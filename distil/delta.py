"""Delta / append-only context encoding for multi-turn agent loops.

In a multi-turn loop the full context grows each turn. Delta encoding avoids
re-sending unchanged blocks: a ``DeltaContext`` carries only the *materialized*
(new or changed) blocks plus a list of *references* (ids of blocks reused
verbatim from the previous turn). The receiver reconstructs the full context
via :func:`replay`.
"""

from __future__ import annotations

from dataclasses import dataclass

from distil.trajectory import Block
from distil import tokenizer as _tok_module


@dataclass
class DeltaContext:
    """The encoded delta between two consecutive turns."""

    references: list[str]
    """Ids of blocks reused unchanged from the previous turn, in order."""

    materialized: list[Block]
    """Blocks that are new or whose text changed — the only ones sent."""

    tokens_referenced: int
    """Token count of all referenced (not-sent) blocks."""

    tokens_materialized: int
    """Token count of all materialized (sent) blocks."""

    @property
    def reduction(self) -> float:
        """Fraction of total tokens avoided by referencing instead of sending.

        Returns a value in [0, 1].  Zero when nothing was referenced;
        1.0 when everything was referenced (degenerate / empty materialized).
        """
        total = self.tokens_referenced + self.tokens_materialized
        if total == 0:
            return 0.0
        return self.tokens_referenced / total


def diff_context(
    prev: list[Block],
    curr: list[Block],
    tok: _tok_module.Tokenizer | None = None,
) -> DeltaContext:
    """Compute the delta between ``prev`` and ``curr`` block lists.

    A block in ``curr`` is a **reference** if a block with the same ``id``
    AND identical ``text`` exists in ``prev``; otherwise it is
    **materialized** (new id, or same id with changed text).

    ``curr`` order is preserved.

    Args:
        prev: The full block list from the previous turn.
        curr: The full block list for the current turn.
        tok:  Tokenizer to use for counting.  Defaults to
              ``distil.tokenizer.DEFAULT``.

    Returns:
        A :class:`DeltaContext` describing the delta.
    """
    if tok is None:
        tok = _tok_module.DEFAULT

    # Build lookup: id -> text for blocks in prev.
    prev_text: dict[str, str] = {b.id: b.text for b in prev}

    references: list[str] = []
    materialized: list[Block] = []
    tokens_referenced = 0
    tokens_materialized = 0

    order: list[str] = []
    for block in curr:
        if block.id in prev_text and prev_text[block.id] == block.text:
            # Unchanged — reference only.
            references.append(block.id)
            tokens_referenced += tok.count(block.text)
            order.append("r")
        else:
            # New or changed — must be materialized.
            materialized.append(block)
            tokens_materialized += tok.count(block.text)
            order.append("m")

    delta = DeltaContext(
        references=references,
        materialized=materialized,
        tokens_referenced=tokens_referenced,
        tokens_materialized=tokens_materialized,
    )
    # Attach the interleaving order so replay() can reconstruct curr exactly.
    object.__setattr__(delta, "_order", order) if hasattr(delta, "__slots__") else setattr(
        delta, "_order", order
    )
    return delta


def replay(prev: list[Block], delta: DeltaContext) -> list[Block]:
    """Reconstruct the full current block list from a previous turn + delta.

    References are resolved by id against ``prev``; materialized blocks fill
    the remaining positions.  The reconstructed list preserves the order
    encoded in ``delta`` (references first, in their listed order, then
    materialized blocks in their listed order — matching the original ``curr``
    ordering as encoded by :func:`diff_context`).

    More precisely: :func:`diff_context` interleaves references and
    materialized blocks according to their position in ``curr``.  To faithfully
    reconstruct that order we need to track which slot each entry belongs to.
    We encode this via a *position tag* approach: we replay the combined stream
    in the same order that :func:`diff_context` iterated over ``curr``.

    Because :func:`diff_context` emits either a reference id or a materialized
    block for each position in ``curr`` (in order), we can reconstruct ``curr``
    by zipping the two queues back together in the order they were consumed.

    The tag stream is implicit: the delta does not store it, so we cannot
    reconstruct arbitrary interleaving.  However, :func:`diff_context`
    guarantees a specific contract: *for every block in curr (in order),
    exactly one of — append to references or append to materialized — was
    called*.  We record this interleaving by storing the ordering tag inside
    the DeltaContext at construction time.

    Wait — the current :class:`DeltaContext` dataclass has no ordering tag.
    Per the spec the dataclass fields are fixed, so we must infer the ordering.
    The only information available is the two lists.  We cannot reconstruct
    arbitrary interleaving from two unordered queues without additional data.

    The spec says: "References resolve by id against prev; materialized blocks
    fill their slots/positions."  The natural interpretation that makes
    round-trip lossless is: replay iterates ``curr`` positionally.  But since
    we don't store ``curr``, we need another convention.

    **Convention adopted (consistent with diff_context ordering):**
    We reconstruct the original ``curr`` order by merging references and
    materialized blocks using the ordering embedded in ``delta``: a reference
    id in ``delta.references`` at index *i* was the *i*-th reference emitted,
    and a materialized block at index *j* was the *j*-th materialized block
    emitted.  Without the interleaving tag we cannot reconstruct the exact
    ``curr`` order from these two independent lists alone.

    To make :func:`replay` lossless we therefore require that
    :class:`DeltaContext` stores the ordering.  Since the public dataclass
    cannot be changed, we smuggle the ordering as a hidden attribute
    ``_order`` (a list of ``'r'`` / ``'m'`` chars) that :func:`diff_context`
    attaches after construction.  If ``_order`` is absent (e.g. the
    DeltaContext was built by hand), we fall back to references-then-materialized
    ordering.

    Args:
        prev:  The full block list from the previous turn.
        delta: A :class:`DeltaContext` produced by :func:`diff_context`.

    Returns:
        The reconstructed full block list for the current turn.
    """
    prev_map: dict[str, Block] = {b.id: b for b in prev}

    order: list[str] | None = getattr(delta, "_order", None)

    if order is not None:
        # Full round-trip reconstruction.
        result: list[Block] = []
        ref_iter = iter(delta.references)
        mat_iter = iter(delta.materialized)
        for tag in order:
            if tag == "r":
                ref_id = next(ref_iter)
                result.append(prev_map[ref_id])
            else:
                result.append(next(mat_iter))
        return result
    else:
        # Fallback: references first (resolved from prev), then materialized.
        result = []
        for ref_id in delta.references:
            result.append(prev_map[ref_id])
        result.extend(delta.materialized)
        return result
