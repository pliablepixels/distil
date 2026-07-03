"""Delta / append-only context encoding for multi-turn agent loops.

In a multi-turn loop the full context grows each turn. Delta encoding avoids
re-sending unchanged blocks: a ``DeltaContext`` carries only the *materialized*
(new or changed) blocks plus a list of *references* (ids of blocks reused
verbatim from the previous turn). The receiver reconstructs the full context
via :func:`replay`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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

    order: list[str] = field(default_factory=list)
    """Interleaving of references ('r') and materialized ('m') blocks, in curr
    order — what makes :func:`replay` an exact reconstruction. A declared field
    (serializes with the dataclass); empty only for hand-built instances, where
    replay falls back to references-then-materialized ordering."""

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

    return DeltaContext(
        references=references,
        materialized=materialized,
        tokens_referenced=tokens_referenced,
        tokens_materialized=tokens_materialized,
        order=order,
    )


def replay(prev: list[Block], delta: DeltaContext) -> list[Block]:
    """Reconstruct the full current block list from a previous turn + delta.

    ``delta.order`` (written by :func:`diff_context`) records how references
    and materialized blocks interleave in the original ``curr``, so the
    round-trip is exact. A hand-built DeltaContext with an empty ``order``
    falls back to references-then-materialized ordering.

    Raises:
        KeyError: if a referenced id is not present in ``prev`` — the delta was
            produced against a different previous turn; reconstructing from it
            would silently corrupt context, so failing loudly is correct.
    """
    prev_map: dict[str, Block] = {b.id: b for b in prev}

    def resolve(ref_id: str) -> Block:
        try:
            return prev_map[ref_id]
        except KeyError:
            raise KeyError(
                f"delta references block {ref_id!r} absent from prev "
                "(delta and prev are from different turns)"
            ) from None

    if delta.order:
        result: list[Block] = []
        ref_iter = iter(delta.references)
        mat_iter = iter(delta.materialized)
        for tag in delta.order:
            result.append(resolve(next(ref_iter)) if tag == "r" else next(mat_iter))
        return result
    return [resolve(r) for r in delta.references] + list(delta.materialized)
