"""The relevance gate — the working-set primitive that wins E8/E11, as a library function.

The relevance-gated reversible tier is distil's highest-accuracy compressor: on a long-horizon
agent it is the only condition statistically non-inferior to full context (E8), and that
result transfers across vendors to a far stronger model (E11). Its mechanism is one decision
made every turn: keep the agent's **working set** — the last ``gate_recent`` user/tool
messages — verbatim, and digest only the aged-out periphery behind a recoverable handle.

This module exposes that decision as a pure, dependency-free function so the winning tier is a
shippable primitive rather than benchmark-only code. ``benchmarks/swe_bench_e2e/compress_proxy``
implements the same selection inline against its proxy state; the contract here is identical
(verified in ``tests/test_gate.py``): given the message roles and ``gate_recent``, return the
set of indices to keep full.

The *operating point* is ``gate_recent``. E11's finding — the safe operating point scales with
agent capability (a weak agent tolerates a small working set; a strong one needs a larger one)
— is why :mod:`distil.calibrate` exists: it selects ``gate_recent`` from a calibration set
instead of leaving it a hand-tuned constant with a silent-loss hazard.
"""

from __future__ import annotations

from collections.abc import Sequence

# Roles whose messages form the agent's working set (its observations and instructions);
# assistant/system turns are not gated periphery.
WORKING_SET_ROLES = ("user", "tool")


def working_set_indices(
    roles: Sequence[str], gate_recent: int, *, working_roles: Sequence[str] = WORKING_SET_ROLES
) -> set[int]:
    """Indices of messages to keep **verbatim** under the relevance gate.

    Args:
        roles: per-message role strings, in conversation order.
        gate_recent: working-set size — keep the last ``gate_recent`` working-role messages
            full; digest the rest. ``gate_recent <= 0`` digests everything (no working set);
            ``gate_recent >= len(working-role messages)`` keeps everything (a no-op gate, the
            regime where short conversations have no periphery to digest).
        working_roles: which roles count as working-set messages (default user/tool).

    Returns the set of indices into ``roles`` that must be kept full. Pure and deterministic.
    """
    if gate_recent <= 0:
        return set()
    ut = [i for i, r in enumerate(roles) if r in working_roles]
    return set(ut[-gate_recent:])


def monotone_gate(gate_recent: int, *, digest=None):
    """A **cache-monotone** relevance-gate strategy (the lossless cost lever, #1).

    Returns a ``Strategy`` (``(blocks, turn) -> blocks``) that keeps the last ``gate_recent``
    blocks (the working set) plus any volatile current-turn block FULL, and replaces every
    older block with a *deterministic* digest. Two properties make it cache-friendly without
    touching the motto:

    * **Deterministic digest** — the same block text always digests to the same bytes, so a
      block's rendering never changes turn-to-turn once it ages out.
    * **Monotone boundary** — a block only ever transitions full -> digested (never back), and
      the digested prefix grows only at its tail as the window slides.

    Together these keep the digested prefix **byte-stable across turns**, so prompt-cache /
    KV-prefix reuse captures it (a cache *read* is ~10x cheaper than fresh input;
    :mod:`distil.compress.cache_aware` prices it). This is lossless relative to the plain gate
    — it changes which bytes are *cached*, not which bytes the agent sees — so it cannot change
    a decision.

    Honest cost scope: the win is over a *cache-hostile* gate (one that re-digests a block
    differently each turn and busts the prefix), not necessarily over no compression at all —
    on content that is already fully cacheable, caching alone can be cheaper than any
    compression, because compressing rewrites cached bytes as fresh/cache-write. The gate's
    primary payoff remains *accuracy* on long-horizon agents (E8/E11); cache-monotonicity is
    what stops compression from throwing cost away on top of that.

    Aging out is recoverable: digests sit behind a content handle (see
    :mod:`distil.skeleton`), so the agent can still expand a block it needs.
    """
    from .skeleton import smart_digest
    from .trajectory import Stability

    do_digest = digest or smart_digest

    def strat(blocks, turn):
        keep_from = max(0, len(blocks) - gate_recent) if gate_recent > 0 else len(blocks)
        out = []
        for i, b in enumerate(blocks):
            if i >= keep_from or b.stability is Stability.VOLATILE:
                out.append(b)
            else:
                out.append(b.copy_with(do_digest(b.text)))
        return out

    return strat


def graded_gate(keep_full: int, keep_light: int, *, light=None, heavy=None):
    """A **graded** relevance gate (#2): per-distance compression tiers, not binary keep/digest.

    The plain gate keeps the working set full and digests everything else uniformly. The graded
    gate digests *more* of the periphery the further it is from the working set, recovering
    extra savings while keeping the decision-bearing recent context intact:

    * last ``keep_full`` blocks (and any volatile turn): FULL,
    * the next ``keep_light`` blocks: a LIGHT digest (generous head/tail), and
    * everything older: a HEAVY digest (tight head/tail).

    All digests are deterministic, so the gate stays cache-monotone (see :func:`monotone_gate`).
    Because the extra compression introduces a *graded* (non-binary) per-message loss, the
    operating point is chosen by the empirical-Bernstein certificate
    (:func:`distil.conformal.tight_risk_bound` with ``method="eb"``), which is tighter than
    Hoeffding–Bentkus exactly in that low-variance graded regime.
    """
    from .skeleton import smart_digest
    from .trajectory import Stability

    # light ~= the plain uniform digest (near periphery keeps full fidelity); heavy crushes the
    # distant past much harder. So the graded gate compresses the FAR periphery more than a
    # uniform gate, spending its fidelity budget where recency makes it likeliest to matter.
    do_light = light or (lambda t: smart_digest(t, head=400, tail=200))
    do_heavy = heavy or (lambda t: smart_digest(t, head=120, tail=60))

    def strat(blocks, turn):
        n = len(blocks)
        full_from = max(0, n - keep_full)
        light_from = max(0, n - keep_full - keep_light)
        out = []
        for i, b in enumerate(blocks):
            if i >= full_from or b.stability is Stability.VOLATILE:
                out.append(b)
            elif i >= light_from:
                out.append(b.copy_with(do_light(b.text)))
            else:
                out.append(b.copy_with(do_heavy(b.text)))
        return out

    return strat


def gate_fraction(
    roles: Sequence[str], gate_recent: int, *, working_roles: Sequence[str] = WORKING_SET_ROLES
) -> float:
    """Fraction of working-role messages the gate would digest (compression aggressiveness).

    1.0 means everything is digested (no working set); 0.0 means nothing is (no-op gate).
    A higher fraction is a more aggressive operating point — the axis E11 calibrates over.
    """
    ut = [i for i, r in enumerate(roles) if r in working_roles]
    if not ut:
        return 0.0
    kept = len(working_set_indices(roles, gate_recent, working_roles=working_roles) & set(ut))
    return (len(ut) - kept) / len(ut)
