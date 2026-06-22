"""Certified-fallback compression — never ship a transform that changes a decision.

The aggressive Tier-1 digest is *recoverable* but not byte-identical to the model's
eye, so on genuinely ambiguous decisions it can occasionally tip the live model to
a different (still reasonable) action. This compressor removes that risk by
construction:

for each turn, it tries the most aggressive transform first and keeps it ONLY IF
the runner's decision on the compressed context matches the decision on the
original. Otherwise it falls back a rung — to **byte-exact** Tier-0 (whitespace/
JSON minification + run-collapse; the model sees essentially the same content),
and finally to **no compression**. The result is certified non-inferior *by
construction* under whichever runner you pass:

  * ``DeterministicRunner`` — free, structural; the result is provably 100%
    decision-equivalent.
  * ``AnthropicRunner(samples=k)`` — graded by the live model itself; the result
    is the most aggressive compression the *real model* agrees preserves the
    decision, with byte-exact fallback everywhere it doesn't.

The cost of safety is honest and visible: where the digest is unsafe you save
less (byte-exact), but you never change an answer.
"""

from __future__ import annotations

from ..trajectory import Block, Stability
from .strategies import Strategy
from .strategies import distil as _distil
from .strategies import none as _none
from .tier0 import Tier0Lossless

_T0 = Tier0Lossless()


def byte_exact(blocks: list[Block], turn: int) -> list[Block]:
    """Tier-0 only on the volatile tail: minify JSON + collapse exact duplicate
    lines. Byte-recoverable, and the model sees semantically identical content —
    the safe floor above 'no compression'."""
    stable = [b for b in blocks if b.stability is not Stability.VOLATILE]
    volatile = [b for b in blocks if b.stability is Stability.VOLATILE]
    return stable + _T0.compress(volatile).blocks


# Most aggressive → safest. distil = lossless fold/template/digest; byte_exact =
# Tier-0 only; none = untouched (always matches).
DEFAULT_LADDER: list[tuple[str, Strategy]] = [
    ("distil", _distil),
    ("byte-exact", byte_exact),
    ("none", _none),
]


def certified_fallback(
    traj,
    runner,
    *,
    ladder: list[tuple[str, Strategy]] | None = None,
) -> Strategy:
    """Build a per-turn strategy that uses the most aggressive ladder rung whose
    decision the *runner* confirms matches the original. Returns a plain
    ``(blocks, turn) -> blocks`` strategy with the per-turn choice precomputed."""
    rungs = ladder if ladder is not None else DEFAULT_LADDER
    choice: dict[int, Strategy] = {}
    for t in traj.turns:
        base_decision = runner.decide(t.blocks)
        picked: Strategy = _none
        for _name, fn in rungs:
            if runner.decide(fn(t.blocks, t.index)) == base_decision:
                picked = fn
                break
        choice[t.index] = picked

    def strat(blocks: list[Block], turn: int) -> list[Block]:
        return choice.get(turn, _none)(blocks, turn)

    return strat


def fallback_breakdown(
    traj, runner, *, ladder: list[tuple[str, Strategy]] | None = None
) -> dict[str, int]:
    """Diagnostic: how many turns each ladder rung was chosen for (visibility into
    where the digest had to fall back to byte-exact)."""
    rungs = ladder if ladder is not None else DEFAULT_LADDER
    counts: dict[str, int] = {name: 0 for name, _ in rungs}
    for t in traj.turns:
        base = runner.decide(t.blocks)
        for name, fn in rungs:
            if runner.decide(fn(t.blocks, t.index)) == base:
                counts[name] += 1
                break
    return counts


__all__ = ["byte_exact", "certified_fallback", "fallback_breakdown", "DEFAULT_LADDER"]
