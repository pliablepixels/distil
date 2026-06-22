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

from dataclasses import dataclass

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


def _trunc_tail(text: str, head: int = 240, tail: int = 240) -> str:
    if len(text) <= head + tail + 40:
        return text
    return f"{text[:head]}\n…[{len(text) - head - tail} chars elided]…\n{text[-tail:]}"


def _aggressive(blocks: list[Block], turn: int) -> list[Block]:
    """Deepest savings: head/tail truncation of the volatile tail, ignoring
    decision-relevance. It CAN drop a decision — which is exactly why it sits
    behind the ``target_equivalence`` dial, used only on the highest-value turns
    when you relax the target below 100%."""
    out: list[Block] = []
    for b in blocks:
        if b.stability is Stability.VOLATILE:
            t = _trunc_tail(b.text)
            out.append(b.copy_with(t) if len(t) < len(b.text) else b)
        else:
            out.append(b)
    return out


# The dial ladder: at target=1.0 the aggressive rung is never used (you get the
# certified lossless result); relax the target and the budget is spent here first,
# on the highest-value turns. Falls back through lossless → byte-exact → none.
PRODUCTION_LADDER: list[tuple[str, Strategy]] = [
    ("aggressive", _aggressive),
    ("lossless", _distil),
    ("byte-exact", byte_exact),
    ("none", _none),
]


def certified_fallback(
    traj,
    runner,
    *,
    target_equivalence: float = 1.0,
    ladder: list[tuple[str, Strategy]] | None = None,
    tok=None,
) -> Strategy:
    """Build a per-turn strategy that uses the most aggressive ladder rung whose
    decision the *runner* confirms matches the original — falling back to byte-exact
    where it doesn't.

    ``target_equivalence`` is the savings-vs-equivalence dial. At ``1.0`` every
    turn must preserve the decision (byte-exact fallback everywhere the digest is
    unsafe). Relax it — say ``0.95`` — and you grant a **divergence budget** of
    ``floor((1-target) * n_turns)`` turns that keep the *most aggressive* transform
    even though it changes the decision; the budget is spent on the **highest-saving**
    turns first. So 95% buys back compression on the turns where the digest saves
    the most, at a measured, bounded cost in decision-equivalence. Returns a plain
    ``(blocks, turn) -> blocks`` strategy with the per-turn choice precomputed."""
    if not 0.0 < target_equivalence <= 1.0:
        raise ValueError(f"target_equivalence must be in (0, 1], got {target_equivalence}")
    if tok is None:
        from ..tokenizer import DEFAULT as tok
    rungs = ladder if ladder is not None else DEFAULT_LADDER
    top_name, top_fn = rungs[0]
    choice: dict[int, Strategy] = {}
    diverging: list[tuple[int, int]] = []  # (turn_index, tokens_aggressive_saves_over_safe)

    for t in traj.turns:
        base_decision = runner.decide(t.blocks)
        safe: Strategy = _none
        for _name, fn in rungs:
            if runner.decide(fn(t.blocks, t.index)) == base_decision:
                safe = fn
                break
        choice[t.index] = safe
        # if the most aggressive rung diverged from the safe choice, note its extra savings
        if top_fn is not safe:
            agg = sum(tok.count(b.text) for b in top_fn(t.blocks, t.index))
            saf = sum(tok.count(b.text) for b in safe(t.blocks, t.index))
            if agg < saf:
                diverging.append((t.index, saf - agg))

    # turns allowed to diverge to hit the target; +epsilon avoids float truncation
    # (1.0 - 0.8 == 0.19999996, which would int()-truncate the budget to zero).
    budget = int((1.0 - target_equivalence) * len(traj.turns) + 1e-6)
    if budget > 0 and diverging:
        diverging.sort(key=lambda x: -x[1])  # highest extra savings first
        for turn_idx, _gain in diverging[:budget]:
            choice[turn_idx] = top_fn  # spend budget: keep the aggressive transform

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


@dataclass
class FrontierPoint:
    target: float  # the equivalence target you asked for
    equivalence: float  # the decision-equivalence actually achieved
    savings: float  # token savings at that point


def frontier(
    entries,
    runner,
    *,
    targets: tuple[float, ...] = (1.0, 0.99, 0.97, 0.95, 0.90),
    ladder: list[tuple[str, Strategy]] | None = None,
    tok=None,
) -> list[FrontierPoint]:
    """Trace the savings-vs-equivalence curve: for each ``target``, build the
    certified-fallback strategy and measure achieved decision-equivalence and
    token savings over ``entries``. This is the dial in action — 100% is the
    certified-safe point; lower targets buy deeper compression at a measured,
    bounded cost in equivalence."""
    from ..certify.gate import certify

    if tok is None:
        from ..tokenizer import DEFAULT as tok
    rungs = ladder if ladder is not None else PRODUCTION_LADDER
    points: list[FrontierPoint] = []
    for target in targets:
        base_tok = comp_tok = 0
        matches: list[float] = []
        for e in entries:
            strat = certified_fallback(
                e.trajectory, runner, target_equivalence=target, ladder=rungs, tok=tok
            )
            matches.append(certify(e.trajectory, strat, runner=runner).match_rate)
            for turn in e.trajectory.turns:
                base_tok += sum(tok.count(b.text) for b in turn.blocks)
                comp_tok += sum(tok.count(b.text) for b in strat(turn.blocks, turn.index))
        points.append(
            FrontierPoint(
                target,
                sum(matches) / len(matches) if matches else 0.0,
                (1.0 - comp_tok / base_tok) if base_tok else 0.0,
            )
        )
    return points


__all__ = [
    "byte_exact",
    "certified_fallback",
    "fallback_breakdown",
    "frontier",
    "FrontierPoint",
    "DEFAULT_LADDER",
    "PRODUCTION_LADDER",
]
