"""Conformal risk-controlled compression — a distribution-free guarantee.

The equivalence *dial* (distil.compress.adaptive) trades savings for a heuristic
per-turn budget. This module replaces the heuristic with a **statistical
certificate**: pick the most aggressive compression level whose *decision-change
rate* is provably bounded by a risk level α — with a finite-sample,
distribution-free guarantee, calibrated on your own traffic.

The machinery is conformal risk control:

  * **Learn-Then-Test (LTT)** — Angelopoulos, Bates, Candès, Jordan & Lei,
    *Ann. Appl. Stat.* 2025 (arXiv:2110.01052). Reframes risk control as multiple
    hypothesis testing; with Hoeffding–Bentkus p-values and fixed-sequence testing
    it yields, for the selected level λ̂,  **P( R(λ̂) ≤ α ) ≥ 1 − δ** — distribution-
    free, finite-sample, no monotonicity assumed.
  * **Conformal Risk Control (CRC)** — Angelopoulos, Bates, Fisch, Lei & Schuster,
    *ICLR 2024* (arXiv:2208.02814). For a monotone 0/1 loss, controls the *expected*
    rate: **E[ L(λ̂) ] ≤ α**, tight to O(1/n).

Mapping: a "level" λ is a compression strategy of known aggressiveness; the loss on
a calibration turn is ``1`` iff the agent's decision changes vs. the original
context (graded by the same runner the certification gate uses — deterministic or
the live model). R(λ) is the decision-change rate. We certify the most aggressive λ
whose risk is controlled at α.

HONEST CAVEAT (the one assumption): conformal guarantees require **exchangeability**
— the calibration traffic must look like the live traffic. Under distribution shift
(a new agent, a prompt change, a workload drift) the bound can silently weaken;
recalibrate on a rolling window of recent traffic. The guarantee is real, not
magic — it holds for the distribution you calibrated on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Hoeffding–Bentkus p-value for the null  H: R(λ) > α  (reject ⇒ certify R ≤ α)
# --------------------------------------------------------------------------- #


def _hoeffding_p(rhat: float, n: int, alpha: float) -> float:
    """Hoeffding tail bound on P(mean ≤ rhat | true risk ≥ α), one-sided."""
    if rhat >= alpha:
        return 1.0
    return math.exp(-2.0 * n * (alpha - rhat) ** 2)


def _bentkus_p(rhat: float, n: int, alpha: float) -> float:
    """Bentkus bound: e · P(Binom(n, α) ≤ ⌈n·rhat⌉) — tighter than Hoeffding in
    the small-rhat regime that matters for certification."""
    if rhat >= alpha:
        return 1.0
    k = math.ceil(n * rhat)
    # exact binomial CDF P(X ≤ k), X ~ Binom(n, α)
    cdf = sum(math.comb(n, i) * alpha**i * (1.0 - alpha) ** (n - i) for i in range(k + 1))
    return min(1.0, math.e * cdf)


def hb_pvalue(rhat: float, n: int, alpha: float) -> float:
    """Hoeffding–Bentkus p-value = min of the two valid bounds (still valid)."""
    if n <= 0:
        return 1.0
    return min(1.0, _hoeffding_p(rhat, n, alpha), _bentkus_p(rhat, n, alpha))


# --------------------------------------------------------------------------- #
# Selection procedures
# --------------------------------------------------------------------------- #


@dataclass
class Certificate:
    method: str  # "ltt" or "crc"
    alpha: float
    delta: float | None
    level: str | None  # name of the certified (most aggressive controlled) level
    index: int  # its index in the ladder, -1 if nothing certifies
    empirical_risk: float  # observed decision-change rate at the certified level
    savings: float  # token savings at the certified level
    n: int  # calibration sample size
    guarantee: str  # human-readable guarantee statement


def ltt_certify(
    level_losses: list[list[float]], *, alpha: float, delta: float
) -> tuple[int, list[float]]:
    """Fixed-sequence Learn-Then-Test. ``level_losses[i]`` are the 0/1 losses at
    level i, with levels ordered LEAST→MOST aggressive (risk non-decreasing).
    Tests H_i: R(λ_i) > α in order at level δ, stopping at the first non-rejection.
    Returns (index of most aggressive certified level or -1, per-level p-values).
    Guarantee: P(R(λ̂) ≤ α) ≥ 1 − δ (fixed-sequence ⇒ no multiplicity penalty)."""
    pvals: list[float] = []
    certified = -1
    stopped = False
    for losses in level_losses:
        n = len(losses)
        rhat = (sum(losses) / n) if n else 1.0
        p = hb_pvalue(rhat, n, alpha)
        pvals.append(p)
        if stopped:
            continue
        if p <= delta:  # reject H ⇒ this level's risk is controlled
            certified = len(pvals) - 1
        else:  # cannot certify ⇒ fixed-sequence stops; nothing past here counts
            stopped = True
    return certified, pvals


def crc_select(level_losses: list[list[float]], *, alpha: float, loss_bound: float = 1.0) -> int:
    """Conformal Risk Control for a monotone 0/1 loss. Returns the most aggressive
    level whose finite-sample-corrected risk (n·R̂ + B)/(n+1) ≤ α. Guarantee:
    E[L(λ̂)] ≤ α. Levels ordered LEAST→MOST aggressive."""
    selected = -1
    for i, losses in enumerate(level_losses):
        n = len(losses)
        if not n:
            continue
        rhat = sum(losses) / n
        corrected = (n * rhat + loss_bound) / (n + 1)
        if corrected <= alpha:
            selected = i
        else:
            break  # monotone risk ⇒ no more-aggressive level can satisfy it
    return selected


# --------------------------------------------------------------------------- #
# Calibration over a corpus — the Decision-Equivalence Risk Certificate (DERC)
# --------------------------------------------------------------------------- #


def _truncate_level(limit: int):
    from .trajectory import Stability

    def strat(blocks, turn):
        return [
            b.copy_with(b.text[:limit]) if b.stability is Stability.VOLATILE else b for b in blocks
        ]

    return strat


def _skeleton_level():
    """Content-aware skeleton digest as a ladder level (see :mod:`distil.skeleton`).

    Unlike truncation it preserves structure — code signatures and traceback tails — so it
    sits at a much better savings/decision-change point on most volatile context, while
    remaining reversible (the original is recoverable behind a content handle)."""
    from .skeleton import smart_digest
    from .trajectory import Stability

    def strat(blocks, turn):
        return [
            b.copy_with(smart_digest(b.text)) if b.stability is Stability.VOLATILE else b
            for b in blocks
        ]

    return strat


def default_ladder():
    """Least → most aggressive compression levels, ordered by expected risk. Reuses
    Distil's safe operating points, then salience-PROTECTED aggressive levels (which
    keep the decision-bearing lines while crushing the rest), then the raw truncation
    sweep that traces the cliff. The certificate picks the highest-savings level whose
    risk is controlled — so a protected-aggressive level can legitimately win."""
    from .compress.adaptive import byte_exact
    from .compress.salience import protect
    from .compress.strategies import distil

    return [
        ("byte-exact", byte_exact),
        ("lossless", distil),
        ("skeleton", _skeleton_level()),
        ("protect+skeleton", protect(_skeleton_level())),
        ("protect+truncate@500", protect(_truncate_level(500))),
        ("protect+truncate@250", protect(_truncate_level(250))),
        ("truncate@1000", _truncate_level(1000)),
        ("truncate@500", _truncate_level(500)),
        ("truncate@250", _truncate_level(250)),
        ("truncate@120", _truncate_level(120)),
    ]


def calibrate(
    entries,
    runner,
    *,
    alpha: float,
    delta: float = 0.05,
    method: str = "ltt",
    ladder=None,
    tok=None,
) -> Certificate:
    """Calibrate a Decision-Equivalence Risk Certificate over ``entries``.

    For each level, the per-turn loss is ``1`` iff the runner's decision on the
    compressed context differs from its decision on the original. Returns the most
    aggressive level whose decision-change rate is certified ≤ ``alpha`` (LTT:
    with confidence 1−``delta``; CRC: in expectation)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    if tok is None:
        from .tokenizer import DEFAULT as tok
    rungs = ladder if ladder is not None else default_ladder()
    level_losses: list[list[float]] = [[] for _ in rungs]
    base_tok = 0
    comp_tok = [0] * len(rungs)

    for e in entries:
        for turn in e.trajectory.turns:
            base = runner.decide(turn.blocks)
            base_tok += sum(tok.count(b.text) for b in turn.blocks)
            for i, (_name, strat) in enumerate(rungs):
                comp = strat(turn.blocks, turn.index)
                level_losses[i].append(1.0 if runner.decide(comp) != base else 0.0)
                comp_tok[i] += sum(tok.count(b.text) for b in comp)

    n = len(level_losses[0]) if level_losses else 0
    if method == "crc":
        idx_end = crc_select(level_losses, alpha=alpha)
    else:
        idx_end, _pvals = ltt_certify(level_losses, alpha=alpha, delta=delta)

    # Every level in the certified prefix [0..idx_end] carries the guarantee (fixed-
    # sequence / monotone). The operating point is the HIGHEST-SAVINGS one of them —
    # savings is not monotone in ladder position once protected levels are mixed in.
    def _savings(i: int) -> float:
        return (1.0 - comp_tok[i] / base_tok) if base_tok else 0.0

    idx = max(range(idx_end + 1), key=_savings) if idx_end >= 0 else -1

    if idx < 0:
        return Certificate(
            method,
            alpha,
            None if method == "crc" else delta,
            None,
            -1,
            0.0,
            0.0,
            n,
            f"No level certifies a decision-change rate ≤ {alpha * 100:.1f}% — "
            "stay at byte-exact (or relax α).",
        )
    name = rungs[idx][0]
    risk = sum(level_losses[idx]) / n if n else 0.0
    savings = (1.0 - comp_tok[idx] / base_tok) if base_tok else 0.0
    if method == "crc":
        guarantee = (
            f"At '{name}' ({savings * 100:.1f}% token savings): the EXPECTED decision-"
            f"change rate vs. uncompressed context is ≤ {alpha * 100:.1f}% (Conformal Risk "
            f"Control, n={n} calibration turns)."
        )
    else:
        guarantee = (
            f"At '{name}' ({savings * 100:.1f}% token savings): the decision-change rate "
            f"vs. uncompressed context is ≤ {alpha * 100:.1f}% with {(1 - delta) * 100:.0f}% "
            f"confidence (Learn-Then-Test, n={n} calibration turns)."
        )
    return Certificate(
        method, alpha, None if method == "crc" else delta, name, idx, risk, savings, n, guarantee
    )
