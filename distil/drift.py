"""Anytime-valid drift detection — fire when live decision-change exceeds the certified budget.

The Decision-Equivalence Risk Certificate (:mod:`distil.conformal`) holds under
*exchangeability*: it is valid for traffic that looks like the calibration set. The standing
GA risk is silent drift — a new model, a prompt change, or a workload shift pushes the true
decision-change rate above the budget α the operating point was certified at, and nothing
notices until quality has already degraded.

This module closes that gap with a sequential test that may be checked **after every turn**
without a multiplicity penalty. It runs a betting e-process for the null ``H0: risk ≤ α``:
capital ``K_t = ∏ (1 + λ_i (X_i − α))`` (predictable stakes ``λ_i``, the same tuning as
:func:`distil.conformal.betting_upper_bound`). Under ``H0`` the capital is a non-negative
supermartingale with ``K_0 = 1``, so by Ville's inequality ``P(∃t: K_t ≥ 1/δ) ≤ δ`` — the
false-alarm probability is at most ``δ`` *no matter how often you peek*. When capital crosses
``1/δ`` the monitor trips: the live risk has exceeded the budget with confidence ``1−δ``, and
the operating point should be recalibrated (:func:`distil.calibrate.calibrate_operating_point`)
or the gate should fall back to full context.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DriftMonitor:
    """A running, anytime-valid alarm for ``risk > alpha``.

    Feed it one per-turn loss at a time (``1.0`` iff the compressed decision diverged from the
    full-context decision, or any value in ``[0, 1]`` for a graded loss). ``tripped`` becomes
    ``True`` once the live risk is certified to exceed ``alpha`` at confidence ``1 − delta``.
    """

    alpha: float
    delta: float = 0.05
    c: float = 0.5

    capital: float = 1.0
    n: int = 0
    tripped: bool = False
    _run_sum: float = 0.0
    _run_sq: float = 0.0
    _sig2_prev: float = 0.25  # σ̂²_0

    def update(self, loss: float) -> bool:
        """Record one turn's loss; return whether the monitor is now tripped."""
        if self.tripped:
            return True
        self.n += 1
        ld = math.log(1.0 / self.delta)
        lam = min(self.c, math.sqrt(2.0 * ld / (self._sig2_prev * self.n)))  # predictable
        self.capital *= 1.0 + lam * (loss - self.alpha)
        if self.capital < 0.0:
            self.capital = 0.0
        # advance the running shrinkage variance for the next bet
        self._run_sum += loss
        mu = (0.5 + self._run_sum) / (1 + self.n)
        self._run_sq += (loss - mu) ** 2
        self._sig2_prev = (0.25 + self._run_sq) / (1 + self.n)
        if self.capital >= 1.0 / self.delta:
            self.tripped = True
        return self.tripped

    def observe(self, losses) -> bool:
        """Feed a batch of losses; return whether tripped after the batch."""
        for x in losses:
            if self.update(x):
                break
        return self.tripped

    @property
    def evalue(self) -> float:
        """Accumulated capital (an e-value); ``>= 1/delta`` means tripped."""
        return self.capital

    def status(self) -> dict:
        return {
            "alpha": self.alpha,
            "delta": self.delta,
            "n": self.n,
            "evalue": round(self.capital, 4),
            "threshold": round(1.0 / self.delta, 2),
            "tripped": self.tripped,
            "action": "recalibrate or fall back to full context" if self.tripped else "ok",
        }
