"""Holdout A/B savings measurement (Roadmap Phase 5).

A synthetic compression ratio is not a savings claim. This measures savings the
honest way: deterministically hold out a control fraction of trajectories, run
distil only on the treatment group, and report the savings with a bootstrap 95%
confidence interval — so the headline number carries its uncertainty, and drift
shows up as the control group diverging.

Determinism: the partition is by SHA-256 of the trajectory id (no RNG state), and
the bootstrap uses a fixed-seed PRNG, so results are reproducible run to run.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from .. import pricing
from ..compress.cache_aware import simulate
from ..tokenizer import DEFAULT, Tokenizer


def _bucket(traj_id: str) -> float:
    """Stable [0,1) hash bucket for a trajectory id."""
    h = hashlib.sha256(traj_id.encode()).hexdigest()[:8]
    return int(h, 16) / 0x100000000


def partition(ids: list[str], control_fraction: float) -> tuple[list[str], list[str]]:
    control = [i for i in ids if _bucket(i) < control_fraction]
    treatment = [i for i in ids if _bucket(i) >= control_fraction]
    return control, treatment


def _savings_fraction(traj, price: pricing.Pricing, tok: Tokenizer) -> float:
    base = simulate(traj, price, strategy="none", caching=False, tok=tok).total_dollars
    dist = simulate(traj, price, strategy="distil", caching=True, tok=tok).total_dollars
    return (1.0 - dist / base) if base else 0.0


def bootstrap_ci(
    values: list[float], iters: int = 2000, alpha: float = 0.05, seed: int = 1234
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via percentile bootstrap, deterministic."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[min(iters - 1, int((1 - alpha / 2) * iters))]
    return mean, lo, hi


@dataclass
class HoldoutReport:
    control_ids: list[str]
    treatment_ids: list[str]
    mean_savings: float
    ci_low: float
    ci_high: float
    control_mean_savings: float

    @property
    def summary(self) -> str:
        return (
            f"treatment savings {self.mean_savings * 100:.1f}% "
            f"(95% CI {self.ci_low * 100:.1f}–{self.ci_high * 100:.1f}%), "
            f"n={len(self.treatment_ids)}; control held out n={len(self.control_ids)}"
        )


def run_holdout(
    entries, price: pricing.Pricing, *, control_fraction: float = 0.2, tok: Tokenizer = DEFAULT
) -> HoldoutReport:
    by_id = {e.trajectory.id: e.trajectory for e in entries}
    control, treatment = partition(list(by_id), control_fraction)
    treat_vals = [_savings_fraction(by_id[i], price, tok) for i in treatment]
    ctrl_vals = [_savings_fraction(by_id[i], price, tok) for i in control]
    mean, lo, hi = bootstrap_ci(treat_vals)
    ctrl_mean = sum(ctrl_vals) / len(ctrl_vals) if ctrl_vals else 0.0
    return HoldoutReport(control, treatment, mean, lo, hi, ctrl_mean)
