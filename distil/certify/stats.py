"""Non-inferiority testing via TOST — the statistical core of the quality contract.

We do NOT eyeball "looks the same." We pre-register an indifference margin and
run two one-sided tests (TOST). Non-inferiority (compressed is not worse than
baseline by more than `margin`) is the lower test; full equivalence requires
both. The Student-t tail is computed from the regularized incomplete beta
function (Numerical-Recipes-style continued fraction) so there is zero
dependency on scipy/numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _betacf(a: float, b: float, x: float) -> float:
    maxit, eps, fpmin = 300, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """Survival P(T > t) for Student-t with df degrees of freedom."""
    x = df / (df + t * t)
    half = 0.5 * _betai(df / 2.0, 0.5, x)
    return half if t > 0 else 1.0 - half


def t_cdf(t: float, df: float) -> float:
    return 1.0 - t_sf(t, df)


@dataclass
class TostResult:
    n: int
    mean_diff: float
    margin: float
    alpha: float
    p_non_inferior: float  # H0: mean <= -margin  (reject -> non-inferior)
    p_not_superiorly_worse: float  # H0: mean >= +margin
    non_inferior: bool
    equivalent: bool

    @property
    def verdict(self) -> str:
        return "PASS" if self.non_inferior else "FAIL"


def tost(diffs: list[float], margin: float = 0.02, alpha: float = 0.05) -> TostResult:
    """Paired TOST. `diffs[i]` = compressed_score - baseline_score for sample i.

    `margin` is the indifference bound on the mean outcome difference (e.g. 0.02
    = tolerate at most a 2-point drop in task success). Non-inferiority is the
    relevant verdict for "compression must not hurt"; equivalence is reported too.
    """
    n = len(diffs)
    if n == 0:
        raise ValueError("need at least one paired sample")
    mean = sum(diffs) / n

    if n == 1:
        var = 0.0
    else:
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    df = max(n - 1, 1)

    if se == 0.0:
        # Degenerate certainty (e.g. perfectly lossless -> every diff is 0).
        p_lower = 0.0 if mean > -margin else 1.0
        p_upper = 0.0 if mean < margin else 1.0
    else:
        t_lower = (mean + margin) / se
        t_upper = (mean - margin) / se
        p_lower = t_sf(t_lower, df)  # P(T > t_lower): reject mean <= -margin
        p_upper = t_cdf(t_upper, df)  # P(T < t_upper): reject mean >= +margin

    non_inferior = p_lower < alpha
    equivalent = non_inferior and p_upper < alpha
    return TostResult(n, mean, margin, alpha, p_lower, p_upper, non_inferior, equivalent)


Z_95 = 1.959963984540054  # two-sided 95% normal quantile


@dataclass(frozen=True)
class McNemarNI:
    n: int
    delta: float  # candidate - reference pass-rate difference (negative => candidate worse)
    ci95_low: float
    ci95_high: float
    se: float
    margin: float
    noninferior: bool

    @property
    def verdict(self) -> str:
        return "PASS" if self.noninferior else "FAIL"


def mcnemar_noninferiority(
    b: int, c: int, n: int, margin: float = 0.05, z: float = Z_95
) -> McNemarNI:
    """Non-inferiority for **paired binary** outcomes (the deployment question).

    This is the test the E8/E11 task-success comparisons use: given the same items scored
    under a reference (e.g. full context) and a candidate (a compressed operating point),
    is the candidate no worse than the reference by more than ``margin``? A McNemar
    p-value only answers "is there *any* difference?"; failing to reject it is *absence of
    evidence*, not equivalence. Non-inferiority is the FDA-standard framing for "compression
    must not hurt".

    Args:
        b: discordant pairs where the **reference** resolved and the candidate did not
           (candidate *losses*).
        c: discordant pairs where the **candidate** resolved and the reference did not
           (candidate *gains*).
        n: total paired instances (both scored on the same items).
        margin: largest tolerated absolute pass-rate drop (proportion, e.g. 0.05 = 5 pp).

    Non-inferior iff the lower 95% CI bound on ``delta`` exceeds ``-margin`` (Wald interval
    using the McNemar variance of correlated proportions). Mirrors
    ``benchmarks/swe_bench_e2e/stats.noninferiority_paired`` — kept here as the shippable
    library home so production calibration does not depend on the benchmark harness.
    """
    if n <= 0:
        return McNemarNI(0, 0.0, 0.0, 0.0, 0.0, margin, False)
    delta = (c - b) / n
    var = ((b + c) - (b - c) ** 2 / n) / (n * n)
    se = math.sqrt(max(var, 0.0))
    lo = delta - z * se
    hi = delta + z * se
    return McNemarNI(n, delta, lo, hi, se, margin, bool(lo > -margin))
