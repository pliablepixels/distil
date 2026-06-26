#!/usr/bin/env python3
"""Statistics helpers for the SWE-bench end-to-end eval (Phase 5 / E7).

Pass@1 is a binomial proportion; we report the **Wilson score** 95% confidence
interval rather than the normal approximation because n is small (30-50) and the
proportion can sit near 0 or 1, where the Wald interval misbehaves.
"""

from __future__ import annotations

import math

Z_95 = 1.959963984540054  # standard normal quantile for a two-sided 95% interval


def wilson_ci(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns ``(low, high)`` in [0,1]."""
    if n <= 0:
        return (0.0, 0.0)
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def pass_at_1(resolved: int, n: int) -> dict[str, float | int]:
    lo, hi = wilson_ci(resolved, n)
    return {
        "resolved": resolved,
        "n": n,
        "pass_at_1": round(resolved / n, 4) if n else 0.0,
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
    }


def noninferiority_paired(
    b: int, c: int, n: int, margin: float, z: float = Z_95
) -> dict[str, float | bool]:
    """Non-inferiority test for **paired** binary outcomes (McNemar discordant layout).

    A McNemar p-value answers "is there *any* difference?"; failing to reject it is
    *absence of evidence*, not evidence of equivalence. This test answers the question a
    deployment actually cares about: "is the candidate no worse than the reference by more
    than ``margin``?" — the FDA-standard non-inferiority framing.

    Args:
        b: discordant pairs where the **reference** resolved and the candidate did not
           (candidate *losses*).
        c: discordant pairs where the **candidate** resolved and the reference did not
           (candidate *gains*).
        n: total paired instances (both scored on the same items).
        margin: the largest absolute drop in pass-rate (as a proportion, e.g. 0.05 for
           5 pp) we are willing to tolerate and still call the candidate non-inferior.

    Returns the candidate−reference pass-rate difference ``delta`` with its two-sided 95%
    CI (Wald interval using the McNemar variance of correlated proportions), and the
    non-inferiority verdict: non-inferior iff the lower CI bound exceeds ``-margin``.
    """
    if n <= 0:
        return {
            "delta": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "se": 0.0,
            "margin": margin,
            "noninferior": False,
        }
    delta = (c - b) / n  # p_candidate - p_reference (negative => candidate worse)
    var = ((b + c) - (b - c) ** 2 / n) / (n * n)  # McNemar variance of the difference
    se = math.sqrt(max(var, 0.0))
    lo = delta - z * se
    hi = delta + z * se
    return {
        "delta": round(delta, 4),
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
        "se": round(se, 5),
        "margin": margin,
        "noninferior": bool(lo > -margin),
    }
