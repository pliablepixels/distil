"""Coverage validation for the tightened conformal bounds.

A certificate is only worth shipping if its (1−δ) bound actually covers the true risk at
least (1−δ) of the time. These are Monte-Carlo coverage tests: draw many calibration sets
from a known distribution, compute the bound, and assert empirical coverage holds. They are
the safety gate for any change to the certificate machinery — an invalid bound silently
breaks the motto. Seeded, so deterministic and fast.
"""

from __future__ import annotations

import random

from distil.conformal import (
    betting_upper_bound,
    certified_risk_bound,
    empirical_bernstein_bound,
    tight_risk_bound,
)

DELTA = 0.1
TRIALS = 3000


def _coverage_binary(p: float, n: int, bound_fn) -> float:
    rng = random.Random(0xC0FFEE ^ (int(p * 1000) << 8) ^ n)
    covered = 0
    for _ in range(TRIALS):
        losses = [1.0 if rng.random() < p else 0.0 for _ in range(n)]
        if p <= bound_fn(losses):
            covered += 1
    return covered / TRIALS


def _coverage_graded(p: float, n: int, bound_fn) -> float:
    # Graded losses ~ Beta(2, 2(1-p)/p), whose population mean is exactly p (analytic),
    # so the bound must cover p at least (1−δ) of the time.
    beta = 2.0 * (1 - p) / p
    rng = random.Random(0xBEEF ^ (int(p * 1000) << 8) ^ n)
    covered = 0
    for _ in range(TRIALS):
        losses = [rng.betavariate(2.0, beta) for _ in range(n)]
        if p <= bound_fn(losses):
            covered += 1
    return covered / TRIALS


def test_eb_bound_covers_graded_losses():
    # Empirical-Bernstein must hold its (1−δ) coverage on genuinely graded losses.
    for p in (0.1, 0.2, 0.35):
        for n in (100, 300):
            cov = _coverage_graded(p, n, lambda L: tight_risk_bound(L, DELTA, method="eb"))
            assert cov >= 1 - DELTA - 0.02, f"EB under-covered at p={p}, n={n}: {cov:.3f}"


def test_hb_bound_covers_binary_losses():
    for p in (0.05, 0.1, 0.2):
        for n in (100, 300):
            cov = _coverage_binary(p, n, lambda L: tight_risk_bound(L, DELTA, method="hb"))
            assert cov >= 1 - DELTA - 0.02, f"HB under-covered at p={p}, n={n}: {cov:.3f}"


def test_auto_picks_hb_for_binary_eb_for_graded():
    binary = [0.0, 1.0, 0.0, 0.0, 1.0] * 20
    graded = [0.1, 0.3, 0.05, 0.2, 0.15] * 20
    # auto on binary == hb; auto on graded == eb
    assert tight_risk_bound(binary, DELTA) == tight_risk_bound(binary, DELTA, method="hb")
    assert tight_risk_bound(graded, DELTA) == tight_risk_bound(graded, DELTA, method="eb")


def test_eb_tighter_than_hb_on_low_variance_graded():
    # On low-variance graded losses, EB should be no looser (usually tighter) than treating
    # them as if binary via HB on the rounded mean. This is the *reason* EB exists.
    # zero-variance graded losses, mean 0.08
    eb = empirical_bernstein_bound(0.08, 0.0, 200, DELTA)
    hb = certified_risk_bound(0.08, 200, DELTA)
    assert eb <= hb + 1e-9, f"EB ({eb:.4f}) should beat HB ({hb:.4f}) at zero variance"


def test_betting_bound_covers_binary_losses():
    # The betting CS must hold its (1−δ) coverage on binary losses (fixed-n check).
    for p in (0.05, 0.1, 0.2):
        for n in (100, 300):
            cov = _coverage_binary(p, n, lambda L: betting_upper_bound(L, DELTA))
            assert cov >= 1 - DELTA - 0.02, f"betting under-covered at p={p}, n={n}: {cov:.3f}"


def test_betting_bound_covers_graded_losses():
    for p in (0.1, 0.25):
        for n in (100, 300):
            cov = _coverage_graded(p, n, lambda L: betting_upper_bound(L, DELTA))
            assert cov >= 1 - DELTA - 0.02, (
                f"betting under-covered (graded) p={p}, n={n}: {cov:.3f}"
            )


def test_betting_anytime_valid_coverage_over_a_stream():
    # The defining property: the bound holds SIMULTANEOUSLY at every t along a stream. We check
    # that the *running* bound never drops below the true mean more than δ of the time (coverage
    # of the whole sequence), which a fixed-n bound peeked-at-every-step would violate.
    p, n = 0.1, 300
    trials = 1500
    breached = 0
    for tr in range(trials):
        rng = random.Random(0xABCD ^ tr)
        xs = [1.0 if rng.random() < p else 0.0 for _ in range(n)]
        # check the bound at a schedule of stopping times along the stream
        ever_below = False
        for t in (50, 100, 150, 200, 250, 300):
            if betting_upper_bound(xs[:t], DELTA) < p:
                ever_below = True
                break
        breached += ever_below
    seq_coverage = 1 - breached / trials
    # anytime-valid => sequence-coverage >= 1−δ (with MC slack)
    assert seq_coverage >= 1 - DELTA - 0.03, f"anytime coverage {seq_coverage:.3f} < target"


def test_betting_comparable_to_hb_on_binary():
    # Honest: for FIXED-n binary losses, Bentkus is already near-optimal and betting pays a
    # small price for anytime-validity, so betting is *comparable* (within a modest factor),
    # not strictly tighter. Its edge is the anytime property (above) + graded-loss adaptivity.
    losses = [1.0 if i % 20 == 0 else 0.0 for i in range(300)]  # rate 0.05
    bet = betting_upper_bound(losses, DELTA)
    hb = certified_risk_bound(sum(losses) / len(losses), len(losses), DELTA)
    assert bet <= 1.3 * hb, f"betting ({bet:.4f}) should be within ~1.3x of HB ({hb:.4f})"
