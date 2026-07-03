"""Trajectory-level risk certificate — certifying the invariant users care about.

The E7 lesson: a valid per-step next-action certificate can pass while
end-to-end success collapses. These tests pin the corrected target: the loss
is the END-TO-END outcome of matched full/compressed runs, the certificate
refuses to overclaim on small samples, and the LTT ladder picks the most
aggressive level whose trajectory risk is actually controlled.
"""

from __future__ import annotations

from distil.certify.trajectory_risk import (
    TrajectoryOutcome,
    certify_trajectory_risk,
    drift_monitor,
    select_compression_level,
)


def _outcomes(n: int, degraded: int) -> list[TrajectoryOutcome]:
    out = []
    for i in range(n):
        regressed = i < degraded
        out.append(
            TrajectoryOutcome(task_id=f"t{i}", full_success=True, compressed_success=not regressed)
        )
    return out


def test_loss_is_end_to_end_degradation_only():
    # full failed too → compression taught us nothing → no loss
    both_fail = TrajectoryOutcome("a", full_success=False, compressed_success=False)
    assert both_fail.loss == 0.0
    regression = TrajectoryOutcome("b", full_success=True, compressed_success=False)
    assert regression.loss == 1.0
    fine = TrajectoryOutcome("c", full_success=True, compressed_success=True)
    assert fine.loss == 0.0


def test_certifies_low_degradation_risk():
    cert = certify_trajectory_risk(_outcomes(200, degraded=1), alpha=0.05)
    assert cert.certified
    assert cert.risk_bound < 0.05
    assert "exchangeable" in cert.assumptions  # the honesty clause travels with it


def test_refuses_high_degradation():
    # The E7 scenario: certificate must FAIL when end-to-end success collapses,
    # even if per-step metrics would have looked fine.
    cert = certify_trajectory_risk(_outcomes(100, degraded=36), alpha=0.05)
    assert not cert.certified
    assert "NOT CERTIFIED" in cert.statement


def test_refuses_small_samples():
    # 5 perfect trajectories are not evidence — refusing beats overclaiming.
    cert = certify_trajectory_risk(_outcomes(5, degraded=0), alpha=0.05)
    assert not cert.certified
    assert cert.risk_bound == 1.0


def test_ladder_selects_most_aggressive_controlled_level():
    levels = [
        _outcomes(200, degraded=0),  # mild: safe
        _outcomes(200, degraded=2),  # medium: safe
        _outcomes(200, degraded=60),  # aggressive: breaks tasks
    ]
    assert select_compression_level(levels, alpha=0.05, delta=0.05) == 1


def test_ladder_returns_minus_one_when_nothing_certifies():
    levels = [_outcomes(50, degraded=30)]
    assert select_compression_level(levels, alpha=0.05, delta=0.05) == -1


def test_drift_monitor_trips_on_shift():
    mon = drift_monitor(alpha=0.05, delta=0.05)
    # calm period at ~2% risk: must not trip
    for i in range(100):
        mon.update(1.0 if i % 50 == 0 else 0.0)
    assert not mon.tripped
    # distribution shift: risk jumps far above alpha → monitor must alarm
    for _ in range(200):
        if mon.update(0.5):
            break
    assert mon.tripped  # certificate is stale — recalibrate
