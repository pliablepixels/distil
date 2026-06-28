"""Anytime-valid drift detection: bounded false alarms, real power, valid under peeking."""

from __future__ import annotations

import random

from distil.drift import DriftMonitor

DELTA = 0.1


def test_no_false_alarm_when_risk_at_or_below_budget():
    # Stream genuinely at the budget (risk == alpha): false-alarm rate must be <= delta, even
    # though we check after every single turn (anytime-valid => no multiplicity penalty).
    alpha = 0.1
    trials, n = 2000, 400
    alarms = 0
    for tr in range(trials):
        rng = random.Random(0xD1F7 ^ tr)
        mon = DriftMonitor(alpha=alpha, delta=DELTA)
        for _ in range(n):
            x = 1.0 if rng.random() < alpha else 0.0  # true risk == alpha (boundary)
            if mon.update(x):
                break
        alarms += mon.tripped
    false_alarm_rate = alarms / trials
    assert false_alarm_rate <= DELTA + 0.03, f"false-alarm rate {false_alarm_rate:.3f} > delta"


def test_detects_real_drift_above_budget():
    # Live risk well above budget -> the monitor should trip on essentially every run.
    alpha = 0.1
    trials, n = 300, 600
    tripped = 0
    for tr in range(trials):
        rng = random.Random(0xBADF00D ^ tr)
        mon = DriftMonitor(alpha=alpha, delta=DELTA)
        for _ in range(n):
            x = 1.0 if rng.random() < 0.30 else 0.0  # drifted to 30% >> 10% budget
            if mon.update(x):
                break
        tripped += mon.tripped
    power = tripped / trials
    assert power >= 0.95, f"drift detection power {power:.3f} too low"


def test_status_reports_action_on_trip():
    mon = DriftMonitor(alpha=0.05, delta=DELTA)
    mon.observe([1.0] * 200)  # a flood of divergences must trip
    st = mon.status()
    assert st["tripped"] is True
    assert "recalibrate" in st["action"]
    assert st["evalue"] >= st["threshold"]


def test_clean_stream_stays_untripped_and_reports_ok():
    mon = DriftMonitor(alpha=0.1, delta=DELTA)
    rng = random.Random(1)
    mon.observe([1.0 if rng.random() < 0.03 else 0.0 for _ in range(400)])  # well below budget
    assert mon.tripped is False
    assert mon.status()["action"] == "ok"
