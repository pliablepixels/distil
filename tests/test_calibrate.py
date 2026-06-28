"""Auto-calibration of the gate operating point — selection, fail-safe, and real E11 data."""

from __future__ import annotations

from pathlib import Path

import pytest

from distil.calibrate import (
    OperatingPoint,
    calibrate_from_scores,
    calibrate_operating_point,
    paired_discordant,
)

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "docs/paper/results/swe_e2e_longhorizon_deepseek/scores"


def test_paired_discordant_counts():
    base = {"a": True, "b": True, "c": False, "d": True}
    cand = {"a": True, "b": False, "c": True, "d": False}
    # losses (base solved, cand not): b, d -> 2 ; gains (cand solved, base not): c -> 1
    assert paired_discordant(base, cand) == (2, 1, 4)


def test_selects_most_aggressive_noninferior_point():
    # Two non-inferior candidates (tiny losses) -> pick the smaller gate_recent (more aggressive).
    pts = [
        OperatingPoint("gate@12", 12, losses=3, gains=2, n=200),
        OperatingPoint("gate@6", 6, losses=4, gains=3, n=200),
    ]
    cert = calibrate_operating_point(pts, margin=0.05)
    assert not cert.fail_safe
    assert cert.selected == "gate@6"
    assert cert.selected_gate_recent == 6


def test_fail_safe_when_nothing_certifies():
    # A single very lossy candidate -> no operating point qualifies -> fall back to full.
    pts = [OperatingPoint("gate@6", 6, losses=70, gains=12, n=200)]
    cert = calibrate_operating_point(pts, margin=0.05)
    assert cert.fail_safe
    assert cert.selected is None
    assert cert.selected_gate_recent is None
    assert "full context" in cert.rationale


def test_rejects_aggressive_keeps_milder():
    # gate@6 is lossy (large net loss); gate@12 is non-inferior -> select gate@12.
    pts = [
        OperatingPoint("gate@6", 6, losses=70, gains=8, n=200),  # -31 pp, fails
        OperatingPoint("gate@12", 12, losses=20, gains=11, n=200),  # -4.5 pp, passes
    ]
    cert = calibrate_operating_point(pts, margin=0.10)
    assert cert.selected == "gate@12"
    # gate@6 must be recorded and marked non-inferior=False
    g6 = next(v for v in cert.levels if v.name == "gate@6")
    assert g6.noninferior is False


@pytest.mark.skipif(not DS.exists(), reason="DeepSeek E11 results not present")
def test_real_e11_deepseek_calibration_selects_gate12():
    """End-to-end on the committed E11 data: full vs gate@6 vs gate@12.

    The whole point of E11: gate@6 is too aggressive for the strong DeepSeek-V3 agent and
    must be rejected; gate@12 is non-inferior and must be selected. Calibration must
    reproduce that decision purely from the harness outputs.
    """
    cert = calibrate_from_scores(
        DS / "full.json",
        [
            ("gate@6", DS / "distil_gated.json", 6),
            ("gate@12", DS / "distil_gated_gr12.json", 12),
        ],
        margin=0.10,
    )
    assert cert.selected == "gate@12"
    assert cert.selected_gate_recent == 12
    g6 = next(v for v in cert.levels if v.name == "gate@6")
    g12 = next(v for v in cert.levels if v.name == "gate@12")
    assert g6.noninferior is False  # -31 pp, correctly rejected
    assert g12.noninferior is True  # -4.5 pp, correctly accepted
    # sanity: the measured deltas match the paper (gate@12 ~ -4.5pp, gate@6 ~ -31pp)
    assert g12.delta == pytest.approx(-0.045, abs=0.01)
    assert g6.delta < -0.25


@pytest.mark.skipif(not DS.exists(), reason="DeepSeek E11 results not present")
def test_real_e11_strict_margin_fails_safe():
    """At a strict 2 pp margin, even gate@12 (-4.5 pp) should not certify -> fail safe."""
    cert = calibrate_from_scores(
        DS / "full.json",
        [("gate@12", DS / "distil_gated_gr12.json", 12)],
        margin=0.02,
    )
    assert cert.fail_safe
    assert cert.selected is None
