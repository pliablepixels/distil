"""Conformal risk-controlled compression — distribution-free decision-equivalence."""

from __future__ import annotations

from distil.conformal import calibrate, crc_select, hb_pvalue, ltt_certify
from distil.corpus import load_corpus
from distil.replay.runner import DeterministicRunner


def test_hb_pvalue_properties():
    # observed risk at/above alpha cannot certify
    assert hb_pvalue(0.10, 100, 0.05) == 1.0
    assert hb_pvalue(0.05, 100, 0.05) == 1.0
    # more calibration data → smaller p-value (stronger certificate) for 0 risk
    assert hb_pvalue(0.0, 1000, 0.05) < hb_pvalue(0.0, 200, 0.05) < hb_pvalue(0.0, 50, 0.05)
    # always a valid probability
    for n in (1, 10, 100):
        for r in (0.0, 0.01, 0.5, 1.0):
            assert 0.0 <= hb_pvalue(r, n, 0.05) <= 1.0


def test_ltt_fixed_sequence_picks_most_aggressive_controlled():
    # levels least→most aggressive; risks 0, 0, 0.10, 0.50 (n=200)
    losses = [[0.0] * 200, [0.0] * 200, [0.10] * 200, [0.50] * 200]
    idx, pvals = ltt_certify(losses, alpha=0.05, delta=0.1)
    assert idx == 1  # most aggressive 0-risk level; the 0.10 level violates α=0.05
    assert pvals[0] <= 0.1 and pvals[1] <= 0.1 and pvals[2] == 1.0


def test_ltt_certifies_a_tolerable_nonzero_risk():
    # a level with 2% empirical risk at large n certifies under α=5%
    losses = [[0.0] * 500, [0.02] * 500, [0.30] * 500]
    idx, _ = ltt_certify(losses, alpha=0.05, delta=0.1)
    assert idx == 1  # the 2% level is controlled at 5%; the 30% level is not


def test_crc_monotone_selection():
    losses = [[0.0] * 100, [0.02] * 100, [0.5] * 100]
    assert crc_select(losses, alpha=0.05) == 1


def test_calibrate_certifies_lossless_and_refuses_aggressive():
    # bundled corpus: lossless preserves decisions (0 risk); truncation does not.
    runner = DeterministicRunner()
    entries = load_corpus()
    cert = calibrate(entries, runner, alpha=0.30, delta=0.10, method="ltt")
    # at a generous α with enough samples, a safe level certifies and reports savings
    assert cert.level is not None
    assert cert.empirical_risk == 0.0  # the certified level changed no decisions
    assert "decision-change rate" in cert.guarantee


def test_certificate_is_honest_about_small_samples():
    # the bundled corpus is small (~28 turns) — it must NOT certify a tight 1% risk,
    # even though the safe levels have 0 empirical risk. Conservative = honest.
    runner = DeterministicRunner()
    cert = calibrate(load_corpus(), runner, alpha=0.01, delta=0.05, method="ltt")
    assert cert.level is None  # not enough data to support a 1% guarantee


def test_calibrate_validates_alpha():
    import pytest

    with pytest.raises(ValueError):
        calibrate(load_corpus(), DeterministicRunner(), alpha=1.5)
