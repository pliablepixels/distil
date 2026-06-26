"""Tests for the trajectory-level certificate (E10) and its conformal primitive."""

from __future__ import annotations

from distil.conformal import certified_risk_bound, hb_pvalue


def test_certified_risk_bound_is_above_rhat_and_certifies():
    # The (1−δ) upper bound must exceed the point estimate and itself certify at δ.
    rhat, n, delta = 0.10, 500, 0.05
    beta = certified_risk_bound(rhat, n, delta)
    assert beta > rhat
    assert hb_pvalue(rhat, n, beta) <= delta + 1e-9  # the bound is reject-able ⇒ certified
    # Just below the bound it should NOT certify (it is the tightest such α).
    assert hb_pvalue(rhat, n, beta - 0.01) > delta


def test_certified_risk_bound_tightens_with_n():
    # More calibration data ⇒ tighter (smaller) upper bound at the same rhat.
    wide = certified_risk_bound(0.10, 100, 0.05)
    tight = certified_risk_bound(0.10, 2000, 0.05)
    assert tight < wide


def test_certified_risk_bound_degenerate():
    assert certified_risk_bound(0.5, 0, 0.05) == 1.0  # no data ⇒ no guarantee


def test_trajectory_certificate_holds_out_of_sample():
    # On the committed E8 outcomes the gated trajectory certificate's OOS coverage must
    # meet the 1−δ target for both divergence and harm (the guarantee actually holds).
    from benchmarks.trajectory_certificate import certificate

    cert = certificate("full", "distil_gated", delta=0.05)
    assert cert["n"] == 500
    for name in ("divergence", "harm"):
        c = cert[name]
        assert c["certified_bound"] > c["empirical_rate"]  # bound above observed
        # OOS coverage at or above target (small tolerance for split-sampling noise).
        assert c["oos"]["coverage"] >= c["oos"]["target"] - 0.02
