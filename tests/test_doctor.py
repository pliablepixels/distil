"""`distil doctor` — setup diagnosis. Checks must never crash, and the proxy
self-test must round-trip a request through an in-process upstream."""

from __future__ import annotations

from distil import doctor


def test_diagnose_runs_every_check_without_crashing() -> None:
    checks = doctor.diagnose()
    assert checks
    names = {c.name for c in checks}
    assert "distil" in names
    assert "proxy self-test" in names
    for c in checks:
        assert c.status in (doctor.OK, doctor.WARN, doctor.INFO, doctor.FAIL)
        assert c.detail  # every check explains itself


def test_proxy_selftest_round_trips() -> None:
    # The headline check: a request must route through the distil proxy to an
    # in-process fake upstream and back — no network, fully self-contained.
    c = doctor._check_proxy_selftest()
    assert c.status == doctor.OK, c.detail


def test_version_check_ok() -> None:
    c = doctor._check_version()
    assert c.status == doctor.OK  # we run on a supported Python


def test_subscription_mode_env_override(monkeypatch) -> None:
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
    assert doctor.subscription_mode() is True
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")
    assert doctor.subscription_mode() is False


def test_subscription_mode_metered_key_means_real_dollars(monkeypatch) -> None:
    # A metered API key set, no explicit override → dollars are real, not notional.
    monkeypatch.delenv("DISTIL_SUBSCRIPTION", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert doctor.subscription_mode() is False
