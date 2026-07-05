"""Version-skew guard warns once when distil is upgraded on disk under a
long-lived proxy (P1-3)."""

from __future__ import annotations

import importlib.metadata

from distil.proxy import _warn_if_version_skew


def test_warns_once_on_skew(capsys, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "9.9.9")
    state = {"running": "1.0.0"}

    _warn_if_version_skew(state)
    err = capsys.readouterr().err
    assert "upgraded on disk to 9.9.9" in err
    assert "still runs 1.0.0" in err
    assert state["warned"] is True

    # Throttled + one-shot: a second call is silent even though disk still differs.
    _warn_if_version_skew(state)
    assert capsys.readouterr().err == ""


def test_silent_when_versions_match(capsys, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "1.2.3")
    state = {"running": "1.2.3"}
    _warn_if_version_skew(state)
    assert capsys.readouterr().err == ""
    assert not state.get("warned")


def test_throttled_between_checks(capsys, monkeypatch):
    calls = {"n": 0}

    def _v(_):
        calls["n"] += 1
        return "2.0.0"

    monkeypatch.setattr(importlib.metadata, "version", _v)
    state = {"running": "2.0.0", "checked": 0.0}
    _warn_if_version_skew(state)  # first call checks disk
    _warn_if_version_skew(state)  # within TTL → skipped before hitting disk
    assert calls["n"] == 1
