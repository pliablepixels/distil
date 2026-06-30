"""`distil statusline` — the Claude Code plugin status line.

It must always print one line, never raise, and reflect the local savings ledger.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path


from distil import ledger
from distil.cli import cmd_statusline


def test_humanize_tokens():
    assert ledger._human(0) == "0"
    assert ledger._human(999) == "999"
    assert ledger._human(14_417) == "14.4K"
    assert ledger._human(1_200_000) == "1.2M"


def _run(monkeypatch, capsys, summary, stdin="{}", no_color=True):
    monkeypatch.setattr(ledger, "summary", lambda *a, **k: summary)
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    rc = cmd_statusline(argparse.Namespace(no_color=no_color))
    return rc, capsys.readouterr().out.strip()


def test_empty_ledger_shows_hint(monkeypatch, capsys):
    rc, out = _run(monkeypatch, capsys, ledger.LedgerSummary(0, 0.0, 0, {}))
    assert rc == 0
    assert out.startswith("distil")
    assert "no savings yet" in out


def test_populated_ledger(monkeypatch, capsys):
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "0")  # force metered: dollars shown
    s = ledger.LedgerSummary(
        3,
        0.0400,
        21_600,
        {"live-proxy": 0.04},
        total_baseline_tokens=50_000,
        total_distil_tokens=28_400,
        total_baseline_dollars=0.10,
        total_distil_dollars=0.06,
    )
    rc, out = _run(monkeypatch, capsys, s)
    assert rc == 0
    assert "50.0K→28.4K tok" in out  # orig → compressed, not just the delta
    assert "$0.10→$0.06" in out
    assert "3 runs" in out


def test_subscription_hides_notional_dollars(monkeypatch, capsys):
    # On a flat-rate subscription the dollar figure is notional, so it's hidden;
    # the orig -> compressed token reduction (the real win) still shows.
    monkeypatch.setenv("DISTIL_SUBSCRIPTION", "1")
    s = ledger.LedgerSummary(
        3,
        0.04,
        21_600,
        {},
        total_baseline_tokens=50_000,
        total_distil_tokens=28_400,
        total_baseline_dollars=0.10,
        total_distil_dollars=0.06,
    )
    rc, out = _run(monkeypatch, capsys, s)
    assert rc == 0
    assert "50.0K→28.4K tok" in out
    assert "$" not in out


def test_singular_run(monkeypatch, capsys):
    s = ledger.LedgerSummary(1, 0.01, 500, {"x": 0.01})
    _rc, out = _run(monkeypatch, capsys, s)
    assert "1 run" in out and "1 runs" not in out


def test_model_name_from_stdin(monkeypatch, capsys):
    s = ledger.LedgerSummary(2, 0.02, 1000, {})
    _rc, out = _run(monkeypatch, capsys, s, stdin='{"model":{"display_name":"Claude Opus 4.8"}}')
    assert "Claude Opus 4.8" in out


def test_never_raises_on_bad_stdin(monkeypatch, capsys):
    s = ledger.LedgerSummary(1, 0.01, 100, {})
    rc, out = _run(monkeypatch, capsys, s, stdin="not json {{{")
    assert rc == 0
    assert out.startswith("distil")


def test_color_codes_present_by_default(monkeypatch, capsys):
    s = ledger.LedgerSummary(1, 0.01, 100, {})
    _rc, out = _run(monkeypatch, capsys, s, no_color=False)
    assert "\033[" in out


_REPO = Path(__file__).resolve().parent.parent


def test_main_statusline_flushes_and_hard_exits(monkeypatch, capsys):
    """The fix's mechanism, version-independently: main(["statusline"]) delivers
    its line, then hard-exits via os._exit so the interpreter's shutdown flush
    can never fault on a pipe the consumer already closed. We patch os._exit
    (which would otherwise kill the test runner) to observe the call."""
    import os

    import pytest

    import distil.cli as cli

    exited = {}

    def fake_exit(code):
        exited["code"] = code
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", fake_exit)
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    monkeypatch.setattr(ledger, "summary", lambda *a, **k: ledger.LedgerSummary(1, 0.01, 100, {}))

    with pytest.raises(SystemExit):
        cli.main(["statusline"])

    assert exited.get("code") == 0
    assert "distil" in capsys.readouterr().out


def test_no_broken_pipe_when_reader_closes_early(tmp_path):
    """End-to-end: invoke distil the way its console-script does — a .py run as
    __main__ that imports the package — with the reader closing the pipe early.
    No BrokenPipeError may reach stderr. (The `-m` form does NOT reproduce this;
    it only manifests on Python 3.13+, so on older interpreters this is a smoke
    test that must simply stay clean.)"""
    wrapper = tmp_path / "run.py"
    wrapper.write_text("import sys\nfrom distil.cli import main\nsys.exit(main())\n")
    for _ in range(8):
        proc = subprocess.run(
            f"{sys.executable} {wrapper} statusline | true",
            shell=True,
            cwd=_REPO,
            input="{}",
            text=True,
            stderr=subprocess.PIPE,
        )
        assert "BrokenPipeError" not in proc.stderr, proc.stderr
        assert "Broken pipe" not in proc.stderr, proc.stderr


def test_render_dashboard_shows_orig_and_compressed():
    s = ledger.LedgerSummary(
        2,
        5.0,
        1_000_000,
        {"live-proxy": 5.0},
        total_baseline_tokens=2_000_000,
        total_distil_tokens=1_000_000,
        total_baseline_dollars=10.0,
        total_distil_dollars=5.0,
    )
    out = ledger.render_dashboard(s, change_rate=0.01, samples=200, color=False)
    assert "2.0M → 1.0M" in out  # orig -> compressed tokens
    assert "50.0% trimmed" in out
    assert "$10.00 → $5.00" in out
    assert "99.0%" in out  # decision-equivalence = 1 - change_rate
    assert "200 samples" in out


def test_render_dashboard_subscription_hides_cost():
    s = ledger.LedgerSummary(
        1,
        5.0,
        1_000_000,
        {},
        total_baseline_tokens=2_000_000,
        total_distil_tokens=1_000_000,
        total_baseline_dollars=10.0,
        total_distil_dollars=5.0,
    )
    out = ledger.render_dashboard(s, subscription=True, color=False)
    assert "notional" in out
    assert "$10.00 → $5.00" not in out
