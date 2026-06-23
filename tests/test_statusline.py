"""`distil statusline` — the Claude Code plugin status line.

It must always print one line, never raise, and reflect the local savings ledger.
"""

from __future__ import annotations

import argparse
import io

from distil import ledger
from distil.cli import _humanize_tokens, cmd_statusline


def test_humanize_tokens():
    assert _humanize_tokens(0) == "0"
    assert _humanize_tokens(999) == "999"
    assert _humanize_tokens(14_417) == "14.4K"
    assert _humanize_tokens(1_200_000) == "1.2M"


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
    s = ledger.LedgerSummary(3, 0.0400, 21_600, {"live-proxy": 0.04})
    rc, out = _run(monkeypatch, capsys, s)
    assert rc == 0
    assert "21.6K tok" in out
    assert "$0.0400" in out
    assert "3 runs" in out


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
