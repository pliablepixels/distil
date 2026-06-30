"""`distil setup` — status-line wiring must be idempotent, conflict-safe, and
preserve other settings."""

from __future__ import annotations

import json

from distil.setup import wire_statusline


def test_wire_fresh(tmp_path) -> None:
    p = tmp_path / "settings.json"
    status, _ = wire_statusline(p)
    assert status == "ok"
    assert "distil" in json.loads(p.read_text())["statusLine"]["command"]


def test_wire_idempotent(tmp_path) -> None:
    p = tmp_path / "settings.json"
    wire_statusline(p)
    status, _ = wire_statusline(p)
    assert status == "exists"


def test_wire_conflict_needs_force_and_does_not_clobber(tmp_path) -> None:
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"statusLine": {"type": "command", "command": "mine.sh"}}))
    status, _ = wire_statusline(p)
    assert status == "conflict"
    assert "mine.sh" in p.read_text()  # untouched
    status, _ = wire_statusline(p, force=True)
    assert status == "ok"
    assert (tmp_path / "settings.json.bak").exists()  # backed up
    assert "distil" in json.loads(p.read_text())["statusLine"]["command"]


def test_wire_preserves_other_settings(tmp_path) -> None:
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"model": "opus", "env": {"X": "1"}}))
    wire_statusline(p)
    data = json.loads(p.read_text())
    assert data["model"] == "opus" and data["env"]["X"] == "1"
    assert "distil" in data["statusLine"]["command"]


def test_wire_rejects_non_object(tmp_path) -> None:
    p = tmp_path / "settings.json"
    p.write_text("[1, 2, 3]")
    status, _ = wire_statusline(p)
    assert status == "error"
