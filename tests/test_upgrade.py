"""`distil version` + `distil upgrade` — installer-aware self-update."""

from __future__ import annotations

import argparse

from distil import __version__
from distil.cli import _detect_installer, cmd_upgrade, cmd_version


def test_version_prints(capsys):
    assert cmd_version(argparse.Namespace()) == 0
    assert __version__ in capsys.readouterr().out


def test_detect_installer_by_path(monkeypatch):
    import distil.cli as cli

    cases = {
        "/usr/local/Cellar/distil/1.8.1/bin/distil": ("homebrew", "brew upgrade distil"),
        "/Users/x/.local/pipx/venvs/distil-llm/bin/distil": ("pipx", "pipx upgrade distil-llm"),
        "/Users/x/.local/share/uv/tools/distil-llm/bin/distil": ("uv", "uv tool upgrade distil-llm"),
    }
    for path, (name, cmd) in cases.items():
        monkeypatch.setattr(cli.shutil if hasattr(cli, "shutil") else __import__("shutil"),
                            "which", lambda _a, _p=path: _p)
        # resolve() is identity for these absolute non-symlink test paths
        monkeypatch.setattr(cli.Path, "resolve", lambda self: self)
        got_name, got_cmd = _detect_installer()
        assert (got_name, got_cmd) == (name, cmd), path


def test_upgrade_dry_run_does_not_run(monkeypatch, capsys):
    import distil.cli as cli

    monkeypatch.setattr(cli, "_detect_installer", lambda: ("pipx", "pipx upgrade distil-llm"))
    called = {"ran": False}
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: called.__setitem__("ran", True)
    )
    rc = cmd_upgrade(argparse.Namespace(dry_run=True))
    assert rc == 0 and not called["ran"]
    assert "pipx upgrade distil-llm" in capsys.readouterr().out


def test_upgrade_unknown_installer_explains(monkeypatch, capsys):
    import distil.cli as cli

    monkeypatch.setattr(cli, "_detect_installer", lambda: ("unknown", None))
    rc = cmd_upgrade(argparse.Namespace(dry_run=False))
    assert rc == 1
    out = capsys.readouterr().out
    assert "pipx upgrade distil-llm" in out and "brew upgrade distil" in out
