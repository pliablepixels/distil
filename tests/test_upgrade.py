"""`distil version` + `distil upgrade` + installer detection (single source)."""

from __future__ import annotations

import argparse

from distil import __version__, onboard
from distil.cli import cmd_upgrade, cmd_version


def test_version_prints(capsys):
    assert cmd_version(argparse.Namespace()) == 0
    assert __version__ in capsys.readouterr().out


def test_install_method_detects_homebrew(monkeypatch):
    # brew wrapper on PATH → homebrew, and the command must be brew (not bare pip)
    monkeypatch.setattr(onboard.shutil, "which", lambda _n: "/usr/local/bin/distil")
    monkeypatch.setattr(onboard.Path, "resolve", lambda self: self)
    assert onboard.install_method() == "homebrew"
    assert onboard.upgrade_command("homebrew") == "brew upgrade distil"
    assert "brew uninstall" in onboard.uninstall_command("homebrew")


def test_command_maps_never_bare_pip_for_managed_installers():
    # brew/pipx/uv commands must be exact (no pip); pip cases carry the venv
    # caveat so a user never hits PEP 668 unguided
    for m in ("homebrew", "pipx", "uv"):
        assert "pip " not in onboard.upgrade_command(m)
        assert "pip " not in onboard.uninstall_command(m)
    assert "venv" in onboard.upgrade_command("pip")
    assert "venv" in onboard.uninstall_command("pip")


def test_upgrade_dry_run_does_not_run(monkeypatch, capsys):
    monkeypatch.setattr(onboard, "install_method", lambda: "pipx")
    ran = {"x": False}
    monkeypatch.setattr("subprocess.run", lambda *a, **k: ran.__setitem__("x", True))
    rc = cmd_upgrade(argparse.Namespace(dry_run=True))
    assert rc == 0 and not ran["x"]
    assert "pipx upgrade distil-llm" in capsys.readouterr().out


def test_upgrade_uvx_is_noop(monkeypatch, capsys):
    monkeypatch.setattr(onboard, "install_method", lambda: "uvx")
    assert cmd_upgrade(argparse.Namespace(dry_run=False)) == 0
    assert "nothing to upgrade" in capsys.readouterr().out.lower()
