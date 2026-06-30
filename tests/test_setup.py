"""`distil setup` — status-line wiring must be idempotent, conflict-safe, and
preserve other settings."""

from __future__ import annotations

import json

from distil.setup import (
    alias_body,
    detect_shell,
    env_body,
    remove_managed,
    service_spec,
    wire_statusline,
    write_managed,
)


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


# ── distil default: managed-block + shell detection (reliable across machines) ──


def test_managed_block_add_idempotent_update_remove(tmp_path) -> None:
    rc = tmp_path / ".zshrc"
    rc.write_text("# my rc\nexport FOO=1\n")

    st, _ = write_managed(rc, "alias claude='distil wrap -- claude'")
    assert st == "ok"
    assert "alias claude=" in rc.read_text()
    assert "export FOO=1" in rc.read_text()  # original preserved

    st, _ = write_managed(rc, "alias claude='distil wrap -- claude'")
    assert st == "exists"  # identical → no churn

    st, _ = write_managed(rc, "alias claude='distil wrap --expand -- claude'")
    assert st == "updated"  # body changed → single block replaced, not duplicated
    assert rc.read_text().count("alias claude=") == 1
    assert (tmp_path / ".zshrc.bak").exists()  # backed up before change

    st, _ = remove_managed(rc)
    assert st == "ok"
    assert "distil" not in rc.read_text()
    assert "export FOO=1" in rc.read_text()  # untouched


def test_remove_managed_absent_is_safe(tmp_path) -> None:
    rc = tmp_path / ".bashrc"
    assert remove_managed(rc)[0] == "absent"  # no file
    rc.write_text("plain\n")
    assert remove_managed(rc)[0] == "absent"  # file but no block
    assert rc.read_text() == "plain\n"


def test_detect_shell_explicit_shell_wins_over_existing_rc(tmp_path, monkeypatch) -> None:
    # Both a .zshrc and a .bashrc exist; $SHELL must decide, not file-existence.
    home = tmp_path
    (home / ".zshrc").write_text("")
    (home / ".bashrc").write_text("")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr("distil.setup._is_windows", lambda: False)
    monkeypatch.setenv("SHELL", "/bin/bash")
    name, rc = detect_shell()
    assert name == "bash" and rc.name == ".bashrc"
    monkeypatch.setenv("SHELL", "/usr/bin/fish")
    assert detect_shell()[0] == "fish"


def test_detect_shell_fallback_when_shell_unset(tmp_path, monkeypatch) -> None:
    home = tmp_path
    (home / ".zshrc").write_text("")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr("distil.setup._is_windows", lambda: False)
    monkeypatch.setenv("SHELL", "")
    assert detect_shell() == ("zsh", home / ".zshrc")


def test_shell_specific_bodies() -> None:
    assert alias_body("claude", "lossless-only", shell="zsh") == (
        "alias claude='distil wrap --lossless-only -- claude'"
    )
    # PowerShell needs a function (forwards @args), not an alias.
    assert "function claude" in alias_body("claude", "expand", shell="powershell")
    # fish uses `set -gx`, posix `export`, powershell `$env:`.
    assert env_body(8788, shell="fish") == "set -gx ANTHROPIC_BASE_URL http://127.0.0.1:8788"
    assert env_body(8788, shell="zsh") == "export ANTHROPIC_BASE_URL=http://127.0.0.1:8788"
    assert env_body(8788, shell="powershell").startswith("$env:ANTHROPIC_BASE_URL")


def test_service_spec_per_platform(monkeypatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    path, content, load = service_spec(8788, "lossless-only")
    assert path.name == "com.distil.proxy.plist"
    assert "--lossless-only" in content and "8788" in content
    assert "launchctl" in load

    monkeypatch.setattr("platform.system", lambda: "Linux")
    path, content, load = service_spec(9000, "expand")
    assert path.name == "distil-proxy.service"
    assert "ExecStart=" in content and "--expand" in content and "9000" in content
    assert "systemctl" in load

    monkeypatch.setattr("platform.system", lambda: "Windows")
    assert service_spec(8788, "expand") == (None, None, None)  # unsupported → graceful


# ── distil default: cmd-level glue (install / undo / always-on), all sandboxed ──


def _default_args(tmp_path, **over):
    import argparse

    base = dict(
        rc=str(tmp_path / ".zshrc"),
        agent="claude",
        mode="lossless-only",
        port=8788,
        undo=False,
        always_on=False,
        no_start=True,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_cmd_default_installs_alias_and_undoes(tmp_path, capsys) -> None:
    from distil import cli

    rc = tmp_path / ".zshrc"
    rc.write_text("# keep me\n")
    assert cli.cmd_default(_default_args(tmp_path)) == 0
    assert "alias claude='distil wrap --lossless-only -- claude'" in rc.read_text()
    assert "# keep me" in rc.read_text()

    assert cli.cmd_default(_default_args(tmp_path, undo=True)) == 0
    assert "distil" not in rc.read_text()
    assert "# keep me" in rc.read_text()


def test_cmd_default_always_on_writes_service_and_env(tmp_path, monkeypatch, capsys) -> None:
    from distil import cli, setup

    svc = tmp_path / "svc.plist"
    monkeypatch.setattr(
        setup, "service_spec", lambda port, mode: (svc, f"PLIST {port} --{mode}", "true")
    )
    # --no-start means we never shell out; assert subprocess is not invoked anyway.
    import subprocess

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no-start"))
    )
    rc = tmp_path / ".zshrc"
    assert cli.cmd_default(_default_args(tmp_path, always_on=True, mode="lossless-only")) == 0
    assert svc.exists() and "8788" in svc.read_text()
    assert "export ANTHROPIC_BASE_URL=http://127.0.0.1:8788" in rc.read_text()

    # undo cleans up both the rc block and the service file
    assert cli.cmd_default(_default_args(tmp_path, undo=True)) == 0
    assert not svc.exists()
    assert "ANTHROPIC_BASE_URL" not in rc.read_text()


def test_cmd_default_always_on_unsupported_platform(tmp_path, monkeypatch, capsys) -> None:
    from distil import cli, setup

    monkeypatch.setattr(setup, "service_spec", lambda port, mode: (None, None, None))
    rc = cli.cmd_default(_default_args(tmp_path, always_on=True))
    assert rc == 1  # graceful, non-zero
    assert "isn't supported" in capsys.readouterr().out
