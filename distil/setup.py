"""``distil setup`` — wire the distil status line into Claude Code settings.

Replaces a manual ``settings.json`` edit. Idempotent, never clobbers an existing
status line without ``--force`` (and backs it up when it does), and preserves every
other setting.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_COMMAND = "distil statusline"


def default_settings_path() -> Path:
    return Path.home() / ".claude" / "settings.json"


def wire_statusline(
    settings_path: Path, *, command: str = DEFAULT_COMMAND, force: bool = False
) -> tuple[str, str]:
    """Wire the distil status line into ``settings_path``.

    Returns ``(status, message)`` where status is one of:
    ``ok`` (wired), ``exists`` (already distil), ``conflict`` (another line set,
    needs ``--force``), ``error`` (unreadable / not an object)."""
    data: object = {}
    if settings_path.exists():
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return ("error", f"{settings_path} is not valid JSON ({exc}) — fix it or edit by hand")
    if not isinstance(data, dict):
        return ("error", f"{settings_path} is not a JSON object")

    sl = data.get("statusLine")
    existing = sl.get("command", "") if isinstance(sl, dict) else ""
    if "distil" in (existing or ""):
        return ("exists", "distil status line already wired")
    if existing and not force:
        return (
            "conflict",
            f"a status line is already set ({existing!r}); "
            "re-run with --force to replace it (it'll be backed up first)",
        )
    if existing:  # force: back up the current settings before replacing
        settings_path.with_name(settings_path.name + ".bak").write_text(
            settings_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    data["statusLine"] = {"type": "command", "command": command, "padding": 0}
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return ("ok", f"wired the distil status line into {settings_path}")
