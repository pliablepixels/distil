"""``distil onboard`` — one command that sets up distil and guides you to use it.

Detects your environment (OS, package managers, agent CLIs, install method, the
optional anthropic extra, Claude Code + subscription), wires the savings status
line, and prints a next-steps guide tailored to what it found — how to route your
agent, validate outcomes with shadow mode, watch savings, and re-verify. Works on
macOS and Windows; mutating actions are gated.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
from dataclasses import dataclass, field

from .doctor import subscription_mode

# Agent CLIs we know how to route, in priority order.
_AGENTS = [("claude", "Claude Code"), ("codex", "Codex"), ("gemini", "Gemini CLI")]
_MANAGERS = ("pipx", "uv", "brew", "scoop", "pip")


@dataclass
class Env:
    os_name: str
    managers: list[str] = field(default_factory=list)
    agents: list[tuple[str, str]] = field(default_factory=list)  # (cmd, label)
    has_anthropic: bool = False
    has_api_key: bool = False
    subscription: bool = False
    installed_version: str = ""
    method: str = "pip"  # how distil is installed: pipx | uv | uvx | pip


def detect() -> Env:
    from . import __version__

    return Env(
        os_name=platform.system() or "unknown",
        managers=[m for m in _MANAGERS if shutil.which(m)],
        agents=[(c, n) for c, n in _AGENTS if shutil.which(c)],
        has_anthropic=importlib.util.find_spec("anthropic") is not None,
        has_api_key=bool(os.environ.get("ANTHROPIC_API_KEY")),
        subscription=subscription_mode(),
        installed_version=__version__,
        method=install_method(),
    )


def install_method() -> str:
    """How the running distil is installed — drives the right upgrade command."""
    from . import __file__ as pkg_file

    p = (pkg_file or "").replace(os.sep, "/").lower()
    if "/pipx/" in p:
        return "pipx"
    if "/uv/tools/" in p:
        return "uv"
    if "/uv/" in p or "/.cache/uv/" in p:
        return "uvx"  # ephemeral run — nothing persistent to upgrade
    return "pip"


def upgrade_command(method: str) -> str:
    return {
        "pipx": "pipx upgrade distil-llm",
        "uv": "uv tool upgrade distil-llm",
        "uvx": "uvx --from distil-llm@latest distil onboard   # uvx runs the latest each time",
        "pip": "pip install --upgrade distil-llm",
    }.get(method, "pip install --upgrade distil-llm")


def latest_pypi_version(timeout: float = 2.5) -> str | None:
    """Latest distil-llm version on PyPI, or None if offline / the check fails."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(
            "https://pypi.org/pypi/distil-llm/json", timeout=timeout
        ) as resp:
            return json.load(resp)["info"]["version"]
    except Exception:  # noqa: BLE001 — offline / DNS / timeout: just skip the check
        return None


def _ver_tuple(s: str) -> tuple[tuple[int, ...], bool]:
    import re

    nums = re.findall(r"\d+", s.split("+")[0])[:3]
    base = tuple(int(n) for n in nums) + (0,) * (3 - len(nums))
    is_pre = bool(re.search(r"(dev|rc|a|b)\d*", s))
    return base, is_pre


def is_outdated(installed: str, latest: str | None) -> bool:
    """True if a newer *released* version than ``installed`` is available."""
    if not latest:
        return False
    bi, pre_i = _ver_tuple(installed)
    bl, _pre_l = _ver_tuple(latest)
    if bi != bl:
        return bi < bl
    return pre_i  # same base number, but ours is a pre-release of it → older


def best_install_command(managers: list[str]) -> str:
    """The recommended way to (re)install distil persistently on this machine."""
    if "pipx" in managers:
        return "pipx install distil-llm"
    if "uv" in managers:
        return "uv tool install distil-llm"
    if "brew" in managers:
        return "brew install pipx && pipx install distil-llm"
    if "scoop" in managers:  # Windows
        return "scoop install pipx && pipx install distil-llm"
    return "python -m pip install --user pipx && pipx install distil-llm"


def next_steps(env: Env) -> list[tuple[str, str, str]]:
    """Tailored guide as (title, command, note) rows."""
    agent = env.agents[0][0] if env.agents else "claude"
    steps: list[tuple[str, str, str]] = []

    if not env.agents:
        steps.append(
            (
                "Install a coding agent",
                "# e.g. Claude Code, Codex, or Gemini CLI",
                "no agent CLI detected on PATH — install one, then re-run distil onboard",
            )
        )

    # Routing — mode depends on billing.
    if env.subscription:
        steps.append(
            (
                "Route your agent (subscription-safe)",
                f"distil wrap --lossless-only -- {agent}",
                "flat-rate plan: trims context, ToS-safe (no lossy digest)",
            )
        )
    else:
        steps.append(
            (
                "Route your agent",
                f"distil wrap --expand -- {agent}",
                "metered key: aggressive reversible digest; the model recovers detail on demand",
            )
        )

    steps.append(
        (
            "Validate it preserved your outcomes (shadow)",
            f"distil wrap --shadow 0.1 -- {agent}",
            "runs 10% of requests twice and checks the next action is unchanged — then: distil shadow-stats",
        )
    )
    steps.append(
        (
            "Watch your savings",
            "distil dashboard",
            "live terminal view (or distil leaderboard for a snapshot)",
        )
    )
    steps.append(
        (
            "Run the test gate anytime",
            "distil bench",
            "corpus-wide non-inferiority gate — no API key needed",
        )
    )
    steps.append(
        (
            "Re-verify your setup",
            "distil doctor",
            "ledger, shadow, proxy self-test, wiring",
        )
    )
    if not env.has_anthropic:
        steps.append(
            (
                "Optional: live grading / billing-grade tokens",
                "pipx inject distil-llm anthropic   # then set ANTHROPIC_API_KEY",
                "only needed for --runner/--tokenizer anthropic",
            )
        )
    return steps


def report(env: Env, latest: str | None) -> dict:
    """A structured snapshot for an agent to reason over (`distil onboard --json`).

    Pure facts + recommendations — no actions taken. The intelligence (deciding,
    asking, running steps) lives in the agent/skill that consumes this."""
    outdated = is_outdated(env.installed_version, latest)
    return {
        "os": env.os_name,
        "agents": [c for c, _ in env.agents],
        "primary_agent": env.agents[0][0] if env.agents else None,
        "package_managers": env.managers,
        "billing": "subscription" if env.subscription else "metered",
        "installed_version": env.installed_version,
        "latest_version": latest,
        "upgrade_available": outdated,
        "install_method": env.method,
        "upgrade_command": upgrade_command(env.method) if outdated else None,
        "anthropic_extra": env.has_anthropic,
        "api_key": env.has_api_key,
        "best_install_command": best_install_command(env.managers),
        "next_steps": [
            {"title": t, "command": cmd, "note": n} for t, cmd, n in next_steps(env)
        ],
    }
