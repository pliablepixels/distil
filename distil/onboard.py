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


def detect() -> Env:
    return Env(
        os_name=platform.system() or "unknown",
        managers=[m for m in _MANAGERS if shutil.which(m)],
        agents=[(c, n) for c, n in _AGENTS if shutil.which(c)],
        has_anthropic=importlib.util.find_spec("anthropic") is not None,
        has_api_key=bool(os.environ.get("ANTHROPIC_API_KEY")),
        subscription=subscription_mode(),
    )


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
