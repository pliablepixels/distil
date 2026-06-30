"""`distil onboard` — detection + a guide tailored to the environment."""

from __future__ import annotations

from distil import onboard
from distil.onboard import Env


def test_detect_returns_env_shape() -> None:
    env = onboard.detect()
    assert env.os_name
    assert isinstance(env.managers, list)
    assert isinstance(env.agents, list)


def test_best_install_prefers_pipx_then_uv_then_fallback() -> None:
    assert onboard.best_install_command(["pipx", "uv"]) == "pipx install distil-llm"
    assert "uv tool install" in onboard.best_install_command(["uv"])
    assert "pipx install distil-llm" in onboard.best_install_command([])  # fallback bootstraps pipx


def test_guide_subscription_uses_lossless() -> None:
    env = Env(os_name="Darwin", agents=[("claude", "Claude Code")], subscription=True, has_anthropic=True)
    cmds = [cmd for _, cmd, _ in onboard.next_steps(env)]
    assert any("--lossless-only -- claude" in c for c in cmds)
    assert not any("--expand -- claude" in c for c in cmds)


def test_guide_metered_uses_expand_and_primary_agent() -> None:
    env = Env(os_name="Darwin", agents=[("codex", "Codex")], subscription=False, has_anthropic=True)
    cmds = [cmd for _, cmd, _ in onboard.next_steps(env)]
    assert any("--expand -- codex" in c for c in cmds)  # routes the detected primary agent


def test_guide_no_agent_prompts_install() -> None:
    env = Env(os_name="Windows", agents=[], subscription=False)
    titles = [t for t, _, _ in onboard.next_steps(env)]
    assert any("Install a coding agent" in t for t in titles)


def test_guide_offers_anthropic_extra_when_missing() -> None:
    env = Env(os_name="Darwin", agents=[("claude", "Claude Code")], has_anthropic=False)
    cmds = [cmd for _, cmd, _ in onboard.next_steps(env)]
    assert any("pipx inject distil-llm anthropic" in c for c in cmds)


def test_guide_always_includes_shadow_and_doctor() -> None:
    env = Env(os_name="Linux", agents=[("claude", "Claude Code")], has_anthropic=True)
    cmds = [cmd for _, cmd, _ in onboard.next_steps(env)]
    assert any("--shadow 0.1" in c for c in cmds)
    assert any(c.strip() == "distil doctor" for c in cmds)
