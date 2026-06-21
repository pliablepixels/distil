"""Tests for distil.ingest — real-trace ingestion.

Covers:
* Anthropic request body → correct Block kinds/stabilities
* OpenAI request body → correct Block kinds/stabilities
* ingest_session over 2 requests → 2-turn Trajectory with byte-stable prefix
* cache_aware.simulate runs on an ingested trajectory and returns total_dollars > 0
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from distil.ingest import (
    ingest_anthropic_request,
    ingest_file,
    ingest_openai_request,
    ingest_session,
)
from distil.trajectory import Kind, Stability


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ANTHROPIC_BODY: dict = {
    "model": "claude-opus-4-8",
    "system": "You are a helpful assistant.",
    "tools": [
        {
            "name": "get_weather",
            "description": "Get the current weather.",
            "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}},
        }
    ],
    "messages": [
        {
            "role": "user",
            "content": "What is the weather in Paris?",
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check the weather for you."},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01",
                    "content": "Sunny, 22°C",
                }
            ],
        },
    ],
}

OPENAI_BODY: dict = {
    "model": "gpt-4o",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather.",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
            },
        }
    ],
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "content": "Let me check the weather for you."},
        {"role": "tool", "tool_call_id": "call_01", "content": "Sunny, 22°C"},
    ],
}


# ---------------------------------------------------------------------------
# Anthropic tests
# ---------------------------------------------------------------------------


def test_anthropic_system_block():
    blocks = ingest_anthropic_request(ANTHROPIC_BODY)
    system_blocks = [b for b in blocks if b.kind == Kind.SYSTEM]
    assert len(system_blocks) == 1
    sb = system_blocks[0]
    assert sb.id == "system"
    assert sb.stability == Stability.STABLE
    assert sb.decision_relevant is False
    assert "helpful assistant" in sb.text


def test_anthropic_tools_block():
    blocks = ingest_anthropic_request(ANTHROPIC_BODY)
    tool_blocks = [b for b in blocks if b.kind == Kind.TOOLS]
    assert len(tool_blocks) == 1
    tb = tool_blocks[0]
    assert tb.id == "tools"
    assert tb.stability == Stability.STABLE
    # the text should be valid JSON containing the tool name
    parsed = json.loads(tb.text)
    assert isinstance(parsed, list)
    assert parsed[0]["name"] == "get_weather"


def test_anthropic_tool_output_volatile():
    blocks = ingest_anthropic_request(ANTHROPIC_BODY)
    tool_output_blocks = [b for b in blocks if b.kind == Kind.TOOL_OUTPUT]
    assert len(tool_output_blocks) == 1
    tob = tool_output_blocks[0]
    assert tob.stability == Stability.VOLATILE
    assert "Sunny" in tob.text


def test_anthropic_ordering_stable_before_volatile():
    """STABLE blocks must precede any VOLATILE blocks in the ordered list."""
    blocks = ingest_anthropic_request(ANTHROPIC_BODY)
    last_stable_idx = max(
        (i for i, b in enumerate(blocks) if b.stability == Stability.STABLE), default=-1
    )
    first_volatile_idx = min(
        (i for i, b in enumerate(blocks) if b.stability == Stability.VOLATILE), default=len(blocks)
    )
    assert last_stable_idx < first_volatile_idx, (
        f"A STABLE block appears after a VOLATILE block "
        f"(last_stable={last_stable_idx}, first_volatile={first_volatile_idx})"
    )


def test_anthropic_no_image_blocks():
    """Images in content must be silently ignored."""
    body = {
        **ANTHROPIC_BODY,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "abc"},
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ],
    }
    blocks = ingest_anthropic_request(body)
    # Only one user text block — no image block
    user_blocks = [b for b in blocks if b.kind == Kind.USER]
    assert len(user_blocks) == 1
    assert "Describe this image" in user_blocks[0].text


def test_anthropic_system_as_list():
    """system field as a list-of-content-blocks should still produce one SYSTEM block."""
    body = {
        **ANTHROPIC_BODY,
        "system": [
            {"type": "text", "text": "Part one."},
            {"type": "text", "text": "Part two."},
        ],
    }
    blocks = ingest_anthropic_request(body)
    system_blocks = [b for b in blocks if b.kind == Kind.SYSTEM]
    assert len(system_blocks) == 1
    assert "Part one" in system_blocks[0].text
    assert "Part two" in system_blocks[0].text


# ---------------------------------------------------------------------------
# OpenAI tests
# ---------------------------------------------------------------------------


def test_openai_system_block():
    blocks = ingest_openai_request(OPENAI_BODY)
    system_blocks = [b for b in blocks if b.kind == Kind.SYSTEM]
    assert len(system_blocks) == 1
    sb = system_blocks[0]
    assert sb.id == "system"
    assert sb.stability == Stability.STABLE
    assert "helpful assistant" in sb.text


def test_openai_tools_block():
    blocks = ingest_openai_request(OPENAI_BODY)
    tool_blocks = [b for b in blocks if b.kind == Kind.TOOLS]
    assert len(tool_blocks) == 1
    tb = tool_blocks[0]
    assert tb.stability == Stability.STABLE
    parsed = json.loads(tb.text)
    assert parsed[0]["function"]["name"] == "get_weather"


def test_openai_tool_output():
    blocks = ingest_openai_request(OPENAI_BODY)
    tool_output_blocks = [b for b in blocks if b.kind == Kind.TOOL_OUTPUT]
    assert len(tool_output_blocks) == 1
    tob = tool_output_blocks[0]
    assert tob.stability == Stability.VOLATILE
    assert "Sunny" in tob.text


def test_openai_ordering_stable_before_volatile():
    blocks = ingest_openai_request(OPENAI_BODY)
    last_stable_idx = max(
        (i for i, b in enumerate(blocks) if b.stability == Stability.STABLE), default=-1
    )
    first_volatile_idx = min(
        (i for i, b in enumerate(blocks) if b.stability == Stability.VOLATILE), default=len(blocks)
    )
    assert last_stable_idx < first_volatile_idx


def test_openai_assistant_history():
    blocks = ingest_openai_request(OPENAI_BODY)
    history_blocks = [b for b in blocks if b.kind == Kind.HISTORY]
    assert len(history_blocks) >= 1
    assert history_blocks[0].stability == Stability.SETTLING


# ---------------------------------------------------------------------------
# ingest_session — multi-turn Trajectory
# ---------------------------------------------------------------------------

# Simulate 2 consecutive API calls that share a stable system+tools prefix
# but grow the message list (as a real agentic session would)
SESSION_REQUEST_1: dict = {
    "model": "claude-opus-4-8",
    "system": "You are an SRE agent.",
    "tools": [{"name": "get_metrics", "description": "Fetch metrics.", "input_schema": {}}],
    "messages": [
        {"role": "user", "content": "Investigate the disk alert."},
    ],
}

SESSION_REQUEST_2: dict = {
    "model": "claude-opus-4-8",
    "system": "You are an SRE agent.",
    "tools": [{"name": "get_metrics", "description": "Fetch metrics.", "input_schema": {}}],
    "messages": [
        {"role": "user", "content": "Investigate the disk alert."},
        {"role": "assistant", "content": "Running get_metrics now."},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "disk=95%"},
            ],
        },
        {"role": "user", "content": "Remediate per policy."},
    ],
}


def test_ingest_session_returns_trajectory():
    traj = ingest_session([SESSION_REQUEST_1, SESSION_REQUEST_2], provider="anthropic")
    assert traj.model == "claude-opus-4-8"
    assert len(traj.turns) == 2
    assert traj.turns[0].index == 0
    assert traj.turns[1].index == 1


def test_ingest_session_stable_prefix_byte_identical():
    """system and tools blocks must have byte-identical text across both turns."""
    traj = ingest_session([SESSION_REQUEST_1, SESSION_REQUEST_2], provider="anthropic")

    def stable_texts(turn_idx: int) -> dict[str, str]:
        return {
            b.id: b.text for b in traj.turns[turn_idx].blocks if b.stability == Stability.STABLE
        }

    t0 = stable_texts(0)
    t1 = stable_texts(1)

    # Both turns must have the same stable block ids
    assert set(t0.keys()) == set(t1.keys()), f"stable ids differ: {set(t0)} vs {set(t1)}"
    for bid in t0:
        assert t0[bid] == t1[bid], f"stable block {bid!r} text differs between turns"


def test_ingest_session_turn_0_has_user_block():
    traj = ingest_session([SESSION_REQUEST_1, SESSION_REQUEST_2], provider="anthropic")
    user_blocks = [b for b in traj.turns[0].blocks if b.kind == Kind.USER]
    assert len(user_blocks) >= 1


def test_ingest_session_turn_1_has_tool_output():
    traj = ingest_session([SESSION_REQUEST_1, SESSION_REQUEST_2], provider="anthropic")
    tool_out = [b for b in traj.turns[1].blocks if b.kind == Kind.TOOL_OUTPUT]
    assert len(tool_out) >= 1


def test_ingest_session_custom_id_and_model():
    traj = ingest_session(
        [SESSION_REQUEST_1],
        provider="anthropic",
        id="my-session",
        model="claude-sonnet-4-6",
    )
    assert traj.id == "my-session"
    assert traj.model == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# cache_aware.simulate integration
# ---------------------------------------------------------------------------


def test_simulate_on_ingested_trajectory():
    """simulate() must accept an ingested trajectory and return total_dollars > 0."""
    from distil.compress import cache_aware
    from distil import pricing

    traj = ingest_session(
        [SESSION_REQUEST_1, SESSION_REQUEST_2],
        provider="anthropic",
        model="claude-opus-4-8",
    )
    p = pricing.get("claude-opus-4-8")
    result = cache_aware.simulate(traj, p, strategy="distil", caching=True)
    assert result.total_dollars > 0, "expected non-zero cost for a real-shape trajectory"
    assert result.total_input_tokens > 0


def test_simulate_distil_vs_naive_on_session():
    """distil strategy must not cost more than naive when caching is on."""
    from distil.compress import cache_aware
    from distil import pricing

    traj = ingest_session(
        [SESSION_REQUEST_1, SESSION_REQUEST_2],
        provider="anthropic",
        model="claude-opus-4-8",
    )
    p = pricing.get("claude-opus-4-8")
    r_distil = cache_aware.simulate(traj, p, strategy="distil", caching=True)
    r_naive = cache_aware.simulate(traj, p, strategy="naive", caching=True)
    # distil preserves the prefix → lower or equal cost vs naive
    assert r_distil.total_dollars <= r_naive.total_dollars + 1e-10


# ---------------------------------------------------------------------------
# ingest_file
# ---------------------------------------------------------------------------


def test_ingest_file_single_json():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(SESSION_REQUEST_1, f)
        fname = f.name

    traj = ingest_file(fname, provider="anthropic")
    assert len(traj.turns) == 1
    assert any(b.kind == Kind.SYSTEM for b in traj.turns[0].blocks)


def test_ingest_file_list_json():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump([SESSION_REQUEST_1, SESSION_REQUEST_2], f)
        fname = f.name

    traj = ingest_file(fname, provider="anthropic")
    assert len(traj.turns) == 2


def test_ingest_file_jsonl():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        f.write(json.dumps(SESSION_REQUEST_1) + "\n")
        f.write(json.dumps(SESSION_REQUEST_2) + "\n")
        fname = f.name

    traj = ingest_file(fname, provider="anthropic")
    assert len(traj.turns) == 2


def test_ingest_file_trajectory_id_from_stem():
    p = Path(tempfile.mktemp(suffix=".json"))
    p.write_text(json.dumps(SESSION_REQUEST_1))
    traj = ingest_file(str(p))
    assert traj.id == p.stem


def test_ingest_file_openai():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(OPENAI_BODY, f)
        fname = f.name

    traj = ingest_file(fname, provider="openai")
    assert len(traj.turns) == 1
    kinds = {b.kind for b in traj.turns[0].blocks}
    assert Kind.SYSTEM in kinds
    assert Kind.TOOLS in kinds


def test_ingest_file_unknown_provider():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(SESSION_REQUEST_1, f)
        fname = f.name

    with pytest.raises(ValueError, match="unknown provider"):
        ingest_file(fname, provider="cohere")
