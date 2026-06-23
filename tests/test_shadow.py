"""Shadow-mode live decision-equivalence — sampling, decision extraction, ledger."""

from __future__ import annotations

from distil.shadow import (
    ShadowLedger,
    ShadowSampler,
    compare_decisions,
    decision_signature,
)


def test_decision_signature_anthropic_tool_use():
    a = {"content": [{"type": "tool_use", "name": "rotate_logs", "input": {"node": "N7"}}]}
    b = {"content": [{"type": "tool_use", "name": "rotate_logs", "input": {"node": "N7"}}]}
    c = {"content": [{"type": "tool_use", "name": "rotate_logs", "input": {"node": "N8"}}]}
    assert decision_signature(a) == decision_signature(b)  # same action+target
    assert decision_signature(a) != decision_signature(c)  # different target
    assert decision_signature(a).startswith("tool:")


def test_decision_signature_openai_tool_call():
    a = {
        "choices": [
            {"message": {"tool_calls": [{"function": {"name": "f", "arguments": '{"x":1}'}}]}}
        ]
    }
    b = {
        "choices": [
            {"message": {"tool_calls": [{"function": {"name": "f", "arguments": '{"x":1}'}}]}}
        ]
    }
    assert decision_signature(a) == decision_signature(b)
    assert decision_signature(a).startswith("tool:")


def test_decision_signature_text_and_none():
    assert decision_signature({"content": [{"type": "text", "text": "hi"}]}) == "text"
    assert decision_signature({"stop_reason": "end_turn"}) == "none"
    assert decision_signature("not a dict") == "none"


def test_compare_decisions():
    tool_a = {"content": [{"type": "tool_use", "name": "x", "input": {"a": 1}}]}
    tool_b = {"content": [{"type": "tool_use", "name": "x", "input": {"a": 2}}]}
    text = {"content": [{"type": "text", "text": "answer"}]}
    assert compare_decisions(tool_a, tool_a) is True
    assert compare_decisions(tool_a, tool_b) is False
    assert compare_decisions(tool_a, text) is False  # compression suppressed the tool call
    assert compare_decisions(text, text) is True


def test_sampler_is_deterministic_one_in_n():
    s = ShadowSampler(0.2)  # 1 in 5
    hits = [s.should_sample() for _ in range(20)]
    assert sum(hits) == 4  # exactly 20/5
    assert ShadowSampler(0.0).should_sample() is False  # disabled


def test_ledger_records_and_rates(tmp_path):
    led = ShadowLedger()
    p = tmp_path / "shadow.jsonl"
    for eq in [True, True, True, False, True]:
        led.record(eq, path=p)
    assert led.samples == 5
    assert led.changes == 1
    assert abs(led.rate() - 0.2) < 1e-9  # 1/5 changed
    # persisted content-free (no prompt/response text)
    text = p.read_text()
    assert "equivalent" in text and "content" not in text


def test_ledger_load_roundtrip(tmp_path):
    p = tmp_path / "shadow.jsonl"
    led = ShadowLedger()
    for eq in [True, False, True]:
        led.record(eq, path=p)
    reloaded = ShadowLedger.load(p)
    assert reloaded.samples == 3
    assert reloaded.changes == 1
