"""Shadow-mode live decision-equivalence — sampling, decision extraction, ledger."""

from __future__ import annotations

from distil.shadow import (
    ShadowLedger,
    ShadowSampler,
    compare_decisions,
    decision_signature,
    decision_signature_from_body,
)


# --- streaming (SSE / chunk-array) decision extraction --------------------- #
# The core property: a STREAMED response must yield the SAME signature as the
# equivalent non-streamed JSON, so shadow-mode works on Claude Code / Codex /
# Gemini sessions (which all stream).

_ANTHROPIC_SSE = (
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":0,'
    '"content_block":{"type":"tool_use","id":"t1","name":"get_weather","input":{}}}\n\n'
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"input_json_delta","partial_json":"{\\"city\\":"}}\n\n'
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"input_json_delta","partial_json":"\\"SF\\"}"}}\n\n'
    'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)
_ANTHROPIC_JSON = {
    "content": [{"type": "tool_use", "name": "get_weather", "input": {"city": "SF"}}]
}

_OPENAI_SSE = (
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1",'
    '"function":{"name":"get_weather","arguments":""}}]}}]}\n\n'
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
    '"function":{"arguments":"{\\"city\\":"}}]}}]}\n\n'
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
    '"function":{"arguments":"\\"SF\\"}"}}]}}]}\n\n'
    "data: [DONE]\n\n"
)
_OPENAI_JSON = {
    "choices": [
        {
            "message": {
                "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city":"SF"}'}}]
            }
        }
    ]
}

_GEMINI_SSE = (
    'data: {"candidates":[{"content":{"parts":['
    '{"functionCall":{"name":"get_weather","args":{"city":"SF"}}}]}}]}\n\n'
)
_GEMINI_JSON = {
    "candidates": [
        {"content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"city": "SF"}}}]}}
    ]
}


def test_anthropic_stream_matches_json():
    sig = decision_signature_from_body(_ANTHROPIC_SSE)
    assert sig.startswith("tool:")
    assert sig == decision_signature(_ANTHROPIC_JSON)


def test_openai_stream_matches_json():
    sig = decision_signature_from_body(_OPENAI_SSE)
    assert sig.startswith("tool:")
    assert sig == decision_signature(_OPENAI_JSON)


def test_gemini_stream_matches_json():
    sig = decision_signature_from_body(_GEMINI_SSE)
    assert sig.startswith("tool:")
    assert sig == decision_signature(_GEMINI_JSON)


def test_gemini_chunk_array_form():
    # Gemini streamGenerateContent without alt=sse returns a JSON array of chunks.
    import json

    body = json.dumps([_GEMINI_JSON])
    assert decision_signature_from_body(body) == decision_signature(_GEMINI_JSON)


def test_stream_text_responses_are_text():
    anth = (
        "event: content_block_start\n"
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n'
    )
    oai = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
    gem = 'data: {"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}\n\n'
    assert decision_signature_from_body(anth) == "text"
    assert decision_signature_from_body(oai) == "text"
    assert decision_signature_from_body(gem) == "text"


def test_body_json_dict_and_bytes_and_empty():
    import json

    assert decision_signature_from_body(json.dumps(_ANTHROPIC_JSON)) == decision_signature(
        _ANTHROPIC_JSON
    )
    assert decision_signature_from_body(_GEMINI_SSE.encode()).startswith("tool:")  # bytes ok
    assert decision_signature_from_body("") == "none"
    assert decision_signature_from_body("not json, not sse") == "none"


def test_compressed_vs_uncompressed_stream_equivalence():
    # Same decision, one streamed one not -> equivalent (shadow records no change).
    assert decision_signature_from_body(_OPENAI_SSE) == decision_signature_from_body(
        __import__("json").dumps(_OPENAI_JSON)
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
