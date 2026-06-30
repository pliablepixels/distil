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


# --- edit-equivalence: AST-normalized code in decision signatures ---------- #


def _anthropic_edit(new_str: str) -> dict:
    return {"content": [{"type": "tool_use", "name": "Edit",
                         "input": {"path": "x.py", "new_str": new_str}}]}


def test_edit_equivalence_ignores_formatting_and_comments():
    a = decision_signature(_anthropic_edit("def f():\n    return 1"))
    b = decision_signature(_anthropic_edit("def f():\n    # a comment\n    return 1"))
    c = decision_signature(_anthropic_edit("def f():\n        return 1"))  # reindented body
    assert a == b == c  # same code, different formatting/comments -> same decision


def test_edit_equivalence_detects_real_logic_change():
    a = decision_signature(_anthropic_edit("def f():\n    return 1"))
    d = decision_signature(_anthropic_edit("def f():\n    return 2"))  # different value
    assert a != d  # a genuine logic change is still a decision change


def test_non_code_inputs_still_distinguished():
    s1 = decision_signature({"content": [{"type": "tool_use", "name": "weather", "input": {"city": "SF"}}]})
    s2 = decision_signature({"content": [{"type": "tool_use", "name": "weather", "input": {"city": "NYC"}}]})
    assert s1 != s2 and s1.startswith("tool:")


def test_edit_equivalence_openai_arguments():
    import json as _j
    a = decision_signature({"choices": [{"message": {"tool_calls": [
        {"function": {"name": "Edit", "arguments": _j.dumps({"new_str": "def f():\n    return 1"})}}]}}]})
    b = decision_signature({"choices": [{"message": {"tool_calls": [
        {"function": {"name": "Edit", "arguments": _j.dumps({"new_str": "def f():\n\n    return 1  # x"})}}]}}]})
    assert a == b


def test_edit_equivalence_holds_across_streaming():
    # Streamed and non-streamed forms of the same edit must still match (shared sig path).
    nonstream = decision_signature(_anthropic_edit("def f():\n    return 1"))
    sse = (
        'event: content_block_start\n'
        'data: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"tool_use","id":"t","name":"Edit","input":{}}}\n\n'
        'event: content_block_delta\n'
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"input_json_delta","partial_json":"{\\"path\\": \\"x.py\\", \\"new_str\\": \\"def f():\\\\n    return 1\\"}"}}\n\n'
    )
    assert decision_signature_from_body(sse) == nonstream


def test_shadow_discriminates_changed_decision_cross_provider():
    """Shadow must flag a *changed* next action (not just confirm matches) for
    every provider's response shape — the basis of cross-provider validation."""
    # OpenAI — same tool, different target argument → changed
    oa_a = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "edit", "arguments": '{"path":"a.py"}'}}]}}]}
    oa_b = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "edit", "arguments": '{"path":"b.py"}'}}]}}]}
    assert compare_decisions(oa_a, oa_a) is True
    assert compare_decisions(oa_a, oa_b) is False

    # Gemini — different function name → changed
    gm_a = {"candidates": [{"content": {"parts": [
        {"functionCall": {"name": "read", "args": {"f": "x"}}}]}}]}
    gm_b = {"candidates": [{"content": {"parts": [
        {"functionCall": {"name": "write", "args": {"f": "x"}}}]}}]}
    assert compare_decisions(gm_a, gm_a) is True
    assert compare_decisions(gm_a, gm_b) is False

    # Anthropic — different tool input → changed
    an_a = {"content": [{"type": "tool_use", "name": "bash", "input": {"cmd": "ls"}}]}
    an_b = {"content": [{"type": "tool_use", "name": "bash", "input": {"cmd": "rm"}}]}
    assert compare_decisions(an_a, an_a) is True
    assert compare_decisions(an_a, an_b) is False
