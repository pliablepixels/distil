"""Gemini generateContent adapter — path detection, compression, recovery, tokens.

Locks in Phase 2: distil compresses Google's generateContent shape (contents/parts/
functionResponse) reversibly, the same way it compresses the Messages API.
"""

from __future__ import annotations

from distil.adapters.gemini import (
    compress_generate_request,
    count_tokens,
    is_gemini_path,
)
from distil.shadow import decision_signature

# A tool output big enough to trigger the Tier-1 reversible digest (>= 6 lines).
_BIG_OUTPUT = "\n".join(f"row {i}: value_{i} status=ok detail=lorem ipsum dolor" for i in range(40))


def _req() -> dict:
    return {
        "contents": [
            {"role": "user", "parts": [{"text": "List the failing rows."}]},
            {
                "role": "model",
                "parts": [{"functionCall": {"name": "query_db", "args": {"q": "SELECT *"}}}],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "query_db",
                            "response": {"output": _BIG_OUTPUT},
                        }
                    }
                ],
            },
        ],
    }


# --- path detection -------------------------------------------------------- #


def test_is_gemini_path_matches_generate_and_stream():
    assert is_gemini_path("/v1beta/models/gemini-1.5-pro:generateContent")
    assert is_gemini_path("/v1beta/models/gemini-2.0-flash:streamGenerateContent")
    assert is_gemini_path("/v1/models/gemini-1.5-pro:generateContent?key=abc123")
    assert is_gemini_path("/v1beta/models/gemini-1.5-pro:generateContent?alt=sse")


def test_is_gemini_path_rejects_other_paths():
    assert not is_gemini_path("/v1/messages")
    assert not is_gemini_path("/v1/chat/completions")
    assert not is_gemini_path("/v1beta/models")
    assert not is_gemini_path("/v1beta/models/gemini-1.5-pro:countTokens")


# --- compression + recovery ------------------------------------------------ #


def test_function_response_is_digested_and_recoverable():
    body = _req()
    new_body, store = compress_generate_request(body)

    # The big tool output was replaced by a shorter digest...
    fr = new_body["contents"][2]["parts"][0]["functionResponse"]["response"]["output"]
    assert fr != _BIG_OUTPUT
    assert len(fr) < len(_BIG_OUTPUT)

    # ...and is byte-exactly recoverable from the store via its handle.
    assert store.handles, "expected at least one digest handle"
    recovered = {store.expand(h) for h in store.handles}
    assert _BIG_OUTPUT in recovered


def test_input_is_not_mutated():
    body = _req()
    before = body["contents"][2]["parts"][0]["functionResponse"]["response"]["output"]
    compress_generate_request(body)
    assert body["contents"][2]["parts"][0]["functionResponse"]["response"]["output"] == before


def test_model_text_is_never_rewritten():
    body = {
        "contents": [
            {"role": "model", "parts": [{"text": '{"a":  1,   "b": 2}'}]},
        ]
    }
    new_body, _ = compress_generate_request(body)
    # Model-authored text passes through byte-exact even though it is minifiable JSON.
    assert new_body["contents"][0]["parts"][0]["text"] == '{"a":  1,   "b": 2}'


def test_function_call_passed_through():
    body = _req()
    new_body, _ = compress_generate_request(body)
    assert new_body["contents"][1]["parts"][0]["functionCall"] == {
        "name": "query_db",
        "args": {"q": "SELECT *"},
    }


def test_no_op_returns_same_body_object():
    body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    new_body, store = compress_generate_request(body)
    assert new_body is body  # nothing compressible -> identity, no copy
    assert not store.handles


# --- token accounting ------------------------------------------------------ #


def test_count_tokens_drops_after_compression():
    body = _req()
    before = count_tokens(body)
    new_body, _ = compress_generate_request(body)
    after = count_tokens(new_body)
    assert before > 0
    assert after < before


# --- robustness (malformed but valid JSON must never crash) ---------------- #


def test_malformed_shapes_do_not_crash():
    for body in [
        {"contents": "not a list"},
        {"contents": [None, 1, "x"]},
        {"contents": [{"role": "user"}]},  # no parts
        {"contents": [{"role": "user", "parts": "nope"}]},
        {"contents": [{"role": "user", "parts": [None, {"text": 5}, {}]}]},
        {"contents": [{"role": "user", "parts": [{"functionResponse": {}}]}]},
        {},
    ]:
        new_body, store = compress_generate_request(body)
        assert isinstance(new_body, dict)
        assert count_tokens(body) >= 0


# --- shadow decision extraction for Gemini --------------------------------- #


def test_decision_signature_gemini_function_call():
    resp = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "f", "args": {"x": 1}}}],
                }
            }
        ]
    }
    sig = decision_signature(resp)
    assert sig.startswith("tool:")
    # Same call -> same signature; different args -> different signature.
    assert sig == decision_signature(resp)
    resp2 = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "f", "args": {"x": 2}}}],
                }
            }
        ]
    }
    assert decision_signature(resp2) != sig


def test_decision_signature_gemini_text_and_none():
    text_resp = {"candidates": [{"content": {"role": "model", "parts": [{"text": "done"}]}}]}
    assert decision_signature(text_resp) == "text"
    assert decision_signature({"candidates": []}) == "none"
    assert decision_signature({}) == "none"
