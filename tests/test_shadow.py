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


def test_sampler_is_probabilistic_and_seedable():
    import random

    # Same seed → identical draw sequence, so shadow tests stay deterministic.
    a = ShadowSampler(0.2, rng=random.Random(42))
    b = ShadowSampler(0.2, rng=random.Random(42))
    assert [a.should_sample() for _ in range(50)] == [b.should_sample() for _ in range(50)]
    # ~10 expected at rate 0.2 over 50 draws; wide band, just not degenerate.
    c = ShadowSampler(0.2, rng=random.Random(7))
    assert 2 <= sum(c.should_sample() for _ in range(50)) <= 18
    assert ShadowSampler(0.0).should_sample() is False  # disabled
    assert all(ShadowSampler(1.0).should_sample() for _ in range(10))  # rate 1 always samples


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
    return {
        "content": [
            {"type": "tool_use", "name": "Edit", "input": {"path": "x.py", "new_str": new_str}}
        ]
    }


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
    s1 = decision_signature(
        {"content": [{"type": "tool_use", "name": "weather", "input": {"city": "SF"}}]}
    )
    s2 = decision_signature(
        {"content": [{"type": "tool_use", "name": "weather", "input": {"city": "NYC"}}]}
    )
    assert s1 != s2 and s1.startswith("tool:")


def test_edit_equivalence_openai_arguments():
    import json as _j

    a = decision_signature(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "Edit",
                                    "arguments": _j.dumps({"new_str": "def f():\n    return 1"}),
                                }
                            }
                        ]
                    }
                }
            ]
        }
    )
    b = decision_signature(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "Edit",
                                    "arguments": _j.dumps(
                                        {"new_str": "def f():\n\n    return 1  # x"}
                                    ),
                                }
                            }
                        ]
                    }
                }
            ]
        }
    )
    assert a == b


def test_edit_equivalence_holds_across_streaming():
    # Streamed and non-streamed forms of the same edit must still match (shared sig path).
    nonstream = decision_signature(_anthropic_edit("def f():\n    return 1"))
    sse = (
        "event: content_block_start\n"
        'data: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"tool_use","id":"t","name":"Edit","input":{}}}\n\n'
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"input_json_delta","partial_json":"{\\"path\\": \\"x.py\\", \\"new_str\\": \\"def f():\\\\n    return 1\\"}"}}\n\n'
    )
    assert decision_signature_from_body(sse) == nonstream


def test_shadow_discriminates_changed_decision_cross_provider():
    """Shadow must flag a *changed* next action (not just confirm matches) for
    every provider's response shape — the basis of cross-provider validation."""
    # OpenAI — same tool, different target argument → changed
    oa_a = {
        "choices": [
            {
                "message": {
                    "tool_calls": [{"function": {"name": "edit", "arguments": '{"path":"a.py"}'}}]
                }
            }
        ]
    }
    oa_b = {
        "choices": [
            {
                "message": {
                    "tool_calls": [{"function": {"name": "edit", "arguments": '{"path":"b.py"}'}}]
                }
            }
        ]
    }
    assert compare_decisions(oa_a, oa_a) is True
    assert compare_decisions(oa_a, oa_b) is False

    # Gemini — different function name → changed
    gm_a = {
        "candidates": [
            {"content": {"parts": [{"functionCall": {"name": "read", "args": {"f": "x"}}}]}}
        ]
    }
    gm_b = {
        "candidates": [
            {"content": {"parts": [{"functionCall": {"name": "write", "args": {"f": "x"}}}]}}
        ]
    }
    assert compare_decisions(gm_a, gm_a) is True
    assert compare_decisions(gm_a, gm_b) is False

    # Anthropic — different tool input → changed
    an_a = {"content": [{"type": "tool_use", "name": "bash", "input": {"cmd": "ls"}}]}
    an_b = {"content": [{"type": "tool_use", "name": "bash", "input": {"cmd": "rm"}}]}
    assert compare_decisions(an_a, an_a) is True
    assert compare_decisions(an_a, an_b) is False


# --- A/A noise baseline (rc4): raw A/B disagreement conflates compression ---
# harm with sampling nondeterminism; the baseline is what makes it readable.


def test_ledger_aa_kind_counts_separately(tmp_path):
    p = tmp_path / "shadow.jsonl"
    led = ShadowLedger()
    led.record(True, path=p)  # default kind: ab
    led.record(False, kind="aa", path=p)
    led.record(True, kind="aa", path=p)
    assert (led.samples, led.changes) == (1, 0)  # ab meaning unchanged
    assert (led.aa_samples, led.aa_changes) == (2, 1)
    reloaded = ShadowLedger.load(p)
    assert (reloaded.samples, reloaded.aa_samples, reloaded.aa_changes) == (1, 2, 1)


def test_ledger_load_pre_rc4_rows_count_as_ab(tmp_path):
    p = tmp_path / "shadow.jsonl"
    p.write_text('{"equivalent": false, "ts": 1.0}\n', encoding="utf-8")  # no "kind"
    led = ShadowLedger.load(p)
    assert (led.samples, led.changes, led.aa_samples) == (1, 1, 0)


def test_aa_agreement_needs_ten_samples(tmp_path):
    led = ShadowLedger()
    p = tmp_path / "shadow.jsonl"
    for _ in range(9):
        led.record(True, kind="aa", path=p)
    assert led.aa_agreement() is None
    led.record(True, kind="aa", path=p)
    assert led.aa_agreement() == 1.0


def test_adjusted_rate_factors_out_model_nondeterminism(tmp_path):
    """47% raw agreement against a 52% self-agreement baseline ≈ compression
    adds ~10% — the exact confusion the raw number invites."""
    led = ShadowLedger()
    p = tmp_path / "shadow.jsonl"
    for i in range(100):
        led.record(i < 47, path=p)  # ab: 47% equivalent
    for i in range(100):
        led.record(i < 52, kind="aa", path=p)  # aa: model agrees with itself 52%
    assert abs(led.rate() - 0.53) < 1e-9
    assert abs(led.aa_agreement() - 0.52) < 1e-9
    assert abs(led.adjusted_rate() - (1 - 0.47 / 0.52)) < 1e-9
    # and a perfect baseline changes nothing
    led2 = ShadowLedger()
    for i in range(100):
        led2.record(i < 47, path=p)
    assert led2.aa_agreement() is None
    assert led2.adjusted_rate() == led2.rate()  # no baseline → raw


def test_record_persists_content_free_evidence(tmp_path):
    import json as _json

    p = tmp_path / "shadow.jsonl"
    ShadowLedger().record(
        False,
        kind="ab",
        evidence={"digest": "ab12", "sig_served": "tool:x1", "sig_replay": "tool:y2"},
        path=p,
    )
    rec = _json.loads(p.read_text().strip())
    assert rec["kind"] == "ab" and rec["digest"] == "ab12"
    assert rec["sig_served"] != rec["sig_replay"]  # the divergence is now diagnosable


def test_proxy_aa_replay_records_baseline_e2e(monkeypatch, tmp_path):
    """Force the A/A branch (random → 0): the proxy replays the SAME compressed
    request and books a kind="aa" self-agreement row with evidence fields."""
    import http.server
    import json as _json
    import os
    import sys
    import threading
    from pathlib import Path

    from distil import proxy as proxy_mod

    RESP = _json.dumps({"content": [{"type": "tool_use", "name": "t", "input": {"x": 1}}]}).encode()

    class Up(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(RESP)))
            self.end_headers()
            self.wfile.write(RESP)

        def log_message(self, *a):  # noqa: ANN002
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Up)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr("random.random", lambda: 0.0)  # sampler fires AND aa branch taken

    child = (
        "import os, json, urllib.request\n"
        "base = os.environ['ANTHROPIC_BASE_URL']\n"
        "body = json.dumps({'model': 'claude-opus-4-8', 'messages': "
        "[{'role': 'user', 'content': 'go'}]}).encode()\n"
        "req = urllib.request.Request(base + '/v1/messages', data=body,"
        " headers={'Content-Type': 'application/json'}, method='POST')\n"
        "urllib.request.urlopen(req, timeout=5)\n"
    )
    try:
        code = proxy_mod.wrap_run(
            [sys.executable, "-c", child],
            upstream=f"http://127.0.0.1:{srv.server_address[1]}",
            record=False,
            shadow_rate=1.0,
        )
    finally:
        srv.shutdown()
    assert code == 0
    sj = Path(os.environ["DISTIL_HOME"]) / "shadow.jsonl"
    rows = [_json.loads(line) for line in sj.read_text().splitlines() if line.strip()]
    aa = [r for r in rows if r.get("kind") == "aa"]
    assert aa, rows  # the A/A branch actually ran and recorded
    assert aa[0]["equivalent"] is True  # fixed upstream → identical decision
    assert aa[0]["digest"] and aa[0]["sig_served"] == aa[0]["sig_replay"]


def _hammer_append(args):
    """Worker for the cross-process append test (must be module-level to pickle)."""
    path_str, worker_id, n_rows = args
    from pathlib import Path

    from distil.shadow import ShadowLedger

    led = ShadowLedger()
    fat = f"sig-{worker_id}-" + "x" * 4096  # well past PIPE_BUF atomicity
    for i in range(n_rows):
        led.record(
            i % 2 == 0,
            kind="ab",
            evidence={"digest": f"d{worker_id}-{i}", "sig_served": fat, "sig_replay": fat},
            path=Path(path_str),
        )
    return worker_id


def test_concurrent_cross_process_appends_stay_intact(tmp_path):
    """Multiple wrap sessions append to one shadow.jsonl. Without the flock a
    >PIPE_BUF line can interleave with another writer's and tear both rows —
    every line must parse and every row must survive."""
    import json as _json
    import sys
    from concurrent.futures import ProcessPoolExecutor

    import pytest

    if sys.platform == "win32":
        pytest.skip("fcntl advisory locking is POSIX-only; unlocked on Windows by design")

    p = tmp_path / "shadow.jsonl"
    workers, rows_each = 4, 25
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_hammer_append, [(str(p), w, rows_each) for w in range(workers)]))

    lines = [line for line in p.read_text().splitlines() if line.strip()]
    assert len(lines) == workers * rows_each  # nothing lost
    for line in lines:
        _json.loads(line)  # nothing torn
