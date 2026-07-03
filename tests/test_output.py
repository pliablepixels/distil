"""Output compression — shaping (gated) + lossless re-entry digest + A/B measurement."""

import json
from pathlib import Path

import pytest

from distil.output import (
    answer_fingerprint,
    digest_output_blocks,
    measure_output_savings,
    shape_request,
)
from distil.trajectory import Block, Kind, Stability

PAIRS_FILE = Path(__file__).resolve().parent.parent / "corpus" / "output_pairs.jsonl"


# --- generation-side shaping ------------------------------------------------
def test_shape_request_anthropic_uses_top_level_system():
    # The Anthropic Messages API 400s on role:"system" inside messages —
    # a Claude body must get the directive via the top-level system field.
    body = {"model": "claude-opus-4-8", "messages": [{"role": "user", "content": "hi"}]}
    out = shape_request(body, level="light", allow=True)
    assert "concise" in out["system"].lower()
    assert all(m["role"] != "system" for m in out["messages"])
    assert body["messages"] == [{"role": "user", "content": "hi"}]  # input not mutated


def test_shape_request_anthropic_appends_to_existing_system():
    body = {"model": "claude-opus-4-8", "system": "You are a bot.", "messages": []}
    out = shape_request(body, level="light", allow=True)
    assert out["system"].startswith("You are a bot.")
    assert "concise" in out["system"].lower()
    # list-form system prompts get a text block appended
    body2 = {"system": [{"type": "text", "text": "core"}], "messages": []}
    out2 = shape_request(body2, level="light", allow=True, shape="anthropic")
    assert out2["system"][0] == {"type": "text", "text": "core"}
    assert "concise" in out2["system"][1]["text"].lower()


def test_shape_request_openai_appends_system_message():
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    out = shape_request(body, level="light", allow=True, shape="openai")
    assert out["messages"][-1]["role"] == "system"
    assert "concise" in out["messages"][-1]["content"].lower()


def test_shape_request_noop_when_off_or_disallowed():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert shape_request(body, level="off", allow=True) is body
    assert shape_request(body, level="aggressive", allow=False) is body  # auth-gated


def test_shape_request_unknown_level_raises():
    with pytest.raises(ValueError):
        shape_request({"messages": []}, level="ludicrous", allow=True)


# --- lossless re-entry digest -----------------------------------------------
def test_digest_output_blocks_is_reversible():
    long_answer = "DECISION: ship it\n" + "\n".join(f"reasoning step {i}" for i in range(20))
    blocks = [Block("h0", Kind.HISTORY, long_answer, Stability.SETTLING)]
    out, restore = digest_output_blocks(blocks)
    assert len(out[0].text) < len(long_answer)  # compressed
    assert "DECISION: ship it" in out[0].text  # decision preserved
    assert long_answer in restore.values()  # original recoverable
    assert out[0].kind is Kind.HISTORY  # kind preserved


def test_digest_output_blocks_skips_short():
    blocks = [Block("h0", Kind.HISTORY, "short answer", Stability.SETTLING)]
    out, restore = digest_output_blocks(blocks)
    assert out[0].text == "short answer" and not restore


# --- A/B measurement (the evaluation) ---------------------------------------
def test_answer_fingerprint_extracts_decision():
    a = "lots of preamble. DECISION: roll back to rev 6. trailing recap."
    b = "DECISION: roll back to rev 6."
    assert answer_fingerprint(a) == answer_fingerprint(b)


def test_measure_output_savings_on_real_fixture():
    pairs = [
        (d["baseline"], d["shaped"])
        for d in (json.loads(line) for line in PAIRS_FILE.read_text().splitlines() if line.strip())
    ]
    report = measure_output_savings(pairs)
    assert report.mean_reduction > 0.4  # verbose -> concise is a big cut
    assert report.answer_match_rate == 1.0  # every answer preserved (the gate)
    assert report.ci_low <= report.mean_reduction <= report.ci_high


def test_measure_flags_dropped_answer():
    # a "compression" that drops the decision must NOT count as a clean saving
    pairs = [("DECISION: do X. blah blah blah blah", "here is a terse but wrong summary")]
    report = measure_output_savings(pairs)
    assert report.answer_match_rate < 1.0
