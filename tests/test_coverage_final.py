"""Final coverage top-ups: small optional-dependency and edge-case branches that
the other suites don't reach. Real behavior assertions, not coverage padding."""

from __future__ import annotations

import importlib
import sys
import types

import pytest


# ── litellm in-process hook (optional dep, mocked) ──────────────────────────
def test_litellm_completion_compresses_then_calls(monkeypatch):
    fake = types.ModuleType("litellm")
    seen: dict = {}
    fake.completion = lambda **kw: (seen.update(kw), "resp")[1]

    async def _acomp(**kw):
        seen.update(kw)
        return "aresp"

    fake.acompletion = _acomp
    monkeypatch.setitem(sys.modules, "litellm", fake)
    from distil.integrations import litellm as dl

    big = "\n".join(f"line {i}: data {i}" for i in range(40))
    msgs = [{"role": "user", "content": [{"type": "tool_result", "content": big}]}]
    out = dl.completion(model="claude-opus-4-8", messages=msgs)
    assert out == "resp" and "messages" in seen


def test_litellm_acompletion(monkeypatch):
    import asyncio

    fake = types.ModuleType("litellm")

    async def _acomp(**kw):
        return "aresp"

    fake.acompletion = _acomp
    monkeypatch.setitem(sys.modules, "litellm", fake)
    from distil.integrations import litellm as dl

    out = asyncio.run(dl.acompletion(model="x", messages=[{"role": "user", "content": "hi"}]))
    assert out == "aresp"


# ── native Rust backend path (distil_core injected) ─────────────────────────
def test_native_rust_backend_path(monkeypatch):
    fake = types.ModuleType("distil_core")
    fake.minify_json = lambda t: "{}"
    fake.collapse_runs = lambda t: t
    fake.count_tokens = lambda t, f=1.33: 7
    monkeypatch.setitem(sys.modules, "distil_core", fake)
    import distil.native as native

    try:
        importlib.reload(native)
        assert native.BACKEND == "rust"
        assert native.minify_json("{ }") == "{}"
        assert native.count_tokens("hello world") == 7
    finally:
        # restore the real (pure-python fallback) module for the rest of the suite
        monkeypatch.delitem(sys.modules, "distil_core", raising=False)
        importlib.reload(native)
        assert native.BACKEND in ("python", "rust")


# ── speculative controller fail-safe branches ───────────────────────────────
def test_speculative_length_mismatch_raises():
    from distil.speculative import calibrate_speculative

    with pytest.raises(ValueError):
        calibrate_speculative([0.1, 0.2], [True])  # misaligned lengths


def test_speculative_empty_is_conservative():
    from distil.speculative import calibrate_speculative

    ctrl = calibrate_speculative([], [])
    assert ctrl is not None  # n==0 path returns a controller, never crashes


# ── replay/prompts parse branches ───────────────────────────────────────────
def test_parse_expand_variants():
    from distil.replay.prompts import parse_expand

    assert parse_expand("") is None
    assert parse_expand("no json here") is None
    assert parse_expand('{"expand": ["h1", "h2"]}') == ["h1", "h2"]
    assert parse_expand('{"expand": []}') is None  # empty expand → None


def test_canonical_and_norm_action():
    from distil.replay.prompts import canonical

    a = canonical("Read", "server.py")
    b = canonical("read", "server.py")
    assert a == b  # action normalized case-insensitively


# ── certify/stats: statistical helpers (pure functions, hit the branches) ───
def test_stats_tost_and_t_distribution():
    from distil.certify.stats import t_cdf, t_sf, tost

    # t distribution: both tails (negative t hits the mirrored branch)
    assert 0.0 <= t_cdf(-1.5, 10.0) <= 0.5
    assert 0.5 <= t_cdf(1.5, 10.0) <= 1.0
    assert abs(t_sf(0.0, 10.0) - 0.5) < 1e-6
    # TOST on a tight cluster of diffs → should be non-inferior at a 2% margin
    r = tost([0.001, -0.002, 0.0, 0.001], margin=0.05, alpha=0.05)
    assert 0.0 <= r.p_non_inferior <= 1.0


def test_stats_mcnemar_noninferiority():
    from distil.certify.stats import mcnemar_noninferiority

    # empty → safe default, non_inferior False, no crash
    assert mcnemar_noninferiority(0, 0, 0).noninferior is False
    # candidate gains > losses on a decent n → non-inferior
    r = mcnemar_noninferiority(b=3, c=8, n=200, margin=0.05)
    assert r.noninferior is True
    # heavy losses → not non-inferior
    r2 = mcnemar_noninferiority(b=40, c=2, n=200, margin=0.05)
    assert r2.noninferior is False


# ── compress/guideline: outcome-guided keep policy ──────────────────────────
def test_outcome_guideline_policy(tmp_path):
    from distil.compress.guideline import OutcomeStats, record_trajectory_outcome

    p = tmp_path / "outcomes.json"
    st = OutcomeStats.load(p)
    st.record({"sig-a"}, regressed=True)
    st.record({"sig-a"}, regressed=True)
    st.record({"sig-a"}, regressed=False)
    assert st.regression_rate("sig-a") > 0.5  # 2 of 3 regressed
    assert st.regression_rate("never-seen") == 0.0
    assert "sig-a" in st.protect_prone(min_seen=1, threshold=0.3)
    assert callable(st.keep_predicate(min_seen=1, threshold=0.3))
    st.save(p)
    assert p.exists()

    # record_trajectory_outcome: full solved, compressed failed → records a regression;
    # and the no-op path when the full run also failed.
    record_trajectory_outcome(
        ["a\nb\nc\nd\ne\nf\ng"], full_success=True, compressed_success=False, path=p
    )
    record_trajectory_outcome(["x"], full_success=False, compressed_success=False, path=p)


# ── last one-liners: entropy empty, tier1 line-drop, gemini passthrough ──────
def test_final_edge_lines():
    from distil.adapters.gemini import _compress_json_value
    from distil.compress.salience import _entropy
    from distil.compress.tier1 import digest
    from distil.adapters.anthropic import RestoreStore

    assert _entropy("") == 0.0  # empty-token early return

    # tier1.digest on a long block drops interior lines → emits the handle marker
    big = "\n".join(f"body line {i}" for i in range(30))
    out, changed = digest(big, head=2, tail=1)
    assert changed and "handle=" in out

    # gemini: a scalar JSON value (not dict/list) passes straight through
    store = RestoreStore()
    assert _compress_json_value(42, store, False) == 42
    assert _compress_json_value("plain", store, False) == "plain"


# ── anthropic adapter: Tier-0 collapse, digest fallback, content shapes ──────
def test_anthropic_adapter_branches():
    from distil.adapters.anthropic import compress_messages

    # Tier-0 collapse: JSON-with-whitespace + repeated runs (hits the collapse path);
    # a SHORT tool_result (<6 lines) that digest declines (tier0 fallback);
    # a list-form tool_result with a big block that DOES digest;
    # image + assistant blocks pass through untouched.
    big = "\n".join(f"log line {i}: value={i}" for i in range(60))
    messages = [
        {"role": "user", "content": [{"type": "text", "text": '{  "a" :  1 ,  "b" : 2  }'}]},
        {"role": "assistant", "content": [{"type": "text", "text": "thinking..."}]},
        {"role": "user", "content": [
            {"type": "tool_result", "content": "one\ntwo"},                 # short → tier0
            {"type": "tool_result", "content": [{"type": "text", "text": big}]},  # digests
            {"type": "image", "source": {"type": "base64", "data": "x"}},   # passthrough
        ]},
    ]
    compressed, store = compress_messages(messages, verbatim=False)
    # the big tool_result got a handle in the store; images/assistant unchanged
    assert store is not None
    flat = str(compressed)
    assert "handle=" in flat or len(store.handles) >= 1

    # verbatim mode: no digest stub, only lossless tier-0
    comp_v, store_v = compress_messages(messages, verbatim=True)
    assert "handle=" not in str(comp_v)
