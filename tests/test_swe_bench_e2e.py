"""Tests for the SWE-bench Verified end-to-end eval (Phase 5 / E7).

Covered:
* deterministic seed-1729 instance sampling (order-independent, reproducible);
* the compression layer for conditions B (distil trunc@500) and C (llmlingua-2):
  the block selector compresses only agent-read context (user/tool text/tool_result
  >= MIN_CHARS) and never system or assistant reasoning, and trunc@500 matches
  distil's ``conformal._truncate_level(500)`` byte-for-byte;
* Wilson 95% CI used for reported pass@1.

These run with no network and no Docker (the heavy harness/agent paths are exercised
by the live eval, not unit tests).
"""

from __future__ import annotations

from benchmarks.swe_bench_e2e import sample as smp
from benchmarks.swe_bench_e2e.compress_proxy import (
    MIN_CHARS,
    CompressStats,
    compress_body,
    trunc_500,
)
from benchmarks.swe_bench_e2e.stats import wilson_ci


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _fake_rows(k: int) -> list[dict]:
    # ids deliberately unsorted to prove the sampler sorts before drawing.
    return [{"instance_id": f"repo__proj-{i:04d}"} for i in reversed(range(k))]


def test_sample_is_deterministic_and_order_independent():
    rows = _fake_rows(500)
    a = smp.sample_instance_ids(rows, 50, seed=1729)
    b = smp.sample_instance_ids(list(reversed(rows)), 50, seed=1729)
    assert a == b  # shuffling the input pool must not change the draw
    assert len(a) == len(set(a)) == 50


def test_sample_changes_with_seed():
    rows = _fake_rows(500)
    assert smp.sample_instance_ids(rows, 50, seed=1729) != smp.sample_instance_ids(rows, 50, seed=1)


# --------------------------------------------------------------------------- #
# trunc@500 == distil's certifying operating point
# --------------------------------------------------------------------------- #
def test_trunc500_matches_conformal_truncate_level():
    from distil.conformal import _truncate_level
    from distil.trajectory import Block, Kind, Stability

    text = "x" * 4000
    strat = _truncate_level(500)
    blk = Block(id="b0", kind=Kind.TOOL_OUTPUT, text=text, stability=Stability.VOLATILE)
    (out_block,) = strat([blk], 0)
    assert trunc_500(text) == out_block.text == text[:500]
    assert len(trunc_500(text)) == 500


def test_trunc500_noop_below_limit():
    assert trunc_500("short") == "short"


# --------------------------------------------------------------------------- #
# Block selection: only agent-read context, never reasoning/instructions
# --------------------------------------------------------------------------- #
def _big(tag: str) -> str:
    return f"{tag}:" + "A" * (MIN_CHARS + 100)


def test_compresses_user_context_not_system_or_assistant():
    big_user = _big("file")
    big_assistant = _big("reasoning")
    body = {
        "model": "claude-sonnet-4-6",
        "system": _big("instructions"),  # system must be untouched
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": big_user}]},
            {"role": "assistant", "content": [{"type": "text", "text": big_assistant}]},
        ],
    }
    stats = CompressStats()
    out = compress_body(body, trunc_500, stats)
    # user file content compressed to 500 chars
    assert out["messages"][0]["content"][0]["text"] == big_user[:500]
    # assistant reasoning left intact
    assert out["messages"][1]["content"][0]["text"] == big_assistant
    # system string left intact (compressor only touches the messages array)
    assert out["system"] == body["system"]
    assert stats.blocks_compressed == 1


def test_small_blocks_untouched():
    body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "tiny instruction"}]}]
    }
    stats = CompressStats()
    out = compress_body(body, trunc_500, stats)
    assert out["messages"][0]["content"][0]["text"] == "tiny instruction"
    assert stats.blocks_compressed == 0


def test_tool_result_string_and_list_compressed():
    payload = _big("cmd-output")
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": payload},
                    {
                        "type": "tool_result",
                        "tool_use_id": "t2",
                        "content": [{"type": "text", "text": payload}],
                    },
                ],
            },
        ],
    }
    stats = CompressStats()
    out = compress_body(body, trunc_500, stats)
    c = out["messages"][0]["content"]
    assert c[0]["content"] == payload[:500]
    assert c[1]["content"][0]["text"] == payload[:500]
    assert stats.blocks_compressed == 2


def test_full_condition_is_passthrough_but_counts():
    big = _big("file")
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": big}]}]}
    stats = CompressStats()
    out = compress_body(body, None, stats)
    assert out["messages"][0]["content"][0]["text"] == big  # unchanged
    assert stats.blocks_seen == 1
    assert stats.blocks_compressed == 0
    assert stats.tokens_before == stats.tokens_after  # no compression accounted


def test_problem_statement_is_protected_from_compression():
    # The problem statement is the task, not file content — it must pass through verbatim
    # even though it is a large user block, while a separate file-content block compresses.
    problem = "ModelBackend.authenticate() should not query the DB when username is None. " * 20
    file_block = _big("source-file")
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": problem + " extra wrapping"}],
            },
            {"role": "user", "content": [{"type": "text", "text": file_block}]},
        ]
    }
    stats = CompressStats()
    out = compress_body(body, trunc_500, stats, protect=problem)
    # problem statement untouched (protected), file content truncated to 500 chars
    assert out["messages"][0]["content"][0]["text"] == problem + " extra wrapping"
    assert out["messages"][1]["content"][0]["text"] == file_block[:500]
    assert stats.blocks_protected == 1
    assert stats.blocks_compressed == 1


def test_compress_body_does_not_mutate_input():
    big = _big("file")
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": big}]}]}
    compress_body(body, trunc_500, CompressStats())
    assert body["messages"][0]["content"][0]["text"] == big  # original intact


# --------------------------------------------------------------------------- #
# Wilson CI
# --------------------------------------------------------------------------- #
def test_wilson_ci_bounds():
    import pytest

    lo, hi = wilson_ci(10, 50)
    assert 0.0 <= lo < 0.20 < hi <= 1.0
    # all-success and all-fail stay in [0,1] (upper at phat=1 is 1.0 up to float error)
    assert wilson_ci(0, 30)[0] == 0.0
    assert wilson_ci(30, 30)[1] == pytest.approx(1.0)


def test_wilson_ci_zero_n():
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_noninferiority_paired_verdict_flips_at_margin():
    from benchmarks.swe_bench_e2e.stats import noninferiority_paired

    # E8 gated-vs-full discordant counts: 42 full-only losses, 30 gated-only gains, n=500.
    r = noninferiority_paired(b=42, c=30, n=500, margin=0.06)
    assert r["delta"] < 0  # candidate (gated) point estimate slightly below reference
    assert r["ci95_low"] > -0.06  # lower bound clears a 6pp margin -> non-inferior
    assert r["noninferior"] is True
    # A stricter 5pp margin is not cleared (CI lower bound ~ -5.7pp).
    assert noninferiority_paired(b=42, c=30, n=500, margin=0.05)["noninferior"] is False
    # A clearly-inferior candidate (ungated reversible: 80 losses, 28 gains) fails even at 10pp.
    assert noninferiority_paired(b=80, c=28, n=500, margin=0.10)["noninferior"] is False


def test_noninferiority_paired_zero_n():
    from benchmarks.swe_bench_e2e.stats import noninferiority_paired

    assert noninferiority_paired(0, 0, 0, 0.05)["noninferior"] is False


def test_trajectory_bound_composition_and_k():
    import pytest

    from benchmarks.trajectory_bound import analyze

    r = analyze()  # E8 gated vs full, committed data
    # The naive per-turn->trajectory composition must be vacuous at agentic horizons,
    # while the observed divergence is far smaller — the gap the analysis quantifies.
    assert r["naive_bound"] > 0.5
    assert r["observed_divergence"] < 0.25
    assert r["naive_bound"] > r["observed_divergence"]
    # Effective consequential turns: a small handful out of ~27, by both estimators.
    assert 1.0 < r["k_consequential_linear"] < 4.0
    assert 1.0 < r["k_consequential_exact"] < 4.0
    # Linear bound reconstructs the observed divergence: d <= k*alpha (tight by fit).
    assert r["k_consequential_linear"] * r["alpha"] == pytest.approx(
        r["observed_divergence"], abs=1e-9
    )


# --- reversible tier: digest + relevance gate (distil_expand / distil_gated) -------- #


def test_digest_block_is_reversible_and_content_addressed():
    import hashlib

    from benchmarks.swe_bench_e2e.compress_proxy import MIN_CHARS, digest_block

    restore: dict[str, str] = {}
    big = "def f():\n" + "x" * 2000
    out = digest_block(big, restore)
    h = hashlib.sha256(big.encode()).hexdigest()[:8]
    assert "distil-digest" in out and h in out and len(out) < len(big)
    assert restore[h] == big  # byte-exact original recoverable
    assert digest_block("short" * (MIN_CHARS // 10), {}) is not None


def test_relevance_gate_keeps_working_set_full_digests_periphery():
    from benchmarks.swe_bench_e2e.compress_proxy import GATE_RECENT, compress_body

    msgs = [{"role": "system", "content": "sys " + "s" * 600}]
    for i in range(10):
        msgs.append(
            {"role": "user", "content": [{"type": "text", "text": f"FILE{i}\n" + "x" * 1500}]}
        )
    restore: dict[str, str] = {}
    out = compress_body(
        {"messages": msgs}, None, CompressStats(), digest_restore=restore, gate_recent=GATE_RECENT
    )
    ut = out["messages"][1:]
    digested = [i for i, m in enumerate(ut) if "distil-digest" in m["content"][0]["text"]]
    full = [i for i, m in enumerate(ut) if "distil-digest" not in m["content"][0]["text"]]
    assert full == list(range(10 - GATE_RECENT, 10))  # last N = working set, kept full
    assert digested == list(range(0, 10 - GATE_RECENT))  # older periphery digested
    assert len(restore) == len(digested)  # only digested blocks are recoverable
    assert "distil-digest" not in out["messages"][0]["content"]  # system never digested


def test_ungated_digest_compresses_all_eligible():
    from benchmarks.swe_bench_e2e.compress_proxy import compress_body

    msgs = [
        {"role": "user", "content": [{"type": "text", "text": f"F{i}\n" + "x" * 1500}]}
        for i in range(5)
    ]
    restore: dict[str, str] = {}
    compress_body(
        {"messages": msgs}, None, CompressStats(), digest_restore=restore, gate_recent=None
    )
    assert len(restore) == 5  # no gate ⇒ every eligible block digested


def test_sticky_expansion_keeps_recovered_block_full_on_later_turns():
    from benchmarks.swe_bench_e2e.compress_proxy import _handle, compress_body

    block = "def f():\n" + "x" * 1500
    msgs = [{"role": "user", "content": [{"type": "text", "text": block}]}]
    h = _handle(block)

    # Turn 1: not yet expanded -> digested.
    r1: dict[str, str] = {}
    out1 = compress_body(
        {"messages": msgs}, None, CompressStats(), digest_restore=r1, expanded=set()
    )
    assert "distil-digest" in out1["messages"][0]["content"][0]["text"]

    # Turn N: the agent already recovered this handle -> kept FULL (no re-expansion thrash),
    # but still recorded as recoverable.
    r2: dict[str, str] = {}
    out2 = compress_body({"messages": msgs}, None, CompressStats(), digest_restore=r2, expanded={h})
    txt = out2["messages"][0]["content"][0]["text"]
    assert "distil-digest" not in txt and txt == block
    assert r2[h] == block


def test_gated_cache_breakpoint_at_stable_boundary():
    from benchmarks.swe_bench_e2e.compress_proxy import GATE_RECENT, compress_body

    msgs = [{"role": "system", "content": "sys"}]
    for i in range(10):
        msgs.append({"role": "user", "content": [{"type": "text", "text": f"F{i}\n" + "x" * 1500}]})
    out = compress_body(
        {"messages": msgs}, None, CompressStats(), digest_restore={}, gate_recent=GATE_RECENT
    )
    marked = [
        i
        for i, m in enumerate(out["messages"])
        if isinstance(m.get("content"), list)
        and any(isinstance(b, dict) and "cache_control" in b for b in m["content"])
    ]
    ut = [i for i, m in enumerate(out["messages"]) if m.get("role") in ("user", "tool")]
    ws_start = ut[-GATE_RECENT]
    # exactly one breakpoint, at the last periphery message before the working set —
    # caches the byte-stable digested prefix (read) instead of re-creating it each turn.
    assert marked == [ws_start - 1]


def test_openai_proxy_gate_keeps_working_set_full():
    # Regression: the OpenAI proxy must implement the relevance gate (it previously did
    # not, silently compressing 100% of blocks and invalidating the distil_gated condition
    # on the openai/DeepSeek backend).
    from benchmarks.swe_bench_e2e.compress_proxy import GATE_RECENT
    from benchmarks.swe_bench_e2e.compress_proxy_openai import compress_body_openai

    msgs = [{"role": "system", "content": "sys"}]
    for i in range(10):
        msgs.append({"role": "user", "content": [{"type": "text", "text": f"F{i}\n" + "x" * 1500}]})
    restore: dict[str, str] = {}
    out = compress_body_openai(
        {"messages": msgs}, None, CompressStats(), digest_restore=restore, gate_recent=GATE_RECENT
    )
    ut = out["messages"][1:]
    digested = [i for i, m in enumerate(ut) if "distil-digest" in m["content"][0]["text"]]
    full = [i for i, m in enumerate(ut) if "distil-digest" not in m["content"][0]["text"]]
    assert full == list(range(10 - GATE_RECENT, 10))  # working set kept full
    assert digested == list(range(0, 10 - GATE_RECENT))  # older periphery digested
    assert len(restore) == len(digested)


def test_openai_proxy_sticky_expansion():
    from benchmarks.swe_bench_e2e.compress_proxy import _handle
    from benchmarks.swe_bench_e2e.compress_proxy_openai import compress_body_openai

    block = "def f():\n" + "x" * 1500
    msgs = [{"role": "user", "content": [{"type": "text", "text": block}]}]
    out = compress_body_openai(
        {"messages": msgs}, None, CompressStats(), digest_restore={}, expanded={_handle(block)}
    )
    assert out["messages"][0]["content"][0]["text"] == block  # sticky: kept full
