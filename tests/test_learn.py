"""The learning flywheel — expand signals → content-free keep policy."""

from __future__ import annotations

import json

from distil.learn import ExpandStats, keep_predicate, signature


def test_signature_is_coarse_and_content_free():
    big_json = json.dumps([{"id": i} for i in range(60)], indent=2)
    assert signature(big_json).startswith("json:")
    assert signature("Traceback (most recent call last):\n  File x").startswith("error:")
    assert signature("\n".join(f"2026-06-22 INFO req {i}" for i in range(50))).startswith("log:")
    # signature carries no content — just class:size
    assert ":" in signature("hello") and "hello" not in signature("hello world")


def test_expand_prone_needs_evidence_and_a_high_rate():
    s = ExpandStats()
    for _ in range(10):
        s.record_digest("json:l")
    for _ in range(4):  # 4/10 = 40% expand rate → over threshold
        s.record_expand("json:l")
    for _ in range(10):
        s.record_digest("log:l")  # never expanded → stays digestible
    prone = s.expand_prone(min_digested=5, threshold=0.25)
    assert prone == {"json:l"}
    # below the sample floor, no policy even at 100% rate
    s.record_digest("code:s")
    s.record_expand("code:s")
    assert "code:s" not in s.expand_prone(min_digested=5, threshold=0.25)


def test_keep_predicate_keeps_only_prone_signatures():
    s = ExpandStats()
    for _ in range(8):
        s.record_digest("error:m")
    for _ in range(4):
        s.record_expand("error:m")
    keep = keep_predicate(s, min_digested=5, threshold=0.25)
    assert (
        keep("Traceback (most recent call last):\n" + "  File a\n" * 20) is True
    )  # error:m → kept
    assert keep("just a short note") is False


def test_stats_persist_atomically(tmp_path):
    p = tmp_path / "stats.json"
    s = ExpandStats()
    s.record_digest("json:l")
    s.record_expand("json:l")
    s.save(p)
    loaded = ExpandStats.load(p)
    assert loaded.digested == {"json:l": 1} and loaded.expanded == {"json:l": 1}
    # corrupt file → graceful empty, never raises
    p.write_text("{ not json")
    assert ExpandStats.load(p).digested == {}
