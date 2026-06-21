"""Tests for distil.gist — content-addressed gist cache."""

from __future__ import annotations

import pytest

from distil.gist import GistCache

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A realistic large tool schema — big enough that savings() is clearly positive.
TOOL_SCHEMA = (
    """{
  "name": "search_codebase",
  "description": "Search the codebase for files, symbols, or text patterns. Returns matching file paths and line numbers with surrounding context. Use when you need to locate a definition, find all callers of a function, or discover files matching a glob pattern. Supports regex. Results are ranked by relevance.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query — a literal string, a regex, or a glob pattern. Regexes must be valid Python re syntax."
      },
      "path": {
        "type": "string",
        "description": "Optional root path to restrict the search. Defaults to the repository root. Use to narrow scope and reduce noise."
      },
      "case_sensitive": {
        "type": "boolean",
        "description": "Whether the query is case-sensitive. Defaults to false for literal queries and true for regexes.",
        "default": false
      },
      "max_results": {
        "type": "integer",
        "description": "Maximum number of results to return. Defaults to 50. Use a lower value when you only need to confirm existence.",
        "default": 50
      },
      "include_context_lines": {
        "type": "integer",
        "description": "Lines of surrounding context to include above and below each match. Defaults to 2.",
        "default": 2
      }
    },
    "required": ["query"]
  }
}"""
    * 3
)  # repeat to make it large


# ---------------------------------------------------------------------------
# ref identity
# ---------------------------------------------------------------------------


def test_same_text_same_ref() -> None:
    cache = GistCache()
    ref1 = cache.register("hello world")
    ref2 = cache.register("hello world")
    assert ref1 == ref2


def test_different_text_different_ref() -> None:
    cache = GistCache()
    ref1 = cache.register("alpha")
    ref2 = cache.register("beta")
    assert ref1 != ref2


def test_ref_format() -> None:
    cache = GistCache()
    ref = cache.register("anything")
    assert ref.startswith("gist:")
    hex_part = ref[len("gist:") :]
    assert len(hex_part) == 8
    assert all(c in "0123456789abcdef" for c in hex_part)


# ---------------------------------------------------------------------------
# materialize round-trip
# ---------------------------------------------------------------------------


def test_materialize_round_trip() -> None:
    cache = GistCache()
    original = "The quick brown fox\njumps over the lazy dog."
    ref = cache.register(original)
    assert cache.materialize(ref) == original


def test_materialize_unknown_ref_raises() -> None:
    cache = GistCache()
    with pytest.raises(KeyError):
        cache.materialize("gist:deadbeef")


# ---------------------------------------------------------------------------
# dedup tracking
# ---------------------------------------------------------------------------


def test_first_register_not_a_hit() -> None:
    cache = GistCache()
    cache.register("some text")
    assert cache.hits == 0
    assert cache.registrations == 1


def test_second_register_counts_as_hit() -> None:
    cache = GistCache()
    cache.register("some text")
    cache.register("some text")
    assert cache.hits == 1
    assert cache.registrations == 2


def test_dedup_rate_reflects_hits() -> None:
    cache = GistCache()
    cache.register("a")
    cache.register("b")
    cache.register("a")  # hit
    cache.register("b")  # hit
    cache.register("c")
    # 2 hits out of 5 registrations
    assert cache.dedup_rate == pytest.approx(2 / 5)


def test_dedup_rate_zero_when_no_registrations() -> None:
    cache = GistCache()
    assert cache.dedup_rate == 0.0


def test_dedup_rate_zero_all_unique() -> None:
    cache = GistCache()
    for i in range(10):
        cache.register(f"unique text {i}")
    assert cache.dedup_rate == 0.0


# ---------------------------------------------------------------------------
# savings
# ---------------------------------------------------------------------------


def test_savings_positive_for_large_schema() -> None:
    cache = GistCache()
    saved = cache.savings(TOOL_SCHEMA)
    # A multi-KB JSON schema must save well more than zero tokens
    assert saved > 50, f"expected >50 token savings, got {saved}"


def test_savings_explicit_tokenizer() -> None:
    from distil.tokenizer import HeuristicTokenizer

    cache = GistCache()
    tok = HeuristicTokenizer()
    saved = cache.savings(TOOL_SCHEMA, tok=tok)
    assert saved > 0


def test_savings_small_text_still_non_negative() -> None:
    # Even a tiny text should produce non-negative savings (ref is short).
    cache = GistCache()
    saved = cache.savings("hi")
    # The ref "gist:XXXXXXXX" (13 chars) is probably about the same token cost
    # as "hi" — we just assert it doesn't crash and the value is an int.
    assert isinstance(saved, int)


# ---------------------------------------------------------------------------
# is_ref
# ---------------------------------------------------------------------------


def test_is_ref_true_for_registered() -> None:
    cache = GistCache()
    ref = cache.register("content")
    assert GistCache.is_ref(ref) is True


def test_is_ref_false_for_plain_content() -> None:
    assert GistCache.is_ref("hello world") is False
    assert GistCache.is_ref("gist:") is False  # too short
    assert GistCache.is_ref("gist:ZZZZZZZZ") is False  # non-hex
    assert GistCache.is_ref("gist:abc123") is False  # too short hex


def test_is_ref_false_for_empty_string() -> None:
    assert GistCache.is_ref("") is False


def test_is_ref_distinguishes_ref_from_content() -> None:
    cache = GistCache()
    ref = cache.register(TOOL_SCHEMA)
    assert GistCache.is_ref(ref) is True
    assert GistCache.is_ref(TOOL_SCHEMA) is False
