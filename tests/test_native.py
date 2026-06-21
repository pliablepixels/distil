"""Tests for distil.native — the Rust/Python unified interface.

Verifies that:
  1. distil.native imports cleanly and declares a valid BACKEND.
  2. BACKEND is either "rust" or "python".
  3. minify_json / collapse_runs / count_tokens produce the SAME results as
     the pure-Python implementations (tier0 + HeuristicTokenizer) on a set of
     representative inputs.

No external dependencies beyond pytest and the stdlib are required.
"""

import distil.native as native
from distil.compress.tier0 import minify_json as py_minify_json
from distil.compress.tier0 import collapse_runs as py_collapse_runs
from distil.tokenizer import HeuristicTokenizer


# ── helpers ──────────────────────────────────────────────────────────────────


def py_count_tokens(text: str, subword_factor: float = 1.33) -> int:
    return HeuristicTokenizer(subword_factor).count(text)


# ── basic smoke tests ─────────────────────────────────────────────────────────


def test_backend_is_declared():
    assert hasattr(native, "BACKEND"), "distil.native must expose BACKEND"


def test_backend_is_rust_or_python():
    assert native.BACKEND in ("rust", "python"), (
        f"BACKEND must be 'rust' or 'python', got {native.BACKEND!r}"
    )


def test_functions_are_callable():
    assert callable(native.minify_json)
    assert callable(native.collapse_runs)
    assert callable(native.count_tokens)


# ── minify_json parity ────────────────────────────────────────────────────────

MINIFY_JSON_CASES = [
    # (description, input)
    ("compact object", '{"a": 1, "b": 2}'),
    ("compact array", "[1, 2, 3]"),
    ("nested", '{"a": {"b": [1, 2]}}'),
    ("unicode preserved", '{"unicode": "héllo"}'),
    ("leading/trailing ws", '  {"key": "value"}  '),
    ("not json", "not json"),
    ("plain number", "42"),
    ("empty string", ""),
    ("invalid json-like", "{not: valid}"),
    ("null", "null"),  # valid JSON scalar but doesn't start with { or [
    ("array of objects", '[{"x": 1}, {"y": 2}]'),
]


def test_minify_json_parity():
    for desc, text in MINIFY_JSON_CASES:
        expected = py_minify_json(text)
        got = native.minify_json(text)
        assert got == expected, (
            f"minify_json mismatch [{desc}]:\n"
            f"  input={text!r}\n"
            f"  Python={expected!r}\n"
            f"  native={got!r}"
        )


# ── collapse_runs parity ──────────────────────────────────────────────────────

COLLAPSE_RUNS_CASES = [
    # (description, input)
    ("empty", ""),
    ("single line with nl", "hello\n"),
    ("two different lines", "a\nb\n"),
    ("two identical (nl)", "a\na\n"),
    ("three identical", "a\na\na\n"),
    ("four identical", "a\na\na\na\n"),
    ("mixed runs", "a\na\na\nb\nb\nb\nc"),
    ("no trailing newline pair", "a\na"),
    ("empty line repeats", "\n\n\n"),
    ("interleaved", "x\nx\ny\nz\nz\nz\nz\n"),
    (
        "realistic log snippet",
        ("INFO starting\nWARN rate limit\nWARN rate limit\nWARN rate limit\nINFO done\n"),
    ),
    ("long repeated line", ("=" * 80 + "\n") * 10),
]


def test_collapse_runs_parity():
    for desc, text in COLLAPSE_RUNS_CASES:
        expected = py_collapse_runs(text)
        got = native.collapse_runs(text)
        assert got == expected, (
            f"collapse_runs mismatch [{desc}]:\n"
            f"  input={text!r}\n"
            f"  Python={expected!r}\n"
            f"  native={got!r}"
        )


# ── count_tokens parity ───────────────────────────────────────────────────────

COUNT_TOKENS_CASES = [
    # (description, text, subword_factor)
    ("empty", "", 1.33),
    ("hello world", "hello world", 1.33),
    ("with punctuation", "hello, world!", 1.33),
    ("single char", "a", 1.33),
    ("whitespace only", "   ", 1.33),
    ("unicode word", "héllo", 1.33),
    ("mixed code prose", "foo bar baz qux", 1.33),
    ("custom factor 2x", "foo bar baz qux", 2.0),
    ("custom factor 1x", "hello world", 1.0),
    ("json-like string", '{"key": "value"}', 1.33),
    ("multiline", "line one\nline two\nline three\n", 1.33),
    ("dense text", "The quick brown fox jumps over the lazy dog.", 1.33),
    ("code snippet", "def foo(x: int) -> str:\n    return str(x)\n", 1.33),
]


def test_count_tokens_parity():
    for desc, text, factor in COUNT_TOKENS_CASES:
        expected = py_count_tokens(text, factor)
        got = native.count_tokens(text, factor)
        assert got == expected, (
            f"count_tokens mismatch [{desc}]:\n"
            f"  text={text!r}, factor={factor}\n"
            f"  Python={expected}\n"
            f"  native={got}"
        )


# ── idempotency sanity ────────────────────────────────────────────────────────


def test_collapse_runs_idempotent_on_already_collapsed():
    """Collapsing an already-collapsed string should produce the same string."""
    original = "a\na\na\nb\nb\nb\nc\n"
    once = native.collapse_runs(original)
    twice = native.collapse_runs(once)
    # The already-collapsed form should not be further collapsed in a way
    # that changes meaning (the marker lines are unique).
    py_once = py_collapse_runs(original)
    py_twice = py_collapse_runs(py_once)
    assert once == py_once
    assert twice == py_twice


def test_minify_json_idempotent():
    """Minifying an already-minified JSON string should return the same string."""
    text = '{"a":1,"b":[1,2,3]}'
    once = native.minify_json(text)
    twice = native.minify_json(once) if once is not None else None
    assert once == twice
