"""Tests for the content-aware skeleton digest (distil/skeleton.py)."""

from __future__ import annotations

from distil.skeleton import code_skeleton, smart_digest, text_window

SAMPLE = '''\
import os
import sys
from typing import Any

CONST = 42


def helper(x: int) -> int:
    """Add one to x."""
    y = x + 1
    z = y * 2
    return z


class Widget:
    """A widget."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.parts = []

    def render(self) -> str:
        out = []
        for p in self.parts:
            out.append(str(p))
        return "".join(out)
'''


def test_code_skeleton_keeps_structure_drops_bodies():
    sk = code_skeleton(SAMPLE)
    assert sk is not None
    # Imports, constants, signatures, class header all survive.
    assert "import os" in sk
    assert "CONST = 42" in sk
    assert "def helper(x: int) -> int:" in sk
    assert "class Widget:" in sk
    assert "def __init__(self, name: str) -> None:" in sk
    assert "def render(self) -> str:" in sk
    # Bodies are elided.
    assert "z = y * 2" not in sk
    assert "for p in self.parts:" not in sk
    assert "..." in sk
    # Docstring hints kept.
    assert "Add one to x." in sk
    # Actually smaller.
    assert len(sk) < len(SAMPLE)


def test_code_skeleton_navigable_lists_every_symbol():
    sk = code_skeleton(SAMPLE)
    for sym in ("helper", "Widget", "__init__", "render"):
        assert sym in sk, f"skeleton must name {sym} so the agent can expand it"


def test_code_skeleton_none_on_non_python():
    assert code_skeleton("this is { not ] python (((") is None


def test_code_skeleton_none_when_no_bodies():
    # Pure imports/assignments: nothing to elide, skeleton wouldn't help.
    assert code_skeleton("import os\nx = 1\ny = 2\n") is None


def test_text_window_keeps_head_and_tail():
    # The traceback failure: the exception lives at the END.
    tb = "Traceback (most recent call last):\n" + ("  frame\n" * 200) + "ValueError: boom"
    w = text_window(tb, head=80, tail=80)
    assert "Traceback" in w  # head
    assert "ValueError: boom" in w  # tail — head-only truncation would drop this
    assert len(w) < len(tb)


def test_text_window_short_text_untouched():
    assert text_window("short", head=400, tail=200) == "short"


def test_smart_digest_routes_code_vs_text():
    assert "def helper" in smart_digest(SAMPLE)  # code path
    tb = "ERROR\n" + ("x\n" * 500) + "AssertionError: nope"
    d = smart_digest(tb)
    assert "AssertionError: nope" in d  # text path keeps the tail


def test_smart_digest_nested_functions_do_not_break():
    src = "def outer():\n    a = 1\n    def inner():\n        return 2\n    return inner()\n"
    sk = code_skeleton(src)
    # Outer signature kept, its whole body (incl. the closure) elided to one '...'.
    assert sk is not None
    assert "def outer():" in sk
    assert sk.count("...") == 1
    assert "return inner()" not in sk
