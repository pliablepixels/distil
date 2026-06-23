"""AST-structural delta — definition-level, formatting/comment/import-order invariant."""

from __future__ import annotations

import difflib

from distil.astdelta import structural_delta

# Realistic: large unchanged bodies (helper, Config) + a small definition that changes.
OLD = """\
import os
import sys


def helper(x):
    a = x + 1
    b = a * 2
    c = b - 3
    d = c ** 2
    e = d % 7
    f = e + a
    g = f * b
    return a + b + c + d + e + f + g


def process(data):
    return [helper(d) for d in data]


class Config:
    DEBUG = False
    VERBOSE = True
    MAX_RETRIES = 5
    TIMEOUT = 30
    NAME = "distil"
    REGION = "us-east-1"
    SHARDS = 8

    def to_dict(self):
        return {"debug": self.DEBUG, "name": self.NAME}
"""

# process() logic changed; helper() gets a comment (same AST); imports reordered.
NEW_LOGIC_CHANGE = """\
import sys
import os


def helper(x):
    # accumulate a polynomial of x
    a = x + 1
    b = a * 2
    c = b - 3
    d = c ** 2
    e = d % 7
    f = e + a
    g = f * b
    return a + b + c + d + e + f + g


def process(data):
    return [helper(d) * 2 for d in data]


class Config:
    DEBUG = False
    VERBOSE = True
    MAX_RETRIES = 5
    TIMEOUT = 30
    NAME = "distil"
    REGION = "us-east-1"
    SHARDS = 8

    def to_dict(self):
        return {"debug": self.DEBUG, "name": self.NAME}
"""

# ONLY formatting/comments/import-order changed — no definition's meaning changed.
NEW_FORMAT_ONLY = """\
import sys

import os


def helper(x):

    # accumulate a polynomial of x
    a = x + 1
    b = a * 2
    c = b - 3
    d = c ** 2
    e = d % 7
    f = e + a
    g = f * b
    return a + b + c + d + e + f + g


def process(data):
    return [helper(d) for d in data]


class Config:
    DEBUG = False
    VERBOSE = True
    MAX_RETRIES = 5
    TIMEOUT = 30
    NAME = "distil"
    REGION = "us-east-1"
    SHARDS = 8

    def to_dict(self):
        return {"debug": self.DEBUG, "name": self.NAME}
"""


def test_isolates_the_one_changed_definition():
    d = structural_delta(OLD, NEW_LOGIC_CHANGE, "AAAA", "BBBB")
    assert d is not None
    assert "changed def process" in d
    assert "* 2" in d  # the new body is carried
    assert "unchanged (still in context above): def helper, class Config" in d
    assert "a = x + 1" not in d  # helper body NOT re-sent
    assert "handle=BBBB" in d  # expand-recoverable


def test_format_only_change_is_recognized_as_no_change():
    d = structural_delta(OLD, NEW_FORMAT_ONLY, "AAAA", "BBBB")
    assert d is not None
    assert "no definition changed" in d  # the AST advantage: format/comment-invariant


def test_structural_beats_textual_on_reformatting():
    # Heavy reformatting of the UNCHANGED helper (a comment before every statement —
    # AST-invariant), plus a real change to process. Textual diff shows every
    # reformatted line; the structural delta references helper untouched and sends
    # only process — so it is dramatically smaller exactly where textual explodes.
    body_lines = [
        "a = x + 1",
        "b = a * 2",
        "c = b - 3",
        "d = c ** 2",
        "e = d % 7",
        "f = e + a",
        "g = f * b",
        "return a + b + c + d + e + f + g",
    ]
    old_helper = "def helper(x):\n" + "".join(f"    {ln}\n" for ln in body_lines)
    new_helper = "def helper(x):\n" + "".join(
        f"    # compute {ln}\n    {ln}\n" for ln in body_lines
    )
    new = OLD.replace(old_helper, new_helper).replace(
        "[helper(d) for d in data]", "[helper(d) * 2 for d in data]"
    )
    assert new != OLD and "# compute" in new

    textual = "".join(
        difflib.unified_diff(OLD.splitlines(keepends=True), new.splitlines(keepends=True))
    )
    structural = structural_delta(OLD, new, "AAAA", "BBBB")
    assert structural is not None
    assert "changed def process" in structural and "# compute" not in structural
    assert len(structural) < len(textual)  # AST ignores the reformat noise; textual can't


def test_added_and_removed_definitions():
    new = OLD + "\n\ndef extra():\n    return 99\n"
    d = structural_delta(OLD, new, "AAAA", "BBBB")
    assert d is not None and "added def extra" in d

    removed_src = OLD.replace("def process(data):\n    return [helper(d) for d in data]\n\n\n", "")
    d2 = structural_delta(OLD, removed_src, "AAAA", "BBBB")
    assert d2 is not None and "removed:" in d2 and "process" in d2


def test_import_reorder_alone_is_no_change():
    reordered = OLD.replace("import os\nimport sys", "import sys\nimport os")
    d = structural_delta(OLD, reordered, "AAAA", "BBBB")
    assert d is not None and "no definition changed" in d


def test_non_python_returns_none():
    assert (
        structural_delta("function f() { return 1; }", "function f() { return 2; }", "A", "B")
        is None
    )


def test_unparseable_returns_none():
    # A mid-edit file that doesn't parse must fall back (return None), never raise.
    assert structural_delta(OLD, "def broken(:\n  pass", "A", "B") is None


def test_integrates_into_cachedelta_for_python():
    from distil.cachedelta import DeltaSession, delta_encode

    def _tr(t):
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t", "content": t}],
        }

    s = DeltaSession()
    delta_encode([{"role": "user", "content": "edit"}, _tr(OLD)], session=s)
    out, store, stats = delta_encode(
        [
            {"role": "user", "content": "edit"},
            _tr(OLD),
            {"role": "user", "content": "again"},
            _tr(NEW_LOGIC_CHANGE),
        ],
        session=s,
    )
    assert stats.delta_refs == 1
    seen = out[3]["content"][0]["content"]
    assert "«distil-ast" in seen  # the AST path was taken, not the textual fallback
    assert any(store.expand(h) == NEW_LOGIC_CHANGE for h in store.handles)  # reversible
