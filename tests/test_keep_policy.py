"""Per-content-type keep policy (compress/keep_policy.py) and category-breakdown
digest markers.

Guards a measured failure: a passing test log's ``N passed`` verdict was dropped
by the content-blind digest because the summary line matched no keep rule and
``tail=1`` missed the multi-line footer. The log policy pins the result summary.
"""

from __future__ import annotations

from distil.compress.keep_policy import (
    ContentKind,
    classify,
    must_keep,
    summarize_dropped,
)
from distil.compress.tier1 import digest


def _vitest_log() -> str:
    """A realistic all-passing vitest run: many per-test lines, some stderr
    noise, and a multi-line summary footer with the pass counts."""
    lines = ["RUN  v3.2.4  /repo/app", ""]
    for i in range(30):
        lines.append(f" ✓ src/lib/__tests__/mod{i}.test.ts ({i % 5 + 1} tests) {i}ms")
    lines += [
        "stderr | src/crypto/__tests__/decrypt.test.ts",
        "ERROR [Crypto] Decryption failed",
        "ERROR [Crypto] Legacy decryption failed",
        "WARN [Time] Timezone conversion failed, falling back",
        "",
        " Test Files  188 passed (188)",
        "      Tests  1955 passed (1955)",
        "   Start at  11:22:29",
        "   Duration  19.27s",
    ]
    return "\n".join(lines)


# ---- classify ----

def test_classify_test_log_is_log():
    assert classify(_vitest_log()) is ContentKind.LOG


def test_classify_traceback():
    tb = (
        "Traceback (most recent call last):\n"
        '  File "a.py", line 3, in <module>\n'
        "    x()\n"
        "ValueError: boom"
    )
    assert classify(tb) is ContentKind.TRACEBACK


def test_classify_diff():
    d = "diff --git a/x b/x\n@@ -1,3 +1,4 @@\n-old\n+new\n context"
    assert classify(d) is ContentKind.DIFF


def test_classify_prose_is_generic():
    prose = "just some notes about the plan\nand more prose\nnothing structured here"
    assert classify(prose) is ContentKind.GENERIC


# ---- per-type must_keep ----

def test_generic_keeps_decision_and_errors_only():
    assert must_keep("DECISION: do X", ContentKind.GENERIC)
    assert must_keep("ERROR: boom", ContentKind.GENERIC)
    assert not must_keep("just a normal line", ContentKind.GENERIC)


def test_log_policy_keeps_pass_summary_that_generic_drops():
    line = "      Tests  1955 passed (1955)"
    assert must_keep(line, ContentKind.LOG)  # the fix
    assert not must_keep(line, ContentKind.GENERIC)  # old content-blind behavior


def test_log_policy_keeps_test_files_and_duration():
    assert must_keep(" Test Files  188 passed (188)", ContentKind.LOG)
    assert must_keep("   Duration  19.27s", ContentKind.LOG)


def test_log_policy_does_not_over_keep_per_test_lines():
    # a per-test line has "(N tests)" but no result summary; must stay droppable
    assert not must_keep(" ✓ src/a.test.ts (3 tests) 5ms", ContentKind.LOG)


def test_traceback_policy_keeps_frames():
    assert must_keep('  File "a.py", line 3, in f', ContentKind.TRACEBACK)
    assert not must_keep("some plain middle line", ContentKind.TRACEBACK)


def test_diff_policy_keeps_hunk_and_file_headers():
    assert must_keep("@@ -1,3 +1,4 @@", ContentKind.DIFF)
    assert must_keep("diff --git a/x b/x", ContentKind.DIFF)
    assert not must_keep(" unchanged context line", ContentKind.DIFF)


# ---- marker category breakdown ----

def test_summarize_dropped_breaks_down_when_flagged_lines_present():
    lines = ["ERROR boom", "ERROR again", "WARN careful", "ordinary line", "another ordinary"]
    s = summarize_dropped(lines)
    assert "2 error" in s
    assert "1 warn" in s
    assert "2 other" in s


def test_summarize_dropped_empty_when_all_mundane():
    # no error/warn in the folded span means no breakdown worth the tokens
    assert summarize_dropped(["plain a", "plain b", "info c"]) == ""


# ---- HEADLINE: digest integration retains the verdict ----

def test_digest_retains_test_verdict():
    log = _vitest_log()
    dtext, changed = digest(log)
    assert changed
    assert len(dtext) < len(log)  # still compresses
    assert "1955 passed" in dtext  # the verdict survives (was dropped before)
    assert "188 passed" in dtext


def test_digest_marker_bare_when_folded_span_is_mundane():
    # the vitest log's dropped lines are all passing tests (no error/warn), so the
    # marker carries no breakdown, just the count and the recovery handle
    log = _vitest_log()
    dtext, _ = digest(log)
    marker = next(line for line in dtext.splitlines() if "omitted" in line)
    assert "handle=" in marker  # still expandable
    assert "(" not in marker  # no breakdown for a mundane fold


def test_digest_tail_zero_flushes_trailing_dropped_run():
    # with tail=0 a block can end on a dropped run, exercising the final flush
    text = "\n".join(["DECISION: keep me"] + [f"drop {i}" for i in range(10)])
    dtext, changed = digest(text, head=1, tail=0)
    assert changed
    assert "DECISION: keep me" in dtext
    assert dtext.rstrip().endswith(">>")  # trailing marker for the final dropped run
    assert "10 lines omitted" in dtext


def test_generic_digest_unchanged_for_prose():
    # a long generic block still digests, and drops non-salient middle lines
    text = "\n".join(["header a", "header b", "header c"] + [f"filler {i}" for i in range(20)] + ["tail z"])
    dtext, changed = digest(text)
    assert changed
    assert "filler 10" not in dtext  # middle folded
    assert "header a" in dtext and "tail z" in dtext
