"""Outcome-guided compression policy + surprise-preserving retention."""

from __future__ import annotations

from distil.compress.guideline import OutcomeStats, record_trajectory_outcome, signature
from distil.compress.salience import salient_lines, surprise_lines


def test_outcome_stats_learns_regression_prone_signatures(tmp_path):
    path = tmp_path / "outcome-stats.json"
    err = "Traceback (most recent call last):\n" + "\n".join(f"  File x{i}" for i in range(12))
    prose = "meeting notes\n" + "\n".join(f"agenda item {i}" for i in range(12))
    # error-class content digested in 5 regressed runs; prose digested in 5 fine runs
    for _ in range(5):
        record_trajectory_outcome([err], full_success=True, compressed_success=False, path=path)
        record_trajectory_outcome([prose], full_success=True, compressed_success=True, path=path)
    stats = OutcomeStats.load(path)
    prone = stats.protect_prone(min_seen=5, threshold=0.3)
    assert signature(err) in prone
    assert signature(prose) not in prone
    keep = stats.keep_predicate(min_seen=5, threshold=0.3)
    assert keep(err) and not keep(prose)


def test_outcome_stats_ignore_tasks_full_context_also_failed(tmp_path):
    path = tmp_path / "outcome-stats.json"
    err = "Error: unfixable\n" + "x\n" * 12
    for _ in range(10):
        record_trajectory_outcome([err], full_success=False, compressed_success=False, path=path)
    stats = OutcomeStats.load(path)
    assert stats.protect_prone(min_seen=1) == set()  # no evidence recorded at all


def test_never_regressing_min_seen_guard(tmp_path):
    path = tmp_path / "outcome-stats.json"
    err = "Error: flaky one-off\n" + "y\n" * 12
    record_trajectory_outcome([err], full_success=True, compressed_success=False, path=path)
    stats = OutcomeStats.load(path)
    # a single unlucky failure must not flip policy — needs min_seen samples
    assert stats.protect_prone(min_seen=5) == set()


def test_surprise_lines_keep_anomalies():
    text = (
        "step 1 completed normally\n"
        "step 2 completed normally\n"
        "AssertionError: expected 200 got 500\n"
        "process exited 1\n"
        "step 4 completed normally\n"
    )
    got = surprise_lines(text)
    assert "AssertionError: expected 200 got 500" in got
    assert "process exited 1" in got
    assert "step 1 completed normally" not in got


def test_surprise_lines_keep_diff_changes():
    diff = "--- a/f.py\n+++ b/f.py\n context line\n-    return a + b\n+    return a - b\n"
    got = surprise_lines(diff)
    assert "-    return a + b" in got
    assert "+    return a - b" in got
    assert "--- a/f.py" not in got  # headers aren't the change


def test_salient_lines_include_surprises_and_paths():
    text = (
        "just some prose here\n"
        "unexpected token in src/distil/compress/salience.py\n"
        "all good otherwise\n"
    )
    got = salient_lines(text)
    assert any("unexpected token" in ln for ln in got)
    assert not any(ln == "all good otherwise" for ln in got)
