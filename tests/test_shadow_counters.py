"""Shadow counter diagnostics — content-free sampling observability."""

from __future__ import annotations

import json
from pathlib import Path


from distil.shadow import ShadowCounters


# ---------------------------------------------------------------------------
# ShadowCounters unit tests
# ---------------------------------------------------------------------------


def test_note_seen_increments_in_memory(tmp_path: Path) -> None:
    p = tmp_path / "ctr.json"
    c = ShadowCounters(path=p)
    c.note_seen()
    c.note_seen()
    with c._lock:
        assert c._pending.get("requests_seen") == 2


def test_note_sampled_increments_in_memory(tmp_path: Path) -> None:
    p = tmp_path / "ctr.json"
    c = ShadowCounters(path=p)
    c.note_sampled()
    with c._lock:
        assert c._pending.get("sampled") == 1


def test_flush_with_record_path(tmp_path: Path) -> None:
    p = tmp_path / "ctr.json"
    c = ShadowCounters(path=p)
    c.note_seen()
    c.note_sampled()
    c.flush_with(replay_attempted=True, recorded=True)
    data = json.loads(p.read_text())
    assert data["requests_seen"] == 1
    assert data["sampled"] == 1
    assert data["replay_attempted"] == 1
    assert data.get("replay_failed", 0) == 0
    assert data["recorded"] == 1
    # pending should be drained
    with c._lock:
        assert c._pending == {}


def test_flush_with_fail_path(tmp_path: Path) -> None:
    p = tmp_path / "ctr.json"
    c = ShadowCounters(path=p)
    c.note_seen()
    c.note_sampled()
    c.flush_with(replay_attempted=True, replay_failed=True, fail_reason="401")
    data = json.loads(p.read_text())
    assert data["replay_attempted"] == 1
    assert data["replay_failed"] == 1
    assert data["last_fail_reason"] == "401"
    assert data.get("recorded", 0) == 0


def test_flush_with_sig_none_skipped(tmp_path: Path) -> None:
    p = tmp_path / "ctr.json"
    c = ShadowCounters(path=p)
    c.flush_with(replay_attempted=True, sig_none_skipped=True)
    data = json.loads(p.read_text())
    assert data["replay_attempted"] == 1
    assert data["signature_none_skipped"] == 1
    assert data.get("replay_failed", 0) == 0


def test_flush_accumulates(tmp_path: Path) -> None:
    p = tmp_path / "ctr.json"
    c = ShadowCounters(path=p)
    c.flush_with(replay_attempted=True, recorded=True)
    c.flush_with(replay_attempted=True, replay_failed=True, fail_reason="401")
    data = json.loads(p.read_text())
    assert data["replay_attempted"] == 2
    assert data["recorded"] == 1
    assert data["replay_failed"] == 1


def test_load_returns_empty_when_missing(tmp_path: Path) -> None:
    p = tmp_path / "nonexistent.json"
    assert ShadowCounters.load(path=p) == {}


def test_load_returns_empty_on_corrupt(tmp_path: Path) -> None:
    p = tmp_path / "ctr.json"
    p.write_text("not json")
    assert ShadowCounters.load(path=p) == {}


def test_load_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "ctr.json"
    c = ShadowCounters(path=p)
    c.note_seen()
    c.note_seen()
    c.note_sampled()
    c.flush_with(replay_attempted=True, recorded=True)
    loaded = ShadowCounters.load(path=p)
    assert loaded["requests_seen"] == 2
    assert loaded["sampled"] == 1
    assert loaded["recorded"] == 1


# ---------------------------------------------------------------------------
# Stats renderer: diagnostic line when recorded==0 but attempted>0
# ---------------------------------------------------------------------------


def _render_shadow_stats(counters: dict, *, samples: int = 0) -> str:
    """Minimal inline reproduction of the cmd_shadow_stats diagnostic logic."""
    lines: list[str] = []
    if samples == 0:
        attempted = counters.get("replay_attempted", 0)
        if attempted > 0:
            seen = counters.get("requests_seen", 0)
            sampled = counters.get("sampled", 0)
            failed = counters.get("replay_failed", 0)
            reason = counters.get("last_fail_reason", "")
            reason_str = f" (last: {reason})" if reason else ""
            lines.append(
                f"Shadow counters: {seen} seen, {sampled} sampled, "
                f"{attempted} replay{'s' if attempted != 1 else ''} attempted, "
                f"{failed} failed{reason_str}"
            )
        else:
            lines.append("No shadow samples yet.")
    else:
        attempted = counters.get("replay_attempted", 0)
        failed = counters.get("replay_failed", 0)
        reason = counters.get("last_fail_reason", "")
        recorded = counters.get("recorded", 0)
        seen = counters.get("requests_seen", 0)
        sampled = counters.get("sampled", 0)
        reason_str = f" (last: {reason})" if reason and failed else ""
        lines.append(
            f"Sampling: {seen} seen, {sampled} sampled, "
            f"{attempted} attempted, {failed} failed{reason_str}, {recorded} recorded"
        )
    return "\n".join(lines)


def test_renderer_shows_diagnostic_when_attempted_but_no_samples() -> None:
    ctrs = {
        "requests_seen": 19,
        "sampled": 2,
        "replay_attempted": 2,
        "replay_failed": 2,
        "last_fail_reason": "401",
    }
    out = _render_shadow_stats(ctrs, samples=0)
    assert "19 seen" in out
    assert "2 sampled" in out
    assert "2 replays attempted" in out
    assert "2 failed" in out
    assert "401" in out
    assert "No shadow samples yet" not in out


def test_renderer_shows_no_samples_when_no_attempts() -> None:
    out = _render_shadow_stats({}, samples=0)
    assert "No shadow samples yet" in out


def test_renderer_shows_counter_line_when_samples_exist() -> None:
    ctrs = {
        "requests_seen": 50,
        "sampled": 5,
        "replay_attempted": 5,
        "replay_failed": 0,
        "recorded": 5,
    }
    out = _render_shadow_stats(ctrs, samples=30)
    assert "50 seen" in out
    assert "5 sampled" in out
    assert "5 recorded" in out
    assert "0 failed" in out
