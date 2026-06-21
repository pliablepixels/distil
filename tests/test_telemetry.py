"""Tests for distil.telemetry — verifiable federated savings telemetry."""

from __future__ import annotations

import json
import tempfile

import pytest

from distil.telemetry import (
    Leaderboard,
    SavingsAggregate,
    _canonical,
    build_leaderboard,
    render_leaderboard_html,
    sign,
    submit,
    verify,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agg(
    instance_id: str = "inst-1",
    tokens_saved: int = 10_000,
    dollars_saved: float = 0.05,
    runs: int = 3,
    certified: bool = True,
    ts: float = 1_000_000.0,
) -> SavingsAggregate:
    return SavingsAggregate(
        instance_id=instance_id,
        tokens_saved=tokens_saved,
        dollars_saved=dollars_saved,
        runs=runs,
        certified=certified,
        ts=ts,
    )


KEY = "super-secret-key"


# ---------------------------------------------------------------------------
# _canonical
# ---------------------------------------------------------------------------


class TestCanonical:
    def test_is_valid_json(self) -> None:
        canon = _canonical(_agg())
        parsed = json.loads(canon)
        assert parsed["instance_id"] == "inst-1"

    def test_sorted_keys(self) -> None:
        canon = _canonical(_agg())
        keys = list(json.loads(canon).keys())
        assert keys == sorted(keys)

    def test_deterministic(self) -> None:
        a = _agg()
        assert _canonical(a) == _canonical(a)

    def test_no_spaces(self) -> None:
        # separators=(",",":") — no whitespace
        assert " " not in _canonical(_agg())


# ---------------------------------------------------------------------------
# sign / verify roundtrip
# ---------------------------------------------------------------------------


class TestSignVerify:
    def test_roundtrip_returns_true(self) -> None:
        agg = _agg()
        signed = sign(agg, KEY)
        assert verify(signed, KEY) is True

    def test_signed_dict_contains_sig(self) -> None:
        signed = sign(_agg(), KEY)
        assert "sig" in signed
        assert len(signed["sig"]) == 64  # hex-encoded SHA-256

    def test_mutate_tokens_fails_verify(self) -> None:
        signed = sign(_agg(tokens_saved=10_000), KEY)
        tampered = {**signed, "tokens_saved": 99_999}
        assert verify(tampered, KEY) is False

    def test_mutate_dollars_fails_verify(self) -> None:
        signed = sign(_agg(dollars_saved=0.05), KEY)
        tampered = {**signed, "dollars_saved": 9.99}
        assert verify(tampered, KEY) is False

    def test_mutate_runs_fails_verify(self) -> None:
        signed = sign(_agg(runs=3), KEY)
        tampered = {**signed, "runs": 9999}
        assert verify(tampered, KEY) is False

    def test_mutate_certified_fails_verify(self) -> None:
        signed = sign(_agg(certified=True), KEY)
        tampered = {**signed, "certified": False}
        assert verify(tampered, KEY) is False

    def test_mutate_instance_id_fails_verify(self) -> None:
        signed = sign(_agg(instance_id="inst-1"), KEY)
        tampered = {**signed, "instance_id": "evil-inst"}
        assert verify(tampered, KEY) is False

    def test_mutate_ts_fails_verify(self) -> None:
        signed = sign(_agg(ts=1_000_000.0), KEY)
        tampered = {**signed, "ts": 0.0}
        assert verify(tampered, KEY) is False

    def test_wrong_key_fails_verify(self) -> None:
        signed = sign(_agg(), KEY)
        assert verify(signed, "wrong-key") is False

    def test_missing_sig_fails_verify(self) -> None:
        signed = sign(_agg(), KEY)
        no_sig = {k: v for k, v in signed.items() if k != "sig"}
        assert verify(no_sig, KEY) is False

    def test_does_not_mutate_caller_dict(self) -> None:
        signed = sign(_agg(), KEY)
        original_keys = set(signed.keys())
        verify(signed, KEY)
        assert set(signed.keys()) == original_keys


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class TestSubmit:
    def test_creates_file_and_returns_path(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            signed = sign(_agg(), KEY)
            path = submit(signed, d)
            assert path.endswith("submissions.jsonl")
            with open(path) as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
            assert len(lines) == 1
            assert json.loads(lines[0])["instance_id"] == "inst-1"

    def test_appends_multiple(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            submit(sign(_agg(instance_id="a"), KEY), d)
            submit(sign(_agg(instance_id="b"), KEY), d)
            with open(f"{d}/submissions.jsonl") as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
            assert len(lines) == 2


# ---------------------------------------------------------------------------
# build_leaderboard
# ---------------------------------------------------------------------------


class TestBuildLeaderboard:
    def _write_submissions(self, d: str, records: list[dict]) -> None:
        path = f"{d}/submissions.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_two_valid_one_tampered(self) -> None:
        """2 valid + 1 tampered → rejected==1, 2 in verified."""
        with tempfile.TemporaryDirectory() as d:
            key_a, key_b = "key-a", "key-b"
            s_a = sign(_agg(instance_id="a", tokens_saved=5_000, certified=True), key_a)
            s_b = sign(_agg(instance_id="b", tokens_saved=8_000, certified=True), key_b)
            tampered = {
                **sign(_agg(instance_id="a", tokens_saved=5_000), key_a),
                "tokens_saved": 999,
            }

            self._write_submissions(d, [s_a, s_b, tampered])
            lb = build_leaderboard(d, {"a": key_a, "b": key_b})

            assert lb.rejected == 1
            assert len(lb.verified) == 2

    def test_rejected_count(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            key_a = "key-a"
            s_a = sign(_agg(instance_id="a", tokens_saved=5_000, certified=True), key_a)
            # tampered record for same instance
            tampered = {**s_a, "tokens_saved": 1}

            self._write_submissions(d, [s_a, tampered])
            lb = build_leaderboard(d, {"a": key_a})

            assert lb.rejected == 1
            assert len(lb.verified) == 1

    def test_only_certified_count_toward_totals(self) -> None:
        """verified contains both; totals only sums certified ones."""
        with tempfile.TemporaryDirectory() as d:
            key_a, key_b = "key-a", "key-b"
            s_cert = sign(
                _agg(
                    instance_id="a", tokens_saved=6_000, dollars_saved=0.06, runs=2, certified=True
                ),
                key_a,
            )
            s_uncert = sign(
                _agg(
                    instance_id="b", tokens_saved=4_000, dollars_saved=0.04, runs=1, certified=False
                ),
                key_b,
            )
            self._write_submissions(d, [s_cert, s_uncert])
            lb = build_leaderboard(d, {"a": key_a, "b": key_b})

            assert len(lb.verified) == 2
            assert lb.totals["tokens_saved"] == 6_000  # only certified "a"
            assert lb.totals["dollars_saved"] == pytest.approx(0.06)
            assert lb.totals["runs"] == 2
            assert lb.totals["instances"] == 2  # both instances present

    def test_sorted_by_tokens_desc(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            key_a, key_b = "key-a", "key-b"
            s_a = sign(_agg(instance_id="a", tokens_saved=1_000), key_a)
            s_b = sign(_agg(instance_id="b", tokens_saved=9_000), key_b)
            self._write_submissions(d, [s_a, s_b])
            lb = build_leaderboard(d, {"a": key_a, "b": key_b})

            assert lb.verified[0]["instance_id"] == "b"
            assert lb.verified[1]["instance_id"] == "a"

    def test_latest_per_instance_wins(self) -> None:
        """Two submissions for same instance: higher ts wins."""
        with tempfile.TemporaryDirectory() as d:
            key_a = "key-a"
            old = sign(_agg(instance_id="a", tokens_saved=1_000, ts=100.0), key_a)
            new = sign(_agg(instance_id="a", tokens_saved=5_000, ts=200.0), key_a)
            self._write_submissions(d, [old, new])
            lb = build_leaderboard(d, {"a": key_a})

            assert len(lb.verified) == 1
            assert lb.verified[0]["tokens_saved"] == 5_000

    def test_unknown_instance_key_rejected(self) -> None:
        """A submission whose instance_id has no key entry is rejected."""
        with tempfile.TemporaryDirectory() as d:
            signed = sign(_agg(instance_id="unknown"), "some-key")
            self._write_submissions(d, [signed])
            lb = build_leaderboard(d, {})  # no keys at all

            assert lb.rejected == 1
            assert len(lb.verified) == 0

    def test_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            lb = build_leaderboard(d, {})
            assert lb.verified == []
            assert lb.rejected == 0
            assert lb.totals["instances"] == 0


# ---------------------------------------------------------------------------
# render_leaderboard_html
# ---------------------------------------------------------------------------


class TestRenderLeaderboardHtml:
    def _simple_lb(self) -> Leaderboard:
        key_a = "key-a"
        signed = sign(
            _agg(instance_id="inst-xyz", tokens_saved=42_000, certified=True),
            key_a,
        )
        with tempfile.TemporaryDirectory() as d:
            submit(signed, d)
            return build_leaderboard(d, {"inst-xyz": key_a})

    def test_returns_nonempty_html(self) -> None:
        lb = self._simple_lb()
        html = render_leaderboard_html(lb)
        assert len(html) > 0
        assert "<!DOCTYPE html>" in html

    def test_contains_instance_id(self) -> None:
        lb = self._simple_lb()
        html = render_leaderboard_html(lb)
        assert "inst-xyz" in html

    def test_contains_certified_marker(self) -> None:
        lb = self._simple_lb()
        html = render_leaderboard_html(lb)
        assert "certified" in html

    def test_title_is_verifiable_savings(self) -> None:
        lb = self._simple_lb()
        html = render_leaderboard_html(lb)
        assert "Verifiable savings" in html

    def test_no_external_assets(self) -> None:
        lb = self._simple_lb()
        html = render_leaderboard_html(lb)
        # No src= or href= pointing to external URLs
        assert "http://" not in html
        assert "https://" not in html

    def test_dark_bg_color_present(self) -> None:
        lb = self._simple_lb()
        html = render_leaderboard_html(lb)
        assert "#06070b" in html

    def test_empty_leaderboard_renders(self) -> None:
        lb = Leaderboard(
            verified=[],
            totals={"tokens_saved": 0, "dollars_saved": 0.0, "runs": 0, "instances": 0},
            rejected=0,
        )
        html = render_leaderboard_html(lb)
        assert "<!DOCTYPE html>" in html
