"""Tests for `distil dissect` — session picker, report math, renderers, and the
proxy/wrap logging it reads (session manifest + per-request detail records).

The proxy round-trip test mirrors tests/test_proxy.py: a stub upstream echoes
the forwarded body, the real handler runs in a thread, and the assertion is on
the *artifact* — sessions/<sid>.requests.jsonl — not on internals.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from distil import dissect as dz
from distil.cli import main
from distil.ledger import (
    append_session_request,
    session_manifest_path,
    session_requests_path,
    write_session_manifest,
)
from distil.proxy import build_handler, wrap_run

_LOG_LINES = "\n".join(
    f"[2026-07-11 12:00:{i:02d}] INFO worker-{i}: heartbeat ok, queue depth {i * 3}"
    for i in range(60)
)


_UPSTREAM_JSON = json.dumps(
    {
        "id": "msg_test",
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 4321, "output_tokens": 87},
    }
).encode()

_UPSTREAM_SSE = (
    b'event: message_start\ndata: {"type":"message_start","message":'
    b'{"usage":{"input_tokens":1234,"output_tokens":1}}}\n\n'
    b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":55}}\n\n'
    b"event: message_stop\ndata: {}\n\n"
)


class _EchoHandler(BaseHTTPRequestHandler):
    """Stub upstream: SSE when the request asked to stream, JSON otherwise —
    both carrying known usage figures so capture can be asserted exactly."""

    def do_POST(self) -> None:  # noqa: N802 — http.server API
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        streaming = b'"stream": true' in body or b'"stream":true' in body
        payload = _UPSTREAM_SSE if streaming else _UPSTREAM_JSON
        self.send_response(200)
        self.send_header(
            "Content-Type", "text/event-stream" if streaming else "application/json"
        )
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:  # quiet
        pass


@pytest.fixture()
def servers() -> Any:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    handler_cls = build_handler(f"http://127.0.0.1:{upstream.server_address[1]}")
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    yield proxy.server_address[1]
    proxy.shutdown()
    upstream.shutdown()


def _post(port: int, payload: dict[str, Any]) -> Any:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req)


def _compressible_payload() -> dict[str, Any]:
    return {
        "model": "claude-test-1",
        "max_tokens": 128,
        "system": "You are a test agent." * 20,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": _LOG_LINES}
                ],
            },
            {"role": "user", "content": "next"},
            {"role": "user", "content": "next again"},
        ],
    }


# --------------------------------------------------------------- proxy logging
class TestRequestDetailLogging:
    def test_detail_record_written_per_request(
        self, servers: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.setenv("DISTIL_SESSION", "s100-1")
        with _post(servers, _compressible_payload()) as resp:
            assert resp.status == 200
        path = session_requests_path("s100-1")
        assert path is not None and path.exists()
        recs = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(recs) == 1
        rec = recs[0]
        assert rec["model"] == "claude-test-1"
        assert rec["mode"] == "digest"
        assert rec["stream"] is False
        assert rec["status"] == 200
        assert rec["booked"] is False  # no SavingsTracker wired in this harness
        assert rec["compressible_tokens"] > 0
        assert rec["tokens_saved"] > 0
        assert rec["overhead_tokens"] > 0  # the system prompt is counted, not compressed
        assert rec["blocks"], "digested blocks should be inventoried"
        blk = rec["blocks"][0]
        assert set(blk) == {"h", "sig", "tokens"} and len(blk["h"]) == 8
        assert ":" in blk["sig"] and blk["tokens"] > 0
        assert "handle=" not in json.dumps(rec), "detail records must stay content-free"
        assert rec["client_stream"] is False
        assert rec["duration_ms"] is not None and rec["duration_ms"] >= 0
        assert rec["usage_input_tokens"] == 4321  # billed usage from the JSON response
        assert rec["usage_output_tokens"] == 87
        assert rec["expanded_handles"] == []

    def test_streamed_request_captures_sse_usage(
        self, servers: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.setenv("DISTIL_SESSION", "s101-1")
        payload = _compressible_payload() | {"stream": True}
        with _post(servers, payload) as resp:
            assert resp.status == 200
            resp.read()
        path = session_requests_path("s101-1")
        assert path is not None
        # The detail record lands after the last relayed byte — poll briefly.
        for _ in range(100):
            if path.exists():
                break
            time.sleep(0.02)
        assert path.exists()
        rec = json.loads(path.read_text().splitlines()[0])
        assert rec["stream"] is True and rec["client_stream"] is True
        assert rec["usage_input_tokens"] == 1234  # from SSE message_start
        assert rec["usage_output_tokens"] == 55  # last (cumulative) message_delta
        assert rec["duration_ms"] is not None

    def test_no_session_no_record(
        self, servers: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        with _post(servers, _compressible_payload()) as resp:
            assert resp.status == 200
        assert not list((tmp_path / "sessions").glob("*.requests.jsonl"))


class TestWrapManifest:
    def test_manifest_written_at_wrap_start(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        assert wrap_run(["true"], expand=True, session_delta=True, shadow_rate=0.25) == 0
        manifests = list((tmp_path / "sessions").glob("s*.json"))
        assert len(manifests) == 1
        man = json.loads(manifests[0].read_text())
        assert man["tool"] == "true" and man["argv"] == ["true"]
        assert man["flags"]["expand"] is True
        assert man["flags"]["session_delta"] is True
        assert man["flags"]["shadow_rate"] == 0.25
        assert man["billing"] in ("subscription", "metered", "unknown")
        assert man["started_ts"] > 0 and man["sid"] == manifests[0].stem

    def test_writers_are_fail_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        # No session -> no path -> both writers are silent no-ops.
        assert session_manifest_path() is None
        write_session_manifest({"sid": "x"})
        append_session_request({"ts": 1})


# ----------------------------------------------------------------- report math
def _seed_state(home: Path) -> None:
    """Two sessions: sA (manifest + detail + restore blobs), sB (ledger only)."""
    row = {
        "trajectory_id": "live-proxy",
        "tokenizer": "heuristic",
        "mode": "digest",
        "acct": 2,
    }
    ledger = [
        {**row, "session": "s200-1", "model": "m-big", "turns": 2, "ts": 1000.0,
         "baseline_input_tokens": 9000, "distil_input_tokens": 3000,
         "baseline_dollars": 0.9, "distil_dollars": 0.3},
        {**row, "session": "s200-1", "model": "m-small", "turns": 1, "ts": 1600.0,
         "baseline_input_tokens": 1000, "distil_input_tokens": 900,
         "baseline_dollars": 0.1, "distil_dollars": 0.09},
        {**row, "session": "s300-9", "model": "m-big", "turns": 1, "ts": 2000.0,
         "baseline_input_tokens": 500, "distil_input_tokens": 400,
         "baseline_dollars": 0.05, "distil_dollars": 0.04},
        {"corrupt": "row missing session"},
    ]
    (home / "savings.jsonl").write_text(
        "\n".join(json.dumps(r) for r in ledger) + "\nnot json\n"
    )
    sess = home / "sessions"
    sess.mkdir()
    (sess / "s200-1.json").write_text(json.dumps({
        "sid": "s200-1", "tool": "codex", "argv": ["codex", "--full-auto"],
        "started_ts": 990.0, "distil_version": "1.15.0", "billing": "subscription",
        "flags": {"expand": True, "session_delta": True, "shadow_rate": 0.1,
                  "lossless_only": False, "verbatim": False, "shape_output": "off",
                  "upstream": "https://api.anthropic.com", "env_var": "ANTHROPIC_BASE_URL"},
    }))
    details = [
        {"ts": 1000.0, "model": "m-big", "stream": True, "client_stream": True,
         "status": 200, "booked": True, "duration_ms": 4000,
         "usage_input_tokens": 2100, "usage_output_tokens": 300, "expanded_handles": [],
         "mode": "digest", "compressible_tokens": 5000, "tokens_saved": 3500,
         "overhead_tokens": 700, "system_tokens": 100, "tools_tokens": 600,
         "tools": [{"name": "bash", "tokens": 300},
                   {"name": "mcp__gmail__search", "tokens": 300}],
         "delta_refs": 0, "delta_tokens_saved": 0,
         "prefix_msgs": 0, "shadow_sampled": True, "expanded": False,
         "output_shaping": "",
         "blocks": [{"h": "aaaa1111", "sig": "log:l", "tokens": 2000},
                    {"h": "bbbb2222", "sig": "prose:m", "tokens": 900}]},
        {"ts": 1600.0, "model": "m-small", "stream": False, "client_stream": True,
         "status": 200, "booked": True, "duration_ms": 9000,
         "usage_input_tokens": 1200, "usage_output_tokens": 150,
         "expanded_handles": ["aaaa1111"],
         "mode": "digest", "compressible_tokens": 2000, "tokens_saved": 1200,
         "overhead_tokens": 500, "system_tokens": 200, "tools_tokens": 300,
         "tools": [{"name": "bash", "tokens": 300}],
         "delta_refs": 3, "delta_tokens_saved": 800,
         "prefix_msgs": 4, "shadow_sampled": False, "expanded": True,
         "output_shaping": "",
         "blocks": [{"h": "aaaa1111", "sig": "log:l", "tokens": 2000}]},
        {"ts": 1700.0, "model": "m-small", "stream": False, "client_stream": False,
         "status": 529, "booked": False, "duration_ms": 500,
         "usage_input_tokens": None, "usage_output_tokens": None, "expanded_handles": [],
         "mode": "verbatim", "compressible_tokens": 0, "tokens_saved": 0,
         "overhead_tokens": 500, "system_tokens": 250, "tools_tokens": 250,
         "tools": [{"name": "bash", "tokens": 250}],
         "delta_refs": 0, "delta_tokens_saved": 0,
         "prefix_msgs": 0, "shadow_sampled": False, "expanded": False,
         "output_shaping": "", "blocks": []},
    ]
    (sess / "s200-1.requests.jsonl").write_text(
        "\n".join(json.dumps(r) for r in details) + "\n"
    )
    (sess / "s200-1").write_text("1")
    (sess / "s300-9.exit").write_text("rc=0")
    restore = home / "restore"
    restore.mkdir()
    (restore / "aaaa1111").write_text("original log content")
    (home / "shadow.jsonl").write_text(
        json.dumps({"ts": 1500.0, "equivalent": True}) + "\n"
        + json.dumps({"ts": 999999.0, "equivalent": False}) + "\n"
    )


class TestDissection:
    @pytest.fixture(autouse=True)
    def _home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        _seed_state(tmp_path)

    def test_list_sessions_newest_first_with_status(self) -> None:
        sessions = dz.list_sessions()
        assert [o.sid for o in sessions] == ["s300-9", "s200-1"]
        s200 = sessions[1]
        assert s200.tool == "codex" and s200.requests == 3
        assert s200.baseline_tokens == 10000 and s200.distil_tokens == 3900
        assert s200.status == "live"
        assert sessions[0].status == "exited"

    def test_resolve_sid(self) -> None:
        assert dz.resolve_sid("latest") == "s300-9"
        assert dz.resolve_sid("s200") == "s200-1"
        assert dz.resolve_sid("s200-1") == "s200-1"
        assert dz.resolve_sid("s") is None  # ambiguous
        assert dz.resolve_sid("nope") is None

    def test_dissection_math(self) -> None:
        d = dz.dissect("s200-1")
        assert d.baseline_tokens == 10000 and d.distil_tokens == 3900
        assert d.pct_saved == pytest.approx(61.0)
        assert d.dollars_saved == pytest.approx(0.61)
        assert d.per_model()[0] == ("m-big", 2, 9000, 3000)
        assert d.detail_available
        assert d.delta_tokens_saved == 800
        assert d.overhead_tokens_avg == 566
        assert d.verbatim_requests == 1 and d.unbooked_requests == 1
        assert d.shadow_sampled == 1 and d.expand_resolved == 1
        assert d.billing == "subscription"
        # Blocks dedup by handle across requests; folds counted per sighting.
        assert set(d.blocks) == {"aaaa1111", "bbbb2222"}
        assert d.blocks["aaaa1111"]["folds"] == 2
        assert d.blocks["aaaa1111"]["recoverable"] is True
        assert d.blocks["bbbb2222"]["recoverable"] is False
        assert d.blocks_by_kind()[0] == ("log:l", 1, 2000)
        # Shadow join is by time window: only the ts=1500 row is inside.
        assert d.shadow_window_rows == 1 and d.shadow_window_agree == 1

    def test_render_text_full(self) -> None:
        d = dz.dissect("s200-1")
        text = dz.render_text(d, color=False)
        assert "s200-1" in text and "codex" in text
        assert "61.0% saved" in text and "notional" in text
        assert "log:l" in text and "aaaa1111" in text
        assert "1/2 blocks still in restore/" in text
        assert "expand: 1 requests" in text and "shadow: 1 requests" in text
        assert "1 verbatim" in text

    def test_render_text_degrades_without_detail(self) -> None:
        d = dz.dissect("s300-9")
        text = dz.render_text(d, color=False)
        assert "manifest not recorded" in text
        assert "not recorded — per-request detail" in text
        assert "rc=0" in text

    def test_to_json_schema(self) -> None:
        payload = dz.to_json(dz.dissect("s200-1"))
        assert payload["session"] == "s200-1"
        assert payload["savings"]["pct_saved"] == 61.0
        assert payload["savings"]["dollars_notional"] is True
        assert payload["requests"]["total"] == 3
        assert payload["blocks"]["unique"] == 2 and payload["blocks"]["recoverable"] == 1
        assert payload["quality"]["shadow_sampled_requests"] == 1

    def test_render_html_self_contained(self) -> None:
        page = dz.render_html(dz.dissect("s200-1"))
        assert page.startswith("<!doctype html>")
        assert "s200-1" in page and "61.0%" in page and "notional" in page
        assert "aaaa1111" in page and "log:l" in page


class TestCli:
    @pytest.fixture(autouse=True)
    def _home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        _seed_state(tmp_path)

    def test_picker_lists_sessions(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["dissect"]) == 0
        out = capsys.readouterr().out
        assert "s200-1" in out and "s300-9" in out and "codex" in out

    def test_picker_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["dissect", "--json"]) == 0
        rows = json.loads(capsys.readouterr().out)
        assert {r["session"] for r in rows} == {"s200-1", "s300-9"}

    def test_report_and_prefix_resolution(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["dissect", "s200", "--no-color"]) == 0
        assert "61.0% saved" in capsys.readouterr().out

    def test_json_report(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["dissect", "latest", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["session"] == "s300-9"

    def test_html_export(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        out_file = tmp_path / "dissect.html"
        assert main(["dissect", "s200-1", "--html", str(out_file)]) == 0
        assert out_file.read_text().startswith("<!doctype html>")

    def test_unknown_session_exits_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["dissect", "zzz"]) == 2
        assert "no session matches" in capsys.readouterr().out


class TestInsights:
    @pytest.fixture(autouse=True)
    def _home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        _seed_state(tmp_path)

    def test_mechanism_decomposition(self) -> None:
        d = dz.dissect("s200-1")
        assert d.tokens_saved_total == 4700
        assert d.delta_tokens_saved == 800 and d.digest_saved == 3900

    def test_overhead_tax(self) -> None:
        d = dz.dissect("s200-1")
        assert d.overhead_tokens_total == 1700
        # 1700 overhead vs 7000 compressible -> 1700/8700
        assert d.overhead_share == pytest.approx(19.54, abs=0.01)

    def test_churn(self) -> None:
        d = dz.dissect("s200-1")
        # aaaa1111 folded twice -> 2000 tokens re-digested across 1 re-folded block
        assert d.churn_tokens == 2000 and d.churned_blocks == 1

    def test_usage_and_calibration(self) -> None:
        d = dz.dissect("s200-1")
        assert d.usage_input_total == 3300 and d.usage_output_total == 450
        assert d.usage_requests == 2
        est, billed = d.calibration()
        # r1: 700 + (5000-3500) = 2200; r2: 500 + (2000-1200) = 1300
        assert est == 3500 and billed == 3300

    def test_headroom_multiplier(self) -> None:
        d = dz.dissect("s200-1")
        assert d.headroom_multiplier == pytest.approx(10000 / 3900)

    def test_latency_by_path_and_forced_buffering(self) -> None:
        d = dz.dissect("s200-1")
        assert d.forced_buffered == 1
        lat = dict((k, (n, ms)) for k, n, ms in d.latency_by_path())
        assert lat["streamed"] == (1, 4000)
        assert lat["buffered (forced by expand)"] == (1, 9000)
        assert lat["buffered"] == (1, 500)

    def test_expansion_regret(self) -> None:
        d = dz.dissect("s200-1")
        assert d.expansion_regret() == [("log:l", 1, 1)]

    def test_no_anomalies_on_healthy_session(self) -> None:
        d = dz.dissect("s200-1")
        assert d.anomalies(dz.list_sessions()) == []

    def test_report_renders_insights(self) -> None:
        d = dz.dissect("s200-1")
        text = dz.render_text(d, color=False, peers=dz.list_sessions())
        assert "digest folds 3.90k (83%)" in text
        assert "20% of everything sent" in text
        assert "2.00k tokens re-digested" in text
        assert "heuristic estimate 3.50k vs billed 3.30k" in text
        assert "~2.6x" in text  # flat-rate headroom
        assert "buffered (forced by expand) 1 req @ 9.0s" in text
        assert "regret: log:l blocks pulled back 1/1" in text
        page = dz.render_html(d, peers=dz.list_sessions())
        assert "Digest folds" in page and "83% of savings" in page
        assert ">2.6\u00d7<" in page  # headroom tile
        payload = dz.to_json(d, dz.list_sessions())
        assert payload["insights"]["churn"]["tokens"] == 2000
        assert payload["insights"]["usage"]["calibration"] == {
            "estimated": 3500,
            "billed": 3300,
        }
        assert payload["insights"]["anomalies"] == []


class TestToolsAndCharts:
    @pytest.fixture(autouse=True)
    def _home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        _seed_state(tmp_path)

    def test_tool_costs_aggregation(self) -> None:
        d = dz.dissect("s200-1")
        # bash: max 300/req, seen on 3 requests -> 900; gmail: 300 on 1 -> 300.
        assert d.tool_costs() == [("bash", 300, 900), ("mcp__gmail__search", 300, 300)]
        assert d.system_tokens_avg == 183 and d.tools_tokens_avg == 383
        assert d.system_growth() == (100, 250)

    def test_text_report_shows_tools_and_prompt_growth(self) -> None:
        text = dz.render_text(dz.dissect("s200-1"), color=False)
        assert "tool definitions (2): bash 300/req, mcp__gmail__search 300/req" in text
        assert "system prompt: 100 → 250 tokens over the session" in text

    def test_html_has_charts(self) -> None:
        page = dz.render_html(dz.dissect("s200-1"))
        assert page.count("<svg") == 3  # timeline, tools, block kinds
        # Validated categorical series, fixed order: overhead / sent / saved.
        assert "#3987e5" in page and "#199e70" in page and "#c98500" in page
        assert "overhead (system + tools)" in page  # legend, not color-alone
        assert "Tool definitions" in page and "bash" in page
        assert "request 2 · m-small · overhead 500 · sent 800 · saved 1,200" in page
        assert "data table" in page  # accessible table view of the timeline
        assert "Request composition" in page

    def test_json_has_tool_breakdown(self) -> None:
        payload = dz.to_json(dz.dissect("s200-1"))
        overhead = payload["insights"]["overhead"]
        assert overhead["tools"][0] == {
            "name": "bash",
            "tokens_per_request": 300,
            "session_tokens": 900,
        }
        assert overhead["system_growth"] == (100, 250)

    def test_charts_absent_without_detail(self) -> None:
        page = dz.render_html(dz.dissect("s300-9"))
        assert "<svg" not in page


class TestInteractivePicker:
    @pytest.fixture(autouse=True)
    def _home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        monkeypatch.delenv("DISTIL_SESSION", raising=False)
        monkeypatch.delenv("NO_COLOR", raising=False)
        _seed_state(tmp_path)

    def _tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

    def test_numbered_pick_renders_report(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._tty(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda prompt="": "2")
        assert main(["dissect"]) == 0
        out = capsys.readouterr().out
        assert "#" in out and "s200-1" in out  # numbered list first
        assert "savings (input tokens, booked 2xx only)" in out  # then the report
        assert "codex" in out

    def test_enter_quits(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._tty(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        assert main(["dissect"]) == 0
        assert "savings (input tokens" not in capsys.readouterr().out

    def test_bad_pick_exits_2(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._tty(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda prompt="": "zzz")
        assert main(["dissect"]) == 2

    def test_non_tty_keeps_list_only(self, capsys: pytest.CaptureFixture[str]) -> None:
        # capsys stdout is not a tty -> no prompt, list only (the old behavior).
        assert main(["dissect"]) == 0
        out = capsys.readouterr().out
        assert "pick one to dissect" in out
        assert "savings (input tokens" not in out


class TestAnomalies:
    def _session(self, home: Path, requests: list[dict[str, Any]], **manifest: Any) -> None:
        sess = home / "sessions"
        sess.mkdir(exist_ok=True)
        man = {
            "sid": "s900-1", "tool": "claude", "argv": ["claude"], "started_ts": 1.0,
            "distil_version": "1.15.0", "billing": "subscription",
            "flags": {"expand": False, "session_delta": False, "shadow_rate": 0.0,
                      "lossless_only": False},
        }
        man["flags"].update(manifest.pop("flags", {}))
        man.update(manifest)
        (sess / "s900-1.json").write_text(json.dumps(man))
        (sess / "s900-1.requests.jsonl").write_text(
            "\n".join(json.dumps(r) for r in requests) + "\n"
        )

    @staticmethod
    def _req(**over: Any) -> dict[str, Any]:
        base = {
            "ts": 10.0, "model": "m", "stream": True, "client_stream": True,
            "status": 200, "booked": True, "duration_ms": 100,
            "usage_input_tokens": None, "usage_output_tokens": None,
            "expanded_handles": [], "mode": "digest", "compressible_tokens": 100,
            "tokens_saved": 50, "overhead_tokens": 10, "delta_refs": 0,
            "delta_tokens_saved": 0, "prefix_msgs": 0, "shadow_sampled": False,
            "expanded": False, "output_shaping": "",
            "blocks": [{"h": "cccc3333", "sig": "log:m", "tokens": 40}],
        }
        base.update(over)
        return base

    def test_silent_shadow_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        self._session(tmp_path, [self._req() for _ in range(12)], flags={"shadow_rate": 0.5})
        warnings = dz.dissect("s900-1").anomalies()
        assert any("shadow may be silently failing" in w for w in warnings)

    def test_expand_never_intercepted_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        # expand on, folds happening, yet every request streamed straight through.
        self._session(tmp_path, [self._req() for _ in range(12)], flags={"expand": True})
        warnings = dz.dissect("s900-1").anomalies()
        assert any("could never be intercepted" in w for w in warnings)

    def test_unbooked_spike_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        reqs = [self._req() for _ in range(6)] + [
            self._req(booked=False, status=529) for _ in range(4)
        ]
        self._session(tmp_path, reqs)
        warnings = dz.dissect("s900-1").anomalies()
        assert any("not booked" in w for w in warnings)

    def test_calibration_drift_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
        # estimate 10+ (100-50) = 60 vs billed 20 -> ratio 3.0
        self._session(tmp_path, [self._req(usage_input_tokens=20)])
        warnings = dz.dissect("s900-1").anomalies()
        assert any("off by >50% vs billed" in w for w in warnings)


class TestScanUsage:
    def test_json_body(self) -> None:
        from distil.streamrelay import scan_usage

        assert scan_usage(_UPSTREAM_JSON) == {"input_tokens": 4321, "output_tokens": 87}

    def test_sse_body_takes_last_output(self) -> None:
        from distil.streamrelay import scan_usage

        assert scan_usage(_UPSTREAM_SSE) == {"input_tokens": 1234, "output_tokens": 55}

    def test_no_usage(self) -> None:
        from distil.streamrelay import scan_usage

        assert scan_usage(b'{"error": "overloaded"}') == {}


class TestTolerantReader:
    def test_read_jsonl_skips_garbage(self, tmp_path: Path) -> None:
        p = tmp_path / "x.jsonl"
        p.write_text('{"a": 1}\nnot json\n[1,2]\n\n{"b": 2}\n')
        assert dz._read_jsonl(p) == [{"a": 1}, {"b": 2}]

    def test_read_jsonl_missing_file(self, tmp_path: Path) -> None:
        assert dz._read_jsonl(tmp_path / "absent.jsonl") == []
