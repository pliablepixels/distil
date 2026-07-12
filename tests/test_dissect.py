"""Tests for `distil dissect` — session picker, report math, renderers, and the
proxy/wrap logging it reads (session manifest + per-request detail records).

The proxy round-trip test mirrors tests/test_proxy.py: a stub upstream echoes
the forwarded body, the real handler runs in a thread, and the assertion is on
the *artifact* — sessions/<sid>.requests.jsonl — not on internals.
"""

from __future__ import annotations

import json
import threading
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


class _EchoHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 — http.server API
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
        {"ts": 1000.0, "model": "m-big", "stream": True, "status": 200, "booked": True,
         "mode": "digest", "compressible_tokens": 5000, "tokens_saved": 3500,
         "overhead_tokens": 700, "delta_refs": 0, "delta_tokens_saved": 0,
         "prefix_msgs": 0, "shadow_sampled": True, "expanded": False,
         "output_shaping": "",
         "blocks": [{"h": "aaaa1111", "sig": "log:l", "tokens": 2000},
                    {"h": "bbbb2222", "sig": "prose:m", "tokens": 900}]},
        {"ts": 1600.0, "model": "m-small", "stream": True, "status": 200, "booked": True,
         "mode": "digest", "compressible_tokens": 2000, "tokens_saved": 1200,
         "overhead_tokens": 500, "delta_refs": 3, "delta_tokens_saved": 800,
         "prefix_msgs": 4, "shadow_sampled": False, "expanded": True,
         "output_shaping": "",
         "blocks": [{"h": "aaaa1111", "sig": "log:l", "tokens": 2000}]},
        {"ts": 1700.0, "model": "m-small", "stream": False, "status": 529, "booked": False,
         "mode": "verbatim", "compressible_tokens": 0, "tokens_saved": 0,
         "overhead_tokens": 500, "delta_refs": 0, "delta_tokens_saved": 0,
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


class TestTolerantReader:
    def test_read_jsonl_skips_garbage(self, tmp_path: Path) -> None:
        p = tmp_path / "x.jsonl"
        p.write_text('{"a": 1}\nnot json\n[1,2]\n\n{"b": 2}\n')
        assert dz._read_jsonl(p) == [{"a": 1}, {"b": 2}]

    def test_read_jsonl_missing_file(self, tmp_path: Path) -> None:
        assert dz._read_jsonl(tmp_path / "absent.jsonl") == []
