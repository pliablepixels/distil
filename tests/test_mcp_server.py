"""Zero-dependency MCP server — JSON-RPC handling, compress/expand round-trip."""

from __future__ import annotations

import io
import json

import pytest

from distil import mcp_server as mcp

BIG = "\n".join(f"line {i}: some content value_{i}" for i in range(40))


@pytest.fixture(autouse=True)
def _isolate_store(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))


def test_initialize_echoes_protocol_and_serverinfo():
    resp = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
    )
    assert resp["id"] == 1
    assert resp["result"]["protocolVersion"] == "2025-06-18"
    assert resp["result"]["serverInfo"]["name"] == "distil"
    assert "tools" in resp["result"]["capabilities"]


def test_tools_list_has_three_tools():
    resp = mcp.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in resp["result"]["tools"]}
    assert names == {"distil_compress", "distil_expand", "distil_savings"}


def test_compress_then_expand_round_trip():
    c = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "distil_compress", "arguments": {"text": BIG}},
        }
    )
    assert c["result"]["isError"] is False
    payload = json.loads(c["result"]["content"][0]["text"])
    assert payload["handle"] and payload["tokens_saved"] > 0
    assert len(payload["compressed"]) < len(BIG)

    e = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "distil_expand", "arguments": {"handle": payload["handle"]}},
        }
    )
    assert e["result"]["content"][0]["text"] == BIG  # byte-exact recovery


def test_expand_unknown_handle_is_error():
    e = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "distil_expand", "arguments": {"handle": "deadbeef"}},
        }
    )
    assert e["result"]["isError"] is True


def test_unknown_tool_is_jsonrpc_error():
    r = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        }
    )
    assert r["error"]["code"] == -32602


def test_notification_returns_none():
    assert mcp.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_unknown_method_is_error():
    r = mcp.handle_message({"jsonrpc": "2.0", "id": 7, "method": "bogus"})
    assert r["error"]["code"] == -32601


def test_serve_loop_over_stdio():
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    stdin = io.StringIO("\n".join(json.dumps(m) for m in msgs) + "\n")
    stdout = io.StringIO()
    mcp.serve(stdin=stdin, stdout=stdout)
    out_lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    # initialize + tools/list answered; the notification produced no line.
    assert [o["id"] for o in out_lines] == [1, 2]
