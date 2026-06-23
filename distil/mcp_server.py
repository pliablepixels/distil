"""Minimal, zero-dependency MCP server for distil.

Exposes distil's reversible compression to any MCP client (Claude Desktop, IDEs,
agents) over stdio JSON-RPC 2.0 — **stdlib only**, no third-party SDK, so it keeps
distil's zero-runtime-deps promise.

Tools
-----
* ``distil_compress(text)`` — reversibly digest a blob; returns the digest, an
  8-hex handle, and tokens saved. The original is kept in a local on-disk store
  (never returned to the model until asked), so it costs zero tokens on the wire.
* ``distil_expand(handle)`` — return the original text for a handle.
* ``distil_savings()`` — cumulative savings from the local ledger.

Run
---
``distil mcp``  (or ``python -m distil.mcp_server``). Wire it into an MCP client's
server config as a stdio command. The protocol is newline-delimited JSON-RPC 2.0;
``handle_message`` is a pure function (testable without real stdio).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .compress.tier1 import _handle, digest
from .tokenizer import DEFAULT as _tokenizer

SERVER_NAME = "distil"
DEFAULT_PROTOCOL = "2025-06-18"


# ---------------------------------------------------------------------------
# Persistent handle store (so expand works across calls / processes)
# ---------------------------------------------------------------------------


def _store_path() -> Path:
    import os

    base = Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))
    return base / "mcp_store.json"


def _load_store() -> dict[str, str]:
    p = _store_path()
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _save_store(store: dict[str, str]) -> None:
    p = _store_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(store))
    except OSError:
        pass  # best-effort; never crash a tool call


# ---------------------------------------------------------------------------
# Tool catalog + implementations
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "distil_compress",
        "description": (
            "Reversibly compress a text blob (e.g. a large tool output) with distil. "
            "Returns a compact digest, an 8-hex handle, and the tokens saved. The "
            "original is kept locally and can be recovered with distil_expand."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "the text to compress"}},
            "required": ["text"],
        },
    },
    {
        "name": "distil_expand",
        "description": "Return the original text for a handle produced by distil_compress.",
        "inputSchema": {
            "type": "object",
            "properties": {"handle": {"type": "string", "description": "the 8-hex content handle"}},
            "required": ["handle"],
        },
    },
    {
        "name": "distil_savings",
        "description": "Report cumulative token/dollar savings from the local distil ledger.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _tool_compress(args: dict[str, Any]) -> str:
    text = args.get("text")
    if not isinstance(text, str):
        return "error: 'text' must be a string"
    digested, changed = digest(text)
    if not changed:
        return json.dumps({"compressed": text, "handle": None, "tokens_saved": 0})
    h = _handle(text)
    store = _load_store()
    store[h] = text
    _save_store(store)
    saved = max(0, _tokenizer.count(text) - _tokenizer.count(digested))
    return json.dumps({"compressed": digested, "handle": h, "tokens_saved": saved})


def _tool_expand(args: dict[str, Any]) -> str:
    handle = args.get("handle")
    if not isinstance(handle, str):
        return "error: 'handle' must be a string"
    original = _load_store().get(handle)
    if original is None:
        return f"error: no original found for handle {handle!r}"
    return original


def _tool_savings(_args: dict[str, Any]) -> str:
    from . import ledger

    s = ledger.summary()
    return json.dumps(
        {
            "runs": s.runs,
            "tokens_saved": s.total_tokens_saved,
            "dollars_saved": round(s.total_dollars_saved, 6),
        }
    )


_DISPATCH = {
    "distil_compress": _tool_compress,
    "distil_expand": _tool_expand,
    "distil_savings": _tool_savings,
}


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 message handling (pure — unit-testable)
# ---------------------------------------------------------------------------


def _result(msg_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def handle_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one JSON-RPC message; return a response dict, or None for notifications."""
    method = msg.get("method")
    msg_id = msg.get("id")

    # Notifications (no id) get no response.
    if method is not None and msg_id is None:
        return None

    if method == "initialize":
        requested = (msg.get("params") or {}).get("protocolVersion")
        return _result(
            msg_id,
            {
                "protocolVersion": requested or DEFAULT_PROTOCOL,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": _server_version()},
            },
        )

    if method == "tools/list":
        return _result(msg_id, {"tools": TOOLS})

    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        fn = _DISPATCH.get(name)
        if fn is None:
            return _error(msg_id, -32602, f"unknown tool: {name!r}")
        try:
            text = fn(args)
            is_error = isinstance(text, str) and text.startswith("error:")
            return _result(
                msg_id, {"content": [{"type": "text", "text": text}], "isError": is_error}
            )
        except Exception as exc:  # noqa: BLE001 — surface as a tool error, never crash the server
            return _result(
                msg_id,
                {"content": [{"type": "text", "text": f"error: {exc}"}], "isError": True},
            )

    if method == "ping":
        return _result(msg_id, {})

    return _error(msg_id, -32601, f"method not found: {method!r}")


def _server_version() -> str:
    from . import __version__

    return __version__


# ---------------------------------------------------------------------------
# stdio transport
# ---------------------------------------------------------------------------


def serve(stdin: Any = None, stdout: Any = None) -> None:
    """Run the newline-delimited JSON-RPC 2.0 stdio loop until EOF."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue  # skip unparseable input rather than crash
        response = handle_message(msg)
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


if __name__ == "__main__":  # pragma: no cover
    serve()
