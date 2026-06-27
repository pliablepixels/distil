#!/usr/bin/env python3
"""OpenAI Chat-Completions compression proxy for the SWE-bench E7 eval.

A thin OpenAI-compatible proxy that sits between the coding agent (aider) and a
local model server (Ollama, vLLM, LM Studio) serving ``/v1/chat/completions``.
Mirrors :mod:`benchmarks.swe_bench_e2e.compress_proxy` (the Anthropic-Messages
variant) in structure and eval conditions — the only differences are the message
format (OpenAI vs Anthropic) and the upstream target (local vs ``api.anthropic.com``).

Four eval conditions
--------------------
* **full**          — transparent pass-through; all accounting still runs.
* **distil_trunc500** — head-truncate each compressible block to 500 chars.
* **llmlingua2**    — LLMLingua-2 at default keep-rate.
* **distil_expand** — digest blocks behind content handles; expose a
  ``distil_expand`` function tool; transparent recover-then-redecide loop inside
  the proxy (the model never sees the recovery detail).

What counts as a *compressible* OpenAI block
---------------------------------------------
OpenAI ``messages`` elements have ``role ∈ {system, user, assistant, tool}`` and
``content`` that is either a plain string or a list of parts.  We compress ONLY:

* messages whose ``role`` is ``"user"`` or ``"tool"``  (same rule as Anthropic);
* ``content`` parts whose ``type`` is ``"text"`` (string or ``{"type":"text","text":…}``);
* blocks of at least :data:`~compress_proxy.MIN_CHARS` characters.

System and assistant content is never touched (same rationale as the Anthropic proxy).

Upstream
--------
Default: ``OPENAI_BASE_URL`` env var, falling back to ``http://127.0.0.1:11434/v1``
(Ollama's default).  Override at runtime via ``serve(upstream=…)`` or ``--upstream``.
Non-streaming only — if a request arrives with ``"stream": true`` the proxy silently
forces it to ``false`` before forwarding (the harness already sets ``stream=false``).
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import sys as _sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

# Re-use everything from the Anthropic proxy — no duplication.
from benchmarks.swe_bench_e2e.compress_proxy import (  # noqa: E402
    COMPRESSORS,
    DISTIL_EXPAND_TOOL,
    EXPAND_CONDITION,
    MAX_EXPAND_ITERS,
    MIN_CHARS,
    CompressStats,
    Compressor,
    _handle,
    digest_block,
)
from distil.tokenizer import DEFAULT as _tokenizer  # noqa: E402

_DEFAULT_UPSTREAM = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")

# The OpenAI function-tool equivalent of the Anthropic DISTIL_EXPAND_TOOL.
_DISTIL_EXPAND_FUNCTION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": DISTIL_EXPAND_TOOL["name"],
        "description": DISTIL_EXPAND_TOOL["description"],
        "parameters": DISTIL_EXPAND_TOOL["input_schema"],
    },
}


# --------------------------------------------------------------------------- #
# Block selection + rewriting (OpenAI message format)
# --------------------------------------------------------------------------- #


def _oai_compressible(role: str, part: Any) -> bool:
    """True iff this OpenAI content part is agent-read context we may compress."""
    if role not in ("user", "tool"):  # never system, never assistant reasoning
        return False
    if isinstance(part, str):
        return len(part) >= MIN_CHARS
    if isinstance(part, dict) and part.get("type") == "text":
        text = part.get("text", "")
        return isinstance(text, str) and len(text) >= MIN_CHARS
    return False


def _part_text(part: Any) -> str | None:
    """Extract the text string from a content part, or None if not a text part."""
    if isinstance(part, str):
        return part
    if isinstance(part, dict) and part.get("type") == "text":
        return part.get("text")
    return None


def compress_body_openai(
    body: dict[str, Any],
    compressor: Compressor | None,
    stats: CompressStats,
    protect: str | None = None,
    digest_restore: dict[str, str] | None = None,
    gate_recent: int | None = None,
    expanded: set[str] | None = None,
) -> dict[str, Any]:
    """Return a new OpenAI chat-completions request body with compressible blocks rewritten.

    Mirrors :func:`compress_proxy.compress_body` for the OpenAI message format.
    ``compressor=None`` (condition *full*) is a transparent pass-through that still
    tallies block/token stats so all conditions report comparable context sizes.

    ``digest_restore`` selects the **reversible** tier (condition ``distil_expand``):
    each compressible block is digested with :func:`~compress_proxy.digest_block` and
    ``handle→original`` is recorded in this dict so the proxy's recovery loop can
    restore it on demand.  When set it takes precedence over ``compressor``.

    ``protect`` is a substring (the SWE-bench problem statement) that must never be
    compressed — see :func:`compress_proxy.compress_body` for full rationale.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    stats.requests += 1
    protect = protect or None

    # Relevance gate (condition distil_gated): keep the last `gate_recent` user/tool
    # messages — the working set — FULL; digest only older periphery. Mirrors
    # compress_proxy.compress_body. Applies only in reversible (digest_restore) mode.
    keep_full: set[int] | None = None
    if gate_recent is not None and digest_restore is not None:
        ut = [
            i
            for i, m in enumerate(messages)
            if isinstance(m, dict) and m.get("role") in ("user", "tool")
        ]
        keep_full = set(ut[-gate_recent:]) if gate_recent > 0 else set()

    def rewrite_text(role: str, text: str, gated_keep: bool = False) -> str:
        stats.blocks_seen += 1
        before = len(text)
        stats.chars_before += before
        stats.tokens_before += _tokenizer.count(text)
        is_protected = protect is not None and protect in text
        if is_protected:
            stats.blocks_protected += 1
        inactive = compressor is None and digest_restore is None
        # Sticky expansion: a block the agent already recovered stays FULL on later turns.
        sticky = False
        if digest_restore is not None and expanded is not None and before > MIN_CHARS:
            if _handle(text) in expanded:
                digest_restore[_handle(text)] = text
                sticky = True
        if (
            inactive
            or gated_keep
            or sticky
            or before < MIN_CHARS
            or role not in ("user", "tool")
            or is_protected
        ):
            stats.chars_after += before
            stats.tokens_after += _tokenizer.count(text)
            return text
        out = digest_block(text, digest_restore) if digest_restore is not None else compressor(text)
        stats.blocks_compressed += 1
        stats.chars_after += len(out)
        stats.tokens_after += _tokenizer.count(out)
        return out

    new_messages: list[Any] = []
    for mi, msg in enumerate(messages):
        if not isinstance(msg, dict):
            new_messages.append(msg)
            continue
        gk = keep_full is not None and mi in keep_full
        role = msg.get("role", "")
        content = msg.get("content")

        if isinstance(content, str):
            # Plain-string content (common for user messages)
            if _oai_compressible(role, content):
                new_messages.append({**msg, "content": rewrite_text(role, content, gk)})
            else:
                new_messages.append(msg)
            continue

        if not isinstance(content, list):
            new_messages.append(msg)
            continue

        # List-of-parts content
        new_parts: list[Any] = []
        for part in content:
            txt = _part_text(part)
            if txt is not None and _oai_compressible(role, part):
                new_txt = rewrite_text(role, txt, gk)
                if isinstance(part, str):
                    new_parts.append(new_txt)
                else:
                    new_parts.append({**part, "text": new_txt})
            else:
                new_parts.append(part)
        new_messages.append({**msg, "content": new_parts})

    return {**body, "messages": new_messages}


# --------------------------------------------------------------------------- #
# The proxy server
# --------------------------------------------------------------------------- #


@dataclass
class ProxyState:
    compressor: Compressor | None
    upstream: str
    stats: CompressStats = field(default_factory=CompressStats)
    capture_path: Path | None = None
    accounting_path: Path | None = None
    protect: str | None = None  # problem statement: never compressed
    expand: bool = False  # distil reversible tier: digest + distil_expand recovery loop
    gate_recent: int | None = None  # distil_gated: keep last N user/tool msgs full
    expanded: set[str] = field(default_factory=set)  # handles recovered this session (sticky)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _make_handler(state: ProxyState):  # noqa: C901
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence default stderr spam
            pass

        def _upstream_headers(self) -> dict[str, str]:
            out: dict[str, str] = {}
            for k, v in self.headers.items():
                if k.lower() in ("host", "content-length", "connection", "accept-encoding"):
                    continue
                out[k] = v
            return out

        def _forward(self, path: str, body_bytes: bytes, hdrs: dict[str, str]):
            """POST to upstream; return (status, [(hdr,val)…], payload). Accounts usage."""
            url = state.upstream.rstrip("/") + path
            req = urllib.request.Request(url, data=body_bytes, method="POST")
            for k, v in hdrs.items():
                req.add_header(k, v)
            req.add_header("Content-Length", str(len(body_bytes)))
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    payload = resp.read()
                    self._account(payload)
                    items = [
                        (k, v)
                        for k, v in resp.headers.items()
                        if k.lower()
                        not in (
                            "transfer-encoding",
                            "connection",
                            "content-length",
                            "content-encoding",
                        )
                    ]
                    return resp.status, items, payload
            except urllib.error.HTTPError as e:
                return (
                    e.code,
                    [("Content-Type", e.headers.get("Content-Type", "application/json"))],
                    e.read(),
                )

        def _send(self, status: int, header_items, payload: bytes):
            self.send_response(status)
            for k, v in header_items:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _expand_loop(self, path: str, raw: bytes, hdrs: dict[str, str]):
            """distil reversible tier for OpenAI format.

            Digest context blocks behind handles, expose ``distil_expand`` as an
            OpenAI function tool, and run the recover-then-redecide loop INSIDE the
            proxy.  The caller sees only the final (post-recovery) assistant message.
            """
            body = json.loads(raw)
            if state.capture_path is not None:
                with state.lock, state.capture_path.open("a") as fh:
                    fh.write(json.dumps(body) + "\n")

            # Force non-streaming (harness sets stream=false, but guard anyway)
            if body.get("stream"):
                body = {**body, "stream": False}

            restore: dict[str, str] = {}
            new_body = compress_body_openai(
                body,
                None,
                state.stats,
                protect=state.protect,
                digest_restore=restore,
                gate_recent=state.gate_recent,
                expanded=state.expanded,
            )

            # Inject the distil_expand function tool (idempotent)
            tools = list(new_body.get("tools") or [])
            already = any(
                isinstance(t, dict)
                and t.get("type") == "function"
                and (t.get("function") or {}).get("name") == "distil_expand"
                for t in tools
            )
            if not already:
                tools.append(_DISTIL_EXPAND_FUNCTION_TOOL)
            new_body["tools"] = tools

            status, items, payload = self._forward(path, json.dumps(new_body).encode(), hdrs)
            did_expand = False

            for _ in range(MAX_EXPAND_ITERS):
                try:
                    resp = json.loads(payload)
                except (ValueError, TypeError):
                    break

                # OpenAI: tool calls live in choices[0].message.tool_calls
                choices = resp.get("choices") or []
                if not choices:
                    break
                choice = choices[0]
                finish = choice.get("finish_reason", "")
                tool_calls = (choice.get("message") or {}).get("tool_calls") or []
                expand_calls = [
                    tc
                    for tc in tool_calls
                    if isinstance(tc, dict)
                    and (tc.get("function") or {}).get("name") == "distil_expand"
                ]
                if not expand_calls or finish != "tool_calls":
                    break

                # Append the assistant message (with its tool_calls) to the conversation
                new_body["messages"].append(choice["message"])

                # Resolve each handle and append a role:"tool" message per call
                for tc in expand_calls:
                    tc_id = tc.get("id", "")
                    raw_args = (tc.get("function") or {}).get("arguments", "{}")
                    try:
                        args = json.loads(raw_args)  # arguments is a JSON STRING in OpenAI
                    except (ValueError, TypeError):
                        args = {}
                    h = args.get("handle", "")
                    full = restore.get(h)
                    if full is None:
                        tool_content = f"(no digested block with handle {h!r})"
                        # still append so the model gets a response
                    else:
                        tool_content = full
                        with state.lock:
                            state.stats.expansions += 1
                            state.expanded.add(h)  # sticky: keep full on later turns
                    new_body["messages"].append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": tool_content,
                        }
                    )

                did_expand = True
                status, items, payload = self._forward(path, json.dumps(new_body).encode(), hdrs)

            if did_expand:
                with state.lock:
                    state.stats.expand_requests += 1
            return status, items, payload

        def _proxy(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            path = self.path
            hdrs = self._upstream_headers()
            try:
                if state.expand and path.endswith("/v1/chat/completions") and raw:
                    status, items, payload = self._expand_loop(path, raw, hdrs)
                    self._send(status, items, payload)
                    return

                body_out = raw
                if path.endswith("/v1/chat/completions") and raw:
                    try:
                        body = json.loads(raw)
                        if state.capture_path is not None:
                            with state.lock, state.capture_path.open("a") as fh:
                                fh.write(json.dumps(body) + "\n")
                        # Force non-streaming
                        if body.get("stream"):
                            body = {**body, "stream": False}
                        new_body = compress_body_openai(
                            body, state.compressor, state.stats, protect=state.protect
                        )
                        body_out = json.dumps(new_body).encode()
                    except (ValueError, TypeError):
                        body_out = raw  # malformed — forward untouched
                status, items, payload = self._forward(path, body_out, hdrs)
                self._send(status, items, payload)
            except Exception as e:  # noqa: BLE001 — surface upstream/network errors as 502
                payload = json.dumps({"error": {"type": "proxy_error", "message": str(e)}}).encode()
                self._send(502, [("Content-Type", "application/json")], payload)

        def _account(self, payload: bytes):
            """Tally OpenAI usage fields into CompressStats."""
            try:
                data = json.loads(payload)
                usage = data.get("usage") or {}
            except (ValueError, TypeError):
                return
            with state.lock:
                # OpenAI: prompt_tokens / completion_tokens
                state.stats.usage_input_tokens += int(usage.get("prompt_tokens", 0) or 0)
                state.stats.usage_output_tokens += int(usage.get("completion_tokens", 0) or 0)
                if state.accounting_path is not None:
                    with state.accounting_path.open("a") as fh:
                        fh.write(json.dumps({"usage": usage}) + "\n")

        do_POST = _proxy
        do_GET = _proxy

    return Handler


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    compressor: Compressor | None,
    upstream: str = _DEFAULT_UPSTREAM,
    capture_path: Path | None = None,
    accounting_path: Path | None = None,
    protect: str | None = None,
    expand: bool = False,
    gate_recent: int | None = None,
) -> tuple[ThreadingHTTPServer, ProxyState]:
    """Start the OpenAI proxy on a background thread. Returns ``(server, state)``.

    Use ``port=0`` to bind an ephemeral port (read it from
    ``server.server_address[1]``). Call ``server.shutdown()`` to stop.

    ``upstream`` is the base URL of the local model server, e.g.
    ``"http://127.0.0.1:11434/v1"`` for Ollama or ``"http://127.0.0.1:8000/v1"``
    for vLLM.  ``protect`` (the problem statement) is never compressed.
    ``expand=True`` selects distil's reversible tier; ``compressor`` is ignored in
    that mode.
    """
    state = ProxyState(
        compressor=compressor,
        upstream=upstream,
        capture_path=capture_path,
        accounting_path=accounting_path,
        protect=protect,
        expand=expand,
        gate_recent=gate_recent,
    )
    httpd = ThreadingHTTPServer((host, port), _make_handler(state))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, state


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="distil E7 OpenAI-compatible compression proxy")
    ap.add_argument("--condition", choices=list(COMPRESSORS), default="full")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8789)
    ap.add_argument("--upstream", default=_DEFAULT_UPSTREAM)
    ap.add_argument("--capture", type=Path, default=None)
    ap.add_argument("--accounting", type=Path, default=None)
    args = ap.parse_args()
    httpd, state = serve(
        host=args.host,
        port=args.port,
        compressor=COMPRESSORS[args.condition],
        upstream=args.upstream,
        capture_path=args.capture,
        accounting_path=args.accounting,
        expand=(args.condition == EXPAND_CONDITION),
    )
    print(
        f"distil-e7 openai proxy: condition={args.condition} "
        f"upstream={args.upstream} "
        f"on {args.host}:{httpd.server_address[1]}"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
