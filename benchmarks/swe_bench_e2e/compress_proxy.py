#!/usr/bin/env python3
"""Compression proxy for the SWE-bench end-to-end eval (Phase 5 / E7).

A thin Anthropic Messages API proxy that sits between the coding agent (aider) and
``api.anthropic.com``. For every ``/v1/messages`` request it optionally rewrites the
large *context* blocks — the file contents and tool/command output the agent reads —
with a pluggable per-block compressor, then forwards the request upstream unchanged in
every other respect. Responses stream straight back.

Three eval conditions are realised by the choice of ``compressor``:

* **A. full**      — ``compressor=None``; the proxy is a transparent pass-through
  (still used so all three conditions share identical network/accounting code).
* **B. distil**    — ``compressor=trunc_500``; distil's phase-2 certifying operating
  point ``trunc@500`` (head-truncate each compressible block to its first 500
  characters), i.e. :func:`distil.conformal._truncate_level` ``(500)`` applied to the
  agent's context blocks.
* **C. llmlingua2** — ``compressor=llmlingua2``; LLMLingua-2 at its default keep-rate,
  the strongest non-distil non-truncation baseline (E5).

What counts as a *compressible context block*
--------------------------------------------
The agent's reasoning (assistant turns) and the harness/system instructions must never
be compressed — only "the file contents the agent reads" (task scope, step 3). We
therefore compress a content block iff **all** of:

* it is a ``text`` or ``tool_result`` block (not ``tool_use``, ``image``, ...);
* it lives in a **non-system, non-assistant** message (user/tool turns carry the file
  reads and command output);
* its text is at least :data:`MIN_CHARS` characters (short blocks are instructions or
  glue, and compressing them neither helps nor is in scope).

Both B and C use this *same* selector, so the only thing that differs between them is
the compressor — an apples-to-apples comparison at the block level.

Token accounting
----------------
Every request records (pre, post) input-token counts via distil's tokenizer plus the
upstream-reported ``usage`` (input/output tokens) so the driver can compute real dollar
cost per condition. Accounting is append-only JSONL, survives interruption, and never
contains prompt text.
"""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

# distil's billing-grade-ish tokenizer (stdlib heuristic; consistent across conditions).
import sys as _sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from distil.skeleton import smart_digest  # noqa: E402
from distil.tokenizer import DEFAULT as _tokenizer  # noqa: E402

UPSTREAM = "https://api.anthropic.com"
MIN_CHARS = 500  # blocks shorter than this are glue/instructions, not file context

# A compressor maps an original block string -> compressed string. Pure and deterministic.
Compressor = Callable[[str], str]


# --------------------------------------------------------------------------- #
# Compressors (the three conditions)
# --------------------------------------------------------------------------- #
def trunc_500(text: str) -> str:
    """distil's phase-2 certifying operating point: head-truncate to 500 chars.

    Mirrors ``distil.conformal._truncate_level(500)`` (``b.text[:limit]``) exactly —
    the only difference is that here the "block" is an agent context block rather than
    a localization-corpus trajectory block.
    """
    return text[:500]


_LLMLINGUA = None


def _llmlingua_compressor():
    global _LLMLINGUA
    if _LLMLINGUA is None:
        # Reuse the exact benchmark adapter (rate 0.5, CPU) so E7's llmlingua-2 matches
        # E5's. Imported lazily — heavy torch + 560MB model, only condition C needs it.
        from benchmarks import llmlingua_adapter

        _LLMLINGUA = llmlingua_adapter
    return _LLMLINGUA


def llmlingua2(text: str) -> str:
    """LLMLingua-2 at the benchmark adapter's default rate (0.5). Memoised per text."""
    adapter = _llmlingua_compressor()
    return adapter.compress([text])[0]


_HEADROOM = None


def _headroom_compressor():
    global _HEADROOM
    if _HEADROOM is None:
        # Reuse the exact E5 adapter so E8's Headroom matches E5's invocation. Headroom is
        # an LLM-compressor: each call drives a Claude model internally (optimize=True), so
        # this condition adds compression-time API cost — imported lazily, only when used.
        from benchmarks import headroom_adapter

        _HEADROOM = headroom_adapter
    return _HEADROOM


def headroom(text: str) -> str:
    """Headroom (headroom-ai) at its default optimize pipeline. One block in, one out."""
    out = _headroom_compressor().compress([text])
    return out[0] if out else text


# --------------------------------------------------------------------------- #
# distil reversible tier: digest-behind-handle + recover-on-demand (distil_expand)
# --------------------------------------------------------------------------- #
DIGEST_HEAD = 400  # chars of head kept verbatim (non-code fallback / tiny blocks)
DIGEST_TAIL = 200  # chars of tail kept for non-code (tracebacks/assertions live there)
# 'skeleton' = content-aware navigable digest (for the active-recovery distil_expand tier);
# 'head' = head-truncation (for the passive distil_gated tier — see digest_block).

DISTIL_EXPAND_TOOL = {
    "name": "distil_expand",
    "description": (
        "Retrieve the FULL original text of a digested context block. Any file or "
        "command output shown with a '<<distil-digest: … handle=\"XXXX\">>' marker has "
        "been shortened; you are seeing only its head. BEFORE you reason about or edit "
        "such a block, call distil_expand with its handle to read the complete content. "
        "Returns the exact original text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"handle": {"type": "string", "description": "the 8-char handle"}},
        "required": ["handle"],
    },
}


def _handle(text: str) -> str:
    """Content-addressed handle for a block — deterministic across turns (same block ->
    same handle), which is what lets the proxy recognise an already-expanded block."""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def digest_block(text: str, restore: dict[str, str]) -> str:
    """distil's reversible tier: replace a block with a *content-aware skeleton* (code
    signatures kept + bodies elided; head+tail for non-code) behind a content handle, and
    record handle->original in ``restore`` so the proxy can recover it on demand.

    Unlike head-truncation, the skeleton is **navigable**: the agent sees every symbol
    that exists and the exception/assertion at a traceback's tail, so it can
    ``distil_expand`` the one block it needs instead of recovering everything. Recovery is
    byte-exact (the original is kept), preserving the reversible-tier contract.

    handle = sha256(original)[:8] — the same content-addressed scheme as
    ``distil.compress.tier1`` / ``distil.replay.expand_runner``, so it is deterministic
    and re-derivable from the same block on a later turn.
    """
    if len(text) <= MIN_CHARS:
        return text
    h = _handle(text)
    restore[h] = text
    # Digest mode (DISTIL_DIGEST_MODE): 'skeleton' (content-aware, navigable) suits the
    # ACTIVE-recovery tier (distil_expand) — the agent expands the bodies it needs. 'head'
    # (head-truncation) suits the PASSIVE tier (distil_gated), which rarely expands: a
    # navigable skeleton there backfires (the agent trusts it, never re-reads, and edits
    # against body-less context). E8 ablation: skeleton lifts expand 28.8->32.4% but
    # collapses gated 36.8->5.6%; head-trunc is gated's correct digest. See
    # docs/paper/results/swe_e2e_longhorizon/skeleton_ablation/.
    import os

    mode = os.environ.get("DISTIL_DIGEST_MODE", "skeleton")
    if mode == "head":
        body = text[:DIGEST_HEAD]
    elif mode == "surprise":
        # v1.7.0 surprise-preserving digest ("lost if surprise", arXiv 2412.17483):
        # head-truncation drops the TAIL of tracebacks — the assertion/anomaly that
        # decides the next action. Keep the head plus any anomaly lines beyond it,
        # bounded so the digest stays a digest.
        from distil.compress.salience import surprise_lines

        body = text[:DIGEST_HEAD]
        kept = [ln for ln in surprise_lines(text[DIGEST_HEAD:]) if ln.strip()][:40]
        if kept:
            body += "\n<<distil-kept anomaly lines>>\n" + "\n".join(kept)
    else:
        body = smart_digest(text, head=DIGEST_HEAD, tail=DIGEST_TAIL)
    if len(body) >= len(text):  # digest not smaller (tiny/odd block) — keep head only
        body = text[:DIGEST_HEAD]
    hidden = len(text) - len(body)
    return (
        body
        + f"\n<<distil-digest: {hidden} more chars hidden; "
        + f'call distil_expand(handle="{h}") to view the full block>>'
    )


COMPRESSORS: dict[str, Compressor | None] = {
    "full": None,
    "distil_trunc500": trunc_500,
    "llmlingua2": llmlingua2,
    "headroom": headroom,
    # distil_expand and distil_gated are realised as a stateful digest + recovery loop
    # in the proxy (not a pure per-block Compressor), so they map to None here and are
    # selected via ProxyState.expand (+ ProxyState.gate_recent for the gated variant).
    "distil_expand": None,
    "distil_gated": None,
    # v1.7.0: the relevance gate + surprise-preserving digest (E12 ablation).
    "distil_gated_surprise": None,
}
EXPAND_CONDITION = "distil_expand"
GATED_CONDITION = "distil_gated"
GATED_SURPRISE_CONDITION = "distil_gated_surprise"
GATED_CONDITIONS = frozenset({GATED_CONDITION, GATED_SURPRISE_CONDITION})
# distil_gated: keep the last N user/tool messages (working set) full, digest older
# periphery. Tunable via DISTIL_E7_GATE_RECENT — on short conversations (<= N user/tool
# turns, e.g. focused SWE-localization) the gate is a no-op (everything is "recent"); its
# payoff is long-horizon agents with large peripheral context. Lower N digests more
# periphery (more savings) but risks digesting the file under edit.
import os as _os  # noqa: E402

GATE_RECENT = int(_os.environ.get("DISTIL_E7_GATE_RECENT", "6"))
MAX_EXPAND_ITERS = 4  # cap the recover-then-redecide round-trips per agent request


# --------------------------------------------------------------------------- #
# Block selection + rewriting
# --------------------------------------------------------------------------- #
def _compressible(role: str, block: Any) -> bool:
    """True iff this block is agent-read context we are allowed to compress."""
    if role not in ("user", "tool"):  # never system, never assistant reasoning
        return False
    if isinstance(block, str):
        return len(block) >= MIN_CHARS
    if not isinstance(block, dict):
        return False
    if block.get("type") not in ("text", "tool_result"):
        return False
    return True


def _block_text(block: Any) -> str | None:
    if isinstance(block, str):
        return block
    if isinstance(block, dict) and block.get("type") == "text":
        return block.get("text")
    return None


@dataclass
class CompressStats:
    requests: int = 0
    blocks_seen: int = 0
    blocks_compressed: int = 0
    chars_before: int = 0
    chars_after: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    usage_input_tokens: int = 0
    usage_output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    blocks_protected: int = 0
    expansions: int = 0  # distil_expand recovery calls the model made (reversible tier)
    expand_requests: int = 0  # agent requests that triggered >=1 recovery loop

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def compress_body(
    body: dict[str, Any],
    compressor: Compressor | None,
    stats: CompressStats,
    protect: str | None = None,
    digest_restore: dict[str, str] | None = None,
    gate_recent: int | None = None,
    expanded: set[str] | None = None,
) -> dict[str, Any]:
    """Return a new request body with compressible context blocks rewritten.

    ``compressor=None`` (condition A) leaves the body byte-identical but still tallies
    block/token stats so all conditions report comparable pre-compression context size.

    ``digest_restore`` selects the **reversible** tier (condition distil_expand): each
    compressible block is digested with :func:`digest_block` and handle->original is
    recorded in this dict so the proxy's recovery loop can restore it on demand. When
    set it takes precedence over ``compressor``.

    ``gate_recent`` is the **relevance gate** (condition distil_gated): when set, the
    last ``gate_recent`` user/tool messages — the agent's *working set* — are left FULL,
    and only the older *periphery* is digested. This keeps the file the agent is actively
    editing intact (so it need not spend a recovery round-trip to re-read it) while still
    compressing the stable older context (which the prompt cache also rewards), and the
    digested periphery remains recoverable via ``distil_expand``. Applies only in the
    reversible (``digest_restore``) mode.

    ``protect`` is a substring (the SWE-bench *problem statement*) that must never be
    compressed: it is the task definition the agent is solving, not "file content the
    agent reads", so truncating it would handicap the compressed conditions for the
    wrong reason. Any block whose text contains ``protect`` is passed through verbatim
    (counted as seen+protected, never compressed) — for every condition identically.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    stats.requests += 1
    protect = protect or None

    # Relevance gate: indices of the last `gate_recent` user/tool messages = working set.
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
        # Sticky expansion: a block the agent already recovered this session stays FULL on
        # every later turn (handles are deterministic). Without this the proxy re-digests
        # it each turn and the agent must re-expand it — the thrash that drives expansion
        # counts into the double digits. Keep it recoverable (record the handle) but verbatim.
        sticky = False
        if digest_restore is not None and expanded is not None and before > MIN_CHARS:
            h = _handle(text)
            if h in expanded:
                digest_restore[h] = text
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
            new_messages.append({**msg, "content": rewrite_text(role, content, gk)})
            continue
        if not isinstance(content, list):
            new_messages.append(msg)
            continue
        new_content: list[Any] = []
        for block in content:
            txt = _block_text(block)
            if txt is not None and _compressible(role, block):
                new_txt = rewrite_text(role, txt, gk)
                if isinstance(block, str):
                    new_content.append(new_txt)
                else:
                    new_content.append({**block, "text": new_txt})
            else:
                # tool_result blocks carry their payload in a nested content list.
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and _compressible(role, block)
                ):
                    new_content.append(_rewrite_tool_result(role, block, rewrite_text, gk))
                else:
                    new_content.append(block)
        new_messages.append({**msg, "content": new_content})

    # Cache-stability for the relevance gate: the digested periphery is deterministic and
    # grows append-only, so the whole prefix BEFORE the working set is byte-stable across
    # turns. Placing a cache breakpoint at that boundary lets Anthropic serve the stable
    # prefix as a 0.1x cache READ instead of re-CREATING it every turn (the failure mode of
    # a single end-of-prompt breakpoint, where the sliding working set invalidates the
    # suffix). Only meaningful in gated mode (keep_full set).
    if keep_full:
        first_ws = min(keep_full)
        if first_ws > 0:
            _mark_block_cache(new_messages[first_ws - 1])
    return {**body, "messages": new_messages}


def _mark_block_cache(msg: Any) -> None:
    """Set an ephemeral cache_control on the last content block of ``msg`` (in place)."""
    if not isinstance(msg, dict):
        return
    content = msg.get("content")
    if isinstance(content, list):
        for b in reversed(content):
            if isinstance(b, dict):
                b["cache_control"] = {"type": "ephemeral"}
                return


def _rewrite_tool_result(
    role: str, block: dict[str, Any], rewrite_text, gated_keep: bool = False
) -> dict[str, Any]:
    inner = block.get("content")
    if isinstance(inner, str):
        return {**block, "content": rewrite_text(role, inner, gated_keep)}
    if isinstance(inner, list):
        new_inner = []
        for sub in inner:
            if (
                isinstance(sub, dict)
                and sub.get("type") == "text"
                and isinstance(sub.get("text"), str)
            ):
                new_inner.append({**sub, "text": rewrite_text(role, sub["text"], gated_keep)})
            else:
                new_inner.append(sub)
        return {**block, "content": new_inner}
    return block


# --------------------------------------------------------------------------- #
# The proxy server
# --------------------------------------------------------------------------- #
@dataclass
class ProxyState:
    compressor: Compressor | None
    stats: CompressStats = field(default_factory=CompressStats)
    capture_path: Path | None = None  # when set, append raw request bodies (debug only)
    accounting_path: Path | None = None
    protect: str | None = None  # problem statement: never compressed (task, not file content)
    expand: bool = False  # distil reversible tier: digest + distil_expand recovery loop
    gate_recent: int | None = None  # distil_gated: keep last N user/tool msgs full
    expanded: set[str] = field(default_factory=set)  # handles recovered this session (sticky)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _make_handler(state: ProxyState):
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
            req = urllib.request.Request(UPSTREAM + path, data=body_bytes, method=self.command)
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
            """distil reversible tier: digest context blocks behind handles, expose a
            distil_expand tool, and run the recover-then-redecide loop INSIDE the proxy.
            aider sees only the final (post-recovery) assistant message — a transparent
            recovery loop, the honest way to grade a reversible compressor end-to-end."""
            body = json.loads(raw)
            if state.capture_path is not None:
                with state.lock, state.capture_path.open("a") as fh:
                    fh.write(json.dumps(body) + "\n")
            restore: dict[str, str] = {}
            new_body = compress_body(
                body,
                None,
                state.stats,
                protect=state.protect,
                digest_restore=restore,
                gate_recent=state.gate_recent,
                expanded=state.expanded,
            )
            tools = list(new_body.get("tools") or [])
            if not any(isinstance(t, dict) and t.get("name") == "distil_expand" for t in tools):
                tools.append(DISTIL_EXPAND_TOOL)
            new_body["tools"] = tools

            status, items, payload = self._forward(path, json.dumps(new_body).encode(), hdrs)
            did_expand = False
            for _ in range(MAX_EXPAND_ITERS):
                try:
                    resp = json.loads(payload)
                except (ValueError, TypeError):
                    break
                content = resp.get("content") or []
                wants = [
                    b
                    for b in content
                    if isinstance(b, dict)
                    and b.get("type") == "tool_use"
                    and b.get("name") == "distil_expand"
                ]
                if not wants or resp.get("stop_reason") != "tool_use":
                    break
                results = []
                for tu in wants:
                    h = (tu.get("input") or {}).get("handle", "")
                    full = restore.get(h)
                    res = {"type": "tool_result", "tool_use_id": tu.get("id")}
                    if full is None:
                        res["content"] = f"(no digested block with handle {h})"
                        res["is_error"] = True
                    else:
                        res["content"] = full
                        with state.lock:
                            state.stats.expansions += 1
                            state.expanded.add(h)  # sticky: keep full on later turns
                    results.append(res)
                did_expand = True
                new_body["messages"].append({"role": "assistant", "content": content})
                new_body["messages"].append({"role": "user", "content": results})
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
                if state.expand and path.endswith("/v1/messages") and raw:
                    status, items, payload = self._expand_loop(path, raw, hdrs)
                    self._send(status, items, payload)
                    return
                body_out = raw
                if path.endswith("/v1/messages") and raw:
                    try:
                        body = json.loads(raw)
                        if state.capture_path is not None:
                            with state.lock, state.capture_path.open("a") as fh:
                                fh.write(json.dumps(body) + "\n")
                        new_body = compress_body(
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
            try:
                data = json.loads(payload)
                usage = data.get("usage") or {}
            except (ValueError, TypeError):
                return
            with state.lock:
                state.stats.usage_input_tokens += int(usage.get("input_tokens", 0) or 0)
                state.stats.usage_output_tokens += int(usage.get("output_tokens", 0) or 0)
                state.stats.cache_read_input_tokens += int(
                    usage.get("cache_read_input_tokens", 0) or 0
                )
                state.stats.cache_creation_input_tokens += int(
                    usage.get("cache_creation_input_tokens", 0) or 0
                )
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
    capture_path: Path | None = None,
    accounting_path: Path | None = None,
    protect: str | None = None,
    expand: bool = False,
    gate_recent: int | None = None,
) -> tuple[ThreadingHTTPServer, ProxyState]:
    """Start the proxy on a background thread. Returns ``(server, state)``.

    Use ``port=0`` to bind an ephemeral port (read it from ``server.server_address[1]``).
    Call ``server.shutdown()`` to stop. ``protect`` (the problem statement) is never
    compressed. ``expand=True`` selects distil's reversible tier (digest + distil_expand
    recovery loop); ``compressor`` is ignored in that mode.
    """
    state = ProxyState(
        compressor=compressor,
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

    ap = argparse.ArgumentParser(description="distil E7 compression proxy")
    ap.add_argument("--condition", choices=list(COMPRESSORS), default="full")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--capture", type=Path, default=None)
    ap.add_argument("--accounting", type=Path, default=None)
    args = ap.parse_args()
    httpd, state = serve(
        host=args.host,
        port=args.port,
        compressor=COMPRESSORS[args.condition],
        capture_path=args.capture,
        accounting_path=args.accounting,
        expand=(args.condition == EXPAND_CONDITION or args.condition in GATED_CONDITIONS),
        gate_recent=(GATE_RECENT if args.condition in GATED_CONDITIONS else None),
    )
    print(f"distil-e7 proxy: condition={args.condition} on {args.host}:{httpd.server_address[1]}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
