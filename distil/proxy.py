"""HTTP proxy that applies distil compression to LLM API requests.

Drop-in for any client that honours a ``base_url`` parameter — Anthropic SDK,
OpenAI SDK, LiteLLM, LangChain, etc. Point the client at the proxy and every
``/v1/messages``, ``/v1/chat/completions``, or ``/v1/responses`` request will
have its ``messages`` array compressed before being forwarded to the real
upstream. All other paths and methods are forwarded unchanged.

Usage
-----
::

    from distil.proxy import serve
    serve(host="127.0.0.1", port=8788, upstream="https://api.anthropic.com")

Or as a module::

    python -m distil.proxy
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .adapters.anthropic import compress_messages
from .tokenizer import DEFAULT as _tokenizer

# ---------------------------------------------------------------------------
# Paths that carry a ``messages`` payload worth compressing
# ---------------------------------------------------------------------------

_COMPRESSIBLE_PATHS = frozenset({"/v1/messages", "/v1/chat/completions", "/v1/responses"})

# Hop-by-hop headers must never be forwarded; they are connection-specific.
_HOP_BY_HOP = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "transfer-encoding",
        "keep-alive",
        "proxy-connection",
        "te",
        "trailers",
        "upgrade",
    }
)


# ---------------------------------------------------------------------------
# Token-saving estimator
# ---------------------------------------------------------------------------


def _count_messages(msgs: list[dict[str, Any]]) -> int:
    """Heuristic token count of an Anthropic/OpenAI messages list."""
    total = 0
    for msg in msgs:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += _tokenizer.count(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                for key in ("text", "content"):
                    val = block.get(key)
                    if isinstance(val, str):
                        total += _tokenizer.count(val)
                    elif isinstance(val, list):
                        for sub in val:
                            if isinstance(sub, dict):
                                sv = sub.get("text", "")
                                if isinstance(sv, str):
                                    total += _tokenizer.count(sv)
    return total


def _tokens_saved(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> int:
    """Rough estimate of tokens saved via the default heuristic tokeniser."""
    return max(0, _count_messages(before) - _count_messages(after))


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------


def build_handler(
    upstream: str,
    *,
    lossless_only: bool = False,
    shape_output: str = "off",
    savings: Any = None,
    flush_every: int = 50,
) -> type[BaseHTTPRequestHandler]:
    """Return a ``BaseHTTPRequestHandler`` subclass configured for *upstream*.

    Parameters
    ----------
    upstream:
        Base URL of the real LLM API, e.g. ``"https://api.anthropic.com"``.
        Must not have a trailing slash.
    lossless_only:
        When *True* only Tier-0 lossless transforms are applied.
    shape_output:
        Output-compression level (``"off"``/``"light"``/``"aggressive"``). When
        not ``"off"`` and lossy compression is permitted, a verbosity-control
        ``role:"system"`` directive is appended so the model emits fewer tokens.
    """

    _upstream = upstream.rstrip("/")

    class _DistilHandler(BaseHTTPRequestHandler):
        # ----------------------------------------------------------------
        # Silence request logs — quiet by design
        # ----------------------------------------------------------------

        def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
            pass

        # ----------------------------------------------------------------
        # HTTP verb dispatch
        # ----------------------------------------------------------------

        def do_POST(self) -> None:  # noqa: N802
            if self.path in _COMPRESSIBLE_PATHS:
                self._handle_compressible()
            else:
                self._passthrough()

        def do_GET(self) -> None:  # noqa: N802
            self._passthrough()

        def do_PUT(self) -> None:  # noqa: N802
            self._passthrough()

        def do_DELETE(self) -> None:  # noqa: N802
            self._passthrough()

        def do_PATCH(self) -> None:  # noqa: N802
            self._passthrough()

        def do_HEAD(self) -> None:  # noqa: N802
            self._passthrough()

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._passthrough()

        # ----------------------------------------------------------------
        # Helpers
        # ----------------------------------------------------------------

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(length) if length else b""

        def _client_headers(self) -> dict[str, str]:
            """Client headers with hop-by-hop stripped (Content-Length excluded
            so we can recompute it after compression)."""
            return {k: v for k, v in self.headers.items() if k.lower() not in _HOP_BY_HOP}

        def _relay(
            self,
            status: int,
            resp_headers: dict[str, str],
            resp_body: bytes,
            extras: dict[str, str] | None = None,
        ) -> None:
            """Write *status*, *resp_headers*, optional *extras*, and *resp_body* to caller."""
            self.send_response(status)
            for k, v in resp_headers.items():
                if k.lower() not in _HOP_BY_HOP:
                    self.send_header(k, v)
            if extras:
                for k, v in extras.items():
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

        def _post_upstream(
            self,
            path: str,
            body: bytes,
            headers: dict[str, str],
        ) -> tuple[int, dict[str, str], bytes]:
            """POST *body* to upstream *path*. Returns (status, headers, body)."""
            url = _upstream + path
            req = urllib.request.Request(
                url,
                data=body,
                headers={**headers, "Content-Length": str(len(body))},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    rbody = resp.read()
                    rhdrs = {k: v for k, v in resp.headers.items() if k.lower() not in _HOP_BY_HOP}
                    return resp.status, rhdrs, rbody
            except urllib.error.HTTPError as exc:
                rbody = exc.read() if exc.fp else b'{"error":"upstream error"}'
                rhdrs = {k: v for k, v in exc.headers.items() if k.lower() not in _HOP_BY_HOP}
                return exc.code, rhdrs, rbody
            except urllib.error.URLError as exc:
                rbody = json.dumps(
                    {"error": "upstream connection failed", "detail": str(exc.reason)}
                ).encode()
                return 502, {"Content-Type": "application/json"}, rbody

        # ----------------------------------------------------------------
        # Compression path
        # ----------------------------------------------------------------

        def _handle_compressible(self) -> None:
            raw = self._read_body()
            headers = self._client_headers()
            extras: dict[str, str] = {}

            try:
                body: dict[str, Any] = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                # Not valid JSON — forward as-is, no extras.
                status, rhdrs, rbody = self._post_upstream(self.path, raw, headers)
                self._relay(status, rhdrs, rbody)
                return

            if "messages" in body and isinstance(body["messages"], list):
                original: list[dict[str, Any]] = body["messages"]
                compressed, _store = compress_messages(original, lossless_only=lossless_only)
                before_tok = _count_messages(original)
                after_tok = _count_messages(compressed)
                saved = max(0, before_tok - after_tok)
                body = {**body, "messages": compressed}
                extras = {
                    "x-distil-compressed": "1",
                    "x-distil-tokens-saved": str(saved),
                }
                # Accumulate GENUINE savings from real traffic into the ledger.
                if savings is not None:
                    savings.record(before_tok, after_tok)
                    if savings.requests >= flush_every:
                        savings.flush()
                # Output compression: gated by lossless_only (only on PAYG-style).
                if shape_output != "off" and not lossless_only:
                    from .output import shape_request

                    body = shape_request(body, level=shape_output, allow=True)
                    extras["x-distil-output-shaping"] = shape_output

            new_raw = json.dumps(body).encode()
            status, rhdrs, rbody = self._post_upstream(self.path, new_raw, headers)
            self._relay(status, rhdrs, rbody, extras=extras)

        # ----------------------------------------------------------------
        # Transparent passthrough (unchanged body, any verb)
        # ----------------------------------------------------------------

        def _passthrough(self) -> None:
            raw = self._read_body()
            headers = self._client_headers()
            url = _upstream + self.path
            req = urllib.request.Request(
                url,
                data=raw or None,
                headers={**headers, **({"Content-Length": str(len(raw))} if raw else {})},
                method=self.command,
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    rbody = resp.read()
                    rhdrs = {k: v for k, v in resp.headers.items() if k.lower() not in _HOP_BY_HOP}
                    self._relay(resp.status, rhdrs, rbody)
            except urllib.error.HTTPError as exc:
                rbody = exc.read() if exc.fp else b'{"error":"upstream error"}'
                rhdrs = {k: v for k, v in exc.headers.items() if k.lower() not in _HOP_BY_HOP}
                self._relay(exc.code, rhdrs, rbody)
            except urllib.error.URLError as exc:
                rbody = json.dumps(
                    {"error": "upstream connection failed", "detail": str(exc.reason)}
                ).encode()
                self._relay(502, {"Content-Type": "application/json"}, rbody)

    return _DistilHandler


# ---------------------------------------------------------------------------
# Blocking server entrypoint
# ---------------------------------------------------------------------------


def serve(
    host: str = "127.0.0.1",
    port: int = 8788,
    upstream: str = "https://api.anthropic.com",
    *,
    lossless_only: bool = False,
    shape_output: str = "off",
    record: bool = True,
    pricing_model: str = "claude-opus-4-8",
) -> None:
    """Run a blocking :class:`ThreadingHTTPServer` proxy.

    Parameters
    ----------
    host:       Interface to bind on.
    port:       Port to listen on.
    upstream:   Real LLM API base URL (no trailing slash).
    lossless_only:
        When *True* only Tier-0 lossless transforms are applied.
    shape_output:
        Output-compression level: ``"off"``/``"light"``/``"aggressive"``.
    record:
        When *True* (default), accumulate GENUINE per-request token savings from
        real traffic into the local ledger (`distil leaderboard`). Numbers only,
        never content.
    pricing_model:
        Model id used to price the genuine dollar savings.
    """
    savings = None
    if record:
        from .runtime import RuntimeSavings

        savings = RuntimeSavings(model=pricing_model)
    handler = build_handler(
        upstream, lossless_only=lossless_only, shape_output=shape_output, savings=savings
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(f"distil proxy listening on http://{host}:{port}")
    print(f"  → upstream: {upstream}")
    if shape_output != "off":
        print(f"  → output shaping: {shape_output}")
    if savings is not None:
        print("  → recording genuine savings → distil leaderboard")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if savings is not None:
            savings.flush()  # persist remaining genuine savings on shutdown
        server.server_close()


if __name__ == "__main__":
    serve()
