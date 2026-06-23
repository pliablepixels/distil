"""Async high-concurrency proxy using aiohttp (optional [async] extra).

Handles thousands of concurrent streaming sessions that the threaded proxy.py
cannot sustain under high fan-out. aiohttp is lazy-imported so the stdlib core
remains dependency-free.

Usage
-----
::

    from distil.aproxy import serve
    serve(host="127.0.0.1", port=8788, upstream="https://api.anthropic.com")

Or build the app yourself::

    from distil.aproxy import make_app
    app = make_app("https://api.anthropic.com", lossless_only=False, shape_output="off")
"""

from __future__ import annotations

import json
from typing import Any

from .adapters.anthropic import compress_messages
from .httpguard import MAX_BODY_BYTES, safe_forward_path
from .tokenizer import DEFAULT as _tokenizer

# ---------------------------------------------------------------------------
# Paths that carry a ``messages`` payload worth compressing
# ---------------------------------------------------------------------------

_COMPRESSIBLE_PATHS = frozenset({"/v1/messages", "/v1/chat/completions", "/v1/responses"})

# Hop-by-hop headers must not be forwarded — they are connection-specific.
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
# Token-saving estimator (mirrors proxy.py._tokens_saved)
# ---------------------------------------------------------------------------


def _tokens_saved(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> int:
    """Rough estimate of tokens saved via the default heuristic tokeniser."""

    def _count(msgs: list[dict[str, Any]]) -> int:
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

    return max(0, _count(before) - _count(after))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_headers(headers: Any) -> dict[str, str]:
    """Strip hop-by-hop headers from a mapping, returning a plain dict."""
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def make_app(
    upstream: str,
    *,
    lossless_only: bool = False,
    shape_output: str = "off",
) -> Any:
    """Build and return an ``aiohttp.web.Application``.

    aiohttp is lazy-imported here; if it is not installed you will get a clear
    ImportError pointing to ``pip install 'distil-llm[async]'``.

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
        directive is appended so the model emits fewer tokens.
    """
    try:
        from aiohttp import web
        import aiohttp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "aiohttp is required for the async proxy. "
            "Install it with: pip install 'distil-llm[async]'"
        ) from exc

    _upstream = upstream.rstrip("/")

    # Typed key avoids NotAppKeyWarning and provides a stable reference for handlers.
    _client_key: web.AppKey[aiohttp.ClientSession] = web.AppKey("client", aiohttp.ClientSession)

    # -----------------------------------------------------------------------
    # Startup / cleanup: one shared ClientSession for the app lifetime
    # -----------------------------------------------------------------------

    async def _on_startup(app: web.Application) -> None:
        # Bound concurrent upstream connections so inbound load can't fan out into
        # an unbounded socket count (backpressure + self-DoS protection).
        connector = aiohttp.TCPConnector(limit=100)
        app[_client_key] = aiohttp.ClientSession(connector=connector)

    async def _on_cleanup(app: web.Application) -> None:
        await app[_client_key].close()

    # -----------------------------------------------------------------------
    # Compression handler
    # -----------------------------------------------------------------------

    async def _handle_compressible(request: web.Request) -> web.Response:
        if safe_forward_path(request.path) is None:
            return web.json_response({"error": "invalid request path"}, status=400)
        raw = await request.read()
        fwd_headers = _filter_headers(request.headers)
        extras: dict[str, str] = {}

        try:
            body: dict[str, Any] = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            body_bytes = raw
        else:
            if "messages" in body and isinstance(body["messages"], list):
                original: list[dict[str, Any]] = body["messages"]
                try:
                    compressed, _store = compress_messages(original, lossless_only=lossless_only)
                except Exception:  # noqa: BLE001 — compression must never break a request
                    compressed = original
                saved = _tokens_saved(original, compressed)
                body = {**body, "messages": compressed}
                extras = {
                    "x-distil-compressed": "1",
                    "x-distil-tokens-saved": str(saved),
                }
                if shape_output != "off" and not lossless_only:
                    from .output import shape_request

                    body = shape_request(body, level=shape_output, allow=True)
                    extras["x-distil-output-shaping"] = shape_output

            body_bytes = json.dumps(body).encode()

        url = _upstream + request.path
        if request.query_string:
            url = f"{url}?{request.query_string}"

        client: aiohttp.ClientSession = request.app[_client_key]
        try:
            async with client.post(
                url,
                data=body_bytes,
                headers={**fwd_headers, "content-length": str(len(body_bytes))},
            ) as resp:
                resp_body = await resp.read()
                resp_headers = _filter_headers(resp.headers)
                resp_headers.update(extras)
                return web.Response(
                    status=resp.status,
                    body=resp_body,
                    headers=resp_headers,
                )
        except aiohttp.ClientError as exc:
            err = json.dumps({"error": "upstream connection failed", "detail": str(exc)}).encode()
            return web.Response(
                status=502,
                body=err,
                content_type="application/json",
            )

    # -----------------------------------------------------------------------
    # Transparent passthrough (all other paths / methods)
    # -----------------------------------------------------------------------

    async def _passthrough(request: web.Request) -> web.Response:
        if safe_forward_path(request.path) is None:
            return web.json_response({"error": "invalid request path"}, status=400)
        raw = await request.read()
        fwd_headers = _filter_headers(request.headers)
        if raw:
            fwd_headers["content-length"] = str(len(raw))

        url = _upstream + request.path
        if request.query_string:
            url = f"{url}?{request.query_string}"

        client: aiohttp.ClientSession = request.app[_client_key]
        try:
            async with client.request(
                request.method,
                url,
                data=raw or None,
                headers=fwd_headers,
            ) as resp:
                resp_body = await resp.read()
                resp_headers = _filter_headers(resp.headers)
                return web.Response(
                    status=resp.status,
                    body=resp_body,
                    headers=resp_headers,
                )
        except aiohttp.ClientError as exc:
            err = json.dumps({"error": "upstream connection failed", "detail": str(exc)}).encode()
            return web.Response(
                status=502,
                body=err,
                content_type="application/json",
            )

    # -----------------------------------------------------------------------
    # Router
    # -----------------------------------------------------------------------

    async def _route_post(request: web.Request) -> web.Response:
        if request.path in _COMPRESSIBLE_PATHS:
            return await _handle_compressible(request)
        return await _passthrough(request)

    # Cap inbound body size so a giant POST can't exhaust memory (aiohttp returns
    # 413 automatically past this); matches the sync servers' guard.
    app = web.Application(client_max_size=MAX_BODY_BYTES)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    # Register compressible POST paths explicitly; everything else hits the wildcard.
    for path in _COMPRESSIBLE_PATHS:
        app.router.add_post(path, _route_post)

    # Wildcard for all other paths/methods (aiohttp doesn't have a true catch-all,
    # so we register explicit wildcards for the common verbs plus a fallback).
    app.router.add_route("*", "/{path_info:.*}", _passthrough)

    return app


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
) -> None:
    """Run an async aiohttp proxy server.

    Parameters
    ----------
    host:       Interface to bind on.
    port:       Port to listen on.
    upstream:   Real LLM API base URL (no trailing slash).
    lossless_only:
        When *True* only Tier-0 lossless transforms are applied.
    shape_output:
        Output-compression level: ``"off"``/``"light"``/``"aggressive"``.
    """
    try:
        from aiohttp import web
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "aiohttp is required for the async proxy. "
            "Install it with: pip install 'distil-llm[async]'"
        ) from exc

    app = make_app(upstream, lossless_only=lossless_only, shape_output=shape_output)
    print(f"distil async proxy listening on http://{host}:{port}")
    print(f"  → upstream: {upstream}")
    if shape_output != "off":
        print(f"  → output shaping: {shape_output}")
    web.run_app(app, host=host, port=port, print=None)
