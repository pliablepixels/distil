"""Incremental upstream→client relay for the sync HTTP servers.

The whole point of fronting an interactive agent is that tokens appear as the
model produces them. Buffering an SSE response start-to-finish turns
time-to-first-token into time-to-last-token, so this module relays the
upstream response chunk-by-chunk instead — while still returning the complete
buffered body to the caller for content-free accounting (shadow-mode decision
signatures). Chunked transfer framing is emitted when the upstream declares no
Content-Length (the SSE case); requires the handler to speak HTTP/1.1.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

_CHUNK = 8192


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Relay 3xx to the client instead of following it: auto-following would
    re-send the client's Authorization/x-api-key to whatever host the upstream
    names — the client's own HTTP stack must decide that."""

    def redirect_request(self, *a, **k):  # noqa: ANN002, ANN003
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _is_timeout(exc: urllib.error.URLError) -> bool:
    import socket

    return isinstance(exc.reason, (socket.timeout, TimeoutError))


def stream_upstream(
    handler: BaseHTTPRequestHandler,
    url: str,
    body: bytes | None,
    headers: dict[str, str],
    *,
    method: str = "POST",
    timeout: float,
    hop_by_hop: frozenset[str],
    extras: dict[str, str] | None = None,
    want_body: bool = False,
) -> tuple[int, bytes | None]:
    """Send the request and relay the response to *handler* incrementally.

    When ``want_body`` is set, returns the complete response body (buffered as
    it streamed) so callers can run post-hoc accounting (shadow sampling); when
    it is not, the body is relayed but never accumulated — N concurrent large
    streams would otherwise pin N full responses in memory. Always returns the
    relayed HTTP status (a synthetic 502/504 on connection failure) as the first
    element so callers can book accounting only on a confirmed 2xx; the body is
    ``None`` unless ``want_body`` was set and a response was read.
    Once streaming has begun, a mid-stream failure closes the connection —
    there is no valid way to append an error to a partially-delivered body.
    """
    req = urllib.request.Request(
        url,
        data=body,
        headers={**headers, **({"Content-Length": str(len(body))} if body else {})},
        method=method,
    )

    def _error(status: int, payload: bytes) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    try:
        resp = _OPENER.open(req, timeout=timeout)  # noqa: S310 — operator-set upstream
    except urllib.error.HTTPError as exc:
        rbody = exc.read() if exc.fp else b'{"error":"upstream error"}'
        handler.send_response(exc.code)
        for k, v in exc.headers.items():
            if k.lower() not in hop_by_hop:
                handler.send_header(k, v)
        handler.send_header("Content-Length", str(len(rbody)))
        handler.end_headers()
        handler.wfile.write(rbody)
        return exc.code, None  # non-2xx relayed to client; not a bookable success
    except urllib.error.URLError as exc:
        status = 504 if _is_timeout(exc) else 502
        _error(
            status,
            json.dumps(
                {"error": "upstream connection failed", "detail": str(exc.reason)[:200]}
            ).encode(),
        )
        return status, None
    except TimeoutError:
        _error(504, b'{"error":"upstream timed out"}')
        return 504, None

    with resp:
        length = resp.headers.get("Content-Length")
        chunked = length is None
        handler.send_response(resp.status)
        for k, v in resp.headers.items():
            if k.lower() not in hop_by_hop:
                handler.send_header(k, v)
        for k, v in (extras or {}).items():
            handler.send_header(k, v)
        if chunked:
            handler.send_header("Transfer-Encoding", "chunked")
        else:
            handler.send_header("Content-Length", length)
        handler.end_headers()

        buf = bytearray()
        try:
            while True:
                # read1: return as soon as ANY bytes arrive (at most one socket
                # read) — resp.read(n) would block until n bytes accumulate,
                # defeating incremental delivery on a dribbling SSE stream.
                chunk = resp.read1(_CHUNK)
                if not chunk:
                    break
                if want_body:
                    buf += chunk
                if chunked:
                    handler.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                else:
                    handler.wfile.write(chunk)
                handler.wfile.flush()
            if chunked:
                handler.wfile.write(b"0\r\n\r\n")
                handler.wfile.flush()
        except OSError:
            # Client disconnected or upstream stalled mid-stream: nothing valid
            # can be appended once bytes have flowed — drop the connection but
            # keep what streamed for accounting.
            handler.close_connection = True
        return resp.status, (bytes(buf) if want_body else None)
