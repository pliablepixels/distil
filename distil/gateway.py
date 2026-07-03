"""Managed multi-tenant gateway with per-tenant savings accounting and a live dashboard.

Drop-in extension of the distil proxy: adds per-tenant token/dollar accounting,
a JSON stats endpoint (/distil/stats), and a self-contained dark HTML dashboard
(/distil/dashboard).  All other paths are handled identically to proxy.py.

Usage
-----
::

    from distil.gateway import serve_gateway
    serve_gateway(host="127.0.0.1", port=8789, upstream="https://api.anthropic.com")

Or from the module::

    python -m distil.gateway
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .adapters.anthropic import compress_messages
from .adapters.gemini import compress_generate_request
from .adapters.gemini import count_tokens as _gemini_count
from .adapters.gemini import is_gemini_path
from .httpguard import parse_content_length, safe_forward_path
from .pricing import Pricing, get as pricing_get
from .proxy import _OPENER, _UPSTREAM_TIMEOUT
from .tokenizer import DEFAULT as _tokenizer

# Safe tenant label: bounded length, no markup / control characters.
_TENANT_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# ---------------------------------------------------------------------------
# Paths that carry a ``messages`` payload worth compressing
# ---------------------------------------------------------------------------

_COMPRESSIBLE_PATHS = frozenset({"/v1/messages", "/v1/chat/completions", "/v1/responses"})

# Hop-by-hop headers must never be forwarded.
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
# Per-tenant stats
# ---------------------------------------------------------------------------


@dataclass
class TenantStats:
    requests: int = 0
    tokens_baseline: int = 0
    tokens_compressed: int = 0

    @property
    def tokens_saved(self) -> int:
        return max(0, self.tokens_baseline - self.tokens_compressed)

    def dollars_saved(self, price: Pricing) -> float:
        """USD saved, based on input token pricing."""
        return self.tokens_saved * price.input

    def pct_saved(self) -> float:
        if self.tokens_baseline == 0:
            return 0.0
        return self.tokens_saved / self.tokens_baseline * 100.0


# ---------------------------------------------------------------------------
# Thread-safe gateway state
# ---------------------------------------------------------------------------


class GatewayState:
    """Thread-safe in-memory map of tenant_id -> TenantStats."""

    def __init__(self, price: Pricing) -> None:
        self._lock = threading.Lock()
        self._tenants: dict[str, TenantStats] = {}
        self._price = price

    def record(self, tenant: str, baseline_tokens: int, compressed_tokens: int) -> None:
        """Accumulate one request's worth of token counts for *tenant*."""
        with self._lock:
            if tenant not in self._tenants:
                self._tenants[tenant] = TenantStats()
            s = self._tenants[tenant]
            s.requests += 1
            s.tokens_baseline += baseline_tokens
            s.tokens_compressed += compressed_tokens

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable dict with per-tenant stats and totals."""
        with self._lock:
            tenants: list[dict[str, Any]] = []
            tot_req = 0
            tot_baseline = 0
            tot_compressed = 0
            for tid, s in sorted(
                self._tenants.items(), key=lambda kv: kv[1].tokens_saved, reverse=True
            ):
                dollars = s.dollars_saved(self._price)
                tenants.append(
                    {
                        "tenant": tid,
                        "requests": s.requests,
                        "tokens_baseline": s.tokens_baseline,
                        "tokens_compressed": s.tokens_compressed,
                        "tokens_saved": s.tokens_saved,
                        "dollars_saved": round(dollars, 6),
                        "pct_saved": round(s.pct_saved(), 2),
                    }
                )
                tot_req += s.requests
                tot_baseline += s.tokens_baseline
                tot_compressed += s.tokens_compressed

            tot_saved = max(0, tot_baseline - tot_compressed)
            tot_dollars = tot_saved * self._price.input
            tot_pct = (tot_saved / tot_baseline * 100.0) if tot_baseline else 0.0

            totals = {
                "requests": tot_req,
                "tokens_baseline": tot_baseline,
                "tokens_compressed": tot_compressed,
                "tokens_saved": tot_saved,
                "dollars_saved": round(tot_dollars, 6),
                "pct_saved": round(tot_pct, 2),
            }
            return {"tenants": tenants, "totals": totals}


# ---------------------------------------------------------------------------
# Tenant identification — no raw key ever stored or logged
# ---------------------------------------------------------------------------


def tenant_of(headers: Any, *, trust_tenant_header: bool = False) -> str:
    """Derive a tenant identifier from request headers.

    Tenant identity comes from the AUTHENTICATED credential (a stable
    ``anon-<sha256(key)[:8]>`` id), never from a client-writable header: any
    caller could otherwise send ``x-distil-tenant: acme-corp`` and book its
    traffic under another tenant's accounting line. The explicit header is
    honored only when the operator opts in (``trust_tenant_header=True`` /
    ``--trust-tenant-header``) for deployments where an upstream gateway they
    control sets it.
    """
    if trust_tenant_header:
        explicit = headers.get("x-distil-tenant")
        if explicit:
            label = explicit.strip()
            # Bounded, safe labels only; anything else falls through to the
            # credential-derived id rather than entering accounting/dashboard.
            if _TENANT_RE.match(label):
                return label

    for header in ("x-api-key", "authorization"):
        val = headers.get(header)
        if val:
            h = hashlib.sha256(val.encode()).hexdigest()[:8]
            return f"anon-{h}"

    return "default"


# ---------------------------------------------------------------------------
# Token-saving estimator (mirrors proxy.py)
# ---------------------------------------------------------------------------


def _count_tokens(msgs: list[dict[str, Any]]) -> int:
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


# ---------------------------------------------------------------------------
# Dashboard HTML generator
# ---------------------------------------------------------------------------


def _dashboard_html(snap: dict[str, Any]) -> str:
    tenants = snap["tenants"]
    totals = snap["totals"]

    rows = ""
    for t in tenants:
        rows += (
            f"<tr>"
            f"<td>{html.escape(str(t['tenant']))}</td>"
            f"<td>{t['requests']}</td>"
            f"<td>{t['tokens_saved']:,}</td>"
            f"<td>${t['dollars_saved']:.4f}</td>"
            f"<td>{t['pct_saved']:.1f}%</td>"
            f"</tr>\n"
        )

    if not rows:
        rows = '<tr><td colspan="5" class="empty">No requests recorded yet.</td></tr>\n'

    totals_row = (
        f"<tr class='total-row'>"
        f"<td><strong>TOTAL</strong></td>"
        f"<td><strong>{totals['requests']}</strong></td>"
        f"<td><strong>{totals['tokens_saved']:,}</strong></td>"
        f"<td><strong>${totals['dollars_saved']:.4f}</strong></td>"
        f"<td><strong>{totals['pct_saved']:.1f}%</strong></td>"
        f"</tr>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>distil gateway — live dashboard</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #06070a;
    color: #e7e9ee;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    padding: 2rem;
    min-height: 100vh;
  }}
  h1 {{
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
    background: linear-gradient(90deg, #8b7bff, #5ad1c9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  .subtitle {{
    color: #6b7280;
    font-size: 0.85rem;
    margin-bottom: 2rem;
  }}
  .headline-cards {{
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
  }}
  .card {{
    background: #0f1117;
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 1.1rem 1.5rem;
    min-width: 160px;
    flex: 1;
  }}
  .card-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin-bottom: 0.35rem;
  }}
  .card-value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: #8b7bff;
  }}
  .card-value.teal {{ color: #5ad1c9; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
  }}
  thead th {{
    text-align: left;
    padding: 0.7rem 1rem;
    border-bottom: 2px solid #1e2130;
    color: #6b7280;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 600;
  }}
  tbody tr {{
    border-bottom: 1px solid #13161f;
    transition: background 0.12s;
  }}
  tbody tr:hover {{ background: #0d1020; }}
  tbody td {{
    padding: 0.75rem 1rem;
    font-variant-numeric: tabular-nums;
  }}
  tbody td:first-child {{
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    color: #8b7bff;
    font-size: 0.82rem;
  }}
  .total-row td {{
    border-top: 2px solid #1e2130;
    color: #5ad1c9;
    padding: 0.75rem 1rem;
    font-variant-numeric: tabular-nums;
  }}
  .total-row td:first-child {{
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.82rem;
  }}
  .empty {{
    color: #4b5563;
    text-align: center;
    padding: 2rem !important;
    font-style: italic;
  }}
  .table-wrap {{
    background: #0f1117;
    border: 1px solid #1e2130;
    border-radius: 10px;
    overflow: hidden;
  }}
  .refresh-note {{
    color: #374151;
    font-size: 0.72rem;
    text-align: right;
    margin-top: 0.75rem;
  }}
</style>
</head>
<body>
<h1>distil gateway</h1>
<p class="subtitle">Per-tenant token compression leaderboard &mdash; refreshes every 5 s</p>

<div class="headline-cards">
  <div class="card">
    <div class="card-label">Total Requests</div>
    <div class="card-value">{totals["requests"]}</div>
  </div>
  <div class="card">
    <div class="card-label">Tokens Saved</div>
    <div class="card-value teal">{totals["tokens_saved"]:,}</div>
  </div>
  <div class="card">
    <div class="card-label">Dollars Saved</div>
    <div class="card-value">${totals["dollars_saved"]:.4f}</div>
  </div>
  <div class="card">
    <div class="card-label">Compression Rate</div>
    <div class="card-value teal">{totals["pct_saved"]:.1f}%</div>
  </div>
</div>

<div class="table-wrap">
<table>
  <thead>
    <tr>
      <th>Tenant</th>
      <th>Requests</th>
      <th>Tokens Saved</th>
      <th>$ Saved</th>
      <th>% Saved</th>
    </tr>
  </thead>
  <tbody>
    {rows}
    {totals_row}
  </tbody>
</table>
</div>
<p class="refresh-note">Auto-refresh every 5 s &bull; distil gateway</p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------


def build_gateway_handler(
    upstream: str,
    state: GatewayState,
    price: Pricing,
    *,
    lossless_only: bool = False,
    verbatim: bool = False,
    admin_token: str | None = None,
    loopback: bool = True,
    trust_tenant_header: bool = False,
) -> type[BaseHTTPRequestHandler]:
    """Return a BaseHTTPRequestHandler subclass for the multi-tenant gateway.

    Parameters
    ----------
    upstream:
        Base URL of the real LLM API, e.g. ``"https://api.anthropic.com"``.
    state:
        Shared ``GatewayState`` instance updated on every compressible request.
    price:
        ``Pricing`` used for dollar calculations in stats / dashboard.
    lossless_only:
        Policy mode (no tool injection). The reversible digest still runs.
    verbatim:
        When *True*, skip the Tier-1 digest (Tier-0 only) — interactive-safe.
    admin_token:
        When set, ``/distil/stats`` and ``/distil/dashboard`` require
        ``Authorization: Bearer <token>``. When unset AND the server is bound
        to a non-loopback interface, those routes are refused (403): per-tenant
        usage metadata must not be readable by anyone on the network.
    loopback:
        Whether the server is bound to a loopback interface (set by
        ``serve_gateway`` from the bind host).
    trust_tenant_header:
        Honor the client-supplied ``x-distil-tenant`` header for accounting.
        Off by default — tenant identity comes from the credential hash.
    """

    _upstream = upstream.rstrip("/")

    class _GatewayHandler(BaseHTTPRequestHandler):
        # HTTP/1.1 so streamed responses can use chunked transfer framing.
        protocol_version = "HTTP/1.1"

        # ----------------------------------------------------------------
        # Silence request logs
        # ----------------------------------------------------------------

        def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
            pass

        # ----------------------------------------------------------------
        # HTTP verb dispatch
        # ----------------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/distil/stats":
                if self._admin_authorized():
                    self._handle_stats()
            elif self.path == "/distil/dashboard":
                if self._admin_authorized():
                    self._handle_dashboard()
            else:
                self._passthrough()

        def _admin_authorized(self) -> bool:
            """Gate the management endpoints. Open on loopback with no token
            configured (local single-operator use); everything else needs the
            bearer token — replies with the error itself when unauthorized."""
            if admin_token:
                supplied = self.headers.get("Authorization", "")
                if hmac.compare_digest(supplied, f"Bearer {admin_token}"):
                    return True
                self._reject(401, "invalid or missing admin token")
                return False
            if loopback:
                return True
            self._reject(
                403,
                "management endpoints are disabled on non-loopback binds "
                "unless --admin-token is set",
            )
            return False

        def do_POST(self) -> None:  # noqa: N802
            # Strip query string for path matching
            path = self.path.split("?", 1)[0]
            if path in _COMPRESSIBLE_PATHS or is_gemini_path(path):
                self._handle_compressible()
            else:
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
        # distil management endpoints
        # ----------------------------------------------------------------

        def _handle_stats(self) -> None:
            snap = state.snapshot()
            body = json.dumps(snap, indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle_dashboard(self) -> None:
            snap = state.snapshot()
            body = _dashboard_html(snap).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ----------------------------------------------------------------
        # Compression path
        # ----------------------------------------------------------------

        def _handle_compressible(self) -> None:
            if safe_forward_path(self.path) is None:
                self._reject(400, "invalid request path")
                return
            raw = self._read_body()
            if raw is None:
                self._reject(413, "request body too large or malformed Content-Length")
                return
            headers = self._client_headers()
            tenant = tenant_of(self.headers, trust_tenant_header=trust_tenant_header)

            try:
                body: dict[str, Any] = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                # Not valid JSON — forward as-is
                status, rhdrs, rbody = self._post_upstream(self.path, raw, headers)
                self._relay(status, rhdrs, rbody)
                return

            # Echo the tenant label back only when it's an operator-trusted
            # explicit label — an anon-<hash> is a stable credential-derived
            # correlator that shouldn't ride response headers.
            extras: dict[str, str] = {}
            if not tenant.startswith("anon-"):
                extras["x-distil-tenant"] = tenant

            if "messages" in body and isinstance(body["messages"], list):
                original: list[dict[str, Any]] = body["messages"]
                try:
                    compressed, _store = compress_messages(original, verbatim=verbatim)
                except Exception:  # noqa: BLE001 — compression must never break a request
                    compressed = original

                baseline_tokens = _count_tokens(original)
                compressed_tokens = _count_tokens(compressed)
                tokens_saved = max(0, baseline_tokens - compressed_tokens)

                state.record(tenant, baseline_tokens, compressed_tokens)

                body = {**body, "messages": compressed}
                extras["x-distil-tokens-saved"] = str(tokens_saved)

            elif "contents" in body and isinstance(body["contents"], list):
                # Gemini generateContent shape.
                baseline_tokens = _gemini_count(body)
                try:
                    body, _store = compress_generate_request(body, verbatim=verbatim)
                except Exception:  # noqa: BLE001 — compression must never break a request
                    pass
                compressed_tokens = _gemini_count(body)
                tokens_saved = max(0, baseline_tokens - compressed_tokens)
                state.record(tenant, baseline_tokens, compressed_tokens)
                extras["x-distil-tokens-saved"] = str(tokens_saved)

            new_raw = json.dumps(body).encode()
            # Streamed requests relay incrementally — TTFT preserved per tenant.
            if bool(body.get("stream")) or ":streamGenerateContent" in self.path:
                from .streamrelay import stream_upstream

                stream_upstream(
                    self,
                    _upstream + self.path,
                    new_raw,
                    headers,
                    timeout=_UPSTREAM_TIMEOUT,
                    hop_by_hop=_HOP_BY_HOP,
                    extras=extras,
                )
                return
            status, rhdrs, rbody = self._post_upstream(self.path, new_raw, headers)
            self._relay(status, rhdrs, rbody, extras=extras)

        # ----------------------------------------------------------------
        # Transparent passthrough (unchanged body, any verb)
        # ----------------------------------------------------------------

        def _passthrough(self) -> None:
            if safe_forward_path(self.path) is None:
                self._reject(400, "invalid request path")
                return
            raw = self._read_body()
            if raw is None:
                self._reject(413, "request body too large or malformed Content-Length")
                return
            headers = self._client_headers()
            url = _upstream + self.path
            req = urllib.request.Request(
                url,
                data=raw or None,
                headers={**headers, **({"Content-Length": str(len(raw))} if raw else {})},
                method=self.command,
            )
            try:
                with _OPENER.open(req, timeout=_UPSTREAM_TIMEOUT) as resp:
                    rbody = resp.read()
                    rhdrs = {k: v for k, v in resp.headers.items() if k.lower() not in _HOP_BY_HOP}
                    self._relay(resp.status, rhdrs, rbody)
            except urllib.error.HTTPError as exc:
                rbody = exc.read() if exc.fp else b'{"error":"upstream error"}'
                rhdrs = {k: v for k, v in exc.headers.items() if k.lower() not in _HOP_BY_HOP}
                self._relay(exc.code, rhdrs, rbody)
            except urllib.error.URLError as exc:
                rbody = json.dumps(
                    {"error": "upstream connection failed", "detail": str(exc.reason)[:200]}
                ).encode()
                self._relay(502, {"Content-Type": "application/json"}, rbody)
            except TimeoutError:
                self._relay(
                    504, {"Content-Type": "application/json"}, b'{"error":"upstream timed out"}'
                )

        # ----------------------------------------------------------------
        # Shared helpers
        # ----------------------------------------------------------------

        def _read_body(self) -> bytes | None:
            length = parse_content_length(self.headers.get("Content-Length"))
            if length is None:
                return None
            return self.rfile.read(length) if length else b""

        def _reject(self, code: int, message: str) -> None:
            body = json.dumps({"error": message}).encode()
            self._relay(code, {"Content-Type": "application/json"}, body)

        def _client_headers(self) -> dict[str, str]:
            """Client headers with hop-by-hop stripped."""
            return {k: v for k, v in self.headers.items() if k.lower() not in _HOP_BY_HOP}

        def _relay(
            self,
            status: int,
            resp_headers: dict[str, str],
            resp_body: bytes,
            extras: dict[str, str] | None = None,
        ) -> None:
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
            if safe_forward_path(path) is None:
                return (
                    400,
                    {"Content-Type": "application/json"},
                    b'{"error":"invalid request path"}',
                )
            url = _upstream + path
            req = urllib.request.Request(
                url,
                data=body,
                headers={**headers, "Content-Length": str(len(body))},
                method="POST",
            )
            try:
                with _OPENER.open(req, timeout=_UPSTREAM_TIMEOUT) as resp:
                    rbody = resp.read()
                    rhdrs = {k: v for k, v in resp.headers.items() if k.lower() not in _HOP_BY_HOP}
                    return resp.status, rhdrs, rbody
            except urllib.error.HTTPError as exc:
                rbody = exc.read() if exc.fp else b'{"error":"upstream error"}'
                rhdrs = {k: v for k, v in exc.headers.items() if k.lower() not in _HOP_BY_HOP}
                return exc.code, rhdrs, rbody
            except urllib.error.URLError as exc:
                rbody = json.dumps(
                    {"error": "upstream connection failed", "detail": str(exc.reason)[:200]}
                ).encode()
                return 502, {"Content-Type": "application/json"}, rbody
            except TimeoutError:
                return 504, {"Content-Type": "application/json"}, b'{"error":"upstream timed out"}'

    return _GatewayHandler


# ---------------------------------------------------------------------------
# Blocking server entrypoint
# ---------------------------------------------------------------------------


def serve_gateway(
    host: str = "127.0.0.1",
    port: int = 8789,
    upstream: str = "https://api.anthropic.com",
    *,
    pricing_model: str = "claude-opus-4-8",
    lossless_only: bool = False,
    verbatim: bool = False,
    admin_token: str | None = None,
    trust_tenant_header: bool = False,
) -> None:
    """Run a blocking ThreadingHTTPServer gateway.

    Parameters
    ----------
    host:           Interface to bind on.
    port:           Port to listen on.
    upstream:       Real LLM API base URL (no trailing slash).
    pricing_model:  Model key from ``distil.pricing.CATALOG`` for dollar accounting.
    lossless_only:  Policy mode (no tool injection); the reversible digest still runs.
    verbatim:       When *True*, skip the Tier-1 digest (Tier-0 only) — interactive-safe.
    admin_token:    Bearer token required for /distil/stats and /distil/dashboard.
                    Mandatory for those routes on non-loopback binds.
    trust_tenant_header:
                    Honor the client-supplied x-distil-tenant header (off by
                    default; tenant identity comes from the credential hash).
    """
    price = pricing_get(pricing_model)
    state = GatewayState(price)
    loopback = host in ("127.0.0.1", "::1", "localhost")
    handler = build_gateway_handler(
        upstream,
        state,
        price,
        lossless_only=lossless_only,
        verbatim=verbatim,
        admin_token=admin_token or os.environ.get("DISTIL_GATEWAY_TOKEN") or None,
        loopback=loopback,
        trust_tenant_header=trust_tenant_header,
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(f"distil gateway listening on http://{host}:{port}")
    print(f"  dashboard: http://{host}:{port}/distil/dashboard")
    if not loopback and not (admin_token or os.environ.get("DISTIL_GATEWAY_TOKEN")):
        print("  ! non-loopback bind without --admin-token: /distil/* routes are disabled")
    print(f"  → upstream: {upstream}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    serve_gateway()
