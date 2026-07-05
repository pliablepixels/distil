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
import os
import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .adapters.anthropic import compress_messages
from .adapters.gemini import compress_generate_request
from .adapters.gemini import count_tokens
from .adapters.gemini import is_gemini_path
from .httpguard import parse_content_length, safe_forward_path, strip_query
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


# Upstream socket timeout (seconds). Generous — LLM generations run minutes —
# but finite, so a wedged upstream can never pin a worker thread forever.
_UPSTREAM_TIMEOUT = float(os.environ.get("DISTIL_UPSTREAM_TIMEOUT", "600"))


def _is_timeout(exc: urllib.error.URLError) -> bool:
    return isinstance(exc.reason, (socket.timeout, TimeoutError))


class QuietHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that doesn't traceback-spam on client disconnects.

    Agents (Claude Code especially) reset/abandon connections constantly —
    cancelled streams, retries, statusline polls. Those surface here as
    ConnectionResetError/BrokenPipeError in the handler thread; they are
    routine, not bugs. Everything else still gets the stdlib traceback.
    """

    def handle_error(self, request, client_address):  # noqa: ANN001 - stdlib signature
        import sys

        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Relay 3xx instead of following: auto-following would re-send the
    client's credentials to whatever host the upstream redirect names."""

    def redirect_request(self, *a, **k):  # noqa: ANN002, ANN003
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


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


def _model_from_path(path: str) -> str | None:
    """Extract the model id from a Gemini-style URL (``.../models/<id>:action``)."""
    marker = "/models/"
    idx = path.find(marker)
    if idx < 0:
        return None
    tail = path[idx + len(marker) :]
    return tail.split(":", 1)[0].split("/", 1)[0] or None


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------


def build_handler(
    upstream: str,
    *,
    lossless_only: bool = False,
    verbatim: bool = False,
    shape_output: str = "off",
    savings: Any = None,
    flush_every: int = 10,
    expand: bool = False,
    shadow_rate: float = 0.0,
    session_delta: bool = False,
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
    # A human-readable mode label, echoed on every compressed response as
    # x-distil-mode so a user seeing ▼0 can tell *why*: verbatim disables the
    # reversible digest (savings come only from lossless whitespace/JSON), so
    # ▼0 there is the mode, not a bug.
    _mode_label = "verbatim" if verbatim else ("lossless-only" if lossless_only else "digest")

    # Learning flywheel state (loaded once when expand is on): the learned
    # keep-byte-exact policy + the accumulating expand stats. See distil.learn.
    _learn_stats = None
    _expand_keep = None
    if expand:
        from .learn import ExpandStats, keep_predicate

        _learn_stats = ExpandStats.load()
        _expand_keep = keep_predicate(_learn_stats)

    # Outcome-guided policy (always on — never-regressing by construction):
    # content classes whose digestion co-occurred with END-TO-END task
    # regressions are kept byte-exact. See distil.compress.guideline.
    from .compress.guideline import OutcomeStats

    _outcome_keep = OutcomeStats.load().keep_predicate()

    def _learn_keep(text: str) -> bool:
        return _outcome_keep(text) or (_expand_keep is not None and _expand_keep(text))

    # Shadow-mode live decision-equivalence: sample a fraction of requests, run the
    # decision uncompressed too (in the background), and record whether it matched.
    _shadow_sampler = None
    _shadow_ledger = None
    _shadow_threads: list[threading.Thread] = []
    if shadow_rate and shadow_rate > 0:
        from .shadow import ShadowLedger, ShadowSampler

        _shadow_sampler = ShadowSampler(shadow_rate)
        _shadow_ledger = ShadowLedger()

    class _DistilHandler(BaseHTTPRequestHandler):
        # HTTP/1.1 so streamed responses can use chunked transfer framing
        # (every non-streaming response still carries an exact Content-Length).
        protocol_version = "HTTP/1.1"

        # ----------------------------------------------------------------
        # Silence request logs — quiet by design
        # ----------------------------------------------------------------

        def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
            pass

        # ----------------------------------------------------------------
        # HTTP verb dispatch
        # ----------------------------------------------------------------

        def do_POST(self) -> None:  # noqa: N802
            p = strip_query(self.path)
            if p in _COMPRESSIBLE_PATHS or is_gemini_path(p):
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

        def _read_body(self) -> bytes | None:
            """Read the request body; on a malformed/oversized/chunked request,
            send the error response itself and return None (caller just returns)."""
            if not self.headers.get("Content-Length") and "chunked" in (
                self.headers.get("Transfer-Encoding") or ""
            ):
                # A chunked body would otherwise be read as empty and silently
                # dropped — fail loudly instead (LLM SDKs always send a length).
                self._reject(411, "chunked request bodies are not supported; send Content-Length")
                return None
            length = parse_content_length(self.headers.get("Content-Length"))
            if length is None:
                self._reject(413, "request body too large or malformed Content-Length")
                return None
            return self.rfile.read(length) if length else b""

        def _reject(self, code: int, message: str) -> None:
            body = json.dumps({"error": message}).encode()
            self._relay(code, {"Content-Type": "application/json"}, body)

        def _client_headers(self, *, identity: bool = False) -> dict[str, str]:
            """Client headers with hop-by-hop stripped (Content-Length excluded
            so we can recompute it after compression). ``identity=True`` also
            drops Accept-Encoding: on compressible paths a gzip upstream body
            would silently defeat the expand loop and shadow decision parsing
            (both read the response), so those requests ask for identity."""
            out = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_BY_HOP}
            if identity:
                out = {k: v for k, v in out.items() if k.lower() != "accept-encoding"}
            return out

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
                status = 504 if _is_timeout(exc) else 502
                rbody = json.dumps(
                    {"error": "upstream connection failed", "detail": str(exc.reason)[:200]}
                ).encode()
                return status, {"Content-Type": "application/json"}, rbody
            except TimeoutError:
                rbody = b'{"error":"upstream timed out"}'
                return 504, {"Content-Type": "application/json"}, rbody

        # ----------------------------------------------------------------
        # Compression path
        # ----------------------------------------------------------------

        def _handle_compressible(self) -> None:
            if safe_forward_path(self.path) is None:
                self._reject(400, "invalid request path")
                return
            raw = self._read_body()
            if raw is None:
                return  # _read_body already sent the error response
            headers = self._client_headers(identity=True)
            extras: dict[str, str] = {}
            store: Any = None  # RestoreStore once messages are compressed (for expand)

            try:
                body: dict[str, Any] = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                # Not valid JSON — forward as-is, no extras.
                status, rhdrs, rbody = self._post_upstream(self.path, raw, headers)
                self._relay(status, rhdrs, rbody)
                return

            if "messages" in body and isinstance(body["messages"], list):
                original: list[dict[str, Any]] = body["messages"]
                # Cache-delta coding (opt-in): cross-turn dedup + cross-version delta,
                # applied to the ORIGINALS before compression so re-reads match across
                # turns. Cache-monotonic (suffix-only) and reversible.
                pre = original
                _dstats = None
                _dstore = None
                if session_delta:
                    try:
                        from .cachedelta import delta_encode, get_session, session_key

                        _sess = get_session(session_key(original))
                        pre, _dstore, _dstats = delta_encode(original, session=_sess)
                    except Exception:  # noqa: BLE001 — never break a request
                        pre, _dstore, _dstats = original, None, None
                try:
                    compressed, store = compress_messages(pre, verbatim=verbatim, keep=_learn_keep)
                except Exception:  # noqa: BLE001 — compression must never break a request
                    compressed, store = pre, None
                # Merge cache-delta references into the store so distil_expand recovers them.
                if _dstore is not None and store is not None:
                    for _h in _dstore.handles:
                        try:
                            store._record(_h, _dstore.expand(_h))
                        except Exception:  # noqa: BLE001
                            pass
                # Learning: tally what we digested, by content-free signature.
                if _learn_stats is not None and getattr(store, "handles", None):
                    from .learn import signature

                    for h in store.handles:
                        try:
                            _learn_stats.record_digest(signature(store.expand(h)))
                        except Exception:  # noqa: BLE001 — learning never breaks a request
                            pass
                before_tok = _count_messages(original)
                after_tok = _count_messages(compressed)
                saved = max(0, before_tok - after_tok)
                body = {**body, "messages": compressed}
                extras = {
                    "x-distil-compressed": "1",
                    "x-distil-tokens-saved": str(saved),
                    "x-distil-mode": _mode_label,
                    # Bytes in the compressible zone (user/tool content distil is
                    # allowed to touch) — when this is ~0, a ▼0 is "nothing large
                    # to compress this turn", not a failure. System prompt, tool
                    # definitions, images and assistant text are never counted.
                    "x-distil-compressible-tokens": str(_count_messages(original)),
                }
                if _dstats is not None:
                    extras["x-distil-cache-refs"] = str(_dstats.exact_refs + _dstats.delta_refs)
                    extras["x-distil-cache-delta"] = str(_dstats.delta_refs)
                    extras["x-distil-cache-tokens-saved"] = str(_dstats.tokens_saved)
                    # Cache-prefix observability: how many leading messages were
                    # byte-stable vs the previous turn (the prompt-cache-read region).
                    # Stateful, content-free — the verifiable benefit of a prefix-freeze
                    # router, without the lossy rewrite (distil is cache-monotonic).
                    extras["x-distil-cache-prefix-msgs"] = str(_dstats.prefix_msgs)
                # Recoverable compression: if anything was digested, offer the model
                # the distil_expand tool so it can pull back detail on demand.
                if expand and getattr(store, "handles", None):
                    from .expand import inject_expand_tool

                    body = inject_expand_tool(body)
                # Accumulate GENUINE savings from real traffic into the ledger,
                # priced per the model THIS request names (agents mix models).
                if savings is not None:
                    savings.record(before_tok, after_tok, model=body.get("model"))
                    savings.maybe_flush(every=flush_every)
                # Output compression: gated by lossless_only (only on PAYG-style).
                if shape_output != "off" and not lossless_only:
                    from .output import shape_request

                    _shape = "anthropic" if strip_query(self.path) == "/v1/messages" else "openai"
                    body = shape_request(body, level=shape_output, allow=True, shape=_shape)
                    extras["x-distil-output-shaping"] = shape_output

            elif "contents" in body and isinstance(body["contents"], list):
                # Gemini generateContent shape. Reversible content compression only;
                # expand-tool / output-shaping are messages-format-only today.
                before_tok = count_tokens(body)
                try:
                    body, store = compress_generate_request(
                        body, verbatim=verbatim, keep=_learn_keep
                    )
                except Exception:  # noqa: BLE001 — compression must never break a request
                    store = None
                if _learn_stats is not None and getattr(store, "handles", None):
                    from .learn import signature

                    for h in store.handles:
                        try:
                            _learn_stats.record_digest(signature(store.expand(h)))
                        except Exception:  # noqa: BLE001 — learning never breaks a request
                            pass
                after_tok = count_tokens(body)
                saved = max(0, before_tok - after_tok)
                extras = {
                    "x-distil-compressed": "1",
                    "x-distil-tokens-saved": str(saved),
                }
                if savings is not None:
                    # Gemini requests carry the model in the URL path, not the body.
                    savings.record(before_tok, after_tok, model=_model_from_path(self.path))
                    savings.maybe_flush(every=flush_every)

            new_raw = json.dumps(body).encode()

            # Decide shadow sampling BEFORE relaying so the marker header can be
            # sent on the streaming path too (headers go out before the body).
            shadow_sampled = _shadow_sampler is not None and _shadow_sampler.should_sample()
            if shadow_sampled:
                extras["x-distil-shadow"] = "sampled"

            # Streaming pass-through: when the client asked for a streamed
            # response, relay upstream bytes as they arrive — time-to-first-token
            # is preserved. The expand loop needs the complete response (it may
            # re-query before answering), so expand-eligible requests stay on
            # the buffered path.
            want_stream = bool(body.get("stream")) or ":streamGenerateContent" in self.path
            if want_stream and not (expand and getattr(store, "handles", None)):
                from .streamrelay import stream_upstream

                rbody_opt = stream_upstream(
                    self,
                    _upstream + self.path,
                    new_raw,
                    headers,
                    timeout=_UPSTREAM_TIMEOUT,
                    hop_by_hop=_HOP_BY_HOP,
                    extras=extras,
                )
                if _learn_stats is not None:
                    _learn_stats.save()
                if shadow_sampled and rbody_opt:
                    self._spawn_shadow(raw, headers, rbody_opt)
                return

            status, rhdrs, rbody = self._post_upstream(self.path, new_raw, headers)

            # Transparent expand loop: resolve any distil_expand tool calls against
            # the local store and re-query, invisibly, before returning to the agent.
            if expand and getattr(store, "handles", None):
                try:
                    resp_json = json.loads(rbody)
                except (ValueError, TypeError):
                    resp_json = None
                if isinstance(resp_json, dict):
                    from .expand import record_signal, run_expand_loop

                    def _post(b: dict[str, Any]) -> dict[str, Any]:
                        _s, _h, rb = self._post_upstream(self.path, json.dumps(b).encode(), headers)
                        return json.loads(rb)

                    def _on_signal(handle: str, original: str) -> None:
                        record_signal(handle, original)  # content-free expand log
                        if _learn_stats is not None:  # learn the expanded signature
                            from .learn import signature

                            _learn_stats.record_expand(signature(original))

                    final = run_expand_loop(body, resp_json, store, _post, on_signal=_on_signal)
                    if final is not resp_json:
                        rbody = json.dumps(final).encode()
                        extras["x-distil-expanded"] = "1"
            if _learn_stats is not None:  # persist the learned policy (atomic)
                _learn_stats.save()

            # Shadow-mode: on a sampled request, re-run the decision UNCOMPRESSED in
            # the background and record whether it matched — a live decision-change
            # signal on real traffic. Never blocks the client's response.
            if shadow_sampled:
                self._spawn_shadow(raw, headers, rbody)
            self._relay(status, rhdrs, rbody, extras=extras)

        def _spawn_shadow(
            self, orig_raw: bytes, headers: dict[str, str], compressed: bytes
        ) -> None:
            """Re-run the request uncompressed in the background and record whether
            the agent's decision matched. Never blocks the client's response."""

            def _shadow_compare() -> None:
                try:
                    from .shadow import decision_signature_from_body

                    _s, _h, orig_rbody = self._post_upstream(self.path, orig_raw, headers)
                    # decision_signature_from_body handles both JSON and streamed
                    # (SSE / chunk-array) bodies, so this works for Claude Code /
                    # Codex / Gemini sessions, which stream their responses.
                    comp_sig = decision_signature_from_body(compressed)
                    orig_sig = decision_signature_from_body(orig_rbody)
                    if _shadow_ledger is not None:
                        _shadow_ledger.record(comp_sig == orig_sig)
                except Exception:  # noqa: BLE001 — shadow must never affect the request
                    pass

            # Track the thread so teardown can drain it: a daemon thread is
            # killed on process exit, which loses the sample on a quick run
            # (e.g. `claude -p`) or right after the last turn. Prune finished
            # ones first so the list stays bounded on long sessions.
            _shadow_threads[:] = [t for t in _shadow_threads if t.is_alive()]
            _t = threading.Thread(target=_shadow_compare, daemon=True)
            _shadow_threads.append(_t)
            _t.start()

        # ----------------------------------------------------------------
        # Transparent passthrough (unchanged body, any verb)
        # ----------------------------------------------------------------

        def _passthrough(self) -> None:
            if safe_forward_path(self.path) is None:
                self._reject(400, "invalid request path")
                return
            raw = self._read_body()
            if raw is None:
                return  # _read_body already sent the error response
            from .streamrelay import stream_upstream

            stream_upstream(
                self,
                _upstream + self.path,
                raw or None,
                self._client_headers(),
                method=self.command,
                timeout=_UPSTREAM_TIMEOUT,
                hop_by_hop=_HOP_BY_HOP,
            )

    _DistilHandler.shadow_threads = _shadow_threads  # type: ignore[attr-defined]  # drained on shutdown
    return _DistilHandler


def _install_sigterm_flush(proc_holder: list | None = None) -> None:
    """Turn SIGTERM into KeyboardInterrupt so the caller's ``finally`` block
    (savings flush, shadow drain) runs on a plain ``kill`` instead of dropping
    up to a flush-window of recorded savings. Forwards the signal to a wrapped
    child process first, if one is registered in *proc_holder*."""
    import signal

    def _on_term(signum: int, frame: object) -> None:  # noqa: ARG001
        if proc_holder:
            try:
                proc_holder[0].terminate()
            except Exception:  # noqa: BLE001 — best-effort child shutdown
                pass
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _on_term)
    except ValueError:
        pass  # not the main thread (embedded use) — finally-blocks still cover Ctrl+C


def _drain_shadow(handler: type, budget: float = 6.0) -> None:
    """Let in-flight shadow comparisons finish recording before the proxy exits.

    Shadow runs each sampled decision uncompressed in a background thread; without
    draining, a quick run (or the last turn before shutdown) loses the sample.
    Bounded by ``budget`` seconds total so a hung upstream can't block teardown."""
    import time

    threads = [t for t in getattr(handler, "shadow_threads", []) or [] if t.is_alive()]
    if not threads:
        return
    deadline = time.monotonic() + budget
    for t in threads:
        t.join(timeout=max(0.0, deadline - time.monotonic()))


# ---------------------------------------------------------------------------
# Blocking server entrypoint
# ---------------------------------------------------------------------------


def serve(
    host: str = "127.0.0.1",
    port: int = 8788,
    upstream: str = "https://api.anthropic.com",
    *,
    lossless_only: bool = False,
    verbatim: bool = False,
    shape_output: str = "off",
    record: bool = True,
    pricing_model: str = "claude-opus-4-8",
    expand: bool = False,
    shadow_rate: float = 0.0,
    session_delta: bool = False,
) -> None:
    """Run a blocking :class:`ThreadingHTTPServer` proxy.

    Parameters
    ----------
    host:       Interface to bind on.
    port:       Port to listen on.
    upstream:   Real LLM API base URL (no trailing slash).
    lossless_only:
        Policy mode: no lossy output-shaping and no tool injection. The reversible
        Tier-1 digest still runs (it is the lossless, certified strategy).
    verbatim:
        When *True*, skip the Tier-1 digest entirely (Tier-0 only) so the model
        sees content verbatim — for interactive sessions / out-of-distribution
        traffic. Lower savings, byte-in-context fidelity.
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
        upstream,
        lossless_only=lossless_only,
        verbatim=verbatim,
        shape_output=shape_output,
        savings=savings,
        expand=expand,
        shadow_rate=shadow_rate,
        session_delta=session_delta,
    )
    server = QuietHTTPServer((host, port), handler)
    print(f"distil proxy listening on http://{host}:{port}")
    print(f"  → upstream: {upstream}")
    if shadow_rate and shadow_rate > 0:
        print(
            f"  → shadow-mode live decision-equivalence: sampling {shadow_rate * 100:.0f}% "
            "(distil shadow-stats)"
        )
    if expand:
        print(
            "  → recoverable compression: distil_expand tool active (agent recovers detail on demand)"
        )
    if shape_output != "off":
        print(f"  → output shaping: {shape_output}")
    if savings is not None:
        print("  → recording genuine savings → distil leaderboard")
    _install_sigterm_flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _drain_shadow(handler)
        if savings is not None:
            savings.flush()  # persist remaining genuine savings on shutdown
        server.server_close()


def wrap_run(
    command: list[str],
    *,
    host: str = "127.0.0.1",
    upstream: str = "https://api.anthropic.com",
    lossless_only: bool = False,
    verbatim: bool = False,
    shape_output: str = "off",
    record: bool = True,
    pricing_model: str = "claude-opus-4-8",
    env_var: str = "ANTHROPIC_BASE_URL",
    expand: bool = False,
    session_delta: bool = False,
    shadow_rate: float = 0.0,
) -> int:
    """Run *command* with its API base URL transparently pointed at a Distil proxy.

    Starts the proxy on an ephemeral local port in a background thread, injects
    ``env_var`` (default ``ANTHROPIC_BASE_URL``) into the child's environment so
    any base-url-honoring SDK routes through compression with no code change,
    runs the command to completion, then tears the proxy down — flushing genuine
    savings to the local ledger. Returns the child process's exit code.
    """
    import subprocess
    import sys
    import time

    # One stable id for THIS wrap invocation, exported so BOTH the in-process
    # proxy (which tags every ledger record) and the wrapped agent — plus the
    # status line the agent spawns — see the same value and can attribute
    # savings to this exact session. Each terminal's wrap gets its own.
    os.environ.setdefault("DISTIL_SESSION", f"s{int(time.time())}-{os.getpid()}")

    savings = None
    if record:
        from .runtime import RuntimeSavings

        savings = RuntimeSavings(model=pricing_model)
    handler = build_handler(
        upstream,
        lossless_only=lossless_only,
        verbatim=verbatim,
        shape_output=shape_output,
        savings=savings,
        expand=expand,
        session_delta=session_delta,
        shadow_rate=shadow_rate,
    )
    server = QuietHTTPServer((host, 0), handler)  # port 0 → OS picks a free port
    base = f"http://{host}:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()

    child_env = dict(os.environ)
    child_env[env_var] = base
    print(f"distil wrap → proxy {base} (upstream {upstream})")
    print(f"  → {env_var}={base}")
    if lossless_only:
        print("  → lossless-only (no shaping / no tool injection)")
    if verbatim:
        print("  → verbatim (Tier-0 only, no digest)")
    if savings is not None:
        print("  → recording genuine savings → distil leaderboard")
    if shadow_rate and shadow_rate > 0:
        print(
            f"  → shadow-mode live decision-equivalence: sampling "
            f"{shadow_rate * 100:.0f}% (distil shadow-stats)"
        )

    code = 0
    proc_holder: list = []
    _install_sigterm_flush(proc_holder)
    try:
        proc = subprocess.Popen(command, env=child_env)
        proc_holder.append(proc)
        code = proc.wait()
    except FileNotFoundError:
        print(f"distil wrap: command not found: {command[0]}", file=sys.stderr)
        code = 127
    except KeyboardInterrupt:
        code = 130
    finally:
        server.shutdown()
        _drain_shadow(handler)
        if savings is not None:
            savings.flush()  # SIGTERM lands here too — no savings are ever dropped
        server.server_close()
    return code


if __name__ == "__main__":
    serve()
