"""Proxy round-trips that exercise the shadow-spawn and expand-gate integration
branches (uncovered by unit tests): a real proxy + stub upstream, shadow at 1.0,
and an expand-mode request carrying a handle stub."""

import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from distil.proxy import build_handler

_RESP = json.dumps(
    {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "done"}],
        "model": "claude-test",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }
).encode()


_LAST_BODY: dict[str, bytes] = {}


class _Echo(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("content-length", 0))
        _LAST_BODY["raw"] = self.rfile.read(n)
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(_RESP)))
        self.end_headers()
        self.wfile.write(_RESP)

    def log_message(self, *a):  # silence
        pass


@pytest.fixture()
def proxy_factory():
    up = ThreadingHTTPServer(("127.0.0.1", 0), _Echo)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    servers = [up]

    def make(**kw):
        h = build_handler(f"http://127.0.0.1:{up.server_address[1]}", **kw)
        px = ThreadingHTTPServer(("127.0.0.1", 0), h)
        threading.Thread(target=px.serve_forever, daemon=True).start()
        servers.append(px)
        return px.server_address[1]

    yield make
    for s in servers:
        s.shutdown()


def _post(port, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req).read()


def _digestible():
    return {
        "model": "claude-test",
        "max_tokens": 64,
        "system": "You are a test agent. " * 30,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t",
                        "content": "\n".join(f"log line number {i} here" for i in range(40)),
                    }
                ],
            },
            {"role": "user", "content": "continue"},
            {"role": "user", "content": "and again"},
        ],
    }


def test_shadow_round_trip(proxy_factory):
    # shadow at 1.0 -> every request sampled -> _spawn_shadow replays temp-0 on a bg thread
    port = proxy_factory(shadow_rate=1.0)
    out = _post(port, _digestible())
    assert b"done" in out
    time.sleep(0.6)  # let the background shadow compare run to completion


def test_expand_gate_round_trip(proxy_factory):
    # expand on + a handle stub in the conversation -> _expand_should_intercept True ->
    # tool injected + response buffered + run_expand_loop (echo has no expand call -> returns)
    port = proxy_factory(expand=True)
    payload = {
        "model": "claude-test",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t",
                        "content": "ran suite\n<< +50 lines, handle=abcd1234 >>\nfinished",
                    }
                ],
            },
            {"role": "user", "content": "go"},
        ],
    }
    out = _post(port, payload)
    assert b"done" in out


def test_lossless_only_round_trip(proxy_factory):
    port = proxy_factory(lossless_only=True)
    out = _post(port, _digestible())
    assert b"done" in out


def test_expand_overrides_lossless_only(proxy_factory):
    # issue #28: an explicit --expand runs the recoverable digest even under
    # lossless-only (subscription) — the injected distil_expand makes stubs
    # recoverable, so the verbatim force no longer applies. Without --expand,
    # lossless-only stays verbatim (the safe default).
    port_expand = proxy_factory(lossless_only=True, expand=True)
    _post(port_expand, _digestible())
    body_expand = _LAST_BODY["raw"].decode()

    port_plain = proxy_factory(lossless_only=True, expand=False)
    _post(port_plain, _digestible())
    body_plain = _LAST_BODY["raw"].decode()

    # The discriminator is the recovery handle: expand's Tier-1 digest emits a
    # handle-bearing stub (recoverable via distil_expand); lossless-only (even with
    # the #24 columnar fold) emits none. So handle= present <=> the recoverable
    # digest ran, which is exactly what --expand must enable here.
    assert "handle=" not in body_plain, "lossless-only alone emits no recovery handle"
    assert "handle=" in body_expand, (
        "explicit --expand must run the recoverable digest even on lossless-only (#28)"
    )


def test_session_delta_round_trip(proxy_factory):
    # session-delta encodes the request against the prior turn -> exercises the
    # cachedelta encode path + cache-* extras (the second post deltas against the first)
    port = proxy_factory(session_delta=True)
    payload = _digestible()
    _post(port, payload)  # establishes the session
    out = _post(port, payload)  # deltas against prior turn
    assert b"done" in out
