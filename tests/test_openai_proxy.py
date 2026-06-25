"""Tests for the OpenAI chat-completions compression proxy (compress_proxy_openai).

Covered:
* ``compress_body_openai``: user/tool blocks compressed, system/assistant untouched,
  protect respected, digest populates restore map.
* Full proxy round-trip with a mock upstream: distil_expand tool-call on the first
  response is resolved by the proxy's recovery loop; the final non-tool answer is
  returned to the caller; stats.expansions >= 1.

No real model or network required — the mock upstream is a tiny in-process
stdlib http.server running on an ephemeral port.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


from benchmarks.swe_bench_e2e.compress_proxy import (
    MIN_CHARS,
    CompressStats,
    digest_block,
    trunc_500,
)
from benchmarks.swe_bench_e2e.compress_proxy_openai import (
    compress_body_openai,
    serve,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _big(tag: str) -> str:
    """A string long enough to pass the MIN_CHARS gate."""
    return f"{tag}:" + "A" * (MIN_CHARS + 100)


# --------------------------------------------------------------------------- #
# Unit tests: compress_body_openai
# --------------------------------------------------------------------------- #


class TestCompressBodyOpenai:
    def test_user_string_content_compressed(self):
        big = _big("file")
        body = {"messages": [{"role": "user", "content": big}]}
        stats = CompressStats()
        out = compress_body_openai(body, trunc_500, stats)
        assert out["messages"][0]["content"] == big[:500]
        assert stats.blocks_compressed == 1

    def test_user_text_part_compressed(self):
        big = _big("file")
        body = {"messages": [{"role": "user", "content": [{"type": "text", "text": big}]}]}
        stats = CompressStats()
        out = compress_body_openai(body, trunc_500, stats)
        assert out["messages"][0]["content"][0]["text"] == big[:500]
        assert stats.blocks_compressed == 1

    def test_tool_role_content_compressed(self):
        big = _big("cmd-output")
        body = {
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": big,
                }
            ]
        }
        stats = CompressStats()
        out = compress_body_openai(body, trunc_500, stats)
        assert out["messages"][0]["content"] == big[:500]
        assert stats.blocks_compressed == 1

    def test_system_content_never_compressed(self):
        big = _big("instructions")
        body = {"messages": [{"role": "system", "content": big}]}
        stats = CompressStats()
        out = compress_body_openai(body, trunc_500, stats)
        assert out["messages"][0]["content"] == big  # untouched
        assert stats.blocks_compressed == 0

    def test_assistant_content_never_compressed(self):
        big = _big("reasoning")
        body = {"messages": [{"role": "assistant", "content": [{"type": "text", "text": big}]}]}
        stats = CompressStats()
        out = compress_body_openai(body, trunc_500, stats)
        assert out["messages"][0]["content"][0]["text"] == big
        assert stats.blocks_compressed == 0

    def test_small_blocks_untouched(self):
        body = {"messages": [{"role": "user", "content": "tiny"}]}
        stats = CompressStats()
        out = compress_body_openai(body, trunc_500, stats)
        assert out["messages"][0]["content"] == "tiny"
        assert stats.blocks_compressed == 0

    def test_protect_substring_blocks_compression(self):
        problem = "Fix the authentication bug in the login flow. " * 20
        file_block = _big("source")
        body = {
            "messages": [
                {"role": "user", "content": problem + " extra text"},
                {"role": "user", "content": [{"type": "text", "text": file_block}]},
            ]
        }
        stats = CompressStats()
        out = compress_body_openai(body, trunc_500, stats, protect=problem)
        # problem statement preserved verbatim
        assert out["messages"][0]["content"] == problem + " extra text"
        # other file block compressed
        assert out["messages"][1]["content"][0]["text"] == file_block[:500]
        assert stats.blocks_protected == 1
        assert stats.blocks_compressed == 1

    def test_full_condition_passthrough_counts_stats(self):
        big = _big("file")
        body = {"messages": [{"role": "user", "content": big}]}
        stats = CompressStats()
        out = compress_body_openai(body, None, stats)
        assert out["messages"][0]["content"] == big  # unchanged
        assert stats.blocks_seen == 1
        assert stats.blocks_compressed == 0
        assert stats.tokens_before == stats.tokens_after

    def test_digest_restore_populates_map(self):
        big = _big("file")
        body = {"messages": [{"role": "user", "content": big}]}
        stats = CompressStats()
        restore: dict[str, str] = {}
        out = compress_body_openai(body, None, stats, digest_restore=restore)
        assert len(restore) == 1
        handle = next(iter(restore))
        assert restore[handle] == big
        # The compressed content contains the handle
        assert handle in out["messages"][0]["content"]

    def test_does_not_mutate_input(self):
        big = _big("file")
        body = {"messages": [{"role": "user", "content": big}]}
        compress_body_openai(body, trunc_500, CompressStats())
        assert body["messages"][0]["content"] == big

    def test_mixed_roles_only_user_tool_compressed(self):
        big = _big("data")
        body = {
            "messages": [
                {"role": "system", "content": big},
                {"role": "user", "content": big},
                {"role": "assistant", "content": big},
                {"role": "tool", "tool_call_id": "c1", "content": big},
            ]
        }
        stats = CompressStats()
        out = compress_body_openai(body, trunc_500, stats)
        msgs = out["messages"]
        assert msgs[0]["content"] == big  # system: untouched
        assert msgs[1]["content"] == big[:500]  # user: compressed
        assert msgs[2]["content"] == big  # assistant: untouched
        assert msgs[3]["content"] == big[:500]  # tool: compressed
        assert stats.blocks_compressed == 2


# --------------------------------------------------------------------------- #
# Integration test: distil_expand recovery loop via mock upstream
# --------------------------------------------------------------------------- #


def _make_mock_upstream(tool_call_response: dict[str, Any], final_response: dict[str, Any]):
    """Return an HTTPServer whose first POST reply triggers distil_expand and the
    second returns a plain assistant message — simulating the reversible tier."""

    call_count = {"n": 0}
    lock = threading.Lock()

    class MockHandler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            self.rfile.read(length)  # consume body (we don't inspect it here)
            with lock:
                n = call_count["n"]
                call_count["n"] += 1
            if n == 0:
                payload = json.dumps(tool_call_response).encode()
            else:
                payload = json.dumps(final_response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", 0), MockHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port, call_count


def test_expand_loop_resolves_handle_and_returns_final_answer():
    """Proxy must:
    1. Digest the user block and inject the distil_expand tool.
    2. Forward to the mock upstream (first call → tool_calls response).
    3. Resolve the handle, append assistant + tool messages, re-POST.
    4. Return the second (final) response to the caller.
    5. Report stats.expansions >= 1.
    """
    big = _big("source-file")

    # --- build a valid distil_expand tool-call response ---
    # We need to know the handle before the proxy digests it, so compute it here.
    restore_tmp: dict[str, str] = {}
    digest_block(big, restore_tmp)
    handle = next(iter(restore_tmp))

    tool_call_id = "call_abc123"
    tool_call_response: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": "distil_expand",
                                "arguments": json.dumps({"handle": handle}),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }

    final_response: dict[str, Any] = {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Here is my analysis."},
            }
        ],
        "usage": {"prompt_tokens": 150, "completion_tokens": 30},
    }

    mock_server, mock_port, call_count = _make_mock_upstream(tool_call_response, final_response)
    try:
        upstream = f"http://127.0.0.1:{mock_port}/v1"
        httpd, state = serve(
            compressor=None,
            upstream=upstream,
            expand=True,
        )
        proxy_port = httpd.server_address[1]

        # Send a request with a large user block to the proxy
        request_body = json.dumps(
            {
                "model": "llama3",
                "messages": [{"role": "user", "content": big}],
            }
        ).encode()
        req = __import__("urllib.request", fromlist=["Request", "urlopen"]).Request(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            data=request_body,
            headers={"Content-Type": "application/json"},
        )
        import urllib.request

        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            body = json.loads(resp.read())

        assert status == 200, f"Expected 200, got {status}"

        # The proxy should have looped: the mock was called at least twice
        assert call_count["n"] >= 2, (
            f"Mock upstream called {call_count['n']} times; expected >= 2 (expand loop)"
        )

        # The final response must be the non-tool answer
        choices = body.get("choices", [])
        assert choices, "Expected at least one choice in final response"
        assert choices[0]["message"]["content"] == "Here is my analysis."
        assert choices[0]["finish_reason"] == "stop"

        # Stats must reflect at least one expansion
        assert state.stats.expansions >= 1, (
            f"Expected stats.expansions >= 1, got {state.stats.expansions}"
        )
        assert state.stats.expand_requests >= 1

        httpd.shutdown()
    finally:
        mock_server.shutdown()


def test_distil_expand_tool_injected_and_idempotent():
    """The proxy injects the distil_expand function tool exactly once even if the
    incoming request already carries other tools."""
    import urllib.request as ureq

    # A mock upstream that just returns a normal (non-tool) answer immediately
    normal_response: dict[str, Any] = {
        "id": "chatcmpl-3",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "done"},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    received_bodies: list[dict[str, Any]] = []
    lock = threading.Lock()

    class _InspectHandler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length)
            with lock:
                received_bodies.append(json.loads(raw))
            payload = json.dumps(normal_response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    insp_server = HTTPServer(("127.0.0.1", 0), _InspectHandler)
    insp_port = insp_server.server_address[1]
    insp_t = threading.Thread(target=insp_server.serve_forever, daemon=True)
    insp_t.start()

    try:
        upstream = f"http://127.0.0.1:{insp_port}/v1"
        httpd, _ = serve(
            compressor=None,
            upstream=upstream,
            expand=True,
        )
        proxy_port = httpd.server_address[1]

        # Request already has one user-defined tool
        request_body = json.dumps(
            {
                "model": "llama3",
                "messages": [{"role": "user", "content": _big("file")}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "my_tool", "parameters": {}},
                    }
                ],
            }
        ).encode()
        req = ureq.Request(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            data=request_body,
            headers={"Content-Type": "application/json"},
        )
        with ureq.urlopen(req, timeout=10):
            pass

        httpd.shutdown()

        assert received_bodies, "Mock upstream received no requests"
        forwarded_tools = received_bodies[0].get("tools", [])
        # user tool still present
        names = [(t.get("function") or {}).get("name") for t in forwarded_tools]
        assert "my_tool" in names
        # distil_expand injected exactly once
        assert names.count("distil_expand") == 1

    finally:
        insp_server.shutdown()
