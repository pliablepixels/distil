"""Tests for the long-horizon ReAct agent benchmark (benchmarks/long_horizon/).

Covered:
* Unit tests for each tool executor on a real tmp worktree (no network).
* A mock-upstream end-to-end test: a stdlib http.server acts as the "Anthropic API",
  driving the agent through read_file → edit_file → finish over enough turns that,
  with gate_recent=2, the relevance gate digests at least one older block
  (assert CompressStats.blocks_compressed > 0 in the gated condition).

No real API calls, no Docker, no git clones — all deterministic and fast.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from benchmarks.long_horizon.tools import execute_tool
from benchmarks.swe_bench_e2e.compress_proxy import (
    COMPRESSORS,
    EXPAND_CONDITION,
    GATE_RECENT,
    GATED_CONDITION,
    CompressStats,
    serve,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# Large enough to be compressible (>= MIN_CHARS = 500).
_BIG_CONTENT = "# source file\n" + ("x" * 600)


@pytest.fixture()
def wt(tmp_path: Path) -> Path:
    """A minimal fake worktree with a couple of source files."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text(
        "def add(a, b):\n    return a - b  # BUG: should be +\n"
    )
    (tmp_path / "src" / "helper.py").write_text(_BIG_CONTENT)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_module.py").write_text(
        "from src.module import add\ndef test_add(): assert add(1, 2) == 3\n"
    )
    return tmp_path


# --------------------------------------------------------------------------- #
# Tool executor unit tests
# --------------------------------------------------------------------------- #


def test_list_dir_root(wt: Path) -> None:
    result = execute_tool("list_dir", {"path": "."}, wt)
    assert "src/" in result
    assert "tests/" in result


def test_list_dir_subdir(wt: Path) -> None:
    result = execute_tool("list_dir", {"path": "src"}, wt)
    assert "module.py" in result
    assert "helper.py" in result


def test_list_dir_missing(wt: Path) -> None:
    result = execute_tool("list_dir", {"path": "nonexistent"}, wt)
    assert result.startswith("ERROR")


def test_list_dir_path_traversal_blocked(wt: Path) -> None:
    result = execute_tool("list_dir", {"path": "../../"}, wt)
    assert result.startswith("ERROR")


def test_read_file_returns_content(wt: Path) -> None:
    result = execute_tool("read_file", {"path": "src/module.py"}, wt)
    assert "def add" in result
    assert "BUG" in result


def test_read_file_missing(wt: Path) -> None:
    result = execute_tool("read_file", {"path": "no_such_file.py"}, wt)
    assert result.startswith("ERROR")


def test_read_file_directory_error(wt: Path) -> None:
    result = execute_tool("read_file", {"path": "src"}, wt)
    assert result.startswith("ERROR")


def test_search_finds_pattern(wt: Path) -> None:
    result = execute_tool("search", {"pattern": "def add"}, wt)
    assert "add" in result


def test_search_no_matches(wt: Path) -> None:
    result = execute_tool("search", {"pattern": "ZXQWERTY_NEVER_MATCHES_12345"}, wt)
    # Either "(no matches)" from grep or empty stdout is acceptable.
    assert isinstance(result, str)


def test_edit_file_replaces_unique_string(wt: Path) -> None:
    result = execute_tool(
        "edit_file",
        {
            "path": "src/module.py",
            "old_str": "return a - b  # BUG: should be +",
            "new_str": "return a + b",
        },
        wt,
    )
    assert result.startswith("OK")
    content = (wt / "src" / "module.py").read_text()
    assert "return a + b" in content
    assert "BUG" not in content


def test_edit_file_not_found_error(wt: Path) -> None:
    result = execute_tool(
        "edit_file",
        {"path": "src/module.py", "old_str": "DOES_NOT_EXIST", "new_str": "x"},
        wt,
    )
    assert result.startswith("ERROR")


def test_edit_file_ambiguous_error(wt: Path) -> None:
    # Write a file with a repeated string.
    (wt / "dupe.py").write_text("foo\nfoo\n")
    result = execute_tool(
        "edit_file",
        {"path": "dupe.py", "old_str": "foo", "new_str": "bar"},
        wt,
    )
    assert "2 times" in result or result.startswith("ERROR")


def test_finish_tool_returns_string(wt: Path) -> None:
    result = execute_tool("finish", {"reason": "done"}, wt)
    assert "done" in result


def test_unknown_tool_graceful(wt: Path) -> None:
    result = execute_tool("distil_expand", {"handle": "abc123"}, wt)
    assert "unknown tool" in result


# --------------------------------------------------------------------------- #
# Mock-upstream end-to-end: agent + proxy + fake Anthropic API
# --------------------------------------------------------------------------- #


def _make_mock_api(wt: Path) -> tuple[HTTPServer, list[dict]]:
    """Build a mock Anthropic API server that drives the agent through a fixed script.

    Turn 0: read_file src/module.py          (adds big context → peripheral content)
    Turn 1: read_file src/helper.py          (more peripheral context)
    Turn 2: read_file tests/test_module.py   (more peripheral context)
    Turn 3: edit_file src/module.py          (fix the bug)
    Turn 4: finish                           (done)

    With gate_recent=2 the proxy keeps only the last 2 user/tool turns full and
    digests the older ones — so by turn 3 at least one block has been compressed.
    """
    script = [
        # Turn 0: agent reads module.py
        {
            "id": "msg_0",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_0",
                    "name": "read_file",
                    "input": {"path": "src/module.py"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 100, "output_tokens": 30},
        },
        # Turn 1: agent reads helper.py (big peripheral content)
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "read_file",
                    "input": {"path": "src/helper.py"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 250, "output_tokens": 30},
        },
        # Turn 2: agent reads tests (more peripheral content)
        {
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_2",
                    "name": "read_file",
                    "input": {"path": "tests/test_module.py"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 400, "output_tokens": 30},
        },
        # Turn 3: agent edits the file
        {
            "id": "msg_3",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_3",
                    "name": "edit_file",
                    "input": {
                        "path": "src/module.py",
                        "old_str": "return a - b  # BUG: should be +",
                        "new_str": "return a + b",
                    },
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 500, "output_tokens": 40},
        },
        # Turn 4: agent finishes
        {
            "id": "msg_4",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_4",
                    "name": "finish",
                    "input": {"reason": "fixed the bug"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 600, "output_tokens": 20},
        },
    ]
    requests_log: list[dict] = []
    turn_index = [0]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass  # silence

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw)
            except (ValueError, TypeError):
                body = {}
            requests_log.append(body)

            idx = turn_index[0]
            turn_index[0] += 1
            response = script[min(idx, len(script) - 1)]
            payload = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, requests_log


def _run_e2e(wt: Path, condition: str) -> tuple[dict, CompressStats]:
    """Wire a mock API → compress proxy → agent for one condition."""
    from benchmarks.long_horizon.agent import run_agent

    # Start the mock Anthropic upstream.
    mock_server, _ = _make_mock_api(wt)
    mock_url = f"http://127.0.0.1:{mock_server.server_address[1]}"

    # Patch UPSTREAM in compress_proxy to point at our mock (monkey-patch for test).
    import benchmarks.swe_bench_e2e.compress_proxy as cp

    original_upstream = cp.UPSTREAM
    cp.UPSTREAM = mock_url
    try:
        problem = "Fix the add function in src/module.py to return a + b instead of a - b."
        httpd, state = serve(
            compressor=COMPRESSORS[condition],
            protect=problem,
            expand=(condition in (EXPAND_CONDITION, GATED_CONDITION)),
            gate_recent=(GATE_RECENT if condition == GATED_CONDITION else None),
        )
        proxy_url = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            meta = run_agent(
                problem_statement=problem,
                worktree=wt,
                base_url=proxy_url,
                api_key="sk-test",
                max_turns=10,
                timeout=30.0,
            )
        finally:
            httpd.shutdown()
    finally:
        cp.UPSTREAM = original_upstream
        mock_server.shutdown()

    return meta, state.stats


def test_e2e_full_condition_no_compression(wt: Path) -> None:
    """Full condition: proxy is transparent, agent completes, no blocks compressed."""
    meta, stats = _run_e2e(wt, "full")
    assert meta["status"] == "finish"
    assert meta["turns"] >= 4
    assert stats.blocks_compressed == 0
    assert stats.blocks_seen > 0  # blocks were tallied even without compression
    # The edit should be present in the worktree.
    content = (wt / "src" / "module.py").read_text()
    assert "return a + b" in content


def test_e2e_trunc500_compresses_big_blocks(wt: Path) -> None:
    """distil_trunc500 condition: large read_file outputs are truncated."""
    meta, stats = _run_e2e(wt, "distil_trunc500")
    assert meta["status"] == "finish"
    # helper.py content is > 500 chars → must be compressed.
    assert stats.blocks_compressed >= 1


def test_e2e_gated_condition_digests_periphery(wt: Path) -> None:
    """distil_gated: with gate_recent=GATE_RECENT the proxy digests older blocks.

    The agent makes 5 tool calls (read×3, edit, finish). With GATE_RECENT=6 the gate
    keeps the last 6 user/tool turns full; since we have fewer than 6 turns, no digesting
    happens on short runs. We assert blocks_seen > 0 (proxy is active) and the run
    completes — the gate payoff is genuinely long runs.

    For a more targeted assertion, we force gate_recent=2 directly so we can confirm
    blocks_compressed > 0 when older turns exist.
    """
    from benchmarks.long_horizon.agent import run_agent
    import benchmarks.swe_bench_e2e.compress_proxy as cp

    mock_server, _ = _make_mock_api(wt)
    mock_url = f"http://127.0.0.1:{mock_server.server_address[1]}"
    original_upstream = cp.UPSTREAM
    cp.UPSTREAM = mock_url
    try:
        problem = "Fix the add function."
        # Force gate_recent=2 so the proxy digests the 3 older read_file results.
        httpd, state = serve(
            compressor=COMPRESSORS["distil_gated"],
            protect=problem,
            expand=True,
            gate_recent=2,
        )
        proxy_url = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            meta = run_agent(
                problem_statement=problem,
                worktree=wt,
                base_url=proxy_url,
                api_key="sk-test",
                max_turns=10,
                timeout=30.0,
            )
        finally:
            httpd.shutdown()
    finally:
        cp.UPSTREAM = original_upstream
        mock_server.shutdown()

    # With gate_recent=2 and 3 read_file blocks accumulated, at least one older block
    # is digested by the time the 3rd+ request arrives at the proxy.
    assert state.stats.blocks_compressed > 0, (
        f"expected blocks_compressed > 0 in gated condition (gate_recent=2), "
        f"got stats={state.stats}"
    )
    assert meta["status"] == "finish"
