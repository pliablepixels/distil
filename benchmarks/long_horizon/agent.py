"""ReAct coding agent for the long-horizon SWE-bench eval.

A multi-turn tool-use loop that drives ``claude-sonnet-4-6`` (temperature 0)
via raw urllib POSTs to ``{base_url}/v1/messages`` — no Anthropic SDK dependency,
so the proxy URL substitution is the only wiring needed.

The agent receives the SWE-bench problem statement as part of its system prompt and
explores the worktree with the tools defined in :mod:`benchmarks.long_horizon.tools`.
Each read_file call adds large peripheral content to the message history, which is
exactly the long-horizon signal that exercises the relevance gate in the
``distil_gated`` condition.

The proxy may inject a ``distil_expand`` tool into the tools list and handle any
``tool_use`` block for it transparently (before the response reaches this agent),
so this loop never sees ``distil_expand`` calls and must not error on unknown tool names.

Returns a metadata dict ``{status, turns, seconds, log_tail}`` matching the shape
of run_agent.run_aider's return value so the driver can log uniformly.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from benchmarks.long_horizon.tools import TOOL_SCHEMAS, execute_tool

MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0
MAX_TOKENS = 4096

# OpenAI function-tool equivalents of the Anthropic TOOL_SCHEMAS.
# Each Anthropic schema {name, description, input_schema} maps to the OpenAI
# {"type":"function","function":{name, description, parameters}} shape.
OPENAI_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in TOOL_SCHEMAS
]

# System prompt template — the problem statement is appended so it is always present
# in the context (and protected from compression by the proxy's ``protect`` parameter).
_SYSTEM_TEMPLATE = """\
You are an expert software engineer solving a real GitHub issue in an open-source \
Python repository. You have been given a worktree checked out at the base commit \
described below.

Your task:
1. Read and understand the issue.
2. Explore the repository to locate the relevant code (use list_dir, read_file, search).
3. Make the minimal necessary code change to fix the issue (use edit_file).
4. Verify your fix with run_tests.
5. Call finish() when done or when you are confident you cannot resolve the issue.

Work methodically: read the files relevant to the issue before editing. Do not guess \
at file paths — verify them first with list_dir or search.

ISSUE
-----
{problem_statement}
"""


def _post(url: str, body: dict[str, Any], api_key: str) -> dict[str, Any]:
    """POST ``body`` as JSON to ``url`` and return the parsed JSON response."""
    raw = json.dumps(body).encode()
    req = urllib.request.Request(url, data=raw, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("x-api-key", api_key)
    req.add_header("anthropic-version", "2023-06-01")
    req.add_header("Content-Length", str(len(raw)))
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        payload = e.read()
        try:
            return json.loads(payload)
        except (ValueError, TypeError):
            return {"error": {"type": "http_error", "message": payload.decode(errors="replace")}}


def run_agent(
    problem_statement: str,
    worktree: Path,
    base_url: str,
    api_key: str,
    *,
    max_turns: int = 30,
    timeout: float = 900.0,
) -> dict[str, Any]:
    """Run the ReAct loop until finish(), max_turns, timeout, or end_turn.

    Parameters
    ----------
    problem_statement:
        The SWE-bench issue text (task definition). Also passed to the proxy
        as ``protect`` so it is never compressed.
    worktree:
        Absolute path to the git worktree where code changes are made.
    base_url:
        The compression proxy URL (e.g. ``http://127.0.0.1:PORT``). The agent
        POSTs to ``{base_url}/v1/messages``.
    api_key:
        Anthropic API key forwarded in every request.
    max_turns:
        Hard cap on assistant turns to bound cost and runtime.
    timeout:
        Wall-clock seconds budget for the whole run.

    Returns
    -------
    dict with keys ``status``, ``turns``, ``seconds``, ``log_tail``.
    """
    endpoint = f"{base_url}/v1/messages"
    system = _SYSTEM_TEMPLATE.format(problem_statement=problem_statement)

    # Anthropic requires >=1 message; seed the conversation with an explicit kickoff so
    # the first request is well-formed. The task itself lives in the system prompt.
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Begin. Explore the repository with the provided tools, localize the "
                "bug described in the task, edit the necessary file(s), run the tests, "
                "and call finish when the issue is resolved."
            ),
        }
    ]
    log_lines: list[str] = []
    t0 = time.time()
    turns = 0
    status = "max_turns"

    def _log(msg: str) -> None:
        log_lines.append(msg)

    for turn in range(max_turns):
        elapsed = time.time() - t0
        if elapsed >= timeout:
            status = "timeout"
            _log(f"[turn {turn}] timeout after {elapsed:.1f}s")
            break

        body: dict[str, Any] = {
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "tools": TOOL_SCHEMAS,
            "messages": messages,
        }

        resp = _post(endpoint, body, api_key)

        # Surface API-level errors early.
        if "error" in resp and "content" not in resp:
            err_msg = resp["error"].get("message", str(resp["error"]))
            _log(f"[turn {turn}] API error: {err_msg[:300]}")
            status = f"api_error:{resp['error'].get('type', 'unknown')}"
            break

        content: list[dict[str, Any]] = resp.get("content") or []
        stop_reason: str = resp.get("stop_reason", "")
        turns = turn + 1

        # Collect assistant text for logging.
        assistant_text = " | ".join(
            b.get("text", "")[:200]
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
        _log(f"[turn {turn}] stop={stop_reason} text={assistant_text[:120]!r}")

        # Append the full assistant response to history.
        messages.append({"role": "assistant", "content": content})

        if stop_reason == "end_turn":
            status = "end_turn"
            break

        if stop_reason != "tool_use":
            # Unexpected stop reason — treat as done.
            status = f"stop:{stop_reason}"
            break

        # Execute every tool_use block in the response.
        tool_results: list[dict[str, Any]] = []
        finished = False
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_name: str = block.get("name", "")
            tool_id: str = block.get("id", "")
            tool_input: dict = block.get("input") or {}

            result_text = execute_tool(tool_name, tool_input, worktree)
            _log(f"  tool={tool_name} result={result_text[:80]!r}")

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_text,
                }
            )

            if tool_name == "finish":
                finished = True

        # Append tool results as a user turn — the accumulated tool_result content
        # is the large peripheral context that grows the message history and exercises
        # the compression proxy.
        messages.append({"role": "user", "content": tool_results})

        if finished:
            status = "finish"
            break

    elapsed = round(time.time() - t0, 1)
    # Keep the last 3000 chars of log as the tail (mirrors run_aider's log_tail).
    log_tail = "\n".join(log_lines)[-3000:]
    return {
        "status": status,
        "turns": turns,
        "seconds": elapsed,
        "log_tail": log_tail,
    }


def _post_openai(url: str, body: dict[str, Any], api_key: str) -> dict[str, Any]:
    """POST ``body`` as JSON to ``url`` using OpenAI auth headers."""
    raw = json.dumps(body).encode()
    req = urllib.request.Request(url, data=raw, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Length", str(len(raw)))
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        payload = e.read()
        try:
            return json.loads(payload)
        except (ValueError, TypeError):
            return {"error": {"type": "http_error", "message": payload.decode(errors="replace")}}


def run_agent_openai(
    worktree: Path,
    problem_statement: str,
    base_url: str,
    api_key: str,
    *,
    model: str,
    max_turns: int = 30,
    timeout: float = 900.0,
) -> dict[str, Any]:
    """Run the ReAct loop in OpenAI Chat-Completions format.

    Mirrors :func:`run_agent` in structure and return shape, but speaks the
    OpenAI ``/v1/chat/completions`` protocol instead of Anthropic ``/v1/messages``.
    Designed for use with Ollama (or any OpenAI-compatible local server) so the
    long-horizon benchmark can run free with a local model.

    Parameters
    ----------
    worktree:
        Absolute path to the git worktree where code changes are made.
    problem_statement:
        The SWE-bench issue text. Placed in the system prompt and protected from
        compression via the proxy's ``protect`` parameter.
    base_url:
        The compression proxy URL (e.g. ``http://127.0.0.1:PORT``). The agent
        POSTs to ``{base_url}/v1/chat/completions``.
    api_key:
        Forwarded as ``Authorization: Bearer <api_key>`` — any value works for
        Ollama (e.g. ``"ollama"``).
    model:
        Model identifier sent in each request (e.g. ``"qwen2.5-coder:32b"``).
    max_turns:
        Hard cap on assistant turns.
    timeout:
        Wall-clock seconds budget for the whole run.

    Returns
    -------
    dict with keys ``status``, ``turns``, ``seconds``, ``log_tail`` — identical
    shape to :func:`run_agent` so the driver can log uniformly.
    """
    endpoint = f"{base_url}/v1/chat/completions"
    system_text = _SYSTEM_TEMPLATE.format(problem_statement=problem_statement)

    # Seed with a system message (problem statement) and an explicit user kickoff —
    # mirrors the Anthropic loop which seeds a single user "Begin…" message.
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": (
                "Begin. Explore the repository with the provided tools, localize the "
                "bug described in the task, edit the necessary file(s), run the tests, "
                "and call finish when the issue is resolved."
            ),
        },
    ]
    log_lines: list[str] = []
    t0 = time.time()
    turns = 0
    status = "max_turns"

    def _log(msg: str) -> None:
        log_lines.append(msg)

    for turn in range(max_turns):
        elapsed = time.time() - t0
        if elapsed >= timeout:
            status = "timeout"
            _log(f"[turn {turn}] timeout after {elapsed:.1f}s")
            break

        body: dict[str, Any] = {
            "model": model,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "tools": OPENAI_TOOLS,
            "tool_choice": "auto",
            "messages": messages,
        }

        resp = _post_openai(endpoint, body, api_key)

        # Surface API-level errors early.
        if "error" in resp and "choices" not in resp:
            err = resp["error"]
            err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            _log(f"[turn {turn}] API error: {err_msg[:300]}")
            status = (
                f"api_error:{err.get('type', 'unknown') if isinstance(err, dict) else 'unknown'}"
            )
            break

        choices = resp.get("choices") or []
        if not choices:
            _log(f"[turn {turn}] empty choices in response")
            status = "api_error:empty_choices"
            break

        choice = choices[0]
        finish_reason: str = choice.get("finish_reason", "")
        assistant_msg: dict[str, Any] = choice.get("message") or {}
        turns = turn + 1

        # Log assistant text content if present.
        text_content = assistant_msg.get("content") or ""
        if isinstance(text_content, str):
            _log(f"[turn {turn}] finish={finish_reason} text={text_content[:120]!r}")
        else:
            _log(f"[turn {turn}] finish={finish_reason}")

        # Append the full assistant message (with any tool_calls) to history.
        messages.append(assistant_msg)

        if finish_reason == "stop":
            status = "end_turn"
            break

        if finish_reason != "tool_calls":
            # Unexpected finish reason — treat as done.
            status = f"stop:{finish_reason}"
            break

        # Execute every tool call in the response.
        tool_calls: list[dict[str, Any]] = assistant_msg.get("tool_calls") or []
        finished = False
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tc_id: str = tc.get("id", "")
            func: dict[str, Any] = tc.get("function") or {}
            tool_name: str = func.get("name", "")
            raw_args: str = func.get("arguments", "{}")
            try:
                tool_input: dict = json.loads(raw_args)
            except (ValueError, TypeError):
                tool_input = {}

            result_text = execute_tool(tool_name, tool_input, worktree)
            _log(f"  tool={tool_name} result={result_text[:80]!r}")

            # Append one role:"tool" message per call — the accumulated tool
            # results are the large peripheral context that exercises the proxy.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_text,
                }
            )

            if tool_name == "finish":
                finished = True

        if finished:
            status = "finish"
            break

    elapsed = round(time.time() - t0, 1)
    log_tail = "\n".join(log_lines)[-3000:]
    return {
        "status": status,
        "turns": turns,
        "seconds": elapsed,
        "log_tail": log_tail,
    }
