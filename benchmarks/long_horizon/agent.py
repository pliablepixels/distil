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
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from benchmarks.long_horizon.tools import TOOL_SCHEMAS, execute_tool

MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0
MAX_TOKENS = 4096
MAX_NUDGES = 4  # how many times to push a prematurely-stopping model to keep working

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


def _mark_cache(messages: list[dict[str, Any]]) -> None:
    """Place a single rolling Anthropic prompt-cache breakpoint on the last content block
    of the conversation (the API caches the whole prefix up to it). Strips any earlier
    breakpoint first so we never exceed the 4-breakpoint limit as the history grows. With
    the cached system prompt this turns each turn's repeated prefix into a 0.1x cache read
    instead of full-price input — the cache-aware lever distil is built around."""
    last_block = None
    for msg in messages:
        c = msg.get("content")
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    b.pop("cache_control", None)
                    last_block = b
    if last_block is not None:
        last_block["cache_control"] = {"type": "ephemeral"}


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
    model: str = MODEL,
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

        # Prompt caching: cache the stable system prompt (large — holds the problem
        # statement) and a rolling breakpoint on the latest turn, so the repeated prefix
        # is billed at the 0.1x cache-read rate instead of full input every turn.
        _mark_cache(messages)
        body: dict[str, Any] = {
            "model": model,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
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


_TOOL_NAMES = {t["name"] for t in TOOL_SCHEMAS}
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def _normalize_tool_calls(structured: list, content: str):
    """Return a list of ``(id, name, args_dict, from_content)`` tool calls.

    Prefers OpenAI structured ``tool_calls``. If there are none, recovers a call that a
    local model spoke as text — a bare ``{"name": …, "arguments": {…}}`` object (optionally
    in a ```json fence or amid prose). ``from_content`` marks the text-recovered path so
    the caller feeds the result back as a user turn (no real tool_call_id to reference)."""
    out = []
    for tc in structured or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments", "{}") or "{}")
        except (ValueError, TypeError):
            args = {}
        out.append(
            (tc.get("id", ""), fn.get("name", ""), args if isinstance(args, dict) else {}, False)
        )
    if out:
        return out
    # Text fallback: find the first JSON object with a known tool name.
    if not content:
        return out
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL) or []
    m = _JSON_OBJ.search(content)
    if m:
        candidates.append(m.group(0))
    for blob in candidates:
        try:
            obj = json.loads(blob)
        except (ValueError, TypeError):
            continue
        name = obj.get("name")
        if name in _TOOL_NAMES:
            args = obj.get("arguments") or obj.get("input") or {}
            return [("", name, args if isinstance(args, dict) else {}, True)]
    return out


def _post_openai(
    url: str, body: dict[str, Any], api_key: str, *, max_retries: int = 6
) -> dict[str, Any]:
    """POST ``body`` as JSON to ``url`` using OpenAI auth headers.

    Retries on HTTP 429 (rate limit, incl. TPM ``type: tokens``) with exponential backoff,
    honoring the ``Retry-After`` header when present. Newly-funded accounts start in a low
    usage tier whose per-minute token budget a long-horizon full-context request can exceed;
    a single request that fits the context window then succeeds once the window resets.
    """
    raw = json.dumps(body).encode()

    def _build() -> urllib.request.Request:
        r = urllib.request.Request(url, data=raw, method="POST")
        r.add_header("Content-Type", "application/json")
        r.add_header("Authorization", f"Bearer {api_key}")
        r.add_header("Content-Length", str(len(raw)))
        return r

    backoff = 5.0
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(_build(), timeout=600) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            payload = e.read()
            if e.code == 429 and attempt < max_retries:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                try:
                    wait = float(retry_after) if retry_after else backoff
                except (ValueError, TypeError):
                    wait = backoff
                wait = min(wait, 60.0)
                print(
                    f"[rate-limit 429] attempt {attempt + 1}/{max_retries}, sleeping {wait:.0f}s",
                    flush=True,
                )
                time.sleep(wait)
                backoff = min(backoff * 2, 60.0)
                continue
            try:
                return json.loads(payload)
            except (ValueError, TypeError):
                return {
                    "error": {"type": "http_error", "message": payload.decode(errors="replace")}
                }


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
    nudges = 0

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

        # Append the full assistant message to history.
        messages.append(assistant_msg)

        # Structured OpenAI tool_calls, OR — for local models (e.g. qwen via Ollama) that
        # emit the call as a JSON blob in `content` instead — recovered from the text.
        structured = assistant_msg.get("tool_calls") or []
        calls = _normalize_tool_calls(
            structured, text_content if isinstance(text_content, str) else ""
        )

        if not calls:
            # No tool call. Weaker (esp. local) models often stop prematurely before
            # editing; nudge them to keep working, up to a small bound, before accepting
            # the turn as final. If an edit has already been made, let it stop.
            made_edit = any("edit_file" in ln for ln in log_lines)
            if nudges < MAX_NUDGES and not made_edit:
                nudges += 1
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have not yet edited a file and called finish. Keep going: "
                            "use the tools to locate the bug, apply the fix with edit_file, "
                            "run the tests, then call finish. Respond with a tool call."
                        ),
                    }
                )
                continue
            status = "end_turn" if finish_reason in ("stop", "") else f"stop:{finish_reason}"
            break

        finished = False
        for tc_id, tool_name, tool_input, from_content in calls:
            result_text = execute_tool(tool_name, tool_input, worktree)
            _log(f"  tool={tool_name} result={result_text[:80]!r}")
            if from_content:
                # The model spoke the call as text (no real tool_call_id) — feed the
                # result back as a user turn so there is no tool_call_id mismatch.
                messages.append(
                    {"role": "user", "content": f"Result of {tool_name}:\n{result_text}"}
                )
            else:
                messages.append({"role": "tool", "tool_call_id": tc_id, "content": result_text})
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
