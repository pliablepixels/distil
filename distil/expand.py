"""Transparent agent-callable expansion — the recoverable-compression moat.

Every other compressor (Headroom, LLMLingua, RTK, plain truncation/summary) is
*lossy*: once a tool output is crushed, the detail is gone. Distil DIGESTS behind
a content handle and keeps the original locally. This module turns that into a
capability competitors structurally cannot match:

  1. It injects a ``distil_expand`` tool into the request, so the model knows it can
     recover any digested block by its handle.
  2. It runs a server-side loop: when the model asks to expand, Distil resolves the
     handle from the local store and re-queries — invisibly. The agent receives one
     normal response; the recovery round-trips never reach it.

Two consequences that change the product, not just the numbers:

  * **Digest fearlessly.** You can compress far more aggressively, because nothing is
    lost — the safety net is the model pulling detail back itself. The dangerous
    failure mode of lossy compression ("it dropped something load-bearing") is gone.
  * **Every expand is a label.** A ``distil_expand`` call is ground truth that the
    digested content was actually needed. Logged, these train the keep-model to stop
    digesting what *your* workload depends on — a compounding, usage-driven moat that
    a lossy tool can't build because it has nothing to expand and no signal to learn
    from.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

EXPAND_TOOL_NAME = "distil_expand"

# Anthropic Messages API tool spec. The proxy injects this so the model can recover.
EXPAND_TOOL: dict[str, Any] = {
    "name": EXPAND_TOOL_NAME,
    "description": (
        "Recover the full original content of a context block that Distil digested to "
        "save tokens. A digested block carries a marker containing 'handle=XXXXXXXX' — "
        "e.g. '<< +N lines, handle=XXXXXXXX >>', a «… handle=XXXXXXXX» columnar/template "
        "marker, or '<<distil elided, handle=XXXXXXXX>>' for a skeletonized block. Call "
        "this with that handle whenever you need detail that was elided to decide."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"handle": {"type": "string", "description": "the 8-char content handle"}},
        "required": ["handle"],
    },
}

# Where expand events are logged — the learning signal / moat data. Content-free by
# default (handle + length only); the proxy can opt into storing the recovered text.
# Resolved lazily so it honors DISTIL_HOME at call time (configurable / isolated tests).
DEFAULT_SIGNAL_PATH = Path.home() / ".distil" / "expand-signals.jsonl"


def _default_signal_path() -> Path:
    import os

    return (
        Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil"))) / "expand-signals.jsonl"
    )


def inject_expand_tool(body: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the request body with the distil_expand tool available."""
    tools = body.get("tools")
    if isinstance(tools, list):
        if any(isinstance(t, dict) and t.get("name") == EXPAND_TOOL_NAME for t in tools):
            return body
        return {**body, "tools": [*tools, EXPAND_TOOL]}
    return {**body, "tools": [EXPAND_TOOL]}


def _expand_calls(resp: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        b
        for b in (resp.get("content") or [])
        if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == EXPAND_TOOL_NAME
    ]


def record_signal(handle: str, original: str, *, path: Path | None = None) -> None:
    """Append a content-free expand event — the label the keep-model learns from.
    Only the handle and recovered length are written; never the content itself."""
    try:
        path = path or _default_signal_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(
                json.dumps({"handle": handle, "recovered_chars": len(original), "ts": time.time()})
                + "\n"
            )
    except OSError:
        pass  # logging the moat signal must never break the request path


def resolve_expands(
    resp: dict[str, Any],
    store: Any,
    *,
    on_signal: Callable[[str, str], None] | None = record_signal,
) -> list[dict[str, Any]] | None:
    """If *resp* contains distil_expand tool calls, resolve each handle against the
    RestoreStore and return the tool_result blocks to feed back. Else None."""
    calls = _expand_calls(resp)
    if not calls:
        return None
    results: list[dict[str, Any]] = []
    for c in calls:
        handle = str((c.get("input") or {}).get("handle", "")).strip()
        try:
            original = store.expand(handle)
        except Exception:  # noqa: BLE001 — unknown/expired handle must not 500 the agent
            original = f"[distil: no original found for handle {handle!r}]"
        if on_signal is not None:
            on_signal(handle, original)
        results.append({"type": "tool_result", "tool_use_id": c.get("id"), "content": original})
    return results


def run_expand_loop(
    body: dict[str, Any],
    first_response: dict[str, Any],
    store: Any,
    post: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    max_iters: int = 4,
    on_signal: Callable[[str, str], None] | None = record_signal,
) -> dict[str, Any]:
    """Server-side recovery loop. While the model asks to expand, resolve the handles
    and re-query ``post`` with the recovered content appended — transparently to the
    agent. Returns the final (non-expand) response. ``post(body) -> parsed response``.

    The loop is bounded by ``max_iters`` so a misbehaving model can't spin forever;
    after the cap, the latest response is returned as-is.
    """
    resp = first_response
    messages = list(body.get("messages") or [])
    for _ in range(max_iters):
        results = resolve_expands(resp, store, on_signal=on_signal)
        if results is None:
            return resp
        messages = [
            *messages,
            {"role": "assistant", "content": resp.get("content", [])},
            {"role": "user", "content": results},
        ]
        resp = post({**body, "messages": messages})
    return resp
