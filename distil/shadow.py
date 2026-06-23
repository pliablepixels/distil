"""Shadow-mode live decision-equivalence — continuous, on real traffic.

The certificate (``distil conformal``) proves decision-equivalence *offline*, on a
calibration corpus. Shadow mode closes the loop *online*: it samples a fraction of
live requests, runs the decision BOTH on the compressed and the uncompressed
context, compares the agent's chosen action, and records a content-free
equivalence signal. You get a rolling, live decision-change rate on your own
production traffic — the thing periodic re-certification can only approximate.

Design constraints (this is in the request path):
  * **Never blocks the user.** The shadow (second, uncompressed) call runs in a
    background thread; the client gets the compressed response immediately.
  * **Sampled.** Only ``rate`` of requests are shadowed, so the cost overhead is
    ``rate`` (e.g. 5%), not 2x.
  * **Content-free.** The ledger stores only a decision *signature* and an
    ``equivalent`` bool — never prompt or response content (same privacy posture
    as the savings ledger / telemetry).

The "decision" is the agent's next action: the first ``tool_use`` block (Anthropic),
``tool_call`` (OpenAI), or ``functionCall`` (Gemini). Two responses are decision-
equivalent iff that action matches — exactly the ``{action, target}`` fingerprint
the certificate uses.

Streaming-aware: real agent sessions (Claude Code, Codex, the Gemini CLI) stream
their responses (SSE), so the decision must be reconstructed from the stream.
:func:`decision_signature_from_body` reads a non-streaming JSON body directly and
reconstructs a streamed (SSE / chunk-array) one via :func:`_decision_from_chunks`,
yielding the same signature either way.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _state_dir() -> Path:
    import os

    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))


def _canon(obj: Any) -> str:
    """A short, stable hash of a JSON-able object — content-free in the ledger."""
    try:
        blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        blob = str(obj)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def decision_signature(resp_json: Any) -> str:
    """A content-free signature of the agent's chosen next action.

    ``tool:<hash>`` when the model called a tool (the decision that matters for an
    agent), ``text`` when it answered without acting, ``none`` when no decision
    could be read. Two responses are decision-equivalent iff their signatures match.
    """
    if not isinstance(resp_json, dict):
        return "none"

    # Anthropic Messages API
    content = resp_json.get("content")
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                return "tool:" + _canon({"name": b.get("name"), "input": b.get("input")})
        return "text"  # answered without calling a tool

    # OpenAI Chat Completions
    choices = resp_json.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        tcs = msg.get("tool_calls")
        if isinstance(tcs, list) and tcs and isinstance(tcs[0], dict):
            fn = tcs[0].get("function") or {}
            return "tool:" + _canon({"name": fn.get("name"), "arguments": fn.get("arguments")})
        return "text"

    # Gemini generateContent
    candidates = resp_json.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        content = candidates[0].get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if isinstance(parts, list):
            for p in parts:
                if isinstance(p, dict) and isinstance(p.get("functionCall"), dict):
                    fc = p["functionCall"]
                    return "tool:" + _canon({"name": fc.get("name"), "args": fc.get("args")})
        return "text"  # responded without calling a function

    return "none"


def _decision_from_chunks(chunks: list[Any]) -> str:
    """Reconstruct the decision signature from a sequence of *streaming* chunks.

    Handles all three providers' streaming shapes by accumulating the first tool
    call across chunks, so the signature matches the non-streaming
    :func:`decision_signature` form exactly:

    * Anthropic SSE — ``content_block_start`` (tool_use name) + ``input_json_delta``
      fragments accumulated into the input object.
    * OpenAI SSE — ``choices[].delta.tool_calls[].function`` name + concatenated
      ``arguments`` string.
    * Gemini ``streamGenerateContent`` — ``candidates[].content.parts[].functionCall``.
    """
    a_name = None
    a_buf = ""
    a_tool = False
    a_text = False
    o_name = None
    o_args = ""
    o_tool = False
    o_text = False
    g_call = None
    g_text = False

    for ch in chunks:
        if not isinstance(ch, dict):
            continue

        # Anthropic streaming events
        ctype = ch.get("type")
        if ctype == "content_block_start":
            cb = ch.get("content_block") or {}
            if cb.get("type") == "tool_use" and not a_tool:
                a_tool = True
                a_name = cb.get("name")
                if isinstance(cb.get("input"), dict) and cb["input"]:
                    a_buf = json.dumps(cb["input"])
            elif cb.get("type") == "text":
                a_text = True
        elif ctype == "content_block_delta":
            delta = ch.get("delta") or {}
            if delta.get("type") == "input_json_delta" and a_tool:
                a_buf += delta.get("partial_json") or ""
            elif delta.get("type") == "text_delta":
                a_text = True

        # OpenAI streaming deltas
        choices = ch.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta") or {}
            tcs = delta.get("tool_calls")
            if isinstance(tcs, list) and tcs and isinstance(tcs[0], dict):
                o_tool = True
                fn = tcs[0].get("function") or {}
                if fn.get("name"):
                    o_name = o_name or fn["name"]
                if fn.get("arguments"):
                    o_args += fn["arguments"]
            elif delta.get("content"):
                o_text = True

        # Gemini streaming chunks
        cands = ch.get("candidates")
        if isinstance(cands, list) and cands and isinstance(cands[0], dict):
            content = cands[0].get("content") or {}
            for p in content.get("parts") or []:
                if isinstance(p, dict):
                    if isinstance(p.get("functionCall"), dict) and g_call is None:
                        g_call = p["functionCall"]
                    elif isinstance(p.get("text"), str):
                        g_text = True

    if a_tool:
        try:
            inp = json.loads(a_buf) if a_buf.strip() else {}
        except (ValueError, TypeError):
            inp = {}
        return "tool:" + _canon({"name": a_name, "input": inp})
    if o_tool:
        return "tool:" + _canon({"name": o_name, "arguments": o_args})
    if g_call is not None:
        return "tool:" + _canon({"name": g_call.get("name"), "args": g_call.get("args")})
    if a_text or o_text or g_text:
        return "text"
    return "none"


def _sse_payloads(text: str) -> list[Any]:
    """Extract the JSON ``data:`` payloads from an SSE stream (skipping ``[DONE]``)."""
    out: list[Any] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            out.append(json.loads(payload))
        except (ValueError, TypeError):
            continue
    return out


def decision_signature_from_body(raw: Any) -> str:
    """Decision signature for a raw response body — JSON, SSE stream, or chunk array.

    This is what makes shadow-mode work on **streaming** sessions (Claude Code,
    Codex, Gemini CLI all stream): a non-streaming JSON body is read directly; an
    SSE stream or a JSON array of chunks is reconstructed via
    :func:`_decision_from_chunks`. Returns the same ``tool:``/``text``/``none``
    signature as :func:`decision_signature`, so streamed and non-streamed responses
    compare correctly.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    if not isinstance(raw, str):
        return decision_signature(raw)
    raw = raw.strip()
    if not raw:
        return "none"
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return _decision_from_chunks(_sse_payloads(raw))
    if isinstance(obj, dict):
        return decision_signature(obj)
    if isinstance(obj, list):
        return _decision_from_chunks(obj)
    return "none"


class ShadowSampler:
    """Deterministic 1-in-N sampling (even, testable, thread-safe). ``rate`` in
    (0,1]; rate<=0 disables shadowing."""

    def __init__(self, rate: float) -> None:
        self.rate = max(0.0, min(1.0, rate))
        self._stride = int(round(1.0 / self.rate)) if self.rate > 0 else 0
        self._n = 0
        self._lock = threading.Lock()

    def should_sample(self) -> bool:
        if self._stride <= 0:
            return False
        with self._lock:
            self._n += 1
            return self._n % self._stride == 0


@dataclass
class ShadowLedger:
    """Rolling, content-free live decision-equivalence stats."""

    window: int = 1000
    samples: int = 0
    changes: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=1000))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, equivalent: bool, *, path: Path | None = None) -> None:
        with self._lock:
            self.samples += 1
            if not equivalent:
                self.changes += 1
            self.recent.append(1 if equivalent else 0)
        self._append(equivalent, path)

    def rate(self) -> float:
        """Live decision-CHANGE rate over the rolling window (0.0 = fully equivalent)."""
        with self._lock:
            if not self.recent:
                return 0.0
            return 1.0 - (sum(self.recent) / len(self.recent))

    def _append(self, equivalent: bool, path: Path | None) -> None:
        try:
            p = path or (_state_dir() / "shadow.jsonl")
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a") as f:
                f.write(json.dumps({"equivalent": bool(equivalent), "ts": time.time()}) + "\n")
        except OSError:
            pass  # telemetry must never break the request path

    @classmethod
    def load(cls, path: Path | None = None) -> ShadowLedger:
        led = cls()
        try:
            p = path or (_state_dir() / "shadow.jsonl")
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                eq = bool(rec.get("equivalent", True))
                led.samples += 1
                if not eq:
                    led.changes += 1
                led.recent.append(1 if eq else 0)
        except OSError:
            pass
        return led


def compare_decisions(compressed_resp: Any, original_resp: Any) -> bool:
    """True iff the agent made the same decision with and without compression."""
    return decision_signature(compressed_resp) == decision_signature(original_resp)
