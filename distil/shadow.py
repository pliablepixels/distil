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

The "decision" is the agent's next action: the first ``tool_use`` block (Anthropic)
or ``tool_call`` (OpenAI). Two responses are decision-equivalent iff that action
matches — exactly the ``{action, target}`` fingerprint the certificate uses.
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
