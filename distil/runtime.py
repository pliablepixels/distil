"""Genuine runtime savings — measure what *using* Distil actually saved.

The corpus numbers are a demonstration; this is the real thing. As live traffic
flows through the proxy/gateway, every compressed request's actual token
reduction is accumulated here and periodically flushed to the local savings
ledger — so `distil leaderboard` reflects *your own usage*, not a synthetic
benchmark. Content is never recorded; only token counts and the priced dollar
estimate. Federate the signed aggregate (`distil/telemetry.py`) and the
community number is genuine, verifiable savings too.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from . import ledger, pricing


@dataclass
class RuntimeSavings:
    model: str = "claude-opus-4-8"
    requests: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    ledger_path: Path | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Accept a str path too — the ledger needs a real Path for `.parent`.
        if self.ledger_path is not None and not isinstance(self.ledger_path, Path):
            self.ledger_path = Path(self.ledger_path)

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def dollars_saved(self) -> float:
        # Genuine, conservative floor: tokens saved priced at the base input rate
        # (the cache-aware benefit is larger; this is the defensible minimum).
        return self.tokens_saved * pricing.get(self.model).input

    def record(self, before: int, after: int) -> None:
        with self._lock:
            self.requests += 1
            self.tokens_before += before
            self.tokens_after += after

    def flush(self) -> bool:
        """Persist accumulated genuine savings to the ledger; reset counters.
        Returns False when there is nothing to flush."""
        with self._lock:
            if self.requests == 0:
                return False
            price = pricing.get(self.model)
            ledger.record(
                trajectory_id="live-proxy",
                model=self.model,
                turns=self.requests,
                baseline_dollars=self.tokens_before * price.input,
                distil_dollars=self.tokens_after * price.input,
                baseline_input_tokens=self.tokens_before,
                distil_input_tokens=self.tokens_after,
                path=self.ledger_path or ledger.DEFAULT_PATH,
            )
            self.requests = 0
            self.tokens_before = 0
            self.tokens_after = 0
            return True
