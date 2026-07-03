"""Outcome-guided compression policy — learn from end-to-end task results.

The expand flywheel (:mod:`distil.learn`) learns from a per-step signal: the
agent asked for a digested block back. This module learns from the signal that
actually matters — the TRAJECTORY outcome (ACON, arXiv 2510.00615: optimize
compression against end-to-end success, not per-step fidelity). Every matched
run contributes evidence: when a task the full context solved fails under
compression, whatever content classes were digested in that trajectory are
suspects; classes that keep showing up in degraded trajectories (relative to
how often they appear in successful ones) get protected — kept byte-exact.

Like the expand policy it composes with, this is never-regressing by
construction: it only makes distil MORE conservative (digest fewer things),
so it can lower savings but never lower fidelity, and it needs no gate to be
safe. All evidence is content-free signatures (``json:l``, ``error:m``) —
never content.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..learn import _state_dir, signature

__all__ = ["OutcomeStats", "record_trajectory_outcome", "signature"]


def _default_path() -> Path:
    return _state_dir() / "outcome-stats.json"


@dataclass
class OutcomeStats:
    """Per-signature counters over matched trajectory outcomes.

    ``degraded[sig]``: trajectories where sig was digested AND the task
    regressed under compression (full succeeded, compressed failed).
    ``ok[sig]``: trajectories where sig was digested and the task did NOT
    regress. The contrast is the learning signal.
    """

    degraded: dict[str, int] = field(default_factory=dict)
    ok: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, sigs: set[str], *, regressed: bool) -> None:
        table = self.degraded if regressed else self.ok
        with self._lock:
            for s in sigs:
                table[s] = table.get(s, 0) + 1

    def regression_rate(self, sig: str) -> float:
        d = self.degraded.get(sig, 0)
        total = d + self.ok.get(sig, 0)
        return d / total if total else 0.0

    def protect_prone(self, *, min_seen: int = 5, threshold: float = 0.3) -> set[str]:
        """Signatures whose digestion co-occurs with end-to-end regressions
        often enough that digesting them is a bad bet. ``min_seen`` guards
        against reacting to noise; the threshold is a rate, so a class that
        appears in many successful trajectories is not penalized for one
        unlucky failure."""
        out: set[str] = set()
        with self._lock:
            for s in set(self.degraded) | set(self.ok):
                d = self.degraded.get(s, 0)
                total = d + self.ok.get(s, 0)
                if total >= min_seen and d / total >= threshold:
                    out.add(s)
        return out

    # -- persistence (local, content-free, failure-tolerant) ------------------
    @classmethod
    def load(cls, path: Path | None = None) -> OutcomeStats:
        try:
            raw = json.loads(Path(path or _default_path()).read_text())
            return cls(dict(raw.get("degraded", {})), dict(raw.get("ok", {})))
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: Path | None = None) -> None:
        try:
            with self._lock:
                payload = json.dumps({"degraded": self.degraded, "ok": self.ok})
            p = Path(path or _default_path())
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(payload)
            tmp.replace(p)
        except Exception:  # noqa: BLE001 — learning must never break anything
            pass

    def keep_predicate(self, *, min_seen: int = 5, threshold: float = 0.3):
        """A ``(text) -> bool`` predicate: True = keep byte-exact. Composes with
        :func:`distil.learn.keep_predicate` (OR the two)."""
        prone = self.protect_prone(min_seen=min_seen, threshold=threshold)

        def keep(text: str) -> bool:
            return signature(text) in prone

        keep.prone = prone  # type: ignore[attr-defined]
        return keep


def record_trajectory_outcome(
    digested_texts: list[str],
    *,
    full_success: bool,
    compressed_success: bool,
    path: Path | None = None,
) -> None:
    """Feed one matched run's evidence into the persistent outcome policy.

    ``digested_texts`` are the original texts distil digested during the
    compressed run (from the restore store) — only their content-free
    signatures are recorded. No-op when the full run also failed: a task the
    agent can't solve anyway teaches nothing about compression.
    """
    if not full_success:
        return
    sigs = {signature(t) for t in digested_texts if t}
    if not sigs:
        return
    stats = OutcomeStats.load(path)
    stats.record(sigs, regressed=not compressed_success)
    stats.save(path)
