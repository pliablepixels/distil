"""The learning flywheel — Distil gets better at *your* workload the more you use it.

Every ``distil_expand`` call (see :mod:`distil.expand`) is ground truth that a
digested block was load-bearing — the agent needed the detail back. This module
turns that signal into a policy: learn which *kinds* of content your agents keep
expanding, and stop digesting those kinds, keeping them byte-exact instead.

Why this is safe AND a moat:

* **Never-regressing by construction.** The learned policy only makes Distil *more*
  conservative (digest fewer things). It can lower savings, never lower
  decision-equivalence — so it needs no gate to be safe.
* **Compounding & private.** It learns from coarse, **content-free signatures**
  (e.g. ``json:l``, ``error:m``) — never the content itself — accumulated locally.
  The more your agents run, the better the fit to *your* data; a lossy competitor
  can't build this because it has nothing to expand and no signal to learn from.

The signature is deliberately coarse (content class × size bucket) so the policy
generalizes and the stored stats leak nothing about the actual content.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path


def _state_dir() -> Path:
    """The local state directory, resolved lazily so it honors ``DISTIL_HOME`` at
    call time (configurable deployments; isolated tests). Defaults to ``~/.distil``."""
    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil")))


def _default_stats_path() -> Path:
    return _state_dir() / "expand-stats.json"


# Back-compat eager constant; runtime paths resolve via _default_stats_path().
DEFAULT_STATS_PATH = Path.home() / ".distil" / "expand-stats.json"

_JSON = re.compile(r"^\s*[\[{]")
_ERROR = re.compile(r"Traceback|^\s*File \".*\", line |Exception|Error:", re.MULTILINE)
_LOG = re.compile(r"\d{4}-\d{2}-\d{2}|\b(INFO|DEBUG|WARN|ERROR|TRACE)\b")
_CODE = re.compile(
    r"^\s*(def |class |import |function |const |let |public |private )", re.MULTILINE
)


def _content_class(text: str) -> str:
    head = text[:400]
    if _JSON.match(text):
        return "json"
    if _ERROR.search(head):
        return "error"
    if _CODE.search(head):
        return "code"
    if _LOG.search(head):
        return "log"
    return "prose"


def _size_bucket(text: str) -> str:
    n = text.count("\n") + 1
    return "s" if n < 10 else "m" if n < 40 else "l" if n < 150 else "xl"


def signature(text: str) -> str:
    """A coarse, content-free fingerprint: ``<class>:<size>`` (e.g. ``json:l``).
    Two blocks share a signature if they're the same kind of content at a similar
    scale — the unit the policy generalizes over. Carries no content."""
    return f"{_content_class(text)}:{_size_bucket(text)}"


@dataclass
class ExpandStats:
    """Per-signature counters: how often we digested it, how often the agent then
    had to expand it. The ratio is the learning signal."""

    digested: dict[str, int] = field(default_factory=dict)
    expanded: dict[str, int] = field(default_factory=dict)
    # The proxy mutates these dicts from ThreadingHTTPServer worker threads and
    # serializes them in save(); without the lock, json.dumps racing a mutation
    # raises "dictionary changed size during iteration" → a 500 to the agent.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record_digest(self, sig: str) -> None:
        with self._lock:
            self.digested[sig] = self.digested.get(sig, 0) + 1

    def record_expand(self, sig: str) -> None:
        with self._lock:
            self.expanded[sig] = self.expanded.get(sig, 0) + 1

    def expand_rate(self, sig: str) -> float:
        d = self.digested.get(sig, 0)
        return self.expanded.get(sig, 0) / d if d else 0.0

    def expand_prone(self, *, min_digested: int = 5, threshold: float = 0.25) -> set[str]:
        """Signatures the agent expands often enough that digesting them is a net
        loss — these should be kept byte-exact. Requires ``min_digested`` samples
        so the policy doesn't react to noise."""
        return {
            s
            for s, d in self.digested.items()
            if d >= min_digested and self.expand_rate(s) >= threshold
        }

    # -- persistence (local, content-free, failure-tolerant) ------------------
    @classmethod
    def load(cls, path: Path | None = None) -> ExpandStats:
        try:
            raw = json.loads(Path(path or _default_stats_path()).read_text(encoding="utf-8"))
            return cls(dict(raw.get("digested", {})), dict(raw.get("expanded", {})))
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: Path | None = None) -> None:
        try:
            with self._lock:  # snapshot under lock so concurrent records can't race dumps
                payload = json.dumps({"digested": self.digested, "expanded": self.expanded})
            p = Path(path or _default_stats_path())
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(p)  # atomic swap so a crash can't corrupt the policy
        except Exception:  # noqa: BLE001 — learning must never break the request path
            pass


def keep_predicate(
    stats: ExpandStats | None = None,
    *,
    path: Path | None = None,
    min_digested: int = 5,
    threshold: float = 0.25,
):
    """Return a ``(text) -> bool`` predicate: True means 'keep byte-exact, don't
    digest' because this signature is expand-prone for your workload. Used by the
    compressor to apply the learned policy."""
    stats = stats if stats is not None else ExpandStats.load(path)
    prone = stats.expand_prone(min_digested=min_digested, threshold=threshold)

    def keep(text: str) -> bool:
        return signature(text) in prone

    keep.prone = prone  # type: ignore[attr-defined]  # exposed for inspection
    return keep
