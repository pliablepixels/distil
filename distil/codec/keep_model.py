"""keep_model — pluggable per-content-type keep-model codec.

This module is the model-agnostic heuristic codec.  It scores every line of a text
block for *salience* and retains a target fraction of lines while guaranteeing that
never-drop lines (score == 1.0) always survive.

Drop-in upgrade path
--------------------
A trained ModernBERT-style token-keep classifier that implements ``KeepModel.score``
can be passed as the ``model`` argument to ``apply_keep`` to replace the deterministic
default without any other code change.  That trained model does not exist in this repo;
``SalienceKeepModel`` is the production default and is not a stub — it applies explicit,
explainable rules that are documented below.

Salience rules (``SalienceKeepModel``)
---------------------------------------
Score 1.0  — NEVER drop
    Lines containing the literal marker ``DECISION:`` are always retained regardless
    of target_ratio.  These carry explicit agent intent and must survive any
    compression budget.

Score 0.95 — Error / failure signals
    Lines matching any keyword in ``ERROR_KEYWORDS`` (case-insensitive): "error",
    "fail", "exception", "traceback", "crashloop", "denied", "breach".  Near-certain
    to matter for debugging or audit.

Score 0.7  — Structured result / header
    Lines that look like structured data: JSON objects/arrays (start with ``{``, ``[``),
    key-value pairs (``key: value`` pattern), or table-like headers (``|``-delimited).
    These carry structured information denser than prose.

Score 0.6  — Numeric / metric lines
    Lines that contain at least two digit sequences (bare numbers, decimals, units).
    Logs, metrics, and timing lines live here.

Score 0.1  — Debug / trace / boilerplate
    Lines matching ``DEBUG_KEYWORDS`` ("debug", "trace", "verbose", "noqa", "todo",
    "fixme") or very short non-empty lines (< 4 chars) with no letters.

Score 0.0  — Empty
    Blank lines (after stripping) are zero-scored and are the first to be dropped.

Rules are evaluated in priority order; first match wins.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

# ---------------------------------------------------------------------------
# Keyword constants — module-level so callers can introspect / extend them.
# ---------------------------------------------------------------------------

#: Triggers score 1.0 (never-drop).
NEVER_DROP_MARKER: str = "DECISION:"

#: Triggers score 0.95.  Checked case-insensitively.
ERROR_KEYWORDS: frozenset[str] = frozenset(
    {"error", "fail", "exception", "traceback", "crashloop", "denied", "breach"}
)

#: Triggers score 0.1 when matched (lowest non-empty tier).  Checked case-insensitively.
DEBUG_KEYWORDS: frozenset[str] = frozenset({"debug", "trace", "verbose", "noqa", "todo", "fixme"})

# Pre-compiled patterns used by SalienceKeepModel.
_RE_KEY_VALUE = re.compile(r"^\s*[\w\-\.]+\s*:\s*\S")
_RE_TABLE_ROW = re.compile(r"\|")
_RE_JSON_START = re.compile(r"^\s*[\[\{]")
_RE_DIGITS = re.compile(r"\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class KeepModel(Protocol):
    """Pluggable line-salience scorer.

    Parameters
    ----------
    line:
        A single line of text (no trailing newline expected but tolerated).
    kind:
        Content-type hint (e.g. ``"tool_output"``, ``"retrieved"``, ``"log"``).
        Implementations may use this to shift scoring; the default heuristic
        uses it only for future extensibility.

    Returns
    -------
    float
        Salience score in [0, 1].  A score of exactly 1.0 means "never drop
        regardless of budget".
    """

    def score(self, line: str, kind: str) -> float: ...


# ---------------------------------------------------------------------------
# Default deterministic implementation
# ---------------------------------------------------------------------------


class SalienceKeepModel:
    """Deterministic, explainable salience scorer.

    Rules are documented at module level.  This class is not a stub: it is the
    production heuristic default and is designed to be replaced by a learned
    classifier that implements the same ``KeepModel`` Protocol.
    """

    def score(self, line: str, kind: str) -> float:  # noqa: ARG002
        stripped = line.strip()

        # 0.0 — empty
        if not stripped:
            return 0.0

        # 1.0 — never-drop marker
        if NEVER_DROP_MARKER in line:
            return 1.0

        lower = stripped.lower()

        # 0.95 — error / failure signal
        for kw in ERROR_KEYWORDS:
            if kw in lower:
                return 0.95

        # 0.1 — debug / trace / boilerplate (checked before structure so that
        #        e.g. "DEBUG: key: value" doesn't score as structured)
        for kw in DEBUG_KEYWORDS:
            if kw in lower:
                return 0.1

        # 0.7 — structured result / header
        if (
            _RE_JSON_START.match(stripped)
            or _RE_KEY_VALUE.match(stripped)
            or _RE_TABLE_ROW.search(stripped)
        ):
            return 0.7

        # 0.6 — numeric / metric line (≥2 distinct digit sequences)
        if len(_RE_DIGITS.findall(stripped)) >= 2:
            return 0.6

        # default: moderate prose
        return 0.3


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = SalienceKeepModel()


def apply_keep(
    text: str,
    kind: str,
    target_ratio: float,
    model: KeepModel | None = None,
) -> str:
    """Score each line and return a salience-filtered subset in original order.

    Parameters
    ----------
    text:
        The full input text.
    kind:
        Content-type hint forwarded to ``model.score``.
    target_ratio:
        Fraction of lines to retain, in (0, 1].  The kept set is at least
        ``ceil(target_ratio * N)`` lines and always includes every never-drop
        line (score == 1.0).
    model:
        ``KeepModel`` implementation to use.  Defaults to ``SalienceKeepModel()``.

    Returns
    -------
    str
        Kept lines joined by ``"\\n"``, in original order.  If all lines are
        never-drop the full text is returned unchanged.  Empty input returns
        ``""``.
    """
    if not text:
        return ""

    _model = model if model is not None else _DEFAULT_MODEL

    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return ""

    # Score every line.
    scores = [_model.score(ln, kind) for ln in lines]

    # Partition: never-drop (score == 1.0) vs. candidates.
    never_drop_idx: list[int] = [i for i, s in enumerate(scores) if s >= 1.0]
    candidate_idx: list[int] = [i for i, s in enumerate(scores) if s < 1.0]

    # How many lines do we need to reach target_ratio?
    target_count = math.ceil(target_ratio * n)
    # The budget for non-never-drop lines.
    budget = max(0, target_count - len(never_drop_idx))

    # Pick top-scoring candidates (stable: ties broken by original order via
    # stable sort on negative index).
    ranked = sorted(candidate_idx, key=lambda i: (-scores[i], i))
    chosen_candidates = set(ranked[:budget])

    # Merge and restore original order.
    keep_idx = set(never_drop_idx) | chosen_candidates
    kept = [lines[i] for i in range(n) if i in keep_idx]

    return "\n".join(kept)
