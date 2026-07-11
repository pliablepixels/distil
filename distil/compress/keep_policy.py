"""Per-content-type keep policy for the Tier-1 reversible digest.

The digest folds a verbose block's droppable middle behind a handle. WHICH lines
are droppable depends on the content: a passing test run's value is its pass/fail
summary, a traceback's is its frames, a diff's is its hunk headers. This module
classifies a block and returns the per-type "must keep" rule, so the digest never
drops the line the content is actually about. It also provides the by-category
count that goes in the digest marker, so the agent can judge whether a folded span
is worth expanding.

Model-free and deterministic. The keep rule is biased toward over-keeping: a stray
kept line costs a little compression, a missed one costs the agent the decisive
line. This is the "per-content-type codec" the tier1 docstring anticipated.
"""

from __future__ import annotations

import re
from enum import Enum


class ContentKind(str, Enum):
    GENERIC = "generic"
    LOG = "log"  # test / build / CI console output
    TRACEBACK = "traceback"
    DIFF = "diff"


# GENERIC salience net (unchanged from the original digest): explicit DECISION
# markers plus the lines an agent most often needs verbatim to react.
_GENERIC_RE = re.compile(r"error|exception|traceback|fail|warn|panic|fatal", re.IGNORECASE)

# LOG result-summary lines a runner prints. Anchored on a count next to an outcome
# word, so per-test lines ("(3 tests) 5ms") are NOT kept, only summaries
# ("1955 passed", "2 failed", "3 skipped"), plus common footer labels and exit codes.
_RESULT_RE = re.compile(
    r"\d+\s+(passed|passing|failed|failing|skipped|pending|errored)\b"
    r"|^\s*(Test Files|Tests|Duration|Start at)\b"
    r"|test result:"  # cargo
    r"|=+.*\b(passed|failed|error)\b.*=+"  # pytest summary rule
    r"|\bexit(?:\s|-)?code\b",
    re.IGNORECASE,
)

# TRACEBACK: keep every stack frame so the call site survives, not just the message.
_FRAME_RE = re.compile(r'^\s*(File ".*", line \d+|at\s+\S+\s*\(.*:\d+\)|\S+\.\w+:\d+)')

# DIFF: keep file and hunk headers so the change location survives.
_DIFF_KEEP_RE = re.compile(r"^(diff --git |@@ |\+\+\+ |--- |index )")

# --- classification signals (distinctive markers of each content kind) ---
_LOG_SIGNAL_RE = re.compile(
    r"\d+\s+(passed|failed|skipped)\b|Test Files|test result:|\bPASS\b|\bFAIL\b"
    r"|^RUN\b|npm (run|test)|vitest|jest|pytest|cargo test|go test",
    re.IGNORECASE | re.MULTILINE,
)
_TB_SIGNAL_RE = re.compile(
    r'Traceback \(most recent call last\)|^\s*File ".*", line \d+', re.MULTILINE
)
_DIFF_SIGNAL_RE = re.compile(r"^(diff --git |@@ -\d)", re.MULTILINE)


def classify(text: str) -> ContentKind:
    """Best-effort content kind for a block. A failing test log also contains a
    traceback, so LOG wins over TRACEBACK (its policy still keeps frames via the
    generic error net) so the run summary is pinned too."""
    if _DIFF_SIGNAL_RE.search(text):
        return ContentKind.DIFF
    if _LOG_SIGNAL_RE.search(text):
        return ContentKind.LOG
    if _TB_SIGNAL_RE.search(text):
        return ContentKind.TRACEBACK
    return ContentKind.GENERIC


def must_keep(line: str, kind: ContentKind = ContentKind.GENERIC) -> bool:
    """True if ``line`` must survive the digest for a block of the given ``kind``.
    Every kind inherits the generic net (DECISION markers + error/failure lines)."""
    if "DECISION:" in line or _GENERIC_RE.search(line):
        return True
    if kind is ContentKind.LOG:
        return _RESULT_RE.search(line) is not None
    if kind is ContentKind.TRACEBACK:
        return _FRAME_RE.search(line) is not None
    if kind is ContentKind.DIFF:
        return _DIFF_KEEP_RE.search(line) is not None
    return False


# --- marker category breakdown ---
_ERR_RE = re.compile(r"error|exception|traceback|fatal|panic", re.IGNORECASE)
_WARN_RE = re.compile(r"warn", re.IGNORECASE)


def _category(line: str) -> str:
    if _ERR_RE.search(line):
        return "error"
    if _WARN_RE.search(line):
        return "warn"
    return "other"


def summarize_dropped(lines: list[str]) -> str:
    """By-category breakdown of dropped lines for the digest marker, e.g.
    ``2 error, 1 warn, 918 other``. Returns ``""`` when nothing flagged (no error
    or warn) was folded: the keep policy already surfaces flagged lines, so a
    mundane fold needs no breakdown and the bare line count carries it. Only
    nonzero categories, fixed order."""
    counts = {"error": 0, "warn": 0, "other": 0}
    for ln in lines:
        counts[_category(ln)] += 1
    if not counts["error"] and not counts["warn"]:
        return ""
    return ", ".join(f"{n} {name}" for name in ("error", "warn", "other") if (n := counts[name]))
