"""Tier 1 — reversible digest behind a retrieval handle.

Large tool outputs / retrieved docs are replaced by a compact, *decision-aware*
digest plus a content handle. The full original is kept locally and can be
re-expanded on demand (`expand(handle)`), so this is lossless in effect.

The codec is decision-aware: any line the runtime cannot prove irrelevant is
preserved verbatim. The keep rules are (1) any line carrying a DECISION: marker,
(2) error/warning/traceback lines an agent reacts to, and (3) the *result* line
of a command — a test/build/exit verdict, the one line that is often the whole
reason the output exists. Rule 3 exists because rules 1–2 alone invert priority
on a passing run: they keep the alarming ERROR stdout and fold the "1955 passed"
verdict. In production this is where a learned per-content-type codec plugs in.
The point is that the *keep* rule is explicit and auditable, not blind truncation.
"""

from __future__ import annotations

import hashlib
import re

from ..trajectory import Block, Kind
from .base import CompressResult

_DIGESTIBLE = {Kind.TOOL_OUTPUT, Kind.RETRIEVED}


def _handle(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


# Heuristic salience net, pending the learned per-content-type salience model the
# module docstring describes. Beyond explicit DECISION: markers, keep the lines an
# agent most often needs verbatim to react: errors, exceptions, tracebacks,
# failures, warnings, panics. Substring + case-insensitive so camelCase exception
# names ("ValueError") and variants ("failed", "errors", "warn") all match — for a
# salience net a stray keep (a rare "terror") only costs a little compression, while
# a miss costs the agent the line, so we deliberately bias toward over-keeping.
_KEEP_RE = re.compile(
    r"error|exception|traceback|fail|warn|panic|fatal",
    re.IGNORECASE,
)

# Result-line net — the single line that carries the OUTCOME of a command, which
# is often the whole reason the output exists (a test run, a build, a script). The
# error/warn net above keeps the *alarming* lines; on its own it inverts priority
# on a passing run, whose verdict ("1955 passed", no error word in it) gets folded
# while the noisy ERROR stdout the run emitted on purpose is kept — the answer
# thrown away, the noise retained. This pins the verdict verbatim. Covers the
# common test runners + build/exit summaries; the learned per-content-type codec
# the module docstring describes would subsume it.
# ponytail: regex covers vitest/jest/pytest/mocha/go/cargo/gradle/maven + exit
# status — the 99% of CI output; add a framework here if one slips through.
_SUMMARY_RE = re.compile(
    r"""
      \b\d+\ +(?:passed|failed|skipped|pending|todo|errors?)\b   # vitest/jest/pytest counts
    | \b\d+\ +(?:passing|failing)\b                              # mocha "1955 passing"
    | \btest\ result:                                            # cargo "test result: ok. 5 passed"
    | ^\s*(?:ok|FAIL|PASS)\b                                     # go package ok/FAIL, bare PASS/FAIL
    | ^\s*---\ +(?:FAIL|PASS|SKIP):                              # go subtest verdicts
    | \bBUILD\ (?:SUCCESS(?:FUL)?|FAIL(?:ED|URE))\b              # gradle/maven
    | \bexit\ (?:code|status)\ +\d+                              # command exit status
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _must_keep(line: str) -> bool:
    return (
        "DECISION:" in line
        or _KEEP_RE.search(line) is not None
        or _SUMMARY_RE.search(line) is not None
    )


def digest(text: str, head: int = 3, tail: int = 1) -> tuple[str, bool]:
    """Return (digest_text, changed). Keeps head/tail context + every must-keep
    line, replacing the dropped middle with a single handle marker."""
    lines = text.splitlines()
    if len(lines) <= head + tail + 1:
        return text, False

    keep_idx = set(range(head)) | set(range(len(lines) - tail, len(lines)))
    keep_idx |= {i for i, ln in enumerate(lines) if _must_keep(ln)}

    out: list[str] = []
    dropped = 0
    i = 0
    n = len(lines)
    while i < n:
        if i in keep_idx:
            if dropped:
                out.append(f"<< +{dropped} lines, handle={_handle(text)} >>")
                dropped = 0
            out.append(lines[i])
        else:
            dropped += 1
        i += 1
    if dropped:
        out.append(f"<< +{dropped} lines, handle={_handle(text)} >>")
    return "\n".join(out), True


class Tier1Reversible:
    tier = 1
    name = "tier1-reversible"

    def __init__(self, min_lines: int = 6) -> None:
        self.min_lines = min_lines

    def compress(self, blocks: list[Block]) -> CompressResult:
        from .structured import fold, template_fold  # local: avoids formatter stripping

        out: list[Block] = []
        restore: dict[str, str] = {}
        for b in blocks:
            if b.kind in _DIGESTIBLE:
                # 1) reversible structured compaction: columnar fold, then template mining
                compact = fold(b.text) or template_fold(b.text)
                if compact is not None:
                    restore[_handle(b.text)] = b.text  # byte-exact original, expandable
                    out.append(b.copy_with(compact))
                    continue
                # 2) otherwise, decision-aware reversible digest for verbose blocks
                if b.text.count("\n") + 1 >= self.min_lines:
                    dtext, changed = digest(b.text)
                    if changed:
                        restore[_handle(b.text)] = b.text
                        out.append(b.copy_with(dtext))
                        continue
            out.append(b)
        return CompressResult(out, restore)
