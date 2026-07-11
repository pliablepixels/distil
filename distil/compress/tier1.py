"""Tier 1 — reversible digest behind a retrieval handle.

Large tool outputs / retrieved docs are replaced by a compact, *decision-aware*
digest plus a content handle. The full original is kept locally and can be
re-expanded on demand (`expand(handle)`), so this is lossless in effect.

The codec is decision-aware: any line the runtime cannot prove irrelevant is
preserved verbatim. The keep rules are (1) any line carrying a DECISION: marker,
(2) error/warning/traceback lines an agent reacts to — deduped, so a loop that
logs the same error 400× keeps its first occurrences plus a fold, not 400 copies —
and (3) the *result* line of a command — a test/build/exit verdict, the one line
that is often the whole reason the output exists. Rule 3 exists because rules 1–2
alone invert priority on a passing run: they keep the alarming ERROR stdout and
fold the "1955 passed" verdict. In production this is where a learned
per-content-type codec plugs in. The point is that the *keep* rule is explicit
and auditable, not blind truncation.
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


# Noise dedup — a run that logs the same ERROR on purpose in a loop produces
# hundreds of near-identical lines that differ only in an index/id/timestamp.
# Keeping them all floods the digest with noise (the second half of the
# "kept the noise, folded the answer" inversion). Normalize away the varying
# numerics to get the line's *shape*; the first few occurrences of a shape are
# kept as the signal "this error happens", the rest fold into the existing
# handle markers (recoverable like any folded line). DECISION: and verdict
# lines are exempt — they are answers, not noise.
_NUM_RE = re.compile(r"0x[0-9a-fA-F]+|\d+")


def _shape(line: str) -> str:
    return _NUM_RE.sub("#", " ".join(line.split())).lower()


# Outcome-aware routing — the first content-type profile. When the log's own
# verdict says GREEN (tests passed / build succeeded, nothing failed), the
# ERROR/WARN stdout is noise the SUT logged on purpose: by definition it did
# not fail the run, so one sample per shape is enough signal. A red or unknown
# outcome keeps the cautious default — there those errors may BE the answer.
# Detection reads only verdict lines (the trustworthy part of the log), and
# everything folded stays recoverable behind the handle.
_RED_RE = re.compile(
    r"""
      \b[1-9]\d*\ +(?:failed|failing|errors?)\b    # non-zero fail/error counts
    | ^\s*---\ +FAIL:                              # go subtest failure
    | ^\s*FAIL\b                                   # go package FAIL / bare FAIL
    | \bBUILD\ FAIL(?:ED|URE)\b                    # gradle/maven
    | \bexit\ (?:code|status)\ +[1-9]              # non-zero exit
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _outcome(lines: list[str]) -> str:
    """'green' | 'red' | 'unknown', judged from verdict lines only."""
    saw_verdict = False
    for ln in lines:
        if _SUMMARY_RE.search(ln) is None:
            continue
        saw_verdict = True
        if _RED_RE.search(ln) is not None:
            return "red"
    return "green" if saw_verdict else "unknown"


def digest(
    text: str, head: int = 3, tail: int = 1, max_repeats: int | None = None
) -> tuple[str, bool]:
    """Return (digest_text, changed). Keeps head/tail context + every must-keep
    line, replacing the dropped middle with a single handle marker. Near-identical
    error/warn repeats beyond `max_repeats` per shape fold with the rest; the
    default routes by the log's own outcome (green run -> 1 sample per shape,
    red/unknown -> 2)."""
    lines = text.splitlines()
    if len(lines) <= head + tail + 1:
        return text, False
    if max_repeats is None:
        max_repeats = 1 if _outcome(lines) == "green" else 2

    keep_idx = set(range(head)) | set(range(len(lines) - tail, len(lines)))
    shape_seen: dict[str, int] = {}
    for i, ln in enumerate(lines):
        if "DECISION:" in ln or _SUMMARY_RE.search(ln) is not None:
            keep_idx.add(i)  # answers — never deduped
        elif _KEEP_RE.search(ln) is not None:
            s = _shape(ln)
            shape_seen[s] = shape_seen.get(s, 0) + 1
            if shape_seen[s] <= max_repeats:
                keep_idx.add(i)

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
