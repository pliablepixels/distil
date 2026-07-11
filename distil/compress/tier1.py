"""Tier 1 — reversible digest behind a retrieval handle.

Large tool outputs / retrieved docs are replaced by a compact, *decision-aware*
digest plus a content handle. The full original is kept locally and can be
re-expanded on demand (`expand(handle)`), so this is lossless in effect.

The codec is decision-aware: any line the runtime cannot prove irrelevant is
preserved verbatim. The keep rule is a *per-content-type* policy (see
``keep_policy``): the generic net keeps DECISION: markers and error/failure lines,
and each content kind adds its own load-bearing lines (a test log's pass/fail
summary, a traceback's frames, a diff's hunk headers). The dropped span is folded
to a marker that also reports what was dropped by category, so the agent can judge
whether to expand. The *keep* rule is explicit and auditable, not a blind truncation.
"""

from __future__ import annotations

import hashlib

from ..trajectory import Block, Kind
from .base import CompressResult
from .keep_policy import classify, must_keep, summarize_dropped

_DIGESTIBLE = {Kind.TOOL_OUTPUT, Kind.RETRIEVED}


def _handle(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _fold_marker(dropped: list[str], handle: str) -> str:
    """Marker for a folded run: the recoverable handle plus a by-category
    breakdown, which is omitted when nothing flagged (error/warn) was folded."""
    breakdown = summarize_dropped(dropped)
    suffix = f" ({breakdown})" if breakdown else ""
    return f"<< +{len(dropped)} lines omitted{suffix}, handle={handle} >>"


def digest(text: str, head: int = 3, tail: int = 1) -> tuple[str, bool]:
    """Return (digest_text, changed). Keeps head/tail context + every must-keep
    line for the block's content kind, replacing each dropped run with a marker
    that carries the recovery handle and a by-category count of what it hides."""
    lines = text.splitlines()
    if len(lines) <= head + tail + 1:
        return text, False

    kind = classify(text)
    keep_idx = set(range(head)) | set(range(len(lines) - tail, len(lines)))
    keep_idx |= {i for i, ln in enumerate(lines) if must_keep(ln, kind)}

    handle = _handle(text)
    out: list[str] = []
    dropped: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if i in keep_idx:
            if dropped:
                out.append(_fold_marker(dropped, handle))
                dropped = []
            out.append(lines[i])
        else:
            dropped.append(lines[i])
        i += 1
    if dropped:
        out.append(_fold_marker(dropped, handle))
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
