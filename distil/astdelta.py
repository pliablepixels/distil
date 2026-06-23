"""AST-structural delta — the deepest layer of cache-delta coding (stdlib only).

When a coding agent re-reads a file after an edit, cache-delta coding sends a delta
instead of the whole file. A *textual* unified diff is line-based, so it explodes on
reformatting, comment churn, or import reordering — noise that changes no decision.
An **AST-structural** delta compares the file's top-level definitions by their parsed
structure, so it isolates exactly the definitions whose *meaning* changed.

The fingerprint is ``ast.dump(node)`` with attributes off — it is invariant to
whitespace, blank lines, comments, and (via set-matching) statement/import ordering.
So a reformat-only change reads as **unchanged** and is referenced, not re-sent;
only definitions whose AST actually changed are sent in full.

Why it is decision-equivalent (the motto): an unchanged definition is still present
verbatim earlier in the (cached) conversation, so a reference carries the same
information the agent needs for its next action; the changed/added definitions are
sent in full. The complete current file is kept locally and recovered byte-exact via
``distil_expand``. Python only (uses the stdlib ``ast``); callers fall back to the
textual delta for other languages or unparseable (mid-edit) source — so it never
fails a request, it just does less.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass
class _Seg:
    kind: str  # "def" | "class" | "stmt"
    name: str  # def/class name, or "" for bare statements
    source: str  # exact source text of the node
    fp: str  # ast.dump fingerprint (formatting/comment-invariant)


def _segments(text: str) -> list[_Seg] | None:
    """Split a Python module into ordered top-level segments, or None if unparseable."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    segs: list[_Seg] = []
    for node in tree.body:
        src = ast.get_source_segment(text, node)
        if src is None:
            return None  # can't recover exact source — let the caller fall back
        fp = ast.dump(node)  # attributes off by default -> position/format invariant
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            segs.append(_Seg("def", node.name, src, fp))
        elif isinstance(node, ast.ClassDef):
            segs.append(_Seg("class", node.name, src, fp))
        else:
            segs.append(_Seg("stmt", "", src, fp))
    return segs


def _label(seg: _Seg) -> str:
    return f"{seg.kind} {seg.name}".strip() if seg.name else seg.kind


def structural_delta(old: str, new: str, base_handle: str, new_handle: str) -> str | None:
    """Return an AST-structural delta marker for *new* vs *old*, or None to fall back.

    None means: not Python, unparseable, or the structural delta isn't smaller than
    re-sending the file — in every case the caller should use the textual delta.
    """
    old_segs = _segments(old)
    new_segs = _segments(new)
    if old_segs is None or new_segs is None:
        return None

    old_named = {(s.kind, s.name): s.fp for s in old_segs if s.name}
    # Bare statements (imports, module code) match by fingerprint, order-insensitive.
    old_stmt_fps: dict[str, int] = {}
    for s in old_segs:
        if not s.name:
            old_stmt_fps[s.fp] = old_stmt_fps.get(s.fp, 0) + 1

    unchanged: list[str] = []
    changed: list[_Seg] = []
    added: list[_Seg] = []
    stmt_used = dict(old_stmt_fps)

    for s in new_segs:
        if s.name:
            prev = old_named.get((s.kind, s.name))
            if prev is None:
                added.append(s)
            elif prev == s.fp:
                unchanged.append(_label(s))
            else:
                changed.append(s)
        else:
            if stmt_used.get(s.fp, 0) > 0:
                stmt_used[s.fp] -= 1
                unchanged.append("module statement")
            else:
                added.append(s)

    new_named = {(s.kind, s.name) for s in new_segs if s.name}
    removed = [f"{k} {n}" for (k, n) in old_named if (k, n) not in new_named]

    # If nothing structural changed, the near-duplicate was reformat/comment-only.
    if not changed and not added and not removed:
        marker = (
            f"«distil-ast base={base_handle} handle={new_handle}» the file you saw as "
            f"{base_handle} was re-read with only formatting/comment changes — no "
            f"definition changed. Call distil_expand with handle={new_handle} for the "
            f"exact current bytes."
        )
        return marker if len(marker) < len(new) else None

    lines = [
        f"«distil-ast base={base_handle} handle={new_handle}» structural delta vs the "
        f"version you saw as {base_handle}:"
    ]
    if unchanged:
        # Collapse repeated "module statement" labels for brevity.
        defs = [u for u in unchanged if u != "module statement"]
        nstmt = sum(1 for u in unchanged if u == "module statement")
        kept = ", ".join(defs) if defs else ""
        if nstmt:
            kept = (kept + ", " if kept else "") + f"{nstmt} module statement(s)"
        lines.append(f"  unchanged (still in context above): {kept}")
    for s in changed:
        lines.append(f"  changed {_label(s)}:\n{s.source}")
    for s in added:
        lines.append(f"  added {_label(s)}:\n{s.source}")
    if removed:
        lines.append(f"  removed: {', '.join(removed)}")
    lines.append(f"Call distil_expand with handle={new_handle} for the full current file.")
    marker = "\n".join(lines)
    return marker if len(marker) < len(new) else None
