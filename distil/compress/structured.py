"""Reversible structured compaction — the token-dense content agents actually
traffic in (JSON arrays of records, tabular tool output) re-encoded into a
compact columnar form that carries the *same information* in far fewer tokens.

Unlike a lossy structural crusher, nothing is discarded: the byte-exact original
is kept in the restore table and is one ``expand(handle)`` away. So this is
reversible in effect — the model sees a smaller, equivalent view; the detail is
never lost. A homogeneous ``[{"id":1,"name":"a","ok":true}, …]`` array of N
records costs the repeated keys + punctuation N times in JSON; the columnar form
states the keys once and lists the values, typically a 40–70% reduction with no
loss of meaning.

Conservative by design: only flat arrays of scalar-valued objects fold, and only
when the result is actually smaller and contains no DECISION marker (so the
deterministic decision signal is never perturbed). Anything else is left for the
existing Tier-0/Tier-1 path.
"""

from __future__ import annotations

import json

_SCALAR = (str, int, float, bool, type(None))
_SEP = "\t"
_HDR = "«"  # « — an unambiguous, rare marker so the compact form is recognisable


def _scalar(v: object) -> bool:
    return isinstance(v, _SCALAR)


def _cell(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def fold(text: str) -> str | None:
    """Columnar-fold a JSON array of homogeneous flat records. Returns the compact
    form, or None if the text isn't a foldable structure or wouldn't shrink."""
    s = text.strip()
    if not (s.startswith("[") and s.endswith("]")) or "DECISION:" in text:
        return None
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, list) or len(obj) < 3:
        return None
    if not all(isinstance(r, dict) and all(_scalar(v) for v in r.values()) for r in obj):
        return None

    # column order: first record's keys, then any extras in first-seen order
    cols: list[str] = []
    for r in obj:
        for k in r:
            if k not in cols:
                cols.append(k)
    # tabs/newlines in any cell would break the columnar layout — bail (rare)
    rows: list[str] = []
    for r in obj:
        cells = [_cell(r.get(c)) for c in cols]
        if any(_SEP in c or "\n" in c for c in cells):
            return None
        rows.append(_SEP.join(cells))

    compact = f"{_HDR}rows={len(obj)} cols={','.join(cols)}{_HDR}\n" + "\n".join(rows)
    return compact if len(compact) < len(s) else None


def is_folded(text: str) -> bool:
    return text.startswith(_HDR)
