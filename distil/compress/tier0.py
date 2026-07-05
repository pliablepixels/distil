"""Tier 0 — provably lossless transforms.

Every transform here is reconstructable: minified JSON re-expands to the same
object; run-length-collapsed lines re-expand from the count markers. We still
record the original in `restore` so reversibility is verifiable in tests.
"""

from __future__ import annotations

import json
import re

from ..trajectory import Block
from .base import CompressResult

_RLE = re.compile(r"^(.*)\n(?:\1\n)+", re.MULTILINE)
# A generated run marker; guard against collapsing content that already looks like one.
_RUN_MARKER = re.compile(r"^<<x\d+>>$")


def _reject_nonfinite(n: str) -> float:
    """parse_float hook: reject overflow/NaN floats (1e400 -> inf, which dumps to the
    invalid-JSON token ``Infinity``). Keeps minify_json genuinely lossless."""
    x = float(n)
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite number {n!r}")
    return x


def _reject_constant(c: str) -> object:
    """parse_constant hook: reject the Infinity / -Infinity / NaN literals outright."""
    raise ValueError(f"non-finite constant {c!r}")


def _no_dup_keys(pairs: list[tuple[str, object]]) -> dict:
    """object_pairs_hook: reject duplicate keys (a round-trip silently drops all but
    the last, changing the object) — such input is left byte-exact instead."""
    d: dict = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"duplicate key {k!r}")
        d[k] = v
    return d


def minify_json(text: str) -> str | None:
    """Re-encode JSON with no incidental whitespace. None if not valid JSON.

    Only applied when the round-trip is provably value-preserving. Inputs where
    ``json.loads``/``json.dumps`` would silently change meaning are rejected (return
    None -> caller keeps the text byte-exact): duplicate object keys (all but the last
    are dropped) and non-finite floats (``1e400`` -> ``Infinity``, invalid JSON).
    """
    s = text.strip()
    if not (s[:1] in "{[" and s[-1:] in "}]"):
        return None
    try:
        obj = json.loads(
            s,
            object_pairs_hook=_no_dup_keys,
            parse_float=_reject_nonfinite,
            parse_constant=_reject_constant,
        )
    except (ValueError, TypeError, RecursionError):
        return None
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def collapse_runs(text: str) -> str:
    """Run-length-encode consecutive identical lines, reversibly.

    Two+ identical lines `L` collapse to `L\\n<<x N>>`. The marker preserves
    the exact count, so the original is fully recoverable.
    """

    def repl(m: re.Match[str]) -> str:
        line = m.group(1)
        # If the repeated content itself looks like a run marker, collapsing it would
        # make the generated marker indistinguishable from original content on reverse.
        # Leave such a run byte-exact (rare; the block stays fully recoverable anyway).
        if _RUN_MARKER.match(line):
            return m.group(0)
        n = m.group(0).count("\n")
        return f"{line}\n<<x{n}>>\n"

    return _RLE.sub(repl, text)


class Tier0Lossless:
    tier = 0
    name = "tier0-lossless"

    def compress(self, blocks: list[Block]) -> CompressResult:
        out: list[Block] = []
        restore: dict[str, str] = {}
        for b in blocks:
            text = b.text
            mj = minify_json(text)
            if mj is not None:
                text = mj
            text = collapse_runs(text)
            if text != b.text:
                restore[b.id] = b.text
            out.append(b.copy_with(text))
        return CompressResult(out, restore)
