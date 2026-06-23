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


def minify_json(text: str) -> str | None:
    """Re-encode JSON with no incidental whitespace. None if not valid JSON."""
    s = text.strip()
    if not (s[:1] in "{[" and s[-1:] in "}]"):
        return None
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return None
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def collapse_runs(text: str) -> str:
    """Run-length-encode consecutive identical lines, reversibly.

    Two+ identical lines `L` collapse to `L\\n<<x N>>`. The marker preserves
    the exact count, so the original is fully recoverable.
    """

    def repl(m: re.Match[str]) -> str:
        line = m.group(1)
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
