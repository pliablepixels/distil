"""Cache stabilization — keep the prefix byte-identical across turns.

The cache-aware engine only pays off if the prefix actually *stays* byte-stable.
Two things silently break that in real agents, and both are fixed here losslessly:

1. **Schema churn.** Tool/JSON payloads whose keys are emitted in a different
   order each call hash differently every turn -> cache miss. `canonicalize_json`
   recursively sorts keys (semantically identical, order is irrelevant) so the
   bytes are stable.

2. **Volatile fields in a stable block.** A system prompt that embeds "Current
   Date: ...", a request UUID, or a JWT changes every turn, so the whole
   cacheable prefix misses. `extract_volatile` lifts those values out into a
   trailing volatile block and leaves a positional placeholder behind — the
   placeholder is the same regardless of the value, so the prefix re-stabilizes.
   Fully reversible via `restore_volatile`.
"""

from __future__ import annotations

import json
import re

from ..trajectory import Block, Kind, Stability

# Order matters: match the longest/most-specific patterns first.
_JWT = r"eyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}"
_UUID = r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}"
_ISO = r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)?"
_VOLATILE = re.compile("|".join(f"(?:{p})" for p in (_JWT, _UUID, _ISO)))

_PLACEHOLDER = "⟦V{}⟧"  # ⟦V1⟧, ⟦V2⟧, ...


def canonicalize_json(text: str) -> str | None:
    """Recursively key-sorted, whitespace-free JSON. None if not JSON."""
    s = text.strip()
    if not (s[:1] in "{[" and s[-1:] in "}]"):
        return None
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return None
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def extract_volatile(text: str) -> tuple[str, list[str]]:
    """Replace volatile tokens with positional placeholders. Returns
    (stabilized_text, values). Same structure -> same placeholders across turns,
    even when the underlying values differ."""
    values: list[str] = []

    def repl(m: re.Match[str]) -> str:
        values.append(m.group(0))
        return _PLACEHOLDER.format(len(values))

    return _VOLATILE.sub(repl, text), values


def restore_volatile(text: str, values: list[str]) -> str:
    for i, v in enumerate(values, start=1):
        text = text.replace(_PLACEHOLDER.format(i), v, 1)
    return text


def stabilize_blocks(blocks: list[Block]) -> list[Block]:
    """Lift volatile fields out of STABLE blocks into one trailing volatile
    block, so the cacheable prefix stops changing turn-to-turn. No-op when the
    stable prefix carries no volatile tokens."""
    out: list[Block] = []
    lifted: list[str] = []
    for b in blocks:
        if b.stability is Stability.STABLE:
            stab, vals = extract_volatile(b.text)
            if vals:
                out.append(b.copy_with(stab))
                lifted += [f"{b.id}:{_PLACEHOLDER.format(i + 1)}={v}" for i, v in enumerate(vals)]
                continue
        out.append(b)
    if lifted:
        out.append(
            Block(
                "volatile-context",
                Kind.USER,
                "[volatile fields lifted from the cached prefix: " + "; ".join(lifted) + "]",
                Stability.VOLATILE,
            )
        )
    return out
