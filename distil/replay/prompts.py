"""Shared decision-prompt rendering + fingerprint parsing for all live runners.

Every runner (Anthropic API, OpenAI-compatible/vLLM, the `claude -p` CLI) must put
the *same* context in front of the model and extract the *same* canonical
``{action,target}`` fingerprint, so their verdicts are comparable. That logic lives
here, once.
"""

from __future__ import annotations

import json
import re

from ..trajectory import Block, Kind, Stability

INSTRUCTION = (
    "You are choosing the single next action an autonomous agent should take given the "
    "context above. Respond with ONLY a compact JSON object and nothing else:\n"
    '{"action": "<tool or operation name>", "target": "<the primary argument>"}'
)


def render(blocks: list[Block]) -> tuple[str, str]:
    """Split blocks into (system, user) text exactly like the Anthropic runner does:
    the stable system/tool schema becomes the system prompt; everything else (history,
    fresh observations) becomes the user turn."""
    system_parts = [
        b.text
        for b in blocks
        if b.stability is Stability.STABLE and b.kind in (Kind.SYSTEM, Kind.TOOLS)
    ]
    rest = [
        b
        for b in blocks
        if not (b.stability is Stability.STABLE and b.kind in (Kind.SYSTEM, Kind.TOOLS))
    ]
    user = "\n\n".join(f"[{b.kind.value}] {b.text}" for b in rest)
    system = "\n\n".join(system_parts) or "You are an autonomous agent."
    return system, user


def decision_prompt(blocks: list[Block]) -> tuple[str, str]:
    """(system, user+instruction) for text/JSON runners that don't use a forced tool."""
    system, user = render(blocks)
    return system, f"{user}\n\n{INSTRUCTION}"


_OBJ = re.compile(r'\{[^{}]*"action"[^{}]*\}', re.DOTALL)


def parse_fingerprint(text: str) -> str:
    """Extract a canonical ``{"action":..,"target":..}`` fingerprint from free model
    text (handles ```json fences, surrounding prose, key reordering). Returns
    ``"<no-decision>"`` if nothing parseable is found."""
    if not text:
        return "<no-decision>"
    candidates = [text.strip()]
    m = _OBJ.search(text)
    if m:
        candidates.insert(0, m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and "action" in obj:
            return json.dumps(
                {"action": str(obj.get("action", "")), "target": str(obj.get("target", ""))},
                sort_keys=True,
                separators=(",", ":"),
            )
    return "<no-decision>"
