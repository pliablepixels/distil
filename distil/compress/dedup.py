"""Cross-turn reversible deduplication — the streaming-loop technique.

Agents re-read the same artifact (a file, a log, a design doc, a dir listing)
across turns. Prompt caching only discounts the *contiguous prefix*; a large
block that recurs in the volatile tail — after the first changed block — is NOT
covered by the cache and is re-billed in full every time it reappears.

This compressor remembers the content it has already sent and, when an inert
block recurs verbatim, replaces it with a compact reference. The byte-exact
original is kept in the restore table and is one ``expand()`` away — reversible,
not lossy. It only references blocks that carry no DECISION marker, so the
decision signal is never perturbed.

It is *stateful* across a trajectory. ``reset()`` (also triggered automatically
when a turn index is non-increasing) clears the memory so a fresh pass starts
clean — which is what the benchmark's repeated measurement passes need.
"""

from __future__ import annotations

import hashlib

from ..trajectory import Block, Stability


def _handle(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


class StreamingDedup:
    def __init__(self, min_chars: int = 160) -> None:
        self.min_chars = min_chars
        self._seen: dict[str, int] = {}
        self._last_turn = -1

    def reset(self) -> None:
        self._seen.clear()
        self._last_turn = -1

    def compress(self, blocks: list[Block], turn: int) -> tuple[list[Block], dict[str, str]]:
        if turn <= self._last_turn:  # a new measurement pass began — start clean
            self.reset()
        self._last_turn = turn

        out: list[Block] = []
        restore: dict[str, str] = {}
        for b in blocks:
            eligible = (
                b.stability is Stability.VOLATILE
                and "DECISION:" not in b.text
                and len(b.text) >= self.min_chars
            )
            if eligible:
                h = _handle(b.text)
                if h in self._seen:
                    restore[h] = b.text  # byte-exact original, expandable
                    out.append(
                        b.copy_with(
                            f"«repeat of earlier tool output {h} (first seen turn {self._seen[h]})»"
                        )
                    )
                    continue
                self._seen[h] = turn
            out.append(b)
        return out, restore
