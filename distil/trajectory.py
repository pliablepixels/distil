"""The trajectory data model — a captured agent run.

A trajectory is an ordered list of turns; each turn is the full context the
model saw at that step, decomposed into blocks. Blocks carry a *stability*
hint (what can live in a cacheable prefix) and an optional ground-truth
`decision_relevant` label used by ablation/certification corpora.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Stability(str, Enum):
    STABLE = "stable"  # system prompt, tool schemas, settled facts -> cacheable prefix
    SETTLING = "settling"  # older history; identical across turns once written -> cacheable
    VOLATILE = "volatile"  # the current user turn / fresh tool output -> never cached


class Kind(str, Enum):
    SYSTEM = "system"
    TOOLS = "tools"
    HISTORY = "history"
    TOOL_OUTPUT = "tool_output"
    USER = "user"
    RETRIEVED = "retrieved"


@dataclass
class Block:
    id: str
    kind: Kind
    text: str
    stability: Stability = Stability.VOLATILE
    decision_relevant: bool = False

    def copy_with(self, text: str) -> "Block":
        return Block(self.id, self.kind, text, self.stability, self.decision_relevant)


@dataclass
class Turn:
    index: int
    blocks: list[Block]


@dataclass
class Trajectory:
    id: str
    model: str
    turns: list[Turn] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> "Trajectory":
        turns = [
            Turn(
                index=t.get("index", i),
                blocks=[
                    Block(
                        id=b["id"],
                        kind=Kind(b["kind"]),
                        text=b["text"],
                        stability=Stability(b.get("stability", "volatile")),
                        decision_relevant=bool(b.get("decision_relevant", False)),
                    )
                    for b in t["blocks"]
                ],
            )
            for i, t in enumerate(d["turns"])
        ]
        return Trajectory(
            id=d.get("id", "trajectory"), model=d.get("model", "claude-opus-4"), turns=turns
        )

    @staticmethod
    def load(path: str | Path) -> "Trajectory":
        return Trajectory.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "model": self.model,
            "turns": [
                {
                    "index": t.index,
                    "blocks": [
                        {
                            "id": b.id,
                            "kind": b.kind.value,
                            "text": b.text,
                            "stability": b.stability.value,
                            "decision_relevant": b.decision_relevant,
                        }
                        for b in t.blocks
                    ],
                }
                for t in self.turns
            ],
        }
