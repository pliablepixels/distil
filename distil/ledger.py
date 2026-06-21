"""Savings ledger — local-first, privacy-preserving community savings tracking.

Every certified run can append an aggregate record (ids + numbers only, never
context content) to a local JSONL. `summary()` rolls it up so you can see
cumulative tokens and dollars saved across an agent fleet over time.

Community aggregation (a shared leaderboard) is a deliberate OPT-IN: it would
mean network egress of your run metadata, so this module never sends anything.
The `share=` seam is where an explicit, consented uploader would plug in.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_PATH = Path.home() / ".distil" / "savings.jsonl"


@dataclass
class SavingsRecord:
    trajectory_id: str
    model: str
    turns: int
    baseline_dollars: float
    distil_dollars: float
    baseline_input_tokens: int
    distil_input_tokens: int
    tokenizer: str
    ts: float

    @property
    def dollars_saved(self) -> float:
        return self.baseline_dollars - self.distil_dollars

    @property
    def tokens_saved(self) -> int:
        return self.baseline_input_tokens - self.distil_input_tokens

    @property
    def pct_saved(self) -> float:
        return (self.dollars_saved / self.baseline_dollars * 100) if self.baseline_dollars else 0.0


def record(
    *,
    trajectory_id: str,
    model: str,
    turns: int,
    baseline_dollars: float,
    distil_dollars: float,
    baseline_input_tokens: int,
    distil_input_tokens: int,
    tokenizer: str = "heuristic",
    path: Path = DEFAULT_PATH,
    share: bool = False,  # opt-in network egress; intentionally unimplemented
) -> SavingsRecord:
    rec = SavingsRecord(
        trajectory_id,
        model,
        turns,
        baseline_dollars,
        distil_dollars,
        baseline_input_tokens,
        distil_input_tokens,
        tokenizer,
        time.time(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(rec)) + "\n")
    return rec


@dataclass
class LedgerSummary:
    runs: int
    total_dollars_saved: float
    total_tokens_saved: int
    by_trajectory: dict[str, float]  # id -> dollars saved


def summary(path: Path = DEFAULT_PATH) -> LedgerSummary:
    if not path.exists():
        return LedgerSummary(0, 0.0, 0, {})
    runs = 0
    dollars = 0.0
    tokens = 0
    by_traj: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        runs += 1
        saved = d["baseline_dollars"] - d["distil_dollars"]
        dollars += saved
        tokens += d["baseline_input_tokens"] - d["distil_input_tokens"]
        by_traj[d["trajectory_id"]] = by_traj.get(d["trajectory_id"], 0.0) + saved
    return LedgerSummary(runs, dollars, tokens, by_traj)
