"""Technique #4 — causal / counterfactual pruning.

For each block, remove it and replay: did any turn's decision change? A block
that is removable in *every* turn it appears in is causally inert — provably
free to drop. This is the sense in which the eval engine is a *discovery*
engine, not a ruler: its output is a pruning policy, and the tokens it frees are
real savings with a causal justification, not a heuristic guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..tokenizer import DEFAULT, Tokenizer
from ..trajectory import Trajectory
from .runner import AgentRunner, DeterministicRunner


@dataclass
class BlockVerdict:
    block_id: str
    occurrences: int
    changed_a_decision: bool
    tokens: int

    @property
    def prunable(self) -> bool:
        return not self.changed_a_decision


@dataclass
class AblationReport:
    verdicts: list[BlockVerdict] = field(default_factory=list)

    @property
    def prunable(self) -> list[BlockVerdict]:
        return [v for v in self.verdicts if v.prunable]

    @property
    def tokens_freed(self) -> int:
        return sum(v.tokens for v in self.prunable)


def discover(
    traj: Trajectory,
    runner: AgentRunner | None = None,
    tok: Tokenizer = DEFAULT,
) -> AblationReport:
    runner = runner or DeterministicRunner()

    occ: dict[str, int] = {}
    changed: dict[str, bool] = {}
    toks: dict[str, int] = {}

    for turn in traj.turns:
        base = runner.decide(turn.blocks)
        for b in turn.blocks:
            occ[b.id] = occ.get(b.id, 0) + 1
            toks[b.id] = max(toks.get(b.id, 0), tok.count(b.text))
            ablated = [x for x in turn.blocks if x.id != b.id]
            if runner.decide(ablated) != base:
                changed[b.id] = True
            else:
                changed.setdefault(b.id, False)

    verdicts = [BlockVerdict(bid, occ[bid], changed[bid], toks[bid]) for bid in occ]
    verdicts.sort(key=lambda v: (v.changed_a_decision, -v.tokens))
    return AblationReport(verdicts)
