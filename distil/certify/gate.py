"""The certification gate — decision-equivalence on a corpus, then TOST.

For each turn we compare the agent's decision on the compressed context against
its decision on the uncompressed context. The per-turn outcome is 1.0 if they
match, else 0.0; the paired difference vs the (always-1.0) baseline feeds TOST.
A strategy ships only if it is certified non-inferior.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..compress.strategies import REGISTRY, Strategy
from ..replay.runner import AgentRunner, DeterministicRunner
from ..trajectory import Trajectory
from .stats import TostResult, tost


@dataclass
class TurnDivergence:
    turn: int
    baseline_decision: str
    compressed_decision: str

    @property
    def matched(self) -> bool:
        return self.baseline_decision == self.compressed_decision


@dataclass
class CertReport:
    strategy: str
    divergences: list[TurnDivergence]
    tost: TostResult

    @property
    def match_rate(self) -> float:
        if not self.divergences:
            return 1.0
        return sum(d.matched for d in self.divergences) / len(self.divergences)

    @property
    def verdict(self) -> str:
        return self.tost.verdict


def certify(
    traj: Trajectory,
    strategy: str | Strategy,
    *,
    runner: AgentRunner | None = None,
    margin: float = 0.02,
    alpha: float = 0.05,
) -> CertReport:
    runner = runner or DeterministicRunner()
    fn: Strategy = REGISTRY[strategy] if isinstance(strategy, str) else strategy
    name = strategy if isinstance(strategy, str) else getattr(strategy, "__name__", "custom")

    divergences: list[TurnDivergence] = []
    diffs: list[float] = []
    for turn in traj.turns:
        base = runner.decide(turn.blocks)
        comp = runner.decide(fn(turn.blocks, turn.index))
        divergences.append(TurnDivergence(turn.index, base, comp))
        # compressed_score - baseline_score; baseline scores 1.0 against itself,
        # compressed scores 1.0 only if its decision matches. Worse -> negative.
        diffs.append((1.0 if base == comp else 0.0) - 1.0)

    return CertReport(name, divergences, tost(diffs, margin=margin, alpha=alpha))
