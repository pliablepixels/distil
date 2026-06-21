"""Technique #1 — cache-aware cost simulation.

Models a multi-turn agent loop where the full context is re-sent every turn.
With prompt caching, the maximal *contiguous prefix* whose text is unchanged
from the previous turn is billed at the cache-read price (~0.10x input); the
first changed block onward is billed fresh, and newly-cacheable blocks pay the
cache-write price (~1.25x input).

Key consequence the simulation makes visible: a compressor that perturbs the
prefix every turn (`naive`) drops the longest-common-prefix to zero and loses
the 10x discount, so it can cost MORE than not compressing at all. Keeping the
prefix byte-stable (`distil`) is worth more than shaving its tokens.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..pricing import Pricing
from ..tokenizer import DEFAULT, Tokenizer
from ..trajectory import Stability, Trajectory
from .strategies import REGISTRY, Strategy


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class TurnCost:
    turn: int
    read_tokens: int
    write_tokens: int
    fresh_tokens: int
    output_tokens: int
    dollars: float


@dataclass
class SimResult:
    strategy: str
    caching: bool
    per_turn: list[TurnCost]
    total_dollars: float
    total_input_tokens: int  # tokens actually sent to the model (sum over turns)

    @property
    def cache_hit_tokens(self) -> int:
        return sum(t.read_tokens for t in self.per_turn)


def simulate(
    traj: Trajectory,
    pricing: Pricing,
    *,
    strategy: str | Strategy = "none",
    caching: bool = True,
    output_tokens_per_turn: int = 0,
    tok: Tokenizer = DEFAULT,
) -> SimResult:
    fn: Strategy = REGISTRY[strategy] if isinstance(strategy, str) else strategy
    name = strategy if isinstance(strategy, str) else getattr(strategy, "__name__", "custom")

    prev_hashes: list[str] | None = None
    per_turn: list[TurnCost] = []
    total = 0.0
    total_input = 0

    for turn in traj.turns:
        blocks = fn(turn.blocks, turn.index)
        hashes = [_h(b.text) for b in blocks]
        toks = [tok.count(b.text) for b in blocks]
        total_input += sum(toks)

        # longest common contiguous prefix with the previous turn (prefix caching)
        lcp = 0
        if caching and prev_hashes is not None:
            limit = min(len(hashes), len(prev_hashes))
            while lcp < limit and hashes[lcp] == prev_hashes[lcp]:
                lcp += 1

        read = sum(toks[:lcp])
        write = 0
        fresh = 0
        for b, t in zip(blocks[lcp:], toks[lcp:]):
            # cacheable blocks past the matched prefix are written to cache;
            # volatile blocks are billed as plain fresh input.
            if caching and b.stability is not Stability.VOLATILE:
                write += t
            else:
                fresh += t

        cost = (
            read * pricing.cache_read
            + write * pricing.cache_write
            + fresh * pricing.input
            + output_tokens_per_turn * pricing.output
        )
        per_turn.append(TurnCost(turn.index, read, write, fresh, output_tokens_per_turn, cost))
        total += cost
        prev_hashes = hashes

    return SimResult(name, caching, per_turn, total, total_input)
