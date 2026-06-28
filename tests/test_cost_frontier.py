"""Cost-frontier techniques (#1 cache-monotone gate, #2 graded gate, #4 speculative, #5 bandit).

These verify the motto-safe cost optimizations: lossless caching behavior, graded compression,
certified speculative escalation, and online operating-point selection.
"""

from __future__ import annotations

import random

from distil.calibrate import bandit_select_operating_point
from distil.gate import graded_gate, monotone_gate
from distil.speculative import calibrate_speculative
from distil.trajectory import Block, Kind, Stability, Trajectory, Turn


# --------------------------------------------------------------------------- #
# #1 cache-monotone gate
# --------------------------------------------------------------------------- #


def _code(i: int) -> str:
    # A digestible block: signatures + bodies (smart_digest keeps signatures, elides bodies).
    return (
        f"import os\n\ndef func_{i}(a, b):\n"
        + "\n".join(f"    x{j} = a + b + {j}  # filler body line {j}" for j in range(40))
        + f"\n    return x0\n\nclass C{i}:\n    def m(self):\n        return {i}\n"
    )


def _growing_trajectory(n_turns: int) -> Trajectory:
    turns = []
    for t in range(n_turns):
        blocks = [
            Block(id=f"b{i}", kind=Kind.TOOL_OUTPUT, text=_code(i), stability=Stability.STABLE)
            for i in range(t + 1)
        ]
        # the current turn's last block is volatile (fresh)
        blocks[-1] = Block(blocks[-1].id, Kind.TOOL_OUTPUT, blocks[-1].text, Stability.VOLATILE)
        turns.append(Turn(index=t, blocks=blocks))
    return Trajectory(id="t", model="m", turns=turns)


def test_monotone_gate_keeps_working_set_full_and_digests_periphery():
    gate = monotone_gate(gate_recent=3)
    blocks = _growing_trajectory(8).turns[-1].blocks  # 8 blocks
    out = gate(blocks, 7)
    assert len(out) == len(blocks)
    # last 3 kept full (byte-identical), earlier ones strictly smaller (digested)
    for i in range(5):
        assert len(out[i].text) < len(blocks[i].text), f"block {i} should be digested"
    for i in range(5, 8):
        assert out[i].text == blocks[i].text, f"block {i} (working set) must stay full"


def test_monotone_gate_digested_prefix_is_byte_stable_across_turns():
    # The cache-monotonicity property: a block, once digested, renders identically every turn.
    gate = monotone_gate(gate_recent=3)
    traj = _growing_trajectory(10)
    renderings: dict[str, set[str]] = {}
    for turn in traj.turns:
        for b in gate(turn.blocks, turn.index):
            renderings.setdefault(b.id, set()).add(b.text)
    # Each block id must have rendered as at most: {full} then {digest} — never multiple digests.
    for bid, texts in renderings.items():
        assert len(texts) <= 2, f"block {bid} rendered {len(texts)} distinct ways (cache-hostile)"


def test_monotone_gate_caches_better_than_churning_gate():
    # The honest #1 claim: a cache-MONOTONE gate (deterministic, stable digests) recovers cache
    # hits that a cache-HOSTILE gate (which re-digests the same block differently each turn)
    # throws away. Both compress identically; only cache behavior differs. (Note: on already
    # fully-cacheable content, compression can cost MORE than caching alone — the gate's payoff
    # is accuracy on long-horizon agents, not cost-domination; see docs/GA_READINESS.md.)
    from distil.compress.cache_aware import simulate
    from distil.pricing import get
    from distil.skeleton import smart_digest

    def churning_gate(gate_recent):
        from distil.trajectory import Stability

        def strat(blocks, turn):
            keep_from = max(0, len(blocks) - gate_recent)
            out = []
            for i, b in enumerate(blocks):
                if i >= keep_from or b.stability is Stability.VOLATILE:
                    out.append(b)
                else:  # cache-hostile: digest text varies by turn -> prefix never matches
                    out.append(b.copy_with(smart_digest(b.text) + f"\n# rev {turn}"))
            return out

        return strat

    traj = _growing_trajectory(12)
    pricing = get("claude-haiku-4-5")
    monotone = simulate(traj, pricing, strategy=monotone_gate(gate_recent=3), caching=True)
    churning = simulate(traj, pricing, strategy=churning_gate(3), caching=True)
    assert monotone.cache_hit_tokens > churning.cache_hit_tokens
    assert monotone.total_dollars < churning.total_dollars


# --------------------------------------------------------------------------- #
# #2 graded gate
# --------------------------------------------------------------------------- #


def test_graded_gate_compresses_more_than_plain_gate():
    # Prose/log periphery (not code): smart_digest falls back to head/tail truncation, so the
    # heavy tier (small head/tail) genuinely compresses more than the light/plain tier. (On
    # code blocks the skeleton is already minimal, so grading is a no-op there — by design.)
    prose = "\n".join(
        f"log line {j}: the agent observed event number {j} in detail" for j in range(200)
    )
    blocks = [
        Block(id=f"b{i}", kind=Kind.TOOL_OUTPUT, text=prose, stability=Stability.STABLE)
        for i in range(10)
    ]
    plain = monotone_gate(gate_recent=3)(blocks, 9)
    graded = graded_gate(keep_full=3, keep_light=2)(blocks, 9)
    plain_chars = sum(len(b.text) for b in plain)
    graded_chars = sum(len(b.text) for b in graded)
    assert graded_chars < plain_chars  # heavy digest on the far periphery saves more
    assert graded[-1].text == blocks[-1].text  # working set still intact


# --------------------------------------------------------------------------- #
# #4 speculative-expand controller
# --------------------------------------------------------------------------- #


def test_speculative_escalates_high_risk_and_certifies_low_miss():
    # Scores correlated with divergence: high score -> likely diverged.
    rng = random.Random(7)
    scores, diverged = [], []
    for _ in range(600):
        d = 1 if rng.random() < 0.15 else 0
        s = rng.uniform(0.6, 1.0) if d else rng.uniform(0.0, 0.5)
        scores.append(s)
        diverged.append(d)
    ctrl = calibrate_speculative(scores, diverged, alpha=0.05, delta=0.1)
    assert ctrl.feasible
    assert ctrl.certified_miss_rate <= 0.05
    assert ctrl.escalation_rate < 0.5  # cheaper than always-full
    assert ctrl.decide(0.9) == "full"
    assert ctrl.decide(0.1) == "compressed"


def test_speculative_no_savings_when_score_uninformative():
    # Score uncorrelated with divergence at a high base rate: to certify a strict 2% miss the
    # controller must escalate ~everything -> high escalation_rate (no real savings). That is
    # the honest, safe outcome (degrade to full context), not a silent lossy shortcut.
    rng = random.Random(11)
    scores = [rng.random() for _ in range(300)]
    diverged = [1 if rng.random() < 0.4 else 0 for _ in range(300)]
    ctrl = calibrate_speculative(scores, diverged, alpha=0.02, delta=0.1)
    assert ctrl.escalation_rate > 0.8


# --------------------------------------------------------------------------- #
# #5 constrained-bandit operating-point selection
# --------------------------------------------------------------------------- #


def test_bandit_selects_aggressive_safe_arm_and_prunes_unsafe():
    # gate@6 is lossy (P(loss) high), gate@12 is near-tie. Bandit should pick gate@12.
    rng = random.Random(3)

    def sample_fn(name: str, gr: int) -> int:
        if gr <= 6:  # aggressive & lossy: many baseline-only wins
            r = rng.random()
            return -1 if r < 0.30 else (1 if r < 0.33 else 0)
        # mild & safe: near-tie
        r = rng.random()
        return -1 if r < 0.07 else (1 if r < 0.12 else 0)

    res = bandit_select_operating_point(
        [("gate@6", 6), ("gate@12", 12)], sample_fn, margin=0.10, budget=3000, batch=50
    )
    assert not res.fail_safe
    assert res.selected == "gate@12"
    g6 = next(v for v in res.arms if v.name == "gate@6")
    assert g6.noninferior is False


def test_bandit_fails_safe_when_all_arms_lossy():
    rng = random.Random(5)

    def sample_fn(name: str, gr: int) -> int:
        r = rng.random()
        return -1 if r < 0.4 else (1 if r < 0.45 else 0)

    res = bandit_select_operating_point(
        [("gate@6", 6), ("gate@12", 12)], sample_fn, margin=0.05, budget=1500, batch=50
    )
    assert res.fail_safe
    assert res.selected is None
