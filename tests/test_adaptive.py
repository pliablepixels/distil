"""Certified-fallback compression — 100% decision-equivalent by construction."""

from __future__ import annotations

from distil.certify.gate import certify
from distil.compress.adaptive import byte_exact, certified_fallback, fallback_breakdown
from distil.corpus import load_corpus
from distil.replay.runner import DeterministicRunner


def _traj():
    return load_corpus()[0].trajectory


def test_certified_fallback_is_100pct_by_construction():
    traj = _traj()
    runner = DeterministicRunner()
    strat = certified_fallback(traj, runner)
    rep = certify(traj, strat, runner=runner)
    assert rep.match_rate == 1.0  # never ships a decision-changing transform
    assert rep.tost.non_inferior


def test_falls_back_when_top_rung_diverges():
    traj = _traj()
    runner = DeterministicRunner()

    # a hostile top rung that strips DECISION markers → always diverges
    def decision_destroyer(blocks, turn):
        return [b.copy_with(b.text.replace("DECISION:", "x")) for b in blocks]

    ladder = [("bad", decision_destroyer), ("byte-exact", byte_exact), ("none", lambda b, t: b)]
    strat = certified_fallback(traj, runner, ladder=ladder)
    rep = certify(traj, strat, runner=runner)
    assert rep.match_rate == 1.0  # fell back past the destroyer, stayed equivalent

    counts = fallback_breakdown(traj, runner, ladder=ladder)
    assert counts["bad"] == 0  # the destroyer was never selected
    assert counts["byte-exact"] + counts["none"] == len(traj.turns)


def test_byte_exact_preserves_decision_lines():
    traj = _traj()
    for turn in traj.turns:
        out = byte_exact(turn.blocks, turn.index)
        before = "\n".join(b.text for b in turn.blocks)
        after = "\n".join(b.text for b in out)
        # every DECISION line survives byte-exact (it only minifies/collapses)
        assert before.count("DECISION:") == after.count("DECISION:")


def test_target_equivalence_is_validated():
    import pytest

    from distil.compress.adaptive import certified_fallback

    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            certified_fallback(_traj(), DeterministicRunner(), target_equivalence=bad)


def test_frontier_is_monotonic_and_dials_savings():
    from distil.compress.adaptive import frontier
    from distil.corpus import load_corpus

    entries = load_corpus()
    pts = frontier(entries, DeterministicRunner(), targets=(1.0, 0.8, 0.6, 0.4))
    # 100% target is fully certified
    assert pts[0].target == 1.0 and pts[0].equivalence == 1.0
    # relaxing the target never *reduces* savings and never *raises* equivalence
    for a, b in zip(pts, pts[1:]):
        assert b.savings >= a.savings - 1e-9
        assert b.equivalence <= a.equivalence + 1e-9
    # and relaxing genuinely buys *some* extra savings by the lowest target
    assert pts[-1].savings > pts[0].savings


def test_relaxed_target_lowers_equivalence_below_one():
    from distil.compress.adaptive import certified_fallback
    from distil.compress.adaptive import PRODUCTION_LADDER

    traj = _traj()
    runner = DeterministicRunner()
    strat = certified_fallback(traj, runner, target_equivalence=0.5, ladder=PRODUCTION_LADDER)
    rep = certify(traj, strat, runner=runner)
    # with a 50% target and a divergent aggressive rung, equivalence drops below 1.0
    assert rep.match_rate < 1.0
