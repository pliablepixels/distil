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
