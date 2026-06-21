"""The gate must PASS lossless strategies and FAIL quality-degrading ones."""

from distil.certify.gate import certify
from distil.certify.stats import tost
from distil.replay.ablation import discover
from distil.trajectory import Trajectory

CORPUS = "corpus/sample_trajectory.json"


def test_tost_passes_when_lossless():
    # every paired difference is zero -> non-inferior at any positive margin
    res = tost([0.0, 0.0, 0.0, 0.0], margin=0.02)
    assert res.non_inferior


def test_tost_fails_on_real_degradation():
    res = tost([0.0, -1.0, -1.0, 0.0], margin=0.02)
    assert not res.non_inferior


def test_distil_strategy_is_certified_non_inferior():
    report = certify(Trajectory.load(CORPUS), "distil")
    assert report.match_rate == 1.0
    assert report.verdict == "PASS"


def test_aggressive_strategy_is_rejected():
    report = certify(Trajectory.load(CORPUS), "aggressive")
    assert report.match_rate < 1.0
    assert report.verdict == "FAIL"


def test_ablation_finds_the_speculative_docs_prunable():
    report = discover(Trajectory.load(CORPUS))
    prunable_ids = {v.block_id for v in report.prunable}
    # the speculative retrieved docs never changed a decision
    assert {"doc-0", "doc-1", "doc-2", "doc-3"} <= prunable_ids
    # the system prompt and tool schema DID drive decisions -> kept
    assert "system" not in prunable_ids
    assert "tools" not in prunable_ids
    assert report.tokens_freed > 0
