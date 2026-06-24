"""Tests for the real-trace adapters + the prove.py proof harness.

These lock in the de-circularization work: τ-/SWE-bench traces load into the
trajectory model with NO planted DECISION markers, the smoke runner distinguishes
recoverable (safe) from irrecoverable (unsafe) compression, and the
frontier/coverage machinery produces a sound out-of-sample certificate.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from distil.compress.adaptive import byte_exact
from distil.conformal import default_ladder
from distil.replay import realtrace
from distil.replay.smoke_runner import SmokeRunner
from distil.trajectory import Stability

FIX = Path(__file__).resolve().parent.parent / "benchmarks" / "fixtures"


def _load_prove():
    path = Path(__file__).resolve().parent.parent / "benchmarks" / "prove.py"
    spec = importlib.util.spec_from_file_location("prove", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# adapters
# --------------------------------------------------------------------------- #


def test_tau_adapter_loads_no_markers():
    entries = realtrace.load_tau_bench(FIX / "tau_bench_sample.json")
    assert len(entries) >= 6
    for e in entries:
        assert e.domain == "tau-bench"
        assert e.trajectory.turns
        # no planted DECISION: oracle anywhere — that is the whole point
        for t in e.trajectory.turns:
            assert all("DECISION:" not in b.text for b in t.blocks)
            # cacheable prefix invariant: volatile blocks come last
            kinds = [b.stability is Stability.VOLATILE for b in t.blocks]
            assert kinds == sorted(kinds, key=lambda v: v)  # all False then all True
            assert any(b.stability is Stability.VOLATILE for b in t.blocks)


def test_swe_adapter_loads_with_resolution():
    entries = realtrace.load_swe_bench(FIX / "swe_bench_sample.json")
    assert len(entries) >= 4
    assert all(e.domain == "swe-bench" for e in entries)
    # resolution status is carried for the downstream task-success metric
    statuses = [realtrace.resolved_status(e) for e in entries]
    assert any(s is True for s in statuses) and any(s is False for s in statuses)


def test_gold_actions_present_and_canonical():
    entries = realtrace.load_tau_bench(FIX / "tau_bench_sample.json")
    gold = realtrace.gold_actions(entries)
    assert gold
    for g in gold.values():
        # fingerprint is the same {action,target} JSON the live runner emits
        assert g.fingerprint.startswith('{"action":')


def test_structural_validation_clean():
    entries = realtrace.load_tau_bench(FIX / "tau_bench_sample.json")
    entries += realtrace.load_swe_bench(FIX / "swe_bench_sample.json")
    assert realtrace.validate_real(entries) == []


# --------------------------------------------------------------------------- #
# smoke runner — recoverable vs irrecoverable
# --------------------------------------------------------------------------- #


def test_smoke_runner_is_not_marker_based():
    # the smoke runner must NOT be the circular DECISION-marker oracle
    from distil.replay.runner import DeterministicRunner

    assert SmokeRunner().name != DeterministicRunner().name
    assert getattr(SmokeRunner(), "evidential", True) is False


def test_byte_exact_preserves_decision_truncation_can_flip():
    entries = realtrace.load_tau_bench(FIX / "tau_bench_sample.json")
    runner = SmokeRunner()

    def truncate120(blocks, turn):
        return [
            b.copy_with(b.text[:120]) if b.stability is Stability.VOLATILE else b for b in blocks
        ]

    flips_byte = flips_trunc = record_turns = 0
    for e in entries:
        for t in e.trajectory.turns:
            base = runner.decide(t.blocks)
            if base == "<no-record>":
                continue
            record_turns += 1
            flips_byte += int(runner.decide(byte_exact(t.blocks, t.index)) != base)
            flips_trunc += int(runner.decide(truncate120(t.blocks, t.index)) != base)
    assert record_turns > 0
    assert flips_byte == 0  # byte-exact never changes a decision
    assert flips_trunc > 0  # aggressive truncation drops the load-bearing record


# --------------------------------------------------------------------------- #
# harness end-to-end
# --------------------------------------------------------------------------- #


def _matrix():
    prove = _load_prove()
    entries = realtrace.load_tau_bench(FIX / "tau_bench_sample.json")
    entries += realtrace.load_swe_bench(FIX / "swe_bench_sample.json")
    gold = realtrace.gold_actions(entries)
    ladder = default_ladder()

    class _Cache:
        def __init__(self):
            self.r = SmokeRunner()

        def decide(self, blocks):
            return self.r.decide(blocks)

    return prove, prove.build_matrix(entries, _Cache(), ladder, gold), ladder


def test_frontier_lossless_safe_aggressive_unsafe():
    prove, matrix, ladder = _matrix()
    rows = {r["level"]: r for r in prove.e1_frontier(matrix, ladder)}
    # the reversible digest saves tokens at ZERO decision change...
    assert rows["lossless"]["savings"] > 0.05
    assert rows["lossless"]["decision_change"] == 0.0
    assert rows["byte-exact"]["decision_change"] == 0.0
    # ...while blind aggressive truncation changes decisions
    assert rows["truncate@120"]["decision_change"] > 0.0
    assert rows["truncate@120"]["savings"] > rows["lossless"]["savings"]


def test_certificate_holds_out_of_sample():
    prove, matrix, ladder = _matrix()
    # at an α the sample can support, the held-out coverage must meet the 1-δ target
    cov = prove.e2_coverage(matrix, ladder, alpha=0.2, delta=0.05, method="ltt", reps=200, seed=0)
    assert cov["certified_frac"] > 0.5
    assert cov["empirical_coverage"] >= 0.95  # the guarantee, validated out-of-sample
    assert cov["mean_realized_risk"] <= 0.2
    assert cov["mean_test_savings"] > 0.05


def test_tiny_sample_refuses_tight_alpha():
    # honesty property: too few calibration turns ⇒ refuse to certify a tight α
    prove, matrix, ladder = _matrix()
    cov = prove.e2_coverage(matrix, ladder, alpha=0.01, delta=0.05, method="ltt", reps=50, seed=0)
    assert cov["certified_frac"] == 0.0
