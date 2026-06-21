"""Tests for the self-distilling online loop (distil/online.py).

Covers:
  1. collect_causal_labels — non-empty, contains both labels {0, 1}.
  2. retrain — accuracy and F1 in [0, 1] with accuracy >= 0.7.
  3. certify_promotion — freshly trained weights pass the non-inferiority gate.
  4. online_round — full round returns a report dict with certified=True.
  5. Degenerate gate proof — an all-drop model (all labels=0) trains fine, but
     when applied WITHOUT the decision_relevant guard (i.e. compressing
     decision-relevant blocks too) the certify gate REJECTS it — proving TOST
     blocks regressions.  certify_promotion itself is safe because its internal
     strategy only compresses causally-inert (non-decision-relevant) blocks.
"""

from __future__ import annotations

import pytest

from distil.certify.gate import certify
from distil.codec.features import FEATURE_NAMES
from distil.codec.keep_model import apply_keep
from distil.codec.learned import LogisticKeepModel
from distil.corpus import load_corpus
from distil.online import (
    _VOLATILE_KINDS,
    certify_promotion,
    collect_causal_labels,
    online_round,
    retrain,
)
from distil.trajectory import Stability


# ---------------------------------------------------------------------------
# Shared fixture — load corpus once per session.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def entries():
    return load_corpus()


# ---------------------------------------------------------------------------
# Shared fixture — causal labels computed once per session.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def causal_labels(entries):
    return collect_causal_labels(entries)


# ---------------------------------------------------------------------------
# 1. collect_causal_labels
# ---------------------------------------------------------------------------


class TestCollectCausalLabels:
    def test_non_empty(self, causal_labels):
        assert len(causal_labels) > 0, "collect_causal_labels returned no labels"

    def test_contains_both_label_values(self, causal_labels):
        label_set = {label for _, label in causal_labels}
        assert 0 in label_set, "No label=0 (safe-to-drop) found — ablation has no prunable blocks"
        assert 1 in label_set, "No label=1 (keep) found — ablation has no kept blocks"

    def test_labels_are_binary(self, causal_labels):
        for line, label in causal_labels:
            assert label in (0, 1), f"Unexpected label {label!r} for line {line!r}"

    def test_no_duplicate_lines(self, causal_labels):
        lines = [line for line, _ in causal_labels]
        assert len(lines) == len(set(lines)), "Duplicate lines found — deduplication failed"


# ---------------------------------------------------------------------------
# 2. retrain
# ---------------------------------------------------------------------------


class TestRetrain:
    @pytest.fixture(scope="class")
    def train_result(self, causal_labels):  # type: ignore[override]
        return retrain(causal_labels)

    def test_returns_required_keys(self, train_result):
        for key in ("weights", "accuracy", "f1", "n_train", "n_test"):
            assert key in train_result, f"Missing key {key!r} in retrain result"

    def test_weights_length(self, train_result):
        assert len(train_result["weights"]) == len(FEATURE_NAMES), (
            f"weights length {len(train_result['weights'])} != {len(FEATURE_NAMES)}"
        )

    def test_accuracy_in_unit_interval(self, train_result):
        acc = train_result["accuracy"]
        assert 0.0 <= acc <= 1.0, f"accuracy {acc} out of [0, 1]"

    def test_f1_in_unit_interval(self, train_result):
        f1 = train_result["f1"]
        assert 0.0 <= f1 <= 1.0, f"F1 {f1} out of [0, 1]"

    def test_accuracy_reasonable_fit(self, train_result):
        acc = train_result["accuracy"]
        assert acc >= 0.7, (
            f"Held-out accuracy {acc:.4f} < 0.7 — the model is not learning the causal labels"
        )

    def test_positive_train_and_test_counts(self, train_result):
        assert train_result["n_train"] > 0, "n_train is zero"
        assert train_result["n_test"] >= 0, "n_test is negative"


# ---------------------------------------------------------------------------
# 3. certify_promotion — freshly trained weights
# ---------------------------------------------------------------------------


class TestCertifyPromotion:
    @pytest.fixture(scope="class")
    def trained_weights(self, causal_labels):  # type: ignore[override]
        return retrain(causal_labels)["weights"]

    def test_trained_weights_pass_gate(self, trained_weights, entries):
        result = certify_promotion(trained_weights, entries)
        assert result is True, (
            "certify_promotion returned False for freshly trained causal weights — "
            "the model causes regressions on the corpus"
        )


# ---------------------------------------------------------------------------
# 4. online_round — full integration
# ---------------------------------------------------------------------------


class TestOnlineRound:
    @pytest.fixture(scope="class")
    def report(self, entries):  # type: ignore[override]
        return online_round(entries)

    def test_returns_dict_with_required_keys(self, report):
        for key in ("n_labels", "accuracy", "f1", "certified", "promoted"):
            assert key in report, f"Missing key {key!r} in online_round report"

    def test_certified_true(self, report):
        assert report["certified"] is True, f"online_round did not certify: {report}"

    def test_n_labels_positive(self, report):
        assert report["n_labels"] > 0, "online_round collected zero labels"

    def test_not_promoted_without_path(self, report):
        # We did not pass promote_to, so promoted must be False.
        assert report["promoted"] is False

    def test_accuracy_and_f1_valid(self, report):
        assert 0.0 <= report["accuracy"] <= 1.0
        assert 0.0 <= report["f1"] <= 1.0


# ---------------------------------------------------------------------------
# 5. Degenerate gate proof — all-drop model trains, but gate blocks regressions
# ---------------------------------------------------------------------------


class TestDegenerateAllDropGateBlocked:
    """Prove that TOST blocks regressions even when the model is maximally wrong.

    We train on an all-zero label set (every line is "safe to drop").  The
    resulting model has low scores for everything, including DECISION: lines.

    certify_promotion's internal strategy only compresses causally-inert
    (decision_relevant=False) blocks — so an all-drop model applied there is
    still safe (those blocks never changed decisions, so dropping their lines
    doesn't cause divergence).

    To prove the GATE itself is the safety net, we build an unrestricted
    strategy that applies the all-drop model to ALL VOLATILE blocks (including
    decision-relevant ones) and directly call certify().  That must FAIL — the
    TOST gate catches the regression.  This separates two orthogonal invariants:
      * certify_promotion is safe because it guards on decision_relevant=False.
      * certify() is the backstop that would catch any escaped regression.
    """

    @pytest.fixture(scope="class")
    def all_drop_labels(self, causal_labels):  # type: ignore[override]
        # Force all labels to 0 (pretend everything is prunable).
        return [(line, 0) for line, _ in causal_labels]

    @pytest.fixture(scope="class")
    def all_drop_weights(self, all_drop_labels):  # type: ignore[override]
        return retrain(all_drop_labels)["weights"]

    def test_all_drop_retrain_succeeds(self, all_drop_weights):
        """Training itself must succeed even on degenerate labels."""
        assert len(all_drop_weights) == len(FEATURE_NAMES)

    def test_all_drop_retrain_accuracy_in_range(self, all_drop_labels):
        # With all-zero labels the model still returns a valid accuracy float.
        result = retrain(all_drop_labels)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_certify_promotion_accepts_all_drop_on_inert_blocks(self, all_drop_weights, entries):
        """certify_promotion is safe: it only touches non-decision-relevant blocks.

        An all-drop model applied exclusively to causally-inert blocks causes no
        decision divergence, so the gate correctly PASSES it.
        """
        result = certify_promotion(all_drop_weights, entries)
        assert result is True, (
            "certify_promotion should pass an all-drop model because it only "
            "compresses causally-inert (non-decision-relevant) blocks"
        )

    def test_gate_rejects_all_drop_applied_to_all_volatile_blocks(self, all_drop_weights, entries):
        """The TOST gate MUST reject an all-drop model when it touches ALL VOLATILE
        blocks (including decision-relevant ones).

        This proves certify() is the backstop: if the decision_relevant guard in
        certify_promotion were ever removed or bypassed, the TOST gate would catch
        the resulting regressions.
        """
        model = LogisticKeepModel(all_drop_weights)
        target_ratio = 0.7

        def _unrestricted_strategy(blocks, _turn):
            out = []
            for block in blocks:
                if block.stability is Stability.VOLATILE and block.kind.value in _VOLATILE_KINDS:
                    # No decision_relevant guard — compress everything.
                    compressed = apply_keep(block.text, block.kind.value, target_ratio, model)
                    out.append(block.copy_with(compressed))
                else:
                    out.append(block)
            return out

        # At least one corpus trajectory must fail certification.
        any_failed = False
        for entry in entries:
            report = certify(entry.trajectory, _unrestricted_strategy, margin=0.02)
            if report.verdict == "FAIL":
                any_failed = True
                break

        assert any_failed, (
            "certify() accepted an all-drop model applied to ALL VOLATILE blocks — "
            "the TOST gate is not catching regressions on decision-relevant content"
        )
