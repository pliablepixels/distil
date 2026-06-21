"""Tests for the learned logistic-regression keep model.

Checks:
  1. Training reduces BCE loss vs. initial (zero) weights.
  2. Held-out accuracy >= 0.90 and F1 >= 0.85.
  3. LogisticKeepModel scores "DECISION: ..." > 0.5 and "debug: trace ..." < 0.5.
  4. Loaded model works as a drop-in KeepModel inside apply_keep.
  5. load() round-trips to_json().
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from distil.codec.features import FEATURE_NAMES, featurize
from distil.codec.keep_model import apply_keep
from distil.codec.learned import (
    DEFAULT_WEIGHTS_PATH,
    LogisticKeepModel,
    _bce_loss,
    _evaluate,
    _split,
    build_dataset,
    train,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dot(w: list[float], x: list[float]) -> float:
    return sum(wi * xi for wi, xi in zip(w, x))


# ---------------------------------------------------------------------------
# 1. Training reduces BCE loss
# ---------------------------------------------------------------------------


class TestTrainingReducesLoss:
    def test_loss_decreases(self) -> None:
        samples, _ = build_dataset()
        assert samples, "Dataset must not be empty"

        w0 = [0.0] * len(FEATURE_NAMES)
        initial_loss = _bce_loss(samples, w0)

        w_trained = train(samples, epochs=400, lr=0.5, l2=1e-4, seed=0)
        final_loss = _bce_loss(samples, w_trained)

        assert final_loss < initial_loss, (
            f"Training did not reduce BCE loss: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_loss_substantial_reduction(self) -> None:
        """Loss should drop by at least 50% from the initial value."""
        samples, _ = build_dataset()
        w0 = [0.0] * len(FEATURE_NAMES)
        initial_loss = _bce_loss(samples, w0)

        w_trained = train(samples, epochs=400, lr=0.5, l2=1e-4, seed=0)
        final_loss = _bce_loss(samples, w_trained)

        assert final_loss < initial_loss * 0.5, (
            f"Loss reduction insufficient: {initial_loss:.4f} -> {final_loss:.4f}"
        )


# ---------------------------------------------------------------------------
# 2. Held-out accuracy >= 0.90 and F1 >= 0.85
# ---------------------------------------------------------------------------


class TestHeldOutMetrics:
    @pytest.fixture(scope="class")
    def trained_weights(self) -> list[float]:
        samples, raw_lines = build_dataset()
        train_set, _ = _split(samples, raw_lines, test_fraction=0.2)
        return train(train_set, epochs=400, lr=0.5, l2=1e-4, seed=0)

    @pytest.fixture(scope="class")
    def test_metrics(self, trained_weights: list[float]) -> dict[str, float]:
        samples, raw_lines = build_dataset()
        _, test_set = _split(samples, raw_lines, test_fraction=0.2)
        return _evaluate(test_set, trained_weights)

    def test_accuracy_at_least_90(self, test_metrics: dict[str, float]) -> None:
        acc = test_metrics["accuracy"]
        assert acc >= 0.90, f"Held-out accuracy {acc:.4f} is below 0.90"

    def test_f1_at_least_85(self, test_metrics: dict[str, float]) -> None:
        f1 = test_metrics["f1"]
        assert f1 >= 0.85, f"Held-out F1 {f1:.4f} is below 0.85"


# ---------------------------------------------------------------------------
# 3. Model scores specific lines correctly
# ---------------------------------------------------------------------------


class TestModelScoresLines:
    @pytest.fixture(scope="class")
    def model(self) -> LogisticKeepModel:
        return LogisticKeepModel.load()

    def test_decision_line_scores_above_half(self, model: LogisticKeepModel) -> None:
        score = model.score("DECISION: roll back to v1.2.3 immediately", "tool_output")
        assert score > 0.5, f"DECISION line scored {score:.4f}, expected > 0.5"

    def test_debug_line_scores_below_half(self, model: LogisticKeepModel) -> None:
        score = model.score("debug: trace entering handler loop", "tool_output")
        assert score < 0.5, f"debug/trace line scored {score:.4f}, expected < 0.5"

    def test_error_line_scores_high(self, model: LogisticKeepModel) -> None:
        """Error lines should be considered highly salient."""
        score = model.score("error: connection refused to database", "tool_output")
        assert score > 0.5, f"error line scored {score:.4f}, expected > 0.5"

    def test_blank_line_scores_low(self, model: LogisticKeepModel) -> None:
        score = model.score("", "tool_output")
        assert score < 0.5, f"blank line scored {score:.4f}, expected < 0.5"


# ---------------------------------------------------------------------------
# 4. Loaded model works as drop-in KeepModel inside apply_keep
# ---------------------------------------------------------------------------


class TestApplyKeepIntegration:
    def test_decision_line_survives_aggressive_ratio(self) -> None:
        """A DECISION line must survive even at target_ratio=0.1."""
        model = LogisticKeepModel.load()
        text = "\n".join(
            [
                "debug: trace start",
                "debug: trace mid",
                "debug: trace end",
                "verbose: lots of boilerplate here",
                "DECISION: approve the rollback to v2.0.1",
                "debug: trace finish",
                "verbose: cleanup",
                "todo: remove this later",
                "noqa: skip checks",
                "fixme: legacy path",
            ]
        )
        result = apply_keep(text, kind="tool_output", target_ratio=0.1, model=model)
        assert "DECISION: approve the rollback to v2.0.1" in result, (
            f"DECISION line was dropped at target_ratio=0.1. Result:\n{result}"
        )

    def test_apply_keep_returns_subset(self) -> None:
        """apply_keep with the learned model returns fewer lines than input."""
        model = LogisticKeepModel.load()
        lines = [f"debug: trace line {i}" for i in range(20)]
        text = "\n".join(lines)
        result = apply_keep(text, kind="tool_output", target_ratio=0.1, model=model)
        result_lines = result.splitlines()
        assert len(result_lines) < len(lines), (
            "apply_keep should return fewer lines than input at ratio=0.1"
        )


# ---------------------------------------------------------------------------
# 5. load() round-trips to_json()
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_to_json_round_trips(self) -> None:
        original = LogisticKeepModel.load()
        serialised = original.to_json()
        data = json.loads(serialised)

        # Reconstruct from the parsed dict.
        restored = LogisticKeepModel(data["weights"])

        # Both models should score identically on a test line.
        line = "DECISION: route to fallback pipeline"
        kind = "tool_output"
        assert original.score(line, kind) == restored.score(line, kind), (
            "Round-tripped model produces different scores"
        )

    def test_to_json_contains_feature_names(self) -> None:
        model = LogisticKeepModel.load()
        data = json.loads(model.to_json())
        assert data["features"] == FEATURE_NAMES

    def test_to_json_contains_correct_weight_count(self) -> None:
        model = LogisticKeepModel.load()
        data = json.loads(model.to_json())
        assert len(data["weights"]) == len(FEATURE_NAMES)

    def test_weights_json_exists(self) -> None:
        assert DEFAULT_WEIGHTS_PATH.exists(), f"weights.json not found at {DEFAULT_WEIGHTS_PATH}"

    def test_load_from_explicit_path(self, tmp_path: Path) -> None:
        """load() accepts an explicit path and round-trips correctly."""
        original = LogisticKeepModel.load()
        tmp_file = tmp_path / "test_weights.json"
        original.save(tmp_file)

        loaded = LogisticKeepModel.load(tmp_file)
        line = "error: disk full"
        kind = "tool_output"
        assert original.score(line, kind) == loaded.score(line, kind)


# ---------------------------------------------------------------------------
# Featurize sanity checks
# ---------------------------------------------------------------------------


class TestFeaturize:
    def test_featurize_length(self) -> None:
        vec = featurize("some line", "tool_output")
        assert len(vec) == len(FEATURE_NAMES)

    def test_blank_line_is_blank_feature(self) -> None:
        vec = featurize("", "tool_output")
        # is_blank is the last feature
        assert vec[-1] == 1.0

    def test_decision_marker_feature(self) -> None:
        vec = featurize("DECISION: do it", "tool_output")
        # has_decision_marker is index 1
        assert vec[1] == 1.0

    def test_all_features_in_unit_interval(self) -> None:
        test_lines = [
            "",
            "DECISION: keep this",
            "error: something failed",
            '{"key": "value"}',
            "debug: trace here",
            "100 requests in 2.5 seconds",
            "A VERY LOUD LINE WITH CAPS",
            "| col1 | col2 | col3 |",
        ]
        for line in test_lines:
            vec = featurize(line, "tool_output")
            for i, v in enumerate(vec):
                assert 0.0 <= v <= 1.0, (
                    f"Feature {FEATURE_NAMES[i]}={v} out of [0,1] for line {line!r}"
                )
