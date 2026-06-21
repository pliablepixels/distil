"""Tests for distil.codec — keep-model codec.

Covers:
- Never-drop lines (DECISION: and error signals) survive at target_ratio=0.1
- Output line order matches input order
- Retained count is >= len(never-drop set) and approximately target_ratio*N
- Pure-noise document compresses hard; decision-dense one barely compresses
- apply_keep is idempotent-ish (re-applying doesn't drop never-drop lines)
- Protocol compliance: a custom KeepModel can be injected
"""

from __future__ import annotations

import math

import pytest

from distil.codec import apply_keep
from distil.codec.keep_model import KeepModel, SalienceKeepModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lines(text: str) -> list[str]:
    return text.splitlines()


# ---------------------------------------------------------------------------
# Unit: SalienceKeepModel.score
# ---------------------------------------------------------------------------


class TestSalienceKeepModelScore:
    def setup_method(self) -> None:
        self.m = SalienceKeepModel()

    def _s(self, line: str) -> float:
        return self.m.score(line, kind="test")

    def test_empty_line_is_zero(self) -> None:
        assert self._s("") == 0.0
        assert self._s("   ") == 0.0

    def test_decision_marker_is_one(self) -> None:
        assert self._s("DECISION: keep this forever") == 1.0
        # marker anywhere in line
        assert self._s("  prefix DECISION: suffix") == 1.0

    def test_error_keywords_score_high(self) -> None:
        for kw in (
            "error",
            "Error",
            "ERROR",
            "fail",
            "exception",
            "traceback",
            "crashloop",
            "denied",
            "breach",
        ):
            score = self._s(f"the {kw} occurred")
            assert score == 0.95, f"expected 0.95 for kw={kw!r}, got {score}"

    def test_debug_keywords_score_low(self) -> None:
        for kw in ("debug", "DEBUG", "trace", "verbose", "noqa", "todo", "fixme"):
            score = self._s(f"{kw}: some detail")
            assert score == 0.1, f"expected 0.1 for kw={kw!r}, got {score}"

    def test_json_object_scores_structured(self) -> None:
        assert self._s('{"key": "value"}') == 0.7

    def test_json_array_scores_structured(self) -> None:
        assert self._s("[1, 2, 3]") == 0.7

    def test_key_value_scores_structured(self) -> None:
        assert self._s("status: ok") == 0.7
        assert self._s("  host-name: localhost") == 0.7

    def test_table_row_scores_structured(self) -> None:
        assert self._s("| col1 | col2 | col3 |") == 0.7

    def test_numeric_line_scores_metric(self) -> None:
        assert self._s("latency 12.3ms p99 45ms") == 0.6
        # "cpu: 0.45 mem: 1024" matches key-value pattern first → 0.7, not 0.6
        assert self._s("cpu: 0.45 mem: 1024") == 0.7
        # pure numbers only, no key-value / json / table → 0.6
        assert self._s("12 34 56") == 0.6

    def test_plain_prose_scores_default(self) -> None:
        score = self._s("this is a plain prose line with no special markers")
        assert score == 0.3

    def test_decision_beats_error(self) -> None:
        # DECISION: checked before error keywords
        score = self._s("DECISION: error occurred")
        assert score == 1.0


# ---------------------------------------------------------------------------
# Integration: apply_keep
# ---------------------------------------------------------------------------


class TestApplyKeepNeverDrop:
    """DECISION: and error lines must survive even at target_ratio=0.1."""

    BODY = "\n".join(
        [
            "some noise line 1",
            "some noise line 2",
            "some noise line 3",
            "some noise line 4",
            "some noise line 5",
            "some noise line 6",
            "some noise line 7",
            "DECISION: use approach A",
            "more noise 8",
            "more noise 9",
            "more noise 10",
            "An error occurred: connection refused",
            "more noise 11",
            "more noise 12",
        ]
    )

    def test_decision_line_survives_low_ratio(self) -> None:
        result = apply_keep(self.BODY, kind="log", target_ratio=0.1)
        assert "DECISION: use approach A" in _lines(result)

    def test_error_line_survives_low_ratio(self) -> None:
        result = apply_keep(self.BODY, kind="log", target_ratio=0.1)
        assert any("error occurred" in ln for ln in _lines(result))

    def test_both_survive_simultaneously(self) -> None:
        result = apply_keep(self.BODY, kind="log", target_ratio=0.1)
        lines = _lines(result)
        assert "DECISION: use approach A" in lines
        assert any("error occurred" in ln for ln in lines)


class TestApplyKeepOrder:
    """Output lines must appear in the same order as input."""

    def test_output_order_matches_input(self) -> None:
        text = "\n".join(
            [
                "line A",
                "DECISION: first decision",
                "line B",
                "An error occurred here",
                "line C",
                "DECISION: second decision",
                "line D",
            ]
        )
        result = apply_keep(text, kind="log", target_ratio=0.5)
        result_lines = _lines(result)
        # Extract positions of decision lines
        d_idx = [i for i, ln in enumerate(result_lines) if "DECISION:" in ln]
        assert len(d_idx) == 2, "both DECISION lines should be kept"
        assert d_idx[0] < d_idx[1], "first decision must come before second"

    def test_kept_lines_are_subsequence_of_input(self) -> None:
        input_lines = [
            "alpha",
            "DECISION: keep",
            "beta",
            "gamma",
            "An error happened",
            "delta",
            "epsilon",
            "zeta",
        ]
        text = "\n".join(input_lines)
        result_lines = _lines(apply_keep(text, kind="log", target_ratio=0.4))
        # Every result line must appear in input_lines in the same relative order.
        it = iter(input_lines)
        for rl in result_lines:
            assert any(rl == il for il in it), f"line {rl!r} not found in expected order"


class TestApplyKeepCount:
    """Retained count must be >= never-drop count and ~= target_ratio * N."""

    def _build_text(self, n_noise: int = 20) -> str:
        lines = [f"noise line {i}" for i in range(n_noise)]
        lines[5] = "DECISION: important choice"
        lines[10] = "error: something bad"
        return "\n".join(lines)

    @pytest.mark.parametrize("ratio", [0.1, 0.25, 0.5, 0.75])
    def test_retained_count_within_budget(self, ratio: float) -> None:
        text = self._build_text(30)
        n = len(_lines(text))
        result_lines = _lines(apply_keep(text, kind="log", target_ratio=ratio))
        kept = len(result_lines)
        target = math.ceil(ratio * n)
        # At most target lines, at least the never-drop count (2 here).
        assert kept >= 2, "never-drop lines must always be present"
        # We allow kept == target (ceil) or slightly over if never-drop pushes us up.
        assert kept <= max(target, 2), f"kept={kept} exceeds target={target}"

    def test_never_drop_set_always_present(self) -> None:
        text = self._build_text(50)
        result_lines = _lines(apply_keep(text, kind="log", target_ratio=0.05))
        assert "DECISION: important choice" in result_lines
        assert any("error: something bad" in ln for ln in result_lines)


class TestApplyKeepCompressionBehavior:
    """Pure-noise compresses hard; decision-dense barely compresses."""

    def test_noise_document_compresses_hard(self) -> None:
        # All debug lines — very low salience, should compress aggressively.
        noise = "\n".join([f"debug: verbose log entry number {i}" for i in range(40)])
        result = apply_keep(noise, kind="log", target_ratio=0.2)
        kept = len(_lines(result))
        original = len(_lines(noise))
        # Should keep roughly 20% (ceil) — definitely well under 50%.
        assert kept <= math.ceil(0.2 * original) + 1

    def test_decision_dense_document_barely_compresses(self) -> None:
        # All DECISION: lines — every line is never-drop, so even at 0.1 ratio
        # the full document is retained.
        decisions = "\n".join([f"DECISION: step {i}" for i in range(20)])
        result = apply_keep(decisions, kind="log", target_ratio=0.1)
        kept = len(_lines(result))
        assert kept == 20, "every DECISION line must survive regardless of ratio"

    def test_mixed_document_partial_compression(self) -> None:
        lines = (
            ["DECISION: approach X"] * 5
            + ["debug: verbose trace {i}" for i in range(10)]
            + ["An error was raised"] * 3
            + [f"noise line {i}" for i in range(20)]
        )
        text = "\n".join(lines)
        result_lines = _lines(apply_keep(text, kind="log", target_ratio=0.3))
        # The 5 DECISION + 3 error lines (8 never-drop-or-near) must all survive.
        decision_count = sum(1 for ln in result_lines if "DECISION:" in ln)
        error_count = sum(1 for ln in result_lines if "error" in ln.lower())
        assert decision_count == 5
        assert error_count == 3


class TestApplyKeepIdempotence:
    """Re-applying at the same ratio must not drop never-drop lines."""

    def test_never_drop_lines_survive_second_pass(self) -> None:
        text = "\n".join(
            [f"noise {i}" for i in range(20)]
            + ["DECISION: final answer", "critical error encountered"]
        )
        first = apply_keep(text, kind="log", target_ratio=0.3)
        second = apply_keep(first, kind="log", target_ratio=0.3)
        lines2 = _lines(second)
        assert "DECISION: final answer" in lines2
        assert any("error" in ln.lower() for ln in lines2)

    def test_idempotent_on_already_filtered_text(self) -> None:
        text = "\n".join(
            [
                "DECISION: keep A",
                "DECISION: keep B",
                "some noise",
                "An error occurred",
            ]
        )
        # First pass — probably keeps everything or nearly so.
        first = apply_keep(text, kind="log", target_ratio=0.8)
        second = apply_keep(first, kind="log", target_ratio=0.8)
        # DECISION lines must survive both passes.
        assert "DECISION: keep A" in _lines(second)
        assert "DECISION: keep B" in _lines(second)


class TestApplyKeepCustomModel:
    """A custom KeepModel can be injected as a drop-in."""

    def test_custom_model_is_respected(self) -> None:
        class AllDropModel:
            """Scores everything 0.0 except lines containing 'KEEP'."""

            def score(self, line: str, kind: str) -> float:  # noqa: ARG002
                return 1.0 if "KEEP" in line else 0.0

        text = "\n".join(
            [
                "this will be dropped",
                "KEEP this line",
                "also dropped",
                "KEEP and this one",
            ]
        )
        result = apply_keep(text, kind="test", target_ratio=0.1, model=AllDropModel())
        result_lines = _lines(result)
        assert "KEEP this line" in result_lines
        assert "KEEP and this one" in result_lines
        assert "this will be dropped" not in result_lines

    def test_protocol_conformance(self) -> None:
        # Verify SalienceKeepModel satisfies KeepModel protocol structurally.
        m: KeepModel = SalienceKeepModel()
        score = m.score("DECISION: test", "tool_output")
        assert score == 1.0


class TestApplyKeepEdgeCases:
    def test_empty_string_returns_empty(self) -> None:
        assert apply_keep("", kind="log", target_ratio=0.5) == ""

    def test_single_line_always_kept(self) -> None:
        result = apply_keep("hello", kind="log", target_ratio=0.5)
        assert result == "hello"

    def test_ratio_one_keeps_everything(self) -> None:
        text = "\n".join(f"line {i}" for i in range(10))
        result = apply_keep(text, kind="log", target_ratio=1.0)
        assert _lines(result) == _lines(text)

    def test_none_model_uses_default(self) -> None:
        text = "DECISION: always keep\nsome noise"
        result = apply_keep(text, kind="log", target_ratio=0.1, model=None)
        assert "DECISION: always keep" in _lines(result)

    def test_all_empty_lines_compress_to_minimum(self) -> None:
        # "\n".join([""] * 20) is 19 newlines → splitlines() yields 19 lines.
        text = "\n".join([""] * 20)
        n = len(_lines(text))  # 19
        result = apply_keep(text, kind="log", target_ratio=0.1)
        # All score 0.0; ceil(0.1 * 19) = 2 kept, but joining 2 empty strings
        # gives "\n" and splitlines() collapses that to 1 item — verify we
        # retain at least 1 line (the minimum non-empty result).
        kept = len(_lines(result))
        assert kept >= 1
        assert kept <= math.ceil(0.1 * n)
