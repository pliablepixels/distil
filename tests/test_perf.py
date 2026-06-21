"""Tests for distil.perf — benchmark correctness and smoke-test."""

from __future__ import annotations


import pytest

from distil.perf import (
    PerfResult,
    _percentiles,
    bench_adapter,
    bench_compression,
    format_table,
    run_perf,
)


# ---------------------------------------------------------------------------
# _percentiles
# ---------------------------------------------------------------------------


class TestPercentiles:
    def test_known_list(self) -> None:
        # [1,2,3,4,5,6,7,8,9,10] -> p50=5, p95=10, mean=5.5
        samples = [float(x) for x in range(1, 11)]
        p50, p95, mean = _percentiles(samples)
        # nearest-rank p50: ceil(50*10/100)-1 = ceil(5)-1 = 4 -> sorted[4] = 5
        assert p50 == pytest.approx(5.0)
        # nearest-rank p95: ceil(95*10/100)-1 = ceil(9.5)-1 = 10-1 = 9 -> sorted[9] = 10
        assert p95 == pytest.approx(10.0)
        assert mean == pytest.approx(5.5)

    def test_single_element(self) -> None:
        p50, p95, mean = _percentiles([42.0])
        assert p50 == pytest.approx(42.0)
        assert p95 == pytest.approx(42.0)
        assert mean == pytest.approx(42.0)

    def test_two_elements(self) -> None:
        p50, p95, mean = _percentiles([10.0, 20.0])
        # p50: ceil(50*2/100)-1 = ceil(1)-1 = 0 -> 10.0
        assert p50 == pytest.approx(10.0)
        # p95: ceil(95*2/100)-1 = ceil(1.9)-1 = 2-1 = 1 -> 20.0
        assert p95 == pytest.approx(20.0)
        assert mean == pytest.approx(15.0)

    def test_empty_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            _percentiles([])

    def test_p95_gte_p50(self) -> None:
        import random

        rng = random.Random(42)
        samples = [rng.uniform(0.1, 100.0) for _ in range(500)]
        p50, p95, mean = _percentiles(samples)
        assert p95 >= p50

    def test_all_same(self) -> None:
        samples = [7.0] * 100
        p50, p95, mean = _percentiles(samples)
        assert p50 == pytest.approx(7.0)
        assert p95 == pytest.approx(7.0)
        assert mean == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# bench_compression
# ---------------------------------------------------------------------------


class TestBenchCompression:
    def test_returns_perf_result(self) -> None:
        result = bench_compression(iterations=5)
        assert isinstance(result, PerfResult)

    def test_positive_latencies(self) -> None:
        result = bench_compression(iterations=5)
        assert result.p50_ms > 0
        assert result.p95_ms > 0
        assert result.mean_ms > 0

    def test_positive_throughput(self) -> None:
        result = bench_compression(iterations=5)
        assert result.items_per_s > 0

    def test_p95_gte_p50(self) -> None:
        result = bench_compression(iterations=5)
        assert result.p95_ms >= result.p50_ms

    def test_name(self) -> None:
        result = bench_compression(iterations=5)
        assert result.name  # non-empty string

    def test_sample_count(self) -> None:
        # n should be iterations * number_of_turns (at least iterations*1)
        result = bench_compression(iterations=5)
        assert result.n >= 5


# ---------------------------------------------------------------------------
# bench_adapter
# ---------------------------------------------------------------------------


class TestBenchAdapter:
    def test_returns_perf_result(self) -> None:
        result = bench_adapter(iterations=10)
        assert isinstance(result, PerfResult)

    def test_positive_latencies(self) -> None:
        result = bench_adapter(iterations=10)
        assert result.p50_ms > 0
        assert result.p95_ms > 0
        assert result.mean_ms > 0

    def test_positive_throughput(self) -> None:
        result = bench_adapter(iterations=10)
        assert result.items_per_s > 0

    def test_p95_gte_p50(self) -> None:
        result = bench_adapter(iterations=10)
        assert result.p95_ms >= result.p50_ms

    def test_token_reduction_positive(self) -> None:
        result = bench_adapter(iterations=10)
        assert "token_reduction" in result.extras
        assert result.extras["token_reduction"] > 0

    def test_token_counts_present(self) -> None:
        result = bench_adapter(iterations=10)
        assert "tokens_before" in result.extras
        assert "tokens_after" in result.extras
        assert result.extras["tokens_before"] > result.extras["tokens_after"]

    def test_sample_count(self) -> None:
        result = bench_adapter(iterations=10)
        assert result.n == 10


# ---------------------------------------------------------------------------
# format_table
# ---------------------------------------------------------------------------


class TestFormatTable:
    def _make_result(self, name: str) -> PerfResult:
        return PerfResult(
            name=name,
            n=100,
            p50_ms=0.123,
            p95_ms=0.456,
            mean_ms=0.200,
            items_per_s=5000.0,
        )

    def test_non_empty(self) -> None:
        results = [self._make_result("distil_strategy")]
        table = format_table(results)
        assert len(table) > 0

    def test_contains_result_names(self) -> None:
        results = [
            self._make_result("distil_strategy"),
            self._make_result("compress_messages"),
        ]
        table = format_table(results)
        assert "distil_strategy" in table
        assert "compress_messages" in table

    def test_contains_headers(self) -> None:
        results = [self._make_result("foo")]
        table = format_table(results)
        assert "p50" in table
        assert "p95" in table

    def test_two_results(self) -> None:
        results = [self._make_result("a"), self._make_result("b")]
        table = format_table(results)
        assert "a" in table
        assert "b" in table

    def test_extras_appear(self) -> None:
        r = PerfResult(
            name="with_extras",
            n=10,
            p50_ms=1.0,
            p95_ms=2.0,
            mean_ms=1.5,
            items_per_s=1000.0,
            extras={"token_reduction": 42},
        )
        table = format_table([r])
        assert "token_reduction" in table
        assert "42" in table


# ---------------------------------------------------------------------------
# run_perf integration
# ---------------------------------------------------------------------------


class TestRunPerf:
    def test_returns_list_of_two(self) -> None:
        results = run_perf(iterations=5)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_all_perf_results(self) -> None:
        results = run_perf(iterations=5)
        for r in results:
            assert isinstance(r, PerfResult)

    def test_positive_values(self) -> None:
        results = run_perf(iterations=5)
        for r in results:
            assert r.p50_ms > 0
            assert r.p95_ms > 0
            assert r.mean_ms > 0
            assert r.items_per_s > 0
