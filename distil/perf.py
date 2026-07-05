"""Production-grade performance benchmark for distil.

Measures latency and throughput of the two hottest paths:
  * The ``distil`` compression strategy (per-turn block compression)
  * The Anthropic adapter ``compress_messages`` (per-request message compression)

All timing uses ``time.perf_counter``; no external dependencies beyond stdlib.

Usage
-----
::

    from distil.perf import run_perf, format_table
    results = run_perf()
    print(format_table(results))

Or via the CLI::

    python -m distil.perf
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .adapters.anthropic import compress_messages
from .compress.strategies import distil as distil_strategy
from .corpus import load_corpus
from .tokenizer import DEFAULT as _tok


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PerfResult:
    name: str
    n: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    items_per_s: float
    # Optional extra fields
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core statistics — hand-rolled; no numpy
# ---------------------------------------------------------------------------


def _percentiles(samples_ms: list[float]) -> tuple[float, float, float]:
    """Return (p50, p95, mean) over *samples_ms*.

    Percentiles use the nearest-rank method on a sorted copy; mean is the
    arithmetic mean.  The list must be non-empty.
    """
    if not samples_ms:
        raise ValueError("samples_ms must be non-empty")

    n = len(samples_ms)
    sorted_s = sorted(samples_ms)

    def _pct(p: float) -> float:
        # Nearest-rank: index = ceil(p/100 * n) - 1 (clamped)
        idx = max(0, min(n - 1, int(-(-int(p * n / 100) // 1))))  # ceiling division
        # Simpler: ceil(p * n / 100) - 1
        import math

        idx = max(0, min(n - 1, math.ceil(p * n / 100) - 1))
        return sorted_s[idx]

    p50 = _pct(50)
    p95 = _pct(95)
    mean = sum(sorted_s) / n
    return p50, p95, mean


# ---------------------------------------------------------------------------
# bench_compression
# ---------------------------------------------------------------------------


def bench_compression(iterations: int = 200) -> PerfResult:
    """Benchmark the ``distil`` strategy across all corpus turns.

    Each sample is the wall-clock time to run ``distil_strategy(blocks, turn_index)``
    on a single turn's block list.  We cycle through all turns across *iterations*
    full passes so the sample count is ``iterations * total_turns``.

    Returns a :class:`PerfResult` with p50/p95/mean ms per-turn and turns/sec.
    """
    entries = load_corpus()

    # Collect all (blocks, turn_index) pairs from the corpus once.
    all_turns: list[tuple[list, int]] = []
    for entry in entries:
        for turn in entry.trajectory.turns:
            all_turns.append((turn.blocks, turn.index))

    if not all_turns:
        raise RuntimeError("Corpus is empty — cannot benchmark compression")

    samples_ms: list[float] = []
    for _ in range(iterations):
        for blocks, turn_idx in all_turns:
            t0 = time.perf_counter()
            distil_strategy(blocks, turn_idx)
            t1 = time.perf_counter()
            samples_ms.append((t1 - t0) * 1_000)

    p50, p95, mean = _percentiles(samples_ms)
    total_s = sum(samples_ms) / 1_000
    items_per_s = len(samples_ms) / total_s if total_s > 0 else float("inf")

    return PerfResult(
        name="distil_strategy",
        n=len(samples_ms),
        p50_ms=p50,
        p95_ms=p95,
        mean_ms=mean,
        items_per_s=items_per_s,
    )


# ---------------------------------------------------------------------------
# bench_adapter
# ---------------------------------------------------------------------------

_LARGE_TOOL_RESULT = "\n".join(
    [
        "Tool: bash",
        "Exit code: 0",
        "Stdout:",
        "  NAME                                   READY   STATUS    RESTARTS   AGE",
        "  frontend-7d9f6b4c8-xk2pl               1/1     Running   0          3d",
        "  backend-api-6c8d5f9b7-mn3qr             1/1     Running   0          3d",
        "  postgres-primary-0                      1/1     Running   0          7d",
        "  redis-cache-5f8c4d7b6-pq9rs             1/1     Running   0          5d",
        "  celery-worker-8b7a6c5d4-uv1wx           1/1     Running   3          3d",
        "  celery-beat-9c8b7d6e5-yz2ab             0/1     Pending   0          1m",
        "Stderr:",
        "  (none)",
        "Previous command output cached.",
        "  cluster: prod-us-east-1",
        "  region: us-east-1",
        "  namespace: application",
        "  context: arn:aws:eks:us-east-1:123456789:cluster/prod",
        "  kubectl version: 1.28.3",
        "  kube-apiserver version: 1.28.2",
        "  node count: 12",
        "  pod count: 47",
    ]
)


def _build_messages() -> list[dict[str, Any]]:
    """Build a realistic multi-turn Anthropic messages list for benchmarking."""
    return [
        {
            "role": "user",
            "content": "Check the status of all pods and identify any that are not running.",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_abc123",
                    "name": "bash",
                    "input": {"command": "kubectl get pods -n application"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_abc123",
                    "content": _LARGE_TOOL_RESULT,
                }
            ],
        },
        {
            "role": "assistant",
            "content": (
                "I can see that `celery-beat-9c8b7d6e5-yz2ab` is in Pending status "
                "with 0 restarts and was started 1 minute ago. All other pods are "
                "Running. Let me describe the pending pod to understand why it is stuck."
            ),
        },
        {
            "role": "user",
            "content": "Good. What is the cause of the Pending status?",
        },
        {
            "role": "assistant",
            "content": "The Pending pod is unschedulable — no node has enough free memory.",
        },
        {
            # Two more turns follow the large tool_result so it is no longer one of
            # the most-recent turns the adapter keeps verbatim (recency exemption)
            # and the benchmark still exercises Tier-1 digestion of older history.
            "role": "user",
            "content": "Thanks, that explains it.",
        },
    ]


def bench_adapter(iterations: int = 500) -> PerfResult:
    """Benchmark ``compress_messages`` over a fixed realistic request.

    Times *iterations* calls to :func:`compress_messages`, then reports
    p50/p95/mean ms and requests/sec.  Also measures token reduction on the
    first compressed result using :data:`distil.tokenizer.DEFAULT`.
    """
    messages = _build_messages()

    # Measure token counts once (before / after) for the reduction extra.
    def _total_tokens(msgs: list[dict[str, Any]]) -> int:
        total = 0
        for msg in msgs:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += _tok.count(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            total += _tok.count(block.get("text", ""))
                        elif block.get("type") == "tool_result":
                            c = block.get("content", "")
                            if isinstance(c, str):
                                total += _tok.count(c)
                            elif isinstance(c, list):
                                for sub in c:
                                    if isinstance(sub, dict) and sub.get("type") == "text":
                                        total += _tok.count(sub.get("text", ""))
                        elif block.get("type") == "tool_use":
                            pass  # skip tool_use input dict
        return total

    tokens_before = _total_tokens(messages)
    compressed_once, _ = compress_messages(messages)
    tokens_after = _total_tokens(compressed_once)
    token_reduction = tokens_before - tokens_after

    # Now benchmark the hot path.
    samples_ms: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        compress_messages(messages)
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1_000)

    p50, p95, mean = _percentiles(samples_ms)
    total_s = sum(samples_ms) / 1_000
    items_per_s = len(samples_ms) / total_s if total_s > 0 else float("inf")

    return PerfResult(
        name="compress_messages",
        n=len(samples_ms),
        p50_ms=p50,
        p95_ms=p95,
        mean_ms=mean,
        items_per_s=items_per_s,
        extras={
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "token_reduction": token_reduction,
        },
    )


# ---------------------------------------------------------------------------
# run_perf / format_table
# ---------------------------------------------------------------------------


def run_perf(iterations: int | None = None) -> list[PerfResult]:
    """Run all benchmarks and return a list of :class:`PerfResult`.

    Parameters
    ----------
    iterations:
        Override the default iteration count for all benchmarks.  ``None``
        uses each benchmark's own default.
    """
    kwargs: dict[str, Any] = {}
    if iterations is not None:
        kwargs["iterations"] = iterations

    results: list[PerfResult] = [
        bench_compression(**kwargs),
        bench_adapter(**kwargs),
    ]
    return results


def format_table(results: list[PerfResult]) -> str:
    """Return a clean fixed-width table of benchmark results.

    Columns: name, n, p50 ms, p95 ms, mean ms, ops/sec.
    """
    # Column headers and widths
    headers = ["benchmark", "n", "p50 ms", "p95 ms", "mean ms", "ops/sec"]
    col_widths = [max(20, max(len(r.name) for r in results) + 2), 8, 10, 10, 10, 12]

    def _fmt_row(cells: list[str]) -> str:
        return "  ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))

    sep = "  ".join("-" * w for w in col_widths)

    rows: list[str] = [
        _fmt_row(headers),
        sep,
    ]
    for r in results:
        rows.append(
            _fmt_row(
                [
                    r.name,
                    str(r.n),
                    f"{r.p50_ms:.3f}",
                    f"{r.p95_ms:.3f}",
                    f"{r.mean_ms:.3f}",
                    f"{r.items_per_s:,.0f}",
                ]
            )
        )

    # Append extras (token reduction etc.) as footnotes
    for r in results:
        if r.extras:
            rows.append("")
            for k, v in r.extras.items():
                rows.append(f"  [{r.name}] {k}: {v}")

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    iters = int(sys.argv[1]) if len(sys.argv) > 1 else None
    results = run_perf(iterations=iters)
    print(format_table(results))
