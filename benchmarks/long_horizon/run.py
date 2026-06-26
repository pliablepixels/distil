#!/usr/bin/env python3
"""Long-horizon ReAct agent driver for the distil E7 benchmark (Phase 5 extension).

Runs a multi-turn ReAct coding agent (see :mod:`benchmarks.long_horizon.agent`) on
SWE-bench Verified instances, routing every Anthropic API call through the compression
proxy from :mod:`benchmarks.swe_bench_e2e.compress_proxy`. The agent explores the
repository over many turns, accumulating large read_file outputs as peripheral context
— the signal that exercises the relevance gate in the ``distil_gated`` condition.

Prediction rows are written to ``--out-dir/predictions/<condition>.jsonl`` in the same
format as run_agent.py so score.py and aggregate.py work unchanged:

    instance_id, model_name_or_path, model_patch, _condition, _agent, _model,
    _temperature, _run, _compress, _empty_patch

Resumable: instances already present in the output JSONL are skipped.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Reuse git helpers and proxy machinery from the existing SWE-bench eval — DO NOT copy.
from benchmarks.swe_bench_e2e.compress_proxy import (
    COMPRESSORS,
    EXPAND_CONDITION,
    GATE_RECENT,
    GATED_CONDITION,
    CompressStats,
    serve,
)
from benchmarks.swe_bench_e2e.run_agent import (
    capture_patch,
    ensure_clone,
    make_worktree,
    remove_worktree,
)
from benchmarks.long_horizon.agent import MODEL, TEMPERATURE, run_agent, run_agent_openai
from benchmarks.swe_bench_e2e.compress_proxy_openai import serve as serve_openai

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("full", "distil_trunc500", "llmlingua2", "distil_expand", "distil_gated")

# Serialise worktree add/remove per clone (mirrors run_agent._CLONE_LOCKS).
_CLONE_LOCKS: dict[str, threading.Lock] = defaultdict(threading.Lock)
_CLONE_LOCKS_GUARD = threading.Lock()


def _get_clone_lock(clone: Path) -> threading.Lock:
    with _CLONE_LOCKS_GUARD:
        return _CLONE_LOCKS[str(clone)]


# --------------------------------------------------------------------------- #
# Per-instance driver
# --------------------------------------------------------------------------- #


def run_instance(
    inst: dict[str, Any],
    condition: str,
    api_key: str,
    cache_dir: Path,
    work_root: Path,
    timeout: float,
    max_turns: int,
    *,
    backend: str = "anthropic",
    upstream: str = "http://127.0.0.1:11434/v1",
    model: str = MODEL,
) -> dict[str, Any]:
    """Run one SWE-bench instance under ``condition`` and return a prediction row.

    ``backend`` selects the agent + proxy pair:
    * ``"anthropic"`` — Anthropic Messages API via :func:`run_agent` (default).
    * ``"openai"``    — OpenAI Chat-Completions via :func:`run_agent_openai`;
      the proxy is :func:`compress_proxy_openai.serve` and ``upstream`` is the
      local model base URL (e.g. Ollama at ``http://127.0.0.1:11434/v1``).
    """
    iid = inst["instance_id"]
    clone = ensure_clone(inst["repo"], cache_dir)
    with _get_clone_lock(clone):
        wt = make_worktree(clone, inst["base_commit"], work_root)

    # The problem statement is the task definition — protect it from compression so
    # conditions are compared on equal footing (same task, only context compressed).
    if backend == "openai":
        httpd, state = serve_openai(
            compressor=COMPRESSORS[condition],
            upstream=upstream,
            protect=inst["problem_statement"],
            expand=(condition in (EXPAND_CONDITION, GATED_CONDITION)),
        )
    else:
        httpd, state = serve(
            compressor=COMPRESSORS[condition],
            protect=inst["problem_statement"],
            expand=(condition in (EXPAND_CONDITION, GATED_CONDITION)),
            gate_recent=(GATE_RECENT if condition == GATED_CONDITION else None),
        )
    base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        if backend == "openai":
            meta = run_agent_openai(
                worktree=wt,
                problem_statement=inst["problem_statement"],
                base_url=base_url,
                api_key=api_key,
                model=model,
                max_turns=max_turns,
                timeout=timeout,
            )
        else:
            meta = run_agent(
                problem_statement=inst["problem_statement"],
                worktree=wt,
                base_url=base_url,
                api_key=api_key,
                max_turns=max_turns,
                timeout=timeout,
            )
        patch = capture_patch(wt)
    finally:
        httpd.shutdown()
        with _get_clone_lock(clone):
            remove_worktree(clone, wt)

    stats: CompressStats = state.stats
    return {
        "instance_id": iid,
        "model_name_or_path": f"distil-lh-{condition}",
        "model_patch": patch,
        "_condition": condition,
        "_agent": "long_horizon_react",
        "_model": model,
        "_temperature": TEMPERATURE,
        "_run": meta,
        "_compress": asdict(stats),
        "_empty_patch": not patch.strip(),
    }


# --------------------------------------------------------------------------- #
# Sample loading
# --------------------------------------------------------------------------- #


def load_sample(path: Path) -> list[dict[str, Any]]:
    """Load instances from the sample JSON (same format as run_agent.load_sample)."""
    data = json.loads(path.read_text())
    return data["instances"]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument(
        "--sample",
        type=Path,
        default=ROOT / "docs/paper/results/swe_e2e/sample_full.json",
        help="path to sample JSON (same schema as run_agent --sample)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs/paper/results/swe_e2e_longhorizon",
        help="output root; predictions written to <out-dir>/predictions/<condition>.jsonl",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".e7_cache/repos",
        help="bare-clone cache (shared with run_agent)",
    )
    ap.add_argument(
        "--work-root",
        type=Path,
        default=ROOT / ".e7_cache/work_lh",
        help="worktree scratch root (separate from run_agent to avoid collisions)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="per-instance wall-clock budget in seconds (default 900)",
    )
    ap.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="max agent turns per instance (default 30; short conversations won't exercise the gate)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only first N instances (smoke test)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent instances; each gets its own proxy + worktree",
    )
    ap.add_argument(
        "--backend",
        choices=("anthropic", "openai"),
        default="anthropic",
        help="agent + proxy backend: 'anthropic' (default) or 'openai' (local model via Ollama/vLLM)",
    )
    ap.add_argument(
        "--upstream",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"),
        help=(
            "OpenAI-compatible upstream base URL (only used when --backend openai). "
            "Defaults to OPENAI_BASE_URL env var or http://127.0.0.1:11434/v1 (Ollama)."
        ),
    )
    ap.add_argument(
        "--model",
        default=None,
        help=(
            "Model identifier sent in each request. Defaults to claude-sonnet-4-6 for "
            "--backend anthropic; required for --backend openai (e.g. qwen2.5-coder:32b)."
        ),
    )
    ap.add_argument(
        "--api-key-env",
        default=None,
        help=(
            "Environment variable holding the API key. Defaults to ANTHROPIC_API_KEY for "
            "--backend anthropic, OPENAI_API_KEY for --backend openai. "
            "For Ollama any non-empty value works; pass --api-key-env '' to use 'ollama'."
        ),
    )
    args = ap.parse_args()

    # Resolve model default per backend.
    effective_model = args.model
    if effective_model is None:
        if args.backend == "openai":
            raise SystemExit(
                "--model is required for --backend openai (e.g. --model qwen2.5-coder:32b)"
            )
        effective_model = MODEL  # claude-sonnet-4-6

    from dotenv import dotenv_values

    env_vals = dotenv_values(ROOT / ".env")

    if args.backend == "openai":
        key_env = args.api_key_env if args.api_key_env is not None else "OPENAI_API_KEY"
        api_key = (env_vals.get(key_env) or os.environ.get(key_env, "")) if key_env else "ollama"
        if not api_key:
            # Ollama accepts any non-empty bearer value — fall back gracefully.
            api_key = "ollama"
    else:
        key_env = args.api_key_env if args.api_key_env is not None else "ANTHROPIC_API_KEY"
        api_key = env_vals.get(key_env) or os.environ.get(key_env, "")
        if not api_key:
            raise SystemExit(f"no {key_env} available (.env or env)")

    instances = load_sample(args.sample)
    if args.limit:
        instances = instances[: args.limit]

    pred_dir = args.out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / f"{args.condition}.jsonl"

    # Resume: skip instances already written.
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["instance_id"])

    todo = [i for i in instances if i["instance_id"] not in done]
    print(
        f"condition={args.condition}: {len(todo)} to run, {len(done)} already done, "
        f"workers={args.workers} max_turns={args.max_turns}",
        flush=True,
    )

    # Pre-clone repos serially — clone races badly across threads.
    for repo in sorted({i["repo"] for i in todo}):
        ensure_clone(repo, args.cache_dir)

    write_lock = threading.Lock()
    done_count = [0]

    def work(inst: dict[str, Any]) -> None:
        iid = inst["instance_id"]
        t0 = time.time()
        try:
            row = run_instance(
                inst,
                args.condition,
                api_key,
                args.cache_dir,
                args.work_root,
                args.timeout,
                args.max_turns,
                backend=args.backend,
                upstream=args.upstream,
                model=effective_model,
            )
        except Exception as e:  # noqa: BLE001 — record failure, keep going
            row = {
                "instance_id": iid,
                "model_name_or_path": f"distil-lh-{args.condition}",
                "model_patch": "",
                "_condition": args.condition,
                "_agent": "long_horizon_react",
                "_model": effective_model,
                "_temperature": TEMPERATURE,
                "_run": {"status": "error", "turns": 0, "seconds": round(time.time() - t0, 1)},
                "_error": str(e)[:500],
                "_empty_patch": True,
            }
        c = row.get("_compress", {})
        r = row.get("_run", {})
        with write_lock:
            with out_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            done_count[0] += 1
            # Print progress line matching run_agent.py's format (+ turns + expansions).
            print(
                f"[{args.condition} {done_count[0]}/{len(todo)}] {iid} "
                f"status={r.get('status', 'err')} "
                f"turns={r.get('turns', 0)} "
                f"empty={row['_empty_patch']} "
                f"blk={c.get('blocks_compressed', 0)}/{c.get('blocks_seen', 0)} "
                f"exp={c.get('expansions', 0)} "
                f"in={c.get('usage_input_tokens', 0)} out={c.get('usage_output_tokens', 0)} "
                f"t={r.get('seconds', 0)}s",
                flush=True,
            )

    if args.workers <= 1:
        for inst in todo:
            work(inst)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))


if __name__ == "__main__":
    main()
