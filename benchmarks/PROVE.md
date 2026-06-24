# `prove.py` — decision-equivalence, measured (not asserted)

This harness turns the central claim into a result by running it on **real agent
traces** graded by a **real model** — i.e. with the circular `DECISION:`-marker
oracle removed (see `docs/PAPER_PLAN.md` for why that matters).

## What it runs

| | | |
|---|---|---|
| **E1** | Frontier | token savings vs. decision-change rate, per ladder level |
| **E2** | Certification coverage — *the proof* | certify at α on a calibration split, then measure the **realized** decision-change rate on a **disjoint held-out** split, over many random trajectory-level splits. The certificate is sound iff empirical `P(realized ≤ α) ≥ 1−δ`. |
| **E3** | Distribution shift | leave-one-domain-out: calibrate on all domains but one, test on the held-out one (the exchangeability stress test) |
| **E4** | Downstream task success | converts per-turn equivalence into the **outcome**: a trajectory keeps its result iff *every* decision is unchanged, so `retained-success(level)` = originally-successful ∧ fully-equivalent, vs. the uncompressed baseline, with a bootstrap CI. Needs outcome labels (τ-bench reward / SWE-bench resolved). |

Decisions are cached on disk per rendered-context hash, so the live-model pass is
paid **once** and the statistics are reproducible.

## Grading backends (`--runner`)

| `--runner` | What it uses | When |
|---|---|---|
| `smoke` | offline heuristic, **non-evidential** | plumbing / CI, no key |
| `claude-cli` | the **`claude -p` CLI** — your Claude Code subscription, **no API key** | easiest real-model run if you already use Claude Code |
| `openai` | any **OpenAI-compatible endpoint** (vLLM/Ollama/LM Studio/OpenAI) via `--base-url` | **free at scale** with a local open model |
| `anthropic` | the Anthropic API (needs `ANTHROPIC_API_KEY`) | billing-grade reference |

```bash
# your Claude subscription, no API key (Haiku = cheap large sweeps; Opus = headline):
python benchmarks/prove.py --dataset tau --path runs.json \
    --runner claude-cli --model claude-haiku-4-5-20251001 --samples 3

# a local open model via vLLM (zero per-call cost):
#   vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
python benchmarks/prove.py --dataset swe --path swe_trajs/ \
    --runner openai --base-url http://localhost:8000/v1 --model meta-llama/Llama-3.1-8B-Instruct
```

## Offline plumbing check (no key, no download)

```bash
python benchmarks/fixtures/make_fixtures.py        # regenerate the fixtures
python benchmarks/prove.py --dataset fixtures --runner smoke --alpha 0.2
```

The `smoke` runner is a **NON-EVIDENTIAL** stand-in (it models "did the load-bearing
record survive?", treating reversible folds as recoverable). It exists only to verify
the harness mechanics; it is not evidence about real agents. On the bundled fixtures
it reproduces the expected shape:

```
E1: lossless = 16% savings @ 0.0% decision-change ; aggressive truncation flips ~43%
E2 (α=0.2):  certified 100% of splits · empirical coverage 100% (≥95%) · 16% held-out savings
E3:          lossless certificate transfers across tau→swe and swe→tau
```

At a tight α (e.g. 0.05) on this tiny fixture it **correctly refuses to certify** —
too few calibration turns — which is the honest, conservative behavior (more turns
certify tighter α, exactly as the README's 320→α2%, 640→α1% result shows).

## The real run (the publishable result)

Needs `ANTHROPIC_API_KEY` and downloaded traces.

```bash
# τ-bench (decisions = real tool calls; nothing tells the model what to pick)
python benchmarks/prove.py --dataset tau --path /data/taubench_runs.json \
    --runner anthropic --samples 3 --alpha 0.05 --report tau_proof.json

# SWE-bench (SWE-agent .traj dir or a single json list of trajectories)
python benchmarks/prove.py --dataset swe --path /data/swe_trajs/ \
    --runner anthropic --samples 3 --alpha 0.05 --report swe_proof.json
```

With `--runner anthropic` the harness also prints **model↔gold next-action
agreement** on the uncompressed context — a sanity gate: if the grader doesn't
reproduce the agents' real actions, fix that before trusting E1/E2.

## Expected trace formats

- **τ-bench**: JSON list of episodes, each `{"messages":[{role, content, tool_calls}], "tools":[...]}`.
  Assistant tool calls are the decision points.
- **SWE-bench**: SWE-agent `.traj` `{"problem_statement", "trajectory":[{action, observation}], "info":{resolved}}`
  — a directory of them, or one JSON file holding a list.

Both adapters live in `distil/replay/realtrace.py`; the parsers are defensive about
the common public shapes. The gold action recorded in each trace is kept for
downstream metrics (agreement, task success) but **never injected into context**.
