# Distil

**Compression with a quality contract.** Cache-aware, causally-pruned context
compression for agentic runtimes — with a statistical non-inferiority gate that
*proves* it didn't hurt. Same potency, less volume.

> Most context compressors ship a token-savings *estimate*. Distil ships a
> **quality contract**: a strategy is allowed to compress only as far as a
> pre-registered non-inferiority test certifies the agent makes the same
> decisions. The eval engine isn't a ruler bolted on the side — it's a
> *discovery engine* that finds what's safe to drop, and a *gate* that blocks
> what isn't.

## The one reframe that makes "100% accuracy" real

Byte-equivalence and high compression are information-theoretically in tension —
you cannot have both. **Decision-equivalence** is the right target: the agent
takes the same actions and produces the same final outputs whether or not the
context was compressed. That is measurable, certifiable, and compatible with
aggressive compression. "100%" becomes a *statistical non-inferiority guarantee
on task outcomes*, not a string diff.

## What's here (runnable today, zero dependencies)

Two highest-leverage techniques, end-to-end, priced in real dollars:

### #1 — Cache-aware compression (`distil/compress/cache_aware.py`)
In a multi-turn agent loop you re-send the growing context every step. With
prompt caching, a cache *read* is ~10× cheaper than fresh input, so the dominant
cost is cache **misses**, not context **size**. Distil keeps the prefix
byte-stable and compresses only the volatile tail. The simulator prices four
strategies and makes the trap visible:

```
$ distil savings --pricing claude-opus-4-8
strategy                               $ / run   vs baseline  cache hits
------------------------------------------------------------------------
baseline (no cache, no compress)       0.01524          0.0%           0
cache only                             0.01115         26.8%       1,028
naive compress + cache                 0.01691        -11.0%           0   ← busts cache, costs MORE
distil (cache-aware lossless)          0.01019         33.1%       1,028
```

Naive recompression sends *fewer* tokens yet costs **more** than not compressing,
because it rewrites the cacheable prefix every turn. Distil doesn't.

### #4 — Causal / counterfactual pruning (`distil/replay/ablation.py`)
The discovery engine. For each context block: remove it, replay, did any
decision change? Blocks that never change a decision are *provably* free to drop.

```
$ distil prune
doc-0  PRUNE (causally inert)        # speculative retrieval, never cited
hist-0 PRUNE (causally inert)        # stale history
obs-0  keep (changed a decision)     # carries the decision-driving signal
system keep (changed a decision)
tokens provably free to drop: 615 across 11 block(s).
```

### The quality contract (`distil/certify/`)
TOST non-inferiority testing (hand-rolled Student-t, no scipy). Lossless
strategies pass; quality-degrading ones are **rejected**:

```
$ distil certify --strategy distil       # VERDICT: PASS  (100% decision-equivalence)
$ distil certify --strategy aggressive   # VERDICT: FAIL  (mean diff -1.0, would degrade quality)
```

### Savings ledger (`distil/ledger.py`)
Local-first cumulative savings tracking (`distil savings --record`,
`distil leaderboard`). Community sharing is **opt-in and not enabled** — the
ledger never sends anything; it records aggregate numbers only, never content.

## Quickstart

```bash
uv run distil savings --pricing claude-opus-4-8   # technique #1, real dollars
uv run distil prune                               # technique #4, causal ablation
uv run distil certify --strategy distil           # the quality contract
uv run distil compress                            # ratio + reversibility per turn
uv run --with pytest python -m pytest -q          # 20 tests

# billing-grade, against the real model (needs `pip install distil[live]` + ANTHROPIC_API_KEY):
uv run distil savings --tokenizer anthropic       # real Claude count_tokens, not an estimate
uv run distil certify --runner anthropic          # certify against the live model
```

## Architecture: risk-graded tiers

| Tier | Module | Loss profile | When |
|---|---|---|---|
| 0 — provably lossless | `compress/tier0.py` | reconstructable (JSON minify, reversible RLE) | always on |
| 1 — reversible digest | `compress/tier1.py` | lossless in effect (full original behind a handle) | large tool outputs / retrieved docs |
| certified lossy | `certify/` gates it | only at ratios the gate certifies non-inferior | history, prose |

## Capability matrix

| Capability | Module | Loss profile |
|---|---|---|
| Cache-aware, priced cost model (proves naive recompression busts the cache) | `compress/cache_aware.py` | — |
| Cache stabilization: recursive JSON-schema canonicalization | `compress/stabilize.py` | lossless |
| Cache stabilization: lift volatile fields (dates/UUIDs/JWTs) out of the cached prefix | `compress/stabilize.py` | lossless, reversible |
| Content-type-aware codecs routed by block `Kind` | `compress/tier0.py`, `tier1.py` | lossless / reversible |
| Reversible digest with on-demand retrieval handles | `compress/tier1.py` | reversible |
| Reject-if-bigger invariant (never emit a block larger than its original) | `compress/strategies.py` | safety |
| **Causal / counterfactual pruning** — discovers causally-inert context | `replay/ablation.py` | certified |
| **TOST non-inferiority gate** — the quality contract | `certify/` | — |
| Local-first, privacy-preserving savings ledger + leaderboard | `ledger.py` | — |

## Honesty / not-yet (this is a discovery scaffold, not a finished product)

- **Default tokenizer is an offline heuristic** so the core runs with zero deps and no
  key; compression *ratios* are robust to it, absolute dollars are not. For billing-grade
  figures use `--tokenizer anthropic` (real Claude `count_tokens` — the correct tokenizer;
  tiktoken undercounts Claude). Numbers shown here use the heuristic.
- **The default runner is a deterministic stand-in** keyed on `DECISION:` markers, so
  ablation and certification run offline with ground truth. `--runner anthropic` certifies
  against the live model (`distil/replay/anthropic_runner.py`) — implemented but
  **UNVERIFIED** here (no API key in this environment).
- **Pricing is current public list price** (Opus 4.8 $5/$25 per Mtok, etc.) — verify before billing use.
- The corpus is one 4-turn trajectory. Real certification needs a trajectory corpus
  across domains (coding, research, ops, support) — that corpus is the next asset.

## Roadmap

1. ~~Real tokenizer + live `AgentRunner`~~ — **done** (`--tokenizer anthropic`, `--runner anthropic`).
2. Trajectory corpus across non-coding domains; CI gate that blocks any strategy
   that fails non-inferiority.
3. Runtime proxy/hook adapter so Distil works with no code changes.
4. **Auth-mode gating** — aggressive lossy compression only on pay-as-you-go;
   lossless-only on subscription/OAuth sessions (never inject tools or alter a
   metered session in a way that could violate provider terms).
5. **Holdout A/B savings measurement** — reserve a control fraction of runs and
   report savings with a confidence interval, not a synthetic ratio.
6. **Byte-fidelity invariants** — SHA-256 equality modulo explicitly-modified
   ranges; preserve numeric precision; append-only (frozen history never mutates).
7. Learned per-content-type keep-model; BM25-filtered partial retrieval from
   handles; delta/append-only context; gist-token caching of static tool schemas.
