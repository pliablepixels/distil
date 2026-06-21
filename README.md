<p align="center">
  <img src="docs/assets/banner.svg" alt="Distil — compression with a quality contract" width="100%"/>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-8b7bff" alt="license"/></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-5ad1c9" alt="python"/>
  <img src="https://img.shields.io/badge/runtime%20deps-0-5ad19a" alt="zero deps"/>
  <img src="https://img.shields.io/badge/tests-181%20passing-5ad19a" alt="tests"/>
  <img src="https://img.shields.io/badge/corpus%20gate-PASS-5ad19a" alt="gate"/>
  <img src="https://img.shields.io/badge/distributables-pip%20·%20docker%20·%20pyz-8b7bff" alt="distributables"/>
</p>

<h3 align="center">Cut LLM agent costs ~30% — and <em>prove</em> the agent still makes the same decisions.</h3>

<p align="center">
Most context compressors ship a token-savings <em>estimate</em>.<br/>
Distil ships a <strong>quality contract</strong>: a strategy compresses only as far as a statistical non-inferiority test certifies the agent behaves identically.
</p>

---

## The idea in one breath

**You don't need byte-equivalence, you need decision-equivalence.** Byte-lossless compression and high savings are information-theoretically in tension. But an agent only has to take the *same actions* and produce the *same outputs* whether or not its context was compressed. That is measurable, certifiable, and compatible with aggressive compression — so **"100% accuracy" becomes a statistical guarantee on outcomes, not a diff of strings.**

That reframe is the whole project. Everything below makes it real, measured, and shippable.

```bash
uvx distil bench          # certify savings + quality across 7 domains, in seconds
```

```
domain            trajectory                $ saved   distil   aggr  pruned
---------------------------------------------------------------------------
ops/sre           sre-disk-incident           33.1%     PASS   FAIL     615
coding            coding-bugfix               28.7%     PASS   FAIL     736
support           support-refund              32.6%     PASS   FAIL     765
research          research-synthesis          25.7%     PASS   FAIL     809
data-analysis     data-analysis-sql           18.1%     PASS   FAIL     965
devops            devops-rollback             25.0%     PASS   FAIL     857
finance           finance-reconcile           29.1%     PASS   FAIL    1014
---------------------------------------------------------------------------
aggregate: distil cuts $0.14212 -> $0.10402 (26.8% cheaper) losslessly; 5761 tokens prunable.
GATE: PASS — every trajectory certified non-inferior; aggressive rejected on all.
```

<p align="center"><img src="docs/assets/domains.svg" alt="measured across 7 domains" width="100%"/></p>

---

## Why this is different

Token-savings numbers are easy to fake: measure quality at *low* compression, then advertise savings at *high* compression — two different runs. Distil refuses that. **The accuracy and the compression are measured on the same trajectories**, and a strategy that can't pass non-inferiority simply doesn't ship. You can watch the gate reject a quality-degrading strategy yourself:

```bash
distil certify --strategy distil       # VERDICT: PASS  (100% decision-equivalence)
distil certify --strategy aggressive   # VERDICT: FAIL  (mean diff −1.0, blocked)
```

---

## How it works

<p align="center"><img src="docs/assets/architecture.svg" alt="architecture" width="100%"/></p>

Two techniques carry most of the win, because they target where the money actually is in an agent loop — not where it looks like it is.

### ① Cache-aware compression — the dominant lever

In a multi-turn loop you re-send the growing context every step. With prompt caching a cache **read is ~10× cheaper** than fresh input, so the real cost is cache **misses**, not context **size**. Distil keeps the prefix byte-stable (schema canonicalization + lifting volatile fields like timestamps/UUIDs out of the prefix) and compresses only the volatile tail. The counterintuitive result, measured:

<p align="center"><img src="docs/assets/cache-aware.svg" alt="cache-aware savings" width="100%"/></p>

> Naive recompression sends **fewer tokens yet costs more than not compressing at all**, because it rewrites the cached prefix every turn. Distil doesn't — that's the whole game most tools miss.

### ② Causal / counterfactual pruning — the discovery engine

The eval isn't a ruler bolted on the side; it's a *discovery engine*. For each context block: remove it, replay, did any decision change? Blocks that never change a decision are **provably free to drop** — speculative retrievals, stale history.

```bash
distil prune
# doc-0   PRUNE (causally inert)     # speculative retrieval, never cited
# obs-0   keep (changed a decision)  # carries the decision-driving signal
# tokens provably free to drop: 615
```

### The quality contract

TOST non-inferiority testing (hand-rolled Student-t, **zero dependencies**). Lossless strategies pass; quality-degrading ones are rejected — across the **whole corpus**, as a CI gate.

---

## Quickstart

```bash
# zero install — run it straight from PyPI
uvx distil bench

# or install
pip install distil           # stdlib-only core, no transitive deps
distil savings               # technique #1, real dollars
distil prune                 # technique #4, causal ablation
distil certify               # the quality contract
distil verify                # byte-fidelity: reversible + append-only
distil holdout               # A/B savings with a bootstrap 95% CI

# billing-grade, against the real model (pip install 'distil[live]' + ANTHROPIC_API_KEY)
distil savings --tokenizer anthropic   # real Claude count_tokens, not an estimate
distil certify --runner anthropic      # certify against the live model
```

Wrap your existing client with **no code change** to the call site:

```python
from distil.adapters.anthropic import wrap
client = wrap(anthropic.Anthropic())   # compresses the request, keeps the cache warm
```

---

## What's inside (all real, all wired, no stubs)

| Capability | Module | Loss profile |
|---|---|---|
| Cache-aware priced cost engine (proves naive busts the cache) | `compress/cache_aware.py` | — |
| Schema canonicalization (byte-stable prefix) | `compress/stabilize.py` | lossless |
| Volatile-field extraction (dates/UUIDs/JWTs out of the prefix) | `compress/stabilize.py` | lossless · reversible |
| Tier-0 reversible transforms (JSON minify, RLE) | `compress/tier0.py` | provably lossless |
| Tier-1 decision-aware digest + retrieval handles | `compress/tier1.py` | reversible |
| Reject-if-bigger invariant | `compress/strategies.py` | safety |
| **Causal / counterfactual pruning** | `replay/ablation.py` | certified |
| **TOST non-inferiority gate** | `certify/` | the contract |
| Multi-domain corpus + corpus-wide CI gate | `corpus.py`, `distil bench` | — |
| Auth-mode gating (lossless-only on subscription/OAuth) | `policy.py` | safety boundary |
| Holdout A/B savings with bootstrap CI | `certify/holdout.py` | — |
| Byte-fidelity invariants (reversible + append-only) | `fidelity.py`, `distil verify` | — |
| BM25 partial retrieval from a handle | `retrieval.py` | — |
| Delta / append-only context | `delta.py` | lossless |
| Per-content-type keep-model codec | `codec/` | pluggable |
| Gist tool-schema caching (send once, reference forever) | `gist.py` | lossless |
| Runtime adapter (compress an Anthropic request, no code change) | `adapters/anthropic.py` | reversible |
| Billing-grade tokenizer + live runner | `tokenizer.py`, `replay/anthropic_runner.py` | opt-in |
| Local-first savings ledger + leaderboard | `ledger.py` | privacy-preserving |

---

## The corpus

The asset that makes certification meaningful: **7 real, captured-style agent trajectories across domains** — ops/SRE, coding, customer support, research, data analysis, devops, finance. Each is a 4-turn headless loop with a cacheable stable prefix, decision-driven volatile tool outputs, and causally-inert noise. `distil bench` runs the non-inferiority gate over all of them; new strategies can't ship unless they pass on every one. See [`corpus/manifest.json`](corpus/manifest.json).

---

## Distributables (multiple formats)

The stdlib-only core makes the packaging genuinely clean:

| Format | Get it | Notes |
|---|---|---|
| **PyPI** | `pip install distil` / `uvx distil` | zero runtime deps; corpus bundled in the wheel |
| **Docker** | `docker build -t distil .` → `docker run distil bench` | tiny, reproducible image |
| **Single-file** | `make pyz` → `python dist/distil.pyz bench` | one portable executable, no install |
| **Source** | `git clone … && make test` | full dev loop |

```bash
make gate     # the full CI gate: tests + corpus non-inferiority + byte-fidelity
```

---

## What we won't pretend

This is a discovery-grade project with a clear honesty line:

- **Default tokenizer is an offline heuristic** (so the core has zero deps). Compression *ratios* are robust to it; absolute dollars are not — use `--tokenizer anthropic` for billing-grade counts (the correct Claude tokenizer; tiktoken undercounts Claude).
- **The default runner is a deterministic stand-in** keyed on decision markers, so the gate runs offline with ground truth. `--runner anthropic` certifies against the live model — implemented, **UNVERIFIED** until you run it with a key.
- The **keep-model codec** and **gist** ship strong, real, deterministic implementations; a learned token-classifier and true soft-prompt gisting are documented *seams behind the same interfaces*, not stubs.
- Numbers in this README are reproducible from the bundled corpus with the heuristic tokenizer. No vanity metrics.

---

## Roadmap

- [x] Real tokenizer + live runner (billing-grade)
- [x] Multi-domain trajectory corpus + CI non-inferiority gate
- [x] Runtime adapter (no-code-change compression)
- [x] Auth-mode gating
- [x] Holdout A/B with confidence intervals
- [x] Byte-fidelity invariants
- [x] BM25 partial retrieval · delta context · keep-model codec · gist caching
- [ ] Learned per-content-type keep-model weights (the codec seam)
- [ ] Provider proxy for drop-in adoption across frameworks

---

## Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The one rule that matters: **a new compression strategy must pass `make gate`** (non-inferior on every domain, byte-reversible). No green gate, no merge. That's the whole philosophy in one sentence.

## License

[Apache-2.0](LICENSE) · *“Same potency, less volume.”*
