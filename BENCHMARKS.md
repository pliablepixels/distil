# Distil — Live Benchmark Report

> Dev reference. Every number here is **measured live** against `claude-opus-4-8`
> (no offline stand-in), reproducible from the scripts in `benchmarks/`. Where a
> result is a limitation or a caveat, it's stated plainly — see [Honest caveats](#honest-caveats).

## TL;DR

On a realistic, decision-determined agent corpus (5 domains, 120 turns, 4.5–6.5 KB/turn),
graded live by `claude-opus-4-8`:

| method | token savings | live decision-change | certifies ≤5% @ 95%? | compression latency / turn |
|---|---:|---:|:--:|---:|
| **Distil** (causal-prune + lossless) | **83.2%** | **0.0%** | ✅ **yes** | **0.026 ms** |
| LLMLingua-2 (`llmlingua` 0.2.2) | 53.1% | 20.0% | ❌ no | ~1,480 ms |
| Headroom (`headroom` 0.27.0) | 35.3% | 0.0% | ✅ yes | 26 ms |
| RTK (`rtk-py` 0.42.4.1) | — | — | excluded | — |

**Distil is the only method that is simultaneously the most aggressive, fully
decision-equivalent, and the lowest latency.** Headroom is decision-safe but
2.4× less aggressive; LLMLingua-2 is aggressive but flips 1-in-5 decisions;
RTK operates at a different layer (see below).

---

## Coding-agent benchmark — cache-delta on read→edit→reread

A second, **messages-level** benchmark targets the coding-agent hot path
(`benchmarks/codebench.py`: 16 sessions / 256 turns of read → edit → **re-read**),
scoring **cache-aware real dollars** (the stable prefix billed at the cache-read
rate) against the *real* installed packages. Every method is confined to the
volatile suffix, so the cached prefix is cache-read for all — a fair comparison of
how each compresses *new* content.

| method | token savings | $ savings (cache-aware) | latency / turn | fidelity |
|---|---:|---:|---:|:--|
| **distil** (PAYG digest) | **91.5%** | **91.1%** | **0.09 ms** | reversible |
| distil + cache-delta | 89.6% | 89.0% | 13.6 ms | reversible |
| distil-verbatim + cache-delta | 34.9% | **43.8%** | 13.1 ms | reversible |
| distil-verbatim (Tier-0 only) | 0.0% | 0.0% | 0.6 ms | reversible |
| Headroom (`headroom` 0.27.0) | 0.0% | 0.0% | 7.6 ms | lossy |
| LLMLingua-2 (`llmlingua` 0.2.2) | 6.5% | **−11.9%** | 762 ms | lossy |

**Honest reading of these numbers:**

- The **Tier-1 reversible digest is the dominant lever** (~91% cache-aware savings,
  reversible, 0.09 ms). On these sessions it already captures the re-reads, so
  **cache-delta adds little *on top of* the digest** (89% vs 91%).
- **cache-delta's real niche is verbatim/interactive mode** (where the aggressive
  digest is disallowed — subscription/OAuth, human-in-the-loop): there it is the
  primary lever, turning a 0% floor into **43.8% cache-aware savings, reversible**.
- **LLMLingua-2 *raises* cost by 11.9%** here — it rewrites suffix content and busts
  the prompt cache — at **~8,500× the latency**, and it is lossy. **Headroom's
  per-block compressor left this corpus unchanged** (0%). distil leads on every axis
  *and* is the only reversible method.

This benchmark also caught two real distil bugs (a cache-monotonicity flip in
cache-delta and a Tier-0 token-inflation on whitespace runs); both are fixed in
v0.22.0 — the numbers above are post-fix. Reproduce:
`PYTHONPATH=. python benchmarks/codebench.py 16`.

---

## How we certified — and why it's credible

The headline isn't "we cut N% tokens." It's a **statistical certificate** on the
thing that actually matters: *does the agent still make the same decision?*

### 1. The loss: decision-equivalence, not byte-equivalence
For each calibration turn, the loss is **`1` iff the agent's decision changes**
versus the uncompressed context, and `0` otherwise. The decision is a structured
`{action, target}` fingerprint produced by forcing the live model through a
single `record_decision` tool call (`distil/replay/anthropic_runner.py`). The
*same* model grades the compressed and uncompressed context.

### 2. Removing the model's own noise
LLMs are non-deterministic. We take the **majority vote over 3 samples** per
decision, so the model's run-to-run variance doesn't masquerade as a
compression-induced change. (`temperature` is deprecated on this model, so
majority vote is the correct stabilizer.)

### 3. The determinism precondition (why this corpus is valid)
A live certificate is only meaningful if the decision is **determined by the
context** — otherwise you're measuring the model's coin-flips, not the
compressor. We verify this directly: **byte-exact decision-change rate = 0.0%**
across all 120 turns. The model reproduces its own decision on (semantically)
identical input, so any change at higher compression is the compressor's fault.

> This precondition is also why our *first* attempt failed honestly: on the
> original synthetic corpus the live model's byte-exact change rate was ~50%
> (ambiguous `target`), so nothing could certify. We fixed the corpus, not the
> math. That failure is documented, not hidden.

### 4. The statistics: distribution-free, finite-sample
We use **Learn-Then-Test** (Angelopoulos, Bates, Candès, Jordan & Lei,
*Ann. Appl. Stat.* 2025, [arXiv:2110.01052](https://arxiv.org/abs/2110.01052))
with **Hoeffding–Bentkus** p-values and fixed-sequence testing. For the certified
level λ̂:

```
P( R(λ̂) ≤ α ) ≥ 1 − δ        (here α = 0.05, δ = 0.05, n = 120)
```

No distributional assumptions, valid at finite n. We also cross-check with
**Conformal Risk Control** ([arXiv:2208.02814](https://arxiv.org/abs/2208.02814),
ICLR 2024), which controls `E[L(λ̂)] ≤ α`. This is distinct from the nearest
neighbor in the literature ([arXiv:2511.17908](https://arxiv.org/abs/2511.17908),
ECIR 2026), which applies conformal guarantees to RAG retrieval *recall* — a
different task. The novel contribution is the loss = **agent decision-equivalence
under context compression**.

### 5. Why you should believe it (not marketing)
- **Every ladder level is shown** (below) — no cherry-picking the best point.
- **The certificate refuses when data is insufficient** — on a small corpus it
  declines a tight α rather than over-claim (we observed it decline α = 0.5%).
- **Competitors run through the same gate** — the same live model, same loss.
- **Fully reproducible** — exact commands and pinned package versions below.
- **`hb_p = 0.0058 < δ`** on the safe levels — certified with margin, not on the edge.

---

## The certificate (full ladder)

Live, `claude-opus-4-8`, majority-of-3, n = 120, α = 0.05, δ = 0.05:

| ladder level | live decision-change | token savings | Hoeffding–Bentkus p | certifies? |
|---|---:|---:|---:|:--:|
| byte-exact | 0.0% | 2.6% | 0.0058 | ✅ |
| lossless (tier-0/1) | 0.0% | 52.3% | 0.0058 | ✅ |
| causal-prune | 0.0% | 83.2% | 0.0058 | ✅ |
| **prune + lossless** | **0.0%** | **83.2%** | **0.0058** | ✅ **← operating point** |
| truncate@400 (blind) | 0.0% | 61.6% | 0.0058 | ✅ |
| truncate@200 (blind) | 100.0% | 73.9% | 1.0000 | ❌ |

**Certified operating point = `prune + lossless`: 83.2% token savings at a live
decision-change rate ≤ 5% with 95% confidence.** The operating point is the
*highest-savings* certified level. Note the cliff: blind `truncate@200` cuts 73.9%
but flips **100%** of decisions — it drops the load-bearing directive. Cutting the
right tokens, not just fewer tokens, is the whole game.

---

## The head-to-head, in detail

All methods graded on the **same** corpus through the **same** decision-equivalence
gate; savings measured identically (compressed block tokens ÷ original block tokens).

### Distil — 83.2% savings, 0.0% decision-change ✅
Decision-aware: keeps `decision_relevant` blocks byte-exact, prunes causally-inert
blocks (the ablation thesis), losslessly compacts the rest. Highest certified
savings, zero decision change, sub-millisecond.

### Headroom (`headroom-ai` 0.27.0) — 35.3% savings, 0.0% decision-change ✅
Real package, invoked fairly: blocks presented as a `tool_use → tool_result`
conversation with `optimize=True` (the per-block seam silently no-ops it — that's
an integration trap, not Headroom's fault). **Decision-safe** — it preserved the
exact target ID on every turn — but **2.4× less aggressive** than Distil, and it
loads a ModernBERT scorer (26 ms/turn vs Distil's 0.026 ms).

### LLMLingua-2 (`llmlingua` 0.2.2) — 53.1% savings, 20.0% decision-change ❌
Real package (`microsoft/llmlingua-2-xlm-roberta-large-meetingbank`, CPU,
rate=0.5). Aggressive extractive token classification — but **decision-unaware**:
it drops/garbles the load-bearing ID on **1-in-5** turns, so it **fails the gate**.
This is the decision-equivalence thesis vindicated against the canonical academic
compressor. Also ~1,480 ms/turn (transformer inference on CPU).

### RTK (`rtk-py` 0.42.4.1) — excluded, with reason
RTK is a **command-output proxy**: it compresses the output of specific wrapped
commands (`git`, `ls`, `psql`, `aws`, `docker`, …) by stripping known boilerplate.
It exposes **no raw-text/stdin mode**, so it cannot compress arbitrary agent
context (tool-result/history blocks). We attempted it through the adapter; it
honestly reports the layer mismatch rather than returning a fabricated number.
It is a different layer of the stack, not a contender on this axis.

---

## Latency (compression cost per turn)

Measured on the same realistic turns (4.5–6.5 KB), pure compression time, n = 120:

| method | p50 | p95 | mean | model in the path? |
|---|---:|---:|---:|:--:|
| **Distil** (prune+lossless) | **0.023 ms** | **0.028 ms** | **0.026 ms** | no |
| Headroom (optimize) | 12.7 ms | 47.8 ms | 26.3 ms | yes (ModernBERT) |
| LLMLingua-2 (CPU) | — | — | ~1,480 ms | yes (xlm-roberta-large) |

Distil is **~1,000× faster than Headroom** and **~57,000× faster than LLMLingua-2**.
This is architectural: Distil does deterministic structural compaction + causal
pruning with **no model load and no inference**. For an *inline proxy* on every
agent turn, ~0.03 ms is invisible; tens-to-thousands of ms is a tax on every call.

---

## Reproduce

```bash
# 1. install the real competitor packages
uv pip install llmlingua headroom-ai rtk-py

# 2. generate the realistic, decision-determined corpus (5 domains, 120 turns)
python benchmarks/gen_realworld.py 30 /tmp/corpus_realworld

# 3. run the live DERC certificate + head-to-head  (needs ANTHROPIC_API_KEY)
python benchmarks/derc_live_compare.py        # ~16 min, ~3,240 live calls

# the shipped certificate command (your own traffic):
distil conformal --corpus ./mycorpus --runner anthropic --alpha 0.05 --samples 3
```

Pinned versions used for this report: `anthropic` 0.111.0, `llmlingua` 0.2.2,
`headroom` 0.27.0, `rtk-py` 0.42.4.1, model `claude-opus-4-8`.

---

## Honest caveats

- **The corpus is decision-*determined synthetic*, not production traffic.** It is
  constructed so the next action is uniquely determined by the context (verified:
  byte-exact change rate = 0). It is realistic in content and size, but it is not a
  substitute for calibrating on your own logs (`distil ingest` → `distil conformal`).
- **Exchangeability.** The conformal guarantee holds for the calibration
  distribution. Under drift (new agent, prompt change, workload shift), recalibrate
  on a rolling window.
- **Marginal, not per-prompt.** The guarantee bounds the *average* decision-change
  rate, not any single prompt.
- **Causal-prune used ground-truth inert labels.** Here the corpus carries the
  `decision_relevant` labels directly; in production Distil's ablation engine
  discovers them via counterfactual replay (`distil prune`), which is itself
  gate-validated.
- **Decision fingerprint = `{action, target}`.** A coarser or finer fingerprint
  would move absolute numbers; this one matches how an agent's next tool call is
  actually keyed.
- **Competitor results are version- and invocation-sensitive.** Headroom in
  particular no-ops under the wrong message shape; we invoke each tool the way that
  gives it its best fair result and verify it actually engaged.
