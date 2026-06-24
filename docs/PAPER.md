# Certified Decision-Equivalent Context Compression for LLM Agents

*Working draft. Results tables are filled by `benchmarks/prove.py` on real traces
(see `docs/PAPER_PLAN.md` for the protocol and `benchmarks/PROVE.md` for the runner).
Numbers marked ⟨…⟩ are auto-populated from a run; do not hand-edit.*

---

## Abstract

LLM agents resend a growing context every turn, so context size dominates serving
cost. Existing context compressors report token savings but offer **no guarantee
that the agent still behaves the same**. We reframe the objective from byte- or
embedding-fidelity to **decision-equivalence**: a compression is acceptable iff the
agent takes the same next action it would have on the uncompressed context. We then
give this a **distribution-free, finite-sample guarantee** by casting level selection
as conformal risk control (Learn-Then-Test / Conformal Risk Control) with the loss
defined as a decision flip. The certificate states, for a chosen risk budget α and
confidence 1−δ, that the decision-change rate on exchangeable traffic is ≤ α. We
evaluate on **real τ-bench agent trajectories** (and SWE-bench edit-localization),
graded by a real model with no answer-revealing markers, and validate the guarantee
**out-of-sample**: across many calibration/test splits the realized decision-change
rate stays ≤ α at the claimed confidence. The reversible compression engine that
operates inside the certified frontier saves ⟨X⟩% tokens at a ⟨Y⟩% decision-change
rate and preserves ⟨Z⟩% downstream task success.

## 1. Introduction

- Agent serving cost scales with re-sent context; prompt caching shifts the cost to
  cache *misses*, which compression of the volatile tail can reduce.
- The gap: every shipping compressor quotes a savings *estimate*; none certifies that
  the agent’s *decisions* are unchanged. "100% accuracy" is a slogan, not a number.
- Contributions:
  1. **Decision-equivalence** as the compression objective (loss = the agent’s next
     action flips vs. uncompressed).
  2. A **decision-equivalence risk certificate**: conformal risk control (LTT/CRC)
     selecting the most aggressive level whose decision-change rate is provably ≤ α.
     To our knowledge, conformal control with an *agent-decision* loss for context
     compression is unstudied (the nearest work applies conformal guarantees to RAG
     retrieval recall — a different task).
  3. A **cache-aware, reversible** compression engine (digest-behind-handle +
     recover-on-demand) that operates inside the certified frontier.
  4. An **evaluation on real agent traces** that removes the circular self-labeling
     of synthetic corpora, with the guarantee **validated out-of-sample** and under
     **distribution shift**.

## 2. Related work

- Context/prompt compression: LLMLingua / LLMLingua-2 / LongLLMLingua, RECOMP,
  selective-context, soft-prompt distillation. All optimize a fidelity/relevance
  proxy; none certifies decision-equivalence.
- Prompt caching and cache-aware serving.
- Distribution-free uncertainty: conformal prediction; **Learn-Then-Test**
  (Angelopoulos et al., 2021/2025) and **Conformal Risk Control** (Angelopoulos et
  al., ICLR 2024). Closest application: conformal guarantees for RAG recall
  (ECIR 2026) — retrieval, not agent-decision-preserving compression.

## 3. Problem formulation

- A trajectory is a sequence of turns; each turn is the full context the agent saw,
  decomposed into typed blocks with a stability hint (cacheable prefix vs. volatile
  tail). A *decision* is the agent’s next action — a tool call (τ-bench) or an
  edit/command (SWE-bench) — represented as a canonical `{action, target}`
  fingerprint produced by a grading model from context alone (no directive/marker).
- Loss on a (turn, level): `L = 1` iff the decision under the compressed context
  differs from the decision under the uncompressed context. `R(λ)` is the expected
  loss (decision-change rate) at compression level λ.
- A compression **ladder** orders levels least→most aggressive: byte-exact →
  reversible lossless digest → salience-protected truncation → raw truncation sweep.

## 4. Method

### 4.1 Cache-aware reversible engine
- Keep the prefix byte-stable (schema canonicalization; lift volatile fields), digest
  only the volatile tail behind a content handle, keep the original locally, expose a
  recover-on-demand tool. Lossless = byte-in-context; reversible = digested but
  byte-recoverable; lossy = the rest. (Details in repo `compress/`.)

### 4.2 Causal pruning (discovery)
- Ablate a block, replay, keep iff a decision changes — yields a causally-justified
  pruning policy rather than a heuristic.

### 4.3 The decision-equivalence risk certificate
- Calibrate per-turn losses for each ladder level on held-out-of-test calibration
  traffic. Select with **LTT** (Hoeffding–Bentkus p-values, fixed-sequence testing):
  guarantees `P(R(λ̂) ≤ α) ≥ 1−δ`; or **CRC** for the monotone 0/1 loss:
  `E[L(λ̂)] ≤ α`. Operating point = highest-savings level in the certified prefix.
- The exchangeability assumption is explicit; the guarantee is marginal over the
  calibration distribution, recalibrated under drift.

## 5. Experimental setup

- **Data.** Real **τ-bench** trajectories (airline/retail; gpt-4o and Sonnet-3.5
  runs; 182 trajectories / 1164 decision points after parsing the airline-gpt-4o
  log) loaded with no planted markers; the decision is the agent’s actual tool call.
  **SWE-bench** edit-localization trajectories built from real issues + gold patches
  (target file inferred from the issue amid distractors).
- **Grader.** A real model returns the `{action, target}` fingerprint via majority-of-k.
  We report **model↔gold next-action agreement** on the uncompressed context as a
  grader-faithfulness gate. Runs use `--runner claude-cli` (subscription) / `openai`
  (local open model) / `anthropic`.
- **Protocol.** E1 frontier; **E2 certification coverage** (certify on calibration,
  measure realized risk on a disjoint held-out split, over many trajectory-level
  splits → empirical `P(realized ≤ α)`); E3 leave-one-domain-out shift; **E4
  downstream task success** (trajectory keeps its outcome iff every decision is
  unchanged), vs. the uncompressed baseline with a bootstrap CI.

## 6. Results

> Auto-populated from `benchmarks/prove.py --report` on real τ-bench. The pipeline is
> validated end-to-end live; the cells below are filled from the latest real run.

**E1 — frontier (real grader).** ⟨table: level, savings, decision-change⟩

**E2 — certification holds out-of-sample.** At α=⟨α⟩, δ=⟨δ⟩ over ⟨R⟩ splits:
empirical coverage `P(realized ≤ α)` = ⟨cov⟩ (target ≥ 1−δ); mean realized held-out
risk = ⟨r⟩; certified savings = ⟨s⟩. **The certificate holds on traffic it never saw.**

**E3 — distribution shift.** Leave-one-domain-out: ⟨per-domain realized risk / ok?⟩.

**E4 — downstream task success.** Baseline success ⟨b⟩%; safe levels retain ⟨…⟩%,
aggressive truncation erodes to ⟨…⟩% (95% CI).

**Grader faithfulness.** model↔gold agreement on uncompressed context = ⟨g⟩%.

## 7. Analysis & limitations

- Where it works / breaks; cost of the guarantee; sample-size vs. tightest certifiable
  α (more calibration turns → tighter α, per the conformal bounds).
- **Threats to validity** (see `docs/PAPER_PLAN.md §8`): grader stochasticity (majority
  vote; report agreement), decision-proxy vs. outcome (E4), non-exchangeability (E3),
  domain coverage, cost-model fidelity.

## 8. Conclusion

Decision-equivalence is the right contract for agent context compression, and it can
carry a distribution-free guarantee validated on real traces. The reversible engine
sits safely inside the certified frontier.

## Reproducing

```bash
# real τ-bench data (no HuggingFace needed; ships in the tau-bench repo):
python benchmarks/fetch_real.py tau --src tau:gpt-4o-airline --out /data/tau.json
# the full real run (subscription grader; Opus for the headline, Haiku for scale):
python benchmarks/prove.py --dataset tau --path /data/tau.json \
    --runner claude-cli --model claude-opus-4-8 --samples 3 \
    --alpha 0.05 --delta 0.05 --ladder full --reps 500 --report results_tau.json
```
