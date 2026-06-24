# Distil → a credible paper: protocol + outline

**Status:** planning doc. The core *idea* (decision-equivalence as a compression
objective, certified with conformal risk control) is novel and publishable. The
current repo's evidence is **circular** — the offline oracle (`DeterministicRunner`)
keys on planted `DECISION:` markers in a synthetic corpus, so "100% decision
equivalence" is a tautology, and the live path is `UNVERIFIED`. This plan specifies
exactly the empirical work needed to close that gap.

---

## 0. The one-line thesis

> You don't need byte- or embedding-fidelity to compress agent context — you need
> **decision-equivalence**, and you can give it a **distribution-free, finite-sample
> guarantee** (P(decision-change rate ≤ α) ≥ 1−δ) via conformal risk control,
> validated on real agent benchmarks against real baselines.

Two contributions to claim, separable so one failing doesn't sink the other:
1. **Conceptual/statistical:** decision-equivalence risk certification (LTT/CRC) for
   context compression. (Novelty headline.)
2. **Systems:** a cache-aware, reversible (digest-behind-handle + recover-on-demand)
   compression engine that operates inside the certified frontier. (Engineering.)

---

## 1. Claims vs. evidence gap (what must change)

| Current claim | What backs it now | What a reviewer needs | Action |
|---|---|---|---|
| "100% decision-equivalence" | Compressor preserved lines containing `DECISION:` | Real model's next action unchanged on non-self-labeling input | Replace oracle + corpus (§2, §3) |
| "Certified 83.2% @ 0%" | Synthetic decision-determined corpus, majority-of-3 | Real traces, real baselines, CIs over seeds | §4, §6 |
| Causal pruning "provably free to drop" | Ablation under the marker oracle | Ablation under a real grader; downstream task metric | §3, §5 |
| Conformal guarantee holds | Math is correct; loss is synthetic | Loss = real decision flip; exchangeability stressed under drift | §5, §7 |
| Cache-aware savings | Internal cost model | Real provider billing / token accounting on real traffic | §6 |

**Rule for the paper:** every headline number is produced by a *real model* deciding
on *input that does not contain the answer*, with baselines run identically.

---

## 2. Kill the circularity (the single most important change)

- **Remove the `DECISION:` marker oracle from the evaluation path.** It can stay as a
  unit-test fixture, never as an experimental grader.
- **Decision = the agent's actual next action**, extracted from real traces or
  produced by a real model via a forced structured tool call (`{action, target}` —
  the `AnthropicRunner` shape is fine), graded by **exact match on a canonical
  fingerprint** plus a semantic-equivalence fallback (LLM-judge, majority-of-k, with
  the judge's own agreement rate reported).
- **No instruction-following shortcuts.** The system prompt must NOT say "obey the
  line beginning DECISION DIRECTIVE." The decision must be *inferred* from context.

---

## 3. Datasets (real agent traffic, no planted answers)

Pick ≥3 across distinct decision types. Priority order:

1. **τ-bench / tau-bench** (airline, retail) — tool-using agents, ground-truth
   actions, multi-turn. Ideal: decisions are real tool calls.
2. **SWE-bench (Verified / Lite)** — coding agent trajectories; decision = next
   edit/command; downstream metric = resolved rate. Pairs well with the AST
   edit-equivalence already in the repo (`astdelta.py`, `shadow.py`).
3. **WebArena / VisualWebArena** or **AgentBench** — web/OS agents, long noisy
   observations (the regime compression actually helps).
4. **GAIA** or **AppWorld** — long-horizon tool use, optional 4th domain.
5. **Captured production logs** (if available) — the most convincing; ingest via the
   existing `distil ingest`. Anonymize.

For each: build trajectories where each turn carries the *accumulated* context an
agent really saw. Keep the held-out split for certification calibration vs. test
**disjoint by trajectory** (no turn leakage).

---

## 4. Baselines (run on the SAME corpus, graded identically)

- **No compression** (upper-bound on quality, reference for decision-change = 0).
- **Truncation / recency window** (structural floor).
- **LLMLingua-2** and **LLMLingua** (real packages — already wired in `benchmarks/`).
- **RECOMP** (abstractive + extractive) and/or **LongLLMLingua**.
- **Headroom** (`headroom-ai`, already wired) as a decision-aware comparator.
- **Distil** variants: lossless, stream (dedup), causal-prune, +reversible/expand.

Every method graded by the *same* runner + same token accounting. Report the method's
own best-fair invocation (as the repo already tries to do) — but on real data.

---

## 5. Metrics

Primary:
- **Decision-change rate** (1 − decision-equivalence) vs. uncompressed, with the
  conformal certificate's α, δ, and the *test-set* realized rate.
- **Downstream task success** (resolved rate / pass@1 / task reward) — this is what
  reviewers trust more than next-action match. Show compression doesn't move it
  beyond the margin.

Secondary:
- **Token savings** (and **$** via real billing token counts; the heuristic tokenizer
  is fine for ratios but use `--tokenizer anthropic` or provider usage for dollars).
- **Cache-aware effective cost** (read vs. miss) — the systems contribution.
- **Expand rate** (for reversible mode) — fraction of digests the model pulled back.
- **Latency / throughput** (already have `distil perf`).

Always with **variance**: ≥3–5 seeds; bootstrap CIs (already implemented in
`certify/holdout.py`). Report judge/grader agreement so equivalence isn't confounded
with grader noise.

---

## 6. Experimental design

- **E1 — Frontier:** savings vs. decision-change across the ladder, per domain, real
  grader. Reproduce the "certified frontier" figure but on real traces. Show distil
  inside the cliff, baselines past it.
- **E2 — Certification validity:** split each domain into calibration / test. Certify
  at chosen α on calibration; **report realized decision-change rate on the held-out
  test set** and verify it's ≤ α at the claimed confidence across many random splits
  (coverage plot: target α vs. empirical risk; LTT should be conservative, CRC tight).
  This is the figure that *proves the guarantee*, not just states it.
- **E3 — Distribution shift (the honest stress test):** calibrate on domain A, test on
  domain B / a later time window / a different agent. Show the bound degrades and
  recalibration restores it. This pre-empts the obvious reviewer attack and turns a
  weakness into a contribution.
- **E4 — Causal pruning ablation:** does dropping "causally inert" blocks (real
  grader) hold downstream task success? Compare to random/recency pruning at equal
  token budget.
- **E5 — Head-to-head vs. baselines** (§4) at matched decision-change budgets.
- **E6 — Systems:** cache-aware cost vs. naive recompression on real prefix dynamics;
  latency; reversible expand-rate behavior.

---

## 7. Statistical plan (mostly already sound — keep it honest)

- LTT with Hoeffding–Bentkus + fixed-sequence (implemented, correct). State the
  guarantee precisely: marginal over the calibration distribution, finite-sample.
- CRC for the monotone-loss expected-risk variant (implemented).
- **Exchangeability** is the load-bearing assumption — discuss it prominently, test it
  in E3. Do not claim a per-prompt guarantee.
- Multiple-comparison hygiene if you sweep many α/ladders; report all, not the best.
- Power/sample-size note: show the n needed to certify each α (the repo already
  observed 320 turns → α=2%, 640 → α=1%; formalize this as a curve).

---

## 8. Threats to validity (write this section explicitly — reviewers will look)

- **Grader as oracle:** real model graders are stochastic and can themselves err;
  mitigate with majority-of-k, report agreement, use exact-match where possible.
- **Synthetic leakage:** ensure no answer-revealing markers; audit the corpus.
- **Decision proxy vs. outcome:** next-action match ≠ task success; report both.
- **Distribution shift / non-exchangeability:** E3.
- **Domain coverage:** ≥3 domains; don't overclaim generality.
- **Cost model fidelity:** validate the cache-aware cost against real provider billing.

---

## 9. Paper outline

1. **Intro** — agents re-send growing context every turn; cost scales; existing
   compressors ship savings *estimates* with no correctness guarantee. Thesis (§0).
2. **Related work** — context compression (LLMLingua family, RECOMP, soft prompts),
   prompt caching, conformal prediction / risk control (LTT, CRC), RAG-conformal
   (the nearest neighbor — distinguish clearly). 
3. **Problem formulation** — decision-equivalence; loss = decision flip; the
   compression-frontier.
4. **Method** — (a) cache-aware reversible engine; (b) causal pruning as discovery;
   (c) conformal decision-equivalence certificate (LTT/CRC).
5. **Experiments** — E1–E6 (§6).
6. **Analysis** — when it works, when it breaks (drift), cost of the guarantee.
7. **Limitations** — §8, honestly.
8. **Conclusion.**

Reusable assets the repo already has: architecture/frontier figures, the statistics
code, the cost model, the baselines wiring, `ingest`. The work is **data + grading +
the validity experiments (E2/E3)**, not new machinery.

---

## 10. Suggested venues

- Strong fit: **EMNLP/ACL Findings or industry track**, **MLSys**, **NeurIPS/ICLR
  workshops** (efficient inference, agents, or distribution-free uncertainty).
- arXiv preprint first (the systems artifact is a real strength), then a workshop to
  get the conformal-for-decision-equivalence framing reviewed, then a main-track
  submission once E2/E3 hold on ≥3 real domains.

---

## 11. Minimal path to a defensible first submission

1. Wire `AnthropicRunner` (or an open-model runner via vLLM for cost) into the eval
   loop; verify it actually runs end-to-end with a key. (Removes the `UNVERIFIED`.)
2. One real dataset (τ-bench is fastest to adopt) with no planted markers.
3. E1 (frontier) + E2 (certification coverage on held-out) + 2 baselines.
4. If E2's realized risk ≤ α holds out-of-sample → you have the headline figure and a
   workshop paper. Add SWE-bench + E3 + full baselines for a main-track version.

The day the held-out decision-change rate comes in at or under the certified α on a
real model and real traces, the claim stops being marketing and becomes a result.
