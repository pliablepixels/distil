# Publishing this, step by step (first-timer guide)

You have the idea, the code, and a paper draft (`docs/paper/main.tex`). Here is the
concrete path from here to a published paper, written for someone doing it for the
first time. Two tracks run in parallel: **(A) finish the experiments**, **(B) get it
out** (arXiv → workshop → conference).

---

## 0. The one thing that gates everything

A reviewer's first question is *"are the numbers real?"* Right now the apparatus is
real and validated; the **headline numbers are not yet run at scale**. Before
submitting anywhere, complete the headline run:

```bash
python benchmarks/fetch_real.py tau --src tau:sonnet-35-airline --out tau.json
python benchmarks/prove.py --dataset tau --path tau.json \
   --runner anthropic --model claude-opus-4-8 --samples 3 \
   --alpha 0.05 --delta 0.05 --ladder full --reps 500 --report no_expand.json
python benchmarks/prove.py --dataset tau --path tau.json \
   --runner anthropic --model claude-opus-4-8 --samples 3 --expand \
   --alpha 0.05 --delta 0.05 --ladder full --reps 500 --report with_expand.json
```
Run a second domain (`tau:sonnet-35-retail`) for E3, then paste E1/E2/E4 into the
paper. Budget: this is the main cost (API tokens + a few hours). Everything else is
writing.

---

## Track A — finish the experiments (what reviewers will check)

1. **Headline run** (above): E1 frontier, E2 out-of-sample coverage, E4 task success,
   no-expand vs with-expand. This is the result.
2. **A second domain** for E3 (distribution shift) — calibrate on one, test on the
   other.
3. **Baselines on the same corpus**: LLMLingua-2, RECOMP, raw truncation — graded
   identically (the harness already supports external strategies; wire them as ladder
   levels). Reviewers want a comparison, not just your method.
4. **Report variance**: seeds + the bootstrap CIs the harness emits; publish the
   grader's model↔gold agreement.

A clean E2 coverage plot (realized risk ≤ α out-of-sample) + a baseline table is
enough for a strong workshop paper. Add E3 + a second task (SWE-bench) for a
main-conference version.

---

## Track B — getting it out

### Step 1 — arXiv preprint (do this first; it's not peer review)
- **What it is:** a public, citable preprint. Establishes priority (timestamp) and
  lets you share a link. Not refereed.
- **Account + endorsement:** create an account at [arxiv.org](https://arxiv.org).
  First-time submitters to a category (here **cs.LG**, cross-list **cs.CL**) may need
  an *endorsement* from an existing arXiv author — ask a colleague/advisor who has
  posted in cs.LG, or arXiv may auto-endorse based on affiliation email.
- **Format:** upload the LaTeX **source** (`main.tex`), not a PDF — arXiv compiles it.
  Because every figure is TikZ, there are no image files to wrangle.
- **License:** CC BY 4.0 is a good default for wide reuse.
- **Timing:** post when the headline numbers are in. You can post v1 and update.

### Step 2 — pick a venue (workshop first, for a first paper)
- **Workshops** (recommended first target): lighter review, fast, friendly, great for
  a focused result. Look for workshops at **NeurIPS / ICLR / ICML** on *efficient
  inference / efficient ML*, *LLM agents*, or *distribution-free uncertainty /
  conformal prediction*; also **ENLSP** (Efficient NLP) and ACL/EMNLP workshops.
  Acceptance is more forgiving and you get real feedback.
- **Findings / industry tracks**: **EMNLP/ACL Findings** and **industry tracks** suit
  a strong applied result that isn't aiming for a main-track spotlight.
- **Main conferences** (higher bar, ~8 pages, months-long cycle): **MLSys** (systems
  + ML — excellent fit for the cache-aware engine), **NeurIPS/ICLR/ICML**, **EMNLP/
  ACL**. Aim here once E1–E4 + baselines + a second task are solid.

### Step 3 — submission mechanics (what surprises first-timers)
- **OpenReview** is the platform for most ML venues (NeurIPS/ICLR/workshops). You
  create a profile; submissions and reviews happen there.
- **Anonymization (double-blind):** main ML venues require an **anonymous** PDF — no
  author names, no obvious self-links. *But you may still post to arXiv beforehand*;
  most venues allow a non-anonymous preprint to exist (check the specific CFP).
  Remove the `\author{}` content and the GitHub link for the submitted version.
- **Page limits & style file:** each venue ships a `.sty`; swap it into `main.tex`
  (see `docs/paper/README.md`). Respect the page limit (refs/appendix usually exempt).
- **Reproducibility checklist & code:** attach the repo (or an anonymized zip). You're
  in good shape — the harness is the artifact.
- **Deadlines:** ML venues have hard abstract + full-paper deadlines (often a week
  apart). Find the CFP, note both, submit a placeholder abstract early.

### Step 4 — after submission
- **Rebuttal:** you get reviews and a short window to respond — answer concretely,
  run the one extra experiment they ask for if you can.
- **Camera-ready:** if accepted, de-anonymize, add acknowledgments, finalize.

---

## A realistic first-timer timeline

| When | Do |
|---|---|
| Week 1–2 | Headline run + second domain + 2 baselines; fill the paper's numbers |
| Week 2 | Post arXiv v1 (cs.LG, cross-list cs.CL) |
| Week 2–4 | Tighten writing; pick the next workshop deadline |
| Deadline | Submit to a workshop (OpenReview, anonymized) |
| +1–2 mo | Reviews → rebuttal → (likely) accept → camera-ready |
| Later | Extend (E3 + SWE-bench + more baselines) → main-track/MLSys |

---

## Honesty checklist (protects you in review)

- [ ] Numbers from a real model on real traces, no answer-revealing markers.
- [ ] Majority-vote grading; model↔gold agreement reported.
- [ ] Reversible tier reported **both** no-expand and with-expand.
- [ ] Baselines run on the *same* corpus with the *same* grader.
- [ ] Exchangeability stated; drift shown (E3); guarantee described as marginal.
- [ ] Code + commands released (they are).

You already did the hard part — the rigorous apparatus and the honest framing. The
rest is running the headline numbers and following the venue's mechanics.
