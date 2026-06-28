# GA-readiness

An honest, current ledger of what is production-grade and what still gates a "drop it on any
agent and trust it unattended" general-availability claim. Updated as items close. The bar:
distil should never *silently* ship a lossy operating point — when it cannot certify safety,
it falls back to full context.

## Status: GA-track — every *design* blocker is closed; what remains is empirical breadth

The headline GA risk surfaced by E11 (the safe operating point is capability-dependent, and a
hand-tuned constant can silently lose 31 pp on a stronger model) is **closed** by
auto-calibration with a fail-safe default. The next two largest items — **drift detection** and
the **single-grader** caveat — are now closed too (anytime-valid monitor + conservative grader
ensemble; see "Recently closed"). What remains is not missing machinery but **empirical breadth
that requires live runs/compute we have not spent**: validating the shipped speculative,
multi-grader, and RL-policy paths end-to-end, and broadening task-success beyond SWE-bench
Verified + τ-bench to more domains and models. The design is domain-agnostic; closing these is
*running it at scale*, not building more. We mark them honestly rather than claim them.

## Closed

| Item | Evidence |
|---|---|
| **Relevance gate is a shippable library primitive** (not benchmark-only) | `distil/gate.py` (`working_set_indices`, `gate_fraction`); contract verified against the benchmark proxy in `tests/test_gate.py` |
| **Operating point is auto-calibrated to agent capability** | `distil/calibrate.py` selects the most aggressive `gate_recent` still non-inferior to full; `distil calibrate` CLI; validated on real E11 data (selects gate@12, rejects gate@6) in `tests/test_calibrate.py` |
| **Fail-safe default (never silently lossy)** | If no operating point certifies non-inferior, calibration returns `fail_safe` → caller keeps full context (`test_fail_safe_when_nothing_certifies`, `test_real_e11_strict_margin_fails_safe`) |
| **Cross-model generality demonstrated** | E11: non-inferiority transfers to DeepSeek-V3 (different vendor, far stronger) at a capability-appropriate point |
| **Engineering maturity** | v0.25.x, 633 tests, full CI (ci/pages/paper-build/release), zero-dependency stdlib core, packaged (`distil` entrypoint) |
| **Per-turn + trajectory certificates, validated out-of-sample** | E2 (coverage 96.6–100%), E10 (trajectory, coverage 95.4/96.7%) |

## Open (tracked GA items)

| Item | Why it gates full GA | Mitigation today |
|---|---|---|
| **Validation breadth** | Task-success is validated on SWE-bench Verified coding agents (E8 n=500 Haiku; E11 n=200 single-seed DeepSeek-V3) and τ-bench. Broad multi-domain production traffic is not yet covered. | The certificate machinery is domain-agnostic; broadening is data, not redesign. |
| **Calibration data requirement** | Auto-calibration needs a small paired full-vs-candidate run on representative traffic before the gate can ship aggressively. | Fail-safe means the *absence* of calibration data degrades to full context (correct, not lossy) rather than to a guessed operating point. |

### Recently closed (were open)

| Item | How it closed |
|---|---|
| **Drift detection / auto-recalibration** | `distil/drift.py:DriftMonitor` — an *anytime-valid* sequential alarm (betting e-process for `H0: risk ≤ α`) that may be checked after every turn with false-alarm probability ≤ δ *no matter how often you peek* (Ville's inequality). Trips when live decision-change exceeds the certified budget → signal to recalibrate or fall back to full context. Validated: bounded false alarms under peeking + high detection power (`tests/test_drift.py`). |
| **Single grader family** | `distil/ensemble.py:EnsembleGrader` — grade with multiple model families, default **"any"-change** aggregation, which is conservative (can only *raise* measured risk), so the certificate stays valid even if one grader family is unfaithful. Aggregation logic shipped + tested (`tests/test_ensemble.py`); multi-family *validation* still needs a live multi-API run. |
| **Anytime-valid / tighter certificate** | `distil/conformal.py:betting_upper_bound` — the hedged-capital betting confidence sequence (Waudby-Smith & Ramdas, JRSSB 2023): variance-adaptive and valid simultaneously at every `t`. Coverage + anytime property Monte-Carlo–validated (`tests/test_conformal_bounds.py`). Honest tradeoff: for one-shot binary losses Bentkus is already near-optimal, so betting is *comparable* there; its edge is continuous monitoring and graded-loss adaptivity. |

## Cost frontier under the motto (advanced techniques)

"Best in class" holds on the motto's axis (certified decision-equivalence + task success), not
on raw cost — an uncertified lossy method can always be cheaper because it is allowed to change
decisions. Within the certified envelope, these techniques cut cost without spending the
certificate. Status is honest about shipped-and-validated vs. framework vs. research.

| # | Technique | Status | Where |
|---|---|---|---|
| 1 | **Cache-monotone gate** — deterministic, append-only digests so the digested prefix is byte-stable and prompt-cache/KV reuse captures it | **Shipped + tested** | `distil/gate.py:monotone_gate`; `tests/test_cost_frontier.py` |
| 2 | **Graded gate** — per-distance compression tiers (crush the far periphery harder), certified with the tighter empirical-Bernstein bound | **Shipped + tested** | `distil/gate.py:graded_gate`; `distil/conformal.py:tight_risk_bound` |
| 3 | **Tighter conformal (empirical-Bernstein)** — certifies more savings at the same confidence on *graded* losses; coverage-validated by Monte-Carlo | **Shipped + coverage-tested** | `distil/conformal.py:empirical_bernstein_bound`; `tests/test_conformal_bounds.py` |
| 4 | **Speculative expansion** — pay for full context only when a certified divergence trigger fires; controller + certified miss-rate | **Framework shipped + tested; needs a live calibration run for end-to-end savings** | `distil/speculative.py` |
| 5 | **Constrained-bandit operating-point search** — online successive-elimination under the NI constraint, fail-safe | **Shipped + tested**; full constrained-RL keep-policy is **research** (needs training data) | `distil/calibrate.py:bandit_select_operating_point` |

Honest cost caveat baked into the design and tests: on content that is *already fully
cacheable*, caching alone can be cheaper than any compression (compressing rewrites cached
bytes as fresh). The cache-monotone gate's win is over a cache-*hostile* gate; the gate's
primary payoff stays accuracy (E8/E11). The certificate's tightening (#3) and the
graded/speculative/bandit machinery cut cost *inside* the certified envelope — they never
trade the guarantee for dollars.

## How to calibrate before shipping the gate

```bash
# 1. Run your agent on a small calibration set under full context and 2–3 candidate
#    working-set sizes (gate_recent), producing swebench-style score JSONs.
# 2. Let distil pick the most aggressive safe operating point (fail-safe to full):
distil calibrate \
  --baseline scores/full.json \
  --candidate gate@6=scores/gate6.json:6 \
  --candidate gate@12=scores/gate12.json:12 \
  --margin 0.05 \
  --json calibration_certificate.json
# 3. Deploy with DISTIL_E7_GATE_RECENT set to the selected value (or keep full context if
#    the certificate is fail-safe).
```
