# GA-readiness

An honest, current ledger of what is production-grade and what still gates a "drop it on any
agent and trust it unattended" general-availability claim. Updated as items close. The bar:
distil should never *silently* ship a lossy operating point — when it cannot certify safety,
it falls back to full context.

## Status: GA-track — the capability-dependent operating-point blocker is closed

The headline GA risk surfaced by E11 (the safe operating point is capability-dependent, and a
hand-tuned constant can silently lose 31 pp on a stronger model) is **closed** by
auto-calibration with a fail-safe default. Remaining items are scoped and tracked below.

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
| **Drift detection / auto-recalibration** | The certificate and the calibrated operating point are valid under *exchangeability* with the calibration distribution; changing model, agent, or task mix requires re-running `distil calibrate`. Today that is manual. | Documented exchangeability caveat; `distil calibrate` is cheap to re-run; shadow-mode (`distil shadow-stats`) surfaces live divergence. GA needs automatic drift alarms that trigger recalibration. |
| **Validation breadth** | Task-success is validated on SWE-bench Verified coding agents (E8 n=500 Haiku; E11 n=200 single-seed DeepSeek-V3) and τ-bench. Broad multi-domain production traffic is not yet covered. | The certificate machinery is domain-agnostic; broadening is data, not redesign. |
| **Single grader family** | Decision-equivalence numbers use one grader model family per task; cross-family ensemble grading is future work. | Reported honestly in-paper; faithfulness diagnostic gates the proxy. |
| **Calibration data requirement** | Auto-calibration needs a small paired full-vs-candidate run on representative traffic before the gate can ship aggressively. | Fail-safe means the *absence* of calibration data degrades to full context (correct, not lossy) rather than to a guessed operating point. |

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
