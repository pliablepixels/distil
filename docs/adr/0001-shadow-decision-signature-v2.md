# ADR 0001 — Shadow decision-signature v2: formatting normalization + algorithm versioning

- Status: Accepted
- Date: 2026-07-08
- Deciders: distil maintainers

## Context

Shadow mode measures compression's effect on agent behavior by comparing a
**content-free decision signature** — `decision_signature` in `distil/shadow.py`
— of the agent's chosen next action, on the compressed request (A/B) versus the
original, and on the original replayed against itself (A/A, the nondeterminism
noise floor). The noise-adjusted decision-equivalence feeds the status-line
health verdict and the trajectory certificate's "decision-equivalence" gate
(`distil certify`, `benchmark.py`).

Field observation (2026-07-08): a live ledger showed **A/A self-agreement of only
72.7%** — identical, uncompressed requests replayed were scored "decision changed"
27% of the time. Root cause: the v1 signature hashes the tool name **plus the full
tool input verbatim**, normalizing only Python-code-shaped strings via AST.
Everything else — `bash` commands, file paths, search queries, natural-language
arguments — is hashed byte-for-byte. LLM wording jitter (`ls -la` vs `ls  -la`,
pretty vs minified JSON, trailing newlines) run-to-run then reads as a *different
decision*. This inflates **both** A/A and A/B change rates, making the equivalence
verdict untrustworthy in both directions and producing false red alarms.

Compounding: the ledger had no per-row algorithm version, so `load()` blended rows
from older code/experiments into a live verdict, and the verdict rendered on very
small samples (n≈36 A/B, n≈11 A/A).

## Decision

1. **Signature v2 — formatting-only normalization.** Canonicalize whitespace on
   every string argument (`_canon_ws`: strip, collapse internal whitespace runs)
   before hashing, and apply it as the fallback when a code string fails Python-AST
   parsing. This removes *formatting* jitter without merging genuinely different
   tokens — different content still hashes differently. We deliberately do **not**
   go coarser (e.g. tool-name-only): that would mask real decision changes and
   weaken the certificate's safety claim.

2. **Algorithm versioning.** Introduce `SIG_VERSION` (now `2`). Every ledger row is
   stamped with `sig` (and `v`, the build). `ShadowLedger.load(current_only=True)`
   counts only rows matching the current `SIG_VERSION`. **v1 and v2 signatures are
   never compared.** Verdicts and certificates issued under v1 remain valid under
   v1; new evidence accumulates fresh under v2. No on-disk row is rewritten.

3. **Robust verdict gate.** The status line / `shadow-stats` render a ✓/⚠/✗ verdict
   only once evidence is robust: `VERDICT_MIN_AB = 50` A/B and `VERDICT_MIN_AA = 30`
   A/A samples; otherwise a neutral `de baseline N/30` / `de N/50` warming state.
   The alarm thresholds themselves (99% / 95% on the noise-adjusted rate) are
   unchanged — the sample gate, not a looser threshold, stops the cry-wolf, so the
   alarm's sensitivity to genuine degradation is preserved.

## Acceptance gate (measured during soak, not offline)

v2 provably removes *formatting* noise (unit tests: whitespace variants collide,
different commands do not). Whether the 72.7% A/A floor was *predominantly*
formatting jitter versus genuine token-level model nondeterminism can only be known
from **live A/A replay under v2**. The rule, evaluated once ≥30 v2 A/A samples
accrue in real traffic:

- **Keep v2 as the equivalence lever** if A/A self-agreement rises materially
  (target ≥90%) AND A/B equivalence does *not* collapse to ~100% (which would mean
  v2 is too coarse and is masking real changes).
- **Escalate** if A/A barely moves: the residual is genuine model nondeterminism,
  not formatting — the lever is then the verdict threshold / sampling, and v2 stays
  only for its (real) formatting wins.

This ADR must be revisited with the measured number before the metric is cited as
evidence that compression is or isn't degrading decisions.

## Consequences

- Existing on-disk ledgers become v1 evidence; live verdicts start warming afresh
  under v2 until 50/30 samples accrue. Expected and correct.
- The certificate's decision-equivalence gate now depends on v2; any tooling that
  compares signatures across versions must key on `sig`.
- Formatting-invariance could, in principle, treat a purely-reindented script as
  the "same decision" — an intended, content-preserving trade.
