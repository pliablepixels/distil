# ADR 0002 — Per-content-type keep policy for the Tier-1 digest

- Status: Accepted
- Date: 2026-07-11
- Deciders: distil maintainers (fork: pliablepixels)

## Context

The Tier-1 reversible digest (`distil/compress/tier1.py`) folds a verbose block's
droppable middle behind a recovery handle, keeping `head`/`tail` context plus a
**content-blind** salience net: lines carrying a `DECISION:` marker or matching
`error|exception|traceback|fail|warn|panic|fatal`. The module docstring already
anticipated the gap: *"in production this is where a learned per-content-type codec
or a salience model plugs in."*

Field observation: on an all-passing test log (~22,971 tokens from a `vitest run`,
1955 tests), the digest compressed roughly 90% but **dropped the run's verdict**.
The summary line `Tests 1955 passed (1955)` matched no keep rule (it contains
neither `DECISION:` nor a failure word), and `tail=1` kept only the final
`Duration` line, so the multi-line footer's decisive line was folded into a handle.
It also kept the incidental `ERROR [Crypto] ...` stdout the code-under-test logs on
purpose. For a "validate tests" task, the pass/fail summary is the entire
decision-relevant payload, so a big reduction that hides it is a quality failure,
even though the content is technically recoverable.

## Decision

Introduce `distil/compress/keep_policy.py`:

- `classify(text) -> ContentKind` where `ContentKind ∈ {GENERIC, LOG, TRACEBACK,
  DIFF}`, using cheap deterministic signals (runner/pass-fail markers → LOG,
  `Traceback`/stack frames → TRACEBACK, `diff --git`/`@@` → DIFF, else GENERIC).
- `must_keep(line, kind)` inherits the generic net for every kind and adds the
  per-kind load-bearing lines: LOG pins result-summary lines (`N passed/failed/
  skipped`, `Test Files`, `Duration`, `test result:`, exit codes) while leaving
  per-test lines droppable; TRACEBACK pins stack frames; DIFF pins file and hunk
  headers.

`digest()` classifies the block once and applies the policy. The dropped-run
marker becomes `<< +N lines omitted, handle=H >>`, with a by-category breakdown
`(E error, W warn, O other)` appended **only when flagged lines were folded**.
Because the keep policy already surfaces error/warn lines, a fold is usually
mundane and the marker stays bare (no wasted tokens); the breakdown appears only
if a future policy ever folds a flagged line.

The policy is model-free and deterministic, keeping the corpus gate offline and
reproducible. It is the "per-content-type codec" the tier1 docstring named; the
learned codec (`distil/codec/`) remains the future path and can consume these
content kinds.

## Consequences

- **Positive.** The digest no longer drops a test run's verdict (regression-tested
  in `tests/test_keep_policy.py`). Markers are self-describing. Reversibility and the
  recovery handle are unchanged: the marker is cosmetic and the restore map is still
  keyed by the SHA-256 of the byte-exact original. Extensible: a new content kind is a
  classify signal plus a keep rule.
- **Cost.** Keeping a few more lines and slightly longer markers reduces compression
  marginally on log/trace/diff blocks. Validated inside the corpus **non-inferiority
  gate** (`make gate` PASS, aggregate 24.2% cheaper, every trajectory certified) and
  the **byte-fidelity gate** (PASS, reversible + append-only).
- **No change for GENERIC blocks.** Prose and other unclassified content digest
  byte-for-byte as before.

## Alternatives considered

- **Wire the existing `salience.py` (`salient_lines`, surprise/high-entropy/
  cross-block anchors) into `digest()`** to unify the two keep mechanisms. Deferred:
  larger surface that touches the salience contract and its tests. The focused
  per-type policy fixes the measured failure with less risk; unifying with
  `salience.py` is a good follow-up.
- **Minimal fix: enlarge `tail` and add a pass/fail regex to the generic net.**
  Rejected: not extensible, and it conflates content types (a pass/fail regex applied
  to non-log blocks is noise).
- **A learned or LLM classifier.** Rejected for now: a model-free, deterministic
  policy keeps the gate offline and reproducible. The learned codec remains the path
  for salience beyond these heuristics.
