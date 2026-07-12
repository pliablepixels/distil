# Spec: Query-Aware Salience

**Status:** implemented (phase 1, deterministic) — shipped in 1.15.0; phase 2 (learned scorer) pending
**Author:** distil core (assisted)
**Date:** 2026-07-12
**Depends on:** `distil/adapters/anthropic.py` (live compress path), `distil/compress/tier1.py` (keep decision), PR #23 content-type keep policy (complementary)

## Context — why

distil's keep rules are **content-agnostic**. They pin lines by intrinsic shape — `DECISION:` markers, error/warn words, and (as of 1.14.x) command *verdict* lines. That closes the "known-format result line" case. It does **not** close the general one an independent review named: *output where one arbitrary line is the whole answer* — a single `grep` hit buried in context, a specific commit SHA in `git log`, one key's value in a config dump. No intrinsic-shape heuristic knows which line matters, because **which line matters depends on what the agent asked, not on the line itself.**

Every competitor has the same blind spot from the other side: rtk is a format filter (knows test/build/`git`, passes prose through), headroom routes by content type, grepai is a vector finder over the repo. **None of them sees the agent's live query paired with the tool output at compression time.** distil does — it is a proxy sitting in the request. The agent's intent is *right there* in the same request as the output being compressed: the `tool_use` call that produced this `tool_result`, and the latest user turn. That is a structural advantage no post-hoc filter has.

**Goal:** use the intent distil already holds to keep the output lines the agent actually needs — turning "content-agnostic" into "content-and-intent-aware" — without touching reversibility or the decision-equivalence certificate.

## The insight

When the agent runs a tool, its **arguments are a precise, free, deterministic statement of intent**:

- `grep MAX_RETRIES config/` → the answer is the line containing `MAX_RETRIES`.
- `git log --oneline --grep=deadlock` → the answer is a commit subject about `deadlock`.
- `cat pkg.json | jq .version` → the answer contains `version`.

The tool_use `input` (and `name`) name the needle. Keep the output lines that match the needle. This is deterministic, model-free, and only a proxy can do it.

## Design

A **query-relevance keep layer**, strictly **additive** on top of the existing keep union (DECISION / error / verdict / dedup, plus PR #23's content-type policy). It only ever keeps *more* lines; it never removes an existing keep.

### 1. Extract intent (once per request)
In `compress_messages` (`anthropic.py:307`) — the one function that holds both the whole `messages` list and each block — derive an `Intent` before compressing blocks:

- **tool_use terms:** for each `tool_result`, find the preceding `tool_use` (same `tool_use_id`); collect salient tokens from its `name` + `input` values — identifiers, quoted strings, paths, flags' operands. This is the primary, highest-precision signal.
- **user terms:** salient tokens from the latest `user` text turn (identifiers, back-ticked spans, quoted strings, paths, error codes, numbers). Secondary.
- Normalize to a lowercased term set; drop stopwords and terms shorter than 3 chars.

Reuse the existing thread-local plumbing that already carries `keep=` into the block compressor (`_keep_tls`, `anthropic.py:337`) — thread `Intent` the same way rather than changing every signature.

### 2. Keep query-relevant lines
Thread `intent` into `_compress_tool_result_text → _tier1_digest → digest(text, ..., intent=…)` and OR a relevance predicate into the keep decision (`tier1._must_keep`):

```
def _query_relevant(line, terms) -> bool:
    # a line is relevant if it contains a salient intent term on an identifier boundary
```

Additive: `_must_keep` returns True if the line matches DECISION / error / verdict **or** `_query_relevant`. Everything else (head/tail, dedup, folding, handle, RestoreStore write) is unchanged.

### 3. Selectivity guard (don't let intent keep everything)
A term that matches most lines isn't discriminating (e.g. the agent grepped a token that appears everywhere). If an intent term matches more than `SELECTIVITY_CAP` (e.g. 40%) of lines, drop that term from the keep set for this block — it's noise, not a needle. This preserves compression on the case where intent doesn't narrow anything.

## Invariants (must hold — verified, not assumed)

Per the architecture trace, recovery is **structural**: whenever a block is digested, `RestoreStore` holds the *entire* original byte-exact, and dropped lines fold under a `<< +N lines, handle=… >>` marker recovered via `distil_expand`. Therefore:

1. **Additive-only.** The layer may only widen the keep set. It must never force a drop past `verbatim` / `_active_keep` / the handle-collision fallback (`anthropic.py:180,187,195`).
2. **Store-write untouched.** Do not add a path that folds lines without the `restore[handle]=full_text` write. Keep the `+N` count honest.
3. **Certificate is the acceptance test.** Changing which lines are inline can change the model's next action, so re-run `certify` (`certify/gate.py:47`) / shadow on the new rule. Because the layer is additive (keeps a superset), decision-equivalence can only stay equal or improve — but it is measured, not asserted.

## Reuse map (don't reinvent)

- `distil/compress/salience.py` — existing "protected-line union" abstraction; `_query_relevant` joins the same union its anomaly regex feeds.
- `distil/compress/guideline.py` — outcome-guided learning flywheel; the same "widen keeps, never narrow, then re-certify" posture. A learned relevance scorer can later ride this.
- `distil/codec/keep_model.py` — `KeepModel` Protocol (`score(line, kind)`); the phase-2 learned relevance scorer plugs in here (add `intent`), model-agnostic.
- `distil/gate.py` — recency working-set ("relevance = recent"); conceptually this feature extends "keep recent" to "keep recent **or** query-relevant".

## Rollout

- **Phase 1 (this spec):** deterministic lexical intent-match on the live Anthropic path. Model-free, additive, behind a re-cert. Ship dark → shadow-measure → enable.
- **Phase 2:** learned relevance scorer via the `KeepModel` Protocol, trained on the expand flywheel (every `distil_expand` call is a label that "this folded line was actually needed"). Same additive contract.

## Verification

- **Unit** (`tests/test_query_aware.py`): grep-hit (agent greps `X`; output has `X` on line 500 amid neutral noise → line kept, middle still folds); git-SHA / config-value analogues; **selectivity guard** (term matching everything → skipped, compression preserved); **additive property** (keep set with intent ⊇ keep set without).
- **Certificate:** `distil bench` stays `GATE: PASS` (additive cannot lower equivalence); add a query-retention check (an intent line survives digestion) alongside the verdict-retention check.
- **Live:** shadow A/B decision-equivalence unchanged-or-up; expand rate should *fall* (fewer round-trips because the needed line is already inline) — a measurable win.

## Why this makes distil win

The verdict fix made distil competitive on *known* formats. Query-aware salience makes it the **only** compressor that keeps the right line in *arbitrary* output — because it's the only one holding the query and the output together. It converts the residual weakness into the differentiator: "keep what the agent is actually looking for."
