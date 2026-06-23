# Changelog

All notable changes to Distil are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

## [0.12.1] — GA hardening

Pre-GA security + correctness pass (no behavior change to the happy path):

- **Request-path safety** (`httpguard.py`, applied across `proxy`, `aproxy`, `gateway`):
  upstream-path validation (blocks `@`/`//`/`..` host-injection SSRF), defensive
  `Content-Length` parsing, an 8 MiB body cap, and a bounded async connector.
- **Crash-resistance**: `compress_messages` and `ingest` no longer raise on
  malformed-but-valid JSON (missing/non-string `text`, non-dict messages, bad
  JSONL lines) — they pass such input through untouched; the compress call in
  every proxy is additionally guarded so compression can never break a request.
- **Gateway**: tenant labels are sanitized to a safe charset (no injection into
  accounting or the dashboard) and all HTML renderers (`gateway`, `telemetry`,
  `ledger`) escape interpolated values (stored-XSS fix).
- **Correctness**: `salience.protect()` now falls back to the byte-exact original
  (never the stripped block) so a salient line is never silently dropped, and uses
  exact line membership; `structured.fold` leaves null-bearing records byte-exact
  (no null-vs-missing ambiguity); the Rust hot-path pins JSON key order to match
  the Python backend.

## [0.12.0]

The Decision-Equivalence Risk Certificate (conformal risk control, `distil conformal`),
salience protection (model-free frontier shifter), and the live head-to-head vs. the
real LLMLingua-2 / Headroom packages. See `BENCHMARKS.md`.

## [0.9.0 – 0.11.0]

Recoverable compression (`distil_expand`), the self-improving learning flywheel
(`distil learn`), and the conformal certificate foundations.

## [0.2.0]

Both sides of the bill, the proof pack, and the leapfrog tracks.

### Added
- **Output compression** — gated generation-side verbosity shaping + lossless
  output-on-re-entry digest + an A/B harness (answer-preservation gate);
  `distil output-savings`, `distil proxy --shape-output`.
- **Certified compression frontier** — `eval.py`, `distil eval`: savings-vs-
  decision-equivalence curve where every point carries its certification verdict.
- **Self-distilling keep-model** — `online.py`, `distil online`: learns from
  causal labels from your own traffic, retrains, promotes only if non-inferior.
- **Verifiable federated telemetry** — `telemetry.py`,
  `distil federated-leaderboard`: HMAC-signed, content-free savings + verdict.
- **Async high-concurrency proxy** — `aproxy.py`, `distil proxy --async` (`[async]`).
- **Rust hot-path core** — `rust/distil-core` (PyO3), `distil/native.py` with a
  pure-Python parity fallback (transparent acceleration when built).
- **Managed gateway** — `gateway.py`, `distil gateway` with a live per-tenant dashboard.
- **Real-trace ingestion** — `ingest.py`, `distil ingest` (Anthropic + OpenAI shapes).
- **Performance benchmark** — `perf.py`, `distil perf` (p50/p95).
- **Transformer keep-model** — ONNX adapter + training pipeline (`distil train-transformer`);
  verified demo checkpoint on the release.
- OpenAI `role:"tool"` messages now get the decision-aware reversible digest.

## [0.1.0]

The first end-to-end cut: compression with a quality contract.

### Added
- **Cache-aware cost engine** (`compress/cache_aware.py`) — prices a multi-turn
  agent loop and proves naive recompression busts the prompt cache.
- **Risk-graded compression** — Tier-0 provably-lossless transforms, Tier-1
  reversible digest with retrieval handles, cache stabilization (schema
  canonicalization + volatile-field extraction), reject-if-bigger invariant.
- **Causal / counterfactual pruning** (`replay/ablation.py`) — discovers
  context that never changes a decision.
- **Quality contract** — TOST non-inferiority gate (`certify/`), decision-equivalence.
- **Multi-domain trajectory corpus** (7 domains) + `distil bench` CI gate.
- **Auth-mode gating** (`policy.py`) — lossless-only on subscription/OAuth.
- **Holdout A/B** (`certify/holdout.py`) — savings with a bootstrap 95% CI.
- **Byte-fidelity gate** (`fidelity.py`) — reversibility + append-only, `distil verify`.
- **Phase-7 building blocks** — BM25 partial retrieval (`retrieval.py`), delta /
  append-only context (`delta.py`), keep-model codec (`codec/`), gist tool-schema
  caching (`gist.py`).
- **Runtime adapter** (`adapters/anthropic.py`) — compress an Anthropic Messages
  request with no caller code change.
- **Billing-grade path** — Anthropic `count_tokens` tokenizer and live
  `AgentRunner` (opt-in `distil[live]`).
- **Distributables** — PyPI wheel/sdist, Docker image, single-file `distil.pyz`,
  CI + release workflows.
