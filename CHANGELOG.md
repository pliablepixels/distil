# Changelog

All notable changes to Distil are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

## [0.18.0] — Streaming-aware shadow mode (Claude Code / Codex / Gemini)

- **Shadow-mode now works on streaming sessions.** Real agent sessions (Claude Code,
  Codex, the Gemini CLI) stream their responses over SSE, which the previous shadow
  comparison couldn't parse — so it silently recorded nothing. `shadow.py` now
  reconstructs the decision from a streamed body: `decision_signature_from_body`
  reads a non-streaming JSON body directly and rebuilds a streamed (SSE or
  chunk-array) one via `_decision_from_chunks`, accumulating the first tool call
  across chunks for all three providers (Anthropic `input_json_delta`, OpenAI
  `tool_calls` argument deltas, Gemini `functionCall`). A streamed response yields
  the same signature as its non-streamed equivalent, so comparisons are valid.
- The proxy shadow path now compares raw bodies via `decision_signature_from_body`,
  so `distil proxy --shadow` measures live decision-equivalence on streaming traffic.
  Verified end-to-end on an SSE tool-call response.

## [0.17.0] — Decouple compression aggression from auth (`--verbatim`)

Resolves an overload introduced in 0.16.0. `--lossless-only` had been redefined to
mean "Tier-0 only," which **contradicted `policy.py`** (where the reversible digest
*is* the lossless strategy that subscription sessions use) and silently de-tuned
autonomous agents on subscription/OAuth from ~70%+ down to ~10%.

- **`--lossless-only` restored** to its policy meaning: lossless *strategies* only
  (no lossy output-shaping) + no tool injection. The reversible, certificate-backed
  Tier-1 digest **still runs** — consistent with `policy.py` and the project's
  definition of "lossless" (reversible + decision-equivalent).
- **New `--verbatim` flag** (proxy / `wrap` / gateway): skips the Tier-1 digest
  entirely (Tier-0 only) so the model sees content un-stubbed. The right mode for
  interactive (human-in-the-loop) sessions or out-of-distribution traffic. Lower
  savings, byte-in-context fidelity.
- Adapter/integration kwargs renamed to match: `compress_messages(..., verbatim=)`,
  `compress_generate_request(..., verbatim=)`; LiteLLM `distil_verbatim`; LangChain
  `compress_messages(..., verbatim=)`. Docs reconciled across CLI / adapters /
  integrations / faq / deploy-security.

## [0.16.0] — Ecosystem hooks: MCP server + LiteLLM/LangChain

- **MCP server** (`mcp_server.py`, `distil mcp`): a zero-dependency, stdlib-only
  Model Context Protocol server over stdio JSON-RPC 2.0. Exposes `distil_compress`
  (reversible digest + handle, original kept in a local on-disk store),
  `distil_expand` (recover by handle), and `distil_savings`. Wire it into any MCP
  client (Claude Desktop, IDEs, agents). The message handler is a pure function and
  is unit-tested without real stdio; the loop is verified end-to-end.
- **In-process framework hooks** (`integrations/`): LiteLLM (`compress`/`completion`/
  `acompletion`) and LangChain (`compress_messages`, duck-typed over message objects
  *and* dicts) compress requests before they leave the process — same reversible
  compression as the proxy, no sidecar required. Both lazy-import their framework, so
  distil stays zero-runtime-deps.

## [0.15.0] — Claude Code plugin + status line

- **`distil statusline`** (new CLI command): renders a compact one-line savings
  summary from the local ledger (tokens, dollars, runs, and live decision-
  equivalence when shadow-mode has samples). Reads the optional Claude Code status-
  line JSON on stdin for the model name; never raises.
- **Claude Code plugin** (`plugins/distil/` + `.claude-plugin/marketplace.json`):
  installable via `/plugin marketplace add dshakes/distil`. Ships a `/distil`
  command (savings report + setup help) and a `statusline.sh` that calls
  `distil statusline`. Honest scope: a plugin cannot reroute a running session or
  set the main status line from its manifest, so the README documents the one-line
  `settings.json` addition; traffic is compressed via `distil wrap` / `distil proxy`.

## [0.14.0] — Google Gemini adapter + true lossless-only

- **Gemini adapter** (`adapters/gemini.py`): the proxy, async proxy, and gateway now
  compress Google's `generateContent` request shape (`contents` / `parts` /
  `functionResponse`) — a third first-class provider alongside Anthropic and the
  OpenAI-compatible family. `text` parts get Tier-0 lossless transforms; large
  `functionResponse` string values get the Tier-1 *reversible* digest (recoverable
  via the local store); `functionCall`, `inlineData`, `fileData`, and model-authored
  text pass through untouched. Path-detected (`:generateContent` /
  `:streamGenerateContent`), so just `--upstream https://generativelanguage.googleapis.com`.
  Shadow-mode live decision-equivalence works for Gemini too. (Expand-tool injection,
  output shaping, and Gemini context caching remain messages-format-only for now.)
- **`--lossless-only` is now genuinely lossless-in-context** (GA correctness fix). It
  previously still applied the Tier-1 digest, replacing tool output the model could not
  recover (tool injection is disallowed on subscription/OAuth) with a stub — despite
  the "safe for subscription" label. It now applies only Tier-0 transforms in this
  mode, so the model sees semantically identical content. The aggressive,
  certificate-backed reversible digest remains the default (PAYG) behavior.

## [0.13.0] — Shadow-mode live decision-equivalence

- **Shadow mode** (`shadow.py`, `distil proxy --shadow RATE`, `distil shadow-stats`):
  samples a fraction of live requests, runs each one **both compressed and
  uncompressed** in a background thread (never blocking the client), and records a
  **content-free live decision-change rate** on real traffic. The continuous online
  counterpart to the offline certificate — decision-equivalence becomes observable
  in production. Decision = the agent's next `tool_use`/`tool_call`; equivalence
  iff that action matches.
- README: a "See it working — real-time savings & live equivalence" section
  (per-request headers, gateway dashboard, genuine-savings ledger, shadow mode,
  and one-env-var org-wide enforcement).

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
