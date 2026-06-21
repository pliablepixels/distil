# Changelog

All notable changes to Distil are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

## [0.1.0] — unreleased

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
