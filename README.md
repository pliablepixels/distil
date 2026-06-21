<p align="center">
  <img src="docs/assets/banner.svg" alt="Distil — compression with a quality contract" width="100%"/>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-8b7bff" alt="license"/></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-5ad1c9" alt="python"/>
  <img src="https://img.shields.io/badge/runtime%20deps-0-5ad19a" alt="zero deps"/>
  <img src="https://img.shields.io/badge/tests-202%20passing-5ad19a" alt="tests"/>
  <img src="https://img.shields.io/badge/corpus%20gate-PASS-5ad19a" alt="gate"/>
  <img src="https://img.shields.io/badge/works%20with-any%20SDK-8b7bff" alt="any sdk"/>
</p>

<h3 align="center">Cut LLM agent costs ~30% — and <em>prove</em> the agent still makes the same decisions.</h3>

<p align="center">
Most context compressors ship a token-savings <em>estimate</em>.<br/>
<strong>Distil ships a quality contract:</strong> a strategy compresses only as far as a statistical non-inferiority test certifies the agent behaves identically — across 7 domains, as a CI gate.
</p>

<p align="center">
  <a href="#-60-second-start">Quickstart</a> ·
  <a href="#-works-with-every-sdk">Integrations</a> ·
  <a href="#-install-your-way">Install</a> ·
  <a href="https://dshakes.github.io/distil/getting-started.html"><b>Full Docs →</b></a>
</p>

---

## 🧭 Pick your lens

<table>
<tr>
<td width="33%" valign="top">

**👔 For decision-makers**

Agents re-send their whole context every turn — you pay for it every turn. Distil cuts that **~30% with zero quality loss**, and *proves* it: the savings and the accuracy are measured on the **same runs**, gated in CI. No "trust us."

</td>
<td width="33%" valign="top">

**🛠️ For developers**

`pip install distil-llm`, point your client's `base_url` at the proxy, done — **no code change, any language or SDK**. Or `wrap(client)` in-process. Lossless by default, reversible on demand.

</td>
<td width="33%" valign="top">

**🔬 For researchers**

Compression reframed as **decision-equivalence** and certified with **TOST non-inferiority** + bootstrap CIs over a multi-domain trajectory corpus. Causal ablation discovers what's safe to drop. Reproducible, zero-dep.

</td>
</tr>
</table>

---

## 💡 The one idea

**You don't need byte-equivalence, you need decision-equivalence.** Byte-lossless compression and high savings are information-theoretically in tension. But an agent only has to take the *same actions* and produce the *same outputs* whether or not its context was compressed. That's measurable and certifiable — so **"100% accuracy" becomes a statistical guarantee on outcomes, not a diff of strings.** Everything here makes that real and measured.

---

## ⚡ 60-second start

```bash
uvx distil bench          # certify savings + quality across 7 domains, in seconds
```

```
domain            trajectory                $ saved   distil   aggr  pruned
---------------------------------------------------------------------------
ops/sre           sre-disk-incident           33.1%     PASS   FAIL     615
coding            coding-bugfix               28.7%     PASS   FAIL     736
support           support-refund              32.6%     PASS   FAIL     765
research          research-synthesis          25.7%     PASS   FAIL     809
data-analysis     data-analysis-sql           18.1%     PASS   FAIL     965
devops            devops-rollback             25.0%     PASS   FAIL     857
finance           finance-reconcile           29.1%     PASS   FAIL    1014
---------------------------------------------------------------------------
aggregate: distil cuts $0.14212 -> $0.10402 (26.8% cheaper) losslessly; 5761 tokens prunable.
GATE: PASS — every trajectory certified non-inferior; aggressive rejected on all.
```

<p align="center"><img src="docs/assets/domains.svg" alt="measured across 7 domains" width="100%"/></p>

> **Why trust the number?** Token-savings numbers are easy to fake — measure quality at *low* compression, advertise savings at *high* compression. Distil refuses that: accuracy and compression are measured on the **same** trajectories, and a strategy that can't pass non-inferiority doesn't ship.
> ```
> distil certify --strategy distil       # VERDICT: PASS  (100% decision-equivalence)
> distil certify --strategy aggressive   # VERDICT: FAIL  (mean diff −1.0, blocked)
> ```

---

## 🔌 Works with every SDK

One proxy. Point any `base_url`-honoring client at it — **Python, TypeScript, any language** — and get cache-aware lossless compression with **no code change**.

<p align="center"><img src="docs/assets/cross-sdk.svg" alt="one proxy, every SDK" width="100%"/></p>

```bash
distil proxy --upstream https://api.anthropic.com   # localhost:8788
```

| SDK / framework | Change | Example |
|---|---|---|
| Anthropic SDK (Py/TS) | `base_url="http://127.0.0.1:8788"` | [`examples/python_anthropic.py`](examples/python_anthropic.py) |
| OpenAI SDK | `base_url="http://127.0.0.1:8788/v1"` | [`examples/python_openai.py`](examples/python_openai.py) |
| Vercel AI SDK | `createAnthropic({ baseURL: '…:8788' })` | [`examples/js_vercel_ai_sdk.ts`](examples/js_vercel_ai_sdk.ts) |
| LangChain (py/js) | `anthropicApiUrl` / base URL | [`examples/js_langchain.ts`](examples/js_langchain.ts) |
| LiteLLM | `api_base="http://127.0.0.1:8788"` | [`examples/python_litellm.py`](examples/python_litellm.py) |

Prefer in-process? Wrap the client directly — still no call-site change:

```python
from distil.adapters.anthropic import wrap
client = wrap(anthropic.Anthropic())   # compresses the request, keeps the cache warm
```

---

## 📦 Install your way

<p align="center"><img src="docs/assets/install.svg" alt="install options" width="100%"/></p>

| Format | Command |
|---|---|
| **Zero install** | `uvx distil bench` |
| **PyPI** | `pip install distil-llm` → `distil bench` |
| **Homebrew** | `brew install dshakes/tap/distil` |
| **Docker** | `docker build -t distil . && docker run distil bench` |
| **Single file** | `make pyz` → `python dist/distil.pyz bench` |
| **Node launcher** | `npx @distil/proxy --upstream https://api.anthropic.com` |

> The import package and CLI are `distil`; the PyPI distribution is `distil-llm` (the bare name was taken). The `npx` path is a thin launcher around the Python proxy — the real cross-language story is pointing your SDK's `base_url` at it.

---

## 🧠 How it works

<p align="center"><img src="docs/assets/architecture.svg" alt="architecture" width="100%"/></p>

Two techniques carry most of the win — they target where the money actually is in an agent loop, not where it looks like it is.

### ① Cache-aware compression — the dominant lever

You re-send the growing context every step. With prompt caching a cache **read is ~10× cheaper** than fresh input, so the real cost is cache **misses**, not context **size**. Distil keeps the prefix byte-stable (schema canonicalization + lifting volatile fields like timestamps/UUIDs out of the prefix) and compresses only the volatile tail.

<p align="center"><img src="docs/assets/cache-aware.svg" alt="cache-aware savings" width="100%"/></p>

> Naive recompression sends **fewer tokens yet costs more than not compressing at all**, because it rewrites the cached prefix every turn. Distil doesn't — that's the whole game most tools miss.

### ② Causal / counterfactual pruning — the discovery engine

The eval isn't a ruler bolted on the side; it's a *discovery engine*. Remove a context block, replay, did any decision change? Blocks that never change a decision are **provably free to drop**.

```bash
distil prune
# doc-0   PRUNE (causally inert)     # speculative retrieval, never cited
# obs-0   keep (changed a decision)  # carries the decision-driving signal
```

---

## 🧩 What's inside (all real, all wired, no stubs)

| Capability | Module | Loss profile |
|---|---|---|
| Cache-aware priced cost engine | `compress/cache_aware.py` | — |
| Schema canonicalization + volatile-field extraction | `compress/stabilize.py` | lossless · reversible |
| Tier-0 reversible transforms · Tier-1 decision-aware digest | `compress/tier0.py`, `tier1.py` | lossless / reversible |
| **Causal / counterfactual pruning** | `replay/ablation.py` | certified |
| **TOST non-inferiority gate** + 7-domain corpus + `distil bench` | `certify/`, `corpus.py` | the contract |
| **Provider proxy** — drop-in across SDKs | `proxy.py`, `distil proxy` | reversible |
| In-process adapter (`wrap`) | `adapters/anthropic.py` | reversible |
| **Learned keep-model** (logistic, 96.4% acc / 0.98 F1 held-out) | `codec/learned.py` | pluggable |
| Auth-mode gating (lossless-only on subscription/OAuth) | `policy.py` | safety |
| Holdout A/B savings + bootstrap CI | `certify/holdout.py` | — |
| Byte-fidelity invariants (reversible + append-only) | `fidelity.py`, `distil verify` | — |
| BM25 partial retrieval · delta context · gist caching | `retrieval.py`, `delta.py`, `gist.py` | lossless |
| Billing-grade tokenizer + live runner | `tokenizer.py`, `replay/anthropic_runner.py` | opt-in |
| Savings ledger + leaderboard (privacy-preserving) | `ledger.py` | local-first |

**Full docs:** [Getting started](https://dshakes.github.io/distil/getting-started.html) · [Concepts](https://dshakes.github.io/distil/concepts.html) · [Techniques](https://dshakes.github.io/distil/techniques.html) · [CLI](https://dshakes.github.io/distil/cli.html) · [Architecture](https://dshakes.github.io/distil/architecture.html) · [Integrations](https://dshakes.github.io/distil/integrations.html) · [Deploy & security](https://dshakes.github.io/distil/deploy-security.html) · [FAQ](https://dshakes.github.io/distil/faq.html)

---

## 🔒 Security & deployment

- **Localhost-only by default** — the proxy binds `127.0.0.1` and forwards only to the single configured upstream (no SSRF).
- **No secret/body logging** — request bodies and credentials are never logged.
- **Auth-mode gating** — `--lossless-only` keeps subscription/OAuth sessions lossless and never injects tools (provider-ToS-safe).
- **Stateless** — nothing is persisted; ZDR-compatible.

See [Deploy & security](https://dshakes.github.io/distil/deploy-security.html) for topologies (local sidecar, container sidecar, shared gateway) and the threat model.

---

## ✅ What we won't pretend

- **Default tokenizer is an offline heuristic** (zero deps); ratios are robust, dollars are approximate. Use `--tokenizer anthropic` for billing-grade counts (the correct Claude tokenizer — tiktoken undercounts Claude).
- **The default runner is a deterministic stand-in** so the gate runs offline with ground truth. `--runner anthropic` certifies against the live model — implemented, **UNVERIFIED** until you run it with a key.
- The learned keep-model is a real trained **logistic** classifier (96.4%/0.98 on held-out lines); a heavier transformer classifier is a documented upgrade behind the same interface.
- Numbers here are reproducible from the bundled corpus with the heuristic tokenizer. No vanity metrics.

---

## 🗺️ Roadmap

- [x] Cache-aware compression · causal pruning · TOST quality gate
- [x] Multi-domain corpus + CI non-inferiority gate
- [x] Real tokenizer + live runner (billing-grade)
- [x] Runtime adapter + **provider proxy** (drop-in across SDKs)
- [x] Auth-mode gating · holdout A/B · byte-fidelity invariants
- [x] **Learned per-content-type keep-model** (trained weights)
- [x] BM25 partial retrieval · delta context · gist caching
- [ ] Transformer keep-model weights (the codec's heavier seam)
- [ ] Managed gateway + per-tenant dashboards

---

## 🤝 Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The one rule that matters: **a new compression strategy must pass `make gate`** (non-inferior on every domain, byte-reversible). No green gate, no merge. That's the whole philosophy in one sentence.

## License

[Apache-2.0](LICENSE) · *“Same potency, less volume.”*
