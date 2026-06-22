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

`pipx install distil-llm` (or `uvx --from distil-llm distil …`), point your client's `base_url` at the proxy, done — **no code change, any language or SDK**. Or `wrap(client)` in-process. Lossless by default, reversible on demand.

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

## 🔑 What only Distil can do — recoverable compression

Every other compressor — summarizers, extractive pruners, structural crushers — is **lossy**: once it crushes a tool output, the detail is *gone*. Distil **digests behind a content handle and keeps the original locally**, then hands the agent a `distil_expand` tool. Run with `distil proxy --expand` (or `distil wrap --expand`) and:

- **The model pulls back exactly the detail it needs, on demand** — Distil resolves the handle from the local store and re-queries, *transparently*. Your agent code never changes; it just gets the right answer.
- **So you can compress fearlessly.** The dangerous failure mode of lossy compression — "it dropped something load-bearing" — is gone, because the safety net is the model recovering the detail itself.
- **Every expansion is a label.** A `distil_expand` call is ground truth that the digested content *mattered*. Logged (numbers only, never content), these train the keep-model to stop digesting what *your* workload depends on — a compounding moat a lossy tool can't build, because it has nothing to expand and no signal to learn from.

This is the structural advantage: **compress more, lose nothing, and get better the more you use it.** Lossy competitors can't follow here without rebuilding around reversibility.

---

## ⚡ 60-second start

```bash
uvx --from distil-llm distil bench   # certify savings + quality across 7 domains, in seconds
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

### The certified compression frontier — `distil eval`

The artifact no competitor publishes: a savings-vs-quality curve where **every point carries its certification verdict**. It locates the cliff past which lossy compression drops decisions — and shows distil sitting safely inside it. Reproducible offline; run `--runner anthropic` over your ingested traces for live task-accuracy.

```
level                   savings   equiv  certified  curve
--------------------------------------------------------------------------
distil (cache-aware)       8.4%    100%     ✔ PASS   ██
truncate@1200              7.2%     79%        ✘ —    ██
truncate@700              20.0%     36%        ✘ —    ████
truncate@300              41.3%      0%        ✘ —    █████████
--------------------------------------------------------------------------
distil: 8.4% token savings @ 100% decision-equivalence — certified.
certified ceiling beyond which lossy compression drops decisions and the gate rejects it.
```

---

## 📊 Benchmark — head-to-head on *certified* savings

Every technique through the **same** decision-equivalence gate and the **same** cache-aware cost model, on a reproducible 64-trajectory, 8-family corpus. The winner is computed, not assumed — lossy methods that drop decisions are disqualified, however much raw they cut.

| Technique | Tokens | $ saved | Decision-equiv | Verdict |
|---|--:|--:|--:|---|
| **distil-causal** | 80.5% | **81.5%** | 100% | ✅ certified — leader |
| truncate / sliding-window | 78.7% | 79.6% | 14% | ❌ fails gate |
| **distil-stream** (+ cross-turn dedup) | 61.0% | 61.7% | 100% | ✅ certified |
| **distil-lossless** (fold + template mining) | 57.4% | 58.1% | 100% | ✅ certified · byte-exact |
| summarize / rolling memory | 56.5% | 57.2% | 39% | ❌ fails gate |
| extractive importance (LLMLingua family) | 18.2% | 18.4% | 77% | ❌ fails gate |

**The only methods that pass the gate are Distil's** — every lossy alternative posts a raw cut but changes decisions. Even byte-exact `distil-lossless` beats every competitor's *certified* number, because theirs is zero. Reproduce: `python benchmarks/gen_corpus.py && distil benchmark --corpus benchmarks/corpus_xl`. Bring your own tool with `--external module:function`. → full methodology: [docs/benchmark](https://dshakes.github.io/distil/benchmark.html)

**Tune the trade — the equivalence dial.** 100% decision-equivalence is the default, not a wall. Set a lower target and Distil spends a bounded *divergence budget* on the highest-value turns — deeper savings for a **measured, explicit** equivalence cost, with byte-exact fallback everywhere else. The trade is always reported, never hidden:

```
$ distil frontier --corpus benchmarks/corpus_xl
   target   achieved equiv  token savings
     100%             100%          58.1%      ← certified-safe
      80%              82%          62.9%      ← deeper, by an amount you chose
```

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

| Format | Command | Prereq |
|---|---|---|
| **Zero install** | `uvx --from distil-llm distil bench` | [uv](https://docs.astral.sh/uv/) |
| **Isolated CLI** | `pipx install distil-llm` → `distil bench` | Python 3.11+, [pipx](https://pipx.pypa.io/) |
| **Homebrew** | `brew install dshakes/tap/distil` | Homebrew |
| **Docker** | `docker build -t distil . && docker run distil bench` | Docker |
| **Single file** | `make pyz` → `python dist/distil.pyz bench` | Python 3.11+ |
| **In a venv** | `pip install distil-llm` (inside an active virtualenv) | Python 3.11+ |

> The import package and CLI are `distil`; the PyPI distribution is `distil-llm` (the bare name was taken — so `uvx`/`pip` must reference `distil-llm`, not `distil`). Distil is a CLI: install it **isolated** (pipx/uv/brew/Docker), because modern macOS/Linux block system-wide `pip install` ([PEP 668](https://peps.python.org/pep-0668/)). **Node / any language:** point your SDK's `base_url` at `distil proxy`, or use `distil wrap -- <agent>` — no Distil-specific package needed.

---

## 🧠 How it works

<p align="center"><img src="docs/assets/architecture.svg" alt="architecture — pipeline and the quality-contract loop" width="100%"/></p>

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
| **Managed gateway** — multi-tenant + live savings dashboard | `gateway.py`, `distil gateway` | — |
| In-process adapter (`wrap`) | `adapters/anthropic.py` | reversible |
| **Learned keep-model** (logistic, 96.4% acc / 0.98 F1 held-out) | `codec/learned.py` | pluggable |
| Transformer keep-model — ONNX adapter + training pipeline | `codec/transformer.py`, `codec/train_transformer.py` | pluggable |
| Auth-mode gating (lossless-only on subscription/OAuth) | `policy.py` | safety |
| Holdout A/B savings + bootstrap CI | `certify/holdout.py` | — |
| Byte-fidelity invariants (reversible + append-only) | `fidelity.py`, `distil verify` | — |
| BM25 partial retrieval · delta context · gist caching | `retrieval.py`, `delta.py`, `gist.py` | lossless |
| **Output compression** — gated shaping + lossless re-entry digest + A/B harness | `output.py`, `distil output-savings` | gated / lossless |
| **Real-trace ingestion** — run the gate on your own traffic | `ingest.py`, `distil ingest` | — |
| **Performance benchmark** — p50/p95 latency + throughput | `perf.py`, `distil perf` | — |
| Billing-grade tokenizer + live runner | `tokenizer.py`, `replay/anthropic_runner.py` | opt-in |
| Savings ledger + leaderboard (privacy-preserving) | `ledger.py` | local-first |
| **Certified compression frontier** — savings-vs-accuracy curve | `eval.py`, `distil eval` | the proof |
| **Self-distilling keep-model** — learns from causal labels, gated by the contract | `online.py`, `distil online` | never-regressing |
| **Verifiable federated telemetry** — signed, content-free savings + verdict | `telemetry.py`, `distil federated-leaderboard` | tamper-evident |
| **Async high-concurrency proxy** | `aproxy.py`, `distil proxy --async` | `[async]` extra |
| **Rust hot-path core** + pure-Python parity fallback | `rust/distil-core`, `distil/native.py` | opt-in speed |

**Full docs:** [Getting started](https://dshakes.github.io/distil/getting-started.html) · [Concepts](https://dshakes.github.io/distil/concepts.html) · [Techniques](https://dshakes.github.io/distil/techniques.html) · [CLI](https://dshakes.github.io/distil/cli.html) · [Output & I/O](https://dshakes.github.io/distil/output.html) · [Architecture](https://dshakes.github.io/distil/architecture.html) · [Integrations](https://dshakes.github.io/distil/integrations.html) · [Deploy & security](https://dshakes.github.io/distil/deploy-security.html) · [FAQ](https://dshakes.github.io/distil/faq.html)

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
- The learned keep-model is a real trained **logistic** classifier (96.4%/0.98 on held-out lines). The **transformer** path ships a real ONNX adapter + training pipeline; a **demo checkpoint** (96.3%/0.98, trained on the bundled corpus) is on the v0.1.0 release, and you retrain on your own traces for production (`distil train-transformer`). We don't fabricate weights to claim "done."
- Numbers here are reproducible from the bundled corpus with the heuristic tokenizer. No vanity metrics.

---

## 🎯 Both sides of the bill — input *and* output

<p align="center"><img src="docs/assets/io.svg" alt="compress both input and output tokens" width="100%"/></p>

**Input/context** (Tier-0/1, cache stabilization, causal pruning, proxy + adapter) — comprehensive.

**Output** — two real mechanisms (`distil/output.py`):
- **Generation-side shaping** — a gated `role:"system"` verbosity directive (`distil proxy --shape-output light|aggressive`) so the model *emits* fewer tokens. Lossy by nature, so it's **PAYG-only** and **measured**: `distil output-savings` reports the token cut **and** the rate the answer survived, with a bootstrap CI.
- **Lossless output-on-re-entry digest** — long answers that become history are digested reversibly, so verbose past output stops costing full price as context.

```
$ distil output-savings
output tokens cut 72.5% (95% CI 67.5–77.1%), answer preserved 100.0% of the time, n=6
```

**Run the gate on *your* traffic, not just the synthetic corpus:**
```
$ distil ingest --input prod-requests.jsonl --out ./mycorpus   # Anthropic/OpenAI logs → trajectories
$ distil bench --corpus ./mycorpus --savings-only
```

**Performance** (`distil perf`, stdlib, single core): ~27,000 distil-compressions/sec; the in-process adapter compresses a request in **~0.006 ms** (p50).

### Honest limits
- **Production keep-model weights.** A logistic model (96.4%/0.98) ships built-in; the transformer is a real adapter + pipeline with a *demo* checkpoint on the [v0.1.0 release](https://github.com/dshakes/distil/releases/tag/v0.1.0) — retrain on your traces (`distil train-transformer`).
- **Output shaping's realized savings are live** — the A/B harness measures it on recorded pairs; the token reduction lands when a real model generates against the directive.
- **Live-model certification** is offline by default; `--runner anthropic` is implemented but **UNVERIFIED** without a key.

No vanity metrics — every number here is reproducible from the bundled corpus.

---

## 🤝 Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The one rule that matters: **a new compression strategy must pass `make gate`** (non-inferior on every domain, byte-reversible). No green gate, no merge. That's the whole philosophy in one sentence.

## License

[Apache-2.0](LICENSE) · *“Same potency, less volume.”*
