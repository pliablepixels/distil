<p align="center">
  <img src="docs/assets/banner.svg" alt="Distil — compression with a quality contract" width="100%"/>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-8b7bff" alt="license"/></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-5ad1c9" alt="python"/>
  <img src="https://img.shields.io/badge/runtime%20deps-0-5ad19a" alt="zero deps"/>
  <img src="https://img.shields.io/badge/tests-737%20passing-5ad19a" alt="tests"/>
  <img src="https://img.shields.io/badge/typed-py.typed%20%C2%B7%20mypy%20clean-8b7bff" alt="typed"/>
  <img src="https://img.shields.io/badge/corpus%20gate-PASS-5ad19a" alt="gate"/>
  <img src="https://img.shields.io/badge/works%20with-any%20SDK-8b7bff" alt="any sdk"/>
</p>

<h3 align="center">The most tokens you can save without losing outcomes — and the only compressor that can prove the second half.</h3>

<p align="center">
Every agent re-sends its whole context every turn — you pay for all of it, every turn. Compressing it is easy; compressing it <strong>without quietly changing what your agent does</strong> is the part everyone skips. Distil ships a <strong>statistical proof</strong> the next action is unchanged — measured on the same runs as the savings, gated in CI — and falls back to full context when it can't certify. <strong>Never silently lossy.</strong></p>

<p align="center"><sub><b>Honest scope:</b> the per-request certificate is a <b>proxy</b> (next-action equivalence) — under <em>aggressive lossy</em> compression it doesn't fully transfer to task success (<a href="#-the-proof">E7</a>). That's why Distil also certifies the level that matters: <a href="#-the-trajectory-level-certificate"><code>distil certify-trajectories</code></a> bounds <b>end-to-end task degradation</b> on matched runs, and Distil calibrates per deployment and fails safe.</sub></p>

<p align="center">
  <a href="#-use-it-now">Use it</a> ·
  <a href="#-works-with-every-sdk">Integrations</a> ·
  <a href="#-install-your-way">Install</a> ·
  <a href="https://dshakes.github.io/distil/getting-started.html"><b>Full Docs →</b></a>
</p>

---

<h3 align="center">Proof first — not a pitch 📊</h3>

<p align="center"><img src="docs/assets/head-to-head.svg" alt="Distil vs LLMLingua-2 vs Headroom — token savings, decision-change rate, latency" width="100%"/></p>

<table align="center">
<tr><th>On a real 500-instance long-horizon agent<br/><sub>(SWE-bench Verified, official harness)</sub></th><th>task success</th><th>tied with full context?</th><th>reversible&nbsp;+&nbsp;certified?</th></tr>
<tr><td><b>Distil</b> (relevance-gated)</td><td align="center"><b>36.8%</b></td><td align="center">✅ <b>only one</b></td><td align="center">✅</td></tr>
<tr><td>Headroom <sub>(lossy)</sub></td><td align="center">32.6%</td><td align="center">❌ −6.6pp</td><td align="center">❌</td></tr>
<tr><td>LLMLingua-2 <sub>(lossy)</sub></td><td align="center">2.4%</td><td align="center">❌ −36.8pp</td><td align="center">❌</td></tr>
<tr><td>no compression <sub>(full)</sub></td><td align="center">39.2%</td><td align="center">—</td><td align="center">—</td></tr>
</table>

<p align="center"><b>Distil is the only compressor statistically tied with full context</b> — every lossy tool craters. And on the live head-to-head above (graded by <code>claude-opus-4-8</code>), it certifies <b>83.2% savings at a 0% decision-change rate</b>, ~1,000× faster than the nearest tool. <a href="#-the-proof">Full breakdown ↓</a></p>

---

## 🚀 Use it now

**One command sets you up and tells you what to do next:**

```bash
pipx install distil-llm
distil onboard      # detects your agent + billing, wires the status line, prints a guided tour
```

`distil onboard` figures out your environment (Claude Code · Codex · Gemini CLI; metered vs subscription) and hands you the exact commands for your setup. Or go straight to wrapping your agent — **no config, no code change:**

```bash
# Claude Code on a metered API key — saves real $$:
distil wrap --expand -- claude

# Claude Code on a Pro/Max subscription — flat-rate, ToS-safe (trims context, not $):
distil wrap --lossless-only -- claude

# Codex, Gemini CLI, or any agent — same pattern:
distil wrap --expand -- codex
```

**Tired of typing `distil wrap` every time?** Make it the default — once:

```bash
distil default            # adds a managed shell alias so `claude` always routes through distil
distil default --undo     # remove it anytime (backed up before any change)
```

It detects your shell (zsh / bash / fish / PowerShell) and billing mode, writes the
right line to the rc file your shell actually reads, and **tells you what it detected**.
Want every SDK covered (not just the agent you type)? `distil default --always-on`
runs a persistent proxy service — powerful, but it's a daemon you keep alive.

Then watch genuine savings from **your** traffic — measured, not estimated:

```bash
distil leaderboard          # cumulative tokens + $ saved, from the local ledger
distil dashboard            # live terminal TUI — token-trim + decision-equiv bars, Ctrl-C to exit
```

**Validate it preserved your outcomes.** Compression is only safe if your agent makes the *same decision* it would on full context. `--shadow` proves it on your live traffic: it samples a fraction of requests, runs each one twice (compressed **and** full prompt), and compares the agent's **chosen next action** — the tool call it decides to make, not the prose:

```bash
distil wrap --shadow 0.1 -- claude   # one command: wraps your agent + shadows 10% of requests
distil shadow-stats                  # live decision-equivalence rate (or /distil-shadow)
```

Honest scope: this is **next-action equivalence — a proxy**, not end-to-end task success ([E7](#-the-proof) shows it doesn't fully transfer under aggressive *lossy* compression). Watch the rate, keep the gate conservative; Distil fails safe to full context.

> **Will it actually save me money?** Only on **metered / pay-as-you-go** billing (an API key): fewer tokens → fewer dollars. On a **subscription** you're flat-rate, so there's no per-token bill to cut — Distil still trims context and latency. **Honest about coding agents:** on short sessions the win is *modest* (~7% — the agent re-expands most of what it edits, [E7](#-the-proof)); the real savings land on **long, many-turn sessions** with large context the model never re-reads. Want an in-process hook or an org-wide proxy instead? See [Works with every SDK](#-works-with-every-sdk).

---

## 💡 Why Distil is different

**You don't need byte-equivalence — you need decision-equivalence.** An agent only has to take the *same actions* whether or not its context was compressed — and *that* is measurable, certifiable, and compatible with aggressive compression. (Honest caveat we measured ourselves: it's a **proxy**; under aggressive *lossy* compression it doesn't fully transfer to task success — [E7](#-the-proof).)

- **Certified, not estimated.** A strategy ships only if a non-inferiority test certifies the next action is unchanged — savings and accuracy on the *same runs*, gated in CI. When it can't certify, it falls back to full context.
- **Certified at the level that matters, too.** `distil certify-trajectories` bounds **end-to-end task degradation** over matched full/compressed runs (Conformal Risk Control — the certificate target our own E7 experiment showed per-step metrics can't stand in for). No other compressor certifies either level.
- **Reversible, not lossy.** Every other compressor *destroys* detail. Distil digests behind a content handle, keeps the original locally, and hands the agent a `distil_expand` tool to pull back exactly what it needs — so you can compress fearlessly.
- **It compounds — on outcomes.** Every expansion is a label that content mattered, and every matched trajectory outcome teaches the policy which content classes break tasks when digested (never content, only signatures). Both flywheels only ever make compression *more* conservative.
- **Streams like it isn't there.** SSE responses relay chunk-by-chunk — time-to-first-token is preserved through the proxy, async proxy, and gateway.

> **Three fidelity tiers:** **lossless** (byte-identical in-context, `--verbatim`) · **reversible** (digested but byte-recoverable on demand — the default) · **lossy** (gone — every other tool). All three Distil modes are certified decision-equivalent; only Distil offers, and certifies, the reversible tier.

---

## ⚡ Prove the numbers yourself — no API key

Don't take the table above on faith. `distil bench` re-certifies savings *and* decision-equivalence on a bundled 7-domain corpus, offline, in seconds — the same gate that runs in CI:

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
aggregate: distil cuts $0.14212 -> $0.10402 (26.8% cheaper) reversibly; 5761 tokens causally prunable.
GATE: PASS — every trajectory certified non-inferior; aggressive rejected on all.
```

<p align="center"><img src="docs/assets/domains.svg" alt="measured across 7 domains" width="100%"/></p>

> **Why trust the number?** Token-savings numbers are easy to fake — measure quality at *low* compression, advertise savings at *high* compression. Distil refuses that: accuracy and compression are measured on the **same** trajectories, and a strategy that can't pass non-inferiority doesn't ship.
> ```
> distil certify --strategy distil       # VERDICT: PASS  (100% decision-equivalence)
> distil certify --strategy aggressive   # VERDICT: FAIL  (mean diff −1.0, blocked)
> ```

`distil eval` plots the **certified compression frontier** — a savings-vs-quality curve where every point carries its certification verdict, locating the cliff past which lossy compression drops decisions. The artifact no competitor publishes: [benchmark.html](https://dshakes.github.io/distil/benchmark.html).

---

## 📊 The proof

Benchmarks & the research — three results, all reproducible, all published with their caveats:

- **Live head-to-head** vs the real `llmlingua` and `headroom-ai` packages (graded by `claude-opus-4-8`): Distil certifies **83.2% savings at a 0% decision-change rate**, ~1,000× faster — the only method that's the most aggressive, fully decision-equivalent, *and* lowest-latency. → [BENCHMARKS.md](BENCHMARKS.md) · [benchmark.html](https://dshakes.github.io/distil/benchmark.html)
- **End-to-end honesty (E7, SWE-bench Verified):** aggressive *lossy* compression **craters** real task success (52% → 16%) — a single-turn certificate does **not** transfer to multi-turn coding. We publish it because it's true. The **reversible** tier survives (56% vs 52%, equal to full). → [research E7](https://dshakes.github.io/distil/research.html#e7)
- **Long-horizon (E8, 500-instance ReAct agent):** the relevance-gated reversible tier is the **only compressor non-inferior to full context** (36.8% vs 39.2%) and beats every lossy tool. The guarantee lifts to whole runs (E10), generalizes across **5 models / 3 vendors** (E11), auto-calibrates per deployment, and ships an anytime-valid drift monitor (E13). → [research E8–E13](https://dshakes.github.io/distil/research.html#e8)

**Full methodology, per-instance data, McNemar tests, and the certificate-non-transfer analysis:** [`docs/PAPER.md`](docs/PAPER.md) · [compiled paper (PDF)](docs/paper/main.pdf) · production status: [`docs/GA_READINESS.md`](docs/GA_READINESS.md).

> **What these numbers are:** the headline savings are **decision-equivalence on a trajectory-corpus proxy** — real and reproducible, but *not* end-to-end task success, which under aggressive *lossy* compression doesn't fully transfer (E7). That's why Distil calibrates per deployment and falls back to full context when it can't certify.

---

## 📡 See it working

Savings you can see — measured on **your** traffic, never estimated, nothing leaves your machine:

- **Per request:** `x-distil-*` response headers (`tokens-saved`, `compressed`, `cache-prefix-msgs`, `expanded`).
- **Per machine:** `distil leaderboard` rolls up the local savings ledger (`--html` for a page).
- **Live decision-equivalence (shadow mode):** `distil proxy --shadow 0.05` runs 5% of traffic uncompressed in the background and reports the rolling decision-change rate — streaming-aware, so it works on real Claude Code / Codex / Gemini sessions.
- **Org-wide:** run `distil proxy` as a sidecar, set `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` once — every client routes through it, zero per-dev change.

The live dashboard, status-line plugin, and privacy-preserving federated leaderboard: [Deploy & observability](https://dshakes.github.io/distil/deploy-security.html).

## 🔌 Works with every SDK

One proxy. Point any `base_url`-honoring client at it — **Python, TypeScript, any language** — and get cache-aware **reversible** compression with **no code change**.

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
| Google Gemini | `--upstream https://generativelanguage.googleapis.com` | [`examples/python_gemini.py`](examples/python_gemini.py) |

Prefer in-process? Wrap the client directly — still no call-site change:

```python
from distil.adapters.anthropic import wrap
client = wrap(anthropic.Anthropic())   # compresses the request, keeps the cache warm
```

**Framework hooks (no proxy, no network hop)** — for agent frameworks that own the message list, compress it where it lives:

| Framework | Hook | Example |
|---|---|---|
| LiteLLM | `distil.integrations.litellm.compress(kwargs)` | [`examples/python_litellm.py`](examples/python_litellm.py) |
| LangChain | `distil.integrations.langchain.compress_messages(msgs)` | — |
| LangGraph | `pre_model_hook=pre_model_hook()` (compresses graph state before the model node) | [`examples/python_langgraph.py`](examples/python_langgraph.py) |

---

## 📦 Install your way

**New here?** `pipx install distil-llm`, then `distil onboard` — it sets you up and guides you (see [Use it now](#-use-it-now)). Want to see it prove itself first instead? `distil bench` runs the certified gate in ~10s, no API key. The matrix below is for picking an *install format* — everything in it is an alternative, not a requirement.

> ⚠️ **The one gotcha — the name.** The PyPI package is **`distil-llm`** but the command is **`distil`** (the bare name was taken). So `pipx install distil-llm` → run `distil …`. `pip install distil` installs something else.

> 🔧 **Seeing `Could not find a version that satisfies the requirement distil-llm (from versions: none)`?** The package **is** on PyPI — that error means your `pip`/`pipx` is on a Python older than the package's floor, so pip filters every release out. **Distil now supports Python 3.9+** (the version macOS ships), so a current install just works; if you still hit this on a very old Python, let **uv provision one for you**: `uvx --python 3.12 --from distil-llm distil bench` (or `uv tool install --python 3.12 distil-llm`). Check yours with `python3 --version`.

> 🔧 **Got an *old* version (e.g. `0.25.1`) instead of the latest?** Public PyPI always serves the newest (`pip index versions distil-llm` lists them). If you got an older one, your `pip`/`pipx` is **not resolving against public PyPI** — almost always a **stale internal mirror** (Artifactory / CodeArtifact / Nexus that hasn't synced the latest yet — common right after a release) or a **`<1.0` version pin** in a constraints file / `pip.conf`. Diagnose and fix:
> ```bash
> pip index versions distil-llm     # stops at an old version? → your index/mirror is stale
> pip config list ; env | grep -i pip   # look for an index-url or PIP_CONSTRAINT pin
> # unblock now — force public PyPI:
> pipx install --pip-args="--index-url https://pypi.org/simple/" distil-llm
> # (or, if you must use the mirror, ask your platform team to sync distil-llm; it exists upstream)
> ```

<p align="center"><img src="docs/assets/install.svg" alt="install options" width="100%"/></p>

| Format | Command | Prereq |
|---|---|---|
| **Zero install** | `uvx --from distil-llm distil bench` | [uv](https://docs.astral.sh/uv/) — **auto-provisions Python 3.9+** |
| **Isolated CLI** | `pipx install distil-llm` → `distil bench` | Python **3.9+** (else `pipx install --python python3.12 distil-llm`) |
| **Homebrew** | `brew install dshakes/tap/distil` | Homebrew |
| **Docker** | `docker build -t distil . && docker run distil bench` | Docker |
| **Single file** | `make pyz` → `python dist/distil.pyz bench` | Python 3.9+ |
| **In a venv** | `pip install distil-llm` (inside an active virtualenv) | Python 3.9+ |

> The import package and CLI are `distil`; the PyPI distribution is `distil-llm` (the bare name was taken — so `uvx`/`pip` must reference `distil-llm`, not `distil`). Distil is a CLI: install it **isolated** (pipx/uv/brew/Docker), because modern macOS/Linux block system-wide `pip install` ([PEP 668](https://peps.python.org/pep-0668/)). **Node / any language:** point your SDK's `base_url` at `distil proxy`, or use `distil wrap -- <agent>` — no Distil-specific package needed.

---

## 🧰 Cheat-sheet

Basics are in [Use it now](#-use-it-now) and [Works with every SDK](#-works-with-every-sdk). Beyond that:

| Goal | Command |
|---|---|
| **Set up + a guided tour (start here)** | `distil onboard` |
| Make distil the default (no per-session `wrap`) | `distil default` · undo: `distil default --undo` |
| Remove distil's footprint (before uninstalling) | `distil offboard` · also clear data: `distil offboard --purge` |
| Diagnose your setup (ledger, shadow, proxy self-test, wiring) | `distil doctor` |
| Wire the savings status line into Claude Code | `distil setup` |
| Watch genuine savings accumulate | `distil leaderboard` · live TUI: `distil dashboard` |
| Live decision-equivalence on real traffic | `distil wrap --shadow 0.1 -- claude` → `distil shadow-stats` |
| Certify on *your* domain | `distil ingest --input prod.jsonl --out ./mycorpus` → `distil conformal --corpus ./mycorpus` |
| Recover digested detail from any agent (MCP) | `distil mcp` |
| Self-improving keep policy | `distil learn` / `distil online` |

> On a flat-rate **Claude subscription** the dollar figures are notional — the status line and
> `distil dashboard` auto-detect it and show the token reduction only (override with
> `DISTIL_SUBSCRIPTION=0/1`).

Rule of thumb: **subscription/interactive → `--lossless-only` (+`--verbatim`)** · **PAYG/autonomous → default digest (+`--expand`)** · **coding re-reads → add `--session-delta`**.

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

## 🎓 The certificate (DERC)

The gate answers *"is this strategy non-inferior on my corpus?"*. The **Decision-Equivalence Risk Certificate** answers the operational one: *"for a risk budget I choose (say ≤5% decision-change), how hard can I compress with a guarantee that holds on my real traffic?"*

```bash
distil conformal --corpus ./mycorpus --alpha 0.05 --delta 0.05
# ✔ CERTIFIED 'lossless' → 57.4% savings; decision-change ≤ 5.0% at 95% confidence (Learn-Then-Test)
```

It's **conformal risk control** (Learn-Then-Test / CRC — distribution-free, finite-sample), not a heuristic threshold. The one load-bearing caveat: the guarantee requires **exchangeability** (calibration traffic ≈ live traffic) and is **marginal** over that distribution — recalibrate on drift. Full theory + citations: [Concepts](https://dshakes.github.io/distil/concepts.html) · [`docs/PAPER.md`](docs/PAPER.md).

### 🏔 The trajectory-level certificate

DERC certifies the *step*; this certifies the *task*. Our E7 experiment — and the 2024–26 agent-compression literature — shows per-step fidelity can pass while end-to-end success collapses, so distil also certifies the level users actually feel: run your eval suite twice (full context vs compressed), feed the matched outcomes in, and get a distribution-free bound on **how many solvable tasks compression may cost you**:

```bash
distil certify-trajectories outcomes.jsonl --alpha 0.05 --delta 0.05
# each line: {"task_id": "...", "full_success": true, "compressed_success": true}
# → With confidence 95%, compression degrades at most 5.0% of tasks the full
#   context would have solved (observed 0.5% over 200 matched trajectories).
```

It refuses to certify on small samples, states its exchangeability assumptions in the certificate itself, and ships an anytime-valid **drift monitor** (`trajectory_risk.drift_monitor`) that tells you when live traffic has shifted enough that the certificate is stale. Matched failures also feed the **outcome-guided policy** (`distil.compress.guideline`): content classes that break tasks when digested get protected byte-exact, automatically.

## 🧩 What's inside

40+ shipped capabilities, all real (no stubs): the cache-aware cost engine, causal pruning, the TOST gate + conformal certificate, the proxy + Anthropic/OpenAI/Gemini adapters, an MCP server, LiteLLM/LangChain/LangGraph hooks, learned keep-models, output compression, and a Rust hot-path core — with **zero runtime dependencies** in the core.

Full module-by-module map: [Architecture](https://dshakes.github.io/distil/architecture.html) · [Techniques](https://dshakes.github.io/distil/techniques.html) · [CLI reference](https://dshakes.github.io/distil/cli.html).

## 🔒 Security & deployment

- **Localhost-only by default** — the proxy binds `127.0.0.1` and forwards only to the single configured upstream (no SSRF).
- **No secret/body logging** — request bodies and credentials are never logged.
- **Auth-mode gating** — `--lossless-only` keeps subscription/OAuth sessions to lossless strategies and never injects tools (provider-ToS-safe); the reversible, certified digest still runs. Add `--verbatim` to skip the digest entirely (Tier-0 only) for interactive sessions.
- **Stateless** — nothing is persisted; ZDR-compatible.

See [Deploy & security](https://dshakes.github.io/distil/deploy-security.html) for topologies (local sidecar, container sidecar, shared gateway) and the threat model.

---

## ✅ What we won't pretend

- **Default tokenizer is an offline heuristic** (zero deps); ratios are robust, dollars are approximate. Use `--tokenizer anthropic` for billing-grade counts (the correct Claude tokenizer — tiktoken undercounts Claude).
- **The default runner is a deterministic stand-in** so the gate runs offline with ground truth. For a *non-circular* evaluation, the **proof harness** (`benchmarks/prove.py`) grades **real agent traces** (τ-bench / SWE-bench) with a **real model** — see [Reproducible evaluation & the paper](#-reproducible-evaluation--the-paper).
- **What the real-trace runs taught us (and we now enforce):** measuring decision-equivalence credibly requires (1) **majority-vote grading** (a single sample lets grader noise masquerade as a decision change), (2) a **faithful grader** (same model family as the trace), and (3) grading the **reversible tier *with* its `distil_expand` recovery loop** — without recovery the digest is only a conservative lower bound. The aggressive headline savings are realized *with* recovery active; the harness measures both bounds.
- The learned keep-model is a real trained **logistic** classifier (96.4%/0.98 on held-out lines). The **transformer** path ships a real ONNX adapter + training pipeline; a **demo checkpoint** (96.3%/0.98, trained on the bundled corpus) is on the v0.1.0 release, and you retrain on your own traces for production (`distil train-transformer`). We don't fabricate weights to claim "done."
- Numbers here are reproducible from the bundled corpus with the heuristic tokenizer. No vanity metrics.

### Deliberately *not* a platform

Distil is a **compression engine with a correctness gate**, not a context-management suite. Some adjacent products bundle a memory/knowledge-graph store, a hosted semantic cache, and editor-auth plumbing. We've looked at each and **declined the ones that can't be put under the certificate** — bundling them would dilute the one promise that matters (decision-equivalence) and add state we can't prove safe:

| Adjacent feature | Our stance |
|---|---|
| Persistent memory / knowledge graph | **Out of scope.** A lossy store of "what mattered" is the opposite of byte-reversible. Use a real memory tool; Distil compresses what you *do* send. |
| Hosted semantic cache | **Out of scope as a service.** We make the *provider's* prompt cache pay off (cache-monotonic prefixes); we don't add a second, lossy, similarity-keyed cache that can return a near-but-wrong hit. |
| Editor/Copilot auth | **Out of scope.** Distil sits on the wire (proxy) or in-process (hooks); it never brokers your credentials. |

What we **did** adopt from that space — because it survives the gate — is on-motto: a **pluggable salience scorer** seam (`salient_tokens(scorer=…)`) so you can bolt on a semantic/NER model to *protect* entities, never to drop them; **cache-prefix observability** (`x-distil-cache-prefix-msgs`) exposing exactly how many leading messages stayed byte-stable; and **first-class framework hooks** (LiteLLM/LangChain/LangGraph). The seam adds *coverage*; the certificate still owns the *guarantee*.

---

## 🎯 Both sides of the bill

Distil compresses **input/context** (comprehensive) **and output** — generation-side verbosity shaping (PAYG, measured with `distil output-savings`) plus a reversible output-on-re-entry digest, so verbose past answers stop costing full price as history. Details: [Output & I/O](https://dshakes.github.io/distil/output.html).

## 🔬 Reproducible evaluation & the paper

Every number reproduces from the bundled corpus (`distil bench`, no key). The non-circular proof harness grades **real agent traces with a real model** (τ-bench / SWE-bench): [`benchmarks/PROVE.md`](benchmarks/PROVE.md). Compiled paper, LaTeX source, and all committed results: [`docs/PAPER.md`](docs/PAPER.md) · [`docs/paper/`](docs/paper/) · [paper PDF](docs/paper/main.pdf).

<h3 align="center">Stop paying to re-send context your agent never reads.</h3>

<p align="center">
<code>pipx install distil-llm && distil bench</code><br/>
<sub>certified savings across 7 domains in ~10 seconds — zero API key, zero runtime deps</sub>
</p>

<p align="center">
<a href="https://dshakes.github.io/distil/getting-started.html"><b>Get started →</b></a> ·
<a href="#-works-with-every-sdk">Wire it into your SDK</a> ·
<a href="docs/PAPER.md">Read the proof</a> ·
<a href="https://pypi.org/project/distil-llm/">PyPI</a>
</p>

---

## 🤝 Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The one rule that matters: **a new compression strategy must pass `make gate`** (non-inferior on every domain, byte-reversible). No green gate, no merge. That's the whole philosophy in one sentence.

## License

[Apache-2.0](LICENSE) · *“Same potency, less volume.”*
