# Benchmarks — head-to-head against real tools

This directory holds the **reproducible head-to-head harness** for comparing
Distil against real, installed compression tools — kept out of the published
`distil-llm` package (the product and its docs stay competitor-neutral; this is
a dev/verification artifact).

Everything here is measured on the **same** machinery Distil holds itself to:
the decision-equivalence + non-inferiority gate (`distil.certify`) and the
cache-aware cost model (`distil.compress.cache_aware`). The harness will rank a
competitor *above* Distil if it earns it. The point is that the result is
reproducible and falsifiable — not a slogan.

## The varied corpus

```bash
python benchmarks/gen_corpus.py     # → benchmarks/corpus_xl/ (24 trajectories, 8 families)
```

Deterministic and seeded, so the numbers reproduce exactly. It spans both
regimes on purpose, so no tool gets a home-turf advantage:

- **structured / repetitive**: JSON record arrays, SQL rows, metrics, logs
- **diagnostic / prose**: Kubernetes incidents, stack traces, RAG chunks, transcripts

Decisions are **buried inside** large tool outputs (as on real agents), so naive
head/tail truncation drops them — exactly as it would in production.

## Run the comparison

```bash
# Distil + the built-in faithful technique-family baselines, on the varied corpus
distil benchmark --corpus benchmarks/corpus_xl

# add a REAL external tool via the --external module:function[:Name] seam
pip install headroom-ai
PYTHONPATH=. distil benchmark --corpus benchmarks/corpus_xl \
  --external benchmarks.headroom_adapter:compress:Headroom
```

## Adapters (and honest caveats)

### `headroom_adapter.py` — headroom-ai

Wraps `headroom.compress`. **Two fairness facts matter** (both learned the hard way):

1. Headroom's router **protects plain user/system messages** and only compresses
   *tool outputs*. Presenting blocks as plain user text yields a misleading 0%.
   The adapter presents each block as a **tool_result** (Headroom's actual
   target) and uses a low `model_limit` — giving it its best, fair shot.
2. The `--external` seam maps texts 1:1, so Headroom's *cross-message* dedup
   isn't exercised; it is compared on per-block structural/text compression.

What we observe: on clean structured JSON, Headroom's SmartCrusher cuts ~55%
**lossily**; on real agent diagnostics it **protects** the content (≈0%). On the
varied corpus it lands ~41% raw but its lossy transforms **flip ~14% of
decisions**, so the gate disqualifies it. Distil reaches higher savings *and*
stays decision-equivalent — and is reversible, not lossy.

### `rtk_adapter.py` — rtk-ai/rtk

RTK is a **command wrapper** (it re-runs `git status`, `cargo test`, … and strips
their known boilerplate), not a general text compressor — as of writing it
exposes **no raw-text/stdin mode**. The adapter shells out to the real `rtk`
binary, probes for a text mode, and — if none exists — **raises a clear error
instead of fabricating a number**. RTK is best compared on trajectories built
from real dev-command outputs; it operates at a different layer than Distil and
Headroom.

## Caveats

- The deterministic runner certifies *structurally* (decisions preserved).
  For live task-accuracy, run with `--runner anthropic` and an API key.
- These adapters are validated against the package versions current at the time
  of writing; if a tool changes its API, update the adapter (each is small and
  self-documenting) — do not trust a silent 0%/fallback.
