# @distil/proxy

Launch the [Distil](https://github.com/dshakes/distil) compression proxy from Node.js — a thin launcher around the Python `distil-llm` package.

## What this is

This package is **a launcher, not a reimplementation**. Distil is a Python package that runs an HTTP proxy; once it is running you point any SDK's `baseURL` at it and every request is transparently compressed before being forwarded to the real upstream.

## Usage

```sh
npx @distil/proxy --port 8788 --upstream https://api.anthropic.com
```

Then in your application — no code changes needed beyond the `baseURL`:

```ts
import { createAnthropic } from "@ai-sdk/anthropic";

const anthropic = createAnthropic({
  baseURL: "http://127.0.0.1:8788",
  apiKey: process.env.ANTHROPIC_API_KEY,
});
```

## Options

All flags are forwarded to `distil proxy`:

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8788` | Port to listen on |
| `--host` | `127.0.0.1` | Interface to bind |
| `--upstream` | `https://api.anthropic.com` | Real LLM API base URL |
| `--lossless-only` | off | Apply only Tier-0 lossless transforms |

## Requirements

- Node.js 18+
- Python 3.11+ with `distil-llm` installed

If `distil-llm` is not installed the launcher will attempt to install it via `pipx run distil-llm` (preferred) or `pip install --user distil-llm`.

## Manual Python install

```sh
pip install distil-llm     # or: pipx install distil-llm
distil proxy --port 8788 --upstream https://api.anthropic.com
```

## SDK integration examples

See [examples/](https://github.com/dshakes/distil/tree/main/examples) in the Distil repository for working snippets for Anthropic SDK, OpenAI SDK, Vercel AI SDK, LangChain.js, and LiteLLM.
