# Distil proxy — integration examples

Start the proxy once, then point any SDK at it — no other code changes needed.

```sh
distil proxy --port 8788 --upstream https://api.anthropic.com
```

## SDK → baseURL mapping

| Example file | SDK / framework | Key setting | Value |
|---|---|---|---|
| `python_anthropic.py` | Anthropic Python SDK | `base_url` | `http://127.0.0.1:8788` |
| `python_openai.py` | OpenAI Python SDK | `base_url` | `http://127.0.0.1:8788/v1` |
| `python_litellm.py` | LiteLLM | `api_base` | `http://127.0.0.1:8788` |
| `js_vercel_ai_sdk.ts` | Vercel AI SDK (`@ai-sdk/anthropic`) | `baseURL` in `createAnthropic({…})` | `http://127.0.0.1:8788` |
| `js_langchain.ts` | LangChain.js (`@langchain/anthropic`) | `anthropicApiUrl` in `ChatAnthropic({…})` | `http://127.0.0.1:8788` |

## Running the examples

### Python

```sh
# Install the SDK you want to use (Distil itself needs no extras)
pip install anthropic           # for python_anthropic.py
pip install openai              # for python_openai.py
pip install litellm             # for python_litellm.py

# Start the proxy (in a separate terminal)
distil proxy --port 8788 --upstream https://api.anthropic.com

# Run the example
ANTHROPIC_API_KEY=sk-ant-… python examples/python_anthropic.py
```

### TypeScript / Node

```sh
# Install dependencies
npm install @ai-sdk/anthropic ai          # for js_vercel_ai_sdk.ts
npm install @langchain/anthropic          # for js_langchain.ts
npm install -D tsx                        # TypeScript runner

# Start the proxy (in a separate terminal)
distil proxy --port 8788 --upstream https://api.anthropic.com

# Run an example
ANTHROPIC_API_KEY=sk-ant-… npx tsx examples/js_vercel_ai_sdk.ts
ANTHROPIC_API_KEY=sk-ant-… npx tsx examples/js_langchain.ts
```

## How the proxy works

The proxy is a local HTTP server (`distil proxy`, default `http://127.0.0.1:8788`).
It intercepts `/v1/messages`, `/v1/chat/completions`, and `/v1/responses` requests,
compresses the `messages` array (lossless Tier-0 + reversible Tier-1 digests), then
forwards the smaller payload to the real upstream. All other paths pass through
unchanged. Your API key travels in the request headers exactly as normal — it is
never logged or stored by the proxy.

Two extra response headers are added for observability:

| Header | Meaning |
|---|---|
| `x-distil-compressed: 1` | Compression was applied this turn |
| `x-distil-tokens-saved: <n>` | Estimated input tokens saved |
