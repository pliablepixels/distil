"""Anthropic SDK + Distil proxy example.

Run `distil proxy` first, then point the SDK at it:

    distil proxy --port 8788 --upstream https://api.anthropic.com

Every request is transparently compressed before being forwarded.
No other code changes are needed — the SDK behaves exactly as normal.
"""

import os
import anthropic

# Point the SDK at the local proxy instead of api.anthropic.com.
# The proxy forwards all traffic (including your API key header) to the
# real upstream after compressing the messages array.
client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="http://127.0.0.1:8788",
)

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Summarise the key ideas behind prompt compression."}],
)

print(response.content[0].text)

# The proxy adds two informational response headers:
#   x-distil-compressed: 1          — compression was applied
#   x-distil-tokens-saved: <n>      — estimated tokens saved this turn
# These are passed through to the client but do not affect the response body.
