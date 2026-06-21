"""LiteLLM + Distil proxy example.

Run `distil proxy` first, then point LiteLLM at it:

    distil proxy --port 8788 --upstream https://api.anthropic.com

LiteLLM's `api_base` overrides the upstream URL for the selected provider,
so all traffic goes through the proxy and is compressed transparently.
"""

import os
import litellm

# Set api_base to the local proxy.  LiteLLM will prefix the appropriate
# path (/v1/messages for Anthropic, /v1/chat/completions for OpenAI-compat)
# which the proxy handles transparently.
response = litellm.completion(
    model="claude-opus-4-5",
    api_base="http://127.0.0.1:8788",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    messages=[{"role": "user", "content": "Summarise the key ideas behind prompt compression."}],
)

print(response.choices[0].message.content)

# To route OpenAI traffic through the proxy instead, use:
#   litellm.completion(
#       model="gpt-4o",
#       api_base="http://127.0.0.1:8788",
#       api_key=os.environ["OPENAI_API_KEY"],
#       messages=[...],
#   )
# but start the proxy with --upstream https://api.openai.com first.
