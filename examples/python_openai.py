"""OpenAI SDK + Distil proxy example.

Run `distil proxy` first, then point the SDK at it:

    distil proxy --port 8788 --upstream https://api.openai.com

Every request is transparently compressed before being forwarded.
No other code changes are needed — the SDK behaves exactly as normal.
"""

import os
import openai

# Point the SDK at the local proxy instead of api.openai.com.
# The proxy handles /v1/chat/completions (and /v1/responses) the same way it
# handles /v1/messages — compressing the messages array before forwarding.
client = openai.OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="http://127.0.0.1:8788/v1",
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarise the key ideas behind prompt compression."},
    ],
)

print(response.choices[0].message.content)
