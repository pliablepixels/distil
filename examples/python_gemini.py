"""Google Gemini + Distil proxy example.

Run `distil proxy` pointed at the Gemini API first:

    distil proxy --port 8788 --upstream https://generativelanguage.googleapis.com

Then call Gemini's REST endpoint through the proxy. Distil compresses the request
``contents`` (text parts + ``functionResponse`` tool outputs) reversibly before
forwarding — no other change needed. Use ``--lossless-only`` for a mode that keeps
every line the model can see (safe when the model's output is read by a human).

The official ``google-genai`` SDK doesn't expose a base-URL override on every
version, so this example uses a plain HTTP call to make the routing explicit.
"""

import json
import os
import urllib.request

API_KEY = os.environ["GEMINI_API_KEY"]
MODEL = "gemini-1.5-pro"

# Point at the local Distil proxy instead of generativelanguage.googleapis.com.
url = f"http://127.0.0.1:8788/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

payload = {
    "contents": [
        {"role": "user", "parts": [{"text": "Summarise the key ideas behind prompt compression."}]},
    ],
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req) as resp:
    print("x-distil-tokens-saved:", resp.headers.get("x-distil-tokens-saved"))
    data = json.load(resp)

print(data["candidates"][0]["content"]["parts"][0]["text"])
