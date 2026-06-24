"""Open-model AgentRunner over any OpenAI-compatible endpoint — vLLM, Ollama,
LM Studio, TGI, llama.cpp server, or OpenAI itself.

Lets you run the frontier/coverage harness at scale with **no per-call API cost**:
serve an open model locally (e.g. ``vllm serve meta-llama/Llama-3.1-8B-Instruct``)
and point ``--base-url`` at it. Zero dependencies — uses stdlib ``urllib`` and the
same decision prompt / fingerprint parser as every other runner.

``decide()`` returns the canonical ``{action,target}`` fingerprint; with
``samples>1`` it takes the majority vote, removing the model's run-to-run variance
that would otherwise masquerade as a compression-induced flip.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections import Counter

from ..trajectory import Block
from . import prompts


class OpenAIRunner:
    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key_env: str = "OPENAI_API_KEY",
        samples: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: float = 120.0,
        json_mode: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.samples = max(1, samples)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.json_mode = json_mode

    def decide(self, blocks: list[Block]) -> str:
        if self.samples == 1:
            return self._sample(blocks)
        votes = Counter(self._sample(blocks) for _ in range(self.samples))
        return votes.most_common(1)[0][0]

    def _sample(self, blocks: list[Block]) -> str:
        system, user = prompts.decision_prompt(blocks)
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:  # some servers reject this; opt-in
            payload["response_format"] = {"type": "json_object"}
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=data, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 (configured URL)
            body = json.loads(resp.read().decode())
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        return prompts.parse_fingerprint(content)
