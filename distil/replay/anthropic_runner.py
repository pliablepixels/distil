"""Live AgentRunner backed by the Claude API — for billing-grade certification.

`decide()` renders a trajectory's blocks into a real Messages request and forces
a single structured decision via a strict tool, returning a canonical fingerprint
(action + target) of what the agent chose. Certification then compares that
fingerprint with and without compression — exactly as the offline
DeterministicRunner does, but against the real model.

Requires the `anthropic` SDK and credentials. Imported lazily so the core stays
dependency-free. NOTE: not exercised in this repo's offline test suite (no API
key); treat live results as UNVERIFIED until you run them against your account.
"""

from __future__ import annotations

import json

from ..trajectory import Block, Kind, Stability

_DECISION_TOOL = {
    "name": "record_decision",
    "description": "Record the single next action the agent will take given the context.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "The tool/operation to invoke next."},
            "target": {"type": "string", "description": "The primary argument or target."},
        },
        "required": ["action", "target"],
        "additionalProperties": False,
    },
}


class AnthropicRunner:
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        client: object | None = None,
        max_tokens: int = 4096,
        samples: int = 1,
    ) -> None:
        self.model = model
        self._client = client
        self.max_tokens = max_tokens
        # Newer models deprecate `temperature`, so we can't pin sampling to 0.
        # Instead, take the MAJORITY decision over `samples` calls — the stable
        # "most-likely action" — which removes the model's own run-to-run variance
        # that would otherwise masquerade as a compression-induced divergence.
        self.samples = max(1, samples)

    def _ensure_client(self) -> object:
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic()
        return self._client

    def decide(self, blocks: list[Block]) -> str:
        if self.samples == 1:
            return self._sample(blocks)
        from collections import Counter

        votes = Counter(self._sample(blocks) for _ in range(self.samples))
        return votes.most_common(1)[0][0]

    def _sample(self, blocks: list[Block]) -> str:
        # Stable system/tool context -> system prompt; everything else -> the user turn.
        system_parts = [
            b.text
            for b in blocks
            if b.stability is Stability.STABLE and b.kind in (Kind.SYSTEM, Kind.TOOLS)
        ]
        rest = [
            b
            for b in blocks
            if not (b.stability is Stability.STABLE and b.kind in (Kind.SYSTEM, Kind.TOOLS))
        ]
        user = "\n\n".join(f"[{b.kind.value}] {b.text}" for b in rest)

        client = self._ensure_client()
        resp = client.messages.create(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=self.max_tokens,
            system="\n\n".join(system_parts) or "You are an autonomous agent.",
            tools=[_DECISION_TOOL],
            tool_choice={"type": "tool", "name": "record_decision"},
            messages=[
                {
                    "role": "user",
                    "content": user + "\n\nRecord the single next action you would take.",
                }
            ],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return json.dumps(block.input, sort_keys=True, separators=(",", ":"))
        return "<no-decision>"

    def _raw(self, system: str, user: str) -> str:
        """Free-form text completion (no forced tool) — used by the expand loop, which
        needs the model to choose between requesting an expansion and committing."""
        client = self._ensure_client()
        resp = client.messages.create(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"
        )
