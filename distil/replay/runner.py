"""AgentRunner — the seam between this harness and a real agent.

`DeterministicRunner` is an offline stand-in whose decision is a pure function
of the decision-relevant content present in context: it reads every line
carrying a `DECISION:` marker. So removing relevant content changes the
decision and removing noise does not — which is exactly what lets causal
ablation and certification produce real, explainable divergence with no API key.

To certify against a live model, implement `AgentRunner.decide` to call the
provider and return a canonical fingerprint of the agent's action (e.g. the
next tool call + its arguments). Everything downstream is unchanged.
"""

from __future__ import annotations

from typing import Protocol

from ..trajectory import Block


class AgentRunner(Protocol):
    def decide(self, blocks: list[Block]) -> str: ...


class DeterministicRunner:
    name = "deterministic"

    def decide(self, blocks: list[Block]) -> str:
        drivers: list[str] = []
        for b in blocks:
            for line in b.text.splitlines():
                if "DECISION:" in line:
                    drivers.append(line.split("DECISION:", 1)[1].strip())
        return " | ".join(sorted(drivers)) or "<no-op>"
