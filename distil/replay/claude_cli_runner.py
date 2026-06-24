"""AgentRunner backed by the `claude -p` CLI (Claude Code in print mode).

Grade decision-equivalence against a real Claude model using your **existing Claude
Code subscription / OAuth** — no ``ANTHROPIC_API_KEY`` required, no separate billing.
The harness shells out to ``claude -p "<prompt>" --output-format json`` per decision,
parses the structured envelope's ``result`` field, and extracts the canonical
``{action,target}`` fingerprint with the shared parser.

This is the lowest-friction way to get a *real-model* result (the proof the paper
needs) when you already run Claude Code. ``samples>1`` takes the majority vote to
remove run-to-run variance.

Notes
-----
* ``claude`` must be on PATH and authenticated (``claude`` interactively once).
* Pick the model with ``--model`` (e.g. a Haiku id for cheap large sweeps, an Opus
  id for the headline run). Defaults to whatever the CLI is configured to use.
* Each call is an independent CLI invocation (``--no-session`` semantics via ``-p``),
  so there is no cross-decision state leakage.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter

from ..trajectory import Block
from . import prompts


class ClaudeCliRunner:
    name = "claude-cli"

    def __init__(
        self,
        *,
        bin: str = "claude",
        model: str | None = None,
        samples: int = 1,
        timeout: float = 180.0,
        extra_args: tuple[str, ...] = (),
    ) -> None:
        self.bin = bin
        self.model = model
        self.samples = max(1, samples)
        self.timeout = timeout
        self.extra_args = tuple(extra_args)

    def decide(self, blocks: list[Block]) -> str:
        if self.samples == 1:
            return self._sample(blocks)
        votes = Counter(self._sample(blocks) for _ in range(self.samples))
        return votes.most_common(1)[0][0]

    def _sample(self, blocks: list[Block]) -> str:
        system, user = prompts.decision_prompt(blocks)
        prompt = f"{system}\n\n{user}"
        cmd = [self.bin, "-p", prompt, "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        cmd += list(self.extra_args)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            return "<no-decision>"
        out = proc.stdout.strip()
        return prompts.parse_fingerprint(self._result_text(out))

    @staticmethod
    def _result_text(stdout: str) -> str:
        """Pull the model's text from the ``--output-format json`` envelope; fall back
        to the raw stdout if it isn't the expected shape."""
        try:
            env = json.loads(stdout)
            if isinstance(env, dict) and "result" in env:
                return str(env["result"])
        except (json.JSONDecodeError, ValueError):
            pass
        return stdout
