---
description: Set up distil and get a guided, tailored tour of how to use it
allowed-tools: Bash(distil *), Bash(uvx *)
---

Run `distil onboard` and walk the user through getting started.

`distil onboard` detects their environment (OS, package managers, agent CLIs, the
optional `anthropic` extra, Claude Code + whether they're on a flat-rate
subscription), wires the savings status line, and prints a next-steps guide
tailored to what it found.

Present it conversationally:
- Summarize what was detected (agent, billing mode, whether the status line got wired).
- Walk through the tailored next steps it printed — especially the **route** command for
  their agent and the **shadow validation** one-liner (`distil wrap --shadow 0.1 -- <agent>`).
- Offer to run the first step for them if they want.

Rules:
- Only report what `distil onboard` actually prints — never invent detected state.
- If it reports a status-line conflict, explain `distil onboard --force` (it backs up the
  existing line first).
- If no agent CLI was detected, the first step is to install one — relay that.
