---
description: Diagnose your distil setup — ledger, shadow validation, proxy round-trip, wiring
allowed-tools: Bash(distil *), Bash(uvx *)
---

Run `distil doctor` and present its diagnosis clearly: which checks pass, which need
attention, and the exact next step for each.

`distil doctor` checks: distil/Python version, the savings ledger (and whether live
traffic is recorded), shadow-validation status, an in-process proxy round-trip
self-test, the optional `anthropic` extra + API key, and — for Claude Code — whether
the status line is wired and whether this is a flat-rate subscription.

Rules:
- Only report what `distil doctor` actually prints — never invent results.
- Lead with anything **failing** (✗) or needing attention (⚠), with its fix hint.
- If everything is healthy, say so in one line.
- If shadow validation isn't running, point to the one-command fix:
  `distil wrap --shadow 0.1 -- claude`.
