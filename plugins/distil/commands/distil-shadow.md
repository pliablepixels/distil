---
description: Decision-equivalence report — did compression preserve your agent's next action?
allowed-tools: Bash(distil *), Bash(uvx *)
---

Run `distil shadow-stats` and report the live **decision-equivalence** result: the percentage of
shadowed requests where the agent's chosen next action was **identical with vs without compression**,
and how many samples that's based on.

Rules:
- Only report what `distil shadow-stats` actually prints — never invent numbers.
- If there are no shadow samples yet, explain that outcome-validation isn't running and show how to
  start it:
  ```bash
  distil proxy --shadow 0.1 --upstream <api>   # shadow 10% of live traffic
  ```
- Briefly remind the user this certifies **next-action equivalence — a proxy**, not end-to-end task
  success, so they should watch the rate and keep the compression gate conservative.

Frame the headline as: *"of the requests I checked, N% produced the identical next action with vs
without compression."*
