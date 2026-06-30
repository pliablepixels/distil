---
description: Full breakdown of your distil savings — orig→compressed tokens, cost, runs, per-trajectory
allowed-tools: Bash(distil *), Bash(uvx *)
---

Run `distil leaderboard` and present a clear breakdown of the user's compression savings:
**original → compressed** input tokens, the same for cost, total runs, and the per-trajectory
split if the command shows one.

Render it as a small table, and add simple Unicode bars to make the compression ratio glanceable,
e.g. `████████░░░░ 55% trimmed`. Keep it tight — this is a status report, not an essay.

Rules:
- Only report what `distil leaderboard` actually prints — never invent or estimate numbers.
- If it reports no savings yet, say so and show the one-line quickstart instead:
  `distil wrap -- <agent>`.
- If the user is on a flat-rate subscription (the `DISTIL_SUBSCRIPTION` env var is set), note that the
  dollar figures are **notional** and lead with the token reduction (the real win).

If the user passed an argument, focus the report on that area.
