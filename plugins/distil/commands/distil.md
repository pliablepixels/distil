---
description: Show your distil compression savings and how to route more traffic through it
allowed-tools: Bash(distil *), Bash(uvx *)
---

Report the user's distil compression savings and how to get more. Be concise and
only report what the commands actually output — do not invent numbers.

1. Run `distil leaderboard` to show genuine cumulative savings from the local ledger
   (tokens saved, dollars saved, runs). If it reports nothing yet, say so.
2. If shadow-mode data exists, run `distil shadow-stats` and report the live
   decision-equivalence. If there are no samples, skip it.
3. Briefly remind the user how to route more traffic through distil:
   - Any agent/CLI: `distil wrap --lossless-only -- <command>` (subscription/OAuth-safe; add `--verbatim` for interactive sessions, where the model should see content un-digested).
   - Any SDK: point its `base_url` at a running `distil proxy`, or set
     `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` org-wide.
   - Google Gemini: `distil proxy --upstream https://generativelanguage.googleapis.com`.

If the user passed an argument (e.g. `/distil shadow`), focus the report on that area.
