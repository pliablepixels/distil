---
description: Open the interactive HTML savings dashboard in your browser
allowed-tools: Bash(distil *), Bash(uvx *), Bash(open *), Bash(xdg-open *)
---

Generate distil's self-contained HTML dashboard and open it in the user's browser.

1. Render it (the path uses the system temp dir):
   ```bash
   distil leaderboard --html "${TMPDIR:-/tmp}/distil-dashboard.html"
   ```
2. Open it — pick the right opener for the platform:
   - macOS: `open "${TMPDIR:-/tmp}/distil-dashboard.html"`
   - Linux: `xdg-open "${TMPDIR:-/tmp}/distil-dashboard.html"`

Then tell the user the dashboard is open and summarize the headline numbers (tokens and, unless on a
subscription, cost) from the leaderboard output.

Rules:
- Only report numbers the command actually produced — never invent them.
- If there are no savings recorded yet, say so and show `distil wrap -- <agent>` instead of opening an
  empty page.
