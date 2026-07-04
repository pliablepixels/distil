---
description: Generate a shareable badge of your measured distil savings
allowed-tools: Bash(distil *), Bash(uvx *)
---

Generate the user's savings badge:

```bash
distil stats --badge
```

Show them the badge URL and the ready-to-paste markdown. The number is their
own locally-measured savings (genuine, content-free) — suggest pasting it into
a project README or sharing it. If there are no savings recorded yet, say so
and suggest `distil wrap -- <agent>` instead of generating an empty badge.
