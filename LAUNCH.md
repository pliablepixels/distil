# distil — GA launch week plan

**Positioning (locked):** *The most tokens you can save without losing outcomes — and
the only compressor that can prove the second half.*

Every asset delivers one of three proofs, never an adjective:

1. **The E8 table** — the only compressor statistically tied with full-context task
   success on SWE-bench Verified (lossy competitors: −6.6pp to −36.8pp). Lives at
   `docs/compare.html` and in the README.
2. **The 10-second local proof** — `uvx --from distil-llm distil bench` (no API key):
   the certified gate runs on the reader's own machine.
3. **The reader's own number** — `distil stats --badge` after routing real traffic.

**Principle:** distil's unfair asset is *runnable proof*. Every touchpoint hands the
reader a command, not a claim.

---

## Day 0 — GA

- [ ] Tag the release (publishes PyPI + `ghcr.io/dshakes/distil` image + Pages site
      with the comparison page).
- [ ] Add GitHub topics: `llm`, `context-compression`, `claude-code`,
      `token-optimization`, `conformal-prediction`, `mcp`.
- [ ] Set the repo social-preview image (the head-to-head SVG).
- [ ] Sanity: `pipx install distil-llm && distil onboard` on a clean machine.

## Day 1 — Show HN (the single highest-leverage shot)

- [ ] Title the **story arc**, not the product:
      *"Our compression certificate passed while agent success collapsed. The fix
      made the compressed agent outperform full context."*
- [ ] Body: E7 (certificate passed, success collapsed) → trajectory certificate →
      E14 (42.0% vs 39.2% on 500 SWE-bench Verified, official harness, paired CI
      −0.6..+6.2pp — say "point estimate above full, superiority not yet
      significant" and let the honesty carry it) → the `bench` one-liner + compare
      page. HN rewards honest engineering stories 10× over launches.
- [ ] Be present in the thread all day; answer with data and commands.

## Day 2 — Community posts

- [ ] **r/ClaudeAI**: the Claude Code angle — `distil wrap -- claude`, session-first
      statusline screenshot, the plugin's 8 commands.
- [ ] **r/LocalLLaMA**: the harness-data angle — "lossy compressors crater on
      SWE-bench (−6.6pp to −36.8pp); here's the official-harness data and the only
      condition that tied with full context."

## Day 3 — Distribution PRs (the Headroom playbook)

- [ ] Submit the LiteLLM integration to LiteLLM's own docs.
- [ ] PR distil into `awesome-claude-code`, `awesome-llm-tools`,
      `awesome-mcp-servers`.
- [ ] Publish the plugin to a Claude Code marketplace repo
      (`plugins/distil` is marketplace-format already).

## Day 4 — Technical deep-dive post

- [ ] dev.to / blog, cross-linked from the repo: the E14 story with real numbers —
      "head-truncation eats the traceback tail; keeping ≤40 anomaly lines changed X"
      — plus the paper link.

## Days 5–7 — The badge flywheel

- [ ] Tweet your own `distil stats --badge` output (a real number), invite replies
      with theirs. Every badge links back to the repo.
- [ ] Repost the best community numbers.

---

## Evergreen loops (already shipped in-product)

- `distil stats --badge` — shareable, verified, links back.
- README star CTA ("a star is how the next engineer finds provable savings").
- `docs/compare.html` — the canonical answer to every "vs Headroom / rtk?" question.
- The statusline itself — every user terminal screenshot is an ad.

## Metrics

- Week-1 GitHub stars (baseline for a good HN technical story: ~300–800).
- PyPI downloads/day; GHCR pulls.
- LiteLLM docs merge; awesome-list merges; marketplace listing live.
- The real one: shadow-mode equivalence reports coming back from strangers' traffic.

## Owner-only actions (nothing else blocks on them)

The HN submit, the Reddit posts, the LiteLLM/awesome-list PRs from your account,
the marketplace listing, and the tweet. All copy, pages, numbers, and commands are
in this repo.
