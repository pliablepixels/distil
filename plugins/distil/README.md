# Distil — Claude Code plugin

A live token-savings **status line** and a **`/distil`** command for
[distil](https://github.com/dshakes/distil) context compression.

> **What this plugin does — honestly.** A Claude Code plugin *cannot* reroute a
> running session's API traffic (the base URL is read at launch) and *cannot* set
> the main status line from its own manifest. So this plugin ships the pieces that
> work cleanly: the `/distil` command, and a status-line script you wire in once.
> To actually compress traffic, run distil as a proxy or via `distil wrap`
> (see below) — then the savings show up here.

## Install

```
/plugin marketplace add dshakes/distil
/plugin install distil@distil
```

This gives you the **`/distil`** command (savings report + setup help) and the
**`distil-setup` skill** — ask Claude Code to "set up distil" or "route my agent
through distil" and it installs the CLI and wires your agent/SDK through compression.

## Enable the live savings status line

Add this to your `~/.claude/settings.json` (the plugin ships the script):

```json
{
  "statusLine": {
    "type": "command",
    "command": "${CLAUDE_PLUGIN_ROOT}/statusline.sh"
  }
}
```

It renders, e.g.:

```
distil · 1.2M→0.5M tok · $3.41→$1.40 · eq 99.5% · 128 runs
```

(original → compressed tokens · original → compressed cost · decision-equivalence
when shadow-mode has samples · runs). With no savings yet it shows a hint instead.
Requires `distil` on `PATH` or `uvx` available.

**On a flat-rate subscription** (Claude Pro/Max) the per-token dollar figure is
notional — set `DISTIL_SUBSCRIPTION=1` in your environment to drop the cost and
show the token reduction only.

### Already have a status line?

Don't replace it — add distil as one more segment. In your existing status-line script:

```bash
distil_seg="$(distil statusline 2>/dev/null || true)"
[ -n "$distil_seg" ] && out="${out}  ·  ${distil_seg}"
```

`distil statusline` prints nothing when distil isn't installed, and `2>/dev/null || true`
keeps your line clean either way.

## Commands

| Command | What it does |
|---|---|
| `/distil` | Your savings report + how to route more traffic through distil |
| `/distil-stats` | Full breakdown — orig→compressed tokens, cost, runs, per-trajectory bars |
| `/distil-shadow` | Decision-equivalence: did compression preserve your agent's next action? |
| `/distil-dashboard` | Open the interactive HTML savings dashboard in your browser |
| `/distil-doctor` | Diagnose your setup — ledger, shadow validation, proxy round-trip, wiring |

Want a **live, refreshing view in your terminal**? Run the CLI directly (outside the
slash-command flow, e.g. in a split pane):

```bash
distil dashboard            # live TUI — token-trim + decision-equiv bars, Ctrl-C to exit
distil dashboard --once     # one static frame (pipe-friendly)
```

## Actually compress traffic

The status line reflects the local savings ledger. Populate it by routing an agent
through distil:

```
# Interactive + subscription/OAuth-safe (no tool injection, content un-digested):
distil wrap --lossless-only --verbatim -- claude

# Or a standalone proxy any SDK can point base_url at:
distil proxy                                   # Anthropic / OpenAI-compatible
distil proxy --upstream https://generativelanguage.googleapis.com   # Google Gemini
```

See the [distil docs](https://dshakes.github.io/distil) for the full story.
