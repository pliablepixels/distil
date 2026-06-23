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

This gives you the **`/distil`** command (savings report + setup help).

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
distil · 1.2M tok · $3.4120 · 128 runs · eq 99.5%
```

(tokens saved · dollars saved · runs · live decision-equivalence when shadow-mode
has samples). With no savings yet it shows a hint instead. Requires `distil`
on `PATH` or `uvx` available.

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
