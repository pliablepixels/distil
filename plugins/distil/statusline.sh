#!/usr/bin/env bash
# Distil status line — renders genuine cumulative savings from the local ledger.
#
# Wire it into ~/.claude/settings.json:
#   { "statusLine": { "type": "command",
#                     "command": "${CLAUDE_PLUGIN_ROOT}/statusline.sh" } }
#
# Claude Code pipes the status-line JSON on stdin; we pass it straight through to
# `distil statusline`, which reads the ledger and prints one line. The command is
# resolved however distil is installed (PATH, then uvx), and degrades gracefully.
set -euo pipefail

if command -v distil >/dev/null 2>&1; then
  exec distil statusline
elif command -v uvx >/dev/null 2>&1; then
  exec uvx --from distil-llm distil statusline
else
  printf 'distil · not installed — pipx install distil-llm'
fi
