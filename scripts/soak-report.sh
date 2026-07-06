#!/usr/bin/env bash
# Soak report: per-session routing outcomes for the rc under soak.
#
# Reads the session traffic markers (written by wrap ≥1.12.0rc1) and the
# savings ledger, prints one line per session — routed / bypassed / too-young —
# and the aggregate bypass rate. For any LIVE bypassed session it also captures
# the diagnostic evidence we need to root-cause the trigger (claude version,
# where the agent's TCP connections actually go), since a bypassed session is
# undiagnosable post-mortem.
#
# Usage: scripts/soak-report.sh          # human report
set -euo pipefail

HOME_DIR="${DISTIL_HOME:-$HOME/.distil}"
LEDGER="$HOME_DIR/savings.jsonl"
GRACE=180
now=$(date +%s)

shopt -s nullglob
markers=("$HOME_DIR"/sessions/*)
if [ ${#markers[@]} -eq 0 ]; then
  echo "no session markers yet — markers appear once wraps run ≥1.12.0rc1 (restart wraps after upgrading)"
  exit 0
fi

total=0 bypassed=0 young=0
for m in "${markers[@]}"; do
  sid=$(basename "$m")
  val=$(cat "$m" 2>/dev/null || echo "?")
  mtime=$(stat -f %m "$m" 2>/dev/null || stat -c %Y "$m")
  age=$((now - mtime))
  rows=0
  [ -f "$LEDGER" ] && rows=$(grep -c "\"$sid\"" "$LEDGER" || true)
  # A wrap session is alive if its pid (sid = s<ts>-<pid>) still runs.
  pid="${sid##*-}"
  alive=""
  kill -0 "$pid" 2>/dev/null && alive=" [wrap alive]"
  total=$((total + 1))
  if [ "$val" = "1" ] || [ "$rows" -gt 0 ]; then
    echo "ROUTED    $sid  ledger_rows=$rows$alive"
  elif [ "$age" -lt "$GRACE" ]; then
    young=$((young + 1))
    echo "TOO-YOUNG $sid  age=${age}s (grace ${GRACE}s)$alive"
  else
    bypassed=$((bypassed + 1))
    echo "BYPASSED  $sid  age=${age}s ledger_rows=0$alive"
    if [ -n "$alive" ]; then
      # Live specimen — capture the evidence that dies with the process.
      cap="$HOME_DIR/bypass-capture-$sid.txt"
      {
        echo "captured: $(date -u +%FT%TZ)"
        echo "claude --version: $(claude --version 2>/dev/null || echo 'claude not on PATH')"
        echo "distil --version: $(distil --version 2>/dev/null || echo '?')"
        agent=$(pgrep -P "$pid" | head -1 || true)
        echo "wrap pid $pid, agent pid ${agent:-none}"
        if [ -n "${agent:-}" ]; then
          echo "--- agent TCP connections (where traffic actually goes) ---"
          lsof -nP -iTCP -a -p "$agent" 2>/dev/null || echo "lsof unavailable"
        fi
      } > "$cap"
      echo "          evidence captured → $cap (attach this when reporting)"
    fi
  fi
done

echo "---"
echo "sessions=$total bypassed=$bypassed too_young=$young  bypass_rate=$bypassed/$((total - young))"
