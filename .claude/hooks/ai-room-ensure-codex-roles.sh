#!/usr/bin/env bash
# SessionStart helper: idempotently ensure the whole HRM room is live for the
# derived channel — codex_co_lead (proven ensure-co-lead path) plus the
# canonical Codex role workers training-dev / curriculum / audit.
#
# Idempotent: each SDK worker handle already live (per `ai-room sdk-agent list
# --json`) is skipped, so repeated session starts never stack duplicate workers.
# Fault-tolerant: ensure-co-lead and every spawn are non-fatal — failures are
# logged and never break Claude startup. Intended to be backgrounded by the
# SessionStart hook in .claude/settings.json. All output goes to the log; stdout
# stays clean.
#
# Role safety: training-dev is full-access (danger-full-access / approval never)
# but spawned IDLE — it is a standing worker, NOT auto-dispatched work. Mutating
# HRM work still requires an explicit Claude gate/task per
# .claude/rules/CLAUDEX_ORCHESTRATION.md. No --fresh-session: a restarted worker
# reuses its SDK state where appropriate (continuity over a cold reset).

set -uo pipefail

LOG=/tmp/ai-room-ensure-codex-roles.log
exec >>"$LOG" 2>&1
echo "=== ensure-codex-roles $(date -u +%FT%TZ) ==="

# Derive PROJECT_DIR / CHANNEL exactly like the SessionStart inline command and
# .claude/hooks/ai-room-session-brief.sh.
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
if [ -n "${AI_ROOM_CHANNEL:-}" ]; then
  CHANNEL="$AI_ROOM_CHANNEL"
else
  CHANNEL=$(basename "$PROJECT_DIR" | sed -E 's/[^A-Za-z0-9_-]+/-/g; s/^[-_]+//; s/[-_]+$//; s/^(.{64}).*/\1/')
  [ -n "$CHANNEL" ] || CHANNEL="claw-code"
fi
echo "channel=$CHANNEL cwd=$PROJECT_DIR"

# 1. codex_co_lead via the proven ensure-co-lead path (unchanged behavior).
ai-room ensure-co-lead --channel "$CHANNEL" --cwd "$PROJECT_DIR" \
  || echo "WARN ensure-co-lead failed (non-fatal)"

# 2. Canonical Codex role workers — handle|role|sandbox triples.
#    approval-policy=never for all; training-dev is full-access but idle.
ROLES="training-dev|training-dev|danger-full-access
curriculum|curriculum|read-only
audit|audit|read-only"

# Snapshot live SDK codex handles once (idempotency: skip-if-live).
LIVE=$(ai-room sdk-agent list --provider codex --channel "$CHANNEL" --json 2>/dev/null \
  | python3 -c 'import json, sys
try:
    d = json.load(sys.stdin)
    print("\n".join(w.get("handle", "") for w in d.get("workers", [])))
except Exception:
    pass' || true)
echo "live SDK codex handles: $(echo "$LIVE" | tr "\n" " ")"

while IFS='|' read -r HANDLE ROLE SANDBOX; do
  [ -n "$HANDLE" ] || continue
  if printf '%s\n' "$LIVE" | grep -qx "$HANDLE"; then
    echo "skip $HANDLE (already live)"
    continue
  fi
  echo "spawn $HANDLE role=$ROLE sandbox=$SANDBOX"
  ai-room sdk-agent spawn \
    --provider codex --channel "$CHANNEL" --cwd "$PROJECT_DIR" \
    --handle "$HANDLE" --role "$ROLE" \
    --sandbox "$SANDBOX" --approval-policy never \
    --start-cursor tail --wake-mode fifo \
    --max-messages 1 --message-retry-limit 2 \
    --error-to claude \
    || echo "WARN spawn $HANDLE failed (non-fatal)"
done <<EOF
$ROLES
EOF

echo "=== ensure-codex-roles done ==="
