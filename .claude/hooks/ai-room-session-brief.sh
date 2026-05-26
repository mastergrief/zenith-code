#!/usr/bin/env bash
# SessionStart hook: emit a brief of ai-room state for the derived channel
# as additionalContext to claude. Auto-closes trivial closures before
# collecting read-only signals (advances the claude cursor for already-
# resolved threads via `ai-room resume-close-trivial`). Fault-tolerant —
# every CLI call falls back to a sensible default on error.
#
# Output: a single line of JSON conforming to Claude Code's hookSpecificOutput
# schema with additionalContext populated. Empty/null context emits nothing.
#
# Ported from zenith-fitness with channel/charter adaptations for claw-code.

set -uo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
if [ -n "${AI_ROOM_CHANNEL:-}" ]; then
  CHANNEL="$AI_ROOM_CHANNEL"
else
  # Mirror ai-room/collab channel mapping: sanitize PROJECT_DIR basename,
  # with AI_ROOM_CHANNEL available for repo names needing an explicit label.
  CHANNEL=$(basename "$PROJECT_DIR" | sed -E 's/[^A-Za-z0-9_-]+/-/g; s/^[-_]+//; s/[-_]+$//; s/^(.{64}).*/\1/')
  [ -n "$CHANNEL" ] || CHANNEL="claw-code"
fi
HANDLE="claude"

# --- auto-close trivial closures (state-changing; runs before signal collection) ---

# Auto-close structural pending-response replies with positive-closure body
# classifier + plain inbox acks so PEEK/RESUME below report post-close state.
# stdout/stderr both redirected to log so the hook's JSON payload on stdout
# stays uncorrupted.
ai-room --channel "$CHANNEL" resume-close-trivial --for "$HANDLE" \
  >>/tmp/ai-room-session-brief.log 2>&1 || true

# --- collect signals (all fault-tolerant, read-only post-close) ---

# Inbox unread: extract leading count from peek output ("N new inbox message(s)...")
PEEK_RAW=$(ai-room --channel "$CHANNEL" peek "$HANDLE" 2>/dev/null | head -1 || echo "0")
PEEK=$(echo "$PEEK_RAW" | grep -oE '^[0-9]+' | head -1)
[ -z "$PEEK" ] && PEEK="0"

RESUME=$(ai-room --channel "$CHANNEL" resume-check --for "$HANDLE" 2>/dev/null | head -1 | sed 's/ *$//' || echo "(resume_check unavailable)")
[ -z "$RESUME" ] && RESUME="(empty)"

# Live codex handles: scan lease files, filter by heartbeat freshness (60s window)
LEASE_DIR="$HOME/.ai-room/channels/$CHANNEL/leases"
LIVE_HANDLES=$(python3 - "$LEASE_DIR" <<'PY'
import json, os, sys, time, glob
from datetime import datetime, timezone
lease_dir = sys.argv[1]
fresh_window_secs = 90  # heartbeat fresher than this counts as live
now = datetime.now(timezone.utc).timestamp()
live = []
for path in sorted(glob.glob(os.path.join(lease_dir, "codex*.json"))):
    try:
        with open(path) as f:
            data = json.load(f)
        hb = data.get("heartbeat")
        if not hb:
            continue
        hb_ts = datetime.fromisoformat(hb.replace("Z", "+00:00")).timestamp()
        if now - hb_ts <= fresh_window_secs:
            live.append(data.get("handle") or os.path.splitext(os.path.basename(path))[0])
    except Exception:
        continue
print(", ".join(live) if live else "(none live; codex peer may still be spawning)")
PY
)

# Live claudex roles: handle:role (role = codex_home basename) from fresh leases.
# These are the lease-backed claudex workers (co_lead + training-dev/curriculum/
# audit). Fault-tolerant.
CLAUDEX_ROLES=$(python3 - "$LEASE_DIR" <<'PY'
import json, os, sys, glob
from datetime import datetime, timezone
lease_dir = sys.argv[1]
fresh_window_secs = 90
now = datetime.now(timezone.utc).timestamp()
out = []
for path in sorted(glob.glob(os.path.join(lease_dir, "codex*.json"))):
    try:
        with open(path) as f:
            data = json.load(f)
        hb = data.get("heartbeat")
        if not hb:
            continue
        if now - datetime.fromisoformat(hb.replace("Z", "+00:00")).timestamp() > fresh_window_secs:
            continue
        handle = data.get("handle") or os.path.splitext(os.path.basename(path))[0]
        ch = data.get("codex_home") or ""
        role = os.path.basename(ch.rstrip("/")) if ch else "?"
        out.append(f"{handle}:{role}")
    except Exception:
        continue
print(", ".join(out) if out else "(none live)")
PY
)
[ -z "$CLAUDEX_ROLES" ] && CLAUDEX_ROLES="(none live)"

# Open tasks owned by claude: short list of subjects
OPEN_TASKS=$(ai-room --channel "$CHANNEL" task list --owner "$HANDLE" --status in_progress 2>/dev/null | head -5 || true)
[ -z "$OPEN_TASKS" ] && OPEN_TASKS="(none)"

# --- compose the brief ---

BRIEF=$(cat <<EOF
[ai-room session brief — channel $CHANNEL]
Inbox unread: $PEEK
Resume directive: $RESUME
Live codex handles: $LIVE_HANDLES
Live claudex roles: $CLAUDEX_ROLES
Open tasks owned by claude (in_progress):
$OPEN_TASKS
Charter: .claude/rules/AI_ROOM_COLLAB.md §"Role" + §"Coordination channel" + §"Before declaring idle — resume_check". Two-session collab via ai_room_* MCP tools, NOT subagent spawning.
EOF
)

# --- emit as hookSpecificOutput JSON ---

python3 -c "
import json, sys
ctx = sys.stdin.read().strip()
if not ctx:
    sys.exit(0)
print(json.dumps({
    'hookSpecificOutput': {
        'hookEventName': 'SessionStart',
        'additionalContext': ctx,
    }
}))
" <<< "$BRIEF"
