#!/usr/bin/env bash
# PreToolUse hook: enforce bin/watch-wrap usage on Monitor calls.
#
# Blocks raw `tail -f | grep` patterns; allows poll-loops and any
# command already using bin/watch-wrap. See bin/watch-wrap for the
# wrapper this enforces — adds exit-code, heartbeat backoff,
# coalesce, categories, replay, and --stop-on.
#
# Hook input: JSON on stdin, {"tool_name": "Monitor", "tool_input": {"command": "..."}}
# Decision: stdout JSON with permissionDecision allow/deny.
# Silent allow (exit 0, no output) for commands that don't match tail -f/F.

set -euo pipefail

# Extract tool_input.command from stdin JSON using python3 (jq may not
# be installed; python3 is stdlib everywhere).
cmd=$(python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get("tool_input", {}).get("command", ""))
except Exception:
    print("")
')

# Fast path: if no tail -f/-F, not our concern
if ! echo "$cmd" | grep -qE 'tail[[:space:]]+-[fF]'; then
    exit 0
fi

# Already using watch-wrap? allow
if echo "$cmd" | grep -q 'watch-wrap'; then
    exit 0
fi

# Poll-loop pattern (while true; ... sleep ...; done) — allow.
# These aren't use cases for watch-wrap; they produce per-poll events.
if echo "$cmd" | grep -qE 'while[[:space:]]+true.*sleep'; then
    exit 0
fi

# Block with actionable message
cat <<'EOF'
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "Raw `tail -f | grep` for Monitor is blocked — use bin/watch-wrap instead. It adds: (1) exit-code event when the process dies [distinguishes crash / clean completion / timeout], (2) heartbeat with exponential backoff on silent processes, (3) coalesce window to group bursty output, (4) --error/--progress/--success category tagging, (5) replay of last N lines on completion, (6) --stop-on REGEX for auto-exit on completion markers. Replace `tail -F /path | grep 'pattern'` with `bin/watch-wrap --log /path [--pid N] --error 'pattern|Traceback|OOM' --progress '...' --stop-on 'DONE$' --heartbeat 180 --replay 20`. Full docs: bin/watch-wrap --help."
  }
}
EOF
