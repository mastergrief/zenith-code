#!/usr/bin/env bash
# Single consolidated SessionStart hook (hooks hygiene: one entry point).
# Children (each self-gates on the payload's `source` where relevant):
#   1. ai-room-ensure-codex-roles.sh       — backgrounded fire-and-forget, no output
#   2. ai-room-session-brief.sh            — always fires
#   3. post-compact-preserve-state.py      — persists compaction summaries to
#      MEMORY/minutes/ (rewired here after being orphaned from settings.json)
#   4. auto-research-resume-directive.py   — startup/resume charter + loop seed,
#      compact standing directive (absorbed post-compact-directive.py)
# Each child emits hookSpecificOutput JSON (or nothing); this wrapper merges
# their additionalContext strings into ONE hookSpecificOutput payload.
# Fault-tolerant: a failing child contributes nothing and never breaks startup.
# Children are invoked through their interpreter, never through their exec bit:
# `core.filemode=false` here, so a child committed 100644 would exec-fail (126)
# and the `|| true` would swallow it silently in a fresh checkout.

set -uo pipefail

HOOKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$(cat 2>/dev/null || true)"

( bash "$HOOKS_DIR/ai-room-ensure-codex-roles.sh" >/dev/null 2>&1 & )

{
  printf '%s' "$PAYLOAD" | bash "$HOOKS_DIR/ai-room-session-brief.sh" 2>/dev/null || true
  printf '%s' "$PAYLOAD" | python3 "$HOOKS_DIR/post-compact-preserve-state.py" 2>/dev/null || true
  printf '%s' "$PAYLOAD" | python3 "$HOOKS_DIR/auto-research-resume-directive.py" 2>/dev/null || true
} | python3 -c "
import json, sys

parts = []
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        ctx = json.loads(line)['hookSpecificOutput']['additionalContext']
    except Exception:
        ctx = line  # tolerate plain-text emitters
    if ctx:
        parts.append(ctx)

if parts:
    print(json.dumps({
        'hookSpecificOutput': {
            'hookEventName': 'SessionStart',
            'additionalContext': '\n'.join(parts),
        }
    }))
"
