#!/usr/bin/env bash
# PreToolUse hook: block background-shell launch patterns from Bash and
# redirect to Monitor.
#
# Enforce Monitor for long-running shell observation. Catches the
# workflow drift where detached Bash launches or run_in_background=true
# hide work from Monitor + bin/watch-wrap. Companion to
# enforce-watch-wrap.sh, which catches the inverse (raw tail -f on
# Monitor).
#
# Hook input: JSON on stdin, {"tool_name": "Bash", "tool_input": {"command": "...", "run_in_background": false|true}}
# Decision: stdout JSON with permissionDecision allow/deny.
# Silent allow (exit 0, no output) for commands that don't match.
#
# Blocks when ANY of:
#   1. tool_input.run_in_background == true
#   2. command invokes `setsid` as a shell command (background launch)
#   3. command invokes `nohup` as a shell command (background launch)
#   4. command invokes `disown` (background launch)
#   5. command uses `&` as a control operator (background launch)
#   6. command body is a poll loop — ANY loop header whose body sleeps,
#      `until` and `while` alike
#
# 4-6 close gaps measured against workflow.md:150, which forbids five
# patterns where this hook previously enforced three: `disown` and a
# trailing `&` both passed. 6 replaces an `until`-keyword match that let
# the identical `while true; do ...; sleep N; done` through — a matcher
# keyed on a spelling rather than on the property (a loop that sleeps).
#
# Allowed: foreground bash, `cd` / `find` / one-off git ops, and any
# command carrying these tokens inside quoted arguments — checks 4-6 run
# against a quote-stripped copy so `grep 'while true' f` is not a launch.
# An over-firing deny wedges ordinary work, so the allow side is covered
# by explicit controls in test_enforce_monitor_on_bg_shell.py.

set -euo pipefail

# Read entire stdin (hook JSON) into a variable BEFORE invoking python,
# so the python heredoc doesn't shadow stdin.
input_json="$(cat)"

INPUT_JSON="$input_json" python3 <<'PYEOF'
import json
import os
import re
import sys

raw = os.environ.get("INPUT_JSON", "")
if not raw:
    sys.exit(0)  # fail-open on empty input
try:
    data = json.loads(raw)
except Exception:
    sys.exit(0)  # fail-open on bad JSON

if not isinstance(data, dict):
    sys.exit(0)  # fail-open on malformed hook shape

ti = data.get("tool_input", {}) or {}
if not isinstance(ti, dict):
    ti = {}

cmd = ti.get("command", "")
if not isinstance(cmd, str):
    cmd = ""
bg = ti.get("run_in_background") is True

block_reason = None

if bg:
    block_reason = "Bash run_in_background=true"

# setsid/nohup as shell commands, not as ordinary arguments to grep/rg/docs.
if block_reason is None and re.search(r"(?:^|[;&|()\n\r])\s*setsid(?:\s|$)", cmd):
    block_reason = "setsid invocation"

if block_reason is None and re.search(r"(?:^|[;&|()\n\r])\s*nohup(?:\s|$)", cmd):
    block_reason = "nohup invocation"

# Checks below run against a quote-stripped copy: `disown`, `&` and the loop
# keywords all appear routinely inside ordinary string arguments
# (`rg 'disown' docs/`, `grep 'while true' f`), and matching those would
# wedge ordinary work. setsid/nohup above keep the raw string because their
# separator anchor already excludes the quoted case.
unquoted = re.sub(r"'[^']*'|\"[^\"]*\"", " ", cmd)

if block_reason is None and re.search(r"(?:^|[;&|()\n\r])\s*disown(?:\s|$)", unquoted):
    block_reason = "disown invocation"

# `&` as a control operator. Excluded: `&&`, and the fd-redirection forms
# `2>&1` / `>&2` / `&> log`, where `&` is part of the redirection token.
if block_reason is None and re.search(r"(?<![&><])&(?![&>])", unquoted):
    block_reason = "background & launch"

# Poll loop by PROPERTY — a loop header whose body sleeps — so `until` and
# `while` are treated alike. [\s\S] rather than `.` so a multi-line command
# cannot slip past on a newline.
if block_reason is None and re.search(
    r"\b(?:until|while)\b[\s\S]*?\bdo\b[\s\S]*?\bsleep\s", unquoted
):
    block_reason = "poll-sleep loop"

if block_reason is None:
    sys.exit(0)  # silent allow

reason = (
    f"Background-shell pattern blocked (matched: {block_reason}). "
    "Per shell_monitor.md + workflow.md GPU bench discipline, long-running "
    "shell observation must go through Monitor + bin/watch-wrap, not Bash "
    "background mode. Use this pattern instead: (a) run the job in a "
    "dedicated foreground shell/session that writes to a log file (no "
    "setsid, nohup, disown, trailing &, or run_in_background=true); (b) "
    "immediately arm Monitor on that log via "
    "bin/watch-wrap with --error / --progress / --success / --stop-on "
    "filters and a heartbeat. For one-shot 'wait until file appears', use "
    "Monitor with --stop-on instead of a Bash poll-sleep loop — `while` "
    "and `until` forms are both blocked. "
    "Hook: .claude/hooks/enforce-monitor-on-bg-shell.sh."
)

out = {
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }
}
json.dump(out, sys.stdout)
sys.stdout.write("\n")
PYEOF
