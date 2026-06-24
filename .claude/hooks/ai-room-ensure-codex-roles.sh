#!/usr/bin/env bash
# SessionStart helper: idempotently ensure the standing co-lead +
# worker lane are live for the derived channel — codex_co_lead via the
# proven ensure-co-lead CLI path, plus plan-dev (standing plan/review +
# bounded implementation lane) in THIS channel via lease-backed auto codex_N
# handles. test-operator is spawned on-demand as a haiku subagent, not
# SessionStart-ensured.
#
# Idempotent: a role whose live-lease codex_home basename already matches is
# skipped, so repeated session starts never stack duplicate workers.
# Fault-tolerant: ensure-co-lead and the role-spawn block are non-fatal; failures
# are logged and never break Claude startup. Backgrounded by .claude/settings.json.
# All output → the log; stdout stays clean.
#
# Role safety: plan-dev is spawned IDLE in this channel — a standing
# worker, NOT auto-dispatched work. plan-dev planning/review and bounded
# edits (require a persisted Claude `+1 implement`) still require an
# explicit Claude task/approval gate per .claude/rules/CLAUDEX_ORCHESTRATION.md.
# test-operator proof runs are on-demand haiku subagents (not SessionStart-
# ensured); full access is temp/log/artifact/tmux only, never source authority.
#
# Coupling note: the role-spawn block loads ~/.ai-room/mcp-server.py as a parity
# module and calls its claudex spawn internals (init_room / ensure_room /
# _discover_live_codex_handles / _resolve_tmux_socket / _spawn_claudex_core),
# mirroring ai_room_lib.mcp_bridge._ensure_co_lead_via_mcp. ~/.ai-room is codex's
# tooling — this hook USES it, never modifies it. If that internal API changes,
# update this block; it fails soft until then.

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

# 1. codex_co_lead via the proven ensure-co-lead CLI path (unchanged).
ai-room ensure-co-lead --channel "$CHANNEL" --cwd "$PROJECT_DIR" \
  || echo "WARN ensure-co-lead failed (non-fatal)"

# 2. Standing claudex lane (plan-dev plan/review/bounded implementation)
#    in THIS channel on PINNED handle codex so the role↔handle mapping stays
#    stable across sessions (codex_co_lead is pinned the same way by
#    ensure-co-lead). test-operator is on-demand haiku subagent spawn, not
#    SessionStart-ensured. Any other role is explicit-dispatch only. Skip a
#    role if already live (on any handle); log-and-skip if its pinned handle
#    is occupied by a different role (never evict).
AI_ROOM_CHANNEL="$CHANNEL" AI_ROOM_CWD="$PROJECT_DIR" python3 - <<'PY' || echo "WARN role-spawn block failed (non-fatal)"
import importlib.util, json, os, pathlib, sys

CHANNEL = os.environ["AI_ROOM_CHANNEL"]
CWD = os.environ["AI_ROOM_CWD"]
# (role, pinned_handle): stable mapping across sessions; handles must match
# the auto-codex pattern accepted by _spawn_claudex_core's collision guard.
ROLES = [
    ("plan-dev", "codex"),
]
SPAWN_TIMEOUT = 120.0


def log(msg):
    print(f"[role-spawn] {msg}", flush=True)


try:
    # Load ~/.ai-room/mcp-server.py as a parity module (mirrors
    # ai_room_lib.mcp_bridge._load_mcp_parity_module); exec'd as a named module
    # so the server main-loop does not run, only its functions are defined.
    mcp_path = pathlib.Path.home() / ".ai-room" / "mcp-server.py"
    spec = importlib.util.spec_from_file_location("ai_room_ensure_roles_parity", mcp_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ai_room_ensure_roles_parity"] = module
    spec.loader.exec_module(module)

    module.init_room(channel=CHANNEL)
    module.ensure_room()

    # Live roles = codex_home basenames of currently-live auto codex handles,
    # via the canonical liveness helper (not a duplicated freshness predicate).
    live_handles = module._discover_live_codex_handles(exclude_handle="")
    leases_dir = pathlib.Path.home() / ".ai-room" / "channels" / CHANNEL / "leases"
    live_roles = set()
    handle_role = {}
    for h in live_handles:
        try:
            d = json.loads((leases_dir / f"{h}.json").read_text())
            ch = d.get("codex_home")
            if ch:
                role_name = os.path.basename(ch.rstrip("/"))
                live_roles.add(role_name)
                handle_role[h] = role_name
        except Exception as e:
            log(f"lease read {h} failed: {e}")
    log(f"live_handles={live_handles} live_roles={sorted(live_roles)} handle_role={handle_role}")

    socket_resolution = module._resolve_tmux_socket(CHANNEL)
    if getattr(socket_resolution, "error", None):
        log(f"tmux socket resolve error: {socket_resolution.error} / {getattr(socket_resolution, 'error_detail', '')}")
        socket_resolution = None

    for role, pinned_handle in ROLES:
        if role in live_roles:
            holder = next((h for h, r in handle_role.items() if r == role), "?")
            log(f"skip {role} (already live on {holder})")
            continue
        occupant = handle_role.get(pinned_handle)
        if pinned_handle in live_handles and occupant != role:
            log(f"skip {role} (pinned handle {pinned_handle} occupied by {occupant!r}; never evict — recycle manually)")
            continue
        log(f"spawn {role} (handle={pinned_handle}, cwd={CWD})")
        kwargs = dict(handle=pinned_handle, cwd=CWD, timeout_seconds=SPAWN_TIMEOUT, role=role)
        if socket_resolution is not None:
            kwargs["tmux_socket"] = socket_resolution.tmux_socket
            kwargs["collab_instance_id"] = socket_resolution.collab_instance_id
        try:
            result = module._spawn_claudex_core(**kwargs)
            log(f"{role}: ok={result.get('ok')} handle={result.get('handle')} error={result.get('error')}")
        except Exception as e:
            log(f"spawn {role} raised: {e}")
except Exception as e:
    log(f"FATAL parity-block error (non-fatal to startup): {e}")
PY

echo "=== ensure-codex-roles done ==="
