#!/usr/bin/env bash
# SessionStart helper: idempotently ensure the whole HRM room is live for the
# derived channel — codex_co_lead (proven ensure-co-lead CLI path) plus the
# canonical claudex role workers training-dev / curriculum / audit, spawned in
# THIS channel as lease-backed auto codex_N handles via the claudex role loader
# (the standing-room model — not the prototype unified-worker path).
#
# Idempotent: a role whose live-lease codex_home basename already matches is
# skipped, so repeated session starts never stack duplicate workers.
# Fault-tolerant: ensure-co-lead and the role-spawn block are non-fatal; failures
# are logged and never break Claude startup. Backgrounded by .claude/settings.json.
# All output → the log; stdout stays clean.
#
# Role safety: training-dev is full-access but spawned IDLE in this channel — a
# standing worker, NOT auto-dispatched work. Mutating HRM work still requires an
# explicit Claude task/approval gate per .claude/rules/CLAUDEX_ORCHESTRATION.md.
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

# 2. Canonical claudex role workers (training-dev / curriculum / audit) in THIS
#    channel as auto codex_N handles. Skip any role already live.
AI_ROOM_CHANNEL="$CHANNEL" AI_ROOM_CWD="$PROJECT_DIR" python3 - <<'PY' || echo "WARN role-spawn block failed (non-fatal)"
import importlib.util, json, os, pathlib, sys

CHANNEL = os.environ["AI_ROOM_CHANNEL"]
CWD = os.environ["AI_ROOM_CWD"]
ROLES = ["training-dev", "curriculum", "audit"]
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
    for h in live_handles:
        try:
            d = json.loads((leases_dir / f"{h}.json").read_text())
            ch = d.get("codex_home")
            if ch:
                live_roles.add(os.path.basename(ch.rstrip("/")))
        except Exception as e:
            log(f"lease read {h} failed: {e}")
    log(f"live_handles={live_handles} live_roles={sorted(live_roles)}")

    socket_resolution = module._resolve_tmux_socket(CHANNEL)
    if getattr(socket_resolution, "error", None):
        log(f"tmux socket resolve error: {socket_resolution.error} / {getattr(socket_resolution, 'error_detail', '')}")
        socket_resolution = None

    for role in ROLES:
        if role in live_roles:
            log(f"skip {role} (already live)")
            continue
        log(f"spawn {role} (handle=auto, cwd={CWD})")
        kwargs = dict(handle=None, cwd=CWD, timeout_seconds=SPAWN_TIMEOUT, role=role)
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
