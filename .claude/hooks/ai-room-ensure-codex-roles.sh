#!/usr/bin/env bash
# SessionStart helper: idempotently ensure the standing co-lead +
# worker lanes are live for the derived channel — codex_co_lead as a
# SOL-BACKEND CLAUDE PEER (spawn_claude sol=true, agent co-lead; gabe-directed,
# NOT the legacy codex-backend ensure-co-lead CLI), plus plan-dev as a
# DEFAULT-BACKEND (Opus) CLAUDE PEER (spawn_claude, agent plan-dev; no
# subagents — gabe-directed, NOT the legacy codex-backend claudex spawn) on
# the pinned handle "codex", plus gate1_audit as a GROK-BACKEND CLAUDE PEER
# (spawn_claude grok=true, agent gate1-auditor; grok via cliproxy in Claude
# Code — gabe-directed gate-1 split, advisor topology ruling) on the pinned
# handle "gate1_audit". test-operator is NOT a peer and NOT a subagent: Claude
# carries it directly, so there is nothing here to ensure for it.
#
# Idempotent: a role whose PINNED HANDLE is already live is skipped, so repeated
# session starts never stack duplicate workers. Two further protections, both
# copied from the retired sibling ai-room-ensure-advisor.sh where they were measured:
#
#   - self-spawn guard (A13 there): a session running AS a managed peer must
#     never ensure ITSELF. A spawned peer runs its own SessionStart, which
#     reaches this hook before its lease is discoverable; the liveness check
#     then sees itself absent and spawns a duplicate. Measured there at ~7.5s
#     after the first. Liveness alone cannot close it, because the racing
#     session is precisely the one that is not yet live. AI_ROOM_HANDLE answers
#     it directly with no timing assumption. Per-handle rather than a whole-file
#     exit, so a peer that is one of these lanes still ensures the OTHERS.
#   - nonblocking per-channel flock: held across the liveness check AND the
#     spawns, so two near-simultaneous session starts cannot both see "not live"
#     and both spawn. Observed firing twice in 20 runs on the advisor's own lock.
# Fault-tolerant: ensure-co-lead and the role-spawn block are non-fatal; failures
# are logged and never break Claude startup. Backgrounded by .claude/settings.json.
# All output → the log; stdout stays clean.
#
# Role safety: plan-dev is spawned IDLE in this channel — a standing
# worker, NOT auto-dispatched work. plan-dev planning/review and bounded
# edits (require a persisted Claude `+1 implement`) still require an
# explicit Claude task/approval gate per .claude/rules/CLAUDEX_ORCHESTRATION.md.
# plan-dev spawns no subagents: it performs every edit, validation run, and
# receipt itself. test-operator proof runs are Claude-carried, not spawned here.
#
# Coupling note: the role-spawn block loads ~/.ai-room/mcp-server.py as a parity
# module and calls its spawn internals (init_room / ensure_room /
# _discover_live_codex_handles / _resolve_tmux_socket / _spawn_claude_core),
# mirroring ai_room_lib.mcp_bridge._ensure_co_lead_via_mcp. ~/.ai-room is codex's
# tooling — this hook USES it, never modifies it. If that internal API changes,
# update this block; it fails soft until then.

set -uo pipefail

LOG="${AI_ROOM_ENSURE_CODEX_ROLES_LOG:-/tmp/ai-room-ensure-codex-roles.log}"
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

# 1. codex_co_lead is ensured inside the parity block below as a sol-backend
#    Claude peer (spawn_claude sol=true, agent co-lead) — the legacy
#    `ai-room ensure-co-lead` CLI spawned a codex-backend claudex on the same
#    handle (wrong credentials) and must NOT be called here.

# 2. Standing plan-dev lane (plan/review/bounded implementation) as a
#    default-backend (Opus) Claude peer (spawn_claude, agent plan-dev) on
#    PINNED handle codex so the role↔handle mapping stays stable across
#    sessions (codex_co_lead is pinned the same way). test-operator is
#    Claude-carried, not a spawnable role. Any other role is
#    explicit-dispatch only. Skip if the handle is already live
#    (never evict; recycle manually if the backend is wrong).
AI_ROOM_CHANNEL="$CHANNEL" AI_ROOM_CWD="$PROJECT_DIR" \
AI_ROOM_SELF_HANDLE="${AI_ROOM_HANDLE:-}" \
python3 - <<'PY' || echo "WARN role-spawn block failed (non-fatal)"
import fcntl, importlib.util, json, os, pathlib, sys

CHANNEL = os.environ["AI_ROOM_CHANNEL"]
CWD = os.environ["AI_ROOM_CWD"]
# Self-spawn guard: the handle this session IS, or "" for the lead / a
# non-room session. Never spawn over ourselves — see the header note.
SELF_HANDLE = os.environ.get("AI_ROOM_SELF_HANDLE", "").strip()
# (role/agent, pinned_handle, grok): standing Claude-peer lanes spawned via
# _spawn_claude_core; stable role↔handle mapping across sessions.
CLAUDE_PEER_ROLES = [
    # orchestrator on the pinned handle "claude": grok via cliproxy. Owns room
    # orchestration, dispatch, gate framing, +1 records, test-operator runs.
    # The interactive Fable session Gabe drives holds handle "advisor"
    # (.mcp.json AI_ROOM_HANDLE default) and is NOT spawned here.
    ("orchestrator", "claude", True),
    # plan-dev: default backend (Opus), no subagents. Gabe-directed.
    ("plan-dev", "codex", False),
    # gate-1 verification+freeze auditor: grok via cliproxy in Claude Code
    # (the proxy env overrides the agent frontmatter model pin). Lane: verify
    # + freeze + external verdict ONLY.
    ("gate1-auditor", "gate1_audit", True),
]
SPAWN_TIMEOUT = 120.0


def log(msg):
    print(f"[role-spawn] {msg}", flush=True)


lock_fd = None
try:
    # Nonblocking, per-channel, acquired BEFORE the liveness discovery and
    # released by FD close, so it is held across liveness AND every spawn.
    # Kernel flock rather than an O_EXCL sentinel: flock ownership dies with the
    # holding process, so a SIGKILL cannot leave it held and silently suppress
    # every future ensure. os.open returns a non-inheritable FD (PEP 446) --
    # load-bearing, since an inherited FD would survive into the spawned peer
    # and hold the lock for that peer's whole lifetime. Do not switch to a shell
    # redirect or pass inheritable=True.
    channel_dir = pathlib.Path.home() / ".ai-room" / "channels" / CHANNEL
    channel_dir.mkdir(parents=True, exist_ok=True)
    lock_path = channel_dir / ".ensure-codex-roles.lock"
    lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log(f"another ensure holds {lock_path}; exiting without spawning")
        raise SystemExit(0)

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

    # Extracted verbatim between the markers below by
    # tests/test_ai_room_ensure_codex_roles_v1.py; keep both markers on their
    # own lines so the extracted text is valid Python as-is.
    # --- BEGIN _handle_is_live
    def _handle_is_live(h, module, live_handles, log):
        """Liveness for a PINNED handle.

        _discover_live_codex_handles filters to AUTO codex handles only
        (^codex$ or ^codex[_-]...), so non-auto pinned handles such as
        "claude" and "gate1_audit" can never appear in it. A membership test
        against that list therefore reads them as dead on every SessionStart
        and spawns a duplicate. Auto handles keep the canonical list; non-auto
        pinned handles resolve through the SAME lease predicates that list
        uses internally, so there is no second freshness rule.

        An unreadable or absent lease reads as NOT live, preserving the prior
        default that a missing peer gets spawned; a stuck-live reading would
        suppress every future spawn silently.
        """
        if module._is_auto_codex_handle(h):
            return h in live_handles
        try:
            lease = module._read_lease(h)
        except Exception as e:
            log(f"lease read {h} failed: {e}; treating as not live")
            return False
        if not lease:
            return False
        return bool(
            module._is_lease_live(lease)
            and module._lease_registry_metadata_live_if_present(lease)
        )
    # --- END _handle_is_live

    socket_resolution = module._resolve_tmux_socket(CHANNEL)
    if getattr(socket_resolution, "error", None):
        log(f"tmux socket resolve error: {socket_resolution.error} / {getattr(socket_resolution, 'error_detail', '')}")
        socket_resolution = None

    # codex_co_lead: sol-backend Claude peer (gabe-directed credentials —
    # spawn_claude sol=true, agent co-lead). Skip if the handle is live on
    # any backend (never evict; recycle manually if the backend is wrong).
    CO_LEAD_HANDLE = "codex_co_lead"
    if SELF_HANDLE == CO_LEAD_HANDLE:
        log(f"skip co-lead (this session IS {CO_LEAD_HANDLE}; not self-spawning)")
    elif _handle_is_live(CO_LEAD_HANDLE, module, live_handles, log):
        log(f"skip co-lead (handle {CO_LEAD_HANDLE} already live)")
    else:
        log(f"spawn co-lead (handle={CO_LEAD_HANDLE}, agent=co-lead, sol=true, cwd={CWD})")
        try:
            result = module._spawn_claude_core(
                handle=CO_LEAD_HANDLE, agent="co-lead", allow_dangerous=True,
                neural=False, sol=True, timeout_seconds=SPAWN_TIMEOUT, cwd=CWD,
                **({"tmux_socket": socket_resolution.tmux_socket,
                    "collab_instance_id": socket_resolution.collab_instance_id}
                   if socket_resolution is not None else {}),
            )
            log(f"co-lead: ok={result.get('ok')} handle={result.get('handle')} sol={result.get('sol')} error={result.get('error')}")
        except Exception as e:
            log(f"spawn co-lead raised: {e}")

    for role, pinned_handle, grok in CLAUDE_PEER_ROLES:
        if SELF_HANDLE == pinned_handle:
            log(f"skip {role} (this session IS {pinned_handle}; not self-spawning)")
            continue
        if _handle_is_live(pinned_handle, module, live_handles, log):
            occupant = handle_role.get(pinned_handle)
            log(f"skip {role} (handle {pinned_handle} already live"
                + (f", claudex role {occupant!r}" if occupant else "")
                + "; never evict — recycle manually if backend is wrong)")
            continue
        log(f"spawn {role} (handle={pinned_handle}, agent={role}, grok={grok}, cwd={CWD})")
        try:
            result = module._spawn_claude_core(
                handle=pinned_handle, agent=role, allow_dangerous=True,
                neural=False, grok=grok, timeout_seconds=SPAWN_TIMEOUT, cwd=CWD,
                **({"tmux_socket": socket_resolution.tmux_socket,
                    "collab_instance_id": socket_resolution.collab_instance_id}
                   if socket_resolution is not None else {}),
            )
            log(f"{role}: ok={result.get('ok')} handle={result.get('handle')} grok={result.get('grok')} error={result.get('error')}")
        except Exception as e:
            log(f"spawn {role} raised: {e}")
except SystemExit:
    raise
except Exception as e:
    log(f"FATAL parity-block error (non-fatal to startup): {e}")
finally:
    # Releases the flock; also runs on the spawn-failure path, so a failure
    # cannot wedge later ensures.
    if lock_fd is not None:
        os.close(lock_fd)
PY

echo "=== ensure-codex-roles done ==="

# 3. `advisor` is no longer a spawned peer: it is the interactive Fable session
#    Gabe drives (handle via .mcp.json AI_ROOM_HANDLE default). Nothing to ensure.
