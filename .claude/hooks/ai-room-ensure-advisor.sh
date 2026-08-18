#!/usr/bin/env bash
# Ensure the standing `advisor` peer is live for the derived channel.
#
# `advisor` is a Claude peer on the DEFAULT backend (no grok/sol/neural), so its
# model comes from .claude/agents/fable-advisor.md frontmatter. That is the only
# route -- ai_room_spawn_claude has no model parameter.
#
# Why this is a separate script rather than a row in ai-room-ensure-codex-roles.sh:
# at HEAD that file's loop calls _spawn_claudex_core, a codex-backend claudex
# spawn, and its own comments warn that path carries the wrong credentials for a
# Claude peer. The _spawn_claude_core form lives only in that file's uncommitted
# worktree conversion, so it is not something this slice can build on.
#
# Idempotent and NEVER evicting: if the handle is live, this exits without
# touching it. Fail-soft throughout: a SessionStart hook must never break
# session startup, so every failure path logs and exits 0. An absent advisor is
# recoverable; a broken startup is not.
#
# Concurrency: a NONBLOCKING kernel flock, keyed per channel, is held across the
# liveness check and the spawn. Kernel flock is required rather than an O_EXCL
# sentinel because flock ownership dies with the holding process -- a SIGKILL
# cannot leave the lock held and silently suppress every future ensure. The key
# is per-channel because handles and liveness are channel-scoped.
#
# Coupling note: this loads ~/.ai-room/mcp-server.py as a parity module and calls
# its internals -- init_room / ensure_room / _read_lease / _is_lease_live /
# _resolve_tmux_socket / _spawn_claude_core. It USES that tooling and never
# modifies it. If the internal API changes, update this script; it fails soft
# until then.
#
# It deliberately does NOT call _discover_live_codex_handles, which the sibling
# role hook uses. That helper screens leases through a codex NAME pattern that
# `advisor` can never match, so it reported the running peer as not-live and the
# hook respawned over it. See the A5 note below for the measurement.

set -uo pipefail

LOG="${AI_ROOM_ENSURE_ADVISOR_LOG:-/tmp/ai-room-ensure-advisor.log}"
exec >>"$LOG" 2>&1
echo "=== ensure-advisor $(date -u +%FT%TZ) ==="

# A2/A3. Byte-for-byte the derivation in ai-room-ensure-codex-roles.sh and
# ai-room-session-brief.sh. A second derivation would be a second source of truth.
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
if [ -n "${AI_ROOM_CHANNEL:-}" ]; then
  CHANNEL="$AI_ROOM_CHANNEL"
else
  CHANNEL=$(basename "$PROJECT_DIR" | sed -E 's/[^A-Za-z0-9_-]+/-/g; s/^[-_]+//; s/[-_]+$//; s/^(.{64}).*/\1/')
  [ -n "$CHANNEL" ] || CHANNEL="claw-code"
fi
echo "channel=$CHANNEL cwd=$PROJECT_DIR"

# A13. A session running AS the advisor must never ensure the advisor.
#
# Measured, not hypothesized: spawning advisor A ran A's own SessionStart, which
# reached this hook before A's lease was discoverable. The liveness check saw
# ['codex', 'codex_co_lead'] -- itself absent -- and spawned a duplicate advisor
# ~7.5s after the first. The flock does not help: the two ensures are sequential,
# not concurrent, so each acquires cleanly. Liveness alone cannot close this,
# because the racing session is precisely the one that is not yet live.
#
# The peer's own environment answers it directly, with no timing assumption.
if [ "${AI_ROOM_HANDLE:-}" = "advisor" ]; then
  echo "skip: this session IS the advisor (AI_ROOM_HANDLE=advisor); not self-spawning"
  echo "=== ensure-advisor done ==="
  exit 0
fi

AI_ROOM_CHANNEL="$CHANNEL" AI_ROOM_CWD="$PROJECT_DIR" python3 - <<'PY' || echo "WARN ensure-advisor block failed (non-fatal)"
import fcntl
import importlib.util
import os
import pathlib
import sys

HANDLE = "advisor"
AGENT = "fable-advisor"
SPAWN_TIMEOUT = float(os.environ.get("AI_ROOM_ADVISOR_SPAWN_TIMEOUT", "90"))
CHANNEL = os.environ["AI_ROOM_CHANNEL"]
CWD = os.environ["AI_ROOM_CWD"]


def log(msg):
    print(f"[ensure-advisor] {msg}", flush=True)


def load_parity_module():
    path = pathlib.Path.home() / ".ai-room" / "mcp-server.py"
    spec = importlib.util.spec_from_file_location("_ai_room_parity_advisor", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ai_room_parity_advisor"] = module
    spec.loader.exec_module(module)
    return module


lock_fd = None
try:
    channel_dir = pathlib.Path.home() / ".ai-room" / "channels" / CHANNEL
    channel_dir.mkdir(parents=True, exist_ok=True)
    lock_path = channel_dir / ".ensure-advisor.lock"

    # A10. Nonblocking, per-channel, held across liveness AND spawn, released by
    # FD close (including on SIGKILL -- that is the whole point of using flock).
    #
    # Two properties this depends on, both MEASURED rather than assumed:
    # - the FD must not survive into the spawned peer, or that peer would hold
    #   the lock for its lifetime and every later hook would see EWOULDBLOCK --
    #   the O_EXCL corpse again, wearing a live process. os.open returns a
    #   non-inheritable FD (PEP 446; os.get_inheritable -> False), and the lock
    #   file is absent from the running peer's /proc/<pid>/fd. Do not switch to
    #   a shell redirect or pass inheritable=True.
    # - flock degrades on some mounts. This path is under ~/.ai-room, on ext4;
    #   a second nonblocking acquire against the real path was observed to
    #   block. Moving the lock under /mnt/c would need that re-measured.
    lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log(f"another ensure holds {lock_path}; exiting without spawning")
        raise SystemExit(0)

    module = load_parity_module()
    module.init_room(channel=CHANNEL)
    module.ensure_room()

    # A5. Liveness via the canonical LEASE predicates, deliberately NOT via
    # _discover_live_codex_handles.
    #
    # Measured: with the advisor running and its lease heartbeat fresh, that
    # helper returned ['codex', 'codex_co_lead'] -- advisor absent. It filters
    # every lease through _is_auto_codex_handle, whose pattern is
    # `^codex(?:$|[_-].+$)` (config.py:141), which `advisor` can never match. It
    # is a NAME filter, not a liveness one, so for this handle it reports "not
    # live" unconditionally and the hook would respawn over a healthy peer on
    # every session. This is not a second freshness predicate: _read_lease and
    # _is_lease_live are the same primitives that helper calls internally.
    lease = module._read_lease(HANDLE)
    if lease is not None and module._is_lease_live(lease):
        log(f"skip: {HANDLE} lease live (owner_pid={lease.get('owner_pid')}, "
            f"heartbeat={lease.get('heartbeat')}); never evict -- "
            "recycle manually if wrong")
        raise SystemExit(0)

    # A6/A8. A stale lease is reported, not deleted: cleanup of a terminal
    # generation is Claude's evidenced action. This script only ever adds.
    if lease is not None:
        log(f"stale lease for {HANDLE} (owner_pid={lease.get('owner_pid')}, "
            f"heartbeat={lease.get('heartbeat')}) not removed; proceeding to spawn")

    # Same resolution + same error handling as the sibling role hook: a
    # resolution carrying `error` is discarded rather than passed through.
    socket_resolution = module._resolve_tmux_socket(CHANNEL)
    if getattr(socket_resolution, "error", None):
        log(f"tmux socket resolve error: {socket_resolution.error} / "
            f"{getattr(socket_resolution, 'error_detail', '')}")
        socket_resolution = None

    log(f"spawn {HANDLE} (agent={AGENT}, default backend, cwd={CWD})")
    result = module._spawn_claude_core(
        handle=HANDLE, agent=AGENT, allow_dangerous=True,
        neural=False, grok=False, sol=False,
        timeout_seconds=SPAWN_TIMEOUT, cwd=CWD,
        **({"tmux_socket": socket_resolution.tmux_socket,
            "collab_instance_id": socket_resolution.collab_instance_id}
           if socket_resolution is not None else {}),
    )
    log(f"{HANDLE}: ok={result.get('ok')} handle={result.get('handle')} "
        f"error={result.get('error')}")
except SystemExit:
    raise
except Exception as exc:  # noqa: BLE001
    # A11. Fail-soft. The lock releases via FD close on this path too, so a
    # spawn failure cannot wedge later ensures.
    log(f"non-fatal error (startup unaffected): {type(exc).__name__}: {exc}")
finally:
    if lock_fd is not None:
        os.close(lock_fd)
PY

echo "=== ensure-advisor done ==="
exit 0
