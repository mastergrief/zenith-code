#!/usr/bin/env python3
"""Re-drive dropped ai-room wakes to CC-hosted role handles (claude-arm lane).

Runs as a PostCompact command hook in the co-lead agent (frontmatter-scoped),
and is safe to run manually from any session in this repo. Scans the channel's
deliveries journal for recent `no_route` wake failures, and for each target
handle whose lease is live but publishes no codex app-server URL (a Claude
Code-hosted role such as codex_co_lead), delivers one rate-limited tmux nudge
via mcp_server_lib.wake_send._claude_arm_tmux_nudge.

Notification-only: always exits 0; never blocks the hook event.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import types
from datetime import datetime, timezone

WINDOW_SECS = 45 * 60
TAIL_LINES = 400


def _parse_ts(ts: str) -> float | None:
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return None


def main() -> int:
    try:
        json.load(sys.stdin)  # hook input; consumed but unused
    except Exception:
        pass

    sys.path.insert(0, str(pathlib.Path.home() / ".ai-room"))
    try:
        from mcp_server_lib import compat, paths, state
        state.ROOM = paths._resolve_room(None)
        from mcp_server_lib import codex_discovery, wake_send
        compat.init(types.SimpleNamespace(_pid_alive=codex_discovery._pid_alive))
    except Exception as exc:
        print(f"[claude-arm redrive] ai-room stack unavailable: {exc}")
        return 0

    journal = state.ROOM / "deliveries.jsonl"
    if not journal.exists():
        return 0
    now = time.time()
    pending: dict[str, int] = {}
    try:
        lines = journal.read_text(encoding="utf-8", errors="replace").splitlines()[-TAIL_LINES:]
    except OSError as exc:
        print(f"[claude-arm redrive] journal unreadable: {exc}")
        return 0
    for raw in lines:
        try:
            rec = json.loads(raw)
        except Exception:
            continue
        if rec.get("phase") != "failed" or rec.get("error_kind") != "no_route":
            continue
        ts = _parse_ts(rec.get("ts", ""))
        if ts is None or now - ts > WINDOW_SECS:
            continue
        handle = rec.get("target_handle")
        if isinstance(handle, str) and handle:
            pending[handle] = pending.get(handle, 0) + 1

    for handle, count in pending.items():
        nudged = wake_send._claude_arm_tmux_nudge(
            handle,
            reason=f"{count} undelivered wake(s) in the last {WINDOW_SECS // 60} min",
        )
        print(f"[claude-arm redrive] {handle}: {count} no_route drop(s); nudge={'sent' if nudged else 'skipped'}")
    if not pending:
        print("[claude-arm redrive] no recent no_route drops")
    return 0


if __name__ == "__main__":
    sys.exit(main())
