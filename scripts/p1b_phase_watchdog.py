#!/usr/bin/env python3
"""Marker-driven Phase B enforcer: per-phase budgets → killpg(trainer group only)."""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from scripts.p1b_o_excl_copy import write_bytes_o_excl

DEFAULT_MARKER_PREFIX = "[P1B_PHASE]"
REQUIRED_MARKER_ORDER = (
    "model_build_start",
    "model_build_end",
    "forward_backward_start",
    "forward_backward_end",
    "vote_apply_start",
    "vote_apply_end",
    "checkpoint_roundtrip_start",
    "checkpoint_roundtrip_end",
    "receipt_mint_start",
    "receipt_mint_end",
    "TERMINAL_OK",
)


def _parse_budgets(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"invalid budget token: {part!r}")
        name, sec = part.split("=", 1)
        out[name.strip()] = float(sec)
    return out


def _append_event(event_log: Path, record: dict) -> None:
    line = json.dumps(record, sort_keys=True) + "\n"
    with open(event_log, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


EXIT_EVENT_LOG_PREEXISTS = 43
EXIT_MARKER_ORDER_MISMATCH = 51


def _ensure_event_log(path: Path) -> None:
    """O_EXCL-create event log; refuse if path already exists (retry = versioned path)."""
    if path.exists():
        print(
            f"WATCHDOG_EVENT_LOG_PREEXISTS path={path} "
            "(retry requires a versioned event-log path)",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(EXIT_EVENT_LOG_PREEXISTS)
    write_bytes_o_excl(path, b"")


def _pid_in_pgid(pid: int, pgid: int) -> bool:
    try:
        return os.getpgid(pid) == pgid
    except ProcessLookupError:
        return False


def _stack_sample(train_pid: int, stack_sample_cmd: str | None) -> str:
    if stack_sample_cmd:
        try:
            proc = subprocess.run(
                stack_sample_cmd,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return (proc.stdout or "") + (proc.stderr or "")
        except Exception as exc:  # noqa: BLE001 — evidence best-effort
            return f"STACK_SAMPLE_CMD_FAILED: {exc}"
    # Fallback
    stack_path = Path(f"/proc/{train_pid}/stack")
    if stack_path.is_file():
        try:
            return stack_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"STACK_SAMPLE_FALLBACK_FAILED: {exc}"
    return "STACK_SAMPLE_UNAVAILABLE"


def _kill_trainer_group(target_pgid: int, target_pid: int | None = None) -> None:
    try:
        os.killpg(target_pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Fallback: signal the process group via negative pid, then the leader.
        try:
            os.kill(-int(target_pgid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    if target_pid is not None:
        try:
            os.kill(int(target_pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P1b marker-driven phase watchdog")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--target-pid", type=int, required=True)
    parser.add_argument("--target-pgid", type=int, required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument(
        "--budgets",
        required=True,
        help="comma list phase=seconds e.g. model_build=180,forward_backward=300",
    )
    parser.add_argument("--marker-prefix", default=DEFAULT_MARKER_PREFIX)
    parser.add_argument(
        "--require-marker-order-before-enforce",
        action="store_true",
        default=False,
    )
    parser.add_argument("--require-monitor-armed-touch", type=Path, required=True)
    parser.add_argument("--supervisor-pgid", type=int, default=None)
    parser.add_argument("--stack-sample-cmd", default=None)
    parser.add_argument(
        "--on-breach",
        default="kill-process-group",
        choices=("kill-process-group",),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="log poll interval seconds",
    )
    parser.add_argument(
        "--idle-exit-sec",
        type=float,
        default=0.0,
        help="if >0, exit cleanly after this many seconds with no open phase (tests)",
    )
    args = parser.parse_args(argv)

    own_pid = os.getpid()
    own_pgid = os.getpgid(0)
    train_pid = int(args.target_pid)
    train_pgid = int(args.target_pgid)

    if own_pgid == train_pgid:
        print(
            f"WATCHDOG_REFUSE SHARED_PGID own_pgid={own_pgid} target_pgid={train_pgid}",
            file=sys.stderr,
            flush=True,
        )
        return 41
    if args.supervisor_pgid is not None and own_pgid == int(args.supervisor_pgid):
        # Watchdog may share supervisor PGID (spawned without process_group=0);
        # only refuse when *target* PGID is wrong. Record membership separately.
        pass
    if not _pid_in_pgid(train_pid, train_pgid):
        print(
            f"WATCHDOG_REFUSE WRONG_MEMBER train_pid={train_pid} "
            f"not in target_pgid={train_pgid}",
            file=sys.stderr,
            flush=True,
        )
        return 42
    # Refuse if our own PID is a member of the trainer PGID (shared group).
    if _pid_in_pgid(own_pid, train_pgid):
        print(
            f"WATCHDOG_REFUSE OWN_PID_IN_TARGET_PGID pid={own_pid} pgid={train_pgid}",
            file=sys.stderr,
            flush=True,
        )
        return 41

    budgets = _parse_budgets(args.budgets)
    prefix = str(args.marker_prefix)
    marker_re = re.compile(
        rf"^{re.escape(prefix)}\s+(\S+)\s*$"
    )

    _ensure_event_log(args.event_log)
    ts = time.time()
    armed_line = (
        f"WATCHDOG_ARMED train_pid={train_pid} train_pgid={train_pgid} "
        f"orch_pid={own_pid} ts={ts:.6f}"
    )
    print(armed_line, flush=True)
    _append_event(
        args.event_log,
        {
            "event": "WATCHDOG_ARMED",
            "train_pid": train_pid,
            "train_pgid": train_pgid,
            "orch_pid": own_pid,
            "orch_pgid": own_pgid,
            "ts": ts,
            "line": armed_line,
        },
    )

    open_phase: Optional[str] = None
    phase_started_at: Optional[float] = None
    seen_markers: list[str] = []
    order_index = 0
    offset = 0
    last_progress = time.monotonic()
    enforce_enabled = False

    while True:
        # Gate full enforcement on monitor-armed touch.
        if not enforce_enabled and args.require_monitor_armed_touch.exists():
            enforce_enabled = True
            _append_event(
                args.event_log,
                {
                    "event": "MONITOR_ARMED_TOUCH_SEEN",
                    "path": str(args.require_monitor_armed_touch),
                    "ts": time.time(),
                },
            )

        if args.log.is_file():
            data = args.log.read_bytes()
            if len(data) > offset:
                chunk = data[offset:]
                offset = len(data)
                text = chunk.decode("utf-8", errors="replace")
                for raw_line in text.splitlines():
                    line = raw_line.strip()
                    m = marker_re.match(line)
                    if not m:
                        # Also accept markers embedded mid-line.
                        if prefix in line:
                            tail = line.split(prefix, 1)[1].strip()
                            token = tail.split()[0] if tail else ""
                        else:
                            continue
                    else:
                        token = m.group(1)
                    if not token:
                        continue
                    seen_markers.append(token)
                    last_progress = time.monotonic()
                    if args.require_marker_order_before_enforce:
                        expected = (
                            REQUIRED_MARKER_ORDER[order_index]
                            if order_index < len(REQUIRED_MARKER_ORDER)
                            else None
                        )
                        if expected is not None and token == expected:
                            order_index += 1
                        elif token.endswith("_start") or token.endswith("_end") or token == "TERMINAL_OK":
                            if expected is not None and token != expected:
                                sample = _stack_sample(train_pid, args.stack_sample_cmd)
                                _append_event(
                                    args.event_log,
                                    {
                                        "event": "MARKER_ORDER_MISMATCH",
                                        "expected": expected,
                                        "got": token,
                                        "stack_sample": sample[:8000],
                                        "ts": time.time(),
                                    },
                                )
                                _kill_trainer_group(train_pgid, train_pid)
                                _append_event(
                                    args.event_log,
                                    {
                                        "event": "KILLPG_TRAINER",
                                        "reason": "MARKER_ORDER_MISMATCH",
                                        "train_pgid": train_pgid,
                                        "train_pid": train_pid,
                                        "ts": time.time(),
                                    },
                                )
                                return EXIT_MARKER_ORDER_MISMATCH
                    if token.endswith("_start"):
                        phase = token[: -len("_start")]
                        open_phase = phase
                        phase_started_at = time.monotonic()
                    elif token.endswith("_end"):
                        phase = token[: -len("_end")]
                        if open_phase == phase:
                            open_phase = None
                            phase_started_at = None
                    elif token == "TERMINAL_OK":
                        _append_event(
                            args.event_log,
                            {"event": "TERMINAL_OK_SEEN", "ts": time.time()},
                        )
                        return 0

        # Budget breach check (only when enforcement armed).
        if (
            enforce_enabled
            and open_phase is not None
            and phase_started_at is not None
            and open_phase in budgets
        ):
            elapsed = time.monotonic() - phase_started_at
            limit = float(budgets[open_phase])
            if elapsed > limit:
                sample = _stack_sample(train_pid, args.stack_sample_cmd)
                _append_event(
                    args.event_log,
                    {
                        "event": "PHASE_BUDGET_BREACH",
                        "phase": open_phase,
                        "elapsed_sec": elapsed,
                        "budget_sec": limit,
                        "train_pid": train_pid,
                        "train_pgid": train_pgid,
                        "stack_sample": sample[:8000],
                        "ts": time.time(),
                    },
                )
                if args.on_breach == "kill-process-group":
                    _kill_trainer_group(train_pgid, train_pid)
                    _append_event(
                        args.event_log,
                        {
                            "event": "KILLPG_TRAINER",
                            "train_pgid": train_pgid,
                            "train_pid": train_pid,
                            "ts": time.time(),
                        },
                    )
                return 50

        # Trainer gone → exit.
        try:
            os.kill(train_pid, 0)
        except ProcessLookupError:
            _append_event(
                args.event_log,
                {"event": "TRAINER_EXITED", "ts": time.time()},
            )
            return 0

        if args.idle_exit_sec > 0 and open_phase is None:
            if time.monotonic() - last_progress >= float(args.idle_exit_sec):
                return 0

        time.sleep(float(args.poll_interval))


if __name__ == "__main__":
    raise SystemExit(main())
