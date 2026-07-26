#!/usr/bin/env python3
"""Thin Phase B supervisor CLI — orchestration only; logic in p1b_supervisor_lib."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from calm.hrm_text_158.native_full_stack import p1b_supervisor_lib as lib

# Re-export exit codes / constants for existing tests.
EXIT_PACKET_MISSING = lib.EXIT_PACKET_MISSING
EXIT_PACKET_SCHEMA = lib.EXIT_PACKET_SCHEMA
EXIT_PACKET_PAYLOAD = lib.EXIT_PACKET_PAYLOAD
EXIT_PACKET_COMMIT = lib.EXIT_PACKET_COMMIT
EXIT_PACKET_FILE_HASH = lib.EXIT_PACKET_FILE_HASH
EXIT_PACKET_ARGV = lib.EXIT_PACKET_ARGV
EXIT_PACKET_PATH_PREEXISTS = lib.EXIT_PACKET_PATH_PREEXISTS
EXIT_PACKET_BUDGET_PATH = lib.EXIT_PACKET_BUDGET_PATH
EXIT_PACKET_FILE_SHA = lib.EXIT_PACKET_FILE_SHA
EXIT_PACKET_SHA_ARG = lib.EXIT_PACKET_SHA_ARG
EXIT_SHARED_PGID = lib.EXIT_SHARED_PGID
EXIT_WATCHDOG_ARMED_TIMEOUT = lib.EXIT_WATCHDOG_ARMED_TIMEOUT
EXIT_ACTIVATION_GATE_TIMEOUT = lib.EXIT_ACTIVATION_GATE_TIMEOUT
EXIT_LOG_PREEXISTS = lib.EXIT_LOG_PREEXISTS
EXIT_TEST_SEAM_REFUSED = lib.EXIT_TEST_SEAM_REFUSED
EXIT_TEST_MODE_CANONICAL_REFUSED = lib.EXIT_TEST_MODE_CANONICAL_REFUSED
EXIT_OWNERSHIP_FAILURE = lib.EXIT_OWNERSHIP_FAILURE
FROZEN_INNER_SCIENCE_ARGV = lib.FROZEN_INNER_SCIENCE_ARGV
ISOL_WT = lib.ISOL_WT
validate_packet_pre_spawn = lib.validate_packet_pre_spawn


def _path_under_tmp(path: Path) -> bool:
    resolved = str(path.resolve())
    return "/pytest-" in resolved or "/tmp/" in resolved or resolved.startswith("/tmp")


def _path_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _assert_production_refuses_test_seams(
    args: argparse.Namespace,
    *,
    test_mode: bool,
) -> None:
    """Production path refuses ambient test inputs unless --test-mode-fixtures."""
    ambient_env = any(
        os.environ.get(k) == "1"
        for k in ("P1B_ALLOW_NONCANONICAL_PATHS", "P1B_ALLOW_TEST_BUDGETS", "P1B_ALLOW_TEST_TRAINER_COMMAND")
    )
    override_flags = any(
        [
            args.trainer_command is not None,
            args.watchdog_command is not None,
            args.monitor_touch is not None,
            args.cwd_override,
            args.watchdog_armed_deadline_sec != 30.0,
            args.monitor_armed_deadline_sec != 60.0,
        ]
    )
    if test_mode:
        return
    if ambient_env or override_flags:
        print(
            "TEST_SEAM_REFUSED production path rejects ambient test inputs "
            "(use --test-mode-fixtures for fixture packets under tmp roots)",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(EXIT_TEST_SEAM_REFUSED)


def _assert_test_mode_packet_is_fixture(packet_path: Path) -> None:
    if not _path_under_tmp(packet_path):
        print(
            f"TEST_MODE_CANONICAL_REFUSED packet={packet_path} "
            "(test mode hard-requires fixture packet under pytest/tmp roots)",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(EXIT_TEST_MODE_CANONICAL_REFUSED)


def _assert_test_mode_write_targets_contained(
    *,
    fixture_root: Path,
    paths: dict,
    env: dict,
    monitor_touch_override: Path | None,
) -> None:
    """Test mode: every write sink must realpath under the fixture tmp root.

    Covers packet.paths entries, env P1B_LIVE_CONVERSION_RECEIPT_JSON, and
    --monitor-touch. Rejects canonical main-tree/isol-wt targets before any
    log touch or child spawn.
    """
    root = fixture_root.resolve()
    targets: list[tuple[str, Path]] = [
        (f"paths.{key}", Path(str(val))) for key, val in paths.items()
    ]
    receipt = env.get("P1B_LIVE_CONVERSION_RECEIPT_JSON")
    if receipt:
        targets.append(("env.P1B_LIVE_CONVERSION_RECEIPT_JSON", Path(str(receipt))))
    if monitor_touch_override is not None:
        targets.append(("--monitor-touch", Path(monitor_touch_override)))

    for label, raw in targets:
        resolved = raw.expanduser().resolve()
        if not _path_under_root(resolved, root):
            print(
                f"TEST_MODE_CANONICAL_REFUSED {label}={resolved} "
                f"fixture_root={root} "
                "(test mode requires all write targets under fixture tmp root "
                "after symlink/realpath resolution)",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(EXIT_TEST_MODE_CANONICAL_REFUSED)


def run_supervisor(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P1b Phase B foreground supervisor")
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--packet-sha256", default=None)
    parser.add_argument("--trainer-command", default=None)
    parser.add_argument("--watchdog-command", default=None)
    parser.add_argument("--monitor-touch", type=Path, default=None)
    parser.add_argument("--watchdog-armed-deadline-sec", type=float, default=30.0)
    parser.add_argument("--monitor-armed-deadline-sec", type=float, default=60.0)
    parser.add_argument("--cwd", type=Path, default=ISOL_WT)
    parser.add_argument(
        "--test-mode-fixtures",
        action="store_true",
        default=False,
        help="Sealed test mode: allows stub overrides; requires fixture packet under tmp",
    )
    args = parser.parse_args(argv)
    args.cwd_override = Path(args.cwd).resolve() != ISOL_WT.resolve()

    test_mode = bool(args.test_mode_fixtures)
    _assert_production_refuses_test_seams(args, test_mode=test_mode)
    if test_mode:
        _assert_test_mode_packet_is_fixture(Path(args.packet))

    cwd = Path(args.cwd).resolve()
    allow_noncanonical = test_mode
    allow_test_budgets = test_mode
    allow_test_trainer = test_mode

    packet = lib.validate_packet_pre_spawn(
        args.packet,
        args.packet_sha256,
        cwd=cwd,
        allow_noncanonical_paths=allow_noncanonical,
        allow_test_budgets=allow_test_budgets,
        allow_test_trainer_command=allow_test_trainer,
    )

    paths = dict(packet["paths"])
    if args.monitor_touch is not None:
        paths["monitor_armed_touch"] = str(args.monitor_touch)

    if test_mode:
        _assert_test_mode_write_targets_contained(
            fixture_root=Path(args.packet).resolve().parent,
            paths=paths,
            env=dict(packet.get("env") or {}),
            monitor_touch_override=args.monitor_touch,
        )

    log_path = Path(paths["phase_b_log"])
    activation_path = Path(paths["activation_receipt"])
    event_log = Path(paths["watchdog_event_log"])
    monitor_touch = Path(paths["monitor_armed_touch"])

    if log_path.exists():
        print(f"LOG_PREEXISTS {log_path}", file=sys.stderr, flush=True)
        return EXIT_LOG_PREEXISTS

    for p in (log_path, event_log, activation_path, monitor_touch):
        p.parent.mkdir(parents=True, exist_ok=True)

    lib.o_excl_touch(log_path)

    env = os.environ.copy()
    for key, val in dict(packet["env"]).items():
        env[str(key)] = str(val)
    env.setdefault("PYTHONPATH", str(cwd))

    if args.trainer_command:
        trainer_argv = lib.parse_shell_argv(args.trainer_command)
    else:
        trainer_argv = lib.parse_shell_argv(str(packet["inner_science_argv"]))

    supervisor_pid = os.getpid()
    supervisor_pgid = os.getpgid(0)

    trainer: subprocess.Popen | None = None
    watchdog: subprocess.Popen | None = None
    train_pgid: int | None = None
    train_pid: int | None = None
    wd_argv: list[str] = []
    ownership_error: str | None = None

    log_f = open(log_path, "a", encoding="utf-8", buffering=1)
    try:
        try:
            trainer = subprocess.Popen(
                trainer_argv,
                cwd=str(cwd),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                process_group=0,
            )
            train_pid = int(trainer.pid)
            train_pgid = os.getpgid(train_pid)
            if train_pgid == supervisor_pgid:
                print("SHARED_PGID STOP", file=sys.stderr, flush=True)
                ownership_error = "SHARED_PGID"
                return EXIT_SHARED_PGID

            if args.watchdog_command:
                wd_argv = lib.parse_shell_argv(args.watchdog_command)
                wd_argv = [
                    a.replace("$TRAIN_PID", str(train_pid)).replace("$TRAIN_PGID", str(train_pgid))
                    for a in wd_argv
                ]
            else:
                wd_argv = lib.build_default_watchdog_cmd(
                    {**packet, "paths": paths},
                    train_pid=train_pid,
                    train_pgid=train_pgid,
                    supervisor_pgid=supervisor_pgid,
                )

            watchdog = subprocess.Popen(
                wd_argv,
                cwd=str(cwd),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            watchdog_pid = int(watchdog.pid)
            watchdog_pgid = os.getpgid(watchdog_pid)
            if watchdog_pgid == train_pgid:
                print("WATCHDOG_SHARED_TRAIN_PGID STOP", file=sys.stderr, flush=True)
                ownership_error = "WATCHDOG_SHARED_TRAIN_PGID"
                return EXIT_SHARED_PGID

            armed_line = lib.wait_for_watchdog_armed(
                event_log,
                float(args.watchdog_armed_deadline_sec),
                watchdog_proc=watchdog,
                log_path=log_path,
            )
            if armed_line is None:
                print("WATCHDOG_ARMED_TIMEOUT", file=sys.stderr, flush=True)
                ownership_error = "WATCHDOG_ARMED_TIMEOUT"
                return EXIT_WATCHDOG_ARMED_TIMEOUT

            deadlines = dict(packet.get("activation_deadlines") or {})
            watch_wrap_cmd = str(packet.get("watch_wrap_command_exact") or "")
            if not watch_wrap_cmd.strip():
                print("PACKET_WATCH_WRAP_COMMAND_EXACT_MISSING", file=sys.stderr, flush=True)
                ownership_error = "WATCH_WRAP_COMMAND_MISSING"
                return EXIT_PACKET_SCHEMA

            receipt = lib.build_activation_receipt(
                train_pid=train_pid,
                train_pgid=train_pgid,
                supervisor_pid=supervisor_pid,
                supervisor_pgid=supervisor_pgid,
                watchdog_pid=watchdog_pid,
                watchdog_pgid=watchdog_pgid,
                watchdog_command_exact=wd_argv,
                watchdog_sha256=lib.sha256_file(cwd / "scripts/p1b_phase_watchdog.py"),
                watch_wrap_command_exact=watch_wrap_cmd,
                watch_wrap_sha256=str(packet["watch_wrap_sha256"]),
                trainer_argv=trainer_argv,
                log_path=log_path,
                packet_path=Path(args.packet),
                packet_file_sha256=str(args.packet_sha256),
                packet_payload_digest=str(packet["packet_payload_digest"]),
                watchdog_activation_line=armed_line,
                activation_deadlines=deadlines,
            )
            if "watch_wrap_command_exact" not in receipt or "watch_wrap_sha256" not in receipt:
                print("ACTIVATION_RECEIPT_WATCH_WRAP_FIELDS_MISSING", file=sys.stderr, flush=True)
                ownership_error = "ACTIVATION_RECEIPT_INCOMPLETE"
                return EXIT_PACKET_SCHEMA
            lib.write_activation_receipt_o_excl(activation_path, receipt)

            mon_deadline = float(
                deadlines.get(
                    "WATCHDOG_ARMED_to_Monitor_armed_sec",
                    args.monitor_armed_deadline_sec,
                )
            )
            if not lib.wait_for_monitor_touch(monitor_touch, mon_deadline):
                print("ACTIVATION_GATE_TIMEOUT", file=sys.stderr, flush=True)
                ownership_error = "ACTIVATION_GATE_TIMEOUT"
                return EXIT_ACTIVATION_GATE_TIMEOUT

            train_rc = trainer.wait()
            if watchdog.poll() is None:
                try:
                    watchdog.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    watchdog.terminate()
                    watchdog.wait(timeout=10)
            watchdog_rc = watchdog.returncode
            print(
                f"SUPERVISOR_REAP train_rc={train_rc} watchdog_rc={watchdog_rc}",
                flush=True,
            )
            # Success requires both children RC 0.
            if train_rc != 0 or watchdog_rc != 0:
                ownership_error = f"CHILD_RC train={train_rc} wd={watchdog_rc}"
                return int(train_rc or watchdog_rc or 1)
            return 0
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 — ownership finally must seal
            print(f"SUPERVISOR_OWNERSHIP_ERROR {exc}", file=sys.stderr, flush=True)
            ownership_error = str(exc)
            return EXIT_OWNERSHIP_FAILURE
    finally:
        # Broad ownership finally: terminate/reap every spawned child; never kill self.
        if ownership_error is not None or (
            trainer is not None and trainer.poll() is None
        ) or (watchdog is not None and watchdog.poll() is None):
            lib.terminate_children(
                train_pgid=train_pgid,
                trainer=trainer,
                watchdog=watchdog,
                protected_pgids={supervisor_pgid},
            )
        log_f.close()


def main(argv: list[str] | None = None) -> int:
    return run_supervisor(argv)


if __name__ == "__main__":
    raise SystemExit(main())
