"""Watchdog breach behavior with synthetic markers (no GPU)."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ISOL = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158-p1b-isol-wt")


def _spawn_trainer_group(tmp_path: Path) -> tuple[subprocess.Popen, int, int]:
    """Spawn a long-sleep child in its own process group."""
    script = tmp_path / "trainer_stub.py"
    script.write_text(
        "import time, os\n"
        "print('trainer_stub_pid', os.getpid(), flush=True)\n"
        "time.sleep(120)\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=str(ISOL),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        process_group=0,
    )
    train_pid = proc.pid
    train_pgid = os.getpgid(train_pid)
    assert train_pgid == train_pid
    return proc, train_pid, train_pgid


def test_budget_breach_kills_trainer_group_only(tmp_path: Path):
    log_path = tmp_path / "phase.log"
    event_log = tmp_path / "events.jsonl"
    touch = tmp_path / "monitor.touch"
    log_path.write_text("", encoding="utf-8")
    touch.write_text("armed\n", encoding="utf-8")

    trainer, train_pid, train_pgid = _spawn_trainer_group(tmp_path)
    supervisor_pgid = os.getpgid(0)
    assert train_pgid != supervisor_pgid

    # Emit an open phase marker; tiny budget forces breach.
    log_path.write_text("[P1B_PHASE] model_build_start\n", encoding="utf-8")

    wd = subprocess.Popen(
        [
            sys.executable,
            "scripts/p1b_phase_watchdog.py",
            "--log",
            str(log_path),
            "--target-pid",
            str(train_pid),
            "--target-pgid",
            str(train_pgid),
            "--event-log",
            str(event_log),
            "--budgets",
            "model_build=0.2",
            "--marker-prefix",
            "[P1B_PHASE]",
            "--require-marker-order-before-enforce",
            "--require-monitor-armed-touch",
            str(touch),
            "--supervisor-pgid",
            str(supervisor_pgid),
            "--poll-interval",
            "0.05",
        ],
        cwd=str(ISOL),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert os.getpgid(wd.pid) != train_pgid

    try:
        wd_rc = wd.wait(timeout=10)
        assert wd_rc == 50, (wd_rc, wd.stdout.read() if wd.stdout else "")
        trainer.wait(timeout=5)
        assert trainer.returncode is not None
        # Trainer must be dead; kill(pid,0) must fail.
        dead = False
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                os.kill(train_pid, 0)
                time.sleep(0.05)
            except ProcessLookupError:
                dead = True
                break
        assert dead, f"trainer pid {train_pid} still alive after breach kill"
        # Watchdog / supervisor process group still alive (this test process).
        os.kill(os.getpid(), 0)
        events = event_log.read_text(encoding="utf-8")
        assert "PHASE_BUDGET_BREACH" in events
        assert "KILLPG_TRAINER" in events
        assert "WATCHDOG_ARMED" in events
    finally:
        if trainer.poll() is None:
            try:
                os.killpg(train_pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.kill(train_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            trainer.wait(timeout=5)
        if wd.poll() is None:
            wd.kill()
            wd.wait(timeout=5)


def test_watchdog_refuses_shared_pgid(tmp_path: Path):
    log_path = tmp_path / "phase.log"
    event_log = tmp_path / "events.jsonl"
    touch = tmp_path / "monitor.touch"
    log_path.write_text("", encoding="utf-8")
    touch.write_text("x", encoding="utf-8")
    own_pgid = os.getpgid(0)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/p1b_phase_watchdog.py",
            "--log",
            str(log_path),
            "--target-pid",
            str(os.getpid()),
            "--target-pgid",
            str(own_pgid),
            "--event-log",
            str(event_log),
            "--budgets",
            "model_build=1",
            "--require-monitor-armed-touch",
            str(touch),
        ],
        cwd=str(ISOL),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 41
    assert "SHARED_PGID" in result.stderr or "OWN_PID_IN_TARGET_PGID" in result.stderr


def test_watchdog_refuses_wrong_member(tmp_path: Path):
    log_path = tmp_path / "phase.log"
    event_log = tmp_path / "events.jsonl"
    touch = tmp_path / "monitor.touch"
    log_path.write_text("", encoding="utf-8")
    touch.write_text("x", encoding="utf-8")
    # Use a fake pgid that does not contain this pid.
    fake_pgid = os.getpid() + 99999
    result = subprocess.run(
        [
            sys.executable,
            "scripts/p1b_phase_watchdog.py",
            "--log",
            str(log_path),
            "--target-pid",
            str(os.getpid()),
            "--target-pgid",
            str(fake_pgid),
            "--event-log",
            str(event_log),
            "--budgets",
            "model_build=1",
            "--require-monitor-armed-touch",
            str(touch),
        ],
        cwd=str(ISOL),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 42
    assert "WRONG_MEMBER" in result.stderr


def test_event_log_preexists_refused(tmp_path: Path):
    log_path = tmp_path / "phase.log"
    event_log = tmp_path / "events.jsonl"
    touch = tmp_path / "monitor.touch"
    log_path.write_text("", encoding="utf-8")
    event_log.write_text("{}\n", encoding="utf-8")  # pre-existing
    touch.write_text("x", encoding="utf-8")
    trainer, train_pid, train_pgid = _spawn_trainer_group(tmp_path)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/p1b_phase_watchdog.py",
                "--log",
                str(log_path),
                "--target-pid",
                str(train_pid),
                "--target-pgid",
                str(train_pgid),
                "--event-log",
                str(event_log),
                "--budgets",
                "model_build=10",
                "--require-monitor-armed-touch",
                str(touch),
                "--supervisor-pgid",
                str(os.getpgid(0)),
            ],
            cwd=str(ISOL),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 43
        assert "WATCHDOG_EVENT_LOG_PREEXISTS" in result.stderr
    finally:
        if trainer.poll() is None:
            try:
                os.killpg(train_pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            trainer.wait(timeout=5)


def test_marker_order_mismatch_kills_and_exits(tmp_path: Path):
    log_path = tmp_path / "phase.log"
    event_log = tmp_path / "events.jsonl"
    touch = tmp_path / "monitor.touch"
    log_path.write_text("", encoding="utf-8")
    touch.write_text("armed\n", encoding="utf-8")
    trainer, train_pid, train_pgid = _spawn_trainer_group(tmp_path)
    # Skip model_build_start — emit model_build_end first (reorder).
    log_path.write_text("[P1B_PHASE] model_build_end\n", encoding="utf-8")
    wd = subprocess.Popen(
        [
            sys.executable,
            "scripts/p1b_phase_watchdog.py",
            "--log",
            str(log_path),
            "--target-pid",
            str(train_pid),
            "--target-pgid",
            str(train_pgid),
            "--event-log",
            str(event_log),
            "--budgets",
            "model_build=30",
            "--marker-prefix",
            "[P1B_PHASE]",
            "--require-marker-order-before-enforce",
            "--require-monitor-armed-touch",
            str(touch),
            "--supervisor-pgid",
            str(os.getpgid(0)),
            "--poll-interval",
            "0.05",
        ],
        cwd=str(ISOL),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wd_rc = wd.wait(timeout=10)
        assert wd_rc == 51, (wd_rc, wd.stdout.read() if wd.stdout else "")
        events = event_log.read_text(encoding="utf-8")
        assert "MARKER_ORDER_MISMATCH" in events
        assert "KILLPG_TRAINER" in events
        # Reap; SIGKILL may leave a zombie until wait().
        try:
            trainer.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        dead = trainer.returncode is not None
        if not dead:
            try:
                os.kill(train_pid, 0)
            except ProcessLookupError:
                dead = True
        assert dead, f"trainer pid {train_pid} still running after marker-order kill"
    finally:
        if trainer.poll() is None:
            try:
                os.killpg(train_pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            trainer.wait(timeout=5)
        if wd.poll() is None:
            wd.kill()
            wd.wait(timeout=5)


def test_marker_order_missing_start_then_later_token_kills(tmp_path: Path):
    """Missing expected start: emitting forward_backward_start before model_build_start."""
    log_path = tmp_path / "phase.log"
    event_log = tmp_path / "events.jsonl"
    touch = tmp_path / "monitor.touch"
    log_path.write_text("", encoding="utf-8")
    touch.write_text("armed\n", encoding="utf-8")
    trainer, train_pid, train_pgid = _spawn_trainer_group(tmp_path)
    log_path.write_text("[P1B_PHASE] forward_backward_start\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/p1b_phase_watchdog.py",
            "--log",
            str(log_path),
            "--target-pid",
            str(train_pid),
            "--target-pgid",
            str(train_pgid),
            "--event-log",
            str(event_log),
            "--budgets",
            "model_build=30,forward_backward=30",
            "--require-marker-order-before-enforce",
            "--require-monitor-armed-touch",
            str(touch),
            "--supervisor-pgid",
            str(os.getpgid(0)),
            "--poll-interval",
            "0.05",
            "--idle-exit-sec",
            "0",
        ],
        cwd=str(ISOL),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    try:
        assert result.returncode == 51
        events = event_log.read_text(encoding="utf-8")
        assert "MARKER_ORDER_MISMATCH" in events
    finally:
        if trainer.poll() is None:
            try:
                os.killpg(train_pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            trainer.wait(timeout=5)
