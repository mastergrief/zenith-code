"""Supervisor lifecycle integration with stub children only."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from scripts.p1b_phase_b_packet_mint import RUNTIME_EXECUTABLE_KEYS, mint_phase_b_packet_o_excl
from scripts.p1b_phaseB_supervisor import (
    EXIT_ACTIVATION_GATE_TIMEOUT,
    EXIT_OWNERSHIP_FAILURE,
    EXIT_PACKET_SHA_ARG,
    FROZEN_INNER_SCIENCE_ARGV,
    run_supervisor,
)

ISOL = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158-p1b-isol-wt")


def _runtime_hashes() -> dict[str, str]:
    return {
        rel: hashlib.sha256((ISOL / rel).read_bytes()).hexdigest()
        for rel in RUNTIME_EXECUTABLE_KEYS
    }


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ISOL, text=True).strip()


def _packet_fields(tmp_path: Path, *, budgets: dict | None = None) -> dict:
    runtime = _runtime_hashes()
    return {
        "plan_id": "p1b_test",
        "plan_revision": "v15",
        "plan_sha256": "b" * 64,
        "gate_msg_ids": {
            "plus1_implement": "g1",
            "implement_gate1": "g2",
            "implement_gate2": "g3",
            "plus1_commit": "g4",
        },
        "commit_sha": _head(),
        "runtime_executable_sha256s": runtime,
        "watch_wrap_sha256": runtime["bin/watch-wrap"],
        "watch_wrap_command_exact": 'bin/watch-wrap --log <path> --heartbeat 30 --error "Traceback|Error|Killed|OOM|FAILED|assert" --progress "[P1B_PHASE]" --success "TERMINAL_OK" --stop-on "TERMINAL_OK" --replay 20',
        "inner_science_argv": FROZEN_INNER_SCIENCE_ARGV,
        "env": {
            "CUDA_VISIBLE_DEVICES": "0",
            "P1B_LIVE_CONVERSION_RECEIPT_JSON": str(tmp_path / "receipt.json"),
            "PYTHONPATH": ".",
        },
        "paths": {
            "phase_b_log": str(tmp_path / "phaseB.log"),
            "watchdog_event_log": str(tmp_path / "wd.jsonl"),
            "activation_receipt": str(tmp_path / "act.json"),
            "monitor_armed_touch": str(tmp_path / "mon.touch"),
            "p1b_receipt": str(tmp_path / "p1b.json"),
        },
        "phase_budgets": budgets
        or {
            "model_build": 180,
            "forward_backward": 300,
            "vote_apply": 120,
            "checkpoint_roundtrip": 180,
            "receipt_mint": 60,
        },
        "activation_deadlines": {
            "trainer_spawn_to_WATCHDOG_ARMED_sec": 30,
            "WATCHDOG_ARMED_to_Monitor_armed_sec": 2,
        },
        "kill_semantics": {"target": "trainer_process_group_only"},
    }


def _mint_packet(tmp_path: Path, fields: dict | None = None) -> tuple[Path, dict]:
    fields = fields or _packet_fields(tmp_path)
    path = tmp_path / "packet.json"
    return path, mint_phase_b_packet_o_excl(path, fields)


def _happy_trainer_stub() -> str:
    # Emit full marker sequence then exit 0.
    markers = [
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
    ]
    body = "; ".join(f"print('[P1B_PHASE] {m}', flush=True)" for m in markers)
    return f"{sys.executable} -c \"import time; {body}; time.sleep(0.3)\""


def _arm_monitor_after_watchdog_armed(
    touch: Path,
    *,
    log_path: Path,
    event_log: Path,
    timeout_sec: float = 15.0,
) -> threading.Thread:
    """Create monitor touch ONLY after WATCHDOG_ARMED (must be absent at pre-spawn)."""

    def _run():
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            for p in (log_path, event_log):
                try:
                    if p.is_file() and "WATCHDOG_ARMED" in p.read_text(encoding="utf-8", errors="replace"):
                        fd = os.open(str(touch), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.write(fd, b"armed\n")
                        os.close(fd)
                        return
                except OSError:
                    pass
            time.sleep(0.05)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def test_packet_refuse_absent_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    path, minted = _mint_packet(tmp_path)
    with pytest.raises(SystemExit) as ei:
        run_supervisor(
            [
                "--test-mode-fixtures",
                "--packet",
                str(path),
                "--cwd",
                str(ISOL),
                "--trainer-command",
                _happy_trainer_stub(),
            ]
        )
    assert ei.value.code == EXIT_PACKET_SHA_ARG


def test_lifecycle_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    monkeypatch.setenv("P1B_ALLOW_TEST_TRAINER_COMMAND", "1")
    fields = _packet_fields(tmp_path)
    path, minted = _mint_packet(tmp_path, fields)
    touch = Path(fields["paths"]["monitor_armed_touch"])
    log_path = Path(fields["paths"]["phase_b_log"])
    event_log = Path(fields["paths"]["watchdog_event_log"])
    _arm_monitor_after_watchdog_armed(touch, log_path=log_path, event_log=event_log)

    rc = run_supervisor(
        [
            "--test-mode-fixtures",
            "--packet",
            str(path),
            "--packet-sha256",
            minted["packet_file_sha256"],
            "--cwd",
            str(ISOL),
            "--trainer-command",
            _happy_trainer_stub(),
            "--watchdog-armed-deadline-sec",
            "10",
            "--monitor-armed-deadline-sec",
            "5",
        ]
    )
    assert rc == 0
    assert log_path.is_file()
    log_text = log_path.read_text(encoding="utf-8")
    assert "WATCHDOG_ARMED" in log_text or event_log.is_file()
    assert "[P1B_PHASE] TERMINAL_OK" in log_text
    act = json.loads(Path(fields["paths"]["activation_receipt"]).read_text(encoding="utf-8"))
    assert act["TRAIN_PGID_ne_SUPERVISOR_PGID"] is True
    assert act["watchdog_pgid_ne_TRAIN_PGID"] is True
    assert act["train_pgid"] != act["supervisor_pgid"]
    assert act["watchdog_pgid"] != act["train_pgid"]
    assert act["packet_file_sha256"] == minted["packet_file_sha256"]
    assert act["watch_wrap_command_exact"]
    assert act["watch_wrap_sha256"]
    assert act["watch_wrap_command_exact"] != act["watch_wrap_sha256"]
    assert "bin/watch-wrap" in act["watch_wrap_command_exact"]


def test_activation_before_watchdog_armed_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Activation receipt must not exist if WATCHDOG_ARMED never appears."""
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    monkeypatch.setenv("P1B_ALLOW_TEST_TRAINER_COMMAND", "1")
    fields = _packet_fields(tmp_path)
    path, minted = _mint_packet(tmp_path, fields)
    # Watchdog stub that never emits WATCHDOG_ARMED / never writes event log.
    wd_stub = f"{sys.executable} -c \"import time; time.sleep(5)\""
    trainer = f"{sys.executable} -c \"import time; time.sleep(5)\""
    rc = run_supervisor(
        [
            "--test-mode-fixtures",
            "--packet",
            str(path),
            "--packet-sha256",
            minted["packet_file_sha256"],
            "--cwd",
            str(ISOL),
            "--trainer-command",
            trainer,
            "--watchdog-command",
            wd_stub,
            "--watchdog-armed-deadline-sec",
            "0.5",
            "--monitor-armed-deadline-sec",
            "1",
        ]
    )
    assert rc == 72  # WATCHDOG_ARMED timeout
    assert not Path(fields["paths"]["activation_receipt"]).exists()


def test_monitor_touch_timeout_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    monkeypatch.setenv("P1B_ALLOW_TEST_TRAINER_COMMAND", "1")
    fields = _packet_fields(tmp_path)
    fields["activation_deadlines"]["WATCHDOG_ARMED_to_Monitor_armed_sec"] = 0.4
    path, minted = _mint_packet(tmp_path, fields)
    trainer = f"{sys.executable} -c \"import time; time.sleep(30)\""
    rc = run_supervisor(
        [
            "--test-mode-fixtures",
            "--packet",
            str(path),
            "--packet-sha256",
            minted["packet_file_sha256"],
            "--cwd",
            str(ISOL),
            "--trainer-command",
            trainer,
            "--watchdog-armed-deadline-sec",
            "10",
            "--monitor-armed-deadline-sec",
            "0.4",
        ]
    )
    assert rc == EXIT_ACTIVATION_GATE_TIMEOUT
    # Activation receipt WAS minted (after WATCHDOG_ARMED) before monitor timeout.
    assert Path(fields["paths"]["activation_receipt"]).is_file()


def test_o_excl_log_and_pgid_distinct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    monkeypatch.setenv("P1B_ALLOW_TEST_TRAINER_COMMAND", "1")
    fields = _packet_fields(tmp_path)
    path, minted = _mint_packet(tmp_path, fields)
    touch = Path(fields["paths"]["monitor_armed_touch"])
    _arm_monitor_after_watchdog_armed(
        touch,
        log_path=Path(fields["paths"]["phase_b_log"]),
        event_log=Path(fields["paths"]["watchdog_event_log"]),
    )
    rc = run_supervisor(
        [
            "--test-mode-fixtures",
            "--packet",
            str(path),
            "--packet-sha256",
            minted["packet_file_sha256"],
            "--cwd",
            str(ISOL),
            "--trainer-command",
            _happy_trainer_stub(),
            "--watchdog-armed-deadline-sec",
            "10",
            "--monitor-armed-deadline-sec",
            "5",
        ]
    )
    assert rc == 0
    act = json.loads(Path(fields["paths"]["activation_receipt"]).read_text(encoding="utf-8"))
    assert act["train_pgid"] != act["supervisor_pgid"]
    assert act["watchdog_pgid"] != act["train_pgid"]
    # Second launch must refuse pre-existing log. Clear other O_EXCL targets first.
    Path(fields["paths"]["monitor_armed_touch"]).unlink(missing_ok=True)
    Path(fields["paths"]["activation_receipt"]).unlink(missing_ok=True)
    Path(fields["paths"]["p1b_receipt"]).unlink(missing_ok=True)
    with pytest.raises(SystemExit) as ei:
        run_supervisor(
            [
            "--test-mode-fixtures",
                "--packet",
                str(path),
                "--packet-sha256",
                minted["packet_file_sha256"],
                "--cwd",
                str(ISOL),
                "--trainer-command",
                _happy_trainer_stub(),
            ]
        )
    assert ei.value.code in (67, 74)


def test_budget_breach_kills_trainer_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    monkeypatch.setenv("P1B_ALLOW_TEST_TRAINER_COMMAND", "1")
    monkeypatch.setenv("P1B_ALLOW_TEST_BUDGETS", "1")
    fields = _packet_fields(
        tmp_path,
        budgets={
            "model_build": 1,
            "forward_backward": 300,
            "vote_apply": 120,
            "checkpoint_roundtrip": 180,
            "receipt_mint": 60,
        },
    )
    path, minted = _mint_packet(tmp_path, fields)
    touch = Path(fields["paths"]["monitor_armed_touch"])
    # Arm monitor AFTER WATCHDOG_ARMED (must be absent at pre-spawn O_EXCL check).
    _arm_monitor_after_watchdog_armed(
        touch,
        log_path=Path(fields["paths"]["phase_b_log"]),
        event_log=Path(fields["paths"]["watchdog_event_log"]),
    )

    trainer = (
        f"{sys.executable} -c "
        "\"import time; print('[P1B_PHASE] model_build_start', flush=True); time.sleep(30)\""
    )
    wd = (
        f"{sys.executable} scripts/p1b_phase_watchdog.py "
        f"--log {fields['paths']['phase_b_log']} "
        f"--target-pid $TRAIN_PID --target-pgid $TRAIN_PGID "
        f"--event-log {fields['paths']['watchdog_event_log']} "
        f"--budgets model_build=0.3 "
        f"--marker-prefix '[P1B_PHASE]' "
        f"--require-marker-order-before-enforce "
        f"--require-monitor-armed-touch {fields['paths']['monitor_armed_touch']} "
        f"--poll-interval 0.05"
    )
    rc = run_supervisor(
        [
            "--test-mode-fixtures",
            "--packet",
            str(path),
            "--packet-sha256",
            minted["packet_file_sha256"],
            "--cwd",
            str(ISOL),
            "--trainer-command",
            trainer,
            "--watchdog-command",
            wd,
            "--watchdog-armed-deadline-sec",
            "10",
            "--monitor-armed-deadline-sec",
            "5",
        ]
    )
    events = Path(fields["paths"]["watchdog_event_log"]).read_text(encoding="utf-8")
    assert "KILLPG_TRAINER" in events or "PHASE_BUDGET_BREACH" in events
    act = json.loads(Path(fields["paths"]["activation_receipt"]).read_text(encoding="utf-8"))
    assert act["supervisor_pid"] > 0
    assert rc != 0 or "KILLPG_TRAINER" in events


def test_activation_oexcl_failure_after_spawn_reaps_both_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Inject activation O_EXCL failure after children spawn → both reaped, nonzero."""
    import scripts.p1b_phaseB_supervisor as sup

    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    monkeypatch.setenv("P1B_ALLOW_TEST_TRAINER_COMMAND", "1")
    fields = _packet_fields(tmp_path)
    path, minted = _mint_packet(tmp_path, fields)
    act_path = Path(fields["paths"]["activation_receipt"])
    log_path = Path(fields["paths"]["phase_b_log"])

    spawned: list[subprocess.Popen] = []
    real_popen = subprocess.Popen

    def tracking_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(sup.subprocess, "Popen", tracking_popen)

    def boom_write(path: Path, receipt: dict) -> str:
        assert len(spawned) >= 2, "activation write must occur after both children spawn"
        assert all(p.poll() is None for p in spawned), "children must still be live at inject"
        raise FileExistsError(str(path))

    monkeypatch.setattr(sup.lib, "write_activation_receipt_o_excl", boom_write)

    trainer = f"{sys.executable} -c \"import time; time.sleep(60)\""
    event_log = fields["paths"]["watchdog_event_log"]
    wd = (
        f"{sys.executable} -c "
        f"\"from pathlib import Path; import time; "
        f"Path({event_log!r}).write_text('WATCHDOG_ARMED stub\\n'); "
        f"time.sleep(60)\""
    )
    rc = run_supervisor(
        [
            "--test-mode-fixtures",
            "--packet",
            str(path),
            "--packet-sha256",
            minted["packet_file_sha256"],
            "--cwd",
            str(ISOL),
            "--trainer-command",
            trainer,
            "--watchdog-command",
            wd,
            "--watchdog-armed-deadline-sec",
            "10",
            "--monitor-armed-deadline-sec",
            "5",
        ]
    )
    assert rc == EXIT_OWNERSHIP_FAILURE or rc != 0
    for proc in spawned:
        assert proc.poll() is not None
    assert log_path.is_file()
    # Injected failure: activation must not be a successful receipt write.
    assert not act_path.exists() or act_path.read_text(encoding="utf-8").strip() == ""
    os.kill(os.getpid(), 0)


def test_trainer_zero_watchdog_nonzero_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Trainer RC 0 + watchdog RC nonzero → supervisor classified nonzero."""
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    monkeypatch.setenv("P1B_ALLOW_TEST_TRAINER_COMMAND", "1")
    fields = _packet_fields(tmp_path)
    path, minted = _mint_packet(tmp_path, fields)
    touch = Path(fields["paths"]["monitor_armed_touch"])
    _arm_monitor_after_watchdog_armed(
        touch,
        log_path=Path(fields["paths"]["phase_b_log"]),
        event_log=Path(fields["paths"]["watchdog_event_log"]),
    )
    event_log = fields["paths"]["watchdog_event_log"]
    wd = (
        f"{sys.executable} -c "
        f"\"from pathlib import Path; import time; import sys; "
        f"Path({event_log!r}).write_text('WATCHDOG_ARMED stub\\n'); "
        f"time.sleep(0.4); sys.exit(9)\""
    )
    rc = run_supervisor(
        [
            "--test-mode-fixtures",
            "--packet",
            str(path),
            "--packet-sha256",
            minted["packet_file_sha256"],
            "--cwd",
            str(ISOL),
            "--trainer-command",
            _happy_trainer_stub(),
            "--watchdog-command",
            wd,
            "--watchdog-armed-deadline-sec",
            "10",
            "--monitor-armed-deadline-sec",
            "5",
        ]
    )
    assert rc != 0
    assert Path(fields["paths"]["activation_receipt"]).is_file()
    assert Path(fields["paths"]["phase_b_log"]).is_file()
