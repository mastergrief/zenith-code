"""Hostile PGID / forbidden orchestration primitives for Phase B supervisor."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ISOL = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158-p1b-isol-wt")
SUPERVISOR = ISOL / "scripts" / "p1b_phaseB_supervisor.py"
WATCHDOG = ISOL / "scripts" / "p1b_phase_watchdog.py"


def test_supervisor_source_forbids_setsid_nohup_trailing_amp():
    src = SUPERVISOR.read_text(encoding="utf-8")
    code_lines = [
        line
        for line in src.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    joined = "\n".join(code_lines)
    assert "setsid" not in joined
    assert "nohup" not in joined
    assert "daemonize" not in joined
    assert not re.search(r"&\s*$", joined, re.M)


def test_watchdog_source_forbids_setsid_nohup():
    src = WATCHDOG.read_text(encoding="utf-8")
    code_lines = [
        line
        for line in src.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    joined = "\n".join(code_lines)
    assert "setsid" not in joined
    assert "nohup" not in joined


def test_watchdog_shared_pgid_refusal_subprocess(tmp_path: Path):
    log_path = tmp_path / "l.log"
    event_log = tmp_path / "e.jsonl"
    touch = tmp_path / "t"
    log_path.write_text("", encoding="utf-8")
    touch.write_text("1", encoding="utf-8")
    own_pgid = os.getpgid(0)
    result = subprocess.run(
        [
            sys.executable,
            str(WATCHDOG),
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


def test_watchdog_wrong_member_refusal_subprocess(tmp_path: Path):
    log_path = tmp_path / "l.log"
    event_log = tmp_path / "e.jsonl"
    touch = tmp_path / "t"
    log_path.write_text("", encoding="utf-8")
    touch.write_text("1", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(WATCHDOG),
            "--log",
            str(log_path),
            "--target-pid",
            str(os.getpid()),
            "--target-pgid",
            str(os.getpid() + 424242),
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


def test_real_supervisor_shared_pgid_never_killpg_self(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Engineered shared-PGID: real supervisor flow; killpg(supervisor_pgid) never called."""
    import hashlib
    import json
    import time

    import pytest

    from calm.hrm_text_158.native_full_stack import p1b_supervisor_lib as lib
    from scripts.p1b_phase_b_packet_mint import RUNTIME_EXECUTABLE_KEYS, mint_phase_b_packet_o_excl
    from scripts.p1b_phaseB_supervisor import EXIT_SHARED_PGID, run_supervisor
    import scripts.p1b_phaseB_supervisor as sup

    runtime = {
        rel: hashlib.sha256((ISOL / rel).read_bytes()).hexdigest()
        for rel in RUNTIME_EXECUTABLE_KEYS
    }
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ISOL, text=True).strip()
    fields = {
        "plan_id": "p1b_test",
        "plan_revision": "v15",
        "plan_sha256": "b" * 64,
        "gate_msg_ids": {
            "plus1_implement": "g1",
            "implement_gate1": "g2",
            "implement_gate2": "g3",
            "plus1_commit": "g4",
        },
        "commit_sha": head,
        "runtime_executable_sha256s": runtime,
        "watch_wrap_sha256": runtime["bin/watch-wrap"],
        "watch_wrap_command_exact": (
            'bin/watch-wrap --log <path> --heartbeat 30 --error "Traceback|Error|Killed|OOM|FAILED|assert" '
            '--progress "[P1B_PHASE]" --success "TERMINAL_OK" --stop-on "TERMINAL_OK" --replay 20'
        ),
        "inner_science_argv": (
            "timeout --kill-after=30 900 python3 scripts/train_hrm_text_158.py "
            "--use-ternary-bulk --sub2-authority-live-conversion-proof "
            "--sub2-authority-eligible-scope all-bitlinear --device cuda "
            "--epochs 1 --batch-size 8 --max-len 256 --seed 1"
        ),
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
        "phase_budgets": {
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
    path = tmp_path / "packet.json"
    minted = mint_phase_b_packet_o_excl(path, fields)

    self_pgid = os.getpgid(0)
    killpg_calls: list[int] = []
    real_killpg = os.killpg

    def audit_killpg(pgid: int, sig: int) -> None:
        killpg_calls.append(int(pgid))
        if int(pgid) == self_pgid:
            raise AssertionError("killpg(supervisor_pgid) must never be called")
        return real_killpg(pgid, sig)

    real_popen = subprocess.Popen

    def popen_inherit_pgid(*args, **kwargs):
        kwargs.pop("process_group", None)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(os, "killpg", audit_killpg)
    monkeypatch.setattr(sup.subprocess, "Popen", popen_inherit_pgid)

    trainer_cmd = f"{sys.executable} -c \"import time; time.sleep(30)\""
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
            trainer_cmd,
            "--watchdog-armed-deadline-sec",
            "2",
            "--monitor-armed-deadline-sec",
            "2",
        ]
    )
    assert rc == EXIT_SHARED_PGID
    assert self_pgid not in killpg_calls
    # Supervisor (this process) still alive after cleanup.
    os.kill(os.getpid(), 0)
    # Evidence sealed: log created before shared-PGID stop.
    assert Path(fields["paths"]["phase_b_log"]).is_file()
    # Give reaper a moment; no orphan sleep trainers from this test's PGID share.
    time.sleep(0.2)
