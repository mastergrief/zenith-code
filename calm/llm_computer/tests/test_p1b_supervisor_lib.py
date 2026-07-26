"""Characterization + hostile tests for p1b_supervisor_lib facade."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack import p1b_supervisor_lib as lib
from scripts.p1b_phase_b_packet_mint import RUNTIME_EXECUTABLE_KEYS, mint_phase_b_packet_o_excl
from scripts.p1b_phaseB_supervisor import (
    EXIT_TEST_MODE_CANONICAL_REFUSED,
    EXIT_TEST_SEAM_REFUSED,
    run_supervisor,
)

ISOL = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158-p1b-isol-wt")
WW_CMD = (
    'bin/watch-wrap --log <path> --heartbeat 30 --error "Traceback|Error|Killed|OOM|FAILED|assert" '
    '--progress "[P1B_PHASE]" --success "TERMINAL_OK" --stop-on "TERMINAL_OK" --replay 20'
)


def _runtime_hashes() -> dict[str, str]:
    return {
        rel: hashlib.sha256((ISOL / rel).read_bytes()).hexdigest()
        for rel in RUNTIME_EXECUTABLE_KEYS
    }


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ISOL, text=True).strip()


def _fields(tmp_path: Path) -> dict:
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
        "watch_wrap_command_exact": WW_CMD,
        "inner_science_argv": lib.FROZEN_INNER_SCIENCE_ARGV,
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


def test_build_activation_receipt_keeps_watch_wrap_fields_distinct(tmp_path: Path):
    receipt = lib.build_activation_receipt(
        train_pid=1,
        train_pgid=2,
        supervisor_pid=3,
        supervisor_pgid=4,
        watchdog_pid=5,
        watchdog_pgid=6,
        watchdog_command_exact=["wd"],
        watchdog_sha256="a" * 64,
        watch_wrap_command_exact=WW_CMD,
        watch_wrap_sha256="b" * 64,
        trainer_argv=["python"],
        log_path=tmp_path / "log",
        packet_path=tmp_path / "pkt",
        packet_file_sha256="c" * 64,
        packet_payload_digest="d" * 64,
        watchdog_activation_line="WATCHDOG_ARMED",
        activation_deadlines={},
    )
    assert receipt["watch_wrap_command_exact"] == WW_CMD
    assert receipt["watch_wrap_sha256"] == "b" * 64
    assert receipt["watch_wrap_command_exact"] != receipt["watch_wrap_sha256"]
    sha = lib.write_activation_receipt_o_excl(tmp_path / "act.json", receipt)
    assert len(sha) == 64
    loaded = json.loads((tmp_path / "act.json").read_text(encoding="utf-8"))
    assert loaded["watch_wrap_command_exact"] == WW_CMD
    assert loaded["watch_wrap_sha256"] == "b" * 64


def test_production_refuses_ambient_test_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    fields = _fields(tmp_path)
    path = tmp_path / "packet.json"
    minted = mint_phase_b_packet_o_excl(path, fields)
    with pytest.raises(SystemExit) as ei:
        run_supervisor(
            [
                "--packet",
                str(path),
                "--packet-sha256",
                minted["packet_file_sha256"],
                "--cwd",
                str(ISOL),
            ]
        )
    assert ei.value.code == EXIT_TEST_SEAM_REFUSED


def test_production_refuses_trainer_command_override(tmp_path: Path):
    fields = _fields(tmp_path)
    path = tmp_path / "packet.json"
    minted = mint_phase_b_packet_o_excl(path, fields)
    with pytest.raises(SystemExit) as ei:
        run_supervisor(
            [
                "--packet",
                str(path),
                "--packet-sha256",
                minted["packet_file_sha256"],
                "--cwd",
                str(ISOL),
                "--trainer-command",
                "python3 -c 'pass'",
            ]
        )
    assert ei.value.code == EXIT_TEST_SEAM_REFUSED


def test_test_mode_refuses_canonical_main_tree_packet(tmp_path: Path):
    # Mint under isol artifacts (canonical-ish path, not pytest tmp).
    canon_dir = ISOL / "artifacts" / "acc_entropy" / "_p1b_test_mode_refuse"
    canon_dir.mkdir(parents=True, exist_ok=True)
    # Use a path that is NOT under /tmp or pytest-
    fields = _fields(tmp_path)
    # Point packet path at isol-wt artifacts (not tmp).
    packet_path = canon_dir / f"refuse_{os.getpid()}.json"
    if packet_path.exists():
        packet_path.unlink()
    # Rebuild fields paths under tmp still, but packet file itself is outside tmp.
    minted = mint_phase_b_packet_o_excl(packet_path, fields)
    try:
        with pytest.raises(SystemExit) as ei:
            run_supervisor(
                [
                    "--test-mode-fixtures",
                    "--packet",
                    str(packet_path),
                    "--packet-sha256",
                    minted["packet_file_sha256"],
                    "--cwd",
                    str(ISOL),
                    "--trainer-command",
                    "python3 -c 'pass'",
                ]
            )
        assert ei.value.code == EXIT_TEST_MODE_CANONICAL_REFUSED
    finally:
        packet_path.unlink(missing_ok=True)


def test_validate_packet_requires_watch_wrap_command_exact(tmp_path: Path):
    fields = _fields(tmp_path)
    del fields["watch_wrap_command_exact"]
    fields["packet_payload_digest"] = "c" * 64
    raw = (json.dumps(fields, sort_keys=True) + "\n").encode()
    path = tmp_path / "missing_ww.json"
    path.write_bytes(raw)
    with pytest.raises(SystemExit) as ei:
        lib.validate_packet_pre_spawn(
            path,
            hashlib.sha256(raw).hexdigest(),
            cwd=ISOL,
            allow_noncanonical_paths=True,
            allow_test_budgets=True,
            allow_test_trainer_command=True,
        )
    assert ei.value.code == lib.EXIT_PACKET_SCHEMA


def test_terminate_children_kills_trainer_group_only(tmp_path: Path):
    script = tmp_path / "sleep.py"
    script.write_text("import time; time.sleep(60)\n", encoding="utf-8")
    trainer = subprocess.Popen(
        [sys.executable, str(script)],
        process_group=0,
    )
    train_pgid = os.getpgid(trainer.pid)
    wd = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
    )
    assert os.getpgid(wd.pid) != train_pgid
    lib.terminate_children(train_pgid=train_pgid, trainer=trainer, watchdog=wd)
    assert trainer.poll() is not None
    assert wd.poll() is not None
    # This test process (supervisor stand-in) still alive.
    os.kill(os.getpid(), 0)


def test_terminate_children_protected_pgid_uses_pid_only_never_killpg_self(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Shared-PGID anomaly: killpg(protected) must never fire; PID kill + reap instead."""
    script = tmp_path / "sleep.py"
    script.write_text("import time; time.sleep(60)\n", encoding="utf-8")
    # Inherit caller PGID (no process_group=0).
    trainer = subprocess.Popen([sys.executable, str(script)])
    train_pgid = os.getpgid(trainer.pid)
    self_pgid = os.getpgid(0)
    assert train_pgid == self_pgid
    wd = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], process_group=0)

    killpg_calls: list[int] = []
    real_killpg = os.killpg

    def audit_killpg(pgid: int, sig: int) -> None:
        killpg_calls.append(int(pgid))
        if int(pgid) == self_pgid:
            raise AssertionError("killpg(supervisor_pgid) forbidden")
        return real_killpg(pgid, sig)

    monkeypatch.setattr(os, "killpg", audit_killpg)
    lib.terminate_children(
        train_pgid=train_pgid,
        trainer=trainer,
        watchdog=wd,
        protected_pgids={self_pgid},
    )
    assert self_pgid not in killpg_calls
    assert trainer.poll() is not None
    assert wd.poll() is not None
    os.kill(os.getpid(), 0)


def test_test_mode_refuses_canonical_inner_path(tmp_path: Path):
    fields = _fields(tmp_path)
    canon = ISOL / "artifacts" / "acc_entropy" / f"_p1b_fixture_refuse_log_{os.getpid()}.log"
    fields["paths"]["phase_b_log"] = str(canon)
    path = tmp_path / "pkt_inner.json"
    minted = mint_phase_b_packet_o_excl(path, fields)
    try:
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
                    "python3 -c 'pass'",
                ]
            )
        assert ei.value.code == EXIT_TEST_MODE_CANONICAL_REFUSED
        assert not canon.exists()
        assert not Path(fields["paths"]["activation_receipt"]).exists()
    finally:
        canon.unlink(missing_ok=True)


def test_test_mode_refuses_canonical_env_receipt_path(tmp_path: Path):
    fields = _fields(tmp_path)
    canon = ISOL / "artifacts" / "acc_entropy" / f"_p1b_fixture_refuse_rcpt_{os.getpid()}.json"
    fields["env"]["P1B_LIVE_CONVERSION_RECEIPT_JSON"] = str(canon)
    path = tmp_path / "pkt_env.json"
    minted = mint_phase_b_packet_o_excl(path, fields)
    try:
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
                    "python3 -c 'pass'",
                ]
            )
        assert ei.value.code == EXIT_TEST_MODE_CANONICAL_REFUSED
        assert not Path(fields["paths"]["phase_b_log"]).exists()
        assert not canon.exists()
    finally:
        canon.unlink(missing_ok=True)


def test_test_mode_refuses_canonical_monitor_touch_override(tmp_path: Path):
    fields = _fields(tmp_path)
    path = tmp_path / "pkt_mon.json"
    minted = mint_phase_b_packet_o_excl(path, fields)
    canon_touch = ISOL / "artifacts" / "acc_entropy" / f"_p1b_fixture_refuse_touch_{os.getpid()}"
    try:
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
                    "--monitor-touch",
                    str(canon_touch),
                    "--trainer-command",
                    "python3 -c 'pass'",
                ]
            )
        assert ei.value.code == EXIT_TEST_MODE_CANONICAL_REFUSED
        assert not Path(fields["paths"]["phase_b_log"]).exists()
        assert not canon_touch.exists()
    finally:
        canon_touch.unlink(missing_ok=True)
