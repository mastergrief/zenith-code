"""O_EXCL copy robustness + cleanup-manifest classifier tests."""
from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from scripts.p1b_o_excl_copy import (
    ShortWriteError,
    classify_cleanup_residual,
    copy_file_o_excl,
    write_bytes_o_excl,
)
from scripts.p1b_phaseB_supervisor import EXIT_LOG_PREEXISTS, run_supervisor


def test_write_bytes_o_excl_no_clobber(tmp_path: Path):
    dest = tmp_path / "artifact.bin"
    sha1 = write_bytes_o_excl(dest, b"hello-world")
    assert dest.read_bytes() == b"hello-world"
    assert len(sha1) == 64
    with pytest.raises(FileExistsError):
        write_bytes_o_excl(dest, b"other")


def test_copy_file_o_excl_dual_sha(tmp_path: Path):
    src = tmp_path / "src.bin"
    dst = tmp_path / "dst.bin"
    src.write_bytes(b"abc123payload")
    src_sha, dst_sha = copy_file_o_excl(src, dst)
    assert src_sha == dst_sha
    assert dst.read_bytes() == src.read_bytes()


def test_short_write_detected(tmp_path: Path):
    dest = tmp_path / "short.bin"
    real_fdopen = os.fdopen

    def wrapping_fdopen(fd, mode="wb"):
        f = real_fdopen(fd, mode)
        orig_write = f.write

        def short_write(data):
            # Force a zero-length write once to trip short-write detection.
            if not getattr(f, "_p1b_shorted", False):
                f._p1b_shorted = True
                return 0
            return orig_write(data)

        f.write = short_write  # type: ignore[method-assign]
        return f

    with mock.patch("os.fdopen", side_effect=wrapping_fdopen):
        with pytest.raises(ShortWriteError):
            write_bytes_o_excl(dest, b"0123456789")


def test_unknown_residual_stop():
    allowed = {
        "/tmp/allowed/a.json",
        "/tmp/allowed/b.log",
    }
    assert classify_cleanup_residual("/tmp/allowed/a.json", allowed) == "ok"
    assert classify_cleanup_residual("/tmp/allowed/evil.json", allowed) == "STOP_unknown"


def test_log_no_clobber_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Supervisor refuses when phase B log already exists."""
    from scripts.p1b_phase_b_packet_mint import mint_phase_b_packet_o_excl
    from scripts.p1b_phase_b_packet_mint import RUNTIME_EXECUTABLE_KEYS
    import hashlib
    import subprocess

    isol = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158-p1b-isol-wt")
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    monkeypatch.setenv("P1B_ALLOW_TEST_TRAINER_COMMAND", "1")
    monkeypatch.chdir(isol)

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=isol, text=True).strip()
    runtime = {
        rel: hashlib.sha256((isol / rel).read_bytes()).hexdigest()
        for rel in RUNTIME_EXECUTABLE_KEYS
    }
    log_path = tmp_path / "phaseB.log"
    log_path.write_text("preexisting\n", encoding="utf-8")
    fields = {
        "plan_id": "p1b_test",
        "plan_revision": "v15",
        "plan_sha256": "a" * 64,
        "gate_msg_ids": {
            "plus1_implement": "x",
            "implement_gate1": "x",
            "implement_gate2": "x",
            "plus1_commit": "x",
        },
        "commit_sha": head,
        "runtime_executable_sha256s": runtime,
        "watch_wrap_sha256": runtime["bin/watch-wrap"],
        "watch_wrap_command_exact": 'bin/watch-wrap --log <path> --heartbeat 30 --error "Traceback|Error|Killed|OOM|FAILED|assert" --progress "[P1B_PHASE]" --success "TERMINAL_OK" --stop-on "TERMINAL_OK" --replay 20',
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
            "phase_b_log": str(log_path),
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
            "WATCHDOG_ARMED_to_Monitor_armed_sec": 60,
        },
        "kill_semantics": {"target": "trainer_process_group_only"},
    }
    packet_path = tmp_path / "packet.json"
    minted = mint_phase_b_packet_o_excl(packet_path, fields)
    # Preexistence of log is checked as EXIT_PACKET_PATH_PREEXISTS (67) during
    # pre-spawn O_EXCL absence checks — stronger than late LOG_PREEXISTS.
    with pytest.raises(SystemExit) as ei:
        run_supervisor(
            [
                "--test-mode-fixtures",
                "--packet",
                str(packet_path),
                "--packet-sha256",
                minted["packet_file_sha256"],
                "--cwd",
                str(isol),
                "--trainer-command",
                "python3 -c 'pass'",
            ]
        )
    assert ei.value.code in (67, EXIT_LOG_PREEXISTS)
