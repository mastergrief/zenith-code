from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import scripts.hrm_text_158_bounded_delta_acquisition_probe as probe


def test_install_probe_durable_run_log_mirrors_stderr_to_run_log(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    log_path = probe.install_probe_durable_run_log(scratch)
    print("mirror-stream-survivability-marker", flush=True)
    sys.stderr.write("stderr-survivability-marker\n")
    sys.stderr.flush()
    text = log_path.read_text(encoding="utf-8")
    assert "mirror-stream-survivability-marker" in text
    assert "stderr-survivability-marker" in text


def test_register_probe_faulthandler_targets_run_log_not_mirrored_stderr_fileno(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    log_path = probe.install_probe_durable_run_log(scratch)
    report = probe.register_probe_faulthandler(run_log_path=log_path)
    assert report["traceback_target"] == str(log_path)
    assert report["enabled_after"] is True


def test_phase_progress_writes_liveness_stack_dump_before_timer_arm(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    last_active = scratch / "last_active_phase.json"
    progress = probe.PhaseProgress(
        enabled=False,
        device=torch.device("cpu"),
        silent_phase_timeout_seconds=300.0,
        last_active_phase_path=last_active,
        arm_faulthandler_timer=False,
    )
    record = {"phase": "step_update", "fields": {"step": 73}}
    progress._arm_current_phase(record, guard_event="enter")
    dump_path = scratch / "liveness_stack_dump.txt"
    assert dump_path.is_file()
    dump_text = dump_path.read_text(encoding="utf-8")
    assert "guard_event=enter" in dump_text
    assert "phase=step_update" in dump_text
    payload = json.loads(dump_text.splitlines()[2])
    assert payload["guard_event"] == "enter"
    assert payload["liveness_failure"] is False
    assert "failure_class" not in payload


def test_h200_replay_confirmation_command_tees_confirmation_stderr_log() -> None:
    replay_path = (
        Path(__file__).resolve().parents[3]
        / "artifacts/consensus_prep/d_recompute_window_feasibility_gpu_launch_packet_v2_h200_replay_commands.json"
    )
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    cmd = replay["confirmation_launch_command"]
    assert "confirmation_stderr.log" in cmd
    assert "tee -a {run_root}/d_recompute_window_diagnostic/confirmation_stderr.log" in cmd
    assert "PIPESTATUS[0]" in cmd
