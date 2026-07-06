"""CPU-static tests for Slice-5 Step-2 liveness-guard packet threading."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from scripts.apply_arc2b_slice5_step2_in_vivo_gpu_launch_packet import (
    HEAD,
    REPO,
    STEP2_MAX_SILENT_PHASE_SECONDS,
    _build_calibration_warmup_retry_command,
    build_liveness_retry_policy,
    build_packet,
    build_replay_commands,
    execute_calibration_warmup_retry,
    self_verify,
    sha256_file,
    verify_explicit_max_silent_phase_seconds,
    verify_warmup_retry_metadata,
)
from scripts.hrm_text_158_d_recompute_calibration_warmup_producer import (
    build_calibration_warmup_probe_argv,
)

CLASSIFIER_MODULE = REPO / "calm/hrm_text_158/native_full_stack/arc2b_slice5_in_vivo_branch.py"


def test_warmup_producer_passthrough_max_silent_phase_seconds() -> None:
    argv = build_calibration_warmup_probe_argv(
        run_root=Path("/tmp/run"),
        parent=Path("parent.pt"),
        parent_sha256="abc",
        warmup_steps=5,
        observations_out=Path("/tmp/run/prelaunch/calibration_warmup_observations.json"),
        max_silent_phase_seconds=600,
    )
    assert "--max-silent-phase-seconds" in argv
    assert argv[argv.index("--max-silent-phase-seconds") + 1] == "600"


def test_warmup_producer_default_preserves_no_explicit_flag() -> None:
    argv = build_calibration_warmup_probe_argv(
        run_root=Path("/tmp/run"),
        parent=Path("parent.pt"),
        parent_sha256="abc",
        warmup_steps=5,
        observations_out=Path("/tmp/run/prelaunch/calibration_warmup_observations.json"),
    )
    assert "--max-silent-phase-seconds" not in argv


def test_replay_commands_carry_explicit_600s_guard() -> None:
    classifier_sha = sha256_file(CLASSIFIER_MODULE)
    replay = build_replay_commands(classifier_sha)
    failures = verify_explicit_max_silent_phase_seconds(replay)
    assert failures == []
    for key in (
        "calibration_warmup_command",
        "scale_smoke_command",
        "confirmation_launch_command",
        "shared_probe_argv",
    ):
        assert f"--max-silent-phase-seconds {STEP2_MAX_SILENT_PHASE_SECONDS}" in str(
            replay[key]
        )


def test_retry_metadata_present_and_liveness_specific() -> None:
    classifier_sha = sha256_file(CLASSIFIER_MODULE)
    replay = build_replay_commands(classifier_sha)
    packet = build_packet(classifier_sha, "deadbeef")
    failures = verify_warmup_retry_metadata(replay, packet)
    assert failures == []
    warmup = str(replay["calibration_warmup_command"])
    assert "execute_calibration_warmup_retry" in warmup
    assert build_liveness_retry_policy()["max_attempts"] == 2


def _write_liveness_phase(run_root: Path) -> None:
    phase_dir = run_root / "calibration_warmup"
    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / "last_active_phase.json").write_text(
        json.dumps({"failure_class": "LIVENESS_FAILURE"}),
        encoding="utf-8",
    )


def test_execute_calibration_warmup_retry_success_on_first_attempt(tmp_path: Path) -> None:
    witness = execute_calibration_warmup_retry(
        run_root=tmp_path,
        producer_command_template="true",
        scratch_wipe_template="true",
        producer_runner=lambda _cmd: 0,
    )
    assert witness["final_reason"] == "success"
    assert witness["retry_used"] is False
    assert witness["final_rc"] == 0


def test_execute_calibration_warmup_retry_liveness_then_success(tmp_path: Path) -> None:
    calls = {"count": 0}

    def runner(_cmd: str) -> int:
        calls["count"] += 1
        if calls["count"] == 1:
            _write_liveness_phase(tmp_path)
            return 1
        return 0

    witness = execute_calibration_warmup_retry(
        run_root=tmp_path,
        producer_command_template="false",
        scratch_wipe_template=f"rm -rf {tmp_path}/calibration_warmup && mkdir -p {tmp_path}/calibration_warmup",
        producer_runner=runner,
    )
    assert witness["final_reason"] == "success"
    assert witness["retry_used"] is True
    assert witness["scratch_wiped_between_attempts"] is True
    assert witness["attempts_used"] == 2


def test_execute_calibration_warmup_retry_liveness_exhausted(tmp_path: Path) -> None:
    def runner(_cmd: str) -> int:
        _write_liveness_phase(tmp_path)
        return 1

    witness = execute_calibration_warmup_retry(
        run_root=tmp_path,
        producer_command_template="false",
        scratch_wipe_template=f"rm -rf {tmp_path}/calibration_warmup && mkdir -p {tmp_path}/calibration_warmup",
        producer_runner=runner,
    )
    assert witness["final_reason"] == "liveness_failure_exhausted_retries"
    assert witness["retry_used"] is True
    assert witness["final_rc"] == 1


def test_execute_calibration_warmup_retry_non_liveness_no_retry(tmp_path: Path) -> None:
    witness = execute_calibration_warmup_retry(
        run_root=tmp_path,
        producer_command_template="false",
        scratch_wipe_template="true",
        producer_runner=lambda _cmd: 1,
    )
    assert witness["final_reason"] == "non_liveness_failure_no_retry"
    assert witness["retry_used"] is False
    assert witness["attempts_used"] == 1


def test_rendered_wrapper_executes_multiword_producer(tmp_path: Path) -> None:
    marker = tmp_path / "prelaunch" / "wrapper_marker.txt"
    producer = (
        "python3 -c "
        "\"import pathlib; pathlib.Path('{run_root}/prelaunch/wrapper_marker.txt')"
        ".parent.mkdir(parents=True, exist_ok=True); "
        "pathlib.Path('{run_root}/prelaunch/wrapper_marker.txt').write_text('ok')\""
    )
    scratch = "rm -rf {run_root}/calibration_warmup && mkdir -p {run_root}/calibration_warmup"
    wrapper = _build_calibration_warmup_retry_command(producer, scratch).replace(
        "{run_root}",
        str(tmp_path),
    )
    proc = subprocess.run(
        wrapper,
        shell=True,
        cwd=REPO,
        check=False,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    assert proc.returncode == 0
    assert marker.is_file()
    witness = json.loads(
        (tmp_path / "prelaunch" / "calibration_warmup_retry_witness.json").read_text()
    )
    assert witness["final_reason"] == "success"
    assert witness["retry_used"] is False


def test_packet_git_head_required_matches_current_head_constant() -> None:
    classifier_sha = sha256_file(CLASSIFIER_MODULE)
    packet = build_packet(classifier_sha, "deadbeef")
    assert packet["git_head_required"] == HEAD
    assert packet["git_head_required"] == "9185823d0f8b1dd1a0352661cac9a8633bc04ddf"


def test_self_verify_passes_after_render() -> None:
    result = self_verify()
    assert result["ok"] is True
    assert result["deterministic_regen"] is True
    assert result["failures"] == []
