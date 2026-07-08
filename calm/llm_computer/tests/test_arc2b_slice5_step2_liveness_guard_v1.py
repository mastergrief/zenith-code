"""CPU-static tests for Slice-5 Step-2 liveness-guard packet threading."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from scripts.apply_arc2b_slice5_step2_in_vivo_gpu_launch_packet import (
    EVENT_CODED_INCOMPATIBLE_FLAGS,
    HEAD,
    REPO,
    git_head,
    STEP2_CARRIER_REQUIRED,
    STEP2_DECAY_FLAGS,
    STEP2_EVENT_CODED_RECOMPUTE_WINDOW_LOG_FLAG,
    STEP2_FORBIDDEN_LAUNCH_SEQUENCE_PATTERNS,
    STEP2_MAX_SILENT_PHASE_SECONDS,
    _build_calibration_warmup_retry_command,
    _strip_event_coded_incompatible_flags,
    build_liveness_retry_policy,
    build_packet,
    build_replay_commands,
    execute_calibration_warmup_retry,
    self_verify,
    sha256_file,
    verify_event_coded_incompatible_flags_absent,
    verify_event_coded_recompute_window_log_flag,
    verify_explicit_max_silent_phase_seconds,
    verify_forbidden_launch_sequence_patterns_reconciled,
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
    assert packet["git_head_required"] == "24c19521e6b453dcb011a1dd57fdc37312196e28"


def test_live_git_head_matches_head_constant() -> None:
    assert git_head() == HEAD


def test_self_verify_fails_when_live_git_head_mismatches_head_constant(
    monkeypatch,
) -> None:
    import scripts.apply_arc2b_slice5_step2_in_vivo_gpu_launch_packet as apply_mod

    monkeypatch.setattr(apply_mod, "git_head", lambda: "0" * 40)
    result = self_verify()
    assert result["pins_match_commit"] is False
    assert "pins_match_commit" in result["failures"]
    assert result["ok"] is False


def test_event_coded_measurement_commands_have_no_incompatible_flags() -> None:
    classifier_sha = sha256_file(CLASSIFIER_MODULE)
    replay = build_replay_commands(classifier_sha)
    failures = verify_event_coded_incompatible_flags_absent(replay)
    assert failures == []
    for key in ("scale_smoke_command", "confirmation_launch_command", "shared_probe_argv"):
        body = str(replay[key])
        assert "--event-coded-sparse-vote-authority" in body
        for flag in EVENT_CODED_INCOMPATIBLE_FLAGS:
            assert flag not in body
        for flag in STEP2_CARRIER_REQUIRED + STEP2_DECAY_FLAGS:
            assert flag in body
        assert STEP2_EVENT_CODED_RECOMPUTE_WINDOW_LOG_FLAG in body


def test_event_coded_recompute_window_log_flag_present_on_measurement_only() -> None:
    classifier_sha = sha256_file(CLASSIFIER_MODULE)
    replay = build_replay_commands(classifier_sha)
    failures = verify_event_coded_recompute_window_log_flag(replay)
    assert failures == []
    for key in ("scale_smoke_command", "confirmation_launch_command", "shared_probe_argv"):
        assert STEP2_EVENT_CODED_RECOMPUTE_WINDOW_LOG_FLAG in str(replay[key])
    warmup = str(replay.get("calibration_warmup_command") or "")
    assert STEP2_EVENT_CODED_RECOMPUTE_WINDOW_LOG_FLAG not in warmup
    assert "--d-recompute-window-instrumentation" not in str(replay["scale_smoke_command"])


def test_forbidden_launch_sequence_patterns_reconciled_for_event_coded() -> None:
    classifier_sha = sha256_file(CLASSIFIER_MODULE)
    replay = build_replay_commands(classifier_sha)
    failures = verify_forbidden_launch_sequence_patterns_reconciled(replay)
    assert failures == []
    patterns = list(replay["forbidden_launch_sequence_patterns"])
    assert patterns == list(STEP2_FORBIDDEN_LAUNCH_SEQUENCE_PATTERNS)
    assert "confirmation_without_d_recompute_window_instrumentation_flag" not in patterns
    assert "scale_smoke_without_d_instrumentation_as_launch_gate" not in patterns
    assert "event_coded_measurement_with_d_recompute_window_instrumentation" in patterns
    assert "event_coded_measurement_with_probe_incompatible_flags" in patterns


def test_warmup_producer_argv_remains_w8_d_calibration() -> None:
    argv = build_calibration_warmup_probe_argv(
        run_root=Path("/tmp/run"),
        parent=Path("parent.pt"),
        parent_sha256="abc",
        warmup_steps=5,
        observations_out=Path("/tmp/run/prelaunch/calibration_warmup_observations.json"),
    )
    assert "--dense-accumulator-w8-clip" in argv
    assert "--d-recompute-window-instrumentation" in argv
    assert "--d-recompute-calibration-warmup-out" in argv
    assert "--event-coded-sparse-vote-authority" not in argv


def test_strip_event_coded_incompatible_flags_handles_boolean_and_value_taking() -> None:
    command = (
        "probe.py --phase d-recompute-window-feasibility "
        "--d-recompute-window-instrumentation "
        "--d-recompute-calibration-warmup-out /tmp/out.json "
        "--event-coded-sparse-vote-authority"
    )
    stripped = _strip_event_coded_incompatible_flags(command)
    assert "--d-recompute-window-instrumentation" not in stripped
    assert "--d-recompute-calibration-warmup-out" not in stripped
    assert "/tmp/out.json" not in stripped
    assert "--event-coded-sparse-vote-authority" in stripped


def test_self_verify_passes_after_render() -> None:
    result = self_verify()
    assert result["ok"] is True
    assert result["deterministic_regen"] is True
    assert result["failures"] == []
