from __future__ import annotations

import json
from pathlib import Path

from calm.hrm_text_158.native_full_stack.d_recompute_input_manifest_bind import (
    compute_spec_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
V2_REPLAY = REPO_ROOT / "artifacts/consensus_prep/d_recompute_window_feasibility_gpu_launch_packet_v2_replay_commands.json"
V2_DRAFT = REPO_ROOT / "artifacts/consensus_prep/d_recompute_window_feasibility_gpu_launch_packet_v2_draft.json"


def _load_replay() -> dict:
    return json.loads(V2_REPLAY.read_text(encoding="utf-8"))


def test_v2_scale_smoke_command_is_d_on_and_manifest_bound() -> None:
    replay = _load_replay()
    smoke = replay["scale_smoke_command"]
    assert "--d-recompute-window-instrumentation" in smoke
    assert "--d-recompute-selector-manifest" in smoke
    assert "{run_root}/prelaunch/calibrated_selector_manifest.json" in smoke
    assert "--d-diagnostic-compact-step-reports" in smoke
    assert "--receipt-emit-profile s3bb_headroom_diagnostic_slim" in smoke
    assert "--steps 5" in smoke


def test_v2_forbids_d_off_smoke_as_launch_gate() -> None:
    replay = _load_replay()
    forbidden = replay["forbidden_launch_sequence_patterns"]
    assert "scale_smoke_without_d_instrumentation_as_launch_gate" in forbidden
    assert "baseline_d_off_smoke_used_as_launch_eligibility_receipt" in forbidden
    assert "scale_smoke_with_d_instrumentation_enabled" not in forbidden


def test_v2_missing_d_on_flag_matches_forbidden_pattern() -> None:
    replay = _load_replay()
    forbidden_pattern = "scale_smoke_without_d_instrumentation_as_launch_gate"
    bad_smoke = replay["scale_smoke_command"].replace(
        "--d-recompute-window-instrumentation",
        "",
    )
    assert "--d-recompute-window-instrumentation" not in bad_smoke
    assert forbidden_pattern in replay["forbidden_launch_sequence_patterns"]
    assert "d_recompute_window_diagnostic" in replay["scale_smoke_command"]


def test_v2_baseline_d_off_not_in_launch_sequence() -> None:
    replay = _load_replay()
    assert "baseline_liveness_telemetry_command" in replay
    assert "baseline_liveness_telemetry_command" not in replay["launch_sequence"]
    baseline = replay["baseline_liveness_telemetry_command"]
    assert "--d-recompute-window-instrumentation" not in baseline


def test_v2_draft_pins_calibration_policy_and_byte_caps() -> None:
    draft = json.loads(V2_DRAFT.read_text(encoding="utf-8"))
    assert draft["packet_revision"] == "v2_rev4c_stratified_horizon_sizing"
    assert draft["run_id"] == "2189e72015"
    assert draft["confirmation_steps"] == 100
    assert draft["scale_smoke"]["scale_smoke_steps"] == 5
    assert draft["scale_smoke"]["d_on_required"] is True
    assert draft["calibration_policy"]["policy_id"] == "horizon_fixed_warmup_calibrated_v0"
    assert draft["min_free_memory_bytes"] == 1610612736
    assert "PROVISIONAL" in draft["min_free_memory_bytes_label"]
    assert draft["in_vivo_alignment_checklist"]["manifest_selected_keys_must_match_emitted_log_state_keys"]


def test_v2_launch_sequence_orders_calibration_before_d_on_smoke() -> None:
    replay = _load_replay()
    sequence = replay["launch_sequence"]
    assert sequence.index("calibration_warmup_command") < sequence.index(
        "calibration_prepass_command"
    )
    assert sequence.index("calibration_prepass_command") < sequence.index("scale_smoke_command")
    assert sequence.index("scale_smoke_command") < sequence.index("scale_smoke_receipt_command")
    assert sequence.index("scale_smoke_receipt_command") < sequence.index("confirmation_launch_command")


def test_launch_packet_v2_h200_draft_pins_horizon_and_caps() -> None:
    h200_draft = REPO_ROOT / (
        "artifacts/consensus_prep/d_recompute_window_feasibility_gpu_launch_packet_v2_h200_draft.json"
    )
    draft = json.loads(h200_draft.read_text(encoding="utf-8"))
    assert draft["packet_revision"] == "v2_rev4d_h200_decensor"
    assert draft["run_id"] == "2189e72016"
    assert draft["confirmation_steps"] == 200
    assert draft["sizing_horizon_h"] == 200
    assert draft["horizon_ladder"] == [25, 50, 100, 200]
    assert draft["postrun_timeout_seconds"] == 1800
    assert draft["byte_caps"]["extrapolated_h100_receipt_bytes_max"] == 104857600
    assert draft["byte_caps"]["extrapolated_h100_recompute_log_bytes_max"] == 67108864
    assert "still_right_censored_at_h200_is_diagnostic_not_promoted_fork" in draft[
        "explicit_non_claims"
    ]


def test_launch_packet_v2_h200_head_at_packet_not_stale_authoring_sha() -> None:
    h200_draft = REPO_ROOT / (
        "artifacts/consensus_prep/d_recompute_window_feasibility_gpu_launch_packet_v2_h200_draft.json"
    )
    draft = json.loads(h200_draft.read_text(encoding="utf-8"))
    stale_pins = {"a40a31e", "7f77640", "e48a5bb"}
    assert draft["head_at_packet"] not in stale_pins
    assert draft["head_at_packet"].startswith("AUTHORING_BASE_SUPERSEDED__")
    assert draft["run_target_head_pin"]["authoritative_source"] == "claude_launch_dispatch"
    spec = draft["expected_native_input_manifest_spec"]
    assert "head_at_packet" not in spec
    assert spec["spec_sha256"] == "e0b009f1d73bb6cef77ff2366faf1c5e4391357f4ad95923781c5f3cd705935c"


def test_launch_packet_v2_h200_replay_wires_caps_and_timeout() -> None:
    h200_replay = REPO_ROOT / (
        "artifacts/consensus_prep/d_recompute_window_feasibility_gpu_launch_packet_v2_h200_replay_commands.json"
    )
    replay = json.loads(h200_replay.read_text(encoding="utf-8"))
    smoke_receipt = replay["scale_smoke_receipt_command"]
    assert "--confirmation-steps 200" in smoke_receipt
    assert "104857600" in smoke_receipt
    assert "67108864" in smoke_receipt
    postrun = replay["postrun_command"]
    assert "timeout 1800" in postrun
    assert "--timeout-seconds 1800" in postrun
    assert (
        "--emit-timeout-receipt" in postrun
        and "v2_h200_draft.json --emit-timeout-receipt" in postrun
    )
    assert replay["launch_sequence"] == json.loads(
        (REPO_ROOT / "artifacts/consensus_prep/d_recompute_window_feasibility_gpu_launch_packet_v2_replay_commands.json").read_text(
            encoding="utf-8"
        )
    )["launch_sequence"]
    assert "--steps 200 --max-steps-hard 200" in replay["confirmation_launch_command"]
    assert "confirmation_stderr.log" in replay["confirmation_launch_command"]


def test_launch_packet_v2_h200_relaunch_run_id_agreement_and_spec_sha() -> None:
    draft_path = REPO_ROOT / (
        "artifacts/consensus_prep/d_recompute_window_feasibility_gpu_launch_packet_v2_h200_relaunch_draft.json"
    )
    replay_path = REPO_ROOT / (
        "artifacts/consensus_prep/d_recompute_window_feasibility_gpu_launch_packet_v2_h200_relaunch_replay_commands.json"
    )
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    spec = draft["expected_native_input_manifest_spec"]

    assert draft["packet_revision"] == "v2_rev4e_h200_decensor_relaunch"
    assert draft["run_id"] == "2189e72017"
    assert spec["run_id"] == "2189e72017"
    assert spec["packet_revision"] == "v2_rev4e_h200_decensor_relaunch"
    assert replay["run_id"] == "2189e72017"
    assert draft["run_id"] == spec["run_id"] == replay["run_id"]

    assert compute_spec_sha256(spec) == spec["spec_sha256"]
    assert spec["spec_sha256"] == "4f368336ba94852fc4262c5508cfc613b32110179ec8235c0bd4055f85b9646c"

    assert draft["never_rm_run_ids"]["2189e72017"]["never_rm"] is True
    assert "reuse_run_id_2189e72001_through_17" in replay["forbidden_launch_sequence_patterns"]
    assert "2189e72001..17" in draft["run_id_policy"]
    assert draft["confirmation_steps"] == 200
    assert draft["sizing_horizon_h"] == 200
    assert draft["horizon_ladder"] == [25, 50, 100, 200]
    assert draft["postrun_timeout_seconds"] == 1800


def test_launch_packet_v2_h200_relaunch_replay_inherits_s1_tee_and_packet_paths() -> None:
    replay_path = REPO_ROOT / (
        "artifacts/consensus_prep/d_recompute_window_feasibility_gpu_launch_packet_v2_h200_relaunch_replay_commands.json"
    )
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    assert replay["binds_main_packet"].endswith("v2_h200_relaunch_draft.json")
    assert "confirmation_stderr.log" in replay["confirmation_launch_command"]
    assert "tee -a" in replay["confirmation_launch_command"]
    assert "PIPESTATUS[0]" in replay["confirmation_launch_command"]
    postrun = replay["postrun_command"]
    assert "v2_h200_relaunch_draft.json" in postrun
    assert "timeout 1800" in postrun
    assert "--timeout-seconds 1800" in postrun
    smoke_receipt = replay["scale_smoke_receipt_command"]
    assert "--confirmation-steps 200" in smoke_receipt
    assert "104857600" in smoke_receipt
    assert "67108864" in smoke_receipt
    assert (
        replay["run_root"]
        == "/home/gabe/claw-code-creditdir/transient_fp_credit/"
        "d_recompute_window_feasibility_seed43_43_2189e72017/"
    )
