from __future__ import annotations

import json
from pathlib import Path


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
