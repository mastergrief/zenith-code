"""Packet self-validate tests for v6i extended-step CLASSIFIER_ONLY launch packet."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DRAFT = REPO / "artifacts/consensus_prep/slice5_step2a_live_carrier_gpu_scale_smoke_launch_packet_v6i_draft.json"
REPLAY = REPO / "artifacts/consensus_prep/slice5_step2a_live_carrier_gpu_scale_smoke_launch_packet_v6i_replay_commands.json"


def _run_self_validate() -> tuple[int, dict]:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    proc = subprocess.run(
        ["bash", "-lc", replay["packet_self_validate_command"]],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    receipt = json.loads(proc.stdout) if proc.stdout.strip() else {}
    return proc.returncode, receipt


def test_v6i_packet_self_validate_passes_clean() -> None:
    exit_code, receipt = _run_self_validate()
    assert exit_code == 0, receipt
    assert receipt.get("pass") is True
    assert receipt.get("packet_revision") == "v6i_extended_step_warmup_classifier"


def test_v6i_draft_classifier_only_and_steps() -> None:
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    assert draft["classifier_only"] is True
    assert draft["source_base_head"] == "3ee73541bb125071b48e46eec5f8d58e833f7e82"
    assert draft["validator_packet_head"] == "fde2051039e80b63aed723e579c8e78419fa17b2"
    assert draft["run_id"] == "2189e7202a"
    assert draft["scale_smoke"]["max_steps_hard"] == 8
    budget = draft["bounded_steps_budget"]
    assert budget["per_step_baseline_seconds"] == 200
    assert budget["per_step_snapshot_premium_seconds"] == 60
    assert budget["per_step_budget_seconds"] == 260
    assert budget["fixed_overhead_seconds"] == 200
    assert budget["phase_timeout_seconds_at_n8"] == 2280
    assert budget["total_timeout_seconds_at_n8"] == 5400
    assert budget["prior_failed_run_id"] == "2189e72029"
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    for key in ("shared_probe_argv", "baseline_gpu_command", "instrumented_gpu_command"):
        text = replay[key]
        assert "--phase-timeout-seconds 2280" in text
        assert "--total-timeout-seconds 5400" in text


def test_v6i_packet_self_validate_fails_on_stale_run_id() -> None:
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    original = draft["run_id"]
    draft["run_id"] = "2189e72028"
    DRAFT.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        exit_code, receipt = _run_self_validate()
        assert exit_code != 0
        assert any("draft_run_id_mismatch" in f for f in receipt.get("failures", []))
    finally:
        draft["run_id"] = original
        DRAFT.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_v6i_packet_self_validate_fails_on_max_steps_hard_not_8() -> None:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    original = replay["baseline_gpu_command"]
    replay["baseline_gpu_command"] = original.replace("--max-steps-hard 8", "--max-steps-hard 6")
    REPLAY.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        exit_code, receipt = _run_self_validate()
        assert exit_code != 0
        assert any("max_steps_hard_not_8_in_baseline_gpu_command" in f for f in receipt.get("failures", []))
    finally:
        replay["baseline_gpu_command"] = original
        REPLAY.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_v6i_packet_self_validate_fails_on_missing_gpu_env_flags() -> None:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    original = replay["instrumented_gpu_command"]
    replay["instrumented_gpu_command"] = original.replace("HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY=1 ", "")
    REPLAY.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        exit_code, receipt = _run_self_validate()
        assert exit_code != 0
        assert any("missing_gpu_q_acc_apply_env_in_instrumented_gpu_command" in f for f in receipt.get("failures", []))
    finally:
        replay["instrumented_gpu_command"] = original
        REPLAY.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_v6i_packet_self_validate_fails_on_classifier_only_not_true() -> None:
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    original = draft["classifier_only"]
    draft["classifier_only"] = False
    DRAFT.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        exit_code, receipt = _run_self_validate()
        assert exit_code != 0
        assert any("classifier_only_not_true" in f for f in receipt.get("failures", []))
    finally:
        draft["classifier_only"] = original
        DRAFT.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_v6i_packet_self_validate_fails_on_stale_v6h_run_id_reference() -> None:
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    original = draft["decision_contract"]["chosen_path"]
    draft["decision_contract"]["chosen_path"] = original + " 2189e72028"
    DRAFT.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        exit_code, receipt = _run_self_validate()
        assert exit_code != 0
        assert any("stale_active_reference:2189e72028" in f for f in receipt.get("failures", []))
    finally:
        draft["decision_contract"]["chosen_path"] = original
        DRAFT.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_v6i_launch_sequence_includes_dispatch_witness_before_gpu_arms() -> None:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    seq = replay["launch_sequence"]
    inj = seq.index("launch_injected_dispatch_witness_command")
    assert inj < seq.index("baseline_gpu_command")
    assert seq.index("v6i_postrun_acceptance_validate_command") > seq.index(
        "live_carrier_scale_smoke_receipt_command"
    )


def test_v6i_packet_self_validate_fails_on_stale_v6i_validator_pin() -> None:
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    original_draft_pins = dict(draft["source_pins_at_launch"])
    original_replay = replay["file_content_pin_witness_command"]
    stale_pins = dict(original_draft_pins)
    stale_pins["scripts/hrm_text_158_slice5_v6i_postrun_acceptance_validate.py"] = (
        "0000000000000000000000000000000000000000000000000000000000000000"
    )
    draft["source_pins_at_launch"] = stale_pins
    DRAFT.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        exit_code, receipt = _run_self_validate()
        assert exit_code != 0
        assert any(
            failure.startswith("source_pins_on_disk_sha_mismatch:")
            or failure.startswith("file_content_pin_witness_command_on_disk_sha_mismatch:")
            or failure.startswith("prelaunch_dry_run_failed:file_content_pin_witness_command")
            for failure in receipt.get("failures", [])
        )
    finally:
        draft["source_pins_at_launch"] = original_draft_pins
        DRAFT.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_v6i_packet_self_validate_fails_when_v6h_dependency_missing_from_pins() -> None:
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    original_pins = dict(draft["source_pins_at_launch"])
    trimmed_pins = {
        k: v
        for k, v in original_pins.items()
        if k != "scripts/hrm_text_158_slice5_v6h_postrun_acceptance_validate.py"
    }
    draft["source_pins_at_launch"] = trimmed_pins
    DRAFT.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        exit_code, receipt = _run_self_validate()
        assert exit_code != 0
        assert any(
            failure == "source_pins_missing_v6h_validator_dependency"
            or failure.startswith("file_content_pin_witness_command_missing_launch_critical:")
            or failure.startswith("prelaunch_dry_run_failed:file_content_pin_witness_command")
            for failure in receipt.get("failures", [])
        )
    finally:
        draft["source_pins_at_launch"] = original_pins
        DRAFT.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_v6i_packet_self_validate_fails_on_stale_phase_timeout_900() -> None:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    original = replay["shared_probe_argv"]
    replay["shared_probe_argv"] = original.replace(
        "--phase-timeout-seconds 2280", "--phase-timeout-seconds 900"
    )
    REPLAY.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        exit_code, receipt = _run_self_validate()
        assert exit_code != 0
        assert any(
            "stale_phase_timeout_900" in f or "missing_phase_timeout_2280" in f
            for f in receipt.get("failures", [])
        )
    finally:
        replay["shared_probe_argv"] = original
        REPLAY.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_v6i_packet_self_validate_fails_on_stale_total_timeout_3600() -> None:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    original = replay["shared_probe_argv"]
    replay["shared_probe_argv"] = original.replace(
        "--total-timeout-seconds 5400", "--total-timeout-seconds 3600"
    )
    REPLAY.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        exit_code, receipt = _run_self_validate()
        assert exit_code != 0
        assert any(
            "stale_total_timeout_3600" in f or "missing_total_timeout_5400" in f
            for f in receipt.get("failures", [])
        )
    finally:
        replay["shared_probe_argv"] = original
        REPLAY.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_v6i_source_pins_include_both_v6i_authority_and_v6h_dependency() -> None:
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    pins = draft["source_pins_at_launch"]
    assert "scripts/hrm_text_158_slice5_v6i_postrun_acceptance_validate.py" in pins
    assert "scripts/hrm_text_158_slice5_v6h_postrun_acceptance_validate.py" in pins
    assert len(pins) == 12
