from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import make_bounded_tensor_state
from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    NarrowCarrierHeadroomBreach,
    W6_SIGNED_MAX,
    W6_SIGNED_MIN,
    pack_w6_tensor,
)
from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    CLASSIFIER_HARNESS_OR_LIVENESS_FAIL,
    CLASSIFIER_HEADROOM_BREACH,
    CLASSIFIER_W6_DYNAMICS_DIVERGES,
    CLASSIFIER_W6_HEADROOM_SUFFICIENT_PARITY_OK,
    HEADROOM_TELEMETRY_SCHEMA_VERSION,
    MEASURED_STEPS_REQUIRED,
    REQUIRED_HEADROOM_TELEMETRY_FIELDS,
    S3BB_W6_HEADROOM_DIAGNOSTIC_PHASE,
    W6_HEADROOM_K_DEFAULT,
    classify_s3bb_run,
    compute_headroom_telemetry_from_accumulators,
    run_vote_materialization_with_s3bb_boundary_catch,
    validate_headroom_telemetry_block,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    attach_s3bb_headroom_telemetry_to_step_report,
)
from scripts.hrm_text_158_s3bb_w6_dynamics_postrun import (
    ORACLE_ARM_DIR,
    TREATMENT_ARM_DIR,
    preflight_arm_receipt_dirs,
    run_postrun,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _telemetry_from_values(values: list[int]) -> dict:
    acc = torch.tensor(values, dtype=torch.int16).reshape(1, -1)
    return compute_headroom_telemetry_from_accumulators(acc, k=W6_HEADROOM_K_DEFAULT)


def _parity_step_report(
    *,
    step_id: str,
    acc_values: list[int],
    q_values: list[int] | None = None,
) -> dict:
    q_values = q_values or [0] * len(acc_values)
    telemetry = _telemetry_from_values(acc_values)
    telemetry["accumulator_snapshots_by_state_key"] = {
        "tiny.proj": acc_values,
    }
    telemetry["q_snapshots_by_state_key"] = {
        "tiny.proj": q_values,
    }
    return {
        "headroom_telemetry": telemetry,
        "step_id": step_id,
    }


def _receipt(
    *,
    steps_completed: int,
    stop_reason: str = "",
    step_reports: dict[str, dict] | None = None,
) -> dict:
    return {
        "steps_completed": steps_completed,
        "stop_reason": stop_reason,
        "step_reports": step_reports or {},
    }


def _build_parity_receipts(steps: int = MEASURED_STEPS_REQUIRED) -> tuple[dict, dict]:
    step_reports: dict[str, dict] = {}
    for step in range(1, steps + 1):
        step_reports[str(step)] = _parity_step_report(
            step_id=str(step),
            acc_values=[5, -9, 21],
            q_values=[0, 1, -1],
        )
    return (
        _receipt(steps_completed=steps, step_reports=dict(step_reports)),
        _receipt(steps_completed=steps, step_reports=copy.deepcopy(step_reports)),
    )


def test_t1_telemetry_field_correctness() -> None:
    values = [5, -9, 21, 31, -31]
    acc = torch.tensor(values, dtype=torch.int16).reshape(1, -1)
    telemetry = compute_headroom_telemetry_from_accumulators(acc, k=W6_HEADROOM_K_DEFAULT)

    assert telemetry["schema_version"] == HEADROOM_TELEMETRY_SCHEMA_VERSION
    assert telemetry["global_max_abs_accumulator"] == 31
    assert telemetry["margin_to_w6_boundary_min"] == 0
    assert telemetry["out_of_domain_lane_count"] == 0
    assert telemetry["would_strict_raise_step"] is False
    assert telemetry["strict_raise_count"] == 0
    boundary_threshold = W6_SIGNED_MAX - W6_HEADROOM_K_DEFAULT
    near_boundary = sum(1 for value in values if abs(value) >= boundary_threshold)
    expected_fraction = near_boundary / len(values)
    assert telemetry["lanes_within_K_of_boundary_fraction"] == pytest.approx(
        expected_fraction
    )


def test_t2_observe_path_never_raises() -> None:
    acc = torch.tensor([[100, -100, 0]], dtype=torch.int16)
    telemetry = compute_headroom_telemetry_from_accumulators(acc)
    assert telemetry["would_strict_raise_step"] is True
    assert telemetry["strict_raise_count"] == 1
    assert telemetry["out_of_domain_lane_count"] == 2
    validate_headroom_telemetry_block(telemetry)


def test_t3_device_preserving() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    acc = torch.tensor([[5, -9, 21]], dtype=torch.int16, device="cuda")
    telemetry = compute_headroom_telemetry_from_accumulators(acc)
    assert telemetry["global_max_abs_accumulator"] == 21


def test_t4_classifier_headroom_before_harness() -> None:
    oracle, treatment = _build_parity_receipts(steps=3)
    treatment["step_reports"]["3"]["headroom_telemetry"]["strict_raise_count"] = 1
    treatment["step_reports"]["3"]["headroom_telemetry"]["would_strict_raise_step"] = True
    treatment["steps_completed"] = 3
    assert classify_s3bb_run(oracle, treatment) == CLASSIFIER_HEADROOM_BREACH


def test_t5_classifier_harness_early_stop() -> None:
    oracle, treatment = _build_parity_receipts(steps=3)
    treatment["steps_completed"] = 3
    treatment["stop_reason"] = "nan_inf"
    assert classify_s3bb_run(oracle, treatment) == CLASSIFIER_HARNESS_OR_LIVENESS_FAIL


def test_t6_classifier_wiring_diverges() -> None:
    oracle, treatment = _build_parity_receipts(steps=MEASURED_STEPS_REQUIRED)
    treatment["step_reports"]["4"]["headroom_telemetry"]["accumulator_snapshots_by_state_key"][
        "tiny.proj"
    ] = [6, -9, 21]
    assert classify_s3bb_run(oracle, treatment) == CLASSIFIER_W6_DYNAMICS_DIVERGES


def test_t7_classifier_parity_ok() -> None:
    oracle, treatment = _build_parity_receipts(steps=MEASURED_STEPS_REQUIRED)
    assert (
        classify_s3bb_run(oracle, treatment)
        == CLASSIFIER_W6_HEADROOM_SUFFICIENT_PARITY_OK
    )


def test_t8_phase_gate_negative() -> None:
    state = make_bounded_tensor_state(
        "tiny.proj",
        torch.tensor([[0, 1, -1]], dtype=torch.int8),
        1.0,
        torch.tensor([[5, -9, 21]], dtype=torch.int16),
    )
    report: dict = {"step_id": "1"}
    result = attach_s3bb_headroom_telemetry_to_step_report(
        report,
        phase="c2p1-real-model-smoke",
        post_update_states={"tiny.proj": state},
    )
    assert result is report
    assert "headroom_telemetry" not in report


def test_t8b_phase_gate_affirmative() -> None:
    state = make_bounded_tensor_state(
        "tiny.proj",
        torch.tensor([[0, 1, -1]], dtype=torch.int8),
        1.0,
        torch.tensor([[5, -9, 21]], dtype=torch.int16),
    )
    report: dict = {"step_id": "1"}
    attach_s3bb_headroom_telemetry_to_step_report(
        report,
        phase=S3BB_W6_HEADROOM_DIAGNOSTIC_PHASE,
        post_update_states={"tiny.proj": state},
    )
    telemetry = report["headroom_telemetry"]
    for field in REQUIRED_HEADROOM_TELEMETRY_FIELDS:
        assert field in telemetry
    validate_headroom_telemetry_block(telemetry)


def test_t9_postrun_cli_schema(tmp_path: Path) -> None:
    oracle, treatment = _build_parity_receipts(steps=MEASURED_STEPS_REQUIRED)
    oracle_dir = tmp_path / "int16_oracle_flag_off"
    treatment_dir = tmp_path / "w6_carrier_flag_on"
    oracle_dir.mkdir(parents=True)
    treatment_dir.mkdir(parents=True)
    oracle_dir.joinpath("receipt.json").write_text(
        json.dumps(oracle, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    treatment_dir.joinpath("receipt.json").write_text(
        json.dumps(treatment, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    classifier_path = tmp_path / "classifier_receipt.json"
    receipt = run_postrun(run_root=tmp_path, json_out=classifier_path)
    assert classifier_path.is_file()
    assert (tmp_path / "s3bb_headroom_summary.json").is_file()
    assert receipt["primary_classifier"] == CLASSIFIER_W6_HEADROOM_SUFFICIENT_PARITY_OK

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "hrm_text_158_s3bb_w6_dynamics_postrun.py"),
            "--run-root",
            str(tmp_path),
            "--json-out",
            str(tmp_path / "classifier_receipt_cli.json"),
        ],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    cli_payload = json.loads(
        (tmp_path / "classifier_receipt_cli.json").read_text(encoding="utf-8")
    )
    assert "primary_classifier" in cli_payload


def test_postrun_explicit_r4_arm_dirs(tmp_path: Path) -> None:
    oracle, treatment = _build_parity_receipts(steps=MEASURED_STEPS_REQUIRED)
    oracle_dir = tmp_path / "w6_on_q_off_oracle"
    treatment_dir = tmp_path / "w6_on_q_on_treatment"
    oracle_dir.mkdir(parents=True)
    treatment_dir.mkdir(parents=True)
    oracle_dir.joinpath("receipt.json").write_text(
        json.dumps(oracle, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    treatment_dir.joinpath("receipt.json").write_text(
        json.dumps(treatment, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    classifier_path = tmp_path / "classifier_receipt.json"
    receipt = run_postrun(
        run_root=tmp_path,
        json_out=classifier_path,
        oracle_arm_dir="w6_on_q_off_oracle",
        treatment_arm_dir="w6_on_q_on_treatment",
    )
    assert receipt["primary_classifier"] == CLASSIFIER_W6_HEADROOM_SUFFICIENT_PARITY_OK
    summary = json.loads((tmp_path / "s3bb_headroom_summary.json").read_text(encoding="utf-8"))
    assert summary["oracle_arm"] == "w6_on_q_off_oracle"
    assert summary["treatment_arm"] == "w6_on_q_on_treatment"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "hrm_text_158_s3bb_w6_dynamics_postrun.py"),
            "--run-root",
            str(tmp_path),
            "--json-out",
            str(tmp_path / "classifier_receipt_cli_r4.json"),
            "--oracle-arm-dir",
            "w6_on_q_off_oracle",
            "--treatment-arm-dir",
            "w6_on_q_on_treatment",
        ],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0


def test_postrun_default_legacy_arm_dir_names(tmp_path: Path) -> None:
    oracle, treatment = _build_parity_receipts(steps=MEASURED_STEPS_REQUIRED)
    (tmp_path / ORACLE_ARM_DIR).mkdir(parents=True)
    (tmp_path / TREATMENT_ARM_DIR).mkdir(parents=True)
    (tmp_path / ORACLE_ARM_DIR / "receipt.json").write_text(
        json.dumps(oracle, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (tmp_path / TREATMENT_ARM_DIR / "receipt.json").write_text(
        json.dumps(treatment, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    receipt = run_postrun(
        run_root=tmp_path,
        json_out=tmp_path / "classifier_receipt.json",
    )
    assert receipt["primary_classifier"] == CLASSIFIER_W6_HEADROOM_SUFFICIENT_PARITY_OK
    summary = json.loads((tmp_path / "s3bb_headroom_summary.json").read_text(encoding="utf-8"))
    assert summary["oracle_arm"] == ORACLE_ARM_DIR
    assert summary["treatment_arm"] == TREATMENT_ARM_DIR


def test_postrun_preflight_raises_on_missing_arm_dir(tmp_path: Path) -> None:
    oracle, _treatment = _build_parity_receipts(steps=MEASURED_STEPS_REQUIRED)
    (tmp_path / ORACLE_ARM_DIR).mkdir(parents=True)
    (tmp_path / ORACLE_ARM_DIR / "receipt.json").write_text(
        json.dumps(oracle, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="postrun arm receipt preflight failed"):
        preflight_arm_receipt_dirs(
            tmp_path,
            oracle_arm_dir=ORACLE_ARM_DIR,
            treatment_arm_dir=TREATMENT_ARM_DIR,
        )
    with pytest.raises(FileNotFoundError, match="postrun arm receipt preflight failed"):
        run_postrun(
            run_root=tmp_path,
            json_out=tmp_path / "classifier_receipt.json",
        )


def test_t10_boundary_catch_path_s3bb_phase() -> None:
    step_report: dict = {}

    def _raise_boundary() -> None:
        pack_w6_tensor(torch.tensor([100], dtype=torch.int16))

    outcome = run_vote_materialization_with_s3bb_boundary_catch(
        phase=S3BB_W6_HEADROOM_DIAGNOSTIC_PHASE,
        step_report=step_report,
        materialize=_raise_boundary,
    )
    assert outcome.terminated is True
    assert outcome.stop_reason == "headroom_breach"
    assert step_report["headroom_telemetry"]["boundary_value_error_caught"] is True

    oracle, treatment = _build_parity_receipts(steps=3)
    treatment["steps_completed"] = 3
    treatment["stop_reason"] = "headroom_breach"
    treatment["step_reports"]["3"]["headroom_telemetry"]["boundary_value_error_caught"] = True
    treatment["step_reports"]["3"]["headroom_telemetry"]["strict_raise_count"] = 1
    treatment["step_reports"]["3"]["headroom_telemetry"]["would_strict_raise_step"] = True
    assert classify_s3bb_run(oracle, treatment) == CLASSIFIER_HEADROOM_BREACH


def test_t11_boundary_catch_non_s3bb_propagates() -> None:
    def _raise_boundary() -> None:
        pack_w6_tensor(torch.tensor([100], dtype=torch.int16))

    with pytest.raises(NarrowCarrierHeadroomBreach, match="pack_w6_tensor requires all values in"):
        run_vote_materialization_with_s3bb_boundary_catch(
            phase="c2p1-real-model-smoke",
            step_report={},
            materialize=_raise_boundary,
        )


def test_t12_s3bb_phase_non_headroom_value_error_propagates() -> None:
    step_report: dict = {}

    def _raise_harness_defect() -> None:
        raise ValueError("vote_specs shape mismatch")

    with pytest.raises(ValueError, match="vote_specs shape mismatch"):
        run_vote_materialization_with_s3bb_boundary_catch(
            phase=S3BB_W6_HEADROOM_DIAGNOSTIC_PHASE,
            step_report=step_report,
            materialize=_raise_harness_defect,
        )
    assert "headroom_telemetry" not in step_report

    def _raise_dtype_defect() -> None:
        raise ValueError("accumulators must be torch.int16, got torch.float32")

    with pytest.raises(ValueError, match="accumulators must be torch.int16"):
        run_vote_materialization_with_s3bb_boundary_catch(
            phase=S3BB_W6_HEADROOM_DIAGNOSTIC_PHASE,
            step_report=step_report,
            materialize=_raise_dtype_defect,
        )

    oracle, treatment = _build_parity_receipts(steps=3)
    treatment["steps_completed"] = 3
    treatment["stop_reason"] = "harness_exception"
    assert classify_s3bb_run(oracle, treatment) == CLASSIFIER_HARNESS_OR_LIVENESS_FAIL


def test_t12b_narrow_carrier_headroom_breach_is_value_error_subclass() -> None:
    with pytest.raises(ValueError):
        pack_w6_tensor(torch.tensor([100], dtype=torch.int16))
    with pytest.raises(NarrowCarrierHeadroomBreach):
        pack_w6_tensor(torch.tensor([100], dtype=torch.int16))
