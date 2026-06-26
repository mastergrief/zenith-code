from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.s3bb_decision_parity import (
    CLASSIFIER_DECISION_MISMATCH,
    CLASSIFIER_DECISION_PARITY_OK,
    CLASSIFIER_DOMAIN_OR_HEADROOM_FAIL,
    CLASSIFIER_FLIP_EQUIVALENT_DYNAMICS_DRIFT,
    CLASSIFIER_HARNESS_OR_LIVENESS_FAIL,
    classify_s3bb_decision_parity_run,
)
from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    MEASURED_STEPS_REQUIRED,
    append_headroom_wiring_sidecar_chunk,
    compare_arm_wiring_guards_streaming,
    diagnose_sidecar_coverage,
)
from calm.hrm_text_158.native_full_stack.w7_dense_acc_in_vivo_confirmation import (
    CLASSIFIER_RUN_HEALTH_FAIL,
    CLASSIFIER_W7_BREAKS_LIVE_PARITY,
    CLASSIFIER_W7_IN_VIVO_CONFIRMED,
    CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24,
    CPU_PHASE0_STRUCTURAL_FLOOR_WIDTH,
    derive_w7_parity_inputs,
    classify_w7_in_vivo_dual_arm,
    resolve_confirmation_envelope,
    verify_dual_arm_w7_configuration,
)
from scripts.hrm_text_158_w7_dense_acc_in_vivo_postrun import emit_w7_in_vivo_classifier_receipt


def _step_report(
    *,
    step_id: str,
    vote_abs_max: int = 24,
    applied_indices: list[int] | None = None,
    q_sha_after: str = "same_sha",
    q_changed_count: int = 2,
) -> dict:
    applied_indices = applied_indices if applied_indices is not None else [0, 2]
    return {
        "step_id": step_id,
        "q_changed_count": q_changed_count,
        "metrics": {"loss": 0.5, "accuracy": 0.25},
        "vote_pressure": {
            "tiny.proj": {
                "vote_abs_max": int(vote_abs_max),
            }
        },
        "step_result": {
            "tensor_stats": {
                "tiny.proj": {
                    "applied_indices": applied_indices,
                    "q_sha256_after": q_sha_after,
                    "flip_count": len(applied_indices),
                }
            }
        },
    }


def _write_matching_sidecars(
    tmp_path: Path,
    *,
    steps: int = MEASURED_STEPS_REQUIRED,
    acc: list[int] | None = None,
) -> tuple[Path, Path]:
    acc = acc or [5, -9, 10]
    q = [0, 1, -1]
    oracle_path = tmp_path / "oracle_sidecar.jsonl"
    treatment_path = tmp_path / "treatment_sidecar.jsonl"
    for step in range(1, steps + 1):
        append_headroom_wiring_sidecar_chunk(
            oracle_path,
            step=step,
            state_key="tiny.proj",
            accumulator_lanes=acc,
            q_lanes=q,
        )
        append_headroom_wiring_sidecar_chunk(
            treatment_path,
            step=step,
            state_key="tiny.proj",
            accumulator_lanes=acc,
            q_lanes=q,
        )
    return oracle_path, treatment_path


def _w7_dual_arm_receipts(
    tmp_path: Path,
    *,
    steps: int = MEASURED_STEPS_REQUIRED,
    vote_abs_max: int = 24,
) -> tuple[dict, dict]:
    oracle_sidecar, treatment_sidecar = _write_matching_sidecars(tmp_path, steps=steps)
    step_reports: dict[str, dict] = {}
    for step in range(1, steps + 1):
        step_reports[str(step)] = _step_report(step_id=str(step), vote_abs_max=vote_abs_max)
    oracle = {
        "steps_completed": steps,
        "stop_reason": "",
        "envelope_id": CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24,
        "dense_accumulator_w7_clip": False,
        "step_reports": copy.deepcopy(step_reports),
        "headroom_wiring_sidecar_path": str(oracle_sidecar),
        "receipt_emit_profile": "s3bb_headroom_diagnostic_slim",
    }
    treatment = {
        "steps_completed": steps,
        "stop_reason": "",
        "envelope_id": CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24,
        "dense_accumulator_w7_clip": True,
        "step_reports": copy.deepcopy(step_reports),
        "headroom_wiring_sidecar_path": str(treatment_sidecar),
        "receipt_emit_profile": "s3bb_headroom_diagnostic_slim",
    }
    return oracle, treatment


def _classify_w7_end_to_end(oracle: dict, treatment: dict) -> dict:
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    coverage = stats["sidecar_coverage_diagnostics"]
    bridge = derive_w7_parity_inputs(primary, stats, coverage)
    envelope = resolve_confirmation_envelope(CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24)
    floor_width, arm_failures = verify_dual_arm_w7_configuration(
        oracle_receipt=oracle,
        treatment_receipt=treatment,
    )
    classifier = classify_w7_in_vivo_dual_arm(
        oracle_receipt=oracle,
        treatment_receipt=treatment,
        envelope=envelope,
        harness_failures=arm_failures,
        parity_break=bool(bridge["parity_break"]),
        structural_fail=bool(bridge["structural_fail"]),
        structural_reason=bridge.get("structural_reason"),
        confirmed_vote_acc_floor_width=floor_width,
    )
    classifier["s3bb_parity_primary_classifier"] = primary
    classifier["w7_parity_bridge"] = bridge
    return classifier


def test_derive_parity_break_on_crossing_only(tmp_path: Path) -> None:
    oracle, treatment = _w7_dual_arm_receipts(tmp_path)
    treatment_sidecar = Path(treatment["headroom_wiring_sidecar_path"])
    treatment_sidecar.unlink(missing_ok=True)
    for step in range(1, MEASURED_STEPS_REQUIRED + 1):
        append_headroom_wiring_sidecar_chunk(
            treatment_sidecar,
            step=step,
            state_key="tiny.proj",
            accumulator_lanes=[11, -9, 10],
            q_lanes=[0, 1, -1],
        )
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_DECISION_MISMATCH
    bridge = derive_w7_parity_inputs(primary, stats, stats["sidecar_coverage_diagnostics"])
    assert bridge["parity_break"] is True
    assert "per_step_crossing_bool_disagreement_count" in bridge["parity_break_driving_keys"]


def test_derive_parity_break_on_applied_mask_only(tmp_path: Path) -> None:
    oracle, treatment = _w7_dual_arm_receipts(tmp_path)
    treatment["step_reports"]["4"]["step_result"]["tensor_stats"]["tiny.proj"]["applied_indices"] = [
        1
    ]
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_DECISION_MISMATCH
    bridge = derive_w7_parity_inputs(primary, stats, stats["sidecar_coverage_diagnostics"])
    assert bridge["parity_break"] is True
    assert "applied_mask_mismatch_count" in bridge["parity_break_driving_keys"]


def test_derive_parity_break_on_q_trajectory_only(tmp_path: Path) -> None:
    oracle, treatment = _w7_dual_arm_receipts(tmp_path)
    treatment["step_reports"]["4"]["step_result"]["tensor_stats"]["tiny.proj"]["q_sha256_after"] = (
        "drift"
    )
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_FLIP_EQUIVALENT_DYNAMICS_DRIFT
    bridge = derive_w7_parity_inputs(primary, stats, stats["sidecar_coverage_diagnostics"])
    assert bridge["parity_break"] is True
    assert "q_sha256_after_mismatch_count" in bridge["parity_break_driving_keys"]


def test_structural_duplicate_oracle_sidecar_is_run_health_fail(tmp_path: Path) -> None:
    oracle, treatment = _w7_dual_arm_receipts(tmp_path)
    oracle_sidecar = Path(oracle["headroom_wiring_sidecar_path"])
    append_headroom_wiring_sidecar_chunk(
        oracle_sidecar,
        step=4,
        state_key="tiny.proj",
        accumulator_lanes=[5, -9, 10],
        q_lanes=[0, 1, -1],
    )
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    coverage = stats["sidecar_coverage_diagnostics"]
    assert coverage["structural_fail"] is True
    assert "oracle_duplicate_keys" in coverage["structural_reasons"]
    bridge = derive_w7_parity_inputs(primary, stats, coverage)
    assert bridge["structural_fail"] is True
    assert bridge["parity_break"] is False
    envelope = resolve_confirmation_envelope(CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24)
    classifier = classify_w7_in_vivo_dual_arm(
        oracle_receipt=oracle,
        treatment_receipt=treatment,
        envelope=envelope,
        structural_fail=True,
        structural_reason=bridge["structural_reason"],
        confirmed_vote_acc_floor_width=CPU_PHASE0_STRUCTURAL_FLOOR_WIDTH,
    )
    assert classifier["primary_classifier"] == CLASSIFIER_RUN_HEALTH_FAIL
    assert classifier["structural_fail"] is True
    assert classifier["rules_promotion_unlock_predicate"] is False


def test_clean_parity_at_peak_33_confirms(tmp_path: Path) -> None:
    oracle, treatment = _w7_dual_arm_receipts(tmp_path, vote_abs_max=24)
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_DECISION_PARITY_OK
    bridge = derive_w7_parity_inputs(primary, stats, stats["sidecar_coverage_diagnostics"])
    assert bridge["structural_fail"] is False
    assert bridge["parity_break"] is False
    envelope = resolve_confirmation_envelope(CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24)
    classifier = classify_w7_in_vivo_dual_arm(
        oracle_receipt=oracle,
        treatment_receipt=treatment,
        envelope=envelope,
        parity_break=False,
        structural_fail=False,
        confirmed_vote_acc_floor_width=CPU_PHASE0_STRUCTURAL_FLOOR_WIDTH,
    )
    assert classifier["primary_classifier"] == CLASSIFIER_W7_IN_VIVO_CONFIRMED
    assert classifier["live_reachable_peak_estimate"] == 33
    assert classifier["rules_promotion_unlock_predicate"] is True


def test_keyed_sidecar_compare_never_raises_on_structural_mismatch(tmp_path: Path) -> None:
    oracle, treatment = _w7_dual_arm_receipts(tmp_path)
    oracle_sidecar = Path(oracle["headroom_wiring_sidecar_path"])
    for _ in range(32):
        append_headroom_wiring_sidecar_chunk(
            oracle_sidecar,
            step=4,
            state_key="tiny.proj",
            accumulator_lanes=[5, -9, 10],
            q_lanes=[0, 1, -1],
        )
    coverage = diagnose_sidecar_coverage(oracle_sidecar, treatment["headroom_wiring_sidecar_path"])
    assert coverage["structural_fail"] is True
    stats = compare_arm_wiring_guards_streaming(
        oracle,
        treatment,
        oracle_sidecar_path=oracle_sidecar,
        treatment_sidecar_path=treatment["headroom_wiring_sidecar_path"],
    )
    assert "sidecar_coverage_diagnostics" in stats
    assert stats["sidecar_coverage_diagnostics"]["structural_fail"] is True


@pytest.mark.parametrize(
    "mutator",
    [
        lambda o, t: [
            receipt.__setitem__(
                "step_reports",
                {
                    step_id: {**report, "step_result": {}}
                    for step_id, report in receipt["step_reports"].items()
                },
            )
            for receipt in (o, t)
        ],
        lambda o, t: [
            receipt["step_reports"]["4"]["step_result"]["tensor_stats"]["tiny.proj"].pop(
                "applied_indices", None
            )
            for receipt in (o, t)
        ],
        lambda o, t: [
            receipt["step_reports"]["4"]["step_result"]["tensor_stats"]["tiny.proj"].__setitem__(
                "q_sha256_after", ""
            )
            for receipt in (o, t)
        ],
    ],
    ids=["missing_tensor_stats", "missing_applied_indices", "missing_q_hashes"],
)
def test_harness_primary_clean_sidecar_is_run_health_fail(
    tmp_path: Path,
    mutator,
) -> None:
    oracle, treatment = _w7_dual_arm_receipts(tmp_path, vote_abs_max=24)
    mutator(oracle, treatment)
    classifier = _classify_w7_end_to_end(oracle, treatment)
    assert classifier["s3bb_parity_primary_classifier"] == CLASSIFIER_HARNESS_OR_LIVENESS_FAIL
    bridge = classifier["w7_parity_bridge"]
    assert bridge["structural_fail"] is True
    assert bridge["parity_break"] is False
    assert bridge["structural_reason"] == "s3bb_harness_or_liveness_fail"
    assert classifier["primary_classifier"] == CLASSIFIER_RUN_HEALTH_FAIL
    assert classifier["primary_classifier"] != CLASSIFIER_W7_BREAKS_LIVE_PARITY
    assert classifier["primary_classifier"] != CLASSIFIER_W7_IN_VIVO_CONFIRMED


def test_insufficient_steps_clean_sidecar_is_run_health_fail(tmp_path: Path) -> None:
    oracle, treatment = _w7_dual_arm_receipts(tmp_path, steps=3, vote_abs_max=24)
    classifier = _classify_w7_end_to_end(oracle, treatment)
    assert classifier["s3bb_parity_primary_classifier"] == CLASSIFIER_HARNESS_OR_LIVENESS_FAIL
    assert classifier["w7_parity_bridge"]["structural_reason"] == "s3bb_harness_or_liveness_fail"
    assert classifier["primary_classifier"] == CLASSIFIER_RUN_HEALTH_FAIL


def test_zero_shared_measured_steps_clean_sidecar_is_run_health_fail(tmp_path: Path) -> None:
    oracle, treatment = _w7_dual_arm_receipts(tmp_path, steps=2, vote_abs_max=24)
    classifier = _classify_w7_end_to_end(oracle, treatment)
    assert classifier["s3bb_parity_primary_classifier"] == CLASSIFIER_HARNESS_OR_LIVENESS_FAIL
    assert classifier["primary_classifier"] == CLASSIFIER_RUN_HEALTH_FAIL
    assert classifier["w7_parity_bridge"]["parity_break"] is False


def test_domain_primary_clean_sidecar_is_run_health_fail(tmp_path: Path) -> None:
    oracle, treatment = _w7_dual_arm_receipts(tmp_path, vote_abs_max=24)
    treatment_sidecar = Path(treatment["headroom_wiring_sidecar_path"])
    treatment_sidecar.unlink(missing_ok=True)
    for step in range(1, MEASURED_STEPS_REQUIRED + 1):
        append_headroom_wiring_sidecar_chunk(
            treatment_sidecar,
            step=step,
            state_key="tiny.proj",
            accumulator_lanes=[5, -9, 16],
            q_lanes=[0, 1, -1],
        )
    classifier = _classify_w7_end_to_end(oracle, treatment)
    assert classifier["s3bb_parity_primary_classifier"] == CLASSIFIER_DOMAIN_OR_HEADROOM_FAIL
    bridge = classifier["w7_parity_bridge"]
    assert bridge["structural_fail"] is True
    assert bridge["parity_break"] is False
    assert bridge["structural_reason"] == "s3bb_domain_or_headroom_fail"
    assert classifier["primary_classifier"] == CLASSIFIER_RUN_HEALTH_FAIL
    assert classifier["primary_classifier"] != CLASSIFIER_W7_BREAKS_LIVE_PARITY


def test_unenumerated_s3bb_primary_clean_sidecar_is_run_health_fail(tmp_path: Path) -> None:
    oracle, treatment = _w7_dual_arm_receipts(tmp_path, vote_abs_max=24)
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_DECISION_PARITY_OK
    coverage = stats["sidecar_coverage_diagnostics"]
    bridge = derive_w7_parity_inputs("SOME_FUTURE_PRIMARY", stats, coverage)
    assert bridge["structural_fail"] is True
    assert bridge["parity_break"] is False
    assert bridge["structural_reason"] == "s3bb_unenumerated_primary_fail"
    envelope = resolve_confirmation_envelope(CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24)
    floor_width, arm_failures = verify_dual_arm_w7_configuration(
        oracle_receipt=oracle,
        treatment_receipt=treatment,
    )
    classifier = classify_w7_in_vivo_dual_arm(
        oracle_receipt=oracle,
        treatment_receipt=treatment,
        envelope=envelope,
        harness_failures=arm_failures,
        parity_break=bool(bridge["parity_break"]),
        structural_fail=bool(bridge["structural_fail"]),
        structural_reason=bridge.get("structural_reason"),
        confirmed_vote_acc_floor_width=floor_width,
    )
    assert classifier["primary_classifier"] == CLASSIFIER_RUN_HEALTH_FAIL
    assert classifier["primary_classifier"] != CLASSIFIER_W7_IN_VIVO_CONFIRMED
    assert classifier["primary_classifier"] != CLASSIFIER_W7_BREAKS_LIVE_PARITY


@pytest.mark.skipif(
    not Path(
        "/home/gabe/claw-code-creditdir/transient_fp_credit/"
        "w7_dense_acc_in_vivo_seed43_43_2189e72006/int16_oracle_flag_off/receipt.json"
    ).is_file(),
    reason="2189e72006 run artifacts not present on this host",
)
def test_reclassify_2189e72006_defaults_structural_run_health_fail(tmp_path: Path) -> None:
    run_root = Path(
        "/home/gabe/claw-code-creditdir/transient_fp_credit/"
        "w7_dense_acc_in_vivo_seed43_43_2189e72006"
    )
    receipt = emit_w7_in_vivo_classifier_receipt(run_root=run_root)
    coverage = receipt["sidecar_coverage_diagnostics"]
    assert coverage["structural_fail"] is True
    assert coverage["oracle_row_count"] == 384
    assert coverage["treatment_row_count"] == 256
    assert coverage["oracle_duplicate_key_count"] > 0
    assert receipt["primary_classifier"] == CLASSIFIER_RUN_HEALTH_FAIL
    assert receipt["structural_fail"] is True
    assert receipt["parity_break"] is False
    assert receipt["rules_promotion_unlock_predicate"] is False
    out_path = tmp_path / "reclassify_2189e72006.json"
    out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    assert out_path.stat().st_size > 0
