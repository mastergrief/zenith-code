from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import make_bounded_tensor_state
from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    HEADROOM_WIRING_SIDECAR_SCHEMA_VERSION,
    MEASURED_STEPS_REQUIRED,
    RECEIPT_EMIT_PROFILE_SLIM,
    SNAPSHOT_MODE_AGGREGATE_ONLY,
    SNAPSHOT_MODE_FULL,
    WARMUP_STEPS,
    append_headroom_wiring_sidecar_chunk,
    attach_s3bb_headroom_telemetry_to_step_report,
    compare_arm_wiring_guards,
    compare_arm_wiring_guards_inline,
    compare_arm_wiring_guards_streaming,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    DEFAULT_CROSSING_THRESHOLD_ABS,
    crossing_bool_w6,
)
from scripts.hrm_text_158_s3bb_w6_dynamics_postrun import run_postrun


def _parity_step_report(
    *,
    step_id: str,
    acc_values: list[int],
    q_values: list[int] | None = None,
    state_key: str = "tiny.proj",
) -> dict:
    q_values = q_values or [0] * len(acc_values)
    telemetry = {
        "schema_version": "hrm_text_158_s3bb_headroom_telemetry/v0",
        "global_max_abs_accumulator": max(abs(int(v)) for v in acc_values),
        "margin_to_w6_boundary_min": 31 - max(abs(int(v)) for v in acc_values),
        "lanes_within_K_of_boundary_fraction": 0.0,
        "out_of_domain_lane_count": 0,
        "would_strict_raise_step": False,
        "strict_raise_count": 0,
        "boundary_value_error_caught": False,
        "eligible_module_count": 1,
        "total_lane_count": len(acc_values),
        "accumulator_snapshots_by_state_key": {state_key: acc_values},
        "q_snapshots_by_state_key": {state_key: q_values},
    }
    return {"headroom_telemetry": telemetry, "step_id": step_id}


def _build_inline_receipts(steps: int = MEASURED_STEPS_REQUIRED) -> tuple[dict, dict]:
    step_reports: dict[str, dict] = {}
    for step in range(1, steps + 1):
        step_reports[str(step)] = _parity_step_report(
            step_id=str(step),
            acc_values=[5, -9, 21],
            q_values=[0, 1, -1],
        )
    oracle = {
        "steps_completed": steps,
        "stop_reason": "",
        "step_reports": dict(step_reports),
        "receipt_emit_profile": "full",
    }
    treatment = {
        "steps_completed": steps,
        "stop_reason": "",
        "step_reports": copy.deepcopy(step_reports),
        "receipt_emit_profile": "full",
    }
    return oracle, treatment


def _write_sidecars_from_inline(
    oracle: dict,
    treatment: dict,
    *,
    oracle_path: Path,
    treatment_path: Path,
) -> None:
    if oracle_path.is_file():
        oracle_path.unlink()
    if treatment_path.is_file():
        treatment_path.unlink()
    for step_id, report in sorted(oracle["step_reports"].items(), key=lambda item: int(item[0])):
        if int(step_id) <= WARMUP_STEPS:
            continue
        telemetry = report["headroom_telemetry"]
        for state_key, acc_values in telemetry["accumulator_snapshots_by_state_key"].items():
            append_headroom_wiring_sidecar_chunk(
                oracle_path,
                step=int(step_id),
                state_key=str(state_key),
                accumulator_lanes=acc_values,
                q_lanes=telemetry["q_snapshots_by_state_key"][state_key],
            )
    for step_id, report in sorted(treatment["step_reports"].items(), key=lambda item: int(item[0])):
        if int(step_id) <= WARMUP_STEPS:
            continue
        telemetry = report["headroom_telemetry"]
        for state_key, acc_values in telemetry["accumulator_snapshots_by_state_key"].items():
            append_headroom_wiring_sidecar_chunk(
                treatment_path,
                step=int(step_id),
                state_key=str(state_key),
                accumulator_lanes=acc_values,
                q_lanes=telemetry["q_snapshots_by_state_key"][state_key],
            )


def _slim_receipt_from_inline(
    receipt: dict,
    *,
    sidecar_path: Path,
) -> dict:
    slim = copy.deepcopy(receipt)
    slim["receipt_emit_profile"] = RECEIPT_EMIT_PROFILE_SLIM
    slim["headroom_wiring_sidecar_path"] = str(sidecar_path)
    slim["headroom_wiring_sidecar_schema"] = HEADROOM_WIRING_SIDECAR_SCHEMA_VERSION
    for report in slim["step_reports"].values():
        telemetry = report["headroom_telemetry"]
        telemetry.pop("accumulator_snapshots_by_state_key", None)
        telemetry.pop("q_snapshots_by_state_key", None)
    return slim


def test_streaming_equivalence_matches_inline(tmp_path: Path) -> None:
    oracle, treatment = _build_inline_receipts(steps=MEASURED_STEPS_REQUIRED)
    inline = compare_arm_wiring_guards_inline(oracle, treatment)

    oracle_sidecar = tmp_path / "oracle.jsonl"
    treatment_sidecar = tmp_path / "treatment.jsonl"
    _write_sidecars_from_inline(oracle, treatment, oracle_path=oracle_sidecar, treatment_path=treatment_sidecar)
    slim_oracle = _slim_receipt_from_inline(oracle, sidecar_path=oracle_sidecar)
    slim_treatment = _slim_receipt_from_inline(treatment, sidecar_path=treatment_sidecar)
    streaming = compare_arm_wiring_guards_streaming(
        slim_oracle,
        slim_treatment,
        oracle_sidecar_path=oracle_sidecar,
        treatment_sidecar_path=treatment_sidecar,
    )
    assert streaming == inline
    assert compare_arm_wiring_guards(oracle, treatment) == inline
    assert compare_arm_wiring_guards(slim_oracle, slim_treatment) == inline


def _treatment_q_crossing_disagreement_count(
    o_vals: list[int],
    t_vals: list[int],
    o_q: list[int],
    t_q: list[int],
    *,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
) -> int:
    """Buggy reference: treatment crossing uses treatment q instead of oracle q."""

    disagreements = 0
    for lane_index, (o_val, t_val) in enumerate(zip(o_vals, t_vals, strict=True)):
        oracle_q = int(o_q[lane_index]) if lane_index < len(o_q) else 0
        treatment_q = int(t_q[lane_index]) if lane_index < len(t_q) else 0
        o_cross = crossing_bool_w6(int(o_val), oracle_q, threshold_abs=int(threshold_abs))
        t_cross = crossing_bool_w6(int(t_val), treatment_q, threshold_abs=int(threshold_abs))
        if o_cross != t_cross:
            disagreements += 1
    return disagreements


def test_crossing_guard_uses_oracle_q_for_both_arms(tmp_path: Path) -> None:
    """Regression: treatment crossing must reference oracle q, not treatment q."""

    o_vals = [15, 5]
    t_vals = [-12, 5]
    o_q = [0, 0]
    t_q = [-1, 0]

    oracle, treatment = _build_inline_receipts(steps=MEASURED_STEPS_REQUIRED)
    for receipt in (oracle, treatment):
        report = receipt["step_reports"]["4"]
        report["headroom_telemetry"]["accumulator_snapshots_by_state_key"]["tiny.proj"] = list(
            o_vals if receipt is oracle else t_vals
        )
        report["headroom_telemetry"]["q_snapshots_by_state_key"]["tiny.proj"] = list(
            o_q if receipt is oracle else t_q
        )

    inline = compare_arm_wiring_guards_inline(oracle, treatment)
    oracle_sidecar = tmp_path / "oracle_qref.jsonl"
    treatment_sidecar = tmp_path / "treatment_qref.jsonl"
    _write_sidecars_from_inline(
        oracle,
        treatment,
        oracle_path=oracle_sidecar,
        treatment_path=treatment_sidecar,
    )
    slim_oracle = _slim_receipt_from_inline(oracle, sidecar_path=oracle_sidecar)
    slim_treatment = _slim_receipt_from_inline(treatment, sidecar_path=treatment_sidecar)
    streaming = compare_arm_wiring_guards_streaming(
        slim_oracle,
        slim_treatment,
        oracle_sidecar_path=oracle_sidecar,
        treatment_sidecar_path=treatment_sidecar,
    )

    assert inline == streaming
    assert int(inline["per_step_crossing_bool_disagreement_count"]) == 0
    assert _treatment_q_crossing_disagreement_count(o_vals, t_vals, o_q, t_q) == 1


def test_streaming_equivalence_detects_divergence(tmp_path: Path) -> None:
    oracle, treatment = _build_inline_receipts(steps=MEASURED_STEPS_REQUIRED)
    treatment["step_reports"]["4"]["headroom_telemetry"]["accumulator_snapshots_by_state_key"][
        "tiny.proj"
    ] = [6, -9, 21]
    inline = compare_arm_wiring_guards_inline(oracle, treatment)

    oracle_sidecar = tmp_path / "oracle_div.jsonl"
    treatment_sidecar = tmp_path / "treatment_div.jsonl"
    _write_sidecars_from_inline(oracle, treatment, oracle_path=oracle_sidecar, treatment_path=treatment_sidecar)
    slim_oracle = _slim_receipt_from_inline(oracle, sidecar_path=oracle_sidecar)
    slim_treatment = _slim_receipt_from_inline(treatment, sidecar_path=treatment_sidecar)
    streaming = compare_arm_wiring_guards_streaming(
        slim_oracle,
        slim_treatment,
        oracle_sidecar_path=oracle_sidecar,
        treatment_sidecar_path=treatment_sidecar,
    )
    assert streaming == inline
    assert float(streaming["per_step_accumulator_l1_max_abs_delta"]) > 0.0


def test_slim_attach_writes_sidecar_without_inline_snapshots(tmp_path: Path) -> None:
    state = make_bounded_tensor_state(
        "tiny.proj",
        torch.tensor([[0, 1, -1]], dtype=torch.int8),
        1.0,
        torch.tensor([[5, -9, 21]], dtype=torch.int16),
    )
    sidecar_path = tmp_path / "headroom_wiring_sidecar.jsonl"
    report: dict = {"step_id": "3"}
    attach_s3bb_headroom_telemetry_to_step_report(
        report,
        phase="s3bb-w6-headroom-diagnostic",
        post_update_states={"tiny.proj": state},
        snapshot_mode=SNAPSHOT_MODE_AGGREGATE_ONLY,
        headroom_wiring_sidecar_path=sidecar_path,
        step=3,
    )
    telemetry = report["headroom_telemetry"]
    assert "accumulator_snapshots_by_state_key" not in telemetry
    assert "q_snapshots_by_state_key" not in telemetry
    assert sidecar_path.is_file()
    records = [json.loads(line) for line in sidecar_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["step"] == 3
    assert records[0]["state_key"] == "tiny.proj"


def test_full_attach_retains_inline_snapshots() -> None:
    state = make_bounded_tensor_state(
        "tiny.proj",
        torch.tensor([[0, 1, -1]], dtype=torch.int8),
        1.0,
        torch.tensor([[5, -9, 21]], dtype=torch.int16),
    )
    report: dict = {"step_id": "3"}
    attach_s3bb_headroom_telemetry_to_step_report(
        report,
        phase="s3bb-w6-headroom-diagnostic",
        post_update_states={"tiny.proj": state},
        snapshot_mode=SNAPSHOT_MODE_FULL,
    )
    telemetry = report["headroom_telemetry"]
    assert telemetry["accumulator_snapshots_by_state_key"]["tiny.proj"] == [5, -9, 21]


def test_slim_postrun_uses_streaming_sidecar(tmp_path: Path) -> None:
    oracle, treatment = _build_inline_receipts(steps=MEASURED_STEPS_REQUIRED)
    oracle_dir = tmp_path / "int16_oracle_flag_off"
    treatment_dir = tmp_path / "w6_carrier_flag_on"
    oracle_dir.mkdir(parents=True)
    treatment_dir.mkdir(parents=True)

    oracle_sidecar = oracle_dir / "headroom_wiring_sidecar.jsonl"
    treatment_sidecar = treatment_dir / "headroom_wiring_sidecar.jsonl"
    _write_sidecars_from_inline(oracle, treatment, oracle_path=oracle_sidecar, treatment_path=treatment_sidecar)
    slim_oracle = _slim_receipt_from_inline(oracle, sidecar_path=oracle_sidecar)
    slim_treatment = _slim_receipt_from_inline(treatment, sidecar_path=treatment_sidecar)
    oracle_dir.joinpath("receipt.json").write_text(json.dumps(slim_oracle), encoding="utf-8")
    treatment_dir.joinpath("receipt.json").write_text(json.dumps(slim_treatment), encoding="utf-8")

    classifier_path = tmp_path / "classifier_receipt.json"
    receipt = run_postrun(run_root=tmp_path, json_out=classifier_path)
    assert receipt["primary_classifier"] == "W6_HEADROOM_SUFFICIENT_PARITY_OK"
    assert receipt["wiring_guards"]["vote_update_state_accumulator_equality_rate"] == 1.0


def test_scale_smoke_mini_profile(tmp_path: Path) -> None:
    from scripts.hrm_text_158_s3c_receipt_emit_scale_smoke import (
        METRICS_SCHEMA_VERSION,
        run_scale_smoke,
    )

    metrics = run_scale_smoke(
        output_dir=tmp_path / "smoke",
        modules=2,
        lanes_per_module=32,
        measured_steps=3,
        warmup_steps=WARMUP_STEPS,
    )
    assert metrics["metrics_schema_version"] == METRICS_SCHEMA_VERSION
    assert metrics["passed"] is True
    assert metrics["wiring_guards"]["vote_update_state_accumulator_equality_rate"] == 1.0
    assert "phase_a_data_gen_wall_seconds" in metrics
    assert "phase_b_sidecar_emit_wall_seconds_by_arm" in metrics
    assert "phase_b_sidecar_emit_wall_seconds_per_arm_max" in metrics
    assert "phase_b_sidecar_emit_wall_seconds_total" in metrics
    assert "phase_b_prime_aggregate_peak_rss_mib" in metrics
    assert "phase_c_compare_wall_seconds" in metrics
    assert metrics["phase_a_data_gen_wall_seconds"] >= 0.0
    assert metrics["phase_b_sidecar_emit_wall_seconds_total"] >= metrics[
        "phase_b_sidecar_emit_wall_seconds_per_arm_max"
    ]


def test_scale_smoke_phase_a_excluded_from_gated_wall(tmp_path: Path) -> None:
    import numpy as np

    from scripts.hrm_text_158_s3c_receipt_emit_scale_smoke import run_scale_smoke

    def slow_lane_factory(
        *,
        module_index: int,
        step: int,
        lanes_per_module: int,
        arm_offset: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        time.sleep(0.05)
        lanes = np.arange(lanes_per_module, dtype=np.int64)
        acc = ((module_index + step + lanes + arm_offset) % 31) - 15
        q = ((lanes + module_index) % 3) - 1
        return acc.astype(np.int16), q.astype(np.int16)

    metrics = run_scale_smoke(
        output_dir=tmp_path / "slow_gen",
        modules=1,
        lanes_per_module=4,
        measured_steps=1,
        warmup_steps=WARMUP_STEPS,
        lane_factory=slow_lane_factory,
        gate_overrides={"phase_b_sidecar_emit_wall_seconds_per_arm_max_lte": 0.05},
    )
    assert metrics["phase_a_data_gen_wall_seconds"] >= 0.08
    assert metrics["phase_b_sidecar_emit_wall_seconds_per_arm_max"] < 0.05
    assert "phase_a_data_gen_wall_seconds" not in metrics["failures"]
    assert "phase_b_sidecar_emit_wall_seconds_total" not in metrics["failures"]
    assert metrics["passed"] is True


def test_scale_smoke_disk_precheck_uses_file_size_gate_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.hrm_text_158_s3c_receipt_emit_scale_smoke as smoke_module

    observed: dict[str, int] = {}

    def capture_precheck(_output_dir: Path, *, sidecar_file_size_budget_bytes: int) -> None:
        observed["budget"] = sidecar_file_size_budget_bytes

    monkeypatch.setattr(smoke_module, "_disk_precheck", capture_precheck)
    smoke_module.run_scale_smoke(
        output_dir=tmp_path / "disk_gate",
        modules=1,
        lanes_per_module=4,
        measured_steps=1,
        warmup_steps=WARMUP_STEPS,
    )
    assert observed["budget"] == smoke_module.GATES["sidecar_file_size_bytes_lte"]


def test_scale_smoke_aggregate_mirror_rss_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.hrm_text_158_s3c_receipt_emit_scale_smoke as smoke_module

    baseline = {"value": 100.0}

    def fake_read_rss_mib() -> float:
        baseline["value"] += 600.0
        return baseline["value"]

    monkeypatch.setattr(smoke_module, "_read_rss_mib", fake_read_rss_mib)
    metrics = smoke_module.run_scale_smoke(
        output_dir=tmp_path / "aggregate_gate",
        modules=1,
        lanes_per_module=8,
        measured_steps=1,
        warmup_steps=WARMUP_STEPS,
    )
    assert metrics["passed"] is False
    assert "phase_b_prime_aggregate_peak_rss_mib" in metrics["failures"]


def test_scale_smoke_cleanup_removes_sidecars_by_default(tmp_path: Path) -> None:
    from scripts.hrm_text_158_s3c_receipt_emit_scale_smoke import run_scale_smoke

    output_dir = tmp_path / "cleanup"
    run_scale_smoke(
        output_dir=output_dir,
        modules=1,
        lanes_per_module=8,
        measured_steps=1,
        warmup_steps=WARMUP_STEPS,
    )
    assert not (output_dir / "oracle_headroom_wiring_sidecar.jsonl").exists()
    assert not (output_dir / "treatment_headroom_wiring_sidecar.jsonl").exists()


def test_scale_smoke_retain_artifacts_keeps_sidecars(tmp_path: Path) -> None:
    from scripts.hrm_text_158_s3c_receipt_emit_scale_smoke import run_scale_smoke

    output_dir = tmp_path / "retain"
    run_scale_smoke(
        output_dir=output_dir,
        modules=1,
        lanes_per_module=8,
        measured_steps=1,
        warmup_steps=WARMUP_STEPS,
        cleanup_sidecars=False,
    )
    assert (output_dir / "oracle_headroom_wiring_sidecar.jsonl").exists()
    assert (output_dir / "treatment_headroom_wiring_sidecar.jsonl").exists()


def test_scale_smoke_per_arm_max_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.hrm_text_158_s3c_receipt_emit_scale_smoke as smoke_module
    from calm.hrm_text_158.native_full_stack import s3bb_headroom_telemetry as telemetry_module

    original_append = telemetry_module.append_headroom_wiring_sidecar_chunk

    def slow_append(*args: object, **kwargs: object) -> None:
        time.sleep(0.08)
        original_append(*args, **kwargs)

    monkeypatch.setattr(telemetry_module, "append_headroom_wiring_sidecar_chunk", slow_append)
    monkeypatch.setattr(smoke_module, "append_headroom_wiring_sidecar_chunk", slow_append)

    metrics = smoke_module.run_scale_smoke(
        output_dir=tmp_path / "per_arm_fail",
        modules=1,
        lanes_per_module=4,
        measured_steps=2,
        warmup_steps=WARMUP_STEPS,
        gate_overrides={"phase_b_sidecar_emit_wall_seconds_per_arm_max_lte": 0.1},
    )
    assert metrics["phase_b_sidecar_emit_wall_seconds_per_arm_max"] >= 0.15
    assert metrics["passed"] is False
    assert "phase_b_sidecar_emit_wall_seconds_per_arm_max" in metrics["failures"]


def test_scale_smoke_total_not_gated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.hrm_text_158_s3c_receipt_emit_scale_smoke as smoke_module
    from calm.hrm_text_158.native_full_stack import s3bb_headroom_telemetry as telemetry_module

    original_append = telemetry_module.append_headroom_wiring_sidecar_chunk

    def slow_append(*args: object, **kwargs: object) -> None:
        time.sleep(0.08)
        original_append(*args, **kwargs)

    monkeypatch.setattr(telemetry_module, "append_headroom_wiring_sidecar_chunk", slow_append)
    monkeypatch.setattr(smoke_module, "append_headroom_wiring_sidecar_chunk", slow_append)

    metrics = smoke_module.run_scale_smoke(
        output_dir=tmp_path / "total_not_gated",
        modules=1,
        lanes_per_module=4,
        measured_steps=2,
        warmup_steps=WARMUP_STEPS,
        gate_overrides={"phase_b_sidecar_emit_wall_seconds_per_arm_max_lte": 0.25},
    )
    assert metrics["phase_b_sidecar_emit_wall_seconds_per_arm_max"] < 0.25
    assert metrics["phase_b_sidecar_emit_wall_seconds_total"] > 0.25
    assert "phase_b_sidecar_emit_wall_seconds_total" not in metrics["failures"]
    assert metrics["passed"] is True
