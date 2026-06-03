"""Native full-loop reference-stitch engineering receipt tests."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.full_loop_receipt import (
    DEFAULT_FULL_LOOP_RECEIPT_ARTIFACT_PATH,
    FULL_LOOP_RECEIPT_ARTIFACT_ENV,
    GLOBAL_CAP_CPU_GLUE_CAVEAT,
    NATIVE_FULL_LOOP_ENGINEERING_RECEIPT_LABEL,
    NEXT_PHYSICAL_SUB2_FORK,
    PRIOR_LARGE_FIXTURE_REFERENCE,
    QSCALE_REFERENCE_MATERIALIZATION_CAVEAT,
    RUN_GPU_FULL_LOOP_RECEIPT_ENV,
    TINY_TWO_PROJECTION_FIXTURE_NAME,
    measure_tiny_two_projection_fixture_budget,
    run_native_full_loop_engineering_receipt,
    tiny_two_projection_vote_cap_fixture,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    PHYSICAL_SUB2_NOT_ACHIEVED_STATEMENT,
)
from calm.hrm_text_158.native_full_stack.vote_update import RUN_GPU_Q_ACC_APPLY_ENV


GPU_FULL_LOOP_RECEIPT = pytest.mark.skipif(
    os.environ.get(RUN_GPU_FULL_LOOP_RECEIPT_ENV) != "1"
    or os.environ.get(RUN_GPU_Q_ACC_APPLY_ENV) != "1"
    or not torch.cuda.is_available(),
    reason=(
        "native full-loop GPU engineering receipt deferred; set "
        f"{RUN_GPU_FULL_LOOP_RECEIPT_ENV}=1 and {RUN_GPU_Q_ACC_APPLY_ENV}=1 "
        "only inside a granted gpu:0 lane"
    ),
)


def test_runner_is_default_off_before_gpu_lane(monkeypatch):
    monkeypatch.delenv(RUN_GPU_FULL_LOOP_RECEIPT_ENV, raising=False)
    monkeypatch.setenv(RUN_GPU_Q_ACC_APPLY_ENV, "1")

    with pytest.raises(RuntimeError, match=RUN_GPU_FULL_LOOP_RECEIPT_ENV):
        run_native_full_loop_engineering_receipt(device="cpu")


def test_tiny_fixture_budget_computes_own_160_entry_ledger_not_prior_large_value():
    fixture = tiny_two_projection_vote_cap_fixture(device="cpu")
    report = measure_tiny_two_projection_fixture_budget(device="cpu")

    assert fixture.name == TINY_TWO_PROJECTION_FIXTURE_NAME
    assert fixture.tensor_shapes == {"proj_in": (8, 16), "proj_out": (4, 8)}
    assert fixture.eligible_weight_count == 160
    assert report.eligible_weight_count == 160
    assert report.q_state_count == 2
    assert report.accumulator_tensor_count == 2
    assert report.q_packed_data_bits_per_weight == pytest.approx(2.0)
    assert report.q_packed_metadata_bits_per_weight == pytest.approx(3.2)
    assert report.q_packed_total_bits_per_weight == pytest.approx(5.2)
    assert report.acc_int16_bits_per_weight == pytest.approx(16.0)
    assert report.frozen_scale_fp32_bits_per_weight == pytest.approx(0.4)
    assert report.packed_inclusive_physical_bits_per_weight == pytest.approx(21.6)
    assert report.target_achieved is False
    assert report.required_acc_bits_per_weight_for_sub2_physical_q_with_scale_and_metadata == pytest.approx(-3.6)
    assert report.dominant_ledger == "acc_int16"
    assert report.receipt_statement == PHYSICAL_SUB2_NOT_ACHIEVED_STATEMENT
    assert report.packed_inclusive_physical_bits_per_weight != pytest.approx(
        PRIOR_LARGE_FIXTURE_REFERENCE["packed_inclusive_physical_bits_per_weight"],
    )


def test_receipt_labels_reference_harness_and_defers_science_not_acquisition():
    assert NATIVE_FULL_LOOP_ENGINEERING_RECEIPT_LABEL.endswith("reference_stitch_only")
    assert "q.float() * scale" in QSCALE_REFERENCE_MATERIALIZATION_CAVEAT
    assert "not a native/custom-kernel speed claim" in QSCALE_REFERENCE_MATERIALIZATION_CAVEAT
    assert "CPU/control-flow glue" in GLOBAL_CAP_CPU_GLUE_CAVEAT
    assert "ternary-hybrid" in NEXT_PHYSICAL_SUB2_FORK
    assert PRIOR_LARGE_FIXTURE_REFERENCE["scope"].endswith("not_tiny_loop_expected_value")
    assert PRIOR_LARGE_FIXTURE_REFERENCE["target_achieved"] is False


@GPU_FULL_LOOP_RECEIPT
def test_cuda_full_loop_receipt_proves_cap_pressure_parity_and_compact_artifact():
    artifact_path = Path(
        os.environ.get(FULL_LOOP_RECEIPT_ARTIFACT_ENV, DEFAULT_FULL_LOOP_RECEIPT_ARTIFACT_PATH),
    )

    receipt = run_native_full_loop_engineering_receipt(artifact_path=artifact_path)

    assert receipt.label == NATIVE_FULL_LOOP_ENGINEERING_RECEIPT_LABEL
    assert receipt.acquisition_claim is False
    assert receipt.retention_claim is False
    assert receipt.native_custom_kernel_speed_claim is False
    assert receipt.eligible_weight_count == 160
    assert len(receipt.step_receipts) == 2
    assert receipt.step_receipts[0].step_consumes_state_mutated_by_prior_step is False
    assert receipt.step_receipts[1].step_consumes_state_mutated_by_prior_step is True
    assert receipt.peak_reserved_bytes >= receipt.peak_allocated_bytes >= 0
    assert receipt.wall_clock_per_step_seconds > 0.0

    for step in receipt.step_receipts:
        assert step.qscale_backend == "cuda"
        assert step.qscale_outputs_finite is True
        assert step.pre_cap_demand_count > step.global_rate_cap_cap
        assert step.global_rate_cap_cap == 2
        assert step.accepted_count == 2
        assert step.deferred_count > 0
        assert step.global_rate_cap_saturated is True
        assert step.cap_provenance_source == "apply_global_rate_cap_reference.accepted_rows"
        assert step.local_plan_rows_used_as_cap_acceptance is False
        assert step.parity_cuda_matches_cpu_global_cap_oracle is True
        assert step.cpu_oracle_mutate_outputs is True
        assert step.q_changed_count > 0
        assert step.q_changed_count == sum(step.q_changed_count_by_state.values())
        assert step.accepted_count_by_state == {"proj_in": 1, "proj_out": 1}
        assert step.cap_single_tensor_winner_state_key is None
        assert step.budget["eligible_weight_count"] == 160
        assert step.budget["target_achieved"] is False

    budget = receipt.terminal_budget
    assert budget["q_packed_data_bits_per_weight"] == pytest.approx(2.0)
    assert budget["q_packed_metadata_bits_per_weight"] == pytest.approx(3.2)
    assert budget["acc_int16_bits_per_weight"] == pytest.approx(16.0)
    assert budget["frozen_scale_fp32_bits_per_weight"] == pytest.approx(0.4)
    assert budget["packed_inclusive_physical_bits_per_weight"] == pytest.approx(21.6)
    assert budget["packed_inclusive_physical_bits_per_weight"] != pytest.approx(
        receipt.prior_large_fixture_reference["packed_inclusive_physical_bits_per_weight"],
    )
    assert budget["target_achieved"] is False
    assert budget["receipt_statement"] == PHYSICAL_SUB2_NOT_ACHIEVED_STATEMENT

    assert artifact_path.exists()
    artifact_text = artifact_path.read_text(encoding="utf-8")
    payload = json.loads(artifact_text)
    assert payload["label"] == NATIVE_FULL_LOOP_ENGINEERING_RECEIPT_LABEL
    assert payload["artifact_hygiene"].startswith("compact runtime proof")
    assert len(artifact_text.encode("utf-8")) < 50_000
    assert '"q_levels"' not in artifact_text
    assert '"accumulators"' not in artifact_text
    print(
        "native_full_loop_engineering_receipt "
        f"artifact={artifact_path} label={receipt.label} "
        f"eligible={receipt.eligible_weight_count} "
        f"inclusive_bpw={budget['packed_inclusive_physical_bits_per_weight']:.6f} "
        f"q_data_bpw={budget['q_packed_data_bits_per_weight']:.6f} "
        f"q_metadata_bpw={budget['q_packed_metadata_bits_per_weight']:.6f} "
        f"acc_bpw={budget['acc_int16_bits_per_weight']:.6f} "
        f"scale_bpw={budget['frozen_scale_fp32_bits_per_weight']:.6f} "
        f"peak_allocated={receipt.peak_allocated_bytes} peak_reserved={receipt.peak_reserved_bytes} "
        f"no_leak_delta={receipt.no_leak_alloc_delta_bytes} "
        f"target_achieved={budget['target_achieved']}"
    )
