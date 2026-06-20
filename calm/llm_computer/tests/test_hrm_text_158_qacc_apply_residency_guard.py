"""CPU tests for B2-4 apply-input row residency guard."""
from __future__ import annotations

import pytest
import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import (
    DeviceGlobalRateCapStateRows,
)
from calm.hrm_text_158.native_full_stack.qacc_apply_residency_guard import (
    QAccApplyResidencyViolation,
    build_apply_inputs_from_python_list_legacy,
    composition_apply_residency_guard,
)


def test_qacc_apply_residency_guard_intercepts_cpu_transfer_in_torch_env() -> None:
    """Step-0 (CUDA): prove TorchDispatchMode intercepts a dispatched CPU copy op."""

    if not torch.cuda.is_available():
        pytest.fail("CUDA required for Step-0 CUDA residency-guard interception proof")
    device = torch.device("cuda")
    with composition_apply_residency_guard() as guard:
        row = guard.register_apply_input_row_tensor(torch.tensor([1, 2, 3], device=device))
        with pytest.raises(QAccApplyResidencyViolation, match="CPU transfer blocked"):
            row.detach().cpu()


def test_qacc_apply_residency_guard_intercepts_dispatched_cpu_copy_without_cuda() -> None:
    """Step-0 (CPU): prove dispatched _to_copy interception without CUDA registration."""

    with composition_apply_residency_guard() as guard:
        row = guard.register_dispatch_probe_tensor(torch.tensor([9, 8, 7], dtype=torch.int64))
        with pytest.raises(QAccApplyResidencyViolation, match="CPU transfer blocked"):
            row.to("cpu", copy=True)


def test_residency_guard_trips_on_registered_row_tensor_cpu_transfer() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with composition_apply_residency_guard() as guard:
        row = guard.register_apply_input_row_tensor(torch.tensor([4], device=device))
        with pytest.raises(QAccApplyResidencyViolation):
            row.to("cpu", copy=True)


def test_residency_guard_trips_on_python_list_row_source() -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA required for structural python-list provenance test")
    device = torch.device("cuda")
    accepted, directions, thresholds = build_apply_inputs_from_python_list_legacy(
        flat_indices=[0],
        directions=[1],
        thresholds=[2],
        device=device,
    )
    with composition_apply_residency_guard() as guard:
        with pytest.raises(QAccApplyResidencyViolation, match="not a registered device rows_by_state view"):
            guard.validate_apply_input_row_provenance(accepted, directions, thresholds)


def test_legacy_accepted_row_tensors_path_trips_before_refactor() -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA required for legacy python-list path test")
    from calm.hrm_text_158.native_full_stack.full_loop_receipt import (
        _legacy_accepted_row_tensors_python_list,
    )

    class _Row:
        def __init__(self, flat_index: int, local_pos: int) -> None:
            self.flat_index = flat_index
            self.local_pos = local_pos

    class _Plan:
        applied_directions = torch.tensor([1], dtype=torch.int16)
        applied_thresholds = torch.tensor([2], dtype=torch.int32)

    accepted, directions, thresholds = _legacy_accepted_row_tensors_python_list(
        _Plan(),
        (_Row(0, 0),),
        device=torch.device("cuda"),
    )
    with composition_apply_residency_guard() as guard:
        with pytest.raises(QAccApplyResidencyViolation):
            guard.validate_apply_input_row_provenance(accepted, directions, thresholds)


def test_post_apply_telemetry_outside_guard_does_not_greenwash() -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA required for telemetry-outside-guard test")
    device = torch.device("cuda")
    positions = torch.tensor([0], dtype=torch.int64, device=device)
    row_state_ids = torch.tensor([0], dtype=torch.int64, device=device)
    row_flat = torch.tensor([7], dtype=torch.int64, device=device)
    pos_cpu = positions.detach().cpu()
    _ = row_state_ids[pos_cpu].detach().cpu().tolist()
    _ = row_flat[pos_cpu].detach().cpu().tolist()

    with composition_apply_residency_guard() as guard:
        guard.register_apply_input_row_tensor(row_flat)
        report = guard.residency_report()
    assert report.cpu_selected_rows_materialized_before_q_acc_apply is False
    assert report.python_row_lists_materialized_before_q_acc_apply is False


def _device_state_rows() -> DeviceGlobalRateCapStateRows:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DeviceGlobalRateCapStateRows(
        state_key="proj_in",
        accepted_indices=torch.tensor([0], dtype=torch.int64, device=device),
        accepted_directions=torch.tensor([1], dtype=torch.int16, device=device),
        accepted_thresholds=torch.tensor([2], dtype=torch.int32, device=device),
        accepted_global_flat_indices=torch.tensor([0], dtype=torch.int64, device=device),
        deferred_indices=torch.empty(0, dtype=torch.int64, device=device),
        deferred_directions=torch.empty(0, dtype=torch.int16, device=device),
        deferred_thresholds=torch.empty(0, dtype=torch.int32, device=device),
        deferred_global_flat_indices=torch.empty(0, dtype=torch.int64, device=device),
    )


def test_clean_device_row_dispatch_reports_no_materialization_on_registered_rows() -> None:
    with composition_apply_residency_guard() as guard:
        state_rows = guard.register_device_state_rows_provenance(_device_state_rows())
        guard.validate_apply_input_row_provenance(
            state_rows.accepted_indices,
            state_rows.accepted_directions,
            state_rows.accepted_thresholds,
        )
        report = guard.residency_report()
    assert report.cpu_selected_rows_materialized_before_q_acc_apply is False
    assert report.python_row_lists_materialized_before_q_acc_apply is False
    assert report.accepted_row_source == "device_selection_result.rows_by_state"


def test_residency_guard_registers_replay_veto_provenance() -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA required for replay-veto provenance registration test")
    device = torch.device("cuda")
    replay_idx = torch.tensor([1], dtype=torch.int64, device=device)
    replay_dir = torch.tensor([-1], dtype=torch.int16, device=device)
    replay_thresh = torch.tensor([40000], dtype=torch.int32, device=device)
    with composition_apply_residency_guard() as guard:
        guard.register_replay_veto_provenance(replay_idx, replay_dir, replay_thresh)
        guard.validate_apply_input_row_provenance(replay_idx, replay_dir, replay_thresh)
        report = guard.residency_report()
    assert report.python_row_lists_materialized_before_q_acc_apply is False
