"""CPU tests for B2-4 composition q_acc_apply dispatcher."""
from __future__ import annotations

import os
from unittest import mock

import pytest
import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import (
    DeviceGlobalRateCapStateRows,
)
from calm.hrm_text_158.native_full_stack.qacc_apply_composition_dispatch import (
    RUN_GPU_Q_ACC_APPLY_NATIVE_ENV,
    _validate_cap_apply_residency,
    apply_cap_row_mutation_with_device_rows,
    q_acc_apply_mutation_under_cap_rows,
)
from calm.hrm_text_158.native_full_stack.qacc_apply_residency_guard import (
    QAccApplyResidencyViolation,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    QAccApplyMutationResult,
    RUN_GPU_Q_ACC_APPLY_ENV,
)


def _device_rows() -> DeviceGlobalRateCapStateRows:
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


def _base_tensors():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.tensor([0, 1], dtype=torch.int8, device=device)
    acc = torch.tensor([0, 0], dtype=torch.int16, device=device)
    return q, acc, device


@pytest.fixture(autouse=True)
def _lane_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RUN_GPU_Q_ACC_APPLY_ENV, "1")
    monkeypatch.delenv(RUN_GPU_Q_ACC_APPLY_NATIVE_ENV, raising=False)


def test_dispatcher_native_off_uses_reference_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    q, acc, device = _base_tensors()
    rows = _device_rows()
    monkeypatch.delenv(RUN_GPU_Q_ACC_APPLY_NATIVE_ENV, raising=False)
    reference = mock.Mock(
        return_value=QAccApplyMutationResult(
            q_levels=q,
            accumulators=acc,
            scope="test",
            backend="cuda",
            stats={},
        )
    )
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.qacc_apply_composition_dispatch.q_acc_apply_mutation_torch_cuda_reference_under_cap_rows",
        reference,
    )
    result = apply_cap_row_mutation_with_device_rows(
        q_levels=q,
        new_accumulators=acc.to(torch.int32),
        state_rows=rows,
        replay_veto_indices=torch.empty(0, dtype=torch.int64, device=device),
        replay_veto_directions=torch.empty(0, dtype=torch.int16, device=device),
        replay_veto_thresholds=torch.empty(0, dtype=torch.int32, device=device),
        mutate_outputs=True,
        original_accumulators=acc,
        scope="test_scope",
    )
    assert result.backend == "cuda"
    reference.assert_called_once()


def test_dispatcher_native_on_calls_triton_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA required for native-on dispatcher test")
    q, acc, device = _base_tensors()
    rows = _device_rows()
    monkeypatch.setenv(RUN_GPU_Q_ACC_APPLY_NATIVE_ENV, "1")
    token = mock.Mock(wrapper_launch_nonce="nonce")
    native = mock.Mock(return_value=(q, acc, token))
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.qacc_apply_composition_dispatch._TRITON_AVAILABLE",
        True,
    )
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.qacc_apply_composition_dispatch.apply_qacc_mutation_triton_native",
        native,
    )
    result = apply_cap_row_mutation_with_device_rows(
        q_levels=q,
        new_accumulators=acc.to(torch.int32),
        state_rows=rows,
        replay_veto_indices=torch.empty(0, dtype=torch.int64, device=device),
        replay_veto_directions=torch.empty(0, dtype=torch.int16, device=device),
        replay_veto_thresholds=torch.empty(0, dtype=torch.int32, device=device),
        mutate_outputs=True,
        original_accumulators=acc,
        scope="test_scope",
    )
    assert result.backend == "cuda_native_triton"
    native.assert_called_once()


def test_dispatcher_native_on_without_triton_fails_no_silent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA required for native-on no-triton test")
    q, acc, device = _base_tensors()
    rows = _device_rows()
    monkeypatch.setenv(RUN_GPU_Q_ACC_APPLY_NATIVE_ENV, "1")
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.qacc_apply_composition_dispatch._TRITON_AVAILABLE",
        False,
    )
    with pytest.raises(RuntimeError, match="reference fallback is forbidden"):
        apply_cap_row_mutation_with_device_rows(
            q_levels=q,
            new_accumulators=acc.to(torch.int32),
            state_rows=rows,
            replay_veto_indices=torch.empty(0, dtype=torch.int64, device=device),
            replay_veto_directions=torch.empty(0, dtype=torch.int16, device=device),
            replay_veto_thresholds=torch.empty(0, dtype=torch.int32, device=device),
            mutate_outputs=True,
            original_accumulators=acc,
            scope="test_scope",
        )


def test_unregistered_apply_inputs_raise_residency_violation() -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA required for unregistered apply-input test")
    q, acc, device = _base_tensors()
    bad_indices = torch.tensor([0], dtype=torch.int64, device=device)
    with pytest.raises(QAccApplyResidencyViolation):
        q_acc_apply_mutation_under_cap_rows(
            q_levels=q,
            new_accumulators=acc.to(torch.int32),
            accepted_indices=bad_indices,
            accepted_directions=torch.tensor([1], dtype=torch.int16, device=device),
            accepted_thresholds=torch.tensor([2], dtype=torch.int32, device=device),
            mutate_outputs=True,
            original_accumulators=acc,
            scope="test_scope",
        )


def test_full_loop_cap_path_uses_device_rows_not_python_lists() -> None:
    from calm.hrm_text_158.native_full_stack import full_loop_receipt

    assert hasattr(full_loop_receipt, "_legacy_accepted_row_tensors_python_list")
    source = open(full_loop_receipt.__file__, encoding="utf-8").read()
    assert "_apply_cap_rows_on_cuda" in source
    assert "apply_cap_row_mutation_with_device_rows" in source
    assert "select_global_rate_cap_rows_torch_cuda_reference" in source


def _device_rows_with_overlap_high_threshold() -> tuple[DeviceGlobalRateCapStateRows, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    overlap_index = 0
    return (
        DeviceGlobalRateCapStateRows(
            state_key="proj_in",
            accepted_indices=torch.tensor([overlap_index], dtype=torch.int64, device=device),
            accepted_directions=torch.tensor([1], dtype=torch.int16, device=device),
            accepted_thresholds=torch.tensor([40000], dtype=torch.int32, device=device),
            accepted_global_flat_indices=torch.tensor([overlap_index], dtype=torch.int64, device=device),
            deferred_indices=torch.empty(0, dtype=torch.int64, device=device),
            deferred_directions=torch.empty(0, dtype=torch.int16, device=device),
            deferred_thresholds=torch.empty(0, dtype=torch.int32, device=device),
            deferred_global_flat_indices=torch.empty(0, dtype=torch.int64, device=device),
        ),
        torch.tensor([overlap_index], dtype=torch.int64, device=device),
        torch.tensor([-1], dtype=torch.int16, device=device),
        torch.tensor([40000], dtype=torch.int32, device=device),
    )


def test_dispatcher_non_empty_replay_passes_through_both_seams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA required for non-empty replay dispatcher test")
    q, acc, device = _base_tensors()
    rows = _device_rows()
    replay_idx = torch.tensor([1], dtype=torch.int64, device=device)
    replay_dir = torch.tensor([-1], dtype=torch.int16, device=device)
    replay_thresh = torch.tensor([2], dtype=torch.int32, device=device)
    reference = mock.Mock(
        return_value=QAccApplyMutationResult(
            q_levels=q,
            accumulators=acc,
            scope="test",
            backend="cuda",
            stats={},
        )
    )
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.qacc_apply_composition_dispatch.q_acc_apply_mutation_torch_cuda_reference_under_cap_rows",
        reference,
    )
    result = apply_cap_row_mutation_with_device_rows(
        q_levels=q,
        new_accumulators=acc.to(torch.int32),
        state_rows=rows,
        replay_veto_indices=replay_idx,
        replay_veto_directions=replay_dir,
        replay_veto_thresholds=replay_thresh,
        mutate_outputs=True,
        original_accumulators=acc,
        scope="test_scope",
    )
    assert result.stats["replay_veto_count"] == 1
    assert result.stats["python_row_lists_materialized_before_q_acc_apply"] is False
    reference.assert_called_once()
    kwargs = reference.call_args.kwargs
    assert kwargs["replay_veto_indices"].numel() == 1


def test_dispatcher_accepted_replay_overlap_high_threshold_passes_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA required for accepted∩replay overlap high-threshold test")
    q = torch.tensor([0, 0, 0, 0], dtype=torch.int8, device="cuda")
    acc = torch.tensor([0, 0, 0, 0], dtype=torch.int16, device="cuda")
    new_acc = torch.tensor([90_000, 0, 0, 0], dtype=torch.int32, device="cuda")
    rows, replay_idx, replay_dir, replay_thresh = _device_rows_with_overlap_high_threshold()
    reference = mock.Mock(
        return_value=QAccApplyMutationResult(
            q_levels=q,
            accumulators=acc,
            scope="test",
            backend="cuda",
            stats={},
        )
    )
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.qacc_apply_composition_dispatch.q_acc_apply_mutation_torch_cuda_reference_under_cap_rows",
        reference,
    )
    result = apply_cap_row_mutation_with_device_rows(
        q_levels=q,
        new_accumulators=new_acc,
        state_rows=rows,
        replay_veto_indices=replay_idx,
        replay_veto_directions=replay_dir,
        replay_veto_thresholds=replay_thresh,
        mutate_outputs=True,
        original_accumulators=acc,
        scope="test_scope",
    )
    assert result.stats["accepted_count"] == 1
    assert result.stats["replay_veto_count"] == 1
    assert result.stats["cpu_selected_rows_materialized_before_q_acc_apply"] is False
    assert int(replay_thresh.item()) == 40000


def test_unregistered_replay_veto_raises_residency_violation() -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA required for unregistered replay-veto test")
    rows = _device_rows()
    bad_replay = torch.tensor([1], dtype=torch.int64, device=rows.accepted_indices.device)
    with pytest.raises(QAccApplyResidencyViolation, match="not a registered device rows_by_state view"):
        _validate_cap_apply_residency(
            state_rows=rows,
            replay_veto_indices=bad_replay,
            replay_veto_directions=torch.tensor([-1], dtype=torch.int16, device=bad_replay.device),
            replay_veto_thresholds=torch.tensor([2], dtype=torch.int32, device=bad_replay.device),
            register_replay_provenance=False,
        )


def test_native_routing_mints_token_without_guard_self_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA required for native routing no-self-trip test")
    from calm.hrm_text_158.native_full_stack.qacc_apply_triton_kernel import (
        _mint_qacc_apply_native_token,
    )

    q, acc, device = _base_tensors()
    rows = _device_rows()
    monkeypatch.setenv(RUN_GPU_Q_ACC_APPLY_NATIVE_ENV, "1")
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.qacc_apply_composition_dispatch._TRITON_AVAILABLE",
        True,
    )

    def _native_apply_with_real_mint(**kwargs):
        q_out = kwargs["q_levels"].clone()
        acc_out = kwargs["new_accumulators"].to(torch.int16)
        token = _mint_qacc_apply_native_token(
            q_levels=kwargs["q_levels"],
            new_accumulators=kwargs["new_accumulators"],
            accepted_indices=kwargs["accepted_indices"],
            accepted_directions=kwargs["accepted_directions"],
            accepted_thresholds=kwargs["accepted_thresholds"],
            replay_veto_indices=kwargs.get("replay_veto_indices"),
            replay_veto_directions=kwargs.get("replay_veto_directions"),
            replay_veto_thresholds=kwargs.get("replay_veto_thresholds"),
            original_accumulators=kwargs.get("original_accumulators"),
            q_out=q_out,
            acc_out=acc_out,
            wrapper_launch_nonce="composition-native-test",
        )
        return q_out, acc_out, token

    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.qacc_apply_composition_dispatch.apply_qacc_mutation_triton_native",
        _native_apply_with_real_mint,
    )
    result = apply_cap_row_mutation_with_device_rows(
        q_levels=q,
        new_accumulators=acc.to(torch.int32),
        state_rows=rows,
        replay_veto_indices=torch.empty(0, dtype=torch.int64, device=device),
        replay_veto_directions=torch.empty(0, dtype=torch.int16, device=device),
        replay_veto_thresholds=torch.empty(0, dtype=torch.int32, device=device),
        mutate_outputs=True,
        original_accumulators=acc,
        scope="test_scope",
    )
    assert result.backend == "cuda_native_triton"
    assert result.stats["wrapper_launch_nonce"] == "composition-native-test"
    assert isinstance(result.stats["wrapper_launch_nonce"], str)


def test_clean_device_row_dispatch_both_seams_report_no_materialization() -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA required for both-seams materialization test")
    q, acc, _device = _base_tensors()
    rows = _device_rows()
    result = apply_cap_row_mutation_with_device_rows(
        q_levels=q,
        new_accumulators=acc.to(torch.int32),
        state_rows=rows,
        replay_veto_indices=torch.empty(0, dtype=torch.int64, device=_device),
        replay_veto_directions=torch.empty(0, dtype=torch.int16, device=_device),
        replay_veto_thresholds=torch.empty(0, dtype=torch.int32, device=_device),
        mutate_outputs=True,
        original_accumulators=acc,
        scope="global_rate_cap_torch_cuda_reference_margin_only_no_policy_change",
    )
    assert result.stats["cpu_selected_rows_materialized_before_q_acc_apply"] is False
    assert result.stats["python_row_lists_materialized_before_q_acc_apply"] is False
