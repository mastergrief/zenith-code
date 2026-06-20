"""B2-4 env-gated composition dispatcher for q_acc_apply under cap rows."""
from __future__ import annotations

import os
from typing import Any

import torch

from calm.hrm_text_158.native_full_stack.qacc_apply_residency_guard import (
    QAccApplyResidencyViolation,
    composition_apply_residency_guard,
)
from calm.hrm_text_158.native_full_stack.qacc_apply_triton_kernel import (
    apply_qacc_mutation_triton_native,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    QAccApplyMutationResult,
    RUN_GPU_Q_ACC_APPLY_ENV,
    q_acc_apply_mutation_torch_cuda_reference_under_cap_rows,
)

RUN_GPU_Q_ACC_APPLY_NATIVE_ENV = "HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY_NATIVE"

try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def _require_lane_env() -> None:
    if os.environ.get(RUN_GPU_Q_ACC_APPLY_ENV) != "1":
        raise RuntimeError(
            f"{RUN_GPU_Q_ACC_APPLY_ENV}=1 is required and must only be set inside "
            "a granted gpu:0 resource lane"
        )


def _native_routing_enabled() -> bool:
    return os.environ.get(RUN_GPU_Q_ACC_APPLY_NATIVE_ENV) == "1"


def _merge_residency_stats(
    stats: dict[str, Any],
    guard_report: dict[str, bool | str],
) -> dict[str, Any]:
    merged = dict(stats)
    merged.update(guard_report)
    return merged


def _validate_cap_apply_residency(
    *,
    state_rows,
    replay_veto_indices: torch.Tensor,
    replay_veto_directions: torch.Tensor,
    replay_veto_thresholds: torch.Tensor,
    register_replay_provenance: bool = True,
) -> dict[str, bool | str]:
    with composition_apply_residency_guard() as guard:
        guard.register_device_state_rows_provenance(state_rows)
        if register_replay_provenance:
            guard.register_replay_veto_provenance(
                replay_veto_indices,
                replay_veto_directions,
                replay_veto_thresholds,
            )
        guard.validate_apply_input_row_provenance(
            state_rows.accepted_indices,
            state_rows.accepted_directions,
            state_rows.accepted_thresholds,
        )
        if replay_veto_indices.numel() > 0:
            guard.validate_apply_input_row_provenance(
                replay_veto_indices,
                replay_veto_directions,
                replay_veto_thresholds,
            )
        return guard.residency_report().to_dict()


def _validate_standalone_apply_residency(
    *,
    accepted_indices: torch.Tensor,
    accepted_directions: torch.Tensor,
    accepted_thresholds: torch.Tensor,
    replay_veto_indices: torch.Tensor | None,
    replay_veto_directions: torch.Tensor | None,
    replay_veto_thresholds: torch.Tensor | None,
) -> dict[str, bool | str]:
    with composition_apply_residency_guard() as guard:
        guard.validate_apply_input_row_provenance(
            accepted_indices,
            accepted_directions,
            accepted_thresholds,
        )
        if replay_veto_indices is not None and replay_veto_indices.numel() > 0:
            guard.validate_apply_input_row_provenance(
                replay_veto_indices,
                replay_veto_directions,
                replay_veto_thresholds,
            )
        return guard.residency_report().to_dict()


def q_acc_apply_mutation_under_cap_rows(
    *,
    q_levels: torch.Tensor,
    new_accumulators: torch.Tensor,
    accepted_indices: torch.Tensor,
    accepted_directions: torch.Tensor,
    accepted_thresholds: torch.Tensor,
    replay_veto_indices: torch.Tensor | None = None,
    replay_veto_directions: torch.Tensor | None = None,
    replay_veto_thresholds: torch.Tensor | None = None,
    mutate_outputs: bool = True,
    original_accumulators: torch.Tensor | None = None,
    scope: str,
    _residency_report: dict[str, bool | str] | None = None,
) -> QAccApplyMutationResult:
    """Dispatch cap-row q_acc_apply to reference or native Triton under env gates."""

    _require_lane_env()

    if _residency_report is None:
        report = _validate_standalone_apply_residency(
            accepted_indices=accepted_indices,
            accepted_directions=accepted_directions,
            accepted_thresholds=accepted_thresholds,
            replay_veto_indices=replay_veto_indices,
            replay_veto_directions=replay_veto_directions,
            replay_veto_thresholds=replay_veto_thresholds,
        )
        return q_acc_apply_mutation_under_cap_rows(
            q_levels=q_levels,
            new_accumulators=new_accumulators,
            accepted_indices=accepted_indices,
            accepted_directions=accepted_directions,
            accepted_thresholds=accepted_thresholds,
            replay_veto_indices=replay_veto_indices,
            replay_veto_directions=replay_veto_directions,
            replay_veto_thresholds=replay_veto_thresholds,
            mutate_outputs=mutate_outputs,
            original_accumulators=original_accumulators,
            scope=scope,
            _residency_report=report,
        )

    if _native_routing_enabled():
        if not _TRITON_AVAILABLE:
            raise RuntimeError(
                f"{RUN_GPU_Q_ACC_APPLY_NATIVE_ENV}=1 requires Triton; "
                "reference fallback is forbidden"
            )
        q_out, acc_out, token = apply_qacc_mutation_triton_native(
            q_levels=q_levels,
            new_accumulators=new_accumulators,
            accepted_indices=accepted_indices,
            accepted_directions=accepted_directions,
            accepted_thresholds=accepted_thresholds,
            replay_veto_indices=replay_veto_indices,
            replay_veto_directions=replay_veto_directions,
            replay_veto_thresholds=replay_veto_thresholds,
            original_accumulators=original_accumulators,
            mutate_outputs=mutate_outputs,
        )
        stats = _merge_residency_stats(
            {
                "scope": scope,
                "backend": "cuda_native_triton",
                "composition_native_routing": True,
                "mutate_outputs": bool(mutate_outputs),
                "accepted_count": int(accepted_indices.numel()),
                "replay_veto_count": int(
                    0 if replay_veto_indices is None else replay_veto_indices.numel()
                ),
                "q_changed_count": int((q_out != q_levels).sum().item()),
                "wrapper_launch_nonce": token.wrapper_launch_nonce,
            },
            _residency_report,
        )
        return QAccApplyMutationResult(
            q_levels=q_out,
            accumulators=acc_out,
            scope=scope,
            backend="cuda_native_triton",
            stats=stats,
        )

    apply_result = q_acc_apply_mutation_torch_cuda_reference_under_cap_rows(
        q_levels=q_levels,
        new_accumulators=new_accumulators,
        accepted_indices=accepted_indices,
        accepted_directions=accepted_directions,
        accepted_thresholds=accepted_thresholds,
        replay_veto_indices=replay_veto_indices,
        replay_veto_directions=replay_veto_directions,
        replay_veto_thresholds=replay_veto_thresholds,
        mutate_outputs=mutate_outputs,
        original_accumulators=original_accumulators,
        scope=scope,
    )
    stats = _merge_residency_stats(dict(apply_result.stats), _residency_report)
    stats["accepted_count"] = int(accepted_indices.numel())
    stats["replay_veto_count"] = int(
        0 if replay_veto_indices is None else replay_veto_indices.numel()
    )
    return QAccApplyMutationResult(
        q_levels=apply_result.q_levels,
        accumulators=apply_result.accumulators,
        scope=apply_result.scope,
        backend=apply_result.backend,
        stats=stats,
    )


def apply_cap_row_mutation_with_device_rows(
    *,
    q_levels: torch.Tensor,
    new_accumulators: torch.Tensor,
    state_rows,
    replay_veto_indices: torch.Tensor,
    replay_veto_directions: torch.Tensor,
    replay_veto_thresholds: torch.Tensor,
    mutate_outputs: bool,
    original_accumulators: torch.Tensor | None,
    scope: str,
) -> QAccApplyMutationResult:
    """Apply using device rows_by_state tensors with a bounded residency guard."""

    residency_report = _validate_cap_apply_residency(
        state_rows=state_rows,
        replay_veto_indices=replay_veto_indices,
        replay_veto_directions=replay_veto_directions,
        replay_veto_thresholds=replay_veto_thresholds,
    )
    return q_acc_apply_mutation_under_cap_rows(
        q_levels=q_levels,
        new_accumulators=new_accumulators,
        accepted_indices=state_rows.accepted_indices,
        accepted_directions=state_rows.accepted_directions,
        accepted_thresholds=state_rows.accepted_thresholds,
        replay_veto_indices=replay_veto_indices,
        replay_veto_directions=replay_veto_directions,
        replay_veto_thresholds=replay_veto_thresholds,
        mutate_outputs=mutate_outputs,
        original_accumulators=original_accumulators,
        scope=scope,
        _residency_report=residency_report,
    )
