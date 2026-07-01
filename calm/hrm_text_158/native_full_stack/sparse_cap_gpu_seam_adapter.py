"""GPU global-rate-cap seam adapter for event-coded sparse cap apply (Slice B)."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
    merge_hot_table_arrays,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    C8StepObservation,
    EventCodedVoteUpdateState,
    apply_event_coded_carrier_step,
    apply_event_coded_integer_vote_update_from_plan,
    assert_c8_runtime_guards,
    c8_runtime_guard_stats,
    event_coded_new_acc_values_at,
    measure_persistent_dense_accumulator_materialized_numel,
    shape_only_accumulator_stub,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    GlobalRateCapResult,
    GlobalRateCapRow,
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    GlobalRateCapTensorResult,
    apply_global_rate_cap_reference,
    tensor_offsets_for_vote_update_states,
    validate_global_tie_rule_mode,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import (
    DeviceGlobalRateCapApplyResult,
    DeviceGlobalRateCapStateRows,
    GLOBAL_RATE_CAP_TORCH_CUDA_REFERENCE_SCOPE,
    RUN_GPU_GLOBAL_RATE_CAP_ENV,
    _selection_with_cpu_telemetry,
    _tensor_sha256,
    select_global_rate_cap_rows_torch_cuda_reference,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    RUN_GPU_Q_ACC_APPLY_ENV,
    VoteUpdateInputs,
    VoteUpdatePlan,
    VoteUpdateSpec,
    VoteUpdateState,
    _apply_threshold_residual_in_place,
)


def sparse_cap_gpu_lane_enabled() -> bool:
    return (
        os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_ENV) == "1"
        and os.environ.get(RUN_GPU_Q_ACC_APPLY_ENV) == "1"
    )


def materialize_sparse_new_acc_on_device(
    plan: VoteUpdatePlan,
    q_levels: torch.Tensor,
) -> torch.Tensor:
    """Scatter sparse event-coded backing into a GPU dense new_acc view for cap seam."""

    device = q_levels.device
    dense = torch.zeros(int(q_levels.numel()), dtype=torch.int32, device=device)
    if (
        plan.event_coded_sparse_active_idx is not None
        and plan.event_coded_sparse_post_active_i32 is not None
        and plan.event_coded_sparse_active_idx.numel() > 0
    ):
        active_idx = plan.event_coded_sparse_active_idx.to(device=device, dtype=torch.int64)
        post_active = plan.event_coded_sparse_post_active_i32.to(
            device=device,
            dtype=torch.int32,
        )
        dense[active_idx] = post_active
    return dense.view_as(q_levels)


def _mirror_plan_to_device(plan: VoteUpdatePlan, device: torch.device) -> VoteUpdatePlan:
    sparse_active = plan.event_coded_sparse_active_idx
    sparse_post = plan.event_coded_sparse_post_active_i32
    mirrored = replace(
        plan,
        q_i16=plan.q_i16.to(device),
        new_acc_i32=plan.new_acc_i32.to(device),
        candidate_indices=plan.candidate_indices.to(device),
        pre_veto_selected_indices=plan.pre_veto_selected_indices.to(device),
        applied_indices=plan.applied_indices.to(device),
        applied_directions=plan.applied_directions.to(device),
        applied_thresholds=plan.applied_thresholds.to(device),
        replay_ce_veto_indices=plan.replay_ce_veto_indices.to(device),
        replay_veto_directions=plan.replay_veto_directions.to(device),
        replay_veto_thresholds=plan.replay_veto_thresholds.to(device),
        pc_aux_negative_indices=plan.pc_aux_negative_indices.to(device),
        pc_aux_veto_indices=plan.pc_aux_veto_indices.to(device),
        event_coded_sparse_active_idx=(
            None if sparse_active is None else sparse_active.to(device)
        ),
        event_coded_sparse_post_active_i32=(
            None if sparse_post is None else sparse_post.to(device)
        ),
    )
    return mirrored


def _cuda_cap_mirror_device() -> torch.device:
    if not torch.cuda.is_available():
        raise ValueError("prepare_gpu_sparse_cap_inputs requires CUDA availability")
    return torch.device("cuda:0")


def _mirror_plan_row_tensors_to_device(
    plan: VoteUpdatePlan,
    device: torch.device,
) -> VoteUpdatePlan:
    """Mirror only cap-row plan tensors to CUDA; no dense new_acc materialization."""

    sparse_active = plan.event_coded_sparse_active_idx
    sparse_post = plan.event_coded_sparse_post_active_i32
    return replace(
        plan,
        candidate_indices=plan.candidate_indices.to(device),
        pre_veto_selected_indices=plan.pre_veto_selected_indices.to(device),
        applied_indices=plan.applied_indices.to(device),
        applied_directions=plan.applied_directions.to(device),
        applied_thresholds=plan.applied_thresholds.to(device),
        replay_ce_veto_indices=plan.replay_ce_veto_indices.to(device),
        replay_veto_directions=plan.replay_veto_directions.to(device),
        replay_veto_thresholds=plan.replay_veto_thresholds.to(device),
        pc_aux_negative_indices=plan.pc_aux_negative_indices.to(device),
        pc_aux_veto_indices=plan.pc_aux_veto_indices.to(device),
        event_coded_sparse_active_idx=(
            None if sparse_active is None else sparse_active.to(device)
        ),
        event_coded_sparse_post_active_i32=(
            None if sparse_post is None else sparse_post.to(device)
        ),
    )


def prepare_sparse_cap_selection_inputs(
    cap_inputs: list[GlobalRateCapTensorInput],
) -> list[GlobalRateCapTensorInput]:
    """CPU q authority with CUDA row tensors for sparse global-cap selection."""

    if not cap_inputs:
        return []
    device = _cuda_cap_mirror_device()
    prepared: list[GlobalRateCapTensorInput] = []
    for item in cap_inputs:
        plan = _mirror_plan_row_tensors_to_device(item.plan, device)
        prepared.append(
            GlobalRateCapTensorInput(
                state_key=item.state_key,
                state=item.state,
                plan=plan,
                vote_inputs=item.vote_inputs,
            )
        )
    return prepared


def prepare_gpu_sparse_cap_inputs(
    cap_inputs: list[GlobalRateCapTensorInput],
) -> list[GlobalRateCapTensorInput]:
    """Mirror persistent CPU q_levels to a transient CUDA view for the GPU cap seam."""

    if not cap_inputs:
        return []
    device = _cuda_cap_mirror_device()
    prepared: list[GlobalRateCapTensorInput] = []
    for item in cap_inputs:
        q_levels = (
            item.state.q_levels.detach()
            .to(device=device, dtype=torch.int8, non_blocking=True)
            .contiguous()
        )
        plan = _mirror_plan_to_device(item.plan, device)
        plan = replace(
            plan,
            new_acc_i32=materialize_sparse_new_acc_on_device(plan, q_levels),
        )
        prepared.append(
            GlobalRateCapTensorInput(
                state_key=item.state_key,
                state=VoteUpdateState(
                    q_levels=q_levels,
                    accumulators=item.state.accumulators.to(device),
                    accumulator_format=item.state.accumulator_format,
                ),
                plan=plan,
                vote_inputs=item.vote_inputs,
            )
        )
    return prepared


def _tuple_to_global_rate_cap_row(
    row_tuple: tuple[str, int, int, int, int, int],
) -> GlobalRateCapRow:
    state_key, flat_index, local_pos, global_flat_index, abs_new_acc, threshold_abs = row_tuple
    return GlobalRateCapRow(
        state_key=state_key,
        flat_index=int(flat_index),
        local_pos=int(local_pos),
        global_flat_index=int(global_flat_index),
        abs_new_acc=int(abs_new_acc),
        threshold_abs=int(threshold_abs),
        margin_abs_over_threshold=int(abs_new_acc) - int(threshold_abs),
    )


@dataclass(frozen=True)
class SparseCapGpuSeamApplyResult:
    tensor_results: list[GlobalRateCapTensorResult]
    step_summary: dict[str, Any]
    accepted_flat_by_key: dict[str, tuple[int, ...]]
    deferred_backlog: dict[str, dict[int, dict[str, int]]]
    gpu_apply: DeviceGlobalRateCapApplyResult


def adapt_device_global_rate_cap_apply_to_sparse_event_coded(
    gpu_apply: DeviceGlobalRateCapApplyResult,
    spec: GlobalRateCapSpec,
    *,
    tie_rule_mode: str = EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    contract_name: str | None = None,
) -> SparseCapGpuSeamApplyResult:
    """Map GPU DeviceGlobalRateCapApplyResult into sparse event-coded learner fields."""

    selection = gpu_apply.selection
    validate_global_tie_rule_mode(tie_rule_mode)
    rows = [_tuple_to_global_rate_cap_row(t) for t in selection.ordered_rows_as_tuples()]
    accepted_rows = [
        _tuple_to_global_rate_cap_row(t) for t in selection.accepted_rows_as_tuples()
    ]
    deferred_rows = [
        _tuple_to_global_rate_cap_row(t) for t in selection.deferred_rows_as_tuples()
    ]
    accepted_flat_by_key: dict[str, tuple[int, ...]] = {}
    for state_key, state_rows in selection.rows_by_state.items():
        accepted_flat_by_key[str(state_key)] = tuple(
            int(idx) for idx in state_rows.accepted_indices.detach().cpu().tolist()
        )
    backlog = copy.deepcopy(selection.deferred_backlog)
    accepted_count = len(accepted_rows)
    deferred_count = len(deferred_rows)
    step_summary = {
        **dict(gpu_apply.stats),
        "global_rate_cap_enabled": True,
        "event_coded_live_carrier_enabled": True,
        "event_coded_sparse_vote_authority": True,
        "event_coded_sparse_cap_enabled": True,
        "sparse_cap_gpu_seam": True,
        "transient_q_mirror_for_gpu_cap": True,
        "persistent_q_authority_device": "cpu",
        "cuda_q_not_saved_state": True,
        "global_rate_cap_accepted_count": accepted_count,
        "global_rate_cap_deferred_count": deferred_count,
        "global_rate_cap_applied_count": accepted_count if spec.mutate_outputs else 0,
        "q_changed_count": int(gpu_apply.stats.get("q_changed_count", 0)),
    }
    if contract_name is not None:
        step_summary["global_rate_cap_contract_name"] = str(contract_name)
    return SparseCapGpuSeamApplyResult(
        tensor_results=list(gpu_apply.tensor_results),
        step_summary=step_summary,
        accepted_flat_by_key=accepted_flat_by_key,
        deferred_backlog=backlog,
        gpu_apply=gpu_apply,
    )


def apply_sparse_event_coded_cap_rows_on_device(
    item: GlobalRateCapTensorInput,
    state_rows: DeviceGlobalRateCapStateRows,
    spec: GlobalRateCapSpec,
    *,
    scope: str = GLOBAL_RATE_CAP_TORCH_CUDA_REFERENCE_SCOPE,
) -> GlobalRateCapTensorResult:
    """Apply accepted/replay cap rows on one state without dense new_acc_i32."""

    q_cpu = item.state.q_levels.detach().cpu().contiguous()
    plan = item.plan

    accepted = state_rows.accepted_indices.detach().cpu().to(torch.int64)
    accepted_dirs = state_rows.accepted_directions.detach().cpu().to(torch.int16)
    accepted_thresholds = state_rows.accepted_thresholds.detach().cpu().to(torch.int32)
    replay = plan.replay_ce_veto_indices.detach().cpu().to(torch.int64).flatten()
    replay_dirs = plan.replay_veto_directions.detach().cpu().to(torch.int16).flatten()
    replay_thresholds = plan.replay_veto_thresholds.detach().cpu().to(torch.int32).flatten()

    if not bool(spec.mutate_outputs):
        q_out = q_cpu.detach().clone().contiguous()
        acc_out = item.state.accumulators.detach().cpu().clone().contiguous()
    else:
        q_i16 = q_cpu.flatten().to(torch.int16).clone()
        if accepted.numel() > 0:
            q_i16[accepted] = (q_i16[accepted] + accepted_dirs.to(torch.int16)).clamp(-1, 1)
            from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
                event_coded_new_acc_values_at,
            )

            acc_accepted = event_coded_new_acc_values_at(
                plan,
                accepted,
                fail_closed=True,
            )
            _apply_threshold_residual_in_place(
                acc_accepted,
                indices=torch.arange(int(accepted.numel()), dtype=torch.int64),
                directions=accepted_dirs,
                thresholds=accepted_thresholds,
            )
        if replay.numel() > 0:
            from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
                event_coded_new_acc_values_at,
            )

            acc_replay = event_coded_new_acc_values_at(
                plan,
                replay,
                fail_closed=True,
            )
            _apply_threshold_residual_in_place(
                acc_replay,
                indices=torch.arange(int(replay.numel()), dtype=torch.int64),
                directions=replay_dirs,
                thresholds=replay_thresholds,
            )
        q_out = q_i16.view_as(q_cpu).to(torch.int8).contiguous()
        acc_out = shape_only_accumulator_stub(q_out)

    q_changed = int((q_out != q_cpu).sum().item())
    stats = {
        **dict(item.plan.stats),
        "event_coded_sparse_gpu_cap_apply_bypass": True,
        "dense_new_acc_materialized_numel": 0,
        "q_changed_count": q_changed,
        "accepted_count": int(accepted.numel()),
        "replay_veto_count": int(replay.numel()),
        "scope": scope,
        "sparse_cap_apply_device": "cpu_row_bypass",
    }
    return GlobalRateCapTensorResult(
        state_key=item.state_key,
        q_levels=q_out,
        accumulators=acc_out,
        stats=stats,
    )


def apply_sparse_event_coded_cap_via_gpu_seam(
    *,
    cap_inputs: list[GlobalRateCapTensorInput],
    spec: GlobalRateCapSpec,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None = None,
    tie_rule_mode: str = EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    contract_name: str | None = None,
    tensor_offsets: dict[str, int] | None = None,
) -> SparseCapGpuSeamApplyResult:
    """Run CUDA margin cap selection+apply without full-numel q_levels CPU shim."""

    if not sparse_cap_gpu_lane_enabled():
        raise RuntimeError(
            f"{RUN_GPU_GLOBAL_RATE_CAP_ENV}=1 and {RUN_GPU_Q_ACC_APPLY_ENV}=1 required"
        )
    offsets = tensor_offsets or tensor_offsets_for_vote_update_states(cap_inputs)
    selection_inputs = prepare_sparse_cap_selection_inputs(cap_inputs)
    selection = select_global_rate_cap_rows_torch_cuda_reference(
        selection_inputs,
        spec,
        tensor_offsets=offsets,
        deferred_backlog=deferred_backlog,
        materialize_cpu_telemetry=False,
        event_coded_sparse_cap_enabled=True,
    )
    selection = _selection_with_cpu_telemetry(
        selection,
        spec,
        deferred_backlog=deferred_backlog,
        cpu_telemetry_timing="after_sparse_event_coded_gpu_cap_apply_bypass",
    )
    tensor_results: list[GlobalRateCapTensorResult] = []
    total_q_changed = 0
    for item in cap_inputs:
        state_rows = selection.rows_by_state[item.state_key]
        result = apply_sparse_event_coded_cap_rows_on_device(
            item,
            state_rows,
            spec,
        )
        tensor_results.append(result)
        total_q_changed += int(result.stats.get("q_changed_count", 0))
    gpu_apply = DeviceGlobalRateCapApplyResult(
        tensor_results=tensor_results,
        selection=selection,
        stats={
            **dict(selection.stats),
            "q_changed_count": total_q_changed,
            "event_coded_sparse_gpu_cap_apply_bypass": True,
            "dense_new_acc_materialized_numel": 0,
        },
    )
    return adapt_device_global_rate_cap_apply_to_sparse_event_coded(
        gpu_apply,
        spec,
        tie_rule_mode=tie_rule_mode,
        contract_name=contract_name,
    )


def _applied_plan_position(
    applied_flat: torch.Tensor,
    flat_index: int,
) -> int | None:
    matches = (applied_flat == int(flat_index)).nonzero(as_tuple=True)[0]
    if matches.numel() == 0:
        return None
    return int(matches[0].item())


def sync_event_coded_carrier_from_gpu_cap(
    carrier: EventCodedAccLiveState,
    q_gpu: torch.Tensor,
    plan: VoteUpdatePlan,
    accepted_local_indices: Sequence[int],
    *,
    q_persistent_cpu: torch.Tensor,
    step_index: int,
    host_allocator_site_emit: Callable[..., None] | None = None,
    optimizer_step_index: int | None = None,
    state_index: int | None = None,
) -> tuple[torch.Tensor, EventCodedAccLiveState]:
    """Sync live carrier from GPU cap q via accepted-index subset gather only."""

    def _site(site_id: str, suffix: str, line: int) -> None:
        if host_allocator_site_emit is None or state_index is None or int(state_index) != 0:
            return
        host_allocator_site_emit(
            site_id,
            suffix,
            origin_file="sparse_cap_gpu_seam_adapter.py",
            origin_line=int(line),
            optimizer_step_index=int(optimizer_step_index or step_index),
            state_index=0,
        )

    _site("C4.S2", "pre", 531)
    updated = carrier.cow_copy()
    _site("C4.S2a", "pre", 441)
    q_out = q_persistent_cpu.detach().cpu().clone().to(torch.int8)
    _site("C4.S2a", "post", 442)
    if not accepted_local_indices:
        apply_event_coded_carrier_step(updated, votes={}, step_index=int(step_index))
        _site("C4.S2", "post", 538)
        return q_out.contiguous(), updated

    device = q_gpu.device
    applied_flat = plan.applied_indices.to(device=device, dtype=torch.int64).flatten()
    threshold_flat = plan.applied_thresholds.to(device=device, dtype=torch.int64).flatten()
    direction_flat = plan.applied_directions.to(device=device, dtype=torch.int16).flatten()

    flat_indices = torch.tensor(
        [int(idx) for idx in accepted_local_indices],
        device=device,
        dtype=torch.int64,
    )
    _site("C4.S2b", "pre", 457)
    q_subset_cpu = q_gpu.flatten().index_select(0, flat_indices).detach().cpu()
    _site("C4.S2b", "post", 457)

    cap_indices: list[int] = []
    cap_values: list[int] = []
    q_flat = q_out.flatten()
    for subset_pos, flat_index in enumerate(accepted_local_indices):
        idx = int(flat_index)
        pos = _applied_plan_position(applied_flat, idx)
        if pos is None:
            continue
        direction = int(direction_flat[pos].item())
        q_val = int(q_subset_cpu[int(subset_pos)].item())
        q_flat[idx] = q_val
        updated.q_levels[idx] = q_val
        carry = int(updated.reconstruct_lane(idx))
        residual = carry - direction * int(threshold_flat[pos].item())
        cap_indices.append(idx)
        cap_values.append(int(residual))

    if cap_indices:
        _site("C4.S2c", "pre", 477)
        cap_idx = np.array(cap_indices, dtype=np.int32)
        cap_val = np.array(cap_values, dtype=np.int16)
        hot_idx, hot_val = merge_hot_table_arrays(
            updated._hot.indices_array(),
            updated._hot.values_array(),
            np.empty(0, dtype=np.int32),
            cap_idx,
            cap_val,
        )
        updated._hot.replace_arrays(hot_idx, hot_val)
        updated._invalidate_packed_caches()
        _site("C4.S2c", "post", 486)

    apply_event_coded_carrier_step(updated, votes={}, step_index=int(step_index))
    observation = C8StepObservation()
    persistent_dense = measure_persistent_dense_accumulator_materialized_numel(
        exact_accumulator_shadow=None,
        event_coded_live_carrier=updated,
        eligible_numel=int(q_out.numel()),
    )
    assert_c8_runtime_guards(
        updated,
        observation=observation,
        persistent_dense_accumulator_materialized_numel=persistent_dense,
    )
    _site("C4.S2", "post", 538)
    return q_out.contiguous(), updated


def apply_cap_tensor_result_gpu(
    item: GlobalRateCapTensorResult,
    *,
    event_states: Mapping[str, EventCodedVoteUpdateState],
    plans_by_key: Mapping[str, VoteUpdatePlan],
    inputs_by_key: Mapping[str, VoteUpdateInputs],
    vote_specs_by_key: Mapping[str, VoteUpdateSpec],
    accepted_flat_by_key: Mapping[str, Sequence[int]],
    local_selection_ordering_step: int,
    cap_boundary_transient: int,
    cap_item_stats: Mapping[str, Any],
    merge_stats_fn: Any,
    state_index: int = -1,
    host_allocator_site_emit: Callable[..., None] | None = None,
) -> tuple[str, EventCodedAccLiveState, torch.Tensor, dict[str, Any]]:
    """Apply one GPU cap tensor result through the sparse event-coded carrier path."""

    state_key = str(item.state_key)
    vu = event_states[state_key]
    plan = plans_by_key[state_key]
    if host_allocator_site_emit is not None and int(state_index) == 0:
        host_allocator_site_emit(
            "C4.S1",
            "pre",
            origin_file="sparse_cap_gpu_seam_adapter.py",
            origin_line=522,
            optimizer_step_index=int(local_selection_ordering_step),
            state_index=0,
        )
    local_result = apply_event_coded_integer_vote_update_from_plan(
        vu,
        inputs_by_key[state_key],
        vote_specs_by_key[state_key],
        plan,
        step_index=int(local_selection_ordering_step),
        cap_boundary_transient_dense=int(cap_boundary_transient),
        lightweight_runtime_stats=True,
    )
    if host_allocator_site_emit is not None and int(state_index) == 0:
        host_allocator_site_emit(
            "C4.S1",
            "post",
            origin_file="sparse_cap_gpu_seam_adapter.py",
            origin_line=530,
            optimizer_step_index=int(local_selection_ordering_step),
            state_index=0,
        )
    q_out, carrier = sync_event_coded_carrier_from_gpu_cap(
        local_result.carrier,
        item.q_levels,
        plan,
        accepted_flat_by_key[state_key],
        q_persistent_cpu=vu.q_levels,
        step_index=int(local_selection_ordering_step),
        host_allocator_site_emit=host_allocator_site_emit,
        optimizer_step_index=int(local_selection_ordering_step),
        state_index=int(state_index),
    )
    stats = merge_stats_fn(dict(cap_item_stats), local_result.stats)
    stats["sparse_cap_gpu_seam_q_source"] = "gpu_cap.accepted_subset_gather"
    stats["sparse_cap_gpu_seam_acc_source"] = "gpu_cap.tensor_results.accumulators"
    return state_key, carrier, q_out, stats


def parity_witness_tensors(
    cpu_result: GlobalRateCapResult,
    gpu_result: SparseCapGpuSeamApplyResult,
) -> dict[str, Any]:
    """Build LOCAL/GLOBAL/TENSOR parity witnesses for fixture assertions."""

    cpu_by_key = {item.state_key: item for item in cpu_result.tensor_results}
    witnesses: dict[str, Any] = {"per_state": {}}
    for gpu_item in gpu_result.tensor_results:
        state_key = gpu_item.state_key
        cpu_item = cpu_by_key[state_key]
        state_rows = gpu_result.gpu_apply.selection.rows_by_state[state_key]
        cpu_accepted_local = [
            int(row.flat_index)
            for row in cpu_result.accepted_rows
            if row.state_key == state_key
        ]
        cpu_deferred_local = [
            int(row.flat_index)
            for row in cpu_result.deferred_rows
            if row.state_key == state_key
        ]
        witnesses["per_state"][state_key] = {
            "accepted_local_gpu": state_rows.accepted_indices.detach().cpu().tolist(),
            "accepted_local_cpu": cpu_accepted_local,
            "deferred_local_gpu": state_rows.deferred_indices.detach().cpu().tolist(),
            "deferred_local_cpu": cpu_deferred_local,
            "accepted_global_sha_gpu": _tensor_sha256(state_rows.accepted_global_flat_indices),
            "accepted_global_sha_cpu": _tensor_sha256(
                torch.tensor(
                    [
                        int(row.global_flat_index)
                        for row in cpu_result.accepted_rows
                        if row.state_key == state_key
                    ],
                    dtype=torch.int64,
                )
            ),
            "q_sha_gpu": _tensor_sha256(gpu_item.q_levels),
            "q_sha_cpu": _tensor_sha256(cpu_item.q_levels),
        }
    return witnesses


def cpu_sparse_cap_oracle(
    cap_inputs: list[GlobalRateCapTensorInput],
    spec: GlobalRateCapSpec,
    *,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None = None,
    tie_rule_mode: str = EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    contract_name: str | None = None,
) -> GlobalRateCapResult:
    """CPU reference oracle for event-coded sparse cap parity."""

    cpu_inputs: list[GlobalRateCapTensorInput] = []
    for item in cap_inputs:
        q_cpu = item.state.q_levels.detach().cpu().contiguous()
        from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
            shape_only_accumulator_stub,
        )

        cpu_inputs.append(
            GlobalRateCapTensorInput(
                state_key=item.state_key,
                state=VoteUpdateState(
                    q_levels=q_cpu,
                    accumulators=shape_only_accumulator_stub(q_cpu),
                    accumulator_format=item.state.accumulator_format,
                ),
                plan=item.plan,
                vote_inputs=item.vote_inputs,
            )
        )
    return apply_global_rate_cap_reference(
        cpu_inputs,
        spec,
        deferred_backlog=deferred_backlog,
        tie_rule_mode=tie_rule_mode,
        contract_name=contract_name,
        event_coded_sparse_cap_enabled=True,
    )
