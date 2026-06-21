"""B2-5a′/B2-5a″ env-gated native MARGIN-selection dispatch.

Single-block tile for row_count<=BLOCK; wider single-block compose for
BLOCK<row_count<=WIDER_CEILING.  Requires lane env + native selection env.
NO torch sort/gather fallback.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    DEFERRED_NON_SCOPE,
    GlobalRateCapOrderingMode,
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    tensor_offsets_for_vote_update_states,
    validate_global_rate_cap_inputs,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import (
    DeviceGlobalRateCapSelectionResult,
    RUN_GPU_GLOBAL_RATE_CAP_ENV,
    _empty_i64,
    _empty_state_rows,
    _rows_by_state_on_device,
    _safe_ratio,
    GLOBAL_RATE_CAP_GPU_SCHEMA_VERSION,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_feasibility_receipt import (
    canonical_tensor_payload_sha256,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_parity_receipt import (
    GLOBAL_RATE_CAP_MARGIN_SELECTION_NATIVE_PARITY_SCHEMA_VERSION,
    GlobalRateCapMarginSelectionNativeParityReceipt,
    build_global_rate_cap_margin_selection_native_parity_receipt,
    new_native_selection_token,
    validate_global_rate_cap_margin_selection_native_parity_receipt,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_path_audit import (
    run_full_native_path_audit,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_packed_key_scaffold import (
    LEGACY_RUN_GPU_GLOBAL_RATE_CAP_NATIVE_ENV,
    _compute_packed_key_budget,
    _device_row_tensors_for_selection,
    _reject_legacy_native_env,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_step0_budget_receipt import (
    TRITON_SINGLE_BLOCK_ROW_CEILING,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_triton_kernel import (
    INT64_MAX,
    MULTIBLOCK_COMPOSITION_SEAM,
    MarginSelectionSingleBlockDeferred,
    MarginSelectionSingleBlockTileResult,
    PADDING_SENTINEL,
    _TRITON_AVAILABLE,
    compute_host_max_full_key_python_int,
    evaluate_padding_headroom,
    margin_selection_single_block_tile,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_wider_single_block_compose import (
    WIDER_SINGLE_BLOCK_COMPOSE_SEAM,
    MarginSelectionWiderSingleBlockDeferred,
    MarginSelectionWiderSingleBlockResult,
    margin_selection_wider_single_block_compose,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_wider_single_block_triton_kernel import (
    WIDER_SINGLE_BLOCK_ROW_CEILING,
    WIDER_SINGLE_BLOCK_SORT_PADDED_N,
)

RUN_GPU_GLOBAL_RATE_CAP_NATIVE_SELECTION_ENV = (
    "HRM_TEXT_158_RUN_GPU_GLOBAL_RATE_CAP_NATIVE_SELECTION"
)
GLOBAL_RATE_CAP_MARGIN_SELECTION_NATIVE_SCOPE = (
    "global_rate_cap_margin_selection_native_single_block_tile"
)
GLOBAL_RATE_CAP_MARGIN_SELECTION_NATIVE_WIDER_SCOPE = (
    "global_rate_cap_margin_selection_native_wider_single_block"
)

_NATIVE_PATH_MODULES_SINGLE_BLOCK = (
    Path(__file__).with_name("global_rate_cap_margin_selection_triton_kernel.py"),
    Path(__file__).with_name("global_rate_cap_margin_selection_native_dispatch.py"),
)
_NATIVE_PATH_MODULES_WIDER = (
    Path(__file__).with_name("global_rate_cap_margin_selection_wider_single_block_triton_kernel.py"),
    Path(__file__).with_name("global_rate_cap_margin_selection_wider_single_block_compose.py"),
    Path(__file__).with_name("global_rate_cap_margin_selection_native_dispatch.py"),
)


def _require_lane_env() -> None:
    if os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_ENV) != "1":
        raise RuntimeError(
            f"{RUN_GPU_GLOBAL_RATE_CAP_ENV}=1 is required and must only be set "
            "inside a granted gpu:0 resource lane"
        )


def _native_selection_enabled() -> bool:
    return os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_NATIVE_SELECTION_ENV) == "1"


def _common_device(inputs: list[GlobalRateCapTensorInput]) -> torch.device:
    devices: set[torch.device] = set()
    for item in inputs:
        tensors = (
            item.state.q_levels,
            item.state.accumulators,
            item.plan.q_i16,
            item.plan.new_acc_i32,
            item.plan.applied_indices,
            item.plan.applied_directions,
            item.plan.applied_thresholds,
        )
        devices.update(tensor.device for tensor in tensors)
    if len(devices) != 1:
        raise ValueError(
            f"native MARGIN selection requires one shared device, got {sorted(map(str, devices))}"
        )
    return next(iter(devices))


def _empty_native_result(
    *,
    inputs: list[GlobalRateCapTensorInput],
    offsets: dict[str, int],
    spec: GlobalRateCapSpec,
    device: torch.device,
    scope: str,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None,
) -> DeviceGlobalRateCapSelectionResult:
    cap = max(0, int(spec.cap))
    empty_i64 = _empty_i64(device)
    empty_dirs = torch.empty(0, dtype=torch.int16, device=device)
    empty_thresh = torch.empty(0, dtype=torch.int32, device=device)
    rows_by_state = {
        item.state_key: _empty_state_rows(item.state_key, device=device) for item in inputs
    }
    stats = {
        "schema_version": GLOBAL_RATE_CAP_GPU_SCHEMA_VERSION,
        "scope": scope,
        "backend": device.type,
        "native_single_block_tile": True,
        "tile_primitive_seam": MULTIBLOCK_COMPOSITION_SEAM,
        "empty_branch_taken": True,
        "global_rate_cap_cap": cap,
        "global_pre_cap_would_apply_count": 0,
        "global_rate_cap_accepted_count": 0,
        "global_rate_cap_deferred_count": 0,
    }
    return DeviceGlobalRateCapSelectionResult(
        scope=scope,
        backend=device.type,
        state_keys=tuple(item.state_key for item in inputs),
        tensor_offsets=dict(offsets),
        cap=cap,
        row_state_ids=empty_i64,
        row_flat_indices=empty_i64,
        row_local_positions=empty_i64,
        row_global_flat_indices=empty_i64,
        row_abs_new_acc=empty_i64,
        row_thresholds=empty_thresh,
        row_directions=empty_dirs,
        accepted_positions=empty_i64,
        deferred_positions=empty_i64,
        rows_by_state=rows_by_state,
        deferred_backlog=dict(deferred_backlog) if deferred_backlog else {},
        stats=stats,
    )


def _defer_receipt(
    *,
    row_count: int,
    multiblock_deferred: bool,
    budget_infeasible: bool,
    host_max_full_key: int = -1,
    padding_headroom_ok: bool = False,
    full_pack_bits: int = -1,
    wider_single_block_regime: bool = False,
    caveats: tuple[str, ...] = (),
) -> GlobalRateCapMarginSelectionNativeParityReceipt:
    receipt = build_global_rate_cap_margin_selection_native_parity_receipt(
        selection_parity_pass=False,
        single_block_regime=row_count <= TRITON_SINGLE_BLOCK_ROW_CEILING,
        wider_single_block_regime=wider_single_block_regime,
        multiblock_deferred=multiblock_deferred,
        row_count=row_count,
        block=TRITON_SINGLE_BLOCK_ROW_CEILING,
        wider_ceiling=WIDER_SINGLE_BLOCK_ROW_CEILING,
        host_max_full_key=host_max_full_key,
        padding_sentinel=PADDING_SENTINEL,
        padding_headroom_ok=padding_headroom_ok,
        full_pack_bits=full_pack_bits,
        budget_infeasible=budget_infeasible,
        native_path_audit_pass=False,
        post_kernel_torch_permutation_detected=False,
        kernel_output_buffers_emitted=False,
        tile_primitive_seam=(
            WIDER_SINGLE_BLOCK_COMPOSE_SEAM
            if wider_single_block_regime
            else MULTIBLOCK_COMPOSITION_SEAM
        ),
        caveats=caveats,
    )
    validate_global_rate_cap_margin_selection_native_parity_receipt(receipt)
    return receipt


def _finalize_native_selection(
    *,
    inputs: list[GlobalRateCapTensorInput],
    offsets: dict[str, int],
    spec: GlobalRateCapSpec,
    device: torch.device,
    scope: str,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None,
    row_count: int,
    row_state_ids: torch.Tensor,
    row_flat_indices: torch.Tensor,
    row_local_positions: torch.Tensor,
    row_global_flat_indices: torch.Tensor,
    row_abs_new_acc: torch.Tensor,
    row_thresholds: torch.Tensor,
    row_directions: torch.Tensor,
    sort_padded_n: int,
    pos_width: int,
    host_max_full_key: int,
    padding_headroom_ok: bool,
    full_pack_bits: int,
    audit,
    bitonic_kernel_symbol: str,
    gather_kernel_symbol: str,
    kernel_source_sha256: str,
    tile_primitive_seam: str,
    single_block_regime: bool,
    wider_single_block_regime: bool,
    abs_new_acc_input: torch.Tensor,
    caveats: tuple[str, ...],
) -> tuple[DeviceGlobalRateCapSelectionResult, GlobalRateCapMarginSelectionNativeParityReceipt]:
    cap = max(0, int(spec.cap))
    take = min(cap, row_count)
    accepted_positions = torch.arange(take, dtype=torch.int64, device=device)
    deferred_positions = torch.arange(take, row_count, dtype=torch.int64, device=device)

    rows_by_state = _rows_by_state_on_device(
        inputs=inputs,
        row_state_ids=row_state_ids,
        row_flat_indices=row_flat_indices,
        row_global_flat_indices=row_global_flat_indices,
        row_thresholds=row_thresholds,
        row_directions=row_directions,
        accepted_positions=accepted_positions,
        deferred_positions=deferred_positions,
        device=device,
    )

    stats = {
        "schema_version": GLOBAL_RATE_CAP_MARGIN_SELECTION_NATIVE_PARITY_SCHEMA_VERSION,
        "scope": scope,
        "backend": device.type,
        "native_single_block_tile": single_block_regime,
        "native_wider_single_block": wider_single_block_regime,
        "tile_primitive_seam": tile_primitive_seam,
        "bitonic_kernel_symbol": bitonic_kernel_symbol,
        "gather_kernel_symbol": gather_kernel_symbol,
        "kernel_source_sha256": kernel_source_sha256,
        "sort_padded_n": sort_padded_n,
        "pos_width": pos_width,
        "host_max_full_key": host_max_full_key,
        "padding_headroom_ok": padding_headroom_ok,
        "functional_veto_policy": DEFERRED_NON_SCOPE,
        "bad_pressure_drain_policy": DEFERRED_NON_SCOPE,
        "global_rate_cap_cap": cap,
        "global_pre_cap_would_apply_count": row_count,
        "global_rate_cap_accepted_count": int(accepted_positions.numel()),
        "global_rate_cap_deferred_count": int(deferred_positions.numel()),
        "global_rate_cap_saturated": row_count > cap,
        "global_rate_cap_fill_ratio": _safe_ratio(int(accepted_positions.numel()), cap),
        "global_deferred_ratio": _safe_ratio(int(deferred_positions.numel()), row_count),
        "native_path_audit_pass": audit.native_path_audit_pass,
    }

    selection = DeviceGlobalRateCapSelectionResult(
        scope=scope,
        backend=device.type,
        state_keys=tuple(item.state_key for item in inputs),
        tensor_offsets=dict(offsets),
        cap=cap,
        row_state_ids=row_state_ids,
        row_flat_indices=row_flat_indices,
        row_local_positions=row_local_positions,
        row_global_flat_indices=row_global_flat_indices,
        row_abs_new_acc=row_abs_new_acc,
        row_thresholds=row_thresholds,
        row_directions=row_directions,
        accepted_positions=accepted_positions,
        deferred_positions=deferred_positions,
        rows_by_state=rows_by_state,
        deferred_backlog=dict(deferred_backlog) if deferred_backlog else {},
        stats=stats,
    )

    token = new_native_selection_token(
        bitonic_kernel_symbol=bitonic_kernel_symbol,
        gather_kernel_symbol=gather_kernel_symbol,
        kernel_source_sha256=kernel_source_sha256,
        selection_input_sha256=canonical_tensor_payload_sha256(abs_new_acc_input),
        ordered_output_sha256=canonical_tensor_payload_sha256(row_global_flat_indices),
        accepted_output_sha256=canonical_tensor_payload_sha256(
            row_global_flat_indices[accepted_positions]
        ),
        deferred_output_sha256=canonical_tensor_payload_sha256(
            row_global_flat_indices[deferred_positions]
            if deferred_positions.numel() > 0
            else row_global_flat_indices[:0]
        ),
        backend=device.type,
    )

    receipt = build_global_rate_cap_margin_selection_native_parity_receipt(
        selection_parity_pass=False,
        single_block_regime=single_block_regime,
        wider_single_block_regime=wider_single_block_regime,
        multiblock_deferred=False,
        row_count=row_count,
        block=TRITON_SINGLE_BLOCK_ROW_CEILING,
        wider_ceiling=WIDER_SINGLE_BLOCK_ROW_CEILING,
        sort_padded_n=sort_padded_n,
        pos_width=pos_width,
        host_max_full_key=host_max_full_key,
        padding_sentinel=INT64_MAX,
        padding_headroom_ok=padding_headroom_ok,
        full_pack_bits=full_pack_bits,
        budget_infeasible=False,
        native_path_audit_pass=audit.native_path_audit_pass,
        post_kernel_torch_permutation_detected=audit.post_kernel_torch_permutation_detected,
        kernel_output_buffers_emitted=True,
        tile_primitive_seam=tile_primitive_seam,
        token=token,
        parity_proof=None,
        caveats=caveats,
    )
    validate_global_rate_cap_margin_selection_native_parity_receipt(receipt)
    return selection, receipt


def select_global_rate_cap_rows_margin_native(
    inputs: list[GlobalRateCapTensorInput],
    spec: GlobalRateCapSpec,
    *,
    tensor_offsets: dict[str, int] | None = None,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None = None,
    scope: str = GLOBAL_RATE_CAP_MARGIN_SELECTION_NATIVE_SCOPE,
) -> tuple[DeviceGlobalRateCapSelectionResult, GlobalRateCapMarginSelectionNativeParityReceipt]:
    """Native single-block MARGIN selection via @triton.jit tile primitive."""

    _reject_legacy_native_env()
    _require_lane_env()
    if not _native_selection_enabled():
        raise RuntimeError(
            f"{RUN_GPU_GLOBAL_RATE_CAP_NATIVE_SELECTION_ENV}=1 is required for native selection"
        )
    if os.environ.get(LEGACY_RUN_GPU_GLOBAL_RATE_CAP_NATIVE_ENV) == "1":
        raise RuntimeError(f"{LEGACY_RUN_GPU_GLOBAL_RATE_CAP_NATIVE_ENV}=1 is fail-closed")

    if not _TRITON_AVAILABLE:
        raise RuntimeError("native selection requires Triton import; no fallback")

    spec.validate()
    if spec.normalized_ordering_mode != GlobalRateCapOrderingMode.MARGIN:
        raise NotImplementedError("native selection supports MARGIN ordering only")
    validate_global_rate_cap_inputs(inputs)

    offsets = tensor_offsets or tensor_offsets_for_vote_update_states(inputs)
    device = _common_device(inputs)
    rows = _device_row_tensors_for_selection(inputs, tensor_offsets=offsets, device=device)
    row_count = int(rows["global_indices"].numel())

    if row_count == 0:
        selection = _empty_native_result(
            inputs=inputs,
            offsets=offsets,
            spec=spec,
            device=device,
            scope=scope,
            deferred_backlog=deferred_backlog,
        )
        receipt = build_global_rate_cap_margin_selection_native_parity_receipt(
            selection_parity_pass=False,
            single_block_regime=True,
            multiblock_deferred=False,
            row_count=0,
            block=TRITON_SINGLE_BLOCK_ROW_CEILING,
            native_path_audit_pass=True,
            kernel_output_buffers_emitted=False,
            tile_primitive_seam=MULTIBLOCK_COMPOSITION_SEAM,
            caveats=("empty branch: no kernel launch",),
        )
        validate_global_rate_cap_margin_selection_native_parity_receipt(receipt)
        return selection, receipt

    if row_count > WIDER_SINGLE_BLOCK_ROW_CEILING:
        selection = _empty_native_result(
            inputs=inputs,
            offsets=offsets,
            spec=spec,
            device=device,
            scope=scope,
            deferred_backlog=deferred_backlog,
        )
        receipt = _defer_receipt(
            row_count=row_count,
            multiblock_deferred=True,
            budget_infeasible=False,
            wider_single_block_regime=True,
            caveats=(
                f"row_count={row_count} > WIDER_CEILING={WIDER_SINGLE_BLOCK_ROW_CEILING}; "
                "fail-closed",
            ),
        )
        return selection, receipt

    if row_count > TRITON_SINGLE_BLOCK_ROW_CEILING:
        budget = _compute_packed_key_budget(
            global_flat_indices=rows["global_indices"],
            abs_new_acc=rows["abs_new_acc"],
        )
        pos_width = max(1, (row_count - 1).bit_length()) if row_count > 0 else 1
        rank_bits = (
            max(1, budget.max_abs_observed.bit_length()) if budget.max_abs_observed > 0 else 1
        )
        full_pack_bits = rank_bits + budget.index_width + pos_width
        host_max = compute_host_max_full_key_python_int(
            abs_new_acc=rows["abs_new_acc"],
            global_flat_indices=rows["global_indices"],
            max_abs_observed=budget.max_abs_observed,
            index_width=budget.index_width,
            row_count=row_count,
        )
        headroom = evaluate_padding_headroom(
            host_max_full_key=host_max,
            full_pack_bits=full_pack_bits,
        )
        if headroom["budget_infeasible"]:
            selection = _empty_native_result(
                inputs=inputs,
                offsets=offsets,
                spec=spec,
                device=device,
                scope=GLOBAL_RATE_CAP_MARGIN_SELECTION_NATIVE_WIDER_SCOPE,
                deferred_backlog=deferred_backlog,
            )
            receipt = _defer_receipt(
                row_count=row_count,
                multiblock_deferred=False,
                budget_infeasible=True,
                wider_single_block_regime=True,
                host_max_full_key=host_max,
                padding_headroom_ok=bool(headroom["padding_headroom_ok"]),
                full_pack_bits=full_pack_bits,
                caveats=("padding headroom / budget infeasible; no kernel launch",),
            )
            return selection, receipt

        try:
            wider = margin_selection_wider_single_block_compose(rows, device=device)
        except MarginSelectionWiderSingleBlockDeferred as exc:
            selection = _empty_native_result(
                inputs=inputs,
                offsets=offsets,
                spec=spec,
                device=device,
                scope=GLOBAL_RATE_CAP_MARGIN_SELECTION_NATIVE_WIDER_SCOPE,
                deferred_backlog=deferred_backlog,
            )
            receipt = _defer_receipt(
                row_count=row_count,
                multiblock_deferred=True,
                budget_infeasible=True,
                wider_single_block_regime=True,
                host_max_full_key=host_max,
                padding_headroom_ok=bool(headroom["padding_headroom_ok"]),
                full_pack_bits=full_pack_bits,
                caveats=(str(exc),),
            )
            return selection, receipt

        audit = run_full_native_path_audit(
            module_paths=_NATIVE_PATH_MODULES_WIDER,
            wider=wider,
        )
        return _finalize_native_selection(
            inputs=inputs,
            offsets=offsets,
            spec=spec,
            device=device,
            scope=GLOBAL_RATE_CAP_MARGIN_SELECTION_NATIVE_WIDER_SCOPE,
            deferred_backlog=deferred_backlog,
            row_count=row_count,
            row_state_ids=wider.row_state_ids,
            row_flat_indices=wider.row_flat_indices,
            row_local_positions=wider.row_local_positions,
            row_global_flat_indices=wider.row_global_flat_indices,
            row_abs_new_acc=wider.row_abs_new_acc,
            row_thresholds=wider.row_thresholds,
            row_directions=wider.row_directions,
            sort_padded_n=wider.sort_padded_n,
            pos_width=wider.pos_width,
            host_max_full_key=wider.host_max_full_key,
            padding_headroom_ok=wider.padding_headroom_ok,
            full_pack_bits=full_pack_bits,
            audit=audit,
            bitonic_kernel_symbol=wider.bitonic_kernel_symbol,
            gather_kernel_symbol=wider.gather_kernel_symbol,
            kernel_source_sha256=wider.kernel_source_sha256,
            tile_primitive_seam=WIDER_SINGLE_BLOCK_COMPOSE_SEAM,
            single_block_regime=False,
            wider_single_block_regime=True,
            abs_new_acc_input=rows["abs_new_acc"],
            caveats=(
                "realistic-size selection_parity_pass requires apply_native_selection_parity_proof",
                f"proven wider ceiling={WIDER_SINGLE_BLOCK_SORT_PADDED_N}",
            ),
        )

    budget = _compute_packed_key_budget(
        global_flat_indices=rows["global_indices"],
        abs_new_acc=rows["abs_new_acc"],
    )
    pos_width = max(1, (row_count - 1).bit_length()) if row_count > 0 else 1
    rank_bits = max(1, budget.max_abs_observed.bit_length()) if budget.max_abs_observed > 0 else 1
    full_pack_bits = rank_bits + budget.index_width + pos_width
    host_max = compute_host_max_full_key_python_int(
        abs_new_acc=rows["abs_new_acc"],
        global_flat_indices=rows["global_indices"],
        max_abs_observed=budget.max_abs_observed,
        index_width=budget.index_width,
        row_count=row_count,
    )
    headroom = evaluate_padding_headroom(
        host_max_full_key=host_max,
        full_pack_bits=full_pack_bits,
    )
    if headroom["budget_infeasible"]:
        selection = _empty_native_result(
            inputs=inputs,
            offsets=offsets,
            spec=spec,
            device=device,
            scope=scope,
            deferred_backlog=deferred_backlog,
        )
        receipt = _defer_receipt(
            row_count=row_count,
            multiblock_deferred=False,
            budget_infeasible=True,
            host_max_full_key=host_max,
            padding_headroom_ok=bool(headroom["padding_headroom_ok"]),
            full_pack_bits=full_pack_bits,
            caveats=("padding headroom / budget infeasible; no kernel launch",),
        )
        return selection, receipt

    try:
        tile = margin_selection_single_block_tile(rows, device=device)
    except MarginSelectionSingleBlockDeferred as exc:
        selection = _empty_native_result(
            inputs=inputs,
            offsets=offsets,
            spec=spec,
            device=device,
            scope=scope,
            deferred_backlog=deferred_backlog,
        )
        receipt = _defer_receipt(
            row_count=row_count,
            multiblock_deferred=row_count > TRITON_SINGLE_BLOCK_ROW_CEILING,
            budget_infeasible=True,
            host_max_full_key=host_max,
            padding_headroom_ok=bool(headroom["padding_headroom_ok"]),
            full_pack_bits=full_pack_bits,
            caveats=(str(exc),),
        )
        return selection, receipt

    audit = run_full_native_path_audit(module_paths=_NATIVE_PATH_MODULES_SINGLE_BLOCK, tile=tile)

    rank_bits = max(1, budget.max_abs_observed.bit_length()) if budget.max_abs_observed > 0 else 1
    full_pack_bits = rank_bits + budget.index_width + tile.pos_width
    return _finalize_native_selection(
        inputs=inputs,
        offsets=offsets,
        spec=spec,
        device=device,
        scope=scope,
        deferred_backlog=deferred_backlog,
        row_count=row_count,
        row_state_ids=tile.row_state_ids,
        row_flat_indices=tile.row_flat_indices,
        row_local_positions=tile.row_local_positions,
        row_global_flat_indices=tile.row_global_flat_indices,
        row_abs_new_acc=tile.row_abs_new_acc,
        row_thresholds=tile.row_thresholds,
        row_directions=tile.row_directions,
        sort_padded_n=tile.sort_padded_n,
        pos_width=tile.pos_width,
        host_max_full_key=tile.host_max_full_key,
        padding_headroom_ok=tile.padding_headroom_ok,
        full_pack_bits=full_pack_bits,
        audit=audit,
        bitonic_kernel_symbol=tile.bitonic_kernel_symbol,
        gather_kernel_symbol=tile.gather_kernel_symbol,
        kernel_source_sha256=tile.kernel_source_sha256,
        tile_primitive_seam=MULTIBLOCK_COMPOSITION_SEAM,
        single_block_regime=True,
        wider_single_block_regime=False,
        abs_new_acc_input=rows["abs_new_acc"],
        caveats=(
            "selection_parity_pass requires apply_native_selection_parity_proof after oracle compare",
        ),
    )


__all__ = [
    "GLOBAL_RATE_CAP_MARGIN_SELECTION_NATIVE_SCOPE",
    "GLOBAL_RATE_CAP_MARGIN_SELECTION_NATIVE_WIDER_SCOPE",
    "RUN_GPU_GLOBAL_RATE_CAP_NATIVE_SELECTION_ENV",
    "select_global_rate_cap_rows_margin_native",
]
