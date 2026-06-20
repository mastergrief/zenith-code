"""B2-5a packed-key CPU scaffold for global-cap MARGIN selection (feasibility null).

COMMITTED NULL after direction convergence (1781986822527): no native
``@triton.jit`` argsort-with-permutation kernel exists in this slice.  This
module retains the packed int64 total-order key
``(rank << index_width) | global_flat_index`` and a clearly labeled **NON-NATIVE
CPU scaffold** that orders via ``torch.topk`` (extrema selection, NOT
``torch.sort`` / ``torch.argsort``).  ``selection_parity_pass`` is PERMANENTLY
False; no native pass is mintable.

Scope = frozen plan ``1781984159546-c894af21``.  Does NOT mutate q/acc, does NOT
edit the ledger, does NOT flip ``global_cap_margin_only_reference``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any

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
    DeviceGlobalRateCapStateRows,
    _empty_i64,
    _empty_state_rows,
    _rows_by_state_on_device,
    _safe_ratio,
    GLOBAL_RATE_CAP_GPU_SCHEMA_VERSION,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_feasibility_receipt import (
    GLOBAL_RATE_CAP_MARGIN_SELECTION_FEASIBILITY_SCHEMA_VERSION,
    GLOBAL_RATE_CAP_MARGIN_SELECTION_FEASIBILITY_NON_CLAIMS,
    GlobalRateCapMarginSelectionFeasibilityReceipt,
    GlobalRateCapSelectionScaffoldToken,
    build_global_rate_cap_margin_selection_feasibility_receipt,
    canonical_tensor_payload_sha256,
    new_selection_token,
    validate_global_rate_cap_margin_selection_feasibility_receipt,
)

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

_KERNEL_FILE = Path(__file__)


def _kernel_source_sha256() -> str:
    return hashlib.sha256(_KERNEL_FILE.read_bytes()).hexdigest()


RUN_GPU_GLOBAL_RATE_CAP_SCAFFOLD_ENV = "HRM_TEXT_158_RUN_GPU_GLOBAL_RATE_CAP_SCAFFOLD"
LEGACY_RUN_GPU_GLOBAL_RATE_CAP_NATIVE_ENV = "HRM_TEXT_158_RUN_GPU_GLOBAL_RATE_CAP_NATIVE"
GLOBAL_RATE_CAP_MARGIN_SELECTION_SCAFFOLD_SCOPE = (
    "global_rate_cap_margin_selection_packed_key_no_policy_change"
)
GLOBAL_RATE_CAP_MARGIN_SELECTION_SCAFFOLD_SCHEMA_VERSION = (
    GLOBAL_RATE_CAP_MARGIN_SELECTION_FEASIBILITY_SCHEMA_VERSION
)
GLOBAL_RATE_CAP_MARGIN_SELECTION_SCAFFOLD_NON_CLAIMS = (
    GLOBAL_RATE_CAP_MARGIN_SELECTION_FEASIBILITY_NON_CLAIMS
)


def _reject_legacy_native_env() -> None:
    """Fail-closed: prior ``..._NATIVE`` env must not arm the CPU scaffold."""
    if os.environ.get(LEGACY_RUN_GPU_GLOBAL_RATE_CAP_NATIVE_ENV) == "1":
        raise RuntimeError(
            f"{LEGACY_RUN_GPU_GLOBAL_RATE_CAP_NATIVE_ENV}=1 is fail-closed on B2-5a null; "
            f"prior native env name removed — use {RUN_GPU_GLOBAL_RATE_CAP_SCAFFOLD_ENV}=1 "
            "for CPU scaffold routing only"
        )


class GlobalRateCapMarginSelectionFeasibilityNull(RuntimeError):
    """Raised (and caught for receipts) when the corrected bit-budget / uniqueness
    / lower-bound / empty-branch checks fail for the observed domain.  Documented
    null, not a forced implementation."""


@dataclass(frozen=True)
class _PackedKeyBudget:
    index_width: int
    max_abs_observed: int
    max_global_flat_index: int


def _device_row_tensors_for_selection(
    inputs: list[GlobalRateCapTensorInput],
    *,
    tensor_offsets: dict[str, int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Materialize per-row device tensors (parallel to the reference
    ``_device_row_tensors`` in ``global_rate_cap_gpu.py``).  Reuses/replicates
    the reference's negative-offset rejection at ``global_rate_cap_gpu.py:534``
    BEFORE key packing."""

    state_ids: list[torch.Tensor] = []
    flat_indices: list[torch.Tensor] = []
    local_positions: list[torch.Tensor] = []
    global_indices: list[torch.Tensor] = []
    abs_new_acc: list[torch.Tensor] = []
    thresholds: list[torch.Tensor] = []
    directions: list[torch.Tensor] = []
    for state_id, item in enumerate(inputs):
        if item.state_key not in tensor_offsets:
            raise ValueError(f"missing tensor offset for {item.state_key!r}")
        indices = item.plan.applied_indices.flatten().to(device=device, dtype=torch.int64)
        count = int(indices.numel())
        if count == 0:
            continue
        offset = int(tensor_offsets[item.state_key])
        if offset < 0:
            raise ValueError(
                f"tensor offset for {item.state_key!r} must be >= 0, got {offset}"
            )
        flat_acc = item.plan.new_acc_i32.flatten().to(device=device, dtype=torch.int64)
        state_ids.append(torch.full((count,), state_id, dtype=torch.int64, device=device))
        flat_indices.append(indices)
        local_positions.append(torch.arange(count, dtype=torch.int64, device=device))
        global_indices.append(indices + offset)
        abs_new_acc.append(flat_acc[indices].abs().to(torch.int64))
        thresholds.append(
            item.plan.applied_thresholds.flatten().to(device=device, dtype=torch.int64)
        )
        directions.append(
            item.plan.applied_directions.flatten().to(device=device, dtype=torch.int16)
        )

    if not state_ids:
        return {
            "state_ids": _empty_i64(device),
            "flat_indices": _empty_i64(device),
            "local_positions": _empty_i64(device),
            "global_indices": _empty_i64(device),
            "abs_new_acc": _empty_i64(device),
            "thresholds": _empty_i64(device),
            "directions": torch.empty(0, dtype=torch.int16, device=device),
        }
    return {
        "state_ids": torch.cat(state_ids),
        "flat_indices": torch.cat(flat_indices),
        "local_positions": torch.cat(local_positions),
        "global_indices": torch.cat(global_indices),
        "abs_new_acc": torch.cat(abs_new_acc),
        "thresholds": torch.cat(thresholds),
        "directions": torch.cat(directions),
    }


def _compute_packed_key_budget(
    *,
    global_flat_indices: torch.Tensor,
    abs_new_acc: torch.Tensor,
) -> _PackedKeyBudget:
    """Compute the corrected packed-key bit-budget.

    ``index_width = bit_length(max_global_flat_index)`` (NOT num_rows-width).

    Five hard asserts split by failure class:

    - Caller-validation asserts -> ``ValueError`` (rejected before pass-mint):
      (a)  every ``global_flat_index < (1 << index_width)`` (structural)
      (a2) every ``global_flat_index >= 0`` (reuses gpu.py:534 neg-offset reject)
      (b)  no duplicate ``global_flat_index`` (overlapping offsets)
    - Domain-feasibility asserts -> ``GlobalRateCapMarginSelectionFeasibilityNull``
      (documented null, NOT a forced impl):
      (c)  all ``rank >= 0`` (abs domain non-negative)
      (d)  ``rank <= ((2**63 - 1) >> index_width)`` (no signed-int64 overflow)
    """

    max_global_flat_index = int(global_flat_indices.max().item())
    if max_global_flat_index < 0:
        raise ValueError(
            f"assert (a2) violated: max_global_flat_index={max_global_flat_index} < 0"
            " (negative tensor offset)"
        )
    index_width = max(1, int(max_global_flat_index).bit_length())
    upper_bound = 1 << index_width
    if not torch.all(global_flat_indices < upper_bound):
        bad = int((global_flat_indices >= upper_bound).sum().item())
        raise ValueError(
            f"assert (a) violated: {bad} global_flat_index values >= 2**index_width"
            f"={upper_bound}"
        )
    if int((global_flat_indices < 0).sum().item()) > 0:
        bad = int((global_flat_indices < 0).sum().item())
        raise ValueError(
            f"assert (a2) violated: {bad} global_flat_index values < 0 (negative offset)"
        )
    if int(global_flat_indices.unique().numel()) != int(global_flat_indices.numel()):
        raise ValueError(
            "assert (b) violated: duplicate global_flat_index values detected"
            " (overlapping tensor_offsets)"
        )
    max_abs_observed = int(abs_new_acc.max().item())
    if max_abs_observed < 0:
        raise GlobalRateCapMarginSelectionFeasibilityNull(
            f"assert (c) violated: max_abs_observed={max_abs_observed} < 0"
            " (abs domain went signed-native)"
        )
    if max_abs_observed > ((2 ** 63 - 1) >> index_width):
        raise GlobalRateCapMarginSelectionFeasibilityNull(
            f"assert (d) violated: max_abs_observed={max_abs_observed} > "
            f"{(2 ** 63 - 1) >> index_width} for index_width={index_width} "
            "(signed-int64 overflow in packed key)"
        )
    return _PackedKeyBudget(
        index_width=index_width,
        max_abs_observed=max_abs_observed,
        max_global_flat_index=max_global_flat_index,
    )


def _packed_total_order_key(
    *,
    abs_new_acc: torch.Tensor,
    global_flat_indices: torch.Tensor,
    budget: _PackedKeyBudget,
) -> torch.Tensor:
    """Build the packed int64 total-order key.

    ``key = (rank << index_width) | global_flat_index`` where
    ``rank = max_abs_observed - abs_new_acc`` (non-negative, assert (c)).
    Ascending order on ``key`` reproduces the oracle tie-break
    (DESC abs, ASC global_flat_index) via the NON-NATIVE CPU scaffold
    (``_scaffold_sort_keys`` / ``torch.topk`` extrema selection).
    """

    rank = budget.max_abs_observed - abs_new_acc
    if int((rank < 0).sum().item()) > 0:
        raise GlobalRateCapMarginSelectionFeasibilityNull(
            "assert (c) violated: negative rank computed (abs domain went signed-native)"
        )
    keys = (rank << budget.index_width) | global_flat_indices.to(torch.int64)
    return keys.to(torch.int64)


def _scaffold_sort_keys(keys: torch.Tensor) -> torch.Tensor:
    """Return the ascending-order permutation via NON-NATIVE CPU scaffold.

  ``torch.topk`` with ``largest=False, sorted=True`` selects the k smallest
  packed keys in ascending order.  This is extrema selection, NOT
  ``torch.sort`` / ``torch.argsort``.  The packed key is a strict total order
  (unique ``global_flat_index``), so stability is irrelevant.

  Triton ``tl.sort`` (standard.py:423-470) is value-returning, block-local,
  static-shape, and returns NO argsort/permutation — it is NOT used here and
  does NOT provide a drop-in native argsort for MARGIN row permutation.
    """

    n = int(keys.numel())
    if n == 0:
        return torch.empty(0, dtype=torch.int64, device=keys.device)
    # NON-NATIVE CPU scaffold only — no @triton.jit kernel, no native pass.
    _, order = torch.topk(keys, n, largest=False, sorted=True)
    return order


def select_global_rate_cap_rows_margin_scaffold(
    inputs: list[GlobalRateCapTensorInput],
    spec: GlobalRateCapSpec,
    *,
    tensor_offsets: dict[str, int] | None = None,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None = None,
    scope: str = GLOBAL_RATE_CAP_MARGIN_SELECTION_SCAFFOLD_SCOPE,
) -> tuple[DeviceGlobalRateCapSelectionResult, GlobalRateCapMarginSelectionFeasibilityReceipt]:
    """NON-NATIVE CPU scaffold MARGIN selection via packed-key total order.

    Returns the device selection result (same shape as the CUDA reference) and
    a feasibility-null receipt.  ``selection_parity_pass`` is PERMANENTLY False;
    no native pass is mintable on this slice.
    """

    _reject_legacy_native_env()
    spec.validate()
    if spec.normalized_ordering_mode != GlobalRateCapOrderingMode.MARGIN:
        raise NotImplementedError(
            "B2-5a scaffold selection supports MARGIN ordering only; "
            "HASH_SHUFFLE and ROUND_ROBIN remain deferred policy science"
        )
    validate_global_rate_cap_inputs(inputs)
    offsets = tensor_offsets or tensor_offsets_for_vote_update_states(inputs)
    device = _common_device(inputs)
    rows = _device_row_tensors_for_selection(
        inputs, tensor_offsets=offsets, device=device
    )
    global_indices = rows["global_indices"]
    abs_new_acc = rows["abs_new_acc"]
    row_count = int(global_indices.numel())

    empty_branch_taken = row_count == 0
    if empty_branch_taken:
        # empty_all_states / no-candidate path: early-return empty BEFORE
        # computing max_global_flat_index / max_abs_observed / index_width.
        receipt = build_global_rate_cap_margin_selection_feasibility_receipt(
            selection_parity_pass=False,
            index_width_bit_budget_pass=False,
            feasibility_null=False,
            empty_branch_taken=True,
            row_count=0,
            accepted_count=0,
            deferred_count=0,
            scaffold_parity_cases=(),
            caveats=("empty_all_states: early-return empty before bit-budget",),
        )
        validate_global_rate_cap_margin_selection_feasibility_receipt(receipt)
        selection = _empty_selection_result(
            inputs=inputs, offsets=offsets, spec=spec, device=device, scope=scope,
            deferred_backlog=deferred_backlog,
        )
        return selection, receipt

    budget = _compute_packed_key_budget(
        global_flat_indices=global_indices, abs_new_acc=abs_new_acc,
    )
    keys = _packed_total_order_key(
        abs_new_acc=abs_new_acc, global_flat_indices=global_indices, budget=budget,
    )
    order = _scaffold_sort_keys(keys)

    row_state_ids = rows["state_ids"][order]
    row_flat_indices = rows["flat_indices"][order]
    row_local_positions = rows["local_positions"][order]
    row_global_flat_indices = rows["global_indices"][order]
    row_abs_new_acc = rows["abs_new_acc"][order]
    row_thresholds = rows["thresholds"][order]
    row_directions = rows["directions"][order]

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

    accepted_count = int(accepted_positions.numel())
    deferred_count = int(deferred_positions.numel())
    stats = {
        "schema_version": GLOBAL_RATE_CAP_MARGIN_SELECTION_SCAFFOLD_SCHEMA_VERSION,
        "scope": scope,
        "backend": device.type,
        "cpu_scaffold_not_native": True,
        "standalone_b2_5a_not_full_loop_migration": True,
        "full_loop_switch_deferred_to_b2_5b": True,
        "global_rate_cap_enabled": True,
        "global_rate_cap_cap": cap,
        "global_rate_cap_ordering_mode": GlobalRateCapOrderingMode.MARGIN.value,
        "global_rate_cap_ordering_seed": int(spec.ordering_seed),
        "functional_veto_policy": DEFERRED_NON_SCOPE,
        "bad_pressure_drain_policy": DEFERRED_NON_SCOPE,
        "policy_modes_supported": [GlobalRateCapOrderingMode.MARGIN.value],
        "policy_modes_rejected": [
            GlobalRateCapOrderingMode.HASH_SHUFFLE.value,
            GlobalRateCapOrderingMode.ROUND_ROBIN.value,
        ],
        "packed_total_order_key": True,
        "lexicographic_stable_sort": False,
        "composite_key_used": True,
        "composite_key_index_width": budget.index_width,
        "composite_key_max_abs_observed": budget.max_abs_observed,
        "composite_key_max_global_flat_index": budget.max_global_flat_index,
        "sort_primitive": "cpu_scaffold_packed_key_torch_topk_extrema",
        "empty_branch_taken": False,
        "order_key": "highest_abs_new_acc_then_lower_global_flat_index",
        "device_row_tensors_emitted_before_cpu_telemetry": True,
        "cpu_telemetry_materialized": False,
        "python_row_lists_materialized_before_q_acc_apply": False,
        "global_pre_cap_would_apply_count": row_count,
        "global_rate_cap_accepted_count": accepted_count,
        "global_rate_cap_deferred_count": deferred_count,
        "global_rate_cap_saturated": row_count > cap,
        "global_rate_cap_fill_ratio": _safe_ratio(accepted_count, cap),
        "global_deferred_ratio": _safe_ratio(deferred_count, row_count),
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
    token = new_selection_token(
        selection_input_sha256=canonical_tensor_payload_sha256(abs_new_acc),
        ordered_output_sha256=canonical_tensor_payload_sha256(row_global_flat_indices),
        accepted_output_sha256=canonical_tensor_payload_sha256(
            row_global_flat_indices[accepted_positions]
        ),
        deferred_output_sha256=canonical_tensor_payload_sha256(
            row_global_flat_indices[deferred_positions]
        ),
    )
    receipt = build_global_rate_cap_margin_selection_feasibility_receipt(
        selection_parity_pass=False,
        index_width_bit_budget_pass=True,
        feasibility_null=False,
        observed_max_abs_observed=budget.max_abs_observed,
        observed_index_width=budget.index_width,
        observed_max_global_flat_index=budget.max_global_flat_index,
        empty_branch_taken=False,
        row_count=row_count,
        accepted_count=accepted_count,
        deferred_count=deferred_count,
        scaffold_parity_cases=(),
        negative_offset_reject_evidence=True,
        token=token,
        caveats=(
            "NON-NATIVE CPU scaffold: selection_parity_pass permanently False",
        ),
    )
    validate_global_rate_cap_margin_selection_feasibility_receipt(receipt)
    return selection, receipt


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
            item.plan.replay_ce_veto_indices,
            item.plan.replay_veto_directions,
            item.plan.replay_veto_thresholds,
        )
        devices.update(tensor.device for tensor in tensors)
    if len(devices) != 1:
        raise ValueError(
            f"scaffold MARGIN selection requires one shared device, got {sorted(map(str, devices))}"
        )
    return next(iter(devices))


def _empty_selection_result(
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
    rows_by_state: dict[str, DeviceGlobalRateCapStateRows] = {}
    for item in inputs:
        rows_by_state[item.state_key] = _empty_state_rows(
            item.state_key, device=device
        )
    stats = {
        "schema_version": GLOBAL_RATE_CAP_MARGIN_SELECTION_SCAFFOLD_SCHEMA_VERSION,
        "scope": scope,
        "backend": device.type,
        "cpu_scaffold_not_native": True,
        "standalone_b2_5a_not_full_loop_migration": True,
        "global_rate_cap_enabled": True,
        "global_rate_cap_cap": cap,
        "global_rate_cap_ordering_mode": GlobalRateCapOrderingMode.MARGIN.value,
        "packed_total_order_key": True,
        "empty_branch_taken": True,
        "global_pre_cap_would_apply_count": 0,
        "global_rate_cap_accepted_count": 0,
        "global_rate_cap_deferred_count": 0,
        "global_rate_cap_saturated": False,
        "global_rate_cap_fill_ratio": 0.0,
        "global_deferred_ratio": 0.0,
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
