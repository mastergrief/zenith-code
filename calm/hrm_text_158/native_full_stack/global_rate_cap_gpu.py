"""Default-off torch-CUDA global-rate-cap reference seam.

This L2-A bridge keeps the live policy narrow: MARGIN ordering only, no trainer
integration, no full-loop migration, and no HASH_SHUFFLE/ROUND_ROBIN policy
science. It emits device row tensors first, then permits compact CPU telemetry.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    DEFERRED_NON_SCOPE,
    GlobalRateCapOrderingMode,
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    GlobalRateCapTensorResult,
    tensor_offsets_for_vote_update_states,
    validate_global_rate_cap_inputs,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    RUN_GPU_Q_ACC_APPLY_ENV,
    q_acc_apply_mutation_torch_cuda_reference_under_cap_rows,
)


RUN_GPU_GLOBAL_RATE_CAP_ENV = "HRM_TEXT_158_RUN_GPU_GLOBAL_RATE_CAP"
GLOBAL_RATE_CAP_GPU_ARTIFACT_ENV = "HRM_TEXT_158_GLOBAL_RATE_CAP_GPU_ARTIFACT"
DEFAULT_GLOBAL_RATE_CAP_GPU_ARTIFACT_PATH = (
    "artifacts/hrm_text_158_native_global_rate_cap_gpu/"
    "global_rate_cap_l2_margin_receipt_gpu.json"
)
GLOBAL_RATE_CAP_GPU_SCHEMA_VERSION = (
    "hrm_text_158_native_global_rate_cap_gpu/v0.margin_only_reference"
)
GLOBAL_RATE_CAP_TORCH_CUDA_REFERENCE_SCOPE = (
    "global_rate_cap_torch_cuda_reference_margin_only_no_policy_change"
)
QACC_KERNEL_STOP_GO_SCHEMA_VERSION = "hrm_text_158_qacc_kernel_stop_go/v0.scale_smoke"
QACC_KERNEL_PROCEED_K1K2 = "proceed_k1k2"
QACC_KERNEL_REVISE_STAGE_SHAPE = "revise_stage_shape"
QACC_KERNEL_STOP_NO_EXPECTED_SPEEDUP = "stop_no_expected_speedup"
QACC_KERNEL_STOP_PARITY_FAILURE = "stop_parity_failure"
QACC_KERNEL_MATERIAL_SHARE_THRESHOLD = 0.25
QACC_KERNEL_HOST_DOMINANT_SHARE_THRESHOLD = 0.50


@dataclass(frozen=True)
class DeviceGlobalRateCapStateRows:
    state_key: str
    accepted_indices: torch.Tensor
    accepted_directions: torch.Tensor
    accepted_thresholds: torch.Tensor
    accepted_global_flat_indices: torch.Tensor
    deferred_indices: torch.Tensor
    deferred_directions: torch.Tensor
    deferred_thresholds: torch.Tensor
    deferred_global_flat_indices: torch.Tensor


@dataclass(frozen=True)
class DeviceGlobalRateCapSelectionResult:
    scope: str
    backend: str
    state_keys: tuple[str, ...]
    tensor_offsets: dict[str, int]
    cap: int
    row_state_ids: torch.Tensor
    row_flat_indices: torch.Tensor
    row_local_positions: torch.Tensor
    row_global_flat_indices: torch.Tensor
    row_abs_new_acc: torch.Tensor
    row_thresholds: torch.Tensor
    row_directions: torch.Tensor
    accepted_positions: torch.Tensor
    deferred_positions: torch.Tensor
    rows_by_state: dict[str, DeviceGlobalRateCapStateRows]
    deferred_backlog: dict[str, dict[int, dict[str, int]]]
    stats: dict[str, Any]

    def _rows_as_tuples(self, positions: torch.Tensor) -> tuple[tuple[str, int, int, int, int, int], ...]:
        if positions.numel() == 0:
            return tuple()
        pos = positions.detach().cpu()
        state_ids = self.row_state_ids[pos].detach().cpu().tolist()
        flat_indices = self.row_flat_indices[pos].detach().cpu().tolist()
        local_positions = self.row_local_positions[pos].detach().cpu().tolist()
        global_indices = self.row_global_flat_indices[pos].detach().cpu().tolist()
        abs_new_acc = self.row_abs_new_acc[pos].detach().cpu().tolist()
        thresholds = self.row_thresholds[pos].detach().cpu().tolist()
        return tuple(
            (
                self.state_keys[int(state_id)],
                int(flat_index),
                int(local_pos),
                int(global_index),
                int(abs_acc),
                int(threshold),
            )
            for state_id, flat_index, local_pos, global_index, abs_acc, threshold in zip(
                state_ids,
                flat_indices,
                local_positions,
                global_indices,
                abs_new_acc,
                thresholds,
                strict=True,
            )
        )

    def ordered_rows_as_tuples(self) -> tuple[tuple[str, int, int, int, int, int], ...]:
        return self._rows_as_tuples(
            torch.arange(self.row_state_ids.numel(), dtype=torch.int64, device=self.row_state_ids.device)
        )

    def accepted_rows_as_tuples(self) -> tuple[tuple[str, int, int, int, int, int], ...]:
        return self._rows_as_tuples(self.accepted_positions)

    def deferred_rows_as_tuples(self) -> tuple[tuple[str, int, int, int, int, int], ...]:
        return self._rows_as_tuples(self.deferred_positions)

    def compact_artifact_payload(self) -> dict[str, Any]:
        return {
            "schema_version": GLOBAL_RATE_CAP_GPU_SCHEMA_VERSION,
            "scope": self.scope,
            "backend": self.backend,
            "state_keys": list(self.state_keys),
            "tensor_offsets": dict(self.tensor_offsets),
            "cap": int(self.cap),
            "stats": dict(self.stats),
        }


@dataclass(frozen=True)
class DeviceGlobalRateCapApplyResult:
    tensor_results: list[GlobalRateCapTensorResult]
    selection: DeviceGlobalRateCapSelectionResult
    stats: dict[str, Any]

    def compact_artifact_payload(self) -> dict[str, Any]:
        return {
            "schema_version": GLOBAL_RATE_CAP_GPU_SCHEMA_VERSION,
            "scope": GLOBAL_RATE_CAP_TORCH_CUDA_REFERENCE_SCOPE,
            "backend": self.selection.backend,
            "selection": self.selection.compact_artifact_payload(),
            "tensor_result_stats_by_state": {
                result.state_key: dict(result.stats) for result in self.tensor_results
            },
            "stats": dict(self.stats),
        }


@dataclass(frozen=True)
class QAccKernelParityReport:
    q_output_exact_match: bool
    accumulator_output_exact_match: bool
    pre_veto_selected_indices_exact_match: bool
    selected_directions_exact_match: bool
    selected_thresholds_exact_match: bool
    accepted_deferred_identity_exact_match: bool
    backlog_keys_exact_match: bool
    q_changed_count_exact_match: bool
    max_abs_diff_q: int
    max_abs_diff_acc: int

    @property
    def exact_pass(self) -> bool:
        return (
            self.q_output_exact_match
            and self.accumulator_output_exact_match
            and self.pre_veto_selected_indices_exact_match
            and self.selected_directions_exact_match
            and self.selected_thresholds_exact_match
            and self.accepted_deferred_identity_exact_match
            and self.backlog_keys_exact_match
            and self.q_changed_count_exact_match
            and int(self.max_abs_diff_q) == 0
            and int(self.max_abs_diff_acc) == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "q_output_exact_match": bool(self.q_output_exact_match),
            "accumulator_output_exact_match": bool(self.accumulator_output_exact_match),
            "pre_veto_selected_indices_exact_match": bool(self.pre_veto_selected_indices_exact_match),
            "selected_directions_exact_match": bool(self.selected_directions_exact_match),
            "selected_thresholds_exact_match": bool(self.selected_thresholds_exact_match),
            "accepted_deferred_identity_exact_match": bool(self.accepted_deferred_identity_exact_match),
            "backlog_keys_exact_match": bool(self.backlog_keys_exact_match),
            "q_changed_count_exact_match": bool(self.q_changed_count_exact_match),
            "max_abs_diff_q": int(self.max_abs_diff_q),
            "max_abs_diff_acc": int(self.max_abs_diff_acc),
            "exact_pass": bool(self.exact_pass),
        }


@dataclass(frozen=True)
class QAccKernelResidencyReport:
    cpu_selected_rows_materialized_before_q_acc_apply: bool
    python_row_lists_materialized_before_q_acc_apply: bool
    accepted_deferred_row_tensors_device_resident_until_receipt: bool
    local_preplan_backend: str
    pre_veto_selection_backend: str
    global_cap_selection_backend: str
    sparse_apply_backend: str
    host_orchestration_backend: str = "python_control"

    @property
    def hot_loop_resident(self) -> bool:
        return (
            not self.cpu_selected_rows_materialized_before_q_acc_apply
            and not self.python_row_lists_materialized_before_q_acc_apply
            and self.accepted_deferred_row_tensors_device_resident_until_receipt
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_selected_rows_materialized_before_q_acc_apply": bool(
                self.cpu_selected_rows_materialized_before_q_acc_apply
            ),
            "python_row_lists_materialized_before_q_acc_apply": bool(
                self.python_row_lists_materialized_before_q_acc_apply
            ),
            "accepted_deferred_row_tensors_device_resident_until_receipt": bool(
                self.accepted_deferred_row_tensors_device_resident_until_receipt
            ),
            "local_preplan_backend": self.local_preplan_backend,
            "pre_veto_selection_backend": self.pre_veto_selection_backend,
            "global_cap_selection_backend": self.global_cap_selection_backend,
            "sparse_apply_backend": self.sparse_apply_backend,
            "host_orchestration_backend": self.host_orchestration_backend,
            "hot_loop_resident": bool(self.hot_loop_resident),
        }


@dataclass(frozen=True)
class QAccKernelStopGoArtifact:
    schema_version: str
    representative_label: str
    tensor_shapes_by_state: dict[str, list[int]]
    total_numel: int
    candidate_count: int
    pre_veto_selected_count: int
    accepted_count: int
    deferred_count: int
    replay_veto_count: int
    phase_wall_ms: dict[str, float]
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    parity: QAccKernelParityReport
    residency: QAccKernelResidencyReport
    kernelizable_wall_ms: float
    kernelizable_share: float
    host_orchestration_share: float
    recommendation: str
    rationale: str
    material_kernelizable_share_threshold: float
    host_orchestration_dominant_share_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "representative_label": self.representative_label,
            "tensor_shapes_by_state": {
                str(key): [int(dim) for dim in value]
                for key, value in self.tensor_shapes_by_state.items()
            },
            "total_numel": int(self.total_numel),
            "candidate_count": int(self.candidate_count),
            "pre_veto_selected_count": int(self.pre_veto_selected_count),
            "accepted_count": int(self.accepted_count),
            "deferred_count": int(self.deferred_count),
            "replay_veto_count": int(self.replay_veto_count),
            "phase_wall_ms": {
                str(key): float(value) for key, value in self.phase_wall_ms.items()
            },
            "peak_allocated_bytes": int(self.peak_allocated_bytes),
            "peak_reserved_bytes": int(self.peak_reserved_bytes),
            "parity": self.parity.to_dict(),
            "residency": self.residency.to_dict(),
            "kernelizable_wall_ms": float(self.kernelizable_wall_ms),
            "kernelizable_share": float(self.kernelizable_share),
            "host_orchestration_share": float(self.host_orchestration_share),
            "recommendation": self.recommendation,
            "rationale": self.rationale,
            "material_kernelizable_share_threshold": float(
                self.material_kernelizable_share_threshold
            ),
            "host_orchestration_dominant_share_threshold": float(
                self.host_orchestration_dominant_share_threshold
            ),
        }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def build_qacc_kernel_stop_go_artifact(
    *,
    representative_label: str,
    tensor_shapes_by_state: dict[str, list[int] | tuple[int, ...]],
    candidate_count: int,
    pre_veto_selected_count: int,
    accepted_count: int,
    deferred_count: int,
    replay_veto_count: int,
    local_preplan_wall_ms: float,
    pre_veto_selection_wall_ms: float,
    global_cap_selection_wall_ms: float,
    sparse_apply_wall_ms: float,
    host_orchestration_wall_ms: float,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    parity: QAccKernelParityReport,
    residency: QAccKernelResidencyReport,
    material_kernelizable_share_threshold: float = QACC_KERNEL_MATERIAL_SHARE_THRESHOLD,
    host_orchestration_dominant_share_threshold: float = (
        QACC_KERNEL_HOST_DOMINANT_SHARE_THRESHOLD
    ),
) -> QAccKernelStopGoArtifact:
    if not representative_label:
        raise ValueError("representative_label must be non-empty")
    if not tensor_shapes_by_state:
        raise ValueError("tensor_shapes_by_state must be non-empty")
    for name, value in {
        "candidate_count": candidate_count,
        "pre_veto_selected_count": pre_veto_selected_count,
        "accepted_count": accepted_count,
        "deferred_count": deferred_count,
        "replay_veto_count": replay_veto_count,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
    }.items():
        if int(value) < 0:
            raise ValueError(f"{name} must be >= 0")
    phase_wall_ms = {
        "local_preplan": float(local_preplan_wall_ms),
        "pre_veto_selection": float(pre_veto_selection_wall_ms),
        "global_cap_selection": float(global_cap_selection_wall_ms),
        "sparse_apply": float(sparse_apply_wall_ms),
        "host_orchestration": float(host_orchestration_wall_ms),
    }
    for name, value in phase_wall_ms.items():
        if value < 0.0:
            raise ValueError(f"{name} wall time must be >= 0")
    if float(material_kernelizable_share_threshold) <= 0.0:
        raise ValueError("material_kernelizable_share_threshold must be > 0")
    if float(host_orchestration_dominant_share_threshold) <= 0.0:
        raise ValueError("host_orchestration_dominant_share_threshold must be > 0")

    normalized_shapes = {
        str(state_key): [int(dim) for dim in shape]
        for state_key, shape in tensor_shapes_by_state.items()
    }
    total_numel = 0
    for shape in normalized_shapes.values():
        numel = 1
        for dim in shape:
            if int(dim) <= 0:
                raise ValueError("tensor shapes must have positive dims")
            numel *= int(dim)
        total_numel += int(numel)

    total_wall_ms = float(sum(phase_wall_ms.values()))
    if total_wall_ms <= 0.0:
        raise ValueError("total wall time must be > 0")
    kernelizable_wall_ms = (
        float(phase_wall_ms["local_preplan"]) + float(phase_wall_ms["sparse_apply"])
    )
    kernelizable_share = float(kernelizable_wall_ms / total_wall_ms)
    host_share = float(phase_wall_ms["host_orchestration"] / total_wall_ms)
    material_subphase_ms = max(
        float(phase_wall_ms["local_preplan"]),
        float(phase_wall_ms["sparse_apply"]),
    )
    material_subphase_share = float(material_subphase_ms / total_wall_ms)

    if not parity.exact_pass:
        recommendation = QACC_KERNEL_STOP_PARITY_FAILURE
        rationale = "exact parity failed on one or more locked qacc surfaces"
    elif not residency.hot_loop_resident:
        recommendation = QACC_KERNEL_REVISE_STAGE_SHAPE
        rationale = "device residency was present but hot-loop residency was not yet honest"
    elif host_share >= float(host_orchestration_dominant_share_threshold):
        recommendation = QACC_KERNEL_REVISE_STAGE_SHAPE
        rationale = "host orchestration dominates total wall time; revise staged shape before K1/K2"
    elif material_subphase_share >= float(material_kernelizable_share_threshold):
        recommendation = QACC_KERNEL_PROCEED_K1K2
        rationale = (
            "exact parity held and a kernelizable subphase is a material share of total wall time"
        )
    else:
        recommendation = QACC_KERNEL_STOP_NO_EXPECTED_SPEEDUP
        rationale = "exact parity held but kernelizable subphases are too small to move end-to-end runtime"

    return QAccKernelStopGoArtifact(
        schema_version=QACC_KERNEL_STOP_GO_SCHEMA_VERSION,
        representative_label=representative_label,
        tensor_shapes_by_state=normalized_shapes,
        total_numel=int(total_numel),
        candidate_count=int(candidate_count),
        pre_veto_selected_count=int(pre_veto_selected_count),
        accepted_count=int(accepted_count),
        deferred_count=int(deferred_count),
        replay_veto_count=int(replay_veto_count),
        phase_wall_ms={**phase_wall_ms, "total": float(total_wall_ms)},
        peak_allocated_bytes=int(peak_allocated_bytes),
        peak_reserved_bytes=int(peak_reserved_bytes),
        parity=parity,
        residency=residency,
        kernelizable_wall_ms=float(kernelizable_wall_ms),
        kernelizable_share=float(kernelizable_share),
        host_orchestration_share=float(host_share),
        recommendation=recommendation,
        rationale=rationale,
        material_kernelizable_share_threshold=float(material_kernelizable_share_threshold),
        host_orchestration_dominant_share_threshold=float(
            host_orchestration_dominant_share_threshold
        ),
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(str(tuple(cpu.shape)).encode("utf-8"))
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def _require_gpu_global_rate_cap_enabled() -> None:
    if os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_ENV) != "1":
        raise RuntimeError(
            f"{RUN_GPU_GLOBAL_RATE_CAP_ENV}=1 is required and must only be set inside "
            "a granted gpu:0 resource lane"
        )


def _validate_margin_only_spec(spec: GlobalRateCapSpec) -> None:
    spec.validate()
    if spec.normalized_ordering_mode != GlobalRateCapOrderingMode.MARGIN:
        raise NotImplementedError(
            "GPU global-rate-cap L2-A supports MARGIN ordering only; "
            "HASH_SHUFFLE and ROUND_ROBIN remain deferred policy science"
        )


def _common_cuda_device(inputs: list[GlobalRateCapTensorInput]) -> torch.device:
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
        raise ValueError(f"global cap CUDA reference requires one shared device, got {sorted(map(str, devices))}")
    device = next(iter(devices))
    if device.type != "cuda":
        raise ValueError("global cap CUDA reference requires q/acc/plan tensors on CUDA")
    return device


def _empty_i64(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.int64, device=device)


def _empty_state_rows(state_key: str, *, device: torch.device) -> DeviceGlobalRateCapStateRows:
    return DeviceGlobalRateCapStateRows(
        state_key=state_key,
        accepted_indices=_empty_i64(device),
        accepted_directions=torch.empty(0, dtype=torch.int16, device=device),
        accepted_thresholds=torch.empty(0, dtype=torch.int32, device=device),
        accepted_global_flat_indices=_empty_i64(device),
        deferred_indices=_empty_i64(device),
        deferred_directions=torch.empty(0, dtype=torch.int16, device=device),
        deferred_thresholds=torch.empty(0, dtype=torch.int32, device=device),
        deferred_global_flat_indices=_empty_i64(device),
    )


def _stable_lexicographic_margin_order(
    *,
    abs_new_acc: torch.Tensor,
    global_flat_indices: torch.Tensor,
) -> torch.Tensor:
    if abs_new_acc.numel() == 0:
        return _empty_i64(abs_new_acc.device)
    global_order = torch.argsort(global_flat_indices, descending=False, stable=True)
    abs_order = torch.argsort(abs_new_acc[global_order], descending=True, stable=True)
    return global_order[abs_order]


def _device_row_tensors(
    inputs: list[GlobalRateCapTensorInput],
    *,
    tensor_offsets: dict[str, int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
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
            raise ValueError(f"tensor offset for {item.state_key!r} must be >= 0, got {offset}")
        flat_acc = item.plan.new_acc_i32.flatten().to(device=device, dtype=torch.int64)
        state_ids.append(torch.full((count,), state_id, dtype=torch.int64, device=device))
        flat_indices.append(indices)
        local_positions.append(torch.arange(count, dtype=torch.int64, device=device))
        global_indices.append(indices + offset)
        abs_new_acc.append(flat_acc[indices].abs().to(torch.int64))
        thresholds.append(item.plan.applied_thresholds.flatten().to(device=device, dtype=torch.int64))
        directions.append(item.plan.applied_directions.flatten().to(device=device, dtype=torch.int16))

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


def _rows_by_state_on_device(
    *,
    inputs: list[GlobalRateCapTensorInput],
    row_state_ids: torch.Tensor,
    row_flat_indices: torch.Tensor,
    row_global_flat_indices: torch.Tensor,
    row_thresholds: torch.Tensor,
    row_directions: torch.Tensor,
    accepted_positions: torch.Tensor,
    deferred_positions: torch.Tensor,
    device: torch.device,
) -> dict[str, DeviceGlobalRateCapStateRows]:
    accepted_state_ids = row_state_ids[accepted_positions]
    accepted_flat = row_flat_indices[accepted_positions]
    accepted_global = row_global_flat_indices[accepted_positions]
    accepted_thresholds = row_thresholds[accepted_positions].to(torch.int32)
    accepted_directions = row_directions[accepted_positions].to(torch.int16)
    deferred_state_ids = row_state_ids[deferred_positions]
    deferred_flat = row_flat_indices[deferred_positions]
    deferred_global = row_global_flat_indices[deferred_positions]
    deferred_thresholds = row_thresholds[deferred_positions].to(torch.int32)
    deferred_directions = row_directions[deferred_positions].to(torch.int16)

    by_state: dict[str, DeviceGlobalRateCapStateRows] = {}
    for state_id, item in enumerate(inputs):
        accepted_mask = accepted_state_ids == state_id
        deferred_mask = deferred_state_ids == state_id
        by_state[item.state_key] = DeviceGlobalRateCapStateRows(
            state_key=item.state_key,
            accepted_indices=accepted_flat[accepted_mask].to(torch.int64),
            accepted_directions=accepted_directions[accepted_mask],
            accepted_thresholds=accepted_thresholds[accepted_mask],
            accepted_global_flat_indices=accepted_global[accepted_mask].to(torch.int64),
            deferred_indices=deferred_flat[deferred_mask].to(torch.int64),
            deferred_directions=deferred_directions[deferred_mask],
            deferred_thresholds=deferred_thresholds[deferred_mask],
            deferred_global_flat_indices=deferred_global[deferred_mask].to(torch.int64),
        )
    return by_state


def _update_deferred_backlog(
    *,
    accepted_rows: tuple[tuple[str, int, int, int, int, int], ...],
    deferred_rows: tuple[tuple[str, int, int, int, int, int], ...],
    prior_backlog: dict[str, dict[int, dict[str, int]]] | None,
    step: int,
) -> tuple[dict[str, dict[int, dict[str, int]]], int, dict[str, int]]:
    backlog = copy.deepcopy(prior_backlog or {})
    accepted_from_prior_deferred = 0
    for state_key, flat_index, *_ in accepted_rows:
        state_backlog = backlog.get(state_key, {})
        if flat_index in state_backlog:
            accepted_from_prior_deferred += 1
            del state_backlog[flat_index]
    for state_key, flat_index, *_ in deferred_rows:
        state_backlog = backlog.setdefault(state_key, {})
        entry = state_backlog.setdefault(
            flat_index,
            {"first_step": int(step), "last_deferred_step": int(step), "defer_count": 0},
        )
        entry["last_deferred_step"] = int(step)
        entry["defer_count"] = int(entry.get("defer_count", 0)) + 1

    entries = [entry for by_index in backlog.values() for entry in by_index.values()]
    if not entries:
        age_summary = {
            "deferred_backlog_size": 0,
            "deferred_backlog_max_age_steps": 0,
            "deferred_backlog_max_defer_count": 0,
        }
    else:
        age_summary = {
            "deferred_backlog_size": len(entries),
            "deferred_backlog_max_age_steps": max(
                int(step) - int(entry["first_step"]) for entry in entries
            ),
            "deferred_backlog_max_defer_count": max(
                int(entry["defer_count"]) for entry in entries
            ),
        }
    return backlog, accepted_from_prior_deferred, age_summary


def _selection_with_cpu_telemetry(
    selection: DeviceGlobalRateCapSelectionResult,
    spec: GlobalRateCapSpec,
    *,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None,
    cpu_telemetry_timing: str,
) -> DeviceGlobalRateCapSelectionResult:
    accepted_rows = selection.accepted_rows_as_tuples()
    deferred_rows = selection.deferred_rows_as_tuples()
    backlog, accepted_from_prior_deferred, age_summary = _update_deferred_backlog(
        accepted_rows=accepted_rows,
        deferred_rows=deferred_rows,
        prior_backlog=deferred_backlog,
        step=spec.step,
    )
    accepted_count = len(accepted_rows)
    deferred_count = len(deferred_rows)
    row_count = int(selection.row_state_ids.numel())
    stats = {
        **selection.stats,
        "cpu_telemetry_materialized": True,
        "cpu_telemetry_timing": cpu_telemetry_timing,
        "accepted_from_prior_deferred_count": accepted_from_prior_deferred,
        "accepted_fresh_count": accepted_count - accepted_from_prior_deferred,
        "global_rate_cap_rows_global_indices_sha256": _tensor_sha256(
            selection.row_global_flat_indices
        ),
        "global_rate_cap_accepted_global_indices_sha256": _tensor_sha256(
            selection.row_global_flat_indices[selection.accepted_positions]
        ),
        "global_rate_cap_deferred_global_indices_sha256": _tensor_sha256(
            selection.row_global_flat_indices[selection.deferred_positions]
        ),
        "global_rate_cap_fill_ratio": _safe_ratio(accepted_count, int(selection.cap)),
        "global_deferred_ratio": _safe_ratio(deferred_count, row_count),
        **age_summary,
    }
    return DeviceGlobalRateCapSelectionResult(
        scope=selection.scope,
        backend=selection.backend,
        state_keys=selection.state_keys,
        tensor_offsets=dict(selection.tensor_offsets),
        cap=selection.cap,
        row_state_ids=selection.row_state_ids,
        row_flat_indices=selection.row_flat_indices,
        row_local_positions=selection.row_local_positions,
        row_global_flat_indices=selection.row_global_flat_indices,
        row_abs_new_acc=selection.row_abs_new_acc,
        row_thresholds=selection.row_thresholds,
        row_directions=selection.row_directions,
        accepted_positions=selection.accepted_positions,
        deferred_positions=selection.deferred_positions,
        rows_by_state=selection.rows_by_state,
        deferred_backlog=backlog,
        stats=stats,
    )


def select_global_rate_cap_rows_torch_cuda_reference(
    inputs: list[GlobalRateCapTensorInput],
    spec: GlobalRateCapSpec,
    *,
    tensor_offsets: dict[str, int] | None = None,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None = None,
    materialize_cpu_telemetry: bool = True,
    scope: str = GLOBAL_RATE_CAP_TORCH_CUDA_REFERENCE_SCOPE,
) -> DeviceGlobalRateCapSelectionResult:
    """Select MARGIN global-cap rows on CUDA and emit device row tensors."""

    _validate_margin_only_spec(spec)
    _require_gpu_global_rate_cap_enabled()
    validate_global_rate_cap_inputs(inputs)
    device = _common_cuda_device(inputs)
    offsets = tensor_offsets or tensor_offsets_for_vote_update_states(inputs)
    rows = _device_row_tensors(inputs, tensor_offsets=offsets, device=device)
    order = _stable_lexicographic_margin_order(
        abs_new_acc=rows["abs_new_acc"],
        global_flat_indices=rows["global_indices"],
    )
    row_state_ids = rows["state_ids"][order]
    row_flat_indices = rows["flat_indices"][order]
    row_local_positions = rows["local_positions"][order]
    row_global_flat_indices = rows["global_indices"][order]
    row_abs_new_acc = rows["abs_new_acc"][order]
    row_thresholds = rows["thresholds"][order]
    row_directions = rows["directions"][order]

    row_count = int(row_state_ids.numel())
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
        "schema_version": GLOBAL_RATE_CAP_GPU_SCHEMA_VERSION,
        "scope": scope,
        "backend": device.type,
        "global_cap_gpu_native": True,
        "standalone_l2a_not_full_loop_migration": True,
        "full_loop_switch_deferred_to_l2b": True,
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
        "lexicographic_stable_sort": True,
        "order_key": "highest_abs_new_acc_then_lower_global_flat_index",
        "composite_key_used": False,
        "composite_key_safe_range": "not_applicable_explicit_stable_lexicographic_sort",
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
        deferred_backlog=copy.deepcopy(deferred_backlog or {}),
        stats=stats,
    )
    if not materialize_cpu_telemetry:
        return selection
    return _selection_with_cpu_telemetry(
        selection,
        spec,
        deferred_backlog=deferred_backlog,
        cpu_telemetry_timing="after_device_row_emission_selection_parity",
    )


def apply_global_rate_cap_torch_cuda_reference_under_margin(
    inputs: list[GlobalRateCapTensorInput],
    spec: GlobalRateCapSpec,
    *,
    tensor_offsets: dict[str, int] | None = None,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None = None,
) -> DeviceGlobalRateCapApplyResult:
    """Feed device-selected cap rows directly into the q/acc apply CUDA seam."""

    selection = select_global_rate_cap_rows_torch_cuda_reference(
        inputs,
        spec,
        tensor_offsets=tensor_offsets,
        deferred_backlog=deferred_backlog,
        materialize_cpu_telemetry=False,
    )
    tensor_results: list[GlobalRateCapTensorResult] = []
    total_q_changed = 0
    for item in inputs:
        state_rows = selection.rows_by_state[item.state_key]
        apply_result = q_acc_apply_mutation_torch_cuda_reference_under_cap_rows(
            q_levels=item.state.q_levels,
            new_accumulators=item.plan.new_acc_i32,
            accepted_indices=state_rows.accepted_indices,
            accepted_directions=state_rows.accepted_directions,
            accepted_thresholds=state_rows.accepted_thresholds,
            replay_veto_indices=item.plan.replay_ce_veto_indices,
            replay_veto_directions=item.plan.replay_veto_directions,
            replay_veto_thresholds=item.plan.replay_veto_thresholds,
            mutate_outputs=spec.mutate_outputs,
            original_accumulators=item.state.accumulators,
            scope=GLOBAL_RATE_CAP_TORCH_CUDA_REFERENCE_SCOPE,
        )
        q_changed = int(apply_result.stats["q_changed_count"])
        total_q_changed += q_changed
        stats = dict(item.plan.stats)
        stats.update(
            {
                "schema_version": GLOBAL_RATE_CAP_GPU_SCHEMA_VERSION,
                "scope": GLOBAL_RATE_CAP_TORCH_CUDA_REFERENCE_SCOPE,
                "backend": apply_result.backend,
                "global_cap_gpu_native": True,
                "global_rate_cap_cap": int(spec.cap),
                "global_rate_cap_ordering_mode": GlobalRateCapOrderingMode.MARGIN.value,
                "global_rate_cap_accepted_count": int(state_rows.accepted_indices.numel()),
                "global_rate_cap_deferred_count": int(state_rows.deferred_indices.numel()),
                "global_rate_cap_accepted_global_indices_sha256": _tensor_sha256(
                    state_rows.accepted_global_flat_indices
                ),
                "global_rate_cap_deferred_global_indices_sha256": _tensor_sha256(
                    state_rows.deferred_global_flat_indices
                ),
                "accepted_row_source": "device_selection_result.rows_by_state",
                "python_row_lists_materialized_before_q_acc_apply": False,
                "q_acc_apply_env_required": RUN_GPU_Q_ACC_APPLY_ENV,
                "ternary_mutation_enabled": bool(spec.mutate_outputs),
                "ternary_mutation_frozen": not bool(spec.mutate_outputs),
                "q_changed_count": q_changed,
            }
        )
        tensor_results.append(
            GlobalRateCapTensorResult(
                state_key=item.state_key,
                q_levels=apply_result.q_levels,
                accumulators=apply_result.accumulators,
                stats=stats,
            )
        )
    selection_with_telemetry = _selection_with_cpu_telemetry(
        selection,
        spec,
        deferred_backlog=deferred_backlog,
        cpu_telemetry_timing="after_q_acc_apply_consumed_device_rows",
    )
    apply_stats = {
        **selection_with_telemetry.stats,
        "q_changed_count": total_q_changed,
        "accepted_row_source": "device_selection_result.rows_by_state",
        "python_row_lists_materialized_before_q_acc_apply": False,
        "q_acc_apply_env_required": RUN_GPU_Q_ACC_APPLY_ENV,
    }
    return DeviceGlobalRateCapApplyResult(
        tensor_results=tensor_results,
        selection=selection_with_telemetry,
        stats=apply_stats,
    )


def write_global_rate_cap_gpu_receipt_artifact(
    payload: dict[str, Any],
    path: str | Path | None = None,
) -> Path:
    artifact_path = Path(
        path
        or os.environ.get(GLOBAL_RATE_CAP_GPU_ARTIFACT_ENV)
        or DEFAULT_GLOBAL_RATE_CAP_GPU_ARTIFACT_PATH
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact_path
