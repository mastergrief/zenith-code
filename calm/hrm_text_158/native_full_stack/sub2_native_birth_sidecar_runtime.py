"""Repo-tracked Step-2 hybrid sidecar runtime helper.

This module owns the durable candidate-path runtime representation for the
`applied_crossing_direction_plus_4bit_residual` persistent mode. It is a pure
CPU/reference helper: persistent state shape, materialize/collapse logic,
per-step re-ledger budgeting, and the hard guard that runtime consumers must
turn into a named stop reason.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.sub2_native_birth_scaffold import (
    ACQUISITION_GATE_UNBLOCKED_NOT_RUN,
    HYBRID_PERSISTENT_MODE_APPLIED_CROSSING_DIRECTION_PLUS_4BIT_RESIDUAL,
    StrictSub2HybridRuntimeMovementOverlay,
    build_strict_sub2_hybrid_runtime_movement_overlay,
    validate_strict_sub2_hybrid_runtime_movement_overlay,
)


SUB2_HYBRID_SIDECAR_RUNTIME_SCHEMA_VERSION = (
    "hrm_text_158_sub2_hybrid_sidecar_runtime/v0"
)
PERSISTENT_SIDECAR_BUDGET_FAIL = "persistent_sidecar_budget_fail"
PERSISTENT_SIDECAR_STATE_AUTHORITY_FAIL = "persistent_sidecar_state_authority_fail"


@dataclass(frozen=True)
class AppliedCrossingDirectionResidualPersistentState:
    state_key: str
    q_levels: torch.Tensor
    frozen_scale: torch.Tensor
    applied_indices: tuple[int, ...]
    applied_directions: tuple[int, ...]
    residual_values: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.state_key:
            raise ValueError("state_key must be non-empty")
        if self.q_levels.dtype != torch.int8:
            raise ValueError(f"q_levels must be torch.int8, got {self.q_levels.dtype}")
        if self.frozen_scale.numel() != 1 or not self.frozen_scale.dtype.is_floating_point:
            raise ValueError("frozen_scale must be a floating scalar tensor")
        if not (
            len(self.applied_indices)
            == len(self.applied_directions)
            == len(self.residual_values)
        ):
            raise ValueError("sidecar indices/directions/residuals must have identical length")
        numel = int(self.q_levels.numel())
        seen = set()
        for raw_index, raw_direction, raw_residual in zip(
            self.applied_indices,
            self.applied_directions,
            self.residual_values,
        ):
            index = int(raw_index)
            direction = int(raw_direction)
            residual = int(raw_residual)
            if index < 0 or index >= numel:
                raise ValueError("sidecar index out of range")
            if index in seen:
                raise ValueError("sidecar indices must be unique")
            seen.add(index)
            if direction not in (-1, 1):
                raise ValueError("sidecar directions must be +/-1")
            if residual < -8 or residual > 7:
                raise ValueError("sidecar residual_values must fit signed 4-bit range [-8, 7]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_key": self.state_key,
            "q_levels_dtype": str(self.q_levels.dtype),
            "q_levels_shape": list(int(dim) for dim in self.q_levels.shape),
            "frozen_scale": float(self.frozen_scale.detach().cpu().item()),
            "applied_indices": [int(v) for v in self.applied_indices],
            "applied_directions": [int(v) for v in self.applied_directions],
            "residual_values": [int(v) for v in self.residual_values],
        }


@dataclass(frozen=True)
class HybridSidecarBudgetGuard:
    schema_version: str
    persistent_mode: str
    pass_guard: bool
    stop_reason: str | None
    failure_classification: str | None
    target_bits_per_weight: float
    inclusive_bits_per_weight: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "persistent_mode": self.persistent_mode,
            "pass": bool(self.pass_guard),
            "stop_reason": self.stop_reason,
            "failure_classification": self.failure_classification,
            "target_bits_per_weight": float(self.target_bits_per_weight),
            "inclusive_bits_per_weight": float(self.inclusive_bits_per_weight),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HybridSidecarPersistentStateReport:
    schema_version: str
    persistent_mode: str
    state_count: int
    total_event_count: int
    per_key_event_count: dict[str, int]
    per_key_max_abs_residual: dict[str, int]
    persistent_dense_shadow_present: bool
    persistent_dense_shadow_bytes: int
    bounded_only_collapse: bool
    movement_overlay: StrictSub2HybridRuntimeMovementOverlay
    budget_guard: HybridSidecarBudgetGuard
    pass_report: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema_version,
            "persistent_mode": self.persistent_mode,
            "state_count": int(self.state_count),
            "total_event_count": int(self.total_event_count),
            "per_key_event_count": {
                str(k): int(v) for k, v in self.per_key_event_count.items()
            },
            "per_key_max_abs_residual": {
                str(k): int(v) for k, v in self.per_key_max_abs_residual.items()
            },
            "persistent_dense_shadow_present": bool(self.persistent_dense_shadow_present),
            "persistent_dense_shadow_bytes": int(self.persistent_dense_shadow_bytes),
            "bounded_only_collapse": bool(self.bounded_only_collapse),
            "movement_overlay": self.movement_overlay.to_dict(),
            "budget_guard": self.budget_guard.to_dict(),
            "pass": bool(self.pass_report),
        }


def make_applied_crossing_direction_residual_persistent_state(
    state_key: str,
    q_levels: torch.Tensor,
    frozen_scale: torch.Tensor | float,
    *,
    applied_indices: Sequence[int],
    applied_directions: Sequence[int],
    residual_values: Sequence[int],
) -> AppliedCrossingDirectionResidualPersistentState:
    scale = (
        torch.tensor(float(frozen_scale), dtype=torch.float32)
        if not isinstance(frozen_scale, torch.Tensor)
        else frozen_scale.detach().cpu().to(torch.float32).reshape(())
    )
    q = q_levels.detach().cpu().to(torch.int8).contiguous()
    return AppliedCrossingDirectionResidualPersistentState(
        state_key=state_key,
        q_levels=q,
        frozen_scale=scale,
        applied_indices=tuple(int(v) for v in applied_indices),
        applied_directions=tuple(int(v) for v in applied_directions),
        residual_values=tuple(int(v) for v in residual_values),
    )


def materialize_transient_sidecar_shadow_state(
    state: AppliedCrossingDirectionResidualPersistentState,
) -> BoundedDeltaTensorState:
    q = state.q_levels.detach().cpu().to(torch.int8).contiguous()
    scale = state.frozen_scale.detach().cpu().to(torch.float32).reshape(())
    acc = torch.zeros_like(q, dtype=torch.int16).flatten()
    for index, direction, residual in zip(
        state.applied_indices,
        state.applied_directions,
        state.residual_values,
    ):
        value = int(residual)
        acc[int(index)] = int(direction if value == 0 else value)
    return make_bounded_tensor_state(
        str(state.state_key),
        q,
        scale,
        acc.view_as(q),
        hot_exact_indices=tuple(int(index) for index in state.applied_indices),
        cold_default_value=0,
    )


def collapse_sidecar_persistent_states(
    states: Mapping[str, BoundedDeltaTensorState],
    *,
    prior_states: Mapping[str, Any] | None = None,
) -> dict[str, AppliedCrossingDirectionResidualPersistentState]:
    out = {}
    for key, state in sorted(states.items()):
        q_after = state.q_levels.detach().cpu().to(torch.int8).flatten().contiguous()
        if prior_states is None:
            changed_indices = [int(index) for index in state.bounded_accumulator.hot_exact_indices]
            changed_directions = []
            for index in changed_indices:
                after = int(q_after[int(index)].item())
                changed_directions.append(1 if after >= 0 else -1)
        else:
            prior_q = (
                prior_states[key]
                .q_levels.detach()
                .cpu()
                .to(torch.int8)
                .flatten()
                .contiguous()
            )
            changed_indices = []
            changed_directions = []
            for flat_index, (before, after) in enumerate(zip(prior_q.tolist(), q_after.tolist())):
                delta = int(after) - int(before)
                if delta != 0:
                    changed_indices.append(int(flat_index))
                    changed_directions.append(1 if delta > 0 else -1)
        acc_after = (
            state.exact_accumulator_shadow.detach().cpu().to(torch.int16).flatten().contiguous()
        )
        residual_values = [
            int(max(-8, min(7, int(acc_after[int(index)].item()))))
            for index in changed_indices
        ]
        out[key] = make_applied_crossing_direction_residual_persistent_state(
            str(state.state_key),
            state.q_levels,
            state.frozen_scale,
            applied_indices=tuple(changed_indices),
            applied_directions=tuple(changed_directions),
            residual_values=tuple(residual_values),
        )
    return out


def _persistent_dense_shadow_present(states: Mapping[str, Any]) -> bool:
    return any(hasattr(state, "exact_accumulator_shadow") for state in states.values())


def _count_dense_shadow_bytes(states: Mapping[str, Any]) -> int:
    total = 0
    for state in states.values():
        shadow = getattr(state, "exact_accumulator_shadow", None)
        if isinstance(shadow, torch.Tensor):
            total += int(shadow.numel()) * int(shadow.element_size())
    return int(total)


def _budget_guard_from_overlay(
    overlay: StrictSub2HybridRuntimeMovementOverlay,
    *,
    persistent_dense_shadow_present: bool,
) -> HybridSidecarBudgetGuard:
    ledger = dict(overlay.persistent_sidecar_ledger)
    inclusive_bpw = float(ledger.get("inclusive_bits_per_weight", 0.0))
    target_bpw = 2.0
    if persistent_dense_shadow_present:
        return HybridSidecarBudgetGuard(
            schema_version=SUB2_HYBRID_SIDECAR_RUNTIME_SCHEMA_VERSION,
            persistent_mode=overlay.persistent_mode,
            pass_guard=False,
            stop_reason=PERSISTENT_SIDECAR_STATE_AUTHORITY_FAIL,
            failure_classification=PERSISTENT_SIDECAR_STATE_AUTHORITY_FAIL,
            target_bits_per_weight=target_bpw,
            inclusive_bits_per_weight=inclusive_bpw,
            reason="candidate persistent state still contains dense shadow authority",
        )
    if not bool(ledger.get("inclusive_lt2")):
        return HybridSidecarBudgetGuard(
            schema_version=SUB2_HYBRID_SIDECAR_RUNTIME_SCHEMA_VERSION,
            persistent_mode=overlay.persistent_mode,
            pass_guard=False,
            stop_reason=PERSISTENT_SIDECAR_BUDGET_FAIL,
            failure_classification=PERSISTENT_SIDECAR_BUDGET_FAIL,
            target_bits_per_weight=target_bpw,
            inclusive_bits_per_weight=inclusive_bpw,
            reason="q+scale+sidecar inclusive bits/weight is >= 2 on the represented event set",
        )
    return HybridSidecarBudgetGuard(
        schema_version=SUB2_HYBRID_SIDECAR_RUNTIME_SCHEMA_VERSION,
        persistent_mode=overlay.persistent_mode,
        pass_guard=True,
        stop_reason=None,
        failure_classification=None,
        target_bits_per_weight=target_bpw,
        inclusive_bits_per_weight=inclusive_bpw,
        reason="represented event set remains within the strict persistent sub-2 budget",
    )


def hybrid_sidecar_persistent_state_report(
    states: Mapping[str, AppliedCrossingDirectionResidualPersistentState],
    *,
    acquisition_science_status: str = ACQUISITION_GATE_UNBLOCKED_NOT_RUN,
    acquisition_achieved: bool = False,
) -> HybridSidecarPersistentStateReport:
    total_events = 0
    per_key_event_count = {}
    per_key_max_abs_residual = {}
    logical_shapes = []
    event_counts = []
    for key, state in sorted(states.items()):
        logical_shapes.append(tuple(int(dim) for dim in state.q_levels.shape))
        event_count = len(state.applied_indices)
        event_counts.append(int(event_count))
        total_events += int(event_count)
        per_key_event_count[key] = int(event_count)
        per_key_max_abs_residual[key] = max(
            (abs(int(value)) for value in state.residual_values),
            default=0,
        )
    persistent_dense_shadow_present = _persistent_dense_shadow_present(states)
    overlay = build_strict_sub2_hybrid_runtime_movement_overlay(
        logical_shapes=logical_shapes,
        event_counts=event_counts,
        persistent_mode=HYBRID_PERSISTENT_MODE_APPLIED_CROSSING_DIRECTION_PLUS_4BIT_RESIDUAL,
        residual_bits_per_event=4,
        persistent_dense_shadow_present=persistent_dense_shadow_present,
        persistent_dense_shadow_bytes=_count_dense_shadow_bytes(states),
        local_update_law_label=ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2,
        acquisition_science_status=acquisition_science_status,
        acquisition_achieved=acquisition_achieved,
    )
    validate_strict_sub2_hybrid_runtime_movement_overlay(overlay)
    budget_guard = _budget_guard_from_overlay(
        overlay,
        persistent_dense_shadow_present=persistent_dense_shadow_present,
    )
    report = HybridSidecarPersistentStateReport(
        schema_version=SUB2_HYBRID_SIDECAR_RUNTIME_SCHEMA_VERSION,
        persistent_mode=HYBRID_PERSISTENT_MODE_APPLIED_CROSSING_DIRECTION_PLUS_4BIT_RESIDUAL,
        state_count=len(states),
        total_event_count=int(total_events),
        per_key_event_count=per_key_event_count,
        per_key_max_abs_residual=per_key_max_abs_residual,
        persistent_dense_shadow_present=persistent_dense_shadow_present,
        persistent_dense_shadow_bytes=_count_dense_shadow_bytes(states),
        bounded_only_collapse=not persistent_dense_shadow_present,
        movement_overlay=overlay,
        budget_guard=budget_guard,
        pass_report=bool(budget_guard.pass_guard and overlay.pass_report),
    )
    return report


def consume_hybrid_sidecar_budget_guard(
    report: HybridSidecarPersistentStateReport | Mapping[str, Any],
) -> tuple[bool, str | None]:
    guard = (
        report.budget_guard.to_dict()
        if isinstance(report, HybridSidecarPersistentStateReport)
        else dict((report.get("budget_guard") or {}))
    )
    return bool(guard.get("pass")), guard.get("stop_reason")


__all__ = [
    "AppliedCrossingDirectionResidualPersistentState",
    "HybridSidecarBudgetGuard",
    "HybridSidecarPersistentStateReport",
    "PERSISTENT_SIDECAR_BUDGET_FAIL",
    "PERSISTENT_SIDECAR_STATE_AUTHORITY_FAIL",
    "SUB2_HYBRID_SIDECAR_RUNTIME_SCHEMA_VERSION",
    "collapse_sidecar_persistent_states",
    "consume_hybrid_sidecar_budget_guard",
    "hybrid_sidecar_persistent_state_report",
    "make_applied_crossing_direction_residual_persistent_state",
    "materialize_transient_sidecar_shadow_state",
]
