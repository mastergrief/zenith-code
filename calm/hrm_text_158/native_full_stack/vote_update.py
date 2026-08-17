"""Repo-native integer q/vote/update bridge for Phase-1 Slice 2A.

Scope is intentionally local/per-tensor: exact int16 update-law contract and
reference, plus a default-off stageable preplan kernel interface. This is not
rank-bucket credit generation, global multi-tensor cap, acquisition proof, or a
GPU-done kernel.
"""
from __future__ import annotations

from dataclasses import InitVar, dataclass
from enum import Enum
import hashlib
import os
from typing import Any, Optional

import torch

from calm.hrm_text_158.native_full_stack.two_tier_step_orchestrator import (
    apply_two_tier_write_backs,
    build_two_tier_step_plan_for_apply,
    plan_two_tier_step,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS,
    assert_two_tier_threshold_receipt_consistent,
)
from calm.hrm_text_158.native_full_stack.two_tier_transient_selection import (
    LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
    carry_after_i32_tensor,
    crossing_eligible_flat_indices,
    crossing_eligible_mask_from_tensors,
    select_flat_indices_by_local_loss_delta_tensor,
    validate_two_tier_selector_inputs,
)

try:  # Keep CPU/static imports working when Triton is unavailable.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - non-Triton hosts.
    triton = None
    tl = None


RUN_GPU_VOTE_UPDATE_ENV = "HRM_TEXT_158_RUN_GPU_VOTE_UPDATE"
RUN_GPU_Q_ACC_APPLY_ENV = "HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY"
Q_ACC_APPLY_MUTATION_TORCH_CUDA_REFERENCE_SCOPE = (
    "q_acc_apply_mutation_torch_cuda_reference_under_cap_rows_only"
)

INT8_Q_TRANSITIONAL_NOTE = (
    "int8_levels q state is transitional and pack-ready; it is not packed "
    "sub-2-bit ternary storage."
)
INT16_ACC_TRANSITIONAL_NOTE = (
    "int16_accumulators are transitional learner-state accumulators; compressed "
    "vote/accumulator storage is not implemented in Slice 2A."
)
DEFERRED_GLOBAL_CAP = "deferred_global_cap"
LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX = "current_abs_new_acc_then_index"
LOCAL_SELECTION_ORDER_DETERMINISTIC_HASH_MATCHED = "deterministic_hash_shuffle_order_matched"


class VoteUpdateQFormat(str, Enum):
    INT8_LEVELS = "int8_levels"
    PACKED_TERNARY = "packed_ternary"


class VoteUpdateAccumulatorFormat(str, Enum):
    INT16_ACCUMULATORS = "int16_accumulators"
    COMPRESSED_ACCUMULATORS = "compressed_accumulators"
    EVENT_CODED_LIVE_CARRIER = "event_coded_live_carrier"


class VoteUpdateVoteFormat(str, Enum):
    INT16_VOTES = "int16_votes"
    COMPRESSED_VOTES = "compressed_votes"


class PcAuxMode(str, Enum):
    TELEMETRY = "telemetry"
    VETO = "veto"


class ReplayCeMode(str, Enum):
    OFF = "off"
    TELEMETRY = "telemetry"
    VETO = "veto"


@dataclass(frozen=True)
class VoteUpdateSpec:
    """Per-tensor local integer update law.

    `global_cap_policy` is explicitly deferred because S1's active global cap is
    a cross-tensor Slice 2B seam, not a hidden part of this local reference.
    """

    threshold_abs: int
    accumulator_clip_min: int
    accumulator_clip_max: int
    decay_numerator: int = 1
    decay_denominator: int = 1
    max_abs_per_tensor: int = 2**31 - 1
    fraction_per_tensor: float = 1.0
    threshold_jitter_enabled: bool = False
    global_cap_policy: str = DEFERRED_GLOBAL_CAP

    @classmethod
    def from_live_spec(cls, spec: dict) -> "VoteUpdateSpec":
        acc_spec = spec["accumulator"]
        flip_budget = spec["per_step_flip_budget"]
        return cls(
            threshold_abs=int(spec["threshold_abs"]),
            accumulator_clip_min=int(acc_spec["clip_min"]),
            accumulator_clip_max=int(acc_spec["clip_max"]),
            decay_numerator=int(spec["decay"]["numerator"]),
            decay_denominator=int(spec["decay"]["denominator"]),
            max_abs_per_tensor=int(flip_budget["max_abs_per_tensor"]),
            fraction_per_tensor=float(flip_budget["fraction_per_tensor"]),
        )

    def validate(self) -> None:
        if self.threshold_jitter_enabled:
            raise NotImplementedError(
                "threshold_jitter_enabled=True is deferred in Slice 2A; "
                "implement exact jitter or keep this explicit rejection"
            )
        if self.global_cap_policy != DEFERRED_GLOBAL_CAP:
            raise NotImplementedError(
                "global multi-tensor cap is Slice 2B; set "
                "global_cap_policy='deferred_global_cap' for this local law"
            )
        if self.threshold_abs <= 0:
            raise ValueError(f"threshold_abs must be > 0, got {self.threshold_abs}")
        if self.accumulator_clip_min > self.accumulator_clip_max:
            raise ValueError("accumulator clip_min must be <= clip_max")
        if self.decay_denominator <= 0:
            raise ValueError("decay_denominator must be > 0")
        if self.decay_numerator < 0:
            raise ValueError("decay_numerator must be >= 0")
        if self.max_abs_per_tensor < 0:
            raise ValueError("max_abs_per_tensor must be >= 0")
        if self.fraction_per_tensor < 0.0:
            raise ValueError("fraction_per_tensor must be >= 0")

    def max_flips(self, numel: int) -> int:
        import math

        self.validate()
        return min(
            int(self.max_abs_per_tensor),
            math.ceil(float(self.fraction_per_tensor) * int(numel)),
        )


@dataclass(frozen=True)
class VoteUpdateState:
    q_levels: torch.Tensor
    accumulators: torch.Tensor
    q_format: VoteUpdateQFormat | str = VoteUpdateQFormat.INT8_LEVELS
    accumulator_format: VoteUpdateAccumulatorFormat | str = (
        VoteUpdateAccumulatorFormat.INT16_ACCUMULATORS
    )

    @property
    def normalized_q_format(self) -> VoteUpdateQFormat:
        return VoteUpdateQFormat(self.q_format)

    @property
    def normalized_accumulator_format(self) -> VoteUpdateAccumulatorFormat:
        return VoteUpdateAccumulatorFormat(self.accumulator_format)


@dataclass(frozen=True)
class VoteUpdateInputs:
    votes: torch.Tensor
    replay_ce_veto_votes: Optional[torch.Tensor] = None
    replay_ce_veto_moves: Optional[torch.Tensor] = None
    pc_aux_votes: Optional[torch.Tensor] = None
    pc_aux_moves: Optional[torch.Tensor] = None
    pc_aux_mode: PcAuxMode | str = PcAuxMode.TELEMETRY
    replay_ce_mode: InitVar[ReplayCeMode | str] = ReplayCeMode.VETO
    vote_format: VoteUpdateVoteFormat | str = VoteUpdateVoteFormat.INT16_VOTES
    local_loss_delta: Optional[torch.Tensor] = None
    vote_active_flat_indices: Optional[torch.Tensor] = None
    sparse_vote_events: Any = None

    def __post_init__(self, replay_ce_mode: ReplayCeMode | str) -> None:
        object.__setattr__(self, "_replay_ce_mode", ReplayCeMode(replay_ce_mode))

    @property
    def normalized_vote_format(self) -> VoteUpdateVoteFormat:
        return VoteUpdateVoteFormat(self.vote_format)

    @property
    def normalized_pc_aux_mode(self) -> PcAuxMode:
        return PcAuxMode(self.pc_aux_mode)

    @property
    def normalized_replay_ce_mode(self) -> ReplayCeMode:
        return self._replay_ce_mode


@dataclass(frozen=True)
class VoteUpdatePlan:
    q_i16: torch.Tensor
    new_acc_i32: torch.Tensor
    candidate_indices: torch.Tensor
    pre_veto_selected_indices: torch.Tensor
    applied_indices: torch.Tensor
    applied_directions: torch.Tensor
    applied_thresholds: torch.Tensor
    replay_ce_veto_indices: torch.Tensor
    replay_veto_directions: torch.Tensor
    replay_veto_thresholds: torch.Tensor
    pc_aux_negative_indices: torch.Tensor
    pc_aux_veto_indices: torch.Tensor
    stats: dict[str, int | float | bool | str]
    event_coded_sparse_active_idx: torch.Tensor | None = None
    event_coded_sparse_post_active_i32: torch.Tensor | None = None


@dataclass(frozen=True)
class VoteUpdateResult:
    q_levels: torch.Tensor
    accumulators: torch.Tensor
    plan: VoteUpdatePlan
    stats: dict[str, int | float | bool | str]


@dataclass(frozen=True)
class QAccApplyMutationResult:
    q_levels: torch.Tensor
    accumulators: torch.Tensor
    scope: str
    backend: str
    stats: dict[str, int | bool | str]


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _replay_ce_telemetry_stat_leaves(
    inputs: VoteUpdateInputs,
    replay_ce_negative: torch.Tensor,
) -> dict[str, int | float | bool | str]:
    if inputs.normalized_replay_ce_mode != ReplayCeMode.TELEMETRY:
        return {}
    return {
        "replay_ce_mode": ReplayCeMode.TELEMETRY.value,
        "replay_ce_negative_count": int(replay_ce_negative.numel()),
    }


def _validate_future_formats(state: VoteUpdateState, inputs: VoteUpdateInputs) -> None:
    if state.normalized_q_format != VoteUpdateQFormat.INT8_LEVELS:
        raise NotImplementedError("future packed q formats are named but not implemented in Slice 2A")
    if state.normalized_accumulator_format != VoteUpdateAccumulatorFormat.INT16_ACCUMULATORS:
        if state.normalized_accumulator_format == VoteUpdateAccumulatorFormat.EVENT_CODED_LIVE_CARRIER:
            raise ValueError(
                "event-coded live carrier must use event_coded_vote_update_adapter plan/apply path"
            )
        raise NotImplementedError("compressed accumulator formats are named but not implemented in Slice 2A")
    if inputs.normalized_vote_format != VoteUpdateVoteFormat.INT16_VOTES:
        raise NotImplementedError("compressed vote formats are named but not implemented in Slice 2A")
    inputs.normalized_pc_aux_mode


def _local_selection_order(
    *,
    candidate_idx: torch.Tensor,
    new_acc_i32: torch.Tensor,
    numel: int,
    mode: str,
    ordering_seed: int,
    ordering_step: int,
) -> torch.Tensor:
    normalized_mode = str(mode)
    if normalized_mode == LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX:
        abs_score = new_acc_i32[candidate_idx].abs().to(torch.int64)
        idx64 = candidate_idx.to(torch.int64)
        composite = abs_score * (int(numel) + 1) + (int(numel) - idx64)
        return torch.argsort(composite, descending=True)
    if normalized_mode == LOCAL_SELECTION_ORDER_DETERMINISTIC_HASH_MATCHED:
        keyed: list[tuple[bytes, int]] = []
        for position, flat_index in enumerate(candidate_idx.detach().cpu().to(torch.int64).tolist()):
            digest = hashlib.sha256(
                (
                    f"hrm_text_158_local_selection_order|seed={int(ordering_seed)}|"
                    f"step={int(ordering_step)}|index={int(flat_index)}"
                ).encode("utf-8")
            ).digest()
            keyed.append((digest, int(position)))
        keyed.sort()
        return torch.tensor(
            [position for _digest, position in keyed],
            dtype=torch.int64,
            device=candidate_idx.device,
        )
    raise ValueError(f"unsupported local_selection_ordering_mode {mode!r}")


def _applied_crossing_direction_from_carry(carry_after: int, *, threshold_abs: int) -> int:
    if int(carry_after) >= int(threshold_abs):
        return 1
    if int(carry_after) <= -int(threshold_abs):
        return -1
    raise ValueError(
        "applied write-back requires crossing carry_after at threshold_abs="
        f"{threshold_abs}, got {carry_after}"
    )


def _partition_pre_veto_by_replay_and_pc_veto(
    pre_veto_selected: torch.Tensor,
    directions: torch.Tensor,
    selected_thresholds: torch.Tensor,
    inputs: VoteUpdateInputs,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    replay_veto_mask = torch.zeros_like(directions, dtype=torch.bool)
    replay_ce_vetoed = pre_veto_selected[:0]
    replay_ce_negative = pre_veto_selected[:0]
    replay_veto_directions = torch.zeros_like(pre_veto_selected[:0], dtype=torch.int16)
    replay_veto_thresholds = torch.zeros_like(pre_veto_selected[:0], dtype=torch.int32)
    if inputs.replay_ce_veto_votes is not None:
        replay_vote = inputs.replay_ce_veto_votes.flatten().to(torch.int32)
        replay_move = inputs.replay_ce_veto_moves.flatten().to(torch.int32)
        replay_direction = torch.sign(replay_vote[pre_veto_selected])
        replay_direction = torch.where(
            replay_direction != 0,
            replay_direction,
            torch.sign(replay_move[pre_veto_selected]),
        ).to(torch.int16)
        replay_support = replay_direction * directions
        replay_veto_mask = replay_support < 0
        if inputs.normalized_replay_ce_mode == ReplayCeMode.TELEMETRY:
            replay_ce_negative = pre_veto_selected[replay_veto_mask]
        elif inputs.normalized_replay_ce_mode == ReplayCeMode.VETO:
            replay_ce_vetoed = pre_veto_selected[replay_veto_mask]
            replay_veto_directions = directions[replay_veto_mask]
            replay_veto_thresholds = selected_thresholds[replay_veto_mask]
    pc_veto_mask = torch.zeros_like(directions, dtype=torch.bool)
    pc_aux_negative = pre_veto_selected[:0]
    pc_aux_vetoed = pre_veto_selected[:0]
    if inputs.pc_aux_votes is not None:
        pc_vote = inputs.pc_aux_votes.flatten().to(torch.int32)
        pc_move = inputs.pc_aux_moves.flatten().to(torch.int32)
        pc_direction = torch.sign(pc_vote[pre_veto_selected])
        pc_direction = torch.where(
            pc_direction != 0,
            pc_direction,
            torch.sign(pc_move[pre_veto_selected]),
        ).to(torch.int16)
        pc_support = pc_direction * directions
        pc_veto_mask = pc_support < 0
        pc_aux_negative = pre_veto_selected[pc_veto_mask]
        if inputs.normalized_pc_aux_mode == PcAuxMode.VETO:
            # Replay remains the first veto layer; PC-veto only accounts for
            # additional flips that survived replay.
            pc_aux_vetoed = pre_veto_selected[pc_veto_mask & ~replay_veto_mask]
    if inputs.normalized_replay_ce_mode == ReplayCeMode.VETO:
        apply_mask = ~replay_veto_mask
    else:
        apply_mask = torch.ones_like(replay_veto_mask)
    if inputs.normalized_pc_aux_mode == PcAuxMode.VETO:
        apply_mask = apply_mask & ~pc_veto_mask
    applied = pre_veto_selected[apply_mask]
    applied_directions = directions[apply_mask]
    applied_thresholds = selected_thresholds[apply_mask]
    return (
        applied,
        applied_directions,
        applied_thresholds,
        replay_ce_vetoed,
        replay_ce_negative,
        replay_veto_directions,
        replay_veto_thresholds,
        pc_aux_negative,
        pc_aux_vetoed,
    )


def _apply_replay_veto_residual_clamp(
    new_acc_i32: torch.Tensor,
    *,
    replay_ce_veto_indices: torch.Tensor,
    replay_veto_directions: torch.Tensor,
    replay_veto_thresholds: torch.Tensor,
) -> None:
    if replay_ce_veto_indices.numel() == 0:
        return
    vetoed = replay_ce_veto_indices.to(new_acc_i32.device)
    directions = replay_veto_directions.to(new_acc_i32.device)
    thresholds = replay_veto_thresholds.to(new_acc_i32.device)
    residual = new_acc_i32[vetoed] - directions.to(torch.int32) * thresholds
    low = -thresholds + 1
    high = thresholds - 1
    new_acc_i32[vetoed] = torch.minimum(torch.maximum(residual, low), high)


def _reject_event_coded_on_dense_vote_path(state: VoteUpdateState, *, site: str) -> None:
    if state.normalized_accumulator_format == VoteUpdateAccumulatorFormat.EVENT_CODED_LIVE_CARRIER:
        raise ValueError(
            f"{site} forbidden on event-coded live carrier path; "
            "use event_coded_vote_update_adapter plan/apply"
        )


def _materialize_two_tier_rows(
    state: VoteUpdateState,
    inputs: VoteUpdateInputs,
) -> list[dict[str, Any]]:
    if state.normalized_accumulator_format == VoteUpdateAccumulatorFormat.EVENT_CODED_LIVE_CARRIER:
        raise ValueError(
            "full-numel two_tier row materialization forbidden on event-coded live carrier path"
        )
    q_levels = state.q_levels.flatten()
    accumulators = state.accumulators.flatten()
    votes = inputs.votes.flatten()
    deltas = inputs.local_loss_delta.flatten()
    rows: list[dict[str, Any]] = []
    for flat_index in range(int(q_levels.numel())):
        rows.append(
            {
                "candidate_id": str(flat_index),
                "flat_index": int(flat_index),
                "vote_value": int(votes[flat_index].item()),
                "pre_accumulator_i16": int(accumulators[flat_index].item()),
                "current_q_level": int(q_levels[flat_index].item()),
                "local_loss_delta": float(deltas[flat_index].item()),
                "in_target_tie_band": True,
            }
        )
    return rows


def _minimal_two_tier_rows_for_flat_indices(
    flat_indices: torch.Tensor,
    *,
    q_levels: torch.Tensor,
    accumulators: torch.Tensor,
    votes: torch.Tensor,
    local_loss_delta: torch.Tensor,
) -> list[dict[str, Any]]:
    q_flat = q_levels.flatten()
    acc_flat = accumulators.flatten()
    vote_flat = votes.flatten()
    delta_flat = local_loss_delta.flatten()
    rows: list[dict[str, Any]] = []
    for flat_index in flat_indices.detach().cpu().tolist():
        idx = int(flat_index)
        rows.append(
            {
                "candidate_id": str(idx),
                "flat_index": idx,
                "vote_value": int(vote_flat[idx].item()),
                "pre_accumulator_i16": int(acc_flat[idx].item()),
                "current_q_level": int(q_flat[idx].item()),
                "local_loss_delta": float(delta_flat[idx].item()),
                "in_target_tie_band": True,
            }
        )
    return rows


def _two_tier_vote_update_stats(
    *,
    votes: torch.Tensor,
    new_acc_i32: torch.Tensor,
    candidate_idx: torch.Tensor,
    pre_veto_selected: torch.Tensor,
    applied: torch.Tensor,
    replay_ce_vetoed: torch.Tensor,
    replay_ce_negative: torch.Tensor,
    pc_aux_negative: torch.Tensor,
    pc_aux_vetoed: torch.Tensor,
    max_flips: int,
    vote_spec_threshold_abs: int,
    crossing_threshold_abs: int,
    local_selection_ordering_mode: str,
    local_selection_ordering_seed: int,
    local_selection_ordering_step: int,
    inputs: VoteUpdateInputs,
) -> dict[str, int | float | bool | str]:
    pre_veto_count = int(pre_veto_selected.numel())
    applied_count = int(applied.numel())
    replay_count = int(replay_ce_vetoed.numel())
    pc_negative_count = int(pc_aux_negative.numel())
    pc_veto_count = int(pc_aux_vetoed.numel())
    stats: dict[str, int | float | bool | str] = {
        "scope": "per_tensor_local_update",
        "global_cap_policy": DEFERRED_GLOBAL_CAP,
        "local_selection_ordering_mode": str(local_selection_ordering_mode),
        "local_selection_ordering_seed": int(local_selection_ordering_seed),
        "local_selection_ordering_step": int(local_selection_ordering_step),
        "threshold_jitter_policy": "deferred_reject",
        "candidate_count": int(candidate_idx.numel()),
        "max_flips": int(max_flips),
        "pre_veto_selected_flip_count": pre_veto_count,
        "post_veto_would_apply_pre_cap_count": applied_count,
        "post_veto_acceptance_ratio_pre_cap": _safe_ratio(applied_count, pre_veto_count),
        "replay_ce_veto_count": replay_count,
        "pc_aux_negative_count": pc_negative_count,
        "pc_aux_mode": inputs.normalized_pc_aux_mode.value,
        "pc_aux_veto_enabled": inputs.normalized_pc_aux_mode == PcAuxMode.VETO,
        "pc_aux_veto_count": pc_veto_count,
        "pc_aux_veto_accumulator_residual_policy": (
            "q_mutation_veto_only_accumulator_retained"
            if inputs.normalized_pc_aux_mode == PcAuxMode.VETO and inputs.pc_aux_votes is not None
            else "not_enabled"
        ),
        "replay_ce_veto_consumes_threshold_event": (
            inputs.normalized_replay_ce_mode == ReplayCeMode.VETO
            and inputs.replay_ce_veto_votes is not None
        ),
        "vetoed_accumulator_residual_policy": (
            "subtract_threshold_then_clamp_without_q_mutation"
            if (
                inputs.normalized_replay_ce_mode == ReplayCeMode.VETO
                and inputs.replay_ce_veto_votes is not None
            )
            else "not_enabled"
        ),
        "vetoed_accumulator_clamp_count": replay_count,
        "vote_nonzero_count": int((votes != 0).sum().item()),
        "acc_abs_max_after_decay_vote": int(new_acc_i32.abs().max().item()) if new_acc_i32.numel() else 0,
        "two_tier_carry_w6_enabled": True,
        "two_tier_threshold_abs": int(crossing_threshold_abs),
        "two_tier_canonical_threshold_abs": int(CROSSING_THRESHOLD_ABS),
        "two_tier_vote_spec_threshold_abs": int(vote_spec_threshold_abs),
    }
    stats.update(_replay_ce_telemetry_stat_leaves(inputs, replay_ce_negative))
    assert_two_tier_threshold_receipt_consistent(stats)
    return stats


def plan_two_tier_vote_update_reference_legacy(
    state: VoteUpdateState,
    inputs: VoteUpdateInputs,
    spec: VoteUpdateSpec,
    *,
    validate_q_levels: bool = True,
    local_selection_ordering_mode: str = LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
    local_selection_ordering_seed: int = 0,
    local_selection_ordering_step: int = 0,
    warmup: bool = False,
) -> VoteUpdatePlan:
    validate_vote_update_contract(state, inputs, spec, validate_q_levels=validate_q_levels)
    if str(local_selection_ordering_mode) != LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA:
        raise ValueError(
            "two_tier_carry_w6_enabled requires "
            f"local_selection_ordering_mode={LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA!r}, "
            f"got {local_selection_ordering_mode!r}"
        )
    failures = validate_two_tier_selector_inputs(inputs, enabled=True)
    if failures:
        raise ValueError("selector input validation failed: " + ",".join(failures))

    q_levels = state.q_levels
    accumulators = state.accumulators
    votes = inputs.votes
    vote_spec_threshold_abs = int(spec.threshold_abs)
    crossing_threshold_abs = int(CROSSING_THRESHOLD_ABS)
    numel = int(q_levels.numel())
    max_flips = spec.max_flips(numel)
    rows = _materialize_two_tier_rows(state, inputs)
    carry_by_flat_index = {
        int(row["flat_index"]): int(row["pre_accumulator_i16"]) for row in rows
    }
    q_level_by_flat_index = {
        int(row["flat_index"]): int(row["current_q_level"]) for row in rows
    }
    two_tier_plan = plan_two_tier_step(
        rows,
        carry_by_flat_index=carry_by_flat_index,
        q_level_by_flat_index=q_level_by_flat_index,
        rate_cap=int(max_flips),
        warmup=bool(warmup),
        local_selection_ordering_mode=str(local_selection_ordering_mode),
        threshold_abs=int(crossing_threshold_abs),
    )
    q_i16 = q_levels.flatten().to(torch.int16)
    new_acc_i32 = torch.zeros(numel, dtype=torch.int32, device=q_levels.device)
    for flat_index, carry_after in sorted(two_tier_plan.carry_after_by_flat_index.items()):
        new_acc_i32[int(flat_index)] = int(carry_after)
    new_acc_i32 = new_acc_i32.view_as(accumulators)
    candidate_idx = torch.tensor(
        crossing_eligible_flat_indices(rows, threshold_abs=int(crossing_threshold_abs)),
        dtype=torch.int64,
        device=q_levels.device,
    )
    pre_veto_selected = torch.tensor(
        list(two_tier_plan.pre_veto_flat_indices),
        dtype=torch.int64,
        device=q_levels.device,
    )
    if pre_veto_selected.numel() == 0:
        applied = pre_veto_selected
        applied_directions = torch.zeros_like(pre_veto_selected, dtype=torch.int16)
        applied_thresholds = torch.zeros_like(pre_veto_selected, dtype=torch.int32)
        replay_ce_vetoed = pre_veto_selected
        replay_ce_negative = pre_veto_selected
        replay_veto_directions = applied_directions
        replay_veto_thresholds = applied_thresholds
        pc_aux_negative = pre_veto_selected
        pc_aux_vetoed = pre_veto_selected
    else:
        selected_thresholds = torch.full_like(
            pre_veto_selected,
            crossing_threshold_abs,
            dtype=torch.int32,
        )
        directions = torch.tensor(
            [
                _applied_crossing_direction_from_carry(
                    int(two_tier_plan.carry_after_by_flat_index[int(flat_index)]),
                    threshold_abs=int(crossing_threshold_abs),
                )
                for flat_index in pre_veto_selected.detach().cpu().tolist()
            ],
            dtype=torch.int16,
            device=q_levels.device,
        )
        (
            applied,
            applied_directions,
            applied_thresholds,
            replay_ce_vetoed,
            replay_ce_negative,
            replay_veto_directions,
            replay_veto_thresholds,
            pc_aux_negative,
            pc_aux_vetoed,
        ) = _partition_pre_veto_by_replay_and_pc_veto(
            pre_veto_selected,
            directions,
            selected_thresholds,
            inputs,
        )

    pre_veto_count = int(pre_veto_selected.numel())
    applied_count = int(applied.numel())
    replay_count = int(replay_ce_vetoed.numel())
    pc_negative_count = int(pc_aux_negative.numel())
    pc_veto_count = int(pc_aux_vetoed.numel())
    stats: dict[str, int | float | bool | str] = {
        "scope": "per_tensor_local_update",
        "global_cap_policy": DEFERRED_GLOBAL_CAP,
        "local_selection_ordering_mode": str(local_selection_ordering_mode),
        "local_selection_ordering_seed": int(local_selection_ordering_seed),
        "local_selection_ordering_step": int(local_selection_ordering_step),
        "threshold_jitter_policy": "deferred_reject",
        "candidate_count": int(candidate_idx.numel()),
        "max_flips": int(max_flips),
        "pre_veto_selected_flip_count": pre_veto_count,
        "post_veto_would_apply_pre_cap_count": applied_count,
        "post_veto_acceptance_ratio_pre_cap": _safe_ratio(applied_count, pre_veto_count),
        "replay_ce_veto_count": replay_count,
        "pc_aux_negative_count": pc_negative_count,
        "pc_aux_mode": inputs.normalized_pc_aux_mode.value,
        "pc_aux_veto_enabled": inputs.normalized_pc_aux_mode == PcAuxMode.VETO,
        "pc_aux_veto_count": pc_veto_count,
        "pc_aux_veto_accumulator_residual_policy": (
            "q_mutation_veto_only_accumulator_retained"
            if inputs.normalized_pc_aux_mode == PcAuxMode.VETO and inputs.pc_aux_votes is not None
            else "not_enabled"
        ),
        "replay_ce_veto_consumes_threshold_event": (
            inputs.normalized_replay_ce_mode == ReplayCeMode.VETO
            and inputs.replay_ce_veto_votes is not None
        ),
        "vetoed_accumulator_residual_policy": (
            "subtract_threshold_then_clamp_without_q_mutation"
            if (
                inputs.normalized_replay_ce_mode == ReplayCeMode.VETO
                and inputs.replay_ce_veto_votes is not None
            )
            else "not_enabled"
        ),
        "vetoed_accumulator_clamp_count": replay_count,
        "vote_nonzero_count": int((votes != 0).sum().item()),
        "acc_abs_max_after_decay_vote": int(new_acc_i32.abs().max().item()) if new_acc_i32.numel() else 0,
        "two_tier_carry_w6_enabled": True,
        "two_tier_threshold_abs": int(crossing_threshold_abs),
        "two_tier_canonical_threshold_abs": int(CROSSING_THRESHOLD_ABS),
        "two_tier_vote_spec_threshold_abs": int(vote_spec_threshold_abs),
    }
    stats.update(_replay_ce_telemetry_stat_leaves(inputs, replay_ce_negative))
    assert_two_tier_threshold_receipt_consistent(stats)
    return VoteUpdatePlan(
        q_i16=q_i16.view_as(q_levels),
        new_acc_i32=new_acc_i32,
        candidate_indices=candidate_idx,
        pre_veto_selected_indices=pre_veto_selected,
        applied_indices=applied,
        applied_directions=applied_directions,
        applied_thresholds=applied_thresholds,
        replay_ce_veto_indices=replay_ce_vetoed,
        replay_veto_directions=replay_veto_directions,
        replay_veto_thresholds=replay_veto_thresholds,
        pc_aux_negative_indices=pc_aux_negative,
        pc_aux_veto_indices=pc_aux_vetoed,
        stats=stats,
    )


def plan_two_tier_vote_update_reference(
    state: VoteUpdateState,
    inputs: VoteUpdateInputs,
    spec: VoteUpdateSpec,
    *,
    validate_q_levels: bool = True,
    local_selection_ordering_mode: str = LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
    local_selection_ordering_seed: int = 0,
    local_selection_ordering_step: int = 0,
    warmup: bool = False,
) -> VoteUpdatePlan:
    _reject_event_coded_on_dense_vote_path(state, site="plan_two_tier_vote_update_reference")
    validate_vote_update_contract(state, inputs, spec, validate_q_levels=validate_q_levels)
    if str(local_selection_ordering_mode) != LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA:
        raise ValueError(
            "two_tier_carry_w6_enabled requires "
            f"local_selection_ordering_mode={LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA!r}, "
            f"got {local_selection_ordering_mode!r}"
        )
    failures = validate_two_tier_selector_inputs(inputs, enabled=True)
    if failures:
        raise ValueError("selector input validation failed: " + ",".join(failures))

    q_levels = state.q_levels
    accumulators = state.accumulators
    votes = inputs.votes
    vote_spec_threshold_abs = int(spec.threshold_abs)
    crossing_threshold_abs = int(CROSSING_THRESHOLD_ABS)
    numel = int(q_levels.numel())
    max_flips = spec.max_flips(numel)

    q_flat = q_levels.flatten().to(torch.int16)
    acc_flat = accumulators.flatten().to(torch.int32)
    vote_flat = votes.flatten().to(torch.int32)
    delta_flat = inputs.local_loss_delta.flatten()

    carry_after_i32 = carry_after_i32_tensor(acc_flat, vote_flat)
    crossing_mask = crossing_eligible_mask_from_tensors(
        q_flat,
        carry_after_i32,
        threshold_abs=int(crossing_threshold_abs),
    )
    candidate_idx = crossing_mask.nonzero(as_tuple=False).flatten().to(
        dtype=torch.int64,
        device=q_levels.device,
    )
    pre_veto_selected = select_flat_indices_by_local_loss_delta_tensor(
        delta_flat,
        crossing_mask,
        rate_cap=int(max_flips),
    ).to(device=q_levels.device)
    new_acc_i32 = carry_after_i32.view_as(accumulators)

    if pre_veto_selected.numel() == 0:
        applied = pre_veto_selected
        applied_directions = torch.zeros_like(pre_veto_selected, dtype=torch.int16)
        applied_thresholds = torch.zeros_like(pre_veto_selected, dtype=torch.int32)
        replay_ce_vetoed = pre_veto_selected
        replay_ce_negative = pre_veto_selected
        replay_veto_directions = applied_directions
        replay_veto_thresholds = applied_thresholds
        pc_aux_negative = pre_veto_selected
        pc_aux_vetoed = pre_veto_selected
    else:
        carry_at_pre_veto = carry_after_i32[pre_veto_selected]
        directions = torch.where(
            carry_at_pre_veto >= int(crossing_threshold_abs),
            torch.ones_like(pre_veto_selected, dtype=torch.int16),
            torch.full_like(pre_veto_selected, -1, dtype=torch.int16),
        )
        selected_thresholds = torch.full_like(
            pre_veto_selected,
            crossing_threshold_abs,
            dtype=torch.int32,
        )
        (
            applied,
            applied_directions,
            applied_thresholds,
            replay_ce_vetoed,
            replay_ce_negative,
            replay_veto_directions,
            replay_veto_thresholds,
            pc_aux_negative,
            pc_aux_vetoed,
        ) = _partition_pre_veto_by_replay_and_pc_veto(
            pre_veto_selected,
            directions,
            selected_thresholds,
            inputs,
        )

    stats = _two_tier_vote_update_stats(
        votes=votes,
        new_acc_i32=new_acc_i32,
        candidate_idx=candidate_idx,
        pre_veto_selected=pre_veto_selected,
        applied=applied,
        replay_ce_vetoed=replay_ce_vetoed,
        replay_ce_negative=replay_ce_negative,
        pc_aux_negative=pc_aux_negative,
        pc_aux_vetoed=pc_aux_vetoed,
        max_flips=int(max_flips),
        vote_spec_threshold_abs=int(vote_spec_threshold_abs),
        crossing_threshold_abs=int(crossing_threshold_abs),
        local_selection_ordering_mode=str(local_selection_ordering_mode),
        local_selection_ordering_seed=int(local_selection_ordering_seed),
        local_selection_ordering_step=int(local_selection_ordering_step),
        inputs=inputs,
    )
    return VoteUpdatePlan(
        q_i16=q_flat.view_as(q_levels),
        new_acc_i32=new_acc_i32,
        candidate_indices=candidate_idx,
        pre_veto_selected_indices=pre_veto_selected,
        applied_indices=applied,
        applied_directions=applied_directions,
        applied_thresholds=applied_thresholds,
        replay_ce_veto_indices=replay_ce_vetoed,
        replay_veto_directions=replay_veto_directions,
        replay_veto_thresholds=replay_veto_thresholds,
        pc_aux_negative_indices=pc_aux_negative,
        pc_aux_veto_indices=pc_aux_vetoed,
        stats=stats,
    )


def apply_two_tier_vote_update_reference(
    state: VoteUpdateState,
    inputs: VoteUpdateInputs,
    spec: VoteUpdateSpec,
    *,
    validate_q_levels: bool = True,
    local_selection_ordering_mode: str = LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
    local_selection_ordering_seed: int = 0,
    local_selection_ordering_step: int = 0,
    warmup: bool = False,
) -> VoteUpdateResult:
    _reject_event_coded_on_dense_vote_path(state, site="apply_two_tier_vote_update_reference")
    plan = plan_two_tier_vote_update_reference(
        state,
        inputs,
        spec,
        validate_q_levels=validate_q_levels,
        local_selection_ordering_mode=str(local_selection_ordering_mode),
        local_selection_ordering_seed=int(local_selection_ordering_seed),
        local_selection_ordering_step=int(local_selection_ordering_step),
        warmup=bool(warmup),
    )
    crossing_threshold_abs = int(CROSSING_THRESHOLD_ABS)
    pre_veto_flat_indices = plan.pre_veto_selected_indices.detach().cpu().tolist()
    carry_flat = plan.new_acc_i32.flatten()
    q_flat = state.q_levels.flatten()
    carry_after_by_flat_index = {
        int(flat_index): int(carry_flat[int(flat_index)].item())
        for flat_index in pre_veto_flat_indices
    }
    q_level_by_flat_index = {
        int(flat_index): int(q_flat[int(flat_index)].item())
        for flat_index in pre_veto_flat_indices
    }
    materialized_rows = _minimal_two_tier_rows_for_flat_indices(
        plan.pre_veto_selected_indices,
        q_levels=state.q_levels,
        accumulators=state.accumulators,
        votes=inputs.votes,
        local_loss_delta=inputs.local_loss_delta,
    )
    two_tier_plan = build_two_tier_step_plan_for_apply(
        carry_after_by_flat_index=carry_after_by_flat_index,
        q_level_by_flat_index=q_level_by_flat_index,
        pre_veto_flat_indices=pre_veto_flat_indices,
        materialized_rows=materialized_rows,
        warmup=bool(warmup),
        local_selection_ordering_mode=str(local_selection_ordering_mode),
        threshold_abs=int(crossing_threshold_abs),
    )
    applied_indices = tuple(int(value) for value in plan.applied_indices.detach().cpu().tolist())
    two_tier_result = apply_two_tier_write_backs(
        two_tier_plan,
        applied_indices,
        threshold_abs=int(crossing_threshold_abs),
    )
    q_i16 = state.q_levels.flatten().to(torch.int16).clone()
    new_acc_i32 = plan.new_acc_i32.flatten().clone()
    for write_back in two_tier_result.applied_write_backs:
        flat_index = int(write_back.flat_index)
        q_i16[flat_index] = int(write_back.current_q_level)
        new_acc_i32[flat_index] = int(write_back.post_accumulator_carry)
    if inputs.normalized_replay_ce_mode == ReplayCeMode.VETO:
        _apply_replay_veto_residual_clamp(
            new_acc_i32,
            replay_ce_veto_indices=plan.replay_ce_veto_indices,
            replay_veto_directions=plan.replay_veto_directions,
            replay_veto_thresholds=plan.replay_veto_thresholds,
        )
    q_out = q_i16.view_as(state.q_levels).to(torch.int8).contiguous()
    acc_out = new_acc_i32.view_as(state.accumulators).to(torch.int16).contiguous()
    stats = dict(plan.stats)
    stats.update({
        "flip_count": int(plan.applied_indices.numel()),
        "post_veto_applied_flip_count": int(plan.applied_indices.numel()),
        "acc_abs_max_after": int(acc_out.abs().max().item()) if acc_out.numel() else 0,
        "q_changed_count": int((q_out != state.q_levels).sum().item()),
    })
    return VoteUpdateResult(q_levels=q_out, accumulators=acc_out, plan=plan, stats=stats)


def validate_vote_update_contract(
    state: VoteUpdateState,
    inputs: VoteUpdateInputs,
    spec: VoteUpdateSpec,
    *,
    validate_q_levels: bool = True,
) -> None:
    """Validate the transitional q:int8, acc:int16, vote:int16 local law."""

    spec.validate()
    _validate_future_formats(state, inputs)
    q = state.q_levels
    acc = state.accumulators
    votes = inputs.votes
    if q.dtype.is_floating_point:
        raise ValueError("FP master tensors are not accepted; pass q:int8 levels")
    if q.dtype != torch.int8:
        raise ValueError(f"q_levels must be torch.int8, got {q.dtype}")
    if acc.dtype.is_floating_point:
        raise ValueError("FP optimizer/moment-like accumulators are not accepted; pass int16 accumulators")
    if acc.dtype != torch.int16:
        raise ValueError(f"accumulators must be torch.int16, got {acc.dtype}")
    if votes.dtype != torch.int16:
        raise ValueError(f"votes must be torch.int16, got {votes.dtype}")
    if q.shape != acc.shape or q.shape != votes.shape:
        raise ValueError("q_levels, accumulators, and votes must have identical shapes")
    if q.device != acc.device or q.device != votes.device:
        raise ValueError("q_levels, accumulators, and votes must be on the same device")
    if validate_q_levels:
        allowed = torch.tensor([-1, 0, 1], dtype=torch.int8, device=q.device)
        if not bool(torch.isin(q, allowed).all().item()):
            raise ValueError("q_levels must contain only ternary int8 levels {-1, 0, +1}")

    optional_shapes = {
        "replay_ce_veto_votes": (inputs.replay_ce_veto_votes, torch.int16),
        "pc_aux_votes": (inputs.pc_aux_votes, torch.int16),
        "replay_ce_veto_moves": (inputs.replay_ce_veto_moves, torch.int8),
        "pc_aux_moves": (inputs.pc_aux_moves, torch.int8),
    }
    if inputs.replay_ce_veto_votes is not None or inputs.replay_ce_veto_moves is not None:
        if inputs.replay_ce_veto_votes is None or inputs.replay_ce_veto_moves is None:
            raise ValueError("replay-CE veto requires both votes:int16 and moves:int8")
    if inputs.pc_aux_votes is not None or inputs.pc_aux_moves is not None:
        if inputs.pc_aux_votes is None or inputs.pc_aux_moves is None:
            raise ValueError("PC auxiliary support requires both votes:int16 and moves:int8")
    for name, (tensor, dtype) in optional_shapes.items():
        if tensor is None:
            continue
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
        if tensor.shape != votes.shape:
            raise ValueError(f"{name} shape must match votes shape")
        if tensor.device != votes.device:
            raise ValueError(f"{name} device must match votes device")


def plan_integer_vote_update_reference(
    state: VoteUpdateState,
    inputs: VoteUpdateInputs,
    spec: VoteUpdateSpec,
    *,
    validate_q_levels: bool = True,
    local_selection_ordering_mode: str = LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    local_selection_ordering_seed: int = 0,
    local_selection_ordering_step: int = 0,
    two_tier_carry_w6_enabled: bool = False,
) -> VoteUpdatePlan:
    """Non-mutating exact local update plan for q:int8 + acc/vote:int16."""

    if two_tier_carry_w6_enabled:
        return plan_two_tier_vote_update_reference(
            state,
            inputs,
            spec,
            validate_q_levels=validate_q_levels,
            local_selection_ordering_mode=str(local_selection_ordering_mode),
            local_selection_ordering_seed=int(local_selection_ordering_seed),
            local_selection_ordering_step=int(local_selection_ordering_step),
        )
    _reject_event_coded_on_dense_vote_path(state, site="plan_integer_vote_update_reference")
    # === OFF body unchanged below this line ===
    validate_vote_update_contract(state, inputs, spec, validate_q_levels=validate_q_levels)
    q_levels = state.q_levels
    accumulators = state.accumulators
    votes = inputs.votes
    threshold = int(spec.threshold_abs)
    numel = int(q_levels.numel())
    max_flips = spec.max_flips(numel)
    if str(local_selection_ordering_mode) not in {
        LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
        LOCAL_SELECTION_ORDER_DETERMINISTIC_HASH_MATCHED,
    }:
        raise ValueError(
            f"unsupported local_selection_ordering_mode {local_selection_ordering_mode!r}"
        )

    q_i16 = q_levels.flatten().to(torch.int16)
    acc_i32 = accumulators.flatten().to(torch.int32)
    vote_i32 = votes.flatten().to(torch.int32)
    decayed = torch.div(
        acc_i32 * int(spec.decay_numerator),
        int(spec.decay_denominator),
        rounding_mode="trunc",
    )
    new_acc_i32 = (decayed + vote_i32).clamp(
        int(spec.accumulator_clip_min),
        int(spec.accumulator_clip_max),
    )
    candidates = ((new_acc_i32 >= threshold) & (q_i16 < 1)) | (
        (new_acc_i32 <= -threshold) & (q_i16 > -1)
    )
    candidate_idx = torch.nonzero(candidates, as_tuple=False).flatten()
    pre_veto_selected = candidate_idx[:0]
    applied = candidate_idx[:0]
    applied_directions = torch.zeros_like(candidate_idx[:0], dtype=torch.int16)
    applied_thresholds = torch.zeros_like(candidate_idx[:0], dtype=torch.int32)
    replay_ce_vetoed = candidate_idx[:0]
    replay_ce_negative = candidate_idx[:0]
    replay_veto_directions = torch.zeros_like(candidate_idx[:0], dtype=torch.int16)
    replay_veto_thresholds = torch.zeros_like(candidate_idx[:0], dtype=torch.int32)
    pc_aux_negative = candidate_idx[:0]
    pc_aux_vetoed = candidate_idx[:0]

    if candidate_idx.numel() > 0 and max_flips > 0:
        order = _local_selection_order(
            candidate_idx=candidate_idx,
            new_acc_i32=new_acc_i32,
            numel=numel,
            mode=str(local_selection_ordering_mode),
            ordering_seed=int(local_selection_ordering_seed),
            ordering_step=int(local_selection_ordering_step),
        )
        pre_veto_selected = candidate_idx[order[:max_flips]]
        selected_thresholds = torch.full_like(pre_veto_selected, threshold, dtype=torch.int32)
        directions = torch.where(new_acc_i32[pre_veto_selected] >= threshold, 1, -1).to(torch.int16)
        (
            applied,
            applied_directions,
            applied_thresholds,
            replay_ce_vetoed,
            replay_ce_negative,
            replay_veto_directions,
            replay_veto_thresholds,
            pc_aux_negative,
            pc_aux_vetoed,
        ) = _partition_pre_veto_by_replay_and_pc_veto(
            pre_veto_selected,
            directions,
            selected_thresholds,
            inputs,
        )

    pre_veto_count = int(pre_veto_selected.numel())
    applied_count = int(applied.numel())
    replay_count = int(replay_ce_vetoed.numel())
    pc_negative_count = int(pc_aux_negative.numel())
    pc_veto_count = int(pc_aux_vetoed.numel())
    stats: dict[str, int | float | bool | str] = {
        "scope": "per_tensor_local_update",
        "global_cap_policy": DEFERRED_GLOBAL_CAP,
        "local_selection_ordering_mode": str(local_selection_ordering_mode),
        "local_selection_ordering_seed": int(local_selection_ordering_seed),
        "local_selection_ordering_step": int(local_selection_ordering_step),
        "threshold_jitter_policy": "deferred_reject",
        "candidate_count": int(candidate_idx.numel()),
        "max_flips": int(max_flips),
        "pre_veto_selected_flip_count": pre_veto_count,
        "post_veto_would_apply_pre_cap_count": applied_count,
        "post_veto_acceptance_ratio_pre_cap": _safe_ratio(applied_count, pre_veto_count),
        "replay_ce_veto_count": replay_count,
        "pc_aux_negative_count": pc_negative_count,
        "pc_aux_mode": inputs.normalized_pc_aux_mode.value,
        "pc_aux_veto_enabled": inputs.normalized_pc_aux_mode == PcAuxMode.VETO,
        "pc_aux_veto_count": pc_veto_count,
        "pc_aux_veto_accumulator_residual_policy": (
            "q_mutation_veto_only_accumulator_retained"
            if inputs.normalized_pc_aux_mode == PcAuxMode.VETO and inputs.pc_aux_votes is not None
            else "not_enabled"
        ),
        "replay_ce_veto_consumes_threshold_event": (
            inputs.normalized_replay_ce_mode == ReplayCeMode.VETO
            and inputs.replay_ce_veto_votes is not None
        ),
        "vetoed_accumulator_residual_policy": (
            "subtract_threshold_then_clamp_without_q_mutation"
            if (
                inputs.normalized_replay_ce_mode == ReplayCeMode.VETO
                and inputs.replay_ce_veto_votes is not None
            )
            else "not_enabled"
        ),
        "vetoed_accumulator_clamp_count": replay_count,
        "vote_nonzero_count": int((votes != 0).sum().item()),
        "acc_abs_max_after_decay_vote": int(new_acc_i32.abs().max().item()) if new_acc_i32.numel() else 0,
    }
    stats.update(_replay_ce_telemetry_stat_leaves(inputs, replay_ce_negative))
    return VoteUpdatePlan(
        q_i16=q_i16.view_as(q_levels),
        new_acc_i32=new_acc_i32.view_as(accumulators),
        candidate_indices=candidate_idx.to(torch.int64),
        pre_veto_selected_indices=pre_veto_selected.to(torch.int64),
        applied_indices=applied.to(torch.int64),
        applied_directions=applied_directions.to(torch.int16),
        applied_thresholds=applied_thresholds.to(torch.int32),
        replay_ce_veto_indices=replay_ce_vetoed.to(torch.int64),
        replay_veto_directions=replay_veto_directions.to(torch.int16),
        replay_veto_thresholds=replay_veto_thresholds.to(torch.int32),
        pc_aux_negative_indices=pc_aux_negative.to(torch.int64),
        pc_aux_veto_indices=pc_aux_vetoed.to(torch.int64),
        stats=stats,
    )


def _validate_ternary_q_levels(state: VoteUpdateState) -> None:
    q = state.q_levels
    if q.dtype.is_floating_point or q.dtype != torch.int8:
        raise ValueError(
            "FP master tensors are not accepted; pass q:int8 levels"
            if q.dtype.is_floating_point
            else f"q_levels must be torch.int8, got {q.dtype}"
        )
    allowed = torch.tensor([-1, 0, 1], dtype=torch.int8, device=q.device)
    if not bool(torch.isin(q, allowed).all().item()):
        raise ValueError("q_levels must contain only ternary int8 levels {-1, 0, +1}")


def _reject_duplicate_or_oor_indices(name: str, indices: torch.Tensor, *, numel: int) -> None:
    if indices.numel() == 0:
        return
    if bool((indices < 0).any().item()) or bool((indices >= int(numel)).any().item()):
        raise ValueError(f"{name} contains out-of-range indices for numel={int(numel)}")
    if indices.numel() > 1:
        sorted_idx, _ = torch.sort(indices)
        if bool((sorted_idx[1:] == sorted_idx[:-1]).any().item()):
            raise ValueError(f"{name} contains duplicate indices")


_SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64)
_SUPPORTED_DIR_THR_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)


def _reject_pm1_dirs_and_positive_thresholds(
    *, directions: torch.Tensor, thresholds: torch.Tensor, dir_name: str, thr_name: str
) -> None:
    if directions.numel() == 0:
        return
    if bool((directions.abs() != 1).any().item()):
        raise ValueError(f"{dir_name} must be exactly ±1")
    if bool((thresholds <= 0).any().item()):
        raise ValueError(f"{thr_name} must be positive")


def _validate_frozen_plan_binding(state: VoteUpdateState, plan: VoteUpdatePlan) -> None:
    """Fail-closed state↔plan binding for public from_frozen_plan callers."""
    if plan.q_i16.dtype != torch.int16 or plan.new_acc_i32.dtype != torch.int32:
        raise ValueError("plan.q_i16 must be int16 and plan.new_acc_i32 must be int32")
    if int(plan.q_i16.numel()) != int(state.q_levels.numel()):
        raise ValueError("plan.q_i16 numel must match state.q_levels")
    if int(plan.new_acc_i32.numel()) != int(state.accumulators.numel()):
        raise ValueError("plan.new_acc_i32 numel must match state.accumulators")
    try:
        q_plan = plan.q_i16.view_as(state.q_levels)
        plan.new_acc_i32.view_as(state.accumulators)
    except RuntimeError as exc:
        raise ValueError("plan q/acc tensors must view_as state q/acc shapes") from exc
    if bool((q_plan != state.q_levels.to(dtype=torch.int16, device=plan.q_i16.device)).any().item()):
        raise ValueError("plan.q_i16 must equal state.q_levels as int16 (q base binding)")

    arrays = (
        ("applied_indices", plan.applied_indices, _SUPPORTED_INDEX_DTYPES, "int32/int64"),
        ("applied_directions", plan.applied_directions, _SUPPORTED_DIR_THR_DTYPES, "explicit integer dtype int8/16/32/64"),
        ("applied_thresholds", plan.applied_thresholds, _SUPPORTED_DIR_THR_DTYPES, "explicit integer dtype int8/16/32/64"),
        ("replay_ce_veto_indices", plan.replay_ce_veto_indices, _SUPPORTED_INDEX_DTYPES, "int32/int64"),
        ("replay_veto_directions", plan.replay_veto_directions, _SUPPORTED_DIR_THR_DTYPES, "explicit integer dtype int8/16/32/64"),
        ("replay_veto_thresholds", plan.replay_veto_thresholds, _SUPPORTED_DIR_THR_DTYPES, "explicit integer dtype int8/16/32/64"),
    )
    for name, tensor, allowed, label in arrays:
        if int(tensor.ndim) != 1:
            raise ValueError(f"{name} must have ndim==1 before coercion, got ndim={int(tensor.ndim)}")
        if tensor.dtype not in allowed:  # dtype before cardinality
            raise ValueError(f"{name} must be {label}, got {tensor.dtype}")

    numel = int(state.q_levels.numel())
    applied, replay = plan.applied_indices, plan.replay_ce_veto_indices
    if int(plan.applied_directions.numel()) != int(applied.numel()):
        raise ValueError("applied_directions length must match applied_indices")
    if int(plan.applied_thresholds.numel()) != int(applied.numel()):
        raise ValueError("applied_thresholds length must match applied_indices")
    if int(plan.replay_veto_directions.numel()) != int(replay.numel()):
        raise ValueError("replay_veto_directions length must match replay_ce_veto_indices")
    if int(plan.replay_veto_thresholds.numel()) != int(replay.numel()):
        raise ValueError("replay_veto_thresholds length must match replay_ce_veto_indices")
    _reject_duplicate_or_oor_indices("applied_indices", applied, numel=numel)
    _reject_duplicate_or_oor_indices("replay_ce_veto_indices", replay, numel=numel)
    _reject_pm1_dirs_and_positive_thresholds(
        directions=plan.applied_directions, thresholds=plan.applied_thresholds,
        dir_name="applied_directions", thr_name="applied_thresholds",
    )
    _reject_pm1_dirs_and_positive_thresholds(
        directions=plan.replay_veto_directions, thresholds=plan.replay_veto_thresholds,
        dir_name="replay_veto_directions", thr_name="replay_veto_thresholds",
    )


def _apply_integer_vote_update_from_frozen_plan_trusted(
    state: VoteUpdateState, plan: VoteUpdatePlan,
) -> VoteUpdateResult:
    """Private writeback-only core (trusted planner output or post-validation public path)."""
    _reject_event_coded_on_dense_vote_path(
        state, site="_apply_integer_vote_update_from_frozen_plan_trusted"
    )
    q_i16 = plan.q_i16.flatten().clone()
    new_acc_i32 = plan.new_acc_i32.flatten().clone()
    if plan.applied_indices.numel() > 0:
        applied = plan.applied_indices.to(q_i16.device)
        directions = plan.applied_directions.to(q_i16.device)
        thresholds = plan.applied_thresholds.to(new_acc_i32.device)
        q_i16[applied] = (q_i16[applied] + directions).clamp(-1, 1)
        residual = new_acc_i32[applied] - directions.to(torch.int32) * thresholds
        new_acc_i32[applied] = torch.minimum(torch.maximum(residual, -thresholds + 1), thresholds - 1)
    _apply_replay_veto_residual_clamp(
        new_acc_i32,
        replay_ce_veto_indices=plan.replay_ce_veto_indices,
        replay_veto_directions=plan.replay_veto_directions,
        replay_veto_thresholds=plan.replay_veto_thresholds,
    )
    q_out = q_i16.view_as(state.q_levels).to(torch.int8).contiguous()
    acc_out = new_acc_i32.view_as(state.accumulators).to(torch.int16).contiguous()
    stats = dict(plan.stats)
    stats.update({
        "flip_count": int(plan.applied_indices.numel()),
        "post_veto_applied_flip_count": int(plan.applied_indices.numel()),
        "acc_abs_max_after": int(acc_out.abs().max().item()) if acc_out.numel() else 0,
        "q_changed_count": int((q_out != state.q_levels).sum().item()),
    })
    return VoteUpdateResult(q_levels=q_out, accumulators=acc_out, plan=plan, stats=stats)


def apply_integer_vote_update_from_frozen_plan(
    state: VoteUpdateState, plan: VoteUpdatePlan,
) -> VoteUpdateResult:
    """Public dense from-frozen-plan apply. Always fail-closed; no validation bypass kwargs."""
    _reject_event_coded_on_dense_vote_path(state, site="apply_integer_vote_update_from_frozen_plan")
    _validate_ternary_q_levels(state)
    _validate_frozen_plan_binding(state, plan)
    return _apply_integer_vote_update_from_frozen_plan_trusted(state, plan)


def apply_integer_vote_update_reference(
    state: VoteUpdateState,
    inputs: VoteUpdateInputs,
    spec: VoteUpdateSpec,
    *,
    validate_q_levels: bool = True,
    local_selection_ordering_mode: str = LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    local_selection_ordering_seed: int = 0,
    local_selection_ordering_step: int = 0,
    two_tier_carry_w6_enabled: bool = False,
) -> VoteUpdateResult:
    """Apply the exact local update law to copies and return new q/acc tensors."""

    _reject_event_coded_on_dense_vote_path(state, site="apply_integer_vote_update_reference")
    if two_tier_carry_w6_enabled:
        return apply_two_tier_vote_update_reference(
            state,
            inputs,
            spec,
            validate_q_levels=validate_q_levels,
            local_selection_ordering_mode=str(local_selection_ordering_mode),
            local_selection_ordering_seed=int(local_selection_ordering_seed),
            local_selection_ordering_step=int(local_selection_ordering_step),
        )
    # === OFF body: plan once, then private trusted writeback (no public revalidation) ===
    plan = plan_integer_vote_update_reference(
        state,
        inputs,
        spec,
        validate_q_levels=validate_q_levels,
        local_selection_ordering_mode=str(local_selection_ordering_mode),
        local_selection_ordering_seed=int(local_selection_ordering_seed),
        local_selection_ordering_step=int(local_selection_ordering_step),
    )
    return _apply_integer_vote_update_from_frozen_plan_trusted(state, plan)


def _coerce_flat_indices(
    name: str,
    values: torch.Tensor,
    *,
    device: torch.device,
    numel: int,
) -> torch.Tensor:
    if values.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be int32/int64, got {values.dtype}")
    out = values.flatten().to(device=device, dtype=torch.int64)
    if out.numel() == 0:
        return out
    if bool((out < 0).any().item()) or bool((out >= int(numel)).any().item()):
        raise ValueError(f"{name} contains out-of-range flat indices")
    return out


def _coerce_flat_i16(name: str, values: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"{name} must be an integer tensor, got {values.dtype}")
    return values.flatten().to(device=device, dtype=torch.int16)


def _coerce_flat_i32(name: str, values: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if values.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError(f"{name} must be an integer tensor, got {values.dtype}")
    return values.flatten().to(device=device, dtype=torch.int32)


def _validate_apply_rows(
    *,
    name: str,
    indices: torch.Tensor,
    directions: torch.Tensor,
    thresholds: torch.Tensor,
) -> None:
    if indices.numel() != directions.numel() or indices.numel() != thresholds.numel():
        raise ValueError(f"{name} indices/directions/thresholds must have matching lengths")
    if thresholds.numel() and bool((thresholds <= 0).any().item()):
        raise ValueError(f"{name} thresholds must be > 0")
    if directions.numel():
        invalid = (directions != 1) & (directions != -1)
        if bool(invalid.any().item()):
            raise ValueError(f"{name} directions must be -1 or +1")


def _apply_threshold_residual_in_place(
    acc_i32: torch.Tensor,
    *,
    indices: torch.Tensor,
    directions: torch.Tensor,
    thresholds: torch.Tensor,
) -> None:
    if indices.numel() == 0:
        return
    residual = acc_i32[indices] - directions.to(torch.int32) * thresholds
    low = -thresholds + 1
    high = thresholds - 1
    acc_i32[indices] = torch.minimum(torch.maximum(residual, low), high)


def q_acc_apply_mutation_torch_cuda_reference_under_cap_rows(
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
    scope: str = Q_ACC_APPLY_MUTATION_TORCH_CUDA_REFERENCE_SCOPE,
) -> QAccApplyMutationResult:
    """Torch-CUDA reference for final cap-accepted q/acc mutation rows.

    Global cap selection stays CPU/control-flow glue. This function consumes the
    final cap-accepted rows plus replay-veto residual rows and applies only the
    sparse q flip/write and accumulator residual update on CUDA copies.
    """

    if os.environ.get(RUN_GPU_Q_ACC_APPLY_ENV) != "1":
        raise RuntimeError(
            f"{RUN_GPU_Q_ACC_APPLY_ENV}=1 is required and must only be set inside "
            "a granted gpu:0 resource lane"
        )
    if q_levels.dtype != torch.int8:
        raise ValueError(f"q_levels must be torch.int8, got {q_levels.dtype}")
    if new_accumulators.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError(
            f"new_accumulators must be int16/int32/int64, got {new_accumulators.dtype}"
        )
    if q_levels.shape != new_accumulators.shape:
        raise ValueError("q_levels and new_accumulators must have identical shapes")
    if q_levels.device.type != "cuda" or new_accumulators.device != q_levels.device:
        raise ValueError("q_acc apply reference requires q/new_acc tensors on the same CUDA device")
    if not bool(mutate_outputs):
        if original_accumulators is None:
            raise ValueError("original_accumulators is required when mutate_outputs=False")
        if original_accumulators.dtype != torch.int16:
            raise ValueError(
                f"original_accumulators must be torch.int16, got {original_accumulators.dtype}"
            )
        if original_accumulators.shape != q_levels.shape:
            raise ValueError("original_accumulators shape must match q_levels")

    device = q_levels.device
    numel = int(q_levels.numel())
    accepted = _coerce_flat_indices(
        "accepted_indices",
        accepted_indices,
        device=device,
        numel=numel,
    )
    accepted_dirs = _coerce_flat_i16(
        "accepted_directions",
        accepted_directions,
        device=device,
    )
    accepted_thresholds_i32 = _coerce_flat_i32(
        "accepted_thresholds",
        accepted_thresholds,
        device=device,
    )
    _validate_apply_rows(
        name="accepted rows",
        indices=accepted,
        directions=accepted_dirs,
        thresholds=accepted_thresholds_i32,
    )

    replay_parts = (replay_veto_indices, replay_veto_directions, replay_veto_thresholds)
    if any(part is not None for part in replay_parts):
        if any(part is None for part in replay_parts):
            raise ValueError("replay-veto rows require indices, directions, and thresholds")
        replay = _coerce_flat_indices(
            "replay_veto_indices",
            replay_veto_indices,
            device=device,
            numel=numel,
        )
        replay_dirs = _coerce_flat_i16(
            "replay_veto_directions",
            replay_veto_directions,
            device=device,
        )
        replay_thresholds_i32 = _coerce_flat_i32(
            "replay_veto_thresholds",
            replay_veto_thresholds,
            device=device,
        )
    else:
        replay = torch.empty(0, dtype=torch.int64, device=device)
        replay_dirs = torch.empty(0, dtype=torch.int16, device=device)
        replay_thresholds_i32 = torch.empty(0, dtype=torch.int32, device=device)
    _validate_apply_rows(
        name="replay-veto rows",
        indices=replay,
        directions=replay_dirs,
        thresholds=replay_thresholds_i32,
    )

    if not bool(mutate_outputs):
        q_out = q_levels.detach().clone().contiguous()
        acc_out = original_accumulators.to(device=device).detach().clone().contiguous()
    else:
        q_i16 = q_levels.flatten().to(torch.int16).clone()
        acc_i32 = new_accumulators.flatten().to(torch.int32).clone()
        if accepted.numel() > 0:
            q_i16[accepted] = (q_i16[accepted] + accepted_dirs).clamp(-1, 1)
            _apply_threshold_residual_in_place(
                acc_i32,
                indices=accepted,
                directions=accepted_dirs,
                thresholds=accepted_thresholds_i32,
            )
        _apply_threshold_residual_in_place(
            acc_i32,
            indices=replay,
            directions=replay_dirs,
            thresholds=replay_thresholds_i32,
        )
        q_out = q_i16.view_as(q_levels).to(torch.int8).contiguous()
        acc_out = acc_i32.view_as(new_accumulators).to(torch.int16).contiguous()

    stats = {
        "scope": scope,
        "cap_rows_fixture_required": True,
        "global_cap_gpu_native": False,
        "packed_state": False,
        "mutate_outputs": bool(mutate_outputs),
        "accepted_count": int(accepted.numel()),
        "replay_veto_count": int(replay.numel()),
        "q_changed_count": int((q_out != q_levels).sum().item()),
    }
    return QAccApplyMutationResult(
        q_levels=q_out,
        accumulators=acc_out,
        scope=scope,
        backend=device.type,
        stats=stats,
    )


if triton is not None:

    @triton.jit
    def _vote_update_preplan_kernel(
        Q,
        ACC,
        VOTES,
        NEW_ACC,
        CANDIDATE,
        DIRECTION,
        N: tl.constexpr,
        THRESHOLD: tl.constexpr,
        CLIP_MIN: tl.constexpr,
        CLIP_MAX: tl.constexpr,
        DECAY_NUM: tl.constexpr,
        DECAY_DEN: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        q = tl.load(Q + offs, mask=mask, other=0).to(tl.int16)
        acc = tl.load(ACC + offs, mask=mask, other=0).to(tl.int32)
        votes = tl.load(VOTES + offs, mask=mask, other=0).to(tl.int32)
        prod = acc * DECAY_NUM
        abs_prod = tl.where(prod < 0, -prod, prod)
        dec_abs = abs_prod // DECAY_DEN
        decayed = tl.where(prod < 0, -dec_abs, dec_abs)
        new_acc = decayed + votes
        new_acc = tl.minimum(tl.maximum(new_acc, CLIP_MIN), CLIP_MAX)
        pos_candidate = (new_acc >= THRESHOLD) & (q < 1)
        neg_candidate = (new_acc <= -THRESHOLD) & (q > -1)
        candidate = pos_candidate | neg_candidate
        direction = tl.where(new_acc >= THRESHOLD, 1, -1)
        tl.store(NEW_ACC + offs, new_acc.to(tl.int16), mask=mask)
        tl.store(CANDIDATE + offs, candidate.to(tl.int8), mask=mask)
        tl.store(DIRECTION + offs, direction.to(tl.int8), mask=mask)

else:
    _vote_update_preplan_kernel = None


def vote_update_preplan_triton(
    state: VoteUpdateState,
    inputs: VoteUpdateInputs,
    spec: VoteUpdateSpec,
    *,
    block: int = 1024,
) -> dict[str, torch.Tensor | str]:
    """Default-off Triton elementwise preplan interface.

    This computes only decayed+vote accumulator values plus candidate/direction
    masks. Global ordering, veto residuals, and q mutation stay in the CPU
    reference until a resource-lane GPU receipt proves a fuller kernel.
    """

    if os.environ.get(RUN_GPU_VOTE_UPDATE_ENV) != "1":
        raise RuntimeError(
            f"{RUN_GPU_VOTE_UPDATE_ENV}=1 is required and must only be set inside "
            "a granted gpu:0 resource lane"
        )
    if _vote_update_preplan_kernel is None:
        raise RuntimeError("vote_update_preplan_triton requires Triton")
    validate_vote_update_contract(state, inputs, spec)
    if state.normalized_accumulator_format == VoteUpdateAccumulatorFormat.EVENT_CODED_LIVE_CARRIER:
        raise ValueError("vote_update_preplan_triton forbidden on event-coded live carrier path")
    if state.q_levels.device.type != "cuda":
        raise ValueError("vote_update_preplan_triton requires CUDA q/acc/votes tensors")
    if block <= 0:
        raise ValueError("block must be > 0")

    q = state.q_levels.contiguous()
    acc = state.accumulators.contiguous()
    votes = inputs.votes.contiguous()
    new_acc = torch.empty_like(acc)
    candidate = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    direction = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    grid = (triton.cdiv(q.numel(), block),)
    _vote_update_preplan_kernel[grid](
        q,
        acc,
        votes,
        new_acc,
        candidate,
        direction,
        int(q.numel()),
        int(spec.threshold_abs),
        int(spec.accumulator_clip_min),
        int(spec.accumulator_clip_max),
        int(spec.decay_numerator),
        int(spec.decay_denominator),
        BLOCK=int(block),
    )
    return {
        "scope": "elementwise_preplan_only_deferred_global_cap",
        "new_accumulators": new_acc,
        "candidate_mask_int8": candidate,
        "direction_int8": direction,
    }
