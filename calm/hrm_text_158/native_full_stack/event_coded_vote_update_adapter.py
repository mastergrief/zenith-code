"""V4-LIVE event-coded vote-update adapter (trainer integration seam)."""
from __future__ import annotations

import hashlib
import os
import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
    PackedEventCodedAccState,
    unpack_event_coded_acc_checkpoint_v1,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    DEFAULT_COLD_DEFAULT,
    DEFAULT_DECAY_DENOMINATOR,
    DEFAULT_DECAY_NUMERATOR,
    EventCodedAccLiveState,
    hot_risk_proxy_indices,
    observed_surfaces_dict,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    DEFAULT_CARRY_WIDTH,
    DEFAULT_CROSSING_THRESHOLD_ABS,
    VOTE_UPDATE_SOURCE_CLIP_MIN,
    VOTE_UPDATE_SOURCE_CLIP_MAX,
    carry_self_update_row,
    crossing_bool_w6,
    effective_clip_bounds,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    VoteUpdateAccumulatorFormat,
    VoteUpdateInputs,
    VoteUpdatePlan,
    VoteUpdateSpec,
    VoteUpdateState,
    _local_selection_order,
    _partition_pre_veto_by_replay_and_pc_veto,
    validate_vote_update_contract,
)

RUN_EVENT_CODED_ACC_LIVE_CARRIER_ENV = "HRM_TEXT_158_RUN_EVENT_CODED_ACC_LIVE_CARRIER"
LIVE_ACC_CARRIER_V4_LIVE = "v4_live"
LIVE_ACC_CARRIER_W5 = "w5"
LIVE_ACC_CARRIER_W6 = "w6"
LIVE_ACC_CARRIER_NONE = "none"

C8_DENSE_ACCUMULATOR_MATERIALIZED_NUMEL_KEY = "dense_accumulator_materialized_numel"
C8_FULL_NUMEL_FLATTEN_COUNT_KEY = "full_numel_flatten_count"
C8_VOTE_UPDATE_PREPLAN_TRITON_INVOKED_KEY = "vote_update_preplan_triton_invoked"
C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY = "transient_dense_compute_numel"
C8_PERSISTENT_AUTHORITY_SCOPE_KEY = "c8_persistent_authority_scope"
C8_PERSISTENT_AUTHORITY_SCOPE_VALUE = (
    "no_dense_persistent_authority; transient O(numel) runtime buffers remain "
    "(separate full_sub2_runtime lane)"
)
C8_LIVE_AUTHORITY_TAG = "event_coded_live_carrier"


@dataclass
class C8StepObservation:
    """Runtime measurements for one V4-LIVE vote-update step (not hardcoded constants)."""

    full_numel_flatten_count: int = 0
    vote_update_preplan_triton_invoked: bool = False
    transient_dense_compute_numel: int = 0

    def record_flatten(self) -> None:
        self.full_numel_flatten_count += 1

    def record_transient_dense(self, numel: int) -> None:
        self.transient_dense_compute_numel = max(
            self.transient_dense_compute_numel,
            int(numel),
        )


def measure_persistent_dense_accumulator_materialized_numel(
    *,
    exact_accumulator_shadow: torch.Tensor | None,
    event_coded_live_carrier: EventCodedAccLiveState | None,
    eligible_numel: int,
) -> int:
    """Count dense int16 accumulator authority persisted in next_states/checkpoint surface."""

    if event_coded_live_carrier is not None:
        if exact_accumulator_shadow is not None:
            return int(exact_accumulator_shadow.numel())
        return 0
    if exact_accumulator_shadow is None:
        return 0
    return int(exact_accumulator_shadow.numel())


def event_coded_live_carrier_enabled(*, enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return os.environ.get(RUN_EVENT_CODED_ACC_LIVE_CARRIER_ENV) == "1"


def resolve_live_acc_carrier_selector(
    *,
    v4_enabled: bool | None = None,
    w5_enabled: bool | None = None,
    w6_enabled: bool | None = None,
) -> str:
    from calm.hrm_text_158.native_full_stack.narrow_carrier_trainer_integration import (
        narrow_carrier_w5_enabled,
        narrow_carrier_w6_enabled,
    )

    use_v4 = event_coded_live_carrier_enabled(enabled=v4_enabled)
    use_w5 = narrow_carrier_w5_enabled(enabled=w5_enabled)
    use_w6 = narrow_carrier_w6_enabled(enabled=w6_enabled)
    selected = sum(int(flag) for flag in (use_v4, use_w5, use_w6))
    if selected > 1:
        raise ValueError(
            "V4-LIVE event-coded carrier is mutually exclusive with W5/W6 narrow carriers"
        )
    if use_v4:
        return LIVE_ACC_CARRIER_V4_LIVE
    if use_w5:
        return LIVE_ACC_CARRIER_W5
    if use_w6:
        return LIVE_ACC_CARRIER_W6
    return LIVE_ACC_CARRIER_NONE


def is_event_coded_vote_update_state(state: VoteUpdateState) -> bool:
    return (
        state.normalized_accumulator_format
        == VoteUpdateAccumulatorFormat.EVENT_CODED_LIVE_CARRIER
    )


def shape_only_accumulator_stub(
    q_levels: torch.Tensor,
    *,
    observation: C8StepObservation | None = None,
) -> torch.Tensor:
    """Non-authoritative shape anchor; values must not be read on the V4 path."""

    if observation is not None:
        observation.record_transient_dense(int(q_levels.numel()))
    return torch.zeros_like(q_levels, dtype=torch.int16)


@dataclass(frozen=True)
class EventCodedVoteUpdateState:
    q_levels: torch.Tensor
    carrier: EventCodedAccLiveState

    @property
    def normalized_accumulator_format(self) -> VoteUpdateAccumulatorFormat:
        return VoteUpdateAccumulatorFormat.EVENT_CODED_LIVE_CARRIER

    def to_vote_update_state(self) -> VoteUpdateState:
        return VoteUpdateState(
            q_levels=self.q_levels,
            accumulators=shape_only_accumulator_stub(self.q_levels),
            accumulator_format=VoteUpdateAccumulatorFormat.EVENT_CODED_LIVE_CARRIER,
        )


@dataclass(frozen=True)
class EventCodedVoteUpdateResult:
    q_levels: torch.Tensor
    carrier: EventCodedAccLiveState
    plan: VoteUpdatePlan
    stats: dict[str, Any]


def carrier_content_sha256(carrier: EventCodedAccLiveState) -> str:
    payload = carrier.to_checkpoint_payload()
    packed = bytes(payload.events_packed.detach().cpu().tolist())
    packed += bytes(payload.backlog_packed.detach().cpu().tolist())
    packed += bytes(payload.hot_exact_packed.detach().cpu().tolist())
    return hashlib.sha256(packed).hexdigest()


def c8_runtime_guard_stats(
    carrier: EventCodedAccLiveState,
    *,
    observation: C8StepObservation,
    persistent_dense_accumulator_materialized_numel: int,
) -> dict[str, Any]:
    return {
        C8_DENSE_ACCUMULATOR_MATERIALIZED_NUMEL_KEY: int(
            persistent_dense_accumulator_materialized_numel
        ),
        C8_FULL_NUMEL_FLATTEN_COUNT_KEY: int(observation.full_numel_flatten_count),
        C8_VOTE_UPDATE_PREPLAN_TRITON_INVOKED_KEY: bool(
            observation.vote_update_preplan_triton_invoked
        ),
        C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY: int(observation.transient_dense_compute_numel),
        C8_PERSISTENT_AUTHORITY_SCOPE_KEY: C8_PERSISTENT_AUTHORITY_SCOPE_VALUE,
        "live_authority": C8_LIVE_AUTHORITY_TAG,
        "event_coded_live_carrier_content_sha256_after": carrier_content_sha256(carrier),
    }


def assert_c8_runtime_guards(
    carrier: EventCodedAccLiveState,
    *,
    observation: C8StepObservation,
    persistent_dense_accumulator_materialized_numel: int,
) -> None:
    stats = c8_runtime_guard_stats(
        carrier,
        observation=observation,
        persistent_dense_accumulator_materialized_numel=int(
            persistent_dense_accumulator_materialized_numel
        ),
    )
    if int(stats[C8_DENSE_ACCUMULATOR_MATERIALIZED_NUMEL_KEY]) != 0:
        raise ValueError(
            "C8 guard failed: dense persistent accumulator authority must be 0 on V4-LIVE path"
        )
    if bool(stats[C8_VOTE_UPDATE_PREPLAN_TRITON_INVOKED_KEY]):
        raise ValueError("C8 guard failed: Triton preplan forbidden on V4-LIVE path")


def hydrate_event_coded_live_carrier_from_packed(
    packed: PackedEventCodedAccState | Any,
    *,
    demotion_band: int = 1,
    cold_default: int = DEFAULT_COLD_DEFAULT,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
) -> EventCodedAccLiveState:
    events, backlog, hot_indices, hot_values = unpack_event_coded_acc_checkpoint_v1(packed)
    hot_exact = {int(i): int(v) for i, v in zip(hot_indices, hot_values)}
    return EventCodedAccLiveState(
        logical_numel=int(packed.logical_numel),
        cold_default=int(cold_default),
        threshold_abs=int(threshold_abs),
        demotion_band=int(demotion_band),
        hot_exact=hot_exact,
        events=list(events),
        backlog=set(int(item) for item in backlog),
    )


def carrier_pre_accumulator_i32_flat(
    carrier: EventCodedAccLiveState,
    numel: int,
) -> torch.Tensor:
    pre_full = torch.full((int(numel),), int(carrier.cold_default), dtype=torch.int32)
    if carrier.hot_exact:
        hot_keys = torch.tensor(list(carrier.hot_exact.keys()), dtype=torch.int64)
        hot_vals = torch.tensor(list(carrier.hot_exact.values()), dtype=torch.int32)
        pre_full[hot_keys] = hot_vals
    return pre_full


def _active_lane_index_tensor(
    carrier: EventCodedAccLiveState,
    votes: torch.Tensor,
) -> torch.Tensor:
    vote_flat = votes.detach().cpu().flatten()
    vote_nz = torch.nonzero(vote_flat != 0, as_tuple=False).flatten().to(torch.int64)
    if not carrier.hot_exact:
        return vote_nz
    hot_idx = torch.tensor(list(carrier.hot_exact.keys()), dtype=torch.int64)
    if vote_nz.numel() == 0:
        return hot_idx
    return torch.unique(torch.cat([hot_idx, vote_nz]))


def _active_lane_indices(
    carrier: EventCodedAccLiveState,
    votes: torch.Tensor,
) -> set[int]:
    return {int(index) for index in _active_lane_index_tensor(carrier, votes).tolist()}


def build_sparse_new_acc_i32_from_carrier_reference(
    carrier: EventCodedAccLiveState,
    q_levels: torch.Tensor,
    votes: torch.Tensor,
    spec: VoteUpdateSpec,
    *,
    observation: C8StepObservation | None = None,
) -> torch.Tensor:
    numel = int(q_levels.numel())
    if observation is not None:
        observation.record_transient_dense(numel)
    new_acc = torch.zeros(numel, dtype=torch.int32)
    active = _active_lane_indices(carrier, votes)
    vote_flat = votes.detach().cpu().flatten()
    for flat_index in sorted(active):
        pre = int(carrier.reconstruct_lane(flat_index))
        vote = int(vote_flat[flat_index].item())
        post = carry_self_update_row(
            pre,
            vote,
            decay_numerator=int(spec.decay_numerator),
            decay_denominator=int(spec.decay_denominator),
        )
        post = max(
            int(spec.accumulator_clip_min),
            min(int(spec.accumulator_clip_max), int(post)),
        )
        new_acc[int(flat_index)] = int(post)
    return new_acc.view_as(q_levels)


def build_sparse_new_acc_i32_from_carrier(
    carrier: EventCodedAccLiveState,
    q_levels: torch.Tensor,
    votes: torch.Tensor,
    spec: VoteUpdateSpec,
    *,
    observation: C8StepObservation | None = None,
) -> torch.Tensor:
    numel = int(q_levels.numel())
    if observation is not None:
        observation.record_transient_dense(numel)
    new_acc = torch.zeros(numel, dtype=torch.int32)
    active_idx = _active_lane_index_tensor(carrier, votes)
    if active_idx.numel() == 0:
        return new_acc.view_as(q_levels)

    vote_flat = votes.detach().cpu().flatten().to(torch.int32)
    pre_full = carrier_pre_accumulator_i32_flat(carrier, numel)
    pre_active = pre_full[active_idx]
    vote_active = vote_flat[active_idx]

    eff_min, eff_max = effective_clip_bounds(
        DEFAULT_CARRY_WIDTH,
        VOTE_UPDATE_SOURCE_CLIP_MIN,
        VOTE_UPDATE_SOURCE_CLIP_MAX,
    )
    decay_num = int(spec.decay_numerator)
    decay_den = int(spec.decay_denominator)
    if decay_den <= 0:
        raise ValueError("decay_denominator must be > 0")
    decayed = (pre_active * decay_num) // decay_den
    post = torch.clamp(decayed + vote_active, eff_min, eff_max)
    post = torch.clamp(
        post,
        int(spec.accumulator_clip_min),
        int(spec.accumulator_clip_max),
    )
    new_acc[active_idx] = post.to(torch.int32)
    return new_acc.view_as(q_levels)


def _validate_event_coded_vote_inputs(
    q_levels: torch.Tensor,
    votes: torch.Tensor,
    spec: VoteUpdateSpec,
    *,
    validate_q_levels: bool = True,
) -> None:
    spec.validate()
    if q_levels.dtype != torch.int8:
        raise ValueError(f"q_levels must be torch.int8, got {q_levels.dtype}")
    if votes.dtype != torch.int16:
        raise ValueError(f"votes must be torch.int16, got {votes.dtype}")
    if q_levels.shape != votes.shape:
        raise ValueError("q_levels and votes must have identical shapes")
    if validate_q_levels:
        allowed = torch.tensor([-1, 0, 1], dtype=torch.int8, device=q_levels.device)
        if not bool(torch.isin(q_levels, allowed).all().item()):
            raise ValueError("q_levels must be in {-1,0,1}")


def plan_event_coded_integer_vote_update_reference(
    state: EventCodedVoteUpdateState,
    inputs: VoteUpdateInputs,
    spec: VoteUpdateSpec,
    *,
    validate_q_levels: bool = True,
    local_selection_ordering_mode: str = LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    local_selection_ordering_seed: int = 0,
    local_selection_ordering_step: int = 0,
    observation: C8StepObservation | None = None,
) -> VoteUpdatePlan:
    q_levels = state.q_levels
    votes = inputs.votes
    _validate_event_coded_vote_inputs(
        q_levels,
        votes,
        spec,
        validate_q_levels=validate_q_levels,
    )
    threshold = int(spec.threshold_abs)
    numel = int(q_levels.numel())
    max_flips = spec.max_flips(numel)

    if observation is not None:
        observation.record_flatten()
    q_i16 = q_levels.flatten().to(torch.int16)
    new_acc_i32 = build_sparse_new_acc_i32_from_carrier(
        state.carrier,
        q_levels,
        votes,
        spec,
        observation=observation,
    ).flatten()

    candidates = ((new_acc_i32 >= threshold) & (q_i16 < 1)) | (
        (new_acc_i32 <= -threshold) & (q_i16 > -1)
    )
    candidate_idx = torch.nonzero(candidates, as_tuple=False).flatten()
    pre_veto_selected = candidate_idx[:0]
    applied = candidate_idx[:0]
    applied_directions = torch.zeros_like(candidate_idx[:0], dtype=torch.int16)
    applied_thresholds = torch.zeros_like(candidate_idx[:0], dtype=torch.int32)
    replay_ce_vetoed = candidate_idx[:0]
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
        directions = torch.where(new_acc_i32[pre_veto_selected] >= threshold, 1, -1).to(
            torch.int16
        )
        (
            applied,
            applied_directions,
            applied_thresholds,
            replay_ce_vetoed,
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

    applied_count = int(applied.numel())
    stats = {
        "event_coded_live_carrier_plan": True,
        "candidate_count": int(candidate_idx.numel()),
        "pre_veto_selected_count": int(pre_veto_selected.numel()),
        "post_veto_would_apply_pre_cap_count": applied_count,
        "post_veto_applied_flip_count": applied_count,
    }
    return VoteUpdatePlan(
        q_i16=q_i16.view_as(q_levels),
        new_acc_i32=new_acc_i32.view_as(q_levels),
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


def _votes_dict_from_tensor(votes: torch.Tensor) -> dict[int, int]:
    vote_flat = votes.detach().cpu().flatten()
    return {
        int(index): int(value)
        for index, value in enumerate(vote_flat.tolist())
        if int(value) != 0
    }


def _sync_q_levels_tensor(
    carrier: EventCodedAccLiveState,
    q_levels: torch.Tensor,
) -> torch.Tensor:
    q_out = q_levels.detach().cpu().clone().to(torch.int8)
    flat = q_out.flatten()
    for flat_index, level in carrier.q_levels.items():
        flat[int(flat_index)] = int(level)
    return flat.view_as(q_levels).contiguous()


def apply_event_coded_carrier_step(
    carrier: EventCodedAccLiveState,
    *,
    votes: Mapping[int, int],
    step_index: int,
) -> None:
    carrier.apply_step(int(step_index), votes=dict(votes))


def apply_event_coded_integer_vote_update_reference(
    state: EventCodedVoteUpdateState,
    inputs: VoteUpdateInputs,
    spec: VoteUpdateSpec,
    *,
    validate_q_levels: bool = True,
    local_selection_ordering_mode: str = LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    local_selection_ordering_seed: int = 0,
    local_selection_ordering_step: int = 0,
    step_index: int = 0,
) -> EventCodedVoteUpdateResult:
    observation = C8StepObservation()
    plan = plan_event_coded_integer_vote_update_reference(
        state,
        inputs,
        spec,
        validate_q_levels=validate_q_levels,
        local_selection_ordering_mode=str(local_selection_ordering_mode),
        local_selection_ordering_seed=int(local_selection_ordering_seed),
        local_selection_ordering_step=int(local_selection_ordering_step),
        observation=observation,
    )
    carrier = copy.deepcopy(state.carrier)
    vote_map = _votes_dict_from_tensor(inputs.votes)
    apply_event_coded_carrier_step(carrier, votes=vote_map, step_index=int(step_index))

    applied_set = {int(item) for item in plan.applied_indices.detach().cpu().tolist()}
    for flat_index in applied_set:
        if flat_index not in carrier.q_levels:
            carry = int(carrier.reconstruct_lane(flat_index))
            carrier.q_levels[int(flat_index)] = 1 if carry >= 0 else -1

    q_out = _sync_q_levels_tensor(carrier, state.q_levels)
    persistent_dense = measure_persistent_dense_accumulator_materialized_numel(
        exact_accumulator_shadow=None,
        event_coded_live_carrier=carrier,
        eligible_numel=int(state.q_levels.numel()),
    )
    assert_c8_runtime_guards(
        carrier,
        observation=observation,
        persistent_dense_accumulator_materialized_numel=persistent_dense,
    )
    stats = dict(plan.stats)
    stats.update(
        c8_runtime_guard_stats(
            carrier,
            observation=observation,
            persistent_dense_accumulator_materialized_numel=persistent_dense,
        )
    )
    stats["logical_numel"] = int(state.q_levels.numel())
    if carrier.step_records:
        stats["v4_live_observed_surfaces"] = observed_surfaces_dict(carrier.step_records[-1])
    stats["flip_count"] = int(plan.applied_indices.numel())
    stats["q_changed_count"] = int((q_out != state.q_levels).sum().item())
    return EventCodedVoteUpdateResult(
        q_levels=q_out,
        carrier=carrier,
        plan=plan,
        stats=stats,
    )


def apply_event_coded_cap_mutations(
    carrier: EventCodedAccLiveState,
    q_levels: torch.Tensor,
    plan: VoteUpdatePlan,
    accepted_indices: Sequence[int],
    *,
    step_index: int,
) -> tuple[torch.Tensor, EventCodedAccLiveState]:
    """Write global-cap accepted rows through the live carrier (not dense acc_out)."""

    if not accepted_indices:
        return q_levels, carrier
    updated = copy.deepcopy(carrier)
    q_out = q_levels.detach().cpu().clone().to(torch.int8)
    q_flat = q_out.flatten()
    threshold_flat = plan.applied_thresholds.detach().cpu().flatten()
    direction_flat = plan.applied_directions.detach().cpu().flatten()
    applied_flat = plan.applied_indices.detach().cpu().flatten()
    index_by_pos = {int(idx): pos for pos, idx in enumerate(applied_flat.tolist())}
    for flat_index in accepted_indices:
        idx = int(flat_index)
        pos = index_by_pos.get(idx)
        if pos is None:
            continue
        direction = int(direction_flat[pos].item())
        q_flat[idx] = int(max(-1, min(1, int(q_flat[idx].item()) + direction)))
        updated.q_levels[idx] = int(q_flat[idx].item())
        carry = int(updated.reconstruct_lane(idx))
        residual = carry - direction * int(threshold_flat[pos].item())
        updated.hot_exact[idx] = int(residual)
    apply_event_coded_carrier_step(
        updated,
        votes={},
        step_index=int(step_index),
    )
    q_synced = _sync_q_levels_tensor(updated, q_out)
    observation = C8StepObservation()
    persistent_dense = measure_persistent_dense_accumulator_materialized_numel(
        exact_accumulator_shadow=None,
        event_coded_live_carrier=updated,
        eligible_numel=int(q_levels.numel()),
    )
    assert_c8_runtime_guards(
        updated,
        observation=observation,
        persistent_dense_accumulator_materialized_numel=persistent_dense,
    )
    return q_synced, updated


def tensor_states_use_event_coded_live_carrier(
    tensor_states: Mapping[str, Any],
) -> bool:
    return all(
        getattr(state, "event_coded_live_carrier", None) is not None
        for state in tensor_states.values()
    )
