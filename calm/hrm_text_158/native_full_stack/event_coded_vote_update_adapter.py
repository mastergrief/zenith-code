"""V4-LIVE event-coded vote-update adapter (trainer integration seam)."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
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
    _PackedHotTable,
    hot_risk_proxy_indices,
    merge_hot_table_arrays,
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
LIVE_ACC_CARRIER_W7 = "w7"
LIVE_ACC_CARRIER_W8 = "w8"
LIVE_ACC_CARRIER_NONE = "none"

C8_DENSE_ACCUMULATOR_MATERIALIZED_NUMEL_KEY = "dense_accumulator_materialized_numel"
C8_FULL_NUMEL_FLATTEN_COUNT_KEY = "full_numel_flatten_count"
C8_VOTE_UPDATE_PREPLAN_TRITON_INVOKED_KEY = "vote_update_preplan_triton_invoked"
C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY = "transient_dense_compute_numel"
C8_PERSISTENT_AUTHORITY_SCOPE_KEY = "c8_persistent_authority_scope"
C8_PERSISTENT_AUTHORITY_SCOPE_VALUE = (
    "no_dense_persistent_authority; transient planner numel-free; cap-ON densifies at "
    "global_rate_cap boundary only (separate full_sub2_runtime lane)"
)
EVENT_CODED_PLANNER_TRANSIENT_DENSE_NUMEL_KEY = "event_coded_planner_transient_dense_numel"
EVENT_CODED_CAP_BOUNDARY_DENSIFIED_KEY = "event_coded_cap_boundary_densified"
_SPARSE_CARRIER_BULK_VOTE_APPLY_MAX_EVENTS = 65_536
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
    w7_enabled: bool | None = None,
    w8_enabled: bool | None = None,
) -> str:
    from calm.hrm_text_158.native_full_stack.narrow_carrier_trainer_integration import (
        narrow_carrier_w5_enabled,
        narrow_carrier_w6_enabled,
        narrow_carrier_w7_enabled,
        narrow_carrier_w8_enabled,
    )

    use_v4 = event_coded_live_carrier_enabled(enabled=v4_enabled)
    use_w5 = narrow_carrier_w5_enabled(enabled=w5_enabled)
    use_w6 = narrow_carrier_w6_enabled(enabled=w6_enabled)
    use_w7 = narrow_carrier_w7_enabled(enabled=w7_enabled)
    use_w8 = narrow_carrier_w8_enabled(enabled=w8_enabled)
    selected = sum(int(flag) for flag in (use_v4, use_w5, use_w6, use_w7, use_w8))
    if selected > 1:
        raise ValueError(
            "V4-LIVE event-coded carrier is mutually exclusive with W5/W6/W7/W8 narrow carriers"
        )
    if use_v4:
        return LIVE_ACC_CARRIER_V4_LIVE
    if use_w5:
        return LIVE_ACC_CARRIER_W5
    if use_w6:
        return LIVE_ACC_CARRIER_W6
    if use_w7:
        return LIVE_ACC_CARRIER_W7
    if use_w8:
        return LIVE_ACC_CARRIER_W8
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
    from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
        encode_event_coded_backlog_indices,
    )

    # Use packed store bytes — do not materialize EventCodedAccEvent shells.
    packed = bytearray()
    packed += carrier._event_store.encode_bytes()
    packed += encode_event_coded_backlog_indices(tuple(sorted(carrier.backlog)))
    packed += carrier.hot_packed_bytes()
    return hashlib.sha256(bytes(packed)).hexdigest()


def c8_runtime_guard_stats(
    carrier: EventCodedAccLiveState,
    *,
    observation: C8StepObservation,
    persistent_dense_accumulator_materialized_numel: int,
    planner_transient_dense_numel: int = 0,
    include_carrier_content_sha256: bool = True,
) -> dict[str, Any]:
    stats = {
        C8_DENSE_ACCUMULATOR_MATERIALIZED_NUMEL_KEY: int(
            persistent_dense_accumulator_materialized_numel
        ),
        C8_FULL_NUMEL_FLATTEN_COUNT_KEY: int(observation.full_numel_flatten_count),
        C8_VOTE_UPDATE_PREPLAN_TRITON_INVOKED_KEY: bool(
            observation.vote_update_preplan_triton_invoked
        ),
        C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY: int(observation.transient_dense_compute_numel),
        EVENT_CODED_PLANNER_TRANSIENT_DENSE_NUMEL_KEY: int(planner_transient_dense_numel),
        C8_PERSISTENT_AUTHORITY_SCOPE_KEY: C8_PERSISTENT_AUTHORITY_SCOPE_VALUE,
        "live_authority": C8_LIVE_AUTHORITY_TAG,
    }
    if include_carrier_content_sha256:
        stats["event_coded_live_carrier_content_sha256_after"] = carrier_content_sha256(
            carrier
        )
    return stats


def assert_c8_runtime_guards(
    carrier: EventCodedAccLiveState,
    *,
    observation: C8StepObservation,
    persistent_dense_accumulator_materialized_numel: int,
) -> None:
    del carrier  # assert path checks authority flags only; content sha is site #3 stats.
    if int(persistent_dense_accumulator_materialized_numel) != 0:
        raise ValueError(
            "C8 guard failed: dense persistent accumulator authority must be 0 on V4-LIVE path"
        )
    if bool(observation.vote_update_preplan_triton_invoked):
        raise ValueError("C8 guard failed: Triton preplan forbidden on V4-LIVE path")


def hydrate_event_coded_live_carrier_from_packed(
    packed: PackedEventCodedAccState | Any,
    *,
    demotion_band: int = 1,
    cold_default: int = DEFAULT_COLD_DEFAULT,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
) -> EventCodedAccLiveState:
    from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
        decode_event_coded_backlog_indices,
        decode_hot_exact_rows,
    )
    from calm.hrm_text_158.native_full_stack.event_coded_acc_event_store import (
        EventCodedAccEventStore,
    )

    # Prefer packed event bytes; do not decode EventCodedAccEvent shells at hydrate.
    backlog = decode_event_coded_backlog_indices(
        packed.backlog_packed,
        backlog_entry_count=int(packed.backlog_entry_count),
    )
    hot_indices, hot_values = decode_hot_exact_rows(
        packed.hot_exact_packed,
        hot_exact_row_count=int(packed.hot_exact_row_count),
    )
    hot_exact = {int(i): int(v) for i, v in zip(hot_indices, hot_values)}
    events_bytes = bytes(packed.events_packed.detach().cpu().contiguous().tolist())
    return EventCodedAccLiveState(
        logical_numel=int(packed.logical_numel),
        cold_default=int(cold_default),
        threshold_abs=int(threshold_abs),
        demotion_band=int(demotion_band),
        _hot=_PackedHotTable.from_dict(hot_exact),
        events=EventCodedAccEventStore.from_packed_bytes(
            events_bytes,
            event_count=int(packed.event_count),
        ),
        backlog=set(int(item) for item in backlog),
    )


def _hot_indices_int64_tensor(carrier: EventCodedAccLiveState) -> torch.Tensor:
    return carrier.hot_lane_indices_tensor()


def pre_accumulator_i32_for_indices(
    carrier: EventCodedAccLiveState,
    indices: torch.Tensor,
) -> torch.Tensor:
    if indices.numel() == 0:
        return torch.empty(0, dtype=torch.int32)
    idx_np = indices.detach().cpu().numpy().astype(np.int64, copy=False)
    hot_idx_t = carrier.hot_lane_indices_tensor()
    hot_val_t = carrier.hot_lane_values_tensor()
    cold = int(carrier.cold_default)
    if hot_idx_t.numel() == 0:
        return torch.full((int(idx_np.size),), cold, dtype=torch.int32)
    hot_idx = hot_idx_t.detach().cpu().numpy().astype(np.int64, copy=False)
    hot_val = hot_val_t.detach().cpu().numpy().astype(np.int32, copy=False)
    pos = np.searchsorted(hot_idx, idx_np)
    in_bounds = pos < hot_idx.size
    matched = np.zeros(idx_np.shape[0], dtype=bool)
    if in_bounds.any():
        matched[in_bounds] = hot_idx[pos[in_bounds]] == idx_np[in_bounds]
    out = np.full(idx_np.shape[0], cold, dtype=np.int32)
    if matched.any():
        out[matched] = hot_val[pos[matched]]
    return torch.from_numpy(out)


def carrier_pre_accumulator_i32_flat(
    carrier: EventCodedAccLiveState,
    numel: int,
) -> torch.Tensor:
    pre_full = torch.full((int(numel),), int(carrier.cold_default), dtype=torch.int32)
    hot_keys = carrier.hot_lane_indices_tensor()
    hot_vals = carrier.hot_lane_values_tensor()
    if hot_keys.numel() > 0:
        pre_full[hot_keys] = hot_vals
    return pre_full


def _active_lane_index_tensor(
    carrier: EventCodedAccLiveState,
    votes: torch.Tensor,
    *,
    vote_active_flat_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if vote_active_flat_indices is not None:
        vote_nz = vote_active_flat_indices.detach().cpu().flatten().to(torch.int64)
    else:
        vote_flat = votes.detach().cpu().flatten()
        vote_nz = torch.nonzero(vote_flat != 0, as_tuple=False).flatten().to(torch.int64)
    hot_idx = carrier.hot_lane_indices_tensor()
    if hot_idx.numel() == 0:
        return vote_nz
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
    pre_active = pre_accumulator_i32_for_indices(carrier, active_idx)
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
        q_flat = q_levels.view(-1)
        if q_flat.numel() > 0:
            min_q = int(q_flat.min().item())
            max_q = int(q_flat.max().item())
            if min_q < -1 or max_q > 1:
                raise ValueError("q_levels must be in {-1,0,1}")


def _shape_stub_int16_votes(q_levels: torch.Tensor) -> torch.Tensor:
    """Shape-compatible int16 votes placeholder; values come from sparse events."""

    flat = torch.zeros(1, dtype=torch.int16, device=q_levels.device).expand(
        int(q_levels.numel())
    )
    return flat.view_as(q_levels)


def event_coded_cold_default_sparse_cap_unsafe(
    cold_default: int,
    spec: VoteUpdateSpec,
) -> bool:
    """True when decayed cold_default could cross threshold for some q in {-1,0,1}."""

    decayed = (int(cold_default) * int(spec.decay_numerator)) // int(
        spec.decay_denominator
    )
    decayed = max(
        int(spec.accumulator_clip_min),
        min(int(spec.accumulator_clip_max), int(decayed)),
    )
    threshold = int(spec.threshold_abs)
    for q in (-1, 0, 1):
        if (decayed >= threshold and q < 1) or (decayed <= -threshold and q > -1):
            return True
    return False


def validate_event_coded_sparse_cap_cold_default(
    carrier: EventCodedAccLiveState,
    spec: VoteUpdateSpec,
) -> None:
    if event_coded_cold_default_sparse_cap_unsafe(int(carrier.cold_default), spec):
        raise ValueError(
            "event-coded sparse global cap unsafe cold_default: decayed cold lane "
            f"can satisfy flip predicate (cold_default={int(carrier.cold_default)}, "
            f"threshold_abs={int(spec.threshold_abs)})"
        )


def _vote_active_values_at_indices(
    active_idx: torch.Tensor,
    votes: torch.Tensor,
    *,
    sparse_vote_events: Any | None = None,
) -> torch.Tensor:
    if sparse_vote_events is not None and int(sparse_vote_events.event_count()) > 0:
        sparse_idx = sparse_vote_events.indices.detach().cpu().to(torch.int64)
        sparse_val = sparse_vote_events.values.detach().cpu().to(torch.int32)
        positions = torch.searchsorted(sparse_idx, active_idx)
        in_bounds = positions < sparse_idx.numel()
        matched = torch.zeros(active_idx.numel(), dtype=torch.bool)
        if in_bounds.any():
            matched[in_bounds] = sparse_idx[positions[in_bounds]] == active_idx[in_bounds]
        vote_active = torch.zeros(active_idx.numel(), dtype=torch.int32)
        if matched.any():
            vote_active[matched] = sparse_val[positions[matched]]
        return vote_active
    vote_flat_view = votes.detach().cpu().view(-1).to(torch.int32)
    return vote_flat_view[active_idx]


def _shape_stub_q_i16(q_levels: torch.Tensor) -> torch.Tensor:
    """Shape-compatible placeholder; materialized at cap boundary when cap is enabled."""

    flat = torch.zeros(1, dtype=torch.int16, device=q_levels.device).expand(
        int(q_levels.numel())
    )
    return flat.view_as(q_levels)


def _shape_stub_new_acc_i32(q_levels: torch.Tensor) -> torch.Tensor:
    """Shape-compatible placeholder; must not be read before cap-boundary densify."""

    flat = torch.zeros(1, dtype=torch.int32, device=q_levels.device).expand(
        int(q_levels.numel())
    )
    return flat.view_as(q_levels)


def _compute_active_lane_post_acc_tensors(
    carrier: EventCodedAccLiveState,
    q_levels: torch.Tensor,
    votes: torch.Tensor,
    spec: VoteUpdateSpec,
    *,
    vote_active_flat_indices: torch.Tensor | None = None,
    sparse_vote_events: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    active_idx = _active_lane_index_tensor(
        carrier,
        votes,
        vote_active_flat_indices=vote_active_flat_indices,
    ).to(torch.int64)
    if active_idx.numel() == 0:
        return active_idx, torch.empty(0, dtype=torch.int32)

    pre_active = pre_accumulator_i32_for_indices(carrier, active_idx)
    if (
        sparse_vote_events is not None
        and vote_active_flat_indices is not None
        and int(carrier.hot_lane_indices_tensor().numel()) == 0
    ):
        vote_nz = vote_active_flat_indices.detach().cpu().to(torch.int64)
        if active_idx.numel() == vote_nz.numel() and torch.equal(active_idx, vote_nz):
            vote_active = sparse_vote_events.values.detach().cpu().to(torch.int32)
        else:
            vote_active = _vote_active_values_at_indices(
                active_idx,
                votes,
                sparse_vote_events=sparse_vote_events,
            )
    else:
        vote_active = _vote_active_values_at_indices(
            active_idx,
            votes,
            sparse_vote_events=sparse_vote_events,
        )

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
    post_active = torch.clamp(decayed + vote_active, eff_min, eff_max)
    post_active = torch.clamp(
        post_active,
        int(spec.accumulator_clip_min),
        int(spec.accumulator_clip_max),
    ).to(torch.int32)
    return active_idx, post_active


def _local_selection_order_active(
    *,
    candidate_idx: torch.Tensor,
    active_idx: torch.Tensor,
    post_active: torch.Tensor,
    numel: int,
    mode: str,
    ordering_seed: int,
    ordering_step: int,
) -> torch.Tensor:
    if candidate_idx.numel() == 0:
        return candidate_idx[:0]
    positions = torch.searchsorted(active_idx, candidate_idx.to(torch.int64))
    post_at_candidates = post_active[positions]
    normalized_mode = str(mode)
    if normalized_mode == LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX:
        abs_score = post_at_candidates.abs().to(torch.int64)
        idx64 = candidate_idx.to(torch.int64)
        composite = abs_score * (int(numel) + 1) + (int(numel) - idx64)
        return torch.argsort(composite, descending=True)
    return _local_selection_order(
        candidate_idx=candidate_idx,
        new_acc_i32=post_at_candidates,
        numel=int(numel),
        mode=normalized_mode,
        ordering_seed=int(ordering_seed),
        ordering_step=int(ordering_step),
    )


def event_coded_new_acc_values_at_device(
    plan: VoteUpdatePlan,
    flat_indices: torch.Tensor,
    device: torch.device,
    *,
    fail_closed: bool = False,
) -> torch.Tensor:
    """Sparse event-coded new_acc lookup at flat indices on the given device."""

    indices = flat_indices.detach().to(device=device, dtype=torch.int64).flatten()
    if indices.numel() == 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    if (
        plan.event_coded_sparse_active_idx is None
        or plan.event_coded_sparse_post_active_i32 is None
    ):
        if fail_closed:
            raise ValueError(
                "event-coded sparse abs_new_acc lookup requires sparse plan backing"
            )
        if plan.new_acc_i32 is None:
            return torch.zeros(indices.numel(), dtype=torch.int32, device=device)
        flat_new_acc = plan.new_acc_i32.detach().to(device=device, dtype=torch.int32).flatten()
        return flat_new_acc[indices]
    active_idx = plan.event_coded_sparse_active_idx.detach().to(
        device=device,
        dtype=torch.int64,
    )
    post_active = plan.event_coded_sparse_post_active_i32.detach().to(
        device=device,
        dtype=torch.int32,
    )
    positions = torch.searchsorted(active_idx, indices)
    in_bounds = positions < active_idx.numel()
    matched = torch.zeros(indices.numel(), dtype=torch.bool, device=device)
    if bool(in_bounds.any().item()):
        matched[in_bounds] = active_idx[positions[in_bounds]] == indices[in_bounds]
    if fail_closed and not bool(matched.all().item()):
        missing = indices[~matched].detach().cpu().tolist()
        raise ValueError(
            "event-coded sparse abs_new_acc lookup miss: flat_index not in "
            f"event_coded_sparse_active_idx (missing={missing})"
        )
    out = torch.zeros(indices.numel(), dtype=torch.int32, device=device)
    if bool(matched.any().item()):
        out[matched] = post_active[positions[matched]]
    return out


def event_coded_new_acc_values_at(
    plan: VoteUpdatePlan,
    flat_indices: torch.Tensor | Sequence[int],
    *,
    fail_closed: bool = False,
) -> torch.Tensor:
    if isinstance(flat_indices, torch.Tensor):
        indices = flat_indices.detach().cpu().to(torch.int64).flatten()
    else:
        indices = torch.tensor([int(item) for item in flat_indices], dtype=torch.int64)
    if indices.numel() == 0:
        return torch.empty(0, dtype=torch.int32)
    if (
        plan.event_coded_sparse_active_idx is not None
        and plan.event_coded_sparse_post_active_i32 is not None
    ):
        active_idx = plan.event_coded_sparse_active_idx.detach().cpu().to(torch.int64)
        post_active = plan.event_coded_sparse_post_active_i32.detach().cpu().to(torch.int32)
        positions = torch.searchsorted(active_idx, indices)
        in_bounds = positions < active_idx.numel()
        matched = torch.zeros(indices.numel(), dtype=torch.bool)
        if in_bounds.any():
            matched[in_bounds] = active_idx[positions[in_bounds]] == indices[in_bounds]
        if fail_closed and not bool(matched.all().item()):
            missing = indices[~matched].tolist()
            raise ValueError(
                "event-coded sparse abs_new_acc lookup miss: flat_index not in "
                f"event_coded_sparse_active_idx (missing={missing})"
            )
        out = torch.zeros(indices.numel(), dtype=torch.int32)
        if matched.any():
            out[matched] = post_active[positions[matched]]
        return out
    flat_new_acc = plan.new_acc_i32.detach().cpu().flatten()
    return flat_new_acc[indices.to(torch.int64)].to(torch.int32)


def event_coded_sparse_abs_new_acc_at(plan: VoteUpdatePlan, flat_index: int) -> int:
    values = event_coded_new_acc_values_at(
        plan,
        [int(flat_index)],
        fail_closed=True,
    )
    return int(values[0].abs().item())


def prepare_event_coded_plan_for_sparse_cap(
    plan: VoteUpdatePlan,
    q_levels: torch.Tensor,
) -> VoteUpdatePlan:
    """Cap-ON sparse path: shape-stub q_i16 without full-numel new_acc densify."""

    stats = dict(plan.stats)
    stats[EVENT_CODED_CAP_BOUNDARY_DENSIFIED_KEY] = False
    stats[C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY] = 0
    stats[EVENT_CODED_PLANNER_TRANSIENT_DENSE_NUMEL_KEY] = 0
    return replace(
        plan,
        q_i16=_shape_stub_q_i16(q_levels),
        stats=stats,
    )


def materialize_event_coded_plan_new_acc_for_indexing(
    plan: VoteUpdatePlan,
    q_levels: torch.Tensor,
) -> VoteUpdatePlan:
    """Emit/indexing helper: scatter sparse backing into a dense view without planner debt."""

    if (
        plan.event_coded_sparse_active_idx is None
        or plan.event_coded_sparse_post_active_i32 is None
    ):
        return plan
    numel = int(q_levels.numel())
    dense = torch.zeros(numel, dtype=torch.int32)
    active_idx = plan.event_coded_sparse_active_idx.detach().cpu().to(torch.int64)
    post_active = plan.event_coded_sparse_post_active_i32.detach().cpu().to(torch.int32)
    if active_idx.numel() > 0:
        dense[active_idx] = post_active
    return replace(plan, new_acc_i32=dense.view_as(q_levels))


def densify_new_acc_i32_at_cap_boundary(
    plan: VoteUpdatePlan,
    q_levels: torch.Tensor,
    observation: C8StepObservation | None = None,
) -> VoteUpdatePlan:
    """The only site allowed to record planner/cap transient dense numel for cap-ON."""

    obs = observation if observation is not None else C8StepObservation()
    numel = int(q_levels.numel())
    obs.record_transient_dense(numel)
    dense = torch.zeros(numel, dtype=torch.int32)
    q_i16 = q_levels.detach().cpu().flatten().to(torch.int16).contiguous().view_as(q_levels)
    if (
        plan.event_coded_sparse_active_idx is not None
        and plan.event_coded_sparse_post_active_i32 is not None
        and plan.event_coded_sparse_active_idx.numel() > 0
    ):
        active_idx = plan.event_coded_sparse_active_idx.detach().cpu().to(torch.int64)
        post_active = plan.event_coded_sparse_post_active_i32.detach().cpu().to(torch.int32)
        dense[active_idx] = post_active
    stats = dict(plan.stats)
    stats[EVENT_CODED_CAP_BOUNDARY_DENSIFIED_KEY] = True
    stats[C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY] = int(obs.transient_dense_compute_numel)
    stats[EVENT_CODED_PLANNER_TRANSIENT_DENSE_NUMEL_KEY] = 0
    return replace(
        plan,
        q_i16=q_i16,
        new_acc_i32=dense.view_as(q_levels),
        stats=stats,
    )


def plan_event_coded_integer_vote_update_dense_oracle(
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
    return plan_event_coded_integer_vote_update_dense_oracle(
        state,
        inputs,
        spec,
        validate_q_levels=validate_q_levels,
        local_selection_ordering_mode=str(local_selection_ordering_mode),
        local_selection_ordering_seed=int(local_selection_ordering_seed),
        local_selection_ordering_step=int(local_selection_ordering_step),
        observation=observation,
    )


def plan_event_coded_integer_vote_update(
    state: EventCodedVoteUpdateState,
    inputs: VoteUpdateInputs,
    spec: VoteUpdateSpec,
    *,
    validate_q_levels: bool = True,
    local_selection_ordering_mode: str = LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    local_selection_ordering_seed: int = 0,
    local_selection_ordering_step: int = 0,
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

    active_idx, post_active = _compute_active_lane_post_acc_tensors(
        state.carrier,
        q_levels,
        votes,
        spec,
        vote_active_flat_indices=inputs.vote_active_flat_indices,
        sparse_vote_events=inputs.sparse_vote_events,
    )

    candidate_idx = active_idx[:0]
    pre_veto_selected = candidate_idx[:0]
    applied = candidate_idx[:0]
    applied_directions = torch.zeros_like(candidate_idx[:0], dtype=torch.int16)
    applied_thresholds = torch.zeros_like(candidate_idx[:0], dtype=torch.int32)
    replay_ce_vetoed = candidate_idx[:0]
    replay_veto_directions = torch.zeros_like(candidate_idx[:0], dtype=torch.int16)
    replay_veto_thresholds = torch.zeros_like(candidate_idx[:0], dtype=torch.int32)
    pc_aux_negative = candidate_idx[:0]
    pc_aux_vetoed = candidate_idx[:0]

    if active_idx.numel() > 0:
        q_active = q_levels.detach().cpu().view(-1)[active_idx].to(torch.int16)
        candidate_mask = ((post_active >= threshold) & (q_active < 1)) | (
            (post_active <= -threshold) & (q_active > -1)
        )
        candidate_idx = active_idx[candidate_mask]
        if candidate_idx.numel() > 0 and max_flips > 0:
            order = _local_selection_order_active(
                candidate_idx=candidate_idx,
                active_idx=active_idx,
                post_active=post_active,
                numel=numel,
                mode=str(local_selection_ordering_mode),
                ordering_seed=int(local_selection_ordering_seed),
                ordering_step=int(local_selection_ordering_step),
            )
            pre_veto_selected = candidate_idx[order[:max_flips]]
            selected_thresholds = torch.full_like(pre_veto_selected, threshold, dtype=torch.int32)
            positions = torch.searchsorted(active_idx, pre_veto_selected.to(torch.int64))
            post_selected = post_active[positions]
            directions = torch.where(post_selected >= threshold, 1, -1).to(torch.int16)
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
    stats: dict[str, int | float | bool | str] = {
        "event_coded_live_carrier_plan": True,
        "candidate_count": int(candidate_idx.numel()),
        "pre_veto_selected_count": int(pre_veto_selected.numel()),
        "post_veto_would_apply_pre_cap_count": applied_count,
        "post_veto_applied_flip_count": applied_count,
        EVENT_CODED_PLANNER_TRANSIENT_DENSE_NUMEL_KEY: 0,
        EVENT_CODED_CAP_BOUNDARY_DENSIFIED_KEY: False,
    }
    sparse_active = (
        active_idx.detach().cpu().clone().contiguous()
        if active_idx.numel() > 0
        else torch.empty(0, dtype=torch.int64)
    )
    sparse_post = (
        post_active.detach().cpu().clone().contiguous()
        if post_active.numel() > 0
        else torch.empty(0, dtype=torch.int32)
    )
    return VoteUpdatePlan(
        q_i16=_shape_stub_q_i16(q_levels),
        new_acc_i32=_shape_stub_new_acc_i32(q_levels),
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
        event_coded_sparse_active_idx=sparse_active,
        event_coded_sparse_post_active_i32=sparse_post,
    )


def _votes_dict_from_tensor(votes: torch.Tensor) -> dict[int, int]:
    vote_flat = votes.detach().cpu().flatten()
    vote_nz = torch.nonzero(vote_flat != 0, as_tuple=False).flatten()
    if vote_nz.numel() == 0:
        return {}
    indices = vote_nz.to(torch.int64).tolist()
    values = vote_flat[vote_nz].tolist()
    return {int(index): int(value) for index, value in zip(indices, values)}


# Slice-10 C2b_app classify site IDs (cpu_reference only). Survival fail-closed:
# every NEW ID below MUST remain as a literal in this module. Do NOT bank OWNER on
# legacy C4.S1c_clone / C4.S1c_contig for Slice-10.
SLICE10_VOTE_FIRST_SYNC_PREFIX = "C2b.S1_vote_first_sync"
SLICE10_CAP_MUT_SYNC_PREFIX = "C2b.S1_cap_mut_sync"
SLICE10_CAP_MUT_Q_CLONE_SITE_ID = "C2b.S1_cap_mut_q_clone"
SLICE10_C2B_APP_CLASSIFY_SITE_IDS: tuple[str, ...] = (
    "C4.S1a",  # existing cow — activated by GAP A
    "C2b.S1_vote_first_sync_clone",
    "C2b.S1_vote_first_sync_hot_list",
    "C2b.S1_vote_first_sync_contig",
    "C2b.S1_cap_mut_q_clone",
    "C2b.S1_cap_mut_sync_clone",
    "C2b.S1_cap_mut_sync_hot_list",
    "C2b.S1_cap_mut_sync_contig",
)
SLICE10_C2B_APP_FORBIDDEN_OWNER_SITE_IDS: tuple[str, ...] = (
    "C4.S1c_clone",
    "C4.S1c_contig",
)


def _sync_q_levels_tensor(
    carrier: EventCodedAccLiveState,
    q_levels: torch.Tensor,
    *,
    host_allocator_site_emit: Callable[..., None] | None = None,
    site_emit_enabled: bool = False,
    optimizer_step_index: int | None = None,
    state_index: int | None = None,
    classify_site_prefix: str | None = None,
) -> torch.Tensor:
    """Sync carrier hot q into a dense tensor.

    When ``classify_site_prefix`` is set (Slice-10 C2b_app classify), emit distinct
    C2b-scoped site IDs for clone / hot_list / contig. When unset, keep legacy
    ``C4.S1c_clone`` / ``C4.S1c_contig`` marks (gpu / non-classify callers).
    """

    def _site(site_id: str, suffix: str, line: int) -> None:
        if host_allocator_site_emit is None or not site_emit_enabled:
            return
        host_allocator_site_emit(
            site_id,
            suffix,
            origin_file="event_coded_vote_update_adapter.py",
            origin_line=int(line),
            optimizer_step_index=int(optimizer_step_index or 0),
            state_index=int(state_index if state_index is not None else -1),
        )

    if classify_site_prefix:
        clone_id = f"{classify_site_prefix}_clone"
        hot_id = f"{classify_site_prefix}_hot_list"
        contig_id = f"{classify_site_prefix}_contig"
    else:
        clone_id = "C4.S1c_clone"
        hot_id = None  # legacy path has no dedicated hot-list mark
        contig_id = "C4.S1c_contig"

    _site(clone_id, "pre", 1021)
    q_out = q_levels.detach().clone().to(torch.int8)
    _site(clone_id, "post", 1021)
    if not carrier.q_levels:
        _site(contig_id, "pre", 1023)
        result = q_out.contiguous()
        _site(contig_id, "post", 1023)
        return result
    if hot_id is not None:
        _site(hot_id, "pre", 1066)
    hot_items = sorted(carrier.q_levels.items(), key=lambda item: int(item[0]))
    flat_indices = torch.tensor(
        [int(index) for index, _ in hot_items],
        device=q_out.device,
        dtype=torch.int64,
    )
    flat_values = torch.tensor(
        [int(level) for _, level in hot_items],
        device=q_out.device,
        dtype=torch.int8,
    )
    if hot_id is not None:
        _site(hot_id, "post", 1076)
    flat = q_out.flatten()
    flat.index_put_((flat_indices,), flat_values)
    _site(contig_id, "pre", 1037)
    result = flat.view_as(q_levels).contiguous()
    _site(contig_id, "post", 1037)
    return result


def apply_event_coded_carrier_step(
    carrier: EventCodedAccLiveState,
    *,
    votes: Mapping[int, int] | None = None,
    sparse_vote_events: Any | None = None,
    step_index: int,
    host_allocator_site_emit: Callable[..., None] | None = None,
    site_emit_enabled: bool = False,
    s1d7_band_counter_emit: Callable[..., None] | None = None,
    optimizer_step_index: int | None = None,
    state_index: int | None = None,
) -> None:
    def _emit_site(site_id: str, suffix: str, line: int) -> None:
        if host_allocator_site_emit is None or not site_emit_enabled:
            return
        host_allocator_site_emit(
            site_id,
            suffix,
            origin_file="event_coded_vote_update_adapter.py",
            origin_line=int(line),
            optimizer_step_index=int(
                optimizer_step_index if optimizer_step_index is not None else step_index
            ),
            state_index=int(state_index if state_index is not None else -1),
        )

    emit_kwargs = {
        "host_allocator_site_emit": host_allocator_site_emit,
        "site_emit_enabled": site_emit_enabled,
        "s1d7_band_counter_emit": s1d7_band_counter_emit,
        "optimizer_step_index": optimizer_step_index,
        "state_index": state_index,
    }
    if sparse_vote_events is not None:
        from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents

        if not isinstance(sparse_vote_events, SparseVoteEvents):
            sparse_vote_events = SparseVoteEvents.from_dict(sparse_vote_events)
        _emit_site("C4.S1d.1", "pre", 1077)
        sparse_indices = sparse_vote_events.indices.detach().cpu().numpy()
        sparse_values = sparse_vote_events.values.detach().cpu().numpy()
        _emit_site("C4.S1d.1", "post", 1081)
        carrier.apply_step(
            int(step_index),
            sparse_vote_indices=sparse_indices,
            sparse_vote_values=sparse_values,
            **emit_kwargs,
        )
        return
    _emit_site("C4.S1d.1", "pre", 1087)
    _emit_site("C4.S1d.1", "post", 1087)
    carrier.apply_step(int(step_index), votes=dict(votes or {}), **emit_kwargs)


def apply_event_coded_integer_vote_update_from_plan(
    state: EventCodedVoteUpdateState,
    inputs: VoteUpdateInputs,
    spec: VoteUpdateSpec,
    plan: VoteUpdatePlan,
    *,
    validate_q_levels: bool = True,
    step_index: int = 0,
    cap_boundary_transient_dense: int = 0,
    lightweight_runtime_stats: bool = False,
    host_allocator_site_emit: Callable[..., None] | None = None,
    optimizer_step_index: int | None = None,
    state_index: int | None = None,
    site_emit_enabled: bool = False,
    classify_site_prefix: str | None = None,
) -> EventCodedVoteUpdateResult:
    def _site(site_id: str, suffix: str, line: int) -> None:
        if host_allocator_site_emit is None or not site_emit_enabled:
            return
        host_allocator_site_emit(
            site_id,
            suffix,
            origin_file="event_coded_vote_update_adapter.py",
            origin_line=int(line),
            optimizer_step_index=int(optimizer_step_index if optimizer_step_index is not None else step_index),
            state_index=int(state_index if state_index is not None else -1),
        )

    _validate_event_coded_vote_inputs(
        state.q_levels,
        inputs.votes,
        spec,
        validate_q_levels=validate_q_levels,
    )
    observation = C8StepObservation()
    observation.record_flatten()
    observation.transient_dense_compute_numel = int(cap_boundary_transient_dense)
    _site("C4.S1a", "pre", 1081)
    carrier = state.carrier.cow_copy()
    _site("C4.S1a", "post", 1081)
    vote_map = (
        inputs.sparse_vote_events
        if inputs.sparse_vote_events is not None
        else _votes_dict_from_tensor(inputs.votes)
    )
    _site("C4.S1d", "pre", 1087)
    carrier_emit_kwargs = {
        "host_allocator_site_emit": host_allocator_site_emit,
        "site_emit_enabled": site_emit_enabled,
        "optimizer_step_index": optimizer_step_index,
        "state_index": state_index,
    }
    if inputs.sparse_vote_events is not None:
        apply_event_coded_carrier_step(
            carrier,
            sparse_vote_events=inputs.sparse_vote_events,
            step_index=int(step_index),
            **carrier_emit_kwargs,
        )
    else:
        apply_event_coded_carrier_step(
            carrier,
            votes=vote_map,
            step_index=int(step_index),
            **carrier_emit_kwargs,
        )
    _site("C4.S1d", "post", 1098)

    _site("C4.S1b", "pre", 1100)
    applied_set = {int(item) for item in plan.applied_indices.detach().cpu().tolist()}
    for flat_index in applied_set:
        if flat_index not in carrier.q_levels:
            carry = int(carrier.reconstruct_lane(flat_index))
            carrier.q_levels[int(flat_index)] = 1 if carry >= 0 else -1
    _site("C4.S1b", "post", 1104)

    q_out = _sync_q_levels_tensor(
        carrier,
        state.q_levels,
        host_allocator_site_emit=host_allocator_site_emit,
        site_emit_enabled=site_emit_enabled,
        optimizer_step_index=optimizer_step_index if optimizer_step_index is not None else step_index,
        state_index=state_index,
        # Only set by cpu_reference vote_and_cap (Slice-10); GPU seam leaves None → legacy C4.S1c_*.
        classify_site_prefix=classify_site_prefix,
    )
    persistent_dense = measure_persistent_dense_accumulator_materialized_numel(
        exact_accumulator_shadow=None,
        event_coded_live_carrier=carrier,
        eligible_numel=int(state.q_levels.numel()),
    )
    _site("C4.S1e", "pre", 1112)
    if not bool(lightweight_runtime_stats):
        assert_c8_runtime_guards(
            carrier,
            observation=observation,
            persistent_dense_accumulator_materialized_numel=persistent_dense,
        )
    _site("C4.S1e", "post", 1117)
    planner_transient = int(
        plan.stats.get(EVENT_CODED_PLANNER_TRANSIENT_DENSE_NUMEL_KEY, 0)
    )
    stats = {
        key: value
        for key, value in plan.stats.items()
        if not isinstance(value, torch.Tensor)
    }
    _site("C4.S1f", "pre", 1126)
    stats.update(
        c8_runtime_guard_stats(
            carrier,
            observation=observation,
            persistent_dense_accumulator_materialized_numel=persistent_dense,
            planner_transient_dense_numel=planner_transient,
            include_carrier_content_sha256=not bool(lightweight_runtime_stats),
        )
    )
    stats["logical_numel"] = int(state.q_levels.numel())
    _site("C4.S1f.2", "pre", 1193)
    if carrier.step_records:
        stats["v4_live_observed_surfaces"] = observed_surfaces_dict(carrier.step_records[-1])
    _site("C4.S1f.2", "post", 1194)
    stats["flip_count"] = int(plan.applied_indices.numel())
    _site("C4.S1f.1", "pre", 1195)
    stats["q_changed_count"] = int((q_out != state.q_levels).sum().item())
    _site("C4.S1f.1", "post", 1197)
    _site("C4.S1f", "post", 1139)
    return EventCodedVoteUpdateResult(
        q_levels=q_out,
        carrier=carrier,
        plan=plan,
        stats=stats,
    )


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
    plan = plan_event_coded_integer_vote_update_dense_oracle(
        state,
        inputs,
        spec,
        validate_q_levels=validate_q_levels,
        local_selection_ordering_mode=str(local_selection_ordering_mode),
        local_selection_ordering_seed=int(local_selection_ordering_seed),
        local_selection_ordering_step=int(local_selection_ordering_step),
        observation=observation,
    )
    carrier = state.carrier.cow_copy()
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
    stats = {
        key: value
        for key, value in plan.stats.items()
        if not isinstance(value, torch.Tensor)
    }
    stats.update(
        c8_runtime_guard_stats(
            carrier,
            observation=observation,
            persistent_dense_accumulator_materialized_numel=persistent_dense,
            planner_transient_dense_numel=int(
                plan.stats.get(EVENT_CODED_PLANNER_TRANSIENT_DENSE_NUMEL_KEY, 0)
            )
            if EVENT_CODED_PLANNER_TRANSIENT_DENSE_NUMEL_KEY in plan.stats
            else int(observation.transient_dense_compute_numel),
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


def _apply_cap_hot_and_q_writes_in_place(
    carrier: EventCodedAccLiveState,
    q_out: torch.Tensor,
    plan: VoteUpdatePlan,
    accepted_indices: Sequence[int],
) -> torch.Tensor:
    """Mutate vote-fork carrier q/hot for accepted cap rows (no cow_copy)."""
    q_flat = q_out.flatten()
    threshold_flat = plan.applied_thresholds.detach().cpu().flatten()
    direction_flat = plan.applied_directions.detach().cpu().flatten()
    applied_flat = plan.applied_indices.detach().cpu().flatten()
    index_by_pos = {int(idx): pos for pos, idx in enumerate(applied_flat.tolist())}
    cap_indices: list[int] = []
    cap_values: list[int] = []
    for flat_index in accepted_indices:
        idx = int(flat_index)
        pos = index_by_pos.get(idx)
        if pos is None:
            continue
        direction = int(direction_flat[pos].item())
        q_flat[idx] = int(max(-1, min(1, int(q_flat[idx].item()) + direction)))
        carrier.q_levels[idx] = int(q_flat[idx].item())
        carry = int(carrier.reconstruct_lane(idx))
        residual = carry - direction * int(threshold_flat[pos].item())
        cap_indices.append(idx)
        cap_values.append(int(residual))
    if cap_indices:
        cap_idx = np.array(cap_indices, dtype=np.int32)
        cap_val = np.array(cap_values, dtype=np.int16)
        hot_idx, hot_val = merge_hot_table_arrays(
            carrier._hot.indices_array(),
            carrier._hot.values_array(),
            np.empty(0, dtype=np.int32),
            cap_idx,
            cap_val,
        )
        carrier._hot.replace_arrays(hot_idx, hot_val)
        carrier._invalidate_packed_caches()
    return q_out


def apply_event_coded_cap_mutations(
    carrier: EventCodedAccLiveState,
    q_levels: torch.Tensor,
    plan: VoteUpdatePlan,
    accepted_indices: Sequence[int],
    *,
    step_index: int,
    host_allocator_site_emit: Callable[..., None] | None = None,
    site_emit_enabled: bool = False,
    optimizer_step_index: int | None = None,
    state_index: int | None = None,
) -> tuple[torch.Tensor, EventCodedAccLiveState]:
    """Write global-cap accepted rows through the live carrier (not dense acc_out).

    Mutates ``carrier`` in-place; input must already be the vote-fork (L1156).
    """

    def _site(site_id: str, suffix: str, line: int) -> None:
        if host_allocator_site_emit is None or not site_emit_enabled:
            return
        host_allocator_site_emit(
            site_id,
            suffix,
            origin_file="event_coded_vote_update_adapter.py",
            origin_line=int(line),
            optimizer_step_index=int(
                optimizer_step_index if optimizer_step_index is not None else step_index
            ),
            state_index=int(state_index if state_index is not None else -1),
        )

    if not accepted_indices:
        return q_levels, carrier
    _site(SLICE10_CAP_MUT_Q_CLONE_SITE_ID, "pre", 1400)
    q_out = q_levels.detach().cpu().clone().to(torch.int8)
    _site(SLICE10_CAP_MUT_Q_CLONE_SITE_ID, "post", 1400)
    _apply_cap_hot_and_q_writes_in_place(carrier, q_out, plan, accepted_indices)
    apply_event_coded_carrier_step(
        carrier,
        votes={},
        step_index=int(step_index),
    )
    q_synced = _sync_q_levels_tensor(
        carrier,
        q_out,
        host_allocator_site_emit=host_allocator_site_emit,
        site_emit_enabled=site_emit_enabled,
        optimizer_step_index=optimizer_step_index if optimizer_step_index is not None else step_index,
        state_index=state_index,
        classify_site_prefix=(
            SLICE10_CAP_MUT_SYNC_PREFIX if site_emit_enabled else None
        ),
    )
    observation = C8StepObservation()
    persistent_dense = measure_persistent_dense_accumulator_materialized_numel(
        exact_accumulator_shadow=None,
        event_coded_live_carrier=carrier,
        eligible_numel=int(q_levels.numel()),
    )
    assert_c8_runtime_guards(
        carrier,
        observation=observation,
        persistent_dense_accumulator_materialized_numel=persistent_dense,
    )
    return q_synced, carrier


def apply_event_coded_vote_and_cap_from_plan(
    state: EventCodedVoteUpdateState,
    inputs: VoteUpdateInputs,
    spec: VoteUpdateSpec,
    plan: VoteUpdatePlan,
    accepted_indices: Sequence[int],
    *,
    validate_q_levels: bool = True,
    step_index: int = 0,
    cap_boundary_transient_dense: int = 0,
    lightweight_runtime_stats: bool = False,
    host_allocator_site_emit: Callable[..., None] | None = None,
    optimizer_step_index: int | None = None,
    state_index: int | None = None,
    site_emit_enabled: bool = False,
) -> EventCodedVoteUpdateResult:
    """Single vote-fork + vote apply + in-place cap on the same carrier."""

    vote_result = apply_event_coded_integer_vote_update_from_plan(
        state,
        inputs,
        spec,
        plan,
        validate_q_levels=validate_q_levels,
        step_index=int(step_index),
        cap_boundary_transient_dense=int(cap_boundary_transient_dense),
        lightweight_runtime_stats=bool(lightweight_runtime_stats),
        host_allocator_site_emit=host_allocator_site_emit,
        optimizer_step_index=optimizer_step_index,
        state_index=state_index,
        site_emit_enabled=site_emit_enabled,
        classify_site_prefix=(
            SLICE10_VOTE_FIRST_SYNC_PREFIX if site_emit_enabled else None
        ),
    )
    if not accepted_indices:
        return vote_result
    q_out, carrier = apply_event_coded_cap_mutations(
        vote_result.carrier,
        vote_result.q_levels,
        plan,
        accepted_indices,
        step_index=int(step_index),
        host_allocator_site_emit=host_allocator_site_emit,
        site_emit_enabled=site_emit_enabled,
        optimizer_step_index=optimizer_step_index,
        state_index=state_index,
    )
    stats = dict(vote_result.stats)
    if carrier.step_records:
        stats["v4_live_observed_surfaces"] = observed_surfaces_dict(carrier.step_records[-1])
    stats["q_changed_count"] = int((q_out != state.q_levels).sum().item())
    return EventCodedVoteUpdateResult(
        q_levels=q_out,
        carrier=carrier,
        plan=plan,
        stats=stats,
    )


def tensor_states_use_event_coded_live_carrier(
    tensor_states: Mapping[str, Any],
) -> bool:
    return all(
        getattr(state, "event_coded_live_carrier", None) is not None
        for state in tensor_states.values()
    )
