"""W6/T=10 grad-proxy audit (M-A) and production ingress helpers."""
from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import torch

from calm.hrm_text_158 import LMHead
from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
    VoteUpdateInputs,
    VoteUpdateSpec,
    make_live_shadow_tensor_state,
)
from calm.hrm_text_158.native_full_stack.oracle_screen_runner import (
    B2B_CANDIDATE_APPLY_POLICY,
    ORACLE_SCREEN_ORDERING_SEED,
    ORACLE_SCREEN_ORDERING_STEP,
    _apply_full_vote_planned_candidate_shadow_update,
    _audit_sparse_singleton_identity_for_candidate,
    _candidate_delta_weight_from_one_flip,
    _candidate_id,
    _compute_activation_credit_candidate_proxies,
    _evaluate_loss,
    _ordered_candidate_indices,
    _sample_candidate_ids,
    _sign_int,
    _single_flip_spec,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_falsifier_battery import (
    kendall_tau_b,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    effective_clip_w6,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS,
)
from calm.hrm_text_158.native_full_stack.two_tier_transient_selection import (
    crossing_eligible_flat_indices,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    LOCAL_SELECTION_ORDER_DETERMINISTIC_HASH_MATCHED,
    VoteUpdateState,
    _local_selection_order,
    apply_integer_vote_update_reference,
    plan_integer_vote_update_reference,
)

_STRICT_SAMPLE_SORT_INT64_MAX = (1 << 63) - 1

GRAD_PROXY_AUDIT_SCHEMA = "hrm_text_158_grad_proxy_audit_receipt/v0"
GRAD_PROXY_AUDIT_ESTIMAND = "first_order_grad_proxy_weighted_local_loss_delta"
GRAD_PROXY_AUDIT_COMPARATOR_SPEC = "threshold_abs_10_w6_carry"
GRAD_PROXY_AUDIT_RECEIPT_NAME = "grad_proxy_audit_receipt.json"
GRAD_PROXY_AUDIT_ABORT_NAME = "grad_proxy_audit_abort.json"
GRAD_PROXY_AUDIT_ABORT_REASON = "no_crossings_within_warmup_cap"
GRAD_PROXY_AUDIT_STATE_SOURCE = "probe_warmup_pre_apply"
DEFAULT_WARMUP_MAX_STEPS = 8
MAX_WARMUP_MAX_STEPS = 8
POPULATION_MODE_SAMPLED_K64 = "sampled_k64"
POPULATION_MODE_FULL_CROSSING_ELIGIBLE = "full_crossing_eligible"
DRIFT_AUDIT_SAMPLE_COUNT = 8
DRIFT_AUDIT_STEP_INTERVAL = 5
GRAD_PROXY_VECTORIZED_DELTA_CHUNK = 64
ACTIVATION_CREDIT_GATHER_TELEMETRY_NOTE = (
    "activation_credit_gather nested phase threads optimizer_step_index when provided; "
    "M1 attempt #6 showed gather PROG lines absent for steps 1-6 and present from step 7 "
    "onward — consistent with late-starting full-population gather work rather than a "
    "hardcoded step counter (fixed in Stage-1 instrumentation)."
)


class GradProxyAuditAborted(RuntimeError):
    """Fail-closed abort for non-comparable grad-proxy audit receipts."""


class GradProxyAuditWarmupCapAborted(GradProxyAuditAborted):
    """Fail-closed abort when no W6 crossings appear within the warm-up cap."""

    reason = GRAD_PROXY_AUDIT_ABORT_REASON

    def __init__(
        self,
        *,
        crossing_eligible_count_by_step: Sequence[int],
        warmup_steps_run: int,
        launch_sha: str,
        parent_sha256: str | None = None,
    ) -> None:
        super().__init__(
            "grad_proxy_audit_aborted: no_crossings_within_warmup_cap"
        )
        self.crossing_eligible_count_by_step = [
            int(value) for value in crossing_eligible_count_by_step
        ]
        self.warmup_steps_run = int(warmup_steps_run)
        self.launch_sha = str(launch_sha)
        self.parent_sha256 = (
            None if parent_sha256 is None else str(parent_sha256)
        )


class UniverseBuildMeasurementAborted(RuntimeError):
    """Bounded measurement abort with partial universe-build timing (slice-0 telemetry)."""

    def __init__(
        self,
        *,
        reason: str,
        timing_out: Mapping[str, float],
        partial_crossing_eligible_count: int,
        partial_candidate_count: int,
    ) -> None:
        super().__init__(f"universe_build_measurement_aborted: {reason}")
        self.reason = str(reason)
        self.timing_out = {str(key): float(value) for key, value in timing_out.items()}
        self.partial_crossing_eligible_count = int(partial_crossing_eligible_count)
        self.partial_candidate_count = int(partial_candidate_count)


def w6_t10_base_spec(*, max_abs_per_tensor: int) -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=int(CROSSING_THRESHOLD_ABS),
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=int(max_abs_per_tensor),
    )


def materialize_selector_rows(
    *,
    votes: torch.Tensor,
    state: BoundedDeltaTensorState,
) -> list[dict[str, Any]]:
    vote_state = state.vote_update_state()
    q_levels = vote_state.q_levels.flatten()
    accumulators = vote_state.accumulators.flatten()
    vote_flat = votes.flatten()
    return [
        {
            "flat_index": int(flat_index),
            "vote_value": int(vote_flat[flat_index].item()),
            "pre_accumulator_i16": int(accumulators[flat_index].item()),
            "current_q_level": int(q_levels[flat_index].item()),
        }
        for flat_index in range(int(q_levels.numel()))
    ]


def _materialize_w6_t10_candidate_dict_entry(
    *,
    state_key: str,
    flat_index: int,
    current_rank_position: int,
    deterministic_hash_rank_position: int,
    vote_flat: torch.Tensor,
    q_flat: torch.Tensor,
    acc_flat: torch.Tensor,
    new_acc: torch.Tensor,
) -> dict[str, Any]:
    new_acc_signed = int(new_acc[int(flat_index)].item())
    candidate_id = _candidate_id(state_key, int(flat_index))
    return {
        "candidate_id": candidate_id,
        "state_key": state_key,
        "flat_index": int(flat_index),
        "vote_value": int(vote_flat[int(flat_index)].item()),
        "current_rank_position": int(current_rank_position),
        "deterministic_hash_rank_position": int(deterministic_hash_rank_position),
        "current_q_level": int(q_flat[int(flat_index)].item()),
        "pre_accumulator_i16": int(acc_flat[int(flat_index)].item()),
        "new_acc_i32_signed": new_acc_signed,
        "proposal_direction": _sign_int(new_acc_signed),
    }


def _canonical_state_key_ord_map(
    tensor_states: Mapping[str, BoundedDeltaTensorState],
) -> tuple[list[str], dict[str, int]]:
    sorted_keys = sorted(tensor_states.keys())
    return sorted_keys, {str(key): int(index) for index, key in enumerate(sorted_keys)}


def _sampled_universe_global_sort_weights(
    *,
    state_count: int,
    max_flat_index: int,
    max_rank: int,
) -> tuple[int, int]:
    w_flat = int(max_flat_index) + 1
    max_state_ord = max(0, int(state_count) - 1)
    w_state = int(state_count) * w_flat
    if w_state <= max_state_ord * w_flat:
        raise RuntimeError(
            "sampled_universe_sort_weight_invariant_failed: "
            f"w_state={w_state} must exceed max_state_ord*w_flat="
            f"{max_state_ord * w_flat}"
        )
    max_composite = (
        int(max_rank) * int(w_state)
        + int(max_state_ord) * int(w_flat)
        + int(max_flat_index)
    )
    if max_composite > _STRICT_SAMPLE_SORT_INT64_MAX:
        raise RuntimeError(
            "sampled_universe_sort_key_int64_overflow: "
            f"max_composite={max_composite} exceeds int64"
        )
    return w_flat, w_state


def _packed_crossing_sort_key(
    rank: torch.Tensor,
    state_key_ord: torch.Tensor,
    flat_index: torch.Tensor,
    *,
    w_state: int,
    w_flat: int,
) -> torch.Tensor:
    rank_i = rank.to(torch.int64)
    ord_i = state_key_ord.to(torch.int64)
    flat_i = flat_index.to(torch.int64)
    composite = rank_i * int(w_state) + ord_i * int(w_flat) + flat_i
    composite_max = int(composite.max().item()) if composite.numel() else 0
    composite_min = int(composite.min().item()) if composite.numel() else 0
    if composite_max > _STRICT_SAMPLE_SORT_INT64_MAX or composite_min < -(1 << 63):
        raise RuntimeError(
            "sampled_universe_sort_key_int64_overflow: "
            f"composite range [{composite_min}, {composite_max}]"
        )
    return composite


def _vote_plan_rank_and_new_acc_at_crossings(
    *,
    vote_state: VoteUpdateState,
    votes: torch.Tensor,
    spec: VoteUpdateSpec,
    crossing_flat: torch.Tensor,
    ordering_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    plan = plan_integer_vote_update_reference(
        vote_state,
        VoteUpdateInputs(votes=votes),
        spec,
        local_selection_ordering_mode=str(ordering_mode),
        local_selection_ordering_seed=int(ORACLE_SCREEN_ORDERING_SEED),
        local_selection_ordering_step=int(ORACLE_SCREEN_ORDERING_STEP),
    )
    candidate_idx = plan.candidate_indices.detach().cpu().to(torch.int64)
    new_acc = plan.new_acc_i32.detach().cpu().to(torch.int32).flatten()
    numel = int(vote_state.q_levels.numel())
    crossing_for_sort = crossing_flat[
        torch.isin(crossing_flat, candidate_idx)
    ].to(torch.int64)
    if candidate_idx.numel() == 0 or int(crossing_for_sort.numel()) == 0:
        return new_acc, torch.empty(0, dtype=torch.int64), crossing_for_sort, candidate_idx
    order = _local_selection_order(
        candidate_idx=candidate_idx,
        new_acc_i32=new_acc,
        numel=numel,
        mode=str(ordering_mode),
        ordering_seed=int(ORACLE_SCREEN_ORDERING_SEED),
        ordering_step=int(ORACLE_SCREEN_ORDERING_STEP),
    )
    ordered = candidate_idx[order]
    rank_full = torch.full((numel,), -1, dtype=torch.int64)
    rank_full[ordered] = torch.arange(int(ordered.numel()), dtype=torch.int64)
    return new_acc, rank_full[crossing_for_sort], crossing_for_sort, candidate_idx


def _sample_crossing_row_indices_interleaved(
    current_perm: torch.Tensor,
    det_perm: torch.Tensor,
    *,
    max_sampled_candidates: int,
) -> list[int]:
    if int(max_sampled_candidates) <= 0:
        return []
    n_current = int(current_perm.numel())
    n_det = int(det_perm.numel())
    if n_current <= int(max_sampled_candidates):
        return [int(row) for row in current_perm.detach().cpu().tolist()]
    sampled: list[int] = []
    seen: set[int] = set()
    max_len = max(n_current, n_det)
    for index in range(max_len):
        for source in (current_perm, det_perm):
            if index >= int(source.numel()):
                continue
            row = int(source[index].item())
            if row in seen:
                continue
            seen.add(row)
            sampled.append(row)
            if len(sampled) >= int(max_sampled_candidates):
                return sampled
    return sampled


def _strict_sampled_universe_materialization_counters(
    *,
    full_python_candidate_id_count: int = 0,
    full_sort_key_entry_count: int = 0,
    full_entry_by_id_count: int = 0,
    crossing_mask_tolist_count: int = 0,
    rank_ties_possible: bool = True,
    bounded_hash_ordering_residual: bool = True,
) -> dict[str, int | bool]:
    return {
        "full_python_candidate_id_count": int(full_python_candidate_id_count),
        "full_sort_key_entry_count": int(full_sort_key_entry_count),
        "full_entry_by_id_count": int(full_entry_by_id_count),
        "crossing_mask_tolist_count": int(crossing_mask_tolist_count),
        "rank_ties_possible": bool(rank_ties_possible),
        "bounded_hash_ordering_residual": bool(bounded_hash_ordering_residual),
    }


def _emit_sampled_universe_build_timing(
    timing_out: dict[str, float] | None,
    *,
    crossing_sets_seconds: float,
    candidate_dict_seconds: float,
    sort_sample_seconds: float,
    universe_total_seconds: float,
    sampled_crossing_mask_list_seconds: float,
    sampled_ordered_indices_seconds: float,
    sampled_rank_maps_seconds: float,
    sampled_sort_key_construction_seconds: float,
    sampled_sample_selection_seconds: float,
    sampled_candidate_dict_k_only_seconds: float,
) -> None:
    if timing_out is None:
        return
    timing_out["universe_crossing_sets"] = float(crossing_sets_seconds)
    timing_out["universe_candidate_dict"] = float(candidate_dict_seconds)
    timing_out["universe_sort_sample"] = float(sort_sample_seconds)
    timing_out["universe_build_total"] = float(universe_total_seconds)
    timing_out["sampled_universe_crossing_mask_list"] = float(
        sampled_crossing_mask_list_seconds
    )
    timing_out["sampled_universe_ordered_indices"] = float(
        sampled_ordered_indices_seconds
    )
    timing_out["sampled_universe_rank_maps"] = float(sampled_rank_maps_seconds)
    timing_out["sampled_universe_candidate_dict_k_only"] = float(
        sampled_candidate_dict_k_only_seconds
    )
    timing_out["sampled_universe_candidate_dict_full"] = 0.0
    timing_out["sampled_universe_sort_key_construction"] = float(
        sampled_sort_key_construction_seconds
    )
    timing_out["sampled_universe_sample_selection"] = float(
        sampled_sample_selection_seconds
    )


def build_w6_t10_crossing_candidate_universe_from_votes(
    *,
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
    max_abs_per_tensor: int,
    max_sampled_candidates: int,
    population_mode: str = POPULATION_MODE_SAMPLED_K64,
    timing_out: dict[str, float] | None = None,
    max_universe_build_seconds: float | None = None,
) -> dict[str, Any]:
    if str(population_mode) == POPULATION_MODE_SAMPLED_K64:
        return _build_w6_t10_crossing_candidate_universe_sample_k_before_materialize(
            tensor_states=tensor_states,
            votes_by_key=votes_by_key,
            max_abs_per_tensor=int(max_abs_per_tensor),
            max_sampled_candidates=int(max_sampled_candidates),
            timing_out=timing_out,
            max_universe_build_seconds=max_universe_build_seconds,
        )
    if str(population_mode) == POPULATION_MODE_FULL_CROSSING_ELIGIBLE:
        return build_w6_t10_crossing_candidate_universe_legacy_full_materialize(
            tensor_states=tensor_states,
            votes_by_key=votes_by_key,
            max_abs_per_tensor=int(max_abs_per_tensor),
            max_sampled_candidates=int(max_sampled_candidates),
            population_mode=POPULATION_MODE_FULL_CROSSING_ELIGIBLE,
            timing_out=timing_out,
            max_universe_build_seconds=max_universe_build_seconds,
        )
    raise ValueError(
        "population_mode must be "
        f"{POPULATION_MODE_SAMPLED_K64!r} or "
        f"{POPULATION_MODE_FULL_CROSSING_ELIGIBLE!r}, got {population_mode!r}"
    )


def build_w6_t10_crossing_candidate_universe_legacy_full_materialize(
    *,
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
    max_abs_per_tensor: int,
    max_sampled_candidates: int,
    population_mode: str = POPULATION_MODE_SAMPLED_K64,
    timing_out: dict[str, float] | None = None,
    max_universe_build_seconds: float | None = None,
) -> dict[str, Any]:
    universe_start = time.perf_counter()
    crossing_sets_seconds = 0.0
    candidate_dict_seconds = 0.0
    sampled_crossing_mask_list_seconds = 0.0
    sampled_ordered_indices_seconds = 0.0
    sampled_rank_maps_seconds = 0.0
    sampled_sort_key_construction_seconds = 0.0
    sampled_sample_selection_seconds = 0.0
    is_sampled_mode = str(population_mode) == POPULATION_MODE_SAMPLED_K64
    base_spec = w6_t10_base_spec(max_abs_per_tensor=int(max_abs_per_tensor))
    one_flip_spec = _single_flip_spec(base_spec)
    candidate_by_id: dict[str, dict[str, Any]] = {}
    crossing_eligible_count = 0

    def _maybe_abort_universe_build(*, reason: str) -> None:
        if max_universe_build_seconds is None:
            return
        elapsed = float(time.perf_counter() - universe_start)
        if elapsed < float(max_universe_build_seconds):
            return
        partial_timing = {
            "universe_crossing_sets": float(crossing_sets_seconds),
            "universe_candidate_dict": float(candidate_dict_seconds),
            "universe_sort_sample": 0.0,
            "universe_build_total": elapsed,
        }
        if is_sampled_mode:
            partial_timing.update(
                {
                    "sampled_universe_crossing_mask_list": float(
                        sampled_crossing_mask_list_seconds
                    ),
                    "sampled_universe_ordered_indices": float(
                        sampled_ordered_indices_seconds
                    ),
                    "sampled_universe_rank_maps": float(sampled_rank_maps_seconds),
                    "sampled_universe_candidate_dict_full": float(
                        candidate_dict_seconds
                    ),
                    "sampled_universe_sort_key_construction": float(
                        sampled_sort_key_construction_seconds
                    ),
                    "sampled_universe_sample_selection": float(
                        sampled_sample_selection_seconds
                    ),
                }
            )
        if timing_out is not None:
            timing_out.update(partial_timing)
        raise UniverseBuildMeasurementAborted(
            reason=str(reason),
            timing_out=partial_timing,
            partial_crossing_eligible_count=int(crossing_eligible_count),
            partial_candidate_count=len(candidate_by_id),
        )

    for state_key, state in sorted(tensor_states.items()):
        vote_state = state.vote_update_state()
        votes = votes_by_key[state_key]
        crossing_sets_start = time.perf_counter()
        crossing_mask = _crossing_eligible_mask_tensor(votes=votes, state=state)
        if not bool(crossing_mask.any().item()):
            crossing_sets_seconds += float(time.perf_counter() - crossing_sets_start)
            continue
        mask_list_start = time.perf_counter()
        crossing_eligible = {
            int(flat_index)
            for flat_index in crossing_mask.nonzero(as_tuple=False).flatten().tolist()
        }
        crossing_eligible_count += len(crossing_eligible)
        if is_sampled_mode:
            sampled_crossing_mask_list_seconds += float(
                time.perf_counter() - mask_list_start
            )
        ordered_indices_start = time.perf_counter()
        current_ordered, unordered, new_acc = _ordered_candidate_indices(
            state=vote_state,
            votes=votes,
            spec=base_spec,
            ordering_mode=LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
        )
        deterministic_ordered, deterministic_unordered, _ = _ordered_candidate_indices(
            state=vote_state,
            votes=votes,
            spec=base_spec,
            ordering_mode=LOCAL_SELECTION_ORDER_DETERMINISTIC_HASH_MATCHED,
        )
        filtered_unordered = [idx for idx in unordered if int(idx) in crossing_eligible]
        filtered_deterministic = [
            idx for idx in deterministic_unordered if int(idx) in crossing_eligible
        ]
        if set(filtered_unordered) != set(filtered_deterministic):
            raise RuntimeError(
                "W6/T=10 crossing candidate set drifted across scheduler orderings"
            )
        if is_sampled_mode:
            sampled_ordered_indices_seconds += float(
                time.perf_counter() - ordered_indices_start
            )
        rank_maps_start = time.perf_counter()
        current_rank = {
            int(flat_index): int(position)
            for position, flat_index in enumerate(current_ordered)
            if int(flat_index) in crossing_eligible
        }
        deterministic_rank = {
            int(flat_index): int(position)
            for position, flat_index in enumerate(deterministic_ordered)
            if int(flat_index) in crossing_eligible
        }
        vote_flat = votes.flatten().to(torch.int32)
        q_flat = vote_state.q_levels.flatten().to(torch.int32)
        acc_flat = vote_state.accumulators.flatten().to(torch.int32)
        if is_sampled_mode:
            sampled_rank_maps_seconds += float(time.perf_counter() - rank_maps_start)
        crossing_sets_seconds += float(time.perf_counter() - crossing_sets_start)

        candidate_dict_start = time.perf_counter()
        for flat_index in filtered_unordered:
            _maybe_abort_universe_build(reason="candidate_dict_loop_timeout")
            new_acc_signed = int(new_acc[int(flat_index)].item())
            candidate_id = _candidate_id(state_key, int(flat_index))
            candidate_by_id[candidate_id] = {
                "candidate_id": candidate_id,
                "state_key": state_key,
                "flat_index": int(flat_index),
                "vote_value": int(vote_flat[int(flat_index)].item()),
                "current_rank_position": int(current_rank[int(flat_index)]),
                "deterministic_hash_rank_position": int(
                    deterministic_rank[int(flat_index)]
                ),
                "current_q_level": int(q_flat[int(flat_index)].item()),
                "pre_accumulator_i16": int(acc_flat[int(flat_index)].item()),
                "new_acc_i32_signed": new_acc_signed,
                "proposal_direction": _sign_int(new_acc_signed),
            }
        candidate_dict_seconds += float(time.perf_counter() - candidate_dict_start)

    sort_sample_start = time.perf_counter()
    sort_keys_start = time.perf_counter()
    current_ordered_ids = [
        str(candidate["candidate_id"])
        for candidate in sorted(
            candidate_by_id.values(),
            key=lambda candidate: (
                int(candidate["current_rank_position"]),
                str(candidate["state_key"]),
                int(candidate["flat_index"]),
            ),
        )
    ]
    deterministic_ordered_ids = [
        str(candidate["candidate_id"])
        for candidate in sorted(
            candidate_by_id.values(),
            key=lambda candidate: (
                int(candidate["deterministic_hash_rank_position"]),
                str(candidate["state_key"]),
                int(candidate["flat_index"]),
            ),
        )
    ]
    if is_sampled_mode:
        sampled_sort_key_construction_seconds = float(
            time.perf_counter() - sort_keys_start
        )
    sample_selection_start = time.perf_counter()
    if population_mode == POPULATION_MODE_FULL_CROSSING_ELIGIBLE:
        sampled_ids = [str(candidate_id) for candidate_id in deterministic_ordered_ids]
    elif population_mode == POPULATION_MODE_SAMPLED_K64:
        sampled_ids = _sample_candidate_ids(
            current_ordered_ids,
            deterministic_ordered_ids,
            max_sampled_candidates=int(max_sampled_candidates),
        )
    else:
        raise ValueError(
            "population_mode must be "
            f"{POPULATION_MODE_SAMPLED_K64!r} or "
            f"{POPULATION_MODE_FULL_CROSSING_ELIGIBLE!r}, got {population_mode!r}"
        )
    if is_sampled_mode:
        sampled_sample_selection_seconds = float(
            time.perf_counter() - sample_selection_start
        )
    sort_sample_seconds = float(time.perf_counter() - sort_sample_start)
    universe_total_seconds = float(time.perf_counter() - universe_start)
    if timing_out is not None:
        timing_out["universe_crossing_sets"] = float(crossing_sets_seconds)
        timing_out["universe_candidate_dict"] = float(candidate_dict_seconds)
        timing_out["universe_sort_sample"] = float(sort_sample_seconds)
        timing_out["universe_build_total"] = float(universe_total_seconds)
        if is_sampled_mode:
            timing_out["sampled_universe_crossing_mask_list"] = float(
                sampled_crossing_mask_list_seconds
            )
            timing_out["sampled_universe_ordered_indices"] = float(
                sampled_ordered_indices_seconds
            )
            timing_out["sampled_universe_rank_maps"] = float(sampled_rank_maps_seconds)
            timing_out["sampled_universe_candidate_dict_full"] = float(
                candidate_dict_seconds
            )
            timing_out["sampled_universe_sort_key_construction"] = float(
                sampled_sort_key_construction_seconds
            )
            timing_out["sampled_universe_sample_selection"] = float(
                sampled_sample_selection_seconds
            )
    return {
        "base_spec": base_spec,
        "one_flip_spec": one_flip_spec,
        "votes_by_key": votes_by_key,
        "candidate_by_id": candidate_by_id,
        "sampled_ids": sampled_ids,
        "crossing_eligible_count": int(crossing_eligible_count),
        "population_mode": str(population_mode),
    }


def _build_w6_t10_crossing_candidate_universe_sample_k_before_materialize(
    *,
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
    max_abs_per_tensor: int,
    max_sampled_candidates: int,
    timing_out: dict[str, float] | None = None,
    max_universe_build_seconds: float | None = None,
) -> dict[str, Any]:
    universe_start = time.perf_counter()
    crossing_sets_seconds = 0.0
    candidate_dict_seconds = 0.0
    sampled_crossing_mask_list_seconds = 0.0
    sampled_ordered_indices_seconds = 0.0
    sampled_rank_maps_seconds = 0.0
    sampled_sort_key_construction_seconds = 0.0
    sampled_sample_selection_seconds = 0.0
    sampled_candidate_dict_k_only_seconds = 0.0
    base_spec = w6_t10_base_spec(max_abs_per_tensor=int(max_abs_per_tensor))
    one_flip_spec = _single_flip_spec(base_spec)
    candidate_by_id: dict[str, dict[str, Any]] = {}
    crossing_eligible_count = 0
    state_materialization_context: dict[str, dict[str, torch.Tensor]] = {}
    sorted_state_keys, state_key_ord_map = _canonical_state_key_ord_map(tensor_states)
    state_key_by_ord = {
        int(ordinal): str(state_key) for state_key, ordinal in state_key_ord_map.items()
    }
    flat_segments: list[torch.Tensor] = []
    state_ord_segments: list[torch.Tensor] = []
    current_rank_segments: list[torch.Tensor] = []
    det_rank_segments: list[torch.Tensor] = []
    max_flat_index = 0
    max_rank_position = 0
    materialization_counters = _strict_sampled_universe_materialization_counters()

    def _maybe_abort_universe_build(*, reason: str) -> None:
        if max_universe_build_seconds is None:
            return
        elapsed = float(time.perf_counter() - universe_start)
        if elapsed < float(max_universe_build_seconds):
            return
        partial_timing = {
            "universe_crossing_sets": float(crossing_sets_seconds),
            "universe_candidate_dict": float(candidate_dict_seconds),
            "universe_sort_sample": 0.0,
            "universe_build_total": elapsed,
            "sampled_universe_crossing_mask_list": float(
                sampled_crossing_mask_list_seconds
            ),
            "sampled_universe_ordered_indices": float(
                sampled_ordered_indices_seconds
            ),
            "sampled_universe_rank_maps": float(sampled_rank_maps_seconds),
            "sampled_universe_candidate_dict_k_only": float(
                sampled_candidate_dict_k_only_seconds
            ),
            "sampled_universe_candidate_dict_full": 0.0,
            "sampled_universe_sort_key_construction": float(
                sampled_sort_key_construction_seconds
            ),
            "sampled_universe_sample_selection": float(
                sampled_sample_selection_seconds
            ),
        }
        if timing_out is not None:
            timing_out.update(partial_timing)
        raise UniverseBuildMeasurementAborted(
            reason=str(reason),
            timing_out=partial_timing,
            partial_crossing_eligible_count=int(crossing_eligible_count),
            partial_candidate_count=len(candidate_by_id),
        )

    for state_key, state in sorted(tensor_states.items()):
        vote_state = state.vote_update_state()
        votes = votes_by_key[state_key]
        crossing_sets_start = time.perf_counter()
        crossing_mask = _crossing_eligible_mask_tensor(votes=votes, state=state)
        if not bool(crossing_mask.any().item()):
            crossing_sets_seconds += float(time.perf_counter() - crossing_sets_start)
            continue
        crossing_flat = crossing_mask.nonzero(as_tuple=False).flatten().to(torch.int64)
        crossing_eligible_count += int(crossing_flat.numel())
        ordered_indices_start = time.perf_counter()
        new_acc, current_rank_at_cross, crossing_for_sort, candidate_idx = (
            _vote_plan_rank_and_new_acc_at_crossings(
                vote_state=vote_state,
                votes=votes,
                spec=base_spec,
                crossing_flat=crossing_flat,
                ordering_mode=LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
            )
        )
        _new_acc_det, det_rank_at_cross, crossing_for_sort_det, candidate_idx_det = (
            _vote_plan_rank_and_new_acc_at_crossings(
                vote_state=vote_state,
                votes=votes,
                spec=base_spec,
                crossing_flat=crossing_flat,
                ordering_mode=LOCAL_SELECTION_ORDER_DETERMINISTIC_HASH_MATCHED,
            )
        )
        if not torch.equal(candidate_idx, candidate_idx_det) or not torch.equal(
            crossing_for_sort, crossing_for_sort_det
        ):
            raise RuntimeError(
                "W6/T=10 crossing candidate set drifted across scheduler orderings"
            )
        if int(crossing_for_sort.numel()) > 0 and (
            int(current_rank_at_cross.numel()) != int(crossing_for_sort.numel())
            or int(det_rank_at_cross.numel()) != int(crossing_for_sort.numel())
            or bool((current_rank_at_cross < 0).any().item())
            or bool((det_rank_at_cross < 0).any().item())
        ):
            raise RuntimeError(
                "W6/T=10 crossing candidate rank lookup failed for scheduler ordering"
            )
        sampled_ordered_indices_seconds += float(
            time.perf_counter() - ordered_indices_start
        )
        _maybe_abort_universe_build(reason="ordered_indices_timeout")
        rank_maps_start = time.perf_counter()
        vote_flat = votes.flatten().to(torch.int32)
        q_flat = vote_state.q_levels.flatten().to(torch.int32)
        acc_flat = vote_state.accumulators.flatten().to(torch.int32)
        state_materialization_context[str(state_key)] = {
            "vote_flat": vote_flat,
            "q_flat": q_flat,
            "acc_flat": acc_flat,
            "new_acc": new_acc,
        }
        if int(crossing_for_sort.numel()) > 0:
            state_ord = int(state_key_ord_map[str(state_key)])
            flat_segments.append(crossing_for_sort)
            state_ord_segments.append(
                torch.full(
                    (int(crossing_for_sort.numel()),),
                    state_ord,
                    dtype=torch.int64,
                )
            )
            current_rank_segments.append(current_rank_at_cross.to(torch.int64))
            det_rank_segments.append(det_rank_at_cross.to(torch.int64))
            max_flat_index = max(max_flat_index, int(crossing_for_sort.max().item()))
            max_rank_position = max(
                max_rank_position,
                int(current_rank_at_cross.max().item()),
                int(det_rank_at_cross.max().item()),
            )
        sampled_rank_maps_seconds += float(time.perf_counter() - rank_maps_start)
        crossing_sets_seconds += float(time.perf_counter() - crossing_sets_start)

    sort_sample_start = time.perf_counter()
    sort_keys_start = time.perf_counter()
    if crossing_eligible_count == 0:
        flat_index_all = torch.empty(0, dtype=torch.int64)
        state_ord_all = torch.empty(0, dtype=torch.int64)
        current_rank_all = torch.empty(0, dtype=torch.int64)
        det_rank_all = torch.empty(0, dtype=torch.int64)
    else:
        flat_index_all = torch.cat(flat_segments)
        state_ord_all = torch.cat(state_ord_segments)
        current_rank_all = torch.cat(current_rank_segments)
        det_rank_all = torch.cat(det_rank_segments)
    w_flat, w_state = _sampled_universe_global_sort_weights(
        state_count=len(sorted_state_keys),
        max_flat_index=max_flat_index,
        max_rank=max_rank_position,
    )
    current_composite = _packed_crossing_sort_key(
        current_rank_all,
        state_ord_all,
        flat_index_all,
        w_state=w_state,
        w_flat=w_flat,
    )
    det_composite = _packed_crossing_sort_key(
        det_rank_all,
        state_ord_all,
        flat_index_all,
        w_state=w_state,
        w_flat=w_flat,
    )
    current_perm = torch.argsort(current_composite, stable=True)
    det_perm = torch.argsort(det_composite, stable=True)
    sampled_sort_key_construction_seconds = float(
        time.perf_counter() - sort_keys_start
    )
    sample_selection_start = time.perf_counter()
    selected_rows = _sample_crossing_row_indices_interleaved(
        current_perm,
        det_perm,
        max_sampled_candidates=int(max_sampled_candidates),
    )
    sampled_ids = [
        _candidate_id(
            state_key_by_ord[int(state_ord_all[int(row)].item())],
            int(flat_index_all[int(row)].item()),
        )
        for row in selected_rows
    ]
    sampled_sample_selection_seconds = float(
        time.perf_counter() - sample_selection_start
    )
    candidate_dict_start = time.perf_counter()
    for row in selected_rows:
        _maybe_abort_universe_build(reason="candidate_dict_k_only_timeout")
        row_index = int(row)
        state_ord = int(state_ord_all[row_index].item())
        flat_index = int(flat_index_all[row_index].item())
        state_key = state_key_by_ord[state_ord]
        ctx = state_materialization_context[state_key]
        candidate_id = _candidate_id(state_key, flat_index)
        candidate_by_id[str(candidate_id)] = _materialize_w6_t10_candidate_dict_entry(
            state_key=state_key,
            flat_index=flat_index,
            current_rank_position=int(current_rank_all[row_index].item()),
            deterministic_hash_rank_position=int(det_rank_all[row_index].item()),
            vote_flat=ctx["vote_flat"],
            q_flat=ctx["q_flat"],
            acc_flat=ctx["acc_flat"],
            new_acc=ctx["new_acc"],
        )
    sampled_candidate_dict_k_only_seconds = float(
        time.perf_counter() - candidate_dict_start
    )
    candidate_dict_seconds = sampled_candidate_dict_k_only_seconds
    sort_sample_seconds = float(time.perf_counter() - sort_sample_start)
    universe_total_seconds = float(time.perf_counter() - universe_start)
    _emit_sampled_universe_build_timing(
        timing_out,
        crossing_sets_seconds=crossing_sets_seconds,
        candidate_dict_seconds=candidate_dict_seconds,
        sort_sample_seconds=sort_sample_seconds,
        universe_total_seconds=universe_total_seconds,
        sampled_crossing_mask_list_seconds=sampled_crossing_mask_list_seconds,
        sampled_ordered_indices_seconds=sampled_ordered_indices_seconds,
        sampled_rank_maps_seconds=sampled_rank_maps_seconds,
        sampled_sort_key_construction_seconds=sampled_sort_key_construction_seconds,
        sampled_sample_selection_seconds=sampled_sample_selection_seconds,
        sampled_candidate_dict_k_only_seconds=sampled_candidate_dict_k_only_seconds,
    )
    if timing_out is not None:
        for key, value in materialization_counters.items():
            if isinstance(value, bool):
                timing_out[f"strict_path_{key}"] = float(int(value))
            else:
                timing_out[f"strict_path_{key}"] = float(value)
    return {
        "base_spec": base_spec,
        "one_flip_spec": one_flip_spec,
        "votes_by_key": votes_by_key,
        "candidate_by_id": candidate_by_id,
        "sampled_ids": sampled_ids,
        "crossing_eligible_count": int(crossing_eligible_count),
        "population_mode": POPULATION_MODE_SAMPLED_K64,
        "strict_path_materialization_counters": dict(materialization_counters),
    }


def count_w6_t10_crossing_eligible_from_votes(
    *,
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
) -> int:
    return int(
        sum(
            crossing_count_by_state_key_from_votes(
                tensor_states=tensor_states,
                votes_by_key=votes_by_key,
            ).values()
        )
    )


def crossing_count_by_state_key_from_votes(
    *,
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state_key, state in sorted(tensor_states.items()):
        crossing_mask = _crossing_eligible_mask_tensor(
            votes=votes_by_key[state_key],
            state=state,
        )
        counts[str(state_key)] = int(crossing_mask.sum().item())
    return counts


def fabricate_all_crossing_tensor_batch(
    *,
    module_count: int,
    numel_per_module: int,
    module_prefix: str = "synthetic.mod",
    weight_shape: tuple[int, ...] | None = None,
    frozen_scale: torch.Tensor | float | None = None,
) -> tuple[dict[str, BoundedDeltaTensorState], dict[str, torch.Tensor]]:
    """CPU-only synthetic batch where every flat index is crossing-eligible."""

    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        make_bounded_tensor_state,
    )

    if weight_shape is not None:
        shape = tuple(int(dim) for dim in weight_shape)
        shape_numel = 1
        for dim in shape:
            shape_numel *= int(dim)
        if int(shape_numel) != int(numel_per_module):
            raise ValueError(
                "weight_shape numel must match numel_per_module: "
                f"{shape} vs {numel_per_module}"
            )
    else:
        shape = (int(numel_per_module),)
    scale = 1.0 if frozen_scale is None else frozen_scale

    tensor_states: dict[str, BoundedDeltaTensorState] = {}
    votes_by_key: dict[str, torch.Tensor] = {}
    q_template = torch.zeros(shape, dtype=torch.int8)
    acc_template = torch.full(shape, 9, dtype=torch.int16)
    votes_template = torch.full(shape, 12, dtype=torch.int16)
    for module_index in range(module_count):
        state_key = f"{module_prefix}{module_index}"
        tensor_states[state_key] = make_bounded_tensor_state(
            state_key,
            q_template.clone(),
            scale,
            acc_template.clone(),
        )
        votes_by_key[state_key] = votes_template.clone()
    expected = int(module_count) * int(numel_per_module)
    actual = count_w6_t10_crossing_eligible_from_votes(
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
    )
    if int(actual) != expected:
        raise ValueError(f"fixture not all-crossing: got {actual}, expected {expected}")
    return tensor_states, votes_by_key


def _attribute_dominant_universe_subphase(
    timing: Mapping[str, float],
) -> tuple[str, dict[str, float]]:
    candidates = {
        "universe_candidate_dict": float(timing.get("universe_candidate_dict", 0.0)),
        "universe_crossing_sets": float(timing.get("universe_crossing_sets", 0.0)),
        "universe_sort_sample": float(timing.get("universe_sort_sample", 0.0)),
    }
    universe_total = float(timing.get("universe_build_total", 0.0))
    denominator = universe_total if universe_total > 0.0 else 1.0
    shares = {key: value / denominator for key, value in candidates.items()}
    dominant = max(candidates, key=candidates.get)
    return dominant, shares


def _attribute_dominant_sampled_universe_subphase(
    timing: Mapping[str, float],
) -> tuple[str, dict[str, float]]:
    candidates = {
        "sampled_universe_candidate_dict_k_only": float(
            timing.get("sampled_universe_candidate_dict_k_only", 0.0)
        ),
        "sampled_universe_candidate_dict_full": float(
            timing.get("sampled_universe_candidate_dict_full", 0.0)
        ),
        "sampled_universe_crossing_mask_list": float(
            timing.get("sampled_universe_crossing_mask_list", 0.0)
        ),
        "sampled_universe_ordered_indices": float(
            timing.get("sampled_universe_ordered_indices", 0.0)
        ),
        "sampled_universe_rank_maps": float(
            timing.get("sampled_universe_rank_maps", 0.0)
        ),
        "sampled_universe_sort_key_construction": float(
            timing.get("sampled_universe_sort_key_construction", 0.0)
        ),
        "sampled_universe_sample_selection": float(
            timing.get("sampled_universe_sample_selection", 0.0)
        ),
    }
    universe_total = float(timing.get("universe_build_total", 0.0))
    denominator = universe_total if universe_total > 0.0 else 1.0
    shares = {key: value / denominator for key, value in candidates.items()}
    dominant = max(candidates, key=candidates.get)
    return dominant, shares


def measure_universe_build_subphase_scale_ladder(
    *,
    ladder_points: Sequence[tuple[int, int]] | None = None,
    max_abs_per_tensor: int = 4096,
    max_wall_seconds: float = 280.0,
    production_crossing_eligible_target: int = 11_640_000,
) -> dict[str, Any]:
    """Bounded slice-0 measurement: scale ladder + timeout-abort partial receipts."""

    if ladder_points is None:
        ladder_points = [
            (1, 10_000),
            (1, 50_000),
            (4, 50_000),
            (8, 100_000),
            (16, 100_000),
            (32, 50_000),
            (32, 100_000),
        ]
    runs: list[dict[str, Any]] = []
    wall_total = 0.0
    for module_count, numel_per_module in ladder_points:
        crossing_count = int(module_count) * int(numel_per_module)
        if wall_total >= max_wall_seconds:
            runs.append(
                {
                    "module_count": module_count,
                    "numel_per_module": numel_per_module,
                    "expected_crossing_eligible": crossing_count,
                    "status": "skipped_wall_budget",
                }
            )
            continue
        tensor_states, votes_by_key = fabricate_all_crossing_tensor_batch(
            module_count=module_count,
            numel_per_module=numel_per_module,
        )
        timing_out: dict[str, float] = {}
        run_wall_start = time.perf_counter()
        timeout_budget: float | None = None
        if crossing_count >= 500_000:
            timeout_budget = min(30.0, max(0.05, max_wall_seconds - wall_total))
        record: dict[str, Any] = {
            "module_count": module_count,
            "numel_per_module": numel_per_module,
            "expected_crossing_eligible": crossing_count,
        }
        try:
            universe = build_w6_t10_crossing_candidate_universe_from_votes(
                tensor_states=tensor_states,
                votes_by_key=votes_by_key,
                max_abs_per_tensor=max_abs_per_tensor,
                max_sampled_candidates=64,
                population_mode=POPULATION_MODE_FULL_CROSSING_ELIGIBLE,
                timing_out=timing_out,
                max_universe_build_seconds=timeout_budget,
            )
            record.update(
                {
                    "status": "completed",
                    "crossing_eligible_count": universe["crossing_eligible_count"],
                    "sampled_ids_count": len(universe["sampled_ids"]),
                    "candidate_dict_size": len(universe["candidate_by_id"]),
                }
            )
        except UniverseBuildMeasurementAborted as exc:
            timing_out = dict(exc.timing_out)
            record.update(
                {
                    "status": "timeout_abort",
                    "abort_reason": exc.reason,
                    "partial_crossing_eligible_count": exc.partial_crossing_eligible_count,
                    "partial_candidate_count": exc.partial_candidate_count,
                }
            )
        run_wall = float(time.perf_counter() - run_wall_start)
        wall_total += run_wall
        dominant, shares = _attribute_dominant_universe_subphase(timing_out)
        record.update(
            {
                "wall_seconds": run_wall,
                "timing": dict(timing_out),
                "dominant_subphase": dominant,
                "subphase_share_of_universe_build": shares,
            }
        )
        runs.append(record)

    projection_points = [
        (
            int(run["expected_crossing_eligible"]),
            float(run["timing"].get("universe_candidate_dict", 0.0)),
        )
        for run in runs
        if run.get("timing", {}).get("universe_candidate_dict")
    ]
    projected_candidate_dict_seconds: float | None = None
    if len(projection_points) >= 2:
        xs = [point[0] for point in projection_points]
        ys = [point[1] for point in projection_points]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denom = sum((x - mean_x) ** 2 for x in xs) or 1.0
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
        projected_candidate_dict_seconds = float(
            slope * float(production_crossing_eligible_target)
        )

    runs_with_timing = [run for run in runs if run.get("timing")]
    overall_dominant = None
    if runs_with_timing:
        largest = max(runs_with_timing, key=lambda item: item["expected_crossing_eligible"])
        overall_dominant = str(largest["dominant_subphase"])
    materialization_dominates = overall_dominant == "universe_candidate_dict"
    return {
        "measurement_method": "scale_ladder_with_timeout_abort_partial_receipts",
        "production_crossing_eligible_target": int(production_crossing_eligible_target),
        "ladder_runs": runs,
        "overall_dominant_subphase": overall_dominant,
        "materialization_dominates": materialization_dominates,
        "projected_universe_candidate_dict_seconds_at_production": (
            projected_candidate_dict_seconds
        ),
        "wall_seconds_total": wall_total,
        "r1_fast_path_gate_recommendation": (
            "authorize_slice_1_fast_path"
            if materialization_dominates
            else "replan_required_proxy_gather_or_other_dominates"
        ),
    }


ORACLE_SCREEN_EXCLUSION_RECORD = {
    "relaunch_scope": "selector_support_consensus_v0",
    "excluded_surface": (
        "oracle_screen_runner.run_candidate_set_viability_oracle_screen "
        "(--oracle-screen-mode launch surface; NOT a selector-chain step)"
    ),
    "forbidden_in_this_relaunch": "oracle_screen_runner.py edit",
    "follow_on": (
        "Separate bounded-measurement/fix gate if an oracle-screen bundle is queued"
    ),
    "same_defect_class": "materialize-before-sample in inline universe build",
}


def capture_legacy_sampled_universe_reference(
    *,
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
    max_abs_per_tensor: int,
    max_sampled_candidates: int,
) -> dict[str, Any]:
    """Golden reference snapshot for fix-slice parity (builder behavior unchanged)."""

    universe = build_w6_t10_crossing_candidate_universe_legacy_full_materialize(
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
        max_abs_per_tensor=int(max_abs_per_tensor),
        max_sampled_candidates=int(max_sampled_candidates),
        population_mode=POPULATION_MODE_SAMPLED_K64,
    )
    sampled_ids = [str(candidate_id) for candidate_id in universe["sampled_ids"]]
    candidate_fields = {
        str(candidate_id): dict(universe["candidate_by_id"][candidate_id])
        for candidate_id in sampled_ids
    }
    return {
        "population_mode": POPULATION_MODE_SAMPLED_K64,
        "max_sampled_candidates": int(max_sampled_candidates),
        "crossing_eligible_count": int(universe["crossing_eligible_count"]),
        "sampled_ids": sampled_ids,
        "sampled_ids_hash16": _candidate_ids_hash16(sampled_ids),
        "candidate_fields_by_id": candidate_fields,
        "candidate_dict_size": len(universe["candidate_by_id"]),
    }


def measure_sampled_universe_build_subphase_scale_ladder(
    *,
    ladder_points: Sequence[tuple[int, int]] | None = None,
    max_abs_per_tensor: int = 4096,
    max_wall_seconds: float = 280.0,
    production_crossing_eligible_target: int = 11_640_000,
    max_sampled_candidates_values: Sequence[int] = (8, 64),
) -> dict[str, Any]:
    """Bounded sampled-mode measurement: scale ladder + timeout-abort @ K=8 and K=64."""

    if ladder_points is None:
        ladder_points = [
            (1, 10_000),
            (1, 50_000),
            (4, 50_000),
            (8, 100_000),
            (16, 100_000),
            (32, 50_000),
            (32, 100_000),
        ]
    measurement_by_k: dict[str, Any] = {}
    wall_grand_total = 0.0
    for max_sampled in max_sampled_candidates_values:
        runs: list[dict[str, Any]] = []
        wall_total = 0.0
        for module_count, numel_per_module in ladder_points:
            crossing_count = int(module_count) * int(numel_per_module)
            if wall_total >= max_wall_seconds:
                runs.append(
                    {
                        "module_count": module_count,
                        "numel_per_module": numel_per_module,
                        "expected_crossing_eligible": crossing_count,
                        "status": "skipped_wall_budget",
                    }
                )
                continue
            tensor_states, votes_by_key = fabricate_all_crossing_tensor_batch(
                module_count=module_count,
                numel_per_module=numel_per_module,
            )
            timing_out: dict[str, float] = {}
            run_wall_start = time.perf_counter()
            timeout_budget: float | None = None
            if crossing_count >= 500_000:
                timeout_budget = min(30.0, max(0.05, max_wall_seconds - wall_total))
            record: dict[str, Any] = {
                "module_count": module_count,
                "numel_per_module": numel_per_module,
                "expected_crossing_eligible": crossing_count,
                "max_sampled_candidates": int(max_sampled),
            }
            try:
                universe = build_w6_t10_crossing_candidate_universe_from_votes(
                    tensor_states=tensor_states,
                    votes_by_key=votes_by_key,
                    max_abs_per_tensor=max_abs_per_tensor,
                    max_sampled_candidates=int(max_sampled),
                    population_mode=POPULATION_MODE_SAMPLED_K64,
                    timing_out=timing_out,
                    max_universe_build_seconds=timeout_budget,
                )
                record.update(
                    {
                        "status": "completed",
                        "crossing_eligible_count": universe["crossing_eligible_count"],
                        "sampled_ids_count": len(universe["sampled_ids"]),
                        "candidate_dict_size": len(universe["candidate_by_id"]),
                        "k_only_materialization": (
                            int(len(universe["candidate_by_id"]))
                            == int(len(universe["sampled_ids"]))
                        ),
                        "materialize_before_sample_observed": (
                            int(len(universe["candidate_by_id"]))
                            > int(len(universe["sampled_ids"]))
                            and int(universe["crossing_eligible_count"])
                            > int(max_sampled)
                        ),
                    }
                )
            except UniverseBuildMeasurementAborted as exc:
                timing_out = dict(exc.timing_out)
                record.update(
                    {
                        "status": "timeout_abort",
                        "abort_reason": exc.reason,
                        "partial_crossing_eligible_count": exc.partial_crossing_eligible_count,
                        "partial_candidate_count": exc.partial_candidate_count,
                        "materialize_before_sample_observed": (
                            int(exc.partial_candidate_count)
                            < int(exc.partial_crossing_eligible_count)
                            and int(exc.partial_crossing_eligible_count) > int(max_sampled)
                        ),
                    }
                )
            run_wall = float(time.perf_counter() - run_wall_start)
            wall_total += run_wall
            dominant, shares = _attribute_dominant_sampled_universe_subphase(timing_out)
            aggregate_dominant, aggregate_shares = _attribute_dominant_universe_subphase(
                timing_out
            )
            record.update(
                {
                    "wall_seconds": run_wall,
                    "timing": dict(timing_out),
                    "dominant_sampled_subphase": dominant,
                    "sampled_subphase_share_of_universe_build": shares,
                    "dominant_aggregate_subphase": aggregate_dominant,
                    "aggregate_subphase_share_of_universe_build": aggregate_shares,
                }
            )
            runs.append(record)

        projection_points = [
            (
                int(run["expected_crossing_eligible"]),
                float(
                    run["timing"].get(
                        "sampled_universe_candidate_dict_k_only",
                        run["timing"].get("sampled_universe_candidate_dict_full", 0.0),
                    )
                ),
            )
            for run in runs
            if run.get("timing")
        ]
        projected_dict_seconds: float | None = None
        if len(projection_points) >= 2:
            xs = [point[0] for point in projection_points]
            ys = [point[1] for point in projection_points]
            mean_x = sum(xs) / len(xs)
            mean_y = sum(ys) / len(ys)
            denom = sum((x - mean_x) ** 2 for x in xs) or 1.0
            slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
            projected_dict_seconds = float(
                slope * float(production_crossing_eligible_target)
            )

        runs_with_timing = [run for run in runs if run.get("timing")]
        overall_dominant = None
        if runs_with_timing:
            largest = max(
                runs_with_timing, key=lambda item: item["expected_crossing_eligible"]
            )
            overall_dominant = str(largest["dominant_sampled_subphase"])
        full_universe_materialization_dominates = overall_dominant in {
            "sampled_universe_candidate_dict_full",
            "sampled_universe_candidate_dict_k_only",
        } and any(
            bool(run.get("materialize_before_sample_observed"))
            for run in runs
            if run.get("status") == "completed"
        )
        materialize_before_sample_confirmed = any(
            bool(run.get("materialize_before_sample_observed"))
            for run in runs
            if run.get("status") in {"completed", "timeout_abort"}
        )
        k_only_materialization_confirmed = all(
            bool(run.get("k_only_materialization"))
            for run in runs
            if run.get("status") == "completed"
        )
        measurement_by_k[str(int(max_sampled))] = {
            "max_sampled_candidates": int(max_sampled),
            "ladder_runs": runs,
            "overall_dominant_sampled_subphase": overall_dominant,
            "full_universe_materialization_dominates": (
                full_universe_materialization_dominates
            ),
            "materialize_before_sample_confirmed": materialize_before_sample_confirmed,
            "k_only_materialization_confirmed": k_only_materialization_confirmed,
            "projected_sampled_universe_candidate_dict_full_seconds_at_production": (
                projected_dict_seconds
            ),
            "wall_seconds_total": wall_total,
            "r1_builder_fix_gate_recommendation": (
                "authorize_sample_before_materialize_fix"
                if materialize_before_sample_confirmed
                else "sample_before_materialize_fix_validated"
                if k_only_materialization_confirmed
                else "replan_required_other_subphase_dominates"
            ),
        }
        wall_grand_total += wall_total

    return {
        "schema": "grad_proxy_sampled_universe_measurement_slice0/v1",
        "measurement_method": (
            "sampled_k64_scale_ladder_with_timeout_abort_partial_receipts"
        ),
        "population_mode": POPULATION_MODE_SAMPLED_K64,
        "production_crossing_eligible_target": int(production_crossing_eligible_target),
        "measurement_by_k": measurement_by_k,
        "wall_seconds_total": wall_grand_total,
        "oracle_screen_exclusion": dict(ORACLE_SCREEN_EXCLUSION_RECORD),
        "bounded_measurement_r4": {
            "max_wall_seconds_per_k": float(max_wall_seconds),
            "fail_closed": True,
            "receipt_producing_on_timeout": True,
        },
    }


def _normalize_votes_by_key(
    votes_by_key: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        str(state_key): votes.detach().cpu().to(torch.int16).contiguous()
        for state_key, votes in votes_by_key.items()
    }


def _probe_warmup_helpers() -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        apply_bounded_delta_vote_step,
    )
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        ARM_A0_RANK_BUCKET_CURRENT,
        SCIENCE_LOCAL_SELECTION_ORDERING_SEED,
        _compute_ce_weighted_grads,
        _science_local_selection_ordering_mode,
        _weighted_grads_to_science_arm_votes,
        default_dry_run_rank_vote_spec,
        default_vote_update_spec,
    )

    return {
        "ARM_A0_RANK_BUCKET_CURRENT": ARM_A0_RANK_BUCKET_CURRENT,
        "SCIENCE_LOCAL_SELECTION_ORDERING_SEED": SCIENCE_LOCAL_SELECTION_ORDERING_SEED,
        "apply_bounded_delta_vote_step": apply_bounded_delta_vote_step,
        "_compute_ce_weighted_grads": _compute_ce_weighted_grads,
        "_science_local_selection_ordering_mode": _science_local_selection_ordering_mode,
        "_weighted_grads_to_science_arm_votes": _weighted_grads_to_science_arm_votes,
        "default_dry_run_rank_vote_spec": default_dry_run_rank_vote_spec,
        "default_vote_update_spec": default_vote_update_spec,
    }


def derive_probe_science_arm_votes(
    *,
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    eligible_modules: Mapping[str, BitLinear],
    device: torch.device,
    extras: Mapping[str, Any],
    max_abs_per_tensor: int,
) -> tuple[float, dict[str, torch.Tensor]]:
    helpers = _probe_warmup_helpers()
    weighted_grads, loss, _metrics = helpers["_compute_ce_weighted_grads"](
        model,
        batch,
        tensor_states,
        eligible_modules,
        device=device,
        extras=extras,
    )
    rank_spec = helpers["default_dry_run_rank_vote_spec"]()
    vote_spec = helpers["default_vote_update_spec"](int(max_abs_per_tensor))
    votes_by_key, _pressure_by_key, finite_weighted_grad = helpers[
        "_weighted_grads_to_science_arm_votes"
    ](
        weighted_grads,
        tensor_states,
        rank_spec=rank_spec,
        vote_spec=vote_spec,
        science_arm=str(helpers["ARM_A0_RANK_BUCKET_CURRENT"]),
    )
    if not bool(finite_weighted_grad):
        raise RuntimeError("probe science-arm votes require finite weighted grads")
    return float(loss.detach().cpu().item()), _normalize_votes_by_key(votes_by_key)


def apply_probe_off_path_warmup_step(
    *,
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
    max_abs_per_tensor: int,
    warmup_step_index: int,
) -> dict[str, BoundedDeltaTensorState]:
    helpers = _probe_warmup_helpers()
    vote_specs = {
        state_key: helpers["default_vote_update_spec"](int(max_abs_per_tensor))
        for state_key in tensor_states
    }
    step_result = helpers["apply_bounded_delta_vote_step"](
        tensor_states,
        votes_by_key,
        vote_specs,
        two_tier_carry_w6_enabled=False,
        local_selection_ordering_mode=helpers["_science_local_selection_ordering_mode"](
            str(helpers["ARM_A0_RANK_BUCKET_CURRENT"])
        ),
        local_selection_ordering_seed=int(helpers["SCIENCE_LOCAL_SELECTION_ORDERING_SEED"]),
        local_selection_ordering_step=int(warmup_step_index),
    )
    return dict(step_result.tensor_states)


def derive_w6_t10_candidate_delta_weight(
    *,
    tensor_state: BoundedDeltaTensorState,
    votes: torch.Tensor,
    candidate: Mapping[str, Any],
    max_abs_per_tensor: int,
) -> float:
    flat_index = int(candidate["flat_index"])
    weight = _vectorized_w6_t10_delta_weights_at_flat_indices(
        tensor_state=tensor_state,
        votes=votes,
        flat_indices=torch.tensor([flat_index], dtype=torch.int64),
        max_abs_per_tensor=int(max_abs_per_tensor),
    )
    return float(weight[0].item())


def _crossing_eligible_mask_tensor(
    *,
    votes: torch.Tensor,
    state: BoundedDeltaTensorState,
    threshold_abs: int = int(CROSSING_THRESHOLD_ABS),
) -> torch.Tensor:
    """Tensor mask matching crossing_eligible_flat_indices W6 carry semantics."""

    vote_state = state.vote_update_state()
    q_i16 = vote_state.q_levels.flatten().to(torch.int16)
    acc_i32 = vote_state.accumulators.flatten().to(torch.int32)
    vote_i32 = votes.flatten().to(torch.int32)
    clip_min, clip_max = effective_clip_w6()
    new_acc_i32 = (acc_i32 + vote_i32).clamp(int(clip_min), int(clip_max))
    threshold = int(threshold_abs)
    return ((new_acc_i32 >= threshold) & (q_i16 < 1)) | (
        (new_acc_i32 <= -threshold) & (q_i16 > -1)
    )


def _crossing_flat_indices_by_state_key(
    *,
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    flat_indices_by_state: dict[str, torch.Tensor] = {}
    for state_key, state in sorted(tensor_states.items()):
        crossing_mask = _crossing_eligible_mask_tensor(
            votes=votes_by_key[state_key],
            state=state,
        )
        if bool(crossing_mask.any().item()):
            flat_indices_by_state[str(state_key)] = (
                crossing_mask.nonzero(as_tuple=False).flatten().to(torch.int64)
            )
    return flat_indices_by_state


def _scatter_vectorized_grad_proxy_ingress_for_state(
    *,
    local_loss_delta: torch.Tensor,
    tensor_state: BoundedDeltaTensorState,
    votes: torch.Tensor,
    flat_indices: torch.Tensor,
    grad_proxies: torch.Tensor,
    max_abs_per_tensor: int,
) -> None:
    flat_indices = flat_indices.flatten().to(dtype=torch.int64)
    grad_proxies = grad_proxies.flatten().to(dtype=torch.float32)
    if flat_indices.numel() == 0:
        return
    if int(flat_indices.numel()) != int(grad_proxies.numel()):
        raise ValueError(
            "grad_proxy_ingress_shape_mismatch: "
            f"flat_indices={int(flat_indices.numel())} "
            f"grad_proxies={int(grad_proxies.numel())}"
        )
    delta_weights = _vectorized_w6_t10_delta_weights_at_flat_indices(
        tensor_state=tensor_state,
        votes=votes,
        flat_indices=flat_indices,
        max_abs_per_tensor=int(max_abs_per_tensor),
    )
    local_loss_delta.view(-1)[flat_indices] = grad_proxies * delta_weights


def _vectorized_w6_t10_delta_weights_at_flat_indices(
    *,
    tensor_state: BoundedDeltaTensorState,
    votes: torch.Tensor,
    flat_indices: torch.Tensor,
    max_abs_per_tensor: int,
) -> torch.Tensor:
    """Exact batched parity with scalar derive_w6_t10 per flat_index.

    O(numel) shared composite + O(batch) 1D gather/scatter — no
    ``(batch,numel)`` materialization.
    """

    base_spec = w6_t10_base_spec(max_abs_per_tensor=int(max_abs_per_tensor))
    vote_state = tensor_state.vote_update_state()
    votes_flat = votes.flatten().to(torch.int32)
    q_flat = vote_state.q_levels.flatten().to(torch.int32)
    acc_i32 = vote_state.accumulators.flatten().to(torch.int32)
    frozen_scale = float(tensor_state.frozen_scale.detach().cpu().item())
    threshold = int(base_spec.threshold_abs)
    clip_min = int(base_spec.accumulator_clip_min)
    clip_max = int(base_spec.accumulator_clip_max)
    numel = int(q_flat.numel())
    device = votes.device
    flat_indices = flat_indices.flatten().to(dtype=torch.int64, device=device)
    if flat_indices.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=device)

    decayed = torch.div(
        acc_i32 * int(base_spec.decay_numerator),
        int(base_spec.decay_denominator),
        rounding_mode="trunc",
    )
    base_new_acc = (decayed + torch.zeros_like(votes_flat)).clamp(clip_min, clip_max)
    candidates_base = ((base_new_acc >= threshold) & (q_flat < 1)) | (
        (base_new_acc <= -threshold) & (q_flat > -1)
    )
    idx_range = torch.arange(numel, dtype=torch.int64, device=device)
    composite_base = base_new_acc.abs() * (numel + 1) + (numel - idx_range)
    composite_base = composite_base.masked_fill(~candidates_base, -1)

    batch_size = int(flat_indices.numel())
    new_acc_at_target = (
        decayed[flat_indices] + votes_flat[flat_indices]
    ).clamp(clip_min, clip_max)
    candidate_at_target = ((new_acc_at_target >= threshold) & (q_flat[flat_indices] < 1)) | (
        (new_acc_at_target <= -threshold) & (q_flat[flat_indices] > -1)
    )
    composite_at_target = new_acc_at_target.abs() * (numel + 1) + (
        numel - flat_indices
    ).to(new_acc_at_target.dtype)
    composite_at_target = composite_at_target.masked_fill(~candidate_at_target, -1)

    global_max_val, global_max_idx = composite_base.max(dim=0)
    if numel >= 2:
        _top2_vals, top2_idxs = torch.topk(composite_base, k=2)
        second_max_val = _top2_vals[1]
        second_max_idx = top2_idxs[1]
    else:
        second_max_val = torch.tensor(-1.0, device=device)
        second_max_idx = torch.tensor(0, dtype=torch.int64, device=device)
    max_other = torch.where(
        flat_indices != global_max_idx,
        global_max_val,
        second_max_val,
    )
    has_candidate = (max_other >= 0) | (composite_at_target >= 0)
    use_target = composite_at_target >= max_other
    winner_from_base = torch.where(
        flat_indices != global_max_idx,
        global_max_idx,
        second_max_idx,
    )
    winner_col = torch.where(use_target, flat_indices, winner_from_base)

    winner_new_acc = torch.where(
        use_target,
        new_acc_at_target,
        base_new_acc[winner_col],
    )
    directions = torch.where(winner_new_acc >= threshold, 1, -1).to(torch.int32)
    target_q_before = q_flat[flat_indices]
    target_q_after = q_flat[flat_indices].clone()
    winner_eq_target = winner_col == flat_indices
    if bool(winner_eq_target.any().item()):
        target_q_after[winner_eq_target] = (
            q_flat[flat_indices[winner_eq_target]] + directions[winner_eq_target]
        ).clamp(-1, 1)
    delta_weights = (target_q_after - target_q_before).to(torch.float32) * frozen_scale
    return delta_weights.masked_fill(~has_candidate, 0.0)


def _ingress_candidate_ids_from_flat_indices_by_state(
    flat_indices_by_state: Mapping[str, torch.Tensor],
) -> list[str]:
    candidate_ids: list[str] = []
    for state_key in sorted(flat_indices_by_state):
        for flat_index in flat_indices_by_state[state_key].flatten().tolist():
            candidate_ids.append(f"{state_key}:{int(flat_index)}")
    return candidate_ids


INGRESS_CANDIDATE_HASH_BASIS_LEGACY_STRING_IDS = "legacy_sorted_candidate_id_strings_v0"
INGRESS_CANDIDATE_HASH_BASIS_TENSOR_STATE_KEY_FLAT_INDEX_V1 = (
    "tensor_state_key_flat_index_v1"
)


def _ingress_candidate_count_from_flat_indices_by_state(
    flat_indices_by_state: Mapping[str, torch.Tensor],
) -> int:
    return sum(
        int(flat_indices_by_state[state_key].numel())
        for state_key in sorted(flat_indices_by_state)
    )


def _tensor_native_ingress_candidate_ids_hash16(
    flat_indices_by_state: Mapping[str, torch.Tensor],
) -> str:
    """Deterministic hash over canonical state_key:flat_index without Python string materialization."""

    hasher = hashlib.sha256()
    for state_key in sorted(flat_indices_by_state):
        flat_indices = (
            flat_indices_by_state[state_key]
            .flatten()
            .to(dtype=torch.int64)
            .cpu()
            .contiguous()
        )
        hasher.update(str(state_key).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(flat_indices.numpy().tobytes())
        hasher.update(b"\0")
    return hasher.hexdigest()[:16]


def _candidate_ids_hash16(candidate_ids: Sequence[str]) -> str:
    values = sorted(str(candidate_id) for candidate_id in candidate_ids)
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()[:16]


def _top_k_overlap(
    *,
    candidate_ids: Sequence[str],
    proxy_by_id: Mapping[str, float],
    shadow_by_id: Mapping[str, float],
    k: int = 8,
) -> float:
    if not candidate_ids:
        return 0.0
    k_eff = min(int(k), len(candidate_ids))
    proxy_top = sorted(
        candidate_ids,
        key=lambda candidate_id: (
            float(proxy_by_id[str(candidate_id)]),
            str(candidate_id),
        ),
    )[:k_eff]
    shadow_top = sorted(
        candidate_ids,
        key=lambda candidate_id: (
            float(shadow_by_id[str(candidate_id)]),
            str(candidate_id),
        ),
    )[:k_eff]
    return len(set(proxy_top) & set(shadow_top)) / float(k_eff)


def _sign_agreement_fraction(
    proxy_values: Sequence[float],
    shadow_values: Sequence[float],
) -> float:
    if not proxy_values:
        return 0.0
    agreements = 0
    for proxy_value, shadow_value in zip(proxy_values, shadow_values):
        if proxy_value == 0.0 and shadow_value == 0.0:
            agreements += 1
        elif proxy_value == 0.0 or shadow_value == 0.0:
            continue
        elif (proxy_value > 0.0) == (shadow_value > 0.0):
            agreements += 1
    return float(agreements) / float(len(proxy_values))


def compute_grad_proxy_pass_bars(
    *,
    per_candidate: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    proxy_values = [float(row["local_loss_delta_proxy"]) for row in per_candidate]
    shadow_values = [float(row["local_loss_delta_shadow"]) for row in per_candidate]
    candidate_ids = [str(row["candidate_id"]) for row in per_candidate]
    proxy_by_id = {
        str(row["candidate_id"]): float(row["local_loss_delta_proxy"]) for row in per_candidate
    }
    shadow_by_id = {
        str(row["candidate_id"]): float(row["local_loss_delta_shadow"]) for row in per_candidate
    }
    tau_b, comparable_pairs, discordant_pairs = kendall_tau_b(proxy_values, shadow_values)
    return {
        "kendall_tau": float(tau_b),
        "kendall_comparable_pairs": int(comparable_pairs),
        "kendall_discordant_pairs": int(discordant_pairs),
        "top8_overlap": float(
            _top_k_overlap(
                candidate_ids=candidate_ids,
                proxy_by_id=proxy_by_id,
                shadow_by_id=shadow_by_id,
                k=8,
            )
        ),
        "sign_agreement_fraction": float(
            _sign_agreement_fraction(proxy_values, shadow_values)
        ),
    }


@contextlib.contextmanager
def _preserve_rng_state(*, device: torch.device) -> Iterator[None]:
    cpu_state = torch.get_rng_state()
    cuda_states: list[torch.Tensor] = []
    if device.type == "cuda":
        cuda_states = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(cuda_states)


def zero_fill_non_crossing_unmeasured_local_loss_deltas(
    *,
    local_loss_delta_by_key: dict[str, torch.Tensor],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
) -> None:
    for state_key, tensor in local_loss_delta_by_key.items():
        crossing_mask = _crossing_eligible_mask_tensor(
            votes=votes_by_key[state_key],
            state=tensor_states[state_key],
        )
        view = tensor.view(-1)
        view[~crossing_mask] = 0.0
        crossing_vals = view[crossing_mask]
        if crossing_vals.numel() > 0 and not bool(torch.isfinite(crossing_vals).all().item()):
            bad = (~torch.isfinite(crossing_vals)).nonzero(as_tuple=False)[0, 0].item()
            flat_index = int(crossing_mask.nonzero(as_tuple=False)[int(bad), 0].item())
            raise ValueError(
                "local_loss_delta_proxy_incomplete_coverage: "
                f"unmeasured_crossing_eligible_row state_key={state_key!r} "
                f"flat_index={flat_index}"
            )


def assert_local_loss_delta_proxy_coverage(
    *,
    local_loss_delta_by_key: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
) -> None:
    missing: list[str] = []
    for state_key, state in sorted(tensor_states.items()):
        crossing_mask = _crossing_eligible_mask_tensor(
            votes=votes_by_key[state_key],
            state=state,
        )
        view = local_loss_delta_by_key[state_key].view(-1)
        crossing_vals = view[crossing_mask]
        if crossing_vals.numel() == 0:
            continue
        bad = ~torch.isfinite(crossing_vals)
        if bool(bad.any().item()):
            bad_positions = crossing_mask.nonzero(as_tuple=False).flatten()
            for position in bad_positions[bad]:
                missing.append(f"{state_key}:{int(position)}")
    if missing:
        raise ValueError(
            "local_loss_delta_proxy_incomplete_coverage: "
            f"unmeasured_crossing_eligible_rows={missing}"
        )


def build_grad_proxy_local_loss_delta_by_key(
    *,
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    eligible_modules: Mapping[str, BitLinear],
    device: torch.device,
    extras: Mapping[str, Any],
    votes_by_key: Mapping[str, torch.Tensor],
    max_abs_per_tensor: int,
    population_mode: str = POPULATION_MODE_FULL_CROSSING_ELIGIBLE,
    max_sampled_candidates: int = 64,
    phase_progress: Any | None = None,
    optimizer_step_index: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    gather_start = time.perf_counter()
    normalize_start = time.perf_counter()
    normalized_votes_by_key = _normalize_votes_by_key(votes_by_key)
    normalize_seconds = float(time.perf_counter() - normalize_start)
    is_full_crossing = str(population_mode) == POPULATION_MODE_FULL_CROSSING_ELIGIBLE
    crossing_count_by_state_key = crossing_count_by_state_key_from_votes(
        tensor_states=tensor_states,
        votes_by_key=normalized_votes_by_key,
    )
    universe_timing: dict[str, float] = {}
    candidate_by_id: dict[str, dict[str, Any]] = {}
    flat_indices_by_state: dict[str, torch.Tensor] = {}
    ingress_candidate_count = 0
    ingress_ids_hash16 = _candidate_ids_hash16([])
    ingress_hash_basis = INGRESS_CANDIDATE_HASH_BASIS_LEGACY_STRING_IDS
    ingress_ids: list[str] = []
    universe_start = time.perf_counter()
    if is_full_crossing:
        flat_indices_by_state = _crossing_flat_indices_by_state_key(
            tensor_states=tensor_states,
            votes_by_key=normalized_votes_by_key,
        )
        crossing_eligible_count = _ingress_candidate_count_from_flat_indices_by_state(
            flat_indices_by_state
        )
        ingress_candidate_count = int(crossing_eligible_count)
        ingress_ids_hash16 = _tensor_native_ingress_candidate_ids_hash16(
            flat_indices_by_state
        )
        ingress_hash_basis = INGRESS_CANDIDATE_HASH_BASIS_TENSOR_STATE_KEY_FLAT_INDEX_V1
        universe_timing = {
            "universe_crossing_sets": 0.0,
            "universe_candidate_dict": 0.0,
            "universe_sort_sample": 0.0,
            "universe_build_total": float(time.perf_counter() - universe_start),
        }
    else:
        universe = build_w6_t10_crossing_candidate_universe_from_votes(
            tensor_states=tensor_states,
            votes_by_key=normalized_votes_by_key,
            max_abs_per_tensor=int(max_abs_per_tensor),
            max_sampled_candidates=int(max_sampled_candidates),
            population_mode=str(population_mode),
            timing_out=universe_timing,
        )
        candidate_by_id = universe["candidate_by_id"]
        ingress_ids = list(universe["sampled_ids"])
        crossing_eligible_count = int(universe["crossing_eligible_count"])
        ingress_candidate_count = int(len(ingress_ids))
        ingress_ids_hash16 = _candidate_ids_hash16(ingress_ids)
        ingress_hash_basis = INGRESS_CANDIDATE_HASH_BASIS_LEGACY_STRING_IDS
    universe_seconds = float(time.perf_counter() - universe_start)
    if ingress_candidate_count == 0:
        local_loss_delta_by_key = {
            state_key: torch.zeros(votes.shape, dtype=torch.float32)
            for state_key, votes in normalized_votes_by_key.items()
        }
        gather_seconds = float(time.perf_counter() - gather_start)
        ingress_receipt: dict[str, Any] = {
            "grad_proxy_ingress_enabled": True,
            "grad_proxy_ingress_estimand": GRAD_PROXY_AUDIT_ESTIMAND,
            "grad_proxy_ingress_population_mode": str(population_mode),
            "grad_proxy_ingress_crossing_eligible_count": int(crossing_eligible_count),
            "crossing_count_by_state_key": dict(crossing_count_by_state_key),
            "grad_proxy_ingress_candidate_count_ingressed": 0,
            "grad_proxy_ingress_candidate_ids_hash16": _candidate_ids_hash16([]),
            "grad_proxy_ingress_candidate_hash_basis": (
                INGRESS_CANDIDATE_HASH_BASIS_TENSOR_STATE_KEY_FLAT_INDEX_V1
                if is_full_crossing
                else INGRESS_CANDIDATE_HASH_BASIS_LEGACY_STRING_IDS
            ),
            "grad_proxy_ingress_gather_seconds": gather_seconds,
            "grad_proxy_gather_seconds": gather_seconds,
            "candidate_count_ingressed": 0,
            "activation_credit_gather_telemetry_note": (
                ACTIVATION_CREDIT_GATHER_TELEMETRY_NOTE
            ),
            "grad_proxy_ingress_phase_seconds": {
                "normalize_votes": normalize_seconds,
                "universe_build": universe_seconds,
                "universe_crossing_sets": float(
                    universe_timing.get("universe_crossing_sets", 0.0)
                ),
                "universe_candidate_dict": float(
                    universe_timing.get("universe_candidate_dict", 0.0)
                ),
                "universe_sort_sample": float(
                    universe_timing.get("universe_sort_sample", 0.0)
                ),
                "flat_index_python_conversion": 0.0,
                "proxy_forward_backward": 0.0,
                "proxy_gather": 0.0,
                "proxy_total": 0.0,
                "delta_weight_scatter": 0.0,
                "coverage": 0.0,
                "total": gather_seconds,
            },
        }
        if optimizer_step_index is not None:
            ingress_receipt["optimizer_step_index"] = int(optimizer_step_index)
        return (
            {
                state_key: tensor.detach().cpu().contiguous()
                for state_key, tensor in local_loss_delta_by_key.items()
            },
            ingress_receipt,
        )
    proxy_start = time.perf_counter()
    local_loss_delta_by_key: dict[str, torch.Tensor] = {
        state_key: torch.full(votes.shape, float("nan"), dtype=torch.float32)
        for state_key, votes in normalized_votes_by_key.items()
    }
    if is_full_crossing:
        flat_index_python_conversion_seconds = 0.0
        proxy_receipt = _compute_activation_credit_candidate_proxies(
            model=model,
            batch=batch,
            tensor_states=tensor_states,
            eligible_modules=eligible_modules,
            device=device,
            extras=extras,
            candidate_by_id={},
            selected_candidate_ids=[],
            flat_indices_by_state_tensors=flat_indices_by_state,
            materialize_proxy_dict=False,
            phase_progress=phase_progress,
            optimizer_step_index=optimizer_step_index,
        )
    else:
        flat_index_python_conversion_seconds = 0.0
        proxy_receipt = _compute_activation_credit_candidate_proxies(
            model=model,
            batch=batch,
            tensor_states=tensor_states,
            eligible_modules=eligible_modules,
            device=device,
            extras=extras,
            candidate_by_id=candidate_by_id,
            selected_candidate_ids=ingress_ids,
            phase_progress=phase_progress,
            optimizer_step_index=optimizer_step_index,
        )
    proxy_fb_seconds = float(proxy_receipt.get("forward_backward_seconds", 0.0))
    proxy_gather_seconds = float(proxy_receipt.get("grad_proxy_accumulation_seconds", 0.0))
    proxy_seconds = float(time.perf_counter() - proxy_start)
    delta_scatter_start = time.perf_counter()
    if is_full_crossing:
        tensor_bundles = proxy_receipt.get("grad_proxy_tensors_by_state") or {}
        for state_key, flat_indices_tensor in sorted(flat_indices_by_state.items()):
            bundle = tensor_bundles.get(state_key)
            if bundle is None:
                raise ValueError(
                    "grad_proxy_ingress_fail_closed: "
                    f"missing grad_proxy_tensors_by_state for {state_key!r}"
                )
            _scatter_vectorized_grad_proxy_ingress_for_state(
                local_loss_delta=local_loss_delta_by_key[state_key],
                tensor_state=tensor_states[state_key],
                votes=normalized_votes_by_key[state_key],
                flat_indices=bundle["flat_indices"],
                grad_proxies=bundle["proxies"],
                max_abs_per_tensor=int(max_abs_per_tensor),
            )
    else:
        grad_proxy_by_id = proxy_receipt["grad_proxy_by_candidate_id"]
        for candidate_id in ingress_ids:
            candidate = candidate_by_id[candidate_id]
            state_key = str(candidate["state_key"])
            flat_index = int(candidate["flat_index"])
            grad_proxy = float(grad_proxy_by_id[str(candidate_id)])
            candidate_delta_weight = derive_w6_t10_candidate_delta_weight(
                tensor_state=tensor_states[state_key],
                votes=normalized_votes_by_key[state_key],
                candidate=candidate,
                max_abs_per_tensor=int(max_abs_per_tensor),
            )
            local_loss_delta_by_key[state_key].view(-1)[flat_index] = float(
                grad_proxy * candidate_delta_weight
            )
    delta_scatter_seconds = float(time.perf_counter() - delta_scatter_start)
    coverage_start = time.perf_counter()
    zero_fill_non_crossing_unmeasured_local_loss_deltas(
        local_loss_delta_by_key=local_loss_delta_by_key,
        tensor_states=tensor_states,
        votes_by_key=normalized_votes_by_key,
    )
    assert_local_loss_delta_proxy_coverage(
        local_loss_delta_by_key=local_loss_delta_by_key,
        tensor_states=tensor_states,
        votes_by_key=normalized_votes_by_key,
    )
    coverage_seconds = float(time.perf_counter() - coverage_start)
    gather_seconds = float(time.perf_counter() - gather_start)
    ingress_receipt: dict[str, Any] = {
        "grad_proxy_ingress_enabled": True,
        "grad_proxy_ingress_estimand": GRAD_PROXY_AUDIT_ESTIMAND,
        "grad_proxy_ingress_population_mode": str(population_mode),
        "grad_proxy_ingress_crossing_eligible_count": int(crossing_eligible_count),
        "crossing_count_by_state_key": dict(crossing_count_by_state_key),
        "grad_proxy_ingress_candidate_count_ingressed": int(ingress_candidate_count),
        "grad_proxy_ingress_candidate_ids_hash16": str(ingress_ids_hash16),
        "grad_proxy_ingress_candidate_hash_basis": str(ingress_hash_basis),
        "grad_proxy_ingress_full_crossing_fast_path": bool(is_full_crossing),
        "grad_proxy_ingress_gather_seconds": gather_seconds,
        "grad_proxy_gather_seconds": gather_seconds,
        "candidate_count_ingressed": int(ingress_candidate_count),
        "activation_credit_gather_telemetry_note": ACTIVATION_CREDIT_GATHER_TELEMETRY_NOTE,
        "cuda_memory_snapshots": list(proxy_receipt.get("cuda_memory_snapshots") or []),
        "grad_proxy_ingress_phase_seconds": {
            "normalize_votes": normalize_seconds,
            "universe_build": universe_seconds,
            "universe_crossing_sets": float(
                universe_timing.get("universe_crossing_sets", 0.0)
            ),
            "universe_candidate_dict": float(
                universe_timing.get("universe_candidate_dict", 0.0)
            ),
            "universe_sort_sample": float(
                universe_timing.get("universe_sort_sample", 0.0)
            ),
            "flat_index_python_conversion": float(flat_index_python_conversion_seconds),
            "proxy_forward_backward": proxy_fb_seconds,
            "proxy_gather": proxy_gather_seconds,
            "proxy_total": proxy_seconds,
            "delta_weight_scatter": delta_scatter_seconds,
            "coverage": coverage_seconds,
            "total": gather_seconds,
        },
    }
    if optimizer_step_index is not None:
        ingress_receipt["optimizer_step_index"] = int(optimizer_step_index)
    return (
        {
            state_key: tensor.detach().cpu().contiguous()
            for state_key, tensor in local_loss_delta_by_key.items()
        },
        ingress_receipt,
    )


def _shadow_local_loss_deltas_for_candidates(
    *,
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    eligible_modules: Mapping[str, BitLinear],
    device: torch.device,
    extras: Mapping[str, Any],
    votes_by_key: Mapping[str, torch.Tensor],
    candidate_by_id: Mapping[str, Mapping[str, Any]],
    sampled_ids: Sequence[str],
    baseline_loss: float,
    max_abs_per_tensor: int,
) -> list[dict[str, Any]]:
    base_spec = w6_t10_base_spec(max_abs_per_tensor=int(max_abs_per_tensor))
    one_flip_spec = _single_flip_spec(base_spec)
    normalized_votes_by_key = _normalize_votes_by_key(votes_by_key)
    per_candidate: list[dict[str, Any]] = []
    for candidate_id in sampled_ids:
        candidate = dict(candidate_by_id[candidate_id])
        state_key = str(candidate["state_key"])
        flat_index = int(candidate["flat_index"])
        grad_proxy = 0.0
        candidate_delta_weight = derive_w6_t10_candidate_delta_weight(
            tensor_state=tensor_states[state_key],
            votes=normalized_votes_by_key[state_key],
            candidate=candidate,
            max_abs_per_tensor=int(max_abs_per_tensor),
        )
        q_levels, accumulators = _apply_full_vote_planned_candidate_shadow_update(
            prior_state=tensor_states[state_key],
            candidate=candidate,
            threshold_abs=int(base_spec.threshold_abs),
        )
        candidate_states = dict(tensor_states)
        candidate_states[state_key] = make_live_shadow_tensor_state(
            tensor_states[state_key],
            q_levels,
            accumulators,
        )
        candidate_loss = _evaluate_loss(
            model,
            batch,
            candidate_states,
            eligible_modules,
            device=device,
            extras=extras,
        )
        local_loss_delta_shadow = float(candidate_loss - float(baseline_loss))
        local_loss_delta_proxy = float(grad_proxy * candidate_delta_weight)
        per_candidate.append(
            {
                "candidate_id": str(candidate_id),
                "state_key": state_key,
                "flat_index": flat_index,
                "grad_proxy": grad_proxy,
                "candidate_delta_weight": float(candidate_delta_weight),
                "local_loss_delta_proxy": local_loss_delta_proxy,
                "local_loss_delta_shadow": local_loss_delta_shadow,
            }
        )
    return per_candidate


def run_proxy_oracle_drift_audit(
    *,
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    eligible_modules: Mapping[str, BitLinear],
    device: torch.device,
    extras: Mapping[str, Any],
    votes_by_key: Mapping[str, torch.Tensor],
    local_loss_delta_by_key: Mapping[str, torch.Tensor],
    baseline_loss: float,
    max_abs_per_tensor: int,
    optimizer_step_index: int,
    drift_sample_count: int = DRIFT_AUDIT_SAMPLE_COUNT,
) -> dict[str, Any]:
    normalized_votes_by_key = _normalize_votes_by_key(votes_by_key)
    universe = build_w6_t10_crossing_candidate_universe_from_votes(
        tensor_states=tensor_states,
        votes_by_key=normalized_votes_by_key,
        max_abs_per_tensor=int(max_abs_per_tensor),
        max_sampled_candidates=int(drift_sample_count),
        population_mode=POPULATION_MODE_SAMPLED_K64,
    )
    candidate_by_id = universe["candidate_by_id"]
    drift_ids = list(universe["sampled_ids"])
    per_candidate: list[dict[str, Any]] = []
    with _preserve_rng_state(device=device), torch.no_grad():
        shadow_rows = _shadow_local_loss_deltas_for_candidates(
            model=model,
            batch=batch,
            tensor_states=tensor_states,
            eligible_modules=eligible_modules,
            device=device,
            extras=extras,
            votes_by_key=normalized_votes_by_key,
            candidate_by_id=candidate_by_id,
            sampled_ids=drift_ids,
            baseline_loss=float(baseline_loss),
            max_abs_per_tensor=int(max_abs_per_tensor),
        )
        for row in shadow_rows:
            candidate_id = str(row["candidate_id"])
            state_key = str(row["state_key"])
            flat_index = int(row["flat_index"])
            proxy_value = float(
                local_loss_delta_by_key[state_key].view(-1)[flat_index].item()
            )
            per_candidate.append(
                {
                    **row,
                    "local_loss_delta_proxy": proxy_value,
                    "local_loss_delta_shadow": float(row["local_loss_delta_shadow"]),
                }
            )
    pass_bars = compute_grad_proxy_pass_bars(per_candidate=per_candidate)
    return {
        "proxy_oracle_drift_step": int(optimizer_step_index),
        "proxy_oracle_drift_sample_count": len(drift_ids),
        "proxy_oracle_drift_candidate_ids_hash16": _candidate_ids_hash16(drift_ids),
        "proxy_oracle_drift_tau": float(pass_bars["kendall_tau"]),
        "proxy_oracle_drift_sign_agreement": float(
            pass_bars["sign_agreement_fraction"]
        ),
        "proxy_oracle_drift_top8_overlap": float(pass_bars["top8_overlap"]),
        "proxy_oracle_drift_gating": False,
        "proxy_oracle_drift_comparator_spec": GRAD_PROXY_AUDIT_COMPARATOR_SPEC,
        "proxy_oracle_drift_estimand": GRAD_PROXY_AUDIT_ESTIMAND,
        "per_candidate": per_candidate,
    }


def run_grad_proxy_audit_at_anchor(
    *,
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
    baseline_loss: float,
    eligible_modules: Mapping[str, BitLinear],
    device: torch.device,
    extras: Mapping[str, Any],
    max_abs_per_tensor: int,
    max_audit_candidates: int,
    launch_sha: str,
    audit_step_index: int,
    audit_warmup_steps_run: int,
    crossing_eligible_count_by_step: Sequence[int],
    comparator_spec: str = GRAD_PROXY_AUDIT_COMPARATOR_SPEC,
) -> dict[str, Any]:
    if int(max_audit_candidates) <= 0:
        raise ValueError("max_audit_candidates must be positive")
    if int(audit_step_index) <= 0:
        raise ValueError("audit_step_index must be positive")
    if int(audit_warmup_steps_run) < 0:
        raise ValueError("audit_warmup_steps_run must be non-negative")
    comparator_spec_mismatch = (
        str(comparator_spec) != GRAD_PROXY_AUDIT_COMPARATOR_SPEC
    )
    audit_start = time.perf_counter()
    normalized_votes_by_key = _normalize_votes_by_key(votes_by_key)
    universe = build_w6_t10_crossing_candidate_universe_from_votes(
        tensor_states=tensor_states,
        votes_by_key=normalized_votes_by_key,
        max_abs_per_tensor=int(max_abs_per_tensor),
        max_sampled_candidates=int(max_audit_candidates),
        population_mode=POPULATION_MODE_SAMPLED_K64,
    )
    base_spec = universe["base_spec"]
    one_flip_spec = universe["one_flip_spec"]
    candidate_by_id = universe["candidate_by_id"]
    sampled_ids = list(universe["sampled_ids"])
    if not sampled_ids:
        raise GradProxyAuditAborted(
            "grad_proxy_audit_aborted: no crossing-eligible candidates sampled"
        )
    for candidate_id in sampled_ids:
        candidate = candidate_by_id[candidate_id]
        audit = _audit_sparse_singleton_identity_for_candidate(
            tensor_state=tensor_states[str(candidate["state_key"])],
            votes=normalized_votes_by_key[str(candidate["state_key"])],
            candidate=candidate,
            one_flip_spec=one_flip_spec,
        )
        if bool(audit["drifted"]):
            raise GradProxyAuditAborted(
                "grad_proxy_audit_aborted: singleton_identity_drift=true "
                f"expected={audit['expected_flat_index']} "
                f"applied={audit['applied_indices']}"
            )
    proxy_receipt = _compute_activation_credit_candidate_proxies(
        model=model,
        batch=batch,
        tensor_states=tensor_states,
        eligible_modules=eligible_modules,
        device=device,
        extras=extras,
        candidate_by_id=candidate_by_id,
        selected_candidate_ids=sampled_ids,
        phase_progress=None,
    )
    per_candidate: list[dict[str, Any]] = []
    for candidate_id in sampled_ids:
        candidate = dict(candidate_by_id[candidate_id])
        state_key = str(candidate["state_key"])
        flat_index = int(candidate["flat_index"])
        grad_proxy = float(
            proxy_receipt["grad_proxy_by_candidate_id"][str(candidate_id)]
        )
        candidate_delta_weight = derive_w6_t10_candidate_delta_weight(
            tensor_state=tensor_states[state_key],
            votes=normalized_votes_by_key[state_key],
            candidate=candidate,
            max_abs_per_tensor=int(max_abs_per_tensor),
        )
        q_levels, accumulators = _apply_full_vote_planned_candidate_shadow_update(
            prior_state=tensor_states[state_key],
            candidate=candidate,
            threshold_abs=int(base_spec.threshold_abs),
        )
        candidate_states = dict(tensor_states)
        candidate_states[state_key] = make_live_shadow_tensor_state(
            tensor_states[state_key],
            q_levels,
            accumulators,
        )
        candidate_loss = _evaluate_loss(
            model,
            batch,
            candidate_states,
            eligible_modules,
            device=device,
            extras=extras,
        )
        local_loss_delta_shadow = float(candidate_loss - float(baseline_loss))
        local_loss_delta_proxy = float(grad_proxy * candidate_delta_weight)
        per_candidate.append(
            {
                "candidate_id": str(candidate_id),
                "state_key": state_key,
                "flat_index": flat_index,
                "grad_proxy": grad_proxy,
                "candidate_delta_weight": float(candidate_delta_weight),
                "local_loss_delta_proxy": local_loss_delta_proxy,
                "local_loss_delta_shadow": local_loss_delta_shadow,
                "candidate_apply_policy": B2B_CANDIDATE_APPLY_POLICY,
            }
        )
    receipt: dict[str, Any] = {
        "schema": GRAD_PROXY_AUDIT_SCHEMA,
        "estimand": GRAD_PROXY_AUDIT_ESTIMAND,
        "comparator_spec": str(comparator_spec),
        "comparator_spec_mismatch": bool(comparator_spec_mismatch),
        "optimizer_step_index": int(audit_step_index),
        "audit_state_source": GRAD_PROXY_AUDIT_STATE_SOURCE,
        "audit_warmup_steps_run": int(audit_warmup_steps_run),
        "audit_step_index": int(audit_step_index),
        "warmup_two_tier_enabled": False,
        "crossing_eligible_count_by_step": [
            int(value) for value in crossing_eligible_count_by_step
        ],
        "baseline_loss": float(baseline_loss),
        "crossing_eligible_count": int(universe["crossing_eligible_count"]),
        "sampled_candidate_count": len(sampled_ids),
        "max_audit_candidates": int(max_audit_candidates),
        "sampled_candidate_ids_hash16": _candidate_ids_hash16(sampled_ids),
        "launch_sha": str(launch_sha),
        "threshold_abs": int(CROSSING_THRESHOLD_ABS),
        "per_candidate": per_candidate,
        "duration_seconds": float(time.perf_counter() - audit_start),
    }
    if comparator_spec_mismatch:
        receipt["pass_bars"] = None
    else:
        receipt["pass_bars"] = compute_grad_proxy_pass_bars(per_candidate=per_candidate)
    return receipt


@dataclass(frozen=True)
class ProbeWarmupAuditAnchor:
    audit_step_index: int
    audit_warmup_steps_run: int
    crossing_eligible_count_by_step: tuple[int, ...]
    crossing_eligible_count: int
    tensor_states: dict[str, BoundedDeltaTensorState]
    votes_by_key: dict[str, torch.Tensor]
    baseline_loss: float


def discover_probe_warmup_audit_anchor(
    *,
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    eligible_modules: Mapping[str, BitLinear],
    device: torch.device,
    extras: Mapping[str, Any],
    max_abs_per_tensor: int,
    launch_sha: str,
    parent_sha256: str | None = None,
    warmup_max_steps: int = DEFAULT_WARMUP_MAX_STEPS,
) -> ProbeWarmupAuditAnchor:
    warmup_cap = min(int(warmup_max_steps), int(MAX_WARMUP_MAX_STEPS))
    if warmup_cap <= 0:
        raise ValueError("warmup_max_steps must be positive")
    states = dict(tensor_states)
    crossing_eligible_count_by_step: list[int] = []
    for step_index in range(1, warmup_cap + 1):
        baseline_loss, votes_by_key = derive_probe_science_arm_votes(
            model=model,
            batch=batch,
            tensor_states=states,
            eligible_modules=eligible_modules,
            device=device,
            extras=extras,
            max_abs_per_tensor=int(max_abs_per_tensor),
        )
        crossing_count = count_w6_t10_crossing_eligible_from_votes(
            tensor_states=states,
            votes_by_key=votes_by_key,
        )
        crossing_eligible_count_by_step.append(int(crossing_count))
        if crossing_count > 0:
            return ProbeWarmupAuditAnchor(
                audit_step_index=int(step_index),
                audit_warmup_steps_run=int(step_index - 1),
                crossing_eligible_count_by_step=tuple(crossing_eligible_count_by_step),
                crossing_eligible_count=int(crossing_count),
                tensor_states=dict(states),
                votes_by_key={key: tensor for key, tensor in votes_by_key.items()},
                baseline_loss=float(baseline_loss),
            )
        if step_index == warmup_cap:
            raise GradProxyAuditWarmupCapAborted(
                crossing_eligible_count_by_step=crossing_eligible_count_by_step,
                warmup_steps_run=int(warmup_cap),
                launch_sha=str(launch_sha),
                parent_sha256=parent_sha256,
            )
        states = apply_probe_off_path_warmup_step(
            tensor_states=states,
            votes_by_key=votes_by_key,
            max_abs_per_tensor=int(max_abs_per_tensor),
            warmup_step_index=int(step_index),
        )
    raise RuntimeError("warmup loop exhausted without audit anchor or abort")


def run_grad_proxy_audit_with_warmup(
    *,
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    eligible_modules: Mapping[str, BitLinear],
    device: torch.device,
    extras: Mapping[str, Any],
    max_abs_per_tensor: int,
    max_audit_candidates: int,
    launch_sha: str,
    parent_sha256: str | None = None,
    comparator_spec: str = GRAD_PROXY_AUDIT_COMPARATOR_SPEC,
    warmup_max_steps: int = DEFAULT_WARMUP_MAX_STEPS,
) -> dict[str, Any]:
    anchor = discover_probe_warmup_audit_anchor(
        model=model,
        batch=batch,
        tensor_states=tensor_states,
        eligible_modules=eligible_modules,
        device=device,
        extras=extras,
        max_abs_per_tensor=int(max_abs_per_tensor),
        launch_sha=str(launch_sha),
        parent_sha256=parent_sha256,
        warmup_max_steps=int(warmup_max_steps),
    )
    return run_grad_proxy_audit_at_anchor(
        model=model,
        batch=batch,
        tensor_states=anchor.tensor_states,
        votes_by_key=anchor.votes_by_key,
        baseline_loss=anchor.baseline_loss,
        eligible_modules=eligible_modules,
        device=device,
        extras=extras,
        max_abs_per_tensor=int(max_abs_per_tensor),
        max_audit_candidates=int(max_audit_candidates),
        launch_sha=str(launch_sha),
        audit_step_index=anchor.audit_step_index,
        audit_warmup_steps_run=anchor.audit_warmup_steps_run,
        crossing_eligible_count_by_step=list(anchor.crossing_eligible_count_by_step),
        comparator_spec=str(comparator_spec),
    )


def run_grad_proxy_audit_step1(
    *,
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    eligible_modules: Mapping[str, BitLinear],
    device: torch.device,
    extras: Mapping[str, Any],
    max_abs_per_tensor: int,
    max_audit_candidates: int,
    launch_sha: str,
    parent_sha256: str | None = None,
    comparator_spec: str = GRAD_PROXY_AUDIT_COMPARATOR_SPEC,
    warmup_max_steps: int = DEFAULT_WARMUP_MAX_STEPS,
) -> dict[str, Any]:
    return run_grad_proxy_audit_with_warmup(
        model=model,
        batch=batch,
        tensor_states=tensor_states,
        eligible_modules=eligible_modules,
        device=device,
        extras=extras,
        max_abs_per_tensor=int(max_abs_per_tensor),
        max_audit_candidates=int(max_audit_candidates),
        launch_sha=str(launch_sha),
        parent_sha256=parent_sha256,
        comparator_spec=str(comparator_spec),
        warmup_max_steps=int(warmup_max_steps),
    )


def resolve_launch_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_grad_proxy_audit_receipt(
    receipt: Mapping[str, Any],
    *,
    artifact_dir: str | Any,
) -> str:
    from pathlib import Path

    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / GRAD_PROXY_AUDIT_RECEIPT_NAME
    out_path.write_text(
        json.dumps(dict(receipt), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(out_path)


def write_grad_proxy_audit_abort_receipt(
    *,
    artifact_dir: str | Any,
    crossing_eligible_count_by_step: Sequence[int],
    warmup_steps_run: int,
    launch_sha: str,
    parent_sha256: str | None,
) -> str:
    from pathlib import Path

    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "reason": GRAD_PROXY_AUDIT_ABORT_REASON,
        "crossing_eligible_count_by_step": [
            int(value) for value in crossing_eligible_count_by_step
        ],
        "audit_state_source": GRAD_PROXY_AUDIT_STATE_SOURCE,
        "warmup_steps_run": int(warmup_steps_run),
        "warmup_two_tier_enabled": False,
        "launch_sha": str(launch_sha),
        "parent_sha256": (
            None if parent_sha256 is None else str(parent_sha256)
        ),
    }
    out_path = out_dir / GRAD_PROXY_AUDIT_ABORT_NAME
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(out_path)
