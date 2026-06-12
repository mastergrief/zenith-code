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
    apply_integer_vote_update_reference,
)

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


def build_w6_t10_crossing_candidate_universe_from_votes(
    *,
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
    max_abs_per_tensor: int,
    max_sampled_candidates: int,
    population_mode: str = POPULATION_MODE_SAMPLED_K64,
) -> dict[str, Any]:
    base_spec = w6_t10_base_spec(max_abs_per_tensor=int(max_abs_per_tensor))
    one_flip_spec = _single_flip_spec(base_spec)
    candidate_by_id: dict[str, dict[str, Any]] = {}
    crossing_eligible_count = 0
    for state_key, state in sorted(tensor_states.items()):
        vote_state = state.vote_update_state()
        votes = votes_by_key[state_key]
        crossing_mask = _crossing_eligible_mask_tensor(votes=votes, state=state)
        if not bool(crossing_mask.any().item()):
            continue
        crossing_eligible = {
            int(flat_index)
            for flat_index in crossing_mask.nonzero(as_tuple=False).flatten().tolist()
        }
        crossing_eligible_count += len(crossing_eligible)
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
        for flat_index in filtered_unordered:
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
    return {
        "base_spec": base_spec,
        "one_flip_spec": one_flip_spec,
        "votes_by_key": votes_by_key,
        "candidate_by_id": candidate_by_id,
        "sampled_ids": sampled_ids,
        "crossing_eligible_count": int(crossing_eligible_count),
        "population_mode": str(population_mode),
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
    normalized_votes_by_key = _normalize_votes_by_key(votes_by_key)
    universe = build_w6_t10_crossing_candidate_universe_from_votes(
        tensor_states=tensor_states,
        votes_by_key=normalized_votes_by_key,
        max_abs_per_tensor=int(max_abs_per_tensor),
        max_sampled_candidates=int(max_sampled_candidates),
        population_mode=str(population_mode),
    )
    candidate_by_id = universe["candidate_by_id"]
    ingress_ids = list(universe["sampled_ids"])
    crossing_count_by_state_key = crossing_count_by_state_key_from_votes(
        tensor_states=tensor_states,
        votes_by_key=normalized_votes_by_key,
    )
    if not ingress_ids:
        local_loss_delta_by_key = {
            state_key: torch.zeros(votes.shape, dtype=torch.float32)
            for state_key, votes in normalized_votes_by_key.items()
        }
        gather_seconds = float(time.perf_counter() - gather_start)
        ingress_receipt: dict[str, Any] = {
            "grad_proxy_ingress_enabled": True,
            "grad_proxy_ingress_estimand": GRAD_PROXY_AUDIT_ESTIMAND,
            "grad_proxy_ingress_population_mode": str(population_mode),
            "grad_proxy_ingress_crossing_eligible_count": int(
                universe["crossing_eligible_count"]
            ),
            "crossing_count_by_state_key": dict(crossing_count_by_state_key),
            "grad_proxy_ingress_candidate_count_ingressed": 0,
            "grad_proxy_ingress_candidate_ids_hash16": _candidate_ids_hash16([]),
            "grad_proxy_ingress_gather_seconds": gather_seconds,
            "grad_proxy_gather_seconds": gather_seconds,
            "candidate_count_ingressed": 0,
            "activation_credit_gather_telemetry_note": (
                ACTIVATION_CREDIT_GATHER_TELEMETRY_NOTE
            ),
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
    universe_seconds = float(time.perf_counter() - gather_start)
    proxy_start = time.perf_counter()
    local_loss_delta_by_key: dict[str, torch.Tensor] = {
        state_key: torch.full(votes.shape, float("nan"), dtype=torch.float32)
        for state_key, votes in normalized_votes_by_key.items()
    }
    if str(population_mode) == POPULATION_MODE_FULL_CROSSING_ELIGIBLE:
        flat_indices_by_state = _crossing_flat_indices_by_state_key(
            tensor_states=tensor_states,
            votes_by_key=normalized_votes_by_key,
        )
        proxy_receipt = _compute_activation_credit_candidate_proxies(
            model=model,
            batch=batch,
            tensor_states=tensor_states,
            eligible_modules=eligible_modules,
            device=device,
            extras=extras,
            candidate_by_id=candidate_by_id,
            selected_candidate_ids=[],
            flat_indices_by_state={
                state_key: [
                    int(flat_index)
                    for flat_index in flat_indices_tensor.tolist()
                ]
                for state_key, flat_indices_tensor in flat_indices_by_state.items()
            },
            materialize_proxy_dict=False,
            phase_progress=phase_progress,
            optimizer_step_index=optimizer_step_index,
        )
    else:
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
    if str(population_mode) == POPULATION_MODE_FULL_CROSSING_ELIGIBLE:
        tensor_bundles = proxy_receipt.get("grad_proxy_tensors_by_state") or {}
        if tensor_bundles:
            for state_key, bundle in sorted(tensor_bundles.items()):
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
            ingress_by_state: dict[str, list[str]] = {}
            for candidate_id in ingress_ids:
                state_key = str(candidate_by_id[candidate_id]["state_key"])
                ingress_by_state.setdefault(state_key, []).append(str(candidate_id))
            for state_key, state_candidate_ids in ingress_by_state.items():
                flat_indices = torch.tensor(
                    [
                        int(candidate_by_id[candidate_id]["flat_index"])
                        for candidate_id in state_candidate_ids
                    ],
                    dtype=torch.int64,
                )
                grad_proxies = torch.tensor(
                    [
                        float(grad_proxy_by_id[candidate_id])
                        for candidate_id in state_candidate_ids
                    ],
                    dtype=torch.float32,
                )
                _scatter_vectorized_grad_proxy_ingress_for_state(
                    local_loss_delta=local_loss_delta_by_key[state_key],
                    tensor_state=tensor_states[state_key],
                    votes=normalized_votes_by_key[state_key],
                    flat_indices=flat_indices,
                    grad_proxies=grad_proxies,
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
        "grad_proxy_ingress_crossing_eligible_count": int(
            universe["crossing_eligible_count"]
        ),
        "crossing_count_by_state_key": dict(crossing_count_by_state_key),
        "grad_proxy_ingress_candidate_count_ingressed": len(ingress_ids),
        "grad_proxy_ingress_candidate_ids_hash16": _candidate_ids_hash16(ingress_ids),
        "grad_proxy_ingress_gather_seconds": gather_seconds,
        "grad_proxy_gather_seconds": gather_seconds,
        "candidate_count_ingressed": len(ingress_ids),
        "activation_credit_gather_telemetry_note": ACTIVATION_CREDIT_GATHER_TELEMETRY_NOTE,
        "cuda_memory_snapshots": list(proxy_receipt.get("cuda_memory_snapshots") or []),
        "grad_proxy_ingress_phase_seconds": {
            "universe": universe_seconds,
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
