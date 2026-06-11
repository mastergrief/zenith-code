"""Read-only W6/T=10 grad-proxy audit for M-A precursor (slice 3a)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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
) -> dict[str, Any]:
    base_spec = w6_t10_base_spec(max_abs_per_tensor=int(max_abs_per_tensor))
    one_flip_spec = _single_flip_spec(base_spec)
    candidate_by_id: dict[str, dict[str, Any]] = {}
    crossing_eligible_count = 0
    for state_key, state in sorted(tensor_states.items()):
        vote_state = state.vote_update_state()
        votes = votes_by_key[state_key]
        rows = materialize_selector_rows(votes=votes, state=state)
        crossing_eligible = set(
            crossing_eligible_flat_indices(rows, threshold_abs=int(CROSSING_THRESHOLD_ABS))
        )
        crossing_eligible_count += len(crossing_eligible)
        if not crossing_eligible:
            continue
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
    sampled_ids = _sample_candidate_ids(
        current_ordered_ids,
        deterministic_ordered_ids,
        max_sampled_candidates=int(max_sampled_candidates),
    )
    return {
        "base_spec": base_spec,
        "one_flip_spec": one_flip_spec,
        "votes_by_key": votes_by_key,
        "candidate_by_id": candidate_by_id,
        "sampled_ids": sampled_ids,
        "crossing_eligible_count": int(crossing_eligible_count),
    }


def count_w6_t10_crossing_eligible_from_votes(
    *,
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
) -> int:
    total = 0
    for state_key, state in tensor_states.items():
        rows = materialize_selector_rows(
            votes=votes_by_key[state_key],
            state=state,
        )
        total += len(
            crossing_eligible_flat_indices(
                rows,
                threshold_abs=int(CROSSING_THRESHOLD_ABS),
            )
        )
    return int(total)


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
    base_spec = w6_t10_base_spec(max_abs_per_tensor=int(max_abs_per_tensor))
    one_flip_spec = _single_flip_spec(base_spec)
    flat_index = int(candidate["flat_index"])
    sparse_votes = torch.zeros_like(votes, dtype=torch.int16)
    sparse_votes.view(-1)[flat_index] = votes.view(-1)[flat_index]
    result = apply_integer_vote_update_reference(
        tensor_state.vote_update_state(),
        VoteUpdateInputs(votes=sparse_votes),
        one_flip_spec,
        local_selection_ordering_mode=LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    )
    return _candidate_delta_weight_from_one_flip(
        q_after_one_flip=result.q_levels,
        flat_index=flat_index,
        current_q_level=int(candidate["current_q_level"]),
        frozen_scale_scalar=float(tensor_state.frozen_scale.detach().cpu().item()),
    )


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
