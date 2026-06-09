"""Thin runtime for the candidate-set-viability oracle screen.

This module keeps the oracle screen off the generic science-arm path. It
reuses the non-mutating vote-update planner to generate one candidate set,
samples a small deterministic subset, and evaluates ephemeral one-flip states
without persisting q or checkpoint artifacts.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import math
import random
from statistics import median
import time
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158 import LMHead
from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
    VoteUpdateInputs,
    VoteUpdateSpec,
    authoritative_forward_context,
    credit_from_weighted_grad,
    default_dry_run_rank_vote_spec,
    make_live_shadow_tensor_state,
    project_s1_gradient_to_moves,
    rank_bucketed_int16_votes,
)
from calm.hrm_text_158.native_full_stack.optimizer_update_law_science import (
    BRANCH_MEASUREMENT_AMBIGUOUS_NO_BRANCH,
    BRANCH_MEASUREMENT_AMBIGUOUS_TIE_BAND_ALIASING,
    BRANCH_ORACLE_INFEASIBLE_OR_TOO_EXPENSIVE,
    BRANCH_PREREGISTERED_CHEAP_LEARNER_FEATURE_FAMILY_CANNOT_PREDICT_REGRET,
    BRANCH_WITHIN_TIE_BAND_AMBIGUOUS_NO_BRANCH,
    BRANCH_WITHIN_TIE_BAND_LEARNER_FEATURES_SEPARATE_REGRET,
    BRANCH_WITHIN_TIE_BAND_NEEDS_NEW_LEARNER_STATE,
    ORACLE_ARM_CURRENT_CREDIT_RANK_BUCKET_CURRENT_ORDER,
    ORACLE_ARM_DETERMINISTIC_HASH_SAME_VOTES,
    ORACLE_ARM_DIAGNOSTIC_LOCAL_LOSS_DELTA,
    ORACLE_SCREEN_FEASIBILITY_MAX_SAMPLED_CANDIDATES,
    ORACLE_SCREEN_FEASIBILITY_MAX_SECONDS,
    PIVOT_MEASUREMENT_ABLATION_SCORE_IDS,
    PIVOT_MEASUREMENT_AUC_NON_PREDICTIVE_MAX,
    PIVOT_MEASUREMENT_AUC_PREDICTIVE_MIN,
    PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_CURRENT_SPEC,
    PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_PREFIX_CAP_1024,
    PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_TOP1,
    PIVOT_MEASUREMENT_NULL_AUC_MARGIN_MIN,
    PIVOT_MEASUREMENT_NULL_HASH_SEEDS,
    PIVOT_MEASUREMENT_NULL_PERCENTILE_MIN,
    PIVOT_MEASUREMENT_NULL_RANDOM_SEEDS,
    PIVOT_MEASUREMENT_POOR_RANK_FRACTION_MIN,
    PIVOT_MEASUREMENT_PREDICTIVE_SEED_LABEL,
    PIVOT_MEASUREMENT_PRIMARY_SCORE_ID,
    PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES,
    PIVOT_MEASUREMENT_TIE_BAND_REGRET_SPREAD_RATIO_MAX,
    PIVOT_MEASUREMENT_TOP_K,
    WITHIN_TIE_BAND_ABLATION_FAMILY_IDS,
    WITHIN_TIE_BAND_FAIL_CLOSED_BUCKET_FRACTION_GT,
    WITHIN_TIE_BAND_FAIL_CLOSED_REGRET_SPREAD_RATIO_GT,
    WITHIN_TIE_BAND_MATCHED_HASH_SIGNAL_MIN,
    WITHIN_TIE_BAND_PREDICTIVE_BUCKET_FRACTION_MAX,
    WITHIN_TIE_BAND_PREDICTIVE_REGRET_CAPTURE_RATIO_MIN,
    WITHIN_TIE_BAND_PREDICTIVE_REGRET_SPREAD_RATIO_MAX,
    WITHIN_TIE_BAND_PRIMARY_FAMILY_ID,
    WITHIN_TIE_BAND_TARGET_TIE_BAND_ID,
    classify_candidate_set_viability_oracle_screen,
    oracle_screen_budget_max_seconds,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    LOCAL_SELECTION_ORDER_DETERMINISTIC_HASH_MATCHED,
    VoteUpdateState,
    _local_selection_order,
    apply_integer_vote_update_reference,
    plan_integer_vote_update_reference,
)


ORACLE_SCREEN_MODE_CANDIDATE_SET_VIABILITY = "candidate_set_viability"
ORACLE_SCREEN_MODE_CREDIT_RANKING_PIVOT_MEASUREMENT = (
    "credit_ranking_pivot_measurement"
)
ORACLE_SCREEN_MODE_WITHIN_TIE_BAND_DISCRIMINATOR = (
    "within_tie_band_discriminator"
)
ORACLE_SCREEN_MODE_CHOICES = (
    ORACLE_SCREEN_MODE_CANDIDATE_SET_VIABILITY,
    ORACLE_SCREEN_MODE_CREDIT_RANKING_PIVOT_MEASUREMENT,
    ORACLE_SCREEN_MODE_WITHIN_TIE_BAND_DISCRIMINATOR,
)
ORACLE_SCREEN_TOP_K = 5
ORACLE_SCREEN_ORDERING_SEED = 17
ORACLE_SCREEN_ORDERING_STEP = 1
ORACLE_SCREEN_IMPROVEMENT_EPS = 1e-12


@contextmanager
def _maybe_phase(phase_progress: Any | None, phase: str, **metadata: Any):
    ctx = phase_progress.phase(phase, **metadata) if phase_progress is not None else nullcontext()
    with ctx:
        yield


def _candidate_id(state_key: str, flat_index: int) -> str:
    return f"{state_key}:{int(flat_index)}"


def _sign_int(value: int | float) -> int:
    if float(value) > 0.0:
        return 1
    if float(value) < 0.0:
        return -1
    return 0


def _rank_decile(position: int | None, total: int) -> int | None:
    if position is None or total <= 0:
        return None
    return min(9, int((int(position) * 10) / max(1, int(total))))


def _sampled_rank_position(
    ordered_candidates: Sequence[Mapping[str, Any]],
    candidate_id: str | None,
) -> int | None:
    if candidate_id is None:
        return None
    for position, candidate in enumerate(ordered_candidates):
        if str(candidate.get("candidate_id")) == str(candidate_id):
            return position
    return None


def _sampled_rank_fraction(position: int | None, sampled_candidate_count: int) -> float | None:
    if position is None or sampled_candidate_count <= 0:
        return None
    return float((int(position) + 1) / int(sampled_candidate_count))


def _ordinal_fraction(position: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float((int(position) + 1) / int(total))


def _quartile_index(position: int, total: int) -> int:
    if total <= 0:
        return 0
    clipped = max(0, min(int(position), max(0, int(total) - 1)))
    return min(3, int((clipped * 4) / max(1, int(total))))


def _pivot_poor_rank_position_threshold(sampled_candidate_count: int) -> int:
    if sampled_candidate_count <= 0:
        return 0
    return int(
        math.ceil(
            float(PIVOT_MEASUREMENT_POOR_RANK_FRACTION_MIN)
            * float(sampled_candidate_count)
        )
    )


def _pivot_is_poor_rank_position(
    position: int | None,
    sampled_candidate_count: int,
) -> bool:
    if position is None or sampled_candidate_count <= 0:
        return False
    return int(position) >= _pivot_poor_rank_position_threshold(sampled_candidate_count)


def _pivot_tie_band_is_ambiguous(
    *,
    oracle_best_candidate_present: bool,
    band_candidate_count: int,
    regret_spread_ratio: float | None,
) -> bool:
    return bool(
        oracle_best_candidate_present
        and int(band_candidate_count) > 1
        and regret_spread_ratio is not None
        and float(regret_spread_ratio)
        > float(PIVOT_MEASUREMENT_TIE_BAND_REGRET_SPREAD_RATIO_MAX)
    )


def _single_flip_spec(base_spec: VoteUpdateSpec) -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=int(base_spec.threshold_abs),
        accumulator_clip_min=int(base_spec.accumulator_clip_min),
        accumulator_clip_max=int(base_spec.accumulator_clip_max),
        max_abs_per_tensor=1,
        fraction_per_tensor=float(base_spec.fraction_per_tensor),
        decay_numerator=int(base_spec.decay_numerator),
        decay_denominator=int(base_spec.decay_denominator),
    )


def _evaluate_loss(
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    eligible_modules: Mapping[str, BitLinear],
    *,
    device: torch.device,
    extras: Mapping[str, Any],
) -> float:
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        with authoritative_forward_context(
            eligible_modules,
            tensor_states,
            device=device,
            requires_grad=False,
        ):
            _carry, loss, _metrics = model(None, dict(batch), **extras)
    model.zero_grad(set_to_none=True)
    return float(loss.detach().cpu().item())


def _compute_baseline_votes(
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    eligible_modules: Mapping[str, BitLinear],
    *,
    device: torch.device,
    extras: Mapping[str, Any],
) -> tuple[float, dict[str, torch.Tensor]]:
    rank_spec = default_dry_run_rank_vote_spec()
    votes_by_key: dict[str, torch.Tensor] = {}
    model.zero_grad(set_to_none=True)
    with authoritative_forward_context(
        eligible_modules,
        tensor_states,
        device=device,
        requires_grad=True,
    ) as handle:
        _carry, loss, _metrics = model(None, dict(batch), **extras)
        loss.backward()
        for state_key in tensor_states:
            weighted_grad = handle.weighted_grad(state_key)
            moves = project_s1_gradient_to_moves(weighted_grad, tensor_states[state_key].q_levels)
            credit = credit_from_weighted_grad(weighted_grad)
            votes_by_key[state_key] = rank_bucketed_int16_votes(
                credit,
                moves,
                rank_spec,
            ).detach().cpu().to(torch.int16).contiguous()
    model.zero_grad(set_to_none=True)
    return float(loss.detach().cpu().item()), votes_by_key


def _ordered_candidate_indices(
    *,
    state: VoteUpdateState,
    votes: torch.Tensor,
    spec: VoteUpdateSpec,
    ordering_mode: str,
) -> tuple[list[int], list[int], torch.Tensor]:
    plan = plan_integer_vote_update_reference(
        state,
        VoteUpdateInputs(votes=votes),
        spec,
        local_selection_ordering_mode=str(ordering_mode),
        local_selection_ordering_seed=ORACLE_SCREEN_ORDERING_SEED,
        local_selection_ordering_step=ORACLE_SCREEN_ORDERING_STEP,
    )
    candidate_idx = plan.candidate_indices.detach().cpu().to(torch.int64)
    if candidate_idx.numel() == 0:
        return [], [], plan.new_acc_i32.detach().cpu().to(torch.int32).flatten()
    new_acc = plan.new_acc_i32.detach().cpu().to(torch.int32).flatten()
    order = _local_selection_order(
        candidate_idx=candidate_idx,
        new_acc_i32=new_acc,
        numel=int(state.q_levels.numel()),
        mode=str(ordering_mode),
        ordering_seed=ORACLE_SCREEN_ORDERING_SEED,
        ordering_step=ORACLE_SCREEN_ORDERING_STEP,
    )
    ordered = candidate_idx[order].detach().cpu().to(torch.int64).tolist()
    unordered = candidate_idx.detach().cpu().to(torch.int64).tolist()
    return [int(index) for index in ordered], [int(index) for index in unordered], new_acc


def _sample_candidate_ids(
    current_ordered_ids: Sequence[str],
    deterministic_ordered_ids: Sequence[str],
    *,
    max_sampled_candidates: int,
) -> list[str]:
    if max_sampled_candidates <= 0:
        return []
    sampled: list[str] = []
    seen: set[str] = set()
    if len(current_ordered_ids) <= max_sampled_candidates:
        return [str(candidate_id) for candidate_id in current_ordered_ids]
    max_len = max(len(current_ordered_ids), len(deterministic_ordered_ids))
    for index in range(max_len):
        for source in (current_ordered_ids, deterministic_ordered_ids):
            if index >= len(source):
                continue
            candidate_id = str(source[index])
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            sampled.append(candidate_id)
            if len(sampled) >= max_sampled_candidates:
                return sampled
    return sampled


def run_candidate_set_viability_oracle_screen(
    *,
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    eligible_modules: Mapping[str, BitLinear],
    device: torch.device,
    max_abs_per_tensor: int,
    extras: Mapping[str, Any],
    max_sampled_candidates: int = ORACLE_SCREEN_FEASIBILITY_MAX_SAMPLED_CANDIDATES,
    max_seconds: float = ORACLE_SCREEN_FEASIBILITY_MAX_SECONDS,
    phase_progress: Any | None = None,
) -> dict[str, Any]:
    if int(max_sampled_candidates) <= 0:
        raise ValueError("oracle screen requires max_sampled_candidates > 0")
    base_spec = VoteUpdateSpec(
        threshold_abs=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=int(max_abs_per_tensor),
    )
    one_flip_spec = _single_flip_spec(base_spec)
    with _maybe_phase(phase_progress, "step_forward_backward", step=1):
        baseline_loss, votes_by_key = _compute_baseline_votes(
            model,
            batch,
            tensor_states,
            eligible_modules,
            device=device,
            extras=extras,
        )
    candidate_by_id: dict[str, dict[str, Any]] = {}
    with _maybe_phase(phase_progress, "step_update", step=1):
        for state_key, state in sorted(tensor_states.items()):
            vote_state = state.vote_update_state()
            votes = votes_by_key[state_key]
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
            if unordered != deterministic_unordered:
                raise RuntimeError("oracle screen candidate set drifted across scheduler orderings")
            current_rank = {
                int(flat_index): int(position)
                for position, flat_index in enumerate(current_ordered)
            }
            deterministic_rank = {
                int(flat_index): int(position)
                for position, flat_index in enumerate(deterministic_ordered)
            }
            vote_flat = votes.flatten().to(torch.int32)
            for flat_index in unordered:
                candidate_id = _candidate_id(state_key, int(flat_index))
                candidate_by_id[candidate_id] = {
                    "candidate_id": candidate_id,
                    "state_key": state_key,
                    "flat_index": int(flat_index),
                    "vote_value": int(vote_flat[int(flat_index)].item()),
                    "current_rank_position": current_rank[int(flat_index)],
                    "deterministic_hash_rank_position": deterministic_rank[int(flat_index)],
                    "current_margin_abs": int(abs(int(new_acc[int(flat_index)].item()))),
                }
        current_ordered_ids = [
            candidate["candidate_id"]
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
            candidate["candidate_id"]
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
    budget_start = time.perf_counter()
    sampled_candidates: list[dict[str, Any]] = []
    budget_exceeded = False
    with _maybe_phase(phase_progress, "audit", step=1):
        for candidate_id in sampled_ids:
            if time.perf_counter() - budget_start > float(max_seconds):
                budget_exceeded = True
                break
            candidate = dict(candidate_by_id[candidate_id])
            state_key = str(candidate["state_key"])
            flat_index = int(candidate["flat_index"])
            sparse_votes = torch.zeros_like(votes_by_key[state_key], dtype=torch.int16)
            sparse_votes.view(-1)[flat_index] = votes_by_key[state_key].view(-1)[flat_index]
            result = apply_integer_vote_update_reference(
                tensor_states[state_key].vote_update_state(),
                VoteUpdateInputs(votes=sparse_votes),
                one_flip_spec,
                local_selection_ordering_mode=LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
                local_selection_ordering_seed=ORACLE_SCREEN_ORDERING_SEED,
                local_selection_ordering_step=ORACLE_SCREEN_ORDERING_STEP,
            )
            applied = result.plan.applied_indices.detach().cpu().to(torch.int64).tolist()
            if applied != [flat_index]:
                raise RuntimeError(
                    "oracle screen single-candidate application drifted from the planned candidate identity"
                )
            candidate_states = dict(tensor_states)
            candidate_states[state_key] = make_live_shadow_tensor_state(
                tensor_states[state_key],
                result.q_levels,
                result.accumulators,
            )
            candidate_loss = _evaluate_loss(
                model,
                batch,
                candidate_states,
                eligible_modules,
                device=device,
                extras=extras,
            )
            candidate["candidate_loss"] = float(candidate_loss)
            candidate["local_loss_delta"] = float(candidate_loss - baseline_loss)
            sampled_candidates.append(candidate)
    sampled_count = len(sampled_candidates)
    sampled_candidate_ids = {candidate["candidate_id"] for candidate in sampled_candidates}
    top_k = min(ORACLE_SCREEN_TOP_K, sampled_count)
    current_top = sorted(
        sampled_candidates,
        key=lambda candidate: (
            int(candidate["current_rank_position"]),
            str(candidate["candidate_id"]),
        ),
    )
    deterministic_top = sorted(
        sampled_candidates,
        key=lambda candidate: (
            int(candidate["deterministic_hash_rank_position"]),
            str(candidate["candidate_id"]),
        ),
    )
    oracle_top = sorted(
        sampled_candidates,
        key=lambda candidate: (
            float(candidate["local_loss_delta"]),
            str(candidate["candidate_id"]),
        ),
    )
    candidate_set_contains_ce_improving_move = any(
        float(candidate["local_loss_delta"]) < -ORACLE_SCREEN_IMPROVEMENT_EPS
        for candidate in sampled_candidates
    )
    current_credit_rank_recovers_improvement = any(
        float(candidate["local_loss_delta"]) < -ORACLE_SCREEN_IMPROVEMENT_EPS
        for candidate in current_top[:top_k]
    )
    deterministic_hash_recovers_improvement = any(
        float(candidate["local_loss_delta"]) < -ORACLE_SCREEN_IMPROVEMENT_EPS
        for candidate in deterministic_top[:top_k]
    )
    current_top1_delta = (
        float(current_top[0]["local_loss_delta"]) if current_top else None
    )
    oracle_top1_delta = float(oracle_top[0]["local_loss_delta"]) if oracle_top else None
    oracle_advantage_over_current = bool(
        oracle_top1_delta is not None
        and current_top1_delta is not None
        and oracle_top1_delta < current_top1_delta - ORACLE_SCREEN_IMPROVEMENT_EPS
    )
    sign_product_values = [
        float(candidate["vote_value"]) * float(-candidate["local_loss_delta"])
        for candidate in sampled_candidates
    ]
    sign_product_mean = (
        float(sum(sign_product_values) / len(sign_product_values))
        if sign_product_values
        else 0.0
    )
    credit_sign_concordance_positive = bool(sign_product_mean > 0.0)
    oracle_feasible = bool(not budget_exceeded)
    branch_inputs = {
        "oracle_feasible": oracle_feasible,
        "candidate_set_contains_ce_improving_move": candidate_set_contains_ce_improving_move,
        "current_credit_rank_recovers_improvement": current_credit_rank_recovers_improvement,
        "deterministic_hash_recovers_improvement": deterministic_hash_recovers_improvement,
        "oracle_advantage_over_current": oracle_advantage_over_current,
        "credit_sign_concordance_positive": credit_sign_concordance_positive,
    }
    branch_classification = classify_candidate_set_viability_oracle_screen(**branch_inputs)
    if budget_exceeded:
        branch_classification = BRANCH_ORACLE_INFEASIBLE_OR_TOO_EXPENSIVE
    improving_current_rank = next(
        (
            int(candidate["current_rank_position"])
            for candidate in current_top
            if float(candidate["local_loss_delta"]) < -ORACLE_SCREEN_IMPROVEMENT_EPS
        ),
        None,
    )
    improving_hash_rank = next(
        (
            int(candidate["deterministic_hash_rank_position"])
            for candidate in deterministic_top
            if float(candidate["local_loss_delta"]) < -ORACLE_SCREEN_IMPROVEMENT_EPS
        ),
        None,
    )
    loss_deltas = [float(candidate["local_loss_delta"]) for candidate in sampled_candidates]
    ce_improving_candidate_count = sum(
        1 for delta in loss_deltas if delta < -ORACLE_SCREEN_IMPROVEMENT_EPS
    )
    ce_improving_candidate_fraction = (
        float(ce_improving_candidate_count / sampled_count) if sampled_count > 0 else None
    )
    oracle_best_candidate_id = str(oracle_top[0]["candidate_id"]) if oracle_top else None
    oracle_best_current_rank_position = (
        int(oracle_top[0]["current_rank_position"]) if oracle_top else None
    )
    oracle_best_deterministic_hash_rank_position = (
        int(oracle_top[0]["deterministic_hash_rank_position"]) if oracle_top else None
    )
    oracle_best_current_sampled_rank_position = _sampled_rank_position(
        current_top,
        oracle_best_candidate_id,
    )
    oracle_best_deterministic_hash_sampled_rank_position = _sampled_rank_position(
        deterministic_top,
        oracle_best_candidate_id,
    )
    oracle_best_current_rank_fraction = _sampled_rank_fraction(
        oracle_best_current_sampled_rank_position,
        sampled_count,
    )
    oracle_best_deterministic_hash_rank_fraction = _sampled_rank_fraction(
        oracle_best_deterministic_hash_sampled_rank_position,
        sampled_count,
    )
    current_vs_oracle_top1_gap = (
        float(current_top1_delta - oracle_top1_delta)
        if current_top1_delta is not None and oracle_top1_delta is not None
        else None
    )
    current_vs_oracle_top1_gap_denominator_abs_oracle_top1_local_loss_delta = (
        abs(float(oracle_top1_delta)) if oracle_top1_delta is not None else None
    )
    current_vs_oracle_top1_gap_ratio = None
    if (
        current_vs_oracle_top1_gap is not None
        and current_vs_oracle_top1_gap_denominator_abs_oracle_top1_local_loss_delta
        is not None
        and current_vs_oracle_top1_gap_denominator_abs_oracle_top1_local_loss_delta > 0.0
    ):
        current_vs_oracle_top1_gap_ratio = float(
            current_vs_oracle_top1_gap
            / current_vs_oracle_top1_gap_denominator_abs_oracle_top1_local_loss_delta
        )
    wider_screen_interpretation_inputs = {
        "sampled_candidate_count": sampled_count,
        "max_sampled_candidates": int(max_sampled_candidates),
        "oracle_best_current_rank_position": oracle_best_current_rank_position,
        "oracle_best_current_sampled_rank_position": (
            oracle_best_current_sampled_rank_position
        ),
        "oracle_best_current_rank_fraction": oracle_best_current_rank_fraction,
        "oracle_best_deterministic_hash_rank_position": (
            oracle_best_deterministic_hash_rank_position
        ),
        "oracle_best_deterministic_hash_sampled_rank_position": (
            oracle_best_deterministic_hash_sampled_rank_position
        ),
        "oracle_best_deterministic_hash_rank_fraction": (
            oracle_best_deterministic_hash_rank_fraction
        ),
        "current_vs_oracle_top1_gap": current_vs_oracle_top1_gap,
        "current_vs_oracle_top1_gap_denominator_abs_oracle_top1_local_loss_delta": (
            current_vs_oracle_top1_gap_denominator_abs_oracle_top1_local_loss_delta
        ),
        "current_vs_oracle_top1_gap_ratio": current_vs_oracle_top1_gap_ratio,
        "ce_improving_candidate_fraction": ce_improving_candidate_fraction,
    }
    compact_summary = {
        "candidate_count": len(candidate_by_id),
        "sampled_candidate_count": sampled_count,
        "top_k": {
            "k": top_k,
            ORACLE_ARM_CURRENT_CREDIT_RANK_BUCKET_CURRENT_ORDER: [
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "local_loss_delta": float(candidate["local_loss_delta"]),
                }
                for candidate in current_top[:top_k]
            ],
            ORACLE_ARM_DETERMINISTIC_HASH_SAME_VOTES: [
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "local_loss_delta": float(candidate["local_loss_delta"]),
                }
                for candidate in deterministic_top[:top_k]
            ],
            ORACLE_ARM_DIAGNOSTIC_LOCAL_LOSS_DELTA: [
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "local_loss_delta": float(candidate["local_loss_delta"]),
                }
                for candidate in oracle_top[:top_k]
            ],
        },
        "sign_concordance": {
            "positive": credit_sign_concordance_positive,
            "weighted_product_mean": sign_product_mean,
            "sampled_candidate_count": sampled_count,
        },
        "credit_rank_deciles": {
            "sampled_candidate_count": sampled_count,
            "best_improving_current_decile": _rank_decile(improving_current_rank, sampled_count),
            "best_improving_deterministic_hash_decile": _rank_decile(
                improving_hash_rank,
                sampled_count,
            ),
        },
        "local_loss_delta_deciles": {
            "sampled_candidate_count": sampled_count,
            "best_local_loss_delta": min(loss_deltas) if loss_deltas else None,
            "median_local_loss_delta": median(loss_deltas) if loss_deltas else None,
            "worst_local_loss_delta": max(loss_deltas) if loss_deltas else None,
            "ce_improving_candidate_count": ce_improving_candidate_count,
        },
        "paired_loss_branch_fields": {
            **branch_inputs,
            "current_top1_local_loss_delta": current_top1_delta,
            "oracle_top1_local_loss_delta": oracle_top1_delta,
        },
        "wider_screen_interpretation_inputs": wider_screen_interpretation_inputs,
    }
    return {
        "schema": "hrm_text_158_candidate_set_viability_oracle_screen_runtime/v0",
        "mode": ORACLE_SCREEN_MODE_CANDIDATE_SET_VIABILITY,
        "same_candidate_set_required": True,
        "screen_rows": int(batch["inputs"].shape[0]),
        "baseline_loss": float(baseline_loss),
        "candidate_count": len(candidate_by_id),
        "sampled_candidate_count": sampled_count,
        "sample_truncated": bool(len(candidate_by_id) > sampled_count),
        "sampled_candidate_ids_hash16": hashlib.sha256(
            "\n".join(sorted(sampled_candidate_ids)).encode("utf-8")
        ).hexdigest()[:16],
        "max_sampled_candidates": int(max_sampled_candidates),
        "max_seconds": float(max_seconds),
        "elapsed_seconds": float(time.perf_counter() - budget_start),
        "budget_exceeded": bool(budget_exceeded),
        "oracle_feasible": bool(oracle_feasible),
        "compact_summary": compact_summary,
        "wider_screen_interpretation_inputs": wider_screen_interpretation_inputs,
        "branch_inputs": branch_inputs,
        "branch_classification": branch_classification,
        "non_persistence": {
            "q_persisted": False,
            "checkpoint_written": False,
            "pt_writes_allowed": False,
            "screen_state_mutated": False,
        },
    }


def _candidate_ids_hash16(candidate_ids: Sequence[str], *, preserve_order: bool = False) -> str:
    values = list(candidate_ids)
    if not preserve_order:
        values = sorted(str(candidate_id) for candidate_id in values)
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()[:16]


def _pivot_tie_band_id(*, abs_vote_value: int, current_margin_abs: int) -> str:
    return f"voteabs={int(abs_vote_value)}|marginabs={int(current_margin_abs)}"


def _pivot_score_key(candidate: Mapping[str, Any], score_id: str) -> tuple[Any, ...]:
    candidate_id = str(candidate["candidate_id"])
    flat_index = int(candidate["flat_index"])
    if score_id == PIVOT_MEASUREMENT_PRIMARY_SCORE_ID:
        return (
            -int(candidate["abs_vote_value"]),
            -int(candidate["current_margin_abs"]),
            int(flat_index),
            candidate_id,
        )
    if score_id == "S_vote_only":
        return (
            -int(candidate["abs_vote_value"]),
            int(flat_index),
            candidate_id,
        )
    if score_id == "S_margin_only":
        return (
            -int(candidate["current_margin_abs"]),
            int(flat_index),
            candidate_id,
        )
    if score_id == "S_current":
        return (
            int(candidate["current_rank_position"]),
            int(flat_index),
            candidate_id,
        )
    raise ValueError(f"unsupported pivot score family {score_id!r}")


def _ordered_candidates_for_pivot_score(
    sampled_candidates: Sequence[Mapping[str, Any]],
    *,
    score_id: str,
) -> list[dict[str, Any]]:
    return [
        dict(candidate)
        for candidate in sorted(
            sampled_candidates,
            key=lambda candidate: _pivot_score_key(candidate, score_id),
        )
    ]


def _pairwise_auc_from_positions(
    sampled_candidates: Sequence[Mapping[str, Any]],
    *,
    position_by_id: Mapping[str, int],
) -> float:
    wins = 0.0
    total = 0
    for left_index, left in enumerate(sampled_candidates):
        left_delta = float(left["local_loss_delta"])
        left_id = str(left["candidate_id"])
        for right in sampled_candidates[left_index + 1 :]:
            right_delta = float(right["local_loss_delta"])
            if abs(left_delta - right_delta) <= ORACLE_SCREEN_IMPROVEMENT_EPS:
                continue
            total += 1
            right_id = str(right["candidate_id"])
            if left_delta < right_delta:
                better_id = left_id
                worse_id = right_id
            else:
                better_id = right_id
                worse_id = left_id
            better_position = int(position_by_id[better_id])
            worse_position = int(position_by_id[worse_id])
            if better_position < worse_position:
                wins += 1.0
            elif better_position == worse_position:
                wins += 0.5
    if total <= 0:
        return 0.5
    return float(wins / total)


def _positive_improvement_mass(candidates: Sequence[Mapping[str, Any]]) -> float:
    return float(
        sum(
            max(0.0, -float(candidate["local_loss_delta"]))
            for candidate in candidates
        )
    )


def _loss_spread_ratio(
    candidates: Sequence[Mapping[str, Any]],
    *,
    oracle_top1_delta: float | None,
) -> float | None:
    if not candidates:
        return None
    deltas = [float(candidate["local_loss_delta"]) for candidate in candidates]
    spread = float(max(deltas) - min(deltas))
    if oracle_top1_delta is None:
        return None
    if abs(float(oracle_top1_delta)) > ORACLE_SCREEN_IMPROVEMENT_EPS:
        return float(spread / abs(float(oracle_top1_delta)))
    if spread <= ORACLE_SCREEN_IMPROVEMENT_EPS:
        return 0.0
    return None


def _fraction_gte_observed(values: Sequence[float], observed: float) -> float:
    if not values:
        return 0.0
    return float(
        sum(1 for value in values if float(value) >= float(observed)) / len(values)
    )


def _fraction_lte_observed(values: Sequence[float], observed: float) -> float:
    if not values:
        return 0.0
    return float(
        sum(1 for value in values if float(value) <= float(observed)) / len(values)
    )


def _deterministic_hash_ordered_candidate_ids(
    sampled_candidates: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    salt: str,
) -> list[str]:
    return [
        str(candidate["candidate_id"])
        for candidate in sorted(
            sampled_candidates,
            key=lambda candidate: (
                hashlib.sha256(
                    (
                        f"{salt}|seed={int(seed)}|candidate_id={str(candidate['candidate_id'])}"
                    ).encode("utf-8")
                ).digest(),
                str(candidate["candidate_id"]),
            ),
        )
    ]


def _hash_ordered_candidate_ids(
    sampled_candidates: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> list[str]:
    return _deterministic_hash_ordered_candidate_ids(
        sampled_candidates,
        seed=int(seed),
        salt="hrm_text_158_credit_ranking_pivot_null_hash",
    )


def _within_tie_band_family_key(
    candidate: Mapping[str, Any],
    *,
    family_id: str,
) -> tuple[Any, ...]:
    if family_id == WITHIN_TIE_BAND_PRIMARY_FAMILY_ID:
        return (
            str(candidate["state_key"]),
            str(candidate["transition_class"]),
            int(candidate["current_rank_quartile_within_state"]),
        )
    if family_id == "F_transition_rankq":
        return (
            str(candidate["transition_class"]),
            int(candidate["current_rank_quartile_within_state"]),
        )
    if family_id == "F_state_transition":
        return (
            str(candidate["state_key"]),
            str(candidate["transition_class"]),
        )
    if family_id == "F_transition_only":
        return (str(candidate["transition_class"]),)
    if family_id == "F_rankq_only":
        return (int(candidate["current_rank_quartile_within_state"]),)
    if family_id == "F_flatq_only":
        return (int(candidate["flat_index_quartile"]),)
    raise ValueError(f"unsupported within-tie-band family_id {family_id!r}")


def _within_tie_band_family_metrics(
    *,
    target_band_candidates: Sequence[Mapping[str, Any]],
    family_id: str,
    oracle_best_candidate: Mapping[str, Any],
    oracle_top1_delta: float | None,
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for candidate in target_band_candidates:
        groups.setdefault(
            _within_tie_band_family_key(candidate, family_id=family_id),
            [],
        ).append(dict(candidate))
    bucket_sizes = sorted((len(group) for group in groups.values()), reverse=True)
    histogram: dict[str, int] = {}
    for size in bucket_sizes:
        key = str(int(size))
        histogram[key] = int(histogram.get(key, 0) + 1)
    oracle_key = _within_tie_band_family_key(
        oracle_best_candidate,
        family_id=family_id,
    )
    oracle_bucket = list(groups.get(oracle_key, []))
    bucket_ids = {
        str(candidate["candidate_id"])
        for candidate in oracle_bucket
    }
    band_count = len(target_band_candidates)
    band_top_k = min(PIVOT_MEASUREMENT_TOP_K, band_count)
    band_top = sorted(
        target_band_candidates,
        key=lambda candidate: (
            float(candidate["local_loss_delta"]),
            str(candidate["candidate_id"]),
        ),
    )[:band_top_k]
    band_top_ids = {
        str(candidate["candidate_id"])
        for candidate in band_top
    }
    band_improvement_mass = _positive_improvement_mass(target_band_candidates)
    bucket_improvement_mass = _positive_improvement_mass(oracle_bucket)
    regret_capture_ratio = (
        float(bucket_improvement_mass / band_improvement_mass)
        if band_improvement_mass > ORACLE_SCREEN_IMPROVEMENT_EPS
        else 0.0
    )
    position_by_id = {
        str(candidate["candidate_id"]): int(position)
        for position, candidate in enumerate(
            sorted(
                target_band_candidates,
                key=lambda candidate: (
                    str(candidate["candidate_id"]) not in bucket_ids,
                    int(candidate["current_rank_position"]),
                    str(candidate["candidate_id"]),
                ),
            )
        )
    }
    null_bucket_fractions: list[float] = []
    null_capture_ratios: list[float] = []
    target_band_by_id = {
        str(candidate["candidate_id"]): dict(candidate)
        for candidate in target_band_candidates
    }
    for seed in PIVOT_MEASUREMENT_NULL_HASH_SEEDS:
        ordered_ids = _deterministic_hash_ordered_candidate_ids(
            target_band_candidates,
            seed=int(seed),
            salt="hrm_text_158_within_tie_band_discriminator_null_hash",
        )
        ordered_candidates = [
            target_band_by_id[candidate_id]
            for candidate_id in ordered_ids
        ]
        cursor = 0
        null_bucket: list[dict[str, Any]] = []
        for size in bucket_sizes:
            bucket = ordered_candidates[cursor : cursor + int(size)]
            cursor += int(size)
            if any(
                str(candidate["candidate_id"]) == str(oracle_best_candidate["candidate_id"])
                for candidate in bucket
            ):
                null_bucket = list(bucket)
                break
        if not null_bucket:
            raise RuntimeError(
                "within-tie-band matched-cardinality null partition dropped the oracle-best candidate"
            )
        null_bucket_fractions.append(float(len(null_bucket) / max(1, band_count)))
        null_bucket_improvement_mass = _positive_improvement_mass(null_bucket)
        null_capture_ratios.append(
            float(null_bucket_improvement_mass / band_improvement_mass)
            if band_improvement_mass > ORACLE_SCREEN_IMPROVEMENT_EPS
            else 0.0
        )
    return {
        "family_id": family_id,
        "bucket_count": len(groups),
        "bucket_cardinality_histogram": histogram,
        "singleton_bucket_count": int(histogram.get("1", 0)),
        "oracle_best_bucket_candidate_count": len(oracle_bucket),
        "oracle_best_bucket_candidate_ids_hash16": _candidate_ids_hash16(
            list(bucket_ids),
        ),
        "oracle_best_bucket_fraction": float(len(oracle_bucket) / max(1, band_count)),
        "oracle_best_bucket_regret_spread_ratio": _loss_spread_ratio(
            oracle_bucket,
            oracle_top1_delta=oracle_top1_delta,
        ),
        "oracle_best_bucket_regret_capture_ratio": regret_capture_ratio,
        "oracle_best_bucket_top_k_capture_fraction": (
            float(len(bucket_ids & band_top_ids) / band_top_k)
            if band_top_k > 0
            else 0.0
        ),
        "within_band_pairwise_auc_report_only": _pairwise_auc_from_positions(
            target_band_candidates,
            position_by_id=position_by_id,
        ),
        "matched_hash_seed_count": len(PIVOT_MEASUREMENT_NULL_HASH_SEEDS),
        "matched_hash_null_fraction_gte_observed_bucket_fraction": (
            _fraction_gte_observed(
                null_bucket_fractions,
                float(len(oracle_bucket) / max(1, band_count)),
            )
        ),
        "matched_hash_null_fraction_lte_observed_regret_capture_ratio": (
            _fraction_lte_observed(
                null_capture_ratios,
                regret_capture_ratio,
            )
        ),
        "null_control_hash_only": True,
    }


def _random_permutation_candidate_ids(
    sampled_candidates: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> list[str]:
    ids = [str(candidate["candidate_id"]) for candidate in sampled_candidates]
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    return ids


def _score_family_metrics(
    *,
    sampled_candidates: Sequence[Mapping[str, Any]],
    score_id: str,
    oracle_top: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str], dict[str, int]]:
    ordered = _ordered_candidates_for_pivot_score(sampled_candidates, score_id=score_id)
    ordered_ids = [str(candidate["candidate_id"]) for candidate in ordered]
    position_by_id = {
        str(candidate_id): int(position)
        for position, candidate_id in enumerate(ordered_ids)
    }
    sampled_count = len(ordered_ids)
    top_k = min(PIVOT_MEASUREMENT_TOP_K, sampled_count)
    oracle_top_ids = {
        str(candidate["candidate_id"])
        for candidate in list(oracle_top)[:top_k]
    }
    predicted_top = ordered[:top_k]
    predicted_top_ids = {
        str(candidate["candidate_id"])
        for candidate in predicted_top
    }
    overlap = 0.0
    if top_k > 0:
        overlap = float(len(oracle_top_ids & predicted_top_ids) / top_k)
    oracle_top_improvement_mass = sum(
        max(0.0, -float(candidate["local_loss_delta"]))
        for candidate in list(oracle_top)[:top_k]
    )
    predicted_top_improvement_mass = sum(
        max(0.0, -float(candidate["local_loss_delta"]))
        for candidate in predicted_top
    )
    if oracle_top_improvement_mass > ORACLE_SCREEN_IMPROVEMENT_EPS:
        regret_capture_ratio = float(
            predicted_top_improvement_mass / oracle_top_improvement_mass
        )
        top_k_gap_ratio = float(
            max(0.0, oracle_top_improvement_mass - predicted_top_improvement_mass)
            / oracle_top_improvement_mass
        )
    else:
        regret_capture_ratio = 1.0
        top_k_gap_ratio = 0.0
    oracle_best_candidate_id = str(oracle_top[0]["candidate_id"]) if oracle_top else None
    oracle_best_sampled_rank_position = (
        int(position_by_id[oracle_best_candidate_id])
        if oracle_best_candidate_id is not None
        else None
    )
    metrics = {
        "score_id": score_id,
        "oracle_top_k_overlap_fraction": overlap,
        "oracle_top_k_regret_capture_ratio": regret_capture_ratio,
        "oracle_top_k_gap_ratio": top_k_gap_ratio,
        "pairwise_auc": _pairwise_auc_from_positions(
            sampled_candidates,
            position_by_id=position_by_id,
        ),
        "oracle_best_sampled_rank_position": oracle_best_sampled_rank_position,
        "oracle_best_sampled_rank_fraction": _sampled_rank_fraction(
            oracle_best_sampled_rank_position,
            sampled_count,
        ),
        "oracle_best_sampled_rank_position_poor_threshold": (
            _pivot_poor_rank_position_threshold(sampled_count)
        ),
        "top_k": top_k,
    }
    return metrics, ordered_ids, position_by_id


def _build_oracle_candidate_universe(
    *,
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    eligible_modules: Mapping[str, BitLinear],
    device: torch.device,
    max_abs_per_tensor: int,
    extras: Mapping[str, Any],
    max_sampled_candidates: int,
    phase_progress: Any | None,
) -> dict[str, Any]:
    base_spec = VoteUpdateSpec(
        threshold_abs=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=int(max_abs_per_tensor),
    )
    one_flip_spec = _single_flip_spec(base_spec)
    with _maybe_phase(phase_progress, "step_forward_backward", step=1):
        baseline_loss, votes_by_key = _compute_baseline_votes(
            model,
            batch,
            tensor_states,
            eligible_modules,
            device=device,
            extras=extras,
        )
    candidate_by_id: dict[str, dict[str, Any]] = {}
    with _maybe_phase(phase_progress, "step_update", step=1):
        for state_key, state in sorted(tensor_states.items()):
            vote_state = state.vote_update_state()
            votes = votes_by_key[state_key]
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
            if unordered != deterministic_unordered:
                raise RuntimeError(
                    "oracle screen candidate set drifted across scheduler orderings"
                )
            current_rank = {
                int(flat_index): int(position)
                for position, flat_index in enumerate(current_ordered)
            }
            deterministic_rank = {
                int(flat_index): int(position)
                for position, flat_index in enumerate(deterministic_ordered)
            }
            vote_flat = votes.flatten().to(torch.int32)
            q_flat = vote_state.q_levels.flatten().to(torch.int32)
            acc_flat = vote_state.accumulators.flatten().to(torch.int32)
            state_candidate_count = len(unordered)
            tensor_numel = int(vote_state.q_levels.numel())
            for flat_index in unordered:
                vote_value = int(vote_flat[int(flat_index)].item())
                new_acc_signed = int(new_acc[int(flat_index)].item())
                current_margin_abs = int(abs(new_acc_signed))
                abs_vote_value = int(abs(vote_value))
                current_rank_position = int(current_rank[int(flat_index)])
                current_q_level = int(q_flat[int(flat_index)].item())
                pre_accumulator_i16 = int(acc_flat[int(flat_index)].item())
                proposal_direction = _sign_int(new_acc_signed)
                threshold_residual_signed = int(
                    new_acc_signed - proposal_direction * int(base_spec.threshold_abs)
                )
                proximity_to_threshold = int(
                    abs(abs(new_acc_signed) - int(base_spec.threshold_abs))
                )
                candidate_id = _candidate_id(state_key, int(flat_index))
                candidate_by_id[candidate_id] = {
                    "candidate_id": candidate_id,
                    "state_key": state_key,
                    "flat_index": int(flat_index),
                    "vote_value": vote_value,
                    "abs_vote_value": abs_vote_value,
                    "current_rank_position": current_rank_position,
                    "deterministic_hash_rank_position": deterministic_rank[int(flat_index)],
                    "current_margin_abs": current_margin_abs,
                    "tie_band_id": _pivot_tie_band_id(
                        abs_vote_value=abs_vote_value,
                        current_margin_abs=current_margin_abs,
                    ),
                    "current_q_level": current_q_level,
                    "pre_accumulator_i16": pre_accumulator_i16,
                    "new_acc_i32_signed": new_acc_signed,
                    "proposal_direction": proposal_direction,
                    "threshold_residual_signed": threshold_residual_signed,
                    "proximity_to_threshold": proximity_to_threshold,
                    "tensor_numel": tensor_numel,
                    "state_candidate_count": state_candidate_count,
                    "current_rank_fraction_within_state": _ordinal_fraction(
                        current_rank_position,
                        state_candidate_count,
                    ),
                    "current_rank_quartile_within_state": _quartile_index(
                        current_rank_position,
                        state_candidate_count,
                    ),
                    "flat_index_fraction": _ordinal_fraction(
                        int(flat_index),
                        tensor_numel,
                    ),
                    "flat_index_quartile": _quartile_index(
                        int(flat_index),
                        tensor_numel,
                    ),
                    "transition_class": f"q{current_q_level}|dir{proposal_direction}",
                }
        current_ordered_ids = [
            candidate["candidate_id"]
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
            candidate["candidate_id"]
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
        "baseline_loss": float(baseline_loss),
        "base_spec": base_spec,
        "one_flip_spec": one_flip_spec,
        "votes_by_key": votes_by_key,
        "candidate_by_id": candidate_by_id,
        "sampled_ids": sampled_ids,
    }


def _evaluate_sparse_selected_candidate_ids(
    *,
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    eligible_modules: Mapping[str, BitLinear],
    device: torch.device,
    extras: Mapping[str, Any],
    votes_by_key: Mapping[str, torch.Tensor],
    candidate_by_id: Mapping[str, Mapping[str, Any]],
    selected_candidate_ids: Sequence[str],
    spec: VoteUpdateSpec,
    ordering_mode: str = LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
) -> dict[str, Any]:
    selected_ids = list(dict.fromkeys(str(candidate_id) for candidate_id in selected_candidate_ids))
    if not selected_ids:
        return {
            "selected_count": 0,
            "applied_count": 0,
            "selected_candidate_ids_hash16": _candidate_ids_hash16(()),
            "applied_candidate_ids_hash16": _candidate_ids_hash16(()),
            "per_state_counts": {},
            "loss": None,
        }
    selected_by_state: dict[str, list[int]] = {}
    for candidate_id in selected_ids:
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise KeyError(f"unknown selected candidate_id {candidate_id!r}")
        state_key = str(candidate["state_key"])
        selected_by_state.setdefault(state_key, []).append(int(candidate["flat_index"]))
    candidate_states = dict(tensor_states)
    applied_candidate_ids: list[str] = []
    per_state_counts: dict[str, dict[str, int]] = {}
    for state_key, flat_indices in sorted(selected_by_state.items()):
        sparse_votes = torch.zeros_like(votes_by_key[state_key], dtype=torch.int16)
        sparse_view = sparse_votes.view(-1)
        full_view = votes_by_key[state_key].view(-1)
        for flat_index in flat_indices:
            sparse_view[int(flat_index)] = full_view[int(flat_index)]
        result = apply_integer_vote_update_reference(
            tensor_states[state_key].vote_update_state(),
            VoteUpdateInputs(votes=sparse_votes),
            spec,
            local_selection_ordering_mode=ordering_mode,
            local_selection_ordering_seed=ORACLE_SCREEN_ORDERING_SEED,
            local_selection_ordering_step=ORACLE_SCREEN_ORDERING_STEP,
        )
        candidate_states[state_key] = make_live_shadow_tensor_state(
            tensor_states[state_key],
            result.q_levels,
            result.accumulators,
        )
        applied_indices = (
            result.plan.applied_indices.detach().cpu().to(torch.int64).tolist()
        )
        applied_ids_for_state = [
            _candidate_id(state_key, int(flat_index))
            for flat_index in applied_indices
        ]
        applied_candidate_ids.extend(applied_ids_for_state)
        per_state_counts[state_key] = {
            "selected_count": len(flat_indices),
            "applied_count": len(applied_ids_for_state),
        }
    variant_loss = _evaluate_loss(
        model,
        batch,
        candidate_states,
        eligible_modules,
        device=device,
        extras=extras,
    )
    return {
        "selected_count": len(selected_ids),
        "applied_count": len(applied_candidate_ids),
        "selected_candidate_ids_hash16": _candidate_ids_hash16(selected_ids),
        "applied_candidate_ids_hash16": _candidate_ids_hash16(applied_candidate_ids),
        "per_state_counts": per_state_counts,
        "loss": float(variant_loss),
    }


def _evaluate_sampled_candidates_for_oracle_screen(
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
    one_flip_spec: VoteUpdateSpec,
    max_seconds: float,
    phase_progress: Any | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, float]:
    budget_start = time.perf_counter()
    sampled_candidates: list[dict[str, Any]] = []
    budget_exceeded = False
    with _maybe_phase(phase_progress, "audit", step=1):
        for candidate_id in sampled_ids:
            if time.perf_counter() - budget_start > float(max_seconds):
                budget_exceeded = True
                break
            candidate = dict(candidate_by_id[candidate_id])
            state_key = str(candidate["state_key"])
            flat_index = int(candidate["flat_index"])
            sparse_votes = torch.zeros_like(votes_by_key[state_key], dtype=torch.int16)
            sparse_votes.view(-1)[flat_index] = votes_by_key[state_key].view(-1)[flat_index]
            result = apply_integer_vote_update_reference(
                tensor_states[state_key].vote_update_state(),
                VoteUpdateInputs(votes=sparse_votes),
                one_flip_spec,
                local_selection_ordering_mode=LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
                local_selection_ordering_seed=ORACLE_SCREEN_ORDERING_SEED,
                local_selection_ordering_step=ORACLE_SCREEN_ORDERING_STEP,
            )
            applied = result.plan.applied_indices.detach().cpu().to(torch.int64).tolist()
            if applied != [flat_index]:
                raise RuntimeError(
                    "oracle screen single-candidate application drifted from the planned candidate identity"
                )
            candidate_states = dict(tensor_states)
            candidate_states[state_key] = make_live_shadow_tensor_state(
                tensor_states[state_key],
                result.q_levels,
                result.accumulators,
            )
            candidate_loss = _evaluate_loss(
                model,
                batch,
                candidate_states,
                eligible_modules,
                device=device,
                extras=extras,
            )
            candidate["candidate_loss"] = float(candidate_loss)
            candidate["local_loss_delta"] = float(candidate_loss - baseline_loss)
            sampled_candidates.append(candidate)
    oracle_top = sorted(
        sampled_candidates,
        key=lambda candidate: (
            float(candidate["local_loss_delta"]),
            str(candidate["candidate_id"]),
        ),
    )
    return (
        sampled_candidates,
        oracle_top,
        bool(budget_exceeded),
        float(time.perf_counter() - budget_start),
    )


def run_credit_ranking_pivot_measurement_oracle_screen(
    *,
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    eligible_modules: Mapping[str, BitLinear],
    device: torch.device,
    max_abs_per_tensor: int,
    extras: Mapping[str, Any],
    max_sampled_candidates: int = PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES,
    max_seconds: float | None = None,
    phase_progress: Any | None = None,
) -> dict[str, Any]:
    if int(max_sampled_candidates) != PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES:
        raise ValueError(
            "credit-ranking pivot measurement requires max_sampled_candidates == 32"
        )
    if max_seconds is None:
        max_seconds = oracle_screen_budget_max_seconds(int(max_sampled_candidates))
    universe = _build_oracle_candidate_universe(
        model=model,
        batch=batch,
        tensor_states=tensor_states,
        eligible_modules=eligible_modules,
        device=device,
        max_abs_per_tensor=int(max_abs_per_tensor),
        extras=extras,
        max_sampled_candidates=int(max_sampled_candidates),
        phase_progress=phase_progress,
    )
    baseline_loss = float(universe["baseline_loss"])
    base_spec = universe["base_spec"]
    one_flip_spec = universe["one_flip_spec"]
    votes_by_key = universe["votes_by_key"]
    candidate_by_id = universe["candidate_by_id"]
    sampled_ids = list(universe["sampled_ids"])
    sampled_candidates, oracle_top, budget_exceeded, elapsed_seconds = (
        _evaluate_sampled_candidates_for_oracle_screen(
            model=model,
            batch=batch,
            tensor_states=tensor_states,
            eligible_modules=eligible_modules,
            device=device,
            extras=extras,
            votes_by_key=votes_by_key,
            candidate_by_id=candidate_by_id,
            sampled_ids=sampled_ids,
            baseline_loss=baseline_loss,
            one_flip_spec=one_flip_spec,
            max_seconds=float(max_seconds),
            phase_progress=phase_progress,
        )
    )
    sampled_count = len(sampled_candidates)
    oracle_top1_delta = float(oracle_top[0]["local_loss_delta"]) if oracle_top else None
    oracle_position_by_id = {
        str(candidate["candidate_id"]): int(position)
        for position, candidate in enumerate(oracle_top)
    }
    for candidate in sampled_candidates:
        oracle_position = oracle_position_by_id.get(str(candidate["candidate_id"]))
        candidate["oracle_best_sampled_rank_position"] = (
            int(oracle_position) if oracle_position is not None else None
        )
        candidate["regret_vs_oracle_top1_local_loss_delta"] = (
            float(candidate["local_loss_delta"] - oracle_top1_delta)
            if oracle_top1_delta is not None
            else None
        )
    score_ids = (
        PIVOT_MEASUREMENT_PRIMARY_SCORE_ID,
        *PIVOT_MEASUREMENT_ABLATION_SCORE_IDS,
    )
    score_family_metrics: dict[str, dict[str, Any]] = {}
    ordered_ids_by_score: dict[str, list[str]] = {}
    positions_by_score: dict[str, dict[str, int]] = {}
    for score_id in score_ids:
        metrics, ordered_ids, position_by_id = _score_family_metrics(
            sampled_candidates=sampled_candidates,
            score_id=score_id,
            oracle_top=oracle_top,
        )
        score_family_metrics[score_id] = metrics
        ordered_ids_by_score[score_id] = ordered_ids
        positions_by_score[score_id] = position_by_id
    primary_metrics = score_family_metrics[PIVOT_MEASUREMENT_PRIMARY_SCORE_ID]
    hash_null_aucs = [
        _pairwise_auc_from_positions(
            sampled_candidates,
            position_by_id={
                candidate_id: int(position)
                for position, candidate_id in enumerate(
                    _hash_ordered_candidate_ids(
                        sampled_candidates,
                        seed=int(seed),
                    )
                )
            },
        )
        for seed in PIVOT_MEASUREMENT_NULL_HASH_SEEDS
    ]
    random_null_aucs = [
        _pairwise_auc_from_positions(
            sampled_candidates,
            position_by_id={
                candidate_id: int(position)
                for position, candidate_id in enumerate(
                    _random_permutation_candidate_ids(
                        sampled_candidates,
                        seed=int(seed),
                    )
                )
            },
        )
        for seed in PIVOT_MEASUREMENT_NULL_RANDOM_SEEDS
    ]
    combined_null_aucs = hash_null_aucs + random_null_aucs
    null_median_auc = median(combined_null_aucs) if combined_null_aucs else 0.5
    null_median_auc_margin = float(primary_metrics["pairwise_auc"] - null_median_auc)
    null_percentile = (
        float(
            sum(
                1
                for null_auc in combined_null_aucs
                if float(null_auc) <= float(primary_metrics["pairwise_auc"])
            )
            / len(combined_null_aucs)
        )
        if combined_null_aucs
        else 0.0
    )
    null_guard_pass = bool(
        null_median_auc_margin >= PIVOT_MEASUREMENT_NULL_AUC_MARGIN_MIN
        and null_percentile >= PIVOT_MEASUREMENT_NULL_PERCENTILE_MIN
    )
    primary_tie_band_groups: dict[str, list[dict[str, Any]]] = {}
    for candidate in sampled_candidates:
        primary_tie_band_groups.setdefault(str(candidate["tie_band_id"]), []).append(candidate)
    oracle_best_candidate = dict(oracle_top[0]) if oracle_top else None
    oracle_best_band = (
        primary_tie_band_groups.get(str(oracle_best_candidate["tie_band_id"]), [])
        if oracle_best_candidate is not None
        else []
    )
    band_deltas = [
        float(candidate["local_loss_delta"])
        for candidate in oracle_best_band
    ]
    band_best_delta = min(band_deltas) if band_deltas else None
    band_worst_delta = max(band_deltas) if band_deltas else None
    band_spread = (
        float(band_worst_delta - band_best_delta)
        if band_best_delta is not None and band_worst_delta is not None
        else None
    )
    if band_spread is None:
        regret_spread_ratio = None
    elif oracle_top1_delta is None:
        regret_spread_ratio = None
    elif abs(float(oracle_top1_delta)) > ORACLE_SCREEN_IMPROVEMENT_EPS:
        regret_spread_ratio = float(band_spread / abs(float(oracle_top1_delta)))
    elif band_spread <= ORACLE_SCREEN_IMPROVEMENT_EPS:
        regret_spread_ratio = 0.0
    else:
        regret_spread_ratio = None
    tie_band_ambiguous = _pivot_tie_band_is_ambiguous(
        oracle_best_candidate_present=oracle_best_candidate is not None,
        band_candidate_count=len(oracle_best_band),
        regret_spread_ratio=regret_spread_ratio,
    )
    primary_ordered_ids = ordered_ids_by_score[PIVOT_MEASUREMENT_PRIMARY_SCORE_ID]
    prefix_ids = primary_ordered_ids[
        : min(len(primary_ordered_ids), 1024)
    ]
    prefix_spec = VoteUpdateSpec(
        threshold_abs=int(base_spec.threshold_abs),
        accumulator_clip_min=int(base_spec.accumulator_clip_min),
        accumulator_clip_max=int(base_spec.accumulator_clip_max),
        max_abs_per_tensor=min(1024, max(1, int(base_spec.max_abs_per_tensor))),
        fraction_per_tensor=float(base_spec.fraction_per_tensor),
        decay_numerator=int(base_spec.decay_numerator),
        decay_denominator=int(base_spec.decay_denominator),
    )
    top1_variant = _evaluate_sparse_selected_candidate_ids(
        model=model,
        batch=batch,
        tensor_states=tensor_states,
        eligible_modules=eligible_modules,
        device=device,
        extras=extras,
        votes_by_key=votes_by_key,
        candidate_by_id=candidate_by_id,
        selected_candidate_ids=primary_ordered_ids[:1],
        spec=one_flip_spec,
    )
    prefix_variant = _evaluate_sparse_selected_candidate_ids(
        model=model,
        batch=batch,
        tensor_states=tensor_states,
        eligible_modules=eligible_modules,
        device=device,
        extras=extras,
        votes_by_key=votes_by_key,
        candidate_by_id=candidate_by_id,
        selected_candidate_ids=prefix_ids,
        spec=prefix_spec,
    )
    current_variant = _evaluate_sparse_selected_candidate_ids(
        model=model,
        batch=batch,
        tensor_states=tensor_states,
        eligible_modules=eligible_modules,
        device=device,
        extras=extras,
        votes_by_key=votes_by_key,
        candidate_by_id=candidate_by_id,
        selected_candidate_ids=[
            str(candidate["candidate_id"])
            for candidate in sorted(
                sampled_candidates,
                key=lambda candidate: (
                    int(candidate["current_rank_position"]),
                    str(candidate["candidate_id"]),
                ),
            )
        ],
        spec=base_spec,
    )
    top1_variant["local_loss_delta"] = (
        float(top1_variant["loss"] - baseline_loss)
        if top1_variant["loss"] is not None
        else None
    )
    prefix_variant["local_loss_delta"] = (
        float(prefix_variant["loss"] - baseline_loss)
        if prefix_variant["loss"] is not None
        else None
    )
    current_variant["local_loss_delta"] = (
        float(current_variant["loss"] - baseline_loss)
        if current_variant["loss"] is not None
        else None
    )
    score_rank_positions = {
        score_id: positions_by_score[score_id]
        for score_id in score_ids
    }
    sampled_candidate_table = [
        {
            "candidate_id": str(candidate["candidate_id"]),
            "state_key": str(candidate["state_key"]),
            "flat_index": int(candidate["flat_index"]),
            "vote_value": int(candidate["vote_value"]),
            "abs_vote_value": int(candidate["abs_vote_value"]),
            "current_margin_abs": int(candidate["current_margin_abs"]),
            "current_rank_position": int(candidate["current_rank_position"]),
            "tie_band_id": str(candidate["tie_band_id"]),
            "score_rank_positions": {
                score_id: int(score_rank_positions[score_id][str(candidate["candidate_id"])])
                for score_id in score_ids
            },
            "candidate_loss": float(candidate["candidate_loss"]),
            "local_loss_delta": float(candidate["local_loss_delta"]),
            "regret_vs_oracle_top1_local_loss_delta": float(
                candidate["regret_vs_oracle_top1_local_loss_delta"]
            )
            if candidate["regret_vs_oracle_top1_local_loss_delta"] is not None
            else None,
            "oracle_best_sampled_rank_position": int(
                candidate["oracle_best_sampled_rank_position"]
            )
            if candidate["oracle_best_sampled_rank_position"] is not None
            else None,
        }
        for candidate in _ordered_candidates_for_pivot_score(
            sampled_candidates,
            score_id=PIVOT_MEASUREMENT_PRIMARY_SCORE_ID,
        )
    ]
    primary_gap_ratio = float(primary_metrics["oracle_top_k_gap_ratio"])
    primary_capture_ratio = float(primary_metrics["oracle_top_k_regret_capture_ratio"])
    primary_rank_position = primary_metrics["oracle_best_sampled_rank_position"]
    primary_poor_rank_threshold = _pivot_poor_rank_position_threshold(sampled_count)
    telemetry = {
        "top_k": min(PIVOT_MEASUREMENT_TOP_K, sampled_count),
        "binary_top_k_ce_improving_capture": {
            score_id: any(
                float(candidate["local_loss_delta"]) < -ORACLE_SCREEN_IMPROVEMENT_EPS
                for candidate in _ordered_candidates_for_pivot_score(
                    sampled_candidates,
                    score_id=score_id,
                )[: min(PIVOT_MEASUREMENT_TOP_K, sampled_count)]
            )
            for score_id in score_ids
        },
        "oracle_top_k_candidate_ids_hash16": _candidate_ids_hash16(
            [
                str(candidate["candidate_id"])
                for candidate in oracle_top[: min(PIVOT_MEASUREMENT_TOP_K, sampled_count)]
            ],
            preserve_order=True,
        ),
        "deterministic_hash_control_only": True,
        "primary_oracle_best_sampled_rank_position_poor_threshold": (
            primary_poor_rank_threshold
        ),
    }
    if not bool(not budget_exceeded and sampled_count > 0):
        branch_classification = BRANCH_MEASUREMENT_AMBIGUOUS_NO_BRANCH
    elif tie_band_ambiguous:
        branch_classification = BRANCH_MEASUREMENT_AMBIGUOUS_TIE_BAND_ALIASING
    elif (
        float(primary_metrics["pairwise_auc"]) >= PIVOT_MEASUREMENT_AUC_PREDICTIVE_MIN
        and null_guard_pass
        and primary_capture_ratio >= 0.50
    ):
        branch_classification = PIVOT_MEASUREMENT_PREDICTIVE_SEED_LABEL
    elif (
        float(primary_metrics["pairwise_auc"]) <= PIVOT_MEASUREMENT_AUC_NON_PREDICTIVE_MAX
        and _pivot_is_poor_rank_position(primary_rank_position, sampled_count)
        and not null_guard_pass
        and primary_gap_ratio > 0.50
    ):
        branch_classification = (
            BRANCH_PREREGISTERED_CHEAP_LEARNER_FEATURE_FAMILY_CANNOT_PREDICT_REGRET
        )
    else:
        branch_classification = BRANCH_MEASUREMENT_AMBIGUOUS_NO_BRANCH
    compact_summary = {
        "candidate_count": len(candidate_by_id),
        "sampled_candidate_count": sampled_count,
        "sampled_candidate_table": sampled_candidate_table,
        "score_family_metrics": {
            "decision_basis": "primary_plus_ablation_report_no_post_hoc_best_of_many",
            "primary_score_id": PIVOT_MEASUREMENT_PRIMARY_SCORE_ID,
            "metrics_by_score_id": score_family_metrics,
        },
        "stage_a_null_guard": {
            "score_id": PIVOT_MEASUREMENT_PRIMARY_SCORE_ID,
            "primary_pairwise_auc": float(primary_metrics["pairwise_auc"]),
            "deterministic_hash_null_aucs": [float(value) for value in hash_null_aucs],
            "random_permutation_null_aucs": [float(value) for value in random_null_aucs],
            "combined_null_median_auc": float(null_median_auc),
            "null_median_auc_margin": float(null_median_auc_margin),
            "null_percentile": float(null_percentile),
            "passes": bool(null_guard_pass),
        },
        "tie_band_ambiguity": {
            "score_id": PIVOT_MEASUREMENT_PRIMARY_SCORE_ID,
            "oracle_best_candidate_id": (
                str(oracle_best_candidate["candidate_id"])
                if oracle_best_candidate is not None
                else None
            ),
            "oracle_best_tie_band_id": (
                str(oracle_best_candidate["tie_band_id"])
                if oracle_best_candidate is not None
                else None
            ),
            "band_candidate_count": len(oracle_best_band),
            "band_candidate_ids_hash16": _candidate_ids_hash16(
                [
                    str(candidate["candidate_id"])
                    for candidate in oracle_best_band
                ]
            ),
            "regret_spread_ratio": float(regret_spread_ratio)
            if regret_spread_ratio is not None
            else None,
            "ambiguous_if_regret_spread_ratio_gt": (
                PIVOT_MEASUREMENT_TIE_BAND_REGRET_SPREAD_RATIO_MAX
            ),
            "ambiguous": bool(tie_band_ambiguous),
        },
        "local_apply_magnitude_smoke": {
            "contract_kind": "local_apply_magnitude_smoke_only",
            "current_spec_is_non_definitive_without_live_full_cap": True,
            "definitive_b_requires_follow_on": True,
            "cap_effective_within_sample": bool(
                int(current_variant["applied_count"]) < int(prefix_variant["applied_count"])
            ),
            "variants": [
                {
                    "variant_id": PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_TOP1,
                    "selection_policy": "primary_score_top1_exact_row",
                    "selected_count": int(top1_variant["selected_count"]),
                    "applied_count": int(top1_variant["applied_count"]),
                    "selected_candidate_ids_hash16": str(
                        top1_variant["selected_candidate_ids_hash16"]
                    ),
                    "applied_candidate_ids_hash16": str(
                        top1_variant["applied_candidate_ids_hash16"]
                    ),
                    "loss": float(top1_variant["loss"])
                    if top1_variant["loss"] is not None
                    else None,
                    "local_loss_delta": float(top1_variant["local_loss_delta"])
                    if top1_variant["local_loss_delta"] is not None
                    else None,
                },
                {
                    "variant_id": PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_PREFIX_CAP_1024,
                    "selection_policy": "primary_score_prefix_cap1024_exact_rows",
                    "selected_count": int(prefix_variant["selected_count"]),
                    "applied_count": int(prefix_variant["applied_count"]),
                    "selected_candidate_ids_hash16": str(
                        prefix_variant["selected_candidate_ids_hash16"]
                    ),
                    "applied_candidate_ids_hash16": str(
                        prefix_variant["applied_candidate_ids_hash16"]
                    ),
                    "loss": float(prefix_variant["loss"])
                    if prefix_variant["loss"] is not None
                    else None,
                    "local_loss_delta": float(prefix_variant["local_loss_delta"])
                    if prefix_variant["local_loss_delta"] is not None
                    else None,
                },
                {
                    "variant_id": PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_CURRENT_SPEC,
                    "selection_policy": "all_sampled_candidates_under_current_local_spec",
                    "selected_count": int(current_variant["selected_count"]),
                    "applied_count": int(current_variant["applied_count"]),
                    "selected_candidate_ids_hash16": str(
                        current_variant["selected_candidate_ids_hash16"]
                    ),
                    "applied_candidate_ids_hash16": str(
                        current_variant["applied_candidate_ids_hash16"]
                    ),
                    "loss": float(current_variant["loss"])
                    if current_variant["loss"] is not None
                    else None,
                    "local_loss_delta": float(current_variant["local_loss_delta"])
                    if current_variant["local_loss_delta"] is not None
                    else None,
                },
            ],
        },
        "telemetry": telemetry,
    }
    return {
        "schema": "hrm_text_158_credit_ranking_pivot_measurement_runtime/v0",
        "mode": ORACLE_SCREEN_MODE_CREDIT_RANKING_PIVOT_MEASUREMENT,
        "same_candidate_set_required": True,
        "screen_rows": int(batch["inputs"].shape[0]),
        "baseline_loss": float(baseline_loss),
        "candidate_count": len(candidate_by_id),
        "sampled_candidate_count": sampled_count,
        "sample_truncated": bool(len(candidate_by_id) > sampled_count),
        "sampled_candidate_ids_hash16": _candidate_ids_hash16(
            [str(candidate["candidate_id"]) for candidate in sampled_candidates]
        ),
        "max_sampled_candidates": int(max_sampled_candidates),
        "max_seconds": float(max_seconds),
        "elapsed_seconds": float(elapsed_seconds),
        "budget_exceeded": bool(budget_exceeded),
        "oracle_feasible": bool(not budget_exceeded),
        "compact_summary": compact_summary,
        "branch_classification": branch_classification,
        "non_persistence": {
            "q_persisted": False,
            "checkpoint_written": False,
            "pt_writes_allowed": False,
            "screen_state_mutated": False,
        },
    }


def run_within_tie_band_discriminator_oracle_screen(
    *,
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    eligible_modules: Mapping[str, BitLinear],
    device: torch.device,
    max_abs_per_tensor: int,
    extras: Mapping[str, Any],
    max_sampled_candidates: int = PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES,
    max_seconds: float | None = None,
    phase_progress: Any | None = None,
) -> dict[str, Any]:
    if int(max_sampled_candidates) != PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES:
        raise ValueError(
            "within-tie-band discriminator requires max_sampled_candidates == 32"
        )
    if max_seconds is None:
        max_seconds = oracle_screen_budget_max_seconds(int(max_sampled_candidates))
    universe = _build_oracle_candidate_universe(
        model=model,
        batch=batch,
        tensor_states=tensor_states,
        eligible_modules=eligible_modules,
        device=device,
        max_abs_per_tensor=int(max_abs_per_tensor),
        extras=extras,
        max_sampled_candidates=int(max_sampled_candidates),
        phase_progress=phase_progress,
    )
    baseline_loss = float(universe["baseline_loss"])
    one_flip_spec = universe["one_flip_spec"]
    votes_by_key = universe["votes_by_key"]
    candidate_by_id = universe["candidate_by_id"]
    sampled_ids = list(universe["sampled_ids"])
    sampled_candidates, oracle_top, budget_exceeded, elapsed_seconds = (
        _evaluate_sampled_candidates_for_oracle_screen(
            model=model,
            batch=batch,
            tensor_states=tensor_states,
            eligible_modules=eligible_modules,
            device=device,
            extras=extras,
            votes_by_key=votes_by_key,
            candidate_by_id=candidate_by_id,
            sampled_ids=sampled_ids,
            baseline_loss=baseline_loss,
            one_flip_spec=one_flip_spec,
            max_seconds=float(max_seconds),
            phase_progress=phase_progress,
        )
    )
    sampled_count = len(sampled_candidates)
    oracle_top1_delta = float(oracle_top[0]["local_loss_delta"]) if oracle_top else None
    oracle_position_by_id = {
        str(candidate["candidate_id"]): int(position)
        for position, candidate in enumerate(oracle_top)
    }
    for candidate in sampled_candidates:
        oracle_position = oracle_position_by_id.get(str(candidate["candidate_id"]))
        candidate["oracle_best_sampled_rank_position"] = (
            int(oracle_position) if oracle_position is not None else None
        )
        candidate["regret_vs_oracle_top1_local_loss_delta"] = (
            float(candidate["local_loss_delta"] - oracle_top1_delta)
            if oracle_top1_delta is not None
            else None
        )
    target_band_candidates = [
        dict(candidate)
        for candidate in sampled_candidates
        if str(candidate["tie_band_id"]) == WITHIN_TIE_BAND_TARGET_TIE_BAND_ID
    ]
    target_band_oracle_top = sorted(
        target_band_candidates,
        key=lambda candidate: (
            float(candidate["local_loss_delta"]),
            str(candidate["candidate_id"]),
        ),
    )
    target_band_oracle_best_candidate = (
        dict(target_band_oracle_top[0]) if target_band_oracle_top else None
    )
    target_band_oracle_top1_delta = (
        float(target_band_oracle_best_candidate["local_loss_delta"])
        if target_band_oracle_best_candidate is not None
        else None
    )
    for candidate in sampled_candidates:
        if (
            str(candidate["tie_band_id"]) == WITHIN_TIE_BAND_TARGET_TIE_BAND_ID
            and target_band_oracle_top1_delta is not None
        ):
            candidate["regret_vs_target_tie_band_oracle_top1_local_loss_delta"] = float(
                candidate["local_loss_delta"] - target_band_oracle_top1_delta
            )
        else:
            candidate["regret_vs_target_tie_band_oracle_top1_local_loss_delta"] = None
    family_ids = (
        WITHIN_TIE_BAND_PRIMARY_FAMILY_ID,
        *WITHIN_TIE_BAND_ABLATION_FAMILY_IDS,
    )
    metrics_by_family_id: dict[str, dict[str, Any]] = {}
    if target_band_oracle_best_candidate is not None:
        for family_id in family_ids:
            metrics_by_family_id[family_id] = _within_tie_band_family_metrics(
                target_band_candidates=target_band_candidates,
                family_id=family_id,
                oracle_best_candidate=target_band_oracle_best_candidate,
                oracle_top1_delta=target_band_oracle_top1_delta,
            )
    primary_metrics = metrics_by_family_id.get(WITHIN_TIE_BAND_PRIMARY_FAMILY_ID)
    predictive = bool(
        primary_metrics is not None
        and float(primary_metrics["oracle_best_bucket_fraction"])
        <= WITHIN_TIE_BAND_PREDICTIVE_BUCKET_FRACTION_MAX
        and primary_metrics["oracle_best_bucket_regret_spread_ratio"] is not None
        and float(primary_metrics["oracle_best_bucket_regret_spread_ratio"])
        <= WITHIN_TIE_BAND_PREDICTIVE_REGRET_SPREAD_RATIO_MAX
        and float(primary_metrics["oracle_best_bucket_regret_capture_ratio"])
        >= WITHIN_TIE_BAND_PREDICTIVE_REGRET_CAPTURE_RATIO_MIN
        and float(
            primary_metrics["matched_hash_null_fraction_gte_observed_bucket_fraction"]
        )
        >= WITHIN_TIE_BAND_MATCHED_HASH_SIGNAL_MIN
        and float(
            primary_metrics["matched_hash_null_fraction_lte_observed_regret_capture_ratio"]
        )
        >= WITHIN_TIE_BAND_MATCHED_HASH_SIGNAL_MIN
    )
    fail_closed = bool(metrics_by_family_id) and all(
        (
            float(metrics["oracle_best_bucket_fraction"])
            > WITHIN_TIE_BAND_FAIL_CLOSED_BUCKET_FRACTION_GT
            or (
                metrics["oracle_best_bucket_regret_spread_ratio"] is not None
                and float(metrics["oracle_best_bucket_regret_spread_ratio"])
                > WITHIN_TIE_BAND_FAIL_CLOSED_REGRET_SPREAD_RATIO_GT
            )
        )
        and float(
            metrics["matched_hash_null_fraction_gte_observed_bucket_fraction"]
        )
        < WITHIN_TIE_BAND_MATCHED_HASH_SIGNAL_MIN
        and float(
            metrics["matched_hash_null_fraction_lte_observed_regret_capture_ratio"]
        )
        < WITHIN_TIE_BAND_MATCHED_HASH_SIGNAL_MIN
        for metrics in metrics_by_family_id.values()
    )
    if not bool(not budget_exceeded and sampled_count > 0):
        branch_classification = BRANCH_WITHIN_TIE_BAND_AMBIGUOUS_NO_BRANCH
    elif not target_band_candidates:
        branch_classification = BRANCH_WITHIN_TIE_BAND_AMBIGUOUS_NO_BRANCH
    elif predictive:
        branch_classification = BRANCH_WITHIN_TIE_BAND_LEARNER_FEATURES_SEPARATE_REGRET
    elif fail_closed:
        branch_classification = BRANCH_WITHIN_TIE_BAND_NEEDS_NEW_LEARNER_STATE
    else:
        branch_classification = BRANCH_WITHIN_TIE_BAND_AMBIGUOUS_NO_BRANCH
    target_band_top_k = min(PIVOT_MEASUREMENT_TOP_K, len(target_band_oracle_top))
    sampled_candidate_table = [
        {
            "candidate_id": str(candidate["candidate_id"]),
            "in_target_tie_band": (
                str(candidate["tie_band_id"]) == WITHIN_TIE_BAND_TARGET_TIE_BAND_ID
            ),
            "state_key": str(candidate["state_key"]),
            "flat_index": int(candidate["flat_index"]),
            "vote_value": int(candidate["vote_value"]),
            "abs_vote_value": int(candidate["abs_vote_value"]),
            "current_margin_abs": int(candidate["current_margin_abs"]),
            "current_rank_position": int(candidate["current_rank_position"]),
            "tie_band_id": str(candidate["tie_band_id"]),
            "current_q_level": int(candidate["current_q_level"]),
            "pre_accumulator_i16": int(candidate["pre_accumulator_i16"]),
            "new_acc_i32_signed": int(candidate["new_acc_i32_signed"]),
            "proposal_direction": int(candidate["proposal_direction"]),
            "threshold_residual_signed": int(candidate["threshold_residual_signed"]),
            "proximity_to_threshold": int(candidate["proximity_to_threshold"]),
            "state_candidate_count": int(candidate["state_candidate_count"]),
            "current_rank_quartile_within_state": int(
                candidate["current_rank_quartile_within_state"]
            ),
            "flat_index_quartile": int(candidate["flat_index_quartile"]),
            "transition_class": str(candidate["transition_class"]),
            "candidate_loss": float(candidate["candidate_loss"]),
            "local_loss_delta": float(candidate["local_loss_delta"]),
            "regret_vs_target_tie_band_oracle_top1_local_loss_delta": (
                float(candidate["regret_vs_target_tie_band_oracle_top1_local_loss_delta"])
                if candidate["regret_vs_target_tie_band_oracle_top1_local_loss_delta"]
                is not None
                else None
            ),
        }
        for candidate in sorted(
            sampled_candidates,
            key=lambda candidate: (
                str(candidate["tie_band_id"]) != WITHIN_TIE_BAND_TARGET_TIE_BAND_ID,
                int(candidate["current_rank_position"]),
                str(candidate["candidate_id"]),
            ),
        )
    ]
    compact_summary = {
        "candidate_count": len(candidate_by_id),
        "sampled_candidate_count": sampled_count,
        "sampled_candidate_table": sampled_candidate_table,
        "target_tie_band": {
            "target_tie_band_id": WITHIN_TIE_BAND_TARGET_TIE_BAND_ID,
            "band_candidate_count": len(target_band_candidates),
            "band_candidate_ids_hash16": _candidate_ids_hash16(
                [
                    str(candidate["candidate_id"])
                    for candidate in target_band_candidates
                ]
            ),
            "band_candidate_fraction_of_sample": (
                float(len(target_band_candidates) / sampled_count)
                if sampled_count > 0
                else 0.0
            ),
            "target_tie_band_oracle_best_candidate_id": (
                str(target_band_oracle_best_candidate["candidate_id"])
                if target_band_oracle_best_candidate is not None
                else None
            ),
            "regret_spread_ratio": _loss_spread_ratio(
                target_band_candidates,
                oracle_top1_delta=target_band_oracle_top1_delta,
            ),
            "top_k": target_band_top_k,
            "top_k_candidate_ids_hash16": _candidate_ids_hash16(
                [
                    str(candidate["candidate_id"])
                    for candidate in target_band_oracle_top[:target_band_top_k]
                ],
                preserve_order=True,
            ),
        },
        "family_metrics": {
            "decision_basis": "primary_plus_ablation_report_no_post_hoc_best_of_many",
            "primary_family_id": WITHIN_TIE_BAND_PRIMARY_FAMILY_ID,
            "metrics_by_family_id": metrics_by_family_id,
        },
        "telemetry": {
            "deterministic_hash_control_only": True,
            "same_candidate_set_required": True,
            "bucket_cardinality_histogram_required": True,
            "singleton_bucket_count_required": True,
        },
    }
    return {
        "schema": "hrm_text_158_within_tie_band_discriminator_runtime/v0",
        "mode": ORACLE_SCREEN_MODE_WITHIN_TIE_BAND_DISCRIMINATOR,
        "same_candidate_set_required": True,
        "screen_rows": int(batch["inputs"].shape[0]),
        "baseline_loss": float(baseline_loss),
        "candidate_count": len(candidate_by_id),
        "sampled_candidate_count": sampled_count,
        "sample_truncated": bool(len(candidate_by_id) > sampled_count),
        "sampled_candidate_ids_hash16": _candidate_ids_hash16(
            [str(candidate["candidate_id"]) for candidate in sampled_candidates]
        ),
        "max_sampled_candidates": int(max_sampled_candidates),
        "max_seconds": float(max_seconds),
        "elapsed_seconds": float(elapsed_seconds),
        "budget_exceeded": bool(budget_exceeded),
        "oracle_feasible": bool(not budget_exceeded),
        "compact_summary": compact_summary,
        "branch_classification": branch_classification,
        "non_persistence": {
            "q_persisted": False,
            "checkpoint_written": False,
            "pt_writes_allowed": False,
            "screen_state_mutated": False,
        },
    }


__all__ = [
    "ORACLE_SCREEN_MODE_CANDIDATE_SET_VIABILITY",
    "ORACLE_SCREEN_MODE_CREDIT_RANKING_PIVOT_MEASUREMENT",
    "ORACLE_SCREEN_MODE_WITHIN_TIE_BAND_DISCRIMINATOR",
    "ORACLE_SCREEN_MODE_CHOICES",
    "_fraction_gte_observed",
    "_fraction_lte_observed",
    "run_credit_ranking_pivot_measurement_oracle_screen",
    "run_candidate_set_viability_oracle_screen",
    "run_within_tie_band_discriminator_oracle_screen",
]
