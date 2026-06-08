"""Thin runtime for the candidate-set-viability oracle screen.

This module keeps the oracle screen off the generic science-arm path. It
reuses the non-mutating vote-update planner to generate one candidate set,
samples a small deterministic subset, and evaluates ephemeral one-flip states
without persisting q or checkpoint artifacts.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
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
    BRANCH_ORACLE_INFEASIBLE_OR_TOO_EXPENSIVE,
    ORACLE_ARM_CURRENT_CREDIT_RANK_BUCKET_CURRENT_ORDER,
    ORACLE_ARM_DETERMINISTIC_HASH_SAME_VOTES,
    ORACLE_ARM_DIAGNOSTIC_LOCAL_LOSS_DELTA,
    ORACLE_SCREEN_FEASIBILITY_MAX_SAMPLED_CANDIDATES,
    ORACLE_SCREEN_FEASIBILITY_MAX_SECONDS,
    classify_candidate_set_viability_oracle_screen,
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
ORACLE_SCREEN_MODE_CHOICES = (ORACLE_SCREEN_MODE_CANDIDATE_SET_VIABILITY,)
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


def _rank_decile(position: int | None, total: int) -> int | None:
    if position is None or total <= 0:
        return None
    return min(9, int((int(position) * 10) / max(1, int(total))))


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
            "ce_improving_candidate_count": sum(
                1 for delta in loss_deltas if delta < -ORACLE_SCREEN_IMPROVEMENT_EPS
            ),
        },
        "paired_loss_branch_fields": {
            **branch_inputs,
            "current_top1_local_loss_delta": current_top1_delta,
            "oracle_top1_local_loss_delta": oracle_top1_delta,
        },
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
        "branch_inputs": branch_inputs,
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
    "ORACLE_SCREEN_MODE_CHOICES",
    "run_candidate_set_viability_oracle_screen",
]
