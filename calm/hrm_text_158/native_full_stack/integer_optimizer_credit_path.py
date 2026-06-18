"""CPU integer optimizer credit-path wire for HRM-Text-1.58 Step 3C-C2b (Option A).

Separate validated wire receipt exercising trainer captures through the integer
3C-A/3C-B chain and banked C2a sparse-authority apply. Does NOT modify public
2C2/2C4a/P1b trainer receipt builders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
    AuthoritativeForwardHandle,
    BoundedDeltaLearnerStepResult,
    BoundedDeltaTensorState,
    RankVoteSpec,
    apply_bounded_delta_vote_step,
    default_dry_run_rank_vote_spec,
)
from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
    integer_marginal_attribution_from_captures,
    projected_moves_from_integer_attribution,
)
from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (
    CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    sparse_rank_votes_from_attribution_events,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_authority_apply import (
    PARITY_CONTRACT_MODE_SPARSE_EVENT_SHAPE_ONLY,
    build_sparse_vote_authority_apply_receipt,
    validate_sparse_vote_authority_apply_receipt,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    derive_trainer_sub2_authority_states,
    select_trainer_eligible_bitlinears,
    trainer_authoritative_forward_context,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec

INTEGER_OPTIMIZER_CREDIT_PATH_ENABLED = False

INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_SCHEMA_VERSION = (
    "hrm_text_158_integer_optimizer_credit_path_wire/v1"
)
INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_TARGET_NAME = (
    "step3c_c2b_integer_optimizer_credit_path_wire"
)
WIRE_SEAM_TRAINER_2C2_EQUIVALENT_CAPTURE_BACKWARD = (
    "trainer_2c2_equivalent_capture_backward"
)

BRANCH_3C_C2B_WIRE_VIABLE = "BR-3C-C2B-WIRE-VIABLE"
BRANCH_3C_C2B_DENSE_LEAK = "BR-3C-C2B-DENSE-LEAK"
BRANCH_3C_C2B_MEASUREMENT_INVALID = "BR-3C-C2B-MEASUREMENT-INVALID"

INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_NON_CLAIMS = (
    "integer optimizer credit path wire is CPU Option-A proof only; no public trainer receipt mutation",
    "parity_contract_mode=SPARSE_EVENT_SHAPE_ONLY does not claim dense-oracle parity",
    "pass_receipt is always false on this receipt class",
    "no GPU runtime receipt; optimizer_credit_state row not flipped",
    "2C4a/P1b in-builder wire deferred to C2b-beta",
)

INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_HARD_FALSE_FIELDS = (
    "ready_to_flip",
    "optimizer_credit_state_sub2_claim",
    "optimizer_credit_state_resolved",
    "readiness_row_flip_authorized",
    "real_native_integer_attribution_present",
    "real_native_integer_credit_ranking_present",
    "gpu_runtime_receipt_present",
)


def default_integer_optimizer_credit_path_vote_update_spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=1,
        max_abs_per_tensor=2,
        fraction_per_tensor=1.0,
    )


def integer_optimizer_credit_path_hard_false_snapshot() -> dict[str, bool]:
    return {field: False for field in INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_HARD_FALSE_FIELDS}


def _sparse_event_counts_by_key(
    sparse_events_by_key: Mapping[str, SparseVoteEvents],
) -> dict[str, int]:
    return {str(key): int(events.event_count()) for key, events in sorted(sparse_events_by_key.items())}


def emit_integer_sparse_vote_events_from_trainer_handle(
    handle: AuthoritativeForwardHandle,
    states: Mapping[str, BoundedDeltaTensorState],
    rank_spec: RankVoteSpec,
    *,
    credit_law_id: str = CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
) -> dict[str, SparseVoteEvents]:
    sparse_events_by_key: dict[str, SparseVoteEvents] = {}
    for state_key, state in sorted(states.items()):
        if state_key not in handle.captures:
            raise KeyError(f"missing capture for state key {state_key!r}")
        capture = handle.captures[state_key]
        if not capture["inputs"] or not capture["grad_outputs"]:
            raise RuntimeError(
                f"integer wire path requires captured inputs/grad_outputs for {state_key!r}"
            )
        weight_shape = tuple(int(dim) for dim in state.q_levels.shape)
        attribution_events = integer_marginal_attribution_from_captures(
            capture["inputs"],
            capture["grad_outputs"],
            weight_shape=weight_shape,
        )
        move_indices, moves = projected_moves_from_integer_attribution(
            attribution_events,
            state.q_levels.reshape(-1),
        )
        sparse_events_by_key[state_key] = sparse_rank_votes_from_attribution_events(
            attribution_events,
            move_indices,
            moves,
            rank_spec,
            credit_law_id=credit_law_id,
        )
    return sparse_events_by_key


def apply_integer_optimizer_credit_path_step(
    states: Mapping[str, BoundedDeltaTensorState],
    sparse_events_by_key: Mapping[str, SparseVoteEvents],
    vote_specs_by_key: Mapping[str, VoteUpdateSpec],
) -> BoundedDeltaLearnerStepResult:
    return apply_bounded_delta_vote_step(
        states,
        None,
        vote_specs_by_key,
        candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        candidate_sparse_vote_events_by_key=sparse_events_by_key,
        candidate_oracle_control_enabled=False,
        sparse_vote_authority_only=True,
    )


@dataclass(frozen=True)
class IntegerOptimizerCreditPathWireReceipt:
    schema_version: str
    target_name: str
    branch_id: str
    integer_optimizer_credit_path_enabled: bool
    wire_seam: str
    sparse_events_emitted_by_key: dict[str, int]
    dense_credit_path_materialized: bool
    oracle_parity_proof_executed: bool
    parity_contract_mode: str
    pass_receipt: bool
    candidate_local_update_pass: bool
    total_sparse_event_count: int
    sparse_vote_authority_apply_receipt: dict[str, Any]
    ready_to_flip: bool
    optimizer_credit_state_sub2_claim: bool
    optimizer_credit_state_resolved: bool
    readiness_row_flip_authorized: bool
    real_native_integer_attribution_present: bool
    real_native_integer_credit_ranking_present: bool
    gpu_runtime_receipt_present: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_name": self.target_name,
            "branch_id": self.branch_id,
            "integer_optimizer_credit_path_enabled": self.integer_optimizer_credit_path_enabled,
            "wire_seam": self.wire_seam,
            "sparse_events_emitted_by_key": dict(self.sparse_events_emitted_by_key),
            "dense_credit_path_materialized": self.dense_credit_path_materialized,
            "oracle_parity_proof_executed": self.oracle_parity_proof_executed,
            "parity_contract_mode": self.parity_contract_mode,
            "pass_receipt": self.pass_receipt,
            "candidate_local_update_pass": self.candidate_local_update_pass,
            "total_sparse_event_count": self.total_sparse_event_count,
            "sparse_vote_authority_apply_receipt": dict(self.sparse_vote_authority_apply_receipt),
            "ready_to_flip": self.ready_to_flip,
            "optimizer_credit_state_sub2_claim": self.optimizer_credit_state_sub2_claim,
            "optimizer_credit_state_resolved": self.optimizer_credit_state_resolved,
            "readiness_row_flip_authorized": self.readiness_row_flip_authorized,
            "real_native_integer_attribution_present": (
                self.real_native_integer_attribution_present
            ),
            "real_native_integer_credit_ranking_present": (
                self.real_native_integer_credit_ranking_present
            ),
            "gpu_runtime_receipt_present": self.gpu_runtime_receipt_present,
            "non_claims": list(self.non_claims),
        }


def classify_integer_optimizer_credit_path_wire_receipt(
    *,
    step_result: BoundedDeltaLearnerStepResult,
    sparse_events_by_key: Mapping[str, SparseVoteEvents],
    dense_leak_detected: bool = False,
) -> str:
    if dense_leak_detected:
        return BRANCH_3C_C2B_DENSE_LEAK
    event_counts = _sparse_event_counts_by_key(sparse_events_by_key)
    if not event_counts or all(int(count) <= 0 for count in event_counts.values()):
        return BRANCH_3C_C2B_MEASUREMENT_INVALID
    summary = step_result.global_summary
    if not bool(summary.get("sparse_vote_authority_only")):
        return BRANCH_3C_C2B_MEASUREMENT_INVALID
    if bool(summary.get("candidate_local_update_pass")):
        return BRANCH_3C_C2B_WIRE_VIABLE
    return BRANCH_3C_C2B_MEASUREMENT_INVALID


def build_integer_optimizer_credit_path_wire_receipt_from_step(
    *,
    step_result: BoundedDeltaLearnerStepResult,
    sparse_events_by_key: Mapping[str, SparseVoteEvents],
    dense_leak_detected: bool = False,
) -> IntegerOptimizerCreditPathWireReceipt:
    sparse_apply_receipt = build_sparse_vote_authority_apply_receipt(
        step_result=step_result,
        sparse_events_by_key=sparse_events_by_key,
        dense_leak_detected=dense_leak_detected,
    )
    event_counts = _sparse_event_counts_by_key(sparse_events_by_key)
    total_sparse_events = sum(int(count) for count in event_counts.values())
    hard_false = integer_optimizer_credit_path_hard_false_snapshot()
    receipt = IntegerOptimizerCreditPathWireReceipt(
        schema_version=INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_SCHEMA_VERSION,
        target_name=INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_TARGET_NAME,
        branch_id=classify_integer_optimizer_credit_path_wire_receipt(
            step_result=step_result,
            sparse_events_by_key=sparse_events_by_key,
            dense_leak_detected=dense_leak_detected,
        ),
        integer_optimizer_credit_path_enabled=True,
        wire_seam=WIRE_SEAM_TRAINER_2C2_EQUIVALENT_CAPTURE_BACKWARD,
        sparse_events_emitted_by_key=event_counts,
        dense_credit_path_materialized=False,
        oracle_parity_proof_executed=False,
        parity_contract_mode=PARITY_CONTRACT_MODE_SPARSE_EVENT_SHAPE_ONLY,
        pass_receipt=False,
        candidate_local_update_pass=bool(
            step_result.global_summary.get("candidate_local_update_pass")
        ),
        total_sparse_event_count=int(total_sparse_events),
        sparse_vote_authority_apply_receipt=sparse_apply_receipt.to_dict(),
        non_claims=INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_NON_CLAIMS,
        **hard_false,
    )
    validate_integer_optimizer_credit_path_wire_receipt(receipt)
    return receipt


def validate_integer_optimizer_credit_path_wire_receipt(
    receipt: IntegerOptimizerCreditPathWireReceipt,
) -> None:
    if receipt.schema_version != INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_SCHEMA_VERSION:
        raise ValueError("integer optimizer credit path wire receipt schema mismatch")
    if receipt.target_name != INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_TARGET_NAME:
        raise ValueError("integer optimizer credit path wire receipt target mismatch")
    if not receipt.integer_optimizer_credit_path_enabled:
        raise ValueError("integer_optimizer_credit_path_enabled must be true on wire receipt")
    if receipt.wire_seam != WIRE_SEAM_TRAINER_2C2_EQUIVALENT_CAPTURE_BACKWARD:
        raise ValueError("wire seam must be trainer_2c2_equivalent_capture_backward")
    if receipt.dense_credit_path_materialized:
        raise ValueError("dense_credit_path_materialized must remain false")
    if receipt.oracle_parity_proof_executed:
        raise ValueError("oracle_parity_proof_executed must remain false")
    if receipt.parity_contract_mode != PARITY_CONTRACT_MODE_SPARSE_EVENT_SHAPE_ONLY:
        raise ValueError("parity_contract_mode must be SPARSE_EVENT_SHAPE_ONLY")
    if receipt.pass_receipt:
        raise ValueError("pass_receipt must remain false on integer optimizer credit path wire")
    for field in INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_HARD_FALSE_FIELDS:
        if bool(getattr(receipt, field)):
            raise ValueError(f"{field} must remain false on integer optimizer credit path wire")
    if receipt.non_claims != INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_NON_CLAIMS:
        raise ValueError("integer optimizer credit path wire non-claims must be exact")
    if not isinstance(receipt.sparse_vote_authority_apply_receipt, dict):
        raise ValueError("sparse_vote_authority_apply_receipt must be embedded dict")
    validate_sparse_vote_authority_apply_receipt_dict(receipt.sparse_vote_authority_apply_receipt)


def validate_sparse_vote_authority_apply_receipt_dict(payload: Mapping[str, Any]) -> None:
    from calm.hrm_text_158.native_full_stack.sparse_vote_authority_apply import (
        SparseVoteAuthorityApplyReceipt,
    )

    validate_sparse_vote_authority_apply_receipt(
        SparseVoteAuthorityApplyReceipt(
            schema_version=str(payload["schema_version"]),
            target_name=str(payload["target_name"]),
            branch_id=str(payload["branch_id"]),
            sparse_vote_authority_only=bool(payload["sparse_vote_authority_only"]),
            parity_contract_mode=str(payload["parity_contract_mode"]),
            pass_receipt=bool(payload["pass_receipt"]),
            candidate_local_update_pass=bool(payload["candidate_local_update_pass"]),
            total_sparse_event_count=int(payload["total_sparse_event_count"]),
            ready_to_flip=bool(payload["ready_to_flip"]),
            optimizer_credit_state_sub2_claim=bool(payload["optimizer_credit_state_sub2_claim"]),
            optimizer_credit_state_resolved=bool(payload["optimizer_credit_state_resolved"]),
            readiness_row_flip_authorized=bool(payload["readiness_row_flip_authorized"]),
            real_native_integer_attribution_present=bool(
                payload["real_native_integer_attribution_present"]
            ),
            real_native_integer_credit_ranking_present=bool(
                payload["real_native_integer_credit_ranking_present"]
            ),
            gpu_runtime_receipt_present=bool(payload["gpu_runtime_receipt_present"]),
            non_claims=tuple(str(item) for item in payload["non_claims"]),
        )
    )


def build_integer_optimizer_credit_path_wire_receipt(
    model: torch.nn.Module,
    *,
    batch: Mapping[str, Any],
    forward_loss_fn: Callable[[torch.nn.Module, Mapping[str, Any]], torch.Tensor],
    use_ternary_bulk: bool,
    eligible_scope: str = "all-bitlinear",
    device: torch.device | str = "cpu",
    rank_spec: RankVoteSpec | None = None,
    vote_update_spec: VoteUpdateSpec | None = None,
    credit_law_id: str = CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
) -> IntegerOptimizerCreditPathWireReceipt:
    eligible = select_trainer_eligible_bitlinears(
        model,
        use_ternary_bulk=use_ternary_bulk,
        eligible_scope=eligible_scope,
    )
    states = derive_trainer_sub2_authority_states(eligible)
    rank_vote_spec = rank_spec or default_dry_run_rank_vote_spec()
    update_spec = vote_update_spec or default_integer_optimizer_credit_path_vote_update_spec()
    vote_specs_by_key = {key: update_spec for key in states}
    prior_training = bool(model.training)
    try:
        model.train(True)
        model.zero_grad(set_to_none=True)
        with trainer_authoritative_forward_context(
            eligible,
            states,
            device=device,
            requires_grad=True,
        ) as handle:
            loss = forward_loss_fn(model, batch)
            if not isinstance(loss, torch.Tensor):
                raise TypeError("integer wire forward_loss_fn must return a torch.Tensor loss")
            loss_to_backward = loss if loss.numel() == 1 else loss.mean()
            if not bool(torch.isfinite(loss_to_backward.detach()).item()):
                raise ValueError("integer wire path requires finite loss")
            loss_to_backward.backward()
            sparse_events_by_key = emit_integer_sparse_vote_events_from_trainer_handle(
                handle,
                states,
                rank_vote_spec,
                credit_law_id=credit_law_id,
            )
    finally:
        model.train(prior_training)

    step_result = apply_integer_optimizer_credit_path_step(
        states,
        sparse_events_by_key,
        vote_specs_by_key,
    )
    return build_integer_optimizer_credit_path_wire_receipt_from_step(
        step_result=step_result,
        sparse_events_by_key=sparse_events_by_key,
    )
