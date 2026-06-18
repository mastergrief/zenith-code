"""CPU sparse-vote-authority apply receipt for HRM-Text-1.58 Step 3C-C2a.

Documents sparse-only production authority through apply_bounded_delta_vote_step
without dense votes_by_key. Does NOT authorize row flip, GPU runtime, or 2C2 pass_receipt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaLearnerStepResult,
)

PARITY_CONTRACT_MODE_SPARSE_EVENT_SHAPE_ONLY = "SPARSE_EVENT_SHAPE_ONLY"
PARITY_CONTRACT_MODE_DEFERRED_3C_GPU = "DEFERRED_3C_GPU"

BRANCH_3C_C2_SPARSE_APPLY_VIABLE = "BR-3C-C2-SPARSE-APPLY-VIABLE"
BRANCH_3C_C2_DENSE_LEAK = "BR-3C-C2-DENSE-LEAK"
BRANCH_3C_C2_MEASUREMENT_INVALID = "BR-3C-C2-MEASUREMENT-INVALID"
BRANCH_3C_C2_PARITY_DEFERRED = "BR-3C-C2-PARITY-DEFERRED"

SPARSE_VOTE_AUTHORITY_APPLY_SCHEMA_VERSION = (
    "hrm_text_158_sparse_vote_authority_apply/v1"
)
SPARSE_VOTE_AUTHORITY_APPLY_TARGET_NAME = "step3c_c2a_sparse_vote_authority_apply"

SPARSE_VOTE_AUTHORITY_APPLY_NON_CLAIMS = (
    "sparse vote authority apply is CPU apply-shim proof only; no trainer wire",
    "parity_contract_mode=SPARSE_EVENT_SHAPE_ONLY does not claim dense-oracle parity",
    "pass_receipt is always false on this receipt class",
    "no GPU runtime receipt; optimizer_credit_state row not flipped",
)

SPARSE_VOTE_AUTHORITY_HARD_FALSE_FIELDS = (
    "ready_to_flip",
    "optimizer_credit_state_sub2_claim",
    "optimizer_credit_state_resolved",
    "readiness_row_flip_authorized",
    "real_native_integer_attribution_present",
    "real_native_integer_credit_ranking_present",
    "gpu_runtime_receipt_present",
)


@dataclass(frozen=True)
class SparseVoteAuthorityApplyReceipt:
    schema_version: str
    target_name: str
    branch_id: str
    sparse_vote_authority_only: bool
    parity_contract_mode: str
    pass_receipt: bool
    candidate_local_update_pass: bool
    total_sparse_event_count: int
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
            "sparse_vote_authority_only": self.sparse_vote_authority_only,
            "parity_contract_mode": self.parity_contract_mode,
            "pass_receipt": self.pass_receipt,
            "candidate_local_update_pass": self.candidate_local_update_pass,
            "total_sparse_event_count": self.total_sparse_event_count,
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


def sparse_vote_authority_hard_false_snapshot() -> dict[str, bool]:
    return {field: False for field in SPARSE_VOTE_AUTHORITY_HARD_FALSE_FIELDS}


def _count_sparse_events(
    sparse_events_by_key: Mapping[str, Any] | None,
) -> int:
    if sparse_events_by_key is None:
        return 0
    total = 0
    for events in sparse_events_by_key.values():
        if hasattr(events, "event_count"):
            total += int(events.event_count())
        elif isinstance(events, Mapping):
            total += sum(1 for vote in events.values() if int(vote) != 0)
    return total


def classify_sparse_vote_authority_apply_receipt(
    *,
    step_result: BoundedDeltaLearnerStepResult,
    sparse_events_by_key: Mapping[str, Any] | None,
    dense_leak_detected: bool = False,
) -> str:
    if dense_leak_detected:
        return BRANCH_3C_C2_DENSE_LEAK
    summary = step_result.global_summary
    if not bool(summary.get("sparse_vote_authority_only")):
        return BRANCH_3C_C2_MEASUREMENT_INVALID
    if _count_sparse_events(sparse_events_by_key) <= 0:
        return BRANCH_3C_C2_MEASUREMENT_INVALID
    if bool(summary.get("candidate_local_update_pass")):
        return BRANCH_3C_C2_SPARSE_APPLY_VIABLE
    return BRANCH_3C_C2_PARITY_DEFERRED


def build_sparse_vote_authority_apply_receipt(
    *,
    step_result: BoundedDeltaLearnerStepResult,
    sparse_events_by_key: Mapping[str, Any] | None,
    dense_leak_detected: bool = False,
) -> SparseVoteAuthorityApplyReceipt:
    summary = step_result.global_summary
    branch_id = classify_sparse_vote_authority_apply_receipt(
        step_result=step_result,
        sparse_events_by_key=sparse_events_by_key,
        dense_leak_detected=dense_leak_detected,
    )
    hard_false = sparse_vote_authority_hard_false_snapshot()
    receipt = SparseVoteAuthorityApplyReceipt(
        schema_version=SPARSE_VOTE_AUTHORITY_APPLY_SCHEMA_VERSION,
        target_name=SPARSE_VOTE_AUTHORITY_APPLY_TARGET_NAME,
        branch_id=branch_id,
        sparse_vote_authority_only=bool(summary.get("sparse_vote_authority_only")),
        parity_contract_mode=str(
            summary.get("parity_contract_mode", PARITY_CONTRACT_MODE_SPARSE_EVENT_SHAPE_ONLY)
        ),
        pass_receipt=False,
        candidate_local_update_pass=bool(summary.get("candidate_local_update_pass")),
        total_sparse_event_count=_count_sparse_events(sparse_events_by_key),
        non_claims=SPARSE_VOTE_AUTHORITY_APPLY_NON_CLAIMS,
        **hard_false,
    )
    validate_sparse_vote_authority_apply_receipt(receipt)
    return receipt


def validate_sparse_vote_authority_apply_receipt(
    receipt: SparseVoteAuthorityApplyReceipt,
) -> None:
    if receipt.schema_version != SPARSE_VOTE_AUTHORITY_APPLY_SCHEMA_VERSION:
        raise ValueError("sparse vote authority apply receipt schema mismatch")
    if receipt.target_name != SPARSE_VOTE_AUTHORITY_APPLY_TARGET_NAME:
        raise ValueError("sparse vote authority apply receipt target mismatch")
    if receipt.parity_contract_mode != PARITY_CONTRACT_MODE_SPARSE_EVENT_SHAPE_ONLY:
        raise ValueError("parity_contract_mode must be SPARSE_EVENT_SHAPE_ONLY")
    if receipt.pass_receipt:
        raise ValueError("pass_receipt must remain false on sparse vote authority apply")
    for field in SPARSE_VOTE_AUTHORITY_HARD_FALSE_FIELDS:
        if bool(getattr(receipt, field)):
            raise ValueError(f"{field} must remain false on sparse vote authority apply")
    if receipt.non_claims != SPARSE_VOTE_AUTHORITY_APPLY_NON_CLAIMS:
        raise ValueError("sparse vote authority apply non-claims must be exact")
