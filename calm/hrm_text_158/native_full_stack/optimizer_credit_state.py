"""Fail-closed optimizer/credit-state blocker receipt for HRM-Text-1.58.

This is a CPU/audit-lane contract only. It names the dense transient credit
debt that keeps optimizer_credit_state blocked and rejects laundering that debt
as an FP exception, row flip, or full-sub2 readiness claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION = (
    "hrm_text_158_optimizer_credit_state_fail_closed/v0.dense_transient_credit_debt"
)
OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_TARGET_NAME = (
    "step3c_optimizer_credit_state_fail_closed"
)

OPTIMIZER_CREDIT_STATE_REQUIRED_DEBT_ANCHORS = (
    "weighted_grad",
    "credit",
    "projected_moves",
    "dense_rank_votes_before_sparse_event_extraction",
    "optimizer_credit_state_resolved_false",
    "credit_ranking_update_law_pivot_deferred",
)
OPTIMIZER_CREDIT_STATE_ALLOWED_DEBT_ANCHORS = (
    OPTIMIZER_CREDIT_STATE_REQUIRED_DEBT_ANCHORS
)

OPTIMIZER_CREDIT_STATE_BLOCKED_REASON = (
    "fail-closed optimizer/credit-state harness only; dense transient "
    "weighted_grad, credit, projected_moves, and dense_rank_votes remain "
    "proof-only over-2 tensors while optimizer_credit_state_resolved is false "
    "and no real native integer attribution/credit/ranking, no-hidden-FP "
    "optimizer-state, or GPU/runtime receipt is present"
)
OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT = (
    "credit_capture_tensors is attribution-only transient FP debt and cannot "
    "satisfy or flip the optimizer_credit_state readiness row"
)
OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS = (
    "optimizer/credit-state blocker refinement is not learning, acquisition, retention, or throughput",
    "dense weighted_grad, credit, projected_moves, and dense rank votes remain proof-only transient over-2 debt",
    "credit_capture_tensors is attribution-only and cannot satisfy the optimizer_credit_state row",
    "optimizer_credit_state_resolved=false remains the current proof boundary",
    "this receipt does not launch GPU, prove runtime residency, write checkpoints, or mutate .pt artifacts",
)

_DEFAULT_DEBT_ANCHORS = (
    {
        "anchor_name": "weighted_grad",
        "source_anchor": "calm/hrm_text_158/native_full_stack/bounded_delta_learner.py:284",
        "evidence": "weighted_grad_from_captures builds a dense float32 tensor from captured inputs and grad_outputs",
    },
    {
        "anchor_name": "credit",
        "source_anchor": "calm/hrm_text_158/native_full_stack/bounded_delta_learner.py:307",
        "evidence": "credit_from_weighted_grad derives dense credit from the dense weighted_grad tensor",
    },
    {
        "anchor_name": "projected_moves",
        "source_anchor": "calm/hrm_text_158/native_full_stack/bounded_delta_learner.py:220",
        "evidence": "projected moves are dense tensor decisions before sparse vote extraction",
    },
    {
        "anchor_name": "dense_rank_votes_before_sparse_event_extraction",
        "source_anchor": "calm/hrm_text_158/native_full_stack/bounded_delta_learner.py:226",
        "evidence": "rank_bucketed_int16_votes allocates dense int16 votes before sparse event extraction",
    },
    {
        "anchor_name": "optimizer_credit_state_resolved_false",
        "source_anchor": "calm/hrm_text_158/native_full_stack/trainer_sub2_authority.py:1402",
        "evidence": "2C4a roundtrip receipt explicitly records optimizer_credit_state_resolved=False",
    },
    {
        "anchor_name": "credit_ranking_update_law_pivot_deferred",
        "source_anchor": "calm/hrm_text_158/native_full_stack/trainer_sub2_authority.py:1403",
        "evidence": "2C4a defers the credit-ranking/update-law pivot",
    },
)


@dataclass(frozen=True)
class OptimizerCreditDebtAnchorObservation:
    anchor_name: str
    source_anchor: str
    evidence: str
    debt_kind: str = "dense_transient_over2_credit_debt"

    def to_dict(self) -> dict[str, str]:
        return {
            "anchor_name": self.anchor_name,
            "source_anchor": self.source_anchor,
            "evidence": self.evidence,
            "debt_kind": self.debt_kind,
        }


@dataclass(frozen=True)
class OptimizerCreditStateFailClosedReceipt:
    schema_version: str
    target_name: str
    allowed_debt_anchors: tuple[str, ...]
    required_debt_anchors: tuple[str, ...]
    optimizer_credit_state_sub2_claim: bool
    optimizer_credit_state_resolved: bool
    readiness_row_flip_authorized: bool
    fp_exception_laundering_claim: bool
    real_native_integer_attribution_present: bool
    real_native_integer_credit_ranking_present: bool
    no_hidden_bf16_fp_optimizer_state_proven: bool
    gpu_runtime_receipt_present: bool
    ready_to_flip: bool
    blocked_reason: str
    debt_anchors: tuple[OptimizerCreditDebtAnchorObservation, ...]
    fp_exception_caveat: str
    smallest_missing_proof: str
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_name": self.target_name,
            "allowed_debt_anchors": list(self.allowed_debt_anchors),
            "required_debt_anchors": list(self.required_debt_anchors),
            "optimizer_credit_state_sub2_claim": self.optimizer_credit_state_sub2_claim,
            "optimizer_credit_state_resolved": self.optimizer_credit_state_resolved,
            "readiness_row_flip_authorized": self.readiness_row_flip_authorized,
            "fp_exception_laundering_claim": self.fp_exception_laundering_claim,
            "real_native_integer_attribution_present": (
                self.real_native_integer_attribution_present
            ),
            "real_native_integer_credit_ranking_present": (
                self.real_native_integer_credit_ranking_present
            ),
            "no_hidden_bf16_fp_optimizer_state_proven": (
                self.no_hidden_bf16_fp_optimizer_state_proven
            ),
            "gpu_runtime_receipt_present": self.gpu_runtime_receipt_present,
            "ready_to_flip": self.ready_to_flip,
            "blocked_reason": self.blocked_reason,
            "debt_anchors": [anchor.to_dict() for anchor in self.debt_anchors],
            "fp_exception_caveat": self.fp_exception_caveat,
            "smallest_missing_proof": self.smallest_missing_proof,
            "non_claims": list(self.non_claims),
        }


def _require_nonempty_string(value: object, *, field_name: str) -> str:
    text = str(value)
    if not text.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _summarize_optimizer_credit_debt_anchors(
    debt_anchors: Sequence[Mapping[str, object]],
) -> tuple[OptimizerCreditDebtAnchorObservation, ...]:
    grouped: dict[str, Mapping[str, object] | None] = {
        anchor: None for anchor in OPTIMIZER_CREDIT_STATE_ALLOWED_DEBT_ANCHORS
    }
    for item in debt_anchors:
        anchor_name = item.get("anchor_name", item.get("name"))
        if anchor_name not in grouped:
            raise ValueError(
                "optimizer_credit_state receipt debt anchors must be exactly the "
                f"Step 3C allowlist {OPTIMIZER_CREDIT_STATE_ALLOWED_DEBT_ANCHORS!r}; "
                f"got {anchor_name!r}"
            )
        if grouped[str(anchor_name)] is not None:
            raise ValueError(f"duplicate optimizer_credit_state debt anchor: {anchor_name!r}")
        grouped[str(anchor_name)] = item

    missing = [anchor for anchor, item in grouped.items() if item is None]
    if missing:
        raise ValueError(
            "optimizer_credit_state receipt missing required debt anchors: "
            + ", ".join(missing)
        )

    observations: list[OptimizerCreditDebtAnchorObservation] = []
    for anchor_name in OPTIMIZER_CREDIT_STATE_REQUIRED_DEBT_ANCHORS:
        item = grouped[anchor_name]
        assert item is not None
        observations.append(
            OptimizerCreditDebtAnchorObservation(
                anchor_name=anchor_name,
                source_anchor=_require_nonempty_string(
                    item.get("source_anchor", ""),
                    field_name=f"{anchor_name}.source_anchor",
                ),
                evidence=_require_nonempty_string(
                    item.get("evidence", ""),
                    field_name=f"{anchor_name}.evidence",
                ),
                debt_kind=_require_nonempty_string(
                    item.get("debt_kind", "dense_transient_over2_credit_debt"),
                    field_name=f"{anchor_name}.debt_kind",
                ),
            )
        )
    return tuple(observations)


def build_optimizer_credit_state_fail_closed_receipt(
    *,
    debt_anchors: Sequence[Mapping[str, object]] = _DEFAULT_DEBT_ANCHORS,
    optimizer_credit_state_sub2_claim: bool = False,
    optimizer_credit_state_resolved: bool = False,
    readiness_row_flip_authorized: bool = False,
    fp_exception_laundering_claim: bool = False,
    real_native_integer_attribution_present: bool = False,
    real_native_integer_credit_ranking_present: bool = False,
    no_hidden_bf16_fp_optimizer_state_proven: bool = False,
    gpu_runtime_receipt_present: bool = False,
    ready_to_flip: bool = False,
    smallest_missing_proof: str = (
        "real native integer attribution/credit/ranking replacement, "
        "no-hidden-BF16/FP optimizer-state proof, and GPU/runtime receipt"
    ),
) -> OptimizerCreditStateFailClosedReceipt:
    """Build the Step 3C fail-closed optimizer/credit-state blocker receipt."""

    receipt = OptimizerCreditStateFailClosedReceipt(
        schema_version=OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION,
        target_name=OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_TARGET_NAME,
        allowed_debt_anchors=OPTIMIZER_CREDIT_STATE_ALLOWED_DEBT_ANCHORS,
        required_debt_anchors=OPTIMIZER_CREDIT_STATE_REQUIRED_DEBT_ANCHORS,
        optimizer_credit_state_sub2_claim=bool(optimizer_credit_state_sub2_claim),
        optimizer_credit_state_resolved=bool(optimizer_credit_state_resolved),
        readiness_row_flip_authorized=bool(readiness_row_flip_authorized),
        fp_exception_laundering_claim=bool(fp_exception_laundering_claim),
        real_native_integer_attribution_present=bool(
            real_native_integer_attribution_present
        ),
        real_native_integer_credit_ranking_present=bool(
            real_native_integer_credit_ranking_present
        ),
        no_hidden_bf16_fp_optimizer_state_proven=bool(
            no_hidden_bf16_fp_optimizer_state_proven
        ),
        gpu_runtime_receipt_present=bool(gpu_runtime_receipt_present),
        ready_to_flip=bool(ready_to_flip),
        blocked_reason=OPTIMIZER_CREDIT_STATE_BLOCKED_REASON,
        debt_anchors=_summarize_optimizer_credit_debt_anchors(debt_anchors),
        fp_exception_caveat=OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT,
        smallest_missing_proof=_require_nonempty_string(
            smallest_missing_proof,
            field_name="smallest_missing_proof",
        ),
        non_claims=OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS,
    )
    validate_optimizer_credit_state_fail_closed_receipt(receipt)
    return receipt


def validate_optimizer_credit_state_fail_closed_receipt(
    receipt: OptimizerCreditStateFailClosedReceipt,
) -> None:
    if (
        receipt.schema_version
        != OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION
    ):
        raise ValueError("optimizer_credit_state fail-closed receipt schema mismatch")
    if receipt.target_name != OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_TARGET_NAME:
        raise ValueError("optimizer_credit_state fail-closed receipt target mismatch")
    if (
        receipt.allowed_debt_anchors
        != OPTIMIZER_CREDIT_STATE_ALLOWED_DEBT_ANCHORS
    ):
        raise ValueError("optimizer_credit_state allowed debt anchors must be exact")
    if (
        receipt.required_debt_anchors
        != OPTIMIZER_CREDIT_STATE_REQUIRED_DEBT_ANCHORS
    ):
        raise ValueError("optimizer_credit_state required debt anchors must be exact")
    observed_names = tuple(anchor.anchor_name for anchor in receipt.debt_anchors)
    if observed_names != OPTIMIZER_CREDIT_STATE_REQUIRED_DEBT_ANCHORS:
        raise ValueError(
            "optimizer_credit_state debt anchors must match the required Step 3C set"
        )
    for anchor in receipt.debt_anchors:
        if not anchor.source_anchor or not anchor.evidence:
            raise ValueError(f"{anchor.anchor_name} is missing anchor evidence")
        if "dense_transient" not in anchor.debt_kind:
            raise ValueError(
                f"{anchor.anchor_name} must remain classified as dense transient debt"
            )

    future_proof_gate = (
        receipt.real_native_integer_attribution_present
        and receipt.real_native_integer_credit_ranking_present
        and receipt.no_hidden_bf16_fp_optimizer_state_proven
        and receipt.gpu_runtime_receipt_present
    )
    laundering_claims = {
        "optimizer_credit_state_sub2_claim": (
            receipt.optimizer_credit_state_sub2_claim
        ),
        "optimizer_credit_state_resolved": receipt.optimizer_credit_state_resolved,
        "readiness_row_flip_authorized": receipt.readiness_row_flip_authorized,
        "fp_exception_laundering_claim": receipt.fp_exception_laundering_claim,
    }
    for label, value in laundering_claims.items():
        if bool(value) and not (future_proof_gate and receipt.ready_to_flip):
            raise ValueError(
                f"{label} requires real native integer attribution/credit/ranking "
                "proof, no-hidden-BF16/FP optimizer-state proof, GPU/runtime "
                "receipt, and ready_to_flip=True"
            )
    if receipt.ready_to_flip and not (
        future_proof_gate
        and receipt.optimizer_credit_state_sub2_claim
        and receipt.optimizer_credit_state_resolved
        and receipt.readiness_row_flip_authorized
        and not receipt.fp_exception_laundering_claim
    ):
        raise ValueError(
            "ready_to_flip cannot be true without native proof gates and explicit "
            "non-laundered optimizer-credit row resolution"
        )
    if receipt.blocked_reason != OPTIMIZER_CREDIT_STATE_BLOCKED_REASON:
        raise ValueError("optimizer_credit_state blocked reason must be exact")
    if receipt.fp_exception_caveat != OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT:
        raise ValueError(
            "optimizer_credit_state must keep credit_capture_tensors attribution-only"
        )
    if receipt.non_claims != OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS:
        raise ValueError("optimizer_credit_state receipt non-claims must be exact")
