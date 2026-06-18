"""CPU no-hidden-FP audit scaffold for HRM-Text-1.58 Step 3C-C1.

Fail-closed audit over optimizer exclusion checks and dense-surface observation.
Does NOT authorize row flip, GPU runtime receipt, or real_native_integer_* present.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence
from unittest import mock

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    default_dry_run_rank_vote_spec,
)
from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
    INTEGER_MARGINAL_ATTRIBUTION_LAW_ID,
    integer_marginal_attribution_from_captures,
    projected_moves_from_integer_attribution,
)
from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (
    CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    sparse_rank_bucketed_vote_events_from_integer_credit,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (
    BRANCH_3C_C_AUDIT_PASS_CPU,
    BRANCH_3C_C_CAPTURE_LAUNDER,
    BRANCH_3C_C_DENSE_LEAK,
    BRANCH_3C_C_MEASUREMENT_INVALID,
    BRANCH_3C_C_OPT_EXCL_FAIL,
    BRANCH_3C_C_PERSISTENT_DEFERRED,
    OBSERVATION_PROBE_COMPLETE_MODES,
    OBSERVATION_PROBE_MODE_ALLOC_GUARD,
    OPTIMIZER_CREDIT_STATE_3C_C1_PARITY_FIXTURE_DESCRIPTOR_SHA256,
    OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS,
    OptimizerCreditStateFailClosedReceipt,
    build_optimizer_credit_state_fail_closed_receipt,
    validate_optimizer_credit_state_fail_closed_receipt,
)

OPTIMIZER_CREDIT_STATE_NO_HIDDEN_FP_AUDIT_SCHEMA_VERSION = (
    "hrm_text_158_optimizer_credit_state_no_hidden_fp_audit/v1"
)

DENSE_SURFACE_WEIGHTED_GRAD = "weighted_grad"
DENSE_SURFACE_CREDIT = "credit"
DENSE_SURFACE_DENSE_RANK_VOTES = "dense_rank_votes_before_sparse_event_extraction"

AUDIT_OPT_EXCL = "AUDIT-OPT-EXCL"
AUDIT_NO_DENSE_WG = "AUDIT-NO-DENSE-WG"
AUDIT_NO_DENSE_CREDIT = "AUDIT-NO-DENSE-CREDIT"
AUDIT_NO_DENSE_VOTES = "AUDIT-NO-DENSE-VOTES"
AUDIT_CAPTURE_LAUNDER = "AUDIT-CAPTURE-LAUNDER"
AUDIT_PERSISTENT_STATE = "AUDIT-PERSISTENT-STATE"

NO_HIDDEN_FP_AUDIT_NON_CLAIMS = (
    "3C-C1 audit is CPU/reference scaffold only; no GPU runtime receipt",
    "audit_observation_complete must be earned by a real probe mode",
    "optimizer_state_eligible_exclusion_proven requires BR-3C-C-AUDIT-PASS-CPU linkage",
    "real_native_integer_attribution_present and real_native_integer_credit_ranking_present stay false on CPU",
)


@dataclass(frozen=True)
class NoHiddenFpAuditResult:
    audit_id: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class OptimizerCreditStateNoHiddenFpAuditReceipt:
    schema_version: str
    target_name: str
    branch_id: str
    audit_results: tuple[NoHiddenFpAuditResult, ...]
    dense_fp_intermediate_tensors_observed: tuple[str, ...]
    audit_observation_complete: bool
    observation_probe_mode: str
    optimizer_state_eligible_exclusion_proven: bool
    fp_exception_laundering_detected: bool
    ready_to_flip: bool
    real_native_integer_attribution_present: bool
    real_native_integer_credit_ranking_present: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_name": self.target_name,
            "branch_id": self.branch_id,
            "audit_results": [item.to_dict() for item in self.audit_results],
            "dense_fp_intermediate_tensors_observed": list(
                self.dense_fp_intermediate_tensors_observed
            ),
            "audit_observation_complete": self.audit_observation_complete,
            "observation_probe_mode": self.observation_probe_mode,
            "optimizer_state_eligible_exclusion_proven": (
                self.optimizer_state_eligible_exclusion_proven
            ),
            "fp_exception_laundering_detected": self.fp_exception_laundering_detected,
            "ready_to_flip": self.ready_to_flip,
            "real_native_integer_attribution_present": (
                self.real_native_integer_attribution_present
            ),
            "real_native_integer_credit_ranking_present": (
                self.real_native_integer_credit_ranking_present
            ),
            "non_claims": list(self.non_claims),
        }


def compute_optimizer_credit_state_no_hidden_fp_audit_receipt_sha256(
    audit: OptimizerCreditStateNoHiddenFpAuditReceipt,
) -> str:
    payload = json.dumps(audit.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_observed_surfaces(
    observed_dense_surfaces: Sequence[str],
) -> tuple[str, ...]:
    return tuple(sorted({str(surface) for surface in observed_dense_surfaces}))


def _run_required_audits(
    *,
    optimizer_checks: Mapping[str, Any],
    observed_dense_surfaces: Sequence[str],
    credit_capture_observed: bool,
    native_attribution_claimed: bool,
) -> tuple[NoHiddenFpAuditResult, ...]:
    observed = set(_normalize_observed_surfaces(observed_dense_surfaces))
    overlap = int(optimizer_checks.get("eligible_params_in_optimizer", -1))
    state_entries = int(optimizer_checks.get("eligible_optimizer_state_entries", -1))
    opt_excl_pass = overlap == 0 and state_entries == 0
    results = [
        NoHiddenFpAuditResult(
            audit_id=AUDIT_OPT_EXCL,
            passed=opt_excl_pass,
            detail=(
                f"eligible_params_in_optimizer={overlap}, "
                f"eligible_optimizer_state_entries={state_entries}"
            ),
        ),
        NoHiddenFpAuditResult(
            audit_id=AUDIT_NO_DENSE_WG,
            passed=DENSE_SURFACE_WEIGHTED_GRAD not in observed,
            detail=f"observed={sorted(observed)}",
        ),
        NoHiddenFpAuditResult(
            audit_id=AUDIT_NO_DENSE_CREDIT,
            passed=DENSE_SURFACE_CREDIT not in observed,
            detail=f"observed={sorted(observed)}",
        ),
        NoHiddenFpAuditResult(
            audit_id=AUDIT_NO_DENSE_VOTES,
            passed=DENSE_SURFACE_DENSE_RANK_VOTES not in observed,
            detail=f"observed={sorted(observed)}",
        ),
        NoHiddenFpAuditResult(
            audit_id=AUDIT_CAPTURE_LAUNDER,
            passed=not (credit_capture_observed and native_attribution_claimed),
            detail=(
                f"credit_capture_observed={credit_capture_observed}, "
                f"native_attribution_claimed={native_attribution_claimed}"
            ),
        ),
        NoHiddenFpAuditResult(
            audit_id=AUDIT_PERSISTENT_STATE,
            passed=True,
            detail="deferred informational only",
        ),
    ]
    return tuple(results)


def _classify_audit_branch(
    *,
    audit_results: Sequence[NoHiddenFpAuditResult],
    observed_dense_surfaces: Sequence[str],
    audit_observation_complete: bool,
    observation_probe_mode: str,
) -> str:
    if not audit_observation_complete:
        return BRANCH_3C_C_MEASUREMENT_INVALID
    if observation_probe_mode not in OBSERVATION_PROBE_COMPLETE_MODES:
        return BRANCH_3C_C_MEASUREMENT_INVALID

    by_id = {item.audit_id: item for item in audit_results}
    if not by_id.get(AUDIT_CAPTURE_LAUNDER, NoHiddenFpAuditResult("", False, "")).passed:
        return BRANCH_3C_C_CAPTURE_LAUNDER
    for audit_id in (AUDIT_NO_DENSE_WG, AUDIT_NO_DENSE_CREDIT, AUDIT_NO_DENSE_VOTES):
        if not by_id.get(audit_id, NoHiddenFpAuditResult("", False, "")).passed:
            return BRANCH_3C_C_DENSE_LEAK
    if not by_id.get(AUDIT_OPT_EXCL, NoHiddenFpAuditResult("", False, "")).passed:
        return BRANCH_3C_C_OPT_EXCL_FAIL
    if observed_dense_surfaces:
        return BRANCH_3C_C_DENSE_LEAK
    return BRANCH_3C_C_AUDIT_PASS_CPU


def run_optimizer_credit_state_no_hidden_fp_audit(
    *,
    optimizer_checks: Mapping[str, Any],
    observed_dense_surfaces: Sequence[str] = (),
    observation_probe_mode: str,
    audit_observation_complete: bool,
    credit_capture_observed: bool = False,
    native_attribution_claimed: bool = False,
    target_name: str = "step3c_optimizer_credit_state_no_hidden_fp_audit",
) -> OptimizerCreditStateNoHiddenFpAuditReceipt:
    observed = _normalize_observed_surfaces(observed_dense_surfaces)
    audit_results = _run_required_audits(
        optimizer_checks=optimizer_checks,
        observed_dense_surfaces=observed,
        credit_capture_observed=credit_capture_observed,
        native_attribution_claimed=native_attribution_claimed,
    )
    branch_id = _classify_audit_branch(
        audit_results=audit_results,
        observed_dense_surfaces=observed,
        audit_observation_complete=audit_observation_complete,
        observation_probe_mode=observation_probe_mode,
    )
    laundering = branch_id == BRANCH_3C_C_CAPTURE_LAUNDER
    exclusion_proven = branch_id == BRANCH_3C_C_AUDIT_PASS_CPU
    return OptimizerCreditStateNoHiddenFpAuditReceipt(
        schema_version=OPTIMIZER_CREDIT_STATE_NO_HIDDEN_FP_AUDIT_SCHEMA_VERSION,
        target_name=target_name,
        branch_id=branch_id,
        audit_results=audit_results,
        dense_fp_intermediate_tensors_observed=observed,
        audit_observation_complete=bool(audit_observation_complete),
        observation_probe_mode=str(observation_probe_mode),
        optimizer_state_eligible_exclusion_proven=exclusion_proven,
        fp_exception_laundering_detected=laundering,
        ready_to_flip=False,
        real_native_integer_attribution_present=False,
        real_native_integer_credit_ranking_present=False,
        non_claims=NO_HIDDEN_FP_AUDIT_NON_CLAIMS,
    )


@contextmanager
def _dense_surface_alloc_guard(
    weight_shape: tuple[int, int],
) -> Iterator[list[str]]:
    observed: list[str] = []
    original_zeros = torch.zeros
    original_empty = torch.empty

    def _shape_from_args(size: tuple[Any, ...] | list[Any]) -> tuple[int, ...]:
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            return tuple(int(dim) for dim in size[0])
        return tuple(int(dim) for dim in size)

    def _maybe_record(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
        if len(shape) != 2 or shape != weight_shape:
            return
        if dtype == torch.float32:
            if DENSE_SURFACE_WEIGHTED_GRAD not in observed:
                observed.append(DENSE_SURFACE_WEIGHTED_GRAD)
            if DENSE_SURFACE_CREDIT not in observed:
                observed.append(DENSE_SURFACE_CREDIT)
        if dtype == torch.int16:
            if DENSE_SURFACE_DENSE_RANK_VOTES not in observed:
                observed.append(DENSE_SURFACE_DENSE_RANK_VOTES)

    def guarded_zeros(*size, **kwargs):
        dtype = kwargs.get("dtype", torch.get_default_dtype())
        _maybe_record(dtype, _shape_from_args(size))
        return original_zeros(*size, **kwargs)

    def guarded_empty(*size, **kwargs):
        dtype = kwargs.get("dtype", torch.get_default_dtype())
        _maybe_record(dtype, _shape_from_args(size))
        return original_empty(*size, **kwargs)

    with mock.patch("torch.zeros", side_effect=guarded_zeros), mock.patch(
        "torch.empty",
        side_effect=guarded_empty,
    ):
        yield observed


def run_integer_path_dense_surface_observation_with_alloc_guard(
    *,
    captures: Mapping[str, Any],
    weight_shape: tuple[int, int],
    q_flat: torch.Tensor,
) -> tuple[tuple[str, ...], str]:
    """Run 3C-A + 3C-B integer path under alloc-guard instrumentation."""
    with _dense_surface_alloc_guard(weight_shape) as observed:
        events = integer_marginal_attribution_from_captures(
            captures["inputs"],
            captures["grad_outputs"],
            weight_shape=weight_shape,
        )
        move_indices, projected_moves = projected_moves_from_integer_attribution(events, q_flat)
        sparse_rank_bucketed_vote_events_from_integer_credit(
            events.attribution_q31,
            projected_moves,
            move_indices,
            spec=default_dry_run_rank_vote_spec(),
            credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
        )
    return tuple(sorted(set(observed))), OBSERVATION_PROBE_MODE_ALLOC_GUARD


def build_optimizer_credit_state_receipt_from_audit(
    audit: OptimizerCreditStateNoHiddenFpAuditReceipt,
    *,
    integer_attribution_law_id: str = INTEGER_MARGINAL_ATTRIBUTION_LAW_ID,
    integer_credit_ranking_law_id: str = CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    parity_fixture_sha256: str = OPTIMIZER_CREDIT_STATE_3C_C1_PARITY_FIXTURE_DESCRIPTOR_SHA256,
) -> OptimizerCreditStateFailClosedReceipt:
    audit_sha = compute_optimizer_credit_state_no_hidden_fp_audit_receipt_sha256(audit)
    exclusion_proven = audit.branch_id == BRANCH_3C_C_AUDIT_PASS_CPU
    receipt = build_optimizer_credit_state_fail_closed_receipt(
        optimizer_state_eligible_exclusion_proven=exclusion_proven,
        no_hidden_bf16_fp_optimizer_state_proven=exclusion_proven,
        no_hidden_fp_audit_branch_id=audit.branch_id,
        no_hidden_fp_audit_receipt_sha256=audit_sha,
        dense_fp_intermediate_tensors_observed=audit.dense_fp_intermediate_tensors_observed,
        integer_attribution_law_id=integer_attribution_law_id,
        integer_credit_ranking_law_id=integer_credit_ranking_law_id,
        parity_fixture_sha256=parity_fixture_sha256,
        cpu_reference_path_only=True,
        audit_observation_complete=audit.audit_observation_complete,
        observation_probe_mode=audit.observation_probe_mode,
    )
    validate_optimizer_credit_state_fail_closed_receipt(receipt)
    return receipt
