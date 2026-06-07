"""Candidate-only persistent-core sub-2 absence receipt.

This module proves a CPU/reference candidate shape. It deliberately does not
convert the live trainer or vote-update authority, and it cannot authorize live
readiness row flips.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.sub2_native_birth_sidecar_runtime import (
    HybridSidecarPersistentStateReport,
    hybrid_sidecar_persistent_state_report,
    make_applied_crossing_direction_residual_persistent_state,
)


PERSISTENT_CORE_SUB2_ABSENCE_SCHEMA_VERSION = (
    "hrm_text_158_persistent_core_sub2_absence/v0.candidate_only"
)
PERSISTENT_CORE_SUB2_ABSENCE_TARGET_NAME = (
    "step2a_candidate_persistent_core_sub2_absence"
)
PERSISTENT_CORE_SUB2_ABSENCE_DENSE_CREDIT_CLASS = "transient_fp_debt"
PERSISTENT_CORE_SUB2_ABSENCE_LIVE_BLOCKED_SURFACES = (
    "persistent_qacc_authority",
    "dense_int16_persistent_accumulator_absence",
    "q_sidecar_vote_carrier",
)
PERSISTENT_CORE_SUB2_ABSENCE_NON_CLAIMS = (
    "candidate receipt only; production_authority_claim_authorized=false",
    "live_runtime_authority_converted=false; trainer_entrypoint_uses_candidate=false",
    "readiness_row_flip_authorized=false; live readiness rows remain debt/blockers until Step 2B",
    "not learning, acquisition, retention, throughput, GPU residency, or a training launch",
)


@dataclass(frozen=True)
class PersistentCoreSub2AbsenceCandidateReceipt:
    schema_version: str
    target_name: str
    pass_receipt: bool
    candidate_persistent_core_absence_proven: bool
    production_authority_claim_authorized: bool
    live_runtime_authority_converted: bool
    trainer_entrypoint_uses_candidate: bool
    readiness_row_flip_authorized: bool
    readiness_row_flip_authorized_surface_names: tuple[str, ...]
    dense_credit_classification: str
    live_rows_remain_debt_or_blocker: tuple[str, ...]
    qacc_authority_inspectable: bool
    q_sidecar_vote_carrier_candidate_sub2: bool
    no_dense_int16_persistent_accumulator_authority_candidate: bool
    optimizer_fp_master_excluded: bool
    physical_persistent_bits_per_weight: float
    effective_persistent_bits_per_weight: float
    target_bits_per_weight: float
    sidecar_report: HybridSidecarPersistentStateReport
    optimizer_identity_proof: dict[str, Any]
    proof_anchors: tuple[str, ...]
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sidecar_report"] = self.sidecar_report.to_dict()
        payload["readiness_row_flip_authorized_surface_names"] = list(
            self.readiness_row_flip_authorized_surface_names
        )
        payload["live_rows_remain_debt_or_blocker"] = list(
            self.live_rows_remain_debt_or_blocker
        )
        payload["proof_anchors"] = list(self.proof_anchors)
        payload["non_claims"] = list(self.non_claims)
        return payload


def _default_optimizer_identity_proof() -> dict[str, Any]:
    return {
        "schema": "hrm_text_158_c2p0_fp_master_identity_snapshot/v0",
        "eligible_master_identity_pass": True,
        "optimizer_step_called": False,
        "optimizer_checks": {
            "eligible_param_count": 0,
            "optimizer_param_count": 0,
            "eligible_params_in_optimizer": 0,
            "eligible_optimizer_state_entries": 0,
            "optimizer_state_entries_total": 0,
            "pass": True,
        },
        "pass": True,
    }


def _optimizer_fp_master_excluded(proof: Mapping[str, Any]) -> bool:
    checks = dict(proof.get("optimizer_checks") or {})
    return bool(
        proof.get("pass")
        and proof.get("eligible_master_identity_pass")
        and int(checks.get("eligible_params_in_optimizer", 0)) == 0
        and int(checks.get("eligible_optimizer_state_entries", 0)) == 0
        and bool(checks.get("pass", True))
    )


def _sample_sidecar_report() -> HybridSidecarPersistentStateReport:
    q_levels = torch.zeros((128, 128), dtype=torch.int8)
    state = make_applied_crossing_direction_residual_persistent_state(
        "candidate.core.weight",
        q_levels,
        1.0,
        applied_indices=(3, 197),
        applied_directions=(1, -1),
        residual_values=(1, -2),
    )
    return hybrid_sidecar_persistent_state_report({state.state_key: state})


def build_persistent_core_sub2_absence_candidate_receipt(
    *,
    sidecar_report: HybridSidecarPersistentStateReport | None = None,
    optimizer_identity_proof: Mapping[str, Any] | None = None,
    production_authority_claim_authorized: bool = False,
    live_runtime_authority_converted: bool = False,
    trainer_entrypoint_uses_candidate: bool = False,
    readiness_row_flip_authorized: bool = False,
    readiness_row_flip_authorized_surface_names: Sequence[str] = (),
    dense_credit_classification: str = PERSISTENT_CORE_SUB2_ABSENCE_DENSE_CREDIT_CLASS,
) -> PersistentCoreSub2AbsenceCandidateReceipt:
    """Build the Step-2A receipt without touching live runtime authority."""

    report = sidecar_report if sidecar_report is not None else _sample_sidecar_report()
    proof = dict(optimizer_identity_proof or _default_optimizer_identity_proof())
    ledger = dict(report.movement_overlay.persistent_sidecar_ledger)
    inclusive_bpw = float(ledger.get("inclusive_bits_per_weight", 0.0))
    optimizer_excluded = _optimizer_fp_master_excluded(proof)
    no_dense_shadow = bool(
        not report.persistent_dense_shadow_present
        and int(report.persistent_dense_shadow_bytes) == 0
    )
    candidate_pass = bool(
        report.pass_report
        and report.budget_guard.pass_guard
        and no_dense_shadow
        and inclusive_bpw < 2.0
        and optimizer_excluded
        and not production_authority_claim_authorized
        and not live_runtime_authority_converted
        and not trainer_entrypoint_uses_candidate
        and not readiness_row_flip_authorized
        and not tuple(readiness_row_flip_authorized_surface_names)
        and str(dense_credit_classification) == PERSISTENT_CORE_SUB2_ABSENCE_DENSE_CREDIT_CLASS
    )
    receipt = PersistentCoreSub2AbsenceCandidateReceipt(
        schema_version=PERSISTENT_CORE_SUB2_ABSENCE_SCHEMA_VERSION,
        target_name=PERSISTENT_CORE_SUB2_ABSENCE_TARGET_NAME,
        pass_receipt=candidate_pass,
        candidate_persistent_core_absence_proven=candidate_pass,
        production_authority_claim_authorized=bool(production_authority_claim_authorized),
        live_runtime_authority_converted=bool(live_runtime_authority_converted),
        trainer_entrypoint_uses_candidate=bool(trainer_entrypoint_uses_candidate),
        readiness_row_flip_authorized=bool(readiness_row_flip_authorized),
        readiness_row_flip_authorized_surface_names=tuple(
            str(name) for name in readiness_row_flip_authorized_surface_names
        ),
        dense_credit_classification=str(dense_credit_classification),
        live_rows_remain_debt_or_blocker=PERSISTENT_CORE_SUB2_ABSENCE_LIVE_BLOCKED_SURFACES,
        qacc_authority_inspectable=True,
        q_sidecar_vote_carrier_candidate_sub2=bool(report.pass_report),
        no_dense_int16_persistent_accumulator_authority_candidate=no_dense_shadow,
        optimizer_fp_master_excluded=optimizer_excluded,
        physical_persistent_bits_per_weight=inclusive_bpw,
        effective_persistent_bits_per_weight=inclusive_bpw,
        target_bits_per_weight=2.0,
        sidecar_report=report,
        optimizer_identity_proof=proof,
        proof_anchors=(
            "sub2_native_birth_sidecar_runtime.py:304",
            "sub2_native_birth_scaffold.py:680",
            "bounded_delta_learner.py:729",
            "vote_update.py:203",
        ),
        non_claims=PERSISTENT_CORE_SUB2_ABSENCE_NON_CLAIMS,
    )
    validate_persistent_core_sub2_absence_candidate_receipt(receipt)
    return receipt


def validate_persistent_core_sub2_absence_candidate_receipt(
    receipt: PersistentCoreSub2AbsenceCandidateReceipt,
) -> None:
    if receipt.schema_version != PERSISTENT_CORE_SUB2_ABSENCE_SCHEMA_VERSION:
        raise ValueError("persistent-core absence schema version mismatch")
    if receipt.target_name != PERSISTENT_CORE_SUB2_ABSENCE_TARGET_NAME:
        raise ValueError("persistent-core absence target name mismatch")
    if receipt.production_authority_claim_authorized:
        raise ValueError("Step 2A cannot authorize production authority claims")
    if receipt.live_runtime_authority_converted:
        raise ValueError("Step 2A cannot claim live runtime authority conversion")
    if receipt.trainer_entrypoint_uses_candidate:
        raise ValueError("Step 2A cannot claim trainer entrypoint use")
    if receipt.readiness_row_flip_authorized:
        raise ValueError("Step 2A cannot authorize live readiness row flips")
    if receipt.readiness_row_flip_authorized_surface_names:
        raise ValueError("Step 2A readiness row flip surface list must be empty")
    if (
        receipt.dense_credit_classification
        != PERSISTENT_CORE_SUB2_ABSENCE_DENSE_CREDIT_CLASS
    ):
        raise ValueError("dense credit must remain classified as transient_fp_debt")
    if (
        tuple(receipt.live_rows_remain_debt_or_blocker)
        != PERSISTENT_CORE_SUB2_ABSENCE_LIVE_BLOCKED_SURFACES
    ):
        raise ValueError("live readiness rows must remain debt/blockers until Step 2B")
    report = receipt.sidecar_report
    if not report.pass_report:
        raise ValueError("sidecar candidate report must pass before absence receipt")
    if not report.budget_guard.pass_guard:
        raise ValueError("sidecar candidate budget guard must pass")
    if report.persistent_dense_shadow_present:
        raise ValueError("candidate persistent state cannot contain dense shadow authority")
    if int(report.persistent_dense_shadow_bytes) != 0:
        raise ValueError("candidate persistent dense shadow bytes must be zero")
    ledger = dict(report.movement_overlay.persistent_sidecar_ledger)
    inclusive_bpw = float(ledger.get("inclusive_bits_per_weight", 0.0))
    if inclusive_bpw >= float(receipt.target_bits_per_weight):
        raise ValueError("candidate inclusive persistent bits/weight must stay < 2")
    if abs(float(receipt.physical_persistent_bits_per_weight) - inclusive_bpw) > 1e-12:
        raise ValueError("physical bits/weight must mirror the sidecar inclusive ledger")
    if abs(float(receipt.effective_persistent_bits_per_weight) - inclusive_bpw) > 1e-12:
        raise ValueError("effective bits/weight must mirror the sidecar inclusive ledger")
    authority_rows = tuple(report.movement_overlay.persistent_authority_row_names)
    if authority_rows != ("q_storage", "frozen_scales_fp32_metadata", "accumulator_sidecar"):
        raise ValueError("candidate authority rows must be q + scale + accumulator_sidecar")
    if any("dense_int16" in row or "int16" in row for row in authority_rows):
        raise ValueError("candidate authority rows cannot include dense int16 accumulator")
    if not receipt.optimizer_fp_master_excluded:
        raise ValueError("eligible FP masters must be excluded from optimizer state")
    if not _optimizer_fp_master_excluded(receipt.optimizer_identity_proof):
        raise ValueError("optimizer identity proof does not exclude eligible masters")
    if bool(receipt.pass_receipt) != bool(
        receipt.candidate_persistent_core_absence_proven
        and receipt.qacc_authority_inspectable
        and receipt.q_sidecar_vote_carrier_candidate_sub2
        and receipt.no_dense_int16_persistent_accumulator_authority_candidate
        and receipt.optimizer_fp_master_excluded
    ):
        raise ValueError("pass_receipt must be computed from explicit candidate-only gates")
    serialized_non_claims = " ".join(receipt.non_claims)
    for required in (
        "production_authority_claim_authorized=false",
        "live_runtime_authority_converted=false",
        "trainer_entrypoint_uses_candidate=false",
        "readiness_row_flip_authorized=false",
    ):
        if required not in serialized_non_claims:
            raise ValueError(f"candidate receipt non-claims must include {required}")


def receipt_with_replaced_sidecar_report(
    receipt: PersistentCoreSub2AbsenceCandidateReceipt,
    report: HybridSidecarPersistentStateReport,
) -> PersistentCoreSub2AbsenceCandidateReceipt:
    return replace(receipt, sidecar_report=report)


__all__ = [
    "PERSISTENT_CORE_SUB2_ABSENCE_DENSE_CREDIT_CLASS",
    "PERSISTENT_CORE_SUB2_ABSENCE_LIVE_BLOCKED_SURFACES",
    "PERSISTENT_CORE_SUB2_ABSENCE_NON_CLAIMS",
    "PERSISTENT_CORE_SUB2_ABSENCE_SCHEMA_VERSION",
    "PERSISTENT_CORE_SUB2_ABSENCE_TARGET_NAME",
    "PersistentCoreSub2AbsenceCandidateReceipt",
    "build_persistent_core_sub2_absence_candidate_receipt",
    "receipt_with_replaced_sidecar_report",
    "validate_persistent_core_sub2_absence_candidate_receipt",
]
