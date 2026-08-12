"""Fail-closed full-sub2 runtime readiness ledger for HRM-Text-1.58.

This module is a read-only gate scaffold. It classifies required runtime
surfaces before mechanism science may launch; it does not implement a trainer,
kernel, checkpoint path, or acquisition proof.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.activation_relief import (
    ACTIVATION_RELIEF_SCHEMA_VERSION,
    ACTIVATION_RESIDUALS_BLOCKED_REASON,
    ACTIVATION_RESIDUALS_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION,
    BACKWARD_RECOMPUTE_RECEIPT_SCHEMA_VERSION,
)
from calm.hrm_text_158.native_full_stack.attention_kv_buffers import (
    ATTENTION_KV_BLOCKED_REASON,
    ATTENTION_KV_BUFFER_SCHEMA_VERSION,
    ATTENTION_KV_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION,
)
from calm.hrm_text_158.native_full_stack.fp_exceptions import FP_EXCEPTION_REGISTRY
from calm.hrm_text_158.native_full_stack.full_loop_receipt import (
    FULL_LOOP_RECEIPT_SCHEMA_VERSION,
)
from calm.hrm_text_158.native_full_stack.native_kernelized_hot_path import (
    NATIVE_KERNELIZED_HOT_PATH_BLOCKED_REASON,
    NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (
    OPTIMIZER_CREDIT_STATE_BLOCKED_REASON,
    OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    PERSISTENT_STATE_BUDGET_SCHEMA_VERSION,
    PHYSICAL_SUB2_NOT_ACHIEVED_STATEMENT,
)
from calm.hrm_text_158.native_full_stack.recurrent_state_buffers import (
    RECURRENT_STATE_BUFFER_SCHEMA_VERSION,
)
from calm.hrm_text_158.native_full_stack.sub2_native_birth_scaffold import (
    STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_SCHEMA_VERSION,
)
from calm.hrm_text_158.native_full_stack.sub2_native_birth_sidecar_runtime import (
    SUB2_HYBRID_SIDECAR_RUNTIME_SCHEMA_VERSION,
)


FULL_SUB2_RUNTIME_READY_FOR_SCIENCE_SCHEMA_VERSION = (
    "hrm_text_158_full_sub2_runtime_ready_for_science/v0.fail_closed"
)
FULL_SUB2_RUNTIME_READY_FOR_SCIENCE_TARGET_NAME = (
    "full_sub2_runtime_ready_for_science"
)

RUNTIME_CLASS_SUB2 = "sub2"
RUNTIME_CLASS_EXPLICIT_EXCEPTION = "explicit_exception"
RUNTIME_CLASS_TRANSIENT_FP_DEBT = "transient_fp_debt"
RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC = "pre_full_stack_diagnostic"
RUNTIME_CLASS_MISSING = "missing"

FULL_SUB2_RUNTIME_CLASSIFICATIONS = (
    RUNTIME_CLASS_SUB2,
    RUNTIME_CLASS_EXPLICIT_EXCEPTION,
    RUNTIME_CLASS_TRANSIENT_FP_DEBT,
    RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
    RUNTIME_CLASS_MISSING,
)

SURFACE_PERSISTENT_QACC_AUTHORITY = "persistent_qacc_authority"
SURFACE_DENSE_INT16_PERSISTENT_ACCUMULATOR_ABSENCE = (
    "dense_int16_persistent_accumulator_absence"
)
SURFACE_Q_SIDECAR_VOTE_CARRIER = "q_sidecar_vote_carrier"
SURFACE_ACTIVATIONS_RESIDUALS = "activations_residuals"
SURFACE_ATTENTION_KV_ATTENTION_BUFFERS = "attention_kv_attention_buffers"
SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS = (
    "backward_saved_tensors_transients"
)
SURFACE_OPTIMIZER_CREDIT_STATE = "optimizer_credit_state"
SURFACE_NATIVE_KERNELIZED_HOT_PATH = "native_kernelized_hot_path"
SURFACE_FP_EXCEPTIONS_LEDGER = "fp_exceptions_ledger"

FULL_SUB2_RUNTIME_REQUIRED_SURFACES = (
    SURFACE_PERSISTENT_QACC_AUTHORITY,
    SURFACE_DENSE_INT16_PERSISTENT_ACCUMULATOR_ABSENCE,
    SURFACE_Q_SIDECAR_VOTE_CARRIER,
    SURFACE_ACTIVATIONS_RESIDUALS,
    SURFACE_ATTENTION_KV_ATTENTION_BUFFERS,
    SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS,
    SURFACE_OPTIMIZER_CREDIT_STATE,
    SURFACE_NATIVE_KERNELIZED_HOT_PATH,
    SURFACE_FP_EXCEPTIONS_LEDGER,
)

FULL_SUB2_RUNTIME_SOURCE_SEAMS = {
    "strict_sub2_scaffold": STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_SCHEMA_VERSION,
    "persistent_state_budget": PERSISTENT_STATE_BUDGET_SCHEMA_VERSION,
    "activation_relief": ACTIVATION_RELIEF_SCHEMA_VERSION,
    "activation_residuals_fail_closed": (
        ACTIVATION_RESIDUALS_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION
    ),
    "backward_recompute": BACKWARD_RECOMPUTE_RECEIPT_SCHEMA_VERSION,
    "attention_kv_buffers": ATTENTION_KV_BUFFER_SCHEMA_VERSION,
    "attention_kv_fail_closed": ATTENTION_KV_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION,
    "optimizer_credit_state": (
        OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION
    ),
    "native_kernelized_hot_path": (
        NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION
    ),
    "recurrent_state_buffers": RECURRENT_STATE_BUFFER_SCHEMA_VERSION,
    "full_loop_receipt": FULL_LOOP_RECEIPT_SCHEMA_VERSION,
    "sidecar_runtime": SUB2_HYBRID_SIDECAR_RUNTIME_SCHEMA_VERSION,
    "fp_exception_count": len(FP_EXCEPTION_REGISTRY),
}

FULL_SUB2_RUNTIME_NON_CLAIMS = (
    "readiness is not learning, acquisition, retention, or throughput",
    "readiness scaffold does not launch training, write checkpoints, or mutate .pt artifacts",
    "explicit exceptions and transient FP debt are visible debt, not sub2 credit",
)

FIXTURE_CURRENT_REPO = "current_repo_scaffold"
FIXTURE_GATED_SUB2_CHECKPOINT_PATH = "gated_sub2_checkpoint_path"
FIXTURE_GATED_SUB2_CHECKPOINT_PATH_BACKWARD_RECOMPUTE = (
    "gated_sub2_checkpoint_path_backward_recompute"
)
FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ACTIVATION_RESIDUALS_BLOCKED = (
    "gated_sub2_checkpoint_path_activation_residuals_blocked"
)
FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ATTENTION_KV_BLOCKED = (
    "gated_sub2_checkpoint_path_attention_kv_blocked"
)
FIXTURE_GATED_SUB2_CHECKPOINT_PATH_OPTIMIZER_CREDIT_STATE_BLOCKED = (
    "gated_sub2_checkpoint_path_optimizer_credit_state_blocked"
)
FIXTURE_GATED_SUB2_CHECKPOINT_PATH_NATIVE_KERNELIZED_HOT_PATH_BLOCKED = (
    "gated_sub2_checkpoint_path_native_kernelized_hot_path_blocked"
)
FIXTURE_MAIN_READY = "main_ready"
FIXTURE_MISSING_ACTIVATIONS = "missing_activations_residuals"
FIXTURE_MISSING_ATTENTION = "missing_attention_kv_attention_buffers"
FIXTURE_MISSING_BACKWARD = "missing_backward_saved_tensors_transients"
FIXTURE_PRE_FULL_STACK_DIAGNOSTIC = "pre_full_stack_diagnostic"
FIXTURE_TRANSIENT_FP_DEBT = "transient_fp_debt"
FIXTURE_STEP2A_CANDIDATE_PERSISTENT_CORE_ABSENCE = (
    "step2a_candidate_persistent_core_absence"
)
FIXTURE_LIVE_P1_AUTHORITY_CONVERSION = "live_p1_authority_conversion"
FIXTURE_LIVE_R1_BACKWARD_LAUNCH = "live_r1_backward_launch"

FULL_SUB2_RUNTIME_FIXTURE_NAMES = (
    FIXTURE_CURRENT_REPO,
    FIXTURE_GATED_SUB2_CHECKPOINT_PATH,
    FIXTURE_GATED_SUB2_CHECKPOINT_PATH_BACKWARD_RECOMPUTE,
    FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ACTIVATION_RESIDUALS_BLOCKED,
    FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ATTENTION_KV_BLOCKED,
    FIXTURE_GATED_SUB2_CHECKPOINT_PATH_OPTIMIZER_CREDIT_STATE_BLOCKED,
    FIXTURE_GATED_SUB2_CHECKPOINT_PATH_NATIVE_KERNELIZED_HOT_PATH_BLOCKED,
    FIXTURE_MAIN_READY,
    FIXTURE_MISSING_ACTIVATIONS,
    FIXTURE_MISSING_ATTENTION,
    FIXTURE_MISSING_BACKWARD,
    FIXTURE_PRE_FULL_STACK_DIAGNOSTIC,
    FIXTURE_TRANSIENT_FP_DEBT,
    FIXTURE_STEP2A_CANDIDATE_PERSISTENT_CORE_ABSENCE,
    FIXTURE_LIVE_P1_AUTHORITY_CONVERSION,
    FIXTURE_LIVE_R1_BACKWARD_LAUNCH,
)

GATED_SUB2_CHECKPOINT_PATH_REASON = (
    "gated default-off sidecar checkpoint path only; default runtime not sub2"
)
GATED_LOSSLESS_RECOMPUTE_REASON = (
    "gated default-off lossless recompute path only; default runtime not sub2"
)


@dataclass(frozen=True)
class FullSub2RuntimeSurfaceReceipt:
    """One required runtime surface classification row."""

    surface_id: str
    classification: str
    reason: str
    source_anchor: str
    proof_artifact_or_test: str
    sunset_condition: str = ""
    diagnostic_exception_reason: str = ""
    why_cheaper_than_full_stack_first: str = ""
    diagnostic_exclusion_reason: str = ""

    def to_dict(self) -> dict[str, str]:
        return {field.name: str(getattr(self, field.name)) for field in fields(self)}


@dataclass(frozen=True)
class FullSub2RuntimeReadyForScienceReceipt:
    """Computed fail-closed readiness receipt."""

    schema_version: str
    target_name: str
    ready_for_main_science: bool
    ready_for_pre_full_stack_diagnostic: bool
    main_science_launch_blocked: bool
    sub2_surface_count: int
    counts_by_class: dict[str, int]
    surface_names_by_class: dict[str, tuple[str, ...]]
    explicit_exception_surface_names: tuple[str, ...]
    transient_fp_debt_surface_names: tuple[str, ...]
    pre_full_stack_diagnostic_surface_names: tuple[str, ...]
    missing_surface_names: tuple[str, ...]
    blocker_surface_names: tuple[str, ...]
    surfaces: tuple[FullSub2RuntimeSurfaceReceipt, ...]
    source_seams: dict[str, Any]
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_name": self.target_name,
            "ready_for_main_science": bool(self.ready_for_main_science),
            "ready_for_pre_full_stack_diagnostic": bool(
                self.ready_for_pre_full_stack_diagnostic
            ),
            "main_science_launch_blocked": bool(self.main_science_launch_blocked),
            "sub2_surface_count": int(self.sub2_surface_count),
            "counts_by_class": {
                key: int(value) for key, value in self.counts_by_class.items()
            },
            "surface_names_by_class": {
                key: list(value) for key, value in self.surface_names_by_class.items()
            },
            "explicit_exception_surface_names": list(
                self.explicit_exception_surface_names
            ),
            "transient_fp_debt_surface_names": list(
                self.transient_fp_debt_surface_names
            ),
            "pre_full_stack_diagnostic_surface_names": list(
                self.pre_full_stack_diagnostic_surface_names
            ),
            "missing_surface_names": list(self.missing_surface_names),
            "blocker_surface_names": list(self.blocker_surface_names),
            "surfaces": [surface.to_dict() for surface in self.surfaces],
            "source_seams": dict(self.source_seams),
            "non_claims": list(self.non_claims),
        }


def _as_surface_receipt(
    surface: FullSub2RuntimeSurfaceReceipt | Mapping[str, Any],
) -> FullSub2RuntimeSurfaceReceipt:
    if isinstance(surface, FullSub2RuntimeSurfaceReceipt):
        return surface
    allowed = {field.name for field in fields(FullSub2RuntimeSurfaceReceipt)}
    extra = set(surface) - allowed
    if extra:
        raise ValueError(f"unknown surface receipt fields: {sorted(extra)}")
    return FullSub2RuntimeSurfaceReceipt(
        surface_id=str(surface.get("surface_id", "")),
        classification=str(surface.get("classification", "")),
        reason=str(surface.get("reason", "")),
        source_anchor=str(surface.get("source_anchor", "")),
        proof_artifact_or_test=str(surface.get("proof_artifact_or_test", "")),
        sunset_condition=str(surface.get("sunset_condition", "")),
        diagnostic_exception_reason=str(
            surface.get("diagnostic_exception_reason", "")
        ),
        why_cheaper_than_full_stack_first=str(
            surface.get("why_cheaper_than_full_stack_first", "")
        ),
        diagnostic_exclusion_reason=str(
            surface.get("diagnostic_exclusion_reason", "")
        ),
    )


def _require_non_empty(value: str, *, field_name: str, surface_id: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{surface_id} requires non-empty {field_name}")


def _validate_surface_receipt(surface: FullSub2RuntimeSurfaceReceipt) -> None:
    if surface.surface_id not in FULL_SUB2_RUNTIME_REQUIRED_SURFACES:
        raise ValueError(f"unknown full-sub2 runtime surface_id: {surface.surface_id!r}")
    if surface.classification not in FULL_SUB2_RUNTIME_CLASSIFICATIONS:
        raise ValueError(
            f"unknown full-sub2 runtime classification for {surface.surface_id!r}: "
            f"{surface.classification!r}"
        )
    _require_non_empty(surface.reason, field_name="reason", surface_id=surface.surface_id)
    _require_non_empty(
        surface.source_anchor,
        field_name="source_anchor",
        surface_id=surface.surface_id,
    )
    _require_non_empty(
        surface.proof_artifact_or_test,
        field_name="proof_artifact_or_test",
        surface_id=surface.surface_id,
    )
    if surface.classification == RUNTIME_CLASS_EXPLICIT_EXCEPTION:
        _require_non_empty(
            surface.sunset_condition,
            field_name="sunset_condition",
            surface_id=surface.surface_id,
        )
    elif surface.sunset_condition:
        raise ValueError(
            f"{surface.surface_id} may set sunset_condition only for explicit_exception"
        )
    if surface.classification == RUNTIME_CLASS_SUB2:
        forbidden = (
            surface.diagnostic_exception_reason,
            surface.why_cheaper_than_full_stack_first,
            surface.diagnostic_exclusion_reason,
        )
        if any(str(value).strip() for value in forbidden):
            raise ValueError(f"{surface.surface_id} sub2 rows cannot carry diagnostic exception fields")


def _bucket_surfaces(
    surfaces: Sequence[FullSub2RuntimeSurfaceReceipt],
) -> dict[str, tuple[str, ...]]:
    buckets: dict[str, list[str]] = {
        classification: [] for classification in FULL_SUB2_RUNTIME_CLASSIFICATIONS
    }
    for surface in surfaces:
        buckets[surface.classification].append(surface.surface_id)
    return {key: tuple(value) for key, value in buckets.items()}


def _diagnostic_row_allowed(surface: FullSub2RuntimeSurfaceReceipt) -> bool:
    if surface.classification in {
        RUNTIME_CLASS_TRANSIENT_FP_DEBT,
        RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
    }:
        return bool(
            surface.diagnostic_exception_reason.strip()
            and surface.why_cheaper_than_full_stack_first.strip()
        )
    if surface.classification == RUNTIME_CLASS_MISSING:
        return bool(
            surface.diagnostic_exception_reason.strip()
            and surface.why_cheaper_than_full_stack_first.strip()
            and surface.diagnostic_exclusion_reason.strip()
        )
    return True


def build_full_sub2_runtime_ready_for_science(
    surfaces: Sequence[FullSub2RuntimeSurfaceReceipt | Mapping[str, Any]],
    *,
    source_seams: Mapping[str, Any] | None = None,
    non_claims: Sequence[str] = FULL_SUB2_RUNTIME_NON_CLAIMS,
) -> FullSub2RuntimeReadyForScienceReceipt:
    """Build a readiness receipt from explicit surface rows.

    The builder never trusts an input readiness flag; readiness is computed from
    the stable surface enum, classifications, and required proof fields.
    """

    normalized = tuple(_as_surface_receipt(surface) for surface in surfaces)
    seen: set[str] = set()
    duplicates: list[str] = []
    for surface in normalized:
        if surface.surface_id in seen:
            duplicates.append(surface.surface_id)
        seen.add(surface.surface_id)
    if duplicates:
        raise ValueError(f"duplicate full-sub2 runtime surface_id values: {sorted(duplicates)}")
    missing_ids = tuple(surface for surface in FULL_SUB2_RUNTIME_REQUIRED_SURFACES if surface not in seen)
    if missing_ids:
        raise ValueError(f"required full-sub2 runtime surfaces missing from receipt: {list(missing_ids)}")
    extra_ids = tuple(surface.surface_id for surface in normalized if surface.surface_id not in FULL_SUB2_RUNTIME_REQUIRED_SURFACES)
    if extra_ids:
        raise ValueError(f"unknown full-sub2 runtime surface_id values: {list(extra_ids)}")
    for surface in normalized:
        _validate_surface_receipt(surface)

    buckets = _bucket_surfaces(normalized)
    counts_by_class = {
        classification: len(buckets[classification])
        for classification in FULL_SUB2_RUNTIME_CLASSIFICATIONS
    }
    ready_for_main_science = bool(
        counts_by_class[RUNTIME_CLASS_MISSING] == 0
        and counts_by_class[RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC] == 0
        and counts_by_class[RUNTIME_CLASS_TRANSIENT_FP_DEBT] == 0
    )
    main_science_launch_blocked = not ready_for_main_science
    diagnostic_blocking_classes = {
        RUNTIME_CLASS_MISSING,
        RUNTIME_CLASS_TRANSIENT_FP_DEBT,
        RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
    }
    diagnostic_rows = tuple(
        surface
        for surface in normalized
        if surface.classification in diagnostic_blocking_classes
    )
    ready_for_pre_full_stack_diagnostic = bool(
        main_science_launch_blocked
        and diagnostic_rows
        and all(_diagnostic_row_allowed(surface) for surface in diagnostic_rows)
    )
    blockers = tuple(
        surface.surface_id
        for surface in normalized
        if surface.classification in diagnostic_blocking_classes
    )
    receipt = FullSub2RuntimeReadyForScienceReceipt(
        schema_version=FULL_SUB2_RUNTIME_READY_FOR_SCIENCE_SCHEMA_VERSION,
        target_name=FULL_SUB2_RUNTIME_READY_FOR_SCIENCE_TARGET_NAME,
        ready_for_main_science=ready_for_main_science,
        ready_for_pre_full_stack_diagnostic=ready_for_pre_full_stack_diagnostic,
        main_science_launch_blocked=main_science_launch_blocked,
        sub2_surface_count=counts_by_class[RUNTIME_CLASS_SUB2],
        counts_by_class=counts_by_class,
        surface_names_by_class=buckets,
        explicit_exception_surface_names=buckets[RUNTIME_CLASS_EXPLICIT_EXCEPTION],
        transient_fp_debt_surface_names=buckets[RUNTIME_CLASS_TRANSIENT_FP_DEBT],
        pre_full_stack_diagnostic_surface_names=buckets[
            RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC
        ],
        missing_surface_names=buckets[RUNTIME_CLASS_MISSING],
        blocker_surface_names=blockers,
        surfaces=normalized,
        source_seams=dict(FULL_SUB2_RUNTIME_SOURCE_SEAMS | dict(source_seams or {})),
        non_claims=tuple(str(non_claim) for non_claim in non_claims),
    )
    validate_full_sub2_runtime_ready_for_science_receipt(receipt)
    return receipt


def validate_full_sub2_runtime_ready_for_science_receipt(
    receipt: FullSub2RuntimeReadyForScienceReceipt,
) -> None:
    if receipt.schema_version != FULL_SUB2_RUNTIME_READY_FOR_SCIENCE_SCHEMA_VERSION:
        raise ValueError("full-sub2 readiness schema version mismatch")
    if receipt.target_name != FULL_SUB2_RUNTIME_READY_FOR_SCIENCE_TARGET_NAME:
        raise ValueError("full-sub2 readiness target name mismatch")
    rebuilt = build_full_sub2_runtime_ready_for_science_without_validation(
        receipt.surfaces,
        source_seams=receipt.source_seams,
        non_claims=receipt.non_claims,
    )
    comparable_fields = (
        "ready_for_main_science",
        "ready_for_pre_full_stack_diagnostic",
        "main_science_launch_blocked",
        "sub2_surface_count",
        "counts_by_class",
        "surface_names_by_class",
        "explicit_exception_surface_names",
        "transient_fp_debt_surface_names",
        "pre_full_stack_diagnostic_surface_names",
        "missing_surface_names",
        "blocker_surface_names",
    )
    for field_name in comparable_fields:
        if getattr(receipt, field_name) != getattr(rebuilt, field_name):
            raise ValueError(f"full-sub2 readiness field {field_name} is not computed from surface rows")
    if bool(receipt.ready_for_main_science) == bool(receipt.main_science_launch_blocked):
        raise ValueError("main_science_launch_blocked must be the inverse of ready_for_main_science")
    if receipt.ready_for_main_science and receipt.ready_for_pre_full_stack_diagnostic:
        raise ValueError("diagnostic readiness must not be true when main readiness is already true")
    for non_claim in receipt.non_claims:
        if "readiness is not learning" in non_claim:
            break
    else:
        raise ValueError("readiness receipt must include the learning/acquisition non-claim")


def build_full_sub2_runtime_ready_for_science_without_validation(
    surfaces: Sequence[FullSub2RuntimeSurfaceReceipt | Mapping[str, Any]],
    *,
    source_seams: Mapping[str, Any] | None = None,
    non_claims: Sequence[str] = FULL_SUB2_RUNTIME_NON_CLAIMS,
) -> FullSub2RuntimeReadyForScienceReceipt:
    """Internal rebuild path used by the validator to avoid recursion."""

    normalized = tuple(_as_surface_receipt(surface) for surface in surfaces)
    seen: set[str] = set()
    for surface in normalized:
        if surface.surface_id in seen:
            raise ValueError(f"duplicate full-sub2 runtime surface_id: {surface.surface_id!r}")
        seen.add(surface.surface_id)
        _validate_surface_receipt(surface)
    if tuple(surface.surface_id for surface in normalized) != FULL_SUB2_RUNTIME_REQUIRED_SURFACES:
        required = list(FULL_SUB2_RUNTIME_REQUIRED_SURFACES)
        actual = [surface.surface_id for surface in normalized]
        raise ValueError(f"full-sub2 readiness surfaces must match required enum order: required={required}, actual={actual}")
    buckets = _bucket_surfaces(normalized)
    counts_by_class = {
        classification: len(buckets[classification])
        for classification in FULL_SUB2_RUNTIME_CLASSIFICATIONS
    }
    ready_for_main_science = bool(
        counts_by_class[RUNTIME_CLASS_MISSING] == 0
        and counts_by_class[RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC] == 0
        and counts_by_class[RUNTIME_CLASS_TRANSIENT_FP_DEBT] == 0
    )
    blocking_classes = {
        RUNTIME_CLASS_MISSING,
        RUNTIME_CLASS_TRANSIENT_FP_DEBT,
        RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
    }
    diagnostic_rows = tuple(
        surface for surface in normalized if surface.classification in blocking_classes
    )
    return FullSub2RuntimeReadyForScienceReceipt(
        schema_version=FULL_SUB2_RUNTIME_READY_FOR_SCIENCE_SCHEMA_VERSION,
        target_name=FULL_SUB2_RUNTIME_READY_FOR_SCIENCE_TARGET_NAME,
        ready_for_main_science=ready_for_main_science,
        ready_for_pre_full_stack_diagnostic=bool(
            (not ready_for_main_science)
            and diagnostic_rows
            and all(_diagnostic_row_allowed(surface) for surface in diagnostic_rows)
        ),
        main_science_launch_blocked=not ready_for_main_science,
        sub2_surface_count=counts_by_class[RUNTIME_CLASS_SUB2],
        counts_by_class=counts_by_class,
        surface_names_by_class=buckets,
        explicit_exception_surface_names=buckets[RUNTIME_CLASS_EXPLICIT_EXCEPTION],
        transient_fp_debt_surface_names=buckets[RUNTIME_CLASS_TRANSIENT_FP_DEBT],
        pre_full_stack_diagnostic_surface_names=buckets[
            RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC
        ],
        missing_surface_names=buckets[RUNTIME_CLASS_MISSING],
        blocker_surface_names=tuple(
            surface.surface_id
            for surface in normalized
            if surface.classification in blocking_classes
        ),
        surfaces=normalized,
        source_seams=dict(FULL_SUB2_RUNTIME_SOURCE_SEAMS | dict(source_seams or {})),
        non_claims=tuple(str(non_claim) for non_claim in non_claims),
    )


def _base_sub2_surfaces() -> tuple[FullSub2RuntimeSurfaceReceipt, ...]:
    return tuple(
        FullSub2RuntimeSurfaceReceipt(
            surface_id=surface_id,
            classification=RUNTIME_CLASS_SUB2,
            reason="static fixture marks this required surface as proven sub2",
            source_anchor="calm/hrm_text_158/native_full_stack/full_sub2_runtime_readiness.py:fixture",
            proof_artifact_or_test="test_hrm_text_158_full_sub2_runtime_readiness.py",
        )
        for surface_id in FULL_SUB2_RUNTIME_REQUIRED_SURFACES
    )


def _with_surface(
    surfaces: Sequence[FullSub2RuntimeSurfaceReceipt],
    surface_id: str,
    *,
    classification: str,
    reason: str,
    source_anchor: str,
    proof_artifact_or_test: str,
    sunset_condition: str = "",
    diagnostic_exception_reason: str = "",
    why_cheaper_than_full_stack_first: str = "",
    diagnostic_exclusion_reason: str = "",
) -> tuple[FullSub2RuntimeSurfaceReceipt, ...]:
    updated = []
    for surface in surfaces:
        if surface.surface_id == surface_id:
            updated.append(
                replace(
                    surface,
                    classification=classification,
                    reason=reason,
                    source_anchor=source_anchor,
                    proof_artifact_or_test=proof_artifact_or_test,
                    sunset_condition=sunset_condition,
                    diagnostic_exception_reason=diagnostic_exception_reason,
                    why_cheaper_than_full_stack_first=why_cheaper_than_full_stack_first,
                    diagnostic_exclusion_reason=diagnostic_exclusion_reason,
                )
            )
        else:
            updated.append(surface)
    return tuple(updated)


def main_ready_fixture_surfaces() -> tuple[FullSub2RuntimeSurfaceReceipt, ...]:
    surfaces = _base_sub2_surfaces()
    return _with_surface(
        surfaces,
        SURFACE_FP_EXCEPTIONS_LEDGER,
        classification=RUNTIME_CLASS_EXPLICIT_EXCEPTION,
        reason="FP exception registry is explicit and bounded; it remains visible debt, not sub2 credit",
        source_anchor="calm/hrm_text_158/native_full_stack/fp_exceptions.py:19",
        proof_artifact_or_test="test_hrm_text_158_full_sub2_runtime_readiness.py::test_main_ready_fixture_allows_only_justified_explicit_exception",
        sunset_condition="replace each registered FP exception with sub2 or remove it from the runtime",
    )


def current_repo_scaffold_surfaces() -> tuple[FullSub2RuntimeSurfaceReceipt, ...]:
    surfaces = main_ready_fixture_surfaces()
    surfaces = _with_surface(
        surfaces,
        SURFACE_PERSISTENT_QACC_AUTHORITY,
        classification=RUNTIME_CLASS_TRANSIENT_FP_DEBT,
        reason=PHYSICAL_SUB2_NOT_ACHIEVED_STATEMENT,
        source_anchor="calm/hrm_text_158/native_full_stack/persistent_state_budget.py:1",
        proof_artifact_or_test="calm/llm_computer/tests/test_hrm_text_158_native_persistent_state_budget.py",
        diagnostic_exception_reason="persistent q/acc authority can be inspected before full sub2 replacement",
        why_cheaper_than_full_stack_first="static ledger check is cheaper than a trainer/kernel launch",
    )
    surfaces = _with_surface(
        surfaces,
        SURFACE_DENSE_INT16_PERSISTENT_ACCUMULATOR_ABSENCE,
        classification=RUNTIME_CLASS_MISSING,
        reason="current scaffold still names dense_int16_accumulator_control as a blocked/off-path control",
        source_anchor="calm/hrm_text_158/native_full_stack/sub2_native_birth_scaffold.py:1015",
        proof_artifact_or_test="calm/llm_computer/tests/test_hrm_text_158_native_sub2_native_birth_scaffold.py",
        diagnostic_exception_reason=(
            "Stability/functional-veto diagnostics measure flip dynamics on the "
            "**existing dense LIVE vote-acc path** that this row already records "
            "as control/baseline, not candidate sub-2 authority. The diagnostic is "
            "informative while absence stays open because it answers whether "
            "ternary still trains under functional-window-veto **on that control "
            "path** — a mechanism-arm branch selection — without reclassifying the "
            "row, without claiming the int16 container is sub-2, and without "
            "promoting it to candidate authority. Closing the absence first means "
            "landing a production sparse/event-coded (or other sub-2) live carrier "
            "for vote-acc with parity and readiness reclass; that stack investment "
            "is only justified after the mechanism arm is known not to be "
            "rate-limited freeze. The diagnostic therefore buys a taxonomy branch "
            "before that investment, not a free pass on the ledger row."
        ),
        why_cheaper_than_full_stack_first=(
            "One bounded stability window (CPU/schema + GPU dynamics under the "
            "lane recipe) is cheaper than multi-phase sparse/event-coded acc "
            "production landing + parity + readiness reclass of this surface to "
            "sub2. Completing closure first also inverts information order: stack "
            "cost without knowing whether the mechanism arm is freeze-class."
        ),
        diagnostic_exclusion_reason=(
            "This diagnostic's claim surface is flip/veto dynamics under the "
            "existing dense LIVE control path only. It does not consume a sub-2 "
            "dense-acc replacement, does not reclassify this row, and does not "
            "authorize main-science or sub-2 claims. The absence remains a "
            "standing MAIN-science blocker (`missing` still blocks "
            "`ready_for_main_science`)."
        ),
    )
    surfaces = _with_surface(
        surfaces,
        SURFACE_Q_SIDECAR_VOTE_CARRIER,
        classification=RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
        reason="sidecar runtime is a reference helper and movement overlay, not a full runtime proof",
        source_anchor="calm/hrm_text_158/native_full_stack/sub2_native_birth_sidecar_runtime.py:1",
        proof_artifact_or_test="calm/llm_computer/tests/test_hrm_text_158_native_sub2_native_birth_scaffold.py",
        diagnostic_exception_reason="carrier semantics can be checked before full kernel residency",
        why_cheaper_than_full_stack_first="CPU/reference carrier fixture is cheaper than GPU science launch",
    )
    surfaces = _with_surface(
        surfaces,
        SURFACE_ACTIVATIONS_RESIDUALS,
        classification=RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
        reason="activation relief is CPU-provable/deferred measurement, not a sub2 runtime proof",
        source_anchor="calm/hrm_text_158/native_full_stack/activation_relief.py:1",
        proof_artifact_or_test="calm/llm_computer/tests/test_hrm_text_158_activation_relief.py",
        diagnostic_exception_reason="activation policy shape can be audited before full-stack runtime",
        why_cheaper_than_full_stack_first="CPU policy validation is cheaper than GPU memory proof",
    )
    surfaces = _with_surface(
        surfaces,
        SURFACE_ATTENTION_KV_ATTENTION_BUFFERS,
        classification=RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
        reason="attention/KV buffer module is an estimator/deferred measurement contract",
        source_anchor="calm/hrm_text_158/native_full_stack/attention_kv_buffers.py:1",
        proof_artifact_or_test="calm/llm_computer/tests/test_hrm_text_158_attention_kv_buffers.py",
        diagnostic_exception_reason="attention/KV byte accounting can be audited before full runtime replacement",
        why_cheaper_than_full_stack_first="static estimator validation is cheaper than GPU science launch",
    )
    surfaces = _with_surface(
        surfaces,
        SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS,
        classification=RUNTIME_CLASS_MISSING,
        reason="no dedicated backward saved-tensor/transient sub2 proof seam is present in Step 1",
        source_anchor="calm/hrm_text_158/native_full_stack/activation_relief.py:1",
        proof_artifact_or_test="test_hrm_text_158_full_sub2_runtime_readiness.py::test_missing_required_surfaces_fail_closed",
        diagnostic_exception_reason=(
            "Weight-side stability diagnostics (flip dynamics / "
            "functional-window-veto) do not execute the backward-saved-tensor "
            "remat/offload seam. The measurement's operands are q/acc flip state "
            "and protected-row surrogates, not saved-activation quantization. "
            "Completing this seam first is activation-runtime stack work the "
            "stability taxonomy does not ask for; the row stays missing for MAIN "
            "science."
        ),
        why_cheaper_than_full_stack_first=(
            "One stability diagnostic window is cheaper than GPU-hot "
            "remat/offload residency proof + live peak-memory receipts that "
            "`activation_relief.py` already defers as out of CPU scope."
        ),
        diagnostic_exclusion_reason=(
            "Diagnostic claim surface excludes backward-saved-tensor sub-2 proof. "
            "Absence remains a MAIN-science blocker. No inference that "
            "activations/backward surfaces \"do not matter\" is licensed "
            "(ternary_hybrid_stack.md:68)."
        ),
    )
    surfaces = _with_surface(
        surfaces,
        SURFACE_OPTIMIZER_CREDIT_STATE,
        classification=RUNTIME_CLASS_TRANSIENT_FP_DEBT,
        reason="credit capture and dense transient credit are engineering debt, not an end-state exception",
        source_anchor="calm/hrm_text_158/native_full_stack/fp_exceptions.py:51",
        proof_artifact_or_test="calm/llm_computer/tests/test_hrm_text_158_credit_bridge.py",
        diagnostic_exception_reason="credit state can be inspected before removing transient FP capture debt",
        why_cheaper_than_full_stack_first="read-only credit/optimizer audit is cheaper than trainer launch",
    )
    surfaces = _with_surface(
        surfaces,
        SURFACE_NATIVE_KERNELIZED_HOT_PATH,
        classification=RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
        reason="strict-sub2 scaffold still discloses qacc hot-loop residency as CPU-reference/not kernelized",
        source_anchor="calm/hrm_text_158/native_full_stack/sub2_native_birth_scaffold.py:1067",
        proof_artifact_or_test="calm/llm_computer/tests/test_hrm_text_158_native_global_rate_cap_gpu.py",
        diagnostic_exception_reason="hot-loop contract can be checked before occupying GPU with science",
        why_cheaper_than_full_stack_first="CPU/static contract proof is cheaper than GPU residency proof",
    )
    return tuple(surfaces)


def gated_sub2_checkpoint_path_surfaces() -> tuple[
    FullSub2RuntimeSurfaceReceipt, ...
]:
    """Readiness variant for the gated, default-off 2C4a checkpoint path."""

    surfaces = current_repo_scaffold_surfaces()
    surfaces = _with_surface(
        surfaces,
        SURFACE_PERSISTENT_QACC_AUTHORITY,
        classification=RUNTIME_CLASS_SUB2,
        reason=(
            f"{GATED_SUB2_CHECKPOINT_PATH_REASON}; 2C4a commit 9600c36 builds a "
            "reconstructable q+scale+bounded sidecar checkpoint path that excludes "
            "eligible BitLinear.weight from authoritative model_state and rejects "
            "raw eligible-weight fallback"
        ),
        source_anchor="calm/hrm_text_158/native_full_stack/trainer_sub2_authority.py:600",
        proof_artifact_or_test=(
            "calm/llm_computer/tests/test_hrm_text_158_trainer_sub2_authority.py::"
            "test_roundtrip_receipt_excludes_fp_masters_and_falsifies_poisoned_forward"
        ),
    )
    surfaces = _with_surface(
        surfaces,
        SURFACE_DENSE_INT16_PERSISTENT_ACCUMULATOR_ABSENCE,
        classification=RUNTIME_CLASS_SUB2,
        reason=(
            f"{GATED_SUB2_CHECKPOINT_PATH_REASON}; 2C4a commit 9600c36 "
            "saves/loads no dense int16 authority and loader rejects "
            "exact_accumulator_shadow plus top-level/per-tensor dense-int16 sidecar flags"
        ),
        source_anchor="calm/hrm_text_158/native_full_stack/trainer_sub2_authority.py:652",
        proof_artifact_or_test=(
            "calm/llm_computer/tests/test_hrm_text_158_trainer_sub2_authority.py::"
            "test_roundtrip_checkpoint_loader_rejects_dense_int16_sidecar_flags"
        ),
    )
    surfaces = _with_surface(
        surfaces,
        SURFACE_Q_SIDECAR_VOTE_CARRIER,
        classification=RUNTIME_CLASS_SUB2,
        reason=(
            f"{GATED_SUB2_CHECKPOINT_PATH_REASON}; 2C4a commit 9600c36 proves "
            "poisoned FP-master bypass falsification and a post-resume shadow-free "
            "authority update whose sidecar payload hash roundtrips after second load"
        ),
        source_anchor="calm/hrm_text_158/native_full_stack/trainer_sub2_authority.py:1151",
        proof_artifact_or_test=(
            "calm/llm_computer/tests/test_hrm_text_158_trainer_sub2_authority.py::"
            "test_roundtrip_receipt_excludes_fp_masters_and_falsifies_poisoned_forward"
        ),
    )
    return tuple(surfaces)


def gated_sub2_checkpoint_path_backward_recompute_surfaces() -> tuple[
    FullSub2RuntimeSurfaceReceipt, ...
]:
    """Step 3A1 readiness variant: backward saved tensors only."""

    surfaces = gated_sub2_checkpoint_path_surfaces()
    return _with_surface(
        surfaces,
        SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS,
        classification=RUNTIME_CLASS_SUB2,
        reason=(
            f"{GATED_LOSSLESS_RECOMPUTE_REASON}; Step 3A1 saved-tensor-hook "
            "receipt proves no extra stored internal recurrence-block saved "
            "payload while boundary z_H/z_L inputs remain accounted under "
            "activations_residuals"
        ),
        source_anchor="calm/hrm_text_158/native_full_stack/activation_relief.py:295",
        proof_artifact_or_test=(
            "calm/llm_computer/tests/test_hrm_text_158_activation_relief.py::"
            "test_backward_recompute_receipt_uses_saved_tensors_hooks_for_no_extra_internal_payload"
        ),
    )


def gated_sub2_checkpoint_path_activation_residuals_blocked_surfaces() -> tuple[
    FullSub2RuntimeSurfaceReceipt, ...
]:
    """Step 3A2 readiness variant: activation/residual row remains blocked."""

    surfaces = gated_sub2_checkpoint_path_backward_recompute_surfaces()
    return _with_surface(
        surfaces,
        SURFACE_ACTIVATIONS_RESIDUALS,
        classification=RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
        reason=(
            f"{ACTIVATION_RESIDUALS_BLOCKED_REASON}; zL_init is cross-referenced "
            "as existing non_eligible_hrm_tensors FP exception debt and is not "
            "solved by activation seam observation"
        ),
        source_anchor="calm/hrm_text_158/native_full_stack/activation_relief.py:1",
        proof_artifact_or_test=(
            "calm/llm_computer/tests/test_hrm_text_158_activation_relief.py::"
            "test_activation_residuals_fail_closed_receipt_enumerates_live_tensor_families_without_flip"
        ),
        diagnostic_exception_reason="activation policy shape can be audited before full-stack runtime",
        why_cheaper_than_full_stack_first="CPU policy validation is cheaper than GPU memory proof",
    )


def gated_sub2_checkpoint_path_attention_kv_blocked_surfaces() -> tuple[
    FullSub2RuntimeSurfaceReceipt, ...
]:
    """Step 3B readiness variant: attention/KV row remains blocked."""

    surfaces = gated_sub2_checkpoint_path_activation_residuals_blocked_surfaces()
    return _with_surface(
        surfaces,
        SURFACE_ATTENTION_KV_ATTENTION_BUFFERS,
        classification=RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
        reason=(
            f"{ATTENTION_KV_BLOCKED_REASON}; q/k/v seam observations, PrefixLM "
            "mask, GQA repeat, SDPA workspace, and runtime KVCache caveats are "
            "blocker evidence, not sub2 credit"
        ),
        source_anchor="calm/hrm_text_158/native_full_stack/attention_kv_buffers.py:1",
        proof_artifact_or_test=(
            "calm/llm_computer/tests/test_hrm_text_158_attention_kv_buffers.py::"
            "test_attention_kv_fail_closed_receipt_enumerates_qkv_allowlist_without_flip"
        ),
        diagnostic_exception_reason="attention/KV byte accounting can be audited before full runtime replacement",
        why_cheaper_than_full_stack_first="static estimator validation is cheaper than GPU science launch",
    )


def gated_sub2_checkpoint_path_optimizer_credit_state_blocked_surfaces() -> tuple[
    FullSub2RuntimeSurfaceReceipt, ...
]:
    """Step 3C readiness variant: optimizer/credit-state row remains blocked."""

    surfaces = gated_sub2_checkpoint_path_attention_kv_blocked_surfaces()
    return _with_surface(
        surfaces,
        SURFACE_OPTIMIZER_CREDIT_STATE,
        classification=RUNTIME_CLASS_TRANSIENT_FP_DEBT,
        reason=(
            f"{OPTIMIZER_CREDIT_STATE_BLOCKED_REASON}; credit_capture_tensors "
            "is attribution-only transient FP debt and cannot satisfy the "
            "optimizer_credit_state row"
        ),
        source_anchor=(
            "calm/hrm_text_158/native_full_stack/optimizer_credit_state.py:1"
        ),
        proof_artifact_or_test=(
            "calm/llm_computer/tests/test_hrm_text_158_optimizer_credit_state.py::"
            "test_optimizer_credit_state_fail_closed_receipt_enumerates_dense_debt_without_flip"
        ),
        diagnostic_exception_reason="credit state can be inspected before removing transient FP capture debt",
        why_cheaper_than_full_stack_first="read-only credit/optimizer audit is cheaper than trainer launch",
    )


def gated_sub2_checkpoint_path_native_kernelized_hot_path_blocked_surfaces() -> tuple[
    FullSub2RuntimeSurfaceReceipt, ...
]:
    """Step 4A readiness variant: native/kernelized hot-path row remains blocked."""

    surfaces = gated_sub2_checkpoint_path_optimizer_credit_state_blocked_surfaces()
    return _with_surface(
        surfaces,
        SURFACE_NATIVE_KERNELIZED_HOT_PATH,
        classification=RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
        reason=(
            f"{NATIVE_KERNELIZED_HOT_PATH_BLOCKED_REASON}; composed-path q_acc_apply "
            "APPLY parity proven (B2-4) does not open cap SELECTION, full-loop, or "
            "hot-loop residency; device=cuda/VRAM residency is not hot-loop "
            "residency"
        ),
        source_anchor=(
            "calm/hrm_text_158/native_full_stack/native_kernelized_hot_path.py:1"
        ),
        proof_artifact_or_test=(
            "calm/llm_computer/tests/test_hrm_text_158_native_kernelized_hot_path.py::"
            "test_native_kernelized_hot_path_receipt_enumerates_current_blockers_without_flip"
        ),
        diagnostic_exception_reason="hot-loop contract can be checked before occupying GPU with science",
        why_cheaper_than_full_stack_first="CPU/static contract proof is cheaper than GPU residency proof",
    )


def step2a_candidate_persistent_core_absence_surfaces() -> tuple[
    FullSub2RuntimeSurfaceReceipt, ...
]:
    """Candidate receipt fixture that deliberately leaves live rows blocking."""

    surfaces = current_repo_scaffold_surfaces()
    surfaces = _with_surface(
        surfaces,
        SURFACE_PERSISTENT_QACC_AUTHORITY,
        classification=RUNTIME_CLASS_TRANSIENT_FP_DEBT,
        reason=(
            "Step 2A proves a CPU/reference candidate receipt only; live q/acc "
            "authority still routes through the existing dense/int16 control seams"
        ),
        source_anchor="calm/hrm_text_158/native_full_stack/persistent_core_sub2_absence.py:1",
        proof_artifact_or_test="calm/llm_computer/tests/test_hrm_text_158_persistent_core_sub2_absence.py",
        diagnostic_exception_reason="candidate q/acc authority can be inspected, but live authority is not converted",
        why_cheaper_than_full_stack_first="candidate-only CPU receipt is cheaper than a trainer authority conversion",
    )
    surfaces = _with_surface(
        surfaces,
        SURFACE_DENSE_INT16_PERSISTENT_ACCUMULATOR_ABSENCE,
        classification=RUNTIME_CLASS_MISSING,
        reason=(
            "candidate sidecar persistence has no dense shadow, but live "
            "persistent_state_budget/vote_update still expose int16 accumulator authority"
        ),
        source_anchor="calm/hrm_text_158/native_full_stack/vote_update.py:203",
        proof_artifact_or_test="calm/llm_computer/tests/test_hrm_text_158_persistent_core_sub2_absence.py::test_step2a_candidate_fixture_does_not_flip_live_rows",
    )
    surfaces = _with_surface(
        surfaces,
        SURFACE_Q_SIDECAR_VOTE_CARRIER,
        classification=RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
        reason=(
            "sidecar carrier is candidate/reference proof only; Step 2B must convert "
            "a trainer-used authority seam before this live row can become sub2"
        ),
        source_anchor="calm/hrm_text_158/native_full_stack/sub2_native_birth_sidecar_runtime.py:1",
        proof_artifact_or_test="calm/llm_computer/tests/test_hrm_text_158_persistent_core_sub2_absence.py",
        diagnostic_exception_reason="candidate carrier can be audited before live authority conversion",
        why_cheaper_than_full_stack_first="CPU/reference candidate proof is cheaper than live vote-update API churn",
    )
    return tuple(surfaces)


def apply_live_p1_conversion_surface_overrides(
    receipt: Any,
    *,
    base_surfaces: Sequence[FullSub2RuntimeSurfaceReceipt] | None = None,
    require_source_at_head: bool = True,
) -> tuple[FullSub2RuntimeSurfaceReceipt, ...]:
    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
        AUTHORIZED_P1B_SURFACE_TUPLE,
        AUTHORIZED_P1B_SURFACE_TUPLE_2ROW,
        validate_trainer_sub2_authority_live_conversion_receipt,
    )

    validate_trainer_sub2_authority_live_conversion_receipt(
        receipt,
        require_source_at_head=require_source_at_head,
    )
    if not receipt.readiness_row_flip_authorized:
        raise ValueError("P1b receipt does not authorize readiness row flips")
    authorized = tuple(receipt.readiness_row_flip_authorized_surface_names)
    if authorized not in (AUTHORIZED_P1B_SURFACE_TUPLE, AUTHORIZED_P1B_SURFACE_TUPLE_2ROW):
        raise ValueError("P1b authorized surface tuple mismatch")

    surfaces = list(base_surfaces or current_repo_scaffold_surfaces())
    reason_prefix = (
        f"P1 live conversion receipt source_commit_sha={receipt.source_commit_sha}; "
        f"p1_envelope_sha256={receipt.p1_envelope_sha256}; "
        f"inner_payload_sha256={receipt.inner_authoritative_state_payload_sha256}"
    )
    proof_test = (
        "calm/llm_computer/tests/test_hrm_text_158_full_sub2_runtime_readiness.py::"
        "test_live_p1_authority_conversion_flips_exactly_authorized_rows"
    )
    if authorized == AUTHORIZED_P1B_SURFACE_TUPLE:
        row_specs = {
            SURFACE_PERSISTENT_QACC_AUTHORITY: (
                "live production P1 checkpoint authority routing + cached parent install; "
                f"{reason_prefix}"
            ),
            SURFACE_DENSE_INT16_PERSISTENT_ACCUMULATOR_ABSENCE: (
                "P1 live envelope saves/loads no dense int16 persistent accumulator authority; "
                f"{reason_prefix}"
            ),
            SURFACE_Q_SIDECAR_VOTE_CARRIER: (
                "P1 live envelope vote-carrier subproof on production load path "
                f"(q_changed_count={receipt.q_changed_count}); {reason_prefix}"
            ),
        }
    else:
        row_specs = {
            SURFACE_PERSISTENT_QACC_AUTHORITY: (
                "live production P1 checkpoint authority routing + cached parent install; "
                f"{reason_prefix}"
            ),
            SURFACE_DENSE_INT16_PERSISTENT_ACCUMULATOR_ABSENCE: (
                "P1 live envelope saves/loads no dense int16 persistent accumulator authority; "
                f"{reason_prefix}"
            ),
        }

    for surface_id, reason in row_specs.items():
        surfaces = list(
            _with_surface(
                tuple(surfaces),
                surface_id,
                classification=RUNTIME_CLASS_SUB2,
                reason=reason,
                source_anchor="calm/hrm_text_158/native_full_stack/trainer_sub2_authority.py:2288",
                proof_artifact_or_test=proof_test,
            )
        )
    return tuple(surfaces)


def live_p1_authority_conversion_surfaces(
    receipt: Any,
) -> FullSub2RuntimeReadyForScienceReceipt:
    overridden = apply_live_p1_conversion_surface_overrides(receipt)
    result = build_full_sub2_runtime_ready_for_science(overridden)
    # Diagnostic eligibility follows the fixture ledger on the recomputed readiness
    # receipt. The source P1 conversion receipt's own non-diagnostic field is a
    # different object and is not rewritten here.
    if result.ready_for_main_science:
        raise ValueError(
            "P1b live conversion must not set ready_for_main_science"
        )
    return result


def post_p1_live_scaffold_surfaces(
    p1_receipt: Any,
    *,
    require_source_at_head: bool = True,
) -> tuple[FullSub2RuntimeSurfaceReceipt, ...]:
    return apply_live_p1_conversion_surface_overrides(
        p1_receipt,
        require_source_at_head=require_source_at_head,
    )


def apply_live_r1_backward_wiring_surface_overrides(
    receipt: Any,
    *,
    base_surfaces: Sequence[FullSub2RuntimeSurfaceReceipt] | None = None,
) -> tuple[FullSub2RuntimeSurfaceReceipt, ...]:
    from calm.hrm_text_158.native_full_stack.activation_relief import (
        AUTHORIZED_R1_L_SURFACE_TUPLE,
        BackwardRecomputeSavedTensorReceipt,
        LaunchRuntimeBackwardValidationReceipt,
        PROOF_KIND_CPU_PRODUCTION_AUTOGAD_WIRING,
        PROOF_KIND_LAUNCH_RUNTIME_VALIDATION,
        TrainerBackwardWiringProofReceipt,
        validate_launch_runtime_backward_receipt,
        validate_trainer_backward_wiring_proof_receipt,
    )

    if isinstance(receipt, BackwardRecomputeSavedTensorReceipt):
        raise ValueError(
            "fixture backward recompute receipt cannot flip live scaffold"
        )
    if isinstance(receipt, TrainerBackwardWiringProofReceipt):
        validate_trainer_backward_wiring_proof_receipt(receipt)
        raise ValueError(
            "CPU production autograd wiring receipt cannot flip live scaffold"
        )

    proof_kind = getattr(receipt, "proof_kind", None)
    if proof_kind == PROOF_KIND_CPU_PRODUCTION_AUTOGAD_WIRING:
        raise ValueError(
            "CPU production autograd wiring receipt cannot flip live scaffold"
        )
    if proof_kind != PROOF_KIND_LAUNCH_RUNTIME_VALIDATION:
        raise ValueError(
            "only launch_runtime_validation receipts may flip backward row"
        )
    if not isinstance(receipt, LaunchRuntimeBackwardValidationReceipt):
        raise TypeError(
            "launch runtime receipt must be LaunchRuntimeBackwardValidationReceipt"
        )
    # Type-1 launch/liveness receipts cannot authorize a readiness row flip.
    # Parked post claim-contract null: no row-flip authority mintable from smoke.
    from calm.hrm_text_158.native_full_stack.activation_relief import (
        R1_ROW_FLIP_AUTHORITY_UNAVAILABLE,
    )

    _ = (receipt, base_surfaces)
    raise ValueError(
        f"{R1_ROW_FLIP_AUTHORITY_UNAVAILABLE}: "
        "Type-1 launch/liveness receipt cannot authorize readiness row flip; "
        "row-flip authority is deferred (no Type-2 in tree)"
    )


def live_r1_backward_wiring_surfaces(
    receipt: Any,
    *,
    base_surfaces: Sequence[FullSub2RuntimeSurfaceReceipt] | None = None,
) -> FullSub2RuntimeReadyForScienceReceipt:
    overridden = apply_live_r1_backward_wiring_surface_overrides(
        receipt,
        base_surfaces=base_surfaces,
    )
    return build_full_sub2_runtime_ready_for_science(overridden)


def apply_live_activation_residuals_surface_overrides(
    receipt: Any,
    *,
    base_surfaces: Sequence[FullSub2RuntimeSurfaceReceipt] | None = None,
) -> tuple[FullSub2RuntimeSurfaceReceipt, ...]:
    """Fail-closed: no CPU/fixture receipt may flip activations_residuals live row."""

    from calm.hrm_text_158.native_full_stack.activation_relief import (
        ActivationResidualsFailClosedReceipt,
        LaunchRuntimeBackwardValidationReceipt,
        TrainerActivationResidualsSeamProofReceipt,
        validate_activation_residuals_fail_closed_receipt,
        validate_trainer_activation_residuals_seam_proof_receipt,
    )
    from calm.hrm_text_158.native_full_stack.activation_residuals_launch import (
        AUTHORIZED_R2A_L_SURFACE_TUPLE,
        LaunchRuntimeActivationResidualsValidationReceipt,
        PROOF_KIND_LAUNCH_RUNTIME_ACTIVATION_RESIDUALS,
        canonicalize_base_sub2_surface_ids,
        load_r2al_base_receipts_from_env,
        validate_launch_runtime_activation_residuals_receipt,
        validate_r2al_live_base_preflight,
    )

    if isinstance(receipt, ActivationResidualsFailClosedReceipt):
        validate_activation_residuals_fail_closed_receipt(receipt)
        raise ValueError(
            "fixture activation/residual fail-closed receipt cannot flip live scaffold"
        )
    if isinstance(receipt, TrainerActivationResidualsSeamProofReceipt):
        validate_trainer_activation_residuals_seam_proof_receipt(receipt)
        raise ValueError(
            "CPU production seam observation receipt cannot flip live scaffold"
        )
    from calm.hrm_text_158.native_full_stack.activation_residuals_m1_remat import (
        TrainerActivationResidualsLosslessEquivalenceReceipt,
        validate_trainer_activation_residuals_lossless_equivalence_receipt,
    )

    if isinstance(receipt, TrainerActivationResidualsLosslessEquivalenceReceipt):
        validate_trainer_activation_residuals_lossless_equivalence_receipt(receipt)
        raise ValueError(
            "CPU lossless equivalence receipt cannot flip live scaffold"
        )
    if isinstance(receipt, LaunchRuntimeBackwardValidationReceipt):
        raise ValueError(
            "R1-L backward launch receipt cannot flip activations_residuals row"
        )

    proof_kind = getattr(receipt, "proof_kind", None)
    if proof_kind != PROOF_KIND_LAUNCH_RUNTIME_ACTIVATION_RESIDUALS:
        raise ValueError(
            "only launch_runtime_activation_residuals receipts may flip activations row"
        )
    if not isinstance(receipt, LaunchRuntimeActivationResidualsValidationReceipt):
        raise TypeError(
            "launch runtime activation/residual receipt must be "
            "LaunchRuntimeActivationResidualsValidationReceipt"
        )
    validate_launch_runtime_activation_residuals_receipt(receipt)
    p1_receipt, r1l_receipt = load_r2al_base_receipts_from_env(receipt.proof_env_embedded)
    p1_path = Path(receipt.proof_env_embedded["R2AL_P1_RECEIPT_JSON"])
    r1l_path = Path(receipt.proof_env_embedded["R2AL_R1L_RECEIPT_JSON"])
    validate_r2al_live_base_preflight(
        receipt,
        p1_receipt=p1_receipt,
        r1l_receipt=r1l_receipt,
        p1_receipt_path=p1_path,
        r1l_receipt_path=r1l_path,
    )

    if base_surfaces is None:
        live_base = live_r1_backward_launch_surfaces(
            r1l_receipt,
            p1_receipt,
            require_source_at_head=False,
        )
        surfaces = list(live_base.surfaces)
    else:
        surfaces = list(base_surfaces)
        live_base = live_r1_backward_launch_surfaces(
            r1l_receipt,
            p1_receipt,
            require_source_at_head=False,
        )
        live_ids = {
            surface.surface_id: surface.classification
            for surface in live_base.surfaces
        }
        provided_sub2 = canonicalize_base_sub2_surface_ids(
            surface.surface_id
            for surface in surfaces
            if surface.classification == RUNTIME_CLASS_SUB2
        )
        if provided_sub2 != tuple(receipt.base_sub2_surface_ids):
            raise ValueError(
                "provided base_surfaces do not match live P1+R1-L base sub2 ids"
            )
        for surface in surfaces:
            if live_ids.get(surface.surface_id) != surface.classification:
                raise ValueError(
                    "provided base_surfaces classification mismatch vs live P1+R1-L base"
                )

    base_by_id = {surface.surface_id: surface for surface in surfaces}
    reason = (
        f"{GATED_LOSSLESS_RECOMPUTE_REASON}; R2-A-L launch/runtime validation "
        f"launch_source_commit_sha={receipt.launch_source_commit_sha}; "
        "M1 saved-tensor-hook remat proves handle-pack authority replacement "
        "with GPU memory measurement on proven P1+R1-L live base"
    )
    proof_test = (
        "calm/llm_computer/tests/test_hrm_text_158_r2al_launch_schema.py::"
        "test_r2al_applier_changes_exactly_one_surface"
    )
    flipped = _with_surface(
        tuple(surfaces),
        SURFACE_ACTIVATIONS_RESIDUALS,
        classification=RUNTIME_CLASS_SUB2,
        reason=reason,
        source_anchor="calm/hrm_text_158/native_full_stack/activation_residuals_m1_remat.py:24",
        proof_artifact_or_test=proof_test,
    )
    for surface in flipped:
        if surface.surface_id == SURFACE_ACTIVATIONS_RESIDUALS:
            continue
        if surface.classification != base_by_id[surface.surface_id].classification:
            raise ValueError(
                "launch runtime flip changed more than activations_residuals"
            )

    result = build_full_sub2_runtime_ready_for_science(flipped)
    if result.ready_for_main_science:
        raise ValueError("launch runtime flip must not set ready_for_main_science")
    if not result.ready_for_pre_full_stack_diagnostic:
        raise ValueError(
            "launch runtime flip must set ready_for_pre_full_stack_diagnostic"
        )
    if tuple(receipt.applier_flipped_surface_ids) != AUTHORIZED_R2A_L_SURFACE_TUPLE:
        raise ValueError("launch runtime authorized surface tuple mismatch")
    if SURFACE_ACTIVATIONS_RESIDUALS not in {
        surface.surface_id
        for surface in flipped
        if surface.classification == RUNTIME_CLASS_SUB2
    }:
        raise ValueError("launch runtime flip must set activations_residuals row to sub2")
    return flipped


def live_r1_backward_launch_surfaces(
    r1l_receipt: Any,
    p1_receipt: Any,
    *,
    require_source_at_head: bool = True,
) -> FullSub2RuntimeReadyForScienceReceipt:
    base = post_p1_live_scaffold_surfaces(
        p1_receipt,
        require_source_at_head=require_source_at_head,
    )
    return live_r1_backward_wiring_surfaces(
        r1l_receipt,
        base_surfaces=base,
    )


def fixture_full_sub2_runtime_ready_for_science(
    fixture_name: str = FIXTURE_CURRENT_REPO,
) -> FullSub2RuntimeReadyForScienceReceipt:
    if fixture_name == FIXTURE_CURRENT_REPO:
        surfaces = current_repo_scaffold_surfaces()
    elif fixture_name == FIXTURE_GATED_SUB2_CHECKPOINT_PATH:
        surfaces = gated_sub2_checkpoint_path_surfaces()
    elif fixture_name == FIXTURE_GATED_SUB2_CHECKPOINT_PATH_BACKWARD_RECOMPUTE:
        surfaces = gated_sub2_checkpoint_path_backward_recompute_surfaces()
    elif fixture_name == FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ACTIVATION_RESIDUALS_BLOCKED:
        surfaces = gated_sub2_checkpoint_path_activation_residuals_blocked_surfaces()
    elif fixture_name == FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ATTENTION_KV_BLOCKED:
        surfaces = gated_sub2_checkpoint_path_attention_kv_blocked_surfaces()
    elif (
        fixture_name
        == FIXTURE_GATED_SUB2_CHECKPOINT_PATH_OPTIMIZER_CREDIT_STATE_BLOCKED
    ):
        surfaces = gated_sub2_checkpoint_path_optimizer_credit_state_blocked_surfaces()
    elif (
        fixture_name
        == FIXTURE_GATED_SUB2_CHECKPOINT_PATH_NATIVE_KERNELIZED_HOT_PATH_BLOCKED
    ):
        surfaces = (
            gated_sub2_checkpoint_path_native_kernelized_hot_path_blocked_surfaces()
        )
    elif fixture_name == FIXTURE_MAIN_READY:
        surfaces = main_ready_fixture_surfaces()
    elif fixture_name == FIXTURE_MISSING_ACTIVATIONS:
        surfaces = _with_surface(
            main_ready_fixture_surfaces(),
            SURFACE_ACTIVATIONS_RESIDUALS,
            classification=RUNTIME_CLASS_MISSING,
            reason="fixture proves missing activation/residual surface blocks main science",
            source_anchor="calm/hrm_text_158/native_full_stack/activation_relief.py:1",
            proof_artifact_or_test="test_hrm_text_158_full_sub2_runtime_readiness.py::test_missing_required_surfaces_fail_closed",
        )
    elif fixture_name == FIXTURE_MISSING_ATTENTION:
        surfaces = _with_surface(
            main_ready_fixture_surfaces(),
            SURFACE_ATTENTION_KV_ATTENTION_BUFFERS,
            classification=RUNTIME_CLASS_MISSING,
            reason="fixture proves missing attention/KV surface blocks main science",
            source_anchor="calm/hrm_text_158/native_full_stack/attention_kv_buffers.py:1",
            proof_artifact_or_test="test_hrm_text_158_full_sub2_runtime_readiness.py::test_missing_required_surfaces_fail_closed",
        )
    elif fixture_name == FIXTURE_MISSING_BACKWARD:
        surfaces = _with_surface(
            main_ready_fixture_surfaces(),
            SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS,
            classification=RUNTIME_CLASS_MISSING,
            reason="fixture proves missing backward saved-tensor/transient surface blocks main science",
            source_anchor="calm/hrm_text_158/native_full_stack/activation_relief.py:1",
            proof_artifact_or_test="test_hrm_text_158_full_sub2_runtime_readiness.py::test_missing_required_surfaces_fail_closed",
        )
    elif fixture_name == FIXTURE_PRE_FULL_STACK_DIAGNOSTIC:
        surfaces = _with_surface(
            main_ready_fixture_surfaces(),
            SURFACE_ACTIVATIONS_RESIDUALS,
            classification=RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
            reason="fixture proves diagnostic surfaces block main science",
            source_anchor="calm/hrm_text_158/native_full_stack/activation_relief.py:1",
            proof_artifact_or_test="test_hrm_text_158_full_sub2_runtime_readiness.py::test_pre_full_stack_diagnostic_allows_only_diagnostic_readiness",
            diagnostic_exception_reason="activation diagnostic is intentionally pre-full-stack",
            why_cheaper_than_full_stack_first="static diagnostic is cheaper than full runtime replacement",
        )
    elif fixture_name == FIXTURE_TRANSIENT_FP_DEBT:
        surfaces = _with_surface(
            main_ready_fixture_surfaces(),
            SURFACE_OPTIMIZER_CREDIT_STATE,
            classification=RUNTIME_CLASS_TRANSIENT_FP_DEBT,
            reason="fixture proves transient FP debt blocks main science",
            source_anchor="calm/hrm_text_158/native_full_stack/fp_exceptions.py:51",
            proof_artifact_or_test="test_hrm_text_158_full_sub2_runtime_readiness.py::test_transient_fp_debt_blocks_main_science",
            diagnostic_exception_reason="transient FP debt can be measured before full replacement",
            why_cheaper_than_full_stack_first="debt classification is cheaper than mechanism science launch",
        )
    elif fixture_name == FIXTURE_STEP2A_CANDIDATE_PERSISTENT_CORE_ABSENCE:
        surfaces = step2a_candidate_persistent_core_absence_surfaces()
    elif fixture_name == FIXTURE_LIVE_P1_AUTHORITY_CONVERSION:
        raise ValueError(
            "live_p1_authority_conversion requires an explicit validated receipt JSON; "
            "use scripts/hrm_text_158_full_sub2_runtime_readiness.py "
            "--fixture live_p1_authority_conversion --live-p1-receipt-json PATH"
        )
    elif fixture_name == FIXTURE_LIVE_R1_BACKWARD_LAUNCH:
        raise ValueError(
            "live_r1_backward_launch requires validated P1 and R1-L receipts; "
            "use scripts/hrm_text_158_full_sub2_runtime_readiness.py "
            "--fixture live_r1_backward_launch --live-p1-receipt-json PATH "
            "--r1l-receipt-json PATH"
        )
    else:
        valid = ", ".join(FULL_SUB2_RUNTIME_FIXTURE_NAMES)
        raise ValueError(f"unknown full-sub2 readiness fixture {fixture_name!r}; valid={valid}")
    return build_full_sub2_runtime_ready_for_science(surfaces)
