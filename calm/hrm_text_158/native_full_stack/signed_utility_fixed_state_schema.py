"""Result/telemetry schema for fixed-state signed-utility diagnostic (PLAN v5)."""
from __future__ import annotations

from typing import Any, Mapping

TERMINAL_CLASSES = (
    "SIGNED_CREDIT_SIGNAL_PRESENT_UNPROVEN",
    "SIGNED_CREDIT_SIGNAL_NULL_OR_HARMFUL",
    "UNVERIFIED_ASYMMETRIC_INTERVENTION",
    "UNVERIFIED_INTEGRITY_OR_EXECUTION",
)

REQUIRED_PHASE_MARKER_NAMES = (
    "PHASE_MATERIALIZE_BEGIN",
    "PHASE_MATERIALIZE_END",
    "PHASE_CAPTURE_BACKWARD_VOTE_BEGIN",
    "PHASE_CAPTURE_BACKWARD_VOTE_END",
    "PHASE_THREE_ARM_EVAL_BEGIN",
    "PHASE_THREE_ARM_EVAL_END",
    "PHASE_EMIT_BEGIN",
    "PHASE_EMIT_END",
)


class SchemaValidationError(ValueError):
    pass


def required_phase_marker_names() -> tuple[str, ...]:
    return REQUIRED_PHASE_MARKER_NAMES


def build_non_authoritative_developer_payload(diag: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": "developer_check",
        "non_authoritative": True,
        "schema": "post_seam_signed_utility_developer_payload_v0",
        "diag": dict(diag),
    }


def validate_authoritative_result_schema_v4_min(payload: Mapping[str, Any]) -> None:
    required = (
        "schema",
        "classifier",
        "L_prod",
        "L_inv",
        "L_noop",
        "epsilon",
        "parent_sha256_pre",
        "parent_sha256_post",
        "phase_markers",
        "nll_per_arm",
        "apply_integer_vote_update_from_frozen_plan_calls",
        "eligible_state_key_count",
    )
    missing = [k for k in required if k not in payload]
    if missing:
        raise SchemaValidationError(f"missing_fields:{missing}")
    if payload["classifier"] not in TERMINAL_CLASSES:
        raise SchemaValidationError("classifier_not_in_terminal_classes")
    nll = payload["nll_per_arm"]
    for arm in ("prod", "inv", "noop"):
        if arm not in nll:
            raise SchemaValidationError(f"nll_per_arm_missing:{arm}")
        row = nll[arm]
        for field in ("numerator_f64", "denominator", "mean"):
            if field not in row:
                raise SchemaValidationError(f"nll_arm_field_missing:{arm}.{field}")
    markers = payload["phase_markers"]
    if not isinstance(markers, Mapping):
        raise SchemaValidationError("phase_markers_not_mapping")
    for name in REQUIRED_PHASE_MARKER_NAMES:
        if name not in markers:
            raise SchemaValidationError(f"phase_marker_missing:{name}")
    n = int(payload["eligible_state_key_count"])
    calls = int(payload["apply_integer_vote_update_from_frozen_plan_calls"])
    if payload["classifier"] in TERMINAL_CLASSES[:2] and calls != 2 * n:
        raise SchemaValidationError("call_count_not_two_times_eligible_keys")


__all__ = [
    "REQUIRED_PHASE_MARKER_NAMES",
    "SchemaValidationError",
    "TERMINAL_CLASSES",
    "build_non_authoritative_developer_payload",
    "required_phase_marker_names",
    "validate_authoritative_result_schema_v4_min",
]
