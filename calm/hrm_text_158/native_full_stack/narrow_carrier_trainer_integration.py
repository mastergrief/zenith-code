"""S2 default-off trainer-facing narrow-carrier (W6) integration seam.

Applies S1 pack_w6/unpack_w6 at the bounded-delta vote_update_state boundary only.
Does not enable vote_update COMPRESSED_ACCUMULATORS or mutate checkpoints.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    clip_then_pack_w6,
    pack_w6,
    strict_roundtrip_w6_tensor,
    unpack_w6,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    DEFAULT_CROSSING_THRESHOLD_ABS,
    carry_self_update_row,
    crossing_bool_w6,
    crosses_threshold,
)

RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV = (
    "HRM_TEXT_158_RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION"
)

CLASSIFIER_S2_HARNESS_FAIL = "S2_HARNESS_FAIL"
CLASSIFIER_S2_PARITY_DIVERGES = "S2_PARITY_DIVERGES"
CLASSIFIER_S2_FLAG_OFF_IDENTITY_AND_PARITY_OK = "S2_FLAG_OFF_IDENTITY_AND_PARITY_OK"

CLASSIFIER_PRECEDENCE: tuple[str, ...] = (
    CLASSIFIER_S2_HARNESS_FAIL,
    CLASSIFIER_S2_PARITY_DIVERGES,
    CLASSIFIER_S2_FLAG_OFF_IDENTITY_AND_PARITY_OK,
)

CLASSIFIER_S3BA_HARNESS_FAIL = "S3BA_HARNESS_FAIL"
CLASSIFIER_S3BA_PARITY_DIVERGES = "S3BA_PARITY_DIVERGES"
CLASSIFIER_S3BA_VECTOR_WIRING_OK = "S3BA_VECTOR_WIRING_OK"

CLASSIFIER_S3BA_PRECEDENCE: tuple[str, ...] = (
    CLASSIFIER_S3BA_HARNESS_FAIL,
    CLASSIFIER_S3BA_PARITY_DIVERGES,
    CLASSIFIER_S3BA_VECTOR_WIRING_OK,
)

S3BA_EXPLICIT_NON_CLAIMS: tuple[str, ...] = (
    "vectorized_trainer_boundary_wiring_proof_only",
    "not_gpu_dynamics_parity_s3bb",
    "not_live_training",
    "not_checkpoint_pt_mutation",
    "not_compressed_vote_update_format_enablement",
    "not_dynamics_stability_full_sub2_readiness",
    "not_physical_sub2",
)

EXPLICIT_NON_CLAIMS: tuple[str, ...] = (
    "trainer_facing_narrow_carrier_integration_proof_only",
    "default_off_not_live_training",
    "not_physical_sub2",
    "not_gpu_parity_s3",
    "not_checkpoint_pt_mutation",
    "not_compressed_vote_update_format_enablement",
    "not_dynamics_stability_full_sub2_readiness",
)

FORBIDDEN_CLAIM_FIELDS: frozenset[str] = frozenset(
    {
        "sub2_win",
        "full_sub2_runtime_ready",
        "gpu_launch_authorized",
        "training_claim",
        "stability_claim",
        "live_training_enabled",
        "checkpoint_mutation",
        "packed_w6_persistent_state",
    }
)

PACKED_W6_STATE_FIELD_MARKERS: tuple[str, ...] = (
    "packed_w6",
    "w6_packed",
    "narrow_carrier_packed",
)
AUTHORIZED_W6_BYTE_PACKED_FIELD_MARKERS: tuple[str, ...] = (
    "w6_byte_packed_accumulator_persisted",
    "w6_byte_packed_payload",
    "w6_byte_packed_schema",
    "w6_byte_packed_logical_shape",
    "w6_byte_packed_logical_numel",
    "w6_byte_packed_persistent_accumulator_saved",
)


def persistent_w6_byte_packed_enabled(*, enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return os.environ.get("HRM_TEXT_158_PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED") == "1"


def narrow_carrier_w6_enabled(*, enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return os.environ.get(RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV) == "1"


def strict_roundtrip_int16_value_through_trainer_boundary(value: int) -> int:
    """Strict in-domain trainer boundary roundtrip using S1 pack_w6/unpack_w6."""

    return unpack_w6(pack_w6(int(value)))


def roundtrip_replay_clip_int16_value(value: int) -> int:
    """Replay-only clip helper; separate from enabled trainer boundary semantics."""

    return unpack_w6(clip_then_pack_w6(int(value)))


def roundtrip_int16_values_through_trainer_boundary(
    values: Sequence[int],
    *,
    enabled: bool | None = None,
) -> list[int]:
    """B2 helper: strict encode/decode each in-domain int16 lane at trainer boundary."""

    if not narrow_carrier_w6_enabled(enabled=enabled):
        return [int(v) for v in values]
    return [strict_roundtrip_int16_value_through_trainer_boundary(int(v)) for v in values]


def apply_trainer_boundary_narrow_carrier(
    accumulators: torch.Tensor,
    *,
    enabled: bool | None = None,
) -> torch.Tensor:
    """Default-off trainer boundary: identity int16 when off; strict W6 roundtrip when on."""

    if accumulators.dtype != torch.int16:
        raise ValueError(f"accumulators must be torch.int16, got {accumulators.dtype}")
    if not narrow_carrier_w6_enabled(enabled=enabled):
        return accumulators
    return strict_roundtrip_w6_tensor(accumulators.detach())


def count_int16_vs_w6_crossing_mismatches(
    rows: Sequence[tuple[int, int, int]],
    *,
    enabled: bool | None = None,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
) -> int:
    """B3 helper: int16 oracle (width=16) vs W6 carrier crossing on fixture rows."""

    if not narrow_carrier_w6_enabled(enabled=enabled):
        return 0

    mismatches = 0
    for pre_acc, vote, q_level in rows:
        new_acc_16 = carry_self_update_row(int(pre_acc), int(vote), width=16)
        new_acc_6 = carry_self_update_row(int(pre_acc), int(vote), width=6)
        oracle_cross = crosses_threshold(
            new_acc_16,
            current_q_level=int(q_level),
            threshold_abs=int(threshold_abs),
        )
        try:
            carrier_acc = strict_roundtrip_int16_value_through_trainer_boundary(new_acc_6)
        except ValueError:
            mismatches += 1
            continue
        carrier_cross = crossing_bool_w6(
            carrier_acc,
            int(q_level),
            threshold_abs=int(threshold_abs),
        )
        if oracle_cross != carrier_cross:
            mismatches += 1
    return mismatches


def assert_no_packed_w6_state_leak(
    payload: Mapping[str, Any],
    *,
    byte_packed_enabled: bool | None = None,
) -> None:
    """B4 helper: reject durable packed-W6 fields in serializable trainer state."""

    authorized = persistent_w6_byte_packed_enabled(enabled=byte_packed_enabled)

    def _walk(value: Any, *, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key).lower()
                if any(marker in key_text for marker in AUTHORIZED_W6_BYTE_PACKED_FIELD_MARKERS):
                    if not authorized:
                        raise ValueError(f"packed W6 state leak at {path}.{key}")
                    _walk(child, path=f"{path}.{key}")
                    continue
                if any(marker in key_text for marker in PACKED_W6_STATE_FIELD_MARKERS):
                    raise ValueError(f"packed W6 state leak at {path}.{key}")
                _walk(child, path=f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                _walk(child, path=f"{path}[{index}]")

    _walk(dict(payload), path="payload")


def emit_s2_classifier_receipt(
    *,
    harness_failures: Sequence[str] | None = None,
    flag_off_identity_pass: bool = True,
    boundary_roundtrip_pass: bool = True,
    parity_mismatch_count: int = 0,
    rollback_pass: bool = True,
) -> dict[str, Any]:
    """Emit S2 classifier receipt with explicit non-claims (B5)."""

    failures = list(dict.fromkeys(harness_failures or ()))
    if failures:
        primary = CLASSIFIER_S2_HARNESS_FAIL
    elif (
        not flag_off_identity_pass
        or not boundary_roundtrip_pass
        or int(parity_mismatch_count) > 0
        or not rollback_pass
    ):
        primary = CLASSIFIER_S2_PARITY_DIVERGES
    else:
        primary = CLASSIFIER_S2_FLAG_OFF_IDENTITY_AND_PARITY_OK

    return {
        "slice_id": "narrow_carrier_trainer_integration_v0",
        "primary_classifier": primary,
        "classifier_precedence": list(CLASSIFIER_PRECEDENCE),
        "narrow_carrier_w6_default_off": True,
        "harness_failures": failures,
        "parity_mismatch_count": int(parity_mismatch_count),
        "explicit_non_claims": list(EXPLICIT_NON_CLAIMS),
        "s2_ok_is_not_live_training": True,
        "s2_ok_is_not_gpu_parity": True,
        "s2_ok_is_not_checkpoint_mutation": True,
        "s2_ok_is_not_compressed_vote_update": True,
    }


def emit_s3ba_classifier_receipt(
    *,
    harness_failures: Sequence[str] | None = None,
    static_inspection_pass: bool = True,
    cpu_regression_pass: bool = True,
    parity_mismatch_count: int = 0,
    flag_off_identity_pass: bool = True,
    no_packed_state_leak_pass: bool = True,
) -> dict[str, Any]:
    """Emit S3ba classifier receipt with explicit non-claims."""

    failures = list(dict.fromkeys(harness_failures or ()))
    if failures or not static_inspection_pass:
        primary = CLASSIFIER_S3BA_HARNESS_FAIL
    elif (
        not cpu_regression_pass
        or int(parity_mismatch_count) > 0
        or not flag_off_identity_pass
        or not no_packed_state_leak_pass
    ):
        primary = CLASSIFIER_S3BA_PARITY_DIVERGES
    else:
        primary = CLASSIFIER_S3BA_VECTOR_WIRING_OK

    return {
        "slice_id": "narrow_carrier_vectorized_wiring_s3ba_v0",
        "primary_classifier": primary,
        "classifier_precedence": list(CLASSIFIER_S3BA_PRECEDENCE),
        "harness_failures": failures,
        "static_inspection_pass": bool(static_inspection_pass),
        "cpu_regression_pass": bool(cpu_regression_pass),
        "parity_mismatch_count": int(parity_mismatch_count),
        "flag_off_identity_pass": bool(flag_off_identity_pass),
        "no_packed_state_leak_pass": bool(no_packed_state_leak_pass),
        "explicit_non_claims": list(S3BA_EXPLICIT_NON_CLAIMS),
        "s3ba_ok_is_not_gpu_dynamics": True,
        "s3ba_ok_is_not_live_training": True,
        "s3ba_ok_is_not_checkpoint_mutation": True,
    }
