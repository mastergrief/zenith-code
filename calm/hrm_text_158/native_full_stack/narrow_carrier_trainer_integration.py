"""S2 default-off trainer-facing narrow-carrier (W6) integration seam.

Applies S1 pack_w6/unpack_w6 at the bounded-delta vote_update_state boundary only.
Does not enable vote_update COMPRESSED_ACCUMULATORS or mutate checkpoints.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    clip_then_pack_w5,
    clip_then_pack_w5_tensor,
    clip_then_roundtrip_w5_tensor,
    clip_then_pack_w6,
    pack_w5,
    pack_w6,
    strict_roundtrip_w6_tensor,
    unpack_w5,
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
RUN_NARROW_CARRIER_W5_TRAINER_INTEGRATION_ENV = (
    "HRM_TEXT_158_RUN_NARROW_CARRIER_W5_TRAINER_INTEGRATION"
)
PERSISTENT_ACCUMULATOR_W5_BYTE_PACKED_ENV = (
    "HRM_TEXT_158_PERSISTENT_ACCUMULATOR_W5_BYTE_PACKED"
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
AUTHORIZED_W5_BYTE_PACKED_FIELD_MARKERS: tuple[str, ...] = (
    "w5_byte_packed_accumulator_persisted",
    "w5_byte_packed_payload",
    "w5_byte_packed_schema",
    "w5_byte_packed_logical_shape",
    "w5_byte_packed_logical_numel",
    "w5_byte_packed_persistent_accumulator_saved",
)
AUTHORIZED_Q_TERNARY_BYTE_PACKED_FIELD_MARKERS: tuple[str, ...] = (
    "q_ternary_byte_packed_persisted",
    "q_ternary_packed_payload",
    "q_ternary_packed_schema",
    "q_ternary_logical_shape",
    "q_ternary_logical_numel",
    "q_ternary_padding_values",
    "q_ternary_byte_packed_persisted_saved",
)
RAW_Q_LEVELS_FIELD = "q_levels"
PERSISTENT_Q_TERNARY_BASE3_CODEC_ENV = "HRM_TEXT_158_PERSISTENT_Q_TERNARY_BASE3_CODEC"
Q_CODEC_SELECTOR_BASE3 = "base3"


def persistent_w6_byte_packed_enabled(*, enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return os.environ.get("HRM_TEXT_158_PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED") == "1"


def persistent_w5_byte_packed_enabled(*, enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return os.environ.get(PERSISTENT_ACCUMULATOR_W5_BYTE_PACKED_ENV) == "1"


def persistent_q_ternary_byte_packed_enabled(*, enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return os.environ.get("HRM_TEXT_158_PERSISTENT_Q_TERNARY_BYTE_PACKED") == "1"


def persistent_q_ternary_base3_codec_enabled(*, enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return os.environ.get(PERSISTENT_Q_TERNARY_BASE3_CODEC_ENV) == "1"


def resolve_q_codec_selector(*, q_codec_selector: str | None = None) -> str:
    if q_codec_selector is not None:
        selector = str(q_codec_selector)
        if selector not in ("2bit", Q_CODEC_SELECTOR_BASE3):
            raise ValueError(
                f"q_codec_selector must be '2bit' or {Q_CODEC_SELECTOR_BASE3!r}, got {selector!r}"
            )
        return selector
    if persistent_q_ternary_base3_codec_enabled():
        return Q_CODEC_SELECTOR_BASE3
    return "2bit"


def narrow_carrier_w6_enabled(*, enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return os.environ.get(RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV) == "1"


def narrow_carrier_w5_enabled(*, enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return os.environ.get(RUN_NARROW_CARRIER_W5_TRAINER_INTEGRATION_ENV) == "1"


def roundtrip_clip_w5_int16_value_through_trainer_boundary(value: int) -> int:
    return unpack_w5(clip_then_pack_w5(int(value)))


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
    w5_enabled: bool | None = None,
    w6_enabled: bool | None = None,
) -> torch.Tensor:
    """Default-off trainer boundary: identity int16 when off; W5 clip roundtrip or W6 strict."""

    if accumulators.dtype != torch.int16:
        raise ValueError(f"accumulators must be torch.int16, got {accumulators.dtype}")
    use_w5 = narrow_carrier_w5_enabled(enabled=w5_enabled)
    use_w6 = narrow_carrier_w6_enabled(enabled=w6_enabled if w6_enabled is not None else enabled)
    if use_w5 and use_w6:
        raise ValueError("W5 and W6 narrow-carrier trainer integration are mutually exclusive")
    if use_w5:
        return clip_then_roundtrip_w5_tensor(accumulators.detach())
    if use_w6:
        return strict_roundtrip_w6_tensor(accumulators.detach())
    return accumulators


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


def assert_no_packed_w5_state_leak(
    payload: Mapping[str, Any],
    *,
    byte_packed_enabled: bool | None = None,
) -> None:
    authorized = persistent_w5_byte_packed_enabled(enabled=byte_packed_enabled)

    def _walk(value: Any, *, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key).lower()
                if any(marker in key_text for marker in AUTHORIZED_W5_BYTE_PACKED_FIELD_MARKERS):
                    if not authorized:
                        raise ValueError(f"packed W5 state leak at {path}.{key}")
                    _walk(child, path=f"{path}.{key}")
                    continue
                if "packed_w5" in key_text or "w5_packed" in key_text:
                    raise ValueError(f"packed W5 state leak at {path}.{key}")
                _walk(child, path=f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                _walk(child, path=f"{path}[{index}]")

    _walk(dict(payload), path="payload")


def assert_no_dual_persistent_acc_byte_packing(
    bounded_payload: Mapping[str, Any],
) -> None:
    w5 = bool(bounded_payload.get("w5_byte_packed_accumulator_persisted"))
    w6 = bool(bounded_payload.get("w6_byte_packed_accumulator_persisted"))
    dense = bool(bounded_payload.get("dense_int16_accumulator_persisted"))
    if sum(int(flag) for flag in (w5, w6, dense)) > 1:
        raise ValueError(
            "dual persistent accumulator encoding: at most one of W5/W6/dense int16 may be saved"
        )


def assert_no_raw_int8_q_dual_persistence(
    payload: Mapping[str, Any],
    *,
    q_packed_enabled: bool | None = None,
    q_codec_selector: str | None = None,
) -> None:
    """Fail-closed: byte-packed q checkpoints must not retain raw int8 q_levels."""

    authorized = persistent_q_ternary_byte_packed_enabled(enabled=q_packed_enabled)
    selector = resolve_q_codec_selector(q_codec_selector=q_codec_selector)
    if selector == Q_CODEC_SELECTOR_BASE3 and not authorized:
        raise ValueError(
            "base-3 q codec selector requires "
            "HRM_TEXT_158_PERSISTENT_Q_TERNARY_BYTE_PACKED=1 before checkpoint save"
        )
    sidecar = payload.get("trainer_sub2_authority")
    if not isinstance(sidecar, Mapping):
        return
    for module_key, module_payload in sorted((sidecar.get("tensor_payloads") or {}).items()):
        if not isinstance(module_payload, Mapping):
            continue
        q_saved = bool(module_payload.get("q_ternary_byte_packed_persisted"))
        if q_saved and RAW_Q_LEVELS_FIELD in module_payload:
            raise ValueError(
                f"raw int8 q_levels dual-persistence in checkpoint payload at {module_key}"
            )
        if q_saved and not authorized:
            raise ValueError(
                f"q-pack payload present without authorization at {module_key}"
            )


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
