"""Canonical trainer_sub2 authoritative sidecar payload sha (IMPLEMENT_v9).

RO re-use of TSA serializer. No TSA edits. Shared by twin_apply emission and
production_binding equality checks.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def eligible_weight_state_keys_from_state_keys(
    state_keys: Sequence[Any],
) -> tuple[str, ...]:
    """Mirror TSA `_eligible_weight_state_keys` for module-name state keys (RO)."""
    return tuple(f"{key}.weight" for key in sorted(str(item) for item in state_keys))


def authoritative_sidecar_payload_sha256(
    tensor_states: Mapping[str, Any],
    *,
    step: int,
    eligible_weight_state_keys: Sequence[str] | None = None,
) -> str:
    """Canonical trainer_sub2 authoritative sidecar sha — same serializer as TSA.

    RO imports of production `_tensor_state_roundtrip_payload` +
    `_roundtrip_payload_sha256`. Sidecar digest covers tensor_payloads + step +
    eligible keys/flags only (model_state is outside the sidecar digest).
    """
    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
        EVENT_CODED_LIVE_CARRIER_PERSISTED_KEY,
        EVENT_CODED_LIVE_CARRIER_SAVED_KEY,
        Q_TERNARY_BYTE_PACKED_PERSISTED_KEY,
        Q_TERNARY_BYTE_PACKED_PERSISTED_SAVED_KEY,
        TRAINER_SUB2_ROUNDTRIP_SCHEMA_VERSION,
        W5_BYTE_PACKED_ACCUMULATOR_PERSISTED_KEY,
        W6_BYTE_PACKED_ACCUMULATOR_PERSISTED_KEY,
        _roundtrip_payload_sha256,
        _tensor_state_roundtrip_payload,
    )

    keys = sorted(str(k) for k in tensor_states)
    tensor_payloads = {
        k: _tensor_state_roundtrip_payload(tensor_states[k]) for k in keys
    }
    w6 = any(
        bool(
            (payload.get("bounded_accumulator") or {}).get(
                W6_BYTE_PACKED_ACCUMULATOR_PERSISTED_KEY
            )
        )
        for payload in tensor_payloads.values()
    )
    w5 = any(
        bool(
            (payload.get("bounded_accumulator") or {}).get(
                W5_BYTE_PACKED_ACCUMULATOR_PERSISTED_KEY
            )
        )
        for payload in tensor_payloads.values()
    )
    q_packed = any(
        bool(payload.get(Q_TERNARY_BYTE_PACKED_PERSISTED_KEY))
        for payload in tensor_payloads.values()
    )
    event_coded = any(
        bool(payload.get(EVENT_CODED_LIVE_CARRIER_PERSISTED_KEY))
        for payload in tensor_payloads.values()
    )
    weight_keys = (
        tuple(str(x) for x in eligible_weight_state_keys)
        if eligible_weight_state_keys is not None
        else eligible_weight_state_keys_from_state_keys(keys)
    )
    sidecar = {
        "schema_version": TRAINER_SUB2_ROUNDTRIP_SCHEMA_VERSION,
        "artifact_role": "trainer_sub2_authoritative_sidecar",
        "step": int(step),
        "eligible_state_keys": tuple(keys),
        "eligible_weight_state_keys": weight_keys,
        "tensor_payloads": tensor_payloads,
        "eligible_fp_masters_authoritative": False,
        "dense_int16_persistent_accumulator_saved": False,
        "w6_byte_packed_persistent_accumulator_saved": bool(w6),
        "w5_byte_packed_persistent_accumulator_saved": bool(w5),
        EVENT_CODED_LIVE_CARRIER_SAVED_KEY: bool(event_coded),
        Q_TERNARY_BYTE_PACKED_PERSISTED_SAVED_KEY: bool(q_packed),
        "normal_bitlinear_weight_forward_not_claimed": True,
    }
    return str(_roundtrip_payload_sha256(sidecar))
