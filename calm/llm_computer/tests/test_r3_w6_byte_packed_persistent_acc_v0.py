"""R3.0 fail-closed proofs for real uint8 W6 accumulator byte packing."""
from __future__ import annotations

import copy
import json
import math
import os

import pytest
import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    decode_bounded_accumulator_to_i16,
)
from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    NarrowCarrierHeadroomBreach,
    PackedW6AccumulatorPayload,
    W6_BYTE_PACKED_SCHEMA,
    W6_WIDTH_BITS,
    pack_w6_lanes_to_bytes,
    pack_w6_tensor,
    reject_int16_tensor_as_packed_acc,
    unpack_w6_lanes_from_bytes,
)
from calm.hrm_text_158.native_full_stack.narrow_carrier_trainer_integration import (
    assert_no_packed_w6_state_leak,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    measure_r3_persistent_state_budget,
    reject_int16_tensors_for_r3_ledger,
)
from calm.hrm_text_158.native_full_stack.qscale_linear import QScaleWeightState
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV,
    W6_BYTE_PACKED_ACCUMULATOR_PERSISTED_KEY,
    W6_BYTE_PACKED_PAYLOAD_KEY,
    _roundtrip_payload_sha256,
    build_trainer_sub2_authority_checkpoint_blob,
    derive_trainer_sub2_authority_states,
    load_trainer_sub2_authority_checkpoint_blob,
    persistent_w6_byte_packed_enabled,
    select_trainer_eligible_bitlinears,
)


class _TinyTernary(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = BitLinear(16, 16, bias=False)
        self.tail = torch.nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.proj(x))


def _in_domain_tensor(n: int) -> torch.Tensor:
    values = [((index % 63) - 31) for index in range(n)]
    return torch.tensor(values, dtype=torch.int16).reshape(n)


def test_pack_produces_uint8_payload_smaller_than_int16_for_n_ge_64() -> None:
    for n in (64, 129, 256):
        acc = _in_domain_tensor(n)
        payload = pack_w6_lanes_to_bytes(acc)
        assert payload.packed.dtype == torch.uint8
        assert payload.packed_data_bytes == math.ceil(n * W6_WIDTH_BITS / 8)
        assert payload.packed_data_bytes < n * 2


def test_roundtrip_int16_in_step_parity_including_edges() -> None:
    edge = torch.tensor([-31, 0, 31], dtype=torch.int16)
    for acc in (_in_domain_tensor(77), edge):
        payload = pack_w6_lanes_to_bytes(acc)
        roundtrip = unpack_w6_lanes_from_bytes(payload)
        assert torch.equal(roundtrip, acc.contiguous())


def test_rejects_fake_int16_container_as_packed_payload() -> None:
    acc = _in_domain_tensor(8)
    fake = pack_w6_tensor(acc)
    with pytest.raises(ValueError, match="not torch.int16"):
        reject_int16_tensor_as_packed_acc(fake)
    with pytest.raises(ValueError, match="not torch.int16"):
        PackedW6AccumulatorPayload(
            packed=fake,
            logical_shape=(8,),
            logical_numel=8,
        )


def test_rejects_out_of_domain_w6_values() -> None:
    with pytest.raises(NarrowCarrierHeadroomBreach):
        pack_w6_lanes_to_bytes(torch.tensor([32], dtype=torch.int16))


def test_rejects_missing_metadata_and_schema_mismatch() -> None:
    acc = _in_domain_tensor(8)
    payload = pack_w6_lanes_to_bytes(acc)
    with pytest.raises(ValueError, match="schema"):
        PackedW6AccumulatorPayload(
            packed=payload.packed,
            logical_shape=payload.logical_shape,
            logical_numel=payload.logical_numel,
            schema="wrong_schema",
        )
    with pytest.raises(ValueError, match="packed byte length"):
        PackedW6AccumulatorPayload(
            packed=payload.packed[:-1],
            logical_shape=payload.logical_shape,
            logical_numel=payload.logical_numel,
        )

def test_ledger_rejects_int16_input_and_counts_real_bytes() -> None:
    acc = _in_domain_tensor(128).reshape(8, 16)
    payload = pack_w6_lanes_to_bytes(acc)
    qstate = QScaleWeightState(
        q_levels=torch.ones((8, 16), dtype=torch.int8),
        scale=torch.tensor(1.0, dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="not torch.int16"):
        reject_int16_tensors_for_r3_ledger([acc.reshape(-1)])
    report = measure_r3_persistent_state_budget([qstate], [payload])
    assert report.r3_acc_physical_bits_per_weight == pytest.approx(6.0, abs=0.25)
    assert report.r3_q_int8_bits_per_weight == pytest.approx(8.0)
    assert report.r3_acc_logical_lane_bits == pytest.approx(6.0)
    assert report.r3_actual_acc_payload_bytes == payload.packed_data_bytes
    assert report.r3_artifact_overhead_bytes >= 0
    assert report.r3_ledger_pass is True


def test_checkpoint_roundtrip_byte_packed_flag_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _TinyTernary()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    monkeypatch.setenv(PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV, "1")
    assert persistent_w6_byte_packed_enabled()
    blob = build_trainer_sub2_authority_checkpoint_blob(
        model,
        eligible_modules=eligible,
        tensor_states=states,
        byte_packed_enabled=True,
    )
    assert blob["trainer_sub2_authority"]["w6_byte_packed_persistent_accumulator_saved"] is True
    payload = blob["trainer_sub2_authority"]["tensor_payloads"]["proj"]["bounded_accumulator"]
    assert payload[W6_BYTE_PACKED_ACCUMULATOR_PERSISTED_KEY] is True
    assert payload[W6_BYTE_PACKED_PAYLOAD_KEY].dtype == torch.uint8
    fresh = _TinyTernary()
    fresh_eligible = select_trainer_eligible_bitlinears(fresh, use_ternary_bulk=True)
    loaded = load_trainer_sub2_authority_checkpoint_blob(
        fresh,
        blob,
        eligible_modules=fresh_eligible,
        byte_packed_enabled=True,
    )
    for key in eligible:
        before = decode_bounded_accumulator_to_i16(states[key].bounded_accumulator)
        after = decode_bounded_accumulator_to_i16(loaded[key].bounded_accumulator)
        assert torch.equal(before, after)


def test_rejects_int16_byte_packed_sidecar_on_load_with_recomputed_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyTernary()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    monkeypatch.setenv(PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV, "1")
    blob = build_trainer_sub2_authority_checkpoint_blob(
        model,
        eligible_modules=eligible,
        tensor_states=states,
        byte_packed_enabled=True,
    )
    bad = copy.deepcopy(blob)
    module_key = next(iter(eligible))
    bounded = bad["trainer_sub2_authority"]["tensor_payloads"][module_key]["bounded_accumulator"]
    uint8_payload = bounded[W6_BYTE_PACKED_PAYLOAD_KEY]
    bounded[W6_BYTE_PACKED_PAYLOAD_KEY] = uint8_payload.to(torch.int16)
    sidecar = bad["trainer_sub2_authority"]
    sidecar_without_hash = dict(sidecar)
    sidecar_without_hash.pop("authoritative_state_payload_sha256", None)
    sidecar["authoritative_state_payload_sha256"] = _roundtrip_payload_sha256(
        sidecar_without_hash
    )
    fresh = _TinyTernary()
    fresh_eligible = select_trainer_eligible_bitlinears(fresh, use_ternary_bulk=True)
    with pytest.raises(ValueError, match="not torch.int16"):
        load_trainer_sub2_authority_checkpoint_blob(
            fresh,
            bad,
            eligible_modules=fresh_eligible,
            byte_packed_enabled=True,
        )


def test_rejects_flag_off_with_byte_packed_sidecar(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _TinyTernary()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    monkeypatch.setenv(PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV, "1")
    blob = build_trainer_sub2_authority_checkpoint_blob(
        model,
        eligible_modules=eligible,
        tensor_states=states,
        byte_packed_enabled=True,
    )
    monkeypatch.delenv(PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV, raising=False)
    fresh = _TinyTernary()
    fresh_eligible = select_trainer_eligible_bitlinears(fresh, use_ternary_bulk=True)
    with pytest.raises(ValueError, match="byte-packed W6"):
        load_trainer_sub2_authority_checkpoint_blob(
            fresh,
            blob,
            eligible_modules=fresh_eligible,
            byte_packed_enabled=False,
        )


def test_rejects_dense_int16_sidecar_fallback() -> None:
    model = _TinyTernary()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    blob = build_trainer_sub2_authority_checkpoint_blob(
        model,
        eligible_modules=eligible,
        tensor_states=states,
    )
    bad = copy.deepcopy(blob)
    bad["trainer_sub2_authority"]["dense_int16_persistent_accumulator_saved"] = True
    fresh = _TinyTernary()
    fresh_eligible = select_trainer_eligible_bitlinears(fresh, use_ternary_bulk=True)
    with pytest.raises(ValueError, match="dense int16 persistent accumulators"):
        load_trainer_sub2_authority_checkpoint_blob(
            fresh,
            bad,
            eligible_modules=fresh_eligible,
        )


def test_authorized_packed_fields_allowed_only_when_flag_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "w6_byte_packed_accumulator_persisted": True,
        "w6_byte_packed_payload": torch.zeros(4, dtype=torch.uint8),
    }
    with pytest.raises(ValueError, match="packed W6 state leak"):
        assert_no_packed_w6_state_leak(payload, byte_packed_enabled=False)
    monkeypatch.setenv(PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV, "1")
    assert_no_packed_w6_state_leak(payload, byte_packed_enabled=True)


def test_lane_domain_pack_w6_tensor_unchanged() -> None:
    acc = torch.tensor([5, -3, 31], dtype=torch.int16)
    packed = pack_w6_tensor(acc)
    assert packed.dtype == torch.int16
    assert packed.tolist() == [5, 61, 31]
