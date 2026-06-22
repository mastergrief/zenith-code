"""R4.0 fail-closed proofs for real uint8 ternary q byte packing."""
from __future__ import annotations

import copy
import hashlib
import math

import pytest
import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    decode_bounded_accumulator_to_i16,
)
from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    pack_w6_lanes_to_bytes,
)
from calm.hrm_text_158.native_full_stack.narrow_carrier_trainer_integration import (
    assert_no_raw_int8_q_dual_persistence,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    PACKED_TERNARY_Q_FORMAT,
    PackedTernaryQState,
    canonical_r4_q_packed_content_sha256,
    measure_r4_persistent_state_budget,
    pack_ternary_q_2bit_reference,
    reject_int8_tensors_for_r4_ledger,
    unpack_ternary_q_2bit_reference,
)
from calm.hrm_text_158.native_full_stack.qscale_linear import QScaleWeightState
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV,
    PERSISTENT_Q_TERNARY_BYTE_PACKED_ENV,
    Q_TERNARY_BYTE_PACKED_PERSISTED_KEY,
    Q_TERNARY_PACKED_PAYLOAD_KEY,
    _roundtrip_payload_sha256,
    build_trainer_sub2_authority_checkpoint_blob,
    derive_trainer_sub2_authority_states,
    load_trainer_sub2_authority_checkpoint_blob,
    persistent_q_ternary_byte_packed_enabled,
    select_trainer_eligible_bitlinears,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    build_r4_persistent_ledger_receipt,
)


class _TinyTernary(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = BitLinear(16, 16, bias=False)
        self.tail = torch.nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.proj(x))


def _make_q(shape: tuple[int, ...]) -> torch.Tensor:
    levels = torch.tensor([-1, 0, 1], dtype=torch.int8)
    idx = torch.arange(math.prod(shape), dtype=torch.long) % 3
    return levels[idx].view(shape).contiguous()


def _large_q_for_inclusive_gate() -> torch.Tensor:
    return _make_q((1024, 1024))


def test_pack_produces_uint8_payload_smaller_than_int8_for_n_ge_64() -> None:
    for shape in ((64,), (8, 16), (16, 16)):
        q = _make_q(shape)
        packed = pack_ternary_q_2bit_reference(q)
        assert packed.packed.dtype == torch.uint8
        assert packed.packed_data_bytes == math.ceil(int(q.numel()) / 4)
        assert packed.packed_data_bytes < int(q.numel())


def test_roundtrip_int8_in_step_parity_and_padding_determinism() -> None:
    q = torch.tensor([-1, 0, 1, 1, 0], dtype=torch.int8)
    packed_a = pack_ternary_q_2bit_reference(q)
    packed_b = pack_ternary_q_2bit_reference(q)
    restored = unpack_ternary_q_2bit_reference(packed_a)
    torch.testing.assert_close(restored, q, atol=0, rtol=0)
    assert packed_a.padding_values == 3
    assert torch.equal(packed_a.packed, packed_b.packed)
    assert canonical_r4_q_packed_content_sha256(
        [
            {
                "state_key": "k",
                "logical_shape": list(packed_a.logical_shape),
                "lanes": int(packed_a.logical_numel),
                "payload_bytes": int(packed_a.packed_data_bytes),
                "metadata_bytes": int(packed_a.metadata_bytes),
                "padding_values": int(packed_a.padding_values),
                "q_bpw": 2.0,
                "payload_sha256": hashlib.sha256(
                    packed_a.packed.detach().cpu().numpy().tobytes()
                ).hexdigest(),
            }
        ]
    ) == canonical_r4_q_packed_content_sha256(
        [
            {
                "state_key": "k",
                "logical_shape": list(packed_b.logical_shape),
                "lanes": int(packed_b.logical_numel),
                "payload_bytes": int(packed_b.packed_data_bytes),
                "metadata_bytes": int(packed_b.metadata_bytes),
                "padding_values": int(packed_b.padding_values),
                "q_bpw": 2.0,
                "payload_sha256": hashlib.sha256(
                    packed_b.packed.detach().cpu().numpy().tobytes()
                ).hexdigest(),
            }
        ]
    )


def test_rejects_fake_int8_container_as_packed_q_payload() -> None:
    fake_packed = _make_q((8,)).contiguous()
    with pytest.raises(ValueError, match="not torch.int8"):
        reject_int8_tensors_for_r4_ledger([fake_packed])
    with pytest.raises(ValueError, match="not torch.int8"):
        unpack_ternary_q_2bit_reference(
            PackedTernaryQState(
                packed=fake_packed,
                logical_shape=(8,),
                logical_numel=8,
                padding_values=0,
            )
        )


def test_ledger_rejects_int8_input_and_counts_real_bytes_at_real_numel() -> None:
    q = _large_q_for_inclusive_gate()
    acc = torch.zeros_like(q, dtype=torch.int16)
    packed_q = pack_ternary_q_2bit_reference(q)
    packed_acc = pack_w6_lanes_to_bytes(acc)
    qstate = QScaleWeightState(
        q_levels=q,
        scale=torch.tensor(1.0, dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="not torch.int8"):
        reject_int8_tensors_for_r4_ledger([q])
    report = measure_r4_persistent_state_budget(
        [qstate],
        [packed_q],
        [packed_acc],
        state_keys=["proj"],
    )
    assert report.eligible_weight_count == 1_048_576
    assert report.r4_q_physical_bits_per_weight == pytest.approx(2.0, abs=0.25)
    assert report.r4_acc_physical_bits_per_weight == pytest.approx(6.0, abs=0.25)
    assert report.r4_checkpoint_inclusive_physical_bits_per_weight <= 8.5
    assert report.r4_actual_q_payload_bytes == packed_q.packed_data_bytes
    assert report.r4_ledger_pass is True
    assert "still NOT sub-2" in report.receipt_statement
    assert report.r4_q_packed_content_sha256


def test_checkpoint_roundtrip_q_pack_flag_gated_with_seam_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyTernary()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    monkeypatch.setenv(PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV, "1")
    monkeypatch.setenv(PERSISTENT_Q_TERNARY_BYTE_PACKED_ENV, "1")
    assert persistent_q_ternary_byte_packed_enabled()
    blob = build_trainer_sub2_authority_checkpoint_blob(
        model,
        eligible_modules=eligible,
        tensor_states=states,
        byte_packed_enabled=True,
        q_packed_enabled=True,
    )
    module_payload = blob["trainer_sub2_authority"]["tensor_payloads"]["proj"]
    assert module_payload[Q_TERNARY_BYTE_PACKED_PERSISTED_KEY] is True
    assert "q_levels" not in module_payload
    assert module_payload[Q_TERNARY_PACKED_PAYLOAD_KEY].dtype == torch.uint8
    assert_no_raw_int8_q_dual_persistence(blob, q_packed_enabled=True)
    fresh = _TinyTernary()
    fresh_eligible = select_trainer_eligible_bitlinears(fresh, use_ternary_bulk=True)
    loaded = load_trainer_sub2_authority_checkpoint_blob(
        fresh,
        blob,
        eligible_modules=fresh_eligible,
        byte_packed_enabled=True,
        q_packed_enabled=True,
    )
    for key in eligible:
        assert torch.equal(states[key].q_levels, loaded[key].q_levels)
        before = decode_bounded_accumulator_to_i16(states[key].bounded_accumulator)
        after = decode_bounded_accumulator_to_i16(loaded[key].bounded_accumulator)
        assert torch.equal(before, after)


def test_rejects_int8_q_packed_sidecar_on_load_with_recomputed_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyTernary()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    monkeypatch.setenv(PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV, "1")
    monkeypatch.setenv(PERSISTENT_Q_TERNARY_BYTE_PACKED_ENV, "1")
    blob = build_trainer_sub2_authority_checkpoint_blob(
        model,
        eligible_modules=eligible,
        tensor_states=states,
        byte_packed_enabled=True,
        q_packed_enabled=True,
    )
    bad = copy.deepcopy(blob)
    module_payload = bad["trainer_sub2_authority"]["tensor_payloads"]["proj"]
    uint8_payload = module_payload[Q_TERNARY_PACKED_PAYLOAD_KEY]
    module_payload[Q_TERNARY_PACKED_PAYLOAD_KEY] = uint8_payload.to(torch.int16)
    sidecar = bad["trainer_sub2_authority"]
    sidecar_without_hash = dict(sidecar)
    sidecar_without_hash.pop("authoritative_state_payload_sha256", None)
    sidecar["authoritative_state_payload_sha256"] = _roundtrip_payload_sha256(
        sidecar_without_hash
    )
    fresh = _TinyTernary()
    fresh_eligible = select_trainer_eligible_bitlinears(fresh, use_ternary_bulk=True)
    with pytest.raises(ValueError, match="not torch.int8|element_size=2"):
        load_trainer_sub2_authority_checkpoint_blob(
            fresh,
            bad,
            eligible_modules=fresh_eligible,
            byte_packed_enabled=True,
            q_packed_enabled=True,
        )


def test_rejects_raw_q_dual_persistence_on_save_shape() -> None:
    payload = {
        "trainer_sub2_authority": {
            "tensor_payloads": {
                "proj": {
                    Q_TERNARY_BYTE_PACKED_PERSISTED_KEY: True,
                    "q_levels": torch.zeros(4, dtype=torch.int8),
                }
            }
        }
    }
    with pytest.raises(ValueError, match="dual-persistence"):
        assert_no_raw_int8_q_dual_persistence(payload, q_packed_enabled=True)


def test_build_r4_persistent_ledger_receipt_disabled_without_flags() -> None:
    q = _large_q_for_inclusive_gate()
    acc = torch.zeros_like(q, dtype=torch.int16)
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        BoundedDeltaAccumulatorState,
        BoundedDeltaTensorState,
    )

    bounded = BoundedDeltaAccumulatorState(
        logical_shape=tuple(q.shape),
        cold_default_value=0,
        hot_exact_indices=(),
        hot_exact_values=(),
        cold_exception_indices=(),
        cold_exception_values=(),
        candidate_name="cold_default",
        raw_arrays_included=False,
    )
    state = BoundedDeltaTensorState(
        state_key="proj",
        q_levels=q,
        frozen_scale=torch.tensor(1.0, dtype=torch.float32),
        bounded_accumulator=bounded,
        exact_accumulator_shadow=acc,
        bounded_accumulator_fresh_for_exact_shadow=False,
    )
    states = {"proj": state}
    assert build_r4_persistent_ledger_receipt(
        states,
        q_packed_enabled=False,
        acc_byte_packed_enabled=True,
    ) == {"enabled": False}
    ledger = build_r4_persistent_ledger_receipt(
        states,
        q_packed_enabled=True,
        acc_byte_packed_enabled=True,
    )
    assert ledger["enabled"] is True
    assert ledger["r4_ledger_pass"] is True
    assert ledger["r4_per_module_q_rows"][0]["lanes"] == 1_048_576
    assert "r3_q_int8_bits_per_weight" not in ledger
