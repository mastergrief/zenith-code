"""R4b base-3 q checkpoint codec proofs (CPU-only, in-memory synthetic tensors)."""
from __future__ import annotations

import copy
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
    PACKED_BASE3_TERNARY_Q_FORMAT,
    PACKED_TERNARY_Q_FORMAT,
    R4B_Q_SCALE_INCLUSIVE_BPW_CEILING,
    measure_r4b_persistent_state_budget,
)
from calm.hrm_text_158.native_full_stack.q_entropy_packing import (
    BASE3_Q_FORMAT,
    pack_ternary_q_base3_5perbyte_reference,
)
from calm.hrm_text_158.native_full_stack.qscale_linear import QScaleWeightState
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV,
    PERSISTENT_Q_TERNARY_BASE3_CODEC_ENV,
    PERSISTENT_Q_TERNARY_BYTE_PACKED_ENV,
    Q_CODEC_SELECTOR_BASE3,
    Q_TERNARY_BYTE_PACKED_PERSISTED_KEY,
    Q_TERNARY_PACKED_PAYLOAD_KEY,
    Q_TERNARY_PACKED_SCHEMA_KEY,
    _pack_q_for_checkpoint,
    _tensor_state_roundtrip_payload,
    _roundtrip_payload_sha256,
    build_trainer_sub2_authority_checkpoint_blob,
    derive_trainer_sub2_authority_states,
    load_trainer_sub2_authority_checkpoint_blob,
    packed_q_state_from_roundtrip_q_payload,
    persistent_q_ternary_base3_codec_enabled,
    select_trainer_eligible_bitlinears,
    tensor_sha256,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    build_r4b_persistent_ledger_receipt,
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


def test_t2_base3_roundtrip_preserves_q_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _TinyTernary()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    monkeypatch.setenv(PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV, "1")
    monkeypatch.setenv(PERSISTENT_Q_TERNARY_BYTE_PACKED_ENV, "1")
    monkeypatch.setenv(PERSISTENT_Q_TERNARY_BASE3_CODEC_ENV, "1")
    assert persistent_q_ternary_base3_codec_enabled()
    blob = build_trainer_sub2_authority_checkpoint_blob(
        model,
        eligible_modules=eligible,
        tensor_states=states,
        byte_packed_enabled=True,
        q_packed_enabled=True,
        q_codec_selector=Q_CODEC_SELECTOR_BASE3,
    )
    module_payload = blob["trainer_sub2_authority"]["tensor_payloads"]["proj"]
    assert module_payload[Q_TERNARY_PACKED_SCHEMA_KEY] == BASE3_Q_FORMAT
    assert module_payload[Q_TERNARY_BYTE_PACKED_PERSISTED_KEY] is True
    assert "q_levels" not in module_payload
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
        assert tensor_sha256(states[key].q_levels) == tensor_sha256(loaded[key].q_levels)
        assert torch.equal(states[key].q_levels, loaded[key].q_levels)


def test_t3_unknown_format_tag_rejected_on_load(monkeypatch: pytest.MonkeyPatch) -> None:
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
    module_payload[Q_TERNARY_PACKED_SCHEMA_KEY] = "packed_unknown_codec/v0"
    sidecar = bad["trainer_sub2_authority"]
    sidecar_without_hash = dict(sidecar)
    sidecar_without_hash.pop("authoritative_state_payload_sha256", None)
    sidecar["authoritative_state_payload_sha256"] = _roundtrip_payload_sha256(
        sidecar_without_hash
    )
    fresh = _TinyTernary()
    fresh_eligible = select_trainer_eligible_bitlinears(fresh, use_ternary_bulk=True)
    with pytest.raises(ValueError, match="unknown format tag"):
        load_trainer_sub2_authority_checkpoint_blob(
            fresh,
            bad,
            eligible_modules=fresh_eligible,
            byte_packed_enabled=True,
            q_packed_enabled=True,
        )


def test_t3_missing_format_tag_rejected_on_load(monkeypatch: pytest.MonkeyPatch) -> None:
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
    module_payload.pop(Q_TERNARY_PACKED_SCHEMA_KEY, None)
    sidecar = bad["trainer_sub2_authority"]
    sidecar_without_hash = dict(sidecar)
    sidecar_without_hash.pop("authoritative_state_payload_sha256", None)
    sidecar["authoritative_state_payload_sha256"] = _roundtrip_payload_sha256(
        sidecar_without_hash
    )
    fresh = _TinyTernary()
    fresh_eligible = select_trainer_eligible_bitlinears(fresh, use_ternary_bulk=True)
    with pytest.raises(ValueError, match="missing q packed format tag"):
        load_trainer_sub2_authority_checkpoint_blob(
            fresh,
            bad,
            eligible_modules=fresh_eligible,
            byte_packed_enabled=True,
            q_packed_enabled=True,
        )


def test_t5_base3_saved_byte_ledger_strictly_below_two_bpw() -> None:
    q = _large_q_for_inclusive_gate()
    acc = torch.zeros_like(q, dtype=torch.int16)
    packed_q = pack_ternary_q_base3_5perbyte_reference(q)
    packed_acc = pack_w6_lanes_to_bytes(acc)
    qstate = QScaleWeightState(
        q_levels=q,
        scale=torch.tensor(1.0, dtype=torch.float32),
    )
    report = measure_r4b_persistent_state_budget(
        [qstate],
        [packed_q],
        [packed_acc],
        state_keys=["proj"],
    )
    assert report.r4b_q_physical_bits_per_weight == pytest.approx(1.6, abs=0.01)
    assert report.r4b_q_scale_inclusive_physical_bits_per_weight < R4B_Q_SCALE_INCLUSIVE_BPW_CEILING
    assert report.r4b_actual_q_payload_bytes == packed_q.packed_data_bytes
    assert report.r4b_ledger_pass is True


def test_t5_metadata_inclusive_ceiling_enforced_on_small_shape() -> None:
    q = _make_q((16, 16))
    acc = torch.zeros_like(q, dtype=torch.int16)
    packed_q = pack_ternary_q_base3_5perbyte_reference(q)
    packed_acc = pack_w6_lanes_to_bytes(acc)
    qstate = QScaleWeightState(
        q_levels=q,
        scale=torch.tensor(1.0, dtype=torch.float32),
    )
    report = measure_r4b_persistent_state_budget(
        [qstate],
        [packed_q],
        [packed_acc],
        state_keys=["proj"],
    )
    q_without_metadata = (
        report.r4b_q_physical_bits_per_weight
        + report.r4b_frozen_scale_fp32_bits / report.eligible_weight_count
    )
    assert report.r4b_q_metadata_bits_per_weight > 0.0
    assert report.r4b_q_scale_inclusive_physical_bits_per_weight > q_without_metadata
    assert report.r4b_q_scale_inclusive_physical_bits_per_weight >= R4B_Q_SCALE_INCLUSIVE_BPW_CEILING
    assert report.r4b_ledger_pass is False


def test_t7_mixed_format_checkpoint_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TwoHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj_a = BitLinear(8, 8, bias=False)
            self.proj_b = BitLinear(8, 8, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj_b(self.proj_a(x))

    model = _TwoHead()
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
        q_codec_selector="2bit",
    )
    sidecar = blob["trainer_sub2_authority"]
    payloads = sidecar["tensor_payloads"]
    second_key = sorted(payloads)[1]
    q_levels = states[second_key].q_levels.detach().cpu().to(torch.int8).contiguous()
    payloads[second_key].update(
        _pack_q_for_checkpoint(
            q_levels,
            q_packed_enabled=True,
            q_codec_selector=Q_CODEC_SELECTOR_BASE3,
        )
    )
    sidecar_without_hash = dict(sidecar)
    sidecar_without_hash.pop("authoritative_state_payload_sha256", None)
    sidecar["authoritative_state_payload_sha256"] = _roundtrip_payload_sha256(
        sidecar_without_hash
    )
    assert payloads[sorted(payloads)[0]][Q_TERNARY_PACKED_SCHEMA_KEY] == PACKED_TERNARY_Q_FORMAT
    assert payloads[second_key][Q_TERNARY_PACKED_SCHEMA_KEY] == PACKED_BASE3_TERNARY_Q_FORMAT
    fresh = _TwoHead()
    fresh_eligible = select_trainer_eligible_bitlinears(fresh, use_ternary_bulk=True)
    with pytest.raises(ValueError, match="mixed q packed formats"):
        load_trainer_sub2_authority_checkpoint_blob(
            fresh,
            blob,
            eligible_modules=fresh_eligible,
            byte_packed_enabled=True,
            q_packed_enabled=True,
        )


def test_t7_uniform_base3_checkpoint_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TwoHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj_a = BitLinear(8, 8, bias=False)
            self.proj_b = BitLinear(8, 8, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj_b(self.proj_a(x))

    model = _TwoHead()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    monkeypatch.setenv(PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV, "1")
    monkeypatch.setenv(PERSISTENT_Q_TERNARY_BYTE_PACKED_ENV, "1")
    monkeypatch.setenv(PERSISTENT_Q_TERNARY_BASE3_CODEC_ENV, "1")
    blob = build_trainer_sub2_authority_checkpoint_blob(
        model,
        eligible_modules=eligible,
        tensor_states=states,
        byte_packed_enabled=True,
        q_packed_enabled=True,
        q_codec_selector=Q_CODEC_SELECTOR_BASE3,
    )
    payloads = blob["trainer_sub2_authority"]["tensor_payloads"]
    for module_payload in payloads.values():
        assert module_payload[Q_TERNARY_PACKED_SCHEMA_KEY] == BASE3_Q_FORMAT
    fresh = _TwoHead()
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


def test_t8_byte_length_validation_rejects_wrong_payload_size() -> None:
    q = _make_q((16, 16))
    packed_q = pack_ternary_q_base3_5perbyte_reference(q)
    bad_payload = copy.deepcopy(packed_q)
    bad_payload = type(packed_q)(
        packed=torch.cat((packed_q.packed, packed_q.packed[:1])),
        logical_shape=packed_q.logical_shape,
        logical_numel=packed_q.logical_numel,
        padding_values=packed_q.padding_values,
        format=packed_q.format,
    )
    acc = torch.zeros_like(q, dtype=torch.int16)
    qstate = QScaleWeightState(q_levels=q, scale=torch.tensor(1.0, dtype=torch.float32))
    with pytest.raises(ValueError, match="packed byte length"):
        measure_r4b_persistent_state_budget(
            [qstate],
            [bad_payload],
            [pack_w6_lanes_to_bytes(acc)],
            state_keys=["proj"],
        )


def test_t10_base3_selector_without_master_q_pack_fails_before_save() -> None:
    q = _make_q((8,))
    with pytest.raises(ValueError, match="base-3 q codec selector requires"):
        _pack_q_for_checkpoint(
            q,
            q_packed_enabled=False,
            q_codec_selector=Q_CODEC_SELECTOR_BASE3,
        )


def test_t10_assert_no_dual_persistence_rejects_base3_without_master() -> None:
    payload = {
        "trainer_sub2_authority": {
            "tensor_payloads": {
                "proj": {
                    Q_TERNARY_BYTE_PACKED_PERSISTED_KEY: True,
                    Q_TERNARY_PACKED_SCHEMA_KEY: PACKED_BASE3_TERNARY_Q_FORMAT,
                    Q_TERNARY_PACKED_PAYLOAD_KEY: torch.zeros(4, dtype=torch.uint8),
                }
            }
        }
    }
    with pytest.raises(ValueError, match="base-3 q codec selector requires"):
        assert_no_raw_int8_q_dual_persistence(
            payload,
            q_packed_enabled=False,
            q_codec_selector=Q_CODEC_SELECTOR_BASE3,
        )


def test_t9_r4b_receipt_uses_saved_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setenv(PERSISTENT_Q_TERNARY_BASE3_CODEC_ENV, "1")
    roundtrip_payload = _tensor_state_roundtrip_payload(
        state,
        byte_packed_enabled=True,
        q_packed_enabled=True,
        q_codec_selector=Q_CODEC_SELECTOR_BASE3,
    )
    saved_packed_q = packed_q_state_from_roundtrip_q_payload(roundtrip_payload)
    ledger = build_r4b_persistent_ledger_receipt(
        {"proj": state},
        q_packed_enabled=True,
        acc_byte_packed_enabled=True,
        q_codec_selector=Q_CODEC_SELECTOR_BASE3,
    )
    assert ledger["enabled"] is True
    assert ledger["r4b_ledger_pass"] is True
    assert ledger["r4b_actual_q_payload_bytes"] == saved_packed_q.packed_data_bytes
    assert ledger["r4b_q_physical_bits_per_weight"] < 2.0

    mutated_packed_q = type(saved_packed_q)(
        packed=torch.cat((saved_packed_q.packed, saved_packed_q.packed[:1])),
        logical_shape=saved_packed_q.logical_shape,
        logical_numel=saved_packed_q.logical_numel,
        padding_values=saved_packed_q.padding_values,
        format=saved_packed_q.format,
    )
    with pytest.raises(ValueError, match="packed byte length"):
        measure_r4b_persistent_state_budget(
            [
                QScaleWeightState(
                    q_levels=q,
                    scale=torch.tensor(1.0, dtype=torch.float32),
                )
            ],
            [mutated_packed_q],
            [pack_w6_lanes_to_bytes(acc)],
            state_keys=["proj"],
        )
