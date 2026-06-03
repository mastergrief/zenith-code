"""Persistent-state q-pack and 3-ledger budget accounting tests."""
from __future__ import annotations

from dataclasses import replace
import os

import pytest
import torch

from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    EFFECTIVE_FORWARD_TERNARY_BITS,
    INCLUSIVE_3LEDGER_TARGET_BASIS,
    PACKED_TERNARY_METADATA_BYTES_PER_DIM,
    PACKED_TERNARY_METADATA_HEADER_BYTES,
    PERSISTENT_STATE_BUDGET_LABEL,
    PHYSICAL_SUB2_NOT_ACHIEVED_STATEMENT,
    RUN_GPU_PERSISTENT_STATE_BUDGET_ENV,
    measure_persistent_state_budget,
    pack_ternary_q_2bit_reference,
    unpack_ternary_q_2bit_reference,
    validate_persistent_state_budget_report,
)
from calm.hrm_text_158.native_full_stack.qscale_linear import (
    QScaleWeightState,
    qscale_linear_reference,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    apply_integer_vote_update_reference,
)


GPU_PERSISTENT_STATE_BUDGET = pytest.mark.skipif(
    os.environ.get(RUN_GPU_PERSISTENT_STATE_BUDGET_ENV) != "1" or not torch.cuda.is_available(),
    reason=(
        "persistent-state q-pack CUDA receipt deferred; set "
        f"{RUN_GPU_PERSISTENT_STATE_BUDGET_ENV}=1 only inside a granted gpu:0 lane"
    ),
)


def _numel(shape: tuple[int, ...]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def _make_q(shape: tuple[int, ...], *, device: str = "cpu") -> torch.Tensor:
    levels = torch.tensor([-1, 0, 1], dtype=torch.int8, device=device)
    idx = torch.arange(_numel(shape), dtype=torch.long, device=device) % 3
    return levels[idx].view(shape).contiguous()


def _qscale_state(shape: tuple[int, int] = (8, 16), *, device: str = "cpu") -> QScaleWeightState:
    return QScaleWeightState(
        q_levels=_make_q(shape, device=device),
        scale=torch.tensor(0.125, dtype=torch.float32, device=device),
    )


def _spec(**kwargs) -> VoteUpdateSpec:
    base = dict(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=1,
        max_abs_per_tensor=64,
        fraction_per_tensor=1.0,
    )
    base.update(kwargs)
    return VoteUpdateSpec(**base)


def test_pack_unpack_roundtrip_counts_actual_padding_and_metadata_bytes():
    q = torch.tensor([-1, 0, 1, 1, 0], dtype=torch.int8)

    packed = pack_ternary_q_2bit_reference(q)
    restored = unpack_ternary_q_2bit_reference(packed)

    torch.testing.assert_close(restored, q, atol=0, rtol=0)
    assert packed.packed.dtype == torch.uint8
    assert packed.packed.numel() == 2
    assert packed.logical_shape == (5,)
    assert packed.logical_numel == 5
    assert packed.padding_values == 3
    assert packed.packed_data_bits == 16
    assert packed.ideal_ternary_2bit_bits == 10
    assert packed.padding_bits == 6
    assert packed.metadata_bytes == PACKED_TERNARY_METADATA_HEADER_BYTES + PACKED_TERNARY_METADATA_BYTES_PER_DIM


def test_unused_active_code_is_rejected_but_padding_lanes_do_not_leak():
    q = torch.tensor([-1, 0, 1, 1, 0], dtype=torch.int8)
    packed = pack_ternary_q_2bit_reference(q)

    bad_active_payload = packed.packed.clone()
    bad_active_payload[0] = int(bad_active_payload[0].item() & 0b11111100) | 0b00000011
    with pytest.raises(ValueError, match="unused code"):
        unpack_ternary_q_2bit_reference(replace(packed, packed=bad_active_payload))

    padding_payload = packed.packed.clone()
    padding_payload[1] = int(padding_payload[1].item() & 0b00000011) | 0b11111100
    restored = unpack_ternary_q_2bit_reference(replace(packed, packed=padding_payload))
    torch.testing.assert_close(restored, q, atol=0, rtol=0)


def test_pack_rejects_non_ternary_or_non_int8_q_levels():
    with pytest.raises(ValueError, match="ternary int8"):
        pack_ternary_q_2bit_reference(torch.tensor([-1, 0, 2], dtype=torch.int8))
    with pytest.raises(ValueError, match="torch.int8"):
        pack_ternary_q_2bit_reference(torch.tensor([-1, 0, 1], dtype=torch.int16))


def test_inclusive_three_ledger_budget_reports_over_target_and_track_b_math_only():
    state = _qscale_state((8, 16))
    accumulators = torch.zeros_like(state.q_levels, dtype=torch.int16)

    report = measure_persistent_state_budget([state], [accumulators])

    validate_persistent_state_budget_report(report)
    assert report.schema_version.endswith("q_pack_reference")
    assert report.label == PERSISTENT_STATE_BUDGET_LABEL
    assert report.target_basis == INCLUSIVE_3LEDGER_TARGET_BASIS
    assert report.target_achieved is False
    assert report.receipt_statement == PHYSICAL_SUB2_NOT_ACHIEVED_STATEMENT
    assert report.eligible_weight_count == 128
    assert report.q_int8_bits_per_weight == pytest.approx(8.0)
    assert report.q_packed_data_bits_per_weight == pytest.approx(2.0)
    assert report.q_packed_padding_bits_per_weight == pytest.approx(0.0)
    assert report.q_packed_metadata_bits_per_weight == pytest.approx(2.0)
    assert report.q_packed_total_bits_per_weight == pytest.approx(4.0)
    assert report.acc_int16_bits_per_weight == pytest.approx(16.0)
    assert report.frozen_scale_fp32_bits_per_weight == pytest.approx(0.25)
    assert report.packed_inclusive_physical_bits_per_weight == pytest.approx(20.25)
    assert report.scale_excluded_diagnostic_bits_per_weight == pytest.approx(20.0)
    assert report.acc_scale_excluded_diagnostic_bits_per_weight == pytest.approx(4.0)
    assert report.q_effective_forward_entropy_bits_per_weight == pytest.approx(EFFECTIVE_FORWARD_TERNARY_BITS)
    assert report.required_acc_bits_per_weight_for_sub2_physical_q_with_scale_and_metadata < 0.0
    assert report.required_acc_bits_per_weight_for_sub2_effective_q_with_scale == pytest.approx(
        2.0 - EFFECTIVE_FORWARD_TERNARY_BITS - 0.25,
    )
    assert report.dominant_ledger == "acc_int16"
    assert report.track_b_status == "design_math_only_no_accumulator_compression_implemented"


def test_honesty_guard_rejects_q_only_or_diagnostic_sub2_target_claims():
    state = _qscale_state((8, 16))
    accumulators = torch.zeros_like(state.q_levels, dtype=torch.int16)
    report = measure_persistent_state_budget([state], [accumulators])

    with pytest.raises(ValueError, match="inclusive 3-ledger"):
        validate_persistent_state_budget_report(
            replace(report, target_basis="q_packed_data_only_diagnostic", target_achieved=True),
        )
    with pytest.raises(ValueError, match="inclusive 3-ledger"):
        validate_persistent_state_budget_report(
            replace(report, target_basis="scale_acc_excluded_diagnostic", target_achieved=True),
        )
    with pytest.raises(ValueError, match="inclusive 3-ledger physical bits/weight"):
        validate_persistent_state_budget_report(replace(report, target_achieved=True))


def test_unpack_qscale_forward_matches_original_reference_exactly():
    state = _qscale_state((4, 7))
    packed = pack_ternary_q_2bit_reference(state.q_levels)
    unpacked_state = QScaleWeightState(
        q_levels=unpack_ternary_q_2bit_reference(packed),
        scale=state.scale,
    )
    x = (torch.arange(14, dtype=torch.float32).view(2, 7) - 3.0) / 11.0

    expected = qscale_linear_reference(x, state)
    actual = qscale_linear_reference(x, unpacked_state)

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_unpack_apply_repack_unpacked_matches_reference_for_non_multiple_of_four_q():
    q = torch.tensor([-1, 0, 0, 0, 1], dtype=torch.int8)
    acc = torch.zeros(5, dtype=torch.int16)
    votes = torch.tensor([0, 12, -12, 12, 0], dtype=torch.int16)
    spec = _spec()

    reference = apply_integer_vote_update_reference(
        VoteUpdateState(q_levels=q, accumulators=acc),
        VoteUpdateInputs(votes=votes),
        spec,
    )
    unpacked_q = unpack_ternary_q_2bit_reference(pack_ternary_q_2bit_reference(q))
    packed_path = apply_integer_vote_update_reference(
        VoteUpdateState(q_levels=unpacked_q, accumulators=acc.clone()),
        VoteUpdateInputs(votes=votes.clone()),
        spec,
    )
    repacked_q = pack_ternary_q_2bit_reference(packed_path.q_levels)
    unpacked_after_apply = unpack_ternary_q_2bit_reference(repacked_q)

    torch.testing.assert_close(unpacked_after_apply, reference.q_levels, atol=0, rtol=0)
    torch.testing.assert_close(packed_path.accumulators, reference.accumulators, atol=0, rtol=0)
    assert repacked_q.padding_values == 3


@GPU_PERSISTENT_STATE_BUDGET
def test_cuda_roundtrip_forward_apply_and_budget_receipt_scaffold():
    q2d = _make_q((3, 5), device="cuda")
    packed = pack_ternary_q_2bit_reference(q2d)
    unpacked = unpack_ternary_q_2bit_reference(packed)
    torch.testing.assert_close(unpacked, q2d, atol=0, rtol=0)
    assert unpacked.device.type == "cuda"

    scale = torch.tensor(0.125, dtype=torch.float32, device="cuda")
    x = (torch.arange(10, dtype=torch.float32, device="cuda").view(2, 5) - 4.0) / 17.0
    original_state = QScaleWeightState(q_levels=q2d, scale=scale)
    unpacked_state = QScaleWeightState(q_levels=unpacked, scale=scale)
    torch.testing.assert_close(
        qscale_linear_reference(x, unpacked_state),
        qscale_linear_reference(x, original_state),
        atol=0.0,
        rtol=0.0,
    )

    q1d = _make_q((5,), device="cuda")
    acc = torch.zeros(5, dtype=torch.int16, device="cuda")
    votes = torch.tensor([0, 12, -12, 12, 0], dtype=torch.int16, device="cuda")
    reference = apply_integer_vote_update_reference(
        VoteUpdateState(q_levels=q1d, accumulators=acc),
        VoteUpdateInputs(votes=votes),
        _spec(),
    )
    packed_path = apply_integer_vote_update_reference(
        VoteUpdateState(
            q_levels=unpack_ternary_q_2bit_reference(pack_ternary_q_2bit_reference(q1d)),
            accumulators=acc.clone(),
        ),
        VoteUpdateInputs(votes=votes.clone()),
        _spec(),
    )
    repacked = pack_ternary_q_2bit_reference(packed_path.q_levels)
    torch.testing.assert_close(
        unpack_ternary_q_2bit_reference(repacked),
        reference.q_levels,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(packed_path.accumulators, reference.accumulators, atol=0, rtol=0)

    budget_state = _qscale_state((128, 128), device="cuda")
    budget_acc = torch.zeros_like(budget_state.q_levels, dtype=torch.int16)
    torch.cuda.reset_peak_memory_stats()
    warm = unpack_ternary_q_2bit_reference(pack_ternary_q_2bit_reference(budget_state.q_levels))
    torch.cuda.synchronize()
    torch.testing.assert_close(warm, budget_state.q_levels, atol=0, rtol=0)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        restored = unpack_ternary_q_2bit_reference(pack_ternary_q_2bit_reference(budget_state.q_levels))
    end.record()
    torch.cuda.synchronize()
    cuda_ms = start.elapsed_time(end) / 20.0
    torch.testing.assert_close(restored, budget_state.q_levels, atol=0, rtol=0)

    report = measure_persistent_state_budget([budget_state], [budget_acc])
    assert report.target_achieved is False
    assert report.receipt_statement == PHYSICAL_SUB2_NOT_ACHIEVED_STATEMENT
    assert report.required_acc_bits_per_weight_for_sub2_physical_q_with_scale_and_metadata < 0.0
    print(
        "persistent_state_q_pack_reference_3ledger_accounting_over_target_receipt_scaffold "
        f"eligible={report.eligible_weight_count} "
        f"q_data_bpw={report.q_packed_data_bits_per_weight:.6f} "
        f"q_metadata_bpw={report.q_packed_metadata_bits_per_weight:.6f} "
        f"acc_bpw={report.acc_int16_bits_per_weight:.6f} "
        f"scale_bpw={report.frozen_scale_fp32_bits_per_weight:.6f} "
        f"inclusive_bpw={report.packed_inclusive_physical_bits_per_weight:.6f} "
        "target_achieved=False "
        f"required_acc_physical_bpw={report.required_acc_bits_per_weight_for_sub2_physical_q_with_scale_and_metadata:.6f} "
        f"cuda_ms={cuda_ms:.4f} "
        f"peak_allocated={torch.cuda.max_memory_allocated()} "
        f"peak_reserved={torch.cuda.max_memory_reserved()} "
        f"statement={report.receipt_statement!r}"
    )
