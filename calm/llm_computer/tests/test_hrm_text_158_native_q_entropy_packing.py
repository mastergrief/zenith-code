"""C1.1a base-3 q-entropy packing/accounting tests."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from calm.hrm_text_158.native_full_stack.accumulator_compression import (
    CandidateClassification,
    required_decision_dimension_names,
)
from calm.hrm_text_158.native_full_stack.q_entropy_packing import (
    BASE3_FULL_GROUP_CODE_COUNT,
    BASE3_Q_ENTROPY_LABEL,
    BASE3_Q_FORMAT,
    BASE3_Q_STORAGE_ONLY_STATUS,
    base3_q_entropy_compact_report,
    base3_q_entropy_ledger_for_shapes,
    base3_q_storage_orthogonality_report,
    default_base3_q_entropy_ledger_table,
    measure_base3_q_entropy_budget,
    pack_ternary_q_base3_5perbyte_reference,
    unpack_ternary_q_base3_5perbyte_reference,
    validate_base3_q_entropy_ledger,
    validate_base3_q_storage_orthogonality,
)
from calm.hrm_text_158.native_full_stack.qscale_linear import (
    QScaleWeightFormat,
    QScaleWeightState,
    qscale_linear_reference,
    validate_qscale_weight_state,
)


def _pattern(length: int, name: str) -> torch.Tensor:
    if name == "all_neg":
        return torch.full((length,), -1, dtype=torch.int8)
    if name == "all_zero":
        return torch.zeros(length, dtype=torch.int8)
    if name == "all_pos":
        return torch.ones(length, dtype=torch.int8)
    if name == "mixed":
        levels = torch.tensor([-1, 0, 1], dtype=torch.int8)
        return levels[(torch.arange(length, dtype=torch.long) * 7 + 1) % 3].contiguous()
    raise AssertionError(name)


def _make_q(shape: tuple[int, int], *, seed: int = 17) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    levels = torch.tensor([-1, 0, 1], dtype=torch.int8)
    idx = torch.randint(0, 3, shape, generator=gen)
    return levels[idx].contiguous()


def _qscale_state(shape: tuple[int, int]) -> QScaleWeightState:
    return QScaleWeightState(
        q_levels=_make_q(shape, seed=sum(shape)),
        scale=torch.tensor(0.125, dtype=torch.float32),
        format=QScaleWeightFormat.INT8_LEVELS,
    )


def _rows_by_name():
    return {row.regime_name: row for row in default_base3_q_entropy_ledger_table()}


def _assert_no_tensors(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        raise AssertionError("compact report must not include raw tensors")
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_tensors(child)


@pytest.mark.parametrize("length", [0, 1, 4, 5, 6, 1027])
@pytest.mark.parametrize("pattern", ["all_neg", "all_zero", "all_pos", "mixed"])
def test_base3_pack_unpack_roundtrips_edge_lengths_and_patterns(length: int, pattern: str):
    q = _pattern(length, pattern)

    packed = pack_ternary_q_base3_5perbyte_reference(q)
    restored = unpack_ternary_q_base3_5perbyte_reference(packed)

    torch.testing.assert_close(restored, q, atol=0, rtol=0)
    assert packed.format == BASE3_Q_FORMAT
    assert packed.packed.dtype == torch.uint8
    assert packed.logical_shape == (length,)
    assert packed.logical_numel == length
    assert packed.padding_values == (-length) % 5
    assert packed.packed.numel() == (length + 4) // 5
    if length > 0 and length % 5 == 0:
        assert bool((packed.packed < BASE3_FULL_GROUP_CODE_COUNT).all().item())


def test_base3_decode_rejects_unused_codes_and_nonzero_padding_trits():
    full = pack_ternary_q_base3_5perbyte_reference(torch.tensor([-1, 0, 1, 1, 0], dtype=torch.int8))
    bad_full = full.packed.clone()
    bad_full[0] = BASE3_FULL_GROUP_CODE_COUNT
    with pytest.raises(ValueError, match="unused base-3 byte code >=243"):
        unpack_ternary_q_base3_5perbyte_reference(replace(full, packed=bad_full))

    partial = pack_ternary_q_base3_5perbyte_reference(torch.zeros(4, dtype=torch.int8))
    bad_partial = partial.packed.clone()
    bad_partial[0] = int(bad_partial[0].item()) + 3**4
    with pytest.raises(ValueError, match="non-zero padded trits"):
        unpack_ternary_q_base3_5perbyte_reference(replace(partial, packed=bad_partial))

    length_six = pack_ternary_q_base3_5perbyte_reference(torch.tensor([-1, 0, 1, 0, -1, 1], dtype=torch.int8))
    bad_tail = length_six.packed.clone()
    bad_tail[-1] = int(bad_tail[-1].item()) + 3
    with pytest.raises(ValueError, match="non-zero padded trits"):
        unpack_ternary_q_base3_5perbyte_reference(replace(length_six, packed=bad_tail))


def test_base3_storage_forward_parity_reenters_existing_int8_levels_boundary():
    state = _qscale_state((4, 7))
    packed = pack_ternary_q_base3_5perbyte_reference(state.q_levels)
    unpacked_state = QScaleWeightState(
        q_levels=unpack_ternary_q_base3_5perbyte_reference(packed),
        scale=state.scale,
        format=QScaleWeightFormat.INT8_LEVELS,
    )
    x = (torch.arange(14, dtype=torch.float32).view(2, 7) - 3.0) / 11.0

    _, _, fmt = validate_qscale_weight_state(unpacked_state)
    expected = qscale_linear_reference(x, state)
    actual = qscale_linear_reference(x, unpacked_state)

    assert fmt == QScaleWeightFormat.INT8_LEVELS
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_base3_ledger_reports_actual_bytes_metadata_padding_and_c1p1b_target():
    rows = _rows_by_name()

    tiny = rows["tiny_two_projection_fixture_base3_q"]
    assert tiny.eligible_weight_count == 160
    assert tiny.packed_q_data_bits == 264
    assert tiny.q_packed_data_bits_per_weight == pytest.approx(1.65)
    assert tiny.q_packed_metadata_bits_per_weight == pytest.approx(3.2)
    assert tiny.q_packed_total_bits_per_weight == pytest.approx(4.85)
    assert tiny.frozen_scale_fp32_bits_per_weight == pytest.approx(0.4)
    assert tiny.remaining_accumulator_budget_bits_per_weight == pytest.approx(-3.25)
    assert tiny.packed_inclusive_physical_bits_per_weight == pytest.approx(21.25)
    assert tiny.target_achieved is False

    len6 = rows["non_multiple_of_five_len6_base3_q"]
    assert len6.eligible_weight_count == 6
    assert len6.packed_q_data_bits == 16
    assert len6.q_packed_data_bits_per_weight == pytest.approx(16 / 6)
    assert len6.q_packed_metadata_bits_per_weight == pytest.approx(32.0)
    assert len6.frozen_scale_fp32_bits_per_weight == pytest.approx(32 / 6)
    assert len6.remaining_accumulator_budget_bits_per_weight == pytest.approx(-38.0)
    assert len6.packed_inclusive_physical_bits_per_weight == pytest.approx(56.0)

    large = rows["prior_large_fixture_base3_q"]
    assert large.eligible_weight_count == 16384
    assert large.packed_q_data_bits == 3277 * 8
    assert large.q_packed_data_bits_per_weight == pytest.approx(1.60009765625)
    assert large.q_packed_metadata_bits_per_weight == pytest.approx(0.015625)
    assert large.q_packed_total_bits_per_weight == pytest.approx(1.61572265625)
    assert large.frozen_scale_fp32_bits_per_weight == pytest.approx(0.001953125)
    assert large.remaining_accumulator_budget_bits_per_weight == pytest.approx(0.38232421875)
    assert large.packed_inclusive_physical_bits_per_weight == pytest.approx(17.61767578125)
    assert large.dominant_ledger == "accumulator"
    assert large.target_achieved is False

    realistic = rows["illustrative_4096x4096_one_tensor_one_scale_base3_q"]
    assert realistic.eligible_weight_count == 4096 * 4096
    assert realistic.q_packed_data_bits_per_weight == pytest.approx(1.6000003814697266)
    assert realistic.q_packed_metadata_bits_per_weight == pytest.approx(256 / (4096 * 4096))
    assert realistic.frozen_scale_fp32_bits_per_weight == pytest.approx(32 / (4096 * 4096))
    assert realistic.remaining_accumulator_budget_bits_per_weight == pytest.approx(
        2.0
        - realistic.q_packed_total_bits_per_weight
        - realistic.frozen_scale_fp32_bits_per_weight,
    )

    per_row = rows["illustrative_4096x4096_one_tensor_per_row_scale_base3_q"]
    assert per_row.frozen_scale_fp32_bits_per_weight == pytest.approx((4096 * 32) / (4096 * 4096))
    assert per_row.remaining_accumulator_budget_bits_per_weight == pytest.approx(
        2.0 - per_row.q_packed_total_bits_per_weight - per_row.frozen_scale_fp32_bits_per_weight,
    )


def test_base3_measurement_uses_actual_packed_bytes_and_zero_length_is_not_budget_row():
    state_a = _qscale_state((8, 16))
    state_b = _qscale_state((4, 8))
    report = measure_base3_q_entropy_budget(
        [state_a, state_b],
        [
            torch.zeros_like(state_a.q_levels, dtype=torch.int16),
            torch.zeros_like(state_b.q_levels, dtype=torch.int16),
        ],
        regime_name="tiny_measured_qscale_base3_q",
    )

    assert report.eligible_weight_count == 160
    assert report.packed_q_data_bits == 264
    assert report.q_packed_data_bits_per_weight == pytest.approx(1.65)
    assert report.q_packed_metadata_bits_per_weight == pytest.approx(3.2)
    assert report.q_packed_total_bits_per_weight == pytest.approx(4.85)
    validate_base3_q_entropy_ledger(report)

    empty_state = QScaleWeightState(
        q_levels=torch.empty((0, 3), dtype=torch.int8),
        scale=torch.tensor(0.125, dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="zero-length q codec cases are not bits-per-weight budget rows"):
        measure_base3_q_entropy_budget([empty_state], [torch.empty((0, 3), dtype=torch.int16)])
    with pytest.raises(ValueError, match="zero-length q codec cases are not bits-per-weight budget rows"):
        base3_q_entropy_ledger_for_shapes(
            regime_name="zero_length_is_codec_only",
            logical_shapes=((0,),),
            scale_count=1,
        )


def test_base3_false_claim_guard_rejects_q_only_and_inclusive_sub2_with_int16_acc():
    large = _rows_by_name()["prior_large_fixture_base3_q"]

    assert large.q_packed_data_bits_per_weight < 2.0
    assert large.packed_inclusive_physical_bits_per_weight > 2.0
    assert large.target_achieved is False
    validate_base3_q_entropy_ledger(large)

    with pytest.raises(ValueError, match="q-payload sub-2"):
        validate_base3_q_entropy_ledger(large, claimed_q_payload_sub2_achieved=True)
    with pytest.raises(ValueError, match="scale/acc-excluded"):
        validate_base3_q_entropy_ledger(large, claimed_scale_acc_excluded_sub2_achieved=True)
    with pytest.raises(ValueError, match="physical sub-2 claim"):
        validate_base3_q_entropy_ledger(large, claimed_physical_sub2_achieved=True)
    with pytest.raises(ValueError, match="target flag"):
        validate_base3_q_entropy_ledger(replace(large, target_achieved=True, claimable_physical_sub2=True))


def test_base3_q_storage_orthogonality_is_executable_not_accumulator_progress():
    report = base3_q_storage_orthogonality_report()
    assessment = report.candidate_assessment

    validate_base3_q_storage_orthogonality(report)
    assert report.storage_only_status == BASE3_Q_STORAGE_ONLY_STATUS
    assert report.not_accumulator_candidate is True
    assert report.not_c2_accumulator_compression_progress is True
    assert report.no_vote_state_compression is True
    assert assessment.normalized_classification == CandidateClassification.BIT_EXACT
    assert set(assessment.covered_decision_dimensions) == set(required_decision_dimension_names())
    assert assessment.missing_decision_dimensions == ()
    assert assessment.compressed_representation is False
    assert assessment.c2_eligible_by_default is False

    with pytest.raises(ValueError, match="not_accumulator_candidate"):
        validate_base3_q_storage_orthogonality(replace(report, not_accumulator_candidate=False))
    with pytest.raises(ValueError, match="not C2 accumulator-compression progress"):
        validate_base3_q_storage_orthogonality(
            replace(report, not_c2_accumulator_compression_progress=False),
        )
    compressed_assessment = replace(assessment, compressed_representation=True)
    with pytest.raises(ValueError, match="compressed representation"):
        validate_base3_q_storage_orthogonality(replace(report, candidate_assessment=compressed_assessment))


def test_base3_compact_report_omits_raw_arrays_and_names_non_claims():
    report = base3_q_entropy_compact_report()
    payload = report.to_dict()

    assert report.raw_arrays_included is False
    assert payload["label"] == BASE3_Q_ENTROPY_LABEL
    assert payload["codec_summary"]["format"] == BASE3_Q_FORMAT
    assert "no qscale boundary semantics change" in payload["non_claims"]
    assert "no physical sub-2 achievement while int16 accumulators remain" in payload["non_claims"]
    assert "packed" not in payload["codec_summary"]
    assert "q_levels" not in payload["codec_summary"]
    _assert_no_tensors(payload)
