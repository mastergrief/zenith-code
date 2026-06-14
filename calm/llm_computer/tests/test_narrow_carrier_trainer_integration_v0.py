from __future__ import annotations

import os

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.narrow_carrier_trainer_integration import (
    CLASSIFIER_S2_FLAG_OFF_IDENTITY_AND_PARITY_OK,
    EXPLICIT_NON_CLAIMS,
    FORBIDDEN_CLAIM_FIELDS,
    RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV,
    apply_trainer_boundary_narrow_carrier,
    assert_no_packed_w6_state_leak,
    count_int16_vs_w6_crossing_mismatches,
    emit_s2_classifier_receipt,
    narrow_carrier_w6_enabled,
    roundtrip_int16_values_through_trainer_boundary,
    roundtrip_replay_clip_int16_value,
)
from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    clip_then_pack_w6,
    pack_w6,
    unpack_w6,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    carry_self_update_row,
    crosses_threshold,
)


def _tiny_state() -> BoundedDeltaTensorState:
    q = torch.tensor([[0, 1, -1]], dtype=torch.int8)
    acc = torch.tensor([[5, -9, 21]], dtype=torch.int16)
    return make_bounded_tensor_state("tiny.proj", q, 1.0, acc)


def _tiny_fixture_rows() -> list[tuple[int, int, int]]:
    return [
        (5, 3, 0),
        (-9, 2, 1),
        (21, -4, -1),
        (10, 0, 0),
        (-10, 0, 0),
    ]


@pytest.fixture(autouse=True)
def _clear_narrow_carrier_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV, raising=False)


def test_b1_flag_off_identity_bit_identical_int16() -> None:
    state = _tiny_state()
    shadow = state.exact_accumulator_shadow
    assert shadow is not None

    vu = state.vote_update_state()
    assert torch.equal(vu.accumulators, shadow)
    assert narrow_carrier_w6_enabled() is False
    assert torch.equal(
        apply_trainer_boundary_narrow_carrier(shadow),
        shadow,
    )


def test_b4_rollback_default_off_no_w6_leak() -> None:
    state = _tiny_state()
    shadow = state.exact_accumulator_shadow
    assert shadow is not None

    os.environ[RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV] = "1"
    try:
        _ = state.vote_update_state()
    finally:
        os.environ.pop(RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV, None)

    disabled_vu = state.vote_update_state()
    assert torch.equal(disabled_vu.accumulators, shadow)

    schema = state.to_schema_dict(parity_check=False)
    assert_no_packed_w6_state_leak(schema)


def test_b2_strict_in_domain_roundtrip_through_trainer_boundary() -> None:
    values = list(range(-31, 32))
    roundtripped = roundtrip_int16_values_through_trainer_boundary(values, enabled=True)
    assert roundtripped == values


def test_enabled_trainer_boundary_rejects_out_of_domain_accumulator() -> None:
    acc = torch.tensor([[100, -100]], dtype=torch.int16)
    with pytest.raises(ValueError, match="pack_w6 requires value"):
        apply_trainer_boundary_narrow_carrier(acc, enabled=True)

    with pytest.raises(ValueError, match="pack_w6 requires value"):
        roundtrip_int16_values_through_trainer_boundary([100], enabled=True)


def test_replay_clip_helper_is_separate_from_trainer_boundary() -> None:
    assert roundtrip_replay_clip_int16_value(100) == 31
    assert roundtrip_replay_clip_int16_value(-100) == -31
    assert unpack_w6(clip_then_pack_w6(100)) == 31
    with pytest.raises(ValueError, match="pack_w6 requires value"):
        pack_w6(100)


def test_b3_int16_oracle_vs_w6_carrier_parity_tiny_cpu_fixture() -> None:
    for pre_acc, vote, q_level in _tiny_fixture_rows():
        new_acc_16 = carry_self_update_row(pre_acc, vote, width=16)
        new_acc_6 = carry_self_update_row(pre_acc, vote, width=6)
        oracle_cross = crosses_threshold(new_acc_16, current_q_level=q_level, threshold_abs=10)
        carrier_acc = unpack_w6(pack_w6(new_acc_6))
        carrier_cross = crosses_threshold(carrier_acc, current_q_level=q_level, threshold_abs=10)
        assert oracle_cross == carrier_cross

    mismatches = count_int16_vs_w6_crossing_mismatches(_tiny_fixture_rows(), enabled=True)
    assert mismatches == 0


def test_b5_explicit_non_claims() -> None:
    receipt = emit_s2_classifier_receipt(
        flag_off_identity_pass=True,
        boundary_roundtrip_pass=True,
        parity_mismatch_count=0,
        rollback_pass=True,
    )
    assert receipt["primary_classifier"] == CLASSIFIER_S2_FLAG_OFF_IDENTITY_AND_PARITY_OK
    assert receipt["explicit_non_claims"] == list(EXPLICIT_NON_CLAIMS)
    assert receipt["s2_ok_is_not_live_training"] is True
    assert receipt["s2_ok_is_not_gpu_parity"] is True
    assert receipt["s2_ok_is_not_checkpoint_mutation"] is True
    assert FORBIDDEN_CLAIM_FIELDS.isdisjoint(receipt.keys())
