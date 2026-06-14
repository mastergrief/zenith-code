from __future__ import annotations

from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.acc_width_recorded_row_sweep import (
    DEFAULT_HEADROOM_FACTOR,
    MIN_NON_DEGENERATE_THRESHOLD_ABS,
    headroom_passes,
)
from calm.hrm_text_158.native_full_stack.b0_recorded_state_inventory import (
    B0_MULTI_TRACE_BUNDLE_SPECS,
)
from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    CLASSIFIER_CODEC_READY_FOR_CPU_PARITY_ONLY,
    CLASSIFIER_HEADROOM_OR_DOMAIN_FAIL,
    CLASSIFIER_HARNESS_FAIL,
    EXPLICIT_NON_CLAIMS,
    FORBIDDEN_CLAIM_FIELDS,
    W6_SIGNED_MAX,
    W6_SIGNED_MIN,
    W6_WIDTH_BITS,
    clip_then_pack_w6,
    clip_to_w6,
    count_codec_crossing_mismatches_on_trace,
    emit_codec_classifier_receipt,
    max_abs_acc_applied_flips_on_trace,
    pack_w6,
    unpack_w6,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    effective_clip_w6,
)


def test_a1_signed_w6_domain() -> None:
    clip_min, clip_max = effective_clip_w6()
    assert clip_min == W6_SIGNED_MIN == -31
    assert clip_max == W6_SIGNED_MAX == 31


def test_a2_full_domain_roundtrip() -> None:
    for value in range(W6_SIGNED_MIN, W6_SIGNED_MAX + 1):
        packed = pack_w6(value)
        assert 0 <= packed < (1 << W6_WIDTH_BITS)
        assert unpack_w6(packed) == value


def test_a3_strict_reject_out_of_domain() -> None:
    with pytest.raises(ValueError, match="pack_w6 requires value"):
        pack_w6(100)
    with pytest.raises(ValueError, match="pack_w6 requires value"):
        pack_w6(-100)
    with pytest.raises(ValueError, match="pack_w6 requires value"):
        pack_w6(W6_SIGNED_MAX + 1)
    with pytest.raises(ValueError, match="pack_w6 requires value"):
        pack_w6(W6_SIGNED_MIN - 1)


def test_unpack_w6_strict_reject_invalid_packed() -> None:
    with pytest.raises(ValueError, match="unpack_w6 requires packed"):
        unpack_w6(64)
    with pytest.raises(ValueError, match="unpack_w6 requires packed"):
        unpack_w6(-1)
    with pytest.raises(ValueError, match="unpack_w6 requires packed"):
        unpack_w6(100)


def test_a4_effective_clip_matches_w6() -> None:
    assert clip_to_w6(100) == 31
    assert clip_to_w6(-100) == -31
    assert clip_to_w6(0) == 0
    assert clip_to_w6(31) == 31
    assert clip_to_w6(-31) == -31
    assert unpack_w6(clip_then_pack_w6(100)) == 31
    assert unpack_w6(clip_then_pack_w6(-100)) == -31


@pytest.mark.parametrize("bundle", B0_MULTI_TRACE_BUNDLE_SPECS, ids=lambda b: b.capture_id)
def test_a5_recorded_row_crossing_equivalence_after_codec(bundle) -> None:
    trace_path = bundle.path(bundle.b2b_trace_ndjson_relpath)
    if not trace_path.is_file():
        pytest.skip(f"baseline trace missing: {trace_path}")

    mismatches, load_failures = count_codec_crossing_mismatches_on_trace(trace_path)
    assert load_failures == []
    assert mismatches == 0


@pytest.mark.parametrize("bundle", B0_MULTI_TRACE_BUNDLE_SPECS, ids=lambda b: b.capture_id)
def test_a6_headroom_rule(bundle) -> None:
    trace_path = bundle.path(bundle.b2b_trace_ndjson_relpath)
    if not trace_path.is_file():
        pytest.skip(f"baseline trace missing: {trace_path}")

    max_abs, load_failures = max_abs_acc_applied_flips_on_trace(trace_path)
    assert load_failures == []
    assert max_abs > 0
    assert headroom_passes(
        W6_WIDTH_BITS,
        max_abs_acc_applied=max_abs,
        headroom_factor=DEFAULT_HEADROOM_FACTOR,
    )


def test_a7_degenerate_guards_route_to_headroom_or_domain_fail() -> None:
    degenerate = emit_codec_classifier_receipt(
        width_bits=1,
        threshold_abs=MIN_NON_DEGENERATE_THRESHOLD_ABS,
        max_abs_acc_applied_flips=100,
        crossing_mismatch_count=0,
        codec_assertions_pass=True,
    )
    assert degenerate["primary_classifier"] == CLASSIFIER_HEADROOM_OR_DOMAIN_FAIL

    headroom_fail = emit_codec_classifier_receipt(
        width_bits=W6_WIDTH_BITS,
        max_abs_acc_applied_flips=20,
        headroom_factor=DEFAULT_HEADROOM_FACTOR,
        crossing_mismatch_count=0,
        codec_assertions_pass=True,
    )
    assert headroom_fail["primary_classifier"] == CLASSIFIER_HEADROOM_OR_DOMAIN_FAIL

    harness = emit_codec_classifier_receipt(
        harness_failures=["trace_load_error:OSError"],
        width_bits=W6_WIDTH_BITS,
        max_abs_acc_applied_flips=9,
        crossing_mismatch_count=0,
        codec_assertions_pass=True,
    )
    assert harness["primary_classifier"] == CLASSIFIER_HARNESS_FAIL


def test_a8_receipt_has_no_sub2_full_runtime_claim_fields() -> None:
    receipt = emit_codec_classifier_receipt(
        width_bits=W6_WIDTH_BITS,
        max_abs_acc_applied_flips=9,
        crossing_mismatch_count=0,
        codec_assertions_pass=True,
    )
    assert receipt["primary_classifier"] == CLASSIFIER_CODEC_READY_FOR_CPU_PARITY_ONLY
    assert receipt["explicit_non_claims"] == list(EXPLICIT_NON_CLAIMS)
    assert receipt["codec_ready_is_not_sub2"] is True
    assert receipt["codec_ready_is_not_trainer_integrated"] is True
    assert receipt["codec_ready_is_not_gpu_parity"] is True
    assert receipt["codec_ready_is_not_full_sub2_runtime"] is True
    assert FORBIDDEN_CLAIM_FIELDS.isdisjoint(receipt.keys())


def test_clip_then_pack_w6_is_separate_from_strict_pack() -> None:
    assert clip_then_pack_w6(100) == pack_w6(31)
    with pytest.raises(ValueError):
        pack_w6(100)
