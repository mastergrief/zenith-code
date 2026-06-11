from __future__ import annotations

import hashlib
import json
import math

import pytest

from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    POST_FLIP_RESIDUAL_PACKED_BIT_WIDTH,
)
from calm.hrm_text_158.native_full_stack.two_tier_persistent_state_ledger import (
    ALLOWED_CLAIM_CLASSES,
    ALLOWED_WIDTH_CLASSES,
    CONTAINER_CARRIER,
    EFFECTIVE_FORWARD_TERNARY_BPW,
    HONEST_LABEL_NOT_SUB_2_BIT_CLAIM,
    HONEST_LABEL_RESIDUAL_BAND_BITS,
    HONEST_LABEL_VOTE_ACCUMULATOR_REPLACEMENT,
    INCUMBENT_CARRIER,
    LEDGER_1_EFFECTIVE_FORWARD,
    LEDGER_2_PHYSICAL_PERSISTENT,
    LEDGER_3_EVAL_EXPORT,
    LOGICAL_NOT_PHYSICAL_UNTIL_PACKING,
    PHYSICAL_ROW_INT8_CONTAINER,
    PHYSICAL_ROW_LOGICAL_WIDTH,
    Q_INT8_BPW,
    RESIDUAL_BAND_BITS_WITHIN_CARRY,
    VOTE_ACCUMULATOR_REPLACEMENT_W6_CARRY,
    W6_CARRY_CONTAINER_BPW,
    W6_CARRY_LOGICAL_BPW,
    assert_residual_band_not_additive_ledger_row,
    compute_container_persistent_bpw,
    compute_incumbent_persistent_bpw,
    compute_logical_persistent_bpw,
    compute_physical_delta_bpw,
    compute_two_tier_persistent_state_ledger_block,
    reject_logical_width_labeled_physical,
    reject_sub_2_physical_overclaim,
    sha256_canonical_ledger_block,
    validate_three_ledgers_present_and_separate,
)


def _canonical_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def test_b7_three_ledgers_always_present_and_never_merged():
    block = compute_two_tier_persistent_state_ledger_block()
    validate_three_ledgers_present_and_separate(block)
    assert block["ledger_count"] == 3
    assert set(block["ledgers"]) == {
        LEDGER_1_EFFECTIVE_FORWARD,
        LEDGER_2_PHYSICAL_PERSISTENT,
        LEDGER_3_EVAL_EXPORT,
    }
    physical = block["ledgers"][LEDGER_2_PHYSICAL_PERSISTENT]
    assert physical["dual_row_discipline"] is True
    assert set(physical["rows"]) == {PHYSICAL_ROW_INT8_CONTAINER, PHYSICAL_ROW_LOGICAL_WIDTH}


def test_b7_physical_ledger_dual_row_exact_values_and_deferral_label():
    block = compute_two_tier_persistent_state_ledger_block()
    rows = block["ledgers"][LEDGER_2_PHYSICAL_PERSISTENT]["rows"]
    container = rows[PHYSICAL_ROW_INT8_CONTAINER]
    logical = rows[PHYSICAL_ROW_LOGICAL_WIDTH]
    assert container["q_bpw"] == Q_INT8_BPW
    assert container["w6_carry_bpw"] == W6_CARRY_CONTAINER_BPW
    assert container["total_bpw"] == 16
    assert logical["q_bpw"] == Q_INT8_BPW
    assert logical["w6_carry_bpw"] == W6_CARRY_LOGICAL_BPW
    assert logical["total_bpw"] == 14
    assert logical["width_class"] == LOGICAL_NOT_PHYSICAL_UNTIL_PACKING
    assert logical["bit_packed_realization"] == "deferred"


def test_b7_incumbent_delta_derived_not_hardcoded():
    incumbent_bpw = compute_incumbent_persistent_bpw()
    container_bpw = compute_container_persistent_bpw()
    block = compute_two_tier_persistent_state_ledger_block()
    comparison = block["incumbent_comparison"]
    assert incumbent_bpw == 24
    assert container_bpw == 16
    assert comparison["incumbent_carrier"] == INCUMBENT_CARRIER
    assert comparison["container_carrier"] == CONTAINER_CARRIER
    assert comparison["physical_delta_bpw"] == compute_physical_delta_bpw(
        incumbent_bpw=incumbent_bpw,
        container_bpw=container_bpw,
    )
    assert comparison["physical_delta_bpw"] == -8
    assert comparison["physical_delta_bpw"] == container_bpw - incumbent_bpw


def test_b7_residual_not_additive_within_carry_width():
    block = compute_two_tier_persistent_state_ledger_block()
    assert_residual_band_not_additive_ledger_row(ledger_block=block)
    physical = block["ledgers"][LEDGER_2_PHYSICAL_PERSISTENT]
    assert physical["residual_not_additive_bpw_row"] is True
    assert physical["residual_packed_bit_width"] == POST_FLIP_RESIDUAL_PACKED_BIT_WIDTH
    assert (
        physical["honest_labels"][HONEST_LABEL_RESIDUAL_BAND_BITS]
        == RESIDUAL_BAND_BITS_WITHIN_CARRY
    )


def test_b7_honest_labels_and_effective_forward_ledger():
    block = compute_two_tier_persistent_state_ledger_block()
    forward = block["ledgers"][LEDGER_1_EFFECTIVE_FORWARD]
    honest = block["ledgers"][LEDGER_2_PHYSICAL_PERSISTENT]["honest_labels"]
    assert forward["bpw"] == pytest.approx(EFFECTIVE_FORWARD_TERNARY_BPW)
    assert forward["bpw"] == pytest.approx(math.log2(3))
    assert forward["unchanged_by_lane"] is True
    assert honest[HONEST_LABEL_NOT_SUB_2_BIT_CLAIM] is True
    assert honest[HONEST_LABEL_VOTE_ACCUMULATOR_REPLACEMENT] == (
        VOTE_ACCUMULATOR_REPLACEMENT_W6_CARRY
    )


def test_b7_eval_export_ledger_is_non_authoritative_recipe_only():
    block = compute_two_tier_persistent_state_ledger_block()
    export = block["ledgers"][LEDGER_3_EVAL_EXPORT]
    assert export["authoritative"] is False
    assert export["regeneration_recipe_only"] is True
    assert "bpw" not in export


def test_b7_anti_overclaim_claim_allowlist_default_deny_fail_closed():
    with pytest.raises(ValueError, match="claim_class not in allowlist"):
        reject_sub_2_physical_overclaim(claim_class="sub_2_bit_persistent")
    with pytest.raises(ValueError, match="claim_class not in allowlist"):
        reject_sub_2_physical_overclaim(claim_class="one_point_nine_bpw_physical")
    with pytest.raises(ValueError, match="claim_class not in allowlist"):
        reject_sub_2_physical_overclaim(claim_class="sub_2_bit_physical_persistent")
    for legal_claim in ALLOWED_CLAIM_CLASSES:
        reject_sub_2_physical_overclaim(claim_class=legal_claim)


def test_b7_anti_overclaim_width_allowlist_and_class_value_pairing_fail_closed():
    with pytest.raises(ValueError, match="width_class not in allowlist"):
        reject_logical_width_labeled_physical(
            width_class="physical_sub_2_or_packed_ternary",
            total_bpw=13.9,
        )
    with pytest.raises(ValueError, match="physical_as_implemented width_class requires container bpw"):
        reject_logical_width_labeled_physical(
            width_class="physical_as_implemented",
            total_bpw=13.9,
        )
    with pytest.raises(ValueError, match="physical_as_implemented width_class requires container bpw"):
        reject_logical_width_labeled_physical(
            width_class="physical_as_implemented",
            total_bpw=14.0,
        )
    with pytest.raises(
        ValueError,
        match="logical_not_physical_until_named_packing_scheme requires logical bpw",
    ):
        reject_logical_width_labeled_physical(
            width_class=LOGICAL_NOT_PHYSICAL_UNTIL_PACKING,
            total_bpw=13.9,
        )
    reject_logical_width_labeled_physical(
        width_class="physical_as_implemented",
        total_bpw=float(compute_container_persistent_bpw()),
    )
    reject_logical_width_labeled_physical(
        width_class=LOGICAL_NOT_PHYSICAL_UNTIL_PACKING,
        total_bpw=float(compute_logical_persistent_bpw()),
    )
    assert set(ALLOWED_WIDTH_CLASSES) == {
        "physical_as_implemented",
        LOGICAL_NOT_PHYSICAL_UNTIL_PACKING,
    }


def test_b7_builder_self_audits_emitted_claim_surfaces():
    block = compute_two_tier_persistent_state_ledger_block()
    forward = block["ledgers"][LEDGER_1_EFFECTIVE_FORWARD]
    export = block["ledgers"][LEDGER_3_EVAL_EXPORT]
    rows = block["ledgers"][LEDGER_2_PHYSICAL_PERSISTENT]["rows"]
    assert forward["claim"] in ALLOWED_CLAIM_CLASSES
    assert export["claim"] in ALLOWED_CLAIM_CLASSES
    assert rows[PHYSICAL_ROW_INT8_CONTAINER]["width_class"] in ALLOWED_WIDTH_CLASSES
    assert rows[PHYSICAL_ROW_LOGICAL_WIDTH]["width_class"] in ALLOWED_WIDTH_CLASSES


def test_b7_canonical_json_ledger_block_sha_stable():
    block = compute_two_tier_persistent_state_ledger_block()
    payload_a = _canonical_json(block)
    payload_b = _canonical_json(compute_two_tier_persistent_state_ledger_block())
    assert payload_a == payload_b
    digest = sha256_canonical_ledger_block(block)
    assert digest == hashlib.sha256(payload_a.encode("utf-8")).hexdigest()
    assert len(digest) == 64
