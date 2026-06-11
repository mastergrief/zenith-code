"""Pure three-ledger persistent-state accounting for two-tier carry (B7).

Accounting-only: makes no stability or training claim. Emits three separate named
ledger rows per eligible weight under ternary_hybrid_stack 3-ledger discipline.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    DEFAULT_CARRY_WIDTH,
    POST_FLIP_RESIDUAL_PACKED_BIT_WIDTH,
)

TWO_TIER_PERSISTENT_STATE_LEDGER_SCHEMA_VERSION = (
    "hrm_text_158_two_tier_persistent_state_ledger/v0"
)

LEDGER_1_EFFECTIVE_FORWARD = "ledger_1_effective_forward_logical_ternary"
LEDGER_2_PHYSICAL_PERSISTENT = "ledger_2_physical_persistent_train_state"
LEDGER_3_EVAL_EXPORT = "ledger_3_eval_export_non_authoritative"

PHYSICAL_ROW_INT8_CONTAINER = "int8_container"
PHYSICAL_ROW_LOGICAL_WIDTH = "logical_width"

LOGICAL_NOT_PHYSICAL_UNTIL_PACKING = "logical_not_physical_until_named_packing_scheme"

Q_INT8_BPW = 8
INT16_VOTE_ACCUMULATOR_BPW = 16
W6_CARRY_CONTAINER_BPW = 8
W6_CARRY_LOGICAL_BPW = int(DEFAULT_CARRY_WIDTH)

EFFECTIVE_FORWARD_TERNARY_BPW = float(math.log2(3))

HONEST_LABEL_NOT_SUB_2_BIT_CLAIM = "not_sub_2_bit_claim"
HONEST_LABEL_VOTE_ACCUMULATOR_REPLACEMENT = "vote_accumulator_replacement"
HONEST_LABEL_RESIDUAL_BAND_BITS = "residual_band_bits"

VOTE_ACCUMULATOR_REPLACEMENT_W6_CARRY = "w6_carry_counter"
RESIDUAL_BAND_BITS_WITHIN_CARRY = "5_within_carry_not_additive"

INCUMBENT_CARRIER = "int8_q_plus_int16_vote_accumulator"
CONTAINER_CARRIER = "int8_q_plus_w6_carry_int8_container"

ALLOWED_CLAIM_CLASSES = frozenset(
    {
        "physical_as_implemented",
        LOGICAL_NOT_PHYSICAL_UNTIL_PACKING,
        "effective_forward_logical_ternary",
        "eval_export_non_authoritative",
    }
)

ALLOWED_WIDTH_CLASSES = frozenset(
    {
        "physical_as_implemented",
        LOGICAL_NOT_PHYSICAL_UNTIL_PACKING,
    }
)


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_canonical_ledger_block(block: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(block)).encode("utf-8")).hexdigest()


def reject_sub_2_physical_overclaim(*, claim_class: str) -> None:
    """Default-deny allowlist: unknown claim_class surfaces are rejected."""
    normalized = str(claim_class).strip()
    if normalized not in ALLOWED_CLAIM_CLASSES:
        raise ValueError(
            "claim_class not in allowlist (default-deny anti-overclaim): "
            f"claim_class={claim_class!r}"
        )


def reject_logical_width_labeled_physical(*, width_class: str, total_bpw: float) -> None:
    """Default-deny allowlist with exact width_class/total_bpw pairing."""
    normalized = str(width_class).strip()
    if normalized not in ALLOWED_WIDTH_CLASSES:
        raise ValueError(
            "width_class not in allowlist (default-deny anti-overclaim): "
            f"width_class={width_class!r}, total_bpw={total_bpw}"
        )
    container_bpw = float(compute_container_persistent_bpw())
    logical_bpw = float(compute_logical_persistent_bpw())
    if normalized == "physical_as_implemented":
        if float(total_bpw) != container_bpw:
            raise ValueError(
                "physical_as_implemented width_class requires container bpw: "
                f"width_class={width_class!r}, total_bpw={total_bpw}, "
                f"expected_total_bpw={container_bpw}"
            )
        return
    if normalized == LOGICAL_NOT_PHYSICAL_UNTIL_PACKING:
        if float(total_bpw) != logical_bpw:
            raise ValueError(
                "logical_not_physical_until_named_packing_scheme requires logical bpw: "
                f"width_class={width_class!r}, total_bpw={total_bpw}, "
                f"expected_total_bpw={logical_bpw}"
            )
        return


def _audit_emitted_claim_surfaces(block: Mapping[str, Any]) -> None:
    """Self-audit builder emission through default-deny guards."""
    ledgers = block["ledgers"]
    reject_sub_2_physical_overclaim(
        claim_class=str(ledgers[LEDGER_1_EFFECTIVE_FORWARD]["claim"]),
    )
    physical_rows = ledgers[LEDGER_2_PHYSICAL_PERSISTENT]["rows"]
    for row in physical_rows.values():
        reject_logical_width_labeled_physical(
            width_class=str(row["width_class"]),
            total_bpw=float(row["total_bpw"]),
        )
        if "claim" in row:
            reject_sub_2_physical_overclaim(claim_class=str(row["claim"]))
    reject_sub_2_physical_overclaim(
        claim_class=str(ledgers[LEDGER_3_EVAL_EXPORT]["claim"]),
    )


def assert_residual_band_not_additive_ledger_row(*, ledger_block: Mapping[str, Any]) -> None:
    physical = ledger_block["ledgers"][LEDGER_2_PHYSICAL_PERSISTENT]
    container_row = physical["rows"][PHYSICAL_ROW_INT8_CONTAINER]
    honest = physical["honest_labels"]
    if honest[HONEST_LABEL_RESIDUAL_BAND_BITS] != RESIDUAL_BAND_BITS_WITHIN_CARRY:
        raise AssertionError(
            "residual_band_bits surface must state within-carry encoding, "
            f"got {honest[HONEST_LABEL_RESIDUAL_BAND_BITS]!r}"
        )
    if int(container_row["total_bpw"]) != int(Q_INT8_BPW + W6_CARRY_CONTAINER_BPW):
        raise AssertionError(
            "residual band must not add a separate ledger bpw row: "
            f"container_total_bpw={container_row['total_bpw']}"
        )
    if int(physical["residual_packed_bit_width"]) != int(POST_FLIP_RESIDUAL_PACKED_BIT_WIDTH):
        raise AssertionError(
            "residual_packed_bit_width surface mismatch: "
            f"expected {POST_FLIP_RESIDUAL_PACKED_BIT_WIDTH}, "
            f"got {physical['residual_packed_bit_width']!r}"
        )


def compute_incumbent_persistent_bpw() -> int:
    return int(Q_INT8_BPW + INT16_VOTE_ACCUMULATOR_BPW)


def compute_container_persistent_bpw() -> int:
    return int(Q_INT8_BPW + W6_CARRY_CONTAINER_BPW)


def compute_logical_persistent_bpw() -> int:
    return int(Q_INT8_BPW + W6_CARRY_LOGICAL_BPW)


def compute_physical_delta_bpw(*, incumbent_bpw: int, container_bpw: int) -> int:
    return int(container_bpw) - int(incumbent_bpw)


def _ledger_1_effective_forward() -> dict[str, Any]:
    return {
        "ledger_id": LEDGER_1_EFFECTIVE_FORWARD,
        "carrier": "q_int8.float32 x frozen_scale",
        "bpw": EFFECTIVE_FORWARD_TERNARY_BPW,
        "unchanged_by_lane": True,
        "claim": "effective_forward_logical_ternary",
    }


def _ledger_2_physical_persistent() -> dict[str, Any]:
    container_total = compute_container_persistent_bpw()
    logical_total = compute_logical_persistent_bpw()
    return {
        "ledger_id": LEDGER_2_PHYSICAL_PERSISTENT,
        "dual_row_discipline": True,
        "rows": {
            PHYSICAL_ROW_INT8_CONTAINER: {
                "row_id": PHYSICAL_ROW_INT8_CONTAINER,
                "carrier": CONTAINER_CARRIER,
                "q_bpw": Q_INT8_BPW,
                "w6_carry_bpw": W6_CARRY_CONTAINER_BPW,
                "w6_carry_representation": "int8_container",
                "total_bpw": container_total,
                "width_class": "physical_as_implemented",
            },
            PHYSICAL_ROW_LOGICAL_WIDTH: {
                "row_id": PHYSICAL_ROW_LOGICAL_WIDTH,
                "carrier": "int8_q_plus_w6_carry_logical_width",
                "q_bpw": Q_INT8_BPW,
                "w6_carry_bpw": W6_CARRY_LOGICAL_BPW,
                "w6_carry_representation": "logical_width",
                "total_bpw": logical_total,
                "width_class": LOGICAL_NOT_PHYSICAL_UNTIL_PACKING,
                "bit_packed_realization": "deferred",
            },
        },
        "honest_labels": {
            HONEST_LABEL_NOT_SUB_2_BIT_CLAIM: True,
            HONEST_LABEL_VOTE_ACCUMULATOR_REPLACEMENT: VOTE_ACCUMULATOR_REPLACEMENT_W6_CARRY,
            HONEST_LABEL_RESIDUAL_BAND_BITS: RESIDUAL_BAND_BITS_WITHIN_CARRY,
        },
        "residual_packed_bit_width": int(POST_FLIP_RESIDUAL_PACKED_BIT_WIDTH),
        "residual_not_additive_bpw_row": True,
    }


def _ledger_3_eval_export() -> dict[str, Any]:
    return {
        "ledger_id": LEDGER_3_EVAL_EXPORT,
        "authoritative": False,
        "regeneration_recipe_only": True,
        "carrier_note": (
            "non_authoritative_probe_export; no persistent bpw number claim"
        ),
        "claim": "eval_export_non_authoritative",
    }


def compute_two_tier_persistent_state_ledger_block() -> dict[str, Any]:
    """Return the canonical three-ledger block for one eligible weight."""

    incumbent_bpw = compute_incumbent_persistent_bpw()
    container_bpw = compute_container_persistent_bpw()
    physical_delta_bpw = compute_physical_delta_bpw(
        incumbent_bpw=incumbent_bpw,
        container_bpw=container_bpw,
    )
    block = {
        "schema": TWO_TIER_PERSISTENT_STATE_LEDGER_SCHEMA_VERSION,
        "ledger_count": 3,
        "ledgers": {
            LEDGER_1_EFFECTIVE_FORWARD: _ledger_1_effective_forward(),
            LEDGER_2_PHYSICAL_PERSISTENT: _ledger_2_physical_persistent(),
            LEDGER_3_EVAL_EXPORT: _ledger_3_eval_export(),
        },
        "incumbent_comparison": {
            "incumbent_carrier": INCUMBENT_CARRIER,
            "incumbent_bpw": incumbent_bpw,
            "container_carrier": CONTAINER_CARRIER,
            "container_bpw": container_bpw,
            "physical_delta_bpw": physical_delta_bpw,
            "delta_derivation": "container_bpw_minus_incumbent_bpw",
        },
        "anti_overclaim": {
            "candidate_not_final": True,
            "accounting_only_lane": True,
        },
    }
    assert_residual_band_not_additive_ledger_row(ledger_block=block)
    _audit_emitted_claim_surfaces(block)
    return block


def validate_three_ledgers_present_and_separate(block: Mapping[str, Any]) -> None:
    ledgers = block.get("ledgers")
    if not isinstance(ledgers, dict):
        raise ValueError("ledgers surface must be a mapping")
    expected = (
        LEDGER_1_EFFECTIVE_FORWARD,
        LEDGER_2_PHYSICAL_PERSISTENT,
        LEDGER_3_EVAL_EXPORT,
    )
    if set(ledgers) != set(expected):
        raise ValueError(
            "three ledgers must be present as separate named rows: "
            f"expected={expected}, got={sorted(ledgers)}"
        )
    if int(block.get("ledger_count", 0)) != 3:
        raise ValueError("ledger_count surface must be 3")
