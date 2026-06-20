"""B2-5a′ Stage-1 Step-0 budget/shape receipt (CPU measurement, no kernel).

Records per-fixture bit-budget / shape measurements and pre-registers the
Stage-2 mechanism decision.  Does NOT mint a native pass, does NOT flip
``global_cap_margin_only_reference``, does NOT touch the ledger.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

GLOBAL_RATE_CAP_MARGIN_SELECTION_STEP0_BUDGET_SCHEMA_VERSION = (
    "hrm_text_158_global_rate_cap_margin_selection_step0_budget/v0.b2_5a_prime"
)

GLOBAL_RATE_CAP_MARGIN_SELECTION_STEP0_NON_CLAIMS: tuple[str, ...] = (
    "B2-5a′ Stage-1 is CPU Step-0 measurement only; ZERO @triton.jit",
    "B2-5a′ Stage-1 does NOT mint a native selection pass",
    "B2-5a′ Stage-1 does NOT flip global_cap_margin_only_reference",
    "B2-5a′ Stage-1 does NOT touch the deferred-backlog ledger",
    "B2-5a′ Stage-1 does NOT claim readiness / optimizer_credit_state / full-loop",
    "B2-5a′ Stage-2 kernel proof is a SEPARATE +1 implement bound to this decision",
)

TRITON_SINGLE_BLOCK_ROW_CEILING = 1024

STAGE2_MECHANISM_FULL_PACKED_KEY = "mechanism_3_full_packed_key"
STAGE2_MECHANISM_KEY_PAYLOAD_COPERMUTE = "mechanism_2_key_payload_copermute"
STAGE2_MECHANISM_MULTIBLOCK_DEFERRED = "row_count_multiblock_deferred"
STAGE2_MECHANISM_BUDGET_INFEASIBLE = "budget_infeasible"
STAGE2_MECHANISM_EMPTY_ROWS = "empty_rows_no_kernel"


class Step0BudgetDecision(str, Enum):
    EMPTY_ROWS = "empty_rows"
    FULL_PACK_VIABLE = "full_pack_viable"
    KEY_PAYLOAD_REQUIRED = "key_payload_required"
    ROW_COUNT_MULTIBLOCK_DEFERRED = "row_count_multiblock_deferred"
    BUDGET_INFEASIBLE = "budget_infeasible"


_DECISION_SEVERITY: dict[Step0BudgetDecision, int] = {
    Step0BudgetDecision.ROW_COUNT_MULTIBLOCK_DEFERRED: 4,
    Step0BudgetDecision.BUDGET_INFEASIBLE: 3,
    Step0BudgetDecision.KEY_PAYLOAD_REQUIRED: 2,
    Step0BudgetDecision.FULL_PACK_VIABLE: 1,
    Step0BudgetDecision.EMPTY_ROWS: 0,
}


@dataclass(frozen=True)
class Step0FixtureMeasurement:
    fixture_name: str
    row_count: int
    max_global_flat_index: int
    index_width: int
    max_abs_observed: int
    rank_bits: int
    pos_width: int
    full_pack_bits: int
    rank_index_bits: int
    cap: int
    total_numel: int
    block: int
    sort_padded_n: int
    row_count_le_block: bool
    int63_full_pack_ok: bool
    int63_rank_index_ok: bool
    assert_d_ok: bool
    empty_branch_taken: bool
    decision: Step0BudgetDecision
    recommended_stage2_mechanism: str
    tensor_offsets_non_contiguous: bool = False


@dataclass(frozen=True)
class GlobalRateCapMarginSelectionStep0BudgetReceipt:
    schema_version: str = GLOBAL_RATE_CAP_MARGIN_SELECTION_STEP0_BUDGET_SCHEMA_VERSION
    fixture_measurements: tuple[Step0FixtureMeasurement, ...] = ()
    worst_case_decision: Step0BudgetDecision = Step0BudgetDecision.BUDGET_INFEASIBLE
    recommended_stage2_mechanism: str = STAGE2_MECHANISM_BUDGET_INFEASIBLE
    block: int = TRITON_SINGLE_BLOCK_ROW_CEILING
    non_claims: tuple[str, ...] = GLOBAL_RATE_CAP_MARGIN_SELECTION_STEP0_NON_CLAIMS


def recommended_mechanism_for_decision(decision: Step0BudgetDecision) -> str:
    if decision == Step0BudgetDecision.FULL_PACK_VIABLE:
        return STAGE2_MECHANISM_FULL_PACKED_KEY
    if decision == Step0BudgetDecision.KEY_PAYLOAD_REQUIRED:
        return STAGE2_MECHANISM_KEY_PAYLOAD_COPERMUTE
    if decision == Step0BudgetDecision.ROW_COUNT_MULTIBLOCK_DEFERRED:
        return STAGE2_MECHANISM_MULTIBLOCK_DEFERRED
    if decision == Step0BudgetDecision.EMPTY_ROWS:
        return STAGE2_MECHANISM_EMPTY_ROWS
    return STAGE2_MECHANISM_BUDGET_INFEASIBLE


def classify_step0_budget_decision(
    *,
    row_count: int,
    int63_full_pack_ok: bool,
    int63_rank_index_ok: bool,
    row_count_le_block: bool,
    assert_d_ok: bool,
) -> Step0BudgetDecision:
    if row_count == 0:
        return Step0BudgetDecision.EMPTY_ROWS
    if row_count > TRITON_SINGLE_BLOCK_ROW_CEILING:
        return Step0BudgetDecision.ROW_COUNT_MULTIBLOCK_DEFERRED
    if int63_full_pack_ok and row_count_le_block and assert_d_ok:
        return Step0BudgetDecision.FULL_PACK_VIABLE
    if int63_rank_index_ok and row_count_le_block:
        return Step0BudgetDecision.KEY_PAYLOAD_REQUIRED
    return Step0BudgetDecision.BUDGET_INFEASIBLE


def worst_case_decision(
    decisions: tuple[Step0BudgetDecision, ...],
) -> Step0BudgetDecision:
    if not decisions:
        return Step0BudgetDecision.BUDGET_INFEASIBLE
    return max(decisions, key=lambda d: _DECISION_SEVERITY[d])


def build_global_rate_cap_margin_selection_step0_budget_receipt(
  measurements: tuple[Step0FixtureMeasurement, ...],
) -> GlobalRateCapMarginSelectionStep0BudgetReceipt:
    decisions = tuple(m.decision for m in measurements)
    worst = worst_case_decision(decisions)
    return GlobalRateCapMarginSelectionStep0BudgetReceipt(
        fixture_measurements=measurements,
        worst_case_decision=worst,
        recommended_stage2_mechanism=recommended_mechanism_for_decision(worst),
    )


def validate_global_rate_cap_margin_selection_step0_budget_receipt(
    receipt: GlobalRateCapMarginSelectionStep0BudgetReceipt,
) -> None:
    if receipt.schema_version != GLOBAL_RATE_CAP_MARGIN_SELECTION_STEP0_BUDGET_SCHEMA_VERSION:
        raise ValueError("schema_version mismatch")
    if receipt.block != TRITON_SINGLE_BLOCK_ROW_CEILING:
        raise ValueError("block must be 1024 for Stage-1 Step-0")
    for measurement in receipt.fixture_measurements:
        if measurement.empty_branch_taken:
            if measurement.row_count != 0:
                raise ValueError("empty_branch_taken requires row_count=0")
            if measurement.max_global_flat_index != -1:
                raise ValueError("empty branch must not fabricate max_global_flat_index")
            if measurement.index_width != -1:
                raise ValueError("empty branch must not fabricate index_width")
            if measurement.max_abs_observed != -1:
                raise ValueError("empty branch must not fabricate max_abs_observed")
            if measurement.decision != Step0BudgetDecision.EMPTY_ROWS:
                raise ValueError("empty branch must carry EMPTY_ROWS decision")
            continue
        if measurement.row_count < 0:
            raise ValueError("row_count must be >= 0")
        if measurement.index_width < 1:
            raise ValueError("non-empty fixture must record index_width >= 1")
        if measurement.max_abs_observed < 0:
            raise ValueError("non-empty fixture must record max_abs_observed >= 0")
        if measurement.recommended_stage2_mechanism != recommended_mechanism_for_decision(
            measurement.decision
        ):
            raise ValueError("recommended_stage2_mechanism inconsistent with decision")
    if receipt.recommended_stage2_mechanism != recommended_mechanism_for_decision(
        receipt.worst_case_decision
    ):
        raise ValueError("aggregate recommended_stage2_mechanism inconsistent with worst_case")
