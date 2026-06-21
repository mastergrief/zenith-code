"""B2-5a″ Stage-A Step-0′ width classifier receipt (CPU budget + compile probe classify).

Records realistic-size budget measurements, compile/resource probe outcomes, and
frozen assertions #1–#4.  Does NOT mint a native pass, does NOT claim runtime
sort correctness, does NOT flip ``global_cap_margin_only_reference``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_step0_budget_receipt import (
    TRITON_SINGLE_BLOCK_ROW_CEILING,
)

GLOBAL_RATE_CAP_MARGIN_SELECTION_MULTIBLOCK_STEP0_BUDGET_SCHEMA_VERSION = (
    "hrm_text_158_global_rate_cap_margin_selection_multiblock_step0_budget/v0.b2_5a_double_prime"
)

GLOBAL_RATE_CAP_MARGIN_SELECTION_MULTIBLOCK_STEP0_NON_CLAIMS: tuple[str, ...] = (
    "B2-5a″ Stage-A Step-0′ is CPU budget + compile classify only; NOT runtime sort parity",
    "B2-5a″ Stage-A does NOT mint selection_parity_pass",
    "B2-5a″ Stage-A does NOT flip global_cap_margin_only_reference",
    "B2-5a″ Stage-A does NOT touch the deferred-backlog ledger",
    "B2-5a″ Stage-A does NOT claim readiness / optimizer_credit_state / full-loop",
    "B2-5a″ Stage-B kernel proof is a SEPARATE +1 implement bound to this decision",
    "B2-5a″ compile probe success does NOT imply runtime sort-key correctness",
)

REALISTIC_ROW_COUNTS: tuple[int, ...] = (1280, 1536, 2048)
WIDER_SINGLE_BLOCK_SORT_PADDED_N = 2048
OPTIONAL_BASELINE_SORT_PADDED_N = 1024

STAGE_B_MECHANISM_WIDER_SINGLE_BLOCK = "mechanism_3_wider_single_block"
STAGE_B_MECHANISM_MULTIBLOCK_MERGE = "mechanism_3_multiblock_merge"
STAGE_B_MECHANISM_BUDGET_INFEASIBLE = "budget_infeasible"


class MultiblockStep0Decision(str, Enum):
    WIDER_SINGLE_BLOCK_COMPILE_VIABLE = "wider_single_block_compile_viable"
    MERGE_TREE_REQUIRED = "merge_tree_required"
    BUDGET_INFEASIBLE = "budget_infeasible"


@dataclass(frozen=True)
class MultiblockStep0FixtureMeasurement:
    fixture_name: str
    row_count: int
    max_global_flat_index: int
    index_width: int
    max_abs_observed: int
    rank_bits: int
    pos_width: int
    full_pack_bits: int
    host_max_full_key: int
    sort_padded_n: int
    int63_full_pack_ok: bool
    padding_headroom_ok: bool
    budget_infeasible: bool
    cap: int
    block: int = TRITON_SINGLE_BLOCK_ROW_CEILING


@dataclass(frozen=True)
class CompileProbeMeasurement:
    sort_padded_n: int
    n_rows_sample: int
    compile_probe_executed: bool
    compile_ok: bool
    kernel_symbol: str
    error_detail: str = ""


@dataclass(frozen=True)
class MultiblockStep0AssertionResults:
    global_pos_permutation_ok: bool
    merge_half_capacity_ok: bool
    staging_layout_ok: bool
    bad_staging_layout_fails: bool
    probe_width_pow2_ok: bool


@dataclass(frozen=True)
class GlobalRateCapMarginSelectionMultiblockStep0BudgetReceipt:
    schema_version: str = GLOBAL_RATE_CAP_MARGIN_SELECTION_MULTIBLOCK_STEP0_BUDGET_SCHEMA_VERSION
    fixture_measurements: tuple[MultiblockStep0FixtureMeasurement, ...] = ()
    compile_probes: tuple[CompileProbeMeasurement, ...] = ()
    assertion_results: MultiblockStep0AssertionResults | None = None
    decision: MultiblockStep0Decision = MultiblockStep0Decision.BUDGET_INFEASIBLE
    recommended_stage_b_mechanism: str = STAGE_B_MECHANISM_BUDGET_INFEASIBLE
    wider_single_block_sort_padded_n: int = WIDER_SINGLE_BLOCK_SORT_PADDED_N
    non_claims: tuple[str, ...] = GLOBAL_RATE_CAP_MARGIN_SELECTION_MULTIBLOCK_STEP0_NON_CLAIMS


def recommended_mechanism_for_multiblock_decision(
    decision: MultiblockStep0Decision,
) -> str:
    if decision == MultiblockStep0Decision.WIDER_SINGLE_BLOCK_COMPILE_VIABLE:
        return STAGE_B_MECHANISM_WIDER_SINGLE_BLOCK
    if decision == MultiblockStep0Decision.MERGE_TREE_REQUIRED:
        return STAGE_B_MECHANISM_MULTIBLOCK_MERGE
    return STAGE_B_MECHANISM_BUDGET_INFEASIBLE


def classify_multiblock_step0_decision(
    *,
    budget_infeasible: bool,
    primary_compile_ok: bool,
) -> MultiblockStep0Decision:
    if budget_infeasible:
        return MultiblockStep0Decision.BUDGET_INFEASIBLE
    if primary_compile_ok:
        return MultiblockStep0Decision.WIDER_SINGLE_BLOCK_COMPILE_VIABLE
    return MultiblockStep0Decision.MERGE_TREE_REQUIRED


def build_global_rate_cap_margin_selection_multiblock_step0_budget_receipt(
    *,
    fixture_measurements: tuple[MultiblockStep0FixtureMeasurement, ...],
    compile_probes: tuple[CompileProbeMeasurement, ...],
    assertion_results: MultiblockStep0AssertionResults,
) -> GlobalRateCapMarginSelectionMultiblockStep0BudgetReceipt:
    budget_infeasible = any(m.budget_infeasible for m in fixture_measurements) or not all(
        (
            assertion_results.global_pos_permutation_ok,
            assertion_results.merge_half_capacity_ok,
            assertion_results.staging_layout_ok,
            assertion_results.bad_staging_layout_fails,
            assertion_results.probe_width_pow2_ok,
        )
    )
    primary = next(
        (p for p in compile_probes if p.sort_padded_n == WIDER_SINGLE_BLOCK_SORT_PADDED_N),
        None,
    )
    primary_compile_ok = bool(primary and primary.compile_ok)
    decision = classify_multiblock_step0_decision(
        budget_infeasible=budget_infeasible,
        primary_compile_ok=primary_compile_ok,
    )
    return GlobalRateCapMarginSelectionMultiblockStep0BudgetReceipt(
        fixture_measurements=fixture_measurements,
        compile_probes=compile_probes,
        assertion_results=assertion_results,
        decision=decision,
        recommended_stage_b_mechanism=recommended_mechanism_for_multiblock_decision(decision),
    )


def validate_global_rate_cap_margin_selection_multiblock_step0_budget_receipt(
    receipt: GlobalRateCapMarginSelectionMultiblockStep0BudgetReceipt,
) -> None:
    if (
        receipt.schema_version
        != GLOBAL_RATE_CAP_MARGIN_SELECTION_MULTIBLOCK_STEP0_BUDGET_SCHEMA_VERSION
    ):
        raise ValueError("schema_version mismatch")
    if receipt.wider_single_block_sort_padded_n != WIDER_SINGLE_BLOCK_SORT_PADDED_N:
        raise ValueError("wider_single_block_sort_padded_n must be 2048")
    if receipt.assertion_results is None:
        raise ValueError("assertion_results required")
    for probe in receipt.compile_probes:
        if probe.sort_padded_n == 1536:
            raise ValueError("illegal SORT_PADDED_N=1536 is forbidden in Step-0′ probes")
        if probe.sort_padded_n not in (
            WIDER_SINGLE_BLOCK_SORT_PADDED_N,
            OPTIONAL_BASELINE_SORT_PADDED_N,
        ):
            raise ValueError(f"unexpected probe width {probe.sort_padded_n}")
    if (
        receipt.recommended_stage_b_mechanism
        != recommended_mechanism_for_multiblock_decision(receipt.decision)
    ):
        raise ValueError("recommended_stage_b_mechanism inconsistent with decision")
    for measurement in receipt.fixture_measurements:
        if measurement.row_count not in REALISTIC_ROW_COUNTS:
            raise ValueError(
                f"fixture row_count must be one of {REALISTIC_ROW_COUNTS}, "
                f"got {measurement.row_count}"
            )
        if measurement.sort_padded_n != WIDER_SINGLE_BLOCK_SORT_PADDED_N:
            raise ValueError("realistic fixture sort_padded_n must be 2048")


__all__ = [
    "GLOBAL_RATE_CAP_MARGIN_SELECTION_MULTIBLOCK_STEP0_BUDGET_SCHEMA_VERSION",
    "GLOBAL_RATE_CAP_MARGIN_SELECTION_MULTIBLOCK_STEP0_NON_CLAIMS",
    "CompileProbeMeasurement",
    "GlobalRateCapMarginSelectionMultiblockStep0BudgetReceipt",
    "MultiblockStep0AssertionResults",
    "MultiblockStep0Decision",
    "MultiblockStep0FixtureMeasurement",
    "OPTIONAL_BASELINE_SORT_PADDED_N",
    "REALISTIC_ROW_COUNTS",
    "STAGE_B_MECHANISM_BUDGET_INFEASIBLE",
    "STAGE_B_MECHANISM_MULTIBLOCK_MERGE",
    "STAGE_B_MECHANISM_WIDER_SINGLE_BLOCK",
    "WIDER_SINGLE_BLOCK_SORT_PADDED_N",
    "build_global_rate_cap_margin_selection_multiblock_step0_budget_receipt",
    "classify_multiblock_step0_decision",
    "recommended_mechanism_for_multiblock_decision",
    "validate_global_rate_cap_margin_selection_multiblock_step0_budget_receipt",
]
