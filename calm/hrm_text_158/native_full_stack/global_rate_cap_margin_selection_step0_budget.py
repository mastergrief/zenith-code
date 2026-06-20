"""B2-5a′ Stage-1 Step-0 budget/shape measurement harness (CPU, no kernel).

Imports banked packed-key helpers from ``packed_key_scaffold.py`` only —
does NOT invoke the scaffold ``torch.topk`` sort path.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    tensor_offsets_for_vote_update_states,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_packed_key_scaffold import (
    GlobalRateCapMarginSelectionFeasibilityNull,
    _compute_packed_key_budget,
    _device_row_tensors_for_selection,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_step0_budget_receipt import (
    GLOBAL_RATE_CAP_MARGIN_SELECTION_STEP0_BUDGET_SCHEMA_VERSION,
    GlobalRateCapMarginSelectionStep0BudgetReceipt,
    Step0BudgetDecision,
    Step0FixtureMeasurement,
    TRITON_SINGLE_BLOCK_ROW_CEILING,
    build_global_rate_cap_margin_selection_step0_budget_receipt,
    classify_step0_budget_decision,
    recommended_mechanism_for_decision,
    validate_global_rate_cap_margin_selection_step0_budget_receipt,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
)


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _bit_length_non_negative(value: int) -> int:
    if value < 0:
        raise ValueError(f"bit_length input must be >= 0, got {value}")
    return max(1, int(value).bit_length()) if value > 0 else 1


def _total_numel(inputs: list[GlobalRateCapTensorInput]) -> int:
    return sum(int(item.state.q_levels.numel()) for item in inputs)


def _offsets_non_contiguous(tensor_offsets: dict[str, int], inputs: list[GlobalRateCapTensorInput]) -> bool:
    if len(inputs) < 2:
        return False
    running = 0
    for item in inputs:
        offset = int(tensor_offsets[item.state_key])
        if offset != running:
            return True
        running += int(item.state.q_levels.numel())
    return False


def measure_step0_fixture(
    *,
    fixture_name: str,
    inputs: list[GlobalRateCapTensorInput],
    spec: GlobalRateCapSpec,
    tensor_offsets: dict[str, int] | None = None,
    device: torch.device | None = None,
) -> Step0FixtureMeasurement:
    """Measure one fixture.  Empty row_count=0 returns documented null row (no fake budget)."""

    dev = device or torch.device("cpu")
    offsets = dict(tensor_offsets or tensor_offsets_for_vote_update_states(inputs))
    rows = _device_row_tensors_for_selection(inputs, tensor_offsets=offsets, device=dev)
    row_count = int(rows["global_indices"].numel())
    cap = max(0, int(spec.cap))
    total_numel = _total_numel(inputs)
    block = TRITON_SINGLE_BLOCK_ROW_CEILING

    if row_count == 0:
        decision = Step0BudgetDecision.EMPTY_ROWS
        return Step0FixtureMeasurement(
            fixture_name=fixture_name,
            row_count=0,
            max_global_flat_index=-1,
            index_width=-1,
            max_abs_observed=-1,
            rank_bits=-1,
            pos_width=-1,
            full_pack_bits=-1,
            rank_index_bits=-1,
            cap=cap,
            total_numel=total_numel,
            block=block,
            sort_padded_n=1,
            row_count_le_block=True,
            int63_full_pack_ok=False,
            int63_rank_index_ok=False,
            assert_d_ok=False,
            empty_branch_taken=True,
            decision=decision,
            recommended_stage2_mechanism=recommended_mechanism_for_decision(decision),
            tensor_offsets_non_contiguous=_offsets_non_contiguous(offsets, inputs),
        )

    try:
        budget = _compute_packed_key_budget(
            global_flat_indices=rows["global_indices"],
            abs_new_acc=rows["abs_new_acc"],
        )
    except GlobalRateCapMarginSelectionFeasibilityNull:
        decision = Step0BudgetDecision.BUDGET_INFEASIBLE
        return Step0FixtureMeasurement(
            fixture_name=fixture_name,
            row_count=row_count,
            max_global_flat_index=-1,
            index_width=-1,
            max_abs_observed=-1,
            rank_bits=-1,
            pos_width=-1,
            full_pack_bits=-1,
            rank_index_bits=-1,
            cap=cap,
            total_numel=total_numel,
            block=block,
            sort_padded_n=_next_power_of_2(row_count),
            row_count_le_block=row_count <= block,
            int63_full_pack_ok=False,
            int63_rank_index_ok=False,
            assert_d_ok=False,
            empty_branch_taken=False,
            decision=decision,
            recommended_stage2_mechanism=recommended_mechanism_for_decision(decision),
            tensor_offsets_non_contiguous=_offsets_non_contiguous(offsets, inputs),
        )

    index_width = budget.index_width
    max_abs_observed = budget.max_abs_observed
    rank_bits = _bit_length_non_negative(max_abs_observed)
    pos_width = _bit_length_non_negative(max(row_count - 1, 0))
    full_pack_bits = rank_bits + index_width + pos_width
    rank_index_bits = rank_bits + index_width
    assert_d_ok = max_abs_observed <= ((2 ** 63 - 1) >> (index_width + pos_width))
    row_count_le_block = row_count <= block
    int63_full_pack_ok = full_pack_bits <= 63
    int63_rank_index_ok = rank_index_bits <= 63
    decision = classify_step0_budget_decision(
        row_count=row_count,
        int63_full_pack_ok=int63_full_pack_ok,
        int63_rank_index_ok=int63_rank_index_ok,
        row_count_le_block=row_count_le_block,
        assert_d_ok=assert_d_ok,
    )

    return Step0FixtureMeasurement(
        fixture_name=fixture_name,
        row_count=row_count,
        max_global_flat_index=budget.max_global_flat_index,
        index_width=index_width,
        max_abs_observed=max_abs_observed,
        rank_bits=rank_bits,
        pos_width=pos_width,
        full_pack_bits=full_pack_bits,
        rank_index_bits=rank_index_bits,
        cap=cap,
        total_numel=total_numel,
        block=block,
        sort_padded_n=_next_power_of_2(row_count),
        row_count_le_block=row_count_le_block,
        int63_full_pack_ok=int63_full_pack_ok,
        int63_rank_index_ok=int63_rank_index_ok,
        assert_d_ok=assert_d_ok,
        empty_branch_taken=False,
        decision=decision,
        recommended_stage2_mechanism=recommended_mechanism_for_decision(decision),
        tensor_offsets_non_contiguous=_offsets_non_contiguous(offsets, inputs),
    )


def build_vote_update_spec(
    *,
    max_abs_per_tensor: int,
    fraction_per_tensor: float = 1.0,
    threshold_abs: int = 10,
) -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=threshold_abs,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=1,
        max_abs_per_tensor=max_abs_per_tensor,
        fraction_per_tensor=fraction_per_tensor,
    )


def build_synthetic_tensor_input(
    *,
    state_key: str,
    numel: int,
    vote_magnitude: int,
    spec: VoteUpdateSpec,
) -> GlobalRateCapTensorInput:
    q = [0] * numel
    acc = [0] * numel
    votes = [vote_magnitude] * numel
    state = VoteUpdateState(
        q_levels=torch.as_tensor(q, dtype=torch.int8),
        accumulators=torch.as_tensor(acc, dtype=torch.int16),
    )
    plan = plan_integer_vote_update_reference(
        state,
        VoteUpdateInputs(votes=torch.as_tensor(votes, dtype=torch.int16)),
        spec,
    )
    return GlobalRateCapTensorInput(state_key=state_key, state=state, plan=plan)


def build_upper_bound_fixture_inputs(
    *,
    numel: int,
    max_abs_per_tensor: int,
    num_states: int,
    fraction_per_tensor: float = 1.0,
) -> list[GlobalRateCapTensorInput]:
    spec = build_vote_update_spec(
        max_abs_per_tensor=max_abs_per_tensor,
        fraction_per_tensor=fraction_per_tensor,
    )
    vote_mag = max(spec.threshold_abs + 1, max_abs_per_tensor)
    return [
        build_synthetic_tensor_input(
            state_key=f"state_{idx}",
            numel=numel,
            vote_magnitude=vote_mag,
            spec=spec,
        )
        for idx in range(num_states)
    ]


def expected_row_count_upper_bound(
    *,
    numel: int,
    max_abs_per_tensor: int,
    num_states: int,
    fraction_per_tensor: float = 1.0,
) -> int:
    per_tensor = min(
        int(max_abs_per_tensor),
        math.ceil(float(fraction_per_tensor) * int(numel)),
    )
    return per_tensor * int(num_states)


def run_step0_budget_fixture_suite(
    fixtures: Iterable[tuple[str, list[GlobalRateCapTensorInput], GlobalRateCapSpec, dict[str, int] | None]],
) -> GlobalRateCapMarginSelectionStep0BudgetReceipt:
    measurements: list[Step0FixtureMeasurement] = []
    for fixture_name, inputs, spec, offsets in fixtures:
        measurements.append(
            measure_step0_fixture(
                fixture_name=fixture_name,
                inputs=inputs,
                spec=spec,
                tensor_offsets=offsets,
            )
        )
    receipt = build_global_rate_cap_margin_selection_step0_budget_receipt(tuple(measurements))
    validate_global_rate_cap_margin_selection_step0_budget_receipt(receipt)
    return receipt


__all__ = [
    "GLOBAL_RATE_CAP_MARGIN_SELECTION_STEP0_BUDGET_SCHEMA_VERSION",
    "GlobalRateCapMarginSelectionStep0BudgetReceipt",
    "Step0BudgetDecision",
    "Step0FixtureMeasurement",
    "TRITON_SINGLE_BLOCK_ROW_CEILING",
    "build_global_rate_cap_margin_selection_step0_budget_receipt",
    "build_synthetic_tensor_input",
    "build_upper_bound_fixture_inputs",
    "build_vote_update_spec",
    "expected_row_count_upper_bound",
    "measure_step0_fixture",
    "run_step0_budget_fixture_suite",
    "validate_global_rate_cap_margin_selection_step0_budget_receipt",
]
