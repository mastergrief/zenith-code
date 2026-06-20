"""B2-5a′ Stage-1 Step-0 budget/shape receipt tests (CPU-only, no kernel).

Validates canonical cross-tie, non-contiguous offsets, upper-bound sweep,
cap annotation, and the empty-row early branch (no fabricated bit-budget).
"""
from __future__ import annotations

import pytest
import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_step0_budget import (
    TRITON_SINGLE_BLOCK_ROW_CEILING,
    build_synthetic_tensor_input,
    build_upper_bound_fixture_inputs,
    build_vote_update_spec,
    expected_row_count_upper_bound,
    measure_step0_fixture,
    run_step0_budget_fixture_suite,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_step0_budget_receipt import (
    STAGE2_MECHANISM_EMPTY_ROWS,
    STAGE2_MECHANISM_MULTIBLOCK_DEFERRED,
    Step0BudgetDecision,
    validate_global_rate_cap_margin_selection_step0_budget_receipt,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
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


def _state(q, acc) -> VoteUpdateState:
    return VoteUpdateState(
        q_levels=torch.as_tensor(q, dtype=torch.int8),
        accumulators=torch.as_tensor(acc, dtype=torch.int16),
    )


def _inputs(votes, **kwargs) -> VoteUpdateInputs:
    converted = {}
    for name, value in kwargs.items():
        converted[name] = (
            None
            if value is None
            else torch.as_tensor(
                value, dtype=torch.int8 if name.endswith("moves") else torch.int16
            )
        )
    return VoteUpdateInputs(
        votes=torch.as_tensor(votes, dtype=torch.int16), **converted
    )


def _tensor_input(state_key, q, acc, votes, **vote_kwargs) -> GlobalRateCapTensorInput:
    state = _state(q, acc)
    return GlobalRateCapTensorInput(
        state_key=state_key,
        state=state,
        plan=plan_integer_vote_update_reference(
            state, _inputs(votes, **vote_kwargs), _spec()
        ),
    )


def _cross_tie_inputs() -> list[GlobalRateCapTensorInput]:
    return [
        _tensor_input("proj_in", [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 30, 30]),
        _tensor_input("proj_out", [0, 0, 0, 0], [0, 0, 0, 0], [30, 30, 0, 0]),
    ]


def _empty_all_inputs() -> list[GlobalRateCapTensorInput]:
    return [
        _tensor_input("a", [0, 0], [0, 0], [0, 0]),
        _tensor_input("b", [0, 0], [0, 0], [0, 0]),
    ]


def test_empty_row_branch_documents_null_without_fake_budget():
    measurement = measure_step0_fixture(
        fixture_name="empty_all_states",
        inputs=_empty_all_inputs(),
        spec=GlobalRateCapSpec(cap=512, step=1),
    )
    assert measurement.empty_branch_taken is True
    assert measurement.row_count == 0
    assert measurement.max_global_flat_index == -1
    assert measurement.index_width == -1
    assert measurement.max_abs_observed == -1
    assert measurement.decision == Step0BudgetDecision.EMPTY_ROWS
    assert measurement.recommended_stage2_mechanism == STAGE2_MECHANISM_EMPTY_ROWS


def test_canonical_cross_tie_fixture_measures_budget_fields():
    measurement = measure_step0_fixture(
        fixture_name="canonical_cross_tie",
        inputs=_cross_tie_inputs(),
        spec=GlobalRateCapSpec(cap=512, step=1),
        tensor_offsets={"proj_in": 0, "proj_out": 4},
    )
    assert measurement.row_count == 4
    assert measurement.index_width == 3
    assert measurement.max_global_flat_index == 5
    assert measurement.cap == 512
    assert measurement.row_count < measurement.cap
    assert measurement.empty_branch_taken is False
    assert measurement.decision in (
        Step0BudgetDecision.FULL_PACK_VIABLE,
        Step0BudgetDecision.KEY_PAYLOAD_REQUIRED,
    )


def test_non_contiguous_offsets_fixture_flags_non_contiguous():
    measurement = measure_step0_fixture(
        fixture_name="non_contiguous_offsets",
        inputs=_cross_tie_inputs(),
        spec=GlobalRateCapSpec(cap=1024, step=1),
        tensor_offsets={"proj_in": 0, "proj_out": 10},
    )
    assert measurement.tensor_offsets_non_contiguous is True
    assert measurement.index_width == 4
    assert measurement.max_global_flat_index == 11


def test_cap_annotation_records_cap_without_bounding_row_count():
    measurement = measure_step0_fixture(
        fixture_name="cap_annotation",
        inputs=_cross_tie_inputs(),
        spec=GlobalRateCapSpec(cap=0, step=1),
        tensor_offsets={"proj_in": 0, "proj_out": 4},
    )
    assert measurement.cap == 0
    assert measurement.row_count == 4


@pytest.mark.parametrize(
    ("numel", "max_abs", "num_states"),
    [
        (4096, 256, 8),
        (65536, 256, 5),
    ],
)
def test_upper_bound_sweep_can_exceed_block_while_cap_stays_small(numel, max_abs, num_states):
    inputs = build_upper_bound_fixture_inputs(
        numel=numel,
        max_abs_per_tensor=max_abs,
        num_states=num_states,
    )
    expected_rows = expected_row_count_upper_bound(
        numel=numel,
        max_abs_per_tensor=max_abs,
        num_states=num_states,
    )
    measurement = measure_step0_fixture(
        fixture_name=f"upper_bound_{numel}_{max_abs}_{num_states}",
        inputs=inputs,
        spec=GlobalRateCapSpec(cap=512, step=1),
    )
    assert measurement.row_count == expected_rows
    assert measurement.row_count > measurement.cap
    assert measurement.row_count > TRITON_SINGLE_BLOCK_ROW_CEILING
    assert measurement.decision == Step0BudgetDecision.ROW_COUNT_MULTIBLOCK_DEFERRED
    assert measurement.recommended_stage2_mechanism == STAGE2_MECHANISM_MULTIBLOCK_DEFERRED


def test_fixture_suite_produces_worst_case_and_validates_receipt():
    fixtures = [
        (
            "canonical_cross_tie",
            _cross_tie_inputs(),
            GlobalRateCapSpec(cap=512, step=1),
            {"proj_in": 0, "proj_out": 4},
        ),
        (
            "empty_all_states",
            _empty_all_inputs(),
            GlobalRateCapSpec(cap=512, step=1),
            None,
        ),
        (
            "upper_bound_multiblock",
            build_upper_bound_fixture_inputs(numel=4096, max_abs_per_tensor=256, num_states=8),
            GlobalRateCapSpec(cap=1024, step=1),
            None,
        ),
    ]
    receipt = run_step0_budget_fixture_suite(fixtures)
    validate_global_rate_cap_margin_selection_step0_budget_receipt(receipt)
    assert receipt.worst_case_decision == Step0BudgetDecision.ROW_COUNT_MULTIBLOCK_DEFERRED
    assert len(receipt.fixture_measurements) == 3


def test_stage1_sources_contain_zero_triton_jit_decorator():
    from calm.hrm_text_158.native_full_stack import (
        global_rate_cap_margin_selection_step0_budget as budget_mod,
        global_rate_cap_margin_selection_step0_budget_receipt as receipt_mod,
    )

    for mod in (budget_mod, receipt_mod):
        source = open(mod.__file__, encoding="utf-8").read()
        assert "import triton" not in source
        assert not any(
            line.strip().startswith("@triton.jit") for line in source.splitlines()
        )
