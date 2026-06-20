"""B2-5a Step-0 CPU feasibility tests for the packed total-order key.

CPU-only (no CUDA, no GPU lane env).  Validates the corrected bit-budget
(``index_width = bit_length(max_global_flat_index)``, NOT num_rows-width), the
FIVE hard asserts ((a) upper-bound, (a2) lower-bound / neg-offset reject,
(b) no-duplicate / overlapping-offset reject, (c) rank >= 0,
(d) signed-int64 overflow), the ``empty_all_states`` no-candidate early-return,
the negative_offset_reject path, and the overlap ValueError.  Does NOT mint a
GPU pass; pass-bits remain False on CPU.
"""
from __future__ import annotations

import pytest
import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapOrderingMode,
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_packed_key_scaffold import (
    GlobalRateCapMarginSelectionFeasibilityNull,
    _compute_packed_key_budget,
    _packed_total_order_key,
    _device_row_tensors_for_selection,
    select_global_rate_cap_rows_margin_scaffold,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_feasibility_receipt import (
    validate_global_rate_cap_margin_selection_feasibility_receipt,
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


_CROSS_TIE = lambda: [
    _tensor_input("proj_in", [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 30, 30]),
    _tensor_input("proj_out", [0, 0, 0, 0], [0, 0, 0, 0], [30, 30, 0, 0]),
]
_EMPTY_ALL = lambda: [
    _tensor_input("a", [0, 0], [0, 0], [0, 0]),
    _tensor_input("b", [0, 0], [0, 0], [0, 0]),
]


def _rows(inputs, *, tensor_offsets):
    return _device_row_tensors_for_selection(
        inputs,
        tensor_offsets=tensor_offsets,
        device=torch.device("cpu"),
    )


def test_bit_budget_index_width_uses_max_global_flat_index_not_num_rows():
    inputs = _CROSS_TIE()
    offsets = {"proj_in": 0, "proj_out": 10}
    rows = _rows(inputs, tensor_offsets=offsets)
    budget = _compute_packed_key_budget(
        global_flat_indices=rows["global_indices"],
        abs_new_acc=rows["abs_new_acc"],
    )
    # max_global_flat_index = 11 (offset 10 + flat_index 1) => bit_length(11) = 4
    assert budget.max_global_flat_index == 11
    assert budget.index_width == 4
    assert budget.max_abs_observed == 30


def test_bit_budget_observe_abs_from_fixture_not_assumed_int32_max():
    rows = _rows(_CROSS_TIE(), tensor_offsets={"proj_in": 0, "proj_out": 4})
    budget = _compute_packed_key_budget(
        global_flat_indices=rows["global_indices"], abs_new_acc=rows["abs_new_acc"]
    )
    assert budget.max_abs_observed == 30
    assert budget.max_abs_observed < (2 ** 31)


def test_packed_key_is_strict_total_order_matching_oracle_tiebreak():
    rows = _rows(_CROSS_TIE(), tensor_offsets={"proj_in": 0, "proj_out": 4})
    budget = _compute_packed_key_budget(
        global_flat_indices=rows["global_indices"], abs_new_acc=rows["abs_new_acc"]
    )
    keys = _packed_total_order_key(
        abs_new_acc=rows["abs_new_acc"],
        global_flat_indices=rows["global_indices"],
        budget=budget,
    )
    # Strict total order: ascending key = DESC abs, ASC global_flat_index.
    sorted_keys, order = torch.topk(keys, keys.numel(), largest=False, sorted=True)
    abs_sorted = rows["abs_new_acc"][order]
    gfi_sorted = rows["global_indices"][order]
    # Within equal abs, gfi must be ascending
    for i in range(int(abs_sorted.numel()) - 1):
        assert abs_sorted[i] >= abs_sorted[i + 1]
        if abs_sorted[i] == abs_sorted[i + 1]:
            assert gfi_sorted[i] < gfi_sorted[i + 1]
    assert keys.numel() == rows["global_indices"].numel()


def test_assert_a_upper_bound_index_below_two_pow_width():
    rows = _rows(_CROSS_TIE(), tensor_offsets={"proj_in": 0, "proj_out": 4})
    budget = _compute_packed_key_budget(
        global_flat_indices=rows["global_indices"], abs_new_acc=rows["abs_new_acc"]
    )
    assert torch.all(rows["global_indices"] < (1 << budget.index_width))


def test_assert_a2_rejects_negative_global_flat_index_in_budget():
    # Exercise the budget-function's own lower-bound check directly with a
    # synthetic negative global_flat_index. The wrapper-entry neg-offset
    # reject at gpu.py:534 is covered separately in
    # test_neg_offset_reject_at_wrapper_entry_matches_reference_gpu_line_534.
    gfi = torch.tensor([0, 1, -1, 2], dtype=torch.int64)
    abs_acc = torch.tensor([3, 5, 7, 9], dtype=torch.int64)
    with pytest.raises(ValueError, match="offset"):
        _compute_packed_key_budget(global_flat_indices=gfi, abs_new_acc=abs_acc)


def test_neg_offset_reject_at_wrapper_entry_matches_reference_gpu_line_534():
    inputs = _CROSS_TIE()
    with pytest.raises(ValueError, match=r"must be >= 0"):
        select_global_rate_cap_rows_margin_scaffold(
            inputs,
            GlobalRateCapSpec(cap=2, step=1),
            tensor_offsets={"proj_in": 0, "proj_out": -5},
        )


def test_assert_b_rejects_overlapping_offsets_value_error_pre_pass_mint():
    inputs = [
        _tensor_input("a", [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 30, 30]),
        _tensor_input("b", [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 30, 30]),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        select_global_rate_cap_rows_margin_scaffold(
            inputs,
            GlobalRateCapSpec(cap=2, step=1),
            tensor_offsets={"a": 0, "b": 0},
        )


def test_non_contiguous_non_overlapping_offsets_pass_budget():
    rows = _rows(_CROSS_TIE(), tensor_offsets={"proj_in": 0, "proj_out": 10})
    budget = _compute_packed_key_budget(
        global_flat_indices=rows["global_indices"], abs_new_acc=rows["abs_new_acc"]
    )
    assert budget.index_width == 4
    assert budget.max_global_flat_index == 11


def test_assert_c_feasibility_null_on_all_negative_abs_domain():
    # Assert (c) fires when the entire observed abs domain is negative
    # (a signed-native accumulator bypassing the .abs() at gpu.py:541).
    gfi = torch.tensor([0, 1, 2], dtype=torch.int64)
    abs_all_neg = torch.tensor([-5, -3, -7], dtype=torch.int64)
    with pytest.raises(GlobalRateCapMarginSelectionFeasibilityNull, match=r"assert \(c\)"):
        _compute_packed_key_budget(global_flat_indices=gfi, abs_new_acc=abs_all_neg)


def test_assert_d_feasibility_null_on_signed_int64_overflow():
    gfi = torch.tensor([0, 1], dtype=torch.int64)  # width=1
    abs_huge = torch.tensor([1 << 62, 1 << 62], dtype=torch.int64)
    with pytest.raises(GlobalRateCapMarginSelectionFeasibilityNull, match=r"assert \(d\)"):
        _compute_packed_key_budget(global_flat_indices=gfi, abs_new_acc=abs_huge)


def test_empty_all_states_early_returns_before_bit_budget_arithmetic():
    inputs = _EMPTY_ALL()
    selection, receipt = select_global_rate_cap_rows_margin_scaffold(
        inputs, GlobalRateCapSpec(cap=2, step=1)
    )
    # No bit-budget computed; fabricated-budget guard
    assert receipt.empty_branch_taken is True
    assert receipt.row_count == 0
    assert receipt.observed_max_abs_observed == -1
    assert receipt.observed_index_width == -1
    assert receipt.observed_max_global_flat_index == -1
    assert receipt.feasibility_null is False
    assert receipt.selection_parity_pass is False
    validate_global_rate_cap_margin_selection_feasibility_receipt(receipt)


def test_empty_all_states_distinct_from_empty_one_state():
    # one empty state, one non-empty: bit-budget IS computed
    inputs = [
        _tensor_input("empty", [0, 0], [0, 0], [0, 0]),
        _tensor_input("nonempty", [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 30, 30]),
    ]
    selection, receipt = select_global_rate_cap_rows_margin_scaffold(
        inputs, GlobalRateCapSpec(cap=2, step=1)
    )
    assert receipt.empty_branch_taken is False
    assert receipt.row_count == 2
    assert receipt.observed_max_abs_observed == 30
    assert receipt.observed_index_width >= 1
    validate_global_rate_cap_margin_selection_feasibility_receipt(receipt)


def test_cpu_receipt_pass_bits_default_false_until_gpu_parity_launch():
    inputs = _CROSS_TIE()
    _, receipt = select_global_rate_cap_rows_margin_scaffold(
        inputs, GlobalRateCapSpec(cap=3, step=1)
    )
    assert receipt.selection_parity_pass is False
    assert receipt.negative_offset_reject_evidence is True
    validate_global_rate_cap_margin_selection_feasibility_receipt(receipt)


def test_receipt_non_claims_do_not_mint_forbidden_claims():
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_packed_key_scaffold import (
        GLOBAL_RATE_CAP_MARGIN_SELECTION_SCAFFOLD_NON_CLAIMS as scaffold_claims,
    )
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_feasibility_receipt import (
        GLOBAL_RATE_CAP_MARGIN_SELECTION_FEASIBILITY_NON_CLAIMS as receipt_claims,
    )
    blob = " ".join(scaffold_claims) + " " + " ".join(receipt_claims)
    forbidden_mentions = [
        term
        for term in (
            "flip global_cap_margin_only_reference",
            "touch the deferred-backlog ledger",
            "mutate q_levels / new_acc_i32 / accumulators",
        )
    ]
    # The non-claims tuple asserts these actions are NOT performed (negation).
    for phrase in forbidden_mentions:
        assert phrase.lower() in blob.lower()
    # Sanity: at least one non-claim asserts no global_cap flip.
    assert any("global_cap_margin_only_reference" in c for c in receipt_claims)


def test_native_named_builder_fail_closed() -> None:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_feasibility_receipt import (
        build_global_rate_cap_margin_selection_native_parity_receipt,
    )

    with pytest.raises(RuntimeError, match="fail-closed"):
        build_global_rate_cap_margin_selection_native_parity_receipt()
