"""B2-5a″ Stage-A Step-0′ width classifier tests (CPU budget + assertions; compile probe optional)."""
from __future__ import annotations

import ast
import py_compile
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap import GlobalRateCapSpec
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_multiblock_step0_budget import (
    REALISTIC_ROW_COUNTS,
    WIDER_SINGLE_BLOCK_SORT_PADDED_N,
    assert_global_pos_permutation,
    assert_merge_half_capacity_for_row_count,
    assert_probe_width_pow2_for_bucket,
    assert_staging_layout_correctness,
    build_realistic_fixture_inputs,
    check_merge_half_capacity,
    measure_multiblock_step0_fixture,
    run_multiblock_step0_budget_suite,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_multiblock_step0_budget_receipt import (
    GLOBAL_RATE_CAP_MARGIN_SELECTION_MULTIBLOCK_STEP0_NON_CLAIMS,
    MultiblockStep0Decision,
    STAGE_B_MECHANISM_MULTIBLOCK_MERGE,
    validate_global_rate_cap_margin_selection_multiblock_step0_budget_receipt,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_multiblock_step0_compile_probe_kernel import (
    _kernel_file_path_for_test,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_path_audit import (
    audit_native_path_module_source,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_triton_kernel import (
    PADDING_SENTINEL,
)

_NATIVE_STACK = Path(__file__).parent.parent.parent / "hrm_text_158" / "native_full_stack"


@pytest.mark.parametrize("row_count", REALISTIC_ROW_COUNTS)
def test_realistic_fixture_row_counts(row_count: int) -> None:
    inputs = build_realistic_fixture_inputs(target_row_count=row_count)
    measurement = measure_multiblock_step0_fixture(
        fixture_name=f"realistic_{row_count}",
        inputs=inputs,
        spec=GlobalRateCapSpec(cap=1024, step=1),
    )
    assert measurement.row_count == row_count
    assert measurement.sort_padded_n == WIDER_SINGLE_BLOCK_SORT_PADDED_N
    assert measurement.int63_full_pack_ok is True
    assert measurement.budget_infeasible is False


def test_assertion_global_pos_permutation() -> None:
    inputs = build_realistic_fixture_inputs(target_row_count=1280)
    from calm.hrm_text_158.native_full_stack.global_rate_cap import (
        tensor_offsets_for_vote_update_states,
    )
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_packed_key_scaffold import (
        _device_row_tensors_for_selection,
    )

    offsets = tensor_offsets_for_vote_update_states(inputs)
    rows = _device_row_tensors_for_selection(inputs, tensor_offsets=offsets, device=torch.device("cpu"))
    measurement = measure_multiblock_step0_fixture(
        fixture_name="realistic_1280",
        inputs=inputs,
        spec=GlobalRateCapSpec(cap=1024, step=1),
    )
    assert assert_global_pos_permutation(
        abs_new_acc=rows["abs_new_acc"],
        global_flat_indices=rows["global_indices"],
        max_abs_observed=measurement.max_abs_observed,
        index_width=measurement.index_width,
        global_row_count=1280,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("row_count", REALISTIC_ROW_COUNTS)
def test_assertion_merge_half_capacity(row_count: int) -> None:
    assert assert_merge_half_capacity_for_row_count(row_count) is True


def test_half_capacity_guard_counterexample() -> None:
    assert check_merge_half_capacity(n_a=1024, n_b=256) is True
    assert check_merge_half_capacity(n_a=2049, n_b=1) is False


def test_assertion_staging_layout_good_and_bad() -> None:
    good_ok, bad_fails = assert_staging_layout_correctness()
    assert good_ok is True
    assert bad_fails is True


@pytest.mark.parametrize("row_count", REALISTIC_ROW_COUNTS)
def test_assertion_probe_width_pow2(row_count: int) -> None:
    assert assert_probe_width_pow2_for_bucket(
        row_count=row_count,
        sort_padded_n=WIDER_SINGLE_BLOCK_SORT_PADDED_N,
    )


def test_suite_cpu_classifies_without_compile_probe() -> None:
    receipt = run_multiblock_step0_budget_suite(run_gpu_compile_probe=False)
    validate_global_rate_cap_margin_selection_multiblock_step0_budget_receipt(receipt)
    assert receipt.assertion_results is not None
    assert receipt.assertion_results.global_pos_permutation_ok is True
    assert receipt.assertion_results.merge_half_capacity_ok is True
    assert receipt.assertion_results.staging_layout_ok is True
    assert receipt.assertion_results.bad_staging_layout_fails is True
    assert receipt.assertion_results.probe_width_pow2_ok is True
    assert len(receipt.fixture_measurements) == 3
    assert all(not p.compile_probe_executed for p in receipt.compile_probes)
    assert receipt.decision == MultiblockStep0Decision.MERGE_TREE_REQUIRED
    assert receipt.recommended_stage_b_mechanism == STAGE_B_MECHANISM_MULTIBLOCK_MERGE
    assert any("NOT mint selection_parity_pass" in c for c in receipt.non_claims)


def test_receipt_rejects_illegal_1536_probe_width() -> None:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_multiblock_step0_budget_receipt import (
        CompileProbeMeasurement,
        GlobalRateCapMarginSelectionMultiblockStep0BudgetReceipt,
        MultiblockStep0AssertionResults,
    )

    receipt = GlobalRateCapMarginSelectionMultiblockStep0BudgetReceipt(
        compile_probes=(
            CompileProbeMeasurement(
                sort_padded_n=1536,
                n_rows_sample=1280,
                compile_probe_executed=True,
                compile_ok=True,
                kernel_symbol="bad",
            ),
        ),
        assertion_results=MultiblockStep0AssertionResults(
            global_pos_permutation_ok=True,
            merge_half_capacity_ok=True,
            staging_layout_ok=True,
            bad_staging_layout_fails=True,
            probe_width_pow2_ok=True,
        ),
    )
    with pytest.raises(ValueError, match="1536"):
        validate_global_rate_cap_margin_selection_multiblock_step0_budget_receipt(receipt)


def test_stage_a_budget_and_receipt_modules_have_no_triton_jit() -> None:
    from calm.hrm_text_158.native_full_stack import (
        global_rate_cap_margin_selection_multiblock_step0_budget as budget_mod,
        global_rate_cap_margin_selection_multiblock_step0_budget_receipt as receipt_mod,
    )

    for mod in (budget_mod, receipt_mod):
        source = open(mod.__file__, encoding="utf-8").read()
        assert "import triton" not in source
        assert not any(line.strip().startswith("@triton.jit") for line in source.splitlines())


def test_probe_kernel_source_has_ji_lt_ki_guard() -> None:
    source = _kernel_file_path_for_test.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "if ji < ki:" in source
    findings = audit_native_path_module_source(source, filepath=str(_kernel_file_path_for_test))
    assert not any(f.kind == "static_denylist" for f in findings)


def test_staging_half_capacity_example_matches_plan() -> None:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_multiblock_step0_budget import (
        _half_capacity_stage_workspace,
    )

    ws = _half_capacity_stage_workspace(
        keys_a=[1, 4],
        keys_b=[2],
        sort_padded_n=4,
        padding_sentinel=PADDING_SENTINEL,
    )
    assert ws == [1, 4, PADDING_SENTINEL, 2]


def test_new_modules_py_compile() -> None:
    paths = (
        _NATIVE_STACK / "global_rate_cap_margin_selection_multiblock_step0_budget.py",
        _NATIVE_STACK / "global_rate_cap_margin_selection_multiblock_step0_budget_receipt.py",
        _NATIVE_STACK / "global_rate_cap_margin_selection_multiblock_step0_compile_probe_kernel.py",
    )
    for path in paths:
        py_compile.compile(str(path), doraise=True)


def test_non_claims_include_no_parity_mint() -> None:
    assert any("NOT mint selection_parity_pass" in c for c in GLOBAL_RATE_CAP_MARGIN_SELECTION_MULTIBLOCK_STEP0_NON_CLAIMS)
    assert any("NOT runtime sort parity" in c for c in GLOBAL_RATE_CAP_MARGIN_SELECTION_MULTIBLOCK_STEP0_NON_CLAIMS)
