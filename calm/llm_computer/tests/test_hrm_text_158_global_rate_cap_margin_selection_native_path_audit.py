"""B2-5a′ Stage-2 §4.5 native-path audit tests (CPU-only).

Five violation fixtures must FAIL the audit; real native modules must PASS.
"""
from __future__ import annotations

import ast
import hashlib
import py_compile
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_path_audit import (
    audit_native_path_module_source,
    run_full_native_path_audit,
)

_NATIVE_STACK = Path(__file__).parent.parent.parent / "hrm_text_158" / "native_full_stack"

_REAL_MODULES = (
    _NATIVE_STACK / "global_rate_cap_margin_selection_triton_kernel.py",
    _NATIVE_STACK / "global_rate_cap_margin_selection_native_dispatch.py",
    _NATIVE_STACK / "global_rate_cap_margin_selection_native_parity_receipt.py",
    _NATIVE_STACK / "global_rate_cap_margin_selection_native_path_audit.py",
)


def test_real_native_modules_static_audit_pass() -> None:
    result = run_full_native_path_audit(module_paths=_REAL_MODULES, tile=None)
    assert result.static_clean, result.static_findings
    assert result.native_path_audit_pass or not result.dynamic_complete


def test_violation_row_tensor_perm() -> None:
    src = """
def bad(row_tensor, perm):
    return row_tensor[perm]
"""
    findings = audit_native_path_module_source(src, filepath="violation_row_tensor_perm.py")
    assert any(f.kind == "ast_subscript_slice_ref" for f in findings)


def test_violation_alias_local_perm() -> None:
    src = """
def bad(rows):
    local = rows["global_indices"]
    return local[perm]
"""
    findings = audit_native_path_module_source(src, filepath="violation_alias.py")
    assert any(f.kind == "ast_subscript_slice_ref" for f in findings)


def test_violation_helper_wrapped_subscript() -> None:
    src = """
def _helper(tensor, idx):
    return tensor[idx]

def bad(rows, perm):
    return _helper(rows["state_ids"], perm)
"""
    findings = audit_native_path_module_source(src, filepath="violation_helper.py")
    assert any(
        f.kind in ("ast_subscript_slice_ref", "ast_call_sort_path_arg") for f in findings
    )


def test_violation_gather() -> None:
    src = """
import torch
def bad(rows, perm):
    return torch.gather(rows["state_ids"], 0, perm)
"""
    findings = audit_native_path_module_source(src, filepath="violation_gather.py")
    assert any(f.kind == "static_denylist" for f in findings)


def test_violation_nonzero_order() -> None:
    src = """
def bad(mask, order):
    return mask.nonzero()[order]
"""
    findings = audit_native_path_module_source(src, filepath="violation_nonzero.py")
    assert any(f.kind in ("static_denylist", "ast_subscript_slice_ref") for f in findings)


def test_kernel_source_sha256_live_file() -> None:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_triton_kernel import (
        _kernel_file_path_for_test,
        _kernel_source_sha256,
    )

    expected = hashlib.sha256(_kernel_file_path_for_test.read_bytes()).hexdigest()
    assert _kernel_source_sha256() == expected


def test_native_modules_py_compile() -> None:
    for path in _REAL_MODULES:
        py_compile.compile(str(path), doraise=True)


def test_sentinel_headroom_case_b_fail_closed() -> None:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_triton_kernel import (
        INT64_MAX,
        evaluate_padding_headroom,
    )

    headroom = evaluate_padding_headroom(host_max_full_key=INT64_MAX, full_pack_bits=40)
    assert headroom["budget_infeasible"] is True
    assert headroom["padding_headroom_ok"] is False


def test_sentinel_headroom_case_a_python_int() -> None:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_triton_kernel import (
        INT64_HEADROOM_LIMIT,
        evaluate_padding_headroom,
    )

    headroom = evaluate_padding_headroom(host_max_full_key=INT64_HEADROOM_LIMIT, full_pack_bits=38)
    assert headroom["budget_infeasible"] is False
    assert headroom["padding_headroom_ok"] is True


def test_violation_perm_to_long() -> None:
    src = """
def bad(row_tensor, perm):
    return row_tensor[perm.to(torch.long)]
"""
    findings = audit_native_path_module_source(src, filepath="violation_perm_to.py")
    assert any(f.kind == "ast_subscript_slice_ref" for f in findings)


def test_violation_alias_helper_param() -> None:
    src = """
def _helper(t, p2):
    return t[p2]

def bad(rows, perm):
    p2 = perm
    return _helper(rows["state_ids"], p2)
"""
    findings = audit_native_path_module_source(src, filepath="violation_alias_helper.py")
    assert any(f.kind == "ast_call_sort_path_arg" for f in findings)


def test_cap_split_arange_not_flagged() -> None:
    src = """
import torch
accepted_positions = torch.arange(4)
def ok(rows):
    return rows["state_ids"][accepted_positions]
"""
    findings = audit_native_path_module_source(src, filepath="allowed_cap_split.py")
    assert findings == []


def _attach_parent_links(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            setattr(child, "parent", node)


def _is_ki_minus_ji_minus_one_shift(node: ast.BinOp) -> bool:
    if not isinstance(node.op, ast.LShift):
        return False
    if not (isinstance(node.left, ast.Constant) and node.left.value == 1):
        return False
    right = node.right
    return (
        isinstance(right, ast.BinOp)
        and isinstance(right.op, ast.Sub)
        and isinstance(right.left, ast.BinOp)
        and isinstance(right.left.op, ast.Sub)
        and isinstance(right.left.left, ast.Name)
        and right.left.left.id == "ki"
        and isinstance(right.left.right, ast.Name)
        and right.left.right.id == "ji"
        and isinstance(right.right, ast.Constant)
        and right.right.value == 1
    )


def _under_ji_lt_ki_guard(node: ast.AST) -> bool:
    cur: ast.AST | None = node
    while cur is not None:
        if isinstance(cur, ast.If):
            test = cur.test
            if (
                isinstance(test, ast.Compare)
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Lt)
                and len(test.comparators) == 1
                and isinstance(test.left, ast.Name)
                and test.left.id == "ji"
                and isinstance(test.comparators[0], ast.Name)
                and test.comparators[0].id == "ki"
            ):
                return True
        cur = getattr(cur, "parent", None)  # type: ignore[arg-type]
    return False


def _find_function_def(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _collect_ki_ji_shifts_in_function(func_def: ast.FunctionDef) -> list[ast.BinOp]:
    shifts: list[ast.BinOp] = []

    class _ShiftFinder(ast.NodeVisitor):
        def visit_BinOp(self, node: ast.BinOp) -> None:
            if _is_ki_minus_ji_minus_one_shift(node):
                shifts.append(node)
            self.generic_visit(node)

    _ShiftFinder().visit(func_def)
    return shifts


def _assert_bitonic_kernel_function_shifts_all_guarded(source: str) -> None:
    tree = ast.parse(source)
    _attach_parent_links(tree)
    func_def = _find_function_def(tree, "_margin_selection_bitonic_sort_kernel")
    assert func_def is not None, "expected _margin_selection_bitonic_sort_kernel FunctionDef"
    shifts = _collect_ki_ji_shifts_in_function(func_def)
    assert shifts, "expected >=1 ki-ji-1 shift inside bitonic sort kernel only"
    unguarded = [n for n in shifts if not _under_ji_lt_ki_guard(n)]
    assert not unguarded, (
        "every bitonic-kernel `1 << (ki - ji - 1)` must be nested under `if ji < ki`"
    )


def test_bitonic_kernel_static_loop_negative_shift_guarded() -> None:
    """CPU gate-1 check: kernel static_range(10)+`if ji < ki` never evaluates negative shift."""

    max_log2n = 10
    unguarded_negative: list[tuple[int, int]] = []
    guarded_shift_counts: list[int] = []
    for ki in range(1, max_log2n + 1):
        for ji in range(max_log2n):
            shift_count = ki - ji - 1
            if ji >= ki:
                if shift_count < 0:
                    unguarded_negative.append((ki, ji))
                continue
            assert shift_count >= 0, f"guarded shift negative at ki={ki} ji={ji}"
            guarded_shift_counts.append(shift_count)
            _ = 1 << shift_count

    assert unguarded_negative, "unguarded path must include negative-shift pairs"
    assert (1, 1) in unguarded_negative
    assert all(sc >= 0 for sc in guarded_shift_counts)


def test_bitonic_kernel_source_shift_nested_under_ji_lt_ki_guard() -> None:
    """AST gate-1: kernel-only `1 << (ki - ji - 1)` sites are all under `if ji < ki`."""

    kernel_path = (
        Path(__file__).parent.parent.parent
        / "hrm_text_158"
        / "native_full_stack"
        / "global_rate_cap_margin_selection_triton_kernel.py"
    )
    _assert_bitonic_kernel_function_shifts_all_guarded(
        kernel_path.read_text(encoding="utf-8")
    )


def test_bitonic_kernel_source_shift_guard_fails_on_unguarded_kernel_variant() -> None:
    """Self-falsification: test must fail when kernel guard removed (reference guard alone insufficient)."""

    guarded_reference_only = """
def bitonic_sort_single_writer_reference():
    for ki in range(1, 11):
        for ji in range(10):
            if ji < ki:
                j = 1 << (ki - ji - 1)

def _margin_selection_bitonic_sort_kernel():
    for ki in range(1, 11):
        for ji in range(10):
            j = 1 << (ki - ji - 1)
"""
    with pytest.raises(AssertionError, match="if ji < ki"):
        _assert_bitonic_kernel_function_shifts_all_guarded(guarded_reference_only)


def test_bitonic_single_writer_reference_sorts_random_and_padded() -> None:
    import random

    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_triton_kernel import (
        PADDING_SENTINEL,
        _next_power_of_2,
        bitonic_sort_single_writer_reference,
    )

    random.seed(17)
    for n in (1, 3, 7, 16, 31, 64, 1024):
        keys = [random.randint(0, 10_000) for _ in range(n)]
        padded = _next_power_of_2(n)
        out = bitonic_sort_single_writer_reference(
            keys, sort_padded_n=padded, padding_sentinel=PADDING_SENTINEL
        )
        assert out[:n] == sorted(keys)
        assert all(v == PADDING_SENTINEL for v in out[n:padded])


def test_native_parity_receipt_refuses_pass_without_audit() -> None:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_parity_receipt import (
        NativeSelectionParityProof,
        build_global_rate_cap_margin_selection_native_parity_receipt,
        new_native_selection_token,
    )

    token = new_native_selection_token(
        bitonic_kernel_symbol="k1",
        gather_kernel_symbol="k2",
        kernel_source_sha256="a" * 64,
        selection_input_sha256="b" * 64,
        ordered_output_sha256="c" * 64,
        accepted_output_sha256="d" * 64,
        deferred_output_sha256="e" * 64,
    )
    with pytest.raises(ValueError, match="native_path_audit_pass"):
        build_global_rate_cap_margin_selection_native_parity_receipt(
            selection_parity_pass=True,
            single_block_regime=True,
            native_path_audit_pass=False,
            kernel_output_buffers_emitted=True,
            parity_proof=NativeSelectionParityProof(
                parity_ok=True,
                ordered_global_indices_match=True,
                accepted_positions_match=True,
                deferred_positions_match=True,
            ),
            token=token,
        )

    with pytest.raises(ValueError, match="parity_proof"):
        build_global_rate_cap_margin_selection_native_parity_receipt(
            selection_parity_pass=True,
            single_block_regime=True,
            native_path_audit_pass=True,
            kernel_output_buffers_emitted=True,
            token=token,
        )


_WIDER_KERNEL = (
    _NATIVE_STACK / "global_rate_cap_margin_selection_wider_single_block_triton_kernel.py"
)
_WIDER_COMPOSE = (
    _NATIVE_STACK / "global_rate_cap_margin_selection_wider_single_block_compose.py"
)


def test_wider_kernel_modules_static_audit_pass() -> None:
    for path in (_WIDER_KERNEL, _WIDER_COMPOSE):
        findings = audit_native_path_module_source(
            path.read_text(encoding="utf-8"),
            filepath=str(path),
        )
        assert findings == []


def test_wider_host_stage_schedule_negative_shift_self_falsification() -> None:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_wider_single_block_triton_kernel import (
        WIDER_MAX_LOG2N,
    )

    unguarded_negative: list[tuple[int, int]] = []
    for ki in range(1, WIDER_MAX_LOG2N + 1):
        for ji in range(WIDER_MAX_LOG2N):
            shift = ki - ji - 1
            if ji < ki:
                assert shift >= 0
            elif shift < 0:
                unguarded_negative.append((ki, ji))
    assert unguarded_negative
    assert (1, 1) in unguarded_negative


def test_wider_stage_kernel_module_static_audit_pass() -> None:
    findings = audit_native_path_module_source(
        _WIDER_KERNEL.read_text(encoding="utf-8"),
        filepath=str(_WIDER_KERNEL),
    )
    assert findings == []


def test_wider_bitonic_reference_sorts_sample_keys() -> None:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_wider_single_block_triton_kernel import (
        bitonic_sort_single_writer_reference_wide,
    )

    keys = [5, 3, 8, 1, 9, 2, 7, 4]
    sorted_ref = bitonic_sort_single_writer_reference_wide(
        keys,
        sort_padded_n=8,
    )[: len(keys)]
    assert sorted_ref == sorted(keys)


def test_wider_regime_pass_mint_requires_row_count_gt_block() -> None:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_parity_receipt import (
        NativeSelectionParityProof,
        RuntimeSortKeyProof,
        build_global_rate_cap_margin_selection_native_parity_receipt,
        new_native_selection_token,
        validate_global_rate_cap_margin_selection_native_parity_receipt,
    )

    token = new_native_selection_token(
        bitonic_kernel_symbol="w",
        gather_kernel_symbol="g",
        kernel_source_sha256="a" * 64,
        selection_input_sha256="b" * 64,
        ordered_output_sha256="c" * 64,
        accepted_output_sha256="d" * 64,
        deferred_output_sha256="e" * 64,
    )
    sort_proof = RuntimeSortKeyProof(
        sort_padded_n=1024,
        n_rows=512,
        native_sorted_sha256="f" * 64,
        cpu_ref_sha256="f" * 64,
        exact=True,
    )
    with pytest.raises(ValueError, match="valid single_block or wider_single_block regime"):
        build_global_rate_cap_margin_selection_native_parity_receipt(
            selection_parity_pass=True,
            single_block_regime=False,
            wider_single_block_regime=True,
            row_count=512,
            sort_padded_n=1024,
            native_path_audit_pass=True,
            kernel_output_buffers_emitted=True,
            runtime_sort_key_exact=True,
            realistic_size_proven=True,
            runtime_sort_key_proof=sort_proof,
            parity_proof=NativeSelectionParityProof(
                parity_ok=True,
                ordered_global_indices_match=True,
                accepted_positions_match=True,
                deferred_positions_match=True,
            ),
            token=token,
        )


def _wider_pass_mint_token_and_proof() -> tuple[object, NativeSelectionParityProof, RuntimeSortKeyProof]:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_parity_receipt import (
        NativeSelectionParityProof,
        RuntimeSortKeyProof,
        new_native_selection_token,
    )

    token = new_native_selection_token(
        bitonic_kernel_symbol="w",
        gather_kernel_symbol="g",
        kernel_source_sha256="a" * 64,
        selection_input_sha256="b" * 64,
        ordered_output_sha256="c" * 64,
        accepted_output_sha256="d" * 64,
        deferred_output_sha256="e" * 64,
    )
    parity_proof = NativeSelectionParityProof(
        parity_ok=True,
        ordered_global_indices_match=True,
        accepted_positions_match=True,
        deferred_positions_match=True,
    )
    sort_proof = RuntimeSortKeyProof(
        sort_padded_n=2048,
        n_rows=1280,
        native_sorted_sha256="f" * 64,
        cpu_ref_sha256="f" * 64,
        exact=True,
    )
    return token, parity_proof, sort_proof


def test_wider_pass_mint_rejected_without_runtime_sort_key_exact() -> None:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_parity_receipt import (
        build_global_rate_cap_margin_selection_native_parity_receipt,
    )

    token, parity_proof, sort_proof = _wider_pass_mint_token_and_proof()
    with pytest.raises(ValueError, match="runtime_sort_key_exact"):
        build_global_rate_cap_margin_selection_native_parity_receipt(
            selection_parity_pass=True,
            wider_single_block_regime=True,
            row_count=1280,
            native_path_audit_pass=True,
            kernel_output_buffers_emitted=True,
            runtime_sort_key_exact=False,
            realistic_size_proven=True,
            parity_proof=parity_proof,
            token=token,
        )


def test_wider_pass_mint_rejected_without_realistic_size_proven() -> None:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_parity_receipt import (
        build_global_rate_cap_margin_selection_native_parity_receipt,
    )

    token, parity_proof, sort_proof = _wider_pass_mint_token_and_proof()
    with pytest.raises(ValueError, match="realistic_size_proven"):
        build_global_rate_cap_margin_selection_native_parity_receipt(
            selection_parity_pass=True,
            wider_single_block_regime=True,
            row_count=1280,
            sort_padded_n=2048,
            native_path_audit_pass=True,
            kernel_output_buffers_emitted=True,
            runtime_sort_key_exact=True,
            realistic_size_proven=False,
            runtime_sort_key_proof=sort_proof,
            parity_proof=parity_proof,
            token=token,
        )


def test_wider_pass_mint_accepted_with_runtime_sort_proof_and_realistic_size() -> None:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_parity_receipt import (
        build_global_rate_cap_margin_selection_native_parity_receipt,
        validate_global_rate_cap_margin_selection_native_parity_receipt,
    )

    token, parity_proof, sort_proof = _wider_pass_mint_token_and_proof()
    receipt = build_global_rate_cap_margin_selection_native_parity_receipt(
        selection_parity_pass=True,
        wider_single_block_regime=True,
        row_count=1280,
        sort_padded_n=2048,
        native_path_audit_pass=True,
        kernel_output_buffers_emitted=True,
        runtime_sort_key_exact=True,
        realistic_size_proven=True,
        runtime_sort_key_proof=sort_proof,
        parity_proof=parity_proof,
        token=token,
    )
    validate_global_rate_cap_margin_selection_native_parity_receipt(receipt)


def test_wider_pass_mint_rejected_when_proof_n_rows_mismatch() -> None:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_parity_receipt import (
        build_global_rate_cap_margin_selection_native_parity_receipt,
    )

    token, parity_proof, sort_proof = _wider_pass_mint_token_and_proof()
    with pytest.raises(ValueError, match="n_rows must match receipt row_count"):
        build_global_rate_cap_margin_selection_native_parity_receipt(
            selection_parity_pass=True,
            wider_single_block_regime=True,
            row_count=999,
            sort_padded_n=2048,
            native_path_audit_pass=True,
            kernel_output_buffers_emitted=True,
            runtime_sort_key_exact=True,
            realistic_size_proven=True,
            runtime_sort_key_proof=sort_proof,
            parity_proof=parity_proof,
            token=token,
        )


def test_wider_pass_mint_rejected_when_proof_sort_padded_n_mismatch() -> None:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_parity_receipt import (
        build_global_rate_cap_margin_selection_native_parity_receipt,
    )

    token, parity_proof, sort_proof = _wider_pass_mint_token_and_proof()
    with pytest.raises(ValueError, match="sort_padded_n must match receipt sort_padded_n"):
        build_global_rate_cap_margin_selection_native_parity_receipt(
            selection_parity_pass=True,
            wider_single_block_regime=True,
            row_count=1280,
            sort_padded_n=4096,
            native_path_audit_pass=True,
            kernel_output_buffers_emitted=True,
            runtime_sort_key_exact=True,
            realistic_size_proven=True,
            runtime_sort_key_proof=sort_proof,
            parity_proof=parity_proof,
            token=token,
        )


def test_wider_pass_mint_rejected_when_receipt_sort_padded_n_default() -> None:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_parity_receipt import (
        build_global_rate_cap_margin_selection_native_parity_receipt,
    )

    token, parity_proof, sort_proof = _wider_pass_mint_token_and_proof()
    with pytest.raises(ValueError, match="sort_padded_n must match receipt sort_padded_n"):
        build_global_rate_cap_margin_selection_native_parity_receipt(
            selection_parity_pass=True,
            wider_single_block_regime=True,
            row_count=1280,
            native_path_audit_pass=True,
            kernel_output_buffers_emitted=True,
            runtime_sort_key_exact=True,
            realistic_size_proven=True,
            runtime_sort_key_proof=sort_proof,
            parity_proof=parity_proof,
            token=token,
        )


def test_validate_rejects_hand_built_wider_pass_without_runtime_sort_proof() -> None:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_parity_receipt import (
        GlobalRateCapMarginSelectionNativeParityReceipt,
        GlobalRateCapSelectionNativeToken,
        NativeSelectionParityProof,
        validate_global_rate_cap_margin_selection_native_parity_receipt,
    )

    token = GlobalRateCapSelectionNativeToken(
        native_invocation_nonce="n",
        bitonic_kernel_symbol="w",
        gather_kernel_symbol="g",
        kernel_source_sha256="a" * 64,
        selection_input_sha256="b" * 64,
        ordered_output_sha256="c" * 64,
        accepted_output_sha256="d" * 64,
        deferred_output_sha256="e" * 64,
    )
    receipt = GlobalRateCapMarginSelectionNativeParityReceipt(
        selection_parity_pass=True,
        wider_single_block_regime=True,
        realistic_size_proven=True,
        runtime_sort_key_exact=False,
        row_count=1280,
        native_path_audit_pass=True,
        kernel_output_buffers_emitted=True,
        parity_proof=NativeSelectionParityProof(
            parity_ok=True,
            ordered_global_indices_match=True,
            accepted_positions_match=True,
            deferred_positions_match=True,
        ),
        token=token,
    )
    with pytest.raises(ValueError, match="runtime_sort_key_exact"):
        validate_global_rate_cap_margin_selection_native_parity_receipt(receipt)


def test_wider_provenance_rejects_mismatched_gather_symbol_hash() -> None:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_path_audit import (
        audit_kernel_buffer_provenance_entry,
    )
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_triton_kernel import (
        KernelBufferProvenance,
        _margin_selection_gather_rows_kernel,
    )
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_wider_single_block_triton_kernel import (
        _kernel_source_sha256 as wider_sha,
    )

    gather_sym = _margin_selection_gather_rows_kernel.__name__
    mismatched = KernelBufferProvenance(
        buffer_role="row_state_ids",
        kernel_symbol=gather_sym,
        kernel_source_sha256=wider_sha(),
    )
    ok, detail = audit_kernel_buffer_provenance_entry(mismatched)
    assert not ok
    assert "hash mismatch" in detail


def test_wider_dynamic_audit_sorted_keys_shape_non_pow2_row_count() -> None:
    """Regression: sorted_keys is trimmed to row_count, not sort_padded_n (1280≠2048)."""

    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_path_audit import (
        WIDER_SORTED_KEYS_BUFFER_ROLE,
        audit_dynamic_wider_kernel_output_buffers,
    )
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_triton_kernel import (
        KernelBufferProvenance,
        _kernel_source_sha256 as banked_sha,
        _margin_selection_gather_rows_kernel,
    )
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_wider_single_block_compose import (
        MarginSelectionWiderSingleBlockResult,
    )
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_wider_single_block_triton_kernel import (
        WIDER_BITONIC_KERNEL_SYMBOL,
        _kernel_source_sha256 as wider_sha,
    )

    row_count = 1280
    sort_padded_n = 2048
    assert sort_padded_n != row_count
    gather_sym = _margin_selection_gather_rows_kernel.__name__
    banked_hash = banked_sha()
    wider_hash = wider_sha()
    provenance = {
        WIDER_SORTED_KEYS_BUFFER_ROLE: KernelBufferProvenance(
            buffer_role=WIDER_SORTED_KEYS_BUFFER_ROLE,
            kernel_symbol=WIDER_BITONIC_KERNEL_SYMBOL,
            kernel_source_sha256=wider_hash,
        ),
        **{
            name: KernelBufferProvenance(
                buffer_role=name,
                kernel_symbol=gather_sym,
                kernel_source_sha256=banked_hash,
            )
            for name in (
                "row_state_ids",
                "row_flat_indices",
                "row_local_positions",
                "row_global_flat_indices",
                "row_abs_new_acc",
                "row_thresholds",
                "row_directions",
            )
        },
    }
    wider = MarginSelectionWiderSingleBlockResult(
        row_count=row_count,
        sort_padded_n=sort_padded_n,
        pos_width=11,
        host_max_full_key=1,
        padding_headroom_ok=True,
        budget_infeasible=False,
        wider_single_block_regime=True,
        realistic_size_proven=False,
        sorted_keys=torch.zeros(row_count, dtype=torch.int64),
        row_state_ids=torch.zeros(row_count, dtype=torch.int64),
        row_flat_indices=torch.zeros(row_count, dtype=torch.int64),
        row_local_positions=torch.zeros(row_count, dtype=torch.int64),
        row_global_flat_indices=torch.zeros(row_count, dtype=torch.int64),
        row_abs_new_acc=torch.zeros(row_count, dtype=torch.int64),
        row_thresholds=torch.zeros(row_count, dtype=torch.int64),
        row_directions=torch.zeros(row_count, dtype=torch.int16),
        kernel_output_provenance=provenance,
        bitonic_kernel_symbol=WIDER_BITONIC_KERNEL_SYMBOL,
        gather_kernel_symbol=gather_sym,
        kernel_source_sha256=wider_hash,
    )
    dynamic_complete, post_perm, findings = audit_dynamic_wider_kernel_output_buffers(wider)
    assert dynamic_complete is True
    assert post_perm is False
    assert findings == ()
