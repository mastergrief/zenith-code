"""B2-5a′ Stage-2 §4.5 native-path audit tests (CPU-only).

Five violation fixtures must FAIL the audit; real native modules must PASS.
"""
from __future__ import annotations

import ast
import hashlib
import py_compile
from pathlib import Path

import pytest

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
