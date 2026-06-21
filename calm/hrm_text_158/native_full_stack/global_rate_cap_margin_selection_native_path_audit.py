"""B2-5a′ Stage-2 §3.2 anti-laundering audit for native MARGIN-selection path.

Static denylist + AST Subscript slice-references visitor + dynamic kernel-output
buffer provenance check.  Gate-1 re-runs both mechanically.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_parity_receipt import (
    ROW_TENSOR_BUFFER_ROLES,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_triton_kernel import (
    KernelBufferProvenance,
    MarginSelectionSingleBlockTileResult,
)

_STATIC_DENYLIST_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\btorch\.sort\b",
        r"\btorch\.argsort\b",
        r"\btorch\.topk\b",
        r"\.sort\s*\(",
        r"\.argsort\s*\(",
        r"\.topk\s*\(",
        r"\btorch\.index_select\b",
        r"\.index_select\s*\(",
        r"\btorch\.gather\b",
        r"\.gather\s*\(",
        r"\btorch\.take_along_dim\b",
        r"\btake_along_dim\s*\(",
        r"\btorch\.masked_select\b",
        r"\.masked_select\s*\(",
        r"nonzero\s*\([^)]*\)\s*\[",
    )
)

_SLICE_REF_NAMES = frozenset({"order", "sorted_pos", "perm", "positions"})
_SORT_PATH_NAMES = frozenset({"perm", "positions", "order", "sorted_pos"})


@dataclass(frozen=True)
class NativePathAuditFinding:
    file: str
    line: int
    kind: str
    detail: str


@dataclass(frozen=True)
class NativePathAuditResult:
    static_clean: bool
    dynamic_complete: bool
    post_kernel_torch_permutation_detected: bool
    native_path_audit_pass: bool
    static_findings: tuple[NativePathAuditFinding, ...]
    dynamic_findings: tuple[NativePathAuditFinding, ...]


def _slice_references_sort_path(slice_node: ast.AST, local_sort_names: set[str]) -> bool:
    if isinstance(slice_node, ast.Constant):
        return False
    for node in ast.walk(slice_node):
        if isinstance(node, ast.Name) and (
            node.id in _SLICE_REF_NAMES or node.id in local_sort_names
        ):
            return True
    return False


def _node_references_sort_path_name(node: ast.AST, local_sort_names: set[str]) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and (
            sub.id in _SLICE_REF_NAMES or sub.id in local_sort_names
        ):
            return True
    return False


class _SubscriptSliceVisitor(ast.NodeVisitor):
    def __init__(self, *, filepath: str, local_sort_names: set[str]) -> None:
        self.filepath = filepath
        self.local_sort_names = set(local_sort_names)
        self.findings: list[NativePathAuditFinding] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Name):
                if (
                    node.value.id in _SORT_PATH_NAMES
                    or node.value.id in _SLICE_REF_NAMES
                    or node.value.id in self.local_sort_names
                ):
                    self.local_sort_names.add(target.id)
            if isinstance(target, ast.Name) and _node_references_sort_path_name(
                node.value, self.local_sort_names
            ):
                self.local_sort_names.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if isinstance(arg, ast.Subscript) and _slice_references_sort_path(
                arg.slice, self.local_sort_names
            ):
                self.findings.append(
                    NativePathAuditFinding(
                        file=self.filepath,
                        line=node.lineno,
                        kind="ast_call_sort_path_arg",
                        detail=ast.unparse(node) if hasattr(ast, "unparse") else "call",
                    )
                )
                continue
            if _node_references_sort_path_name(arg, self.local_sort_names):
                self.findings.append(
                    NativePathAuditFinding(
                        file=self.filepath,
                        line=node.lineno,
                        kind="ast_call_sort_path_arg",
                        detail=ast.unparse(node) if hasattr(ast, "unparse") else "call",
                    )
                )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.slice, ast.Constant):
            self.generic_visit(node)
            return
        if _slice_references_sort_path(node.slice, self.local_sort_names):
            self.findings.append(
                NativePathAuditFinding(
                    file=self.filepath,
                    line=node.lineno,
                    kind="ast_subscript_slice_ref",
                    detail=ast.unparse(node) if hasattr(ast, "unparse") else "subscript",
                )
            )
        self.generic_visit(node)


def _docstring_line_numbers(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module, ast.ClassDef)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            start = first.lineno
            end = first.end_lineno or start
            lines.update(range(start, end + 1))
    return lines


def audit_native_path_module_source(
    source: str,
    *,
    filepath: str = "<unknown>",
) -> list[NativePathAuditFinding]:
    findings: list[NativePathAuditFinding] = []
    tree = ast.parse(source, filename=filepath)
    doc_lines = _docstring_line_numbers(tree)
    for lineno, line in enumerate(source.splitlines(), start=1):
        if lineno in doc_lines:
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern in _STATIC_DENYLIST_PATTERNS:
            if pattern.search(line):
                findings.append(
                    NativePathAuditFinding(
                        file=filepath,
                        line=lineno,
                        kind="static_denylist",
                        detail=pattern.pattern,
                    )
                )
                break

    visitor = _SubscriptSliceVisitor(filepath=filepath, local_sort_names=set(_SORT_PATH_NAMES))
    visitor.visit(tree)
    findings.extend(visitor.findings)
    return findings


def audit_native_path_modules(
    module_paths: Iterable[Path],
) -> NativePathAuditResult:
    static_findings: list[NativePathAuditFinding] = []
    for path in module_paths:
        static_findings.extend(
            audit_native_path_module_source(path.read_text(encoding="utf-8"), filepath=str(path))
        )
    static_clean = len(static_findings) == 0
    return NativePathAuditResult(
        static_clean=static_clean,
        dynamic_complete=False,
        post_kernel_torch_permutation_detected=False,
        native_path_audit_pass=False,
        static_findings=tuple(static_findings),
        dynamic_findings=(),
    )


def audit_dynamic_kernel_output_buffers(
    tile: MarginSelectionSingleBlockTileResult | None,
) -> tuple[bool, bool, tuple[NativePathAuditFinding, ...]]:
    """Return (dynamic_complete, post_kernel_torch_permutation_detected, findings)."""

    if tile is None or tile.row_count == 0:
        return True, False, ()

    findings: list[NativePathAuditFinding] = []
    provenance = tile.kernel_output_provenance
    for role in ROW_TENSOR_BUFFER_ROLES:
        if role not in provenance:
            findings.append(
                NativePathAuditFinding(
                    file="dynamic",
                    line=0,
                    kind="missing_kernel_buffer_provenance",
                    detail=role,
                )
            )
            continue
        entry: KernelBufferProvenance = provenance[role]
        if not entry.kernel_symbol or not entry.kernel_source_sha256:
            findings.append(
                NativePathAuditFinding(
                    file="dynamic",
                    line=0,
                    kind="incomplete_provenance",
                    detail=role,
                )
            )

    buffers = {
        "row_state_ids": tile.row_state_ids,
        "row_flat_indices": tile.row_flat_indices,
        "row_local_positions": tile.row_local_positions,
        "row_global_flat_indices": tile.row_global_flat_indices,
        "row_abs_new_acc": tile.row_abs_new_acc,
        "row_thresholds": tile.row_thresholds,
        "row_directions": tile.row_directions,
    }
    for role, tensor in buffers.items():
        if tensor is None or int(tensor.numel()) != tile.row_count:
            findings.append(
                NativePathAuditFinding(
                    file="dynamic",
                    line=0,
                    kind="buffer_shape",
                    detail=f"{role} numel mismatch",
                )
            )

    dynamic_complete = len(findings) == 0
    return dynamic_complete, False, tuple(findings)


def run_full_native_path_audit(
    *,
    module_paths: Iterable[Path],
    tile: MarginSelectionSingleBlockTileResult | None = None,
) -> NativePathAuditResult:
    static = audit_native_path_modules(module_paths)
    dynamic_complete, post_perm, dynamic_findings = audit_dynamic_kernel_output_buffers(tile)
    native_pass = static.static_clean and dynamic_complete and not post_perm
    return NativePathAuditResult(
        static_clean=static.static_clean,
        dynamic_complete=dynamic_complete,
        post_kernel_torch_permutation_detected=post_perm,
        native_path_audit_pass=native_pass,
        static_findings=static.static_findings,
        dynamic_findings=dynamic_findings,
    )


__all__ = [
    "NativePathAuditFinding",
    "NativePathAuditResult",
    "audit_dynamic_kernel_output_buffers",
    "audit_native_path_module_source",
    "audit_native_path_modules",
    "run_full_native_path_audit",
]
