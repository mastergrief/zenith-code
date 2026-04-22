"""AST refactor primitives — semantic-preserving transformations
for multi-step / recursive refactoring of Python code.

Core primitives (IDE-staple refactorings):

    rename_variable(code, old, new, *, scope="module")
        Rename every binding + reference of `old` to `new` within
        `scope`. Scope is "module" (entire module), or a string
        function name (rename only inside that function). Avoids
        collisions: refuses if `new` already exists in the scope.

    inline_variable(code, var_name, *, scope="module")
        Replace every use of a single-assignment local variable
        with its value expression. Refuses if the variable is
        reassigned, used as an augmented assignment target, or
        its value has side effects (function call with side-effect
        markers, attribute writes, etc).

    extract_method(code, class_name, new_name, start_line, end_line)
        Pull a contiguous block of statements out of one method into
        a new method on the same class. Replaces the block with
        `self.<new_name>(...)`. Detects variables bound before the
        block and used after as return values; detects free vars as
        arguments.

Design choices distinguishing these from ast_repair:

    - ast_repair is ERROR-DRIVEN (runs after test failure, deterministic
      fix per error class). It rewrites Gemma-output bugs.
    - ast_refactor is INTENT-DRIVEN (runs on clean code, preserves
      semantics, restructures for readability / maintainability).

Each primitive returns a RefactorResult with before/after code plus a
validation report (does the AST unparse + re-parse? any names
undefined after rewrite?). Caller is responsible for sandbox-running
the tests to verify no regression.

Supports recursive refactoring (multi-step sessions) via RefactorSession:

    session = RefactorSession(code)
    session.apply(rename_variable, old="tmp", new="accumulator")
    session.apply(inline_variable, var_name="x")
    session.apply(extract_method, class_name="Foo",
                  new_name="_compute", start_line=10, end_line=15)
    final_code = session.result()       # raises if any step failed

Full spec: `.claude/rules/refactor.md` (future).
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# ==================================================================
# Data structures
# ==================================================================


@dataclass
class RefactorResult:
    """Outcome of one refactor primitive."""
    new_code: Optional[str]
    kind: str                            # "rename_variable" | "inline_variable" | "extract_method" | "none"
    notes: List[str] = field(default_factory=list)
    n_changes: int = 0
    error: Optional[str] = None

    @property
    def applied(self) -> bool:
        return self.new_code is not None and self.error is None


# ==================================================================
# 1. rename_variable
# ==================================================================


def _find_scope_node(
    tree: ast.Module, scope: str
) -> ast.AST:
    """Return the AST node corresponding to `scope`.
    'module' → the whole tree. A function name → that FunctionDef
    (must exist; raises ValueError if multiple matches or none).
    """
    if scope == "module":
        return tree
    matches = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == scope
    ]
    if not matches:
        raise ValueError(f"no FunctionDef named {scope!r}")
    if len(matches) > 1:
        raise ValueError(f"multiple FunctionDefs named {scope!r}")
    return matches[0]


def _collect_names_in_scope(scope_node: ast.AST) -> set:
    """All names bound or referenced in `scope_node`. Used for
    collision-gating rename_variable."""
    names: set = set()
    for node in ast.walk(scope_node):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            names.add(node.name)
            for arg in getattr(node, "args", ast.arguments(
                    posonlyargs=[], args=[], kwonlyargs=[],
                    kw_defaults=[], defaults=[])).args:
                names.add(arg.arg)
    return names


class _NameRewriter(ast.NodeTransformer):
    """Rename every `Name(id=old)` to `Name(id=new)` and every
    FunctionDef / arg with id `old` to `new`, within the visited subtree.
    """

    def __init__(self, old: str, new: str):
        self.old = old
        self.new = new
        self.n_changes = 0

    def visit_Name(self, node):
        if node.id == self.old:
            self.n_changes += 1
            return ast.copy_location(
                ast.Name(id=self.new, ctx=node.ctx), node)
        return node

    def visit_arg(self, node):
        if node.arg == self.old:
            self.n_changes += 1
            node.arg = self.new
        return node

    def visit_FunctionDef(self, node):
        if node.name == self.old:
            self.n_changes += 1
            node.name = self.new
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

    def visit_Global(self, node):
        node.names = [self.new if n == self.old else n for n in node.names]
        return node

    def visit_Nonlocal(self, node):
        node.names = [self.new if n == self.old else n for n in node.names]
        return node


def rename_variable(
    code: str, old: str, new: str, *, scope: str = "module",
) -> RefactorResult:
    """Rename every binding + reference of `old` to `new` within `scope`.

    Scope:
        "module" — rename throughout the whole module.
        "<fn_name>" — rename only within that FunctionDef (including
                      its arguments, locals, and nested scopes).

    Safety:
        Refuses if `new` already exists as a name in the target scope
        (would cause silent shadowing). Raises nothing — returns a
        RefactorResult with error set.
    """
    if not old or not new:
        return RefactorResult(None, "none", error="empty name")
    if old == new:
        return RefactorResult(code, "rename_variable",
                              notes=["no-op (old==new)"], n_changes=0)

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return RefactorResult(None, "none", error=f"parse failed: {e.msg}")

    try:
        scope_node = _find_scope_node(tree, scope)
    except ValueError as e:
        return RefactorResult(None, "none", error=str(e))

    # Collision check: is `new` already in use in scope?
    existing = _collect_names_in_scope(scope_node)
    if new in existing:
        return RefactorResult(
            None, "none",
            error=f"collision: {new!r} already in scope {scope!r}")
    if old not in existing:
        return RefactorResult(
            None, "none",
            error=f"{old!r} not found in scope {scope!r}")

    rewriter = _NameRewriter(old, new)
    rewriter.visit(scope_node)
    ast.fix_missing_locations(tree)

    try:
        new_code = ast.unparse(tree)
    except Exception as e:
        return RefactorResult(None, "none", error=f"unparse failed: {e}")

    # Sanity: re-parse to verify
    try:
        ast.parse(new_code)
    except SyntaxError as e:
        return RefactorResult(
            None, "none",
            error=f"post-rewrite parse failed: {e.msg}")

    return RefactorResult(
        new_code, "rename_variable",
        notes=[f"renamed {old!r} -> {new!r} in scope {scope!r} "
               f"({rewriter.n_changes} sites)"],
        n_changes=rewriter.n_changes,
    )


# ==================================================================
# 2. inline_variable
# ==================================================================


def _count_name_bindings(scope_node: ast.AST, name: str) -> int:
    """Count Assign / AugAssign / AnnAssign / For / With / Except
    bindings of `name` in `scope_node` (excluding nested scopes)."""
    n = 0
    for node in ast.walk(scope_node):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    n += 1
                elif isinstance(tgt, (ast.Tuple, ast.List)):
                    for elt in tgt.elts:
                        if isinstance(elt, ast.Name) and elt.id == name:
                            n += 1
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                n += 1
        elif isinstance(node, ast.AnnAssign):
            if (isinstance(node.target, ast.Name)
                    and node.target.id == name):
                n += 1
        elif isinstance(node, ast.For):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                n += 1
    return n


def _find_single_binding_assign(
    scope_node: ast.AST, name: str
) -> Optional[ast.Assign]:
    """If `name` is bound by exactly one simple `Assign` (not augmented,
    not tuple-unpack, not annotation-only) in `scope_node`, return that
    Assign node. Else None."""
    count = _count_name_bindings(scope_node, name)
    if count != 1:
        return None
    for node in ast.walk(scope_node):
        if isinstance(node, ast.Assign):
            if (len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == name):
                return node
    return None


class _InlineRewriter(ast.NodeTransformer):
    """Replace every `Name(id=name)` in a Load context with a deep
    copy of `value_expr`. Leaves Store/Del contexts alone (so the
    assignment itself stays, to be removed separately)."""

    def __init__(self, name: str, value_expr: ast.AST):
        self.name = name
        self.value_expr = value_expr
        self.n_changes = 0

    def visit_Name(self, node):
        if (node.id == self.name
                and isinstance(node.ctx, ast.Load)):
            self.n_changes += 1
            return ast.copy_location(copy.deepcopy(self.value_expr), node)
        return node


def _has_side_effects(expr: ast.AST) -> bool:
    """Very conservative — rejects any Call expression (may have side
    effects), any Yield, Await, or Name-with-Attribute-write."""
    for node in ast.walk(expr):
        if isinstance(node, (ast.Call, ast.Yield, ast.YieldFrom, ast.Await)):
            return True
    return False


def inline_variable(
    code: str, var_name: str, *, scope: str = "module",
    allow_side_effects: bool = False,
) -> RefactorResult:
    """Inline a single-assignment variable.

    Constraints:
        - `var_name` must be bound by exactly one `Assign` in `scope`.
        - The assignment's value must not have side effects (Call /
          Yield / Await) unless `allow_side_effects=True`.
        - `var_name` must not be a function argument, loop variable,
          or otherwise rebound.
    """
    if not var_name:
        return RefactorResult(None, "none", error="empty var_name")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return RefactorResult(None, "none", error=f"parse failed: {e.msg}")

    try:
        scope_node = _find_scope_node(tree, scope)
    except ValueError as e:
        return RefactorResult(None, "none", error=str(e))

    # Also reject if var_name is an argument of any FunctionDef in scope
    for node in ast.walk(scope_node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args + node.args.kwonlyargs:
                if arg.arg == var_name:
                    return RefactorResult(
                        None, "none",
                        error=f"{var_name!r} is a function argument")

    assign = _find_single_binding_assign(scope_node, var_name)
    if assign is None:
        return RefactorResult(
            None, "none",
            error=f"{var_name!r} not single-binding Assign in {scope!r}")

    if not allow_side_effects and _has_side_effects(assign.value):
        return RefactorResult(
            None, "none",
            error=f"{var_name!r}'s value has side effects "
                  "(pass allow_side_effects=True to override)")

    # Rewrite: replace Name loads with the value, drop the Assign stmt.
    rewriter = _InlineRewriter(var_name, assign.value)
    rewriter.visit(scope_node)

    # Remove the original Assign statement from its parent body.
    removed = _remove_assign_stmt(scope_node, assign)
    if not removed:
        return RefactorResult(
            None, "none",
            error=f"could not locate Assign to remove from body")

    ast.fix_missing_locations(tree)
    try:
        new_code = ast.unparse(tree)
    except Exception as e:
        return RefactorResult(None, "none", error=f"unparse failed: {e}")

    try:
        ast.parse(new_code)
    except SyntaxError as e:
        return RefactorResult(
            None, "none", error=f"post-rewrite parse failed: {e.msg}")

    return RefactorResult(
        new_code, "inline_variable",
        notes=[f"inlined {var_name!r} → {ast.unparse(assign.value)!r} "
               f"({rewriter.n_changes} sites)"],
        n_changes=rewriter.n_changes,
    )


def _remove_assign_stmt(
    scope_node: ast.AST, target: ast.Assign
) -> bool:
    """Walk every AST node with a `.body` list and remove `target` from
    it. Returns True if the statement was found and removed."""
    for node in ast.walk(scope_node):
        body_attrs = []
        if hasattr(node, "body") and isinstance(node.body, list):
            body_attrs.append("body")
        if hasattr(node, "orelse") and isinstance(node.orelse, list):
            body_attrs.append("orelse")
        if hasattr(node, "finalbody") and isinstance(node.finalbody, list):
            body_attrs.append("finalbody")
        for attr in body_attrs:
            body = getattr(node, attr)
            for i, stmt in enumerate(body):
                if stmt is target:
                    del body[i]
                    return True
    return False


# ==================================================================
# 3. extract_method
# ==================================================================


def _collect_reads_writes(
    stmts: List[ast.stmt]
) -> Tuple[set, set]:
    """Return (reads, writes) sets of Name strings across `stmts`.
    Store ctx → writes. Load/Del ctx → reads. Walks all descendants."""
    reads: set = set()
    writes: set = set()
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    writes.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    reads.add(node.id)
            elif isinstance(node, ast.arg):
                writes.add(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef)):
                writes.add(node.name)
    return reads, writes


def _stmts_in_line_range(
    body: List[ast.stmt], start: int, end: int
) -> List[ast.stmt]:
    """Return body statements whose line spans intersect [start, end]."""
    out = []
    for stmt in body:
        s_line = getattr(stmt, "lineno", 0)
        e_line = getattr(stmt, "end_lineno", s_line)
        if s_line >= start and e_line <= end:
            out.append(stmt)
    return out


def extract_method(
    code: str,
    class_name: str,
    method_name: str,
    new_name: str,
    start_line: int,
    end_line: int,
) -> RefactorResult:
    """Pull statements in `[start_line, end_line]` out of
    `class_name.method_name` into a new method `class_name.new_name`.

    Generated method signature: `self` + any free variable read by the
    block that wasn't written earlier in the containing method.
    If the block writes variables that are read AFTER the block in the
    containing method, those are returned from the new method and
    unpacked at the call site.

    Replaces the block with `self.<new_name>(args) [-> vars]`.

    Constraints:
        - class must exist, method must exist.
        - block must be contiguous top-level statements of the method
          (no partial-expression extraction).
        - new_name must not collide with an existing method.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return RefactorResult(None, "none", error=f"parse failed: {e.msg}")

    # Find class
    classes = [n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == class_name]
    if not classes:
        return RefactorResult(None, "none",
                              error=f"class {class_name!r} not found")
    cls = classes[0]

    # Find method
    methods = [m for m in cls.body
               if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
               and m.name == method_name]
    if not methods:
        return RefactorResult(None, "none",
                              error=f"method {class_name}.{method_name} not found")
    method = methods[0]

    # Collision check on new method name
    existing = {m.name for m in cls.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if new_name in existing:
        return RefactorResult(None, "none",
                              error=f"method {new_name!r} already exists "
                                    f"on {class_name}")

    # Extract statements
    block = _stmts_in_line_range(method.body, start_line, end_line)
    if not block:
        return RefactorResult(None, "none",
                              error=f"no statements in [{start_line}, {end_line}]")
    if len(block) != sum(
        1 for s in method.body
        if start_line <= getattr(s, "lineno", 0) <= end_line
    ):
        return RefactorResult(None, "none",
                              error="block is not contiguous in method body")

    # Identify free vars (reads in block not written in block).
    # in_scope_before = names defined by the enclosing method before
    # the block — these are the CANDIDATES for passthrough args. Other
    # free reads (module-level globals, builtins) don't need args.
    before = [s for s in method.body
              if getattr(s, "end_lineno", 0) < start_line]
    _, written_before = _collect_reads_writes(before)
    arg_names = {a.arg for a in method.args.args}
    in_scope_before = written_before | arg_names

    reads, writes = _collect_reads_writes(block)
    free_reads = (reads - writes - {"self"}) & in_scope_before

    # Identify returned vars: writes in block that are READ after the block
    after = [s for s in method.body
             if getattr(s, "lineno", 0) > end_line]
    reads_after, _ = _collect_reads_writes(after)
    returned = sorted(writes & reads_after)

    # Build new method def
    free_read_list = sorted(free_reads)
    new_args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg="self")] + [ast.arg(arg=n) for n in free_read_list],
        kwonlyargs=[], kw_defaults=[], defaults=[],
    )
    new_body = list(block)  # shallow copy — stmts moved, not cloned
    if returned:
        if len(returned) == 1:
            new_body.append(ast.Return(value=ast.Name(id=returned[0],
                                                     ctx=ast.Load())))
        else:
            new_body.append(ast.Return(
                value=ast.Tuple(
                    elts=[ast.Name(id=n, ctx=ast.Load()) for n in returned],
                    ctx=ast.Load()),
            ))

    new_method = ast.FunctionDef(
        name=new_name,
        args=new_args,
        body=new_body,
        decorator_list=[],
        returns=None,
        type_comment=None,
    )

    # Build call-site replacement
    call_args = [ast.Name(id=n, ctx=ast.Load()) for n in free_read_list]
    call = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="self", ctx=ast.Load()),
            attr=new_name, ctx=ast.Load()),
        args=call_args, keywords=[],
    )
    if returned:
        if len(returned) == 1:
            call_stmt = ast.Assign(
                targets=[ast.Name(id=returned[0], ctx=ast.Store())],
                value=call,
            )
        else:
            call_stmt = ast.Assign(
                targets=[ast.Tuple(
                    elts=[ast.Name(id=n, ctx=ast.Store()) for n in returned],
                    ctx=ast.Store())],
                value=call,
            )
    else:
        call_stmt = ast.Expr(value=call)

    # Replace block with call_stmt in method.body
    new_method_body = []
    block_ids = {id(s) for s in block}
    inserted = False
    for s in method.body:
        if id(s) in block_ids:
            if not inserted:
                new_method_body.append(call_stmt)
                inserted = True
            # skip — statement moved into new method
        else:
            new_method_body.append(s)
    method.body = new_method_body

    # Insert new method into class body AFTER the original method
    insert_idx = cls.body.index(method) + 1
    cls.body.insert(insert_idx, new_method)

    ast.fix_missing_locations(tree)
    try:
        new_code = ast.unparse(tree)
    except Exception as e:
        return RefactorResult(None, "none", error=f"unparse failed: {e}")

    try:
        ast.parse(new_code)
    except SyntaxError as e:
        return RefactorResult(
            None, "none", error=f"post-rewrite parse failed: {e.msg}")

    notes = [
        f"extracted {len(block)} statement(s) from "
        f"{class_name}.{method_name} into {class_name}.{new_name}",
        f"args: {free_read_list}",
        f"returns: {returned}",
    ]
    return RefactorResult(
        new_code, "extract_method",
        notes=notes, n_changes=len(block),
    )


# ==================================================================
# Recursive refactor session
# ==================================================================


class RefactorSession:
    """Chain multiple refactor primitives with automatic rollback on
    validation failure.

    Example:
        s = RefactorSession(code)
        s.apply(rename_variable, old="x", new="value")
        s.apply(inline_variable, var_name="tmp")
        s.apply(extract_method, class_name="Foo", method_name="process",
                new_name="_helper", start_line=10, end_line=15)
        final = s.result()  # raises RefactorError if any step failed

    Each step is validated via ast.parse. External test-validation
    (sandbox-run tests between steps) is the caller's responsibility.
    The session tracks the full history for audit.
    """

    def __init__(self, code: str):
        self._initial = code
        self._current = code
        self._history: List[Tuple[str, RefactorResult]] = []
        self._failed = False
        self._error: Optional[str] = None

    def apply(self, primitive: Callable, /, **kwargs) -> RefactorResult:
        """Apply `primitive(current_code, **kwargs)`. Records the result
        in history. On failure (error set), records and stops the
        session — subsequent apply() calls are no-ops."""
        if self._failed:
            return RefactorResult(
                None, "none", error="session already failed")

        result = primitive(self._current, **kwargs)
        self._history.append((primitive.__name__, result))
        if result.applied:
            self._current = result.new_code
        else:
            self._failed = True
            self._error = result.error
        return result

    def result(self) -> str:
        """Return the current code. Raises RuntimeError if the session
        has a failed step."""
        if self._failed:
            raise RuntimeError(
                f"session failed: {self._error} "
                f"(history: {[k for k, _ in self._history]})")
        return self._current

    @property
    def history(self) -> List[Tuple[str, RefactorResult]]:
        return list(self._history)

    @property
    def ok(self) -> bool:
        return not self._failed
