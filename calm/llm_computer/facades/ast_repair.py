"""Tier-2 AST-walker card — deterministic repair of Gemma's
persistent code failure modes on the R53 corpus.

Two rewrites, both driven by runtime error text (not by a spec):

1. **Shadow rename** (TypeError: 'X' object is not callable).
   Gemma writes `self.tokens = capacity` in __init__ but the test
   calls `tb.tokens()` as a method. Find every class-attribute assign
   `self.<name> = ...` where <name> is also a method on the same
   class; rename the attribute to `_<name>` and rewrite all non-call
   read sites. The method body stays intact, callers see the method.
   R53.33 receipt: this exact pattern on `token_bucket_rate_limiter`.

2. **Dict-key synonym rewrite** (KeyError: 'X').
   Test does `r['age']['mean']`; Gemma emitted `{'age': {'avg': ...}}`.
   Walk every Dict literal in the tree, find a key that's a known
   synonym of the missing key, rename it. Conservative: only rewrites
   string-literal keys with a synonym from a small curated table.
   R53.33 receipt: `csv_column_stats` emits 'avg' for 'mean', etc.

The walker is post-generation and post-extraction — it runs AFTER the
test output has been captured, so the error text guides which rewrite
fires. On no match it returns (code, None) and the caller falls back
to the LLM-repair path.

Intentionally narrow. Each rewrite handles one failure class, leaves
everything else alone. Wider inference (e.g. guessing `'score'`'s
column index from the header) is deferred to a future pass.

Wired from scripts/r53_21_import_inject.py alongside import injection
and R53.19 structured repair.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


# Curated synonym table. Keys are the "target" names tests expect;
# values are alternate spellings Gemma emits. Bidirectional lookup
# happens below — if target='mean' is missing and code has 'avg',
# we rewrite 'avg' -> 'mean'. Conservative list; expand with receipts.
DICT_KEY_SYNONYMS = {
    "mean":    {"avg", "average", "mu"},
    "stdev":   {"std", "stddev", "sigma", "deviation", "std_dev"},
    "median":  {"med", "middle"},
    "variance": {"var", "variance"},
    "min":     {"minimum", "low", "smallest", "lo"},
    "max":     {"maximum", "high", "largest", "hi"},
    "sum":     {"total"},
    "count":   {"cnt", "n", "num"},
    "first":   {"head", "start"},
    "last":    {"tail", "end"},
}


@dataclass
class RepairResult:
    """Outcome of an AST-walker pass."""
    new_code: Optional[str]    # None if no rewrite applied
    kind: str                  # "shadow_rename" | "dict_synonym" | "none"
    notes: List[str] = field(default_factory=list)  # human-readable trail

    @property
    def applied(self) -> bool:
        return self.new_code is not None


# --------------------------------------------------------------------
# 1. Shadow rename
# --------------------------------------------------------------------


def _collect_methods_per_class(tree: ast.Module) -> dict:
    """Map class name → set of method names defined inside."""
    out: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.add(item.name)
            out[node.name] = methods
    return out


def _find_shadowed_attrs(tree: ast.Module) -> List[Tuple[str, str]]:
    """Return (class_name, attr_name) pairs where self.<attr> is assigned
    inside a method AND <attr> is also a method on the same class.

    Uses the enclosing-class walk rather than ast.walk so the same
    attribute name on two different classes doesn't collide.
    """
    found: List[Tuple[str, str]] = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        methods = {
            m.name for m in cls.body
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if not methods:
            continue
        shadowed: Set[str] = set()
        for m in cls.body:
            if not isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(m):
                # Assignment `self.<name> = ...`
                if isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if (isinstance(tgt, ast.Attribute)
                                and isinstance(tgt.value, ast.Name)
                                and tgt.value.id == "self"
                                and tgt.attr in methods):
                            shadowed.add(tgt.attr)
                elif isinstance(node, ast.AugAssign):
                    tgt = node.target
                    if (isinstance(tgt, ast.Attribute)
                            and isinstance(tgt.value, ast.Name)
                            and tgt.value.id == "self"
                            and tgt.attr in methods):
                        shadowed.add(tgt.attr)
        for name in sorted(shadowed):
            found.append((cls.name, name))
    return found


class _ShadowRewriter(ast.NodeTransformer):
    """Rename `self.<old>` to `self.<new>` EXCEPT in call positions
    (where it's still the method). Scoped to the class whose method
    we're rewriting."""

    def __init__(self, class_name: str, old: str, new: str):
        self.class_name = class_name
        self.old = old
        self.new = new
        self._in_target_class = 0

    def visit_ClassDef(self, node):
        if node.name == self.class_name:
            self._in_target_class += 1
            self.generic_visit(node)
            self._in_target_class -= 1
            return node
        return self.generic_visit(node) or node

    def visit_Call(self, node):
        # If the call target is self.<old>, keep it as-is (method call).
        # Recurse into args and into .func children but NOT into the
        # self.<old> Attribute itself.
        if (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr == self.old
                and self._in_target_class > 0):
            # Visit args/keywords normally, leave node.func alone
            node.args = [self.visit(a) for a in node.args]
            node.keywords = [self.visit(k) for k in node.keywords]
            return node
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if (self._in_target_class > 0
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr == self.old):
            return ast.copy_location(
                ast.Attribute(value=node.value, attr=self.new, ctx=node.ctx),
                node)
        return node


def rename_shadow(code: str) -> RepairResult:
    """If any class has `self.<name> = ...` where <name> is also a
    method on the same class, rename the attribute to `_<name>`
    (or `<name>_value` on fallback collision). Rewrites all non-call
    read sites.

    Returns RepairResult with kind='shadow_rename' + new_code when a
    rewrite fires, or kind='none' otherwise.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return RepairResult(None, "none", ["parse failed"])

    shadows = _find_shadowed_attrs(tree)
    if not shadows:
        return RepairResult(None, "none", ["no shadow detected"])

    notes: List[str] = []
    methods_per_class = _collect_methods_per_class(tree)
    for class_name, attr in shadows:
        new = f"_{attr}"
        # Guarantee uniqueness — if _<attr> also collides, append suffix
        while new in methods_per_class.get(class_name, set()):
            new += "_value"
        tree = _ShadowRewriter(class_name, attr, new).visit(tree)
        notes.append(f"{class_name}.{attr} -> {class_name}.{new}")
    ast.fix_missing_locations(tree)

    try:
        new_code = ast.unparse(tree)
    except Exception as e:
        return RepairResult(None, "none", [f"unparse failed: {e}"])

    return RepairResult(new_code, "shadow_rename", notes)


# --------------------------------------------------------------------
# 2. Dict-key synonym rewrite
# --------------------------------------------------------------------


def _resolve_synonym(missing_key: str) -> Set[str]:
    """Return the set of alternate spellings that, when found in the
    code, should be renamed to `missing_key`. Falls back to empty
    when `missing_key` isn't in the synonym table."""
    direct = DICT_KEY_SYNONYMS.get(missing_key.lower())
    if direct:
        return direct
    # Reverse lookup: maybe missing_key is a synonym and target is the key
    for target, alts in DICT_KEY_SYNONYMS.items():
        if missing_key.lower() in alts:
            return {target} | (alts - {missing_key.lower()})
    return set()


class _DictKeyRewriter(ast.NodeTransformer):
    """Rename string literal keys in every Dict literal OR Subscript
    access when they match one of the synonym alternates. Also
    rewrites `d.get('avg', ...)` / `d['avg']` to the target key.
    """

    def __init__(self, target: str, alternates: Set[str]):
        self.target = target
        self.alternates = {a.lower() for a in alternates}
        self.n_replaced = 0

    def _maybe_rewrite_constant(self, node):
        if (isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.lower() in self.alternates):
            self.n_replaced += 1
            return ast.copy_location(ast.Constant(value=self.target), node)
        return node

    def visit_Dict(self, node):
        self.generic_visit(node)
        node.keys = [self._maybe_rewrite_constant(k) if k is not None else k
                     for k in node.keys]
        return node

    def visit_Subscript(self, node):
        self.generic_visit(node)
        node.slice = self._maybe_rewrite_constant(node.slice)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        # d.get('avg', ...) / d.setdefault('avg', ...) / d.pop('avg', ...)
        if (isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "setdefault", "pop")
                and node.args):
            node.args[0] = self._maybe_rewrite_constant(node.args[0])
        return node


def rewrite_dict_synonym(code: str, missing_key: str) -> RepairResult:
    """If `missing_key` has known synonyms AND any of them appear as
    string-literal keys/subscripts in `code`, rename them to
    `missing_key`. Returns RepairResult with kind='dict_synonym' when
    a rewrite fires.
    """
    alternates = _resolve_synonym(missing_key)
    if not alternates:
        return RepairResult(None, "none",
                            [f"no synonyms for '{missing_key}'"])

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return RepairResult(None, "none", ["parse failed"])

    writer = _DictKeyRewriter(missing_key, alternates)
    tree = writer.visit(tree)
    if writer.n_replaced == 0:
        return RepairResult(None, "none",
                            [f"no synonyms of '{missing_key}' found "
                             f"in {sorted(alternates)}"])

    ast.fix_missing_locations(tree)
    try:
        new_code = ast.unparse(tree)
    except Exception as e:
        return RepairResult(None, "none", [f"unparse failed: {e}"])

    return RepairResult(
        new_code, "dict_synonym",
        [f"rewrote {writer.n_replaced} occurrence(s) → '{missing_key}'"])


# --------------------------------------------------------------------
# Error-text parsing
# --------------------------------------------------------------------


_KEYERROR_RE = re.compile(r"KeyError:\s*['\"]([^'\"]+)['\"]")
_TYPEERROR_CALLABLE_RE = re.compile(
    r"TypeError:\s*'(\w+)' object is not callable")


def extract_missing_key(error_output: str) -> Optional[str]:
    m = _KEYERROR_RE.search(error_output or "")
    return m.group(1) if m else None


def has_typeerror_callable(error_output: str) -> bool:
    return bool(_TYPEERROR_CALLABLE_RE.search(error_output or ""))


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------


def repair(code: str, error_output: str) -> RepairResult:
    """Dispatch on error kind. Returns first successful repair, or
    RepairResult(None, 'none', ...) if neither applies.

    Caller responsibility: re-run tests on the returned code. Walker
    does no sandboxing itself.
    """
    if not code:
        return RepairResult(None, "none", ["empty code"])

    # 1. Shadow rename — driven by TypeError callable. We also run the
    # detector on code with no such error: sometimes Gemma's code has
    # the shadow but the error surfaces as AttributeError on a later
    # call. Safe to run unconditionally — no-op if no shadow exists.
    shadow = rename_shadow(code)
    if shadow.applied:
        return shadow

    # 2. Dict synonym — driven by KeyError key
    missing = extract_missing_key(error_output or "")
    if missing:
        syn = rewrite_dict_synonym(code, missing)
        if syn.applied:
            return syn

    return RepairResult(None, "none",
                        ["no applicable rewrite",
                         f"shadow: {shadow.notes}",
                         f"missing_key: {missing}"])
