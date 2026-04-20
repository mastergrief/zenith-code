"""Tier-2 AST-walker card — deterministic repair of Gemma's
persistent code failure modes on the R53 corpus.

Five rewrites, all driven by runtime error text (not by a spec):

1. **Syntax repair** (SyntaxError at parse time).
   Iteratively parse code, find each SyntaxError, try to fix:
   (a) mismatch — `closing X doesn't match opening Y` → insert Y's
   matching closer at the mismatch offset; (b) append-at-end —
   line has unbalanced brackets → append or insert-before-colon.
   R53.35 receipt: csv_column_stats missing `)` before `:`, 0/0 → 8/8.

2. **Shadow rename** (TypeError: 'X' object is not callable).
   Gemma writes `self.tokens = capacity` in __init__ but the test
   calls `tb.tokens()` as a method. Find every class-attribute assign
   `self.<name> = ...` where <name> is also a method on the same
   class; rename the attribute to `_<name>` and rewrite all non-call
   read sites. The method body stays intact, callers see the method.
   R53.33 receipt: this exact pattern on `token_bucket_rate_limiter`.

3. **Dict-key synonym rewrite** (KeyError: 'X').
   Test does `r['age']['mean']`; Gemma emitted `{'age': {'avg': ...}}`.
   Walk every Dict literal in the tree, find a key that's a known
   synonym of the missing key, rename it. Conservative: only rewrites
   string-literal keys with a synonym from a small curated table.
   R53.33 receipt: `csv_column_stats` emits 'avg' for 'mean', etc.

4. **Off-by-one range** (IndexError: list index out of range).
   Gemma writes `for i in range(len(xs) + 1): xs[i]`. The `+1` is a
   fencepost bug. Conservative: require both (a) an IndexError in the
   error text and (b) an actual `container[loopvar]` subscript inside
   the loop body before rewriting. Handles `range(len(X)+1)` and
   `range(0, len(X)+1)`.

5. **Missing return** (AssertionError / NoneType).
   Gemma computes the answer as a bare expression at the end of the
   function (e.g. `result` on its own line) and forgets to return it.
   If the function has zero `return <value>` statements AND its last
   statement is `Expr(value=X)` where X is a plausible return
   expression (Name, BinOp, Call, Subscript, IfExp), rewrite the last
   Expr into `Return(X)`. Conservative — requires no existing
   value-return anywhere in the function body.

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
_INDEXERROR_OOB_RE = re.compile(
    r"IndexError:\s*(?:list |tuple |string |)index out of range")
# Signals that a function returned None when the caller / test expected
# a value. Covers: AssertionError referencing None, TypeError on
# NoneType operations, "got None", generic AttributeError on NoneType.
_NONE_RETURN_RE = re.compile(
    r"(?:NoneType|got None|expected \S+, got None|"
    r"AssertionError.*None|AttributeError:.*NoneType)",
    re.IGNORECASE)


def extract_missing_key(error_output: str) -> Optional[str]:
    m = _KEYERROR_RE.search(error_output or "")
    return m.group(1) if m else None


def has_typeerror_callable(error_output: str) -> bool:
    return bool(_TYPEERROR_CALLABLE_RE.search(error_output or ""))


def has_indexerror_oob(error_output: str) -> bool:
    return bool(_INDEXERROR_OOB_RE.search(error_output or ""))


def has_none_return_signal(error_output: str) -> bool:
    return bool(_NONE_RETURN_RE.search(error_output or ""))


# --------------------------------------------------------------------
# 3. Balanced-paren syntax repair (pre-walker, pre-extractor)
# --------------------------------------------------------------------


_BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = {")", "]", "}"}


_TRAILING_SUFFIX_RE = re.compile(
    r"""
    (?P<suffix>
        (?:\s*->\s*\w+)?   # return-type annotation (-> Foo)
        \s*:\s*$           # mandatory trailing colon
      |
        \s*:\s*$           # or just colon
    )
    """,
    re.VERBOSE,
)


def _balance_brackets_on_line(line: str) -> Optional[str]:
    """If `line` has more openers than closers of some bracket class,
    insert the missing closers (in reverse-stack order). Returns the
    fixed line, or None if no fix applicable (more closers than
    openers, or already balanced).

    Insertion point: BEFORE any trailing `:`/`->:` (statement-opener
    suffixes that commonly appear on `for`/`if`/`def`/`with`/`while`
    lines where Gemma missed a paren). This handles the canonical
    R53.35v2 pattern: `for i in range(min(a, len(row)):` with missing
    `)` before the `:`.

    Falls back to append-at-end for lines without a trailing suffix
    (e.g. `x = func(a, b` expression without terminator).

    Assumes lexically naive parsing: brackets inside strings are NOT
    distinguished. False positives on code like `s = "("` are possible
    but rare.
    """
    # Strip trailing comment first (naive — ignores strings).
    code_part = line
    comment = ""
    hash_idx = code_part.find("#")
    if hash_idx >= 0 and hash_idx > code_part.rfind('"'):
        comment = code_part[hash_idx:]
        code_part = code_part[:hash_idx]

    stripped = code_part.rstrip()
    trailing_ws = code_part[len(stripped):]

    # Identify trailing statement-suffix (`:` or `-> T:`) to insert before
    suffix_match = _TRAILING_SUFFIX_RE.search(stripped)
    if suffix_match:
        suffix = suffix_match.group(0)
        code_head = stripped[:-len(suffix)]
    else:
        suffix = ""
        code_head = stripped

    stack: List[str] = []
    in_str: Optional[str] = None
    for ch in code_head:
        if in_str:
            if ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
            continue
        if ch in _BRACKET_PAIRS:
            stack.append(ch)
        elif ch in _CLOSERS:
            if stack and _BRACKET_PAIRS[stack[-1]] == ch:
                stack.pop()
            else:
                return None

    if not stack:
        return None   # balanced

    fix = "".join(_BRACKET_PAIRS[b] for b in reversed(stack))
    return code_head + fix + suffix + trailing_ws + comment


MAX_SYNTAX_REPAIR_LINES = 10


_MISMATCH_RE = re.compile(
    r"closing parenthesis '([)}\]])' does not match "
    r"opening parenthesis '([({\[])'"
)


def _repair_mismatch(line: str, offset: int, closer: str, opener: str) -> Optional[str]:
    """Python says the bracket at `offset` (`closer`, e.g. '}') doesn't
    match a preceding unclosed `opener` (e.g. '('). Insert the correct
    closer for `opener` right BEFORE the mismatched closer.

    Canonical R53.35 csv case:
      `{h[i]: [] for i in range(len(h)}`  — `}` at offset 40
      Python: "closing '}' does not match opening '('"
      Fix: insert `)` at offset 39 → `{h[i]: [] for i in range(len(h))}`.

    Returns None if offset is out of bounds.
    """
    if opener not in _BRACKET_PAIRS:
        return None
    need = _BRACKET_PAIRS[opener]
    # Python's offset is 1-based for column; index into line is offset-1.
    # Sanity: check the char at offset-1 is the reported closer.
    idx = offset - 1
    if idx < 0 or idx >= len(line):
        return None
    if line[idx] != closer:
        # Offset may differ by 1 in some Python versions; try offset
        if 0 <= offset < len(line) and line[offset] == closer:
            idx = offset
        else:
            return None
    return line[:idx] + need + line[idx:]


def repair_syntax(code: str) -> RepairResult:
    """Iteratively parse `code`, find each SyntaxError, and try to
    auto-fix it. Two strategies:

    1. **Mismatch**: error says `closing X does not match opening Y` —
       insert Y's matching closer before X. Handles the canonical
       R53.35 csv pattern (`{... range(len(h)}` missing `)`).
    2. **Unclosed-at-end**: error's line has more openers than closers
       of some bracket class — append the missing closers. Handles
       the simpler `func(a, b` case.

    Stops when ast.parse succeeds or repair can't make progress.

    Returns RepairResult with kind='syntax_repair' when any fix was
    applied, 'none' otherwise.
    """
    if not code:
        return RepairResult(None, "none", ["empty code"])

    try:
        ast.parse(code)
        return RepairResult(None, "none", ["code already parses"])
    except SyntaxError:
        pass

    lines = code.split("\n")
    fixes: List[str] = []
    # Line numbers where each strategy has already been attempted
    # (some lines may need both passes).
    seen_mismatch: set = set()
    seen_append: set = set()

    for _ in range(MAX_SYNTAX_REPAIR_LINES):
        try:
            ast.parse("\n".join(lines))
            break
        except SyntaxError as e:
            lineno = e.lineno or 0
            offset = e.offset or 0
            msg = e.msg or ""
            if lineno < 1 or lineno > len(lines):
                break

            fixed_this_round = False

            # Strategy 1: mismatch repair
            mm = _MISMATCH_RE.search(msg)
            if mm and lineno not in seen_mismatch:
                seen_mismatch.add(lineno)
                closer, opener = mm.group(1), mm.group(2)
                fixed = _repair_mismatch(
                    lines[lineno - 1], offset, closer, opener)
                if fixed is not None and fixed != lines[lineno - 1]:
                    fixes.append(
                        f"line {lineno}: inserted '{_BRACKET_PAIRS[opener]}' "
                        f"before mismatched '{closer}'")
                    lines[lineno - 1] = fixed
                    fixed_this_round = True

            # Strategy 2: append-at-end repair (only if mismatch didn't fire)
            if not fixed_this_round and lineno not in seen_append:
                seen_append.add(lineno)
                fixed = _balance_brackets_on_line(lines[lineno - 1])
                if fixed is not None and fixed != lines[lineno - 1]:
                    fixes.append(
                        f"line {lineno}: closed unbalanced brackets")
                    lines[lineno - 1] = fixed
                    fixed_this_round = True

            if not fixed_this_round:
                break

    if not fixes:
        return RepairResult(None, "none",
                            ["syntax error not fixable by bracket balancing"])

    new_code = "\n".join(lines)
    try:
        ast.parse(new_code)
    except SyntaxError as e:
        return RepairResult(None, "none",
                            [f"post-repair parse still failed: {e.msg}"]
                            + fixes)

    return RepairResult(new_code, "syntax_repair", fixes)


# --------------------------------------------------------------------
# 4. Off-by-one range rewrite (IndexError-driven)
# --------------------------------------------------------------------


def _len_plus_one_call(node: ast.AST) -> Optional[ast.Call]:
    """If `node` is `len(X) + 1` or `1 + len(X)`, return the `len(X)`
    Call node. Otherwise None."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)):
        return None
    left, right = node.left, node.right

    def _is_len_call(n):
        return (isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "len"
                and len(n.args) == 1)

    def _is_one(n):
        return isinstance(n, ast.Constant) and n.value == 1

    if _is_len_call(left) and _is_one(right):
        return left
    if _is_len_call(right) and _is_one(left):
        return right
    return None


def _body_subscripts_container_by_loopvar(
    body: List[ast.stmt], container_name: str, loopvar: str
) -> bool:
    """True iff any `container_name[loopvar]` subscript appears in
    `body` (recursively)."""
    for stmt in body:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Subscript):
                continue
            val = node.value
            idx = node.slice  # Python 3.9+: slice is the expr directly
            if (isinstance(val, ast.Name) and val.id == container_name
                    and isinstance(idx, ast.Name) and idx.id == loopvar):
                return True
    return False


class _OffByOneRewriter(ast.NodeTransformer):
    """For each For-loop of form
       `for <loopvar> in range(len(<container>) + 1):`
       (or `range(0, len(<container>) + 1)`) where
       `<container>[<loopvar>]` appears in the body, rewrite the
       `len(X) + 1` subexpression to `len(X)`.

    Conservative: the body-subscript check filters loops where the
    `+1` is intentional (e.g. fencepost iteration without indexing).
    """

    def __init__(self):
        self.n_replaced = 0

    def visit_For(self, node):
        # Depth-first so nested loops rewrite first.
        self.generic_visit(node)

        # Loop var must be a simple Name
        if not isinstance(node.target, ast.Name):
            return node
        loopvar = node.target.id

        # Iter must be range(...) call
        if not (isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"):
            return node

        args = node.iter.args
        # Supported forms:
        #   range(len(X) + 1)       — args=[BinOp]
        #   range(0, len(X) + 1)    — args=[0, BinOp]
        if len(args) == 1:
            idx = 0
        elif (len(args) == 2
              and isinstance(args[0], ast.Constant)
              and args[0].value == 0):
            idx = 1
        else:
            return node

        len_call = _len_plus_one_call(args[idx])
        if len_call is None:
            return node

        # len() arg must be a simple Name (container identifier)
        if not (len_call.args and isinstance(len_call.args[0], ast.Name)):
            return node
        container_name = len_call.args[0].id

        # Body must actually subscript container[loopvar] — gate signal
        if not _body_subscripts_container_by_loopvar(
                node.body, container_name, loopvar):
            return node

        # Rewrite: replace the `len(X) + 1` BinOp with the len() Call
        args[idx] = len_call
        self.n_replaced += 1
        return node


def rewrite_off_by_one(code: str) -> RepairResult:
    """If the code has any `for i in range(len(X) + 1):` loop with
    `X[i]` in the body, rewrite `range(len(X) + 1)` → `range(len(X))`.

    Returns RepairResult with kind='off_by_one' when a rewrite fires,
    'none' otherwise.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return RepairResult(None, "none", ["parse failed"])

    writer = _OffByOneRewriter()
    tree = writer.visit(tree)
    if writer.n_replaced == 0:
        return RepairResult(None, "none", ["no off-by-one pattern found"])

    ast.fix_missing_locations(tree)
    try:
        new_code = ast.unparse(tree)
    except Exception as e:
        return RepairResult(None, "none", [f"unparse failed: {e}"])

    return RepairResult(
        new_code, "off_by_one",
        [f"rewrote {writer.n_replaced} range(len(X)+1) → range(len(X))"])


# --------------------------------------------------------------------
# 5. Missing-return rewrite (None-return-signal-driven)
# --------------------------------------------------------------------


# Expressions that plausibly represent a function's return value.
# Deliberately narrow — bare literals alone (e.g. `42`) are excluded
# because a trailing constant on its own line is usually dead code
# or an interactive-style expression, not a forgotten return.
_RETURN_EXPR_TYPES = (
    ast.Name, ast.BinOp, ast.Call, ast.Subscript, ast.IfExp,
    ast.BoolOp, ast.UnaryOp, ast.Compare, ast.List, ast.Tuple,
    ast.Set, ast.Dict, ast.ListComp, ast.SetComp, ast.DictComp,
    ast.GeneratorExp, ast.Attribute,
)


def _function_has_value_return(fn: ast.AST) -> bool:
    """True iff `fn` (FunctionDef/AsyncFunctionDef) contains any
    `return <value>` statement (bare `return` doesn't count — that's
    still an implicit None return).

    Excludes returns inside nested function/lambda definitions.
    """
    for node in ast.walk(fn):
        if node is fn:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            # Don't descend into nested scopes
            continue
    # ast.walk doesn't let us skip subtrees. Do an explicit recursive
    # walk that respects scope boundaries.
    stack: List[ast.AST] = [fn]
    while stack:
        cur = stack.pop()
        for child in ast.iter_child_nodes(cur):
            if child is fn:
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.Lambda)):
                continue   # nested scope: skip
            if isinstance(child, ast.Return) and child.value is not None:
                return True
            stack.append(child)
    return False


def _last_stmt_as_expression(fn_body: List[ast.stmt]) -> Optional[ast.Expr]:
    """If the last statement of `fn_body` is an `Expr(value=X)` where
    X is a plausible return expression, return that Expr. Else None.
    """
    if not fn_body:
        return None
    last = fn_body[-1]
    if not isinstance(last, ast.Expr):
        return None
    if not isinstance(last.value, _RETURN_EXPR_TYPES):
        return None
    # Skip docstring-like standalone strings (unlikely at function tail
    # but defensive)
    if isinstance(last.value, ast.Constant) and isinstance(last.value.value, str):
        return None
    return last


class _MissingReturnRewriter(ast.NodeTransformer):
    """Walk every FunctionDef / AsyncFunctionDef. If it has zero
    value-returns AND its last stmt is `Expr(plausible_expr)`, convert
    that last stmt to `Return(plausible_expr)`.
    """

    def __init__(self):
        self.n_replaced = 0

    def _maybe_rewrite(self, node):
        # Recurse into nested defs first
        self.generic_visit(node)

        if _function_has_value_return(node):
            return node
        last_expr = _last_stmt_as_expression(node.body)
        if last_expr is None:
            return node

        new_return = ast.copy_location(
            ast.Return(value=last_expr.value), last_expr)
        node.body[-1] = new_return
        self.n_replaced += 1
        return node

    def visit_FunctionDef(self, node):
        return self._maybe_rewrite(node)

    def visit_AsyncFunctionDef(self, node):
        return self._maybe_rewrite(node)


def rewrite_missing_return(code: str) -> RepairResult:
    """Find functions whose last stmt is a bare expression and whose
    body has no `return <value>` anywhere, and convert that last stmt
    into `Return(...)`.

    Returns RepairResult with kind='missing_return' on any change.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return RepairResult(None, "none", ["parse failed"])

    writer = _MissingReturnRewriter()
    tree = writer.visit(tree)
    if writer.n_replaced == 0:
        return RepairResult(None, "none", ["no missing-return pattern found"])

    ast.fix_missing_locations(tree)
    try:
        new_code = ast.unparse(tree)
    except Exception as e:
        return RepairResult(None, "none", [f"unparse failed: {e}"])

    return RepairResult(
        new_code, "missing_return",
        [f"added return to {writer.n_replaced} function(s)"])


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------


def repair(code: str, error_output: str) -> RepairResult:
    """Dispatch on error kind. Returns first successful repair, or
    RepairResult(None, 'none', ...) if none applies.

    Caller responsibility: re-run tests on the returned code. Walker
    does no sandboxing itself.
    """
    if not code:
        return RepairResult(None, "none", ["empty code"])

    # 0. Syntax repair — runs FIRST so downstream rewrites see parseable
    # code. Fires when ast.parse fails with a bracket imbalance; no-op
    # when code already parses.
    syntax = repair_syntax(code)
    if syntax.applied:
        return syntax

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

    # 3. Off-by-one — driven by IndexError OOB text. Gated on both the
    # error text and a body-subscript signal (in the walker) to avoid
    # breaking legitimate `range(len(X)+1)` loops.
    if has_indexerror_oob(error_output or ""):
        obo = rewrite_off_by_one(code)
        if obo.applied:
            return obo

    # 4. Missing return — driven by None-return signal in error text.
    # Gated on both error text and walker's no-existing-return check.
    if has_none_return_signal(error_output or ""):
        mret = rewrite_missing_return(code)
        if mret.applied:
            return mret

    return RepairResult(None, "none",
                        ["no applicable rewrite",
                         f"shadow: {shadow.notes}",
                         f"missing_key: {missing}",
                         f"indexerror: {has_indexerror_oob(error_output or '')}",
                         f"none_return: {has_none_return_signal(error_output or '')}"])
