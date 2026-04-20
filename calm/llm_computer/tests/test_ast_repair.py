"""Unit tests for the tier-2 AST walker (ast_repair.py).

Each test uses the actual failure pattern from R53.33:
- token_bucket_rate_limiter: self.tokens attribute shadowing method
- csv_column_stats: dict literal key 'avg' when tests need 'mean'
"""

from __future__ import annotations

import ast
import textwrap

import pytest

from calm.llm_computer.facades.ast_repair import (
    DICT_KEY_SYNONYMS,
    RepairResult,
    _balance_brackets_on_line,
    _find_shadowed_attrs,
    extract_missing_key,
    has_typeerror_callable,
    rename_shadow,
    repair,
    repair_syntax,
    rewrite_dict_synonym,
)


# -- shadow rename ---------------------------------------------------


GEMMA_TOKEN_BUCKET_BUG = textwrap.dedent('''
    import time

    class TokenBucket:
        def __init__(self, rate, capacity):
            self.rate = rate
            self.capacity = capacity
            self.tokens = capacity
            self.last = time.monotonic()

        def _refill(self):
            now = time.monotonic()
            elapsed = now - self.last
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last = now

        def allow(self):
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

        def tokens(self):
            self._refill()
            return self.tokens
''').strip()


def test_shadow_detect():
    tree = ast.parse(GEMMA_TOKEN_BUCKET_BUG)
    pairs = _find_shadowed_attrs(tree)
    assert ("TokenBucket", "tokens") in pairs


def test_shadow_rename_applied():
    r = rename_shadow(GEMMA_TOKEN_BUCKET_BUG)
    assert r.applied
    assert r.kind == "shadow_rename"
    assert "TokenBucket.tokens -> TokenBucket._tokens" in r.notes[0]


def test_shadow_rename_compiles_and_runs():
    r = rename_shadow(GEMMA_TOKEN_BUCKET_BUG)
    assert r.applied
    # Compile + execute the rewritten code
    ns: dict = {}
    exec(r.new_code, ns)
    TokenBucket = ns["TokenBucket"]
    tb = TokenBucket(rate=10, capacity=5)
    # Method call works — this was the R53.33 TypeError site
    assert abs(tb.tokens() - 5.0) < 1e-6
    # Drain capacity
    for _ in range(5):
        assert tb.allow() is True
    # Empty bucket — may True if refill during sleep; at least test no crash
    _ = tb.allow()


def test_shadow_rename_preserves_method_body():
    """The rewriter must NOT touch self.tokens inside the method's
    own body at the Attribute read-sites that REFER to the attr.
    But inside the method body, `return self.tokens` must become
    `return self._tokens` — because the method needs to read its
    own backing attribute. Non-call reads always get renamed."""
    r = rename_shadow(GEMMA_TOKEN_BUCKET_BUG)
    # Method def `def tokens(self):` intact
    assert "def tokens(self)" in r.new_code
    # Read site renamed: returns the backing attribute
    assert "return self._tokens" in r.new_code
    # Write sites renamed: attr assignment uses new name
    assert "self._tokens = capacity" in r.new_code
    assert "self._tokens -= 1" in r.new_code
    # Call sites preserved: self._refill() on method call in allow()
    assert "self._refill()" in r.new_code


def test_shadow_no_op_when_no_shadow():
    code = textwrap.dedent('''
        class Foo:
            def __init__(self):
                self.x = 1
            def bar(self):
                return self.x
    ''').strip()
    r = rename_shadow(code)
    assert not r.applied
    assert r.kind == "none"


def test_shadow_handles_augassign():
    code = textwrap.dedent('''
        class C:
            def __init__(self):
                self.counter = 0
            def counter(self):
                return self.counter
            def inc(self):
                self.counter += 1
    ''').strip()
    r = rename_shadow(code)
    assert r.applied
    assert "self._counter += 1" in r.new_code


def test_shadow_skips_other_class_same_attr():
    code = textwrap.dedent('''
        class A:
            def __init__(self):
                self.x = 1
            def x(self):
                return self.x

        class B:
            def __init__(self):
                self.x = 2   # B has no method x — must NOT rename
            def y(self):
                return self.x
    ''').strip()
    r = rename_shadow(code)
    assert r.applied
    # A got renamed, B left alone. Reparse and inspect.
    tree = ast.parse(r.new_code)
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        methods = {m.name for m in cls.body
                   if isinstance(m, ast.FunctionDef)}
        attr_writes = set()
        for m in cls.body:
            if not isinstance(m, ast.FunctionDef):
                continue
            for node in ast.walk(m):
                if (isinstance(node, ast.Assign) and
                        len(node.targets) == 1 and
                        isinstance(node.targets[0], ast.Attribute) and
                        isinstance(node.targets[0].value, ast.Name) and
                        node.targets[0].value.id == "self"):
                    attr_writes.add(node.targets[0].attr)
        if cls.name == "A":
            assert "x" not in attr_writes  # renamed
            assert "_x" in attr_writes
            assert "x" in methods            # method preserved
        elif cls.name == "B":
            assert "x" in attr_writes        # untouched


# -- dict synonym ----------------------------------------------------


GEMMA_CSV_BUG = textwrap.dedent('''
    import csv
    import statistics
    from io import StringIO

    def csv_column_stats(text):
        if not text:
            return {}
        reader = csv.reader(StringIO(text))
        rows = list(reader)
        header = rows[0]
        data = rows[1:]
        result = {}
        for i, col in enumerate(header):
            vals = []
            ok = True
            for row in data:
                try:
                    vals.append(float(row[i]))
                except ValueError:
                    ok = False
                    break
            if not ok or not vals:
                continue
            std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
            result[col] = {"avg": sum(vals)/len(vals), "std": std,
                           "min": min(vals), "max": max(vals)}
        return result
''').strip()


def test_synonym_mean_detected():
    r = rewrite_dict_synonym(GEMMA_CSV_BUG, "mean")
    assert r.applied
    assert "'mean'" in r.new_code or '"mean"' in r.new_code
    assert "'avg'" not in r.new_code and '"avg"' not in r.new_code


def test_synonym_stdev_detected():
    r = rewrite_dict_synonym(GEMMA_CSV_BUG, "stdev")
    assert r.applied
    assert "'stdev'" in r.new_code or '"stdev"' in r.new_code


def test_synonym_no_op_when_no_synonyms():
    # 'score' is not in the synonym table
    r = rewrite_dict_synonym(GEMMA_CSV_BUG, "score")
    assert not r.applied
    assert r.kind == "none"


def test_synonym_no_op_when_key_already_present():
    code = textwrap.dedent('''
        def f():
            return {"mean": 1, "stdev": 2}
    ''').strip()
    r = rewrite_dict_synonym(code, "mean")
    # No synonym of 'mean' appears, so nothing to rewrite
    assert not r.applied


def test_synonym_rewrites_subscript_access():
    """If Gemma BOTH constructs and reads 'avg', we should rewrite
    both sides. Test: the constructed dict + any subscript access."""
    code = textwrap.dedent('''
        def f():
            d = {"avg": 1}
            return d["avg"] + 1
    ''').strip()
    r = rewrite_dict_synonym(code, "mean")
    assert r.applied
    # Both the literal AND the subscript are now 'mean'
    assert "'mean': 1" in r.new_code or '"mean": 1' in r.new_code
    assert "d['mean']" in r.new_code or 'd["mean"]' in r.new_code


def test_synonym_rewrites_get_method():
    code = textwrap.dedent('''
        def f(d):
            return d.get("average", 0)
    ''').strip()
    r = rewrite_dict_synonym(code, "mean")
    assert r.applied
    assert "'mean'" in r.new_code or '"mean"' in r.new_code


# -- error parsing ---------------------------------------------------


def test_extract_missing_key():
    assert extract_missing_key("KeyError: 'mean'") == "mean"
    assert extract_missing_key('KeyError: "score"') == "score"
    assert extract_missing_key("nothing") is None


def test_has_typeerror_callable():
    assert has_typeerror_callable("TypeError: 'float' object is not callable")
    assert has_typeerror_callable("TypeError: 'int' object is not callable")
    assert not has_typeerror_callable("TypeError: other")


# -- integration entry point ----------------------------------------


def test_repair_dispatches_shadow_on_callable_error():
    r = repair(GEMMA_TOKEN_BUCKET_BUG,
               "TypeError: 'float' object is not callable")
    assert r.applied
    assert r.kind == "shadow_rename"


def test_repair_dispatches_synonym_on_keyerror():
    r = repair(GEMMA_CSV_BUG, "KeyError: 'mean'")
    assert r.applied
    assert r.kind == "dict_synonym"


def test_repair_returns_none_when_no_applicable():
    code = textwrap.dedent('''
        def f(x): return x + 1
    ''').strip()
    r = repair(code, "ValueError: bad input")
    assert not r.applied
    assert r.kind == "none"


def test_repair_shadow_runs_unconditionally_without_error_text():
    """Shadow detector runs even when error text is empty — it's
    static, doesn't need a trigger. This lets the walker rescue code
    whose symptom surfaces as something other than TypeError callable."""
    r = repair(GEMMA_TOKEN_BUCKET_BUG, "")
    assert r.applied
    assert r.kind == "shadow_rename"


# -- end-to-end on the real R53 failure ------------------------------


def test_end_to_end_token_bucket_passes_after_repair():
    """Rewrite the Gemma-buggy code and run the R53 token_bucket
    tests against it. Must pass."""
    r = repair(GEMMA_TOKEN_BUCKET_BUG,
               "TypeError: 'float' object is not callable")
    assert r.applied

    # Execute the rewritten code + the actual R53 test sub-case
    ns: dict = {}
    exec(r.new_code, ns)
    TokenBucket = ns["TokenBucket"]

    # Start full
    tb = TokenBucket(rate=10, capacity=5)
    assert abs(tb.tokens() - 5.0) < 1e-6

    # Consume up to capacity (slow refill)
    tb2 = TokenBucket(rate=0.1, capacity=3)
    successes = sum(tb2.allow() for _ in range(3))
    assert successes == 3
    assert not tb2.allow()


# -- syntax repair (balanced brackets) ------------------------------


def test_balance_mid_expression_imbalance_returns_none():
    """_balance_brackets_on_line is naive append-at-end only. A
    mid-expression imbalance where `}` closes a `{` before all inner
    `(` are closed returns None — let _repair_mismatch handle it via
    the SyntaxError offset path."""
    line = "    column_data = {header[i]: [] for i in range(len(header)}"
    fixed = _balance_brackets_on_line(line)
    # `}` with an unclosed `(` ahead of it → naive balancer bails.
    assert fixed is None


def test_balance_append_missing_closer_at_end():
    line = "x = func(a, b, c"
    fixed = _balance_brackets_on_line(line)
    assert fixed == "x = func(a, b, c)"


def test_balance_nested_missing():
    line = "y = f(g(h(z"
    fixed = _balance_brackets_on_line(line)
    assert fixed == "y = f(g(h(z)))"


def test_balance_already_balanced_is_noop():
    line = "x = func(a, b)"
    fixed = _balance_brackets_on_line(line)
    assert fixed is None


def test_balance_preserves_trailing_comment():
    line = "x = func(a  # missing close"
    fixed = _balance_brackets_on_line(line)
    assert fixed is not None
    assert fixed.endswith("# missing close")
    assert "func(a)" in fixed


def test_balance_returns_none_on_excess_closer():
    line = "x = func(a))"
    fixed = _balance_brackets_on_line(line)
    assert fixed is None


def test_balance_ignores_brackets_in_strings():
    line = "msg = 'this has ( unmatched paren in string'"
    fixed = _balance_brackets_on_line(line)
    # String-internal `(` doesn't count as an opener
    assert fixed is None


def test_repair_syntax_single_missing_paren():
    code = "def f():\n    x = func(a, b\n    return x\n"
    r = repair_syntax(code)
    assert r.applied
    assert r.kind == "syntax_repair"
    ast.parse(r.new_code)


def test_repair_syntax_noop_on_valid():
    code = "def f():\n    return 42\n"
    r = repair_syntax(code)
    assert not r.applied
    assert "already parses" in r.notes[0]


def test_repair_syntax_unfixable_returns_none():
    # Genuine syntax error that isn't a bracket imbalance — missing colon
    code = "def f()\n    pass\n"
    r = repair_syntax(code)
    assert not r.applied


def test_repair_syntax_missing_paren_before_colon():
    """The R53.35v2 csv pattern: `for i in range(min(a, len(row)):` —
    missing `)` before the `:`. Python reports "invalid syntax" with
    offset at the `:`. Balancer must insert `)` BEFORE the trailing
    colon, not append at end."""
    code = (
        "def f(a, data):\n"
        "    for i in range(min(a, len(data)):\n"
        "        pass\n"
    )
    r = repair_syntax(code)
    assert r.applied, f"expected syntax_repair: {r.notes}"
    ast.parse(r.new_code)
    assert "range(min(a, len(data)))" in r.new_code


def test_repair_syntax_def_with_missing_close_paren():
    """def f(: pattern — fixed to def f(): by insert-before-colon."""
    code = "def f(:\n    pass\n"
    r = repair_syntax(code)
    assert r.applied
    ast.parse(r.new_code)
    assert "def f():" in r.new_code


def test_repair_syntax_gemma_csv_pattern():
    """The exact R53.35 bug pattern — dict comp with unclosed paren.
    `{header[i]: [] for i in range(len(header)}`: the `{/}` pair is
    balanced but the inner `range(len(header)` is missing its `)`.

    Python's parser reports: "closing parenthesis '}' does not match
    opening parenthesis '('" at the `}` offset. _repair_mismatch
    inserts `)` before the `}` → `range(len(header))}`, which parses."""
    code = (
        "def f(header, rows):\n"
        "    column_data = {header[i]: [] for i in range(len(header)}\n"
        "    return column_data\n"
    )
    r = repair_syntax(code)
    assert r.applied, f"expected mismatch repair to fire: {r.notes}"
    assert r.kind == "syntax_repair"
    assert "inserted ')'" in r.notes[0]
    ast.parse(r.new_code)   # parses now
    # Confirm the fix is minimal — only one `)` added on line 2
    assert "range(len(header))" in r.new_code


def test_repair_syntax_gemma_csv_three_copies():
    """Gemma's actual raw output had THREE identical unclosed parens.
    Repair must iterate through all of them."""
    code = (
        "def f(h, data):\n"
        "    a = {h[i]: [] for i in range(len(h)}\n"
        "    b = {h[i]: [] for i in range(len(h)}\n"
        "    c = {h[i]: [] for i in range(len(h)}\n"
        "    return a, b, c\n"
    )
    r = repair_syntax(code)
    assert r.applied
    assert len(r.notes) == 3   # one fix per duplicated bug line
    ast.parse(r.new_code)


def test_repair_dispatches_syntax_first():
    """Syntax repair runs before shadow/synonym — broken code can't
    be walked for shadow anyway."""
    code = "def f():\n    x = func(a, b\n    return x\n"
    r = repair(code, "SyntaxError: '(' was never closed")
    assert r.applied
    assert r.kind == "syntax_repair"


def test_end_to_end_csv_column_stats_passes_after_repair():
    """Rewrite the Gemma-buggy csv code for both synonym sub-keys
    ('mean' then 'stdev') and run the R53 csv test sub-case."""
    # First pass: KeyError 'mean'
    r1 = repair(GEMMA_CSV_BUG, "KeyError: 'mean'")
    assert r1.applied
    # Second pass on the rewritten code: KeyError 'stdev'
    r2 = repair(r1.new_code, "KeyError: 'stdev'")
    assert r2.applied

    ns: dict = {}
    exec(r2.new_code, ns)
    csv_column_stats = ns["csv_column_stats"]
    t = "name,age,score\nAlice,30,95.5\nBob,25,82.0\nCarol,35,88.5"
    r = csv_column_stats(t)
    assert "name" not in r     # non-numeric skipped
    assert "age" in r
    assert abs(r["age"]["mean"] - 30.0) < 1e-9   # THE fix works
    assert r["age"]["min"] == 25.0
    assert r["age"]["max"] == 35.0
    assert abs(r["score"]["mean"] - 88.666666) < 1e-3
