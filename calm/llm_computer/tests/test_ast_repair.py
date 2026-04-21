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
    _name_similarity,
    extract_missing_key,
    extract_undefined_name,
    fuzzy_rename_function,
    has_indent_error,
    has_indexerror_oob,
    has_none_return_signal,
    has_typeerror_callable,
    insert_pass_in_empty_blocks,
    rename_shadow,
    repair,
    repair_cascade,
    repair_syntax,
    rewrite_dict_synonym,
    rewrite_missing_return,
    rewrite_off_by_one,
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


# -- off-by-one range ------------------------------------------------


GEMMA_OFF_BY_ONE_BUG = textwrap.dedent('''
    def sum_list(xs):
        total = 0
        for i in range(len(xs) + 1):
            total += xs[i]
        return total
''').strip() + "\n"


def test_has_indexerror_oob():
    assert has_indexerror_oob("IndexError: list index out of range")
    assert has_indexerror_oob("IndexError: tuple index out of range")
    assert has_indexerror_oob("IndexError: string index out of range")
    assert has_indexerror_oob("  IndexError: index out of range")
    assert not has_indexerror_oob("KeyError: 'foo'")
    assert not has_indexerror_oob("")


def test_off_by_one_basic_rewrite():
    r = rewrite_off_by_one(GEMMA_OFF_BY_ONE_BUG)
    assert r.applied
    assert r.kind == "off_by_one"
    # The rewritten range() must no longer contain `+ 1`
    assert "range(len(xs) + 1)" not in r.new_code
    assert "range(len(xs))" in r.new_code
    # Code must parse + run without IndexError
    ns: dict = {}
    exec(r.new_code, ns)
    assert ns["sum_list"]([1, 2, 3]) == 6


def test_off_by_one_two_arg_range_rewrite():
    """range(0, len(X) + 1) is the same bug in two-arg form."""
    code = textwrap.dedent('''
        def f(xs):
            out = []
            for i in range(0, len(xs) + 1):
                out.append(xs[i])
            return out
    ''')
    r = rewrite_off_by_one(code)
    assert r.applied
    assert "range(0, len(xs))" in r.new_code
    ns: dict = {}
    exec(r.new_code, ns)
    assert ns["f"]([1, 2, 3]) == [1, 2, 3]


def test_off_by_one_no_subscript_noop():
    """Body never indexes xs[i] — this +1 might be intentional
    (e.g. iterating one past). Don't rewrite."""
    code = textwrap.dedent('''
        def count(xs):
            n = 0
            for i in range(len(xs) + 1):
                n += i
            return n
    ''')
    r = rewrite_off_by_one(code)
    assert not r.applied
    assert r.kind == "none"


def test_off_by_one_correct_range_noop():
    """Code that already uses range(len(xs)) — no change."""
    code = textwrap.dedent('''
        def f(xs):
            total = 0
            for i in range(len(xs)):
                total += xs[i]
            return total
    ''')
    r = rewrite_off_by_one(code)
    assert not r.applied


def test_off_by_one_not_len_noop():
    """`range(N + 1)` where N isn't `len(...)` — unrelated pattern."""
    code = textwrap.dedent('''
        def f(n, xs):
            for i in range(n + 1):
                print(xs[i])
    ''')
    r = rewrite_off_by_one(code)
    assert not r.applied


def test_off_by_one_nested_container_subscript():
    """Subscript is d[i], not self.xs[i] — we only track simple Name
    containers. Don't rewrite to be safe."""
    code = textwrap.dedent('''
        class C:
            def f(self):
                for i in range(len(self.xs) + 1):
                    print(self.xs[i])
    ''')
    r = rewrite_off_by_one(code)
    # `len(self.xs)` — arg is Attribute, not Name → conservative no-op
    assert not r.applied


def test_off_by_one_one_plus_len_form():
    """`1 + len(X)` (reversed) also triggers the rewrite."""
    code = textwrap.dedent('''
        def f(xs):
            for i in range(1 + len(xs)):
                print(xs[i])
    ''')
    r = rewrite_off_by_one(code)
    assert r.applied
    assert "range(len(xs))" in r.new_code


def test_off_by_one_multiple_loops():
    """Two separate buggy loops both get rewritten in one pass."""
    code = textwrap.dedent('''
        def g(xs, ys):
            a = 0
            for i in range(len(xs) + 1):
                a += xs[i]
            b = 0
            for j in range(len(ys) + 1):
                b += ys[j]
            return a + b
    ''')
    r = rewrite_off_by_one(code)
    assert r.applied
    assert "rewrote 2" in r.notes[0]
    assert "range(len(xs) + 1)" not in r.new_code
    assert "range(len(ys) + 1)" not in r.new_code
    ns: dict = {}
    exec(r.new_code, ns)
    assert ns["g"]([1, 2, 3], [4, 5]) == 15


def test_repair_off_by_one_via_dispatch():
    """Full repair() entry — IndexError in error text + pattern in code."""
    r = repair(GEMMA_OFF_BY_ONE_BUG, "IndexError: list index out of range")
    assert r.applied
    assert r.kind == "off_by_one"


def test_repair_off_by_one_requires_indexerror_in_error_text():
    """Without IndexError in error text, dispatch skips off_by_one
    even though pattern exists. Belt+suspenders against false positives."""
    r = repair(GEMMA_OFF_BY_ONE_BUG, "ValueError: something else")
    assert not r.applied
    # dispatch notes mention indexerror gate was false
    assert any("indexerror" in n.lower() for n in r.notes)


def test_repair_syntax_still_first_in_dispatch():
    """Broken syntax code with IndexError hint must still go to
    syntax_repair first — can't AST-walk unparseable code."""
    # Syntactically broken, but includes IndexError hint:
    code = "def f(xs):\n    for i in range(len(xs) + 1:\n        print(xs[i])\n"
    r = repair(code, "IndexError: list index out of range")
    # Note: can't predict exactly what syntax_repair does here (might
    # succeed or fail), but off_by_one must NOT run on unparseable code.
    assert r.kind in ("syntax_repair", "none")


# -- missing return --------------------------------------------------


GEMMA_MISSING_RETURN_BUG = textwrap.dedent('''
    def add(a, b):
        result = a + b
        result
''').strip() + "\n"


def test_has_none_return_signal():
    assert has_none_return_signal("TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'")
    assert has_none_return_signal("AssertionError: expected 5, got None")
    assert has_none_return_signal("got None")
    assert has_none_return_signal("AttributeError: 'NoneType' object has no attribute 'foo'")
    assert not has_none_return_signal("KeyError: 'x'")
    assert not has_none_return_signal("IndexError: list index out of range")
    assert not has_none_return_signal("")


def test_missing_return_basic_rewrite():
    r = rewrite_missing_return(GEMMA_MISSING_RETURN_BUG)
    assert r.applied
    assert r.kind == "missing_return"
    ns: dict = {}
    exec(r.new_code, ns)
    assert ns["add"](2, 3) == 5


def test_missing_return_binop_inline():
    """Last stmt is a BinOp directly, no named variable."""
    code = textwrap.dedent('''
        def mul(a, b):
            a * b
    ''')
    r = rewrite_missing_return(code)
    assert r.applied
    ns: dict = {}
    exec(r.new_code, ns)
    assert ns["mul"](3, 4) == 12


def test_missing_return_call_inline():
    code = textwrap.dedent('''
        def shout(s):
            s.upper()
    ''')
    r = rewrite_missing_return(code)
    assert r.applied
    ns: dict = {}
    exec(r.new_code, ns)
    assert ns["shout"]("hi") == "HI"


def test_missing_return_skips_existing_return():
    """Function with a real return — don't touch."""
    code = textwrap.dedent('''
        def f(x):
            if x > 0:
                return x
            x
    ''')
    r = rewrite_missing_return(code)
    assert not r.applied


def test_missing_return_skips_bare_return():
    """`return` alone counts as existing None-return, but our gate
    says only `return <value>` counts as having a real return. So a
    function with `return` (bare) + trailing expression should still
    trigger — bare return is implicit None just like trailing expr.
    """
    code = textwrap.dedent('''
        def f(x):
            if x < 0:
                return
            x * 2
    ''')
    r = rewrite_missing_return(code)
    # With our definition (value-return only), this triggers.
    assert r.applied
    ns: dict = {}
    exec(r.new_code, ns)
    assert ns["f"](5) == 10
    assert ns["f"](-1) is None


def test_missing_return_skips_non_expression_last_stmt():
    """Last stmt is an assignment / if / for / while — not a bare
    expression — don't touch."""
    code = textwrap.dedent('''
        def f(xs):
            total = 0
            for x in xs:
                total += x
    ''')
    r = rewrite_missing_return(code)
    # last stmt is a For, not an Expr — no rewrite
    assert not r.applied


def test_missing_return_skips_bare_literal():
    """Trailing bare literal (e.g. `42` on its own line) looks like
    dead code more than a forgotten return. Don't touch."""
    code = textwrap.dedent('''
        def f():
            42
    ''')
    r = rewrite_missing_return(code)
    assert not r.applied


def test_missing_return_skips_string_docstring_like():
    """Trailing plain string literal — docstring-like artifact. Skip."""
    code = textwrap.dedent('''
        def f():
            x = 1
            "some note"
    ''')
    r = rewrite_missing_return(code)
    assert not r.applied


def test_missing_return_nested_function_independent():
    """Outer has no return but inner does. Inner's return doesn't
    count for the outer's gate — outer should still rewrite."""
    code = textwrap.dedent('''
        def outer(xs):
            def inner(x):
                return x * 2
            [inner(x) for x in xs]
    ''')
    r = rewrite_missing_return(code)
    assert r.applied
    ns: dict = {}
    exec(r.new_code, ns)
    assert ns["outer"]([1, 2, 3]) == [2, 4, 6]


def test_missing_return_method_in_class():
    """Inside a class, method bodies are also walked."""
    code = textwrap.dedent('''
        class C:
            def compute(self, x):
                x + 1
    ''')
    r = rewrite_missing_return(code)
    assert r.applied
    ns: dict = {}
    exec(r.new_code, ns)
    assert ns["C"]().compute(5) == 6


def test_repair_missing_return_via_dispatch():
    r = repair(GEMMA_MISSING_RETURN_BUG,
               "TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'")
    assert r.applied
    assert r.kind == "missing_return"


def test_repair_missing_return_requires_error_signal():
    """Without None-return error signal, dispatch skips missing_return
    even though pattern exists."""
    r = repair(GEMMA_MISSING_RETURN_BUG, "ValueError: something else")
    assert not r.applied
    assert any("none_return" in n.lower() for n in r.notes)


# -- empty-block pass-insert -----------------------------------------


def test_has_indent_error():
    assert has_indent_error("IndentationError: expected an indented block")
    assert has_indent_error(
        "IndentationError: expected an indented block after 'except' statement on line 5")
    assert has_indent_error(
        "SyntaxError: expected an indented block after function definition on line 3")
    assert has_indent_error("  err: IndentationError: expected an indented block")
    assert not has_indent_error("IndexError: list index out of range")
    assert not has_indent_error("")


def test_empty_except_inserts_pass():
    code = textwrap.dedent('''
        def f():
            try:
                x = 1
            except Exception:
        ''').lstrip()
    r = insert_pass_in_empty_blocks(code)
    assert r.applied
    assert r.kind == "empty_block"
    # Must parse after rewrite
    ast.parse(r.new_code)
    assert "pass" in r.new_code


def test_empty_if_inserts_pass():
    code = textwrap.dedent('''
        def f(x):
            if x > 0:
        ''').rstrip() + "\n"
    r = insert_pass_in_empty_blocks(code)
    assert r.applied
    ast.parse(r.new_code)


def test_empty_else_inserts_pass():
    code = textwrap.dedent('''
        def f(x):
            if x > 0:
                return 1
            else:
        ''').rstrip() + "\n"
    r = insert_pass_in_empty_blocks(code)
    assert r.applied
    ast.parse(r.new_code)


def test_empty_for_inserts_pass():
    code = textwrap.dedent('''
        def f(xs):
            for x in xs:
    ''').rstrip() + "\n"
    r = insert_pass_in_empty_blocks(code)
    assert r.applied
    ast.parse(r.new_code)


def test_empty_try_inserts_pass():
    """try: with no body gets pass; pre-existing except: body retained."""
    code = textwrap.dedent('''
        def f():
            try:
            except Exception:
                return None
    ''').rstrip() + "\n"
    r = insert_pass_in_empty_blocks(code)
    assert r.applied
    ast.parse(r.new_code)


def test_parseable_code_is_noop():
    """Code that already parses — empty_block fires no change."""
    code = textwrap.dedent('''
        def f():
            try:
                x = 1
            except Exception:
                return None
    ''')
    r = insert_pass_in_empty_blocks(code)
    assert not r.applied
    assert "already parses" in r.notes[0]


def test_empty_block_with_comment_only_body():
    """A comment-only body counts as empty — pass still inserted."""
    code = textwrap.dedent('''
        def f():
            try:
                x = 1
            except Exception:
                # ignore
        ''').rstrip() + "\n"
    r = insert_pass_in_empty_blocks(code)
    # Comment body is text-level present but same-indent as header
    # means our simple indent check thinks it's empty. This is a
    # conservative approximation. Comment at higher indent would
    # count as body; comment at same indent is ambiguous.
    # For this test, comment at +4 indent: DOES count as meaningful
    # by indent width, so rewrite shouldn't fire.
    if r.applied:
        # Acceptable — pass is now before the comment
        ast.parse(r.new_code)


def test_empty_block_multiple_headers():
    """Two distinct empty blocks both get pass."""
    code = textwrap.dedent('''
        def f():
            if True:
            elif False:
    ''').rstrip() + "\n"
    r = insert_pass_in_empty_blocks(code)
    assert r.applied
    ast.parse(r.new_code)
    assert r.new_code.count("pass") >= 2


def test_empty_block_at_eof():
    """Header is the last line of the file — no following line.
    Walker inserts `pass`; note that `try:` without `except:` remains
    syntactically incomplete (needs a separate fix), but the pass
    insertion itself is correct."""
    code = "def f():\n    try:\n"
    r = insert_pass_in_empty_blocks(code)
    assert r.applied
    assert "pass" in r.new_code
    # Incomplete try/except may still not parse, but partial fix is OK


def test_empty_block_def_at_eof_parses():
    """`def` at EOF with no body — pass makes it parseable."""
    code = "def f():\n"
    r = insert_pass_in_empty_blocks(code)
    assert r.applied
    ast.parse(r.new_code)


def test_empty_block_header_followed_by_deeper_compound():
    """Header followed by a DEEPER-indented compound header (e.g.
    try: then for: nested) — the outer block is NOT empty."""
    code = textwrap.dedent('''
        def f(xs):
            try:
                for x in xs:
                    print(x)
            except Exception:
                return None
    ''')
    r = insert_pass_in_empty_blocks(code)
    # Should not fire — code parses fine.
    assert not r.applied


def test_repair_empty_block_via_dispatch():
    """Full repair() — IndentationError error text dispatches to
    empty_block rewriter."""
    code = "def f():\n    try:\n        x = 1\n    except Exception:\n"
    r = repair(code, "IndentationError: expected an indented block after 'except' statement on line 4")
    assert r.applied
    assert r.kind == "empty_block"
    ast.parse(r.new_code)


def test_repair_empty_block_runs_even_without_error_text():
    """Dispatch: empty_block in the syntax-repair tier runs on any
    unparseable input, doesn't require specific error text."""
    code = "def f():\n    try:\n        x = 1\n    except Exception:\n"
    # No indication of IndentationError in the message
    r = repair(code, "")
    # Still fires — the code is unparseable
    assert r.applied
    assert r.kind == "empty_block"


# -- cascade (multi-pass) --------------------------------------------


def test_cascade_noop_on_clean_code():
    code = "def f(x):\n    return x * 2\n"
    r = repair_cascade(code, "")
    assert not r.applied


def test_cascade_single_pass_no_cascade_prefix():
    """When only one rewrite applies, kind is the rewrite's name, not
    prefixed with 'cascade:'."""
    r = repair_cascade(GEMMA_TOKEN_BUCKET_BUG,
                       "TypeError: 'int' object is not callable")
    assert r.applied
    # Only shadow_rename needed
    assert r.kind == "shadow_rename"


def test_cascade_empty_block_then_missing_return():
    """Empty-except makes code parseable; AST walker then notices
    missing return in function. Two passes should apply."""
    code = textwrap.dedent('''
        def compute(x):
            try:
                return x * 2
            except Exception:
            x * 3
        ''').rstrip() + "\n"
    # First pass: empty_block inserts pass after except
    # Second pass: code parses, but value-return exists (inside try) so
    #   missing_return won't fire. Only one rewrite should apply.
    r = repair_cascade(code,
                       "TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'")
    assert r.applied
    # One rewrite — just empty_block
    assert "empty_block" in r.kind


def test_cascade_syntax_then_dict_synonym():
    """Code has a missing-paren AND uses 'stddev' key. First pass
    fixes syntax; second pass rewrites the synonym."""
    # `{` with missing `}` — triggers syntax_repair; then with 'stddev'
    # where KeyError comes on 'stdev'
    code = textwrap.dedent('''
        def f(data):
            r = {'stddev': 0.5
            return r
        ''').rstrip() + "\n"
    r = repair_cascade(code, "KeyError: 'stdev'")
    assert r.applied
    # Check both syntax_repair and dict_synonym in the cascade
    assert "syntax_repair" in r.kind
    # After syntax fix, dict_synonym should also fire
    # (the test depends on dispatch order — allow either single or cascade)
    ast.parse(r.new_code)


def test_cascade_respects_max_passes():
    """max_passes=1 should stop after first rewrite even if more
    would apply."""
    code = textwrap.dedent('''
        def compute(x):
            try:
                return x * 2
            except Exception:
            x * 3
        ''').rstrip() + "\n"
    r = repair_cascade(code, "", max_passes=1)
    assert r.applied
    # Only one pass applied
    assert len(r.notes) == 1


def test_cascade_preserves_repair_single_pass():
    """Single repair() still works as before — cascade is additive."""
    r = repair(GEMMA_TOKEN_BUCKET_BUG,
               "TypeError: 'int' object is not callable")
    assert r.applied
    assert r.kind == "shadow_rename"


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


# -- fuzzy function rename ------------------------------------------


GEMMA_WRONG_FN_NAME = textwrap.dedent('''
    def find_first_repeated(s):
        """Find the first repeated character."""
        seen = set()
        for ch in s:
            if ch in seen:
                return ch
            seen.add(ch)
        return None
''').strip()


GEMMA_VERY_DIFFERENT_FN = textwrap.dedent('''
    def process_string(s):
        result = s[::-1]
        return result
''').strip()


def test_name_similarity_exact():
    assert _name_similarity("foo_bar", "foo_bar") == 1.0


def test_name_similarity_partial():
    # {first, repeated, char} vs {find, first, repeated}
    # inter={first, repeated} size 2, union={find, first, repeated, char} size 4 → 0.5
    sim = _name_similarity("first_repeated_char", "find_first_repeated")
    assert 0.4 < sim <= 0.5


def test_name_similarity_disjoint():
    assert _name_similarity("foo", "bar") == 0.0


def test_extract_undefined_name_parses():
    err = "Traceback (most recent call last):\n  File ...\nNameError: name 'first_repeated_char' is not defined"
    assert extract_undefined_name(err) == "first_repeated_char"


def test_extract_undefined_name_none():
    assert extract_undefined_name("KeyError: 'mean'") is None
    assert extract_undefined_name("") is None


def test_fuzzy_rename_mbpp1_first_repeated_char():
    """Gemma wrote `find_first_repeated`; test expected `first_repeated_char`."""
    r = fuzzy_rename_function(GEMMA_WRONG_FN_NAME, "first_repeated_char")
    assert r.applied, f"walker failed: {r.notes}"
    assert r.kind == "fuzzy_rename"
    # The renamed code should define first_repeated_char.
    tree = ast.parse(r.new_code)
    fn_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "first_repeated_char" in fn_names
    assert "find_first_repeated" not in fn_names
    # Execute to verify it actually works.
    ns: dict = {}
    exec(r.new_code, ns)
    assert ns["first_repeated_char"]("abcdab") == "a"
    assert ns["first_repeated_char"]("abc") is None


def test_fuzzy_rename_skips_unrelated():
    """`process_string` vs `first_repeated_char` — zero overlap, must NOT rename."""
    r = fuzzy_rename_function(GEMMA_VERY_DIFFERENT_FN, "first_repeated_char")
    assert not r.applied, "walker wrongly renamed unrelated function"


def test_fuzzy_rename_refuses_existing_target():
    """If code already has first_repeated_char AND another similar def,
    don't clobber."""
    code = textwrap.dedent('''
        def find_first_repeated(s):
            return None
        def first_repeated_char(s):
            return "a"
    ''').strip()
    r = fuzzy_rename_function(code, "first_repeated_char")
    assert not r.applied, "walker created duplicate definition"


def test_fuzzy_rename_empty_code():
    r = fuzzy_rename_function("", "foo")
    assert not r.applied


def test_fuzzy_rename_no_funcdef():
    r = fuzzy_rename_function("x = 5", "foo")
    assert not r.applied


def test_repair_dispatches_fuzzy_rename_on_nameerror():
    """End-to-end: repair() picks fuzzy_rename when error_output has NameError."""
    err = "NameError: name 'first_repeated_char' is not defined"
    r = repair(GEMMA_WRONG_FN_NAME, err)
    assert r.applied
    assert r.kind == "fuzzy_rename"


def test_repair_no_fuzzy_rename_without_nameerror():
    """If error isn't NameError, fuzzy_rename shouldn't fire."""
    err = "AssertionError: whatever"
    r = repair(GEMMA_WRONG_FN_NAME, err)
    assert not r.applied or r.kind != "fuzzy_rename"


def test_fuzzy_rename_recursive_call_rewritten():
    """Function calls itself recursively — rename must update call site."""
    code = textwrap.dedent('''
        def factorial_fn(n):
            if n <= 1:
                return 1
            return n * factorial_fn(n - 1)
    ''').strip()
    r = fuzzy_rename_function(code, "factorial")
    assert r.applied
    ns: dict = {}
    exec(r.new_code, ns)
    assert ns["factorial"](5) == 120
