"""Unit tests for ast_refactor primitives."""
import ast

import pytest

from calm.llm_computer.ast_refactor import (
    RefactorSession,
    convert_loop_to_comprehension,
    detect_refactor_opportunities,
    extract_method,
    inline_variable,
    rename_variable,
)


# ==================================================================
# rename_variable
# ==================================================================


def test_rename_variable_module_scope():
    code = """
x = 1
y = x + 2
def f():
    return x
"""
    r = rename_variable(code, "x", "value")
    assert r.applied
    assert "x" not in r.new_code
    assert "value" in r.new_code
    # Sanity: result compiles + executes
    ns = {}
    exec(r.new_code, ns)
    assert ns["f"]() == 1
    assert ns["y"] == 3


def test_rename_variable_scope_restricted_to_function():
    code = """
x = 100

def f():
    x = 1
    return x

def g():
    return x
"""
    r = rename_variable(code, "x", "local", scope="f")
    assert r.applied
    # Module-level x and g's reference untouched
    assert "x = 100" in r.new_code
    assert "return x" in r.new_code  # g's return
    # f's local renamed
    assert "local = 1" in r.new_code
    ns = {}
    exec(r.new_code, ns)
    assert ns["f"]() == 1
    assert ns["g"]() == 100


def test_rename_variable_collision_refused():
    code = """
x = 1
y = 2
z = x + y
"""
    r = rename_variable(code, "x", "y")
    assert not r.applied
    assert "collision" in r.error


def test_rename_variable_noop_on_same_name():
    code = "x = 1\n"
    r = rename_variable(code, "x", "x")
    assert r.applied
    assert r.n_changes == 0


def test_rename_variable_nonexistent_var():
    r = rename_variable("x = 1", "nonesuch", "something_else")
    assert not r.applied
    assert "not found" in r.error


def test_rename_variable_function_args():
    code = """
def f(x, y):
    return x + y
"""
    r = rename_variable(code, "x", "alpha", scope="f")
    assert r.applied
    ns = {}
    exec(r.new_code, ns)
    assert ns["f"](1, 2) == 3
    assert "def f(alpha, y)" in r.new_code


def test_rename_variable_preserves_semantics_across_scope():
    """Rename inside a function must not touch outer references."""
    code = """
counter = 0

def increment():
    counter = 1
    return counter
"""
    r = rename_variable(code, "counter", "local_counter", scope="increment")
    assert r.applied
    ns = {}
    exec(r.new_code, ns)
    assert ns["counter"] == 0
    assert ns["increment"]() == 1


# ==================================================================
# inline_variable
# ==================================================================


def test_inline_variable_simple():
    code = """
x = 5
y = x + 1
z = x * 2
"""
    r = inline_variable(code, "x")
    assert r.applied
    assert "x = 5" not in r.new_code
    ns = {}
    exec(r.new_code, ns)
    assert ns["y"] == 6
    assert ns["z"] == 10


def test_inline_variable_refused_on_reassign():
    code = """
x = 1
x = 2
y = x
"""
    r = inline_variable(code, "x")
    assert not r.applied
    assert "not single-binding" in r.error


def test_inline_variable_refused_on_call_value():
    code = """
x = input()
y = x
"""
    r = inline_variable(code, "x")
    assert not r.applied
    assert "side effects" in r.error


def test_inline_variable_allow_side_effects():
    code = """
def get_value():
    return 42

x = get_value()
y = x + 1
"""
    r = inline_variable(code, "x", allow_side_effects=True)
    assert r.applied
    # After inlining, the call moves to the use site
    assert "get_value()" in r.new_code
    # The assignment line is gone
    assert "x = get_value()" not in r.new_code


def test_inline_variable_refused_on_fn_arg():
    code = """
def f(x):
    y = x + 1
    return y
"""
    r = inline_variable(code, "x", scope="f")
    assert not r.applied
    assert "function argument" in r.error


def test_inline_variable_complex_expression():
    code = """
base = 10
result = base ** 2 + base * 3 + 1
"""
    r = inline_variable(code, "base")
    assert r.applied
    ns = {}
    exec(r.new_code, ns)
    assert ns["result"] == 131


# ==================================================================
# extract_method
# ==================================================================


def test_extract_method_simple():
    code = """
class Foo:
    def process(self, data):
        # header
        x = data * 2
        y = x + 10
        return y
""".strip()
    # Extract lines 4-5 (x = ..., y = ...)
    r = extract_method(code, "Foo", "process", "_compute", 4, 5)
    assert r.applied, r.error
    # New method exists
    assert "def _compute" in r.new_code
    # Call site replaces extracted lines
    assert "self._compute" in r.new_code
    # Semantics preserved
    ns = {}
    exec(r.new_code, ns)
    assert ns["Foo"]().process(5) == 20


def test_extract_method_with_args():
    code = """
class Calc:
    def solve(self, a, b):
        c = a + b
        d = c * 2
        return d
""".strip()
    r = extract_method(code, "Calc", "solve", "_double", 4, 4)
    assert r.applied, r.error
    ns = {}
    exec(r.new_code, ns)
    assert ns["Calc"]().solve(3, 4) == 14


def test_extract_method_no_class():
    r = extract_method("x = 1", "Foo", "bar", "baz", 1, 1)
    assert not r.applied
    assert "class" in r.error


def test_extract_method_collision_refused():
    code = """
class Foo:
    def a(self):
        x = 1
        return x
    def b(self):
        pass
""".strip()
    r = extract_method(code, "Foo", "a", "b", 3, 3)
    assert not r.applied
    assert "already exists" in r.error


# ==================================================================
# RefactorSession (multi-step chains)
# ==================================================================


def test_session_chain_success():
    code = """
def f():
    x = 1
    y = x + 2
    z = y * 3
    return z
"""
    s = RefactorSession(code)
    r1 = s.apply(rename_variable, old="x", new="first", scope="f")
    assert r1.applied
    r2 = s.apply(rename_variable, old="y", new="second", scope="f")
    assert r2.applied
    r3 = s.apply(inline_variable, var_name="first", scope="f")
    assert r3.applied
    assert s.ok
    final = s.result()
    ns = {}
    exec(final, ns)
    assert ns["f"]() == 9


def test_session_stops_on_failure():
    code = "x = 1\ny = 2\n"
    s = RefactorSession(code)
    # Collision - fails
    r = s.apply(rename_variable, old="x", new="y")
    assert not r.applied
    # Subsequent apply should no-op
    r2 = s.apply(rename_variable, old="y", new="zz")
    assert not r2.applied
    assert not s.ok
    with pytest.raises(RuntimeError):
        s.result()


def test_session_preserves_initial_on_failure():
    code = "x = 1\ny = 2\n"
    s = RefactorSession(code)
    r1 = s.apply(rename_variable, old="x", new="alpha")
    assert r1.applied
    # Second step fails — session reports failure but history shows both
    r2 = s.apply(rename_variable, old="y", new="alpha")  # collision
    assert not r2.applied
    assert len(s.history) == 2
    assert s.history[0][1].applied
    assert not s.history[1][1].applied


# ==================================================================
# Integration — combined refactor scenario
# ==================================================================


# ==================================================================
# convert_loop_to_comprehension
# ==================================================================


def test_loop_to_comprehension_simple():
    code = """
def squares(xs):
    result = []
    for x in xs:
        result.append(x * x)
    return result
"""
    r = convert_loop_to_comprehension(code)
    assert r.applied
    assert "[x * x for x in xs]" in r.new_code
    ns = {}
    exec(r.new_code, ns)
    assert ns["squares"]([1, 2, 3]) == [1, 4, 9]


def test_loop_to_comprehension_with_guard():
    code = """
def evens(xs):
    result = []
    for x in xs:
        if x % 2 == 0:
            result.append(x)
    return result
"""
    r = convert_loop_to_comprehension(code)
    assert r.applied
    assert "[x for x in xs if x % 2 == 0]" in r.new_code
    ns = {}
    exec(r.new_code, ns)
    assert ns["evens"]([1, 2, 3, 4, 5]) == [2, 4]


def test_loop_to_comprehension_refuses_multi_statement_body():
    """Body with more than the append → not rewritten."""
    code = """
def f(xs):
    result = []
    for x in xs:
        y = x * 2
        result.append(y + 1)
    return result
"""
    r = convert_loop_to_comprehension(code)
    assert not r.applied


def test_loop_to_comprehension_matches_paired_accumulator():
    """When `other = []` immediately precedes `for x in xs: other.append(x)`,
    the rewriter correctly pairs them and leaves the unrelated `result = []`
    untouched.
    """
    code = """
def f(xs):
    result = []
    other = []
    for x in xs:
        other.append(x)
    return other
"""
    r = convert_loop_to_comprehension(code)
    assert r.applied
    # `other` got the comprehension; `result` stays
    assert "other = [x for x in xs]" in r.new_code
    assert "result = []" in r.new_code


def test_loop_to_comprehension_inside_method():
    code = """
class Analyzer:
    def flag(self, items):
        result = []
        for item in items:
            if item > 10:
                result.append(item)
        return result
"""
    r = convert_loop_to_comprehension(code)
    assert r.applied
    ns = {}
    exec(r.new_code, ns)
    assert ns["Analyzer"]().flag([5, 15, 20]) == [15, 20]


# ==================================================================
# detect_refactor_opportunities
# ==================================================================


def test_detect_long_method():
    long_body = "\n".join([f"        x{i} = {i}" for i in range(35)])
    code = f"""
class Big:
    def monster_method(self):
{long_body}
"""
    opps = detect_refactor_opportunities(code, long_method_threshold=30)
    long_opps = [o for o in opps if o.kind == "long_method"]
    assert len(long_opps) == 1
    assert "Big.monster_method" in long_opps[0].location


def test_detect_loop_comprehension_pattern():
    code = """
def f(xs):
    result = []
    for x in xs:
        result.append(x + 1)
    return result
"""
    opps = detect_refactor_opportunities(code)
    kinds = {o.kind for o in opps}
    assert "loop_to_comprehension" in kinds


def test_detect_single_use_local():
    code = """
def f():
    x = compute()
    return x
"""
    opps = detect_refactor_opportunities(code)
    inline_opps = [o for o in opps if o.kind == "single_use_local"]
    assert any(o.detail.startswith("var 'x'") for o in inline_opps)


def test_detect_clean_code_no_opportunities():
    code = """
def f(xs):
    return [x * 2 for x in xs if x > 0]
"""
    opps = detect_refactor_opportunities(code, long_method_threshold=100)
    # No long methods, no loop-to-comp (already a comp), no single-use
    # locals (no assigns).
    assert len(opps) == 0


def test_combined_session_full_refactor():
    """Realistic scenario: rename a var, inline a temp, extract a method."""
    code = """
class Analytics:
    def summarize(self, data):
        tmp = [x * 2 for x in data]
        total = sum(tmp)
        count = len(data)
        avg = total / count if count else 0
        return avg
""".strip()
    s = RefactorSession(code)
    # 1. Rename 'tmp' to 'doubled'
    s.apply(rename_variable, old="tmp", new="doubled", scope="summarize")
    # 2. Inline the doubled variable (side-effect-free list comp)
    s.apply(inline_variable, var_name="doubled", scope="summarize")
    assert s.ok, str(s.history)

    final = s.result()
    ns = {}
    exec(final, ns)
    assert ns["Analytics"]().summarize([1, 2, 3]) == 4.0
