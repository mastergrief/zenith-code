"""Tests for the auto-refactor planner."""
from calm.llm_computer.refactor_planner import (
    PlanOutcome, build_plan, execute_plan,
)


def test_build_plan_detects_loop_comprehension():
    code = """
def f(xs):
    result = []
    for x in xs:
        result.append(x * 2)
    return result
"""
    plan = build_plan(code)
    names = [p.primitive_name for p in plan]
    assert "convert_loop_to_comprehension" in names


def test_build_plan_single_use_local_queued():
    code = """
def f():
    tmp = compute_something_pure(42)
    return tmp
"""
    # `tmp` is single-use single-assignment but value has side effects —
    # the base inline_variable will refuse. Plan still queues it; the
    # session reports the error gracefully.
    plan = build_plan(code)
    inline_steps = [p for p in plan if p.primitive_name == "inline_variable"]
    assert len(inline_steps) >= 1


def test_execute_plan_runs_session_with_tests():
    code = """
def squares(xs):
    result = []
    for x in xs:
        result.append(x * x)
    return result
"""
    tests = """
assert squares([1, 2, 3]) == [1, 4, 9]
assert squares([]) == []
print("ALL_PASS")
"""
    out = execute_plan(code, tests)
    assert out.error is None
    assert out.tests_pass
    assert "[x * x for x in xs]" in out.final_code


def test_execute_plan_rolls_back_broken_step():
    """An inline that breaks tests should roll back; the final code is
    the last-good version, and tests pass on it."""
    code = """
def f(x):
    doubled = x * 2
    print("hello")
    return doubled
"""
    tests = """
assert f(3) == 6
print("ALL_PASS")
"""
    out = execute_plan(code, tests)
    # Regardless of what ran, the final code must pass tests
    assert out.tests_pass


def test_execute_plan_baseline_failure_returns_error():
    code = "def f():\n    return 0\n"
    tests = 'assert f() == 99\nprint("ALL_PASS")\n'
    out = execute_plan(code, tests)
    assert out.error is not None
    assert "baseline" in out.error


def test_planner_idempotent_on_clean_code():
    """Running the planner on its own output should produce no changes."""
    code = """
def clean(xs):
    return [x * 2 for x in xs if x > 0]
"""
    tests = 'assert clean([1, -1, 2]) == [2, 4]\nprint("ALL_PASS")\n'
    out1 = execute_plan(code, tests)
    assert out1.tests_pass
    out2 = execute_plan(out1.final_code, tests)
    # Second run should have no new applications
    assert out2.n_applied == 0
    assert out2.final_code == out1.final_code


def test_plan_outcome_summary_readable():
    code = """
def f(xs):
    result = []
    for x in xs:
        result.append(x)
    return result
"""
    tests = 'assert f([1,2,3]) == [1,2,3]\nprint("ALL_PASS")\n'
    out = execute_plan(code, tests)
    assert "/" in out.summary
    assert "opportunities" in out.summary
