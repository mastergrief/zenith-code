"""Test VerifiedRefactorSession (refactor + sandbox tests)."""
from calm.llm_computer.ast_refactor import (
    extract_method, inline_variable, rename_variable,
)
from calm.llm_computer.refactor_session import VerifiedRefactorSession


GOOD_TESTS = """
assert f() == 15
print("ALL_PASS")
"""


SOURCE_SIMPLE = """
def f():
    x = 5
    y = x + 10
    return y
"""


def test_session_baseline_must_pass():
    source = """
def f():
    return 999
"""
    s = VerifiedRefactorSession(source, GOOD_TESTS)
    assert not s.ok
    assert "baseline" in s.last_error


def test_session_chain_ok():
    s = VerifiedRefactorSession(SOURCE_SIMPLE, GOOD_TESTS)
    assert s.ok, s.last_error
    step = s.apply(rename_variable, old="x", new="base", scope="f")
    assert step.test_passed
    step = s.apply(rename_variable, old="y", new="result", scope="f")
    assert step.test_passed
    step = s.apply(inline_variable, var_name="base", scope="f")
    assert step.test_passed
    assert s.ok
    # Result code compiles + passes tests
    final = s.result()
    exec(final + "\n\nassert f() == 15")


def test_session_rolls_back_on_test_failure():
    """Rename a function that tests depend on → tests break → rollback."""
    tests = """
assert f() == 15
print("ALL_PASS")
"""
    s = VerifiedRefactorSession(SOURCE_SIMPLE, tests)
    step = s.apply(rename_variable, old="f", new="g")  # breaks tests
    # Refactor applied but tests fail (tests call f(), not g())
    assert step.refactor_result.applied
    assert step.test_passed is False
    assert not s.ok
    # Current code stays at last-good (initial)
    assert "def f" in s.result()


def test_session_stops_on_refactor_error():
    tests = 'assert True\nprint("ALL_PASS")\n'
    s = VerifiedRefactorSession("x = 1\ny = 2\n", tests)
    step = s.apply(rename_variable, old="x", new="y")  # collision
    assert not step.refactor_result.applied
    assert step.test_passed is None  # tests never run
    assert not s.ok


def test_session_history_tracks_every_step():
    s = VerifiedRefactorSession(SOURCE_SIMPLE, GOOD_TESTS)
    s.apply(rename_variable, old="x", new="base", scope="f")
    s.apply(rename_variable, old="y", new="result", scope="f")
    assert len(s.history) == 2
    for step in s.history:
        assert step.refactor_result.applied
        assert step.test_passed


def test_session_summary_reflects_outcome():
    s = VerifiedRefactorSession(SOURCE_SIMPLE, GOOD_TESTS)
    s.apply(rename_variable, old="x", new="base", scope="f")
    assert s.summary.startswith("OK")


def test_session_summary_on_failure():
    s = VerifiedRefactorSession("x = 1\ny = 2\n",
                                 'assert True\nprint("ALL_PASS")\n')
    s.apply(rename_variable, old="x", new="y")  # collision
    assert "FAILED" in s.summary
