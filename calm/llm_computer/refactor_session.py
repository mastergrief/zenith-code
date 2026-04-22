"""Test-driven recursive refactor — chain ast_refactor primitives with
sandbox test verification between steps.

Extends RefactorSession with an oracle: after each applied primitive,
run the user-supplied test harness. If tests fail, roll back. This is
the code-oriented analog of CALM's safe_eval oracle — substrate
refactoring with execution-level verification.

Pipeline per step:

    primitive(code, **kwargs) → RefactorResult
        ↓ (applied? — else record failure, stop session)
    sandbox.run_python(code + tests) → SandboxResult
        ↓ (ok? — else roll back this step's rewrite, stop session)
    next step

The session's final code is the last version where ALL steps passed
both AST validation AND sandbox tests. History records every attempt.

This is the frontier-coding equivalent of the autonomous loop:
    - auto-facade generator: Gemma-fail prompt → spec → validate → ship
    - refactor session: user refactor plan → primitive → sandbox → ship
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from calm.llm_computer.ast_refactor import RefactorResult


@dataclass
class VerifiedStep:
    """One step of a VerifiedRefactorSession."""
    primitive_name: str
    kwargs: dict
    refactor_result: RefactorResult
    test_passed: Optional[bool]         # None if refactor didn't apply
    test_output: Optional[str] = None
    test_error: Optional[str] = None


class VerifiedRefactorSession:
    """RefactorSession + sandbox test-harness verification between steps.

    Usage:
        tests_code = '''
        assert process(5) == 10
        assert process(0) == 0
        print("ALL_PASS")
        '''
        session = VerifiedRefactorSession(source_code, tests_code)
        session.apply(rename_variable, old="x", new="value")
        session.apply(inline_variable, var_name="tmp")
        if session.ok:
            final = session.result()
        else:
            # session.last_error reports which step + why
            pass

    Test harness contract: tests_code runs AS-IS (no substitution) and
    is CONCATENATED to the current source code with a blank line
    between. Tests must reference names from the source verbatim.
    """

    # Line printed by well-formed test harness on success. Callers can
    # override via `sentinel` ctor param.
    DEFAULT_SENTINEL = "ALL_PASS"

    def __init__(
        self,
        initial_code: str,
        tests_code: str,
        *,
        sentinel: str = DEFAULT_SENTINEL,
        timeout: float = 10.0,
    ):
        self._initial = initial_code
        self._current = initial_code
        self._tests = tests_code
        self._sentinel = sentinel
        self._timeout = timeout
        self._history: List[VerifiedStep] = []
        self._last_good_code = initial_code
        self._failed = False
        self._error: Optional[str] = None
        self._verify_initial()

    def _verify_initial(self):
        """Run tests on initial code. If the baseline doesn't pass, the
        session is invalid — can't detect regressions."""
        ok, output, err = self._run_tests(self._initial)
        if not ok:
            self._failed = True
            self._error = (f"baseline tests don't pass — cannot detect "
                           f"regressions (err: {err or 'missing sentinel'})")

    def _run_tests(self, code: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Run `code + tests` in sandbox. Return (ok, stdout, error)."""
        from calm.sandbox import run_python
        combined = code + "\n\n" + self._tests
        result = run_python(combined, timeout=self._timeout)
        if result.error:
            return False, result.raw, result.error
        if self._sentinel not in (result.raw or ""):
            return False, result.raw, f"sentinel {self._sentinel!r} missing"
        return True, result.raw, None

    def apply(self, primitive: Callable, /, **kwargs) -> VerifiedStep:
        """Apply primitive + verify. If either step fails, record and
        stop the session."""
        if self._failed:
            step = VerifiedStep(
                primitive_name=primitive.__name__, kwargs=kwargs,
                refactor_result=RefactorResult(
                    None, "none",
                    error="session already failed"),
                test_passed=None,
            )
            self._history.append(step)
            return step

        result = primitive(self._current, **kwargs)
        if not result.applied:
            step = VerifiedStep(
                primitive_name=primitive.__name__, kwargs=kwargs,
                refactor_result=result, test_passed=None,
            )
            self._history.append(step)
            # No-op (new_code=None, error=None) is session-safe: nothing
            # to apply but no failure either. Only fail on explicit error.
            if result.error is not None:
                self._failed = True
                self._error = f"{primitive.__name__}: {result.error}"
            return step

        # Refactor applied; verify via sandbox
        ok, output, err = self._run_tests(result.new_code)
        step = VerifiedStep(
            primitive_name=primitive.__name__, kwargs=kwargs,
            refactor_result=result,
            test_passed=ok, test_output=output, test_error=err,
        )
        self._history.append(step)
        if ok:
            self._current = result.new_code
            self._last_good_code = result.new_code
        else:
            # Roll back this step — keep _current at last good
            self._failed = True
            self._error = (f"{primitive.__name__} broke tests: "
                           f"{err or 'sentinel missing'}")
        return step

    def result(self) -> str:
        """Current code if session ok, else the last good version
        (rolled back to before the failing step)."""
        return self._current

    @property
    def ok(self) -> bool:
        return not self._failed

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    @property
    def history(self) -> List[VerifiedStep]:
        return list(self._history)

    @property
    def summary(self) -> str:
        """One-line summary of the session outcome."""
        applied = sum(1 for s in self._history
                      if s.refactor_result.applied and s.test_passed)
        total = len(self._history)
        if self._failed:
            return f"FAILED at step {applied + 1}/{total}: {self._error}"
        return f"OK: {applied}/{total} steps applied & verified"
