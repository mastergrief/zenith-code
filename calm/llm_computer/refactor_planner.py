"""Auto-refactor planner — plan-and-execute refactor sessions from
detected opportunities.

Given a Python source + test harness, this module:

  1. Scans for refactor opportunities via `detect_refactor_opportunities`.
  2. Builds an ordered `SessionPlan` — a list of `(primitive_callable,
     kwargs)` tuples, ordered by safety (cheapest-to-verify first).
  3. Runs the plan through `VerifiedRefactorSession` so each step is
     sandbox-gated. Failures don't poison subsequent steps.
  4. Returns a `PlanOutcome` report with applied / skipped / failed
     counts + final code.

Closes the last frontier-coding loop: the substrate REFACTORS CODE
WITHOUT HUMAN INTERVENTION beyond the test harness. Analogous to the
autonomous-loop pattern on compute facades (oracle_inference +
llm_synthesizer + MetaFacade), but operating on real source.

Design:
  - Ordering: loop_to_comprehension BEFORE single_use_local (simpler,
    higher value, less state to track). Long methods reported but not
    auto-extracted yet (extract_method needs line-range inference from
    the long-method detector, which we haven't wired).
  - Safety: every step is sandbox-verified before accepting. Test
    failures roll back the step; session continues with remaining plan.
  - Idempotent: running the planner again on its own output should be
    a no-op (all opportunities resolved).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from calm.llm_computer.ast_refactor import (
    RefactorOpportunity,
    convert_loop_to_comprehension,
    detect_refactor_opportunities,
    inline_variable,
)
from calm.llm_computer.refactor_session import (
    VerifiedRefactorSession, VerifiedStep,
)


@dataclass
class PlanStep:
    """One scheduled refactor operation."""
    primitive_name: str
    primitive: Callable
    kwargs: dict
    opportunity: RefactorOpportunity


@dataclass
class PlanOutcome:
    """Full report of a planner run."""
    opportunities_found: List[RefactorOpportunity] = field(default_factory=list)
    plan: List[PlanStep] = field(default_factory=list)
    applied_steps: List[VerifiedStep] = field(default_factory=list)
    initial_code: str = ""
    final_code: str = ""
    tests_pass: bool = False
    error: Optional[str] = None

    @property
    def n_applied(self) -> int:
        return sum(1 for s in self.applied_steps
                   if s.refactor_result.applied and s.test_passed)

    @property
    def n_plan(self) -> int:
        return len(self.plan)

    @property
    def summary(self) -> str:
        if self.error:
            return f"ERROR: {self.error}"
        return (f"{self.n_applied}/{self.n_plan} steps applied & verified, "
                f"{len(self.opportunities_found)} opportunities detected")


def build_plan(
    code: str,
    opportunities: Optional[List[RefactorOpportunity]] = None,
) -> List[PlanStep]:
    """Translate a list of opportunities into an executable SessionPlan.

    Priority (first applied first):
      1. loop_to_comprehension (single pass, no scope tracking, big
         readability win)
      2. single_use_local inline (scope-aware, rebuilds each invocation)

    long_method opportunities are RECORDED but not executed — they
    require line-range inference + extract_method, which currently
    needs human input for new method name. Future work.
    """
    if opportunities is None:
        opportunities = detect_refactor_opportunities(code)

    plan: List[PlanStep] = []

    # Group 1: loop_to_comprehension — single module-wide primitive.
    # It's self-contained (scans every body), so we enqueue it once
    # even if there are multiple loop_to_comp opportunities.
    lc_opps = [o for o in opportunities
               if o.kind == "loop_to_comprehension"]
    if lc_opps:
        plan.append(PlanStep(
            primitive_name="convert_loop_to_comprehension",
            primitive=convert_loop_to_comprehension,
            kwargs={},
            opportunity=lc_opps[0],     # representative
        ))

    # Group 2: inline single-use locals. Each is scoped to its
    # enclosing function; the opportunity's location encodes
    # "function:var".
    for opp in opportunities:
        if opp.kind != "single_use_local":
            continue
        loc = opp.location
        if ":" not in loc:
            continue
        fn_scope, var_name = loc.split(":", 1)
        plan.append(PlanStep(
            primitive_name="inline_variable",
            primitive=inline_variable,
            kwargs={"var_name": var_name, "scope": fn_scope},
            opportunity=opp,
        ))

    return plan


def execute_plan(
    code: str,
    tests_code: str,
    plan: Optional[List[PlanStep]] = None,
    *,
    timeout: float = 10.0,
) -> PlanOutcome:
    """Run the plan through `VerifiedRefactorSession`. If `plan` is
    None, auto-build from `detect_refactor_opportunities(code)`.

    Each step's result is recorded. Failures (sandbox test regression
    or refactor-primitive error) are NOT session-fatal — we continue
    to the next step. This is different from the default Session
    behavior: the planner assumes independent steps, so one broken
    step shouldn't halt the rest.
    """
    opps = detect_refactor_opportunities(code)
    if plan is None:
        plan = build_plan(code, opps)

    out = PlanOutcome(
        opportunities_found=opps,
        plan=plan,
        initial_code=code,
        final_code=code,
    )

    session = VerifiedRefactorSession(code, tests_code, timeout=timeout)
    if not session.ok:
        out.error = f"baseline tests don't pass: {session.last_error}"
        return out

    for step in plan:
        # Apply the step. If it fails (error or test regression), record
        # but don't halt.
        v_step = session.apply(step.primitive, **step.kwargs)
        out.applied_steps.append(v_step)
        # If session state went failed due to this step, clear the flag
        # and continue. The session's internal `_current` already rolled
        # back to last-good.
        if not session.ok:
            # Reset the session error/failed flag so next step can run
            session._failed = False
            session._error = None

    out.final_code = session.result()
    # Final sandbox sanity — do tests still pass on final code?
    ok, _, _ = session._run_tests(out.final_code)
    out.tests_pass = ok
    return out
