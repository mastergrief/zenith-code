"""Pure Step-C characterization reducers (CPU-static; no I/O/GPU/launch)."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

REQUIRED_KEYS = {"U": {1, 2, 3, 4}, "E": {3, 4}, "R0": {3, 4}, "RW": {3, 4}}
MAX_PHYSICAL = 10
MAX_RUNNER_ATTEMPTS = 4
PHASE_BUDGETS = {
    "materialize": 360.0,
    "arm_U": 480.0,
    "arm_E": 300.0,
    "arm_R0": 300.0,
    "arm_RW": 300.0,
    "cut_fork_serialize_load": 120.0,
    "receipt_emission": 60.0,
    "postflight": 90.0,
}


class ReducerViolation(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code, self.detail = str(code), str(detail)
        super().__init__(f"{self.code}" + (f": {detail}" if detail else ""))


def exact_int_steps(value: Any) -> int:
    if type(value) is not int or isinstance(value, bool) or value <= 0:
        raise ReducerViolation("STEPS_MALFORMED", repr(value))
    return value


def assert_planned_physical_cap(*, planned: int, steps: int) -> None:
    if int(planned) + int(steps) > MAX_PHYSICAL:
        raise ReducerViolation(
            "PLANNED_PHYSICAL_OVERFLOW", f"{planned}+{steps}>{MAX_PHYSICAL}"
        )


def assert_runner_attempt_cap(*, next_attempt: int) -> None:
    if int(next_attempt) > MAX_RUNNER_ATTEMPTS:
        raise ReducerViolation("RUNNER_ATTEMPT_OVERFLOW", str(next_attempt))


def assert_keysets(observed: Mapping[str, set[int]]) -> None:
    for arm, need in REQUIRED_KEYS.items():
        got = set(observed.get(arm, ()))
        if got != need:
            raise ReducerViolation("STEP_KEYSET_MISMATCH", f"{arm}:{sorted(got)}!={sorted(need)}")


def assert_scratch_paths_equal(*, got: list[Path], allow: list[Path]) -> None:
    need = sorted(p.resolve() for p in allow)
    have = sorted(p.resolve() for p in got)
    if have != need:
        code = (
            "BANKED_ARTIFACT_MUTATION"
            if any(p.suffix == ".pt" for p in have)
            else "UNEXPECTED_SCRATCH_ARTIFACT"
        )
        raise ReducerViolation(code, f"{have}!={need}")


def assert_scratch_pre_empty(got: list[Path]) -> None:
    if got:
        raise ReducerViolation("ARTIFACT_COLLISION", f"pre-existing:{got[0]}")


def phase_budget_code(name: str) -> str:
    return "CUT_FORK_PHASE_TIMEOUT" if name == "cut_fork_serialize_load" else "PHASE_BUDGET_EXCEEDED"


def assert_phase_budget(*, name: str, duration_s: float) -> None:
    budget = float(PHASE_BUDGETS[name])
    if float(duration_s) > budget:
        raise ReducerViolation(phase_budget_code(name), f"{name}:{duration_s:.3f}>{budget}")


def assert_geometry_tuple_admitted(admitted: bool, *, geom: tuple[int, int, int]) -> None:
    if not admitted:
        raise ReducerViolation("UNADMITTED_GEOMETRY", repr(geom))


__all__ = [
    "REQUIRED_KEYS",
    "MAX_PHYSICAL",
    "MAX_RUNNER_ATTEMPTS",
    "PHASE_BUDGETS",
    "ReducerViolation",
    "exact_int_steps",
    "assert_planned_physical_cap",
    "assert_runner_attempt_cap",
    "assert_keysets",
    "assert_scratch_paths_equal",
    "assert_scratch_pre_empty",
    "assert_phase_budget",
    "phase_budget_code",
    "assert_geometry_tuple_admitted",
]
