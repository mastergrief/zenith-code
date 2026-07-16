"""CPU-static tests for Step-C characterization reducers."""
from __future__ import annotations

from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.forgotten_accum_step_c_characterization_reducers import (
    MAX_PHYSICAL,
    ReducerViolation,
    assert_geometry_tuple_admitted,
    assert_keysets,
    assert_phase_budget,
    assert_planned_physical_cap,
    assert_runner_attempt_cap,
    assert_scratch_paths_equal,
    assert_scratch_pre_empty,
    exact_int_steps,
)


def test_exact_int_steps_and_malformed() -> None:
    assert exact_int_steps(2) == 2
    with pytest.raises(ReducerViolation, match="STEPS_MALFORMED"):
        exact_int_steps(True)
    with pytest.raises(ReducerViolation, match="STEPS_MALFORMED"):
        exact_int_steps(0)


def test_planned_physical_cap_before_work() -> None:
    assert_planned_physical_cap(planned=8, steps=2)
    with pytest.raises(ReducerViolation, match="PLANNED_PHYSICAL_OVERFLOW"):
        assert_planned_physical_cap(planned=8, steps=3)
    assert MAX_PHYSICAL == 10


def test_runner_attempt_cap() -> None:
    assert_runner_attempt_cap(next_attempt=4)
    with pytest.raises(ReducerViolation, match="RUNNER_ATTEMPT_OVERFLOW"):
        assert_runner_attempt_cap(next_attempt=5)


def test_keysets() -> None:
    good = {"U": {1, 2, 3, 4}, "E": {3, 4}, "R0": {3, 4}, "RW": {3, 4}}
    assert_keysets(good)
    with pytest.raises(ReducerViolation, match="STEP_KEYSET_MISMATCH"):
        assert_keysets({"U": {1}, "E": {3, 4}, "R0": {3, 4}, "RW": {3, 4}})


def test_scratch_allowlist() -> None:
    allow = [Path("/tmp/a.json"), Path("/tmp/b.json")]
    assert_scratch_paths_equal(got=list(allow), allow=list(allow))
    with pytest.raises(ReducerViolation, match="UNEXPECTED_SCRATCH_ARTIFACT"):
        assert_scratch_paths_equal(got=[Path("/tmp/a.json"), Path("/tmp/extra.json")], allow=allow)
    with pytest.raises(ReducerViolation, match="BANKED_ARTIFACT_MUTATION"):
        assert_scratch_paths_equal(
            got=[Path("/tmp/a.json"), Path("/tmp/b.json"), Path("/tmp/c.pt")],
            allow=allow,
        )
    assert_scratch_pre_empty([])
    with pytest.raises(ReducerViolation, match="ARTIFACT_COLLISION"):
        assert_scratch_pre_empty([Path("/tmp/x")])


def test_phase_budgets() -> None:
    assert_phase_budget(name="arm_U", duration_s=1.0)
    with pytest.raises(ReducerViolation, match="PHASE_BUDGET_EXCEEDED"):
        assert_phase_budget(name="arm_U", duration_s=481.0)
    with pytest.raises(ReducerViolation, match="CUT_FORK_PHASE_TIMEOUT"):
        assert_phase_budget(name="cut_fork_serialize_load", duration_s=121.0)


def test_geometry_admitted_predicate() -> None:
    assert_geometry_tuple_admitted(True, geom=(2, 4, 1))
    with pytest.raises(ReducerViolation, match="UNADMITTED_GEOMETRY"):
        assert_geometry_tuple_admitted(False, geom=(2, 4, 2))
