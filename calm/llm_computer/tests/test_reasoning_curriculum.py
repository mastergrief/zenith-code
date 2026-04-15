"""Tests for reasoning curriculum."""

from __future__ import annotations

import pytest

from calm.llm_computer.calm_training_bridge import (
    default_oracle, recipe_adder_small, recipe_is_sum_prime,
)
from calm.llm_computer.compositional_data import CompositionalTask
from calm.llm_computer.reasoning_curriculum import (
    CorrectionSet, Decomposer, ErrorRecord, ErrorTracker,
    ReasoningCurriculum, grade_difficulty,
)


# ----- Decomposer tests -----

def test_decomposer_splits_is_sum_prime():
    d = Decomposer()
    trace = d.decompose("calm_is_sum_prime")
    assert trace.goal_recipe == "calm_is_sum_prime"
    # Expect 3 steps: adder, is_prime, joint
    assert len(trace.steps) == 3
    recipe_names = [s.recipe_name for s in trace.steps]
    assert "calm_adder_small" in recipe_names
    assert "calm_is_prime" in recipe_names
    assert "calm_is_sum_prime" in recipe_names


def test_decomposer_unknown_returns_atomic():
    d = Decomposer()
    trace = d.decompose("never_seen_recipe")
    assert len(trace.steps) == 1
    assert trace.steps[0].recipe_name == "never_seen_recipe"
    assert not trace.steps[0].depends_on


def test_leaf_recipes_identified():
    d = Decomposer()
    trace = d.decompose("calm_is_sum_prime")
    leaves = trace.leaf_recipes
    assert "calm_adder_small" in leaves  # depends on nothing
    assert "calm_is_prime" not in leaves  # depends on adder


# ----- Difficulty grading -----

def test_difficulty_grading_composition_bumps():
    adder = recipe_adder_small()
    sum_prime = recipe_is_sum_prime()
    g_adder = grade_difficulty(adder)
    g_sp = grade_difficulty(sum_prime)
    assert g_sp > g_adder  # composition should grade harder


# ----- Error tracker -----

def test_error_tracker_records_misses():
    tracker = ErrorTracker()
    task = CompositionalTask(prompt=(1, 2), answer=3,
                             cards_required=frozenset({"adder"}))
    tracker.record_outcome("calm_adder_small", task, predicted=3, correct=3)
    tracker.record_outcome("calm_adder_small", task, predicted=0, correct=3)
    tracker.record_outcome("calm_adder_small", task, predicted=0, correct=3)
    assert tracker.error_rate("calm_adder_small") == 2 / 3
    errors = tracker.errors("calm_adder_small")
    assert len(errors) == 2


def test_error_tracker_hardest_recipe():
    tracker = ErrorTracker()
    t = CompositionalTask(prompt=(0,), answer=0, cards_required=frozenset())
    # easy: 1 error / 10 attempts = 10%
    for _ in range(10):
        tracker.record_outcome("easy", t, predicted=0, correct=0)
    tracker.record_outcome("easy", t, predicted=1, correct=0)
    # hard: 5 errors / 10 attempts = 50%
    for _ in range(5):
        tracker.record_outcome("hard", t, predicted=0, correct=0)
    for _ in range(5):
        tracker.record_outcome("hard", t, predicted=1, correct=0)
    assert tracker.hardest_recipe() == "hard"


def test_error_tracker_no_errors_recipe_never_fires():
    tracker = ErrorTracker()
    assert tracker.hardest_recipe() is None


# ----- Correction set -----

def test_correction_set_add_and_sample():
    cs = CorrectionSet()
    task1 = CompositionalTask(prompt=(1, 1), answer=2,
                              cards_required=frozenset({"adder"}))
    task2 = CompositionalTask(prompt=(2, 3), answer=5,
                              cards_required=frozenset({"adder"}))
    cs.add(task1)
    cs.add(task2)
    assert len(cs) == 2
    sampled = cs.as_training_data(n=10, seed=0)
    assert len(sampled) == 10
    # All samples should be from the 2 added tasks
    for s in sampled:
        assert s.prompt in {(1, 1), (2, 3)}


def test_correction_set_empty_returns_empty():
    cs = CorrectionSet()
    assert cs.as_training_data(n=10) == []


def test_correction_set_from_error_records():
    cs = CorrectionSet()
    task = CompositionalTask(prompt=(3, 4), answer=7,
                             cards_required=frozenset({"adder"}))
    errs = [ErrorRecord("calm_adder_small", task, predicted=0, correct=7)]
    cs.add_from_errors(errs)
    assert len(cs) == 1


# ----- Curriculum orchestrator -----

def test_curriculum_add_goal_enqueues_steps():
    oracle = default_oracle()
    curr = ReasoningCurriculum(oracle)
    trace = curr.add_goal("calm_is_sum_prime")
    # Should queue all 3 steps of the decomposition
    assert len(trace.steps) == 3
    pending = curr.schedule()
    assert "calm_adder_small" in pending
    assert "calm_is_prime" in pending
    assert "calm_is_sum_prime" in pending


def test_curriculum_schedule_orders_by_difficulty():
    oracle = default_oracle()
    curr = ReasoningCurriculum(oracle)
    curr.add_goal("calm_is_sum_prime")
    order = curr.schedule()
    # Atomic recipes should come first, composition last
    # calm_adder_small is difficulty 1 (single card)
    # calm_is_prime is difficulty 1
    # calm_is_sum_prime is difficulty 2 + composition = higher
    assert order[-1] == "calm_is_sum_prime"


def test_curriculum_next_recipe_pops_queue():
    oracle = default_oracle()
    curr = ReasoningCurriculum(oracle)
    curr.add_goal("calm_is_sum_prime")
    first = curr.next_recipe()
    assert first is not None
    assert first not in curr.schedule()  # removed from pending


def test_curriculum_next_recipe_returns_none_when_done():
    oracle = default_oracle()
    curr = ReasoningCurriculum(oracle)
    assert curr.next_recipe() is None


def test_curriculum_record_outcome_marks_complete():
    oracle = default_oracle()
    curr = ReasoningCurriculum(oracle)
    curr.add_goal("calm_is_sum_prime")
    first = curr.next_recipe()
    curr.record_outcome(first, accuracy=0.95)
    assert first in curr.completed_recipes()
    assert curr.outcomes()[first] == 0.95


def test_curriculum_training_data_includes_fresh():
    oracle = default_oracle()
    curr = ReasoningCurriculum(oracle)
    data = curr.training_data_for("calm_adder_small", n=20)
    assert len(data) == 20
    # All should be valid adder tasks (answer = sum)
    for t in data:
        a, b = t.prompt
        assert t.answer == a + b


def test_curriculum_training_data_mixes_corrections():
    oracle = default_oracle()
    curr = ReasoningCurriculum(oracle)
    # Seed the correction set with a known error
    bad_task = CompositionalTask(prompt=(0, 0), answer=7,  # wrong but labeled
                                  cards_required=frozenset({"adder"}))
    curr.corrections.add(bad_task)
    # Request training data with 30% corrections
    data = curr.training_data_for(
        "calm_adder_small", n=20,
        include_corrections=True, correction_fraction=0.3,
    )
    # Count how many came from corrections (they have the marked prompt)
    n_corrections = sum(1 for t in data if t.answer == 7)
    # Should be ~6 (30% of 20)
    assert n_corrections == 6


def test_curriculum_records_errors_into_corrections():
    oracle = default_oracle()
    curr = ReasoningCurriculum(oracle)
    curr.add_goal("calm_adder_small")
    task = CompositionalTask(prompt=(1, 2), answer=3,
                             cards_required=frozenset({"adder"}))
    err = ErrorRecord("calm_adder_small", task, predicted=0, correct=3)
    curr.record_outcome("calm_adder_small", accuracy=0.5, errors=[err])
    # Correction set should have the failed task
    assert len(curr.corrections) == 1


if __name__ == "__main__":
    test_decomposer_splits_is_sum_prime()
    print("[ok] decomposer splits is_sum_prime")
    test_decomposer_unknown_returns_atomic()
    print("[ok] unknown goal → atomic")
    test_leaf_recipes_identified()
    print("[ok] leaf recipes identified")
    test_difficulty_grading_composition_bumps()
    print("[ok] difficulty grading bumps composition")
    test_error_tracker_records_misses()
    print("[ok] error tracker")
    test_error_tracker_hardest_recipe()
    print("[ok] hardest_recipe detection")
    test_error_tracker_no_errors_recipe_never_fires()
    print("[ok] no-attempt recipes handled")
    test_correction_set_add_and_sample()
    print("[ok] correction set sampling")
    test_correction_set_empty_returns_empty()
    print("[ok] empty correction set")
    test_correction_set_from_error_records()
    print("[ok] add_from_errors")
    test_curriculum_add_goal_enqueues_steps()
    print("[ok] add_goal enqueues decomposition")
    test_curriculum_schedule_orders_by_difficulty()
    print("[ok] schedule orders by difficulty")
    test_curriculum_next_recipe_pops_queue()
    print("[ok] next_recipe pops")
    test_curriculum_next_recipe_returns_none_when_done()
    print("[ok] none when done")
    test_curriculum_record_outcome_marks_complete()
    print("[ok] record_outcome completion")
    test_curriculum_training_data_includes_fresh()
    print("[ok] fresh training data")
    test_curriculum_training_data_mixes_corrections()
    print("[ok] mixes corrections at fraction")
    test_curriculum_records_errors_into_corrections()
    print("[ok] errors feed corrections")
