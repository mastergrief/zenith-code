"""Reasoning curriculum — use CALM's cognitive modules' STRUCTURE to
drive phased training for SubstrateHRLM.

CALM's cognitive modules (calm/router.py, calm/adaptive.py,
calm/module_learning.py, etc.) are designed for TEXT responses from
full-scale LLMs. At SubstrateHRLM's toy scale (1-bit outputs), they
can't directly analyze outputs. But their STRUCTURE maps cleanly onto
our phased training:

  CALM cognitive module          ↔  Our training equivalent
  ------------------------       -------------------------
  decompose (breaks problem)     Decomposer: recipe → sub-recipes
  adaptive (estimates difficulty) DifficultyGrader: recipe → int
  module_learning (tracks issues) ErrorTracker: per-recipe miss counts
  MathCollector/BoolCollector     CorrectionSet: errors → next phase

This module ships the bridge. Each piece is pure Python (no model
inference needed) so works at any scale.

Usage:
  curriculum = ReasoningCurriculum(default_oracle())
  curriculum.add_goal("calm_is_sum_prime")
  for phase_recipe in curriculum.schedule():
      # train on phase_recipe...
      curriculum.record_outcome(phase_recipe, accuracy)

Produces a sequence of phases ordered by difficulty, with harder
tasks introduced only after easier prerequisites pass.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from calm.llm_computer.calm_training_bridge import (
    CALMOracle, CALMRecipe,
)
from calm.llm_computer.compositional_data import CompositionalTask


# ----- Decomposition -----

@dataclass
class DecomposedStep:
    """One step of a decomposed reasoning trace.

    Attributes:
        step_id: ordinal (0 = first).
        recipe_name: CALM recipe to evaluate at this step.
        depends_on: step_ids this step needs to have run first.
        description: human-readable note (for logging/trace).
    """
    step_id: int
    recipe_name: str
    depends_on: tuple[int, ...] = ()
    description: str = ""


@dataclass
class ReasoningTrace:
    """The sequence of steps a reasoning curriculum produces for a goal."""
    goal_recipe: str
    steps: tuple[DecomposedStep, ...]

    @property
    def leaf_recipes(self) -> frozenset[str]:
        """Recipes that don't depend on other steps — train first."""
        return frozenset(
            s.recipe_name for s in self.steps if not s.depends_on
        )

    @property
    def final_step(self) -> DecomposedStep:
        return self.steps[-1]


class Decomposer:
    """Break compositional goal recipes into sub-recipe chains.

    Hard-coded for the MVP; real CALM decompose module would use NL
    reasoning. Knows: is_sum_prime = adder + is_prime, sum_mod2 = adder
    + modulo, etc.
    """

    # Static decomposition map: goal_recipe -> reasoning trace
    _DECOMPOSITIONS: dict[str, list[DecomposedStep]] = {
        "calm_is_sum_prime": [
            DecomposedStep(0, "calm_adder_small", (), "first compute a+b"),
            DecomposedStep(1, "calm_is_prime", (0,), "then check is_prime(sum)"),
            DecomposedStep(2, "calm_is_sum_prime", (0, 1),
                          "joint: is_prime(a+b) end-to-end"),
        ],
        "calm_sum_parity": [
            DecomposedStep(0, "calm_adder_small", (), "first compute a+b"),
            DecomposedStep(1, "calm_sum_parity", (0,),
                          "joint: (a+b) mod 2 end-to-end"),
        ],
    }

    def decompose(self, goal_recipe: str) -> ReasoningTrace:
        """Return a ReasoningTrace for the goal. If no decomposition is
        known, returns a single-step trace (the recipe itself)."""
        steps = self._DECOMPOSITIONS.get(goal_recipe)
        if steps is None:
            steps = [DecomposedStep(0, goal_recipe, (), "atomic recipe")]
        return ReasoningTrace(
            goal_recipe=goal_recipe,
            steps=tuple(steps),
        )


# ----- Difficulty grading -----

def grade_difficulty(recipe: CALMRecipe) -> int:
    """Rough grade 1..5 based on recipe metadata.

    Maps to: 1=trivial, 2=easy, 3=medium, 4=hard, 5=deep.
    Uses recipe.difficulty as base; bumps for many variables or
    compositional nature.
    """
    base = recipe.difficulty
    n_vars = len(recipe.var_ranges)
    n_cards = len(recipe.cards_required)
    # Composition bumps grade
    grade = base
    if n_cards > 1:
        grade += 1
    if n_vars > 2:
        grade += 1
    return min(5, max(1, grade))


# ----- Error tracking (module_learning analog) -----

@dataclass
class ErrorRecord:
    recipe_name: str
    task: CompositionalTask
    predicted: int
    correct: int


class ErrorTracker:
    """Tracks recipe-level miss rates. Analogous to CALM's
    module_learning which tracks recurring quality issues."""

    def __init__(self):
        self._by_recipe: dict[str, list[ErrorRecord]] = defaultdict(list)
        self._attempts: dict[str, int] = defaultdict(int)

    def record_outcome(
        self, recipe_name: str, task: CompositionalTask,
        predicted: int, correct: int,
    ) -> None:
        self._attempts[recipe_name] += 1
        if predicted != correct:
            self._by_recipe[recipe_name].append(
                ErrorRecord(recipe_name=recipe_name, task=task,
                            predicted=predicted, correct=correct)
            )

    def error_rate(self, recipe_name: str) -> float:
        n = self._attempts.get(recipe_name, 0)
        if n == 0:
            return 0.0
        return len(self._by_recipe[recipe_name]) / n

    def errors(self, recipe_name: str) -> list[ErrorRecord]:
        return list(self._by_recipe[recipe_name])

    def hardest_recipe(self) -> Optional[str]:
        """Recipe with highest error rate (ignoring recipes with 0 attempts)."""
        scored = [
            (n, self.error_rate(n))
            for n in self._attempts if self._attempts[n] > 0
        ]
        if not scored:
            return None
        return max(scored, key=lambda x: x[1])[0]

    def all_recipe_error_rates(self) -> dict[str, float]:
        return {n: self.error_rate(n) for n in self._attempts}


# ----- Correction set (MathCollector analog) -----

@dataclass
class CorrectionSet:
    """Mistakes from prior phase become oversampled training data
    for the next phase. Analogous to CALM's MathCollector/BoolCollector.
    """
    _entries: list[CompositionalTask] = field(default_factory=list)

    def add(self, task: CompositionalTask) -> None:
        self._entries.append(task)

    def add_from_errors(self, errors: Iterable[ErrorRecord]) -> None:
        for e in errors:
            self.add(e.task)

    def as_training_data(
        self, n: int, seed: int = 0,
        oversample_factor: int = 3,
    ) -> list[CompositionalTask]:
        """Sample n examples from the correction set with replacement.
        If set is small, we cycle; if empty, returns empty list."""
        if not self._entries:
            return []
        rng = random.Random(seed)
        # Each correction example gets oversample_factor weight
        pool = self._entries * oversample_factor
        return [rng.choice(pool) for _ in range(n)]

    def __len__(self) -> int:
        return len(self._entries)


# ----- Reasoning curriculum orchestrator -----

class ReasoningCurriculum:
    """Decomposes goals into phased sub-recipe ladders, schedules by
    difficulty, tracks outcomes, drives the next phase.

    Pattern:
      curriculum.add_goal("calm_is_sum_prime")
      for recipe_name in curriculum.schedule():
          train_phase_on(recipe_name)
          curriculum.record_outcome(recipe_name, accuracy)
    """

    def __init__(
        self,
        oracle: CALMOracle,
        decomposer: Optional[Decomposer] = None,
    ):
        self.oracle = oracle
        self.decomposer = decomposer or Decomposer()
        self._pending: list[str] = []  # recipes queued for training
        self._completed: set[str] = set()
        self._outcomes: dict[str, float] = {}  # recipe → final accuracy
        self.error_tracker = ErrorTracker()
        self.corrections = CorrectionSet()

    def add_goal(self, goal_recipe: str) -> ReasoningTrace:
        """Decompose a goal and enqueue its steps in dependency order."""
        trace = self.decomposer.decompose(goal_recipe)
        # Topological order via step dependencies (steps already in order
        # in our MVP decompositions, but validate)
        for step in trace.steps:
            if step.recipe_name not in self._pending \
                and step.recipe_name not in self._completed:
                self._pending.append(step.recipe_name)
        return trace

    def schedule(self) -> list[str]:
        """Return pending recipes ordered by difficulty (easier first)."""
        def key(recipe_name: str) -> int:
            recipe = self.oracle._recipes.get(recipe_name)
            if recipe is None:
                return 99
            return grade_difficulty(recipe)
        return sorted(self._pending, key=key)

    def next_recipe(self) -> Optional[str]:
        """Return the next recipe to train on, or None if done.
        Removes the recipe from pending."""
        ordered = self.schedule()
        if not ordered:
            return None
        chosen = ordered[0]
        self._pending.remove(chosen)
        return chosen

    def record_outcome(
        self, recipe_name: str, accuracy: float,
        errors: Iterable[ErrorRecord] = (),
    ) -> None:
        """Mark a recipe as completed with its final accuracy. Optionally
        feed any error records into the correction set for next phase."""
        self._outcomes[recipe_name] = accuracy
        self._completed.add(recipe_name)
        self.corrections.add_from_errors(errors)

    def completed_recipes(self) -> list[str]:
        return sorted(self._completed)

    def outcomes(self) -> dict[str, float]:
        return dict(self._outcomes)

    def training_data_for(
        self, recipe_name: str, n: int,
        include_corrections: bool = True,
        correction_fraction: float = 0.3,
        seed: int = 0,
    ) -> list[CompositionalTask]:
        """Build training data for a recipe. If `include_corrections` is
        True and the correction set is non-empty, mix in correction
        examples at `correction_fraction` of total."""
        n_corrections = (
            int(n * correction_fraction) if include_corrections
            and len(self.corrections) > 0
            else 0
        )
        n_fresh = n - n_corrections
        fresh = self.oracle.sample(recipe_name, n=n_fresh, seed=seed)
        corrections = (
            self.corrections.as_training_data(n_corrections, seed=seed + 1)
            if n_corrections > 0 else []
        )
        return fresh + corrections
