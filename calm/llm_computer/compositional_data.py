"""Compositional training data pipeline — multi-card task generation.

The unified CHRLM architecture enables cards to be stacked, but cards
only LEARN to compose if training data exercises composition. A dataset
where each example uses only one card teaches cards in isolation; a
dataset where each example requires MULTIPLE cards to cooperate teaches
routing + composition.

This module generates compositional tasks against the available card
library. Each task is a `(prompt, answer, cards_required, trace)`
tuple where:
  - prompt: input sequence
  - answer: expected output
  - cards_required: the set of cards whose outputs compose to produce
    the answer (used as metadata for gating + curriculum filtering)
  - trace: optional reasoning string documenting the composition
    (used as supervised scratchpad target if desired)

Tasks are declarative — `TaskTemplate` generators produce new (a, b, ...)
random fillings. Curriculum: simpler tasks (1 card) first, building to
multi-card compositions.

MVP covers:
  - Single-card echoes (one card per task)
  - 2-card compositions (adder + is_prime, adder + echo, etc.)
  - Card routing (task requires selecting which card to use based on
    input pattern — e.g., "sum these" vs "is this prime")

Not in MVP: natural language wrapping, multi-turn reasoning, open-ended
problem solving.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional


@dataclass
class CompositionalTask:
    """One generated training example."""
    prompt: tuple[int, ...]
    answer: int
    cards_required: frozenset[str]
    difficulty: int = 1
    trace: str = ""

    def as_tuple(self) -> tuple[tuple[int, ...], int]:
        """Flat (prompt, answer) for direct training."""
        return self.prompt, self.answer


@dataclass
class TaskTemplate:
    """Recipe for generating tasks of a particular type.

    Attributes:
        name: human-readable label.
        cards_required: which cards the answer depends on.
        difficulty: 1 = single-card trivial, 2 = 2-card composition, etc.
        sample_fn: callable(rng) → CompositionalTask producing one random
            instance.
    """
    name: str
    cards_required: frozenset[str]
    difficulty: int
    sample_fn: Callable[[random.Random], CompositionalTask]


# ----- Single-card task templates -----

def template_echo_a() -> TaskTemplate:
    """Return `a` verbatim. Uses just the embedding/echo card."""
    def sample(rng):
        a = rng.randint(0, 3)
        b = rng.randint(0, 3)
        return CompositionalTask(
            prompt=(a, b),
            answer=a,
            cards_required=frozenset({"echo_a"}),
            difficulty=1,
            trace=f"echo_a({a})={a}",
        )
    return TaskTemplate("echo_a", frozenset({"echo_a"}), 1, sample)


def template_echo_b() -> TaskTemplate:
    def sample(rng):
        a = rng.randint(0, 3)
        b = rng.randint(0, 3)
        return CompositionalTask(
            prompt=(a, b),
            answer=b,
            cards_required=frozenset({"echo_b"}),
            difficulty=1,
            trace=f"echo_b({b})={b}",
        )
    return TaskTemplate("echo_b", frozenset({"echo_b"}), 1, sample)


def template_adder() -> TaskTemplate:
    """Compute a+b using the compiled adder."""
    def sample(rng):
        a = rng.randint(0, 3)
        b = rng.randint(0, 3)
        return CompositionalTask(
            prompt=(a, b),
            answer=a + b,
            cards_required=frozenset({"adder"}),
            difficulty=1,
            trace=f"adder({a},{b})={a+b}",
        )
    return TaskTemplate("adder", frozenset({"adder"}), 1, sample)


# ----- Composition templates -----

def template_add_then_echo() -> TaskTemplate:
    """Compute a+b, then echo the result. Requires adder + echo routing."""
    def sample(rng):
        a = rng.randint(0, 3)
        b = rng.randint(0, 3)
        s = a + b
        return CompositionalTask(
            prompt=(a, b),
            answer=s,  # same as adder alone, but conceptually composed
            cards_required=frozenset({"adder", "echo_result"}),
            difficulty=2,
            trace=f"adder({a},{b})={s}; echo({s})={s}",
        )
    return TaskTemplate("add_then_echo", frozenset({"adder", "echo_result"}), 2, sample)


def template_is_sum_prime() -> TaskTemplate:
    """Compute a+b, check if prime. Requires adder + is_prime."""
    def sample(rng):
        a = rng.randint(0, 3)
        b = rng.randint(0, 3)
        s = a + b
        is_prime = s in {2, 3, 5, 7}
        return CompositionalTask(
            prompt=(a, b),
            answer=1 if is_prime else 0,
            cards_required=frozenset({"adder", "is_prime"}),
            difficulty=2,
            trace=f"adder({a},{b})={s}; is_prime({s})={is_prime}",
        )
    return TaskTemplate("is_sum_prime", frozenset({"adder", "is_prime"}), 2, sample)


def template_max_of_ab() -> TaskTemplate:
    """Return max(a, b). Requires echo + compare."""
    def sample(rng):
        a = rng.randint(0, 3)
        b = rng.randint(0, 3)
        return CompositionalTask(
            prompt=(a, b),
            answer=max(a, b),
            cards_required=frozenset({"echo_a", "echo_b", "compare"}),
            difficulty=2,
            trace=f"compare({a},{b}); max={max(a,b)}",
        )
    return TaskTemplate("max_of_ab", frozenset({"echo_a", "echo_b", "compare"}), 2, sample)


# ----- Routing templates (single-turn dispatching) -----

def template_routed(
    task_tag: int,
    inner_template: TaskTemplate,
) -> TaskTemplate:
    """Wraps an inner template with a routing tag as the first prompt
    token. The model must learn to dispatch based on the tag to the
    right card set. Example: tag 0 = add, tag 1 = is_prime.
    """
    def sample(rng):
        inner = inner_template.sample_fn(rng)
        # Prepend tag to prompt
        return CompositionalTask(
            prompt=(task_tag,) + inner.prompt,
            answer=inner.answer,
            cards_required=inner.cards_required | frozenset({"router"}),
            difficulty=inner.difficulty + 1,
            trace=f"route(tag={task_tag}) → {inner.trace}",
        )
    return TaskTemplate(
        name=f"routed_{inner_template.name}",
        cards_required=inner_template.cards_required | frozenset({"router"}),
        difficulty=inner_template.difficulty + 1,
        sample_fn=sample,
    )


# ----- Dataset assembly -----

@dataclass
class DatasetMix:
    """Weighted mix of templates; sample_n produces N tasks with
    frequencies proportional to weights."""
    templates: list[TaskTemplate]
    weights: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.weights:
            self.weights = [1.0] * len(self.templates)
        assert len(self.weights) == len(self.templates)

    def sample_n(self, n: int, seed: int = 0) -> list[CompositionalTask]:
        rng = random.Random(seed)
        total = sum(self.weights)
        out = []
        for _ in range(n):
            r = rng.random() * total
            acc = 0.0
            for tmpl, w in zip(self.templates, self.weights):
                acc += w
                if r <= acc:
                    out.append(tmpl.sample_fn(rng))
                    break
        return out


def curriculum_single_to_compositional(
    single_templates: Iterable[TaskTemplate],
    compositional_templates: Iterable[TaskTemplate],
    phase_split: float = 0.7,
) -> tuple[DatasetMix, DatasetMix]:
    """Return (phase_1_mix, phase_2_mix) where phase 1 emphasizes
    single-card tasks (at `phase_split`) and phase 2 adds compositional
    tasks (1 - phase_split weight on singles, rest on compositionals)."""
    singles = list(single_templates)
    compos = list(compositional_templates)
    # Phase 1: all singles equally
    p1 = DatasetMix(singles, weights=[1.0] * len(singles))
    # Phase 2: singles + compositionals weighted
    p2_templates = singles + compos
    single_weight = (phase_split / len(singles)) if singles else 0.0
    comp_weight = ((1 - phase_split) / len(compos)) if compos else 0.0
    p2_weights = (
        [single_weight] * len(singles) + [comp_weight] * len(compos)
    )
    p2 = DatasetMix(p2_templates, weights=p2_weights)
    return p1, p2


# ----- Filtering / introspection -----

def filter_by_cards(
    tasks: Iterable[CompositionalTask],
    available_cards: Iterable[str],
) -> list[CompositionalTask]:
    """Keep only tasks whose required cards are ALL in available_cards.

    Use during phased training: phase N might only have cards installed
    for phases 0..N-1; filter compositional templates to those the
    current model can actually solve.
    """
    avail = frozenset(available_cards)
    return [t for t in tasks if t.cards_required.issubset(avail)]


def group_by_difficulty(
    tasks: Iterable[CompositionalTask],
) -> dict[int, list[CompositionalTask]]:
    groups: dict[int, list[CompositionalTask]] = {}
    for t in tasks:
        groups.setdefault(t.difficulty, []).append(t)
    return groups
