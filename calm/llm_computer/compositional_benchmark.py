"""Compositional generalization benchmark — does the model handle UNSEEN
compositions of already-installed cards?

The compositional data pipeline (compositional_data.py) generates tasks
that require multiple cards. If we train on a subset of card-pair
compositions and eval on the rest, we measure whether the model
*generalizes compositions* versus *memorizes specific pairs*.

Without this, we're shipping sub-card infrastructure whose core value
proposition — composition — is unmeasured.

Split strategies:
  - Leave-one-out: hold out all tasks whose cards_required set equals
    a specific held-out set (e.g. hold out all {adder, is_prime} tasks
    from training, test on them).
  - Random fraction: hold out k% of templates uniformly at random.
  - Difficulty stratified: hold out the hardest composition tier.

Result: per-template accuracy, overall pass rate, and a "generalization
gap" metric (train_acc - test_acc) for the composition category.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterable

from calm.llm_computer.compositional_data import (
    CompositionalTask, DatasetMix, TaskTemplate,
)


@dataclass
class TemplateEvalResult:
    template_name: str
    n_samples: int
    n_correct: int
    cards_required: frozenset[str]
    difficulty: int

    @property
    def accuracy(self) -> float:
        return self.n_correct / max(1, self.n_samples)


@dataclass
class BenchmarkResult:
    """Result of running a compositional benchmark."""
    train_templates: list[str]
    held_out_templates: list[str]
    per_template: dict[str, TemplateEvalResult] = field(default_factory=dict)

    @property
    def train_accuracy(self) -> float:
        if not self.train_templates:
            return 0.0
        accs = [self.per_template[n].accuracy for n in self.train_templates
                if n in self.per_template]
        return sum(accs) / max(1, len(accs))

    @property
    def held_out_accuracy(self) -> float:
        if not self.held_out_templates:
            return 0.0
        accs = [self.per_template[n].accuracy for n in self.held_out_templates
                if n in self.per_template]
        return sum(accs) / max(1, len(accs))

    @property
    def generalization_gap(self) -> float:
        """Train acc - held-out acc. Positive = memorization signal."""
        return self.train_accuracy - self.held_out_accuracy


def split_leave_one_out(
    templates: Iterable[TaskTemplate],
    held_out_names: Iterable[str],
) -> tuple[list[TaskTemplate], list[TaskTemplate]]:
    """Partition templates by name: held_out_names in the held-out set,
    everything else in train."""
    held_names = set(held_out_names)
    train, held = [], []
    for t in templates:
        if t.name in held_names:
            held.append(t)
        else:
            train.append(t)
    return train, held


def split_random_fraction(
    templates: Iterable[TaskTemplate],
    fraction: float,
    seed: int = 0,
) -> tuple[list[TaskTemplate], list[TaskTemplate]]:
    """Randomly hold out `fraction` of templates."""
    rng = random.Random(seed)
    lst = list(templates)
    rng.shuffle(lst)
    n_held = max(1, int(len(lst) * fraction))
    return lst[n_held:], lst[:n_held]


def split_by_difficulty(
    templates: Iterable[TaskTemplate],
    min_held_difficulty: int,
) -> tuple[list[TaskTemplate], list[TaskTemplate]]:
    """Hold out all templates with difficulty >= min_held_difficulty."""
    train, held = [], []
    for t in templates:
        if t.difficulty >= min_held_difficulty:
            held.append(t)
        else:
            train.append(t)
    return train, held


class CompositionalBenchmark:
    """Benchmark with fixed template split. Call `evaluate(predict_fn)`
    with any predictor; returns BenchmarkResult."""

    def __init__(
        self,
        train_templates: list[TaskTemplate],
        held_out_templates: list[TaskTemplate],
        samples_per_template: int = 20,
        seed: int = 42,
    ):
        self.train_templates = train_templates
        self.held_out_templates = held_out_templates
        self.samples_per_template = samples_per_template
        self.seed = seed

    def generate_eval_samples(self) -> dict[str, list[CompositionalTask]]:
        """Per-template sample bank (deterministic given seed)."""
        out = {}
        base_seed = self.seed
        for t in self.train_templates + self.held_out_templates:
            rng = random.Random(base_seed + hash(t.name) % 10000)
            out[t.name] = [t.sample_fn(rng) for _ in range(self.samples_per_template)]
        return out

    def evaluate(
        self,
        predict_fn: Callable[[CompositionalTask], int],
    ) -> BenchmarkResult:
        """Run predict_fn on every sample, compute per-template accuracy.

        predict_fn takes a CompositionalTask (has .prompt), returns the
        predicted answer int. Compared against task.answer.
        """
        samples = self.generate_eval_samples()
        per_template = {}
        for t in self.train_templates + self.held_out_templates:
            correct = 0
            for task in samples[t.name]:
                try:
                    pred = predict_fn(task)
                    if pred == task.answer:
                        correct += 1
                except Exception:
                    pass  # count as miss
            per_template[t.name] = TemplateEvalResult(
                template_name=t.name,
                n_samples=len(samples[t.name]),
                n_correct=correct,
                cards_required=t.cards_required,
                difficulty=t.difficulty,
            )
        return BenchmarkResult(
            train_templates=[t.name for t in self.train_templates],
            held_out_templates=[t.name for t in self.held_out_templates],
            per_template=per_template,
        )


def report(result: BenchmarkResult) -> str:
    """Pretty-print a BenchmarkResult."""
    lines = ["=== Compositional Benchmark ===",
             f"  train acc:       {result.train_accuracy*100:.1f}%",
             f"  held-out acc:    {result.held_out_accuracy*100:.1f}%",
             f"  generalization gap: {result.generalization_gap*100:+.1f}pp",
             "",
             "Per-template:"]
    for name in result.train_templates + result.held_out_templates:
        r = result.per_template.get(name)
        if not r:
            continue
        tag = "train" if name in result.train_templates else "HELD"
        lines.append(f"  [{tag}] {name:30s} {r.accuracy*100:5.1f}% "
                    f"({r.n_correct}/{r.n_samples})  diff={r.difficulty}  "
                    f"cards={sorted(r.cards_required)}")
    return "\n".join(lines)
