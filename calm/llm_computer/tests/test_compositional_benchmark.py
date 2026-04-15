"""Tests for compositional generalization benchmark."""

from __future__ import annotations

import random

from calm.llm_computer.compositional_benchmark import (
    BenchmarkResult, CompositionalBenchmark, report,
    split_by_difficulty, split_leave_one_out, split_random_fraction,
)
from calm.llm_computer.compositional_data import (
    CompositionalTask,
    template_add_then_echo, template_adder, template_echo_a, template_echo_b,
    template_is_sum_prime, template_max_of_ab,
)


def _all_templates():
    return [
        template_echo_a(),
        template_echo_b(),
        template_adder(),
        template_add_then_echo(),
        template_is_sum_prime(),
        template_max_of_ab(),
    ]


def test_split_leave_one_out():
    templates = _all_templates()
    train, held = split_leave_one_out(templates, ["is_sum_prime", "max_of_ab"])
    assert len(train) == 4
    assert len(held) == 2
    assert {t.name for t in held} == {"is_sum_prime", "max_of_ab"}


def test_split_random_fraction_deterministic():
    templates = _all_templates()
    a_train, a_held = split_random_fraction(templates, fraction=0.3, seed=0)
    b_train, b_held = split_random_fraction(templates, fraction=0.3, seed=0)
    assert [t.name for t in a_train] == [t.name for t in b_train]
    assert [t.name for t in a_held] == [t.name for t in b_held]


def test_split_by_difficulty_holds_compositions():
    templates = _all_templates()
    train, held = split_by_difficulty(templates, min_held_difficulty=2)
    assert all(t.difficulty < 2 for t in train)
    assert all(t.difficulty >= 2 for t in held)
    assert len(held) >= 1


def test_evaluate_perfect_predictor_scores_100():
    templates = _all_templates()
    train, held = split_leave_one_out(templates, ["is_sum_prime"])
    bench = CompositionalBenchmark(train, held, samples_per_template=10, seed=42)
    # Perfect predictor returns the correct answer
    result = bench.evaluate(predict_fn=lambda task: task.answer)
    assert result.train_accuracy == 1.0
    assert result.held_out_accuracy == 1.0
    assert result.generalization_gap == 0.0
    for r in result.per_template.values():
        assert r.n_correct == r.n_samples


def test_evaluate_wrong_predictor_scores_zero_on_adder():
    templates = _all_templates()
    train, held = split_leave_one_out(templates, ["is_sum_prime"])
    bench = CompositionalBenchmark(train, held, samples_per_template=10, seed=42)
    # Wrong predictor always returns -1
    result = bench.evaluate(predict_fn=lambda task: -1)
    for r in result.per_template.values():
        assert r.n_correct == 0


def test_evaluate_handles_exceptions_as_misses():
    templates = [template_adder()]
    bench = CompositionalBenchmark(templates, [], samples_per_template=5, seed=1)
    def raising(_task):
        raise RuntimeError("boom")
    result = bench.evaluate(predict_fn=raising)
    assert result.per_template["adder"].n_correct == 0
    assert result.per_template["adder"].n_samples == 5


def test_per_template_cards_required_populated():
    templates = [template_is_sum_prime()]
    bench = CompositionalBenchmark(templates, [], samples_per_template=5, seed=0)
    result = bench.evaluate(predict_fn=lambda task: task.answer)
    r = result.per_template["is_sum_prime"]
    assert r.cards_required == frozenset({"adder", "is_prime"})
    assert r.difficulty == 2


def test_generalization_gap_computes_correctly():
    templates = _all_templates()
    train, held = split_leave_one_out(templates, ["is_sum_prime", "max_of_ab"])
    bench = CompositionalBenchmark(train, held, samples_per_template=10, seed=1)
    # Predictor gets train right, held wrong
    def partial(task):
        if task.cards_required in (frozenset({"adder", "is_prime"}),
                                    frozenset({"echo_a", "echo_b", "compare"})):
            return -1  # wrong on held-out
        return task.answer  # right on train
    result = bench.evaluate(predict_fn=partial)
    assert result.train_accuracy == 1.0
    assert result.held_out_accuracy == 0.0
    assert result.generalization_gap == 1.0  # pure memorization signal


def test_report_string_includes_key_info():
    templates = [template_adder(), template_is_sum_prime()]
    bench = CompositionalBenchmark([templates[0]], [templates[1]],
                                    samples_per_template=5, seed=0)
    result = bench.evaluate(predict_fn=lambda task: task.answer)
    out = report(result)
    assert "Compositional Benchmark" in out
    assert "train acc" in out
    assert "held-out acc" in out
    assert "adder" in out
    assert "is_sum_prime" in out


def test_deterministic_samples_across_runs():
    templates = [template_adder()]
    b1 = CompositionalBenchmark(templates, [], samples_per_template=5, seed=99)
    b2 = CompositionalBenchmark(templates, [], samples_per_template=5, seed=99)
    s1 = b1.generate_eval_samples()
    s2 = b2.generate_eval_samples()
    assert [t.prompt for t in s1["adder"]] == [t.prompt for t in s2["adder"]]


if __name__ == "__main__":
    test_split_leave_one_out()
    print("[ok] leave-one-out split")
    test_split_random_fraction_deterministic()
    print("[ok] random split is deterministic")
    test_split_by_difficulty_holds_compositions()
    print("[ok] difficulty-stratified split")
    test_evaluate_perfect_predictor_scores_100()
    print("[ok] perfect predictor scores 100%")
    test_evaluate_wrong_predictor_scores_zero_on_adder()
    print("[ok] wrong predictor scores 0%")
    test_evaluate_handles_exceptions_as_misses()
    print("[ok] exceptions counted as misses")
    test_per_template_cards_required_populated()
    print("[ok] per-template cards + difficulty populated")
    test_generalization_gap_computes_correctly()
    print("[ok] generalization gap correct")
    test_report_string_includes_key_info()
    print("[ok] report string readable")
    test_deterministic_samples_across_runs()
    print("[ok] eval samples deterministic")
