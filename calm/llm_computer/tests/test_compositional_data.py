"""Tests for compositional data pipeline."""

from __future__ import annotations

import random

import pytest

from calm.llm_computer.compositional_data import (
    CompositionalTask, DatasetMix, TaskTemplate,
    curriculum_single_to_compositional,
    filter_by_cards, group_by_difficulty,
    template_add_then_echo, template_adder, template_echo_a, template_echo_b,
    template_is_sum_prime, template_max_of_ab, template_routed,
)


def test_single_card_templates_produce_correct_answers():
    rng = random.Random(0)
    echo_a = template_echo_a().sample_fn(rng)
    assert echo_a.answer == echo_a.prompt[0]
    echo_b = template_echo_b().sample_fn(rng)
    assert echo_b.answer == echo_b.prompt[1]
    adder = template_adder().sample_fn(rng)
    assert adder.answer == adder.prompt[0] + adder.prompt[1]


def test_composition_templates_declare_cards_required():
    t = template_is_sum_prime()
    assert t.cards_required == frozenset({"adder", "is_prime"})
    assert t.difficulty == 2

    t2 = template_add_then_echo()
    assert "adder" in t2.cards_required
    assert "echo_result" in t2.cards_required

    t3 = template_max_of_ab()
    assert t3.cards_required == frozenset({"echo_a", "echo_b", "compare"})


def test_is_sum_prime_answer_correct():
    rng = random.Random(42)
    for _ in range(20):
        task = template_is_sum_prime().sample_fn(rng)
        a, b = task.prompt
        s = a + b
        expected = 1 if s in {2, 3, 5, 7} else 0
        assert task.answer == expected, (
            f"is_sum_prime({a},{b})={task.answer}, expected {expected}"
        )


def test_routed_wrapper_prepends_tag_and_inherits_inner():
    routed_adder = template_routed(task_tag=0, inner_template=template_adder())
    rng = random.Random(0)
    task = routed_adder.sample_fn(rng)
    assert task.prompt[0] == 0  # routing tag
    assert task.answer == task.prompt[1] + task.prompt[2]
    assert "router" in task.cards_required
    assert "adder" in task.cards_required
    assert task.difficulty == 2  # adder difficulty (1) + routing overhead


def test_dataset_mix_samples_n_with_weights():
    templates = [
        template_echo_a(),
        template_adder(),
    ]
    mix = DatasetMix(templates, weights=[1.0, 3.0])
    tasks = mix.sample_n(400, seed=1)
    # Count by template name (inferred from cards_required)
    echo_count = sum(1 for t in tasks if t.cards_required == frozenset({"echo_a"}))
    adder_count = sum(1 for t in tasks if t.cards_required == frozenset({"adder"}))
    assert echo_count + adder_count == 400
    # Adder should be ~3x more frequent than echo (allow 20% deviation)
    ratio = adder_count / max(1, echo_count)
    assert 2.3 < ratio < 3.9, f"adder:echo ratio = {ratio:.2f}, expected ~3"


def test_dataset_mix_default_weights_uniform():
    templates = [template_echo_a(), template_echo_b(), template_adder()]
    mix = DatasetMix(templates)
    assert mix.weights == [1.0, 1.0, 1.0]


def test_curriculum_builder():
    singles = [template_echo_a(), template_adder()]
    compos = [template_is_sum_prime()]
    p1, p2 = curriculum_single_to_compositional(
        singles, compos, phase_split=0.8,
    )
    # p1 should only contain singles
    assert len(p1.templates) == 2
    # p2 should contain all + re-weighted
    assert len(p2.templates) == 3
    # Singles weight 0.8/2=0.4 each, compositional weight 0.2/1=0.2
    assert p2.weights[0] == pytest.approx(0.4)
    assert p2.weights[1] == pytest.approx(0.4)
    assert p2.weights[2] == pytest.approx(0.2)


def test_filter_by_cards():
    tasks = [
        template_echo_a().sample_fn(random.Random(0)),
        template_is_sum_prime().sample_fn(random.Random(0)),
        template_max_of_ab().sample_fn(random.Random(0)),
    ]
    # Only "echo_a" available
    kept = filter_by_cards(tasks, available_cards={"echo_a"})
    assert len(kept) == 1
    assert kept[0].cards_required == frozenset({"echo_a"})

    # adder + is_prime available
    kept = filter_by_cards(tasks, available_cards={"adder", "is_prime"})
    assert len(kept) == 1
    assert kept[0].cards_required == frozenset({"adder", "is_prime"})


def test_group_by_difficulty():
    rng = random.Random(0)
    tasks = [
        template_echo_a().sample_fn(rng),     # diff 1
        template_adder().sample_fn(rng),      # diff 1
        template_is_sum_prime().sample_fn(rng),  # diff 2
    ]
    groups = group_by_difficulty(tasks)
    assert set(groups.keys()) == {1, 2}
    assert len(groups[1]) == 2
    assert len(groups[2]) == 1


def test_compositional_task_as_tuple():
    task = CompositionalTask(
        prompt=(1, 2), answer=3, cards_required=frozenset({"adder"}),
    )
    prompt, answer = task.as_tuple()
    assert prompt == (1, 2)
    assert answer == 3


def test_trace_is_descriptive():
    task = template_is_sum_prime().sample_fn(random.Random(42))
    assert "adder" in task.trace
    assert "is_prime" in task.trace


if __name__ == "__main__":
    test_single_card_templates_produce_correct_answers()
    print("[ok] single-card templates correct")
    test_composition_templates_declare_cards_required()
    print("[ok] compositions declare required cards")
    test_is_sum_prime_answer_correct()
    print("[ok] is_sum_prime answers correct")
    test_routed_wrapper_prepends_tag_and_inherits_inner()
    print("[ok] routed wrapper prepends tag")
    test_dataset_mix_samples_n_with_weights()
    print("[ok] DatasetMix respects weights")
    test_dataset_mix_default_weights_uniform()
    print("[ok] default weights are uniform")
    test_curriculum_builder()
    print("[ok] curriculum builder")
    test_filter_by_cards()
    print("[ok] filter_by_cards")
    test_group_by_difficulty()
    print("[ok] group_by_difficulty")
    test_compositional_task_as_tuple()
    print("[ok] CompositionalTask tuple interface")
    test_trace_is_descriptive()
    print("[ok] trace field is descriptive")
