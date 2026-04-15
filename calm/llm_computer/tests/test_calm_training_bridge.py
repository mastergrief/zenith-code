"""Tests for CALM training bridge."""

from __future__ import annotations

import random

import pytest

from calm.llm_computer.calm_training_bridge import (
    CALMOracle, CALMRecipe, calm_verified_gate, default_oracle,
    recipe_adder_small, recipe_gcd_small, recipe_is_prime_small,
    recipe_is_sum_prime, recipe_max_ab, recipe_sum_mod2,
)


def test_recipe_adder_generates_correct_labels():
    oracle = CALMOracle()
    oracle.register(recipe_adder_small())
    tasks = oracle.sample("calm_adder_small", n=20, seed=0)
    assert len(tasks) == 20
    # Every label must equal a+b via Python (CALM matches Python on simple ops)
    for t in tasks:
        a, b = t.prompt
        assert t.answer == a + b, (
            f"CALM-labeled adder wrong: {a}+{b}={t.answer}, expected {a+b}"
        )


def test_recipe_is_prime_labels_correct():
    oracle = CALMOracle()
    oracle.register(recipe_is_prime_small())
    tasks = oracle.sample("calm_is_prime", n=30, seed=1)
    primes = {2, 3, 5, 7}
    for t in tasks:
        n = t.prompt[0]
        expected = 1 if n in primes else 0
        assert t.answer == expected, (
            f"is_prime({n}) labeled {t.answer}, expected {expected}"
        )


def test_recipe_is_sum_prime_correct():
    """Compositional recipe: is_prime(a+b)."""
    oracle = CALMOracle()
    oracle.register(recipe_is_sum_prime())
    tasks = oracle.sample("calm_is_sum_prime", n=50, seed=2)
    primes = {2, 3, 5, 7}
    for t in tasks:
        a, b = t.prompt
        expected = 1 if (a + b) in primes else 0
        assert t.answer == expected, (
            f"is_prime({a}+{b})={t.answer}, expected {expected}"
        )
    # Must have cards_required declared
    assert all(t.cards_required == frozenset({"adder", "is_prime"})
               for t in tasks)
    assert all(t.difficulty == 2 for t in tasks)


def test_recipe_gcd_correct():
    import math
    oracle = CALMOracle()
    oracle.register(recipe_gcd_small())
    tasks = oracle.sample("calm_gcd", n=30, seed=3)
    for t in tasks:
        a, b = t.prompt
        assert t.answer == math.gcd(a, b)


def test_recipe_sum_mod2_correct():
    oracle = CALMOracle()
    oracle.register(recipe_sum_mod2())
    tasks = oracle.sample("calm_sum_parity", n=30, seed=4)
    for t in tasks:
        a, b = t.prompt
        assert t.answer == (a + b) % 2


def test_recipe_max_ab_correct():
    oracle = CALMOracle()
    oracle.register(recipe_max_ab())
    tasks = oracle.sample("calm_max_ab", n=30, seed=5)
    for t in tasks:
        a, b = t.prompt
        assert t.answer == max(a, b)


def test_trace_field_populated():
    oracle = CALMOracle()
    oracle.register(recipe_is_sum_prime())
    tasks = oracle.sample("calm_is_sum_prime", n=3, seed=0)
    for t in tasks:
        assert "calm_eval" in t.trace
        assert "is_prime" in t.trace


def test_unknown_recipe_raises():
    oracle = CALMOracle()
    with pytest.raises(KeyError, match="unknown recipe"):
        oracle.sample("does_not_exist", n=5)


def test_duplicate_recipe_raises():
    oracle = CALMOracle()
    oracle.register(recipe_adder_small())
    with pytest.raises(ValueError, match="already registered"):
        oracle.register(recipe_adder_small())


def test_build_template_produces_task_template():
    oracle = CALMOracle()
    oracle.register(recipe_is_sum_prime())
    tmpl = oracle.build_template("calm_is_sum_prime")
    # Use it like a normal TaskTemplate
    rng = random.Random(99)
    for _ in range(10):
        task = tmpl.sample_fn(rng)
        a, b = task.prompt
        primes = {2, 3, 5, 7}
        expected = 1 if (a + b) in primes else 0
        assert task.answer == expected


def test_deterministic_given_seed():
    oracle = CALMOracle()
    oracle.register(recipe_adder_small())
    s1 = oracle.sample("calm_adder_small", n=20, seed=42)
    s2 = oracle.sample("calm_adder_small", n=20, seed=42)
    assert [t.prompt for t in s1] == [t.prompt for t in s2]
    assert [t.answer for t in s1] == [t.answer for t in s2]


def test_default_oracle_has_all_recipes():
    oracle = default_oracle()
    names = set(oracle.recipes())
    assert "calm_adder_small" in names
    assert "calm_is_prime" in names
    assert "calm_is_sum_prime" in names
    assert "calm_gcd" in names
    assert "calm_sum_parity" in names
    assert "calm_max_ab" in names


def test_calm_verified_gate_perfect_predictor_scores_1():
    """Gate passes when predictor matches CALM."""
    recipe = recipe_is_sum_prime()
    score = calm_verified_gate(
        predict_fn=lambda task: task.answer,  # perfect
        recipe=recipe, n_samples=20, seed=7,
    )
    assert score == 1.0


def test_calm_verified_gate_wrong_predictor_scores_0():
    recipe = recipe_adder_small()
    score = calm_verified_gate(
        predict_fn=lambda task: -1,  # always wrong
        recipe=recipe, n_samples=10, seed=8,
    )
    assert score == 0.0


def test_calm_verified_gate_partial_predictor():
    """Predictor right half the time → gate score ~0.5."""
    recipe = recipe_adder_small()
    call_count = [0]
    def alternating(task):
        call_count[0] += 1
        return task.answer if call_count[0] % 2 == 0 else -1
    score = calm_verified_gate(
        predict_fn=alternating, recipe=recipe, n_samples=20, seed=9,
    )
    # Exactly half (10/20) should match
    assert score == 0.5


if __name__ == "__main__":
    test_recipe_adder_generates_correct_labels()
    print("[ok] adder recipe labels correct")
    test_recipe_is_prime_labels_correct()
    print("[ok] is_prime recipe labels correct")
    test_recipe_is_sum_prime_correct()
    print("[ok] is_sum_prime compositional recipe correct")
    test_recipe_gcd_correct()
    print("[ok] gcd recipe correct")
    test_recipe_sum_mod2_correct()
    print("[ok] sum_mod2 recipe correct")
    test_recipe_max_ab_correct()
    print("[ok] max_ab recipe correct")
    test_trace_field_populated()
    print("[ok] trace field has CALM eval string")
    test_unknown_recipe_raises()
    print("[ok] unknown recipe raises")
    test_duplicate_recipe_raises()
    print("[ok] duplicate recipe raises")
    test_build_template_produces_task_template()
    print("[ok] build_template integrates with TaskTemplate")
    test_deterministic_given_seed()
    print("[ok] deterministic given seed")
    test_default_oracle_has_all_recipes()
    print("[ok] default_oracle populated")
    test_calm_verified_gate_perfect_predictor_scores_1()
    print("[ok] calm_verified_gate perfect → 1")
    test_calm_verified_gate_wrong_predictor_scores_0()
    print("[ok] calm_verified_gate wrong → 0")
    test_calm_verified_gate_partial_predictor()
    print("[ok] calm_verified_gate partial predictor")
