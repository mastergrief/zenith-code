"""Closed-loop feedback tests — does the learning cycle actually close?

The hypothesis: `AutoLearner` implements
    correction → store pattern → match future prompt → precompute
and the loop produces the correct value on the downstream prompt.

These are the first tests for `auto_learn.py` — the core of Vector 1
(closing the feedback loop). Without this coverage, we have no way to
detect if the loop silently breaks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from calm.auto_learn import AutoLearner, LearnedPattern


@dataclass
class _FakeClaim:
    """Minimal Claim surrogate for learn_from_correction."""
    expression: str
    actual_value: Optional[object] = 0
    correct: bool = False


@pytest.fixture
def tmp_db(tmp_path) -> Path:
    return tmp_path / "patterns.jsonl"


def test_learn_generalizes_multiplication(tmp_db):
    """Correcting 17*23 should learn a generalized arithmetic pattern,
    not the specific '17 * 23'. The generalizer uses chr(78 + idx) so
    two operands → 'N * O'."""
    learner = AutoLearner(db_path=tmp_db)
    learner.learn_from_correction(_FakeClaim(expression="17 * 23"))
    assert "N * O" in learner._patterns, \
        f"expected 'N * O', got {list(learner._patterns)}"


def test_learn_generalizes_function_call(tmp_db):
    """Correcting is_prime(391) should learn 'is_prime(N)'."""
    learner = AutoLearner(db_path=tmp_db)
    learner.learn_from_correction(_FakeClaim(expression="is_prime(391)"))
    assert "is_prime(N)" in learner._patterns, \
        f"expected 'is_prime(N)', got {list(learner._patterns)}"


def test_learn_generalizes_two_arg_function(tmp_db):
    """gcd(48, 180) → 'gcd(N, O)'."""
    learner = AutoLearner(db_path=tmp_db)
    learner.learn_from_correction(_FakeClaim(expression="gcd(48, 180)"))
    assert "gcd(N, O)" in learner._patterns, \
        f"expected 'gcd(N, O)', got {list(learner._patterns)}"


def test_frequency_increments_on_repeat(tmp_db):
    """Same correction twice → frequency bumps, one pattern entry."""
    learner = AutoLearner(db_path=tmp_db)
    learner.learn_from_correction(_FakeClaim(expression="17 * 23"))
    learner.learn_from_correction(_FakeClaim(expression="5 * 8"))
    # Both should generalize to same 'N * M' pattern.
    matching = [p for k, p in learner._patterns.items() if "*" in k]
    assert len(matching) == 1, f"expected 1 pattern, got {len(matching)}: {list(learner._patterns)}"
    assert matching[0].frequency == 2


def test_correct_claims_do_not_learn(tmp_db):
    """Claims marked correct should not generate patterns."""
    learner = AutoLearner(db_path=tmp_db)
    learner.learn_from_correction(_FakeClaim(expression="17 * 23", correct=True))
    assert len(learner._patterns) == 0


def test_persistence_roundtrip(tmp_db):
    """Learn, create new AutoLearner, patterns should survive."""
    l1 = AutoLearner(db_path=tmp_db)
    l1.learn_from_correction(_FakeClaim(expression="factorial(7)"))
    l1.learn_from_correction(_FakeClaim(expression="is_prime(391)"))
    del l1

    l2 = AutoLearner(db_path=tmp_db)
    assert len(l2._patterns) == 2
    assert "factorial(N)" in l2._patterns
    assert "is_prime(N)" in l2._patterns


def test_suggest_precomputes_empty_db(tmp_db):
    """No patterns → no suggestions."""
    learner = AutoLearner(db_path=tmp_db)
    result = learner.suggest_precomputes("what is 347 * 289")
    assert result == {}


def test_loop_closes_for_arithmetic(tmp_db):
    """THE FEEDBACK LOOP TEST — learn from one correction, verify next
    similar prompt gets the precomputed (correct) value.

    Round 1: prompt '17 * 23' would produce error → correction fires.
             AutoLearner records 'N * M' pattern.
    Round 2: prompt 'what is 347 * 289?' arrives. Pattern matches, the
             engine precomputes 347 * 289 = 100283 and would inject as
             a verified fact, skipping the error cycle entirely.

    The test proves the precompute returns the correct value."""
    learner = AutoLearner(db_path=tmp_db)

    # Step 1: simulate a correction on '17 * 23'.
    learner.learn_from_correction(_FakeClaim(expression="17 * 23"))

    # Step 2: a DIFFERENT multiplication prompt arrives.
    precomputes = learner.suggest_precomputes("what is 347 * 289?")

    # Loop closed? Expected '347 * 289' key with value 100283.
    assert any("347" in k and "289" in k and "*" in k for k in precomputes), \
        f"expected 347*289 in precomputes, got {precomputes}"
    matching_key = next(k for k in precomputes if "347" in k and "289" in k and "*" in k)
    assert precomputes[matching_key] == 100283, \
        f"expected 100283, got {precomputes[matching_key]}"


def test_loop_closes_for_function_call(tmp_db):
    """Same loop test but for function-call patterns."""
    learner = AutoLearner(db_path=tmp_db)
    learner.learn_from_correction(_FakeClaim(expression="is_prime(391)"))

    # Next prompt — check if 523 is prime (523 IS prime, for the record).
    precomputes = learner.suggest_precomputes("is 523 prime")
    assert "is_prime(523)" in precomputes, \
        f"expected is_prime(523), got {precomputes}"
    assert precomputes["is_prime(523)"] is True


def test_large_number_guard(tmp_db):
    """Guard: a pattern instantiated with >10M operand should skip
    (avoids hanging on factorial(big_number) from a credit-card number)."""
    learner = AutoLearner(db_path=tmp_db)
    learner.learn_from_correction(_FakeClaim(expression="factorial(5)"))
    # Prompt contains a big number that shouldn't be factorial'd.
    precomputes = learner.suggest_precomputes(
        "my card number is 4532015112830366 and i want to know about factorial")
    # Guard should have skipped anything over 10M.
    for key in precomputes:
        if "factorial" in key:
            # extract the argument
            import re
            m = re.search(r"factorial\((\d+)\)", key)
            if m:
                assert int(m.group(1)) <= 10_000_000, \
                    f"guard failed: computed {key}"


def test_hit_counter_increments_on_fire(tmp_db):
    """Every successful precompute should bump the firing pattern's hits."""
    learner = AutoLearner(db_path=tmp_db)
    learner.learn_from_correction(_FakeClaim(expression="17 * 23"))
    pat = learner._patterns["N * O"]
    assert pat.hits == 0

    learner.suggest_precomputes("what is 5 * 7?")
    assert pat.hits == 1
    learner.suggest_precomputes("what is 100 * 200?")
    assert pat.hits == 2


def test_hit_counter_not_incremented_on_miss(tmp_db):
    """Prompts with no matching numbers shouldn't bump hits."""
    learner = AutoLearner(db_path=tmp_db)
    learner.learn_from_correction(_FakeClaim(expression="17 * 23"))
    pat = learner._patterns["N * O"]
    learner.suggest_precomputes("this prompt has no numbers in it")
    assert pat.hits == 0


def test_hit_counter_persists(tmp_db):
    """Hit counters should survive a save/load cycle."""
    l1 = AutoLearner(db_path=tmp_db)
    l1.learn_from_correction(_FakeClaim(expression="17 * 23"))
    l1.suggest_precomputes("what is 5 * 7?")
    l1.suggest_precomputes("what is 100 * 200?")
    del l1

    l2 = AutoLearner(db_path=tmp_db)
    assert l2._patterns["N * O"].hits == 2


def test_prune_cold_patterns(tmp_db):
    """A pattern learned once and never fired should be pruned. A hot
    one should survive."""
    learner = AutoLearner(db_path=tmp_db)
    learner.learn_from_correction(_FakeClaim(expression="factorial(7)"))
    learner.learn_from_correction(_FakeClaim(expression="17 * 23"))
    learner.suggest_precomputes("what is 5 * 7?")  # hits 'N * O' but not 'factorial(N)'

    n_pruned = learner.prune_cold_patterns(min_hits=1, min_frequency=1)
    assert n_pruned == 1
    assert "factorial(N)" not in learner._patterns
    assert "N * O" in learner._patterns


def test_prune_respects_high_frequency(tmp_db):
    """A pattern learned many times (even if never fired) should survive
    pruning when min_frequency is lower than its frequency."""
    learner = AutoLearner(db_path=tmp_db)
    # Simulate the same error being seen 5 times.
    for _ in range(5):
        learner.learn_from_correction(_FakeClaim(expression="factorial(7)"))
    pat = learner._patterns["factorial(N)"]
    assert pat.frequency == 5
    assert pat.hits == 0

    # Default prune: min_frequency=1 means "frequency > 1 is safe" →
    # frequency 5 > 1 → survives.
    n_pruned = learner.prune_cold_patterns(min_hits=1, min_frequency=1)
    assert n_pruned == 0
    assert "factorial(N)" in learner._patterns


def test_stats_reports_hits(tmp_db):
    """stats() should surface total_hits and cold_patterns counts."""
    learner = AutoLearner(db_path=tmp_db)
    learner.learn_from_correction(_FakeClaim(expression="17 * 23"))
    learner.learn_from_correction(_FakeClaim(expression="factorial(7)"))
    learner.suggest_precomputes("what is 5 * 7?")

    s = learner.stats()
    assert s["total"] == 2
    assert s["total_hits"] == 1
    assert s["cold_patterns"] == 1  # factorial never fired


def test_no_regression_in_learned_pattern_db(tmp_db):
    """Writing patterns to disk should not corrupt existing JSON, and the
    hits field should be persisted."""
    learner = AutoLearner(db_path=tmp_db)
    learner.learn_from_correction(_FakeClaim(expression="17 * 23"))
    learner.learn_from_correction(_FakeClaim(expression="is_prime(391)"))

    # Verify on-disk format (now includes hits field).
    with open(tmp_db) as f:
        lines = [json.loads(ln) for ln in f if ln.strip()]
    assert len(lines) == 2
    for entry in lines:
        assert {"pattern_type", "expression", "frequency", "hits"} <= set(entry.keys())


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "patterns.jsonl"
        test_learn_generalizes_multiplication(db)
        print("[ok] generalize multiplication")
        db.unlink()
        test_loop_closes_for_arithmetic(db)
        print("[ok] loop closes for arithmetic")
        db.unlink()
        test_loop_closes_for_function_call(db)
        print("[ok] loop closes for function call")
        db.unlink()
        test_persistence_roundtrip(db)
        print("[ok] persistence roundtrip")
    print("\nall passed")
