"""Tests for R9 programmatic rare-class synthesis."""
import pytest

from calm.hrm.code_dt_data import CodeProblem
from calm.hrm.rare_class_synth import (
    canonical_arg,
    infer_semantic,
    parse_arg_names,
    synthesize_rare_class_pairs,
    _generate_for_skeleton,
)


def test_infer_semantic_int_names():
    assert infer_semantic("n") == "int"
    assert infer_semantic("N") == "int"
    assert infer_semantic("num") == "int"
    assert infer_semantic("limit") == "int"


def test_infer_semantic_string_names():
    assert infer_semantic("s") == "string"
    assert infer_semantic("text") == "string"


def test_infer_semantic_list_names():
    assert infer_semantic("xs") == "list"
    assert infer_semantic("nums") == "list"
    assert infer_semantic("arr") == "list"


def test_infer_semantic_with_annotation():
    assert infer_semantic("n: int") == "int"
    assert infer_semantic("text: str") == "string"
    assert infer_semantic("l: list") == "list"


def test_infer_semantic_unknown():
    assert infer_semantic("xyz_unknown_arg_name") == "generic"


def test_canonical_arg_strips_annotation():
    assert canonical_arg("n: int") == "n"
    assert canonical_arg("n: int = 10") == "n"
    assert canonical_arg("*args") == "args"


def test_parse_arg_names_empty():
    assert parse_arg_names("def FN():") == []


def test_parse_arg_names_single():
    assert parse_arg_names("def FN(n):") == ["n"]


def test_parse_arg_names_multi():
    assert parse_arg_names("def FN(a, b):") == ["a", "b"]
    assert parse_arg_names("def FN(a,b):") == ["a", "b"]


def test_generate_for_skeleton_single_arg_int():
    import random
    out = _generate_for_skeleton("def FN(n):", n=10, rng=random.Random(42))
    assert len(out) == 10
    # All have target skeleton
    assert all(p.expression == "def FN(n):" for p in out)
    # All prompts mention 'n' (copy target)
    assert all("n" in p.question.lower() for p in out)


def test_generate_for_skeleton_two_arg_pair():
    import random
    out = _generate_for_skeleton("def FN(a, b):", n=8, rng=random.Random(42))
    assert len(out) == 8
    assert all(p.expression == "def FN(a, b):" for p in out)


def test_synthesize_rare_class_pairs_targets_rare():
    """Rare classes in [3, 20] range get synthesized; common ones skipped."""
    pairs = (
        # Common: 100 pairs for FN(n)
        [CodeProblem(question="common", expression="def FN(n):")] * 100 +
        # Rare: 4 pairs for FN(s) — should get synthesized
        [CodeProblem(question="rare", expression="def FN(s):")] * 4 +
        # Rare: 3 pairs for FN(xs)
        [CodeProblem(question="rare2", expression="def FN(xs):")] * 3
    )
    synthetic = synthesize_rare_class_pairs(
        pairs, min_count=3, max_count=20, target_per_class=20, seed=42,
    )
    # Only FN(s) and FN(xs) are rare; FN(n) is too common (100)
    synthetic_classes = {p.expression for p in synthetic}
    assert synthetic_classes == {"def FN(s):", "def FN(xs):"}
    assert len(synthetic) == 40  # 2 classes × 20 each


def test_synthesize_rare_skips_too_common():
    pairs = [CodeProblem(question="q", expression="def FN(n):")] * 50
    synthetic = synthesize_rare_class_pairs(
        pairs, min_count=3, max_count=20, target_per_class=20,
    )
    # FN(n) has 50 > max_count=20 → skipped
    assert synthetic == []


def test_synthesize_rare_skips_too_rare():
    """Below min_count — typos / one-offs should not be synthesized."""
    pairs = [CodeProblem(question="q", expression="def FN(weird_arg):")] * 2
    synthetic = synthesize_rare_class_pairs(
        pairs, min_count=3, max_count=20, target_per_class=20,
    )
    assert synthetic == []


def test_synthesize_deterministic_with_seed():
    pairs = [CodeProblem(question="q", expression="def FN(n):")] * 5
    s1 = synthesize_rare_class_pairs(pairs, target_per_class=10, seed=42)
    s2 = synthesize_rare_class_pairs(pairs, target_per_class=10, seed=42)
    assert [p.question for p in s1] == [p.question for p in s2]


def test_synthesize_different_seeds_differ():
    pairs = [CodeProblem(question="q", expression="def FN(n):")] * 5
    s1 = synthesize_rare_class_pairs(pairs, target_per_class=10, seed=1)
    s2 = synthesize_rare_class_pairs(pairs, target_per_class=10, seed=2)
    assert [p.question for p in s1] != [p.question for p in s2]


def test_synthesize_skips_zero_arg():
    """0-arg has trivial output (only FN() exists). Not synthesized."""
    pairs = [CodeProblem(question="q", expression="def FN():")] * 5
    synthetic = synthesize_rare_class_pairs(
        pairs, min_count=3, max_count=20, target_per_class=10,
    )
    assert synthetic == []


def test_synthesize_three_arg_number_triple():
    """R11: 3-arg with templates DOES synthesize."""
    pairs = [CodeProblem(question="q", expression="def FN(a, b, c):")] * 5
    synthetic = synthesize_rare_class_pairs(
        pairs, min_count=3, max_count=20, target_per_class=10,
    )
    # Number triple templates exist → should synthesize
    assert len(synthetic) == 10
    assert all(p.expression == "def FN(a, b, c):" for p in synthetic)
    # All prompts mention a, b, and c
    for p in synthetic:
        assert "a" in p.question and "b" in p.question and "c" in p.question


def test_synthesize_four_arg_skipped():
    """4+ args still skipped (no templates yet)."""
    pairs = [CodeProblem(question="q", expression="def FN(a, b, c, d):")] * 5
    synthetic = synthesize_rare_class_pairs(
        pairs, min_count=3, max_count=20, target_per_class=10,
    )
    assert synthetic == []
