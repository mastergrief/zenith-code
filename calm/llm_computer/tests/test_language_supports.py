"""Unit tests for language-wrapper finite-support audit infrastructure
(codex msg 1779559495228-f863199b +1 implement L0a as first
language-axis rung).

Pure tests: no model inference, no ckpt load. Asserts L0a support
shape, per-source-rung bucket counts, multiplicity floor at default
recipe, parallel-aggregate independence from math A0.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from calm.hrm_text_158.curriculum.exhaustive_supports import (
    EXHAUSTIVE_ACTIVE_RUNGS,
    EXHAUSTIVE_EXPECTED_AGGREGATE,
)
from calm.hrm_text_158.curriculum.language_supports import (
    LANGUAGE_ACTIVE_RUNGS,
    LANGUAGE_EXPECTED_AGGREGATE,
    LANGUAGE_EXPECTED_COUNTS,
    build_language_supports,
    language_source_rung_buckets,
)


def test_language_active_rungs_contains_only_l0a() -> None:
    """First language-axis rung is L0a only; L0b/L0c are future slices."""
    assert LANGUAGE_ACTIVE_RUNGS == ("L0a",)


def test_language_aggregate_equals_230() -> None:
    """L0a bounded stratified support total = 230 rows
    (per codex msg 1779559495228 spec)."""
    assert LANGUAGE_EXPECTED_AGGREGATE == 230
    assert LANGUAGE_EXPECTED_COUNTS["L0a"] == 230


def test_build_language_supports_l0a_shape() -> None:
    """L0a 230 (question, expected, source_rung) triples."""
    supports = build_language_supports()
    assert "L0a" in supports
    assert len(supports["L0a"]) == 230
    for row in supports["L0a"]:
        assert isinstance(row, tuple) and len(row) == 3, (
            f"row must be (question, expected, source_rung) triple; got {row!r}"
        )
        q, exp, src = row
        assert isinstance(q, str) and q.startswith("what's ")
        assert isinstance(exp, int)
        assert isinstance(src, str)


def test_l0a_per_source_rung_counts() -> None:
    """Per-source-rung counts match the bounded stratified spec."""
    supports = build_language_supports()
    by_source: dict[str, int] = {}
    for q, exp, src in supports["L0a"]:
        by_source[src] = by_source.get(src, 0) + 1
    expected = {
        "R0": 20,
        "R1_plus_0": 10,
        "R1_0_plus_A": 10,
        "R1_minus_0": 10,
        "R1b1": 20, "R1b2": 20, "R1b3": 20, "R1b4v2": 20,
        "R1b5": 20, "R1b6": 20, "R1b7": 20, "R1b8": 20, "R1b9": 20,
    }
    assert by_source == expected, f"per-source counts mismatch: {by_source}"
    assert sum(expected.values()) == 230


def test_l0a_buckets_helper_returns_canonical_order() -> None:
    buckets = language_source_rung_buckets("L0a")
    assert buckets == [
        "R0",
        "R1_plus_0", "R1_0_plus_A", "R1_minus_0",
        "R1b1", "R1b2", "R1b3", "R1b4v2",
        "R1b5", "R1b6", "R1b7", "R1b8", "R1b9",
    ]


def test_l0a_buckets_unknown_rung_raises() -> None:
    with pytest.raises(ValueError, match="unknown language rung"):
        language_source_rung_buckets("L0b")


def test_math_a0_unchanged_at_1255() -> None:
    """Codex msg 1779559495228 invariant: math A0 export stays pure
    and stable; language supports are a PARALLEL audit surface, not
    blended into math aggregate."""
    assert EXHAUSTIVE_EXPECTED_AGGREGATE == 1255, (
        f"math A0 aggregate must stay 1255 (R0..R1b9); got {EXHAUSTIVE_EXPECTED_AGGREGATE}"
    )
    # And L0a-as-source-rungs are NOT in math active rungs.
    assert "L0a" not in EXHAUSTIVE_ACTIVE_RUNGS


def test_l0a_train_held_disjoint() -> None:
    """L0a train ∩ L0a held = ∅ (within-L0a disjoint invariant)."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0a
    train, held = _enumerate_partition_l0a(seed=42)
    train_qs = {r["question"] for r in train}
    held_qs = {r["question"] for r in held}
    overlap = train_qs & held_qs
    assert not overlap, f"L0a train ∩ held overlap: {sorted(overlap)[:5]}"


def test_l0a_partition_counts_184_train_46_held() -> None:
    """Exact partition spec: 184 train + 46 held = 230 total."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0a
    train, held = _enumerate_partition_l0a(seed=42)
    assert len(train) == 184, f"L0a train: {len(train)}"
    assert len(held) == 46, f"L0a held: {len(held)}"


def test_l0a_per_bucket_train_held_split() -> None:
    """Per-source-rung train/held split matches codex spec:
    R0 16/4, R1 sub-templates 8/2 each, R1bN each 16/4."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0a
    train, held = _enumerate_partition_l0a(seed=42)

    def count_by_source(rows: list[dict]) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            out[r["source_rung"]] = out.get(r["source_rung"], 0) + 1
        return out

    train_by = count_by_source(train)
    held_by = count_by_source(held)

    assert train_by["R0"] == 16 and held_by["R0"] == 4
    for sub in ("R1_plus_0", "R1_0_plus_A", "R1_minus_0"):
        assert train_by[sub] == 8, f"{sub} train: {train_by[sub]}"
        assert held_by[sub] == 2, f"{sub} held: {held_by[sub]}"
    for rung in ("R1b1", "R1b2", "R1b3", "R1b4v2", "R1b5", "R1b6", "R1b7", "R1b8", "R1b9"):
        assert train_by[rung] == 16, f"{rung} train: {train_by[rung]}"
        assert held_by[rung] == 4, f"{rung} held: {held_by[rung]}"


def test_l0a_one_digit_exhaustive_in_train() -> None:
    """All R0 A∈{0..9} and R1bN A∈{1..9} one_digit picks must be in
    train (codex spec: "include all one-digit ...")."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0a
    train, held = _enumerate_partition_l0a(seed=42)
    train_qs = {r["question"] for r in train}
    held_qs = {r["question"] for r in held}

    # R0 one_digit: A=0..9
    for a in range(0, 10):
        assert f"what's {a}?" in train_qs, f"R0 one_digit {a} missing from train"
        assert f"what's {a}?" not in held_qs

    # R1bN one_digit: A=1..9 for each K=1..K=8 plus rung + K=-1 minus
    r1b_ops = [(" plus 1",), (" minus 1",), (" plus 2",), (" plus 3",),
               (" plus 4",), (" plus 5",), (" plus 6",), (" plus 7",), (" plus 8",)]
    for (op,) in r1b_ops:
        for a in range(1, 10):
            q = f"what's {a}{op}?"
            assert q in train_qs, f"{q!r} missing from L0a train"
            assert q not in held_qs


def test_l0a_template_shape_whats_math() -> None:
    """Every L0a row starts with `what's ` (contraction, NOT canonical
    `what is `). Distinguishes L0a paraphrase from R0..R1b9 math rows."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0a
    train, held = _enumerate_partition_l0a(seed=42)
    for r in train + held:
        q = r["question"]
        assert q.startswith("what's "), f"L0a row must start with `what's `: {q!r}"
        assert not q.startswith("what is "), f"L0a row must NOT start with `what is `: {q!r}"


def test_l0a_math_semantics_preserved() -> None:
    """Expected values match parent R0..R1b9 math semantics exactly.
    L0a does NOT introduce new math operations or values."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0a
    train, held = _enumerate_partition_l0a(seed=42)
    for r in train + held:
        q, exp, src = r["question"], r["expected"], r["source_rung"]
        # Parse semantics by template shape
        if src == "R0":
            # `what's N?` → N
            n = int(q[len("what's "):-1])
            assert exp == n, f"R0 row expected mismatch: {r}"
        elif src == "R1_plus_0":
            a = int(q[len("what's "):-len(" plus 0?")])
            assert exp == a
        elif src == "R1_0_plus_A":
            a = int(q[len("what's 0 plus "):-1])
            assert exp == a
        elif src == "R1_minus_0":
            a = int(q[len("what's "):-len(" minus 0?")])
            assert exp == a
        elif src.startswith("R1b"):
            # `what's A {plus,minus} K?`
            if " plus " in q:
                a_str, k_str = q[len("what's "):-1].split(" plus ")
                a, k = int(a_str), int(k_str)
                assert exp == a + k, f"{src} row expected mismatch: {r}"
            elif " minus " in q:
                a_str, k_str = q[len("what's "):-1].split(" minus ")
                a, k = int(a_str), int(k_str)
                assert exp == a - k, f"{src} row expected mismatch: {r}"


def test_l0a_multiplicity_meets_10x_floor() -> None:
    """Default recipe (n_train=10000, rr=0.65) yields n_new=3500
    against unique_train_count=184; multiplicity >= 10x floor."""
    n_train = 10000
    replay_ratio = 0.65
    n_new = int(n_train * (1.0 - replay_ratio))
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0a
    train, _ = _enumerate_partition_l0a(seed=42)
    unique_train = len(train)
    multiplicity = n_new / unique_train
    assert multiplicity >= 10.0, (
        f"L0a multiplicity {multiplicity:.2f}x below 10x floor; "
        f"n_new={n_new} unique_train={unique_train}"
    )
    # Sanity: expected ~19x
    assert 18.0 <= multiplicity <= 20.0, (
        f"L0a expected ~19x multiplicity; got {multiplicity:.2f}x"
    )


def test_l0a_partition_stable_across_pythonhashseed() -> None:
    """Deterministic partition: must produce identical train/held
    sequences across PYTHONHASHSEED values (using _stable_seed infra)."""
    snippet = (
        "from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0a; "
        "train, held = _enumerate_partition_l0a(seed=42); "
        "ts = sorted(r['question'] for r in train); "
        "hs = sorted(r['question'] for r in held); "
        "print('||'.join(ts) + '###' + '||'.join(hs))"
    )
    out1 = subprocess.check_output(
        [sys.executable, "-c", snippet],
        env={**os.environ, "PYTHONHASHSEED": "0"},
    ).decode().strip()
    out2 = subprocess.check_output(
        [sys.executable, "-c", snippet],
        env={**os.environ, "PYTHONHASHSEED": "777"},
    ).decode().strip()
    out3 = subprocess.check_output(
        [sys.executable, "-c", snippet],
        env={**os.environ, "PYTHONHASHSEED": "random"},
    ).decode().strip()
    assert out1 == out2 == out3, "L0a partition diverged across PYTHONHASHSEED"


def test_l0a_partition_changes_with_seed() -> None:
    """Different seeds produce different two_digit picks (one_digit
    exhaustive coverage is the same; two_digit sampling is seed-deterministic)."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0a
    train_a, _ = _enumerate_partition_l0a(seed=42)
    train_b, _ = _enumerate_partition_l0a(seed=17)
    qs_a = {r["question"] for r in train_a}
    qs_b = {r["question"] for r in train_b}
    # Some overlap (one_digit exhaustive), but not identical
    assert qs_a != qs_b, "L0a partition must depend on seed"
