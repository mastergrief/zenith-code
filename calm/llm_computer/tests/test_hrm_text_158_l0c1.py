"""F.1 — L0c1 one_digit-stratum precursor rung scaffolding tests.

Codex msg 1779636434289-de29e525 +1 Slice F.1 implement (gabe greenlit
"ok lets go for the next slice then", relay 1779636434289). L0c1 is the
one_digit-stratum precursor SUBSET of L0c (121 rows = L0c's one_digit
stratum). These tests assert codex's acceptance gate:
  - L0c1 support count = 121
  - canonical language aggregate still 690 (L0c1 NOT in it)
  - L0c still full 230
  - train/held split deterministic (115/6)
  - L0c1 ⊂ L0c by construction
  - L0c1 is DIAGNOSIS_ONLY (subset, not a disjoint axis → excluded from
    positional priors; L0c's priors stay unchanged)
The probe's separate `surface='l0c1'` JSON is validated via a live probe
run in the slice receipt (real product path), not here.
"""
from __future__ import annotations

import pytest

from calm.hrm_text_158.curriculum.generators import (
    RUNG_NAMES,
    _enumerate_partition_l0c,
    _enumerate_partition_l0c1,
    make_rung_examples,
)
from calm.hrm_text_158.curriculum.language_supports import (
    LANGUAGE_ACTIVE_RUNGS,
    LANGUAGE_EXPECTED_AGGREGATE,
    LANGUAGE_EXPECTED_COUNTS,
    L0C1_EXPECTED_COUNT,
    _l0c1_support,
    _l0c_support,
    build_l0c1_support,
    language_source_rung_buckets,
)
from calm.hrm_text_158.curriculum.replay import (
    DIAGNOSIS_ONLY_RUNGS,
    _resolve_prior_rungs,
)


# --- position / RUNG_NAMES -------------------------------------------------


def test_l0c1_in_rung_names_index_16() -> None:
    """L0c1 inserted at index 16, between L0b (15) and L0c (now 17)."""
    assert RUNG_NAMES[16] == "L0c1", f"L0c1 must be at index 16; got {RUNG_NAMES}"
    assert RUNG_NAMES[15] == "L0b", f"L0b must stay at index 15; got {RUNG_NAMES}"
    assert RUNG_NAMES[17] == "L0c", f"L0c must shift to index 17; got {RUNG_NAMES}"


# --- count / split ---------------------------------------------------------


def test_l0c1_support_count_121() -> None:
    assert L0C1_EXPECTED_COUNT == 121
    assert len(_l0c1_support(seed=42)) == 121


def test_l0c1_partition_115_train_6_held() -> None:
    train, held = _enumerate_partition_l0c1(seed=42)
    assert len(train) == 115, f"L0c1 train must be 115; got {len(train)}"
    assert len(held) == 6, f"L0c1 held must be 6; got {len(held)}"
    assert len(train) + len(held) == 121


def test_l0c1_split_deterministic() -> None:
    a = _enumerate_partition_l0c1(seed=42)
    b = _enumerate_partition_l0c1(seed=42)
    assert a == b
    # support builder deterministic too
    assert _l0c1_support(seed=42) == _l0c1_support(seed=42)


def test_l0c1_changes_with_seed() -> None:
    # The R1 identity-bridge rows are seed-sampled; a different seed must
    # change the L0c1 set (R0/R1b one_digit strata are exhaustive/stable,
    # but the bridge rows differ).
    assert _l0c1_support(seed=42) != _l0c1_support(seed=43)


# --- subset / stratum ------------------------------------------------------


def test_l0c1_is_subset_of_l0c() -> None:
    l0c1 = set(_l0c1_support(seed=42))
    l0c = set(_l0c_support(seed=42))
    assert len(l0c1) == 121
    assert l0c1 <= l0c, "L0c1 must be a strict subset of L0c"


def test_l0c1_equals_l0c_one_digit_stratum() -> None:
    """L0c1 is exactly L0c's one_digit-stratum rows (train+held)."""
    l0c_train, l0c_held = _enumerate_partition_l0c(seed=42)
    one_digit = [
        (r["question"], r["expected"], r["source_rung"])
        for r in l0c_train + l0c_held
        if r["stratum"] == "one_digit"
    ]
    assert len(one_digit) == 121
    assert set(_l0c1_support(seed=42)) == set(one_digit)


def test_l0c_stratum_partition_121_109() -> None:
    """L0c rows split 121 one_digit + 109 two_digit = 230."""
    train, held = _enumerate_partition_l0c(seed=42)
    rows = train + held
    n_one = sum(1 for r in rows if r["stratum"] == "one_digit")
    n_two = sum(1 for r in rows if r["stratum"] == "two_digit")
    assert (n_one, n_two) == (121, 109)
    assert len(rows) == 230


# --- canonical language aggregate untouched --------------------------------


def test_l0c1_not_in_active_language_aggregate() -> None:
    assert LANGUAGE_ACTIVE_RUNGS == ("L0a", "L0b", "L0c"), (
        f"L0c1 must NOT be an active language rung; got {LANGUAGE_ACTIVE_RUNGS}"
    )
    assert "L0c1" not in LANGUAGE_ACTIVE_RUNGS
    assert "L0c1" not in LANGUAGE_EXPECTED_COUNTS
    assert LANGUAGE_EXPECTED_AGGREGATE == 690, (
        f"canonical language aggregate must stay 690; got {LANGUAGE_EXPECTED_AGGREGATE}"
    )


def test_l0c_still_full_230() -> None:
    assert len(_l0c_support(seed=42)) == 230
    assert LANGUAGE_EXPECTED_COUNTS["L0c"] == 230


# --- replay classification -------------------------------------------------


def test_l0c1_in_diagnosis_only() -> None:
    """L0c1 ⊂ L0c (not a disjoint axis) → excluded from positional priors."""
    assert "L0c1" in DIAGNOSIS_ONLY_RUNGS


def test_l0c1_positional_priors_are_math_plus_l0a_l0b() -> None:
    priors = _resolve_prior_rungs("L0c1", None)
    assert priors == [
        "R0", "R1", "R1b1", "R1b2", "R1b3", "R1b4v2",
        "R1b5", "R1b6", "R1b7", "R1b8", "R1b9", "L0a", "L0b",
    ], f"L0c1 positional priors unexpected: {priors}"


def test_l0c_priors_unchanged_exclude_l0c1() -> None:
    """Inserting L0c1 before L0c must NOT change L0c's resolved priors
    (L0c1 is diagnosis-only, so it is filtered out)."""
    priors = _resolve_prior_rungs("L0c", None)
    assert "L0c1" not in priors, f"L0c must not positionally replay L0c1; got {priors}"
    assert priors == [
        "R0", "R1", "R1b1", "R1b2", "R1b3", "R1b4v2",
        "R1b5", "R1b6", "R1b7", "R1b8", "R1b9", "L0a", "L0b",
    ], f"L0c positional priors changed: {priors}"


# --- generation surface ----------------------------------------------------


def test_l0c1_make_rung_examples() -> None:
    rows = make_rung_examples("L0c1", n=60, seed=42, split="train")
    assert len(rows) == 60
    pool = {(q, e) for q, e, _s in _l0c1_support(seed=42)}
    for ex in rows:
        assert ex["rung"] == "L0c1"
        assert ex["question"].endswith("equals what?"), ex["question"]
        assert (ex["question"], ex["expected"]) in pool


def test_l0c1_buckets_helper_returns_13_source_buckets() -> None:
    buckets = language_source_rung_buckets("L0c1")
    assert buckets == [
        "R0",
        "R1_plus_0", "R1_0_plus_A", "R1_minus_0",
        "R1b1", "R1b2", "R1b3", "R1b4v2",
        "R1b5", "R1b6", "R1b7", "R1b8", "R1b9",
    ]


def test_build_l0c1_support_single_key_shape() -> None:
    surf = build_l0c1_support(seed=42)
    assert set(surf.keys()) == {"L0c1"}
    assert len(surf["L0c1"]) == 121
