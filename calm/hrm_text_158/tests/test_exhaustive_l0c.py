"""Tests for the exhaustive-L0c language-density support (codex plan-gate
msg 1779692896701 / Step 1): the `<expr> equals what?` wrapper over the full
math-A0 exhaustive set (1255), derived by transforming math-A0 rows so
count / per-source counts / expected values match math A0 by construction,
and L0c1's 121 one_digit rows are a subset.

NOTE on uniqueness: math A0 itself contains exactly ONE duplicate question —
`what is 0 plus 0?` appears twice in R1 (the A=0 collision between the
`A plus 0` and `0 plus A` additive-identity sub-templates). Exhaustive L0c
mirrors math A0 faithfully for density parity, so it inherits that single
dup (`0 plus 0 equals what?` ×2): 1255 rows, 1254 unique. The test pins this
explicitly so a future math-A0 dedup re-triggers review.

Pure data assembly; no model / no GPU.
"""
import os
import sys
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from calm.hrm_text_158.curriculum.language_supports import (  # noqa: E402
    build_exhaustive_l0c_supports,
    _math_q_to_l0c,
    _l0c1_support,
    L0C_EXHAUSTIVE_EXPECTED_COUNT,
)
from calm.hrm_text_158.curriculum.exhaustive_supports import (  # noqa: E402
    build_exhaustive_supports,
)

_KNOWN_INHERITED_DUP = "0 plus 0 equals what?"


def _flat(supports):
    return [(q, e) for v in supports.values() for (q, e) in v]


# --------------------------------------------------------------------------- #
# Count / per-source / expected — match math A0 by construction
# --------------------------------------------------------------------------- #

def test_total_count_is_1255():
    l0c = build_exhaustive_l0c_supports()
    assert sum(len(v) for v in l0c.values()) == 1255 == L0C_EXHAUSTIVE_EXPECTED_COUNT


def test_per_source_counts_and_expected_match_math_a0():
    l0c = build_exhaustive_l0c_supports()
    math = build_exhaustive_supports()
    assert list(l0c.keys()) == list(math.keys()), "source rungs (keys) must match math A0"
    for rung in math:
        assert len(l0c[rung]) == len(math[rung]), rung
        # expected values identical and in order (same underlying math row).
        assert [e for _q, e in l0c[rung]] == [e for _q, e in math[rung]], rung


# --------------------------------------------------------------------------- #
# Uniqueness — exactly one inherited dup from math A0 (pinned)
# --------------------------------------------------------------------------- #

def test_unique_except_single_inherited_math_a0_dup():
    l0c = build_exhaustive_l0c_supports()
    qs = [q for q, _ in _flat(l0c)]
    dups = {q: n for q, n in Counter(qs).items() if n > 1}
    assert dups == {_KNOWN_INHERITED_DUP: 2}, f"unexpected dup set: {dups}"
    assert len(set(qs)) == 1254


def test_inherited_dup_traces_to_math_a0():
    # The dup is math A0's, not introduced by the wrapper transform.
    math_qs = [q for q, _ in _flat(build_exhaustive_supports())]
    math_dups = {q: n for q, n in Counter(math_qs).items() if n > 1}
    assert math_dups == {"what is 0 plus 0?": 2}


# --------------------------------------------------------------------------- #
# L0c1 subset + source_rung preservation
# --------------------------------------------------------------------------- #

def test_l0c1_is_subset_of_exhaustive_l0c():
    pairs = set(_flat(build_exhaustive_l0c_supports()))
    for seed in (17, 42):
        l0c1 = {(q, e) for (q, e, _sr) in _l0c1_support(seed)}
        assert len(l0c1) == 121
        assert l0c1 <= pairs, f"seed {seed}: L0c1 not a subset"


def test_source_rung_preserved_as_keys():
    assert list(build_exhaustive_l0c_supports().keys()) == list(
        build_exhaustive_supports().keys())


# --------------------------------------------------------------------------- #
# Transform helper — produces the L0c partition template form; fails loud
# --------------------------------------------------------------------------- #

def test_transform_matches_l0c_template_form():
    assert _math_q_to_l0c("what is 10 minus 1?") == "10 minus 1 equals what?"
    assert _math_q_to_l0c("what is 7?") == "7 equals what?"
    assert _math_q_to_l0c("what is 0 plus 9?") == "0 plus 9 equals what?"


def test_transform_rejects_bad_format():
    import pytest
    with pytest.raises(ValueError, match="unexpected math-A0 question format"):
        _math_q_to_l0c("10 minus 1 equals what?")  # already L0c form
    with pytest.raises(ValueError):
        _math_q_to_l0c("what is 5")  # no trailing ?


def test_known_template_rows_present():
    flat = {q for q, _ in _flat(build_exhaustive_l0c_supports())}
    for s in ("10 minus 1 equals what?", "99 equals what?",
              "1 plus 8 equals what?", "9 minus 0 equals what?"):
        assert s in flat, s


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"  {_name}: PASS")
    print("exhaustive-L0c tests: PASS")
