"""Tests for the L0b hard-row guard anchor set (math_fragile_v2).

Proves aggregate counts + bucket/source labels + per-subset training repeat
(v1@3 + hard@CLI-repeat) without 5x-ing v1. No GPU / model load.
"""
import importlib.util
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from calm.hrm_text_158.curriculum.retention_anchors import (
    load_anchor_set,
    anchor_set_source_rung_buckets,
    RETENTION_ANCHOR_EXPECTED_COUNTS,
    L0B_HARDROW_V1,
    MATH_FRAGILE_V1,
    MATH_FRAGILE_V2,
)

_spec = importlib.util.spec_from_file_location(
    "_train_hrm_text_158", os.path.join(_REPO, "scripts", "train_hrm_text_158.py")
)
_thr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_thr)
_compose = _thr._compose_anchor_rows


def test_l0b_hardrow_set_shape():
    rows = load_anchor_set("l0b_hardrow_v1")
    assert len(rows) == 5 == RETENTION_ANCHOR_EXPECTED_COUNTS["l0b_hardrow_v1"]
    # All on the L0b R1b3 "calculate {A} plus 2." surface, expected = A+2,
    # distinct hardrow source bucket, unique anchor_ids.
    for r in rows:
        assert r.source_rung == "R1b3_hardrow"
        assert r.question.startswith("calculate ") and r.question.endswith(" plus 2.")
        a = int(r.question.split()[1])
        assert r.expected == a + 2
    assert {int(r.question.split()[1]) for r in rows} == {12, 13, 14, 15, 16}
    assert len({r.anchor_id for r in rows}) == 5
    # The confirmed persistent hard row is present.
    assert any(r.question == "calculate 14 plus 2." and r.expected == 16 for r in rows)


def test_math_fragile_v2_is_v1_plus_hardrows():
    v2 = load_anchor_set("math_fragile_v2")
    assert len(v2) == 26 == RETENTION_ANCHOR_EXPECTED_COUNTS["math_fragile_v2"]
    assert MATH_FRAGILE_V2 == MATH_FRAGILE_V1 + L0B_HARDROW_V1
    # v1 rows preserved unchanged inside v2.
    assert v2[: len(MATH_FRAGILE_V1)] == MATH_FRAGILE_V1
    buckets = anchor_set_source_rung_buckets("math_fragile_v2")
    assert buckets == ["R1b2", "R1_zero_left", "R1_zero_right", "R1b3_hardrow"]


def test_v2_compose_per_subset_repeat():
    # v2 with CLI repeat 5: v1 fixed @3, hard @5.
    composed = _compose("math_fragile_v2", 5)
    assert len(composed) == 21 * 3 + 5 * 5 == 88
    from collections import Counter
    c = Counter(r["anchor_id"] for r in composed)
    # Every v1 row appears exactly 3x; every hard row exactly 5x.
    for r in MATH_FRAGILE_V1:
        assert c[r.anchor_id] == 3, f"v1 row {r.anchor_id} repeat {c[r.anchor_id]} != 3"
    for r in L0B_HARDROW_V1:
        assert c[r.anchor_id] == 5, f"hard row {r.anchor_id} repeat {c[r.anchor_id]} != 5"
    # v1 coverage identical to standalone v1@3 (not weakened, not 5x'd).
    v1_in_v2 = [r for r in composed if r["source_rung"] != "R1b3_hardrow"]
    assert len(v1_in_v2) == 63


def test_v1_compose_unchanged_no_regression():
    # Existing v1 path must be byte-identical: 21 unique * 3 = 63.
    composed = _compose("math_fragile_v1", 3)
    assert len(composed) == 63
    from collections import Counter
    c = Counter(r["anchor_id"] for r in composed)
    assert all(v == 3 for v in c.values()) and len(c) == 21
    assert _compose("none", 5) == []


if __name__ == "__main__":
    test_l0b_hardrow_set_shape()
    test_math_fragile_v2_is_v1_plus_hardrows()
    test_v2_compose_per_subset_repeat()
    test_v1_compose_unchanged_no_regression()
    print("hardrow-anchor tests: PASS")
