"""F.4-audit — standalone `--l0c2-audit` surface (codex msg 1779707615251 +1
with the composite-bucket adjustment).

L0c2 is wired as a SEPARATE audit surface (mirroring L0c1): deliberately NOT in
LANGUAGE_ACTIVE_RUNGS, so the canonical 690 language aggregate (L0a+L0b+L0c) is
PRESERVED. The per-bucket axis is the 12 COMPOSITE `source_rung:operator`
buckets — NOT the collapsed 11 source-rungs — so the operator-specific
`R1b2:minus` failure class (`10 minus 1 -> 9`) cannot hide (the load-bearing
correction). No model / no GPU.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from collections import Counter  # noqa: E402

from calm.hrm_text_158.curriculum.generators import (  # noqa: E402
    _l0c_is_hard,
    _enumerate_partition_l0c2,
)
from calm.hrm_text_158.curriculum.language_supports import (  # noqa: E402
    build_l0c2_support,
    _l0c2_support,
    L0C2_AUDIT_EXPECTED_COUNT,
    build_language_supports,
    LANGUAGE_ACTIVE_RUNGS,
    LANGUAGE_EXPECTED_AGGREGATE,
    language_source_rung_buckets,
    build_l0c1_support,
    L0C1_EXPECTED_COUNT,
    build_exhaustive_l0c_supports,
    L0C_EXHAUSTIVE_EXPECTED_COUNT,
)

_OPERAND_ONLY = ("10 minus 1 equals what?", 9, "R1b2:minus")  # the singleton, composite-tagged


# --------------------------------------------------------------------------- #
# Canonical 690 language aggregate PRESERVED (L0c2 is a separate surface)
# --------------------------------------------------------------------------- #

def test_canonical_690_aggregate_preserved():
    # The preservation proof: L0c2 is NOT added to LANGUAGE_ACTIVE_RUNGS, so the
    # canonical language aggregate stays L0a+L0b+L0c = 690.
    assert LANGUAGE_ACTIVE_RUNGS == ("L0a", "L0b", "L0c")
    assert "L0c2" not in LANGUAGE_ACTIVE_RUNGS
    supports = build_language_supports()
    assert set(supports.keys()) == {"L0a", "L0b", "L0c"}
    assert sum(len(v) for v in supports.values()) == 690
    assert LANGUAGE_EXPECTED_AGGREGATE == 690


# --------------------------------------------------------------------------- #
# L0c2 surface shape: single key, 230 rows, composite bucket, all hard
# --------------------------------------------------------------------------- #

def test_l0c2_support_single_key_230():
    s = build_l0c2_support()
    assert list(s.keys()) == ["L0c2"]
    assert len(s["L0c2"]) == 230
    assert L0C2_AUDIT_EXPECTED_COUNT == 230


def test_l0c2_rows_shape_and_all_hard():
    rows = _l0c2_support()
    for (q, e, bucket) in rows:
        assert q.endswith(" equals what?")
        assert _l0c_is_hard(q, e), (q, e)
        assert ":" in bucket, f"bucket not composite: {bucket!r}"


def test_l0c2_support_deterministic():
    assert _l0c2_support(42) == _l0c2_support(42)


# --------------------------------------------------------------------------- #
# 12 COMPOSITE buckets (codex 1779707615251) — operator axis preserved
# --------------------------------------------------------------------------- #

_EXPECTED_BUCKETS = [
    "R0:identity",
    "R1:minus", "R1:plus",
    "R1b1:plus", "R1b2:minus", "R1b3:plus", "R1b4v2:plus",
    "R1b5:plus", "R1b6:plus", "R1b7:plus", "R1b8:plus", "R1b9:plus",
]


def test_language_buckets_l0c2_is_12_composite():
    buckets = language_source_rung_buckets("L0c2")
    assert buckets == _EXPECTED_BUCKETS
    assert len(buckets) == 12
    assert "R1b2:minus" in buckets  # the operator-specific class must be visible


def test_buckets_match_support_no_drift_no_keyerror():
    # Every support row's bucket label must be in language_source_rung_buckets
    # ("L0c2") — else the probe's by_source[src] dict KeyErrors (line ~1393).
    # AND the bucket list must equal the labels actually present (drift guard).
    rows = _l0c2_support()
    present = {bucket for (_q, _e, bucket) in rows}
    declared = set(language_source_rung_buckets("L0c2"))
    assert present == declared, f"drift: present={present} declared={declared}"


def test_buckets_derive_from_partition():
    # The composite labels are exactly source_rung:operator from the F.4a partition.
    train, held = _enumerate_partition_l0c2(42)
    derived = {f"{r['source_rung']}:{r['operator']}" for r in train + held}
    assert derived == set(_EXPECTED_BUCKETS)


def test_per_bucket_19_20_pattern_inherited():
    # F.4a allocation: 12 buckets, ten of 19 + two of 20 = 230.
    rows = _l0c2_support()
    per_bucket = Counter(bucket for (_q, _e, bucket) in rows)
    assert set(per_bucket.keys()) == set(_EXPECTED_BUCKETS)
    counts = Counter(per_bucket.values())
    assert counts[20] == 2 and counts[19] == 10, dict(per_bucket)
    assert sum(per_bucket.values()) == 230


# --------------------------------------------------------------------------- #
# Regression class visibility: R1b2:minus carries the operand-only-hard singleton
# --------------------------------------------------------------------------- #

def test_r1b2_minus_bucket_carries_operand_only_singleton():
    rows = _l0c2_support()
    assert _OPERAND_ONLY in rows, "the `10 minus 1 -> 9` singleton must be in the L0c2 audit surface, bucket R1b2:minus"
    # And it lands in the R1b2:minus bucket (operator preserved, not collapsed).
    r1b2_minus = [(q, e) for (q, e, b) in rows if b == "R1b2:minus"]
    assert ("10 minus 1 equals what?", 9) in r1b2_minus


# --------------------------------------------------------------------------- #
# Sibling surfaces UNCHANGED
# --------------------------------------------------------------------------- #

def test_l0c1_surface_unchanged_121():
    s = build_l0c1_support()
    assert list(s.keys()) == ["L0c1"]
    assert len(s["L0c1"]) == 121
    assert L0C1_EXPECTED_COUNT == 121


def test_l0c_exhaustive_unchanged_1255():
    n = sum(len(v) for v in build_exhaustive_l0c_supports().values())
    assert n == 1255
    assert L0C_EXHAUSTIVE_EXPECTED_COUNT == 1255


def test_existing_language_buckets_unchanged():
    # L0a/L0b/L0c/L0c1 still report the 13 source-rung buckets (NOT composite).
    for rung in ("L0a", "L0b", "L0c", "L0c1"):
        b = language_source_rung_buckets(rung)
        assert b[0] == "R0" and "R1_plus_0" in b and len(b) == 13


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"  {_name}: PASS")
    print("L0c2 audit-surface tests: PASS")
