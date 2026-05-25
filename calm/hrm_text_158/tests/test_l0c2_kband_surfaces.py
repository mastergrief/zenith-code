"""F.4d — explicit L0c2 result-magnitude band rungs and audit surfaces.

The bands are filtered partitions over the existing L0c2 hard pool. They do not
re-sample or re-split: L0c2 owns rare-first reservation, the train/held split,
and the `10 minus 1 -> 9` singleton pin. No model / no GPU.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from collections import Counter  # noqa: E402

from calm.hrm_text_158.curriculum.generators import (  # noqa: E402
    L0C2_BAND_EXPECTED_COUNTS,
    L0C2_EXPECTED_COUNT,
    RUNG_NAMES,
    _RUNG_SPEC,
    _enumerate_partition_l0c2,
    _enumerate_partition_l0c2_band,
    _enumerate_partition_l0c2k1,
    _enumerate_partition_l0c2k2,
    _enumerate_partition_l0c2k3,
    _l0c_is_hard,
    _l0c2_result_band,
    make_rung_examples,
)
from calm.hrm_text_158.curriculum.language_supports import (  # noqa: E402
    LANGUAGE_ACTIVE_RUNGS,
    LANGUAGE_EXPECTED_AGGREGATE,
    L0C2K1_AUDIT_EXPECTED_COUNT,
    L0C2K2_AUDIT_EXPECTED_COUNT,
    L0C2K3_AUDIT_EXPECTED_COUNT,
    _l0c2k1_support,
    _l0c2k2_support,
    _l0c2k3_support,
    build_l0c2k1_support,
    build_l0c2k2_support,
    build_l0c2k3_support,
    build_language_supports,
    language_source_rung_buckets,
)

_OPERAND_ONLY_QE = ("10 minus 1 equals what?", 9)


def _qe(rows):
    return {(r["question"], r["expected"]) for r in rows}


def _band_qe(seed: int, band: str):
    train, held = _enumerate_partition_l0c2_band(seed, band)  # type: ignore[arg-type]
    return _qe(train + held)


def _assert_rows_in_band(rows, band: str):
    for row in rows:
        assert _l0c_is_hard(row["question"], row["expected"]), row
        assert _l0c2_result_band(row) == band, row


# --------------------------------------------------------------------------- #
# Rung registration + training sampler recognition
# --------------------------------------------------------------------------- #


def test_kband_rungs_registered_after_l0c2_before_l0c():
    expected = ["L0c2-K1", "L0c2-K2", "L0c2-K3"]
    for rung in expected:
        assert rung in RUNG_NAMES
        assert rung in _RUNG_SPEC
    assert list(RUNG_NAMES[RUNG_NAMES.index("L0c2") + 1:RUNG_NAMES.index("L0c")]) == expected


def test_make_rung_examples_tags_and_band_membership():
    for rung, band in (("L0c2-K1", "K1"), ("L0c2-K2", "K2"), ("L0c2-K3", "K3")):
        rows = make_rung_examples(rung, 48, seed=42, split="train")
        assert len(rows) == 48
        assert all(r["rung"] == rung for r in rows)
        assert all(r["question"].endswith(" equals what?") for r in rows)
        for r in rows:
            pseudo = {"expected": r["expected"], "hard_reason": "result_2digit"}
            if (r["question"], r["expected"]) == _OPERAND_ONLY_QE:
                pseudo["hard_reason"] = "operand_2digit_result_1digit"
            assert _l0c_is_hard(r["question"], r["expected"]), r
            assert _l0c2_result_band(pseudo) == band, r


# --------------------------------------------------------------------------- #
# Seed-42 support counts + disjoint union over the existing L0c2 partition
# --------------------------------------------------------------------------- #


def test_seed42_band_counts_match_gate_contract():
    assert L0C2_BAND_EXPECTED_COUNTS == {"K1": 24, "K2": 79, "K3": 127}
    train1, held1 = _enumerate_partition_l0c2k1(42)
    train2, held2 = _enumerate_partition_l0c2k2(42)
    train3, held3 = _enumerate_partition_l0c2k3(42)
    assert len(train1) + len(held1) == 24 == L0C2K1_AUDIT_EXPECTED_COUNT
    assert len(train2) + len(held2) == 79 == L0C2K2_AUDIT_EXPECTED_COUNT
    assert len(train3) + len(held3) == 127 == L0C2K3_AUDIT_EXPECTED_COUNT


def test_bands_are_disjoint_and_union_l0c2_seed42():
    l0c2_train, l0c2_held = _enumerate_partition_l0c2(42)
    k1 = _band_qe(42, "K1")
    k2 = _band_qe(42, "K2")
    k3 = _band_qe(42, "K3")
    assert k1.isdisjoint(k2)
    assert k1.isdisjoint(k3)
    assert k2.isdisjoint(k3)
    assert k1 | k2 | k3 == _qe(l0c2_train + l0c2_held)
    assert len(k1 | k2 | k3) == L0C2_EXPECTED_COUNT


def test_band_train_held_split_preserved_and_disjoint():
    for band in ("K1", "K2", "K3"):
        train, held = _enumerate_partition_l0c2_band(42, band)  # type: ignore[arg-type]
        assert _qe(train).isdisjoint(_qe(held))
        _assert_rows_in_band(train + held, band)


def test_k1_singleton_present_and_pinned_to_train():
    train, held = _enumerate_partition_l0c2k1(42)
    assert _OPERAND_ONLY_QE in _qe(train)
    assert _OPERAND_ONLY_QE not in _qe(held)
    for r in train:
        if (r["question"], r["expected"]) == _OPERAND_ONLY_QE:
            assert r["hard_reason"] == "operand_2digit_result_1digit"
            assert _l0c2_result_band(r) == "K1"
            break
    else:
        raise AssertionError("singleton row not found in K1 train")


# --------------------------------------------------------------------------- #
# Audit support surfaces + composite bucket axis
# --------------------------------------------------------------------------- #


def test_kband_audit_builders_single_key_and_counts_seed42():
    supports = [
        (build_l0c2k1_support(), "L0c2-K1", 24),
        (build_l0c2k2_support(), "L0c2-K2", 79),
        (build_l0c2k3_support(), "L0c2-K3", 127),
    ]
    for support, key, expected in supports:
        assert list(support.keys()) == [key]
        assert len(support[key]) == expected


def test_kband_surfaces_do_not_change_canonical_language_aggregate():
    assert LANGUAGE_ACTIVE_RUNGS == ("L0a", "L0b", "L0c")
    for rung in ("L0c2-K1", "L0c2-K2", "L0c2-K3"):
        assert rung not in LANGUAGE_ACTIVE_RUNGS
    assert sum(len(v) for v in build_language_supports().values()) == 690
    assert LANGUAGE_EXPECTED_AGGREGATE == 690


def test_kband_support_rows_are_composite_and_bucket_declared():
    for rung, rows in (
        ("L0c2-K1", _l0c2k1_support()),
        ("L0c2-K2", _l0c2k2_support()),
        ("L0c2-K3", _l0c2k3_support()),
    ):
        declared = set(language_source_rung_buckets(rung))
        present = {bucket for (_q, _e, bucket) in rows}
        assert present <= declared
        assert all(":" in bucket for bucket in present)
        assert "R1b2:minus" in declared
        for q, e, bucket in rows:
            assert q.endswith(" equals what?")
            assert _l0c_is_hard(q, e), (q, e, bucket)


def test_kband_bucket_axis_keeps_full_l0c2_composite_shape():
    expected = language_source_rung_buckets("L0c2")
    for rung in ("L0c2-K1", "L0c2-K2", "L0c2-K3"):
        buckets = language_source_rung_buckets(rung)
        assert buckets == expected
        assert len(buckets) == 12


def test_seed42_k1_sparse_bucket_pattern_is_explicitly_safe():
    rows = _l0c2k1_support(42)
    per_bucket = Counter(bucket for (_q, _e, bucket) in rows)
    declared = language_source_rung_buckets("L0c2-K1")
    assert sum(per_bucket.values()) == 24
    assert set(per_bucket) <= set(declared)
    assert any(per_bucket[b] == 0 for b in declared)


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"  {_name}: PASS")
    print("L0c2 K-band surface tests: PASS")
