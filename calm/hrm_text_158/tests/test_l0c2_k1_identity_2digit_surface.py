"""L0c2-K1-identity-2digit finite identity precursor surface tests.

The rung is a 90-row identity-only suffix-copy probe over
`<n> equals what?`, n=10..99. It has no arithmetic rows, pins the legacy
11/17 identity misses held-only, and exposes finite train/held audit surfaces.
"""
import os
import re
import sys
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from calm.hrm_text_158.curriculum.generators import (  # noqa: E402
    L0C2K1_IDENTITY_FRESH_HELD_COUNT,
    L0C2K1_IDENTITY_HELD_COUNT,
    L0C2K1_IDENTITY_LEGACY_COUNT,
    L0C2K1_IDENTITY_LEGACY_ROWS,
    L0C2K1_IDENTITY_TEEN_HELD_COUNT,
    L0C2K1_IDENTITY_TEEN_TRAIN_COUNT,
    L0C2K1_IDENTITY_TOTAL_COUNT,
    L0C2K1_IDENTITY_TRAIN_COUNT,
    RUNG_NAMES,
    _RUNG_SPEC,
    _enumerate_partition_l0c2k1_identity,
    _l0c2k1_identity_enumerate,
    make_rung_examples,
)
from calm.hrm_text_158.curriculum.language_supports import (  # noqa: E402
    L0C2K1_IDENTITY_AUDIT_EXPECTED_COUNT,
    build_l0c1_support,
    build_l0c2k1_identity_support,
    language_source_rung_buckets,
)
from calm.hrm_text_158.curriculum.replay import DIAGNOSIS_ONLY_RUNGS  # noqa: E402

_SEEDS = (17, 42, 99)
_EXPLICIT_REPLAY_RUNGS = (
    "R0", "R1", "R1b1", "R1b2", "R1b3", "R1b4v2", "R1b5",
    "R1b6", "R1b7", "R1b8", "R1b9", "L0a", "L0b", "L0c1",
)


def _qe(rows):
    return {(r["question"], r["expected"]) for r in rows}


def test_identity_rung_registered_and_diagnosis_only():
    assert "L0c2-K1-identity-2digit" in RUNG_NAMES
    assert "L0c2-K1-identity-2digit" in _RUNG_SPEC
    assert "L0c2-K1-identity-2digit" in DIAGNOSIS_ONLY_RUNGS


def test_train_script_argparse_choices_include_identity_rung():
    train_src = os.path.join(_REPO, "scripts", "train_hrm_text_158.py")
    with open(train_src, "r", encoding="utf-8") as fh:
        src = fh.read()
    m = re.search(r'"--curriculum-rung"[^)]*?choices=\[([^\]]*)\]', src, re.DOTALL)
    assert m, "could not locate --curriculum-rung choices= in train script"
    choices = set(re.findall(r'"([^"]+)"', m.group(1)))
    assert choices == set(RUNG_NAMES) - {"R7"}
    assert "L0c2-K1-identity-2digit" in choices


def test_enumeration_is_90_identity_only_rows():
    rows = _l0c2k1_identity_enumerate()
    assert len(rows) == L0C2K1_IDENTITY_TOTAL_COUNT == 90
    assert len(_qe(rows)) == 90
    assert {r["tens"] for r in rows} == set(range(1, 10))
    assert all(10 <= r["n"] <= 99 for r in rows)
    assert all(r["expected"] == r["n"] for r in rows)
    assert all(r["question"] == f"{r['n']} equals what?" for r in rows)
    assert all(" plus " not in r["question"] and " minus " not in r["question"] for r in rows)
    buckets = Counter(r["tens"] for r in rows)
    assert all(buckets[tens] == 10 for tens in range(1, 10))


def test_partition_counts_and_disjoint_all_seeds():
    for seed in _SEEDS:
        train, held = _enumerate_partition_l0c2k1_identity(seed)
        assert len(train) == L0C2K1_IDENTITY_TRAIN_COUNT == 70
        assert len(held) == L0C2K1_IDENTITY_HELD_COUNT == 20
        assert _qe(train).isdisjoint(_qe(held))
        assert _qe(train) | _qe(held) == _qe(_l0c2k1_identity_enumerate())


def test_legacy_11_17_are_held_never_train_all_seeds():
    legacy_qe = set(L0C2K1_IDENTITY_LEGACY_ROWS)
    assert legacy_qe == {("11 equals what?", 11), ("17 equals what?", 17)}
    for seed in _SEEDS:
        train, held = _enumerate_partition_l0c2k1_identity(seed)
        assert legacy_qe <= _qe(held)
        assert legacy_qe.isdisjoint(_qe(train))
        legacy_rows = [r for r in held if r["hold_kind"] == "legacy"]
        assert len(legacy_rows) == L0C2K1_IDENTITY_LEGACY_COUNT == 2
        assert _qe(legacy_rows) == legacy_qe


def test_teen_and_nonteen_bucket_counts_all_seeds():
    for seed in _SEEDS:
        train, held = _enumerate_partition_l0c2k1_identity(seed)
        assert sum(1 for r in train if r["tens"] == 1) == L0C2K1_IDENTITY_TEEN_TRAIN_COUNT == 6
        assert sum(1 for r in held if r["tens"] == 1) == L0C2K1_IDENTITY_TEEN_HELD_COUNT == 4
        assert sum(1 for r in held if r["hold_kind"] == "fresh") == L0C2K1_IDENTITY_FRESH_HELD_COUNT == 18
        for tens in range(2, 10):
            assert sum(1 for r in train if r["tens"] == tens) == 8
            assert sum(1 for r in held if r["tens"] == tens) == 2


def test_make_rung_examples_samples_only_requested_split():
    train, held = _enumerate_partition_l0c2k1_identity(17)
    train_qe = _qe(train)
    held_qe = _qe(held)
    train_rows = make_rung_examples("L0c2-K1-identity-2digit", 400, seed=17, split="train")
    held_rows = make_rung_examples("L0c2-K1-identity-2digit", 120, seed=17, split="held_out")
    assert all(r["rung"] == "L0c2-K1-identity-2digit" for r in train_rows + held_rows)
    assert _qe(train_rows) <= train_qe
    assert _qe(train_rows).isdisjoint(held_qe)
    assert _qe(held_rows) <= held_qe
    assert _qe(held_rows).isdisjoint(train_qe)


def test_audit_builder_two_surfaces_with_counts_and_buckets():
    support = build_l0c2k1_identity_support(17)
    assert set(support) == {
        "L0c2-K1-identity-2digit-train",
        "L0c2-K1-identity-2digit-held",
    }
    assert len(support["L0c2-K1-identity-2digit-train"]) == 70
    assert len(support["L0c2-K1-identity-2digit-held"]) == 20
    assert L0C2K1_IDENTITY_AUDIT_EXPECTED_COUNT == 90 == 70 + 20

    held_buckets = Counter(b for (_q, _e, b) in support["L0c2-K1-identity-2digit-held"])
    declared_held = set(language_source_rung_buckets("L0c2-K1-identity-2digit-held"))
    assert set(held_buckets) <= declared_held
    assert held_buckets["held_legacy_teen"] == 2
    assert held_buckets["held_fresh_teen"] == 2
    for tens in range(2, 10):
        assert held_buckets[f"held_fresh_tens_{tens}"] == 2

    train_buckets = Counter(b for (_q, _e, b) in support["L0c2-K1-identity-2digit-train"])
    declared_train = set(language_source_rung_buckets("L0c2-K1-identity-2digit-train"))
    assert set(train_buckets) <= declared_train
    assert train_buckets["train_teen"] == 6
    for tens in range(2, 10):
        assert train_buckets[f"train_tens_{tens}"] == 8


def test_held_targets_do_not_leak_into_replay_or_retained_support_rows():
    _train, held = _enumerate_partition_l0c2k1_identity(17)
    held_targets = _qe(held)

    replay_rows = []
    for rung in _EXPLICIT_REPLAY_RUNGS:
        replay_rows.extend(make_rung_examples(rung, 800, seed=17, split="train"))
        replay_rows.extend(make_rung_examples(rung, 200, seed=17, split="held_out"))
    replay_rows.extend(
        {"question": q, "expected": e}
        for rows in build_l0c1_support(17).values()
        for (q, e, _bucket) in rows
    )
    assert held_targets.isdisjoint(_qe(replay_rows))

    from scripts.train_hrm_text_158 import _retained_support  # noqa: E402

    retained_rows = []
    for name in ("L0b", "math_a0", "math_r1b2_minus_one"):
        rows, _hash = _retained_support(name, 17)
        retained_rows.extend({"question": q, "expected": e} for q, e, _bucket in rows)
    assert held_targets.isdisjoint(_qe(retained_rows))


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"  {_name}: PASS")
    print("L0c2-K1-identity-2digit surface tests: PASS")
