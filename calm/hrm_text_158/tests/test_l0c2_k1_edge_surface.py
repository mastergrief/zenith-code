"""F.4d-edge — L0c2-K1-edge held-generalization micro-slice surface tests.

The edge rung is a SEPARATE dense 65-row same-template surface (identity +
small-m plus), NOT a filter of the L0c2 pool. It pins the 4 persistent K1 held
edges audit-only (no held->train laundering) + 9 fresh held rows, with a fixed
52 train / 13 held split (only WHICH rows are fresh-held varies by seed). No
model / no GPU. (codex_2 design 1779728324177 + co-lead finite-train amendment;
claude-takeover implementation per gabe 2026-05-25.)
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from collections import Counter  # noqa: E402

from calm.hrm_text_158.curriculum.generators import (  # noqa: E402
    L0C2K1_EDGE_FRESH_HELD_COUNT,
    L0C2K1_EDGE_HELD_COUNT,
    L0C2K1_EDGE_LEGACY_COUNT,
    L0C2K1_EDGE_LEGACY_ROWS,
    L0C2K1_EDGE_TOTAL_COUNT,
    L0C2K1_EDGE_TRAIN_COUNT,
    RUNG_NAMES,
    _RUNG_SPEC,
    _enumerate_partition_l0c2k1_edge,
    _l0c2k1_edge_enumerate,
    make_rung_examples,
)
from calm.hrm_text_158.curriculum.language_supports import (  # noqa: E402
    L0C2K1_EDGE_AUDIT_EXPECTED_COUNT,
    build_l0c2k1_edge_support,
    language_source_rung_buckets,
)
from calm.hrm_text_158.curriculum.replay import DIAGNOSIS_ONLY_RUNGS  # noqa: E402

_SEEDS = (17, 42, 99)


def _qe(rows):
    return {(r["question"], r["expected"]) for r in rows}


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def test_edge_rung_registered_and_diagnosis_only():
    assert "L0c2-K1-edge" in RUNG_NAMES
    assert "L0c2-K1-edge" in _RUNG_SPEC
    # Acquisition target: must be excluded from positional replay derivation.
    assert "L0c2-K1-edge" in DIAGNOSIS_ONLY_RUNGS


def test_train_script_argparse_choices_in_sync_with_rung_names():
    """Guard: a new rung must be registered in BOTH RUNG_NAMES and the train
    script's --curriculum-rung argparse choices=. The F.4d-edge launch caught
    this drift the hard way — the rung was in RUNG_NAMES but missing from the
    hardcoded choices=, so argparse rejected it (zero GPU wasted, but a failed
    launch). R7 (GSM8k) is intentionally excluded from choices (served
    separately, not via make_rung_examples)."""
    import re
    train_src = os.path.join(_REPO, "scripts", "train_hrm_text_158.py")
    with open(train_src, "r", encoding="utf-8") as fh:
        src = fh.read()
    m = re.search(r'"--curriculum-rung"[^)]*?choices=\[([^\]]*)\]', src, re.DOTALL)
    assert m, "could not locate --curriculum-rung choices= in train script"
    choices = set(re.findall(r'"([^"]+)"', m.group(1)))
    expected = set(RUNG_NAMES) - {"R7"}
    assert choices == expected, (
        "train --curriculum-rung choices out of sync with RUNG_NAMES; "
        f"missing_from_choices={sorted(expected - choices)} "
        f"extra_in_choices={sorted(choices - expected)}"
    )
    assert "L0c2-K1-edge" in choices


# --------------------------------------------------------------------------- #
# Finite enumeration shape
# --------------------------------------------------------------------------- #


def test_enumeration_is_65_rows_all_in_band():
    rows = _l0c2k1_edge_enumerate()
    assert len(rows) == L0C2K1_EDGE_TOTAL_COUNT == 65
    assert all(10 <= r["expected"] <= 19 for r in rows)
    # 10 identity + 55 triangular plus = 65.
    assert sum(1 for r in rows if r["stratum"] == "identity") == 10
    assert sum(1 for r in rows if r["stratum"] != "identity") == 55
    # No duplicate (question, expected).
    assert len(_qe(rows)) == 65


def test_question_format_byte_identical_to_l0c_surface():
    rows = _l0c2k1_edge_enumerate()
    for r in rows:
        assert r["question"].endswith(" equals what?")
    qmap = {r["question"]: r["expected"] for r in rows}
    # Identity + plus templates render exactly like the `<expr> equals what?`
    # audit surface.
    assert qmap["17 equals what?"] == 17
    assert qmap["13 plus 0 equals what?"] == 13
    assert qmap["14 plus 1 equals what?"] == 15
    assert qmap["16 plus 1 equals what?"] == 17


# --------------------------------------------------------------------------- #
# Partition: 52 train / 13 held, legacy pinned audit-only, 9 fresh
# --------------------------------------------------------------------------- #


def test_partition_counts_and_disjoint_all_seeds():
    for seed in _SEEDS:
        train, held = _enumerate_partition_l0c2k1_edge(seed)
        assert len(train) == L0C2K1_EDGE_TRAIN_COUNT == 52
        assert len(held) == L0C2K1_EDGE_HELD_COUNT == 13
        assert _qe(train).isdisjoint(_qe(held))
        assert _qe(train) | _qe(held) == _qe(_l0c2k1_edge_enumerate())
        assert all(10 <= e <= 19 for (_q, e) in _qe(train) | _qe(held))


def test_legacy_rows_are_held_never_train_all_seeds():
    legacy_qe = set(L0C2K1_EDGE_LEGACY_ROWS)
    assert len(legacy_qe) == L0C2K1_EDGE_LEGACY_COUNT == 4
    for seed in _SEEDS:
        train, held = _enumerate_partition_l0c2k1_edge(seed)
        held_qe = _qe(held)
        train_qe = _qe(train)
        assert legacy_qe <= held_qe, "a legacy edge leaked out of held"
        assert legacy_qe.isdisjoint(train_qe), "a legacy edge laundered into train"
        legacy_in_held = [r for r in held if r["hold_kind"] == "legacy"]
        assert len(legacy_in_held) == 4
        assert {(r["question"], r["expected"]) for r in legacy_in_held} == legacy_qe


def test_fresh_held_count_and_kind_all_seeds():
    for seed in _SEEDS:
        _train, held = _enumerate_partition_l0c2k1_edge(seed)
        fresh = [r for r in held if r["hold_kind"] == "fresh"]
        legacy = [r for r in held if r["hold_kind"] == "legacy"]
        assert len(fresh) == L0C2K1_EDGE_FRESH_HELD_COUNT == 9
        assert len(legacy) == L0C2K1_EDGE_LEGACY_COUNT == 4
        # Fresh held are NOT legacy rows.
        assert _qe(fresh).isdisjoint(set(L0C2K1_EDGE_LEGACY_ROWS))


def test_every_stratum_keeps_train_coverage():
    # m=9 singleton must stay in train; no m (0..9) or identity orphaned.
    for seed in _SEEDS:
        train, _held = _enumerate_partition_l0c2k1_edge(seed)
        train_m = {r["m"] for r in train}
        assert train_m == set(range(0, 10)) | {None}


def test_partition_deterministic_per_seed():
    a = _enumerate_partition_l0c2k1_edge(17)
    b = _enumerate_partition_l0c2k1_edge(17)
    assert _qe(a[0]) == _qe(b[0])
    assert _qe(a[1]) == _qe(b[1])


def test_different_seeds_shift_fresh_held_not_counts():
    # Counts fixed; fresh-held identity may differ across seeds (so the gate
    # measures generalization, not a fixed memorized hold set).
    _t17, h17 = _enumerate_partition_l0c2k1_edge(17)
    _t42, h42 = _enumerate_partition_l0c2k1_edge(42)
    assert len(h17) == len(h42) == 13
    fresh17 = {(r["question"], r["expected"]) for r in h17 if r["hold_kind"] == "fresh"}
    fresh42 = {(r["question"], r["expected"]) for r in h42 if r["hold_kind"] == "fresh"}
    assert len(fresh17) == len(fresh42) == 9
    # Legacy held are seed-invariant.
    leg17 = {(r["question"], r["expected"]) for r in h17 if r["hold_kind"] == "legacy"}
    assert leg17 == set(L0C2K1_EDGE_LEGACY_ROWS)


# --------------------------------------------------------------------------- #
# make_rung_examples sampling
# --------------------------------------------------------------------------- #


def test_make_rung_examples_train_samples_only_52_unique():
    train, _held = _enumerate_partition_l0c2k1_edge(17)
    train_qe = _qe(train)
    rows = make_rung_examples("L0c2-K1-edge", 400, seed=17, split="train")
    assert len(rows) == 400
    assert all(r["rung"] == "L0c2-K1-edge" for r in rows)
    sampled = {(r["question"], r["expected"]) for r in rows}
    # With-replacement sampling draws ONLY from the 52 train rows; no held row.
    assert sampled <= train_qe
    held_qe = _qe(_held)
    assert sampled.isdisjoint(held_qe)


def test_make_rung_examples_held_samples_only_13_unique():
    _train, held = _enumerate_partition_l0c2k1_edge(17)
    held_qe = _qe(held)
    rows = make_rung_examples("L0c2-K1-edge", 120, seed=17, split="held_out")
    sampled = {(r["question"], r["expected"]) for r in rows}
    assert sampled <= held_qe


# --------------------------------------------------------------------------- #
# Audit surfaces (finite, per-surface)
# --------------------------------------------------------------------------- #


def test_audit_builder_two_surfaces_with_counts():
    support = build_l0c2k1_edge_support(17)
    assert set(support.keys()) == {"L0c2-K1-edge-train", "L0c2-K1-edge-held"}
    assert len(support["L0c2-K1-edge-train"]) == 52
    assert len(support["L0c2-K1-edge-held"]) == 13
    assert L0C2K1_EDGE_AUDIT_EXPECTED_COUNT == 65 == 52 + 13


def test_audit_held_bucket_axis_legacy_fresh():
    support = build_l0c2k1_edge_support(17)
    held = support["L0c2-K1-edge-held"]
    buckets = Counter(b for (_q, _e, b) in held)
    declared = set(language_source_rung_buckets("L0c2-K1-edge-held"))
    assert set(buckets) <= declared
    assert buckets["legacy"] == 4
    assert buckets["fresh"] == 9
    # All held rows render in the `<expr> equals what?` surface.
    for q, e, _b in held:
        assert q.endswith(" equals what?")
        assert 10 <= e <= 19


def test_audit_train_bucket_axis_strata():
    support = build_l0c2k1_edge_support(17)
    train = support["L0c2-K1-edge-train"]
    declared = set(language_source_rung_buckets("L0c2-K1-edge-train"))
    present = {b for (_q, _e, b) in train}
    assert present <= declared
    assert len(train) == 52
    for q, e, _b in train:
        assert q.endswith(" equals what?")
        assert 10 <= e <= 19


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"  {_name}: PASS")
    print("L0c2-K1-edge surface tests: PASS")
