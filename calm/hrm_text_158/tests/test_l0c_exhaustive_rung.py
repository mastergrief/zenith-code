"""F.3a — trainable `L0c_exhaustive` rung scaffold (codex msg 1779694585312 Q1).

Covers generator/split/rung-recognition only; the AUDIT support (count 1255,
L0c1 subset, math-density transform) lives in test_exhaustive_l0c.py and the
retained-support registry entry in test_retained_support_registry.py. The
trainable rung is DISTINCT from bounded 230-row L0c. No model / no GPU.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from calm.hrm_text_158.curriculum.generators import (  # noqa: E402
    make_rung_examples,
    _enumerate_partition_l0c_exhaustive,
    _enumerate_partition_l0c,
    RUNG_NAMES,
)

_DUP = ("0 plus 0 equals what?", 0)  # the inherited math-A0 duplicate (R1)


def _pairs(rows):
    return [(r["question"], r["expected"]) for r in rows]


# --------------------------------------------------------------------------- #
# Rung recognition
# --------------------------------------------------------------------------- #

def test_rung_registered_in_names():
    assert "L0c_exhaustive" in RUNG_NAMES


def test_make_rung_examples_recognizes_and_tags():
    rows = make_rung_examples("L0c_exhaustive", 64, seed=17, split="train")
    assert len(rows) == 64
    assert all(r["rung"] == "L0c_exhaustive" for r in rows)
    assert all(r["question"].endswith(" equals what?") for r in rows)


# --------------------------------------------------------------------------- #
# Deterministic, stratified split — invariant-safe (splits.py: train ∩ held ∅)
# --------------------------------------------------------------------------- #

def test_partition_deterministic():
    assert _enumerate_partition_l0c_exhaustive(17) == _enumerate_partition_l0c_exhaustive(17)


def test_partition_total_is_1255():
    train, held = _enumerate_partition_l0c_exhaustive(17)
    assert len(train) + len(held) == 1255


def test_train_held_no_overlap_on_qe():
    train, held = _enumerate_partition_l0c_exhaustive(17)
    assert train and held
    assert set(_pairs(train)).isdisjoint(set(_pairs(held))), \
        "train ∩ held must be empty on (question, expected)"


def test_inherited_dup_kept_atomic():
    # Both physical copies of `0 plus 0 equals what?` must land on ONE side,
    # so the split does not violate the no-overlap invariant.
    train, held = _enumerate_partition_l0c_exhaustive(17)
    n_train = _pairs(train).count(_DUP)
    n_held = _pairs(held).count(_DUP)
    assert {n_train, n_held} == {0, 2}, f"dup not atomic: train={n_train} held={n_held}"


def test_split_stratified_by_source_rung():
    train, held = _enumerate_partition_l0c_exhaustive(17)
    assert {r["source_rung"] for r in train} == {r["source_rung"] for r in held}


def test_split_seed_dependent_but_pool_stable():
    # The split (which rows are held) is seed-dependent; the union of (q,e) is
    # NOT (it's the full 1255-derived set regardless of seed).
    def _union(seed):
        tr, hl = _enumerate_partition_l0c_exhaustive(seed)
        return set(_pairs(tr)) | set(_pairs(hl))
    assert _union(17) == _union(42)
    # different seed generally reshuffles the held set
    _, h17 = _enumerate_partition_l0c_exhaustive(17)
    _, h42 = _enumerate_partition_l0c_exhaustive(42)
    assert set(_pairs(h17)) != set(_pairs(h42))


# --------------------------------------------------------------------------- #
# Regression: the new rung must not perturb bounded L0c
# --------------------------------------------------------------------------- #

def test_bounded_l0c_unchanged_230():
    train, held = _enumerate_partition_l0c(17)
    assert len(train) + len(held) == 230


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"  {_name}: PASS")
    print("L0c_exhaustive-rung tests: PASS")
