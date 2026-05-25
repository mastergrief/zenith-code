"""F.4a — bounded-2-digit stair-step rung `L0c2` (codex msg 1779705530223 +1,
option B after F.3e/F.3f-b continuation+rewarm of the full-1255 exhaustive
surface proved net-negative).

L0c2 is a bounded (~230), auditable HARD subset of exhaustive-L0c, stratified
equal-ish over (source_rung x operator) with secondary hard_reason coverage.
Codex correction pinned here (1779705530223): `_l0c_is_hard` is NOT uniformly
result-2-digit — the operand_2digit_result_1digit class (singleton
`10 minus 1 -> 9`) must NOT be averaged away; it is guaranteed in the support
and pinned to TRAIN as the regression class. No model / no GPU. Audit-wiring is
a SEPARATE slice (F.4-audit); this slice is generator/partition/tests only.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from collections import Counter  # noqa: E402

from calm.hrm_text_158.curriculum.generators import (  # noqa: E402
    make_rung_examples,
    _enumerate_partition_l0c2,
    _enumerate_partition_l0c,
    _enumerate_partition_l0c1,
    _l0c_operator,
    _l0c_hard_reason,
    _l0c_is_hard,
    L0C2_EXPECTED_COUNT,
    RUNG_NAMES,
    _RUNG_SPEC,
)
from calm.hrm_text_158.curriculum.language_supports import (  # noqa: E402
    build_exhaustive_l0c_supports,
)
from calm.hrm_text_158.curriculum.replay import _resolve_prior_rungs  # noqa: E402

_OPERAND_ONLY = ("10 minus 1 equals what?", 9)  # the singleton operand_2digit_result_1digit


def _qe(rows):
    return [(r["question"], r["expected"]) for r in rows]


def _pool_buckets():
    """(source_rung, operator) buckets present in the HARD exhaustive-L0c pool."""
    buckets = set()
    for source_rung, rows in build_exhaustive_l0c_supports().items():
        for (q, e) in rows:
            if _l0c_is_hard(q, e):
                buckets.add((source_rung, _l0c_operator(q)))
    return buckets


# --------------------------------------------------------------------------- #
# Registration / placement (must NOT break the F.3d-b adjacency assertion)
# --------------------------------------------------------------------------- #

def test_rung_registered_after_l0c1():
    assert "L0c2" in RUNG_NAMES
    assert "L0c2" in _RUNG_SPEC
    # Bounded sibling of L0c1: placed after L0c1, before L0c. Positional DEFAULT
    # priors for L0c2 are {R0..R1b9, L0a, L0b} — L0c1 is DIAGNOSIS_ONLY (replay.py)
    # so it is EXCLUDED from positional derivation (see the dedicated prior-set
    # tests below). L0c / L0c_exhaustive / L0c_exhaustive_2digit sit positionally
    # AFTER L0c2 (never priors). F.4b must opt L0c1 in EXPLICITLY via
    # --replay-rungs ...,L0c1 — non-redundant here because L0c1 is the one-digit
    # wrapper stratum, DISJOINT from L0c2's two-digit stratum (unlike L0c1 ⊂ L0c).
    assert RUNG_NAMES.index("L0c1") < RUNG_NAMES.index("L0c2") < RUNG_NAMES.index("L0c")
    # F.3d-b adjacency invariant still holds (relative).
    assert RUNG_NAMES.index("L0c_exhaustive_2digit") == RUNG_NAMES.index("L0c_exhaustive") + 1


def test_default_positional_priors_exclude_l0c1():
    # Codex correction (msg 1779706385596): the dry-run shows
    # prior_rungs=[R0..R1b9,L0a,L0b]; L0c1 is in DIAGNOSIS_ONLY_RUNGS so the
    # positional default for L0c2 EXCLUDES it. F.4b must not assume L0c1 replay.
    priors = _resolve_prior_rungs("L0c2", None)
    assert "L0c1" not in priors, "L0c1 is DIAGNOSIS_ONLY; excluded from positional default"
    assert set(priors) == {
        "R0", "R1", "R1b1", "R1b2", "R1b3", "R1b4v2",
        "R1b5", "R1b6", "R1b7", "R1b8", "R1b9", "L0a", "L0b",
    }, priors


def test_explicit_replay_can_opt_in_l0c1_with_warn():
    # F.4b stair-step retention: L0c1 (one-digit wrapper) is DISJOINT from L0c2
    # (two-digit), so replaying it is NON-redundant and must be opted in
    # explicitly. Explicit override accepts diagnosis-only rungs with a WARN.
    warns: list[str] = []
    explicit = "R0,R1,R1b1,R1b2,R1b3,R1b4v2,R1b5,R1b6,R1b7,R1b8,R1b9,L0a,L0b,L0c1"
    priors = _resolve_prior_rungs("L0c2", explicit, warn_callback=warns.append)
    assert "L0c1" in priors
    assert any("L0c1" in w for w in warns), warns


def test_make_rung_recognizes_and_tags():
    rows = make_rung_examples("L0c2", 64, seed=17, split="train")
    assert len(rows) == 64
    assert all(r["rung"] == "L0c2" for r in rows)
    assert all(r["question"].endswith(" equals what?") for r in rows)
    assert all(_l0c_is_hard(r["question"], r["expected"]) for r in rows)


# --------------------------------------------------------------------------- #
# Classifiers (operator + hard_reason) — unit cases + consistency
# --------------------------------------------------------------------------- #

def test_operator_classifier():
    assert _l0c_operator("5 plus 7 equals what?") == "plus"
    assert _l0c_operator("11 minus 5 equals what?") == "minus"
    assert _l0c_operator("0 plus 42 equals what?") == "plus"
    assert _l0c_operator("42 equals what?") == "identity"
    assert _l0c_operator("13 minus 0 equals what?") == "minus"


def test_hard_reason_classifier_cases():
    assert _l0c_hard_reason("9 plus 6 equals what?", 15) == "result_2digit"        # operands 1-digit, result 2-digit
    assert _l0c_hard_reason("10 minus 1 equals what?", 9) == "operand_2digit_result_1digit"
    assert _l0c_hard_reason("42 equals what?", 42) == "result_2digit"
    assert _l0c_hard_reason("3 plus 4 equals what?", 7) == "easy"


def test_hard_reason_consistent_with_is_hard_over_full_support():
    # A row is hard iff hard_reason != 'easy' (the two predicates must agree).
    for _rung, rows in build_exhaustive_l0c_supports().items():
        for (q, e) in rows:
            assert _l0c_is_hard(q, e) == (_l0c_hard_reason(q, e) != "easy"), (q, e)


# --------------------------------------------------------------------------- #
# Partition: deterministic, exactly 230, disjoint, all-hard
# --------------------------------------------------------------------------- #

def test_partition_total_is_230():
    train, held = _enumerate_partition_l0c2(17)
    assert L0C2_EXPECTED_COUNT == 230
    assert len(train) + len(held) == 230


def test_partition_deterministic():
    a_tr, a_hl = _enumerate_partition_l0c2(17)
    b_tr, b_hl = _enumerate_partition_l0c2(17)
    assert _qe(a_tr) == _qe(b_tr)
    assert _qe(a_hl) == _qe(b_hl)


def test_train_held_disjoint_on_qe():
    train, held = _enumerate_partition_l0c2(17)
    assert set(_qe(train)).isdisjoint(set(_qe(held)))


def test_all_rows_are_hard():
    train, held = _enumerate_partition_l0c2(17)
    assert all(_l0c_is_hard(r["question"], r["expected"]) for r in train + held)
    # And no operand-or-result is single-digit-only (i.e. no 'easy' leaked in).
    assert all(_l0c_hard_reason(r["question"], r["expected"]) != "easy" for r in train + held)


# --------------------------------------------------------------------------- #
# Stratification: equal-ish (source_rung x operator) coverage + allocation
# --------------------------------------------------------------------------- #

def test_every_source_rung_operator_bucket_represented():
    train, held = _enumerate_partition_l0c2(17)
    present = {(r["source_rung"], r["operator"]) for r in train + held}
    pool = _pool_buckets()
    assert present == pool, f"missing buckets: {pool - present}; extra: {present - pool}"
    # Sanity: this pool is the known 12-bucket shape (R0/identity, R1/{plus,minus},
    # R1b1..R1b9). Pin the count + a few load-bearing buckets explicitly.
    assert len(pool) == 12
    assert ("R0", "identity") in pool
    assert ("R1", "plus") in pool and ("R1", "minus") in pool
    assert ("R1b2", "minus") in pool


def test_equal_ish_per_bucket_allocation():
    train, held = _enumerate_partition_l0c2(17)
    per_bucket = Counter((r["source_rung"], r["operator"]) for r in train + held)
    # 230 / 12 = 19.17 -> ten buckets of 19, two of 20.
    counts = Counter(per_bucket.values())
    assert set(counts) <= {19, 20}, dict(per_bucket)
    assert counts[20] == 2 and counts[19] == 10, dict(per_bucket)


def test_split_ratio_approx_80_20():
    train, held = _enumerate_partition_l0c2(17)
    frac_held = len(held) / (len(train) + len(held))
    assert 0.17 <= frac_held <= 0.23, "held fraction %.3f not ~0.20" % frac_held


# --------------------------------------------------------------------------- #
# Regression class: operand_2digit_result_1digit singleton (`10 minus 1 -> 9`)
# present, in TRAIN, and never dropped (codex msg 1779705530223 correction).
# --------------------------------------------------------------------------- #

def test_operand_only_hard_row_present_and_in_train():
    train, held = _enumerate_partition_l0c2(17)
    train_qe = set(_qe(train))
    held_qe = set(_qe(held))
    assert _OPERAND_ONLY in train_qe, "the `10 minus 1 -> 9` regression row is missing from TRAIN"
    assert _OPERAND_ONLY not in held_qe, "singleton class must not be held-split"
    # It is exactly the operand_2digit_result_1digit class.
    q, e = _OPERAND_ONLY
    assert _l0c_hard_reason(q, e) == "operand_2digit_result_1digit"


def test_operand_only_class_covered_where_present():
    # Every bucket that HAS an operand_2digit_result_1digit row in the pool must
    # carry at least one in the L0c2 support (rare-first reservation guarantee).
    train, held = _enumerate_partition_l0c2(17)
    chosen_o2r1 = {
        (r["source_rung"], r["operator"])
        for r in train + held
        if r["hard_reason"] == "operand_2digit_result_1digit"
    }
    pool_o2r1 = set()
    for source_rung, rows in build_exhaustive_l0c_supports().items():
        for (q, e) in rows:
            if _l0c_is_hard(q, e) and _l0c_hard_reason(q, e) == "operand_2digit_result_1digit":
                pool_o2r1.add((source_rung, _l0c_operator(q)))
    assert chosen_o2r1 == pool_o2r1
    assert pool_o2r1 == {("R1b2", "minus")}  # the only such bucket in this pool


# --------------------------------------------------------------------------- #
# Regression: sibling rungs UNCHANGED
# --------------------------------------------------------------------------- #

def test_bounded_l0c_unchanged_230():
    train, held = _enumerate_partition_l0c(17)
    assert len(train) + len(held) == 230


def test_l0c1_unchanged_121():
    train, held = _enumerate_partition_l0c1(17)
    assert len(train) + len(held) == 121


def test_l0c_exhaustive_support_unchanged_1255():
    n = sum(len(v) for v in build_exhaustive_l0c_supports().values())
    assert n == 1255


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"  {_name}: PASS")
    print("L0c2-rung tests: PASS")
