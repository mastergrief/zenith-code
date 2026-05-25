"""F.3d-b — 2-digit-emphasis ACQUISITION sampler/rung `L0c_exhaustive_2digit`
(codex msg 1779701225492; per-row ratio 1779701860482; slice-split 1779701431738).

Wires the weighted sampler on top of the F.3d-a `_l0c_is_hard` predicate. The
variant reuses the SAME exhaustive-L0c partition/support/audit surface as
`L0c_exhaustive`; the ONLY change is TRAIN sampling with a per-ROW hard weight
(3.0 hard / 1.0 easy -> ~97.6% hard by draw, NOT pool-70/30 since the support is
already 93.2% hard). Held-out stays uniform/full-support; bank gate is still the
full `--l0c-exhaustive-audit`. Acquisition variant ONLY (never a retained
support). No model / no GPU.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import random  # noqa: E402

from calm.hrm_text_158.curriculum.generators import (  # noqa: E402
    make_rung_examples,
    _gen_l0c_exhaustive_2digit,
    _enumerate_partition_l0c_exhaustive,
    _enumerate_partition_l0c,
    _l0c_is_hard,
    _RUNG_SPEC,
    RUNG_NAMES,
)
from calm.hrm_text_158.curriculum.language_supports import (  # noqa: E402
    build_exhaustive_l0c_supports,
)


def _qe(rows):
    return [(r["question"], r["expected"]) for r in rows]


def _hard_frac(rows):
    return sum(1 for r in rows if _l0c_is_hard(r["question"], r["expected"])) / len(rows)


# --------------------------------------------------------------------------- #
# Rung recognition / tagging / placement (codex placement caution 1779701936543)
# --------------------------------------------------------------------------- #

def test_rung_registered_after_l0c_exhaustive():
    assert "L0c_exhaustive_2digit" in RUNG_NAMES
    # Placed AFTER L0c_exhaustive so the step-4000 surface is a valid (past)
    # prior for optional F.3e replay; bounded L0c / L0c1 precede both.
    assert RUNG_NAMES.index("L0c_exhaustive_2digit") == RUNG_NAMES.index("L0c_exhaustive") + 1
    assert RUNG_NAMES.index("L0c") < RUNG_NAMES.index("L0c_exhaustive_2digit")
    assert RUNG_NAMES.index("L0c1") < RUNG_NAMES.index("L0c_exhaustive_2digit")


def test_make_rung_recognizes_and_tags():
    rows = make_rung_examples("L0c_exhaustive_2digit", 64, seed=17, split="train")
    assert len(rows) == 64
    assert all(r["rung"] == "L0c_exhaustive_2digit" for r in rows)
    assert all(r["question"].endswith(" equals what?") for r in rows)


# --------------------------------------------------------------------------- #
# Determinism + per-row weighted draw share (NOT pool-70/30)
# --------------------------------------------------------------------------- #

def test_sampler_deterministic():
    a = make_rung_examples("L0c_exhaustive_2digit", 300, seed=17, split="train")
    b = make_rung_examples("L0c_exhaustive_2digit", 300, seed=17, split="train")
    assert _qe(a) == _qe(b)


def test_train_hard_share_approx_97_6_at_3x():
    # Default spec hard_weight=3.0 -> 3*1170/(3*1170+85) ~ 0.976.
    spec = _RUNG_SPEC["L0c_exhaustive_2digit"]["train"]
    assert spec.get("hard_weight") == 3.0
    rng = random.Random(17)
    rows = _gen_l0c_exhaustive_2digit(rng, spec, 8000, seed=17, split="train")
    frac = _hard_frac(rows)
    assert 0.96 <= frac <= 0.99, "3x hard-share %.4f not ~0.976" % frac


def test_train_hard_share_2x_backoff():
    # Back-off lever: hard_weight=2.0 -> 2*1170/(2*1170+85) ~ 0.965.
    spec = {"partition": "enumerate_stratified_l0c_exhaustive", "hard_weight": 2.0}
    rng = random.Random(17)
    rows = _gen_l0c_exhaustive_2digit(rng, spec, 8000, seed=17, split="train")
    frac = _hard_frac(rows)
    assert 0.95 <= frac <= 0.975, "2x hard-share %.4f not ~0.965" % frac


def test_train_share_is_emphasis_over_uniform():
    # The whole point: 3x weighting samples HARDER than uniform (93.2%).
    spec = _RUNG_SPEC["L0c_exhaustive_2digit"]["train"]
    rng = random.Random(17)
    rows = _gen_l0c_exhaustive_2digit(rng, spec, 8000, seed=17, split="train")
    train_pool, _ = _enumerate_partition_l0c_exhaustive(17)
    uniform_hard = sum(1 for r in train_pool if _l0c_is_hard(r["question"], r["expected"])) / len(train_pool)
    assert _hard_frac(rows) > uniform_hard + 0.02


# --------------------------------------------------------------------------- #
# Held-out stays UNIFORM (3x is train-only) so the audit surface is unbiased
# --------------------------------------------------------------------------- #

def test_held_is_uniform_not_weighted():
    spec = _RUNG_SPEC["L0c_exhaustive_2digit"]["held_out"]
    rng = random.Random(17)
    rows = _gen_l0c_exhaustive_2digit(rng, spec, 8000, seed=17, split="held_out")
    _, held_pool = _enumerate_partition_l0c_exhaustive(17)
    pool_hard = sum(1 for r in held_pool if _l0c_is_hard(r["question"], r["expected"])) / len(held_pool)
    assert abs(_hard_frac(rows) - pool_hard) < 0.05  # tracks natural pool, not 0.976


# --------------------------------------------------------------------------- #
# Reuses the existing partition: disjoint on (q,e); train ⊆ pool; audit 1255
# --------------------------------------------------------------------------- #

def test_train_held_disjoint_on_qe():
    tr = make_rung_examples("L0c_exhaustive_2digit", 2000, seed=17, split="train")
    hl = make_rung_examples("L0c_exhaustive_2digit", 2000, seed=17, split="held_out")
    assert set(_qe(tr)).isdisjoint(set(_qe(hl)))


def test_train_outputs_subset_of_train_pool():
    train_pool, _ = _enumerate_partition_l0c_exhaustive(17)
    pool_qe = set((r["question"], r["expected"]) for r in train_pool)
    tr = make_rung_examples("L0c_exhaustive_2digit", 2000, seed=17, split="train")
    assert set(_qe(tr)).issubset(pool_qe)


def test_audit_support_unchanged_1255():
    n = sum(len(v) for v in build_exhaustive_l0c_supports().values())
    assert n == 1255


# --------------------------------------------------------------------------- #
# Regression: default L0c_exhaustive + bounded L0c are UNCHANGED
# --------------------------------------------------------------------------- #

def test_default_l0c_exhaustive_unchanged():
    a = make_rung_examples("L0c_exhaustive", 300, seed=17, split="train")
    b = make_rung_examples("L0c_exhaustive", 300, seed=17, split="train")
    assert _qe(a) == _qe(b)
    assert all(r["rung"] == "L0c_exhaustive" for r in a)
    # uniform (NOT weighted): hard-fraction tracks the pool hardness, not 0.976.
    train_pool, _ = _enumerate_partition_l0c_exhaustive(17)
    pool_hard = sum(1 for r in train_pool if _l0c_is_hard(r["question"], r["expected"])) / len(train_pool)
    assert abs(_hard_frac(a) - pool_hard) < 0.12


def test_bounded_l0c_unchanged_230():
    train, held = _enumerate_partition_l0c(17)
    assert len(train) + len(held) == 230


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"  {_name}: PASS")
    print("L0c_exhaustive_2digit-rung tests: PASS")
