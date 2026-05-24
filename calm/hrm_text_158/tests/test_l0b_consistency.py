"""Tests for the L0b retained-support KL-only consistency slice (F.2f).

Codex +1 msg 1779647554279-522ba519 (shape) + determinism design_proposal
1779647581438-d34c44db. Covers the cheap gates: canonical support snapshot +
content hash determinism, the K-cyclic sampler determinism/coverage, KL-only
(no-CE) helper semantics on an all-prior side batch, and the train() guard
rejections. No GPU / no model load required.

Key invariant under test: the 230-row L0b support is SEED-DEPENDENT as a set
(two_digit picks in `_enumerate_partition_l0b` are seeded), so the consistency
support MUST be built with curriculum_seed=17 for the F.2f run — only then are
the held holes observed under seed 17 actually in the protected set.
"""
import importlib.util
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Load the trainer module by path (scripts/ is not a package).
_spec = importlib.util.spec_from_file_location(
    "_train_hrm_text_158", os.path.join(_REPO, "scripts", "train_hrm_text_158.py")
)
_thr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_thr)

from calm.hrm_text_158.lm_head import IGNORE_LABEL_ID
from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0b

_support = _thr._l0b_consistency_support
_Sampler = _thr._L0bConsistencySampler
_pc_kl = _thr._parent_consistency_kl

# Held L0b rows confirmed as F.2d/F.2e moving-hole corruptions under seed 17
# (the whack-a-mole the broad fix subsumes): the persistent hard-row hole and
# the F.2e final-step hole. Both are HELD rows (replay is train-only, so they
# were never guarded before this slice).
_CONFIRMED_HELD_HOLES = [
    ("calculate 14 plus 2.", 16),
    ("calculate 40 minus 1.", 39),
]


# --------------------------------------------------------------------------- #
# Support snapshot — determinism, count, canonical order, held coverage
# --------------------------------------------------------------------------- #

def test_support_count_and_surface():
    rows, h = _support(17)
    assert len(rows) == 230, f"L0b support must be 230, got {len(rows)}"
    assert len(h) == 16 and all(c in "0123456789abcdef" for c in h)
    # All rows are the L0b `calculate <expr>.` surface.
    for q, e, sr in rows:
        assert q.startswith("calculate ") and q.endswith(".")
        assert isinstance(e, int) and isinstance(sr, str)


def test_support_is_deterministic_same_seed():
    r1, h1 = _support(17)
    r2, h2 = _support(17)
    assert r1 == r2 and h1 == h2, "same seed must yield identical support + hash"


def test_support_canonical_order():
    rows, _ = _support(17)
    assert rows == sorted(rows, key=lambda r: (r[2], r[0], r[1])), \
        "support must be canonical-sorted by (source_rung, question, expected)"


def test_support_equals_train_union_held_and_covers_held():
    # The whole point of the broad fix: ALL held rows are in the protected set
    # (replay is train-only, so held rows were previously unguarded).
    train, held = _enumerate_partition_l0b(17)
    rows, _ = _support(17)
    sup_q = {(q, e) for (q, e, _sr) in rows}
    train_q = {(r["question"], r["expected"]) for r in train}
    held_q = {(r["question"], r["expected"]) for r in held}
    assert held_q <= sup_q, "every L0b held row must be in the consistency support"
    assert train_q <= sup_q
    assert sup_q == (train_q | held_q)
    assert len(held_q) == 46 and len(train_q) == 184


def test_confirmed_held_holes_present_in_support_17():
    # Codex test req: "all recent held holes present in _l0b_support(17)."
    _train, held = _enumerate_partition_l0b(17)
    held_q = {(r["question"], r["expected"]) for r in held}
    rows, _ = _support(17)
    sup_q = {(q, e) for (q, e, _sr) in rows}
    for q, e in _CONFIRMED_HELD_HOLES:
        assert (q, e) in sup_q, f"confirmed hole {q!r}->{e} missing from support(17)"
        assert (q, e) in held_q, f"confirmed hole {q!r}->{e} should be a HELD row"


def test_support_set_is_seed_dependent():
    # Documents WHY curriculum_seed=17 is load-bearing: a different seed picks
    # different two_digit rows, so the protected set genuinely differs.
    rows_17, h17 = _support(17)
    rows_42, h42 = _support(42)
    assert h17 != h42, "different seed must change the support hash (seed-dependent set)"
    assert len(rows_17) == len(rows_42) == 230


# --------------------------------------------------------------------------- #
# K-cyclic sampler — determinism, coverage, wrap
# --------------------------------------------------------------------------- #

def test_sampler_same_seed_identical_first_3_batches():
    s1 = _Sampler(n=230, seed=17, batch=8)
    s2 = _Sampler(n=230, seed=17, batch=8)
    assert s1.perm == s2.perm, "same seed must yield identical permutation"
    b1 = [s1.next_indices() for _ in range(3)]
    b2 = [s2.next_indices() for _ in range(3)]
    assert b1 == b2, "same seed must yield identical first 3 side batches"


def test_sampler_different_seed_changes_perm_only():
    a = _Sampler(n=230, seed=17, batch=8)
    b = _Sampler(n=230, seed=18, batch=8)
    assert a.n == b.n == 230
    assert a.perm != b.perm, "different seed must change the sampler permutation"


def test_sampler_perm_is_a_permutation_full_cycle_coverage():
    s = _Sampler(n=230, seed=17, batch=8)
    assert sorted(s.perm) == list(range(230)), "perm must cover every row exactly once"
    # Walking one full cycle (ceil(230/8) batches) covers all rows ≥1x.
    seen = set()
    n_batches = (230 + 8 - 1) // 8
    for _ in range(n_batches):
        seen.update(s.next_indices())
    assert seen == set(range(230)), "one cyclic pass must cover the full support"


def test_sampler_cyclic_wrap_and_coverage_counter():
    s = _Sampler(n=230, seed=17, batch=8)
    total = 0
    for _ in range(40):  # 40*8 = 320 > 230 → wraps
        total += len(s.next_indices())
    cov = s.coverage()
    assert cov["rows_seen"] == total == 320
    assert cov["full_cycles"] == 320 // 230 == 1
    assert cov["cursor"] == 320 % 230
    assert cov["support_seed"] == _thr._stable_curriculum_seed(17, "l0b_consistency")


def test_sampler_rejects_bad_args():
    import pytest
    with pytest.raises(ValueError):
        _Sampler(n=0, seed=17, batch=8)
    with pytest.raises(ValueError):
        _Sampler(n=230, seed=17, batch=0)


# --------------------------------------------------------------------------- #
# KL-only (no-CE) semantics on the all-prior side batch
# --------------------------------------------------------------------------- #

def _mk_labels(B, L, resp_from=3):
    labels = torch.full((B, L), IGNORE_LABEL_ID, dtype=torch.long)
    labels[:, resp_from:] = torch.randint(0, 260, (B, L - resp_from))
    return labels


def test_side_batch_all_prior_kl_only_on_response_positions():
    # The side batch tags every row is_prior=True; KL fires on response
    # positions only (labels != IGNORE), never on the prompt prefix, and there
    # is NO cross-entropy term — only the KL flows gradient.
    torch.manual_seed(0)
    B, L, V = 4, 6, 260
    labels = _mk_labels(B, L, resp_from=3)
    is_prior = torch.ones(B, dtype=torch.bool)

    logits = torch.randn(B, L, V)
    kl_same = _pc_kl(logits, logits.clone(), labels, is_prior, temp=1.0)
    assert kl_same.item() < 1e-6, f"child==parent must be ~0, got {kl_same.item()}"

    child = torch.randn(B, L, V, requires_grad=True)
    parent = torch.randn(B, L, V)
    kl = _pc_kl(child, parent, labels, is_prior, temp=1.0)
    assert kl.item() > 0.0
    kl.backward()
    assert torch.isfinite(child.grad).all()
    # Prefix positions (ignored labels) get ZERO grad — KL is response-only.
    assert child.grad[:, :3].abs().sum().item() == 0.0, "prefix must get 0 grad"
    assert child.grad[:, 3:].abs().sum().item() > 0.0, "response positions must get grad"


# --------------------------------------------------------------------------- #
# train() guard rejections (fire BEFORE any data/model work)
# --------------------------------------------------------------------------- #

def test_rejects_negative_l0b_weight():
    import pytest
    with pytest.raises(ValueError, match="must be >= 0"):
        _thr.train(l0b_consistency_weight=-1.0)


def test_l0b_weight_requires_load_from():
    import pytest
    with pytest.raises(ValueError, match="requires --load-from"):
        _thr.train(l0b_consistency_weight=1.0)


def test_l0b_weight_requires_curriculum_mode():
    import pytest
    with pytest.raises(ValueError, match="requires curriculum mode"):
        _thr.train(l0b_consistency_weight=1.0, load_from="x", curriculum_rung=None)


def test_l0b_batch_must_be_positive_when_enabled():
    import pytest
    with pytest.raises(ValueError, match="must be >= 1"):
        _thr.train(l0b_consistency_weight=1.0, load_from="x",
                   curriculum_rung="L0c1", l0b_consistency_batch=0)


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"  {_name}: PASS")
    print("l0b-consistency tests: PASS")
