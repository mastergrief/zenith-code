"""CPU exactness fence for `apply_sparse_hot` (F3 / arm3_sparse_hot_forgettable_cold).

Pins live semantics from forgetting_laws.apply_sparse_hot (UNTOUCHED):
  - flatten |acc| across the name→tensor dict (GLOBAL top-H)
  - k = min(hot_h, numel); k<=0 → all-zero; k>=numel → identity
  - thresh = min(topk(|flat|, k)); keep = |flat| >= thresh
  - if keep.sum() > k: drop later flat indices (nonzero ascending) so |kept|==k
  - cold → 0; hot retain original signed values

Also grounds Z1/Z2 mix-in discrimination: sparse_hot ≠ decay_leak and ≠ ttl on a
shared fixture. Does NOT add receipt counters (FINDING restated in receipt).
"""
from __future__ import annotations

from typing import Mapping

import torch

from calm.hrm_text_158.native_full_stack.forgetting_laws import (
    apply_decay_leak,
    apply_sparse_hot,
    apply_ttl_age_drain,
)

HOT_H_PREREG = 8192


def _exact_oracle(
    arms_acc: Mapping[str, list],
    *,
    hot_h: int = HOT_H_PREREG,
) -> dict[str, list]:
    """Pure-Python oracle mirroring live apply_sparse_hot (incl. flat-index trim)."""
    names = list(arms_acc.keys())
    flats: list[int] = []
    meta: list[tuple[str, int]] = []
    for n in names:
        vals = [int(x) for x in arms_acc[n]]
        flats.extend(abs(v) for v in vals)
        meta.append((n, len(vals)))
    n_all = len(flats)
    k = min(int(hot_h), n_all)
    if k <= 0:
        return {n: [0] * len(arms_acc[n]) for n in names}
    # kth-largest threshold (min of top-k)
    ordered = sorted(flats, reverse=True)
    thresh = ordered[k - 1]
    keep = [abs_v >= thresh for abs_v in flats]
    keep_idx = [i for i, flag in enumerate(keep) if flag]
    if len(keep_idx) > k:
        for i in keep_idx[k:]:
            keep[i] = False
    out: dict[str, list] = {}
    off = 0
    for n, nn in meta:
        src = [int(x) for x in arms_acc[n]]
        out[n] = [src[j] if keep[off + j] else 0 for j in range(nn)]
        off += nn
    return out


def _as_tensors(arms: Mapping[str, list], *, dtype=torch.int16) -> dict[str, torch.Tensor]:
    return {n: torch.tensor(v, dtype=dtype) for n, v in arms.items()}


def _as_lists(arms: Mapping[str, torch.Tensor]) -> dict[str, list]:
    return {n: t.detach().cpu().tolist() for n, t in arms.items()}


def test_oracle_matches_impl_on_basic_fixture() -> None:
    arms = {
        "a": [10, -3, 1, 0, -8],
        "b": [2, 7, -1],
    }
    hot_h = 3
    got = _as_lists(apply_sparse_hot(_as_tensors(arms), hot_h=hot_h))
    want = _exact_oracle(arms, hot_h=hot_h)
    assert got == want
    # global top-3 by |acc|: 10, -8, 7
    assert got["a"] == [10, 0, 0, 0, -8]
    assert got["b"] == [0, 7, 0]


def test_true_negative_k_le_zero_all_zero() -> None:
    arms = {"a": [5, -4, 3], "b": [2]}
    for hot_h in (0, -1, -100):
        got = _as_lists(apply_sparse_hot(_as_tensors(arms), hot_h=hot_h))
        want = _exact_oracle(arms, hot_h=hot_h)
        assert got == want
        assert all(v == 0 for row in got.values() for v in row)


def test_true_negative_h_ge_numel_identity() -> None:
    arms = {"a": [5, -4], "b": [3, -2, 1]}
    n = sum(len(v) for v in arms.values())
    for hot_h in (n, n + 1, HOT_H_PREREG):
        got = _as_lists(apply_sparse_hot(_as_tensors(arms), hot_h=hot_h))
        want = _exact_oracle(arms, hot_h=hot_h)
        assert got == want == arms


def test_tie_trim_exactly_h_and_subset_of_eligible() -> None:
    """Forced ties at threshold: |kept|==H; kept ⊆ {|x|>=thresh}."""
    # 6 equal |acc|=5 → thresh=5, keep.sum()=6 > H=3 → trim to 3
    arms = {"a": [5, -5, 5, -5, 5, -5]}
    hot_h = 3
    got = _as_lists(apply_sparse_hot(_as_tensors(arms), hot_h=hot_h))
    want = _exact_oracle(arms, hot_h=hot_h)
    assert got == want
    flats_abs = [abs(x) for x in arms["a"]]
    ordered = sorted(flats_abs, reverse=True)
    thresh = ordered[hot_h - 1]
    eligible = {i for i, a in enumerate(flats_abs) if a >= thresh}
    kept = {i for i, (pre, post) in enumerate(zip(arms["a"], got["a"], strict=True)) if post == pre and pre != 0}
    # also count retained zeros? none here
    assert len(kept) == hot_h
    assert kept <= eligible
    # live trim keeps earliest flat indices among eligible
    assert kept == {0, 1, 2}


def test_cross_tensor_global_top_h() -> None:
    """Hot set is GLOBAL: one tensor may be all-cold; masks reassemble to top-H."""
    arms = {
        "hot_only": [100, 90, 80],
        "cold_only": [1, 2, 3],
        "mixed": [50, 4, 70],
    }
    hot_h = 4
    got = _as_lists(apply_sparse_hot(_as_tensors(arms), hot_h=hot_h))
    want = _exact_oracle(arms, hot_h=hot_h)
    assert got == want
    # top-4 abs: 100,90,80,70 — cold_only all zeroed; mixed keeps only 70
    assert got["hot_only"] == [100, 90, 80]
    assert got["cold_only"] == [0, 0, 0]
    assert got["mixed"] == [0, 0, 70]
    retained = sum(1 for row in got.values() for v in row if v != 0)
    assert retained == hot_h


def test_bounded_dual_oracle_sweep_zero_mismatches() -> None:
    """Bounded fixture family: impl == pure-Python oracle, 0 mismatches."""
    mismatches = 0
    cases = 0
    shapes = [
        {"a": [1]},
        {"a": [0, 0, 0]},
        {"a": list(range(-7, 8))},
        {"a": [5] * 10, "b": [-5] * 10},
        {"u": [9, 1], "v": [8, 2, 7], "w": [3]},
        {"x": [0, 1, 0, 2, 0, 3, 0, 4]},
    ]
    hot_hs = [0, 1, 2, 3, 5, 8, 16, 64, HOT_H_PREREG]
    for arms in shapes:
        n = sum(len(v) for v in arms.values())
        for hot_h in hot_hs:
            got = _as_lists(apply_sparse_hot(_as_tensors(arms), hot_h=hot_h))
            want = _exact_oracle(arms, hot_h=hot_h)
            cases += 1
            if got != want:
                mismatches += 1
                raise AssertionError(
                    f"mismatch hot_h={hot_h} n={n} arms={arms}: got={got} want={want}"
                )
            # |kept nonzero-or-retained-nonzero| bound
            k = min(int(hot_h), n)
            if k > 0:
                # count positions where |pre|>0 and post==pre, plus we allow pre==0 retained
                kept_nonzero = sum(
                    1
                    for n_ in arms
                    for pre, post in zip(arms[n_], got[n_], strict=True)
                    if int(pre) != 0 and int(post) == int(pre)
                )
                assert kept_nonzero <= k
    assert cases == len(shapes) * len(hot_hs)
    assert mismatches == 0


def test_no_mixin_contrast_vs_decay_and_ttl() -> None:
    """Law-identity discriminator: sparse_hot ≠ decay and ≠ TTL on shared fixture."""
    pre = torch.tensor([20, -18, 5, 4, 3, 2, 8, 0], dtype=torch.int16)
    hot_h = 3
    sparse = apply_sparse_hot({"a": pre.clone()}, hot_h=hot_h)["a"]
    decayed = apply_decay_leak(pre.clone(), lam=1.0 / 32.0)
    ep = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0], dtype=torch.int32)
    ttl_acc, _ = apply_ttl_age_drain(pre.clone(), ep, step=100, ttl=32)
    assert sparse.tolist() != decayed.tolist()
    assert sparse.tolist() != ttl_acc.tolist()
    # sparse keeps global top-3 |acc|: 20, -18, 8
    assert sparse.tolist() == [20, -18, 0, 0, 0, 0, 8, 0]
    # decay is age-independent shrink: cold-ish mid values survive (not hard-zeroed as a set)
    assert int(decayed[0].item()) != 0 and int(decayed[3].item()) != 0
    # index 3 (|4|) is cold under sparse_hot but non-zero under decay
    assert int(sparse[3].item()) == 0 and int(decayed[3].item()) != 0
    # ttl with age=99>32 drains active episodes → differs from sparse retain pattern
    assert ttl_acc.tolist() != sparse.tolist()


def test_finding_flat_trim_can_drop_above_thresh() -> None:
    """FINDING (characterization only): flat-index trim can drop |x|>thresh.

    Fixture [5,5,5,10] hot_h=2 → thresh=5; live keeps earliest two 5s and drops 10.
    Fence does NOT change the law — documents live behavior for ns5 D-branch awareness.
    """
    arms = {"a": [5, 5, 5, 10]}
    got = _as_lists(apply_sparse_hot(_as_tensors(arms), hot_h=2))
    want = _exact_oracle(arms, hot_h=2)
    assert got == want
    assert got["a"] == [5, 5, 0, 0]
    assert got["a"][3] == 0  # 10 dropped despite |10|>thresh
