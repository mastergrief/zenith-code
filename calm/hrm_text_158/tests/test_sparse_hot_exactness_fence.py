"""CPU exactness fence for `apply_sparse_hot` (F3 / arm3_sparse_hot_forgettable_cold).

Law-conformance (post repair): exact global top-H |acc| via torch.topk indices.
  - flatten |acc| across the name→tensor dict (GLOBAL top-H)
  - k = min(hot_h, numel); k<=0 → all-zero; k>=numel → identity
  - keep = top-k indices of |flat| (not threshold+flat-trim)
  - tie-break: torch.topk native index order; kept |values| multiset == true top-k
  - cold → 0; hot retain original signed values

Also grounds Z1/Z2 mix-in discrimination: sparse_hot ≠ decay_leak and ≠ ttl on a
shared fixture. Does NOT add receipt counters.
"""
from __future__ import annotations

from collections import Counter
from typing import Mapping

import torch

from calm.hrm_text_158.native_full_stack.forgetting_laws import (
    apply_decay_leak,
    apply_sparse_hot,
    apply_ttl_age_drain,
)

HOT_H_PREREG = 8192


def _flat_abs(arms_acc: Mapping[str, list]) -> list[int]:
    out: list[int] = []
    for n in arms_acc.keys():
        out.extend(abs(int(v)) for v in arms_acc[n])
    return out


def _is_tie_free(arms_acc: Mapping[str, list], hot_h: int) -> bool:
    """True when the top-k set is unique (strict gap between k-th and (k+1)-th)."""
    flats = _flat_abs(arms_acc)
    n = len(flats)
    k = min(int(hot_h), n)
    if k <= 0 or k >= n:
        return True
    ordered = sorted(flats, reverse=True)
    return ordered[k - 1] > ordered[k]


def _exact_oracle_tie_free(
    arms_acc: Mapping[str, list],
    *,
    hot_h: int = HOT_H_PREREG,
) -> dict[str, list]:
    """Pure-Python true top-H for tie-free fixtures (unique keep set)."""
    names = list(arms_acc.keys())
    flats = _flat_abs(arms_acc)
    meta = [(n, len(arms_acc[n])) for n in names]
    n_all = len(flats)
    k = min(int(hot_h), n_all)
    if k <= 0:
        return {n: [0] * len(arms_acc[n]) for n in names}
    # Unique top-k set: all indices with abs > thresh, plus all == thresh when count fits
    ordered = sorted(flats, reverse=True)
    thresh = ordered[k - 1]
    keep = [abs_v >= thresh for abs_v in flats]
    assert sum(keep) == k  # tie-free invariant
    out: dict[str, list] = {}
    off = 0
    for n, nn in meta:
        src = [int(x) for x in arms_acc[n]]
        out[n] = [src[j] if keep[off + j] else 0 for j in range(nn)]
        off += nn
    return out


def _assert_true_topk_properties(
    arms_in: Mapping[str, list],
    arms_out: Mapping[str, list],
    *,
    hot_h: int,
) -> None:
    """Property oracle: true top-H multiset / above-thresh / |kept|==k / ⊆ eligible."""
    flats = _flat_abs(arms_in)
    n = len(flats)
    k = min(int(hot_h), n)
    retained_abs: list[int] = []
    kept_idx: list[int] = []
    off = 0
    for name in arms_in.keys():
        for j, (pre, post) in enumerate(zip(arms_in[name], arms_out[name], strict=True)):
            if int(post) == int(pre) and int(pre) != 0:
                kept_idx.append(off + j)
                retained_abs.append(abs(int(pre)))
            elif int(post) != 0 and int(post) != int(pre):
                raise AssertionError("sparse_hot must retain original signed values or zero")
            elif int(pre) != 0 and int(post) == 0:
                pass  # cold
            elif int(pre) == 0 and int(post) == 0:
                pass
            else:
                raise AssertionError(f"unexpected pre/post ({pre},{post})")
        off += len(arms_in[name])

    if k <= 0:
        assert all(v == 0 for row in arms_out.values() for v in row)
        return

    # Identity path: k >= n retains every element (including zeros).
    if k >= n:
        assert arms_out == arms_in
        return

    # All-zero input: top-k retains zeros in-place (identity); no nonzero kept set.
    if all(a == 0 for a in flats):
        assert arms_out == arms_in
        return

    ordered = sorted(flats, reverse=True)
    topk_multiset = Counter(ordered[:k])
    n_nonzero = sum(1 for a in flats if a != 0)
    # Output-visible kept count: nonzero retained. Zeros in the top-k mask are invisible.
    if n_nonzero >= k:
        assert len(kept_idx) == k
        assert Counter(retained_abs) == topk_multiset
    else:
        assert len(kept_idx) == n_nonzero
        assert Counter(retained_abs) == Counter(a for a in flats if a != 0)

    thresh = ordered[k - 1]
    eligible = {i for i, a in enumerate(flats) if a >= thresh}
    assert set(kept_idx) <= eligible
    # every strictly-above-thresh element retained
    above = {i for i, a in enumerate(flats) if a > thresh}
    assert above <= set(kept_idx)


def _as_tensors(arms: Mapping[str, list], *, dtype=torch.int16) -> dict[str, torch.Tensor]:
    return {n: torch.tensor(v, dtype=dtype) for n, v in arms.items()}


def _as_lists(arms: Mapping[str, torch.Tensor]) -> dict[str, list]:
    return {n: t.detach().cpu().tolist() for n, t in arms.items()}


def test_oracle_matches_impl_on_basic_fixture() -> None:
    """Tie-free fixture: exact pure-Python oracle match + known keep set."""
    arms = {
        "a": [10, -3, 1, 0, -8],
        "b": [2, 7, -1],
    }
    hot_h = 3
    assert _is_tie_free(arms, hot_h)
    got = _as_lists(apply_sparse_hot(_as_tensors(arms), hot_h=hot_h))
    want = _exact_oracle_tie_free(arms, hot_h=hot_h)
    assert got == want
    # global top-3 by |acc|: 10, -8, 7
    assert got["a"] == [10, 0, 0, 0, -8]
    assert got["b"] == [0, 7, 0]
    _assert_true_topk_properties(arms, got, hot_h=hot_h)


def test_true_negative_k_le_zero_all_zero() -> None:
    arms = {"a": [5, -4, 3], "b": [2]}
    for hot_h in (0, -1, -100):
        got = _as_lists(apply_sparse_hot(_as_tensors(arms), hot_h=hot_h))
        _assert_true_topk_properties(arms, got, hot_h=hot_h)
        assert all(v == 0 for row in got.values() for v in row)


def test_true_negative_h_ge_numel_identity() -> None:
    arms = {"a": [5, -4], "b": [3, -2, 1]}
    n = sum(len(v) for v in arms.values())
    for hot_h in (n, n + 1, HOT_H_PREREG):
        got = _as_lists(apply_sparse_hot(_as_tensors(arms), hot_h=hot_h))
        assert got == arms
        _assert_true_topk_properties(arms, got, hot_h=hot_h)


def test_exact_topk_tie_break_properties() -> None:
    """Forced ties: |kept|==H; kept ⊆ eligible; |values| multiset == top-k; above-thresh kept."""
    arms = {"a": [5, -5, 5, -5, 5, -5]}
    hot_h = 3
    got = _as_lists(apply_sparse_hot(_as_tensors(arms), hot_h=hot_h))
    _assert_true_topk_properties(arms, got, hot_h=hot_h)
    # All |values| are 5 → any 3 indices; count exact
    retained = [v for v in got["a"] if v != 0]
    assert len(retained) == hot_h
    assert all(abs(v) == 5 for v in retained)


def test_cross_tensor_global_top_h() -> None:
    """Hot set is GLOBAL: one tensor may be all-cold; masks reassemble to top-H."""
    arms = {
        "hot_only": [100, 90, 80],
        "cold_only": [1, 2, 3],
        "mixed": [50, 4, 70],
    }
    hot_h = 4
    assert _is_tie_free(arms, hot_h)
    got = _as_lists(apply_sparse_hot(_as_tensors(arms), hot_h=hot_h))
    want = _exact_oracle_tie_free(arms, hot_h=hot_h)
    assert got == want
    # top-4 abs: 100,90,80,70 — cold_only all zeroed; mixed keeps only 70
    assert got["hot_only"] == [100, 90, 80]
    assert got["cold_only"] == [0, 0, 0]
    assert got["mixed"] == [0, 0, 70]
    retained = sum(1 for row in got.values() for v in row if v != 0)
    assert retained == hot_h
    _assert_true_topk_properties(arms, got, hot_h=hot_h)


def test_bounded_dual_oracle_sweep_zero_mismatches() -> None:
    """Bounded fixture family: true top-H properties always; exact match on tie-free."""
    mismatches = 0
    cases = 0
    exact_checked = 0
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
            cases += 1
            try:
                _assert_true_topk_properties(arms, got, hot_h=hot_h)
            except AssertionError as exc:
                mismatches += 1
                raise AssertionError(
                    f"property mismatch hot_h={hot_h} n={n} arms={arms}: {exc}"
                ) from exc
            if _is_tie_free(arms, hot_h) and min(int(hot_h), n) > 0:
                want = _exact_oracle_tie_free(arms, hot_h=hot_h)
                exact_checked += 1
                if got != want:
                    mismatches += 1
                    raise AssertionError(
                        f"exact mismatch hot_h={hot_h} n={n} arms={arms}: "
                        f"got={got} want={want}"
                    )
    assert cases == len(shapes) * len(hot_hs)
    assert mismatches == 0
    assert exact_checked > 0


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
    assert int(decayed[0].item()) != 0 and int(decayed[3].item()) != 0
    assert int(sparse[3].item()) == 0 and int(decayed[3].item()) != 0
    assert ttl_acc.tolist() != sparse.tolist()


def test_conformance_above_thresh_retained_not_flat_trimmed() -> None:
    """CONFORMANCE (i): [5,5,5,10] H=2 MUST keep the 10 (exact top-k, not flat-trim)."""
    arms = {"a": [5, 5, 5, 10]}
    got = _as_lists(apply_sparse_hot(_as_tensors(arms), hot_h=2))
    _assert_true_topk_properties(arms, got, hot_h=2)
    assert got["a"][3] == 10  # acceptance (i)
    assert sorted(abs(v) for v in got["a"] if v != 0) == [5, 10]
    assert sum(1 for v in got["a"] if v != 0) == 2
