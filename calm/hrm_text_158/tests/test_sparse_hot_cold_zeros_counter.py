"""CPU characterization: n_sparse_hot_cold_zeros == oracle force-zero count + ARM3 emit.

PLAN: artifacts/acc_entropy/arm3_sparse_hot_cold_zeros_counter_PLAN_v1.json
Bindings: output-derived (pre!=0)&(post==0); non-ARM3 omit; no fake-zero defaults.
"""
from __future__ import annotations

from typing import Mapping

import torch

from calm.hrm_text_158.native_full_stack.family_classifier import (
    ARM0,
    ARM1,
    ARM2,
    ARM3,
)
from calm.hrm_text_158.native_full_stack.forgetting_laws import (
    apply_sparse_hot,
    apply_sparse_hot_with_count,
)
from calm.hrm_text_158.native_full_stack.phase_receipt_contracts import (
    arm_metrics_for_classifier,
)
from calm.hrm_text_158.native_full_stack.screen_receipt_output import (
    arm3_sparse_hot_cold_zeros_measurement_fields,
)


def _oracle_cold_zeros(
    arms_pre: Mapping[str, list],
    arms_post: Mapping[str, list],
) -> int:
    """Pure-Python: count (pre!=0)&(post==0) across named tensors."""
    n = 0
    for name in arms_pre.keys():
        for pre, post in zip(arms_pre[name], arms_post[name], strict=True):
            if int(pre) != 0 and int(post) == 0:
                n += 1
    return n


def _as_tensors(arms: Mapping[str, list]) -> dict[str, torch.Tensor]:
    return {n: torch.tensor(v, dtype=torch.int16) for n, v in arms.items()}


def _as_lists(arms: Mapping[str, torch.Tensor]) -> dict[str, list]:
    return {n: t.detach().cpu().tolist() for n, t in arms.items()}


def test_wrapper_matches_oracle_and_primitive_outputs() -> None:
    arms = {"a": [10, -3, 1, 0, -8], "b": [2, 7, -1]}
    hot_h = 3
    tens = _as_tensors(arms)
    prim = apply_sparse_hot(tens, hot_h=hot_h)
    got, n = apply_sparse_hot_with_count(tens, hot_h=hot_h)
    assert _as_lists(got) == _as_lists(prim)
    want_n = _oracle_cold_zeros(arms, _as_lists(got))
    assert n == want_n
    # top-3 keep 10,-8,7 → force-zero 1,2,-1,-3 (4); pre-zero at index3 never counts
    assert n == 4


def test_prezero_never_counts() -> None:
    """Zeros-in-topk-mask / pre-zero elements NEVER increment counter."""
    arms = {"x": [0, 1, 0, 2, 0, 3, 0, 4]}
    hot_h = 5  # only 4 nonzeros → at least one zero "kept"; pre-zeros must not count
    got, n = apply_sparse_hot_with_count(_as_tensors(arms), hot_h=hot_h)
    assert n == _oracle_cold_zeros(arms, _as_lists(got))
    # With H>=4 nonzeros and n=8, all nonzeros retained → n==0
    assert n == 0
    # Force some cold: H=2 keeps top |acc| 4 and 3
    got2, n2 = apply_sparse_hot_with_count(_as_tensors(arms), hot_h=2)
    assert n2 == _oracle_cold_zeros(arms, _as_lists(got2)) == 2  # 1 and 2 force-zeroed
    assert _as_lists(got2)["x"][0] == 0  # pre-zero stays 0, not counted


def test_identity_path_counter_zero() -> None:
    arms = {"a": [5, -4], "b": [3, -2, 1]}
    n_el = sum(len(v) for v in arms.values())
    got, n = apply_sparse_hot_with_count(_as_tensors(arms), hot_h=n_el)
    assert _as_lists(got) == arms
    assert n == 0


def test_k_le_zero_counts_all_prnonzero() -> None:
    arms = {"a": [5, -4, 0, 3]}
    got, n = apply_sparse_hot_with_count(_as_tensors(arms), hot_h=0)
    assert all(v == 0 for row in _as_lists(got).values() for v in row)
    assert n == _oracle_cold_zeros(arms, _as_lists(got)) == 3  # three pre-nonzero


def test_tie_fixture_output_derived_not_numel_minus_k() -> None:
    arms = {"a": [5, -5, 5, -5, 5, -5]}
    hot_h = 3
    got, n = apply_sparse_hot_with_count(_as_tensors(arms), hot_h=hot_h)
    assert n == _oracle_cold_zeros(arms, _as_lists(got))
    # numel-k would be 3; force-zeros among nonzero pre is also 3 here — OK
    assert n == 3
    # Include pre-zeros so numel-k diverges from event count
    arms2 = {"a": [5, 5, 5, 5, 0, 0]}
    got2, n2 = apply_sparse_hot_with_count(_as_tensors(arms2), hot_h=2)
    assert n2 == _oracle_cold_zeros(arms2, _as_lists(got2)) == 2
    assert n2 != (6 - 2)  # not numel-k (=4)


def test_conformance_keeps_above_thresh_count() -> None:
    arms = {"a": [5, 5, 5, 10]}
    got, n = apply_sparse_hot_with_count(_as_tensors(arms), hot_h=2)
    assert _as_lists(got)["a"][3] == 10
    assert n == _oracle_cold_zeros(arms, _as_lists(got)) == 2


def test_non_arm3_omit_key_present_and_zero_distinct() -> None:
    """BINDING 2: non-ARM3 OMIT key; ARM3 present-and-zero ≠ absent."""
    for arm in (ARM0, ARM1, ARM2):
        fields = arm3_sparse_hot_cold_zeros_measurement_fields(arm, 0)
        assert fields == {}
        assert "n_sparse_hot_cold_zeros" not in fields
    arm3_zero = arm3_sparse_hot_cold_zeros_measurement_fields(ARM3, 0)
    assert arm3_zero == {"n_sparse_hot_cold_zeros": 0}
    assert "n_sparse_hot_cold_zeros" in arm3_zero
    assert arm3_sparse_hot_cold_zeros_measurement_fields(ARM3, 9) == {
        "n_sparse_hot_cold_zeros": 9
    }


def test_consumer_pass_through_no_fake_zero() -> None:
    """BINDING 3: rebuild sites must not invent 0 when key absent."""
    base = {
        "measurements": {
            "n_flips": 1,
            "q_changed_count": 1,
            "n_applied_drains": 10,
            "lifetime_censored_frac": 0.0,
            "H_bits_per_weight": 1.0,
        },
        "probes": {"retention_ok": True, "acq_delta_count": 0},
    }
    out_absent = arm_metrics_for_classifier(base)
    assert "n_sparse_hot_cold_zeros" not in out_absent

    with_zero = {
        **base,
        "measurements": {**base["measurements"], "n_sparse_hot_cold_zeros": 0},
    }
    out_zero = arm_metrics_for_classifier(with_zero)
    assert out_zero["n_sparse_hot_cold_zeros"] == 0
    assert "n_sparse_hot_cold_zeros" in out_zero

    with_pos = {
        **base,
        "measurements": {**base["measurements"], "n_sparse_hot_cold_zeros": 42},
    }
    assert arm_metrics_for_classifier(with_pos)["n_sparse_hot_cold_zeros"] == 42
