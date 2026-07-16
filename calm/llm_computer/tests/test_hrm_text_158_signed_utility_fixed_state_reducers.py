"""CPU-static tests for signed_utility_fixed_state_reducers (PLAN v5)."""
from __future__ import annotations

from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_reducers import (
    PRIVATE_TRUSTED_CORE,
    SignedUtilityReducerError,
    classify_signed_utility,
    make_raw_front_c_observation_holder_observer,
    mean_nll_f64_from_metrics_loss,
    mutation_parity_report,
    static_private_core_prohibition_pass,
)

MOD = Path(__file__).resolve().parents[2] / "hrm_text_158/native_full_stack/signed_utility_fixed_state_reducers.py"


def test_loc_budget():
    assert sum(1 for _ in MOD.open()) <= 180


def test_classifier_present_null_eps():
    clf, eps = classify_signed_utility(0.5, 0.75, 0.625)
    assert clf == "SIGNED_CREDIT_SIGNAL_PRESENT_UNPROVEN"
    assert eps == 1e-7 * max(1.0, abs(0.625))
    assert classify_signed_utility(1.0, 1.0, 1.0)[0] == "SIGNED_CREDIT_SIGNAL_NULL_OR_HARMFUL"
    assert classify_signed_utility(0.1, 0.2, 0.2)[0] == "SIGNED_CREDIT_SIGNAL_PRESENT_UNPROVEN"


def test_nll_den_lt_1_fail_closed():
    num, den, mean = mean_nll_f64_from_metrics_loss((2.0, 4))
    assert (num, den, mean) == (2.0, 4, 0.5)
    with pytest.raises(SignedUtilityReducerError, match="nll_denominator_lt_1"):
        mean_nll_f64_from_metrics_loss((1.0, 0))


def test_holder_second_call_fail():
    holder, count = [], [0]
    obs = make_raw_front_c_observation_holder_observer(holder, count)
    obs({"plans_by_key": {"k": "p0"}})
    assert count[0] == 1
    with pytest.raises(SignedUtilityReducerError, match="raw_holder_second_call"):
        obs({"plans_by_key": {"k": "p1"}})


def test_private_core_ast_ban():
    src = MOD.read_text(encoding="utf-8")
    assert static_private_core_prohibition_pass(src) is True
    bad = "from x import " + PRIVATE_TRUSTED_CORE + "\n"
    assert static_private_core_prohibition_pass(bad) is False


def test_mutation_parity_pass_fail():
    class _S:
        def __init__(self, q):
            import torch
            self.q_levels = torch.tensor(q, dtype=torch.int8)

    base = {"m0": _S([0, 0, 0, 0])}
    prod = {"m0": _S([1, 0, 0, 0])}
    inv_ok = {"m0": _S([-1, 0, 0, 0])}
    inv_bad = {"m0": _S([0, -1, 0, 0])}  # different changed index => asymmetry
    assert mutation_parity_report(base, prod, inv_ok)["pass"] is True
    assert mutation_parity_report(base, prod, inv_bad)["pass"] is False


def test_mutation_parity_rank2_production_like_pass_fail():
    class _S:
        def __init__(self, q):
            import torch
            self.q_levels = torch.tensor(q, dtype=torch.int8)

    # 2x2 matrix-shaped q (production-like rank>=2)
    base = {"m0": _S([[0, 0], [0, 0]])}
    prod = {"m0": _S([[1, 0], [0, 0]])}
    inv_ok = {"m0": _S([[-1, 0], [0, 0]])}
    inv_bad = {"m0": _S([[0, -1], [0, 0]])}
    assert mutation_parity_report(base, prod, inv_ok)["pass"] is True
    assert mutation_parity_report(base, prod, inv_ok)["per_key"]["m0"]["shape"] == [2, 2]
    assert mutation_parity_report(base, prod, inv_bad)["pass"] is False


def test_mutation_parity_key_and_shape_mismatch_fail_closed():
    class _S:
        def __init__(self, q):
            import torch
            self.q_levels = torch.tensor(q, dtype=torch.int8)

    base = {"m0": _S([0, 0])}
    with pytest.raises(SignedUtilityReducerError, match="parity_key_mismatch"):
        mutation_parity_report(base, {"m0": _S([1, 0])}, {"m1": _S([-1, 0])})
    with pytest.raises(SignedUtilityReducerError, match="parity_shape_mismatch"):
        mutation_parity_report(
            {"m0": _S([0, 0])},
            {"m0": _S([[1, 0], [0, 0]])},
            {"m0": _S([-1, 0])},
        )
