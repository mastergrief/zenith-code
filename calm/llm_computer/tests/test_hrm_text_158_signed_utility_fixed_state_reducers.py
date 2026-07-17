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
        def __init__(self, q, acc=None):
            import torch
            self.q_levels = torch.tensor(q, dtype=torch.int8)
            self.exact_accumulator_shadow = torch.tensor(
                acc if acc is not None else [0] * len(q), dtype=torch.int16
            )
            self.frozen_scale = torch.tensor(1.0)

    base = {"m0": _S([0, 0, 0, 0])}
    prod = {"m0": _S([1, 0, 0, 0])}
    inv_ok = {"m0": _S([-1, 0, 0, 0])}
    inv_bad = {"m0": _S([0, -1, 0, 0])}
    ok = mutation_parity_report(base, prod, inv_ok)
    assert ok["pass"] is True and ok["frozen_scale"]["pass"] is True
    assert "changed_prod" not in ok["q_levels"]["per_key"]["m0"]
    assert mutation_parity_report(base, prod, inv_bad)["pass"] is False


def test_mutation_parity_rank2_production_like_pass_fail():
    class _S:
        def __init__(self, q):
            import torch
            self.q_levels = torch.tensor(q, dtype=torch.int8)
            self.exact_accumulator_shadow = torch.zeros_like(self.q_levels, dtype=torch.int16)
            self.frozen_scale = torch.tensor(1.0)

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
            self.exact_accumulator_shadow = torch.zeros_like(self.q_levels, dtype=torch.int16)
            self.frozen_scale = torch.tensor(1.0)

    base = {"m0": _S([0, 0])}
    with pytest.raises(SignedUtilityReducerError, match="parity_key_mismatch"):
        mutation_parity_report(base, {"m0": _S([1, 0])}, {"m1": _S([-1, 0])})
    with pytest.raises(SignedUtilityReducerError, match="parity_shape_mismatch"):
        mutation_parity_report(
            {"m0": _S([0, 0])},
            {"m0": _S([[1, 0], [0, 0]])},
            {"m0": _S([-1, 0])},
        )


def test_mutation_parity_requires_exact_acc_and_detects_acc_asymmetry():
    class _S:
        def __init__(self, q, acc):
            import torch
            self.q_levels = torch.tensor(q, dtype=torch.int8)
            self.exact_accumulator_shadow = torch.tensor(acc, dtype=torch.int16)
            self.frozen_scale = torch.tensor(1.0)

    base = {"m0": _S([0, 0], [0, 0])}
    prod = {"m0": _S([1, 0], [2, 0])}
    inv_q_ok_acc_bad = {"m0": _S([-1, 0], [0, 3])}
    rep = mutation_parity_report(base, prod, inv_q_ok_acc_bad)
    assert rep["q_levels"]["pass"] is True and rep["exact_accumulator_shadow"]["pass"] is False
    assert rep["pass"] is False


def test_mutation_parity_frozen_scale_byte_identical_no_broadcast():
    class _S:
        def __init__(self, q, scale):
            import torch
            self.q_levels = torch.tensor(q, dtype=torch.int8)
            self.exact_accumulator_shadow = torch.zeros_like(self.q_levels, dtype=torch.int16)
            self.frozen_scale = scale if torch.is_tensor(scale) else torch.tensor(scale)

    import torch
    base = {"m0": _S([0, 0], torch.tensor(1.0))}
    prod = {"m0": _S([1, 0], torch.tensor([1.0]))}
    inv = {"m0": _S([-1, 0], torch.tensor([[1.0]]))}
    # broadcast-equal but shape-differing scales must FAIL
    assert mutation_parity_report(base, prod, inv)["frozen_scale"]["pass"] is False
    ok_scale = torch.tensor(1.0)
    ok = mutation_parity_report(
        {"m0": _S([0, 0], ok_scale)},
        {"m0": _S([1, 0], ok_scale.clone())},
        {"m0": _S([-1, 0], ok_scale.clone())},
    )
    assert ok["frozen_scale"]["pass"] is True
    assert ok["frozen_scale"]["per_key"]["m0"]["base_sha256"] == ok["frozen_scale"]["per_key"]["m0"]["prod_sha256"]


def test_loc_budget():
    assert sum(1 for _ in MOD.open()) <= 200
