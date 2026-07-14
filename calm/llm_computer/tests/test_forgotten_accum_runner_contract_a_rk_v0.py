"""A-RK / A-EFF CPU tests — decisive negatives + real-signature bind."""
from __future__ import annotations

import inspect

import pytest

from calm.hrm_text_158.native_full_stack.forgotten_accum_runner_contract import (
    EXIT_RUN_ARMS_RUNNER_CONTRACT,
    RUNNER_CONTRACT_INVALID,
    RunnerContractRefuse,
    assert_pins_explicit_in_bound,
    assert_three_way_equal,
    bind_real_run_bounded_delta_steps,
    build_forgotten_accum_runner_contract,
    finalize_a_eff_stamp_from_observations,
    pins_from_bound,
)


def test_exit_25_no_collision_with_prior_exits():
    from calm.hrm_text_158.native_full_stack import forgotten_accum_run_arms_launch as launch

    assert EXIT_RUN_ARMS_RUNNER_CONTRACT == 25
    prior = {
        launch.EXIT_RUN_ARMS_NO_AUTHORITY,
        launch.EXIT_RUN_ARMS_PREFLIGHT,
        launch.EXIT_RUN_ARMS_CONTROL_INVALID,
        launch.EXIT_RUN_ARMS_FAILURE,
        launch.EXIT_RUN_ARMS_IDENTITY,
    }
    assert EXIT_RUN_ARMS_RUNNER_CONTRACT not in prior


def test_build_contract_horizon_equals_runway_not_fork_b_32():
    c = build_forgotten_accum_runner_contract(runway_steps=4)
    assert c.global_horizon == 4
    assert c.runway_steps == 4
    assert c.max_abs_per_tensor == 4096
    assert c.r7_deferred_backlog_carry_enabled is True
    assert c.require_q_change is False
    c1500 = build_forgotten_accum_runner_contract(runway_steps=1500)
    assert c1500.global_horizon == 1500


def test_real_signature_bind_requires_explicit_pins():
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import run_bounded_delta_steps

    model = object()
    batch = {}
    states = {}
    eligible = {}
    args = (model, batch, states, eligible)
    base_kw = {
        "device": "cpu",
        "steps": 1,
        "require_q_change": False,
        "max_abs_per_tensor": 4096,
        "global_horizon": 4,
        "global_cap_contract": "c1_banked_faithful_long_run_global_cap",
        "r7_deferred_backlog_carry_enabled": True,
    }
    bound = bind_real_run_bounded_delta_steps(run_bounded_delta_steps, args, base_kw)
    assert_pins_explicit_in_bound(bound)
    pins = pins_from_bound(bound)
    assert pins["global_horizon"] == 4

    # Omit a silent-default pin → must refuse (would inherit OFF/False).
    incomplete = dict(base_kw)
    del incomplete["global_cap_contract"]
    bound2 = bind_real_run_bounded_delta_steps(run_bounded_delta_steps, args, incomplete)
    with pytest.raises(RunnerContractRefuse, match="omitted"):
        assert_pins_explicit_in_bound(bound2)


def test_decisive_negative_cap_resolver_no_spec_mismatches_requested():
    """requested+bound correct but effective resolved_spec absent => RUNNER_CONTRACT_INVALID."""

    contract = build_forgotten_accum_runner_contract(runway_steps=4)
    bound_pins = contract.as_pins_dict()
    effective = finalize_a_eff_stamp_from_observations(
        horizon_obs=[4],
        cap_name_obs=[None],
        cap_resolved_obs=[False],
        max_abs_per_tensor=4096,
        r7_consumed=True,
        require_q_change_consumed=False,
    )
    assert effective["within_arm_consistent"] is True
    with pytest.raises(RunnerContractRefuse, match="resolved_spec_present=False"):
        assert_three_way_equal(
            requested=contract, bound_pins=bound_pins, effective=effective
        )


def test_decisive_negative_r7_consumed_false_vs_requested_true():
    contract = build_forgotten_accum_runner_contract(runway_steps=4)
    bound_pins = contract.as_pins_dict()
    effective = finalize_a_eff_stamp_from_observations(
        horizon_obs=[4],
        cap_name_obs=[contract.global_cap_contract],
        cap_resolved_obs=[True],
        max_abs_per_tensor=4096,
        r7_consumed=False,  # consumed gate false
        require_q_change_consumed=False,
    )
    with pytest.raises(RunnerContractRefuse, match="requested!=effective"):
        assert_three_way_equal(
            requested=contract, bound_pins=bound_pins, effective=effective
        )


def test_three_way_pass_when_a_eff_matches():
    contract = build_forgotten_accum_runner_contract(runway_steps=4)
    bound_pins = contract.as_pins_dict()
    effective = finalize_a_eff_stamp_from_observations(
        horizon_obs=[4],
        cap_name_obs=[contract.global_cap_contract],
        cap_resolved_obs=[True],
        max_abs_per_tensor=4096,
        r7_consumed=True,
        require_q_change_consumed=False,
    )
    out = assert_three_way_equal(
        requested=contract, bound_pins=bound_pins, effective=effective
    )
    assert out["three_way_equality_pass"] is True


def test_fake_runner_cannot_be_conformance_proof():
    """A **kw fake must not be used as bind target — real probe signature required."""

    def fake_runner(*_a, **_kw):
        return None

    sig = inspect.signature(fake_runner)
    assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import run_bounded_delta_steps

    real_sig = inspect.signature(run_bounded_delta_steps)
    assert "global_cap_contract" in real_sig.parameters
    assert real_sig.parameters["global_cap_contract"].default is not inspect.Parameter.empty


def test_runner_contract_invalid_constant():
    assert RUNNER_CONTRACT_INVALID == "RUNNER_CONTRACT_INVALID"
