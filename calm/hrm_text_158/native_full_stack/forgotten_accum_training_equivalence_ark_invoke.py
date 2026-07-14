"""A-RK arm invoke wire: real-signature bind + runner call + three-way effective check.

Split from science_driver to keep that module under the <500 stop threshold.
"""
from __future__ import annotations

from typing import Any

from calm.hrm_text_158.native_full_stack.forgotten_accum_runner_contract import (
    ForgottenAccumRunnerContract,
    assert_horizon_matches_runway,
    assert_three_way_equal,
    bind_real_run_bounded_delta_steps,
    extract_effective_from_runner_return,
    pins_from_bound,
)


def invoke_arm_with_a_rk(
    runner,
    model,
    batch,
    states,
    eligible,
    device,
    steps,
    start_step,
    global_horizon,
    hook,
    backlog,
    flip,
    schedule,
    rk,
    arm,
    log,
    *,
    runner_contract: ForgottenAccumRunnerContract,
    a_rk_receipts: list[dict[str, Any]],
):
    """Assemble kwargs, REAL-signature bind, call runner, A-RK three-way on return."""

    kw = dict(rk)
    kw.update(
        {
            "start_step": int(start_step),
            "global_horizon": int(global_horizon),
            "post_step_hook": hook,
            "initial_deferred_backlog": backlog,
            "flip_application_deferred": bool(flip),
            "flip_application_deferred_schedule": schedule,
        }
    )
    args = (model, batch, states, eligible)
    call_kwargs = dict(device=device, steps=int(steps), **kw)
    # Conformance bind uses the REAL probe signature — never a fake runner's **kw.
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        run_bounded_delta_steps as _real_run_bounded_delta_steps,
    )

    bound = bind_real_run_bounded_delta_steps(
        _real_run_bounded_delta_steps, args, call_kwargs
    )
    bound_pins = pins_from_bound(bound)
    log.append(
        {
            "arm": arm,
            "start_step": int(start_step),
            "steps": int(steps),
            "has_schedule": schedule is not None,
        }
    )
    result = runner(*args, **call_kwargs)
    effective = extract_effective_from_runner_return(result)
    three = assert_three_way_equal(
        requested=runner_contract, bound_pins=bound_pins, effective=effective
    )
    assert_horizon_matches_runway(
        observed_horizon=int(effective["global_horizon"]),
        runway_steps=int(runner_contract.runway_steps),
        arm=str(arm),
    )
    a_rk_receipts.append(
        {
            "arm": arm,
            "bound_pins": bound_pins,
            "effective": {
                k: effective[k]
                for k in (
                    "global_horizon",
                    "global_cap_contract",
                    "global_cap_resolved_spec_present",
                    "max_abs_per_tensor",
                    "r7_deferred_backlog_carry_enabled",
                    "require_q_change",
                )
            },
            "three_way_equality_pass": bool(three["three_way_equality_pass"]),
            "horizon_spy_total_steps_observed": int(effective["global_horizon"]),
        }
    )
    return result


__all__ = ["invoke_arm_with_a_rk"]
