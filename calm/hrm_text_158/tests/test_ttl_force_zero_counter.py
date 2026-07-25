"""CPU characterization: n_ttl_force_zero_drains == oracle drained-count + ARM2 emit surface.

PLAN: artifacts/acc_entropy/arm2_ttl_force_zero_counter_PLAN_v1.json
Bindings: wrapper-with-count; non-ARM2 omit asserted now; no fake-zero consumer defaults.
"""
from __future__ import annotations

import torch

from calm.hrm_text_158.native_full_stack.family_classifier import (
    ARM0,
    ARM1,
    ARM2,
    ARM3,
)
from calm.hrm_text_158.native_full_stack.forgetting_laws import (
    apply_ttl_age_drain,
    apply_ttl_age_drain_with_count,
)
from calm.hrm_text_158.native_full_stack.phase_receipt_contracts import (
    arm_metrics_for_classifier,
)
from calm.hrm_text_158.native_full_stack.screen_receipt_output import (
    arm2_ttl_force_zero_measurement_fields,
)
from calm.hrm_text_158.tests.test_ttl_age_drain_exactness_fence import (
    TTL_PREREG,
    _exact_oracle,
)


def test_wrapper_count_matches_oracle_boundary() -> None:
    step = 200
    ages = [1, 16, 31, 32, 33, 40, 100]
    acc = torch.tensor([7, -5, 3, 9, -2, 4, 1], dtype=torch.int16)
    ep = torch.tensor([step - a for a in ages], dtype=torch.int32)
    want_acc, want_ep, drained = _exact_oracle(
        acc.tolist(), ep.tolist(), step=step, ttl=TTL_PREREG
    )
    got_acc, got_ep, got_n = apply_ttl_age_drain_with_count(
        acc, ep, step=step, ttl=TTL_PREREG
    )
    prim_acc, prim_ep = apply_ttl_age_drain(acc, ep, step=step, ttl=TTL_PREREG)
    assert got_acc.tolist() == want_acc == prim_acc.tolist()
    assert got_ep.tolist() == want_ep == prim_ep.tolist()
    assert got_n == sum(1 for d in drained if d) == 3
    # age==ttl contributes 0; age==ttl+1 contributes 1
    assert drained[3] is False and drained[4] is True


def test_wrapper_count_inactive_zero() -> None:
    step = 10_000
    acc = torch.tensor([11, -9, 5], dtype=torch.int16)
    ep = torch.zeros(3, dtype=torch.int32)
    _, _, got_n = apply_ttl_age_drain_with_count(acc, ep, step=step, ttl=TTL_PREREG)
    assert got_n == 0


def test_wrapper_count_exhaustive_grid_spot_strata() -> None:
    ttl = TTL_PREREG
    mismatches = 0
    cases = 0
    for step in range(0, 80):
        for ep0 in range(0, 80):
            for aval in (-127, -1, 0, 1, 63, 127):
                acc = torch.tensor([aval], dtype=torch.int16)
                ep = torch.tensor([ep0], dtype=torch.int32)
                want_acc, want_ep, drained = _exact_oracle(
                    [aval], [ep0], step=step, ttl=ttl
                )
                got_acc, got_ep, got_n = apply_ttl_age_drain_with_count(
                    acc, ep, step=step, ttl=ttl
                )
                prim_acc, prim_ep = apply_ttl_age_drain(acc, ep, step=step, ttl=ttl)
                cases += 1
                want_n = int(drained[0])
                if (
                    got_acc.tolist() != want_acc
                    or got_ep.tolist() != want_ep
                    or got_n != want_n
                    or got_acc.tolist() != prim_acc.tolist()
                    or got_ep.tolist() != prim_ep.tolist()
                ):
                    mismatches += 1
                    if mismatches <= 5:
                        raise AssertionError(
                            f"mismatch step={step} ep={ep0} aval={aval}: "
                            f"got=({got_acc.tolist()},{got_ep.tolist()},{got_n}) "
                            f"want=({want_acc},{want_ep},{want_n})"
                        )
    assert cases == 80 * 80 * 6
    assert mismatches == 0


def test_non_arm2_omit_key_present_and_zero_distinct() -> None:
    """BINDING 2: non-ARM2 receipts OMIT the key; ARM2 present-and-zero ≠ absent."""
    for arm in (ARM0, ARM1, ARM3):
        fields = arm2_ttl_force_zero_measurement_fields(arm, 0)
        assert fields == {}
        assert "n_ttl_force_zero_drains" not in fields
    arm2_zero = arm2_ttl_force_zero_measurement_fields(ARM2, 0)
    assert arm2_zero == {"n_ttl_force_zero_drains": 0}
    assert "n_ttl_force_zero_drains" in arm2_zero
    arm2_pos = arm2_ttl_force_zero_measurement_fields(ARM2, 7)
    assert arm2_pos == {"n_ttl_force_zero_drains": 7}


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
    assert "n_ttl_force_zero_drains" not in out_absent

    with_zero = {
        **base,
        "measurements": {**base["measurements"], "n_ttl_force_zero_drains": 0},
    }
    out_zero = arm_metrics_for_classifier(with_zero)
    assert out_zero["n_ttl_force_zero_drains"] == 0
    assert "n_ttl_force_zero_drains" in out_zero

    with_pos = {
        **base,
        "measurements": {**base["measurements"], "n_ttl_force_zero_drains": 42},
    }
    assert arm_metrics_for_classifier(with_pos)["n_ttl_force_zero_drains"] == 42
