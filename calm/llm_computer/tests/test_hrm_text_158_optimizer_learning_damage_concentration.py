"""Focused tests for optimizer learning-damage concentration audit."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.optimizer_learning_damage_concentration import (
    BRANCH_CONCENTRATED_KEYS,
    BRANCH_DIFFUSE,
    BRANCH_MEASUREMENT_INVALID,
    BRANCH_UNRESOLVED,
    PARENT_SHA_LOCKED_ARM_A_9DB27EE4,
    build_optimizer_learning_damage_concentration_audit_receipt,
    classify_from_parent_receipt_file,
    validate_optimizer_learning_damage_concentration_audit_receipt,
)

ARM_A_PARENT = Path(
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "racc_real_credit_drain_armA_20260616T191949ZZ/receipts/"
    "racc_real_credit_drain_run_receipt.json"
)


def _key_report(*, ucc: int, cold_delta: int, numel: int) -> dict:
    return {
        "numel": numel,
        "pressure_diagnostics": {
            "unapplied_crossing_count": ucc,
            "cold_exception_row_count_delta": cold_delta,
        },
    }


def _step(
    step_id: int,
    *,
    delta: float,
    within: bool,
    keys: dict[str, dict],
) -> dict:
    return {
        "step_id": step_id,
        "ce_proxy_delta_rel": delta,
        "ce_proxy_delta_within_tolerance": within,
        "per_key": keys,
    }


def _parent_from_steps(steps: list[dict], eps: float = 0.01) -> dict:
    return {"ce_proxy_eps_ce": eps, "per_step_reports": steps}


def _classify_parent(parent: dict, parent_sha: str = "a" * 64) -> object:
    from calm.hrm_text_158.native_full_stack.optimizer_learning_damage_concentration import (
        _parse_parent_steps,
    )

    steps, eps = _parse_parent_steps(parent)
    return build_optimizer_learning_damage_concentration_audit_receipt(
        parent_receipt_sha256=parent_sha,
        steps=steps,
        ce_proxy_eps_ce=eps,
    )


@pytest.mark.skipif(not ARM_A_PARENT.is_file(), reason="arm-A parent receipt missing")
def test_live_parent_9db27ee4_integration():
    receipt = classify_from_parent_receipt_file(ARM_A_PARENT)
    validate_optimizer_learning_damage_concentration_audit_receipt(receipt)
    assert receipt.parent_receipt_sha256 == PARENT_SHA_LOCKED_ARM_A_9DB27EE4
    assert receipt.branch_id == BRANCH_DIFFUSE
    assert receipt.fail_step_count == 6
    assert receipt.pass_step_count == 6
    assert receipt.dominant_family_lift_share == 0.0
    assert receipt.dominant_family_rate_lift_share == 0.0
    assert receipt.ready_to_flip is False
    assert receipt.optimizer_credit_state_sub2_claim is False
    assert receipt.readiness_row_flip_authorized is False
    assert receipt.mechanism_built is False
    assert receipt.mint_authority is False


def test_missing_field_measurement_invalid(tmp_path: Path):
    parent = _parent_from_steps(
        [
            _step(
                0,
                delta=0.05,
                within=False,
                keys={
                    "model.H_level.core.layers.0.attn.o_proj": {
                        "pressure_diagnostics": {
                            "unapplied_crossing_count": 1,
                            "cold_exception_row_count_delta": 0,
                        }
                    }
                },
            )
        ]
    )
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(parent), encoding="utf-8")
    receipt = classify_from_parent_receipt_file(path)
    assert receipt.branch_id == BRANCH_MEASUREMENT_INVALID


def test_uniform_size_bias_not_concentrated_keys():
    large = "model.H_level.core.layers.0.mlp.gate_up_proj"
    small = "model.H_level.core.layers.0.attn.o_proj"
    keys_fail = {
        large: _key_report(ucc=1000, cold_delta=0, numel=1_000_000),
        small: _key_report(ucc=100, cold_delta=0, numel=10_000),
    }
    keys_pass = {
        large: _key_report(ucc=1000, cold_delta=0, numel=1_000_000),
        small: _key_report(ucc=100, cold_delta=0, numel=10_000),
    }
    parent = _parent_from_steps(
        [
            _step(0, delta=0.0, within=True, keys=keys_pass),
            _step(1, delta=0.05, within=False, keys=keys_fail),
            _step(2, delta=0.0, within=True, keys=keys_pass),
            _step(3, delta=0.05, within=False, keys=keys_fail),
        ]
    )
    receipt = _classify_parent(parent)
    assert receipt.branch_id != BRANCH_CONCENTRATED_KEYS


def test_proportional_size_bias_not_concentrated_keys():
    large = "model.H_level.core.layers.0.mlp.gate_up_proj"
    small = "model.H_level.core.layers.0.attn.o_proj"
    parent = _parent_from_steps(
        [
            _step(
                0,
                delta=0.0,
                within=True,
                keys={
                    large: _key_report(ucc=900, cold_delta=0, numel=1_000_000),
                    small: _key_report(ucc=90, cold_delta=0, numel=10_000),
                },
            ),
            _step(
                1,
                delta=0.05,
                within=False,
                keys={
                    large: _key_report(ucc=1000, cold_delta=0, numel=1_000_000),
                    small: _key_report(ucc=100, cold_delta=0, numel=10_000),
                },
            ),
            _step(
                2,
                delta=0.0,
                within=True,
                keys={
                    large: _key_report(ucc=900, cold_delta=0, numel=1_000_000),
                    small: _key_report(ucc=90, cold_delta=0, numel=10_000),
                },
            ),
            _step(
                3,
                delta=0.05,
                within=False,
                keys={
                    large: _key_report(ucc=1000, cold_delta=0, numel=1_000_000),
                    small: _key_report(ucc=100, cold_delta=0, numel=10_000),
                },
            ),
        ]
    )
    receipt = _classify_parent(parent)
    assert receipt.branch_id != BRANCH_CONCENTRATED_KEYS


def test_same_family_mismatch_not_concentrated_keys():
    lift_dominant_key = "model.H_level.core.layers.0.attn.o_proj"
    rate_dominant_key = "model.H_level.core.layers.0.mlp.gate_up_proj"
    parent = _parent_from_steps(
        [
            _step(
                0,
                delta=0.0,
                within=True,
                keys={
                    lift_dominant_key: _key_report(ucc=10, cold_delta=0, numel=100_000),
                    rate_dominant_key: _key_report(ucc=0, cold_delta=0, numel=100),
                },
            ),
            _step(
                1,
                delta=0.05,
                within=False,
                keys={
                    lift_dominant_key: _key_report(ucc=110, cold_delta=0, numel=100_000),
                    rate_dominant_key: _key_report(ucc=50, cold_delta=0, numel=100),
                },
            ),
            _step(
                2,
                delta=0.0,
                within=True,
                keys={
                    lift_dominant_key: _key_report(ucc=10, cold_delta=0, numel=100_000),
                    rate_dominant_key: _key_report(ucc=0, cold_delta=0, numel=100),
                },
            ),
            _step(
                3,
                delta=0.05,
                within=False,
                keys={
                    lift_dominant_key: _key_report(ucc=110, cold_delta=0, numel=100_000),
                    rate_dominant_key: _key_report(ucc=50, cold_delta=0, numel=100),
                },
            ),
        ]
    )
    receipt = _classify_parent(parent)
    assert receipt.lift_dominant_family == "H/attn/o_proj"
    assert receipt.rate_dominant_family == "H/mlp/gate_up_proj"
    assert receipt.branch_id != BRANCH_CONCENTRATED_KEYS


def test_single_spike_unresolved_not_concentrated_keys():
    key = "model.H_level.core.layers.0.attn.o_proj"
    parent = _parent_from_steps(
        [
            _step(0, delta=0.0, within=True, keys={key: _key_report(ucc=10, cold_delta=0, numel=1000)}),
            _step(1, delta=0.05, within=False, keys={key: _key_report(ucc=200, cold_delta=0, numel=1000)}),
            _step(2, delta=0.0, within=True, keys={key: _key_report(ucc=10, cold_delta=0, numel=1000)}),
        ]
    )
    receipt = _classify_parent(parent)
    assert receipt.branch_id in {BRANCH_UNRESOLVED, BRANCH_DIFFUSE}
    assert receipt.branch_id != BRANCH_CONCENTRATED_KEYS


def test_middle_zone_family_bucket_unresolved_not_diffuse():
    localized = "model.H_level.core.layers.0.attn.o_proj"
    background = "model.H_level.core.layers.0.mlp.gate_up_proj"
    steps = []
    for step_id in range(10):
        fail = step_id % 2 == 1
        steps.append(
            _step(
                step_id,
                delta=0.05 if fail else 0.0,
                within=not fail,
                keys={
                    localized: _key_report(
                        ucc=40 if fail else 10,
                        cold_delta=0,
                        numel=1_000,
                    ),
                    background: _key_report(
                        ucc=62 if fail else 40,
                        cold_delta=0,
                        numel=1_000,
                    ),
                },
            )
        )
    receipt = _classify_parent(_parent_from_steps(steps))
    assert receipt.fail_step_count == 5
    assert receipt.top1_excess_share <= 0.40
    assert 0.40 < receipt.dominant_family_lift_share < 0.70
    assert receipt.branch_id == BRANCH_UNRESOLVED
    assert receipt.branch_id != BRANCH_DIFFUSE
    assert receipt.branch_id != BRANCH_CONCENTRATED_KEYS


def test_hard_false_fields():
    key = "model.H_level.core.layers.0.attn.o_proj"
    parent = _parent_from_steps(
        [_step(0, delta=0.05, within=False, keys={key: _key_report(ucc=1, cold_delta=0, numel=1000)})]
    )
    receipt = _classify_parent(parent)
    assert receipt.ready_to_flip is False
    assert receipt.optimizer_credit_state_sub2_claim is False
    assert receipt.readiness_row_flip_authorized is False
    assert receipt.mechanism_built is False
    assert receipt.mint_authority is False
