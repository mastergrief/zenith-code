"""Arm proof reducer tests (D2c3 S2)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_arm_proofs import (
    ArmProofError,
    PLAN_V6_MUTABLE_ARMS,
    calibrate_capture_vs_public,
    canonical_invert_plans_v4,
    hash_current_weights_tensors,
    mutable_arms_for_isolation,
)

MOD = Path(__file__).resolve().parents[2] / "hrm_text_158/native_full_stack/signed_utility_fixed_state_arm_proofs.py"


@dataclass
class _Plan:
    applied_directions: torch.Tensor
    replay_veto_directions: torch.Tensor
    other: int = 7


def test_loc_budget():
    assert sum(1 for _ in MOD.open()) <= 220


def test_inversion_requires_both_direction_fields():
    @dataclass
    class Incomplete:
        applied_directions: torch.Tensor

    with pytest.raises(ArmProofError, match="invert_missing_direction_field"):
        canonical_invert_plans_v4({"k": Incomplete(torch.tensor([1]))})
    plan = _Plan(torch.tensor([1, -1], dtype=torch.int16), torch.tensor([1, 1], dtype=torch.int16))
    inv = canonical_invert_plans_v4({"k": plan})["k"]
    assert torch.equal(inv.applied_directions, -plan.applied_directions)
    assert torch.equal(inv.replay_veto_directions, -plan.replay_veto_directions)
    assert inv.other == 7


def test_weight_hash_bytes_and_calibration():
    a = {"w": torch.zeros(2)}; b = {"w": torch.zeros(2)}; b["w"][0] = 1.0
    assert hash_current_weights_tensors(a) != hash_current_weights_tensors(b)
    st = lambda: SimpleNamespace(
        q_levels=torch.zeros(2, dtype=torch.int8),
        exact_accumulator_shadow=torch.zeros(2, dtype=torch.int16),
    )
    c, p = {"k": st()}, {"k": st()}
    assert calibrate_capture_vs_public(c, p)["pass"] is True
    p["k"].exact_accumulator_shadow[0] = 2
    assert calibrate_capture_vs_public(c, p)["pass"] is False


def test_mutable_arms_require_calibration_shadow():
    arms = {k: {"q_levels": torch.zeros(1)} for k in PLAN_V6_MUTABLE_ARMS if k != "calibration_shadow"}
    with pytest.raises(ArmProofError, match="isolation_arms_missing"):
        mutable_arms_for_isolation(arms)
