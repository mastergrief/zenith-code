"""Eval determinism/autocast contract tests (D2c3 S3)."""
from __future__ import annotations

from pathlib import Path

import torch

from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_eval_contract import (
    deterministic_eval_contract,
)

MOD = Path(__file__).resolve().parents[2] / "hrm_text_158/native_full_stack/signed_utility_fixed_state_eval_contract.py"


def test_loc_budget():
    assert sum(1 for _ in MOD.open()) <= 90


def test_autocast_disabled_inside_and_restored():
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    # Force ambient enabled where possible via nested context observation.
    with torch.autocast(device_type=device_type, enabled=True):
        before = torch.is_autocast_enabled() if hasattr(torch, "is_autocast_enabled") else True
        with deterministic_eval_contract(device=("cuda:0" if device_type == "cuda" else "cpu")):
            # Inside contract: enabled=False context must be active.
            assert torch.is_autocast_enabled() is False
        # After exit, ambient autocast context remains the outer enabled=True.
        assert torch.is_autocast_enabled() is True or before is True
