"""Partition/leakage pure reducer tests (D2c3 S1)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_partition_leakage import (
    PartitionLeakageError,
    compute_partition_leakage_compact,
    surface_values,
)

MOD = Path(__file__).resolve().parents[2] / "hrm_text_158/native_full_stack/signed_utility_fixed_state_partition_leakage.py"


def _batch(prompt, resp, sep, rid):
    inputs = torch.tensor([prompt + resp], dtype=torch.long)
    labels = torch.tensor([[-100] * sep + resp], dtype=torch.long)
    return {
        "batch": {"inputs": inputs, "labels": labels, "sep_positions": torch.tensor([sep])},
        "metadata": {"row_ids": [rid]},
    }


def test_loc_budget():
    assert sum(1 for _ in MOD.open()) <= 160


def test_same_prompt_different_response_detects_prompt_overlap():
    b0 = _batch([1, 2, 3], [10, 11], 3, "c0")
    b1 = _batch([1, 2, 3], [20, 21], 3, "e0")
    b2 = _batch([9, 9, 9], [30], 3, "e1")
    leak = compute_partition_leakage_compact([b0, b1, b2])
    assert leak["normalized_prompt_hash_overlap"] == 1
    assert leak["pass"] is False


def test_disjoint_prompts_pass():
    b0 = _batch([1, 2, 3], [10], 3, "c0")
    b1 = _batch([4, 5, 6], [20], 3, "e0")
    b2 = _batch([7, 8, 9], [30], 3, "e1")
    leak = compute_partition_leakage_compact([b0, b1, b2])
    assert leak["normalized_prompt_hash_overlap"] == 0
    assert leak["pass"] is True


def test_missing_sep_raises():
    b = {"batch": {"inputs": torch.tensor([[1, 2, 3]])}, "metadata": {"row_ids": ["r"]}}
    with pytest.raises(PartitionLeakageError, match="sep_positions_missing"):
        surface_values(b, "normalized_prompt_hash")
