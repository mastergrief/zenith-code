"""CPU tests for W7 dense-acc trainer boundary integration."""
from __future__ import annotations

import os

import pytest
import torch

from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    resolve_live_acc_carrier_selector,
)
from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    W7_SIGNED_MAX,
    W7_SIGNED_MIN,
    W7_WIDTH_BITS,
    clip_then_roundtrip_w7_tensor,
    count_w7_clip_events_tensor,
)
from calm.hrm_text_158.native_full_stack.narrow_carrier_trainer_integration import (
    RUN_NARROW_CARRIER_W5_TRAINER_INTEGRATION_ENV,
    RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV,
    RUN_NARROW_CARRIER_W7_TRAINER_INTEGRATION_ENV,
    apply_trainer_boundary_narrow_carrier,
    narrow_carrier_w7_enabled,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import effective_clip_bounds


def test_w7_effective_clip_is_sixty_three() -> None:
    clip_min, clip_max = effective_clip_bounds(7, -127, 127)
    assert clip_min == -63 and clip_max == 63


def test_w7_clip_roundtrip_clips_out_of_domain_values() -> None:
    acc = torch.tensor([70, -70, 33, 9], dtype=torch.int16)
    clipped = clip_then_roundtrip_w7_tensor(acc)
    assert int(clipped[0]) == 63
    assert int(clipped[1]) == -63
    assert int(clipped[2]) == 33
    assert count_w7_clip_events_tensor(acc, clipped) == 2


def test_w7_trainer_boundary_default_off_identity() -> None:
    acc = torch.tensor([33, -31], dtype=torch.int16)
    out = apply_trainer_boundary_narrow_carrier(acc, w7_enabled=False)
    assert torch.equal(out, acc)


def test_w7_trainer_boundary_enabled_clips() -> None:
    acc = torch.tensor([80, -80], dtype=torch.int16)
    out = apply_trainer_boundary_narrow_carrier(acc, w7_enabled=True)
    assert int(out[0]) == W7_SIGNED_MAX
    assert int(out[1]) == W7_SIGNED_MIN
    assert W7_WIDTH_BITS == 7


def test_w7_mutually_exclusive_with_w5_w6(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RUN_NARROW_CARRIER_W7_TRAINER_INTEGRATION_ENV, "1")
    monkeypatch.delenv(RUN_NARROW_CARRIER_W5_TRAINER_INTEGRATION_ENV, raising=False)
    monkeypatch.delenv(RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV, raising=False)
    assert narrow_carrier_w7_enabled() is True
    acc = torch.tensor([10], dtype=torch.int16)
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_trainer_boundary_narrow_carrier(acc, w5_enabled=True, w7_enabled=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_trainer_boundary_narrow_carrier(acc, w6_enabled=True, w7_enabled=True)


def test_w7_mutually_exclusive_with_v4() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_live_acc_carrier_selector(v4_enabled=True, w7_enabled=True)
