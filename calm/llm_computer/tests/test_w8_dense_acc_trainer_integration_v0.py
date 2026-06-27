"""CPU tests for W8 dense-acc trainer boundary materialization (lane 2)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    LIVE_ACC_CARRIER_W8,
    resolve_live_acc_carrier_selector,
)
from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    W8_SIGNED_MAX,
    W8_SIGNED_MIN,
    W8_WIDTH_BITS,
    clip_then_roundtrip_w8_tensor,
    count_w8_clip_events_tensor,
)
from calm.hrm_text_158.native_full_stack.narrow_carrier_trainer_integration import (
    RUN_NARROW_CARRIER_W8_TRAINER_INTEGRATION_ENV,
    apply_trainer_boundary_narrow_carrier,
    narrow_carrier_w8_enabled,
)

IN_VIVO_ORACLE_SIDECAR = Path(
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "w7_dense_acc_in_vivo_seed43_43_2189e72008/int16_oracle_flag_off/"
    "headroom_wiring_sidecar.jsonl"
)


def test_w8_selector_returns_w8_carrier_when_enabled() -> None:
    assert resolve_live_acc_carrier_selector(w8_enabled=True) == LIVE_ACC_CARRIER_W8


def test_w8_trainer_boundary_default_off_identity() -> None:
    acc = torch.tensor([120, -96, 127], dtype=torch.int16)
    out = apply_trainer_boundary_narrow_carrier(acc, w8_enabled=False)
    assert torch.equal(out, acc)


def test_w8_trainer_boundary_enabled_in_domain_identity() -> None:
    acc = torch.tensor([-127, -126, 0, 126, 127, 120, 87, 33], dtype=torch.int16)
    out = apply_trainer_boundary_narrow_carrier(acc, w8_enabled=True)
    assert torch.equal(out, acc)
    assert count_w8_clip_events_tensor(acc, out) == 0


def test_w8_trainer_boundary_enabled_out_of_domain_matches_clipped_reference() -> None:
    acc = torch.tensor([200, -200], dtype=torch.int16)
    expected = clip_then_roundtrip_w8_tensor(acc)
    out = apply_trainer_boundary_narrow_carrier(acc, w8_enabled=True)
    assert torch.equal(out, expected)
    assert out.tolist() == [W8_SIGNED_MAX, W8_SIGNED_MIN]
    assert not torch.equal(out, acc)


@pytest.mark.skipif(not IN_VIVO_ORACLE_SIDECAR.is_file(), reason="2189e72008 read-only artifact absent")
def test_w8_trainer_boundary_recorded_oracle_lanes_bit_identical() -> None:
    recorded: set[int] = set()
    for line in IN_VIVO_ORACLE_SIDECAR.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for lane in row.get("accumulator_lanes") or []:
            recorded.add(int(lane))
    assert recorded
    assert max(abs(v) for v in recorded) <= 127
    tensor = torch.tensor(sorted(recorded), dtype=torch.int16)
    out = apply_trainer_boundary_narrow_carrier(tensor, w8_enabled=True)
    assert torch.equal(out, tensor)


def test_w8_trainer_boundary_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RUN_NARROW_CARRIER_W8_TRAINER_INTEGRATION_ENV, "1")
    assert narrow_carrier_w8_enabled() is True
    acc = torch.tensor([127, -127], dtype=torch.int16)
    out = apply_trainer_boundary_narrow_carrier(acc)
    assert torch.equal(out, acc)
    assert W8_WIDTH_BITS == 8


def test_w8_mutually_exclusive_with_w5_w6_w7() -> None:
    acc = torch.tensor([10], dtype=torch.int16)
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_trainer_boundary_narrow_carrier(acc, w5_enabled=True, w8_enabled=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_trainer_boundary_narrow_carrier(acc, w6_enabled=True, w8_enabled=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_trainer_boundary_narrow_carrier(acc, w7_enabled=True, w8_enabled=True)


def test_w8_mutually_exclusive_with_v4() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_live_acc_carrier_selector(v4_enabled=True, w8_enabled=True)
