"""CPU tests for W8 narrow-accumulator codec seam (lane 1)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.accumulator_real_dynamics_verdict import (
    default_vote_update_spec,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    LIVE_ACC_CARRIER_W8,
    resolve_live_acc_carrier_selector,
)
from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    W8_INT8_EXCLUDED_LOGICAL,
    W8_LOGICAL_MAX,
    W8_LOGICAL_MIN,
    W8_SIGNED_MAX,
    W8_SIGNED_MIN,
    W8_WIDTH_BITS,
    W8NarrowCarrierContractInvalid,
    assert_w8_source_clip_contract,
    clip_then_roundtrip_w8_tensor,
    clip_to_w8,
    clip_to_w8_tensor,
    count_w8_clip_events_tensor,
    decode_w8_logical_lane_from_int8,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    effective_clip_bounds,
    signed_w_max,
)

IN_VIVO_ORACLE_SIDECAR = Path(
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "w7_dense_acc_in_vivo_seed43_43_2189e72008/int16_oracle_flag_off/"
    "headroom_wiring_sidecar.jsonl"
)


def test_w8_fail_closed_source_clip_contract() -> None:
    assert signed_w_max(8) == 127
    assert effective_clip_bounds(8, -127, 127) == (-127, 127)
    spec = default_vote_update_spec()
    assert int(spec.accumulator_clip_min) == -127
    assert int(spec.accumulator_clip_max) == 127
    assert_w8_source_clip_contract()


def test_w8_boundary_values_roundtrip_exactly() -> None:
    boundaries = (-127, -126, 0, 126, 127)
    for value in boundaries:
        assert clip_to_w8(value) == value
        tensor = torch.tensor([value], dtype=torch.int16)
        out = clip_then_roundtrip_w8_tensor(tensor)
        assert int(out.item()) == value


def test_w8_excludes_minus_128_as_logical_lane() -> None:
    assert clip_to_w8(-128) == -127
    assert clip_to_w8(W8_INT8_EXCLUDED_LOGICAL) == -127
    clipped = clip_to_w8_tensor(torch.tensor([-128, 127], dtype=torch.int16))
    assert int(clipped[0]) == -127
    assert int(clipped[1]) == 127
    roundtripped = clip_then_roundtrip_w8_tensor(clipped)
    assert int(roundtripped[0]) == -127
    assert all(int(v) != W8_INT8_EXCLUDED_LOGICAL for v in roundtripped.tolist())


def test_w8_synthetic_boundary_rows_lossless() -> None:
    values = torch.tensor(
        [-127, -126, -1, 0, 1, 126, 127, 120, 87, 33],
        dtype=torch.int16,
    )
    out = clip_then_roundtrip_w8_tensor(values)
    assert torch.equal(out, values)
    assert count_w8_clip_events_tensor(values, out) == 0


def test_w8_clips_values_beyond_source_clip() -> None:
    acc = torch.tensor([200, -200, 127, -127], dtype=torch.int16)
    clipped = clip_then_roundtrip_w8_tensor(acc)
    assert clipped.tolist() == [127, -127, 127, -127]
    assert count_w8_clip_events_tensor(acc, clipped) == 2


@pytest.mark.skipif(not IN_VIVO_ORACLE_SIDECAR.is_file(), reason="2189e72008 read-only artifact absent")
def test_w8_recorded_in_vivo_sidecar_lanes_lossless_roundtrip() -> None:
    recorded: set[int] = set()
    for line in IN_VIVO_ORACLE_SIDECAR.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for lane in row.get("accumulator_lanes") or []:
            recorded.add(int(lane))
    assert recorded, "expected recorded sidecar lane values"
    assert max(abs(v) for v in recorded) <= 127
    tensor = torch.tensor(sorted(recorded), dtype=torch.int16)
    out = clip_then_roundtrip_w8_tensor(tensor)
    assert torch.equal(out, tensor)


def test_w8_strict_ingest_decode_accepts_symmetric_range() -> None:
    for value in (-127, -126, 0, 126, 127):
        assert decode_w8_logical_lane_from_int8(value) == value


def test_w8_strict_ingest_decode_rejects_minus_128() -> None:
    with pytest.raises(W8NarrowCarrierContractInvalid, match="rejects raw int8 -128"):
        decode_w8_logical_lane_from_int8(-128)


def test_w8_strict_ingest_decode_rejects_out_of_logical_range() -> None:
    with pytest.raises(W8NarrowCarrierContractInvalid, match="out of symmetric range"):
        decode_w8_logical_lane_from_int8(128)


def test_w8_selector_returns_w8_label() -> None:
    assert resolve_live_acc_carrier_selector(w8_enabled=True) == LIVE_ACC_CARRIER_W8


def test_w8_mutually_exclusive_with_v4() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_live_acc_carrier_selector(v4_enabled=True, w8_enabled=True)


def test_w8_constants() -> None:
    assert W8_WIDTH_BITS == 8
    assert W8_SIGNED_MIN == -127 and W8_SIGNED_MAX == 127
    assert W8_LOGICAL_MIN == -127 and W8_LOGICAL_MAX == 127


def test_w8_contract_invalid_if_source_clip_widens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.narrow_accumulator_codec.effective_clip_bounds",
        lambda width, lo, hi: (-255, 255),
    )
    with pytest.raises(W8NarrowCarrierContractInvalid, match="effective_clip_bounds"):
        assert_w8_source_clip_contract()
