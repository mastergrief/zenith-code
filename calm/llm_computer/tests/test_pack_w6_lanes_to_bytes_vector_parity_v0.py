"""CPU parity proofs for vectorized W6 bytes-from-lanes packing."""
from __future__ import annotations

import torch

from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    W6_PACK_MASK,
    W6_WIDTH_BITS,
    _pack_w6_lanes_to_bytes_scalar_reference,
    _pack_w6_lanes_to_bytes_vectorized,
    pack_w6_lanes_to_bytes,
    pack_w6_tensor,
    unpack_w6_lanes_from_bytes,
)

# Hand-computed via frozen scalar reference for lanes [-31..-25] (masked 33..39).
GOLDEN_N7_MASKED_LANES = [33, 34, 35, 36, 37, 38, 39]
GOLDEN_N7_BYTES = bytes([161, 56, 146, 165, 121, 2])


def _in_domain_tensor(n: int) -> torch.Tensor:
    values = [((index % 63) - 31) for index in range(n)]
    return torch.tensor(values, dtype=torch.int16)


def _masked_lane_ints(acc: torch.Tensor) -> list[int]:
    lane_tensor = pack_w6_tensor(acc)
    return [int(value) & W6_PACK_MASK for value in lane_tensor.reshape(-1).tolist()]


def _pack_msb_first_wrong_order(lanes: list[int]) -> bytes:
    """Negative control: MSB-first within each lane (must diverge from wire format)."""

    lane_count = len(lanes)
    nbytes = (lane_count * W6_WIDTH_BITS + 7) // 8
    out = bytearray(nbytes)
    bit_pos = 0
    for lane in lanes:
        unsigned_lane = int(lane) & W6_PACK_MASK
        for bit_idx in range(W6_WIDTH_BITS - 1, -1, -1):
            if (unsigned_lane >> bit_idx) & 1:
                byte_idx = bit_pos // 8
                bit_in_byte = bit_pos % 8
                out[byte_idx] |= 1 << bit_in_byte
            bit_pos += 1
    return bytes(out)


def test_scalar_reference_matches_golden_n7_vector() -> None:
    assert _pack_w6_lanes_to_bytes_scalar_reference(GOLDEN_N7_MASKED_LANES) == GOLDEN_N7_BYTES


def test_vectorized_matches_scalar_reference_golden_n7() -> None:
    lanes_u8 = torch.tensor(GOLDEN_N7_MASKED_LANES, dtype=torch.uint8)
    vector_bytes = _pack_w6_lanes_to_bytes_vectorized(lanes_u8).tolist()
    assert bytes(vector_bytes) == GOLDEN_N7_BYTES


def test_pack_w6_lanes_to_bytes_matches_scalar_reference_sweep() -> None:
    edge = torch.tensor([-31, 0, 31], dtype=torch.int16)
    for n in (5, 7, 11, 64, 77, 129):
        acc = _in_domain_tensor(n)
        masked = _masked_lane_ints(acc)
        expected = _pack_w6_lanes_to_bytes_scalar_reference(masked)
        payload = pack_w6_lanes_to_bytes(acc)
        assert payload.packed.tolist() == list(expected)
    masked_edge = _masked_lane_ints(edge)
    expected_edge = _pack_w6_lanes_to_bytes_scalar_reference(masked_edge)
    payload_edge = pack_w6_lanes_to_bytes(edge)
    assert payload_edge.packed.tolist() == list(expected_edge)


def test_msb_first_negative_control_fails_parity() -> None:
    acc = _in_domain_tensor(7)
    masked = _masked_lane_ints(acc)
    correct = _pack_w6_lanes_to_bytes_scalar_reference(masked)
    wrong = _pack_msb_first_wrong_order(masked)
    assert wrong != correct


def test_roundtrip_pack_unpack_identity() -> None:
    for n in (5, 7, 11, 64, 77):
        acc = _in_domain_tensor(n)
        payload = pack_w6_lanes_to_bytes(acc)
        roundtrip = unpack_w6_lanes_from_bytes(payload)
        assert torch.equal(roundtrip, acc.contiguous())
