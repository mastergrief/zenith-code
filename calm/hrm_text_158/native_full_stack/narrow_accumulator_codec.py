"""Int16 vote-acc → 6-bit signed-lane accumulator codec seam (scalar + tensor paths)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.acc_width_recorded_row_sweep import (
    CANONICAL_VOTE_UPDATE_THRESHOLD_ABS,
    DEFAULT_HEADROOM_FACTOR,
    MIN_NON_DEGENERATE_THRESHOLD_ABS,
    VoteSpecParsed,
    build_teacher_forced_applied_candidate_ids,
    headroom_passes,
    load_acc_width_trace_steps,
    replay_width_lane,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    VOTE_UPDATE_SOURCE_CLIP_MAX,
    VOTE_UPDATE_SOURCE_CLIP_MIN,
    carry_self_update_row,
    crossing_bool_w6,
    effective_clip_bounds,
    signed_w_max,
)

W6_WIDTH_BITS = 6
W6_SIGNED_MIN = -signed_w_max(W6_WIDTH_BITS)
W6_SIGNED_MAX = signed_w_max(W6_WIDTH_BITS)
W6_PACK_MASK = (1 << W6_WIDTH_BITS) - 1
W6_PACKED_MIN = 0
W6_PACKED_MAX = W6_PACK_MASK
W6_SIGN_BIT = 1 << (W6_WIDTH_BITS - 1)
W6_SIGN_EXTEND_OFFSET = 1 << W6_WIDTH_BITS


class NarrowCarrierHeadroomBreach(ValueError):
    """Strict W6 narrow-carrier boundary rejection (out-of-domain accumulator lane)."""


CLASSIFIER_S3A_COST_OR_HARNESS_FAIL = "S3A_COST_OR_HARNESS_FAIL"
CLASSIFIER_S3A_PARITY_DIVERGES = "S3A_PARITY_DIVERGES"
CLASSIFIER_S3A_VECTOR_PARITY_OK_COST_BOUNDED = "S3A_VECTOR_PARITY_OK_COST_BOUNDED"

CLASSIFIER_S3A_PRECEDENCE: tuple[str, ...] = (
    CLASSIFIER_S3A_COST_OR_HARNESS_FAIL,
    CLASSIFIER_S3A_PARITY_DIVERGES,
    CLASSIFIER_S3A_VECTOR_PARITY_OK_COST_BOUNDED,
)

S3A_EXPLICIT_NON_CLAIMS: tuple[str, ...] = (
    "cpu_vectorized_w6_codec_prerequisite_proof_only",
    "not_gpu_parity_s3b",
    "not_trainer_boundary_wiring",
    "not_live_training",
    "not_checkpoint_pt_mutation",
    "not_dynamics_stability_full_sub2_readiness",
    "not_physical_sub2",
)

CLASSIFIER_HARNESS_FAIL = "HARNESS_FAIL"
CLASSIFIER_HEADROOM_OR_DOMAIN_FAIL = "HEADROOM_OR_DOMAIN_FAIL"
CLASSIFIER_CODEC_READY_FOR_CPU_PARITY_ONLY = "CODEC_READY_FOR_CPU_PARITY_ONLY"

CLASSIFIER_PRECEDENCE: tuple[str, ...] = (
    CLASSIFIER_HARNESS_FAIL,
    CLASSIFIER_HEADROOM_OR_DOMAIN_FAIL,
    CLASSIFIER_CODEC_READY_FOR_CPU_PARITY_ONLY,
)

EXPLICIT_NON_CLAIMS: tuple[str, ...] = (
    "16_to_6_persistent_carrier_codec_proof_only",
    "not_physical_sub2",
    "not_tensor_wide",
    "not_trainer_integrated",
    "not_gpu",
    "not_dynamics_stability_full_sub2_readiness",
)

FORBIDDEN_CLAIM_FIELDS: frozenset[str] = frozenset(
    {
        "sub2_win",
        "full_sub2_runtime_ready",
        "gpu_launch_authorized",
        "training_claim",
        "stability_claim",
    }
)


def pack_w6(value: int) -> int:
    """Strict pack: fail-closed outside signed W6 domain [-31, 31]."""

    v = int(value)
    if v < W6_SIGNED_MIN or v > W6_SIGNED_MAX:
        raise NarrowCarrierHeadroomBreach(
            f"pack_w6 requires value in [{W6_SIGNED_MIN}, {W6_SIGNED_MAX}], got {value}"
        )
    return v & W6_PACK_MASK


def unpack_w6(packed: int) -> int:
    """Unpack a 6-bit two's-complement lane to signed int (strict fail-closed)."""

    raw = int(packed)
    if raw < W6_PACKED_MIN or raw > W6_PACKED_MAX:
        raise ValueError(
            f"unpack_w6 requires packed in [{W6_PACKED_MIN}, {W6_PACKED_MAX}], got {packed}"
        )
    sign_bit = 1 << (W6_WIDTH_BITS - 1)
    if raw >= sign_bit:
        return raw - (1 << W6_WIDTH_BITS)
    return raw


def clip_to_w6(value: int) -> int:
    """Clamp to effective W6 clip bounds (replay / headroom helper only)."""

    clip_min, clip_max = effective_clip_bounds(
        W6_WIDTH_BITS,
        VOTE_UPDATE_SOURCE_CLIP_MIN,
        VOTE_UPDATE_SOURCE_CLIP_MAX,
    )
    return max(clip_min, min(clip_max, int(value)))


def clip_then_pack_w6(value: int) -> int:
    """Explicit clip-then-pack replay path; separate from strict pack_w6."""

    return pack_w6(clip_to_w6(value))


def pack_w6_tensor(acc: torch.Tensor) -> torch.Tensor:
    """Strict vectorized pack: fail-closed outside signed W6 domain [-31, 31]."""

    if acc.dtype != torch.int16:
        raise ValueError(f"pack_w6_tensor requires torch.int16, got {acc.dtype}")
    values = acc.to(dtype=torch.int32)
    out_of_domain = (values < W6_SIGNED_MIN) | (values > W6_SIGNED_MAX)
    if int(out_of_domain.max()) > 0:
        raise NarrowCarrierHeadroomBreach(
            "pack_w6_tensor requires all values in "
            f"[{W6_SIGNED_MIN}, {W6_SIGNED_MAX}]; "
            f"pack_w6 requires value in [{W6_SIGNED_MIN}, {W6_SIGNED_MAX}]"
        )
    return (values & W6_PACK_MASK).to(torch.int16)


def unpack_w6_tensor(packed: torch.Tensor) -> torch.Tensor:
    """Strict vectorized unpack from 6-bit lanes to signed int16."""

    if packed.dtype != torch.int16:
        raise ValueError(f"unpack_w6_tensor requires torch.int16, got {packed.dtype}")
    values = packed.to(dtype=torch.int32)
    out_of_domain = (values < W6_PACKED_MIN) | (values > W6_PACKED_MAX)
    if int(out_of_domain.max()) > 0:
        raise ValueError(
            f"unpack_w6_tensor requires packed lanes in [{W6_PACKED_MIN}, {W6_PACKED_MAX}]"
        )
    unsigned = values & W6_PACK_MASK
    signed = torch.where(
        unsigned >= W6_SIGN_BIT,
        unsigned - W6_SIGN_EXTEND_OFFSET,
        unsigned,
    )
    return signed.to(torch.int16)


def strict_roundtrip_w6_tensor(acc: torch.Tensor) -> torch.Tensor:
    """Vectorized strict roundtrip using pack_w6_tensor/unpack_w6_tensor."""

    return unpack_w6_tensor(pack_w6_tensor(acc))


def clip_to_w6_tensor(acc: torch.Tensor) -> torch.Tensor:
    """Replay-only clamp to effective W6 clip bounds."""

    clip_min, clip_max = effective_clip_bounds(
        W6_WIDTH_BITS,
        VOTE_UPDATE_SOURCE_CLIP_MIN,
        VOTE_UPDATE_SOURCE_CLIP_MAX,
    )
    return torch.clamp(acc.to(torch.int32), clip_min, clip_max).to(torch.int16)


def clip_then_pack_w6_tensor(acc: torch.Tensor) -> torch.Tensor:
    """Replay-only clip-then-pack; separate from strict pack_w6_tensor."""

    return pack_w6_tensor(clip_to_w6_tensor(acc))


W6_BYTE_PACKED_SCHEMA = "w6_lanes_byte_packed/v0"


@dataclass(frozen=True)
class PackedW6AccumulatorPayload:
    """Physical uint8 byte payload for 6-bit signed accumulator lanes."""

    packed: torch.Tensor
    logical_shape: tuple[int, ...]
    logical_numel: int
    lane_bits: int = W6_WIDTH_BITS
    schema: str = W6_BYTE_PACKED_SCHEMA

    def __post_init__(self) -> None:
        _validate_packed_w6_payload_metadata(self)

    @property
    def packed_data_bytes(self) -> int:
        return int(self.packed.numel() * self.packed.element_size())

    @property
    def expected_packed_data_bytes(self) -> int:
        return int((int(self.logical_numel) * int(self.lane_bits) + 7) // 8)


def _numel_from_shape(shape: Sequence[int]) -> int:
    numel = 1
    for dim in shape:
        if int(dim) < 0:
            raise ValueError(f"logical_shape dims must be non-negative, got {tuple(shape)}")
        numel *= int(dim)
    return int(numel)


def reject_int16_tensor_as_packed_acc(
    value: torch.Tensor,
    *,
    context: str = "accumulator",
) -> None:
    """Fail-closed guard: W6-valued int16 containers are not physical byte packing."""

    if value.dtype == torch.int16:
        raise ValueError(
            f"{context} must be a real uint8 byte payload with schema "
            f"{W6_BYTE_PACKED_SCHEMA!r}, not torch.int16"
        )
    if value.element_size() > 1:
        raise ValueError(
            f"{context} physical payload must be 1-byte elements, got "
            f"element_size={value.element_size()}"
        )


def _validate_packed_w6_payload_metadata(payload: PackedW6AccumulatorPayload) -> None:
    if str(payload.schema) != W6_BYTE_PACKED_SCHEMA:
        raise ValueError(
            f"packed acc schema must be {W6_BYTE_PACKED_SCHEMA!r}, got {payload.schema!r}"
        )
    reject_int16_tensor_as_packed_acc(payload.packed, context="packed acc payload")
    if payload.packed.dtype != torch.uint8:
        raise ValueError(f"packed acc payload must be torch.uint8, got {payload.packed.dtype}")
    if payload.packed.ndim != 1:
        raise ValueError(f"packed acc payload must be 1-D bytes, got shape {tuple(payload.packed.shape)}")
    if int(payload.logical_numel) <= 0:
        raise ValueError("logical_numel must be positive")
    if int(payload.lane_bits) != W6_WIDTH_BITS:
        raise ValueError(f"lane_bits must be {W6_WIDTH_BITS}, got {payload.lane_bits}")
    if _numel_from_shape(payload.logical_shape) != int(payload.logical_numel):
        raise ValueError("logical_shape product must match logical_numel")
    expected_bytes = payload.expected_packed_data_bytes
    if int(payload.packed.numel()) != expected_bytes:
        raise ValueError(
            "packed byte length must equal ceil(logical_numel * lane_bits / 8); "
            f"got {int(payload.packed.numel())}, expected {expected_bytes}"
        )


def _pack_w6_lanes_to_bytes_scalar_reference(lanes: Sequence[int]) -> bytes:
    """Frozen scalar packer: lane bits 0..5, global LSB-first by bit_pos%8."""

    lane_count = len(lanes)
    if lane_count <= 0:
        return b""
    total_bits = int(lane_count * W6_WIDTH_BITS)
    nbytes = (total_bits + 7) // 8
    out = bytearray(nbytes)
    bit_pos = 0
    for lane in lanes:
        unsigned_lane = int(lane) & W6_PACK_MASK
        for bit_idx in range(W6_WIDTH_BITS):
            if (unsigned_lane >> bit_idx) & 1:
                byte_idx = bit_pos // 8
                bit_in_byte = bit_pos % 8
                out[byte_idx] |= 1 << bit_in_byte
            bit_pos += 1
    return bytes(out)


def _pack_w6_lanes_to_bytes_vectorized(lanes_u8: torch.Tensor) -> torch.Tensor:
    """Memory-bounded vector pack: 4 lanes -> 3 bytes, scalar tail for r<=3."""

    lanes_flat = lanes_u8.reshape(-1).contiguous()
    lane_count = int(lanes_flat.numel())
    nbytes = (lane_count * W6_WIDTH_BITS + 7) // 8
    out = torch.zeros(nbytes, dtype=torch.uint8)

    full_lane_count = (lane_count // 4) * 4
    if full_lane_count > 0:
        groups = lanes_flat[:full_lane_count].view(-1, 4).to(torch.int32)
        packed24 = (
            groups[:, 0]
            | (groups[:, 1] << 6)
            | (groups[:, 2] << 12)
            | (groups[:, 3] << 18)
        )
        group_count = full_lane_count // 4
        packed_bytes = torch.empty(group_count * 3, dtype=torch.uint8)
        packed_bytes[0::3] = (packed24 & 0xFF).to(torch.uint8)
        packed_bytes[1::3] = ((packed24 >> 8) & 0xFF).to(torch.uint8)
        packed_bytes[2::3] = ((packed24 >> 16) & 0xFF).to(torch.uint8)
        out[: group_count * 3] = packed_bytes

    tail_count = lane_count - full_lane_count
    if tail_count > 0:
        tail_lanes = lanes_flat[full_lane_count:].tolist()
        tail_bytes = _pack_w6_lanes_to_bytes_scalar_reference(tail_lanes)
        start_byte = full_lane_count * W6_WIDTH_BITS // 8
        out[start_byte : start_byte + len(tail_bytes)] = torch.tensor(
            list(tail_bytes),
            dtype=torch.uint8,
        )
    return out.contiguous()


def pack_w6_lanes_to_bytes(acc: torch.Tensor) -> PackedW6AccumulatorPayload:
    """Pack in-domain int16 lanes into a real uint8 payload of ceil(N*6/8) bytes."""

    lane_tensor = pack_w6_tensor(acc)
    logical_shape = tuple(int(dim) for dim in acc.shape)
    logical_numel = int(acc.numel())
    lanes_u8 = (lane_tensor.reshape(-1) & W6_PACK_MASK).to(torch.uint8)
    packed = _pack_w6_lanes_to_bytes_vectorized(lanes_u8)
    payload = PackedW6AccumulatorPayload(
        packed=packed,
        logical_shape=logical_shape,
        logical_numel=logical_numel,
    )
    _validate_packed_w6_payload_metadata(payload)
    return payload


def unpack_w6_lanes_from_bytes(payload: PackedW6AccumulatorPayload) -> torch.Tensor:
    """Decode a byte-packed W6 payload back to torch.int16 in-step lanes."""

    _validate_packed_w6_payload_metadata(payload)
    packed = payload.packed.to(torch.int64)
    logical_numel = int(payload.logical_numel)
    signed_lanes: list[int] = []
    bit_pos = 0
    for _lane_idx in range(logical_numel):
        unsigned_lane = 0
        for bit_idx in range(W6_WIDTH_BITS):
            byte_idx = bit_pos // 8
            bit_in_byte = bit_pos % 8
            bit = int((packed[byte_idx] >> bit_in_byte) & 1)
            unsigned_lane |= bit << bit_idx
            bit_pos += 1
        signed_lanes.append(unpack_w6(unsigned_lane))
    return torch.tensor(signed_lanes, dtype=torch.int16).view(payload.logical_shape).contiguous()


def default_vote_spec() -> VoteSpecParsed:
    return VoteSpecParsed(
        threshold_abs=CANONICAL_VOTE_UPDATE_THRESHOLD_ABS,
        decay_numerator=1,
        decay_denominator=1,
        accumulator_clip_min=VOTE_UPDATE_SOURCE_CLIP_MIN,
        accumulator_clip_max=VOTE_UPDATE_SOURCE_CLIP_MAX,
    )


def count_codec_crossing_mismatches_on_trace(
    trace_path: Path,
    *,
    vote_spec: VoteSpecParsed | None = None,
) -> tuple[int, list[str]]:
    """A5 helper: crossing_bool_w6 equivalence after clip_then_pack_w6 roundtrip."""

    steps, load_failures = load_acc_width_trace_steps(trace_path)
    if load_failures:
        return 0, list(load_failures)

    spec = vote_spec or default_vote_spec()
    mismatches = 0
    for step in steps:
        for row in step.get("sampled_candidate_table") or ():
            if not isinstance(row, Mapping):
                continue
            pre_acc = int(row["pre_accumulator_i16"])
            vote = int(row["vote_value"])
            q_level = int(row["current_q_level"])
            new_acc = carry_self_update_row(pre_acc, vote, width=W6_WIDTH_BITS)
            original_cross = crossing_bool_w6(new_acc, q_level, threshold_abs=spec.threshold_abs)
            decoded = unpack_w6(clip_then_pack_w6(new_acc))
            codec_cross = crossing_bool_w6(
                decoded,
                q_level,
                threshold_abs=spec.threshold_abs,
            )
            if original_cross != codec_cross:
                mismatches += 1
    return mismatches, []


def max_abs_acc_applied_flips_on_trace(
    trace_path: Path,
    *,
    vote_spec: VoteSpecParsed | None = None,
) -> tuple[int, list[str]]:
    """A6 helper: max |applied flip acc| from width-6 replay lane."""

    steps, load_failures = load_acc_width_trace_steps(trace_path)
    if load_failures:
        return 0, list(load_failures)

    spec = vote_spec or default_vote_spec()
    applied_candidate_ids = build_teacher_forced_applied_candidate_ids(steps)
    lane = replay_width_lane(
        steps,
        vote_spec=spec,
        width=W6_WIDTH_BITS,
        applied_candidate_ids_by_step=applied_candidate_ids,
    )
    return int(lane.get("max_abs_acc_applied_flips", 0)), []


def emit_codec_classifier_receipt(
    *,
    harness_failures: Sequence[str] | None = None,
    width_bits: int = W6_WIDTH_BITS,
    max_abs_acc_applied_flips: int = 0,
    headroom_factor: float = DEFAULT_HEADROOM_FACTOR,
    threshold_abs: int = CANONICAL_VOTE_UPDATE_THRESHOLD_ABS,
    crossing_mismatch_count: int = 0,
    codec_assertions_pass: bool = True,
) -> dict[str, Any]:
    """Emit classifier receipt with explicit non-claims (A7/A8)."""

    failures = list(dict.fromkeys(harness_failures or ()))
    degenerate = (
        int(width_bits) < MIN_NON_DEGENERATE_THRESHOLD_ABS
        or int(threshold_abs) < MIN_NON_DEGENERATE_THRESHOLD_ABS
    )
    if degenerate:
        headroom_ok = False
    else:
        headroom_ok = headroom_passes(
            int(width_bits),
            max_abs_acc_applied=int(max_abs_acc_applied_flips),
            headroom_factor=float(headroom_factor),
        )

    if failures:
        primary = CLASSIFIER_HARNESS_FAIL
    elif (
        degenerate
        or not headroom_ok
        or int(crossing_mismatch_count) > 0
        or not codec_assertions_pass
    ):
        primary = CLASSIFIER_HEADROOM_OR_DOMAIN_FAIL
    else:
        primary = CLASSIFIER_CODEC_READY_FOR_CPU_PARITY_ONLY

    return {
        "slice_id": "narrow_accumulator_codec_cpu_v0",
        "primary_classifier": primary,
        "classifier_precedence": list(CLASSIFIER_PRECEDENCE),
        "width_bits": int(width_bits),
        "signed_domain": [W6_SIGNED_MIN, W6_SIGNED_MAX],
        "headroom_pass": headroom_ok,
        "max_abs_acc_applied_flips": int(max_abs_acc_applied_flips),
        "headroom_factor": float(headroom_factor),
        "crossing_mismatch_count": int(crossing_mismatch_count),
        "harness_failures": failures,
        "explicit_non_claims": list(EXPLICIT_NON_CLAIMS),
        "codec_ready_is_not_sub2": True,
        "codec_ready_is_not_trainer_integrated": True,
        "codec_ready_is_not_gpu_parity": True,
        "codec_ready_is_not_full_sub2_runtime": True,
    }


def emit_s3a_classifier_receipt(
    *,
    harness_failures: Sequence[str] | None = None,
    parity_pass: bool = True,
    static_inspection_pass: bool = True,
    cost_ratio_at_12288: float | None = None,
    cost_ratio_pass: bool = True,
    cost_model_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit S3a classifier receipt with explicit non-claims."""

    failures = list(dict.fromkeys(harness_failures or ()))
    if failures or not static_inspection_pass or not cost_ratio_pass:
        primary = CLASSIFIER_S3A_COST_OR_HARNESS_FAIL
    elif not parity_pass:
        primary = CLASSIFIER_S3A_PARITY_DIVERGES
    else:
        primary = CLASSIFIER_S3A_VECTOR_PARITY_OK_COST_BOUNDED

    return {
        "slice_id": "vectorized_w6_codec_s3a_v0",
        "primary_classifier": primary,
        "classifier_precedence": list(CLASSIFIER_S3A_PRECEDENCE),
        "harness_failures": failures,
        "parity_pass": bool(parity_pass),
        "static_inspection_pass": bool(static_inspection_pass),
        "cost_ratio_at_12288": cost_ratio_at_12288,
        "cost_ratio_pass": bool(cost_ratio_pass),
        "cost_model_receipt": dict(cost_model_receipt or {}),
        "explicit_non_claims": list(S3A_EXPLICIT_NON_CLAIMS),
        "s3a_ok_is_not_gpu_parity": True,
        "s3a_ok_is_not_trainer_wiring": True,
        "s3a_ok_is_not_live_training": True,
    }
