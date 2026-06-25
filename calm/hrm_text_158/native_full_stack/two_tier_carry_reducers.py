"""Pure carry reducers for two-tier accumulator width semantics."""
from __future__ import annotations

import numpy as np

VOTE_UPDATE_SOURCE_CLIP_MIN = -127
VOTE_UPDATE_SOURCE_CLIP_MAX = 127
DEFAULT_CARRY_WIDTH = 6
DEFAULT_CROSSING_THRESHOLD_ABS = 10

POST_FLIP_RESIDUAL_ENCODING_NAME = "applied_crossing_direction_plus_4bit_residual"
POST_FLIP_RESIDUAL_PACKED_BIT_WIDTH = 5
POST_FLIP_RESIDUAL_PACKED_MAX = (1 << POST_FLIP_RESIDUAL_PACKED_BIT_WIDTH) - 1
POST_FLIP_RESIDUAL_MAG_MAX_AT_THRESHOLD_10 = DEFAULT_CROSSING_THRESHOLD_ABS - 1

# Spec §4.4 rejects 4-bit signed for post_flip_residual: range is -8..+7 (|.|≤7),
# which cannot hold the threshold_minus_one band magnitude 9 at T=10.
REJECTED_POST_FLIP_RESIDUAL_ENCODING_FOUR_BIT_SIGNED = "4bit_signed"
FOUR_BIT_SIGNED_RESIDUAL_MAX_ABS = 7


def signed_w_max(width: int) -> int:
    if width < 2:
        raise ValueError(f"width must be >= 2, got {width}")
    return (1 << (int(width) - 1)) - 1


def effective_clip_bounds(
    width: int,
    source_clip_min: int,
    source_clip_max: int,
) -> tuple[int, int]:
    """effective_clip(W)=±min(source_clip_max, 2^(W-1)-1) composed with source clip."""

    w_max = signed_w_max(width)
    w_min = -w_max
    eff_max = min(int(source_clip_max), w_max)
    eff_min = max(int(source_clip_min), w_min)
    eff_max = min(eff_max, w_max)
    eff_min = max(eff_min, w_min)
    return eff_min, eff_max


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(int(lo), min(int(hi), int(value)))


def _decay_accumulator(
    acc: int,
    *,
    decay_numerator: int,
    decay_denominator: int,
) -> int:
    if decay_denominator <= 0:
        raise ValueError("decay_denominator must be > 0")
    return (int(acc) * int(decay_numerator)) // int(decay_denominator)


def decay_vote_clamp(
    pre_accumulator: int,
    vote_value: int,
    *,
    clip_min: int,
    clip_max: int,
    decay_numerator: int,
    decay_denominator: int,
) -> int:
    decayed = _decay_accumulator(
        pre_accumulator,
        decay_numerator=decay_numerator,
        decay_denominator=decay_denominator,
    )
    return _clamp(decayed + int(vote_value), clip_min, clip_max)


def crosses_threshold(
    new_acc: int,
    *,
    current_q_level: int,
    threshold_abs: int,
) -> bool:
    q = int(current_q_level)
    acc = int(new_acc)
    threshold = int(threshold_abs)
    return (acc >= threshold and q < 1) or (acc <= -threshold and q > -1)


def effective_clip_w6(
    source_clip_min: int = VOTE_UPDATE_SOURCE_CLIP_MIN,
    source_clip_max: int = VOTE_UPDATE_SOURCE_CLIP_MAX,
) -> tuple[int, int]:
    return effective_clip_bounds(6, source_clip_min, source_clip_max)


def carry_self_update_row(
    pre_acc: int,
    vote: int,
    *,
    width: int = DEFAULT_CARRY_WIDTH,
    source_clip_min: int = VOTE_UPDATE_SOURCE_CLIP_MIN,
    source_clip_max: int = VOTE_UPDATE_SOURCE_CLIP_MAX,
    decay_numerator: int = 1,
    decay_denominator: int = 1,
) -> int:
    clip_min, clip_max = effective_clip_bounds(
        width,
        source_clip_min,
        source_clip_max,
    )
    return decay_vote_clamp(
        pre_acc,
        vote,
        clip_min=clip_min,
        clip_max=clip_max,
        decay_numerator=decay_numerator,
        decay_denominator=decay_denominator,
    )


def crossing_bool_w6(
    new_acc: int,
    q: int,
    *,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
) -> bool:
    return crosses_threshold(
        new_acc,
        current_q_level=int(q),
        threshold_abs=threshold_abs,
    )


def vectorized_carry_self_update_row(
    pre_acc: np.ndarray,
    vote: np.ndarray,
    *,
    width: int = DEFAULT_CARRY_WIDTH,
    source_clip_min: int = VOTE_UPDATE_SOURCE_CLIP_MIN,
    source_clip_max: int = VOTE_UPDATE_SOURCE_CLIP_MAX,
    decay_numerator: int = 1,
    decay_denominator: int = 1,
) -> np.ndarray:
    clip_min, clip_max = effective_clip_bounds(
        width,
        source_clip_min,
        source_clip_max,
    )
    decayed = (pre_acc.astype(np.int64) * int(decay_numerator)) // int(decay_denominator)
    post = decayed + vote.astype(np.int64)
    return np.clip(post, clip_min, clip_max).astype(np.int32)


def vectorized_crosses_threshold(
    new_acc: np.ndarray,
    q: np.ndarray,
    *,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
) -> np.ndarray:
    q_arr = q.astype(np.int32)
    acc = new_acc.astype(np.int32)
    threshold = int(threshold_abs)
    return ((acc >= threshold) & (q_arr < 1)) | ((acc <= -threshold) & (q_arr > -1))


def _validate_post_flip_residual_domain(
    direction_sign: int,
    residual_mag: int,
    *,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
) -> None:
    sign = int(direction_sign)
    mag = int(residual_mag)
    if sign not in (-1, 1):
        raise ValueError(f"direction_sign must be -1 or +1, got {direction_sign}")
    max_mag = int(threshold_abs) - 1
    if mag < 0 or mag > max_mag:
        raise ValueError(
            f"residual_mag must be in [0, {max_mag}] at threshold_abs={threshold_abs}, "
            f"got {residual_mag}"
        )


def encode_post_flip_residual(
    direction_sign: int,
    residual_mag: int,
    *,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
) -> int:
    """Pack direction bit + 4-bit magnitude for post_flip_residual (§4.4)."""

    _validate_post_flip_residual_domain(
        direction_sign,
        residual_mag,
        threshold_abs=threshold_abs,
    )
    direction_bit = 1 if int(direction_sign) > 0 else 0
    return (direction_bit << 4) | (int(residual_mag) & 0xF)


def decode_post_flip_residual(
    packed: int,
    *,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
) -> tuple[int, int]:
    """Unpack (direction_sign, residual_mag) from a 5-bit post_flip_residual code."""

    value = int(packed)
    if value < 0 or value > POST_FLIP_RESIDUAL_PACKED_MAX:
        raise ValueError(
            f"packed post_flip_residual must fit in "
            f"{POST_FLIP_RESIDUAL_PACKED_BIT_WIDTH} bits (0.."
            f"{POST_FLIP_RESIDUAL_PACKED_MAX}), got {packed}"
        )
    direction_sign = 1 if (value >> 4) & 1 else -1
    residual_mag = value & 0xF
    _validate_post_flip_residual_domain(
        direction_sign,
        residual_mag,
        threshold_abs=threshold_abs,
    )
    return direction_sign, residual_mag


def encode_post_flip_residual_from_clamped(
    clamped_residual: int,
    *,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
) -> int:
    """Encode a signed residual already produced by post_flip_residual_clamp."""

    residual = int(clamped_residual)
    lo = -int(threshold_abs) + 1
    hi = int(threshold_abs) - 1
    if residual < lo or residual > hi:
        raise ValueError(
            f"clamped_residual must be in [{lo}, {hi}] at threshold_abs={threshold_abs}, "
            f"got {clamped_residual}"
        )
    direction_sign = -1 if residual < 0 else 1
    return encode_post_flip_residual(
        direction_sign,
        abs(residual),
        threshold_abs=threshold_abs,
    )
