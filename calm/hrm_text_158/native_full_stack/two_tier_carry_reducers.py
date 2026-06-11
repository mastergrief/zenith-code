"""Pure carry reducers for two-tier accumulator width semantics."""
from __future__ import annotations

VOTE_UPDATE_SOURCE_CLIP_MIN = -127
VOTE_UPDATE_SOURCE_CLIP_MAX = 127
DEFAULT_CARRY_WIDTH = 6
DEFAULT_CROSSING_THRESHOLD_ABS = 10


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
