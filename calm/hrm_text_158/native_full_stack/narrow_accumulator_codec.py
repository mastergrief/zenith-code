"""CPU-only int16 vote-acc → 6-bit signed-lane accumulator codec seam."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

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
        raise ValueError(
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
