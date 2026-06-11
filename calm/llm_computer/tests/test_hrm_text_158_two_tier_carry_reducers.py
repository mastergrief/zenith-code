from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from calm.hrm_text_158.native_full_stack.acc_width_recorded_row_sweep import (
    CANONICAL_VOTE_UPDATE_THRESHOLD_ABS,
    VoteSpecParsed,
    load_acc_width_trace_steps,
    replay_width_lane,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    VOTE_UPDATE_SOURCE_CLIP_MAX,
    VOTE_UPDATE_SOURCE_CLIP_MIN,
    carry_self_update_row,
    crossing_bool_w6,
    crosses_threshold,
    decay_vote_clamp,
    effective_clip_bounds,
    effective_clip_w6,
)

TRACE1_PATH = Path(
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "b2b_recapture_20260610T145044Z/b2b_seed43/b2b_sequential_trace.ndjson"
)
CAPTURE2_PATH = Path(
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "b2b_recapture_20260610T204129Z/b2b_seed44/b2b_sequential_trace.ndjson"
)

BASELINE_CAPTURES = (
    pytest.param(
        TRACE1_PATH,
        "cb373de78030c5a9",
        id="trace1",
    ),
    pytest.param(
        CAPTURE2_PATH,
        "34310c423c2ed05c",
        id="capture2",
    ),
)


def _vote_spec() -> VoteSpecParsed:
    return VoteSpecParsed(
        threshold_abs=CANONICAL_VOTE_UPDATE_THRESHOLD_ABS,
        decay_numerator=1,
        decay_denominator=1,
        accumulator_clip_min=VOTE_UPDATE_SOURCE_CLIP_MIN,
        accumulator_clip_max=VOTE_UPDATE_SOURCE_CLIP_MAX,
    )


def _iter_recorded_rows(
    steps: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int, int, int, int]]:
    rows: list[tuple[int, int, int, int, int]] = []
    for step in steps:
        step_index = int(step["optimizer_step_index"])
        for row in step.get("sampled_candidate_table") or ():
            if not isinstance(row, Mapping):
                continue
            rows.append(
                (
                    step_index,
                    int(row["flat_index"]),
                    int(row["pre_accumulator_i16"]),
                    int(row["vote_value"]),
                    int(row["current_q_level"]),
                )
            )
    return rows


def test_decay_vote_clamp_matches_recorded_values_for_fixture_rows() -> None:
    vote_spec = _vote_spec()
    for pre_acc, vote, expected in ((10, 15, 25), (3, 1, 4)):
        assert (
            decay_vote_clamp(
                pre_acc,
                vote,
                clip_min=vote_spec.accumulator_clip_min,
                clip_max=vote_spec.accumulator_clip_max,
                decay_numerator=1,
                decay_denominator=1,
            )
            == expected
        )


@pytest.mark.parametrize("width", (8, 16))
def test_effective_clip_preserves_vote_update_bounds_for_w_ge_8(width: int) -> None:
    assert effective_clip_bounds(
        width,
        VOTE_UPDATE_SOURCE_CLIP_MIN,
        VOTE_UPDATE_SOURCE_CLIP_MAX,
    ) == (VOTE_UPDATE_SOURCE_CLIP_MIN, VOTE_UPDATE_SOURCE_CLIP_MAX)


def test_effective_clip_w6_matches_signed_w6_bounds() -> None:
    assert effective_clip_w6() == (-31, 31)


@pytest.mark.parametrize("width", (8, 16))
def test_carry_self_update_matches_vote_update_reference_for_w_ge_8(width: int) -> None:
    clip_min, clip_max = effective_clip_bounds(
        width,
        VOTE_UPDATE_SOURCE_CLIP_MIN,
        VOTE_UPDATE_SOURCE_CLIP_MAX,
    )
    for pre_acc, vote in ((10, 15), (3, 1), (-120, 5), (100, -50)):
        reference = decay_vote_clamp(
            pre_acc,
            vote,
            clip_min=clip_min,
            clip_max=clip_max,
            decay_numerator=1,
            decay_denominator=1,
        )
        assert carry_self_update_row(pre_acc, vote, width=width) == reference


def test_four_bit_signed_overflow_surfaces_as_clip_not_wrap() -> None:
    clip_min, clip_max = effective_clip_bounds(
        4,
        VOTE_UPDATE_SOURCE_CLIP_MIN,
        VOTE_UPDATE_SOURCE_CLIP_MAX,
    )
    assert (clip_min, clip_max) == (-7, 7)
    clipped = decay_vote_clamp(
        7,
        1,
        clip_min=clip_min,
        clip_max=clip_max,
        decay_numerator=1,
        decay_denominator=1,
    )
    assert clipped == 7
    assert clipped != ((7 + 1) % 16) - 8


@pytest.mark.parametrize(("trace_path", "expected_trace_hash"), BASELINE_CAPTURES)
def test_w6_vs_w16_crossing_equivalence_over_recorded_rows(
    trace_path: Path,
    expected_trace_hash: str,
) -> None:
    if not trace_path.is_file():
        pytest.skip(
            f"baseline trace missing for {expected_trace_hash}: {trace_path}"
        )

    steps, load_failures = load_acc_width_trace_steps(trace_path)
    assert load_failures == []
    assert steps

    vote_spec = _vote_spec()
    lane16 = replay_width_lane(
        steps,
        vote_spec=vote_spec,
        width=16,
        applied_candidate_ids_by_step={},
    )
    lane6 = replay_width_lane(
        steps,
        vote_spec=vote_spec,
        width=6,
        applied_candidate_ids_by_step={},
    )

    mismatches: list[dict[str, int | bool]] = []
    for step_index, flat_index, pre_acc, vote, q_level in _iter_recorded_rows(steps):
        key = (step_index, flat_index)
        w6_new = carry_self_update_row(pre_acc, vote, width=6)
        w16_new = carry_self_update_row(pre_acc, vote, width=16)
        w6_cross = crossing_bool_w6(w6_new, q_level)
        w16_cross = crosses_threshold(
            w16_new,
            current_q_level=q_level,
            threshold_abs=CANONICAL_VOTE_UPDATE_THRESHOLD_ABS,
        )
        if w6_cross != w16_cross:
            mismatches.append(
                {
                    "optimizer_step_index": step_index,
                    "flat_index": flat_index,
                    "w6_cross": w6_cross,
                    "w16_cross": w16_cross,
                }
            )
        assert lane16["row_crossings"][key] == w16_cross
        assert lane6["row_crossings"][key] == w6_cross

    assert mismatches == []
