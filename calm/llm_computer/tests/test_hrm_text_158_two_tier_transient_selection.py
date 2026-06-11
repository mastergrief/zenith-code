from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from calm.hrm_text_158.native_full_stack.acc_width_recorded_row_sweep import (
    CANONICAL_VOTE_UPDATE_THRESHOLD_ABS,
    VoteSpecParsed,
    load_acc_width_trace_steps,
)
from calm.hrm_text_158.native_full_stack.transient_selection_information_audit import (
    partition_steps,
    reconstruct_transient_target,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_falsifier_battery import (
    HELD_STEP_END,
    HELD_STEP_START,
    LABEL_SELECTION_MUST_STAY_TRANSIENT_BROAD,
    run_falsifier_battery,
)
from calm.hrm_text_158.native_full_stack.two_tier_transient_selection import (
    FORBIDDEN_PERSIST_SELECTOR_SURFACES,
    LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
    crossing_eligible_flat_indices,
    crossing_eligible_flat_indices_w16_reference,
    rank_eligible_by_transient_score,
    select_by_local_loss_delta,
    select_candidate_ids_by_local_loss_delta,
    transient_score_from_local_loss_delta,
    validate_two_tier_selector_inputs,
)

TRACE1_PATH = Path(
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "b2b_recapture_20260610T145044Z/b2b_seed43/b2b_sequential_trace.ndjson"
)


def _candidate_row(
    candidate_id: str,
    *,
    local_loss_delta: float,
    flat_index: int,
    pre_acc: int = 5,
    vote: int = 1,
    new_acc: int = 12,
    q_level: int = 0,
    in_band: bool = True,
) -> dict[str, object]:
    proposal_direction = 1 if int(new_acc) >= 0 else -1
    threshold = CANONICAL_VOTE_UPDATE_THRESHOLD_ABS
    return {
        "candidate_id": candidate_id,
        "flat_index": flat_index,
        "vote_value": vote,
        "pre_accumulator_i16": pre_acc,
        "new_acc_i32_signed": new_acc,
        "proposal_direction": proposal_direction,
        "current_q_level": q_level,
        "in_target_tie_band": in_band,
        "threshold_residual_signed": int(new_acc) - proposal_direction * threshold,
        "proximity_to_threshold": abs(abs(int(new_acc)) - threshold),
        "current_rank_position": flat_index,
        "local_loss_delta": local_loss_delta,
    }


def _step(step_index: int, rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "optimizer_step_index": step_index,
        "source_kind": "within_tie_band_discriminator",
        "source_table_hash": f"hash-{step_index}",
        "sampled_candidate_table": rows,
    }


def _fixture_stream(*, oracle_prefix: str = "oracle") -> list[dict[str, object]]:
    steps: list[dict[str, object]] = []
    for step_index in range(1, 51):
        oracle_id = f"{oracle_prefix}-{step_index}"
        rows = [
            _candidate_row(
                oracle_id,
                local_loss_delta=-0.95,
                flat_index=1,
                pre_acc=18,
                vote=3,
                new_acc=20,
            ),
            _candidate_row(
                f"decoy-{step_index}",
                local_loss_delta=-0.10,
                flat_index=9,
                pre_acc=4,
                vote=1,
                new_acc=12,
            ),
        ]
        steps.append(_step(step_index, rows))
    return steps


def test_transient_score_is_negated_local_loss_delta() -> None:
    assert transient_score_from_local_loss_delta(-0.5) == 0.5
    assert transient_score_from_local_loss_delta(0.25) == -0.25


def test_reconstruct_transient_target_parity_on_audit_held_split_without_ties() -> None:
    steps = _fixture_stream()
    held_steps = partition_steps(steps)["held_steps"]
    audit_selected, _ = reconstruct_transient_target(held_steps, rate_cap=1)
    reducer_selected = [
        select_candidate_ids_by_local_loss_delta(
            step["sampled_candidate_table"],
            rate_cap=1,
        )
        for step in held_steps
    ]
    assert reducer_selected == audit_selected


def test_tie_break_divergence_class_documents_flat_index_as_forward_normative() -> None:
    """Audit ties on candidate_id; reducer ties on flat_index when delta ties."""

    rows = [
        _candidate_row(
            "aaa",
            local_loss_delta=-0.5,
            flat_index=5,
            pre_acc=9,
            vote=5,
            new_acc=14,
        ),
        _candidate_row(
            "zzz",
            local_loss_delta=-0.5,
            flat_index=2,
            pre_acc=9,
            vote=5,
            new_acc=14,
        ),
    ]
    audit_rank = sorted(rows, key=lambda row: (float(row["local_loss_delta"]), str(row["candidate_id"])))
    reducer_rank = rank_eligible_by_transient_score(rows)
    assert [row["candidate_id"] for row in audit_rank] == ["aaa", "zzz"]
    assert [row["flat_index"] for row in reducer_rank] == [2, 5]
    assert select_candidate_ids_by_local_loss_delta(rows, rate_cap=1) == ("zzz",)
    assert select_by_local_loss_delta(rows, rate_cap=1) == (2,)


def test_non_crossing_better_local_loss_delta_loses_to_crossing_row() -> None:
    rows = [
        _candidate_row(
            "better-non-crossing",
            local_loss_delta=-0.9,
            flat_index=1,
            pre_acc=0,
            vote=1,
            new_acc=1,
        ),
        _candidate_row(
            "worse-crossing",
            local_loss_delta=-0.1,
            flat_index=2,
            pre_acc=9,
            vote=5,
            new_acc=14,
        ),
    ]
    assert crossing_eligible_flat_indices(rows) == [2]
    assert select_by_local_loss_delta(rows, rate_cap=1) == (2,)


def test_validate_two_tier_selector_inputs_fail_closed_when_enabled() -> None:
    rows = [_candidate_row("a", local_loss_delta=-0.1, flat_index=1)]
    assert validate_two_tier_selector_inputs(rows, enabled=True) == []
    assert validate_two_tier_selector_inputs(rows, enabled=False) == []

    missing = [{"flat_index": 1}]
    assert "row_0_missing_local_loss_delta" in validate_two_tier_selector_inputs(missing)

    none_row = [_candidate_row("a", local_loss_delta=-0.1, flat_index=1)]
    none_row[0]["local_loss_delta"] = None
    assert "row_0_local_loss_delta_none" in validate_two_tier_selector_inputs(none_row)

    bad_dtype = [_candidate_row("a", local_loss_delta=-0.1, flat_index=1)]
    bad_dtype[0]["local_loss_delta"] = "nan"
    assert "row_0_local_loss_delta_bad_dtype" in validate_two_tier_selector_inputs(bad_dtype)

    non_finite = [_candidate_row("a", local_loss_delta=-0.1, flat_index=1)]
    non_finite[0]["local_loss_delta"] = float("inf")
    assert "row_0_local_loss_delta_non_finite" in validate_two_tier_selector_inputs(non_finite)


def test_select_by_local_loss_delta_honors_in_target_tie_band_filter() -> None:
    rows = [
        _candidate_row(
            "in",
            local_loss_delta=-0.2,
            flat_index=1,
            pre_acc=9,
            vote=5,
            new_acc=14,
            in_band=True,
        ),
        _candidate_row(
            "out",
            local_loss_delta=-0.9,
            flat_index=2,
            pre_acc=9,
            vote=5,
            new_acc=14,
            in_band=False,
        ),
    ]
    assert select_by_local_loss_delta(rows, rate_cap=1) == (2,)
    assert select_by_local_loss_delta(rows, rate_cap=1, in_target_tie_band_only=True) == (1,)


def test_validate_planned_seam_object_votes_and_local_loss_delta() -> None:
    torch = pytest.importorskip("torch")

    class _SeamInputs:
        def __init__(self, votes: torch.Tensor, local_loss_delta: torch.Tensor) -> None:
            self.votes = votes
            self.local_loss_delta = local_loss_delta

    good = _SeamInputs(
        votes=torch.tensor([1, -2], dtype=torch.int16),
        local_loss_delta=torch.tensor([-0.5, -0.6], dtype=torch.float32),
    )
    assert validate_two_tier_selector_inputs(good, enabled=True) == []
    assert validate_two_tier_selector_inputs(good, enabled=False) == []

    bad_shape = _SeamInputs(
        votes=torch.tensor([1, -2], dtype=torch.int16),
        local_loss_delta=torch.tensor([-0.5], dtype=torch.float32),
    )
    assert "seam_votes_local_loss_delta_shape_mismatch" in validate_two_tier_selector_inputs(
        bad_shape
    )

    bad_votes_dtype = _SeamInputs(
        votes=torch.tensor([-0.1, -0.2], dtype=torch.float32),
        local_loss_delta=torch.tensor([-0.5, -0.6], dtype=torch.float32),
    )
    assert "seam_votes_bad_dtype" in validate_two_tier_selector_inputs(bad_votes_dtype)

    bad_delta_dtype = _SeamInputs(
        votes=torch.tensor([1, -2], dtype=torch.int16),
        local_loss_delta=torch.tensor([-0.5, -0.6], dtype=torch.float64),
    )
    assert "seam_local_loss_delta_bad_dtype" in validate_two_tier_selector_inputs(
        bad_delta_dtype
    )

    non_finite = _SeamInputs(
        votes=torch.tensor([1, -2], dtype=torch.int16),
        local_loss_delta=torch.tensor([float("inf"), -0.6], dtype=torch.float32),
    )
    assert "seam_local_loss_delta_non_finite" in validate_two_tier_selector_inputs(non_finite)


def test_w6_crossing_eligibility_matches_w16_on_recorded_rows() -> None:
    if not TRACE1_PATH.is_file():
        pytest.skip(f"trace-1 fixture missing: {TRACE1_PATH}")

    steps, load_failures = load_acc_width_trace_steps(TRACE1_PATH)
    assert load_failures == []
    mismatches: list[tuple[int, int]] = []
    for step in steps:
        rows = [
            row
            for row in step.get("sampled_candidate_table") or ()
            if isinstance(row, Mapping)
        ]
        w6 = set(crossing_eligible_flat_indices(rows))
        w16 = set(crossing_eligible_flat_indices_w16_reference(rows))
        if w6 != w16:
            mismatches.append((int(step["optimizer_step_index"]), len(w6 ^ w16)))
    assert mismatches == []


@pytest.mark.parametrize(("trace_path",), ((TRACE1_PATH,),))
def test_m2a_trace1_held_rows_align_selection_must_stay_transient_broad_row(
    trace_path: Path,
) -> None:
    if not trace_path.is_file():
        pytest.skip(f"trace-1 fixture missing: {trace_path}")

    steps, load_failures = load_acc_width_trace_steps(trace_path)
    assert load_failures == []
    held_steps = [
        step
        for step in steps
        if HELD_STEP_START <= int(step["optimizer_step_index"]) <= HELD_STEP_END
    ]
    vote_spec = VoteSpecParsed(
        threshold_abs=CANONICAL_VOTE_UPDATE_THRESHOLD_ABS,
        decay_numerator=1,
        decay_denominator=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
    )
    battery = run_falsifier_battery(held_steps, vote_spec=vote_spec)
    assert battery["classifier"]["primary_label"] == LABEL_SELECTION_MUST_STAY_TRANSIENT_BROAD
    assert battery["classifier"]["matched_row"] == 4


def test_forbidden_persist_surfaces_and_ordering_constant_are_frozen() -> None:
    assert LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA == (
        "transient_local_loss_delta_then_flat_index"
    )
    assert "local_loss_delta" in FORBIDDEN_PERSIST_SELECTOR_SURFACES
    assert "rate_cap_queue" in FORBIDDEN_PERSIST_SELECTOR_SURFACES
