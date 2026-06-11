from __future__ import annotations

import pytest

from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    carry_self_update_row,
    decode_post_flip_residual,
    encode_post_flip_residual,
)
from calm.hrm_text_158.native_full_stack.two_tier_step_orchestrator import (
    WARMUP_APPLY_CLASS_CANONICAL,
    WARMUP_APPLY_CLASS_SUBTHRESHOLD_BOOTSTRAP,
    ZERO_RESIDUAL_POST_FLIP_PACKED,
    apply_two_tier_write_backs,
    derive_warmup_apply_tags_from_applied_abs_new_acc,
    plan_two_tier_step,
    run_two_tier_optimizer_step,
    validate_two_tier_step_ordering_mode,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS,
)
from calm.hrm_text_158.native_full_stack.two_tier_transient_selection import (
    LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
    select_by_local_loss_delta,
)


def _row(
    candidate_id: str,
    *,
    flat_index: int,
    local_loss_delta: float,
    pre_acc: int,
    vote: int,
    proposal_direction: int = 1,
    q_level: int = 0,
    in_band: bool = True,
) -> dict[str, object]:
    new_acc = carry_self_update_row(pre_acc, vote)
    return {
        "candidate_id": candidate_id,
        "flat_index": flat_index,
        "vote_value": vote,
        "pre_accumulator_i16": pre_acc,
        "new_acc_i32_signed": new_acc,
        "proposal_direction": proposal_direction,
        "current_q_level": q_level,
        "in_target_tie_band": in_band,
        "local_loss_delta": local_loss_delta,
    }


def test_carry_applies_to_all_rows_including_non_selected() -> None:
    rows = [
        _row("selected", flat_index=1, local_loss_delta=-0.9, pre_acc=9, vote=5),
        _row("non-selected", flat_index=2, local_loss_delta=-0.1, pre_acc=2, vote=1),
    ]
    result = run_two_tier_optimizer_step(
        rows,
        carry_by_flat_index={1: 9, 2: 2},
        q_level_by_flat_index={1: 0, 2: 0},
        rate_cap=1,
        warmup=False,
    )
    raw_carry_selected = carry_self_update_row(9, 5)
    assert result.carry_after_by_flat_index[2] == carry_self_update_row(2, 1)
    assert result.carry_after_by_flat_index[1] == raw_carry_selected - CROSSING_THRESHOLD_ABS
    assert result.applied_flat_indices == (1,)


def test_write_back_only_applied_rows() -> None:
    rows = [
        _row("better", flat_index=1, local_loss_delta=-0.9, pre_acc=9, vote=5),
        _row("worse", flat_index=2, local_loss_delta=-0.2, pre_acc=9, vote=5),
    ]
    carry_only_1 = carry_self_update_row(9, 5)
    carry_only_2 = carry_self_update_row(9, 5)
    result = run_two_tier_optimizer_step(
        rows,
        carry_by_flat_index={1: 9, 2: 9},
        q_level_by_flat_index={1: 0, 2: 0},
        rate_cap=1,
        warmup=False,
    )
    assert result.applied_flat_indices == (1,)
    assert result.carry_after_by_flat_index[2] == carry_only_2
    assert result.q_level_after_by_flat_index[2] == 0
    assert result.carry_after_by_flat_index[1] == carry_only_1 - CROSSING_THRESHOLD_ABS
    assert result.q_level_after_by_flat_index[1] == 1
    assert len(result.applied_write_backs) == 1
    assert result.applied_write_backs[0].flat_index == 1


def test_warmup_flag_propagates_without_suppressing_carry_or_selection() -> None:
    rows = [
        _row("warm", flat_index=1, local_loss_delta=-0.5, pre_acc=9, vote=5),
    ]
    result = run_two_tier_optimizer_step(
        rows,
        carry_by_flat_index={1: 9},
        q_level_by_flat_index={1: 0},
        rate_cap=1,
        warmup=True,
    )
    assert result.warmup is True
    assert result.warmup_apply_class == WARMUP_APPLY_CLASS_CANONICAL
    assert result.applied_flat_indices == (1,)


def test_warmup_subthreshold_bootstrap_tag_surface() -> None:
    tags = derive_warmup_apply_tags_from_applied_abs_new_acc([9])
    assert tags["warmup_apply_class"] == WARMUP_APPLY_CLASS_SUBTHRESHOLD_BOOTSTRAP
    assert tags["effective_apply_threshold_abs"] == 9


def test_warmup_tag_canonical_when_applied_abs_meets_threshold() -> None:
    tags = derive_warmup_apply_tags_from_applied_abs_new_acc([CROSSING_THRESHOLD_ABS])
    assert tags["warmup_apply_class"] == WARMUP_APPLY_CLASS_CANONICAL
    assert tags["effective_apply_threshold_abs"] is None


def test_zero_residual_encodes_to_packed_16_via_explicit_encode_path() -> None:
    rows = [
        _row(
            "exact-threshold",
            flat_index=1,
            local_loss_delta=-0.1,
            pre_acc=9,
            vote=1,
            proposal_direction=1,
        ),
    ]
    assert carry_self_update_row(9, 1) == CROSSING_THRESHOLD_ABS
    result = run_two_tier_optimizer_step(
        rows,
        carry_by_flat_index={1: 9},
        q_level_by_flat_index={1: 0},
        rate_cap=1,
        warmup=False,
    )
    assert result.applied_write_backs[0].post_flip_residual_packed == ZERO_RESIDUAL_POST_FLIP_PACKED
    assert ZERO_RESIDUAL_POST_FLIP_PACKED == 16
    assert decode_post_flip_residual(ZERO_RESIDUAL_POST_FLIP_PACKED) == (1, 0)
    assert (
        result.applied_write_backs[0].post_flip_residual_packed
        == encode_post_flip_residual(1, 0, threshold_abs=CROSSING_THRESHOLD_ABS)
    )


def test_negative_crossing_applied_q_flip_and_residual_share_computed_direction() -> None:
    rows = [
        _row(
            "neg-cross",
            flat_index=1,
            local_loss_delta=-0.1,
            pre_acc=-9,
            vote=-2,
            proposal_direction=-1,
            q_level=1,
        ),
    ]
    carry_after = carry_self_update_row(-9, -2)
    assert carry_after <= -CROSSING_THRESHOLD_ABS
    expected_residual = carry_after - (-1) * CROSSING_THRESHOLD_ABS

    result = run_two_tier_optimizer_step(
        rows,
        carry_by_flat_index={1: -9},
        q_level_by_flat_index={1: 1},
        rate_cap=1,
        warmup=False,
    )
    write_back = result.applied_write_backs[0]
    assert write_back.applied_crossing_direction == -1
    assert write_back.post_accumulator_carry == expected_residual
    assert write_back.current_q_level == 0
    assert result.q_level_after_by_flat_index[1] == write_back.current_q_level


def test_mismatched_proposal_direction_fail_closed() -> None:
    rows = [
        _row(
            "mismatch",
            flat_index=1,
            local_loss_delta=-0.1,
            pre_acc=9,
            vote=3,
            proposal_direction=-1,
            q_level=0,
        ),
    ]
    assert carry_self_update_row(9, 3) >= CROSSING_THRESHOLD_ABS
    with pytest.raises(ValueError, match="proposal_direction disagrees with computed crossing authority"):
        run_two_tier_optimizer_step(
            rows,
            carry_by_flat_index={1: 9},
            q_level_by_flat_index={1: 0},
            rate_cap=1,
            warmup=False,
        )


def test_applied_row_without_proposal_direction_uses_computed_crossing_only() -> None:
    row = _row("no-proposal", flat_index=1, local_loss_delta=-0.1, pre_acc=9, vote=5)
    row.pop("proposal_direction")
    result = run_two_tier_optimizer_step(
        [row],
        carry_by_flat_index={1: 9},
        q_level_by_flat_index={1: 0},
        rate_cap=1,
        warmup=False,
    )
    write_back = result.applied_write_backs[0]
    assert write_back.applied_crossing_direction == 1
    assert write_back.post_accumulator_carry == carry_self_update_row(9, 5) - CROSSING_THRESHOLD_ABS


def test_zero_residual_negative_crossing_encodes_canonical_packed_16() -> None:
    rows = [
        _row(
            "neg-zero",
            flat_index=1,
            local_loss_delta=-0.1,
            pre_acc=-9,
            vote=-1,
            proposal_direction=-1,
            q_level=1,
        ),
    ]
    assert carry_self_update_row(-9, -1) == -CROSSING_THRESHOLD_ABS
    result = run_two_tier_optimizer_step(
        rows,
        carry_by_flat_index={1: -9},
        q_level_by_flat_index={1: 1},
        rate_cap=1,
        warmup=False,
    )
    write_back = result.applied_write_backs[0]
    assert write_back.applied_crossing_direction == -1
    assert write_back.post_accumulator_carry == 0
    assert write_back.post_flip_residual_packed == ZERO_RESIDUAL_POST_FLIP_PACKED
    assert ZERO_RESIDUAL_POST_FLIP_PACKED == 16


def test_composition_equals_manual_leg_by_leg_reference() -> None:
    rows = [
        _row("a", flat_index=1, local_loss_delta=-0.4, pre_acc=9, vote=5),
        _row("b", flat_index=2, local_loss_delta=-0.8, pre_acc=9, vote=5),
        _row("c", flat_index=3, local_loss_delta=-0.1, pre_acc=1, vote=1),
    ]
    carry_state = {1: 9, 2: 9, 3: 1}
    q_state = {1: 0, 2: -1, 3: 0}

    manual_carry = {
        flat_index: carry_self_update_row(carry_state[flat_index], int(row["vote_value"]))
        for flat_index, row in ((1, rows[0]), (2, rows[1]), (3, rows[2]))
    }
    manual_applied = select_by_local_loss_delta(rows, rate_cap=1)
    manual_q = dict(q_state)
    manual_acc = dict(manual_carry)
    manual_write_backs: list[tuple[int, int, int]] = []
    for flat_index in manual_applied:
        row = next(row for row in rows if int(row["flat_index"]) == int(flat_index))
        carry_after = manual_carry[int(flat_index)]
        direction = 1 if carry_after >= CROSSING_THRESHOLD_ABS else -1
        residual = carry_after - direction * CROSSING_THRESHOLD_ABS
        packed = encode_post_flip_residual(1, 0) if residual == 0 else encode_post_flip_residual(
            1 if residual > 0 else -1,
            abs(residual),
            threshold_abs=CROSSING_THRESHOLD_ABS,
        )
        manual_q[int(flat_index)] = max(-1, min(1, manual_q[int(flat_index)] + direction))
        manual_acc[int(flat_index)] = residual
        manual_write_backs.append((int(flat_index), packed, residual))

    result = run_two_tier_optimizer_step(
        rows,
        carry_by_flat_index=carry_state,
        q_level_by_flat_index=q_state,
        rate_cap=1,
        warmup=False,
    )
    assert result.applied_flat_indices == manual_applied
    assert result.carry_after_by_flat_index == manual_acc
    assert result.q_level_after_by_flat_index == manual_q
    assert [
        (wb.flat_index, wb.post_flip_residual_packed, wb.post_accumulator_carry)
        for wb in result.applied_write_backs
    ] == manual_write_backs


def test_fail_closed_on_unsupported_ordering_mode() -> None:
    with pytest.raises(ValueError, match="unsupported local_selection_ordering_mode"):
        validate_two_tier_step_ordering_mode("current_abs_new_acc_then_index")


def test_fail_closed_on_bad_selector_rows() -> None:
    rows = [_row("bad", flat_index=1, local_loss_delta=-0.1, pre_acc=9, vote=5)]
    rows[0].pop("local_loss_delta")
    with pytest.raises(ValueError, match="selector input validation failed"):
        run_two_tier_optimizer_step(
            rows,
            carry_by_flat_index={1: 9},
            q_level_by_flat_index={1: 0},
            rate_cap=1,
            warmup=False,
        )


def test_fail_closed_on_negative_rate_cap() -> None:
    rows = [_row("ok", flat_index=1, local_loss_delta=-0.1, pre_acc=9, vote=5)]
    with pytest.raises(ValueError, match="rate_cap must be >= 0"):
        run_two_tier_optimizer_step(
            rows,
            carry_by_flat_index={1: 9},
            q_level_by_flat_index={1: 0},
            rate_cap=-1,
            warmup=False,
        )


def test_required_ordering_mode_constant_surface() -> None:
    assert LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA == (
        "transient_local_loss_delta_then_flat_index"
    )


def test_plan_phase_does_not_mutate_caller_maps_or_write_back() -> None:
    rows = [
        _row("a", flat_index=1, local_loss_delta=-0.4, pre_acc=9, vote=5),
        _row("b", flat_index=2, local_loss_delta=-0.8, pre_acc=9, vote=5),
    ]
    carry_input = {1: 9, 2: 9}
    q_input = {1: 0, 2: -1}
    carry_snapshot = dict(carry_input)
    q_snapshot = dict(q_input)

    plan = plan_two_tier_step(
        rows,
        carry_by_flat_index=carry_input,
        q_level_by_flat_index=q_input,
        rate_cap=1,
        warmup=False,
    )

    assert carry_input == carry_snapshot
    assert q_input == q_snapshot
    assert plan.q_level_by_flat_index == q_snapshot
    assert plan.pre_veto_flat_indices == (2,)
    assert plan.carry_after_by_flat_index[1] == carry_self_update_row(9, 5)
    assert plan.carry_after_by_flat_index[2] == carry_self_update_row(9, 5)
    assert plan.carry_after_by_flat_index[1] == plan.carry_after_by_flat_index[2]


def test_apply_on_subset_writes_back_only_given_indices() -> None:
    rows = [
        _row("a", flat_index=1, local_loss_delta=-0.4, pre_acc=9, vote=5),
        _row("b", flat_index=2, local_loss_delta=-0.8, pre_acc=9, vote=5),
    ]
    plan = plan_two_tier_step(
        rows,
        carry_by_flat_index={1: 9, 2: 9},
        q_level_by_flat_index={1: 0, 2: 0},
        rate_cap=2,
        warmup=False,
    )
    carry_only_1 = plan.carry_after_by_flat_index[1]
    carry_only_2 = plan.carry_after_by_flat_index[2]

    result = apply_two_tier_write_backs(plan, (2,))

    assert result.applied_flat_indices == (2,)
    assert result.carry_after_by_flat_index[1] == carry_only_1
    assert result.q_level_after_by_flat_index[1] == 0
    assert result.carry_after_by_flat_index[2] == carry_only_2 - CROSSING_THRESHOLD_ABS
    assert result.q_level_after_by_flat_index[2] == 1
    assert len(result.applied_write_backs) == 1
    assert result.applied_write_backs[0].flat_index == 2


def test_recomposition_equals_run_two_tier_optimizer_step() -> None:
    rows = [
        _row("a", flat_index=1, local_loss_delta=-0.4, pre_acc=9, vote=5),
        _row("b", flat_index=2, local_loss_delta=-0.8, pre_acc=9, vote=5),
        _row("c", flat_index=3, local_loss_delta=-0.1, pre_acc=1, vote=1),
    ]
    kwargs = {
        "carry_by_flat_index": {1: 9, 2: 9, 3: 1},
        "q_level_by_flat_index": {1: 0, 2: -1, 3: 0},
        "rate_cap": 1,
        "warmup": False,
    }
    composed = run_two_tier_optimizer_step(rows, **kwargs)
    plan = plan_two_tier_step(rows, **kwargs)
    manual = apply_two_tier_write_backs(plan, plan.pre_veto_flat_indices)

    assert manual.applied_flat_indices == composed.applied_flat_indices
    assert manual.carry_after_by_flat_index == composed.carry_after_by_flat_index
    assert manual.q_level_after_by_flat_index == composed.q_level_after_by_flat_index
    assert manual.applied_write_backs == composed.applied_write_backs
    assert manual.warmup_apply_class == composed.warmup_apply_class


def test_fail_closed_surfaces_preserved_through_plan_entry_point() -> None:
    rows = [_row("bad", flat_index=1, local_loss_delta=-0.1, pre_acc=9, vote=5)]
    rows[0].pop("local_loss_delta")
    with pytest.raises(ValueError, match="selector input validation failed"):
        plan_two_tier_step(
            rows,
            carry_by_flat_index={1: 9},
            q_level_by_flat_index={1: 0},
            rate_cap=1,
            warmup=False,
        )


def test_fail_closed_surfaces_preserved_through_apply_entry_point() -> None:
    rows = [
        _row(
            "mismatch",
            flat_index=1,
            local_loss_delta=-0.1,
            pre_acc=9,
            vote=3,
            proposal_direction=-1,
        ),
    ]
    plan = plan_two_tier_step(
        rows,
        carry_by_flat_index={1: 9},
        q_level_by_flat_index={1: 0},
        rate_cap=1,
        warmup=False,
    )
    with pytest.raises(ValueError, match="proposal_direction disagrees with computed crossing authority"):
        apply_two_tier_write_backs(plan, plan.pre_veto_flat_indices)


def test_apply_rejects_crossing_row_outside_pre_veto_flat_indices() -> None:
    rows = [
        _row("selected", flat_index=1, local_loss_delta=-0.9, pre_acc=9, vote=5),
        _row("not-selected", flat_index=2, local_loss_delta=-0.1, pre_acc=9, vote=5),
    ]
    plan = plan_two_tier_step(
        rows,
        carry_by_flat_index={1: 9, 2: 9},
        q_level_by_flat_index={1: 0, 2: 0},
        rate_cap=1,
        warmup=False,
    )
    assert plan.pre_veto_flat_indices == (1,)
    with pytest.raises(ValueError, match="applied_flat_indices flat_index=2 not in pre_veto_flat_indices"):
        apply_two_tier_write_backs(plan, (2,))


def test_apply_rejects_duplicate_applied_flat_indices() -> None:
    rows = [
        _row("a", flat_index=1, local_loss_delta=-0.4, pre_acc=9, vote=5),
    ]
    plan = plan_two_tier_step(
        rows,
        carry_by_flat_index={1: 9},
        q_level_by_flat_index={1: 0},
        rate_cap=1,
        warmup=False,
    )
    with pytest.raises(ValueError, match="applied_flat_indices contains duplicate flat_index=1"):
        apply_two_tier_write_backs(plan, (1, 1))


def test_apply_empty_applied_tuple_is_legal_full_veto() -> None:
    rows = [
        _row("a", flat_index=1, local_loss_delta=-0.4, pre_acc=9, vote=5),
        _row("b", flat_index=2, local_loss_delta=-0.8, pre_acc=9, vote=5),
    ]
    plan = plan_two_tier_step(
        rows,
        carry_by_flat_index={1: 9, 2: 9},
        q_level_by_flat_index={1: 0, 2: -1},
        rate_cap=2,
        warmup=False,
    )
    result = apply_two_tier_write_backs(plan, ())
    assert result.applied_flat_indices == ()
    assert result.applied_write_backs == ()
    assert result.carry_after_by_flat_index == plan.carry_after_by_flat_index
    assert result.q_level_after_by_flat_index == plan.q_level_by_flat_index
