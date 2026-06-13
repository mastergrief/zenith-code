"""Pure two-tier step orchestrator composing B1–B4 reducers (spec §4.1–§4.2).

Carry self-update runs on every row every step. Selection write-back (post-flip
residual, q flip, applied carry sync) runs on the applied set only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    carry_self_update_row,
    encode_post_flip_residual,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS,
)
from calm.hrm_text_158.native_full_stack.two_tier_transient_selection import (
    LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
    select_by_local_loss_delta,
    validate_two_tier_selector_inputs,
)

WARMUP_APPLY_CLASS_CANONICAL = "canonical"
WARMUP_APPLY_CLASS_SUBTHRESHOLD_BOOTSTRAP = "subthreshold_bootstrap"

ZERO_RESIDUAL_DIRECTION_CANONICAL = 1
ZERO_RESIDUAL_POST_FLIP_PACKED = encode_post_flip_residual(
    ZERO_RESIDUAL_DIRECTION_CANONICAL,
    0,
    threshold_abs=CROSSING_THRESHOLD_ABS,
)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(int(lo), min(int(hi), int(value)))


def _row_mapping(row: Any) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    raise TypeError(f"step row must be a mapping, got {type(row)!r}")


def _ternary_q_after_flip(current_q_level: int, applied_crossing_direction: int) -> int:
    return _clamp(
        int(current_q_level) + int(applied_crossing_direction),
        -1,
        1,
    )


def _post_flip_residual_signed(
    carry_after: int,
    *,
    applied_crossing_direction: int,
    threshold_abs: int = CROSSING_THRESHOLD_ABS,
) -> int:
    """Threshold-minus-one residual band using computed applied crossing direction."""

    direction = int(applied_crossing_direction)
    if direction not in (-1, 1):
        raise ValueError(
            "applied_crossing_direction must be -1 or +1, "
            f"got {applied_crossing_direction}"
        )
    residual = int(carry_after) - direction * int(threshold_abs)
    lo = -int(threshold_abs) + 1
    hi = int(threshold_abs) - 1
    return _clamp(residual, lo, hi)


def _applied_crossing_direction(carry_after: int, *, threshold_abs: int) -> int:
    if int(carry_after) >= int(threshold_abs):
        return 1
    if int(carry_after) <= -int(threshold_abs):
        return -1
    raise ValueError(
        "applied write-back requires crossing carry_after at threshold_abs="
        f"{threshold_abs}, got {carry_after}"
    )


def _assert_applied_subset_of_pre_veto(
    applied_flat_indices: Sequence[int],
    pre_veto_flat_indices: Sequence[int],
) -> None:
    """Fail closed: applied indices must be a duplicate-free subset of pre-veto selection."""

    pre_veto_set = {int(value) for value in pre_veto_flat_indices}
    seen: set[int] = set()
    for flat_index in applied_flat_indices:
        idx = int(flat_index)
        if idx in seen:
            raise ValueError(
                "applied_flat_indices contains duplicate flat_index="
                f"{idx}; pre_veto_flat_indices={tuple(int(v) for v in pre_veto_flat_indices)!r}"
            )
        seen.add(idx)
        if idx not in pre_veto_set:
            raise ValueError(
                "applied_flat_indices flat_index="
                f"{idx} not in pre_veto_flat_indices="
                f"{tuple(int(v) for v in pre_veto_flat_indices)!r}"
            )


def _assert_proposal_direction_agrees_with_crossing(
    row: Mapping[str, Any],
    crossing_direction: int,
) -> None:
    """When row telemetry carries proposal_direction it must match computed crossing."""

    if "proposal_direction" not in row:
        return
    proposal_sign = 1 if int(row["proposal_direction"]) >= 0 else -1
    if proposal_sign != int(crossing_direction):
        raise ValueError(
            "proposal_direction disagrees with computed crossing authority: "
            f"proposal_sign={proposal_sign}, "
            f"computed_crossing_direction={int(crossing_direction)}, "
            f"flat_index={int(row['flat_index'])}"
        )


def _encode_post_flip_residual_for_applied_row(
    residual_signed: int,
    *,
    threshold_abs: int = CROSSING_THRESHOLD_ABS,
) -> int:
    if int(residual_signed) == 0:
        return encode_post_flip_residual(
            ZERO_RESIDUAL_DIRECTION_CANONICAL,
            0,
            threshold_abs=int(threshold_abs),
        )
    direction_sign = -1 if int(residual_signed) < 0 else 1
    return encode_post_flip_residual(
        direction_sign,
        abs(int(residual_signed)),
        threshold_abs=int(threshold_abs),
    )


def validate_two_tier_step_ordering_mode(local_selection_ordering_mode: str) -> None:
    if str(local_selection_ordering_mode) != LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA:
        raise ValueError(
            "unsupported local_selection_ordering_mode "
            f"{local_selection_ordering_mode!r}; expected "
            f"{LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA!r}"
        )


def derive_warmup_apply_tags_from_applied_abs_new_acc(
    applied_abs_new_acc: Sequence[int],
    *,
    threshold_abs: int = CROSSING_THRESHOLD_ABS,
) -> dict[str, Any]:
    """Tag-only warmup classification; does not suppress carry/selection legs."""

    values = [abs(int(value)) for value in applied_abs_new_acc]
    if any(value < int(threshold_abs) for value in values):
        return {
            "warmup_apply_class": WARMUP_APPLY_CLASS_SUBTHRESHOLD_BOOTSTRAP,
            "effective_apply_threshold_abs": max(values) if values else None,
        }
    return {
        "warmup_apply_class": WARMUP_APPLY_CLASS_CANONICAL,
        "effective_apply_threshold_abs": None,
    }


@dataclass(frozen=True)
class TwoTierAppliedWriteBack:
    flat_index: int
    applied_crossing_direction: int
    post_flip_residual_packed: int
    post_accumulator_carry: int
    current_q_level: int


@dataclass(frozen=True)
class TwoTierStepPlan:
    """Pre-veto plan: carry-all-rows state + selection; no write-back."""

    warmup: bool
    local_selection_ordering_mode: str
    threshold_abs: int
    materialized_rows: tuple[dict[str, Any], ...]
    carry_after_by_flat_index: dict[int, int]
    q_level_by_flat_index: dict[int, int]
    pre_veto_flat_indices: tuple[int, ...]


@dataclass(frozen=True)
class TwoTierStepResult:
    warmup: bool
    warmup_apply_class: str
    effective_apply_threshold_abs: int | None
    local_selection_ordering_mode: str
    applied_flat_indices: tuple[int, ...]
    carry_after_by_flat_index: dict[int, int]
    q_level_after_by_flat_index: dict[int, int]
    applied_write_backs: tuple[TwoTierAppliedWriteBack, ...]


def _materialize_step_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    carry_by_flat_index: Mapping[int, int],
    q_level_by_flat_index: Mapping[int, int],
) -> list[dict[str, Any]]:
    materialized_rows: list[dict[str, Any]] = []
    for row in rows:
        mapping = _row_mapping(row)
        flat_index = int(mapping["flat_index"])
        materialized_rows.append(
            {
                **dict(mapping),
                "pre_accumulator_i16": int(
                    carry_by_flat_index.get(
                        flat_index,
                        int(mapping["pre_accumulator_i16"]),
                    )
                ),
                "current_q_level": int(
                    q_level_by_flat_index.get(
                        flat_index,
                        int(mapping["current_q_level"]),
                    )
                ),
            }
        )
    return materialized_rows


def build_two_tier_step_plan_for_apply(
    *,
    carry_after_by_flat_index: Mapping[int, int],
    q_level_by_flat_index: Mapping[int, int],
    pre_veto_flat_indices: Sequence[int],
    materialized_rows: Sequence[Mapping[str, Any]],
    warmup: bool,
    local_selection_ordering_mode: str,
    threshold_abs: int = CROSSING_THRESHOLD_ABS,
) -> TwoTierStepPlan:
    """Compact TwoTierStepPlan for apply write-backs without full-map re-materialization."""

    validate_two_tier_step_ordering_mode(local_selection_ordering_mode)
    return TwoTierStepPlan(
        warmup=bool(warmup),
        local_selection_ordering_mode=str(local_selection_ordering_mode),
        threshold_abs=int(threshold_abs),
        materialized_rows=tuple(dict(row) for row in materialized_rows),
        carry_after_by_flat_index=dict(carry_after_by_flat_index),
        q_level_by_flat_index=dict(q_level_by_flat_index),
        pre_veto_flat_indices=tuple(int(value) for value in pre_veto_flat_indices),
    )


def _snapshot_q_level_by_flat_index(
    materialized_rows: Sequence[Mapping[str, Any]],
    q_level_by_flat_index: Mapping[int, int],
) -> dict[int, int]:
    q_snapshot = {int(flat_index): int(q_level) for flat_index, q_level in q_level_by_flat_index.items()}
    for row in materialized_rows:
        flat_index = int(row["flat_index"])
        if flat_index not in q_snapshot:
            q_snapshot[flat_index] = int(row["current_q_level"])
    return q_snapshot


def plan_two_tier_step(
    rows: Sequence[Mapping[str, Any]],
    *,
    carry_by_flat_index: Mapping[int, int],
    q_level_by_flat_index: Mapping[int, int],
    rate_cap: int,
    warmup: bool,
    local_selection_ordering_mode: str = LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
    threshold_abs: int = CROSSING_THRESHOLD_ABS,
    in_target_tie_band_only: bool = False,
) -> TwoTierStepPlan:
    """§4.1 carry-all-rows + pre-veto selection; no write-back or caller-map mutation."""

    validate_two_tier_step_ordering_mode(local_selection_ordering_mode)
    materialized_rows = _materialize_step_rows(
        rows,
        carry_by_flat_index=carry_by_flat_index,
        q_level_by_flat_index=q_level_by_flat_index,
    )

    failures = validate_two_tier_selector_inputs(materialized_rows, enabled=True)
    if failures:
        raise ValueError("selector input validation failed: " + ",".join(failures))
    if int(rate_cap) < 0:
        raise ValueError(f"rate_cap must be >= 0, got {rate_cap}")

    carry_after_by_flat_index: dict[int, int] = {}
    for row in materialized_rows:
        flat_index = int(row["flat_index"])
        carry_after_by_flat_index[flat_index] = carry_self_update_row(
            int(row["pre_accumulator_i16"]),
            int(row["vote_value"]),
        )

    q_snapshot = _snapshot_q_level_by_flat_index(materialized_rows, q_level_by_flat_index)

    pre_veto_flat_indices = select_by_local_loss_delta(
        materialized_rows,
        rate_cap=int(rate_cap),
        threshold_abs=int(threshold_abs),
        in_target_tie_band_only=bool(in_target_tie_band_only),
    )

    return TwoTierStepPlan(
        warmup=bool(warmup),
        local_selection_ordering_mode=str(local_selection_ordering_mode),
        threshold_abs=int(threshold_abs),
        materialized_rows=tuple(dict(row) for row in materialized_rows),
        carry_after_by_flat_index=dict(carry_after_by_flat_index),
        q_level_by_flat_index=dict(q_snapshot),
        pre_veto_flat_indices=tuple(int(value) for value in pre_veto_flat_indices),
    )


def apply_two_tier_write_backs(
    plan: TwoTierStepPlan,
    applied_flat_indices: Sequence[int],
    *,
    threshold_abs: int | None = None,
) -> TwoTierStepResult:
    """§4.2 write-back on the post-veto applied subset only."""

    resolved_threshold_abs = (
        int(plan.threshold_abs) if threshold_abs is None else int(threshold_abs)
    )

    _assert_applied_subset_of_pre_veto(
        applied_flat_indices,
        plan.pre_veto_flat_indices,
    )

    carry_after_by_flat_index = dict(plan.carry_after_by_flat_index)
    q_level_after_by_flat_index = dict(plan.q_level_by_flat_index)

    applied_write_backs: list[TwoTierAppliedWriteBack] = []
    applied_abs_new_acc: list[int] = []
    rows_by_flat = {int(row["flat_index"]): row for row in plan.materialized_rows}
    for flat_index in applied_flat_indices:
        row = rows_by_flat[int(flat_index)]
        carry_after = int(carry_after_by_flat_index[int(flat_index)])
        applied_abs_new_acc.append(abs(carry_after))
        crossing_direction = _applied_crossing_direction(
            carry_after,
            threshold_abs=resolved_threshold_abs,
        )
        _assert_proposal_direction_agrees_with_crossing(row, crossing_direction)
        residual_signed = _post_flip_residual_signed(
            carry_after,
            applied_crossing_direction=int(crossing_direction),
            threshold_abs=resolved_threshold_abs,
        )
        post_flip_residual_packed = _encode_post_flip_residual_for_applied_row(
            residual_signed,
            threshold_abs=resolved_threshold_abs,
        )
        post_accumulator_carry = int(residual_signed)
        current_q_level = _ternary_q_after_flip(
            int(q_level_after_by_flat_index[int(flat_index)]),
            crossing_direction,
        )
        carry_after_by_flat_index[int(flat_index)] = post_accumulator_carry
        q_level_after_by_flat_index[int(flat_index)] = current_q_level
        applied_write_backs.append(
            TwoTierAppliedWriteBack(
                flat_index=int(flat_index),
                applied_crossing_direction=int(crossing_direction),
                post_flip_residual_packed=int(post_flip_residual_packed),
                post_accumulator_carry=int(post_accumulator_carry),
                current_q_level=int(current_q_level),
            )
        )

    warmup_tags = derive_warmup_apply_tags_from_applied_abs_new_acc(
        applied_abs_new_acc,
        threshold_abs=resolved_threshold_abs,
    )
    return TwoTierStepResult(
        warmup=bool(plan.warmup),
        warmup_apply_class=str(warmup_tags["warmup_apply_class"]),
        effective_apply_threshold_abs=warmup_tags["effective_apply_threshold_abs"],
        local_selection_ordering_mode=str(plan.local_selection_ordering_mode),
        applied_flat_indices=tuple(int(value) for value in applied_flat_indices),
        carry_after_by_flat_index=carry_after_by_flat_index,
        q_level_after_by_flat_index=q_level_after_by_flat_index,
        applied_write_backs=tuple(applied_write_backs),
    )


def run_two_tier_optimizer_step(
    rows: Sequence[Mapping[str, Any]],
    *,
    carry_by_flat_index: Mapping[int, int],
    q_level_by_flat_index: Mapping[int, int],
    rate_cap: int,
    warmup: bool,
    local_selection_ordering_mode: str = LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
    threshold_abs: int = CROSSING_THRESHOLD_ABS,
    in_target_tie_band_only: bool = False,
) -> TwoTierStepResult:
    """Execute one pure two-tier optimizer step (carry all rows → select → write-back applied)."""

    plan = plan_two_tier_step(
        rows,
        carry_by_flat_index=carry_by_flat_index,
        q_level_by_flat_index=q_level_by_flat_index,
        rate_cap=rate_cap,
        warmup=warmup,
        local_selection_ordering_mode=local_selection_ordering_mode,
        threshold_abs=threshold_abs,
        in_target_tie_band_only=in_target_tie_band_only,
    )
    return apply_two_tier_write_backs(
        plan,
        plan.pre_veto_flat_indices,
        threshold_abs=int(threshold_abs),
    )
