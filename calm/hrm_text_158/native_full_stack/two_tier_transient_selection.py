"""Pure transient-selection reducers for the two-tier carry/selection boundary.

Returns step-local applied tuples only. Forbidden persist surfaces (packet §4):
- Rank order (F2), cap top-k (F1), argmax abs (F3), rate-cap queue, full 32-row tables
  as persistent authority (spec §4.6).
- Transient reference / outcome scoring fields: local_loss_delta, candidate_loss,
  regret_vs_target_tie_band_oracle_top1_local_loss_delta, current_rank_position.
- Selector telemetry: threshold_residual_signed, proximity_to_threshold.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    carry_self_update_row,
    crossing_bool_w6,
    crosses_threshold,
    effective_clip_w6,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS,
)

LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA = "transient_local_loss_delta_then_flat_index"

FORBIDDEN_PERSIST_SELECTOR_SURFACES: tuple[str, ...] = (
    "rank_order_f2",
    "cap_top_k_f1",
    "argmax_abs_f3",
    "rate_cap_queue",
    "full_candidate_table_persistent_authority",
    "local_loss_delta",
    "candidate_loss",
    "regret_vs_target_tie_band_oracle_top1_local_loss_delta",
    "current_rank_position",
    "threshold_residual_signed",
    "proximity_to_threshold",
)


def transient_score_from_local_loss_delta(local_loss_delta: float) -> float:
    """Pinned convention: lower loss delta is better → score = -delta."""

    return -float(local_loss_delta)


def _row_mapping(row: Any) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    raise TypeError(f"selector row must be a mapping, got {type(row)!r}")


def _extract_local_loss_delta(row: Mapping[str, Any]) -> float:
    if "local_loss_delta" not in row:
        raise ValueError("local_loss_delta missing from selector row")
    value = row["local_loss_delta"]
    if value is None:
        raise ValueError("local_loss_delta must not be None when validation is enabled")
    return float(value)


def _validate_planned_seam_local_loss_delta(
    inputs: Any,
    *,
    require_local_loss_delta: bool,
) -> list[str]:
    failures: list[str] = []
    if not hasattr(inputs, "votes"):
        failures.append("seam_missing_votes")
    if require_local_loss_delta and not hasattr(inputs, "local_loss_delta"):
        failures.append("seam_missing_local_loss_delta")
    if failures:
        return failures

    votes = getattr(inputs, "votes")
    deltas = getattr(inputs, "local_loss_delta", None)
    if deltas is None:
        failures.append("seam_local_loss_delta_none")
        return failures
    if votes is None:
        failures.append("seam_votes_none")
        return failures

    votes_shape = getattr(votes, "shape", None)
    deltas_shape = getattr(deltas, "shape", None)
    if votes_shape is None or deltas_shape is None:
        failures.append("seam_shape_missing")
        return failures
    if tuple(votes_shape) != tuple(deltas_shape):
        failures.append("seam_votes_local_loss_delta_shape_mismatch")
        return failures

    votes_dtype = getattr(votes, "dtype", None)
    deltas_dtype = getattr(deltas, "dtype", None)
    if votes_dtype is None or deltas_dtype is None:
        failures.append("seam_dtype_missing")
        return failures

    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    if torch is not None:
        if votes_dtype != torch.int16:
            failures.append("seam_votes_bad_dtype")
        if deltas_dtype != torch.float32:
            failures.append("seam_local_loss_delta_bad_dtype")
        if not torch.isfinite(deltas).all().item():
            failures.append("seam_local_loss_delta_non_finite")
    else:
        votes_flat = list(getattr(votes, "flatten", lambda: votes)())
        deltas_flat = list(getattr(deltas, "flatten", lambda: deltas)())
        if any(isinstance(value, bool) for value in votes_flat) or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in votes_flat
        ):
            failures.append("seam_votes_bad_dtype")
        if any(isinstance(value, bool) for value in deltas_flat) or not all(
            isinstance(value, float) and not isinstance(value, bool) for value in deltas_flat
        ):
            failures.append("seam_local_loss_delta_bad_dtype")
        elif not all(math.isfinite(float(value)) for value in deltas_flat):
            failures.append("seam_local_loss_delta_non_finite")
    return failures


def _validate_row_sequence_local_loss_delta(
    rows: Sequence[Any],
    *,
    require_local_loss_delta: bool,
) -> list[str]:
    failures: list[str] = []
    if not rows:
        failures.append("selector_rows_empty")
        return failures
    for index, row in enumerate(rows):
        mapping = _row_mapping(row)
        if require_local_loss_delta and "local_loss_delta" not in mapping:
            failures.append(f"row_{index}_missing_local_loss_delta")
            continue
        if "local_loss_delta" not in mapping:
            continue
        value = mapping["local_loss_delta"]
        if value is None:
            failures.append(f"row_{index}_local_loss_delta_none")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            failures.append(f"row_{index}_local_loss_delta_bad_dtype")
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            failures.append(f"row_{index}_local_loss_delta_non_finite")
    return failures


def validate_two_tier_selector_inputs(
    inputs: Any,
    *,
    enabled: bool = True,
    require_local_loss_delta: bool = True,
) -> list[str]:
    """Fail-closed validator for row dicts or the planned B6 votes+delta seam."""

    if not enabled:
        return []
    if hasattr(inputs, "votes") and hasattr(inputs, "local_loss_delta"):
        return _validate_planned_seam_local_loss_delta(
            inputs,
            require_local_loss_delta=require_local_loss_delta,
        )
    if isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes, bytearray)):
        return _validate_row_sequence_local_loss_delta(
            inputs,
            require_local_loss_delta=require_local_loss_delta,
        )
    return ["selector_inputs_unsupported_shape"]


def carry_after_i32_tensor(
    acc_i32: torch.Tensor,
    vote_i32: torch.Tensor,
    *,
    decay_numerator: int = 1,
    decay_denominator: int = 1,
) -> torch.Tensor:
    """Vectorized W6 carry-after matching carry_self_update_row(decay=1/1)."""

    clip_min, clip_max = effective_clip_w6()
    decayed = acc_i32
    if int(decay_numerator) != 1 or int(decay_denominator) != 1:
        decayed = (acc_i32 * int(decay_numerator)) // int(decay_denominator)
    return (decayed + vote_i32).clamp(int(clip_min), int(clip_max))


def crossing_eligible_mask_from_tensors(
    q_i16: torch.Tensor,
    carry_after_i32: torch.Tensor,
    *,
    threshold_abs: int = CROSSING_THRESHOLD_ABS,
) -> torch.Tensor:
    """Tensor mask matching crossing_eligible_flat_indices W6 semantics."""

    threshold = int(threshold_abs)
    return ((carry_after_i32 >= threshold) & (q_i16 < 1)) | (
        (carry_after_i32 <= -threshold) & (q_i16 > -1)
    )


def select_flat_indices_by_local_loss_delta_tensor(
    local_loss_delta: torch.Tensor,
    crossing_mask: torch.Tensor,
    *,
    rate_cap: int,
) -> torch.Tensor:
    """Select up to rate_cap crossing flat indices by (delta, flat_index) ascending."""

    if int(rate_cap) < 0:
        raise ValueError(f"rate_cap must be >= 0, got {rate_cap}")
    crossing_flat = crossing_mask.nonzero(as_tuple=False).flatten().to(torch.int64)
    if crossing_flat.numel() == 0 or int(rate_cap) == 0:
        return crossing_flat[:0]
    deltas_at_crossing = local_loss_delta[crossing_flat]
    flat_order = torch.argsort(crossing_flat, stable=True)
    crossing_flat = crossing_flat[flat_order]
    deltas_at_crossing = deltas_at_crossing[flat_order]
    delta_order = torch.argsort(deltas_at_crossing, stable=True)
    return crossing_flat[delta_order[: int(rate_cap)]]


def crossing_eligible_flat_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold_abs: int = CROSSING_THRESHOLD_ABS,
    width: int = 6,
) -> list[int]:
    """W6 crossing authority for eligibility (normative carry width)."""

    eligible: list[int] = []
    for row in rows:
        flat_index = int(row["flat_index"])
        new_acc = carry_self_update_row(
            int(row["pre_accumulator_i16"]),
            int(row["vote_value"]),
            width=int(width),
        )
        if crossing_bool_w6(
            new_acc,
            int(row["current_q_level"]),
            threshold_abs=int(threshold_abs),
        ):
            eligible.append(flat_index)
    return sorted(eligible)


def rank_eligible_by_transient_score(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Sort by (local_loss_delta, flat_index) ascending — flat_index tie-break."""

    materialized = [dict(row) for row in rows]
    return sorted(
        materialized,
        key=lambda row: (float(row["local_loss_delta"]), int(row["flat_index"])),
    )


def select_by_local_loss_delta(
    rows: Sequence[Mapping[str, Any]],
    *,
    rate_cap: int,
    threshold_abs: int = CROSSING_THRESHOLD_ABS,
    in_target_tie_band_only: bool = False,
) -> tuple[int, ...]:
    """Return step-local applied flat_index tuple; does not persist selector surfaces."""

    failures = validate_two_tier_selector_inputs(rows, enabled=True)
    if failures:
        raise ValueError("selector input validation failed: " + ",".join(failures))
    if int(rate_cap) < 0:
        raise ValueError(f"rate_cap must be >= 0, got {rate_cap}")

    candidates = list(rows)
    if in_target_tie_band_only:
        candidates = [
            row for row in candidates if bool(row.get("in_target_tie_band", False))
        ]
    crossing_eligible = set(
        crossing_eligible_flat_indices(
            candidates,
            threshold_abs=int(threshold_abs),
        )
    )
    candidates = [
        row for row in candidates if int(row["flat_index"]) in crossing_eligible
    ]
    ranked = rank_eligible_by_transient_score(candidates)
    return tuple(int(row["flat_index"]) for row in ranked[: int(rate_cap)])


def select_candidate_ids_by_local_loss_delta(
    rows: Sequence[Mapping[str, Any]],
    *,
    rate_cap: int,
    threshold_abs: int = CROSSING_THRESHOLD_ABS,
    in_target_tie_band_only: bool = False,
) -> tuple[str, ...]:
    """Audit-bridge helper: map flat_index selection back to candidate_id strings."""

    selected_flat = select_by_local_loss_delta(
        rows,
        rate_cap=rate_cap,
        threshold_abs=threshold_abs,
        in_target_tie_band_only=in_target_tie_band_only,
    )
    by_flat = {int(row["flat_index"]): str(row["candidate_id"]) for row in rows}
    return tuple(by_flat[flat_index] for flat_index in selected_flat)


def crossing_eligible_flat_indices_w16_reference(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold_abs: int = CROSSING_THRESHOLD_ABS,
) -> list[int]:
    """Reference W16 crossing membership for equivalence audits only."""

    eligible: list[int] = []
    for row in rows:
        flat_index = int(row["flat_index"])
        new_acc = carry_self_update_row(
            int(row["pre_accumulator_i16"]),
            int(row["vote_value"]),
            width=16,
        )
        if crosses_threshold(
            new_acc,
            current_q_level=int(row["current_q_level"]),
            threshold_abs=int(threshold_abs),
        ):
            eligible.append(flat_index)
    return sorted(eligible)
