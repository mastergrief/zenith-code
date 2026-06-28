"""Offline CPU decay-grid acc-bpw sizing core for D recompute-window horizon growth.

Envelope construction (conservative upper bound)
------------------------------------------------
Slice 4 does **not** call `EventCodedAccLiveState.apply_step` or
`apply_event_coded_carrier_step` — the live carrier hardcodes default decay and
would make the decay grid decorative.

Instead, for each decay-grid point `(decay_num, decay_den)` and sized window `K`,
we build explicit checkpoint byte payloads via `pack_event_coded_acc_checkpoint_v1`
and measure them with `measure_r4v_event_coded_acc_budget`.

Per grid point:
1. **Decay-parameterized surface** — simulate `K` worst-case steps with the grid
   decay applied as `(acc * decay_num) // decay_den + vote` (clipped), rotating
   lanes with alternating `±1` votes to maximize carry churn.
2. **Event log upper bound** — one crossing event per step (max in-K density),
   with `residual_mag=15` and `event_type=1`.
3. **Index adversaries** — encode the same event-count bound twice:
   - dense low-index (`flat_index = step % numel`)
   - high-index varint adversary (`flat_index = numel - 1`)
   and take whichever yields **more** event bytes (varint width + flags).
4. **Hot/backlog** — from the decay simulation: hot rows for
   `|carry| >= promote_threshold`, backlog varints for remaining nonzero lanes.
   High-index adversary duplicates hot/backlog indices at `numel - 1` when that
   weakens the bound.

The chosen envelope is a conservative **envelope model** over (K, decay, numel):
monotone in retention and index-adversary-dominant within this synthetic builder.
It is **not** a proved absolute upper bound over multi-lane crossing density,
global-cap accepted density, or live-carrier backlog/hot surfaces. In-vivo sub-2
validation is slice 5's job (real logs/cap/manifests).

Budget note
-----------
`effective_acc_budget_bpw = min(ACC_BUDGET_BPW_UNDER_BASE3_Q,
TARGET_PHYSICAL_BITS_PER_WEIGHT - measured_q_scale_bpw)` is **tighter** than
`sub2_carrier_family_discriminator.dual_boolean_record` line ~147, which checks
only `inclusive_acc_bpw < ACC_BUDGET_BPW_UNDER_BASE3_Q`. When
`measured_q_scale_bpw > R4B_Q_PHYSICAL_BITS_PER_WEIGHT_BASE3` (i.e. > 1.6 bpw),
slice-4 effective budget is below 0.4 — do not expect identity with
`dual_boolean_record.sub2_candidate`.

Right-censor alignment: consumes slice-3 summaries with
`measurement_start_step=1` (slice-2b manifest pin); `k_star == H` is censored.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.d_recompute_window_horizon_analyzer import (
    GROWTH_ACCELERATING_OR_RIGHT_CENSORED,
    GROWTH_INCONCLUSIVE_COST_OR_COVERAGE,
    GROWTH_LINEAR_SIZED_WITH_DECAY,
    GROWTH_PLATEAU_SIZED,
    GROWTH_RIGHT_CENSORED_LOWER_BOUND,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
    EventCodedAccEvent,
    PackedEventCodedAccState,
    encode_event_coded_acc_events,
    encode_event_coded_backlog_indices,
    encode_hot_exact_rows,
    pack_event_coded_acc_checkpoint_v1,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    R4B_Q_PHYSICAL_BITS_PER_WEIGHT_BASE3,
    TARGET_PHYSICAL_BITS_PER_WEIGHT,
    measure_r4v_event_coded_acc_budget,
)
from calm.hrm_text_158.native_full_stack.qscale_linear import QScaleWeightState
from calm.hrm_text_158.native_full_stack.sub2_carrier_family_discriminator import (
    ACC_BUDGET_BPW_UNDER_BASE3_Q,
    DECLARED_Q_BPW_BASE3,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    promotion_carry_threshold,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    DEFAULT_CROSSING_THRESHOLD_ABS,
)

ACC_SIZING_SCHEMA_VERSION = "hrm_text_158_d_recompute_window_acc_sizing/v0"

VERDICT_SCOPE_ENVELOPE_MODEL_ONLY = "envelope_model_only"

DEFAULT_DECAY_GRID: tuple[tuple[int, int], ...] = (
    (1, 1),
    (9, 10),
    (1, 2),
    (4, 5),
    (19, 20),
)

SIZING_VERDICT_RECOMMENDED_LAW = "recommended_law"
SIZING_VERDICT_SIZED_NOT_SUB2 = "SIZED_WINDOW_ONLY_NOT_SUB2"
SIZING_VERDICT_INCONCLUSIVE = "INCONCLUSIVE_BUDGET_MAPPING_MISSING"

SIZED_GROWTH_BRANCHES: frozenset[str] = frozenset(
    {
        GROWTH_PLATEAU_SIZED,
        GROWTH_LINEAR_SIZED_WITH_DECAY,
    }
)

ACCUMULATOR_CLIP_MIN = -127
ACCUMULATOR_CLIP_MAX = 127


@dataclass(frozen=True)
class DecayEnvelopeSurface:
    events: tuple[EventCodedAccEvent, ...]
    backlog_indices: tuple[int, ...]
    hot_indices: tuple[int, ...]
    hot_values: tuple[int, ...]


def effective_acc_budget_bpw(*, measured_q_scale_bpw: float) -> float:
    q_room = float(TARGET_PHYSICAL_BITS_PER_WEIGHT) - float(measured_q_scale_bpw)
    return min(float(ACC_BUDGET_BPW_UNDER_BASE3_Q), float(q_room))


def _apply_decay_carry(
    acc: int,
    vote: int,
    *,
    decay_num: int,
    decay_den: int,
) -> int:
    if int(decay_den) <= 0:
        raise ValueError("decay_denominator must be > 0")
    decayed = (int(acc) * int(decay_num)) // int(decay_den)
    value = int(decayed) + int(vote)
    return max(ACCUMULATOR_CLIP_MIN, min(ACCUMULATOR_CLIP_MAX, value))


def simulate_decay_worst_case_surface(
    *,
    window_k: int,
    decay_num: int,
    decay_den: int,
    numel: int,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
) -> DecayEnvelopeSurface:
    """Decay-parameterized worst-case carry surface over K steps (no live carrier)."""

    k = int(window_k)
    n = int(numel)
    if k <= 0 or n <= 0:
        raise ValueError("window_k and numel must be positive")
    if int(decay_den) <= 0:
        raise ValueError("decay_denominator must be > 0")
    promote_at = int(promotion_carry_threshold(threshold_abs=int(threshold_abs)))
    retention = float(int(decay_num)) / float(int(decay_den))

    # Concentrated worst-case churn on lane 0 (decay affects terminal carry magnitude).
    carry0 = 0
    for step in range(1, k + 1):
        vote = 1 if step % 2 == 1 else -1
        carry0 = _apply_decay_carry(
            carry0,
            vote,
            decay_num=int(decay_num),
            decay_den=int(decay_den),
        )

    # Conservative decay-parameterized backlog width: slower decay retains more lanes.
    backlog_lane_count = min(n, max(1, int(math.ceil(float(k) * retention))))
    backlog_indices = tuple(range(int(backlog_lane_count)))

    hot_indices: tuple[int, ...] = ()
    hot_values: tuple[int, ...] = ()
    if abs(int(carry0)) >= promote_at:
        hot_indices = (0,)
        hot_values = (int(carry0),)

    events: list[EventCodedAccEvent] = []
    for step in range(1, k + 1):
        lane = (step - 1) % n
        events.append(
            EventCodedAccEvent(
                flat_index=int(lane),
                direction=1,
                residual_mag=15,
                event_type=1,
            )
        )
    return DecayEnvelopeSurface(
        events=tuple(events),
        backlog_indices=backlog_indices,
        hot_indices=hot_indices,
        hot_values=hot_values,
    )


def _remap_indices_to_high_adversary(
    indices: Sequence[int],
    *,
    numel: int,
) -> tuple[int, ...]:
    if int(numel) <= 0:
        raise ValueError("numel must be positive")
    if not indices:
        return ()
    return tuple(int(numel) - 1 for _ in indices)


def _events_with_index_pattern(
    event_count: int,
    *,
    numel: int,
    high_index: bool,
) -> tuple[EventCodedAccEvent, ...]:
    n = int(numel)
    hi = int(n) - 1
    out: list[EventCodedAccEvent] = []
    for step in range(1, int(event_count) + 1):
        flat_index = hi if high_index else (step - 1) % n
        out.append(
            EventCodedAccEvent(
                flat_index=int(flat_index),
                direction=1,
                residual_mag=15,
                event_type=1,
            )
        )
    return tuple(out)


def _payload_acc_bytes(payload: PackedEventCodedAccState) -> int:
    return int(
        payload.events_packed.numel()
        + payload.backlog_packed.numel()
        + payload.hot_exact_packed.numel()
        + int(payload.metadata_bytes)
    )


def build_conservative_envelope_payload(
    *,
    window_k: int,
    decay_num: int,
    decay_den: int,
    numel: int,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
) -> PackedEventCodedAccState:
    """Return the byte-max conservative envelope for (K, decay, numel)."""

    surface = simulate_decay_worst_case_surface(
        window_k=int(window_k),
        decay_num=int(decay_num),
        decay_den=int(decay_den),
        numel=int(numel),
        threshold_abs=int(threshold_abs),
    )
    event_count = len(surface.events)
    candidates: list[PackedEventCodedAccState] = []
    for high_index in (False, True):
        events = _events_with_index_pattern(
            event_count,
            numel=int(numel),
            high_index=bool(high_index),
        )
        if high_index:
            hot_idx = _remap_indices_to_high_adversary(
                surface.hot_indices,
                numel=int(numel),
            )
            backlog_idx = _remap_indices_to_high_adversary(
                surface.backlog_indices,
                numel=int(numel),
            )
        else:
            hot_idx = surface.hot_indices
            backlog_idx = surface.backlog_indices
        candidates.append(
            pack_event_coded_acc_checkpoint_v1(
                logical_numel=int(numel),
                events=events,
                backlog_indices=backlog_idx,
                hot_exact_indices=hot_idx,
                hot_exact_values=surface.hot_values,
            )
        )
    return max(candidates, key=_payload_acc_bytes)


def measure_envelope_inclusive_bpw(
    *,
    window_k: int,
    decay_num: int,
    decay_den: int,
    numel: int,
    state_key: str = "envelope.acc",
) -> dict[str, Any]:
    payload = build_conservative_envelope_payload(
        window_k=int(window_k),
        decay_num=int(decay_num),
        decay_den=int(decay_den),
        numel=int(numel),
    )
    q = torch.zeros(int(numel), dtype=torch.int8)
    qstate = QScaleWeightState(
        q_levels=q.view(1, int(numel)),
        scale=torch.tensor(1.0, dtype=torch.float32),
    )
    report = measure_r4v_event_coded_acc_budget(
        [qstate],
        [payload],
        state_keys=[str(state_key)],
    )
    return {
        "decay_num": int(decay_num),
        "decay_den": int(decay_den),
        "window_k": int(window_k),
        "numel": int(numel),
        "acc_term_bpw": float(report.r4v_acc_physical_bits_per_weight),
        "acc_metadata_bpw": float(report.r4v_acc_metadata_bits_per_weight),
        "inclusive_acc_bpw": float(report.r4v_acc_inclusive_physical_bits_per_weight),
        "r4v_actual_events_payload_bytes": int(report.r4v_actual_events_payload_bytes),
        "r4v_actual_backlog_payload_bytes": int(report.r4v_actual_backlog_payload_bytes),
        "r4v_actual_hot_exact_payload_bytes": int(
            report.r4v_actual_hot_exact_payload_bytes
        ),
        "r4v_actual_acc_metadata_bytes": int(report.r4v_actual_acc_metadata_bytes),
    }


def compare_envelope_pattern_bytes(
    *,
    event_count: int,
    numel: int,
    backlog_indices: Sequence[int] = (),
    hot_indices: Sequence[int] = (),
    hot_values: Sequence[int] = (),
) -> dict[str, int]:
    """Diagnostic helper: compare dense-low vs high-index adversary event bytes."""

    dense_events = _events_with_index_pattern(
        int(event_count),
        numel=int(numel),
        high_index=False,
    )
    high_events = _events_with_index_pattern(
        int(event_count),
        numel=int(numel),
        high_index=True,
    )
    hi = int(numel) - 1
    return {
        "dense_low_event_bytes": len(encode_event_coded_acc_events(dense_events)),
        "high_index_event_bytes": len(encode_event_coded_acc_events(high_events)),
        "high_index_flat_index": hi,
        "backlog_high_bytes": len(
            encode_event_coded_backlog_indices(
                _remap_indices_to_high_adversary(backlog_indices, numel=int(numel))
            )
        ),
        "hot_high_bytes": len(
            encode_hot_exact_rows(
                _remap_indices_to_high_adversary(hot_indices, numel=int(numel)),
                tuple(int(v) for v in hot_values),
            )
        ),
    }


def extract_sizing_window_k(
    summary: Mapping[str, Any],
    *,
    sizing_horizon_h: int,
) -> int | None:
    horizon_h = int(sizing_horizon_h)
    k99 = summary.get("k99_weighted")
    kworst = summary.get("kworst_weighted")
    if k99 is None or kworst is None:
        return None
    if float(kworst) >= float(horizon_h) or float(k99) >= float(horizon_h):
        return None
    return int(math.ceil(float(kworst)))


def size_acc_bpw_from_horizon_growth(
    horizon_growth: Mapping[str, Any],
    *,
    measured_q_scale_bpw: float | None = None,
    decay_grid: Sequence[tuple[int, int]] = DEFAULT_DECAY_GRID,
    sizing_horizon_h: int = 100,
    measurement_start_step: int = 1,
    numel_for_bpw: int,
    state_key: str = "envelope.acc",
) -> dict[str, Any]:
    _ = int(measurement_start_step)  # pinned contract; summaries already at start=1
    growth_branch = str(horizon_growth.get("growth_branch", ""))
    summaries = dict(horizon_growth.get("summaries_by_h") or {})
    summary = summaries.get(int(sizing_horizon_h))
    q_scale_bpw = (
        float(DECLARED_Q_BPW_BASE3)
        if measured_q_scale_bpw is None
        else float(measured_q_scale_bpw)
    )
    budget_bpw = effective_acc_budget_bpw(measured_q_scale_bpw=float(q_scale_bpw))

    base: dict[str, Any] = {
        "schema_version": ACC_SIZING_SCHEMA_VERSION,
        "growth_branch": growth_branch,
        "sizing_horizon_h": int(sizing_horizon_h),
        "measurement_start_step": 1,
        "measured_q_scale_bpw": float(q_scale_bpw),
        "effective_acc_budget_bpw": float(budget_bpw),
        "acc_budget_bpw_under_base3_q": float(ACC_BUDGET_BPW_UNDER_BASE3_Q),
        "target_physical_bits_per_weight": float(TARGET_PHYSICAL_BITS_PER_WEIGHT),
        "r4b_q_physical_bits_per_weight_base3": float(
            R4B_Q_PHYSICAL_BITS_PER_WEIGHT_BASE3
        ),
        "numel_for_bpw": int(numel_for_bpw),
        "decay_grid": [list(item) for item in decay_grid],
        "verdict_scope": VERDICT_SCOPE_ENVELOPE_MODEL_ONLY,
        "not_in_vivo_bound": True,
        "requires_slice5_live_validation": True,
    }

    if growth_branch == GROWTH_RIGHT_CENSORED_LOWER_BOUND:
        return base | {
            "sizing_verdict": SIZING_VERDICT_INCONCLUSIVE,
            "reason": "right_censored_growth_branch",
        }
    if growth_branch not in SIZED_GROWTH_BRANCHES:
        return base | {
            "sizing_verdict": SIZING_VERDICT_INCONCLUSIVE,
            "reason": "growth_branch_not_sized",
            "growth_branch": growth_branch,
        }
    if summary is None:
        return base | {
            "sizing_verdict": SIZING_VERDICT_INCONCLUSIVE,
            "reason": "missing_sizing_horizon_summary",
        }

    window_k = extract_sizing_window_k(summary, sizing_horizon_h=int(sizing_horizon_h))
    if window_k is None:
        return base | {
            "sizing_verdict": SIZING_VERDICT_INCONCLUSIVE,
            "reason": "censored_or_missing_kworst_at_horizon",
        }

    grid_rows: list[dict[str, Any]] = []
    mapping_ok = True
    for decay_num, decay_den in decay_grid:
        try:
            row = measure_envelope_inclusive_bpw(
                window_k=int(window_k),
                decay_num=int(decay_num),
                decay_den=int(decay_den),
                numel=int(numel_for_bpw),
                state_key=str(state_key),
            )
        except (ValueError, TypeError) as exc:
            mapping_ok = False
            row = {
                "decay_num": int(decay_num),
                "decay_den": int(decay_den),
                "mapping_error": str(exc),
            }
        grid_rows.append(row)

    if not mapping_ok:
        return base | {
            "sizing_verdict": SIZING_VERDICT_INCONCLUSIVE,
            "reason": "budget_mapping_failed",
            "window_k": int(window_k),
            "grid_rows": grid_rows,
        }

    measurable = [
        row
        for row in grid_rows
        if "inclusive_acc_bpw" in row and row.get("mapping_error") is None
    ]
    if not measurable:
        return base | {
            "sizing_verdict": SIZING_VERDICT_INCONCLUSIVE,
            "reason": "budget_mapping_missing",
            "window_k": int(window_k),
            "grid_rows": grid_rows,
        }

    best_row = min(measurable, key=lambda row: float(row["inclusive_acc_bpw"]))
    strict_pass = any(
        float(row["inclusive_acc_bpw"]) < float(budget_bpw) for row in measurable
    )
    if strict_pass:
        verdict = SIZING_VERDICT_RECOMMENDED_LAW
        reason = "inclusive_acc_bpw_strictly_under_effective_budget"
    else:
        verdict = SIZING_VERDICT_SIZED_NOT_SUB2
        reason = "sized_window_bpw_at_or_above_effective_budget"

    return base | {
        "sizing_verdict": verdict,
        "reason": reason,
        "window_k": int(window_k),
        "grid_rows": grid_rows,
        "best_grid_row": dict(best_row),
        "strict_less_than_budget": bool(strict_pass),
    }
