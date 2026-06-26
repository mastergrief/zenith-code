"""CPU dense-accumulator width parity screen (C3 Phase-0).

Decision-parity contract over the dense update law with width-specific
``effective_clip_bounds`` — not a packing/storage shortcut.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    apply_global_rate_cap_reference,
    tensor_offsets_for_vote_update_states,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    DEFAULT_CROSSING_THRESHOLD_ABS,
    VOTE_UPDATE_SOURCE_CLIP_MAX,
    VOTE_UPDATE_SOURCE_CLIP_MIN,
    crosses_threshold,
    decay_vote_clamp,
    effective_clip_bounds,
    signed_w_max,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
)

DENSE_ACC_WIDTH_PARITY_SCHEMA_VERSION = "hrm_text_158_dense_accumulator_width_parity_screen/v1"
MANDATORY_WIDTH_GRID: tuple[int, ...] = (16, 8, 7, 6, 5, 4)
BOUNDARY_OVERSHOOT_SCENARIO_CLASS = "boundary_overshoot"
SUB_FLOOR_BOUNDARY_TESTED_WIDTHS: tuple[int, ...] = (5, 6)
BOUNDARY_TESTED_WIDTH = 5  # legacy alias for decisive narrow width in tests/receipts
STRUCTURALLY_LOSSLESS_WIDTHS: tuple[int, ...] = (7, 8, 10, 12, 16)
DECISIVE_NARROW_WIDTHS: tuple[int, ...] = (5,)
STORAGE_LOSSLESS_WIDTHS: tuple[int, ...] = (8, 10, 12)
BELOW_THRESHOLD_TRIVIAL_WIDTH = 4
REFERENCE_WIDTH = 16
BPW_REDUCTION_TARGET_FRACTION = 0.25
# 3-ledger persistent state: int8 q + int16 vote-acc + FP32 scale (north star = sub-2 inclusive).
PERSISTENT_Q_TERM_BPW = 8
PERSISTENT_SCALE_TERM_BPW_ESTIMATE = 0.0
SUB2_SCOPE_CAVEAT = (
    "bpw_reduction_at_structural_floor is vote-accumulator term only (16->7 bits); "
    "does NOT achieve sub-2 inclusive persistent — int8 q (~8b) and FP32 scale remain; "
    "at W7 structural floor persistent state is ~15 bits/weight. "
    "Narrows the int16 vote-acc dominator only; q term + scale must be addressed for sub-2."
)
CLASSIFIER_LABEL_C3_DENSE_WIDTH_REDUCIBLE_TO_FLOOR = (
    "reducible to structural clip floor W7 (~56% bpw off int16 vote-acc term only, "
    "not inclusive persistent budget); W5 breaks O3 on overshoot"
)
# ``accumulator_real_dynamics_verdict.PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE`` uses vote_abs up to 24.
CANONICAL_MAX_RANK_VOTE_ABS = 24
DRY_RUN_MAX_RANK_VOTE_ABS = 4

CLASSIFIER_C3_RUN_HEALTH_FAIL = "C3_RUN_HEALTH_FAIL"
CLASSIFIER_C3_MISSING_OBSERVABLES = "C3_MISSING_OBSERVABLES"
CLASSIFIER_C3_INT16_STORAGE_OVERWIDE = "C3_INT16_STORAGE_OVERWIDE"
CLASSIFIER_C3_DENSE_WIDTH_REDUCIBLE = "C3_DENSE_WIDTH_REDUCIBLE"
CLASSIFIER_C3_NARROW_WIDTH_BREAKS_DECISION_PARITY = "C3_NARROW_WIDTH_BREAKS_DECISION_PARITY"
CLASSIFIER_C3_DENSE_WIDTH_REDUCIBLE_TO_FLOOR = "C3_DENSE_WIDTH_REDUCIBLE_TO_FLOOR"
CLASSIFIER_C3_DENSE_WIDTH_INTRINSIC_UNDER_CONTRACT = (
    "C3_DENSE_WIDTH_INTRINSIC_UNDER_CONTRACT"
)
CLASSIFIER_C3_CLIP_EXCEEDS_REACHABLE_RANGE = "C3_CLIP_EXCEEDS_REACHABLE_RANGE"


def reachable_pre_crossing_accumulator_peak(
    *,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
    max_vote_abs: int = CANONICAL_MAX_RANK_VOTE_ABS,
) -> int:
    """Upper bound on crossing-step accumulator before clip: (T-1)+max_vote."""

    return int(threshold_abs) - 1 + int(max_vote_abs)


@dataclass(frozen=True)
class SyntheticWidthScenario:
    name: str
    votes: tuple[int, ...]
    initial_acc: int = 0
    initial_q: int = 0
    scenario_class: str = "baseline"


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(int(lo), min(int(hi), int(value)))


def post_flip_residual_clamp(
    new_acc: int,
    *,
    proposal_direction: int,
    threshold_abs: int,
) -> int:
    direction = 1 if int(proposal_direction) >= 0 else -1
    residual = int(new_acc) - direction * int(threshold_abs)
    lo = -int(threshold_abs) + 1
    hi = int(threshold_abs) - 1
    return _clamp(residual, lo, hi)


@dataclass(frozen=True)
class LaneStepRecord:
    step_index: int
    vote: int
    pre_acc: int
    pre_q: int
    new_acc: int
    crossing: bool
    applied_flip: bool
    post_acc: int
    post_q: int


@dataclass(frozen=True)
class LaneTrajectory:
    width: int
    scenario_name: str
    effective_clip_min: int
    effective_clip_max: int
    width_regime: str
    steps: tuple[LaneStepRecord, ...]

    @property
    def crossing_steps(self) -> tuple[int, ...]:
        return tuple(record.step_index for record in self.steps if record.crossing)

    @property
    def applied_flip_steps(self) -> tuple[int, ...]:
        return tuple(record.step_index for record in self.steps if record.applied_flip)

    @property
    def q_history(self) -> tuple[int, ...]:
        return tuple(record.post_q for record in self.steps)

    @property
    def acc_history(self) -> tuple[int, ...]:
        return tuple(record.post_acc for record in self.steps)


@dataclass(frozen=True)
class WidthDriftRow:
    scenario_name: str
    scenario_class: str
    width: int
    width_regime: str
    drift: bool
    o1_crossing_mismatch: bool
    o2_flip_mismatch: bool
    o3_acc_mismatch: bool
    o4_q_mismatch: bool
    excluded_from_parity_failure: bool


@dataclass(frozen=True)
class O5WidthSurface:
    width: int
    accepted_identities: tuple[tuple[str, int], ...]
    deferred_identities: tuple[tuple[str, int], ...]
    global_cap_saturated: bool
    global_cap_row_count: int


@dataclass(frozen=True)
class O5FixtureResult:
    observed: bool
    reason: str
    reference_width: int
    surfaces_by_width: tuple[O5WidthSurface, ...]
    o5_drift_vs_reference: tuple[tuple[int, bool], ...]


@dataclass(frozen=True)
class WidthParityScreenResult:
    schema_version: str
    width_grid: tuple[int, ...]
    scenarios: tuple[str, ...]
    drift_rows: tuple[WidthDriftRow, ...]
    parity_failure_count: int
    screen_complete: bool
    o5: O5FixtureResult
    classifier: str
    classifier_basis: str
    storage_overwide: bool
    decisive_narrow_all_parity_safe: bool
    bpw_by_width: dict[str, float]
    reachable_pre_crossing_peak: int
    canonical_max_rank_vote_abs: int
    boundary_overshoot_tested: bool
    structural_clip_floor_width: int | None
    minimum_safe_width_empirical: int | None
    minimum_safe_width_structural: int | None
    sub_floor_breaking_widths: tuple[int, ...]
    sub_floor_parity_safe_widths: tuple[int, ...]
    bpw_reduction_at_structural_floor: float | None
    sub2_scope_caveat: str
    persistent_state_bpw_at_structural_floor_estimate: float | None
    classifier_label: str


def is_structurally_lossless_width(
    width: int,
    *,
    reachable_peak: int | None = None,
) -> bool:
    peak = (
        int(reachable_peak)
        if reachable_peak is not None
        else reachable_pre_crossing_accumulator_peak()
    )
    return clip_abs_for_width(int(width)) >= int(peak)


def width_regime_label(width: int, *, reachable_peak: int | None = None) -> str:
    peak = (
        int(reachable_peak)
        if reachable_peak is not None
        else reachable_pre_crossing_accumulator_peak()
    )
    if int(width) == BELOW_THRESHOLD_TRIVIAL_WIDTH:
        return "below_threshold_trivial"
    if int(width) == 8:
        return "source_clip_lossless"
    if int(width) in SUB_FLOOR_BOUNDARY_TESTED_WIDTHS:
        return "sub_floor_boundary_tested"
    if is_structurally_lossless_width(width, reachable_peak=peak):
        return "clip_exceeds_reachable_range"
    if int(width) in DECISIVE_NARROW_WIDTHS:
        return "narrow_dynamics_decisive"
    if int(width) == REFERENCE_WIDTH:
        return "reference"
    return "optional_control"


def effective_clip_for_width(width: int) -> tuple[int, int]:
    return effective_clip_bounds(
        int(width),
        VOTE_UPDATE_SOURCE_CLIP_MIN,
        VOTE_UPDATE_SOURCE_CLIP_MAX,
    )


def clip_abs_for_width(width: int) -> int:
    clip_min, clip_max = effective_clip_for_width(int(width))
    return int(max(abs(int(clip_min)), abs(int(clip_max))))


def sub_floor_widths_for_peak(
    width_grid: Sequence[int],
    *,
    reachable_peak: int,
) -> tuple[int, ...]:
    return tuple(
        int(width)
        for width in width_grid
        if int(width) != BELOW_THRESHOLD_TRIVIAL_WIDTH
        and clip_abs_for_width(int(width)) < int(reachable_peak)
    )


def structural_clip_floor_width(
    width_grid: Sequence[int],
    *,
    reachable_peak: int,
) -> int | None:
    """Smallest grid width whose clip magnitude is >= reachable_peak."""

    candidates = [
        int(width)
        for width in width_grid
        if clip_abs_for_width(int(width)) >= int(reachable_peak)
    ]
    return min(candidates) if candidates else None


def boundary_overshoot_scenarios(
    scenarios: Sequence[SyntheticWidthScenario],
) -> tuple[SyntheticWidthScenario, ...]:
    return tuple(
        scenario
        for scenario in scenarios
        if str(scenario.scenario_class) == BOUNDARY_OVERSHOOT_SCENARIO_CLASS
    )


def width_passes_all_boundary_overshoot(
    width: int,
    *,
    scenarios: Sequence[SyntheticWidthScenario],
    reference_width: int = REFERENCE_WIDTH,
) -> bool:
    for scenario in boundary_overshoot_scenarios(scenarios):
        reference = simulate_lane_trajectory(scenario, width=int(reference_width))
        candidate = simulate_lane_trajectory(scenario, width=int(width))
        if _trajectory_drift(reference, candidate, scenario_class=scenario.scenario_class).drift:
            return False
    return True


def minimum_safe_width_empirical(
    width_grid: Sequence[int],
    *,
    scenarios: Sequence[SyntheticWidthScenario],
    reference_width: int = REFERENCE_WIDTH,
) -> int | None:
    """Smallest grid width that matches reference on all boundary-overshoot scenarios."""

    ordered = sorted(int(width) for width in width_grid if int(width) != BELOW_THRESHOLD_TRIVIAL_WIDTH)
    for width in ordered:
        if width_passes_all_boundary_overshoot(
            width,
            scenarios=scenarios,
            reference_width=reference_width,
        ):
            return int(width)
    return None


def accumulator_bpw_for_width(width: int) -> float:
    return float(int(width))


def persistent_state_bpw_at_structural_floor(structural_floor_width: int) -> float:
    return float(
        PERSISTENT_Q_TERM_BPW
        + int(structural_floor_width)
        + PERSISTENT_SCALE_TERM_BPW_ESTIMATE
    )


def classifier_label_for(classifier: str) -> str:
    if classifier == CLASSIFIER_C3_DENSE_WIDTH_REDUCIBLE_TO_FLOOR:
        return CLASSIFIER_LABEL_C3_DENSE_WIDTH_REDUCIBLE_TO_FLOOR
    return classifier


def default_mandatory_scenarios() -> tuple[SyntheticWidthScenario, ...]:
    return (
        SyntheticWidthScenario(
            name="delayed_crossing_sparse_votes",
            votes=(5, 0, 0, 5),
            scenario_class="baseline",
        ),
        SyntheticWidthScenario(
            name="decay_only",
            votes=(6, 0, 0, 0, 0),
            scenario_class="baseline",
        ),
        SyntheticWidthScenario(
            name="oscillation",
            votes=(5, 0, -5, 0, 6, 0),
            scenario_class="baseline",
        ),
        SyntheticWidthScenario(
            name="sub_threshold_staircase",
            votes=(1, 1, 1, 1, 1, 1, 1, 1, 2),
            scenario_class="baseline",
        ),
        SyntheticWidthScenario(
            name="near_saturation_carry",
            votes=(3, 3, 3, 3, 3, 3, 4),
            scenario_class="baseline",
        ),
        SyntheticWidthScenario(
            name="boundary_overshoot_positive",
            votes=(1, 1, 1, 1, 1, 1, 1, 1, 1, 7),
            scenario_class=BOUNDARY_OVERSHOOT_SCENARIO_CLASS,
        ),
        SyntheticWidthScenario(
            name="boundary_overshoot_negative",
            votes=(-1, -1, -1, -1, -1, -1, -1, -1, -1, -7),
            scenario_class=BOUNDARY_OVERSHOOT_SCENARIO_CLASS,
        ),
        SyntheticWidthScenario(
            name="boundary_overshoot_oscillation_past_clip",
            votes=(8, 8, 0, -8, -8),
            scenario_class=BOUNDARY_OVERSHOOT_SCENARIO_CLASS,
        ),
        SyntheticWidthScenario(
            name="boundary_overshoot_canonical_vote24",
            votes=(1, 1, 1, 1, 1, 1, 1, 1, 1, CANONICAL_MAX_RANK_VOTE_ABS),
            scenario_class=BOUNDARY_OVERSHOOT_SCENARIO_CLASS,
        ),
    )


def simulate_lane_trajectory(
    scenario: SyntheticWidthScenario,
    *,
    width: int,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
    decay_numerator: int = 1,
    decay_denominator: int = 1,
) -> LaneTrajectory:
    clip_min, clip_max = effective_clip_for_width(width)
    acc = int(scenario.initial_acc)
    q = int(scenario.initial_q)
    records: list[LaneStepRecord] = []
    for step_index, vote in enumerate(scenario.votes):
        pre_acc = acc
        pre_q = q
        new_acc = decay_vote_clamp(
            pre_acc,
            int(vote),
            clip_min=clip_min,
            clip_max=clip_max,
            decay_numerator=decay_numerator,
            decay_denominator=decay_denominator,
        )
        crossing = crosses_threshold(
            new_acc,
            current_q_level=pre_q,
            threshold_abs=threshold_abs,
        )
        applied_flip = False
        post_acc = new_acc
        post_q = pre_q
        if crossing:
            direction = 1 if new_acc >= threshold_abs else -1
            post_q = _clamp(pre_q + direction, -1, 1)
            post_acc = post_flip_residual_clamp(
                new_acc,
                proposal_direction=direction,
                threshold_abs=threshold_abs,
            )
            applied_flip = True
            acc = post_acc
            q = post_q
        else:
            acc = new_acc
            q = pre_q
        records.append(
            LaneStepRecord(
                step_index=step_index,
                vote=int(vote),
                pre_acc=pre_acc,
                pre_q=pre_q,
                new_acc=new_acc,
                crossing=bool(crossing),
                applied_flip=applied_flip,
                post_acc=post_acc,
                post_q=post_q,
            )
        )
    return LaneTrajectory(
        width=int(width),
        scenario_name=str(scenario.name),
        effective_clip_min=int(clip_min),
        effective_clip_max=int(clip_max),
        width_regime=width_regime_label(width, reachable_peak=reachable_pre_crossing_accumulator_peak()),
        steps=tuple(records),
    )


def _trajectory_drift(
    reference: LaneTrajectory,
    candidate: LaneTrajectory,
    *,
    scenario_class: str,
) -> WidthDriftRow:
    o1 = reference.crossing_steps != candidate.crossing_steps
    o2 = reference.applied_flip_steps != candidate.applied_flip_steps
    o3 = reference.acc_history != candidate.acc_history
    o4 = reference.q_history != candidate.q_history
    drift = bool(o1 or o2 or o3 or o4)
    width = int(candidate.width)
    excluded = (
        width == BELOW_THRESHOLD_TRIVIAL_WIDTH
        or is_structurally_lossless_width(width)
        or (
            width in SUB_FLOOR_BOUNDARY_TESTED_WIDTHS
            and str(scenario_class) != BOUNDARY_OVERSHOOT_SCENARIO_CLASS
        )
        or (
            width in STRUCTURALLY_LOSSLESS_WIDTHS
            and str(scenario_class) == "baseline"
        )
    )
    return WidthDriftRow(
        scenario_name=reference.scenario_name,
        scenario_class=str(scenario_class),
        width=width,
        width_regime=candidate.width_regime,
        drift=drift,
        o1_crossing_mismatch=o1,
        o2_flip_mismatch=o2,
        o3_acc_mismatch=o3,
        o4_q_mismatch=o4,
        excluded_from_parity_failure=excluded,
    )


def vote_spec_for_width(width: int, *, threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS) -> VoteUpdateSpec:
    clip_min, clip_max = effective_clip_for_width(width)
    return VoteUpdateSpec(
        threshold_abs=int(threshold_abs),
        accumulator_clip_min=int(clip_min),
        accumulator_clip_max=int(clip_max),
    )


def _identity_rows(rows: Sequence[Any]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(row.state_key), int(row.flat_index)) for row in rows))


def measure_o5_global_cap_surface_for_width(
    width: int,
    *,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
    global_cap: int = 1,
) -> O5WidthSurface:
    """Build W-specific local plans and observe global-cap surfaces."""

    spec = vote_spec_for_width(width, threshold_abs=threshold_abs)
    states: list[tuple[str, VoteUpdateState, VoteUpdateInputs]] = []
    numel = 8
    votes_a = torch.zeros(numel, dtype=torch.int16)
    votes_a[0] = 12
    votes_a[1] = 11
    votes_b = torch.zeros(numel, dtype=torch.int16)
    votes_b[2] = 13
    votes_b[3] = 10
    for key, votes in (("tensor_a", votes_a), ("tensor_b", votes_b)):
        q = torch.zeros(numel, dtype=torch.int8)
        acc = torch.zeros(numel, dtype=torch.int16)
        state = VoteUpdateState(q_levels=q, accumulators=acc)
        inputs = VoteUpdateInputs(votes=votes.to(torch.int16))
        states.append((key, state, inputs))
    cap_inputs: list[GlobalRateCapTensorInput] = []
    for key, state, inputs in states:
        plan = plan_integer_vote_update_reference(state, inputs, spec)
        cap_inputs.append(
            GlobalRateCapTensorInput(
                state_key=key,
                state=state,
                plan=plan,
                vote_inputs=inputs,
            )
        )
    cap_spec = GlobalRateCapSpec(cap=int(global_cap), step=1)
    cap_result = apply_global_rate_cap_reference(
        cap_inputs,
        cap_spec,
        tensor_offsets=tensor_offsets_for_vote_update_states(cap_inputs),
    )
    return O5WidthSurface(
        width=int(width),
        accepted_identities=_identity_rows(cap_result.accepted_rows),
        deferred_identities=_identity_rows(cap_result.deferred_rows),
        global_cap_saturated=bool(cap_result.step_summary.get("global_rate_cap_saturated", False)),
        global_cap_row_count=len(cap_result.rows),
    )


def build_o5_fixture_result(
    *,
    widths: Sequence[int] = MANDATORY_WIDTH_GRID,
    reference_width: int = REFERENCE_WIDTH,
) -> O5FixtureResult:
    try:
        surfaces = tuple(
            measure_o5_global_cap_surface_for_width(int(width)) for width in widths
        )
    except Exception as exc:  # pragma: no cover - fail-closed path tested explicitly
        return O5FixtureResult(
            observed=False,
            reason=f"o5_fixture_construction_failed: {exc}",
            reference_width=int(reference_width),
            surfaces_by_width=(),
            o5_drift_vs_reference=(),
        )
    ref = next(surface for surface in surfaces if surface.width == int(reference_width))
    drift_pairs: list[tuple[int, bool]] = []
    for surface in surfaces:
        if surface.width == ref.width:
            drift_pairs.append((surface.width, False))
            continue
        drift = (
            surface.accepted_identities != ref.accepted_identities
            or surface.deferred_identities != ref.deferred_identities
            or surface.global_cap_saturated != ref.global_cap_saturated
        )
        drift_pairs.append((surface.width, drift))
    return O5FixtureResult(
        observed=True,
        reason="w_specific_plan_global_cap_fixture",
        reference_width=int(reference_width),
        surfaces_by_width=surfaces,
        o5_drift_vs_reference=tuple(drift_pairs),
    )


def run_width_parity_screen(
    *,
    width_grid: Sequence[int] = MANDATORY_WIDTH_GRID,
    scenarios: Sequence[SyntheticWidthScenario] | None = None,
    reference_width: int = REFERENCE_WIDTH,
    include_o5: bool = True,
    screen_complete: bool = True,
) -> WidthParityScreenResult:
    scenario_list = tuple(scenarios or default_mandatory_scenarios())
    grid = tuple(int(width) for width in width_grid)
    drift_rows: list[WidthDriftRow] = []
    trajectories: dict[tuple[str, int], LaneTrajectory] = {}
    for scenario in scenario_list:
        for width in grid:
            trajectories[(scenario.name, width)] = simulate_lane_trajectory(scenario, width=width)
    for scenario in scenario_list:
        reference = trajectories[(scenario.name, int(reference_width))]
        for width in grid:
            if int(width) == int(reference_width):
                continue
            candidate = trajectories[(scenario.name, int(width))]
            drift_rows.append(
                _trajectory_drift(
                    reference,
                    candidate,
                    scenario_class=scenario.scenario_class,
                )
            )
    parity_failure_count = sum(
        1 for row in drift_rows if row.drift and not row.excluded_from_parity_failure
    )
    o5 = build_o5_fixture_result(widths=grid, reference_width=reference_width) if include_o5 else O5FixtureResult(
        observed=False,
        reason="o5_disabled",
        reference_width=int(reference_width),
        surfaces_by_width=(),
        o5_drift_vs_reference=(),
    )
    bpw_by_width = {str(width): accumulator_bpw_for_width(width) for width in grid}
    reachable_peak = reachable_pre_crossing_accumulator_peak()
    structural_floor = structural_clip_floor_width(grid, reachable_peak=reachable_peak)
    empirical_min = minimum_safe_width_empirical(
        grid,
        scenarios=scenario_list,
        reference_width=int(reference_width),
    )
    boundary_probe_rows = [
        row
        for row in drift_rows
        if int(row.width) in SUB_FLOOR_BOUNDARY_TESTED_WIDTHS
        and row.scenario_class == BOUNDARY_OVERSHOOT_SCENARIO_CLASS
    ]
    sub_floor_breaking = tuple(
        sorted(
            {
                int(row.width)
                for row in boundary_probe_rows
                if row.drift and not row.excluded_from_parity_failure
            }
        )
    )
    sub_floor_parity_safe = tuple(
        width
        for width in SUB_FLOOR_BOUNDARY_TESTED_WIDTHS
        if int(width) in grid and int(width) not in sub_floor_breaking
    )
    bpw_reduction_at_floor: float | None = None
    persistent_bpw_at_floor: float | None = None
    if structural_floor is not None:
        ref_bpw = float(bpw_by_width[str(reference_width)])
        floor_bpw = float(bpw_by_width[str(structural_floor)])
        bpw_reduction_at_floor = (ref_bpw - floor_bpw) / ref_bpw
        persistent_bpw_at_floor = persistent_state_bpw_at_structural_floor(structural_floor)
    classifier, basis, storage_overwide, decisive_safe = classify_c3_dense_width_screen(
        drift_rows=tuple(drift_rows),
        width_grid=grid,
        screen_complete=screen_complete,
        o5=o5,
        bpw_by_width=bpw_by_width,
        reference_width=int(reference_width),
        boundary_probe_rows=tuple(boundary_probe_rows),
        reachable_peak=reachable_peak,
        structural_floor_width=structural_floor,
        minimum_safe_width_empirical=empirical_min,
        sub_floor_breaking_widths=sub_floor_breaking,
    )
    return WidthParityScreenResult(
        schema_version=DENSE_ACC_WIDTH_PARITY_SCHEMA_VERSION,
        width_grid=grid,
        scenarios=tuple(scenario.name for scenario in scenario_list),
        drift_rows=tuple(drift_rows),
        parity_failure_count=int(parity_failure_count),
        screen_complete=bool(screen_complete),
        o5=o5,
        classifier=classifier,
        classifier_basis=basis,
        storage_overwide=storage_overwide,
        decisive_narrow_all_parity_safe=decisive_safe,
        bpw_by_width=bpw_by_width,
        reachable_pre_crossing_peak=int(reachable_peak),
        canonical_max_rank_vote_abs=int(CANONICAL_MAX_RANK_VOTE_ABS),
        boundary_overshoot_tested=bool(boundary_probe_rows),
        structural_clip_floor_width=structural_floor,
        minimum_safe_width_empirical=empirical_min,
        minimum_safe_width_structural=structural_floor,
        sub_floor_breaking_widths=sub_floor_breaking,
        sub_floor_parity_safe_widths=sub_floor_parity_safe,
        bpw_reduction_at_structural_floor=bpw_reduction_at_floor,
        sub2_scope_caveat=SUB2_SCOPE_CAVEAT,
        persistent_state_bpw_at_structural_floor_estimate=persistent_bpw_at_floor,
        classifier_label=classifier_label_for(classifier),
    )


def classify_c3_dense_width_screen(
    *,
    drift_rows: Sequence[WidthDriftRow],
    width_grid: Sequence[int],
    screen_complete: bool,
    o5: O5FixtureResult,
    bpw_by_width: Mapping[str, float],
    reference_width: int = REFERENCE_WIDTH,
    boundary_probe_rows: Sequence[WidthDriftRow] = (),
    reachable_peak: int | None = None,
    structural_floor_width: int | None = None,
    minimum_safe_width_empirical: int | None = None,
    sub_floor_breaking_widths: Sequence[int] = (),
) -> tuple[str, str, bool, bool]:
    if not screen_complete:
        return (
            CLASSIFIER_C3_RUN_HEALTH_FAIL,
            "screen_incomplete",
            False,
            False,
        )
    grid = {int(width) for width in width_grid}
    required = set(MANDATORY_WIDTH_GRID)
    if not required.issubset(grid):
        return (
            CLASSIFIER_C3_RUN_HEALTH_FAIL,
            "missing_mandatory_width",
            False,
            False,
        )
    overshoot_rows = [
        row for row in boundary_probe_rows if row.scenario_class == BOUNDARY_OVERSHOOT_SCENARIO_CLASS
    ]
    if not overshoot_rows:
        return (
            CLASSIFIER_C3_RUN_HEALTH_FAIL,
            "missing_boundary_overshoot_scenarios",
            False,
            False,
        )
    w8_rows = [row for row in drift_rows if int(row.width) == 8]
    storage_overwide = bool(w8_rows) and all(not row.drift for row in w8_rows)
    peak = (
        int(reachable_peak)
        if reachable_peak is not None
        else reachable_pre_crossing_accumulator_peak()
    )
    structural_floor = (
        int(structural_floor_width)
        if structural_floor_width is not None
        else structural_clip_floor_width(width_grid, reachable_peak=peak)
    )
    empirical_min = minimum_safe_width_empirical
    sub_floor_breaking = tuple(int(width) for width in sub_floor_breaking_widths)
    w5_overshoot_rows = [row for row in overshoot_rows if int(row.width) == BOUNDARY_TESTED_WIDTH]
    w5_boundary_safe = all(not row.drift for row in w5_overshoot_rows)
    w5_boundary_drift = any(row.drift for row in w5_overshoot_rows)
    ref_bpw = float(bpw_by_width[str(reference_width)])
    w5_bpw = float(bpw_by_width[str(BOUNDARY_TESTED_WIDTH)])
    w5_reduction_ok = (ref_bpw - w5_bpw) / ref_bpw >= BPW_REDUCTION_TARGET_FRACTION
    floor_reduction_ok = False
    if structural_floor is not None:
        floor_bpw = float(bpw_by_width[str(structural_floor)])
        floor_reduction_fraction = (ref_bpw - floor_bpw) / ref_bpw
        floor_reduction_ok = floor_reduction_fraction >= BPW_REDUCTION_TARGET_FRACTION
    if not o5.observed:
        classifier = CLASSIFIER_C3_MISSING_OBSERVABLES
        basis = o5.reason
    elif structural_floor is not None and sub_floor_breaking and floor_reduction_ok:
        classifier = CLASSIFIER_C3_DENSE_WIDTH_REDUCIBLE_TO_FLOOR
        basis = (
            f"structural_clip_floor_w{structural_floor}_at_reachable_peak_{peak};"
            f"vote_acc_bpw_reduction_fraction={floor_reduction_fraction:.4f};"
            f"sub_floor_breaking={list(sub_floor_breaking)};"
            f"empirical_overshoot_safe_min_w{empirical_min};"
            "vote_acc_term_only_not_sub2_inclusive"
        )
    elif w5_boundary_safe and w5_reduction_ok:
        classifier = CLASSIFIER_C3_DENSE_WIDTH_REDUCIBLE
        basis = "boundary_overshoot_parity_safe_and_bpw_reduction"
    elif w5_boundary_drift or sub_floor_breaking:
        classifier = CLASSIFIER_C3_NARROW_WIDTH_BREAKS_DECISION_PARITY
        basis = "sub_floor_drift_on_boundary_overshoot_scenarios"
    elif storage_overwide:
        classifier = CLASSIFIER_C3_INT16_STORAGE_OVERWIDE
        basis = "w8_source_clip_shrink_only"
    else:
        classifier = CLASSIFIER_C3_DENSE_WIDTH_INTRINSIC_UNDER_CONTRACT
        basis = "complete_grid_non_trivial_break_or_no_reduction"
    decisive_safe = bool(w5_boundary_safe)
    return classifier, basis, storage_overwide, decisive_safe


def result_to_report_dict(result: WidthParityScreenResult) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "width_grid": list(result.width_grid),
        "scenarios": list(result.scenarios),
        "parity_failure_count": result.parity_failure_count,
        "screen_complete": result.screen_complete,
        "classifier": result.classifier,
        "classifier_basis": result.classifier_basis,
        "storage_overwide": result.storage_overwide,
        "decisive_narrow_all_parity_safe": result.decisive_narrow_all_parity_safe,
        "bpw_by_width": dict(result.bpw_by_width),
        "reachable_pre_crossing_peak": result.reachable_pre_crossing_peak,
        "canonical_max_rank_vote_abs": result.canonical_max_rank_vote_abs,
        "boundary_overshoot_tested": result.boundary_overshoot_tested,
        "structural_clip_floor_width": result.structural_clip_floor_width,
        "minimum_safe_width_empirical": result.minimum_safe_width_empirical,
        "minimum_safe_width_structural": result.minimum_safe_width_structural,
        "sub_floor_breaking_widths": list(result.sub_floor_breaking_widths),
        "sub_floor_parity_safe_widths": list(result.sub_floor_parity_safe_widths),
        "bpw_reduction_at_structural_floor": result.bpw_reduction_at_structural_floor,
        "vote_acc_bpw_reduction_at_structural_floor": result.bpw_reduction_at_structural_floor,
        "sub2_scope_caveat": result.sub2_scope_caveat,
        "persistent_state_bpw_at_structural_floor_estimate": (
            result.persistent_state_bpw_at_structural_floor_estimate
        ),
        "classifier_label": result.classifier_label,
        "drift_rows": [asdict(row) for row in result.drift_rows],
        "o5": {
            "observed": result.o5.observed,
            "reason": result.o5.reason,
            "reference_width": result.o5.reference_width,
            "o5_drift_vs_reference": list(result.o5.o5_drift_vs_reference),
            "surfaces_by_width": [asdict(surface) for surface in result.o5.surfaces_by_width],
        },
    }


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def clip_table_for_grid(width_grid: Sequence[int] = MANDATORY_WIDTH_GRID) -> list[dict[str, Any]]:
    peak = reachable_pre_crossing_accumulator_peak()
    rows: list[dict[str, Any]] = []
    for width in width_grid:
        clip_min, clip_max = effective_clip_for_width(int(width))
        rows.append(
            {
                "width": int(width),
                "effective_clip_min": int(clip_min),
                "effective_clip_max": int(clip_max),
                "signed_w_max": int(signed_w_max(int(width))),
                "clip_abs": clip_abs_for_width(int(width)),
                "clips_reachable_peak": clip_abs_for_width(int(width)) < int(peak),
                "width_regime": width_regime_label(int(width), reachable_peak=peak),
            }
        )
    return rows
