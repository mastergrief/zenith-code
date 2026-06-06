"""C1.1c representative drift-vs-budget verdict for bounded-delta state.

This composes the C1.1b pre-registered native-loop pressure schedule with the
C1.1c bounded-delta ledger/oracle. Local one-step reports are diagnostic only;
the terminal report is cumulative, carrying exact and bounded q/acc/backlog
states independently across the fixed four-step schedule.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.accumulator_real_dynamics_verdict import (
    BINDING_FOR_IN_TREE_NATIVE_LOOP_DISTRIBUTION,
    PARTIAL_FOR_S1_REAL_DYNAMICS,
    PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE,
    PRIMARY_STATE_KEYS,
    SOURCE_KIND_GENERATED_NATIVE_LOOP,
    SourceFieldCoverage,
    VotePressureStepSpec,
    _cap_inputs_for_density_inputs,
    _density_inputs_for_step,
    _initial_states,
    default_vote_update_spec,
    pre_register_source_bindingness,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE,
    EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE,
    HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE,
    BoundedDeltaGuardSpec,
    BoundedDeltaMeasuredReport,
    BoundedDeltaOracleInput,
    BoundedDeltaReferenceReport,
    BoundedDeltaStorageProjection,
    _backlog_key_set,
    _build_measured_report_from_paths,
    _evaluate_bounded_delta_admission,
    _evaluate_guardrail,
    _identity_sha256,
    _run_reference_path,
    bounded_delta_admission_contract,
    bounded_delta_inclusive_ledger,
    compare_bounded_delta_paths_to_int16_oracle,
    compare_bounded_delta_step_to_int16_oracle,
    decode_bounded_accumulator_to_i16,
    encode_budget_capped_hybrid_reference,
    project_bounded_delta_accumulator_bpw,
)
from calm.hrm_text_158.native_full_stack.full_loop_receipt import (
    TINY_LOOP_GLOBAL_CAP,
    measure_tiny_two_projection_fixture_budget,
    tiny_full_loop_vote_update_spec,
    tiny_full_loop_votes_for_step,
    tiny_two_projection_vote_cap_fixture,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapSpec,
    tensor_offsets_for_vote_update_states,
)
from calm.hrm_text_158.native_full_stack.q_entropy_packing import (
    Base3QEntropyLedgerRow,
    default_base3_q_entropy_ledger_table,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateInputs, VoteUpdateState


REPRESENTATIVE_VERDICT_SCHEMA_VERSION = (
    "hrm_text_158_c1p1c_representative_bounded_delta_verdict/v0.cumulative"
)
REPRESENTATIVE_VERDICT_LABEL = (
    "c1p1c_representative_drift_vs_budget_cumulative_partial_for_s1"
)
ONE_STEP_LOCAL_DIAGNOSTIC_MODE = "one_step_local_diagnostic_only"
CUMULATIVE_SCHEDULE_MODE = "cumulative_exact_vs_bounded_carry_forward"
PRIMARY_CURVE_LABEL = "hot_max_backlog_k32"
HOT_BUDGET_POINT_LABELS = ("hot0", "hot64", "hot128", "hotmax")
BACKLOG_K_POLICIES = (0, 32)
STRICT_CONTROL_LABEL = "same_backlog_exact_control"
BOUNDED_BACKLOG_LABEL_TEMPLATE = "bounded_backlog_k{backlog_k}"
DEFAULT_TENSOR_METADATA_BITS_PER_INPUT = 64
DEFAULT_BUCKET_METADATA_BITS = 64
DEFAULT_SCALE_METADATA_BITS = 0
DEFAULT_GUARDRAIL_METADATA_BITS = 64
ORACLE_UPPER_BOUND_ADMISSION_DIAGNOSTIC = "oracle_upper_bound_admission_diagnostic"
ACCUMULATOR_FREE_NULL_BASELINE = "accumulator_free_null_baseline"
CANDIDATE_ADMISSION_DIAGNOSTIC_SCHEMA_VERSION = (
    "hrm_text_158_c1p1c_candidate_admission_diagnostic/v0"
)
CANDIDATE_ADMISSION_DIAGNOSTIC_LABEL = (
    "c1p1c_candidate_admission_oracle_upper_bound_diagnostic"
)
CAPACITY_LOCALIZATION_DIAGNOSTIC_SCHEMA_VERSION = (
    "hrm_text_158_c1p1c_capacity_localization_diagnostic/v0"
)
CAPACITY_LOCALIZATION_DIAGNOSTIC_LABEL = (
    "c1p1c_candidate_capacity_localization_oracle_upper_bound_diagnostic"
)
COARSE_SIGNED_CHARGE_BLOCK_SIZE = 8
A_COLD_EXCEPTION_BUDGET_LEVER_LABEL = (
    "cold_exception_budget_lever_requires_surface_faithful_tighter_encoding"
)
A_FUNDAMENTALLY_OVER_LABEL = "fundamentally_over_hot_cold_split_wrong_shape"
K_SWEEP_MINIMAL_VIABLE_PASS = "minimal_viable_k_upper_bound_pass"
K_SWEEP_JOINT_INFEASIBLE = "joint_infeasible_surface_faithful_breaks_sub2"
K_SWEEP_REPRESENTATION_WALL = "representation_level_capacity_wall"
REAL_BACKLOG_LOWER_BOUND_SCHEMA_VERSION = (
    "hrm_text_158_c1p1c_real_backlog_lower_bound_diagnostic/v0"
)
REAL_BACKLOG_LOWER_BOUND_LABEL = (
    "c1p1c_sparse_amortized_real_backlog_lower_bound_diagnostic"
)
PER_ROW_COMPRESSION_CLOSED_TINY_FIXTURE_LOWER_BOUND_ONLY = (
    "per_row_compression_closed_tiny_fixture_lower_bound_only"
)
PER_ROW_COMPRESSION_CLOSED_BY_EASY_CASE_LOWER_BOUND = (
    PER_ROW_COMPRESSION_CLOSED_TINY_FIXTURE_LOWER_BOUND_ONLY
)
SPARSE_AMORTIZED_CANDIDATE_RESURRECTED_FOR_HARDER_TRACE = (
    "sparse_amortized_candidate_resurrected_for_harder_trace"
)
REPRESENTATIVE_TRACE_UNDERPOWERED_FOR_CLOSURE = (
    "representative_trace_underpowered_for_closure"
)
TINY_FIXTURE_HEADROOM_SOURCE = "tiny_two_projection_fixture_budget"
LOWER_BOUND_TRACE_STOP_NONTRIVIAL = "nontrivial_backlog_reached"
LOWER_BOUND_TRACE_STOP_PLATEAU = "backlog_identity_plateau"
LOWER_BOUND_TRACE_STOP_CPU_SECONDS = "cpu_seconds_budget_hit"
LOWER_BOUND_TRACE_STOP_MAX_STEPS = "max_steps_budget_hit"
LOWER_BOUND_TRACE_PLATEAU_PATIENCE_STEPS = 2
LOWER_BOUND_TRACE_MAX_STEPS = 8
LOWER_BOUND_TRACE_MAX_SECONDS = 2.0
LOWER_BOUND_TRACE_NONTRIVIAL_BACKLOG_SIZE = 2
LOWER_BOUND_TRACE_NONTRIVIAL_UNIQUE_IDENTITIES = 2
LOWER_BOUND_TRACE_NONTRIVIAL_MEMBERSHIP_CHANGES = 2
SCALE_APPROPRIATE_B_STORAGE_SCHEMA_VERSION = (
    "hrm_text_158_c1p1d_scale_appropriate_b_storage_comparison/v0"
)
SCALE_APPROPRIATE_B_STORAGE_LABEL = (
    "c1p1d_scale_appropriate_b_storage_rate_comparison"
)
ABSOLUTE_COUNT_LOWER_BOUND_DIAGNOSTIC = "absolute_count_lower_bound_non_decisive"
RATE_HELD_B_STORAGE_DIAGNOSTIC = "rate_held_b_storage_bpw"
RATE_HELD_B_STILL_OVER_SCALE_HEADROOM_CANDIDATE_BRANCH_A = (
    "rate_held_b_still_over_scale_headroom_candidate_branch_a"
)
SCALE_APPROPRIATE_COMPARISON_AMBIGUOUS_NEEDS_BACKLOG_DENSITY_TRACE = (
    "scale_appropriate_comparison_ambiguous_needs_backlog_density_trace"
)
RATE_HELD_COUNT_ROUNDING_POLICY = (
    "ceil_positive_density_per_step_to_target_eligible_weight_count"
)
SCALE_REQUIRED_Q_LEDGER_ROWS = (
    "prior_large_fixture_base3_q",
    "illustrative_4096x4096_one_tensor_one_scale_base3_q",
)
SCALE_SENSITIVITY_Q_LEDGER_ROWS = (
    "illustrative_4096x4096_one_tensor_per_row_scale_base3_q",
)
DECISION_STATISTIC_UPPER_BOUND_SCHEMA_VERSION = (
    "hrm_text_158_c1p1e_decision_statistic_upper_bound_diagnostic/v0"
)
DECISION_STATISTIC_UPPER_BOUND_LABEL = (
    "c1p1e_branch_a_virtual_decision_statistic_upper_bound_diagnostic"
)
VIRTUAL_DECISION_STATISTIC_CANDIDATE = "branch_a_virtual_decision_statistic"
DECISION_STATISTIC_UPPER_BOUND_PASS = "decision_statistic_upper_bound_pass"
OBSERVABLE_RANK_FEATURES_INSUFFICIENT = "observable_rank_features_insufficient"
STATISTIC_BUDGET_BREAKS_SUB2 = "statistic_budget_breaks_sub2"
DECISION_STATISTIC_BUCKET_KEY_DIMENSIONS = (
    "state_key",
    "current_q_level",
    "move_direction",
)
DECISION_STATISTIC_COUNT_ONLY_MODE = "per_bucket_accepted_and_deferred_counts_only"
DECISION_STATISTIC_SHUFFLE_FALSIFIER = "per_bucket_reverse_order_tie_falsifier"
DECISION_STATISTIC_SEED_BITS = 0
DECISION_STATISTIC_CUTOFF_BIT_WIDTH = 0
DECISION_STATISTIC_METADATA_BITS = 64
TIE_FRONTIER_RESERVATION_SCHEMA_VERSION = (
    "hrm_text_158_c1p1f_tie_frontier_reservation_lower_bound_diagnostic/v0"
)
TIE_FRONTIER_RESERVATION_LABEL = (
    "c1p1f_tie_frontier_reservation_lower_bound_diagnostic"
)
TIE_FRONTIER_RESERVATION_CANDIDATE = (
    "branch_a_tie_frontier_exact_reservation_candidate_hybrid"
)
THEORETICAL_LOWER_BOUND_NON_DECISIVE = "theoretical_lower_bound_non_decisive"
TIE_MEMBERSHIP_MASK_ENCODING = "exact_tie_membership_mask"
TIE_SELECTED_OFFSET_ENCODING = (
    "accepted_selected_offsets_from_current_transient_rank_order"
)
OBSERVED_TIE_RESERVATION_DIAGNOSTIC = (
    "observed_frontier_tie_reservation_absolute_count_diagnostic_only"
)
RATE_HELD_TIE_RESERVATION_DIAGNOSTIC = (
    "joint_ta_rate_held_frontier_tie_reservation"
)
FULL_PLATEAU_JOINT_TA_SCALING_MODEL = (
    "full_plateau_joint_ta_density_rate_hold_from_observed_quantized_frontier"
)
TIE_RESERVATION_FITS_HEADROOM_CANDIDATE_HYBRID = (
    "tie_reservation_fits_headroom_candidate_hybrid"
)
TIE_FRONTIER_RESERVATION_FITS_HEADROOM_CANDIDATE_HYBRID = (
    TIE_RESERVATION_FITS_HEADROOM_CANDIDATE_HYBRID
)
TIE_RESERVATION_BREAKS_SUB2 = "tie_reservation_breaks_sub2"
TIE_DENSITY_AMBIGUOUS_NEEDS_TRACE = "tie_density_ambiguous_needs_trace"


def representative_engineering_guard_spec() -> BoundedDeltaGuardSpec:
    """Pre-registered diagnostic guard; adequacy is intentionally deferred to C2."""

    return BoundedDeltaGuardSpec(
        name="c1p1c_representative_diagnostic_guard_c2_adequacy_deferred",
        max_candidate_changed_fraction=0.05,
        max_accepted_changed_fraction=0.10,
        max_deferred_changed_fraction=0.10,
        max_q_changed_fraction=0.10,
        max_backlog_key_changed_fraction=0.10,
        max_cap_frontier_rank_delta=32,
        hot_risk_rows_require_zero_drift=True,
    )


def _prior_large_q_ledger() -> Base3QEntropyLedgerRow:
    for row in default_base3_q_entropy_ledger_table():
        if row.regime_name == "prior_large_fixture_base3_q":
            return row
    raise ValueError("prior_large_fixture_base3_q ledger row missing")


def _q_ledger_row_by_name(regime_name: str) -> Base3QEntropyLedgerRow:
    for row in default_base3_q_entropy_ledger_table():
        if row.regime_name == str(regime_name):
            return row
    raise ValueError(f"{regime_name} ledger row missing")


def _backlog_entry_count(backlog: Mapping[str, Mapping[int, Mapping[str, int]]]) -> int:
    return len(_backlog_key_set(backlog))


def _copy_backlog(
    backlog: Mapping[str, Mapping[int, Mapping[str, int]]],
) -> dict[str, dict[int, dict[str, int]]]:
    return {
        str(state_key): {int(index): dict(entry) for index, entry in by_index.items()}
        for state_key, by_index in backlog.items()
    }


def _states_from_path(path: Any) -> dict[str, VoteUpdateState]:
    return {
        state_key: VoteUpdateState(
            q_levels=path.output_q_by_key[state_key].detach().clone().contiguous(),
            accumulators=path.output_acc_by_key[state_key].detach().clone().contiguous(),
        )
        for state_key in PRIMARY_STATE_KEYS
    }


def _make_step_inputs(
    states: Mapping[str, VoteUpdateState],
    schedule_step: Any,
) -> tuple[list[BoundedDeltaOracleInput], dict[str, int]]:
    spec = default_vote_update_spec()
    density_inputs = _density_inputs_for_step(states, schedule_step, spec)
    cap_inputs = _cap_inputs_for_density_inputs(density_inputs)
    offsets = tensor_offsets_for_vote_update_states(cap_inputs)
    return [
        BoundedDeltaOracleInput(
            state_key=item.state_key,
            state=item.state,
            vote_inputs=item.vote_inputs,
            vote_spec=item.spec,
        )
        for item in density_inputs
    ], offsets


def _unique_row_identities(rows: Sequence[Any]) -> list[tuple[str, int]]:
    seen: set[tuple[str, int]] = set()
    out: list[tuple[str, int]] = []
    for row in rows:
        identity = (str(row.state_key), int(row.flat_index))
        if identity in seen:
            continue
        seen.add(identity)
        out.append(identity)
    return out


def _hot_indices_by_state_from_exact_path(path: Any, hot_budget: int) -> dict[str, tuple[int, ...]]:
    if path.cap_result is None:
        raise ValueError("representative verdict requires a global cap result")
    budget = max(0, int(hot_budget))
    priority_rows = (
        list(path.cap_result.accepted_rows)
        + list(path.cap_result.deferred_rows)
        + list(path.cap_result.rows)
    )
    identities = _unique_row_identities(priority_rows)[:budget]
    by_state: dict[str, list[int]] = {key: [] for key in PRIMARY_STATE_KEYS}
    for state_key, flat_index in identities:
        by_state.setdefault(state_key, []).append(int(flat_index))
    return {state_key: tuple(indices) for state_key, indices in by_state.items()}


def _all_candidate_hot_indices_by_state(path: Any) -> dict[str, tuple[int, ...]]:
    by_state: dict[str, list[int]] = {key: [] for key in PRIMARY_STATE_KEYS}
    for state_key, flat_index in sorted(path.candidate_ids):
        by_state.setdefault(state_key, []).append(int(flat_index))
    return {state_key: tuple(indices) for state_key, indices in by_state.items()}


def _hot_count(hot_by_state: Mapping[str, Sequence[int]]) -> int:
    return sum(len(tuple(indices)) for indices in hot_by_state.values())


def _select_stored_backlog(
    backlog: Mapping[str, Mapping[int, Mapping[str, int]]],
    *,
    priority_identities: Sequence[tuple[str, int]],
    max_entries: int,
) -> dict[str, dict[int, dict[str, int]]]:
    limit = int(max_entries)
    if limit < 0:
        raise ValueError("bounded backlog K policy must be >= 0")
    if limit == 0:
        return {}
    available = _backlog_key_set(backlog)
    ordered: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for identity in priority_identities:
        normalized = (str(identity[0]), int(identity[1]))
        if normalized in available and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    for identity in sorted(available):
        if identity not in seen:
            ordered.append(identity)
    keep = set(ordered[:limit])
    stored: dict[str, dict[int, dict[str, int]]] = {}
    for state_key, by_index in backlog.items():
        for flat_index, entry in by_index.items():
            identity = (str(state_key), int(flat_index))
            if identity in keep:
                stored.setdefault(str(state_key), {})[int(flat_index)] = dict(entry)
    return stored


def _max_hot_budget_for_backlog_k(
    q_ledger: Base3QEntropyLedgerRow,
    *,
    backlog_k: int,
    tensor_count: int,
) -> int:
    eligible = int(q_ledger.eligible_weight_count)
    index_bits = int(math.ceil(math.log2(float(eligible))))
    hot_bits_per_row = index_bits + 16 + 2
    backlog_bits_per_entry = index_bits + 16 + 16
    metadata_bits = (
        int(tensor_count) * DEFAULT_TENSOR_METADATA_BITS_PER_INPUT
        + DEFAULT_BUCKET_METADATA_BITS
        + DEFAULT_SCALE_METADATA_BITS
        + DEFAULT_GUARDRAIL_METADATA_BITS
    )
    remaining_bits = (
        float(q_ledger.remaining_accumulator_budget_bits_per_weight) * float(eligible)
        - float(metadata_bits)
        - float(int(backlog_k) * backlog_bits_per_entry)
    )
    candidate = max(0, min(eligible, int(math.floor((remaining_bits - 1e-9) / hot_bits_per_row))))
    while candidate > 0:
        projection = project_bounded_delta_accumulator_bpw(
            eligible_weight_count=eligible,
            hot_exact_row_count=candidate,
            backlog_entry_count=int(backlog_k),
            tensor_metadata_bits=int(tensor_count) * DEFAULT_TENSOR_METADATA_BITS_PER_INPUT,
            bucket_metadata_bits=DEFAULT_BUCKET_METADATA_BITS,
            scale_metadata_bits=DEFAULT_SCALE_METADATA_BITS,
            guardrail_metadata_bits=DEFAULT_GUARDRAIL_METADATA_BITS,
        )
        ledger = bounded_delta_inclusive_ledger(q_ledger, projection)
        if ledger.claimable_physical_sub2:
            return candidate
        candidate -= 1
    return 0


def _requested_hot_budget(label: str, max_hot_budget: int) -> tuple[int, bool]:
    if label == "hotmax":
        return int(max_hot_budget), True
    if not label.startswith("hot"):
        raise ValueError(f"unexpected hot budget label {label!r}")
    return min(int(label.removeprefix("hot")), int(max_hot_budget)), False


def _build_cumulative_reference_report(
    *,
    exact_path: Any,
    bounded_path: Any,
    exact_input_states: Mapping[str, VoteUpdateState],
    bounded_input_states: Mapping[str, VoteUpdateState],
    exact_backlog: Mapping[str, Mapping[int, Mapping[str, int]]],
    bounded_input_backlog: Mapping[str, Mapping[int, Mapping[str, int]]],
    bounded_stored_backlog: Mapping[str, Mapping[int, Mapping[str, int]]],
    inputs: Sequence[BoundedDeltaOracleInput],
    q_ledger_row: Base3QEntropyLedgerRow,
    projection: BoundedDeltaStorageProjection,
    guard_spec: BoundedDeltaGuardSpec,
    global_cap_spec: GlobalRateCapSpec,
    tensor_offsets: Mapping[str, int],
    hot_by_state: Mapping[str, Sequence[int]],
) -> BoundedDeltaReferenceReport:
    report_inputs = tuple(
        BoundedDeltaOracleInput(
            state_key=item.state_key,
            state=bounded_input_states[item.state_key],
            vote_inputs=item.vote_inputs,
            vote_spec=item.vote_spec,
            hot_exact_indices=tuple(int(idx) for idx in hot_by_state.get(item.state_key, ())),
            cold_default_value=0,
        )
        for item in inputs
    )
    return compare_bounded_delta_paths_to_int16_oracle(
        inputs=report_inputs,
        q_ledger_row=q_ledger_row,
        exact_path=exact_path,
        bounded_path=bounded_path,
        storage_projection=projection,
        guard_spec=guard_spec,
        candidate_name=HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE,
        global_cap_spec=global_cap_spec,
        exact_input_states=exact_input_states,
        bounded_input_states=bounded_input_states,
        exact_input_backlog=exact_backlog,
        bounded_input_backlog=bounded_input_backlog,
        bounded_stored_backlog=bounded_stored_backlog,
        tensor_offsets=tensor_offsets,
        bounded_backlog_policy_active=True,
        path_difference=(
            "cumulative path differs by bounded accumulator encode_decode, "
            "bounded-backlog encode/drop, and prior bounded q/acc/backlog carry-forward"
        ),
        oracle_parity_overrides={
            "cumulative_carry_forward": True,
            "bounded_reinitialized_from_exact": False,
            "tensor_offsets_checksum": str(math.fsum(float(value) for value in tensor_offsets.values())),
        },
        non_claims=(
            "cumulative representative verdict over generated in-tree schedule only",
            "no production vote_update/global_rate_cap replacement",
            "no GPU lane",
            "no trainer/live-run/checkpoint/creditdir mutation",
            "no acquisition, retention, or stability claim",
            "guard-bound adequacy deferred to C2",
            "compact counts/hashes only; no raw per-weight arrays",
        ),
    )


@dataclass(frozen=True)
class RepresentativeStepReport:
    schedule_name: str
    step: int
    mode: str
    curve_label: str
    hot_budget_label: str
    requested_hot_budget: int
    is_max_hot_budget_point: bool
    max_hot_budget_for_policy: int
    actual_hot_exact_row_count: int
    backlog_policy_k: int | None
    exact_backlog_entry_count: int
    bounded_stored_backlog_entry_count: int
    bounded_reinitialized_from_exact: bool
    classification: str
    guard_passed: bool
    failed_metrics: tuple[str, ...]
    bounded_delta_report: BoundedDeltaReferenceReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_name": self.schedule_name,
            "step": int(self.step),
            "mode": self.mode,
            "curve_label": self.curve_label,
            "hot_budget_label": self.hot_budget_label,
            "requested_hot_budget": int(self.requested_hot_budget),
            "is_max_hot_budget_point": bool(self.is_max_hot_budget_point),
            "max_hot_budget_for_policy": int(self.max_hot_budget_for_policy),
            "actual_hot_exact_row_count": int(self.actual_hot_exact_row_count),
            "backlog_policy_k": self.backlog_policy_k,
            "exact_backlog_entry_count": int(self.exact_backlog_entry_count),
            "bounded_stored_backlog_entry_count": int(self.bounded_stored_backlog_entry_count),
            "bounded_reinitialized_from_exact": bool(self.bounded_reinitialized_from_exact),
            "classification": self.classification,
            "guard_passed": bool(self.guard_passed),
            "failed_metrics": list(self.failed_metrics),
            "bounded_delta_report": self.bounded_delta_report.to_dict(),
        }


@dataclass(frozen=True)
class RepresentativeCurveRunReport:
    curve_label: str
    mode: str
    hot_budget_label: str
    requested_hot_budget: int
    is_max_hot_budget_point: bool
    max_hot_budget_for_policy: int
    backlog_policy_k: int
    per_step_reports: tuple[RepresentativeStepReport, ...]
    terminal_classification: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "curve_label": self.curve_label,
            "mode": self.mode,
            "hot_budget_label": self.hot_budget_label,
            "requested_hot_budget": int(self.requested_hot_budget),
            "is_max_hot_budget_point": bool(self.is_max_hot_budget_point),
            "max_hot_budget_for_policy": int(self.max_hot_budget_for_policy),
            "backlog_policy_k": int(self.backlog_policy_k),
            "per_step_reports": [step.to_dict() for step in self.per_step_reports],
            "terminal_classification": self.terminal_classification,
        }


@dataclass(frozen=True)
class RepresentativeDriftVerdictReport:
    schema_version: str
    label: str
    terminal_mode: str
    terminal_science_question_closed: bool
    terminal_classification: str
    primary_curve_label: str
    source_bindingness: Any
    field_coverage: SourceFieldCoverage
    q_ledger_regime_name: str
    guard_spec: BoundedDeltaGuardSpec
    hot_budget_points: tuple[str, ...]
    backlog_k_policies: tuple[int, ...]
    one_step_local_diagnostic_reports: tuple[RepresentativeStepReport, ...]
    cumulative_curve_reports: tuple[RepresentativeCurveRunReport, ...]
    bindingness_statement: str
    residual_diversity_caveat: str
    guard_bound_adequacy_statement: str
    raw_arrays_included: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "terminal_mode": self.terminal_mode,
            "terminal_science_question_closed": bool(self.terminal_science_question_closed),
            "terminal_classification": self.terminal_classification,
            "primary_curve_label": self.primary_curve_label,
            "source_bindingness": self.source_bindingness.to_dict(),
            "field_coverage": self.field_coverage.to_dict(),
            "q_ledger_regime_name": self.q_ledger_regime_name,
            "guard_spec": self.guard_spec.to_dict(),
            "hot_budget_points": list(self.hot_budget_points),
            "backlog_k_policies": list(self.backlog_k_policies),
            "one_step_local_diagnostic_reports": [
                step.to_dict() for step in self.one_step_local_diagnostic_reports
            ],
            "cumulative_curve_reports": [run.to_dict() for run in self.cumulative_curve_reports],
            "bindingness_statement": self.bindingness_statement,
            "residual_diversity_caveat": self.residual_diversity_caveat,
            "guard_bound_adequacy_statement": self.guard_bound_adequacy_statement,
            "raw_arrays_included": bool(self.raw_arrays_included),
            "non_claims": list(self.non_claims),
        }


@dataclass(frozen=True)
class AdmissionNullBaselineComparison:
    candidate_beats_null: bool
    compared_surfaces: tuple[str, ...]
    strict_improvement_surfaces: tuple[str, ...]
    regressed_surfaces: tuple[str, ...]
    candidate_surface_counts: dict[str, int]
    null_surface_counts: dict[str, int]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_beats_null": bool(self.candidate_beats_null),
            "compared_surfaces": list(self.compared_surfaces),
            "strict_improvement_surfaces": list(self.strict_improvement_surfaces),
            "regressed_surfaces": list(self.regressed_surfaces),
            "candidate_surface_counts": dict(self.candidate_surface_counts),
            "null_surface_counts": dict(self.null_surface_counts),
            "summary": self.summary,
        }


@dataclass(frozen=True)
class BacklogTruncationAttribution:
    bounded_input_truncation_count: int
    bounded_input_truncation_identities_sha256: str
    bounded_stored_truncation_count: int
    bounded_stored_truncation_identities_sha256: str
    paired_rejection_summary: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "bounded_input_truncation_count": int(self.bounded_input_truncation_count),
            "bounded_input_truncation_identities_sha256": self.bounded_input_truncation_identities_sha256,
            "bounded_stored_truncation_count": int(self.bounded_stored_truncation_count),
            "bounded_stored_truncation_identities_sha256": self.bounded_stored_truncation_identities_sha256,
            "paired_rejection_summary": self.paired_rejection_summary,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class CandidateAdmissionDiagnosticStepReport:
    schedule_name: str
    step: int
    candidate_name: str
    builder_label: str
    backlog_policy_k: int | None
    bounded_delta_report: BoundedDeltaReferenceReport
    null_baseline_comparison: AdmissionNullBaselineComparison
    backlog_truncation_attribution: BacklogTruncationAttribution

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_name": self.schedule_name,
            "step": int(self.step),
            "candidate_name": self.candidate_name,
            "builder_label": self.builder_label,
            "backlog_policy_k": self.backlog_policy_k,
            "bounded_delta_report": self.bounded_delta_report.to_dict(),
            "null_baseline_comparison": self.null_baseline_comparison.to_dict(),
            "backlog_truncation_attribution": self.backlog_truncation_attribution.to_dict(),
        }


@dataclass(frozen=True)
class CandidatePromotionDecision:
    candidate_name: str
    earns_dyn200_consideration: bool
    status: str
    failed_step_names: tuple[str, ...]
    oracle_upper_bound_only: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "earns_dyn200_consideration": bool(self.earns_dyn200_consideration),
            "status": self.status,
            "failed_step_names": list(self.failed_step_names),
            "oracle_upper_bound_only": bool(self.oracle_upper_bound_only),
        }


@dataclass(frozen=True)
class CandidateAdmissionDiagnosticRunReport:
    candidate_name: str
    builder_label: str
    backlog_policy_k: int | None
    per_step_reports: tuple[CandidateAdmissionDiagnosticStepReport, ...]
    terminal_decision: CandidatePromotionDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "builder_label": self.builder_label,
            "backlog_policy_k": self.backlog_policy_k,
            "per_step_reports": [step.to_dict() for step in self.per_step_reports],
            "terminal_decision": self.terminal_decision.to_dict(),
        }


@dataclass(frozen=True)
class CandidateAdmissionDiagnosticReport:
    schema_version: str
    label: str
    source_bindingness: Any
    field_coverage: SourceFieldCoverage
    q_ledger_regime_name: str
    guard_spec: BoundedDeltaGuardSpec
    null_baseline_label: str
    pre_registered_schedule: tuple[VotePressureStepSpec, ...]
    candidate_runs: tuple[CandidateAdmissionDiagnosticRunReport, ...]
    raw_arrays_included: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "source_bindingness": self.source_bindingness.to_dict(),
            "field_coverage": self.field_coverage.to_dict(),
            "q_ledger_regime_name": self.q_ledger_regime_name,
            "guard_spec": self.guard_spec.to_dict(),
            "null_baseline_label": self.null_baseline_label,
            "pre_registered_schedule": [step.to_dict() for step in self.pre_registered_schedule],
            "candidate_runs": [run.to_dict() for run in self.candidate_runs],
            "raw_arrays_included": bool(self.raw_arrays_included),
            "non_claims": list(self.non_claims),
        }


@dataclass(frozen=True)
class CandidateCapacityStepReport:
    schedule_name: str
    step: int
    k_label: str
    k_value: int | None
    bounded_delta_report: BoundedDeltaReferenceReport
    backlog_truncation_attribution: BacklogTruncationAttribution
    protected_surface_destructive_approximation_present: bool
    surface_fidelity_clears: bool
    packed_inclusive_physical_bits_per_weight: float
    delta_over_2bpw: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_name": self.schedule_name,
            "step": int(self.step),
            "k_label": self.k_label,
            "k_value": self.k_value,
            "bounded_delta_report": self.bounded_delta_report.to_dict(),
            "backlog_truncation_attribution": self.backlog_truncation_attribution.to_dict(),
            "protected_surface_destructive_approximation_present": bool(
                self.protected_surface_destructive_approximation_present
            ),
            "surface_fidelity_clears": bool(self.surface_fidelity_clears),
            "packed_inclusive_physical_bits_per_weight": float(
                self.packed_inclusive_physical_bits_per_weight
            ),
            "delta_over_2bpw": float(self.delta_over_2bpw),
        }


@dataclass(frozen=True)
class CandidateABudgetReadout:
    schedule_name: str
    step: int
    packed_inclusive_physical_bits_per_weight: float
    delta_over_2bpw: float
    hot_exact_bits: int
    cold_exception_bits: int
    backlog_bits: int
    metadata_bits: int
    dense_cold_bits: float
    original_classification: str
    original_rejection_summary: str
    cold_zero_counterfactual_bits_per_weight: float
    cold_zero_counterfactual_delta_over_2bpw: float
    cold_zero_counterfactual_clears_sub2: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_name": self.schedule_name,
            "step": int(self.step),
            "packed_inclusive_physical_bits_per_weight": float(
                self.packed_inclusive_physical_bits_per_weight
            ),
            "delta_over_2bpw": float(self.delta_over_2bpw),
            "hot_exact_bits": int(self.hot_exact_bits),
            "cold_exception_bits": int(self.cold_exception_bits),
            "backlog_bits": int(self.backlog_bits),
            "metadata_bits": int(self.metadata_bits),
            "dense_cold_bits": float(self.dense_cold_bits),
            "original_classification": self.original_classification,
            "original_rejection_summary": self.original_rejection_summary,
            "cold_zero_counterfactual_bits_per_weight": float(
                self.cold_zero_counterfactual_bits_per_weight
            ),
            "cold_zero_counterfactual_delta_over_2bpw": float(
                self.cold_zero_counterfactual_delta_over_2bpw
            ),
            "cold_zero_counterfactual_clears_sub2": bool(
                self.cold_zero_counterfactual_clears_sub2
            ),
        }


@dataclass(frozen=True)
class CandidateABudgetLocalizationReport:
    candidate_name: str
    per_step_readouts: tuple[CandidateABudgetReadout, ...]
    terminal_budget_direction_label: str
    original_terminal_classification: str
    original_terminal_rejection_summary: str
    cold_zero_counterfactual_terminal_bits_per_weight: float
    cold_zero_counterfactual_terminal_delta_over_2bpw: float
    cold_zero_counterfactual_terminal_clears_sub2: bool
    non_claim: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "per_step_readouts": [step.to_dict() for step in self.per_step_readouts],
            "terminal_budget_direction_label": self.terminal_budget_direction_label,
            "original_terminal_classification": self.original_terminal_classification,
            "original_terminal_rejection_summary": self.original_terminal_rejection_summary,
            "cold_zero_counterfactual_terminal_bits_per_weight": float(
                self.cold_zero_counterfactual_terminal_bits_per_weight
            ),
            "cold_zero_counterfactual_terminal_delta_over_2bpw": float(
                self.cold_zero_counterfactual_terminal_delta_over_2bpw
            ),
            "cold_zero_counterfactual_terminal_clears_sub2": bool(
                self.cold_zero_counterfactual_terminal_clears_sub2
            ),
            "non_claim": self.non_claim,
        }


@dataclass(frozen=True)
class CandidateKSweepEntry:
    candidate_name: str
    k_label: str
    k_value: int | None
    per_step_reports: tuple[CandidateCapacityStepReport, ...]
    all_steps_surface_fidelity_clears: bool
    all_steps_claimable_physical_sub2_with_guardrail: bool
    terminal_surface_fidelity_clears: bool
    terminal_claimable_physical_sub2_with_guardrail: bool
    terminal_protected_surface_destructive_approximation_present: bool
    terminal_packed_inclusive_physical_bits_per_weight: float
    terminal_delta_over_2bpw: float
    terminal_rejection_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "k_label": self.k_label,
            "k_value": self.k_value,
            "per_step_reports": [step.to_dict() for step in self.per_step_reports],
            "all_steps_surface_fidelity_clears": bool(self.all_steps_surface_fidelity_clears),
            "all_steps_claimable_physical_sub2_with_guardrail": bool(
                self.all_steps_claimable_physical_sub2_with_guardrail
            ),
            "terminal_surface_fidelity_clears": bool(self.terminal_surface_fidelity_clears),
            "terminal_claimable_physical_sub2_with_guardrail": bool(
                self.terminal_claimable_physical_sub2_with_guardrail
            ),
            "terminal_protected_surface_destructive_approximation_present": bool(
                self.terminal_protected_surface_destructive_approximation_present
            ),
            "terminal_packed_inclusive_physical_bits_per_weight": float(
                self.terminal_packed_inclusive_physical_bits_per_weight
            ),
            "terminal_delta_over_2bpw": float(self.terminal_delta_over_2bpw),
            "terminal_rejection_summary": self.terminal_rejection_summary,
        }


@dataclass(frozen=True)
class CandidateKSweepDecision:
    candidate_name: str
    status: str
    decisive_k_label: str
    decisive_k_value: int | None
    oracle_upper_bound_only: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "status": self.status,
            "decisive_k_label": self.decisive_k_label,
            "decisive_k_value": self.decisive_k_value,
            "oracle_upper_bound_only": bool(self.oracle_upper_bound_only),
        }


@dataclass(frozen=True)
class CandidateKSweepRunReport:
    candidate_name: str
    sweep_entries: tuple[CandidateKSweepEntry, ...]
    terminal_decision: CandidateKSweepDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "sweep_entries": [entry.to_dict() for entry in self.sweep_entries],
            "terminal_decision": self.terminal_decision.to_dict(),
        }


@dataclass(frozen=True)
class CandidateCapacityLocalizationReport:
    schema_version: str
    label: str
    source_bindingness: Any
    field_coverage: SourceFieldCoverage
    q_ledger_regime_name: str
    guard_spec: BoundedDeltaGuardSpec
    candidate_a_budget_report: CandidateABudgetLocalizationReport
    backlog_k_schedule: tuple[str, ...]
    sweep_runs: tuple[CandidateKSweepRunReport, ...]
    raw_arrays_included: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "source_bindingness": self.source_bindingness.to_dict(),
            "field_coverage": self.field_coverage.to_dict(),
            "q_ledger_regime_name": self.q_ledger_regime_name,
            "guard_spec": self.guard_spec.to_dict(),
            "candidate_a_budget_report": self.candidate_a_budget_report.to_dict(),
            "backlog_k_schedule": list(self.backlog_k_schedule),
            "sweep_runs": [run.to_dict() for run in self.sweep_runs],
            "raw_arrays_included": bool(self.raw_arrays_included),
            "non_claims": list(self.non_claims),
        }


@dataclass(frozen=True)
class RealBacklogTraceStepReport:
    schedule_name: str
    step: int
    vote_pattern_step: int
    pre_cap_demand_count: int
    exact_candidate_count: int
    exact_accepted_count: int
    exact_deferred_count: int
    exact_fired_count: int
    exact_output_backlog_count: int
    exact_output_backlog_identities_sha256: str
    backlog_membership_changed_count_from_prior_step: int
    cumulative_unique_backlog_identity_count: int
    backlog_max_age_steps: int
    backlog_max_defer_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_name": self.schedule_name,
            "step": int(self.step),
            "vote_pattern_step": int(self.vote_pattern_step),
            "pre_cap_demand_count": int(self.pre_cap_demand_count),
            "exact_candidate_count": int(self.exact_candidate_count),
            "exact_accepted_count": int(self.exact_accepted_count),
            "exact_deferred_count": int(self.exact_deferred_count),
            "exact_fired_count": int(self.exact_fired_count),
            "exact_output_backlog_count": int(self.exact_output_backlog_count),
            "exact_output_backlog_identities_sha256": self.exact_output_backlog_identities_sha256,
            "backlog_membership_changed_count_from_prior_step": int(
                self.backlog_membership_changed_count_from_prior_step
            ),
            "cumulative_unique_backlog_identity_count": int(
                self.cumulative_unique_backlog_identity_count
            ),
            "backlog_max_age_steps": int(self.backlog_max_age_steps),
            "backlog_max_defer_count": int(self.backlog_max_defer_count),
        }


@dataclass(frozen=True)
class RealBacklogTraceSummaryReport:
    stop_reason: str
    stop_step: int
    plateau_patience_steps: int
    max_steps_budget: int
    cpu_seconds_budget: float
    elapsed_seconds: float
    saw_any_backlog: bool
    nontrivial_backlog_reached: bool
    plateau_detected: bool
    max_exact_output_backlog_count: int
    cumulative_unique_backlog_identity_count: int
    cumulative_backlog_membership_changed_count: int
    max_exact_backlog_age_steps: int
    max_exact_backlog_defer_count: int
    per_step_reports: tuple[RealBacklogTraceStepReport, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stop_reason": self.stop_reason,
            "stop_step": int(self.stop_step),
            "plateau_patience_steps": int(self.plateau_patience_steps),
            "max_steps_budget": int(self.max_steps_budget),
            "cpu_seconds_budget": float(self.cpu_seconds_budget),
            "elapsed_seconds": float(self.elapsed_seconds),
            "saw_any_backlog": bool(self.saw_any_backlog),
            "nontrivial_backlog_reached": bool(self.nontrivial_backlog_reached),
            "plateau_detected": bool(self.plateau_detected),
            "max_exact_output_backlog_count": int(self.max_exact_output_backlog_count),
            "cumulative_unique_backlog_identity_count": int(
                self.cumulative_unique_backlog_identity_count
            ),
            "cumulative_backlog_membership_changed_count": int(
                self.cumulative_backlog_membership_changed_count
            ),
            "max_exact_backlog_age_steps": int(self.max_exact_backlog_age_steps),
            "max_exact_backlog_defer_count": int(self.max_exact_backlog_defer_count),
            "per_step_reports": [step.to_dict() for step in self.per_step_reports],
        }


@dataclass(frozen=True)
class RealBacklogLowerBoundStepReport:
    schedule_name: str
    step: int
    k_label: str
    k_value: int | None
    guard_passed: bool
    admission_passed: bool
    surface_fidelity_clears: bool
    failed_metrics: tuple[str, ...]
    admission_failed_surfaces: tuple[str, ...]
    rejection_summary: str
    protected_surface_destructive_approximation_present: bool
    bounded_delta_acc_bits_per_weight: float
    backlog_entry_count: int
    hot_exact_row_count: int
    event_delta_count: int
    measured_report: BoundedDeltaMeasuredReport
    backlog_truncation_attribution: BacklogTruncationAttribution

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_name": self.schedule_name,
            "step": int(self.step),
            "k_label": self.k_label,
            "k_value": self.k_value,
            "guard_passed": bool(self.guard_passed),
            "admission_passed": bool(self.admission_passed),
            "surface_fidelity_clears": bool(self.surface_fidelity_clears),
            "failed_metrics": list(self.failed_metrics),
            "admission_failed_surfaces": list(self.admission_failed_surfaces),
            "rejection_summary": self.rejection_summary,
            "protected_surface_destructive_approximation_present": bool(
                self.protected_surface_destructive_approximation_present
            ),
            "bounded_delta_acc_bits_per_weight": float(self.bounded_delta_acc_bits_per_weight),
            "backlog_entry_count": int(self.backlog_entry_count),
            "hot_exact_row_count": int(self.hot_exact_row_count),
            "event_delta_count": int(self.event_delta_count),
            "measured_report": self.measured_report.to_dict(),
            "backlog_truncation_attribution": self.backlog_truncation_attribution.to_dict(),
        }


@dataclass(frozen=True)
class RealBacklogLowerBoundSweepEntry:
    candidate_name: str
    k_label: str
    k_value: int | None
    per_step_reports: tuple[RealBacklogLowerBoundStepReport, ...]
    all_steps_surface_fidelity_clears: bool
    peak_bounded_delta_acc_bits_per_weight: float
    terminal_bounded_delta_acc_bits_per_weight: float
    terminal_rejection_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "k_label": self.k_label,
            "k_value": self.k_value,
            "per_step_reports": [step.to_dict() for step in self.per_step_reports],
            "all_steps_surface_fidelity_clears": bool(self.all_steps_surface_fidelity_clears),
            "peak_bounded_delta_acc_bits_per_weight": float(
                self.peak_bounded_delta_acc_bits_per_weight
            ),
            "terminal_bounded_delta_acc_bits_per_weight": float(
                self.terminal_bounded_delta_acc_bits_per_weight
            ),
            "terminal_rejection_summary": self.terminal_rejection_summary,
        }


@dataclass(frozen=True)
class RealBacklogLowerBoundDecision:
    terminal_label: str
    headroom_source: str
    eligible_weight_count: int
    q_packed_data_bits_per_weight: float
    q_packed_metadata_bits_per_weight: float
    q_packed_total_bits_per_weight: float
    frozen_scale_fp32_bits_per_weight: float
    actual_remaining_accumulator_headroom_bits_per_weight: float
    minimal_surface_faithful_k_label: str | None
    minimal_surface_faithful_k_value: int | None
    minimal_surface_faithful_peak_bounded_delta_acc_bits_per_weight: float | None
    headroom_minus_minimal_surface_faithful_peak_bits_per_weight: float | None
    minimal_surface_faithful_k_fits_headroom: bool
    global_per_row_compression_closed: bool
    branch_a_trigger: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "terminal_label": self.terminal_label,
            "headroom_source": self.headroom_source,
            "eligible_weight_count": int(self.eligible_weight_count),
            "q_packed_data_bits_per_weight": float(self.q_packed_data_bits_per_weight),
            "q_packed_metadata_bits_per_weight": float(self.q_packed_metadata_bits_per_weight),
            "q_packed_total_bits_per_weight": float(self.q_packed_total_bits_per_weight),
            "frozen_scale_fp32_bits_per_weight": float(
                self.frozen_scale_fp32_bits_per_weight
            ),
            "actual_remaining_accumulator_headroom_bits_per_weight": float(
                self.actual_remaining_accumulator_headroom_bits_per_weight
            ),
            "minimal_surface_faithful_k_label": self.minimal_surface_faithful_k_label,
            "minimal_surface_faithful_k_value": self.minimal_surface_faithful_k_value,
            "minimal_surface_faithful_peak_bounded_delta_acc_bits_per_weight": (
                None
                if self.minimal_surface_faithful_peak_bounded_delta_acc_bits_per_weight is None
                else float(self.minimal_surface_faithful_peak_bounded_delta_acc_bits_per_weight)
            ),
            "headroom_minus_minimal_surface_faithful_peak_bits_per_weight": (
                None
                if self.headroom_minus_minimal_surface_faithful_peak_bits_per_weight is None
                else float(self.headroom_minus_minimal_surface_faithful_peak_bits_per_weight)
            ),
            "minimal_surface_faithful_k_fits_headroom": bool(
                self.minimal_surface_faithful_k_fits_headroom
            ),
            "global_per_row_compression_closed": bool(
                self.global_per_row_compression_closed
            ),
            "branch_a_trigger": bool(self.branch_a_trigger),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RealBacklogLowerBoundReport:
    schema_version: str
    label: str
    source_bindingness: Any
    field_coverage: SourceFieldCoverage
    q_persistent_budget_label: str
    candidate_name: str
    guard_spec: BoundedDeltaGuardSpec
    exact_trace_summary: RealBacklogTraceSummaryReport
    backlog_k_schedule: tuple[str, ...]
    sweep_entries: tuple[RealBacklogLowerBoundSweepEntry, ...]
    terminal_decision: RealBacklogLowerBoundDecision
    raw_arrays_included: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "source_bindingness": self.source_bindingness.to_dict(),
            "field_coverage": self.field_coverage.to_dict(),
            "q_persistent_budget_label": self.q_persistent_budget_label,
            "candidate_name": self.candidate_name,
            "guard_spec": self.guard_spec.to_dict(),
            "exact_trace_summary": self.exact_trace_summary.to_dict(),
            "backlog_k_schedule": list(self.backlog_k_schedule),
            "sweep_entries": [entry.to_dict() for entry in self.sweep_entries],
            "terminal_decision": self.terminal_decision.to_dict(),
            "raw_arrays_included": bool(self.raw_arrays_included),
            "non_claims": list(self.non_claims),
        }


@dataclass(frozen=True)
class ScaleAppropriateProjectionStepReport:
    schedule_name: str
    step: int
    projection_label: str
    source_hot_exact_row_count: int
    source_event_delta_count: int
    source_backlog_entry_count: int
    target_hot_exact_row_count: int
    target_event_delta_count: int
    target_backlog_entry_count: int
    target_index_bits_per_row: int
    tensor_metadata_bits: int
    bucket_metadata_bits: int
    scale_metadata_bits: int
    guardrail_metadata_bits: int
    bounded_delta_acc_bits_per_weight: float
    exceeds_scale_headroom: bool
    decisive_for_branch: bool
    rounding_policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_name": self.schedule_name,
            "step": int(self.step),
            "projection_label": self.projection_label,
            "source_hot_exact_row_count": int(self.source_hot_exact_row_count),
            "source_event_delta_count": int(self.source_event_delta_count),
            "source_backlog_entry_count": int(self.source_backlog_entry_count),
            "target_hot_exact_row_count": int(self.target_hot_exact_row_count),
            "target_event_delta_count": int(self.target_event_delta_count),
            "target_backlog_entry_count": int(self.target_backlog_entry_count),
            "target_index_bits_per_row": int(self.target_index_bits_per_row),
            "tensor_metadata_bits": int(self.tensor_metadata_bits),
            "bucket_metadata_bits": int(self.bucket_metadata_bits),
            "scale_metadata_bits": int(self.scale_metadata_bits),
            "guardrail_metadata_bits": int(self.guardrail_metadata_bits),
            "bounded_delta_acc_bits_per_weight": float(
                self.bounded_delta_acc_bits_per_weight
            ),
            "exceeds_scale_headroom": bool(self.exceeds_scale_headroom),
            "decisive_for_branch": bool(self.decisive_for_branch),
            "rounding_policy": self.rounding_policy,
        }


@dataclass(frozen=True)
class ScaleAppropriateLedgerComparisonReport:
    q_regime_name: str
    row_role: str
    eligible_weight_count: int
    q_packed_data_bits_per_weight: float
    q_packed_metadata_bits_per_weight: float
    q_packed_total_bits_per_weight: float
    frozen_scale_fp32_bits_per_weight: float
    scale_appropriate_headroom_bits_per_weight: float
    density_assumption: str
    absolute_count_lower_bound_step_reports: tuple[ScaleAppropriateProjectionStepReport, ...]
    rate_held_b_storage_step_reports: tuple[ScaleAppropriateProjectionStepReport, ...]
    absolute_count_lower_bound_peak_bounded_delta_acc_bits_per_weight: float
    rate_held_b_storage_peak_bounded_delta_acc_bits_per_weight: float
    absolute_count_lower_bound_exceeds_scale_headroom: bool
    rate_held_b_storage_exceeds_scale_headroom: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "q_regime_name": self.q_regime_name,
            "row_role": self.row_role,
            "eligible_weight_count": int(self.eligible_weight_count),
            "q_packed_data_bits_per_weight": float(self.q_packed_data_bits_per_weight),
            "q_packed_metadata_bits_per_weight": float(
                self.q_packed_metadata_bits_per_weight
            ),
            "q_packed_total_bits_per_weight": float(self.q_packed_total_bits_per_weight),
            "frozen_scale_fp32_bits_per_weight": float(
                self.frozen_scale_fp32_bits_per_weight
            ),
            "scale_appropriate_headroom_bits_per_weight": float(
                self.scale_appropriate_headroom_bits_per_weight
            ),
            "density_assumption": self.density_assumption,
            "absolute_count_lower_bound_step_reports": [
                step.to_dict() for step in self.absolute_count_lower_bound_step_reports
            ],
            "rate_held_b_storage_step_reports": [
                step.to_dict() for step in self.rate_held_b_storage_step_reports
            ],
            "absolute_count_lower_bound_peak_bounded_delta_acc_bits_per_weight": float(
                self.absolute_count_lower_bound_peak_bounded_delta_acc_bits_per_weight
            ),
            "rate_held_b_storage_peak_bounded_delta_acc_bits_per_weight": float(
                self.rate_held_b_storage_peak_bounded_delta_acc_bits_per_weight
            ),
            "absolute_count_lower_bound_exceeds_scale_headroom": bool(
                self.absolute_count_lower_bound_exceeds_scale_headroom
            ),
            "rate_held_b_storage_exceeds_scale_headroom": bool(
                self.rate_held_b_storage_exceeds_scale_headroom
            ),
        }


@dataclass(frozen=True)
class ScaleAppropriateComparisonDecision:
    terminal_label: str
    required_rows: tuple[str, ...]
    rate_held_density_assumption_explicit: bool
    required_rows_all_rate_held_exceed_scale_headroom: bool
    any_required_absolute_count_lower_bound_exceeds_scale_headroom: bool
    candidate_branch_a_trigger_earned: bool
    global_per_row_compression_closed: bool
    branch_a_trigger: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "terminal_label": self.terminal_label,
            "required_rows": list(self.required_rows),
            "rate_held_density_assumption_explicit": bool(
                self.rate_held_density_assumption_explicit
            ),
            "required_rows_all_rate_held_exceed_scale_headroom": bool(
                self.required_rows_all_rate_held_exceed_scale_headroom
            ),
            "any_required_absolute_count_lower_bound_exceeds_scale_headroom": bool(
                self.any_required_absolute_count_lower_bound_exceeds_scale_headroom
            ),
            "candidate_branch_a_trigger_earned": bool(
                self.candidate_branch_a_trigger_earned
            ),
            "global_per_row_compression_closed": bool(
                self.global_per_row_compression_closed
            ),
            "branch_a_trigger": bool(self.branch_a_trigger),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ScaleAppropriateBStorageComparisonReport:
    schema_version: str
    label: str
    source_bindingness: Any
    field_coverage: SourceFieldCoverage
    candidate_name: str
    source_lower_bound_label: str
    source_terminal_label: str
    source_minimal_surface_faithful_k_label: str
    source_minimal_surface_faithful_k_value: int
    source_tiny_eligible_weight_count: int
    density_rounding_policy: str
    required_q_ledger_rows: tuple[str, ...]
    sensitivity_q_ledger_rows: tuple[str, ...]
    row_comparisons: tuple[ScaleAppropriateLedgerComparisonReport, ...]
    terminal_decision: ScaleAppropriateComparisonDecision
    raw_arrays_included: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "source_bindingness": self.source_bindingness.to_dict(),
            "field_coverage": self.field_coverage.to_dict(),
            "candidate_name": self.candidate_name,
            "source_lower_bound_label": self.source_lower_bound_label,
            "source_terminal_label": self.source_terminal_label,
            "source_minimal_surface_faithful_k_label": self.source_minimal_surface_faithful_k_label,
            "source_minimal_surface_faithful_k_value": int(
                self.source_minimal_surface_faithful_k_value
            ),
            "source_tiny_eligible_weight_count": int(self.source_tiny_eligible_weight_count),
            "density_rounding_policy": self.density_rounding_policy,
            "required_q_ledger_rows": list(self.required_q_ledger_rows),
            "sensitivity_q_ledger_rows": list(self.sensitivity_q_ledger_rows),
            "row_comparisons": [row.to_dict() for row in self.row_comparisons],
            "terminal_decision": self.terminal_decision.to_dict(),
            "raw_arrays_included": bool(self.raw_arrays_included),
            "non_claims": list(self.non_claims),
        }


@dataclass(frozen=True)
class DecisionStatisticSchemaReport:
    bucket_key_dimensions: tuple[str, ...]
    bucket_cardinality_bound: int
    observed_bucket_count: int
    bucket_key_bit_width: int
    accepted_count_bit_width: int
    deferred_count_bit_width: int
    cutoff_bit_width: int
    seed_bits: int
    metadata_bits: int
    total_bits: int
    strictest_required_q_regime_name: str
    strictest_required_eligible_weight_count: int
    strictest_required_headroom_bits_per_weight: float
    total_bits_per_weight_strictest_required_row: float
    fits_strictest_required_headroom: bool
    inclusive_sub2_if_installed: bool
    statistic_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket_key_dimensions": list(self.bucket_key_dimensions),
            "bucket_cardinality_bound": int(self.bucket_cardinality_bound),
            "observed_bucket_count": int(self.observed_bucket_count),
            "bucket_key_bit_width": int(self.bucket_key_bit_width),
            "accepted_count_bit_width": int(self.accepted_count_bit_width),
            "deferred_count_bit_width": int(self.deferred_count_bit_width),
            "cutoff_bit_width": int(self.cutoff_bit_width),
            "seed_bits": int(self.seed_bits),
            "metadata_bits": int(self.metadata_bits),
            "total_bits": int(self.total_bits),
            "strictest_required_q_regime_name": self.strictest_required_q_regime_name,
            "strictest_required_eligible_weight_count": int(
                self.strictest_required_eligible_weight_count
            ),
            "strictest_required_headroom_bits_per_weight": float(
                self.strictest_required_headroom_bits_per_weight
            ),
            "total_bits_per_weight_strictest_required_row": float(
                self.total_bits_per_weight_strictest_required_row
            ),
            "fits_strictest_required_headroom": bool(
                self.fits_strictest_required_headroom
            ),
            "inclusive_sub2_if_installed": bool(self.inclusive_sub2_if_installed),
            "statistic_mode": self.statistic_mode,
        }


@dataclass(frozen=True)
class DecisionStatisticBucketSummary:
    state_key: str
    current_q_level: int
    move_direction: int
    accepted_count: int
    deferred_count: int
    candidate_row_count: int
    decisive_bucket: bool
    frontier_tie_crosses_boundary: bool

    def statistic_input_dict(self) -> dict[str, Any]:
        return {
            "state_key": self.state_key,
            "current_q_level": int(self.current_q_level),
            "move_direction": int(self.move_direction),
            "accepted_count": int(self.accepted_count),
            "deferred_count": int(self.deferred_count),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.statistic_input_dict()
        payload.update(
            {
                "candidate_row_count": int(self.candidate_row_count),
                "decisive_bucket": bool(self.decisive_bucket),
                "frontier_tie_crosses_boundary": bool(
                    self.frontier_tie_crosses_boundary
                ),
            }
        )
        return payload


@dataclass(frozen=True)
class DecisionStatisticStepReport:
    schedule_name: str
    step: int
    global_cap: int
    candidate_row_count: int
    accepted_row_count: int
    deferred_row_count: int
    candidate_rows_fully_transient_observable: bool
    bucket_summaries: tuple[DecisionStatisticBucketSummary, ...]
    statistic_schema: DecisionStatisticSchemaReport
    shuffle_falsifier: str
    frontier_tie_bucket_count: int
    canonical_matches_exact: bool
    shuffled_matches_exact: bool
    shuffle_preserves_outcome: bool
    observable_rank_features_sufficient: bool
    insufficiency_reason: str | None
    exact_accepted_identities_sha256: str
    canonical_accepted_identities_sha256: str
    shuffled_accepted_identities_sha256: str
    exact_deferred_identities_sha256: str
    canonical_deferred_identities_sha256: str
    shuffled_deferred_identities_sha256: str
    exact_q_changed_identities_sha256: str
    canonical_q_changed_identities_sha256: str
    shuffled_q_changed_identities_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_name": self.schedule_name,
            "step": int(self.step),
            "global_cap": int(self.global_cap),
            "candidate_row_count": int(self.candidate_row_count),
            "accepted_row_count": int(self.accepted_row_count),
            "deferred_row_count": int(self.deferred_row_count),
            "candidate_rows_fully_transient_observable": bool(
                self.candidate_rows_fully_transient_observable
            ),
            "bucket_summaries": [bucket.to_dict() for bucket in self.bucket_summaries],
            "statistic_schema": self.statistic_schema.to_dict(),
            "shuffle_falsifier": self.shuffle_falsifier,
            "frontier_tie_bucket_count": int(self.frontier_tie_bucket_count),
            "canonical_matches_exact": bool(self.canonical_matches_exact),
            "shuffled_matches_exact": bool(self.shuffled_matches_exact),
            "shuffle_preserves_outcome": bool(self.shuffle_preserves_outcome),
            "observable_rank_features_sufficient": bool(
                self.observable_rank_features_sufficient
            ),
            "insufficiency_reason": self.insufficiency_reason,
            "exact_accepted_identities_sha256": self.exact_accepted_identities_sha256,
            "canonical_accepted_identities_sha256": self.canonical_accepted_identities_sha256,
            "shuffled_accepted_identities_sha256": self.shuffled_accepted_identities_sha256,
            "exact_deferred_identities_sha256": self.exact_deferred_identities_sha256,
            "canonical_deferred_identities_sha256": self.canonical_deferred_identities_sha256,
            "shuffled_deferred_identities_sha256": self.shuffled_deferred_identities_sha256,
            "exact_q_changed_identities_sha256": self.exact_q_changed_identities_sha256,
            "canonical_q_changed_identities_sha256": self.canonical_q_changed_identities_sha256,
            "shuffled_q_changed_identities_sha256": self.shuffled_q_changed_identities_sha256,
        }


@dataclass(frozen=True)
class DecisionStatisticUpperBoundDecision:
    terminal_label: str
    strictest_required_q_regime_name: str
    strictest_required_headroom_bits_per_weight: float
    peak_statistic_bits_per_weight: float
    peak_statistic_step: str
    budget_fits_strictest_required_headroom: bool
    inclusive_sub2_if_installed: bool
    first_budget_failure_step: str | None
    first_insufficient_step: str | None
    any_step_frontier_tie_crosses_boundary: bool
    all_steps_shuffle_preserve_outcome: bool
    global_per_row_compression_closed: bool
    branch_a_trigger: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "terminal_label": self.terminal_label,
            "strictest_required_q_regime_name": self.strictest_required_q_regime_name,
            "strictest_required_headroom_bits_per_weight": float(
                self.strictest_required_headroom_bits_per_weight
            ),
            "peak_statistic_bits_per_weight": float(self.peak_statistic_bits_per_weight),
            "peak_statistic_step": self.peak_statistic_step,
            "budget_fits_strictest_required_headroom": bool(
                self.budget_fits_strictest_required_headroom
            ),
            "inclusive_sub2_if_installed": bool(self.inclusive_sub2_if_installed),
            "first_budget_failure_step": self.first_budget_failure_step,
            "first_insufficient_step": self.first_insufficient_step,
            "any_step_frontier_tie_crosses_boundary": bool(
                self.any_step_frontier_tie_crosses_boundary
            ),
            "all_steps_shuffle_preserve_outcome": bool(
                self.all_steps_shuffle_preserve_outcome
            ),
            "global_per_row_compression_closed": bool(
                self.global_per_row_compression_closed
            ),
            "branch_a_trigger": bool(self.branch_a_trigger),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DecisionStatisticUpperBoundReport:
    schema_version: str
    label: str
    source_bindingness: Any
    field_coverage: SourceFieldCoverage
    candidate_name: str
    source_scale_comparison_label: str
    source_scale_terminal_label: str
    strictest_required_q_regime_name: str
    strictest_required_eligible_weight_count: int
    strictest_required_headroom_bits_per_weight: float
    bucket_key_dimensions: tuple[str, ...]
    statistic_mode: str
    shuffle_falsifier: str
    step_reports: tuple[DecisionStatisticStepReport, ...]
    terminal_decision: DecisionStatisticUpperBoundDecision
    raw_arrays_included: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "source_bindingness": self.source_bindingness.to_dict(),
            "field_coverage": self.field_coverage.to_dict(),
            "candidate_name": self.candidate_name,
            "source_scale_comparison_label": self.source_scale_comparison_label,
            "source_scale_terminal_label": self.source_scale_terminal_label,
            "strictest_required_q_regime_name": self.strictest_required_q_regime_name,
            "strictest_required_eligible_weight_count": int(
                self.strictest_required_eligible_weight_count
            ),
            "strictest_required_headroom_bits_per_weight": float(
                self.strictest_required_headroom_bits_per_weight
            ),
            "bucket_key_dimensions": list(self.bucket_key_dimensions),
            "statistic_mode": self.statistic_mode,
            "shuffle_falsifier": self.shuffle_falsifier,
            "step_reports": [step.to_dict() for step in self.step_reports],
            "terminal_decision": self.terminal_decision.to_dict(),
            "raw_arrays_included": bool(self.raw_arrays_included),
            "non_claims": list(self.non_claims),
        }


@dataclass(frozen=True)
class TieFrontierObservedBucketReport:
    schedule_name: str
    step: int
    state_key: str
    current_q_level: int
    move_direction: int
    candidate_row_count: int
    accepted_row_count: int
    boundary_abs_new_acc: int
    tie_group_size: int
    exact_accepted_within_tie_count: int
    tie_group_density_per_eligible_weight: float
    accepted_within_tie_density_per_eligible_weight: float
    theoretical_lower_bound_bits: int
    mask_bits: int
    selected_offset_bits: int
    decisive_practical_encoding_label: str
    decisive_practical_bits: int
    plateau_covers_entire_bucket: bool
    exact_tie_members_sha256: str
    exact_tie_accepted_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_name": self.schedule_name,
            "step": int(self.step),
            "state_key": self.state_key,
            "current_q_level": int(self.current_q_level),
            "move_direction": int(self.move_direction),
            "candidate_row_count": int(self.candidate_row_count),
            "accepted_row_count": int(self.accepted_row_count),
            "boundary_abs_new_acc": int(self.boundary_abs_new_acc),
            "tie_group_size": int(self.tie_group_size),
            "exact_accepted_within_tie_count": int(
                self.exact_accepted_within_tie_count
            ),
            "tie_group_density_per_eligible_weight": float(
                self.tie_group_density_per_eligible_weight
            ),
            "accepted_within_tie_density_per_eligible_weight": float(
                self.accepted_within_tie_density_per_eligible_weight
            ),
            "theoretical_lower_bound_bits": int(self.theoretical_lower_bound_bits),
            "mask_bits": int(self.mask_bits),
            "selected_offset_bits": int(self.selected_offset_bits),
            "decisive_practical_encoding_label": self.decisive_practical_encoding_label,
            "decisive_practical_bits": int(self.decisive_practical_bits),
            "plateau_covers_entire_bucket": bool(self.plateau_covers_entire_bucket),
            "exact_tie_members_sha256": self.exact_tie_members_sha256,
            "exact_tie_accepted_sha256": self.exact_tie_accepted_sha256,
        }


@dataclass(frozen=True)
class TieReservationProjectionBucketReport:
    schedule_name: str
    step: int
    state_key: str
    current_q_level: int
    move_direction: int
    source_tie_group_size: int
    source_exact_accepted_within_tie_count: int
    source_tie_group_density_per_eligible_weight: float
    source_accepted_within_tie_density_per_eligible_weight: float
    target_tie_group_size: int
    target_exact_accepted_within_tie_count: int
    tie_group_density_per_eligible_weight: float
    accepted_within_tie_density_per_eligible_weight: float
    theoretical_lower_bound_bits: int
    mask_bits: int
    selected_offset_bits: int
    decisive_practical_encoding_label: str
    decisive_practical_bits: int
    joint_ta_scaling_model: str
    scaling_model_defensible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_name": self.schedule_name,
            "step": int(self.step),
            "state_key": self.state_key,
            "current_q_level": int(self.current_q_level),
            "move_direction": int(self.move_direction),
            "source_tie_group_size": int(self.source_tie_group_size),
            "source_exact_accepted_within_tie_count": int(
                self.source_exact_accepted_within_tie_count
            ),
            "source_tie_group_density_per_eligible_weight": float(
                self.source_tie_group_density_per_eligible_weight
            ),
            "source_accepted_within_tie_density_per_eligible_weight": float(
                self.source_accepted_within_tie_density_per_eligible_weight
            ),
            "target_tie_group_size": int(self.target_tie_group_size),
            "target_exact_accepted_within_tie_count": int(
                self.target_exact_accepted_within_tie_count
            ),
            "tie_group_density_per_eligible_weight": float(
                self.tie_group_density_per_eligible_weight
            ),
            "accepted_within_tie_density_per_eligible_weight": float(
                self.accepted_within_tie_density_per_eligible_weight
            ),
            "theoretical_lower_bound_bits": int(self.theoretical_lower_bound_bits),
            "mask_bits": int(self.mask_bits),
            "selected_offset_bits": int(self.selected_offset_bits),
            "decisive_practical_encoding_label": self.decisive_practical_encoding_label,
            "decisive_practical_bits": int(self.decisive_practical_bits),
            "joint_ta_scaling_model": self.joint_ta_scaling_model,
            "scaling_model_defensible": bool(self.scaling_model_defensible),
        }


@dataclass(frozen=True)
class TieReservationStepProjectionReport:
    schedule_name: str
    step: int
    projection_label: str
    target_q_regime_name: str
    source_eligible_weight_count: int
    target_eligible_weight_count: int
    source_candidate_row_count: int
    source_accepted_row_count: int
    source_deferred_row_count: int
    target_candidate_row_count: int
    target_accepted_row_count: int
    target_deferred_row_count: int
    decision_statistic_total_bits: int
    decision_statistic_bits_per_weight: float
    bucket_reports: tuple[TieReservationProjectionBucketReport, ...]
    theoretical_lower_bound_total_bits: int
    theoretical_lower_bound_bits_per_weight: float
    mask_total_bits: int
    mask_bits_per_weight: float
    selected_offset_total_bits: int
    selected_offset_bits_per_weight: float
    decisive_practical_encoding_label: str
    decisive_tie_reservation_total_bits: int
    decisive_tie_reservation_bits_per_weight: float
    combined_decisive_bits_per_weight: float
    strictest_headroom_bits_per_weight: float
    fits_strictest_headroom: bool
    diagnostic_only: bool
    joint_ta_scaling_model: str
    scaling_model_defensible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_name": self.schedule_name,
            "step": int(self.step),
            "projection_label": self.projection_label,
            "target_q_regime_name": self.target_q_regime_name,
            "source_eligible_weight_count": int(self.source_eligible_weight_count),
            "target_eligible_weight_count": int(self.target_eligible_weight_count),
            "source_candidate_row_count": int(self.source_candidate_row_count),
            "source_accepted_row_count": int(self.source_accepted_row_count),
            "source_deferred_row_count": int(self.source_deferred_row_count),
            "target_candidate_row_count": int(self.target_candidate_row_count),
            "target_accepted_row_count": int(self.target_accepted_row_count),
            "target_deferred_row_count": int(self.target_deferred_row_count),
            "decision_statistic_total_bits": int(self.decision_statistic_total_bits),
            "decision_statistic_bits_per_weight": float(
                self.decision_statistic_bits_per_weight
            ),
            "bucket_reports": [bucket.to_dict() for bucket in self.bucket_reports],
            "theoretical_lower_bound_total_bits": int(
                self.theoretical_lower_bound_total_bits
            ),
            "theoretical_lower_bound_bits_per_weight": float(
                self.theoretical_lower_bound_bits_per_weight
            ),
            "mask_total_bits": int(self.mask_total_bits),
            "mask_bits_per_weight": float(self.mask_bits_per_weight),
            "selected_offset_total_bits": int(self.selected_offset_total_bits),
            "selected_offset_bits_per_weight": float(self.selected_offset_bits_per_weight),
            "decisive_practical_encoding_label": self.decisive_practical_encoding_label,
            "decisive_tie_reservation_total_bits": int(
                self.decisive_tie_reservation_total_bits
            ),
            "decisive_tie_reservation_bits_per_weight": float(
                self.decisive_tie_reservation_bits_per_weight
            ),
            "combined_decisive_bits_per_weight": float(
                self.combined_decisive_bits_per_weight
            ),
            "strictest_headroom_bits_per_weight": float(
                self.strictest_headroom_bits_per_weight
            ),
            "fits_strictest_headroom": bool(self.fits_strictest_headroom),
            "diagnostic_only": bool(self.diagnostic_only),
            "joint_ta_scaling_model": self.joint_ta_scaling_model,
            "scaling_model_defensible": bool(self.scaling_model_defensible),
        }


@dataclass(frozen=True)
class TieReservationRowComparisonReport:
    q_regime_name: str
    row_role: str
    eligible_weight_count: int
    row_headroom_bits_per_weight: float
    strictest_headroom_bits_per_weight: float
    observed_tie_density_assumption: str
    joint_ta_scaling_model: str
    joint_ta_scaling_model_defensible: bool
    absolute_count_step_reports: tuple[TieReservationStepProjectionReport, ...]
    rate_held_step_reports: tuple[TieReservationStepProjectionReport, ...]
    absolute_count_peak_combined_bits_per_weight: float
    rate_held_peak_combined_bits_per_weight: float
    rate_held_fits_strictest_headroom: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "q_regime_name": self.q_regime_name,
            "row_role": self.row_role,
            "eligible_weight_count": int(self.eligible_weight_count),
            "row_headroom_bits_per_weight": float(self.row_headroom_bits_per_weight),
            "strictest_headroom_bits_per_weight": float(
                self.strictest_headroom_bits_per_weight
            ),
            "observed_tie_density_assumption": self.observed_tie_density_assumption,
            "joint_ta_scaling_model": self.joint_ta_scaling_model,
            "joint_ta_scaling_model_defensible": bool(
                self.joint_ta_scaling_model_defensible
            ),
            "absolute_count_step_reports": [
                step.to_dict() for step in self.absolute_count_step_reports
            ],
            "rate_held_step_reports": [
                step.to_dict() for step in self.rate_held_step_reports
            ],
            "absolute_count_peak_combined_bits_per_weight": float(
                self.absolute_count_peak_combined_bits_per_weight
            ),
            "rate_held_peak_combined_bits_per_weight": float(
                self.rate_held_peak_combined_bits_per_weight
            ),
            "rate_held_fits_strictest_headroom": bool(
                self.rate_held_fits_strictest_headroom
            ),
        }


@dataclass(frozen=True)
class TieFrontierReservationDecision:
    terminal_label: str
    required_rows: tuple[str, ...]
    strictest_required_q_regime_name: str
    strictest_headroom_bits_per_weight: float
    joint_ta_scaling_model: str
    joint_ta_scaling_model_defensible: bool
    peak_rate_held_combined_bits_per_weight: float
    peak_rate_held_step: str
    peak_rate_held_q_regime_name: str
    peak_rate_held_encoding_label: str
    theoretical_lower_bound_non_decisive: bool
    required_rows_all_rate_held_fit_strictest_headroom: bool
    any_required_row_joint_ta_ambiguous: bool
    candidate_hybrid_alive: bool
    global_per_row_compression_closed: bool
    branch_a_trigger: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "terminal_label": self.terminal_label,
            "required_rows": list(self.required_rows),
            "strictest_required_q_regime_name": self.strictest_required_q_regime_name,
            "strictest_headroom_bits_per_weight": float(
                self.strictest_headroom_bits_per_weight
            ),
            "joint_ta_scaling_model": self.joint_ta_scaling_model,
            "joint_ta_scaling_model_defensible": bool(
                self.joint_ta_scaling_model_defensible
            ),
            "peak_rate_held_combined_bits_per_weight": float(
                self.peak_rate_held_combined_bits_per_weight
            ),
            "peak_rate_held_step": self.peak_rate_held_step,
            "peak_rate_held_q_regime_name": self.peak_rate_held_q_regime_name,
            "peak_rate_held_encoding_label": self.peak_rate_held_encoding_label,
            "theoretical_lower_bound_non_decisive": bool(
                self.theoretical_lower_bound_non_decisive
            ),
            "required_rows_all_rate_held_fit_strictest_headroom": bool(
                self.required_rows_all_rate_held_fit_strictest_headroom
            ),
            "any_required_row_joint_ta_ambiguous": bool(
                self.any_required_row_joint_ta_ambiguous
            ),
            "candidate_hybrid_alive": bool(self.candidate_hybrid_alive),
            "global_per_row_compression_closed": bool(
                self.global_per_row_compression_closed
            ),
            "branch_a_trigger": bool(self.branch_a_trigger),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TieFrontierReservationLowerBoundReport:
    schema_version: str
    label: str
    source_bindingness: Any
    field_coverage: SourceFieldCoverage
    candidate_name: str
    source_decision_statistic_label: str
    source_decision_statistic_terminal_label: str
    strictest_required_q_regime_name: str
    strictest_headroom_bits_per_weight: float
    source_eligible_weight_count: int
    required_q_ledger_rows: tuple[str, ...]
    sensitivity_q_ledger_rows: tuple[str, ...]
    observed_failing_bucket_reports: tuple[TieFrontierObservedBucketReport, ...]
    row_comparisons: tuple[TieReservationRowComparisonReport, ...]
    terminal_decision: TieFrontierReservationDecision
    raw_arrays_included: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "source_bindingness": self.source_bindingness.to_dict(),
            "field_coverage": self.field_coverage.to_dict(),
            "candidate_name": self.candidate_name,
            "source_decision_statistic_label": self.source_decision_statistic_label,
            "source_decision_statistic_terminal_label": self.source_decision_statistic_terminal_label,
            "strictest_required_q_regime_name": self.strictest_required_q_regime_name,
            "strictest_headroom_bits_per_weight": float(
                self.strictest_headroom_bits_per_weight
            ),
            "source_eligible_weight_count": int(self.source_eligible_weight_count),
            "required_q_ledger_rows": list(self.required_q_ledger_rows),
            "sensitivity_q_ledger_rows": list(self.sensitivity_q_ledger_rows),
            "observed_failing_bucket_reports": [
                bucket.to_dict() for bucket in self.observed_failing_bucket_reports
            ],
            "row_comparisons": [row.to_dict() for row in self.row_comparisons],
            "terminal_decision": self.terminal_decision.to_dict(),
            "raw_arrays_included": bool(self.raw_arrays_included),
            "non_claims": list(self.non_claims),
        }


@dataclass(frozen=True)
class _ExactScheduleTraceStep:
    schedule_step: VotePressureStepSpec
    inputs: tuple[BoundedDeltaOracleInput, ...]
    tensor_offsets: dict[str, int]
    cap_spec: GlobalRateCapSpec
    exact_input_states: dict[str, VoteUpdateState]
    exact_input_backlog: dict[str, dict[int, dict[str, int]]]
    exact_path: Any
    exact_output_backlog: dict[str, dict[int, dict[str, int]]]


@dataclass(frozen=True)
class _RealBacklogTraceStep:
    schedule_name: str
    step: int
    vote_pattern_step: int
    inputs: tuple[BoundedDeltaOracleInput, ...]
    tensor_offsets: dict[str, int]
    cap_spec: GlobalRateCapSpec
    exact_input_states: dict[str, VoteUpdateState]
    exact_input_backlog: dict[str, dict[int, dict[str, int]]]
    exact_path: Any
    exact_output_backlog: dict[str, dict[int, dict[str, int]]]


def _step_report_from_reference(
    *,
    schedule_name: str,
    step: int,
    mode: str,
    curve_label: str,
    hot_budget_label: str,
    requested_hot_budget: int,
    is_max_hot_budget_point: bool,
    max_hot_budget_for_policy: int,
    actual_hot_exact_row_count: int,
    backlog_policy_k: int | None,
    report: BoundedDeltaReferenceReport,
    bounded_reinitialized_from_exact: bool,
) -> RepresentativeStepReport:
    parity = report.measured_report.oracle_parity
    return RepresentativeStepReport(
        schedule_name=schedule_name,
        step=int(step),
        mode=mode,
        curve_label=curve_label,
        hot_budget_label=hot_budget_label,
        requested_hot_budget=int(requested_hot_budget),
        is_max_hot_budget_point=bool(is_max_hot_budget_point),
        max_hot_budget_for_policy=int(max_hot_budget_for_policy),
        actual_hot_exact_row_count=int(actual_hot_exact_row_count),
        backlog_policy_k=backlog_policy_k,
        exact_backlog_entry_count=int(parity.get("exact_output_deferred_backlog_count", 0)),
        bounded_stored_backlog_entry_count=int(
            parity.get("bounded_stored_deferred_backlog_count", 0)
        ),
        bounded_reinitialized_from_exact=bool(bounded_reinitialized_from_exact),
        classification=report.classification,
        guard_passed=bool(report.guard_passed),
        failed_metrics=tuple(report.failed_metrics),
        bounded_delta_report=report,
    )


def _copy_state_map(
    states: Mapping[str, VoteUpdateState],
) -> dict[str, VoteUpdateState]:
    return {
        state_key: VoteUpdateState(
            q_levels=state.q_levels.detach().clone().contiguous(),
            accumulators=state.accumulators.detach().clone().contiguous(),
        )
        for state_key, state in states.items()
    }


def _zero_accumulator_state_map(
    states: Mapping[str, VoteUpdateState],
) -> dict[str, VoteUpdateState]:
    return {
        state_key: VoteUpdateState(
            q_levels=state.q_levels.detach().clone().contiguous(),
            accumulators=torch.zeros_like(state.accumulators, dtype=torch.int16),
        )
        for state_key, state in states.items()
    }


def _ids_by_state(
    identities: Sequence[tuple[str, int]],
) -> dict[str, tuple[int, ...]]:
    out: dict[str, list[int]] = {key: [] for key in PRIMARY_STATE_KEYS}
    for state_key, flat_index in sorted(identities):
        out.setdefault(str(state_key), []).append(int(flat_index))
    return {state_key: tuple(indices) for state_key, indices in out.items()}


def _candidate_hot_identities(
    *,
    candidate_name: str,
    exact_path: Any,
    carried_backlog: Mapping[str, Mapping[int, Mapping[str, int]]],
) -> set[tuple[str, int]]:
    carried_ids = _backlog_key_set(carried_backlog)
    if candidate_name == HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE:
        return exact_path.candidate_ids | carried_ids
    if candidate_name == EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE:
        return exact_path.fired_ids | carried_ids
    if candidate_name == COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE:
        return exact_path.candidate_ids | exact_path.accepted_ids | exact_path.deferred_ids | carried_ids
    raise ValueError(f"unsupported candidate_name {candidate_name!r}")


def _hot_exact_candidate_inputs_and_states(
    *,
    inputs: Sequence[BoundedDeltaOracleInput],
    source_states: Mapping[str, VoteUpdateState],
    hot_by_state: Mapping[str, Sequence[int]],
) -> tuple[tuple[BoundedDeltaOracleInput, ...], dict[str, VoteUpdateState]]:
    candidate_inputs: list[BoundedDeltaOracleInput] = []
    bounded_states: dict[str, VoteUpdateState] = {}
    for item in inputs:
        state = source_states[item.state_key]
        hot = tuple(int(idx) for idx in hot_by_state.get(item.state_key, ()))
        encoded = encode_budget_capped_hybrid_reference(
            state,
            hot_exact_indices=hot,
            cold_default_value=0,
        )
        bounded_states[item.state_key] = VoteUpdateState(
            q_levels=state.q_levels.detach().clone().contiguous(),
            accumulators=decode_bounded_accumulator_to_i16(encoded),
        )
        candidate_inputs.append(
            BoundedDeltaOracleInput(
                state_key=item.state_key,
                state=state,
                vote_inputs=item.vote_inputs,
                vote_spec=item.vote_spec,
                hot_exact_indices=hot,
                cold_default_value=0,
            )
        )
    return tuple(candidate_inputs), bounded_states


def _coarse_block_charge(block: torch.Tensor, *, threshold_abs: int) -> int:
    mean_value = float(block.detach().to(torch.float32).mean().item())
    charge = max(1, int(threshold_abs) - 1)
    if mean_value > 0.5:
        return charge
    if mean_value < -0.5:
        return -charge
    return 0


def _coarse_signed_candidate_inputs_and_states(
    *,
    inputs: Sequence[BoundedDeltaOracleInput],
    source_states: Mapping[str, VoteUpdateState],
    hot_by_state: Mapping[str, Sequence[int]],
) -> tuple[tuple[BoundedDeltaOracleInput, ...], dict[str, VoteUpdateState], float]:
    candidate_inputs: list[BoundedDeltaOracleInput] = []
    bounded_states: dict[str, VoteUpdateState] = {}
    total_dense_bits = 0
    total_eligible = 0
    for item in inputs:
        state = source_states[item.state_key]
        hot = tuple(int(idx) for idx in hot_by_state.get(item.state_key, ()))
        flat = state.accumulators.detach().cpu().to(torch.int16).flatten()
        approx = torch.zeros_like(flat)
        for block_start in range(0, int(flat.numel()), COARSE_SIGNED_CHARGE_BLOCK_SIZE):
            block = flat[block_start : block_start + COARSE_SIGNED_CHARGE_BLOCK_SIZE]
            block_charge = _coarse_block_charge(
                block,
                threshold_abs=int(item.vote_spec.threshold_abs),
            )
            approx[block_start : block_start + int(block.numel())] = int(block_charge)
        for index in hot:
            approx[int(index)] = flat[int(index)]
        bounded_states[item.state_key] = VoteUpdateState(
            q_levels=state.q_levels.detach().clone().contiguous(),
            accumulators=approx.view_as(state.accumulators).contiguous(),
        )
        candidate_inputs.append(
            BoundedDeltaOracleInput(
                state_key=item.state_key,
                state=state,
                vote_inputs=item.vote_inputs,
                vote_spec=item.vote_spec,
                hot_exact_indices=hot,
                cold_default_value=0,
            )
        )
        total_dense_bits += int(math.ceil(float(flat.numel()) / COARSE_SIGNED_CHARGE_BLOCK_SIZE) * 2)
        total_eligible += int(flat.numel())
    dense_bpw = float(total_dense_bits) / float(total_eligible) if total_eligible else 0.0
    return tuple(candidate_inputs), bounded_states, dense_bpw


def _candidate_inputs_and_states(
    *,
    candidate_name: str,
    inputs: Sequence[BoundedDeltaOracleInput],
    source_states: Mapping[str, VoteUpdateState],
    carried_backlog: Mapping[str, Mapping[int, Mapping[str, int]]],
    exact_path: Any,
) -> tuple[tuple[BoundedDeltaOracleInput, ...], dict[str, VoteUpdateState], int, float]:
    hot_by_state = _ids_by_state(
        _candidate_hot_identities(
            candidate_name=candidate_name,
            exact_path=exact_path,
            carried_backlog=carried_backlog,
        )
    )
    if candidate_name == COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE:
        candidate_inputs, bounded_states, dense_bpw = _coarse_signed_candidate_inputs_and_states(
            inputs=inputs,
            source_states=source_states,
            hot_by_state=hot_by_state,
        )
        return candidate_inputs, bounded_states, 0, dense_bpw
    candidate_inputs, bounded_states = _hot_exact_candidate_inputs_and_states(
        inputs=inputs,
        source_states=source_states,
        hot_by_state=hot_by_state,
    )
    event_delta_count = (
        len(exact_path.fired_ids)
        if candidate_name == EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE
        else 0
    )
    return candidate_inputs, bounded_states, event_delta_count, 0.0


def _continuity_surface_counts(measured: BoundedDeltaMeasuredReport) -> dict[str, int]:
    return {
        "accepted_rows": int(measured.accepted_changed_count),
        "deferred_rows": int(measured.deferred_changed_count),
        "final_q_changes": int(measured.q_changed_count),
        "backlog_carry": int(measured.backlog_key_changed_count),
    }


def _compare_against_null(
    *,
    candidate_report: BoundedDeltaReferenceReport,
    null_report: BoundedDeltaReferenceReport,
) -> AdmissionNullBaselineComparison:
    candidate_counts = _continuity_surface_counts(candidate_report.measured_report)
    null_counts = _continuity_surface_counts(null_report.measured_report)
    compared_surfaces = tuple(candidate_counts)
    strict_improvement = tuple(
        name for name in compared_surfaces if candidate_counts[name] < null_counts[name]
    )
    regressed = tuple(
        name for name in compared_surfaces if candidate_counts[name] > null_counts[name]
    )
    beats_null = not regressed and bool(strict_improvement)
    summary = (
        "beats_accumulator_free_null_on_continuity"
        if beats_null
        else "does_not_beat_accumulator_free_null_on_continuity"
    )
    return AdmissionNullBaselineComparison(
        candidate_beats_null=beats_null,
        compared_surfaces=compared_surfaces,
        strict_improvement_surfaces=strict_improvement,
        regressed_surfaces=regressed,
        candidate_surface_counts=candidate_counts,
        null_surface_counts=null_counts,
        summary=summary,
    )


def _backlog_truncation_report(
    *,
    report: BoundedDeltaReferenceReport,
    exact_input_backlog: Mapping[str, Mapping[int, Mapping[str, int]]],
    bounded_input_backlog: Mapping[str, Mapping[int, Mapping[str, int]]],
    exact_output_backlog: Mapping[str, Mapping[int, Mapping[str, int]]],
    bounded_stored_backlog: Mapping[str, Mapping[int, Mapping[str, int]]],
) -> BacklogTruncationAttribution:
    input_truncated = _backlog_key_set(exact_input_backlog) - _backlog_key_set(bounded_input_backlog)
    stored_truncated = _backlog_key_set(exact_output_backlog) - _backlog_key_set(bounded_stored_backlog)
    if input_truncated or stored_truncated:
        summary = (
            "backlog_truncation_continuity_miss"
            if int(report.measured_report.backlog_key_changed_count) > 0
            else "backlog_truncation_present_without_backlog_key_miss"
        )
    else:
        summary = "no_backlog_truncation"
    return BacklogTruncationAttribution(
        bounded_input_truncation_count=len(input_truncated),
        bounded_input_truncation_identities_sha256=_identity_sha256(input_truncated),
        bounded_stored_truncation_count=len(stored_truncated),
        bounded_stored_truncation_identities_sha256=_identity_sha256(stored_truncated),
        paired_rejection_summary=report.rejection_telemetry.summary,
        summary=summary,
    )


def _oracle_upper_bound_non_claims(*, candidate_name: str) -> tuple[str, ...]:
    return (
        f"{ORACLE_UPPER_BOUND_ADMISSION_DIAGNOSTIC} only; exact-path oracle identities guide {candidate_name}",
        "PASS is an oracle-informed upper bound, not a deployable online codec",
        "promotion means earns dyn200 consideration / next implementation design only",
        "no live codec exists from this diagnostic alone",
        "no production vote_update/global_rate_cap replacement",
        "no GPU lane",
        "no trainer/live-run/checkpoint/creditdir mutation",
        "compact counts/hashes only; no raw per-weight arrays",
    )


def _delta_over_2bpw(bits_per_weight: float) -> float:
    return float(bits_per_weight) - 2.0


def _candidate_path_difference(
    *,
    candidate_name: str,
    backlog_policy_k: int | None,
) -> str:
    if candidate_name == HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE:
        return (
            "oracle upper bound candidate path differs by exact-path-guided hot frontier/backlog "
            "selection carried through the bounded accumulator state"
        )
    if candidate_name == EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE:
        return (
            "oracle upper bound candidate path differs by fired-row event retention plus "
            f"bounded backlog truncate_k{int(backlog_policy_k or 0)} carry-forward"
        )
    return (
        "oracle upper bound candidate path differs by coarse signed cold-field approximation plus "
        f"sparse exact frontier overrides and bounded backlog truncate_k{int(backlog_policy_k or 0)} carry-forward"
    )


def _build_exact_schedule_trace() -> tuple[tuple[_ExactScheduleTraceStep, ...], int]:
    exact_states = _initial_states()
    exact_backlog: dict[str, dict[int, dict[str, int]]] = {}
    trace_steps: list[_ExactScheduleTraceStep] = []
    max_exact_output_backlog_count = 0
    for schedule_step in PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE:
        inputs, offsets = _make_step_inputs(exact_states, schedule_step)
        cap_spec = GlobalRateCapSpec(cap=int(schedule_step.cap), step=int(schedule_step.step))
        exact_path = _run_reference_path(
            inputs,
            states_by_key=exact_states,
            global_cap_spec=cap_spec,
            deferred_backlog=exact_backlog,
            tensor_offsets=offsets,
        )
        if exact_path.cap_result is None:
            raise ValueError("capacity localization trace requires global cap results")
        exact_output_backlog = _copy_backlog(exact_path.cap_result.deferred_backlog)
        max_exact_output_backlog_count = max(
            max_exact_output_backlog_count,
            _backlog_entry_count(exact_output_backlog),
        )
        trace_steps.append(
            _ExactScheduleTraceStep(
                schedule_step=schedule_step,
                inputs=tuple(inputs),
                tensor_offsets=dict(offsets),
                cap_spec=cap_spec,
                exact_input_states=_copy_state_map(exact_states),
                exact_input_backlog=_copy_backlog(exact_backlog),
                exact_path=exact_path,
                exact_output_backlog=exact_output_backlog,
            )
        )
        exact_states = _states_from_path(exact_path)
        exact_backlog = exact_output_backlog
    return tuple(trace_steps), int(max_exact_output_backlog_count)


def _protected_surface_destructive_approximation_present(
    report: BoundedDeltaReferenceReport,
) -> bool:
    protected = set(report.admission_contract.exact_surfaces)
    return any(
        item.surface in protected and item.status == "destructive_approximation"
        for item in report.rejection_telemetry.surfaces
    )


def _capacity_step_report(
    *,
    schedule_name: str,
    step: int,
    k_label: str,
    k_value: int | None,
    report: BoundedDeltaReferenceReport,
    exact_input_backlog: Mapping[str, Mapping[int, Mapping[str, int]]],
    bounded_input_backlog: Mapping[str, Mapping[int, Mapping[str, int]]],
    exact_output_backlog: Mapping[str, Mapping[int, Mapping[str, int]]],
    bounded_stored_backlog: Mapping[str, Mapping[int, Mapping[str, int]]],
) -> CandidateCapacityStepReport:
    truncation = _backlog_truncation_report(
        report=report,
        exact_input_backlog=exact_input_backlog,
        bounded_input_backlog=bounded_input_backlog,
        exact_output_backlog=exact_output_backlog,
        bounded_stored_backlog=bounded_stored_backlog,
    )
    bpw = float(report.ledger.packed_inclusive_physical_bits_per_weight)
    return CandidateCapacityStepReport(
        schedule_name=schedule_name,
        step=int(step),
        k_label=k_label,
        k_value=k_value,
        bounded_delta_report=report,
        backlog_truncation_attribution=truncation,
        protected_surface_destructive_approximation_present=(
            _protected_surface_destructive_approximation_present(report)
        ),
        surface_fidelity_clears=bool(report.guard_passed and report.admission_passed),
        packed_inclusive_physical_bits_per_weight=bpw,
        delta_over_2bpw=_delta_over_2bpw(bpw),
    )


def _cold_zero_counterfactual_projection(
    projection: BoundedDeltaStorageProjection,
) -> BoundedDeltaStorageProjection:
    return project_bounded_delta_accumulator_bpw(
        eligible_weight_count=int(projection.eligible_weight_count),
        hot_exact_row_count=int(projection.hot_exact_row_count),
        cold_exception_row_count=0,
        event_delta_count=int(projection.event_delta_count),
        backlog_entry_count=int(projection.backlog_entry_count),
        index_bits_per_row=int(projection.index_bits_per_row),
        hot_value_bits_per_row=int(projection.hot_value_bits_per_row),
        hot_flag_bits_per_row=int(projection.hot_flag_bits_per_row),
        cold_exception_value_bits_per_row=int(projection.cold_exception_value_bits_per_row),
        cold_exception_flag_bits_per_row=int(projection.cold_exception_flag_bits_per_row),
        event_delta_bits_per_entry=int(projection.event_delta_bits_per_entry),
        event_delta_flag_bits_per_entry=int(projection.event_delta_flag_bits_per_entry),
        backlog_age_bits_per_entry=int(projection.backlog_age_bits_per_entry),
        backlog_defer_count_bits_per_entry=int(projection.backlog_defer_count_bits_per_entry),
        tensor_metadata_bits=int(projection.tensor_metadata_bits),
        bucket_metadata_bits=int(projection.bucket_metadata_bits),
        scale_metadata_bits=int(projection.scale_metadata_bits),
        guardrail_metadata_bits=int(projection.guardrail_metadata_bits),
        dense_cold_bits_per_weight=float(projection.dense_cold_bits_per_weight),
    )


def _candidate_a_budget_readout(
    step_report: CandidateCapacityStepReport,
) -> CandidateABudgetReadout:
    projection = step_report.bounded_delta_report.storage_projection
    counterfactual = _cold_zero_counterfactual_projection(projection)
    return CandidateABudgetReadout(
        schedule_name=step_report.schedule_name,
        step=int(step_report.step),
        packed_inclusive_physical_bits_per_weight=float(
            step_report.packed_inclusive_physical_bits_per_weight
        ),
        delta_over_2bpw=float(step_report.delta_over_2bpw),
        hot_exact_bits=int(projection.hot_exact_bits),
        cold_exception_bits=int(projection.cold_exception_bits),
        backlog_bits=int(projection.backlog_bits),
        metadata_bits=int(projection.metadata_bits),
        dense_cold_bits=float(projection.dense_cold_bits),
        original_classification=step_report.bounded_delta_report.classification,
        original_rejection_summary=step_report.bounded_delta_report.rejection_telemetry.summary,
        cold_zero_counterfactual_bits_per_weight=float(
            counterfactual.bounded_delta_acc_bits_per_weight
            + step_report.bounded_delta_report.ledger.q_packed_total_bits_per_weight
            + step_report.bounded_delta_report.ledger.frozen_scale_fp32_bits_per_weight
        ),
        cold_zero_counterfactual_delta_over_2bpw=_delta_over_2bpw(
            counterfactual.bounded_delta_acc_bits_per_weight
            + step_report.bounded_delta_report.ledger.q_packed_total_bits_per_weight
            + step_report.bounded_delta_report.ledger.frozen_scale_fp32_bits_per_weight
        ),
        cold_zero_counterfactual_clears_sub2=bool(
            (
                counterfactual.bounded_delta_acc_bits_per_weight
                + step_report.bounded_delta_report.ledger.q_packed_total_bits_per_weight
                + step_report.bounded_delta_report.ledger.frozen_scale_fp32_bits_per_weight
            )
            < 2.0
        ),
    )


def _candidate_a_terminal_budget_direction(
    readouts: Sequence[CandidateABudgetReadout],
) -> str:
    if any(
        item.delta_over_2bpw > 0.0 and item.cold_zero_counterfactual_delta_over_2bpw > 0.0
        for item in readouts
    ):
        return A_FUNDAMENTALLY_OVER_LABEL
    return A_COLD_EXCEPTION_BUDGET_LEVER_LABEL


def _backlog_k_values(max_exact_output_backlog_count: int) -> tuple[int, ...]:
    values: list[int] = []
    current = 32
    target = max(32, int(max_exact_output_backlog_count))
    while current < target:
        values.append(int(current))
        current *= 2
    values.append(int(current))
    return tuple(values)


def _trace_step_schedule_name(trace_step: Any) -> str:
    if hasattr(trace_step, "schedule_name"):
        return str(trace_step.schedule_name)
    return str(trace_step.schedule_step.name)


def _trace_step_number(trace_step: Any) -> int:
    if hasattr(trace_step, "step"):
        return int(trace_step.step)
    return int(trace_step.schedule_step.step)


def _run_candidate_capacity_sweep(
    *,
    candidate_name: str,
    trace_steps: Sequence[_ExactScheduleTraceStep],
    q_ledger: Base3QEntropyLedgerRow,
    guard_spec: BoundedDeltaGuardSpec,
    backlog_policy_k: int | None,
    k_label: str,
) -> CandidateKSweepEntry:
    bounded_states = _copy_state_map(_initial_states())
    bounded_backlog: dict[str, dict[int, dict[str, int]]] = {}
    step_reports: list[CandidateCapacityStepReport] = []
    eligible = int(q_ledger.eligible_weight_count)
    for trace_step in trace_steps:
        candidate_input_backlog = _copy_backlog(bounded_backlog)
        candidate_inputs, bounded_input_states, event_delta_count, dense_cold_bpw = (
            _candidate_inputs_and_states(
                candidate_name=candidate_name,
                inputs=trace_step.inputs,
                source_states=bounded_states,
                carried_backlog=candidate_input_backlog,
                exact_path=trace_step.exact_path,
            )
        )
        bounded_path = _run_reference_path(
            candidate_inputs,
            states_by_key=bounded_input_states,
            global_cap_spec=trace_step.cap_spec,
            deferred_backlog=candidate_input_backlog,
            tensor_offsets=trace_step.tensor_offsets,
        )
        if bounded_path.cap_result is None:
            raise ValueError("candidate capacity sweep requires bounded cap results")
        if backlog_policy_k is None:
            bounded_stored_backlog = _copy_backlog(bounded_path.cap_result.deferred_backlog)
        else:
            bounded_stored_backlog = _select_stored_backlog(
                bounded_path.cap_result.deferred_backlog,
                priority_identities=trace_step.exact_path.ordered_row_ids,
                max_entries=int(backlog_policy_k),
            )
        hot_row_count = _hot_count(
            {item.state_key: item.hot_exact_indices for item in candidate_inputs}
        )
        report = compare_bounded_delta_paths_to_int16_oracle(
            inputs=candidate_inputs,
            q_ledger_row=q_ledger,
            exact_path=trace_step.exact_path,
            bounded_path=bounded_path,
            storage_projection=project_bounded_delta_accumulator_bpw(
                eligible_weight_count=eligible,
                hot_exact_row_count=hot_row_count,
                event_delta_count=event_delta_count,
                backlog_entry_count=_backlog_entry_count(bounded_stored_backlog),
                tensor_metadata_bits=len(PRIMARY_STATE_KEYS) * DEFAULT_TENSOR_METADATA_BITS_PER_INPUT,
                bucket_metadata_bits=DEFAULT_BUCKET_METADATA_BITS,
                scale_metadata_bits=DEFAULT_SCALE_METADATA_BITS,
                guardrail_metadata_bits=DEFAULT_GUARDRAIL_METADATA_BITS,
                dense_cold_bits_per_weight=dense_cold_bpw,
            ),
            guard_spec=guard_spec,
            candidate_name=candidate_name,
            global_cap_spec=trace_step.cap_spec,
            exact_input_states=trace_step.exact_input_states,
            bounded_input_states=bounded_input_states,
            exact_input_backlog=trace_step.exact_input_backlog,
            bounded_input_backlog=candidate_input_backlog,
            bounded_stored_backlog=bounded_stored_backlog,
            tensor_offsets=trace_step.tensor_offsets,
            bounded_backlog_policy_active=(
                _backlog_key_set(trace_step.exact_input_backlog)
                != _backlog_key_set(candidate_input_backlog)
                or _backlog_key_set(trace_step.exact_output_backlog)
                != _backlog_key_set(bounded_stored_backlog)
            ),
            path_difference=_candidate_path_difference(
                candidate_name=candidate_name,
                backlog_policy_k=backlog_policy_k,
            ),
            oracle_parity_overrides={
                "cumulative_carry_forward": True,
                "bounded_reinitialized_from_exact": False,
                "builder_label": ORACLE_UPPER_BOUND_ADMISSION_DIAGNOSTIC,
                "swept_backlog_k_label": k_label,
            },
            non_claims=_oracle_upper_bound_non_claims(candidate_name=candidate_name),
        )
        step_reports.append(
            _capacity_step_report(
                schedule_name=_trace_step_schedule_name(trace_step),
                step=_trace_step_number(trace_step),
                k_label=k_label,
                k_value=backlog_policy_k,
                report=report,
                exact_input_backlog=trace_step.exact_input_backlog,
                bounded_input_backlog=candidate_input_backlog,
                exact_output_backlog=trace_step.exact_output_backlog,
                bounded_stored_backlog=bounded_stored_backlog,
            )
        )
        bounded_states = _states_from_path(bounded_path)
        bounded_backlog = _copy_backlog(bounded_stored_backlog)
    terminal = step_reports[-1]
    return CandidateKSweepEntry(
        candidate_name=candidate_name,
        k_label=k_label,
        k_value=backlog_policy_k,
        per_step_reports=tuple(step_reports),
        all_steps_surface_fidelity_clears=all(
            step.surface_fidelity_clears for step in step_reports
        ),
        all_steps_claimable_physical_sub2_with_guardrail=all(
            step.bounded_delta_report.claimable_physical_sub2_with_guardrail
            for step in step_reports
        ),
        terminal_surface_fidelity_clears=bool(terminal.surface_fidelity_clears),
        terminal_claimable_physical_sub2_with_guardrail=bool(
            terminal.bounded_delta_report.claimable_physical_sub2_with_guardrail
        ),
        terminal_protected_surface_destructive_approximation_present=bool(
            terminal.protected_surface_destructive_approximation_present
        ),
        terminal_packed_inclusive_physical_bits_per_weight=float(
            terminal.packed_inclusive_physical_bits_per_weight
        ),
        terminal_delta_over_2bpw=float(terminal.delta_over_2bpw),
        terminal_rejection_summary=terminal.bounded_delta_report.rejection_telemetry.summary,
    )


def _candidate_k_sweep_decision(
    *,
    candidate_name: str,
    sweep_entries: Sequence[CandidateKSweepEntry],
) -> CandidateKSweepDecision:
    viable = next(
        (
            entry
            for entry in sweep_entries
            if entry.all_steps_surface_fidelity_clears
            and entry.all_steps_claimable_physical_sub2_with_guardrail
        ),
        None,
    )
    if viable is not None:
        return CandidateKSweepDecision(
            candidate_name=candidate_name,
            status=K_SWEEP_MINIMAL_VIABLE_PASS,
            decisive_k_label=viable.k_label,
            decisive_k_value=viable.k_value,
            oracle_upper_bound_only=True,
        )
    surface_clear = next(
        (entry for entry in sweep_entries if entry.all_steps_surface_fidelity_clears),
        None,
    )
    if surface_clear is not None:
        return CandidateKSweepDecision(
            candidate_name=candidate_name,
            status=K_SWEEP_JOINT_INFEASIBLE,
            decisive_k_label=surface_clear.k_label,
            decisive_k_value=surface_clear.k_value,
            oracle_upper_bound_only=True,
        )
    unbounded = sweep_entries[-1]
    return CandidateKSweepDecision(
        candidate_name=candidate_name,
        status=K_SWEEP_REPRESENTATION_WALL,
        decisive_k_label=unbounded.k_label,
        decisive_k_value=unbounded.k_value,
        oracle_upper_bound_only=True,
    )


def _run_one_step_strict_control(
    q_ledger: Base3QEntropyLedgerRow,
    guard_spec: BoundedDeltaGuardSpec,
) -> tuple[RepresentativeStepReport, ...]:
    states = _initial_states()
    backlog: dict[str, dict[int, dict[str, int]]] = {}
    reports: list[RepresentativeStepReport] = []
    for schedule_step in PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE:
        inputs, offsets = _make_step_inputs(states, schedule_step)
        cap_spec = GlobalRateCapSpec(cap=int(schedule_step.cap), step=int(schedule_step.step))
        exact_path = _run_reference_path(
            inputs,
            states_by_key=states,
            global_cap_spec=cap_spec,
            deferred_backlog=backlog,
            tensor_offsets=offsets,
        )
        hot_by_state = _all_candidate_hot_indices_by_state(exact_path)
        control_inputs = [
            BoundedDeltaOracleInput(
                state_key=item.state_key,
                state=item.state,
                vote_inputs=item.vote_inputs,
                vote_spec=item.vote_spec,
                hot_exact_indices=hot_by_state.get(item.state_key, ()),
            )
            for item in inputs
        ]
        report = compare_bounded_delta_step_to_int16_oracle(
            control_inputs,
            q_ledger_row=q_ledger,
            guard_spec=guard_spec,
            global_cap_spec=cap_spec,
            deferred_backlog=backlog,
            tensor_offsets=offsets,
        )
        reports.append(
            _step_report_from_reference(
                schedule_name=schedule_step.name,
                step=int(schedule_step.step),
                mode=ONE_STEP_LOCAL_DIAGNOSTIC_MODE,
                curve_label=STRICT_CONTROL_LABEL,
                hot_budget_label="all_candidate_hot",
                requested_hot_budget=_hot_count(hot_by_state),
                is_max_hot_budget_point=False,
                max_hot_budget_for_policy=_hot_count(hot_by_state),
                actual_hot_exact_row_count=_hot_count(hot_by_state),
                backlog_policy_k=None,
                report=report,
                bounded_reinitialized_from_exact=True,
            )
        )
        states = _states_from_path(exact_path)
        if exact_path.cap_result is None:
            raise ValueError("strict control requires global cap result")
        backlog = exact_path.cap_result.deferred_backlog
    return tuple(reports)


def _run_cumulative_curve(
    q_ledger: Base3QEntropyLedgerRow,
    guard_spec: BoundedDeltaGuardSpec,
    *,
    hot_budget_label: str,
    backlog_k: int,
) -> RepresentativeCurveRunReport:
    exact_states = _initial_states()
    bounded_states = _initial_states()
    exact_backlog: dict[str, dict[int, dict[str, int]]] = {}
    bounded_backlog: dict[str, dict[int, dict[str, int]]] = {}
    max_hot_budget = _max_hot_budget_for_backlog_k(
        q_ledger,
        backlog_k=int(backlog_k),
        tensor_count=len(PRIMARY_STATE_KEYS),
    )
    requested_hot, is_max = _requested_hot_budget(hot_budget_label, max_hot_budget)
    curve_label = (
        f"{hot_budget_label}_"
        f"{BOUNDED_BACKLOG_LABEL_TEMPLATE.format(backlog_k=int(backlog_k))}"
    )
    step_reports: list[RepresentativeStepReport] = []
    for schedule_step in PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE:
        inputs, offsets = _make_step_inputs(exact_states, schedule_step)
        cap_spec = GlobalRateCapSpec(cap=int(schedule_step.cap), step=int(schedule_step.step))
        exact_path = _run_reference_path(
            inputs,
            states_by_key=exact_states,
            global_cap_spec=cap_spec,
            deferred_backlog=exact_backlog,
            tensor_offsets=offsets,
        )
        hot_by_state = _hot_indices_by_state_from_exact_path(exact_path, requested_hot)
        decoded_bounded_states: dict[str, VoteUpdateState] = {}
        for state_key in PRIMARY_STATE_KEYS:
            encoded = encode_budget_capped_hybrid_reference(
                bounded_states[state_key],
                hot_exact_indices=hot_by_state.get(state_key, ()),
                cold_default_value=0,
            )
            decoded_bounded_states[state_key] = VoteUpdateState(
                q_levels=bounded_states[state_key].q_levels.detach().clone().contiguous(),
                accumulators=decode_bounded_accumulator_to_i16(encoded),
            )
        bounded_path = _run_reference_path(
            inputs,
            states_by_key=decoded_bounded_states,
            global_cap_spec=cap_spec,
            deferred_backlog=bounded_backlog,
            tensor_offsets=offsets,
        )
        if exact_path.cap_result is None or bounded_path.cap_result is None:
            raise ValueError("cumulative representative verdict requires global cap results")
        bounded_stored_backlog = _select_stored_backlog(
            bounded_path.cap_result.deferred_backlog,
            priority_identities=exact_path.ordered_row_ids,
            max_entries=int(backlog_k),
        )
        projection = project_bounded_delta_accumulator_bpw(
            eligible_weight_count=int(q_ledger.eligible_weight_count),
            hot_exact_row_count=_hot_count(hot_by_state),
            backlog_entry_count=_backlog_entry_count(bounded_stored_backlog),
            tensor_metadata_bits=len(PRIMARY_STATE_KEYS) * DEFAULT_TENSOR_METADATA_BITS_PER_INPUT,
            bucket_metadata_bits=DEFAULT_BUCKET_METADATA_BITS,
            scale_metadata_bits=DEFAULT_SCALE_METADATA_BITS,
            guardrail_metadata_bits=DEFAULT_GUARDRAIL_METADATA_BITS,
        )
        report = _build_cumulative_reference_report(
            exact_path=exact_path,
            bounded_path=bounded_path,
            exact_input_states=exact_states,
            bounded_input_states=decoded_bounded_states,
            exact_backlog=exact_backlog,
            bounded_input_backlog=bounded_backlog,
            bounded_stored_backlog=bounded_stored_backlog,
            inputs=inputs,
            q_ledger_row=q_ledger,
            projection=projection,
            guard_spec=guard_spec,
            global_cap_spec=cap_spec,
            tensor_offsets=offsets,
            hot_by_state=hot_by_state,
        )
        step_reports.append(
            _step_report_from_reference(
                schedule_name=schedule_step.name,
                step=int(schedule_step.step),
                mode=CUMULATIVE_SCHEDULE_MODE,
                curve_label=curve_label,
                hot_budget_label=hot_budget_label,
                requested_hot_budget=requested_hot,
                is_max_hot_budget_point=is_max,
                max_hot_budget_for_policy=max_hot_budget,
                actual_hot_exact_row_count=_hot_count(hot_by_state),
                backlog_policy_k=int(backlog_k),
                report=report,
                bounded_reinitialized_from_exact=False,
            )
        )
        exact_states = _states_from_path(exact_path)
        bounded_states = _states_from_path(bounded_path)
        exact_backlog = _copy_backlog(exact_path.cap_result.deferred_backlog)
        bounded_backlog = _copy_backlog(bounded_stored_backlog)
    terminal = step_reports[-1]
    return RepresentativeCurveRunReport(
        curve_label=curve_label,
        mode=CUMULATIVE_SCHEDULE_MODE,
        hot_budget_label=hot_budget_label,
        requested_hot_budget=requested_hot,
        is_max_hot_budget_point=is_max,
        max_hot_budget_for_policy=max_hot_budget,
        backlog_policy_k=int(backlog_k),
        per_step_reports=tuple(step_reports),
        terminal_classification=terminal.classification,
    )


def run_representative_bounded_delta_drift_verdict() -> RepresentativeDriftVerdictReport:
    """Run the pre-registered representative C1.1c drift-vs-budget verdict."""

    q_ledger = _prior_large_q_ledger()
    guard_spec = representative_engineering_guard_spec()
    local_control = _run_one_step_strict_control(q_ledger, guard_spec)
    curve_reports = tuple(
        _run_cumulative_curve(
            q_ledger,
            guard_spec,
            hot_budget_label=hot_label,
            backlog_k=backlog_k,
        )
        for backlog_k in BACKLOG_K_POLICIES
        for hot_label in HOT_BUDGET_POINT_LABELS
    )
    primary_curve = next(
        run for run in curve_reports if run.curve_label == f"hotmax_{BOUNDED_BACKLOG_LABEL_TEMPLATE.format(backlog_k=32)}"
    )
    bindingness = pre_register_source_bindingness(
        source_kind=SOURCE_KIND_GENERATED_NATIVE_LOOP,
        coverage=SourceFieldCoverage.full_generated_native_loop(),
    )
    return RepresentativeDriftVerdictReport(
        schema_version=REPRESENTATIVE_VERDICT_SCHEMA_VERSION,
        label=REPRESENTATIVE_VERDICT_LABEL,
        terminal_mode=CUMULATIVE_SCHEDULE_MODE,
        terminal_science_question_closed=True,
        terminal_classification=primary_curve.terminal_classification,
        primary_curve_label=primary_curve.curve_label,
        source_bindingness=bindingness,
        field_coverage=SourceFieldCoverage.full_generated_native_loop(),
        q_ledger_regime_name=q_ledger.regime_name,
        guard_spec=guard_spec,
        hot_budget_points=HOT_BUDGET_POINT_LABELS,
        backlog_k_policies=BACKLOG_K_POLICIES,
        one_step_local_diagnostic_reports=local_control,
        cumulative_curve_reports=curve_reports,
        bindingness_statement=(
            f"{BINDING_FOR_IN_TREE_NATIVE_LOOP_DISTRIBUTION}; "
            f"{PARTIAL_FOR_S1_REAL_DYNAMICS}; not upgraded to live-S1 authority"
        ),
        residual_diversity_caveat=(
            "The generator starts from zero accumulators and deterministic disjoint vote ranges, "
            "so it stresses cap/backlog pressure more than residual accumulator-value diversity; "
            "low drift here would not imply low drift on real residual-diverse dynamics."
        ),
        guard_bound_adequacy_statement=(
            "The guard bounds are a pre-registered engineering diagnostic; C2 owns whether "
            "this drift is learnable/acquisition-stable."
        ),
        raw_arrays_included=False,
        non_claims=(
            "no production vote_update/global_rate_cap replacement",
            "no GPU lane",
            "no trainer/live-run/checkpoint/creditdir mutation",
            "no acquisition, retention, or stability claim",
            "guard-bound adequacy deferred to C2",
            "compact counts/hashes only; no raw per-weight arrays",
        ),
    )


def run_candidate_admission_diagnostic() -> CandidateAdmissionDiagnosticReport:
    """Run the CPU-only A/B/C oracle-upper-bound candidate admission diagnostic."""

    q_ledger = _prior_large_q_ledger()
    guard_spec = representative_engineering_guard_spec()
    bindingness = pre_register_source_bindingness(
        source_kind=SOURCE_KIND_GENERATED_NATIVE_LOOP,
        coverage=SourceFieldCoverage.full_generated_native_loop(),
    )
    exact_states = _initial_states()
    exact_backlog: dict[str, dict[int, dict[str, int]]] = {}
    null_states = _zero_accumulator_state_map(_initial_states())
    candidate_runtime: dict[str, dict[str, Any]] = {
        HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE: {
            "states": _copy_state_map(_initial_states()),
            "backlog": {},
            "backlog_policy_k": None,
            "steps": [],
        },
        EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE: {
            "states": _copy_state_map(_initial_states()),
            "backlog": {},
            "backlog_policy_k": 32,
            "steps": [],
        },
        COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE: {
            "states": _copy_state_map(_initial_states()),
            "backlog": {},
            "backlog_policy_k": 32,
            "steps": [],
        },
    }
    eligible = int(q_ledger.eligible_weight_count)

    for schedule_step in PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE:
        inputs, offsets = _make_step_inputs(exact_states, schedule_step)
        cap_spec = GlobalRateCapSpec(cap=int(schedule_step.cap), step=int(schedule_step.step))
        exact_path = _run_reference_path(
            inputs,
            states_by_key=exact_states,
            global_cap_spec=cap_spec,
            deferred_backlog=exact_backlog,
            tensor_offsets=offsets,
        )
        if exact_path.cap_result is None:
            raise ValueError("candidate admission diagnostic requires global cap results")
        exact_output_backlog = _copy_backlog(exact_path.cap_result.deferred_backlog)

        null_inputs = tuple(
            BoundedDeltaOracleInput(
                state_key=item.state_key,
                state=null_states[item.state_key],
                vote_inputs=item.vote_inputs,
                vote_spec=item.vote_spec,
                hot_exact_indices=(),
                cold_default_value=0,
            )
            for item in inputs
        )
        null_path = _run_reference_path(
            null_inputs,
            states_by_key=null_states,
            global_cap_spec=cap_spec,
            deferred_backlog={},
            tensor_offsets=offsets,
        )
        if null_path.cap_result is None:
            raise ValueError("null baseline diagnostic requires global cap results")
        null_report = compare_bounded_delta_paths_to_int16_oracle(
            inputs=null_inputs,
            q_ledger_row=q_ledger,
            exact_path=exact_path,
            bounded_path=null_path,
            storage_projection=project_bounded_delta_accumulator_bpw(
                eligible_weight_count=eligible,
                hot_exact_row_count=0,
                backlog_entry_count=0,
                tensor_metadata_bits=len(PRIMARY_STATE_KEYS) * DEFAULT_TENSOR_METADATA_BITS_PER_INPUT,
                bucket_metadata_bits=DEFAULT_BUCKET_METADATA_BITS,
                scale_metadata_bits=DEFAULT_SCALE_METADATA_BITS,
                guardrail_metadata_bits=DEFAULT_GUARDRAIL_METADATA_BITS,
            ),
            guard_spec=guard_spec,
            candidate_name=HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE,
            global_cap_spec=cap_spec,
            exact_input_states=exact_states,
            bounded_input_states=null_states,
            exact_input_backlog=exact_backlog,
            bounded_input_backlog={},
            bounded_stored_backlog={},
            tensor_offsets=offsets,
            bounded_backlog_policy_active=True,
            path_difference=(
                "accumulator-free null baseline removes accumulator and backlog carry "
                "from the bounded path while preserving the same q/vote/cap pipeline"
            ),
            oracle_parity_overrides={
                "cumulative_carry_forward": True,
                "bounded_reinitialized_from_exact": False,
                "baseline_label": ACCUMULATOR_FREE_NULL_BASELINE,
            },
            non_claims=(
                "accumulator-free null baseline is a continuity anchor only",
                "no production vote_update/global_rate_cap replacement",
                "no GPU lane",
                "no trainer/live-run/checkpoint/creditdir mutation",
                "compact counts/hashes only; no raw per-weight arrays",
            ),
        )
        null_states = _zero_accumulator_state_map(_states_from_path(null_path))

        for candidate_name in (
            HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE,
            EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE,
            COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE,
        ):
            runtime = candidate_runtime[candidate_name]
            candidate_input_backlog = _copy_backlog(runtime["backlog"])
            candidate_inputs, bounded_input_states, event_delta_count, dense_cold_bpw = (
                _candidate_inputs_and_states(
                    candidate_name=candidate_name,
                    inputs=inputs,
                    source_states=runtime["states"],
                    carried_backlog=candidate_input_backlog,
                    exact_path=exact_path,
                )
            )
            bounded_path = _run_reference_path(
                candidate_inputs,
                states_by_key=bounded_input_states,
                global_cap_spec=cap_spec,
                deferred_backlog=candidate_input_backlog,
                tensor_offsets=offsets,
            )
            if bounded_path.cap_result is None:
                raise ValueError("candidate admission diagnostic requires bounded cap results")
            backlog_policy_k = runtime["backlog_policy_k"]
            if backlog_policy_k is None:
                bounded_stored_backlog = _copy_backlog(bounded_path.cap_result.deferred_backlog)
            else:
                bounded_stored_backlog = _select_stored_backlog(
                    bounded_path.cap_result.deferred_backlog,
                    priority_identities=exact_path.ordered_row_ids,
                    max_entries=int(backlog_policy_k),
                )
            hot_row_count = _hot_count(
                {item.state_key: item.hot_exact_indices for item in candidate_inputs}
            )
            bounded_backlog_policy_active = (
                _backlog_key_set(exact_backlog) != _backlog_key_set(candidate_input_backlog)
                or _backlog_key_set(exact_output_backlog) != _backlog_key_set(bounded_stored_backlog)
            )
            candidate_report = compare_bounded_delta_paths_to_int16_oracle(
                inputs=candidate_inputs,
                q_ledger_row=q_ledger,
                exact_path=exact_path,
                bounded_path=bounded_path,
                storage_projection=project_bounded_delta_accumulator_bpw(
                    eligible_weight_count=eligible,
                    hot_exact_row_count=hot_row_count,
                    event_delta_count=event_delta_count,
                    backlog_entry_count=_backlog_entry_count(bounded_stored_backlog),
                    tensor_metadata_bits=len(PRIMARY_STATE_KEYS) * DEFAULT_TENSOR_METADATA_BITS_PER_INPUT,
                    bucket_metadata_bits=DEFAULT_BUCKET_METADATA_BITS,
                    scale_metadata_bits=DEFAULT_SCALE_METADATA_BITS,
                    guardrail_metadata_bits=DEFAULT_GUARDRAIL_METADATA_BITS,
                    dense_cold_bits_per_weight=dense_cold_bpw,
                ),
                guard_spec=guard_spec,
                candidate_name=candidate_name,
                global_cap_spec=cap_spec,
                exact_input_states=exact_states,
                bounded_input_states=bounded_input_states,
                exact_input_backlog=exact_backlog,
                bounded_input_backlog=candidate_input_backlog,
                bounded_stored_backlog=bounded_stored_backlog,
                tensor_offsets=offsets,
                bounded_backlog_policy_active=bounded_backlog_policy_active,
                path_difference=_candidate_path_difference(
                    candidate_name=candidate_name,
                    backlog_policy_k=backlog_policy_k,
                ),
                oracle_parity_overrides={
                    "cumulative_carry_forward": True,
                    "bounded_reinitialized_from_exact": False,
                    "builder_label": ORACLE_UPPER_BOUND_ADMISSION_DIAGNOSTIC,
                },
                non_claims=_oracle_upper_bound_non_claims(candidate_name=candidate_name),
            )
            runtime["steps"].append(
                CandidateAdmissionDiagnosticStepReport(
                    schedule_name=schedule_step.name,
                    step=int(schedule_step.step),
                    candidate_name=candidate_name,
                    builder_label=ORACLE_UPPER_BOUND_ADMISSION_DIAGNOSTIC,
                    backlog_policy_k=backlog_policy_k,
                    bounded_delta_report=candidate_report,
                    null_baseline_comparison=_compare_against_null(
                        candidate_report=candidate_report,
                        null_report=null_report,
                    ),
                    backlog_truncation_attribution=_backlog_truncation_report(
                        report=candidate_report,
                        exact_input_backlog=exact_backlog,
                        bounded_input_backlog=candidate_input_backlog,
                        exact_output_backlog=exact_output_backlog,
                        bounded_stored_backlog=bounded_stored_backlog,
                    ),
                )
            )
            runtime["states"] = _states_from_path(bounded_path)
            runtime["backlog"] = _copy_backlog(bounded_stored_backlog)

        exact_states = _states_from_path(exact_path)
        exact_backlog = exact_output_backlog

    candidate_runs: list[CandidateAdmissionDiagnosticRunReport] = []
    for candidate_name in (
        HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE,
        EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE,
        COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE,
    ):
        runtime = candidate_runtime[candidate_name]
        step_reports = tuple(runtime["steps"])
        failed_step_names = tuple(
            step.schedule_name
            for step in step_reports
            if not (
                step.bounded_delta_report.claimable_physical_sub2_with_guardrail
                and step.null_baseline_comparison.candidate_beats_null
            )
        )
        candidate_runs.append(
            CandidateAdmissionDiagnosticRunReport(
                candidate_name=candidate_name,
                builder_label=ORACLE_UPPER_BOUND_ADMISSION_DIAGNOSTIC,
                backlog_policy_k=runtime["backlog_policy_k"],
                per_step_reports=step_reports,
                terminal_decision=CandidatePromotionDecision(
                    candidate_name=candidate_name,
                    earns_dyn200_consideration=not failed_step_names,
                    status=(
                        "earns_dyn200_consideration"
                        if not failed_step_names
                        else "classified_null_with_rejection_telemetry"
                    ),
                    failed_step_names=failed_step_names,
                    oracle_upper_bound_only=True,
                ),
            )
        )

    return CandidateAdmissionDiagnosticReport(
        schema_version=CANDIDATE_ADMISSION_DIAGNOSTIC_SCHEMA_VERSION,
        label=CANDIDATE_ADMISSION_DIAGNOSTIC_LABEL,
        source_bindingness=bindingness,
        field_coverage=SourceFieldCoverage.full_generated_native_loop(),
        q_ledger_regime_name=q_ledger.regime_name,
        guard_spec=guard_spec,
        null_baseline_label=ACCUMULATOR_FREE_NULL_BASELINE,
        pre_registered_schedule=PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE,
        candidate_runs=tuple(candidate_runs),
        raw_arrays_included=False,
        non_claims=(
            "CPU-only oracle-upper-bound admission diagnostic",
            "PASS proves an upper bound, not a deployable online codec",
            "dyn200 remains a later gate outside this slice",
            "no GPU lane",
            "compact counts/hashes only; no raw per-weight arrays",
        ),
    )


def run_candidate_capacity_localization_diagnostic() -> CandidateCapacityLocalizationReport:
    """Localize whether A/B/C nulls are budget-lever, k-starvation, or redesign walls."""

    q_ledger = _prior_large_q_ledger()
    guard_spec = representative_engineering_guard_spec()
    bindingness = pre_register_source_bindingness(
        source_kind=SOURCE_KIND_GENERATED_NATIVE_LOOP,
        coverage=SourceFieldCoverage.full_generated_native_loop(),
    )
    baseline_report = run_candidate_admission_diagnostic()
    by_name = {run.candidate_name: run for run in baseline_report.candidate_runs}

    candidate_a_run = by_name[HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE]
    a_readouts = tuple(
        _candidate_a_budget_readout(step)
        for step in (
            CandidateCapacityStepReport(
                schedule_name=step.schedule_name,
                step=int(step.step),
                k_label="baseline",
                k_value=None,
                bounded_delta_report=step.bounded_delta_report,
                backlog_truncation_attribution=step.backlog_truncation_attribution,
                protected_surface_destructive_approximation_present=(
                    _protected_surface_destructive_approximation_present(step.bounded_delta_report)
                ),
                surface_fidelity_clears=bool(
                    step.bounded_delta_report.guard_passed
                    and step.bounded_delta_report.admission_passed
                ),
                packed_inclusive_physical_bits_per_weight=float(
                    step.bounded_delta_report.ledger.packed_inclusive_physical_bits_per_weight
                ),
                delta_over_2bpw=_delta_over_2bpw(
                    step.bounded_delta_report.ledger.packed_inclusive_physical_bits_per_weight
                ),
            )
            for step in candidate_a_run.per_step_reports
        )
    )
    a_terminal = a_readouts[-1]
    candidate_a_budget_report = CandidateABudgetLocalizationReport(
        candidate_name=HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE,
        per_step_readouts=a_readouts,
        terminal_budget_direction_label=_candidate_a_terminal_budget_direction(a_readouts),
        original_terminal_classification=(
            candidate_a_run.per_step_reports[-1].bounded_delta_report.classification
        ),
        original_terminal_rejection_summary=(
            candidate_a_run.per_step_reports[-1].bounded_delta_report.rejection_telemetry.summary
        ),
        cold_zero_counterfactual_terminal_bits_per_weight=float(
            a_terminal.cold_zero_counterfactual_bits_per_weight
        ),
        cold_zero_counterfactual_terminal_delta_over_2bpw=float(
            a_terminal.cold_zero_counterfactual_delta_over_2bpw
        ),
        cold_zero_counterfactual_terminal_clears_sub2=bool(
            a_terminal.cold_zero_counterfactual_clears_sub2
        ),
        non_claim=(
            "cold-zero counterfactual isolates the budget lever only; salvage still requires a "
            "future surface-faithful tighter-cold encoding"
        ),
    )

    trace_steps, max_exact_output_backlog_count = _build_exact_schedule_trace()
    backlog_k_values = _backlog_k_values(max_exact_output_backlog_count)
    backlog_k_schedule = tuple([str(value) for value in backlog_k_values] + ["unbounded"])
    sweep_runs: list[CandidateKSweepRunReport] = []
    for candidate_name in (
        EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE,
        COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE,
    ):
        sweep_entries = tuple(
            [
                _run_candidate_capacity_sweep(
                    candidate_name=candidate_name,
                    trace_steps=trace_steps,
                    q_ledger=q_ledger,
                    guard_spec=guard_spec,
                    backlog_policy_k=int(k_value),
                    k_label=str(int(k_value)),
                )
                for k_value in backlog_k_values
            ]
            + [
                _run_candidate_capacity_sweep(
                    candidate_name=candidate_name,
                    trace_steps=trace_steps,
                    q_ledger=q_ledger,
                    guard_spec=guard_spec,
                    backlog_policy_k=None,
                    k_label="unbounded",
                )
            ]
        )
        sweep_runs.append(
            CandidateKSweepRunReport(
                candidate_name=candidate_name,
                sweep_entries=sweep_entries,
                terminal_decision=_candidate_k_sweep_decision(
                    candidate_name=candidate_name,
                    sweep_entries=sweep_entries,
                ),
            )
        )

    return CandidateCapacityLocalizationReport(
        schema_version=CAPACITY_LOCALIZATION_DIAGNOSTIC_SCHEMA_VERSION,
        label=CAPACITY_LOCALIZATION_DIAGNOSTIC_LABEL,
        source_bindingness=bindingness,
        field_coverage=SourceFieldCoverage.full_generated_native_loop(),
        q_ledger_regime_name=q_ledger.regime_name,
        guard_spec=guard_spec,
        candidate_a_budget_report=candidate_a_budget_report,
        backlog_k_schedule=backlog_k_schedule,
        sweep_runs=tuple(sweep_runs),
        raw_arrays_included=False,
        non_claims=(
            "CPU-only oracle-upper-bound capacity localization",
            "A cold-zero counterfactual is a budget-lever direction signal only",
            "B/C k-sweep varies backlog capacity only; all other schedule/oracle terms are fixed",
            "PASS would still be oracle-upper-bound only, not a deployable codec",
            "no GPU lane",
            "compact counts/hashes only; no raw per-weight arrays",
        ),
    )


def _protected_surface_destructive_approximation_present_from_rejection(
    *,
    exact_surfaces: Sequence[str],
    rejection_telemetry: Any,
) -> bool:
    protected = set(str(surface) for surface in exact_surfaces)
    return any(
        item.surface in protected and item.status == "destructive_approximation"
        for item in rejection_telemetry.surfaces
    )


def _tensor_offsets_for_state_map(
    states: Mapping[str, VoteUpdateState],
) -> dict[str, int]:
    offsets: dict[str, int] = {}
    cursor = 0
    for state_key, state in states.items():
        offsets[str(state_key)] = cursor
        cursor += int(state.q_levels.numel())
    return offsets


def _lower_bound_trace_schedule_name(*, step: int, vote_pattern_step: int) -> str:
    return f"tiny_native_full_loop_step_{int(step)}_pattern_{int(vote_pattern_step)}"


def _lower_bound_trace_is_nontrivial(
    *,
    max_exact_output_backlog_count: int,
    cumulative_unique_backlog_identity_count: int,
    cumulative_backlog_membership_changed_count: int,
    max_exact_backlog_defer_count: int,
) -> bool:
    return bool(
        int(max_exact_output_backlog_count) >= LOWER_BOUND_TRACE_NONTRIVIAL_BACKLOG_SIZE
        or int(cumulative_unique_backlog_identity_count)
        >= LOWER_BOUND_TRACE_NONTRIVIAL_UNIQUE_IDENTITIES
        or int(cumulative_backlog_membership_changed_count)
        >= LOWER_BOUND_TRACE_NONTRIVIAL_MEMBERSHIP_CHANGES
        or int(max_exact_backlog_defer_count) >= 2
    )


def _lower_bound_backlog_k_values(max_exact_output_backlog_count: int) -> tuple[int, ...]:
    upper = max(0, int(max_exact_output_backlog_count))
    return tuple(range(0, upper + 1))


def _build_real_backlog_trace() -> tuple[
    tuple[_RealBacklogTraceStep, ...],
    RealBacklogTraceSummaryReport,
]:
    fixture = tiny_two_projection_vote_cap_fixture(device="cpu")
    state_keys = tuple(fixture.qscale_states.keys())
    states = {
        state_key: VoteUpdateState(
            q_levels=fixture.qscale_states[state_key].q_levels.detach().clone().contiguous(),
            accumulators=fixture.accumulators[state_key].detach().clone().contiguous(),
        )
        for state_key in state_keys
    }
    exact_backlog: dict[str, dict[int, dict[str, int]]] = {}
    spec = tiny_full_loop_vote_update_spec()
    started = time.perf_counter()
    previous_backlog_ids: set[tuple[str, int]] = set()
    previous_backlog_hash: str | None = None
    plateau_streak = 0
    saw_any_backlog = False
    nontrivial_backlog_reached = False
    plateau_detected = False
    max_exact_output_backlog_count = 0
    cumulative_unique_backlog_ids: set[tuple[str, int]] = set()
    cumulative_backlog_membership_changed_count = 0
    max_exact_backlog_age_steps = 0
    max_exact_backlog_defer_count = 0
    trace_steps: list[_RealBacklogTraceStep] = []
    step_reports: list[RealBacklogTraceStepReport] = []
    stop_reason = LOWER_BOUND_TRACE_STOP_MAX_STEPS

    for step in range(1, LOWER_BOUND_TRACE_MAX_STEPS + 1):
        vote_pattern_step = ((int(step) - 1) % 2) + 1
        votes_by_state = tiny_full_loop_votes_for_step(
            step,
            device="cpu",
            repeat_cycle=True,
        )
        inputs = tuple(
            BoundedDeltaOracleInput(
                state_key=state_key,
                state=states[state_key],
                vote_inputs=VoteUpdateInputs(votes=votes_by_state[state_key]),
                vote_spec=spec,
            )
            for state_key in state_keys
        )
        tensor_offsets = _tensor_offsets_for_state_map(states)
        cap_spec = GlobalRateCapSpec(cap=TINY_LOOP_GLOBAL_CAP, step=int(step))
        exact_path = _run_reference_path(
            inputs,
            states_by_key=states,
            global_cap_spec=cap_spec,
            deferred_backlog=exact_backlog,
            tensor_offsets=tensor_offsets,
        )
        if exact_path.cap_result is None:
            raise ValueError("real backlog lower-bound trace requires global cap results")
        exact_output_backlog = _copy_backlog(exact_path.cap_result.deferred_backlog)
        current_backlog_ids = _backlog_key_set(exact_output_backlog)
        saw_any_backlog = saw_any_backlog or bool(current_backlog_ids)
        backlog_membership_changed_count = len(previous_backlog_ids ^ current_backlog_ids)
        cumulative_backlog_membership_changed_count += backlog_membership_changed_count
        cumulative_unique_backlog_ids |= current_backlog_ids
        backlog_count = len(current_backlog_ids)
        max_exact_output_backlog_count = max(max_exact_output_backlog_count, backlog_count)
        backlog_age_steps = int(
            exact_path.cap_result.step_summary["deferred_backlog_max_age_steps"]
        )
        backlog_defer_count = int(
            exact_path.cap_result.step_summary["deferred_backlog_max_defer_count"]
        )
        max_exact_backlog_age_steps = max(max_exact_backlog_age_steps, backlog_age_steps)
        max_exact_backlog_defer_count = max(
            max_exact_backlog_defer_count,
            backlog_defer_count,
        )
        current_backlog_hash = _identity_sha256(current_backlog_ids)
        if current_backlog_ids and previous_backlog_hash == current_backlog_hash:
            plateau_streak += 1
        else:
            plateau_streak = 0
        nontrivial_backlog_reached = _lower_bound_trace_is_nontrivial(
            max_exact_output_backlog_count=max_exact_output_backlog_count,
            cumulative_unique_backlog_identity_count=len(cumulative_unique_backlog_ids),
            cumulative_backlog_membership_changed_count=(
                cumulative_backlog_membership_changed_count
            ),
            max_exact_backlog_defer_count=max_exact_backlog_defer_count,
        )
        plateau_detected = bool(current_backlog_ids) and (
            plateau_streak >= LOWER_BOUND_TRACE_PLATEAU_PATIENCE_STEPS
        )
        trace_steps.append(
            _RealBacklogTraceStep(
                schedule_name=_lower_bound_trace_schedule_name(
                    step=step,
                    vote_pattern_step=vote_pattern_step,
                ),
                step=int(step),
                vote_pattern_step=vote_pattern_step,
                inputs=inputs,
                tensor_offsets=tensor_offsets,
                cap_spec=cap_spec,
                exact_input_states=_copy_state_map(states),
                exact_input_backlog=_copy_backlog(exact_backlog),
                exact_path=exact_path,
                exact_output_backlog=exact_output_backlog,
            )
        )
        step_reports.append(
            RealBacklogTraceStepReport(
                schedule_name=_lower_bound_trace_schedule_name(
                    step=step,
                    vote_pattern_step=vote_pattern_step,
                ),
                step=int(step),
                vote_pattern_step=vote_pattern_step,
                pre_cap_demand_count=int(
                    exact_path.cap_result.step_summary["global_pre_cap_would_apply_count"]
                ),
                exact_candidate_count=len(exact_path.candidate_ids),
                exact_accepted_count=len(exact_path.accepted_ids),
                exact_deferred_count=len(exact_path.deferred_ids),
                exact_fired_count=len(exact_path.fired_ids),
                exact_output_backlog_count=backlog_count,
                exact_output_backlog_identities_sha256=current_backlog_hash,
                backlog_membership_changed_count_from_prior_step=(
                    backlog_membership_changed_count
                ),
                cumulative_unique_backlog_identity_count=len(cumulative_unique_backlog_ids),
                backlog_max_age_steps=backlog_age_steps,
                backlog_max_defer_count=backlog_defer_count,
            )
        )
        states = _states_from_path(exact_path)
        exact_backlog = exact_output_backlog
        previous_backlog_ids = current_backlog_ids
        previous_backlog_hash = current_backlog_hash
        if nontrivial_backlog_reached:
            stop_reason = LOWER_BOUND_TRACE_STOP_NONTRIVIAL
            break
        if plateau_detected:
            stop_reason = LOWER_BOUND_TRACE_STOP_PLATEAU
            break
        if (time.perf_counter() - started) >= LOWER_BOUND_TRACE_MAX_SECONDS:
            stop_reason = LOWER_BOUND_TRACE_STOP_CPU_SECONDS
            break

    elapsed_seconds = time.perf_counter() - started
    return (
        tuple(trace_steps),
        RealBacklogTraceSummaryReport(
            stop_reason=stop_reason,
            stop_step=len(trace_steps),
            plateau_patience_steps=LOWER_BOUND_TRACE_PLATEAU_PATIENCE_STEPS,
            max_steps_budget=LOWER_BOUND_TRACE_MAX_STEPS,
            cpu_seconds_budget=LOWER_BOUND_TRACE_MAX_SECONDS,
            elapsed_seconds=elapsed_seconds,
            saw_any_backlog=saw_any_backlog,
            nontrivial_backlog_reached=nontrivial_backlog_reached,
            plateau_detected=plateau_detected,
            max_exact_output_backlog_count=max_exact_output_backlog_count,
            cumulative_unique_backlog_identity_count=len(cumulative_unique_backlog_ids),
            cumulative_backlog_membership_changed_count=(
                cumulative_backlog_membership_changed_count
            ),
            max_exact_backlog_age_steps=max_exact_backlog_age_steps,
            max_exact_backlog_defer_count=max_exact_backlog_defer_count,
            per_step_reports=tuple(step_reports),
        ),
    )


def _run_real_backlog_lower_bound_sweep(
    *,
    trace_steps: Sequence[_RealBacklogTraceStep],
    guard_spec: BoundedDeltaGuardSpec,
    backlog_policy_k: int | None,
    k_label: str,
) -> RealBacklogLowerBoundSweepEntry:
    if not trace_steps:
        raise ValueError("real backlog lower-bound sweep requires at least one trace step")
    candidate_name = EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE
    contract = bounded_delta_admission_contract(candidate_name=candidate_name)
    bounded_states = _copy_state_map(trace_steps[0].exact_input_states)
    bounded_backlog: dict[str, dict[int, dict[str, int]]] = {}
    step_reports: list[RealBacklogLowerBoundStepReport] = []

    for trace_step in trace_steps:
        candidate_input_backlog = _copy_backlog(bounded_backlog)
        candidate_inputs, bounded_input_states, event_delta_count, _ = (
            _candidate_inputs_and_states(
                candidate_name=candidate_name,
                inputs=trace_step.inputs,
                source_states=bounded_states,
                carried_backlog=candidate_input_backlog,
                exact_path=trace_step.exact_path,
            )
        )
        bounded_path = _run_reference_path(
            candidate_inputs,
            states_by_key=bounded_input_states,
            global_cap_spec=trace_step.cap_spec,
            deferred_backlog=candidate_input_backlog,
            tensor_offsets=trace_step.tensor_offsets,
        )
        if bounded_path.cap_result is None:
            raise ValueError("real backlog lower-bound sweep requires bounded cap results")
        if backlog_policy_k is None:
            bounded_stored_backlog = _copy_backlog(bounded_path.cap_result.deferred_backlog)
        else:
            bounded_stored_backlog = _select_stored_backlog(
                bounded_path.cap_result.deferred_backlog,
                priority_identities=trace_step.exact_path.ordered_row_ids,
                max_entries=int(backlog_policy_k),
            )
        hot_row_count = _hot_count(
            {item.state_key: item.hot_exact_indices for item in candidate_inputs}
        )
        projection = project_bounded_delta_accumulator_bpw(
            eligible_weight_count=sum(
                int(item.state.q_levels.numel()) for item in candidate_inputs
            ),
            hot_exact_row_count=hot_row_count,
            event_delta_count=event_delta_count,
            backlog_entry_count=_backlog_entry_count(bounded_stored_backlog),
            tensor_metadata_bits=len(candidate_inputs) * DEFAULT_TENSOR_METADATA_BITS_PER_INPUT,
            bucket_metadata_bits=DEFAULT_BUCKET_METADATA_BITS,
            scale_metadata_bits=DEFAULT_SCALE_METADATA_BITS,
            guardrail_metadata_bits=DEFAULT_GUARDRAIL_METADATA_BITS,
        )
        measured = _build_measured_report_from_paths(
            inputs=candidate_inputs,
            candidate_name=candidate_name,
            exact_path=trace_step.exact_path,
            bounded_path=bounded_path,
            global_cap_spec=trace_step.cap_spec,
            exact_input_states=trace_step.exact_input_states,
            bounded_input_states=bounded_input_states,
            exact_input_backlog=trace_step.exact_input_backlog,
            bounded_input_backlog=candidate_input_backlog,
            bounded_stored_backlog=bounded_stored_backlog,
            tensor_offsets=trace_step.tensor_offsets,
            bounded_backlog_policy_active=(
                _backlog_key_set(trace_step.exact_input_backlog)
                != _backlog_key_set(candidate_input_backlog)
                or _backlog_key_set(trace_step.exact_output_backlog)
                != _backlog_key_set(bounded_stored_backlog)
            ),
            path_difference=_candidate_path_difference(
                candidate_name=candidate_name,
                backlog_policy_k=backlog_policy_k,
            ),
            oracle_parity_overrides={
                "cumulative_carry_forward": True,
                "bounded_reinitialized_from_exact": False,
                "builder_label": REAL_BACKLOG_LOWER_BOUND_LABEL,
                "swept_backlog_k_label": k_label,
                "trace_source": "tiny_native_full_loop_reference_stitch",
            },
        )
        guard_eval = _evaluate_guardrail(guard_spec, measured)
        admission_eval = _evaluate_bounded_delta_admission(contract, measured)
        step_reports.append(
            RealBacklogLowerBoundStepReport(
                schedule_name=trace_step.schedule_name,
                step=int(trace_step.step),
                k_label=k_label,
                k_value=backlog_policy_k,
                guard_passed=bool(guard_eval.guard_passed),
                admission_passed=bool(admission_eval.admission_passed),
                surface_fidelity_clears=bool(
                    guard_eval.guard_passed and admission_eval.admission_passed
                ),
                failed_metrics=tuple(guard_eval.failed_metrics),
                admission_failed_surfaces=tuple(admission_eval.failed_surfaces),
                rejection_summary=admission_eval.rejection_telemetry.summary,
                protected_surface_destructive_approximation_present=(
                    _protected_surface_destructive_approximation_present_from_rejection(
                        exact_surfaces=contract.exact_surfaces,
                        rejection_telemetry=admission_eval.rejection_telemetry,
                    )
                ),
                bounded_delta_acc_bits_per_weight=float(
                    projection.bounded_delta_acc_bits_per_weight
                ),
                backlog_entry_count=int(projection.backlog_entry_count),
                hot_exact_row_count=int(projection.hot_exact_row_count),
                event_delta_count=int(projection.event_delta_count),
                measured_report=measured,
                backlog_truncation_attribution=_backlog_truncation_report(
                    report=type(
                        "_SurfaceOnlyReport",
                        (),
                        {
                            "measured_report": measured,
                            "rejection_telemetry": admission_eval.rejection_telemetry,
                        },
                    )(),
                    exact_input_backlog=trace_step.exact_input_backlog,
                    bounded_input_backlog=candidate_input_backlog,
                    exact_output_backlog=trace_step.exact_output_backlog,
                    bounded_stored_backlog=bounded_stored_backlog,
                ),
            )
        )
        bounded_states = _states_from_path(bounded_path)
        bounded_backlog = _copy_backlog(bounded_stored_backlog)

    terminal = step_reports[-1]
    return RealBacklogLowerBoundSweepEntry(
        candidate_name=candidate_name,
        k_label=k_label,
        k_value=backlog_policy_k,
        per_step_reports=tuple(step_reports),
        all_steps_surface_fidelity_clears=all(
            step.surface_fidelity_clears for step in step_reports
        ),
        peak_bounded_delta_acc_bits_per_weight=max(
            step.bounded_delta_acc_bits_per_weight for step in step_reports
        ),
        terminal_bounded_delta_acc_bits_per_weight=float(
            terminal.bounded_delta_acc_bits_per_weight
        ),
        terminal_rejection_summary=terminal.rejection_summary,
    )


def _real_backlog_lower_bound_decision(
    *,
    trace_summary: RealBacklogTraceSummaryReport,
    sweep_entries: Sequence[RealBacklogLowerBoundSweepEntry],
    budget_report: Any,
) -> RealBacklogLowerBoundDecision:
    actual_headroom_bits_per_weight = float(
        budget_report.required_acc_bits_per_weight_for_sub2_physical_q_with_scale_and_metadata
    )
    minimal_surface_faithful = next(
        (entry for entry in sweep_entries if entry.all_steps_surface_fidelity_clears),
        None,
    )
    minimal_peak = (
        None
        if minimal_surface_faithful is None
        else float(minimal_surface_faithful.peak_bounded_delta_acc_bits_per_weight)
    )
    headroom_minus_minimal = (
        None
        if minimal_peak is None
        else float(actual_headroom_bits_per_weight) - float(minimal_peak)
    )
    minimal_fits_headroom = bool(
        minimal_peak is not None and float(minimal_peak) <= float(actual_headroom_bits_per_weight)
    )
    if trace_summary.nontrivial_backlog_reached and minimal_fits_headroom:
        label = SPARSE_AMORTIZED_CANDIDATE_RESURRECTED_FOR_HARDER_TRACE
        reason = (
            "nontrivial backlog/churn surfaced on the tiny native stitch and the minimal "
            f"surface-faithful k={minimal_surface_faithful.k_label} peak acc bpw "
            f"{minimal_peak:.6f} fits tiny-fixture headroom {float(actual_headroom_bits_per_weight):.6f}; "
            "this is still not a global closure or branch trigger"
        )
    elif trace_summary.saw_any_backlog and (
        trace_summary.plateau_detected or trace_summary.nontrivial_backlog_reached
    ):
        label = PER_ROW_COMPRESSION_CLOSED_BY_EASY_CASE_LOWER_BOUND
        if minimal_surface_faithful is None:
            reason = (
                "even the unbounded lower-bound sweep never cleared the protected "
                "decision surfaces, so backlog capacity is not the missing lever here"
            )
        else:
            reason = (
                f"minimal surface-faithful k={minimal_surface_faithful.k_label} still peaks at "
                f"{minimal_peak:.6f} acc bpw against tiny-fixture headroom "
                f"{float(actual_headroom_bits_per_weight):.6f}; tiny-fixture-only lower bound, "
                "not global closure"
            )
    else:
        label = REPRESENTATIVE_TRACE_UNDERPOWERED_FOR_CLOSURE
        reason = (
            f"trace stop={trace_summary.stop_reason}; saw_any_backlog={trace_summary.saw_any_backlog}; "
            f"minimal surface-faithful k={None if minimal_surface_faithful is None else minimal_surface_faithful.k_label}"
        )
    return RealBacklogLowerBoundDecision(
        terminal_label=label,
        headroom_source=TINY_FIXTURE_HEADROOM_SOURCE,
        eligible_weight_count=int(budget_report.eligible_weight_count),
        q_packed_data_bits_per_weight=float(budget_report.q_packed_data_bits_per_weight),
        q_packed_metadata_bits_per_weight=float(
            budget_report.q_packed_metadata_bits_per_weight
        ),
        q_packed_total_bits_per_weight=float(budget_report.q_packed_total_bits_per_weight),
        frozen_scale_fp32_bits_per_weight=float(
            budget_report.frozen_scale_fp32_bits_per_weight
        ),
        actual_remaining_accumulator_headroom_bits_per_weight=float(
            actual_headroom_bits_per_weight
        ),
        minimal_surface_faithful_k_label=(
            None if minimal_surface_faithful is None else minimal_surface_faithful.k_label
        ),
        minimal_surface_faithful_k_value=(
            None if minimal_surface_faithful is None else minimal_surface_faithful.k_value
        ),
        minimal_surface_faithful_peak_bounded_delta_acc_bits_per_weight=minimal_peak,
        headroom_minus_minimal_surface_faithful_peak_bits_per_weight=(
            headroom_minus_minimal
        ),
        minimal_surface_faithful_k_fits_headroom=minimal_fits_headroom,
        global_per_row_compression_closed=False,
        branch_a_trigger=False,
        reason=reason,
    )


def run_real_backlog_lower_bound_diagnostic() -> RealBacklogLowerBoundReport:
    """Characterize the B candidate against the cheapest real backlog-bearing stitch."""

    guard_spec = representative_engineering_guard_spec()
    bindingness = pre_register_source_bindingness(
        source_kind=SOURCE_KIND_GENERATED_NATIVE_LOOP,
        coverage=SourceFieldCoverage.full_generated_native_loop(),
    )
    trace_steps, trace_summary = _build_real_backlog_trace()
    backlog_k_values = _lower_bound_backlog_k_values(
        trace_summary.max_exact_output_backlog_count
    )
    backlog_k_schedule = tuple([str(value) for value in backlog_k_values] + ["unbounded"])
    sweep_entries = tuple(
        [
            _run_real_backlog_lower_bound_sweep(
                trace_steps=trace_steps,
                guard_spec=guard_spec,
                backlog_policy_k=int(k_value),
                k_label=str(int(k_value)),
            )
            for k_value in backlog_k_values
        ]
        + [
            _run_real_backlog_lower_bound_sweep(
                trace_steps=trace_steps,
                guard_spec=guard_spec,
                backlog_policy_k=None,
                k_label="unbounded",
            )
        ]
    )
    budget_report = measure_tiny_two_projection_fixture_budget(device="cpu")
    return RealBacklogLowerBoundReport(
        schema_version=REAL_BACKLOG_LOWER_BOUND_SCHEMA_VERSION,
        label=REAL_BACKLOG_LOWER_BOUND_LABEL,
        source_bindingness=bindingness,
        field_coverage=SourceFieldCoverage.full_generated_native_loop(),
        q_persistent_budget_label=budget_report.label,
        candidate_name=EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE,
        guard_spec=guard_spec,
        exact_trace_summary=trace_summary,
        backlog_k_schedule=backlog_k_schedule,
        sweep_entries=sweep_entries,
        terminal_decision=_real_backlog_lower_bound_decision(
            trace_summary=trace_summary,
            sweep_entries=sweep_entries,
            budget_report=budget_report,
        ),
        raw_arrays_included=False,
        non_claims=(
            "CPU-only tiny native full-loop lower-bound diagnostic",
            "uses the current tiny fixture q+scale headroom, not a sub-2 q upgrade claim",
            "global_per_row_compression_closed=false",
            "branch_a_trigger=false",
            "surface-faithful pass is a routing signal only, not a deployable codec",
            "no GPU lane",
            "no trainer/live-run/checkpoint/creditdir mutation",
            "compact counts/hashes only; no raw per-weight arrays",
        ),
    )


def _scale_count_with_density(
    *,
    source_count: int,
    source_eligible_weight_count: int,
    target_eligible_weight_count: int,
) -> int:
    source = int(source_count)
    if source <= 0:
        return 0
    scaled = math.ceil(
        float(source) / float(int(source_eligible_weight_count)) * float(int(target_eligible_weight_count))
    )
    return min(int(target_eligible_weight_count), int(max(1, scaled)))


def _scale_density_assumption(
    source_step_reports: Sequence[RealBacklogLowerBoundStepReport],
    *,
    source_eligible_weight_count: int,
) -> str:
    parts = []
    for step in source_step_reports:
        parts.append(
            f"{step.schedule_name}: hot={int(step.hot_exact_row_count)}/{int(source_eligible_weight_count)} "
            f"event={int(step.event_delta_count)}/{int(source_eligible_weight_count)} "
            f"backlog={int(step.backlog_entry_count)}/{int(source_eligible_weight_count)}"
        )
    return "; ".join(parts)


def _scale_projection_step_report(
    *,
    source_step: RealBacklogLowerBoundStepReport,
    q_ledger_row: Base3QEntropyLedgerRow,
    projection_label: str,
    target_hot_exact_row_count: int,
    target_event_delta_count: int,
    target_backlog_entry_count: int,
    decisive_for_branch: bool,
    rounding_policy: str,
) -> ScaleAppropriateProjectionStepReport:
    projection = project_bounded_delta_accumulator_bpw(
        eligible_weight_count=int(q_ledger_row.eligible_weight_count),
        hot_exact_row_count=int(target_hot_exact_row_count),
        event_delta_count=int(target_event_delta_count),
        backlog_entry_count=int(target_backlog_entry_count),
        tensor_metadata_bits=int(q_ledger_row.q_state_count)
        * DEFAULT_TENSOR_METADATA_BITS_PER_INPUT,
        bucket_metadata_bits=DEFAULT_BUCKET_METADATA_BITS,
        scale_metadata_bits=DEFAULT_SCALE_METADATA_BITS,
        guardrail_metadata_bits=DEFAULT_GUARDRAIL_METADATA_BITS,
    )
    headroom = float(q_ledger_row.remaining_accumulator_budget_bits_per_weight)
    bpw = float(projection.bounded_delta_acc_bits_per_weight)
    return ScaleAppropriateProjectionStepReport(
        schedule_name=source_step.schedule_name,
        step=int(source_step.step),
        projection_label=projection_label,
        source_hot_exact_row_count=int(source_step.hot_exact_row_count),
        source_event_delta_count=int(source_step.event_delta_count),
        source_backlog_entry_count=int(source_step.backlog_entry_count),
        target_hot_exact_row_count=int(target_hot_exact_row_count),
        target_event_delta_count=int(target_event_delta_count),
        target_backlog_entry_count=int(target_backlog_entry_count),
        target_index_bits_per_row=int(projection.index_bits_per_row),
        tensor_metadata_bits=int(projection.tensor_metadata_bits),
        bucket_metadata_bits=int(projection.bucket_metadata_bits),
        scale_metadata_bits=int(projection.scale_metadata_bits),
        guardrail_metadata_bits=int(projection.guardrail_metadata_bits),
        bounded_delta_acc_bits_per_weight=bpw,
        exceeds_scale_headroom=bool(bpw > headroom),
        decisive_for_branch=bool(decisive_for_branch),
        rounding_policy=rounding_policy,
    )


def _scale_appropriate_comparison_for_row(
    *,
    q_ledger_row: Base3QEntropyLedgerRow,
    row_role: str,
    source_step_reports: Sequence[RealBacklogLowerBoundStepReport],
    source_eligible_weight_count: int,
) -> ScaleAppropriateLedgerComparisonReport:
    density_assumption = _scale_density_assumption(
        source_step_reports,
        source_eligible_weight_count=source_eligible_weight_count,
    )
    absolute_reports = tuple(
        _scale_projection_step_report(
            source_step=step,
            q_ledger_row=q_ledger_row,
            projection_label=ABSOLUTE_COUNT_LOWER_BOUND_DIAGNOSTIC,
            target_hot_exact_row_count=int(step.hot_exact_row_count),
            target_event_delta_count=int(step.event_delta_count),
            target_backlog_entry_count=int(step.backlog_entry_count),
            decisive_for_branch=False,
            rounding_policy="hold_absolute_counts_fixed_under_target_index_meta",
        )
        for step in source_step_reports
    )
    rate_reports = tuple(
        _scale_projection_step_report(
            source_step=step,
            q_ledger_row=q_ledger_row,
            projection_label=RATE_HELD_B_STORAGE_DIAGNOSTIC,
            target_hot_exact_row_count=_scale_count_with_density(
                source_count=int(step.hot_exact_row_count),
                source_eligible_weight_count=source_eligible_weight_count,
                target_eligible_weight_count=int(q_ledger_row.eligible_weight_count),
            ),
            target_event_delta_count=_scale_count_with_density(
                source_count=int(step.event_delta_count),
                source_eligible_weight_count=source_eligible_weight_count,
                target_eligible_weight_count=int(q_ledger_row.eligible_weight_count),
            ),
            target_backlog_entry_count=_scale_count_with_density(
                source_count=int(step.backlog_entry_count),
                source_eligible_weight_count=source_eligible_weight_count,
                target_eligible_weight_count=int(q_ledger_row.eligible_weight_count),
            ),
            decisive_for_branch=True,
            rounding_policy=RATE_HELD_COUNT_ROUNDING_POLICY,
        )
        for step in source_step_reports
    )
    absolute_peak = max(step.bounded_delta_acc_bits_per_weight for step in absolute_reports)
    rate_peak = max(step.bounded_delta_acc_bits_per_weight for step in rate_reports)
    headroom = float(q_ledger_row.remaining_accumulator_budget_bits_per_weight)
    return ScaleAppropriateLedgerComparisonReport(
        q_regime_name=q_ledger_row.regime_name,
        row_role=row_role,
        eligible_weight_count=int(q_ledger_row.eligible_weight_count),
        q_packed_data_bits_per_weight=float(q_ledger_row.q_packed_data_bits_per_weight),
        q_packed_metadata_bits_per_weight=float(
            q_ledger_row.q_packed_metadata_bits_per_weight
        ),
        q_packed_total_bits_per_weight=float(q_ledger_row.q_packed_total_bits_per_weight),
        frozen_scale_fp32_bits_per_weight=float(
            q_ledger_row.frozen_scale_fp32_bits_per_weight
        ),
        scale_appropriate_headroom_bits_per_weight=headroom,
        density_assumption=density_assumption,
        absolute_count_lower_bound_step_reports=absolute_reports,
        rate_held_b_storage_step_reports=rate_reports,
        absolute_count_lower_bound_peak_bounded_delta_acc_bits_per_weight=float(
            absolute_peak
        ),
        rate_held_b_storage_peak_bounded_delta_acc_bits_per_weight=float(rate_peak),
        absolute_count_lower_bound_exceeds_scale_headroom=bool(absolute_peak > headroom),
        rate_held_b_storage_exceeds_scale_headroom=bool(rate_peak > headroom),
    )


def _scale_appropriate_b_storage_decision(
    row_comparisons: Sequence[ScaleAppropriateLedgerComparisonReport],
) -> ScaleAppropriateComparisonDecision:
    required_rows = tuple(
        row for row in row_comparisons if row.row_role == "required_gate"
    )
    if not required_rows:
        raise ValueError("scale-appropriate B comparison requires required gate rows")
    required_row_names = tuple(row.q_regime_name for row in required_rows)
    all_required_rate_held_exceed = all(
        row.rate_held_b_storage_exceeds_scale_headroom for row in required_rows
    )
    any_required_absolute_exceed = any(
        row.absolute_count_lower_bound_exceeds_scale_headroom for row in required_rows
    )
    if all_required_rate_held_exceed:
        label = RATE_HELD_B_STILL_OVER_SCALE_HEADROOM_CANDIDATE_BRANCH_A
        reason = (
            "rate-held B storage stays above scale-appropriate accumulator headroom on every "
            f"required row ({', '.join(required_row_names)}); this earns a candidate structural-pivot "
            "trigger under the explicit tiny-density assumption, while branch_a_trigger remains false "
            "until Claude routes it"
        )
        candidate_branch = True
    else:
        failing = ", ".join(
            row.q_regime_name
            for row in required_rows
            if not row.rate_held_b_storage_exceeds_scale_headroom
        )
        label = SCALE_APPROPRIATE_COMPARISON_AMBIGUOUS_NEEDS_BACKLOG_DENSITY_TRACE
        reason = (
            "rate-held B storage does not cleanly exceed scale headroom on every required row "
            f"(non-exceeding: {failing or 'none'}); absolute-count rows remain non-decisive, so the "
            "next honest step is a backlog-density trace"
        )
        candidate_branch = False
    return ScaleAppropriateComparisonDecision(
        terminal_label=label,
        required_rows=required_row_names,
        rate_held_density_assumption_explicit=True,
        required_rows_all_rate_held_exceed_scale_headroom=bool(
            all_required_rate_held_exceed
        ),
        any_required_absolute_count_lower_bound_exceeds_scale_headroom=bool(
            any_required_absolute_exceed
        ),
        candidate_branch_a_trigger_earned=bool(candidate_branch),
        global_per_row_compression_closed=False,
        branch_a_trigger=False,
        reason=reason,
    )


def run_scale_appropriate_b_storage_comparison() -> ScaleAppropriateBStorageComparisonReport:
    """Compare Slice 1c's minimal surface-faithful B storage against scale headroom."""

    lower_bound = run_real_backlog_lower_bound_diagnostic()
    minimal_surface_faithful = next(
        (
            entry
            for entry in lower_bound.sweep_entries
            if entry.k_label == lower_bound.terminal_decision.minimal_surface_faithful_k_label
        ),
        None,
    )
    if minimal_surface_faithful is None:
        raise ValueError("Slice 1d requires the committed Slice 1c minimal surface-faithful B entry")
    source_eligible = int(lower_bound.terminal_decision.eligible_weight_count)
    bindingness = pre_register_source_bindingness(
        source_kind=SOURCE_KIND_GENERATED_NATIVE_LOOP,
        coverage=SourceFieldCoverage.full_generated_native_loop(),
    )
    row_comparisons = tuple(
        [
            _scale_appropriate_comparison_for_row(
                q_ledger_row=_q_ledger_row_by_name(regime_name),
                row_role="required_gate",
                source_step_reports=minimal_surface_faithful.per_step_reports,
                source_eligible_weight_count=source_eligible,
            )
            for regime_name in SCALE_REQUIRED_Q_LEDGER_ROWS
        ]
        + [
            _scale_appropriate_comparison_for_row(
                q_ledger_row=_q_ledger_row_by_name(regime_name),
                row_role="sensitivity_only",
                source_step_reports=minimal_surface_faithful.per_step_reports,
                source_eligible_weight_count=source_eligible,
            )
            for regime_name in SCALE_SENSITIVITY_Q_LEDGER_ROWS
        ]
    )
    return ScaleAppropriateBStorageComparisonReport(
        schema_version=SCALE_APPROPRIATE_B_STORAGE_SCHEMA_VERSION,
        label=SCALE_APPROPRIATE_B_STORAGE_LABEL,
        source_bindingness=bindingness,
        field_coverage=SourceFieldCoverage.full_generated_native_loop(),
        candidate_name=EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE,
        source_lower_bound_label=lower_bound.label,
        source_terminal_label=lower_bound.terminal_decision.terminal_label,
        source_minimal_surface_faithful_k_label=(
            lower_bound.terminal_decision.minimal_surface_faithful_k_label or "missing"
        ),
        source_minimal_surface_faithful_k_value=int(
            lower_bound.terminal_decision.minimal_surface_faithful_k_value or -1
        ),
        source_tiny_eligible_weight_count=source_eligible,
        density_rounding_policy=RATE_HELD_COUNT_ROUNDING_POLICY,
        required_q_ledger_rows=SCALE_REQUIRED_Q_LEDGER_ROWS,
        sensitivity_q_ledger_rows=SCALE_SENSITIVITY_Q_LEDGER_ROWS,
        row_comparisons=row_comparisons,
        terminal_decision=_scale_appropriate_b_storage_decision(row_comparisons),
        raw_arrays_included=False,
        non_claims=(
            "CPU-only scale-appropriate comparison built on the Slice 1c oracle-upper-bound B seam",
            "rate-held rows assume tiny-trace hot/event/backlog density scales by eligible weight",
            "absolute_count_lower_bound rows are non-decisive diagnostics only",
            "global_per_row_compression_closed=false",
            "branch_a_trigger=false",
            "candidate_branch_a_trigger_earned is advisory-only until Claude routes it",
            "no GPU lane",
            "no dyn200, no heavier backlog-density trace in this slice",
            "compact counts/hashes only; no raw per-weight arrays",
        ),
    )


@dataclass(frozen=True)
class _ObservableDecisionRow:
    state_key: str
    flat_index: int
    current_q_level: int
    move_direction: int
    abs_new_acc: int

    @property
    def identity(self) -> tuple[str, int]:
        return (self.state_key, int(self.flat_index))

    @property
    def bucket_key(self) -> tuple[str, int, int]:
        return (self.state_key, int(self.current_q_level), int(self.move_direction))


def _enum_bit_width(num_values: int) -> int:
    count = int(num_values)
    if count <= 1:
        return 0
    return int(math.ceil(math.log2(count)))


def _count_bit_width(max_value: int) -> int:
    value = int(max_value)
    if value <= 0:
        return 0
    return int(math.ceil(math.log2(value + 1)))


def _decision_statistic_bucket_cardinality_bound() -> int:
    return len(PRIMARY_STATE_KEYS) * 3 * 2


def _decision_statistic_bucket_key_bit_width() -> int:
    return (
        _enum_bit_width(len(PRIMARY_STATE_KEYS))
        + _enum_bit_width(3)
        + _enum_bit_width(2)
    )


def _strictest_required_scale_row(
    report: ScaleAppropriateBStorageComparisonReport,
) -> ScaleAppropriateLedgerComparisonReport:
    required_rows = [row for row in report.row_comparisons if row.row_role == "required_gate"]
    if not required_rows:
        raise ValueError("decision statistic diagnostic requires the Slice 1d required rows")
    return min(
        required_rows,
        key=lambda row: float(row.scale_appropriate_headroom_bits_per_weight),
    )


def _decision_statistic_observable_rows(
    trace_step: _ExactScheduleTraceStep,
) -> tuple[_ObservableDecisionRow, ...]:
    cap_result = trace_step.exact_path.cap_result
    if cap_result is None:
        raise ValueError("decision statistic diagnostic requires global cap rows")
    rows: list[_ObservableDecisionRow] = []
    for row in cap_result.rows:
        state = trace_step.exact_input_states[row.state_key]
        current_q_level = int(state.q_levels.flatten()[int(row.flat_index)].item())
        direction = int(
            trace_step.exact_path.plans[row.state_key]
            .applied_directions[int(row.local_pos)]
            .item()
        )
        rows.append(
            _ObservableDecisionRow(
                state_key=row.state_key,
                flat_index=int(row.flat_index),
                current_q_level=current_q_level,
                move_direction=direction,
                abs_new_acc=int(row.abs_new_acc),
            )
        )
    return tuple(rows)


def _decision_statistic_bucket_summaries(
    observable_rows: Sequence[_ObservableDecisionRow],
    *,
    exact_accepted: set[tuple[str, int]],
    exact_deferred: set[tuple[str, int]],
) -> tuple[DecisionStatisticBucketSummary, ...]:
    by_bucket: dict[tuple[str, int, int], list[_ObservableDecisionRow]] = {}
    for row in observable_rows:
        by_bucket.setdefault(row.bucket_key, []).append(row)
    summaries: list[DecisionStatisticBucketSummary] = []
    for bucket_key in sorted(by_bucket):
        rows = by_bucket[bucket_key]
        candidate_row_count = len(rows)
        accepted_count = sum(row.identity in exact_accepted for row in rows)
        deferred_count = sum(row.identity in exact_deferred for row in rows)
        decisive_bucket = 0 < accepted_count < candidate_row_count
        scores = sorted((row.abs_new_acc for row in rows), reverse=True)
        frontier_tie_crosses_boundary = bool(
            decisive_bucket
            and scores[accepted_count - 1] == scores[accepted_count]
        )
        summaries.append(
            DecisionStatisticBucketSummary(
                state_key=bucket_key[0],
                current_q_level=int(bucket_key[1]),
                move_direction=int(bucket_key[2]),
                accepted_count=int(accepted_count),
                deferred_count=int(deferred_count),
                candidate_row_count=int(candidate_row_count),
                decisive_bucket=bool(decisive_bucket),
                frontier_tie_crosses_boundary=bool(frontier_tie_crosses_boundary),
            )
        )
    return tuple(summaries)


def _decision_statistic_schema_report(
    *,
    bucket_summaries: Sequence[DecisionStatisticBucketSummary],
    candidate_row_count: int,
    global_cap: int,
    strictest_required_row: ScaleAppropriateLedgerComparisonReport,
) -> DecisionStatisticSchemaReport:
    observed_bucket_count = len(tuple(bucket_summaries))
    bucket_key_bit_width = _decision_statistic_bucket_key_bit_width()
    accepted_count_bit_width = _count_bit_width(global_cap)
    deferred_count_bit_width = _count_bit_width(candidate_row_count)
    total_bits = (
        observed_bucket_count
        * (
            bucket_key_bit_width
            + accepted_count_bit_width
            + deferred_count_bit_width
            + DECISION_STATISTIC_CUTOFF_BIT_WIDTH
        )
        + DECISION_STATISTIC_SEED_BITS
        + DECISION_STATISTIC_METADATA_BITS
    )
    eligible = int(strictest_required_row.eligible_weight_count)
    bits_per_weight = float(total_bits) / float(eligible) if eligible else 0.0
    headroom = float(strictest_required_row.scale_appropriate_headroom_bits_per_weight)
    fits = bits_per_weight <= headroom + 1e-12
    return DecisionStatisticSchemaReport(
        bucket_key_dimensions=DECISION_STATISTIC_BUCKET_KEY_DIMENSIONS,
        bucket_cardinality_bound=_decision_statistic_bucket_cardinality_bound(),
        observed_bucket_count=int(observed_bucket_count),
        bucket_key_bit_width=int(bucket_key_bit_width),
        accepted_count_bit_width=int(accepted_count_bit_width),
        deferred_count_bit_width=int(deferred_count_bit_width),
        cutoff_bit_width=int(DECISION_STATISTIC_CUTOFF_BIT_WIDTH),
        seed_bits=int(DECISION_STATISTIC_SEED_BITS),
        metadata_bits=int(DECISION_STATISTIC_METADATA_BITS),
        total_bits=int(total_bits),
        strictest_required_q_regime_name=strictest_required_row.q_regime_name,
        strictest_required_eligible_weight_count=int(eligible),
        strictest_required_headroom_bits_per_weight=headroom,
        total_bits_per_weight_strictest_required_row=float(bits_per_weight),
        fits_strictest_required_headroom=bool(fits),
        inclusive_sub2_if_installed=bool(fits),
        statistic_mode=DECISION_STATISTIC_COUNT_ONLY_MODE,
    )


def _reconstruct_decision_sets_from_bucket_counts(
    observable_rows: Sequence[_ObservableDecisionRow],
    bucket_summaries: Sequence[DecisionStatisticBucketSummary],
    *,
    reverse_bucket_order: bool,
) -> tuple[set[tuple[str, int]], set[tuple[str, int]], set[tuple[str, int]]]:
    summary_by_key = {
        (bucket.state_key, int(bucket.current_q_level), int(bucket.move_direction)): bucket
        for bucket in bucket_summaries
    }
    by_bucket: dict[tuple[str, int, int], list[_ObservableDecisionRow]] = {}
    for row in observable_rows:
        by_bucket.setdefault(row.bucket_key, []).append(row)
    accepted: set[tuple[str, int]] = set()
    deferred: set[tuple[str, int]] = set()
    for bucket_key in sorted(by_bucket):
        rows = list(by_bucket[bucket_key])
        if reverse_bucket_order:
            rows.reverse()
        ordered = sorted(rows, key=lambda item: -item.abs_new_acc)
        summary = summary_by_key[bucket_key]
        if summary.accepted_count + summary.deferred_count != len(ordered):
            raise ValueError("decision statistic reconstruction must cover every cap row")
        accepted |= {row.identity for row in ordered[: int(summary.accepted_count)]}
        deferred |= {
            row.identity
            for row in ordered[
                int(summary.accepted_count) : int(summary.accepted_count)
                + int(summary.deferred_count)
            ]
        }
    return accepted, deferred, set(accepted)


def _decision_statistic_step_report(
    *,
    trace_step: _ExactScheduleTraceStep,
    strictest_required_row: ScaleAppropriateLedgerComparisonReport,
) -> DecisionStatisticStepReport:
    cap_result = trace_step.exact_path.cap_result
    if cap_result is None:
        raise ValueError("decision statistic diagnostic requires cap-result exact paths")
    observable_rows = _decision_statistic_observable_rows(trace_step)
    exact_accepted = {(row.state_key, int(row.flat_index)) for row in cap_result.accepted_rows}
    exact_deferred = {(row.state_key, int(row.flat_index)) for row in cap_result.deferred_rows}
    exact_q_changed = set(trace_step.exact_path.q_changed_ids)
    bucket_summaries = _decision_statistic_bucket_summaries(
        observable_rows,
        exact_accepted=exact_accepted,
        exact_deferred=exact_deferred,
    )
    schema = _decision_statistic_schema_report(
        bucket_summaries=bucket_summaries,
        candidate_row_count=len(observable_rows),
        global_cap=int(trace_step.cap_spec.cap),
        strictest_required_row=strictest_required_row,
    )
    canonical_accepted, canonical_deferred, canonical_q_changed = (
        _reconstruct_decision_sets_from_bucket_counts(
            observable_rows,
            bucket_summaries,
            reverse_bucket_order=False,
        )
    )
    shuffled_accepted, shuffled_deferred, shuffled_q_changed = (
        _reconstruct_decision_sets_from_bucket_counts(
            observable_rows,
            bucket_summaries,
            reverse_bucket_order=True,
        )
    )
    canonical_matches_exact = (
        canonical_accepted == exact_accepted
        and canonical_deferred == exact_deferred
        and canonical_q_changed == exact_q_changed
    )
    shuffled_matches_exact = (
        shuffled_accepted == exact_accepted
        and shuffled_deferred == exact_deferred
        and shuffled_q_changed == exact_q_changed
    )
    shuffle_preserves_outcome = (
        canonical_accepted == shuffled_accepted
        and canonical_deferred == shuffled_deferred
        and canonical_q_changed == shuffled_q_changed
    )
    frontier_tie_bucket_count = sum(
        1 for bucket in bucket_summaries if bucket.frontier_tie_crosses_boundary
    )
    observable_rank_features_sufficient = bool(
        canonical_matches_exact and shuffled_matches_exact and shuffle_preserves_outcome
    )
    insufficiency_reason: str | None = None
    if not observable_rank_features_sufficient:
        if frontier_tie_bucket_count > 0 and not shuffle_preserves_outcome:
            insufficiency_reason = (
                "counts-only statistic depends on row-identity order inside decisive tied buckets"
            )
        elif not canonical_matches_exact:
            insufficiency_reason = (
                "counts-only statistic failed canonical exact accepted/deferred/q reconstruction"
            )
        elif not shuffled_matches_exact:
            insufficiency_reason = (
                "reverse-order tie falsifier changed accepted/deferred/q reconstruction"
            )
        else:
            insufficiency_reason = (
                "observable rank features were not sufficient under the declared falsifier"
            )
    return DecisionStatisticStepReport(
        schedule_name=trace_step.schedule_step.name,
        step=int(trace_step.schedule_step.step),
        global_cap=int(trace_step.cap_spec.cap),
        candidate_row_count=len(observable_rows),
        accepted_row_count=len(exact_accepted),
        deferred_row_count=len(exact_deferred),
        candidate_rows_fully_transient_observable=True,
        bucket_summaries=bucket_summaries,
        statistic_schema=schema,
        shuffle_falsifier=DECISION_STATISTIC_SHUFFLE_FALSIFIER,
        frontier_tie_bucket_count=int(frontier_tie_bucket_count),
        canonical_matches_exact=bool(canonical_matches_exact),
        shuffled_matches_exact=bool(shuffled_matches_exact),
        shuffle_preserves_outcome=bool(shuffle_preserves_outcome),
        observable_rank_features_sufficient=bool(observable_rank_features_sufficient),
        insufficiency_reason=insufficiency_reason,
        exact_accepted_identities_sha256=_identity_sha256(exact_accepted),
        canonical_accepted_identities_sha256=_identity_sha256(canonical_accepted),
        shuffled_accepted_identities_sha256=_identity_sha256(shuffled_accepted),
        exact_deferred_identities_sha256=_identity_sha256(exact_deferred),
        canonical_deferred_identities_sha256=_identity_sha256(canonical_deferred),
        shuffled_deferred_identities_sha256=_identity_sha256(shuffled_deferred),
        exact_q_changed_identities_sha256=_identity_sha256(exact_q_changed),
        canonical_q_changed_identities_sha256=_identity_sha256(canonical_q_changed),
        shuffled_q_changed_identities_sha256=_identity_sha256(shuffled_q_changed),
    )


def _decision_statistic_terminal_decision(
    step_reports: Sequence[DecisionStatisticStepReport],
    *,
    strictest_required_row: ScaleAppropriateLedgerComparisonReport,
) -> DecisionStatisticUpperBoundDecision:
    if not step_reports:
        raise ValueError("decision statistic diagnostic requires at least one step report")
    peak_step = max(
        step_reports,
        key=lambda step: step.statistic_schema.total_bits_per_weight_strictest_required_row,
    )
    budget_failures = [
        step.schedule_name
        for step in step_reports
        if not step.statistic_schema.fits_strictest_required_headroom
    ]
    insufficiencies = [
        step.schedule_name
        for step in step_reports
        if not step.observable_rank_features_sufficient
    ]
    if budget_failures:
        terminal_label = STATISTIC_BUDGET_BREAKS_SUB2
        reason = (
            "low-cardinality decision statistic exceeded the strictest Slice 1d headroom "
            f"first at {budget_failures[0]}"
        )
    elif insufficiencies:
        terminal_label = OBSERVABLE_RANK_FEATURES_INSUFFICIENT
        reason = (
            "counts-only bucket statistic fits the strictest Slice 1d headroom but fails "
            f"the reverse-order tie falsifier first at {insufficiencies[0]}; exact recovery "
            "depends on row-identity order inside decisive tied buckets"
        )
    else:
        terminal_label = DECISION_STATISTIC_UPPER_BOUND_PASS
        reason = (
            "counts-only bucket statistic fits the strictest Slice 1d headroom and survives "
            "the declared reverse-order tie falsifier on every preregistered step"
        )
    return DecisionStatisticUpperBoundDecision(
        terminal_label=terminal_label,
        strictest_required_q_regime_name=strictest_required_row.q_regime_name,
        strictest_required_headroom_bits_per_weight=float(
            strictest_required_row.scale_appropriate_headroom_bits_per_weight
        ),
        peak_statistic_bits_per_weight=float(
            peak_step.statistic_schema.total_bits_per_weight_strictest_required_row
        ),
        peak_statistic_step=peak_step.schedule_name,
        budget_fits_strictest_required_headroom=not budget_failures,
        inclusive_sub2_if_installed=not budget_failures,
        first_budget_failure_step=budget_failures[0] if budget_failures else None,
        first_insufficient_step=insufficiencies[0] if insufficiencies else None,
        any_step_frontier_tie_crosses_boundary=any(
            step.frontier_tie_bucket_count > 0 for step in step_reports
        ),
        all_steps_shuffle_preserve_outcome=all(
            step.shuffle_preserves_outcome for step in step_reports
        ),
        global_per_row_compression_closed=False,
        branch_a_trigger=False,
        reason=reason,
    )


def _decision_statistic_non_claims() -> tuple[str, ...]:
    return (
        "CPU-only oracle-derived upper bound over the preregistered exact cap trace",
        "persistent input is limited to low-cardinality bucket counts keyed by state/q/direction",
        "reverse-order tie falsifier is a hard anti-leak gate, not an optional sensitivity check",
        "PASS is an oracle-derived upper bound only, not a deployable online estimator",
        "global_per_row_compression_closed=false",
        "branch_a_trigger=false",
        "no dyn200, no GPU lane, no kernel path",
        "compact aggregate hashes only in validation output; no raw per-weight arrays",
    )


def run_decision_statistic_upper_bound_diagnostic() -> DecisionStatisticUpperBoundReport:
    """Test branch-(a) with a low-cardinality, identity-free decision statistic."""

    scale_report = run_scale_appropriate_b_storage_comparison()
    strictest_required_row = _strictest_required_scale_row(scale_report)
    trace_steps, _ = _build_exact_schedule_trace()
    step_reports = tuple(
        _decision_statistic_step_report(
            trace_step=trace_step,
            strictest_required_row=strictest_required_row,
        )
        for trace_step in trace_steps
    )
    bindingness = pre_register_source_bindingness(
        source_kind=SOURCE_KIND_GENERATED_NATIVE_LOOP,
        coverage=SourceFieldCoverage.full_generated_native_loop(),
    )
    return DecisionStatisticUpperBoundReport(
        schema_version=DECISION_STATISTIC_UPPER_BOUND_SCHEMA_VERSION,
        label=DECISION_STATISTIC_UPPER_BOUND_LABEL,
        source_bindingness=bindingness,
        field_coverage=SourceFieldCoverage.full_generated_native_loop(),
        candidate_name=VIRTUAL_DECISION_STATISTIC_CANDIDATE,
        source_scale_comparison_label=scale_report.label,
        source_scale_terminal_label=scale_report.terminal_decision.terminal_label,
        strictest_required_q_regime_name=strictest_required_row.q_regime_name,
        strictest_required_eligible_weight_count=int(
            strictest_required_row.eligible_weight_count
        ),
        strictest_required_headroom_bits_per_weight=float(
            strictest_required_row.scale_appropriate_headroom_bits_per_weight
        ),
        bucket_key_dimensions=DECISION_STATISTIC_BUCKET_KEY_DIMENSIONS,
        statistic_mode=DECISION_STATISTIC_COUNT_ONLY_MODE,
        shuffle_falsifier=DECISION_STATISTIC_SHUFFLE_FALSIFIER,
        step_reports=step_reports,
        terminal_decision=_decision_statistic_terminal_decision(
            step_reports,
            strictest_required_row=strictest_required_row,
        ),
        raw_arrays_included=False,
        non_claims=_decision_statistic_non_claims(),
    )


def _log2_choose_ceil(total_count: int, selected_count: int) -> int:
    total = int(total_count)
    selected = int(selected_count)
    if total <= 0 or selected < 0 or selected > total:
        raise ValueError("tie-frontier lower bound requires 0 <= A <= T and T > 0")
    if selected == 0 or selected == total:
        return 0
    value = (
        math.lgamma(total + 1)
        - math.lgamma(selected + 1)
        - math.lgamma(total - selected + 1)
    ) / math.log(2.0)
    return int(math.ceil(value - 1e-12))


def _practical_tie_encoding_choice(*, tie_group_size: int, accepted_count: int) -> tuple[str, int]:
    mask_bits = int(tie_group_size)
    offset_bits = int(accepted_count) * _count_bit_width(int(tie_group_size) - 1)
    if mask_bits <= offset_bits:
        return TIE_MEMBERSHIP_MASK_ENCODING, mask_bits
    return TIE_SELECTED_OFFSET_ENCODING, offset_bits


def _decision_statistic_projection_bits(
    *,
    source_step_report: DecisionStatisticStepReport,
    target_eligible_weight_count: int,
    source_eligible_weight_count: int,
) -> tuple[int, float, int, int, int]:
    target_candidate_row_count = _scale_count_with_density(
        source_count=int(source_step_report.candidate_row_count),
        source_eligible_weight_count=source_eligible_weight_count,
        target_eligible_weight_count=target_eligible_weight_count,
    )
    target_accepted_row_count = _scale_count_with_density(
        source_count=int(source_step_report.accepted_row_count),
        source_eligible_weight_count=source_eligible_weight_count,
        target_eligible_weight_count=target_eligible_weight_count,
    )
    target_deferred_row_count = _scale_count_with_density(
        source_count=int(source_step_report.deferred_row_count),
        source_eligible_weight_count=source_eligible_weight_count,
        target_eligible_weight_count=target_eligible_weight_count,
    )
    total_bits = (
        len(source_step_report.bucket_summaries)
        * (
            _decision_statistic_bucket_key_bit_width()
            + _count_bit_width(target_accepted_row_count)
            + _count_bit_width(target_candidate_row_count)
            + DECISION_STATISTIC_CUTOFF_BIT_WIDTH
        )
        + DECISION_STATISTIC_SEED_BITS
        + DECISION_STATISTIC_METADATA_BITS
    )
    bits_per_weight = float(total_bits) / float(int(target_eligible_weight_count))
    return (
        int(total_bits),
        float(bits_per_weight),
        int(target_candidate_row_count),
        int(target_accepted_row_count),
        int(target_deferred_row_count),
    )


def _tie_bucket_identity_sets(
    *,
    trace_step: _ExactScheduleTraceStep,
    bucket: DecisionStatisticBucketSummary,
) -> tuple[set[tuple[str, int]], set[tuple[str, int]], int, bool]:
    cap_result = trace_step.exact_path.cap_result
    if cap_result is None:
        raise ValueError("tie-frontier lower-bound diagnostic requires a cap-result trace")
    exact_accepted = {(row.state_key, int(row.flat_index)) for row in cap_result.accepted_rows}
    matching_rows = [
        row
        for row in _decision_statistic_observable_rows(trace_step)
        if row.state_key == bucket.state_key
        and int(row.current_q_level) == int(bucket.current_q_level)
        and int(row.move_direction) == int(bucket.move_direction)
    ]
    accepted_count = int(bucket.accepted_count)
    sorted_scores = sorted((row.abs_new_acc for row in matching_rows), reverse=True)
    boundary_score = int(sorted_scores[accepted_count - 1])
    tie_rows = {row.identity for row in matching_rows if int(row.abs_new_acc) == boundary_score}
    accepted_within_tie = tie_rows & exact_accepted
    plateau_covers_entire_bucket = len(tie_rows) == len(matching_rows)
    return tie_rows, accepted_within_tie, boundary_score, plateau_covers_entire_bucket


def _observed_tie_frontier_bucket_reports(
    *,
    source_step_reports: Sequence[DecisionStatisticStepReport],
    trace_steps: Sequence[_ExactScheduleTraceStep],
    source_eligible_weight_count: int,
) -> tuple[TieFrontierObservedBucketReport, ...]:
    trace_by_name = {step.schedule_step.name: step for step in trace_steps}
    reports: list[TieFrontierObservedBucketReport] = []
    for step_report in source_step_reports:
        if int(step_report.frontier_tie_bucket_count) <= 0:
            continue
        trace_step = trace_by_name[step_report.schedule_name]
        for bucket in step_report.bucket_summaries:
            if not bool(bucket.frontier_tie_crosses_boundary):
                continue
            tie_rows, accepted_within_tie, boundary_score, plateau_covers_entire_bucket = (
                _tie_bucket_identity_sets(trace_step=trace_step, bucket=bucket)
            )
            tie_group_size = len(tie_rows)
            exact_accepted_within_tie_count = len(accepted_within_tie)
            theoretical_lower_bound_bits = _log2_choose_ceil(
                tie_group_size,
                exact_accepted_within_tie_count,
            )
            decisive_label, decisive_bits = _practical_tie_encoding_choice(
                tie_group_size=tie_group_size,
                accepted_count=exact_accepted_within_tie_count,
            )
            reports.append(
                TieFrontierObservedBucketReport(
                    schedule_name=step_report.schedule_name,
                    step=int(step_report.step),
                    state_key=bucket.state_key,
                    current_q_level=int(bucket.current_q_level),
                    move_direction=int(bucket.move_direction),
                    candidate_row_count=int(bucket.candidate_row_count),
                    accepted_row_count=int(bucket.accepted_count),
                    boundary_abs_new_acc=int(boundary_score),
                    tie_group_size=int(tie_group_size),
                    exact_accepted_within_tie_count=int(exact_accepted_within_tie_count),
                    tie_group_density_per_eligible_weight=float(tie_group_size)
                    / float(source_eligible_weight_count),
                    accepted_within_tie_density_per_eligible_weight=float(
                        exact_accepted_within_tie_count
                    )
                    / float(source_eligible_weight_count),
                    theoretical_lower_bound_bits=int(theoretical_lower_bound_bits),
                    mask_bits=int(tie_group_size),
                    selected_offset_bits=int(exact_accepted_within_tie_count)
                    * _count_bit_width(int(tie_group_size) - 1),
                    decisive_practical_encoding_label=decisive_label,
                    decisive_practical_bits=int(decisive_bits),
                    plateau_covers_entire_bucket=bool(plateau_covers_entire_bucket),
                    exact_tie_members_sha256=_identity_sha256(tie_rows),
                    exact_tie_accepted_sha256=_identity_sha256(accepted_within_tie),
                )
            )
    return tuple(reports)


def _observed_tie_density_assumption(
    bucket_reports: Sequence[TieFrontierObservedBucketReport],
    *,
    source_eligible_weight_count: int,
) -> str:
    parts = []
    for bucket in bucket_reports:
        parts.append(
            f"{bucket.schedule_name}:{bucket.state_key}/q{int(bucket.current_q_level)}/d{int(bucket.move_direction)} "
            f"T={int(bucket.tie_group_size)}/{int(source_eligible_weight_count)} "
            f"A={int(bucket.exact_accepted_within_tie_count)}/{int(source_eligible_weight_count)}"
        )
    return "; ".join(parts)


def _joint_ta_scaling_model_defensible(
    bucket_reports: Sequence[TieFrontierObservedBucketReport],
) -> bool:
    return bool(bucket_reports) and all(
        bool(bucket.plateau_covers_entire_bucket) for bucket in bucket_reports
    )


def _project_tie_reservation_bucket(
    *,
    source_bucket: TieFrontierObservedBucketReport,
    source_eligible_weight_count: int,
    target_eligible_weight_count: int,
    projection_label: str,
    scaling_model_defensible: bool,
) -> TieReservationProjectionBucketReport:
    if projection_label == OBSERVED_TIE_RESERVATION_DIAGNOSTIC:
        target_tie_group_size = int(source_bucket.tie_group_size)
        target_exact_accepted_within_tie_count = int(
            source_bucket.exact_accepted_within_tie_count
        )
        scaling_model = "hold_observed_TA_fixed_absolute_count_diagnostic_only"
    elif projection_label == RATE_HELD_TIE_RESERVATION_DIAGNOSTIC:
        target_tie_group_size = _scale_count_with_density(
            source_count=int(source_bucket.tie_group_size),
            source_eligible_weight_count=source_eligible_weight_count,
            target_eligible_weight_count=target_eligible_weight_count,
        )
        target_exact_accepted_within_tie_count = _scale_count_with_density(
            source_count=int(source_bucket.exact_accepted_within_tie_count),
            source_eligible_weight_count=source_eligible_weight_count,
            target_eligible_weight_count=target_eligible_weight_count,
        )
        scaling_model = FULL_PLATEAU_JOINT_TA_SCALING_MODEL
    else:
        raise ValueError(f"unsupported tie-reservation projection label {projection_label!r}")
    target_exact_accepted_within_tie_count = min(
        int(target_tie_group_size),
        int(target_exact_accepted_within_tie_count),
    )
    theoretical_lower_bound_bits = _log2_choose_ceil(
        target_tie_group_size,
        target_exact_accepted_within_tie_count,
    )
    decisive_label, decisive_bits = _practical_tie_encoding_choice(
        tie_group_size=target_tie_group_size,
        accepted_count=target_exact_accepted_within_tie_count,
    )
    return TieReservationProjectionBucketReport(
        schedule_name=source_bucket.schedule_name,
        step=int(source_bucket.step),
        state_key=source_bucket.state_key,
        current_q_level=int(source_bucket.current_q_level),
        move_direction=int(source_bucket.move_direction),
        source_tie_group_size=int(source_bucket.tie_group_size),
        source_exact_accepted_within_tie_count=int(
            source_bucket.exact_accepted_within_tie_count
        ),
        source_tie_group_density_per_eligible_weight=float(
            source_bucket.tie_group_density_per_eligible_weight
        ),
        source_accepted_within_tie_density_per_eligible_weight=float(
            source_bucket.accepted_within_tie_density_per_eligible_weight
        ),
        target_tie_group_size=int(target_tie_group_size),
        target_exact_accepted_within_tie_count=int(
            target_exact_accepted_within_tie_count
        ),
        tie_group_density_per_eligible_weight=float(target_tie_group_size)
        / float(target_eligible_weight_count),
        accepted_within_tie_density_per_eligible_weight=float(
            target_exact_accepted_within_tie_count
        )
        / float(target_eligible_weight_count),
        theoretical_lower_bound_bits=int(theoretical_lower_bound_bits),
        mask_bits=int(target_tie_group_size),
        selected_offset_bits=int(target_exact_accepted_within_tie_count)
        * _count_bit_width(int(target_tie_group_size) - 1),
        decisive_practical_encoding_label=decisive_label,
        decisive_practical_bits=int(decisive_bits),
        joint_ta_scaling_model=scaling_model,
        scaling_model_defensible=bool(scaling_model_defensible),
    )


def _projection_decisive_label(
    bucket_reports: Sequence[TieReservationProjectionBucketReport],
) -> str:
    labels = {bucket.decisive_practical_encoding_label for bucket in bucket_reports}
    if len(labels) == 1:
        return next(iter(labels))
    return "mixed_per_bucket_min_practical_exact_retention"


def _tie_reservation_step_projection_report(
    *,
    source_step_report: DecisionStatisticStepReport,
    source_bucket_reports: Sequence[TieFrontierObservedBucketReport],
    source_eligible_weight_count: int,
    q_ledger_row: Base3QEntropyLedgerRow,
    strictest_headroom_bits_per_weight: float,
    projection_label: str,
    scaling_model_defensible: bool,
) -> TieReservationStepProjectionReport:
    bucket_reports = tuple(
        _project_tie_reservation_bucket(
            source_bucket=bucket,
            source_eligible_weight_count=source_eligible_weight_count,
            target_eligible_weight_count=int(q_ledger_row.eligible_weight_count),
            projection_label=projection_label,
            scaling_model_defensible=scaling_model_defensible,
        )
        for bucket in source_bucket_reports
    )
    decision_statistic_total_bits, decision_statistic_bits_per_weight, target_candidate_row_count, target_accepted_row_count, target_deferred_row_count = (
        _decision_statistic_projection_bits(
            source_step_report=source_step_report,
            target_eligible_weight_count=int(q_ledger_row.eligible_weight_count),
            source_eligible_weight_count=source_eligible_weight_count,
        )
    )
    theoretical_total_bits = sum(bucket.theoretical_lower_bound_bits for bucket in bucket_reports)
    mask_total_bits = sum(bucket.mask_bits for bucket in bucket_reports)
    selected_offset_total_bits = sum(bucket.selected_offset_bits for bucket in bucket_reports)
    decisive_total_bits = sum(bucket.decisive_practical_bits for bucket in bucket_reports)
    target_eligible = int(q_ledger_row.eligible_weight_count)
    return TieReservationStepProjectionReport(
        schedule_name=source_step_report.schedule_name,
        step=int(source_step_report.step),
        projection_label=projection_label,
        target_q_regime_name=q_ledger_row.regime_name,
        source_eligible_weight_count=int(source_eligible_weight_count),
        target_eligible_weight_count=target_eligible,
        source_candidate_row_count=int(source_step_report.candidate_row_count),
        source_accepted_row_count=int(source_step_report.accepted_row_count),
        source_deferred_row_count=int(source_step_report.deferred_row_count),
        target_candidate_row_count=int(target_candidate_row_count),
        target_accepted_row_count=int(target_accepted_row_count),
        target_deferred_row_count=int(target_deferred_row_count),
        decision_statistic_total_bits=int(decision_statistic_total_bits),
        decision_statistic_bits_per_weight=float(decision_statistic_bits_per_weight),
        bucket_reports=bucket_reports,
        theoretical_lower_bound_total_bits=int(theoretical_total_bits),
        theoretical_lower_bound_bits_per_weight=float(theoretical_total_bits)
        / float(target_eligible),
        mask_total_bits=int(mask_total_bits),
        mask_bits_per_weight=float(mask_total_bits) / float(target_eligible),
        selected_offset_total_bits=int(selected_offset_total_bits),
        selected_offset_bits_per_weight=float(selected_offset_total_bits)
        / float(target_eligible),
        decisive_practical_encoding_label=_projection_decisive_label(bucket_reports),
        decisive_tie_reservation_total_bits=int(decisive_total_bits),
        decisive_tie_reservation_bits_per_weight=float(decisive_total_bits)
        / float(target_eligible),
        combined_decisive_bits_per_weight=float(decision_statistic_bits_per_weight)
        + float(decisive_total_bits) / float(target_eligible),
        strictest_headroom_bits_per_weight=float(strictest_headroom_bits_per_weight),
        fits_strictest_headroom=(
            float(decision_statistic_bits_per_weight)
            + float(decisive_total_bits) / float(target_eligible)
            <= float(strictest_headroom_bits_per_weight) + 1e-12
        ),
        diagnostic_only=projection_label == OBSERVED_TIE_RESERVATION_DIAGNOSTIC,
        joint_ta_scaling_model=(
            "hold_observed_TA_fixed_absolute_count_diagnostic_only"
            if projection_label == OBSERVED_TIE_RESERVATION_DIAGNOSTIC
            else FULL_PLATEAU_JOINT_TA_SCALING_MODEL
        ),
        scaling_model_defensible=bool(scaling_model_defensible),
    )


def _failing_step_bucket_groups(
    observed_bucket_reports: Sequence[TieFrontierObservedBucketReport],
) -> dict[str, tuple[TieFrontierObservedBucketReport, ...]]:
    grouped: dict[str, list[TieFrontierObservedBucketReport]] = {}
    for bucket in observed_bucket_reports:
        grouped.setdefault(bucket.schedule_name, []).append(bucket)
    return {
        name: tuple(sorted(reports, key=lambda item: (item.state_key, item.move_direction)))
        for name, reports in grouped.items()
    }


def _tie_reservation_row_comparison_for_row(
    *,
    q_ledger_row: Base3QEntropyLedgerRow,
    row_role: str,
    source_step_reports: Sequence[DecisionStatisticStepReport],
    observed_bucket_reports: Sequence[TieFrontierObservedBucketReport],
    source_eligible_weight_count: int,
    strictest_headroom_bits_per_weight: float,
) -> TieReservationRowComparisonReport:
    bucket_groups = _failing_step_bucket_groups(observed_bucket_reports)
    scaling_model_defensible = _joint_ta_scaling_model_defensible(observed_bucket_reports)
    absolute_reports = tuple(
        _tie_reservation_step_projection_report(
            source_step_report=step,
            source_bucket_reports=bucket_groups[step.schedule_name],
            source_eligible_weight_count=source_eligible_weight_count,
            q_ledger_row=q_ledger_row,
            strictest_headroom_bits_per_weight=strictest_headroom_bits_per_weight,
            projection_label=OBSERVED_TIE_RESERVATION_DIAGNOSTIC,
            scaling_model_defensible=True,
        )
        for step in source_step_reports
        if step.schedule_name in bucket_groups
    )
    rate_held_reports = tuple(
        _tie_reservation_step_projection_report(
            source_step_report=step,
            source_bucket_reports=bucket_groups[step.schedule_name],
            source_eligible_weight_count=source_eligible_weight_count,
            q_ledger_row=q_ledger_row,
            strictest_headroom_bits_per_weight=strictest_headroom_bits_per_weight,
            projection_label=RATE_HELD_TIE_RESERVATION_DIAGNOSTIC,
            scaling_model_defensible=scaling_model_defensible,
        )
        for step in source_step_reports
        if step.schedule_name in bucket_groups
    )
    absolute_peak = max(
        step.combined_decisive_bits_per_weight for step in absolute_reports
    )
    rate_peak = max(step.combined_decisive_bits_per_weight for step in rate_held_reports)
    return TieReservationRowComparisonReport(
        q_regime_name=q_ledger_row.regime_name,
        row_role=row_role,
        eligible_weight_count=int(q_ledger_row.eligible_weight_count),
        row_headroom_bits_per_weight=float(
            q_ledger_row.remaining_accumulator_budget_bits_per_weight
        ),
        strictest_headroom_bits_per_weight=float(strictest_headroom_bits_per_weight),
        observed_tie_density_assumption=_observed_tie_density_assumption(
            observed_bucket_reports,
            source_eligible_weight_count=source_eligible_weight_count,
        ),
        joint_ta_scaling_model=FULL_PLATEAU_JOINT_TA_SCALING_MODEL,
        joint_ta_scaling_model_defensible=bool(scaling_model_defensible),
        absolute_count_step_reports=absolute_reports,
        rate_held_step_reports=rate_held_reports,
        absolute_count_peak_combined_bits_per_weight=float(absolute_peak),
        rate_held_peak_combined_bits_per_weight=float(rate_peak),
        rate_held_fits_strictest_headroom=bool(
            rate_peak <= float(strictest_headroom_bits_per_weight) + 1e-12
        ),
    )


def _tie_frontier_reservation_decision(
    row_comparisons: Sequence[TieReservationRowComparisonReport],
    *,
    strictest_required_row: ScaleAppropriateLedgerComparisonReport,
) -> TieFrontierReservationDecision:
    required_rows = [row for row in row_comparisons if row.row_role == "required_gate"]
    if not required_rows:
        raise ValueError("tie-frontier lower-bound diagnostic requires the Slice 1d required rows")
    any_ambiguous = any(not row.joint_ta_scaling_model_defensible for row in required_rows)
    all_fit = all(row.rate_held_fits_strictest_headroom for row in required_rows)
    peak_row = max(
        required_rows,
        key=lambda row: row.rate_held_peak_combined_bits_per_weight,
    )
    peak_step = max(
        peak_row.rate_held_step_reports,
        key=lambda step: step.combined_decisive_bits_per_weight,
    )
    if any_ambiguous:
        terminal_label = TIE_DENSITY_AMBIGUOUS_NEEDS_TRACE
        reason = (
            "no defensible joint T/A scaling model was available for every required row; "
            "honest terminal is ambiguity rather than fixed tiny-multiplicity fit"
        )
    elif not all_fit:
        terminal_label = TIE_RESERVATION_BREAKS_SUB2
        reason = (
            "decision-statistic bpw plus the decisive practical exact-tie reservation bpw "
            f"exceeded the strictest Slice 1d headroom first at {peak_row.q_regime_name}/"
            f"{peak_step.schedule_name}"
        )
    else:
        terminal_label = TIE_RESERVATION_FITS_HEADROOM_CANDIDATE_HYBRID
        reason = (
            "under the full-plateau joint T/A rate-held model, the decisive practical exact-tie "
            "reservation plus the branch-(a) decision statistic stays below the strictest Slice 1d "
            "headroom on every required row; hybrid survives as a candidate only"
        )
    return TieFrontierReservationDecision(
        terminal_label=terminal_label,
        required_rows=tuple(SCALE_REQUIRED_Q_LEDGER_ROWS),
        strictest_required_q_regime_name=strictest_required_row.q_regime_name,
        strictest_headroom_bits_per_weight=float(
            strictest_required_row.scale_appropriate_headroom_bits_per_weight
        ),
        joint_ta_scaling_model=FULL_PLATEAU_JOINT_TA_SCALING_MODEL,
        joint_ta_scaling_model_defensible=not any_ambiguous,
        peak_rate_held_combined_bits_per_weight=float(
            peak_row.rate_held_peak_combined_bits_per_weight
        ),
        peak_rate_held_step=peak_step.schedule_name,
        peak_rate_held_q_regime_name=peak_row.q_regime_name,
        peak_rate_held_encoding_label=peak_step.decisive_practical_encoding_label,
        theoretical_lower_bound_non_decisive=True,
        required_rows_all_rate_held_fit_strictest_headroom=bool(all_fit),
        any_required_row_joint_ta_ambiguous=bool(any_ambiguous),
        candidate_hybrid_alive=(
            terminal_label == TIE_RESERVATION_FITS_HEADROOM_CANDIDATE_HYBRID
        ),
        global_per_row_compression_closed=False,
        branch_a_trigger=False,
        reason=reason,
    )


def _tie_frontier_reservation_non_claims() -> tuple[str, ...]:
    return (
        "CPU-only lower-bound diagnostic built on the committed branch-(a) exact trace",
        "theoretical lower bound is non-decisive unless a recoverable enumerative codec is separately validated",
        "practical exact-retention encodings are mask and selected-offset list only",
        "selected offsets are recomputed from current transient observable rank order, not persisted identity state",
        "candidate_hybrid only; no dyn200, no online-estimability claim, no global closure",
        "global_per_row_compression_closed=false",
        "branch_a_trigger=false",
        "compact validation hashes only; no row IDs or ordered IDs in the persisted payload",
    )


def run_tie_frontier_reservation_lower_bound_diagnostic() -> TieFrontierReservationLowerBoundReport:
    """Measure the honest lower bound for exact frontier-tie retention on top of branch-(a)."""

    scale_report = run_scale_appropriate_b_storage_comparison()
    strictest_required_row = _strictest_required_scale_row(scale_report)
    decision_report = run_decision_statistic_upper_bound_diagnostic()
    trace_steps, _ = _build_exact_schedule_trace()
    source_step_reports = tuple(
        step
        for step in decision_report.step_reports
        if int(step.frontier_tie_bucket_count) > 0
    )
    observed_bucket_reports = _observed_tie_frontier_bucket_reports(
        source_step_reports=source_step_reports,
        trace_steps=trace_steps,
        source_eligible_weight_count=int(
            decision_report.strictest_required_eligible_weight_count
        ),
    )
    bindingness = pre_register_source_bindingness(
        source_kind=SOURCE_KIND_GENERATED_NATIVE_LOOP,
        coverage=SourceFieldCoverage.full_generated_native_loop(),
    )
    row_comparisons = tuple(
        [
            _tie_reservation_row_comparison_for_row(
                q_ledger_row=_q_ledger_row_by_name(regime_name),
                row_role="required_gate",
                source_step_reports=source_step_reports,
                observed_bucket_reports=observed_bucket_reports,
                source_eligible_weight_count=int(
                    decision_report.strictest_required_eligible_weight_count
                ),
                strictest_headroom_bits_per_weight=float(
                    strictest_required_row.scale_appropriate_headroom_bits_per_weight
                ),
            )
            for regime_name in SCALE_REQUIRED_Q_LEDGER_ROWS
        ]
        + [
            _tie_reservation_row_comparison_for_row(
                q_ledger_row=_q_ledger_row_by_name(regime_name),
                row_role="sensitivity_only",
                source_step_reports=source_step_reports,
                observed_bucket_reports=observed_bucket_reports,
                source_eligible_weight_count=int(
                    decision_report.strictest_required_eligible_weight_count
                ),
                strictest_headroom_bits_per_weight=float(
                    strictest_required_row.scale_appropriate_headroom_bits_per_weight
                ),
            )
            for regime_name in SCALE_SENSITIVITY_Q_LEDGER_ROWS
        ]
    )
    return TieFrontierReservationLowerBoundReport(
        schema_version=TIE_FRONTIER_RESERVATION_SCHEMA_VERSION,
        label=TIE_FRONTIER_RESERVATION_LABEL,
        source_bindingness=bindingness,
        field_coverage=SourceFieldCoverage.full_generated_native_loop(),
        candidate_name=TIE_FRONTIER_RESERVATION_CANDIDATE,
        source_decision_statistic_label=decision_report.label,
        source_decision_statistic_terminal_label=decision_report.terminal_decision.terminal_label,
        strictest_required_q_regime_name=strictest_required_row.q_regime_name,
        strictest_headroom_bits_per_weight=float(
            strictest_required_row.scale_appropriate_headroom_bits_per_weight
        ),
        source_eligible_weight_count=int(
            decision_report.strictest_required_eligible_weight_count
        ),
        required_q_ledger_rows=SCALE_REQUIRED_Q_LEDGER_ROWS,
        sensitivity_q_ledger_rows=SCALE_SENSITIVITY_Q_LEDGER_ROWS,
        observed_failing_bucket_reports=observed_bucket_reports,
        row_comparisons=row_comparisons,
        terminal_decision=_tie_frontier_reservation_decision(
            row_comparisons,
            strictest_required_row=strictest_required_row,
        ),
        raw_arrays_included=False,
        non_claims=_tie_frontier_reservation_non_claims(),
    )


def _assert_no_tensors(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        raise ValueError("representative verdict payload must not include raw tensors")
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_tensors(child)


def validate_representative_bounded_delta_drift_verdict_report(
    report: RepresentativeDriftVerdictReport,
) -> None:
    if report.schema_version != REPRESENTATIVE_VERDICT_SCHEMA_VERSION:
        raise ValueError("unexpected representative verdict schema version")
    if report.terminal_science_question_closed and report.terminal_mode != CUMULATIVE_SCHEDULE_MODE:
        raise ValueError("C1.1c science closure requires cumulative mode")
    if not report.cumulative_curve_reports:
        raise ValueError("representative verdict requires cumulative drift curve reports")
    if not any(run.curve_label == report.primary_curve_label for run in report.cumulative_curve_reports):
        raise ValueError("primary curve label must name one cumulative curve report")
    for run in report.cumulative_curve_reports:
        if run.mode != CUMULATIVE_SCHEDULE_MODE:
            raise ValueError("cumulative curve reports must use cumulative mode")
        if len(run.per_step_reports) != len(PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE):
            raise ValueError("each cumulative run must cover the full pre-registered schedule")
        for step_report in run.per_step_reports:
            if step_report.bounded_reinitialized_from_exact:
                raise ValueError("cumulative bounded path must not reinitialize from exact per step")
            parity = step_report.bounded_delta_report.measured_report.oracle_parity
            if parity.get("bounded_reinitialized_from_exact") is not False:
                raise ValueError("oracle parity must record no bounded reinitialization")
            if not bool(parity.get("cumulative_carry_forward")):
                raise ValueError("oracle parity must record cumulative carry-forward")
            if (
                step_report.bounded_delta_report.storage_projection.backlog_entry_count
                != step_report.bounded_stored_backlog_entry_count
            ):
                raise ValueError("ledger projection must charge actual bounded stored backlog")
    _assert_no_tensors(report.to_dict())


def validate_candidate_admission_diagnostic_report(
    report: CandidateAdmissionDiagnosticReport,
) -> None:
    if report.schema_version != CANDIDATE_ADMISSION_DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError("unexpected candidate admission diagnostic schema version")
    if report.null_baseline_label != ACCUMULATOR_FREE_NULL_BASELINE:
        raise ValueError("candidate admission diagnostic must name the accumulator-free null baseline")
    if len(report.pre_registered_schedule) != len(PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE):
        raise ValueError("candidate admission diagnostic must cover the full pre-registered schedule")
    if len(report.candidate_runs) != 3:
        raise ValueError("candidate admission diagnostic must report the three preregistered candidates")
    for run in report.candidate_runs:
        if run.builder_label != ORACLE_UPPER_BOUND_ADMISSION_DIAGNOSTIC:
            raise ValueError("candidate admission diagnostic must stay oracle-upper-bound labeled")
        if len(run.per_step_reports) != len(PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE):
            raise ValueError("each candidate run must cover the full pre-registered schedule")
        for step in run.per_step_reports:
            if step.builder_label != ORACLE_UPPER_BOUND_ADMISSION_DIAGNOSTIC:
                raise ValueError("step report builder label drifted from the preregistered oracle-upper-bound tag")
            parity = step.bounded_delta_report.measured_report.oracle_parity
            if parity.get("builder_label") != ORACLE_UPPER_BOUND_ADMISSION_DIAGNOSTIC:
                raise ValueError("oracle parity must record the oracle-upper-bound builder label")
            if not bool(parity.get("cumulative_carry_forward")):
                raise ValueError("candidate admission diagnostic must record cumulative carry-forward")
    _assert_no_tensors(report.to_dict())


def validate_candidate_capacity_localization_report(
    report: CandidateCapacityLocalizationReport,
) -> None:
    if report.schema_version != CAPACITY_LOCALIZATION_DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError("unexpected candidate capacity localization schema version")
    if report.candidate_a_budget_report.candidate_name != HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE:
        raise ValueError("capacity localization must carry the A budget report")
    if report.candidate_a_budget_report.terminal_budget_direction_label not in {
        A_COLD_EXCEPTION_BUDGET_LEVER_LABEL,
        A_FUNDAMENTALLY_OVER_LABEL,
    }:
        raise ValueError("unexpected A terminal budget-direction label")
    if not report.backlog_k_schedule or report.backlog_k_schedule[-1] != "unbounded":
        raise ValueError("capacity localization backlog schedule must end with unbounded")
    if len(report.sweep_runs) != 2:
        raise ValueError("capacity localization must include exactly the B and C sweeps")
    for run in report.sweep_runs:
        if run.terminal_decision.status not in {
            K_SWEEP_MINIMAL_VIABLE_PASS,
            K_SWEEP_JOINT_INFEASIBLE,
            K_SWEEP_REPRESENTATION_WALL,
        }:
            raise ValueError("unexpected k-sweep terminal decision")
        if len(run.sweep_entries) != len(report.backlog_k_schedule):
            raise ValueError("each k-sweep run must cover the preregistered k schedule")
        for entry in run.sweep_entries:
            if len(entry.per_step_reports) != len(PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE):
                raise ValueError("each k entry must cover the full pre-registered schedule")
            for step in entry.per_step_reports:
                parity = step.bounded_delta_report.measured_report.oracle_parity
                if parity.get("builder_label") != ORACLE_UPPER_BOUND_ADMISSION_DIAGNOSTIC:
                    raise ValueError("capacity localization must preserve the oracle-upper-bound builder label")
                if not bool(parity.get("cumulative_carry_forward")):
                    raise ValueError("capacity localization must preserve cumulative carry-forward")
    _assert_no_tensors(report.to_dict())


def validate_real_backlog_lower_bound_diagnostic_report(
    report: RealBacklogLowerBoundReport,
) -> None:
    if report.schema_version != REAL_BACKLOG_LOWER_BOUND_SCHEMA_VERSION:
        raise ValueError("unexpected real backlog lower-bound schema version")
    if report.candidate_name != EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE:
        raise ValueError("real backlog lower-bound must stay B-only")
    if report.terminal_decision.terminal_label not in {
        PER_ROW_COMPRESSION_CLOSED_BY_EASY_CASE_LOWER_BOUND,
        SPARSE_AMORTIZED_CANDIDATE_RESURRECTED_FOR_HARDER_TRACE,
        REPRESENTATIVE_TRACE_UNDERPOWERED_FOR_CLOSURE,
    }:
        raise ValueError("unexpected real backlog lower-bound terminal label")
    if report.terminal_decision.headroom_source != TINY_FIXTURE_HEADROOM_SOURCE:
        raise ValueError("real backlog lower-bound must declare the tiny-fixture headroom source")
    if bool(report.terminal_decision.global_per_row_compression_closed):
        raise ValueError("tiny-fixture lower-bound must not claim global per-row compression closure")
    if bool(report.terminal_decision.branch_a_trigger):
        raise ValueError("tiny-fixture lower-bound must not trigger branch (a)")
    recomputed_headroom = (
        2.0
        - float(report.terminal_decision.q_packed_total_bits_per_weight)
        - float(report.terminal_decision.frozen_scale_fp32_bits_per_weight)
    )
    if not math.isclose(
        float(report.terminal_decision.actual_remaining_accumulator_headroom_bits_per_weight),
        recomputed_headroom,
        abs_tol=1e-12,
    ):
        raise ValueError("real backlog lower-bound headroom must be 2.0 - q_total - scale")
    if report.exact_trace_summary.stop_reason not in {
        LOWER_BOUND_TRACE_STOP_NONTRIVIAL,
        LOWER_BOUND_TRACE_STOP_PLATEAU,
        LOWER_BOUND_TRACE_STOP_CPU_SECONDS,
        LOWER_BOUND_TRACE_STOP_MAX_STEPS,
    }:
        raise ValueError("unexpected lower-bound trace stop reason")
    if not report.exact_trace_summary.per_step_reports:
        raise ValueError("real backlog lower-bound requires at least one trace step")
    if not report.backlog_k_schedule or report.backlog_k_schedule[-1] != "unbounded":
        raise ValueError("real backlog lower-bound backlog schedule must end with unbounded")
    if len(report.sweep_entries) != len(report.backlog_k_schedule):
        raise ValueError("real backlog lower-bound must sweep the declared k schedule")
    if (
        report.terminal_decision.minimal_surface_faithful_k_label is not None
        and report.terminal_decision.minimal_surface_faithful_k_label
        not in report.backlog_k_schedule
    ):
        raise ValueError("minimal surface-faithful k must come from the declared schedule")
    if report.terminal_decision.terminal_label == SPARSE_AMORTIZED_CANDIDATE_RESURRECTED_FOR_HARDER_TRACE:
        if not report.exact_trace_summary.nontrivial_backlog_reached:
            raise ValueError("resurrection requires a nontrivial backlog trace")
        if not report.terminal_decision.minimal_surface_faithful_k_fits_headroom:
            raise ValueError("resurrection requires the minimal surface-faithful k to fit headroom")
    if report.terminal_decision.terminal_label == PER_ROW_COMPRESSION_CLOSED_BY_EASY_CASE_LOWER_BOUND:
        if report.terminal_decision.minimal_surface_faithful_k_fits_headroom:
            raise ValueError("easy-case closure must not claim the minimal k fits headroom")
    for entry, expected_k_label in zip(report.sweep_entries, report.backlog_k_schedule):
        if entry.k_label != expected_k_label:
            raise ValueError("sweep entries must preserve backlog schedule order")
        if len(entry.per_step_reports) != len(report.exact_trace_summary.per_step_reports):
            raise ValueError("each lower-bound sweep entry must cover the traced steps")
        for step in entry.per_step_reports:
            parity = step.measured_report.oracle_parity
            if parity.get("builder_label") != REAL_BACKLOG_LOWER_BOUND_LABEL:
                raise ValueError("lower-bound sweep must preserve the lower-bound builder label")
            if not bool(parity.get("cumulative_carry_forward")):
                raise ValueError("lower-bound sweep must preserve cumulative carry-forward")
    _assert_no_tensors(report.to_dict())


def validate_scale_appropriate_b_storage_comparison_report(
    report: ScaleAppropriateBStorageComparisonReport,
) -> None:
    if report.schema_version != SCALE_APPROPRIATE_B_STORAGE_SCHEMA_VERSION:
        raise ValueError("unexpected scale-appropriate B comparison schema version")
    if report.candidate_name != EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE:
        raise ValueError("scale-appropriate B comparison must stay B-only")
    if tuple(report.required_q_ledger_rows) != SCALE_REQUIRED_Q_LEDGER_ROWS:
        raise ValueError("required q-ledger rows drifted from the gated Slice 1d set")
    if tuple(report.sensitivity_q_ledger_rows) != SCALE_SENSITIVITY_Q_LEDGER_ROWS:
        raise ValueError("sensitivity q-ledger rows drifted from the gated Slice 1d set")
    if report.terminal_decision.terminal_label not in {
        RATE_HELD_B_STILL_OVER_SCALE_HEADROOM_CANDIDATE_BRANCH_A,
        SCALE_APPROPRIATE_COMPARISON_AMBIGUOUS_NEEDS_BACKLOG_DENSITY_TRACE,
    }:
        raise ValueError("unexpected scale-appropriate B comparison terminal label")
    if bool(report.terminal_decision.global_per_row_compression_closed):
        raise ValueError("scale-appropriate B comparison must not claim global closure")
    if bool(report.terminal_decision.branch_a_trigger):
        raise ValueError("scale-appropriate B comparison must not self-trigger branch routing")
    required_rows = {
        row.q_regime_name: row for row in report.row_comparisons if row.row_role == "required_gate"
    }
    if set(required_rows) != set(SCALE_REQUIRED_Q_LEDGER_ROWS):
        raise ValueError("scale-appropriate B comparison must cover the required gate rows exactly")
    for row in report.row_comparisons:
        if not row.absolute_count_lower_bound_step_reports:
            raise ValueError("each scale row must keep the absolute-count diagnostic path")
        if not row.rate_held_b_storage_step_reports:
            raise ValueError("each scale row must keep the rate-held decisive path")
        if row.absolute_count_lower_bound_step_reports[0].projection_label != ABSOLUTE_COUNT_LOWER_BOUND_DIAGNOSTIC:
            raise ValueError("absolute-count diagnostic label drifted")
        if row.rate_held_b_storage_step_reports[0].projection_label != RATE_HELD_B_STORAGE_DIAGNOSTIC:
            raise ValueError("rate-held diagnostic label drifted")
        if any(step.decisive_for_branch for step in row.absolute_count_lower_bound_step_reports):
            raise ValueError("absolute-count diagnostics must stay non-decisive")
        if not all(step.decisive_for_branch for step in row.rate_held_b_storage_step_reports):
            raise ValueError("rate-held rows must stay the decisive comparator")
        if not all(
            step.rounding_policy == RATE_HELD_COUNT_ROUNDING_POLICY
            for step in row.rate_held_b_storage_step_reports
        ):
            raise ValueError("rate-held rows must declare the gated density rounding policy")
        if row.rate_held_b_storage_peak_bounded_delta_acc_bits_per_weight < row.absolute_count_lower_bound_peak_bounded_delta_acc_bits_per_weight:
            raise ValueError("rate-held peak bpw must not undercut the absolute-count lower bound")
    if report.terminal_decision.candidate_branch_a_trigger_earned:
        if not report.terminal_decision.required_rows_all_rate_held_exceed_scale_headroom:
            raise ValueError("candidate branch-a trigger requires all required rows to exceed headroom")
        if report.terminal_decision.terminal_label != RATE_HELD_B_STILL_OVER_SCALE_HEADROOM_CANDIDATE_BRANCH_A:
            raise ValueError("candidate branch-a trigger must carry the rate-held over-headroom label")
    else:
        if report.terminal_decision.terminal_label != SCALE_APPROPRIATE_COMPARISON_AMBIGUOUS_NEEDS_BACKLOG_DENSITY_TRACE:
            raise ValueError("non-triggered scale comparison must land on the ambiguity label")
    _assert_no_tensors(report.to_dict())


def _validate_decision_statistic_statistic_input(
    step_report: DecisionStatisticStepReport,
) -> None:
    allowed_schema_keys = {
        "bucket_key_dimensions",
        "bucket_cardinality_bound",
        "observed_bucket_count",
        "bucket_key_bit_width",
        "accepted_count_bit_width",
        "deferred_count_bit_width",
        "cutoff_bit_width",
        "seed_bits",
        "metadata_bits",
        "total_bits",
        "strictest_required_q_regime_name",
        "strictest_required_eligible_weight_count",
        "strictest_required_headroom_bits_per_weight",
        "total_bits_per_weight_strictest_required_row",
        "fits_strictest_required_headroom",
        "inclusive_sub2_if_installed",
        "statistic_mode",
    }
    schema_payload = step_report.statistic_schema.to_dict()
    if set(schema_payload) != allowed_schema_keys:
        raise ValueError("decision statistic schema payload drifted from the preregistered fields")
    allowed_bucket_keys = {
        "state_key",
        "current_q_level",
        "move_direction",
        "accepted_count",
        "deferred_count",
    }
    for bucket in step_report.bucket_summaries:
        payload = bucket.statistic_input_dict()
        if set(payload) != allowed_bucket_keys:
            raise ValueError("decision statistic bucket payload drifted from the aggregate-only field set")


def validate_decision_statistic_upper_bound_report(
    report: DecisionStatisticUpperBoundReport,
) -> None:
    if report.schema_version != DECISION_STATISTIC_UPPER_BOUND_SCHEMA_VERSION:
        raise ValueError("unexpected decision statistic upper-bound schema version")
    if report.label != DECISION_STATISTIC_UPPER_BOUND_LABEL:
        raise ValueError("unexpected decision statistic upper-bound label")
    if report.candidate_name != VIRTUAL_DECISION_STATISTIC_CANDIDATE:
        raise ValueError("decision statistic diagnostic must stay on the branch-(a) virtual candidate")
    if report.source_scale_comparison_label != SCALE_APPROPRIATE_B_STORAGE_LABEL:
        raise ValueError("decision statistic diagnostic must cite the committed Slice 1d source label")
    if report.source_scale_terminal_label != RATE_HELD_B_STILL_OVER_SCALE_HEADROOM_CANDIDATE_BRANCH_A:
        raise ValueError("decision statistic diagnostic must inherit the Slice 1d branch-(a) source trigger")
    if tuple(report.bucket_key_dimensions) != DECISION_STATISTIC_BUCKET_KEY_DIMENSIONS:
        raise ValueError("decision statistic bucket keys drifted from the low-cardinality contract")
    if report.statistic_mode != DECISION_STATISTIC_COUNT_ONLY_MODE:
        raise ValueError("decision statistic mode drifted from the counts-only contract")
    if report.shuffle_falsifier != DECISION_STATISTIC_SHUFFLE_FALSIFIER:
        raise ValueError("decision statistic shuffle falsifier drifted from the gated anti-leak check")
    if report.terminal_decision.terminal_label not in {
        DECISION_STATISTIC_UPPER_BOUND_PASS,
        OBSERVABLE_RANK_FEATURES_INSUFFICIENT,
        STATISTIC_BUDGET_BREAKS_SUB2,
    }:
        raise ValueError("unexpected decision statistic terminal label")
    if bool(report.terminal_decision.global_per_row_compression_closed):
        raise ValueError("decision statistic diagnostic must not claim global closure")
    if bool(report.terminal_decision.branch_a_trigger):
        raise ValueError("decision statistic diagnostic must not self-trigger branch routing")
    if len(report.step_reports) != len(PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE):
        raise ValueError("decision statistic diagnostic must cover the full preregistered schedule")
    expected_names = [step.name for step in PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE]
    actual_names = [step.schedule_name for step in report.step_reports]
    if actual_names != expected_names:
        raise ValueError("decision statistic step order drifted from the preregistered schedule")
    for step_report, schedule_step in zip(report.step_reports, PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE):
        if int(step_report.step) != int(schedule_step.step):
            raise ValueError("decision statistic step number drifted from the preregistered schedule")
        if not step_report.candidate_rows_fully_transient_observable:
            raise ValueError("branch-(a) candidate rows must remain transient-observable only")
        if step_report.candidate_row_count != step_report.accepted_row_count + step_report.deferred_row_count:
            raise ValueError("decision statistic step must partition every cap row into accepted or deferred")
        if step_report.statistic_schema.observed_bucket_count != len(step_report.bucket_summaries):
            raise ValueError("decision statistic observed bucket count must match the reported buckets")
        if step_report.statistic_schema.bucket_cardinality_bound != _decision_statistic_bucket_cardinality_bound():
            raise ValueError("decision statistic bucket bound drifted from the low-cardinality feature lattice")
        if step_report.statistic_schema.observed_bucket_count > step_report.statistic_schema.bucket_cardinality_bound:
            raise ValueError("decision statistic used more buckets than the low-cardinality bound allows")
        if tuple(step_report.statistic_schema.bucket_key_dimensions) != DECISION_STATISTIC_BUCKET_KEY_DIMENSIONS:
            raise ValueError("decision statistic schema bucket dimensions drifted")
        if step_report.statistic_schema.cutoff_bit_width != DECISION_STATISTIC_CUTOFF_BIT_WIDTH:
            raise ValueError("decision statistic cutoff bit-width drifted from the counts-only contract")
        if step_report.statistic_schema.seed_bits != DECISION_STATISTIC_SEED_BITS:
            raise ValueError("decision statistic seed bits drifted from the gated plan")
        if step_report.statistic_schema.metadata_bits != DECISION_STATISTIC_METADATA_BITS:
            raise ValueError("decision statistic metadata bits drifted from the preregistered ledger")
        if step_report.statistic_schema.statistic_mode != DECISION_STATISTIC_COUNT_ONLY_MODE:
            raise ValueError("decision statistic schema mode drifted from the counts-only contract")
        if bool(step_report.statistic_schema.fits_strictest_required_headroom) != bool(
            step_report.statistic_schema.inclusive_sub2_if_installed
        ):
            raise ValueError("decision statistic headroom fit and inclusive-sub2 flags must agree")
        if step_report.frontier_tie_bucket_count != sum(
            1 for bucket in step_report.bucket_summaries if bucket.frontier_tie_crosses_boundary
        ):
            raise ValueError("decision statistic frontier-tie count drifted from the bucket summaries")
        seen_bucket_keys: set[tuple[str, int, int]] = set()
        for bucket in step_report.bucket_summaries:
            bucket_key = (bucket.state_key, int(bucket.current_q_level), int(bucket.move_direction))
            if bucket_key in seen_bucket_keys:
                raise ValueError("decision statistic bucket keys must stay unique")
            seen_bucket_keys.add(bucket_key)
            if bucket.state_key not in PRIMARY_STATE_KEYS:
                raise ValueError("decision statistic bucket used an unknown state_key")
            if int(bucket.current_q_level) not in (-1, 0, 1):
                raise ValueError("decision statistic bucket used an invalid q level")
            if int(bucket.move_direction) not in (-1, 1):
                raise ValueError("decision statistic bucket used an invalid move direction")
            if bucket.accepted_count + bucket.deferred_count != bucket.candidate_row_count:
                raise ValueError("decision statistic bucket counts must partition the bucket candidates")
            if bool(bucket.decisive_bucket) != bool(0 < bucket.accepted_count < bucket.candidate_row_count):
                raise ValueError("decision statistic decisive-bucket flag drifted from the counts")
            if bucket.frontier_tie_crosses_boundary and not bucket.decisive_bucket:
                raise ValueError("decision statistic frontier ties are only meaningful on decisive buckets")
        _validate_decision_statistic_statistic_input(step_report)
        if step_report.observable_rank_features_sufficient:
            if not (
                step_report.canonical_matches_exact
                and step_report.shuffled_matches_exact
                and step_report.shuffle_preserves_outcome
            ):
                raise ValueError("decision statistic sufficiency requires exact canonical+shuffled agreement")
        else:
            if step_report.insufficiency_reason is None:
                raise ValueError("decision statistic insufficiency must name a reason")
    budget_failures = [
        step.schedule_name
        for step in report.step_reports
        if not step.statistic_schema.fits_strictest_required_headroom
    ]
    insufficiencies = [
        step.schedule_name
        for step in report.step_reports
        if not step.observable_rank_features_sufficient
    ]
    if report.terminal_decision.terminal_label == DECISION_STATISTIC_UPPER_BOUND_PASS:
        if budget_failures or insufficiencies:
            raise ValueError("decision statistic PASS requires no budget failures and no sufficiency failures")
    elif report.terminal_decision.terminal_label == STATISTIC_BUDGET_BREAKS_SUB2:
        if not budget_failures:
            raise ValueError("budget-break terminal requires an actual budget failure")
    else:
        if budget_failures:
            raise ValueError("observable-rank insufficiency must not mask a prior budget failure")
        if not insufficiencies:
            raise ValueError("observable-rank insufficiency terminal requires a real sufficiency failure")
    if report.terminal_decision.first_budget_failure_step != (
        budget_failures[0] if budget_failures else None
    ):
        raise ValueError("decision statistic first budget failure step drifted from the step reports")
    if report.terminal_decision.first_insufficient_step != (
        insufficiencies[0] if insufficiencies else None
    ):
        raise ValueError("decision statistic first insufficiency step drifted from the step reports")
    if report.terminal_decision.peak_statistic_step not in actual_names:
        raise ValueError("decision statistic peak step must name one of the traced steps")
    _assert_no_tensors(report.to_dict())


def _expected_bucket_decisive_bits(
    bucket: TieReservationProjectionBucketReport,
) -> int:
    label = bucket.decisive_practical_encoding_label
    if label == TIE_MEMBERSHIP_MASK_ENCODING:
        return int(bucket.mask_bits)
    if label == TIE_SELECTED_OFFSET_ENCODING:
        return int(bucket.selected_offset_bits)
    raise ValueError("bucket decisive practical encoding must be an exact practical encoding")


def validate_tie_frontier_reservation_lower_bound_report(
    report: TieFrontierReservationLowerBoundReport,
) -> None:
    if report.schema_version != TIE_FRONTIER_RESERVATION_SCHEMA_VERSION:
        raise ValueError("unexpected tie-frontier reservation schema version")
    if report.label != TIE_FRONTIER_RESERVATION_LABEL:
        raise ValueError("unexpected tie-frontier reservation label")
    if report.candidate_name != TIE_FRONTIER_RESERVATION_CANDIDATE:
        raise ValueError("tie-frontier reservation diagnostic candidate drifted")
    if report.source_decision_statistic_label != DECISION_STATISTIC_UPPER_BOUND_LABEL:
        raise ValueError("tie-frontier reservation diagnostic must cite the committed branch-(a) label")
    if report.source_decision_statistic_terminal_label != OBSERVABLE_RANK_FEATURES_INSUFFICIENT:
        raise ValueError("tie-frontier reservation diagnostic must inherit the branch-(a) insufficiency source")
    if tuple(report.required_q_ledger_rows) != SCALE_REQUIRED_Q_LEDGER_ROWS:
        raise ValueError("tie-frontier reservation required rows drifted from Slice 1d")
    if tuple(report.sensitivity_q_ledger_rows) != SCALE_SENSITIVITY_Q_LEDGER_ROWS:
        raise ValueError("tie-frontier reservation sensitivity rows drifted from Slice 1d")
    if report.terminal_decision.terminal_label not in {
        TIE_RESERVATION_FITS_HEADROOM_CANDIDATE_HYBRID,
        TIE_RESERVATION_BREAKS_SUB2,
        TIE_DENSITY_AMBIGUOUS_NEEDS_TRACE,
    }:
        raise ValueError("unexpected tie-frontier reservation terminal label")
    if bool(report.terminal_decision.global_per_row_compression_closed):
        raise ValueError("tie-frontier reservation diagnostic must not claim global closure")
    if bool(report.terminal_decision.branch_a_trigger):
        raise ValueError("tie-frontier reservation diagnostic must not self-trigger branch routing")
    if not report.observed_failing_bucket_reports:
        raise ValueError("tie-frontier reservation diagnostic requires observed failing buckets")
    for bucket in report.observed_failing_bucket_reports:
        if bucket.tie_group_size <= 0:
            raise ValueError("observed tie buckets must have positive tie group size")
        if bucket.exact_accepted_within_tie_count <= 0:
            raise ValueError("observed tie buckets must retain at least one exact accepted row")
        if bucket.exact_accepted_within_tie_count > bucket.tie_group_size:
            raise ValueError("observed tie bucket accepted count cannot exceed tie group size")
        if bucket.decisive_practical_encoding_label not in {
            TIE_MEMBERSHIP_MASK_ENCODING,
            TIE_SELECTED_OFFSET_ENCODING,
        }:
            raise ValueError("observed tie bucket decisive encoding drifted")
        if bucket.theoretical_lower_bound_bits > bucket.decisive_practical_bits:
            raise ValueError("practical exact-retention encoding cannot beat the theoretical lower bound")
    row_map = {row.q_regime_name: row for row in report.row_comparisons}
    if set(name for name in row_map if row_map[name].row_role == "required_gate") != set(SCALE_REQUIRED_Q_LEDGER_ROWS):
        raise ValueError("tie-frontier reservation diagnostic must cover the required rows exactly")
    for row in report.row_comparisons:
        if not row.absolute_count_step_reports or not row.rate_held_step_reports:
            raise ValueError("each tie-frontier row must keep both absolute-count and rate-held reports")
        for step in row.absolute_count_step_reports:
            if not bool(step.diagnostic_only):
                raise ValueError("absolute-count tie reservation reports must stay diagnostic only")
            if step.projection_label != OBSERVED_TIE_RESERVATION_DIAGNOSTIC:
                raise ValueError("absolute-count tie reservation label drifted")
        for step in row.rate_held_step_reports:
            if bool(step.diagnostic_only):
                raise ValueError("rate-held tie reservation reports must stay decisive")
            if step.projection_label != RATE_HELD_TIE_RESERVATION_DIAGNOSTIC:
                raise ValueError("rate-held tie reservation label drifted")
            if step.decisive_practical_encoding_label == THEORETICAL_LOWER_BOUND_NON_DECISIVE:
                raise ValueError("theoretical lower bound must not be used as the decisive practical encoding")
            bucket_expected_decisive_total = 0
            bucket_mask_total = 0
            bucket_offset_total = 0
            bucket_theoretical_total = 0
            bucket_labels: set[str] = set()
            for bucket in step.bucket_reports:
                expected_bucket_decisive_bits = _expected_bucket_decisive_bits(bucket)
                if int(bucket.decisive_practical_bits) != int(expected_bucket_decisive_bits):
                    raise ValueError("bucket decisive practical bits must equal the chosen practical encoding bits")
                if int(bucket.theoretical_lower_bound_bits) > int(bucket.decisive_practical_bits):
                    raise ValueError("bucket theoretical lower bound must stay <= decisive practical bits")
                bucket_expected_decisive_total += int(expected_bucket_decisive_bits)
                bucket_mask_total += int(bucket.mask_bits)
                bucket_offset_total += int(bucket.selected_offset_bits)
                bucket_theoretical_total += int(bucket.theoretical_lower_bound_bits)
                bucket_labels.add(str(bucket.decisive_practical_encoding_label))
            if int(step.mask_total_bits) != int(bucket_mask_total):
                raise ValueError("step mask total must equal the sum over projected buckets")
            if int(step.selected_offset_total_bits) != int(bucket_offset_total):
                raise ValueError("step offset total must equal the sum over projected buckets")
            if int(step.theoretical_lower_bound_total_bits) != int(bucket_theoretical_total):
                raise ValueError("step theoretical lower-bound total must equal the sum over projected buckets")
            if int(step.decisive_tie_reservation_total_bits) != int(bucket_expected_decisive_total):
                raise ValueError("step decisive tie total must equal the sum of chosen practical bucket encodings")
            target_eligible = int(step.target_eligible_weight_count)
            if not math.isclose(
                float(step.mask_bits_per_weight),
                float(step.mask_total_bits) / float(target_eligible),
                abs_tol=1e-12,
            ):
                raise ValueError("step mask bpw must equal mask total / eligible weights")
            if not math.isclose(
                float(step.selected_offset_bits_per_weight),
                float(step.selected_offset_total_bits) / float(target_eligible),
                abs_tol=1e-12,
            ):
                raise ValueError("step offset bpw must equal offset total / eligible weights")
            if not math.isclose(
                float(step.theoretical_lower_bound_bits_per_weight),
                float(step.theoretical_lower_bound_total_bits) / float(target_eligible),
                abs_tol=1e-12,
            ):
                raise ValueError("step theoretical lower-bound bpw must equal total / eligible weights")
            if not math.isclose(
                float(step.decisive_tie_reservation_bits_per_weight),
                float(step.decisive_tie_reservation_total_bits) / float(target_eligible),
                abs_tol=1e-12,
            ):
                raise ValueError("step decisive tie bpw must equal decisive total / eligible weights")
            if not math.isclose(
                float(step.combined_decisive_bits_per_weight),
                float(step.decision_statistic_bits_per_weight)
                + float(step.decisive_tie_reservation_bits_per_weight),
                abs_tol=1e-12,
            ):
                raise ValueError("step combined bpw must equal decision-statistic bpw + decisive tie bpw")
            if len(bucket_labels) == 1:
                expected_label = next(iter(bucket_labels))
            else:
                expected_label = "mixed_per_bucket_min_practical_exact_retention"
            if step.decisive_practical_encoding_label != expected_label:
                raise ValueError("step decisive encoding label must reflect the chosen bucket practical encodings")
            if step.scaling_model_defensible:
                if step.joint_ta_scaling_model != FULL_PLATEAU_JOINT_TA_SCALING_MODEL:
                    raise ValueError("defensible rate-held tie model drifted from the full-plateau contract")
        if not math.isclose(
            row.rate_held_peak_combined_bits_per_weight,
            max(step.combined_decisive_bits_per_weight for step in row.rate_held_step_reports),
            abs_tol=1e-12,
        ):
            raise ValueError("rate-held peak bpw drifted from the step reports")
    required_rows = [row for row in report.row_comparisons if row.row_role == "required_gate"]
    any_ambiguous = any(not row.joint_ta_scaling_model_defensible for row in required_rows)
    all_fit = all(row.rate_held_fits_strictest_headroom for row in required_rows)
    if report.terminal_decision.terminal_label == TIE_DENSITY_AMBIGUOUS_NEEDS_TRACE:
        if not any_ambiguous:
            raise ValueError("ambiguous terminal requires at least one required-row scaling ambiguity")
    elif report.terminal_decision.terminal_label == TIE_RESERVATION_BREAKS_SUB2:
        if any_ambiguous or all_fit:
            raise ValueError("break terminal requires defensible scaling and an actual strictest-headroom miss")
    else:
        if any_ambiguous or not all_fit:
            raise ValueError("candidate-hybrid fit requires defensible scaling and fits on every required row")
    _assert_no_tensors(report.to_dict())


__all__ = [
    "ABSOLUTE_COUNT_LOWER_BOUND_DIAGNOSTIC",
    "ACCUMULATOR_FREE_NULL_BASELINE",
    "A_COLD_EXCEPTION_BUDGET_LEVER_LABEL",
    "A_FUNDAMENTALLY_OVER_LABEL",
    "BACKLOG_K_POLICIES",
    "CAPACITY_LOCALIZATION_DIAGNOSTIC_LABEL",
    "CAPACITY_LOCALIZATION_DIAGNOSTIC_SCHEMA_VERSION",
    "CANDIDATE_ADMISSION_DIAGNOSTIC_LABEL",
    "CANDIDATE_ADMISSION_DIAGNOSTIC_SCHEMA_VERSION",
    "CUMULATIVE_SCHEDULE_MODE",
    "DECISION_STATISTIC_UPPER_BOUND_LABEL",
    "DECISION_STATISTIC_UPPER_BOUND_PASS",
    "DECISION_STATISTIC_UPPER_BOUND_SCHEMA_VERSION",
    "HOT_BUDGET_POINT_LABELS",
    "OBSERVED_TIE_RESERVATION_DIAGNOSTIC",
    "K_SWEEP_JOINT_INFEASIBLE",
    "K_SWEEP_MINIMAL_VIABLE_PASS",
    "K_SWEEP_REPRESENTATION_WALL",
    "OBSERVABLE_RANK_FEATURES_INSUFFICIENT",
    "ONE_STEP_LOCAL_DIAGNOSTIC_MODE",
    "ORACLE_UPPER_BOUND_ADMISSION_DIAGNOSTIC",
    "PER_ROW_COMPRESSION_CLOSED_BY_EASY_CASE_LOWER_BOUND",
    "PER_ROW_COMPRESSION_CLOSED_TINY_FIXTURE_LOWER_BOUND_ONLY",
    "PRIMARY_CURVE_LABEL",
    "RATE_HELD_B_STILL_OVER_SCALE_HEADROOM_CANDIDATE_BRANCH_A",
    "RATE_HELD_B_STORAGE_DIAGNOSTIC",
    "RATE_HELD_COUNT_ROUNDING_POLICY",
    "REAL_BACKLOG_LOWER_BOUND_LABEL",
    "RATE_HELD_TIE_RESERVATION_DIAGNOSTIC",
    "REAL_BACKLOG_LOWER_BOUND_SCHEMA_VERSION",
    "REPRESENTATIVE_TRACE_UNDERPOWERED_FOR_CLOSURE",
    "REPRESENTATIVE_VERDICT_LABEL",
    "REPRESENTATIVE_VERDICT_SCHEMA_VERSION",
    "SCALE_APPROPRIATE_B_STORAGE_LABEL",
    "THEORETICAL_LOWER_BOUND_NON_DECISIVE",
    "TIE_DENSITY_AMBIGUOUS_NEEDS_TRACE",
    "TIE_FRONTIER_RESERVATION_CANDIDATE",
    "TIE_FRONTIER_RESERVATION_FITS_HEADROOM_CANDIDATE_HYBRID",
    "TIE_FRONTIER_RESERVATION_FITS_HEADROOM_CANDIDATE_HYBRID",
    "TIE_FRONTIER_RESERVATION_LABEL",
    "TIE_FRONTIER_RESERVATION_SCHEMA_VERSION",
    "TIE_MEMBERSHIP_MASK_ENCODING",
    "TIE_RESERVATION_BREAKS_SUB2",
    "TIE_SELECTED_OFFSET_ENCODING",
    "SCALE_APPROPRIATE_B_STORAGE_SCHEMA_VERSION",
    "SCALE_APPROPRIATE_COMPARISON_AMBIGUOUS_NEEDS_BACKLOG_DENSITY_TRACE",
    "DecisionStatisticBucketSummary",
    "DecisionStatisticSchemaReport",
    "DecisionStatisticStepReport",
    "DecisionStatisticUpperBoundDecision",
    "DecisionStatisticUpperBoundReport",
    "TieFrontierObservedBucketReport",
    "TieFrontierReservationDecision",
    "TieFrontierReservationLowerBoundReport",
    "TieReservationProjectionBucketReport",
    "TieReservationRowComparisonReport",
    "TieReservationStepProjectionReport",
    "RealBacklogLowerBoundDecision",
    "RealBacklogLowerBoundReport",
    "RealBacklogLowerBoundStepReport",
    "RealBacklogLowerBoundSweepEntry",
    "RealBacklogTraceStepReport",
    "RealBacklogTraceSummaryReport",
    "ScaleAppropriateBStorageComparisonReport",
    "ScaleAppropriateComparisonDecision",
    "ScaleAppropriateLedgerComparisonReport",
    "ScaleAppropriateProjectionStepReport",
    "SPARSE_AMORTIZED_CANDIDATE_RESURRECTED_FOR_HARDER_TRACE",
    "STATISTIC_BUDGET_BREAKS_SUB2",
    "TINY_FIXTURE_HEADROOM_SOURCE",
    "CandidateABudgetLocalizationReport",
    "CandidateABudgetReadout",
    "CandidateAdmissionDiagnosticReport",
    "CandidateAdmissionDiagnosticRunReport",
    "CandidateAdmissionDiagnosticStepReport",
    "CandidateCapacityLocalizationReport",
    "CandidateCapacityStepReport",
    "CandidateKSweepDecision",
    "CandidateKSweepEntry",
    "CandidateKSweepRunReport",
    "CandidatePromotionDecision",
    "RepresentativeCurveRunReport",
    "RepresentativeDriftVerdictReport",
    "RepresentativeStepReport",
    "representative_engineering_guard_spec",
    "run_candidate_admission_diagnostic",
    "run_candidate_capacity_localization_diagnostic",
    "run_decision_statistic_upper_bound_diagnostic",
    "run_real_backlog_lower_bound_diagnostic",
    "run_tie_frontier_reservation_lower_bound_diagnostic",
    "run_scale_appropriate_b_storage_comparison",
    "run_representative_bounded_delta_drift_verdict",
    "validate_candidate_admission_diagnostic_report",
    "validate_candidate_capacity_localization_report",
    "validate_decision_statistic_upper_bound_report",
    "validate_real_backlog_lower_bound_diagnostic_report",
    "validate_tie_frontier_reservation_lower_bound_report",
    "validate_scale_appropriate_b_storage_comparison_report",
    "validate_representative_bounded_delta_drift_verdict_report",
    "VIRTUAL_DECISION_STATISTIC_CANDIDATE",
]
