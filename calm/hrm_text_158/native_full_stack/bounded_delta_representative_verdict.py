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


__all__ = [
    "ACCUMULATOR_FREE_NULL_BASELINE",
    "A_COLD_EXCEPTION_BUDGET_LEVER_LABEL",
    "A_FUNDAMENTALLY_OVER_LABEL",
    "BACKLOG_K_POLICIES",
    "CAPACITY_LOCALIZATION_DIAGNOSTIC_LABEL",
    "CAPACITY_LOCALIZATION_DIAGNOSTIC_SCHEMA_VERSION",
    "CANDIDATE_ADMISSION_DIAGNOSTIC_LABEL",
    "CANDIDATE_ADMISSION_DIAGNOSTIC_SCHEMA_VERSION",
    "CUMULATIVE_SCHEDULE_MODE",
    "HOT_BUDGET_POINT_LABELS",
    "K_SWEEP_JOINT_INFEASIBLE",
    "K_SWEEP_MINIMAL_VIABLE_PASS",
    "K_SWEEP_REPRESENTATION_WALL",
    "ONE_STEP_LOCAL_DIAGNOSTIC_MODE",
    "ORACLE_UPPER_BOUND_ADMISSION_DIAGNOSTIC",
    "PER_ROW_COMPRESSION_CLOSED_BY_EASY_CASE_LOWER_BOUND",
    "PER_ROW_COMPRESSION_CLOSED_TINY_FIXTURE_LOWER_BOUND_ONLY",
    "PRIMARY_CURVE_LABEL",
    "REAL_BACKLOG_LOWER_BOUND_LABEL",
    "REAL_BACKLOG_LOWER_BOUND_SCHEMA_VERSION",
    "REPRESENTATIVE_TRACE_UNDERPOWERED_FOR_CLOSURE",
    "REPRESENTATIVE_VERDICT_LABEL",
    "REPRESENTATIVE_VERDICT_SCHEMA_VERSION",
    "RealBacklogLowerBoundDecision",
    "RealBacklogLowerBoundReport",
    "RealBacklogLowerBoundStepReport",
    "RealBacklogLowerBoundSweepEntry",
    "RealBacklogTraceStepReport",
    "RealBacklogTraceSummaryReport",
    "SPARSE_AMORTIZED_CANDIDATE_RESURRECTED_FOR_HARDER_TRACE",
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
    "run_real_backlog_lower_bound_diagnostic",
    "run_representative_bounded_delta_drift_verdict",
    "validate_candidate_admission_diagnostic_report",
    "validate_candidate_capacity_localization_report",
    "validate_real_backlog_lower_bound_diagnostic_report",
    "validate_representative_bounded_delta_drift_verdict_report",
]
