"""C1.1c representative drift-vs-budget verdict for bounded-delta state.

This composes the C1.1b pre-registered native-loop pressure schedule with the
C1.1c bounded-delta ledger/oracle. Local one-step reports are diagnostic only;
the terminal report is cumulative, carrying exact and bounded q/acc/backlog
states independently across the fixed four-step schedule.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
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
    _identity_sha256,
    _run_reference_path,
    bounded_delta_inclusive_ledger,
    compare_bounded_delta_paths_to_int16_oracle,
    compare_bounded_delta_step_to_int16_oracle,
    decode_bounded_accumulator_to_i16,
    encode_budget_capped_hybrid_reference,
    project_bounded_delta_accumulator_bpw,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapSpec,
    tensor_offsets_for_vote_update_states,
)
from calm.hrm_text_158.native_full_stack.q_entropy_packing import (
    Base3QEntropyLedgerRow,
    default_base3_q_entropy_ledger_table,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateState


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
COARSE_SIGNED_CHARGE_BLOCK_SIZE = 8


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


__all__ = [
    "ACCUMULATOR_FREE_NULL_BASELINE",
    "BACKLOG_K_POLICIES",
    "CANDIDATE_ADMISSION_DIAGNOSTIC_LABEL",
    "CANDIDATE_ADMISSION_DIAGNOSTIC_SCHEMA_VERSION",
    "CUMULATIVE_SCHEDULE_MODE",
    "HOT_BUDGET_POINT_LABELS",
    "ONE_STEP_LOCAL_DIAGNOSTIC_MODE",
    "ORACLE_UPPER_BOUND_ADMISSION_DIAGNOSTIC",
    "PRIMARY_CURVE_LABEL",
    "REPRESENTATIVE_VERDICT_LABEL",
    "REPRESENTATIVE_VERDICT_SCHEMA_VERSION",
    "CandidateAdmissionDiagnosticReport",
    "CandidateAdmissionDiagnosticRunReport",
    "CandidateAdmissionDiagnosticStepReport",
    "CandidatePromotionDecision",
    "RepresentativeCurveRunReport",
    "RepresentativeDriftVerdictReport",
    "RepresentativeStepReport",
    "representative_engineering_guard_spec",
    "run_candidate_admission_diagnostic",
    "run_representative_bounded_delta_drift_verdict",
    "validate_candidate_admission_diagnostic_report",
    "validate_representative_bounded_delta_drift_verdict_report",
]
