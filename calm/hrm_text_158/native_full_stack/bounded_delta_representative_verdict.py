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
    _cap_inputs_for_density_inputs,
    _density_inputs_for_step,
    _initial_states,
    default_vote_update_spec,
    pre_register_source_bindingness,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    BOUNDED_DELTA_ACCUMULATOR_LABEL,
    BOUNDED_DELTA_ACCUMULATOR_SCHEMA_VERSION,
    BOUNDED_DELTA_GUARDRAIL_FAILED,
    BOUNDED_DELTA_LEDGER_FAILED,
    BOUNDED_DELTA_WITH_REPORT,
    BoundedDeltaGuardSpec,
    BoundedDeltaInclusiveLedger,
    BoundedDeltaMeasuredReport,
    BoundedDeltaOracleInput,
    BoundedDeltaReferenceReport,
    BoundedDeltaStorageProjection,
    _backlog_key_set,
    _evaluate_guardrail,
    _hash_cap_spec,
    _hash_vote_inputs,
    _identity_sha256,
    _p95,
    _rank_delta,
    _run_reference_path,
    _symmetric_fraction,
    _tensor_sha256,
    bounded_delta_candidate_assessment,
    bounded_delta_inclusive_ledger,
    compare_bounded_delta_step_to_int16_oracle,
    decode_bounded_accumulator_to_i16,
    encode_budget_capped_hybrid_reference,
    project_bounded_delta_accumulator_bpw,
    validate_bounded_delta_inclusive_ledger,
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


def _classify_from_guard_and_ledger(
    guard_passed: bool,
    ledger: BoundedDeltaInclusiveLedger,
) -> str:
    if not guard_passed:
        return BOUNDED_DELTA_GUARDRAIL_FAILED
    if not ledger.claimable_physical_sub2:
        return BOUNDED_DELTA_LEDGER_FAILED
    return BOUNDED_DELTA_WITH_REPORT


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
    ledger = bounded_delta_inclusive_ledger(q_ledger_row, projection)
    validate_bounded_delta_inclusive_ledger(ledger)

    candidate_changed_count, candidate_fraction = _symmetric_fraction(
        exact_path.candidate_ids,
        bounded_path.candidate_ids,
    )
    accepted_changed_count, accepted_fraction = _symmetric_fraction(
        exact_path.accepted_ids,
        bounded_path.accepted_ids,
    )
    deferred_changed_count, deferred_fraction = _symmetric_fraction(
        exact_path.deferred_ids,
        bounded_path.deferred_ids,
    )
    q_changed_count, q_fraction = _symmetric_fraction(
        exact_path.q_changed_ids,
        bounded_path.q_changed_ids,
    )
    exact_backlog_ids = _backlog_key_set(
        exact_path.cap_result.deferred_backlog if exact_path.cap_result is not None else exact_backlog
    )
    bounded_backlog_ids = _backlog_key_set(bounded_stored_backlog)
    backlog_changed_count, backlog_fraction = _symmetric_fraction(
        exact_backlog_ids,
        bounded_backlog_ids,
    )
    direction_keys = set(exact_path.candidate_direction_by_id) | set(
        bounded_path.candidate_direction_by_id
    )
    direction_changed = sum(
        1
        for identity in direction_keys
        if exact_path.candidate_direction_by_id.get(identity)
        != bounded_path.candidate_direction_by_id.get(identity)
    )
    acc_errors: list[torch.Tensor] = []
    exact_hashes: dict[str, str] = {}
    bounded_hashes: dict[str, str] = {}
    residual_hash_match = True
    for state_key in PRIMARY_STATE_KEYS:
        exact_acc = exact_path.output_acc_by_key[state_key].detach().cpu().to(torch.int32)
        bounded_acc = bounded_path.output_acc_by_key[state_key].detach().cpu().to(torch.int32)
        exact_hashes[state_key] = _tensor_sha256(exact_acc)
        bounded_hashes[state_key] = _tensor_sha256(bounded_acc)
        residual_hash_match = residual_hash_match and exact_hashes[state_key] == bounded_hashes[state_key]
        acc_errors.append((exact_acc - bounded_acc).abs().flatten())
    all_errors = torch.cat(acc_errors) if acc_errors else torch.empty(0, dtype=torch.int32)
    max_abs_error = int(all_errors.max().item()) if int(all_errors.numel()) else 0

    hot_ids = {
        (state_key, int(index))
        for state_key, indices in hot_by_state.items()
        for index in indices
    }
    decision_symdiff = (
        (exact_path.candidate_ids ^ bounded_path.candidate_ids)
        | (exact_path.accepted_ids ^ bounded_path.accepted_ids)
        | (exact_path.deferred_ids ^ bounded_path.deferred_ids)
        | (exact_path.q_changed_ids ^ bounded_path.q_changed_ids)
    )
    same_initial_q = all(
        _tensor_sha256(exact_input_states[key].q_levels)
        == _tensor_sha256(bounded_input_states[key].q_levels)
        for key in PRIMARY_STATE_KEYS
    )
    vote_hash = _hash_vote_inputs(inputs)
    offsets_hash = math.fsum(float(value) for value in tensor_offsets.values())
    measured = BoundedDeltaMeasuredReport(
        schema_version=BOUNDED_DELTA_ACCUMULATOR_SCHEMA_VERSION,
        label=BOUNDED_DELTA_ACCUMULATOR_LABEL,
        candidate_name="cumulative_hot_exact_cold_default_bounded_backlog_k_policy",
        candidate_changed_count=candidate_changed_count,
        candidate_union_count=len(exact_path.candidate_ids | bounded_path.candidate_ids),
        candidate_changed_fraction=candidate_fraction,
        direction_changed_count=direction_changed,
        accepted_changed_count=accepted_changed_count,
        accepted_union_count=len(exact_path.accepted_ids | bounded_path.accepted_ids),
        accepted_changed_fraction=accepted_fraction,
        deferred_changed_count=deferred_changed_count,
        deferred_union_count=len(exact_path.deferred_ids | bounded_path.deferred_ids),
        deferred_changed_fraction=deferred_fraction,
        q_changed_count=q_changed_count,
        q_changed_union_count=len(exact_path.q_changed_ids | bounded_path.q_changed_ids),
        q_changed_fraction=q_fraction,
        backlog_key_changed_count=backlog_changed_count,
        backlog_key_union_count=len(exact_backlog_ids | bounded_backlog_ids),
        backlog_key_changed_fraction=backlog_fraction,
        cap_frontier_rank_delta=_rank_delta(
            exact_path.ordered_row_ids,
            bounded_path.ordered_row_ids,
        ),
        hot_risk_changed_count=len(decision_symdiff & hot_ids),
        max_abs_acc_error=max_abs_error,
        p95_abs_acc_error=_p95(all_errors),
        accumulator_residual_hash_match=residual_hash_match,
        exact_accumulator_residuals_sha256=exact_hashes,
        bounded_accumulator_residuals_sha256=bounded_hashes,
        exact_candidate_identities_sha256=_identity_sha256(exact_path.candidate_ids),
        bounded_candidate_identities_sha256=_identity_sha256(bounded_path.candidate_ids),
        exact_accepted_identities_sha256=_identity_sha256(exact_path.accepted_ids),
        bounded_accepted_identities_sha256=_identity_sha256(bounded_path.accepted_ids),
        exact_deferred_identities_sha256=_identity_sha256(exact_path.deferred_ids),
        bounded_deferred_identities_sha256=_identity_sha256(bounded_path.deferred_ids),
        oracle_parity={
            "same_initial_q": same_initial_q,
            "same_votes_sha256": True,
            "votes_sha256": vote_hash,
            "same_cap_spec": True,
            "cap_spec_sha256": _hash_cap_spec(global_cap_spec),
            "same_deferred_backlog": False,
            "bounded_backlog_policy_active": True,
            "cumulative_carry_forward": True,
            "bounded_reinitialized_from_exact": False,
            "exact_input_deferred_backlog_count": _backlog_entry_count(exact_backlog),
            "bounded_input_deferred_backlog_count": _backlog_entry_count(bounded_input_backlog),
            "exact_output_deferred_backlog_count": len(exact_backlog_ids),
            "bounded_stored_deferred_backlog_count": len(bounded_backlog_ids),
            "same_tensor_offsets": True,
            "tensor_offsets_checksum": str(offsets_hash),
            "path_difference": (
                "cumulative path differs by bounded accumulator encode_decode, "
                "bounded-backlog encode/drop, and prior bounded q/acc/backlog carry-forward"
            ),
        },
    )
    guard_eval = _evaluate_guardrail(guard_spec, measured)
    classification = _classify_from_guard_and_ledger(guard_eval.guard_passed, ledger)
    return BoundedDeltaReferenceReport(
        schema_version=BOUNDED_DELTA_ACCUMULATOR_SCHEMA_VERSION,
        label=BOUNDED_DELTA_ACCUMULATOR_LABEL,
        candidate_name="cumulative_hot_exact_cold_default_bounded_backlog_k_policy",
        classification=classification,
        ledger=ledger,
        storage_projection=projection,
        guard_spec=guard_spec,
        measured_report=measured,
        guard_passed=guard_eval.guard_passed,
        failed_metrics=guard_eval.failed_metrics,
        candidate_assessment=bounded_delta_candidate_assessment(),
        raw_arrays_included=False,
        non_claims=(
            "cumulative representative verdict over generated in-tree schedule only",
            "no production vote_update/global_rate_cap replacement",
            "no GPU lane",
            "no trainer/live-run/checkpoint/creditdir mutation",
            "no acquisition, retention, or stability claim",
            "guard-bound adequacy deferred to C2",
            "compact counts/hashes only; no raw per-weight arrays",
        ),
        next_candidate_if_failed="event_coded_crossing_residual_log",
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


__all__ = [
    "BACKLOG_K_POLICIES",
    "CUMULATIVE_SCHEDULE_MODE",
    "HOT_BUDGET_POINT_LABELS",
    "ONE_STEP_LOCAL_DIAGNOSTIC_MODE",
    "PRIMARY_CURVE_LABEL",
    "REPRESENTATIVE_VERDICT_LABEL",
    "REPRESENTATIVE_VERDICT_SCHEMA_VERSION",
    "RepresentativeCurveRunReport",
    "RepresentativeDriftVerdictReport",
    "RepresentativeStepReport",
    "representative_engineering_guard_spec",
    "run_representative_bounded_delta_drift_verdict",
    "validate_representative_bounded_delta_drift_verdict_report",
]
