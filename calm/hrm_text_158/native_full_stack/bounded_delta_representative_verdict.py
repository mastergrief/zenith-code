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
ONLINE_ESTIMABILITY_TIE_MASK_SCHEMA_VERSION = (
    "hrm_text_158_c1p1g_strict_observable_tie_mask_online_estimability_diagnostic/v0"
)
ONLINE_ESTIMABILITY_TIE_MASK_LABEL = (
    "c1p1g_strict_observable_tie_mask_online_estimability_diagnostic"
)
ONLINE_ESTIMABLE_TIE_MASK_CANDIDATE = (
    "branch_a_online_estimable_tie_mask_candidate_hybrid"
)
STRICT_OBSERVABLE_TIE_MASK_NOT_IDENTIFIABLE_IDENTITY_BOUND = (
    "strict_observable_tie_mask_not_identifiable_identity_bound"
)
STRICT_OBSERVABLE_TIE_MASK_EXACT_RECOVERABLE_IDENTITY_FREE_CANDIDATE_ONLY = (
    "strict_observable_tie_mask_exact_recoverable_identity_free_candidate_only"
)
STRICT_OBSERVABLE_TIE_MASK_PARTIALLY_RECOVERABLE_NOT_EXACT = (
    "strict_observable_tie_mask_partially_recoverable_not_exact"
)
STRICT_OBSERVABLE_TIE_MASK_SHUFFLE_FALSIFIER = (
    "within_equal_feature_group_reverse_order_falsifier"
)
STRICT_OBSERVABLE_TIE_MASK_ALLOWED_BUCKET_KEY_DIMENSIONS = (
    "state_key",
    "current_q_level",
    "move_direction",
)
STRICT_OBSERVABLE_TIE_MASK_ALLOWED_WITHIN_BUCKET_FEATURE_KEYS = (
    "vote_sign",
    "vote_value",
    "vote_abs",
    "abs_new_acc",
    "threshold_abs",
    "margin_abs_over_threshold",
    "replay_ce_veto_vote_sign",
    "replay_ce_veto_vote_value",
    "replay_ce_veto_move_sign",
    "pc_aux_vote_sign",
    "pc_aux_vote_value",
    "pc_aux_move_sign",
)
STRICT_OBSERVABLE_TIE_MASK_ALLOWED_BUCKET_AGGREGATE_KEYS = (
    "global_cap",
    "candidate_row_count",
    "higher_priority_row_count",
    "residual_cap_slots_entering_bucket",
)
STRICT_OBSERVABLE_TIE_MASK_FORBIDDEN_PREDICTOR_INPUT_KEY_FRAGMENTS = (
    "flat_index",
    "global_flat_index",
    "local_pos",
    "identity",
    "row_id",
    "rank_position",
    "rank",
    "order",
    "backlog",
    "accepted",
    "oracle",
)
PATH_B_IDENTITY_FREE_TIE_RULE_CLASSIFIER_SCHEMA_VERSION = (
    "hrm_text_158_c1p1h_path_b_identity_free_tie_rule_classifier/v0"
)
PATH_B_IDENTITY_FREE_TIE_RULE_CLASSIFIER_LABEL = (
    "c1p1h_path_b_identity_free_tie_rule_classifier"
)
PATH_B_IDENTITY_FREE_TIE_RULE_CLASSIFIER_CANDIDATE = (
    "path_b_identity_free_tie_rule_family_classifier"
)
CLASS_ACTION_ACCEPT_ALL_MIXED_CLASSES = "class_action_accept_all_mixed_classes"
CLASS_ACTION_DEFER_ALL_MIXED_CLASSES_NO_BACKFILL = (
    "class_action_defer_all_mixed_classes_no_backfill"
)
STRICTLY_NEW_EMITTED_IDENTITY_FREE_OBSERVABLE_SPLIT = (
    "strictly_new_emitted_identity_free_observable_split"
)
AGGREGATE_STATE_REDEFINITION = "aggregate_state_redefinition"
CANDIDATE_FAMILY_REQUIRES_IDENTITY_OR_ORDER_SUBSET_SELECTION = (
    "candidate_family_requires_identity_or_order_subset_selection"
)
CANDIDATE_FAMILY_CLASS_UNIFORM_CAP_OVERFLOW_NEGATIVE = (
    "candidate_family_class_uniform_cap_overflow_negative"
)
CANDIDATE_FAMILY_CLASS_UNIFORM_BOUNDED_DEVIATION_CANDIDATE_ONLY = (
    "candidate_family_class_uniform_bounded_deviation_candidate_only"
)
CANDIDATE_FAMILY_NO_EMITTED_IDENTITY_FREE_SPLIT_OBSERVABLE = (
    "candidate_family_no_emitted_identity_free_split_observable"
)
CANDIDATE_FAMILY_EMITTED_IDENTITY_FREE_SPLIT_CANDIDATE_ONLY = (
    "candidate_family_emitted_identity_free_split_candidate_only"
)
CANDIDATE_FAMILY_AGGREGATE_STATE_UNBOUNDED_PERSISTENT_BITS_NEGATIVE = (
    "candidate_family_aggregate_state_unbounded_persistent_bits_negative"
)
CANDIDATE_FAMILY_AGGREGATE_STATE_BOUNDED_PERSISTENT_BITS_CANDIDATE_ONLY = (
    "candidate_family_aggregate_state_bounded_persistent_bits_candidate_only"
)
CANDIDATE_FAMILY_AGGREGATE_STATE_RUNTIME_SEMANTICS_UNSPECIFIED = (
    "candidate_family_aggregate_state_runtime_semantics_unspecified"
)
RUNTIME_TIE_RULE_MUTATION_PARITY_PROBE = "runtime_tie_rule_mutation_parity_probe"
LEARNING_RETENTION_TOLERANCE_PROBE = "learning_retention_tolerance_probe"
CAP_PRESSURE_FRONTIER_ONLY_UNDERFILL_NO_REALLOCATION = (
    "frontier_only_underfill_no_reallocation"
)
CAP_PRESSURE_FRONTIER_OVERFLOW_REQUIRES_ILLEGAL_SUBSET_SELECTION = (
    "frontier_overflow_requires_illegal_subset_selection"
)
PATH_B_AGGREGATE_STATE_METADATA_BITS = 64
PATH_B_AGGREGATE_STATE_CARRY_BITS = 16
PATH_B_DEFER_ALL_BASELINE_PARITY_PROBE_SCHEMA_VERSION = (
    "hrm_text_158_c1p1i_path_b_defer_all_zero_bit_baseline_parity_probe/v0"
)
PATH_B_DEFER_ALL_BASELINE_PARITY_PROBE_LABEL = (
    "c1p1i_path_b_defer_all_zero_bit_baseline_parity_probe"
)
PATH_B_DEFER_ALL_BASELINE_PARITY_PROBE_CANDIDATE = (
    CLASS_ACTION_DEFER_ALL_MIXED_CLASSES_NO_BACKFILL
)
BASELINE_SUFFICIENT_NO_CARRY_NEEDED = "baseline_sufficient_no_carry_needed"
CARRY_CANDIDATE_EARNED = "carry_candidate_earned"
INCONCLUSIVE_NEEDS_LOOP_MEASUREMENT = "inconclusive_needs_loop_measurement"
AGGREGATE_RUNTIME_SEMANTICS_DEFINITION_PLAN = (
    "aggregate_runtime_semantics_definition_plan"
)
PATH_B_AGGREGATE_STATE_RUNTIME_SEMANTICS_SCHEMA_VERSION = (
    "hrm_text_158_c1p1j_path_b_aggregate_state_runtime_semantics_definition/v0"
)
PATH_B_AGGREGATE_STATE_RUNTIME_SEMANTICS_LABEL = (
    "c1p1j_path_b_aggregate_state_runtime_semantics_definition"
)
PATH_B_AGGREGATE_STATE_RUNTIME_SEMANTICS_CANDIDATE = (
    "path_b_aggregate_state_runtime_semantics_definition"
)
CARRY_SEMANTICS_CANDIDATE = "carry_semantics_candidate"
CARRY_SEMANTICS_UNLAWFUL_REQUIRES_IDENTITY_OR_ORDER = (
    "carry_semantics_unlawful_requires_identity_or_order"
)
CARRY_SEMANTICS_CAP_OVERFLOW_OR_BITS_UNBOUNDED = (
    "carry_semantics_cap_overflow_or_bits_unbounded"
)
IMMEDIATE_RECOVERY_SEMANTICS_EXHAUSTED_ON_THIS_STATE_PATH = (
    "immediate_recovery_semantics_exhausted_on_this_state_path"
)
CARRY_FAMILY_QUOTA_RELEASE = "carry_quota_release"
CARRY_FAMILY_CLASS_UNIFORM_ACCEPT_ALL_IF_FIT = (
    "carry_class_uniform_accept_all_if_fit"
)
CARRY_FAMILY_CLASS_UNIFORM_WITH_EXTRA_DEVIATION = (
    "carry_class_uniform_with_extra_deviation"
)
CARRY_FAMILY_CLASS_UNIFORM_DEFER_UNTIL_FIT = (
    "carry_class_uniform_defer_until_fit"
)
CARRY_SUBCASE_EXACT_RECOVERY = "exact_recovery"
CARRY_SUBCASE_CLASS_UNIFORM_BOUNDED_EXTRA_DEVIATION = (
    "class_uniform_recovery_with_bounded_extra_deviation"
)
PATH_B_DEFER_UNTIL_FIT_TTL2_PLAUSIBILITY_PRECHECK_SCHEMA_VERSION = (
    "hrm_text_158_c1p1k_path_b_defer_until_fit_ttl2_fit_plausibility_precheck/v0"
)
PATH_B_DEFER_UNTIL_FIT_TTL2_PLAUSIBILITY_PRECHECK_LABEL = (
    "c1p1k_path_b_defer_until_fit_ttl2_fit_plausibility_precheck"
)
PATH_B_DEFER_UNTIL_FIT_TTL2_PLAUSIBILITY_PRECHECK_CANDIDATE = (
    "defer_until_fit_ttl2_fit_plausibility_precheck"
)
DEFER_UNTIL_FIT_FIRST_FIT_PLAUSIBLE_CANDIDATE_ONLY = (
    "defer_until_fit_first_fit_plausible_candidate_only"
)
DEFER_UNTIL_FIT_TTL2_NO_FIT_ON_STATE_PATH = (
    "defer_until_fit_ttl2_no_fit_on_state_path"
)
DEFER_UNTIL_FIT_HORIZON_MEASUREMENT_INCONCLUSIVE = (
    "defer_until_fit_horizon_measurement_inconclusive"
)
DEFER_UNTIL_FIT_TTL2_ALLOWED_ACTION_INPUT_DIMENSIONS = (
    "state_key",
    "current_q_level",
    "move_direction",
    "stored_debt_count",
    "age_steps",
    "residual_cap_entering_packet",
    "projected_class_count",
    "projected_class_mass",
    "projected_class_cardinality",
    "projection_bits",
)
DEFER_UNTIL_FIT_TTL2_FORBIDDEN_ACTION_INPUT_KEY_FRAGMENTS = (
    "identity",
    "order",
    "prefix",
    "row_id",
    "flat_index",
)
DEFER_UNTIL_FIT_TTL2_HORIZON_STEPS = 2


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
class ObservableTieMaskFeatureClassReport:
    feature_payload: dict[str, Any]
    row_count: int
    accepted_count: int
    deferred_count: int
    mixed_acceptance: bool
    best_case_identity_free_correct_count: int
    best_case_identity_free_hamming_lower_bound: int
    canonical_prefix_matches_exact: bool
    reversed_prefix_matches_exact: bool
    exact_accepted_identities_sha256: str
    canonical_prefix_accepted_identities_sha256: str
    reversed_prefix_accepted_identities_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_payload": dict(self.feature_payload),
            "row_count": int(self.row_count),
            "accepted_count": int(self.accepted_count),
            "deferred_count": int(self.deferred_count),
            "mixed_acceptance": bool(self.mixed_acceptance),
            "best_case_identity_free_correct_count": int(
                self.best_case_identity_free_correct_count
            ),
            "best_case_identity_free_hamming_lower_bound": int(
                self.best_case_identity_free_hamming_lower_bound
            ),
            "canonical_prefix_matches_exact": bool(
                self.canonical_prefix_matches_exact
            ),
            "reversed_prefix_matches_exact": bool(
                self.reversed_prefix_matches_exact
            ),
            "exact_accepted_identities_sha256": self.exact_accepted_identities_sha256,
            "canonical_prefix_accepted_identities_sha256": self.canonical_prefix_accepted_identities_sha256,
            "reversed_prefix_accepted_identities_sha256": self.reversed_prefix_accepted_identities_sha256,
        }


@dataclass(frozen=True)
class ObservableTieMaskBucketReport:
    schedule_name: str
    step: int
    state_key: str
    current_q_level: int
    move_direction: int
    global_cap: int
    candidate_row_count: int
    accepted_row_count: int
    deferred_row_count: int
    higher_priority_row_count: int
    residual_cap_slots_entering_bucket: int
    feature_class_reports: tuple[ObservableTieMaskFeatureClassReport, ...]
    mixed_feature_class_count: int
    mixed_feature_class_row_count: int
    exact_identity_free_recovery_possible: bool
    exact_mask_recovery_rate: float
    best_case_identity_free_correct_count: int
    best_case_identity_free_hamming_lower_bound: int
    best_case_identity_free_mask_accuracy_upper_bound: float
    canonical_order_leaky_matches_exact: bool
    reversed_order_leaky_matches_exact: bool
    within_class_reverse_order_changes_order_leaky_mask: bool
    order_dependence_witnessed: bool
    exact_accepted_identities_sha256: str
    canonical_order_leaky_accepted_identities_sha256: str
    reversed_order_leaky_accepted_identities_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_name": self.schedule_name,
            "step": int(self.step),
            "state_key": self.state_key,
            "current_q_level": int(self.current_q_level),
            "move_direction": int(self.move_direction),
            "global_cap": int(self.global_cap),
            "candidate_row_count": int(self.candidate_row_count),
            "accepted_row_count": int(self.accepted_row_count),
            "deferred_row_count": int(self.deferred_row_count),
            "higher_priority_row_count": int(self.higher_priority_row_count),
            "residual_cap_slots_entering_bucket": int(
                self.residual_cap_slots_entering_bucket
            ),
            "feature_class_reports": [
                report.to_dict() for report in self.feature_class_reports
            ],
            "mixed_feature_class_count": int(self.mixed_feature_class_count),
            "mixed_feature_class_row_count": int(self.mixed_feature_class_row_count),
            "exact_identity_free_recovery_possible": bool(
                self.exact_identity_free_recovery_possible
            ),
            "exact_mask_recovery_rate": float(self.exact_mask_recovery_rate),
            "best_case_identity_free_correct_count": int(
                self.best_case_identity_free_correct_count
            ),
            "best_case_identity_free_hamming_lower_bound": int(
                self.best_case_identity_free_hamming_lower_bound
            ),
            "best_case_identity_free_mask_accuracy_upper_bound": float(
                self.best_case_identity_free_mask_accuracy_upper_bound
            ),
            "canonical_order_leaky_matches_exact": bool(
                self.canonical_order_leaky_matches_exact
            ),
            "reversed_order_leaky_matches_exact": bool(
                self.reversed_order_leaky_matches_exact
            ),
            "within_class_reverse_order_changes_order_leaky_mask": bool(
                self.within_class_reverse_order_changes_order_leaky_mask
            ),
            "order_dependence_witnessed": bool(self.order_dependence_witnessed),
            "exact_accepted_identities_sha256": self.exact_accepted_identities_sha256,
            "canonical_order_leaky_accepted_identities_sha256": self.canonical_order_leaky_accepted_identities_sha256,
            "reversed_order_leaky_accepted_identities_sha256": self.reversed_order_leaky_accepted_identities_sha256,
        }


@dataclass(frozen=True)
class ObservableTieMaskOnlineEstimabilityDecision:
    terminal_label: str
    decisive_bucket_count: int
    exact_recoverable_bucket_count: int
    first_failure_bucket: str | None
    worst_bucket_best_case_identity_free_mask_accuracy_upper_bound: float
    any_mixed_feature_class_split: bool
    any_order_dependence_witnessed: bool
    online_realizable_candidate_hybrid: bool
    implementation_design_earned: bool
    path_b_identity_free_redesign_earned: bool
    global_per_row_compression_closed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "terminal_label": self.terminal_label,
            "decisive_bucket_count": int(self.decisive_bucket_count),
            "exact_recoverable_bucket_count": int(self.exact_recoverable_bucket_count),
            "first_failure_bucket": self.first_failure_bucket,
            "worst_bucket_best_case_identity_free_mask_accuracy_upper_bound": float(
                self.worst_bucket_best_case_identity_free_mask_accuracy_upper_bound
            ),
            "any_mixed_feature_class_split": bool(
                self.any_mixed_feature_class_split
            ),
            "any_order_dependence_witnessed": bool(
                self.any_order_dependence_witnessed
            ),
            "online_realizable_candidate_hybrid": bool(
                self.online_realizable_candidate_hybrid
            ),
            "implementation_design_earned": bool(
                self.implementation_design_earned
            ),
            "path_b_identity_free_redesign_earned": bool(
                self.path_b_identity_free_redesign_earned
            ),
            "global_per_row_compression_closed": bool(
                self.global_per_row_compression_closed
            ),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ObservableTieMaskOnlineEstimabilityReport:
    schema_version: str
    label: str
    source_bindingness: Any
    field_coverage: SourceFieldCoverage
    candidate_name: str
    source_tie_frontier_reservation_label: str
    source_tie_frontier_reservation_terminal_label: str
    source_decision_statistic_terminal_label: str
    strictest_required_q_regime_name: str
    strictest_headroom_bits_per_weight: float
    allowed_bucket_key_dimensions: tuple[str, ...]
    allowed_within_bucket_feature_keys: tuple[str, ...]
    allowed_bucket_aggregate_keys: tuple[str, ...]
    forbidden_predictor_input_key_fragments: tuple[str, ...]
    shuffle_falsifier: str
    bucket_reports: tuple[ObservableTieMaskBucketReport, ...]
    terminal_decision: ObservableTieMaskOnlineEstimabilityDecision
    raw_arrays_included: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "source_bindingness": self.source_bindingness.to_dict(),
            "field_coverage": self.field_coverage.to_dict(),
            "candidate_name": self.candidate_name,
            "source_tie_frontier_reservation_label": self.source_tie_frontier_reservation_label,
            "source_tie_frontier_reservation_terminal_label": self.source_tie_frontier_reservation_terminal_label,
            "source_decision_statistic_terminal_label": self.source_decision_statistic_terminal_label,
            "strictest_required_q_regime_name": self.strictest_required_q_regime_name,
            "strictest_headroom_bits_per_weight": float(
                self.strictest_headroom_bits_per_weight
            ),
            "allowed_bucket_key_dimensions": list(self.allowed_bucket_key_dimensions),
            "allowed_within_bucket_feature_keys": list(
                self.allowed_within_bucket_feature_keys
            ),
            "allowed_bucket_aggregate_keys": list(
                self.allowed_bucket_aggregate_keys
            ),
            "forbidden_predictor_input_key_fragments": list(
                self.forbidden_predictor_input_key_fragments
            ),
            "shuffle_falsifier": self.shuffle_falsifier,
            "bucket_reports": [bucket.to_dict() for bucket in self.bucket_reports],
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


@dataclass(frozen=True)
class _StrictObservableTieMaskRow:
    state_key: str
    flat_index: int
    current_q_level: int
    move_direction: int
    vote_sign: int
    vote_value: int
    vote_abs: int
    abs_new_acc: int
    threshold_abs: int
    margin_abs_over_threshold: int
    replay_ce_veto_vote_sign: int | None = None
    replay_ce_veto_vote_value: int | None = None
    replay_ce_veto_move_sign: int | None = None
    pc_aux_vote_sign: int | None = None
    pc_aux_vote_value: int | None = None
    pc_aux_move_sign: int | None = None

    @property
    def identity(self) -> tuple[str, int]:
        return (self.state_key, int(self.flat_index))

    @property
    def bucket_key(self) -> tuple[str, int, int]:
        return (self.state_key, int(self.current_q_level), int(self.move_direction))

    def feature_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "vote_sign": int(self.vote_sign),
            "vote_value": int(self.vote_value),
            "vote_abs": int(self.vote_abs),
            "abs_new_acc": int(self.abs_new_acc),
            "threshold_abs": int(self.threshold_abs),
            "margin_abs_over_threshold": int(self.margin_abs_over_threshold),
        }
        optional_fields = {
            "replay_ce_veto_vote_sign": self.replay_ce_veto_vote_sign,
            "replay_ce_veto_vote_value": self.replay_ce_veto_vote_value,
            "replay_ce_veto_move_sign": self.replay_ce_veto_move_sign,
            "pc_aux_vote_sign": self.pc_aux_vote_sign,
            "pc_aux_vote_value": self.pc_aux_vote_value,
            "pc_aux_move_sign": self.pc_aux_move_sign,
        }
        for key, value in optional_fields.items():
            if value is not None:
                payload[key] = int(value)
        return payload

    def feature_key(self) -> tuple[tuple[str, Any], ...]:
        payload = self.feature_payload()
        return tuple((key, payload[key]) for key in sorted(payload))


def _sign_int(value: int | None) -> int | None:
    if value is None:
        return None
    value_i = int(value)
    if value_i > 0:
        return 1
    if value_i < 0:
        return -1
    return 0


def _flat_optional_int(tensor: torch.Tensor | None, flat_index: int) -> int | None:
    if tensor is None:
        return None
    return int(tensor.flatten()[int(flat_index)].item())


def _strict_observable_tie_mask_rows(
    trace_step: _ExactScheduleTraceStep,
) -> tuple[_StrictObservableTieMaskRow, ...]:
    cap_result = trace_step.exact_path.cap_result
    if cap_result is None:
        raise ValueError("online-estimability diagnostic requires global cap rows")
    inputs_by_state = {item.state_key: item for item in trace_step.inputs}
    rows: list[_StrictObservableTieMaskRow] = []
    for row in cap_result.rows:
        input_item = inputs_by_state[row.state_key]
        state = trace_step.exact_input_states[row.state_key]
        flat_index = int(row.flat_index)
        vote_value = int(input_item.vote_inputs.votes.flatten()[flat_index].item())
        rows.append(
            _StrictObservableTieMaskRow(
                state_key=row.state_key,
                flat_index=flat_index,
                current_q_level=int(state.q_levels.flatten()[flat_index].item()),
                move_direction=int(
                    trace_step.exact_path.plans[row.state_key]
                    .applied_directions[int(row.local_pos)]
                    .item()
                ),
                vote_sign=int(_sign_int(vote_value)),
                vote_value=int(vote_value),
                vote_abs=abs(int(vote_value)),
                abs_new_acc=int(row.abs_new_acc),
                threshold_abs=int(row.threshold_abs),
                margin_abs_over_threshold=int(row.margin_abs_over_threshold),
                replay_ce_veto_vote_sign=_sign_int(
                    _flat_optional_int(
                        input_item.vote_inputs.replay_ce_veto_votes,
                        flat_index,
                    )
                ),
                replay_ce_veto_vote_value=_flat_optional_int(
                    input_item.vote_inputs.replay_ce_veto_votes,
                    flat_index,
                ),
                replay_ce_veto_move_sign=_sign_int(
                    _flat_optional_int(
                        input_item.vote_inputs.replay_ce_veto_moves,
                        flat_index,
                    )
                ),
                pc_aux_vote_sign=_sign_int(
                    _flat_optional_int(input_item.vote_inputs.pc_aux_votes, flat_index)
                ),
                pc_aux_vote_value=_flat_optional_int(
                    input_item.vote_inputs.pc_aux_votes,
                    flat_index,
                ),
                pc_aux_move_sign=_sign_int(
                    _flat_optional_int(input_item.vote_inputs.pc_aux_moves, flat_index)
                ),
            )
        )
    return tuple(rows)


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


def _observable_tie_bucket_locator(
    schedule_name: str,
    state_key: str,
    current_q_level: int,
    move_direction: int,
) -> str:
    return (
        f"{schedule_name}:{state_key}/q{int(current_q_level)}/d{int(move_direction)}"
    )


def _strict_observable_tie_mask_bucket_report(
    *,
    source_bucket: TieFrontierObservedBucketReport,
    trace_step: _ExactScheduleTraceStep,
) -> ObservableTieMaskBucketReport:
    cap_result = trace_step.exact_path.cap_result
    if cap_result is None:
        raise ValueError("online-estimability diagnostic requires a cap-result trace")
    bucket_key = (
        source_bucket.state_key,
        int(source_bucket.current_q_level),
        int(source_bucket.move_direction),
    )
    observable_rows = [
        row
        for row in _strict_observable_tie_mask_rows(trace_step)
        if row.bucket_key == bucket_key
    ]
    if len(observable_rows) != int(source_bucket.candidate_row_count):
        raise ValueError("online-estimability bucket drifted from the observed tie-frontier candidates")
    exact_accepted_all = {
        (row.state_key, int(row.flat_index)) for row in cap_result.accepted_rows
    }
    exact_accepted = {
        row.identity for row in observable_rows if row.identity in exact_accepted_all
    }
    if len(exact_accepted) != int(source_bucket.accepted_row_count):
        raise ValueError("online-estimability bucket drifted from the exact accepted count")
    sorted_scores = sorted((row.abs_new_acc for row in observable_rows), reverse=True)
    accepted_row_count = int(source_bucket.accepted_row_count)
    boundary_score = int(sorted_scores[accepted_row_count - 1])
    higher_priority_row_count = sum(
        int(row.abs_new_acc) > boundary_score for row in observable_rows
    )
    residual_cap_slots_entering_bucket = (
        accepted_row_count - int(higher_priority_row_count)
    )
    if residual_cap_slots_entering_bucket < 0:
        raise ValueError("residual cap slots must not go negative")
    by_feature: dict[tuple[tuple[str, Any], ...], list[_StrictObservableTieMaskRow]] = {}
    for row in observable_rows:
        by_feature.setdefault(row.feature_key(), []).append(row)
    feature_class_reports: list[ObservableTieMaskFeatureClassReport] = []
    canonical_order_leaky_accepted: set[tuple[str, int]] = set()
    reversed_order_leaky_accepted: set[tuple[str, int]] = set()
    best_case_correct = 0
    best_case_hamming = 0
    mixed_feature_class_count = 0
    mixed_feature_class_row_count = 0
    exact_identity_free_recovery_possible = True
    for feature_key in sorted(by_feature):
        class_rows = by_feature[feature_key]
        exact_class_accepted = {
            row.identity for row in class_rows if row.identity in exact_accepted
        }
        class_row_count = len(class_rows)
        class_accepted_count = len(exact_class_accepted)
        class_deferred_count = class_row_count - class_accepted_count
        mixed_acceptance = bool(0 < class_accepted_count < class_row_count)
        if mixed_acceptance:
            mixed_feature_class_count += 1
            mixed_feature_class_row_count += class_row_count
            exact_identity_free_recovery_possible = False
        best_case_correct += max(class_accepted_count, class_deferred_count)
        best_case_hamming += min(class_accepted_count, class_deferred_count)
        canonical_prefix = {
            row.identity for row in class_rows[:class_accepted_count]
        }
        reversed_prefix = {
            row.identity
            for row in list(reversed(class_rows))[:class_accepted_count]
        }
        canonical_order_leaky_accepted |= canonical_prefix
        reversed_order_leaky_accepted |= reversed_prefix
        feature_class_reports.append(
            ObservableTieMaskFeatureClassReport(
                feature_payload=dict(class_rows[0].feature_payload()),
                row_count=int(class_row_count),
                accepted_count=int(class_accepted_count),
                deferred_count=int(class_deferred_count),
                mixed_acceptance=bool(mixed_acceptance),
                best_case_identity_free_correct_count=int(
                    max(class_accepted_count, class_deferred_count)
                ),
                best_case_identity_free_hamming_lower_bound=int(
                    min(class_accepted_count, class_deferred_count)
                ),
                canonical_prefix_matches_exact=bool(
                    canonical_prefix == exact_class_accepted
                ),
                reversed_prefix_matches_exact=bool(
                    reversed_prefix == exact_class_accepted
                ),
                exact_accepted_identities_sha256=_identity_sha256(exact_class_accepted),
                canonical_prefix_accepted_identities_sha256=_identity_sha256(
                    canonical_prefix
                ),
                reversed_prefix_accepted_identities_sha256=_identity_sha256(
                    reversed_prefix
                ),
            )
        )
    candidate_row_count = len(observable_rows)
    best_case_accuracy = float(best_case_correct) / float(candidate_row_count)
    canonical_matches_exact = canonical_order_leaky_accepted == exact_accepted
    reversed_matches_exact = reversed_order_leaky_accepted == exact_accepted
    within_class_reverse_order_changes = (
        canonical_order_leaky_accepted != reversed_order_leaky_accepted
    )
    order_dependence_witnessed = bool(
        canonical_matches_exact
        and not reversed_matches_exact
        and within_class_reverse_order_changes
    )
    return ObservableTieMaskBucketReport(
        schedule_name=source_bucket.schedule_name,
        step=int(source_bucket.step),
        state_key=source_bucket.state_key,
        current_q_level=int(source_bucket.current_q_level),
        move_direction=int(source_bucket.move_direction),
        global_cap=int(trace_step.cap_spec.cap),
        candidate_row_count=int(candidate_row_count),
        accepted_row_count=int(accepted_row_count),
        deferred_row_count=int(candidate_row_count - accepted_row_count),
        higher_priority_row_count=int(higher_priority_row_count),
        residual_cap_slots_entering_bucket=int(residual_cap_slots_entering_bucket),
        feature_class_reports=tuple(feature_class_reports),
        mixed_feature_class_count=int(mixed_feature_class_count),
        mixed_feature_class_row_count=int(mixed_feature_class_row_count),
        exact_identity_free_recovery_possible=bool(
            exact_identity_free_recovery_possible
        ),
        exact_mask_recovery_rate=1.0 if exact_identity_free_recovery_possible else 0.0,
        best_case_identity_free_correct_count=int(best_case_correct),
        best_case_identity_free_hamming_lower_bound=int(best_case_hamming),
        best_case_identity_free_mask_accuracy_upper_bound=float(best_case_accuracy),
        canonical_order_leaky_matches_exact=bool(canonical_matches_exact),
        reversed_order_leaky_matches_exact=bool(reversed_matches_exact),
        within_class_reverse_order_changes_order_leaky_mask=bool(
            within_class_reverse_order_changes
        ),
        order_dependence_witnessed=bool(order_dependence_witnessed),
        exact_accepted_identities_sha256=_identity_sha256(exact_accepted),
        canonical_order_leaky_accepted_identities_sha256=_identity_sha256(
            canonical_order_leaky_accepted
        ),
        reversed_order_leaky_accepted_identities_sha256=_identity_sha256(
            reversed_order_leaky_accepted
        ),
    )


def _strict_observable_tie_mask_terminal_decision(
    bucket_reports: Sequence[ObservableTieMaskBucketReport],
) -> ObservableTieMaskOnlineEstimabilityDecision:
    if not bucket_reports:
        raise ValueError("online-estimability diagnostic requires at least one decisive bucket")
    exact_recoverable_buckets = [
        bucket
        for bucket in bucket_reports
        if bucket.exact_identity_free_recovery_possible
        and not bucket.order_dependence_witnessed
    ]
    failed_buckets = [
        bucket
        for bucket in bucket_reports
        if not (
            bucket.exact_identity_free_recovery_possible
            and not bucket.order_dependence_witnessed
        )
    ]
    first_failure = failed_buckets[0] if failed_buckets else None
    any_mixed = any(
        int(bucket.mixed_feature_class_count) > 0 for bucket in bucket_reports
    )
    any_order_dependence = any(
        bool(bucket.order_dependence_witnessed) for bucket in bucket_reports
    )
    worst_bucket = min(
        bucket_reports,
        key=lambda bucket: bucket.best_case_identity_free_mask_accuracy_upper_bound,
    )
    first_failure_bucket = None
    if first_failure is not None:
        first_failure_bucket = _observable_tie_bucket_locator(
            first_failure.schedule_name,
            first_failure.state_key,
            int(first_failure.current_q_level),
            int(first_failure.move_direction),
        )
    if len(exact_recoverable_buckets) == len(bucket_reports):
        terminal_label = (
            STRICT_OBSERVABLE_TIE_MASK_EXACT_RECOVERABLE_IDENTITY_FREE_CANDIDATE_ONLY
        )
        reason = (
            "the pinned current-step observable schema exactly recovers every decisive tie bucket "
            "and survives the within-class reverse-order falsifier; the hybrid earns an "
            "implementation-design slice only"
        )
        online_realizable = True
        implementation_design_earned = True
        path_b_earned = False
    elif exact_recoverable_buckets:
        terminal_label = STRICT_OBSERVABLE_TIE_MASK_PARTIALLY_RECOVERABLE_NOT_EXACT
        reason = (
            "some decisive buckets are identity-free recoverable, but exact tie-mask recovery still "
            f"fails first at {first_failure_bucket}; the current state path still earns path (b)"
        )
        online_realizable = False
        implementation_design_earned = False
        path_b_earned = True
    else:
        terminal_label = STRICT_OBSERVABLE_TIE_MASK_NOT_IDENTIFIABLE_IDENTITY_BOUND
        reason = (
            "the strongest allowed current-step observable schema still leaves mixed equal-feature "
            f"classes first at {first_failure_bucket}; exact recovery is identity-bound and any "
            "apparent prefix match depends on stable cap order"
        )
        online_realizable = False
        implementation_design_earned = False
        path_b_earned = True
    return ObservableTieMaskOnlineEstimabilityDecision(
        terminal_label=terminal_label,
        decisive_bucket_count=int(len(bucket_reports)),
        exact_recoverable_bucket_count=int(len(exact_recoverable_buckets)),
        first_failure_bucket=first_failure_bucket,
        worst_bucket_best_case_identity_free_mask_accuracy_upper_bound=float(
            worst_bucket.best_case_identity_free_mask_accuracy_upper_bound
        ),
        any_mixed_feature_class_split=bool(any_mixed),
        any_order_dependence_witnessed=bool(any_order_dependence),
        online_realizable_candidate_hybrid=bool(online_realizable),
        implementation_design_earned=bool(implementation_design_earned),
        path_b_identity_free_redesign_earned=bool(path_b_earned),
        global_per_row_compression_closed=False,
        reason=reason,
    )


def _online_estimable_tie_mask_non_claims() -> tuple[str, ...]:
    return (
        "CPU-only strict observable/transient identifiability diagnostic over the committed exact cap trace",
        "negative is limited to this pinned observable schema and state path, not a global impossibility claim",
        "positive would earn implementation-design only; still candidate-only and not a learner claim",
        "global_per_row_compression_closed=false",
        "no dyn200, no GPU lane, no kernel path",
        "compact hashes and aggregate counts only; no raw per-weight arrays",
    )


def run_online_estimable_tie_mask_diagnostic() -> ObservableTieMaskOnlineEstimabilityReport:
    tie_report = run_tie_frontier_reservation_lower_bound_diagnostic()
    if tie_report.terminal_decision.terminal_label != TIE_FRONTIER_RESERVATION_FITS_HEADROOM_CANDIDATE_HYBRID:
        raise ValueError("online-estimability diagnostic requires the candidate-hybrid tie-frontier source")
    trace_steps, _ = _build_exact_schedule_trace()
    trace_by_name = {step.schedule_step.name: step for step in trace_steps}
    bucket_reports = tuple(
        _strict_observable_tie_mask_bucket_report(
            source_bucket=source_bucket,
            trace_step=trace_by_name[source_bucket.schedule_name],
        )
        for source_bucket in tie_report.observed_failing_bucket_reports
    )
    bindingness = pre_register_source_bindingness(
        source_kind=SOURCE_KIND_GENERATED_NATIVE_LOOP,
        coverage=SourceFieldCoverage.full_generated_native_loop(),
    )
    return ObservableTieMaskOnlineEstimabilityReport(
        schema_version=ONLINE_ESTIMABILITY_TIE_MASK_SCHEMA_VERSION,
        label=ONLINE_ESTIMABILITY_TIE_MASK_LABEL,
        source_bindingness=bindingness,
        field_coverage=SourceFieldCoverage.full_generated_native_loop(),
        candidate_name=ONLINE_ESTIMABLE_TIE_MASK_CANDIDATE,
        source_tie_frontier_reservation_label=tie_report.label,
        source_tie_frontier_reservation_terminal_label=(
            tie_report.terminal_decision.terminal_label
        ),
        source_decision_statistic_terminal_label=(
            tie_report.source_decision_statistic_terminal_label
        ),
        strictest_required_q_regime_name=tie_report.strictest_required_q_regime_name,
        strictest_headroom_bits_per_weight=float(
            tie_report.strictest_headroom_bits_per_weight
        ),
        allowed_bucket_key_dimensions=(
            STRICT_OBSERVABLE_TIE_MASK_ALLOWED_BUCKET_KEY_DIMENSIONS
        ),
        allowed_within_bucket_feature_keys=(
            STRICT_OBSERVABLE_TIE_MASK_ALLOWED_WITHIN_BUCKET_FEATURE_KEYS
        ),
        allowed_bucket_aggregate_keys=(
            STRICT_OBSERVABLE_TIE_MASK_ALLOWED_BUCKET_AGGREGATE_KEYS
        ),
        forbidden_predictor_input_key_fragments=(
            STRICT_OBSERVABLE_TIE_MASK_FORBIDDEN_PREDICTOR_INPUT_KEY_FRAGMENTS
        ),
        shuffle_falsifier=STRICT_OBSERVABLE_TIE_MASK_SHUFFLE_FALSIFIER,
        bucket_reports=bucket_reports,
        terminal_decision=_strict_observable_tie_mask_terminal_decision(
            bucket_reports
        ),
        raw_arrays_included=False,
        non_claims=_online_estimable_tie_mask_non_claims(),
    )


def _validate_online_estimable_tie_mask_feature_schema(
    report: ObservableTieMaskOnlineEstimabilityReport,
) -> None:
    if tuple(report.allowed_bucket_key_dimensions) != (
        STRICT_OBSERVABLE_TIE_MASK_ALLOWED_BUCKET_KEY_DIMENSIONS
    ):
        raise ValueError("online-estimability bucket-key schema drifted from the approved input set")
    if tuple(report.allowed_within_bucket_feature_keys) != (
        STRICT_OBSERVABLE_TIE_MASK_ALLOWED_WITHIN_BUCKET_FEATURE_KEYS
    ):
        raise ValueError("online-estimability feature-key schema drifted from the approved input set")
    if tuple(report.allowed_bucket_aggregate_keys) != (
        STRICT_OBSERVABLE_TIE_MASK_ALLOWED_BUCKET_AGGREGATE_KEYS
    ):
        raise ValueError("online-estimability aggregate-key schema drifted from the approved input set")
    if tuple(report.forbidden_predictor_input_key_fragments) != (
        STRICT_OBSERVABLE_TIE_MASK_FORBIDDEN_PREDICTOR_INPUT_KEY_FRAGMENTS
    ):
        raise ValueError("online-estimability forbidden-key fragments drifted from the co-lead gate")
    allowed_keys = (
        set(report.allowed_bucket_key_dimensions)
        | set(report.allowed_within_bucket_feature_keys)
        | set(report.allowed_bucket_aggregate_keys)
    )
    for key in allowed_keys:
        lowered = key.lower()
        if any(
            fragment in lowered
            for fragment in report.forbidden_predictor_input_key_fragments
        ):
            raise ValueError("online-estimability allowed schema leaked a forbidden predictor key")


def validate_online_estimable_tie_mask_report(
    report: ObservableTieMaskOnlineEstimabilityReport,
) -> None:
    if report.schema_version != ONLINE_ESTIMABILITY_TIE_MASK_SCHEMA_VERSION:
        raise ValueError("unexpected online-estimability schema version")
    if report.label != ONLINE_ESTIMABILITY_TIE_MASK_LABEL:
        raise ValueError("unexpected online-estimability label")
    if report.candidate_name != ONLINE_ESTIMABLE_TIE_MASK_CANDIDATE:
        raise ValueError("online-estimability candidate drifted from the branch-(a) hybrid tie mask")
    if report.source_tie_frontier_reservation_label != TIE_FRONTIER_RESERVATION_LABEL:
        raise ValueError("online-estimability diagnostic must cite the committed tie-frontier source")
    if report.source_tie_frontier_reservation_terminal_label != TIE_FRONTIER_RESERVATION_FITS_HEADROOM_CANDIDATE_HYBRID:
        raise ValueError("online-estimability diagnostic must inherit the candidate-hybrid-alive source")
    if report.source_decision_statistic_terminal_label != OBSERVABLE_RANK_FEATURES_INSUFFICIENT:
        raise ValueError("online-estimability diagnostic must keep the branch-(a) source lineage")
    if report.shuffle_falsifier != STRICT_OBSERVABLE_TIE_MASK_SHUFFLE_FALSIFIER:
        raise ValueError("online-estimability shuffle falsifier drifted from the plan gate")
    if report.terminal_decision.terminal_label not in {
        STRICT_OBSERVABLE_TIE_MASK_NOT_IDENTIFIABLE_IDENTITY_BOUND,
        STRICT_OBSERVABLE_TIE_MASK_EXACT_RECOVERABLE_IDENTITY_FREE_CANDIDATE_ONLY,
        STRICT_OBSERVABLE_TIE_MASK_PARTIALLY_RECOVERABLE_NOT_EXACT,
    }:
        raise ValueError("unexpected online-estimability terminal label")
    if bool(report.terminal_decision.global_per_row_compression_closed):
        raise ValueError("online-estimability diagnostic must not claim global closure")
    _validate_online_estimable_tie_mask_feature_schema(report)
    if not report.bucket_reports:
        raise ValueError("online-estimability diagnostic must report at least one decisive bucket")
    seen_bucket_keys: set[tuple[str, str, int, int]] = set()
    exact_recoverable_bucket_count = 0
    any_mixed = False
    any_order_dependence = False
    for bucket in report.bucket_reports:
        bucket_key = (
            bucket.schedule_name,
            bucket.state_key,
            int(bucket.current_q_level),
            int(bucket.move_direction),
        )
        if bucket_key in seen_bucket_keys:
            raise ValueError("online-estimability bucket keys must stay unique")
        seen_bucket_keys.add(bucket_key)
        if bucket.state_key not in PRIMARY_STATE_KEYS:
            raise ValueError("online-estimability bucket used an unknown state_key")
        if int(bucket.current_q_level) not in (-1, 0, 1):
            raise ValueError("online-estimability bucket used an invalid q level")
        if int(bucket.move_direction) not in (-1, 1):
            raise ValueError("online-estimability bucket used an invalid move direction")
        if bucket.candidate_row_count != bucket.accepted_row_count + bucket.deferred_row_count:
            raise ValueError("online-estimability bucket must partition every candidate row")
        if bucket.residual_cap_slots_entering_bucket != (
            bucket.accepted_row_count - bucket.higher_priority_row_count
        ):
            raise ValueError("online-estimability residual-cap slots drifted from accepted minus higher-priority rows")
        feature_row_total = sum(
            group.row_count for group in bucket.feature_class_reports
        )
        feature_accepted_total = sum(
            group.accepted_count for group in bucket.feature_class_reports
        )
        feature_deferred_total = sum(
            group.deferred_count for group in bucket.feature_class_reports
        )
        mixed_feature_class_count = sum(
            1 for group in bucket.feature_class_reports if group.mixed_acceptance
        )
        mixed_feature_class_row_count = sum(
            group.row_count for group in bucket.feature_class_reports if group.mixed_acceptance
        )
        best_case_correct = sum(
            group.best_case_identity_free_correct_count
            for group in bucket.feature_class_reports
        )
        best_case_hamming = sum(
            group.best_case_identity_free_hamming_lower_bound
            for group in bucket.feature_class_reports
        )
        if feature_row_total != bucket.candidate_row_count:
            raise ValueError("online-estimability feature classes must cover every bucket row")
        if feature_accepted_total != bucket.accepted_row_count:
            raise ValueError("online-estimability feature classes drifted from the exact accepted count")
        if feature_deferred_total != bucket.deferred_row_count:
            raise ValueError("online-estimability feature classes drifted from the exact deferred count")
        if mixed_feature_class_count != bucket.mixed_feature_class_count:
            raise ValueError("online-estimability mixed-class count drifted from the feature reports")
        if mixed_feature_class_row_count != bucket.mixed_feature_class_row_count:
            raise ValueError("online-estimability mixed-class row count drifted from the feature reports")
        if best_case_correct != bucket.best_case_identity_free_correct_count:
            raise ValueError("online-estimability best-case correct count drifted from the feature reports")
        if best_case_hamming != bucket.best_case_identity_free_hamming_lower_bound:
            raise ValueError("online-estimability best-case Hamming bound drifted from the feature reports")
        expected_accuracy = float(best_case_correct) / float(bucket.candidate_row_count)
        if abs(expected_accuracy - bucket.best_case_identity_free_mask_accuracy_upper_bound) > 1e-12:
            raise ValueError("online-estimability best-case accuracy upper bound drifted from the counts")
        expected_exact = mixed_feature_class_count == 0
        if bool(bucket.exact_identity_free_recovery_possible) != bool(expected_exact):
            raise ValueError("online-estimability exact-recovery flag drifted from the mixed-class test")
        if float(bucket.exact_mask_recovery_rate) != (1.0 if expected_exact else 0.0):
            raise ValueError("online-estimability exact-recovery rate drifted from the bucket verdict")
        if expected_exact and bucket.best_case_identity_free_hamming_lower_bound != 0:
            raise ValueError("exactly recoverable buckets must carry zero Hamming lower bound")
        if not expected_exact and bucket.best_case_identity_free_hamming_lower_bound <= 0:
            raise ValueError("non-identifiable buckets must carry a positive Hamming lower bound")
        for group in bucket.feature_class_reports:
            payload_keys = set(group.feature_payload)
            if not payload_keys:
                raise ValueError("online-estimability feature classes must carry an explicit payload")
            if not payload_keys <= set(report.allowed_within_bucket_feature_keys):
                raise ValueError("online-estimability feature payload used a non-approved key")
            for key in payload_keys:
                lowered = key.lower()
                if any(
                    fragment in lowered
                    for fragment in report.forbidden_predictor_input_key_fragments
                ):
                    raise ValueError("online-estimability feature payload leaked a forbidden predictor key")
            if group.row_count != group.accepted_count + group.deferred_count:
                raise ValueError("online-estimability feature-class counts must partition the class")
            if bool(group.mixed_acceptance) != bool(0 < group.accepted_count < group.row_count):
                raise ValueError("online-estimability mixed-acceptance flag drifted from the class counts")
            if group.best_case_identity_free_correct_count != max(
                group.accepted_count,
                group.deferred_count,
            ):
                raise ValueError("online-estimability best-case class correct count drifted from majority labeling")
            if group.best_case_identity_free_hamming_lower_bound != min(
                group.accepted_count,
                group.deferred_count,
            ):
                raise ValueError("online-estimability class Hamming bound drifted from majority labeling")
            if group.mixed_acceptance and (
                group.canonical_prefix_accepted_identities_sha256
                == group.reversed_prefix_accepted_identities_sha256
            ):
                raise ValueError("mixed feature classes must change under the reverse-order falsifier")
            if not group.mixed_acceptance and not (
                group.canonical_prefix_matches_exact
                and group.reversed_prefix_matches_exact
            ):
                raise ValueError("pure feature classes must stay exact under either within-class order")
        if bool(bucket.within_class_reverse_order_changes_order_leaky_mask) != bool(
            bucket.canonical_order_leaky_accepted_identities_sha256
            != bucket.reversed_order_leaky_accepted_identities_sha256
        ):
            raise ValueError("online-estimability order-change flag drifted from the bucket hashes")
        expected_order_dependence = bool(
            bucket.canonical_order_leaky_matches_exact
            and not bucket.reversed_order_leaky_matches_exact
            and bucket.within_class_reverse_order_changes_order_leaky_mask
        )
        if bool(bucket.order_dependence_witnessed) != expected_order_dependence:
            raise ValueError("online-estimability order-dependence witness drifted from the falsifier hashes")
        if bucket.exact_identity_free_recovery_possible and bucket.order_dependence_witnessed:
            raise ValueError("exactly recoverable buckets must not rely on order dependence")
        exact_recoverable_bucket_count += int(
            bucket.exact_identity_free_recovery_possible
            and not bucket.order_dependence_witnessed
        )
        any_mixed = any_mixed or bool(bucket.mixed_feature_class_count)
        any_order_dependence = any_order_dependence or bool(bucket.order_dependence_witnessed)
    failures = [
        bucket
        for bucket in report.bucket_reports
        if not (
            bucket.exact_identity_free_recovery_possible
            and not bucket.order_dependence_witnessed
        )
    ]
    first_failure_bucket = None
    if failures:
        first_failure = failures[0]
        first_failure_bucket = _observable_tie_bucket_locator(
            first_failure.schedule_name,
            first_failure.state_key,
            int(first_failure.current_q_level),
            int(first_failure.move_direction),
        )
    worst_bucket = min(
        report.bucket_reports,
        key=lambda bucket: bucket.best_case_identity_free_mask_accuracy_upper_bound,
    )
    if report.terminal_decision.decisive_bucket_count != len(report.bucket_reports):
        raise ValueError("online-estimability decisive-bucket count drifted from the report")
    if report.terminal_decision.exact_recoverable_bucket_count != exact_recoverable_bucket_count:
        raise ValueError("online-estimability exact-recoverable bucket count drifted from the report")
    if report.terminal_decision.first_failure_bucket != first_failure_bucket:
        raise ValueError("online-estimability first-failure bucket drifted from the report order")
    if abs(
        report.terminal_decision.worst_bucket_best_case_identity_free_mask_accuracy_upper_bound
        - worst_bucket.best_case_identity_free_mask_accuracy_upper_bound
    ) > 1e-12:
        raise ValueError("online-estimability worst-bucket accuracy bound drifted from the bucket reports")
    if bool(report.terminal_decision.any_mixed_feature_class_split) != bool(any_mixed):
        raise ValueError("online-estimability mixed-feature split flag drifted from the bucket reports")
    if bool(report.terminal_decision.any_order_dependence_witnessed) != bool(any_order_dependence):
        raise ValueError("online-estimability order-dependence flag drifted from the bucket reports")
    label = report.terminal_decision.terminal_label
    if label == STRICT_OBSERVABLE_TIE_MASK_EXACT_RECOVERABLE_IDENTITY_FREE_CANDIDATE_ONLY:
        if failures:
            raise ValueError("online-estimability exact-recoverable terminal requires every bucket to pass")
        if not (
            report.terminal_decision.online_realizable_candidate_hybrid
            and report.terminal_decision.implementation_design_earned
            and not report.terminal_decision.path_b_identity_free_redesign_earned
        ):
            raise ValueError("exact-recoverable terminal must earn implementation design and not path (b)")
    elif label == STRICT_OBSERVABLE_TIE_MASK_PARTIALLY_RECOVERABLE_NOT_EXACT:
        if not (0 < exact_recoverable_bucket_count < len(report.bucket_reports)):
            raise ValueError("partial terminal requires some but not all decisive buckets to be exactly recoverable")
        if (
            report.terminal_decision.online_realizable_candidate_hybrid
            or report.terminal_decision.implementation_design_earned
            or not report.terminal_decision.path_b_identity_free_redesign_earned
        ):
            raise ValueError("partial terminal must still reject online realizability and earn path (b)")
    else:
        if exact_recoverable_bucket_count != 0:
            raise ValueError("identity-bound terminal requires zero exactly recoverable decisive buckets")
        if (
            report.terminal_decision.online_realizable_candidate_hybrid
            or report.terminal_decision.implementation_design_earned
            or not report.terminal_decision.path_b_identity_free_redesign_earned
        ):
            raise ValueError("identity-bound terminal must reject online realizability and earn path (b)")
        if not (any_mixed or any_order_dependence):
            raise ValueError("identity-bound terminal requires a real mixed-class or order-dependence failure")
    _assert_no_tensors(report.to_dict())


@dataclass(frozen=True)
class PathBStepDeviationReport:
    schedule_name: str
    step: int
    global_cap: int
    oracle_accepted_count: int
    rule_accepted_count: int
    mixed_class_candidate_row_count: int
    mixed_class_oracle_accepted_count: int
    missed_oracle_accepts: int
    extra_accepts: int
    accepted_set_symmetric_difference: int
    cap_overflow: int
    cap_underfill: int
    q_delta_footprint_bound: int
    class_local: bool
    cap_pressure_effect: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_name": self.schedule_name,
            "step": int(self.step),
            "global_cap": int(self.global_cap),
            "oracle_accepted_count": int(self.oracle_accepted_count),
            "rule_accepted_count": int(self.rule_accepted_count),
            "mixed_class_candidate_row_count": int(self.mixed_class_candidate_row_count),
            "mixed_class_oracle_accepted_count": int(self.mixed_class_oracle_accepted_count),
            "missed_oracle_accepts": int(self.missed_oracle_accepts),
            "extra_accepts": int(self.extra_accepts),
            "accepted_set_symmetric_difference": int(
                self.accepted_set_symmetric_difference
            ),
            "cap_overflow": int(self.cap_overflow),
            "cap_underfill": int(self.cap_underfill),
            "q_delta_footprint_bound": int(self.q_delta_footprint_bound),
            "class_local": bool(self.class_local),
            "cap_pressure_effect": self.cap_pressure_effect,
        }


@dataclass(frozen=True)
class PathBDeviationVectorSummary:
    peak_missed_oracle_accepts: int
    peak_extra_accepts: int
    peak_accepted_set_symmetric_difference: int
    peak_cap_overflow: int
    peak_cap_underfill: int
    peak_q_delta_footprint_bound: int
    all_steps_class_local: bool
    cap_pressure_effects: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_missed_oracle_accepts": int(self.peak_missed_oracle_accepts),
            "peak_extra_accepts": int(self.peak_extra_accepts),
            "peak_accepted_set_symmetric_difference": int(
                self.peak_accepted_set_symmetric_difference
            ),
            "peak_cap_overflow": int(self.peak_cap_overflow),
            "peak_cap_underfill": int(self.peak_cap_underfill),
            "peak_q_delta_footprint_bound": int(self.peak_q_delta_footprint_bound),
            "all_steps_class_local": bool(self.all_steps_class_local),
            "cap_pressure_effects": list(self.cap_pressure_effects),
        }


@dataclass(frozen=True)
class PathBPersistentLedgerCharge:
    total_bits: int
    bits_per_eligible_weight: float
    bounded_under_strictest_headroom: bool
    purely_transient_recomputed: bool
    concurrent_class_count: int
    class_key_bits: int
    aggregate_payload_bits: int
    metadata_bits: int
    carry_bits: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_bits": int(self.total_bits),
            "bits_per_eligible_weight": float(self.bits_per_eligible_weight),
            "bounded_under_strictest_headroom": bool(
                self.bounded_under_strictest_headroom
            ),
            "purely_transient_recomputed": bool(self.purely_transient_recomputed),
            "concurrent_class_count": int(self.concurrent_class_count),
            "class_key_bits": int(self.class_key_bits),
            "aggregate_payload_bits": int(self.aggregate_payload_bits),
            "metadata_bits": int(self.metadata_bits),
            "carry_bits": int(self.carry_bits),
        }


@dataclass(frozen=True)
class PathBMechanismFamilyReport:
    family_name: str
    variant_name: str
    terminal_label: str
    uses_only_emitted_current_step_observables: bool
    acts_uniformly_per_equal_feature_class: bool
    requires_forbidden_identity_or_order: bool
    additional_emitted_observable_keys_checked: tuple[str, ...]
    step_deviation_reports: tuple[PathBStepDeviationReport, ...]
    deviation_vector: PathBDeviationVectorSummary | None
    persistent_ledger_charge: PathBPersistentLedgerCharge | None
    earned_downstream_test: str | None
    persistent_state_bits_delta: int | None
    why_not_oracle_mask: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_name": self.family_name,
            "variant_name": self.variant_name,
            "terminal_label": self.terminal_label,
            "uses_only_emitted_current_step_observables": bool(
                self.uses_only_emitted_current_step_observables
            ),
            "acts_uniformly_per_equal_feature_class": bool(
                self.acts_uniformly_per_equal_feature_class
            ),
            "requires_forbidden_identity_or_order": bool(
                self.requires_forbidden_identity_or_order
            ),
            "additional_emitted_observable_keys_checked": list(
                self.additional_emitted_observable_keys_checked
            ),
            "step_deviation_reports": [
                step.to_dict() for step in self.step_deviation_reports
            ],
            "deviation_vector": (
                None if self.deviation_vector is None else self.deviation_vector.to_dict()
            ),
            "persistent_ledger_charge": (
                None
                if self.persistent_ledger_charge is None
                else self.persistent_ledger_charge.to_dict()
            ),
            "earned_downstream_test": self.earned_downstream_test,
            "persistent_state_bits_delta": self.persistent_state_bits_delta,
            "why_not_oracle_mask": self.why_not_oracle_mask,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PathBClassifierDecision:
    candidate_variants: tuple[str, ...]
    negative_variants: tuple[str, ...]
    candidate_family_count: int
    negative_family_count: int
    dyn200_earned: bool
    oracle_mask_hybrid_revived: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_variants": list(self.candidate_variants),
            "negative_variants": list(self.negative_variants),
            "candidate_family_count": int(self.candidate_family_count),
            "negative_family_count": int(self.negative_family_count),
            "dyn200_earned": bool(self.dyn200_earned),
            "oracle_mask_hybrid_revived": bool(self.oracle_mask_hybrid_revived),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PathBIdentityFreeTieRuleClassifierReport:
    schema_version: str
    label: str
    source_bindingness: Any
    field_coverage: SourceFieldCoverage
    candidate_name: str
    source_online_estimability_label: str
    source_online_estimability_terminal_label: str
    strictest_required_q_regime_name: str
    strictest_headroom_bits_per_weight: float
    family_reports: tuple[PathBMechanismFamilyReport, ...]
    terminal_decision: PathBClassifierDecision
    raw_arrays_included: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "source_bindingness": self.source_bindingness.to_dict(),
            "field_coverage": self.field_coverage.to_dict(),
            "candidate_name": self.candidate_name,
            "source_online_estimability_label": self.source_online_estimability_label,
            "source_online_estimability_terminal_label": self.source_online_estimability_terminal_label,
            "strictest_required_q_regime_name": self.strictest_required_q_regime_name,
            "strictest_headroom_bits_per_weight": float(
                self.strictest_headroom_bits_per_weight
            ),
            "family_reports": [family.to_dict() for family in self.family_reports],
            "terminal_decision": self.terminal_decision.to_dict(),
            "raw_arrays_included": bool(self.raw_arrays_included),
            "non_claims": list(self.non_claims),
        }


@dataclass(frozen=True)
class _PathBStepContext:
    schedule_name: str
    step: int
    global_cap: int
    oracle_accepted_count: int
    non_mixed_oracle_accepted_count: int
    mixed_class_candidate_row_count: int
    mixed_class_oracle_accepted_count: int
    mixed_class_count: int


def _path_b_decisive_step_contexts(
    online_report: ObservableTieMaskOnlineEstimabilityReport,
) -> tuple[_PathBStepContext, ...]:
    decision_report = run_decision_statistic_upper_bound_diagnostic()
    decision_by_name = {step.schedule_name: step for step in decision_report.step_reports}
    by_name: dict[str, list[ObservableTieMaskBucketReport]] = {}
    for bucket in online_report.bucket_reports:
        by_name.setdefault(bucket.schedule_name, []).append(bucket)
    contexts: list[_PathBStepContext] = []
    for step_report in decision_report.step_reports:
        step_buckets = by_name.get(step_report.schedule_name)
        if not step_buckets:
            continue
        mixed_candidate_row_count = sum(
            int(bucket.candidate_row_count) for bucket in step_buckets
        )
        mixed_oracle_accepted_count = sum(
            int(bucket.accepted_row_count) for bucket in step_buckets
        )
        contexts.append(
            _PathBStepContext(
                schedule_name=step_report.schedule_name,
                step=int(step_report.step),
                global_cap=int(step_report.global_cap),
                oracle_accepted_count=int(step_report.accepted_row_count),
                non_mixed_oracle_accepted_count=int(step_report.accepted_row_count)
                - int(mixed_oracle_accepted_count),
                mixed_class_candidate_row_count=int(mixed_candidate_row_count),
                mixed_class_oracle_accepted_count=int(mixed_oracle_accepted_count),
                mixed_class_count=len(step_buckets),
            )
        )
    return tuple(contexts)


def _path_b_step_deviation_accept_all(
    context: _PathBStepContext,
) -> PathBStepDeviationReport:
    rule_accepted_count = (
        int(context.non_mixed_oracle_accepted_count)
        + int(context.mixed_class_candidate_row_count)
    )
    extra_accepts = max(0, rule_accepted_count - int(context.oracle_accepted_count))
    accepted_set_symmetric_difference = int(extra_accepts)
    return PathBStepDeviationReport(
        schedule_name=context.schedule_name,
        step=int(context.step),
        global_cap=int(context.global_cap),
        oracle_accepted_count=int(context.oracle_accepted_count),
        rule_accepted_count=int(rule_accepted_count),
        mixed_class_candidate_row_count=int(context.mixed_class_candidate_row_count),
        mixed_class_oracle_accepted_count=int(context.mixed_class_oracle_accepted_count),
        missed_oracle_accepts=0,
        extra_accepts=int(extra_accepts),
        accepted_set_symmetric_difference=int(accepted_set_symmetric_difference),
        cap_overflow=max(0, int(rule_accepted_count) - int(context.global_cap)),
        cap_underfill=0,
        q_delta_footprint_bound=int(accepted_set_symmetric_difference),
        class_local=True,
        cap_pressure_effect=(
            CAP_PRESSURE_FRONTIER_OVERFLOW_REQUIRES_ILLEGAL_SUBSET_SELECTION
        ),
    )


def _path_b_step_deviation_defer_all_no_backfill(
    context: _PathBStepContext,
) -> PathBStepDeviationReport:
    rule_accepted_count = int(context.non_mixed_oracle_accepted_count)
    missed_oracle_accepts = int(context.mixed_class_oracle_accepted_count)
    accepted_set_symmetric_difference = int(missed_oracle_accepts)
    return PathBStepDeviationReport(
        schedule_name=context.schedule_name,
        step=int(context.step),
        global_cap=int(context.global_cap),
        oracle_accepted_count=int(context.oracle_accepted_count),
        rule_accepted_count=int(rule_accepted_count),
        mixed_class_candidate_row_count=int(context.mixed_class_candidate_row_count),
        mixed_class_oracle_accepted_count=int(context.mixed_class_oracle_accepted_count),
        missed_oracle_accepts=int(missed_oracle_accepts),
        extra_accepts=0,
        accepted_set_symmetric_difference=int(accepted_set_symmetric_difference),
        cap_overflow=0,
        cap_underfill=max(0, int(context.global_cap) - int(rule_accepted_count)),
        q_delta_footprint_bound=int(accepted_set_symmetric_difference),
        class_local=True,
        cap_pressure_effect=CAP_PRESSURE_FRONTIER_ONLY_UNDERFILL_NO_REALLOCATION,
    )


def _path_b_deviation_vector_summary(
    step_reports: Sequence[PathBStepDeviationReport],
) -> PathBDeviationVectorSummary:
    if not step_reports:
        raise ValueError("path-(b) deviation summary requires at least one step report")
    return PathBDeviationVectorSummary(
        peak_missed_oracle_accepts=max(
            int(step.missed_oracle_accepts) for step in step_reports
        ),
        peak_extra_accepts=max(int(step.extra_accepts) for step in step_reports),
        peak_accepted_set_symmetric_difference=max(
            int(step.accepted_set_symmetric_difference) for step in step_reports
        ),
        peak_cap_overflow=max(int(step.cap_overflow) for step in step_reports),
        peak_cap_underfill=max(int(step.cap_underfill) for step in step_reports),
        peak_q_delta_footprint_bound=max(
            int(step.q_delta_footprint_bound) for step in step_reports
        ),
        all_steps_class_local=all(bool(step.class_local) for step in step_reports),
        cap_pressure_effects=tuple(
            dict.fromkeys(step.cap_pressure_effect for step in step_reports)
        ),
    )


def _path_b_zero_persistent_ledger_charge() -> PathBPersistentLedgerCharge:
    return PathBPersistentLedgerCharge(
        total_bits=0,
        bits_per_eligible_weight=0.0,
        bounded_under_strictest_headroom=True,
        purely_transient_recomputed=True,
        concurrent_class_count=0,
        class_key_bits=0,
        aggregate_payload_bits=0,
        metadata_bits=0,
        carry_bits=0,
    )


def _path_b_additional_emitted_feature_keys() -> tuple[str, ...]:
    trace_steps, _ = _build_exact_schedule_trace()
    emitted_keys: set[str] = set()
    for trace_step in trace_steps:
        if trace_step.schedule_step.name not in {"cap_saturated", "backlog_growth"}:
            continue
        for row in _strict_observable_tie_mask_rows(trace_step):
            emitted_keys |= set(row.feature_payload())
    return tuple(
        sorted(
            emitted_keys - set(STRICT_OBSERVABLE_TIE_MASK_ALLOWED_WITHIN_BUCKET_FEATURE_KEYS)
        )
    )


def _path_b_aggregate_state_class_key_bits() -> int:
    return (
        _enum_bit_width(len(PRIMARY_STATE_KEYS))
        + _enum_bit_width(3)
        + _enum_bit_width(2)
        + _enum_bit_width(3)
        + 16
        + 16
        + 32
        + 32
        + 32
    )


def _path_b_aggregate_state_ledger_charge(
    online_report: ObservableTieMaskOnlineEstimabilityReport,
) -> PathBPersistentLedgerCharge:
    contexts = _path_b_decisive_step_contexts(online_report)
    if not contexts:
        raise ValueError("path-(b) aggregate-state ledger charge requires decisive step contexts")
    strict_row = _q_ledger_row_by_name(online_report.strictest_required_q_regime_name)
    max_mixed_class_row_count = max(
        int(context.mixed_class_candidate_row_count / context.mixed_class_count)
        for context in contexts
        if int(context.mixed_class_count) > 0
    )
    concurrent_class_count = max(int(context.mixed_class_count) for context in contexts)
    class_key_bits = _path_b_aggregate_state_class_key_bits()
    aggregate_payload_bits = _count_bit_width(max_mixed_class_row_count) + _count_bit_width(
        max(int(context.global_cap) for context in contexts)
    )
    total_bits = (
        class_key_bits
        + aggregate_payload_bits
        + int(PATH_B_AGGREGATE_STATE_METADATA_BITS)
        + int(PATH_B_AGGREGATE_STATE_CARRY_BITS)
    ) * int(concurrent_class_count)
    bits_per_eligible_weight = float(total_bits) / float(int(strict_row.eligible_weight_count))
    return PathBPersistentLedgerCharge(
        total_bits=int(total_bits),
        bits_per_eligible_weight=float(bits_per_eligible_weight),
        bounded_under_strictest_headroom=bool(
            bits_per_eligible_weight
            <= float(online_report.strictest_headroom_bits_per_weight) + 1e-12
        ),
        purely_transient_recomputed=False,
        concurrent_class_count=int(concurrent_class_count),
        class_key_bits=int(class_key_bits),
        aggregate_payload_bits=int(aggregate_payload_bits),
        metadata_bits=int(PATH_B_AGGREGATE_STATE_METADATA_BITS),
        carry_bits=int(PATH_B_AGGREGATE_STATE_CARRY_BITS),
    )


def _path_b_class_action_accept_all_report(
    online_report: ObservableTieMaskOnlineEstimabilityReport,
) -> PathBMechanismFamilyReport:
    step_reports = tuple(
        _path_b_step_deviation_accept_all(context)
        for context in _path_b_decisive_step_contexts(online_report)
    )
    return PathBMechanismFamilyReport(
        family_name="class_action",
        variant_name=CLASS_ACTION_ACCEPT_ALL_MIXED_CLASSES,
        terminal_label=CANDIDATE_FAMILY_CLASS_UNIFORM_CAP_OVERFLOW_NEGATIVE,
        uses_only_emitted_current_step_observables=True,
        acts_uniformly_per_equal_feature_class=True,
        requires_forbidden_identity_or_order=False,
        additional_emitted_observable_keys_checked=(),
        step_deviation_reports=step_reports,
        deviation_vector=_path_b_deviation_vector_summary(step_reports),
        persistent_ledger_charge=None,
        earned_downstream_test=None,
        persistent_state_bits_delta=None,
        why_not_oracle_mask=None,
        reason=(
            "accepting every row in each mixed equal-feature class is identity-free and uniform, but it "
            "immediately overflows the global cap on the committed decisive buckets"
        ),
    )


def _path_b_class_action_defer_all_report(
    online_report: ObservableTieMaskOnlineEstimabilityReport,
) -> PathBMechanismFamilyReport:
    step_reports = tuple(
        _path_b_step_deviation_defer_all_no_backfill(context)
        for context in _path_b_decisive_step_contexts(online_report)
    )
    zero_charge = _path_b_zero_persistent_ledger_charge()
    return PathBMechanismFamilyReport(
        family_name="class_action",
        variant_name=CLASS_ACTION_DEFER_ALL_MIXED_CLASSES_NO_BACKFILL,
        terminal_label=CANDIDATE_FAMILY_CLASS_UNIFORM_BOUNDED_DEVIATION_CANDIDATE_ONLY,
        uses_only_emitted_current_step_observables=True,
        acts_uniformly_per_equal_feature_class=True,
        requires_forbidden_identity_or_order=False,
        additional_emitted_observable_keys_checked=(),
        step_deviation_reports=step_reports,
        deviation_vector=_path_b_deviation_vector_summary(step_reports),
        persistent_ledger_charge=zero_charge,
        earned_downstream_test=RUNTIME_TIE_RULE_MUTATION_PARITY_PROBE,
        persistent_state_bits_delta=0,
        why_not_oracle_mask=(
            "defer-all/no-backfill changes the tie rule itself and never attempts to recover the "
            "unidentifiable per-row oracle mask"
        ),
        reason=(
            "deferring every mixed equal-feature class with no downstream refill is deterministic, "
            "identity-free, and carries a bounded class-local deviation vector worth a runtime parity probe"
        ),
    )


def _path_b_emitted_observable_split_report(
    online_report: ObservableTieMaskOnlineEstimabilityReport,
) -> PathBMechanismFamilyReport:
    additional_keys = _path_b_additional_emitted_feature_keys()
    terminal = CANDIDATE_FAMILY_NO_EMITTED_IDENTITY_FREE_SPLIT_OBSERVABLE
    reason = (
        "the current approved observable schema already exhausts the emitted current-step cap/vote "
        "features used on the committed decisive buckets; there is no additional emitted identity-free "
        "observable left in this slice to split the mixed classes"
    )
    if additional_keys:
        terminal = CANDIDATE_FAMILY_EMITTED_IDENTITY_FREE_SPLIT_CANDIDATE_ONLY
        reason = (
            "an emitted current-step observable outside the current audited schema is available and would "
            "need a separate gated admission before this family could advance"
        )
    return PathBMechanismFamilyReport(
        family_name="emitted_observable_split",
        variant_name=STRICTLY_NEW_EMITTED_IDENTITY_FREE_OBSERVABLE_SPLIT,
        terminal_label=terminal,
        uses_only_emitted_current_step_observables=True,
        acts_uniformly_per_equal_feature_class=False,
        requires_forbidden_identity_or_order=False,
        additional_emitted_observable_keys_checked=additional_keys,
        step_deviation_reports=(),
        deviation_vector=None,
        persistent_ledger_charge=None,
        earned_downstream_test=None,
        persistent_state_bits_delta=None,
        why_not_oracle_mask=None,
        reason=reason,
    )


def _path_b_aggregate_state_redefinition_report(
    online_report: ObservableTieMaskOnlineEstimabilityReport,
) -> PathBMechanismFamilyReport:
    ledger_charge = _path_b_aggregate_state_ledger_charge(online_report)
    if ledger_charge.bounded_under_strictest_headroom:
        terminal = CANDIDATE_FAMILY_AGGREGATE_STATE_RUNTIME_SEMANTICS_UNSPECIFIED
        reason = (
            "the aggregate-state family has a bounded full persistent representation under the strict "
            "sub-2 headroom, but it does not advance in this slice because its induced runtime action "
            "semantics are still unspecified"
        )
    else:
        terminal = CANDIDATE_FAMILY_AGGREGATE_STATE_UNBOUNDED_PERSISTENT_BITS_NEGATIVE
        reason = (
            "the aggregate-state redefinition would replace an unidentifiable mask with over-budget or "
            "unbounded persistent state, so it does not advance"
        )
    return PathBMechanismFamilyReport(
        family_name="aggregate_state_redefinition",
        variant_name=AGGREGATE_STATE_REDEFINITION,
        terminal_label=terminal,
        uses_only_emitted_current_step_observables=True,
        acts_uniformly_per_equal_feature_class=True,
        requires_forbidden_identity_or_order=False,
        additional_emitted_observable_keys_checked=(),
        step_deviation_reports=(),
        deviation_vector=None,
        persistent_ledger_charge=ledger_charge,
        earned_downstream_test=None,
        persistent_state_bits_delta=None,
        why_not_oracle_mask=None,
        reason=reason,
    )


def _path_b_classifier_non_claims() -> tuple[str, ...]:
    return (
        "CPU-only analytic classifier over the committed c593a4d decisive buckets",
        "candidate advances earn only downstream tolerance probes, not learner success or dyn200",
        "class-action candidates change the tie rule rather than recovering the unidentifiable oracle mask",
        "aggregate-state candidates must charge the full persistent representation against the strict sub-2 headroom",
        "no oracle-mask hybrid revival",
        "no GPU lane, no kernel path, no raw per-weight arrays",
    )


def _path_b_classifier_terminal_decision(
    family_reports: Sequence[PathBMechanismFamilyReport],
) -> PathBClassifierDecision:
    candidate_variants = tuple(
        report.variant_name
        for report in family_reports
        if report.terminal_label in {
            CANDIDATE_FAMILY_CLASS_UNIFORM_BOUNDED_DEVIATION_CANDIDATE_ONLY,
            CANDIDATE_FAMILY_EMITTED_IDENTITY_FREE_SPLIT_CANDIDATE_ONLY,
        }
    )
    negative_variants = tuple(
        report.variant_name
        for report in family_reports
        if report.variant_name not in candidate_variants
    )
    return PathBClassifierDecision(
        candidate_variants=candidate_variants,
        negative_variants=negative_variants,
        candidate_family_count=len(candidate_variants),
        negative_family_count=len(negative_variants),
        dyn200_earned=False,
        oracle_mask_hybrid_revived=False,
        reason=(
            "path-(b) classification complete: only identity-free rules with bounded, explicit deviation "
            "vectors and bounded persistent-state charges advance to downstream tolerance probes"
        ),
    )


def run_path_b_identity_free_tie_rule_classifier() -> PathBIdentityFreeTieRuleClassifierReport:
    online_report = run_online_estimable_tie_mask_diagnostic()
    if online_report.terminal_decision.terminal_label != STRICT_OBSERVABLE_TIE_MASK_NOT_IDENTIFIABLE_IDENTITY_BOUND:
        raise ValueError("path-(b) classifier requires the committed online-estimability identity-bound source")
    family_reports = (
        _path_b_class_action_accept_all_report(online_report),
        _path_b_class_action_defer_all_report(online_report),
        _path_b_emitted_observable_split_report(online_report),
        _path_b_aggregate_state_redefinition_report(online_report),
    )
    bindingness = pre_register_source_bindingness(
        source_kind=SOURCE_KIND_GENERATED_NATIVE_LOOP,
        coverage=SourceFieldCoverage.full_generated_native_loop(),
    )
    return PathBIdentityFreeTieRuleClassifierReport(
        schema_version=PATH_B_IDENTITY_FREE_TIE_RULE_CLASSIFIER_SCHEMA_VERSION,
        label=PATH_B_IDENTITY_FREE_TIE_RULE_CLASSIFIER_LABEL,
        source_bindingness=bindingness,
        field_coverage=SourceFieldCoverage.full_generated_native_loop(),
        candidate_name=PATH_B_IDENTITY_FREE_TIE_RULE_CLASSIFIER_CANDIDATE,
        source_online_estimability_label=online_report.label,
        source_online_estimability_terminal_label=(
            online_report.terminal_decision.terminal_label
        ),
        strictest_required_q_regime_name=online_report.strictest_required_q_regime_name,
        strictest_headroom_bits_per_weight=float(
            online_report.strictest_headroom_bits_per_weight
        ),
        family_reports=family_reports,
        terminal_decision=_path_b_classifier_terminal_decision(family_reports),
        raw_arrays_included=False,
        non_claims=_path_b_classifier_non_claims(),
    )


def validate_path_b_identity_free_tie_rule_classifier_report(
    report: PathBIdentityFreeTieRuleClassifierReport,
) -> None:
    if report.schema_version != PATH_B_IDENTITY_FREE_TIE_RULE_CLASSIFIER_SCHEMA_VERSION:
        raise ValueError("unexpected path-(b) classifier schema version")
    if report.label != PATH_B_IDENTITY_FREE_TIE_RULE_CLASSIFIER_LABEL:
        raise ValueError("unexpected path-(b) classifier label")
    if report.candidate_name != PATH_B_IDENTITY_FREE_TIE_RULE_CLASSIFIER_CANDIDATE:
        raise ValueError("path-(b) classifier candidate drifted")
    if report.source_online_estimability_label != ONLINE_ESTIMABILITY_TIE_MASK_LABEL:
        raise ValueError("path-(b) classifier must cite the committed online-estimability source")
    if report.source_online_estimability_terminal_label != STRICT_OBSERVABLE_TIE_MASK_NOT_IDENTIFIABLE_IDENTITY_BOUND:
        raise ValueError("path-(b) classifier must inherit the identity-bound online-estimability source")
    expected_variants = (
        CLASS_ACTION_ACCEPT_ALL_MIXED_CLASSES,
        CLASS_ACTION_DEFER_ALL_MIXED_CLASSES_NO_BACKFILL,
        STRICTLY_NEW_EMITTED_IDENTITY_FREE_OBSERVABLE_SPLIT,
        AGGREGATE_STATE_REDEFINITION,
    )
    actual_variants = tuple(report_entry.variant_name for report_entry in report.family_reports)
    if actual_variants != expected_variants:
        raise ValueError("path-(b) classifier family ordering drifted from the gated plan")
    by_variant = {entry.variant_name: entry for entry in report.family_reports}

    accept_all = by_variant[CLASS_ACTION_ACCEPT_ALL_MIXED_CLASSES]
    if accept_all.terminal_label != CANDIDATE_FAMILY_CLASS_UNIFORM_CAP_OVERFLOW_NEGATIVE:
        raise ValueError("accept-all class-action rule must stay a cap-overflow negative on the committed trace")
    if accept_all.earned_downstream_test is not None or accept_all.persistent_state_bits_delta is not None:
        raise ValueError("negative accept-all class-action rule must not earn a downstream test or persistent delta")
    if len(accept_all.step_deviation_reports) != 2:
        raise ValueError("accept-all class-action rule must report both decisive steps")
    if any(step.cap_overflow <= 0 for step in accept_all.step_deviation_reports):
        raise ValueError("accept-all class-action rule must overflow the cap on every decisive step")
    if any(
        step.cap_pressure_effect != CAP_PRESSURE_FRONTIER_OVERFLOW_REQUIRES_ILLEGAL_SUBSET_SELECTION
        for step in accept_all.step_deviation_reports
    ):
        raise ValueError("accept-all class-action rule must carry the overflow cap-pressure effect")

    defer_all = by_variant[CLASS_ACTION_DEFER_ALL_MIXED_CLASSES_NO_BACKFILL]
    if defer_all.terminal_label != CANDIDATE_FAMILY_CLASS_UNIFORM_BOUNDED_DEVIATION_CANDIDATE_ONLY:
        raise ValueError("defer-all/no-backfill class-action rule must stay the bounded-deviation candidate")
    if defer_all.earned_downstream_test != RUNTIME_TIE_RULE_MUTATION_PARITY_PROBE:
        raise ValueError("defer-all/no-backfill class-action rule must earn the runtime parity probe")
    if defer_all.persistent_state_bits_delta != 0:
        raise ValueError("defer-all/no-backfill class-action rule must carry zero persistent-state bits delta")
    if defer_all.why_not_oracle_mask is None:
        raise ValueError("candidate class-action rule must emit the why-not-oracle-mask non-claim")
    if defer_all.persistent_ledger_charge is None or not defer_all.persistent_ledger_charge.purely_transient_recomputed:
        raise ValueError("candidate class-action rule must explicitly declare its zero-bit transient ledger charge")
    if defer_all.deviation_vector is None:
        raise ValueError("candidate class-action rule must emit the full deviation vector")
    if any(step.cap_underfill <= 0 for step in defer_all.step_deviation_reports):
        raise ValueError("defer-all/no-backfill class-action rule must underfill the cap on every decisive step")
    if any(step.extra_accepts != 0 for step in defer_all.step_deviation_reports):
        raise ValueError("defer-all/no-backfill class-action rule must not add extra accepts on the committed trace")
    if any(
        step.cap_pressure_effect != CAP_PRESSURE_FRONTIER_ONLY_UNDERFILL_NO_REALLOCATION
        for step in defer_all.step_deviation_reports
    ):
        raise ValueError("defer-all/no-backfill class-action rule must declare the no-reallocation cap-pressure effect")

    emitted_split = by_variant[STRICTLY_NEW_EMITTED_IDENTITY_FREE_OBSERVABLE_SPLIT]
    if emitted_split.terminal_label != CANDIDATE_FAMILY_NO_EMITTED_IDENTITY_FREE_SPLIT_OBSERVABLE:
        raise ValueError("family-(2) must stay negative until a real emitted split observable exists")
    if emitted_split.additional_emitted_observable_keys_checked:
        raise ValueError("family-(2) should have no extra emitted split observables on the committed trace")
    if emitted_split.earned_downstream_test is not None:
        raise ValueError("family-(2) negative must not earn a downstream test")

    aggregate = by_variant[AGGREGATE_STATE_REDEFINITION]
    if aggregate.terminal_label not in {
        CANDIDATE_FAMILY_AGGREGATE_STATE_RUNTIME_SEMANTICS_UNSPECIFIED,
        CANDIDATE_FAMILY_AGGREGATE_STATE_UNBOUNDED_PERSISTENT_BITS_NEGATIVE,
    }:
        raise ValueError("family-(3) must stay non-advancing until concrete runtime semantics are defined")
    if aggregate.earned_downstream_test is not None:
        raise ValueError("family-(3) must not earn a downstream test while runtime semantics are unspecified")
    if aggregate.persistent_state_bits_delta is not None:
        raise ValueError("family-(3) must not claim a persistent-state delta advance while runtime semantics are unspecified")
    if aggregate.persistent_ledger_charge is None:
        raise ValueError("family-(3) feasibility note must emit the full persistent ledger charge")
    if aggregate.persistent_ledger_charge.total_bits <= 0:
        raise ValueError("family-(3) feasibility note must charge a positive persistent aggregate representation")
    if aggregate.terminal_label == CANDIDATE_FAMILY_AGGREGATE_STATE_RUNTIME_SEMANTICS_UNSPECIFIED and not aggregate.persistent_ledger_charge.bounded_under_strictest_headroom:
        raise ValueError("family-(3) bounded feasibility note must stay under the strictest headroom")
    if aggregate.deviation_vector is not None:
        raise ValueError("family-(3) must not borrow a runtime deviation vector before defining runtime semantics")
    if aggregate.step_deviation_reports:
        raise ValueError("family-(3) must not borrow step-level runtime deviation reports before defining runtime semantics")
    if aggregate.why_not_oracle_mask is not None:
        raise ValueError("family-(3) should stay a feasibility note rather than a candidate rule in this slice")

    candidate_variants = {
        report_entry.variant_name
        for report_entry in report.family_reports
        if report_entry.terminal_label in {
            CANDIDATE_FAMILY_CLASS_UNIFORM_BOUNDED_DEVIATION_CANDIDATE_ONLY,
            CANDIDATE_FAMILY_EMITTED_IDENTITY_FREE_SPLIT_CANDIDATE_ONLY,
        }
    }
    negative_variants = {
        report_entry.variant_name
        for report_entry in report.family_reports
        if report_entry.variant_name not in candidate_variants
    }
    if tuple(report.terminal_decision.candidate_variants) != tuple(
        report_entry.variant_name
        for report_entry in report.family_reports
        if report_entry.variant_name in candidate_variants
    ):
        raise ValueError("path-(b) classifier candidate list drifted from the family terminals")
    if tuple(report.terminal_decision.negative_variants) != tuple(
        report_entry.variant_name
        for report_entry in report.family_reports
        if report_entry.variant_name in negative_variants
    ):
        raise ValueError("path-(b) classifier negative list drifted from the family terminals")
    if report.terminal_decision.candidate_family_count != len(candidate_variants):
        raise ValueError("path-(b) classifier candidate-family count drifted from the family terminals")
    if report.terminal_decision.negative_family_count != len(negative_variants):
        raise ValueError("path-(b) classifier negative-family count drifted from the family terminals")
    if report.terminal_decision.dyn200_earned or report.terminal_decision.oracle_mask_hybrid_revived:
        raise ValueError("path-(b) classifier must not claim dyn200 or revive the oracle-mask hybrid")
    _assert_no_tensors(report.to_dict())


@dataclass(frozen=True)
class PathBDeferAllBaselineStepReport:
    schedule_name: str
    step: int
    global_cap: int
    exact_candidate_row_count: int
    replay_candidate_row_count: int
    exact_accepted_count: int
    replay_accepted_count: int
    accepted_surface_symmetric_difference: int
    exact_deferred_count: int
    replay_deferred_count: int
    deferred_surface_symmetric_difference: int
    exact_backlog_count: int
    replay_backlog_count: int
    backlog_surface_symmetric_difference: int
    q_divergence_count: int
    mixed_feature_class_count: int
    mixed_feature_class_row_count: int
    max_mixed_class_cardinality: int
    dropped_mass_count: int
    exact_accepted_identities_sha256: str
    replay_accepted_identities_sha256: str
    exact_deferred_identities_sha256: str
    replay_deferred_identities_sha256: str
    exact_backlog_identities_sha256: str
    replay_backlog_identities_sha256: str
    dropped_mass_identities_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_name": self.schedule_name,
            "step": int(self.step),
            "global_cap": int(self.global_cap),
            "exact_candidate_row_count": int(self.exact_candidate_row_count),
            "replay_candidate_row_count": int(self.replay_candidate_row_count),
            "exact_accepted_count": int(self.exact_accepted_count),
            "replay_accepted_count": int(self.replay_accepted_count),
            "accepted_surface_symmetric_difference": int(
                self.accepted_surface_symmetric_difference
            ),
            "exact_deferred_count": int(self.exact_deferred_count),
            "replay_deferred_count": int(self.replay_deferred_count),
            "deferred_surface_symmetric_difference": int(
                self.deferred_surface_symmetric_difference
            ),
            "exact_backlog_count": int(self.exact_backlog_count),
            "replay_backlog_count": int(self.replay_backlog_count),
            "backlog_surface_symmetric_difference": int(
                self.backlog_surface_symmetric_difference
            ),
            "q_divergence_count": int(self.q_divergence_count),
            "mixed_feature_class_count": int(self.mixed_feature_class_count),
            "mixed_feature_class_row_count": int(self.mixed_feature_class_row_count),
            "max_mixed_class_cardinality": int(self.max_mixed_class_cardinality),
            "dropped_mass_count": int(self.dropped_mass_count),
            "exact_accepted_identities_sha256": self.exact_accepted_identities_sha256,
            "replay_accepted_identities_sha256": self.replay_accepted_identities_sha256,
            "exact_deferred_identities_sha256": self.exact_deferred_identities_sha256,
            "replay_deferred_identities_sha256": self.replay_deferred_identities_sha256,
            "exact_backlog_identities_sha256": self.exact_backlog_identities_sha256,
            "replay_backlog_identities_sha256": self.replay_backlog_identities_sha256,
            "dropped_mass_identities_sha256": self.dropped_mass_identities_sha256,
        }


@dataclass(frozen=True)
class PathBDroppedMassOriginReport:
    origin_schedule_name: str
    origin_step: int
    dropped_mass_count: int
    dropped_mass_identities_sha256: str
    re_presented_later_count: int
    re_presented_later_identities_sha256: str
    eventually_accepted_under_baseline_count: int
    eventually_accepted_under_baseline_identities_sha256: str
    recoverable_but_unrecovered_count: int
    recoverable_but_unrecovered_identities_sha256: str
    never_recovered_count: int
    never_recovered_identities_sha256: str
    terminal_censored_mass_count: int
    terminal_censored_mass_identities_sha256: str
    future_schedule_names: tuple[str, ...]
    bounded_class_count_upper_bound: int
    max_bounded_class_cardinality: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_schedule_name": self.origin_schedule_name,
            "origin_step": int(self.origin_step),
            "dropped_mass_count": int(self.dropped_mass_count),
            "dropped_mass_identities_sha256": self.dropped_mass_identities_sha256,
            "re_presented_later_count": int(self.re_presented_later_count),
            "re_presented_later_identities_sha256": self.re_presented_later_identities_sha256,
            "eventually_accepted_under_baseline_count": int(
                self.eventually_accepted_under_baseline_count
            ),
            "eventually_accepted_under_baseline_identities_sha256": (
                self.eventually_accepted_under_baseline_identities_sha256
            ),
            "recoverable_but_unrecovered_count": int(
                self.recoverable_but_unrecovered_count
            ),
            "recoverable_but_unrecovered_identities_sha256": (
                self.recoverable_but_unrecovered_identities_sha256
            ),
            "never_recovered_count": int(self.never_recovered_count),
            "never_recovered_identities_sha256": self.never_recovered_identities_sha256,
            "terminal_censored_mass_count": int(self.terminal_censored_mass_count),
            "terminal_censored_mass_identities_sha256": (
                self.terminal_censored_mass_identities_sha256
            ),
            "future_schedule_names": list(self.future_schedule_names),
            "bounded_class_count_upper_bound": int(
                self.bounded_class_count_upper_bound
            ),
            "max_bounded_class_cardinality": int(
                self.max_bounded_class_cardinality
            ),
        }


@dataclass(frozen=True)
class PathBDeferAllBaselineDecision:
    terminal_label: str
    final_step_schedule_name: str
    final_q_divergence_count: int
    final_accepted_surface_symmetric_difference: int
    final_deferred_surface_symmetric_difference: int
    final_backlog_surface_symmetric_difference: int
    total_dropped_mass_count: int
    total_re_presented_later_count: int
    total_eventually_accepted_under_baseline_count: int
    total_recoverable_but_unrecovered_count: int
    total_never_recovered_count: int
    total_terminal_censored_mass_count: int
    peak_bounded_class_count_upper_bound: int
    peak_bounded_class_cardinality: int
    aggregate_runtime_semantics_definition_plan_earned: bool
    candidate_only: bool
    dyn200_earned: bool
    learner_sub2_claimed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "terminal_label": self.terminal_label,
            "final_step_schedule_name": self.final_step_schedule_name,
            "final_q_divergence_count": int(self.final_q_divergence_count),
            "final_accepted_surface_symmetric_difference": int(
                self.final_accepted_surface_symmetric_difference
            ),
            "final_deferred_surface_symmetric_difference": int(
                self.final_deferred_surface_symmetric_difference
            ),
            "final_backlog_surface_symmetric_difference": int(
                self.final_backlog_surface_symmetric_difference
            ),
            "total_dropped_mass_count": int(self.total_dropped_mass_count),
            "total_re_presented_later_count": int(
                self.total_re_presented_later_count
            ),
            "total_eventually_accepted_under_baseline_count": int(
                self.total_eventually_accepted_under_baseline_count
            ),
            "total_recoverable_but_unrecovered_count": int(
                self.total_recoverable_but_unrecovered_count
            ),
            "total_never_recovered_count": int(self.total_never_recovered_count),
            "total_terminal_censored_mass_count": int(
                self.total_terminal_censored_mass_count
            ),
            "peak_bounded_class_count_upper_bound": int(
                self.peak_bounded_class_count_upper_bound
            ),
            "peak_bounded_class_cardinality": int(
                self.peak_bounded_class_cardinality
            ),
            "aggregate_runtime_semantics_definition_plan_earned": bool(
                self.aggregate_runtime_semantics_definition_plan_earned
            ),
            "candidate_only": bool(self.candidate_only),
            "dyn200_earned": bool(self.dyn200_earned),
            "learner_sub2_claimed": bool(self.learner_sub2_claimed),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PathBDeferAllBaselineParityProbeReport:
    schema_version: str
    label: str
    source_bindingness: Any
    field_coverage: SourceFieldCoverage
    candidate_name: str
    source_classifier_label: str
    source_classifier_candidate_variant: str
    source_classifier_downstream_test: str
    step_reports: tuple[PathBDeferAllBaselineStepReport, ...]
    dropped_mass_origin_reports: tuple[PathBDroppedMassOriginReport, ...]
    terminal_decision: PathBDeferAllBaselineDecision
    raw_arrays_included: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "source_bindingness": self.source_bindingness.to_dict(),
            "field_coverage": self.field_coverage.to_dict(),
            "candidate_name": self.candidate_name,
            "source_classifier_label": self.source_classifier_label,
            "source_classifier_candidate_variant": self.source_classifier_candidate_variant,
            "source_classifier_downstream_test": self.source_classifier_downstream_test,
            "step_reports": [step.to_dict() for step in self.step_reports],
            "dropped_mass_origin_reports": [
                report.to_dict() for report in self.dropped_mass_origin_reports
            ],
            "terminal_decision": self.terminal_decision.to_dict(),
            "raw_arrays_included": bool(self.raw_arrays_included),
            "non_claims": list(self.non_claims),
        }


@dataclass(frozen=True)
class _PathBReplayStepState:
    schedule_name: str
    step: int
    global_cap: int
    candidate_row_ids: set[tuple[str, int]]
    accepted_ids: set[tuple[str, int]]
    deferred_ids: set[tuple[str, int]]
    backlog_ids: set[tuple[str, int]]
    output_states: dict[str, VoteUpdateState]
    output_backlog: dict[str, dict[int, dict[str, int]]]
    q_changed_ids: set[tuple[str, int]]
    dropped_ids: set[tuple[str, int]]
    mixed_feature_class_count: int
    mixed_feature_class_row_count: int
    max_mixed_class_cardinality: int
    observable_rows: tuple[_StrictObservableTieMaskRow, ...]


def _path_b_ids_from_indices(
    state_key: str,
    indices: torch.Tensor,
) -> set[tuple[str, int]]:
    return {
        (str(state_key), int(index))
        for index in indices.detach().cpu().to(torch.int64).tolist()
    }


def _path_b_apply_threshold_residual(
    new_acc_i32: torch.Tensor,
    indices: torch.Tensor,
    directions: torch.Tensor,
    thresholds: torch.Tensor,
) -> None:
    if indices.numel() == 0:
        return
    residual = new_acc_i32[indices] - directions.to(torch.int32) * thresholds
    low = -thresholds + 1
    high = thresholds - 1
    new_acc_i32[indices] = torch.minimum(torch.maximum(residual, low), high)


def _path_b_current_trace_step(
    *,
    states_by_key: Mapping[str, VoteUpdateState],
    deferred_backlog: Mapping[str, Mapping[int, Mapping[str, int]]],
    schedule_step: VotePressureStepSpec,
) -> _ExactScheduleTraceStep:
    inputs, offsets = _make_step_inputs(states_by_key, schedule_step)
    cap_spec = GlobalRateCapSpec(cap=int(schedule_step.cap), step=int(schedule_step.step))
    exact_path = _run_reference_path(
        inputs,
        states_by_key=_copy_state_map(states_by_key),
        global_cap_spec=cap_spec,
        deferred_backlog=_copy_backlog(deferred_backlog),
        tensor_offsets=offsets,
    )
    if exact_path.cap_result is None:
        raise ValueError("path-(b) baseline replay requires cap-result exact paths")
    return _ExactScheduleTraceStep(
        schedule_step=schedule_step,
        inputs=tuple(inputs),
        tensor_offsets=dict(offsets),
        cap_spec=cap_spec,
        exact_input_states=_copy_state_map(states_by_key),
        exact_input_backlog=_copy_backlog(deferred_backlog),
        exact_path=exact_path,
        exact_output_backlog=_copy_backlog(exact_path.cap_result.deferred_backlog),
    )


def _path_b_mutated_partition_for_trace_step(
    trace_step: _ExactScheduleTraceStep,
) -> tuple[
    tuple[_StrictObservableTieMaskRow, ...],
    set[tuple[str, int]],
    set[tuple[str, int]],
    set[tuple[str, int]],
    int,
    int,
    int,
]:
    cap_result = trace_step.exact_path.cap_result
    if cap_result is None:
        raise ValueError("path-(b) baseline replay requires a cap-result trace")
    observable_rows = _strict_observable_tie_mask_rows(trace_step)
    exact_accepted = {
        (row.state_key, int(row.flat_index)) for row in cap_result.accepted_rows
    }
    exact_deferred = {
        (row.state_key, int(row.flat_index)) for row in cap_result.deferred_rows
    }
    by_bucket: dict[tuple[str, int, int], list[_StrictObservableTieMaskRow]] = {}
    for row in observable_rows:
        by_bucket.setdefault(row.bucket_key, []).append(row)
    dropped_ids: set[tuple[str, int]] = set()
    mixed_feature_class_count = 0
    mixed_feature_class_row_count = 0
    max_mixed_class_cardinality = 0
    for bucket_rows in by_bucket.values():
        by_feature: dict[tuple[tuple[str, Any], ...], list[_StrictObservableTieMaskRow]] = {}
        for row in bucket_rows:
            by_feature.setdefault(row.feature_key(), []).append(row)
        for class_rows in by_feature.values():
            class_ids = {row.identity for row in class_rows}
            class_accepted = class_ids & exact_accepted
            if 0 < len(class_accepted) < len(class_ids):
                mixed_feature_class_count += 1
                mixed_feature_class_row_count += len(class_ids)
                max_mixed_class_cardinality = max(
                    max_mixed_class_cardinality,
                    len(class_ids),
                )
                dropped_ids |= class_accepted
    replay_accepted = set(exact_accepted) - set(dropped_ids)
    replay_deferred = set(exact_deferred) | set(dropped_ids)
    return (
        observable_rows,
        replay_accepted,
        replay_deferred,
        dropped_ids,
        int(mixed_feature_class_count),
        int(mixed_feature_class_row_count),
        int(max_mixed_class_cardinality),
    )


def _path_b_replay_step_from_mutated_partition(
    trace_step: _ExactScheduleTraceStep,
) -> _PathBReplayStepState:
    cap_result = trace_step.exact_path.cap_result
    if cap_result is None:
        raise ValueError("path-(b) baseline replay requires a cap-result trace")
    (
        observable_rows,
        replay_accepted,
        replay_deferred,
        dropped_ids,
        mixed_feature_class_count,
        mixed_feature_class_row_count,
        max_mixed_class_cardinality,
    ) = _path_b_mutated_partition_for_trace_step(trace_step)
    backlog = _copy_backlog(trace_step.exact_input_backlog)
    for row in cap_result.rows:
        identity = row.identity()
        if identity in replay_accepted:
            state_backlog = backlog.get(row.state_key, {})
            if row.flat_index in state_backlog:
                del state_backlog[row.flat_index]
        if identity in replay_deferred:
            state_backlog = backlog.setdefault(row.state_key, {})
            entry = state_backlog.setdefault(
                int(row.flat_index),
                {
                    "first_step": int(trace_step.schedule_step.step),
                    "last_deferred_step": int(trace_step.schedule_step.step),
                    "defer_count": 0,
                },
            )
            entry["last_deferred_step"] = int(trace_step.schedule_step.step)
            entry["defer_count"] = int(entry.get("defer_count", 0)) + 1
    output_states: dict[str, VoteUpdateState] = {}
    q_changed_ids: set[tuple[str, int]] = set()
    for item in trace_step.inputs:
        state_key = item.state_key
        input_state = trace_step.exact_input_states[state_key]
        plan = trace_step.exact_path.plans[state_key]
        q_i16 = plan.q_i16.flatten().clone()
        new_acc_i32 = plan.new_acc_i32.flatten().clone().to(torch.int32)
        pre_cap_indices = plan.applied_indices.to(torch.int64)
        pre_cap_directions = plan.applied_directions.to(torch.int16)
        pre_cap_thresholds = plan.applied_thresholds.to(torch.int32)
        accepted_mask = torch.tensor(
            [
                (state_key, int(idx)) in replay_accepted
                for idx in pre_cap_indices.detach().cpu().tolist()
            ],
            dtype=torch.bool,
        )
        accepted_indices = pre_cap_indices[accepted_mask]
        accepted_directions = pre_cap_directions[accepted_mask]
        accepted_thresholds = pre_cap_thresholds[accepted_mask]
        if accepted_indices.numel() > 0:
            q_i16[accepted_indices] = (
                q_i16[accepted_indices] + accepted_directions
            ).clamp(-1, 1)
            _path_b_apply_threshold_residual(
                new_acc_i32,
                accepted_indices,
                accepted_directions,
                accepted_thresholds,
            )
        replay_indices = plan.replay_ce_veto_indices.to(torch.int64)
        replay_directions = plan.replay_veto_directions.to(torch.int16)
        replay_thresholds = plan.replay_veto_thresholds.to(torch.int32)
        _path_b_apply_threshold_residual(
            new_acc_i32,
            replay_indices,
            replay_directions,
            replay_thresholds,
        )
        q_out = q_i16.view_as(input_state.q_levels).to(torch.int8).contiguous()
        acc_out = new_acc_i32.view_as(input_state.accumulators).to(torch.int16).contiguous()
        changed = torch.nonzero(
            q_out.flatten() != input_state.q_levels.flatten(),
            as_tuple=False,
        ).flatten()
        q_changed_ids |= _path_b_ids_from_indices(state_key, changed)
        output_states[state_key] = VoteUpdateState(
            q_levels=q_out,
            accumulators=acc_out,
        )
    return _PathBReplayStepState(
        schedule_name=trace_step.schedule_step.name,
        step=int(trace_step.schedule_step.step),
        global_cap=int(trace_step.cap_spec.cap),
        candidate_row_ids={row.identity for row in observable_rows},
        accepted_ids=set(replay_accepted),
        deferred_ids=set(replay_deferred),
        backlog_ids=_backlog_key_set(backlog),
        output_states=output_states,
        output_backlog=backlog,
        q_changed_ids=q_changed_ids,
        dropped_ids=set(dropped_ids),
        mixed_feature_class_count=int(mixed_feature_class_count),
        mixed_feature_class_row_count=int(mixed_feature_class_row_count),
        max_mixed_class_cardinality=int(max_mixed_class_cardinality),
        observable_rows=observable_rows,
    )


def _path_b_q_divergence_count(
    *,
    replay_states: Mapping[str, VoteUpdateState],
    exact_output_q_by_key: Mapping[str, torch.Tensor],
) -> int:
    divergence_ids: set[tuple[str, int]] = set()
    for state_key in PRIMARY_STATE_KEYS:
        replay_q = replay_states[state_key].q_levels.flatten()
        exact_q = exact_output_q_by_key[state_key].flatten()
        changed = torch.nonzero(replay_q != exact_q, as_tuple=False).flatten()
        divergence_ids |= _path_b_ids_from_indices(state_key, changed)
    return int(len(divergence_ids))


def _path_b_defer_all_baseline_step_report(
    *,
    exact_trace_step: _ExactScheduleTraceStep,
    replay_step: _PathBReplayStepState,
) -> PathBDeferAllBaselineStepReport:
    cap_result = exact_trace_step.exact_path.cap_result
    if cap_result is None:
        raise ValueError("path-(b) baseline step report requires an exact cap-result trace")
    exact_accepted = {
        (row.state_key, int(row.flat_index)) for row in cap_result.accepted_rows
    }
    exact_deferred = {
        (row.state_key, int(row.flat_index)) for row in cap_result.deferred_rows
    }
    exact_backlog_ids = _backlog_key_set(exact_trace_step.exact_output_backlog)
    return PathBDeferAllBaselineStepReport(
        schedule_name=exact_trace_step.schedule_step.name,
        step=int(exact_trace_step.schedule_step.step),
        global_cap=int(exact_trace_step.cap_spec.cap),
        exact_candidate_row_count=len(cap_result.rows),
        replay_candidate_row_count=len(replay_step.candidate_row_ids),
        exact_accepted_count=len(exact_accepted),
        replay_accepted_count=len(replay_step.accepted_ids),
        accepted_surface_symmetric_difference=len(
            exact_accepted ^ replay_step.accepted_ids
        ),
        exact_deferred_count=len(exact_deferred),
        replay_deferred_count=len(replay_step.deferred_ids),
        deferred_surface_symmetric_difference=len(
            exact_deferred ^ replay_step.deferred_ids
        ),
        exact_backlog_count=len(exact_backlog_ids),
        replay_backlog_count=len(replay_step.backlog_ids),
        backlog_surface_symmetric_difference=len(
            exact_backlog_ids ^ replay_step.backlog_ids
        ),
        q_divergence_count=_path_b_q_divergence_count(
            replay_states=replay_step.output_states,
            exact_output_q_by_key=exact_trace_step.exact_path.output_q_by_key,
        ),
        mixed_feature_class_count=int(replay_step.mixed_feature_class_count),
        mixed_feature_class_row_count=int(replay_step.mixed_feature_class_row_count),
        max_mixed_class_cardinality=int(replay_step.max_mixed_class_cardinality),
        dropped_mass_count=len(replay_step.dropped_ids),
        exact_accepted_identities_sha256=_identity_sha256(exact_accepted),
        replay_accepted_identities_sha256=_identity_sha256(replay_step.accepted_ids),
        exact_deferred_identities_sha256=_identity_sha256(exact_deferred),
        replay_deferred_identities_sha256=_identity_sha256(replay_step.deferred_ids),
        exact_backlog_identities_sha256=_identity_sha256(exact_backlog_ids),
        replay_backlog_identities_sha256=_identity_sha256(replay_step.backlog_ids),
        dropped_mass_identities_sha256=_identity_sha256(replay_step.dropped_ids),
    )


def _path_b_future_class_support_summary(
    *,
    dropped_ids: set[tuple[str, int]],
    future_steps: Sequence[_PathBReplayStepState],
) -> tuple[set[tuple[str, int]], int, int, tuple[str, ...]]:
    re_presented: set[tuple[str, int]] = set()
    peak_class_count = 0
    peak_class_cardinality = 0
    future_schedule_names: list[str] = []
    for future_step in future_steps:
        identity_to_group = {
            row.identity: (row.bucket_key, row.feature_key())
            for row in future_step.observable_rows
        }
        re_presented_now = dropped_ids & future_step.candidate_row_ids
        if not re_presented_now:
            continue
        future_schedule_names.append(future_step.schedule_name)
        re_presented |= re_presented_now
        by_group: dict[
            tuple[tuple[str, int, int], tuple[tuple[str, Any], ...]],
            set[tuple[str, int]],
        ] = {}
        for identity in re_presented_now:
            group_key = identity_to_group.get(identity)
            if group_key is None:
                raise ValueError(
                    "re-presented identity must stay observable inside the replay step"
                )
            by_group.setdefault(group_key, set()).add(identity)
        peak_class_count = max(peak_class_count, len(by_group))
        if by_group:
            peak_class_cardinality = max(
                peak_class_cardinality,
                max(len(group) for group in by_group.values()),
            )
    return (
        re_presented,
        int(peak_class_count),
        int(peak_class_cardinality),
        tuple(dict.fromkeys(future_schedule_names)),
    )


def _path_b_dropped_mass_origin_reports(
    replay_steps: Sequence[_PathBReplayStepState],
) -> tuple[PathBDroppedMassOriginReport, ...]:
    reports: list[PathBDroppedMassOriginReport] = []
    for index, origin_step in enumerate(replay_steps):
        dropped_ids = set(origin_step.dropped_ids)
        if not dropped_ids:
            continue
        future_steps = tuple(replay_steps[index + 1 :])
        re_presented_ids, peak_class_count, peak_class_cardinality, future_schedule_names = (
            _path_b_future_class_support_summary(
                dropped_ids=dropped_ids,
                future_steps=future_steps,
            )
        )
        eventually_accepted = set()
        for future_step in future_steps:
            eventually_accepted |= dropped_ids & future_step.accepted_ids
        terminal_censored = set(dropped_ids) if not future_steps else set()
        recoverable_but_unrecovered = re_presented_ids - eventually_accepted
        never_recovered = dropped_ids - eventually_accepted - terminal_censored
        reports.append(
            PathBDroppedMassOriginReport(
                origin_schedule_name=origin_step.schedule_name,
                origin_step=int(origin_step.step),
                dropped_mass_count=len(dropped_ids),
                dropped_mass_identities_sha256=_identity_sha256(dropped_ids),
                re_presented_later_count=len(re_presented_ids),
                re_presented_later_identities_sha256=_identity_sha256(re_presented_ids),
                eventually_accepted_under_baseline_count=len(eventually_accepted),
                eventually_accepted_under_baseline_identities_sha256=_identity_sha256(
                    eventually_accepted
                ),
                recoverable_but_unrecovered_count=len(recoverable_but_unrecovered),
                recoverable_but_unrecovered_identities_sha256=_identity_sha256(
                    recoverable_but_unrecovered
                ),
                never_recovered_count=len(never_recovered),
                never_recovered_identities_sha256=_identity_sha256(never_recovered),
                terminal_censored_mass_count=len(terminal_censored),
                terminal_censored_mass_identities_sha256=_identity_sha256(
                    terminal_censored
                ),
                future_schedule_names=future_schedule_names,
                bounded_class_count_upper_bound=int(peak_class_count),
                max_bounded_class_cardinality=int(peak_class_cardinality),
            )
        )
    return tuple(reports)


def _path_b_defer_all_baseline_non_claims() -> tuple[str, ...]:
    return (
        "CPU-only mutated recurrence replay over the fixed preregistered native reference schedule",
        "rule inputs remain limited to current-step bucket plus within-bucket observable feature classes",
        "the replay mutates the accepted/deferred partition and recomputes downstream backlog/q/acc from that mutated partition",
        "aggregate-state is non-competing in this slice; carry_candidate_earned only routes to a future runtime-semantics-definition plan",
        "nonzero terminal-step censored dropped mass blocks baseline_sufficient_no_carry_needed",
        "candidate-only; no dyn200, no learner/sub-2 claim, no raw per-weight arrays",
    )


def _path_b_defer_all_baseline_terminal_decision(
    *,
    step_reports: Sequence[PathBDeferAllBaselineStepReport],
    origin_reports: Sequence[PathBDroppedMassOriginReport],
) -> PathBDeferAllBaselineDecision:
    if not step_reports:
        raise ValueError("path-(b) baseline parity probe requires at least one step report")
    final_step = step_reports[-1]
    total_dropped_mass = sum(report.dropped_mass_count for report in origin_reports)
    total_re_presented = sum(report.re_presented_later_count for report in origin_reports)
    total_eventually_accepted = sum(
        report.eventually_accepted_under_baseline_count for report in origin_reports
    )
    total_recoverable_but_unrecovered = sum(
        report.recoverable_but_unrecovered_count for report in origin_reports
    )
    total_never_recovered = sum(report.never_recovered_count for report in origin_reports)
    total_terminal_censored = sum(
        report.terminal_censored_mass_count for report in origin_reports
    )
    peak_class_count = max(
        (report.bounded_class_count_upper_bound for report in origin_reports),
        default=0,
    )
    peak_class_cardinality = max(
        (report.max_bounded_class_cardinality for report in origin_reports),
        default=0,
    )
    if total_recoverable_but_unrecovered > 0:
        terminal = CARRY_CANDIDATE_EARNED
        reason = (
            "non-terminal defer-all dropped mass re-presented later as a bounded identity-free "
            "class aggregate but still failed to get accepted under the 0-bit baseline, so the "
            "aggregate runtime-semantics-definition plan is earned next"
        )
    elif total_terminal_censored > 0:
        terminal = INCONCLUSIVE_NEEDS_LOOP_MEASUREMENT
        reason = (
            "terminal-step dropped mass remained horizon-censored on backlog_growth, so the hard "
            "baseline cannot claim sufficiency on the fixed four-step schedule"
        )
    elif (
        final_step.q_divergence_count == 0
        and final_step.accepted_surface_symmetric_difference == 0
        and final_step.deferred_surface_symmetric_difference == 0
        and final_step.backlog_surface_symmetric_difference == 0
        and total_never_recovered == 0
    ):
        terminal = BASELINE_SUFFICIENT_NO_CARRY_NEEDED
        reason = (
            "the 0-bit defer-all replay closed back onto the committed exact-path surfaces by the "
            "end of the preregistered schedule with no unrecovered or censored dropped mass"
        )
    else:
        terminal = INCONCLUSIVE_NEEDS_LOOP_MEASUREMENT
        reason = (
            "the replay exposed residual divergence without an observed bounded recoverable class "
            "aggregate that would justify carry, so the honest next step is more measurement rather "
            "than a sufficiency or carry claim"
        )
    return PathBDeferAllBaselineDecision(
        terminal_label=terminal,
        final_step_schedule_name=final_step.schedule_name,
        final_q_divergence_count=int(final_step.q_divergence_count),
        final_accepted_surface_symmetric_difference=int(
            final_step.accepted_surface_symmetric_difference
        ),
        final_deferred_surface_symmetric_difference=int(
            final_step.deferred_surface_symmetric_difference
        ),
        final_backlog_surface_symmetric_difference=int(
            final_step.backlog_surface_symmetric_difference
        ),
        total_dropped_mass_count=int(total_dropped_mass),
        total_re_presented_later_count=int(total_re_presented),
        total_eventually_accepted_under_baseline_count=int(total_eventually_accepted),
        total_recoverable_but_unrecovered_count=int(
            total_recoverable_but_unrecovered
        ),
        total_never_recovered_count=int(total_never_recovered),
        total_terminal_censored_mass_count=int(total_terminal_censored),
        peak_bounded_class_count_upper_bound=int(peak_class_count),
        peak_bounded_class_cardinality=int(peak_class_cardinality),
        aggregate_runtime_semantics_definition_plan_earned=(
            terminal == CARRY_CANDIDATE_EARNED
        ),
        candidate_only=True,
        dyn200_earned=False,
        learner_sub2_claimed=False,
        reason=reason,
    )


def _path_b_exact_and_defer_all_replay_steps(
    ) -> tuple[tuple[_ExactScheduleTraceStep, _PathBReplayStepState], ...]:
    exact_trace_steps, _ = _build_exact_schedule_trace()
    replay_states = _initial_states()
    replay_backlog: dict[str, dict[int, dict[str, int]]] = {}
    replay_pairs: list[tuple[_ExactScheduleTraceStep, _PathBReplayStepState]] = []
    for exact_trace_step in exact_trace_steps:
        replay_trace_step = _path_b_current_trace_step(
            states_by_key=replay_states,
            deferred_backlog=replay_backlog,
            schedule_step=exact_trace_step.schedule_step,
        )
        replay_step = _path_b_replay_step_from_mutated_partition(replay_trace_step)
        replay_pairs.append((exact_trace_step, replay_step))
        replay_states = _copy_state_map(replay_step.output_states)
        replay_backlog = _copy_backlog(replay_step.output_backlog)
    return tuple(replay_pairs)


def run_path_b_defer_all_baseline_parity_probe() -> PathBDeferAllBaselineParityProbeReport:
    classifier_report = run_path_b_identity_free_tie_rule_classifier()
    validate_path_b_identity_free_tie_rule_classifier_report(classifier_report)
    classifier_by_variant = {
        entry.variant_name: entry for entry in classifier_report.family_reports
    }
    defer_all = classifier_by_variant[CLASS_ACTION_DEFER_ALL_MIXED_CLASSES_NO_BACKFILL]
    if defer_all.earned_downstream_test != RUNTIME_TIE_RULE_MUTATION_PARITY_PROBE:
        raise ValueError(
            "path-(b) baseline parity probe requires the committed defer-all runtime mutation trigger"
        )
    replay_pairs = _path_b_exact_and_defer_all_replay_steps()
    replay_steps = [replay_step for _, replay_step in replay_pairs]
    step_reports = [
        _path_b_defer_all_baseline_step_report(
            exact_trace_step=exact_trace_step,
            replay_step=replay_step,
        )
        for exact_trace_step, replay_step in replay_pairs
    ]
    origin_reports = _path_b_dropped_mass_origin_reports(replay_steps)
    bindingness = pre_register_source_bindingness(
        source_kind=SOURCE_KIND_GENERATED_NATIVE_LOOP,
        coverage=SourceFieldCoverage.full_generated_native_loop(),
    )
    return PathBDeferAllBaselineParityProbeReport(
        schema_version=PATH_B_DEFER_ALL_BASELINE_PARITY_PROBE_SCHEMA_VERSION,
        label=PATH_B_DEFER_ALL_BASELINE_PARITY_PROBE_LABEL,
        source_bindingness=bindingness,
        field_coverage=SourceFieldCoverage.full_generated_native_loop(),
        candidate_name=PATH_B_DEFER_ALL_BASELINE_PARITY_PROBE_CANDIDATE,
        source_classifier_label=classifier_report.label,
        source_classifier_candidate_variant=CLASS_ACTION_DEFER_ALL_MIXED_CLASSES_NO_BACKFILL,
        source_classifier_downstream_test=RUNTIME_TIE_RULE_MUTATION_PARITY_PROBE,
        step_reports=tuple(step_reports),
        dropped_mass_origin_reports=origin_reports,
        terminal_decision=_path_b_defer_all_baseline_terminal_decision(
            step_reports=step_reports,
            origin_reports=origin_reports,
        ),
        raw_arrays_included=False,
        non_claims=_path_b_defer_all_baseline_non_claims(),
    )


def validate_path_b_defer_all_baseline_parity_probe_report(
    report: PathBDeferAllBaselineParityProbeReport,
) -> None:
    if report.schema_version != PATH_B_DEFER_ALL_BASELINE_PARITY_PROBE_SCHEMA_VERSION:
        raise ValueError("unexpected path-(b) defer-all baseline parity schema version")
    if report.label != PATH_B_DEFER_ALL_BASELINE_PARITY_PROBE_LABEL:
        raise ValueError("unexpected path-(b) defer-all baseline parity label")
    if report.candidate_name != PATH_B_DEFER_ALL_BASELINE_PARITY_PROBE_CANDIDATE:
        raise ValueError("unexpected path-(b) defer-all baseline parity candidate")
    if report.source_classifier_label != PATH_B_IDENTITY_FREE_TIE_RULE_CLASSIFIER_LABEL:
        raise ValueError("baseline parity probe must cite the committed path-(b) classifier label")
    if report.source_classifier_candidate_variant != CLASS_ACTION_DEFER_ALL_MIXED_CLASSES_NO_BACKFILL:
        raise ValueError("baseline parity probe must stay on the defer-all class-action candidate")
    if report.source_classifier_downstream_test != RUNTIME_TIE_RULE_MUTATION_PARITY_PROBE:
        raise ValueError("baseline parity probe must inherit the runtime mutation downstream test")
    if report.terminal_decision.terminal_label not in {
        BASELINE_SUFFICIENT_NO_CARRY_NEEDED,
        CARRY_CANDIDATE_EARNED,
        INCONCLUSIVE_NEEDS_LOOP_MEASUREMENT,
    }:
        raise ValueError("unexpected path-(b) defer-all baseline terminal label")
    if not report.step_reports:
        raise ValueError("baseline parity probe requires at least one step report")
    if len(report.step_reports) != len(PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE):
        raise ValueError("baseline parity probe must cover the full preregistered schedule")
    expected_names = [step.name for step in PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE]
    actual_names = [step.schedule_name for step in report.step_reports]
    if actual_names != expected_names:
        raise ValueError("baseline parity probe step order drifted from the preregistered schedule")
    dropped_by_step = {
        (step.schedule_name, int(step.step)): int(step.dropped_mass_count)
        for step in report.step_reports
    }
    final_step = report.step_reports[-1]
    for step_report, schedule_step in zip(report.step_reports, PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE):
        if int(step_report.step) != int(schedule_step.step):
            raise ValueError("baseline parity probe step number drifted from the preregistered schedule")
        if step_report.global_cap != int(schedule_step.cap):
            raise ValueError("baseline parity probe step cap drifted from the preregistered schedule")
        if step_report.exact_accepted_count + step_report.exact_deferred_count != step_report.exact_candidate_row_count:
            raise ValueError("exact accepted/deferred counts must partition the exact candidate rows")
        if step_report.replay_accepted_count + step_report.replay_deferred_count != step_report.replay_candidate_row_count:
            raise ValueError("replay accepted/deferred counts must partition the replay candidate rows")
        if step_report.dropped_mass_count > step_report.mixed_feature_class_row_count:
            raise ValueError("dropped mass cannot exceed the rows carried by mixed feature classes")
        if step_report.mixed_feature_class_count == 0 and step_report.dropped_mass_count != 0:
            raise ValueError("dropping mass without a mixed feature class would violate the gated rule")
        if step_report.max_mixed_class_cardinality < 0:
            raise ValueError("mixed class cardinality must stay non-negative")
    for origin_report in report.dropped_mass_origin_reports:
        key = (origin_report.origin_schedule_name, int(origin_report.origin_step))
        if key not in dropped_by_step:
            raise ValueError("origin report must cite one of the traced replay steps")
        if int(origin_report.dropped_mass_count) != int(dropped_by_step[key]):
            raise ValueError("origin report dropped-mass count drifted from the step report")
        if origin_report.re_presented_later_count < origin_report.eventually_accepted_under_baseline_count:
            raise ValueError("later acceptance must be a subset of later re-presentation")
        if origin_report.recoverable_but_unrecovered_count > origin_report.re_presented_later_count:
            raise ValueError("recoverable-but-unrecovered mass must be bounded by later re-presentation")
        if origin_report.future_schedule_names != tuple(dict.fromkeys(origin_report.future_schedule_names)):
            raise ValueError("future schedule names must stay unique and ordered")
        if key == (final_step.schedule_name, int(final_step.step)):
            if origin_report.terminal_censored_mass_count != origin_report.dropped_mass_count:
                raise ValueError("terminal-step dropped mass must remain fully censored on the fixed schedule")
        else:
            if origin_report.terminal_censored_mass_count != 0:
                raise ValueError("only terminal-step dropped mass may be marked censored")
    summed_dropped = sum(report_entry.dropped_mass_count for report_entry in report.dropped_mass_origin_reports)
    summed_re_presented = sum(
        report_entry.re_presented_later_count for report_entry in report.dropped_mass_origin_reports
    )
    summed_eventually_accepted = sum(
        report_entry.eventually_accepted_under_baseline_count
        for report_entry in report.dropped_mass_origin_reports
    )
    summed_recoverable_but_unrecovered = sum(
        report_entry.recoverable_but_unrecovered_count
        for report_entry in report.dropped_mass_origin_reports
    )
    summed_never_recovered = sum(
        report_entry.never_recovered_count for report_entry in report.dropped_mass_origin_reports
    )
    summed_terminal_censored = sum(
        report_entry.terminal_censored_mass_count
        for report_entry in report.dropped_mass_origin_reports
    )
    if report.terminal_decision.total_dropped_mass_count != int(summed_dropped):
        raise ValueError("baseline parity total dropped-mass count drifted from the origin reports")
    if report.terminal_decision.total_re_presented_later_count != int(summed_re_presented):
        raise ValueError("baseline parity total re-presented count drifted from the origin reports")
    if report.terminal_decision.total_eventually_accepted_under_baseline_count != int(summed_eventually_accepted):
        raise ValueError("baseline parity total eventual-accept count drifted from the origin reports")
    if report.terminal_decision.total_recoverable_but_unrecovered_count != int(summed_recoverable_but_unrecovered):
        raise ValueError("baseline parity total recoverable-but-unrecovered count drifted from the origin reports")
    if report.terminal_decision.total_never_recovered_count != int(summed_never_recovered):
        raise ValueError("baseline parity total never-recovered count drifted from the origin reports")
    if report.terminal_decision.total_terminal_censored_mass_count != int(summed_terminal_censored):
        raise ValueError("baseline parity total terminal-censored count drifted from the origin reports")
    if report.terminal_decision.final_step_schedule_name != final_step.schedule_name:
        raise ValueError("baseline parity final-step schedule name drifted from the step reports")
    if report.terminal_decision.final_q_divergence_count != int(final_step.q_divergence_count):
        raise ValueError("baseline parity final q divergence drifted from the final step report")
    if report.terminal_decision.final_accepted_surface_symmetric_difference != int(
        final_step.accepted_surface_symmetric_difference
    ):
        raise ValueError("baseline parity final accepted-surface divergence drifted from the final step report")
    if report.terminal_decision.final_deferred_surface_symmetric_difference != int(
        final_step.deferred_surface_symmetric_difference
    ):
        raise ValueError("baseline parity final deferred-surface divergence drifted from the final step report")
    if report.terminal_decision.final_backlog_surface_symmetric_difference != int(
        final_step.backlog_surface_symmetric_difference
    ):
        raise ValueError("baseline parity final backlog divergence drifted from the final step report")
    if report.terminal_decision.candidate_only is not True:
        raise ValueError("baseline parity probe must stay candidate-only")
    if report.terminal_decision.dyn200_earned or report.terminal_decision.learner_sub2_claimed:
        raise ValueError("baseline parity probe must not claim dyn200 or learner/sub-2 success")
    label = report.terminal_decision.terminal_label
    if label == BASELINE_SUFFICIENT_NO_CARRY_NEEDED:
        if report.terminal_decision.total_recoverable_but_unrecovered_count != 0:
            raise ValueError("baseline sufficiency requires zero recoverable-but-unrecovered mass")
        if report.terminal_decision.total_terminal_censored_mass_count != 0:
            raise ValueError("baseline sufficiency requires zero terminal-censored mass")
        if any(
            value != 0
            for value in (
                report.terminal_decision.final_q_divergence_count,
                report.terminal_decision.final_accepted_surface_symmetric_difference,
                report.terminal_decision.final_deferred_surface_symmetric_difference,
                report.terminal_decision.final_backlog_surface_symmetric_difference,
                report.terminal_decision.total_never_recovered_count,
            )
        ):
            raise ValueError("baseline sufficiency requires closure of final divergences and unrecovered mass")
        if report.terminal_decision.aggregate_runtime_semantics_definition_plan_earned:
            raise ValueError("baseline sufficiency must not earn the aggregate runtime-semantics-definition plan")
    elif label == CARRY_CANDIDATE_EARNED:
        if report.terminal_decision.total_recoverable_but_unrecovered_count <= 0:
            raise ValueError("carry candidate requires observed recoverable-but-unrecovered mass")
        if not report.terminal_decision.aggregate_runtime_semantics_definition_plan_earned:
            raise ValueError("carry candidate must earn the aggregate runtime-semantics-definition plan")
    else:
        if report.terminal_decision.aggregate_runtime_semantics_definition_plan_earned:
            raise ValueError("inconclusive terminal must not earn the aggregate runtime-semantics-definition plan")
    _assert_no_tensors(report.to_dict())


@dataclass(frozen=True)
class AggregateStateRuntimeLedgerCharge:
    base_total_bits: int
    total_bits: int
    bits_per_eligible_weight: float
    bounded_under_strictest_headroom: bool
    concurrent_class_count: int
    class_key_bits: int
    aggregate_payload_bits: int
    metadata_bits: int
    carry_bits: int
    projection_bits: int
    ttl_bits: int
    age_bits: int
    merge_counter_bits: int
    active_slot_bits: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_total_bits": int(self.base_total_bits),
            "total_bits": int(self.total_bits),
            "bits_per_eligible_weight": float(self.bits_per_eligible_weight),
            "bounded_under_strictest_headroom": bool(
                self.bounded_under_strictest_headroom
            ),
            "concurrent_class_count": int(self.concurrent_class_count),
            "class_key_bits": int(self.class_key_bits),
            "aggregate_payload_bits": int(self.aggregate_payload_bits),
            "metadata_bits": int(self.metadata_bits),
            "carry_bits": int(self.carry_bits),
            "projection_bits": int(self.projection_bits),
            "ttl_bits": int(self.ttl_bits),
            "age_bits": int(self.age_bits),
            "merge_counter_bits": int(self.merge_counter_bits),
            "active_slot_bits": int(self.active_slot_bits),
        }


@dataclass(frozen=True)
class AggregateStateLifecycleSpec:
    create_policy: str
    merge_policy: str
    consume_policy: str
    expire_policy: str
    ttl_steps: int | None
    active_carry_count_bound: int
    predeclared_inter_class_priority: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "create_policy": self.create_policy,
            "merge_policy": self.merge_policy,
            "consume_policy": self.consume_policy,
            "expire_policy": self.expire_policy,
            "ttl_steps": self.ttl_steps,
            "active_carry_count_bound": int(self.active_carry_count_bound),
            "predeclared_inter_class_priority": self.predeclared_inter_class_priority,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class AggregateStateProjectionClassReport:
    origin_schedule_name: str
    future_schedule_name: str
    state_key: str
    current_q_level: int
    move_direction: int
    origin_feature_payload: dict[str, Any]
    future_feature_payload: dict[str, Any]
    origin_class_cardinality: int
    future_class_cardinality: int
    future_class_count_in_bucket: int
    projection_bits_for_class: int
    carry_matched_row_count: int
    recovered_dropped_count: int
    matched_not_recovered_count: int
    future_observable_split_possible: bool
    future_observable_split_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_schedule_name": self.origin_schedule_name,
            "future_schedule_name": self.future_schedule_name,
            "state_key": self.state_key,
            "current_q_level": int(self.current_q_level),
            "move_direction": int(self.move_direction),
            "origin_feature_payload": dict(self.origin_feature_payload),
            "future_feature_payload": dict(self.future_feature_payload),
            "origin_class_cardinality": int(self.origin_class_cardinality),
            "future_class_cardinality": int(self.future_class_cardinality),
            "future_class_count_in_bucket": int(self.future_class_count_in_bucket),
            "projection_bits_for_class": int(self.projection_bits_for_class),
            "carry_matched_row_count": int(self.carry_matched_row_count),
            "recovered_dropped_count": int(self.recovered_dropped_count),
            "matched_not_recovered_count": int(self.matched_not_recovered_count),
            "future_observable_split_possible": bool(
                self.future_observable_split_possible
            ),
            "future_observable_split_reason": self.future_observable_split_reason,
        }


@dataclass(frozen=True)
class AggregateStateRuntimeFamilyReport:
    family_name: str
    variant_name: str
    terminal_label: str
    candidate_subcase: str | None
    action_order_policy: str
    immediate_action_policy: str
    initial_residual_cap: int
    post_action_residual_cap: int
    carry_matched_row_count: int
    recovered_dropped_count: int
    false_positive_accept_count: int
    missed_represented_count: int
    collision_class_cardinalities: tuple[int, ...]
    future_observable_split_possible: bool
    projection_class_reports: tuple[AggregateStateProjectionClassReport, ...]
    lifecycle_spec: AggregateStateLifecycleSpec
    ledger_charge: AggregateStateRuntimeLedgerCharge
    earned_downstream_test: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_name": self.family_name,
            "variant_name": self.variant_name,
            "terminal_label": self.terminal_label,
            "candidate_subcase": self.candidate_subcase,
            "action_order_policy": self.action_order_policy,
            "immediate_action_policy": self.immediate_action_policy,
            "initial_residual_cap": int(self.initial_residual_cap),
            "post_action_residual_cap": int(self.post_action_residual_cap),
            "carry_matched_row_count": int(self.carry_matched_row_count),
            "recovered_dropped_count": int(self.recovered_dropped_count),
            "false_positive_accept_count": int(self.false_positive_accept_count),
            "missed_represented_count": int(self.missed_represented_count),
            "collision_class_cardinalities": [
                int(value) for value in self.collision_class_cardinalities
            ],
            "future_observable_split_possible": bool(
                self.future_observable_split_possible
            ),
            "projection_class_reports": [
                report.to_dict() for report in self.projection_class_reports
            ],
            "lifecycle_spec": self.lifecycle_spec.to_dict(),
            "ledger_charge": self.ledger_charge.to_dict(),
            "earned_downstream_test": self.earned_downstream_test,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AggregateStateRuntimeSemanticsDecision:
    terminal_label: str
    immediate_recovery_semantics_exhausted_on_this_state_path: bool
    candidate_family_names: tuple[str, ...]
    inconclusive_family_names: tuple[str, ...]
    negative_family_names: tuple[str, ...]
    future_observable_split_possible_any: bool
    peak_total_bits: int
    peak_bits_per_eligible_weight: float
    learning_retention_tolerance_probe_earned: bool
    candidate_only: bool
    dyn200_earned: bool
    learner_sub2_claimed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "terminal_label": self.terminal_label,
            "immediate_recovery_semantics_exhausted_on_this_state_path": bool(
                self.immediate_recovery_semantics_exhausted_on_this_state_path
            ),
            "candidate_family_names": list(self.candidate_family_names),
            "inconclusive_family_names": list(self.inconclusive_family_names),
            "negative_family_names": list(self.negative_family_names),
            "future_observable_split_possible_any": bool(
                self.future_observable_split_possible_any
            ),
            "peak_total_bits": int(self.peak_total_bits),
            "peak_bits_per_eligible_weight": float(self.peak_bits_per_eligible_weight),
            "learning_retention_tolerance_probe_earned": bool(
                self.learning_retention_tolerance_probe_earned
            ),
            "candidate_only": bool(self.candidate_only),
            "dyn200_earned": bool(self.dyn200_earned),
            "learner_sub2_claimed": bool(self.learner_sub2_claimed),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AggregateStateRuntimeSemanticsReport:
    schema_version: str
    label: str
    source_bindingness: Any
    field_coverage: SourceFieldCoverage
    candidate_name: str
    source_baseline_label: str
    source_baseline_terminal_label: str
    family_reports: tuple[AggregateStateRuntimeFamilyReport, ...]
    terminal_decision: AggregateStateRuntimeSemanticsDecision
    raw_arrays_included: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "source_bindingness": self.source_bindingness.to_dict(),
            "field_coverage": self.field_coverage.to_dict(),
            "candidate_name": self.candidate_name,
            "source_baseline_label": self.source_baseline_label,
            "source_baseline_terminal_label": self.source_baseline_terminal_label,
            "family_reports": [family.to_dict() for family in self.family_reports],
            "terminal_decision": self.terminal_decision.to_dict(),
            "raw_arrays_included": bool(self.raw_arrays_included),
            "non_claims": list(self.non_claims),
        }


@dataclass(frozen=True)
class _AggregateStateProjectionContext:
    origin_schedule_name: str
    future_schedule_name: str
    future_global_cap: int
    projection_class_reports: tuple[AggregateStateProjectionClassReport, ...]


def _path_b_first_represented_projection_context() -> _AggregateStateProjectionContext:
    replay_pairs = _path_b_exact_and_defer_all_replay_steps()
    for origin_exact, origin_replay in replay_pairs[:-1]:
        for _, future_replay in replay_pairs[replay_pairs.index((origin_exact, origin_replay)) + 1 :]:
            re_presented = origin_replay.dropped_ids & future_replay.candidate_row_ids
            if not re_presented:
                continue
            origin_rows_by_id = {row.identity: row for row in origin_replay.observable_rows}
            future_rows_by_id = {row.identity: row for row in future_replay.observable_rows}
            future_bucket_feature_sets: dict[tuple[str, int, int], set[tuple[tuple[str, Any], ...]]] = {}
            future_class_rows: dict[
                tuple[tuple[str, int, int], tuple[tuple[str, Any], ...]],
                list[_StrictObservableTieMaskRow],
            ] = {}
            for row in future_replay.observable_rows:
                future_bucket_feature_sets.setdefault(row.bucket_key, set()).add(
                    row.feature_key()
                )
                future_class_rows.setdefault(
                    (row.bucket_key, row.feature_key()), []
                ).append(row)
            by_future_group: dict[
                tuple[tuple[str, int, int], tuple[tuple[str, Any], ...]],
                list[tuple[str, int]],
            ] = {}
            for identity in sorted(re_presented):
                future_row = future_rows_by_id.get(identity)
                if future_row is None:
                    raise ValueError("future projection identity missing from the replay rows")
                by_future_group.setdefault(
                    (future_row.bucket_key, future_row.feature_key()), []
                ).append(identity)
            projection_reports: list[AggregateStateProjectionClassReport] = []
            for (bucket_key, feature_key), identities in sorted(
                by_future_group.items(),
                key=lambda item: (item[0][0], item[0][1]),
            ):
                class_rows = future_class_rows[(bucket_key, feature_key)]
                origin_feature_rows = [origin_rows_by_id[identity] for identity in identities]
                origin_bucket_key = origin_feature_rows[0].bucket_key
                origin_feature_key = origin_feature_rows[0].feature_key()
                origin_class_rows = [
                    row
                    for row in origin_replay.observable_rows
                    if row.bucket_key == origin_bucket_key
                    and row.feature_key() == origin_feature_key
                ]
                future_class_count_in_bucket = len(
                    future_bucket_feature_sets[bucket_key]
                )
                projection_reports.append(
                    AggregateStateProjectionClassReport(
                        origin_schedule_name=origin_replay.schedule_name,
                        future_schedule_name=future_replay.schedule_name,
                        state_key=str(bucket_key[0]),
                        current_q_level=int(bucket_key[1]),
                        move_direction=int(bucket_key[2]),
                        origin_feature_payload=dict(origin_feature_rows[0].feature_payload()),
                        future_feature_payload=dict(class_rows[0].feature_payload()),
                        origin_class_cardinality=len(origin_class_rows),
                        future_class_cardinality=len(class_rows),
                        future_class_count_in_bucket=int(future_class_count_in_bucket),
                        projection_bits_for_class=_enum_bit_width(
                            int(future_class_count_in_bucket)
                        ),
                        carry_matched_row_count=len(class_rows),
                        recovered_dropped_count=len(identities),
                        matched_not_recovered_count=len(class_rows) - len(identities),
                        future_observable_split_possible=False,
                        future_observable_split_reason=(
                            "the best-case future projection can target the 384-row future class, "
                            "but every row inside that class shares the full allowed future-step "
                            "observable signature, so no lawful 384->128 split remains"
                        ),
                    )
                )
            return _AggregateStateProjectionContext(
                origin_schedule_name=origin_replay.schedule_name,
                future_schedule_name=future_replay.schedule_name,
                future_global_cap=int(future_replay.global_cap),
                projection_class_reports=tuple(projection_reports),
            )
    raise ValueError(
        "aggregate-state runtime semantics require a deferred origin that re-presents later"
    )


def _path_b_projection_collision_class_cardinalities(
    projection_reports: Sequence[AggregateStateProjectionClassReport],
) -> tuple[int, ...]:
    return tuple(
        int(report.future_class_cardinality) for report in projection_reports
    )


def _path_b_projection_order_key(
    report: AggregateStateProjectionClassReport,
) -> tuple[int, str, int, int, tuple[tuple[str, Any], ...]]:
    return (
        -int(report.recovered_dropped_count),
        str(report.state_key),
        int(report.current_q_level),
        int(report.move_direction),
        tuple(sorted(report.future_feature_payload.items())),
    )


def _path_b_aggregate_state_runtime_ledger_charge(
    online_report: ObservableTieMaskOnlineEstimabilityReport,
    projection_reports: Sequence[AggregateStateProjectionClassReport],
    *,
    ttl_bits: int = 0,
    age_bits: int = 0,
    merge_counter_bits: int = 0,
    active_slot_bits: int = 0,
) -> AggregateStateRuntimeLedgerCharge:
    base = _path_b_aggregate_state_ledger_charge(online_report)
    strict_row = _q_ledger_row_by_name(online_report.strictest_required_q_regime_name)
    projection_bits = sum(
        int(report.projection_bits_for_class) for report in projection_reports
    )
    total_bits = (
        int(base.total_bits)
        + int(projection_bits)
        + int(ttl_bits)
        + int(age_bits)
        + int(merge_counter_bits)
        + int(active_slot_bits)
    )
    bits_per_weight = float(total_bits) / float(int(strict_row.eligible_weight_count))
    return AggregateStateRuntimeLedgerCharge(
        base_total_bits=int(base.total_bits),
        total_bits=int(total_bits),
        bits_per_eligible_weight=float(bits_per_weight),
        bounded_under_strictest_headroom=bool(
            bits_per_weight
            <= float(online_report.strictest_headroom_bits_per_weight) + 1e-12
        ),
        concurrent_class_count=int(base.concurrent_class_count),
        class_key_bits=int(base.class_key_bits),
        aggregate_payload_bits=int(base.aggregate_payload_bits),
        metadata_bits=int(base.metadata_bits),
        carry_bits=int(base.carry_bits),
        projection_bits=int(projection_bits),
        ttl_bits=int(ttl_bits),
        age_bits=int(age_bits),
        merge_counter_bits=int(merge_counter_bits),
        active_slot_bits=int(active_slot_bits),
    )


def _path_b_immediate_aggregate_lifecycle_spec(
    *,
    projection_reports: Sequence[AggregateStateProjectionClassReport],
    action_description: str,
) -> AggregateStateLifecycleSpec:
    return AggregateStateLifecycleSpec(
        create_policy=(
            "create one carry packet per origin mixed class with stored debt equal to the "
            "dropped-row count and a charged future-class projection selector"
        ),
        merge_policy=(
            "none needed on the observed immediate-recovery path; no duplicate future class key "
            "is created before the first future projection"
        ),
        consume_policy=action_description,
        expire_policy=(
            "none needed for immediate-recovery semantics on this state path because the packet is "
            "consumed or ruled out at the first future projection"
        ),
        ttl_steps=None,
        active_carry_count_bound=len(tuple(projection_reports)),
        predeclared_inter_class_priority=(
            "carried classes ordered by stored debt descending then class key lexicographic over "
            "(state_key, current_q_level, move_direction, projected future feature payload)"
        ),
        notes=(
            "immediate-recovery families pay only the charged future projection selector beyond the "
            "base aggregate-state feasibility note"
        ),
    )


def _path_b_defer_until_fit_lifecycle_spec(
    *,
    projection_reports: Sequence[AggregateStateProjectionClassReport],
) -> AggregateStateLifecycleSpec:
    return AggregateStateLifecycleSpec(
        create_policy=(
            "when a projected carried class exceeds residual cap, persist the full class packet with "
            "its stored debt and charged future-class projection selector"
        ),
        merge_policy=(
            "merge identical carried class packets by summing debt counts under the same class key; "
            "no separate merge counter is needed on this state path"
        ),
        consume_policy=(
            "accept the full carried class only when its class_cardinality <= residual_cap under the "
            "predeclared class-level priority order, then consume the packet"
        ),
        expire_policy=(
            "expire the packet when its TTL elapses without a lawful full-class fit"
        ),
        ttl_steps=2,
        active_carry_count_bound=len(tuple(projection_reports)),
        predeclared_inter_class_priority=(
            "carried classes ordered by stored debt descending then class key lexicographic over "
            "(state_key, current_q_level, move_direction, projected future feature payload)"
        ),
        notes=(
            "this family is lawful only as an all-or-none class packet; if a full class does not fit, "
            "the whole class is deferred with no within-class quota or subset selection"
        ),
    )


def _path_b_aggregate_projection_totals(
    projection_reports: Sequence[AggregateStateProjectionClassReport],
) -> tuple[int, int]:
    return (
        sum(int(report.carry_matched_row_count) for report in projection_reports),
        sum(int(report.recovered_dropped_count) for report in projection_reports),
    )


def _path_b_aggregate_quota_release_report(
    online_report: ObservableTieMaskOnlineEstimabilityReport,
    context: _AggregateStateProjectionContext,
) -> AggregateStateRuntimeFamilyReport:
    projection_reports = context.projection_class_reports
    carry_matched_row_count, recovered_dropped_count = (
        _path_b_aggregate_projection_totals(projection_reports)
    )
    ledger_charge = _path_b_aggregate_state_runtime_ledger_charge(
        online_report,
        projection_reports,
    )
    return AggregateStateRuntimeFamilyReport(
        family_name="aggregate_state_runtime_semantics",
        variant_name=CARRY_FAMILY_QUOTA_RELEASE,
        terminal_label=CARRY_SEMANTICS_UNLAWFUL_REQUIRES_IDENTITY_OR_ORDER,
        candidate_subcase=None,
        action_order_policy=(
            "carried classes ordered by stored debt descending then class key lexicographic"
        ),
        immediate_action_policy=(
            "release only the stored debt-count quota inside each projected future class"
        ),
        initial_residual_cap=int(context.future_global_cap),
        post_action_residual_cap=int(context.future_global_cap),
        carry_matched_row_count=int(carry_matched_row_count),
        recovered_dropped_count=int(recovered_dropped_count),
        false_positive_accept_count=0,
        missed_represented_count=0,
        collision_class_cardinalities=_path_b_projection_collision_class_cardinalities(
            projection_reports
        ),
        future_observable_split_possible=False,
        projection_class_reports=projection_reports,
        lifecycle_spec=_path_b_immediate_aggregate_lifecycle_spec(
            projection_reports=projection_reports,
            action_description=(
                "consume the packet by releasing its stored debt-count quota inside the matched class"
            ),
        ),
        ledger_charge=ledger_charge,
        earned_downstream_test=None,
        reason=(
            "even after charging a best-case future-class projection selector, each matched future "
            "class is 384 rows while only 128 rows are the recovered dropped mass, so quota release "
            "would require a within-class A<T subset selection that reintroduces identity/order"
        ),
    )


def _path_b_aggregate_full_class_accept_report(
    online_report: ObservableTieMaskOnlineEstimabilityReport,
    context: _AggregateStateProjectionContext,
    *,
    variant_name: str,
    allow_extra_deviation: bool,
) -> AggregateStateRuntimeFamilyReport:
    projection_reports = tuple(
        sorted(context.projection_class_reports, key=_path_b_projection_order_key)
    )
    carry_matched_row_count, recovered_dropped_count = (
        _path_b_aggregate_projection_totals(projection_reports)
    )
    false_positive_accept_count = sum(
        int(report.matched_not_recovered_count) for report in projection_reports
    )
    residual_cap = int(context.future_global_cap)
    class_sizes = [int(report.future_class_cardinality) for report in projection_reports]
    total_class_mass = sum(class_sizes)
    fits_residual_cap = total_class_mass <= residual_cap
    ledger_charge = _path_b_aggregate_state_runtime_ledger_charge(
        online_report,
        projection_reports,
    )
    if not fits_residual_cap:
        terminal_label = CARRY_SEMANTICS_CAP_OVERFLOW_OR_BITS_UNBOUNDED
        candidate_subcase = None
        earned_downstream_test = None
        reason = (
            "the best-case projected future classes total "
            f"{int(total_class_mass)} rows while the global residual cap is only {int(context.future_global_cap)}, "
            "so a lawful all-or-none accept-all action cannot recover them without overflowing the cap"
        )
    elif allow_extra_deviation:
        terminal_label = CARRY_SEMANTICS_CANDIDATE
        candidate_subcase = (
            CARRY_SUBCASE_EXACT_RECOVERY
            if false_positive_accept_count == 0
            else CARRY_SUBCASE_CLASS_UNIFORM_BOUNDED_EXTRA_DEVIATION
        )
        earned_downstream_test = LEARNING_RETENTION_TOLERANCE_PROBE
        reason = (
            "the projected future classes fit the residual cap and can be accepted uniformly; the "
            "remaining question is whether the charged false positives are behaviorally tolerable"
        )
    else:
        terminal_label = CARRY_SEMANTICS_UNLAWFUL_REQUIRES_IDENTITY_OR_ORDER
        candidate_subcase = None
        earned_downstream_test = None
        reason = (
            "the projected future classes only recover the dropped rows by accepting an additional "
            f"{int(false_positive_accept_count)} non-dropped rows, so exact recovery would still need a "
            "forbidden within-class subset"
        )
    return AggregateStateRuntimeFamilyReport(
        family_name="aggregate_state_runtime_semantics",
        variant_name=variant_name,
        terminal_label=terminal_label,
        candidate_subcase=candidate_subcase,
        action_order_policy=(
            "carried classes ordered by stored debt descending then class key lexicographic"
        ),
        immediate_action_policy=(
            "accept the full projected future class packet(s) iff their cumulative class mass fits within "
            "residual_cap under the predeclared priority order; otherwise the family fails immediate cap-fit"
        ),
        initial_residual_cap=int(context.future_global_cap),
        post_action_residual_cap=int(
            context.future_global_cap - total_class_mass
            if fits_residual_cap
            else context.future_global_cap
        ),
        carry_matched_row_count=int(carry_matched_row_count),
        recovered_dropped_count=int(recovered_dropped_count),
        false_positive_accept_count=int(false_positive_accept_count),
        missed_represented_count=0,
        collision_class_cardinalities=_path_b_projection_collision_class_cardinalities(
            projection_reports
        ),
        future_observable_split_possible=False,
        projection_class_reports=projection_reports,
        lifecycle_spec=_path_b_immediate_aggregate_lifecycle_spec(
            projection_reports=projection_reports,
            action_description=(
                "consume the packet by accepting the full projected future class as one class-uniform unit"
            ),
        ),
        ledger_charge=ledger_charge,
        earned_downstream_test=earned_downstream_test,
        reason=reason,
    )


def _path_b_aggregate_defer_until_fit_report(
    online_report: ObservableTieMaskOnlineEstimabilityReport,
    context: _AggregateStateProjectionContext,
) -> AggregateStateRuntimeFamilyReport:
    projection_reports = tuple(
        sorted(context.projection_class_reports, key=_path_b_projection_order_key)
    )
    carry_matched_row_count, recovered_dropped_count = (
        _path_b_aggregate_projection_totals(projection_reports)
    )
    ledger_charge = _path_b_aggregate_state_runtime_ledger_charge(
        online_report,
        projection_reports,
        ttl_bits=len(projection_reports) * 2,
        age_bits=len(projection_reports) * 2,
        merge_counter_bits=0,
        active_slot_bits=_count_bit_width(len(projection_reports)),
    )
    if not ledger_charge.bounded_under_strictest_headroom:
        terminal_label = CARRY_SEMANTICS_CAP_OVERFLOW_OR_BITS_UNBOUNDED
        reason = (
            "the defer-until-fit lifecycle requires more persistent bits than the strictest remaining "
            "sub-2 accumulator headroom allows"
        )
    else:
        terminal_label = INCONCLUSIVE_NEEDS_LOOP_MEASUREMENT
        reason = (
            "the class-packet defer-until-fit lifecycle is lawful and still bounded after charging its "
            "TTL/age/active-slot fields, but the one-future-step trace cannot observe a later step where "
            "the full 384-row projected classes might fit the residual cap"
        )
    return AggregateStateRuntimeFamilyReport(
        family_name="aggregate_state_runtime_semantics",
        variant_name=CARRY_FAMILY_CLASS_UNIFORM_DEFER_UNTIL_FIT,
        terminal_label=terminal_label,
        candidate_subcase=None,
        action_order_policy=(
            "carried classes ordered by stored debt descending then class key lexicographic"
        ),
        immediate_action_policy=(
            "defer the full projected future class when class_cardinality > residual_cap; accept only on a later "
            "step where the full class fits"
        ),
        initial_residual_cap=int(context.future_global_cap),
        post_action_residual_cap=int(context.future_global_cap),
        carry_matched_row_count=int(carry_matched_row_count),
        recovered_dropped_count=0,
        false_positive_accept_count=0,
        missed_represented_count=int(recovered_dropped_count),
        collision_class_cardinalities=_path_b_projection_collision_class_cardinalities(
            projection_reports
        ),
        future_observable_split_possible=False,
        projection_class_reports=projection_reports,
        lifecycle_spec=_path_b_defer_until_fit_lifecycle_spec(
            projection_reports=projection_reports,
        ),
        ledger_charge=ledger_charge,
        earned_downstream_test=None,
        reason=reason,
    )


def _path_b_aggregate_state_runtime_non_claims() -> tuple[str, ...]:
    return (
        "CPU-first analytic theorem over the committed 2702312 defer-all replay traces",
        "future projection is a charged best-case class-level selector, not a free identity-like row pointer",
        "exact row recovery may not be claimed when future_observable_split_possible is false inside the projected class",
        "no terminal claims global carry exhaustion; negatives are scoped to immediate recovery on this state path unless a deferred lawful lifecycle is itself impossible or over-budget",
        "candidate-only; no dyn200, no learner/sub-2 claim, no aggregate-state retention run in this slice",
    )


def _path_b_aggregate_state_runtime_terminal_decision(
    family_reports: Sequence[AggregateStateRuntimeFamilyReport],
) -> AggregateStateRuntimeSemanticsDecision:
    candidate_families = tuple(
        report.variant_name
        for report in family_reports
        if report.terminal_label == CARRY_SEMANTICS_CANDIDATE
    )
    inconclusive_families = tuple(
        report.variant_name
        for report in family_reports
        if report.terminal_label == INCONCLUSIVE_NEEDS_LOOP_MEASUREMENT
    )
    negative_families = tuple(
        report.variant_name
        for report in family_reports
        if report.variant_name not in candidate_families
        and report.variant_name not in inconclusive_families
    )
    immediate_families = {
        CARRY_FAMILY_QUOTA_RELEASE,
        CARRY_FAMILY_CLASS_UNIFORM_ACCEPT_ALL_IF_FIT,
        CARRY_FAMILY_CLASS_UNIFORM_WITH_EXTRA_DEVIATION,
    }
    immediate_negative = all(
        next(report for report in family_reports if report.variant_name == variant_name).terminal_label
        != CARRY_SEMANTICS_CANDIDATE
        for variant_name in immediate_families
    )
    future_observable_split_possible_any = any(
        report.future_observable_split_possible for report in family_reports
    )
    peak_total_bits = max(int(report.ledger_charge.total_bits) for report in family_reports)
    peak_bits_per_weight = max(
        float(report.ledger_charge.bits_per_eligible_weight) for report in family_reports
    )
    if candidate_families:
        terminal_label = CARRY_SEMANTICS_CANDIDATE
        reason = (
            "at least one aggregate-state runtime semantics family survives the theorem and earns the "
            "downstream learning_retention_tolerance_probe"
        )
    elif inconclusive_families:
        terminal_label = INCONCLUSIVE_NEEDS_LOOP_MEASUREMENT
        reason = (
            "immediate recovery semantics are exhausted on this state path, but a lawful defer-until-fit "
            "class-packet lifecycle remains bounded and unobserved beyond the current horizon"
        )
    else:
        terminal_label = IMMEDIATE_RECOVERY_SEMANTICS_EXHAUSTED_ON_THIS_STATE_PATH
        reason = (
            "every immediate aggregate-state recovery family is negative on this state path and no lawful "
            "deferred class-packet lifecycle survives the budget/cap checks"
        )
    return AggregateStateRuntimeSemanticsDecision(
        terminal_label=terminal_label,
        immediate_recovery_semantics_exhausted_on_this_state_path=bool(immediate_negative),
        candidate_family_names=candidate_families,
        inconclusive_family_names=inconclusive_families,
        negative_family_names=negative_families,
        future_observable_split_possible_any=bool(future_observable_split_possible_any),
        peak_total_bits=int(peak_total_bits),
        peak_bits_per_eligible_weight=float(peak_bits_per_weight),
        learning_retention_tolerance_probe_earned=bool(candidate_families),
        candidate_only=True,
        dyn200_earned=False,
        learner_sub2_claimed=False,
        reason=reason,
    )


def run_path_b_aggregate_state_runtime_semantics_definition() -> AggregateStateRuntimeSemanticsReport:
    baseline_report = run_path_b_defer_all_baseline_parity_probe()
    validate_path_b_defer_all_baseline_parity_probe_report(baseline_report)
    if baseline_report.terminal_decision.terminal_label != CARRY_CANDIDATE_EARNED:
        raise ValueError(
            "aggregate-state runtime semantics require the committed defer-all carry candidate source"
        )
    online_report = run_online_estimable_tie_mask_diagnostic()
    projection_context = _path_b_first_represented_projection_context()
    family_reports = (
        _path_b_aggregate_quota_release_report(online_report, projection_context),
        _path_b_aggregate_full_class_accept_report(
            online_report,
            projection_context,
            variant_name=CARRY_FAMILY_CLASS_UNIFORM_ACCEPT_ALL_IF_FIT,
            allow_extra_deviation=False,
        ),
        _path_b_aggregate_full_class_accept_report(
            online_report,
            projection_context,
            variant_name=CARRY_FAMILY_CLASS_UNIFORM_WITH_EXTRA_DEVIATION,
            allow_extra_deviation=True,
        ),
        _path_b_aggregate_defer_until_fit_report(online_report, projection_context),
    )
    bindingness = pre_register_source_bindingness(
        source_kind=SOURCE_KIND_GENERATED_NATIVE_LOOP,
        coverage=SourceFieldCoverage.full_generated_native_loop(),
    )
    return AggregateStateRuntimeSemanticsReport(
        schema_version=PATH_B_AGGREGATE_STATE_RUNTIME_SEMANTICS_SCHEMA_VERSION,
        label=PATH_B_AGGREGATE_STATE_RUNTIME_SEMANTICS_LABEL,
        source_bindingness=bindingness,
        field_coverage=SourceFieldCoverage.full_generated_native_loop(),
        candidate_name=PATH_B_AGGREGATE_STATE_RUNTIME_SEMANTICS_CANDIDATE,
        source_baseline_label=baseline_report.label,
        source_baseline_terminal_label=baseline_report.terminal_decision.terminal_label,
        family_reports=family_reports,
        terminal_decision=_path_b_aggregate_state_runtime_terminal_decision(
            family_reports
        ),
        raw_arrays_included=False,
        non_claims=_path_b_aggregate_state_runtime_non_claims(),
    )


def validate_path_b_aggregate_state_runtime_semantics_report(
    report: AggregateStateRuntimeSemanticsReport,
) -> None:
    if report.schema_version != PATH_B_AGGREGATE_STATE_RUNTIME_SEMANTICS_SCHEMA_VERSION:
        raise ValueError("unexpected aggregate-state runtime semantics schema version")
    if report.label != PATH_B_AGGREGATE_STATE_RUNTIME_SEMANTICS_LABEL:
        raise ValueError("unexpected aggregate-state runtime semantics label")
    if report.candidate_name != PATH_B_AGGREGATE_STATE_RUNTIME_SEMANTICS_CANDIDATE:
        raise ValueError("unexpected aggregate-state runtime semantics candidate")
    if report.source_baseline_label != PATH_B_DEFER_ALL_BASELINE_PARITY_PROBE_LABEL:
        raise ValueError("aggregate-state runtime semantics must cite the committed baseline label")
    if report.source_baseline_terminal_label != CARRY_CANDIDATE_EARNED:
        raise ValueError("aggregate-state runtime semantics must inherit the carry-candidate baseline")
    expected_variants = (
        CARRY_FAMILY_QUOTA_RELEASE,
        CARRY_FAMILY_CLASS_UNIFORM_ACCEPT_ALL_IF_FIT,
        CARRY_FAMILY_CLASS_UNIFORM_WITH_EXTRA_DEVIATION,
        CARRY_FAMILY_CLASS_UNIFORM_DEFER_UNTIL_FIT,
    )
    actual_variants = tuple(report_entry.variant_name for report_entry in report.family_reports)
    if actual_variants != expected_variants:
        raise ValueError("aggregate-state runtime semantics family ordering drifted from the gated plan")
    by_variant = {entry.variant_name: entry for entry in report.family_reports}
    for family in report.family_reports:
        class_totals = sum(
            int(class_report.carry_matched_row_count)
            for class_report in family.projection_class_reports
        )
        recovered_totals = sum(
            int(class_report.recovered_dropped_count)
            for class_report in family.projection_class_reports
        )
        if int(family.carry_matched_row_count) != int(class_totals):
            raise ValueError("family carry-matched total drifted from the projection class reports")
        if family.variant_name == CARRY_FAMILY_CLASS_UNIFORM_DEFER_UNTIL_FIT:
            if int(family.recovered_dropped_count) != 0:
                raise ValueError("defer-until-fit should not claim recovered dropped mass on this observed horizon")
            if int(family.missed_represented_count) != int(recovered_totals):
                raise ValueError("defer-until-fit missed represented mass must equal the projected recoverable mass on this horizon")
        elif int(family.recovered_dropped_count) != int(recovered_totals):
            raise ValueError("family recovered-dropped total drifted from the projection class reports")
        if tuple(family.collision_class_cardinalities) != tuple(
            int(class_report.future_class_cardinality)
            for class_report in family.projection_class_reports
        ):
            raise ValueError("family collision class cardinalities drifted from the projection class reports")
        if family.variant_name == CARRY_FAMILY_CLASS_UNIFORM_DEFER_UNTIL_FIT:
            if family.lifecycle_spec.ttl_steps is None:
                raise ValueError("defer-until-fit must carry an explicit TTL")
            if family.ledger_charge.total_bits <= family.ledger_charge.base_total_bits:
                raise ValueError("defer-until-fit must recharge beyond the base feasibility note")
        else:
            if family.lifecycle_spec.ttl_steps is not None:
                raise ValueError("immediate families must not smuggle a TTL")
        if family.variant_name == CARRY_FAMILY_QUOTA_RELEASE:
            if family.terminal_label != CARRY_SEMANTICS_UNLAWFUL_REQUIRES_IDENTITY_OR_ORDER:
                raise ValueError("quota release must stay unlawful when future observable split is impossible")
            if family.false_positive_accept_count != 0:
                raise ValueError("quota release exact-claim family must not accept false positives")
        if family.variant_name in {
            CARRY_FAMILY_CLASS_UNIFORM_ACCEPT_ALL_IF_FIT,
            CARRY_FAMILY_CLASS_UNIFORM_WITH_EXTRA_DEVIATION,
        }:
            if family.initial_residual_cap <= 0:
                raise ValueError("full-class accept families require a positive residual cap")
            projected_class_mass = sum(
                int(class_report.future_class_cardinality)
                for class_report in family.projection_class_reports
            )
            if family.post_action_residual_cap < 0:
                raise ValueError("full-class accept family cannot drive residual cap negative")
            if family.post_action_residual_cap > family.initial_residual_cap:
                raise ValueError("full-class accept family cannot increase residual cap above its initial value")
            accepted_class_mass = int(family.initial_residual_cap) - int(
                family.post_action_residual_cap
            )
            if accepted_class_mass not in {0, int(projected_class_mass)}:
                raise ValueError(
                    "full-class accept family residual accounting must be all-or-none over the projected class mass"
                )
            if (
                family.terminal_label == CARRY_SEMANTICS_CANDIDATE
                and int(projected_class_mass) > int(family.initial_residual_cap)
            ):
                raise ValueError(
                    "candidate full-class accept family exceeds the initial residual cap"
                )
        if family.variant_name == CARRY_FAMILY_CLASS_UNIFORM_ACCEPT_ALL_IF_FIT and family.terminal_label == CARRY_SEMANTICS_CANDIDATE:
            if family.false_positive_accept_count != 0:
                raise ValueError("exact full-class family may only be a candidate when false positives are zero")
            if family.post_action_residual_cap != family.initial_residual_cap - projected_class_mass:
                raise ValueError(
                    "candidate full-class accept residual cap must equal the initial residual minus the accepted class mass"
                )
        if family.variant_name == CARRY_FAMILY_CLASS_UNIFORM_WITH_EXTRA_DEVIATION and family.terminal_label == CARRY_SEMANTICS_CANDIDATE:
            if family.earned_downstream_test != LEARNING_RETENTION_TOLERANCE_PROBE:
                raise ValueError("bounded-extra-deviation candidate must earn the tolerance probe")
            if family.post_action_residual_cap != family.initial_residual_cap - projected_class_mass:
                raise ValueError(
                    "candidate full-class accept residual cap must equal the initial residual minus the accepted class mass"
                )
        if family.variant_name == CARRY_FAMILY_CLASS_UNIFORM_DEFER_UNTIL_FIT and family.terminal_label == INCONCLUSIVE_NEEDS_LOOP_MEASUREMENT:
            if family.missed_represented_count <= 0:
                raise ValueError("inconclusive defer-until-fit must leave represented mass unconsumed on this horizon")
        for class_report in family.projection_class_reports:
            if class_report.future_class_count_in_bucket <= 0:
                raise ValueError("projection class must report a positive future class-count in its bucket")
            if class_report.projection_bits_for_class != _enum_bit_width(
                int(class_report.future_class_count_in_bucket)
            ):
                raise ValueError("projection bits must match the charged future-class selector width")
            if class_report.recovered_dropped_count > class_report.future_class_cardinality:
                raise ValueError("recovered dropped mass cannot exceed the projected future class")
            if class_report.matched_not_recovered_count != (
                class_report.future_class_cardinality - class_report.recovered_dropped_count
            ):
                raise ValueError("matched-not-recovered count drifted from the projected future class size")
            if class_report.future_observable_split_possible:
                raise ValueError("the committed trace should not permit a future 384->128 identity-free split")
    decision = report.terminal_decision
    if decision.candidate_only is not True or decision.dyn200_earned or decision.learner_sub2_claimed:
        raise ValueError("aggregate-state runtime semantics must stay candidate-only with no dyn200/learner claim")
    if tuple(decision.candidate_family_names) != tuple(
        family.variant_name for family in report.family_reports if family.terminal_label == CARRY_SEMANTICS_CANDIDATE
    ):
        raise ValueError("candidate family list drifted from the family terminals")
    if tuple(decision.inconclusive_family_names) != tuple(
        family.variant_name
        for family in report.family_reports
        if family.terminal_label == INCONCLUSIVE_NEEDS_LOOP_MEASUREMENT
    ):
        raise ValueError("inconclusive family list drifted from the family terminals")
    if tuple(decision.negative_family_names) != tuple(
        family.variant_name
        for family in report.family_reports
        if family.terminal_label not in {CARRY_SEMANTICS_CANDIDATE, INCONCLUSIVE_NEEDS_LOOP_MEASUREMENT}
    ):
        raise ValueError("negative family list drifted from the family terminals")
    if decision.terminal_label == CARRY_SEMANTICS_CANDIDATE:
        if not decision.learning_retention_tolerance_probe_earned:
            raise ValueError("candidate terminal must earn the downstream tolerance probe")
    elif decision.terminal_label == INCONCLUSIVE_NEEDS_LOOP_MEASUREMENT:
        if not decision.inconclusive_family_names:
            raise ValueError("inconclusive terminal requires at least one inconclusive family")
    elif decision.terminal_label == IMMEDIATE_RECOVERY_SEMANTICS_EXHAUSTED_ON_THIS_STATE_PATH:
        if decision.inconclusive_family_names:
            raise ValueError("immediate-recovery exhaustion must not keep an inconclusive family alive")
        if not decision.immediate_recovery_semantics_exhausted_on_this_state_path:
            raise ValueError("immediate-recovery exhaustion label requires the immediate-negative flag")
    else:
        raise ValueError("unexpected aggregate-state runtime semantics terminal label")
    _assert_no_tensors(report.to_dict())


@dataclass(frozen=True)
class DeferUntilFitTtl2PacketOriginReport:
    packet_name: str
    priority_rank: int
    state_key: str
    current_q_level: int
    move_direction: int
    created_from_origin_schedule_name: str
    created_from_origin_step: int
    first_projection_schedule_name: str
    first_projection_step: int
    stored_debt_count: int
    first_projection_class_cardinality: int
    first_projection_class_count_in_bucket: int
    first_projection_false_positive_mass: int
    projection_bits_for_class: int
    projected_feature_payload: dict[str, Any]
    audit_represented_mass_count: int
    audit_represented_identities_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_name": self.packet_name,
            "priority_rank": int(self.priority_rank),
            "state_key": self.state_key,
            "current_q_level": int(self.current_q_level),
            "move_direction": int(self.move_direction),
            "created_from_origin_schedule_name": self.created_from_origin_schedule_name,
            "created_from_origin_step": int(self.created_from_origin_step),
            "first_projection_schedule_name": self.first_projection_schedule_name,
            "first_projection_step": int(self.first_projection_step),
            "stored_debt_count": int(self.stored_debt_count),
            "first_projection_class_cardinality": int(self.first_projection_class_cardinality),
            "first_projection_class_count_in_bucket": int(self.first_projection_class_count_in_bucket),
            "first_projection_false_positive_mass": int(self.first_projection_false_positive_mass),
            "projection_bits_for_class": int(self.projection_bits_for_class),
            "projected_feature_payload": dict(self.projected_feature_payload),
            "audit_represented_mass_count": int(self.audit_represented_mass_count),
            "audit_represented_identities_sha256": self.audit_represented_identities_sha256,
        }


@dataclass(frozen=True)
class DeferUntilFitTtl2PacketStepReport:
    packet_name: str
    priority_rank: int
    schedule_name: str
    step: int
    age_steps: int
    residual_cap_entering_packet: int
    projected_class_count: int
    projected_class_cardinalities: tuple[int, ...]
    total_projected_class_mass: int
    projection_bits_for_class: int
    recovered_dropped_audit_mass: int
    false_positive_mass: int
    missed_represented_mass: int
    full_class_consume_legal: bool
    uses_row_identity_or_order_as_action_input: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_name": self.packet_name,
            "priority_rank": int(self.priority_rank),
            "schedule_name": self.schedule_name,
            "step": int(self.step),
            "age_steps": int(self.age_steps),
            "residual_cap_entering_packet": int(self.residual_cap_entering_packet),
            "projected_class_count": int(self.projected_class_count),
            "projected_class_cardinalities": [
                int(value) for value in self.projected_class_cardinalities
            ],
            "total_projected_class_mass": int(self.total_projected_class_mass),
            "projection_bits_for_class": int(self.projection_bits_for_class),
            "recovered_dropped_audit_mass": int(self.recovered_dropped_audit_mass),
            "false_positive_mass": int(self.false_positive_mass),
            "missed_represented_mass": int(self.missed_represented_mass),
            "full_class_consume_legal": bool(self.full_class_consume_legal),
            "uses_row_identity_or_order_as_action_input": bool(
                self.uses_row_identity_or_order_as_action_input
            ),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DeferUntilFitTtl2FutureStepReport:
    schedule_name: str
    step: int
    age_steps: int
    global_cap: int
    candidate_row_count: int
    backlog_count: int
    packet_step_reports: tuple[DeferUntilFitTtl2PacketStepReport, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_name": self.schedule_name,
            "step": int(self.step),
            "age_steps": int(self.age_steps),
            "global_cap": int(self.global_cap),
            "candidate_row_count": int(self.candidate_row_count),
            "backlog_count": int(self.backlog_count),
            "packet_step_reports": [
                report.to_dict() for report in self.packet_step_reports
            ],
        }


@dataclass(frozen=True)
class DeferUntilFitTtl2Decision:
    terminal_label: str
    ttl_steps_charged: int
    ttl_boundary_evaluated: bool
    first_fit_schedule_name: str | None
    first_fit_step: int | None
    first_fit_packet_names: tuple[str, ...]
    uses_row_identity_or_order_as_action_input: bool
    candidate_only: bool
    dyn200_earned: bool
    learner_sub2_claimed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "terminal_label": self.terminal_label,
            "ttl_steps_charged": int(self.ttl_steps_charged),
            "ttl_boundary_evaluated": bool(self.ttl_boundary_evaluated),
            "first_fit_schedule_name": self.first_fit_schedule_name,
            "first_fit_step": self.first_fit_step,
            "first_fit_packet_names": list(self.first_fit_packet_names),
            "uses_row_identity_or_order_as_action_input": bool(
                self.uses_row_identity_or_order_as_action_input
            ),
            "candidate_only": bool(self.candidate_only),
            "dyn200_earned": bool(self.dyn200_earned),
            "learner_sub2_claimed": bool(self.learner_sub2_claimed),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DeferUntilFitTtl2FitPlausibilityPrecheckReport:
    schema_version: str
    label: str
    source_bindingness: Any
    field_coverage: SourceFieldCoverage
    candidate_name: str
    source_baseline_label: str
    source_baseline_terminal_label: str
    source_aggregate_runtime_label: str
    source_aggregate_runtime_terminal_label: str
    source_defer_family_variant_name: str
    defer_family_lifecycle_spec: AggregateStateLifecycleSpec
    defer_family_ledger_charge: AggregateStateRuntimeLedgerCharge
    ttl_steps_charged: int
    allowed_action_input_dimensions: tuple[str, ...]
    forbidden_action_input_key_fragments: tuple[str, ...]
    packet_origin_reports: tuple[DeferUntilFitTtl2PacketOriginReport, ...]
    future_step_reports: tuple[DeferUntilFitTtl2FutureStepReport, ...]
    terminal_decision: DeferUntilFitTtl2Decision
    raw_arrays_included: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "source_bindingness": self.source_bindingness.to_dict(),
            "field_coverage": self.field_coverage.to_dict(),
            "candidate_name": self.candidate_name,
            "source_baseline_label": self.source_baseline_label,
            "source_baseline_terminal_label": self.source_baseline_terminal_label,
            "source_aggregate_runtime_label": self.source_aggregate_runtime_label,
            "source_aggregate_runtime_terminal_label": self.source_aggregate_runtime_terminal_label,
            "source_defer_family_variant_name": self.source_defer_family_variant_name,
            "defer_family_lifecycle_spec": self.defer_family_lifecycle_spec.to_dict(),
            "defer_family_ledger_charge": self.defer_family_ledger_charge.to_dict(),
            "ttl_steps_charged": int(self.ttl_steps_charged),
            "allowed_action_input_dimensions": list(
                self.allowed_action_input_dimensions
            ),
            "forbidden_action_input_key_fragments": list(
                self.forbidden_action_input_key_fragments
            ),
            "packet_origin_reports": [
                report.to_dict() for report in self.packet_origin_reports
            ],
            "future_step_reports": [
                report.to_dict() for report in self.future_step_reports
            ],
            "terminal_decision": self.terminal_decision.to_dict(),
            "raw_arrays_included": bool(self.raw_arrays_included),
            "non_claims": list(self.non_claims),
        }


@dataclass(frozen=True)
class _DeferUntilFitTtl2PacketContext:
    packet_name: str
    priority_rank: int
    state_key: str
    current_q_level: int
    move_direction: int
    created_from_origin_schedule_name: str
    created_from_origin_step: int
    first_projection_schedule_name: str
    first_projection_step: int
    stored_debt_count: int
    first_projection_class_cardinality: int
    first_projection_class_count_in_bucket: int
    first_projection_false_positive_mass: int
    projection_bits_for_class: int
    projected_feature_payload: dict[str, Any]
    audit_identities: frozenset[tuple[str, int]]


def _path_b_defer_until_fit_packet_name(
    *,
    state_key: str,
    current_q_level: int,
    move_direction: int,
) -> str:
    return f"{state_key}:q{int(current_q_level)}:d{int(move_direction)}"


def _path_b_defer_until_fit_packet_contexts() -> tuple[_DeferUntilFitTtl2PacketContext, ...]:
    replay_pairs = _path_b_exact_and_defer_all_replay_steps()
    by_schedule_name = {
        replay_step.schedule_name: (exact_trace_step, replay_step)
        for exact_trace_step, replay_step in replay_pairs
    }
    if "cap_saturated" not in by_schedule_name or "backlog_growth" not in by_schedule_name:
        raise ValueError(
            "defer-until-fit TTL2 precheck requires the committed cap_saturated/backlog_growth path"
        )
    origin_exact, origin_replay = by_schedule_name["cap_saturated"]
    future_exact, future_replay = by_schedule_name["backlog_growth"]
    future_bucket_feature_sets: dict[
        tuple[str, int, int], set[tuple[tuple[str, Any], ...]]
    ] = {}
    future_class_rows: dict[
        tuple[tuple[str, int, int], tuple[tuple[str, Any], ...]],
        list[_StrictObservableTieMaskRow],
    ] = {}
    for row in future_replay.observable_rows:
        future_bucket_feature_sets.setdefault(row.bucket_key, set()).add(
            row.feature_key()
        )
        future_class_rows.setdefault((row.bucket_key, row.feature_key()), []).append(row)
    identity_to_group = {
        row.identity: (row.bucket_key, row.feature_key())
        for row in future_replay.observable_rows
    }
    by_future_group: dict[
        tuple[tuple[str, int, int], tuple[tuple[str, Any], ...]],
        set[tuple[str, int]],
    ] = {}
    for identity in sorted(origin_replay.dropped_ids & future_replay.candidate_row_ids):
        group_key = identity_to_group.get(identity)
        if group_key is None:
            raise ValueError(
                "defer-until-fit TTL2 packet audit identities must stay observable on backlog_growth"
            )
        by_future_group.setdefault(group_key, set()).add(identity)
    contexts: list[_DeferUntilFitTtl2PacketContext] = []
    for priority_rank, ((bucket_key, feature_key), identities) in enumerate(
        sorted(by_future_group.items(), key=lambda item: (item[0][0], item[0][1])),
        start=1,
    ):
        class_rows = future_class_rows[(bucket_key, feature_key)]
        contexts.append(
            _DeferUntilFitTtl2PacketContext(
                packet_name=_path_b_defer_until_fit_packet_name(
                    state_key=str(bucket_key[0]),
                    current_q_level=int(bucket_key[1]),
                    move_direction=int(bucket_key[2]),
                ),
                priority_rank=int(priority_rank),
                state_key=str(bucket_key[0]),
                current_q_level=int(bucket_key[1]),
                move_direction=int(bucket_key[2]),
                created_from_origin_schedule_name=origin_replay.schedule_name,
                created_from_origin_step=int(origin_exact.schedule_step.step),
                first_projection_schedule_name=future_replay.schedule_name,
                first_projection_step=int(future_exact.schedule_step.step),
                stored_debt_count=len(identities),
                first_projection_class_cardinality=len(class_rows),
                first_projection_class_count_in_bucket=len(
                    future_bucket_feature_sets[bucket_key]
                ),
                first_projection_false_positive_mass=len(class_rows) - len(identities),
                projection_bits_for_class=_enum_bit_width(
                    len(future_bucket_feature_sets[bucket_key])
                ),
                projected_feature_payload=dict(class_rows[0].feature_payload()),
                audit_identities=frozenset(identities),
            )
        )
    if not contexts:
        raise ValueError(
            "defer-until-fit TTL2 precheck requires the committed carry packets from backlog_growth"
        )
    return tuple(contexts)


def _path_b_defer_until_fit_ttl2_future_step_specs() -> tuple[VotePressureStepSpec, ...]:
    base_step = PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE[-1]
    tensor_numel = int(next(iter(_initial_states().values())).q_levels.numel())
    specs: list[VotePressureStepSpec] = []
    for age_steps in range(1, DEFER_UNTIL_FIT_TTL2_HORIZON_STEPS + 1):
        start_index = int(base_step.start_index) + int(base_step.rows_per_tensor) * int(
            age_steps
        )
        if start_index < 0 or start_index + int(base_step.rows_per_tensor) > tensor_numel:
            break
        specs.append(
            VotePressureStepSpec(
                name=f"{base_step.name}_ttl2_future_{int(age_steps)}",
                step=int(base_step.step) + int(age_steps),
                rows_per_tensor=int(base_step.rows_per_tensor),
                start_index=int(start_index),
                vote_abs=int(base_step.vote_abs),
                cap=int(base_step.cap),
                expected_regime=(
                    f"ttl2 natural continuation {int(age_steps)} from {base_step.name}; "
                    "same pressure/cap shape, advanced row window"
                ),
            )
        )
    return tuple(specs)


def _path_b_exact_and_defer_all_replay_steps_ttl2_extension(
) -> tuple[_PathBReplayStepState, ...]:
    replay_pairs = _path_b_exact_and_defer_all_replay_steps()
    replay_states = _copy_state_map(replay_pairs[-1][1].output_states)
    replay_backlog = _copy_backlog(replay_pairs[-1][1].output_backlog)
    future_steps: list[_PathBReplayStepState] = []
    for schedule_step in _path_b_defer_until_fit_ttl2_future_step_specs():
        replay_trace_step = _path_b_current_trace_step(
            states_by_key=replay_states,
            deferred_backlog=replay_backlog,
            schedule_step=schedule_step,
        )
        replay_step = _path_b_replay_step_from_mutated_partition(replay_trace_step)
        future_steps.append(replay_step)
        replay_states = _copy_state_map(replay_step.output_states)
        replay_backlog = _copy_backlog(replay_step.output_backlog)
    return tuple(future_steps)


def _path_b_defer_until_fit_ttl2_packet_origin_reports(
    packet_contexts: Sequence[_DeferUntilFitTtl2PacketContext],
) -> tuple[DeferUntilFitTtl2PacketOriginReport, ...]:
    return tuple(
        DeferUntilFitTtl2PacketOriginReport(
            packet_name=context.packet_name,
            priority_rank=int(context.priority_rank),
            state_key=context.state_key,
            current_q_level=int(context.current_q_level),
            move_direction=int(context.move_direction),
            created_from_origin_schedule_name=context.created_from_origin_schedule_name,
            created_from_origin_step=int(context.created_from_origin_step),
            first_projection_schedule_name=context.first_projection_schedule_name,
            first_projection_step=int(context.first_projection_step),
            stored_debt_count=int(context.stored_debt_count),
            first_projection_class_cardinality=int(
                context.first_projection_class_cardinality
            ),
            first_projection_class_count_in_bucket=int(
                context.first_projection_class_count_in_bucket
            ),
            first_projection_false_positive_mass=int(
                context.first_projection_false_positive_mass
            ),
            projection_bits_for_class=int(context.projection_bits_for_class),
            projected_feature_payload=dict(context.projected_feature_payload),
            audit_represented_mass_count=int(len(context.audit_identities)),
            audit_represented_identities_sha256=_identity_sha256(
                set(context.audit_identities)
            ),
        )
        for context in packet_contexts
    )


def _path_b_defer_until_fit_ttl2_packet_step_report(
    *,
    packet_context: _DeferUntilFitTtl2PacketContext,
    future_step: _PathBReplayStepState,
    age_steps: int,
    residual_cap_entering_packet: int,
) -> DeferUntilFitTtl2PacketStepReport:
    identity_to_group = {
        row.identity: (row.bucket_key, row.feature_key())
        for row in future_step.observable_rows
    }
    class_rows_by_group: dict[
        tuple[tuple[str, int, int], tuple[tuple[str, Any], ...]],
        list[_StrictObservableTieMaskRow],
    ] = {}
    for row in future_step.observable_rows:
        class_rows_by_group.setdefault((row.bucket_key, row.feature_key()), []).append(
            row
        )
    present_ids = set(packet_context.audit_identities) & future_step.candidate_row_ids
    by_group: dict[
        tuple[tuple[str, int, int], tuple[tuple[str, Any], ...]],
        set[tuple[str, int]],
    ] = {}
    for identity in sorted(present_ids):
        group_key = identity_to_group.get(identity)
        if group_key is None:
            raise ValueError(
                "TTL2 future packet audit mass must stay observable when represented"
            )
        by_group.setdefault(group_key, set()).add(identity)
    projected_group_keys = sorted(by_group, key=lambda item: (item[0], item[1]))
    projected_class_cardinalities = tuple(
        len(class_rows_by_group[group_key]) for group_key in projected_group_keys
    )
    total_projected_class_mass = int(sum(projected_class_cardinalities))
    recovered_dropped_audit_mass = int(len(present_ids))
    false_positive_mass = int(
        total_projected_class_mass - recovered_dropped_audit_mass
    )
    full_class_consume_legal = bool(
        len(projected_group_keys) == 1
        and total_projected_class_mass > 0
        and total_projected_class_mass <= int(residual_cap_entering_packet)
    )
    missed_represented_mass = int(
        0 if full_class_consume_legal else recovered_dropped_audit_mass
    )
    if not projected_group_keys:
        reason = (
            "packet audit mass is not represented as a candidate class on this future step"
        )
    elif len(projected_group_keys) > 1:
        reason = (
            "packet audit mass split across multiple future observable classes, so no single class packet can be consumed lawfully"
        )
    elif total_projected_class_mass > int(residual_cap_entering_packet):
        reason = (
            f"projected class mass {int(total_projected_class_mass)} exceeds residual cap {int(residual_cap_entering_packet)}"
        )
    else:
        reason = "single projected class fits residual cap and is lawful to consume"
    return DeferUntilFitTtl2PacketStepReport(
        packet_name=packet_context.packet_name,
        priority_rank=int(packet_context.priority_rank),
        schedule_name=future_step.schedule_name,
        step=int(future_step.step),
        age_steps=int(age_steps),
        residual_cap_entering_packet=int(residual_cap_entering_packet),
        projected_class_count=int(len(projected_group_keys)),
        projected_class_cardinalities=tuple(int(value) for value in projected_class_cardinalities),
        total_projected_class_mass=int(total_projected_class_mass),
        projection_bits_for_class=int(packet_context.projection_bits_for_class),
        recovered_dropped_audit_mass=int(recovered_dropped_audit_mass),
        false_positive_mass=int(false_positive_mass),
        missed_represented_mass=int(missed_represented_mass),
        full_class_consume_legal=bool(full_class_consume_legal),
        uses_row_identity_or_order_as_action_input=False,
        reason=reason,
    )


def _path_b_defer_until_fit_ttl2_future_step_report(
    *,
    packet_contexts: Sequence[_DeferUntilFitTtl2PacketContext],
    future_step: _PathBReplayStepState,
    age_steps: int,
) -> DeferUntilFitTtl2FutureStepReport:
    residual_cap = int(future_step.global_cap)
    packet_step_reports: list[DeferUntilFitTtl2PacketStepReport] = []
    for packet_context in packet_contexts:
        packet_report = _path_b_defer_until_fit_ttl2_packet_step_report(
            packet_context=packet_context,
            future_step=future_step,
            age_steps=age_steps,
            residual_cap_entering_packet=residual_cap,
        )
        packet_step_reports.append(packet_report)
        if packet_report.full_class_consume_legal:
            residual_cap -= int(packet_report.total_projected_class_mass)
    return DeferUntilFitTtl2FutureStepReport(
        schedule_name=future_step.schedule_name,
        step=int(future_step.step),
        age_steps=int(age_steps),
        global_cap=int(future_step.global_cap),
        candidate_row_count=int(len(future_step.candidate_row_ids)),
        backlog_count=int(len(future_step.backlog_ids)),
        packet_step_reports=tuple(packet_step_reports),
    )


def _path_b_defer_until_fit_ttl2_non_claims() -> tuple[str, ...]:
    return (
        "CPU-only TTL2 fit-plausibility precheck over the committed defer-all replay path",
        "TTL2 is a new local seam layered on top of the banked 4-step baseline; the fixed baseline validator semantics remain unchanged",
        "the natural future horizon is derived from backlog_growth with the same pressure/cap shape and advancing row windows",
        "no cap-relief control is present in the candidate decision",
        "action inputs remain class-level only; row identity/order are audit-only and never consume inputs",
        "candidate-only; no dyn200, no learner/sub-2 claim, no raw per-weight arrays",
    )


def _path_b_defer_until_fit_ttl2_terminal_decision(
    future_step_reports: Sequence[DeferUntilFitTtl2FutureStepReport],
) -> DeferUntilFitTtl2Decision:
    ttl_boundary_evaluated = bool(
        len(future_step_reports) == DEFER_UNTIL_FIT_TTL2_HORIZON_STEPS
    )
    first_fit_step = None
    first_fit_packets: tuple[str, ...] = ()
    for future_step in future_step_reports:
        fit_packets = tuple(
            packet.packet_name
            for packet in future_step.packet_step_reports
            if packet.full_class_consume_legal
        )
        if fit_packets:
            first_fit_step = future_step
            first_fit_packets = fit_packets
            break
    if not ttl_boundary_evaluated:
        return DeferUntilFitTtl2Decision(
            terminal_label=DEFER_UNTIL_FIT_HORIZON_MEASUREMENT_INCONCLUSIVE,
            ttl_steps_charged=DEFER_UNTIL_FIT_TTL2_HORIZON_STEPS,
            ttl_boundary_evaluated=False,
            first_fit_schedule_name=None,
            first_fit_step=None,
            first_fit_packet_names=(),
            uses_row_identity_or_order_as_action_input=False,
            candidate_only=True,
            dyn200_earned=False,
            learner_sub2_claimed=False,
            reason=(
                "the TTL boundary could not be evaluated from the local natural extension steps, so the honest result is horizon inconclusive"
            ),
        )
    if first_fit_step is not None:
        return DeferUntilFitTtl2Decision(
            terminal_label=DEFER_UNTIL_FIT_FIRST_FIT_PLAUSIBLE_CANDIDATE_ONLY,
            ttl_steps_charged=DEFER_UNTIL_FIT_TTL2_HORIZON_STEPS,
            ttl_boundary_evaluated=True,
            first_fit_schedule_name=first_fit_step.schedule_name,
            first_fit_step=int(first_fit_step.step),
            first_fit_packet_names=tuple(first_fit_packets),
            uses_row_identity_or_order_as_action_input=False,
            candidate_only=True,
            dyn200_earned=False,
            learner_sub2_claimed=False,
            reason=(
                "at least one full carried class packet fits before TTL expiry on the natural defer-all replay path, so the charged defer family earns a later full horizon/tolerance slice only"
            ),
        )
    ttl_boundary = future_step_reports[-1]
    ttl_boundary_masses = tuple(
        int(packet.total_projected_class_mass)
        for packet in ttl_boundary.packet_step_reports
    )
    return DeferUntilFitTtl2Decision(
        terminal_label=DEFER_UNTIL_FIT_TTL2_NO_FIT_ON_STATE_PATH,
        ttl_steps_charged=DEFER_UNTIL_FIT_TTL2_HORIZON_STEPS,
        ttl_boundary_evaluated=True,
        first_fit_schedule_name=None,
        first_fit_step=None,
        first_fit_packet_names=(),
        uses_row_identity_or_order_as_action_input=False,
        candidate_only=True,
        dyn200_earned=False,
        learner_sub2_claimed=False,
        reason=(
            f"no carried class packet fits by ttl_steps={int(DEFER_UNTIL_FIT_TTL2_HORIZON_STEPS)} on this state path; at the TTL boundary {ttl_boundary.schedule_name} the projected class masses {ttl_boundary_masses} still exceed residual cap {int(ttl_boundary.global_cap)}"
        ),
    )


def run_defer_until_fit_ttl2_fit_plausibility_precheck(
) -> DeferUntilFitTtl2FitPlausibilityPrecheckReport:
    baseline_report = run_path_b_defer_all_baseline_parity_probe()
    validate_path_b_defer_all_baseline_parity_probe_report(baseline_report)
    if baseline_report.terminal_decision.terminal_label != CARRY_CANDIDATE_EARNED:
        raise ValueError(
            "TTL2 defer-until-fit precheck requires the committed carry-candidate baseline source"
        )
    aggregate_report = run_path_b_aggregate_state_runtime_semantics_definition()
    validate_path_b_aggregate_state_runtime_semantics_report(aggregate_report)
    if (
        aggregate_report.terminal_decision.terminal_label
        != INCONCLUSIVE_NEEDS_LOOP_MEASUREMENT
    ):
        raise ValueError(
            "TTL2 defer-until-fit precheck requires the committed aggregate-runtime inconclusive source"
        )
    family_by_variant = {
        family.variant_name: family for family in aggregate_report.family_reports
    }
    defer_family = family_by_variant[CARRY_FAMILY_CLASS_UNIFORM_DEFER_UNTIL_FIT]
    if defer_family.lifecycle_spec.ttl_steps != DEFER_UNTIL_FIT_TTL2_HORIZON_STEPS:
        raise ValueError(
            "TTL2 defer-until-fit precheck requires the charged ttl_steps=2 lifecycle"
        )
    packet_contexts = _path_b_defer_until_fit_packet_contexts()
    packet_origin_reports = _path_b_defer_until_fit_ttl2_packet_origin_reports(
        packet_contexts
    )
    future_replay_steps = _path_b_exact_and_defer_all_replay_steps_ttl2_extension()
    future_step_reports = tuple(
        _path_b_defer_until_fit_ttl2_future_step_report(
            packet_contexts=packet_contexts,
            future_step=future_step,
            age_steps=age_steps,
        )
        for age_steps, future_step in enumerate(future_replay_steps, start=1)
    )
    bindingness = pre_register_source_bindingness(
        source_kind=SOURCE_KIND_GENERATED_NATIVE_LOOP,
        coverage=SourceFieldCoverage.full_generated_native_loop(),
    )
    return DeferUntilFitTtl2FitPlausibilityPrecheckReport(
        schema_version=PATH_B_DEFER_UNTIL_FIT_TTL2_PLAUSIBILITY_PRECHECK_SCHEMA_VERSION,
        label=PATH_B_DEFER_UNTIL_FIT_TTL2_PLAUSIBILITY_PRECHECK_LABEL,
        source_bindingness=bindingness,
        field_coverage=SourceFieldCoverage.full_generated_native_loop(),
        candidate_name=PATH_B_DEFER_UNTIL_FIT_TTL2_PLAUSIBILITY_PRECHECK_CANDIDATE,
        source_baseline_label=baseline_report.label,
        source_baseline_terminal_label=baseline_report.terminal_decision.terminal_label,
        source_aggregate_runtime_label=aggregate_report.label,
        source_aggregate_runtime_terminal_label=(
            aggregate_report.terminal_decision.terminal_label
        ),
        source_defer_family_variant_name=CARRY_FAMILY_CLASS_UNIFORM_DEFER_UNTIL_FIT,
        defer_family_lifecycle_spec=defer_family.lifecycle_spec,
        defer_family_ledger_charge=defer_family.ledger_charge,
        ttl_steps_charged=DEFER_UNTIL_FIT_TTL2_HORIZON_STEPS,
        allowed_action_input_dimensions=(
            DEFER_UNTIL_FIT_TTL2_ALLOWED_ACTION_INPUT_DIMENSIONS
        ),
        forbidden_action_input_key_fragments=(
            DEFER_UNTIL_FIT_TTL2_FORBIDDEN_ACTION_INPUT_KEY_FRAGMENTS
        ),
        packet_origin_reports=packet_origin_reports,
        future_step_reports=future_step_reports,
        terminal_decision=_path_b_defer_until_fit_ttl2_terminal_decision(
            future_step_reports
        ),
        raw_arrays_included=False,
        non_claims=_path_b_defer_until_fit_ttl2_non_claims(),
    )


def validate_defer_until_fit_ttl2_fit_plausibility_precheck_report(
    report: DeferUntilFitTtl2FitPlausibilityPrecheckReport,
) -> None:
    if (
        report.schema_version
        != PATH_B_DEFER_UNTIL_FIT_TTL2_PLAUSIBILITY_PRECHECK_SCHEMA_VERSION
    ):
        raise ValueError("unexpected defer-until-fit TTL2 precheck schema version")
    if report.label != PATH_B_DEFER_UNTIL_FIT_TTL2_PLAUSIBILITY_PRECHECK_LABEL:
        raise ValueError("unexpected defer-until-fit TTL2 precheck label")
    if (
        report.candidate_name
        != PATH_B_DEFER_UNTIL_FIT_TTL2_PLAUSIBILITY_PRECHECK_CANDIDATE
    ):
        raise ValueError("unexpected defer-until-fit TTL2 precheck candidate")
    if report.source_baseline_label != PATH_B_DEFER_ALL_BASELINE_PARITY_PROBE_LABEL:
        raise ValueError("TTL2 precheck must cite the committed baseline label")
    if report.source_baseline_terminal_label != CARRY_CANDIDATE_EARNED:
        raise ValueError("TTL2 precheck must inherit the carry-candidate baseline")
    if report.source_aggregate_runtime_label != PATH_B_AGGREGATE_STATE_RUNTIME_SEMANTICS_LABEL:
        raise ValueError("TTL2 precheck must cite the committed aggregate-runtime label")
    if report.source_aggregate_runtime_terminal_label != INCONCLUSIVE_NEEDS_LOOP_MEASUREMENT:
        raise ValueError("TTL2 precheck must inherit the aggregate-runtime inconclusive source")
    if report.source_defer_family_variant_name != CARRY_FAMILY_CLASS_UNIFORM_DEFER_UNTIL_FIT:
        raise ValueError("TTL2 precheck must stay on the charged defer-until-fit family")
    if int(report.ttl_steps_charged) != int(DEFER_UNTIL_FIT_TTL2_HORIZON_STEPS):
        raise ValueError("TTL2 precheck must keep the charged ttl_steps=2 horizon")
    if report.defer_family_lifecycle_spec.ttl_steps != DEFER_UNTIL_FIT_TTL2_HORIZON_STEPS:
        raise ValueError("TTL2 precheck must report the charged ttl_steps=2 lifecycle")
    if report.defer_family_ledger_charge.ttl_bits != 4:
        raise ValueError("TTL2 precheck must report the charged ttl bits")
    if report.defer_family_ledger_charge.age_bits != 4:
        raise ValueError("TTL2 precheck must report the charged age bits")
    if report.defer_family_ledger_charge.active_slot_bits != 2:
        raise ValueError("TTL2 precheck must report the charged active-slot bits")
    if report.defer_family_ledger_charge.projection_bits != 2:
        raise ValueError("TTL2 precheck must report the charged projection bits")
    if tuple(report.forbidden_action_input_key_fragments) != (
        DEFER_UNTIL_FIT_TTL2_FORBIDDEN_ACTION_INPUT_KEY_FRAGMENTS
    ):
        raise ValueError("TTL2 forbidden action-input fragments drifted from the gate")
    for key in report.allowed_action_input_dimensions:
        lowered = key.lower()
        if any(
            fragment in lowered
            for fragment in report.forbidden_action_input_key_fragments
        ):
            raise ValueError(
                "TTL2 action-input schema leaked a forbidden identity/order key"
            )
    if tuple(report.allowed_action_input_dimensions) != (
        DEFER_UNTIL_FIT_TTL2_ALLOWED_ACTION_INPUT_DIMENSIONS
    ):
        raise ValueError("TTL2 action-input schema drifted from the approved class-level input set")
    if not report.packet_origin_reports:
        raise ValueError("TTL2 precheck requires at least one carried packet")
    seen_packet_names: set[str] = set()
    for expected_rank, packet in enumerate(report.packet_origin_reports, start=1):
        if packet.packet_name in seen_packet_names:
            raise ValueError("TTL2 packet names must stay unique")
        seen_packet_names.add(packet.packet_name)
        if int(packet.priority_rank) != int(expected_rank):
            raise ValueError("TTL2 packet priority ranks must stay contiguous and ordered")
        if int(packet.audit_represented_mass_count) != int(packet.stored_debt_count):
            raise ValueError("stored debt must equal the audited dropped mass for each packet")
        if int(packet.first_projection_false_positive_mass) != (
            int(packet.first_projection_class_cardinality)
            - int(packet.audit_represented_mass_count)
        ):
            raise ValueError("first-projection false-positive mass drifted from class minus audit mass")
        if int(packet.projection_bits_for_class) != _enum_bit_width(
            int(packet.first_projection_class_count_in_bucket)
        ):
            raise ValueError("TTL2 packet projection bits drifted from the first projection selector width")
    expected_future_specs = _path_b_defer_until_fit_ttl2_future_step_specs()
    if report.terminal_decision.terminal_label != DEFER_UNTIL_FIT_HORIZON_MEASUREMENT_INCONCLUSIVE:
        if len(report.future_step_reports) != len(expected_future_specs):
            raise ValueError("TTL2 precheck must evaluate the full charged horizon when not inconclusive")
    residual_packet_names = tuple(packet.packet_name for packet in report.packet_origin_reports)
    for future_step_report, expected_spec in zip(
        report.future_step_reports, expected_future_specs
    ):
        if future_step_report.schedule_name != expected_spec.name:
            raise ValueError("TTL2 future step names drifted from the natural extension spec")
        if int(future_step_report.step) != int(expected_spec.step):
            raise ValueError("TTL2 future step numbers drifted from the natural extension spec")
        if int(future_step_report.age_steps) != int(
            expected_spec.step - PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE[-1].step
        ):
            raise ValueError("TTL2 future step ages drifted from the charged horizon index")
        if int(future_step_report.global_cap) != int(expected_spec.cap):
            raise ValueError("TTL2 future step cap drifted from backlog_growth")
        if tuple(packet.packet_name for packet in future_step_report.packet_step_reports) != residual_packet_names:
            raise ValueError("TTL2 future steps must report every carried packet in priority order")
        residual_cap = int(future_step_report.global_cap)
        for packet_step_report, packet_origin in zip(
            future_step_report.packet_step_reports,
            report.packet_origin_reports,
        ):
            if int(packet_step_report.priority_rank) != int(packet_origin.priority_rank):
                raise ValueError("TTL2 packet-step priority drifted from the packet origin ordering")
            if int(packet_step_report.age_steps) != int(future_step_report.age_steps):
                raise ValueError("TTL2 packet-step age drifted from the future step")
            if int(packet_step_report.residual_cap_entering_packet) != int(residual_cap):
                raise ValueError("TTL2 residual cap entering packet drifted from the priority simulation")
            if int(packet_step_report.projected_class_count) != len(
                packet_step_report.projected_class_cardinalities
            ):
                raise ValueError("TTL2 projected class count drifted from the cardinality tuple")
            if int(packet_step_report.total_projected_class_mass) != sum(
                int(value)
                for value in packet_step_report.projected_class_cardinalities
            ):
                raise ValueError("TTL2 projected class mass drifted from the class cardinalities")
            if int(packet_step_report.false_positive_mass) != (
                int(packet_step_report.total_projected_class_mass)
                - int(packet_step_report.recovered_dropped_audit_mass)
            ):
                raise ValueError("TTL2 false-positive mass drifted from projected mass minus audit mass")
            if bool(packet_step_report.uses_row_identity_or_order_as_action_input):
                raise ValueError("TTL2 packet consume path must stay class-level with no identity/order input")
            if packet_step_report.full_class_consume_legal:
                if int(packet_step_report.projected_class_count) != 1:
                    raise ValueError("TTL2 lawful consume requires exactly one projected class packet")
                if int(packet_step_report.total_projected_class_mass) <= 0:
                    raise ValueError("TTL2 lawful consume requires a positive projected class mass")
                if int(packet_step_report.total_projected_class_mass) > int(residual_cap):
                    raise ValueError("TTL2 lawful consume cannot exceed residual cap")
                if int(packet_step_report.missed_represented_mass) != 0:
                    raise ValueError("TTL2 lawful consume must clear the represented audit mass on that step")
                residual_cap -= int(packet_step_report.total_projected_class_mass)
            else:
                expected_missed = int(packet_step_report.recovered_dropped_audit_mass)
                if int(packet_step_report.missed_represented_mass) != expected_missed:
                    raise ValueError("TTL2 non-consuming packet must mark all represented audit mass as missed")
    decision = report.terminal_decision
    if bool(decision.uses_row_identity_or_order_as_action_input):
        raise ValueError("TTL2 terminal decision must stay class-level with no identity/order input")
    if decision.candidate_only is not True or decision.dyn200_earned or decision.learner_sub2_claimed:
        raise ValueError("TTL2 precheck must stay candidate-only with no dyn200/learner claim")
    any_legal = any(
        packet.full_class_consume_legal
        for future_step in report.future_step_reports
        for packet in future_step.packet_step_reports
    )
    if decision.terminal_label == DEFER_UNTIL_FIT_FIRST_FIT_PLAUSIBLE_CANDIDATE_ONLY:
        if not any_legal:
            raise ValueError("TTL2 first-fit candidate terminal requires a lawful consume event")
        if decision.first_fit_schedule_name is None or decision.first_fit_step is None:
            raise ValueError("TTL2 first-fit candidate terminal must record the first-fit step")
        if not decision.first_fit_packet_names:
            raise ValueError("TTL2 first-fit candidate terminal must record the fitting packet names")
    elif decision.terminal_label == DEFER_UNTIL_FIT_TTL2_NO_FIT_ON_STATE_PATH:
        if any_legal:
            raise ValueError("TTL2 no-fit terminal cannot coexist with a lawful consume event")
        if not decision.ttl_boundary_evaluated:
            raise ValueError("TTL2 no-fit terminal requires the TTL boundary to be evaluated")
        if decision.first_fit_schedule_name is not None or decision.first_fit_step is not None:
            raise ValueError("TTL2 no-fit terminal must not report a first-fit step")
    elif decision.terminal_label == DEFER_UNTIL_FIT_HORIZON_MEASUREMENT_INCONCLUSIVE:
        if decision.ttl_boundary_evaluated:
            raise ValueError("TTL2 horizon inconclusive must only be used when the TTL boundary is unevaluated")
    else:
        raise ValueError("unexpected TTL2 precheck terminal label")
    if bool(report.raw_arrays_included):
        raise ValueError("TTL2 precheck must not include raw arrays")
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
    "ONLINE_ESTIMABILITY_TIE_MASK_LABEL",
    "PATH_B_IDENTITY_FREE_TIE_RULE_CLASSIFIER_LABEL",
    "PATH_B_IDENTITY_FREE_TIE_RULE_CLASSIFIER_SCHEMA_VERSION",
    "PATH_B_IDENTITY_FREE_TIE_RULE_CLASSIFIER_CANDIDATE",
    "ONLINE_ESTIMABILITY_TIE_MASK_SCHEMA_VERSION",
    "ONLINE_ESTIMABLE_TIE_MASK_CANDIDATE",
    "DECISION_STATISTIC_UPPER_BOUND_PASS",
    "DECISION_STATISTIC_UPPER_BOUND_SCHEMA_VERSION",
    "HOT_BUDGET_POINT_LABELS",
    "OBSERVED_TIE_RESERVATION_DIAGNOSTIC",
    "K_SWEEP_JOINT_INFEASIBLE",
    "K_SWEEP_MINIMAL_VIABLE_PASS",
    "K_SWEEP_REPRESENTATION_WALL",
    "OBSERVABLE_RANK_FEATURES_INSUFFICIENT",
    "STRICT_OBSERVABLE_TIE_MASK_EXACT_RECOVERABLE_IDENTITY_FREE_CANDIDATE_ONLY",
    "STRICT_OBSERVABLE_TIE_MASK_NOT_IDENTIFIABLE_IDENTITY_BOUND",
    "STRICT_OBSERVABLE_TIE_MASK_PARTIALLY_RECOVERABLE_NOT_EXACT",
    "CLASS_ACTION_ACCEPT_ALL_MIXED_CLASSES",
    "CLASS_ACTION_DEFER_ALL_MIXED_CLASSES_NO_BACKFILL",
    "STRICTLY_NEW_EMITTED_IDENTITY_FREE_OBSERVABLE_SPLIT",
    "AGGREGATE_STATE_REDEFINITION",
    "CANDIDATE_FAMILY_REQUIRES_IDENTITY_OR_ORDER_SUBSET_SELECTION",
    "CANDIDATE_FAMILY_CLASS_UNIFORM_CAP_OVERFLOW_NEGATIVE",
    "CANDIDATE_FAMILY_CLASS_UNIFORM_BOUNDED_DEVIATION_CANDIDATE_ONLY",
    "CANDIDATE_FAMILY_NO_EMITTED_IDENTITY_FREE_SPLIT_OBSERVABLE",
    "CANDIDATE_FAMILY_EMITTED_IDENTITY_FREE_SPLIT_CANDIDATE_ONLY",
    "CANDIDATE_FAMILY_AGGREGATE_STATE_UNBOUNDED_PERSISTENT_BITS_NEGATIVE",
    "CANDIDATE_FAMILY_AGGREGATE_STATE_BOUNDED_PERSISTENT_BITS_CANDIDATE_ONLY",
    "CANDIDATE_FAMILY_AGGREGATE_STATE_RUNTIME_SEMANTICS_UNSPECIFIED",
    "RUNTIME_TIE_RULE_MUTATION_PARITY_PROBE",
    "LEARNING_RETENTION_TOLERANCE_PROBE",
    "PATH_B_DEFER_ALL_BASELINE_PARITY_PROBE_SCHEMA_VERSION",
    "PATH_B_DEFER_ALL_BASELINE_PARITY_PROBE_LABEL",
    "PATH_B_DEFER_ALL_BASELINE_PARITY_PROBE_CANDIDATE",
    "PATH_B_AGGREGATE_STATE_RUNTIME_SEMANTICS_SCHEMA_VERSION",
    "PATH_B_AGGREGATE_STATE_RUNTIME_SEMANTICS_LABEL",
    "PATH_B_AGGREGATE_STATE_RUNTIME_SEMANTICS_CANDIDATE",
    "PATH_B_DEFER_UNTIL_FIT_TTL2_PLAUSIBILITY_PRECHECK_SCHEMA_VERSION",
    "PATH_B_DEFER_UNTIL_FIT_TTL2_PLAUSIBILITY_PRECHECK_LABEL",
    "PATH_B_DEFER_UNTIL_FIT_TTL2_PLAUSIBILITY_PRECHECK_CANDIDATE",
    "BASELINE_SUFFICIENT_NO_CARRY_NEEDED",
    "CARRY_CANDIDATE_EARNED",
    "CARRY_SEMANTICS_CANDIDATE",
    "CARRY_SEMANTICS_UNLAWFUL_REQUIRES_IDENTITY_OR_ORDER",
    "CARRY_SEMANTICS_CAP_OVERFLOW_OR_BITS_UNBOUNDED",
    "INCONCLUSIVE_NEEDS_LOOP_MEASUREMENT",
    "DEFER_UNTIL_FIT_FIRST_FIT_PLAUSIBLE_CANDIDATE_ONLY",
    "DEFER_UNTIL_FIT_TTL2_NO_FIT_ON_STATE_PATH",
    "DEFER_UNTIL_FIT_HORIZON_MEASUREMENT_INCONCLUSIVE",
    "IMMEDIATE_RECOVERY_SEMANTICS_EXHAUSTED_ON_THIS_STATE_PATH",
    "CARRY_FAMILY_QUOTA_RELEASE",
    "CARRY_FAMILY_CLASS_UNIFORM_ACCEPT_ALL_IF_FIT",
    "CARRY_FAMILY_CLASS_UNIFORM_WITH_EXTRA_DEVIATION",
    "CARRY_FAMILY_CLASS_UNIFORM_DEFER_UNTIL_FIT",
    "CARRY_SUBCASE_EXACT_RECOVERY",
    "CARRY_SUBCASE_CLASS_UNIFORM_BOUNDED_EXTRA_DEVIATION",
    "AGGREGATE_RUNTIME_SEMANTICS_DEFINITION_PLAN",
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
    "STRICT_OBSERVABLE_TIE_MASK_SHUFFLE_FALSIFIER",
    "CAP_PRESSURE_FRONTIER_ONLY_UNDERFILL_NO_REALLOCATION",
    "CAP_PRESSURE_FRONTIER_OVERFLOW_REQUIRES_ILLEGAL_SUBSET_SELECTION",
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
    "ObservableTieMaskBucketReport",
    "ObservableTieMaskFeatureClassReport",
    "ObservableTieMaskOnlineEstimabilityDecision",
    "ObservableTieMaskOnlineEstimabilityReport",
    "PathBStepDeviationReport",
    "PathBDeviationVectorSummary",
    "PathBPersistentLedgerCharge",
    "PathBMechanismFamilyReport",
    "PathBClassifierDecision",
    "PathBIdentityFreeTieRuleClassifierReport",
    "PathBDeferAllBaselineStepReport",
    "PathBDroppedMassOriginReport",
    "PathBDeferAllBaselineDecision",
    "PathBDeferAllBaselineParityProbeReport",
    "AggregateStateRuntimeLedgerCharge",
    "DeferUntilFitTtl2PacketOriginReport",
    "DeferUntilFitTtl2PacketStepReport",
    "DeferUntilFitTtl2FutureStepReport",
    "DeferUntilFitTtl2Decision",
    "DeferUntilFitTtl2FitPlausibilityPrecheckReport",
    "AggregateStateLifecycleSpec",
    "AggregateStateProjectionClassReport",
    "AggregateStateRuntimeFamilyReport",
    "AggregateStateRuntimeSemanticsDecision",
    "AggregateStateRuntimeSemanticsReport",
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
    "run_online_estimable_tie_mask_diagnostic",
    "run_path_b_identity_free_tie_rule_classifier",
    "run_path_b_defer_all_baseline_parity_probe",
    "run_path_b_aggregate_state_runtime_semantics_definition",
    "run_defer_until_fit_ttl2_fit_plausibility_precheck",
    "run_real_backlog_lower_bound_diagnostic",
    "run_tie_frontier_reservation_lower_bound_diagnostic",
    "run_scale_appropriate_b_storage_comparison",
    "run_representative_bounded_delta_drift_verdict",
    "validate_candidate_admission_diagnostic_report",
    "validate_candidate_capacity_localization_report",
    "validate_decision_statistic_upper_bound_report",
    "validate_online_estimable_tie_mask_report",
    "validate_path_b_identity_free_tie_rule_classifier_report",
    "validate_path_b_defer_all_baseline_parity_probe_report",
    "validate_path_b_aggregate_state_runtime_semantics_report",
    "validate_defer_until_fit_ttl2_fit_plausibility_precheck_report",
    "validate_real_backlog_lower_bound_diagnostic_report",
    "validate_tie_frontier_reservation_lower_bound_report",
    "validate_scale_appropriate_b_storage_comparison_report",
    "validate_representative_bounded_delta_drift_verdict_report",
    "VIRTUAL_DECISION_STATISTIC_CANDIDATE",
]
