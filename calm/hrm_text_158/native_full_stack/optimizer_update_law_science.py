"""Pre-full-stack diagnostic packet contract for optimizer/update-law science.

This module is not a readiness row and not a launch runner. It encodes the
Step-1 authoring packet for the rank-bucket vs rank-free sign-pressure
diagnostic, including the ordering-matched A1 control that isolates the
vote-law variable from tie/order effects.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION = (
    "hrm_text_158_optimizer_update_law_science/v0.pre_full_stack_diagnostic"
)
OPTIMIZER_UPDATE_LAW_SCIENCE_TARGET_NAME = "step1_optimizer_update_law_science_packet"
DIAGNOSTIC_CLASS_PRE_FULL_STACK = "pre_full_stack_diagnostic"
STEP1_DRY_RUN_PACKET_KIND = "step1_dry_run_packet"
STEP2_LAUNCH_BUNDLE_PACKET_KIND = "step2_launch_bundle"
STEP3_MEASUREMENT_POWER_TRUST_REGION_PACKET_KIND = (
    "step3_measurement_power_then_trust_region_packet"
)
STEP4_POWERED_RANK_SIGNAL_DECOMPOSITION_PACKET_KIND = (
    "step4_powered_rank_signal_decomposition_packet"
)
STEP5_SUPPORT_ORDER_TRAJECTORY_ROBUSTNESS_PACKET_KIND = (
    "support_order_trajectory_robustness"
)
STEP6_ORDER_AVERAGED_A0_COMPONENT_DECOMPOSITION_PACKET_KIND = (
    "step6_order_averaged_a0_component_decomposition_packet"
)
ORACLE_SCREEN_PACKET_KIND = (
    "pre_full_stack_diagnostic__candidate_set_viability_oracle_screen"
)
ORACLE_SCREEN_LAUNCH_BUNDLE_PACKET_KIND = (
    "candidate_set_viability_oracle_screen_launch_bundle"
)
CREDIT_RANKING_PIVOT_MEASUREMENT_PACKET_KIND = (
    "pre_full_stack_diagnostic__credit_ranking_pivot_measurement"
)
CREDIT_RANKING_PIVOT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND = (
    "credit_ranking_pivot_measurement_launch_bundle"
)
WITHIN_TIE_BAND_DISCRIMINATOR_PACKET_KIND = (
    "pre_full_stack_diagnostic__within_tie_band_discriminator_measurement"
)
WITHIN_TIE_BAND_DISCRIMINATOR_LAUNCH_BUNDLE_PACKET_KIND = (
    "within_tie_band_discriminator_measurement_launch_bundle"
)
ACTIVATION_CREDIT_MEASUREMENT_PACKET_KIND = (
    "pre_full_stack_diagnostic__activation_credit_measurement"
)
ACTIVATION_CREDIT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND = (
    "activation_credit_measurement_launch_bundle"
)
ACTIVATION_CREDIT_SCALE_SMOKE_LAUNCH_BUNDLE_PACKET_KIND = (
    "activation_credit_scale_smoke_launch_bundle"
)

SCIENCE_MODE_PRETERMINAL_SCREEN = "preterminal_screen"
SCIENCE_MODE_BRANCH_VERDICT = "branch_verdict"
SCIENCE_MODE_ROWS = {
    SCIENCE_MODE_PRETERMINAL_SCREEN: 20,
    SCIENCE_MODE_BRANCH_VERDICT: 50,
}

ARM_A0_RANK_BUCKET_CURRENT = "A0_rank_bucket_current_ordering"
ARM_A1_RANK_BUCKET_ORDER_MATCHED = "A1_rank_bucket_order_matched"
ARM_B_RANK_FREE_SIGN_PRESSURE = "B_rank_free_sign_pressure"
ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER = "C_rank_free_sign_current_margin_order"
ARM_B_CAP_MAX_ABS_1024 = "B_cap_max_abs_1024"
ARM_INVERTED_SIGN_PRESSURE = "inverted_sign_pressure"
ORACLE_ARM_CURRENT_CREDIT_RANK_BUCKET_CURRENT_ORDER = (
    "current_credit_rank_bucket_current_order"
)
ORACLE_ARM_DETERMINISTIC_HASH_SAME_VOTES = "deterministic_hash_same_votes"
ORACLE_ARM_DIAGNOSTIC_LOCAL_LOSS_DELTA = (
    "diagnostic_local_loss_delta_oracle_same_candidates"
)
SCIENCE_ARM_IDS = (
    ARM_A0_RANK_BUCKET_CURRENT,
    ARM_A1_RANK_BUCKET_ORDER_MATCHED,
    ARM_B_RANK_FREE_SIGN_PRESSURE,
    ARM_INVERTED_SIGN_PRESSURE,
)
ORACLE_SCREEN_ARM_IDS = (
    ORACLE_ARM_CURRENT_CREDIT_RANK_BUCKET_CURRENT_ORDER,
    ORACLE_ARM_DETERMINISTIC_HASH_SAME_VOTES,
    ORACLE_ARM_DIAGNOSTIC_LOCAL_LOSS_DELTA,
)

TIE_POLICY_CURRENT_MARGIN_INDEX = "current_abs_new_acc_then_index"
TIE_POLICY_DETERMINISTIC_HASH_MATCHED = "deterministic_hash_shuffle_order_matched"
TIE_POLICY_SCREEN_ONLY_IF_A1_MISSING = "screen_only_if_a1_missing"
FIXED_RANK_BUCKET_NON_TARGET_AUX = "fixed_rank_bucket_non_target_aux"

CONTROL_PARITY_FRACTION_MIN = 0.15
CONTROL_PARITY_FRACTION_MAX = 0.45

BRANCH_RANK_FREE_POSITIVE = "rank_free_sign_pressure_positive"
BRANCH_RANKING_STILL_REQUIRED = "ranking_still_required_or_prior_null_artifact"
BRANCH_DIRECTION_PROJECTION_WRONG = "direction_projection_wrong"
BRANCH_TIE_POLICY_OR_OVERUPDATE = "tie_policy_or_overupdate_limited"
BRANCH_CREDIT_SOURCE_NOT_SUFFICIENT = "credit_source_not_sufficient"
BRANCH_INSUFFICIENT_SEPARATION = "insufficient_separation"
BRANCH_PRIOR_NULL_SETUP_UNVERIFIED = "prior_null_setup_unverified"
BRANCH_CAP_NOOP = "cap_noop"
BRANCH_MEASUREMENT_UNDERPOWERED = "measurement_underpowered"
BRANCH_MEASUREMENT_POWERED = "measurement_powered"
BRANCH_MEASUREMENT_LOSS_POWERED = "measurement_loss_powered"
BRANCH_POWERED_NEGATIVE_OR_LOSS_ONLY = "powered_negative_or_loss_only"
BRANCH_CURRENT_QACC_MARGIN_ORDER_BUNDLE_CARRIER = (
    "current_order_qacc_margin_bundle_carrier_candidate"
)
BRANCH_RANK_MAGNITUDE_CONDITIONED_ON_CURRENT_ORDER = (
    "rank_magnitude_conditioned_on_current_order"
)
BRANCH_CURRENT_ORDER_NOT_NECESSARY = "current_order_not_necessary"
BRANCH_PARTIAL_LOCAL_SIGNAL = "partial_local_signal_not_carrier"
BRANCH_NO_MATCH_DIFFERENT_CREDIT_SOURCE = "no_match_pivot_different_credit_source"
BRANCH_MASS_CONFOUNDED_CURRENT_ORDER_SIGNAL = "mass_confounded_current_order_signal"
BRANCH_A0_COMPONENT_ORDER_ROBUST = "A0_component_order_robust"
BRANCH_MEASUREMENT_ORDER_SENSITIVE = "measurement_order_sensitive__redesign_scheduler_or_update_law"
BRANCH_CANDIDATE_SET_VIABLE_CREDIT_RANKING_BAD = (
    "candidate_set_viable_credit_ranking_bad"
)
BRANCH_SCHEDULER_ONLY_ORDER_SENSITIVE = "scheduler_only_order_sensitive"
BRANCH_CREDIT_MAGNITUDE_BAD_SIGN_USABLE = "credit_magnitude_bad_sign_usable"
BRANCH_CANDIDATE_GENERATION_BAD_OR_NO_LOCAL_SIGNAL = (
    "candidate_generation_bad_or_no_local_signal"
)
BRANCH_ORACLE_INFEASIBLE_OR_TOO_EXPENSIVE = "oracle_infeasible_or_too_expensive"
BRANCH_PREREGISTERED_CHEAP_LEARNER_FEATURE_FAMILY_CANNOT_PREDICT_REGRET = (
    "preregistered_cheap_learner_feature_family_cannot_predict_regret"
)
BRANCH_MEASUREMENT_AMBIGUOUS_TIE_BAND_ALIASING = (
    "measurement_ambiguous_tie_band_aliasing"
)
BRANCH_MEASUREMENT_AMBIGUOUS_NO_BRANCH = "measurement_ambiguous_no_branch"
BRANCH_WITHIN_TIE_BAND_LEARNER_FEATURES_SEPARATE_REGRET = (
    "within_tie_band_learner_features_separate_regret"
)
BRANCH_WITHIN_TIE_BAND_NEEDS_NEW_LEARNER_STATE = (
    "within_tie_band_needs_new_learner_state"
)
BRANCH_WITHIN_TIE_BAND_AMBIGUOUS_NO_BRANCH = (
    "within_tie_band_ambiguous_no_branch"
)
ORACLE_SCREEN_BRANCHES = (
    BRANCH_CANDIDATE_SET_VIABLE_CREDIT_RANKING_BAD,
    BRANCH_SCHEDULER_ONLY_ORDER_SENSITIVE,
    BRANCH_CREDIT_MAGNITUDE_BAD_SIGN_USABLE,
    BRANCH_CANDIDATE_GENERATION_BAD_OR_NO_LOCAL_SIGNAL,
    BRANCH_ORACLE_INFEASIBLE_OR_TOO_EXPENSIVE,
)
ORACLE_WIDER_SCREEN_VERDICT_RANKING_EFFECTIVELY_OK = (
    "candidate_set_viable__ranking_effectively_ok__reopen_scheduler_cap"
)
ORACLE_WIDER_SCREEN_VERDICT_RANKING_SUBOPTIMAL = (
    "candidate_set_viable__ranking_suboptimal__promote_credit_magnitude_calibration"
)
ORACLE_WIDER_SCREEN_VERDICT_CREDIT_RANKING_BAD = (
    "candidate_set_viable__credit_ranking_bad__pivot_update_law"
)
ORACLE_WIDER_SCREEN_INTERPRETATION_VERDICTS = (
    ORACLE_WIDER_SCREEN_VERDICT_RANKING_EFFECTIVELY_OK,
    ORACLE_WIDER_SCREEN_VERDICT_RANKING_SUBOPTIMAL,
    ORACLE_WIDER_SCREEN_VERDICT_CREDIT_RANKING_BAD,
)
ORACLE_WIDER_SCREEN_NEGATIVE_PASSTHROUGH_BRANCHES = (
    BRANCH_CANDIDATE_GENERATION_BAD_OR_NO_LOCAL_SIGNAL,
    BRANCH_ORACLE_INFEASIBLE_OR_TOO_EXPENSIVE,
)
PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES = 32
PIVOT_MEASUREMENT_TOP_K = 5
PIVOT_MEASUREMENT_PRIMARY_SCORE_ID = "S_vote_margin"
PIVOT_MEASUREMENT_ABLATION_SCORE_IDS = (
    "S_vote_only",
    "S_margin_only",
    "S_current",
)
PIVOT_MEASUREMENT_NULL_HASH_SEEDS = (3, 5, 7, 11, 13, 17, 19, 23)
PIVOT_MEASUREMENT_NULL_RANDOM_SEEDS = (31, 37, 41, 43, 47, 53, 59, 61)
PIVOT_MEASUREMENT_AUC_NON_PREDICTIVE_MAX = 0.55
PIVOT_MEASUREMENT_AUC_PREDICTIVE_MIN = 0.60
PIVOT_MEASUREMENT_NULL_AUC_MARGIN_MIN = 0.05
PIVOT_MEASUREMENT_NULL_PERCENTILE_MIN = 0.75
PIVOT_MEASUREMENT_POOR_RANK_FRACTION_MIN = 0.25
PIVOT_MEASUREMENT_TIE_BAND_REGRET_SPREAD_RATIO_MAX = 0.25
PIVOT_MEASUREMENT_PREDICTIVE_SEED_LABEL = "primary_score_predictive_for_local_regret"
PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_TOP1 = "top1_unit"
PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_PREFIX_CAP_1024 = "prefix_cap1024"
PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_CURRENT_SPEC = "current_spec"
CREDIT_RANKING_PIVOT_MEASUREMENT_BRANCHES = (
    BRANCH_PREREGISTERED_CHEAP_LEARNER_FEATURE_FAMILY_CANNOT_PREDICT_REGRET,
    PIVOT_MEASUREMENT_PREDICTIVE_SEED_LABEL,
    BRANCH_MEASUREMENT_AMBIGUOUS_TIE_BAND_ALIASING,
    BRANCH_MEASUREMENT_AMBIGUOUS_NO_BRANCH,
)
WITHIN_TIE_BAND_TARGET_TIE_BAND_ID = "voteabs=4|marginabs=4"
WITHIN_TIE_BAND_PRIMARY_FAMILY_ID = "F_state_transition_rankq"
WITHIN_TIE_BAND_ABLATION_FAMILY_IDS = (
    "F_transition_rankq",
    "F_state_transition",
    "F_transition_only",
    "F_rankq_only",
    "F_flatq_only",
)
WITHIN_TIE_BAND_MATCHED_HASH_SIGNAL_MIN = 0.75
WITHIN_TIE_BAND_PREDICTIVE_BUCKET_FRACTION_MAX = 0.25
WITHIN_TIE_BAND_PREDICTIVE_REGRET_SPREAD_RATIO_MAX = 0.25
WITHIN_TIE_BAND_PREDICTIVE_REGRET_CAPTURE_RATIO_MIN = 0.50
WITHIN_TIE_BAND_FAIL_CLOSED_BUCKET_FRACTION_GT = 0.50
WITHIN_TIE_BAND_FAIL_CLOSED_REGRET_SPREAD_RATIO_GT = 0.50
WITHIN_TIE_BAND_DISCRIMINATOR_BRANCHES = (
    BRANCH_WITHIN_TIE_BAND_LEARNER_FEATURES_SEPARATE_REGRET,
    BRANCH_WITHIN_TIE_BAND_NEEDS_NEW_LEARNER_STATE,
    BRANCH_WITHIN_TIE_BAND_AMBIGUOUS_NO_BRANCH,
)
BRANCH_ACTIVATION_CREDIT_CANDIDATE_SIGNAL = "activation_credit_candidate_signal"
BRANCH_ACTIVATION_CREDIT_MISSING_SIGNAL_DEEPER_THAN_FIRST_ORDER_CREDIT_STORAGE = (
    "activation_credit_missing_signal_deeper_than_first_order_credit_storage"
)
BRANCH_ACTIVATION_CREDIT_AMBIGUOUS_NO_BRANCH = (
    "activation_credit_ambiguous_no_branch"
)
ACTIVATION_CREDIT_BRANCHES = (
    BRANCH_ACTIVATION_CREDIT_CANDIDATE_SIGNAL,
    BRANCH_ACTIVATION_CREDIT_MISSING_SIGNAL_DEEPER_THAN_FIRST_ORDER_CREDIT_STORAGE,
    BRANCH_ACTIVATION_CREDIT_AMBIGUOUS_NO_BRANCH,
)
ACTIVATION_CREDIT_PRIMARY_FAMILY_ID = "F_taylor_benefit_q5"
ACTIVATION_CREDIT_SNR_Q5_ABLATION_FAMILY_ID = "F_snr_q5"
ACTIVATION_CREDIT_DIAG_FISHER_Q5_ABLATION_FAMILY_ID = "F_diagfisher_q5"
ACTIVATION_CREDIT_ABLATION_FAMILY_IDS = (
    ACTIVATION_CREDIT_SNR_Q5_ABLATION_FAMILY_ID,
    ACTIVATION_CREDIT_DIAG_FISHER_Q5_ABLATION_FAMILY_ID,
)
ACTIVATION_CREDIT_TOPOLOGY_CONTROL_FAMILY_ID = (
    "F_topology_lane_head_row_block128"
)
ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_FIELD = "taylor_benefit_q5_bin"
ACTIVATION_CREDIT_SNR_Q5_FIELD = "snr_q5_bin"
ACTIVATION_CREDIT_DIAG_FISHER_Q5_FIELD = "diagfisher_q5_bin"
ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_PREFIX = "taylor_benefit_q5"
ACTIVATION_CREDIT_SNR_Q5_PREFIX = "snr_q5"
ACTIVATION_CREDIT_DIAG_FISHER_Q5_PREFIX = "diagfisher_q5"
ACTIVATION_CREDIT_SECOND_ORDER_SNR_EPS = 1e-12
ACTIVATION_CREDIT_TARGET_TIE_BAND_ID = WITHIN_TIE_BAND_TARGET_TIE_BAND_ID
ACTIVATION_CREDIT_MATCHED_HASH_SIGNAL_MIN = WITHIN_TIE_BAND_MATCHED_HASH_SIGNAL_MIN
ACTIVATION_CREDIT_PREDICTIVE_BUCKET_FRACTION_MAX = (
    WITHIN_TIE_BAND_PREDICTIVE_BUCKET_FRACTION_MAX
)
ACTIVATION_CREDIT_PREDICTIVE_REGRET_SPREAD_RATIO_MAX = (
    WITHIN_TIE_BAND_PREDICTIVE_REGRET_SPREAD_RATIO_MAX
)
ACTIVATION_CREDIT_PREDICTIVE_REGRET_CAPTURE_RATIO_MIN = (
    WITHIN_TIE_BAND_PREDICTIVE_REGRET_CAPTURE_RATIO_MIN
)
ACTIVATION_CREDIT_FAIL_CLOSED_BUCKET_FRACTION_GT = (
    WITHIN_TIE_BAND_FAIL_CLOSED_BUCKET_FRACTION_GT
)
ACTIVATION_CREDIT_FAIL_CLOSED_REGRET_SPREAD_RATIO_GT = (
    WITHIN_TIE_BAND_FAIL_CLOSED_REGRET_SPREAD_RATIO_GT
)
ACTIVATION_CREDIT_SMOKE_MAX_SAMPLED_CANDIDATES = 8
ACTIVATION_CREDIT_SMOKE_BATCH_SIZE = 4
ACTIVATION_CREDIT_SMOKE_REQUIRED_PER_SEED_RECEIPT_FIELDS = (
    "target_band_candidate_count",
    "grad_proxy_candidate_count",
)
ACTIVATION_CREDIT_SMOKE_OCCUPANCY_MISS_LABEL = "occupancy_miss"
ACTIVATION_CREDIT_SMOKE_INCONCLUSIVE_GATHER_TIMING_ONLY = (
    "inconclusive_on_gather_timing_only"
)
ACTIVATION_CREDIT_SMOKE_GATHER_FAILURE_REPAIR_SIGNAL = (
    "smoke_failure_repair_signal"
)
ACTIVATION_CREDIT_SMOKE_RESMOKE_BUDGETS = (12, 16)
ACTIVATION_CREDIT_MAGNITUDE_BIN_COUNT = 2
ACTIVATION_CREDIT_MAGNITUDE_Q5_BIN_COUNT = 5
ACTIVATION_CREDIT_MAGNITUDE_Q5_MIN_BUCKET_SIZE = 5
ACTIVATION_CREDIT_TOPOLOGY_ROW_BLOCK_SIZE = 128
ACTIVATION_CREDIT_FRESH_CONFIRMATION_SEED = 71
ACTIVATION_CREDIT_STDOUT_PATH_ENV = "HRM_TEXT_158_STDOUT_PATH"
ACTIVATION_CREDIT_STDERR_PATH_ENV = "HRM_TEXT_158_STDERR_PATH"
OPTIMIZER_UPDATE_LAW_BRANCHES = (
    BRANCH_RANK_FREE_POSITIVE,
    BRANCH_RANKING_STILL_REQUIRED,
    BRANCH_DIRECTION_PROJECTION_WRONG,
    BRANCH_TIE_POLICY_OR_OVERUPDATE,
    BRANCH_CREDIT_SOURCE_NOT_SUFFICIENT,
    BRANCH_INSUFFICIENT_SEPARATION,
)

_READINESS_FORBIDDEN_TRUE_FIELDS = (
    "readiness_claim",
    "full_sub2_claim",
    "full_sub2_runtime_readiness_claim",
    "ready_for_main_science",
    "checkpoint_promotion_claim",
    "optimizer_credit_state_row_flip",
    "readiness_row_flip_authorized",
)
_RAW_ARRAY_KEY_FRAGMENTS = (
    "raw_per_proposal",
    "per_proposal_array",
    "raw_proposal_array",
)
_ORACLE_GLOBAL_FORBIDDEN_TRUE_KEYS = {
    "q_persisted",
    "q_persist_allowed",
    "oracle_state_survives_into_learner",
    "learner_teacher_promotion",
    "learner_teacher_promotion_allowed",
    "checkpoint_promotional",
    "checkpoint_promotion_claim",
    "readiness_fullsub2_carrier_claim_allowed",
}
_ORACLE_GLOBAL_PT_PATH_KEYS = {
    "oracle_artifact_path",
    "oracle_checkpoint_path",
    "oracle_state_path",
    "q_persist_path",
}
_AUTHOR_PACKET_RUNTIME_RESULT_FIELDS = (
    "runtime_results",
    "arm_metrics",
    "branch_classification",
    "terminal_receipt",
    "command_outputs",
)
_COMMAND_REQUIRED_FIELDS = (
    "cwd",
    "env",
    "argv",
    "stdout_path",
    "stderr_path",
    "receipt_path",
    "scratch_root",
    "expected_exit_policy",
)
_PHASE_BUDGET_KEYS = (
    "forward_backward",
    "vote_gen_update",
    "emission_accounting",
    "artifact_flush",
)
STEP3_PHASE_POWER_150 = "measurement_power_150"
STEP3_PHASE_POWER_300 = "measurement_power_300"
STEP3_PHASE_TRUST_REGION_150 = "trust_region_cap_150"
STEP3_PHASE_TRUST_REGION_300 = "trust_region_cap_300"
STEP3_PHASE_STEPS = {
    STEP3_PHASE_POWER_150: 150,
    STEP3_PHASE_POWER_300: 300,
    STEP3_PHASE_TRUST_REGION_150: 150,
    STEP3_PHASE_TRUST_REGION_300: 300,
}
STEP3_POWER_PHASES = (STEP3_PHASE_POWER_150, STEP3_PHASE_POWER_300)
STEP3_TRUST_REGION_PHASES = (STEP3_PHASE_TRUST_REGION_150, STEP3_PHASE_TRUST_REGION_300)
STEP3_MAX_STEPS_HARD = 300
STEP3_STRICT_EXACT_FLOOR_COUNT = 10
STEP3_STRICT_EXACT_FLOOR_TOTAL = 90
STEP3_BASELINE_MAX_ABS_PER_TENSOR = 4096
STEP3_CAP_MAX_ABS_PER_TENSOR = 1024
STEP3_FRACTION_PER_TENSOR = 1.0
STEP3_EFFECTIVE_CAP_TARGET_TENSOR_NUMELS = (2048 * 512,)
STEP4_PHASE_RANK_SIGNAL_150 = "rank_signal_150"
STEP4_PHASE_RANK_SIGNAL_300 = "rank_signal_300"
STEP4_PHASE_STEPS = {
    STEP4_PHASE_RANK_SIGNAL_150: 150,
    STEP4_PHASE_RANK_SIGNAL_300: 300,
}
STEP4_PHASES = (STEP4_PHASE_RANK_SIGNAL_150, STEP4_PHASE_RANK_SIGNAL_300)
STEP4_ARM_IDS = (
    ARM_A0_RANK_BUCKET_CURRENT,
    ARM_A1_RANK_BUCKET_ORDER_MATCHED,
    ARM_B_RANK_FREE_SIGN_PRESSURE,
    ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER,
    ARM_INVERTED_SIGN_PRESSURE,
)
STEP5_PHASE_STEPS = {
    STEP4_PHASE_RANK_SIGNAL_150: 150,
}
STEP5_PHASES = tuple(STEP5_PHASE_STEPS)
STEP5_ARM_IDS = (
    ARM_A0_RANK_BUCKET_CURRENT,
    ARM_B_RANK_FREE_SIGN_PRESSURE,
    ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER,
    ARM_INVERTED_SIGN_PRESSURE,
)
STEP5_SUPPORT_ORDER_SEED = 29
STEP5_CURRICULUM_SEED = 17
STEP5_STRICT_FLOOR_COUNT = 10
STEP5_STRICT_TOTAL = 90
STEP5_STRICT_MARGIN_COUNT = 5
STEP6_PHASE_STEPS = {
    STEP4_PHASE_RANK_SIGNAL_150: 150,
}
STEP6_PHASES = tuple(STEP6_PHASE_STEPS)
STEP6_ARM_IDS = (
    ARM_A0_RANK_BUCKET_CURRENT,
    ARM_A1_RANK_BUCKET_ORDER_MATCHED,
    ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER,
)
STEP6_SUPPORT_ORDER_SEEDS: tuple[int | None, ...] = (None, 29, 43)
STEP6_FIXED_PREREG_NEW_SEED = 43
STEP6_CURRICULUM_SEED = 17
STEP6_MAX_ARM_RUNS = 9
STEP6_GPU_HOUR_CEILING = 2.0
ORACLE_SCREEN_CONTRAST_SEEDS = (43, 29)
ORACLE_SCREEN_PROMOTION_ORDER_SEEDS: tuple[int | None, ...] = (None, 29, 43)
ORACLE_SCREEN_N20_ROWS = 20
ORACLE_SCREEN_PROMOTION_ROWS = 50
ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES = (8, 32, 64)
ORACLE_SCREEN_FEASIBILITY_MAX_SAMPLED_CANDIDATES = 8
ORACLE_SCREEN_FEASIBILITY_MAX_SECONDS = 30.0
ORACLE_SCREEN_MAX_SECONDS_BY_BUDGET = {
    8: 30.0,
    32: 120.0,
    64: 240.0,
}
ORACLE_SCREEN_SCIENCE_CONTRACT_COMMIT_SHA = (
    "afbe598de6d81a776bf2bd9fc12115cf1293f9d6"
)
ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE = "first-bitlinear"
ACTIVATION_CREDIT_MEASUREMENT_BATCH_SIZE = ORACLE_SCREEN_N20_ROWS
ACTIVATION_CREDIT_GRAD_PROXY_COMPUTE_MODE = "candidate_only_gather"
ACTIVATION_CREDIT_CAPTURE_DEVICE_MODE = "device_resident"
STEP4_MATCH_STRICT_GAP_MAX = 3
STEP4_MATCH_STRICT_TOTAL = 90
STEP4_MASS_RATIO_MIN = 0.75
STEP4_MASS_RATIO_MAX = 1.25
STEP4_MASS_ABS_DELTA_MIN = 4.0
STEP4_MASS_COUNT_METRICS = (
    "q_changed_count",
    "candidate_count",
    "pre_veto_selected_count",
    "applied_count",
    "vote_nonzero_count",
)
STEP4_MASS_PRESSURE_METRICS = (
    "vote_abs_median",
    "vote_abs_max",
)


def _path_join(root: str | Path, *parts: str) -> str:
    return str(Path(root).joinpath(*parts))


def oracle_screen_budget_max_seconds(max_sampled_candidates: int) -> float:
    budget = int(max_sampled_candidates)
    if budget not in ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES:
        raise ValueError(
            "oracle-screen max_sampled_candidates must be one of "
            f"{ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES}"
        )
    return float(ORACLE_SCREEN_MAX_SECONDS_BY_BUDGET[budget])


def oracle_screen_effectively_ok_rank_position_exclusive_bound(
    sampled_candidate_count: int,
) -> int:
    sampled = max(1, int(sampled_candidate_count))
    return max(5, int(math.ceil(0.10 * sampled)))


def default_science_arms(*, include_inverted: bool = True) -> list[dict[str, Any]]:
    arms = [
        {
            "arm_id": ARM_A0_RANK_BUCKET_CURRENT,
            "vote_law": "current_rank_bucket",
            "ordering_role": "control_parity_calibrator",
            "tie_policy_id": TIE_POLICY_CURRENT_MARGIN_INDEX,
            "required": True,
        },
        {
            "arm_id": ARM_A1_RANK_BUCKET_ORDER_MATCHED,
            "vote_law": "current_rank_bucket",
            "ordering_role": "ordering_matched_control",
            "tie_policy_id": TIE_POLICY_DETERMINISTIC_HASH_MATCHED,
            "required": True,
        },
        {
            "arm_id": ARM_B_RANK_FREE_SIGN_PRESSURE,
            "vote_law": "rank_free_sign_pressure",
            "ordering_role": "candidate_same_ordering_as_A1",
            "tie_policy_id": TIE_POLICY_DETERMINISTIC_HASH_MATCHED,
            "required": True,
        },
    ]
    if include_inverted:
        arms.append(
            {
                "arm_id": ARM_INVERTED_SIGN_PRESSURE,
                "vote_law": "inverted_rank_free_sign_pressure",
                "ordering_role": "direction_falsifier_same_ordering_as_A1",
                "tie_policy_id": TIE_POLICY_DETERMINISTIC_HASH_MATCHED,
                "required": False,
            },
        )
    return arms


def default_control_parity_gate() -> dict[str, Any]:
    return {
        "metric": "current_beats_competitors_fraction",
        "min_inclusive": CONTROL_PARITY_FRACTION_MIN,
        "max_inclusive": CONTROL_PARITY_FRACTION_MAX,
        "qualitative_prior_null_signature_required": True,
        "requires_current_improves_vs_baseline": True,
        "requires_random_matches_or_beats_current": True,
        "failure_branch": BRANCH_INSUFFICIENT_SEPARATION,
        "candidate_arms_read_on_failure": False,
    }


def default_verdict_rule() -> dict[str, Any]:
    return {
        "positive_branch": BRANCH_RANK_FREE_POSITIVE,
        "positive_requires": [
            "control_parity_gate_pass",
            "B_beats_A0",
            "B_beats_A1",
            "B_beats_falsifiers",
        ],
        "B_beats_A0_but_not_A1_branch": BRANCH_TIE_POLICY_OR_OVERUPDATE,
        "A1_helped_branch": BRANCH_TIE_POLICY_OR_OVERUPDATE,
        "A1_missing_policy": TIE_POLICY_SCREEN_ONLY_IF_A1_MISSING,
        "allowed_branches": list(OPTIMIZER_UPDATE_LAW_BRANCHES),
    }


def default_hash_gate_policy() -> dict[str, Any]:
    return {
        "gate_type": "read_only_source_banked_hash_before_after",
        "required_sources": [
            "banked_parent_checkpoint",
            "banked_initial_q_state",
        ],
        "explicitly_not_sources": [
            "live_q_after_update",
            "post_arm_live_q_mutation",
        ],
        "live_q_delta_fields_required_separately": True,
        "hash_before_after_must_match": True,
    }


def build_optimizer_update_law_science_packet(
    *,
    parent_path: str | Path,
    parent_sha256: str,
    mode: str = SCIENCE_MODE_PRETERMINAL_SCREEN,
    launch_gate_id: str | None = None,
    include_inverted: bool = True,
) -> dict[str, Any]:
    normalized_mode = str(mode)
    if normalized_mode not in SCIENCE_MODE_ROWS:
        raise ValueError(f"unsupported science packet mode {mode!r}")
    packet = {
        "schema_version": OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION,
        "target_name": OPTIMIZER_UPDATE_LAW_SCIENCE_TARGET_NAME,
        "artifact_role": "optimizer_update_law_science_launch_packet",
        "diagnostic_class": DIAGNOSTIC_CLASS_PRE_FULL_STACK,
        "pre_full_stack_diagnostic": True,
        "mode": normalized_mode,
        "n_rows": SCIENCE_MODE_ROWS[normalized_mode],
        "launch_gate_id": launch_gate_id,
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "gpu_launched": False,
        "checkpoint_written": False,
        "pt_mutated": False,
        "readiness_claim": False,
        "full_sub2_claim": False,
        "optimizer_credit_state_row_flip": False,
        "aux_vote_law": FIXED_RANK_BUCKET_NON_TARGET_AUX,
        "arms": default_science_arms(include_inverted=include_inverted),
        "control_parity_gate": default_control_parity_gate(),
        "verdict_rule": default_verdict_rule(),
        "hash_gate_policy": default_hash_gate_policy(),
        "compact_instrumentation_only": True,
        "raw_per_proposal_arrays_included": False,
        "non_claims": [
            "no readiness row flip",
            "no full-sub2 runtime claim",
            "no checkpoint promotion",
            "no GPU launch from Step-1 authoring",
            "no .pt mutation",
            "default rank-bucket trainer path unchanged",
        ],
    }
    validate_optimizer_update_law_science_packet(packet)
    return packet


def _build_probe_command_record(
    *,
    repo_root: str | Path,
    run_root: str | Path,
    parent_path: str | Path,
    parent_sha256: str,
    mode: str,
    arm_id: str,
    device: str,
    phase_timeout_seconds: int | float,
    total_timeout_seconds: int | float,
    max_silent_phase_seconds: int | float,
) -> dict[str, Any]:
    n_rows = int(SCIENCE_MODE_ROWS[str(mode)])
    scratch_root = _path_join(run_root, str(mode), str(arm_id))
    receipt_path = _path_join(scratch_root, "receipt.json")
    stdout_path = _path_join(scratch_root, "stdout.ndjson")
    stderr_path = _path_join(scratch_root, "stderr.log")
    argv = [
        "python3",
        "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--phase",
        f"optimizer-update-law-{mode}-{arm_id}",
        "--device",
        str(device),
        "--parent",
        str(parent_path),
        "--parent-sha256",
        str(parent_sha256),
        "--scratch-root",
        scratch_root,
        "--steps",
        str(n_rows),
        "--max-steps-hard",
        str(max(SCIENCE_MODE_ROWS.values())),
        "--audit-interval",
        str(n_rows),
        "--science-arm",
        str(arm_id),
        "--emit-progress",
        "--phase-timeout-seconds",
        str(phase_timeout_seconds),
        "--total-timeout-seconds",
        str(total_timeout_seconds),
        "--max-silent-phase-seconds",
        str(max_silent_phase_seconds),
    ]
    return {
        "mode": str(mode),
        "arm_id": str(arm_id),
        "n_rows": n_rows,
        "steps_requested": n_rows,
        "steps_source": "SCIENCE_MODE_ROWS[mode]",
        "cwd": str(repo_root),
        "env": {
            "HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE": "1",
            "HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH": "1",
        },
        "argv": argv,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "receipt_path": receipt_path,
        "scratch_root": scratch_root,
        "expected_exit_policy": "exit_0_required_else_stop_no_retry_no_verdict",
    }


def _build_step3_probe_command_record(
    *,
    repo_root: str | Path,
    run_root: str | Path,
    parent_path: str | Path,
    parent_sha256: str,
    phase: str,
    arm_id: str,
    science_arm: str,
    device: str,
    max_abs_per_tensor: int,
    phase_timeout_seconds: int | float,
    total_timeout_seconds: int | float,
    max_silent_phase_seconds: int | float,
    enabled_if: str,
) -> dict[str, Any]:
    steps_requested = int(STEP3_PHASE_STEPS[str(phase)])
    scratch_root = _path_join(run_root, str(phase), str(arm_id))
    receipt_path = _path_join(scratch_root, "receipt.json")
    stdout_path = _path_join(scratch_root, "stdout.ndjson")
    stderr_path = _path_join(scratch_root, "stderr.log")
    argv = [
        "python3",
        "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--phase",
        f"optimizer-update-law-step3-{phase}-{arm_id}",
        "--device",
        str(device),
        "--parent",
        str(parent_path),
        "--parent-sha256",
        str(parent_sha256),
        "--scratch-root",
        scratch_root,
        "--steps",
        str(steps_requested),
        "--max-steps-hard",
        str(STEP3_MAX_STEPS_HARD),
        "--audit-interval",
        str(steps_requested),
        "--science-arm",
        str(science_arm),
        "--max-abs-per-tensor",
        str(int(max_abs_per_tensor)),
        "--emit-progress",
        "--phase-timeout-seconds",
        str(phase_timeout_seconds),
        "--total-timeout-seconds",
        str(total_timeout_seconds),
        "--max-silent-phase-seconds",
        str(max_silent_phase_seconds),
    ]
    return {
        "mode": str(phase),
        "phase_role": "measurement_power" if str(phase) in STEP3_POWER_PHASES else "trust_region_cap",
        "arm_id": str(arm_id),
        "science_arm": str(science_arm),
        "n_rows": steps_requested,
        "steps_requested": steps_requested,
        "steps_source": "STEP3_PHASE_STEPS[mode]",
        "max_abs_per_tensor": int(max_abs_per_tensor),
        "fraction_per_tensor": STEP3_FRACTION_PER_TENSOR,
        "global_cap_contract": "off",
        "cwd": str(repo_root),
        "env": {
            "HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE": "1",
            "HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH": "1",
        },
        "argv": argv,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "receipt_path": receipt_path,
        "scratch_root": scratch_root,
        "enabled_if": str(enabled_if),
        "expected_exit_policy": "exit_0_required_else_stop_no_retry_no_verdict",
    }


def default_phase_budgets() -> dict[str, Any]:
    return {
        "forward_backward": {
            "first_milestone_seconds": 180,
            "probe_phase_markers": ["step_forward_backward"],
        },
        "vote_gen_update": {
            "first_milestone_seconds": 120,
            "probe_phase_markers": ["step_update"],
        },
        "emission_accounting": {
            "first_milestone_seconds": 90,
            "probe_phase_markers": ["audit", "prior_final_audit", "front_c_identity_artifact"],
        },
        "artifact_flush": {
            "first_milestone_seconds": 60,
            "probe_phase_markers": ["checkpoint_payload", "receipt_write"],
        },
    }


def default_activation_credit_phase_budgets(*, smoke: bool) -> dict[str, Any]:
    forward_backward_seconds = 60 if smoke else 180
    gather_seconds = 30 if smoke else 120
    emit_seconds = 30 if smoke else 90
    flush_seconds = 20 if smoke else 60
    return {
        "forward_backward": {
            "first_milestone_seconds": forward_backward_seconds,
            "probe_phase_markers": ["activation_credit_forward_backward"],
        },
        "vote_gen_update": {
            "first_milestone_seconds": gather_seconds,
            "probe_phase_markers": ["activation_credit_gather"],
        },
        "emission_accounting": {
            "first_milestone_seconds": emit_seconds,
            "probe_phase_markers": ["activation_credit_emit"],
        },
        "artifact_flush": {
            "first_milestone_seconds": flush_seconds,
            "probe_phase_markers": ["checkpoint_payload", "receipt_write"],
        },
        "background_candidate_generation": {
            "first_milestone_seconds": forward_backward_seconds,
            "probe_phase_markers": ["step_forward_backward", "step_update", "audit"],
        },
    }


def default_watcher_bundle() -> dict[str, Any]:
    return {
        "watcher_kind": "line_oriented_progress_monitor",
        "input_streams": ["stdout_path", "stderr_path"],
        "progress_filters": [
            "phase_telemetry",
            "bounded_steps",
            "step_forward_backward",
            "step_update",
            "receipt_write",
        ],
        "error_filters": [
            "Traceback",
            "RuntimeError",
            "CUDA out of memory",
            "nonfinite",
            "parent checkpoint hash changed",
            "phase timeout",
            "silent phase timeout",
        ],
        "heartbeat_policy": {
            "source": "phase_telemetry",
            "stall_decision": "phase_budget_not_total_timeout",
        },
        "terminal_receipt_required": True,
    }


def default_activation_credit_watcher_bundle() -> dict[str, Any]:
    bundle = deepcopy(default_watcher_bundle())
    progress_filters = list(bundle.get("progress_filters") or ())
    for marker in (
        "activation_credit_forward_backward",
        "activation_credit_gather",
        "activation_credit_emit",
    ):
        if marker not in progress_filters:
            progress_filters.append(marker)
    bundle["progress_filters"] = progress_filters
    return bundle


def default_screen_before_verdict_dependency() -> dict[str, Any]:
    return {
        "screen_mode": SCIENCE_MODE_PRETERMINAL_SCREEN,
        "verdict_mode": SCIENCE_MODE_BRANCH_VERDICT,
        "verdict_blocked_until": {
            "screen_terminal_receipt_pass": True,
            "parent_hash_unchanged": True,
            "all_four_arms_completed": True,
            "no_nonfinite": True,
            "no_cuda_oom": True,
            "no_timeout": True,
            "pt_mutated": False,
            "qualitative_control_parity_screen_not": BRANCH_INSUFFICIENT_SEPARATION,
            "prior_null_setup_verified": True,
        },
    }


def default_terminal_criteria() -> dict[str, Any]:
    return {
        "branch_classifier": list(OPTIMIZER_UPDATE_LAW_BRANCHES),
        "verdict_rule": default_verdict_rule(),
        "control_parity_gate": default_control_parity_gate(),
        "hash_gate_policy": default_hash_gate_policy(),
        "prior_null_setup_gate": {
            "verified_required_before_parity_band": True,
            "unverified_branch": BRANCH_PRIOR_NULL_SETUP_UNVERIFIED,
            "fallback_branch": BRANCH_INSUFFICIENT_SEPARATION,
            "blocks_control_parity_band": True,
        },
        "live_q_delta_fields_required_separately": True,
        "no_verdict_outside_prereg": True,
    }


def default_resource_lane_contract(*, symbolic_lane: str = "gpu:0") -> dict[str, Any]:
    return {
        "resource_lane_required": True,
        "lane_name": str(symbolic_lane),
        "resolved_at_launch": False,
        "acquire_at_launch": True,
        "release_on_terminal_receipt": True,
        "conflict_check_required": True,
        "author_packet_does_not_acquire": True,
    }


def default_step3_power_floor() -> dict[str, Any]:
    return {
        "strict_exact_floor": {
            "non_inverted_only": True,
            "count": STEP3_STRICT_EXACT_FLOOR_COUNT,
            "total": STEP3_STRICT_EXACT_FLOOR_TOTAL,
            "not_an_acquisition_claim": True,
        },
        "paired_loss_floor": {
            "bootstrap_ci": "95%",
            "comparisons": ["A1_minus_B", "B_minus_A0"],
            "ci_must_exclude_zero": True,
        },
        "classifications": {
            "no_floor": BRANCH_MEASUREMENT_UNDERPOWERED,
            "strict_exact_floor": BRANCH_MEASUREMENT_POWERED,
            "favorable_paired_loss_ci": BRANCH_MEASUREMENT_LOSS_POWERED,
            "strict_below_floor_and_only_b_minus_a0_loss_favors_a0": BRANCH_POWERED_NEGATIVE_OR_LOSS_ONLY,
        },
        "phase2_unlock_rule": (
            "Phase 2 may make acquisition-capable claims only after strict_exact floor "
            "or favorable paired-loss separation; B-A0-loss-only favoring A0 is loss-rescue "
            "screen only/no acquisition claim or STOP."
        ),
    }


def default_step3_effective_cap_audit(
    *,
    target_tensor_numels: Sequence[int] = STEP3_EFFECTIVE_CAP_TARGET_TENSOR_NUMELS,
    baseline_max_abs_per_tensor: int = STEP3_BASELINE_MAX_ABS_PER_TENSOR,
    cap_max_abs_per_tensor: int = STEP3_CAP_MAX_ABS_PER_TENSOR,
    fraction_per_tensor: float = STEP3_FRACTION_PER_TENSOR,
) -> dict[str, Any]:
    allowed_baseline = [
        min(int(baseline_max_abs_per_tensor), int(float(fraction_per_tensor) * int(numel) + 0.999999999))
        for numel in target_tensor_numels
    ]
    allowed_cap = [
        min(int(cap_max_abs_per_tensor), int(float(fraction_per_tensor) * int(numel) + 0.999999999))
        for numel in target_tensor_numels
    ]
    reduced = [cap < baseline for baseline, cap in zip(allowed_baseline, allowed_cap)]
    return {
        "schema": "hrm_text_158_optimizer_update_law_effective_cap_audit/v0",
        "basis": "author-side first-bitlinear target tensor numel from existing C2.1 default eligible_scope",
        "target_tensor_numels": [int(numel) for numel in target_tensor_numels],
        "baseline_max_abs_per_tensor": int(baseline_max_abs_per_tensor),
        "cap_max_abs_per_tensor": int(cap_max_abs_per_tensor),
        "fraction_per_tensor": float(fraction_per_tensor),
        "tensor_count": len(target_tensor_numels),
        "tensor_count_reduced": sum(1 for value in reduced if value),
        "total_allowed_flips_baseline": sum(allowed_baseline),
        "total_allowed_flips_cap": sum(allowed_cap),
        "cap_effective": any(reduced) and sum(allowed_cap) < sum(allowed_baseline),
        "if_cap_effective_false": BRANCH_CAP_NOOP,
    }


def default_step4_science_arms() -> list[dict[str, Any]]:
    arms = [
        arm
        for arm in default_science_arms(include_inverted=False)
    ]
    arms.append(
        {
            "arm_id": ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER,
            "vote_law": "rank_free_sign_pressure",
            "ordering_role": "current_qacc_margin_order_bundle_probe",
            "tie_policy_id": TIE_POLICY_CURRENT_MARGIN_INDEX,
            "claim_caveat": "current qacc-margin/order bundle; not pure current-order rank",
            "required": True,
        },
    )
    arms.append(
        {
            "arm_id": ARM_INVERTED_SIGN_PRESSURE,
            "vote_law": "inverted_rank_free_sign_pressure",
            "ordering_role": "direction_falsifier_same_ordering_as_A1",
            "tie_policy_id": TIE_POLICY_DETERMINISTIC_HASH_MATCHED,
            "required": False,
        },
    )
    return arms


def default_step4_match_to_a0_rule() -> dict[str, Any]:
    return {
        "strict_gap_max": STEP4_MATCH_STRICT_GAP_MAX,
        "strict_total": STEP4_MATCH_STRICT_TOTAL,
        "paired_loss_comparison": "arm_minus_A0",
        "paired_loss_ci": "95% bootstrap",
        "paired_loss_match_if": "ci_crosses_zero_or_entirely_below_zero",
        "carrier_named_only_on_match_to_A0": True,
        "beat_A1_or_B_is_not_a_carrier_claim": True,
    }


def default_step4_mass_confound_rule() -> dict[str, Any]:
    return {
        "classification": BRANCH_MASS_CONFOUNDED_CURRENT_ORDER_SIGNAL,
        "compares": ["C_vs_A0", "C_vs_B"],
        "count_metrics": list(STEP4_MASS_COUNT_METRICS),
        "pressure_metrics": list(STEP4_MASS_PRESSURE_METRICS),
        "ratio_min_inclusive": STEP4_MASS_RATIO_MIN,
        "ratio_max_inclusive": STEP4_MASS_RATIO_MAX,
        "absolute_delta_min_inclusive": STEP4_MASS_ABS_DELTA_MIN,
        "material_if": (
            "any metric has abs(delta)>=4 and ratio outside [0.75,1.25]; "
            "missing metric is fail-closed"
        ),
        "if_match_to_A0_but_material_mass_difference": BRANCH_MASS_CONFOUNDED_CURRENT_ORDER_SIGNAL,
        "not_carrier_ready": True,
    }


def default_step5_science_arms() -> list[dict[str, Any]]:
    by_id = {
        str(arm["arm_id"]): dict(arm)
        for arm in default_step4_science_arms()
    }
    return [by_id[arm_id] for arm_id in STEP5_ARM_IDS]


def default_step5_support_order_proof_contract() -> dict[str, Any]:
    return {
        "support_order_seed": STEP5_SUPPORT_ORDER_SEED,
        "curriculum_seed": STEP5_CURRICULUM_SEED,
        "ordered_hash_fields_required": [
            "support_order_original_ordered_traversal_hash16",
            "support_order_permuted_ordered_traversal_hash16",
        ],
        "ordered_hashes_must_differ": True,
        "order_invariant_hash_fields_required": [
            "support_order_original_invariant_multiset_hash16",
            "support_order_permuted_invariant_multiset_hash16",
        ],
        "order_invariant_hashes_must_match": True,
        "support_content_unchanged_basis": "order_invariant_multiset_hash16",
        "legacy_support_content_hash16_semantics": "ordered_batch_hashes_order_sensitive",
        "ordered_support_content_hash16_is_invariant": False,
        "false_invariant_trap": (
            "support_content_unchanged must not be derived from support_content_hash16; "
            "support_content_hash16 is ordered and must change when traversal changes"
        ),
    }


def default_step5_pass_rule() -> dict[str, Any]:
    return {
        "label": "support_order_trajectory_robustness_pass_rule",
        "strict_total": STEP5_STRICT_TOTAL,
        "C_strict_floor_count": STEP5_STRICT_FLOOR_COUNT,
        "C_margin_over_max_A0_B_count": STEP5_STRICT_MARGIN_COUNT,
        "paired_loss_required": {
            "comparisons": ["C_minus_A0", "C_minus_B"],
            "mean_must_be_less_than": 0.0,
            "ci": "95% bootstrap",
            "ci_high_must_be_less_than": 0.0,
        },
        "mass_confound_rule": default_step4_mass_confound_rule(),
        "mass_confound_pass_rule": (
            "C_vs_A0 and C_vs_B ratios must stay in [0.75,1.25] unless "
            "abs_delta<4; missing metric fails closed"
        ),
        "inverted_falsifier": "inverted_sign_pressure must not exceed C strict-exact",
        "ready_for_main_science_after_pass": False,
        "qacc_kernelized": False,
    }


def _support_order_seed_label(seed: int | None) -> str:
    return "original" if seed is None else f"seed{int(seed)}"


def default_step6_science_arms() -> list[dict[str, Any]]:
    by_id = {
        str(arm["arm_id"]): dict(arm)
        for arm in default_step4_science_arms()
    }
    return [by_id[arm_id] for arm_id in STEP6_ARM_IDS]


def default_step6_support_order_proof_contract() -> dict[str, Any]:
    return {
        "seed_matrix": [
            {
                "support_order_seed": seed,
                "seed_label": _support_order_seed_label(seed),
                "argv_omits_support_order_seed": seed is None,
                "support_order_permutation_required": seed is not None,
            }
            for seed in STEP6_SUPPORT_ORDER_SEEDS
        ],
        "curriculum_seed": STEP6_CURRICULUM_SEED,
        "fixed_preregistered_new_seed": STEP6_FIXED_PREREG_NEW_SEED,
        "post_hoc_seed_selection_allowed": False,
        "original_trajectory_contract": (
            "support_order_seed=null and argv omits --support-order-seed"
        ),
        "seeded_trajectory_contract": (
            "seed29 and seed43 include --support-order-seed with the fixed value"
        ),
        "support_content_unchanged_basis": "order_invariant_multiset_hash16",
        "legacy_support_content_hash16_semantics": "ordered_batch_hashes_order_sensitive",
        "ordered_support_content_hash16_is_invariant": False,
    }


def default_step6_stability_rule() -> dict[str, Any]:
    return {
        "label": "order_averaged_a0_component_stability_rule",
        "primary_evidence": "seed_level_dominance",
        "dominant_arm": ARM_A0_RANK_BUCKET_CURRENT,
        "must_beat_arms": [
            ARM_A1_RANK_BUCKET_ORDER_MATCHED,
            ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER,
        ],
        "min_seeds_dominating": 2,
        "total_seeds": len(STEP6_SUPPORT_ORDER_SEEDS),
        "paired_loss_support_required": True,
        "paired_loss_comparisons": ["A0_minus_A1", "A0_minus_C"],
        "paired_loss_ci": "95% bootstrap",
        "pooled_paired_row_loss_role": "secondary_supporting_only",
        "pooled_loss_cannot_override_seed_level_instability": True,
        "positive_classification": BRANCH_A0_COMPONENT_ORDER_ROBUST,
        "negative_or_unstable_classification": BRANCH_MEASUREMENT_ORDER_SENSITIVE,
        "partial_result_allowed_if_only_one_comparator_clears": True,
        "no_carrier_readiness_or_full_sub2_claim": True,
    }


def default_step6_mass_confound_rule() -> dict[str, Any]:
    rule = default_step4_mass_confound_rule()
    rule["compares"] = ["A0_vs_A1", "A0_vs_C"]
    rule["step6_role"] = "A0 component attribution mass-confound guard"
    return rule


def default_oracle_screen_arms() -> list[dict[str, Any]]:
    return [
        {
            "arm_id": ORACLE_ARM_CURRENT_CREDIT_RANK_BUCKET_CURRENT_ORDER,
            "role": "A0 baseline path",
            "candidate_set": "same_projected_move_candidate_set",
            "vote_source": "current_credit_rank_bucket",
            "ordering_role": "current_order",
            "oracle_applied": False,
            "q_persisted": False,
            "raw_per_proposal_arrays_included": False,
            "required": True,
        },
        {
            "arm_id": ORACLE_ARM_DETERMINISTIC_HASH_SAME_VOTES,
            "role": "A1-style scheduler anchor only",
            "candidate_set": "same_projected_move_candidate_set",
            "vote_source": "same_votes_as_current_credit_rank_bucket",
            "ordering_role": "deterministic_hash_same_votes",
            "oracle_applied": False,
            "q_persisted": False,
            "raw_per_proposal_arrays_included": False,
            "required": True,
        },
        {
            "arm_id": ORACLE_ARM_DIAGNOSTIC_LOCAL_LOSS_DELTA,
            "role": "bounded objective-local CE-delta oracle",
            "candidate_set": "same_projected_move_candidate_set",
            "vote_source": "diagnostic_local_loss_delta",
            "ordering_role": "oracle_rank_same_candidates",
            "oracle_applied": False,
            "q_persisted": False,
            "learner_teacher_promotion": False,
            "checkpoint_promotional": False,
            "raw_per_proposal_arrays_included": False,
            "required": True,
        },
    ]


def default_oracle_feasibility_budget() -> dict[str, Any]:
    return default_oracle_feasibility_budget_for(
        max_sampled_candidates=ORACLE_SCREEN_FEASIBILITY_MAX_SAMPLED_CANDIDATES
    )


def default_oracle_feasibility_budget_for(*, max_sampled_candidates: int) -> dict[str, Any]:
    budget = int(max_sampled_candidates)
    return {
        "probe_required_before_full_screen": True,
        "budget_present": True,
        "allowed_max_sampled_candidates": list(ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES),
        "max_sampled_candidates": budget,
        "max_seconds_by_budget": {
            str(candidate_budget): float(seconds)
            for candidate_budget, seconds in ORACLE_SCREEN_MAX_SECONDS_BY_BUDGET.items()
        },
        "max_seconds": oracle_screen_budget_max_seconds(budget),
        "reject_if_over_budget": True,
        "reject_if_unsafe": True,
        "classify_branch_on_missing_overrun_or_unsafe": BRANCH_ORACLE_INFEASIBLE_OR_TOO_EXPENSIVE,
    }


def default_oracle_non_persistence_contract() -> dict[str, Any]:
    return {
        "q_persist_allowed": False,
        "q_persisted": False,
        "oracle_state_survives_into_learner": False,
        "learner_teacher_promotion_allowed": False,
        "learner_teacher_promotion": False,
        "checkpoint_promotional": False,
        "checkpoint_written": False,
        "pt_writes_allowed": False,
        "readiness_fullsub2_carrier_claim_allowed": False,
    }


def default_oracle_compact_summary_schema() -> dict[str, Any]:
    allowed_fields = [
        "candidate_count",
        "sampled_candidate_count",
        "top_k",
        "sign_concordance",
        "credit_rank_deciles",
        "local_loss_delta_deciles",
        "paired_loss_branch_fields",
        "wider_screen_interpretation_inputs",
    ]
    return {
        "compact_summary_only": True,
        "allowed_fields": allowed_fields,
        "required_fields": allowed_fields,
        "raw_per_proposal_arrays": False,
        "raw_candidate_scores": False,
        "raw_local_loss_deltas": False,
    }


def default_oracle_screen_seed_order_contract() -> dict[str, Any]:
    return {
        "contrast_support_order_seeds": list(ORACLE_SCREEN_CONTRAST_SEEDS),
        "contrast_seed_roles": {
            "seed43": "A0-bad contrast",
            "seed29": "A0-good contrast",
        },
        "n20_screen_rows": ORACLE_SCREEN_N20_ROWS,
        "n20_screen_is_launch_gated": True,
        "promotion_condition": {
            "promote_to_n50_x_3_orderings": True,
            "promotion_rows": ORACLE_SCREEN_PROMOTION_ROWS,
            "support_order_seeds": list(ORACLE_SCREEN_PROMOTION_ORDER_SEEDS),
            "only_if_non_null": True,
            "only_if_not_artifact_confounded": True,
            "post_hoc_seed_selection_allowed": False,
        },
    }


def default_oracle_screen_classifier_contract() -> dict[str, Any]:
    return {
        "exactly_one_branch": True,
        "allowed_branches": list(ORACLE_SCREEN_BRANCHES),
        "priority_order": [
            BRANCH_ORACLE_INFEASIBLE_OR_TOO_EXPENSIVE,
            BRANCH_CANDIDATE_GENERATION_BAD_OR_NO_LOCAL_SIGNAL,
            BRANCH_SCHEDULER_ONLY_ORDER_SENSITIVE,
            BRANCH_CREDIT_MAGNITUDE_BAD_SIGN_USABLE,
            BRANCH_CANDIDATE_SET_VIABLE_CREDIT_RANKING_BAD,
        ],
    }


def default_oracle_wider_screen_interpretation_contract(
    *,
    max_sampled_candidates: int,
) -> dict[str, Any]:
    budget = int(max_sampled_candidates)
    return {
        "contract_kind": "wider_confirmation_diagnostic_interpretation",
        "runtime_branch_classification_semantics_frozen": True,
        "runtime_branch_classification_field": "branch_classification",
        "positive_interpretation_verdicts": list(
            ORACLE_WIDER_SCREEN_INTERPRETATION_VERDICTS
        ),
        "negative_low_level_passthrough": list(
            ORACLE_WIDER_SCREEN_NEGATIVE_PASSTHROUGH_BRANCHES
        ),
        "allowed_max_sampled_candidates": list(
            ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES
        ),
        "max_sampled_candidates": budget,
        "tier_max_seconds": oracle_screen_budget_max_seconds(budget),
        "contrast_support_order_seeds": list(ORACLE_SCREEN_CONTRAST_SEEDS),
        "rank_position_index_base": 0,
        "ranking_effectively_ok": {
            "every_feasible_contrast_cell_required": True,
            "current_credit_rank_recovers_improvement_required": True,
            "oracle_best_current_rank_position_lt_rule": {
                "position_source": "oracle_best_current_sampled_rank_position",
                "absolute_floor_positions": 5,
                "fraction_of_sampled_candidate_count": 0.10,
                "comparison": "<",
                "uses_sampled_candidate_count": True,
            },
            "oracle_best_current_rank_position_lt_examples": {
                str(candidate_budget): oracle_screen_effectively_ok_rank_position_exclusive_bound(
                    candidate_budget
                )
                for candidate_budget in ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES
            },
            "current_vs_oracle_top1_gap_ratio_max_inclusive": 0.25,
        },
        "credit_ranking_bad": {
            "any_feasible_contrast_cell_rank_recovers_false": True,
            "widest_executed_budget_all_contrast_seeds_required": True,
            "rank_fraction_source": "oracle_best_current_sampled_rank_position",
            "oracle_best_current_rank_fraction_gt": 0.25,
            "current_vs_oracle_top1_gap_ratio_gt": 0.50,
        },
        "ranking_suboptimal": {
            "feasible_between_positive_and_negative_bands": True,
        },
        "next_branch_by_interpretation": {
            ORACLE_WIDER_SCREEN_VERDICT_RANKING_EFFECTIVELY_OK: (
                "ranking_not_the_bottleneck__reopen_scheduler_cap_backlog_multi_step"
            ),
            ORACLE_WIDER_SCREEN_VERDICT_RANKING_SUBOPTIMAL: (
                "credit_magnitude_or_rank_bin_calibration"
            ),
            ORACLE_WIDER_SCREEN_VERDICT_CREDIT_RANKING_BAD: (
                "update_law_or_credit_ranking_pivot"
            ),
        },
    }


def default_credit_ranking_pivot_compact_summary_schema() -> dict[str, Any]:
    allowed_fields = [
        "candidate_count",
        "sampled_candidate_count",
        "sampled_candidate_table",
        "score_family_metrics",
        "stage_a_null_guard",
        "tie_band_ambiguity",
        "local_apply_magnitude_smoke",
        "telemetry",
    ]
    return {
        "compact_summary_only": True,
        "allowed_fields": allowed_fields,
        "required_fields": allowed_fields,
        "raw_per_proposal_arrays": False,
        "raw_candidate_scores": False,
        "raw_local_loss_deltas": False,
    }


def default_credit_ranking_pivot_measurement_contract() -> dict[str, Any]:
    return {
        "contract_kind": "credit_ranking_pivot_separating_measurement",
        "candidate_generation_fixed": True,
        "same_candidate_set_required": True,
        "contrast_support_order_seeds": list(ORACLE_SCREEN_CONTRAST_SEEDS),
        "required_max_sampled_candidates": PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES,
        "top_k": PIVOT_MEASUREMENT_TOP_K,
        "learner_available_ranking_input_fields": [
            "candidate_id",
            "state_key",
            "flat_index",
            "vote_value",
            "abs_vote_value",
            "current_margin_abs",
            "current_rank_position",
            "tie_band_id",
        ],
        "oracle_only_label_fields": [
            "candidate_loss",
            "local_loss_delta",
            "regret_vs_oracle_top1_local_loss_delta",
            "oracle_best_sampled_rank_position",
        ],
        "score_family": {
            "primary": PIVOT_MEASUREMENT_PRIMARY_SCORE_ID,
            "ablations": list(PIVOT_MEASUREMENT_ABLATION_SCORE_IDS),
            "decision_basis": "primary_plus_ablation_report_no_post_hoc_best_of_many",
            "hash_control_role": "null_distribution_only",
            "null_distribution": {
                "deterministic_hash_seeds": list(PIVOT_MEASUREMENT_NULL_HASH_SEEDS),
                "random_permutation_seeds": list(PIVOT_MEASUREMENT_NULL_RANDOM_SEEDS),
                "comparison_basis": "combined_null_median_and_percentile",
            },
        },
        "stage_a": {
            "decision_metrics": [
                "oracle_top_k_overlap_fraction",
                "oracle_top_k_regret_capture_ratio",
                "oracle_top_k_gap_ratio",
                "pairwise_auc",
                "null_median_auc_margin",
                "null_percentile",
            ],
            "pairwise_auc_non_predictive_max": PIVOT_MEASUREMENT_AUC_NON_PREDICTIVE_MAX,
            "pairwise_auc_predictive_min": PIVOT_MEASUREMENT_AUC_PREDICTIVE_MIN,
            "oracle_best_sampled_rank_position_poor_fraction": (
                PIVOT_MEASUREMENT_POOR_RANK_FRACTION_MIN
            ),
            "oracle_best_sampled_rank_position_poor_threshold_rule": (
                "ceil(fraction * sampled_candidate_count)"
            ),
            "null_guard": {
                "median_auc_margin_min": PIVOT_MEASUREMENT_NULL_AUC_MARGIN_MIN,
                "percentile_min": PIVOT_MEASUREMENT_NULL_PERCENTILE_MIN,
                "not_heavy_ci": True,
            },
            "non_predictive_branch_label": (
                BRANCH_PREREGISTERED_CHEAP_LEARNER_FEATURE_FAMILY_CANNOT_PREDICT_REGRET
            ),
            "predictive_seed_label": PIVOT_MEASUREMENT_PREDICTIVE_SEED_LABEL,
        },
        "tie_band_ambiguity_guard": {
            "score_family": PIVOT_MEASUREMENT_PRIMARY_SCORE_ID,
            "oracle_best_in_band_required": True,
            "ambiguous_if_regret_spread_ratio_gt": (
                PIVOT_MEASUREMENT_TIE_BAND_REGRET_SPREAD_RATIO_MAX
            ),
            "ambiguous_branch_label": BRANCH_MEASUREMENT_AMBIGUOUS_TIE_BAND_ALIASING,
        },
        "stage_b_local_apply_magnitude_smoke": {
            "contract_kind": "local_apply_magnitude_smoke_only",
            "variant_ids": [
                PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_TOP1,
                PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_PREFIX_CAP_1024,
                PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_CURRENT_SPEC,
            ],
            "current_spec_is_non_definitive_without_live_full_cap": True,
            "definitive_b_requires_follow_on": True,
        },
        "stage_c_follow_on": {
            "separately_gated_required": True,
            "broad_arm_forbidden": True,
            "no_acquisition_or_retention_claim_before_receipt": True,
        },
        "allowed_seed_local_labels": [
            BRANCH_PREREGISTERED_CHEAP_LEARNER_FEATURE_FAMILY_CANNOT_PREDICT_REGRET,
            PIVOT_MEASUREMENT_PREDICTIVE_SEED_LABEL,
            BRANCH_MEASUREMENT_AMBIGUOUS_TIE_BAND_ALIASING,
            BRANCH_MEASUREMENT_AMBIGUOUS_NO_BRANCH,
        ],
    }


def default_within_tie_band_discriminator_compact_summary_schema() -> dict[str, Any]:
    allowed_fields = [
        "candidate_count",
        "sampled_candidate_count",
        "sampled_candidate_table",
        "target_tie_band",
        "family_metrics",
        "telemetry",
    ]
    return {
        "compact_summary_only": True,
        "allowed_fields": allowed_fields,
        "required_fields": allowed_fields,
        "raw_per_proposal_arrays": False,
        "raw_candidate_scores": False,
        "raw_local_loss_deltas": False,
    }


def default_within_tie_band_discriminator_measurement_contract() -> dict[str, Any]:
    learner_available_fields = [
        "candidate_id",
        "state_key",
        "flat_index",
        "vote_value",
        "abs_vote_value",
        "current_margin_abs",
        "current_rank_position",
        "tie_band_id",
        "current_q_level",
        "pre_accumulator_i16",
        "new_acc_i32_signed",
        "proposal_direction",
        "threshold_residual_signed",
        "proximity_to_threshold",
        "tensor_numel",
        "state_candidate_count",
        "current_rank_fraction_within_state",
        "current_rank_quartile_within_state",
        "flat_index_fraction",
        "flat_index_quartile",
        "transition_class",
    ]
    return {
        "contract_kind": "within_tie_band_learner_available_discriminator_measurement",
        "candidate_generation_fixed": True,
        "same_candidate_set_required": True,
        "contrast_support_order_seeds": list(ORACLE_SCREEN_CONTRAST_SEEDS),
        "required_max_sampled_candidates": PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES,
        "target_tie_band_id": WITHIN_TIE_BAND_TARGET_TIE_BAND_ID,
        "oracle_best_in_target_tie_band_required": True,
        "top_k": PIVOT_MEASUREMENT_TOP_K,
        "rank_position_index_base": 0,
        "learner_available_ranking_input_fields": learner_available_fields,
        "learner_available_field_provenance": {
            "candidate_id": "_build_oracle_candidate_universe:_candidate_id(state_key, flat_index)",
            "state_key": "_build_oracle_candidate_universe: loop key before oracle evaluation",
            "flat_index": "_build_oracle_candidate_universe: unordered candidate index from _ordered_candidate_indices",
            "vote_value": "_build_oracle_candidate_universe: vote_flat[flat_index] before _evaluate_loss",
            "abs_vote_value": "_build_oracle_candidate_universe: abs(vote_value) before _evaluate_loss",
            "current_margin_abs": "_build_oracle_candidate_universe: abs(new_acc[flat_index]) before _evaluate_loss",
            "current_rank_position": "_build_oracle_candidate_universe: enumerate(current_ordered) within state",
            "tie_band_id": "_build_oracle_candidate_universe: _pivot_tie_band_id(abs_vote_value, current_margin_abs)",
            "current_q_level": "_build_oracle_candidate_universe: state.vote_update_state().q_levels[flat_index]",
            "pre_accumulator_i16": "_build_oracle_candidate_universe: state.vote_update_state().accumulators[flat_index]",
            "new_acc_i32_signed": "_build_oracle_candidate_universe: new_acc[flat_index] from _ordered_candidate_indices",
            "proposal_direction": "_build_oracle_candidate_universe: sign(new_acc_i32_signed)",
            "threshold_residual_signed": "_build_oracle_candidate_universe: new_acc_i32_signed - proposal_direction * threshold_abs",
            "proximity_to_threshold": "_build_oracle_candidate_universe: abs(abs(new_acc_i32_signed) - threshold_abs)",
            "tensor_numel": "_build_oracle_candidate_universe: state.vote_update_state().q_levels.numel()",
            "state_candidate_count": "_build_oracle_candidate_universe: len(unordered) within state",
            "current_rank_fraction_within_state": "_build_oracle_candidate_universe: (current_rank_position + 1) / state_candidate_count",
            "current_rank_quartile_within_state": "_build_oracle_candidate_universe: quartile(current_rank_position, state_candidate_count)",
            "flat_index_fraction": "_build_oracle_candidate_universe: (flat_index + 1) / tensor_numel",
            "flat_index_quartile": "_build_oracle_candidate_universe: quartile(flat_index, tensor_numel)",
            "transition_class": "_build_oracle_candidate_universe: tuple(current_q_level, proposal_direction)",
        },
        "oracle_only_label_fields": [
            "candidate_loss",
            "local_loss_delta",
            "regret_vs_target_tie_band_oracle_top1_local_loss_delta",
            "target_tie_band_oracle_best_candidate_id",
        ],
        "family_discriminator": {
            "primary": WITHIN_TIE_BAND_PRIMARY_FAMILY_ID,
            "ablations": list(WITHIN_TIE_BAND_ABLATION_FAMILY_IDS),
            "fields_by_family_id": {
                WITHIN_TIE_BAND_PRIMARY_FAMILY_ID: [
                    "state_key",
                    "transition_class",
                    "current_rank_quartile_within_state",
                ],
                "F_transition_rankq": [
                    "transition_class",
                    "current_rank_quartile_within_state",
                ],
                "F_state_transition": [
                    "state_key",
                    "transition_class",
                ],
                "F_transition_only": [
                    "transition_class",
                ],
                "F_rankq_only": [
                    "current_rank_quartile_within_state",
                ],
                "F_flatq_only": [
                    "flat_index_quartile",
                ],
            },
            "decision_basis": "primary_plus_ablation_report_no_post_hoc_best_of_many",
            "hash_control_role": "null_distribution_only",
            "null_distribution": {
                "deterministic_hash_seeds": list(PIVOT_MEASUREMENT_NULL_HASH_SEEDS),
                "matched_cardinality_bucket_partitions": True,
                "smaller_bucket_fraction_guard_field": (
                    "matched_hash_null_fraction_gte_observed_bucket_fraction"
                ),
                "smaller_bucket_fraction_guard_comparison": "fraction_gte_observed",
                "larger_regret_capture_guard_field": (
                    "matched_hash_null_fraction_lte_observed_regret_capture_ratio"
                ),
                "larger_regret_capture_guard_comparison": "fraction_lte_observed",
            },
        },
        "within_band_decision": {
            "decision_metrics": [
                "oracle_best_bucket_fraction",
                "oracle_best_bucket_regret_spread_ratio",
                "oracle_best_bucket_regret_capture_ratio",
                "oracle_best_bucket_top_k_capture_fraction",
                "matched_hash_null_fraction_gte_observed_bucket_fraction",
                "matched_hash_null_fraction_lte_observed_regret_capture_ratio",
                "within_band_pairwise_auc_report_only",
            ],
            "target_tie_band_missing_branch_label": (
                BRANCH_WITHIN_TIE_BAND_AMBIGUOUS_NO_BRANCH
            ),
            "predictive_branch_label": (
                BRANCH_WITHIN_TIE_BAND_LEARNER_FEATURES_SEPARATE_REGRET
            ),
            "predictive_family_id": WITHIN_TIE_BAND_PRIMARY_FAMILY_ID,
            "predictive_bucket_fraction_max": (
                WITHIN_TIE_BAND_PREDICTIVE_BUCKET_FRACTION_MAX
            ),
            "predictive_regret_spread_ratio_max": (
                WITHIN_TIE_BAND_PREDICTIVE_REGRET_SPREAD_RATIO_MAX
            ),
            "predictive_regret_capture_ratio_min": (
                WITHIN_TIE_BAND_PREDICTIVE_REGRET_CAPTURE_RATIO_MIN
            ),
            "predictive_matched_hash_guard_min": (
                WITHIN_TIE_BAND_MATCHED_HASH_SIGNAL_MIN
            ),
            "fail_closed_branch_label": (
                BRANCH_WITHIN_TIE_BAND_NEEDS_NEW_LEARNER_STATE
            ),
            "fail_closed_requires_all_preregistered_families": True,
            "fail_closed_bucket_fraction_gt": (
                WITHIN_TIE_BAND_FAIL_CLOSED_BUCKET_FRACTION_GT
            ),
            "fail_closed_regret_spread_ratio_gt": (
                WITHIN_TIE_BAND_FAIL_CLOSED_REGRET_SPREAD_RATIO_GT
            ),
            "fail_closed_matched_hash_signal_min": (
                WITHIN_TIE_BAND_MATCHED_HASH_SIGNAL_MIN
            ),
            "ambiguous_branch_label": BRANCH_WITHIN_TIE_BAND_AMBIGUOUS_NO_BRANCH,
        },
        "fragmentation_audit": {
            "bucket_cardinality_histogram_required": True,
            "singleton_bucket_count_required": True,
        },
        "allowed_seed_local_labels": list(WITHIN_TIE_BAND_DISCRIMINATOR_BRANCHES),
    }


def default_activation_credit_compact_summary_schema() -> dict[str, Any]:
    allowed_fields = [
        "candidate_count",
        "sampled_candidate_count",
        "sampled_candidate_table",
        "target_tie_band",
        "family_metrics",
        "telemetry",
    ]
    return {
        "compact_summary_only": True,
        "allowed_fields": allowed_fields,
        "required_fields": allowed_fields,
        "raw_per_proposal_arrays": False,
        "raw_candidate_scores": False,
        "raw_local_loss_deltas": False,
    }


def default_activation_credit_scale_smoke_compact_summary_schema() -> dict[str, Any]:
    allowed_fields = [
        "target_tie_band_id",
        "target_band_candidate_count",
        "grad_proxy_candidate_count",
        "magnitude_bin_threshold",
        "magnitude_bin_histogram",
        "magnitude_bin_degenerate",
        "singleton_magnitude_source_count",
        "sampled_target_band_rows",
    ]
    return {
        "compact_summary_only": True,
        "allowed_fields": allowed_fields,
        "required_fields": allowed_fields,
        "raw_per_proposal_arrays": False,
        "raw_candidate_scores": False,
        "raw_local_loss_deltas": False,
    }


def default_activation_credit_measurement_contract() -> dict[str, Any]:
    learner_available_fields = [
        "candidate_id",
        "state_key",
        "flat_index",
        "vote_value",
        "current_margin_abs",
        "current_rank_position",
        "tie_band_id",
        "transition_class",
        "candidate_delta_sign",
        "candidate_delta_weight",
        ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_FIELD,
        ACTIVATION_CREDIT_SNR_Q5_FIELD,
        ACTIVATION_CREDIT_DIAG_FISHER_Q5_FIELD,
        "topology_row_block_128",
        "activation_feature_valid",
    ]
    return {
        "contract_kind": "activation_credit_within_tie_band_measurement",
        "candidate_generation_fixed": True,
        "same_candidate_set_required": True,
        "contrast_support_order_seeds": list(ORACLE_SCREEN_CONTRAST_SEEDS),
        "required_max_sampled_candidates": PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES,
        "required_eligible_scope": ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE,
        "target_tie_band_id": ACTIVATION_CREDIT_TARGET_TIE_BAND_ID,
        "oracle_best_in_target_tie_band_required": True,
        "top_k": PIVOT_MEASUREMENT_TOP_K,
        "rank_position_index_base": 0,
        "activation_credit_source": {
            "grad_proxy_formula": "sum(dL/dy_row * x_col) over autograd-captured invocations/positions; no extra response-label mask",
            "diag_fisher_formula": "sum((dL/dy_row)^2 * (x_col)^2) over the exact grad_proxy captures; empirical Fisher / Gauss-Newton diagonal surrogate only",
            "capture_device_mode": ACTIVATION_CREDIT_CAPTURE_DEVICE_MODE,
            "grad_proxy_compute_mode": ACTIVATION_CREDIT_GRAD_PROXY_COMPUTE_MODE,
            "candidate_only_gather_required": True,
            "diag_fisher_reuses_grad_proxy_captures": True,
            "second_backward_forbidden": True,
            "no_extra_response_label_mask": True,
            "fixed_background_candidate_generation_allowed": True,
            "fixed_background_must_be_labeled_non_hot_loop": True,
            "fp_proxy_transient_eval_only": True,
            "transient_eval_only_scalar_fields": [
                "grad_proxy",
                "diag_fisher",
                "taylor_benefit",
                "snr",
            ],
            "policy_facing_fields": [
                ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_FIELD,
                ACTIVATION_CREDIT_SNR_Q5_FIELD,
                ACTIVATION_CREDIT_DIAG_FISHER_Q5_FIELD,
            ],
        },
        "feature_construction": {
            "candidate_delta_sign_source": "sign(q_after_one_flip[flat_index] - current_q_level)",
            "candidate_delta_weight_source": "((q_after_one_flip[flat_index] - current_q_level) * frozen_scale_scalar)",
            "candidate_delta_weight_effective_weight_space": True,
            "candidate_delta_sign_zero_invalidates_row": True,
            "diag_fisher_surrogate_kind": "empirical_fisher_gauss_newton_diagonal",
            "taylor_benefit_formula": "-grad_proxy * candidate_delta_weight - 0.5 * diag_fisher * candidate_delta_weight^2",
            "snr_formula": "abs(grad_proxy) / sqrt(diag_fisher + second_order_snr_eps)",
            "second_order_snr_eps": ACTIVATION_CREDIT_SECOND_ORDER_SNR_EPS,
            "q5_bin_count": ACTIVATION_CREDIT_MAGNITUDE_Q5_BIN_COUNT,
            "q5_strategy": "equal_frequency_quintiles_over_target_band_scalar_values",
            "q5_value_field_by_family_id": {
                ACTIVATION_CREDIT_PRIMARY_FAMILY_ID: "taylor_benefit",
                ACTIVATION_CREDIT_SNR_Q5_ABLATION_FAMILY_ID: "snr",
                ACTIVATION_CREDIT_DIAG_FISHER_Q5_ABLATION_FAMILY_ID: "diag_fisher",
            },
            "q5_output_field_by_family_id": {
                ACTIVATION_CREDIT_PRIMARY_FAMILY_ID: ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_FIELD,
                ACTIVATION_CREDIT_SNR_Q5_ABLATION_FAMILY_ID: ACTIVATION_CREDIT_SNR_Q5_FIELD,
                ACTIVATION_CREDIT_DIAG_FISHER_Q5_ABLATION_FAMILY_ID: ACTIVATION_CREDIT_DIAG_FISHER_Q5_FIELD,
            },
            "q5_min_bucket_size": ACTIVATION_CREDIT_MAGNITUDE_Q5_MIN_BUCKET_SIZE,
            "q5_ties_force_ambiguous": True,
            "q5_singleton_buckets_forbidden": True,
            "degenerate_branch_label": BRANCH_ACTIVATION_CREDIT_AMBIGUOUS_NO_BRANCH,
        },
        "learner_available_ranking_input_fields": learner_available_fields,
        "learner_available_field_provenance": {
            "candidate_id": "_build_oracle_candidate_universe:_candidate_id(state_key, flat_index)",
            "state_key": "_build_oracle_candidate_universe: loop key before oracle evaluation",
            "flat_index": "_build_oracle_candidate_universe: unordered candidate index from _ordered_candidate_indices",
            "vote_value": "_build_oracle_candidate_universe: vote_flat[flat_index] before _evaluate_loss",
            "current_margin_abs": "_build_oracle_candidate_universe: abs(new_acc[flat_index]) before _evaluate_loss",
            "current_rank_position": "_build_oracle_candidate_universe: enumerate(current_ordered) within state",
            "tie_band_id": "_build_oracle_candidate_universe: _pivot_tie_band_id(abs(vote_value), current_margin_abs)",
            "transition_class": "_build_oracle_candidate_universe: tuple(current_q_level, proposal_direction)",
            "candidate_delta_sign": "_evaluate_sampled_candidates_for_oracle_screen: sign(q_after_one_flip[flat_index] - current_q_level)",
            "candidate_delta_weight": "run_activation_credit_measurement_oracle_screen: (q_after_one_flip[flat_index] - current_q_level) * frozen_scale_scalar in effective-weight space",
            ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_FIELD: "run_activation_credit_measurement_oracle_screen: equal-frequency quintile bin over target-band taylor_benefit when the primary q5 guard clears",
            ACTIVATION_CREDIT_SNR_Q5_FIELD: "run_activation_credit_measurement_oracle_screen: equal-frequency quintile bin over target-band snr when the report-only q5 guard clears",
            ACTIVATION_CREDIT_DIAG_FISHER_Q5_FIELD: "run_activation_credit_measurement_oracle_screen: equal-frequency quintile bin over target-band diag_fisher (empirical Fisher / Gauss-Newton diagonal surrogate) when the report-only q5 guard clears",
            "topology_row_block_128": "run_activation_credit_measurement_oracle_screen: row_index // 128 from flat_index decomposition",
            "activation_feature_valid": "run_activation_credit_measurement_oracle_screen: grad_proxy and diag_fisher available and candidate_delta_sign != 0",
        },
        "oracle_only_label_fields": [
            "candidate_loss",
            "local_loss_delta",
            "regret_vs_target_tie_band_oracle_top1_local_loss_delta",
            "target_tie_band_oracle_best_candidate_id",
        ],
        "family_discriminator": {
            "primary": ACTIVATION_CREDIT_PRIMARY_FAMILY_ID,
            "ablations": list(ACTIVATION_CREDIT_ABLATION_FAMILY_IDS),
            "topology_control": ACTIVATION_CREDIT_TOPOLOGY_CONTROL_FAMILY_ID,
            "fields_by_family_id": {
                ACTIVATION_CREDIT_PRIMARY_FAMILY_ID: [
                    ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_FIELD,
                ],
                ACTIVATION_CREDIT_SNR_Q5_ABLATION_FAMILY_ID: [
                    ACTIVATION_CREDIT_SNR_Q5_FIELD,
                ],
                ACTIVATION_CREDIT_DIAG_FISHER_Q5_ABLATION_FAMILY_ID: [
                    ACTIVATION_CREDIT_DIAG_FISHER_Q5_FIELD,
                ],
                ACTIVATION_CREDIT_TOPOLOGY_CONTROL_FAMILY_ID: [
                    "topology_row_block_128",
                ],
            },
            "decision_basis": "primary_plus_ablation_report_no_post_hoc_best_of_many",
            "hash_control_role": "null_distribution_only",
            "null_distribution": {
                "deterministic_hash_seeds": list(PIVOT_MEASUREMENT_NULL_HASH_SEEDS),
                "matched_cardinality_bucket_partitions": True,
                "smaller_bucket_fraction_guard_field": (
                    "matched_hash_null_fraction_gte_observed_bucket_fraction"
                ),
                "smaller_bucket_fraction_guard_comparison": "fraction_gte_observed",
                "larger_regret_capture_guard_field": (
                    "matched_hash_null_fraction_lte_observed_regret_capture_ratio"
                ),
                "larger_regret_capture_guard_comparison": "fraction_lte_observed",
            },
        },
        "within_band_decision": {
            "decision_metrics": [
                "oracle_best_bucket_fraction",
                "oracle_best_bucket_regret_spread_ratio",
                "oracle_best_bucket_regret_capture_ratio",
                "oracle_best_bucket_top_k_capture_fraction",
                "matched_hash_null_fraction_gte_observed_bucket_fraction",
                "matched_hash_null_fraction_lte_observed_regret_capture_ratio",
                "within_band_pairwise_auc_report_only",
            ],
            "target_tie_band_missing_branch_label": (
                BRANCH_ACTIVATION_CREDIT_AMBIGUOUS_NO_BRANCH
            ),
            "predictive_branch_label": BRANCH_ACTIVATION_CREDIT_CANDIDATE_SIGNAL,
            "predictive_family_id": ACTIVATION_CREDIT_PRIMARY_FAMILY_ID,
            "predictive_bucket_fraction_max": (
                ACTIVATION_CREDIT_PREDICTIVE_BUCKET_FRACTION_MAX
            ),
            "predictive_regret_spread_ratio_max": (
                ACTIVATION_CREDIT_PREDICTIVE_REGRET_SPREAD_RATIO_MAX
            ),
            "predictive_regret_capture_ratio_min": (
                ACTIVATION_CREDIT_PREDICTIVE_REGRET_CAPTURE_RATIO_MIN
            ),
            "predictive_matched_hash_guard_min": (
                ACTIVATION_CREDIT_MATCHED_HASH_SIGNAL_MIN
            ),
            "fail_closed_branch_label": (
                BRANCH_ACTIVATION_CREDIT_MISSING_SIGNAL_DEEPER_THAN_FIRST_ORDER_CREDIT_STORAGE
            ),
            "fail_closed_requires_all_preregistered_families": True,
            "fail_closed_bucket_fraction_gt": (
                ACTIVATION_CREDIT_FAIL_CLOSED_BUCKET_FRACTION_GT
            ),
            "fail_closed_regret_spread_ratio_gt": (
                ACTIVATION_CREDIT_FAIL_CLOSED_REGRET_SPREAD_RATIO_GT
            ),
            "fail_closed_matched_hash_signal_min": (
                ACTIVATION_CREDIT_MATCHED_HASH_SIGNAL_MIN
            ),
            "topology_control_family_id": ACTIVATION_CREDIT_TOPOLOGY_CONTROL_FAMILY_ID,
            "topology_control_positive_forces_ambiguous": True,
            "ambiguous_branch_label": BRANCH_ACTIVATION_CREDIT_AMBIGUOUS_NO_BRANCH,
        },
        "fragmentation_audit": {
            "bucket_cardinality_histogram_required": True,
            "singleton_bucket_count_required": True,
            "candidate_delta_weight_support_required": True,
            "q5_primary_prefix": ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_PREFIX,
            "q5_report_only_prefixes": [
                ACTIVATION_CREDIT_SNR_Q5_PREFIX,
                ACTIVATION_CREDIT_DIAG_FISHER_Q5_PREFIX,
            ],
            "q5_min_bucket_candidate_count_required": ACTIVATION_CREDIT_MAGNITUDE_Q5_MIN_BUCKET_SIZE,
            "q5_singleton_buckets_forbidden": True,
            "q5_ties_force_ambiguous": True,
            "q5_guard_failure_branch_label": BRANCH_ACTIVATION_CREDIT_AMBIGUOUS_NO_BRANCH,
        },
        "scale_smoke_gate": {
            "required_before_full_eval": True,
            "smoke_launch_bundle_packet_kind": (
                ACTIVATION_CREDIT_SCALE_SMOKE_LAUNCH_BUNDLE_PACKET_KIND
            ),
            "required_max_sampled_candidates": (
                ACTIVATION_CREDIT_SMOKE_MAX_SAMPLED_CANDIDATES
            ),
            "required_batch_size": ACTIVATION_CREDIT_SMOKE_BATCH_SIZE,
            "smoke_branch_classification_must_be_null": True,
            "required_grad_proxy_candidate_count_positive": True,
        },
        "fresh_confirmation_gate": {
            "required_seed_before_persistent_followup": (
                ACTIVATION_CREDIT_FRESH_CONFIRMATION_SEED
            ),
            "two_seed_positive_label": BRANCH_ACTIVATION_CREDIT_CANDIDATE_SIGNAL,
            "persistent_followup_forbidden_before_fresh_confirmation": True,
        },
        "allowed_seed_local_labels": list(ACTIVATION_CREDIT_BRANCHES),
    }


def default_activation_credit_scale_smoke_contract() -> dict[str, Any]:
    return {
        "contract_kind": "activation_credit_scale_smoke_only",
        "same_candidate_set_required": True,
        "contrast_support_order_seeds": list(ORACLE_SCREEN_CONTRAST_SEEDS),
        "required_max_sampled_candidates": ACTIVATION_CREDIT_SMOKE_MAX_SAMPLED_CANDIDATES,
        "required_batch_size": ACTIVATION_CREDIT_SMOKE_BATCH_SIZE,
        "required_eligible_scope": ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE,
        "target_tie_band_id": ACTIVATION_CREDIT_TARGET_TIE_BAND_ID,
        "capture_device_mode": ACTIVATION_CREDIT_CAPTURE_DEVICE_MODE,
        "grad_proxy_compute_mode": ACTIVATION_CREDIT_GRAD_PROXY_COMPUTE_MODE,
        "no_extra_response_label_mask": True,
        "fixed_background_must_be_labeled_non_hot_loop": True,
        "smoke_branch_classification_must_be_null": True,
        "required_grad_proxy_candidate_count_positive": True,
        "policy_verdict_forbidden": True,
        "occupancy_outcome_contract": default_activation_credit_scale_smoke_outcome_contract(),
        "allowed_policy_facing_fields": [
            "credit_sign",
            "credit_magnitude_bin",
            "signed_alignment",
        ],
    }


def default_activation_credit_scale_smoke_outcome_contract() -> dict[str, Any]:
    return {
        "per_seed_receipt_fields_required": list(
            ACTIVATION_CREDIT_SMOKE_REQUIRED_PER_SEED_RECEIPT_FIELDS
        ),
        "pass_requires_any_seed_positive_fields": list(
            ACTIVATION_CREDIT_SMOKE_REQUIRED_PER_SEED_RECEIPT_FIELDS
        ),
        "per_seed_target_band_zero_label": ACTIVATION_CREDIT_SMOKE_OCCUPANCY_MISS_LABEL,
        "per_seed_target_band_zero_is_code_failure": False,
        "all_seeds_target_band_zero_outcome": (
            ACTIVATION_CREDIT_SMOKE_INCONCLUSIVE_GATHER_TIMING_ONLY
        ),
        "all_seeds_target_band_zero_reprobe_budgets": list(
            ACTIVATION_CREDIT_SMOKE_RESMOKE_BUDGETS
        ),
        "all_seeds_target_band_zero_blocks_full_read": True,
        "target_band_positive_grad_proxy_zero_outcome": (
            ACTIVATION_CREDIT_SMOKE_GATHER_FAILURE_REPAIR_SIGNAL
        ),
        "target_band_positive_grad_proxy_zero_is_occupancy_miss": False,
        "target_band_positive_grad_proxy_zero_is_code_failure": True,
    }


def default_activation_credit_scale_smoke_terminal_criteria() -> dict[str, Any]:
    return {
        "branch_classifier": None,
        "same_candidate_set_required": True,
        "support_order_seeds": list(ORACLE_SCREEN_CONTRAST_SEEDS),
        "required_max_sampled_candidates": ACTIVATION_CREDIT_SMOKE_MAX_SAMPLED_CANDIDATES,
        "required_batch_size": ACTIVATION_CREDIT_SMOKE_BATCH_SIZE,
        "required_eligible_scope": ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE,
        "required_grad_proxy_candidate_count_positive": True,
        "smoke_branch_classification_must_be_null": True,
        "policy_verdict_forbidden": True,
        "qacc_kernelized": False,
        "device_residency_not_hot_loop_residency": True,
        "occupancy_outcome_contract": default_activation_credit_scale_smoke_outcome_contract(),
    }


def default_activation_credit_measurement_terminal_criteria() -> dict[str, Any]:
    return {
        "branch_classifier": list(ACTIVATION_CREDIT_BRANCHES),
        "same_candidate_set_required": True,
        "support_order_seeds": list(ORACLE_SCREEN_CONTRAST_SEEDS),
        "n20_screen_rows": ACTIVATION_CREDIT_MEASUREMENT_BATCH_SIZE,
        "required_max_sampled_candidates": PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES,
        "required_eligible_scope": ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE,
        "fresh_confirmation_seed_required_for_persistent_followup": (
            ACTIVATION_CREDIT_FRESH_CONFIRMATION_SEED
        ),
        "topology_control_positive_forces_ambiguous": True,
        "qacc_kernelized": False,
        "device_residency_not_hot_loop_residency": True,
    }


def step4_arm_matches_a0(
    *,
    arm_strict_exact_count: int,
    a0_strict_exact_count: int,
    paired_loss_ci_low: float,
    paired_loss_ci_high: float,
    strict_gap_max: int = STEP4_MATCH_STRICT_GAP_MAX,
) -> bool:
    strict_gap = int(a0_strict_exact_count) - int(arm_strict_exact_count)
    loss_ci_matches = float(paired_loss_ci_low) <= 0.0 <= float(paired_loss_ci_high)
    loss_ci_favors_arm = float(paired_loss_ci_high) < 0.0
    return strict_gap <= int(strict_gap_max) and (loss_ci_matches or loss_ci_favors_arm)


def step4_mass_confound_detected(
    *,
    reference: Mapping[str, int | float],
    candidate: Mapping[str, int | float],
    rule: Mapping[str, Any] | None = None,
) -> bool:
    active_rule = dict(rule or default_step4_mass_confound_rule())
    metrics = list(active_rule.get("count_metrics") or ()) + list(active_rule.get("pressure_metrics") or ())
    if not metrics:
        raise ValueError("Step-4 mass-confound rule must name metrics")
    ratio_min = float(active_rule.get("ratio_min_inclusive", STEP4_MASS_RATIO_MIN))
    ratio_max = float(active_rule.get("ratio_max_inclusive", STEP4_MASS_RATIO_MAX))
    abs_delta_min = float(active_rule.get("absolute_delta_min_inclusive", STEP4_MASS_ABS_DELTA_MIN))
    for metric in metrics:
        if metric not in reference or metric not in candidate:
            raise ValueError(f"Step-4 mass-confound metric missing: {metric}")
        ref_value = float(reference[metric])
        cand_value = float(candidate[metric])
        abs_delta = abs(cand_value - ref_value)
        denominator = max(abs(ref_value), 1.0)
        ratio = abs(cand_value) / denominator
        if abs_delta >= abs_delta_min and (ratio < ratio_min or ratio > ratio_max):
            return True
    return False


def classify_step4_rank_signal_decomposition(
    *,
    c_matches_a0: bool,
    c_mass_confounded: bool,
    a1_matches_a0: bool = False,
    a0_beats_c: bool = False,
    a1_beats_b: bool = False,
    any_non_reference_matches_a0: bool = False,
) -> str:
    if bool(c_matches_a0) and bool(c_mass_confounded):
        return BRANCH_MASS_CONFOUNDED_CURRENT_ORDER_SIGNAL
    if bool(c_matches_a0):
        return BRANCH_CURRENT_QACC_MARGIN_ORDER_BUNDLE_CARRIER
    if bool(a1_matches_a0):
        return BRANCH_CURRENT_ORDER_NOT_NECESSARY
    if bool(a0_beats_c) and not bool(a1_beats_b):
        return BRANCH_RANK_MAGNITUDE_CONDITIONED_ON_CURRENT_ORDER
    if not bool(any_non_reference_matches_a0):
        return BRANCH_NO_MATCH_DIFFERENT_CREDIT_SOURCE
    return BRANCH_PARTIAL_LOCAL_SIGNAL


def classify_candidate_set_viability_oracle_screen(
    *,
    oracle_feasible: bool,
    candidate_set_contains_ce_improving_move: bool,
    current_credit_rank_recovers_improvement: bool = False,
    deterministic_hash_recovers_improvement: bool = False,
    credit_sign_concordance_positive: bool = False,
    oracle_advantage_over_current: bool = False,
) -> str:
    if not bool(oracle_feasible):
        return BRANCH_ORACLE_INFEASIBLE_OR_TOO_EXPENSIVE
    if not bool(candidate_set_contains_ce_improving_move):
        return BRANCH_CANDIDATE_GENERATION_BAD_OR_NO_LOCAL_SIGNAL
    if bool(deterministic_hash_recovers_improvement) and not bool(current_credit_rank_recovers_improvement):
        return BRANCH_SCHEDULER_ONLY_ORDER_SENSITIVE
    if bool(credit_sign_concordance_positive) and not bool(current_credit_rank_recovers_improvement):
        return BRANCH_CREDIT_MAGNITUDE_BAD_SIGN_USABLE
    if bool(oracle_advantage_over_current) or not bool(current_credit_rank_recovers_improvement):
        return BRANCH_CANDIDATE_SET_VIABLE_CREDIT_RANKING_BAD
    return BRANCH_CANDIDATE_GENERATION_BAD_OR_NO_LOCAL_SIGNAL


def default_prior_verdict_parent_ref(
    *,
    parent_path: str | Path,
    parent_sha256: str,
) -> dict[str, Any]:
    return {
        "verdict_label": "credit_ranking_uninformative_update_law_pivot",
        "candidate_label": "credit_ranking_uninformative_update_law_pivot",
        "artifact_path": "/tmp/hrm158_shadow_prefix_lane3_hybrid_gpu_n50_trace.tierb.final.json",
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "support_name": "L0c2-K2-addition-120",
        "support_sha": "21c8a2f8c15fd68571407e6d1f215ab045ffc5a2a91e4b5a44b50bcd46b6faf0",
        "row_count": 120,
        "lane": "lane3",
        "acc_mode": "applied_crossing_direction_plus_4bit_residual",
        "vote_fidelity": "dry2",
        "activation_mode": "ternary_group128_codec_from_step0",
        "rank_vote_spec_sha256": "6c109e0482292edf72d3cc4ada6bda0840e67e8dbfac4ad7fd64d353602806a5",
        "vote_mapping_family": "rank_bucketed",
        "artifact_literal_parent_path": str(parent_path),
        "artifact_literal_parent_sha256": str(parent_sha256),
        "artifact_literal_support_sha256": "21c8a2f8c15fd68571407e6d1f215ab045ffc5a2a91e4b5a44b50bcd46b6faf0",
        "artifact_literal_acc_mode": "applied_crossing_direction_plus_4bit_residual",
        "artifact_literal_activation_mode": "ternary_group128_codec_from_step0",
        "artifact_literal_vote_fidelity": "dry2",
        "artifact_literal_rank_vote_spec_sha256": "6c109e0482292edf72d3cc4ada6bda0840e67e8dbfac4ad7fd64d353602806a5",
        "artifact_literal_vote_mapping_family": "rank_bucketed",
        "artifact_literal_lane": "lane3",
        "artifact_literal_random_seed": 17,
        "artifact_literal_step_or_sample_count": 50,
        "human_readable_labels": {
            "support_sha_prefix": "21c8a2f8",
            "acc_mode_family": "hybrid",
            "vote_fidelity_family": "tierb",
            "activation_family": "shadow_prefix",
        },
        "support_seed": 17,
        "random_seed": 17,
        "step_or_sample_count": 50,
        "current_beats_competitors_fraction": 0.3,
        "prior_null_setup_verified": False,
        "verification_state": "pending_launch_terminal_receipt",
        "unverified_branch": BRANCH_PRIOR_NULL_SETUP_UNVERIFIED,
        "fallback_branch": BRANCH_INSUFFICIENT_SEPARATION,
        "blocks_control_parity_band": True,
        "verified_at_launch_required": True,
        "control_parity_self_protects": True,
    }


def build_optimizer_update_law_launch_bundle(
    *,
    parent_path: str | Path,
    parent_sha256: str,
    repo_root: str | Path,
    run_root: str | Path,
    device: str = "cuda:0",
    launch_gate_id: str | None = None,
    symbolic_resource_lane: str = "gpu:0",
    phase_timeout_seconds: int | float = 1800,
    total_timeout_seconds: int | float = 14400,
    max_silent_phase_seconds: int | float = 300,
) -> dict[str, Any]:
    commands = [
        _build_probe_command_record(
            repo_root=repo_root,
            run_root=run_root,
            parent_path=parent_path,
            parent_sha256=parent_sha256,
            mode=mode,
            arm_id=arm_id,
            device=device,
            phase_timeout_seconds=phase_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
            max_silent_phase_seconds=max_silent_phase_seconds,
        )
        for mode in (SCIENCE_MODE_PRETERMINAL_SCREEN, SCIENCE_MODE_BRANCH_VERDICT)
        for arm_id in SCIENCE_ARM_IDS
    ]
    bundle = {
        "schema_version": OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION,
        "packet_kind": STEP2_LAUNCH_BUNDLE_PACKET_KIND,
        "target_name": "step2_optimizer_update_law_gpu_launch_bundle",
        "artifact_role": "optimizer_update_law_science_gpu_launch_author_packet",
        "diagnostic_class": DIAGNOSTIC_CLASS_PRE_FULL_STACK,
        "pre_full_stack_diagnostic": True,
        "author_only": True,
        "commands_executed": False,
        "gpu_launched": False,
        "launch_gate_id": launch_gate_id,
        "pt_mutated": False,
        "readiness_claim": False,
        "full_sub2_claim": False,
        "checkpoint_written": False,
        "optimizer_credit_state_row_flip": False,
        "branch_result": None,
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "prior_verdict_parent_ref": default_prior_verdict_parent_ref(
            parent_path=parent_path,
            parent_sha256=parent_sha256,
        ),
        "mode_sequence": [
            SCIENCE_MODE_PRETERMINAL_SCREEN,
            SCIENCE_MODE_BRANCH_VERDICT,
        ],
        "screen_before_verdict": default_screen_before_verdict_dependency(),
        "arms": default_science_arms(include_inverted=True),
        "commands": commands,
        "resource_lane": default_resource_lane_contract(
            symbolic_lane=symbolic_resource_lane,
        ),
        "watcher_audit_bundle": default_watcher_bundle(),
        "phase_budgets": default_phase_budgets(),
        "terminal_criteria": default_terminal_criteria(),
        "hash_gate_policy": default_hash_gate_policy(),
        "compact_instrumentation_only": True,
        "raw_per_proposal_arrays_included": False,
        "artifact_policy": {
            "compact_json_ndjson_only": True,
            "raw_per_proposal_arrays": False,
            "pt_writes_allowed": False,
        },
        "non_claims": [
            "author-only launch packet",
            "no GPU launch from this packet-authoring step",
            "no resource lane acquired by this packet",
            "no .pt mutation",
            "no readiness row flip",
            "no full-sub2 runtime claim",
            "no branch verdict yet",
        ],
    }
    validate_optimizer_update_law_launch_bundle(bundle)
    return bundle


def build_measurement_power_then_trust_region_packet(
    *,
    parent_path: str | Path,
    parent_sha256: str,
    repo_root: str | Path,
    run_root: str | Path,
    device: str = "cuda:0",
    launch_gate_id: str | None = None,
    symbolic_resource_lane: str = "gpu:0",
    phase_timeout_seconds: int | float = 1800,
    total_timeout_seconds: int | float = 14400,
    max_silent_phase_seconds: int | float = 300,
) -> dict[str, Any]:
    phase1_arms = (
        (ARM_A0_RANK_BUCKET_CURRENT, ARM_A0_RANK_BUCKET_CURRENT),
        (ARM_A1_RANK_BUCKET_ORDER_MATCHED, ARM_A1_RANK_BUCKET_ORDER_MATCHED),
        (ARM_B_RANK_FREE_SIGN_PRESSURE, ARM_B_RANK_FREE_SIGN_PRESSURE),
        (ARM_INVERTED_SIGN_PRESSURE, ARM_INVERTED_SIGN_PRESSURE),
    )
    phase2_arms = (
        (ARM_A0_RANK_BUCKET_CURRENT, ARM_A0_RANK_BUCKET_CURRENT, STEP3_BASELINE_MAX_ABS_PER_TENSOR),
        (ARM_A1_RANK_BUCKET_ORDER_MATCHED, ARM_A1_RANK_BUCKET_ORDER_MATCHED, STEP3_BASELINE_MAX_ABS_PER_TENSOR),
        (ARM_B_RANK_FREE_SIGN_PRESSURE, ARM_B_RANK_FREE_SIGN_PRESSURE, STEP3_BASELINE_MAX_ABS_PER_TENSOR),
        (ARM_B_CAP_MAX_ABS_1024, ARM_B_RANK_FREE_SIGN_PRESSURE, STEP3_CAP_MAX_ABS_PER_TENSOR),
        (ARM_INVERTED_SIGN_PRESSURE, ARM_INVERTED_SIGN_PRESSURE, STEP3_BASELINE_MAX_ABS_PER_TENSOR),
    )
    commands = [
        _build_step3_probe_command_record(
            repo_root=repo_root,
            run_root=run_root,
            parent_path=parent_path,
            parent_sha256=parent_sha256,
            phase=phase,
            arm_id=arm_id,
            science_arm=science_arm,
            device=device,
            max_abs_per_tensor=STEP3_BASELINE_MAX_ABS_PER_TENSOR,
            phase_timeout_seconds=phase_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
            max_silent_phase_seconds=max_silent_phase_seconds,
            enabled_if="always for phase-1 power measurement" if phase == STEP3_PHASE_POWER_150 else (
                "only if 150-step rung is measurement_underpowered and safety gates remain clean"
            ),
        )
        for phase in STEP3_POWER_PHASES
        for arm_id, science_arm in phase1_arms
    ]
    commands.extend(
        _build_step3_probe_command_record(
            repo_root=repo_root,
            run_root=run_root,
            parent_path=parent_path,
            parent_sha256=parent_sha256,
            phase=phase,
            arm_id=arm_id,
            science_arm=science_arm,
            device=device,
            max_abs_per_tensor=max_abs,
            phase_timeout_seconds=phase_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
            max_silent_phase_seconds=max_silent_phase_seconds,
            enabled_if=(
                "only if matching phase-1 rung clears power floor and effective_cap.cap_effective=true; "
                "B-A0-loss-only negative floor is loss-rescue only/no acquisition claim"
            ),
        )
        for phase in STEP3_TRUST_REGION_PHASES
        for arm_id, science_arm, max_abs in phase2_arms
    )
    packet = {
        "schema_version": OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION,
        "packet_kind": STEP3_MEASUREMENT_POWER_TRUST_REGION_PACKET_KIND,
        "target_name": "step3_measurement_power_then_trust_region_packet",
        "artifact_role": "optimizer_update_law_measurement_power_trust_region_author_packet",
        "diagnostic_class": DIAGNOSTIC_CLASS_PRE_FULL_STACK,
        "pre_full_stack_diagnostic": True,
        "author_only": True,
        "commands_executed": False,
        "gpu_launched": False,
        "launch_gate_id": launch_gate_id,
        "pt_mutated": False,
        "readiness_claim": False,
        "full_sub2_claim": False,
        "checkpoint_written": False,
        "optimizer_credit_state_row_flip": False,
        "branch_result": None,
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "prior_verdict_parent_ref": default_prior_verdict_parent_ref(
            parent_path=parent_path,
            parent_sha256=parent_sha256,
        ),
        "mode_sequence": [
            STEP3_PHASE_POWER_150,
            STEP3_PHASE_POWER_300,
            STEP3_PHASE_TRUST_REGION_150,
            STEP3_PHASE_TRUST_REGION_300,
        ],
        "power_ladder": {
            "steps_first": 150,
            "steps_optional_continuation": 300,
            "max_steps_hard": STEP3_MAX_STEPS_HARD,
            "continuation_enabled_if": "150-step rung is measurement_underpowered and safety gates clean",
            "floor": default_step3_power_floor(),
        },
        "trust_region": {
            "variable": "max_abs_per_tensor",
            "baseline_value": STEP3_BASELINE_MAX_ABS_PER_TENSOR,
            "cap_value": STEP3_CAP_MAX_ABS_PER_TENSOR,
            "fraction_per_tensor": STEP3_FRACTION_PER_TENSOR,
            "global_cap_contract": "off",
            "global_cap_deferred": True,
            "held_fixed": [
                "rank_free_sign_pressure vote law",
                "A1/order-matched deterministic tie policy",
                "sign direction",
                "parent/support/prior-null",
            ],
            "effective_cap": default_step3_effective_cap_audit(),
        },
        "success_boundary": {
            "requires_power_floor": True,
            "positive_requires": [
                "B_cap_beats_B_baseline_on_paired_exact",
                "B_cap_beats_B_baseline_on_paired_loss",
                "B_cap_beats_A1_on_paired_exact",
                "B_cap_beats_A1_on_paired_loss",
            ],
            "null_pivot": "credit-generation/ranking reformulation",
            "no_additional_rate_cap_variant_on_null": True,
        },
        "arms": default_science_arms(include_inverted=True)
        + [
            {
                "arm_id": ARM_B_CAP_MAX_ABS_1024,
                "vote_law": "rank_free_sign_pressure",
                "ordering_role": "candidate_same_ordering_as_A1",
                "tie_policy_id": TIE_POLICY_DETERMINISTIC_HASH_MATCHED,
                "max_abs_per_tensor": STEP3_CAP_MAX_ABS_PER_TENSOR,
                "required": False,
            },
        ],
        "commands": commands,
        "resource_lane": default_resource_lane_contract(
            symbolic_lane=symbolic_resource_lane,
        ),
        "watcher_audit_bundle": default_watcher_bundle(),
        "phase_budgets": default_phase_budgets(),
        "terminal_criteria": {
            **default_terminal_criteria(),
            "branch_classifier": list(OPTIMIZER_UPDATE_LAW_BRANCHES)
            + [
                BRANCH_CAP_NOOP,
                BRANCH_MEASUREMENT_UNDERPOWERED,
                BRANCH_MEASUREMENT_POWERED,
                BRANCH_MEASUREMENT_LOSS_POWERED,
                BRANCH_POWERED_NEGATIVE_OR_LOSS_ONLY,
            ],
            "step3_power_floor": default_step3_power_floor(),
            "effective_cap_required": True,
            "cap_noop_branch": BRANCH_CAP_NOOP,
        },
        "hash_gate_policy": default_hash_gate_policy(),
        "compact_instrumentation_only": True,
        "raw_per_proposal_arrays_included": False,
        "artifact_policy": {
            "compact_json_ndjson_only": True,
            "raw_per_proposal_arrays": False,
            "pt_writes_allowed": False,
        },
        "non_claims": [
            "author-only Step-3 packet",
            "no GPU launch from this packet-authoring step",
            "no resource lane acquired by this packet",
            "no .pt mutation",
            "no readiness row flip",
            "no full-sub2 runtime claim",
            "power floor is not an acquisition claim",
        ],
    }
    validate_measurement_power_then_trust_region_packet(packet)
    return packet


def _build_step4_probe_command_record(
    *,
    repo_root: str | Path,
    run_root: str | Path,
    parent_path: str | Path,
    parent_sha256: str,
    phase: str,
    arm_id: str,
    device: str,
    phase_timeout_seconds: int | float,
    total_timeout_seconds: int | float,
    max_silent_phase_seconds: int | float,
) -> dict[str, Any]:
    steps_requested = int(STEP4_PHASE_STEPS[str(phase)])
    scratch_root = _path_join(run_root, str(phase), str(arm_id))
    receipt_path = _path_join(scratch_root, "receipt.json")
    stdout_path = _path_join(scratch_root, "stdout.ndjson")
    stderr_path = _path_join(scratch_root, "stderr.log")
    argv = [
        "python3",
        "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--phase",
        f"optimizer-update-law-step4-{phase}-{arm_id}",
        "--device",
        str(device),
        "--parent",
        str(parent_path),
        "--parent-sha256",
        str(parent_sha256),
        "--scratch-root",
        scratch_root,
        "--steps",
        str(steps_requested),
        "--max-steps-hard",
        str(max(STEP4_PHASE_STEPS.values())),
        "--audit-interval",
        str(steps_requested),
        "--science-arm",
        str(arm_id),
        "--max-abs-per-tensor",
        str(STEP3_BASELINE_MAX_ABS_PER_TENSOR),
        "--emit-progress",
        "--phase-timeout-seconds",
        str(phase_timeout_seconds),
        "--total-timeout-seconds",
        str(total_timeout_seconds),
        "--max-silent-phase-seconds",
        str(max_silent_phase_seconds),
    ]
    return {
        "mode": str(phase),
        "phase_role": "rank_signal_decomposition",
        "arm_id": str(arm_id),
        "science_arm": str(arm_id),
        "n_rows": steps_requested,
        "steps_requested": steps_requested,
        "steps_source": "STEP4_PHASE_STEPS[mode]",
        "max_abs_per_tensor": STEP3_BASELINE_MAX_ABS_PER_TENSOR,
        "fraction_per_tensor": STEP3_FRACTION_PER_TENSOR,
        "global_cap_contract": "off",
        "cwd": str(repo_root),
        "env": {
            "HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE": "1",
            "HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH": "1",
        },
        "argv": argv,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "receipt_path": receipt_path,
        "scratch_root": scratch_root,
        "enabled_if": "150 first; 300 only if 150 is powered but ambiguous",
        "expected_exit_policy": "exit_0_required_else_stop_no_retry_no_verdict",
    }


def build_powered_rank_signal_decomposition_packet(
    *,
    parent_path: str | Path,
    parent_sha256: str,
    repo_root: str | Path,
    run_root: str | Path,
    device: str = "cuda:0",
    launch_gate_id: str | None = None,
    symbolic_resource_lane: str = "gpu:0",
    phase_timeout_seconds: int | float = 1800,
    total_timeout_seconds: int | float = 14400,
    max_silent_phase_seconds: int | float = 300,
) -> dict[str, Any]:
    commands = [
        _build_step4_probe_command_record(
            repo_root=repo_root,
            run_root=run_root,
            parent_path=parent_path,
            parent_sha256=parent_sha256,
            phase=phase,
            arm_id=arm_id,
            device=device,
            phase_timeout_seconds=phase_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
            max_silent_phase_seconds=max_silent_phase_seconds,
        )
        for phase in STEP4_PHASES
        for arm_id in STEP4_ARM_IDS
    ]
    packet = {
        "schema_version": OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION,
        "packet_kind": STEP4_POWERED_RANK_SIGNAL_DECOMPOSITION_PACKET_KIND,
        "target_name": "step4_powered_rank_signal_decomposition_packet",
        "artifact_role": "optimizer_update_law_rank_signal_decomposition_author_packet",
        "diagnostic_class": DIAGNOSTIC_CLASS_PRE_FULL_STACK,
        "pre_full_stack_diagnostic": True,
        "author_only": True,
        "commands_executed": False,
        "gpu_launched": False,
        "launch_gate_id": launch_gate_id,
        "pt_mutated": False,
        "readiness_claim": False,
        "full_sub2_claim": False,
        "ready_for_main_science": False,
        "checkpoint_written": False,
        "optimizer_credit_state_row_flip": False,
        "optimizer_credit_state_science_dependent": True,
        "branch_result": None,
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "prior_verdict_parent_ref": default_prior_verdict_parent_ref(
            parent_path=parent_path,
            parent_sha256=parent_sha256,
        ),
        "mode_sequence": list(STEP4_PHASES),
        "power_ladder": {
            "steps_first": 150,
            "steps_optional_continuation": 300,
            "max_steps_hard": max(STEP4_PHASE_STEPS.values()),
            "continuation_enabled_if": (
                "150-step rung is powered but match-to-A0 result is ambiguous; "
                "clear misses stop at 150"
            ),
            "floor": default_step3_power_floor(),
        },
        "match_to_A0_rule": default_step4_match_to_a0_rule(),
        "mass_confound_rule": default_step4_mass_confound_rule(),
        "success_boundary": {
            "carrier_named_only_on_match_to_A0": True,
            "C_claim": BRANCH_CURRENT_QACC_MARGIN_ORDER_BUNDLE_CARRIER,
            "C_claim_caveat": "current qacc-margin/order bundle; margin-vs-index split deferred",
            "C_mass_confounded_branch": BRANCH_MASS_CONFOUNDED_CURRENT_ORDER_SIGNAL,
            "A1_matches_A0_branch": BRANCH_CURRENT_ORDER_NOT_NECESSARY,
            "rank_magnitude_branch": BRANCH_RANK_MAGNITUDE_CONDITIONED_ON_CURRENT_ORDER,
            "no_match_pivot": BRANCH_NO_MATCH_DIFFERENT_CREDIT_SOURCE,
        },
        "arms": default_step4_science_arms(),
        "commands": commands,
        "resource_lane": default_resource_lane_contract(
            symbolic_lane=symbolic_resource_lane,
        ),
        "watcher_audit_bundle": default_watcher_bundle(),
        "phase_budgets": default_phase_budgets(),
        "terminal_criteria": {
            **default_terminal_criteria(),
            "branch_classifier": [
                BRANCH_CURRENT_QACC_MARGIN_ORDER_BUNDLE_CARRIER,
                BRANCH_RANK_MAGNITUDE_CONDITIONED_ON_CURRENT_ORDER,
                BRANCH_CURRENT_ORDER_NOT_NECESSARY,
                BRANCH_PARTIAL_LOCAL_SIGNAL,
                BRANCH_NO_MATCH_DIFFERENT_CREDIT_SOURCE,
                BRANCH_MASS_CONFOUNDED_CURRENT_ORDER_SIGNAL,
                BRANCH_MEASUREMENT_UNDERPOWERED,
                BRANCH_MEASUREMENT_POWERED,
                BRANCH_MEASUREMENT_LOSS_POWERED,
                BRANCH_POWERED_NEGATIVE_OR_LOSS_ONLY,
            ],
            "step4_power_floor": default_step3_power_floor(),
            "match_to_A0_rule": default_step4_match_to_a0_rule(),
            "mass_confound_rule": default_step4_mass_confound_rule(),
            "no_carrier_claim_on_beating_A1_or_B_only": True,
        },
        "hash_gate_policy": default_hash_gate_policy(),
        "compact_instrumentation_only": True,
        "raw_per_proposal_arrays_included": False,
        "artifact_policy": {
            "compact_json_ndjson_only": True,
            "raw_per_proposal_arrays": False,
            "pt_writes_allowed": False,
        },
        "non_claims": [
            "author-only Step-4 packet",
            "no GPU launch from this packet-authoring step",
            "no resource lane acquired by this packet",
            "no .pt mutation",
            "no readiness row flip",
            "no full-sub2 runtime claim",
            "optimizer_credit_state remains science-dependent",
            "C is current qacc-margin/order bundle, not pure current-order rank",
        ],
    }
    validate_powered_rank_signal_decomposition_packet(packet)
    return packet


def _build_step5_probe_command_record(
    *,
    repo_root: str | Path,
    run_root: str | Path,
    parent_path: str | Path,
    parent_sha256: str,
    arm_id: str,
    device: str,
    phase_timeout_seconds: int | float,
    total_timeout_seconds: int | float,
    max_silent_phase_seconds: int | float,
) -> dict[str, Any]:
    phase = STEP4_PHASE_RANK_SIGNAL_150
    steps_requested = int(STEP5_PHASE_STEPS[phase])
    scratch_root = _path_join(run_root, phase, str(arm_id))
    receipt_path = _path_join(scratch_root, "receipt.json")
    stdout_path = _path_join(scratch_root, "stdout.ndjson")
    stderr_path = _path_join(scratch_root, "stderr.log")
    argv = [
        "python3",
        "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--phase",
        f"optimizer-update-law-step5-support-order-{arm_id}",
        "--device",
        str(device),
        "--parent",
        str(parent_path),
        "--parent-sha256",
        str(parent_sha256),
        "--scratch-root",
        scratch_root,
        "--curriculum-seed",
        str(STEP5_CURRICULUM_SEED),
        "--support-order-seed",
        str(STEP5_SUPPORT_ORDER_SEED),
        "--steps",
        str(steps_requested),
        "--max-steps-hard",
        str(max(STEP5_PHASE_STEPS.values())),
        "--audit-interval",
        str(steps_requested),
        "--science-arm",
        str(arm_id),
        "--max-abs-per-tensor",
        str(STEP3_BASELINE_MAX_ABS_PER_TENSOR),
        "--emit-progress",
        "--phase-timeout-seconds",
        str(phase_timeout_seconds),
        "--total-timeout-seconds",
        str(total_timeout_seconds),
        "--max-silent-phase-seconds",
        str(max_silent_phase_seconds),
    ]
    return {
        "mode": phase,
        "phase_role": "support_order_trajectory_robustness",
        "arm_id": str(arm_id),
        "science_arm": str(arm_id),
        "n_rows": steps_requested,
        "steps_requested": steps_requested,
        "steps_source": "STEP5_PHASE_STEPS[mode]",
        "curriculum_seed": STEP5_CURRICULUM_SEED,
        "support_order_seed": STEP5_SUPPORT_ORDER_SEED,
        "support_order_permutation_required": True,
        "qacc_kernelized": False,
        "max_abs_per_tensor": STEP3_BASELINE_MAX_ABS_PER_TENSOR,
        "fraction_per_tensor": STEP3_FRACTION_PER_TENSOR,
        "global_cap_contract": "off",
        "cwd": str(repo_root),
        "env": {
            "HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE": "1",
            "HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH": "1",
        },
        "argv": argv,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "receipt_path": receipt_path,
        "scratch_root": scratch_root,
        "enabled_if": "Step-5 fixed 150-only support-order robustness packet",
        "expected_exit_policy": "exit_0_required_else_stop_no_retry_no_verdict",
    }


def build_support_order_trajectory_robustness_packet(
    *,
    parent_path: str | Path,
    parent_sha256: str,
    repo_root: str | Path,
    run_root: str | Path,
    device: str = "cuda:0",
    launch_gate_id: str | None = None,
    symbolic_resource_lane: str = "gpu:0",
    phase_timeout_seconds: int | float = 1800,
    total_timeout_seconds: int | float = 7200,
    max_silent_phase_seconds: int | float = 300,
) -> dict[str, Any]:
    commands = [
        _build_step5_probe_command_record(
            repo_root=repo_root,
            run_root=run_root,
            parent_path=parent_path,
            parent_sha256=parent_sha256,
            arm_id=arm_id,
            device=device,
            phase_timeout_seconds=phase_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
            max_silent_phase_seconds=max_silent_phase_seconds,
        )
        for arm_id in STEP5_ARM_IDS
    ]
    packet = {
        "schema_version": OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION,
        "packet_kind": STEP5_SUPPORT_ORDER_TRAJECTORY_ROBUSTNESS_PACKET_KIND,
        "target_name": "support_order_trajectory_robustness_packet",
        "artifact_role": "optimizer_update_law_support_order_trajectory_robustness_author_packet",
        "diagnostic_class": DIAGNOSTIC_CLASS_PRE_FULL_STACK,
        "pre_full_stack_diagnostic": True,
        "author_only": True,
        "commands_executed": False,
        "gpu_launched": False,
        "launch_gate_id": launch_gate_id,
        "pt_mutated": False,
        "readiness_claim": False,
        "full_sub2_claim": False,
        "ready_for_main_science": False,
        "checkpoint_written": False,
        "optimizer_credit_state_row_flip": False,
        "optimizer_credit_state_science_dependent": True,
        "branch_result": None,
        "qacc_kernelized": False,
        "qacc_cpu_reference_caveat": (
            "qacc vote/select/apply/update remains CPU-reference/default-off; "
            "this packet is not a hot-loop kernel residency proof"
        ),
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "prior_verdict_parent_ref": default_prior_verdict_parent_ref(
            parent_path=parent_path,
            parent_sha256=parent_sha256,
        ),
        "mode_sequence": list(STEP5_PHASES),
        "support_order_seed": STEP5_SUPPORT_ORDER_SEED,
        "curriculum_seed": STEP5_CURRICULUM_SEED,
        "support_order_proof_contract": default_step5_support_order_proof_contract(),
        "power_ladder": {
            "steps_first": 150,
            "steps_optional_continuation": None,
            "max_steps_hard": max(STEP5_PHASE_STEPS.values()),
            "continuation_enabled_if": "disabled; Step-5 is 150-only",
            "floor": default_step3_power_floor(),
        },
        "pass_rule": default_step5_pass_rule(),
        "mass_confound_rule": default_step4_mass_confound_rule(),
        "success_boundary": {
            "positive_label": BRANCH_CURRENT_QACC_MARGIN_ORDER_BUNDLE_CARRIER,
            "positive_requires": [
                "C.strict >= 10/90",
                "C.strict - max(A0.strict,B.strict) >= 5",
                "C_minus_A0 paired-loss mean<0 and 95% CI high<0",
                "C_minus_B paired-loss mean<0 and 95% CI high<0",
                "C_vs_A0 and C_vs_B mass-confound pass",
                "inverted_sign_pressure does not exceed C",
            ],
            "readiness_after_pass": False,
            "not_independent_seed_robustness": True,
        },
        "arms": default_step5_science_arms(),
        "commands": commands,
        "resource_lane": default_resource_lane_contract(
            symbolic_lane=symbolic_resource_lane,
        ),
        "watcher_audit_bundle": default_watcher_bundle(),
        "phase_budgets": default_phase_budgets(),
        "terminal_criteria": {
            **default_terminal_criteria(),
            "branch_classifier": [
                BRANCH_CURRENT_QACC_MARGIN_ORDER_BUNDLE_CARRIER,
                BRANCH_MASS_CONFOUNDED_CURRENT_ORDER_SIGNAL,
                BRANCH_POWERED_NEGATIVE_OR_LOSS_ONLY,
            ],
            "pass_rule": default_step5_pass_rule(),
            "mass_confound_rule": default_step4_mass_confound_rule(),
            "terminal_receipt_required_tables": [
                "strict_exact_by_arm",
                "paired_loss_C_minus_A0_and_C_minus_B",
                "mass_confound_C_vs_A0_and_C_vs_B",
                "device_vs_hot_loop_qacc_kernelized_false",
            ],
            "terminal_receipt_required_proofs": [
                "parent_sha256_pre_post_unchanged",
                "resource_lane_released",
                "artifact_paths",
                "support_order_seed",
                "support_order_original_ordered_traversal_hash16",
                "support_order_permuted_ordered_traversal_hash16",
                "support_order_original_invariant_multiset_hash16",
                "support_order_permuted_invariant_multiset_hash16",
                "support_content_unchanged_from_order_invariant_hash",
            ],
            "qacc_kernelized": False,
            "device_residency_not_hot_loop_residency": True,
            "ready_for_main_science_after_pass": False,
        },
        "hash_gate_policy": default_hash_gate_policy(),
        "compact_instrumentation_only": True,
        "raw_per_proposal_arrays_included": False,
        "artifact_policy": {
            "compact_json_ndjson_only": True,
            "raw_per_proposal_arrays": False,
            "pt_writes_allowed": False,
        },
        "non_claims": [
            "author-only Step-5 packet",
            "support_order_trajectory_robustness, not independent-seed robustness",
            "no GPU launch from this packet-authoring step",
            "no resource lane acquired by this packet",
            "no .pt mutation",
            "no readiness row flip",
            "no full-sub2 runtime claim",
            "ready_for_main_science remains false",
            "qacc_vote_select_apply_update remains CPU-reference/default-off, not kernelized",
        ],
    }
    validate_support_order_trajectory_robustness_packet(packet)
    return packet


def _build_step6_probe_command_record(
    *,
    repo_root: str | Path,
    run_root: str | Path,
    parent_path: str | Path,
    parent_sha256: str,
    arm_id: str,
    support_order_seed: int | None,
    device: str,
    phase_timeout_seconds: int | float,
    total_timeout_seconds: int | float,
    max_silent_phase_seconds: int | float,
) -> dict[str, Any]:
    phase = STEP4_PHASE_RANK_SIGNAL_150
    steps_requested = int(STEP6_PHASE_STEPS[phase])
    seed_label = _support_order_seed_label(support_order_seed)
    scratch_root = _path_join(run_root, phase, seed_label, str(arm_id))
    receipt_path = _path_join(scratch_root, "receipt.json")
    stdout_path = _path_join(scratch_root, "stdout.ndjson")
    stderr_path = _path_join(scratch_root, "stderr.log")
    argv = [
        "python3",
        "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--phase",
        f"optimizer-update-law-step6-order-averaged-{seed_label}-{arm_id}",
        "--device",
        str(device),
        "--parent",
        str(parent_path),
        "--parent-sha256",
        str(parent_sha256),
        "--scratch-root",
        scratch_root,
        "--curriculum-seed",
        str(STEP6_CURRICULUM_SEED),
    ]
    if support_order_seed is not None:
        argv.extend(["--support-order-seed", str(int(support_order_seed))])
    argv.extend(
        [
            "--steps",
            str(steps_requested),
            "--max-steps-hard",
            str(max(STEP6_PHASE_STEPS.values())),
            "--audit-interval",
            str(steps_requested),
            "--science-arm",
            str(arm_id),
            "--max-abs-per-tensor",
            str(STEP3_BASELINE_MAX_ABS_PER_TENSOR),
            "--emit-progress",
            "--phase-timeout-seconds",
            str(phase_timeout_seconds),
            "--total-timeout-seconds",
            str(total_timeout_seconds),
            "--max-silent-phase-seconds",
            str(max_silent_phase_seconds),
        ],
    )
    return {
        "mode": phase,
        "phase_role": "order_averaged_a0_component_decomposition",
        "arm_id": str(arm_id),
        "science_arm": str(arm_id),
        "seed_label": seed_label,
        "support_order_seed": support_order_seed,
        "support_order_permutation_required": support_order_seed is not None,
        "fresh_step6_evidence": True,
        "context_only": False,
        "classifier_evidence": True,
        "n_rows": steps_requested,
        "steps_requested": steps_requested,
        "steps_source": "STEP6_PHASE_STEPS[mode]",
        "curriculum_seed": STEP6_CURRICULUM_SEED,
        "qacc_kernelized": False,
        "max_abs_per_tensor": STEP3_BASELINE_MAX_ABS_PER_TENSOR,
        "fraction_per_tensor": STEP3_FRACTION_PER_TENSOR,
        "global_cap_contract": "off",
        "cwd": str(repo_root),
        "env": {
            "HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE": "1",
            "HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH": "1",
        },
        "argv": argv,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "receipt_path": receipt_path,
        "scratch_root": scratch_root,
        "enabled_if": "Step-6 fixed fresh-all-9 order-averaged A0 decomposition packet",
        "expected_exit_policy": "exit_0_required_else_stop_no_retry_no_verdict",
    }


def build_order_averaged_a0_component_decomposition_packet(
    *,
    parent_path: str | Path,
    parent_sha256: str,
    repo_root: str | Path,
    run_root: str | Path,
    device: str = "cuda:0",
    launch_gate_id: str | None = None,
    symbolic_resource_lane: str = "gpu:0",
    phase_timeout_seconds: int | float = 1800,
    total_timeout_seconds: int | float = 7200,
    max_silent_phase_seconds: int | float = 300,
) -> dict[str, Any]:
    commands = [
        _build_step6_probe_command_record(
            repo_root=repo_root,
            run_root=run_root,
            parent_path=parent_path,
            parent_sha256=parent_sha256,
            arm_id=arm_id,
            support_order_seed=support_order_seed,
            device=device,
            phase_timeout_seconds=phase_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
            max_silent_phase_seconds=max_silent_phase_seconds,
        )
        for support_order_seed in STEP6_SUPPORT_ORDER_SEEDS
        for arm_id in STEP6_ARM_IDS
    ]
    packet = {
        "schema_version": OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION,
        "packet_kind": STEP6_ORDER_AVERAGED_A0_COMPONENT_DECOMPOSITION_PACKET_KIND,
        "target_name": "step6_order_averaged_a0_component_decomposition_packet",
        "artifact_role": "optimizer_update_law_order_averaged_a0_component_author_packet",
        "diagnostic_class": DIAGNOSTIC_CLASS_PRE_FULL_STACK,
        "pre_full_stack_diagnostic": True,
        "author_only": True,
        "commands_executed": False,
        "gpu_launched": False,
        "launch_gate_id": launch_gate_id,
        "pt_mutated": False,
        "readiness_claim": False,
        "full_sub2_claim": False,
        "ready_for_main_science": False,
        "checkpoint_written": False,
        "optimizer_credit_state_row_flip": False,
        "optimizer_credit_state_science_dependent": True,
        "branch_result": None,
        "qacc_kernelized": False,
        "qacc_cpu_reference_caveat": (
            "qacc vote/select/apply/update remains CPU-reference/default-off; "
            "this packet is not a hot-loop kernel residency proof"
        ),
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "prior_verdict_parent_ref": default_prior_verdict_parent_ref(
            parent_path=parent_path,
            parent_sha256=parent_sha256,
        ),
        "mode_sequence": list(STEP6_PHASES),
        "support_order_seeds": list(STEP6_SUPPORT_ORDER_SEEDS),
        "curriculum_seed": STEP6_CURRICULUM_SEED,
        "support_order_proof_contract": default_step6_support_order_proof_contract(),
        "power_ladder": {
            "steps_first": 150,
            "steps_optional_continuation": None,
            "max_steps_hard": max(STEP6_PHASE_STEPS.values()),
            "continuation_enabled_if": "disabled; Step-6 is 150-only",
            "floor": default_step3_power_floor(),
        },
        "order_averaged_stability_rule": default_step6_stability_rule(),
        "mass_confound_rule": default_step6_mass_confound_rule(),
        "context_only_prior_receipts": [
            {
                "label": "step4_original_order_context",
                "context_only": True,
                "classifier_evidence": False,
                "reason": "motivated order-averaged design; not Step-6 verdict evidence",
            },
            {
                "label": "step5_seed29_context",
                "context_only": True,
                "classifier_evidence": False,
                "reason": "falsified C single-order claim; not Step-6 verdict evidence",
            },
        ],
        "success_boundary": {
            "positive_label": BRANCH_A0_COMPONENT_ORDER_ROBUST,
            "positive_requires": [
                "A0 beats A1 and C in at least 2/3 preregistered seeds",
                "per-seed paired loss supports A0 over A1 and C",
                "pooled paired-row loss is secondary and cannot override seed instability",
                "A0_vs_A1 and A0_vs_C mass-confound tables reported",
                "fresh Step-6 receipts only; Step-4/5 receipts are context-only",
            ],
            "unstable_label": BRANCH_MEASUREMENT_ORDER_SENSITIVE,
            "readiness_after_pass": False,
            "carrier_claim_after_pass": False,
        },
        "arms": default_step6_science_arms(),
        "commands": commands,
        "cost_ceiling": {
            "max_arm_runs": STEP6_MAX_ARM_RUNS,
            "max_gpu_hours": STEP6_GPU_HOUR_CEILING,
            "stop_before_launch_if_exceeded": True,
        },
        "resource_lane": default_resource_lane_contract(
            symbolic_lane=symbolic_resource_lane,
        ),
        "watcher_audit_bundle": default_watcher_bundle(),
        "phase_budgets": default_phase_budgets(),
        "terminal_criteria": {
            **default_terminal_criteria(),
            "branch_classifier": [
                BRANCH_A0_COMPONENT_ORDER_ROBUST,
                BRANCH_MEASUREMENT_ORDER_SENSITIVE,
                BRANCH_PARTIAL_LOCAL_SIGNAL,
            ],
            "order_averaged_stability_rule": default_step6_stability_rule(),
            "mass_confound_rule": default_step6_mass_confound_rule(),
            "terminal_receipt_required_tables": [
                "strict_count_distributions_by_arm_across_seeds",
                "paired_loss_A0_minus_A1_and_A0_minus_C_per_seed",
                "pooled_paired_row_loss_secondary_only",
                "seed_wise_rank_ordering",
                "mass_confound_A0_vs_A1_and_A0_vs_C",
                "fresh_step6_vs_context_only_evidence_ledger",
                "device_vs_hot_loop_qacc_kernelized_false",
            ],
            "terminal_receipt_required_proofs": [
                "parent_sha256_pre_post_unchanged",
                "resource_lane_released",
                "artifact_paths",
                "support_order_seed_matrix_original_29_43",
                "original_argv_omits_support_order_seed",
                "seed29_seed43_argv_include_support_order_seed",
                "fresh_step6_receipts_only_for_classifier",
            ],
            "qacc_kernelized": False,
            "device_residency_not_hot_loop_residency": True,
            "ready_for_main_science_after_pass": False,
            "carrier_claim_after_pass": False,
        },
        "hash_gate_policy": default_hash_gate_policy(),
        "compact_instrumentation_only": True,
        "raw_per_proposal_arrays_included": False,
        "artifact_policy": {
            "compact_json_ndjson_only": True,
            "raw_per_proposal_arrays": False,
            "pt_writes_allowed": False,
        },
        "non_claims": [
            "author-only Step-6 packet",
            "fresh-all-9 order-averaged A0/A1/C measurement-validity diagnostic",
            "no GPU launch from this packet-authoring step",
            "no resource lane acquired by this packet",
            "no .pt mutation",
            "no readiness row flip",
            "no carrier claim even on pass",
            "no full-sub2 runtime claim",
            "ready_for_main_science remains false",
            "qacc_vote_select_apply_update remains CPU-reference/default-off, not kernelized",
            "Step-4/5 receipts are context-only rationale, not classifier evidence",
        ],
    }
    validate_order_averaged_a0_component_decomposition_packet(packet)
    return packet


def build_candidate_set_viability_oracle_screen_packet(
    *,
    parent_path: str | Path,
    parent_sha256: str,
    launch_gate_id: str | None = None,
    max_sampled_candidates: int = ORACLE_SCREEN_FEASIBILITY_MAX_SAMPLED_CANDIDATES,
) -> dict[str, Any]:
    budget = int(max_sampled_candidates)
    packet = {
        "schema_version": OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION,
        "packet_kind": ORACLE_SCREEN_PACKET_KIND,
        "target_name": ORACLE_SCREEN_PACKET_KIND,
        "artifact_role": "candidate_set_viability_oracle_screen_author_packet",
        "diagnostic_class": DIAGNOSTIC_CLASS_PRE_FULL_STACK,
        "pre_full_stack_diagnostic": True,
        "author_only": True,
        "commands_executed": False,
        "gpu_launched": False,
        "launch_gate_id": launch_gate_id,
        "pt_mutated": False,
        "readiness_claim": False,
        "full_sub2_claim": False,
        "ready_for_main_science": False,
        "carrier_claim": False,
        "checkpoint_written": False,
        "optimizer_credit_state_row_flip": False,
        "optimizer_credit_state_science_dependent": True,
        "branch_result": None,
        "qacc_kernelized": False,
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "arms": default_oracle_screen_arms(),
        "same_candidate_set_required": True,
        "seed_order_contract": default_oracle_screen_seed_order_contract(),
        "oracle_feasibility_budget": default_oracle_feasibility_budget_for(
            max_sampled_candidates=budget
        ),
        "oracle_non_persistence_contract": default_oracle_non_persistence_contract(),
        "compact_summary_schema": default_oracle_compact_summary_schema(),
        "classifier_contract": default_oracle_screen_classifier_contract(),
        "wider_screen_interpretation_contract": (
            default_oracle_wider_screen_interpretation_contract(
                max_sampled_candidates=budget
            )
        ),
        "fallback": {
            "fallback_mode": "decile_only_concordance",
            "oracle_applied_arm_allowed": False,
            "enabled_if": "oracle_applied_arm_reads_as_leakage_risky",
        },
        "artifact_policy": {
            "compact_json_ndjson_only": True,
            "raw_per_proposal_arrays": False,
            "pt_writes_allowed": False,
            "oracle_artifact_path": None,
        },
        "non_claims": [
            "packet scaffold only; no oracle execution",
            "no GPU launch from packet authoring",
            "no .pt mutation or checkpoint promotion",
            "diagnostic_local_loss_delta never persists q",
            "oracle is not a learner or teacher",
            "no readiness row flip",
            "no carrier or full-sub2 runtime claim",
            "optimizer_credit_state remains science-dependent",
        ],
    }
    validate_candidate_set_viability_oracle_screen_packet(packet)
    return packet


def _build_oracle_screen_probe_command_record(
    *,
    repo_root: str | Path,
    run_root: str | Path,
    parent_path: str | Path,
    parent_sha256: str,
    support_order_seed: int,
    max_sampled_candidates: int,
    device: str,
    phase_timeout_seconds: int | float,
    total_timeout_seconds: int | float,
    max_silent_phase_seconds: int | float,
) -> dict[str, Any]:
    budget = int(max_sampled_candidates)
    seed_label = _support_order_seed_label(int(support_order_seed))
    scratch_root = _path_join(run_root, seed_label)
    receipt_path = _path_join(scratch_root, "receipt.json")
    stdout_path = _path_join(scratch_root, "stdout.ndjson")
    stderr_path = _path_join(scratch_root, "stderr.log")
    argv = [
        "python3",
        "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--oracle-screen-mode",
        "candidate_set_viability",
        "--phase",
        f"optimizer-update-law-oracle-screen-n20-{seed_label}",
        "--device",
        str(device),
        "--parent",
        str(parent_path),
        "--parent-sha256",
        str(parent_sha256),
        "--scratch-root",
        scratch_root,
        "--curriculum-seed",
        str(STEP6_CURRICULUM_SEED),
        "--support-order-seed",
        str(int(support_order_seed)),
        "--oracle-screen-max-sampled-candidates",
        str(budget),
        "--batch-size",
        str(ORACLE_SCREEN_N20_ROWS),
        "--steps",
        "1",
        "--max-steps-hard",
        "1",
        "--max-abs-per-tensor",
        str(STEP3_BASELINE_MAX_ABS_PER_TENSOR),
        "--emit-progress",
        "--phase-timeout-seconds",
        str(phase_timeout_seconds),
        "--total-timeout-seconds",
        str(total_timeout_seconds),
        "--max-silent-phase-seconds",
        str(max_silent_phase_seconds),
    ]
    return {
        "mode": "oracle_screen_n20",
        "phase_role": "candidate_set_viability_oracle_screen",
        "support_order_seed": int(support_order_seed),
        "seed_label": seed_label,
        "oracle_screen_mode": "candidate_set_viability",
        "screen_rows": ORACLE_SCREEN_N20_ROWS,
        "n_rows": ORACLE_SCREEN_N20_ROWS,
        "steps_requested": 1,
        "steps_source": "fixed_single_support_batch_oracle_screen",
        "batch_size": ORACLE_SCREEN_N20_ROWS,
        "curriculum_seed": STEP6_CURRICULUM_SEED,
        "same_candidate_set_required": True,
        "max_sampled_candidates": budget,
        "oracle_max_seconds": oracle_screen_budget_max_seconds(budget),
        "max_abs_per_tensor": STEP3_BASELINE_MAX_ABS_PER_TENSOR,
        "fraction_per_tensor": STEP3_FRACTION_PER_TENSOR,
        "global_cap_contract": "off",
        "cwd": str(repo_root),
        "env": {
            "HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE": "1",
            "HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH": "1",
        },
        "argv": argv,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "receipt_path": receipt_path,
        "scratch_root": scratch_root,
        "enabled_if": "fixed contrast seeds 43 and 29 only; same candidate set once per seed",
        "expected_exit_policy": "exit_0_required_else_stop_no_retry_no_verdict",
    }


def build_candidate_set_viability_oracle_screen_launch_bundle(
    *,
    parent_path: str | Path,
    parent_sha256: str,
    repo_root: str | Path,
    run_root: str | Path,
    max_sampled_candidates: int = ORACLE_SCREEN_FEASIBILITY_MAX_SAMPLED_CANDIDATES,
    device: str = "cuda:0",
    launch_gate_id: str | None = None,
    symbolic_resource_lane: str = "gpu:0",
    phase_timeout_seconds: int | float = 1800,
    total_timeout_seconds: int | float = 7200,
    max_silent_phase_seconds: int | float = 300,
) -> dict[str, Any]:
    budget = int(max_sampled_candidates)
    science_contract = packet_without_runtime_results(
        build_candidate_set_viability_oracle_screen_packet(
            parent_path=parent_path,
            parent_sha256=parent_sha256,
            launch_gate_id=None,
            max_sampled_candidates=budget,
        ),
    )
    commands = [
        _build_oracle_screen_probe_command_record(
            repo_root=repo_root,
            run_root=run_root,
            parent_path=parent_path,
            parent_sha256=parent_sha256,
            support_order_seed=int(seed),
            max_sampled_candidates=budget,
            device=device,
            phase_timeout_seconds=phase_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
            max_silent_phase_seconds=max_silent_phase_seconds,
        )
        for seed in ORACLE_SCREEN_CONTRAST_SEEDS
    ]
    packet = {
        "schema_version": OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION,
        "packet_kind": ORACLE_SCREEN_LAUNCH_BUNDLE_PACKET_KIND,
        "target_name": ORACLE_SCREEN_LAUNCH_BUNDLE_PACKET_KIND,
        "artifact_role": "candidate_set_viability_oracle_screen_launch_bundle_author_packet",
        "diagnostic_class": DIAGNOSTIC_CLASS_PRE_FULL_STACK,
        "pre_full_stack_diagnostic": True,
        "author_only": True,
        "commands_executed": False,
        "gpu_launched": False,
        "launch_gate_id": launch_gate_id,
        "pt_mutated": False,
        "readiness_claim": False,
        "full_sub2_claim": False,
        "ready_for_main_science": False,
        "carrier_claim": False,
        "checkpoint_written": False,
        "optimizer_credit_state_row_flip": False,
        "optimizer_credit_state_science_dependent": True,
        "branch_result": None,
        "qacc_kernelized": False,
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "science_contract_commit_sha": ORACLE_SCREEN_SCIENCE_CONTRACT_COMMIT_SHA,
        "science_contract": science_contract,
        "screen_rows": ORACLE_SCREEN_N20_ROWS,
        "curriculum_seed": STEP6_CURRICULUM_SEED,
        "support_order_seeds": list(ORACLE_SCREEN_CONTRAST_SEEDS),
        "same_candidate_set_required": True,
        "oracle_feasibility_budget": default_oracle_feasibility_budget_for(
            max_sampled_candidates=budget
        ),
        "oracle_non_persistence_contract": default_oracle_non_persistence_contract(),
        "compact_summary_schema": default_oracle_compact_summary_schema(),
        "classifier_contract": default_oracle_screen_classifier_contract(),
        "wider_screen_interpretation_contract": (
            default_oracle_wider_screen_interpretation_contract(
                max_sampled_candidates=budget
            )
        ),
        "commands": commands,
        "resource_lane": default_resource_lane_contract(
            symbolic_lane=symbolic_resource_lane,
        ),
        "watcher_audit_bundle": default_watcher_bundle(),
        "phase_budgets": default_phase_budgets(),
        "terminal_criteria": {
            **default_terminal_criteria(),
            "branch_classifier": list(ORACLE_SCREEN_BRANCHES),
            "same_candidate_set_required": True,
            "support_order_seeds": list(ORACLE_SCREEN_CONTRAST_SEEDS),
            "n20_screen_rows": ORACLE_SCREEN_N20_ROWS,
            "qacc_kernelized": False,
            "device_residency_not_hot_loop_residency": True,
        },
        "hash_gate_policy": default_hash_gate_policy(),
        "compact_instrumentation_only": True,
        "raw_per_proposal_arrays_included": False,
        "artifact_policy": {
            "compact_json_ndjson_only": True,
            "raw_per_proposal_arrays": False,
            "pt_writes_allowed": False,
        },
        "non_claims": [
            "author-only oracle-screen launch bundle",
            "embeds afbe598 science contract without mutating it",
            "fixed N=20 contrast screen only; promotion to N=50x3 remains separately gated",
            "same candidate set generated once per seed and rescored under three arms",
            "no GPU launch from this packet-authoring step",
            "no resource lane acquired by this packet",
            "no .pt mutation or checkpoint promotion",
            "no readiness row flip",
            "no carrier or full-sub2 runtime claim",
            "qacc_vote_select_apply_update remains CPU-reference/default-off, not kernelized",
        ],
    }
    validate_candidate_set_viability_oracle_screen_launch_bundle(packet)
    return packet


def build_credit_ranking_pivot_measurement_packet(
    *,
    parent_path: str | Path,
    parent_sha256: str,
    launch_gate_id: str | None = None,
) -> dict[str, Any]:
    budget = PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES
    packet = {
        "schema_version": OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION,
        "packet_kind": CREDIT_RANKING_PIVOT_MEASUREMENT_PACKET_KIND,
        "target_name": CREDIT_RANKING_PIVOT_MEASUREMENT_PACKET_KIND,
        "artifact_role": "credit_ranking_pivot_measurement_author_packet",
        "diagnostic_class": DIAGNOSTIC_CLASS_PRE_FULL_STACK,
        "pre_full_stack_diagnostic": True,
        "author_only": True,
        "commands_executed": False,
        "gpu_launched": False,
        "launch_gate_id": launch_gate_id,
        "pt_mutated": False,
        "readiness_claim": False,
        "full_sub2_claim": False,
        "ready_for_main_science": False,
        "carrier_claim": False,
        "checkpoint_written": False,
        "optimizer_credit_state_row_flip": False,
        "optimizer_credit_state_science_dependent": True,
        "branch_result": None,
        "qacc_kernelized": False,
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "arms": default_oracle_screen_arms(),
        "same_candidate_set_required": True,
        "seed_order_contract": default_oracle_screen_seed_order_contract(),
        "oracle_feasibility_budget": default_oracle_feasibility_budget_for(
            max_sampled_candidates=budget
        ),
        "oracle_non_persistence_contract": default_oracle_non_persistence_contract(),
        "compact_summary_schema": default_credit_ranking_pivot_compact_summary_schema(),
        "measurement_contract": default_credit_ranking_pivot_measurement_contract(),
        "artifact_policy": {
            "compact_json_ndjson_only": True,
            "raw_per_proposal_arrays": False,
            "pt_writes_allowed": False,
            "oracle_artifact_path": None,
        },
        "non_claims": [
            "packet scaffold only; no measurement execution",
            "no GPU launch from packet authoring",
            "no .pt mutation or checkpoint promotion",
            "measurement keeps candidate generation fixed",
            "deterministic hash remains null/control only",
            "no readiness row flip",
            "no carrier or full-sub2 runtime claim",
            "optimizer_credit_state remains science-dependent",
        ],
    }
    validate_credit_ranking_pivot_measurement_packet(packet)
    return packet


def _build_credit_ranking_pivot_measurement_command_record(
    *,
    repo_root: str | Path,
    run_root: str | Path,
    parent_path: str | Path,
    parent_sha256: str,
    support_order_seed: int,
    device: str,
    phase_timeout_seconds: int | float,
    total_timeout_seconds: int | float,
    max_silent_phase_seconds: int | float,
) -> dict[str, Any]:
    budget = PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES
    seed_label = _support_order_seed_label(int(support_order_seed))
    scratch_root = _path_join(run_root, seed_label)
    receipt_path = _path_join(scratch_root, "receipt.json")
    stdout_path = _path_join(scratch_root, "stdout.ndjson")
    stderr_path = _path_join(scratch_root, "stderr.log")
    argv = [
        "python3",
        "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--oracle-screen-mode",
        "credit_ranking_pivot_measurement",
        "--phase",
        f"credit-ranking-pivot-measurement-n20-{seed_label}",
        "--device",
        str(device),
        "--parent",
        str(parent_path),
        "--parent-sha256",
        str(parent_sha256),
        "--scratch-root",
        scratch_root,
        "--curriculum-seed",
        str(STEP6_CURRICULUM_SEED),
        "--support-order-seed",
        str(int(support_order_seed)),
        "--oracle-screen-max-sampled-candidates",
        str(budget),
        "--batch-size",
        str(ORACLE_SCREEN_N20_ROWS),
        "--steps",
        "1",
        "--max-steps-hard",
        "1",
        "--max-abs-per-tensor",
        str(STEP3_BASELINE_MAX_ABS_PER_TENSOR),
        "--emit-progress",
        "--phase-timeout-seconds",
        str(phase_timeout_seconds),
        "--total-timeout-seconds",
        str(total_timeout_seconds),
        "--max-silent-phase-seconds",
        str(max_silent_phase_seconds),
    ]
    return {
        "mode": "credit_ranking_pivot_measurement_n20",
        "phase_role": "credit_ranking_pivot_measurement",
        "support_order_seed": int(support_order_seed),
        "seed_label": seed_label,
        "oracle_screen_mode": "credit_ranking_pivot_measurement",
        "screen_rows": ORACLE_SCREEN_N20_ROWS,
        "n_rows": ORACLE_SCREEN_N20_ROWS,
        "steps_requested": 1,
        "steps_source": "fixed_single_support_batch_credit_ranking_pivot_measurement",
        "batch_size": ORACLE_SCREEN_N20_ROWS,
        "curriculum_seed": STEP6_CURRICULUM_SEED,
        "same_candidate_set_required": True,
        "max_sampled_candidates": budget,
        "oracle_max_seconds": oracle_screen_budget_max_seconds(budget),
        "max_abs_per_tensor": STEP3_BASELINE_MAX_ABS_PER_TENSOR,
        "fraction_per_tensor": STEP3_FRACTION_PER_TENSOR,
        "global_cap_contract": "off",
        "cwd": str(repo_root),
        "env": {
            "HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE": "1",
            "HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH": "1",
        },
        "argv": argv,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "receipt_path": receipt_path,
        "scratch_root": scratch_root,
        "enabled_if": "fixed contrast seeds 43 and 29 only; same candidate set once per seed",
        "expected_exit_policy": "exit_0_required_else_stop_no_retry_no_verdict",
    }


def build_credit_ranking_pivot_measurement_launch_bundle(
    *,
    parent_path: str | Path,
    parent_sha256: str,
    repo_root: str | Path,
    run_root: str | Path,
    device: str = "cuda:0",
    launch_gate_id: str | None = None,
    symbolic_resource_lane: str = "gpu:0",
    phase_timeout_seconds: int | float = 1800,
    total_timeout_seconds: int | float = 7200,
    max_silent_phase_seconds: int | float = 300,
) -> dict[str, Any]:
    budget = PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES
    science_contract = packet_without_runtime_results(
        build_credit_ranking_pivot_measurement_packet(
            parent_path=parent_path,
            parent_sha256=parent_sha256,
            launch_gate_id=None,
        ),
    )
    commands = [
        _build_credit_ranking_pivot_measurement_command_record(
            repo_root=repo_root,
            run_root=run_root,
            parent_path=parent_path,
            parent_sha256=parent_sha256,
            support_order_seed=int(seed),
            device=device,
            phase_timeout_seconds=phase_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
            max_silent_phase_seconds=max_silent_phase_seconds,
        )
        for seed in ORACLE_SCREEN_CONTRAST_SEEDS
    ]
    packet = {
        "schema_version": OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION,
        "packet_kind": CREDIT_RANKING_PIVOT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND,
        "target_name": CREDIT_RANKING_PIVOT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND,
        "artifact_role": "credit_ranking_pivot_measurement_launch_bundle_author_packet",
        "diagnostic_class": DIAGNOSTIC_CLASS_PRE_FULL_STACK,
        "pre_full_stack_diagnostic": True,
        "author_only": True,
        "commands_executed": False,
        "gpu_launched": False,
        "launch_gate_id": launch_gate_id,
        "pt_mutated": False,
        "readiness_claim": False,
        "full_sub2_claim": False,
        "ready_for_main_science": False,
        "carrier_claim": False,
        "checkpoint_written": False,
        "optimizer_credit_state_row_flip": False,
        "optimizer_credit_state_science_dependent": True,
        "branch_result": None,
        "qacc_kernelized": False,
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "science_contract": science_contract,
        "screen_rows": ORACLE_SCREEN_N20_ROWS,
        "curriculum_seed": STEP6_CURRICULUM_SEED,
        "support_order_seeds": list(ORACLE_SCREEN_CONTRAST_SEEDS),
        "same_candidate_set_required": True,
        "oracle_feasibility_budget": default_oracle_feasibility_budget_for(
            max_sampled_candidates=budget
        ),
        "oracle_non_persistence_contract": default_oracle_non_persistence_contract(),
        "compact_summary_schema": default_credit_ranking_pivot_compact_summary_schema(),
        "measurement_contract": default_credit_ranking_pivot_measurement_contract(),
        "commands": commands,
        "resource_lane": default_resource_lane_contract(
            symbolic_lane=symbolic_resource_lane,
        ),
        "watcher_audit_bundle": default_watcher_bundle(),
        "phase_budgets": default_phase_budgets(),
        "terminal_criteria": {
            **default_terminal_criteria(),
            "branch_classifier": [
                BRANCH_PREREGISTERED_CHEAP_LEARNER_FEATURE_FAMILY_CANNOT_PREDICT_REGRET,
                "primary_score_predictive_for_local_regret",
                BRANCH_MEASUREMENT_AMBIGUOUS_TIE_BAND_ALIASING,
                BRANCH_MEASUREMENT_AMBIGUOUS_NO_BRANCH,
            ],
            "same_candidate_set_required": True,
            "support_order_seeds": list(ORACLE_SCREEN_CONTRAST_SEEDS),
            "n20_screen_rows": ORACLE_SCREEN_N20_ROWS,
            "required_max_sampled_candidates": budget,
            "qacc_kernelized": False,
            "device_residency_not_hot_loop_residency": True,
        },
        "hash_gate_policy": default_hash_gate_policy(),
        "compact_instrumentation_only": True,
        "raw_per_proposal_arrays_included": False,
        "artifact_policy": {
            "compact_json_ndjson_only": True,
            "raw_per_proposal_arrays": False,
            "pt_writes_allowed": False,
        },
        "non_claims": [
            "author-only credit-ranking pivot launch bundle",
            "same candidate set generated once per seed and evaluated under learner-score diagnostics",
            "deterministic hash remains a null/control distribution only",
            "no GPU launch from this packet-authoring step",
            "no resource lane acquired by this packet",
            "no .pt mutation or checkpoint promotion",
            "no readiness row flip",
            "no carrier or full-sub2 runtime claim",
            "qacc_vote_select_apply_update remains CPU-reference/default-off, not kernelized",
        ],
    }
    validate_credit_ranking_pivot_measurement_launch_bundle(packet)
    return packet


def build_activation_credit_measurement_packet(
    *,
    parent_path: str | Path,
    parent_sha256: str,
    launch_gate_id: str | None = None,
) -> dict[str, Any]:
    budget = PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES
    packet = {
        "schema_version": OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION,
        "packet_kind": ACTIVATION_CREDIT_MEASUREMENT_PACKET_KIND,
        "target_name": ACTIVATION_CREDIT_MEASUREMENT_PACKET_KIND,
        "artifact_role": "activation_credit_measurement_author_packet",
        "diagnostic_class": DIAGNOSTIC_CLASS_PRE_FULL_STACK,
        "pre_full_stack_diagnostic": True,
        "author_only": True,
        "commands_executed": False,
        "gpu_launched": False,
        "launch_gate_id": launch_gate_id,
        "pt_mutated": False,
        "readiness_claim": False,
        "full_sub2_claim": False,
        "ready_for_main_science": False,
        "carrier_claim": False,
        "checkpoint_written": False,
        "optimizer_credit_state_row_flip": False,
        "optimizer_credit_state_science_dependent": True,
        "branch_result": None,
        "qacc_kernelized": False,
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "eligible_scope": ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE,
        "arms": default_oracle_screen_arms(),
        "same_candidate_set_required": True,
        "seed_order_contract": default_oracle_screen_seed_order_contract(),
        "oracle_feasibility_budget": default_oracle_feasibility_budget_for(
            max_sampled_candidates=budget
        ),
        "oracle_non_persistence_contract": default_oracle_non_persistence_contract(),
        "compact_summary_schema": default_activation_credit_compact_summary_schema(),
        "measurement_contract": default_activation_credit_measurement_contract(),
        "scale_smoke_required_before_full_eval": True,
        "scale_smoke_launch_bundle_packet_kind": (
            ACTIVATION_CREDIT_SCALE_SMOKE_LAUNCH_BUNDLE_PACKET_KIND
        ),
        "fresh_confirmation_seed_required_for_persistent_followup": (
            ACTIVATION_CREDIT_FRESH_CONFIRMATION_SEED
        ),
        "artifact_policy": {
            "compact_json_ndjson_only": True,
            "raw_per_proposal_arrays": False,
            "pt_writes_allowed": False,
            "oracle_artifact_path": None,
        },
        "non_claims": [
            "packet scaffold only; no measurement execution",
            "no GPU launch from packet authoring",
            "no .pt mutation or checkpoint promotion",
            "measurement keeps candidate generation fixed and target tie-band fixed",
            "device-resident activation-credit path only; legacy background path must remain explicitly non-hot-loop if unchanged",
            "deterministic hash remains null/control only",
            "no readiness row flip",
            "no carrier or full-sub2 runtime claim",
            "optimizer_credit_state remains science-dependent",
            "no persistent eligibility build before fresh seed 71 confirmation",
            "full activation-credit read remains blocked on the de-risk smoke receipt",
        ],
    }
    validate_activation_credit_measurement_packet(packet)
    return packet


def _build_activation_credit_measurement_command_record(
    *,
    repo_root: str | Path,
    run_root: str | Path,
    parent_path: str | Path,
    parent_sha256: str,
    support_order_seed: int,
    device: str,
    phase_timeout_seconds: int | float,
    total_timeout_seconds: int | float,
    max_silent_phase_seconds: int | float,
) -> dict[str, Any]:
    budget = PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES
    seed_label = _support_order_seed_label(int(support_order_seed))
    scratch_root = _path_join(run_root, seed_label)
    receipt_path = _path_join(scratch_root, "receipt.json")
    stdout_path = _path_join(scratch_root, "stdout.ndjson")
    stderr_path = _path_join(scratch_root, "stderr.log")
    argv = [
        "python3",
        "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--oracle-screen-mode",
        "activation_credit_measurement",
        "--phase",
        f"activation-credit-measurement-n20-{seed_label}",
        "--device",
        str(device),
        "--parent",
        str(parent_path),
        "--parent-sha256",
        str(parent_sha256),
        "--scratch-root",
        scratch_root,
        "--curriculum-seed",
        str(STEP6_CURRICULUM_SEED),
        "--support-order-seed",
        str(int(support_order_seed)),
        "--oracle-screen-max-sampled-candidates",
        str(budget),
        "--batch-size",
        str(ACTIVATION_CREDIT_MEASUREMENT_BATCH_SIZE),
        "--eligible-scope",
        ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE,
        "--steps",
        "1",
        "--max-steps-hard",
        "1",
        "--max-abs-per-tensor",
        str(STEP3_BASELINE_MAX_ABS_PER_TENSOR),
        "--emit-progress",
        "--phase-timeout-seconds",
        str(phase_timeout_seconds),
        "--total-timeout-seconds",
        str(total_timeout_seconds),
        "--max-silent-phase-seconds",
        str(max_silent_phase_seconds),
    ]
    return {
        "mode": "activation_credit_measurement_n20",
        "phase_role": "activation_credit_measurement",
        "support_order_seed": int(support_order_seed),
        "seed_label": seed_label,
        "oracle_screen_mode": "activation_credit_measurement",
        "screen_rows": ACTIVATION_CREDIT_MEASUREMENT_BATCH_SIZE,
        "n_rows": ACTIVATION_CREDIT_MEASUREMENT_BATCH_SIZE,
        "steps_requested": 1,
        "steps_source": "fixed_single_support_batch_activation_credit_measurement",
        "batch_size": ACTIVATION_CREDIT_MEASUREMENT_BATCH_SIZE,
        "eligible_scope": ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE,
        "curriculum_seed": STEP6_CURRICULUM_SEED,
        "same_candidate_set_required": True,
        "max_sampled_candidates": budget,
        "oracle_max_seconds": oracle_screen_budget_max_seconds(budget),
        "max_abs_per_tensor": STEP3_BASELINE_MAX_ABS_PER_TENSOR,
        "fraction_per_tensor": STEP3_FRACTION_PER_TENSOR,
        "global_cap_contract": "off",
        "capture_device_mode": ACTIVATION_CREDIT_CAPTURE_DEVICE_MODE,
        "grad_proxy_compute_mode": ACTIVATION_CREDIT_GRAD_PROXY_COMPUTE_MODE,
        "no_extra_response_label_mask": True,
        "fp_proxy_transient_eval_only": True,
        "cwd": str(repo_root),
        "env": {
            "HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE": "1",
            "HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH": "1",
            ACTIVATION_CREDIT_STDOUT_PATH_ENV: stdout_path,
            ACTIVATION_CREDIT_STDERR_PATH_ENV: stderr_path,
        },
        "argv": argv,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "receipt_path": receipt_path,
        "scratch_root": scratch_root,
        "enabled_if": "fixed contrast seeds 43 and 29 only; same candidate set once per seed; smoke receipt must clear first",
        "expected_exit_policy": "exit_0_required_else_stop_no_retry_no_verdict",
    }


def build_activation_credit_measurement_launch_bundle(
    *,
    parent_path: str | Path,
    parent_sha256: str,
    repo_root: str | Path,
    run_root: str | Path,
    device: str = "cuda:0",
    launch_gate_id: str | None = None,
    symbolic_resource_lane: str = "gpu:0",
    phase_timeout_seconds: int | float = 1800,
    total_timeout_seconds: int | float = 7200,
    max_silent_phase_seconds: int | float = 300,
) -> dict[str, Any]:
    budget = PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES
    science_contract = packet_without_runtime_results(
        build_activation_credit_measurement_packet(
            parent_path=parent_path,
            parent_sha256=parent_sha256,
            launch_gate_id=None,
        ),
    )
    commands = [
        _build_activation_credit_measurement_command_record(
            repo_root=repo_root,
            run_root=run_root,
            parent_path=parent_path,
            parent_sha256=parent_sha256,
            support_order_seed=int(seed),
            device=device,
            phase_timeout_seconds=phase_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
            max_silent_phase_seconds=max_silent_phase_seconds,
        )
        for seed in ORACLE_SCREEN_CONTRAST_SEEDS
    ]
    packet = {
        "schema_version": OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION,
        "packet_kind": ACTIVATION_CREDIT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND,
        "target_name": ACTIVATION_CREDIT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND,
        "artifact_role": "activation_credit_measurement_launch_bundle_author_packet",
        "diagnostic_class": DIAGNOSTIC_CLASS_PRE_FULL_STACK,
        "pre_full_stack_diagnostic": True,
        "author_only": True,
        "commands_executed": False,
        "gpu_launched": False,
        "launch_gate_id": launch_gate_id,
        "pt_mutated": False,
        "readiness_claim": False,
        "full_sub2_claim": False,
        "ready_for_main_science": False,
        "carrier_claim": False,
        "checkpoint_written": False,
        "optimizer_credit_state_row_flip": False,
        "optimizer_credit_state_science_dependent": True,
        "branch_result": None,
        "qacc_kernelized": False,
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "eligible_scope": ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE,
        "science_contract": science_contract,
        "screen_rows": ACTIVATION_CREDIT_MEASUREMENT_BATCH_SIZE,
        "curriculum_seed": STEP6_CURRICULUM_SEED,
        "support_order_seeds": list(ORACLE_SCREEN_CONTRAST_SEEDS),
        "same_candidate_set_required": True,
        "oracle_feasibility_budget": default_oracle_feasibility_budget_for(
            max_sampled_candidates=budget
        ),
        "oracle_non_persistence_contract": default_oracle_non_persistence_contract(),
        "compact_summary_schema": default_activation_credit_compact_summary_schema(),
        "measurement_contract": default_activation_credit_measurement_contract(),
        "scale_smoke_required_before_full_eval": True,
        "scale_smoke_launch_bundle_packet_kind": (
            ACTIVATION_CREDIT_SCALE_SMOKE_LAUNCH_BUNDLE_PACKET_KIND
        ),
        "commands": commands,
        "resource_lane": default_resource_lane_contract(
            symbolic_lane=symbolic_resource_lane,
        ),
        "watcher_audit_bundle": default_activation_credit_watcher_bundle(),
        "phase_budgets": default_activation_credit_phase_budgets(smoke=False),
        "terminal_criteria": default_activation_credit_measurement_terminal_criteria(),
        "hash_gate_policy": default_hash_gate_policy(),
        "compact_instrumentation_only": True,
        "raw_per_proposal_arrays_included": False,
        "artifact_policy": {
            "compact_json_ndjson_only": True,
            "raw_per_proposal_arrays": False,
            "pt_writes_allowed": False,
        },
        "non_claims": [
            "author-only activation-credit launch bundle",
            "same candidate set generated once per seed and evaluated inside the fixed target tie band",
            "deterministic hash remains a null/control distribution only",
            "legacy candidate generation may remain fixed-background reference and must not be described as GPU-hot-loop if unchanged",
            "new activation-credit grad_proxy path must stay device-resident",
            "no GPU launch from this packet-authoring step",
            "no resource lane acquired by this packet",
            "no .pt mutation or checkpoint promotion",
            "no readiness row flip",
            "no carrier or full-sub2 runtime claim",
            "qacc_vote_select_apply_update remains CPU-reference/default-off, not kernelized",
            "no persistent eligibility build before fresh seed 71 confirmation",
            "full activation-credit read remains blocked on the de-risk smoke receipt",
        ],
    }
    validate_activation_credit_measurement_launch_bundle(packet)
    return packet


def _build_activation_credit_scale_smoke_command_record(
    *,
    repo_root: str | Path,
    run_root: str | Path,
    parent_path: str | Path,
    parent_sha256: str,
    support_order_seed: int,
    device: str,
    phase_timeout_seconds: int | float,
    total_timeout_seconds: int | float,
    max_silent_phase_seconds: int | float,
) -> dict[str, Any]:
    budget = ACTIVATION_CREDIT_SMOKE_MAX_SAMPLED_CANDIDATES
    seed_label = _support_order_seed_label(int(support_order_seed))
    scratch_root = _path_join(run_root, seed_label)
    receipt_path = _path_join(scratch_root, "receipt.json")
    stdout_path = _path_join(scratch_root, "stdout.ndjson")
    stderr_path = _path_join(scratch_root, "stderr.log")
    argv = [
        "python3",
        "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--oracle-screen-mode",
        "activation_credit_scale_smoke",
        "--phase",
        f"activation-credit-scale-smoke-b4-k8-{seed_label}",
        "--device",
        str(device),
        "--parent",
        str(parent_path),
        "--parent-sha256",
        str(parent_sha256),
        "--scratch-root",
        scratch_root,
        "--curriculum-seed",
        str(STEP6_CURRICULUM_SEED),
        "--support-order-seed",
        str(int(support_order_seed)),
        "--oracle-screen-max-sampled-candidates",
        str(budget),
        "--batch-size",
        str(ACTIVATION_CREDIT_SMOKE_BATCH_SIZE),
        "--eligible-scope",
        ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE,
        "--steps",
        "1",
        "--max-steps-hard",
        "1",
        "--max-abs-per-tensor",
        str(STEP3_BASELINE_MAX_ABS_PER_TENSOR),
        "--emit-progress",
        "--phase-timeout-seconds",
        str(phase_timeout_seconds),
        "--total-timeout-seconds",
        str(total_timeout_seconds),
        "--max-silent-phase-seconds",
        str(max_silent_phase_seconds),
    ]
    return {
        "mode": "activation_credit_scale_smoke_b4_k8",
        "phase_role": "activation_credit_scale_smoke",
        "support_order_seed": int(support_order_seed),
        "seed_label": seed_label,
        "oracle_screen_mode": "activation_credit_scale_smoke",
        "screen_rows": ACTIVATION_CREDIT_SMOKE_BATCH_SIZE,
        "n_rows": ACTIVATION_CREDIT_SMOKE_BATCH_SIZE,
        "steps_requested": 1,
        "steps_source": "fixed_single_support_batch_activation_credit_scale_smoke",
        "batch_size": ACTIVATION_CREDIT_SMOKE_BATCH_SIZE,
        "eligible_scope": ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE,
        "curriculum_seed": STEP6_CURRICULUM_SEED,
        "same_candidate_set_required": True,
        "max_sampled_candidates": budget,
        "oracle_max_seconds": oracle_screen_budget_max_seconds(budget),
        "max_abs_per_tensor": STEP3_BASELINE_MAX_ABS_PER_TENSOR,
        "fraction_per_tensor": STEP3_FRACTION_PER_TENSOR,
        "global_cap_contract": "off",
        "capture_device_mode": ACTIVATION_CREDIT_CAPTURE_DEVICE_MODE,
        "grad_proxy_compute_mode": ACTIVATION_CREDIT_GRAD_PROXY_COMPUTE_MODE,
        "no_extra_response_label_mask": True,
        "fp_proxy_transient_eval_only": True,
        "cwd": str(repo_root),
        "env": {
            "HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE": "1",
            "HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH": "1",
            ACTIVATION_CREDIT_STDOUT_PATH_ENV: stdout_path,
            ACTIVATION_CREDIT_STDERR_PATH_ENV: stderr_path,
        },
        "argv": argv,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "receipt_path": receipt_path,
        "scratch_root": scratch_root,
        "enabled_if": "fixed contrast seeds 43 and 29 only; scale-smoke only; no policy verdict",
        "expected_exit_policy": "exit_0_required_else_stop_no_retry_no_verdict",
    }


def build_activation_credit_scale_smoke_launch_bundle(
    *,
    parent_path: str | Path,
    parent_sha256: str,
    repo_root: str | Path,
    run_root: str | Path,
    device: str = "cuda:0",
    launch_gate_id: str | None = None,
    symbolic_resource_lane: str = "gpu:0",
    phase_timeout_seconds: int | float = 1800,
    total_timeout_seconds: int | float = 7200,
    max_silent_phase_seconds: int | float = 300,
) -> dict[str, Any]:
    budget = ACTIVATION_CREDIT_SMOKE_MAX_SAMPLED_CANDIDATES
    commands = [
        _build_activation_credit_scale_smoke_command_record(
            repo_root=repo_root,
            run_root=run_root,
            parent_path=parent_path,
            parent_sha256=parent_sha256,
            support_order_seed=int(seed),
            device=device,
            phase_timeout_seconds=phase_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
            max_silent_phase_seconds=max_silent_phase_seconds,
        )
        for seed in ORACLE_SCREEN_CONTRAST_SEEDS
    ]
    packet = {
        "schema_version": OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION,
        "packet_kind": ACTIVATION_CREDIT_SCALE_SMOKE_LAUNCH_BUNDLE_PACKET_KIND,
        "target_name": ACTIVATION_CREDIT_SCALE_SMOKE_LAUNCH_BUNDLE_PACKET_KIND,
        "artifact_role": "activation_credit_scale_smoke_launch_bundle_author_packet",
        "diagnostic_class": DIAGNOSTIC_CLASS_PRE_FULL_STACK,
        "pre_full_stack_diagnostic": True,
        "author_only": True,
        "commands_executed": False,
        "gpu_launched": False,
        "launch_gate_id": launch_gate_id,
        "pt_mutated": False,
        "readiness_claim": False,
        "full_sub2_claim": False,
        "ready_for_main_science": False,
        "carrier_claim": False,
        "checkpoint_written": False,
        "optimizer_credit_state_row_flip": False,
        "optimizer_credit_state_science_dependent": True,
        "branch_result": None,
        "qacc_kernelized": False,
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "eligible_scope": ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE,
        "screen_rows": ACTIVATION_CREDIT_SMOKE_BATCH_SIZE,
        "curriculum_seed": STEP6_CURRICULUM_SEED,
        "support_order_seeds": list(ORACLE_SCREEN_CONTRAST_SEEDS),
        "same_candidate_set_required": True,
        "oracle_feasibility_budget": default_oracle_feasibility_budget_for(
            max_sampled_candidates=budget
        ),
        "oracle_non_persistence_contract": default_oracle_non_persistence_contract(),
        "compact_summary_schema": default_activation_credit_scale_smoke_compact_summary_schema(),
        "scale_smoke_contract": default_activation_credit_scale_smoke_contract(),
        "commands": commands,
        "resource_lane": default_resource_lane_contract(
            symbolic_lane=symbolic_resource_lane,
        ),
        "watcher_audit_bundle": default_activation_credit_watcher_bundle(),
        "phase_budgets": default_activation_credit_phase_budgets(smoke=True),
        "terminal_criteria": default_activation_credit_scale_smoke_terminal_criteria(),
        "hash_gate_policy": default_hash_gate_policy(),
        "compact_instrumentation_only": True,
        "raw_per_proposal_arrays_included": False,
        "artifact_policy": {
            "compact_json_ndjson_only": True,
            "raw_per_proposal_arrays": False,
            "pt_writes_allowed": False,
        },
        "non_claims": [
            "author-only activation-credit scale-smoke launch bundle",
            "same candidate set generated once per seed and evaluated only for scale/de-risk telemetry",
            "deterministic hash remains a null/control distribution only",
            "legacy candidate generation may remain fixed-background reference and must not be described as GPU-hot-loop if unchanged",
            "new activation-credit grad_proxy path must stay device-resident",
            "no policy verdict or persistent eligibility claim from the smoke",
            "no GPU launch from this packet-authoring step",
            "no resource lane acquired by this packet",
            "no .pt mutation or checkpoint promotion",
            "no readiness row flip",
            "qacc_vote_select_apply_update remains CPU-reference/default-off, not kernelized",
        ],
    }
    validate_activation_credit_scale_smoke_launch_bundle(packet)
    return packet


def build_within_tie_band_discriminator_packet(
    *,
    parent_path: str | Path,
    parent_sha256: str,
    launch_gate_id: str | None = None,
) -> dict[str, Any]:
    budget = PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES
    packet = {
        "schema_version": OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION,
        "packet_kind": WITHIN_TIE_BAND_DISCRIMINATOR_PACKET_KIND,
        "target_name": WITHIN_TIE_BAND_DISCRIMINATOR_PACKET_KIND,
        "artifact_role": "within_tie_band_discriminator_author_packet",
        "diagnostic_class": DIAGNOSTIC_CLASS_PRE_FULL_STACK,
        "pre_full_stack_diagnostic": True,
        "author_only": True,
        "commands_executed": False,
        "gpu_launched": False,
        "launch_gate_id": launch_gate_id,
        "pt_mutated": False,
        "readiness_claim": False,
        "full_sub2_claim": False,
        "ready_for_main_science": False,
        "carrier_claim": False,
        "checkpoint_written": False,
        "optimizer_credit_state_row_flip": False,
        "optimizer_credit_state_science_dependent": True,
        "branch_result": None,
        "qacc_kernelized": False,
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "arms": default_oracle_screen_arms(),
        "same_candidate_set_required": True,
        "seed_order_contract": default_oracle_screen_seed_order_contract(),
        "oracle_feasibility_budget": default_oracle_feasibility_budget_for(
            max_sampled_candidates=budget
        ),
        "oracle_non_persistence_contract": default_oracle_non_persistence_contract(),
        "compact_summary_schema": default_within_tie_band_discriminator_compact_summary_schema(),
        "measurement_contract": default_within_tie_band_discriminator_measurement_contract(),
        "artifact_policy": {
            "compact_json_ndjson_only": True,
            "raw_per_proposal_arrays": False,
            "pt_writes_allowed": False,
            "oracle_artifact_path": None,
        },
        "non_claims": [
            "packet scaffold only; no measurement execution",
            "no GPU launch from packet authoring",
            "no .pt mutation or checkpoint promotion",
            "measurement keeps candidate generation fixed and target tie-band fixed",
            "deterministic hash remains null/control only",
            "no readiness row flip",
            "no carrier or full-sub2 runtime claim",
            "optimizer_credit_state remains science-dependent",
            "no stage-b apply-magnitude or budget-64 expansion on this slice",
        ],
    }
    validate_within_tie_band_discriminator_packet(packet)
    return packet


def _build_within_tie_band_discriminator_command_record(
    *,
    repo_root: str | Path,
    run_root: str | Path,
    parent_path: str | Path,
    parent_sha256: str,
    support_order_seed: int,
    device: str,
    phase_timeout_seconds: int | float,
    total_timeout_seconds: int | float,
    max_silent_phase_seconds: int | float,
) -> dict[str, Any]:
    budget = PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES
    seed_label = _support_order_seed_label(int(support_order_seed))
    scratch_root = _path_join(run_root, seed_label)
    receipt_path = _path_join(scratch_root, "receipt.json")
    stdout_path = _path_join(scratch_root, "stdout.ndjson")
    stderr_path = _path_join(scratch_root, "stderr.log")
    argv = [
        "python3",
        "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--oracle-screen-mode",
        "within_tie_band_discriminator",
        "--phase",
        f"within-tie-band-discriminator-n20-{seed_label}",
        "--device",
        str(device),
        "--parent",
        str(parent_path),
        "--parent-sha256",
        str(parent_sha256),
        "--scratch-root",
        scratch_root,
        "--curriculum-seed",
        str(STEP6_CURRICULUM_SEED),
        "--support-order-seed",
        str(int(support_order_seed)),
        "--oracle-screen-max-sampled-candidates",
        str(budget),
        "--batch-size",
        str(ORACLE_SCREEN_N20_ROWS),
        "--steps",
        "1",
        "--max-steps-hard",
        "1",
        "--max-abs-per-tensor",
        str(STEP3_BASELINE_MAX_ABS_PER_TENSOR),
        "--emit-progress",
        "--phase-timeout-seconds",
        str(phase_timeout_seconds),
        "--total-timeout-seconds",
        str(total_timeout_seconds),
        "--max-silent-phase-seconds",
        str(max_silent_phase_seconds),
    ]
    return {
        "mode": "within_tie_band_discriminator_n20",
        "phase_role": "within_tie_band_discriminator",
        "support_order_seed": int(support_order_seed),
        "seed_label": seed_label,
        "oracle_screen_mode": "within_tie_band_discriminator",
        "screen_rows": ORACLE_SCREEN_N20_ROWS,
        "n_rows": ORACLE_SCREEN_N20_ROWS,
        "steps_requested": 1,
        "steps_source": "fixed_single_support_batch_within_tie_band_discriminator",
        "batch_size": ORACLE_SCREEN_N20_ROWS,
        "curriculum_seed": STEP6_CURRICULUM_SEED,
        "same_candidate_set_required": True,
        "max_sampled_candidates": budget,
        "oracle_max_seconds": oracle_screen_budget_max_seconds(budget),
        "max_abs_per_tensor": STEP3_BASELINE_MAX_ABS_PER_TENSOR,
        "fraction_per_tensor": STEP3_FRACTION_PER_TENSOR,
        "global_cap_contract": "off",
        "cwd": str(repo_root),
        "env": {
            "HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE": "1",
            "HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH": "1",
        },
        "argv": argv,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "receipt_path": receipt_path,
        "scratch_root": scratch_root,
        "enabled_if": "fixed contrast seeds 43 and 29 only; same candidate set once per seed",
        "expected_exit_policy": "exit_0_required_else_stop_no_retry_no_verdict",
    }


def build_within_tie_band_discriminator_launch_bundle(
    *,
    parent_path: str | Path,
    parent_sha256: str,
    repo_root: str | Path,
    run_root: str | Path,
    device: str = "cuda:0",
    launch_gate_id: str | None = None,
    symbolic_resource_lane: str = "gpu:0",
    phase_timeout_seconds: int | float = 1800,
    total_timeout_seconds: int | float = 7200,
    max_silent_phase_seconds: int | float = 300,
) -> dict[str, Any]:
    budget = PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES
    science_contract = packet_without_runtime_results(
        build_within_tie_band_discriminator_packet(
            parent_path=parent_path,
            parent_sha256=parent_sha256,
            launch_gate_id=None,
        ),
    )
    commands = [
        _build_within_tie_band_discriminator_command_record(
            repo_root=repo_root,
            run_root=run_root,
            parent_path=parent_path,
            parent_sha256=parent_sha256,
            support_order_seed=int(seed),
            device=device,
            phase_timeout_seconds=phase_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
            max_silent_phase_seconds=max_silent_phase_seconds,
        )
        for seed in ORACLE_SCREEN_CONTRAST_SEEDS
    ]
    packet = {
        "schema_version": OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION,
        "packet_kind": WITHIN_TIE_BAND_DISCRIMINATOR_LAUNCH_BUNDLE_PACKET_KIND,
        "target_name": WITHIN_TIE_BAND_DISCRIMINATOR_LAUNCH_BUNDLE_PACKET_KIND,
        "artifact_role": "within_tie_band_discriminator_launch_bundle_author_packet",
        "diagnostic_class": DIAGNOSTIC_CLASS_PRE_FULL_STACK,
        "pre_full_stack_diagnostic": True,
        "author_only": True,
        "commands_executed": False,
        "gpu_launched": False,
        "launch_gate_id": launch_gate_id,
        "pt_mutated": False,
        "readiness_claim": False,
        "full_sub2_claim": False,
        "ready_for_main_science": False,
        "carrier_claim": False,
        "checkpoint_written": False,
        "optimizer_credit_state_row_flip": False,
        "optimizer_credit_state_science_dependent": True,
        "branch_result": None,
        "qacc_kernelized": False,
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "science_contract": science_contract,
        "screen_rows": ORACLE_SCREEN_N20_ROWS,
        "curriculum_seed": STEP6_CURRICULUM_SEED,
        "support_order_seeds": list(ORACLE_SCREEN_CONTRAST_SEEDS),
        "same_candidate_set_required": True,
        "oracle_feasibility_budget": default_oracle_feasibility_budget_for(
            max_sampled_candidates=budget
        ),
        "oracle_non_persistence_contract": default_oracle_non_persistence_contract(),
        "compact_summary_schema": default_within_tie_band_discriminator_compact_summary_schema(),
        "measurement_contract": default_within_tie_band_discriminator_measurement_contract(),
        "commands": commands,
        "resource_lane": default_resource_lane_contract(
            symbolic_lane=symbolic_resource_lane,
        ),
        "watcher_audit_bundle": default_watcher_bundle(),
        "phase_budgets": default_phase_budgets(),
        "terminal_criteria": {
            **default_terminal_criteria(),
            "branch_classifier": list(WITHIN_TIE_BAND_DISCRIMINATOR_BRANCHES),
            "same_candidate_set_required": True,
            "support_order_seeds": list(ORACLE_SCREEN_CONTRAST_SEEDS),
            "n20_screen_rows": ORACLE_SCREEN_N20_ROWS,
            "required_max_sampled_candidates": budget,
            "qacc_kernelized": False,
            "device_residency_not_hot_loop_residency": True,
        },
        "hash_gate_policy": default_hash_gate_policy(),
        "compact_instrumentation_only": True,
        "raw_per_proposal_arrays_included": False,
        "artifact_policy": {
            "compact_json_ndjson_only": True,
            "raw_per_proposal_arrays": False,
            "pt_writes_allowed": False,
        },
        "non_claims": [
            "author-only within-tie-band discriminator launch bundle",
            "same candidate set generated once per seed and evaluated inside the fixed target tie band",
            "deterministic hash remains a null/control distribution only",
            "no GPU launch from this packet-authoring step",
            "no resource lane acquired by this packet",
            "no .pt mutation or checkpoint promotion",
            "no readiness row flip",
            "no carrier or full-sub2 runtime claim",
            "qacc_vote_select_apply_update remains CPU-reference/default-off, not kernelized",
            "no stage-b apply-magnitude or budget-64 expansion on this slice",
        ],
    }
    validate_within_tie_band_discriminator_launch_bundle(packet)
    return packet


def _walk_items(value: Any) -> Sequence[Any]:
    if isinstance(value, Mapping):
        return list(value.items())
    if isinstance(value, list):
        return list(enumerate(value))
    if isinstance(value, tuple):
        return list(enumerate(value))
    return ()


def _contains_nested_key(value: Any, key_name: str) -> bool:
    if isinstance(value, Mapping):
        if key_name in value:
            return True
        return any(_contains_nested_key(child, key_name) for child in value.values())
    if isinstance(value, list):
        return any(_contains_nested_key(child, key_name) for child in value)
    if isinstance(value, tuple):
        return any(_contains_nested_key(child, key_name) for child in value)
    return False


def _reject_raw_arrays(value: Any, *, path: str = "packet") -> None:
    for key, child in _walk_items(value):
        key_text = str(key)
        lowered = key_text.lower()
        if any(fragment in lowered for fragment in _RAW_ARRAY_KEY_FRAGMENTS):
            if child not in (False, None, "not_included"):
                raise ValueError(f"{path}.{key_text} contains raw per-proposal arrays")
        _reject_raw_arrays(child, path=f"{path}.{key_text}")


def _require_false(packet: Mapping[str, Any], field: str) -> None:
    if bool(packet.get(field, False)):
        raise ValueError(f"{field} must be false for pre_full_stack_diagnostic packet")


def _validate_arms(arms: Sequence[Mapping[str, Any]]) -> None:
    by_id = {str(arm.get("arm_id")): dict(arm) for arm in arms}
    required = {
        ARM_A0_RANK_BUCKET_CURRENT,
        ARM_A1_RANK_BUCKET_ORDER_MATCHED,
        ARM_B_RANK_FREE_SIGN_PRESSURE,
    }
    missing = sorted(required - set(by_id))
    if missing:
        raise ValueError(f"science packet missing required arms: {missing}")
    if by_id[ARM_A0_RANK_BUCKET_CURRENT].get("tie_policy_id") != TIE_POLICY_CURRENT_MARGIN_INDEX:
        raise ValueError("A0 must keep current abs(new_acc)+index ordering")
    a1_policy = by_id[ARM_A1_RANK_BUCKET_ORDER_MATCHED].get("tie_policy_id")
    b_policy = by_id[ARM_B_RANK_FREE_SIGN_PRESSURE].get("tie_policy_id")
    if a1_policy != b_policy or a1_policy != TIE_POLICY_DETERMINISTIC_HASH_MATCHED:
        raise ValueError("A1 and B must share the deterministic ordering-matched tie policy")
    if ARM_INVERTED_SIGN_PRESSURE in by_id:
        if by_id[ARM_INVERTED_SIGN_PRESSURE].get("tie_policy_id") != a1_policy:
            raise ValueError("inverted-sign falsifier must share A1/B tie policy")


def _validate_hash_gate_policy(policy: Mapping[str, Any]) -> None:
    required = set(policy.get("required_sources") or ())
    forbidden_text = " ".join(str(item).lower() for item in policy.get("required_sources") or ())
    if "live_q" in forbidden_text or "post_arm" in forbidden_text:
        raise ValueError("hash gate cannot use live post-arm q mutation as read-only source")
    if "banked_parent_checkpoint" not in required or "banked_initial_q_state" not in required:
        raise ValueError("hash gate requires read-only banked parent and initial q-state sources")
    if not bool(policy.get("live_q_delta_fields_required_separately")):
        raise ValueError("live q deltas must be recorded separately from read-only hash gates")
    if not bool(policy.get("hash_before_after_must_match")):
        raise ValueError("read-only hash gates must assert before/after match")


def validate_optimizer_update_law_science_packet(packet: Mapping[str, Any]) -> None:
    if packet.get("schema_version") != OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION:
        raise ValueError("unsupported optimizer update-law science schema")
    if packet.get("diagnostic_class") != DIAGNOSTIC_CLASS_PRE_FULL_STACK:
        raise ValueError("packet must be pre_full_stack_diagnostic")
    if packet.get("launch_gate_id", "missing") is not None:
        raise ValueError("Step-1 dry-run packet must carry launch_gate_id=null")
    mode = str(packet.get("mode", ""))
    if mode not in SCIENCE_MODE_ROWS:
        raise ValueError(f"unsupported science mode {mode!r}")
    if int(packet.get("n_rows", -1)) != SCIENCE_MODE_ROWS[mode]:
        raise ValueError(f"{mode} must use N={SCIENCE_MODE_ROWS[mode]}")
    for field in _READINESS_FORBIDDEN_TRUE_FIELDS:
        _require_false(packet, field)
    if packet.get("aux_vote_law") != FIXED_RANK_BUCKET_NON_TARGET_AUX:
        raise ValueError("aux_vote_law must be fixed_rank_bucket_non_target_aux")
    _validate_arms(packet.get("arms") or ())
    gate = packet.get("control_parity_gate") or {}
    if float(gate.get("min_inclusive", -1.0)) != CONTROL_PARITY_FRACTION_MIN:
        raise ValueError("control parity min tolerance must be pinned to 0.15")
    if float(gate.get("max_inclusive", -1.0)) != CONTROL_PARITY_FRACTION_MAX:
        raise ValueError("control parity max tolerance must be pinned to 0.45")
    if not bool(gate.get("qualitative_prior_null_signature_required")):
        raise ValueError("qualitative prior-null signature is required")
    if not bool(gate.get("requires_current_improves_vs_baseline")):
        raise ValueError("control parity requires current improves vs baseline")
    if not bool(gate.get("requires_random_matches_or_beats_current")):
        raise ValueError("control parity requires random matches or beats current")
    _validate_hash_gate_policy(packet.get("hash_gate_policy") or {})
    verdict = packet.get("verdict_rule") or {}
    if verdict.get("positive_branch") != BRANCH_RANK_FREE_POSITIVE:
        raise ValueError("positive branch must be rank_free_sign_pressure_positive")
    if verdict.get("B_beats_A0_but_not_A1_branch") != BRANCH_TIE_POLICY_OR_OVERUPDATE:
        raise ValueError("B beats A0 but not A1 must classify tie/order limited")
    if verdict.get("A1_missing_policy") != TIE_POLICY_SCREEN_ONLY_IF_A1_MISSING:
        raise ValueError("A1 missing must demote to screen only")
    _reject_raw_arrays(packet)


def classify_optimizer_update_law_branch(
    *,
    mode: str,
    control_parity_pass: bool,
    b_beats_a0: bool,
    b_beats_a1: bool,
    b_beats_falsifiers: bool,
    a1_helped: bool = False,
    current_rank_bucket_beats_b: bool = False,
    inverted_beats_candidate: bool = False,
    any_arm_improves: bool = True,
    prior_null_setup_verified: bool = True,
) -> str | None:
    if str(mode) == SCIENCE_MODE_PRETERMINAL_SCREEN:
        return None
    if not bool(prior_null_setup_verified):
        return BRANCH_INSUFFICIENT_SEPARATION
    if not bool(control_parity_pass):
        return BRANCH_INSUFFICIENT_SEPARATION
    if bool(inverted_beats_candidate):
        return BRANCH_DIRECTION_PROJECTION_WRONG
    if bool(a1_helped) or (bool(b_beats_a0) and not bool(b_beats_a1)):
        return BRANCH_TIE_POLICY_OR_OVERUPDATE
    if bool(b_beats_a0) and bool(b_beats_a1) and bool(b_beats_falsifiers):
        return BRANCH_RANK_FREE_POSITIVE
    if bool(current_rank_bucket_beats_b):
        return BRANCH_RANKING_STILL_REQUIRED
    if not bool(any_arm_improves):
        return BRANCH_CREDIT_SOURCE_NOT_SUFFICIENT
    return BRANCH_INSUFFICIENT_SEPARATION


def classify_step3_power_floor(
    *,
    non_inverted_strict_exact_counts: Mapping[str, int] | Sequence[int],
    paired_loss_ci_excludes_zero: Mapping[str, bool] | Sequence[str] | bool = False,
    paired_loss_winner: Mapping[str, str] | None = None,
    separated_comparison: str | None = None,
    separated_direction: str | None = None,
) -> str:
    """Classify whether Step-3 has enough measurement power to enter phase 2."""
    if isinstance(non_inverted_strict_exact_counts, Mapping):
        counts = [int(value) for value in non_inverted_strict_exact_counts.values()]
    else:
        counts = [int(value) for value in non_inverted_strict_exact_counts]
    if counts and max(counts) >= STEP3_STRICT_EXACT_FLOOR_COUNT:
        return BRANCH_MEASUREMENT_POWERED

    separated: set[str] = set()
    if isinstance(paired_loss_ci_excludes_zero, Mapping):
        separated = {str(name) for name, excludes in paired_loss_ci_excludes_zero.items() if bool(excludes)}
    elif isinstance(paired_loss_ci_excludes_zero, bool):
        if paired_loss_ci_excludes_zero:
            separated.add(str(separated_comparison or "unspecified"))
    else:
        separated = {str(name) for name in paired_loss_ci_excludes_zero}
    if not separated:
        return BRANCH_MEASUREMENT_UNDERPOWERED

    winner = str(
        separated_direction
        or (paired_loss_winner or {}).get("B_minus_A0")
        or "",
    ).lower()
    a0_favored = winner in {"a0", "a0_lower_loss", "favors_a0", "favor_a0"}
    if separated == {"B_minus_A0"} and a0_favored:
        return BRANCH_POWERED_NEGATIVE_OR_LOSS_ONLY
    return BRANCH_MEASUREMENT_LOSS_POWERED


def _validate_author_only_fields(
    packet: Mapping[str, Any],
    *,
    expected_packet_kind: str = STEP2_LAUNCH_BUNDLE_PACKET_KIND,
    label: str = "author-only launch bundle",
) -> None:
    required_values = {
        "packet_kind": expected_packet_kind,
        "author_only": True,
        "commands_executed": False,
        "gpu_launched": False,
        "launch_gate_id": None,
        "pt_mutated": False,
        "readiness_claim": False,
        "branch_result": None,
    }
    for field, expected in required_values.items():
        if packet.get(field) != expected:
            raise ValueError(f"{field} must be {expected!r} for {label}")
    for field in _AUTHOR_PACKET_RUNTIME_RESULT_FIELDS:
        if field in packet:
            raise ValueError(f"{label} rejects runtime field {field}")
    if bool(packet.get("full_sub2_claim", False)):
        raise ValueError(f"full_sub2_claim must be false for {label}")
    if bool(packet.get("checkpoint_written", False)):
        raise ValueError(f"checkpoint_written must be false for {label}")
    if bool(packet.get("optimizer_credit_state_row_flip", False)):
        raise ValueError(f"optimizer_credit_state_row_flip must be false for {label}")


def _validate_step3_power_floor(floor: Mapping[str, Any]) -> None:
    strict = floor.get("strict_exact_floor") or {}
    if not bool(strict.get("non_inverted_only")):
        raise ValueError("Step-3 strict_exact floor must be non_inverted_only")
    if int(strict.get("count", -1)) != STEP3_STRICT_EXACT_FLOOR_COUNT:
        raise ValueError("Step-3 strict_exact floor count must be 10")
    if int(strict.get("total", -1)) != STEP3_STRICT_EXACT_FLOOR_TOTAL:
        raise ValueError("Step-3 strict_exact floor total must be 90")
    if not bool(strict.get("not_an_acquisition_claim")):
        raise ValueError("Step-3 strict_exact floor must disclaim acquisition")
    paired_loss = floor.get("paired_loss_floor") or {}
    if paired_loss.get("bootstrap_ci") != "95%":
        raise ValueError("Step-3 paired-loss floor must use 95% bootstrap CI")
    if set(paired_loss.get("comparisons") or ()) != {"A1_minus_B", "B_minus_A0"}:
        raise ValueError("Step-3 paired-loss floor must cover A1_minus_B and B_minus_A0")
    if not bool(paired_loss.get("ci_must_exclude_zero")):
        raise ValueError("Step-3 paired-loss floor must require CI excluding zero")
    classifications = floor.get("classifications") or {}
    expected = {
        "no_floor": BRANCH_MEASUREMENT_UNDERPOWERED,
        "strict_exact_floor": BRANCH_MEASUREMENT_POWERED,
        "favorable_paired_loss_ci": BRANCH_MEASUREMENT_LOSS_POWERED,
        "strict_below_floor_and_only_b_minus_a0_loss_favors_a0": BRANCH_POWERED_NEGATIVE_OR_LOSS_ONLY,
    }
    for field, expected_value in expected.items():
        if classifications.get(field) != expected_value:
            raise ValueError(f"Step-3 power floor missing {field}={expected_value!r}")
    unlock_rule = str(floor.get("phase2_unlock_rule", ""))
    if "B-A0-loss-only" not in unlock_rule or "no acquisition claim" not in unlock_rule:
        raise ValueError("Step-3 power floor must block B-A0-loss-only acquisition-capable phase2")


def _validate_step3_effective_cap(audit: Mapping[str, Any]) -> None:
    required = {
        "baseline_max_abs_per_tensor": STEP3_BASELINE_MAX_ABS_PER_TENSOR,
        "cap_max_abs_per_tensor": STEP3_CAP_MAX_ABS_PER_TENSOR,
        "fraction_per_tensor": STEP3_FRACTION_PER_TENSOR,
        "if_cap_effective_false": BRANCH_CAP_NOOP,
    }
    for field, expected in required.items():
        if audit.get(field) != expected:
            raise ValueError(f"Step-3 effective_cap must carry {field}={expected!r}")
    if int(audit.get("tensor_count_reduced", 0)) <= 0:
        raise ValueError("Step-3 effective_cap tensor_count_reduced must be positive")
    baseline = int(audit.get("total_allowed_flips_baseline", -1))
    cap = int(audit.get("total_allowed_flips_cap", -1))
    if baseline <= 0 or cap <= 0 or cap >= baseline:
        raise ValueError("Step-3 effective_cap total_allowed_flips_cap must be below baseline")
    if not bool(audit.get("cap_effective")):
        raise ValueError("Step-3 effective_cap cap_effective must be true; otherwise classify cap_noop")


def _validate_step3_command_record(command: Mapping[str, Any]) -> None:
    missing = [field for field in _COMMAND_REQUIRED_FIELDS if field not in command]
    if missing:
        raise ValueError(f"Step-3 command record missing required fields: {missing}")
    mode = str(command.get("mode"))
    arm_id = str(command.get("arm_id"))
    science_arm = str(command.get("science_arm"))
    if mode not in STEP3_PHASE_STEPS:
        raise ValueError(f"Step-3 command record has unsupported mode {mode!r}")
    if arm_id not in set(SCIENCE_ARM_IDS) | {ARM_B_CAP_MAX_ABS_1024}:
        raise ValueError(f"Step-3 command record has unsupported arm_id {arm_id!r}")
    if science_arm not in SCIENCE_ARM_IDS:
        raise ValueError(f"Step-3 command record has unsupported science_arm {science_arm!r}")
    if arm_id == ARM_B_CAP_MAX_ABS_1024:
        if science_arm != ARM_B_RANK_FREE_SIGN_PRESSURE:
            raise ValueError("B_cap command must execute B rank-free science arm")
        expected_max_abs = STEP3_CAP_MAX_ABS_PER_TENSOR
    else:
        if science_arm != arm_id:
            raise ValueError("Step-3 non-cap command science_arm must match arm_id")
        expected_max_abs = STEP3_BASELINE_MAX_ABS_PER_TENSOR
    if mode in STEP3_POWER_PHASES and arm_id == ARM_B_CAP_MAX_ABS_1024:
        raise ValueError("Step-3 power phase must not include B_cap arm")
    if int(command.get("max_abs_per_tensor", -1)) != expected_max_abs:
        raise ValueError(f"Step-3 command max_abs_per_tensor must be {expected_max_abs}")
    if float(command.get("fraction_per_tensor", -1.0)) != STEP3_FRACTION_PER_TENSOR:
        raise ValueError("Step-3 command fraction_per_tensor must be 1.0")
    if command.get("global_cap_contract") != "off":
        raise ValueError("Step-3 command global cap must stay off")
    n_rows = int(command.get("n_rows", -1))
    steps_requested = int(command.get("steps_requested", -2))
    expected_steps = int(STEP3_PHASE_STEPS[mode])
    if n_rows != expected_steps or steps_requested != expected_steps:
        raise ValueError("Step-3 command steps_requested must match phase steps")
    if command.get("steps_source") != "STEP3_PHASE_STEPS[mode]":
        raise ValueError("Step-3 command steps_source must document STEP3_PHASE_STEPS")
    env = command.get("env")
    if not isinstance(env, Mapping):
        raise ValueError("Step-3 command env must be a mapping")
    if env.get("HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE") != "1":
        raise ValueError("Step-3 command env missing HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE=1")
    if env.get("HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH") != "1":
        raise ValueError("Step-3 command env missing HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH=1")
    argv = command.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("Step-3 command argv must be a list[str]")
    required_args = {
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--device",
        "--parent",
        "--parent-sha256",
        "--scratch-root",
        "--steps",
        "--max-steps-hard",
        "--audit-interval",
        "--science-arm",
        "--max-abs-per-tensor",
        "--emit-progress",
    }
    if not required_args.issubset(set(argv)):
        raise ValueError("Step-3 command argv missing required probe launch arguments")
    expected_flag_values = (
        ("--science-arm", science_arm),
        ("--steps", str(expected_steps)),
        ("--max-steps-hard", str(STEP3_MAX_STEPS_HARD)),
        ("--max-abs-per-tensor", str(expected_max_abs)),
    )
    for flag, expected in expected_flag_values:
        try:
            observed = argv[argv.index(flag) + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Step-3 command argv missing {flag} value") from exc
        if observed != expected:
            raise ValueError(f"Step-3 command argv {flag} must be {expected!r}, got {observed!r}")
    try:
        device = argv[argv.index("--device") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("Step-3 command argv missing --device value") from exc
    if not device.startswith("cuda:"):
        raise ValueError("Step-3 command argv --device must target CUDA for launch packet")
    for path_field in ("stdout_path", "stderr_path", "receipt_path", "scratch_root"):
        value = str(command.get(path_field))
        if not value:
            raise ValueError(f"Step-3 command {path_field} must be non-empty")
        if value.endswith(".pt"):
            raise ValueError(f"Step-3 command {path_field} cannot target .pt artifacts")
    if command.get("expected_exit_policy") != "exit_0_required_else_stop_no_retry_no_verdict":
        raise ValueError("Step-3 command expected_exit_policy must fail closed")


def _validate_step4_command_record(command: Mapping[str, Any]) -> None:
    missing = [field for field in _COMMAND_REQUIRED_FIELDS if field not in command]
    if missing:
        raise ValueError(f"Step-4 command record missing required fields: {missing}")
    mode = str(command.get("mode"))
    arm_id = str(command.get("arm_id"))
    science_arm = str(command.get("science_arm"))
    if mode not in STEP4_PHASE_STEPS:
        raise ValueError(f"Step-4 command record has unsupported mode {mode!r}")
    if arm_id not in STEP4_ARM_IDS:
        raise ValueError(f"Step-4 command record has unsupported arm_id {arm_id!r}")
    if science_arm != arm_id or science_arm not in STEP4_ARM_IDS:
        raise ValueError("Step-4 command science_arm must match a Step-4 arm_id")
    if int(command.get("max_abs_per_tensor", -1)) != STEP3_BASELINE_MAX_ABS_PER_TENSOR:
        raise ValueError("Step-4 command max_abs_per_tensor must keep baseline 4096")
    if float(command.get("fraction_per_tensor", -1.0)) != STEP3_FRACTION_PER_TENSOR:
        raise ValueError("Step-4 command fraction_per_tensor must be 1.0")
    if command.get("global_cap_contract") != "off":
        raise ValueError("Step-4 command global cap must stay off")
    n_rows = int(command.get("n_rows", -1))
    steps_requested = int(command.get("steps_requested", -2))
    expected_steps = int(STEP4_PHASE_STEPS[mode])
    if n_rows != expected_steps or steps_requested != expected_steps:
        raise ValueError("Step-4 command steps_requested must match phase steps")
    if command.get("steps_source") != "STEP4_PHASE_STEPS[mode]":
        raise ValueError("Step-4 command steps_source must document STEP4_PHASE_STEPS")
    env = command.get("env")
    if not isinstance(env, Mapping):
        raise ValueError("Step-4 command env must be a mapping")
    if env.get("HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE") != "1":
        raise ValueError("Step-4 command env missing HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE=1")
    if env.get("HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH") != "1":
        raise ValueError("Step-4 command env missing HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH=1")
    argv = command.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("Step-4 command argv must be a list[str]")
    required_args = {
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--device",
        "--parent",
        "--parent-sha256",
        "--scratch-root",
        "--steps",
        "--max-steps-hard",
        "--audit-interval",
        "--science-arm",
        "--max-abs-per-tensor",
        "--emit-progress",
    }
    if not required_args.issubset(set(argv)):
        raise ValueError("Step-4 command argv missing required probe launch arguments")
    expected_flag_values = (
        ("--science-arm", science_arm),
        ("--steps", str(expected_steps)),
        ("--max-steps-hard", str(max(STEP4_PHASE_STEPS.values()))),
        ("--max-abs-per-tensor", str(STEP3_BASELINE_MAX_ABS_PER_TENSOR)),
    )
    for flag, expected in expected_flag_values:
        try:
            observed = argv[argv.index(flag) + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Step-4 command argv missing {flag} value") from exc
        if observed != expected:
            raise ValueError(f"Step-4 command argv {flag} must be {expected!r}, got {observed!r}")
    try:
        device = argv[argv.index("--device") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("Step-4 command argv missing --device value") from exc
    if not device.startswith("cuda:"):
        raise ValueError("Step-4 command argv --device must target CUDA for launch packet")
    for path_field in ("stdout_path", "stderr_path", "receipt_path", "scratch_root"):
        value = str(command.get(path_field))
        if not value:
            raise ValueError(f"Step-4 command {path_field} must be non-empty")
        if value.endswith(".pt"):
            raise ValueError(f"Step-4 command {path_field} cannot target .pt artifacts")
    if command.get("expected_exit_policy") != "exit_0_required_else_stop_no_retry_no_verdict":
        raise ValueError("Step-4 command expected_exit_policy must fail closed")


def _validate_command_record(command: Mapping[str, Any]) -> None:
    missing = [field for field in _COMMAND_REQUIRED_FIELDS if field not in command]
    if missing:
        raise ValueError(f"command record missing required fields: {missing}")
    mode = str(command.get("mode"))
    arm_id = str(command.get("arm_id"))
    if mode not in SCIENCE_MODE_ROWS:
        raise ValueError(f"command record has unsupported mode {mode!r}")
    if arm_id not in SCIENCE_ARM_IDS:
        raise ValueError(f"command record has unsupported arm_id {arm_id!r}")
    n_rows = int(command.get("n_rows", -1))
    steps_requested = int(command.get("steps_requested", -2))
    if n_rows != SCIENCE_MODE_ROWS[mode]:
        raise ValueError(f"command n_rows must equal SCIENCE_MODE_ROWS[{mode!r}]")
    if steps_requested != n_rows:
        raise ValueError("command steps_requested must equal n_rows explicitly")
    if command.get("steps_source") != "SCIENCE_MODE_ROWS[mode]":
        raise ValueError("command steps_source must document SCIENCE_MODE_ROWS[mode]")
    env = command.get("env")
    if not isinstance(env, Mapping):
        raise ValueError("command env must be a mapping")
    if env.get("HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE") != "1":
        raise ValueError("command env missing HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE=1")
    if env.get("HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH") != "1":
        raise ValueError("command env missing HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH=1")
    argv = command.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("command argv must be a list[str]")
    required_args = {
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--device",
        "--parent",
        "--parent-sha256",
        "--scratch-root",
        "--steps",
        "--max-steps-hard",
        "--audit-interval",
        "--science-arm",
        "--emit-progress",
    }
    if not required_args.issubset(set(argv)):
        raise ValueError("command argv missing required probe launch arguments")
    expected_flag_values = (
        ("--science-arm", arm_id),
        ("--steps", str(steps_requested)),
        ("--max-steps-hard", str(max(SCIENCE_MODE_ROWS.values()))),
    )
    for flag, expected in expected_flag_values:
        try:
            observed = argv[argv.index(flag) + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"command argv missing {flag} value") from exc
        if observed != expected:
            raise ValueError(f"command argv {flag} must be {expected!r}, got {observed!r}")
    try:
        device = argv[argv.index("--device") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("command argv missing --device value") from exc
    if not device.startswith("cuda:"):
        raise ValueError("command argv --device must target CUDA for launch bundle")
    for path_field in ("stdout_path", "stderr_path", "receipt_path", "scratch_root"):
        value = str(command.get(path_field))
        if not value:
            raise ValueError(f"command {path_field} must be non-empty")
        if value.endswith(".pt"):
            raise ValueError(f"command {path_field} cannot target .pt artifacts")
    if command.get("expected_exit_policy") != "exit_0_required_else_stop_no_retry_no_verdict":
        raise ValueError("command expected_exit_policy must fail closed")


def _validate_resource_lane(resource_lane: Mapping[str, Any]) -> None:
    if not bool(resource_lane.get("resource_lane_required")):
        raise ValueError("resource_lane_required must be true")
    if not str(resource_lane.get("lane_name", "")):
        raise ValueError("resource lane must name a symbolic lane")
    if bool(resource_lane.get("resolved_at_launch")):
        raise ValueError("author packet cannot mark resource lane resolved")
    if not bool(resource_lane.get("acquire_at_launch")):
        raise ValueError("resource lane must be acquired at launch")
    if not bool(resource_lane.get("release_on_terminal_receipt")):
        raise ValueError("resource lane must release on terminal receipt")
    if not bool(resource_lane.get("author_packet_does_not_acquire")):
        raise ValueError("author packet must not acquire resource lane")


def _validate_screen_before_verdict(packet: Mapping[str, Any]) -> None:
    if packet.get("mode_sequence") != [SCIENCE_MODE_PRETERMINAL_SCREEN, SCIENCE_MODE_BRANCH_VERDICT]:
        raise ValueError("mode_sequence must enforce screen before branch verdict")
    dependency = packet.get("screen_before_verdict") or {}
    blocked = dependency.get("verdict_blocked_until") or {}
    if dependency.get("screen_mode") != SCIENCE_MODE_PRETERMINAL_SCREEN:
        raise ValueError("screen_before_verdict missing preterminal screen mode")
    if dependency.get("verdict_mode") != SCIENCE_MODE_BRANCH_VERDICT:
        raise ValueError("screen_before_verdict missing branch verdict mode")
    required_blockers = {
        "screen_terminal_receipt_pass": True,
        "parent_hash_unchanged": True,
        "all_four_arms_completed": True,
        "no_nonfinite": True,
        "no_cuda_oom": True,
        "no_timeout": True,
        "pt_mutated": False,
        "qualitative_control_parity_screen_not": BRANCH_INSUFFICIENT_SEPARATION,
        "prior_null_setup_verified": True,
    }
    for field, expected in required_blockers.items():
        if blocked.get(field) != expected:
            raise ValueError(f"branch verdict missing blocked_until {field}={expected!r}")


def _validate_phase_budgets(phase_budgets: Mapping[str, Any]) -> None:
    missing = [key for key in _PHASE_BUDGET_KEYS if key not in phase_budgets]
    if missing:
        raise ValueError(f"phase_budgets missing required first-milestone budgets: {missing}")
    for key in _PHASE_BUDGET_KEYS:
        budget = phase_budgets[key]
        if int(budget.get("first_milestone_seconds", 0)) <= 0:
            raise ValueError(f"phase budget {key} must have positive first_milestone_seconds")
        if not budget.get("probe_phase_markers"):
            raise ValueError(f"phase budget {key} must name probe phase markers")


def _validate_activation_credit_phase_budgets(
    phase_budgets: Mapping[str, Any],
    *,
    smoke: bool,
) -> None:
    _validate_phase_budgets(phase_budgets)
    expected = default_activation_credit_phase_budgets(smoke=smoke)
    for key, expected_budget in expected.items():
        budget = phase_budgets.get(key)
        if not isinstance(budget, Mapping):
            raise ValueError(f"activation-credit phase budget {key} missing")
        if int(budget.get("first_milestone_seconds", -1)) != int(
            expected_budget["first_milestone_seconds"]
        ):
            raise ValueError(
                f"activation-credit phase budget {key} first_milestone_seconds drifted"
            )
        if list(budget.get("probe_phase_markers") or ()) != list(
            expected_budget["probe_phase_markers"]
        ):
            raise ValueError(
                f"activation-credit phase budget {key} probe_phase_markers drifted"
            )


def _validate_author_hash_gates(packet: Mapping[str, Any]) -> None:
    _validate_hash_gate_policy(packet.get("hash_gate_policy") or {})
    terminal = packet.get("terminal_criteria") or {}
    _validate_hash_gate_policy(terminal.get("hash_gate_policy") or {})
    if not bool(terminal.get("live_q_delta_fields_required_separately")):
        raise ValueError("terminal criteria must keep live-q deltas separate")
    prior_gate = terminal.get("prior_null_setup_gate") or {}
    required_prior_gate = {
        "verified_required_before_parity_band": True,
        "unverified_branch": BRANCH_PRIOR_NULL_SETUP_UNVERIFIED,
        "fallback_branch": BRANCH_INSUFFICIENT_SEPARATION,
        "blocks_control_parity_band": True,
    }
    for field, expected in required_prior_gate.items():
        if prior_gate.get(field) != expected:
            raise ValueError(f"terminal prior-null gate must carry {field}={expected!r}")
    prior = packet.get("prior_verdict_parent_ref") or {}
    if str(prior.get("parent_sha256")) != str(packet.get("parent_sha256")):
        raise ValueError("prior verdict parent sha must match launch parent sha")
    if str(prior.get("parent_path")) != str(packet.get("parent_path")):
        raise ValueError("prior verdict parent path must match launch parent path")
    if str(prior.get("artifact_literal_parent_sha256")) != str(packet.get("parent_sha256")):
        raise ValueError("artifact literal parent sha must match launch parent sha")
    if str(prior.get("artifact_literal_parent_path")) != str(packet.get("parent_path")):
        raise ValueError("artifact literal parent path must match launch parent path")
    expected_prior = {
        "verdict_label": "credit_ranking_uninformative_update_law_pivot",
        "candidate_label": "credit_ranking_uninformative_update_law_pivot",
        "artifact_path": "/tmp/hrm158_shadow_prefix_lane3_hybrid_gpu_n50_trace.tierb.final.json",
        "support_name": "L0c2-K2-addition-120",
        "support_sha": "21c8a2f8c15fd68571407e6d1f215ab045ffc5a2a91e4b5a44b50bcd46b6faf0",
        "row_count": 120,
        "lane": "lane3",
        "acc_mode": "applied_crossing_direction_plus_4bit_residual",
        "vote_fidelity": "dry2",
        "activation_mode": "ternary_group128_codec_from_step0",
        "rank_vote_spec_sha256": "6c109e0482292edf72d3cc4ada6bda0840e67e8dbfac4ad7fd64d353602806a5",
        "vote_mapping_family": "rank_bucketed",
        "artifact_literal_support_sha256": "21c8a2f8c15fd68571407e6d1f215ab045ffc5a2a91e4b5a44b50bcd46b6faf0",
        "artifact_literal_acc_mode": "applied_crossing_direction_plus_4bit_residual",
        "artifact_literal_activation_mode": "ternary_group128_codec_from_step0",
        "artifact_literal_vote_fidelity": "dry2",
        "artifact_literal_rank_vote_spec_sha256": "6c109e0482292edf72d3cc4ada6bda0840e67e8dbfac4ad7fd64d353602806a5",
        "artifact_literal_vote_mapping_family": "rank_bucketed",
        "artifact_literal_lane": "lane3",
        "artifact_literal_random_seed": 17,
        "artifact_literal_step_or_sample_count": 50,
        "support_seed": 17,
        "random_seed": 17,
        "step_or_sample_count": 50,
        "current_beats_competitors_fraction": 0.3,
        "prior_null_setup_verified": False,
        "verification_state": "pending_launch_terminal_receipt",
        "unverified_branch": BRANCH_PRIOR_NULL_SETUP_UNVERIFIED,
        "fallback_branch": BRANCH_INSUFFICIENT_SEPARATION,
        "blocks_control_parity_band": True,
        "verified_at_launch_required": True,
        "control_parity_self_protects": True,
    }
    for field, expected in expected_prior.items():
        if prior.get(field) != expected:
            raise ValueError(f"prior verdict parent ref must carry {field}={expected!r}")


def validate_optimizer_update_law_launch_bundle(packet: Mapping[str, Any]) -> None:
    if packet.get("schema_version") != OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION:
        raise ValueError("unsupported optimizer update-law launch bundle schema")
    if packet.get("diagnostic_class") != DIAGNOSTIC_CLASS_PRE_FULL_STACK:
        raise ValueError("launch bundle must be pre_full_stack_diagnostic")
    _validate_author_only_fields(packet)
    _validate_arms(packet.get("arms") or ())
    _reject_raw_arrays(packet)
    _validate_resource_lane(packet.get("resource_lane") or {})
    _validate_screen_before_verdict(packet)
    _validate_phase_budgets(packet.get("phase_budgets") or {})
    _validate_author_hash_gates(packet)
    terminal = packet.get("terminal_criteria") or {}
    if terminal.get("branch_classifier") != list(OPTIMIZER_UPDATE_LAW_BRANCHES):
        raise ValueError("terminal criteria must carry the preregistered branch classifier")
    if not bool(terminal.get("no_verdict_outside_prereg")):
        raise ValueError("terminal criteria must reject verdicts outside prereg")
    commands = packet.get("commands")
    if not isinstance(commands, list):
        raise ValueError("launch bundle commands must be a list")
    seen = {(str(cmd.get("mode")), str(cmd.get("arm_id"))) for cmd in commands}
    expected = {
        (mode, arm_id)
        for mode in (SCIENCE_MODE_PRETERMINAL_SCREEN, SCIENCE_MODE_BRANCH_VERDICT)
        for arm_id in SCIENCE_ARM_IDS
    }
    if seen != expected:
        raise ValueError("launch bundle must include every arm for screen and branch modes")
    for command in commands:
        _validate_command_record(command)
    artifact_policy = packet.get("artifact_policy") or {}
    if not bool(artifact_policy.get("compact_json_ndjson_only")):
        raise ValueError("artifact policy must require compact JSON/NDJSON")
    if bool(artifact_policy.get("pt_writes_allowed")):
        raise ValueError("artifact policy must reject .pt writes")


def validate_measurement_power_then_trust_region_packet(packet: Mapping[str, Any]) -> None:
    if packet.get("schema_version") != OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION:
        raise ValueError("unsupported optimizer update-law Step-3 packet schema")
    if packet.get("diagnostic_class") != DIAGNOSTIC_CLASS_PRE_FULL_STACK:
        raise ValueError("Step-3 packet must be pre_full_stack_diagnostic")
    _validate_author_only_fields(
        packet,
        expected_packet_kind=STEP3_MEASUREMENT_POWER_TRUST_REGION_PACKET_KIND,
        label="author-only Step-3 packet",
    )
    _validate_arms(packet.get("arms") or ())
    by_arm = {str(arm.get("arm_id")): dict(arm) for arm in packet.get("arms") or ()}
    cap_arm = by_arm.get(ARM_B_CAP_MAX_ABS_1024)
    if not cap_arm:
        raise ValueError("Step-3 packet must include B_cap max_abs 1024 arm")
    if cap_arm.get("vote_law") != "rank_free_sign_pressure":
        raise ValueError("B_cap arm must hold rank_free_sign_pressure vote law fixed")
    if cap_arm.get("tie_policy_id") != TIE_POLICY_DETERMINISTIC_HASH_MATCHED:
        raise ValueError("B_cap arm must share A1/B deterministic tie policy")
    if int(cap_arm.get("max_abs_per_tensor", -1)) != STEP3_CAP_MAX_ABS_PER_TENSOR:
        raise ValueError("B_cap arm max_abs_per_tensor must be 1024")
    _reject_raw_arrays(packet)
    _validate_resource_lane(packet.get("resource_lane") or {})
    _validate_phase_budgets(packet.get("phase_budgets") or {})
    _validate_author_hash_gates(packet)

    if packet.get("mode_sequence") != [
        STEP3_PHASE_POWER_150,
        STEP3_PHASE_POWER_300,
        STEP3_PHASE_TRUST_REGION_150,
        STEP3_PHASE_TRUST_REGION_300,
    ]:
        raise ValueError("Step-3 mode_sequence must be power-150, power-300, trust-150, trust-300")
    power = packet.get("power_ladder") or {}
    if int(power.get("steps_first", -1)) != 150:
        raise ValueError("Step-3 power ladder first rung must be 150 steps")
    if int(power.get("steps_optional_continuation", -1)) != 300:
        raise ValueError("Step-3 power ladder continuation must be 300 steps")
    if int(power.get("max_steps_hard", -1)) != STEP3_MAX_STEPS_HARD:
        raise ValueError("Step-3 power ladder max_steps_hard must be 300")
    _validate_step3_power_floor(power.get("floor") or {})

    trust_region = packet.get("trust_region") or {}
    if trust_region.get("variable") != "max_abs_per_tensor":
        raise ValueError("Step-3 trust_region variable must be max_abs_per_tensor")
    if int(trust_region.get("baseline_value", -1)) != STEP3_BASELINE_MAX_ABS_PER_TENSOR:
        raise ValueError("Step-3 trust_region baseline max_abs must be 4096")
    if int(trust_region.get("cap_value", -1)) != STEP3_CAP_MAX_ABS_PER_TENSOR:
        raise ValueError("Step-3 trust_region cap max_abs must be 1024")
    if float(trust_region.get("fraction_per_tensor", -1.0)) != STEP3_FRACTION_PER_TENSOR:
        raise ValueError("Step-3 trust_region fraction_per_tensor must be 1.0")
    if trust_region.get("global_cap_contract") != "off" or not bool(trust_region.get("global_cap_deferred")):
        raise ValueError("Step-3 trust_region global cap must remain off/deferred")
    _validate_step3_effective_cap(trust_region.get("effective_cap") or {})

    success = packet.get("success_boundary") or {}
    required_success = {
        "B_cap_beats_B_baseline_on_paired_exact",
        "B_cap_beats_B_baseline_on_paired_loss",
        "B_cap_beats_A1_on_paired_exact",
        "B_cap_beats_A1_on_paired_loss",
    }
    if not bool(success.get("requires_power_floor")):
        raise ValueError("Step-3 success boundary must require power floor")
    if set(success.get("positive_requires") or ()) != required_success:
        raise ValueError("Step-3 success boundary must require B_cap to beat B baseline and A1")
    if success.get("null_pivot") != "credit-generation/ranking reformulation":
        raise ValueError("Step-3 null must pivot to credit-generation/ranking reformulation")

    terminal = packet.get("terminal_criteria") or {}
    terminal_branches = set(terminal.get("branch_classifier") or ())
    for branch in (
        BRANCH_CAP_NOOP,
        BRANCH_MEASUREMENT_UNDERPOWERED,
        BRANCH_MEASUREMENT_POWERED,
        BRANCH_MEASUREMENT_LOSS_POWERED,
        BRANCH_POWERED_NEGATIVE_OR_LOSS_ONLY,
    ):
        if branch not in terminal_branches:
            raise ValueError(f"Step-3 terminal classifier missing {branch}")
    if terminal.get("cap_noop_branch") != BRANCH_CAP_NOOP:
        raise ValueError("Step-3 terminal criteria must classify ineffective cap as cap_noop")
    if not bool(terminal.get("effective_cap_required")):
        raise ValueError("Step-3 terminal criteria must require effective cap")
    _validate_step3_power_floor(terminal.get("step3_power_floor") or {})

    commands = packet.get("commands")
    if not isinstance(commands, list):
        raise ValueError("Step-3 packet commands must be a list")
    seen = {(str(cmd.get("mode")), str(cmd.get("arm_id"))) for cmd in commands}
    expected_power = {
        (phase, arm_id)
        for phase in STEP3_POWER_PHASES
        for arm_id in SCIENCE_ARM_IDS
    }
    expected_trust = {
        (phase, arm_id)
        for phase in STEP3_TRUST_REGION_PHASES
        for arm_id in set(SCIENCE_ARM_IDS) | {ARM_B_CAP_MAX_ABS_1024}
    }
    if seen != expected_power | expected_trust:
        raise ValueError("Step-3 packet must include power arms and trust-region B_cap matrix")
    for command in commands:
        _validate_step3_command_record(command)
    artifact_policy = packet.get("artifact_policy") or {}
    if not bool(artifact_policy.get("compact_json_ndjson_only")):
        raise ValueError("Step-3 artifact policy must require compact JSON/NDJSON")
    if bool(artifact_policy.get("pt_writes_allowed")):
        raise ValueError("Step-3 artifact policy must reject .pt writes")


def _validate_step4_match_rule(rule: Mapping[str, Any]) -> None:
    if int(rule.get("strict_gap_max", -1)) != STEP4_MATCH_STRICT_GAP_MAX:
        raise ValueError("Step-4 match rule strict_gap_max must be 3")
    if int(rule.get("strict_total", -1)) != STEP4_MATCH_STRICT_TOTAL:
        raise ValueError("Step-4 match rule strict_total must be 90")
    if rule.get("paired_loss_comparison") != "arm_minus_A0":
        raise ValueError("Step-4 match rule must compare arm_minus_A0")
    if rule.get("paired_loss_ci") != "95% bootstrap":
        raise ValueError("Step-4 match rule must pin 95% bootstrap CI")
    if rule.get("paired_loss_match_if") != "ci_crosses_zero_or_entirely_below_zero":
        raise ValueError("Step-4 match rule must allow CI crosses zero or favors arm")
    if not bool(rule.get("carrier_named_only_on_match_to_A0")):
        raise ValueError("Step-4 match rule must name carrier only on match-to-A0")
    if not bool(rule.get("beat_A1_or_B_is_not_a_carrier_claim")):
        raise ValueError("Step-4 match rule must reject beat-A1/B-only carrier claims")


def _validate_step4_mass_rule(rule: Mapping[str, Any]) -> None:
    if rule.get("classification") != BRANCH_MASS_CONFOUNDED_CURRENT_ORDER_SIGNAL:
        raise ValueError("Step-4 mass rule must classify mass-confounded current-order signal")
    if set(rule.get("compares") or ()) != {"C_vs_A0", "C_vs_B"}:
        raise ValueError("Step-4 mass rule must compare C vs A0 and C vs B")
    if tuple(rule.get("count_metrics") or ()) != STEP4_MASS_COUNT_METRICS:
        raise ValueError("Step-4 mass rule count metrics drifted")
    if tuple(rule.get("pressure_metrics") or ()) != STEP4_MASS_PRESSURE_METRICS:
        raise ValueError("Step-4 mass rule pressure metrics drifted")
    if float(rule.get("ratio_min_inclusive", -1.0)) != STEP4_MASS_RATIO_MIN:
        raise ValueError("Step-4 mass rule ratio_min must be 0.75")
    if float(rule.get("ratio_max_inclusive", -1.0)) != STEP4_MASS_RATIO_MAX:
        raise ValueError("Step-4 mass rule ratio_max must be 1.25")
    if float(rule.get("absolute_delta_min_inclusive", -1.0)) != STEP4_MASS_ABS_DELTA_MIN:
        raise ValueError("Step-4 mass rule absolute delta minimum must be 4")
    if not bool(rule.get("not_carrier_ready")):
        raise ValueError("Step-4 mass-confounded branch must not be carrier-ready")


def _validate_step6_mass_rule(rule: Mapping[str, Any]) -> None:
    if rule.get("classification") != BRANCH_MASS_CONFOUNDED_CURRENT_ORDER_SIGNAL:
        raise ValueError("Step-6 mass rule must classify mass-confounded current-order signal")
    if set(rule.get("compares") or ()) != {"A0_vs_A1", "A0_vs_C"}:
        raise ValueError("Step-6 mass rule must compare A0 vs A1 and A0 vs C")
    if tuple(rule.get("count_metrics") or ()) != STEP4_MASS_COUNT_METRICS:
        raise ValueError("Step-6 mass rule count metrics drifted")
    if tuple(rule.get("pressure_metrics") or ()) != STEP4_MASS_PRESSURE_METRICS:
        raise ValueError("Step-6 mass rule pressure metrics drifted")
    if float(rule.get("ratio_min_inclusive", -1.0)) != STEP4_MASS_RATIO_MIN:
        raise ValueError("Step-6 mass rule ratio_min must be 0.75")
    if float(rule.get("ratio_max_inclusive", -1.0)) != STEP4_MASS_RATIO_MAX:
        raise ValueError("Step-6 mass rule ratio_max must be 1.25")
    if float(rule.get("absolute_delta_min_inclusive", -1.0)) != STEP4_MASS_ABS_DELTA_MIN:
        raise ValueError("Step-6 mass rule absolute delta minimum must be 4")
    if not bool(rule.get("not_carrier_ready")):
        raise ValueError("Step-6 mass-confounded branch must not be carrier-ready")


def validate_powered_rank_signal_decomposition_packet(packet: Mapping[str, Any]) -> None:
    if packet.get("schema_version") != OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION:
        raise ValueError("unsupported optimizer update-law Step-4 packet schema")
    if packet.get("diagnostic_class") != DIAGNOSTIC_CLASS_PRE_FULL_STACK:
        raise ValueError("Step-4 packet must be pre_full_stack_diagnostic")
    _validate_author_only_fields(
        packet,
        expected_packet_kind=STEP4_POWERED_RANK_SIGNAL_DECOMPOSITION_PACKET_KIND,
        label="author-only Step-4 packet",
    )
    if bool(packet.get("ready_for_main_science")):
        raise ValueError("Step-4 packet must keep ready_for_main_science=false")
    if not bool(packet.get("optimizer_credit_state_science_dependent")):
        raise ValueError("Step-4 packet must state optimizer_credit_state remains science-dependent")
    _reject_raw_arrays(packet)
    _validate_resource_lane(packet.get("resource_lane") or {})
    _validate_phase_budgets(packet.get("phase_budgets") or {})
    _validate_author_hash_gates(packet)

    by_arm = {str(arm.get("arm_id")): dict(arm) for arm in packet.get("arms") or ()}
    if set(by_arm) != set(STEP4_ARM_IDS):
        raise ValueError("Step-4 packet must include exactly A0/A1/B/C/inverted arms")
    _validate_arms([by_arm[arm_id] for arm_id in SCIENCE_ARM_IDS])
    c_arm = by_arm[ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER]
    if c_arm.get("vote_law") != "rank_free_sign_pressure":
        raise ValueError("Step-4 C arm must use rank_free_sign_pressure vote law")
    if c_arm.get("tie_policy_id") != TIE_POLICY_CURRENT_MARGIN_INDEX:
        raise ValueError("Step-4 C arm must use current qacc-margin/order bundle")
    if "not pure current-order rank" not in str(c_arm.get("claim_caveat", "")):
        raise ValueError("Step-4 C arm must disclaim pure current-order rank")

    if packet.get("mode_sequence") != list(STEP4_PHASES):
        raise ValueError("Step-4 mode_sequence must be rank_signal_150 then rank_signal_300")
    power = packet.get("power_ladder") or {}
    if int(power.get("steps_first", -1)) != 150:
        raise ValueError("Step-4 power ladder first rung must be 150 steps")
    if int(power.get("steps_optional_continuation", -1)) != 300:
        raise ValueError("Step-4 power ladder continuation must be 300 steps")
    if int(power.get("max_steps_hard", -1)) != max(STEP4_PHASE_STEPS.values()):
        raise ValueError("Step-4 power ladder max_steps_hard must be 300")
    continuation = str(power.get("continuation_enabled_if", ""))
    if "ambiguous" not in continuation or "clear misses stop at 150" not in continuation:
        raise ValueError("Step-4 continuation rule must be 300-only-if-ambiguous")
    _validate_step3_power_floor(power.get("floor") or {})
    _validate_step4_match_rule(packet.get("match_to_A0_rule") or {})
    _validate_step4_mass_rule(packet.get("mass_confound_rule") or {})

    success = packet.get("success_boundary") or {}
    if not bool(success.get("carrier_named_only_on_match_to_A0")):
        raise ValueError("Step-4 success boundary must require match-to-A0")
    if success.get("C_claim") != BRANCH_CURRENT_QACC_MARGIN_ORDER_BUNDLE_CARRIER:
        raise ValueError("Step-4 C claim branch drifted")
    if success.get("C_mass_confounded_branch") != BRANCH_MASS_CONFOUNDED_CURRENT_ORDER_SIGNAL:
        raise ValueError("Step-4 mass-confounded branch drifted")
    if "margin-vs-index split deferred" not in str(success.get("C_claim_caveat", "")):
        raise ValueError("Step-4 C claim caveat must defer margin-vs-index split")

    terminal = packet.get("terminal_criteria") or {}
    terminal_branches = set(terminal.get("branch_classifier") or ())
    expected_branches = {
        BRANCH_CURRENT_QACC_MARGIN_ORDER_BUNDLE_CARRIER,
        BRANCH_RANK_MAGNITUDE_CONDITIONED_ON_CURRENT_ORDER,
        BRANCH_CURRENT_ORDER_NOT_NECESSARY,
        BRANCH_PARTIAL_LOCAL_SIGNAL,
        BRANCH_NO_MATCH_DIFFERENT_CREDIT_SOURCE,
        BRANCH_MASS_CONFOUNDED_CURRENT_ORDER_SIGNAL,
        BRANCH_MEASUREMENT_UNDERPOWERED,
        BRANCH_MEASUREMENT_POWERED,
        BRANCH_MEASUREMENT_LOSS_POWERED,
        BRANCH_POWERED_NEGATIVE_OR_LOSS_ONLY,
    }
    if terminal_branches != expected_branches:
        raise ValueError("Step-4 terminal branch classifier drifted")
    _validate_step3_power_floor(terminal.get("step4_power_floor") or {})
    _validate_step4_match_rule(terminal.get("match_to_A0_rule") or {})
    _validate_step4_mass_rule(terminal.get("mass_confound_rule") or {})
    if not bool(terminal.get("no_carrier_claim_on_beating_A1_or_B_only")):
        raise ValueError("Step-4 terminal criteria must block beat-A1/B-only carrier claims")

    commands = packet.get("commands")
    if not isinstance(commands, list):
        raise ValueError("Step-4 packet commands must be a list")
    seen = {(str(cmd.get("mode")), str(cmd.get("arm_id"))) for cmd in commands}
    expected = {
        (phase, arm_id)
        for phase in STEP4_PHASES
        for arm_id in STEP4_ARM_IDS
    }
    if seen != expected:
        raise ValueError("Step-4 packet must include each arm at 150 and 300")
    for command in commands:
        _validate_step4_command_record(command)
    artifact_policy = packet.get("artifact_policy") or {}
    if not bool(artifact_policy.get("compact_json_ndjson_only")):
        raise ValueError("Step-4 artifact policy must require compact JSON/NDJSON")
    if bool(artifact_policy.get("pt_writes_allowed")):
        raise ValueError("Step-4 artifact policy must reject .pt writes")


def _validate_step5_command_record(command: Mapping[str, Any]) -> None:
    missing = [field for field in _COMMAND_REQUIRED_FIELDS if field not in command]
    if missing:
        raise ValueError(f"Step-5 command record missing required fields: {missing}")
    mode = str(command.get("mode"))
    arm_id = str(command.get("arm_id"))
    science_arm = str(command.get("science_arm"))
    if mode != STEP4_PHASE_RANK_SIGNAL_150:
        raise ValueError("Step-5 command mode must be rank_signal_150 only")
    if arm_id not in STEP5_ARM_IDS:
        raise ValueError(f"Step-5 command record has unsupported arm_id {arm_id!r}")
    if science_arm != arm_id or science_arm not in STEP5_ARM_IDS:
        raise ValueError("Step-5 command science_arm must match a Step-5 arm_id")
    if int(command.get("n_rows", -1)) != 150 or int(command.get("steps_requested", -2)) != 150:
        raise ValueError("Step-5 command steps_requested must be exactly 150")
    if command.get("steps_source") != "STEP5_PHASE_STEPS[mode]":
        raise ValueError("Step-5 command steps_source must document STEP5_PHASE_STEPS")
    if int(command.get("curriculum_seed", -1)) != STEP5_CURRICULUM_SEED:
        raise ValueError("Step-5 command curriculum_seed must be 17")
    if int(command.get("support_order_seed", -1)) != STEP5_SUPPORT_ORDER_SEED:
        raise ValueError("Step-5 command support_order_seed must be 29")
    if not bool(command.get("support_order_permutation_required")):
        raise ValueError("Step-5 command must require support-order permutation")
    if bool(command.get("qacc_kernelized")):
        raise ValueError("Step-5 command must keep qacc_kernelized=false")
    if int(command.get("max_abs_per_tensor", -1)) != STEP3_BASELINE_MAX_ABS_PER_TENSOR:
        raise ValueError("Step-5 command max_abs_per_tensor must keep baseline 4096")
    if float(command.get("fraction_per_tensor", -1.0)) != STEP3_FRACTION_PER_TENSOR:
        raise ValueError("Step-5 command fraction_per_tensor must be 1.0")
    if command.get("global_cap_contract") != "off":
        raise ValueError("Step-5 command global cap must stay off")
    env = command.get("env")
    if not isinstance(env, Mapping):
        raise ValueError("Step-5 command env must be a mapping")
    if env.get("HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE") != "1":
        raise ValueError("Step-5 command env missing HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE=1")
    if env.get("HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH") != "1":
        raise ValueError("Step-5 command env missing HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH=1")
    argv = command.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("Step-5 command argv must be a list[str]")
    required_args = {
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--device",
        "--parent",
        "--parent-sha256",
        "--scratch-root",
        "--curriculum-seed",
        "--support-order-seed",
        "--steps",
        "--max-steps-hard",
        "--audit-interval",
        "--science-arm",
        "--max-abs-per-tensor",
        "--emit-progress",
    }
    if not required_args.issubset(set(argv)):
        raise ValueError("Step-5 command argv missing required probe launch arguments")
    expected_flag_values = (
        ("--science-arm", science_arm),
        ("--curriculum-seed", str(STEP5_CURRICULUM_SEED)),
        ("--support-order-seed", str(STEP5_SUPPORT_ORDER_SEED)),
        ("--steps", "150"),
        ("--max-steps-hard", "150"),
        ("--audit-interval", "150"),
        ("--max-abs-per-tensor", str(STEP3_BASELINE_MAX_ABS_PER_TENSOR)),
    )
    for flag, expected in expected_flag_values:
        try:
            observed = argv[argv.index(flag) + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Step-5 command argv missing {flag} value") from exc
        if observed != expected:
            raise ValueError(f"Step-5 command argv {flag} must be {expected!r}, got {observed!r}")
    if STEP4_PHASE_RANK_SIGNAL_300 in argv or "300" in {
        str(command.get("n_rows")),
        str(command.get("steps_requested")),
    }:
        raise ValueError("Step-5 command must not include 300-step continuation")
    try:
        device = argv[argv.index("--device") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("Step-5 command argv missing --device value") from exc
    if not device.startswith("cuda:"):
        raise ValueError("Step-5 command argv --device must target CUDA for launch packet")
    for path_field in ("stdout_path", "stderr_path", "receipt_path", "scratch_root"):
        value = str(command.get(path_field))
        if not value:
            raise ValueError(f"Step-5 command {path_field} must be non-empty")
        if value.endswith(".pt"):
            raise ValueError(f"Step-5 command {path_field} cannot target .pt artifacts")
    if command.get("expected_exit_policy") != "exit_0_required_else_stop_no_retry_no_verdict":
        raise ValueError("Step-5 command expected_exit_policy must fail closed")


def _validate_step5_support_order_contract(contract: Mapping[str, Any]) -> None:
    if int(contract.get("support_order_seed", -1)) != STEP5_SUPPORT_ORDER_SEED:
        raise ValueError("Step-5 support-order contract must pin support_order_seed=29")
    if int(contract.get("curriculum_seed", -1)) != STEP5_CURRICULUM_SEED:
        raise ValueError("Step-5 support-order contract must pin curriculum_seed=17")
    if contract.get("support_content_unchanged_basis") != "order_invariant_multiset_hash16":
        raise ValueError("Step-5 support_content_unchanged basis must be order-invariant")
    if bool(contract.get("ordered_support_content_hash16_is_invariant")):
        raise ValueError("Step-5 ordered support_content_hash16 must not be treated as invariant")
    if contract.get("legacy_support_content_hash16_semantics") != "ordered_batch_hashes_order_sensitive":
        raise ValueError("Step-5 must document legacy support_content_hash16 as order-sensitive")
    if not bool(contract.get("ordered_hashes_must_differ")):
        raise ValueError("Step-5 contract must require ordered traversal hashes to differ")
    if not bool(contract.get("order_invariant_hashes_must_match")):
        raise ValueError("Step-5 contract must require invariant hashes to match")
    false_trap = str(contract.get("false_invariant_trap", ""))
    if "support_content_hash16" not in false_trap or "must not be derived" not in false_trap:
        raise ValueError("Step-5 false-invariant trap must reject support_content_hash16 basis")


def validate_support_order_trajectory_robustness_packet(packet: Mapping[str, Any]) -> None:
    if packet.get("schema_version") != OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION:
        raise ValueError("unsupported optimizer update-law Step-5 packet schema")
    if packet.get("diagnostic_class") != DIAGNOSTIC_CLASS_PRE_FULL_STACK:
        raise ValueError("Step-5 packet must be pre_full_stack_diagnostic")
    _validate_author_only_fields(
        packet,
        expected_packet_kind=STEP5_SUPPORT_ORDER_TRAJECTORY_ROBUSTNESS_PACKET_KIND,
        label="author-only Step-5 packet",
    )
    if bool(packet.get("ready_for_main_science")):
        raise ValueError("Step-5 packet must keep ready_for_main_science=false")
    if bool(packet.get("qacc_kernelized")):
        raise ValueError("Step-5 packet must keep qacc_kernelized=false")
    if not bool(packet.get("optimizer_credit_state_science_dependent")):
        raise ValueError("Step-5 packet must state optimizer_credit_state remains science-dependent")
    _reject_raw_arrays(packet)
    _validate_resource_lane(packet.get("resource_lane") or {})
    _validate_phase_budgets(packet.get("phase_budgets") or {})
    _validate_author_hash_gates(packet)
    if int(packet.get("support_order_seed", -1)) != STEP5_SUPPORT_ORDER_SEED:
        raise ValueError("Step-5 packet support_order_seed must be 29")
    if int(packet.get("curriculum_seed", -1)) != STEP5_CURRICULUM_SEED:
        raise ValueError("Step-5 packet curriculum_seed must be 17")
    _validate_step5_support_order_contract(packet.get("support_order_proof_contract") or {})

    by_arm = {str(arm.get("arm_id")): dict(arm) for arm in packet.get("arms") or ()}
    if set(by_arm) != set(STEP5_ARM_IDS):
        raise ValueError("Step-5 packet must include exactly A0/B/C/inverted arms and no A1")
    if ARM_A1_RANK_BUCKET_ORDER_MATCHED in by_arm:
        raise ValueError("Step-5 packet must not include A1")
    if by_arm[ARM_A0_RANK_BUCKET_CURRENT].get("tie_policy_id") != TIE_POLICY_CURRENT_MARGIN_INDEX:
        raise ValueError("Step-5 A0 must keep current qacc-margin/index ordering")
    if by_arm[ARM_B_RANK_FREE_SIGN_PRESSURE].get("tie_policy_id") != TIE_POLICY_DETERMINISTIC_HASH_MATCHED:
        raise ValueError("Step-5 B must keep deterministic order-matched tie policy")
    c_arm = by_arm[ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER]
    if c_arm.get("vote_law") != "rank_free_sign_pressure":
        raise ValueError("Step-5 C arm must use rank_free_sign_pressure vote law")
    if c_arm.get("tie_policy_id") != TIE_POLICY_CURRENT_MARGIN_INDEX:
        raise ValueError("Step-5 C arm must use current qacc-margin/order bundle")
    if by_arm[ARM_INVERTED_SIGN_PRESSURE].get("tie_policy_id") != TIE_POLICY_DETERMINISTIC_HASH_MATCHED:
        raise ValueError("Step-5 inverted falsifier must share B deterministic tie policy")

    if packet.get("mode_sequence") != list(STEP5_PHASES):
        raise ValueError("Step-5 mode_sequence must be rank_signal_150 only")
    power = packet.get("power_ladder") or {}
    if int(power.get("steps_first", -1)) != 150:
        raise ValueError("Step-5 power ladder first rung must be 150 steps")
    if power.get("steps_optional_continuation") is not None:
        raise ValueError("Step-5 power ladder must not define a 300-step continuation")
    if int(power.get("max_steps_hard", -1)) != 150:
        raise ValueError("Step-5 power ladder max_steps_hard must be 150")
    continuation = str(power.get("continuation_enabled_if", ""))
    if "disabled" not in continuation or "150-only" not in continuation:
        raise ValueError("Step-5 continuation rule must be disabled 150-only")
    _validate_step3_power_floor(power.get("floor") or {})
    _validate_step4_mass_rule(packet.get("mass_confound_rule") or {})
    pass_rule = packet.get("pass_rule") or {}
    if int(pass_rule.get("C_strict_floor_count", -1)) != STEP5_STRICT_FLOOR_COUNT:
        raise ValueError("Step-5 pass rule must require C.strict >= 10/90")
    if int(pass_rule.get("C_margin_over_max_A0_B_count", -1)) != STEP5_STRICT_MARGIN_COUNT:
        raise ValueError("Step-5 pass rule must require C margin >= 5 over A0/B")
    if bool(pass_rule.get("qacc_kernelized")):
        raise ValueError("Step-5 pass rule must keep qacc_kernelized=false")
    paired = pass_rule.get("paired_loss_required") or {}
    if set(paired.get("comparisons") or ()) != {"C_minus_A0", "C_minus_B"}:
        raise ValueError("Step-5 pass rule must require C-A0 and C-B paired loss")
    if float(paired.get("mean_must_be_less_than", 1.0)) != 0.0:
        raise ValueError("Step-5 paired-loss mean threshold must be 0")
    if float(paired.get("ci_high_must_be_less_than", 1.0)) != 0.0:
        raise ValueError("Step-5 paired-loss CI-high threshold must be 0")

    terminal = packet.get("terminal_criteria") or {}
    _validate_step4_mass_rule(terminal.get("mass_confound_rule") or {})
    if bool(terminal.get("qacc_kernelized")):
        raise ValueError("Step-5 terminal criteria must keep qacc_kernelized=false")
    if bool(terminal.get("ready_for_main_science_after_pass")):
        raise ValueError("Step-5 terminal criteria must keep ready_for_main_science=false")
    terminal_tables = set(terminal.get("terminal_receipt_required_tables") or ())
    required_tables = {
        "strict_exact_by_arm",
        "paired_loss_C_minus_A0_and_C_minus_B",
        "mass_confound_C_vs_A0_and_C_vs_B",
        "device_vs_hot_loop_qacc_kernelized_false",
    }
    if terminal_tables != required_tables:
        raise ValueError("Step-5 terminal receipt table requirements drifted")
    terminal_proofs = set(terminal.get("terminal_receipt_required_proofs") or ())
    required_proofs = {
        "parent_sha256_pre_post_unchanged",
        "resource_lane_released",
        "artifact_paths",
        "support_order_seed",
        "support_order_original_ordered_traversal_hash16",
        "support_order_permuted_ordered_traversal_hash16",
        "support_order_original_invariant_multiset_hash16",
        "support_order_permuted_invariant_multiset_hash16",
        "support_content_unchanged_from_order_invariant_hash",
    }
    if terminal_proofs != required_proofs:
        raise ValueError("Step-5 terminal receipt proof requirements drifted")

    commands = packet.get("commands")
    if not isinstance(commands, list):
        raise ValueError("Step-5 packet commands must be a list")
    seen = {(str(cmd.get("mode")), str(cmd.get("arm_id"))) for cmd in commands}
    expected = {
        (STEP4_PHASE_RANK_SIGNAL_150, arm_id)
        for arm_id in STEP5_ARM_IDS
    }
    if seen != expected:
        raise ValueError("Step-5 packet must include exactly four 150-only commands")
    for command in commands:
        _validate_step5_command_record(command)
    artifact_policy = packet.get("artifact_policy") or {}
    if not bool(artifact_policy.get("compact_json_ndjson_only")):
        raise ValueError("Step-5 artifact policy must require compact JSON/NDJSON")
    if bool(artifact_policy.get("pt_writes_allowed")):
        raise ValueError("Step-5 artifact policy must reject .pt writes")


def _validate_step6_command_record(command: Mapping[str, Any]) -> None:
    missing = [field for field in _COMMAND_REQUIRED_FIELDS if field not in command]
    if missing:
        raise ValueError(f"Step-6 command record missing required fields: {missing}")
    mode = str(command.get("mode"))
    arm_id = str(command.get("arm_id"))
    science_arm = str(command.get("science_arm"))
    if mode != STEP4_PHASE_RANK_SIGNAL_150:
        raise ValueError("Step-6 command mode must be rank_signal_150 only")
    if arm_id not in STEP6_ARM_IDS:
        raise ValueError(f"Step-6 command record has unsupported arm_id {arm_id!r}")
    if science_arm != arm_id or science_arm not in STEP6_ARM_IDS:
        raise ValueError("Step-6 command science_arm must match a Step-6 arm_id")
    if int(command.get("n_rows", -1)) != 150 or int(command.get("steps_requested", -2)) != 150:
        raise ValueError("Step-6 command steps_requested must be exactly 150")
    if command.get("steps_source") != "STEP6_PHASE_STEPS[mode]":
        raise ValueError("Step-6 command steps_source must document STEP6_PHASE_STEPS")
    if int(command.get("curriculum_seed", -1)) != STEP6_CURRICULUM_SEED:
        raise ValueError("Step-6 command curriculum_seed must be 17")
    if bool(command.get("qacc_kernelized")):
        raise ValueError("Step-6 command must keep qacc_kernelized=false")
    if int(command.get("max_abs_per_tensor", -1)) != STEP3_BASELINE_MAX_ABS_PER_TENSOR:
        raise ValueError("Step-6 command max_abs_per_tensor must keep baseline 4096")
    if float(command.get("fraction_per_tensor", -1.0)) != STEP3_FRACTION_PER_TENSOR:
        raise ValueError("Step-6 command fraction_per_tensor must be 1.0")
    if command.get("global_cap_contract") != "off":
        raise ValueError("Step-6 command global cap must stay off")
    if not bool(command.get("fresh_step6_evidence")):
        raise ValueError("Step-6 command must be fresh Step-6 evidence")
    if bool(command.get("context_only")):
        raise ValueError("Step-6 command cannot be context_only reused evidence")
    if not bool(command.get("classifier_evidence")):
        raise ValueError("Step-6 command must be classifier evidence")
    env = command.get("env")
    if not isinstance(env, Mapping):
        raise ValueError("Step-6 command env must be a mapping")
    if env.get("HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE") != "1":
        raise ValueError("Step-6 command env missing HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE=1")
    if env.get("HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH") != "1":
        raise ValueError("Step-6 command env missing HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH=1")
    argv = command.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("Step-6 command argv must be a list[str]")
    required_args = {
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--device",
        "--parent",
        "--parent-sha256",
        "--scratch-root",
        "--curriculum-seed",
        "--steps",
        "--max-steps-hard",
        "--audit-interval",
        "--science-arm",
        "--max-abs-per-tensor",
        "--emit-progress",
    }
    if not required_args.issubset(set(argv)):
        raise ValueError("Step-6 command argv missing required probe launch arguments")
    expected_flag_values = (
        ("--science-arm", science_arm),
        ("--curriculum-seed", str(STEP6_CURRICULUM_SEED)),
        ("--steps", "150"),
        ("--max-steps-hard", "150"),
        ("--audit-interval", "150"),
        ("--max-abs-per-tensor", str(STEP3_BASELINE_MAX_ABS_PER_TENSOR)),
    )
    for flag, expected in expected_flag_values:
        try:
            observed = argv[argv.index(flag) + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Step-6 command argv missing {flag} value") from exc
        if observed != expected:
            raise ValueError(f"Step-6 command argv {flag} must be {expected!r}, got {observed!r}")
    seed = command.get("support_order_seed")
    seed_label = str(command.get("seed_label"))
    if seed not in STEP6_SUPPORT_ORDER_SEEDS:
        raise ValueError("Step-6 command support_order_seed must be one of null, 29, 43")
    if seed is None:
        if seed_label != "original":
            raise ValueError("Step-6 original trajectory seed_label must be original")
        if "--support-order-seed" in argv:
            raise ValueError("Step-6 original trajectory argv must omit --support-order-seed")
        if bool(command.get("support_order_permutation_required")):
            raise ValueError("Step-6 original trajectory must not require support-order permutation")
    else:
        seed_int = int(seed)
        if seed_label != f"seed{seed_int}":
            raise ValueError("Step-6 seeded trajectory seed_label must match support_order_seed")
        if "--support-order-seed" not in argv:
            raise ValueError("Step-6 seeded trajectory argv must include --support-order-seed")
        try:
            observed_seed = argv[argv.index("--support-order-seed") + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError("Step-6 seeded trajectory argv missing --support-order-seed value") from exc
        if observed_seed != str(seed_int):
            raise ValueError("Step-6 seeded trajectory argv support-order seed drifted")
        if not bool(command.get("support_order_permutation_required")):
            raise ValueError("Step-6 seeded trajectory must require support-order permutation")
    if STEP4_PHASE_RANK_SIGNAL_300 in argv or "300" in {
        str(command.get("n_rows")),
        str(command.get("steps_requested")),
    }:
        raise ValueError("Step-6 command must not include 300-step continuation")
    try:
        device = argv[argv.index("--device") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("Step-6 command argv missing --device value") from exc
    if not device.startswith("cuda:"):
        raise ValueError("Step-6 command argv --device must target CUDA for launch packet")
    for path_field in ("stdout_path", "stderr_path", "receipt_path", "scratch_root"):
        value = str(command.get(path_field))
        if not value:
            raise ValueError(f"Step-6 command {path_field} must be non-empty")
        if value.endswith(".pt"):
            raise ValueError(f"Step-6 command {path_field} cannot target .pt artifacts")
    if command.get("expected_exit_policy") != "exit_0_required_else_stop_no_retry_no_verdict":
        raise ValueError("Step-6 command expected_exit_policy must fail closed")


def _validate_step6_context_only_receipts(entries: Sequence[Mapping[str, Any]]) -> None:
    expected_labels = {"step4_original_order_context", "step5_seed29_context"}
    if len(entries) != len(expected_labels):
        raise ValueError("Step-6 context-only prior receipt ledger must list exactly Step-4 and Step-5 context")
    observed_labels = {str(entry.get("label")) for entry in entries}
    if observed_labels != expected_labels:
        raise ValueError("Step-6 context-only prior receipt ledger labels must be Step-4 and Step-5")
    for entry in entries:
        if not bool(entry.get("context_only")):
            raise ValueError("Step-6 reused historical receipt evidence must be marked context_only")
        if bool(entry.get("classifier_evidence")):
            raise ValueError("Step-6 classifier evidence must not come from reused Step-4/5 receipts")


def _validate_step6_support_order_contract(contract: Mapping[str, Any]) -> None:
    seeds = contract.get("seed_matrix") or ()
    if not isinstance(seeds, list) or len(seeds) != len(STEP6_SUPPORT_ORDER_SEEDS):
        raise ValueError("Step-6 support-order contract must define exactly three seed specs")
    observed = [item.get("support_order_seed") for item in seeds]
    if observed != list(STEP6_SUPPORT_ORDER_SEEDS):
        raise ValueError("Step-6 seed specs must be [null, 29, 43] in preregistered order")
    original = seeds[0]
    if original.get("seed_label") != "original" or not bool(original.get("argv_omits_support_order_seed")):
        raise ValueError("Step-6 original seed spec must omit --support-order-seed")
    for item, expected_seed in zip(seeds[1:], STEP6_SUPPORT_ORDER_SEEDS[1:]):
        if item.get("seed_label") != f"seed{expected_seed}":
            raise ValueError("Step-6 seeded specs must have stable seed labels")
        if bool(item.get("argv_omits_support_order_seed")):
            raise ValueError("Step-6 seeded specs must include --support-order-seed")
        if not bool(item.get("support_order_permutation_required")):
            raise ValueError("Step-6 seeded specs must require support-order permutation")
    if int(contract.get("curriculum_seed", -1)) != STEP6_CURRICULUM_SEED:
        raise ValueError("Step-6 support-order contract must pin curriculum_seed=17")
    if int(contract.get("fixed_preregistered_new_seed", -1)) != STEP6_FIXED_PREREG_NEW_SEED:
        raise ValueError("Step-6 support-order contract must pin seed43 as fixed prereg")
    if bool(contract.get("post_hoc_seed_selection_allowed")):
        raise ValueError("Step-6 support-order contract must reject post-hoc seed selection")
    if contract.get("support_content_unchanged_basis") != "order_invariant_multiset_hash16":
        raise ValueError("Step-6 support_content_unchanged basis must be order-invariant")
    if bool(contract.get("ordered_support_content_hash16_is_invariant")):
        raise ValueError("Step-6 ordered support_content_hash16 must not be treated as invariant")


def validate_order_averaged_a0_component_decomposition_packet(packet: Mapping[str, Any]) -> None:
    if packet.get("schema_version") != OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION:
        raise ValueError("unsupported optimizer update-law Step-6 packet schema")
    if packet.get("diagnostic_class") != DIAGNOSTIC_CLASS_PRE_FULL_STACK:
        raise ValueError("Step-6 packet must be pre_full_stack_diagnostic")
    _validate_author_only_fields(
        packet,
        expected_packet_kind=STEP6_ORDER_AVERAGED_A0_COMPONENT_DECOMPOSITION_PACKET_KIND,
        label="author-only Step-6 packet",
    )
    if bool(packet.get("ready_for_main_science")):
        raise ValueError("Step-6 packet must keep ready_for_main_science=false")
    if bool(packet.get("qacc_kernelized")):
        raise ValueError("Step-6 packet must keep qacc_kernelized=false")
    if not bool(packet.get("optimizer_credit_state_science_dependent")):
        raise ValueError("Step-6 packet must state optimizer_credit_state remains science-dependent")
    _reject_raw_arrays(packet)
    _validate_resource_lane(packet.get("resource_lane") or {})
    _validate_phase_budgets(packet.get("phase_budgets") or {})
    _validate_author_hash_gates(packet)
    if packet.get("mode_sequence") != list(STEP6_PHASES):
        raise ValueError("Step-6 mode_sequence must be rank_signal_150 only")
    if packet.get("support_order_seeds") != list(STEP6_SUPPORT_ORDER_SEEDS):
        raise ValueError("Step-6 support_order_seeds must be [null, 29, 43]")
    if int(packet.get("curriculum_seed", -1)) != STEP6_CURRICULUM_SEED:
        raise ValueError("Step-6 packet must pin curriculum_seed=17")
    _validate_step6_support_order_contract(packet.get("support_order_proof_contract") or {})
    _validate_step6_context_only_receipts(packet.get("context_only_prior_receipts") or ())

    by_arm = {str(arm.get("arm_id")): dict(arm) for arm in packet.get("arms") or ()}
    if set(by_arm) != set(STEP6_ARM_IDS):
        raise ValueError("Step-6 packet must include exactly A0/A1/C arms")
    if ARM_B_RANK_FREE_SIGN_PRESSURE in by_arm or ARM_INVERTED_SIGN_PRESSURE in by_arm:
        raise ValueError("Step-6 packet must not include B or inverted arms")
    if by_arm[ARM_A0_RANK_BUCKET_CURRENT].get("tie_policy_id") != TIE_POLICY_CURRENT_MARGIN_INDEX:
        raise ValueError("Step-6 A0 must keep current qacc-margin/index ordering")
    if by_arm[ARM_A1_RANK_BUCKET_ORDER_MATCHED].get("tie_policy_id") != TIE_POLICY_DETERMINISTIC_HASH_MATCHED:
        raise ValueError("Step-6 A1 must keep deterministic order-matched tie policy")
    c_arm = by_arm[ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER]
    if c_arm.get("vote_law") != "rank_free_sign_pressure":
        raise ValueError("Step-6 C arm must use rank_free_sign_pressure vote law")
    if c_arm.get("tie_policy_id") != TIE_POLICY_CURRENT_MARGIN_INDEX:
        raise ValueError("Step-6 C arm must use current qacc-margin/order bundle")

    power = packet.get("power_ladder") or {}
    if int(power.get("steps_first", -1)) != 150:
        raise ValueError("Step-6 power ladder first rung must be 150 steps")
    if power.get("steps_optional_continuation") is not None:
        raise ValueError("Step-6 power ladder must not define a 300-step continuation")
    if int(power.get("max_steps_hard", -1)) != 150:
        raise ValueError("Step-6 power ladder max_steps_hard must be 150")
    continuation = str(power.get("continuation_enabled_if", ""))
    if "disabled" not in continuation or "150-only" not in continuation:
        raise ValueError("Step-6 continuation rule must be disabled 150-only")
    _validate_step3_power_floor(power.get("floor") or {})
    _validate_step6_mass_rule(packet.get("mass_confound_rule") or {})

    stability = packet.get("order_averaged_stability_rule") or {}
    if stability.get("primary_evidence") != "seed_level_dominance":
        raise ValueError("Step-6 stability rule must use seed-level dominance as primary evidence")
    if stability.get("dominant_arm") != ARM_A0_RANK_BUCKET_CURRENT:
        raise ValueError("Step-6 stability rule must test A0 dominance")
    if set(stability.get("must_beat_arms") or ()) != {
        ARM_A1_RANK_BUCKET_ORDER_MATCHED,
        ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER,
    }:
        raise ValueError("Step-6 stability rule must require A0 beating A1 and C")
    if int(stability.get("min_seeds_dominating", -1)) != 2:
        raise ValueError("Step-6 stability rule must require dominance in at least 2/3 seeds")
    if int(stability.get("total_seeds", -1)) != 3:
        raise ValueError("Step-6 stability rule must cover exactly three seeds")
    if not bool(stability.get("pooled_loss_cannot_override_seed_level_instability")):
        raise ValueError("Step-6 pooled loss must not override seed-level instability")
    if stability.get("positive_classification") != BRANCH_A0_COMPONENT_ORDER_ROBUST:
        raise ValueError("Step-6 positive classification drifted")
    if stability.get("negative_or_unstable_classification") != BRANCH_MEASUREMENT_ORDER_SENSITIVE:
        raise ValueError("Step-6 unstable classification drifted")
    if not bool(stability.get("no_carrier_readiness_or_full_sub2_claim")):
        raise ValueError("Step-6 stability rule must forbid carrier/readiness/full-sub2 claims")

    cost = packet.get("cost_ceiling") or {}
    if int(cost.get("max_arm_runs", -1)) != STEP6_MAX_ARM_RUNS:
        raise ValueError("Step-6 cost ceiling must cap at 9 arm-runs")
    if float(cost.get("max_gpu_hours", -1.0)) > STEP6_GPU_HOUR_CEILING:
        raise ValueError("Step-6 cost ceiling must stay <=2 GPU-hours")
    if not bool(cost.get("stop_before_launch_if_exceeded")):
        raise ValueError("Step-6 cost ceiling must stop before launch if exceeded")

    terminal = packet.get("terminal_criteria") or {}
    if bool(terminal.get("qacc_kernelized")):
        raise ValueError("Step-6 terminal criteria must keep qacc_kernelized=false")
    if bool(terminal.get("ready_for_main_science_after_pass")):
        raise ValueError("Step-6 terminal criteria must keep ready_for_main_science=false")
    if bool(terminal.get("carrier_claim_after_pass")):
        raise ValueError("Step-6 terminal criteria must not make a carrier claim")
    if terminal.get("order_averaged_stability_rule") != stability:
        raise ValueError("Step-6 terminal stability rule must match packet stability rule")
    _validate_step6_mass_rule(terminal.get("mass_confound_rule") or {})
    terminal_tables = set(terminal.get("terminal_receipt_required_tables") or ())
    required_tables = {
        "strict_count_distributions_by_arm_across_seeds",
        "paired_loss_A0_minus_A1_and_A0_minus_C_per_seed",
        "pooled_paired_row_loss_secondary_only",
        "seed_wise_rank_ordering",
        "mass_confound_A0_vs_A1_and_A0_vs_C",
        "fresh_step6_vs_context_only_evidence_ledger",
        "device_vs_hot_loop_qacc_kernelized_false",
    }
    if terminal_tables != required_tables:
        raise ValueError("Step-6 terminal receipt table requirements drifted")
    terminal_proofs = set(terminal.get("terminal_receipt_required_proofs") or ())
    required_proofs = {
        "parent_sha256_pre_post_unchanged",
        "resource_lane_released",
        "artifact_paths",
        "support_order_seed_matrix_original_29_43",
        "original_argv_omits_support_order_seed",
        "seed29_seed43_argv_include_support_order_seed",
        "fresh_step6_receipts_only_for_classifier",
    }
    if terminal_proofs != required_proofs:
        raise ValueError("Step-6 terminal receipt proof requirements drifted")

    commands = packet.get("commands")
    if not isinstance(commands, list):
        raise ValueError("Step-6 packet commands must be a list")
    if len(commands) != STEP6_MAX_ARM_RUNS:
        raise ValueError("Step-6 packet must render exactly 9 commands")
    seen = {
        (cmd.get("support_order_seed"), str(cmd.get("arm_id")))
        for cmd in commands
    }
    expected = {
        (seed, arm_id)
        for seed in STEP6_SUPPORT_ORDER_SEEDS
        for arm_id in STEP6_ARM_IDS
    }
    if seen != expected:
        raise ValueError("Step-6 packet must render seeds [null, 29, 43] x arms [A0,A1,C]")
    for command in commands:
        _validate_step6_command_record(command)
    artifact_policy = packet.get("artifact_policy") or {}
    if not bool(artifact_policy.get("compact_json_ndjson_only")):
        raise ValueError("Step-6 artifact policy must require compact JSON/NDJSON")
    if bool(artifact_policy.get("pt_writes_allowed")):
        raise ValueError("Step-6 artifact policy must reject .pt writes")


def _validate_oracle_screen_arms(arms: Sequence[Mapping[str, Any]]) -> None:
    by_id = {str(arm.get("arm_id")): dict(arm) for arm in arms}
    if set(by_id) != set(ORACLE_SCREEN_ARM_IDS):
        raise ValueError("oracle screen packet must include exactly the 3 registered arms")
    for arm_id, arm in by_id.items():
        if arm.get("candidate_set") != "same_projected_move_candidate_set":
            raise ValueError("oracle screen arms must use the same projected-move candidate set")
        if bool(arm.get("q_persisted")):
            raise ValueError("oracle screen arm must not persist q")
        if bool(arm.get("raw_per_proposal_arrays_included")):
            raise ValueError("oracle screen arm must not include raw per-proposal arrays")
        if arm_id != ORACLE_ARM_DIAGNOSTIC_LOCAL_LOSS_DELTA and bool(arm.get("oracle_applied")):
            raise ValueError("only diagnostic_local_loss_delta may name an oracle arm")
    oracle_arm = by_id[ORACLE_ARM_DIAGNOSTIC_LOCAL_LOSS_DELTA]
    if bool(oracle_arm.get("learner_teacher_promotion")):
        raise ValueError("diagnostic oracle must not be learner/teacher promotional")
    if bool(oracle_arm.get("checkpoint_promotional")):
        raise ValueError("diagnostic oracle must not be checkpoint-promotional")
    if oracle_arm.get("vote_source") != "diagnostic_local_loss_delta":
        raise ValueError("diagnostic oracle arm must use diagnostic_local_loss_delta")


def _validate_oracle_feasibility_budget(budget: Mapping[str, Any]) -> None:
    required = {
        "probe_required_before_full_screen",
        "budget_present",
        "allowed_max_sampled_candidates",
        "max_sampled_candidates",
        "max_seconds_by_budget",
        "max_seconds",
        "reject_if_over_budget",
        "reject_if_unsafe",
        "classify_branch_on_missing_overrun_or_unsafe",
    }
    missing = sorted(required - set(budget))
    if missing:
        raise ValueError(f"oracle feasibility budget missing required fields: {missing}")
    if not bool(budget.get("budget_present")):
        raise ValueError("oracle feasibility budget must be present")
    if not bool(budget.get("probe_required_before_full_screen")):
        raise ValueError("oracle feasibility probe must precede full screen")
    if list(budget.get("allowed_max_sampled_candidates") or ()) != list(
        ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES
    ):
        raise ValueError("oracle feasibility budget allowed_max_sampled_candidates drifted")
    expected_seconds_by_budget = {
        str(candidate_budget): float(seconds)
        for candidate_budget, seconds in ORACLE_SCREEN_MAX_SECONDS_BY_BUDGET.items()
    }
    if dict(budget.get("max_seconds_by_budget") or {}) != expected_seconds_by_budget:
        raise ValueError("oracle feasibility budget max_seconds_by_budget drifted")
    selected_budget = int(budget.get("max_sampled_candidates", 0))
    if selected_budget not in ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES:
        raise ValueError("oracle feasibility max_sampled_candidates must be one of {8,32,64}")
    if float(budget.get("max_seconds", 0.0)) != oracle_screen_budget_max_seconds(
        selected_budget
    ):
        raise ValueError("oracle feasibility max_seconds must match the pinned budget tier")
    if not bool(budget.get("reject_if_over_budget")) or not bool(budget.get("reject_if_unsafe")):
        raise ValueError("oracle feasibility budget must reject overrun or unsafe probes")
    if budget.get("classify_branch_on_missing_overrun_or_unsafe") != BRANCH_ORACLE_INFEASIBLE_OR_TOO_EXPENSIVE:
        raise ValueError("oracle feasibility failure must classify infeasible/too expensive")


def _validate_oracle_non_persistence(contract: Mapping[str, Any]) -> None:
    required_false = {
        "q_persist_allowed",
        "q_persisted",
        "oracle_state_survives_into_learner",
        "learner_teacher_promotion_allowed",
        "learner_teacher_promotion",
        "checkpoint_promotional",
        "checkpoint_written",
        "pt_writes_allowed",
        "readiness_fullsub2_carrier_claim_allowed",
    }
    missing = sorted(required_false - set(contract))
    if missing:
        raise ValueError(f"oracle non-persistence contract missing required fields: {missing}")
    for field in required_false:
        if bool(contract.get(field)):
            raise ValueError(f"oracle non-persistence contract requires {field}=false")


def _validate_oracle_compact_summary_schema(schema: Mapping[str, Any]) -> None:
    expected_fields = set(default_oracle_compact_summary_schema()["allowed_fields"])
    if not bool(schema.get("compact_summary_only")):
        raise ValueError("oracle receipt must be compact-summary-only")
    if set(schema.get("allowed_fields") or ()) != expected_fields:
        raise ValueError("oracle compact summary allowed fields drifted")
    if set(schema.get("required_fields") or ()) != expected_fields:
        raise ValueError("oracle compact summary required fields drifted")
    for field in ("raw_per_proposal_arrays", "raw_candidate_scores", "raw_local_loss_deltas"):
        if bool(schema.get(field)):
            raise ValueError("oracle compact summary must reject raw proposal arrays")


def _validate_oracle_seed_order_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("contrast_support_order_seeds") != list(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError("oracle screen contrast seeds must be [43, 29]")
    if int(contract.get("n20_screen_rows", -1)) != ORACLE_SCREEN_N20_ROWS:
        raise ValueError("oracle screen N=20 row contract drifted")
    if not bool(contract.get("n20_screen_is_launch_gated")):
        raise ValueError("oracle N=20 screen must remain separately launch-gated")
    promotion = contract.get("promotion_condition") or {}
    if not bool(promotion.get("promote_to_n50_x_3_orderings")):
        raise ValueError("oracle screen promotion condition must name N=50 x 3 orderings")
    if int(promotion.get("promotion_rows", -1)) != ORACLE_SCREEN_PROMOTION_ROWS:
        raise ValueError("oracle screen promotion rows must be 50")
    if promotion.get("support_order_seeds") != list(ORACLE_SCREEN_PROMOTION_ORDER_SEEDS):
        raise ValueError("oracle screen promotion order seeds must be [null, 29, 43]")
    if not bool(promotion.get("only_if_non_null")):
        raise ValueError("oracle screen promotion requires non-null result")
    if not bool(promotion.get("only_if_not_artifact_confounded")):
        raise ValueError("oracle screen promotion requires no artifact confound")
    if bool(promotion.get("post_hoc_seed_selection_allowed")):
        raise ValueError("oracle screen must reject post-hoc seed selection")


def _validate_oracle_classifier_contract(contract: Mapping[str, Any]) -> None:
    if not bool(contract.get("exactly_one_branch")):
        raise ValueError("oracle classifier must return exactly one branch")
    if set(contract.get("allowed_branches") or ()) != set(ORACLE_SCREEN_BRANCHES):
        raise ValueError("oracle classifier allowed branches drifted")
    if not (contract.get("priority_order") or ()):
        raise ValueError("oracle classifier must document priority order")


def _validate_oracle_wider_screen_interpretation_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("contract_kind") != "wider_confirmation_diagnostic_interpretation":
        raise ValueError("oracle wider-screen interpretation contract kind drifted")
    if not bool(contract.get("runtime_branch_classification_semantics_frozen")):
        raise ValueError("oracle wider-screen interpretation must freeze runtime branch semantics")
    if contract.get("runtime_branch_classification_field") != "branch_classification":
        raise ValueError("oracle wider-screen interpretation must read branch_classification")
    if list(contract.get("positive_interpretation_verdicts") or ()) != list(
        ORACLE_WIDER_SCREEN_INTERPRETATION_VERDICTS
    ):
        raise ValueError("oracle wider-screen interpretation positive verdicts drifted")
    if list(contract.get("negative_low_level_passthrough") or ()) != list(
        ORACLE_WIDER_SCREEN_NEGATIVE_PASSTHROUGH_BRANCHES
    ):
        raise ValueError("oracle wider-screen interpretation negative passthrough drifted")
    if list(contract.get("allowed_max_sampled_candidates") or ()) != list(
        ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES
    ):
        raise ValueError("oracle wider-screen interpretation allowed budgets drifted")
    selected_budget = int(contract.get("max_sampled_candidates", 0))
    if selected_budget not in ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES:
        raise ValueError("oracle wider-screen interpretation budget must be one of {8,32,64}")
    if float(contract.get("tier_max_seconds", 0.0)) != oracle_screen_budget_max_seconds(
        selected_budget
    ):
        raise ValueError("oracle wider-screen interpretation tier_max_seconds drifted")
    if contract.get("contrast_support_order_seeds") != list(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError("oracle wider-screen interpretation contrast seeds drifted")
    if int(contract.get("rank_position_index_base", -1)) != 0:
        raise ValueError("oracle wider-screen interpretation rank positions must stay zero-based")
    effectively_ok = contract.get("ranking_effectively_ok") or {}
    if not bool(effectively_ok.get("every_feasible_contrast_cell_required")):
        raise ValueError("oracle wider-screen OK band must require every feasible contrast cell")
    if not bool(effectively_ok.get("current_credit_rank_recovers_improvement_required")):
        raise ValueError("oracle wider-screen OK band must require rank_recovers=true")
    ok_rule = effectively_ok.get("oracle_best_current_rank_position_lt_rule") or {}
    if ok_rule.get("position_source") != "oracle_best_current_sampled_rank_position":
        raise ValueError("oracle wider-screen OK band must read sampled-order rank position")
    if int(ok_rule.get("absolute_floor_positions", -1)) != 5:
        raise ValueError("oracle wider-screen OK band absolute floor must stay 5")
    if float(ok_rule.get("fraction_of_sampled_candidate_count", -1.0)) != 0.10:
        raise ValueError("oracle wider-screen OK band sampled fraction must stay 0.10")
    if ok_rule.get("comparison") != "<":
        raise ValueError("oracle wider-screen OK band rank comparison must stay <")
    if not bool(ok_rule.get("uses_sampled_candidate_count")):
        raise ValueError("oracle wider-screen OK band must use sampled_candidate_count")
    expected_examples = {
        str(candidate_budget): oracle_screen_effectively_ok_rank_position_exclusive_bound(
            candidate_budget
        )
        for candidate_budget in ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES
    }
    if dict(effectively_ok.get("oracle_best_current_rank_position_lt_examples") or {}) != expected_examples:
        raise ValueError("oracle wider-screen OK band examples drifted")
    if float(effectively_ok.get("current_vs_oracle_top1_gap_ratio_max_inclusive", -1.0)) != 0.25:
        raise ValueError("oracle wider-screen OK band gap-ratio ceiling must stay 0.25")
    ranking_bad = contract.get("credit_ranking_bad") or {}
    if not bool(ranking_bad.get("any_feasible_contrast_cell_rank_recovers_false")):
        raise ValueError("oracle wider-screen BAD band must guard rank_recovers=false")
    if not bool(ranking_bad.get("widest_executed_budget_all_contrast_seeds_required")):
        raise ValueError("oracle wider-screen BAD band must require both widest-budget contrast seeds")
    if ranking_bad.get("rank_fraction_source") != "oracle_best_current_sampled_rank_position":
        raise ValueError("oracle wider-screen BAD band must read sampled-order rank fraction")
    if float(ranking_bad.get("oracle_best_current_rank_fraction_gt", -1.0)) != 0.25:
        raise ValueError("oracle wider-screen BAD band rank-fraction threshold must stay 0.25")
    if float(ranking_bad.get("current_vs_oracle_top1_gap_ratio_gt", -1.0)) != 0.50:
        raise ValueError("oracle wider-screen BAD band gap-ratio threshold must stay 0.50")
    ranking_suboptimal = contract.get("ranking_suboptimal") or {}
    if not bool(ranking_suboptimal.get("feasible_between_positive_and_negative_bands")):
        raise ValueError("oracle wider-screen suboptimal band must be the feasible-between-bands fallback")
    expected_next = {
        ORACLE_WIDER_SCREEN_VERDICT_RANKING_EFFECTIVELY_OK: (
            "ranking_not_the_bottleneck__reopen_scheduler_cap_backlog_multi_step"
        ),
        ORACLE_WIDER_SCREEN_VERDICT_RANKING_SUBOPTIMAL: (
            "credit_magnitude_or_rank_bin_calibration"
        ),
        ORACLE_WIDER_SCREEN_VERDICT_CREDIT_RANKING_BAD: (
            "update_law_or_credit_ranking_pivot"
        ),
    }
    if dict(contract.get("next_branch_by_interpretation") or {}) != expected_next:
        raise ValueError("oracle wider-screen next-branch mapping drifted")


def _validate_credit_ranking_pivot_compact_summary_schema(schema: Mapping[str, Any]) -> None:
    expected_fields = set(default_credit_ranking_pivot_compact_summary_schema()["allowed_fields"])
    if not bool(schema.get("compact_summary_only")):
        raise ValueError("credit-ranking pivot receipt must be compact-summary-only")
    if set(schema.get("allowed_fields") or ()) != expected_fields:
        raise ValueError("credit-ranking pivot compact summary allowed fields drifted")
    if set(schema.get("required_fields") or ()) != expected_fields:
        raise ValueError("credit-ranking pivot compact summary required fields drifted")
    for field in ("raw_per_proposal_arrays", "raw_candidate_scores", "raw_local_loss_deltas"):
        if bool(schema.get(field)):
            raise ValueError("credit-ranking pivot compact summary must reject raw proposal arrays")


def _validate_credit_ranking_pivot_measurement_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("contract_kind") != "credit_ranking_pivot_separating_measurement":
        raise ValueError("credit-ranking pivot contract kind drifted")
    if not bool(contract.get("candidate_generation_fixed")):
        raise ValueError("credit-ranking pivot contract must keep candidate generation fixed")
    if not bool(contract.get("same_candidate_set_required")):
        raise ValueError("credit-ranking pivot contract must require the same candidate set")
    if contract.get("contrast_support_order_seeds") != list(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError("credit-ranking pivot contract contrast seeds drifted")
    if int(contract.get("required_max_sampled_candidates", -1)) != PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES:
        raise ValueError("credit-ranking pivot contract must pin budget 32")
    if int(contract.get("top_k", -1)) != PIVOT_MEASUREMENT_TOP_K:
        raise ValueError("credit-ranking pivot contract top_k drifted")
    if list(contract.get("learner_available_ranking_input_fields") or ()) != [
        "candidate_id",
        "state_key",
        "flat_index",
        "vote_value",
        "abs_vote_value",
        "current_margin_abs",
        "current_rank_position",
        "tie_band_id",
    ]:
        raise ValueError("credit-ranking pivot learner-available input fields drifted")
    if list(contract.get("oracle_only_label_fields") or ()) != [
        "candidate_loss",
        "local_loss_delta",
        "regret_vs_oracle_top1_local_loss_delta",
        "oracle_best_sampled_rank_position",
    ]:
        raise ValueError("credit-ranking pivot oracle-only label fields drifted")
    score_family = contract.get("score_family") or {}
    if score_family.get("primary") != PIVOT_MEASUREMENT_PRIMARY_SCORE_ID:
        raise ValueError("credit-ranking pivot primary score id drifted")
    if list(score_family.get("ablations") or ()) != list(PIVOT_MEASUREMENT_ABLATION_SCORE_IDS):
        raise ValueError("credit-ranking pivot ablation score ids drifted")
    if score_family.get("decision_basis") != "primary_plus_ablation_report_no_post_hoc_best_of_many":
        raise ValueError("credit-ranking pivot score-family decision basis drifted")
    if score_family.get("hash_control_role") != "null_distribution_only":
        raise ValueError("credit-ranking pivot hash control must remain null-only")
    null_distribution = score_family.get("null_distribution") or {}
    if null_distribution.get("deterministic_hash_seeds") != list(PIVOT_MEASUREMENT_NULL_HASH_SEEDS):
        raise ValueError("credit-ranking pivot null hash seeds drifted")
    if null_distribution.get("random_permutation_seeds") != list(PIVOT_MEASUREMENT_NULL_RANDOM_SEEDS):
        raise ValueError("credit-ranking pivot null random seeds drifted")
    stage_a = contract.get("stage_a") or {}
    if list(stage_a.get("decision_metrics") or ()) != [
        "oracle_top_k_overlap_fraction",
        "oracle_top_k_regret_capture_ratio",
        "oracle_top_k_gap_ratio",
        "pairwise_auc",
        "null_median_auc_margin",
        "null_percentile",
    ]:
        raise ValueError("credit-ranking pivot Stage-A decision metrics drifted")
    if float(stage_a.get("pairwise_auc_non_predictive_max", -1.0)) != PIVOT_MEASUREMENT_AUC_NON_PREDICTIVE_MAX:
        raise ValueError("credit-ranking pivot non-predictive AUC threshold drifted")
    if float(stage_a.get("pairwise_auc_predictive_min", -1.0)) != PIVOT_MEASUREMENT_AUC_PREDICTIVE_MIN:
        raise ValueError("credit-ranking pivot predictive AUC threshold drifted")
    if float(
        stage_a.get("oracle_best_sampled_rank_position_poor_fraction", -1.0)
    ) != PIVOT_MEASUREMENT_POOR_RANK_FRACTION_MIN:
        raise ValueError("credit-ranking pivot poor-rank position fraction drifted")
    if stage_a.get("oracle_best_sampled_rank_position_poor_threshold_rule") != (
        "ceil(fraction * sampled_candidate_count)"
    ):
        raise ValueError("credit-ranking pivot poor-rank threshold rule drifted")
    null_guard = stage_a.get("null_guard") or {}
    if float(null_guard.get("median_auc_margin_min", -1.0)) != PIVOT_MEASUREMENT_NULL_AUC_MARGIN_MIN:
        raise ValueError("credit-ranking pivot null median margin drifted")
    if float(null_guard.get("percentile_min", -1.0)) != PIVOT_MEASUREMENT_NULL_PERCENTILE_MIN:
        raise ValueError("credit-ranking pivot null percentile threshold drifted")
    if not bool(null_guard.get("not_heavy_ci")):
        raise ValueError("credit-ranking pivot null guard must stay lightweight")
    if stage_a.get("non_predictive_branch_label") != BRANCH_PREREGISTERED_CHEAP_LEARNER_FEATURE_FAMILY_CANNOT_PREDICT_REGRET:
        raise ValueError("credit-ranking pivot non-predictive branch label drifted")
    if stage_a.get("predictive_seed_label") != PIVOT_MEASUREMENT_PREDICTIVE_SEED_LABEL:
        raise ValueError("credit-ranking pivot predictive seed label drifted")
    ambiguity = contract.get("tie_band_ambiguity_guard") or {}
    if ambiguity.get("score_family") != PIVOT_MEASUREMENT_PRIMARY_SCORE_ID:
        raise ValueError("credit-ranking pivot tie-band score family drifted")
    if not bool(ambiguity.get("oracle_best_in_band_required")):
        raise ValueError("credit-ranking pivot tie-band guard must require oracle-best in-band")
    if float(
        ambiguity.get("ambiguous_if_regret_spread_ratio_gt", -1.0)
    ) != PIVOT_MEASUREMENT_TIE_BAND_REGRET_SPREAD_RATIO_MAX:
        raise ValueError("credit-ranking pivot tie-band regret-spread threshold drifted")
    if ambiguity.get("ambiguous_branch_label") != BRANCH_MEASUREMENT_AMBIGUOUS_TIE_BAND_ALIASING:
        raise ValueError("credit-ranking pivot tie-band ambiguous branch drifted")
    stage_b = contract.get("stage_b_local_apply_magnitude_smoke") or {}
    if stage_b.get("contract_kind") != "local_apply_magnitude_smoke_only":
        raise ValueError("credit-ranking pivot Stage-B contract kind drifted")
    if list(stage_b.get("variant_ids") or ()) != [
        PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_TOP1,
        PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_PREFIX_CAP_1024,
        PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_CURRENT_SPEC,
    ]:
        raise ValueError("credit-ranking pivot Stage-B variant ids drifted")
    if not bool(stage_b.get("current_spec_is_non_definitive_without_live_full_cap")):
        raise ValueError("credit-ranking pivot Stage-B must remain non-definitive without live full cap")
    if not bool(stage_b.get("definitive_b_requires_follow_on")):
        raise ValueError("credit-ranking pivot Stage-B must require follow-on for definitive b")
    stage_c = contract.get("stage_c_follow_on") or {}
    if not bool(stage_c.get("separately_gated_required")):
        raise ValueError("credit-ranking pivot Stage-C must remain separately gated")
    if not bool(stage_c.get("broad_arm_forbidden")):
        raise ValueError("credit-ranking pivot Stage-C must forbid broad-arm expansion")
    if not bool(stage_c.get("no_acquisition_or_retention_claim_before_receipt")):
        raise ValueError("credit-ranking pivot Stage-C must forbid acquisition/retention claims")
    if list(contract.get("allowed_seed_local_labels") or ()) != [
        BRANCH_PREREGISTERED_CHEAP_LEARNER_FEATURE_FAMILY_CANNOT_PREDICT_REGRET,
        PIVOT_MEASUREMENT_PREDICTIVE_SEED_LABEL,
        BRANCH_MEASUREMENT_AMBIGUOUS_TIE_BAND_ALIASING,
        BRANCH_MEASUREMENT_AMBIGUOUS_NO_BRANCH,
    ]:
        raise ValueError("credit-ranking pivot allowed seed-local labels drifted")


def _validate_activation_credit_compact_summary_schema(schema: Mapping[str, Any]) -> None:
    expected_fields = set(default_activation_credit_compact_summary_schema()["allowed_fields"])
    if not bool(schema.get("compact_summary_only")):
        raise ValueError("activation-credit receipt must be compact-summary-only")
    if set(schema.get("allowed_fields") or ()) != expected_fields:
        raise ValueError("activation-credit compact summary allowed fields drifted")
    if set(schema.get("required_fields") or ()) != expected_fields:
        raise ValueError("activation-credit compact summary required fields drifted")
    for field in ("raw_per_proposal_arrays", "raw_candidate_scores", "raw_local_loss_deltas"):
        if bool(schema.get(field)):
            raise ValueError("activation-credit compact summary must reject raw proposal arrays")


def _validate_activation_credit_scale_smoke_compact_summary_schema(
    schema: Mapping[str, Any],
) -> None:
    expected_fields = set(
        default_activation_credit_scale_smoke_compact_summary_schema()["allowed_fields"]
    )
    if not bool(schema.get("compact_summary_only")):
        raise ValueError("activation-credit smoke receipt must be compact-summary-only")
    if set(schema.get("allowed_fields") or ()) != expected_fields:
        raise ValueError("activation-credit smoke compact summary allowed fields drifted")
    if set(schema.get("required_fields") or ()) != expected_fields:
        raise ValueError("activation-credit smoke compact summary required fields drifted")
    for field in ("raw_per_proposal_arrays", "raw_candidate_scores", "raw_local_loss_deltas"):
        if bool(schema.get(field)):
            raise ValueError(
                "activation-credit smoke compact summary must reject raw proposal arrays"
            )


def _validate_activation_credit_measurement_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("contract_kind") != "activation_credit_within_tie_band_measurement":
        raise ValueError("activation-credit contract kind drifted")
    if not bool(contract.get("candidate_generation_fixed")):
        raise ValueError("activation-credit contract must keep candidate generation fixed")
    if not bool(contract.get("same_candidate_set_required")):
        raise ValueError("activation-credit contract must require the same candidate set")
    if contract.get("contrast_support_order_seeds") != list(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError("activation-credit contract contrast seeds drifted")
    if int(contract.get("required_max_sampled_candidates", -1)) != (
        PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES
    ):
        raise ValueError("activation-credit contract must pin budget 32")
    if contract.get("required_eligible_scope") != ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE:
        raise ValueError("activation-credit contract eligible_scope drifted")
    if contract.get("target_tie_band_id") != ACTIVATION_CREDIT_TARGET_TIE_BAND_ID:
        raise ValueError("activation-credit contract target tie band drifted")
    if not bool(contract.get("oracle_best_in_target_tie_band_required")):
        raise ValueError(
            "activation-credit contract must require oracle-best inside the target tie band"
        )
    if int(contract.get("top_k", -1)) != PIVOT_MEASUREMENT_TOP_K:
        raise ValueError("activation-credit contract top_k drifted")
    if int(contract.get("rank_position_index_base", -1)) != 0:
        raise ValueError("activation-credit contract must keep zero-based rank positions")
    source = contract.get("activation_credit_source") or {}
    if source.get("capture_device_mode") != ACTIVATION_CREDIT_CAPTURE_DEVICE_MODE:
        raise ValueError("activation-credit source must pin device_resident capture")
    if source.get("grad_proxy_compute_mode") != ACTIVATION_CREDIT_GRAD_PROXY_COMPUTE_MODE:
        raise ValueError("activation-credit source must pin candidate-only gather")
    if not bool(source.get("candidate_only_gather_required")):
        raise ValueError("activation-credit source must require candidate-only gather")
    if not bool(source.get("no_extra_response_label_mask")):
        raise ValueError("activation-credit source must forbid extra response-label masking")
    if not bool(source.get("fixed_background_candidate_generation_allowed")):
        raise ValueError(
            "activation-credit source must preserve honest fixed-background candidate generation wording"
        )
    if not bool(source.get("fixed_background_must_be_labeled_non_hot_loop")):
        raise ValueError(
            "activation-credit source must label fixed-background candidate generation non-hot-loop"
        )
    if not bool(source.get("fp_proxy_transient_eval_only")):
        raise ValueError("activation-credit source must keep fp proxy transient-eval-only")
    if source.get("diag_fisher_formula") != (
        "sum((dL/dy_row)^2 * (x_col)^2) over the exact grad_proxy captures; empirical Fisher / Gauss-Newton diagonal surrogate only"
    ):
        raise ValueError("activation-credit diag_fisher formula drifted")
    if not bool(source.get("diag_fisher_reuses_grad_proxy_captures")):
        raise ValueError("activation-credit diag_fisher must reuse grad_proxy captures")
    if not bool(source.get("second_backward_forbidden")):
        raise ValueError("activation-credit diag_fisher must forbid a second backward")
    if list(source.get("transient_eval_only_scalar_fields") or ()) != [
        "grad_proxy",
        "diag_fisher",
        "taylor_benefit",
        "snr",
    ]:
        raise ValueError("activation-credit transient eval-only scalar fields drifted")
    if list(source.get("policy_facing_fields") or ()) != [
        ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_FIELD,
        ACTIVATION_CREDIT_SNR_Q5_FIELD,
        ACTIVATION_CREDIT_DIAG_FISHER_Q5_FIELD,
    ]:
        raise ValueError("activation-credit policy-facing fields drifted")
    construction = contract.get("feature_construction") or {}
    if construction.get("candidate_delta_sign_source") != (
        "sign(q_after_one_flip[flat_index] - current_q_level)"
    ):
        raise ValueError("activation-credit candidate_delta_sign source drifted")
    if construction.get("candidate_delta_weight_source") != (
        "((q_after_one_flip[flat_index] - current_q_level) * frozen_scale_scalar)"
    ):
        raise ValueError("activation-credit candidate_delta_weight source drifted")
    if not bool(construction.get("candidate_delta_weight_effective_weight_space")):
        raise ValueError("activation-credit candidate_delta_weight must stay in effective-weight space")
    if not bool(construction.get("candidate_delta_sign_zero_invalidates_row")):
        raise ValueError("activation-credit zero candidate_delta_sign must invalidate the row")
    if construction.get("diag_fisher_surrogate_kind") != "empirical_fisher_gauss_newton_diagonal":
        raise ValueError("activation-credit diag_fisher surrogate kind drifted")
    if construction.get("taylor_benefit_formula") != (
        "-grad_proxy * candidate_delta_weight - 0.5 * diag_fisher * candidate_delta_weight^2"
    ):
        raise ValueError("activation-credit taylor_benefit formula drifted")
    if construction.get("snr_formula") != "abs(grad_proxy) / sqrt(diag_fisher + second_order_snr_eps)":
        raise ValueError("activation-credit snr formula drifted")
    if float(construction.get("second_order_snr_eps", -1.0)) != ACTIVATION_CREDIT_SECOND_ORDER_SNR_EPS:
        raise ValueError("activation-credit second-order snr eps drifted")
    if int(construction.get("q5_bin_count", -1)) != ACTIVATION_CREDIT_MAGNITUDE_Q5_BIN_COUNT:
        raise ValueError("activation-credit q5 bin count drifted")
    if construction.get("q5_strategy") != "equal_frequency_quintiles_over_target_band_scalar_values":
        raise ValueError("activation-credit q5 strategy drifted")
    if construction.get("q5_value_field_by_family_id") != {
        ACTIVATION_CREDIT_PRIMARY_FAMILY_ID: "taylor_benefit",
        ACTIVATION_CREDIT_SNR_Q5_ABLATION_FAMILY_ID: "snr",
        ACTIVATION_CREDIT_DIAG_FISHER_Q5_ABLATION_FAMILY_ID: "diag_fisher",
    }:
        raise ValueError("activation-credit q5 value fields drifted")
    if construction.get("q5_output_field_by_family_id") != {
        ACTIVATION_CREDIT_PRIMARY_FAMILY_ID: ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_FIELD,
        ACTIVATION_CREDIT_SNR_Q5_ABLATION_FAMILY_ID: ACTIVATION_CREDIT_SNR_Q5_FIELD,
        ACTIVATION_CREDIT_DIAG_FISHER_Q5_ABLATION_FAMILY_ID: ACTIVATION_CREDIT_DIAG_FISHER_Q5_FIELD,
    }:
        raise ValueError("activation-credit q5 output fields drifted")
    if int(construction.get("q5_min_bucket_size", -1)) != ACTIVATION_CREDIT_MAGNITUDE_Q5_MIN_BUCKET_SIZE:
        raise ValueError("activation-credit q5 min bucket size drifted")
    if not bool(construction.get("q5_ties_force_ambiguous")):
        raise ValueError("activation-credit q5 tie guard drifted")
    if not bool(construction.get("q5_singleton_buckets_forbidden")):
        raise ValueError("activation-credit q5 singleton guard drifted")
    if construction.get("degenerate_branch_label") != BRANCH_ACTIVATION_CREDIT_AMBIGUOUS_NO_BRANCH:
        raise ValueError("activation-credit degenerate branch label drifted")
    learner_fields = [
        "candidate_id",
        "state_key",
        "flat_index",
        "vote_value",
        "current_margin_abs",
        "current_rank_position",
        "tie_band_id",
        "transition_class",
        "candidate_delta_sign",
        "candidate_delta_weight",
        ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_FIELD,
        ACTIVATION_CREDIT_SNR_Q5_FIELD,
        ACTIVATION_CREDIT_DIAG_FISHER_Q5_FIELD,
        "topology_row_block_128",
        "activation_feature_valid",
    ]
    if list(contract.get("learner_available_ranking_input_fields") or ()) != learner_fields:
        raise ValueError("activation-credit learner-available input fields drifted")
    provenance = contract.get("learner_available_field_provenance") or {}
    if set(provenance.keys()) != set(learner_fields):
        raise ValueError("activation-credit learner-available provenance keys drifted")
    if not all(isinstance(provenance[field], str) and provenance[field] for field in learner_fields):
        raise ValueError("activation-credit learner-available provenance must stay source-backed text")
    if list(contract.get("oracle_only_label_fields") or ()) != [
        "candidate_loss",
        "local_loss_delta",
        "regret_vs_target_tie_band_oracle_top1_local_loss_delta",
        "target_tie_band_oracle_best_candidate_id",
    ]:
        raise ValueError("activation-credit oracle-only label fields drifted")
    family = contract.get("family_discriminator") or {}
    if family.get("primary") != ACTIVATION_CREDIT_PRIMARY_FAMILY_ID:
        raise ValueError("activation-credit primary family id drifted")
    if list(family.get("ablations") or ()) != list(ACTIVATION_CREDIT_ABLATION_FAMILY_IDS):
        raise ValueError("activation-credit ablation family ids drifted")
    if family.get("topology_control") != ACTIVATION_CREDIT_TOPOLOGY_CONTROL_FAMILY_ID:
        raise ValueError("activation-credit topology control family drifted")
    fields_by_family = family.get("fields_by_family_id") or {}
    if fields_by_family.get(ACTIVATION_CREDIT_PRIMARY_FAMILY_ID) != [
        ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_FIELD,
    ]:
        raise ValueError("activation-credit second-order q5 primary family drifted")
    if fields_by_family.get(ACTIVATION_CREDIT_SNR_Q5_ABLATION_FAMILY_ID) != [
        ACTIVATION_CREDIT_SNR_Q5_FIELD,
    ]:
        raise ValueError("activation-credit snr q5 ablation drifted")
    if fields_by_family.get(ACTIVATION_CREDIT_DIAG_FISHER_Q5_ABLATION_FAMILY_ID) != [
        ACTIVATION_CREDIT_DIAG_FISHER_Q5_FIELD,
    ]:
        raise ValueError("activation-credit diag_fisher q5 ablation drifted")
    if fields_by_family.get(ACTIVATION_CREDIT_TOPOLOGY_CONTROL_FAMILY_ID) != [
        "topology_row_block_128"
    ]:
        raise ValueError("activation-credit topology control fields drifted")
    if family.get("decision_basis") != "primary_plus_ablation_report_no_post_hoc_best_of_many":
        raise ValueError("activation-credit family decision basis drifted")
    if family.get("hash_control_role") != "null_distribution_only":
        raise ValueError("activation-credit hash control must remain null-only")
    null_distribution = family.get("null_distribution") or {}
    if null_distribution.get("deterministic_hash_seeds") != list(PIVOT_MEASUREMENT_NULL_HASH_SEEDS):
        raise ValueError("activation-credit null hash seeds drifted")
    if not bool(null_distribution.get("matched_cardinality_bucket_partitions")):
        raise ValueError("activation-credit null distribution must keep matched partitions")
    if null_distribution.get("smaller_bucket_fraction_guard_field") != (
        "matched_hash_null_fraction_gte_observed_bucket_fraction"
    ):
        raise ValueError("activation-credit smaller-bucket guard field drifted")
    if null_distribution.get("smaller_bucket_fraction_guard_comparison") != "fraction_gte_observed":
        raise ValueError("activation-credit smaller-bucket guard comparison drifted")
    if null_distribution.get("larger_regret_capture_guard_field") != (
        "matched_hash_null_fraction_lte_observed_regret_capture_ratio"
    ):
        raise ValueError("activation-credit regret-capture guard field drifted")
    if null_distribution.get("larger_regret_capture_guard_comparison") != "fraction_lte_observed":
        raise ValueError("activation-credit regret-capture guard comparison drifted")
    decision = contract.get("within_band_decision") or {}
    if list(decision.get("decision_metrics") or ()) != [
        "oracle_best_bucket_fraction",
        "oracle_best_bucket_regret_spread_ratio",
        "oracle_best_bucket_regret_capture_ratio",
        "oracle_best_bucket_top_k_capture_fraction",
        "matched_hash_null_fraction_gte_observed_bucket_fraction",
        "matched_hash_null_fraction_lte_observed_regret_capture_ratio",
        "within_band_pairwise_auc_report_only",
    ]:
        raise ValueError("activation-credit decision metrics drifted")
    if decision.get("target_tie_band_missing_branch_label") != BRANCH_ACTIVATION_CREDIT_AMBIGUOUS_NO_BRANCH:
        raise ValueError("activation-credit target-tie-band missing branch drifted")
    if decision.get("predictive_branch_label") != BRANCH_ACTIVATION_CREDIT_CANDIDATE_SIGNAL:
        raise ValueError("activation-credit predictive branch label drifted")
    if decision.get("predictive_family_id") != ACTIVATION_CREDIT_PRIMARY_FAMILY_ID:
        raise ValueError("activation-credit predictive family id drifted")
    if float(decision.get("predictive_bucket_fraction_max", -1.0)) != ACTIVATION_CREDIT_PREDICTIVE_BUCKET_FRACTION_MAX:
        raise ValueError("activation-credit predictive bucket fraction drifted")
    if float(decision.get("predictive_regret_spread_ratio_max", -1.0)) != ACTIVATION_CREDIT_PREDICTIVE_REGRET_SPREAD_RATIO_MAX:
        raise ValueError("activation-credit predictive regret spread threshold drifted")
    if float(decision.get("predictive_regret_capture_ratio_min", -1.0)) != ACTIVATION_CREDIT_PREDICTIVE_REGRET_CAPTURE_RATIO_MIN:
        raise ValueError("activation-credit predictive regret capture threshold drifted")
    if float(decision.get("predictive_matched_hash_guard_min", -1.0)) != ACTIVATION_CREDIT_MATCHED_HASH_SIGNAL_MIN:
        raise ValueError("activation-credit predictive matched-hash threshold drifted")
    if decision.get("fail_closed_branch_label") != (
        BRANCH_ACTIVATION_CREDIT_MISSING_SIGNAL_DEEPER_THAN_FIRST_ORDER_CREDIT_STORAGE
    ):
        raise ValueError("activation-credit fail-closed branch label drifted")
    if not bool(decision.get("fail_closed_requires_all_preregistered_families")):
        raise ValueError("activation-credit fail-closed rule must require all preregistered families")
    if float(decision.get("fail_closed_bucket_fraction_gt", -1.0)) != ACTIVATION_CREDIT_FAIL_CLOSED_BUCKET_FRACTION_GT:
        raise ValueError("activation-credit fail-closed bucket fraction drifted")
    if float(decision.get("fail_closed_regret_spread_ratio_gt", -1.0)) != ACTIVATION_CREDIT_FAIL_CLOSED_REGRET_SPREAD_RATIO_GT:
        raise ValueError("activation-credit fail-closed regret spread threshold drifted")
    if float(decision.get("fail_closed_matched_hash_signal_min", -1.0)) != ACTIVATION_CREDIT_MATCHED_HASH_SIGNAL_MIN:
        raise ValueError("activation-credit fail-closed matched-hash threshold drifted")
    if decision.get("topology_control_family_id") != ACTIVATION_CREDIT_TOPOLOGY_CONTROL_FAMILY_ID:
        raise ValueError("activation-credit topology control family id drifted")
    if not bool(decision.get("topology_control_positive_forces_ambiguous")):
        raise ValueError("activation-credit topology control must force ambiguous on positive")
    if decision.get("ambiguous_branch_label") != BRANCH_ACTIVATION_CREDIT_AMBIGUOUS_NO_BRANCH:
        raise ValueError("activation-credit ambiguous branch label drifted")
    fragmentation = contract.get("fragmentation_audit") or {}
    if not bool(fragmentation.get("bucket_cardinality_histogram_required")):
        raise ValueError("activation-credit fragmentation audit must require bucket histogram")
    if not bool(fragmentation.get("singleton_bucket_count_required")):
        raise ValueError("activation-credit fragmentation audit must require singleton bucket count")
    if not bool(fragmentation.get("candidate_delta_weight_support_required")):
        raise ValueError("activation-credit fragmentation audit must require candidate_delta_weight support")
    if fragmentation.get("q5_primary_prefix") != ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_PREFIX:
        raise ValueError("activation-credit fragmentation audit primary q5 prefix drifted")
    if list(fragmentation.get("q5_report_only_prefixes") or ()) != [
        ACTIVATION_CREDIT_SNR_Q5_PREFIX,
        ACTIVATION_CREDIT_DIAG_FISHER_Q5_PREFIX,
    ]:
        raise ValueError("activation-credit fragmentation audit report-only q5 prefixes drifted")
    if int(fragmentation.get("q5_min_bucket_candidate_count_required", -1)) != ACTIVATION_CREDIT_MAGNITUDE_Q5_MIN_BUCKET_SIZE:
        raise ValueError("activation-credit fragmentation audit q5 min bucket size drifted")
    if not bool(fragmentation.get("q5_singleton_buckets_forbidden")):
        raise ValueError("activation-credit fragmentation audit q5 singleton guard drifted")
    if not bool(fragmentation.get("q5_ties_force_ambiguous")):
        raise ValueError("activation-credit fragmentation audit q5 tie guard drifted")
    if fragmentation.get("q5_guard_failure_branch_label") != BRANCH_ACTIVATION_CREDIT_AMBIGUOUS_NO_BRANCH:
        raise ValueError("activation-credit fragmentation audit q5 guard branch drifted")
    smoke_gate = contract.get("scale_smoke_gate") or {}
    if not bool(smoke_gate.get("required_before_full_eval")):
        raise ValueError("activation-credit full eval must require the de-risk smoke first")
    if smoke_gate.get("smoke_launch_bundle_packet_kind") != (
        ACTIVATION_CREDIT_SCALE_SMOKE_LAUNCH_BUNDLE_PACKET_KIND
    ):
        raise ValueError("activation-credit smoke launch bundle kind drifted")
    if int(smoke_gate.get("required_max_sampled_candidates", -1)) != ACTIVATION_CREDIT_SMOKE_MAX_SAMPLED_CANDIDATES:
        raise ValueError("activation-credit smoke gate must pin budget 8")
    if int(smoke_gate.get("required_batch_size", -1)) != ACTIVATION_CREDIT_SMOKE_BATCH_SIZE:
        raise ValueError("activation-credit smoke gate must pin batch size 4")
    if not bool(smoke_gate.get("smoke_branch_classification_must_be_null")):
        raise ValueError("activation-credit smoke gate must keep branch_classification=null")
    if not bool(smoke_gate.get("required_grad_proxy_candidate_count_positive")):
        raise ValueError("activation-credit smoke gate must require grad_proxy_candidate_count > 0")
    fresh = contract.get("fresh_confirmation_gate") or {}
    if int(fresh.get("required_seed_before_persistent_followup", -1)) != ACTIVATION_CREDIT_FRESH_CONFIRMATION_SEED:
        raise ValueError("activation-credit fresh confirmation seed drifted")
    if fresh.get("two_seed_positive_label") != BRANCH_ACTIVATION_CREDIT_CANDIDATE_SIGNAL:
        raise ValueError("activation-credit two-seed positive label drifted")
    if not bool(fresh.get("persistent_followup_forbidden_before_fresh_confirmation")):
        raise ValueError("activation-credit must forbid persistent followup before fresh confirmation")
    if list(contract.get("allowed_seed_local_labels") or ()) != list(ACTIVATION_CREDIT_BRANCHES):
        raise ValueError("activation-credit allowed seed-local labels drifted")


def _validate_activation_credit_scale_smoke_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("contract_kind") != "activation_credit_scale_smoke_only":
        raise ValueError("activation-credit smoke contract kind drifted")
    if not bool(contract.get("same_candidate_set_required")):
        raise ValueError("activation-credit smoke contract must require the same candidate set")
    if contract.get("contrast_support_order_seeds") != list(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError("activation-credit smoke contract contrast seeds drifted")
    if int(contract.get("required_max_sampled_candidates", -1)) != ACTIVATION_CREDIT_SMOKE_MAX_SAMPLED_CANDIDATES:
        raise ValueError("activation-credit smoke contract must pin budget 8")
    if int(contract.get("required_batch_size", -1)) != ACTIVATION_CREDIT_SMOKE_BATCH_SIZE:
        raise ValueError("activation-credit smoke contract must pin batch size 4")
    if contract.get("required_eligible_scope") != ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE:
        raise ValueError("activation-credit smoke contract eligible_scope drifted")
    if contract.get("target_tie_band_id") != ACTIVATION_CREDIT_TARGET_TIE_BAND_ID:
        raise ValueError("activation-credit smoke contract target tie band drifted")
    if contract.get("capture_device_mode") != ACTIVATION_CREDIT_CAPTURE_DEVICE_MODE:
        raise ValueError("activation-credit smoke contract must pin device_resident capture")
    if contract.get("grad_proxy_compute_mode") != ACTIVATION_CREDIT_GRAD_PROXY_COMPUTE_MODE:
        raise ValueError("activation-credit smoke contract must pin candidate-only gather")
    if not bool(contract.get("no_extra_response_label_mask")):
        raise ValueError("activation-credit smoke contract must forbid extra response-label masking")
    if not bool(contract.get("fixed_background_must_be_labeled_non_hot_loop")):
        raise ValueError(
            "activation-credit smoke contract must label fixed-background candidate generation non-hot-loop"
        )
    if not bool(contract.get("smoke_branch_classification_must_be_null")):
        raise ValueError("activation-credit smoke contract must keep branch_classification=null")
    if not bool(contract.get("required_grad_proxy_candidate_count_positive")):
        raise ValueError("activation-credit smoke contract must require grad_proxy_candidate_count > 0")
    if not bool(contract.get("policy_verdict_forbidden")):
        raise ValueError("activation-credit smoke contract must forbid policy verdicts")
    _validate_activation_credit_scale_smoke_outcome_contract(
        contract.get("occupancy_outcome_contract") or {},
        label="activation-credit smoke contract occupancy outcome contract",
    )
    if list(contract.get("allowed_policy_facing_fields") or ()) != [
        "credit_sign",
        "credit_magnitude_bin",
        "signed_alignment",
    ]:
        raise ValueError("activation-credit smoke policy-facing fields drifted")


def _validate_activation_credit_scale_smoke_outcome_contract(
    contract: Mapping[str, Any],
    *,
    label: str,
) -> None:
    required_fields = list(ACTIVATION_CREDIT_SMOKE_REQUIRED_PER_SEED_RECEIPT_FIELDS)
    if list(contract.get("per_seed_receipt_fields_required") or ()) != required_fields:
        raise ValueError(
            f"{label} must require per-seed target_band_candidate_count and grad_proxy_candidate_count"
        )
    if list(contract.get("pass_requires_any_seed_positive_fields") or ()) != required_fields:
        raise ValueError(
            f"{label} must require at least one seed with positive target_band_candidate_count and grad_proxy_candidate_count"
        )
    if contract.get("per_seed_target_band_zero_label") != ACTIVATION_CREDIT_SMOKE_OCCUPANCY_MISS_LABEL:
        raise ValueError(f"{label} occupancy_miss label drifted")
    if bool(contract.get("per_seed_target_band_zero_is_code_failure")):
        raise ValueError(f"{label} must not treat occupancy_miss as a code failure")
    if contract.get("all_seeds_target_band_zero_outcome") != (
        ACTIVATION_CREDIT_SMOKE_INCONCLUSIVE_GATHER_TIMING_ONLY
    ):
        raise ValueError(
            f"{label} must classify both-seed occupancy misses as inconclusive_on_gather_timing_only"
        )
    if list(contract.get("all_seeds_target_band_zero_reprobe_budgets") or ()) != list(
        ACTIVATION_CREDIT_SMOKE_RESMOKE_BUDGETS
    ):
        raise ValueError(f"{label} must require minimal re-smoke budgets [12, 16]")
    if not bool(contract.get("all_seeds_target_band_zero_blocks_full_read")):
        raise ValueError(
            f"{label} must block the full read after an inconclusive timing-only smoke"
        )
    if contract.get("target_band_positive_grad_proxy_zero_outcome") != (
        ACTIVATION_CREDIT_SMOKE_GATHER_FAILURE_REPAIR_SIGNAL
    ):
        raise ValueError(
            f"{label} must classify target-band-positive/grad-proxy-zero as a repair signal"
        )
    if bool(contract.get("target_band_positive_grad_proxy_zero_is_occupancy_miss")):
        raise ValueError(
            f"{label} must not treat target-band-positive/grad-proxy-zero as occupancy_miss"
        )
    if not bool(contract.get("target_band_positive_grad_proxy_zero_is_code_failure")):
        raise ValueError(
            f"{label} must treat target-band-positive/grad-proxy-zero as a code failure"
        )


def _validate_within_tie_band_discriminator_compact_summary_schema(
    schema: Mapping[str, Any],
) -> None:
    expected_fields = set(
        default_within_tie_band_discriminator_compact_summary_schema()["allowed_fields"]
    )
    if not bool(schema.get("compact_summary_only")):
        raise ValueError("within-tie-band receipt must be compact-summary-only")
    if set(schema.get("allowed_fields") or ()) != expected_fields:
        raise ValueError("within-tie-band compact summary allowed fields drifted")
    if set(schema.get("required_fields") or ()) != expected_fields:
        raise ValueError("within-tie-band compact summary required fields drifted")
    for field in ("raw_per_proposal_arrays", "raw_candidate_scores", "raw_local_loss_deltas"):
        if bool(schema.get(field)):
            raise ValueError("within-tie-band compact summary must reject raw proposal arrays")


def _validate_within_tie_band_discriminator_measurement_contract(
    contract: Mapping[str, Any],
) -> None:
    if contract.get("contract_kind") != "within_tie_band_learner_available_discriminator_measurement":
        raise ValueError("within-tie-band contract kind drifted")
    if not bool(contract.get("candidate_generation_fixed")):
        raise ValueError("within-tie-band contract must keep candidate generation fixed")
    if not bool(contract.get("same_candidate_set_required")):
        raise ValueError("within-tie-band contract must require the same candidate set")
    if contract.get("contrast_support_order_seeds") != list(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError("within-tie-band contract contrast seeds drifted")
    if int(contract.get("required_max_sampled_candidates", -1)) != PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES:
        raise ValueError("within-tie-band contract must pin budget 32")
    if contract.get("target_tie_band_id") != WITHIN_TIE_BAND_TARGET_TIE_BAND_ID:
        raise ValueError("within-tie-band contract target tie band drifted")
    if not bool(contract.get("oracle_best_in_target_tie_band_required")):
        raise ValueError("within-tie-band contract must require oracle-best inside the target tie band")
    if int(contract.get("top_k", -1)) != PIVOT_MEASUREMENT_TOP_K:
        raise ValueError("within-tie-band contract top_k drifted")
    if int(contract.get("rank_position_index_base", -1)) != 0:
        raise ValueError("within-tie-band contract must keep zero-based rank positions")
    learner_fields = [
        "candidate_id",
        "state_key",
        "flat_index",
        "vote_value",
        "abs_vote_value",
        "current_margin_abs",
        "current_rank_position",
        "tie_band_id",
        "current_q_level",
        "pre_accumulator_i16",
        "new_acc_i32_signed",
        "proposal_direction",
        "threshold_residual_signed",
        "proximity_to_threshold",
        "tensor_numel",
        "state_candidate_count",
        "current_rank_fraction_within_state",
        "current_rank_quartile_within_state",
        "flat_index_fraction",
        "flat_index_quartile",
        "transition_class",
    ]
    if list(contract.get("learner_available_ranking_input_fields") or ()) != learner_fields:
        raise ValueError("within-tie-band learner-available input fields drifted")
    provenance = contract.get("learner_available_field_provenance") or {}
    if set(provenance.keys()) != set(learner_fields):
        raise ValueError("within-tie-band learner-available provenance keys drifted")
    if not all(isinstance(provenance[field], str) and provenance[field] for field in learner_fields):
        raise ValueError("within-tie-band learner-available provenance must stay source-backed text")
    if list(contract.get("oracle_only_label_fields") or ()) != [
        "candidate_loss",
        "local_loss_delta",
        "regret_vs_target_tie_band_oracle_top1_local_loss_delta",
        "target_tie_band_oracle_best_candidate_id",
    ]:
        raise ValueError("within-tie-band oracle-only label fields drifted")
    family = contract.get("family_discriminator") or {}
    if family.get("primary") != WITHIN_TIE_BAND_PRIMARY_FAMILY_ID:
        raise ValueError("within-tie-band primary family id drifted")
    if list(family.get("ablations") or ()) != list(WITHIN_TIE_BAND_ABLATION_FAMILY_IDS):
        raise ValueError("within-tie-band ablation family ids drifted")
    expected_family_fields = {
        WITHIN_TIE_BAND_PRIMARY_FAMILY_ID: [
            "state_key",
            "transition_class",
            "current_rank_quartile_within_state",
        ],
        "F_transition_rankq": [
            "transition_class",
            "current_rank_quartile_within_state",
        ],
        "F_state_transition": [
            "state_key",
            "transition_class",
        ],
        "F_transition_only": [
            "transition_class",
        ],
        "F_rankq_only": [
            "current_rank_quartile_within_state",
        ],
        "F_flatq_only": [
            "flat_index_quartile",
        ],
    }
    if dict(family.get("fields_by_family_id") or {}) != expected_family_fields:
        raise ValueError("within-tie-band family field mapping drifted")
    if family.get("decision_basis") != "primary_plus_ablation_report_no_post_hoc_best_of_many":
        raise ValueError("within-tie-band decision basis drifted")
    if family.get("hash_control_role") != "null_distribution_only":
        raise ValueError("within-tie-band hash control must remain null-only")
    null_distribution = family.get("null_distribution") or {}
    if null_distribution.get("deterministic_hash_seeds") != list(PIVOT_MEASUREMENT_NULL_HASH_SEEDS):
        raise ValueError("within-tie-band deterministic-hash seeds drifted")
    if not bool(null_distribution.get("matched_cardinality_bucket_partitions")):
        raise ValueError("within-tie-band null control must keep matched-cardinality partitions")
    if null_distribution.get("smaller_bucket_fraction_guard_field") != (
        "matched_hash_null_fraction_gte_observed_bucket_fraction"
    ):
        raise ValueError("within-tie-band smaller-bucket guard field drifted")
    if null_distribution.get("smaller_bucket_fraction_guard_comparison") != "fraction_gte_observed":
        raise ValueError("within-tie-band smaller-bucket guard comparator drifted")
    if null_distribution.get("larger_regret_capture_guard_field") != (
        "matched_hash_null_fraction_lte_observed_regret_capture_ratio"
    ):
        raise ValueError("within-tie-band regret-capture guard field drifted")
    if null_distribution.get("larger_regret_capture_guard_comparison") != "fraction_lte_observed":
        raise ValueError("within-tie-band regret-capture guard comparator drifted")
    decision = contract.get("within_band_decision") or {}
    if list(decision.get("decision_metrics") or ()) != [
        "oracle_best_bucket_fraction",
        "oracle_best_bucket_regret_spread_ratio",
        "oracle_best_bucket_regret_capture_ratio",
        "oracle_best_bucket_top_k_capture_fraction",
        "matched_hash_null_fraction_gte_observed_bucket_fraction",
        "matched_hash_null_fraction_lte_observed_regret_capture_ratio",
        "within_band_pairwise_auc_report_only",
    ]:
        raise ValueError("within-tie-band decision metrics drifted")
    if any("percentile" in str(metric) for metric in decision.get("decision_metrics") or ()):
        raise ValueError("within-tie-band decision metrics must encode comparator direction explicitly")
    if decision.get("target_tie_band_missing_branch_label") != BRANCH_WITHIN_TIE_BAND_AMBIGUOUS_NO_BRANCH:
        raise ValueError("within-tie-band target-band-missing branch drifted")
    if decision.get("predictive_branch_label") != BRANCH_WITHIN_TIE_BAND_LEARNER_FEATURES_SEPARATE_REGRET:
        raise ValueError("within-tie-band predictive branch label drifted")
    if decision.get("predictive_family_id") != WITHIN_TIE_BAND_PRIMARY_FAMILY_ID:
        raise ValueError("within-tie-band predictive family id drifted")
    if float(decision.get("predictive_bucket_fraction_max", -1.0)) != WITHIN_TIE_BAND_PREDICTIVE_BUCKET_FRACTION_MAX:
        raise ValueError("within-tie-band predictive bucket-fraction threshold drifted")
    if float(decision.get("predictive_regret_spread_ratio_max", -1.0)) != WITHIN_TIE_BAND_PREDICTIVE_REGRET_SPREAD_RATIO_MAX:
        raise ValueError("within-tie-band predictive regret-spread threshold drifted")
    if float(decision.get("predictive_regret_capture_ratio_min", -1.0)) != WITHIN_TIE_BAND_PREDICTIVE_REGRET_CAPTURE_RATIO_MIN:
        raise ValueError("within-tie-band predictive regret-capture threshold drifted")
    if float(decision.get("predictive_matched_hash_guard_min", -1.0)) != WITHIN_TIE_BAND_MATCHED_HASH_SIGNAL_MIN:
        raise ValueError("within-tie-band predictive matched-hash threshold drifted")
    if decision.get("fail_closed_branch_label") != BRANCH_WITHIN_TIE_BAND_NEEDS_NEW_LEARNER_STATE:
        raise ValueError("within-tie-band fail-closed branch label drifted")
    if not bool(decision.get("fail_closed_requires_all_preregistered_families")):
        raise ValueError("within-tie-band fail-closed branch must require all preregistered families")
    if float(decision.get("fail_closed_bucket_fraction_gt", -1.0)) != WITHIN_TIE_BAND_FAIL_CLOSED_BUCKET_FRACTION_GT:
        raise ValueError("within-tie-band fail-closed bucket-fraction threshold drifted")
    if float(decision.get("fail_closed_regret_spread_ratio_gt", -1.0)) != WITHIN_TIE_BAND_FAIL_CLOSED_REGRET_SPREAD_RATIO_GT:
        raise ValueError("within-tie-band fail-closed regret-spread threshold drifted")
    if float(decision.get("fail_closed_matched_hash_signal_min", -1.0)) != WITHIN_TIE_BAND_MATCHED_HASH_SIGNAL_MIN:
        raise ValueError("within-tie-band fail-closed matched-hash threshold drifted")
    if decision.get("ambiguous_branch_label") != BRANCH_WITHIN_TIE_BAND_AMBIGUOUS_NO_BRANCH:
        raise ValueError("within-tie-band ambiguous branch label drifted")
    fragmentation = contract.get("fragmentation_audit") or {}
    if not bool(fragmentation.get("bucket_cardinality_histogram_required")):
        raise ValueError("within-tie-band contract must require bucket cardinality histograms")
    if not bool(fragmentation.get("singleton_bucket_count_required")):
        raise ValueError("within-tie-band contract must require singleton bucket counts")
    if list(contract.get("allowed_seed_local_labels") or ()) != list(WITHIN_TIE_BAND_DISCRIMINATOR_BRANCHES):
        raise ValueError("within-tie-band allowed seed-local labels drifted")


def _validate_oracle_global_non_persistence(value: Any, *, path: str = "packet") -> None:
    for key, child in _walk_items(value):
        key_text = str(key)
        child_path = f"{path}.{key_text}"
        if key_text in _ORACLE_GLOBAL_FORBIDDEN_TRUE_KEYS and bool(child):
            raise ValueError(f"{child_path} violates oracle non-persistence boundary")
        if key_text in _ORACLE_GLOBAL_PT_PATH_KEYS and str(child).endswith(".pt"):
            raise ValueError(f"{child_path} must not target .pt artifacts")
        _validate_oracle_global_non_persistence(child, path=child_path)


def validate_candidate_set_viability_oracle_screen_packet(packet: Mapping[str, Any]) -> None:
    if packet.get("schema_version") != OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION:
        raise ValueError("unsupported optimizer update-law oracle-screen packet schema")
    if packet.get("diagnostic_class") != DIAGNOSTIC_CLASS_PRE_FULL_STACK:
        raise ValueError("oracle-screen packet must be pre_full_stack_diagnostic")
    _validate_author_only_fields(
        packet,
        expected_packet_kind=ORACLE_SCREEN_PACKET_KIND,
        label="author-only oracle-screen packet",
    )
    for field in _READINESS_FORBIDDEN_TRUE_FIELDS:
        _require_false(packet, field)
    if bool(packet.get("carrier_claim")) or bool(packet.get("q_sidecar_carrier_claim")):
        raise ValueError("oracle-screen packet must not make a carrier claim")
    if bool(packet.get("qacc_kernelized")):
        raise ValueError("oracle-screen packet must keep qacc_kernelized=false")
    if not bool(packet.get("optimizer_credit_state_science_dependent")):
        raise ValueError("oracle-screen packet must keep optimizer_credit_state science-dependent")
    if not bool(packet.get("same_candidate_set_required")):
        raise ValueError("oracle-screen packet must require the same candidate set")
    if bool(packet.get("oracle_state_survives_into_learner")):
        raise ValueError("oracle state must not survive into learner fields")
    _reject_raw_arrays(packet)
    _validate_oracle_global_non_persistence(packet)
    _validate_oracle_screen_arms(packet.get("arms") or ())
    _validate_oracle_seed_order_contract(packet.get("seed_order_contract") or {})
    _validate_oracle_feasibility_budget(packet.get("oracle_feasibility_budget") or {})
    _validate_oracle_non_persistence(packet.get("oracle_non_persistence_contract") or {})
    _validate_oracle_compact_summary_schema(packet.get("compact_summary_schema") or {})
    _validate_oracle_classifier_contract(packet.get("classifier_contract") or {})
    _validate_oracle_wider_screen_interpretation_contract(
        packet.get("wider_screen_interpretation_contract") or {}
    )
    fallback = packet.get("fallback") or {}
    if fallback.get("fallback_mode") != "decile_only_concordance":
        raise ValueError("oracle fallback must be decile-only concordance")
    if bool(fallback.get("oracle_applied_arm_allowed")):
        raise ValueError("decile-only fallback must not allow oracle-applied arm")
    artifact_policy = packet.get("artifact_policy") or {}
    if not bool(artifact_policy.get("compact_json_ndjson_only")):
        raise ValueError("oracle artifact policy must require compact JSON/NDJSON")
    if bool(artifact_policy.get("pt_writes_allowed")):
        raise ValueError("oracle artifact policy must reject .pt writes")
    oracle_artifact_path = artifact_policy.get("oracle_artifact_path")
    if oracle_artifact_path is not None and str(oracle_artifact_path).endswith(".pt"):
        raise ValueError("oracle artifact path must not target .pt artifacts")


def _validate_oracle_screen_command_record(command: Mapping[str, Any]) -> None:
    missing = [field for field in _COMMAND_REQUIRED_FIELDS if field not in command]
    if missing:
        raise ValueError(f"oracle-screen command record missing required fields: {missing}")
    if str(command.get("mode")) != "oracle_screen_n20":
        raise ValueError("oracle-screen command mode must be oracle_screen_n20")
    if str(command.get("phase_role")) != "candidate_set_viability_oracle_screen":
        raise ValueError("oracle-screen command phase_role drifted")
    if str(command.get("oracle_screen_mode")) != "candidate_set_viability":
        raise ValueError("oracle-screen command must pin candidate_set_viability mode")
    if not bool(command.get("same_candidate_set_required")):
        raise ValueError("oracle-screen command must require same candidate set")
    if int(command.get("support_order_seed", -1)) not in ORACLE_SCREEN_CONTRAST_SEEDS:
        raise ValueError("oracle-screen command support_order_seed must be one of the contrast seeds")
    if str(command.get("seed_label")) != _support_order_seed_label(int(command["support_order_seed"])):
        raise ValueError("oracle-screen command seed_label drifted from support_order_seed")
    if int(command.get("screen_rows", -1)) != ORACLE_SCREEN_N20_ROWS:
        raise ValueError("oracle-screen command screen_rows must be 20")
    if int(command.get("n_rows", -1)) != ORACLE_SCREEN_N20_ROWS:
        raise ValueError("oracle-screen command n_rows must equal screen_rows")
    if int(command.get("steps_requested", -1)) != 1:
        raise ValueError("oracle-screen command steps_requested must be 1")
    if command.get("steps_source") != "fixed_single_support_batch_oracle_screen":
        raise ValueError("oracle-screen command steps_source drifted")
    if int(command.get("batch_size", -1)) != ORACLE_SCREEN_N20_ROWS:
        raise ValueError("oracle-screen command batch_size must equal N=20 screen rows")
    if int(command.get("curriculum_seed", -1)) != STEP6_CURRICULUM_SEED:
        raise ValueError("oracle-screen command curriculum_seed must stay pinned to 17")
    budget = int(command.get("max_sampled_candidates", -1))
    if budget not in ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES:
        raise ValueError("oracle-screen command max_sampled_candidates must be one of {8,32,64}")
    if float(command.get("oracle_max_seconds", -1.0)) != oracle_screen_budget_max_seconds(
        budget
    ):
        raise ValueError("oracle-screen command oracle_max_seconds drifted")
    if int(command.get("max_abs_per_tensor", -1)) != STEP3_BASELINE_MAX_ABS_PER_TENSOR:
        raise ValueError("oracle-screen command max_abs_per_tensor must stay at the baseline cap")
    if float(command.get("fraction_per_tensor", -1.0)) != STEP3_FRACTION_PER_TENSOR:
        raise ValueError("oracle-screen command fraction_per_tensor must stay at 1.0")
    if command.get("global_cap_contract") != "off":
        raise ValueError("oracle-screen command must keep global cap off")
    env = command.get("env")
    if not isinstance(env, Mapping):
        raise ValueError("oracle-screen command env must be a mapping")
    if env.get("HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE") != "1":
        raise ValueError("oracle-screen command env missing HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE=1")
    if env.get("HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH") != "1":
        raise ValueError("oracle-screen command env missing HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH=1")
    argv = command.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("oracle-screen command argv must be a list[str]")
    required_args = {
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--oracle-screen-mode",
        "--device",
        "--parent",
        "--parent-sha256",
        "--scratch-root",
        "--curriculum-seed",
        "--support-order-seed",
        "--oracle-screen-max-sampled-candidates",
        "--batch-size",
        "--steps",
        "--max-steps-hard",
        "--max-abs-per-tensor",
        "--emit-progress",
    }
    if not required_args.issubset(set(argv)):
        raise ValueError("oracle-screen command argv missing required probe launch arguments")
    if "--science-arm" in argv:
        raise ValueError("oracle-screen command must route through --oracle-screen-mode, not --science-arm")
    expected_flag_values = (
        ("--oracle-screen-mode", "candidate_set_viability"),
        ("--curriculum-seed", str(STEP6_CURRICULUM_SEED)),
        ("--support-order-seed", str(int(command["support_order_seed"]))),
        ("--oracle-screen-max-sampled-candidates", str(budget)),
        ("--batch-size", str(ORACLE_SCREEN_N20_ROWS)),
        ("--steps", "1"),
        ("--max-steps-hard", "1"),
        ("--max-abs-per-tensor", str(STEP3_BASELINE_MAX_ABS_PER_TENSOR)),
    )
    for flag, expected in expected_flag_values:
        try:
            observed = argv[argv.index(flag) + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"oracle-screen command argv missing {flag} value") from exc
        if observed != expected:
            raise ValueError(
                f"oracle-screen command argv {flag} must be {expected!r}, got {observed!r}"
            )
    try:
        device = argv[argv.index("--device") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("oracle-screen command argv missing --device value") from exc
    if not device.startswith("cuda:"):
        raise ValueError("oracle-screen command argv --device must target CUDA for launch bundle")
    for path_field in ("stdout_path", "stderr_path", "receipt_path", "scratch_root"):
        value = str(command.get(path_field))
        if not value:
            raise ValueError(f"oracle-screen command {path_field} must be non-empty")
        if value.endswith(".pt"):
            raise ValueError(f"oracle-screen command {path_field} cannot target .pt artifacts")
    if command.get("expected_exit_policy") != "exit_0_required_else_stop_no_retry_no_verdict":
        raise ValueError("oracle-screen command expected_exit_policy must fail closed")


def validate_candidate_set_viability_oracle_screen_launch_bundle(packet: Mapping[str, Any]) -> None:
    if packet.get("schema_version") != OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION:
        raise ValueError("unsupported optimizer update-law oracle-screen launch bundle schema")
    if packet.get("diagnostic_class") != DIAGNOSTIC_CLASS_PRE_FULL_STACK:
        raise ValueError("oracle-screen launch bundle must be pre_full_stack_diagnostic")
    _validate_author_only_fields(
        packet,
        expected_packet_kind=ORACLE_SCREEN_LAUNCH_BUNDLE_PACKET_KIND,
        label="author-only oracle-screen launch bundle",
    )
    for field in _READINESS_FORBIDDEN_TRUE_FIELDS:
        _require_false(packet, field)
    if bool(packet.get("carrier_claim")) or bool(packet.get("q_sidecar_carrier_claim")):
        raise ValueError("oracle-screen launch bundle must not make a carrier claim")
    if bool(packet.get("qacc_kernelized")):
        raise ValueError("oracle-screen launch bundle must keep qacc_kernelized=false")
    if not bool(packet.get("optimizer_credit_state_science_dependent")):
        raise ValueError(
            "oracle-screen launch bundle must keep optimizer_credit_state science-dependent"
        )
    if int(packet.get("screen_rows", -1)) != ORACLE_SCREEN_N20_ROWS:
        raise ValueError("oracle-screen launch bundle must pin N=20 rows")
    if int(packet.get("curriculum_seed", -1)) != STEP6_CURRICULUM_SEED:
        raise ValueError("oracle-screen launch bundle curriculum_seed must stay pinned to 17")
    if packet.get("support_order_seeds") != list(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError("oracle-screen launch bundle must pin the contrast support-order seeds")
    if not bool(packet.get("same_candidate_set_required")):
        raise ValueError("oracle-screen launch bundle must require the same candidate set")
    if packet.get("science_contract_commit_sha") != ORACLE_SCREEN_SCIENCE_CONTRACT_COMMIT_SHA:
        raise ValueError("oracle-screen launch bundle must embed the committed afbe598 science contract")
    _reject_raw_arrays(packet)
    _validate_resource_lane(packet.get("resource_lane") or {})
    _validate_phase_budgets(packet.get("phase_budgets") or {})
    _validate_oracle_global_non_persistence(packet)
    _validate_oracle_feasibility_budget(packet.get("oracle_feasibility_budget") or {})
    _validate_oracle_non_persistence(packet.get("oracle_non_persistence_contract") or {})
    _validate_oracle_compact_summary_schema(packet.get("compact_summary_schema") or {})
    _validate_oracle_classifier_contract(packet.get("classifier_contract") or {})
    _validate_oracle_wider_screen_interpretation_contract(
        packet.get("wider_screen_interpretation_contract") or {}
    )
    science_contract = packet.get("science_contract")
    if not isinstance(science_contract, Mapping):
        raise ValueError("oracle-screen launch bundle must embed the science_contract packet")
    validate_candidate_set_viability_oracle_screen_packet(science_contract)
    if str(science_contract.get("parent_path")) != str(packet.get("parent_path")):
        raise ValueError("embedded oracle-screen science contract parent_path must match launch bundle")
    if str(science_contract.get("parent_sha256")) != str(packet.get("parent_sha256")):
        raise ValueError("embedded oracle-screen science contract parent_sha256 must match launch bundle")
    if (
        science_contract.get("wider_screen_interpretation_contract")
        != packet.get("wider_screen_interpretation_contract")
    ):
        raise ValueError(
            "oracle-screen launch bundle wider_screen_interpretation_contract must match science_contract"
        )
    terminal = packet.get("terminal_criteria") or {}
    if terminal.get("branch_classifier") != list(ORACLE_SCREEN_BRANCHES):
        raise ValueError("oracle-screen launch bundle terminal branch classifier drifted")
    if not bool(terminal.get("same_candidate_set_required")):
        raise ValueError("oracle-screen launch bundle terminal criteria must require same candidate set")
    if terminal.get("support_order_seeds") != list(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError("oracle-screen launch bundle terminal criteria support_order_seeds drifted")
    if int(terminal.get("n20_screen_rows", -1)) != ORACLE_SCREEN_N20_ROWS:
        raise ValueError("oracle-screen launch bundle terminal criteria must pin N=20 rows")
    if bool(terminal.get("qacc_kernelized")):
        raise ValueError("oracle-screen launch bundle terminal criteria must keep qacc_kernelized=false")
    if not bool(terminal.get("device_residency_not_hot_loop_residency")):
        raise ValueError("oracle-screen launch bundle must disclaim device residency vs hot-loop residency")
    commands = packet.get("commands")
    if not isinstance(commands, list):
        raise ValueError("oracle-screen launch bundle commands must be a list")
    if len(commands) != len(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError("oracle-screen launch bundle must include exactly one command per contrast seed")
    seen = {int(command.get("support_order_seed", -1)) for command in commands}
    if seen != set(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError("oracle-screen launch bundle command seeds drifted from the contrast contract")
    for command in commands:
        _validate_oracle_screen_command_record(command)
    artifact_policy = packet.get("artifact_policy") or {}
    if not bool(artifact_policy.get("compact_json_ndjson_only")):
        raise ValueError("oracle-screen launch bundle artifact policy must require compact JSON/NDJSON")
    if bool(artifact_policy.get("pt_writes_allowed")):
        raise ValueError("oracle-screen launch bundle artifact policy must reject .pt writes")


def validate_credit_ranking_pivot_measurement_packet(packet: Mapping[str, Any]) -> None:
    if packet.get("schema_version") != OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION:
        raise ValueError("unsupported credit-ranking pivot measurement packet schema")
    if packet.get("diagnostic_class") != DIAGNOSTIC_CLASS_PRE_FULL_STACK:
        raise ValueError("credit-ranking pivot packet must be pre_full_stack_diagnostic")
    _validate_author_only_fields(
        packet,
        expected_packet_kind=CREDIT_RANKING_PIVOT_MEASUREMENT_PACKET_KIND,
        label="author-only credit-ranking pivot measurement packet",
    )
    for field in _READINESS_FORBIDDEN_TRUE_FIELDS:
        _require_false(packet, field)
    if bool(packet.get("carrier_claim")) or bool(packet.get("q_sidecar_carrier_claim")):
        raise ValueError("credit-ranking pivot packet must not make a carrier claim")
    if bool(packet.get("qacc_kernelized")):
        raise ValueError("credit-ranking pivot packet must keep qacc_kernelized=false")
    if not bool(packet.get("optimizer_credit_state_science_dependent")):
        raise ValueError(
            "credit-ranking pivot packet must keep optimizer_credit_state science-dependent"
        )
    if not bool(packet.get("same_candidate_set_required")):
        raise ValueError("credit-ranking pivot packet must require the same candidate set")
    if bool(packet.get("oracle_state_survives_into_learner")):
        raise ValueError("oracle state must not survive into learner fields")
    _reject_raw_arrays(packet)
    _validate_oracle_global_non_persistence(packet)
    _validate_oracle_screen_arms(packet.get("arms") or ())
    _validate_oracle_seed_order_contract(packet.get("seed_order_contract") or {})
    feasibility_budget = packet.get("oracle_feasibility_budget") or {}
    selected_budget = int(feasibility_budget.get("max_sampled_candidates", -1))
    if selected_budget != PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES:
        raise ValueError("credit-ranking pivot packet must pin budget 32")
    _validate_oracle_feasibility_budget(feasibility_budget)
    _validate_oracle_non_persistence(packet.get("oracle_non_persistence_contract") or {})
    _validate_credit_ranking_pivot_compact_summary_schema(
        packet.get("compact_summary_schema") or {}
    )
    _validate_credit_ranking_pivot_measurement_contract(
        packet.get("measurement_contract") or {}
    )
    artifact_policy = packet.get("artifact_policy") or {}
    if not bool(artifact_policy.get("compact_json_ndjson_only")):
        raise ValueError(
            "credit-ranking pivot packet artifact policy must require compact JSON/NDJSON"
        )
    if bool(artifact_policy.get("raw_per_proposal_arrays")):
        raise ValueError("credit-ranking pivot packet must reject raw per-proposal arrays")
    if bool(artifact_policy.get("pt_writes_allowed")):
        raise ValueError("credit-ranking pivot packet artifact policy must reject .pt writes")
    oracle_artifact_path = artifact_policy.get("oracle_artifact_path")
    if oracle_artifact_path is not None and str(oracle_artifact_path).endswith(".pt"):
        raise ValueError("credit-ranking pivot packet artifact path must not target .pt artifacts")


def _validate_credit_ranking_pivot_measurement_command_record(
    command: Mapping[str, Any],
) -> None:
    missing = [field for field in _COMMAND_REQUIRED_FIELDS if field not in command]
    if missing:
        raise ValueError(
            f"credit-ranking pivot command record missing required fields: {missing}"
        )
    if str(command.get("mode")) != "credit_ranking_pivot_measurement_n20":
        raise ValueError(
            "credit-ranking pivot command mode must be credit_ranking_pivot_measurement_n20"
        )
    if str(command.get("phase_role")) != "credit_ranking_pivot_measurement":
        raise ValueError("credit-ranking pivot command phase_role drifted")
    if str(command.get("oracle_screen_mode")) != "credit_ranking_pivot_measurement":
        raise ValueError(
            "credit-ranking pivot command must pin credit_ranking_pivot_measurement mode"
        )
    if not bool(command.get("same_candidate_set_required")):
        raise ValueError("credit-ranking pivot command must require same candidate set")
    if int(command.get("support_order_seed", -1)) not in ORACLE_SCREEN_CONTRAST_SEEDS:
        raise ValueError(
            "credit-ranking pivot command support_order_seed must be one of the contrast seeds"
        )
    if str(command.get("seed_label")) != _support_order_seed_label(
        int(command["support_order_seed"])
    ):
        raise ValueError("credit-ranking pivot command seed_label drifted from support_order_seed")
    if int(command.get("screen_rows", -1)) != ORACLE_SCREEN_N20_ROWS:
        raise ValueError("credit-ranking pivot command screen_rows must be 20")
    if int(command.get("n_rows", -1)) != ORACLE_SCREEN_N20_ROWS:
        raise ValueError("credit-ranking pivot command n_rows must equal screen_rows")
    if int(command.get("steps_requested", -1)) != 1:
        raise ValueError("credit-ranking pivot command steps_requested must be 1")
    if (
        command.get("steps_source")
        != "fixed_single_support_batch_credit_ranking_pivot_measurement"
    ):
        raise ValueError("credit-ranking pivot command steps_source drifted")
    if int(command.get("batch_size", -1)) != ORACLE_SCREEN_N20_ROWS:
        raise ValueError("credit-ranking pivot command batch_size must equal N=20 screen rows")
    if int(command.get("curriculum_seed", -1)) != STEP6_CURRICULUM_SEED:
        raise ValueError("credit-ranking pivot command curriculum_seed must stay pinned to 17")
    budget = int(command.get("max_sampled_candidates", -1))
    if budget != PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES:
        raise ValueError("credit-ranking pivot command max_sampled_candidates must be 32")
    if float(command.get("oracle_max_seconds", -1.0)) != oracle_screen_budget_max_seconds(
        budget
    ):
        raise ValueError("credit-ranking pivot command oracle_max_seconds drifted")
    if int(command.get("max_abs_per_tensor", -1)) != STEP3_BASELINE_MAX_ABS_PER_TENSOR:
        raise ValueError(
            "credit-ranking pivot command max_abs_per_tensor must stay at the baseline cap"
        )
    if float(command.get("fraction_per_tensor", -1.0)) != STEP3_FRACTION_PER_TENSOR:
        raise ValueError("credit-ranking pivot command fraction_per_tensor must stay at 1.0")
    if command.get("global_cap_contract") != "off":
        raise ValueError("credit-ranking pivot command must keep global cap off")
    env = command.get("env")
    if not isinstance(env, Mapping):
        raise ValueError("credit-ranking pivot command env must be a mapping")
    if env.get("HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE") != "1":
        raise ValueError(
            "credit-ranking pivot command env missing HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE=1"
        )
    if env.get("HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH") != "1":
        raise ValueError(
            "credit-ranking pivot command env missing HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH=1"
        )
    argv = command.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("credit-ranking pivot command argv must be a list[str]")
    required_args = {
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--oracle-screen-mode",
        "--device",
        "--parent",
        "--parent-sha256",
        "--scratch-root",
        "--curriculum-seed",
        "--support-order-seed",
        "--oracle-screen-max-sampled-candidates",
        "--batch-size",
        "--steps",
        "--max-steps-hard",
        "--max-abs-per-tensor",
        "--emit-progress",
    }
    if not required_args.issubset(set(argv)):
        raise ValueError(
            "credit-ranking pivot command argv missing required probe launch arguments"
        )
    if "--science-arm" in argv:
        raise ValueError(
            "credit-ranking pivot command must route through --oracle-screen-mode, not --science-arm"
        )
    expected_flag_values = (
        ("--oracle-screen-mode", "credit_ranking_pivot_measurement"),
        ("--curriculum-seed", str(STEP6_CURRICULUM_SEED)),
        ("--support-order-seed", str(int(command["support_order_seed"]))),
        (
            "--oracle-screen-max-sampled-candidates",
            str(PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES),
        ),
        ("--batch-size", str(ORACLE_SCREEN_N20_ROWS)),
        ("--steps", "1"),
        ("--max-steps-hard", "1"),
        ("--max-abs-per-tensor", str(STEP3_BASELINE_MAX_ABS_PER_TENSOR)),
    )
    for flag, expected in expected_flag_values:
        try:
            observed = argv[argv.index(flag) + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"credit-ranking pivot command argv missing {flag} value") from exc
        if observed != expected:
            raise ValueError(
                f"credit-ranking pivot command argv {flag} must be {expected!r}, got {observed!r}"
            )
    try:
        device = argv[argv.index("--device") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("credit-ranking pivot command argv missing --device value") from exc
    if not device.startswith("cuda:"):
        raise ValueError(
            "credit-ranking pivot command argv --device must target CUDA for launch bundle"
        )
    for path_field in ("stdout_path", "stderr_path", "receipt_path", "scratch_root"):
        value = str(command.get(path_field))
        if not value:
            raise ValueError(f"credit-ranking pivot command {path_field} must be non-empty")
        if value.endswith(".pt"):
            raise ValueError(
                f"credit-ranking pivot command {path_field} cannot target .pt artifacts"
            )
    if command.get("expected_exit_policy") != "exit_0_required_else_stop_no_retry_no_verdict":
        raise ValueError("credit-ranking pivot command expected_exit_policy must fail closed")


def validate_credit_ranking_pivot_measurement_launch_bundle(
    packet: Mapping[str, Any],
) -> None:
    if packet.get("schema_version") != OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION:
        raise ValueError(
            "unsupported credit-ranking pivot measurement launch bundle schema"
        )
    if packet.get("diagnostic_class") != DIAGNOSTIC_CLASS_PRE_FULL_STACK:
        raise ValueError(
            "credit-ranking pivot measurement launch bundle must be pre_full_stack_diagnostic"
        )
    _validate_author_only_fields(
        packet,
        expected_packet_kind=CREDIT_RANKING_PIVOT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND,
        label="author-only credit-ranking pivot measurement launch bundle",
    )
    for field in _READINESS_FORBIDDEN_TRUE_FIELDS:
        _require_false(packet, field)
    if bool(packet.get("carrier_claim")) or bool(packet.get("q_sidecar_carrier_claim")):
        raise ValueError(
            "credit-ranking pivot measurement launch bundle must not make a carrier claim"
        )
    if bool(packet.get("qacc_kernelized")):
        raise ValueError(
            "credit-ranking pivot measurement launch bundle must keep qacc_kernelized=false"
        )
    if not bool(packet.get("optimizer_credit_state_science_dependent")):
        raise ValueError(
            "credit-ranking pivot measurement launch bundle must keep optimizer_credit_state science-dependent"
        )
    if int(packet.get("screen_rows", -1)) != ORACLE_SCREEN_N20_ROWS:
        raise ValueError(
            "credit-ranking pivot measurement launch bundle must pin N=20 rows"
        )
    if int(packet.get("curriculum_seed", -1)) != STEP6_CURRICULUM_SEED:
        raise ValueError(
            "credit-ranking pivot measurement launch bundle curriculum_seed must stay pinned to 17"
        )
    if packet.get("support_order_seeds") != list(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError(
            "credit-ranking pivot measurement launch bundle must pin the contrast support-order seeds"
        )
    if not bool(packet.get("same_candidate_set_required")):
        raise ValueError(
            "credit-ranking pivot measurement launch bundle must require the same candidate set"
        )
    _reject_raw_arrays(packet)
    _validate_resource_lane(packet.get("resource_lane") or {})
    _validate_phase_budgets(packet.get("phase_budgets") or {})
    _validate_oracle_global_non_persistence(packet)
    feasibility_budget = packet.get("oracle_feasibility_budget") or {}
    selected_budget = int(feasibility_budget.get("max_sampled_candidates", -1))
    if selected_budget != PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES:
        raise ValueError(
            "credit-ranking pivot measurement launch bundle must pin budget 32"
        )
    _validate_oracle_feasibility_budget(feasibility_budget)
    _validate_oracle_non_persistence(packet.get("oracle_non_persistence_contract") or {})
    _validate_credit_ranking_pivot_compact_summary_schema(
        packet.get("compact_summary_schema") or {}
    )
    _validate_credit_ranking_pivot_measurement_contract(
        packet.get("measurement_contract") or {}
    )
    science_contract = packet.get("science_contract")
    if not isinstance(science_contract, Mapping):
        raise ValueError(
            "credit-ranking pivot measurement launch bundle must embed the science_contract packet"
        )
    validate_credit_ranking_pivot_measurement_packet(science_contract)
    if str(science_contract.get("parent_path")) != str(packet.get("parent_path")):
        raise ValueError(
            "embedded credit-ranking pivot science contract parent_path must match launch bundle"
        )
    if str(science_contract.get("parent_sha256")) != str(packet.get("parent_sha256")):
        raise ValueError(
            "embedded credit-ranking pivot science contract parent_sha256 must match launch bundle"
        )
    if science_contract.get("measurement_contract") != packet.get("measurement_contract"):
        raise ValueError(
            "credit-ranking pivot measurement_contract must match embedded science_contract"
        )
    terminal = packet.get("terminal_criteria") or {}
    if terminal.get("branch_classifier") != list(CREDIT_RANKING_PIVOT_MEASUREMENT_BRANCHES):
        raise ValueError(
            "credit-ranking pivot launch bundle terminal branch classifier drifted"
        )
    if not bool(terminal.get("same_candidate_set_required")):
        raise ValueError(
            "credit-ranking pivot launch bundle terminal criteria must require same candidate set"
        )
    if terminal.get("support_order_seeds") != list(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError(
            "credit-ranking pivot launch bundle terminal criteria support_order_seeds drifted"
        )
    if int(terminal.get("n20_screen_rows", -1)) != ORACLE_SCREEN_N20_ROWS:
        raise ValueError(
            "credit-ranking pivot launch bundle terminal criteria must pin N=20 rows"
        )
    if int(terminal.get("required_max_sampled_candidates", -1)) != (
        PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES
    ):
        raise ValueError(
            "credit-ranking pivot launch bundle terminal criteria must pin budget 32"
        )
    if bool(terminal.get("qacc_kernelized")):
        raise ValueError(
            "credit-ranking pivot launch bundle terminal criteria must keep qacc_kernelized=false"
        )
    if not bool(terminal.get("device_residency_not_hot_loop_residency")):
        raise ValueError(
            "credit-ranking pivot launch bundle must disclaim device residency vs hot-loop residency"
        )
    commands = packet.get("commands")
    if not isinstance(commands, list):
        raise ValueError("credit-ranking pivot launch bundle commands must be a list")
    if len(commands) != len(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError(
            "credit-ranking pivot launch bundle must include exactly one command per contrast seed"
        )
    seen = {int(command.get("support_order_seed", -1)) for command in commands}
    if seen != set(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError(
            "credit-ranking pivot launch bundle command seeds drifted from the contrast seeds contract"
        )
    for command in commands:
        _validate_credit_ranking_pivot_measurement_command_record(command)
    artifact_policy = packet.get("artifact_policy") or {}
    if not bool(artifact_policy.get("compact_json_ndjson_only")):
        raise ValueError(
            "credit-ranking pivot launch bundle artifact policy must require compact JSON/NDJSON"
        )
    if bool(artifact_policy.get("raw_per_proposal_arrays")):
        raise ValueError(
            "credit-ranking pivot launch bundle must reject raw per-proposal arrays"
        )
    if bool(artifact_policy.get("pt_writes_allowed")):
        raise ValueError(
            "credit-ranking pivot launch bundle artifact policy must reject .pt writes"
        )


def validate_activation_credit_measurement_packet(packet: Mapping[str, Any]) -> None:
    if packet.get("schema_version") != OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION:
        raise ValueError("unsupported activation-credit measurement packet schema")
    if packet.get("diagnostic_class") != DIAGNOSTIC_CLASS_PRE_FULL_STACK:
        raise ValueError("activation-credit packet must be pre_full_stack_diagnostic")
    _validate_author_only_fields(
        packet,
        expected_packet_kind=ACTIVATION_CREDIT_MEASUREMENT_PACKET_KIND,
        label="author-only activation-credit measurement packet",
    )
    for field in _READINESS_FORBIDDEN_TRUE_FIELDS:
        _require_false(packet, field)
    if bool(packet.get("carrier_claim")) or bool(packet.get("q_sidecar_carrier_claim")):
        raise ValueError("activation-credit packet must not make a carrier claim")
    if bool(packet.get("qacc_kernelized")):
        raise ValueError("activation-credit packet must keep qacc_kernelized=false")
    if not bool(packet.get("optimizer_credit_state_science_dependent")):
        raise ValueError(
            "activation-credit packet must keep optimizer_credit_state science-dependent"
        )
    if packet.get("eligible_scope") != ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE:
        raise ValueError("activation-credit packet must pin eligible_scope=first-bitlinear")
    if not bool(packet.get("same_candidate_set_required")):
        raise ValueError("activation-credit packet must require the same candidate set")
    if bool(packet.get("oracle_state_survives_into_learner")):
        raise ValueError("oracle state must not survive into learner fields")
    _reject_raw_arrays(packet)
    _validate_oracle_global_non_persistence(packet)
    _validate_oracle_screen_arms(packet.get("arms") or ())
    _validate_oracle_seed_order_contract(packet.get("seed_order_contract") or {})
    feasibility_budget = packet.get("oracle_feasibility_budget") or {}
    selected_budget = int(feasibility_budget.get("max_sampled_candidates", -1))
    if selected_budget != PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES:
        raise ValueError("activation-credit packet must pin budget 32")
    _validate_oracle_feasibility_budget(feasibility_budget)
    _validate_oracle_non_persistence(packet.get("oracle_non_persistence_contract") or {})
    _validate_activation_credit_compact_summary_schema(
        packet.get("compact_summary_schema") or {}
    )
    _validate_activation_credit_measurement_contract(
        packet.get("measurement_contract") or {}
    )
    if not bool(packet.get("scale_smoke_required_before_full_eval")):
        raise ValueError("activation-credit packet must require the de-risk smoke first")
    if packet.get("scale_smoke_launch_bundle_packet_kind") != (
        ACTIVATION_CREDIT_SCALE_SMOKE_LAUNCH_BUNDLE_PACKET_KIND
    ):
        raise ValueError("activation-credit packet smoke launch bundle kind drifted")
    if int(packet.get("fresh_confirmation_seed_required_for_persistent_followup", -1)) != (
        ACTIVATION_CREDIT_FRESH_CONFIRMATION_SEED
    ):
        raise ValueError("activation-credit packet fresh confirmation seed drifted")
    artifact_policy = packet.get("artifact_policy") or {}
    if not bool(artifact_policy.get("compact_json_ndjson_only")):
        raise ValueError(
            "activation-credit packet artifact policy must require compact JSON/NDJSON"
        )
    if bool(artifact_policy.get("raw_per_proposal_arrays")):
        raise ValueError("activation-credit packet must reject raw per-proposal arrays")
    if bool(artifact_policy.get("pt_writes_allowed")):
        raise ValueError("activation-credit packet artifact policy must reject .pt writes")
    oracle_artifact_path = artifact_policy.get("oracle_artifact_path")
    if oracle_artifact_path is not None and str(oracle_artifact_path).endswith(".pt"):
        raise ValueError("activation-credit packet artifact path must not target .pt artifacts")


def _validate_activation_credit_command_record(
    command: Mapping[str, Any],
    *,
    mode: str,
    phase_role: str,
    oracle_screen_mode: str,
    batch_size: int,
    budget: int,
    steps_source: str,
    label: str,
) -> None:
    missing = [field for field in _COMMAND_REQUIRED_FIELDS if field not in command]
    if missing:
        raise ValueError(f"{label} command record missing required fields: {missing}")
    if str(command.get("mode")) != mode:
        raise ValueError(f"{label} command mode drifted")
    if str(command.get("phase_role")) != phase_role:
        raise ValueError(f"{label} command phase_role drifted")
    if str(command.get("oracle_screen_mode")) != oracle_screen_mode:
        raise ValueError(f"{label} command must pin {oracle_screen_mode} mode")
    if not bool(command.get("same_candidate_set_required")):
        raise ValueError(f"{label} command must require same candidate set")
    if int(command.get("support_order_seed", -1)) not in ORACLE_SCREEN_CONTRAST_SEEDS:
        raise ValueError(f"{label} command support_order_seed must be one of the contrast seeds")
    if str(command.get("seed_label")) != _support_order_seed_label(
        int(command["support_order_seed"])
    ):
        raise ValueError(f"{label} command seed_label drifted from support_order_seed")
    if int(command.get("screen_rows", -1)) != batch_size:
        raise ValueError(f"{label} command screen_rows drifted")
    if int(command.get("n_rows", -1)) != batch_size:
        raise ValueError(f"{label} command n_rows must equal screen_rows")
    if int(command.get("steps_requested", -1)) != 1:
        raise ValueError(f"{label} command steps_requested must be 1")
    if command.get("steps_source") != steps_source:
        raise ValueError(f"{label} command steps_source drifted")
    if int(command.get("batch_size", -1)) != batch_size:
        raise ValueError(f"{label} command batch_size drifted")
    if command.get("eligible_scope") != ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE:
        raise ValueError(f"{label} command must pin eligible_scope=first-bitlinear")
    if int(command.get("curriculum_seed", -1)) != STEP6_CURRICULUM_SEED:
        raise ValueError(f"{label} command curriculum_seed must stay pinned to 17")
    if int(command.get("max_sampled_candidates", -1)) != budget:
        raise ValueError(f"{label} command max_sampled_candidates drifted")
    if float(command.get("oracle_max_seconds", -1.0)) != oracle_screen_budget_max_seconds(
        budget
    ):
        raise ValueError(f"{label} command oracle_max_seconds drifted")
    if int(command.get("max_abs_per_tensor", -1)) != STEP3_BASELINE_MAX_ABS_PER_TENSOR:
        raise ValueError(f"{label} command max_abs_per_tensor must stay at the baseline cap")
    if float(command.get("fraction_per_tensor", -1.0)) != STEP3_FRACTION_PER_TENSOR:
        raise ValueError(f"{label} command fraction_per_tensor must stay at 1.0")
    if command.get("global_cap_contract") != "off":
        raise ValueError(f"{label} command must keep global cap off")
    if command.get("capture_device_mode") != ACTIVATION_CREDIT_CAPTURE_DEVICE_MODE:
        raise ValueError(f"{label} command must pin device_resident capture")
    if command.get("grad_proxy_compute_mode") != ACTIVATION_CREDIT_GRAD_PROXY_COMPUTE_MODE:
        raise ValueError(f"{label} command must pin candidate-only gather")
    if not bool(command.get("no_extra_response_label_mask")):
        raise ValueError(f"{label} command must forbid extra response-label masking")
    if not bool(command.get("fp_proxy_transient_eval_only")):
        raise ValueError(f"{label} command must keep fp proxy transient-eval-only")
    env = command.get("env")
    if not isinstance(env, Mapping):
        raise ValueError(f"{label} command env must be a mapping")
    if env.get("HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE") != "1":
        raise ValueError(
            f"{label} command env missing HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE=1"
        )
    if env.get("HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH") != "1":
        raise ValueError(
            f"{label} command env missing HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH=1"
        )
    stdout_path = str(command.get("stdout_path"))
    stderr_path = str(command.get("stderr_path"))
    if env.get(ACTIVATION_CREDIT_STDOUT_PATH_ENV) != stdout_path:
        raise ValueError(f"{label} command env stdout path must match stdout_path")
    if env.get(ACTIVATION_CREDIT_STDERR_PATH_ENV) != stderr_path:
        raise ValueError(f"{label} command env stderr path must match stderr_path")
    argv = command.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError(f"{label} command argv must be a list[str]")
    required_args = {
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--oracle-screen-mode",
        "--device",
        "--parent",
        "--parent-sha256",
        "--scratch-root",
        "--curriculum-seed",
        "--support-order-seed",
        "--oracle-screen-max-sampled-candidates",
        "--batch-size",
        "--eligible-scope",
        "--steps",
        "--max-steps-hard",
        "--max-abs-per-tensor",
        "--emit-progress",
    }
    if not required_args.issubset(set(argv)):
        raise ValueError(f"{label} command argv missing required probe launch arguments")
    if "--science-arm" in argv:
        raise ValueError(
            f"{label} command must route through --oracle-screen-mode, not --science-arm"
        )
    expected_flag_values = (
        ("--oracle-screen-mode", oracle_screen_mode),
        ("--curriculum-seed", str(STEP6_CURRICULUM_SEED)),
        ("--support-order-seed", str(int(command["support_order_seed"]))),
        ("--oracle-screen-max-sampled-candidates", str(budget)),
        ("--batch-size", str(batch_size)),
        ("--eligible-scope", ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE),
        ("--steps", "1"),
        ("--max-steps-hard", "1"),
        ("--max-abs-per-tensor", str(STEP3_BASELINE_MAX_ABS_PER_TENSOR)),
    )
    for flag, expected in expected_flag_values:
        try:
            observed = argv[argv.index(flag) + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{label} command argv missing {flag} value") from exc
        if observed != expected:
            raise ValueError(
                f"{label} command argv {flag} must be {expected!r}, got {observed!r}"
            )
    try:
        device = argv[argv.index("--device") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"{label} command argv missing --device value") from exc
    if not device.startswith("cuda:"):
        raise ValueError(f"{label} command argv --device must target CUDA for launch bundle")
    for path_field in ("stdout_path", "stderr_path", "receipt_path", "scratch_root"):
        value = str(command.get(path_field))
        if not value:
            raise ValueError(f"{label} command {path_field} must be non-empty")
        if value.endswith(".pt"):
            raise ValueError(f"{label} command {path_field} cannot target .pt artifacts")
    if command.get("expected_exit_policy") != "exit_0_required_else_stop_no_retry_no_verdict":
        raise ValueError(f"{label} command expected_exit_policy must fail closed")


def _validate_activation_credit_measurement_command_record(
    command: Mapping[str, Any],
) -> None:
    _validate_activation_credit_command_record(
        command,
        mode="activation_credit_measurement_n20",
        phase_role="activation_credit_measurement",
        oracle_screen_mode="activation_credit_measurement",
        batch_size=ACTIVATION_CREDIT_MEASUREMENT_BATCH_SIZE,
        budget=PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES,
        steps_source="fixed_single_support_batch_activation_credit_measurement",
        label="activation-credit measurement",
    )


def validate_activation_credit_measurement_launch_bundle(
    packet: Mapping[str, Any],
) -> None:
    if packet.get("schema_version") != OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION:
        raise ValueError("unsupported activation-credit measurement launch bundle schema")
    if packet.get("diagnostic_class") != DIAGNOSTIC_CLASS_PRE_FULL_STACK:
        raise ValueError(
            "activation-credit measurement launch bundle must be pre_full_stack_diagnostic"
        )
    _validate_author_only_fields(
        packet,
        expected_packet_kind=ACTIVATION_CREDIT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND,
        label="author-only activation-credit measurement launch bundle",
    )
    for field in _READINESS_FORBIDDEN_TRUE_FIELDS:
        _require_false(packet, field)
    if bool(packet.get("carrier_claim")) or bool(packet.get("q_sidecar_carrier_claim")):
        raise ValueError("activation-credit measurement launch bundle must not make a carrier claim")
    if bool(packet.get("qacc_kernelized")):
        raise ValueError(
            "activation-credit measurement launch bundle must keep qacc_kernelized=false"
        )
    if not bool(packet.get("optimizer_credit_state_science_dependent")):
        raise ValueError(
            "activation-credit measurement launch bundle must keep optimizer_credit_state science-dependent"
        )
    if packet.get("eligible_scope") != ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE:
        raise ValueError(
            "activation-credit measurement launch bundle must pin eligible_scope=first-bitlinear"
        )
    if int(packet.get("screen_rows", -1)) != ACTIVATION_CREDIT_MEASUREMENT_BATCH_SIZE:
        raise ValueError("activation-credit measurement launch bundle must pin N=20 rows")
    if int(packet.get("curriculum_seed", -1)) != STEP6_CURRICULUM_SEED:
        raise ValueError(
            "activation-credit measurement launch bundle curriculum_seed must stay pinned to 17"
        )
    if packet.get("support_order_seeds") != list(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError(
            "activation-credit measurement launch bundle must pin the contrast support-order seeds"
        )
    if not bool(packet.get("same_candidate_set_required")):
        raise ValueError(
            "activation-credit measurement launch bundle must require the same candidate set"
        )
    if not bool(packet.get("scale_smoke_required_before_full_eval")):
        raise ValueError(
            "activation-credit measurement launch bundle must require the de-risk smoke first"
        )
    if packet.get("scale_smoke_launch_bundle_packet_kind") != (
        ACTIVATION_CREDIT_SCALE_SMOKE_LAUNCH_BUNDLE_PACKET_KIND
    ):
        raise ValueError("activation-credit measurement launch bundle smoke packet kind drifted")
    _reject_raw_arrays(packet)
    _validate_resource_lane(packet.get("resource_lane") or {})
    _validate_activation_credit_phase_budgets(packet.get("phase_budgets") or {}, smoke=False)
    _validate_oracle_global_non_persistence(packet)
    feasibility_budget = packet.get("oracle_feasibility_budget") or {}
    selected_budget = int(feasibility_budget.get("max_sampled_candidates", -1))
    if selected_budget != PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES:
        raise ValueError("activation-credit measurement launch bundle must pin budget 32")
    _validate_oracle_feasibility_budget(feasibility_budget)
    _validate_oracle_non_persistence(packet.get("oracle_non_persistence_contract") or {})
    _validate_activation_credit_compact_summary_schema(
        packet.get("compact_summary_schema") or {}
    )
    _validate_activation_credit_measurement_contract(
        packet.get("measurement_contract") or {}
    )
    watcher = packet.get("watcher_audit_bundle") or {}
    progress_filters = set(watcher.get("progress_filters") or ())
    if not {
        "activation_credit_forward_backward",
        "activation_credit_gather",
        "activation_credit_emit",
    }.issubset(progress_filters):
        raise ValueError(
            "activation-credit measurement launch bundle watcher must include activation_credit phase markers"
        )
    science_contract = packet.get("science_contract")
    if not isinstance(science_contract, Mapping):
        raise ValueError(
            "activation-credit measurement launch bundle must embed the science_contract packet"
        )
    validate_activation_credit_measurement_packet(science_contract)
    if str(science_contract.get("parent_path")) != str(packet.get("parent_path")):
        raise ValueError(
            "embedded activation-credit science contract parent_path must match launch bundle"
        )
    if str(science_contract.get("parent_sha256")) != str(packet.get("parent_sha256")):
        raise ValueError(
            "embedded activation-credit science contract parent_sha256 must match launch bundle"
        )
    if science_contract.get("measurement_contract") != packet.get("measurement_contract"):
        raise ValueError(
            "activation-credit measurement_contract must match embedded science_contract"
        )
    terminal = packet.get("terminal_criteria") or {}
    if terminal.get("branch_classifier") != list(ACTIVATION_CREDIT_BRANCHES):
        raise ValueError(
            "activation-credit launch bundle terminal branch classifier drifted"
        )
    for stale_key in ("control_parity_gate", "prior_null_setup_gate", "verdict_rule"):
        if _contains_nested_key(terminal, stale_key):
            raise ValueError(
                f"activation-credit launch bundle terminal criteria must not contain {stale_key}"
            )
    if not bool(terminal.get("same_candidate_set_required")):
        raise ValueError(
            "activation-credit launch bundle terminal criteria must require same candidate set"
        )
    if terminal.get("support_order_seeds") != list(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError(
            "activation-credit launch bundle terminal criteria support_order_seeds drifted"
        )
    if int(terminal.get("n20_screen_rows", -1)) != ACTIVATION_CREDIT_MEASUREMENT_BATCH_SIZE:
        raise ValueError(
            "activation-credit launch bundle terminal criteria must pin N=20 rows"
        )
    if int(terminal.get("required_max_sampled_candidates", -1)) != (
        PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES
    ):
        raise ValueError(
            "activation-credit launch bundle terminal criteria must pin budget 32"
        )
    if terminal.get("required_eligible_scope") != ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE:
        raise ValueError(
            "activation-credit launch bundle terminal criteria eligible_scope drifted"
        )
    if int(
        terminal.get("fresh_confirmation_seed_required_for_persistent_followup", -1)
    ) != ACTIVATION_CREDIT_FRESH_CONFIRMATION_SEED:
        raise ValueError(
            "activation-credit launch bundle terminal fresh confirmation seed drifted"
        )
    if not bool(terminal.get("topology_control_positive_forces_ambiguous")):
        raise ValueError(
            "activation-credit topology control must force ambiguous on positive"
        )
    if bool(terminal.get("qacc_kernelized")):
        raise ValueError(
            "activation-credit launch bundle terminal criteria must keep qacc_kernelized=false"
        )
    if not bool(terminal.get("device_residency_not_hot_loop_residency")):
        raise ValueError(
            "activation-credit launch bundle must disclaim device residency vs hot-loop residency"
        )
    commands = packet.get("commands")
    if not isinstance(commands, list):
        raise ValueError("activation-credit launch bundle commands must be a list")
    if len(commands) != len(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError(
            "activation-credit launch bundle must include exactly one command per contrast seed"
        )
    seen = {int(command.get("support_order_seed", -1)) for command in commands}
    if seen != set(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError(
            "activation-credit launch bundle command seeds drifted from the contrast seeds contract"
        )
    for command in commands:
        _validate_activation_credit_measurement_command_record(command)
    artifact_policy = packet.get("artifact_policy") or {}
    if not bool(artifact_policy.get("compact_json_ndjson_only")):
        raise ValueError(
            "activation-credit launch bundle artifact policy must require compact JSON/NDJSON"
        )
    if bool(artifact_policy.get("raw_per_proposal_arrays")):
        raise ValueError("activation-credit launch bundle must reject raw per-proposal arrays")
    if bool(artifact_policy.get("pt_writes_allowed")):
        raise ValueError("activation-credit launch bundle artifact policy must reject .pt writes")


def _validate_activation_credit_scale_smoke_command_record(
    command: Mapping[str, Any],
) -> None:
    _validate_activation_credit_command_record(
        command,
        mode="activation_credit_scale_smoke_b4_k8",
        phase_role="activation_credit_scale_smoke",
        oracle_screen_mode="activation_credit_scale_smoke",
        batch_size=ACTIVATION_CREDIT_SMOKE_BATCH_SIZE,
        budget=ACTIVATION_CREDIT_SMOKE_MAX_SAMPLED_CANDIDATES,
        steps_source="fixed_single_support_batch_activation_credit_scale_smoke",
        label="activation-credit smoke",
    )


def validate_activation_credit_scale_smoke_launch_bundle(
    packet: Mapping[str, Any],
) -> None:
    if packet.get("schema_version") != OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION:
        raise ValueError("unsupported activation-credit smoke launch bundle schema")
    if packet.get("diagnostic_class") != DIAGNOSTIC_CLASS_PRE_FULL_STACK:
        raise ValueError(
            "activation-credit smoke launch bundle must be pre_full_stack_diagnostic"
        )
    _validate_author_only_fields(
        packet,
        expected_packet_kind=ACTIVATION_CREDIT_SCALE_SMOKE_LAUNCH_BUNDLE_PACKET_KIND,
        label="author-only activation-credit smoke launch bundle",
    )
    for field in _READINESS_FORBIDDEN_TRUE_FIELDS:
        _require_false(packet, field)
    if bool(packet.get("carrier_claim")) or bool(packet.get("q_sidecar_carrier_claim")):
        raise ValueError("activation-credit smoke launch bundle must not make a carrier claim")
    if bool(packet.get("qacc_kernelized")):
        raise ValueError("activation-credit smoke launch bundle must keep qacc_kernelized=false")
    if not bool(packet.get("optimizer_credit_state_science_dependent")):
        raise ValueError(
            "activation-credit smoke launch bundle must keep optimizer_credit_state science-dependent"
        )
    if packet.get("eligible_scope") != ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE:
        raise ValueError(
            "activation-credit smoke launch bundle must pin eligible_scope=first-bitlinear"
        )
    if int(packet.get("screen_rows", -1)) != ACTIVATION_CREDIT_SMOKE_BATCH_SIZE:
        raise ValueError("activation-credit smoke launch bundle must pin batch size 4")
    if int(packet.get("curriculum_seed", -1)) != STEP6_CURRICULUM_SEED:
        raise ValueError(
            "activation-credit smoke launch bundle curriculum_seed must stay pinned to 17"
        )
    if packet.get("support_order_seeds") != list(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError(
            "activation-credit smoke launch bundle must pin the contrast support-order seeds"
        )
    if not bool(packet.get("same_candidate_set_required")):
        raise ValueError(
            "activation-credit smoke launch bundle must require the same candidate set"
        )
    _reject_raw_arrays(packet)
    _validate_resource_lane(packet.get("resource_lane") or {})
    _validate_activation_credit_phase_budgets(packet.get("phase_budgets") or {}, smoke=True)
    _validate_oracle_global_non_persistence(packet)
    feasibility_budget = packet.get("oracle_feasibility_budget") or {}
    selected_budget = int(feasibility_budget.get("max_sampled_candidates", -1))
    if selected_budget != ACTIVATION_CREDIT_SMOKE_MAX_SAMPLED_CANDIDATES:
        raise ValueError("activation-credit smoke launch bundle must pin budget 8")
    _validate_oracle_feasibility_budget(feasibility_budget)
    _validate_oracle_non_persistence(packet.get("oracle_non_persistence_contract") or {})
    _validate_activation_credit_scale_smoke_compact_summary_schema(
        packet.get("compact_summary_schema") or {}
    )
    _validate_activation_credit_scale_smoke_contract(
        packet.get("scale_smoke_contract") or {}
    )
    watcher = packet.get("watcher_audit_bundle") or {}
    progress_filters = set(watcher.get("progress_filters") or ())
    if not {
        "activation_credit_forward_backward",
        "activation_credit_gather",
        "activation_credit_emit",
    }.issubset(progress_filters):
        raise ValueError(
            "activation-credit smoke launch bundle watcher must include activation_credit phase markers"
        )
    terminal = packet.get("terminal_criteria") or {}
    if terminal.get("branch_classifier") is not None:
        raise ValueError(
            "activation-credit smoke launch bundle terminal branch_classifier must stay null"
        )
    for stale_key in ("control_parity_gate", "prior_null_setup_gate", "verdict_rule"):
        if _contains_nested_key(terminal, stale_key):
            raise ValueError(
                f"activation-credit smoke launch bundle terminal criteria must not contain {stale_key}"
            )
    if not bool(terminal.get("same_candidate_set_required")):
        raise ValueError(
            "activation-credit smoke launch bundle terminal criteria must require same candidate set"
        )
    if terminal.get("support_order_seeds") != list(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError(
            "activation-credit smoke launch bundle terminal criteria support_order_seeds drifted"
        )
    if int(terminal.get("required_max_sampled_candidates", -1)) != ACTIVATION_CREDIT_SMOKE_MAX_SAMPLED_CANDIDATES:
        raise ValueError(
            "activation-credit smoke launch bundle terminal criteria must pin budget 8"
        )
    if int(terminal.get("required_batch_size", -1)) != ACTIVATION_CREDIT_SMOKE_BATCH_SIZE:
        raise ValueError(
            "activation-credit smoke launch bundle terminal criteria must pin batch size 4"
        )
    if terminal.get("required_eligible_scope") != ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE:
        raise ValueError(
            "activation-credit smoke launch bundle terminal criteria eligible_scope drifted"
        )
    if not bool(terminal.get("required_grad_proxy_candidate_count_positive")):
        raise ValueError(
            "activation-credit smoke launch bundle terminal criteria must require grad_proxy_candidate_count > 0"
        )
    if not bool(terminal.get("smoke_branch_classification_must_be_null")):
        raise ValueError(
            "activation-credit smoke launch bundle terminal criteria must keep branch_classification=null"
        )
    if not bool(terminal.get("policy_verdict_forbidden")):
        raise ValueError(
            "activation-credit smoke launch bundle terminal criteria must forbid policy verdicts"
        )
    if bool(terminal.get("qacc_kernelized")):
        raise ValueError(
            "activation-credit smoke launch bundle terminal criteria must keep qacc_kernelized=false"
        )
    if not bool(terminal.get("device_residency_not_hot_loop_residency")):
        raise ValueError(
            "activation-credit smoke launch bundle must disclaim device residency vs hot-loop residency"
        )
    _validate_activation_credit_scale_smoke_outcome_contract(
        terminal.get("occupancy_outcome_contract") or {},
        label="activation-credit smoke launch bundle terminal occupancy outcome contract",
    )
    commands = packet.get("commands")
    if not isinstance(commands, list):
        raise ValueError("activation-credit smoke launch bundle commands must be a list")
    if len(commands) != len(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError(
            "activation-credit smoke launch bundle must include exactly one command per contrast seed"
        )
    seen = {int(command.get("support_order_seed", -1)) for command in commands}
    if seen != set(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError(
            "activation-credit smoke launch bundle command seeds drifted from the contrast seeds contract"
        )
    for command in commands:
        _validate_activation_credit_scale_smoke_command_record(command)
    artifact_policy = packet.get("artifact_policy") or {}
    if not bool(artifact_policy.get("compact_json_ndjson_only")):
        raise ValueError(
            "activation-credit smoke launch bundle artifact policy must require compact JSON/NDJSON"
        )
    if bool(artifact_policy.get("raw_per_proposal_arrays")):
        raise ValueError(
            "activation-credit smoke launch bundle must reject raw per-proposal arrays"
        )
    if bool(artifact_policy.get("pt_writes_allowed")):
        raise ValueError(
            "activation-credit smoke launch bundle artifact policy must reject .pt writes"
        )


def validate_within_tie_band_discriminator_packet(packet: Mapping[str, Any]) -> None:
    if packet.get("schema_version") != OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION:
        raise ValueError("unsupported within-tie-band packet schema")
    if packet.get("diagnostic_class") != DIAGNOSTIC_CLASS_PRE_FULL_STACK:
        raise ValueError("within-tie-band packet must be pre_full_stack_diagnostic")
    _validate_author_only_fields(
        packet,
        expected_packet_kind=WITHIN_TIE_BAND_DISCRIMINATOR_PACKET_KIND,
        label="author-only within-tie-band discriminator packet",
    )
    for field in _READINESS_FORBIDDEN_TRUE_FIELDS:
        _require_false(packet, field)
    if bool(packet.get("carrier_claim")) or bool(packet.get("q_sidecar_carrier_claim")):
        raise ValueError("within-tie-band packet must not make a carrier claim")
    if bool(packet.get("qacc_kernelized")):
        raise ValueError("within-tie-band packet must keep qacc_kernelized=false")
    if not bool(packet.get("optimizer_credit_state_science_dependent")):
        raise ValueError(
            "within-tie-band packet must keep optimizer_credit_state science-dependent"
        )
    if not bool(packet.get("same_candidate_set_required")):
        raise ValueError("within-tie-band packet must require the same candidate set")
    if bool(packet.get("oracle_state_survives_into_learner")):
        raise ValueError("oracle state must not survive into learner fields")
    _reject_raw_arrays(packet)
    _validate_oracle_global_non_persistence(packet)
    _validate_oracle_screen_arms(packet.get("arms") or ())
    _validate_oracle_seed_order_contract(packet.get("seed_order_contract") or {})
    feasibility_budget = packet.get("oracle_feasibility_budget") or {}
    selected_budget = int(feasibility_budget.get("max_sampled_candidates", -1))
    if selected_budget != PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES:
        raise ValueError("within-tie-band packet must pin budget 32")
    _validate_oracle_feasibility_budget(feasibility_budget)
    _validate_oracle_non_persistence(packet.get("oracle_non_persistence_contract") or {})
    _validate_within_tie_band_discriminator_compact_summary_schema(
        packet.get("compact_summary_schema") or {}
    )
    _validate_within_tie_band_discriminator_measurement_contract(
        packet.get("measurement_contract") or {}
    )
    artifact_policy = packet.get("artifact_policy") or {}
    if not bool(artifact_policy.get("compact_json_ndjson_only")):
        raise ValueError(
            "within-tie-band packet artifact policy must require compact JSON/NDJSON"
        )
    if bool(artifact_policy.get("raw_per_proposal_arrays")):
        raise ValueError("within-tie-band packet must reject raw per-proposal arrays")
    if bool(artifact_policy.get("pt_writes_allowed")):
        raise ValueError("within-tie-band packet artifact policy must reject .pt writes")
    oracle_artifact_path = artifact_policy.get("oracle_artifact_path")
    if oracle_artifact_path is not None and str(oracle_artifact_path).endswith(".pt"):
        raise ValueError("within-tie-band packet artifact path must not target .pt artifacts")


def _validate_within_tie_band_discriminator_command_record(
    command: Mapping[str, Any],
) -> None:
    missing = [field for field in _COMMAND_REQUIRED_FIELDS if field not in command]
    if missing:
        raise ValueError(
            f"within-tie-band command record missing required fields: {missing}"
        )
    if str(command.get("mode")) != "within_tie_band_discriminator_n20":
        raise ValueError(
            "within-tie-band command mode must be within_tie_band_discriminator_n20"
        )
    if str(command.get("phase_role")) != "within_tie_band_discriminator":
        raise ValueError("within-tie-band command phase_role drifted")
    if str(command.get("oracle_screen_mode")) != "within_tie_band_discriminator":
        raise ValueError(
            "within-tie-band command must pin within_tie_band_discriminator mode"
        )
    if not bool(command.get("same_candidate_set_required")):
        raise ValueError("within-tie-band command must require same candidate set")
    if int(command.get("support_order_seed", -1)) not in ORACLE_SCREEN_CONTRAST_SEEDS:
        raise ValueError(
            "within-tie-band command support_order_seed must be one of the contrast seeds"
        )
    if str(command.get("seed_label")) != _support_order_seed_label(
        int(command["support_order_seed"])
    ):
        raise ValueError("within-tie-band command seed_label drifted from support_order_seed")
    if int(command.get("screen_rows", -1)) != ORACLE_SCREEN_N20_ROWS:
        raise ValueError("within-tie-band command screen_rows must be 20")
    if int(command.get("n_rows", -1)) != ORACLE_SCREEN_N20_ROWS:
        raise ValueError("within-tie-band command n_rows must equal screen_rows")
    if int(command.get("steps_requested", -1)) != 1:
        raise ValueError("within-tie-band command steps_requested must be 1")
    if (
        command.get("steps_source")
        != "fixed_single_support_batch_within_tie_band_discriminator"
    ):
        raise ValueError("within-tie-band command steps_source drifted")
    if int(command.get("batch_size", -1)) != ORACLE_SCREEN_N20_ROWS:
        raise ValueError("within-tie-band command batch_size must equal N=20 screen rows")
    if int(command.get("curriculum_seed", -1)) != STEP6_CURRICULUM_SEED:
        raise ValueError("within-tie-band command curriculum_seed must stay pinned to 17")
    budget = int(command.get("max_sampled_candidates", -1))
    if budget != PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES:
        raise ValueError("within-tie-band command max_sampled_candidates must be 32")
    if float(command.get("oracle_max_seconds", -1.0)) != oracle_screen_budget_max_seconds(
        budget
    ):
        raise ValueError("within-tie-band command oracle_max_seconds drifted")
    if int(command.get("max_abs_per_tensor", -1)) != STEP3_BASELINE_MAX_ABS_PER_TENSOR:
        raise ValueError(
            "within-tie-band command max_abs_per_tensor must stay at the baseline cap"
        )
    if float(command.get("fraction_per_tensor", -1.0)) != STEP3_FRACTION_PER_TENSOR:
        raise ValueError("within-tie-band command fraction_per_tensor must stay at 1.0")
    if command.get("global_cap_contract") != "off":
        raise ValueError("within-tie-band command must keep global cap off")
    env = command.get("env")
    if not isinstance(env, Mapping):
        raise ValueError("within-tie-band command env must be a mapping")
    if env.get("HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE") != "1":
        raise ValueError(
            "within-tie-band command env missing HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE=1"
        )
    if env.get("HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH") != "1":
        raise ValueError(
            "within-tie-band command env missing HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH=1"
        )
    argv = command.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("within-tie-band command argv must be a list[str]")
    required_args = {
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--oracle-screen-mode",
        "--device",
        "--parent",
        "--parent-sha256",
        "--scratch-root",
        "--curriculum-seed",
        "--support-order-seed",
        "--oracle-screen-max-sampled-candidates",
        "--batch-size",
        "--steps",
        "--max-steps-hard",
        "--max-abs-per-tensor",
        "--emit-progress",
    }
    if not required_args.issubset(set(argv)):
        raise ValueError(
            "within-tie-band command argv missing required probe launch arguments"
        )
    if "--science-arm" in argv:
        raise ValueError(
            "within-tie-band command must route through --oracle-screen-mode, not --science-arm"
        )
    expected_flag_values = (
        ("--oracle-screen-mode", "within_tie_band_discriminator"),
        ("--curriculum-seed", str(STEP6_CURRICULUM_SEED)),
        ("--support-order-seed", str(int(command["support_order_seed"]))),
        (
            "--oracle-screen-max-sampled-candidates",
            str(PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES),
        ),
        ("--batch-size", str(ORACLE_SCREEN_N20_ROWS)),
        ("--steps", "1"),
        ("--max-steps-hard", "1"),
        ("--max-abs-per-tensor", str(STEP3_BASELINE_MAX_ABS_PER_TENSOR)),
    )
    for flag, expected in expected_flag_values:
        try:
            observed = argv[argv.index(flag) + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"within-tie-band command argv missing {flag} value") from exc
        if observed != expected:
            raise ValueError(
                f"within-tie-band command argv {flag} must be {expected!r}, got {observed!r}"
            )
    try:
        device = argv[argv.index("--device") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("within-tie-band command argv missing --device value") from exc
    if not device.startswith("cuda:"):
        raise ValueError(
            "within-tie-band command argv --device must target CUDA for launch bundle"
        )
    for path_field in ("stdout_path", "stderr_path", "receipt_path", "scratch_root"):
        value = str(command.get(path_field))
        if not value:
            raise ValueError(f"within-tie-band command {path_field} must be non-empty")
        if value.endswith(".pt"):
            raise ValueError(
                f"within-tie-band command {path_field} cannot target .pt artifacts"
            )
    if command.get("expected_exit_policy") != "exit_0_required_else_stop_no_retry_no_verdict":
        raise ValueError("within-tie-band command expected_exit_policy must fail closed")


def validate_within_tie_band_discriminator_launch_bundle(
    packet: Mapping[str, Any],
) -> None:
    if packet.get("schema_version") != OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION:
        raise ValueError("unsupported within-tie-band launch bundle schema")
    if packet.get("diagnostic_class") != DIAGNOSTIC_CLASS_PRE_FULL_STACK:
        raise ValueError(
            "within-tie-band launch bundle must be pre_full_stack_diagnostic"
        )
    _validate_author_only_fields(
        packet,
        expected_packet_kind=WITHIN_TIE_BAND_DISCRIMINATOR_LAUNCH_BUNDLE_PACKET_KIND,
        label="author-only within-tie-band discriminator launch bundle",
    )
    for field in _READINESS_FORBIDDEN_TRUE_FIELDS:
        _require_false(packet, field)
    if bool(packet.get("carrier_claim")) or bool(packet.get("q_sidecar_carrier_claim")):
        raise ValueError(
            "within-tie-band launch bundle must not make a carrier claim"
        )
    if bool(packet.get("qacc_kernelized")):
        raise ValueError("within-tie-band launch bundle must keep qacc_kernelized=false")
    if not bool(packet.get("optimizer_credit_state_science_dependent")):
        raise ValueError(
            "within-tie-band launch bundle must keep optimizer_credit_state science-dependent"
        )
    if int(packet.get("screen_rows", -1)) != ORACLE_SCREEN_N20_ROWS:
        raise ValueError("within-tie-band launch bundle must pin N=20 rows")
    if int(packet.get("curriculum_seed", -1)) != STEP6_CURRICULUM_SEED:
        raise ValueError(
            "within-tie-band launch bundle curriculum_seed must stay pinned to 17"
        )
    if packet.get("support_order_seeds") != list(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError(
            "within-tie-band launch bundle must pin the contrast support-order seeds"
        )
    if not bool(packet.get("same_candidate_set_required")):
        raise ValueError(
            "within-tie-band launch bundle must require the same candidate set"
        )
    _reject_raw_arrays(packet)
    _validate_resource_lane(packet.get("resource_lane") or {})
    _validate_phase_budgets(packet.get("phase_budgets") or {})
    _validate_oracle_global_non_persistence(packet)
    feasibility_budget = packet.get("oracle_feasibility_budget") or {}
    selected_budget = int(feasibility_budget.get("max_sampled_candidates", -1))
    if selected_budget != PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES:
        raise ValueError("within-tie-band launch bundle must pin budget 32")
    _validate_oracle_feasibility_budget(feasibility_budget)
    _validate_oracle_non_persistence(packet.get("oracle_non_persistence_contract") or {})
    _validate_within_tie_band_discriminator_compact_summary_schema(
        packet.get("compact_summary_schema") or {}
    )
    _validate_within_tie_band_discriminator_measurement_contract(
        packet.get("measurement_contract") or {}
    )
    science_contract = packet.get("science_contract")
    if not isinstance(science_contract, Mapping):
        raise ValueError(
            "within-tie-band launch bundle must embed the science_contract packet"
        )
    validate_within_tie_band_discriminator_packet(science_contract)
    if str(science_contract.get("parent_path")) != str(packet.get("parent_path")):
        raise ValueError(
            "embedded within-tie-band science contract parent_path must match launch bundle"
        )
    if str(science_contract.get("parent_sha256")) != str(packet.get("parent_sha256")):
        raise ValueError(
            "embedded within-tie-band science contract parent_sha256 must match launch bundle"
        )
    if science_contract.get("measurement_contract") != packet.get("measurement_contract"):
        raise ValueError(
            "within-tie-band measurement_contract must match embedded science_contract"
        )
    terminal = packet.get("terminal_criteria") or {}
    if terminal.get("branch_classifier") != list(WITHIN_TIE_BAND_DISCRIMINATOR_BRANCHES):
        raise ValueError(
            "within-tie-band launch bundle terminal branch classifier drifted"
        )
    if not bool(terminal.get("same_candidate_set_required")):
        raise ValueError(
            "within-tie-band launch bundle terminal criteria must require same candidate set"
        )
    if terminal.get("support_order_seeds") != list(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError(
            "within-tie-band launch bundle terminal criteria support_order_seeds drifted"
        )
    if int(terminal.get("n20_screen_rows", -1)) != ORACLE_SCREEN_N20_ROWS:
        raise ValueError(
            "within-tie-band launch bundle terminal criteria must pin N=20 rows"
        )
    if int(terminal.get("required_max_sampled_candidates", -1)) != PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES:
        raise ValueError(
            "within-tie-band launch bundle terminal criteria must pin budget 32"
        )
    if bool(terminal.get("qacc_kernelized")):
        raise ValueError(
            "within-tie-band launch bundle terminal criteria must keep qacc_kernelized=false"
        )
    if not bool(terminal.get("device_residency_not_hot_loop_residency")):
        raise ValueError(
            "within-tie-band launch bundle terminal criteria must preserve device-vs-hot-loop wording"
        )
    if not bool(packet.get("compact_instrumentation_only")):
        raise ValueError(
            "within-tie-band launch bundle must stay compact-instrumentation-only"
        )
    if bool(packet.get("raw_per_proposal_arrays_included")):
        raise ValueError(
            "within-tie-band launch bundle must reject raw per-proposal arrays"
        )
    commands = packet.get("commands")
    if not isinstance(commands, list):
        raise ValueError("within-tie-band launch bundle commands must be a list")
    if len(commands) != len(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError(
            "within-tie-band launch bundle must include exactly one command per contrast seed"
        )
    seen = {int(command.get("support_order_seed", -1)) for command in commands}
    if seen != set(ORACLE_SCREEN_CONTRAST_SEEDS):
        raise ValueError(
            "within-tie-band launch bundle command seeds drifted from the contrast seeds contract"
        )
    for command in commands:
        _validate_within_tie_band_discriminator_command_record(command)
    artifact_policy = packet.get("artifact_policy") or {}
    if not bool(artifact_policy.get("compact_json_ndjson_only")):
        raise ValueError(
            "within-tie-band launch bundle artifact policy must require compact JSON/NDJSON"
        )
    if bool(artifact_policy.get("raw_per_proposal_arrays")):
        raise ValueError(
            "within-tie-band launch bundle must reject raw per-proposal arrays"
        )
    if bool(artifact_policy.get("pt_writes_allowed")):
        raise ValueError(
            "within-tie-band launch bundle artifact policy must reject .pt writes"
        )


def packet_without_runtime_results(packet: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(dict(packet))
    out.pop("runtime_results", None)
    out.pop("arm_metrics", None)
    return out


__all__ = [
    "ARM_A0_RANK_BUCKET_CURRENT",
    "ARM_A1_RANK_BUCKET_ORDER_MATCHED",
    "ARM_B_CAP_MAX_ABS_1024",
    "ARM_B_RANK_FREE_SIGN_PRESSURE",
    "ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER",
    "ARM_INVERTED_SIGN_PRESSURE",
    "BRANCH_CANDIDATE_GENERATION_BAD_OR_NO_LOCAL_SIGNAL",
    "BRANCH_CANDIDATE_SET_VIABLE_CREDIT_RANKING_BAD",
    "BRANCH_CREDIT_MAGNITUDE_BAD_SIGN_USABLE",
    "BRANCH_CREDIT_SOURCE_NOT_SUFFICIENT",
    "BRANCH_DIRECTION_PROJECTION_WRONG",
    "BRANCH_INSUFFICIENT_SEPARATION",
    "BRANCH_MEASUREMENT_LOSS_POWERED",
    "BRANCH_MEASUREMENT_AMBIGUOUS_NO_BRANCH",
    "BRANCH_MEASUREMENT_AMBIGUOUS_TIE_BAND_ALIASING",
    "BRANCH_MEASUREMENT_ORDER_SENSITIVE",
    "BRANCH_MEASUREMENT_POWERED",
    "BRANCH_MEASUREMENT_UNDERPOWERED",
    "BRANCH_ORACLE_INFEASIBLE_OR_TOO_EXPENSIVE",
    "BRANCH_POWERED_NEGATIVE_OR_LOSS_ONLY",
    "BRANCH_PREREGISTERED_CHEAP_LEARNER_FEATURE_FAMILY_CANNOT_PREDICT_REGRET",
    "BRANCH_ACTIVATION_CREDIT_AMBIGUOUS_NO_BRANCH",
    "BRANCH_ACTIVATION_CREDIT_CANDIDATE_SIGNAL",
    "BRANCH_ACTIVATION_CREDIT_MISSING_SIGNAL_DEEPER_THAN_FIRST_ORDER_CREDIT_STORAGE",
    "BRANCH_WITHIN_TIE_BAND_AMBIGUOUS_NO_BRANCH",
    "BRANCH_WITHIN_TIE_BAND_LEARNER_FEATURES_SEPARATE_REGRET",
    "BRANCH_WITHIN_TIE_BAND_NEEDS_NEW_LEARNER_STATE",
    "ACTIVATION_CREDIT_BRANCHES",
    "ACTIVATION_CREDIT_CAPTURE_DEVICE_MODE",
    "ACTIVATION_CREDIT_FRESH_CONFIRMATION_SEED",
    "ACTIVATION_CREDIT_GRAD_PROXY_COMPUTE_MODE",
    "ACTIVATION_CREDIT_MAGNITUDE_BIN_COUNT",
    "ACTIVATION_CREDIT_MEASUREMENT_BATCH_SIZE",
    "ACTIVATION_CREDIT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND",
    "ACTIVATION_CREDIT_MEASUREMENT_PACKET_KIND",
    "ACTIVATION_CREDIT_PRIMARY_FAMILY_ID",
    "ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE",
    "ACTIVATION_CREDIT_SCALE_SMOKE_LAUNCH_BUNDLE_PACKET_KIND",
    "ACTIVATION_CREDIT_SMOKE_BATCH_SIZE",
    "ACTIVATION_CREDIT_SMOKE_MAX_SAMPLED_CANDIDATES",
    "ACTIVATION_CREDIT_TOPOLOGY_CONTROL_FAMILY_ID",
    "CREDIT_RANKING_PIVOT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND",
    "CREDIT_RANKING_PIVOT_MEASUREMENT_PACKET_KIND",
    "BRANCH_CURRENT_ORDER_NOT_NECESSARY",
    "BRANCH_CURRENT_QACC_MARGIN_ORDER_BUNDLE_CARRIER",
    "BRANCH_MASS_CONFOUNDED_CURRENT_ORDER_SIGNAL",
    "BRANCH_NO_MATCH_DIFFERENT_CREDIT_SOURCE",
    "BRANCH_PARTIAL_LOCAL_SIGNAL",
    "BRANCH_PRIOR_NULL_SETUP_UNVERIFIED",
    "BRANCH_A0_COMPONENT_ORDER_ROBUST",
    "BRANCH_RANK_FREE_POSITIVE",
    "BRANCH_RANK_MAGNITUDE_CONDITIONED_ON_CURRENT_ORDER",
    "BRANCH_RANKING_STILL_REQUIRED",
    "BRANCH_SCHEDULER_ONLY_ORDER_SENSITIVE",
    "BRANCH_TIE_POLICY_OR_OVERUPDATE",
    "CONTROL_PARITY_FRACTION_MAX",
    "CONTROL_PARITY_FRACTION_MIN",
    "DIAGNOSTIC_CLASS_PRE_FULL_STACK",
    "FIXED_RANK_BUCKET_NON_TARGET_AUX",
    "ORACLE_ARM_CURRENT_CREDIT_RANK_BUCKET_CURRENT_ORDER",
    "ORACLE_ARM_DETERMINISTIC_HASH_SAME_VOTES",
    "ORACLE_ARM_DIAGNOSTIC_LOCAL_LOSS_DELTA",
    "ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES",
    "ORACLE_SCREEN_ARM_IDS",
    "ORACLE_SCREEN_BRANCHES",
    "ORACLE_SCREEN_CONTRAST_SEEDS",
    "ORACLE_SCREEN_LAUNCH_BUNDLE_PACKET_KIND",
    "ORACLE_SCREEN_MAX_SECONDS_BY_BUDGET",
    "ORACLE_SCREEN_PACKET_KIND",
    "ORACLE_SCREEN_PROMOTION_ORDER_SEEDS",
    "ORACLE_SCREEN_SCIENCE_CONTRACT_COMMIT_SHA",
    "ORACLE_WIDER_SCREEN_INTERPRETATION_VERDICTS",
    "ORACLE_WIDER_SCREEN_VERDICT_CREDIT_RANKING_BAD",
    "ORACLE_WIDER_SCREEN_VERDICT_RANKING_EFFECTIVELY_OK",
    "ORACLE_WIDER_SCREEN_VERDICT_RANKING_SUBOPTIMAL",
    "OPTIMIZER_UPDATE_LAW_BRANCHES",
    "OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION",
    "SCIENCE_MODE_BRANCH_VERDICT",
    "SCIENCE_MODE_PRETERMINAL_SCREEN",
    "STEP1_DRY_RUN_PACKET_KIND",
    "STEP2_LAUNCH_BUNDLE_PACKET_KIND",
    "STEP3_CAP_MAX_ABS_PER_TENSOR",
    "STEP3_BASELINE_MAX_ABS_PER_TENSOR",
    "STEP3_MEASUREMENT_POWER_TRUST_REGION_PACKET_KIND",
    "STEP4_MATCH_STRICT_GAP_MAX",
    "STEP4_POWERED_RANK_SIGNAL_DECOMPOSITION_PACKET_KIND",
    "STEP5_ARM_IDS",
    "STEP5_CURRICULUM_SEED",
    "STEP5_STRICT_FLOOR_COUNT",
    "STEP5_STRICT_MARGIN_COUNT",
    "STEP5_STRICT_TOTAL",
    "STEP5_SUPPORT_ORDER_SEED",
    "STEP5_SUPPORT_ORDER_TRAJECTORY_ROBUSTNESS_PACKET_KIND",
    "STEP6_ARM_IDS",
    "STEP6_CURRICULUM_SEED",
    "STEP6_FIXED_PREREG_NEW_SEED",
    "STEP6_MAX_ARM_RUNS",
    "STEP6_ORDER_AVERAGED_A0_COMPONENT_DECOMPOSITION_PACKET_KIND",
    "STEP6_SUPPORT_ORDER_SEEDS",
    "TIE_POLICY_CURRENT_MARGIN_INDEX",
    "TIE_POLICY_DETERMINISTIC_HASH_MATCHED",
    "WITHIN_TIE_BAND_ABLATION_FAMILY_IDS",
    "WITHIN_TIE_BAND_DISCRIMINATOR_BRANCHES",
    "WITHIN_TIE_BAND_DISCRIMINATOR_LAUNCH_BUNDLE_PACKET_KIND",
    "WITHIN_TIE_BAND_DISCRIMINATOR_PACKET_KIND",
    "WITHIN_TIE_BAND_PRIMARY_FAMILY_ID",
    "WITHIN_TIE_BAND_TARGET_TIE_BAND_ID",
    "build_measurement_power_then_trust_region_packet",
    "build_activation_credit_measurement_launch_bundle",
    "build_activation_credit_measurement_packet",
    "build_activation_credit_scale_smoke_launch_bundle",
    "build_candidate_set_viability_oracle_screen_launch_bundle",
    "build_candidate_set_viability_oracle_screen_packet",
    "build_credit_ranking_pivot_measurement_launch_bundle",
    "build_credit_ranking_pivot_measurement_packet",
    "build_optimizer_update_law_launch_bundle",
    "build_optimizer_update_law_science_packet",
    "build_order_averaged_a0_component_decomposition_packet",
    "build_powered_rank_signal_decomposition_packet",
    "build_support_order_trajectory_robustness_packet",
    "build_within_tie_band_discriminator_launch_bundle",
    "build_within_tie_band_discriminator_packet",
    "classify_candidate_set_viability_oracle_screen",
    "classify_optimizer_update_law_branch",
    "classify_step4_rank_signal_decomposition",
    "classify_step3_power_floor",
    "default_control_parity_gate",
    "default_hash_gate_policy",
    "default_prior_verdict_parent_ref",
    "default_resource_lane_contract",
    "default_science_arms",
    "default_step3_effective_cap_audit",
    "default_step3_power_floor",
    "default_step4_mass_confound_rule",
    "default_step4_match_to_a0_rule",
    "default_step4_science_arms",
    "default_step5_pass_rule",
    "default_step5_science_arms",
    "default_step5_support_order_proof_contract",
    "default_step6_mass_confound_rule",
    "default_step6_science_arms",
    "default_step6_stability_rule",
    "default_step6_support_order_proof_contract",
    "default_screen_before_verdict_dependency",
    "default_terminal_criteria",
    "default_verdict_rule",
    "default_watcher_bundle",
    "oracle_screen_budget_max_seconds",
    "oracle_screen_effectively_ok_rank_position_exclusive_bound",
    "packet_without_runtime_results",
    "step4_arm_matches_a0",
    "step4_mass_confound_detected",
    "validate_measurement_power_then_trust_region_packet",
    "validate_activation_credit_measurement_launch_bundle",
    "validate_activation_credit_measurement_packet",
    "validate_activation_credit_scale_smoke_launch_bundle",
    "validate_candidate_set_viability_oracle_screen_launch_bundle",
    "validate_candidate_set_viability_oracle_screen_packet",
    "validate_credit_ranking_pivot_measurement_launch_bundle",
    "validate_credit_ranking_pivot_measurement_packet",
    "validate_optimizer_update_law_launch_bundle",
    "validate_optimizer_update_law_science_packet",
    "validate_order_averaged_a0_component_decomposition_packet",
    "validate_powered_rank_signal_decomposition_packet",
    "validate_support_order_trajectory_robustness_packet",
    "validate_within_tie_band_discriminator_launch_bundle",
    "validate_within_tie_band_discriminator_packet",
]
