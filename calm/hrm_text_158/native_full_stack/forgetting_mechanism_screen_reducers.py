"""Compat re-export facade for forgetting-mechanism screen reducers.

Behavior-preserving extraction (r6): public names unchanged. Prefer importing
from the named seams directly for new code; this module keeps caller churn at
zero for tests/script during the refactor.

Seams (flat under native_full_stack/):
  fixed_qscale_credit.py
  forgetting_laws.py
  family_classifier.py
  phase_probe_sets.py
  phase_receipt_contracts.py
  screen_model_runtime.py
  screen_execution_loop.py
  screen_receipt_output.py
  screen_run_loop.py  (thin re-export / glue shim)

Bound by PLAN_v9 sha 07a02aff… (authority dispatch 1784812148229).
Do NOT mutate vote_lifetime_screen_reducers.py (frozen-q zero-drain contract).
"""
from __future__ import annotations

from calm.hrm_text_158.native_full_stack.family_classifier import (  # noqa: F401
    ARM0,
    ARM1,
    ARM2,
    ARM3,
    CENSOR_CLEAR_MAX,
    EPS,
    FAMILY_F1,
    FAMILY_F2,
    FAMILY_F3,
    FAMILY_F4,
    H_PROGRESS_BAR_FRAC,
    N_FLIPS_VACUOUS,
    Q_MOTION_MIN_FRAC,
    RETENTION_SLOP_COUNTS,
    SUB2_ACC_BUDGET_BPW,
    TIE_TOLERANCE_BPW,
    classify_forgetting_family_screen,
    g0_valid,
    g0b_q_motion_ok,
    g1_survival_ok,
    retention_ok,
)
from calm.hrm_text_158.native_full_stack.fixed_qscale_credit import (  # noqa: F401
    CreditGradStore,
    FixedQScaleLinearWithCredit,
    begin_credit_step,
    bitlinear_absmean_quantize,
    cumulative_q_transitions,
    fixed_qscale_linear_with_credit,
    flattened_nd_dW,
    get_credit_store,
    mechanical_dynamic_scale_diverges,
    qscale_reference_weight,
    snapshot_route_counters,
)
from calm.hrm_text_158.native_full_stack.forgetting_laws import (  # noqa: F401
    H_TRAJECTORY_EVERY,
    apply_decay_leak,
    apply_live_flip_writeback,
    apply_sparse_hot,
    apply_ttl_age_drain,
    entropy_bits,
    should_record_h_trajectory,
    threshold_residual_writeback,
)
from calm.hrm_text_158.native_full_stack.phase_probe_sets import (  # noqa: F401
    ACQ_N,
    ACQ_SHUFFLE_SEED,
    ACQUISITION_SELECTION_SHA256,
    IDENTITY_PARENT_SUPPORT_HASH,
    IDENTITY_SELECTION_SHA256,
    MATH_A0_PARENT_SUPPORT_HASH,
    RET_ID_N,
    RET_MATH_N,
    RET_SHUFFLE_SEED,
    Row3,
    build_phase1_probe_sets,
    compact_rows_sha256,
    load_identity_full_rows,
    load_math_a0_rows,
    parent_support_hash16,
    sample_batch_excluding_acquisition,
    select_acquisition_rows,
    select_retention_identity_rows,
    select_retention_math_rows,
)
from calm.hrm_text_158.native_full_stack.phase_receipt_contracts import (  # noqa: F401
    DEFAULT_AUTHORITY_DISPATCH,
    DEFAULT_PARENT_SHA256,
    DEFAULT_PLAN_SHA256,
    PHASE0_ALLOWED_STEPS,
    PHASE0_ARM_ID,
    PHASE0_SCREEN_ID,
    PHASE0_STEPS,
    PHASE0B_STEPS_FALLBACK_ONCE,
    PHASE_BATCH,
    PHASE_TOPK,
    ArmReceiptContractError,
    arm_metrics_for_classifier,
    build_phase1_terminal_receipt,
    decide_phase0_aggregate_transition,
    validate_phase0_receipt_for_aggregate,
    validate_shared_held_fixed_arm_receipts,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (  # noqa: F401
    CROSSING_THRESHOLD_ABS,
)
from calm.hrm_text_158.native_full_stack.vote_lifetime_screen_reducers import (  # noqa: F401
    apply_drain_resets,
    lifetime_censored_frac,
    update_episode_starts,
)

__all__ = [
    "ACQ_N",
    "ACQ_SHUFFLE_SEED",
    "ACQUISITION_SELECTION_SHA256",
    "ARM0",
    "ARM1",
    "ARM2",
    "ARM3",
    "ArmReceiptContractError",
    "CENSOR_CLEAR_MAX",
    "CROSSING_THRESHOLD_ABS",
    "CreditGradStore",
    "DEFAULT_AUTHORITY_DISPATCH",
    "DEFAULT_PARENT_SHA256",
    "DEFAULT_PLAN_SHA256",
    "EPS",
    "FAMILY_F1",
    "FAMILY_F2",
    "FAMILY_F3",
    "FAMILY_F4",
    "FixedQScaleLinearWithCredit",
    "H_PROGRESS_BAR_FRAC",
    "H_TRAJECTORY_EVERY",
    "IDENTITY_PARENT_SUPPORT_HASH",
    "IDENTITY_SELECTION_SHA256",
    "MATH_A0_PARENT_SUPPORT_HASH",
    "N_FLIPS_VACUOUS",
    "PHASE0_ALLOWED_STEPS",
    "PHASE0_ARM_ID",
    "PHASE0_SCREEN_ID",
    "PHASE0_STEPS",
    "PHASE0B_STEPS_FALLBACK_ONCE",
    "PHASE_BATCH",
    "PHASE_TOPK",
    "Q_MOTION_MIN_FRAC",
    "RETENTION_SLOP_COUNTS",
    "RET_ID_N",
    "RET_MATH_N",
    "RET_SHUFFLE_SEED",
    "Row3",
    "SUB2_ACC_BUDGET_BPW",
    "TIE_TOLERANCE_BPW",
    "apply_decay_leak",
    "apply_drain_resets",
    "apply_live_flip_writeback",
    "apply_sparse_hot",
    "apply_ttl_age_drain",
    "arm_metrics_for_classifier",
    "begin_credit_step",
    "bitlinear_absmean_quantize",
    "build_phase1_probe_sets",
    "build_phase1_terminal_receipt",
    "classify_forgetting_family_screen",
    "compact_rows_sha256",
    "cumulative_q_transitions",
    "decide_phase0_aggregate_transition",
    "entropy_bits",
    "fixed_qscale_linear_with_credit",
    "flattened_nd_dW",
    "g0_valid",
    "g0b_q_motion_ok",
    "g1_survival_ok",
    "get_credit_store",
    "lifetime_censored_frac",
    "load_identity_full_rows",
    "load_math_a0_rows",
    "mechanical_dynamic_scale_diverges",
    "parent_support_hash16",
    "qscale_reference_weight",
    "retention_ok",
    "sample_batch_excluding_acquisition",
    "select_acquisition_rows",
    "select_retention_identity_rows",
    "select_retention_math_rows",
    "should_record_h_trajectory",
    "snapshot_route_counters",
    "threshold_residual_writeback",
    "update_episode_starts",
    "validate_phase0_receipt_for_aggregate",
    "validate_shared_held_fixed_arm_receipts",
]
