"""Compat re-export facade + PLAN_v10 contract re-exports + file IO/load.

Historical PLAN_v9 seam re-exports remain for characterization imports.
Live v10 authority lives in forgetting_screen_v10_contract.py (pure mapping).
File load/hash for control baseline + three-arm shared-geometry live here.

Do NOT mutate vote_lifetime_screen_reducers.py (frozen-q zero-drain contract).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

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
    classify_forgetting_family_screen,  # PLAN_v9 historical
    g0_valid,  # PLAN_v9 historical (censor-era)
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
    apply_sparse_hot_with_count,
    apply_ttl_age_drain,
    apply_ttl_age_drain_with_count,
    entropy_bits,
    should_record_h_trajectory,
    threshold_residual_writeback,
)
from calm.hrm_text_158.native_full_stack.forgetting_screen_v10_contract import (  # noqa: F401
    BACKLOG_PROGRESS_BAR,
    CLASS_EXIT_WINNERS,
    FORMAL150_CONTROL_SHA256,
    HIGH_DEMAND_RATIO,
    MIN_COHORT_N,
    PRESSURE_PROGRESS_BAR,
    RECOGNIZED_DEFERRED_SURVIVAL_ENUM,
    SUSTAINED_HIGH_DEMAND_FRAC_STEPS,
    arm_metrics_for_v10_classifier,
    backlog_bar_v10,
    build_v10_terminal_receipt,
    classify_forgetting_family_screen_v10,
    g0_valid_v10,
    h_bar_v10,
    pressure_bar_v10,
    recompute_deferred_survival_class,
    regime_exit_v10,
    validate_control_baseline_bind,
)
from calm.hrm_text_158.native_full_stack.forgetting_screen_v10_1_contract import (  # noqa: F401
    AUTHORITY_DISPATCH_V10_1,
    CONTROL_CREDITED_MASS,
    CREDITED_MASS_BAND,
    PLAN_V10_1_PATH,
    PLAN_V10_1_SHA256,
    PRE_POST_SCHEMA,
    SUPPRESSION_ARM,
    TERMINAL_LABEL_CANONICAL,
    TERMINAL_MODE_DEGENERATE_FULL_SUPPRESSION,
    TRANSFER_CARDINALITY,
    TRANSFER_LAW,
    build_exhaustive_transfer_table,
    build_terminal_label_canonical,
    classify_discriminator_branch,
    credited_mass_ratio,
    credited_mass_ratio_in_band,
    g0_valid_v10_1,
    pre_post_evidence_schema_valid,
    suppression_diagnostic_match,
    suppression_disposition,
    transfer_pair,
    trunc_toward_zero_mul_31_32,
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
    optional_json_float,
    sanitize_receipt_for_strict_json,
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


def load_and_validate_control_baseline(
    path: str | Path,
    *,
    expected_sha256: str,
    require_exact_c2_echo: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Facade owns file IO/hash — contract stays mapping-pure."""
    raw = Path(path).read_bytes()
    return validate_control_baseline_bind(
        json.loads(raw.decode("utf-8")),
        expected_sha256=expected_sha256,
        actual_sha256=hashlib.sha256(raw).hexdigest(),
        require_exact_c2_echo=require_exact_c2_echo,
    )


def pin_and_load_formal_control_baseline(
    path: str | Path,
    *,
    supplied_sha256: str,
    require_exact_c2_echo: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pin supplied sha to FORMAL150 BEFORE load; then file bytes must match."""
    if str(supplied_sha256) != FORMAL150_CONTROL_SHA256:
        return {
            "ok": False,
            "reason": "control_baseline_sha_not_pinned",
            "action": "stop",
            "expected_sha256": FORMAL150_CONTROL_SHA256,
            "supplied_sha256": str(supplied_sha256),
        }
    return load_and_validate_control_baseline(
        path,
        expected_sha256=FORMAL150_CONTROL_SHA256,
        require_exact_c2_echo=require_exact_c2_echo,
    )


class V10ArmReceiptContractError(ValueError):
    """Fail-closed three-mechanism-arm shared geometry (v10 surface)."""


def validate_three_mechanism_arm_receipts_v10(
    by_arm: Mapping[str, Mapping[str, Any]],
    *,
    expected_plan_sha256: str,
    expected_parent_sha256: str,
    expected_authority_dispatch: str = DEFAULT_AUTHORITY_DISPATCH,
    expected_acq_sha: str = ACQUISITION_SELECTION_SHA256,
    expected_id_sha: str = IDENTITY_SELECTION_SHA256,
    expected_batch: int = PHASE_BATCH,
    expected_topk: int = PHASE_TOPK,
    expected_steps: int = PHASE0_STEPS,
) -> dict[str, Any]:
    """Exactly ARM1/ARM2/ARM3 — formal artifact is sole control (never arm0)."""
    required = (ARM1, ARM2, ARM3)
    for arm in required:
        if arm not in by_arm:
            raise V10ArmReceiptContractError(f"missing arm receipt for {arm}")
    ref = by_arm[ARM1]
    ref_scale = str((ref.get("frozen_scale_sha") or {}).get("before") or "")
    ref_q = str((ref.get("q_sha") or {}).get("before") or "")
    if not ref_scale:
        raise V10ArmReceiptContractError("arm1 missing frozen_scale_sha.before")
    if not ref_q:
        raise V10ArmReceiptContractError("arm1 missing q_sha.before")
    for arm in required:
        r = by_arm[arm]
        if str(r.get("arm")) != arm:
            raise V10ArmReceiptContractError(
                f"arm label mismatch: key={arm} receipt.arm={r.get('arm')!r}"
            )
        if str(r.get("plan_sha256")) != str(expected_plan_sha256):
            raise V10ArmReceiptContractError(f"{arm} plan_sha256 mismatch")
        if str(r.get("authority_dispatch")) != str(expected_authority_dispatch):
            raise V10ArmReceiptContractError(f"{arm} authority_dispatch mismatch")
        banked = r.get("banked_sha") or {}
        if not bool(banked.get("match")):
            raise V10ArmReceiptContractError(f"{arm} banked_sha.match is not true")
        if str(banked.get("before")) != str(expected_parent_sha256):
            raise V10ArmReceiptContractError(f"{arm} banked parent mismatch")
        if str(banked.get("after")) != str(expected_parent_sha256):
            raise V10ArmReceiptContractError(f"{arm} banked after mismatch")
        scale = r.get("frozen_scale_sha") or {}
        if not bool(scale.get("match")):
            raise V10ArmReceiptContractError(f"{arm} frozen_scale_sha.match is not true")
        if str(scale.get("before")) != ref_scale:
            raise V10ArmReceiptContractError(f"{arm} frozen_scale_sha.before divergence")
        qsha = r.get("q_sha") or {}
        if str(qsha.get("before")) != ref_q:
            raise V10ArmReceiptContractError(f"{arm} q_sha.before divergence")
        try:
            arm_steps = int(r.get("steps", -1))
            arm_batch = int(r.get("batch", -1))
            arm_topk = int(r.get("topk", -1))
        except (TypeError, ValueError) as e:
            raise V10ArmReceiptContractError(f"{arm} geometry invalid: {e}") from e
        if arm_steps != int(expected_steps):
            raise V10ArmReceiptContractError(f"{arm} steps {arm_steps} != {expected_steps}")
        if arm_batch != int(expected_batch):
            raise V10ArmReceiptContractError(f"{arm} batch {arm_batch} != {expected_batch}")
        if arm_topk != int(expected_topk):
            raise V10ArmReceiptContractError(f"{arm} topk {arm_topk} != {expected_topk}")
        if str(r.get("screen")) != str(PHASE0_SCREEN_ID):
            raise V10ArmReceiptContractError(f"{arm} screen mismatch")
        if bool(r.get("schema_only", False)) or bool(r.get("correctness_smoke", False)):
            raise V10ArmReceiptContractError(f"{arm} schema_only/correctness_smoke must be false")
        route = r.get("route_counters") or {}
        try:
            n_fixed = int(route.get("n_fixed_qscale_forwards", 0))
            n_dyn = int(route.get("n_bitlinear_dynamic_forwards", -1))
            n_elig = int(route.get("n_eligible_keys", -1))
            n_cred = int(route.get("n_credit_grads_present", -2))
        except (TypeError, ValueError) as e:
            raise V10ArmReceiptContractError(f"{arm} route_counters invalid: {e}") from e
        if n_fixed <= 0 or n_dyn != 0 or n_elig <= 0 or n_cred <= 0 or n_elig != n_cred:
            raise V10ArmReceiptContractError(f"{arm} route/probe coverage fail-closed")
        probes = r.get("probes") or {}
        if bool(probes.get("skipped", True)):
            raise V10ArmReceiptContractError(f"{arm} probes.skipped must be false")
        if str(probes.get("acquisition_selection_sha256")) != str(expected_acq_sha):
            raise V10ArmReceiptContractError(f"{arm} acquisition selection sha mismatch")
        if str(probes.get("identity_selection_sha256")) != str(expected_id_sha):
            raise V10ArmReceiptContractError(f"{arm} identity selection sha mismatch")
        if int(probes.get("acquisition_n", -1)) != ACQ_N:
            raise V10ArmReceiptContractError(f"{arm} acquisition_n != {ACQ_N}")
        if int(probes.get("retention_n", -1)) != ACQ_N:
            raise V10ArmReceiptContractError(f"{arm} retention_n != {ACQ_N}")
        for need in (
            "acq_step0_count", "acq_final_count", "acq_delta_count",
            "retention_step0_count", "retention_final_count", "retention_ok",
        ):
            if need not in probes:
                raise V10ArmReceiptContractError(f"{arm} probes missing {need}")
        # R1 surface must be present on mechanism receipts (producer contract).
        meas = r.get("measurements") or {}
        demand = meas.get("demand") or {}
        ds = meas.get("deferred_survival") or {}
        for k in ("mean_ratio", "max_ratio", "frac_steps_ratio_ge_2", "n_steps"):
            if k not in demand:
                raise V10ArmReceiptContractError(f"{arm} measurements.demand missing {k}")
        for k in (
            "N_events_evaluable", "N_survived_applied_within_H", "N_never_applied_within_H",
            "N_events_censored_insufficient_followup", "N_events_evaluable_early",
            "N_events_evaluable_late", "N_never_applied_within_H_early",
            "N_never_applied_within_H_late", "deferred_never_apply_within_H_frac",
            "deferred_never_apply_within_H_frac_early",
            "deferred_never_apply_within_H_frac_late", "delta_never_apply",
            "deferred_survival_class",
        ):
            if k not in ds:
                raise V10ArmReceiptContractError(
                    f"{arm} measurements.deferred_survival missing {k}"
                )
    return {
        "steps": int(expected_steps),
        "batch": int(expected_batch),
        "topk": int(expected_topk),
        "parent_sha256": expected_parent_sha256,
        "plan_sha256": expected_plan_sha256,
        "authority_dispatch": expected_authority_dispatch,
        "frozen_scale_sha_before": ref_scale,
        "q_sha_before": ref_q,
        "mechanism_arms": list(required),
        "control": "formal150_artifact_sole",
    }


__all__ = [
    "ACQ_N",
    "ACQ_SHUFFLE_SEED",
    "ACQUISITION_SELECTION_SHA256",
    "ARM0",
    "ARM1",
    "ARM2",
    "ARM3",
    "ArmReceiptContractError",
    "BACKLOG_PROGRESS_BAR",
    "CENSOR_CLEAR_MAX",
    "CLASS_EXIT_WINNERS",
    "CONTROL_CREDITED_MASS",
    "CREDITED_MASS_BAND",
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
    "FORMAL150_CONTROL_SHA256",
    "FixedQScaleLinearWithCredit",
    "H_PROGRESS_BAR_FRAC",
    "H_TRAJECTORY_EVERY",
    "HIGH_DEMAND_RATIO",
    "IDENTITY_PARENT_SUPPORT_HASH",
    "IDENTITY_SELECTION_SHA256",
    "MATH_A0_PARENT_SUPPORT_HASH",
    "MIN_COHORT_N",
    "N_FLIPS_VACUOUS",
    "PHASE0_ALLOWED_STEPS",
    "PHASE0_ARM_ID",
    "PHASE0_SCREEN_ID",
    "PHASE0_STEPS",
    "PHASE0B_STEPS_FALLBACK_ONCE",
    "PHASE_BATCH",
    "PHASE_TOPK",
    "PLAN_V10_1_PATH",
    "PLAN_V10_1_SHA256",
    "AUTHORITY_DISPATCH_V10_1",
    "TERMINAL_LABEL_CANONICAL",
    "build_terminal_label_canonical",
    "PRE_POST_SCHEMA",
    "PRESSURE_PROGRESS_BAR",
    "Q_MOTION_MIN_FRAC",
    "RECOGNIZED_DEFERRED_SURVIVAL_ENUM",
    "RETENTION_SLOP_COUNTS",
    "RET_ID_N",
    "RET_MATH_N",
    "RET_SHUFFLE_SEED",
    "Row3",
    "SUB2_ACC_BUDGET_BPW",
    "SUPPRESSION_ARM",
    "SUSTAINED_HIGH_DEMAND_FRAC_STEPS",
    "TERMINAL_MODE_DEGENERATE_FULL_SUPPRESSION",
    "TIE_TOLERANCE_BPW",
    "TRANSFER_CARDINALITY",
    "TRANSFER_LAW",
    "V10ArmReceiptContractError",
    "apply_decay_leak",
    "apply_drain_resets",
    "apply_live_flip_writeback",
    "apply_sparse_hot",
    "apply_sparse_hot_with_count",
    "apply_ttl_age_drain",
    "apply_ttl_age_drain_with_count",
    "arm_metrics_for_classifier",
    "arm_metrics_for_v10_classifier",
    "backlog_bar_v10",
    "begin_credit_step",
    "bitlinear_absmean_quantize",
    "build_exhaustive_transfer_table",
    "build_phase1_probe_sets",
    "build_phase1_terminal_receipt",
    "build_v10_terminal_receipt",
    "classify_discriminator_branch",
    "classify_forgetting_family_screen",
    "classify_forgetting_family_screen_v10",
    "compact_rows_sha256",
    "credited_mass_ratio",
    "credited_mass_ratio_in_band",
    "cumulative_q_transitions",
    "decide_phase0_aggregate_transition",
    "entropy_bits",
    "fixed_qscale_linear_with_credit",
    "flattened_nd_dW",
    "g0_valid",
    "g0_valid_v10",
    "g0_valid_v10_1",
    "g0b_q_motion_ok",
    "g1_survival_ok",
    "get_credit_store",
    "h_bar_v10",
    "lifetime_censored_frac",
    "load_and_validate_control_baseline",
    "load_identity_full_rows",
    "load_math_a0_rows",
    "mechanical_dynamic_scale_diverges",
    "optional_json_float",
    "parent_support_hash16",
    "pin_and_load_formal_control_baseline",
    "pre_post_evidence_schema_valid",
    "pressure_bar_v10",
    "qscale_reference_weight",
    "recompute_deferred_survival_class",
    "regime_exit_v10",
    "retention_ok",
    "sample_batch_excluding_acquisition",
    "sanitize_receipt_for_strict_json",
    "select_acquisition_rows",
    "select_retention_identity_rows",
    "select_retention_math_rows",
    "should_record_h_trajectory",
    "snapshot_route_counters",
    "suppression_diagnostic_match",
    "suppression_disposition",
    "threshold_residual_writeback",
    "transfer_pair",
    "trunc_toward_zero_mul_31_32",
    "update_episode_starts",
    "validate_control_baseline_bind",
    "validate_phase0_receipt_for_aggregate",
    "validate_shared_held_fixed_arm_receipts",
    "validate_three_mechanism_arm_receipts_v10",
]
