"""W8 dense-accumulator in-vivo confirmation envelope + postrun classifier helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import signed_w_max
from calm.hrm_text_158.native_full_stack.w7_dense_acc_in_vivo_confirmation import (
    CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24,
    CONFIRMATION_VOTE_PATH_RANK_BUCKETED,
    ConfirmationEnvelope,
    MEANINGFUL_PARITY_PRESSURE_PEAK,
    NATIVE_LOOP_INJECTION_CONFIRMATION_PENDING,
    ObservedVotePressure,
    S3BB_DOMAIN_HEALTH_PRIMARIES,
    S3BB_HARNESS_HEALTH_PRIMARIES,
    S3BB_PARITY_OK_PRIMARIES,
    S3BB_SCIENCE_PARITY_BREAK_PRIMARIES,
    STRUCTURAL_REASON_O1_MISSING_EVIDENCE,
    _vote_abs_max_from_pressure_entry,
    clip_abs_for_width_bits,
    live_reachable_peak_estimate,
    live_reachable_peak_estimate_from_observed,
    observed_vote_pressure_from_receipt,
    resolve_confirmation_envelope,
)

CLASSIFIER_RUN_HEALTH_FAIL = "RUN_HEALTH_FAIL"
CLASSIFIER_HARNESS_INVALID = "HARNESS_INVALID"
CLASSIFIER_OBSERVER_TOO_EXPENSIVE = "OBSERVER_TOO_EXPENSIVE"
CLASSIFIER_W8_FLOOR_RISES_WITH_LIVE_VOTE_MAX = "W8_FLOOR_RISES_WITH_LIVE_VOTE_MAX"
CLASSIFIER_W8_BREAKS_LIVE_PARITY = "W8_BREAKS_LIVE_PARITY"
CLASSIFIER_ENVELOPE_UNDER_PRESSED = "ENVELOPE_UNDER_PRESSED"
CLASSIFIER_LIVE_FLOOR_MUCH_BELOW_W8 = "LIVE_FLOOR_MUCH_BELOW_W8"
CLASSIFIER_W8_IN_VIVO_TRANSPARENT = "W8_IN_VIVO_TRANSPARENT"
CLASSIFIER_W8_IN_VIVO_CONFIRMED = "W8_IN_VIVO_CONFIRMED"

CLASSIFIER_PRECEDENCE: tuple[str, ...] = (
    CLASSIFIER_RUN_HEALTH_FAIL,
    CLASSIFIER_HARNESS_INVALID,
    CLASSIFIER_OBSERVER_TOO_EXPENSIVE,
    CLASSIFIER_W8_FLOOR_RISES_WITH_LIVE_VOTE_MAX,
    CLASSIFIER_W8_BREAKS_LIVE_PARITY,
    CLASSIFIER_ENVELOPE_UNDER_PRESSED,
    CLASSIFIER_LIVE_FLOOR_MUCH_BELOW_W8,
    CLASSIFIER_W8_IN_VIVO_TRANSPARENT,
    CLASSIFIER_W8_IN_VIVO_CONFIRMED,
)

CPU_PHASE0_STRUCTURAL_FLOOR_WIDTH = 8
CPU_PHASE0_REACHABLE_PEAK = 33
W8_CLIP_ABS = signed_w_max(8)
W8_IN_VIVO_CONFIRMED_MIN_PEAK = 33

EXPLICIT_NON_CLAIMS: tuple[str, ...] = (
    "w8_carrier_faithfulness_to_current_plus_minus_127_clamped_production_not_universal_transparency",
    "oracle_and_treatment_both_share_vote_update_plus_minus_127_storage_clamp",
    "w8_2x_dominator_reduction_not_sub2_inclusive",
    "not_held_rules_unlock",
    "canonical_w7_negative_2189e72008_stands",
    "not_gpu_launch_in_code_readiness_slice",
    "w8_transparency_banked_only_on_w8_in_vivo_confirmed",
)

PREREG_PACKET_W8_BREAKS_PARITY_CITATION = (
    "artifacts/consensus_prep/w8_dense_acc_in_vivo_gpu_launch_packet_v1_draft.json:"
    "classifier_labels.W8_BREAKS_LIVE_PARITY"
)
PREREG_W8_BREAKS_PARITY_PREDICATES = (
    "crossing/applied-mask/q/final parity on clean sidecars"
)
DIVERGENCE_CHARACTERIZATION_W8_INT8_ROUNDTRIP_FEEDBACK = (
    "int8_roundtrip_feedback_loop_divergence"
)

W8_ACCUMULATOR_CLIP_CONTRACT: dict[str, Any] = {
    "contract_id": "source_clip_lossless_w8_alignment",
    "storage_clamp": {
        "operation": "vote_update (decay+vote).clamp(±127) into exact_accumulator_shadow",
        "bounds": "±127 vote_update spec",
        "shared_by_both_arms": True,
        "citations": [
            "vote_update.py:968-970",
            "accumulator_real_dynamics_verdict.py:560-561",
            "bounded_delta_learner.py:1080,1101",
        ],
    },
    "w8_read_boundary": {
        "operation": "clip_then_roundtrip_w8_tensor per int16 lane",
        "bounds": "±127 (source-clip-lossless; no shrink vs storage clamp)",
        "site": "vote_update_state via apply_trainer_boundary_narrow_carrier",
        "citations": [
            "narrow_accumulator_codec.py:562-566",
            "narrow_carrier_trainer_integration.py",
        ],
    },
    "clamp_vs_clamp_note": (
        "Oracle int16 WITHOUT W8 trainer clip; BOTH arms share ±127 storage clamp. "
        "W8_IN_VIVO_CONFIRMED proves W8-carrier faithfulness to CURRENT ±127-clamped "
        "production dynamics — NOT universal transparency."
    ),
    "conditional_on_production_clamp": "[-127,+127]",
    "w8_floor_witness": "vote-derived live_reachable_peak_estimate (not raw sidecar max)",
}


def _bit_equality_lane_witness(w8_o1_stats: Mapping[str, Any]) -> dict[str, Any]:
    total_lane_count = int(w8_o1_stats.get("total_lane_count") or 0)
    matched_key_compared_lane_count = int(
        (w8_o1_stats.get("sidecar_coverage_diagnostics") or {}).get(
            "matched_key_compared_lane_count"
        )
        or 0
    )
    if matched_key_compared_lane_count == 0:
        matched_key_compared_lane_count = total_lane_count
    equality_rate = float(
        w8_o1_stats.get("vote_update_state_accumulator_equality_rate") or 0.0
    )
    vacuous = total_lane_count <= 0
    return {
        "total_lane_count": total_lane_count,
        "matched_key_compared_lane_count": matched_key_compared_lane_count,
        "vote_update_state_accumulator_equality_rate": equality_rate,
        "o1_lane_equality_vacuous": bool(vacuous),
        "o1_lane_equality_load_bearing": not vacuous,
        "o1_witness_domain": w8_o1_stats.get("o1_witness_domain"),
        "o1_skip_policy": w8_o1_stats.get("o1_skip_policy"),
    }


def _compute_w8_o1_lane_stats(
    *,
    oracle_receipt: Mapping[str, Any] | None,
    treatment_receipt: Mapping[str, Any] | None,
    sidecar_coverage: Mapping[str, Any],
) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
        resolve_headroom_wiring_sidecar_path,
    )
    from calm.hrm_text_158.native_full_stack.w8_o1_lane_equality_witness import (
        compare_w8_o1_lane_equality_streaming,
    )

    if oracle_receipt is None or treatment_receipt is None:
        return {
            "total_lane_count": 0,
            "vote_update_state_accumulator_equality_rate": 0.0,
            "sidecar_coverage_diagnostics": dict(sidecar_coverage),
            "o1_witness_domain": "w8_signed_max_127",
            "o1_skip_policy": "warmup_only_not_w6_strict_raise",
        }
    oracle_path = resolve_headroom_wiring_sidecar_path(oracle_receipt)
    treatment_path = resolve_headroom_wiring_sidecar_path(treatment_receipt)
    if oracle_path is None or treatment_path is None:
        return {
            "total_lane_count": 0,
            "vote_update_state_accumulator_equality_rate": 0.0,
            "sidecar_coverage_diagnostics": dict(sidecar_coverage),
            "o1_witness_domain": "w8_signed_max_127",
            "o1_skip_policy": "warmup_only_not_w6_strict_raise",
        }
    return compare_w8_o1_lane_equality_streaming(
        oracle_receipt,
        treatment_receipt,
        oracle_sidecar_path=oracle_path,
        treatment_sidecar_path=treatment_path,
        sidecar_coverage=sidecar_coverage,
    )


def extract_w8_parity_signals(s3bb_stats: Mapping[str, Any]) -> dict[str, Any]:
    """Prereg O1-O4 parity keys: crossing, applied_mask, q_trajectory, final_metrics."""

    driving_keys: list[str] = []
    crossing = s3bb_stats.get("crossing_parity") or {}
    if int(crossing.get("per_step_crossing_bool_disagreement_count") or 0) > 0:
        driving_keys.append("per_step_crossing_bool_disagreement_count")

    applied = s3bb_stats.get("applied_mask_parity") or {}
    if int(applied.get("applied_mask_mismatch_count") or 0) > 0:
        driving_keys.append("applied_mask_mismatch_count")

    q_traj = s3bb_stats.get("q_trajectory_parity") or {}
    if int(q_traj.get("q_sha256_after_mismatch_count") or 0) > 0:
        driving_keys.append("q_sha256_after_mismatch_count")
    if q_traj.get("q_changed_count_mismatch_steps"):
        driving_keys.append("q_changed_count_mismatch_steps")
    if bool(q_traj.get("steps_completed_mismatch")):
        driving_keys.append("steps_completed_mismatch")
    if bool(q_traj.get("stop_reason_mismatch")):
        driving_keys.append("stop_reason_mismatch")
    if bool(q_traj.get("final_metrics_mismatch")):
        driving_keys.append("final_metrics_mismatch")

    return {
        "parity_break": bool(driving_keys),
        "parity_break_driving_keys": driving_keys,
        "crossing_parity": dict(crossing) if isinstance(crossing, Mapping) else {},
        "applied_mask_parity": dict(applied) if isinstance(applied, Mapping) else {},
        "q_trajectory_parity": dict(q_traj) if isinstance(q_traj, Mapping) else {},
    }


def prereg_o1_o4_adjudicable(
    *,
    o1_lane_witness: Mapping[str, Any],
    parity_signals: Mapping[str, Any],
) -> bool:
    if bool(o1_lane_witness.get("o1_lane_equality_load_bearing")):
        return True
    return bool(parity_signals.get("parity_break_driving_keys"))


def verify_dual_arm_w8_configuration(
    *,
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
) -> tuple[int | None, list[str]]:
    """Derive confirmed floor width only when arm flags match the dual-arm contract."""

    failures: list[str] = []
    if bool(oracle_receipt.get("dense_accumulator_w8_clip")):
        failures.append("oracle_dense_accumulator_w8_clip_must_be_false")
    if not bool(treatment_receipt.get("dense_accumulator_w8_clip")):
        failures.append("treatment_dense_accumulator_w8_clip_must_be_true")
    for label, receipt in (("oracle", oracle_receipt), ("treatment", treatment_receipt)):
        if bool(receipt.get("dense_accumulator_w7_clip")):
            failures.append(f"{label}_dense_accumulator_w7_clip_active")
        if bool(receipt.get("persistent_accumulator_w5_byte_packed")):
            failures.append(f"{label}_persistent_accumulator_w5_active")
        if bool(receipt.get("persistent_accumulator_w6_byte_packed")):
            failures.append(f"{label}_persistent_accumulator_w6_active")
        if bool(receipt.get("persistent_accumulator_event_coded_live")):
            failures.append(f"{label}_persistent_accumulator_event_coded_live_active")
    if failures:
        return None, failures
    return CPU_PHASE0_STRUCTURAL_FLOOR_WIDTH, []


def _max_sidecar_abs_from_receipt(receipt: Mapping[str, Any]) -> int | None:
    from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
        _index_sidecar_file,
        resolve_headroom_wiring_sidecar_path,
    )

    sidecar_path = resolve_headroom_wiring_sidecar_path(receipt)
    if sidecar_path is None or not Path(sidecar_path).is_file():
        return None
    keyed, _, _ = _index_sidecar_file(sidecar_path)
    max_abs = 0
    for row in keyed.values():
        lanes = row.get("accumulator_lanes") or []
        for lane in lanes:
            max_abs = max(max_abs, abs(int(lane)))
    return int(max_abs)


def derive_w8_parity_inputs(
    s3bb_primary: str,
    s3bb_stats: Mapping[str, Any],
    sidecar_coverage: Mapping[str, Any],
    *,
    oracle_receipt: Mapping[str, Any] | None = None,
    treatment_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Map S3BB decision-parity stats to W8 structural/parity bridge inputs."""

    coverage = dict(sidecar_coverage)
    base: dict[str, Any] = {
        "sidecar_coverage_diagnostics": coverage,
        "s3bb_primary_classifier": str(s3bb_primary),
        "w8_accumulator_clip_contract": dict(W8_ACCUMULATOR_CLIP_CONTRACT),
        "prereg_w8_breaks_parity_citation": PREREG_PACKET_W8_BREAKS_PARITY_CITATION,
        "prereg_w8_breaks_parity_predicates": PREREG_W8_BREAKS_PARITY_PREDICATES,
        "s3bb_w5w6_domain_primary_inapplicable": False,
        "s3bb_w5w6_domain_primary_recorded": None,
    }

    if coverage.get("structural_fail"):
        return {
            **base,
            "structural_fail": True,
            "structural_reason": str(
                coverage.get("structural_reason") or "sidecar_structural_coverage_fail"
            ),
            "parity_break": False,
            "parity_break_driving_keys": [],
            "o1_lane_equality_vacuous": True,
            "o1_lane_equality_load_bearing": False,
            "prereg_o1_o4_adjudicable": False,
        }

    primary = str(s3bb_primary)
    if primary in S3BB_HARNESS_HEALTH_PRIMARIES:
        return {
            **base,
            "structural_fail": True,
            "structural_reason": "s3bb_harness_or_liveness_fail",
            "parity_break": False,
            "parity_break_driving_keys": [],
            "o1_lane_equality_vacuous": True,
            "o1_lane_equality_load_bearing": False,
            "prereg_o1_o4_adjudicable": False,
        }

    domain_inapplicable = primary in S3BB_DOMAIN_HEALTH_PRIMARIES
    if domain_inapplicable:
        base["s3bb_w5w6_domain_primary_inapplicable"] = True
        base["s3bb_w5w6_domain_primary_recorded"] = primary
    elif (
        primary not in S3BB_PARITY_OK_PRIMARIES
        and primary not in S3BB_SCIENCE_PARITY_BREAK_PRIMARIES
    ):
        return {
            **base,
            "structural_fail": True,
            "structural_reason": "s3bb_unenumerated_primary_fail",
            "parity_break": False,
            "parity_break_driving_keys": [],
            "o1_lane_equality_vacuous": True,
            "o1_lane_equality_load_bearing": False,
            "prereg_o1_o4_adjudicable": False,
        }

    w8_o1_stats = _compute_w8_o1_lane_stats(
        oracle_receipt=oracle_receipt,
        treatment_receipt=treatment_receipt,
        sidecar_coverage=coverage,
    )
    if w8_o1_stats.get("structural_fail"):
        return {
            **base,
            "structural_fail": True,
            "structural_reason": str(
                w8_o1_stats.get("structural_reason") or "w8_o1_lane_witness_structural_fail"
            ),
            "parity_break": False,
            "parity_break_driving_keys": [],
            "o1_lane_equality_vacuous": True,
            "o1_lane_equality_load_bearing": False,
            "prereg_o1_o4_adjudicable": False,
            "w8_o1_lane_stats": dict(w8_o1_stats),
        }

    o1_lane_witness = _bit_equality_lane_witness(w8_o1_stats)
    parity_signals = extract_w8_parity_signals(s3bb_stats)
    driving_keys = list(parity_signals["parity_break_driving_keys"])
    if primary in S3BB_SCIENCE_PARITY_BREAK_PRIMARIES:
        driving_keys.insert(0, f"s3bb_primary:{primary}")
    if (
        bool(o1_lane_witness.get("o1_lane_equality_load_bearing"))
        and float(o1_lane_witness.get("vote_update_state_accumulator_equality_rate") or 0.0)
        < 1.0
    ):
        driving_keys.insert(0, "o1_accumulator_lane_inequality")
    parity_signals = {
        **parity_signals,
        "parity_break": bool(driving_keys),
        "parity_break_driving_keys": driving_keys,
    }

    adjudicable = prereg_o1_o4_adjudicable(
        o1_lane_witness=o1_lane_witness,
        parity_signals=parity_signals,
    )
    if not adjudicable:
        return {
            **base,
            "structural_fail": True,
            "structural_reason": STRUCTURAL_REASON_O1_MISSING_EVIDENCE,
            "parity_break": False,
            "parity_break_driving_keys": [],
            "o1_lane_equality_vacuous": bool(o1_lane_witness["o1_lane_equality_vacuous"]),
            "o1_lane_equality_load_bearing": False,
            "prereg_o1_o4_adjudicable": False,
            **o1_lane_witness,
            **parity_signals,
        }

    oracle_max_abs = (
        _max_sidecar_abs_from_receipt(oracle_receipt) if oracle_receipt is not None else None
    )
    treatment_max_abs = (
        _max_sidecar_abs_from_receipt(treatment_receipt)
        if treatment_receipt is not None
        else None
    )

    return {
        **base,
        "structural_fail": False,
        "structural_reason": None,
        "parity_break": bool(driving_keys),
        "parity_break_driving_keys": driving_keys,
        "o1_lane_equality_vacuous": bool(o1_lane_witness["o1_lane_equality_vacuous"]),
        "o1_lane_equality_load_bearing": bool(o1_lane_witness["o1_lane_equality_load_bearing"]),
        "prereg_o1_o4_adjudicable": True,
        "oracle_max_sidecar_abs": oracle_max_abs,
        "treatment_max_sidecar_abs": treatment_max_abs,
        "w8_o1_lane_stats": dict(w8_o1_stats),
        **o1_lane_witness,
        **parity_signals,
    }


def classify_w8_in_vivo_dual_arm(
    *,
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
    envelope: ConfirmationEnvelope | None,
    observer_too_expensive: bool = False,
    harness_failures: Sequence[str] = (),
    parity_break: bool = False,
    structural_fail: bool = False,
    structural_reason: str | None = None,
    confirmed_vote_acc_floor_width: int | None = None,
    oracle_max_sidecar_abs: int | None = None,
    treatment_max_sidecar_abs: int | None = None,
) -> dict[str, Any]:
    failures = list(harness_failures)
    if structural_fail:
        failures.append(str(structural_reason or "sidecar_structural_coverage_fail"))
    if envelope is None:
        failures.append("missing_confirmation_envelope")
    elif str(oracle_receipt.get("envelope_id") or "") != envelope.envelope_id:
        failures.append("oracle_envelope_id_mismatch")
    elif str(treatment_receipt.get("envelope_id") or "") != envelope.envelope_id:
        failures.append("treatment_envelope_id_mismatch")
    oracle_pressure = observed_vote_pressure_from_receipt(oracle_receipt)
    treatment_pressure = observed_vote_pressure_from_receipt(treatment_receipt)
    if oracle_pressure.source == "none":
        failures.append("oracle_missing_observed_vote_pressure")
    if treatment_pressure.source == "none":
        failures.append("treatment_missing_observed_vote_pressure")
    if failures:
        return _classifier_payload(
            primary=CLASSIFIER_RUN_HEALTH_FAIL,
            envelope=envelope,
            oracle_receipt=oracle_receipt,
            treatment_receipt=treatment_receipt,
            failures=failures,
            confirmed_vote_acc_floor_width=confirmed_vote_acc_floor_width,
            structural_fail=structural_fail,
            structural_reason=structural_reason,
            oracle_pressure=oracle_pressure,
            treatment_pressure=treatment_pressure,
        )
    if observer_too_expensive:
        return _classifier_payload(
            primary=CLASSIFIER_OBSERVER_TOO_EXPENSIVE,
            envelope=envelope,
            oracle_receipt=oracle_receipt,
            treatment_receipt=treatment_receipt,
            failures=[],
            confirmed_vote_acc_floor_width=confirmed_vote_acc_floor_width,
            structural_fail=structural_fail,
            structural_reason=structural_reason,
            oracle_pressure=oracle_pressure,
            treatment_pressure=treatment_pressure,
        )
    assert envelope is not None
    threshold = int(envelope.live_threshold_abs)
    max_vote_obs = max(
        int(oracle_pressure.max_vote_abs),
        int(treatment_pressure.max_vote_abs),
    )
    peak_obs = live_reachable_peak_estimate(
        threshold_abs=threshold,
        max_vote_abs_observed=max_vote_obs,
    )
    if peak_obs > W8_CLIP_ABS:
        primary = CLASSIFIER_W8_FLOOR_RISES_WITH_LIVE_VOTE_MAX
    elif parity_break:
        primary = CLASSIFIER_W8_BREAKS_LIVE_PARITY
    elif peak_obs < MEANINGFUL_PARITY_PRESSURE_PEAK:
        primary = CLASSIFIER_ENVELOPE_UNDER_PRESSED
    elif (
        peak_obs >= W8_IN_VIVO_CONFIRMED_MIN_PEAK
        and confirmed_vote_acc_floor_width == CPU_PHASE0_STRUCTURAL_FLOOR_WIDTH
        and not parity_break
    ):
        primary = CLASSIFIER_W8_IN_VIVO_CONFIRMED
    elif (
        peak_obs < W8_IN_VIVO_CONFIRMED_MIN_PEAK
        and not parity_break
        and oracle_max_sidecar_abs is not None
        and treatment_max_sidecar_abs is not None
        and oracle_max_sidecar_abs <= W8_CLIP_ABS
        and treatment_max_sidecar_abs <= W8_CLIP_ABS
    ):
        primary = CLASSIFIER_W8_IN_VIVO_TRANSPARENT
    elif peak_obs < W8_IN_VIVO_CONFIRMED_MIN_PEAK:
        primary = CLASSIFIER_LIVE_FLOOR_MUCH_BELOW_W8
    else:
        primary = CLASSIFIER_W8_BREAKS_LIVE_PARITY
    return _classifier_payload(
        primary=primary,
        envelope=envelope,
        oracle_receipt=oracle_receipt,
        treatment_receipt=treatment_receipt,
        failures=[],
        live_max_vote_abs_observed=max_vote_obs,
        live_reachable_peak_estimate=peak_obs,
        confirmed_vote_acc_floor_width=confirmed_vote_acc_floor_width,
        parity_break=parity_break,
        structural_fail=structural_fail,
        structural_reason=structural_reason,
        oracle_pressure=oracle_pressure,
        treatment_pressure=treatment_pressure,
        oracle_max_sidecar_abs=oracle_max_sidecar_abs,
        treatment_max_sidecar_abs=treatment_max_sidecar_abs,
    )


def _classifier_payload(
    *,
    primary: str,
    envelope: ConfirmationEnvelope | None,
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
    failures: Sequence[str],
    live_max_vote_abs_observed: int | None = None,
    live_reachable_peak_estimate: int | None = None,
    confirmed_vote_acc_floor_width: int | None = None,
    parity_break: bool = False,
    structural_fail: bool = False,
    structural_reason: str | None = None,
    oracle_pressure: ObservedVotePressure | None = None,
    treatment_pressure: ObservedVotePressure | None = None,
    oracle_max_sidecar_abs: int | None = None,
    treatment_max_sidecar_abs: int | None = None,
) -> dict[str, Any]:
    threshold = int(envelope.live_threshold_abs) if envelope is not None else None
    if oracle_pressure is None:
        oracle_pressure = observed_vote_pressure_from_receipt(oracle_receipt)
    if treatment_pressure is None:
        treatment_pressure = observed_vote_pressure_from_receipt(treatment_receipt)
    if live_max_vote_abs_observed is None:
        if oracle_pressure.source == "none" and treatment_pressure.source == "none":
            live_max_vote_abs_observed = None
        else:
            live_max_vote_abs_observed = max(
                int(oracle_pressure.max_vote_abs),
                int(treatment_pressure.max_vote_abs),
            )
    if live_reachable_peak_estimate is None and threshold is not None:
        if live_max_vote_abs_observed is None:
            live_reachable_peak_estimate = None
        else:
            live_reachable_peak_estimate = live_reachable_peak_estimate_from_observed(
                threshold_abs=threshold,
                max_vote_abs_observed=int(live_max_vote_abs_observed),
            )
    observed_step_count = max(
        int(oracle_pressure.observed_step_count),
        int(treatment_pressure.observed_step_count),
    )
    if oracle_pressure.source == "none" and treatment_pressure.source == "none":
        live_max_vote_abs_source = "none"
    elif oracle_pressure.source != "none" and treatment_pressure.source != "none":
        live_max_vote_abs_source = "step_reports.vote_pressure"
    else:
        live_max_vote_abs_source = "partial_step_reports.vote_pressure"
    floor_width = confirmed_vote_acc_floor_width
    floor_clip = (
        clip_abs_for_width_bits(int(floor_width)) if floor_width is not None else None
    )
    banks_w8_transparency = primary == CLASSIFIER_W8_IN_VIVO_CONFIRMED
    banks_w8_carrier_faithfulness = primary == CLASSIFIER_W8_IN_VIVO_CONFIRMED
    informative_only = primary in {
        CLASSIFIER_W8_IN_VIVO_TRANSPARENT,
        CLASSIFIER_LIVE_FLOOR_MUCH_BELOW_W8,
        CLASSIFIER_ENVELOPE_UNDER_PRESSED,
    }
    return {
        "schema_version": "hrm_text_158_w8_dense_acc_in_vivo_classifier/v1",
        "primary_classifier": primary,
        "classifier_precedence": list(CLASSIFIER_PRECEDENCE),
        "explicit_non_claims": list(EXPLICIT_NON_CLAIMS),
        "confirmation_vote_path": (
            envelope.confirmation_vote_path if envelope is not None else None
        ),
        "envelope_id": envelope.envelope_id if envelope is not None else None,
        "live_threshold_abs": threshold,
        "wired_max_vote_abs": (
            int(envelope.wired_max_vote_abs) if envelope is not None else None
        ),
        "wired_reachable_peak_estimate": (
            int(envelope.wired_reachable_peak_estimate) if envelope is not None else None
        ),
        "live_max_vote_abs_observed": live_max_vote_abs_observed,
        "live_max_vote_abs_source": live_max_vote_abs_source,
        "observed_step_count": int(observed_step_count),
        "live_reachable_peak_estimate": live_reachable_peak_estimate,
        "confirmed_vote_acc_floor_width": floor_width,
        "confirmed_vote_acc_floor_clip_abs": floor_clip,
        "cpu_phase0_structural_floor_width": CPU_PHASE0_STRUCTURAL_FLOOR_WIDTH,
        "cpu_phase0_reachable_peak": CPU_PHASE0_REACHABLE_PEAK,
        "native_loop_injection_confirmation": NATIVE_LOOP_INJECTION_CONFIRMATION_PENDING,
        "banks_w8_transparency": bool(banks_w8_transparency),
        "banks_w8_carrier_faithfulness": bool(banks_w8_carrier_faithfulness),
        "informative_only": bool(informative_only),
        "eager_tier_rules_unlock": False,
        "parity_break": bool(parity_break),
        "structural_fail": bool(structural_fail),
        "structural_reason": structural_reason,
        "harness_failures": list(failures),
        "oracle_max_sidecar_abs": oracle_max_sidecar_abs,
        "treatment_max_sidecar_abs": treatment_max_sidecar_abs,
        "w8_clip_abs": int(W8_CLIP_ABS),
    }


__all__ = [
    "CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24",
    "CONFIRMATION_VOTE_PATH_RANK_BUCKETED",
    "CLASSIFIER_ENVELOPE_UNDER_PRESSED",
    "CLASSIFIER_LIVE_FLOOR_MUCH_BELOW_W8",
    "CLASSIFIER_RUN_HEALTH_FAIL",
    "CLASSIFIER_W8_BREAKS_LIVE_PARITY",
    "CLASSIFIER_W8_FLOOR_RISES_WITH_LIVE_VOTE_MAX",
    "CLASSIFIER_W8_IN_VIVO_CONFIRMED",
    "CLASSIFIER_W8_IN_VIVO_TRANSPARENT",
    "ConfirmationEnvelope",
    "W8_ACCUMULATOR_CLIP_CONTRACT",
    "classify_w8_in_vivo_dual_arm",
    "derive_w8_parity_inputs",
    "extract_w8_parity_signals",
    "prereg_o1_o4_adjudicable",
    "resolve_confirmation_envelope",
    "verify_dual_arm_w8_configuration",
]
