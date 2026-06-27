"""W7 dense-accumulator in-vivo confirmation envelope + postrun classifier helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.accumulator_real_dynamics_verdict import (
    default_vote_update_spec as canonical_acquisition_vote_update_spec,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    RankVoteSpec,
    canonical_acquisition_peak_reachable,
    canonical_acquisition_rank_vote_spec,
    default_dry_run_rank_vote_spec,
    dry_run_rank_vote_peak_reachable,
    max_vote_abs_for_rank_spec,
)
from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (
    canonical_rank_bin_spec_sha256,
    canonical_rank_vote_spec,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    effective_clip_bounds,
    signed_w_max,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec

CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24 = "canonical_t10_prereg_v24"
CONFIRMATION_VOTE_PATH_RANK_BUCKETED = "rank_bucketed_acquisition_canonical"
NATIVE_LOOP_INJECTION_CONFIRMATION_PENDING = "pending"

CLASSIFIER_RUN_HEALTH_FAIL = "RUN_HEALTH_FAIL"
CLASSIFIER_OBSERVER_TOO_EXPENSIVE = "OBSERVER_TOO_EXPENSIVE"
CLASSIFIER_W7_FLOOR_RISES_WITH_LIVE_VOTE_MAX = "W7_FLOOR_RISES_WITH_LIVE_VOTE_MAX"
CLASSIFIER_W7_BREAKS_LIVE_PARITY = "W7_BREAKS_LIVE_PARITY"
CLASSIFIER_W7_IN_VIVO_CONFIRMED = "W7_IN_VIVO_CONFIRMED"
CLASSIFIER_LIVE_FLOOR_MUCH_BELOW_W7 = "LIVE_FLOOR_MUCH_BELOW_W7"
CLASSIFIER_ENVELOPE_UNDER_PRESSED = "ENVELOPE_UNDER_PRESSED"

S3BB_PARITY_OK_PRIMARIES: frozenset[str] = frozenset(
    {
        "DECISION_PARITY_OK",
        "W6_HEADROOM_SUFFICIENT_PARITY_OK",
        "W5_DECISION_PARITY_OK",
        "PARITY_OK",
    }
)
S3BB_SCIENCE_PARITY_BREAK_PRIMARIES: frozenset[str] = frozenset(
    {
        "DECISION_MISMATCH",
        "FLIP_EQUIVALENT_DYNAMICS_DRIFT",
    }
)
S3BB_HARNESS_HEALTH_PRIMARIES: frozenset[str] = frozenset({"HARNESS_OR_LIVENESS_FAIL"})
S3BB_DOMAIN_HEALTH_PRIMARIES: frozenset[str] = frozenset({"DOMAIN_OR_HEADROOM_FAIL"})

CLASSIFIER_PRECEDENCE: tuple[str, ...] = (
    CLASSIFIER_RUN_HEALTH_FAIL,
    CLASSIFIER_OBSERVER_TOO_EXPENSIVE,
    CLASSIFIER_W7_FLOOR_RISES_WITH_LIVE_VOTE_MAX,
    CLASSIFIER_W7_BREAKS_LIVE_PARITY,
    CLASSIFIER_ENVELOPE_UNDER_PRESSED,
    CLASSIFIER_LIVE_FLOOR_MUCH_BELOW_W7,
    CLASSIFIER_W7_IN_VIVO_CONFIRMED,
)

CPU_PHASE0_STRUCTURAL_FLOOR_WIDTH = 7
CPU_PHASE0_REACHABLE_PEAK = 33
W7_CLIP_ABS = signed_w_max(7)
MEANINGFUL_PARITY_PRESSURE_PEAK = 15
W7_IN_VIVO_CONFIRMED_MIN_PEAK = 33

EXPLICIT_NON_CLAIMS: tuple[str, ...] = (
    "dynamics_only_clip_boundary_not_byte_pack",
    "not_native_loop_injection_confirmation",
    "not_rules_promotion_unlock_in_this_slice",
    "not_inclusive_sub2",
    "not_gpu_launch_in_code_readiness_slice",
)

PREREG_PACKET_W7_BREAKS_PARITY_CITATION = (
    "artifacts/consensus_prep/w7_dense_acc_in_vivo_gpu_launch_packet_v3_draft.json:"
    "classifier_labels.W7_BREAKS_LIVE_PARITY"
)
PREREG_W7_BREAKS_PARITY_PREDICATES = (
    "crossing/applied-mask/q/final parity on clean sidecars"
)
STRUCTURAL_REASON_O1_MISSING_EVIDENCE = "o1_missing_evidence"
DIVERGENCE_ONSET_STATE_KEY = "model.H_level.core.layers.0.attn.gqkv_proj"
DIVERGENCE_CHARACTERIZATION_W7_READ_CLAMP_ASYMMETRY = (
    "dynamics_drift_from_w7_read_clamp_asymmetry"
)

W7_ACCUMULATOR_CLIP_CONTRACT_C: dict[str, Any] = {
    "contract_id": "C_hybrid_read_clamp_write_spec",
    "read_boundary": {
        "operation": "clip_to_w7_tensor per int16 lane",
        "bounds": "±63",
        "site": "vote_update_state via apply_trainer_boundary_narrow_carrier",
        "citations": [
            "narrow_accumulator_codec.py:474-486",
            "narrow_carrier_trainer_integration.py:266-267",
            "bounded_delta_learner.py:926-931",
        ],
    },
    "write_storage": {
        "operation": "vote_update (decay+vote).clamp(±127) into exact_accumulator_shadow",
        "bounds": "±127 vote_update spec",
        "no_w7_reclip_on_write": True,
        "citations": [
            "vote_update.py:968-970",
            "accumulator_real_dynamics_verdict.py:560-561",
            "bounded_delta_learner.py:1080,1101",
        ],
    },
    "o1_sidecar_observable": (
        "post-step exact_accumulator_shadow lanes in headroom_wiring_sidecar "
        "(not post-boundary W7-re-clipped storage)"
    ),
    "w7_floor_witness": "vote-derived live_reachable_peak_estimate (not raw sidecar max)",
    "packet_non_claim": "dynamics_only_clip_boundary_not_byte_pack",
}


@dataclass(frozen=True)
class ConfirmationEnvelope:
    envelope_id: str
    confirmation_vote_path: str
    rank_spec: RankVoteSpec
    vote_update_spec_factory: Any
    live_threshold_abs: int
    wired_max_vote_abs: int
    wired_reachable_peak_estimate: int

    def vote_update_spec(self, *, max_abs_per_tensor: int) -> VoteUpdateSpec:
        spec = self.vote_update_spec_factory()
        return VoteUpdateSpec(
            threshold_abs=int(spec.threshold_abs),
            accumulator_clip_min=int(spec.accumulator_clip_min),
            accumulator_clip_max=int(spec.accumulator_clip_max),
            decay_numerator=int(spec.decay_numerator),
            decay_denominator=int(spec.decay_denominator),
            max_abs_per_tensor=int(max_abs_per_tensor),
            fraction_per_tensor=float(spec.fraction_per_tensor),
        )

    def receipt_fields(self) -> dict[str, Any]:
        canonical_bins = canonical_rank_vote_spec(self.rank_spec)
        return {
            "envelope_id": self.envelope_id,
            "confirmation_vote_path": self.confirmation_vote_path,
            "live_threshold_abs": int(self.live_threshold_abs),
            "wired_max_vote_abs": int(self.wired_max_vote_abs),
            "live_reachable_peak_estimate": int(self.wired_reachable_peak_estimate),
            "rank_vote_spec_sha256": canonical_rank_bin_spec_sha256(canonical_bins),
            "native_loop_injection_confirmation": NATIVE_LOOP_INJECTION_CONFIRMATION_PENDING,
            "cpu_phase0_structural_floor_width": int(CPU_PHASE0_STRUCTURAL_FLOOR_WIDTH),
            "cpu_phase0_reachable_peak": int(CPU_PHASE0_REACHABLE_PEAK),
        }


def resolve_confirmation_envelope(envelope_id: str | None) -> ConfirmationEnvelope | None:
    if envelope_id is None or str(envelope_id).strip() == "":
        return None
    normalized = str(envelope_id).strip()
    if normalized != CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24:
        raise ValueError(
            f"unsupported confirmation_envelope {envelope_id!r}; "
            f"expected {CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24!r}"
        )
    rank_spec = canonical_acquisition_rank_vote_spec()
    threshold = int(canonical_acquisition_vote_update_spec().threshold_abs)
    return ConfirmationEnvelope(
        envelope_id=normalized,
        confirmation_vote_path=CONFIRMATION_VOTE_PATH_RANK_BUCKETED,
        rank_spec=rank_spec,
        vote_update_spec_factory=canonical_acquisition_vote_update_spec,
        live_threshold_abs=threshold,
        wired_max_vote_abs=max_vote_abs_for_rank_spec(rank_spec),
        wired_reachable_peak_estimate=canonical_acquisition_peak_reachable(
            threshold_abs=threshold
        ),
    )


def clip_abs_for_width_bits(width_bits: int) -> int:
    clip_min, clip_max = effective_clip_bounds(
        int(width_bits),
        -127,
        127,
    )
    return int(max(abs(int(clip_min)), abs(int(clip_max))))


def clip_table_for_peak(*, reachable_peak: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for width_bits in (4, 5, 6, 7, 16):
        clip_abs = clip_abs_for_width_bits(width_bits)
        rows[f"W{width_bits}"] = {
            "width_bits": int(width_bits),
            "clip_abs": int(clip_abs),
            "clips_reachable_peak": int(clip_abs) < int(reachable_peak),
        }
    return rows


@dataclass(frozen=True)
class ObservedVotePressure:
    max_vote_abs: int
    observed_step_count: int
    source: str


def _vote_abs_max_from_pressure_entry(entry: Mapping[str, Any]) -> int | None:
    if "vote_abs_max" in entry:
        return int(entry["vote_abs_max"])
    for key in ("pressure_shape_summary", "vote_abs_summary"):
        summary = entry.get(key)
        if isinstance(summary, Mapping) and "vote_abs_max" in summary:
            return int(summary["vote_abs_max"])
    return None


def observed_vote_pressure_from_receipt(receipt: Mapping[str, Any]) -> ObservedVotePressure:
    """Return per-step vote-pressure evidence only — never configured rank bins."""

    observed = 0
    steps_with_pressure = 0
    step_reports = receipt.get("step_reports") or {}
    if not isinstance(step_reports, Mapping):
        return ObservedVotePressure(0, 0, "none")
    for report in step_reports.values():
        if not isinstance(report, Mapping):
            continue
        vote_pressure = report.get("vote_pressure") or {}
        if not isinstance(vote_pressure, Mapping) or not vote_pressure:
            continue
        step_max = 0
        step_has = False
        for entry in vote_pressure.values():
            if not isinstance(entry, Mapping):
                continue
            candidate = _vote_abs_max_from_pressure_entry(entry)
            if candidate is None:
                continue
            step_has = True
            step_max = max(step_max, int(candidate))
        if step_has:
            steps_with_pressure += 1
            observed = max(observed, step_max)
    source = "step_reports.vote_pressure" if steps_with_pressure > 0 else "none"
    return ObservedVotePressure(
        max_vote_abs=int(observed),
        observed_step_count=int(steps_with_pressure),
        source=source,
    )


def verify_dual_arm_w7_configuration(
    *,
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
) -> tuple[int | None, list[str]]:
    """Derive confirmed floor width only when arm flags match the dual-arm contract."""

    failures: list[str] = []
    if bool(oracle_receipt.get("dense_accumulator_w7_clip")):
        failures.append("oracle_dense_accumulator_w7_clip_must_be_false")
    if not bool(treatment_receipt.get("dense_accumulator_w7_clip")):
        failures.append("treatment_dense_accumulator_w7_clip_must_be_true")
    for label, receipt in (("oracle", oracle_receipt), ("treatment", treatment_receipt)):
        if bool(receipt.get("persistent_accumulator_w5_byte_packed")):
            failures.append(f"{label}_persistent_accumulator_w5_active")
        if bool(receipt.get("persistent_accumulator_w6_byte_packed")):
            failures.append(f"{label}_persistent_accumulator_w6_byte_packed_active")
        if bool(receipt.get("persistent_accumulator_event_coded_live")):
            failures.append(f"{label}_persistent_accumulator_event_coded_live_active")
    if failures:
        return None, failures
    return CPU_PHASE0_STRUCTURAL_FLOOR_WIDTH, []


def live_reachable_peak_estimate(*, threshold_abs: int, max_vote_abs_observed: int) -> int:
    return int(threshold_abs) - 1 + int(max_vote_abs_observed)


def _bit_equality_lane_witness(s3bb_stats: Mapping[str, Any]) -> dict[str, Any]:
    bit_equality = s3bb_stats.get("bit_equality_diagnostics") or {}
    total_lane_count = int(bit_equality.get("total_lane_count") or 0)
    matched_key_compared_lane_count = int(
        (bit_equality.get("sidecar_coverage_diagnostics") or {}).get(
            "matched_key_compared_lane_count"
        )
        or 0
    )
    if matched_key_compared_lane_count == 0:
        matched_key_compared_lane_count = total_lane_count
    equality_rate = float(bit_equality.get("vote_update_state_accumulator_equality_rate") or 0.0)
    vacuous = total_lane_count <= 0
    return {
        "total_lane_count": total_lane_count,
        "matched_key_compared_lane_count": matched_key_compared_lane_count,
        "vote_update_state_accumulator_equality_rate": equality_rate,
        "o1_lane_equality_vacuous": bool(vacuous),
        "o1_lane_equality_load_bearing": not vacuous,
    }


def extract_w7_parity_signals(s3bb_stats: Mapping[str, Any]) -> dict[str, Any]:
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
    """REQ-1: packet :381 allows BREAKS on enumerated O1-O4 predicates without lane-equality."""

    if bool(o1_lane_witness.get("o1_lane_equality_load_bearing")):
        return True
    return bool(parity_signals.get("parity_break_driving_keys"))


def characterize_accumulator_divergence(
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
    *,
    focus_state_key: str = DIVERGENCE_ONSET_STATE_KEY,
) -> dict[str, Any] | None:
    from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
        _index_sidecar_file,
        resolve_headroom_wiring_sidecar_path,
    )

    oracle_path = resolve_headroom_wiring_sidecar_path(oracle_receipt)
    treatment_path = resolve_headroom_wiring_sidecar_path(treatment_receipt)
    if oracle_path is None or treatment_path is None:
        return None
    if not Path(oracle_path).is_file() or not Path(treatment_path).is_file():
        return None

    oracle_keyed, _, _ = _index_sidecar_file(oracle_path)
    treatment_keyed, _, _ = _index_sidecar_file(treatment_path)
    shared_keys = sorted(
        key
        for key in set(oracle_keyed).intersection(treatment_keyed)
        if key[1] == focus_state_key
    )
    if not shared_keys:
        return None

    per_step: list[dict[str, Any]] = []
    onset_step: int | None = None
    for step_id, _state_key in shared_keys:
        oracle_lanes = [int(v) for v in oracle_keyed[(step_id, focus_state_key)]["accumulator_lanes"]]
        treatment_lanes = [
            int(v) for v in treatment_keyed[(step_id, focus_state_key)]["accumulator_lanes"]
        ]
        if len(oracle_lanes) != len(treatment_lanes):
            continue
        lane_count = len(oracle_lanes)
        diff_count = sum(1 for o_val, t_val in zip(oracle_lanes, treatment_lanes, strict=True) if o_val != t_val)
        row = {
            "step": int(step_id),
            "lane_count": int(lane_count),
            "diff_count": int(diff_count),
            "diff_fraction": float(diff_count) / float(lane_count) if lane_count else 0.0,
            "oracle_max_abs": int(max(abs(v) for v in oracle_lanes)) if oracle_lanes else 0,
            "treatment_max_abs": int(max(abs(v) for v in treatment_lanes)) if treatment_lanes else 0,
        }
        per_step.append(row)
        if onset_step is None and diff_count > 0:
            onset_step = int(step_id)

    if not per_step:
        return None

    per_step.sort(key=lambda row: int(row["step"]))
    terminal = per_step[-1]
    return {
        "focus_state_key": focus_state_key,
        "characterization": DIVERGENCE_CHARACTERIZATION_W7_READ_CLAMP_ASYMMETRY,
        "onset_step": onset_step,
        "per_step_series": per_step,
        "terminal_step": int(terminal["step"]),
        "terminal_diff_count": int(terminal["diff_count"]),
        "terminal_diff_fraction": float(terminal["diff_fraction"]),
        "terminal_oracle_max_abs": int(terminal["oracle_max_abs"]),
        "terminal_treatment_max_abs": int(terminal["treatment_max_abs"]),
        "treatment_plateau_max_abs": int(
            max(int(row["treatment_max_abs"]) for row in per_step)
        ),
    }


def derive_w7_parity_inputs(
    s3bb_primary: str,
    s3bb_stats: Mapping[str, Any],
    sidecar_coverage: Mapping[str, Any],
    *,
    oracle_receipt: Mapping[str, Any] | None = None,
    treatment_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Map S3BB decision-parity stats to W7 structural/parity bridge inputs."""

    coverage = dict(sidecar_coverage)
    base: dict[str, Any] = {
        "sidecar_coverage_diagnostics": coverage,
        "s3bb_primary_classifier": str(s3bb_primary),
        "w7_accumulator_clip_contract": dict(W7_ACCUMULATOR_CLIP_CONTRACT_C),
        "prereg_w7_breaks_parity_citation": PREREG_PACKET_W7_BREAKS_PARITY_CITATION,
        "prereg_w7_breaks_parity_predicates": PREREG_W7_BREAKS_PARITY_PREDICATES,
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
            "divergence_characterization": None,
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
            "divergence_characterization": None,
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
            "divergence_characterization": None,
        }

    o1_lane_witness = _bit_equality_lane_witness(s3bb_stats)
    parity_signals = extract_w7_parity_signals(s3bb_stats)
    driving_keys = list(parity_signals["parity_break_driving_keys"])
    if primary in S3BB_SCIENCE_PARITY_BREAK_PRIMARIES:
        driving_keys.insert(0, f"s3bb_primary:{primary}")

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
            "divergence_characterization": None,
            **o1_lane_witness,
            **parity_signals,
        }

    divergence_characterization = None
    if oracle_receipt is not None and treatment_receipt is not None:
        divergence_characterization = characterize_accumulator_divergence(
            oracle_receipt,
            treatment_receipt,
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
        "divergence_characterization": divergence_characterization,
        **o1_lane_witness,
        **parity_signals,
    }


def classify_w7_in_vivo_dual_arm(
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
        primary = CLASSIFIER_RUN_HEALTH_FAIL
        return _classifier_payload(
            primary=primary,
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
        primary = CLASSIFIER_OBSERVER_TOO_EXPENSIVE
        return _classifier_payload(
            primary=primary,
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
    if peak_obs > W7_CLIP_ABS:
        primary = CLASSIFIER_W7_FLOOR_RISES_WITH_LIVE_VOTE_MAX
    elif parity_break:
        primary = CLASSIFIER_W7_BREAKS_LIVE_PARITY
    elif peak_obs < MEANINGFUL_PARITY_PRESSURE_PEAK:
        primary = CLASSIFIER_ENVELOPE_UNDER_PRESSED
    elif (
        peak_obs >= W7_IN_VIVO_CONFIRMED_MIN_PEAK
        and confirmed_vote_acc_floor_width == CPU_PHASE0_STRUCTURAL_FLOOR_WIDTH
        and not parity_break
    ):
        primary = CLASSIFIER_W7_IN_VIVO_CONFIRMED
    elif peak_obs < W7_IN_VIVO_CONFIRMED_MIN_PEAK:
        primary = CLASSIFIER_LIVE_FLOOR_MUCH_BELOW_W7
    else:
        primary = CLASSIFIER_W7_BREAKS_LIVE_PARITY
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
    rules_promotion_unlock_predicate = (
        primary == CLASSIFIER_W7_IN_VIVO_CONFIRMED
        and confirmed_vote_acc_floor_width == CPU_PHASE0_STRUCTURAL_FLOOR_WIDTH
        and live_max_vote_abs_source == "step_reports.vote_pressure"
        and live_max_vote_abs_observed is not None
        and live_reachable_peak_estimate is not None
        and int(live_reachable_peak_estimate) >= W7_IN_VIVO_CONFIRMED_MIN_PEAK
    )
    return {
        "schema_version": "hrm_text_158_w7_dense_acc_in_vivo_classifier/v1",
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
        "rules_promotion_unlock_predicate": bool(rules_promotion_unlock_predicate),
        "parity_break": bool(parity_break),
        "structural_fail": bool(structural_fail),
        "structural_reason": structural_reason,
        "harness_failures": list(failures),
        "dry_run_peak_reference": dry_run_rank_vote_peak_reachable(threshold_abs=1),
        "wired_envelope_peak_reference": (
            envelope.wired_reachable_peak_estimate if envelope is not None else None
        ),
    }


def live_reachable_peak_estimate_from_observed(
    *,
    threshold_abs: int,
    max_vote_abs_observed: int,
) -> int:
    return live_reachable_peak_estimate(
        threshold_abs=int(threshold_abs),
        max_vote_abs_observed=int(max_vote_abs_observed),
    )
