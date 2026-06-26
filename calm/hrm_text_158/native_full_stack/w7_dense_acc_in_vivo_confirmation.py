"""W7 dense-accumulator in-vivo confirmation envelope + postrun classifier helpers."""
from __future__ import annotations

from dataclasses import dataclass
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


def classify_w7_in_vivo_dual_arm(
    *,
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
    envelope: ConfirmationEnvelope | None,
    observer_too_expensive: bool = False,
    harness_failures: Sequence[str] = (),
    parity_break: bool = False,
    confirmed_vote_acc_floor_width: int | None = None,
) -> dict[str, Any]:
    failures = list(harness_failures)
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
