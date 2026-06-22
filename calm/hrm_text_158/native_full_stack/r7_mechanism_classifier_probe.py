"""Read-only R7 mechanism classifier over compact cap/defer instrumentation sidecars."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.r7_cap_defer_pressure_instrumentation import (
    R7_STEP_CHUNK_SCHEMA_VERSION,
    iter_sidecar_chunks,
    validate_accounting_invariant,
)

R7_PROBE_SCHEMA_VERSION = "hrm_text_158_r7_mechanism_classifier_probe/v1"

DEFERRED_AGE_MIN = 1
DEFER_SATURATION_MIN = 0.10
Q_TRANSITION_MASS_RATIO_MAX = 0.05
PRESSURE_GROWTH_MIN = 1.5
MIN_MEASURED_STEPS = 8

BRANCH_HARNESS_FAIL = "R7_HARNESS_FAIL"
BRANCH_SCHEMA_FAIL = "R7_SCHEMA_FAIL"
BRANCH_ARTIFACT_INSUFFICIENT = "R7_ARTIFACT_INSUFFICIENT"
BRANCH_CAP_DEFER = "R7_CAP_DEFER_BINDING"
BRANCH_VOTE_AMPLITUDE = "R7_VOTE_AMPLITUDE_DECAY_BINDING"
BRANCH_MIXED = "R7_MIXED_OR_INCONCLUSIVE"
BRANCH_NO_PRESSURE_GROWTH = "R7_NO_PRESSURE_GROWTH"

EXPLICIT_NON_CLAIMS: tuple[str, ...] = (
    "proxy_not_proof",
    "no_backlog_age_proof_without_carry",
    "no_mechanism_reducible",
    "no_stability_verdict",
    "no_sub2_win",
    "no_readiness_flip",
    "no_trainer_or_gpu",
    "no_decision_surface_claim",
    "diagnostic_10_step_not_mechanism_proof",
    "prior_deferred_accepted_not_excluded_from_candidate_count",
)

NEXT_ACTION_BY_BRANCH: dict[str, str] = {
    BRANCH_CAP_DEFER: "global_cap_relaxation_diagnostic_run",
    BRANCH_VOTE_AMPLITUDE: "lower_amplitude_decay_sign_compressed_mechanism_design",
    BRANCH_MIXED: "stop_and_redesign_prereg",
    BRANCH_ARTIFACT_INSUFFICIENT: "instrumentation_not_interpretation",
    BRANCH_HARNESS_FAIL: "stop_and_fix_inputs",
    BRANCH_SCHEMA_FAIL: "stop_and_fix_inputs",
    BRANCH_NO_PRESSURE_GROWTH: "instrumentation_not_interpretation",
}


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def compute_run_metrics(chunks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(chunks, key=lambda chunk: int(chunk["step"]))
    if not ordered:
        return {
            "steps_observed": 0,
            "pressure_mass_first": 0,
            "pressure_mass_last": 0,
            "pressure_growth_ratio": 0.0,
            "run_max_deferred_backlog_max_age_steps": 0,
            "run_mean_deferred_saturation": 0.0,
            "q_transition_mass_ratio": 0.0,
            "accepted_from_prior_deferred_total": 0,
            "accepted_fresh_total": 0,
            "per_step": [],
        }
    pressure_first = int(ordered[0].get("pressure_mass", 0))
    pressure_last = int(ordered[-1].get("pressure_mass", 0))
    deferred_saturation_values: list[float] = []
    age_values: list[int] = []
    per_step: list[dict[str, Any]] = []
    q_transition_total = 0
    pressure_lane_steps_total = 0
    for index, chunk in enumerate(ordered):
        candidate = max(int(chunk.get("candidate_count", 0)), 1)
        deferred = int(chunk.get("deferred_count", 0))
        deferred_saturation_values.append(_safe_ratio(deferred, candidate))
        age_values.append(int(chunk.get("deferred_backlog_max_age_steps", 0)))
        per_step.append(
            {
                "step": int(chunk["step"]),
                "candidate_count": int(chunk.get("candidate_count", 0)),
                "accepted_count": int(chunk.get("accepted_count", 0)),
                "deferred_count": deferred,
                "accepted_from_prior_deferred_count": int(
                    chunk.get("accepted_from_prior_deferred_count", 0)
                ),
                "accepted_fresh_count": int(chunk.get("accepted_fresh_count", 0)),
                "deferred_backlog_max_age_steps": int(
                    chunk.get("deferred_backlog_max_age_steps", 0)
                ),
                "pressure_mass": int(chunk.get("pressure_mass", 0)),
            }
        )
        if index > 0:
            prev = ordered[index - 1]
            q_transition_total += int(chunk.get("q_apply_count", 0))
            pressure_lane_steps_total += int(prev.get("pressure_mass", 0))
    return {
        "steps_observed": len(ordered),
        "pressure_mass_first": pressure_first,
        "pressure_mass_last": pressure_last,
        "pressure_growth_ratio": _safe_ratio(pressure_last, max(pressure_first, 1)),
        "run_max_deferred_backlog_max_age_steps": max(age_values) if age_values else 0,
        "run_mean_deferred_saturation": (
            sum(deferred_saturation_values) / len(deferred_saturation_values)
            if deferred_saturation_values
            else 0.0
        ),
        "q_transition_mass_ratio": _safe_ratio(
            q_transition_total,
            pressure_lane_steps_total,
        ),
        "accepted_from_prior_deferred_total": sum(
            int(chunk.get("accepted_from_prior_deferred_count", 0)) for chunk in ordered
        ),
        "accepted_fresh_total": sum(int(chunk.get("accepted_fresh_count", 0)) for chunk in ordered),
        "per_step": per_step,
        "thresholds": {
            "DEFERRED_AGE_MIN": DEFERRED_AGE_MIN,
            "DEFER_SATURATION_MIN": DEFER_SATURATION_MIN,
            "Q_TRANSITION_MASS_RATIO_MAX": Q_TRANSITION_MASS_RATIO_MAX,
            "PRESSURE_GROWTH_MIN": PRESSURE_GROWTH_MIN,
        },
    }


def select_branch(
    *,
    harness_fail: bool,
    schema_fail: bool,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if harness_fail:
        return {"branch": BRANCH_HARNESS_FAIL, "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_HARNESS_FAIL]}
    if schema_fail:
        return {"branch": BRANCH_SCHEMA_FAIL, "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_SCHEMA_FAIL]}
    if int(metrics["steps_observed"]) < MIN_MEASURED_STEPS:
        return {
            "branch": BRANCH_ARTIFACT_INSUFFICIENT,
            "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_ARTIFACT_INSUFFICIENT],
            "reason": "fewer_than_eight_measured_steps",
        }
    pressure_growth = float(metrics["pressure_growth_ratio"])
    if pressure_growth < PRESSURE_GROWTH_MIN:
        return {
            "branch": BRANCH_NO_PRESSURE_GROWTH,
            "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_NO_PRESSURE_GROWTH],
            "pressure_growth_ratio": pressure_growth,
        }
    run_max_age = int(metrics["run_max_deferred_backlog_max_age_steps"])
    run_mean_deferred = float(metrics["run_mean_deferred_saturation"])
    q_ratio = float(metrics["q_transition_mass_ratio"])
    cap_defer = (
        run_max_age >= DEFERRED_AGE_MIN
        and run_mean_deferred >= DEFER_SATURATION_MIN
        and pressure_growth >= PRESSURE_GROWTH_MIN
    )
    vote_amplitude = (
        run_max_age == 0
        and run_mean_deferred == 0.0
        and q_ratio <= Q_TRANSITION_MASS_RATIO_MAX
        and pressure_growth >= PRESSURE_GROWTH_MIN
    )
    if cap_defer:
        return {
            "branch": BRANCH_CAP_DEFER,
            "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_CAP_DEFER],
            "run_max_deferred_backlog_max_age_steps": run_max_age,
            "run_mean_deferred_saturation": run_mean_deferred,
            "pressure_growth_ratio": pressure_growth,
        }
    if vote_amplitude:
        return {
            "branch": BRANCH_VOTE_AMPLITUDE,
            "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_VOTE_AMPLITUDE],
            "q_transition_mass_ratio": q_ratio,
            "pressure_growth_ratio": pressure_growth,
        }
    return {
        "branch": BRANCH_MIXED,
        "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_MIXED],
        "pressure_growth_ratio": pressure_growth,
        "run_max_deferred_backlog_max_age_steps": run_max_age,
        "run_mean_deferred_saturation": run_mean_deferred,
        "q_transition_mass_ratio": q_ratio,
    }


def build_classifier_from_chunks(
    *,
    chunks: Sequence[Mapping[str, Any]],
    harness_fail: bool = False,
    run_root: str | None = None,
    head_sha256: str | None = None,
    sidecar_path: str | None = None,
) -> dict[str, Any]:
    schema_failures: list[str] = []
    for chunk in chunks:
        if chunk.get("schema_version") != R7_STEP_CHUNK_SCHEMA_VERSION:
            schema_failures.append(f"step{chunk.get('step')}:schema_version")
            continue
        failures = list(chunk.get("accounting_invariant_failures") or [])
        if failures:
            schema_failures.extend(f"step{chunk.get('step')}:{item}" for item in failures)
    metrics = compute_run_metrics(chunks) if chunks else compute_run_metrics([])
    branch = select_branch(
        harness_fail=harness_fail,
        schema_fail=bool(schema_failures),
        metrics=metrics,
    )
    result: dict[str, Any] = {
        "schema_version": R7_PROBE_SCHEMA_VERSION,
        "raw_arrays_included": False,
        "run_root": run_root,
        "head_sha256": head_sha256,
        "sidecar_path": sidecar_path,
        "run_metrics": metrics,
        "branch_selection": branch,
        "explicit_non_claims": list(EXPLICIT_NON_CLAIMS),
    }
    if schema_failures:
        result["schema_failures"] = schema_failures
    return result


def build_classifier_probe_receipt(
    *,
    run_root: Path,
    head_sha256: str,
    sidecar_name: str = "r7_cap_defer_pressure_sidecar.jsonl",
) -> dict[str, Any]:
    sidecar_path = run_root / sidecar_name
    if not sidecar_path.is_file():
        return build_classifier_from_chunks(
            chunks=[],
            harness_fail=True,
            run_root=str(run_root),
            head_sha256=head_sha256,
            sidecar_path=str(sidecar_path),
        )
    chunks = iter_sidecar_chunks(sidecar_path)
    return build_classifier_from_chunks(
        chunks=chunks,
        harness_fail=False,
        run_root=str(run_root),
        head_sha256=head_sha256,
        sidecar_path=str(sidecar_path),
    )


def validate_step_summary_for_field_presence(step_summary: Mapping[str, Any]) -> list[str]:
    return validate_accounting_invariant(step_summary)
