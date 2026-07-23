"""Receipt assembly only for pressure/metric diagnostic (PLAN_v6 rev4).

Owns: build_diagnostic_receipt (allowlist + seam_map + classifier invoke).
Readiness lives in pressure_metric_readiness.py.
Dependency: receipt → readiness + classifier + telemetry.
Bound by PLAN_v6 sha 346b67d8…; rev4 re-scope 1784829182373.
"""
from __future__ import annotations

from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.pressure_metric_classifier import (
    classify_pressure_metric_family,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_readiness import (
    evaluate_readiness,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_telemetry import (
    AUTHORITY_DISPATCH,
    CROSSING_THRESHOLD_ABS,
    PARENT_SHA256,
    PLAN_SHA256,
    optional_json_float,
    sanitize_receipt_for_strict_json,
    summarize_demand_totals,
)

RECEIPT_MEASUREMENT_ALLOWLIST = (
    "n_flips",
    "q_changed_count",
    "credited_mass",
    "lifetime_censored_frac",
    "p50_flip_lifetime",
    "H_bits_per_weight",
    "H_trajectory",
    "n_applied_drains",
    "margin_trajectory",
    "episode_trajectory",
)


def build_diagnostic_receipt(
    *,
    store: Any,
    measurements: Mapping[str, Any],
    probes: Mapping[str, Any] | None = None,
    route_counters: Mapping[str, Any] | None = None,
    banked_sha: Mapping[str, Any] | None = None,
    frozen_scale_sha: Mapping[str, Any] | None = None,
    paired_determinism_cost_ok: bool | None = None,
    paired_proof: Mapping[str, Any] | None = None,
    expected_parent_sha: str | None = None,
    schema_only: bool = False,
    steps: int | None = None,
    require_probes: bool = True,
) -> dict[str, Any]:
    """Compact receipt — aggregates only; never identity tensors."""
    surv = store.survival_summary()
    demand = summarize_demand_totals(store.per_step_ratios)
    H_traj = list(measurements.get("H_trajectory") or [])
    H_final = optional_json_float(measurements.get("H_bits_per_weight"))
    H_step25 = None
    for row in H_traj:
        if int(row.get("step", -1)) == 25:
            H_step25 = optional_json_float(row.get("H_bits_per_weight"))
            break

    probes = probes or {}
    retention_ok_flag = probes.get("retention_ok")
    if retention_ok_flag is None and probes.get("ret_final_count") is not None:
        from calm.hrm_text_158.native_full_stack.family_classifier import retention_ok

        retention_ok_flag = retention_ok(
            final_count=int(probes["ret_final_count"]),
            step0_count=int(probes["ret_step0_count"]),
        )

    margin_trajectory = list(measurements.get("margin_trajectory") or [])
    episode_trajectory = list(measurements.get("episode_trajectory") or [])

    every = []
    for row in store.per_step_ratios:
        if int(row["step"]) % 25 == 0 or int(row["step"]) == int(store.steps):
            every.append(dict(row))

    required_telemetry = {
        "demand": demand,
        "deferred_survival": surv,
        "margin_trajectory": margin_trajectory,
        "episode_trajectory": episode_trajectory,
        "demand_per_25": every,
        "H_trajectory": H_traj,
    }

    readiness = evaluate_readiness(
        expected_parent_sha=expected_parent_sha or PARENT_SHA256,
        banked_sha=banked_sha,
        frozen_scale_sha=frozen_scale_sha,
        route_counters=route_counters,
        paired_proof=paired_proof,
        paired_determinism_cost_ok=paired_determinism_cost_ok,
        H_step25=H_step25,
        required_telemetry=required_telemetry,
        schema_only=schema_only,
        steps=int(steps if steps is not None else store.steps),
        probes=probes,
        require_probes=bool(require_probes) and not schema_only,
    )

    telemetry_ok = bool(store.per_step_ratios) or schema_only
    verdict = classify_pressure_metric_family(
        telemetry_ok=telemetry_ok and not schema_only,
        two_tier_threshold_assert_pass=bool(store.two_tier_threshold_assert_pass),
        paired_determinism_cost_ok=paired_determinism_cost_ok,
        N_events_evaluable=int(surv["N_events_evaluable"]),
        mean_ratio=demand["mean_ratio"],
        frac_steps_ratio_ge_2=demand["frac_steps_ratio_ge_2"],
        deferred_survival_class=str(surv["deferred_survival_class"]),
        delta_never_apply=surv["delta_never_apply"],
        N_events_evaluable_early=int(surv["N_events_evaluable_early"]),
        N_events_evaluable_late=int(surv["N_events_evaluable_late"]),
        deferred_never_apply_within_H_frac=surv["deferred_never_apply_within_H_frac"],
        lcf=optional_json_float(measurements.get("lifetime_censored_frac")),
        H_final=H_final,
        H_step25=H_step25,
        retention_ok_flag=bool(retention_ok_flag) if retention_ok_flag is not None else False,
        readiness_ok=bool(readiness["ok"]) if not schema_only else True,
        readiness_stop_reason=readiness.get("stop_reason"),
    )

    meas_out = {k: measurements.get(k) for k in RECEIPT_MEASUREMENT_ALLOWLIST}
    meas_out["demand"] = demand
    meas_out["deferred_survival"] = surv
    meas_out["demand_per_25"] = every
    meas_out["margin_trajectory"] = margin_trajectory
    meas_out["episode_trajectory"] = episode_trajectory

    receipt = {
        "screen": "censor_null_pressure_metric_diagnostic/v1",
        "plan_sha256": PLAN_SHA256,
        "authority_dispatch": AUTHORITY_DISPATCH,
        "schema_only": bool(schema_only),
        "two_tier_threshold_assert_pass": bool(store.two_tier_threshold_assert_pass),
        "crossing_threshold_abs": int(CROSSING_THRESHOLD_ABS),
        "measurements": meas_out,
        "probes": dict(probes) if probes else None,
        "route_counters": dict(route_counters) if route_counters else None,
        "banked_sha": dict(banked_sha) if banked_sha else None,
        "frozen_scale_sha": dict(frozen_scale_sha) if frozen_scale_sha else None,
        "paired_proof": dict(paired_proof) if paired_proof else None,
        "readiness": readiness,
        "classifier": verdict,
        "family": verdict["family"],
        "stop_reason": verdict["stop_reason"],
        "seam_map": {
            "selection_margins": "pressure_metric_telemetry",
            "lifecycle_store": "pressure_metric_lifecycle",
            "classifier": "pressure_metric_classifier",
            "readiness": "pressure_metric_readiness",
            "receipt_assembly": "pressure_metric_receipt",
            "warmup_runtime": "pressure_metric_warmup_runtime",
            "proof": "pressure_metric_proof",
            "benchmark": "pressure_metric_benchmark",
            "dependency_direction": (
                "telemetry→∅; lifecycle→two_tier; classifier→telemetry; "
                "readiness→telemetry; receipt→readiness+classifier; "
                "warmup_runtime→loop+model_runtime+lifecycle+telemetry; "
                "proof→warmup+receipt; benchmark→warmup+proof; "
                "model_runtime→fixed_qscale (assert/rebind); CLI→all thin"
            ),
        },
        "explicit_non_claims": [
            "observability_only — no forgetting law",
            "no representation_limit verdict from this screen",
            "no sub-2 achievability claim",
            "identity trackers internal-only (not emitted)",
        ],
    }
    return sanitize_receipt_for_strict_json(receipt)
