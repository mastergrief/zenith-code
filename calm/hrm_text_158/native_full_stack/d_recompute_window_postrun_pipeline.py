"""Postrun pipeline chaining horizon growth, acc sizing, and in-vivo validation."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.d_recompute_window_acc_sizing import (
    SIZING_VERDICT_INCONCLUSIVE,
    SIZING_VERDICT_RECOMMENDED_LAW,
    SIZING_VERDICT_SIZED_NOT_SUB2,
    VERDICT_SCOPE_ENVELOPE_MODEL_ONLY,
    quantile_size_acc_bpw_from_horizon_growth,
    size_acc_bpw_from_horizon_growth,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_horizon_analyzer import (
    analyze_horizon_k_star_growth,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_in_vivo_bound_validator import (
    IN_VIVO_DOMINANCE_PROVEN,
    VERDICT_SCOPE_IN_VIVO_VALIDATED,
    validate_in_vivo_acc_bound,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_stratified_selector import (
    StratifiedSelectorManifest,
)

POSTRUN_PIPELINE_SCHEMA_VERSION = "hrm_text_158_d_recompute_postrun_pipeline/v0"


def _compact_horizon_summary(horizon_growth: Mapping[str, Any]) -> dict[str, Any]:
    summaries = dict(horizon_growth.get("summaries_by_h") or {})
    compact_summaries: dict[str, Any] = {}
    for horizon_h, summary in summaries.items():
        if not isinstance(summary, Mapping):
            continue
        compact_summaries[str(horizon_h)] = {
            key: summary.get(key)
            for key in (
                "k95_weighted",
                "k99_weighted",
                "kworst_weighted",
                "right_censor_rate",
                "parity_fail_count",
                "gapped_lane_count",
            )
        }
    return {
        "growth_branch": horizon_growth.get("growth_branch"),
        "coverage_tier": horizon_growth.get("coverage_tier"),
        "stress_tail_policy": horizon_growth.get("stress_tail_policy"),
        "summaries_by_h": compact_summaries,
    }


def _compact_acc_sizing_summary(acc_sizing: Mapping[str, Any]) -> dict[str, Any]:
    best_row = acc_sizing.get("best_grid_row")
    compact_best = None
    if isinstance(best_row, Mapping):
        compact_best = {
            key: best_row.get(key)
            for key in (
                "decay_num",
                "decay_den",
                "window_k",
                "inclusive_acc_bpw",
            )
        }
    return {
        "sizing_verdict": acc_sizing.get("sizing_verdict"),
        "reason": acc_sizing.get("reason"),
        "verdict_scope": acc_sizing.get("verdict_scope"),
        "window_k": acc_sizing.get("window_k"),
        "effective_acc_budget_bpw": acc_sizing.get("effective_acc_budget_bpw"),
        "strict_less_than_budget": acc_sizing.get("strict_less_than_budget"),
        "best_grid_row": compact_best,
    }


def _compact_quantile_acc_sizing_summary(
    quantile_acc_sizing: Mapping[str, Any],
) -> dict[str, Any]:
    best_row = quantile_acc_sizing.get("best_grid_row")
    compact_best = None
    if isinstance(best_row, Mapping):
        compact_best = {
            key: best_row.get(key)
            for key in (
                "decay_num",
                "decay_den",
                "window_k",
                "inclusive_acc_bpw",
            )
        }
    return {
        key: quantile_acc_sizing.get(key)
        for key in (
            "claim_scope",
            "quantile",
            "quantile_k",
            "quantile_window_k",
            "quantile_uncensored",
            "censored_weight_fraction",
            "censor_mass_max",
            "tail_policy",
            "not_worst_case_bound",
            "growth_branch",
            "parity_fail_count",
            "gapped_lane_count",
            "eligible_lane_count",
            "coverage_tier",
            "selector_log_key_aligned",
            "effective_acc_budget_bpw",
            "strict_less_than_budget",
            "quantile_sizing_verdict",
            "quantile_sub2_candidate",
            "reason",
        )
    } | {"best_grid_row": compact_best}


def merge_postrun_verdict(
    *,
    horizon_growth: Mapping[str, Any],
    acc_sizing: Mapping[str, Any],
    in_vivo_validation: Mapping[str, Any],
) -> dict[str, Any]:
    envelope_verdict = str(acc_sizing.get("sizing_verdict") or "")
    in_vivo_scope = str(in_vivo_validation.get("verdict_scope") or "")
    strict_pass = bool(acc_sizing.get("strict_less_than_budget"))

    final_sizing_verdict = envelope_verdict
    final_verdict_scope = str(
        in_vivo_validation.get("verdict_scope")
        or acc_sizing.get("verdict_scope")
        or VERDICT_SCOPE_ENVELOPE_MODEL_ONLY
    )
    downgrade_reason = None

    if (
        envelope_verdict == SIZING_VERDICT_RECOMMENDED_LAW
        and in_vivo_scope == VERDICT_SCOPE_IN_VIVO_VALIDATED
        and strict_pass
        and in_vivo_validation.get("in_vivo_verdict") == IN_VIVO_DOMINANCE_PROVEN
    ):
        final_sizing_verdict = SIZING_VERDICT_RECOMMENDED_LAW
        final_verdict_scope = VERDICT_SCOPE_IN_VIVO_VALIDATED
    elif envelope_verdict == SIZING_VERDICT_RECOMMENDED_LAW:
        final_sizing_verdict = SIZING_VERDICT_SIZED_NOT_SUB2
        downgrade_reason = "envelope_recommended_law_without_in_vivo_validation"
        final_verdict_scope = VERDICT_SCOPE_ENVELOPE_MODEL_ONLY
    elif envelope_verdict == SIZING_VERDICT_SIZED_NOT_SUB2:
        final_sizing_verdict = SIZING_VERDICT_SIZED_NOT_SUB2
    else:
        final_sizing_verdict = SIZING_VERDICT_INCONCLUSIVE
        if downgrade_reason is None:
            downgrade_reason = str(acc_sizing.get("reason") or in_vivo_validation.get("reason"))

    return {
        "final_sizing_verdict": final_sizing_verdict,
        "final_verdict_scope": final_verdict_scope,
        "recommended_law_eligible": bool(
            final_sizing_verdict == SIZING_VERDICT_RECOMMENDED_LAW
            and final_verdict_scope == VERDICT_SCOPE_IN_VIVO_VALIDATED
        ),
        "downgrade_reason": downgrade_reason,
    }


def run_postrun_arc2b_analysis(
    records: Sequence[Mapping[str, Any]],
    *,
    manifest: StratifiedSelectorManifest,
    numel_for_bpw: int,
    measured_q_scale_bpw: float | None = None,
    sizing_horizon_h: int = 100,
    measurement_start_step: int = 1,
) -> dict[str, Any]:
    horizon_growth = analyze_horizon_k_star_growth(
        records,
        stratum_weights=manifest.stratum_weights,
        stress_tail_policy=str(manifest.manifest_spec.get("stress_tail_policy") or ""),
        coverage_tier=str(manifest.coverage_tier),
        measurement_start_step=int(measurement_start_step),
    )
    acc_sizing = size_acc_bpw_from_horizon_growth(
        horizon_growth,
        numel_for_bpw=int(numel_for_bpw),
        measured_q_scale_bpw=measured_q_scale_bpw,
        sizing_horizon_h=int(sizing_horizon_h),
        measurement_start_step=int(measurement_start_step),
    )
    in_vivo_validation = validate_in_vivo_acc_bound(
        records,
        manifest=manifest,
        envelope_sizing=acc_sizing,
        sizing_horizon_h=int(sizing_horizon_h),
        measurement_start_step=int(measurement_start_step),
        numel_for_bpw=int(numel_for_bpw),
    )
    logged_keys = sorted({str(record["state_key"]) for record in records})
    manifest_keys = sorted(manifest.entry_by_key().keys())
    selector_log_key_aligned = set(logged_keys) == set(manifest_keys)
    quantile_acc_sizing = quantile_size_acc_bpw_from_horizon_growth(
        horizon_growth,
        numel_for_bpw=int(numel_for_bpw),
        measured_q_scale_bpw=measured_q_scale_bpw,
        sizing_horizon_h=int(sizing_horizon_h),
        selector_log_key_aligned=bool(selector_log_key_aligned),
        stratum_weights=manifest.stratum_weights,
    )
    merged = merge_postrun_verdict(
        horizon_growth=horizon_growth,
        acc_sizing=acc_sizing,
        in_vivo_validation=in_vivo_validation,
    )
    return {
        "schema_version": POSTRUN_PIPELINE_SCHEMA_VERSION,
        "horizon_growth_summary": _compact_horizon_summary(horizon_growth),
        "acc_sizing_summary": _compact_acc_sizing_summary(acc_sizing),
        "in_vivo_validation": {
            key: in_vivo_validation.get(key)
            for key in (
                "in_vivo_verdict",
                "verdict_scope",
                "not_in_vivo_bound",
                "requires_slice5_live_validation",
                "reason",
                "logged_density_surface",
                "dominating_density_evidence",
                "byte_comparison",
            )
        },
        "arc2b_verdict": merged,
        "quantile_acc_sizing": _compact_quantile_acc_sizing_summary(quantile_acc_sizing),
        "growth_branch": horizon_growth.get("growth_branch"),
        "final_sizing_verdict": merged["final_sizing_verdict"],
        "final_verdict_scope": merged["final_verdict_scope"],
    }
