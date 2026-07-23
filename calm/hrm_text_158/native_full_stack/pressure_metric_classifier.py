"""Pure classifier + precedence for pressure/metric diagnostic (PLAN_v6).

Owns: select_family_from_predicates + classify_pressure_metric_family.
Readiness + receipt assembly live in pressure_metric_receipt.py.
Dependency: classifier → telemetry (constants only). Never imports lifecycle/CLI.
Bound by PLAN_v6 sha 346b67d8…; rev3 re-scope 1784828063166.
"""
from __future__ import annotations

from typing import Any

from calm.hrm_text_158.native_full_stack.pressure_metric_telemetry import (
    HIGH_DEMAND_RATIO,
    HIGH_LCF,
    LABEL_R0,
    LABEL_R1,
    LABEL_R2,
    LABEL_R3,
    LABEL_R4,
    LOW_MODERATE_DEMAND_RATIO_MAX,
    MATERIAL_H_MOTION_BPW,
    MIN_COHORT_N,
    REPRESENTATION_IMMOVABLE_H_DELTA_MAX,
    SUSTAINED_HIGH_DEMAND_FRAC_STEPS,
)


def select_family_from_predicates(
    *,
    r1: bool,
    r2_raw: bool,
    r3: bool,
) -> dict[str, Any]:
    """Pure precedence selector — R1 ≻ R2 ≻ R3 ≻ R4; multi_match when R1∧R2_raw."""
    out: dict[str, Any] = {
        "family": LABEL_R4,
        "stop_reason": "R4_else",
        "multi_match": False,
        "R1": bool(r1),
        "R2": bool(r2_raw),
        "R3": bool(r3),
        "label": LABEL_R4,
    }
    if r1 and r2_raw:
        out.update(
            multi_match=True,
            family=LABEL_R1,
            label=LABEL_R1,
            stop_reason="R1_pressure_multi_match_prefers_R1",
        )
        return out
    if r1:
        out.update(family=LABEL_R1, label=LABEL_R1, stop_reason="R1_pressure")
        return out
    if r2_raw:
        out.update(family=LABEL_R2, label=LABEL_R2, stop_reason="R2_metric")
        return out
    if r3:
        out.update(
            family=LABEL_R3,
            label=LABEL_R3,
            stop_reason="R3_observation_unresolved",
        )
        return out
    return out


def classify_pressure_metric_family(
    *,
    telemetry_ok: bool,
    two_tier_threshold_assert_pass: bool,
    paired_determinism_cost_ok: bool | None,
    N_events_evaluable: int,
    mean_ratio: float | None,
    frac_steps_ratio_ge_2: float | None,
    deferred_survival_class: str | None,
    delta_never_apply: float | None,
    N_events_evaluable_early: int,
    N_events_evaluable_late: int,
    deferred_never_apply_within_H_frac: float | None,
    lcf: float | None,
    H_final: float | None,
    H_step25: float | None,
    retention_ok_flag: bool | None,
    readiness_ok: bool | None = True,
    readiness_stop_reason: str | None = None,
) -> dict[str, Any]:
    """PLAN_v6 classifier — never emits representation_limit.

    Fail-closed: paired_determinism_cost_ok must be Exactly True (None/False → R0).
    readiness_ok must also be Exactly True when supplied.
    """
    out: dict[str, Any] = {
        "family": LABEL_R4,
        "stop_reason": None,
        "multi_match": False,
        "R1": False,
        "R2": False,
        "R3": False,
        "label": LABEL_R4,
    }
    if readiness_ok is not True:
        out.update(
            family=LABEL_R0,
            stop_reason=str(readiness_stop_reason or "readiness_fail"),
            label=LABEL_R0,
        )
        return out
    if not telemetry_ok:
        out.update(family=LABEL_R0, stop_reason="missing_telemetry", label=LABEL_R0)
        return out
    if not two_tier_threshold_assert_pass:
        out.update(
            family=LABEL_R0,
            stop_reason="two_tier_threshold_assert_fail",
            label=LABEL_R0,
        )
        return out
    if paired_determinism_cost_ok is not True:
        reason = (
            "paired_determinism_or_cost_missing"
            if paired_determinism_cost_ok is None
            else "paired_determinism_or_cost_fail"
        )
        out.update(family=LABEL_R0, stop_reason=reason, label=LABEL_R0)
        return out
    if int(N_events_evaluable) == 0:
        out.update(
            family=LABEL_R0,
            stop_reason="deferred_survival_denominator_zero",
            label=LABEL_R0,
        )
        return out
    if (
        mean_ratio is None
        or frac_steps_ratio_ge_2 is None
        or lcf is None
        or H_final is None
        or H_step25 is None
    ):
        missing = []
        if mean_ratio is None:
            missing.append("mean_ratio")
        if frac_steps_ratio_ge_2 is None:
            missing.append("frac_steps_ratio_ge_2")
        if lcf is None:
            missing.append("lcf")
        if H_final is None:
            missing.append("H_final")
        if H_step25 is None:
            missing.append("H_step25")
        out.update(
            family=LABEL_R0,
            stop_reason=f"missing_telemetry:{','.join(missing)}",
            label=LABEL_R0,
        )
        return out

    H_control_ref = float(H_step25)
    growing_or_stable = False
    if (
        int(N_events_evaluable_early) >= MIN_COHORT_N
        and int(N_events_evaluable_late) >= MIN_COHORT_N
        and deferred_survival_class in ("growing", "stable_high")
    ):
        growing_or_stable = True

    r1 = (
        float(mean_ratio) >= HIGH_DEMAND_RATIO
        and float(frac_steps_ratio_ge_2) >= SUSTAINED_HIGH_DEMAND_FRAC_STEPS
        and growing_or_stable
    )
    r2_raw = (
        float(mean_ratio) <= LOW_MODERATE_DEMAND_RATIO_MAX
        and float(lcf) >= HIGH_LCF
        and abs(float(H_final) - float(H_step25)) >= MATERIAL_H_MOTION_BPW
    )
    r3 = (
        (not r1)
        and (not r2_raw)
        and float(mean_ratio) <= LOW_MODERATE_DEMAND_RATIO_MAX
        and bool(retention_ok_flag)
        and abs(float(H_final) - H_control_ref) <= REPRESENTATION_IMMOVABLE_H_DELTA_MAX
    )

    selected = select_family_from_predicates(r1=r1, r2_raw=r2_raw, r3=r3)
    selected["H_control_ref"] = H_control_ref
    return selected
