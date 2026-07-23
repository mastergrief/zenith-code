"""Forgetting-family classifier + eligibility guards (PLAN_v9).

Extracted behavior-preservingly from forgetting_mechanism_screen_reducers.
"""
from __future__ import annotations

from typing import Any, Mapping

EPS = 1e-9
TIE_TOLERANCE_BPW = 0.02
SUB2_ACC_BUDGET_BPW = 0.4
H_PROGRESS_BAR_FRAC = 0.50
CENSOR_CLEAR_MAX = 0.50
N_FLIPS_VACUOUS = 1000
Q_MOTION_MIN_FRAC = 0.01
RETENTION_SLOP_COUNTS = 2

FAMILY_F1 = "F1_decay_leak"
FAMILY_F2 = "F2_ttl_age_drain"
FAMILY_F3 = "F3_sparse_hot_forgettable_cold"
FAMILY_F4 = "F4_design_family_null"

ARM1 = "arm1_decay_leak"
ARM2 = "arm2_ttl_age_drain"
ARM3 = "arm3_sparse_hot_forgettable_cold"
ARM0 = "arm0_coupled_q_no_forget"


def retention_ok(
    *,
    final_count: int,
    step0_count: int,
    slop: int = RETENTION_SLOP_COUNTS,
) -> bool:
    """COUNT units only — reject rate-style misuse by requiring ints."""
    if not isinstance(final_count, int) or not isinstance(step0_count, int):
        raise TypeError("retention_ok requires integer exact-match counts (not rates)")
    return int(final_count) >= int(step0_count) - int(slop)


def g0_valid(*, lifetime_censored_frac_value: float, n_flips: int) -> bool:
    return not (
        float(lifetime_censored_frac_value) >= CENSOR_CLEAR_MAX
        or int(n_flips) < N_FLIPS_VACUOUS
    )


def g0b_q_motion_ok(*, q_changed_count: int, n_applied_drains: int) -> bool:
    floor = max(1, int(Q_MOTION_MIN_FRAC * int(n_applied_drains)))
    return int(q_changed_count) >= floor


def g1_survival_ok(*, retention_ok_flag: bool, acq_delta_count: int) -> bool:
    return bool(retention_ok_flag) and int(acq_delta_count) >= 0


def classify_forgetting_family_screen(
    *,
    phase0_censor_cleared: bool,
    H_control_final: float,
    arm_metrics: Mapping[str, Mapping[str, Any]],
    tie_tolerance_bpw: float = TIE_TOLERANCE_BPW,
) -> dict[str, Any]:
    """PLAN_v9 classifier: control-budget → E → S → F3≥F1≥F2."""
    out: dict[str, Any] = {
        "eps": EPS,
        "tie_tolerance_bpw": float(tie_tolerance_bpw),
        "family": FAMILY_F4,
        "stop_reason": None,
        "E": [],
        "S": [],
        "H_progress": {},
        "multi_match": False,
    }

    if not phase0_censor_cleared:
        out["stop_reason"] = "phase0_censor_uncleared"
        return out

    if float(H_control_final) <= SUB2_ACC_BUDGET_BPW:
        out["family"] = FAMILY_F4
        out["stop_reason"] = "control_already_at_budget_no_forgetting_family"
        return out

    gap = max(EPS, float(H_control_final) - SUB2_ACC_BUDGET_BPW)
    bar = H_PROGRESS_BAR_FRAC * gap

    # Per-arm guards → E
    E: list[str] = []
    H_prog: dict[str, float] = {}
    for arm in (ARM1, ARM2, ARM3):
        m = arm_metrics[arm]
        Hp = max(0.0, float(H_control_final) - float(m["H_final"]))
        H_prog[arm] = Hp
        ok = (
            g0_valid(
                lifetime_censored_frac_value=float(m["lifetime_censored_frac"]),
                n_flips=int(m["n_flips"]),
            )
            and g0b_q_motion_ok(
                q_changed_count=int(m["q_changed_count"]),
                n_applied_drains=int(m["n_applied_drains"]),
            )
            and g1_survival_ok(
                retention_ok_flag=bool(m["retention_ok"]),
                acq_delta_count=int(m["acq_delta_count"]),
            )
        )
        if ok:
            E.append(arm)
    out["E"] = list(E)
    out["H_progress"] = dict(H_prog)

    if not E or not any(H_prog[a] >= bar for a in E):
        out["stop_reason"] = "R0_null_guard"
        return out

    max_E = max(H_prog[a] for a in E)
    S = [
        a
        for a in E
        if H_prog[a] >= bar and H_prog[a] >= max_E - float(tie_tolerance_bpw)
    ]
    out["S"] = list(S)

    if ARM3 in S:
        out["family"] = FAMILY_F3
        out["stop_reason"] = "R1_F3"
        out["multi_match"] = len(S) > 1
        return out
    if ARM1 in S and ARM3 not in S:
        out["family"] = FAMILY_F1
        out["stop_reason"] = "R2_F1"
        out["multi_match"] = len(S) > 1
        return out
    if S == [ARM2] or set(S) == {ARM2}:
        out["family"] = FAMILY_F2
        out["stop_reason"] = "R3_F2"
        return out

    out["stop_reason"] = "R4_ambiguous_null"
    return out
