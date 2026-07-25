"""PLAN_v10 contract: bind/G0/bars/classifier. Pure mapping (no file IO)."""
from __future__ import annotations

import math
from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.family_classifier import (
    ARM1, ARM2, ARM3, EPS, FAMILY_F1, FAMILY_F2, FAMILY_F3, FAMILY_F4,
    H_PROGRESS_BAR_FRAC, N_FLIPS_VACUOUS, SUB2_ACC_BUDGET_BPW, TIE_TOLERANCE_BPW,
    g0b_q_motion_ok, g1_survival_ok,
)

MIN_COHORT_N = 100
GROWING_DEFERRED_SURVIVAL_DELTA = 0.10
STABLE_HIGH_DEFERRED_NEVER_APPLY_FLOOR = 0.50
HIGH_DEMAND_RATIO = 2.0
SUSTAINED_HIGH_DEMAND_FRAC_STEPS = 0.50
PRESSURE_PROGRESS_BAR = 0.50
BACKLOG_PROGRESS_BAR = 0.10
RATIO_COUNT_EPS = 1e-6
RECOGNIZED_DEFERRED_SURVIVAL_ENUM = frozenset(
    {"vacuous", "other", "growing", "stable_high", "collapsing"}
)
CLASS_EXIT_WINNERS = frozenset({"other", "collapsing"})
FORMAL150_CONTROL_SHA256 = "5e593454f0ddffb946692e09913da5df1ddfe0f2f11aaaf3fb663a2f00fbcfdb"

_C2_PATHS: tuple[tuple[str, ...], ...] = (
    ("measurements", "demand", "mean_ratio"),
    ("measurements", "demand", "max_ratio"),
    ("measurements", "demand", "frac_steps_ratio_ge_2"),
    ("measurements", "demand", "n_steps"),
    ("measurements", "deferred_survival", "N_events_evaluable"),
    ("measurements", "deferred_survival", "N_survived_applied_within_H"),
    ("measurements", "deferred_survival", "N_never_applied_within_H"),
    ("measurements", "deferred_survival", "N_events_censored_insufficient_followup"),
    ("measurements", "deferred_survival", "N_events_evaluable_early"),
    ("measurements", "deferred_survival", "N_events_evaluable_late"),
    ("measurements", "deferred_survival", "N_never_applied_within_H_early"),
    ("measurements", "deferred_survival", "N_never_applied_within_H_late"),
    ("measurements", "deferred_survival", "deferred_never_apply_within_H_frac"),
    ("measurements", "deferred_survival", "deferred_never_apply_within_H_frac_early"),
    ("measurements", "deferred_survival", "deferred_never_apply_within_H_frac_late"),
    ("measurements", "deferred_survival", "delta_never_apply"),
    ("measurements", "deferred_survival", "deferred_survival_class"),
    ("measurements", "H_bits_per_weight"),
    ("probes", "acq_step0_count"),
    ("probes", "acq_final_count"),
    ("probes", "acq_delta_count"),
    ("probes", "ret_step0_count"),
    ("probes", "ret_final_count"),
    ("probes", "retention_ok"),
    ("classifier", "family"),
    ("classifier", "stop_reason"),
)

def _dig(obj: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            raise KeyError(".".join(path))
        cur = cur[key]
    return cur

def _finite(value: Any) -> float:
    x = float(value)
    if math.isnan(x) or math.isinf(x):
        raise ValueError("non-finite")
    return x

def _fail(reason: str, **extra: Any) -> dict[str, Any]:
    out = {"ok": False, "reason": reason, "action": "stop"}
    out.update(extra)
    return out

def validate_control_baseline_bind(
    baseline_obj: Mapping[str, Any] | None,
    *,
    expected_sha256: str,
    actual_sha256: str | None = None,
    require_exact_c2_echo: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """CONTROL_BASELINE_BIND — fail-closed; no censor fallback. Mapping-in only."""
    if baseline_obj is None:
        return _fail("control_baseline_missing")
    if actual_sha256 is not None and str(actual_sha256) != str(expected_sha256):
        return _fail(
            "control_baseline_sha_mismatch",
            expected_sha256=str(expected_sha256),
            actual_sha256=str(actual_sha256),
        )
    try:
        if (
            int(baseline_obj["steps"]) != 150
            or int(baseline_obj["batch"]) != 8
            or int(baseline_obj["topk"]) != 1024
            or str(baseline_obj["arm"]) != "arm0_coupled_q_no_forget"
            or baseline_obj["telemetry"] is not True
        ):
            return _fail("geometry_field_mismatch")
    except (KeyError, TypeError, ValueError):
        return _fail("geometry_field_mismatch")
    try:
        if str(_dig(baseline_obj, ("paired_proof", "device"))) != "cuda:0":
            return _fail("paired_proof_device_mismatch")
    except KeyError:
        return _fail("paired_proof_device_missing")

    extracted: dict[str, Any] = {}
    try:
        for path in _C2_PATHS:
            extracted[".".join(path)] = _dig(baseline_obj, path)
        traj = baseline_obj["measurements"]["H_trajectory"]
        if not isinstance(traj, list) or not traj:
            return _fail("H_trajectory_missing")
        extracted["measurements.H_trajectory[0].H_bits_per_weight"] = traj[0][
            "H_bits_per_weight"
        ]
    except (KeyError, TypeError, IndexError):
        return _fail("c2_key_missing")
    if len(extracted) != 27:
        return _fail("c2_key_count_mismatch", n_keys=len(extracted))
    if require_exact_c2_echo is not None:
        for key, expected in require_exact_c2_echo.items():
            if key not in extracted or extracted[key] != expected:
                return _fail("c2_echo_mismatch", key=key)
    return {
        "ok": True,
        "reason": None,
        "action": "ok",
        "c2_key_count": 27,
        "extracted": extracted,
        "H_control_final": float(extracted["measurements.H_bits_per_weight"]),
        "H_step25": float(extracted["measurements.H_trajectory[0].H_bits_per_weight"]),
        "mean_ratio_control": float(extracted["measurements.demand.mean_ratio"]),
        "never_frac_control": float(
            extracted["measurements.deferred_survival.deferred_never_apply_within_H_frac"]
        ),
        "frac_ge2_control": float(extracted["measurements.demand.frac_steps_ratio_ge_2"]),
    }

def _num(m: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in m and m[k] is not None:
            return float(m[k])
    return float(default)

def _inum(m: Mapping[str, Any], *keys: str, default: int = 0) -> int:
    for k in keys:
        if k in m and m[k] is not None:
            return int(m[k])
    return int(default)

def recompute_deferred_survival_class(
    *, n_eval: int, n_early: int, n_late: int, never_frac: float, delta: float | None,
) -> str:
    """Lifecycle thresholds — claimed class mismatch fails closed."""
    if n_eval == 0:
        return "vacuous"
    if n_early < MIN_COHORT_N or n_late < MIN_COHORT_N:
        return "vacuous" if (n_early == 0 and n_late == 0) else "other"
    if delta is None:
        return "other"
    if delta >= GROWING_DEFERRED_SURVIVAL_DELTA:
        return "growing"
    if (
        abs(delta) < GROWING_DEFERRED_SURVIVAL_DELTA
        and never_frac >= STABLE_HIGH_DEFERRED_NEVER_APPLY_FLOOR
        and n_eval >= MIN_COHORT_N
    ):
        return "stable_high"
    if delta <= -GROWING_DEFERRED_SURVIVAL_DELTA:
        return "collapsing"
    return "other"

def arm_metrics_for_v10_classifier(arm_receipt: Mapping[str, Any]) -> dict[str, Any]:
    m = arm_receipt.get("measurements") or {}
    probes = arm_receipt.get("probes") or {}
    ds = m.get("deferred_survival") or {}
    demand = m.get("demand") or {}
    h_raw = m.get("H_bits_per_weight", m.get("H_final"))
    try:
        h_final = float(h_raw) if h_raw is not None else None
        if h_final is not None and (math.isnan(h_final) or math.isinf(h_final)):
            h_final = None
    except (TypeError, ValueError):
        h_final = None
    early_nf = ds.get("deferred_never_apply_within_H_frac_early", m.get("early_never_frac"))
    late_nf = ds.get("deferred_never_apply_within_H_frac_late", m.get("late_never_frac"))
    delta = ds.get("delta_never_apply", m.get("delta_never_apply"))
    return {
        "H_final": h_final,
        "n_flips": int(m.get("n_flips", 0)),
        "q_changed_count": int(m.get("q_changed_count", 0)),
        "n_applied_drains": int(m.get("n_applied_drains", 0)),
        "lifetime_censored_frac": float(m.get("lifetime_censored_frac", 1.0)),
        "retention_ok": bool(probes.get("retention_ok", False)),
        "acq_delta_count": int(probes.get("acq_delta_count", -10**9)),
        "mean_ratio": _num(demand, "mean_ratio", default=_num(m, "mean_ratio")),
        "max_ratio": _num(demand, "max_ratio", default=_num(m, "max_ratio", default=float("nan"))),
        "frac_steps_ratio_ge_2": _num(
            demand, "frac_steps_ratio_ge_2", default=_num(m, "frac_steps_ratio_ge_2", default=1.0)
        ),
        "n_steps": _inum(demand, "n_steps", default=_inum(m, "n_steps")),
        "N_events_evaluable": _inum(ds, "N_events_evaluable", default=_inum(m, "N_events_evaluable")),
        "N_survived_applied_within_H": _inum(
            ds, "N_survived_applied_within_H", default=_inum(m, "N_survived_applied_within_H")
        ),
        "N_never_applied_within_H": _inum(
            ds, "N_never_applied_within_H", default=_inum(m, "N_never_applied_within_H")
        ),
        "N_events_censored_insufficient_followup": _inum(
            ds, "N_events_censored_insufficient_followup",
            default=_inum(m, "N_events_censored_insufficient_followup"),
        ),
        "N_events_evaluable_early": _inum(
            ds, "N_events_evaluable_early", default=_inum(m, "N_events_evaluable_early")
        ),
        "N_events_evaluable_late": _inum(
            ds, "N_events_evaluable_late", default=_inum(m, "N_events_evaluable_late")
        ),
        "N_never_applied_within_H_early": _inum(
            ds, "N_never_applied_within_H_early", default=_inum(m, "N_never_applied_within_H_early")
        ),
        "N_never_applied_within_H_late": _inum(
            ds, "N_never_applied_within_H_late", default=_inum(m, "N_never_applied_within_H_late")
        ),
        "never_frac": _num(
            ds, "deferred_never_apply_within_H_frac",
            default=_num(m, "never_frac", "deferred_never_apply_within_H_frac", default=1.0),
        ),
        "early_never_frac": None if early_nf is None else float(early_nf),
        "late_never_frac": None if late_nf is None else float(late_nf),
        "delta_never_apply": None if delta is None else float(delta),
        "deferred_survival_class": ds.get(
            "deferred_survival_class", m.get("deferred_survival_class")
        ),
        "receipt_steps": int(arm_receipt.get("steps", -1)),
        # ARM2-conditional: pass through only when present — NEVER m.get(..., 0).
        **(
            {"n_ttl_force_zero_drains": int(m["n_ttl_force_zero_drains"])}
            if "n_ttl_force_zero_drains" in m
            else {}
        ),
        # ARM3-conditional: pass through only when present — NEVER m.get(..., 0).
        **(
            {"n_sparse_hot_cold_zeros": int(m["n_sparse_hot_cold_zeros"])}
            if "n_sparse_hot_cold_zeros" in m
            else {}
        ),
    }

def g0_valid_v10(m: Mapping[str, Any]) -> tuple[bool, str | None]:
    """v10 G0: full R1 surface + cohorts + conservation + class recompute; lcf non-gating."""
    req = (
        "n_flips", "N_events_evaluable", "N_survived_applied_within_H",
        "N_never_applied_within_H", "N_events_evaluable_early", "N_events_evaluable_late",
        "N_never_applied_within_H_early", "N_never_applied_within_H_late",
        "N_events_censored_insufficient_followup", "never_frac", "deferred_survival_class",
        "mean_ratio", "max_ratio", "frac_steps_ratio_ge_2", "n_steps",
        "early_never_frac", "late_never_frac", "delta_never_apply", "H_final",
        "receipt_steps",
    )
    for k in req:
        if k not in m or m[k] is None:
            return False, f"missing_field:{k}"
    try:
        n_eval = int(m["N_events_evaluable"])
        n_surv = int(m["N_survived_applied_within_H"])
        n_never = int(m["N_never_applied_within_H"])
        n_early = int(m["N_events_evaluable_early"])
        n_late = int(m["N_events_evaluable_late"])
        n_never_e = int(m["N_never_applied_within_H_early"])
        n_never_l = int(m["N_never_applied_within_H_late"])
        n_cens = int(m["N_events_censored_insufficient_followup"])
        n_flips = int(m["n_flips"])
        n_steps = int(m["n_steps"])
        receipt_steps = int(m["receipt_steps"])
        never_frac = _finite(m["never_frac"])
        early_nf = _finite(m["early_never_frac"])
        late_nf = _finite(m["late_never_frac"])
        delta = _finite(m["delta_never_apply"])
        mean_r = _finite(m["mean_ratio"])
        max_r = _finite(m["max_ratio"])
        frac = _finite(m["frac_steps_ratio_ge_2"])
        _finite(m["H_final"])
        klass = m["deferred_survival_class"]
    except (TypeError, ValueError):
        return False, "non_finite_or_bad_type"
    for n in (n_eval, n_surv, n_never, n_early, n_late, n_never_e, n_never_l, n_cens, n_flips):
        if n < 0:
            return False, "negative_count"
    if n_steps != 150:
        return False, "n_steps_not_150"
    if receipt_steps != n_steps:
        return False, "n_steps_receipt_steps_mismatch"
    if not (0.0 <= never_frac <= 1.0) or not (0.0 <= frac <= 1.0):
        return False, "fraction_out_of_range"
    if not (0.0 <= early_nf <= 1.0) or not (0.0 <= late_nf <= 1.0):
        return False, "cohort_fraction_out_of_range"
    if max_r < mean_r:
        return False, "max_ratio_lt_mean_ratio"
    if n_flips < N_FLIPS_VACUOUS:
        return False, "n_flips_vacuous"
    if n_eval < MIN_COHORT_N or n_early < MIN_COHORT_N or n_late < MIN_COHORT_N:
        return False, "cohort_below_min"
    if n_eval != n_surv + n_never:
        return False, "conservation_eval_surv_never"
    if n_early + n_late != n_eval:
        return False, "conservation_early_late_eval"
    if n_never_e + n_never_l != n_never:
        return False, "conservation_early_late_never"
    if abs(never_frac - (n_never / max(1, n_eval))) > RATIO_COUNT_EPS:
        return False, "never_frac_count_mismatch"
    if abs(early_nf - (n_never_e / max(1, n_early))) > RATIO_COUNT_EPS:
        return False, "early_never_frac_count_mismatch"
    if abs(late_nf - (n_never_l / max(1, n_late))) > RATIO_COUNT_EPS:
        return False, "late_never_frac_count_mismatch"
    if abs(delta - (late_nf - early_nf)) > RATIO_COUNT_EPS:
        return False, "delta_never_apply_mismatch"
    if not isinstance(klass, str) or klass not in RECOGNIZED_DEFERRED_SURVIVAL_ENUM:
        return False, "class_unrecognized"
    if klass == "vacuous":
        return False, "class_vacuous"
    expected = recompute_deferred_survival_class(
        n_eval=n_eval, n_early=n_early, n_late=n_late, never_frac=never_frac, delta=delta
    )
    if klass != expected:
        return False, "class_recompute_mismatch"
    return True, None

def regime_exit_v10(*, mean_ratio_arm: float, frac_ge2_arm: float) -> bool:
    return (float(mean_ratio_arm) < HIGH_DEMAND_RATIO) or (
        float(frac_ge2_arm) < SUSTAINED_HIGH_DEMAND_FRAC_STEPS
    )

def pressure_bar_v10(
    *, mean_ratio_arm: float, frac_ge2_arm: float, mean_ratio_control: float
) -> bool:
    progress = max(0.0, float(mean_ratio_control) - float(mean_ratio_arm)) / max(
        EPS, float(mean_ratio_control)
    )
    return progress >= PRESSURE_PROGRESS_BAR or regime_exit_v10(
        mean_ratio_arm=mean_ratio_arm, frac_ge2_arm=frac_ge2_arm
    )

def backlog_bar_v10(*, never_frac_arm: float, never_frac_control: float, klass: str) -> bool:
    if klass not in RECOGNIZED_DEFERRED_SURVIVAL_ENUM:
        return False
    progress = max(0.0, float(never_frac_control) - float(never_frac_arm))
    return progress >= BACKLOG_PROGRESS_BAR or klass in CLASS_EXIT_WINNERS

def h_bar_v10(*, H_arm_final: float, H_control_final: float) -> bool:
    gap = max(EPS, float(H_control_final) - SUB2_ACC_BUDGET_BPW)
    return max(0.0, float(H_control_final) - float(H_arm_final)) >= H_PROGRESS_BAR_FRAC * gap

def classify_forgetting_family_screen_v10(
    *,
    control_baseline_ok: bool,
    H_control_final: float,
    mean_ratio_control: float,
    never_frac_control: float,
    arm_metrics: Mapping[str, Mapping[str, Any]],
    tie_tolerance_bpw: float = TIE_TOLERANCE_BPW,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "eps": EPS, "tie_tolerance_bpw": float(tie_tolerance_bpw),
        "family": FAMILY_F4, "stop_reason": None,
        "E": [], "W": [], "S": [], "H_progress": {},
        "pressure_bar": {}, "backlog_bar": {}, "h_bar": {}, "g0_reason": {},
        "multi_match": False, "plan": "v10",
    }
    if not control_baseline_ok:
        out["stop_reason"] = "control_baseline_not_ok"
        return out
    if float(H_control_final) <= SUB2_ACC_BUDGET_BPW:
        out["stop_reason"] = "control_already_at_budget_no_forgetting_family"
        return out
    E: list[str] = []
    W: list[str] = []
    H_prog: dict[str, float] = {}
    for arm in (ARM1, ARM2, ARM3):
        m = arm_metrics[arm]
        ok_g0, g0_reason = g0_valid_v10(m)
        out["g0_reason"][arm] = g0_reason
        ok = (
            ok_g0
            and g0b_q_motion_ok(
                q_changed_count=int(m["q_changed_count"]),
                n_applied_drains=int(m["n_applied_drains"]),
            )
            and g1_survival_ok(
                retention_ok_flag=bool(m["retention_ok"]),
                acq_delta_count=int(m["acq_delta_count"]),
            )
        )
        Hp = max(0.0, float(H_control_final) - float(m["H_final"]))
        H_prog[arm] = Hp
        hb = h_bar_v10(H_arm_final=float(m["H_final"]), H_control_final=float(H_control_final))
        pb = pressure_bar_v10(
            mean_ratio_arm=float(m["mean_ratio"]),
            frac_ge2_arm=float(m["frac_steps_ratio_ge_2"]),
            mean_ratio_control=float(mean_ratio_control),
        )
        bb = backlog_bar_v10(
            never_frac_arm=float(m["never_frac"]),
            never_frac_control=float(never_frac_control),
            klass=str(m["deferred_survival_class"]),
        )
        out["h_bar"][arm] = hb
        out["pressure_bar"][arm] = pb
        out["backlog_bar"][arm] = bb
        if ok:
            E.append(arm)
            if hb and pb and bb:
                W.append(arm)
    out["E"] = list(E)
    out["W"] = list(W)
    out["H_progress"] = dict(H_prog)
    if not W:
        out["stop_reason"] = "R0_null_no_arm_clears_pressure_backlog_and_H_while_surviving"
        return out
    max_W = max(H_prog[a] for a in W)
    S = [a for a in W if H_prog[a] >= max_W - float(tie_tolerance_bpw)]
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

def build_v10_terminal_receipt(
    *,
    control_bind: Mapping[str, Any],
    arm_receipts: Mapping[str, Mapping[str, Any]] | None,
    plan_sha256: str,
    authority_dispatch: str,
    force_null_reason: str | None = None,
) -> dict[str, Any]:
    arm_receipts = arm_receipts or {}
    non_claims = [
        "v10 control-bind screen — no forgetting-law ship",
        "no sub-2 achievability claim",
        "lcf non-gating under v10",
    ]
    if force_null_reason is not None or not bool(control_bind.get("ok")):
        reason = force_null_reason or str(control_bind.get("reason") or "control_baseline_not_ok")
        return {
            "screen": "forgetting_mechanism_phase1/v10",
            "plan_sha256": plan_sha256,
            "authority_dispatch": authority_dispatch,
            "authoritative": False,
            "control_baseline_ok": bool(control_bind.get("ok")),
            "control_bind": dict(control_bind),
            "transition": None,
            "arms_classified": False,
            "family": FAMILY_F4,
            "stop_reason": reason,
            "classifier": {
                "family": FAMILY_F4, "stop_reason": reason,
                "E": [], "W": [], "S": [], "plan": "v10",
            },
            "explicit_non_claims": non_claims,
        }
    metrics = {
        ARM1: arm_metrics_for_v10_classifier(arm_receipts[ARM1]),
        ARM2: arm_metrics_for_v10_classifier(arm_receipts[ARM2]),
        ARM3: arm_metrics_for_v10_classifier(arm_receipts[ARM3]),
    }
    verdict = classify_forgetting_family_screen_v10(
        control_baseline_ok=True,
        H_control_final=float(control_bind["H_control_final"]),
        mean_ratio_control=float(control_bind["mean_ratio_control"]),
        never_frac_control=float(control_bind["never_frac_control"]),
        arm_metrics=metrics,
    )
    bind_keep = {
        k: control_bind[k]
        for k in (
            "ok", "c2_key_count", "H_control_final", "H_step25",
            "mean_ratio_control", "never_frac_control", "frac_ge2_control",
        )
        if k in control_bind
    }
    return {
        "screen": "forgetting_mechanism_phase1/v10",
        "plan_sha256": plan_sha256,
        "authority_dispatch": authority_dispatch,
        "authoritative": True,
        "control_baseline_ok": True,
        "control_bind": bind_keep,
        "transition": None,
        "arms_classified": True,
        "H_control_final": float(control_bind["H_control_final"]),
        "arm_metrics": metrics,
        "classifier": verdict,
        "family": verdict["family"],
        "stop_reason": verdict["stop_reason"],
        "explicit_non_claims": non_claims,
    }
