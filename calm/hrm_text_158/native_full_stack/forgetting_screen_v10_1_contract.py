"""PLAN_v10.1r8 pure contract: two-branch G0 + transfer table + mass band.

Ordinary branch delegates to g0_valid_v10 BYTE-UNCHANGED (imported, never edited).
All predicate helpers are total/fail-closed: malformed/None → (False, reason), never raise.
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.forgetting_screen_v10_contract import (
    FORMAL150_CONTROL_SHA256,
    g0_valid_v10,
)

TERMINAL_MODE_DEGENERATE_FULL_SUPPRESSION = "degenerate_full_suppression"
CONTROL_CREDITED_MASS = 2_910_513_700
CREDITED_MASS_BAND = (0.90, 1.10)
FORMAL150_CONTROL_SHA256_V10_1 = FORMAL150_CONTROL_SHA256
PLAN_V10_1_SHA256 = "d55a21c44ab604adb547e947207774a55c244ab82a471eaf0891203bd0e643fa"
PLAN_V10_1_PATH = "artifacts/acc_entropy/forgetting_mechanism_screen_PLAN_v10_1.json"
AUTHORITY_DISPATCH_V10_1 = "1784910279857-8a789566"
SUPPRESSION_ARM = "arm1_decay_leak"
PRE_POST_SCHEMA = "v10_1_pre_post_transform_v1"
TRANSFER_LAW = "pre=clamp(acc+move,-127,127); out=trunc_toward_zero(pre*31/32)"
LAW_LABEL_TOKEN = "law_trunc_toward_zero_lambda_1_32_int16_carrier"
ACC_LO, ACC_HI = -127, 127
MOVES = (-1, 0, 1)
TRANSFER_CARDINALITY = 255 * 3
FORMAL_STEPS, FORMAL_BATCH, FORMAL_TOPK = 150, 8, 1024
FORMAL_DEVICE = "cuda:0"
H_SAMPLE_STEPS = (25, 50, 75, 100, 125, 150)
TERMINAL_LABEL_CANONICAL = (
    f"{SUPPRESSION_ARM}"
    f"__{LAW_LABEL_TOKEN}"
    f"__geom_steps{FORMAL_STEPS}_batch{FORMAL_BATCH}_topk{FORMAL_TOPK}_"
    f"{FORMAL_DEVICE.replace(':', '')}"
    f"__{TERMINAL_MODE_DEGENERATE_FULL_SUPPRESSION}"
)
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def trunc_toward_zero_mul_31_32(pre: int) -> int:
    return int(math.trunc(float(pre) * 31.0 / 32.0))


def transfer_pair(acc: int, move: int) -> tuple[int, int]:
    pre = max(ACC_LO, min(ACC_HI, int(acc) + int(move)))
    return pre, trunc_toward_zero_mul_31_32(pre)


def build_exhaustive_transfer_table() -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for acc in range(ACC_LO, ACC_HI + 1):
        for move in MOVES:
            pre, out = transfer_pair(acc, move)
            rows.append({"acc": acc, "move": move, "pre": pre, "out": out})
    return rows


def _as_finite_number(x: Any) -> float | None:
    """Authoritative numeric: type(x) in {int, float}, bool excluded, finite."""
    if type(x) not in (int, float):
        return None
    v = float(x)
    if not math.isfinite(v):
        return None
    return v


def _as_int(x: Any) -> int | None:
    """Exact JSON-int: type(x) is int (bool excluded). No float/string coercion."""
    if type(x) is int:
        return x
    return None


def credited_mass_ratio(
    arm_credited_mass: Any, *, control_credited_mass: Any = CONTROL_CREDITED_MASS
) -> float | None:
    den = _as_finite_number(control_credited_mass)
    if den is None or den <= 0.0:
        return None
    num = _as_finite_number(arm_credited_mass)
    if num is None:
        return None
    return num / den


def credited_mass_ratio_in_band(
    arm_credited_mass: Any,
    *,
    control_credited_mass: Any = CONTROL_CREDITED_MASS,
    band: tuple[float, float] = CREDITED_MASS_BAND,
) -> tuple[bool, float | None, str | None]:
    try:
        ratio = credited_mass_ratio(
            arm_credited_mass, control_credited_mass=control_credited_mass
        )
    except Exception:  # noqa: BLE001 — total fail-closed
        return False, None, "credited_mass_ratio_exception"
    if ratio is None:
        return False, None, "credited_mass_denominator_invalid_or_nonfinite"
    try:
        lo, hi = float(band[0]), float(band[1])
    except (TypeError, ValueError, IndexError):
        return False, ratio, "credited_mass_band_malformed"
    if ratio < lo or ratio > hi:
        return False, ratio, "credited_mass_ratio_outside_band"
    return True, ratio, None


def _side_stats_ok(block: Any, *, side: str) -> tuple[bool, str | None]:
    if not isinstance(block, Mapping):
        return False, f"pre_post_{side}_not_mapping"
    for sk in ("nonzero", "abs_max"):
        if sk not in block:
            return False, f"pre_post_{side}_missing:{sk}"
        iv = _as_int(block[sk])
        if iv is None or iv < 0:
            return False, f"pre_post_{side}_bad_int:{sk}"
    vals: dict[str, float] = {
        "nonzero": float(block["nonzero"]),
        "abs_max": float(block["abs_max"]),
    }
    for sk in ("abs_p50", "abs_p90"):
        if sk not in block:
            return False, f"pre_post_{side}_missing:{sk}"
        v = _as_finite_number(block[sk])
        if v is None:
            return False, f"pre_post_{side}_non_finite:{sk}"
        if v < 0.0:
            return False, f"pre_post_{side}_negative:{sk}"
        vals[sk] = v
    if vals["abs_p50"] > vals["abs_p90"] + 1e-12:
        return False, f"pre_post_{side}_p50_gt_p90"
    if vals["abs_p90"] > vals["abs_max"] + 1e-12:
        return False, f"pre_post_{side}_p90_gt_abs_max"
    return True, None


def pre_post_evidence_schema_valid(
    ev: Mapping[str, Any] | None, *, require_steps: int | None = None
) -> tuple[bool, str | None]:
    if ev is None or not isinstance(ev, Mapping):
        return False, "pre_post_evidence_missing"
    if ev.get("schema") != PRE_POST_SCHEMA:
        return False, "pre_post_schema_mismatch"
    if str(ev.get("law", "")) != TRANSFER_LAW:
        return False, "pre_post_law_mismatch"
    for forbidden in ("raw_acc", "raw_moves", "acc_tensor", "move_tensor", "arrays"):
        if forbidden in ev:
            return False, f"pre_post_raw_array_forbidden:{forbidden}"
    for k in (
        "move_nonzero_count",
        "post_projection",
        "post_decay",
        "pre_nonzero_to_post_zero_count",
        "post_decay_candidate_count",
        "law_mismatch_count",
        "steps_accumulated",
        "move_abs_bins",
    ):
        if k not in ev:
            return False, f"pre_post_missing:{k}"
    bins = ev.get("move_abs_bins")
    if not isinstance(bins, Mapping):
        return False, "pre_post_move_abs_bins_missing"
    bin_sum = 0
    for bk, bv in bins.items():
        # Bin keys are decimal strings of exact ints 1..127 (bin 0 forbidden).
        if type(bk) is not str or not bk.isdigit():
            return False, "pre_post_bin_key_non_int"
        bi = int(bk)  # key parse only; counts still require exact JSON-int
        if bi < 1 or bi > 127:
            return False, "pre_post_bin_key_out_of_domain_or_zero_bin"
        bi_c = _as_int(bv)
        if bi_c is None or bi_c < 0:
            return False, "pre_post_bin_count_bad"
        bin_sum += bi_c
    move_nz = _as_int(ev.get("move_nonzero_count"))
    if move_nz is None or move_nz < 0:
        return False, "pre_post_move_nonzero_bad"
    # Conservation is unconditional: empty bins valid ONLY when move_nonzero_count==0.
    if bin_sum != move_nz:
        return False, "pre_post_bin_sum_ne_move_nonzero"
    ok_p, r_p = _side_stats_ok(ev.get("post_projection"), side="post_projection")
    if not ok_p:
        return False, r_p
    ok_d, r_d = _side_stats_ok(ev.get("post_decay"), side="post_decay")
    if not ok_d:
        return False, r_d
    erased = _as_int(ev.get("pre_nonzero_to_post_zero_count"))
    cand = _as_int(ev.get("post_decay_candidate_count"))
    mismatch = _as_int(ev.get("law_mismatch_count"))
    steps_acc = _as_int(ev.get("steps_accumulated"))
    if erased is None or erased < 0:
        return False, "pre_post_erasure_bad"
    if cand is None or cand < 0:
        return False, "pre_post_cand_bad"
    if mismatch is None or mismatch < 0:
        return False, "pre_post_law_mismatch_bad"
    if steps_acc is None or steps_acc < 0:
        return False, "pre_post_steps_bad"
    if require_steps is not None and steps_acc != int(require_steps):
        return False, "pre_post_steps_accumulated_mismatch"
    proj_nz = _as_int(ev["post_projection"].get("nonzero"))
    decay_nz = _as_int(ev["post_decay"].get("nonzero"))
    if proj_nz is None or decay_nz is None:
        return False, "pre_post_side_nonzero_bad"
    frac = ev.get("pre_nonzero_to_post_zero_frac")
    if decay_nz == 0 and proj_nz > 0:
        if erased != proj_nz:
            return False, "pre_post_erasure_ne_proj_nonzero"
        fv = _as_finite_number(frac)
        if fv is None or abs(fv - 1.0) > 1e-9:
            return False, "pre_post_frac_not_one_on_full_erase"
    elif frac is not None:
        fv = _as_finite_number(frac)
        if fv is None or fv < 0.0 or fv > 1.0:
            return False, "pre_post_frac_out_of_range"
    return True, None


def _zero_lifecycle_ok(ds: Mapping[str, Any]) -> tuple[bool, str | None]:
    keys = (
        "N_events_evaluable",
        "N_survived_applied_within_H",
        "N_never_applied_within_H",
        "N_events_evaluable_early",
        "N_events_evaluable_late",
        "N_never_applied_within_H_early",
        "N_never_applied_within_H_late",
        "N_events_censored_insufficient_followup",
    )
    for k in keys:
        if k not in ds:
            return False, f"lifecycle_missing:{k}"
        v = _as_int(ds[k])
        if v is None or v < 0:
            return False, f"lifecycle_bad:{k}"
        if v != 0:
            return False, "lifecycle_nonzero"
    for fk in (
        "deferred_never_apply_within_H_frac",
        "deferred_survival_frac",
        "deferred_never_apply_within_H_frac_early",
        "deferred_never_apply_within_H_frac_late",
        "delta_never_apply",
    ):
        if fk in ds and ds[fk] is not None:
            return False, f"zero_cohort_fraction_coerced:{fk}"
    return True, None


def _h_trajectory_formal_zero(traj: Any) -> tuple[bool, str | None]:
    if not isinstance(traj, list) or len(traj) != len(H_SAMPLE_STEPS):
        return False, "H_trajectory_wrong_len"
    seen: list[int] = []
    for row, expected_step in zip(traj, H_SAMPLE_STEPS):
        if not isinstance(row, Mapping):
            return False, "H_trajectory_row_bad"
        st = _as_int(row.get("step"))
        if st != int(expected_step):
            return False, "H_trajectory_step_mismatch"
        h = _as_finite_number(row.get("H_bits_per_weight"))
        if h is None or h != 0.0:
            return False, "H_trajectory_nonzero"
        seen.append(st)
    return True, None


def _valid_hex64(x: Any) -> bool:
    return isinstance(x, str) and _HEX64.match(x) is not None


def suppression_diagnostic_match(receipt: Mapping[str, Any]) -> tuple[bool, str | None]:
    if not isinstance(receipt, Mapping):
        return False, "receipt_not_mapping"
    if str(receipt.get("arm", "")) != SUPPRESSION_ARM:
        return False, "arm_not_arm1_decay_leak"
    if _as_int(receipt.get("steps")) != FORMAL_STEPS:
        return False, "steps_not_150"
    if _as_int(receipt.get("batch")) != FORMAL_BATCH:
        return False, "batch_not_8"
    if _as_int(receipt.get("topk")) != FORMAL_TOPK:
        return False, "topk_not_1024"
    if str(receipt.get("device", "")) != FORMAL_DEVICE:
        return False, "device_not_cuda0"
    if str(receipt.get("plan_v10_1_sha256", "")) != PLAN_V10_1_SHA256:
        return False, "plan_v10_1_sha_mismatch"
    if str(receipt.get("plan_v10_1_path", "")) != PLAN_V10_1_PATH:
        return False, "plan_v10_1_path_mismatch"
    if str(receipt.get("authority_dispatch_v10_1", "")) != AUTHORITY_DISPATCH_V10_1:
        return False, "authority_dispatch_v10_1_mismatch"
    if str(receipt.get("pinned_control_sha256", "")) != FORMAL150_CONTROL_SHA256_V10_1:
        return False, "pinned_control_sha_mismatch"
    if receipt.get("pre_post_telemetry") is not True:
        return False, "pre_post_telemetry_not_on"
    meas = receipt.get("measurements")
    if not isinstance(meas, Mapping):
        return False, "measurements_missing"
    ds = meas.get("deferred_survival")
    demand = meas.get("demand")
    if not isinstance(ds, Mapping) or not isinstance(demand, Mapping):
        return False, "r1_surface_missing"
    for k in ("n_flips", "n_applied_drains", "q_changed_count"):
        if _as_int(meas.get(k)) != 0:
            return False, f"nonzero_{k}"
    ok_life, life_reason = _zero_lifecycle_ok(ds)
    if not ok_life:
        return False, life_reason
    mean_r = _as_finite_number(demand.get("mean_ratio"))
    max_r = _as_finite_number(demand.get("max_ratio"))
    if mean_r is None or max_r is None or mean_r != 0.0 or max_r != 0.0:
        return False, "demand_nonzero"
    ok_h, h_reason = _h_trajectory_formal_zero(meas.get("H_trajectory"))
    if not ok_h:
        return False, h_reason
    q_sha = receipt.get("q_sha")
    if not isinstance(q_sha, Mapping):
        return False, "q_sha_missing"
    qb, qa = q_sha.get("before"), q_sha.get("after")
    if not _valid_hex64(qb) or not _valid_hex64(qa):
        return False, "q_sha_not_hex64"
    if qb != qa:
        return False, "q_sha_changed"
    probes = receipt.get("probes")
    if not isinstance(probes, Mapping):
        return False, "probes_missing"
    if "skipped" not in probes:
        return False, "probes_skipped_missing"
    if probes.get("skipped") is not False:
        return False, "probes_skipped_not_false"
    acq0, acq1 = _as_int(probes.get("acq_step0_count")), _as_int(probes.get("acq_final_count"))
    ret0, ret1 = _as_int(probes.get("retention_step0_count")), _as_int(
        probes.get("retention_final_count")
    )
    if acq0 is None or acq1 is None or acq0 != acq1:
        return False, "acq_step0_final_mismatch"
    if ret0 is None or ret1 is None or ret0 != ret1:
        return False, "retention_step0_final_mismatch"
    if probes.get("retention_ok") is not True:
        return False, "retention_not_ok"
    route = receipt.get("route_counters")
    if not isinstance(route, Mapping):
        return False, "route_missing"
    if (_as_int(route.get("n_fixed_qscale_forwards")) or 0) <= 0:
        return False, "route_fixed_qscale_dead"
    if _as_int(route.get("n_bitlinear_dynamic_forwards")) != 0:
        return False, "route_dynamic_nonzero"
    ok_mass, _ratio, mass_reason = credited_mass_ratio_in_band(meas.get("credited_mass"))
    if not ok_mass:
        return False, mass_reason or "credited_mass_band_fail"
    ok_pp, pp_reason = pre_post_evidence_schema_valid(
        meas.get("pre_post_transform"), require_steps=FORMAL_STEPS
    )
    if not ok_pp:
        return False, pp_reason
    pp = meas["pre_post_transform"]
    if _as_int(pp.get("move_nonzero_count")) <= 0:
        return False, "pre_post_no_moves"
    if _as_int(pp["post_projection"].get("nonzero")) <= 0:
        return False, "pre_post_projection_zero_despite_moves"
    if _as_int(pp["post_decay"].get("nonzero")) != 0:
        return False, "pre_post_decay_nonzero"
    if _as_int(pp.get("post_decay_candidate_count")) != 0:
        return False, "pre_post_candidates_nonzero"
    if _as_int(pp.get("law_mismatch_count")) != 0:
        return False, "pre_post_law_mismatch_nonzero"
    # strong suppression evidence: all nonzero pre magnitudes must be 1
    if _as_int(pp["post_projection"].get("abs_max")) != 1:
        return False, "pre_post_proj_abs_max_not_1"
    if _as_int(pp.get("pre_nonzero_to_post_zero_count")) <= 0:
        return False, "pre_post_no_erasure"
    return True, None


def build_terminal_label_canonical(
    *,
    arm: str = SUPPRESSION_ARM,
    law: str = TRANSFER_LAW,
    steps: int = FORMAL_STEPS,
    batch: int = FORMAL_BATCH,
    topk: int = FORMAL_TOPK,
    device: str = FORMAL_DEVICE,
) -> str:
    """Mechanical join of arm/law/geometry into the terminal label.

    Fail-closed on mismatch vs frozen values: returns a non-canonical joined
    string (never silently substitutes the arm1 canonical label).
    """
    law_tok = LAW_LABEL_TOKEN if str(law) == TRANSFER_LAW else "law_MISMATCH"
    steps_i = _as_int(steps)
    batch_i = _as_int(batch)
    topk_i = _as_int(topk)
    if steps_i is None or batch_i is None or topk_i is None:
        return (
            f"{arm}__{law_tok}__geom_INVALID__"
            f"{TERMINAL_MODE_DEGENERATE_FULL_SUPPRESSION}"
        )
    device_tok = str(device).replace(":", "")
    return (
        f"{arm}"
        f"__{law_tok}"
        f"__geom_steps{steps_i}_batch{batch_i}_topk{topk_i}_{device_tok}"
        f"__{TERMINAL_MODE_DEGENERATE_FULL_SUPPRESSION}"
    )


def suppression_disposition(
    *,
    arm: str = SUPPRESSION_ARM,
    law: str = TRANSFER_LAW,
    steps: int = FORMAL_STEPS,
    batch: int = FORMAL_BATCH,
    topk: int = FORMAL_TOPK,
    device: str = FORMAL_DEVICE,
) -> dict[str, Any]:
    label = build_terminal_label_canonical(
        arm=arm, law=law, steps=steps, batch=batch, topk=topk, device=device
    )
    return {
        "terminal_mode": TERMINAL_MODE_DEGENERATE_FULL_SUPPRESSION,
        "receipt_valid_for_diagnosis": label == TERMINAL_LABEL_CANONICAL,
        "mechanism_eligible": False,
        "family_winner": False,
        "terminal_label": label,
        "excluded_from": ["E", "W", "S", "H_bar", "pressure_bar", "backlog_bar", "family_winner"],
        "deferred_survival_class": None,
        "arm": arm,
    }


def classify_discriminator_branch(pre_post: Any) -> str:
    ok, _reason = pre_post_evidence_schema_valid(pre_post)
    if not ok or not isinstance(pre_post, Mapping):
        return "wiring_or_representation"
    mismatch = _as_int(pre_post.get("law_mismatch_count"))
    if mismatch is None or mismatch > 0:
        return "wiring_or_representation"
    move_nz = _as_int(pre_post.get("move_nonzero_count")) or 0
    proj_nz = _as_int(pre_post["post_projection"].get("nonzero")) or 0
    decay_nz = _as_int(pre_post["post_decay"].get("nonzero")) or 0
    erased = _as_int(pre_post.get("pre_nonzero_to_post_zero_count")) or 0
    abs_max = _as_int(pre_post["post_projection"].get("abs_max")) or 0
    if move_nz <= 0 or proj_nz <= 0:
        return "wiring_or_representation"
    if decay_nz != 0:
        return "wiring_or_representation"
    # Under frozen law, |pre|>=2 cannot become 0 via trunc(pre*31/32)
    if abs_max >= 2:
        return "wiring_or_representation"
    if abs_max != 1 or erased != proj_nz:
        return "wiring_or_representation"
    return "strong_S1"


def g0_valid_v10_1(receipt: Mapping[str, Any]) -> tuple[bool, str | None, dict[str, Any]]:
    try:
        ok_sup, _sup_reason = suppression_diagnostic_match(receipt)
    except Exception as exc:  # noqa: BLE001
        return False, f"suppression_match_exception:{type(exc).__name__}", {
            "branch": "none",
            "disposition": None,
        }
    if ok_sup:
        return True, None, {
            "branch": "suppression_diagnostic",
            "disposition": suppression_disposition(
                arm=str(receipt.get("arm", "")),
                law=TRANSFER_LAW,
                steps=_as_int(receipt.get("steps")) or FORMAL_STEPS,
                batch=_as_int(receipt.get("batch")) or FORMAL_BATCH,
                topk=_as_int(receipt.get("topk")) or FORMAL_TOPK,
                device=str(receipt.get("device", FORMAL_DEVICE)),
            ),
        }
    try:
        from calm.hrm_text_158.native_full_stack.forgetting_screen_v10_contract import (
            arm_metrics_for_v10_classifier,
        )

        metrics = arm_metrics_for_v10_classifier(receipt)
        ok, reason = g0_valid_v10(metrics)
    except Exception as exc:  # noqa: BLE001
        return False, f"ordinary_metrics_build_failed:{type(exc).__name__}", {
            "branch": "none",
            "disposition": None,
        }
    if ok:
        return True, None, {"branch": "ordinary", "disposition": None}
    if reason in ("class_vacuous", "n_flips_vacuous", "cohort_below_min") or (
        reason is not None and "vacuous" in str(reason)
    ):
        return False, f"vacuous_unresolved:{reason}", {"branch": "none", "disposition": None}
    return False, reason, {"branch": "none", "disposition": None}
