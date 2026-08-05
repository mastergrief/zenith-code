"""A′ slice-4 Rung-4 support-split residual densify pure reducer (STEP-1).

D1/D2/D3 densify axes. IMPORT_ONLY residual arithmetic. No IO/CLI.
PLAN v6: feea775c3b3bb1bee6f0d5775d4da783b09560b72b4a1b6cd8500af5f56329a9
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

from scripts.a_prime_slice4_residual_classification_reducer_v0 import (
    residual_composite,
    residual_per_support_label,
)
from scripts.a_prime_slice4_support_split_residual_densify_schema_v0 import (
    AUTHORITY_ARM,
    AUTHORITY_HORIZON,
    FROZEN_NEUTRAL_SUCCESSOR_TEXTS as T,
    MIN_BUCKET_DENOMINATOR_ROWS,
    PRODUCT,
    REQUIRED_CLAIM_BOUNDARY,
    SCHEMA_ID,
    SUPPORTS,
    admit_horizon_view,
    ceil_0_70,
    enrichment_ge_1_5,
    enrichment_le_0_5,
    extract_bucket_map,
    extract_survivors,
    extract_universe,
    is_exact_dict,
    is_exact_list,
    is_exact_str,
)

NON_AUTHORITATIVE_KEYS: frozenset[str] = frozenset()


def admit_support_projections(proj: Mapping[str, Any], support: str) -> list[str]:
    if not is_exact_dict(proj):
        return ["support_proj_not_dict"]
    reasons: list[str] = []
    for arm in ("package", "out"):
        if arm not in proj or not is_exact_dict(proj[arm]):
            reasons.append(f"missing_arm:{arm}")
            continue
        for h in (10, 20, 50):
            if h not in proj[arm]:
                reasons.append(f"missing_horizon:{arm}:{h}")
                continue
            reasons.extend(
                f"{arm}:{h}:{r}" for r in admit_horizon_view(proj[arm][h], support=support)
            )
    return reasons


def check_identity(projections: Mapping[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    """Universe + cross-receipt source-bucket map equality (all six arm×horizon views)."""
    reasons: list[str] = []
    raw: dict[str, Any] = {}
    if not is_exact_dict(projections):
        return "IDENTITY_BIND_FAIL", ["projections_not_dict"], raw
    for support in SUPPORTS:
        if support not in projections:
            reasons.append(f"missing_support:{support}")
            continue
        sp = projections[support]
        rs = admit_support_projections(sp, support)
        if rs:
            reasons.extend(f"{support}:{r}" for r in rs)
            continue
        ref_u = extract_universe(sp["package"][50])
        ref_set = set(ref_u)
        try:
            ref_bmap = extract_bucket_map(sp["package"][50])
        except ValueError as e:
            reasons.append(f"{support}:package:50:bucket_map:{e}")
            continue
        raw[support] = {"universe_n": len(ref_u)}
        for arm in ("package", "out"):
            for h in (10, 20, 50):
                u = extract_universe(sp[arm][h])
                if set(u) != ref_set or len(u) != len(ref_u):
                    reasons.append(f"{support}:{arm}:{h}:universe_ne_package_N50")
                try:
                    bmap = extract_bucket_map(sp[arm][h])
                except ValueError as e:
                    reasons.append(f"{support}:{arm}:{h}:bucket_map:{e}")
                    continue
                # Exact cross-receipt bucket-map equality vs package-N50 reference.
                if bmap != ref_bmap:
                    reasons.append(f"{support}:{arm}:{h}:bucket_map_ne_reference")
    if reasons:
        return "IDENTITY_BIND_FAIL", reasons, raw
    return "IDENTITY_OK", [], raw


def d1_per_support(
    universe: list[str], bucket_map: Mapping[str, str], survivors: set[str]
) -> tuple[str, dict[str, Any]]:
    S = len(survivors)
    raw: dict[str, Any] = {
        "S_s_total_support_survivors": S,
        "branch_denominator": "TOTAL_PACKAGE_N50_SUPPORT_SURVIVORS",
    }
    if S == 0:
        return "DEGENERATE_NO_SURVIVORS", raw
    bucket_rows: dict[str, int] = {}
    bucket_surv: dict[str, int] = {}
    for rid in universe:
        b = bucket_map[rid]
        bucket_rows[b] = bucket_rows.get(b, 0) + 1
        if rid in survivors:
            bucket_surv[b] = bucket_surv.get(b, 0) + 1
    eligible = [b for b, n in bucket_rows.items() if n >= MIN_BUCKET_DENOMINATOR_ROWS]
    ranked = sorted(eligible, key=lambda b: (-bucket_surv.get(b, 0), b))
    top1_b = ranked[0] if ranked else None
    top2_b = ranked[1] if len(ranked) > 1 else None
    top1 = bucket_surv.get(top1_b, 0) if top1_b is not None else 0
    top2_only = bucket_surv.get(top2_b, 0) if top2_b is not None else 0
    top2_sum = top1 + top2_only if top1_b is not None else 0
    S_elig = sum(bucket_surv.get(b, 0) for b in eligible)
    n_u = len(universe)
    enr_b, dep_b, enr_n, dep_n = [], [], 0, 0
    for b in eligible:
        bs, br = bucket_surv.get(b, 0), bucket_rows[b]
        if enrichment_ge_1_5(bs, br, S, n_u):
            enr_b.append(b)
            enr_n += bs
        if enrichment_le_0_5(bs, br, S, n_u):
            dep_b.append(b)
            dep_n += bs
    raw.update(
        {
            "S_eligible_sum_report_only": S_elig,
            "top1_bucket": top1_b,
            "top1_survivors": top1,
            "top2_bucket": top2_b,
            "top2_survivors": top2_only,
            "top2_sum": top2_sum,
            "enriched_survivors_eligible_scope": enr_n,
            "depleted_survivors_eligible_scope": dep_n,
            "ineligible_bucket_survivors_sum_report_only": S - S_elig,
            "enriched_buckets": list(enr_b),
            "depleted_buckets": list(dep_b),
            "enriched_is_top1": bool(top1_b in enr_b) if top1_b else False,
            "enriched_in_top2": bool(
                any(b in enr_b for b in (top1_b, top2_b) if b is not None)
            ),
            "eligible_ranking": [
                {"bucket": b, "survivors": bucket_surv.get(b, 0), "rows": bucket_rows[b]}
                for b in ranked
            ],
        }
    )
    if top1 * 2 >= S:
        return "HEAD1_MAJORITY", raw
    if top2_sum * 2 >= S:
        return "HEAD2_MAJORITY", raw
    if top2_sum * 3 >= S:
        return "HEAD2_THIRD", raw
    return "HEAD_DIFFUSE", raw


def d1_composite(a: str, b: str) -> str:
    if a == "DEGENERATE_NO_SURVIVORS" or b == "DEGENERATE_NO_SURVIVORS":
        return "DEGENERATE_NO_SURVIVORS"
    return a if a == b else "SPLIT_SUPPORTS"


def d2_per_support(
    r50: set[str],
    bucket_map: Mapping[str, str],
    enriched_buckets: list[str],
    pkg20: set[str],
    pkg10: set[str],
) -> tuple[str, dict[str, Any]]:
    enr = set(enriched_buckets)
    e50 = {rid for rid in r50 if bucket_map.get(rid) in enr}
    n = len(e50)
    present20, absent20, present10 = len(e50 & pkg20), len(e50 - pkg20), len(e50 & pkg10)
    thr = ceil_0_70(n) if n > 0 else 0
    raw = {
        "|E50|_row_ids": n,
        "present_at_package_N20_row_id_intersection": present20,
        "absent_from_package_N20_row_id_difference": absent20,
        "present_at_package_N10_row_id_intersection": present10,
        "ceil_0_70_|E50|": thr,
        "membership_carrier": "row_id",
        "enriched_buckets": list(enriched_buckets),
    }
    if n == 0:
        return "E_EMPTY", raw
    if present20 >= thr:
        return "E_PERSISTENT", raw
    if absent20 >= thr:
        return "E_TRANSIENT", raw
    return "E_MIXED", raw


def d2_composite(a: str, b: str) -> str:
    if a == "E_EMPTY" or b == "E_EMPTY":
        return "E_EMPTY" if a == b == "E_EMPTY" else "SPLIT_SUPPORTS"
    return a if a == b else "SPLIT_SUPPORTS"


def d3_per_support(r50: set[str], o50: set[str]) -> tuple[str, dict[str, Any]]:
    inter = len(r50 & o50)
    raw = {
        "|R50|_row_ids": len(r50),
        "|O50|_row_ids": len(o50),
        "inter_row_id_intersection": inter,
        "only_package_row_id_difference": len(r50 - o50),
        "only_out_row_id_difference": len(o50 - r50),
        "membership_carrier": "row_id",
    }
    if not r50:
        return "DEGENERATE_NO_SURVIVORS", raw
    if inter * 2 >= len(r50):
        return "CO_MAJORITY", raw
    if inter > 0:
        return "CO_PARTIAL", raw
    return "CO_DISJOINT", raw


def d3_composite(a: str, b: str) -> str:
    if a == "DEGENERATE_NO_SURVIVORS" or b == "DEGENERATE_NO_SURVIVORS":
        return "DEGENERATE_NO_SURVIVORS"
    return a if a == b else "SPLIT_SUPPORTS"


def _ed_from_raw(raw: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    enr, dep = [], []
    per = raw.get("per_bucket") or {}
    if not is_exact_dict(per):
        return enr, dep
    for b, info in per.items():
        if is_exact_dict(info) and info.get("eligible") is True:
            if info.get("enrichment_ge_1_5") is True:
                enr.append(b)
            if info.get("enrichment_le_0_5") is True:
                dep.append(b)
    return sorted(enr), sorted(dep)


def check_authority(
    projections: Mapping[str, Any], terminal: Mapping[str, Any] | None
) -> tuple[str, list[str], dict[str, Any]]:
    raw: dict[str, Any] = {"recomputed": {}, "terminal_bind": {}}
    if terminal is None:
        raw["authority_mode"] = "no_terminal_fixture"
        return "AUTHORITY_OK", [], raw
    if not is_exact_dict(terminal):
        return "AUTHORITY_BIND_FAIL", ["terminal_not_dict"], raw
    rbp = terminal.get("residual_bucket_profile")
    if not is_exact_dict(rbp):
        return "AUTHORITY_BIND_FAIL", ["missing_residual_bucket_profile"], raw
    t_ps, t_raw, t_comp = rbp.get("per_support"), rbp.get("raw"), rbp.get("composite")
    if not is_exact_dict(t_ps) or not is_exact_dict(t_raw):
        return "AUTHORITY_BIND_FAIL", ["terminal_residual_shape"], raw
    reasons: list[str] = []
    if terminal.get("composite_terminal") != "IDENTITY_OK__CHURNED__TRANSIENT__STRATIFIED":
        reasons.append("composite_terminal_ne_expected")
    if terminal.get("successor") != "support-split residual densify; no mechanism mint":
        reasons.append("successor_ne_expected")
    if t_comp != "STRATIFIED":
        reasons.append(f"residual_composite_ne_STRATIFIED:{t_comp!r}")
    re_labels, re_raws = {}, {}
    for support in SUPPORTS:
        view = projections[support]["package"][50]
        u, bm, sv = extract_universe(view), extract_bucket_map(view), extract_survivors(view)
        lab, rraw = residual_per_support_label(
            u, bm, sv, arm=AUTHORITY_ARM, horizon=AUTHORITY_HORIZON
        )
        re_labels[support], re_raws[support] = lab, rraw
        if t_ps.get(support) != lab:
            reasons.append(f"{support}:label_ne_terminal:{lab}!={t_ps.get(support)}")
        tenr, tdep = _ed_from_raw(t_raw.get(support) or {})
        renr, rdep = _ed_from_raw(rraw)
        if tenr != renr or tdep != rdep:
            reasons.append(f"{support}:enriched_depleted_ne_terminal")
    re_comp = residual_composite(re_labels["L0b"], re_labels["math_a0"])
    raw["recomputed"] = {"per_support": re_labels, "composite": re_comp, "raw": re_raws}
    raw["terminal_bind"] = {"per_support": dict(t_ps), "composite": t_comp}
    if re_comp != t_comp:
        reasons.append(f"residual_composite_recompute_ne:{re_comp}!={t_comp}")
    return ("AUTHORITY_OK", [], raw) if not reasons else ("AUTHORITY_BIND_FAIL", reasons, raw)


def lookup_successor(identity: str, d1: str, d2: str, d3: str) -> str:
    if identity == "INSTRUMENT_OR_BIND_FAIL":
        return T["instrument"]
    if identity == "IDENTITY_BIND_FAIL":
        return T["identity"]
    if identity == "AUTHORITY_BIND_FAIL":
        return T["authority"]
    if d1 == "SPLIT_SUPPORTS" or d2 == "SPLIT_SUPPORTS" or d3 == "SPLIT_SUPPORTS":
        return T["split"]
    if d1 in ("HEAD1_MAJORITY", "HEAD2_MAJORITY") and d3 == "CO_DISJOINT":
        return T["step_5"]
    if d2 == "E_TRANSIENT" and d3 == "CO_DISJOINT":
        return T["step_6"]
    if d2 == "E_PERSISTENT" and d3 == "CO_MAJORITY":
        return T["step_7"]
    return T["default"]


def densify_from_projections(
    projections: Mapping[str, Any],
    *,
    authority_terminal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity, id_reasons, id_raw = check_identity(projections)
    claim = dict(REQUIRED_CLAIM_BOUNDARY)

    def _fail(kind: str, reasons: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
        out = {
            "schema": SCHEMA_ID,
            "product": PRODUCT,
            "identity_profile": kind,
            "identity_reasons": list(reasons),
            "identity_raw": id_raw,
            "D1_profile": None,
            "D2_profile": None,
            "D3_profile": None,
            "composite_terminal": kind,
            "successor": lookup_successor(kind, "HEAD_DIFFUSE", "E_MIXED", "CO_DISJOINT"),
            "claim_boundary": claim,
            "instrument_fail": kind == "INSTRUMENT_OR_BIND_FAIL",
        }
        if extra:
            out.update(extra)
        return out

    if identity != "IDENTITY_OK":
        return _fail("IDENTITY_BIND_FAIL", id_reasons)
    auth_status, auth_reasons, auth_raw = check_authority(projections, authority_terminal)
    if auth_status != "AUTHORITY_OK":
        return _fail("AUTHORITY_BIND_FAIL", auth_reasons, {"authority_raw": auth_raw})

    d1_ps, d1_raw, d2_ps, d2_raw, d3_ps, d3_raw = {}, {}, {}, {}, {}, {}
    report_only: dict[str, Any] = {}
    for support in SUPPORTS:
        sp = projections[support]
        u = extract_universe(sp["package"][50])
        bmap = extract_bucket_map(sp["package"][50])
        r50 = extract_survivors(sp["package"][50])
        pkg20 = extract_survivors(sp["package"][20])
        pkg10 = extract_survivors(sp["package"][10])
        o50 = extract_survivors(sp["out"][50])
        o20 = extract_survivors(sp["out"][20])
        lab1, raw1 = d1_per_support(u, bmap, r50)
        d1_ps[support], d1_raw[support] = lab1, raw1
        enr_b = list(raw1.get("enriched_buckets") or [])
        dep_b = list(raw1.get("depleted_buckets") or [])
        lab2, raw2 = d2_per_support(r50, bmap, enr_b, pkg20, pkg10)
        d2_ps[support], d2_raw[support] = lab2, raw2
        dep_set = set(dep_b)
        e_dep = {rid for rid in r50 if bmap.get(rid) in dep_set}
        report_only[support] = {
            "depleted_stratum_package": {
                "|D50|": len(e_dep),
                "present_at_package_N20": len(e_dep & pkg20),
                "absent_from_package_N20": len(e_dep - pkg20),
                "present_at_package_N10": len(e_dep & pkg10),
                "absent_from_package_N10": len(e_dep - pkg10),
            },
            "present_at_package_N10_enriched": raw2[
                "present_at_package_N10_row_id_intersection"
            ],
            "out_enriched_N50": {
                "|E_out|": len({rid for rid in o50 if bmap.get(rid) in set(enr_b)})
            },
            "cosurvival_package_N20_out_N20": {
                "|R20|_row_ids": len(pkg20),
                "|O20|_row_ids": len(o20),
                "inter_row_id_intersection": len(pkg20 & o20),
                "only_package_row_id_difference": len(pkg20 - o20),
                "only_out_row_id_difference": len(o20 - pkg20),
            },
        }
        lab3, raw3 = d3_per_support(r50, o50)
        d3_ps[support], d3_raw[support] = lab3, raw3
        report_only[support]["cosurvival_enriched_only"] = len(
            {rid for rid in (r50 & o50) if bmap.get(rid) in set(enr_b)}
        )
        report_only[support]["cosurvival_depleted_only"] = len(
            {rid for rid in (r50 & o50) if bmap.get(rid) in dep_set}
        )

    d1c = d1_composite(d1_ps["L0b"], d1_ps["math_a0"])
    d2c = d2_composite(d2_ps["L0b"], d2_ps["math_a0"])
    d3c = d3_composite(d3_ps["L0b"], d3_ps["math_a0"])
    return {
        "schema": SCHEMA_ID,
        "product": PRODUCT,
        "identity_profile": "IDENTITY_OK",
        "identity_reasons": [],
        "identity_raw": id_raw,
        "authority_profile": "AUTHORITY_OK",
        "authority_reasons": [],
        "authority_raw": auth_raw,
        "D1_profile": {
            "composite": d1c,
            "per_support": d1_ps,
            "raw": d1_raw,
            "branch_denominator": "TOTAL_PACKAGE_N50_SUPPORT_SURVIVORS",
        },
        "D2_profile": {
            "composite": d2c,
            "per_support": d2_ps,
            "raw": d2_raw,
            "membership_carrier": "row_id",
        },
        "D3_profile": {
            "composite": d3c,
            "per_support": d3_ps,
            "raw": d3_raw,
            "membership_carrier": "row_id",
        },
        "report_only": report_only,
        "composite_terminal": f"IDENTITY_OK__{d1c}__{d2c}__{d3c}",
        "successor": lookup_successor("IDENTITY_OK", d1c, d2c, d3c),
        "claim_boundary": claim,
        "instrument_fail": False,
    }


def densify_core(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        k: copy.deepcopy(v)
        for k, v in result.items()
        if k not in NON_AUTHORITATIVE_KEYS
    }
