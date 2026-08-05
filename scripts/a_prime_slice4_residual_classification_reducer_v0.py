"""Rung-3 residual classification pure reducer (STEP-1). reducer→schema; no IO."""
from __future__ import annotations

import copy
from typing import Any, Mapping

from scripts.a_prime_slice4_residual_classification_schema_v0 import (
    ARMS,
    CL_LABELS,
    EXPECTED_CARDINALITY,
    HORIZONS,
    MIN_BUCKET_DENOMINATOR_ROWS,
    NINE_CELL_TABLE,
    OVERLAP_COMPOSITE,
    OVERLAP_PER_SUPPORT,
    PREEMPTING_ONLY,
    Q3_BRANCH_ARM,
    Q3_BRANCH_HORIZON,
    REQUIRED_CLAIM_BOUNDARY,
    RESCUE_COMPOSITE,
    RESCUE_PER_SUPPORT,
    RESIDUAL_COMPOSITE,
    RESIDUAL_PER_SUPPORT,
    SUPPORTS,
    admit_horizon_view,
    ceil_0_70,
    coverage_buckets_ok,
    coverage_rows_ok,
    enrichment_ge_1_5,
    enrichment_le_0_5,
    extract_bucket_map,
    extract_survivors,
    extract_universe,
    is_exact_dict,
    is_exact_int,
    is_exact_list,
    is_exact_str,
    j_ge_0_8,
    j_le_0_3,
)

def jaccard_raw(a: set[str], b: set[str]) -> dict[str, int]:
    inter = a & b
    union = a | b
    return {
        "intersect": len(inter),
        "a": len(a),
        "b": len(b),
        "union": len(union),
    }

def _admit_all(projections: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not is_exact_dict(projections):
        return ["projections_not_dict"]
    for support in SUPPORTS:
        if support not in projections or not is_exact_dict(projections[support]):
            reasons.append(f"missing_support:{support}")
            continue
        for arm in ARMS:
            if arm not in projections[support] or not is_exact_dict(projections[support][arm]):
                reasons.append(f"missing_arm:{support}:{arm}")
                continue
            for h in HORIZONS:
                if h not in projections[support][arm]:
                    reasons.append(f"missing_horizon:{support}:{arm}:{h}")
                    continue
                r = admit_horizon_view(projections[support][arm][h], support=support)
                for x in r:
                    reasons.append(f"{support}:{arm}:{h}:{x}")
    return reasons

def check_identity(projections: Mapping[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    """Return (IDENTITY_OK|IDENTITY_BIND_FAIL, reasons, raw)."""
    reasons = _admit_all(projections)
    raw: dict[str, Any] = {"per_support": {}}
    if reasons:
        return "IDENTITY_BIND_FAIL", reasons, raw

    for support in SUPPORTS:
        universes: dict[str, set[str]] = {}
        bucket_maps: dict[str, dict[str, str]] = {}
        for arm in ARMS:
            for h in HORIZONS:
                key = f"{arm}:N{h}"
                view = projections[support][arm][h]
                universes[key] = set(extract_universe(view))
                bucket_maps[key] = extract_bucket_map(view)
        # all six universes equal
        keys = list(universes.keys())
        base = universes[keys[0]]
        for k in keys[1:]:
            if universes[k] != base:
                reasons.append(f"universe_mismatch:{support}:{keys[0]}!={k}")
        # cross-receipt bucket map invariance
        for rid in base:
            buckets = {k: bucket_maps[k].get(rid) for k in keys}
            vals = set(buckets.values())
            if len(vals) != 1 or None in vals:
                reasons.append(f"bucket_map_drift:{support}:{rid}:{buckets}")
        raw["per_support"][support] = {
            "universe_size": len(base),
            "expected": EXPECTED_CARDINALITY[support],
        }
        if len(base) != EXPECTED_CARDINALITY[support]:
            reasons.append(
                f"cardinality:{support}:{len(base)}!={EXPECTED_CARDINALITY[support]}"
            )
    if reasons:
        return "IDENTITY_BIND_FAIL", reasons, raw
    return "IDENTITY_OK", [], raw

def overlap_per_support_label(
    j20: dict[str, int], j50: dict[str, int]
) -> tuple[str, dict[str, Any]]:
    """Per-support overlap label from Jaccard raw counts at N20 and N50."""
    raw = {"N20": j20, "N50": j50}
    if j20["union"] == 0 or j50["union"] == 0:
        return "DEGENERATE_EMPTY", raw
    stable = j_ge_0_8(j20["intersect"], j20["union"]) and j_ge_0_8(
        j50["intersect"], j50["union"]
    )
    churned = j_le_0_3(j20["intersect"], j20["union"]) and j_le_0_3(
        j50["intersect"], j50["union"]
    )
    if stable:
        return "STABLE_CORE", raw
    if churned:
        return "CHURNED", raw
    return "PARTIAL_CORE", raw

def overlap_composite(l0b: str, math: str) -> str:
    if l0b == "DEGENERATE_EMPTY" or math == "DEGENERATE_EMPTY":
        return "DEGENERATE_EMPTY"
    key = (l0b, math)
    if key not in NINE_CELL_TABLE:
        raise ValueError(f"overlap_composite_unknown_pair:{key}")
    return NINE_CELL_TABLE[key]

def rescue_per_support_label(
    gross: set[str],
    survivors_pkg_n20: set[str],
    survivors_pkg_n50: set[str],
) -> tuple[str, dict[str, Any]]:
    n = len(gross)
    thr = ceil_0_70(n)
    absent_n20 = gross - survivors_pkg_n20
    present_n50 = gross & survivors_pkg_n50
    raw = {
        "gross": n,
        "threshold_ceil_0_70": thr,
        "absent_from_package_N20": len(absent_n20),
        "present_at_package_N50": len(present_n50),
    }
    if n == 0:
        raw["empty_gross"] = True
        return "MIXED", raw
    transient = len(absent_n20) >= thr
    persistent = len(present_n50) >= thr
    if transient and persistent:
        return "NONMONOTONE_RESCUE", raw
    if transient:
        return "TRANSIENT", raw
    if persistent:
        return "PERSISTENT", raw
    return "MIXED", raw

def rescue_composite(l0b: str, math: str) -> str:
    if l0b == math:
        return l0b
    return "SPLIT_SUPPORTS"

def counter_loss_label(
    counter_loss: set[str],
    survivors_out_n20: set[str],
    survivors_out_n50: set[str],
) -> tuple[str, dict[str, Any]]:
    """OUT-arm CL_* table; NEVER branch input."""
    n = len(counter_loss)
    thr = ceil_0_70(n)
    gone_n20 = counter_loss - survivors_out_n20
    still_n50 = counter_loss & survivors_out_n50
    raw = {
        "counter_loss": n,
        "threshold_ceil_0_70": thr,
        "absent_from_out_N20": len(gone_n20),
        "present_at_out_N50": len(still_n50),
        "branch_input": False,
        "measured_in_arm": "out",
    }
    if n == 0:
        raw["empty_counter_loss"] = True
        return "CL_MIXED", raw
    cl_t = len(gone_n20) >= thr
    cl_p = len(still_n50) >= thr
    if cl_t and cl_p:
        return "CL_NONMONOTONE", raw
    if cl_t:
        return "CL_TRANSIENT", raw
    if cl_p:
        return "CL_PERSISTENT", raw
    return "CL_MIXED", raw

def residual_per_support_label(
    universe: list[str],
    bucket_map: Mapping[str, str],
    survivors: set[str],
    *,
    arm: str,
    horizon: int,
) -> tuple[str, dict[str, Any]]:
    """Q3 residual label for one support surface; raw stamps the examined arm/horizon."""
    if not is_exact_str(arm) or not is_exact_int(horizon):
        raise ValueError(f"residual_surface_types:{arm!r}:{horizon!r}")
    support_rows = len(universe)
    support_surv = len(survivors)
    raw: dict[str, Any] = {
        "support_rows": support_rows,
        "support_survivors": support_surv,
        "arm": arm,
        "horizon": horizon,
    }
    if support_surv == 0:
        return "DEGENERATE_NO_SURVIVORS", raw

    bucket_rows: dict[str, int] = {}
    bucket_surv: dict[str, int] = {}
    for rid in universe:
        b = bucket_map.get(rid)
        if b is None or not is_exact_str(b) or b == "":
            raw["metadata_reason"] = "missing_bucket"
            return "METADATA_INSUFFICIENT", raw
        bucket_rows[b] = bucket_rows.get(b, 0) + 1
        if rid in survivors:
            bucket_surv[b] = bucket_surv.get(b, 0) + 1

    nonempty = [b for b, n in bucket_rows.items() if n > 0]
    eligible = [b for b in nonempty if bucket_rows[b] >= MIN_BUCKET_DENOMINATOR_ROWS]
    eligible_rows = sum(bucket_rows[b] for b in eligible)
    raw["eligible_rows"] = eligible_rows
    raw["eligible_buckets"] = len(eligible)
    raw["all_nonempty_buckets"] = len(nonempty)
    raw["coverage_rows_ok"] = coverage_rows_ok(eligible_rows, support_rows) if support_rows else False
    raw["coverage_buckets_ok"] = (
        coverage_buckets_ok(len(eligible), len(nonempty)) if nonempty else False
    )
    raw["per_bucket"] = {
        b: {
            "bucket_rows": bucket_rows[b],
            "bucket_survivors": bucket_surv.get(b, 0),
            "eligible": b in eligible,
        }
        for b in nonempty
    }

    if not nonempty or not eligible:
        raw["metadata_reason"] = "zero_eligible_buckets"
        return "METADATA_INSUFFICIENT", raw
    if not raw["coverage_rows_ok"] or not raw["coverage_buckets_ok"]:
        raw["metadata_reason"] = "coverage_below_min"
        return "METADATA_INSUFFICIENT", raw

    enriches = []
    for b in eligible:
        bs = bucket_surv.get(b, 0)
        br = bucket_rows[b]
        ge = enrichment_ge_1_5(bs, br, support_surv, support_rows)
        le = enrichment_le_0_5(bs, br, support_surv, support_rows)
        enriches.append((b, ge, le, bs, br))
        raw["per_bucket"][b]["enrichment_ge_1_5"] = ge
        raw["per_bucket"][b]["enrichment_le_0_5"] = le
    if any(ge or le for _, ge, le, _, _ in enriches):
        return "STRATIFIED", raw
    # all inside (1/2, 3/2): not ge and not le (any false ⇒ all true)
    return "UNIFORM", raw

def residual_composite(l0b: str, math: str) -> str:
    if l0b == "METADATA_INSUFFICIENT" or math == "METADATA_INSUFFICIENT":
        return "METADATA_INSUFFICIENT"
    if l0b == "DEGENERATE_NO_SURVIVORS" or math == "DEGENERATE_NO_SURVIVORS":
        return "DEGENERATE_NO_SURVIVORS"
    if l0b == "STRATIFIED" and math == "STRATIFIED":
        return "STRATIFIED"
    if l0b == "UNIFORM" and math == "UNIFORM":
        return "UNIFORM"
    return "SPLIT_SUPPORTS"

def lookup_successor(
    identity: str,
    overlap: str,
    rescue: str,
    residual: str,
) -> str:
    """First-match-wins successor mapping (plan v6)."""
    if identity == "INSTRUMENT_OR_BIND_FAIL":
        return "instrument repair only; no science successor"
    if identity == "IDENTITY_BIND_FAIL":
        return "receipt/field schema repair; no residual science"
    if overlap == "SPLIT_SUPPORTS" or rescue == "SPLIT_SUPPORTS" or residual == "SPLIT_SUPPORTS":
        return (
            "support-split residual densify (each support routed separately "
            "on its own per-support labels, both reported); no mechanism mint"
        )
    if residual == "METADATA_INSUFFICIENT":
        return "bucket metadata repair or drop Q3; no mechanism mint"
    if residual == "DEGENERATE_NO_SURVIVORS" or overlap == "DEGENERATE_EMPTY":
        return "classify residual; no mechanism mint"
    if overlap == "STABLE_CORE" and rescue in ("TRANSIENT", "MIXED"):
        return "densify residual OUT-stable core rows (Rung-1-style densify); no mechanism mint"
    if overlap == "CHURNED":
        return "support-split residual densify; no mechanism mint"
    if rescue == "NONMONOTONE_RESCUE":
        return "multi-horizon rescue map densify; no mechanism mint"
    if residual == "STRATIFIED":
        return "bucket-targeted densify on enriched residual strata; no mechanism mint"
    if residual == "UNIFORM" and overlap == "PARTIAL_CORE":
        return "broad residual densify; no mechanism mint"
    return "classify residual; no mechanism mint"

def classify_from_projections(projections: Mapping[str, Any]) -> dict[str, Any]:
    """Main pure classifier. projections[support][arm][horizon] = horizon view dict."""
    identity, id_reasons, id_raw = check_identity(projections)
    if identity == "IDENTITY_BIND_FAIL":
        return {
            "identity_profile": identity,
            "identity_reasons": id_reasons,
            "identity_raw": id_raw,
            "survivor_overlap_profile": None,
            "rescue_persistence_profile": None,
            "residual_bucket_profile": None,
            "counter_loss_table": None,
            "composite_terminal": "IDENTITY_BIND_FAIL",
            "successor": lookup_successor(identity, "", "", ""),
            "claim_boundary": dict(REQUIRED_CLAIM_BOUNDARY),
            "instrument_fail": False,
        }

    # survivors per support/arm/horizon
    surv: dict[str, dict[str, dict[int, set[str]]]] = {}
    bucket_maps: dict[str, dict[str, dict[int, dict[str, str]]]] = {}
    universes: dict[str, dict[str, dict[int, list[str]]]] = {}
    for support in SUPPORTS:
        surv[support] = {}
        bucket_maps[support] = {}
        universes[support] = {}
        for arm in ARMS:
            surv[support][arm] = {}
            bucket_maps[support][arm] = {}
            universes[support][arm] = {}
            for h in HORIZONS:
                view = projections[support][arm][h]
                surv[support][arm][h] = extract_survivors(view)
                bucket_maps[support][arm][h] = extract_bucket_map(view)
                universes[support][arm][h] = extract_universe(view)

    # Q1 overlap
    per_overlap: dict[str, str] = {}
    overlap_raw: dict[str, Any] = {}
    for support in SUPPORTS:
        j20 = jaccard_raw(surv[support]["package"][20], surv[support]["out"][20])
        j50 = jaccard_raw(surv[support]["package"][50], surv[support]["out"][50])
        # also report N10
        j10 = jaccard_raw(surv[support]["package"][10], surv[support]["out"][10])
        label, raw = overlap_per_support_label(j20, j50)
        raw["N10"] = j10
        per_overlap[support] = label
        overlap_raw[support] = raw
        if label not in OVERLAP_PER_SUPPORT:
            raise ValueError(f"bad_overlap_label:{label}")
    o_comp = overlap_composite(per_overlap["L0b"], per_overlap["math_a0"])
    if o_comp not in OVERLAP_COMPOSITE:
        raise ValueError(f"bad_overlap_composite:{o_comp}")

    # Q2 rescue (gross in PACKAGE; per support dens only)
    per_rescue: dict[str, str] = {}
    rescue_raw: dict[str, Any] = {}
    cl_table: dict[str, Any] = {}
    for support in SUPPORTS:
        gross = surv[support]["package"][10] - surv[support]["out"][10]
        counter = surv[support]["out"][10] - surv[support]["package"][10]
        rlab, rraw = rescue_per_support_label(
            gross, surv[support]["package"][20], surv[support]["package"][50]
        )
        rraw["gross_set_size"] = len(gross)
        # never use cross-support total as dens — already per-support
        per_rescue[support] = rlab
        rescue_raw[support] = rraw
        clab, craw = counter_loss_label(
            counter, surv[support]["out"][20], surv[support]["out"][50]
        )
        if clab not in CL_LABELS:
            raise ValueError(f"bad_cl_label:{clab}")
        cl_table[support] = {"label": clab, "raw": craw}
        if rlab not in RESCUE_PER_SUPPORT:
            raise ValueError(f"bad_rescue_label:{rlab}")
    r_comp = rescue_composite(per_rescue["L0b"], per_rescue["math_a0"])
    if r_comp not in RESCUE_COMPOSITE:
        raise ValueError(f"bad_rescue_composite:{r_comp}")

    # Q3 residual — PACKAGE N50 ONLY for branch
    per_residual: dict[str, str] = {}
    residual_raw: dict[str, Any] = {}
    report_only: dict[str, Any] = {}
    for support in SUPPORTS:
        lab, raw = residual_per_support_label(
            universes[support][Q3_BRANCH_ARM][Q3_BRANCH_HORIZON],
            bucket_maps[support][Q3_BRANCH_ARM][Q3_BRANCH_HORIZON],
            surv[support][Q3_BRANCH_ARM][Q3_BRANCH_HORIZON],
            arm=Q3_BRANCH_ARM,
            horizon=Q3_BRANCH_HORIZON,
        )
        per_residual[support] = lab
        residual_raw[support] = raw
        # report-only: PACKAGE N20 + OUT all horizons enrichment tables
        ro: dict[str, Any] = {}
        for arm, h in (("package", 20), ("out", 10), ("out", 20), ("out", 50)):
            rlab, rraw = residual_per_support_label(
                universes[support][arm][h],
                bucket_maps[support][arm][h],
                surv[support][arm][h],
                arm=arm,
                horizon=h,
            )
            ro[f"{arm}:N{h}"] = {"label_report_only": rlab, "raw": rraw}
        report_only[support] = ro
        if lab not in RESIDUAL_PER_SUPPORT:
            raise ValueError(f"bad_residual_label:{lab}")
    res_comp = residual_composite(per_residual["L0b"], per_residual["math_a0"])
    if res_comp not in RESIDUAL_COMPOSITE:
        raise ValueError(f"bad_residual_composite:{res_comp}")

    terminal = f"IDENTITY_OK__{o_comp}__{r_comp}__{res_comp}"
    successor = lookup_successor("IDENTITY_OK", o_comp, r_comp, res_comp)

    return {
        "identity_profile": "IDENTITY_OK",
        "identity_reasons": [],
        "identity_raw": id_raw,
        "survivor_overlap_profile": {
            "composite": o_comp,
            "per_support": per_overlap,
            "raw": overlap_raw,
        },
        "rescue_persistence_profile": {
            "composite": r_comp,
            "per_support": per_rescue,
            "raw": rescue_raw,
        },
        "residual_bucket_profile": {
            "composite": res_comp,
            "per_support": per_residual,
            "raw": residual_raw,
            "branch_input_surface": {
                "arm": Q3_BRANCH_ARM,
                "horizon": Q3_BRANCH_HORIZON,
                "exclusive": True,
            },
            "report_only": report_only,
        },
        "counter_loss_table": cl_table,
        "composite_terminal": terminal,
        "successor": successor,
        "claim_boundary": dict(REQUIRED_CLAIM_BOUNDARY),
        "instrument_fail": False,
    }

# Empty by design: every classify_from_projections top-level key is authoritative.
NON_AUTHORITATIVE_KEYS: frozenset[str] = frozenset()


def classification_core(result: Mapping[str, Any]) -> dict[str, Any]:
    """Deep snapshot of all authoritative emitted keys (incl. identity_reasons/raw)."""
    return {
        k: copy.deepcopy(v)
        for k, v in result.items()
        if k not in NON_AUTHORITATIVE_KEYS
    }
