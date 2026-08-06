"""Rung-5 shared-component pure reducer. IMPORT densify/residual; no IO/CLI.
PLAN v4 a2e7420aeaee715ed181b46f4f1de4d0b93deb47a29da6e3bded0fd431e48421 REIMPLEMENTATION of residual/D2 arithmetic FORBIDDEN.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

from scripts.a_prime_slice4_support_split_residual_densify_reducer_v0 import (
    check_authority as densify_check_authority,
    check_identity,
    d1_per_support,
    d2_per_support,
)
from scripts.a_prime_slice4_shared_component_decomposition_schema_v0 import (
    FROZEN_NEUTRAL_SUCCESSOR_TEXTS as T,
    MASS_HEAD_COMPARATOR_BUCKET,
    MASS_HEAD_COMPARATOR_SUPPORT,
    PRODUCT,
    REQUIRED_CLAIM_BOUNDARY,
    RUNG3_COMPOSITE_EXPECTED,
    RUNG4_COMPOSITE_EXPECTED,
    RUNG4_D2_COMPOSITE_EXPECTED,
    RUNG4_D2_PER_SUPPORT_EXPECTED,
    RUNG4_D2_RAW_COUNTS_EXPECTED,
    SCHEMA_ID,
    SHARED_ENRICHED_COMPONENTS,
    SUPPORTS,
    extract_bucket_map,
    extract_survivors,
    extract_universe,
    is_exact_dict,
)

NON_AUTHORITATIVE_KEYS: frozenset[str] = frozenset()

_COUNT_KEYS = (
    "|E50|_row_ids",
    "present_at_package_N20_row_id_intersection",
    "absent_from_package_N20_row_id_difference",
)

def c1_component_raw(
    r50: set[str],
    bucket_map: Mapping[str, str],
    component: str,
    pkg20: set[str],
    pkg10: set[str],
) -> tuple[str, dict[str, Any]]:
    """Per-component endpoint label via densify d2_per_support([component])."""
    lab, raw = d2_per_support(r50, bucket_map, [component], pkg20, pkg10)
    out = {
        "|B50|_row_ids": raw["|E50|_row_ids"],
        "present_at_package_N20_row_id_intersection": raw["present_at_package_N20_row_id_intersection"],
        "absent_from_package_N20_row_id_difference": raw["absent_from_package_N20_row_id_difference"],
        "present_at_package_N10_row_id_intersection": raw["present_at_package_N10_row_id_intersection"],
        "ceil_0_70_|B50|": raw["ceil_0_70_|E50|"],
        "component_label": lab,
        "membership_carrier": "row_id",
        "component": component,
    }
    return lab, out

def cross_support_alignment(
    labels: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Per-component label equality across supports (no row_id ops)."""
    out: dict[str, Any] = {}
    both = True
    for c in SHARED_ENRICHED_COMPONENTS:
        a = labels["L0b"][c]
        b = labels["math_a0"][c]
        aligned = a == b and a != "E_EMPTY" and b != "E_EMPTY"
        out[f"{c}_aligned"] = aligned
        out[f"{c}_labels"] = {"L0b": a, "math_a0": b}
        if not aligned:
            both = False
    out["both_components_aligned"] = both
    return out

def a1_secondary(
    r50: set[str],
    bucket_map: Mapping[str, str],
    pkg20: set[str],
    pkg10: set[str],
    math_component_labels: Mapping[str, str],
) -> tuple[str, dict[str, Any]]:
    lab, raw_d2 = d2_per_support(
        r50, bucket_map, [MASS_HEAD_COMPARATOR_BUCKET], pkg20, pkg10
    )
    raw = {
        "|B50_R1|": raw_d2["|E50|_row_ids"],
        "present_at_package_N20": raw_d2["present_at_package_N20_row_id_intersection"],
        "absent_from_package_N20": raw_d2["absent_from_package_N20_row_id_difference"],
        "present_at_package_N10": raw_d2["present_at_package_N10_row_id_intersection"],
        "ceil_0_70": raw_d2["ceil_0_70_|E50|"],
        "R1_label": lab,
    }
    r0 = math_component_labels.get("R0")
    r1b = math_component_labels.get("R1b4v2")
    if lab == r0 == r1b:
        sec = "R1_LABEL_EQ_R0_AND_R1b4v2"
    elif lab == r0 and lab != r1b:
        sec = "R1_LABEL_EQ_R0"
    elif lab == r1b and lab != r0:
        sec = "R1_LABEL_EQ_R1b4v2"
    else:
        sec = "R1_LABEL_EQ_NEITHER"
    return sec, raw

def b_l0b_row_table(projections: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Report-only L0b row membership; never a branch input."""
    sp = projections["L0b"]
    bmap = extract_bucket_map(sp["package"][50])
    r50 = extract_survivors(sp["package"][50])
    sets = {
        ("package", 10): extract_survivors(sp["package"][10]),
        ("package", 20): extract_survivors(sp["package"][20]),
        ("package", 50): r50,
        ("out", 10): extract_survivors(sp["out"][10]),
        ("out", 20): extract_survivors(sp["out"][20]),
        ("out", 50): extract_survivors(sp["out"][50]),
    }
    rows = []
    for rid in sorted(r50):
        entry: dict[str, Any] = {"row_id": rid, "source_bucket": bmap.get(rid)}
        for (arm, h), s in sets.items():
            entry[f"{arm}_N{h}"] = rid in s
        rows.append(entry)
    return rows

def check_rung4_authority(
    projections: Mapping[str, Any],
    rung4_terminal: Mapping[str, Any] | None,
) -> tuple[str, list[str], dict[str, Any]]:
    """Rung-4 pin + densify D2 recompute bind."""
    raw: dict[str, Any] = {"recomputed_d2": {}, "terminal_d2": {}}
    if rung4_terminal is None:
        raw["authority_mode"] = "no_rung4_terminal_fixture"
        return "AUTHORITY_OK", [], raw
    if not is_exact_dict(rung4_terminal):
        return "AUTHORITY_BIND_FAIL", ["rung4_terminal_not_dict"], raw
    reasons: list[str] = []
    if rung4_terminal.get("composite_terminal") != RUNG4_COMPOSITE_EXPECTED:
        reasons.append("rung4_composite_terminal_ne_expected")
    d2p = rung4_terminal.get("D2_profile")
    if not is_exact_dict(d2p):
        return "AUTHORITY_BIND_FAIL", ["rung4_missing_D2_profile"], raw
    if d2p.get("composite") != RUNG4_D2_COMPOSITE_EXPECTED:
        reasons.append(f"rung4_d2_composite_ne:{d2p.get('composite')!r}")
    t_ps = d2p.get("per_support")
    t_raw = d2p.get("raw")
    if not is_exact_dict(t_ps) or not is_exact_dict(t_raw):
        return "AUTHORITY_BIND_FAIL", ["rung4_d2_shape"], raw
    re_ps: dict[str, str] = {}
    re_raw: dict[str, Any] = {}
    for support in SUPPORTS:
        sp = projections[support]
        u = extract_universe(sp["package"][50])
        bmap = extract_bucket_map(sp["package"][50])
        r50 = extract_survivors(sp["package"][50])
        pkg20 = extract_survivors(sp["package"][20])
        pkg10 = extract_survivors(sp["package"][10])
        _lab1, raw1 = d1_per_support(u, bmap, r50)
        enr = list(raw1.get("enriched_buckets") or [])
        if enr != list(SHARED_ENRICHED_COMPONENTS):
            reasons.append(f"{support}:enriched_buckets_ne_pin:{enr!r}")
        lab2, raw2 = d2_per_support(r50, bmap, enr, pkg20, pkg10)
        re_ps[support], re_raw[support] = lab2, raw2
        if t_ps.get(support) != lab2:
            reasons.append(
                f"{support}:d2_label_ne_terminal:{lab2}!={t_ps.get(support)}"
            )
        if t_ps.get(support) != RUNG4_D2_PER_SUPPORT_EXPECTED[support]:
            reasons.append(
                f"{support}:d2_terminal_label_ne_pin:{t_ps.get(support)!r}"
            )
        exp = RUNG4_D2_RAW_COUNTS_EXPECTED[support]
        tr = t_raw.get(support) or {}
        if not is_exact_dict(tr):
            reasons.append(f"{support}:d2_terminal_raw_not_dict")
            continue
        for k, ev in exp.items():
            if tr.get(k) != ev:
                reasons.append(f"{support}:d2_terminal_raw_{k}_ne_pin")
        # whole-map equality: any field drift (incl. schema-grown keys) binds
        if tr != raw2:
            reasons.append(f"{support}:d2_terminal_raw_ne_recompute")
    raw["recomputed_d2"] = {"per_support": re_ps, "raw": re_raw}
    raw["terminal_d2"] = {"per_support": dict(t_ps), "raw": dict(t_raw)}
    return ("AUTHORITY_OK", [], raw) if not reasons else ("AUTHORITY_BIND_FAIL", reasons, raw)

def check_component_authority(
    projections: Mapping[str, Any],
    *,
    rung3_terminal: Mapping[str, Any] | None = None,
    rung4_terminal: Mapping[str, Any] | None = None,
) -> tuple[str, list[str], dict[str, Any]]:
    """Rung-3 residual + Rung-4 D2 authority bind."""
    s3, r3, raw3 = densify_check_authority(projections, rung3_terminal)
    raw: dict[str, Any] = {"rung3": raw3}
    if s3 != "AUTHORITY_OK":
        return s3, list(r3), raw
    if rung3_terminal is not None and is_exact_dict(rung3_terminal):
        if rung3_terminal.get("composite_terminal") != RUNG3_COMPOSITE_EXPECTED:
            return (
                "AUTHORITY_BIND_FAIL",
                ["rung3_composite_terminal_ne_expected"],
                raw,
            )
    s4, r4, raw4 = check_rung4_authority(projections, rung4_terminal)
    raw["rung4"] = raw4
    if s4 != "AUTHORITY_OK":
        return s4, list(r4), raw
    return "AUTHORITY_OK", [], raw

def recompose_check(
    component_raws: Mapping[str, Mapping[str, Mapping[str, Any]]],
    aggregate_d2_raw: Mapping[str, Mapping[str, Any]],
) -> tuple[str, list[str], dict[str, Any]]:
    """C2 recomposition: component sums == aggregate E50 raw."""
    reasons: list[str] = []
    raw: dict[str, Any] = {}
    for support in SUPPORTS:
        comps = component_raws[support]
        sums = {k: 0 for k in (
            "|E50|_row_ids",
            "present_at_package_N20_row_id_intersection",
            "absent_from_package_N20_row_id_difference",
            "present_at_package_N10_row_id_intersection",
        )}
        key_map = (
            ("|E50|_row_ids", "|B50|_row_ids"),
            ("present_at_package_N20_row_id_intersection", "present_at_package_N20_row_id_intersection"),
            ("absent_from_package_N20_row_id_difference", "absent_from_package_N20_row_id_difference"),
            ("present_at_package_N10_row_id_intersection", "present_at_package_N10_row_id_intersection"),
        )
        for c in SHARED_ENRICHED_COMPONENTS:
            cr = comps[c]
            for sk, ck in key_map:
                sums[sk] += int(cr[ck])
        agg = aggregate_d2_raw[support]
        raw[support] = {"component_sums": sums, "aggregate": dict(agg)}
        for k in _COUNT_KEYS:
            if sums[k] != int(agg[k]):
                reasons.append(f"{support}:recomp_{k}:{sums[k]}!={agg[k]}")
        if sums["present_at_package_N10_row_id_intersection"] != int(
            agg["present_at_package_N10_row_id_intersection"]
        ):
            reasons.append(f"{support}:recomp_present10")
    if reasons:
        return "RECOMPOSITION_BIND_FAIL", reasons, raw
    return "RECOMPOSITION_OK", [], raw

def lookup_successor(kind: str, primary: str | None = None) -> str:
    m = {
        "INSTRUMENT_OR_BIND_FAIL": "instrument",
        "IDENTITY_BIND_FAIL": "identity",
        "AUTHORITY_BIND_FAIL": "authority",
        "RECOMPOSITION_BIND_FAIL": "recomposition",
        "DEGENERATE_EMPTY_COMPONENT": "step_5",
    }
    if kind in m:
        return T[m[kind]]
    if primary == "ALIGNED_COMPONENT_LABELS__AGGREGATE_SPLIT":
        return T["step_6"]
    if primary == "COMPONENT_LABEL_SPLIT":
        return T["step_7"]
    return T["step_8"]

def component_from_projections(
    projections: Mapping[str, Any],
    *,
    rung3_terminal: Mapping[str, Any] | None = None,
    rung4_terminal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    claim = dict(REQUIRED_CLAIM_BOUNDARY)
    identity, id_reasons, id_raw = check_identity(projections)

    def _fail(kind: str, reasons: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
        out = {
            "schema": SCHEMA_ID,
            "product": PRODUCT,
            "identity_profile": "IDENTITY_BIND_FAIL" if kind == "IDENTITY_BIND_FAIL" else "IDENTITY_OK",
            "identity_reasons": list(id_reasons) if kind == "IDENTITY_BIND_FAIL" else [],
            "identity_raw": id_raw,
            "terminal_reasons": list(reasons),
            "terminal_kind": kind,
            "C1_profile": None,
            "C2_profile": None,
            "A1_secondary": None,
            "B_report_only": None,
            "composite_terminal": kind,
            "successor": lookup_successor(kind),
            "claim_boundary": claim,
            "instrument_fail": kind == "INSTRUMENT_OR_BIND_FAIL",
        }
        if extra:
            out.update(extra)
        return out

    if identity != "IDENTITY_OK":
        return _fail("IDENTITY_BIND_FAIL", id_reasons)

    auth_status, auth_reasons, auth_raw = check_component_authority(
        projections, rung3_terminal=rung3_terminal, rung4_terminal=rung4_terminal
    )
    if auth_status != "AUTHORITY_OK":
        return _fail(
            "AUTHORITY_BIND_FAIL",
            auth_reasons,
            {"authority_raw": auth_raw, "authority_profile": auth_status},
        )

    c1_labels: dict[str, dict[str, str]] = {}
    c1_raws: dict[str, dict[str, dict[str, Any]]] = {}
    agg_d2_labels: dict[str, str] = {}
    agg_d2_raws: dict[str, dict[str, Any]] = {}
    empty_components: list[str] = []

    for support in SUPPORTS:
        sp = projections[support]
        u = extract_universe(sp["package"][50])
        bmap = extract_bucket_map(sp["package"][50])
        r50 = extract_survivors(sp["package"][50])
        pkg20 = extract_survivors(sp["package"][20])
        pkg10 = extract_survivors(sp["package"][10])
        enr = list(SHARED_ENRICHED_COMPONENTS)  # pinned pair; not dynamic d1 enr
        lab2, raw2 = d2_per_support(r50, bmap, enr, pkg20, pkg10)
        agg_d2_labels[support] = lab2
        agg_d2_raws[support] = raw2
        c1_labels[support] = {}
        c1_raws[support] = {}
        for c in SHARED_ENRICHED_COMPONENTS:
            clab, craw = c1_component_raw(r50, bmap, c, pkg20, pkg10)
            c1_labels[support][c] = clab
            c1_raws[support][c] = craw
            if clab == "E_EMPTY" or int(craw["|B50|_row_ids"]) == 0:
                empty_components.append(f"{support}:{c}")

    recomp_status, recomp_reasons, recomp_raw = recompose_check(c1_raws, agg_d2_raws)
    if recomp_status != "RECOMPOSITION_OK":
        return _fail(
            "RECOMPOSITION_BIND_FAIL",
            recomp_reasons,
            {
                "authority_profile": "AUTHORITY_OK",
                "authority_raw": auth_raw,
                "C1_profile": {
                    "per_support_per_component": c1_labels,
                    "raw": c1_raws,
                },
                "recomposition_raw": recomp_raw,
            },
        )

    if empty_components:
        return _fail(
            "DEGENERATE_EMPTY_COMPONENT",
            empty_components,
            {
                "authority_profile": "AUTHORITY_OK",
                "authority_raw": auth_raw,
                "C1_profile": {
                    "per_support_per_component": c1_labels,
                    "raw": c1_raws,
                    "alignment": cross_support_alignment(c1_labels),
                },
            },
        )

    alignment = cross_support_alignment(c1_labels)
    agg_differ = agg_d2_labels["L0b"] != agg_d2_labels["math_a0"]
    # C2: aligned+differ→ALIGNED; not aligned→SPLIT; else hold (no values_primary).
    if alignment["both_components_aligned"] and agg_differ:
        primary: str | None = "ALIGNED_COMPONENT_LABELS__AGGREGATE_SPLIT"
        composite = f"IDENTITY_OK__{primary}"
    elif not alignment["both_components_aligned"]:
        primary = "COMPONENT_LABEL_SPLIT"
        composite = f"IDENTITY_OK__{primary}"
    else:
        primary = None
        composite = "IDENTITY_OK"  # else-hold; not a values_primary token

    # A1 secondary (math_a0 only)
    spm = projections[MASS_HEAD_COMPARATOR_SUPPORT]
    a1_lab, a1_raw = a1_secondary(
        extract_survivors(spm["package"][50]),
        extract_bucket_map(spm["package"][50]),
        extract_survivors(spm["package"][20]),
        extract_survivors(spm["package"][10]),
        c1_labels[MASS_HEAD_COMPARATOR_SUPPORT],
    )
    b_table = b_l0b_row_table(projections)

    return {
        "schema": SCHEMA_ID,
        "product": PRODUCT,
        "identity_profile": "IDENTITY_OK",
        "identity_reasons": [],
        "identity_raw": id_raw,
        "terminal_reasons": [],
        "terminal_kind": None,
        "authority_profile": "AUTHORITY_OK",
        "authority_reasons": [],
        "authority_raw": auth_raw,
        "C1_profile": {
            "per_support_per_component": c1_labels,
            "raw": c1_raws,
            "alignment": alignment,
            "membership_carrier": "row_id",
        },
        "C2_profile": {
            "status": "RECOMPOSITION_OK",
            "raw": recomp_raw,
            "aggregate_d2_labels": agg_d2_labels,
            "aggregate_labels_differ": agg_differ,
            "primary": primary,
        },
        "A1_secondary": {
            "label": a1_lab,
            "raw": a1_raw,
            "branch_authority": "NONE",
        },
        "B_report_only": {
            "L0b_row_table": b_table,
            "row_count": len(b_table),
            "branch_authority": "NONE",
        },
        "composite_terminal": composite,
        "successor": lookup_successor("IDENTITY_OK", primary),
        "claim_boundary": claim,
        "instrument_fail": False,
    }

def component_core(result: Mapping[str, Any]) -> dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in result.items() if k not in NON_AUTHORITATIVE_KEYS}
