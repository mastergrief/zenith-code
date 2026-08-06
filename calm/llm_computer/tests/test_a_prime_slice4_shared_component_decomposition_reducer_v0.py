"""STEP-1 shared-component reducer tests. PLAN v4."""
from __future__ import annotations

import copy

import pytest

from scripts.a_prime_slice4_residual_classification_reducer_v0 import (
    residual_composite,
    residual_per_support_label,
)
from scripts.a_prime_slice4_shared_component_decomposition_reducer_v0 import (
    a1_secondary,
    c1_component_raw,
    check_component_authority,
    check_identity,
    check_rung4_authority,
    component_core,
    component_from_projections,
    cross_support_alignment,
    lookup_successor,
    recompose_check,
)
from scripts.a_prime_slice4_shared_component_decomposition_schema_v0 import (
    FROZEN_NEUTRAL_SUCCESSOR_TEXTS as T,
    REQUIRED_CLAIM_BOUNDARY,
    SHARED_ENRICHED_COMPONENTS,
    ceil_0_70,
    extract_bucket_map,
    extract_survivors,
    extract_universe,
)
from scripts.a_prime_slice4_support_split_residual_densify_reducer_v0 import (
    d1_per_support,
    d2_per_support,
)

def _view(rows, bucks, fails=None):
    fails = list(fails or [])
    return {
        "row_ids": list(rows),
        "sample_hashes": [r.rsplit(":", 1)[-1] for r in rows],
        "source_buckets": list(bucks),
        "strict_failure_row_ids": fails,
        "support_rows_audited": len(rows),
    }

def _proj(rows, bucks, pkg50, pkg20=None, pkg10=None, out50=None):
    U = set(rows)

    def f(s):
        return sorted(U - set(s))

    p50, p20 = set(pkg50), set(pkg20 if pkg20 is not None else pkg50)
    p10 = set(pkg10 if pkg10 is not None else p20)
    o50 = set(out50 if out50 is not None else ())
    return {
        "package": {
            50: _view(rows, bucks, f(p50)),
            20: _view(rows, bucks, f(p20)),
            10: _view(rows, bucks, f(p10)),
        },
        "out": {
            50: _view(rows, bucks, f(o50)),
            20: _view(rows, bucks, f(o50)),
            10: _view(rows, bucks, f(U)),
        },
    }

def l0b_cal():
    elig = [
        "R0",
        "R1b1",
        "R1b2",
        "R1b3",
        "R1b4v2",
        "R1b5",
        "R1b6",
        "R1b7",
        "R1b8",
        "R1b9",
    ]
    inelig = ["R1_0_plus_A", "R1_minus_0", "R1_plus_0"]
    rows, bucks = [], []
    for b in elig:
        for j in range(20):
            rows.append(f"L0b:{b}_{j:02d}")
            bucks.append(b)
    for b in inelig:
        for j in range(10):
            rows.append(f"L0b:{b}_{j:02d}")
            bucks.append(b)
    surv = []
    for b, k in [
        ("R0", 5),
        ("R1b4v2", 3),
        ("R1b1", 2),
        ("R1b3", 2),
        ("R1b7", 2),
        ("R1b5", 1),
        ("R1b9", 1),
    ]:
        surv += [f"L0b:{b}_{j:02d}" for j in range(k)]
    surv += [f"L0b:R1_minus_0_{j:02d}" for j in range(3)] + ["L0b:R1_0_plus_A_00"]
    assert len(rows) == 230 and len(surv) == 20
    # N20 present: R0:1, R1b4v2:2 → present20=3 for enriched; matches live
    p20 = [f"L0b:R0_00"] + [f"L0b:R1b4v2_{j:02d}" for j in range(2)] + surv[8:12]
    p10 = p20[:3] + surv[8:14]
    return _proj(rows, bucks, surv, p20, p10, surv[:5]), rows, bucks, surv

def math_cal():
    sizes = {
        "R0": 100,
        "R1": 300,
        "R1b1": 98,
        "R1b2": 99,
        "R1b3": 97,
        "R1b4v2": 96,
        "R1b5": 95,
        "R1b6": 94,
        "R1b7": 93,
        "R1b8": 92,
        "R1b9": 91,
    }
    rows, bucks = [], []
    for b, n in sizes.items():
        for j in range(n):
            rows.append(f"math_a0:{b}_{j:03d}")
            bucks.append(b)
    assign = {
        "R1": 19,
        "R0": 14,
        "R1b4v2": 9,
        "R1b3": 8,
        "R1b1": 5,
        "R1b5": 5,
        "R1b8": 5,
        "R1b9": 5,
        "R1b7": 4,
        "R1b6": 2,
        "R1b2": 0,
    }
    surv = []
    for b, k in assign.items():
        surv += [f"math_a0:{b}_{j:03d}" for j in range(k)]
    assert len(rows) == 1255 and len(surv) == 76
    # live-shaped: R0 present20=1, R1b4v2 present20=5, R1 present20=5
    p20 = (
        [f"math_a0:R0_000"]
        + [f"math_a0:R1b4v2_{j:03d}" for j in range(5)]
        + [f"math_a0:R1_{j:03d}" for j in range(5)]
        + [f"math_a0:R1b3_{j:03d}" for j in range(4)]
    )
    p10 = p20 + [f"math_a0:R0_{j:03d}" for j in range(1, 9)] + [
        f"math_a0:R1b4v2_{j:03d}" for j in range(5, 7)
    ]
    return _proj(rows, bucks, surv, p20, p10, surv[:12])

def both(a, b):
    return {"L0b": a, "math_a0": b}

def term_rung3(l0b, math):
    ps, raws = {}, {}
    for name, proj in (("L0b", l0b), ("math_a0", math)):
        v = proj["package"][50]
        lab, raw = residual_per_support_label(
            extract_universe(v),
            extract_bucket_map(v),
            extract_survivors(v),
            arm="package",
            horizon=50,
        )
        ps[name], raws[name] = lab, raw
    return {
        "composite_terminal": "IDENTITY_OK__CHURNED__TRANSIENT__STRATIFIED",
        "successor": "support-split residual densify; no mechanism mint",
        "residual_bucket_profile": {
            "composite": residual_composite(ps["L0b"], ps["math_a0"]),
            "per_support": ps,
            "raw": raws,
        },
        "source_shas": {},
    }

def term_rung4(l0b, math):
    d2_ps, d2_raw = {}, {}
    for name, proj in (("L0b", l0b), ("math_a0", math)):
        v = proj["package"][50]
        u = extract_universe(v)
        bm = extract_bucket_map(v)
        r50 = extract_survivors(v)
        p20 = extract_survivors(proj["package"][20])
        p10 = extract_survivors(proj["package"][10])
        _l1, raw1 = d1_per_support(u, bm, r50)
        enr = list(raw1.get("enriched_buckets") or [])
        lab2, raw2 = d2_per_support(r50, bm, enr, p20, p10)
        d2_ps[name], d2_raw[name] = lab2, raw2
    d2c = d2_ps["L0b"] if d2_ps["L0b"] == d2_ps["math_a0"] else "SPLIT_SUPPORTS"
    return {
        "composite_terminal": "IDENTITY_OK__HEAD2_THIRD__SPLIT_SUPPORTS__CO_PARTIAL",
        "D2_profile": {
            "composite": d2c,
            "per_support": d2_ps,
            "raw": d2_raw,
            "membership_carrier": "row_id",
        },
        "source_shas": {},
    }

def cal_pair():
    l0b, _, _, _ = l0b_cal()
    math = math_cal()
    return l0b, math, term_rung3(l0b, math), term_rung4(l0b, math)

def test_ceil_imported():
    assert ceil_0_70(8) == 6 and ceil_0_70(23) == 17

def test_c1_label_boundaries():
    r50 = {f"id{i}" for i in range(10)}
    bmap = {f"id{i}": "R0" for i in range(10)}
    lab, raw = c1_component_raw(r50, bmap, "R0", set(list(r50)[:7]), set())
    assert raw["ceil_0_70_|B50|"] == 7 and lab == "E_PERSISTENT"
    assert c1_component_raw(r50, bmap, "R0", set(list(r50)[:3]), set())[0] == "E_TRANSIENT"
    assert c1_component_raw(r50, bmap, "R0", set(list(r50)[:5]), set())[0] == "E_MIXED"
    lab4, raw4 = c1_component_raw(r50, bmap, "OTHER", set(), set())
    assert lab4 == "E_EMPTY" and raw4["|B50|_row_ids"] == 0

def test_c1_live_shaped_labels():
    l0b, math, r3, r4 = cal_pair()
    res = component_from_projections(
        both(l0b, math), rung3_terminal=r3, rung4_terminal=r4
    )
    assert res["composite_terminal"] == (
        "IDENTITY_OK__ALIGNED_COMPONENT_LABELS__AGGREGATE_SPLIT"
    )
    labs = res["C1_profile"]["per_support_per_component"]
    assert labs["L0b"]["R0"] == "E_TRANSIENT"
    assert labs["L0b"]["R1b4v2"] == "E_MIXED"
    assert labs["math_a0"]["R0"] == "E_TRANSIENT"
    assert labs["math_a0"]["R1b4v2"] == "E_MIXED"
    assert res["C1_profile"]["alignment"]["both_components_aligned"] is True
    assert res["A1_secondary"]["label"] == "R1_LABEL_EQ_R0"
    assert res["A1_secondary"]["branch_authority"] == "NONE"
    assert res["B_report_only"]["branch_authority"] == "NONE"
    assert res["B_report_only"]["row_count"] == 20
    assert res["successor"] == T["step_6"]
    for k, v in REQUIRED_CLAIM_BOUNDARY.items():
        assert res["claim_boundary"][k] is v

def test_c2_recomposition_ok_and_flip_hostile():
    l0b, math, r3, r4 = cal_pair()
    res = component_from_projections(
        both(l0b, math), rung3_terminal=r3, rung4_terminal=r4
    )
    assert res["C2_profile"]["status"] == "RECOMPOSITION_OK"
    # flip hostile: mutate a component raw sum path by injecting empty component via
    # recompose_check directly
    good = res["C1_profile"]["raw"]
    agg = {
        s: res["C2_profile"]["raw"][s]["aggregate"] for s in ("L0b", "math_a0")
    }
    # reconstruct aggregate from recomp raw
    # simpler: call recompose_check with tampered sum
    tampered = copy.deepcopy(good)
    tampered["L0b"]["R0"]["|B50|_row_ids"] = int(tampered["L0b"]["R0"]["|B50|_row_ids"]) + 1
    st, reasons, _ = recompose_check(tampered, {
        s: res["C2_profile"]["raw"][s]["aggregate"] for s in ("L0b", "math_a0")
    })
    assert st == "RECOMPOSITION_BIND_FAIL"
    assert any("L0b:recomp_" in r for r in reasons)

def test_degenerate_empty_component():
    l0b, rows, bucks, surv = l0b_cal()
    math = math_cal()
    surv2 = [s for s in surv if not s.startswith("L0b:R0_")]
    l0b2 = _proj(rows, bucks, surv2, surv2[:3], surv2[:5], surv2[:2])
    res = component_from_projections(both(l0b2, math), rung3_terminal=None, rung4_terminal=None)
    assert res["composite_terminal"] == "DEGENERATE_EMPTY_COMPONENT"
    assert res["successor"] == T["step_5"]
    assert res["identity_profile"] == "IDENTITY_OK"
    assert res["identity_reasons"] == []
    reasons = res.get("terminal_reasons") or []
    assert any("L0b:R0" in r for r in reasons)
    assert res.get("terminal_kind") == "DEGENERATE_EMPTY_COMPONENT"

def test_component_label_split():
    l0b, math, r3, r4 = cal_pair()
    math2 = copy.deepcopy(math)
    r50 = extract_survivors(math2["package"][50])
    bmap = extract_bucket_map(math2["package"][50])
    r0_ids = {rid for rid in r50 if bmap.get(rid) == "R0"}
    p20_old = extract_survivors(math2["package"][20])
    p20_new = sorted(p20_old | r0_ids)
    U = extract_universe(math2["package"][50])
    fails20 = sorted(set(U) - set(p20_new))
    view20 = dict(math2["package"][20])
    view20["strict_failure_row_ids"] = fails20
    math2 = dict(math2)
    math2["package"] = dict(math2["package"])
    math2["package"][20] = view20
    r3b = term_rung3(l0b, math2)
    res = component_from_projections(
        both(l0b, math2), rung3_terminal=r3b, rung4_terminal=None
    )
    assert res["composite_terminal"] == "IDENTITY_OK__COMPONENT_LABEL_SPLIT"
    assert res["C1_profile"]["alignment"]["both_components_aligned"] is False
    assert res["successor"] == T["step_7"]

def test_a1_secondary_eq_variants():
    r50 = {f"r{i}" for i in range(10)}
    bmap = {f"r{i}": "R1" for i in range(10)}
    sec, raw = a1_secondary(
        r50, bmap, set(list(r50)[:3]), set(), {"R0": "E_TRANSIENT", "R1b4v2": "E_MIXED"}
    )
    assert sec == "R1_LABEL_EQ_R0" and raw["R1_label"] == "E_TRANSIENT"
    sec2, raw2 = a1_secondary(
        r50, bmap, set(list(r50)[:7]), set(), {"R0": "E_MIXED", "R1b4v2": "E_PERSISTENT"}
    )
    assert raw2["R1_label"] == "E_PERSISTENT"
    assert sec2 == "R1_LABEL_EQ_R1b4v2"
    sec3, _ = a1_secondary(
        r50, bmap, set(list(r50)[:7]), set(), {"R0": "E_PERSISTENT", "R1b4v2": "E_PERSISTENT"}
    )
    assert sec3 == "R1_LABEL_EQ_R0_AND_R1b4v2"
    sec4, _ = a1_secondary(
        r50, bmap, set(list(r50)[:5]), set(), {"R0": "E_TRANSIENT", "R1b4v2": "E_TRANSIENT"}
    )
    assert sec4 == "R1_LABEL_EQ_NEITHER"

def test_a1_and_b_cannot_change_primary():
    l0b, math, r3, r4 = cal_pair()
    res = component_from_projections(
        both(l0b, math), rung3_terminal=r3, rung4_terminal=r4
    )
    primary = res["composite_terminal"]
    assert res["A1_secondary"]["branch_authority"] == "NONE"
    assert res["B_report_only"]["branch_authority"] == "NONE"
    assert primary == "IDENTITY_OK__ALIGNED_COMPONENT_LABELS__AGGREGATE_SPLIT"

def test_identity_bucket_map_drift():
    l0b, rows, bucks, surv = l0b_cal()
    math = math_cal()
    view = dict(l0b["package"][20])
    sb = list(view["source_buckets"])
    sb[0] = "R1b9" if sb[0] != "R1b9" else "R0"
    view["source_buckets"] = sb
    l0b2 = dict(l0b)
    l0b2["package"] = dict(l0b["package"])
    l0b2["package"][20] = view
    idn, reasons, _ = check_identity(both(l0b2, math))
    assert idn == "IDENTITY_BIND_FAIL"
    assert any("bucket_map_ne_reference" in r for r in reasons)

def test_identity_length_mismatch():
    l0b, _, _, _ = l0b_cal()
    math = math_cal()
    view = dict(l0b["package"][50])
    view["sample_hashes"] = view["sample_hashes"][:-1]
    l0b2 = dict(l0b)
    l0b2["package"] = dict(l0b["package"])
    l0b2["package"][50] = view
    res = component_from_projections(both(l0b2, math))
    assert res["composite_terminal"] == "IDENTITY_BIND_FAIL"

# Authoritative densify D2 raw fields (whole-map bind in check_rung4_authority).
_D2_RAW_HOSTILES = (
    ("|E50|_row_ids", 999),
    ("present_at_package_N20_row_id_intersection", 999),
    ("absent_from_package_N20_row_id_difference", 999),
    ("present_at_package_N10_row_id_intersection", 999),
    ("ceil_0_70_|E50|", 999),
    ("membership_carrier", "sample_hash"),
    ("enriched_buckets", ["WRONG"]),
)


@pytest.mark.parametrize("support", ["L0b", "math_a0"])
@pytest.mark.parametrize("field,bad", _D2_RAW_HOSTILES)
def test_authority_rung4_raw_field_drift_whole_map(support, field, bad):
    """Any terminal D2 raw field drift → AUTHORITY_BIND_FAIL; no science terminal."""
    l0b, math, r3, r4 = cal_pair()
    r4b = copy.deepcopy(r4)
    r4b["D2_profile"]["raw"][support][field] = bad
    st, reasons, _ = check_rung4_authority(both(l0b, math), r4b)
    assert st == "AUTHORITY_BIND_FAIL"
    assert any("d2_terminal_raw" in r for r in reasons)
    res = component_from_projections(
        both(l0b, math), rung3_terminal=r3, rung4_terminal=r4b
    )
    assert res["composite_terminal"] == "AUTHORITY_BIND_FAIL"
    assert not str(res["composite_terminal"]).startswith("IDENTITY_OK__")

def test_authority_rung3_composite_mismatch():
    l0b, math, r3, r4 = cal_pair()
    r3b = copy.deepcopy(r3)
    r3b["composite_terminal"] = "WRONG"
    res = component_from_projections(
        both(l0b, math), rung3_terminal=r3b, rung4_terminal=r4
    )
    assert res["composite_terminal"] == "AUTHORITY_BIND_FAIL"

def test_successor_frozen_texts():
    assert lookup_successor("INSTRUMENT_OR_BIND_FAIL") == T["instrument"]
    assert lookup_successor("IDENTITY_BIND_FAIL") == T["identity"]
    assert lookup_successor("AUTHORITY_BIND_FAIL") == T["authority"]
    assert lookup_successor("RECOMPOSITION_BIND_FAIL") == T["recomposition"]
    assert lookup_successor("DEGENERATE_EMPTY_COMPONENT") == T["step_5"]
    assert (
        lookup_successor("IDENTITY_OK", "ALIGNED_COMPONENT_LABELS__AGGREGATE_SPLIT")
        == T["step_6"]
    )
    assert lookup_successor("IDENTITY_OK", "COMPONENT_LABEL_SPLIT") == T["step_7"]
    assert lookup_successor("IDENTITY_OK", None) == T["step_8"]

def test_component_core_snapshot():
    l0b, math, r3, r4 = cal_pair()
    res = component_from_projections(
        both(l0b, math), rung3_terminal=r3, rung4_terminal=r4
    )
    core = component_core(res)
    assert core["composite_terminal"] == res["composite_terminal"]
    assert "schema" in core and "C1_profile" in core

def test_alignment_helper():
    labels = {
        "L0b": {"R0": "E_TRANSIENT", "R1b4v2": "E_MIXED"},
        "math_a0": {"R0": "E_TRANSIENT", "R1b4v2": "E_MIXED"},
    }
    a = cross_support_alignment(labels)
    assert a["both_components_aligned"] is True
    labels["math_a0"]["R0"] = "E_MIXED"
    assert cross_support_alignment(labels)["both_components_aligned"] is False

def test_no_cross_support_row_id_intersection_in_core_keys():
    l0b, math, r3, r4 = cal_pair()
    res = component_from_projections(
        both(l0b, math), rung3_terminal=r3, rung4_terminal=r4
    )
    blob = repr(res["C1_profile"]) + repr(res["C2_profile"])
    assert "cross_support_row_id" not in blob

def test_aligned_aggregates_equal_hold_no_science_branch():
    """aligned AND NOT agg_differ → IDENTITY_OK + step_8 (no values_primary)."""
    l0b, math, _, _ = cal_pair()
    math2 = copy.deepcopy(math)
    r50 = extract_survivors(math2["package"][50])
    bmap = extract_bucket_map(math2["package"][50])
    r0 = sorted(rid for rid in r50 if bmap.get(rid) == "R0")
    r1b = sorted(rid for rid in r50 if bmap.get(rid) == "R1b4v2")
    other = sorted(rid for rid in r50 if bmap.get(rid) not in ("R0", "R1b4v2"))
    # R0 present4→TRANSIENT; R1b4v2 present5→MIXED; agg present9/absent14 → E_MIXED
    p20 = set(r0[:4] + r1b[:5] + other[:5])
    U = extract_universe(math2["package"][50])
    fails20 = sorted(set(U) - p20)
    view20 = dict(math2["package"][20])
    view20["strict_failure_row_ids"] = fails20
    math2 = dict(math2)
    math2["package"] = dict(math2["package"])
    math2["package"][20] = view20
    res = component_from_projections(
        both(l0b, math2), rung3_terminal=None, rung4_terminal=None
    )
    assert res["C1_profile"]["alignment"]["both_components_aligned"] is True
    assert res["C2_profile"]["aggregate_labels_differ"] is False
    assert res["C2_profile"]["primary"] is None
    assert res["composite_terminal"] == "IDENTITY_OK"
    assert res["successor"] == T["step_8"]
    assert "COMPONENT_LABEL_SPLIT" not in res["composite_terminal"]

def test_imports_densify_not_reimplement():
    import scripts.a_prime_slice4_shared_component_decomposition_reducer_v0 as r
    import scripts.a_prime_slice4_support_split_residual_densify_reducer_v0 as d

    assert r.d2_per_support is d.d2_per_support
    assert r.d1_per_support is d.d1_per_support
    assert r.check_identity is d.check_identity
    assert r.densify_check_authority is d.check_authority
