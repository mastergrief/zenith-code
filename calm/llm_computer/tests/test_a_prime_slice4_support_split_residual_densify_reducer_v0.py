"""STEP-1 densify reducer tests. PLAN v6 feea775c…29a9"""
from __future__ import annotations
from pathlib import Path
import pytest
from scripts.a_prime_slice4_support_split_residual_densify_reducer_v0 import (
    check_identity, d1_per_support, d2_per_support, d3_per_support,
    d1_composite, d2_composite, d3_composite, densify_core,
    densify_from_projections, lookup_successor,
)
from scripts.a_prime_slice4_support_split_residual_densify_schema_v0 import (
    D1_L0B_CALIBRATION, FROZEN_NEUTRAL_SUCCESSOR_TEXTS as T,
    REQUIRED_CLAIM_BOUNDARY, ceil_0_70, extract_bucket_map,
    extract_survivors, extract_universe,
)
from scripts.a_prime_slice4_residual_classification_reducer_v0 import (
    residual_composite, residual_per_support_label,
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
    arms = {
        "package": {50: _view(rows, bucks, f(p50)), 20: _view(rows, bucks, f(p20)), 10: _view(rows, bucks, f(p10))},
        "out": {50: _view(rows, bucks, f(o50)), 20: _view(rows, bucks, f(o50)), 10: _view(rows, bucks, f(U))},
    }
    return arms

def l0b_cal():
    elig = ["R0","R1b1","R1b2","R1b3","R1b4v2","R1b5","R1b6","R1b7","R1b8","R1b9"]
    inelig = ["R1_0_plus_A","R1_minus_0","R1_plus_0"]
    rows, bucks = [], []
    for b in elig:
        for j in range(20):
            rows.append(f"L0b:{b}_{j:02d}"); bucks.append(b)
    for b in inelig:
        for j in range(10):
            rows.append(f"L0b:{b}_{j:02d}"); bucks.append(b)
    surv = []
    for b,k in [("R0",5),("R1b4v2",3),("R1b1",2),("R1b3",2),("R1b7",2),("R1b5",1),("R1b9",1)]:
        surv += [f"L0b:{b}_{j:02d}" for j in range(k)]
    surv += [f"L0b:R1_minus_0_{j:02d}" for j in range(3)] + ["L0b:R1_0_plus_A_00"]
    assert len(rows)==230 and len(surv)==20
    return _proj(rows, bucks, surv, surv[:8], surv[:12], surv[:2]), rows, bucks, surv

def math_cal():
    sizes = {"R0":100,"R1":300,"R1b1":98,"R1b2":99,"R1b3":97,"R1b4v2":96,"R1b5":95,"R1b6":94,"R1b7":93,"R1b8":92,"R1b9":91}
    rows, bucks = [], []
    for b,n in sizes.items():
        for j in range(n):
            rows.append(f"math_a0:{b}_{j:03d}"); bucks.append(b)
    assign = {"R1":19,"R0":14,"R1b4v2":9,"R1b3":8,"R1b1":5,"R1b5":5,"R1b8":5,"R1b9":5,"R1b7":4,"R1b6":2,"R1b2":0}
    surv = []
    for b,k in assign.items():
        surv += [f"math_a0:{b}_{j:03d}" for j in range(k)]
    assert len(rows)==1255 and len(surv)==76
    return _proj(rows, bucks, surv, surv[:30], surv[:50], surv[:10])

def both(a,b): return {"L0b": a, "math_a0": b}

def term_for(l0b, math):
    ps, raws = {}, {}
    for name, proj in (("L0b", l0b), ("math_a0", math)):
        v = proj["package"][50]
        lab, raw = residual_per_support_label(
            extract_universe(v), extract_bucket_map(v), extract_survivors(v),
            arm="package", horizon=50)
        ps[name], raws[name] = lab, raw
    return {
        "composite_terminal": "IDENTITY_OK__CHURNED__TRANSIENT__STRATIFIED",
        "successor": "support-split residual densify; no mechanism mint",
        "residual_bucket_profile": {
            "composite": residual_composite(ps["L0b"], ps["math_a0"]),
            "per_support": ps, "raw": raws},
        "source_shas": {},
    }

def test_ceil_0_70_integer():
    assert ceil_0_70(0)==0 and ceil_0_70(1)==1 and ceil_0_70(10)==7 and ceil_0_70(47)==33

def test_d1_head_thresholds_total_S():
    rows = [f"L0b:h{i:04d}" for i in range(40)]
    bucks = ["A"]*20+["B"]*20
    lab,_ = d1_per_support(rows, dict(zip(rows,bucks)), set(rows[:6]+rows[20:24]))
    assert lab=="HEAD1_MAJORITY"
    rows2 = rows+[f"L0b:c{i:04d}" for i in range(10)]; bucks2 = bucks+["C"]*10
    lab2,_ = d1_per_support(rows2, dict(zip(rows2,bucks2)), set(rows[:4]+rows[20:23]+rows2[40:42]))
    assert lab2=="HEAD2_MAJORITY"
    lab4, raw4 = d1_per_support(rows2, dict(zip(rows2,bucks2)), set(rows[:3]+rows[20:22]+rows2[40:47]))
    assert raw4["top2_sum"]==5 and lab4=="HEAD2_THIRD"
    lab5,_ = d1_per_support(rows2, dict(zip(rows2,bucks2)), set(rows[:2]+rows[20:22]+rows2[40:50]))
    assert lab5=="HEAD_DIFFUSE"

def test_d1_l0b_calibration_total_vs_eligible_flip():
    cal = D1_L0B_CALIBRATION
    proj,_,_,_ = l0b_cal()
    v = proj["package"][50]
    u, bm, sv = extract_universe(v), extract_bucket_map(v), extract_survivors(v)
    assert len(sv)==cal["S_total"]
    lab, raw = d1_per_support(u, bm, sv)
    assert raw["S_s_total_support_survivors"]==cal["S_total"]
    assert raw["S_eligible_sum_report_only"]==cal["S_eligible_sum"]
    assert raw["top2_sum"]==cal["top2_sum"]
    assert lab==cal["must_label"]
    assert raw["top2_sum"]*2 >= raw["S_eligible_sum_report_only"]
    assert lab != cal["forbidden_label_under_eligible_only_denom"]

def test_d1_degenerate():
    rows=[f"L0b:h{i:04d}" for i in range(40)]; bucks=["A"]*20+["B"]*20
    assert d1_per_support(rows, dict(zip(rows,bucks)), set())[0]=="DEGENERATE_NO_SURVIVORS"

def test_d2_ceil_boundaries_and_empty():
    r50={f"id{i}" for i in range(10)}; bmap={f"id{i}":"R0" for i in range(10)}
    lab, raw = d2_per_support(r50, bmap, ["R0"], set(list(r50)[:7]), set())
    assert raw["ceil_0_70_|E50|"]==7 and lab=="E_PERSISTENT"
    assert d2_per_support(r50, bmap, ["R0"], set(list(r50)[:3]), set())[0]=="E_TRANSIENT"
    assert d2_per_support(r50, bmap, ["R0"], set(list(r50)[:5]), set())[0]=="E_MIXED"
    lab4, raw4 = d2_per_support(r50, bmap, ["OTHER"], set(), set())
    assert lab4=="E_EMPTY" and raw4["|E50|_row_ids"]==0

def test_d3_cosurvival():
    r50={f"r{i}" for i in range(10)}
    assert d3_per_support(r50, {f"r{i}" for i in range(5)}|{f"o{i}" for i in range(3)})[0]=="CO_MAJORITY"
    assert d3_per_support(r50, {f"r{i}" for i in range(1)})[0]=="CO_PARTIAL"
    assert d3_per_support(r50, {"zz"})[0]=="CO_DISJOINT"
    assert d3_per_support(set(), r50)[0]=="DEGENERATE_NO_SURVIVORS"

def test_composites_split():
    assert d1_composite("HEAD2_THIRD","HEAD2_THIRD")=="HEAD2_THIRD"
    assert d1_composite("HEAD2_THIRD","HEAD_DIFFUSE")=="SPLIT_SUPPORTS"
    assert d2_composite("E_EMPTY","E_TRANSIENT")=="SPLIT_SUPPORTS"
    assert d2_composite("E_EMPTY","E_EMPTY")=="E_EMPTY"
    assert d3_composite("CO_DISJOINT","CO_PARTIAL")=="SPLIT_SUPPORTS"

def test_successor_frozen_texts():
    assert lookup_successor("INSTRUMENT_OR_BIND_FAIL","X","Y","Z")==T["instrument"]
    assert lookup_successor("IDENTITY_BIND_FAIL","X","Y","Z")==T["identity"]
    assert lookup_successor("AUTHORITY_BIND_FAIL","X","Y","Z")==T["authority"]
    assert lookup_successor("IDENTITY_OK","SPLIT_SUPPORTS","E_MIXED","CO_PARTIAL")==T["split"]
    assert lookup_successor("IDENTITY_OK","HEAD1_MAJORITY","E_MIXED","CO_DISJOINT")==T["step_5"]
    assert lookup_successor("IDENTITY_OK","HEAD2_THIRD","E_TRANSIENT","CO_DISJOINT")==T["step_6"]
    assert lookup_successor("IDENTITY_OK","HEAD2_THIRD","E_PERSISTENT","CO_MAJORITY")==T["step_7"]
    assert lookup_successor("IDENTITY_OK","HEAD_DIFFUSE","E_MIXED","CO_PARTIAL")==T["default"]

def test_identity_length_mismatch_short_sample_hashes():
    proj,_,_,_ = l0b_cal(); math=math_cal()
    view=dict(proj["package"][50]); view["sample_hashes"]=view["sample_hashes"][:-1]
    proj=dict(proj); proj["package"]=dict(proj["package"]); proj["package"][50]=view
    idn, reasons, _ = check_identity(both(proj, math))
    assert idn=="IDENTITY_BIND_FAIL" and any("sample_hashes" in r or "length_mismatch" in r for r in reasons)

def test_identity_length_mismatch_long_source_buckets():
    proj,_,_,_ = l0b_cal(); math=math_cal()
    view=dict(proj["package"][50]); view["source_buckets"]=list(view["source_buckets"])+["EXTRA"]
    proj=dict(proj); proj["package"]=dict(proj["package"]); proj["package"][50]=view
    idn, reasons, _ = check_identity(both(proj, math))
    assert idn=="IDENTITY_BIND_FAIL" and any("source_buckets" in r or "length_mismatch" in r for r in reasons)

def test_identity_universe_mismatch_across_horizons():
    proj,_,_,_ = l0b_cal(); math=math_cal()
    view=dict(proj["package"][20]); rids=list(view["row_ids"]); rids[0]="L0b:TAMPERED_0000"
    view["row_ids"]=rids; view["sample_hashes"]=[r.rsplit(":",1)[-1] for r in rids]
    # keep parallel list lens: buckets already len-matched; sample_hashes rebuilt
    proj=dict(proj); proj["package"]=dict(proj["package"]); proj["package"][20]=view
    idn, reasons, _ = check_identity(both(proj, math))
    assert idn=="IDENTITY_BIND_FAIL"
    assert any("universe_ne_package_N50" in r or "suffix_ne" in r or "length" in r for r in reasons)

@pytest.mark.parametrize("arm,h", [
    ("package", 10), ("package", 20), ("package", 50),
    ("out", 10), ("out", 20), ("out", 50),
])
def test_identity_bucket_map_drift_any_view(arm, h):
    """Class cure: one source_bucket flip in ANY arm×horizon → IDENTITY_BIND_FAIL."""
    proj, rows, bucks, _ = l0b_cal(); math = math_cal()
    if arm == "package" and h == 50:
        # drift a non-reference view would be preferred, but package-N50 is the
        # reference: flipping it makes all others fail vs reference — still FAIL.
        pass
    view = dict(proj[arm][h])
    sb = list(view["source_buckets"])
    # flip first row's bucket to a different existing name (preserve lengths/hashes/row_ids)
    sb[0] = "R1b9" if sb[0] != "R1b9" else "R0"
    view["source_buckets"] = sb
    proj = dict(proj)
    proj[arm] = dict(proj[arm])
    proj[arm][h] = view
    # if we only flip package-N50 reference, other views mismatch reference — still fail
    idn, reasons, _ = check_identity(both(proj, math))
    assert idn == "IDENTITY_BIND_FAIL"
    assert any("bucket_map_ne_reference" in r for r in reasons)

def test_end_to_end_known_good_and_core_keys():
    l0b,_,_,_=l0b_cal(); math=math_cal(); term=term_for(l0b, math)
    res=densify_from_projections(both(l0b, math), authority_terminal=term)
    assert res["identity_profile"]=="IDENTITY_OK" and res["authority_profile"]=="AUTHORITY_OK"
    assert res["D1_profile"]["per_support"]["L0b"]=="HEAD2_THIRD"
    assert res["D1_profile"]["branch_denominator"]=="TOTAL_PACKAGE_N50_SUPPORT_SURVIVORS"
    assert res["composite_terminal"].startswith("IDENTITY_OK__")
    assert res["claim_boundary"]==REQUIRED_CLAIM_BOUNDARY
    core=densify_core(res)
    assert set(core)==set(res) and "identity_reasons" in core
    assert res["D2_profile"]["membership_carrier"]=="row_id"
    assert res["D3_profile"]["membership_carrier"]=="row_id"
    assert "report_only" in res

def test_authority_bind_fail_on_label_tamper():
    l0b,_,_,_=l0b_cal(); math=math_cal(); term=term_for(l0b, math)
    term=dict(term); rbp=dict(term["residual_bucket_profile"]); ps=dict(rbp["per_support"])
    ps["L0b"]="UNIFORM"; rbp["per_support"]=ps; term["residual_bucket_profile"]=rbp
    res=densify_from_projections(both(l0b, math), authority_terminal=term)
    assert res["composite_terminal"]=="AUTHORITY_BIND_FAIL"
    assert res["successor"]==T["authority"]

def test_authority_ok_without_terminal_fixture():
    l0b,_,_,_=l0b_cal(); math=math_cal()
    res=densify_from_projections(both(l0b, math), authority_terminal=None)
    assert res["identity_profile"]=="IDENTITY_OK" and res["authority_profile"]=="AUTHORITY_OK"
    assert res["D1_profile"]["per_support"]["L0b"]=="HEAD2_THIRD"

def test_report_only_out_enrichment_does_not_change_d1_d2():
    l0b,_,_,_=l0b_cal(); math=math_cal(); term=term_for(l0b, math)
    res1=densify_from_projections(both(l0b, math), authority_terminal=term)
    math2=math_cal(); view=dict(math2["out"][50]); view["strict_failure_row_ids"]=[]
    math2=dict(math2); math2["out"]=dict(math2["out"]); math2["out"][50]=view
    res2=densify_from_projections(both(l0b, math2), authority_terminal=term)
    assert res1["D1_profile"]["per_support"]==res2["D1_profile"]["per_support"]
    assert res1["D2_profile"]["per_support"]==res2["D2_profile"]["per_support"]

def test_report_only_tables_complete_and_non_branching():
    l0b,_,_,_=l0b_cal(); math=math_cal(); term=term_for(l0b, math)
    res=densify_from_projections(both(l0b, math), authority_terminal=term)
    for s in ("L0b", "math_a0"):
        dep = res["report_only"][s]["depleted_stratum_package"]
        for k in ("|D50|", "present_at_package_N20", "absent_from_package_N20",
                  "present_at_package_N10", "absent_from_package_N10"):
            assert k in dep and type(dep[k]) is int
        co = res["report_only"][s]["cosurvival_package_N20_out_N20"]
        for k in ("|R20|_row_ids", "|O20|_row_ids", "inter_row_id_intersection",
                  "only_package_row_id_difference", "only_out_row_id_difference"):
            assert k in co and type(co[k]) is int
    d2 = res["D2_profile"]["per_support"]
    d3 = res["D3_profile"]["per_support"]
    # mutate package-N10 / out-N20 survivors → report-only changes; D2/D3 labels hold
    # (D2 branch uses E50 vs pkg20; D3 uses r50 vs o50 — N10/out20 report-only)
    l0b2 = dict(l0b); l0b2["package"]=dict(l0b["package"]); l0b2["out"]=dict(l0b["out"])
    v10 = dict(l0b2["package"][10]); v10["strict_failure_row_ids"]=list(v10["row_ids"])  # zero N10 surv
    l0b2["package"][10] = v10
    v20o = dict(l0b2["out"][20]); v20o["strict_failure_row_ids"]=[]  # all out N20 survive
    l0b2["out"][20] = v20o
    res2=densify_from_projections(both(l0b2, math), authority_terminal=term)
    assert res2["D2_profile"]["per_support"] == d2
    assert res2["D3_profile"]["per_support"] == d3
    # report-only fields actually moved
    assert res2["report_only"]["L0b"]["depleted_stratum_package"]["present_at_package_N10"] == 0
    assert res2["report_only"]["L0b"]["cosurvival_package_N20_out_N20"]["|O20|_row_ids"] == 230

def test_d2_uses_row_id_not_index_alignment():
    r50={"a","b","c","d","e","f","g","h","i","j"}; bmap={x:"R0" for x in r50}
    lab, raw = d2_per_support(r50, bmap, ["R0"], {"j","i","h","g","f","e","d"}, set())
    assert lab=="E_PERSISTENT" and raw["present_at_package_N20_row_id_intersection"]==7

def test_line_cap_source_files():
    root = Path(__file__).resolve().parents[3]
    for rel in (
        "scripts/a_prime_slice4_support_split_residual_densify_schema_v0.py",
        "scripts/a_prime_slice4_support_split_residual_densify_reducer_v0.py",
        "calm/llm_computer/tests/test_a_prime_slice4_support_split_residual_densify_reducer_v0.py",
    ):
        n=(root/rel).read_text().count("\n")+1
        assert n<500, f"{rel} lines={n}"

def test_densify_core_emits_all_keys():
    l0b,_,_,_=l0b_cal(); math=math_cal()
    res=densify_from_projections(both(l0b, math))
    assert set(densify_core(res).keys())==set(res.keys())
