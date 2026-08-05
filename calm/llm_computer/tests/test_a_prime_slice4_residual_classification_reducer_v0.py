"""STEP-1 residual reducer battery (PLAN v6)."""
from __future__ import annotations
import copy
import pytest
from scripts.a_prime_slice4_residual_classification_reducer_v0 import (

    NINE_CELL_TABLE,
    NON_AUTHORITATIVE_KEYS,
    classification_core,
    classify_from_projections,
    counter_loss_label,
    jaccard_raw,
    lookup_successor,
    overlap_composite,
    overlap_per_support_label,
    rescue_composite,
    rescue_per_support_label,
    residual_composite,
    residual_per_support_label,
)
from scripts.a_prime_slice4_residual_classification_schema_v0 import (
    EXPECTED_CARDINALITY,
    PREEMPTING_ONLY,
    ceil_0_70,
    j_ge_0_8,
    j_le_0_3,
)
def _mk_view(row_ids, fails=None, buckets=None):
    fails = fails or []
    buckets = ["b0"] * len(row_ids) if buckets is None else list(buckets)
    return {
        "row_ids": list(row_ids),
        "sample_hashes": [rid.rsplit(":", 1)[-1] for rid in row_ids],
        "source_buckets": buckets,
        "strict_failure_row_ids": list(fails),
        "support_rows_audited": len(row_ids),
    }
def _ids(prefix: str, n: int) -> list[str]:
    return [f"{i:04d}:{prefix}{i:04x}" for i in range(n)]
def _full_grid(l0b_ids, math_ids, *, l0b_fails=None, math_fails=None, l0b_buckets=None, math_buckets=None):
    l0b_fails, math_fails = l0b_fails or {}, math_fails or {}
    out = {"L0b": {}, "math_a0": {}}
    for arm in ("package", "out"):
        out["L0b"][arm] = {h: _mk_view(l0b_ids, fails=l0b_fails.get((arm, h), []), buckets=l0b_buckets) for h in (10, 20, 50)}
        out["math_a0"][arm] = {h: _mk_view(math_ids, fails=math_fails.get((arm, h), []), buckets=math_buckets) for h in (10, 20, 50)}
    return out
def _pad(prefix: str, n: int) -> list[str]:
    return _ids(prefix, n)
L0B = _pad("l0", EXPECTED_CARDINALITY["L0b"])
MATH = _pad("ma", EXPECTED_CARDINALITY["math_a0"])
def test_ceil_0_70_integer_only_boundaries():
    assert ceil_0_70(0) == 0
    assert ceil_0_70(1) == 1  # ceil(0.7)=1
    assert ceil_0_70(10) == 7
    assert ceil_0_70(11) == 8  # 7.7 -> 8
    assert ceil_0_70(100) == 70
    for n in range(0, 50):
        assert ceil_0_70(n) == (7 * n + 9) // 10
def test_jaccard_exact_rational():
    assert j_ge_0_8(4, 5) is True
    assert j_ge_0_8(3, 5) is False
    assert j_le_0_3(3, 10) is True
    assert j_le_0_3(4, 10) is False
def test_identity_ok_clean_grid():
    proj = _full_grid(L0B, MATH)
    r = classify_from_projections(proj)
    assert r["identity_profile"] == "IDENTITY_OK"
    assert r["composite_terminal"].startswith("IDENTITY_OK__")
def test_identity_duplicate_row_id():
    ids = list(L0B)
    ids[1] = ids[0]
    proj = _full_grid(ids, MATH)
    r = classify_from_projections(proj)
    assert r["identity_profile"] == "IDENTITY_BIND_FAIL"
    assert r["composite_terminal"] == "IDENTITY_BIND_FAIL"
def test_identity_cardinality_mismatch():
    short = L0B[:100]
    proj = _full_grid(short, MATH)
    r = classify_from_projections(proj)
    assert r["identity_profile"] == "IDENTITY_BIND_FAIL"
def test_identity_universe_pkg_ne_out():
    proj = _full_grid(L0B, MATH)
    other = list(L0B)
    other[0] = "9999:deadbeefdeadbeef"
    proj["L0b"]["out"][10] = _mk_view(other)
    r = classify_from_projections(proj)
    assert r["identity_profile"] == "IDENTITY_BIND_FAIL"
def test_identity_suffix_ne_sample_hash():
    proj = _full_grid(L0B, MATH)
    view = proj["L0b"]["package"][10]
    view["sample_hashes"] = list(view["sample_hashes"])
    view["sample_hashes"][0] = "ffffffff"
    r = classify_from_projections(proj)
    assert r["identity_profile"] == "IDENTITY_BIND_FAIL"
def test_identity_bucket_map_drift_pkg_vs_out():
    buckets_a = ["b0"] * len(L0B)
    buckets_b = ["b0"] * len(L0B)
    buckets_b[3] = "b1"
    proj = _full_grid(L0B, MATH, l0b_buckets=buckets_a)
    for h in (10, 20, 50):
        proj["L0b"]["out"][h] = _mk_view(L0B, buckets=buckets_b)
    r = classify_from_projections(proj)
    assert r["identity_profile"] == "IDENTITY_BIND_FAIL"
    assert any("bucket_map_drift" in x for x in r["identity_reasons"])
def test_identity_bucket_map_drift_n20_vs_n50():
    buckets_a = ["b0"] * len(L0B)
    buckets_b = ["b0"] * len(L0B)
    buckets_b[5] = "b9"
    proj = _full_grid(L0B, MATH, l0b_buckets=buckets_a)
    proj["L0b"]["package"][50] = _mk_view(L0B, buckets=buckets_b)
    r = classify_from_projections(proj)
    assert r["identity_profile"] == "IDENTITY_BIND_FAIL"
def test_nine_cell_table_exhaustiveness():
    labels = ("STABLE_CORE", "PARTIAL_CORE", "CHURNED")
    pairs = {(a, b) for a in labels for b in labels}
    assert set(NINE_CELL_TABLE.keys()) == pairs
    assert NINE_CELL_TABLE[("CHURNED", "PARTIAL_CORE")] == "SPLIT_SUPPORTS"
    assert NINE_CELL_TABLE[("PARTIAL_CORE", "STABLE_CORE")] == "PARTIAL_CORE"
    assert overlap_composite("CHURNED", "PARTIAL_CORE") == "SPLIT_SUPPORTS"
    assert overlap_composite("PARTIAL_CORE", "STABLE_CORE") == "PARTIAL_CORE"
    assert overlap_composite("STABLE_CORE", "STABLE_CORE") == "STABLE_CORE"
    assert overlap_composite("DEGENERATE_EMPTY", "STABLE_CORE") == "DEGENERATE_EMPTY"
def test_overlap_stable_core_fixture():
    proj = _full_grid(L0B, MATH)
    r = classify_from_projections(proj)
    assert r["identity_profile"] == "IDENTITY_OK"
    assert r["survivor_overlap_profile"]["composite"] == "STABLE_CORE"
    for s in ("L0b", "math_a0"):
        assert r["survivor_overlap_profile"]["per_support"][s] == "STABLE_CORE"
        raw = r["survivor_overlap_profile"]["raw"][s]
        assert "intersect" in raw["N20"]
        assert raw["N20"]["intersect"] == raw["N20"]["union"]
def test_overlap_churned_fixture():
    half = len(L0B) // 2
    l0_pkg_fail = L0B[half:]
    l0_out_fail = L0B[:half]
    mhalf = len(MATH) // 2
    m_pkg_fail = MATH[mhalf:]
    m_out_fail = MATH[:mhalf]
    fails = {
        ("package", 20): l0_pkg_fail,
        ("package", 50): l0_pkg_fail,
        ("out", 20): l0_out_fail,
        ("out", 50): l0_out_fail,
    }
    mfails = {
        ("package", 20): m_pkg_fail,
        ("package", 50): m_pkg_fail,
        ("out", 20): m_out_fail,
        ("out", 50): m_out_fail,
    }
    proj = _full_grid(L0B, MATH, l0b_fails=fails, math_fails=mfails)
    r = classify_from_projections(proj)
    assert r["identity_profile"] == "IDENTITY_OK"
    assert r["survivor_overlap_profile"]["composite"] == "CHURNED"
def test_overlap_split_supports_fixture():
    half = len(MATH) // 2
    mf = {("package", 20): MATH[half:], ("package", 50): MATH[half:], ("out", 20): MATH[:half], ("out", 50): MATH[:half]}
    r = classify_from_projections(_full_grid(L0B, MATH, math_fails=mf))
    assert r["survivor_overlap_profile"]["per_support"] == {"L0b": "STABLE_CORE", "math_a0": "CHURNED"}
    assert r["survivor_overlap_profile"]["composite"] == "SPLIT_SUPPORTS"
def test_overlap_degenerate_empty_union():
    j = jaccard_raw(set(), set())
    assert j["union"] == 0
    lab, _ = overlap_per_support_label(j, j)
    assert lab == "DEGENERATE_EMPTY"
    assert overlap_composite("DEGENERATE_EMPTY", "STABLE_CORE") == "DEGENERATE_EMPTY"
def test_rescue_transient_boundary():
    gross = {f"g{i}" for i in range(10)}
    n20 = {f"g{i}" for i in range(3)}  # 7 absent
    n50 = set()
    lab, raw = rescue_per_support_label(gross, n20, n50)
    assert raw["threshold_ceil_0_70"] == 7
    assert lab == "TRANSIENT"
def test_rescue_persistent_boundary():
    gross = {f"g{i}" for i in range(10)}
    n20 = set(gross)  # none absent → not transient
    n50 = {f"g{i}" for i in range(7)}  # 7 present
    lab, raw = rescue_per_support_label(gross, n20, n50)
    assert lab == "PERSISTENT"
    assert raw["present_at_package_N50"] == 7
def test_rescue_nonmonotone():
    gross = {f"g{i}" for i in range(10)}
    n20 = {f"g{i}" for i in range(3)}  # 7 absent
    n50 = {f"g{i}" for i in range(7)}  # 7 present (resurrection)
    lab, _ = rescue_per_support_label(gross, n20, n50)
    assert lab == "NONMONOTONE_RESCUE"
def test_rescue_empty_gross_mixed():
    lab, raw = rescue_per_support_label(set(), set(), set())
    assert lab == "MIXED"
    assert raw.get("empty_gross") is True
def test_rescue_composite_split():
    assert rescue_composite("TRANSIENT", "MIXED") == "SPLIT_SUPPORTS"
    assert rescue_composite("TRANSIENT", "TRANSIENT") == "TRANSIENT"
def test_counter_loss_out_arm_non_branching():
    cl = {f"c{i}" for i in range(10)}
    out20 = {f"c{i}" for i in range(3)}  # 7 gone
    out50 = set()
    lab, raw = counter_loss_label(cl, out20, out50)
    assert lab == "CL_TRANSIENT"
    assert raw["branch_input"] is False
    assert raw["measured_in_arm"] == "out"
def test_counter_loss_persistent():
    cl = {f"c{i}" for i in range(10)}
    out20 = set(cl)
    out50 = {f"c{i}" for i in range(7)}
    lab, _ = counter_loss_label(cl, out20, out50)
    assert lab == "CL_PERSISTENT"
def test_counter_loss_empty():
    lab, raw = counter_loss_label(set(), set(), set())
    assert lab == "CL_MIXED"
    assert raw.get("empty_counter_loss") is True
def test_q2_cross_support_total_never_denominator():
    proj = _full_grid(L0B, MATH)
    fails_out = L0B[:10]
    for h in (10,):
        proj["L0b"]["out"][h] = _mk_view(L0B, fails=fails_out)
    r = classify_from_projections(proj)
    assert r["identity_profile"] == "IDENTITY_OK"
    for s in ("L0b", "math_a0"):
        assert "gross" in r["rescue_persistence_profile"]["raw"][s]
    assert r["rescue_persistence_profile"]["raw"]["L0b"]["gross"] == 10
    assert r["rescue_persistence_profile"]["raw"]["math_a0"]["gross"] == 0
def _bucketed(ids: list[str], n_buckets: int = 5) -> list[str]:
    return [f"b{i % n_buckets}" for i in range(len(ids))]
def test_residual_stratified_enrichment():
    buckets = _bucketed(L0B, 5)
    fails = [rid for rid, b in zip(L0B, buckets) if b != "b0"]
    lab, raw = residual_per_support_label(
        L0B, dict(zip(L0B, buckets)), set(L0B) - set(fails), arm="package", horizon=50
    )
    assert lab in ("STRATIFIED", "UNIFORM", "METADATA_INSUFFICIENT")
    assert raw["arm"] == "package" and raw["horizon"] == 50
    if raw.get("coverage_rows_ok") and raw.get("coverage_buckets_ok"):
        assert lab == "STRATIFIED"
def test_residual_bucket_rows_below_20_excluded():
    ids = L0B[:50]
    lab, raw = residual_per_support_label(ids, {rid: f"b{i}" for i, rid in enumerate(ids)}, set(ids), arm="package", horizon=50)
    assert lab == "METADATA_INSUFFICIENT" and raw.get("eligible_buckets", 0) == 0
def test_residual_coverage_rows_below_min_mi():
    buckets = [f"t{i}" for i in range(210)] + ["big"] * 20
    lab, raw = residual_per_support_label(L0B, dict(zip(L0B, buckets)), set(L0B), arm="package", horizon=50)
    assert lab == "METADATA_INSUFFICIENT"
    assert raw.get("metadata_reason") in ("coverage_below_min", "zero_eligible_buckets")
def test_residual_degenerate_no_survivors():
    lab, raw = residual_per_support_label(
        L0B, {rid: "b0" for rid in L0B}, set(), arm="package", horizon=50
    )
    assert lab == "DEGENERATE_NO_SURVIVORS"
    assert raw["support_survivors"] == 0
    assert raw["arm"] == "package" and raw["horizon"] == 50
def test_residual_composite_order():
    assert residual_composite("METADATA_INSUFFICIENT", "UNIFORM") == "METADATA_INSUFFICIENT"
    assert residual_composite("DEGENERATE_NO_SURVIVORS", "STRATIFIED") == "DEGENERATE_NO_SURVIVORS"
    assert residual_composite("STRATIFIED", "STRATIFIED") == "STRATIFIED"
    assert residual_composite("UNIFORM", "UNIFORM") == "UNIFORM"
    assert residual_composite("STRATIFIED", "UNIFORM") == "SPLIT_SUPPORTS"
def test_q3_report_only_cannot_change_branch():
    proj = _full_grid(L0B, MATH, l0b_buckets=_bucketed(L0B), math_buckets=_bucketed(MATH))
    branch = classify_from_projections(proj)["residual_bucket_profile"]["composite"]
    proj3 = copy.deepcopy(proj)
    for arm, h in (("package", 20), ("out", 10), ("out", 20), ("out", 50)):
        proj3["L0b"][arm][h] = _mk_view(L0B, fails=L0B[10:], buckets=_bucketed(L0B))
        proj3["math_a0"][arm][h] = _mk_view(MATH, fails=MATH[10:], buckets=_bucketed(MATH))
    alt3 = classify_from_projections(proj3)
    assert alt3["identity_profile"] == "IDENTITY_OK"
    assert alt3["residual_bucket_profile"]["composite"] == branch
    assert alt3["residual_bucket_profile"]["branch_input_surface"] == {"arm": "package", "horizon": 50, "exclusive": True}
def test_q3_report_only_raws_stamp_own_surface():
    r = classify_from_projections(_full_grid(L0B, MATH, l0b_buckets=_bucketed(L0B), math_buckets=_bucketed(MATH)))
    assert r["identity_profile"] == "IDENTITY_OK"
    exp = {"package:N20": ("package", 20), "out:N10": ("out", 10), "out:N20": ("out", 20), "out:N50": ("out", 50)}
    for support in ("L0b", "math_a0"):
        br = r["residual_bucket_profile"]["raw"][support]
        assert br["arm"] == "package" and br["horizon"] == 50
        ro = r["residual_bucket_profile"]["report_only"][support]
        assert set(ro) == set(exp)
        for key, (arm, h) in exp.items():
            assert ro[key]["raw"]["arm"] == arm and ro[key]["raw"]["horizon"] == h
def test_successor_split_from_overlap():
    s = lookup_successor("IDENTITY_OK", "SPLIT_SUPPORTS", "MIXED", "UNIFORM")
    assert "support-split" in s
    assert "no mechanism mint" in s
def test_successor_split_from_rescue():
    s = lookup_successor("IDENTITY_OK", "PARTIAL_CORE", "SPLIT_SUPPORTS", "UNIFORM")
    assert "support-split" in s
def test_successor_split_from_residual():
    s = lookup_successor("IDENTITY_OK", "PARTIAL_CORE", "MIXED", "SPLIT_SUPPORTS")
    assert "support-split" in s
def test_successor_split_before_stable_core_rules():
    s = lookup_successor("IDENTITY_OK", "SPLIT_SUPPORTS", "TRANSIENT", "STRATIFIED")
    assert "support-split" in s
    assert "OUT-stable" not in s
def test_successor_first_match_stable_transient_stratified():
    s = lookup_successor("IDENTITY_OK", "STABLE_CORE", "TRANSIENT", "STRATIFIED")
    assert "OUT-stable" in s or "Rung-1" in s
def test_mi_not_in_preempting_only():
    assert "METADATA_INSUFFICIENT" not in PREEMPTING_ONLY
    assert PREEMPTING_ONLY == ("IDENTITY_BIND_FAIL", "INSTRUMENT_OR_BIND_FAIL")
def test_mi_in_residual_slot_only():
    proj = _full_grid(L0B, MATH)
    assert residual_composite("METADATA_INSUFFICIENT", "UNIFORM") == "METADATA_INSUFFICIENT"
    r = classify_from_projections(proj)
    assert r["identity_profile"] != "METADATA_INSUFFICIENT"
def test_classification_core_projection():
    proj = _full_grid(L0B, MATH)
    r = classify_from_projections(proj)
    core = classification_core(r)
    assert core["identity_profile"] == "IDENTITY_OK"
    assert "composite_terminal" in core
    assert core["claim_boundary"]["no_mechanism_mint"] is True
def test_jaccard_raw_counts_always():
    a = {"a", "b", "c"}
    b = {"b", "c", "d"}
    j = jaccard_raw(a, b)
    assert j == {"intersect": 2, "a": 3, "b": 3, "union": 4}
@pytest.mark.parametrize(
    "pair,expected",
    [
        (("STABLE_CORE", "STABLE_CORE"), "STABLE_CORE"),
        (("STABLE_CORE", "PARTIAL_CORE"), "PARTIAL_CORE"),
        (("STABLE_CORE", "CHURNED"), "SPLIT_SUPPORTS"),
        (("PARTIAL_CORE", "STABLE_CORE"), "PARTIAL_CORE"),
        (("PARTIAL_CORE", "PARTIAL_CORE"), "PARTIAL_CORE"),
        (("PARTIAL_CORE", "CHURNED"), "SPLIT_SUPPORTS"),
        (("CHURNED", "STABLE_CORE"), "SPLIT_SUPPORTS"),
        (("CHURNED", "PARTIAL_CORE"), "SPLIT_SUPPORTS"),
        (("CHURNED", "CHURNED"), "CHURNED"),
    ],
)
def test_nine_cell_parametrized(pair, expected):
    assert NINE_CELL_TABLE[pair] == expected
    assert overlap_composite(*pair) == expected
def _mutate_view_meta(proj, support, arm, h, **fields):
    view = dict(proj[support][arm][h])
    for k, v in fields.items():
        view[k] = v
    proj[support][arm][h] = view
    return proj
@pytest.mark.parametrize(
    "support,arm,h,field,mode",
    [
        ("L0b", "package", 10, "sample_hashes", "short"),
        ("L0b", "package", 10, "sample_hashes", "long"),
        ("L0b", "out", 20, "source_buckets", "short"),
        ("math_a0", "package", 50, "source_buckets", "long"),
    ],
)
def test_identity_meta_length_mismatch_fail_closed(support, arm, h, field, mode):
    proj = _full_grid(L0B, MATH)
    base = list(proj[support][arm][h][field])
    val = base[:-1] if mode == "short" else base + ["xtra"]
    _mutate_view_meta(proj, support, arm, h, **{field: val})
    r = classify_from_projections(proj)
    assert r["identity_profile"] == "IDENTITY_BIND_FAIL"
    assert r["composite_terminal"] == "IDENTITY_BIND_FAIL"
def test_identity_known_good_still_admitted():
    assert classify_from_projections(_full_grid(L0B, MATH))["identity_profile"] == "IDENTITY_OK"
def test_residual_coverage_buckets_below_min_mi():
    buckets = ["big"] * 200 + [f"t{i}" for i in range(30)]
    lab, raw = residual_per_support_label(
        L0B, dict(zip(L0B, buckets)), set(L0B), arm="package", horizon=50
    )
    assert raw["coverage_rows_ok"] is True
    assert raw["coverage_buckets_ok"] is False
    assert raw["eligible_buckets"] == 1
    assert raw["all_nonempty_buckets"] == 31
    assert lab == "METADATA_INSUFFICIENT"
    assert raw.get("metadata_reason") == "coverage_below_min"
def test_classification_core_includes_full_q3_payload():
    proj = _full_grid(L0B, MATH, l0b_buckets=_bucketed(L0B), math_buckets=_bucketed(MATH))
    r = classify_from_projections(proj)
    core = classification_core(r)
    residual = core["residual_bucket_profile"]
    for k in ("composite", "per_support", "raw", "branch_input_surface", "report_only"):
        assert k in residual, k
    assert residual["report_only"]["L0b"]["out:N10"]["raw"]["arm"] == "out"
def test_classification_core_mutation_battery():
    proj = _full_grid(L0B, MATH, l0b_buckets=_bucketed(L0B), math_buckets=_bucketed(MATH))
    r = classify_from_projections(proj)
    base = classification_core(r)
    assert classification_core(r) == base
    paths = [
        ("residual_bucket_profile", "raw", "L0b", "support_survivors"),
        ("residual_bucket_profile", "report_only", "L0b", "out:N10", "raw", "support_rows"),
        ("residual_bucket_profile", "raw", "L0b", "arm"),
        ("residual_bucket_profile", "raw", "L0b", "coverage_rows_ok"),
    ]
    for path in paths:
        m = copy.deepcopy(r)
        cur = m
        for k in path[:-1]:
            cur = cur[k]
        leaf = path[-1]
        cur[leaf] = (cur[leaf] + 1) if isinstance(cur[leaf], int) else (
            "out" if cur[leaf] == "package" else (not cur[leaf])
        )
        assert classification_core(m) != base, path
    m = copy.deepcopy(r)
    m["residual_bucket_profile"]["report_only"]["L0b"]["out:N10"]["raw"][
        "support_rows"
    ] = 0
    assert classification_core(m) != base
    m = copy.deepcopy(r)
    m["residual_bucket_profile"]["raw"]["L0b"]["arm"] = "out"
    assert classification_core(m) != base
    m = copy.deepcopy(r)
    m["residual_bucket_profile"]["raw"]["L0b"]["coverage_rows_ok"] = not m[
        "residual_bucket_profile"
    ]["raw"]["L0b"].get("coverage_rows_ok", True)
    assert classification_core(m) != base
def _dup_l0b_proj():
    ids = list(L0B); ids[1] = ids[0]
    return _full_grid(ids, MATH)
def test_classification_core_identity_fields_bound_both_shapes():
    for proj, want in ((_full_grid(L0B, MATH), "IDENTITY_OK"), (_dup_l0b_proj(), "IDENTITY_BIND_FAIL")):
        r = classify_from_projections(proj)
        assert r["identity_profile"] == want
        c = classification_core(r)
        assert c["identity_reasons"] == r["identity_reasons"]
        assert c["identity_raw"] == r["identity_raw"]
def test_classification_core_identity_mutation_battery():
    good = classify_from_projections(_full_grid(L0B, MATH))
    base_g = classification_core(good)
    assert classification_core(good) == base_g
    m = copy.deepcopy(good)
    m["identity_raw"]["per_support"]["L0b"]["universe_size"] += 1
    assert classification_core(m) != base_g
    bad = classify_from_projections(_dup_l0b_proj())
    base_b = classification_core(bad)
    assert classification_core(bad) == base_b
    m = copy.deepcopy(bad); m["identity_reasons"] = list(m["identity_reasons"]) + ["fabricated"]
    assert classification_core(m) != base_b
    m = copy.deepcopy(bad); m["identity_raw"] = {"fabricated": True}
    assert classification_core(m) != base_b
def test_classification_core_emitted_key_coverage_mechanical():
    assert NON_AUTHORITATIVE_KEYS == frozenset()
    for r in (classify_from_projections(_full_grid(L0B, MATH)), classify_from_projections(_dup_l0b_proj())):
        core = classification_core(r)
        for k in r:
            if k in NON_AUTHORITATIVE_KEYS:
                assert k not in core
            else:
                assert k in core, k
