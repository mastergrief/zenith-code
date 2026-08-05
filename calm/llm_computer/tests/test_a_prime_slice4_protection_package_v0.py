"""Tests: A' slice-4 Rung-2 protection-package (PLAN v5, cycle-4 method)."""
from __future__ import annotations

import inspect
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from scripts.a_prime_slice4_protection_package_classifier_v0 import (
    REQUIRED_OUT_AUTHORITY,
    build_terminal_receipt,
    finalize_dual_key,
    main as classifier_main,
    mint_exclusive_run_root,
    validate_candidate_receipt,
)
from scripts.a_prime_slice4_protection_package_reducer_v0 import (
    FROZEN_OUT_TERMINAL_SHA256 as OUT_SHA,
    HORIZONS,
    PARENT_SHA_EXPECTED as PSHA,
    REQUIRED_CLAIM_BOUNDARY,
    REQUIRED_PACKAGE_BINDING,
    REQUIRED_RUN_GEOMETRY as GEOM,
    START_SURVIVOR_DENOMINATORS as SURV,
    bind_and_classify_package,
    check_package_binding,
    classify_from_counts,
    extract_final_strict_count,
)

F = {
    "CLEARS__BOTH_HELD": {10: (220, 1200), 20: (210, 1150), 50: (205, 1140)},
    "PREVENTS__BOTH_HELD": {10: (150, 900), 20: (210, 1150), 50: (205, 1140)},
    "NULL__NO_MIX": {10: (122, 816), 20: (15, 67), 50: (15, 47)},
    "DELAYS__L0B_ONLY": {10: (220, 1200), 20: (210, 100), 50: (205, 90)},
    "PREVENTS__MATH_ONLY": {10: (150, 900), 20: (115, 1254), 50: (110, 1250)},
    "OTHER__L0B_ONLY": {10: (220, 900), 20: (210, 100), 50: (200, 90)},
    "OTHER__REDISTRIBUTED": {10: (230, 708), 20: (15, 67), 50: (15, 47)},
    "DELAYS__NO_MIX": {10: (220, 1200), 20: (20, 100), 50: (18, 80)},
    "DELAYS_edge_N20_580__NO_MIX": {10: (220, 1117), 20: (20, 560), 50: (18, 500)},
    "OTHER_not_delays_N20_582__NO_MIX": {10: (220, 1117), 20: (20, 562), 50: (18, 500)},
}
TERM = {k: k for k in F}
TERM["DELAYS_edge_N20_580__NO_MIX"] = "DELAYS__NO_MIX"
TERM["OTHER_not_delays_N20_582__NO_MIX"] = "OTHER__NO_MIX"
S1 = "Rung-1 densify before mechanism (MIXED-support precedence override)"
SD = "DECOMPOSE replay-veto vs PC-veto (NOT Rung-1 by default)"
SDEL = "DECOMPOSE or tighter package; Rung-1 optional only if geometry ambiguity named"
SUCC = {
    "CLEARS__BOTH_HELD": SD, "PREVENTS__BOTH_HELD": SD, "NULL__NO_MIX": "Rung-1 densify next",
    "DELAYS__L0B_ONLY": S1, "PREVENTS__MATH_ONLY": S1, "OTHER__L0B_ONLY": S1,
    "OTHER__REDISTRIBUTED": S1, "DELAYS__NO_MIX": SDEL,
    "DELAYS_edge_N20_580__NO_MIX": SDEL,
    "OTHER_not_delays_N20_582__NO_MIX": "classify residual; no mechanism mint",
}
GOOD_OUT = {"branch": "NONMONOTONE_OR_MULTI_CLIFF", "terminal_authority": "manifest+marker", "synthetic": False}


def _c(t):
    return {n: {"L0b": a, "math_a0": b} for n, (a, b) in t.items()}


def _bind(**o):
    b = {
        "requested_supports": ["L0b", "math_a0"], "replay_ce_veto": True, "pc_aux_enabled": True,
        "parent_consistency_weight": 1.0, "pc_aux_mode": "veto",
        "prior_batches_fed_to_bounded_steps": True, "target_parent_kl": False,
        "target_rows_excluded_from_pc": True,
    }
    b.update(o)
    return b


def _rec(n, c, *, gh=50, parent_ok=True, binding=None, parent_before=PSHA, parent_after=PSHA,
         geometry=None, baseline=None, steps=None):
    b = _bind() if binding is None else binding
    g = dict(GEOM) if geometry is None else geometry
    base = baseline or {"L0b": "230/230", "math_a0": "1254/1255"}
    return {
        "steps_completed": n if steps is None else steps,
        "parent_hash_unchanged": parent_ok, "parent_hash_before": parent_before,
        "parent_hash_after": parent_after,
        "device": g.get("device", GEOM["device"]),
        "eligible_scope": g.get("eligible_scope", GEOM["eligible_scope"]),
        "local_selection_ordering_seed": g.get(
            "local_selection_ordering_seed", GEOM["local_selection_ordering_seed"]),
        "banked_pt_mutated": g.get("banked_pt_mutated", GEOM["banked_pt_mutated"]),
        "step_reports": {str(s): {"global_horizon": gh} for s in range(1, n + 1)},
        "b2_retention": {"schema": "b2_retained_support/v0", "enabled": True, **b},
        "prior_audit": {
            "prior_batches_fed_to_bounded_steps": b.get("prior_batches_fed_to_bounded_steps"),
            "target_parent_kl": b.get("target_parent_kl"),
            "deltas": {
                "L0b": {"parent_baseline_vs_final": {
                    "baseline_strict_exact": base["L0b"],
                    "final_strict_exact": f"{c['L0b']}/230"}},
                "math_a0": {"parent_baseline_vs_final": {
                    "baseline_strict_exact": base["math_a0"],
                    "final_strict_exact": f"{c['math_a0']}/1255"}},
            },
        },
    }


def _pack(table, **kw):
    ct = _c(table)
    return {n: json.dumps(_rec(n, ct[n], **kw), sort_keys=True).encode() for n in HORIZONS}


def _bind_call(table, **kw):
    return bind_and_classify_package(
        package_receipt_bytes_by_n=_pack(table, **kw), out_terminal=GOOD_OUT,
        out_terminal_sha256=OUT_SHA, require_frozen_out_terminal_sha=True)


def _cls(table):
    c = dict(classify_from_counts(_c(table)))
    shas = {f"package/N{n}": ("ab" * 32) for n in HORIZONS}
    shas["out/terminal"] = OUT_SHA
    c.update(source_shas=shas, package_binding=_bind(), out_authority=dict(REQUIRED_OUT_AUTHORITY),
             instrument_fail=False, reasons=[])
    return c, shas


def _fin(root, c, shas, **kw):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = finalize_dual_key(root, c, source_shas=shas, **kw)
    return rc, buf.getvalue()


def _assert_inst(r, needle):
    assert r["branch"] == "INSTRUMENT_OR_BIND_FAIL" and any(needle in x for x in r["reasons"])


def _assert_hostile(rc, out, wrap=2):
    assert rc == wrap
    assert not any(ln.startswith("PACKET_TERMINAL ") for ln in out.splitlines())
    assert any(ln.startswith("INCOMPLETE_FINALIZATION") for ln in out.splitlines())
    assert any(ln == f"WRAPPER_RC {wrap}" for ln in out.splitlines())


@pytest.mark.parametrize("name,table", list(F.items()))
def test_positive_fixture(name, table):
    r = classify_from_counts(_c(table))
    assert r["branch"] == TERM[name]
    assert r["successor"] == SUCC[name]
    b0 = SURV["L0b"]
    assert r["N20_own_loss"]["L0b"] == pytest.approx((b0 - table[20][0]) / b0)
    assert r["claim_boundary"] == REQUIRED_CLAIM_BOUNDARY


@pytest.mark.parametrize("name,eff,sup", [
    ("DELAYS__L0B_ONLY", "DELAYS", "L0B_ONLY"),
    ("PREVENTS__MATH_ONLY", "PREVENTS", "MATH_ONLY"),
    ("OTHER__REDISTRIBUTED", "OTHER", "REDISTRIBUTED"),
])
def test_mixed_rung1(name, eff, sup):
    r = classify_from_counts(_c(F[name]))
    assert (r["package_effect_profile"], r["support_response_profile"], r["successor"]) == (eff, sup, S1)


def test_edges_and_bind_ok():
    assert classify_from_counts(_c(F["DELAYS_edge_N20_580__NO_MIX"]))["branch"] == "DELAYS__NO_MIX"
    assert classify_from_counts(_c(F["OTHER_not_delays_N20_582__NO_MIX"]))["lift_N20"] == 500
    r = _bind_call(F["PREVENTS__BOTH_HELD"])
    assert r["instrument_fail"] is False and r["source_shas"]["out/terminal"] == OUT_SHA
    assert check_package_binding(r["package_binding"]) == []
    assert extract_final_strict_count(_rec(10, {"L0b": 122, "math_a0": 816}), "L0b") == 122


@pytest.mark.parametrize("field,value,needle", [
    ("requested_supports", ["L0b"], "requested_supports"),
    ("replay_ce_veto", False, "replay_ce_veto"), ("pc_aux_enabled", False, "pc_aux_enabled"),
    ("parent_consistency_weight", 0.0, "parent_consistency_weight"),
    ("pc_aux_mode", "telemetry", "pc_aux_mode"),
    ("prior_batches_fed_to_bounded_steps", False, "prior_batches_fed_to_bounded_steps"),
    ("target_parent_kl", True, "target_parent_kl"),
    ("target_rows_excluded_from_pc", False, "target_rows_excluded_from_pc"),
    ("replay_ce_veto", 1, "replay_ce_veto"), ("target_parent_kl", 0, "target_parent_kl"),
    ("parent_consistency_weight", "1.0", "parent_consistency_weight"),
    ("pc_aux_enabled", 1, "pc_aux_enabled"),
])
def test_binding_negatives(field, value, needle):
    _assert_inst(_bind_call(F["NULL__NO_MIX"], binding=_bind(**{field: value})), needle)


@pytest.mark.parametrize("gk,bv,needle", [
    ("device", "cpu", "device"), ("eligible_scope", "all", "eligible_scope"),
    ("local_selection_ordering_seed", 0, "local_selection_ordering_seed"),
    ("banked_pt_mutated", True, "banked_pt_mutated"),
    ("banked_pt_mutated", 0, "banked_pt_mutated"),
    ("local_selection_ordering_seed", True, "local_selection_ordering_seed"),
    ("device", 0, "device"),
])
def test_geometry_negatives(gk, bv, needle):
    g = dict(GEOM); g[gk] = bv
    _assert_inst(_bind_call(F["NULL__NO_MIX"], geometry=g), needle)


@pytest.mark.parametrize("kw,needle", [
    ({"gh": 49}, "global_horizon"),
    ({"gh": 50.0}, "global_horizon"),
    ({"parent_before": PSHA, "parent_after": "0" * 64}, "parent_hash_after"),
    ({"parent_after": None}, "parent_hash_after"),
    ({"steps": 10.0}, "steps_completed"),
])
def test_parent_horizon_steps_negatives(kw, needle):
    if "steps" in kw:
        r = bind_and_classify_package(
            package_receipt_bytes_by_n=_pack(F["NULL__NO_MIX"], steps=kw["steps"]),
            out_terminal=GOOD_OUT, out_terminal_sha256=OUT_SHA,
            require_frozen_out_terminal_sha=True)
        _assert_inst(r, needle)
    else:
        _assert_inst(_bind_call(F["NULL__NO_MIX"], **kw), needle)


@pytest.mark.parametrize("l0b,math,needle", [
    (-1, 67, "count_extract"),  # negative fails digit parse before domain
    (231, 67, "L0b_out_of_domain"),
    (15, 1256, "math_a0_out_of_domain"),
])
def test_count_domain(l0b, math, needle):
    bad = {10: (122, 816), 20: (l0b, math), 50: (15, 47)}
    with pytest.raises(ValueError):
        classify_from_counts(_c(bad))
    _assert_inst(_bind_call(bad), needle)


def test_baseline_and_out_pin():
    r = _bind_call(F["NULL__NO_MIX"], baseline={"L0b": "0/999", "math_a0": "999/999"})
    _assert_inst(r, "baseline_mismatch")
    r2 = bind_and_classify_package(
        package_receipt_bytes_by_n=_pack(F["NULL__NO_MIX"]),
        out_terminal=None, out_terminal_sha256=None, require_frozen_out_terminal_sha=True)
    assert any("out_terminal_missing" in x for x in r2["reasons"])


def test_dual_key_positive(tmp_path: Path):
    c, shas = _cls(F["PREVENTS__BOTH_HELD"])
    rc, out = _fin(tmp_path / "ok", c, shas)
    assert rc == 0 and "PACKET_TERMINAL PREVENTS__BOTH_HELD" in out
    rec = json.loads((tmp_path / "ok" / "terminal_receipt.json").read_text())
    assert rec["source_shas"]["out/terminal"] == OUT_SHA
    c2, s2 = _cls(F["CLEARS__BOTH_HELD"])
    _assert_hostile(*_fin(tmp_path / "pp", c2, s2, inject_postpub_fail=True), wrap=4)


def _g2_muts():
    def bare_instrument(r):
        r["branch"] = "INSTRUMENT_OR_BIND_FAIL"
        r["package_effect_profile"] = None
        r["support_response_profile"] = None
        r["successor"] = "instrument repair only"
        r["classification"] = {"instrument_fail": True}

    def instr_empty_shas_no_auth(r):
        # co_lead hole 1: instrument with source_shas={} + out_authority=None
        r["branch"] = "INSTRUMENT_OR_BIND_FAIL"
        r["package_effect_profile"] = None
        r["support_response_profile"] = None
        r["successor"] = "instrument repair only"
        r["source_shas"] = {}
        r["classification"] = {
            "branch": "INSTRUMENT_OR_BIND_FAIL",
            "instrument_fail": True,
            "reasons": ["x"],
            "source_shas": {},
            "out_authority": None,
            "successor": "instrument repair only",
            "claim_boundary": dict(REQUIRED_CLAIM_BOUNDARY),
        }

    def emb_sha_flip(r):
        # co_lead hole 2: embedded package/N10 flip vs intact top-level
        emb = dict(r["classification"]["source_shas"])
        emb["package/N10"] = "0" * 64
        r["classification"]["source_shas"] = emb

    def reasons_bad(r):
        # co_lead hole 3: reasons=["bad"] on attribution
        r["classification"]["reasons"] = ["bad"]

    def succ_flip(r):
        # co_lead hole 4: top-level successor flipped to Rung-1
        r["successor"] = "Rung-1 densify before mechanism (MIXED-support precedence override)"

    return {
        "g2_str_l0b": lambda r: r["classification"]["counts"]["20"].__setitem__("L0b", "210"),
        "g2_float_l0b": lambda r: r["classification"]["counts"]["20"].__setitem__("L0b", 210.0),
        "g2_out_totals": lambda r: r["classification"].__setitem__(
            "out_totals", {"10": 999, "20": 999, "50": 999}),
        "g2_out_support": lambda r: r["classification"].__setitem__(
            "out_support_counts", {str(n): {"L0b": 999, "math_a0": 999} for n in (10, 20, 50)}),
        "g2_bare_instrument": bare_instrument,
        "c5_instr_empty_shas": instr_empty_shas_no_auth,
        "c5_emb_sha_flip": emb_sha_flip,
        "c5_reasons_bad": reasons_bad,
        "c5_succ_flip": succ_flip,
        "del_cb": lambda r: r.pop("claim_boundary", None),
        "empty_bind": lambda r: r["classification"].__setitem__("package_binding", {}),
        "veto_int1": lambda r: r["classification"].__setitem__(
            "package_binding", _bind(replay_ce_veto=1)),
        "weight_str": lambda r: r["classification"].__setitem__(
            "package_binding", _bind(parent_consistency_weight="1.0")),
        "drop_out_sha": lambda r: r["source_shas"].pop("out/terminal", None),
        "flip_out_sha": lambda r: r["source_shas"].__setitem__("out/terminal", "0" * 64),
        "drop_out_auth": lambda r: r["classification"].__setitem__("out_authority", None),
    }


@pytest.mark.parametrize("label", list(_g2_muts()))
def test_finalize_gate2_and_hostiles(tmp_path: Path, label):
    c, shas = _cls(F["CLEARS__BOTH_HELD"])
    rc, out = _fin(tmp_path / f"h_{label}", c, shas, inject_receipt_mutator=_g2_muts()[label])
    _assert_hostile(rc, out)


def test_run_root_e2e_cli(tmp_path: Path):
    c, shas = _cls(F["NULL__NO_MIX"])
    ne = tmp_path / "ne"; ne.mkdir(); (ne / "x").write_text("x")
    assert mint_exclusive_run_root(ne)[0] is False
    _assert_hostile(*_fin(ne, c, shas))
    r = _bind_call(F["OTHER__REDISTRIBUTED"])
    rc, out = _fin(tmp_path / "e2e", r, r["source_shas"])
    assert rc == 0 and "PACKET_TERMINAL OTHER__REDISTRIBUTED" in out
    import scripts.a_prime_slice4_protection_package_classifier_v0 as mod
    assert "exclusive_mint" not in inspect.signature(finalize_dual_key).parameters
    assert "skip_frozen" not in Path(mod.__file__).read_text()
    rbytes = _pack(F["NULL__NO_MIX"])
    for n, raw in rbytes.items():
        (tmp_path / f"pkg_N{n}.json").write_bytes(raw)
    argv = ["--run-root", str(tmp_path / "no_out")]
    for n in HORIZONS:
        argv.extend(["--package-receipt", f"{n}={tmp_path / f'pkg_N{n}.json'}"])
    outb, err = io.StringIO(), io.StringIO()
    with pytest.raises(SystemExit) as ei:
        with redirect_stdout(outb), redirect_stderr(err):
            classifier_main(argv)
    assert ei.value.code != 0 and "PACKET_TERMINAL" not in outb.getvalue() + err.getvalue()
    bad = tmp_path / "bad.json"; bad.write_text("{invalid")
    argv2 = ["--run-root", str(tmp_path / "bad_out"), "--out-terminal-receipt", str(bad)]
    for n in HORIZONS:
        argv2.extend(["--package-receipt", f"{n}={tmp_path / f'pkg_N{n}.json'}"])
    out2 = io.StringIO()
    with redirect_stdout(out2):
        assert classifier_main(argv2) == 2
    assert "INCOMPLETE_FINALIZATION" in out2.getvalue() and "PACKET_TERMINAL" not in out2.getvalue()


def test_known_good_instrument_envelope(tmp_path: Path):
    """Instrument terminal with real bound shas + out_authority validates and mints."""
    r = _bind_call(F["NULL__NO_MIX"], binding=_bind(pc_aux_mode="telemetry"))
    assert r["branch"] == "INSTRUMENT_OR_BIND_FAIL"
    assert r["out_authority"] is not None
    assert r["source_shas"].get("out/terminal") == OUT_SHA
    assert r["source_shas"]
    rc, out = _fin(tmp_path / "instr_ok", r, r["source_shas"])
    assert rc == 0
    assert "PACKET_TERMINAL INSTRUMENT_OR_BIND_FAIL" in out


@pytest.mark.parametrize("label,mut", [
    ("c6_reasons", lambda r: r["classification"].__setitem__("reasons", ["fabricated repair"])),
    ("c6_binding", lambda r: r["classification"].__setitem__("package_binding", {"arbitrary": True})),
    ("c6_counts", lambda r: r["classification"].__setitem__(
        "counts", {"10": {"L0b": "fake", "math_a0": 999}})),
])
def test_instrument_snapshot_hostiles(tmp_path: Path, label, mut):
    """co_lead cycle-6: mutate production-shaped instrument → zero markers."""
    r = _bind_call(F["NULL__NO_MIX"], binding=_bind(pc_aux_mode="telemetry"))
    assert r["branch"] == "INSTRUMENT_OR_BIND_FAIL"
    rc, out = _fin(tmp_path / f"h_{label}", r, r["source_shas"], inject_receipt_mutator=mut)
    _assert_hostile(rc, out)


def test_constants_and_validate():
    assert REQUIRED_PACKAGE_BINDING["pc_aux_mode"] == "veto"
    assert OUT_SHA.startswith("f587cee0") and GEOM["device"] == "cuda:0"
    c, shas = _cls(F["DELAYS__NO_MIX"])
    ok, reason = validate_candidate_receipt(
        build_terminal_receipt(c, run_root=Path("/tmp/x"), source_shas=shas), source_shas=shas)
    assert ok, reason
