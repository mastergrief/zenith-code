"""Tests: A′ slice-4 Rung-0 support-loss attribution (cycle-4; ≤500 lines)."""
from __future__ import annotations

import inspect
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from scripts.a_prime_slice4_support_loss_attribution_classifier_v0 import (
    build_terminal_receipt,
    finalize_dual_key,
    main as classifier_main,
    mint_exclusive_run_root,
    validate_candidate_receipt,
)
from scripts.a_prime_slice4_support_loss_attribution_reducer_v0 import (
    FROZEN_TERMINAL_SHA256,
    HORIZONS,
    START_SURVIVOR_DENOMINATORS,
    SUPPORT_ROWS_EXPECTED,
    bind_and_extract,
    classify_from_counts,
    extract_final_strict_count,
    sha256_hex,
)

# plan v2 integer-count fixtures (rates derived only)
F = {
    "CO_COLLAPSE__CLIFF_SPECIFIC": {
        1: (230, 1254), 5: (230, 1254), 10: (120, 900), 20: (20, 100), 35: (20, 100), 50: (20, 100)
    },
    "CONTAINED__L0B_ENRICHED_BOTH": {
        1: (230, 1254), 5: (230, 1254), 10: (160, 1100), 20: (100, 980), 35: (100, 980), 50: (100, 980)
    },
    "CONTAINED__MATH_A0_ENRICHED_BOTH": {
        1: (230, 1254), 5: (230, 1254), 10: (200, 900), 20: (170, 600), 35: (170, 600), 50: (170, 600)
    },
    "CONTAINED__BALANCED": {
        1: (230, 1254), 5: (230, 1254), 10: (170, 930), 20: (120, 650), 35: (120, 650), 50: (120, 650)
    },
    "CONTAINED__SUB_THRESHOLD": {
        1: (230, 1254), 5: (230, 1254), 10: (225, 1240), 20: (220, 1225), 35: (220, 1225), 50: (220, 1225)
    },
    "PARTIAL__CLIFF_SPECIFIC": {
        1: (230, 1254), 5: (230, 1254), 10: (120, 1100), 20: (15, 400), 35: (15, 400), 50: (15, 400)
    },
}


def _counts(table):
    return {n: {"L0b": t[0], "math_a0": t[1]} for n, t in table.items()}


def _receipt(n, c, *, gh=50, parent_ok=True, replay_pc="OUT", direct_kl=False, pbf=False):
    return {
        "steps_completed": n,
        "parent_hash_unchanged": parent_ok,
        "parent_hash_before": "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec",
        "parent_hash_after": "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec",
        "step_reports": {str(s): {"global_horizon": gh} for s in range(1, n + 1)},
        "prior_audit": {
            "replay_pc": replay_pc,
            "direct_kl": direct_kl,
            "prior_batches_fed_to_bounded_steps": pbf,
            "deltas": {
                "L0b": {"parent_baseline_vs_final": {
                    "baseline_strict_exact": "230/230",
                    "final_strict_exact": f"{c['L0b']}/230",
                }},
                "math_a0": {"parent_baseline_vs_final": {
                    "baseline_strict_exact": "1254/1255",
                    "final_strict_exact": f"{c['math_a0']}/1255",
                }},
            },
        },
    }


def _pack(table, **kw):
    ct = _counts(table)
    rbytes = {n: json.dumps(_receipt(n, ct[n], **kw), sort_keys=True).encode() for n in HORIZONS}
    shas = {f"input/N{n}": sha256_hex(rbytes[n]) for n in HORIZONS}
    term = {
        "branch": "NONMONOTONE_OR_MULTI_CLIFF",
        "terminal_authority": "manifest+marker",
        "synthetic": False,
        "source_shas": shas,
        "classification": {
            "branch": "NONMONOTONE_OR_MULTI_CLIFF",
            "details": {"counts_by_n": {str(n): ct[n]["L0b"] + ct[n]["math_a0"] for n in HORIZONS}},
        },
    }
    return term, json.dumps(term, sort_keys=True).encode(), rbytes, shas


def _cls(table):
    c = classify_from_counts(_counts(table))
    shas = {f"input/N{n}": ("ab" * 32) for n in HORIZONS}
    c = dict(c)
    c["source_shas"] = shas
    c["instrument_fail"] = False
    c["reasons"] = []
    return c, shas


def _fin(root, classification, source_shas, **kw):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = finalize_dual_key(root, classification, source_shas=source_shas, **kw)
    return rc, buf.getvalue()


def _lines(out, prefix):
    return [ln for ln in out.splitlines() if ln.startswith(prefix)]


def _assert_hostile(rc, out, *, wrap=2):
    assert rc == wrap
    assert _lines(out, "PACKET_TERMINAL ") == []
    assert any(ln.startswith("INCOMPLETE_FINALIZATION") for ln in out.splitlines())
    assert any(ln == f"WRAPPER_RC {wrap}" for ln in out.splitlines())


@pytest.mark.parametrize("branch,table", list(F.items()))
def test_positive_fixture_from_integer_counts(branch, table):
    r = classify_from_counts(_counts(table))
    assert r["branch"] == branch
    assert f'{r["endpoint_profile"]}__{r["cliff_profile"]}' == branch
    b0, bm = START_SURVIVOR_DENOMINATORS["L0b"], START_SURVIVOR_DENOMINATORS["math_a0"]
    assert r["endpoint"]["L0b_own_loss"] == pytest.approx((b0 - table[20][0]) / b0)
    assert r["endpoint"]["math_a0_own_loss"] == pytest.approx((bm - table[20][1]) / bm)
    assert r["survivor_denominators"] == {"L0b": 230, "math_a0": 1254}
    assert SUPPORT_ROWS_EXPECTED["math_a0"] == 1255


def test_refuses_math_denom_1255():
    with pytest.raises(ValueError, match="1255"):
        classify_from_counts(_counts(F["CONTAINED__SUB_THRESHOLD"]), survivor_denoms={"L0b": 230, "math_a0": 1255})


def test_bind_parses_hashed_raw_only():
    term, traw, rbytes, shas = _pack(F["CONTAINED__BALANCED"])
    r = bind_and_extract(terminal=term, terminal_sha256=sha256_hex(traw), receipt_bytes_by_n=rbytes, require_frozen_terminal_sha=False)
    assert r["instrument_fail"] is False and r["branch"] == "CONTAINED__BALANCED" and r["source_shas"] == shas


@pytest.mark.parametrize(
    "mut,needle",
    [
        (lambda t, s: {**t, "source_shas": {**s, "input/N5": "0" * 64}}, "source_sha_mismatch"),
        (lambda t, s: {**t, "terminal_authority": "receipt-only"}, "terminal_authority"),
        (lambda t, s: {**t, "synthetic": True}, "synthetic"),
        (
            lambda t, s: {
                **t,
                "classification": {"branch": t["branch"], "details": {"counts_by_n": {str(n): 0 for n in HORIZONS}}},
            },
            "compose_fail",
        ),
    ],
)
def test_bind_instrument_negatives(mut, needle):
    term, traw, rbytes, shas = _pack(F["CONTAINED__SUB_THRESHOLD"])
    term = mut(term, shas)
    r = bind_and_extract(
        terminal=term,
        terminal_sha256=sha256_hex(json.dumps(term, sort_keys=True).encode()),
        receipt_bytes_by_n=rbytes,
        require_frozen_terminal_sha=False,
    )
    assert r["branch"] == "INSTRUMENT_OR_BIND_FAIL"
    assert any(needle in x for x in r["reasons"])


def test_bind_prior_batches_and_horizon_and_survivor():
    term, traw, rbytes, _ = _pack(F["CONTAINED__BALANCED"], pbf=True)
    r = bind_and_extract(terminal=term, terminal_sha256=sha256_hex(traw), receipt_bytes_by_n=rbytes, require_frozen_terminal_sha=False)
    assert any("prior_batches_fed_to_bounded_steps" in x for x in r["reasons"])

    term, traw, rbytes, _ = _pack(F["CONTAINED__SUB_THRESHOLD"], gh=49)
    r = bind_and_extract(terminal=term, terminal_sha256=sha256_hex(traw), receipt_bytes_by_n=rbytes, require_frozen_terminal_sha=False)
    assert any("global_horizon" in x for x in r["reasons"])

    bad = dict(F["CONTAINED__SUB_THRESHOLD"])
    bad[1] = (230, 1255)
    term, traw, rbytes, _ = _pack(bad)
    r = bind_and_extract(terminal=term, terminal_sha256=sha256_hex(traw), receipt_bytes_by_n=rbytes, require_frozen_terminal_sha=False)
    assert any("N1_survivor_math_a0" in x for x in r["reasons"])


def test_frozen_terminal_sha_enforced():
    term, traw, rbytes, _ = _pack(F["CONTAINED__BALANCED"])
    r = bind_and_extract(terminal=term, terminal_sha256=sha256_hex(traw), receipt_bytes_by_n=rbytes, require_frozen_terminal_sha=True)
    assert r["branch"] == "INSTRUMENT_OR_BIND_FAIL" and any("terminal_sha_mismatch" in x for x in r["reasons"])


def test_cli_no_skip_flag_and_no_exclusive_mint_param():
    import scripts.a_prime_slice4_support_loss_attribution_classifier_v0 as mod

    src = Path(mod.__file__).read_text()
    assert "skip_frozen_terminal_sha" not in src and "exclusive_mint" not in src
    assert "if exclusive_mint" not in src
    assert "exclusive_mint" not in inspect.signature(finalize_dual_key).parameters
    buf = io.StringIO()
    with pytest.raises(SystemExit) as ei:
        with redirect_stdout(buf):
            classifier_main(["--help"])
    assert ei.value.code == 0 and "skip-frozen" not in buf.getvalue()


def test_factorization_and_extract():
    r = classify_from_counts(_counts(F["CO_COLLAPSE__CLIFF_SPECIFIC"]))
    assert r["endpoint_profile"] == "CO_COLLAPSE" and r["cliff_profile"] == "CLIFF_SPECIFIC"
    rec = _receipt(10, {"L0b": 122, "math_a0": 816})
    assert extract_final_strict_count(rec, "L0b") == 122


def test_dual_key_positive_single_marker(tmp_path: Path):
    c, shas = _cls(F["CONTAINED__BALANCED"])
    rc, out = _fin(tmp_path / "ok", c, shas)
    assert rc == 0
    assert _lines(out, "PACKET_TERMINAL ") == ["PACKET_TERMINAL CONTAINED__BALANCED"]
    assert _lines(out, "WRAPPER_RC ") == ["WRAPPER_RC 0"]
    assert "INCOMPLETE_FINALIZATION" not in out


def test_dual_key_hostile_postpub(tmp_path: Path):
    c, shas = _cls(F["CONTAINED__BALANCED"])
    rc, out = _fin(tmp_path / "h", c, shas, inject_postpub_fail=True)
    _assert_hostile(rc, out, wrap=4)


def test_finalize_top_claim_boundary_deleted(tmp_path: Path):
    """Cycle-6: missing top-level claim_boundary → fail-closed, zero markers."""
    c, shas = _cls(F["CONTAINED__BALANCED"])

    def _del_boundary(receipt):
        del receipt["claim_boundary"]

    rc, out = _fin(tmp_path / "del_cb", c, shas, inject_receipt_mutator=_del_boundary)
    _assert_hostile(rc, out)
    assert any("claim_boundary" in ln for ln in out.splitlines())


def test_finalize_top_claim_boundary_flipped(tmp_path: Path):
    """Cycle-6: top-level flipped while embedded stays true → fail-closed."""
    c, shas = _cls(F["CONTAINED__BALANCED"])

    def _flip_boundary(receipt):
        receipt["claim_boundary"] = {
            "attribution_only": False,
            "pre_cause": False,
            "pre_carrier": False,
            "absolute_share_not_branch_input": True,
        }

    rc, out = _fin(tmp_path / "flip_cb", c, shas, inject_receipt_mutator=_flip_boundary)
    _assert_hostile(rc, out)
    assert any("claim_boundary" in ln for ln in out.splitlines())


def test_run_root_nonempty_and_empty_collision(tmp_path: Path):
    c, shas = _cls(F["CONTAINED__SUB_THRESHOLD"])
    nonempty = tmp_path / "ne"
    nonempty.mkdir()
    (nonempty / "x").write_text("x")
    assert mint_exclusive_run_root(nonempty)[0] is False
    rc, out = _fin(nonempty, c, shas)
    _assert_hostile(rc, out)
    assert any("run_root_exists" in ln for ln in out.splitlines())

    empty = tmp_path / "empty"
    empty.mkdir()
    ok, reason = mint_exclusive_run_root(empty)
    assert ok is False and "run_root_exists" in reason
    rc, out = _fin(empty, c, shas)
    _assert_hostile(rc, out)
    assert any("run_root_exists" in ln for ln in out.splitlines())


def test_candidate_validator_honest_and_surface_negatives():
    c, shas = _cls(F["CONTAINED__BALANCED"])
    receipt = build_terminal_receipt(c, run_root=Path("/tmp/x"), source_shas=shas)
    assert validate_candidate_receipt(receipt, source_shas=shas)[0] is True
    assert receipt["claim_boundary"] == {
        "attribution_only": True,
        "pre_cause": True,
        "pre_carrier": True,
        "absolute_share_not_branch_input": True,
    }
    for mut in (
        lambda r: {**r, "branch": "CO_COLLAPSE__BALANCED"},
        lambda r: {**r, "endpoint_profile": "NOPE"},
        lambda r: {**r, "schema": "wrong"},
        lambda r: {**r, "source_shas": {"input/N1": shas["input/N1"]}},
        lambda r: {k: v for k, v in r.items() if k != "claim_boundary"},
        lambda r: {
            **r,
            "claim_boundary": {
                "attribution_only": False,
                "pre_cause": False,
                "pre_carrier": False,
                "absolute_share_not_branch_input": True,
            },
        },
    ):
        assert validate_candidate_receipt(mut(receipt), source_shas=shas)[0] is False


def _mutate_cls(receipt, mut_fn):
    bad = dict(receipt)
    cls = dict(bad["classification"])
    # deep-copy nested containers we may mutate
    if "cliffs" in cls:
        cls["cliffs"] = [dict(x) for x in cls["cliffs"]]
    if "counts" in cls:
        cls["counts"] = {k: dict(v) for k, v in cls["counts"].items()}
    if "endpoint" in cls:
        cls["endpoint"] = dict(cls["endpoint"])
    if "claim_boundary" in cls:
        cls["claim_boundary"] = dict(cls["claim_boundary"])
    if "support_rows_expected" in cls:
        cls["support_rows_expected"] = dict(cls["support_rows_expected"])
    mut_fn(cls)
    bad["classification"] = cls
    return bad


def _cliff0(cls, key, value):
    cliffs = [dict(x) for x in cls["cliffs"]]
    cliffs[0] = dict(cliffs[0])
    cliffs[0][key] = value
    cls["cliffs"] = cliffs


def _mut_claim(c):
    c["claim_boundary"] = {
        "attribution_only": False,
        "pre_cause": False,
        "pre_carrier": False,
    }


def _mut_del35(c):
    del c["counts"]["35"]


def _mut_cliffs_empty(c):
    c["cliffs"] = []


def _mut_cliffs_third(c):
    c["cliffs"] = list(c["cliffs"]) + [dict(c["cliffs"][0])]


def _mut_branch(c):
    c["branch"] = "TAMPERED__BRANCH"


def _mut_endpoint(c):
    c["endpoint"] = {**c["endpoint"], "L0b_own_loss": 0.0}


def _mut_support(c):
    c["support_rows_expected"] = {"L0b": 230, "math_a0": 9999}


@pytest.mark.parametrize(
    "name,mut",
    [
        ("abs_share_math_a0", lambda c: _cliff0(c, "abs_share_math_a0", 0.0)),
        ("l0b_enriched", lambda c: _cliff0(c, "l0b_enriched", not c["cliffs"][0]["l0b_enriched"])),
        ("claim_boundary", _mut_claim),
        ("counts_35_deleted", _mut_del35),
        ("cliffs_empty", _mut_cliffs_empty),
        ("cliffs_third", _mut_cliffs_third),
        ("embedded_branch", _mut_branch),
        ("endpoint_loss", _mut_endpoint),
        ("rate_tamper", lambda c: _cliff0(c, "L0b_own_baseline_loss_rate", 0.999)),
        ("support_rows", _mut_support),
    ],
)
def test_candidate_core_projection_mutation_fail_closed(name, mut, tmp_path: Path):
    """Cycle-5: ANY projected core field mutation fails closed (no field-list holes)."""
    c, shas = _cls(F["CONTAINED__BALANCED"])
    receipt = build_terminal_receipt(c, run_root=Path("/tmp/x"), source_shas=shas)
    assert validate_candidate_receipt(receipt, source_shas=shas)[0] is True
    bad = _mutate_cls(receipt, mut)
    ok, reason = validate_candidate_receipt(bad, source_shas=shas)
    assert ok is False, (name, reason)
    assert reason
    # finalize path: embed same mutated classification → zero PACKET_TERMINAL
    c_bad = dict(c)
    if "cliffs" in c_bad:
        c_bad["cliffs"] = [dict(x) for x in c_bad["cliffs"]]
    if "counts" in c_bad:
        c_bad["counts"] = {k: dict(v) for k, v in c_bad["counts"].items()}
    if "endpoint" in c_bad:
        c_bad["endpoint"] = dict(c_bad["endpoint"])
    if "claim_boundary" in c_bad:
        c_bad["claim_boundary"] = dict(c_bad["claim_boundary"])
    if "support_rows_expected" in c_bad:
        c_bad["support_rows_expected"] = dict(c_bad["support_rows_expected"])
    mut(c_bad)
    rc, out = _fin(tmp_path / f"mut_{name}", c_bad, shas)
    _assert_hostile(rc, out)


def test_bind_success_partial():
    term, traw, rbytes, shas = _pack(F["PARTIAL__CLIFF_SPECIFIC"])
    r = bind_and_extract(terminal=term, terminal_sha256=sha256_hex(traw), receipt_bytes_by_n=rbytes, require_frozen_terminal_sha=False)
    assert r["instrument_fail"] is False and r["branch"] == "PARTIAL__CLIFF_SPECIFIC"


LIVE = Path("/home/gabe/claw-code-creditdir/a_prime_slice3_onset/run_fb2fcec5")


@pytest.mark.skipif(not LIVE.is_dir(), reason="live run_fb2fcec5 absent")
def test_live_run_fb2fcec5_expected_composite():
    traw = (LIVE / "terminal_receipt.json").read_bytes()
    term = json.loads(traw)
    assert sha256_hex(traw) == FROZEN_TERMINAL_SHA256
    rbytes = {n: (LIVE / f"horizon_N{n}/c2p1_impl_cpu/receipt.json").read_bytes() for n in HORIZONS}
    r = bind_and_extract(terminal=term, terminal_sha256=sha256_hex(traw), receipt_bytes_by_n=rbytes, require_frozen_terminal_sha=True)
    assert r["instrument_fail"] is False, r.get("reasons")
    assert r["branch"] == "CO_COLLAPSE__CLIFF_SPECIFIC"
    assert r["survivor_denominators"]["math_a0"] == 1254
    assert r["endpoint"]["L0b_own_loss"] == pytest.approx(0.9347826086956522)
    assert r["endpoint"]["math_a0_own_loss"] == pytest.approx(0.9465709736842105)
