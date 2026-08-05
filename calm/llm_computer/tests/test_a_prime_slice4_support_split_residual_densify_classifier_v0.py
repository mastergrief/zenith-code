"""Classifier integration: routing + markerless fail-closed + envelope hostiles (STEP-2)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.a_prime_slice4_support_split_residual_densify_classifier_v0 import (
    DECLARED_TOP_EMBEDDED_FIELDS,
    finalize_dual_key,
    main,
    validate_candidate_receipt,
    verify_published_manifest,
)
from scripts.a_prime_slice4_support_split_residual_densify_reducer_v0 import (
    densify_from_projections,
)
from scripts.a_prime_slice4_support_split_residual_densify_classifier_v0 import (
    build_projections,
    load_receipts_same_byte,
)
from scripts.a_prime_slice4_support_split_runtime_source_contract_v0 import (
    ALGORITHM,
    MANIFEST_SCHEMA_ID,
    MINTED_BY,
    ORDERED_RUNTIME_PATHS,
    TASK_ID,
    ordered_concat_v0,
)

ROOT = Path(__file__).resolve().parents[3]
PINS = {
    "package": {
        10: Path("/home/gabe/claw-code-creditdir/a_prime_slice4_protect/run_702cc34b/horizon_N10/c2p1_impl_cpu/receipt.json"),
        20: Path("/home/gabe/claw-code-creditdir/a_prime_slice4_protect/run_702cc34b/horizon_N20/c2p1_impl_cpu/receipt.json"),
        50: Path("/home/gabe/claw-code-creditdir/a_prime_slice4_protect/run_702cc34b/horizon_N50/c2p1_impl_cpu/receipt.json"),
    },
    "out": {
        10: Path("/home/gabe/claw-code-creditdir/a_prime_slice3_onset/run_fb2fcec5/horizon_N10/c2p1_impl_cpu/receipt.json"),
        20: Path("/home/gabe/claw-code-creditdir/a_prime_slice3_onset/run_fb2fcec5/horizon_N20/c2p1_impl_cpu/receipt.json"),
        50: Path("/home/gabe/claw-code-creditdir/a_prime_slice3_onset/run_fb2fcec5/horizon_N50/c2p1_impl_cpu/receipt.json"),
    },
}
TERM = Path("/home/gabe/claw-code-creditdir/a_prime_slice4_rung3/run_residual_v3/terminal_receipt.json")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _good_manifest_file(tmp: Path) -> tuple[Path, str]:
    per = {p: _sha((ROOT / p).read_bytes()) for p in ORDERED_RUNTIME_PATHS}
    dig = ordered_concat_v0(list(ORDERED_RUNTIME_PATHS), per)
    obj = {
        "schema_id": MANIFEST_SCHEMA_ID,
        "ordered_runtime_paths": list(ORDERED_RUNTIME_PATHS),
        "per_file_sha256": per,
        "runtime_source_digest": dig,
        "algorithm": ALGORITHM,
        "implementation_content_digest": "b" * 64,
        "minted_by": MINTED_BY,
        "task_id": TASK_ID,
        "plan_revision_binding": "v6_rung4_20260805",
    }
    raw = (json.dumps(obj, sort_keys=True) + "\n").encode()
    path = tmp / "rung4_runtime_source_manifest_v0.json"
    path.write_bytes(raw)
    return path, _sha(raw)


def _argv(run_root: Path, man: Path, man_sha: str) -> list[str]:
    return [
        "--run-root", str(run_root),
        "--package-receipt", f"10={PINS['package'][10]}",
        "--package-receipt", f"20={PINS['package'][20]}",
        "--package-receipt", f"50={PINS['package'][50]}",
        "--out-receipt", f"10={PINS['out'][10]}",
        "--out-receipt", f"20={PINS['out'][20]}",
        "--out-receipt", f"50={PINS['out'][50]}",
        "--rung3-terminal-receipt", str(TERM),
        "--runtime-source-manifest", str(man),
        "--runtime-source-manifest-sha256", man_sha,
    ]


def _live_classification_and_source_shas(man_sha: str):
    pkg = {n: PINS["package"][n] for n in (10, 20, 50)}
    out = {n: PINS["out"][n] for n in (10, 20, 50)}
    pkg_objs, pkg_shas = load_receipts_same_byte(pkg)  # type: ignore[misc]
    out_objs, out_shas = load_receipts_same_byte(out)  # type: ignore[misc]
    term_raw = TERM.read_bytes()
    term_obj = json.loads(term_raw.decode())
    proj = build_projections(pkg_objs, out_objs)
    classification = densify_from_projections(proj, authority_terminal=term_obj)
    source_shas = {
        **{f"package/N{n}": pkg_shas[n] for n in (10, 20, 50)},
        **{f"out/N{n}": out_shas[n] for n in (10, 20, 50)},
        "rung3_terminal": _sha(term_raw),
        "runtime_source_manifest_sha256_expected": man_sha,
        "runtime_source_manifest_sha256_observed": man_sha,
        "runtime_source_manifest_sha256_equal": True,
        "runtime_source": {
            "per_file": {p: _sha((ROOT / p).read_bytes()) for p in ORDERED_RUNTIME_PATHS},
            "runtime_source_digest": ordered_concat_v0(
                list(ORDERED_RUNTIME_PATHS),
                {p: _sha((ROOT / p).read_bytes()) for p in ORDERED_RUNTIME_PATHS},
            ),
            "algorithm": ALGORITHM,
            "manifest_schema_id": MANIFEST_SCHEMA_ID,
        },
    }
    return classification, source_shas


def _assert_markerless(out: str, rc: int, rr: Path) -> None:
    assert rc == 2
    assert "INCOMPLETE_FINALIZATION" in out
    assert "WRAPPER_RC 2" in out
    assert "PACKET_TERMINAL" not in out
    assert not (rr / "terminal_manifest.json").exists()


def test_happy_path_live_bytes(tmp_path, capsys):
    man, man_sha = _good_manifest_file(tmp_path)
    rr = tmp_path / "run_happy"
    rc = main(_argv(rr, man, man_sha))
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("PACKET_TERMINAL ") == 1
    assert "WRAPPER_RC 0" in out
    assert (rr / "terminal_receipt.json").is_file()
    assert (rr / "terminal_manifest.json").is_file()
    rec = json.loads((rr / "terminal_receipt.json").read_text())
    assert rec["schema"].startswith("a_prime_slice4_support_split_residual_densify")
    assert rec["source_shas"]["runtime_source_manifest_sha256_equal"] is True
    assert rec["source_shas"]["runtime_source_manifest_sha256_expected"] == man_sha
    assert rec["identity_profile"] == "IDENTITY_OK"
    for f in DECLARED_TOP_EMBEDDED_FIELDS:
        assert rec[f] == rec["classification"][f]
    vok, _ = verify_published_manifest(
        rr / "terminal_manifest.json",
        receipt_branch=rec["branch"],
        expected_hashes={"terminal_receipt.json": _sha((rr / "terminal_receipt.json").read_bytes())},
    )
    assert vok is True


def test_preexisting_run_root_markerless(tmp_path, capsys):
    man, man_sha = _good_manifest_file(tmp_path)
    rr = tmp_path / "exists"
    rr.mkdir()
    rc = main(_argv(rr, man, man_sha))
    out = capsys.readouterr().out
    assert rc == 2 and "INCOMPLETE_FINALIZATION" in out and "PACKET_TERMINAL" not in out


def test_expected_manifest_sha_mismatch_markerless(tmp_path, capsys):
    man, man_sha = _good_manifest_file(tmp_path)
    rc = main(_argv(tmp_path / "r", man, "00" * 32))
    out = capsys.readouterr().out
    assert rc == 2 and "runtime_source_manifest_sha_mismatch" in out
    assert "PACKET_TERMINAL" not in out
    assert not (tmp_path / "r").exists()


def test_malformed_horizon_positional_markerless(tmp_path, capsys):
    man, man_sha = _good_manifest_file(tmp_path)
    a = _argv(tmp_path / "r", man, man_sha)
    i = a.index("--package-receipt")
    a[i + 1] = str(PINS["package"][10])
    rc = main(a)
    out = capsys.readouterr().out
    assert rc == 2 and "INCOMPLETE_FINALIZATION" in out and "PACKET_TERMINAL" not in out


def test_duplicate_horizon_markerless(tmp_path, capsys):
    man, man_sha = _good_manifest_file(tmp_path)
    a = _argv(tmp_path / "r", man, man_sha)
    a += ["--package-receipt", f"10={PINS['package'][20]}"]
    rc = main(a)
    out = capsys.readouterr().out
    assert rc == 2 and "PACKET_TERMINAL" not in out


def test_missing_horizon_markerless(tmp_path, capsys):
    man, man_sha = _good_manifest_file(tmp_path)
    a = [
        "--run-root", str(tmp_path / "r"),
        "--package-receipt", f"10={PINS['package'][10]}",
        "--package-receipt", f"20={PINS['package'][20]}",
        "--out-receipt", f"10={PINS['out'][10]}",
        "--out-receipt", f"20={PINS['out'][20]}",
        "--out-receipt", f"50={PINS['out'][50]}",
        "--rung3-terminal-receipt", str(TERM),
        "--runtime-source-manifest", str(man),
        "--runtime-source-manifest-sha256", man_sha,
    ]
    rc = main(a)
    out = capsys.readouterr().out
    assert rc == 2 and "horizon_set" in out


def test_swapped_horizon_pin_equality_markerless(tmp_path, capsys):
    """N20 path under N10 key fails frozen path pin equality (strict markerless)."""
    man, man_sha = _good_manifest_file(tmp_path)
    rr = tmp_path / "swap"
    a = _argv(rr, man, man_sha)
    idxs = [i for i, x in enumerate(a) if x == "--package-receipt"]
    a[idxs[0] + 1] = f"10={PINS['package'][20]}"
    a[idxs[1] + 1] = f"20={PINS['package'][10]}"
    rc = main(a)
    out = capsys.readouterr().out
    assert rc == 2
    assert "INCOMPLETE_FINALIZATION" in out
    assert "INSTRUMENT_OR_BIND_FAIL" in out
    assert "path_ne_pin" in out
    assert "PACKET_TERMINAL" not in out
    assert not rr.exists()


def test_missing_terminal_markerless(tmp_path, capsys):
    man, man_sha = _good_manifest_file(tmp_path)
    a = _argv(tmp_path / "r", man, man_sha)
    i = a.index("--rung3-terminal-receipt")
    a[i + 1] = str(tmp_path / "nope.json")
    rc = main(a)
    out = capsys.readouterr().out
    assert rc == 2 and "missing_terminal" in out


def test_valid_shaped_terminal_substitution_markerless(tmp_path, capsys):
    """Structurally valid terminal at wrong path/bytes → AUTHORITY_BIND_FAIL class."""
    man, man_sha = _good_manifest_file(tmp_path)
    # copy real terminal bytes to a different path (path pin fails first)
    fake = tmp_path / "fake_terminal.json"
    fake.write_bytes(TERM.read_bytes())
    rr = tmp_path / "term_sub"
    a = _argv(rr, man, man_sha)
    i = a.index("--rung3-terminal-receipt")
    a[i + 1] = str(fake)
    rc = main(a)
    out = capsys.readouterr().out
    assert rc == 2
    assert "AUTHORITY_BIND_FAIL" in out
    assert "terminal_path_ne_pin" in out
    assert "PACKET_TERMINAL" not in out
    assert not rr.exists()


def test_sha_substitution_hostile_markerless(tmp_path, capsys):
    """Correct pin path but different bytes (tmp copy under expected path is impossible
    without mutating creditdir). Calibrate via path pointing at wrong-horizon file
    already covered by swap; here assert out/N50 path under out/N10 key fails path pin.
    """
    man, man_sha = _good_manifest_file(tmp_path)
    rr = tmp_path / "out_swap"
    a = _argv(rr, man, man_sha)
    idxs = [i for i, x in enumerate(a) if x == "--out-receipt"]
    a[idxs[0] + 1] = f"10={PINS['out'][50]}"
    a[idxs[2] + 1] = f"50={PINS['out'][10]}"
    rc = main(a)
    out = capsys.readouterr().out
    assert rc == 2 and "INSTRUMENT_OR_BIND_FAIL" in out and "path_ne_pin" in out
    assert "PACKET_TERMINAL" not in out
    assert not rr.exists()


@pytest.mark.parametrize(
    "name,mut,needle",
    [
        ("top_ne_embedded", lambda r: r.__setitem__("identity_profile", "TAMPERED"), "top_ne_embedded"),
        ("claim_boundary_missing", lambda r: r.pop("claim_boundary", None), "claim_boundary_missing"),
        (
            "source_shas_mismatch",
            lambda r: r["source_shas"].__setitem__("rung3_terminal", "00" * 32),
            "source_shas_mismatch",
        ),
        (
            "core_snapshot_ne_embedded",
            lambda r: (
                r["classification"]["D1_profile"].__setitem__(
                    "composite", "TAMPERED_D1_COMPOSITE"
                )
                if isinstance(r["classification"].get("D1_profile"), dict)
                else r["classification"].__setitem__("instrument_fail", True)
            ),
            "core_snapshot_ne_embedded",
        ),
    ],
)
def test_finalize_inject_receipt_mutator_markerless(tmp_path, capsys, name, mut, needle):
    man, man_sha = _good_manifest_file(tmp_path)
    classification, source_shas = _live_classification_and_source_shas(man_sha)
    rr = tmp_path / f"mut_{name}"

    def mutator(receipt):
        mut(receipt)

    rc = finalize_dual_key(
        rr, classification, source_shas=source_shas, inject_receipt_mutator=mutator
    )
    out = capsys.readouterr().out
    _assert_markerless(out, rc, rr)
    assert needle in out


def test_validate_candidate_runtime_source_sha_not_equal_direct(tmp_path):
    """equal=False is only reachable when top source_shas still equals activation map."""
    man, man_sha = _good_manifest_file(tmp_path)
    classification, source_shas = _live_classification_and_source_shas(man_sha)
    rr = tmp_path / "direct_eq"
    # Build a valid receipt via finalize happy path first.
    assert finalize_dual_key(rr, classification, source_shas=source_shas) == 0
    rec = json.loads((rr / "terminal_receipt.json").read_text())
    bad_src = copy.deepcopy(rec["source_shas"])
    bad_src["runtime_source_manifest_sha256_equal"] = False
    rec["source_shas"] = bad_src
    rec["classification"]["source_shas"] = copy.deepcopy(bad_src)
    snap = copy.deepcopy(rec["classification"])
    ok, reason = validate_candidate_receipt(
        rec, source_shas=bad_src, canonical_snapshot=snap, expected_run_root=rr
    )
    assert ok is False and reason == "runtime_source_manifest_sha_not_equal"


def test_verify_published_manifest_unit_negatives(tmp_path, capsys):
    man, man_sha = _good_manifest_file(tmp_path)
    rr = tmp_path / "pub"
    assert main(_argv(rr, man, man_sha)) == 0
    capsys.readouterr()
    mp = rr / "terminal_manifest.json"
    rec = json.loads((rr / "terminal_receipt.json").read_text())
    branch = rec["branch"]
    good_hash = _sha((rr / "terminal_receipt.json").read_bytes())
    exp = {"terminal_receipt.json": good_hash}
    assert verify_published_manifest(mp, receipt_branch=branch, expected_hashes=exp)[0] is True

    payload = json.loads(mp.read_text())

    def _write(p):
        mp.write_text(json.dumps(p, indent=2, sort_keys=True) + "\n")

    p = dict(payload)
    p["schema"] = "wrong"
    _write(p)
    ok, reason = verify_published_manifest(mp, receipt_branch=branch, expected_hashes=exp)
    assert ok is False and "manifest_schema" in reason

    p = dict(payload)
    p["synthetic"] = True
    _write(p)
    ok, reason = verify_published_manifest(mp, receipt_branch=branch, expected_hashes=exp)
    assert ok is False and reason == "manifest_synthetic"

    p = dict(payload)
    p["run_root"] = "/tmp/not_this_run"
    _write(p)
    ok, reason = verify_published_manifest(mp, receipt_branch=branch, expected_hashes=exp)
    assert ok is False and reason == "manifest_run_root_mismatch"

    p = dict(payload)
    p["outputs"] = dict(exp)
    p["outputs"]["extra.json"] = "ab" * 32
    _write(p)
    ok, reason = verify_published_manifest(mp, receipt_branch=branch, expected_hashes=exp)
    assert ok is False and reason == "manifest_outputs_set_ne_expected"

    p = dict(payload)
    p["outputs"] = {}
    _write(p)
    ok, reason = verify_published_manifest(mp, receipt_branch=branch, expected_hashes=exp)
    assert ok is False and reason == "manifest_outputs_set_ne_expected"

    # restore good outputs then stale hash
    p = dict(payload)
    p["outputs"] = {"terminal_receipt.json": "cd" * 32}
    _write(p)
    ok, reason = verify_published_manifest(
        mp, receipt_branch=branch, expected_hashes={"terminal_receipt.json": "cd" * 32}
    )
    assert ok is False and "stale_output_hash" in reason

    # direct validate_candidate_receipt positive on published receipt bytes
    rec2 = json.loads((rr / "terminal_receipt.json").read_text())
    snap = copy.deepcopy(rec2["classification"])
    vok, vreason = validate_candidate_receipt(
        rec2, source_shas=rec2["source_shas"], canonical_snapshot=snap, expected_run_root=rr
    )
    assert vok is True and vreason == "ok"


def test_line_cap_classifier_and_tests():
    for rel in (
        "scripts/a_prime_slice4_support_split_residual_densify_classifier_v0.py",
        "scripts/a_prime_slice4_support_split_runtime_source_contract_v0.py",
        "calm/llm_computer/tests/test_a_prime_slice4_support_split_residual_densify_classifier_v0.py",
        "calm/llm_computer/tests/test_a_prime_slice4_support_split_runtime_source_contract_v0.py",
    ):
        n = (ROOT / rel).read_text().count("\n") + 1
        assert n < 500, f"{rel}={n}"
