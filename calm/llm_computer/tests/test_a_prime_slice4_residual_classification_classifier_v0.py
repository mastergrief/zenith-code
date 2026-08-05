"""STEP-2 residual classification classifier dual-key battery (PLAN v6)."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from scripts.a_prime_slice4_residual_classification_classifier_v0 import (

    DECLARED_TOP_EMBEDDED_FIELDS,
    HORIZONS,
    MANIFEST_SCHEMA,
    TERMINAL_MANIFEST_NAME,
    TERMINAL_RECEIPT_NAME,
    build_projections,
    build_terminal_receipt,
    extract_horizon_view,
    finalize_dual_key,
    main,
    mint_exclusive_run_root,
    sha256_file,
    sha256_hex,
    validate_candidate_receipt,
    verify_published_manifest,
)
from scripts.a_prime_slice4_residual_classification_reducer_v0 import (
    classification_core,
    classify_from_projections,
)
from scripts.a_prime_slice4_residual_classification_schema_v0 import (
    EXPECTED_CARDINALITY,
    REQUIRED_CLAIM_BOUNDARY,
)
# argv template conformance (PLAN v6 six_receipt_live_classification)
PLAN_ARGV_FLAGS = (
    "--run-root",
    "--package-receipt",
    "--out-receipt",
)
def _ids(prefix: str, n: int) -> list[str]:
    return [f"{i:04d}:{prefix}{i:04x}" for i in range(n)]
def _mk_receipt(support_ids: dict[str, list[str]], fails: dict[str, list[str]] | None = None) -> dict:
    fails = fails or {}
    final_reports = {}
    for support, ids in support_ids.items():
        batches = []
        for rid in ids:
            h = rid.rsplit(":", 1)[-1]
            batches.append(
                {
                    "metadata": {
                        "row_ids": [rid],
                        "sample_hashes": [h],
                        "source_buckets": ["b0"],
                    }
                }
            )
        final_reports[support] = {
            "batch_reports": batches,
            "strict_failure_row_ids": list(fails.get(support, [])),
            "support_rows_audited": len(ids),
        }
    return {"prior_audit": {"final_reports": final_reports}}
def _write_six(tmp: Path, mutate_pkg_n10=None) -> tuple[list[str], list[str], dict[str, str]]:
    tmp.mkdir(parents=True, exist_ok=True)
    l0b = _ids("l0", EXPECTED_CARDINALITY["L0b"])
    math = _ids("ma", EXPECTED_CARDINALITY["math_a0"])
    base = _mk_receipt({"L0b": l0b, "math_a0": math})
    pkg_args, out_args = [], []
    shas = {}
    for h in HORIZONS:
        for arm, args in (("package", pkg_args), ("out", out_args)):
            obj = json.loads(json.dumps(base))
            if arm == "package" and h == 10 and mutate_pkg_n10 is not None:
                mutate_pkg_n10(obj)
            p = tmp / f"{arm}_N{h}.json"
            text = json.dumps(obj, sort_keys=True) + "\n"
            p.write_text(text)
            shas[f"{arm}/N{h}"] = sha256_hex(text.encode())
            args.append(f"{h}={p}")
    return pkg_args, out_args, shas
def test_argv_template_conformance_flags():
    import scripts.a_prime_slice4_residual_classification_classifier_v0 as mod
    import argparse
    p = argparse.ArgumentParser()
    # rebuild parser actions from main's expected flags
    help_src = Path(mod.__file__).read_text()
    for flag in PLAN_ARGV_FLAGS:
        assert flag in help_src
    # sole run-root placeholder pattern in plan: __EXCLUSIVE_RUN_ROOT__ not hardcoded here
    assert "--package-receipt" in help_src and "--out-receipt" in help_src
def test_extract_horizon_view_and_classify_synthetic(tmp_path: Path):
    l0b = _ids("l0", EXPECTED_CARDINALITY["L0b"])
    math = _ids("ma", EXPECTED_CARDINALITY["math_a0"])
    rec = _mk_receipt({"L0b": l0b, "math_a0": math})
    view = extract_horizon_view(rec, "L0b")
    assert view["support_rows_audited"] == EXPECTED_CARDINALITY["L0b"]
    assert len(view["row_ids"]) == EXPECTED_CARDINALITY["L0b"]
    pkg = {h: rec for h in HORIZONS}
    out = {h: rec for h in HORIZONS}
    proj = build_projections(pkg, out)
    r = classify_from_projections(proj)
    assert r["identity_profile"] == "IDENTITY_OK"
    assert r["composite_terminal"].startswith("IDENTITY_OK__")
def test_mint_exclusive_run_root_exists(tmp_path: Path):
    root = tmp_path / "rr"
    root.mkdir()
    ok, reason = mint_exclusive_run_root(root)
    assert ok is False and "run_root_exists" in reason
def test_exclusive_mint_via_main_preexisting(tmp_path: Path, capsys):
    pkg, out, _ = _write_six(tmp_path / "rcpts")
    run_root = tmp_path / "exists"
    run_root.mkdir()
    rc = main(
        [
            "--run-root",
            str(run_root),
            *sum([["--package-receipt", x] for x in pkg], []),
            *sum([["--out-receipt", x] for x in out], []),
        ]
    )
    captured = capsys.readouterr().out
    assert rc == 2
    assert "INCOMPLETE_FINALIZATION" in captured
    assert "WRAPPER_RC 2" in captured
    assert "PACKET_TERMINAL" not in captured
    assert not (run_root / TERMINAL_RECEIPT_NAME).exists()
def test_dual_key_happy_path_single_marker(tmp_path: Path, capsys):
    pkg, out, shas = _write_six(tmp_path / "rcpts")
    run_root = tmp_path / "run_ok"
    rc = main(
        [
            "--run-root",
            str(run_root),
            *sum([["--package-receipt", x] for x in pkg], []),
            *sum([["--out-receipt", x] for x in out], []),
        ]
    )
    out_txt = capsys.readouterr().out
    assert rc == 0
    markers = [ln for ln in out_txt.splitlines() if ln.startswith("PACKET_TERMINAL ")]
    assert len(markers) == 1
    assert "WRAPPER_RC 0" in out_txt
    assert (run_root / TERMINAL_RECEIPT_NAME).is_file()
    assert (run_root / TERMINAL_MANIFEST_NAME).is_file()
    receipt = json.loads((run_root / TERMINAL_RECEIPT_NAME).read_text())
    assert receipt["source_shas"] == shas
    assert receipt["classification"]["source_shas"] == shas
    assert receipt["claim_boundary"] == REQUIRED_CLAIM_BOUNDARY
    assert receipt["terminal_authority"] == "manifest+marker"
    assert receipt["branch"] == receipt["composite_terminal"]
    assert receipt["branch"].startswith("IDENTITY_")
def _load_six(tmp: Path):
    from scripts.a_prime_slice4_residual_classification_classifier_v0 import (
        _parse_horizon_paths,
        load_receipts_same_byte,
    )
    pkg, out, _ = _write_six(tmp / "rcpts")
    pkg_paths, _ = _parse_horizon_paths(pkg, label="package")
    out_paths, _ = _parse_horizon_paths(out, label="out")
    pkg_objs, pkg_shas = load_receipts_same_byte(pkg_paths)
    out_objs, out_shas = load_receipts_same_byte(out_paths)
    classification = classify_from_projections(build_projections(pkg_objs, out_objs))
    source_shas = {
        **{f"package/N{n}": pkg_shas[n] for n in HORIZONS},
        **{f"out/N{n}": out_shas[n] for n in HORIZONS},
    }
    return classification, source_shas, pkg, out
def test_source_sha_bind_hostile_mismatch(tmp_path: Path):
    classification, true_shas, _, _ = _load_six(tmp_path)
    bad_shas = dict(true_shas)
    bad_shas["package/N10"] = "0" * 64
    snap = dict(classification)
    snap["source_shas"] = dict(bad_shas)
    root = tmp_path / "x"
    receipt = build_terminal_receipt(
        classification, run_root=root, source_shas=bad_shas
    )
    ok, reason = validate_candidate_receipt(
        receipt,
        source_shas=true_shas,
        canonical_snapshot=snap,
        expected_run_root=root,
    )
    assert ok is False
    assert "source_shas" in reason
def test_core_equality_tamper_hostile(tmp_path: Path, capsys):
    classification, source_shas, _, _ = _load_six(tmp_path)
    def mutator(receipt):
        receipt["classification"]["identity_profile"] = "TAMPERED"
    rc = finalize_dual_key(
        tmp_path / "tamper",
        classification,
        source_shas=source_shas,
        inject_receipt_mutator=mutator,
    )
    out_txt = capsys.readouterr().out
    assert rc == 2
    assert "candidate_invalid" in out_txt
    assert "PACKET_TERMINAL" not in out_txt
def test_missing_horizon_fail_closed(tmp_path: Path, capsys):
    pkg, out, _ = _write_six(tmp_path / "rcpts")
    pkg = [x for x in pkg if not x.startswith("50=")]
    rc = main(
        [
            "--run-root",
            str(tmp_path / "miss"),
            *sum([["--package-receipt", x] for x in pkg], []),
            *sum([["--out-receipt", x] for x in out], []),
        ]
    )
    assert rc == 2
    assert "PACKET_TERMINAL" not in capsys.readouterr().out
def test_duplicate_horizon_fail_closed(tmp_path: Path, capsys):
    pkg, out, _ = _write_six(tmp_path / "rcpts")
    pkg = pkg + [pkg[0]]
    rc = main(
        [
            "--run-root",
            str(tmp_path / "dup"),
            *sum([["--package-receipt", x] for x in pkg], []),
            *sum([["--out-receipt", x] for x in out], []),
        ]
    )
    assert rc == 2
    assert "PACKET_TERMINAL" not in capsys.readouterr().out
def test_claim_boundary_constants_in_receipt(tmp_path: Path, capsys):
    pkg, out, _ = _write_six(tmp_path / "rcpts")
    run_root = tmp_path / "cb"
    assert main([
        "--run-root", str(run_root),
        *sum([["--package-receipt", x] for x in pkg], []),
        *sum([["--out-receipt", x] for x in out], []),
    ]) == 0
    rec = json.loads((run_root / TERMINAL_RECEIPT_NAME).read_text())
    for k, v in REQUIRED_CLAIM_BOUNDARY.items():
        assert rec["claim_boundary"][k] is v
        assert rec["classification"]["claim_boundary"][k] is v
def test_classification_core_roundtrip_in_receipt(tmp_path: Path, capsys):
    pkg, out, _ = _write_six(tmp_path / "rcpts")
    run_root = tmp_path / "core"
    assert main([
        "--run-root", str(run_root),
        *sum([["--package-receipt", x] for x in pkg], []),
        *sum([["--out-receipt", x] for x in out], []),
    ]) == 0
    rec = json.loads((run_root / TERMINAL_RECEIPT_NAME).read_text())
    core = classification_core(rec["classification"])
    assert core["composite_terminal"] == rec["branch"]
    assert "identity_reasons" in core and "identity_raw" in core
def test_main_instrument_envelope_malformed_receipt(tmp_path: Path, capsys):
    """Malformed receipt → main still dual-key terminals as INSTRUMENT_OR_BIND_FAIL."""
    pkg, out, _ = _write_six(tmp_path / "rcpts")
    # corrupt package N10 structure (missing prior_audit)
    bad = tmp_path / "rcpts" / "package_N10.json"
    bad.write_text(json.dumps({"not_prior_audit": True}) + "\n")
    run_root = tmp_path / "instr"
    rc = main([
        "--run-root", str(run_root),
        *sum([["--package-receipt", x] for x in pkg], []),
        *sum([["--out-receipt", x] for x in out], []),
    ])
    out_txt = capsys.readouterr().out
    assert rc == 0
    markers = [ln for ln in out_txt.splitlines() if ln.startswith("PACKET_TERMINAL ")]
    assert len(markers) == 1
    assert markers[0] == "PACKET_TERMINAL INSTRUMENT_OR_BIND_FAIL"
    assert "WRAPPER_RC 0" in out_txt
    rec = json.loads((run_root / TERMINAL_RECEIPT_NAME).read_text())
    man = json.loads((run_root / TERMINAL_MANIFEST_NAME).read_text())
    assert rec["branch"] == "INSTRUMENT_OR_BIND_FAIL"
    assert rec["identity_profile"] == "INSTRUMENT_OR_BIND_FAIL"
    assert rec["classification"]["instrument_fail"] is True
    assert man["branch"] == "INSTRUMENT_OR_BIND_FAIL"
    assert man["schema"] == MANIFEST_SCHEMA
    assert man["synthetic"] is False
@pytest.mark.parametrize("field", list(DECLARED_TOP_EMBEDDED_FIELDS))
def test_top_embedded_field_mismatch_fail_closed(tmp_path: Path, field: str):
    classification, source_shas, _, _ = _load_six(tmp_path)
    root = tmp_path / "bind"
    receipt = build_terminal_receipt(
        classification, run_root=root, source_shas=source_shas
    )
    snap = dict(classification)
    snap["source_shas"] = dict(source_shas)
    # known-good first
    ok, reason = validate_candidate_receipt(
        receipt,
        source_shas=source_shas,
        canonical_snapshot=snap,
        expected_run_root=root,
    )
    assert ok is True, reason
    # tamper top-level only
    if field == "composite_terminal":
        receipt["composite_terminal"] = "TAMPERED_COMPOSITE"
        receipt["branch"] = "TAMPERED_COMPOSITE"
    else:
        receipt[field] = "TAMPERED_TOP"
    ok, reason = validate_candidate_receipt(
        receipt,
        source_shas=source_shas,
        canonical_snapshot=snap,
        expected_run_root=root,
    )
    assert ok is False
    assert field in reason or "top_ne_embedded" in reason or "composite_terminal" in reason
def test_run_root_mismatch_fail_closed(tmp_path: Path):
    classification, source_shas, _, _ = _load_six(tmp_path)
    root = tmp_path / "rr"
    receipt = build_terminal_receipt(
        classification, run_root=root, source_shas=source_shas
    )
    snap = dict(classification)
    snap["source_shas"] = dict(source_shas)
    ok, reason = validate_candidate_receipt(
        receipt,
        source_shas=source_shas,
        canonical_snapshot=snap,
        expected_run_root=tmp_path / "other_root",
    )
    assert ok is False
    assert "run_root_mismatch" in reason
def test_manifest_self_contract_negative_battery(tmp_path: Path):
    classification, source_shas, _, _ = _load_six(tmp_path)
    run_root = tmp_path / "man"
    assert (
        finalize_dual_key(run_root, classification, source_shas=source_shas) == 0
    )
    man_path = run_root / TERMINAL_MANIFEST_NAME
    # known-good silent
    branch = json.loads((run_root / TERMINAL_RECEIPT_NAME).read_text())["branch"]
    outputs = {TERMINAL_RECEIPT_NAME: sha256_file(run_root / TERMINAL_RECEIPT_NAME)}
    ok, reason = verify_published_manifest(
        man_path, receipt_branch=branch, expected_hashes=outputs
    )
    assert ok is True, reason
    def _mutate_and_check(mutator, needle: str):
        payload = json.loads(man_path.read_text())
        mutator(payload)
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        man_path.write_text(text)
        cand = sha256_hex(text.encode())
        ok2, reason2 = verify_published_manifest(
            man_path,
            receipt_branch=branch,
            expected_hashes=outputs,
            candidate_sha256=cand,
        )
        assert ok2 is False
        assert needle in reason2
    _mutate_and_check(lambda p: p.__setitem__("schema", "wrong.schema"), "manifest_schema")
    # rewrite good then synthetic
    payload = {
        "schema": MANIFEST_SCHEMA,
        "branch": branch,
        "terminal_authority": "manifest+marker",
        "run_root": str(man_path.parent.resolve()),
        "outputs": outputs,
        "synthetic": False,
    }
    man_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _mutate_and_check(lambda p: p.__setitem__("synthetic", True), "manifest_synthetic")
    man_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _mutate_and_check(
        lambda p: p.__setitem__("run_root", "/not/the/mint/root"),
        "manifest_run_root_mismatch",
    )
def test_known_good_envelope_and_manifest_still_pass(tmp_path: Path, capsys):
    pkg, out, shas = _write_six(tmp_path / "rcpts")
    run_root = tmp_path / "good"
    rc = main([
        "--run-root", str(run_root),
        *sum([["--package-receipt", x] for x in pkg], []),
        *sum([["--out-receipt", x] for x in out], []),
    ])
    assert rc == 0
    rec = json.loads((run_root / TERMINAL_RECEIPT_NAME).read_text())
    man = json.loads((run_root / TERMINAL_MANIFEST_NAME).read_text())
    for field in DECLARED_TOP_EMBEDDED_FIELDS:
        assert rec[field] == rec["classification"][field]
    assert rec["run_root"] == str(run_root.resolve())
    assert man["schema"] == MANIFEST_SCHEMA
    assert man["synthetic"] is False
    assert man["run_root"] == str(run_root.resolve())
    assert man["branch"] == rec["branch"]
    # outputs exact-set equality still silent on known-good
    outputs = {TERMINAL_RECEIPT_NAME: sha256_file(run_root / TERMINAL_RECEIPT_NAME)}
    ok, reason = verify_published_manifest(
        run_root / TERMINAL_MANIFEST_NAME,
        receipt_branch=rec["branch"],
        expected_hashes=outputs,
    )
    assert ok is True, reason
def test_manifest_outputs_extra_entry_fail_closed(tmp_path: Path):
    classification, source_shas, _, _ = _load_six(tmp_path)
    run_root = tmp_path / "extra"
    assert finalize_dual_key(run_root, classification, source_shas=source_shas) == 0
    man_path = run_root / TERMINAL_MANIFEST_NAME
    branch = json.loads((run_root / TERMINAL_RECEIPT_NAME).read_text())["branch"]
    expected = {TERMINAL_RECEIPT_NAME: sha256_file(run_root / TERMINAL_RECEIPT_NAME)}
    # plant an extra real file with correct hash, claim it in outputs
    extra = run_root / "extra_artifact.json"
    extra.write_text('{"extra": true}\n')
    payload = json.loads(man_path.read_text())
    payload["outputs"] = {
        **expected,
        "extra_artifact.json": sha256_file(extra),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    man_path.write_text(text)
    ok, reason = verify_published_manifest(
        man_path,
        receipt_branch=branch,
        expected_hashes=expected,
        candidate_sha256=sha256_hex(text.encode()),
    )
    assert ok is False
    assert reason == "manifest_outputs_set_ne_expected"
@pytest.mark.parametrize("bad_payload", (["list"], "scalar", 42, None))
def test_manifest_payload_not_exact_dict_fail_closed(tmp_path: Path, bad_payload):
    man_path = tmp_path / TERMINAL_MANIFEST_NAME
    man_path.write_text(json.dumps(bad_payload) + "\n")
    ok, reason = verify_published_manifest(
        man_path,
        receipt_branch="ANY",
        expected_hashes={TERMINAL_RECEIPT_NAME: "deadbeef"},
        candidate_sha256=sha256_file(man_path),
    )
    assert ok is False
    assert "manifest_payload_not_exact_dict" in reason
@pytest.mark.parametrize("bad_outputs", ("scalar", [1, 2], None))
def test_manifest_outputs_not_exact_dict_fail_closed(tmp_path: Path, bad_outputs):
    classification, source_shas, _, _ = _load_six(tmp_path)
    run_root = tmp_path / f"outs_{type(bad_outputs).__name__}"
    assert finalize_dual_key(run_root, classification, source_shas=source_shas) == 0
    man_path = run_root / TERMINAL_MANIFEST_NAME
    branch = json.loads((run_root / TERMINAL_RECEIPT_NAME).read_text())["branch"]
    expected = {TERMINAL_RECEIPT_NAME: sha256_file(run_root / TERMINAL_RECEIPT_NAME)}
    payload = json.loads(man_path.read_text())
    if bad_outputs is None:
        payload.pop("outputs", None)
    else:
        payload["outputs"] = bad_outputs
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    man_path.write_text(text)
    ok, reason = verify_published_manifest(
        man_path,
        receipt_branch=branch,
        expected_hashes=expected,
        candidate_sha256=sha256_hex(text.encode()),
    )
    assert ok is False
    assert "manifest_outputs_not_exact_dict" in reason
