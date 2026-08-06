"""STEP-2 tests: Rung-6 classifier dual-key + pins (PLAN v6)."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import scripts.a_prime_slice4_count_standardization_classifier_v0 as clf
import scripts.a_prime_slice4_count_standardization_runtime_source_contract_v0 as contract
from scripts.a_prime_slice4_count_standardization_schema_v0 import (
    BRANCH_RATE_PROFILE_SELECTS,
    CLAIM_BOUNDARY_REQUIRED,
    PLAN_REVISION_BINDING,
)

REPO = Path(__file__).resolve().parents[3]
PLAN_BINDING = (
    "PLAN_v6 ee9628cdcc45515dd8007de065960cae344b43f5ccaa600b3d8bafaa3066b900"
)


def test_plan_binding_constants():
    assert PLAN_BINDING == PLAN_REVISION_BINDING
    assert contract.PLAN_REVISION_BINDING == PLAN_BINDING
    assert clf.PLAN_REVISION_BINDING == PLAN_BINDING


def _real_rung5_receipt() -> dict:
    pin = contract.FROZEN_RUNG5_TERMINAL_PIN
    return json.loads(Path(pin["path"]).read_text())


def test_extract_and_standardize_live_rung5():
    receipt = _real_rung5_receipt()
    c1_raw, aggregates, published = clf.extract_inputs_from_rung5_terminal(receipt)
    from scripts.a_prime_slice4_count_standardization_reducer_v0 import (
        standardize_from_c1_raw,
    )

    out = standardize_from_c1_raw(c1_raw, aggregates, published)
    assert out["primary"] == BRANCH_RATE_PROFILE_SELECTS
    assert out["plan_revision_binding"] == PLAN_BINDING
    assert set(out["claim_boundary"]) == set(CLAIM_BOUNDARY_REQUIRED)
    assert published["L0b"] == "E_MIXED"
    assert published["math_a0"] == "E_TRANSIENT"


def _strip_label(receipt: dict, support: str) -> dict:
    """Deep-copy receipt and remove published D2 label for one support (both sites)."""
    import copy

    r = copy.deepcopy(receipt)
    for root in (r, r.get("classification") or {}):
        c2 = root.get("C2_profile") if isinstance(root, dict) else None
        if not isinstance(c2, dict):
            continue
        labels = c2.get("aggregate_d2_labels")
        if isinstance(labels, dict) and support in labels:
            del labels[support]
        raw = c2.get("raw")
        if isinstance(raw, dict) and support in raw:
            agg = (raw[support] or {}).get("aggregate")
            if isinstance(agg, dict) and "label" in agg:
                del agg["label"]
    return r


def test_l0b_only_label_strip_fails_closed():
    """Observed counter-case (gate-1 bounce): missing L0b label must NOT mint RATE."""
    from scripts.a_prime_slice4_count_standardization_reducer_v0 import (
        standardize_from_c1_raw,
    )
    from scripts.a_prime_slice4_count_standardization_schema_v0 import (
        BRANCH_STANDARDIZATION_BIND_FAIL,
    )

    receipt = _strip_label(_real_rung5_receipt(), "L0b")
    try:
        c1_raw, aggregates, published = clf.extract_inputs_from_rung5_terminal(receipt)
    except ValueError as e:
        assert "missing_published_d2_label:L0b" in str(e)
        return
    out = standardize_from_c1_raw(c1_raw, aggregates, published)
    assert out["primary"] == BRANCH_STANDARDIZATION_BIND_FAIL
    assert out["primary"] != BRANCH_RATE_PROFILE_SELECTS


def test_both_labels_strip_fails_closed():
    from scripts.a_prime_slice4_count_standardization_reducer_v0 import (
        standardize_from_c1_raw,
    )
    from scripts.a_prime_slice4_count_standardization_schema_v0 import (
        BRANCH_STANDARDIZATION_BIND_FAIL,
    )

    receipt = _strip_label(_real_rung5_receipt(), "L0b")
    receipt = _strip_label(receipt, "math_a0")
    try:
        c1_raw, aggregates, published = clf.extract_inputs_from_rung5_terminal(receipt)
    except ValueError as e:
        assert "missing_published_d2_label" in str(e)
        return
    out = standardize_from_c1_raw(c1_raw, aggregates, published)
    assert out["primary"] == BRANCH_STANDARDIZATION_BIND_FAIL
    assert out["primary"] != BRANCH_RATE_PROFILE_SELECTS


def test_dual_key_preexisting_root_fails(tmp_path: Path):
    root = tmp_path / "exists"
    root.mkdir()
    ok, reason = clf.mint_exclusive_run_root(root)
    assert not ok and "run_root_exists" in reason


def test_dual_key_happy_fixture(tmp_path: Path):
    from scripts.a_prime_slice4_count_standardization_reducer_v0 import (
        standardize_live_shaped,
    )

    classification = standardize_live_shaped()
    root = tmp_path / "run_v1"
    rc = clf.finalize_dual_key(
        root,
        classification,
        source_shas={"plan_revision_binding": PLAN_BINDING},
    )
    assert rc == 0
    assert (root / clf.TERMINAL_RECEIPT_NAME).is_file()
    assert (root / clf.TERMINAL_MANIFEST_NAME).is_file()
    rec = json.loads((root / clf.TERMINAL_RECEIPT_NAME).read_text())
    assert rec["plan_revision_binding"] == PLAN_BINDING
    assert rec["claim_boundary"] == CLAIM_BOUNDARY_REQUIRED
    assert rec["branch"].startswith("IDENTITY_OK__") or rec["branch"] == classification[
        "composite_terminal"
    ]
    man = json.loads((root / clf.TERMINAL_MANIFEST_NAME).read_text())
    assert man["branch"] == rec["branch"]
    assert man["plan_revision_binding"] == PLAN_BINDING


def test_marker_cardinality_concept():
    # dual-key success prints exactly one PACKET_TERMINAL (enforced by finalize path)
    src = (
        REPO / "scripts/a_prime_slice4_count_standardization_classifier_v0.py"
    ).read_text()
    assert src.count('print(f"PACKET_TERMINAL') == 1
    assert "WRAPPER_RC 0" in src


def test_import_closure_four_path():
    src = (
        REPO / "scripts/a_prime_slice4_count_standardization_classifier_v0.py"
    ).read_text()
    tree = ast.parse(src)
    allowed = {
        "scripts.a_prime_slice4_count_standardization_schema_v0",
        "scripts.a_prime_slice4_count_standardization_reducer_v0",
        "scripts.a_prime_slice4_count_standardization_runtime_source_contract_v0",
        "scripts.a_prime_slice4_count_standardization_classifier_v0",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("scripts."):
                assert node.module in allowed, node.module


def test_no_forbidden_prior_imports():
    for rel in (
        "scripts/a_prime_slice4_count_standardization_classifier_v0.py",
        "scripts/a_prime_slice4_count_standardization_runtime_source_contract_v0.py",
    ):
        src = (REPO / rel).read_text()
        assert "shared_component" not in src
        assert "residual_classification" not in src
        assert "support_split_residual" not in src


def test_argv_placeholders_three():
    # live template in plan has exactly 3 placeholders; classifier argparse has no more
    src = (
        REPO / "scripts/a_prime_slice4_count_standardization_classifier_v0.py"
    ).read_text()
    assert src.count("__EXCLUSIVE_RUN_ROOT__") == 0  # substituted at launch
    assert "--run-root" in src
    assert "--runtime-source-manifest" in src
    assert "--runtime-source-manifest-sha256" in src
    assert "--rung5-terminal-receipt" in src


def test_line_caps_step2():
    files = [
        REPO / "scripts/a_prime_slice4_count_standardization_runtime_source_contract_v0.py",
        REPO / "scripts/a_prime_slice4_count_standardization_classifier_v0.py",
        REPO
        / "calm/llm_computer/tests/test_a_prime_slice4_count_standardization_runtime_source_contract_v0.py",
        REPO
        / "calm/llm_computer/tests/test_a_prime_slice4_count_standardization_classifier_v0.py",
    ]
    for f in files:
        n = len(f.read_text().splitlines())
        assert n < 500, f"{f.name} {n}"


def test_step1_binding_and_line_caps():
    """STEP-1 modules share PLAN_v6 binding; each file stays under 500L."""
    schema = REPO / "scripts/a_prime_slice4_count_standardization_schema_v0.py"
    reducer = REPO / "scripts/a_prime_slice4_count_standardization_reducer_v0.py"
    tests = (
        REPO
        / "calm/llm_computer/tests/test_a_prime_slice4_count_standardization_reducer_v0.py"
    )
    assert PLAN_BINDING in schema.read_text()
    assert "PLAN_v6 ee9628cd" in schema.read_text()
    assert "dcd51995" not in schema.read_text()
    for f in (schema, reducer, tests):
        assert len(f.read_text().splitlines()) < 500, f.name


def _good_receipt(tmp_path: Path) -> tuple[dict, dict]:
    from scripts.a_prime_slice4_count_standardization_reducer_v0 import (
        standardize_live_shaped,
    )

    classification = standardize_live_shaped()
    root = tmp_path / "snap"
    root.mkdir()
    source_shas = {"plan_revision_binding": PLAN_BINDING, "tag": "good"}
    rec = clf.build_terminal_receipt(
        classification, run_root=root, source_shas=source_shas
    )
    return rec, classification


def test_hostile_missing_top_primary(tmp_path: Path):
    rec, classification = _good_receipt(tmp_path)
    del rec["primary"]
    ok, reason = clf.validate_candidate_receipt(
        rec, classification=classification, expected_run_root=tmp_path / "snap"
    )
    assert not ok and "top_missing:primary" in reason


def test_hostile_missing_embedded_primary(tmp_path: Path):
    rec, classification = _good_receipt(tmp_path)
    del rec["classification"]["primary"]
    ok, reason = clf.validate_candidate_receipt(
        rec, classification=classification, expected_run_root=tmp_path / "snap"
    )
    assert not ok and "embedded_missing:primary" in reason


def test_hostile_missing_claim_boundary_no_fabricate():
    from scripts.a_prime_slice4_count_standardization_reducer_v0 import (
        standardize_live_shaped,
    )

    classification = dict(standardize_live_shaped())
    del classification["claim_boundary"]
    try:
        clf.build_terminal_receipt(
            classification,
            run_root=Path("/tmp/unused_rung6"),
            source_shas={"plan_revision_binding": PLAN_BINDING},
        )
        assert False, "expected ValueError claim_boundary_missing"
    except ValueError as e:
        assert "claim_boundary_missing" in str(e)


def test_hostile_missing_composite_terminal_no_primary_fallback():
    from scripts.a_prime_slice4_count_standardization_reducer_v0 import (
        standardize_live_shaped,
    )

    classification = dict(standardize_live_shaped())
    del classification["composite_terminal"]
    # primary still present — must NOT fall back
    assert classification.get("primary")
    try:
        clf.build_terminal_receipt(
            classification,
            run_root=Path("/tmp/unused_rung6"),
            source_shas={"plan_revision_binding": PLAN_BINDING},
        )
        assert False, "expected ValueError composite_terminal_missing"
    except ValueError as e:
        assert "composite_terminal_missing" in str(e)


def test_hostile_embedded_plan_binding_mismatch(tmp_path: Path):
    rec, classification = _good_receipt(tmp_path)
    rec["classification"]["plan_revision_binding"] = "PLAN_v6 " + ("0" * 64)
    ok, reason = clf.validate_candidate_receipt(
        rec, classification=classification, expected_run_root=tmp_path / "snap"
    )
    assert not ok and "embedded_plan_revision_binding" in reason


def test_hostile_source_shas_mismatch(tmp_path: Path):
    rec, classification = _good_receipt(tmp_path)
    rec["classification"]["source_shas"] = {"mutated": True}
    ok, reason = clf.validate_candidate_receipt(
        rec, classification=classification, expected_run_root=tmp_path / "snap"
    )
    assert not ok and "source_shas_mismatch" in reason


def test_hostile_missing_runtime_manifest_main(tmp_path: Path, capsys):
    root = tmp_path / "no_create_root"
    missing = tmp_path / "does_not_exist_manifest.json"
    rc = clf.main(
        [
            "--run-root",
            str(root),
            "--rung5-terminal-receipt",
            str(contract.FROZEN_RUNG5_TERMINAL_PIN["path"]),
            "--runtime-source-manifest",
            str(missing),
            "--runtime-source-manifest-sha256",
            "a" * 64,
        ]
    )
    assert rc == 2
    out = capsys.readouterr().out
    assert "INCOMPLETE_FINALIZATION" in out
    assert "WRAPPER_RC 2" in out
    assert not root.exists()
