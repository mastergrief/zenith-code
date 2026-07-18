"""CPU-static tests for canonical terminal receipt consumer (PLAN v28 Step A)."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack import signed_utility_fixed_state_schema as schema
from calm.hrm_text_158.native_full_stack import signed_utility_fixed_state_terminal_consumer as cons
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_legal_subset import (
    ESTIMAND_NAME,
    MAX_AUTHORITATIVE_RESULT_BYTES,
)

REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
CONSUMER = REPO / "calm/hrm_text_158/native_full_stack/signed_utility_fixed_state_terminal_consumer.py"
A0 = Path(
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "forgotten_accum_A_LEDGER_ACCOUNTING_v2_POST_SEAM_SIGNED_UTILITY_D2C16_"
    "TERMINAL_CONSUMER_V18_CHARACTERIZATION_RECEIPT.json"
)
A0_SHA = "1ca2b80d0151a989e74893dd44579baf2032ef772d44f58fb24e7f9d3cb4697d"
DELTA_IDS = frozenset({"UNVERIFIED_V4_INTEGRITY", "UNVERIFIED_V4_ASYMMETRIC"})
BYTES_RE = re.compile(r"^bytes\(len=(\d+), sha256=([0-9a-f]{64})\)$")
STDLIB_ROOTS = frozenset(sys.stdlib_module_names) | {"__future__"}


def _dumps(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _reconstruct_raw(row: dict):
    call = row["call"]
    raw_repr = call.get("raw_repr")
    if raw_repr is None:
        return None
    if isinstance(raw_repr, str) and raw_repr.startswith("empty_bytes"):
        return b""
    if row["id"] == "OVERSIZE_RECEIPT":
        return _dumps(row["CURRENT_V18_OUTPUT"]["parsed"])
    text = call.get("raw_json_or_text")
    assert text is not None
    raw = text.encode("utf-8")
    m = BYTES_RE.match(raw_repr)
    assert m and len(raw) == int(m.group(1)) and hashlib.sha256(raw).hexdigest() == m.group(2)
    return raw


@pytest.fixture(scope="module")
def a0_rows():
    assert hashlib.sha256(A0.read_bytes()).hexdigest() == A0_SHA
    data = json.loads(A0.read_text())
    assert data["row_count"] == 19
    return data["rows"]


def test_consumer_loc_cap():
    assert len(CONSUMER.read_text().splitlines()) <= 250


def test_ast_zero_non_stdlib_imports():
    tree = ast.parse(CONSUMER.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                assert root in STDLIB_ROOTS, alias.name
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert node.module is not None
            root = node.module.split(".", 1)[0]
            assert root in STDLIB_ROOTS, node.module


def test_vocabulary_parity_vs_schema():
    assert cons.SCHEMA_SCIENCE == schema.SCHEMA_SCIENCE
    assert cons.SCHEMA_UNVERIFIED == schema.SCHEMA_UNVERIFIED
    assert cons.ESTIMAND == ESTIMAND_NAME
    assert cons.INTEGRITY in schema.TERMINAL_CLASSES
    assert cons.ASYMMETRIC in schema.TERMINAL_CLASSES
    assert cons.MAX_RECEIPT == MAX_AUTHORITATIVE_RESULT_BYTES
    assert cons.ALLOWED_UNVERIFIED_CLASSIFIERS == frozenset({cons.INTEGRITY, cons.ASYMMETRIC})


def test_a0_non_delta_rows_exact(a0_rows):
    for row in a0_rows:
        if row["id"] in DELTA_IDS:
            continue
        if row["fn"] == "support_trichotomy_from_bytes":
            raw = _reconstruct_raw(row)
            got = cons.support_trichotomy_from_bytes(
                raw, exists=bool(row["call"]["exists"]), saw_begin=bool(row["call"]["saw_begin"])
            )
            assert got == row["CURRENT_V18_OUTPUT"], row["id"]
        else:
            got = cons.cross_check_pair_receipt(row["call"]["pair_kind"], row["call"]["support_meta"])
            assert got == row["CURRENT_V18_OUTPUT"], row["id"]


def test_intended_delta_unverified_integrity_accepted_nonscience(a0_rows):
    row = next(r for r in a0_rows if r["id"] == "UNVERIFIED_V4_INTEGRITY")
    raw = _reconstruct_raw(row)
    got = cons.support_trichotomy_from_bytes(raw, exists=True, saw_begin=True)
    assert got["trichotomy_enum"] == "nonempty_parsed"
    assert got["chronology_ok"] is True
    assert got["integrity_reasons"] == []
    assert got.get("nonscience") is True
    assert got["parsed"]["classifier"] == cons.INTEGRITY
    assert got["parsed"]["classifier"] not in cons.SCIENCE_ELIGIBILITY_CLASSIFIERS


def test_intended_delta_unverified_asymmetric_accepted_nonscience(a0_rows):
    row = next(r for r in a0_rows if r["id"] == "UNVERIFIED_V4_ASYMMETRIC")
    raw = _reconstruct_raw(row)
    got = cons.support_trichotomy_from_bytes(raw, exists=True, saw_begin=True)
    assert got["trichotomy_enum"] == "nonempty_parsed"
    assert got["chronology_ok"] is True
    assert got.get("nonscience") is True
    assert got["parsed"]["classifier"] == cons.ASYMMETRIC
    assert got["parsed"]["classifier"] not in (cons.ELIGIBLE, cons.PRESENT, cons.NULL_OR_HARMFUL)


def test_invented_capture_phase_fail_closed(a0_rows):
    row = next(r for r in a0_rows if r["id"] == "UNVERIFIED_V4_INVENTED_CAPTURE_PHASE_NEGATIVE")
    raw = _reconstruct_raw(row)
    got = cons.support_trichotomy_from_bytes(raw, exists=True, saw_begin=True)
    assert got == row["CURRENT_V18_OUTPUT"]
    assert got["trichotomy_enum"] == "nonempty_unparsed"
    assert got.get("nonscience") is not True


def test_unverified_never_eligible_present_null():
    for clf in (cons.INTEGRITY, cons.ASYMMETRIC):
        raw = _dumps(
            {
                "schema": cons.SCHEMA_UNVERIFIED,
                "estimand": cons.ESTIMAND,
                "classifier": clf,
                "status": "unverified",
            }
        )
        got = cons.support_trichotomy_from_bytes(raw, exists=True, saw_begin=True)
        assert got["trichotomy_enum"] == "nonempty_parsed"
        assert got["parsed"]["classifier"] not in cons.SCIENCE_ELIGIBILITY_CLASSIFIERS
        assert got.get("nonscience") is True
        # pair 0/0 must not treat as eligible
        reasons = cons.cross_check_pair_receipt("completed_eligible", got)
        assert "pair_0_0_requires_ELIGIBLE_receipt" in reasons or "pair_0_0_forbids_nonscience_unverified" in reasons


def test_nll_bearing_unverified_integrity_fail():
    raw = _dumps(
        {
            "schema": cons.SCHEMA_UNVERIFIED,
            "estimand": cons.ESTIMAND,
            "classifier": cons.INTEGRITY,
            "L_prod": 1.0,
            "L_inv": 2.0,
            "L_noop": 1.5,
            "L_noop_repeat": 1.5,
            "epsilon": 0.1,
        }
    )
    got = cons.support_trichotomy_from_bytes(raw, exists=True, saw_begin=True)
    assert got["trichotomy_enum"] == "nonempty_parsed"
    assert got["chronology_ok"] is False
    assert any(str(x).startswith("nll_bearing_unverified:") for x in got["integrity_reasons"])
    assert got.get("nonscience") is True
    assert got["parsed"]["classifier"] != cons.ELIGIBLE
    pair = cons.cross_check_pair_receipt("completed_cli_noneligible_or_integrity_receipt", got)
    assert "pair_2_0_requires_chronology_ok" in pair or "pair_2_0_forbids_integrity_reasons" in pair


@pytest.mark.parametrize("key", sorted(cons.NLL_BEARING_KEYS))
@pytest.mark.parametrize("value", [None, False, 0, 0.0])
def test_nll_key_presence_fails_regardless_of_value(key, value):
    payload = {
        "schema": cons.SCHEMA_UNVERIFIED,
        "estimand": cons.ESTIMAND,
        "classifier": cons.INTEGRITY,
        "status": "unverified",
        key: value,
    }
    got = cons.support_trichotomy_from_bytes(_dumps(payload), exists=True, saw_begin=True)
    assert got["chronology_ok"] is False
    assert f"nll_bearing_unverified:{key}" in got["integrity_reasons"]
    pair = cons.cross_check_pair_receipt("completed_cli_noneligible_or_integrity_receipt", got)
    assert pair != []


@pytest.mark.parametrize("key", sorted(cons.SCIENCE_MARKER_KEYS))
@pytest.mark.parametrize("value", [None, False, 0, 0.0, ""])
def test_science_marker_key_presence_fails_regardless_of_value(key, value):
    payload = {
        "schema": cons.SCHEMA_UNVERIFIED,
        "estimand": cons.ESTIMAND,
        "classifier": cons.ASYMMETRIC,
        "status": "unverified",
        key: value,
    }
    got = cons.support_trichotomy_from_bytes(_dumps(payload), exists=True, saw_begin=True)
    assert got["chronology_ok"] is False
    assert f"science_marker_unverified:{key}" in got["integrity_reasons"]
    pair = cons.cross_check_pair_receipt("completed_cli_noneligible_or_integrity_receipt", got)
    assert pair != []


def test_science_markers_unverified_integrity_fail_and_pair_reject():
    raw = _dumps(
        {
            "schema": cons.SCHEMA_UNVERIFIED,
            "estimand": cons.ESTIMAND,
            "classifier": cons.INTEGRITY,
            "status": "eligible",
            "science_classifier": cons.PRESENT,
            "signed_credit_present": True,
            "utility_delta": 0.5,
        }
    )
    got = cons.support_trichotomy_from_bytes(raw, exists=True, saw_begin=True)
    assert got["trichotomy_enum"] == "nonempty_parsed"
    assert got["chronology_ok"] is False
    reasons = got["integrity_reasons"]
    assert any(str(x).startswith("science_marker_unverified:") for x in reasons)
    assert any(str(x).startswith("science_status_unverified:") for x in reasons)
    pair = cons.cross_check_pair_receipt("completed_cli_noneligible_or_integrity_receipt", got)
    assert pair != []
    assert "pair_2_0_requires_chronology_ok" in pair or "pair_2_0_forbids_integrity_reasons" in pair


def test_clean_unverified_pair_2_0_positive_both_classifiers():
    for clf in (cons.INTEGRITY, cons.ASYMMETRIC):
        raw = _dumps(
            {
                "schema": cons.SCHEMA_UNVERIFIED,
                "estimand": cons.ESTIMAND,
                "classifier": clf,
                "status": "unverified",
            }
        )
        got = cons.support_trichotomy_from_bytes(raw, exists=True, saw_begin=True)
        assert got["trichotomy_enum"] == "nonempty_parsed"
        assert got["chronology_ok"] is True
        assert got["integrity_reasons"] == []
        assert got.get("nonscience") is True
        pair = cons.cross_check_pair_receipt("completed_cli_noneligible_or_integrity_receipt", got)
        assert pair == []


@pytest.mark.parametrize(
    "chronology_ok,expect_reject",
    [
        ("MISSING", True),
        (None, True),
        (False, True),
        (True, False),
    ],
)
def test_pair_2_0_chronology_must_be_strictly_true(chronology_ok, expect_reject):
    meta = {
        "trichotomy_enum": "nonempty_parsed",
        "integrity_reasons": [],
        "nonscience": True,
        "parsed": {"classifier": cons.INTEGRITY, "schema": cons.SCHEMA_UNVERIFIED},
    }
    if chronology_ok != "MISSING":
        meta["chronology_ok"] = chronology_ok
    pair = cons.cross_check_pair_receipt("completed_cli_noneligible_or_integrity_receipt", meta)
    if expect_reject:
        assert "pair_2_0_requires_chronology_ok" in pair
    else:
        assert pair == []


def test_pair_cross_checks_preserve_non_delta(a0_rows):
    for rid in ("PAIR_CROSS_CHECK_2_0", "PAIR_CROSS_CHECK_0_0"):
        row = next(r for r in a0_rows if r["id"] == rid)
        got = cons.cross_check_pair_receipt(row["call"]["pair_kind"], row["call"]["support_meta"])
        assert got == row["CURRENT_V18_OUTPUT"]
