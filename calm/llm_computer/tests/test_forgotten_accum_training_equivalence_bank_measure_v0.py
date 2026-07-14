"""Unit tests for bank_measure parse/refuse seam (Option B schema v2)."""
from __future__ import annotations

import pytest

from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_bank_measure import (
    BankInputsRefuse,
    evaluate_parsed_bank_blobs,
    parse_complete_bank_inputs,
    parse_required_arm_bank_blob,
    refuse_formal_unresolved_policy,
    resolve_bank_blobs_for_driver,
)


def _sibling(**kw):
    base = {
        "numeric_close_sibling_clear": True,
        "same_surface_parent_relative_hole_count": 0,
        "parent_floor_status": "UNRESOLVED_POLICY",
    }
    base.update(kw)
    return base


def _blob(**kw):
    base = {
        "acquire_pct": 95.0,
        "retain_pct_by_support": {"L0b": 91.0, "math_a0": 92.0},
        "clears_by_save": {250: True, 1500: False},
        "close_sibling_report": _sibling(),
    }
    base.update(kw)
    return base


def test_parse_and_evaluate_complete_injected_bank():
    raw = {a: _blob() for a in ("U", "E", "R0", "RW")}
    parsed = parse_complete_bank_inputs(raw)
    receipts = evaluate_parsed_bank_blobs(parsed)
    assert all(r.bank_clear for r in receipts.values())
    assert receipts["U"].earliest_all_clear_save == 250


def test_v1_boolean_fields_schema_invalid():
    blob = _blob(parent_consistency_ok=True, close_sibling_ok=True)
    with pytest.raises(BankInputsRefuse, match="SCHEMA_INVALID"):
        parse_required_arm_bank_blob("U", blob)


def test_literal_default_path_cannot_synthesize():
    blob = _blob()
    del blob["close_sibling_report"]
    with pytest.raises(BankInputsRefuse, match="PARTIAL"):
        parse_required_arm_bank_blob("U", blob)


def test_malformed_clears_refuse():
    with pytest.raises(BankInputsRefuse, match="PARTIAL"):
        parse_required_arm_bank_blob("U", _blob(clears_by_save={}))


def test_formal_resolve_always_refuses_even_complete():
    raw = {a: _blob() for a in ("U", "E", "R0", "RW")}
    with pytest.raises(BankInputsRefuse, match="UNRESOLVED_POLICY"):
        resolve_bank_blobs_for_driver(bank_inputs=raw, developer_validation=False)
    with pytest.raises(BankInputsRefuse, match="MISSING"):
        resolve_bank_blobs_for_driver(bank_inputs=None, developer_validation=False)


def test_smoke_suppresses_absent_and_allows_injected_parse():
    blobs, section = resolve_bank_blobs_for_driver(
        bank_inputs=None, developer_validation=True
    )
    assert blobs is None and section == "suppressed"
    raw = {a: _blob() for a in ("U", "E", "R0", "RW")}
    blobs, section = resolve_bank_blobs_for_driver(
        bank_inputs=raw, developer_validation=True
    )
    assert section == "injected" and set(blobs) == {"U", "E", "R0", "RW"}


def test_refuse_formal_helper():
    with pytest.raises(BankInputsRefuse, match="UNRESOLVED_POLICY"):
        refuse_formal_unresolved_policy(bank_inputs={"U": _blob()})
