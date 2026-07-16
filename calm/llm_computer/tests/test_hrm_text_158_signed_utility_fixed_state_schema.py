"""CPU-static tests for signed_utility_fixed_state_schema (PLAN v5)."""
from __future__ import annotations

from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_schema import (
    REQUIRED_PHASE_MARKER_NAMES,
    SchemaValidationError,
    TERMINAL_CLASSES,
    build_non_authoritative_developer_payload,
    validate_authoritative_result_schema_v4_min,
)

MOD = Path(__file__).resolve().parents[2] / "hrm_text_158/native_full_stack/signed_utility_fixed_state_schema.py"


def _valid(**over):
    markers = {n: True for n in REQUIRED_PHASE_MARKER_NAMES}
    payload = {
        "schema": "v4_min",
        "classifier": "SIGNED_CREDIT_SIGNAL_NULL_OR_HARMFUL",
        "L_prod": 1.0,
        "L_inv": 1.0,
        "L_noop": 1.0,
        "epsilon": 1e-7,
        "parent_sha256_pre": "a" * 64,
        "parent_sha256_post": "a" * 64,
        "phase_markers": markers,
        "nll_per_arm": {
            "prod": {"numerator_f64": 1.0, "denominator": 1, "mean": 1.0},
            "inv": {"numerator_f64": 1.0, "denominator": 1, "mean": 1.0},
            "noop": {"numerator_f64": 1.0, "denominator": 1, "mean": 1.0},
        },
        "apply_integer_vote_update_from_frozen_plan_calls": 4,
        "eligible_state_key_count": 2,
    }
    payload.update(over)
    return payload


def test_loc_budget():
    assert sum(1 for _ in MOD.open()) <= 160


def test_terminal_classes_and_developer_payload():
    assert "UNVERIFIED_ASYMMETRIC_INTERVENTION" in TERMINAL_CLASSES
    payload = build_non_authoritative_developer_payload({"classifier": "X"})
    assert payload["non_authoritative"] is True
    assert payload["mode"] == "developer_check"


def test_schema_requires_nll_parent_hashes_phase_markers():
    validate_authoritative_result_schema_v4_min(_valid())
    with pytest.raises(SchemaValidationError, match="missing_fields"):
        validate_authoritative_result_schema_v4_min({"classifier": "SIGNED_CREDIT_SIGNAL_NULL_OR_HARMFUL"})
    with pytest.raises(SchemaValidationError, match="phase_marker_missing"):
        bad = _valid()
        bad["phase_markers"] = {}
        validate_authoritative_result_schema_v4_min(bad)
    with pytest.raises(SchemaValidationError, match="call_count_not_two_times"):
        validate_authoritative_result_schema_v4_min(_valid(apply_integer_vote_update_from_frozen_plan_calls=3))
