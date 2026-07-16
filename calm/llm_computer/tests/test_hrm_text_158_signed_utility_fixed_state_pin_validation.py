"""CPU-static tests for signed_utility_fixed_state_pin_validation (PLAN v5)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_pin_validation import (
    PinValidationError,
    WATCH_WRAP_CLAW_CODE_FORBIDDEN_SHA256,
    WATCH_WRAP_HRM158_SHA256,
    rehash_path,
    validate_proof_packet_source_pins,
)

MOD = Path(__file__).resolve().parents[2] / "hrm_text_158/native_full_stack/signed_utility_fixed_state_pin_validation.py"
WATCH = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158/bin/watch-wrap")
VOTE = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158/calm/hrm_text_158/native_full_stack/vote_update.py")


def test_loc_budget():
    assert sum(1 for _ in MOD.open()) <= 160


def test_absolute_path_and_sha_required(tmp_path: Path):
    with pytest.raises(PinValidationError, match="source_pins_missing"):
        validate_proof_packet_source_pins({})
    with pytest.raises(PinValidationError, match="pin_requires_absolute_path_and_sha256"):
        validate_proof_packet_source_pins({"source_pins": {"x": {"sha256": "abc"}}})


def test_pin_mismatch_fail_closed(tmp_path: Path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    with pytest.raises(PinValidationError, match="pin_sha_mismatch"):
        validate_proof_packet_source_pins(
            {"source_pins": {"a": {"absolute_path": str(f), "sha256": "0" * 64}}}
        )


def test_watch_wrap_must_be_hrm158_not_claw_code():
    digest = rehash_path(WATCH)
    assert digest == WATCH_WRAP_HRM158_SHA256
    assert digest != WATCH_WRAP_CLAW_CODE_FORBIDDEN_SHA256
    observed = validate_proof_packet_source_pins(
        {
            "source_pins": {
                "watch_wrap": {"absolute_path": str(WATCH), "sha256": digest},
                "vote_update": {"absolute_path": str(VOTE), "sha256": rehash_path(VOTE)},
            }
        }
    )
    assert observed["watch_wrap"] == WATCH_WRAP_HRM158_SHA256
