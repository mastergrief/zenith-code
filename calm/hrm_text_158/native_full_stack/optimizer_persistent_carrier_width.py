"""Fail-closed persistent carrier width narrowability receipt for HRM-Text-1.58.

CPU/audit lane only. Classifies whether persistent accumulator train-state WIDTH
can be encoded in a sub-2-bit carrier under a declared proof law. Separate from
optimizer_credit_state row flip, learning preservation, and readiness claims.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

OPTIMIZER_PERSISTENT_CARRIER_WIDTH_NARROWABILITY_SCHEMA_VERSION = (
    "hrm_text_158_optimizer_persistent_carrier_width_narrowability/v0.fail_closed"
)
OPTIMIZER_PERSISTENT_CARRIER_WIDTH_NARROWABILITY_TARGET_NAME = (
    "optimizer_persistent_carrier_width_narrowability"
)

LOCKED_THRESHOLD_LOCK_GATE_ID = "1781642632005-06bc55a8"

PROOF_LAW_LOCKED_ARM_A = "LOCKED_ARM_A"
PROOF_LAW_HISTORICAL_NO_DRAIN_CONTROL = "HISTORICAL_NO_DRAIN_CONTROL"
PROOF_LAW_IDS = (PROOF_LAW_LOCKED_ARM_A, PROOF_LAW_HISTORICAL_NO_DRAIN_CONTROL)

PARENT_SHA_LOCKED_ARM_A_9DB27EE4 = (
    "9db27ee4543dac49954873fe586ba1d6769000e4081fbb8b155ef5bdc7ef45ef"
)
PARENT_SHA_HISTORICAL_V8C2_4DDEACC8 = (
    "4ddeacc84a4bca05e1a75307af967500c39d4491189141c6749d21e1372bc5be"
)

BRANCH_INT4_VIABLE = "BR-CARRIER-INT4-VIABLE"
BRANCH_INT8_REQUIRED = "BR-CARRIER-INT8-REQUIRED"
BRANCH_INT16_REQUIRED = "BR-CARRIER-INT16-REQUIRED"
BRANCH_BLOCKED_BY_LEARNING = "BR-CARRIER-NARROW-BLOCKED-BY-LEARNING"
BRANCH_BLOCKED_BY_LAW_MISMATCH = "BR-CARRIER-NARROW-BLOCKED-BY-LAW-MISMATCH"
BRANCH_MEASUREMENT_INVALID = "BR-CARRIER-MEASUREMENT-INVALID"

LEARNING_NOT_MEASURED = "not_measured"
LEARNING_PRESERVES = "preserves_learning"
LEARNING_DAMAGES = "damages_learning"
LEARNING_UNCALIBRATED = "uncalibrated_proxy"

INT4_MAX_ABS = 7
INT8_MAX_ABS = 127

NARROWABILITY_NON_CLAIMS = (
    "persistent carrier width narrowability is not learning, acquisition, retention, or throughput",
    "width encoding viable is a candidate encoding observation, not a carrier win or trainability claim",
    "ready_to_flip and optimizer_credit_state_sub2_claim remain false in this receipt class",
    "learning_co_gate_verdict is separate from width encoding viability",
    "this receipt does not launch GPU, flip readiness rows, write checkpoints, or mutate .pt artifacts",
)


@dataclass(frozen=True)
class OptimizerPersistentCarrierWidthNarrowabilityReceipt:
    schema_version: str
    target_name: str
    lock_gate_id: str
    proof_law_id: str
    parent_receipt_sha256: str
    control_only: bool
    peak_decoded_abs_max_over_run: int
    peak_frac_over_int4: float
    peak_frac_over_int8: float
    peak_plateau_verified: bool
    carrier_int4_viable: bool
    carrier_int8_viable: bool
    carrier_int16_required: bool
    encoding_round_trip_max_delta: int
    width_encoding_viable: bool
    branch_id: str
    learning_co_gate_verdict: str
    ready_to_flip: bool
    optimizer_credit_state_sub2_claim: bool
    readiness_row_flip_authorized: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            if field.name != "non_claims"
            else list(self.non_claims)
            for field in fields(self)
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encoding_round_trip_max_delta(peak_abs: int) -> tuple[int, bool, bool, bool]:
    """Return (max_delta, int4_viable, int8_viable, int16_required) for peak scalar."""

    peak = int(peak_abs)
    if peak < 0:
        raise ValueError("peak_decoded_abs_max_over_run must be non-negative")
    int4_clamped = max(-INT4_MAX_ABS, min(INT4_MAX_ABS, peak))
    int8_clamped = max(-INT8_MAX_ABS, min(INT8_MAX_ABS, peak))
    int4_delta = abs(peak - int4_clamped)
    int8_delta = abs(peak - int8_clamped)
    int4_viable = peak <= INT4_MAX_ABS and int4_delta == 0
    int8_viable = peak <= INT8_MAX_ABS and int8_delta == 0
    int16_required = peak > INT8_MAX_ABS or int8_delta > 0
    return max(int4_delta, int8_delta), int4_viable, int8_viable, int16_required


def _learning_verdict_from_parent(parent: Mapping[str, Any]) -> str:
    fail_class = str(parent.get("real_credit_fail_class", "") or "")
    if fail_class == "bounds_but_damages_learning":
        return LEARNING_DAMAGES
    if fail_class == "bounds_and_preserves_learning":
        return LEARNING_PRESERVES
    if parent.get("ce_proxy_all_steps_within_tolerance") is True:
        return LEARNING_PRESERVES
    if parent.get("ce_proxy_all_steps_within_tolerance") is False:
        return LEARNING_DAMAGES
    return LEARNING_NOT_MEASURED


def _expected_proof_law_for_parent(parent_sha: str) -> str | None:
    normalized = parent_sha.lower()
    if normalized == PARENT_SHA_LOCKED_ARM_A_9DB27EE4:
        return PROOF_LAW_LOCKED_ARM_A
    if normalized == PARENT_SHA_HISTORICAL_V8C2_4DDEACC8:
        return PROOF_LAW_HISTORICAL_NO_DRAIN_CONTROL
    return None


def _classify_branch(
    *,
    proof_law_id: str,
    parent_sha: str,
    peak: int,
    frac_over_int4: float,
    int4_viable: bool,
    int8_viable: bool,
    int16_required: bool,
    learning_verdict: str,
    measurement_valid: bool,
) -> str:
    expected = _expected_proof_law_for_parent(parent_sha)
    if expected is not None and proof_law_id != expected:
        return BRANCH_BLOCKED_BY_LAW_MISMATCH
    if not measurement_valid:
        return BRANCH_MEASUREMENT_INVALID
    if int16_required:
        return BRANCH_INT16_REQUIRED
    width_viable = int4_viable or int8_viable
    if width_viable and learning_verdict == LEARNING_DAMAGES:
        return BRANCH_BLOCKED_BY_LEARNING
    if int4_viable and frac_over_int4 == 0.0:
        return BRANCH_INT4_VIABLE
    if int8_viable:
        return BRANCH_INT8_REQUIRED
    return BRANCH_MEASUREMENT_INVALID


def load_parent_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"parent receipt missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("parent receipt must be a JSON object")
    return payload


def extract_peak_fields(parent: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "peak_global_max_abs",
        "peak_frac_over_int4",
        "peak_frac_over_int8",
    )
    missing = [key for key in required if key not in parent]
    if missing:
        raise ValueError(f"parent receipt missing peak fields: {missing}")
    return {
        "peak_decoded_abs_max_over_run": int(parent["peak_global_max_abs"]),
        "peak_frac_over_int4": float(parent["peak_frac_over_int4"]),
        "peak_frac_over_int8": float(parent["peak_frac_over_int8"]),
        "peak_plateau_verified": bool(parent.get("plateau_valid", False)),
    }


def build_optimizer_persistent_carrier_width_narrowability_receipt(
    *,
    proof_law_id: str,
    parent_receipt_sha256: str,
    peak_decoded_abs_max_over_run: int,
    peak_frac_over_int4: float,
    peak_frac_over_int8: float,
    peak_plateau_verified: bool = False,
    control_only: bool = False,
    learning_co_gate_verdict: str = LEARNING_NOT_MEASURED,
    lock_gate_id: str = LOCKED_THRESHOLD_LOCK_GATE_ID,
) -> OptimizerPersistentCarrierWidthNarrowabilityReceipt:
    if proof_law_id not in PROOF_LAW_IDS:
        raise ValueError(f"unknown proof_law_id: {proof_law_id!r}")
    if lock_gate_id != LOCKED_THRESHOLD_LOCK_GATE_ID:
        raise ValueError("lock_gate_id mismatch")
    parent_sha = str(parent_receipt_sha256).lower()
    expected = _expected_proof_law_for_parent(parent_sha)
    if expected is not None and proof_law_id != expected:
        raise ValueError("parent_receipt_sha256 incompatible with proof_law_id")
    if proof_law_id == PROOF_LAW_HISTORICAL_NO_DRAIN_CONTROL and not control_only:
        raise ValueError(
            "HISTORICAL_NO_DRAIN_CONTROL requires control_only=true"
        )
    if proof_law_id == PROOF_LAW_LOCKED_ARM_A and control_only:
        raise ValueError("LOCKED_ARM_A cannot set control_only=true")

    round_trip_delta, int4_viable, int8_viable, int16_required = (
        _encoding_round_trip_max_delta(peak_decoded_abs_max_over_run)
    )
    if int4_viable and peak_frac_over_int4 != 0.0:
        int4_viable = False
    measurement_valid = peak_decoded_abs_max_over_run >= 0
    branch_id = _classify_branch(
        proof_law_id=proof_law_id,
        parent_sha=parent_sha,
        peak=peak_decoded_abs_max_over_run,
        frac_over_int4=peak_frac_over_int4,
        int4_viable=int4_viable,
        int8_viable=int8_viable,
        int16_required=int16_required,
        learning_verdict=learning_co_gate_verdict,
        measurement_valid=measurement_valid,
    )
    width_encoding_viable = bool(int4_viable or int8_viable)
    receipt = OptimizerPersistentCarrierWidthNarrowabilityReceipt(
        schema_version=OPTIMIZER_PERSISTENT_CARRIER_WIDTH_NARROWABILITY_SCHEMA_VERSION,
        target_name=OPTIMIZER_PERSISTENT_CARRIER_WIDTH_NARROWABILITY_TARGET_NAME,
        lock_gate_id=lock_gate_id,
        proof_law_id=proof_law_id,
        parent_receipt_sha256=parent_sha,
        control_only=bool(control_only),
        peak_decoded_abs_max_over_run=int(peak_decoded_abs_max_over_run),
        peak_frac_over_int4=float(peak_frac_over_int4),
        peak_frac_over_int8=float(peak_frac_over_int8),
        peak_plateau_verified=bool(peak_plateau_verified),
        carrier_int4_viable=bool(int4_viable),
        carrier_int8_viable=bool(int8_viable),
        carrier_int16_required=bool(int16_required),
        encoding_round_trip_max_delta=int(round_trip_delta),
        width_encoding_viable=bool(width_encoding_viable),
        branch_id=branch_id,
        learning_co_gate_verdict=str(learning_co_gate_verdict),
        ready_to_flip=False,
        optimizer_credit_state_sub2_claim=False,
        readiness_row_flip_authorized=False,
        non_claims=NARROWABILITY_NON_CLAIMS,
    )
    validate_optimizer_persistent_carrier_width_narrowability_receipt(receipt)
    return receipt


def classify_from_parent_receipt_file(
    path: Path,
    *,
    proof_law_id: str,
    control_only: bool = False,
    learning_co_gate_verdict: str | None = None,
) -> OptimizerPersistentCarrierWidthNarrowabilityReceipt:
    parent = load_parent_receipt(path)
    parent_sha = _sha256_file(path)
    peaks = extract_peak_fields(parent)
    learning = (
        learning_co_gate_verdict
        if learning_co_gate_verdict is not None
        else _learning_verdict_from_parent(parent)
    )
    return build_optimizer_persistent_carrier_width_narrowability_receipt(
        proof_law_id=proof_law_id,
        parent_receipt_sha256=parent_sha,
        learning_co_gate_verdict=learning,
        control_only=control_only,
        **peaks,
    )


def validate_optimizer_persistent_carrier_width_narrowability_receipt(
    receipt: OptimizerPersistentCarrierWidthNarrowabilityReceipt,
) -> None:
    if (
        receipt.schema_version
        != OPTIMIZER_PERSISTENT_CARRIER_WIDTH_NARROWABILITY_SCHEMA_VERSION
    ):
        raise ValueError("narrowability receipt schema mismatch")
    if receipt.target_name != OPTIMIZER_PERSISTENT_CARRIER_WIDTH_NARROWABILITY_TARGET_NAME:
        raise ValueError("narrowability receipt target mismatch")
    if receipt.lock_gate_id != LOCKED_THRESHOLD_LOCK_GATE_ID:
        raise ValueError("narrowability receipt lock_gate_id mismatch")
    if receipt.proof_law_id not in PROOF_LAW_IDS:
        raise ValueError("narrowability receipt proof_law_id invalid")
    if receipt.ready_to_flip or receipt.optimizer_credit_state_sub2_claim:
        raise ValueError("narrowability receipt forbids flip/sub2 claims")
    if receipt.readiness_row_flip_authorized:
        raise ValueError("narrowability receipt forbids readiness_row_flip_authorized")
    expected = _expected_proof_law_for_parent(receipt.parent_receipt_sha256)
    if expected is not None and receipt.proof_law_id != expected:
        raise ValueError("parent_receipt_sha256 incompatible with proof_law_id")
    if (
        receipt.proof_law_id == PROOF_LAW_HISTORICAL_NO_DRAIN_CONTROL
        and not receipt.control_only
    ):
        raise ValueError(
            "HISTORICAL_NO_DRAIN_CONTROL requires control_only=true"
        )
    if receipt.carrier_int4_viable and receipt.encoding_round_trip_max_delta > 0:
        raise ValueError("int4 viability requires zero encoding round-trip delta")
    if receipt.branch_id == BRANCH_INT4_VIABLE and not receipt.carrier_int4_viable:
        raise ValueError("BR-CARRIER-INT4-VIABLE requires carrier_int4_viable")
    encoding_viable = receipt.carrier_int4_viable or receipt.carrier_int8_viable
    if receipt.width_encoding_viable != encoding_viable:
        raise ValueError(
            "width_encoding_viable must equal carrier_int4_viable or carrier_int8_viable"
        )
    if receipt.non_claims != NARROWABILITY_NON_CLAIMS:
        raise ValueError("narrowability receipt non_claims must be exact")
