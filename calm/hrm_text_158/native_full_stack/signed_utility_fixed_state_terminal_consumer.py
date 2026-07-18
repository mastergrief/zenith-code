"""Canonical stdlib-only terminal receipt consumer (PLAN v28 Step A).

Owns support_trichotomy_from_bytes + cross_check_pair_receipt. Preserves V18
non-delta behavior; accepts unverified_v4 with exactly the two allowed
unverified classifiers as nonscience; fails closed on invented classifiers and
NLL-bearing unverified payloads.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

# Embedded vocabulary (parity asserted by tests against schema / legal_subset).
ESTIMAND = "full_state_legal_subset_signed_direction_fixed_state_heldout_utility"
SCHEMA_SCIENCE = "post_seam_signed_utility_authoritative_result_science_v4"
SCHEMA_UNVERIFIED = "post_seam_signed_utility_authoritative_result_unverified_v4"
SCHEMA_OPERATOR_TERMINAL = "three_arm_heldout_nll_operator_terminal_v1"
INTEGRITY = "UNVERIFIED_INTEGRITY_OR_EXECUTION"
ASYMMETRIC = "UNVERIFIED_ASYMMETRIC_INTERVENTION"
ELIGIBLE = "SIGNED_CREDIT_SIGNAL_PRESENT_UNPROVEN"
PRESENT = "SIGNED_CREDIT_SIGNAL_PRESENT_UNPROVEN"
NULL_OR_HARMFUL = "SIGNED_CREDIT_SIGNAL_NULL_OR_HARMFUL"
ALLOWED_UNVERIFIED_CLASSIFIERS = frozenset({INTEGRITY, ASYMMETRIC})
SCIENCE_ELIGIBILITY_CLASSIFIERS = frozenset({ELIGIBLE, PRESENT, NULL_OR_HARMFUL})
SUPPORT_CLASSIFIERS = frozenset({INTEGRITY, ASYMMETRIC, NULL_OR_HARMFUL, PRESENT})
NLL_BEARING_KEYS = frozenset({"L_prod", "L_inv", "L_noop", "L_noop_repeat", "epsilon"})
SCIENCE_MARKER_KEYS = frozenset({"science_classifier", "signed_credit_present", "utility_delta"})
FORBIDDEN_UNVERIFIED_STATUS = frozenset({
    "eligible",
    "present",
    "null",
    "science",
    ELIGIBLE,
    PRESENT,
    NULL_OR_HARMFUL,
})
MAX_RECEIPT = 256 * 1024


def support_trichotomy_from_bytes(raw: bytes | None, *, exists: bool, saw_begin: bool) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "exists": exists,
        "size_bytes": 0 if raw is None else len(raw),
        "sha256_or_null": None,
        "trichotomy_enum": "absent_pre_begin",
        "chronology_ok": True,
        "chronology_proof": "absent_and_no_SUPPORT_BEGIN",
        "parsed": None,
        "integrity_reasons": [],
    }
    if not exists:
        if saw_begin:
            meta.update(
                trichotomy_enum="integrity_absent_after_begin",
                chronology_ok=False,
                chronology_proof="SUPPORT_BEGIN_seen_but_receipt_absent",
            )
            meta["integrity_reasons"].append("absent_after_begin")
        return meta
    assert raw is not None
    meta["sha256_or_null"] = hashlib.sha256(raw).hexdigest()
    if len(raw) == 0:
        if not saw_begin:
            meta.update(
                trichotomy_enum="integrity_zero_without_begin",
                chronology_ok=False,
                chronology_proof="zero_byte_without_SUPPORT_BEGIN",
            )
            meta["integrity_reasons"].append("zero_byte_without_begin")
        else:
            meta.update(
                trichotomy_enum="zero_byte_reserved_interrupted",
                chronology_ok=True,
                chronology_proof="SUPPORT_BEGIN_then_zero_byte_reserved",
            )
        return meta
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        meta.update(trichotomy_enum="nonempty_unparsed", chronology_ok=False)
        meta["integrity_reasons"].append(f"unparsed:{exc}")
        return meta
    if not isinstance(obj, dict):
        meta.update(trichotomy_enum="nonempty_unparsed", chronology_ok=False)
        meta["integrity_reasons"].append("not_object")
        return meta
    schema = obj.get("schema") or obj.get("schema_version")
    if schema == SCHEMA_UNVERIFIED:
        clf = obj.get("classifier")
        if clf in ALLOWED_UNVERIFIED_CLASSIFIERS:
            return _consume_unverified_v4(meta, obj, raw)
        # Preserve V18 fail-closed shape for invented/unknown unverified classifiers.
        meta.update(trichotomy_enum="nonempty_unparsed", chronology_ok=False)
        meta["integrity_reasons"].append(f"bad_schema:{schema}")
        return meta
    if schema not in (SCHEMA_SCIENCE, SCHEMA_OPERATOR_TERMINAL):
        meta.update(trichotomy_enum="nonempty_unparsed", chronology_ok=False)
        meta["integrity_reasons"].append(f"bad_schema:{schema}")
        return meta

    # science_v4 / operator_terminal_v1 — preserve V18 non-delta behavior
    if obj.get("estimand") not in (None, ESTIMAND) and obj.get("estimand") != ESTIMAND:
        meta["integrity_reasons"].append("bad_estimand")
        meta["chronology_ok"] = False
    for k, v in obj.items():
        if isinstance(v, list) and len(v) > 32:
            meta["integrity_reasons"].append(f"raw_array_field:{k}")
            meta["chronology_ok"] = False
    if obj.get("classifier") not in SUPPORT_CLASSIFIERS:
        meta["integrity_reasons"].append(f"bad_classifier:{obj.get('classifier')}")
        meta["chronology_ok"] = False
    if len(raw) > MAX_RECEIPT:
        meta["integrity_reasons"].append("oversize_support_receipt")
        meta["chronology_ok"] = False
    meta.update(
        parsed=obj,
        trichotomy_enum="nonempty_parsed",
        chronology_proof="parsed_post_seam_signed_utility_authoritative_result_science_v4",
    )
    if meta["integrity_reasons"]:
        meta["chronology_ok"] = False
    return meta


def _consume_unverified_v4(meta: dict[str, Any], obj: dict[str, Any], raw: bytes) -> dict[str, Any]:
    if obj.get("estimand") not in (None, ESTIMAND) and obj.get("estimand") != ESTIMAND:
        meta["integrity_reasons"].append("bad_estimand")
    for k, v in obj.items():
        if isinstance(v, list) and len(v) > 32:
            meta["integrity_reasons"].append(f"raw_array_field:{k}")
    for key in NLL_BEARING_KEYS:
        if key in obj:
            meta["integrity_reasons"].append(f"nll_bearing_unverified:{key}")
    for key in SCIENCE_MARKER_KEYS:
        if key in obj:
            meta["integrity_reasons"].append(f"science_marker_unverified:{key}")
    status = obj.get("status")
    if isinstance(status, str) and status in FORBIDDEN_UNVERIFIED_STATUS:
        meta["integrity_reasons"].append(f"science_status_unverified:{status}")
    elif isinstance(status, str) and status.lower() in {"eligible", "present", "null", "science"}:
        meta["integrity_reasons"].append(f"science_status_unverified:{status}")
    if len(raw) > MAX_RECEIPT:
        meta["integrity_reasons"].append("oversize_support_receipt")
    meta.update(
        parsed=obj,
        trichotomy_enum="nonempty_parsed",
        chronology_proof="parsed_post_seam_signed_utility_authoritative_result_unverified_v4_nonscience",
        nonscience=True,
    )
    if meta["integrity_reasons"]:
        meta["chronology_ok"] = False
    else:
        meta["chronology_ok"] = True
    return meta


def cross_check_pair_receipt(pair_kind: str, support_meta: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    parsed = support_meta.get("parsed") or {}
    clf = parsed.get("classifier")
    if pair_kind == "completed_eligible":
        if support_meta.get("trichotomy_enum") != "nonempty_parsed" or clf != ELIGIBLE:
            reasons.append("pair_0_0_requires_ELIGIBLE_receipt")
        if support_meta.get("nonscience") is True:
            reasons.append("pair_0_0_forbids_nonscience_unverified")
    if pair_kind == "completed_cli_noneligible_or_integrity_receipt":
        if support_meta.get("trichotomy_enum") != "nonempty_parsed":
            reasons.append("pair_2_0_requires_parsed_noneligible_or_integrity_receipt")
        elif support_meta.get("chronology_ok") is not True:
            reasons.append("pair_2_0_requires_chronology_ok")
        elif support_meta.get("integrity_reasons"):
            reasons.append("pair_2_0_forbids_integrity_reasons")
        elif clf == ELIGIBLE:
            reasons.append("pair_2_0_with_ELIGIBLE_receipt")
        elif clf not in (SUPPORT_CLASSIFIERS - {ELIGIBLE}):
            reasons.append("pair_2_0_bad_classifier")
    return reasons


__all__ = [
    "ALLOWED_UNVERIFIED_CLASSIFIERS",
    "ASYMMETRIC",
    "ELIGIBLE",
    "ESTIMAND",
    "FORBIDDEN_UNVERIFIED_STATUS",
    "INTEGRITY",
    "MAX_RECEIPT",
    "NULL_OR_HARMFUL",
    "PRESENT",
    "SCHEMA_OPERATOR_TERMINAL",
    "SCHEMA_SCIENCE",
    "SCHEMA_UNVERIFIED",
    "SCIENCE_ELIGIBILITY_CLASSIFIERS",
    "SCIENCE_MARKER_KEYS",
    "SUPPORT_CLASSIFIERS",
    "cross_check_pair_receipt",
    "support_trichotomy_from_bytes",
]
