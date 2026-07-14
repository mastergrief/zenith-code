"""A-SPEC Option B bank evidence + claim authority (schema v2).

Strict parse, no field defaults. Formal refuses UNRESOLVED_POLICY before
measure (safety landing). v1 boolean blobs are SCHEMA_INVALID.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_bank_eval import (
    B2_RETAINED_SUPPORTS,
    CloseSiblingReport,
    ParentFloorStatus,
    claim_blockers_from_close_sibling,
    evaluate_arm_bank_gate,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    ArmId,
    FailClosedClass,
    ParentConsistencyMechanismReceipt,
)

BANK_INPUTS_INVALID = FailClosedClass.BANK_INPUTS_INVALID.value
UNRESOLVED_POLICY = "UNRESOLVED_POLICY"
BANK_EVIDENCE_SCHEMA_VERSION = 2
REQUIRED_ARM_KEYS = tuple(a.value for a in ArmId)
V1_FORBIDDEN_FIELDS = ("parent_consistency_ok", "close_sibling_ok")
REQUIRED_V2_BLOB_FIELDS = (
    "acquire_pct", "retain_pct_by_support", "clears_by_save", "close_sibling_report",
)
BASE_FORMAL_CLAIM_BLOCKERS = ("RULE_CONFLICT_UNRESOLVED", "A_LEDGER_SYNTHETIC")


class BankInputsRefuse(ValueError):
    def __init__(self, message: str, *, kind: str = "MISSING"):
        self.kind = str(kind)
        super().__init__(f"{BANK_INPUTS_INVALID}:{self.kind}: {message}")


def refuse_formal_unresolved_policy(
    *, bank_inputs: Mapping[str, Any] | None = None
) -> None:
    kind = "MISSING" if not bank_inputs else (
        "PARTIAL" if not isinstance(bank_inputs, Mapping) else "UNRESOLVED_POLICY"
    )
    raise BankInputsRefuse(
        "formal bank evidence forbidden while measured CloseSibling producer "
        "unwired and A-SPEC Option B schema is readiness-only",
        kind=kind,
    )


def _parse_parent_floor_status(raw: Any) -> ParentFloorStatus:
    if isinstance(raw, ParentFloorStatus):
        return raw
    if raw is None:
        return ParentFloorStatus.UNRESOLVED_POLICY
    try:
        return ParentFloorStatus(str(raw).strip().upper())
    except ValueError as exc:
        raise BankInputsRefuse(
            f"parent_floor_status invalid {raw!r}", kind="PARTIAL"
        ) from exc


def parse_close_sibling_report(arm: str, raw: Any) -> CloseSiblingReport:
    if not isinstance(raw, Mapping):
        raise BankInputsRefuse(
            f"arm {arm}: close_sibling_report must be a mapping", kind="PARTIAL"
        )
    try:
        hole_count = int(raw["same_surface_parent_relative_hole_count"])
        numeric_clear = bool(raw["numeric_close_sibling_clear"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BankInputsRefuse(
            f"arm {arm}: close_sibling_report malformed ({exc})", kind="PARTIAL"
        ) from exc
    return CloseSiblingReport(
        numeric_close_sibling_clear=numeric_clear,
        same_surface_parent_relative_hole_count=hole_count,
        parent_floor_status=_parse_parent_floor_status(raw.get("parent_floor_status")),
        surfaces=tuple(str(s) for s in (raw.get("surfaces") or ())),
        candidate_scores={
            str(k): float(v) for k, v in dict(raw.get("candidate_scores") or {}).items()
        },
        parent_scores={
            str(k): float(v) for k, v in dict(raw.get("parent_scores") or {}).items()
        },
        deltas={str(k): float(v) for k, v in dict(raw.get("deltas") or {}).items()},
        evaluator_identity=str(raw.get("evaluator_identity") or ""),
        notes=dict(raw.get("notes") or {}),
    )


def parse_required_arm_bank_blob(arm: str, blob: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(blob, Mapping):
        raise BankInputsRefuse(f"arm {arm}: blob must be a mapping", kind="PARTIAL")
    forbidden = [k for k in V1_FORBIDDEN_FIELDS if k in blob]
    if forbidden:
        raise BankInputsRefuse(
            f"arm {arm}: v1 boolean fields {forbidden} are SCHEMA_INVALID "
            "(no True-migration)",
            kind="SCHEMA_INVALID",
        )
    missing = [k for k in REQUIRED_V2_BLOB_FIELDS if k not in blob]
    if missing:
        raise BankInputsRefuse(
            f"arm {arm}: missing required fields {missing}", kind="PARTIAL"
        )
    retain = blob["retain_pct_by_support"]
    if not isinstance(retain, Mapping):
        raise BankInputsRefuse(
            f"arm {arm}: retain_pct_by_support must be mapping", kind="PARTIAL"
        )
    retain_missing = [k for k in B2_RETAINED_SUPPORTS if k not in retain]
    if retain_missing:
        raise BankInputsRefuse(
            f"arm {arm}: retain_pct_by_support missing {retain_missing}", kind="PARTIAL"
        )
    clears = blob["clears_by_save"]
    if not isinstance(clears, Mapping) or not clears:
        raise BankInputsRefuse(
            f"arm {arm}: clears_by_save must be non-empty mapping", kind="PARTIAL"
        )
    try:
        clears_by_save = {int(k): bool(v) for k, v in clears.items()}
        acquire_pct = float(blob["acquire_pct"])
        retain_pct = {str(k): float(v) for k, v in retain.items()}
    except (TypeError, ValueError) as exc:
        raise BankInputsRefuse(
            f"arm {arm}: malformed fields ({exc})", kind="PARTIAL"
        ) from exc
    report = parse_close_sibling_report(arm, blob["close_sibling_report"])
    return {
        "schema_version": BANK_EVIDENCE_SCHEMA_VERSION,
        "acquire_pct": acquire_pct,
        "retain_pct_by_support": retain_pct,
        "clears_by_save": clears_by_save,
        "close_sibling_report": report,
        "hashes_diagnostic": dict(blob.get("hashes_diagnostic") or {}),
        "claim_blockers_extra": claim_blockers_from_close_sibling(report),
    }


def parse_complete_bank_inputs(
    bank_inputs: Mapping[str, Mapping[str, Any]] | None,
    *,
    require_arms: Sequence[str] = REQUIRED_ARM_KEYS,
) -> dict[str, dict[str, Any]]:
    if not bank_inputs:
        raise BankInputsRefuse(
            "bank_inputs missing/empty (fabrication forbidden)", kind="MISSING"
        )
    if not isinstance(bank_inputs, Mapping):
        raise BankInputsRefuse("bank_inputs must be a mapping", kind="PARTIAL")
    missing_arms = [a for a in require_arms if a not in bank_inputs]
    if missing_arms:
        raise BankInputsRefuse(f"missing arm keys {missing_arms}", kind="PARTIAL")
    extra = [k for k in bank_inputs if k not in require_arms]
    if extra:
        raise BankInputsRefuse(f"unknown arm keys {extra}", kind="PARTIAL")
    return {a: parse_required_arm_bank_blob(a, bank_inputs[a]) for a in require_arms}


def evaluate_parsed_bank_blobs(bank_blobs: Mapping[str, Mapping[str, Any]]):
    return {
        name: evaluate_arm_bank_gate(
            arm=name,
            acquire_pct=float(blob["acquire_pct"]),
            retain_pct_by_support=dict(blob["retain_pct_by_support"]),
            clears_by_save=dict(blob["clears_by_save"]),
            close_sibling_report=blob.get("close_sibling_report"),
            hashes_diagnostic=dict(blob.get("hashes_diagnostic") or {}),
        )
        for name, blob in bank_blobs.items()
    }


def resolve_bank_blobs_for_driver(
    *,
    bank_inputs: Mapping[str, Mapping[str, Any]] | None,
    developer_validation: bool,
) -> tuple[dict[str, dict[str, Any]] | None, str]:
    if not developer_validation:
        refuse_formal_unresolved_policy(bank_inputs=bank_inputs)
    if bank_inputs:
        return parse_complete_bank_inputs(bank_inputs), "injected"
    return None, "suppressed"


def build_parent_consistency_mechanism_receipt(
    *,
    expected_parent_sha256: str,
    observed_parent_sha256: str,
    notes: Mapping[str, Any] | None = None,
) -> ParentConsistencyMechanismReceipt:
    match = str(expected_parent_sha256) == str(observed_parent_sha256)
    return ParentConsistencyMechanismReceipt(
        expected_parent_sha256=str(expected_parent_sha256),
        observed_parent_sha256=str(observed_parent_sha256),
        match=match,
        recipe_compliance_ok=match,
        notes=dict(notes or {}),
    )


def collect_formal_claim_blockers(
    *,
    bank_blobs: Mapping[str, Mapping[str, Any]] | None = None,
    ledger_claimable: bool | None = None,
    extra: Sequence[str] = (),
) -> list[str]:
    blockers = list(BASE_FORMAL_CLAIM_BLOCKERS)
    if bank_blobs:
        for blob in bank_blobs.values():
            for b in blob.get("claim_blockers_extra") or ():
                if b not in blockers:
                    blockers.append(str(b))
    for b in extra:
        if b not in blockers:
            blockers.append(str(b))
    return blockers


def stamp_mode_labels(receipt: dict, *, mode: str) -> dict:
    """Non-claim fields only — never sets claimable_science/bankable True."""

    out = dict(receipt)
    if mode == "smoke":
        out["run_kind"] = "REAL_DEVICE_SMOKE"
        out["science_label"] = None
        for key in (
            "bank_gate_pass", "acquire_rate", "retain_rate", "bpw",
            "earliest_all_clear", "90/90", "acquire_pct", "retain_pct_by_support",
        ):
            out.pop(key, None)
        if isinstance(out.get("bank_receipts"), dict):
            out["bank_receipts"] = None
            out["bank_receipts_suppressed"] = True
    else:
        out["run_kind"] = "FORMAL_SCIENCE"
        out.setdefault("science_label", "forgotten_accum_training_equivalence_4arm")
    return out


def apply_claim_coupling(receipt: dict, *, mode: str) -> dict:
    """Safe claim reducer — never True from mode alone."""

    out = stamp_mode_labels(receipt, mode=mode)
    notes = out.get("notes") if isinstance(out.get("notes"), dict) else {}
    ledger_claimable = notes.get("ledger_claimable")
    bank_section = notes.get("bank_section")
    bank_receipts = out.get("bank_receipts")
    bank_ok = (
        isinstance(bank_receipts, dict)
        and bank_receipts
        and all(bool(v.get("bank_clear")) for v in bank_receipts.values())
        and bank_section == "measured"
        and not notes.get("bank_receipts_suppressed")
        and all(not list(v.get("bank_blockers") or []) for v in bank_receipts.values())
    )
    synthetic_ledger = ledger_claimable is False or (
        isinstance(notes.get("ledger_field_provenance"), dict)
        and any(v == "SYNTHETIC" for v in notes["ledger_field_provenance"].values())
    )
    blockers = collect_formal_claim_blockers(
        ledger_claimable=False if synthetic_ledger else ledger_claimable,
        extra=list(notes.get("formal_claim_blockers") or ()),
    )
    notes = dict(notes)
    notes["formal_claim_blockers"] = list(blockers)
    out["notes"] = notes
    out["claimable_science"] = False
    out["bankable"] = False
    if blockers:
        out["claim_blocked_reason"] = blockers[0]
    elif mode == "formal":
        out["claim_blocked_reason"] = notes.get("bank_refuse_kind") or UNRESOLVED_POLICY
    elif not bank_ok:
        out["claim_blocked_reason"] = "BANK_EVIDENCE_INCOMPLETE"
    if synthetic_ledger:
        out["ledger_synthetic"] = True
    return out


def synthetic_ledger_notes() -> dict[str, Any]:
    return {
        "ledger_field_provenance": {
            "forward_count": "SYNTHETIC",
            "backward_count": "SYNTHETIC",
            "update_count": "SYNTHETIC",
            "gpu_time_seconds": "SYNTHETIC",
            "rewarm_examples_seen": "SYNTHETIC",
        },
        "ledger_claimable": False,
        "formal_claim_blockers": list(BASE_FORMAL_CLAIM_BLOCKERS),
    }


__all__ = [
    "BANK_INPUTS_INVALID", "BANK_EVIDENCE_SCHEMA_VERSION", "UNRESOLVED_POLICY",
    "BASE_FORMAL_CLAIM_BLOCKERS", "V1_FORBIDDEN_FIELDS", "BankInputsRefuse",
    "ParentConsistencyMechanismReceipt", "REQUIRED_ARM_KEYS", "REQUIRED_V2_BLOB_FIELDS",
    "refuse_formal_unresolved_policy", "parse_close_sibling_report",
    "parse_required_arm_bank_blob", "parse_complete_bank_inputs",
    "evaluate_parsed_bank_blobs", "resolve_bank_blobs_for_driver",
    "build_parent_consistency_mechanism_receipt", "collect_formal_claim_blockers",
    "stamp_mode_labels", "apply_claim_coupling", "synthetic_ledger_notes",
]
