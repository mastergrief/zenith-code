"""A-BANK fail-closed bank evidence (scope A safety landing).

Strict parse with NO field defaults. Formal path refuses UNRESOLVED_POLICY
before any measured producer. Does NOT synthesize parent/sibling booleans.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_bank_eval import (
    B2_RETAINED_SUPPORTS,
    evaluate_arm_bank_gate,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    ArmId,
    FailClosedClass,
)

BANK_INPUTS_INVALID = FailClosedClass.BANK_INPUTS_INVALID.value
UNRESOLVED_POLICY = "UNRESOLVED_POLICY"
REQUIRED_ARM_KEYS = tuple(a.value for a in ArmId)
REQUIRED_BLOB_FIELDS = (
    "acquire_pct",
    "retain_pct_by_support",
    "clears_by_save",
    "parent_consistency_ok",
    "close_sibling_ok",
)


class BankInputsRefuse(ValueError):
    """Maps to BANK_INPUTS_INVALID / exit 26."""

    def __init__(self, message: str, *, kind: str = "MISSING"):
        self.kind = str(kind)
        super().__init__(f"{BANK_INPUTS_INVALID}:{self.kind}: {message}")


def refuse_formal_unresolved_policy(
    *, bank_inputs: Mapping[str, Any] | None = None
) -> None:
    """Scope (A): formal NEVER measures/claims — completeness is not authority."""

    kind = "UNRESOLVED_POLICY"
    if not bank_inputs:
        kind = "MISSING"
    elif not isinstance(bank_inputs, Mapping):
        kind = "PARTIAL"
    else:
        # Caller-supplied "complete" blob still refuses — A-SPEC unresolved.
        kind = "UNRESOLVED_POLICY"
    raise BankInputsRefuse(
        "parent_consistency_ok/close_sibling_ok have no banked post-hoc "
        "provenance while A-SPEC is unresolved; formal bank evidence forbidden",
        kind=kind,
    )


def parse_required_arm_bank_blob(arm: str, blob: Mapping[str, Any]) -> dict[str, Any]:
    """Strict parse — NO field defaults (closes A-BANK-FIELD)."""

    if not isinstance(blob, Mapping):
        raise BankInputsRefuse(f"arm {arm}: blob must be a mapping", kind="PARTIAL")
    missing = [k for k in REQUIRED_BLOB_FIELDS if k not in blob]
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
            f"arm {arm}: retain_pct_by_support missing {retain_missing}",
            kind="PARTIAL",
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
        parent_ok = blob["parent_consistency_ok"]
        sibling_ok = blob["close_sibling_ok"]
        if not isinstance(parent_ok, bool) or not isinstance(sibling_ok, bool):
            raise TypeError("parent/sibling must be bool")
    except (TypeError, ValueError) as exc:
        raise BankInputsRefuse(
            f"arm {arm}: malformed fields ({exc})", kind="PARTIAL"
        ) from exc
    return {
        "acquire_pct": acquire_pct,
        "retain_pct_by_support": retain_pct,
        "clears_by_save": clears_by_save,
        "parent_consistency_ok": bool(parent_ok),
        "close_sibling_ok": bool(sibling_ok),
        "hashes_diagnostic": dict(blob.get("hashes_diagnostic") or {}),
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
            parent_consistency_ok=bool(blob["parent_consistency_ok"]),
            close_sibling_ok=bool(blob["close_sibling_ok"]),
            hashes_diagnostic=dict(blob.get("hashes_diagnostic") or {}),
        )
        for name, blob in bank_blobs.items()
    }


def resolve_bank_blobs_for_driver(
    *,
    bank_inputs: Mapping[str, Mapping[str, Any]] | None,
    developer_validation: bool,
) -> tuple[dict[str, dict[str, Any]] | None, str]:
    """Smoke may inject/suppress; formal always refuses (scope A)."""

    if not developer_validation:
        refuse_formal_unresolved_policy(bank_inputs=bank_inputs)
    if bank_inputs:
        return parse_complete_bank_inputs(bank_inputs), "injected"
    return None, "suppressed"


def apply_claim_coupling(receipt: dict, *, mode: str) -> dict:
    """A-CLAIM conjunction — materialization True is never effective authority."""

    out = dict(receipt)
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
    )
    synthetic_ledger = ledger_claimable is False or (
        isinstance(notes.get("ledger_field_provenance"), dict)
        and any(v == "SYNTHETIC" for v in notes["ledger_field_provenance"].values())
    )
    # Scope (A): measured bank is impossible → never claimable/bankable.
    out["claimable_science"] = False
    out["bankable"] = False
    if synthetic_ledger:
        out["ledger_synthetic"] = True
        out["claim_blocked_reason"] = "A-LEDGER_SYNTHETIC"
    elif mode == "formal":
        out["claim_blocked_reason"] = notes.get("bank_refuse_kind") or UNRESOLVED_POLICY
    elif not bank_ok:
        out["claim_blocked_reason"] = "BANK_EVIDENCE_INCOMPLETE"
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
    }


__all__ = [
    "BANK_INPUTS_INVALID",
    "UNRESOLVED_POLICY",
    "BankInputsRefuse",
    "REQUIRED_ARM_KEYS",
    "REQUIRED_BLOB_FIELDS",
    "refuse_formal_unresolved_policy",
    "parse_required_arm_bank_blob",
    "parse_complete_bank_inputs",
    "evaluate_parsed_bank_blobs",
    "resolve_bank_blobs_for_driver",
    "apply_claim_coupling",
    "synthetic_ledger_notes",
]
