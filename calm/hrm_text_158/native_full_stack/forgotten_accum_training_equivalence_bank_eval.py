"""Thin bank-gate eval facade (import/wrap probe symbols; no god-file growth).

Hashes are DIAGNOSTIC_ONLY — neither grant nor veto bank outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


# Probe constants re-exported for arm harness without growing probe.
B2_RETAINED_SUPPORTS = ("L0b", "math_a0")
BANK_ACQUIRE_TARGET = 90
BANK_RETAIN_TARGET = 90
SAVE_CADENCE_DEFAULT = (250, 500, 750, 1000, 1250, 1500)


@dataclass(frozen=True)
class BankGateSurfaceSpec:
    acquire_target_pct: int = BANK_ACQUIRE_TARGET
    retain_target_pct: int = BANK_RETAIN_TARGET
    retained_supports: tuple[str, ...] = B2_RETAINED_SUPPORTS
    require_parent_consistency: bool = True
    require_close_sibling: bool = True
    final_checkpoint_privilege: bool = False
    hashes_diagnostic_only: bool = True
    save_cadence: tuple[int, ...] = SAVE_CADENCE_DEFAULT


@dataclass(frozen=True)
class ArmBankEvalReceipt:
    arm: str
    earliest_all_clear_save: int | None
    acquire_pct: float | None
    retain_pct_by_support: dict[str, float]
    parent_consistency_ok: bool | None
    close_sibling_ok: bool | None
    hashes_diagnostic: dict[str, str]
    bank_clear: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "earliest_all_clear_save": self.earliest_all_clear_save,
            "acquire_pct": self.acquire_pct,
            "retain_pct_by_support": dict(self.retain_pct_by_support),
            "parent_consistency_ok": self.parent_consistency_ok,
            "close_sibling_ok": self.close_sibling_ok,
            "hashes_diagnostic": dict(self.hashes_diagnostic),
            "hashes_grant_or_veto": False,
            "bank_clear": bool(self.bank_clear),
        }


def select_earliest_all_clear(
    clears_by_save: Mapping[int, bool],
    *,
    cadence: Sequence[int] = SAVE_CADENCE_DEFAULT,
    final_privilege: bool = False,
) -> int | None:
    """Earliest cadence save that is all-clear. Final has no privilege."""

    ordered = [int(s) for s in cadence]
    if not final_privilege and ordered:
        # final may clear but does not outrank earlier; still eligible if sole clear
        pass
    for step in ordered:
        if bool(clears_by_save.get(step)):
            return int(step)
    return None


def evaluate_arm_bank_gate(
    *,
    arm: str,
    acquire_pct: float,
    retain_pct_by_support: Mapping[str, float],
    clears_by_save: Mapping[int, bool],
    parent_consistency_ok: bool,
    close_sibling_ok: bool,
    hashes_diagnostic: Mapping[str, str] | None = None,
    spec: BankGateSurfaceSpec | None = None,
) -> ArmBankEvalReceipt:
    spec = spec or BankGateSurfaceSpec()
    retain_ok = all(
        float(retain_pct_by_support.get(name, -1.0)) >= float(spec.retain_target_pct)
        for name in spec.retained_supports
    )
    acquire_ok = float(acquire_pct) >= float(spec.acquire_target_pct)
    pc_ok = bool(parent_consistency_ok) if spec.require_parent_consistency else True
    sib_ok = bool(close_sibling_ok) if spec.require_close_sibling else True
    earliest = select_earliest_all_clear(
        clears_by_save,
        cadence=spec.save_cadence,
        final_privilege=spec.final_checkpoint_privilege,
    )
    bank_clear = bool(
        acquire_ok and retain_ok and pc_ok and sib_ok and earliest is not None
    )
    return ArmBankEvalReceipt(
        arm=str(arm),
        earliest_all_clear_save=earliest,
        acquire_pct=float(acquire_pct),
        retain_pct_by_support={k: float(v) for k, v in retain_pct_by_support.items()},
        parent_consistency_ok=bool(parent_consistency_ok),
        close_sibling_ok=bool(close_sibling_ok),
        hashes_diagnostic=dict(hashes_diagnostic or {}),
        bank_clear=bank_clear,
    )


def e_must_match_u_bank(
    u: ArmBankEvalReceipt, e: ArmBankEvalReceipt
) -> bool:
    """CONTROL_INVALID if E bank outcome differs from U."""

    return bool(u.bank_clear) == bool(e.bank_clear) and (
        u.earliest_all_clear_save == e.earliest_all_clear_save
        or (not u.bank_clear and not e.bank_clear)
    )


def probe_bank_gate_import_surface() -> dict[str, str]:
    """Document the probe symbols this facade is intended to wrap at GPU time."""

    return {
        "probe": "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
        "acquire": "audit_identity_full_support",
        "retain": "finalize_b2_full_verdict_state / record_b2_full_prior_snapshot",
        "close_sibling": "audit_prior_support*",
        "note": "CPU Phase-B uses local evaluate_arm_bank_gate; probe stay thin",
    }


__all__ = [
    "BankGateSurfaceSpec",
    "ArmBankEvalReceipt",
    "select_earliest_all_clear",
    "evaluate_arm_bank_gate",
    "e_must_match_u_bank",
    "probe_bank_gate_import_surface",
    "B2_RETAINED_SUPPORTS",
]
