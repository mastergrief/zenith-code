"""Thin bank-gate eval facade (Option B — no PC/sibling boolean AND-gates).

Hashes are DIAGNOSTIC_ONLY. Global retain uses existing numeric >=90 path
(temporary conservative behavior; NOT declared canonical).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

B2_RETAINED_SUPPORTS = ("L0b", "math_a0")
BANK_ACQUIRE_TARGET = 90
BANK_RETAIN_TARGET = 90
SAVE_CADENCE_DEFAULT = (250, 500, 750, 1000, 1250, 1500)
CLOSE_SIBLING_BROAD_CLUSTER_THRESHOLD = 3


class ParentFloorStatus(str, Enum):
    """Tri-state — absence must not default benignly."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNRESOLVED_POLICY = "UNRESOLVED_POLICY"


@dataclass(frozen=True)
class BankGateSurfaceSpec:
    acquire_target_pct: int = BANK_ACQUIRE_TARGET
    retain_target_pct: int = BANK_RETAIN_TARGET
    retained_supports: tuple[str, ...] = B2_RETAINED_SUPPORTS
    final_checkpoint_privilege: bool = False
    hashes_diagnostic_only: bool = True
    save_cadence: tuple[int, ...] = SAVE_CADENCE_DEFAULT
    broad_cluster_threshold: int = CLOSE_SIBLING_BROAD_CLUSTER_THRESHOLD


@dataclass(frozen=True)
class CloseSiblingReport:
    """L0c1 evidence/report — never an independent veto boolean."""

    numeric_close_sibling_clear: bool
    same_surface_parent_relative_hole_count: int
    parent_floor_status: ParentFloorStatus
    surfaces: tuple[str, ...] = ()
    candidate_scores: Mapping[str, float] = field(default_factory=dict)
    parent_scores: Mapping[str, float] = field(default_factory=dict)
    deltas: Mapping[str, float] = field(default_factory=dict)
    evaluator_identity: str = ""
    notes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def no_new_broad_cluster(self) -> bool:
        return int(self.same_surface_parent_relative_hole_count) < int(
            CLOSE_SIBLING_BROAD_CLUSTER_THRESHOLD
        )

    @property
    def sibling_local_clear(self) -> bool:
        if bool(self.numeric_close_sibling_clear):
            return True
        if self.parent_floor_status is not ParentFloorStatus.PASS:
            return False
        return bool(self.no_new_broad_cluster)

    def as_dict(self) -> dict[str, Any]:
        return {
            "numeric_close_sibling_clear": bool(self.numeric_close_sibling_clear),
            "same_surface_parent_relative_hole_count": int(
                self.same_surface_parent_relative_hole_count
            ),
            "no_new_broad_cluster": bool(self.no_new_broad_cluster),
            "parent_floor_status": self.parent_floor_status.value,
            "sibling_local_clear": bool(self.sibling_local_clear),
            "surfaces": list(self.surfaces),
            "candidate_scores": dict(self.candidate_scores),
            "parent_scores": dict(self.parent_scores),
            "deltas": dict(self.deltas),
            "evaluator_identity": str(self.evaluator_identity),
            "notes": dict(self.notes),
        }


def reduce_close_sibling_blockers(
    report: CloseSiblingReport | None,
    *,
    threshold: int = CLOSE_SIBLING_BROAD_CLUSTER_THRESHOLD,
) -> list[str]:
    if report is None:
        return []
    out: list[str] = []
    if int(report.same_surface_parent_relative_hole_count) >= int(threshold):
        out.append("CLOSE_SIBLING_BROAD_CLUSTER")
    if report.parent_floor_status is ParentFloorStatus.FAIL:
        out.append("CLOSE_SIBLING_PARENT_FLOOR")
    return out


def claim_blockers_from_close_sibling(report: CloseSiblingReport | None) -> list[str]:
    if report is None or report.parent_floor_status is ParentFloorStatus.UNRESOLVED_POLICY:
        return ["PARENT_FLOOR_POLICY_UNRESOLVED"]
    return []


@dataclass(frozen=True)
class ArmBankEvalReceipt:
    arm: str
    earliest_all_clear_save: int | None
    acquire_pct: float | None
    retain_pct_by_support: dict[str, float]
    close_sibling_report: dict[str, Any] | None
    bank_blockers: tuple[str, ...]
    hashes_diagnostic: dict[str, str]
    bank_clear: bool
    schema_version: int = 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "schema_version": int(self.schema_version),
            "earliest_all_clear_save": self.earliest_all_clear_save,
            "acquire_pct": self.acquire_pct,
            "retain_pct_by_support": dict(self.retain_pct_by_support),
            "close_sibling_report": (
                None if self.close_sibling_report is None else dict(self.close_sibling_report)
            ),
            "bank_blockers": list(self.bank_blockers),
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
    for step in [int(s) for s in cadence]:
        if bool(clears_by_save.get(step)):
            return int(step)
    return None


def evaluate_arm_bank_gate(
    *,
    arm: str,
    acquire_pct: float,
    retain_pct_by_support: Mapping[str, float],
    clears_by_save: Mapping[int, bool],
    close_sibling_report: CloseSiblingReport | None = None,
    hashes_diagnostic: Mapping[str, str] | None = None,
    spec: BankGateSurfaceSpec | None = None,
) -> ArmBankEvalReceipt:
    spec = spec or BankGateSurfaceSpec()
    retain_ok = all(
        float(retain_pct_by_support.get(name, -1.0)) >= float(spec.retain_target_pct)
        for name in spec.retained_supports
    )
    acquire_ok = float(acquire_pct) >= float(spec.acquire_target_pct)
    earliest = select_earliest_all_clear(
        clears_by_save, cadence=spec.save_cadence,
        final_privilege=spec.final_checkpoint_privilege,
    )
    blockers = tuple(
        reduce_close_sibling_blockers(
            close_sibling_report, threshold=int(spec.broad_cluster_threshold)
        )
    )
    bank_clear = bool(acquire_ok and retain_ok and earliest is not None and not blockers)
    return ArmBankEvalReceipt(
        arm=str(arm),
        earliest_all_clear_save=earliest,
        acquire_pct=float(acquire_pct),
        retain_pct_by_support={k: float(v) for k, v in retain_pct_by_support.items()},
        close_sibling_report=(
            None if close_sibling_report is None else close_sibling_report.as_dict()
        ),
        bank_blockers=blockers,
        hashes_diagnostic=dict(hashes_diagnostic or {}),
        bank_clear=bank_clear,
    )


def e_must_match_u_bank(u: ArmBankEvalReceipt, e: ArmBankEvalReceipt) -> bool:
    if bool(u.bank_clear) != bool(e.bank_clear):
        return False
    if u.earliest_all_clear_save != e.earliest_all_clear_save and (
        u.bank_clear or e.bank_clear
    ):
        return False
    return tuple(u.bank_blockers) == tuple(e.bank_blockers)


def probe_bank_gate_import_surface() -> dict[str, str]:
    return {
        "probe": "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
        "acquire": "audit_identity_full_support",
        "retain": "finalize_b2_full_verdict_state / record_b2_full_prior_snapshot",
        "close_sibling": "audit_prior_support*",
        "note": "CPU Phase-B uses local evaluate_arm_bank_gate; probe stay thin",
    }


__all__ = [
    "BankGateSurfaceSpec", "ParentFloorStatus", "CloseSiblingReport",
    "ArmBankEvalReceipt", "select_earliest_all_clear",
    "reduce_close_sibling_blockers", "claim_blockers_from_close_sibling",
    "evaluate_arm_bank_gate", "e_must_match_u_bank",
    "probe_bank_gate_import_surface", "B2_RETAINED_SUPPORTS",
    "CLOSE_SIBLING_BROAD_CLUSTER_THRESHOLD",
]
