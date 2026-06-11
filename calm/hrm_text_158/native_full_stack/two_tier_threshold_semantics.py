"""Frozen §2C threshold semantics and §2E audit-resolution helpers."""
from __future__ import annotations

from typing import Any, Mapping

# Frozen §2C from transient_selection_interface.md (R2 1781122431786).
CROSSING_THRESHOLD_ABS = 10
FROZEN_THRESHOLD_SEMANTICS: dict[str, Any] = {
    "crossing_threshold_abs": 10,
    "crossing_threshold_source": "canonical_default_spec_accumulator_real_dynamics_verdict",
    "crossing_authority": "vote_update_spec",
    "residual_band_encoding": "threshold_minus_one",
    "row_fields_authority": "telemetry_not_crossing",
    "row_crosscheck_policy": "informational",
}

THRESHOLD_CROSSCHECK_MISMATCH = "threshold_row_derivation_mismatch"
THRESHOLD_CROSSCHECK_INFORMATIONAL_POLICY = "informational"


def frozen_threshold_semantics_block() -> dict[str, Any]:
    """Return the verbatim §2C threshold_semantics block."""

    return dict(FROZEN_THRESHOLD_SEMANTICS)


def assert_two_tier_threshold_receipt_consistent(stats: Mapping[str, Any]) -> None:
    """Fail-closed: two-tier receipts must not split effective vs canonical threshold."""

    if not bool(stats.get("two_tier_carry_w6_enabled")):
        return
    effective = int(stats["two_tier_threshold_abs"])
    canonical = int(stats["two_tier_canonical_threshold_abs"])
    if effective != canonical:
        raise ValueError(
            "two_tier_threshold_receipt_inconsistent: "
            f"two_tier_threshold_abs={effective} "
            f"two_tier_canonical_threshold_abs={canonical}"
        )


def resolve_threshold_crosscheck_authority(
    crosscheck_status: str,
    *,
    threshold_semantics: Mapping[str, Any] | None = None,
) -> str:
    """§2E: threshold_crosscheck is informational when frozen §2C is present."""

    semantics = (
        dict(threshold_semantics)
        if threshold_semantics is not None
        else frozen_threshold_semantics_block()
    )
    if (
        str(crosscheck_status) == THRESHOLD_CROSSCHECK_MISMATCH
        and semantics.get("row_crosscheck_policy")
        == THRESHOLD_CROSSCHECK_INFORMATIONAL_POLICY
    ):
        return THRESHOLD_CROSSCHECK_INFORMATIONAL_POLICY
    return str(crosscheck_status)
