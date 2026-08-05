"""A′ slice-4 Rung-4 support-split residual densify schema (STEP-1).

Constants, labels, claim-boundary, and re-exports of residual pure helpers.
No branch rules, no IO/CLI. Densify-specific; residual arithmetic IMPORTED only.
PLAN v6: feea775c3b3bb1bee6f0d5775d4da783b09560b72b4a1b6cd8500af5f56329a9
"""
from __future__ import annotations

from typing import Any

# IMPORT_ONLY residual pure helpers (pinned Rung-3 modules; read-only).
from scripts.a_prime_slice4_residual_classification_schema_v0 import (  # noqa: F401
    ARMS,
    EXPECTED_CARDINALITY,
    HORIZONS,
    MIN_BUCKET_DENOMINATOR_ROWS,
    SUPPORTS,
    admit_horizon_view,
    ceil_0_70,
    coverage_buckets_ok,
    coverage_rows_ok,
    enrichment_ge_1_5,
    enrichment_le_0_5,
    extract_bucket_map,
    extract_survivors,
    extract_universe,
    is_exact_bool,
    is_exact_dict,
    is_exact_int,
    is_exact_list,
    is_exact_set,
    is_exact_str,
)

SCHEMA_ID = "a_prime_slice4_support_split_residual_densify/v0"
PRODUCT = "SUPPORT_SPLIT_RESIDUAL_DENSIFY_RECEIPTS_ONLY"

PREEMPTING_ONLY: tuple[str, ...] = (
    "INSTRUMENT_OR_BIND_FAIL",
    "IDENTITY_BIND_FAIL",
    "AUTHORITY_BIND_FAIL",
)

D1_PER_SUPPORT = (
    "HEAD1_MAJORITY",
    "HEAD2_MAJORITY",
    "HEAD2_THIRD",
    "HEAD_DIFFUSE",
    "DEGENERATE_NO_SURVIVORS",
)
D1_COMPOSITE = D1_PER_SUPPORT + ("SPLIT_SUPPORTS",)

D2_PER_SUPPORT = ("E_PERSISTENT", "E_TRANSIENT", "E_MIXED", "E_EMPTY")
D2_COMPOSITE = D2_PER_SUPPORT + ("SPLIT_SUPPORTS",)

D3_PER_SUPPORT = (
    "CO_MAJORITY",
    "CO_PARTIAL",
    "CO_DISJOINT",
    "DEGENERATE_NO_SURVIVORS",
)
D3_COMPOSITE = D3_PER_SUPPORT + ("SPLIT_SUPPORTS",)

# D1 TOTAL denominator pin (PLAN v6): S_s = |R50_s| including ineligible survivors.
D1_BRANCH_DENOMINATOR = "TOTAL_PACKAGE_N50_SUPPORT_SURVIVORS"
D1_L0B_CALIBRATION = {
    "S_total": 20,
    "S_eligible_sum": 16,
    "top2_sum": 8,
    "must_label": "HEAD2_THIRD",
    "forbidden_label_under_eligible_only_denom": "HEAD2_MAJORITY",
}

MEMBERSHIP_CARRIER = "row_id"

REQUIRED_CLAIM_BOUNDARY: dict[str, bool] = {
    "receipts_only_descriptive_densify": True,
    "no_cause": True,
    "no_individual_replay_vs_pc_attribution": True,
    "no_mechanism_mint": True,
    "no_carrier_claim": True,
    "no_readiness_acquisition_bank": True,
    "pre_carrier": True,
}

FROZEN_NEUTRAL_SUCCESSOR_TEXTS: dict[str, str] = {
    "step_5": (
        "D1 head-majority and R50∩O50 empty at N50; "
        "hold for design review; no mechanism mint"
    ),
    "step_6": (
        "E50 rows are mostly absent from PACKAGE N20 (row_id difference) "
        "and R50 is disjoint from OUT N50; hold for design review; no mechanism mint"
    ),
    "step_7": (
        "E50 rows meet the PACKAGE-N20 membership threshold and R50 meets "
        "the OUT-N50 co-membership threshold; next = shared-row residual audit; "
        "no mechanism mint"
    ),
    "default": "classify residual densify complete; no mechanism mint",
    "split": (
        "per-support densify report already emitted; next = design review of "
        "support-asymmetric residual densify (still no mechanism mint)"
    ),
    "instrument": "instrument repair only; no science successor",
    "identity": "receipt/field schema repair; no densify science",
    "authority": "Rung-3 terminal/recompute bind repair; no densify science",
}

# Authority residual surface (PACKAGE N50 exclusive) — matches Rung-3 Q3.
AUTHORITY_ARM = "package"
AUTHORITY_HORIZON = 50


def is_exact_mapping(v: Any) -> bool:
    return is_exact_dict(v)
