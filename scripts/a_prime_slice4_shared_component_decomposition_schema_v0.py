"""A′ slice-4 Rung-5 shared-component decomposition schema (STEP-1).

Constants + re-exports of densify/residual pure helpers. No branch rules, no IO/CLI.
PLAN v4: a2e7420aeaee715ed181b46f4f1de4d0b93deb47a29da6e3bded0fd431e48421
"""
from __future__ import annotations

from typing import Any

# IMPORT_ONLY densify schema (itself imports residual pure helpers).
from scripts.a_prime_slice4_support_split_residual_densify_schema_v0 import (  # noqa: F401
    ARMS,
    AUTHORITY_ARM,
    AUTHORITY_HORIZON,
    EXPECTED_CARDINALITY,
    HORIZONS,
    MEMBERSHIP_CARRIER,
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

SCHEMA_ID = "a_prime_slice4_shared_component_decomposition/v0"
PRODUCT = "SHARED_ENRICHED_COMPONENT_DECOMPOSITION_RECEIPTS_ONLY"

SHARED_ENRICHED_COMPONENTS: tuple[str, ...] = ("R0", "R1b4v2")
MASS_HEAD_COMPARATOR_BUCKET = "R1"
MASS_HEAD_COMPARATOR_SUPPORT = "math_a0"

PREEMPTING: tuple[str, ...] = (
    "INSTRUMENT_OR_BIND_FAIL",
    "IDENTITY_BIND_FAIL",
    "AUTHORITY_BIND_FAIL",
    "RECOMPOSITION_BIND_FAIL",
    "DEGENERATE_EMPTY_COMPONENT",
)

D2_COMPONENT_LABELS: tuple[str, ...] = (
    "E_PERSISTENT",
    "E_TRANSIENT",
    "E_MIXED",
    "E_EMPTY",
)

PRIMARY_SCIENCE: tuple[str, ...] = (
    "ALIGNED_COMPONENT_LABELS__AGGREGATE_SPLIT",
    "COMPONENT_LABEL_SPLIT",
)

A1_SECONDARY: tuple[str, ...] = (
    "R1_LABEL_EQ_R0_AND_R1b4v2",
    "R1_LABEL_EQ_R0",
    "R1_LABEL_EQ_R1b4v2",
    "R1_LABEL_EQ_NEITHER",
)

REQUIRED_CLAIM_BOUNDARY: dict[str, bool] = {
    "receipts_only_descriptive_component_decomposition": True,
    "pre_carrier": True,
    "pre_mechanism_mint": True,
    "no_cause": True,
    "no_individual_replay_vs_pc_attribution": True,
    "no_mechanism_mint": True,
    "no_carrier_claim": True,
    "no_readiness_acquisition_bank": True,
}

FROZEN_NEUTRAL_SUCCESSOR_TEXTS: dict[str, str] = {
    "instrument": "instrument repair only; no science successor",
    "identity": "receipt/field schema repair; no component science",
    "authority": "Rung-3/Rung-4 terminal or D2 recompute bind repair; no component science",
    "recomposition": "recomposition arithmetic/instrument repair; no component science",
    "step_5": "empty named component; hold for design review; no mechanism mint",
    "step_6": (
        "shared enriched component labels align across supports; aggregate endpoint "
        "labels differ under exact per-support count recomposition; design review "
        "complete; no mechanism mint"
    ),
    "step_7": (
        "shared enriched component labels differ across supports; hold for design "
        "review of component-label split; no mechanism mint"
    ),
    "step_8": "component-decomposition terminal is the measurement; no mechanism mint",
}

RUNG3_COMPOSITE_EXPECTED = "IDENTITY_OK__CHURNED__TRANSIENT__STRATIFIED"
RUNG4_COMPOSITE_EXPECTED = "IDENTITY_OK__HEAD2_THIRD__SPLIT_SUPPORTS__CO_PARTIAL"
RUNG4_D2_COMPOSITE_EXPECTED = "SPLIT_SUPPORTS"
RUNG4_D2_PER_SUPPORT_EXPECTED: dict[str, str] = {
    "L0b": "E_MIXED",
    "math_a0": "E_TRANSIENT",
}
RUNG4_D2_RAW_COUNTS_EXPECTED: dict[str, dict[str, int]] = {
    "L0b": {
        "|E50|_row_ids": 8,
        "present_at_package_N20_row_id_intersection": 3,
        "absent_from_package_N20_row_id_difference": 5,
    },
    "math_a0": {
        "|E50|_row_ids": 23,
        "present_at_package_N20_row_id_intersection": 6,
        "absent_from_package_N20_row_id_difference": 17,
    },
}


def is_exact_mapping(v: Any) -> bool:
    return is_exact_dict(v)
