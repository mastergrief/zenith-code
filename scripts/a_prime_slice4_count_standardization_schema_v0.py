"""Schema constants for A′ slice-4 Rung-6 count-standardization (PLAN v6).

Pure constants only — no I/O, no CLI.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Final

SCHEMA_ID: Final[str] = "a_prime_slice4_count_standardization/v0"
PRODUCT: Final[str] = "COUNT_STANDARDIZATION_DECOMPOSITION"
PLAN_REVISION_BINDING: Final[str] = (
    "PLAN_v6 ee9628cdcc45515dd8007de065960cae344b43f5ccaa600b3d8bafaa3066b900"
)
TASK_ID: Final[str] = "1786004998450-f6569bd2"

SUPPORTS: Final[tuple[str, ...]] = ("L0b", "math_a0")
COMPONENTS: Final[tuple[str, ...]] = ("R0", "R1b4v2")
COMPONENTS_SET: Final[frozenset[str]] = frozenset(COMPONENTS)

THRESHOLD_TRANSIENT: Final[Fraction] = Fraction(3, 10)
THRESHOLD_PERSISTENT: Final[Fraction] = Fraction(7, 10)

BRANCH_STANDARDIZATION_BIND_FAIL: Final[str] = "STANDARDIZATION_BIND_FAIL"
BRANCH_BOUNDARY_TIE: Final[str] = "BOUNDARY_TIE"
BRANCH_RATE_PROFILE_SELECTS: Final[str] = (
    "RATE_PROFILE_SELECTS_D2_LABEL_UNDER_BOTH_WEIGHT_PROFILES"
)
BRANCH_WEIGHT_PROFILE_SELECTS: Final[str] = (
    "WEIGHT_PROFILE_SELECTS_D2_LABEL_UNDER_BOTH_RATE_PROFILES"
)
BRANCH_BOTH_AXES_OR_INTERACTION: Final[str] = "BOTH_AXES_OR_INTERACTION"
BRANCH_NEITHER_AXIS_SELECTS: Final[str] = "NEITHER_AXIS_SELECTS_LABEL"

FIRST_MATCH_ORDER: Final[tuple[str, ...]] = (
    BRANCH_STANDARDIZATION_BIND_FAIL,
    BRANCH_BOUNDARY_TIE,
    BRANCH_RATE_PROFILE_SELECTS,
    BRANCH_WEIGHT_PROFILE_SELECTS,
    BRANCH_BOTH_AXES_OR_INTERACTION,
    BRANCH_NEITHER_AXIS_SELECTS,
)

LABEL_PERSISTENT: Final[str] = "PERSISTENT"
LABEL_TRANSIENT: Final[str] = "TRANSIENT"
LABEL_MIXED: Final[str] = "MIXED"

# Frozen Rung-5 published counts (diagnostic pins; live path re-extracts from receipt).
FROZEN_COUNTS: Final[dict] = {
    "L0b": {
        "R0": {"N50": 5, "present_N20": 1, "absent_N20": 4},
        "R1b4v2": {"N50": 3, "present_N20": 2, "absent_N20": 1},
        "aggregate": {
            "N50": 8,
            "present_N20": 3,
            "absent_N20": 5,
            "ceil_0_70": 6,
            "margin": -1,
        },
    },
    "math_a0": {
        "R0": {"N50": 14, "present_N20": 1, "absent_N20": 13},
        "R1b4v2": {"N50": 9, "present_N20": 5, "absent_N20": 4},
        "aggregate": {
            "N50": 23,
            "present_N20": 6,
            "absent_N20": 17,
            "ceil_0_70": 17,
            "margin": 0,
        },
    },
}

PUBLISHED_D2_LABELS_RAW: Final[dict[str, str]] = {
    "L0b": "E_MIXED",
    "math_a0": "E_TRANSIENT",
}

CLAIM_BOUNDARY_KEY_SET: Final[tuple[str, ...]] = (
    "receipts_only_descriptive_count_standardization",
    "pre_carrier",
    "pre_mechanism_mint",
    "no_cause_claim",
    "no_mechanism_claim",
    "no_labeler_cause_claim",
    "no_individual_fate_or_transfer_claim",
    "no_carrier_readiness_acquisition_bank_claim",
)

CLAIM_BOUNDARY_REQUIRED: Final[dict[str, bool]] = {
    k: True for k in CLAIM_BOUNDARY_KEY_SET
}

CLAIM_CEILING_SENTENCE_A: Final[str] = (
    "under the existing D2 threshold, swapping the named component rate profile "
    "versus weight profile changes/preserves the standardized categorical endpoint "
    "as preregistered."
)
CLAIM_CEILING_SENTENCE_B: Final[str] = (
    "the decomposition attributes the standardized rate gap arithmetically; each "
    "standardized categorical endpoint is the existing 3/10–7/10 boundary "
    "classification of its level, and their observed difference is reported "
    "separately — no claim about labeler cause, individual fate, transfer, or "
    "mechanism."
)

SUCCESSOR_MAPPING: Final[dict[str, str]] = {
    BRANCH_RATE_PROFILE_SELECTS: (
        "rate profile selects standardized D2 label under both weight profiles; "
        "next design boundary is whether to close slice-4 (C) or open "
        "threshold-sensitivity (A) only if residual design asks it — still no "
        "mechanism mint"
    ),
    BRANCH_WEIGHT_PROFILE_SELECTS: (
        "weight profile selects standardized D2 label under both rate profiles; "
        "next design boundary is C or mixture-focused follow-up — still no "
        "mechanism mint"
    ),
    BRANCH_BOTH_AXES_OR_INTERACTION: (
        "interaction/unresolved under this instrument (background-dependent flips) "
        "— C with unresolved result or A if design reopens threshold; still no "
        "mechanism mint"
    ),
    BRANCH_NEITHER_AXIS_SELECTS: (
        "neither axis selects (all four labels equal); residual asymmetry not "
        "categorical under this boundary — C with that characterization; still no "
        "mechanism mint"
    ),
    BRANCH_STANDARDIZATION_BIND_FAIL: (
        "instrument/bind failure — cure instrument; do not mint science"
    ),
    BRANCH_BOUNDARY_TIE: (
        "exact boundary cell — fallback A (threshold-sensitivity) candidacy; no "
        "mechanism mint"
    ),
}


APPROVED_PHRASING_EXAMPLES: Final[tuple[str, ...]] = (
    "rate profile selects the standardized D2 label under both weight profiles",
    "share_term and rate_term partition delta_q exactly",
    "diagonal standardized labels reproduce published D2 labels",
)

# Exact allowlist for claim-field silence (ws-normalized membership).
_SCIENCE_BRANCHES: Final[tuple[str, ...]] = (
    BRANCH_RATE_PROFILE_SELECTS,
    BRANCH_WEIGHT_PROFILE_SELECTS,
    BRANCH_BOTH_AXES_OR_INTERACTION,
    BRANCH_NEITHER_AXIS_SELECTS,
)
# Preregistered prohibition calibrations (PLAN v4:260 PASS side via membership).
PROHIBITION_CALIBRATION_STRINGS: Final[tuple[str, ...]] = (
    "no claim about rates causing the label split",
    "without claiming that weights cause the split",
    "it is not a claim that the rate profile causes the split",
    "no claim about labeler cause or mechanism is made here",
    "no claim about whether rates cause or weights cause the split",
)
CLAIM_SILENT_ALLOWLIST: Final[tuple[str, ...]] = (
    CLAIM_CEILING_SENTENCE_A,
    CLAIM_CEILING_SENTENCE_B,
    *APPROVED_PHRASING_EXAMPLES,
    *tuple(SUCCESSOR_MAPPING.values()),
    *FIRST_MATCH_ORDER,
    *tuple(f"IDENTITY_OK__{b}" for b in _SCIENCE_BRANCHES),
    *PROHIBITION_CALIBRATION_STRINGS,
)

# Exact Kitagawa values on live-shaped counts (Fraction strings for pin asserts).
KITAGAWA_LIVE_EXACT: Final[dict] = {
    "delta_q": Fraction(-21, 184),
    "L0b": {"share": Fraction(7, 920), "rate": Fraction(-14, 115)},
    "math_a0": {"share": Fraction(61, 7728), "rate": Fraction(-41, 336)},
    "symmetric_average": {
        "share": Fraction(599, 77280),
        "rate": Fraction(-9419, 77280),
    },
}

RUNG5_TERMINAL_PIN: Final[dict[str, object]] = {
    "path": (
        "/home/gabe/claw-code-creditdir/a_prime_slice4_rung5/"
        "run_component_v1/terminal_receipt.json"
    ),
    "sha256": "9b9939b52c6fa984582c93604d8033385bb6ecd154399d782da49ba013096a6c",
    "bytes": 35775,
}
