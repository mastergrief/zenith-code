"""B2-5b Step-0 optimizer_credit_state consumer measurement receipt.

CPU/read-only diagnostic: measures whether banked B2-5a″ native MARGIN selection
is sufficient for the real optimizer_credit_state consumer and classifies the
integration seam.  Does NOT wire the native selector or flip readiness rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

OPTIMIZER_CREDIT_STATE_GLOBAL_CAP_CONSUMER_STEP0_SCHEMA_VERSION = (
    "hrm_text_158_optimizer_credit_state_global_cap_consumer_step0/v0.b2_5b"
)

WIDER_SINGLE_BLOCK_CEILING = 2048

OPTIMIZER_CREDIT_STATE_GLOBAL_CAP_CONSUMER_STEP0_NON_CLAIMS: tuple[str, ...] = (
    "B2-5b Step-0 is CPU/read-only consumer measurement only; no native selector wiring",
    "B2-5b Step-0 does NOT mint selection_parity_pass",
    "B2-5b Step-0 does NOT flip global_cap_margin_only_reference",
    "B2-5b Step-0 does NOT flip optimizer_credit_state / readiness rows",
    "B2-5b Step-0 does NOT claim full-loop or GPU runtime proof",
    "B2-5b aggregate_branch_id reflects representative consumer rows only",
    "classifier_negative fixtures are test-only and never drive the default aggregate",
)

OPTIMIZER_CREDIT_STATE_GLOBAL_CAP_CONSUMER_STEP0_HARD_FALSE_FIELDS: tuple[str, ...] = (
    "selection_parity_pass",
    "native_selector_wired",
    "readiness_flip_authorized",
    "global_cap_margin_only_reference_flipped",
    "optimizer_credit_state_sub2_claim",
    "optimizer_credit_state_resolved",
)

PINNED_CALL_SITE_IOCP_DEFAULT_OFF_FLAG = "IOCP_DEFAULT_OFF_FLAG"
PINNED_CALL_SITE_IOCP_SPARSE_EMIT_STEP = "IOCP_SPARSE_EMIT_STEP"
PINNED_CALL_SITE_IOCP_RECEIPT_FLAGS = "IOCP_RECEIPT_FLAGS"
PINNED_CALL_SITE_BDL_GLOBAL_CAP_REFERENCE = "BDL_GLOBAL_CAP_REFERENCE"
PINNED_CALL_SITE_BDL_CANDIDATE_GLOBAL_CAP_REJECT = "BDL_CANDIDATE_GLOBAL_CAP_REJECT"
PINNED_CALL_SITE_GRC_STEP_SUMMARY_SURFACE = "GRC_STEP_SUMMARY_SURFACE"
PINNED_CALL_SITE_GRC_SPEC_DEFAULT_MARGIN = "GRC_SPEC_DEFAULT_MARGIN"

PINNED_CALL_SITE_IDS: tuple[str, ...] = (
    PINNED_CALL_SITE_IOCP_DEFAULT_OFF_FLAG,
    PINNED_CALL_SITE_IOCP_SPARSE_EMIT_STEP,
    PINNED_CALL_SITE_IOCP_RECEIPT_FLAGS,
    PINNED_CALL_SITE_BDL_GLOBAL_CAP_REFERENCE,
    PINNED_CALL_SITE_BDL_CANDIDATE_GLOBAL_CAP_REJECT,
    PINNED_CALL_SITE_GRC_STEP_SUMMARY_SURFACE,
    PINNED_CALL_SITE_GRC_SPEC_DEFAULT_MARGIN,
)

FixtureRole = Literal["representative_consumer", "classifier_negative"]

ConsumerPathClass = Literal[
    "PATH_A_INTEGER_WIRE",
    "PATH_B_GLOBAL_CAP_REFERENCE",
    "STRUCTURAL_SEAM_FACT",
    "GRC_SPEC_SURFACE",
    "CLASSIFIER_NEGATIVE_PROBE",
]

CandidateModeClass = Literal[
    "CANDIDATE_ONLY",
    "NON_CANDIDATE_GLOBAL_CAP_REFERENCE",
    "STRUCTURAL_REJECTION",
    "NOT_APPLICABLE",
]


class ConsumerStep0BranchId(str, Enum):
    MEASUREMENT_INVALID = "BR-OCGS-MEASUREMENT-INVALID"
    CEILING_LIFT_FIRST = "BR-OCGS-CEILING-LIFT-FIRST"
    ORDERING_MODE_FIRST = "BR-OCGS-ORDERING-MODE-FIRST"
    CANDIDATE_GCAP_SEAM = "BR-OCGS-CANDIDATE-GCAP-SEAM"
    INTEGRATION_PLAN = "BR-OCGS-INTEGRATION-PLAN"


@dataclass(frozen=True)
class ConsumerStep0FixtureMeasurement:
    fixture_name: str
    fixture_role: FixtureRole
    pinned_call_site_id: str
    source_anchor: str
    consumer_path_class: ConsumerPathClass
    candidate_mode_class: CandidateModeClass
    total_sparse_event_count: int
    projected_full_demand_count: int
    projected_global_pre_cap_would_apply_count: int
    max_row_count: int
    ordering_mode: str
    ordering_mode_source: str
    cap: int
    deferred_count: int
    saturation_observed: bool
    candidate_rejects_global_cap: bool
    seam_resolved: bool


@dataclass(frozen=True)
class ConsumerStep0BranchProbeResult:
    fixture_name: str
    fixture_role: FixtureRole
    branch_id: ConsumerStep0BranchId


@dataclass(frozen=True)
class OptimizerCreditStateGlobalCapConsumerStep0Receipt:
    schema_version: str = OPTIMIZER_CREDIT_STATE_GLOBAL_CAP_CONSUMER_STEP0_SCHEMA_VERSION
    fixture_measurements: tuple[ConsumerStep0FixtureMeasurement, ...] = ()
    representative_measurements: tuple[ConsumerStep0FixtureMeasurement, ...] = ()
    classifier_negative_results: tuple[ConsumerStep0BranchProbeResult, ...] = ()
    sampled_call_sites: tuple[str, ...] = ()
    aggregate_branch_id: ConsumerStep0BranchId = ConsumerStep0BranchId.MEASUREMENT_INVALID
    measurement_representative: bool = False
    any_row_count_above_ceiling: bool = False
    any_non_margin_ordering: bool = False
    any_candidate_rejects_global_cap: bool = False
    wider_ceiling: int = WIDER_SINGLE_BLOCK_CEILING
    include_classifier_negatives: bool = False
    selection_parity_pass: bool = False
    native_selector_wired: bool = False
    readiness_flip_authorized: bool = False
    global_cap_margin_only_reference_flipped: bool = False
    optimizer_credit_state_sub2_claim: bool = False
    optimizer_credit_state_resolved: bool = False
    non_claims: tuple[str, ...] = OPTIMIZER_CREDIT_STATE_GLOBAL_CAP_CONSUMER_STEP0_NON_CLAIMS


def _representative_rows(
    measurements: tuple[ConsumerStep0FixtureMeasurement, ...],
) -> tuple[ConsumerStep0FixtureMeasurement, ...]:
    return tuple(
        row for row in measurements if row.fixture_role == "representative_consumer"
    )


def all_pinned_sites_sampled(representative_rows: tuple[ConsumerStep0FixtureMeasurement, ...]) -> bool:
    sampled = {row.pinned_call_site_id for row in representative_rows}
    return all(site_id in sampled for site_id in PINNED_CALL_SITE_IDS)


def classify_fixture_branch_probe(
    row: ConsumerStep0FixtureMeasurement,
) -> ConsumerStep0BranchId:
    """Classify a single fixture row without the representativeness gate (test-only probes)."""

    if row.max_row_count > WIDER_SINGLE_BLOCK_CEILING:
        return ConsumerStep0BranchId.CEILING_LIFT_FIRST
    if row.ordering_mode != "margin":
        return ConsumerStep0BranchId.ORDERING_MODE_FIRST
    if row.candidate_rejects_global_cap:
        return ConsumerStep0BranchId.CANDIDATE_GCAP_SEAM
    if (
        row.max_row_count <= WIDER_SINGLE_BLOCK_CEILING
        and row.ordering_mode == "margin"
        and row.seam_resolved
    ):
        return ConsumerStep0BranchId.INTEGRATION_PLAN
    return ConsumerStep0BranchId.MEASUREMENT_INVALID


def classify_aggregate_branch(
    representative_rows: tuple[ConsumerStep0FixtureMeasurement, ...],
) -> ConsumerStep0BranchId:
    if not all_pinned_sites_sampled(representative_rows):
        return ConsumerStep0BranchId.MEASUREMENT_INVALID
    if any(row.max_row_count > WIDER_SINGLE_BLOCK_CEILING for row in representative_rows):
        return ConsumerStep0BranchId.CEILING_LIFT_FIRST
    if any(row.ordering_mode != "margin" for row in representative_rows):
        return ConsumerStep0BranchId.ORDERING_MODE_FIRST
    if any(row.candidate_rejects_global_cap for row in representative_rows):
        return ConsumerStep0BranchId.CANDIDATE_GCAP_SEAM
    if all(
        row.max_row_count <= WIDER_SINGLE_BLOCK_CEILING
        and row.ordering_mode == "margin"
        and row.seam_resolved
        for row in representative_rows
    ):
        return ConsumerStep0BranchId.INTEGRATION_PLAN
    return ConsumerStep0BranchId.MEASUREMENT_INVALID


def derive_aggregate_flags(
    representative_rows: tuple[ConsumerStep0FixtureMeasurement, ...],
) -> dict[str, bool | tuple[str, ...]]:
    return {
        "any_row_count_above_ceiling": any(
            row.max_row_count > WIDER_SINGLE_BLOCK_CEILING for row in representative_rows
        ),
        "any_non_margin_ordering": any(
            row.ordering_mode != "margin" for row in representative_rows
        ),
        "any_candidate_rejects_global_cap": any(
            row.candidate_rejects_global_cap for row in representative_rows
        ),
        "sampled_call_sites": tuple(
            sorted({row.pinned_call_site_id for row in representative_rows})
        ),
        "measurement_representative": all_pinned_sites_sampled(representative_rows),
    }


def build_optimizer_credit_state_global_cap_consumer_step0_receipt(
    *,
    fixture_measurements: tuple[ConsumerStep0FixtureMeasurement, ...],
    include_classifier_negatives: bool = False,
) -> OptimizerCreditStateGlobalCapConsumerStep0Receipt:
    representative = _representative_rows(fixture_measurements)
    if not include_classifier_negatives:
        unexpected = [
            row.fixture_name
            for row in fixture_measurements
            if row.fixture_role == "classifier_negative"
        ]
        if unexpected:
            raise ValueError(
                "classifier_negative fixtures require include_classifier_negatives=True"
            )
    flags = derive_aggregate_flags(representative)
    negative_rows = tuple(
        row for row in fixture_measurements if row.fixture_role == "classifier_negative"
    )
    classifier_negative_results = tuple(
        ConsumerStep0BranchProbeResult(
            fixture_name=row.fixture_name,
            fixture_role=row.fixture_role,
            branch_id=classify_fixture_branch_probe(row),
        )
        for row in negative_rows
    )
    return OptimizerCreditStateGlobalCapConsumerStep0Receipt(
        fixture_measurements=fixture_measurements,
        representative_measurements=representative,
        classifier_negative_results=classifier_negative_results,
        aggregate_branch_id=classify_aggregate_branch(representative),
        include_classifier_negatives=include_classifier_negatives,
        **flags,
    )


def validate_optimizer_credit_state_global_cap_consumer_step0_receipt(
    receipt: OptimizerCreditStateGlobalCapConsumerStep0Receipt,
) -> None:
    if receipt.schema_version != OPTIMIZER_CREDIT_STATE_GLOBAL_CAP_CONSUMER_STEP0_SCHEMA_VERSION:
        raise ValueError("schema_version mismatch")
    if receipt.wider_ceiling != WIDER_SINGLE_BLOCK_CEILING:
        raise ValueError("wider_ceiling must be 2048")
    for field in OPTIMIZER_CREDIT_STATE_GLOBAL_CAP_CONSUMER_STEP0_HARD_FALSE_FIELDS:
        if bool(getattr(receipt, field)):
            raise ValueError(f"{field} must remain hard-false on Step-0 receipt")
    representative = _representative_rows(receipt.fixture_measurements)
    if receipt.representative_measurements != representative:
        raise ValueError("representative_measurements must match fixture_role filter")
    flags = derive_aggregate_flags(representative)
    for key in (
        "any_row_count_above_ceiling",
        "any_non_margin_ordering",
        "any_candidate_rejects_global_cap",
        "sampled_call_sites",
        "measurement_representative",
    ):
        if getattr(receipt, key) != flags[key]:
            raise ValueError(f"{key} inconsistent with representative measurements")
    expected_branch = classify_aggregate_branch(representative)
    if receipt.aggregate_branch_id != expected_branch:
        raise ValueError("aggregate_branch_id inconsistent with representative measurements")
    if not receipt.include_classifier_negatives:
        if any(
            row.fixture_role == "classifier_negative" for row in receipt.fixture_measurements
        ):
            raise ValueError(
                "classifier_negative fixtures forbidden when include_classifier_negatives=False"
            )
        if receipt.classifier_negative_results:
            raise ValueError("classifier_negative_results must be empty by default")
    for row in receipt.fixture_measurements:
        if row.pinned_call_site_id not in PINNED_CALL_SITE_IDS and row.fixture_role == "representative_consumer":
            raise ValueError(f"unknown pinned call site {row.pinned_call_site_id!r}")
        if row.projected_full_demand_count < 0:
            raise ValueError("projected_full_demand_count must be >= 0")
        if row.max_row_count < 0:
            raise ValueError("max_row_count must be >= 0")
        if row.fixture_role == "representative_consumer" and row.ordering_mode not in {
            "margin",
            "hash_shuffle",
            "round_robin",
        }:
            raise ValueError(f"invalid ordering_mode {row.ordering_mode!r}")
