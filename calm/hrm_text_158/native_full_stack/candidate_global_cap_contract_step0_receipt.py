"""B2-5c Step-0 candidate↔global-cap contract characterization receipt.

CPU/read-only diagnostic classifying whether candidate_mode's rejection of
global_cap_spec is proof-extension, reconciliation-contract, or bridge-implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

CANDIDATE_GLOBAL_CAP_CONTRACT_STEP0_SCHEMA_VERSION = (
    "hrm_text_158_candidate_global_cap_contract_step0/v0.b2_5c"
)

COMPOSITION_ENTRY_SYMBOL = "bounded_delta_learner.apply_bounded_delta_vote_step"
COMPOSITION_GUARD_ANCHOR = "bounded_delta_learner.py:1646-1647"

CANDIDATE_GLOBAL_CAP_CONTRACT_STEP0_NON_CLAIMS: tuple[str, ...] = (
    "B2-5c Step-0 is CPU/read-only contract characterization only",
    "B2-5c does NOT wire candidate_mode + global_cap_spec together",
    "B2-5c does NOT mint selection_parity_pass",
    "B2-5c does NOT flip global_cap_margin_only_reference",
    "B2-5c does NOT flip optimizer_credit_state / readiness rows",
    "B2-5c aggregate_branch_id reflects representative paired fixtures only",
    "classifier_negative fixtures are test-only and never drive the default aggregate",
)

CANDIDATE_GLOBAL_CAP_CONTRACT_STEP0_HARD_FALSE_FIELDS: tuple[str, ...] = (
    "selection_parity_pass",
    "native_selector_wired",
    "readiness_flip_authorized",
    "global_cap_margin_only_reference_flipped",
    "optimizer_credit_state_sub2_claim",
    "wiring_authorized",
)

PINNED_SURFACE_CANDIDATE = "PATH_CANDIDATE"
PINNED_SURFACE_EXACT_LOCAL = "PATH_EXACT_LOCAL"
PINNED_SURFACE_GCAP_SHADOW = "PATH_GCAP_SHADOW"

PINNED_SURFACES_FULL_EXECUTION: tuple[str, ...] = (
    PINNED_SURFACE_CANDIDATE,
    PINNED_SURFACE_EXACT_LOCAL,
    PINNED_SURFACE_GCAP_SHADOW,
)

FixtureRole = Literal["representative_consumer", "classifier_negative"]
FixtureTier = Literal["minimal", "saturated", "structural"]


class CandidateGlobalCapContractBranchId(str, Enum):
    MEASUREMENT_INVALID = "BR-CGC-MEASUREMENT-INVALID"
    RECONCILIATION_CONTRACT = "BR-CGC-RECONCILIATION-CONTRACT"
    PROOF_EXTENSION = "BR-CGC-PROOF-EXTENSION-RUNG"
    BRIDGE_IMPLEMENTATION = "BR-CGC-BRIDGE-IMPLEMENTATION-RUNG"


def composition_path_exists() -> bool:
    """Static auditable check — True only if candidate+global_cap composition is wired."""

    return False


@dataclass(frozen=True)
class CandidateGlobalCapContractFixtureMeasurement:
    fixture_name: str
    fixture_role: FixtureRole
    fixture_tier: FixtureTier
    pinned_surfaces: tuple[str, ...]
    total_sparse_event_count: int
    candidate_applied_row_identities_sha256: str
    candidate_residual_after_threshold_sha256: str
    candidate_q_changed_count: int
    candidate_local_update_pass: bool
    candidate_global_rate_cap_enabled: bool
    exact_local_applied_row_identities_sha256: str
    exact_local_residual_after_threshold_sha256: str
    exact_local_pre_cap_demand_count: int
    shadow_pre_cap_demand_sha256: str
    shadow_accepted_identities_sha256: str
    shadow_deferred_identities_sha256: str
    shadow_mutation_observed: bool
    cap: int
    accepted_count: int
    deferred_count: int
    identity_set_match: bool
    direction_match: bool
    residual_hash_match: bool
    ordering_match: bool
    global_cap_pure_subset_of_local_universe: bool
    deferred_backlog_authority_defined: bool
    saturation_exercised: bool
    structural_candidate_global_cap_reject: bool = False


@dataclass(frozen=True)
class CandidateGlobalCapContractBranchProbeResult:
    fixture_name: str
    fixture_role: FixtureRole
    branch_id: CandidateGlobalCapContractBranchId


@dataclass(frozen=True)
class CandidateGlobalCapContractStep0Receipt:
    schema_version: str = CANDIDATE_GLOBAL_CAP_CONTRACT_STEP0_SCHEMA_VERSION
    fixture_measurements: tuple[CandidateGlobalCapContractFixtureMeasurement, ...] = ()
    representative_measurements: tuple[CandidateGlobalCapContractFixtureMeasurement, ...] = ()
    classifier_negative_results: tuple[CandidateGlobalCapContractBranchProbeResult, ...] = ()
    aggregate_branch_id: CandidateGlobalCapContractBranchId = (
        CandidateGlobalCapContractBranchId.MEASUREMENT_INVALID
    )
    measurement_representative: bool = False
    composition_path_exists: bool = False
    composition_guard_anchor: str = COMPOSITION_GUARD_ANCHOR
    composition_entry_symbol: str = COMPOSITION_ENTRY_SYMBOL
    selection_parity_pass: bool = False
    native_selector_wired: bool = False
    readiness_flip_authorized: bool = False
    global_cap_margin_only_reference_flipped: bool = False
    optimizer_credit_state_sub2_claim: bool = False
    wiring_authorized: bool = False
    include_classifier_negatives: bool = False
    non_claims: tuple[str, ...] = CANDIDATE_GLOBAL_CAP_CONTRACT_STEP0_NON_CLAIMS


def _representative_rows(
    measurements: tuple[CandidateGlobalCapContractFixtureMeasurement, ...],
) -> tuple[CandidateGlobalCapContractFixtureMeasurement, ...]:
    return tuple(row for row in measurements if row.fixture_role == "representative_consumer")


def _full_execution_rows(
    representative_rows: tuple[CandidateGlobalCapContractFixtureMeasurement, ...],
) -> tuple[CandidateGlobalCapContractFixtureMeasurement, ...]:
    required = set(PINNED_SURFACES_FULL_EXECUTION)
    return tuple(
        row
        for row in representative_rows
        if row.fixture_tier != "structural"
        and required.issubset(set(row.pinned_surfaces))
    )


def representativeness_gate(
    representative_rows: tuple[CandidateGlobalCapContractFixtureMeasurement, ...],
) -> bool:
    execution_rows = _full_execution_rows(representative_rows)
    if not execution_rows:
        return False
    tiers = {row.fixture_tier for row in execution_rows}
    if "minimal" not in tiers or "saturated" not in tiers:
        return False
    if any(row.total_sparse_event_count <= 0 for row in execution_rows):
        return False
    if not any(row.saturation_exercised for row in execution_rows):
        return False
    if not any(row.structural_candidate_global_cap_reject for row in representative_rows):
        return False
    return all(
        set(PINNED_SURFACES_FULL_EXECUTION).issubset(set(row.pinned_surfaces))
        for row in execution_rows
    )


def classify_fixture_branch_probe(
    row: CandidateGlobalCapContractFixtureMeasurement,
) -> CandidateGlobalCapContractBranchId:
    if row.total_sparse_event_count <= 0 or not row.pinned_surfaces:
        return CandidateGlobalCapContractBranchId.MEASUREMENT_INVALID
    if not (
        row.identity_set_match
        and row.direction_match
        and row.residual_hash_match
        and row.ordering_match
        and row.global_cap_pure_subset_of_local_universe
        and row.deferred_backlog_authority_defined
    ):
        return CandidateGlobalCapContractBranchId.RECONCILIATION_CONTRACT
    if composition_path_exists():
        return CandidateGlobalCapContractBranchId.PROOF_EXTENSION
    return CandidateGlobalCapContractBranchId.BRIDGE_IMPLEMENTATION


def classify_aggregate_branch(
    representative_rows: tuple[CandidateGlobalCapContractFixtureMeasurement, ...],
) -> CandidateGlobalCapContractBranchId:
    if not representativeness_gate(representative_rows):
        return CandidateGlobalCapContractBranchId.MEASUREMENT_INVALID
    execution_rows = _full_execution_rows(representative_rows)
    if any(
        not (
            row.identity_set_match
            and row.direction_match
            and row.residual_hash_match
            and row.ordering_match
        )
        for row in execution_rows
    ):
        return CandidateGlobalCapContractBranchId.RECONCILIATION_CONTRACT
    if any(not row.deferred_backlog_authority_defined for row in execution_rows):
        return CandidateGlobalCapContractBranchId.RECONCILIATION_CONTRACT
    if not all(row.global_cap_pure_subset_of_local_universe for row in execution_rows):
        return CandidateGlobalCapContractBranchId.RECONCILIATION_CONTRACT
    if composition_path_exists():
        return CandidateGlobalCapContractBranchId.PROOF_EXTENSION
    return CandidateGlobalCapContractBranchId.BRIDGE_IMPLEMENTATION


def build_candidate_global_cap_contract_step0_receipt(
    *,
    fixture_measurements: tuple[CandidateGlobalCapContractFixtureMeasurement, ...],
    include_classifier_negatives: bool = False,
) -> CandidateGlobalCapContractStep0Receipt:
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
    negative_rows = tuple(
        row for row in fixture_measurements if row.fixture_role == "classifier_negative"
    )
    return CandidateGlobalCapContractStep0Receipt(
        fixture_measurements=fixture_measurements,
        representative_measurements=representative,
        classifier_negative_results=tuple(
            CandidateGlobalCapContractBranchProbeResult(
                fixture_name=row.fixture_name,
                fixture_role=row.fixture_role,
                branch_id=classify_fixture_branch_probe(row),
            )
            for row in negative_rows
        ),
        aggregate_branch_id=classify_aggregate_branch(representative),
        measurement_representative=representativeness_gate(representative),
        composition_path_exists=composition_path_exists(),
        include_classifier_negatives=include_classifier_negatives,
    )


def validate_candidate_global_cap_contract_step0_receipt(
    receipt: CandidateGlobalCapContractStep0Receipt,
) -> None:
    if receipt.schema_version != CANDIDATE_GLOBAL_CAP_CONTRACT_STEP0_SCHEMA_VERSION:
        raise ValueError("schema_version mismatch")
    for field in CANDIDATE_GLOBAL_CAP_CONTRACT_STEP0_HARD_FALSE_FIELDS:
        if bool(getattr(receipt, field)):
            raise ValueError(f"{field} must remain hard-false on Step-0 receipt")
    representative = _representative_rows(receipt.fixture_measurements)
    if receipt.representative_measurements != representative:
        raise ValueError("representative_measurements must match fixture_role filter")
    if receipt.composition_path_exists != composition_path_exists():
        raise ValueError("composition_path_exists inconsistent with static guard")
    if receipt.composition_guard_anchor != COMPOSITION_GUARD_ANCHOR:
        raise ValueError("composition_guard_anchor mismatch")
    expected_branch = classify_aggregate_branch(representative)
    if receipt.aggregate_branch_id != expected_branch:
        raise ValueError("aggregate_branch_id inconsistent with representative measurements")
    if receipt.measurement_representative != representativeness_gate(representative):
        raise ValueError("measurement_representative inconsistent with representative rows")
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
        if len(row.candidate_applied_row_identities_sha256) != 64:
            raise ValueError("candidate_applied_row_identities_sha256 must be sha256 hex")
        if len(row.exact_local_applied_row_identities_sha256) != 64:
            raise ValueError("exact_local_applied_row_identities_sha256 must be sha256 hex")
        if row.fixture_role == "representative_consumer" and row.fixture_tier != "structural":
            if row.total_sparse_event_count <= 0:
                raise ValueError("representative paired fixture requires non-zero sparse events")
