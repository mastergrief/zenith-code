"""B2-5c Step-1a candidate+global-cap bridge reference receipt (CPU/read-only)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CANDIDATE_GLOBAL_CAP_BRIDGE_SCHEMA_VERSION = (
    "hrm_text_158_candidate_global_cap_bridge/v0.b2_5c_step1a"
)

COMPOSITION_ENTRY_SYMBOL = "bounded_delta_learner.apply_bounded_delta_vote_step"
COMPOSITION_GUARD_ANCHOR = "bounded_delta_learner.py:1646-1647"

CANDIDATE_GLOBAL_CAP_BRIDGE_NON_CLAIMS: tuple[str, ...] = (
    "B2-5c Step-1a is CPU/read-only bridge reference characterization only",
    "B2-5c Step-1a does NOT wire candidate_mode + global_cap_spec in the trainer loop",
    "B2-5c Step-1a does NOT un-raise bounded_delta_learner.py:1646-1647",
    "B2-5c Step-1a does NOT mint selection_parity_pass",
    "B2-5c Step-1a does NOT flip global_cap_margin_only_reference",
    "B2-5c Step-1a does NOT flip optimizer_credit_state / readiness rows",
    "Step-1a novel claim (iii) saturated MARGIN identity is confirmatory not independent",
    "Post-cap equivalence on no-clip fixtures is cross-path composability not tautological discovery",
    "classifier_negative fixtures are test-only and never drive the default aggregate",
)

CANDIDATE_GLOBAL_CAP_BRIDGE_HARD_FALSE_FIELDS: tuple[str, ...] = (
    "selection_parity_pass",
    "native_selector_wired",
    "readiness_flip_authorized",
    "global_cap_margin_only_reference_flipped",
    "optimizer_credit_state_sub2_claim",
    "wiring_authorized",
    "trainer_guard_unraised",
)

FixtureRole = Literal["representative_consumer", "classifier_negative"]
MagnitudeRegime = Literal["no_clip_exact_add_back", "clip_boundary_reconciliation"]


def composition_path_exists() -> bool:
    """Static auditable check — True only if candidate+global_cap composition is wired."""

    return False


@dataclass(frozen=True)
class CandidateGlobalCapBridgeFixtureMeasurement:
    fixture_name: str
    fixture_role: FixtureRole
    total_sparse_event_count: int
    magnitude_regime: MagnitudeRegime
    add_back_clip_boundary_reconciliation: bool
    fidelity_lattice_pass: bool
    bridge_equivalent: bool
    accepted_identities_match: bool
    deferred_identities_match: bool
    accepted_order_match: bool
    cap_counts_match: bool
    step1a_novel_claim_materialization_fidelity: bool
    step1a_novel_claim_cap_api_composability: bool
    step1a_novel_claim_saturated_margin_ordering_identity: bool
    candidate_applied_row_identities_sha256: str
    bridge_accepted_identities_sha256: str
    oracle_accepted_identities_sha256: str
    structural_candidate_global_cap_reject: bool = False


@dataclass(frozen=True)
class CandidateGlobalCapBridgeReceipt:
    schema_version: str
    representative_measurements: tuple[CandidateGlobalCapBridgeFixtureMeasurement, ...]
    classifier_negative_results: tuple[CandidateGlobalCapBridgeFixtureMeasurement, ...]
    include_classifier_negatives: bool
    measurement_representative: bool
    aggregate_bridge_equivalent: bool
    composition_path_exists: bool
    composition_guard_anchor: str
    selection_parity_pass: bool
    native_selector_wired: bool
    readiness_flip_authorized: bool
    global_cap_margin_only_reference_flipped: bool
    optimizer_credit_state_sub2_claim: bool
    wiring_authorized: bool
    trainer_guard_unraised: bool
    non_claims: tuple[str, ...]


def _execution_rows(
    rows: tuple[CandidateGlobalCapBridgeFixtureMeasurement, ...],
) -> tuple[CandidateGlobalCapBridgeFixtureMeasurement, ...]:
    return tuple(
        row
        for row in rows
        if row.fixture_role == "representative_consumer"
        and not row.structural_candidate_global_cap_reject
    )


def representativeness_gate(
    representative_rows: tuple[CandidateGlobalCapBridgeFixtureMeasurement, ...],
) -> bool:
    execution_rows = _execution_rows(representative_rows)
    if len(execution_rows) < 2:
        return False
    names = {row.fixture_name for row in execution_rows}
    if "F_BRIDGE_MINIMAL" not in names or "F_BRIDGE_SATURATED" not in names:
        return False
    if any(row.total_sparse_event_count <= 0 for row in execution_rows):
        return False
    if not any(row.add_back_clip_boundary_reconciliation is False for row in execution_rows):
        return False
    if not any(row.structural_candidate_global_cap_reject for row in representative_rows):
        return False
    return all(row.magnitude_regime == "no_clip_exact_add_back" for row in execution_rows)


def validate_candidate_global_cap_bridge_receipt(
    receipt: CandidateGlobalCapBridgeReceipt,
) -> None:
    if receipt.schema_version != CANDIDATE_GLOBAL_CAP_BRIDGE_SCHEMA_VERSION:
        raise ValueError(f"unexpected schema_version {receipt.schema_version!r}")
    for field in CANDIDATE_GLOBAL_CAP_BRIDGE_HARD_FALSE_FIELDS:
        if getattr(receipt, field) is not False:
            raise ValueError(f"hard-false field {field!r} must be False")
    if receipt.composition_path_exists is not False:
        raise ValueError("composition_path_exists must remain False")
    if receipt.composition_guard_anchor != COMPOSITION_GUARD_ANCHOR:
        raise ValueError("composition_guard_anchor mismatch")
    if not receipt.include_classifier_negatives and receipt.classifier_negative_results:
        raise ValueError("classifier negatives present without include_classifier_negatives=True")
    for row in (*receipt.representative_measurements, *receipt.classifier_negative_results):
        if row.fixture_role == "classifier_negative" and not receipt.include_classifier_negatives:
            raise ValueError(
                f"classifier_negative fixture {row.fixture_name!r} requires include_classifier_negatives=True",
            )
        if row.add_back_clip_boundary_reconciliation and row.bridge_equivalent:
            raise ValueError(
                f"{row.fixture_name}: clip-boundary reconciliation must not be bridge_equivalent",
            )
        if row.fidelity_lattice_pass and not row.step1a_novel_claim_materialization_fidelity:
            raise ValueError(
                f"{row.fixture_name}: fidelity pass requires materialization claim",
            )
        if not row.fidelity_lattice_pass and row.step1a_novel_claim_materialization_fidelity:
            raise ValueError(
                f"{row.fixture_name}: materialization claim requires fidelity pass",
            )
    execution_rows = _execution_rows(receipt.representative_measurements)
    expected_rep = representativeness_gate(receipt.representative_measurements)
    if receipt.measurement_representative is not expected_rep:
        raise ValueError("measurement_representative flag mismatch")
    if expected_rep:
        if not all(row.bridge_equivalent for row in execution_rows):
            raise ValueError("representative execution rows must be bridge_equivalent")
        if receipt.aggregate_bridge_equivalent is not True:
            raise ValueError("aggregate_bridge_equivalent must be True for representative suite")


def build_candidate_global_cap_bridge_receipt(
    *,
    fixture_measurements: tuple[CandidateGlobalCapBridgeFixtureMeasurement, ...],
    include_classifier_negatives: bool = False,
) -> CandidateGlobalCapBridgeReceipt:
    representative = tuple(
        row for row in fixture_measurements if row.fixture_role == "representative_consumer"
    )
    negatives = tuple(
        row for row in fixture_measurements if row.fixture_role == "classifier_negative"
    )
    if negatives and not include_classifier_negatives:
        raise ValueError("include_classifier_negatives=True required for classifier_negative rows")
    execution_rows = _execution_rows(representative)
    rep_ok = representativeness_gate(representative)
    aggregate_ok = rep_ok and all(row.bridge_equivalent for row in execution_rows)
    return CandidateGlobalCapBridgeReceipt(
        schema_version=CANDIDATE_GLOBAL_CAP_BRIDGE_SCHEMA_VERSION,
        representative_measurements=representative,
        classifier_negative_results=negatives if include_classifier_negatives else (),
        include_classifier_negatives=include_classifier_negatives,
        measurement_representative=rep_ok,
        aggregate_bridge_equivalent=aggregate_ok,
        composition_path_exists=composition_path_exists(),
        composition_guard_anchor=COMPOSITION_GUARD_ANCHOR,
        selection_parity_pass=False,
        native_selector_wired=False,
        readiness_flip_authorized=False,
        global_cap_margin_only_reference_flipped=False,
        optimizer_credit_state_sub2_claim=False,
        wiring_authorized=False,
        trainer_guard_unraised=False,
        non_claims=CANDIDATE_GLOBAL_CAP_BRIDGE_NON_CLAIMS,
    )
