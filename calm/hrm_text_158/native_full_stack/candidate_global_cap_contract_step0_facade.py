"""Thin facade for B2-5c Step-0 candidate↔global-cap contract measurement."""
from __future__ import annotations

from calm.hrm_text_158.native_full_stack.candidate_global_cap_contract_step0_measurement import (
    build_classifier_negative_measurements,
    build_representative_consumer_measurements,
)
from calm.hrm_text_158.native_full_stack.candidate_global_cap_contract_step0_receipt import (
    CandidateGlobalCapContractStep0Receipt,
    build_candidate_global_cap_contract_step0_receipt,
    validate_candidate_global_cap_contract_step0_receipt,
)


def run_candidate_global_cap_contract_step0_suite(
    *,
    include_classifier_negatives: bool = False,
) -> CandidateGlobalCapContractStep0Receipt:
    measurements = list(build_representative_consumer_measurements())
    if include_classifier_negatives:
        measurements.extend(build_classifier_negative_measurements())
    receipt = build_candidate_global_cap_contract_step0_receipt(
        fixture_measurements=tuple(measurements),
        include_classifier_negatives=include_classifier_negatives,
    )
    validate_candidate_global_cap_contract_step0_receipt(receipt)
    return receipt


__all__ = [
    "run_candidate_global_cap_contract_step0_suite",
]
