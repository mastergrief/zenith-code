"""Thin facade for B2-5c Step-1a candidate+global-cap bridge reference suite."""
from __future__ import annotations

from calm.hrm_text_158.native_full_stack.candidate_global_cap_bridge_receipt import (
    build_candidate_global_cap_bridge_receipt,
    validate_candidate_global_cap_bridge_receipt,
    CandidateGlobalCapBridgeReceipt,
)
from calm.hrm_text_158.native_full_stack.candidate_global_cap_bridge_reference import (
    build_classifier_negative_bridge_measurements,
    build_representative_bridge_measurements,
)


def run_candidate_global_cap_bridge_suite(
    *,
    include_classifier_negatives: bool = False,
) -> CandidateGlobalCapBridgeReceipt:
    measurements = list(build_representative_bridge_measurements())
    if include_classifier_negatives:
        measurements.extend(build_classifier_negative_bridge_measurements())
    receipt = build_candidate_global_cap_bridge_receipt(
        fixture_measurements=tuple(measurements),
        include_classifier_negatives=include_classifier_negatives,
    )
    validate_candidate_global_cap_bridge_receipt(receipt)
    return receipt
