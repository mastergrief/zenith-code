"""Thin facade for B2-5b Step-0 optimizer_credit_state consumer measurement."""
from __future__ import annotations

from calm.hrm_text_158.native_full_stack.optimizer_credit_state_global_cap_consumer_step0_measurement import (
    build_classifier_negative_measurements,
    build_representative_consumer_measurements,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state_global_cap_consumer_step0_receipt import (
    OptimizerCreditStateGlobalCapConsumerStep0Receipt,
    build_optimizer_credit_state_global_cap_consumer_step0_receipt,
    validate_optimizer_credit_state_global_cap_consumer_step0_receipt,
)


def run_optimizer_credit_state_global_cap_consumer_step0_suite(
    *,
    include_classifier_negatives: bool = False,
) -> OptimizerCreditStateGlobalCapConsumerStep0Receipt:
    measurements = list(build_representative_consumer_measurements())
    if include_classifier_negatives:
        measurements.extend(build_classifier_negative_measurements())
    receipt = build_optimizer_credit_state_global_cap_consumer_step0_receipt(
        fixture_measurements=tuple(measurements),
        include_classifier_negatives=include_classifier_negatives,
    )
    validate_optimizer_credit_state_global_cap_consumer_step0_receipt(receipt)
    return receipt


__all__ = [
    "run_optimizer_credit_state_global_cap_consumer_step0_suite",
]
