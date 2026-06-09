from __future__ import annotations

from calm.hrm_text_158.native_full_stack.accumulator_policy_shadow_screen import (
    ARM_ACCUMULATOR_ONLY,
    ARM_ACCUMULATOR_PLUS_TRANSIENT,
    ARM_INT16_BASELINE,
    ARM_TRANSIENT_RESOLVER_ONLY,
    CLAIM_ALGORITHMIC_PROXY_NOT_PHYSICAL_SUB2,
    CLAIM_INT16_REFERENCE,
    CLAIM_SUB2,
    CLAIM_TRANSIENT_FP_DEBT,
    DEFAULT_PREREG_THRESHOLDS,
    LABEL_ACCUMULATOR_TRACKS_INT16_POLICY,
    LABEL_SCREEN_HARNESS_OR_GATE_FAIL,
    LABEL_TRANSIENT_CARRIES_SELECTION,
    PRE_FULL_STACK_DIAGNOSTIC_ONLY,
    REQUIRED_SHADOW_ARMS,
    REQUIRED_THRESHOLD_FIELDS,
    run_accumulator_policy_shadow_screen,
)


def test_cpu_synthetic_shadow_screen_tracks_same_stream_and_embeds_readiness() -> None:
    receipt = run_accumulator_policy_shadow_screen(steps=50)

    assert receipt["pre_full_stack_diagnostic_only"] is True
    assert receipt["runtime_readiness_claim"] is False
    assert receipt["training_or_acquisition_claim"] is False
    assert receipt["q_mutation_applied_to_model"] is False
    assert receipt["compact_receipt"] is True
    assert receipt["diagnostic_contract"]["satisfied"] is True
    assert receipt["primary_label"] == LABEL_ACCUMULATOR_TRACKS_INT16_POLICY
    assert receipt["taxonomy_labels"] == [
        PRE_FULL_STACK_DIAGNOSTIC_ONLY,
        LABEL_ACCUMULATOR_TRACKS_INT16_POLICY,
    ]
    assert set(receipt["candidate_stream_hashes_by_arm"].values()) == {
        receipt["candidate_stream_hash"]
    }
    assert receipt["divergent_arm_state_hashes_allowed"] is True
    assert set(receipt["arms"]) == set(REQUIRED_SHADOW_ARMS)
    readiness = receipt["readiness_current_repo"]
    assert readiness["ready_for_main_science"] is False
    assert readiness["ready_for_pre_full_stack_diagnostic"] is False
    assert readiness["main_science_launch_blocked"] is True
    assert "persistent_qacc_authority" in readiness["blocker_surface_names"]


def test_arm_ledgers_separate_update_selection_and_label_proxy_vs_reference() -> None:
    receipt = run_accumulator_policy_shadow_screen(steps=50)
    arms = receipt["arms"]

    assert arms[ARM_INT16_BASELINE]["persistent_state_claim_class"] == CLAIM_INT16_REFERENCE
    assert arms[ARM_INT16_BASELINE]["selection_reads_decoded_int16"] is True
    assert (
        arms[ARM_ACCUMULATOR_ONLY]["persistent_state_claim_class"]
        == CLAIM_ALGORITHMIC_PROXY_NOT_PHYSICAL_SUB2
    )
    assert arms[ARM_ACCUMULATOR_ONLY]["fp_transient_used_for_update"] is True
    assert arms[ARM_ACCUMULATOR_ONLY]["fp_transient_used_for_selection"] is False
    assert arms[ARM_ACCUMULATOR_ONLY]["selection_reads_decoded_int16"] is False
    assert (
        arms[ARM_TRANSIENT_RESOLVER_ONLY]["persistent_state_claim_class"]
        == CLAIM_TRANSIENT_FP_DEBT
    )
    assert arms[ARM_TRANSIENT_RESOLVER_ONLY]["fp_transient_used_for_selection"] is True
    assert (
        arms[ARM_ACCUMULATOR_PLUS_TRANSIENT]["persistent_state_claim_class"]
        == CLAIM_TRANSIENT_FP_DEBT
    )


def test_required_threshold_fields_and_liveness_verdict_boundary_are_explicit() -> None:
    receipt = run_accumulator_policy_shadow_screen(steps=20)

    assert set(REQUIRED_THRESHOLD_FIELDS).issubset(receipt["thresholds"])
    assert receipt["thresholds"]["min_steps_for_verdict"] == 50
    assert receipt["thresholds"]["n20_liveness_only"] is True
    assert DEFAULT_PREREG_THRESHOLDS["min_jaccard_vs_int16"] == 0.90
    assert receipt["steps"] == 20
    assert receipt["verdict_allowed"] is False
    assert PRE_FULL_STACK_DIAGNOSTIC_ONLY in receipt["taxonomy_labels"]


def test_transient_carries_selection_taxonomy_when_accumulator_fails() -> None:
    receipt = run_accumulator_policy_shadow_screen(
        steps=50,
        synthetic_mode="transient_carries",
    )

    assert receipt["diagnostic_contract"]["satisfied"] is True
    assert receipt["primary_label"] == LABEL_TRANSIENT_CARRIES_SELECTION
    assert (
        receipt["aggregate_metrics"]["transient_only_advantage_vs_accumulator"]
        > receipt["thresholds"]["max_transient_only_advantage_allowed"]
    )


def test_contract_fail_closed_on_candidate_stream_drift() -> None:
    receipt = run_accumulator_policy_shadow_screen(
        steps=50,
        candidate_stream_hash_overrides={ARM_ACCUMULATOR_ONLY: "drifted-stream"},
    )

    assert receipt["screen_harness_or_gate_fail"] is True
    assert receipt["primary_label"] == LABEL_SCREEN_HARNESS_OR_GATE_FAIL
    assert receipt["diagnostic_contract"]["satisfied"] is False
    assert "same_candidate_stream" in receipt["failure_reasons"]


def test_contract_fail_closed_on_illegal_physical_sub2_selection_read() -> None:
    receipt = run_accumulator_policy_shadow_screen(
        steps=50,
        arm_ledger_overrides={
            ARM_ACCUMULATOR_ONLY: {
                "persistent_state_claim_class": CLAIM_SUB2,
                "selection_reads_decoded_int16": True,
            }
        },
    )

    assert receipt["screen_harness_or_gate_fail"] is True
    assert receipt["primary_label"] == LABEL_SCREEN_HARNESS_OR_GATE_FAIL
    assert "accumulator_physical_sub2_selection_clean" in receipt["failure_reasons"]


def test_contract_fail_closed_on_runtime_or_mutation_claims() -> None:
    receipt = run_accumulator_policy_shadow_screen(
        steps=50,
        q_mutation_applied_to_model=True,
    )

    assert receipt["screen_harness_or_gate_fail"] is True
    assert "no_q_mutation" in receipt["failure_reasons"]
