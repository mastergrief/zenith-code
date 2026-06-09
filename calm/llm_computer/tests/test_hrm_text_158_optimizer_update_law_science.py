"""Step-1 optimizer/update-law science packet tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.optimizer_update_law_science import (
    ACTIVATION_CREDIT_BRANCHES,
    ACTIVATION_CREDIT_DIAG_FISHER_Q5_ABLATION_FAMILY_ID,
    ACTIVATION_CREDIT_DIAG_FISHER_Q5_FIELD,
    ACTIVATION_CREDIT_FRESH_CONFIRMATION_SEED,
    ACTIVATION_CREDIT_MAGNITUDE_Q5_BIN_COUNT,
    ACTIVATION_CREDIT_MAGNITUDE_Q5_MIN_BUCKET_SIZE,
    ACTIVATION_CREDIT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND,
    ACTIVATION_CREDIT_MEASUREMENT_PACKET_KIND,
    ACTIVATION_CREDIT_PRIMARY_FAMILY_ID,
    ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE,
    ACTIVATION_CREDIT_SCALE_SMOKE_LAUNCH_BUNDLE_PACKET_KIND,
    ACTIVATION_CREDIT_SECOND_ORDER_SNR_EPS,
    ACTIVATION_CREDIT_SNR_Q5_ABLATION_FAMILY_ID,
    ACTIVATION_CREDIT_SNR_Q5_FIELD,
    ACTIVATION_CREDIT_SMOKE_BATCH_SIZE,
    ACTIVATION_CREDIT_SMOKE_MAX_SAMPLED_CANDIDATES,
    ACTIVATION_CREDIT_STDERR_PATH_ENV,
    ACTIVATION_CREDIT_STDOUT_PATH_ENV,
    ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_FIELD,
    B2B_SEQUENTIAL_STEPS_FOR_VERDICT,
    B2B_SEQUENTIAL_WITHIN_TIE_BAND_LAUNCH_BUNDLE_PACKET_KIND,
    B2B_SEQUENTIAL_WITHIN_TIE_BAND_PACKET_KIND,
    ARM_A0_RANK_BUCKET_CURRENT,
    ARM_A1_RANK_BUCKET_ORDER_MATCHED,
    ARM_B_CAP_MAX_ABS_1024,
    ARM_B_RANK_FREE_SIGN_PRESSURE,
    ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER,
    ARM_INVERTED_SIGN_PRESSURE,
    BRANCH_ACTIVATION_CREDIT_AMBIGUOUS_NO_BRANCH,
    BRANCH_ACTIVATION_CREDIT_CANDIDATE_SIGNAL,
    BRANCH_ACTIVATION_CREDIT_MISSING_SIGNAL_DEEPER_THAN_FIRST_ORDER_CREDIT_STORAGE,
    BRANCH_CANDIDATE_GENERATION_BAD_OR_NO_LOCAL_SIGNAL,
    BRANCH_CANDIDATE_SET_VIABLE_CREDIT_RANKING_BAD,
    BRANCH_CREDIT_MAGNITUDE_BAD_SIGN_USABLE,
    BRANCH_CURRENT_ORDER_NOT_NECESSARY,
    BRANCH_CURRENT_QACC_MARGIN_ORDER_BUNDLE_CARRIER,
    BRANCH_INSUFFICIENT_SEPARATION,
    BRANCH_MEASUREMENT_AMBIGUOUS_NO_BRANCH,
    BRANCH_MEASUREMENT_AMBIGUOUS_TIE_BAND_ALIASING,
    BRANCH_MASS_CONFOUNDED_CURRENT_ORDER_SIGNAL,
    BRANCH_MEASUREMENT_LOSS_POWERED,
    BRANCH_MEASUREMENT_ORDER_SENSITIVE,
    BRANCH_MEASUREMENT_POWERED,
    BRANCH_MEASUREMENT_UNDERPOWERED,
    BRANCH_NO_MATCH_DIFFERENT_CREDIT_SOURCE,
    BRANCH_ORACLE_INFEASIBLE_OR_TOO_EXPENSIVE,
    BRANCH_PARTIAL_LOCAL_SIGNAL,
    BRANCH_POWERED_NEGATIVE_OR_LOSS_ONLY,
    BRANCH_PREREGISTERED_CHEAP_LEARNER_FEATURE_FAMILY_CANNOT_PREDICT_REGRET,
    BRANCH_WITHIN_TIE_BAND_AMBIGUOUS_NO_BRANCH,
    BRANCH_WITHIN_TIE_BAND_LEARNER_FEATURES_SEPARATE_REGRET,
    BRANCH_WITHIN_TIE_BAND_NEEDS_NEW_LEARNER_STATE,
    CREDIT_RANKING_PIVOT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND,
    CREDIT_RANKING_PIVOT_MEASUREMENT_PACKET_KIND,
    BRANCH_PRIOR_NULL_SETUP_UNVERIFIED,
    BRANCH_RANK_FREE_POSITIVE,
    BRANCH_RANK_MAGNITUDE_CONDITIONED_ON_CURRENT_ORDER,
    BRANCH_SCHEDULER_ONLY_ORDER_SENSITIVE,
    BRANCH_TIE_POLICY_OR_OVERUPDATE,
    BRANCH_A0_COMPONENT_ORDER_ROBUST,
    CONTROL_PARITY_FRACTION_MAX,
    CONTROL_PARITY_FRACTION_MIN,
    FIXED_RANK_BUCKET_NON_TARGET_AUX,
    ORACLE_ARM_CURRENT_CREDIT_RANK_BUCKET_CURRENT_ORDER,
    ORACLE_ARM_DETERMINISTIC_HASH_SAME_VOTES,
    ORACLE_ARM_DIAGNOSTIC_LOCAL_LOSS_DELTA,
    ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES,
    ORACLE_SCREEN_BRANCHES,
    ORACLE_SCREEN_CONTRAST_SEEDS,
    ORACLE_SCREEN_LAUNCH_BUNDLE_PACKET_KIND,
    ORACLE_SCREEN_MAX_SECONDS_BY_BUDGET,
    ORACLE_SCREEN_PACKET_KIND,
    ORACLE_SCREEN_PROMOTION_ORDER_SEEDS,
    ORACLE_SCREEN_SCIENCE_CONTRACT_COMMIT_SHA,
    ORACLE_WIDER_SCREEN_INTERPRETATION_VERDICTS,
    ORACLE_WIDER_SCREEN_VERDICT_CREDIT_RANKING_BAD,
    ORACLE_WIDER_SCREEN_VERDICT_RANKING_EFFECTIVELY_OK,
    ORACLE_WIDER_SCREEN_VERDICT_RANKING_SUBOPTIMAL,
    SCIENCE_MODE_BRANCH_VERDICT,
    SCIENCE_MODE_PRETERMINAL_SCREEN,
    STEP1_DRY_RUN_PACKET_KIND,
    STEP2_LAUNCH_BUNDLE_PACKET_KIND,
    STEP3_BASELINE_MAX_ABS_PER_TENSOR,
    STEP3_CAP_MAX_ABS_PER_TENSOR,
    STEP3_MEASUREMENT_POWER_TRUST_REGION_PACKET_KIND,
    STEP4_POWERED_RANK_SIGNAL_DECOMPOSITION_PACKET_KIND,
    STEP5_SUPPORT_ORDER_SEED,
    STEP5_SUPPORT_ORDER_TRAJECTORY_ROBUSTNESS_PACKET_KIND,
    STEP6_FIXED_PREREG_NEW_SEED,
    STEP6_ORDER_AVERAGED_A0_COMPONENT_DECOMPOSITION_PACKET_KIND,
    STEP6_SUPPORT_ORDER_SEEDS,
    build_activation_credit_measurement_launch_bundle,
    build_activation_credit_measurement_packet,
    build_activation_credit_scale_smoke_launch_bundle,
    build_b2b_sequential_within_tie_band_launch_bundle,
    build_b2b_sequential_within_tie_band_packet,
    build_candidate_set_viability_oracle_screen_packet,
    build_candidate_set_viability_oracle_screen_launch_bundle,
    build_credit_ranking_pivot_measurement_launch_bundle,
    build_credit_ranking_pivot_measurement_packet,
    build_within_tie_band_discriminator_launch_bundle,
    build_within_tie_band_discriminator_packet,
    build_measurement_power_then_trust_region_packet,
    build_powered_rank_signal_decomposition_packet,
    build_order_averaged_a0_component_decomposition_packet,
    build_support_order_trajectory_robustness_packet,
    TIE_POLICY_CURRENT_MARGIN_INDEX,
    TIE_POLICY_DETERMINISTIC_HASH_MATCHED,
    build_optimizer_update_law_launch_bundle,
    build_optimizer_update_law_science_packet,
    classify_candidate_set_viability_oracle_screen,
    classify_optimizer_update_law_branch,
    classify_step4_rank_signal_decomposition,
    classify_step3_power_floor,
    default_step4_mass_confound_rule,
    oracle_screen_budget_max_seconds,
    oracle_screen_effectively_ok_rank_position_exclusive_bound,
    step4_arm_matches_a0,
    step4_mass_confound_detected,
    validate_measurement_power_then_trust_region_packet,
    validate_activation_credit_measurement_launch_bundle,
    validate_activation_credit_measurement_packet,
    validate_activation_credit_scale_smoke_launch_bundle,
    validate_b2b_sequential_within_tie_band_launch_bundle,
    validate_b2b_sequential_within_tie_band_packet,
    validate_candidate_set_viability_oracle_screen_launch_bundle,
    validate_candidate_set_viability_oracle_screen_packet,
    validate_credit_ranking_pivot_measurement_launch_bundle,
    validate_credit_ranking_pivot_measurement_packet,
    validate_within_tie_band_discriminator_launch_bundle,
    validate_within_tie_band_discriminator_packet,
    validate_optimizer_update_law_launch_bundle,
    validate_optimizer_update_law_science_packet,
    validate_order_averaged_a0_component_decomposition_packet,
    validate_powered_rank_signal_decomposition_packet,
    validate_support_order_trajectory_robustness_packet,
    WITHIN_TIE_BAND_DISCRIMINATOR_LAUNCH_BUNDLE_PACKET_KIND,
    WITHIN_TIE_BAND_DISCRIMINATOR_PACKET_KIND,
    WITHIN_TIE_BAND_PRIMARY_FAMILY_ID,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import build_support_order_trajectory_proof
from scripts.hrm_text_158_optimizer_update_law_science_packet import main as packet_main


def test_science_packet_declares_a0_a1_b_and_falsifier_with_fixed_gates():
    packet = build_optimizer_update_law_science_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
        mode=SCIENCE_MODE_BRANCH_VERDICT,
    )
    arms = {arm["arm_id"]: arm for arm in packet["arms"]}

    assert packet["diagnostic_class"] == "pre_full_stack_diagnostic"
    assert packet["launch_gate_id"] is None
    assert packet["n_rows"] == 50
    assert packet["gpu_launched"] is False
    assert packet["checkpoint_written"] is False
    assert packet["pt_mutated"] is False
    assert packet["readiness_claim"] is False
    assert packet["full_sub2_claim"] is False
    assert packet["optimizer_credit_state_row_flip"] is False
    assert packet["aux_vote_law"] == FIXED_RANK_BUCKET_NON_TARGET_AUX
    assert arms[ARM_A0_RANK_BUCKET_CURRENT]["tie_policy_id"] == TIE_POLICY_CURRENT_MARGIN_INDEX
    assert arms[ARM_A1_RANK_BUCKET_ORDER_MATCHED]["tie_policy_id"] == TIE_POLICY_DETERMINISTIC_HASH_MATCHED
    assert arms[ARM_B_RANK_FREE_SIGN_PRESSURE]["tie_policy_id"] == TIE_POLICY_DETERMINISTIC_HASH_MATCHED
    assert arms[ARM_INVERTED_SIGN_PRESSURE]["tie_policy_id"] == TIE_POLICY_DETERMINISTIC_HASH_MATCHED
    gate = packet["control_parity_gate"]
    assert gate["min_inclusive"] == CONTROL_PARITY_FRACTION_MIN
    assert gate["max_inclusive"] == CONTROL_PARITY_FRACTION_MAX
    assert gate["qualitative_prior_null_signature_required"] is True
    assert gate["requires_current_improves_vs_baseline"] is True
    assert gate["requires_random_matches_or_beats_current"] is True
    validate_optimizer_update_law_science_packet(packet)


@pytest.mark.parametrize(
    "mutation,error",
    [
        ({"launch_gate_id": "future-launch"}, "launch_gate_id=null"),
        ({"readiness_claim": True}, "readiness_claim"),
        ({"full_sub2_claim": True}, "full_sub2_claim"),
        ({"raw_per_proposal_arrays_included": ["bad"]}, "raw per-proposal arrays"),
        ({"aux_vote_law": "rank_free_aux"}, "aux_vote_law"),
    ],
)
def test_science_packet_validator_rejects_laundering_fields(mutation, error):
    packet = build_optimizer_update_law_science_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
    )
    packet.update(mutation)
    with pytest.raises(ValueError, match=error):
        validate_optimizer_update_law_science_packet(packet)


def test_science_packet_validator_rejects_live_q_as_banked_hash_gate():
    packet = build_optimizer_update_law_science_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
    )
    packet["hash_gate_policy"]["required_sources"] = [
        "banked_parent_checkpoint",
        "live_q_after_update",
    ]

    with pytest.raises(ValueError, match="live post-arm q"):
        validate_optimizer_update_law_science_packet(packet)


@pytest.mark.parametrize(
    "gate_field,error",
    [
        ("requires_current_improves_vs_baseline", "current improves vs baseline"),
        ("requires_random_matches_or_beats_current", "random matches or beats current"),
    ],
)
def test_science_packet_validator_requires_qualitative_parity_fields(gate_field, error):
    packet = build_optimizer_update_law_science_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
    )
    packet["control_parity_gate"][gate_field] = False

    with pytest.raises(ValueError, match=error):
        validate_optimizer_update_law_science_packet(packet)


def test_branch_classifier_requires_b_to_beat_a0_and_a1():
    assert classify_optimizer_update_law_branch(
        mode=SCIENCE_MODE_PRETERMINAL_SCREEN,
        control_parity_pass=True,
        b_beats_a0=True,
        b_beats_a1=True,
        b_beats_falsifiers=True,
    ) is None
    assert classify_optimizer_update_law_branch(
        mode=SCIENCE_MODE_BRANCH_VERDICT,
        control_parity_pass=False,
        b_beats_a0=True,
        b_beats_a1=True,
        b_beats_falsifiers=True,
    ) == BRANCH_INSUFFICIENT_SEPARATION
    assert classify_optimizer_update_law_branch(
        mode=SCIENCE_MODE_BRANCH_VERDICT,
        control_parity_pass=True,
        b_beats_a0=True,
        b_beats_a1=False,
        b_beats_falsifiers=True,
    ) == BRANCH_TIE_POLICY_OR_OVERUPDATE
    assert classify_optimizer_update_law_branch(
        mode=SCIENCE_MODE_BRANCH_VERDICT,
        control_parity_pass=True,
        b_beats_a0=True,
        b_beats_a1=True,
        b_beats_falsifiers=True,
    ) == BRANCH_RANK_FREE_POSITIVE


def test_branch_classifier_fails_closed_when_prior_null_setup_unverified():
    branch = classify_optimizer_update_law_branch(
        mode=SCIENCE_MODE_BRANCH_VERDICT,
        control_parity_pass=True,
        b_beats_a0=True,
        b_beats_a1=True,
        b_beats_falsifiers=True,
        prior_null_setup_verified=False,
    )

    assert branch == BRANCH_INSUFFICIENT_SEPARATION
    assert branch != BRANCH_RANK_FREE_POSITIVE


def test_step2_launch_bundle_is_author_only_and_structured():
    bundle = build_optimizer_update_law_launch_bundle(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-step2",
    )

    validate_optimizer_update_law_launch_bundle(bundle)
    assert bundle["packet_kind"] == STEP2_LAUNCH_BUNDLE_PACKET_KIND
    assert bundle["author_only"] is True
    assert bundle["commands_executed"] is False
    assert bundle["gpu_launched"] is False
    assert bundle["launch_gate_id"] is None
    assert bundle["pt_mutated"] is False
    assert bundle["readiness_claim"] is False
    assert bundle["branch_result"] is None
    assert bundle["mode_sequence"] == [
        SCIENCE_MODE_PRETERMINAL_SCREEN,
        SCIENCE_MODE_BRANCH_VERDICT,
    ]
    assert bundle["resource_lane"]["resource_lane_required"] is True
    assert bundle["resource_lane"]["lane_name"] == "gpu:0"
    assert bundle["resource_lane"]["resolved_at_launch"] is False
    assert bundle["screen_before_verdict"]["verdict_blocked_until"]["prior_null_setup_verified"] is True
    assert bundle["terminal_criteria"]["prior_null_setup_gate"]["unverified_branch"] == (
        BRANCH_PRIOR_NULL_SETUP_UNVERIFIED
    )
    assert "live_q_after_update" in bundle["hash_gate_policy"]["explicitly_not_sources"]
    assert "post_arm_live_q_mutation" in bundle["hash_gate_policy"]["explicitly_not_sources"]

    prior = bundle["prior_verdict_parent_ref"]
    assert prior["artifact_path"] == "/tmp/hrm158_shadow_prefix_lane3_hybrid_gpu_n50_trace.tierb.final.json"
    assert prior["candidate_label"] == "credit_ranking_uninformative_update_law_pivot"
    assert prior["support_name"] == "L0c2-K2-addition-120"
    assert prior["support_sha"] == "21c8a2f8c15fd68571407e6d1f215ab045ffc5a2a91e4b5a44b50bcd46b6faf0"
    assert prior["row_count"] == 120
    assert prior["lane"] == "lane3"
    assert prior["acc_mode"] == "applied_crossing_direction_plus_4bit_residual"
    assert prior["vote_fidelity"] == "dry2"
    assert prior["activation_mode"] == "ternary_group128_codec_from_step0"
    assert prior["rank_vote_spec_sha256"] == "6c109e0482292edf72d3cc4ada6bda0840e67e8dbfac4ad7fd64d353602806a5"
    assert prior["vote_mapping_family"] == "rank_bucketed"
    assert prior["artifact_literal_support_sha256"] == (
        "21c8a2f8c15fd68571407e6d1f215ab045ffc5a2a91e4b5a44b50bcd46b6faf0"
    )
    assert prior["artifact_literal_acc_mode"] == "applied_crossing_direction_plus_4bit_residual"
    assert prior["artifact_literal_activation_mode"] == "ternary_group128_codec_from_step0"
    assert prior["artifact_literal_vote_fidelity"] == "dry2"
    assert prior["artifact_literal_rank_vote_spec_sha256"] == (
        "6c109e0482292edf72d3cc4ada6bda0840e67e8dbfac4ad7fd64d353602806a5"
    )
    assert prior["artifact_literal_vote_mapping_family"] == "rank_bucketed"
    assert prior["artifact_literal_lane"] == "lane3"
    assert prior["artifact_literal_random_seed"] == 17
    assert prior["artifact_literal_step_or_sample_count"] == 50
    assert prior["human_readable_labels"]["acc_mode_family"] == "hybrid"
    assert prior["random_seed"] == 17
    assert prior["step_or_sample_count"] == 50
    assert prior["current_beats_competitors_fraction"] == 0.3
    assert prior["prior_null_setup_verified"] is False
    assert prior["blocks_control_parity_band"] is True

    commands = bundle["commands"]
    assert len(commands) == 8
    assert {(cmd["mode"], cmd["arm_id"]) for cmd in commands} == {
        (mode, arm)
        for mode in (SCIENCE_MODE_PRETERMINAL_SCREEN, SCIENCE_MODE_BRANCH_VERDICT)
        for arm in (
            ARM_A0_RANK_BUCKET_CURRENT,
            ARM_A1_RANK_BUCKET_ORDER_MATCHED,
            ARM_B_RANK_FREE_SIGN_PRESSURE,
            ARM_INVERTED_SIGN_PRESSURE,
        )
    }
    for command in commands:
        assert command["cwd"] == "/repo"
        assert command["env"]["HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE"] == "1"
        assert command["env"]["HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH"] == "1"
        assert command["steps_requested"] == command["n_rows"]
        assert command["steps_source"] == "SCIENCE_MODE_ROWS[mode]"
        assert isinstance(command["argv"], list)
        assert command["argv"][command["argv"].index("--steps") + 1] == str(command["steps_requested"])
        assert command["argv"][command["argv"].index("--max-steps-hard") + 1] == "50"
        assert command["argv"][command["argv"].index("--device") + 1] == "cuda:0"
        assert command["argv"][command["argv"].index("--science-arm") + 1] == command["arm_id"]
        assert command["expected_exit_policy"] == "exit_0_required_else_stop_no_retry_no_verdict"


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda packet: packet.update({"commands_executed": True}), "commands_executed"),
        (lambda packet: packet.update({"gpu_launched": True}), "gpu_launched"),
        (lambda packet: packet.update({"branch_result": BRANCH_RANK_FREE_POSITIVE}), "branch_result"),
        (lambda packet: packet.update({"runtime_results": {}}), "runtime field runtime_results"),
        (
            lambda packet: packet["resource_lane"].update({"resolved_at_launch": True}),
            "resource lane resolved",
        ),
        (
            lambda packet: packet.update(
                {"mode_sequence": [SCIENCE_MODE_BRANCH_VERDICT, SCIENCE_MODE_PRETERMINAL_SCREEN]},
            ),
            "screen before branch verdict",
        ),
        (
            lambda packet: packet["phase_budgets"].pop("artifact_flush"),
            "phase_budgets missing",
        ),
        (
            lambda packet: packet["commands"][0].update({"steps_requested": 999}),
            "steps_requested must equal n_rows",
        ),
        (
            lambda packet: packet["commands"][0]["argv"].__setitem__(
                packet["commands"][0]["argv"].index("--device") + 1,
                "cpu",
            ),
            "--device must target CUDA",
        ),
        (
            lambda packet: packet["commands"][0]["argv"].__setitem__(
                packet["commands"][0]["argv"].index("--max-steps-hard") + 1,
                "20",
            ),
            "--max-steps-hard",
        ),
        (
            lambda packet: packet["commands"][0]["argv"].remove("--steps"),
            "missing --steps value|missing required probe launch arguments",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update({"support_sha": "wrong"}),
            "support_sha",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update({"acc_mode": "hybrid"}),
            "acc_mode",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update({"activation_mode": "shadow_prefix"}),
            "activation_mode",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update({"vote_fidelity": "tierb"}),
            "vote_fidelity",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update({"rank_vote_spec_sha256": "wrong"}),
            "rank_vote_spec_sha256",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update({"vote_mapping_family": "wrong"}),
            "vote_mapping_family",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update({"lane": "lane4"}),
            "lane",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update({"random_seed": 18}),
            "random_seed",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update({"step_or_sample_count": 49}),
            "step_or_sample_count",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update({"support_seed": 18}),
            "support_seed",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update(
                {"artifact_literal_parent_path": "wrong.pt"},
            ),
            "artifact literal parent path",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update(
                {"artifact_literal_support_sha256": "wrong"},
            ),
            "artifact_literal_support_sha256",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update(
                {"artifact_literal_acc_mode": "hybrid"},
            ),
            "artifact_literal_acc_mode",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update(
                {"artifact_literal_activation_mode": "shadow_prefix"},
            ),
            "artifact_literal_activation_mode",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update(
                {"artifact_literal_vote_fidelity": "tierb"},
            ),
            "artifact_literal_vote_fidelity",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update(
                {"artifact_literal_rank_vote_spec_sha256": "wrong"},
            ),
            "artifact_literal_rank_vote_spec_sha256",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update(
                {"artifact_literal_vote_mapping_family": "wrong"},
            ),
            "artifact_literal_vote_mapping_family",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update(
                {"artifact_literal_lane": "lane4"},
            ),
            "artifact_literal_lane",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update(
                {"artifact_literal_random_seed": 18},
            ),
            "artifact_literal_random_seed",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update(
                {"artifact_literal_step_or_sample_count": 49},
            ),
            "artifact_literal_step_or_sample_count",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update(
                {"artifact_literal_parent_sha256": "wrong"},
            ),
            "artifact literal parent sha",
        ),
        (
            lambda packet: packet["prior_verdict_parent_ref"].update({"prior_null_setup_verified": True}),
            "prior_null_setup_verified",
        ),
        (
            lambda packet: packet["terminal_criteria"]["prior_null_setup_gate"].update(
                {"blocks_control_parity_band": False},
            ),
            "blocks_control_parity_band",
        ),
    ],
)
def test_step2_launch_bundle_validator_rejects_runtime_and_dependency_confusion(mutation, error):
    bundle = build_optimizer_update_law_launch_bundle(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-step2",
    )
    mutation(bundle)

    with pytest.raises(ValueError, match=error):
        validate_optimizer_update_law_launch_bundle(bundle)


def test_step3_measurement_power_trust_region_packet_pins_cap_and_power_floor():
    packet = build_measurement_power_then_trust_region_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-step3",
    )

    validate_measurement_power_then_trust_region_packet(packet)
    assert packet["packet_kind"] == STEP3_MEASUREMENT_POWER_TRUST_REGION_PACKET_KIND
    assert packet["author_only"] is True
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["launch_gate_id"] is None
    assert packet["pt_mutated"] is False
    assert packet["readiness_claim"] is False
    assert packet["full_sub2_claim"] is False
    assert packet["branch_result"] is None
    assert packet["mode_sequence"] == [
        "measurement_power_150",
        "measurement_power_300",
        "trust_region_cap_150",
        "trust_region_cap_300",
    ]

    power = packet["power_ladder"]
    assert power["steps_first"] == 150
    assert power["steps_optional_continuation"] == 300
    assert power["max_steps_hard"] == 300
    floor = power["floor"]
    assert floor["strict_exact_floor"]["non_inverted_only"] is True
    assert floor["strict_exact_floor"]["count"] == 10
    assert floor["strict_exact_floor"]["total"] == 90
    assert floor["paired_loss_floor"]["bootstrap_ci"] == "95%"
    assert floor["paired_loss_floor"]["ci_must_exclude_zero"] is True
    assert floor["classifications"]["no_floor"] == BRANCH_MEASUREMENT_UNDERPOWERED
    assert floor["classifications"]["strict_exact_floor"] == BRANCH_MEASUREMENT_POWERED
    assert floor["classifications"]["favorable_paired_loss_ci"] == BRANCH_MEASUREMENT_LOSS_POWERED
    assert floor["classifications"]["strict_below_floor_and_only_b_minus_a0_loss_favors_a0"] == (
        BRANCH_POWERED_NEGATIVE_OR_LOSS_ONLY
    )
    assert "B-A0-loss-only" in floor["phase2_unlock_rule"]
    assert "no acquisition claim" in floor["phase2_unlock_rule"]

    trust_region = packet["trust_region"]
    assert trust_region["variable"] == "max_abs_per_tensor"
    assert trust_region["baseline_value"] == STEP3_BASELINE_MAX_ABS_PER_TENSOR
    assert trust_region["cap_value"] == STEP3_CAP_MAX_ABS_PER_TENSOR
    assert trust_region["fraction_per_tensor"] == 1.0
    assert trust_region["global_cap_contract"] == "off"
    assert trust_region["global_cap_deferred"] is True
    cap = trust_region["effective_cap"]
    assert cap["baseline_max_abs_per_tensor"] == 4096
    assert cap["cap_max_abs_per_tensor"] == 1024
    assert cap["fraction_per_tensor"] == 1.0
    assert cap["tensor_count_reduced"] == 1
    assert cap["total_allowed_flips_baseline"] == 4096
    assert cap["total_allowed_flips_cap"] == 1024
    assert cap["cap_effective"] is True
    assert cap["if_cap_effective_false"] == "cap_noop"

    arms = {arm["arm_id"]: arm for arm in packet["arms"]}
    assert arms[ARM_B_CAP_MAX_ABS_1024]["vote_law"] == "rank_free_sign_pressure"
    assert arms[ARM_B_CAP_MAX_ABS_1024]["tie_policy_id"] == TIE_POLICY_DETERMINISTIC_HASH_MATCHED
    assert arms[ARM_B_CAP_MAX_ABS_1024]["max_abs_per_tensor"] == 1024

    commands = packet["commands"]
    assert len(commands) == 18
    assert sum(1 for cmd in commands if cmd["phase_role"] == "measurement_power") == 8
    assert sum(1 for cmd in commands if cmd["phase_role"] == "trust_region_cap") == 10
    cap_commands = [cmd for cmd in commands if cmd["arm_id"] == ARM_B_CAP_MAX_ABS_1024]
    assert len(cap_commands) == 2
    for command in cap_commands:
        assert command["mode"] in {"trust_region_cap_150", "trust_region_cap_300"}
        assert command["science_arm"] == ARM_B_RANK_FREE_SIGN_PRESSURE
        assert command["max_abs_per_tensor"] == 1024
        assert command["argv"][command["argv"].index("--science-arm") + 1] == ARM_B_RANK_FREE_SIGN_PRESSURE
        assert command["argv"][command["argv"].index("--max-abs-per-tensor") + 1] == "1024"
    for command in commands:
        assert command["argv"][command["argv"].index("--max-steps-hard") + 1] == "300"
        assert command["global_cap_contract"] == "off"
        assert command["fraction_per_tensor"] == 1.0


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda packet: packet["trust_region"]["effective_cap"].update({"cap_effective": False}),
            "cap_effective",
        ),
        (
            lambda packet: packet["trust_region"]["effective_cap"].update({"tensor_count_reduced": 0}),
            "tensor_count_reduced",
        ),
        (
            lambda packet: packet["trust_region"]["effective_cap"].update(
                {"total_allowed_flips_cap": 4096},
            ),
            "total_allowed_flips_cap",
        ),
        (
            lambda packet: packet["commands"][0]["argv"].__setitem__(
                packet["commands"][0]["argv"].index("--max-abs-per-tensor") + 1,
                "1024",
            ),
            "--max-abs-per-tensor",
        ),
    ],
)
def test_step3_validator_rejects_ineffective_cap_or_mismatched_command(mutation, error):
    packet = build_measurement_power_then_trust_region_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-step3",
    )
    mutation(packet)

    with pytest.raises(ValueError, match=error):
        validate_measurement_power_then_trust_region_packet(packet)


def test_step3_power_floor_classifier_blocks_b_a0_loss_only_negative_unlock():
    assert classify_step3_power_floor(
        non_inverted_strict_exact_counts={
            ARM_A0_RANK_BUCKET_CURRENT: 9,
            ARM_A1_RANK_BUCKET_ORDER_MATCHED: 4,
            ARM_B_RANK_FREE_SIGN_PRESSURE: 3,
        },
        paired_loss_ci_excludes_zero={"B_minus_A0": True},
        paired_loss_winner={"B_minus_A0": "A0"},
    ) == BRANCH_POWERED_NEGATIVE_OR_LOSS_ONLY
    assert classify_step3_power_floor(
        non_inverted_strict_exact_counts=[9, 4, 3],
        paired_loss_ci_excludes_zero=False,
    ) == BRANCH_MEASUREMENT_UNDERPOWERED
    assert classify_step3_power_floor(
        non_inverted_strict_exact_counts=[10, 4, 3],
        paired_loss_ci_excludes_zero=False,
    ) == BRANCH_MEASUREMENT_POWERED
    assert classify_step3_power_floor(
        non_inverted_strict_exact_counts=[9, 4, 3],
        paired_loss_ci_excludes_zero={"A1_minus_B": True},
        paired_loss_winner={"A1_minus_B": "B"},
    ) == BRANCH_MEASUREMENT_LOSS_POWERED


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda packet: packet["power_ladder"]["floor"]["strict_exact_floor"].update({"count": 9}),
            "strict_exact floor count",
        ),
        (
            lambda packet: packet["power_ladder"]["floor"]["classifications"].pop(
                "strict_below_floor_and_only_b_minus_a0_loss_favors_a0",
            ),
            "powered_negative_or_loss_only",
        ),
        (
            lambda packet: packet["power_ladder"]["floor"].update({"phase2_unlock_rule": "loss CI clears"}),
            "B-A0-loss-only acquisition-capable phase2",
        ),
    ],
)
def test_step3_validator_rejects_power_floor_drift(mutation, error):
    packet = build_measurement_power_then_trust_region_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-step3",
    )
    mutation(packet)

    with pytest.raises(ValueError, match=error):
        validate_measurement_power_then_trust_region_packet(packet)


def test_step4_rank_signal_packet_adds_c_only_for_step4_and_keeps_legacy_stable():
    step1 = build_optimizer_update_law_science_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
    )
    step3 = build_measurement_power_then_trust_region_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-step3",
    )
    step4 = build_powered_rank_signal_decomposition_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-step4",
    )

    assert ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER not in {
        arm["arm_id"] for arm in step1["arms"]
    }
    assert ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER not in {
        arm["arm_id"] for arm in step3["arms"]
    }
    validate_powered_rank_signal_decomposition_packet(step4)
    assert step4["packet_kind"] == STEP4_POWERED_RANK_SIGNAL_DECOMPOSITION_PACKET_KIND
    assert step4["author_only"] is True
    assert step4["gpu_launched"] is False
    assert step4["launch_gate_id"] is None
    assert step4["pt_mutated"] is False
    assert step4["checkpoint_written"] is False
    assert step4["readiness_claim"] is False
    assert step4["full_sub2_claim"] is False
    assert step4["ready_for_main_science"] is False
    assert step4["branch_result"] is None
    assert step4["optimizer_credit_state_science_dependent"] is True

    arms = {arm["arm_id"]: arm for arm in step4["arms"]}
    assert set(arms) == {
        ARM_A0_RANK_BUCKET_CURRENT,
        ARM_A1_RANK_BUCKET_ORDER_MATCHED,
        ARM_B_RANK_FREE_SIGN_PRESSURE,
        ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER,
        ARM_INVERTED_SIGN_PRESSURE,
    }
    c_arm = arms[ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER]
    assert c_arm["vote_law"] == "rank_free_sign_pressure"
    assert c_arm["tie_policy_id"] == TIE_POLICY_CURRENT_MARGIN_INDEX
    assert "not pure current-order rank" in c_arm["claim_caveat"]

    assert step4["mode_sequence"] == ["rank_signal_150", "rank_signal_300"]
    assert step4["power_ladder"]["steps_first"] == 150
    assert step4["power_ladder"]["steps_optional_continuation"] == 300
    assert "clear misses stop at 150" in step4["power_ladder"]["continuation_enabled_if"]
    assert step4["match_to_A0_rule"]["strict_gap_max"] == 3
    assert step4["match_to_A0_rule"]["carrier_named_only_on_match_to_A0"] is True
    assert step4["mass_confound_rule"]["ratio_min_inclusive"] == 0.75
    assert step4["mass_confound_rule"]["ratio_max_inclusive"] == 1.25
    assert step4["mass_confound_rule"]["absolute_delta_min_inclusive"] == 4.0
    assert step4["success_boundary"]["C_claim"] == BRANCH_CURRENT_QACC_MARGIN_ORDER_BUNDLE_CARRIER
    assert step4["success_boundary"]["C_mass_confounded_branch"] == (
        BRANCH_MASS_CONFOUNDED_CURRENT_ORDER_SIGNAL
    )
    assert "margin-vs-index split deferred" in step4["success_boundary"]["C_claim_caveat"]

    commands = step4["commands"]
    assert len(commands) == 10
    assert {(cmd["mode"], cmd["arm_id"]) for cmd in commands} == {
        (phase, arm)
        for phase in {"rank_signal_150", "rank_signal_300"}
        for arm in set(arms)
    }
    c_commands = [
        command for command in commands
        if command["arm_id"] == ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER
    ]
    assert len(c_commands) == 2
    for command in c_commands:
        assert command["science_arm"] == ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER
        assert command["argv"][command["argv"].index("--science-arm") + 1] == (
            ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER
        )
        assert command["argv"][command["argv"].index("--max-abs-per-tensor") + 1] == "4096"
        assert command["global_cap_contract"] == "off"


def test_step4_match_to_a0_and_mass_confounded_classification_are_quantitative():
    assert step4_arm_matches_a0(
        arm_strict_exact_count=18,
        a0_strict_exact_count=21,
        paired_loss_ci_low=-0.05,
        paired_loss_ci_high=0.08,
    )
    assert step4_arm_matches_a0(
        arm_strict_exact_count=20,
        a0_strict_exact_count=21,
        paired_loss_ci_low=-0.20,
        paired_loss_ci_high=-0.01,
    )
    assert not step4_arm_matches_a0(
        arm_strict_exact_count=17,
        a0_strict_exact_count=21,
        paired_loss_ci_low=-0.05,
        paired_loss_ci_high=0.08,
    )
    assert not step4_arm_matches_a0(
        arm_strict_exact_count=20,
        a0_strict_exact_count=21,
        paired_loss_ci_low=0.01,
        paired_loss_ci_high=0.20,
    )

    reference = {
        "q_changed_count": 100,
        "candidate_count": 200,
        "pre_veto_selected_count": 100,
        "applied_count": 90,
        "vote_nonzero_count": 1000,
        "vote_abs_median": 2,
        "vote_abs_max": 4,
    }
    close_candidate = {
        "q_changed_count": 105,
        "candidate_count": 205,
        "pre_veto_selected_count": 100,
        "applied_count": 91,
        "vote_nonzero_count": 1005,
        "vote_abs_median": 2,
        "vote_abs_max": 4,
    }
    mass_shifted_candidate = {
        **close_candidate,
        "q_changed_count": 140,
    }
    assert step4_mass_confound_detected(reference=reference, candidate=close_candidate) is False
    assert step4_mass_confound_detected(reference=reference, candidate=mass_shifted_candidate) is True
    with pytest.raises(ValueError, match="vote_abs_max"):
        step4_mass_confound_detected(
            reference=reference,
            candidate={k: v for k, v in close_candidate.items() if k != "vote_abs_max"},
            rule=default_step4_mass_confound_rule(),
        )

    assert classify_step4_rank_signal_decomposition(
        c_matches_a0=True,
        c_mass_confounded=True,
        any_non_reference_matches_a0=True,
    ) == BRANCH_MASS_CONFOUNDED_CURRENT_ORDER_SIGNAL
    assert classify_step4_rank_signal_decomposition(
        c_matches_a0=True,
        c_mass_confounded=False,
        any_non_reference_matches_a0=True,
    ) == BRANCH_CURRENT_QACC_MARGIN_ORDER_BUNDLE_CARRIER
    assert classify_step4_rank_signal_decomposition(
        c_matches_a0=False,
        c_mass_confounded=False,
        a1_matches_a0=True,
        any_non_reference_matches_a0=True,
    ) == BRANCH_CURRENT_ORDER_NOT_NECESSARY
    assert classify_step4_rank_signal_decomposition(
        c_matches_a0=False,
        c_mass_confounded=False,
        a0_beats_c=True,
        a1_beats_b=False,
        any_non_reference_matches_a0=False,
    ) == BRANCH_RANK_MAGNITUDE_CONDITIONED_ON_CURRENT_ORDER
    assert classify_step4_rank_signal_decomposition(
        c_matches_a0=False,
        c_mass_confounded=False,
        any_non_reference_matches_a0=False,
    ) == BRANCH_NO_MATCH_DIFFERENT_CREDIT_SOURCE
    assert classify_step4_rank_signal_decomposition(
        c_matches_a0=False,
        c_mass_confounded=False,
        any_non_reference_matches_a0=True,
    ) == BRANCH_PARTIAL_LOCAL_SIGNAL


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda packet: packet["arms"].pop(ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER),
            "exactly A0/A1/B/C/inverted",
        ),
        (
            lambda packet: packet["arms"][ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER].update(
                {"claim_caveat": "pure current-order rank"},
            ),
            "pure current-order rank",
        ),
        (
            lambda packet: packet["match_to_A0_rule"].update({"strict_gap_max": 4}),
            "strict_gap_max",
        ),
        (
            lambda packet: packet["mass_confound_rule"].update({"ratio_max_inclusive": 1.5}),
            "ratio_max",
        ),
        (
            lambda packet: packet.update({"ready_for_main_science": True}),
            "ready_for_main_science",
        ),
    ],
)
def test_step4_validator_rejects_fold_drift(mutation, error):
    packet = build_powered_rank_signal_decomposition_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-step4",
    )
    packet["arms"] = {arm["arm_id"]: arm for arm in packet["arms"]}
    mutation(packet)
    if isinstance(packet["arms"], dict):
        packet["arms"] = list(packet["arms"].values())

    with pytest.raises(ValueError, match=error):
        validate_powered_rank_signal_decomposition_packet(packet)


def test_step5_support_order_trajectory_packet_is_four_arm_150_only():
    packet = build_support_order_trajectory_robustness_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-step5",
    )

    validate_support_order_trajectory_robustness_packet(packet)
    assert packet["packet_kind"] == STEP5_SUPPORT_ORDER_TRAJECTORY_ROBUSTNESS_PACKET_KIND
    assert packet["author_only"] is True
    assert packet["gpu_launched"] is False
    assert packet["launch_gate_id"] is None
    assert packet["pt_mutated"] is False
    assert packet["checkpoint_written"] is False
    assert packet["readiness_claim"] is False
    assert packet["full_sub2_claim"] is False
    assert packet["ready_for_main_science"] is False
    assert packet["qacc_kernelized"] is False
    assert "CPU-reference/default-off" in packet["qacc_cpu_reference_caveat"]
    assert packet["support_order_seed"] == STEP5_SUPPORT_ORDER_SEED
    assert packet["curriculum_seed"] == 17

    arms = {arm["arm_id"]: arm for arm in packet["arms"]}
    assert set(arms) == {
        ARM_A0_RANK_BUCKET_CURRENT,
        ARM_B_RANK_FREE_SIGN_PRESSURE,
        ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER,
        ARM_INVERTED_SIGN_PRESSURE,
    }
    assert ARM_A1_RANK_BUCKET_ORDER_MATCHED not in arms
    assert packet["mode_sequence"] == ["rank_signal_150"]
    assert packet["power_ladder"]["steps_first"] == 150
    assert packet["power_ladder"]["steps_optional_continuation"] is None
    assert packet["power_ladder"]["max_steps_hard"] == 150
    assert packet["pass_rule"]["C_strict_floor_count"] == 10
    assert packet["pass_rule"]["C_margin_over_max_A0_B_count"] == 5
    assert set(packet["pass_rule"]["paired_loss_required"]["comparisons"]) == {
        "C_minus_A0",
        "C_minus_B",
    }
    assert packet["support_order_proof_contract"]["support_content_unchanged_basis"] == (
        "order_invariant_multiset_hash16"
    )
    assert packet["support_order_proof_contract"]["ordered_support_content_hash16_is_invariant"] is False

    commands = packet["commands"]
    assert len(commands) == 4
    assert {(command["mode"], command["arm_id"]) for command in commands} == {
        ("rank_signal_150", ARM_A0_RANK_BUCKET_CURRENT),
        ("rank_signal_150", ARM_B_RANK_FREE_SIGN_PRESSURE),
        ("rank_signal_150", ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER),
        ("rank_signal_150", ARM_INVERTED_SIGN_PRESSURE),
    }
    for command in commands:
        argv = command["argv"]
        assert command["steps_requested"] == 150
        assert command["support_order_seed"] == 29
        assert command["curriculum_seed"] == 17
        assert command["qacc_kernelized"] is False
        assert "--support-order-seed" in argv
        assert argv[argv.index("--support-order-seed") + 1] == "29"
        assert argv[argv.index("--curriculum-seed") + 1] == "17"
        assert argv[argv.index("--steps") + 1] == "150"
        assert argv[argv.index("--audit-interval") + 1] == "150"
        assert argv[argv.index("--max-steps-hard") + 1] == "150"
        assert ARM_A1_RANK_BUCKET_ORDER_MATCHED not in argv
        assert "rank_signal_300" not in argv


def test_step5_support_order_hash_proof_uses_invariant_not_ordered_hash():
    original = [
        {"metadata": {"row_ids": ["0:aaa"], "batch_content_hash16": "aaa", "batch_index": 0}},
        {"metadata": {"row_ids": ["1:bbb"], "batch_content_hash16": "bbb", "batch_index": 1}},
        {"metadata": {"row_ids": ["2:ccc"], "batch_content_hash16": "ccc", "batch_index": 2}},
    ]
    permuted = [original[2], original[0], original[1]]

    proof = build_support_order_trajectory_proof(
        original,
        permuted,
        support_order_seed=STEP5_SUPPORT_ORDER_SEED,
    )

    assert proof["support_order_seed"] == STEP5_SUPPORT_ORDER_SEED
    assert proof["support_order_permutation_enabled"] is True
    assert proof["support_order_changed"] is True
    assert proof["support_content_unchanged"] is True
    assert proof["support_content_unchanged_basis"] == "order_invariant_multiset_hash16"
    assert proof["support_order_original_ordered_traversal_hash16"] != (
        proof["support_order_permuted_ordered_traversal_hash16"]
    )
    assert proof["support_order_original_invariant_multiset_hash16"] == (
        proof["support_order_permuted_invariant_multiset_hash16"]
    )
    assert proof["ordered_support_content_hash16_is_invariant"] is False


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda packet: packet["arms"].append(
                {
                    "arm_id": ARM_A1_RANK_BUCKET_ORDER_MATCHED,
                    "vote_law": "current_rank_bucket",
                    "tie_policy_id": TIE_POLICY_DETERMINISTIC_HASH_MATCHED,
                    "required": True,
                },
            ),
            "no A1",
        ),
        (
            lambda packet: packet["commands"].append(
                {
                    **packet["commands"][0],
                    "mode": "rank_signal_300",
                    "n_rows": 300,
                    "steps_requested": 300,
                },
            ),
            "four 150-only commands",
        ),
        (
            lambda packet: packet["support_order_proof_contract"].update(
                {"support_content_unchanged_basis": "support_content_hash16"},
            ),
            "order-invariant",
        ),
        (
            lambda packet: packet["support_order_proof_contract"].update(
                {"ordered_support_content_hash16_is_invariant": True},
            ),
            "ordered support_content_hash16",
        ),
        (
            lambda packet: packet.update({"ready_for_main_science": True}),
            "ready_for_main_science",
        ),
        (
            lambda packet: packet.update({"full_sub2_claim": True}),
            "full_sub2",
        ),
        (
            lambda packet: packet.update({"qacc_kernelized": True}),
            "qacc_kernelized",
        ),
        (
            lambda packet: packet.update({"raw_per_proposal_arrays_included": True}),
            "raw per-proposal arrays",
        ),
    ],
)
def test_step5_validator_rejects_false_invariant_and_scope_drift(mutation, error):
    packet = build_support_order_trajectory_robustness_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-step5",
    )
    mutation(packet)

    with pytest.raises(ValueError, match=error):
        validate_support_order_trajectory_robustness_packet(packet)


def test_step6_order_averaged_packet_is_fresh_all_9_author_only():
    packet = build_order_averaged_a0_component_decomposition_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-step6",
    )

    validate_order_averaged_a0_component_decomposition_packet(packet)
    assert packet["packet_kind"] == STEP6_ORDER_AVERAGED_A0_COMPONENT_DECOMPOSITION_PACKET_KIND
    assert packet["author_only"] is True
    assert packet["gpu_launched"] is False
    assert packet["launch_gate_id"] is None
    assert packet["pt_mutated"] is False
    assert packet["checkpoint_written"] is False
    assert packet["readiness_claim"] is False
    assert packet["full_sub2_claim"] is False
    assert packet["ready_for_main_science"] is False
    assert packet["qacc_kernelized"] is False
    assert "CPU-reference/default-off" in packet["qacc_cpu_reference_caveat"]
    assert packet["support_order_seeds"] == list(STEP6_SUPPORT_ORDER_SEEDS)
    assert packet["support_order_proof_contract"]["fixed_preregistered_new_seed"] == (
        STEP6_FIXED_PREREG_NEW_SEED
    )
    assert packet["support_order_proof_contract"]["post_hoc_seed_selection_allowed"] is False

    arms = {arm["arm_id"]: arm for arm in packet["arms"]}
    assert set(arms) == {
        ARM_A0_RANK_BUCKET_CURRENT,
        ARM_A1_RANK_BUCKET_ORDER_MATCHED,
        ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER,
    }
    assert ARM_B_RANK_FREE_SIGN_PRESSURE not in arms
    assert ARM_INVERTED_SIGN_PRESSURE not in arms

    assert packet["mode_sequence"] == ["rank_signal_150"]
    assert packet["power_ladder"]["steps_first"] == 150
    assert packet["power_ladder"]["steps_optional_continuation"] is None
    assert packet["power_ladder"]["max_steps_hard"] == 150
    assert packet["cost_ceiling"]["max_arm_runs"] == 9
    assert packet["cost_ceiling"]["max_gpu_hours"] == 2.0

    stability = packet["order_averaged_stability_rule"]
    assert stability["primary_evidence"] == "seed_level_dominance"
    assert stability["positive_classification"] == BRANCH_A0_COMPONENT_ORDER_ROBUST
    assert stability["negative_or_unstable_classification"] == BRANCH_MEASUREMENT_ORDER_SENSITIVE
    assert stability["min_seeds_dominating"] == 2
    assert stability["pooled_loss_cannot_override_seed_level_instability"] is True
    assert stability["no_carrier_readiness_or_full_sub2_claim"] is True
    assert set(packet["mass_confound_rule"]["compares"]) == {"A0_vs_A1", "A0_vs_C"}
    assert set(packet["terminal_criteria"]["mass_confound_rule"]["compares"]) == {
        "A0_vs_A1",
        "A0_vs_C",
    }

    context = packet["context_only_prior_receipts"]
    assert context
    assert all(entry["context_only"] is True for entry in context)
    assert all(entry["classifier_evidence"] is False for entry in context)

    commands = packet["commands"]
    assert len(commands) == 9
    assert {
        (command["support_order_seed"], command["arm_id"])
        for command in commands
    } == {
        (seed, arm)
        for seed in (None, 29, 43)
        for arm in {
            ARM_A0_RANK_BUCKET_CURRENT,
            ARM_A1_RANK_BUCKET_ORDER_MATCHED,
            ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER,
        }
    }
    for command in commands:
        argv = command["argv"]
        assert command["steps_requested"] == 150
        assert command["curriculum_seed"] == 17
        assert command["qacc_kernelized"] is False
        assert command["fresh_step6_evidence"] is True
        assert command["context_only"] is False
        assert command["classifier_evidence"] is True
        assert argv[argv.index("--curriculum-seed") + 1] == "17"
        assert argv[argv.index("--steps") + 1] == "150"
        assert argv[argv.index("--audit-interval") + 1] == "150"
        assert argv[argv.index("--max-steps-hard") + 1] == "150"
        assert "rank_signal_300" not in argv
        assert ARM_B_RANK_FREE_SIGN_PRESSURE not in argv
        assert ARM_INVERTED_SIGN_PRESSURE not in argv
        if command["support_order_seed"] is None:
            assert command["seed_label"] == "original"
            assert command["support_order_permutation_required"] is False
            assert "--support-order-seed" not in argv
        else:
            assert command["support_order_seed"] in {29, 43}
            assert command["support_order_permutation_required"] is True
            assert argv[argv.index("--support-order-seed") + 1] == str(
                command["support_order_seed"],
            )


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda packet: packet["arms"].append(
                {
                    "arm_id": ARM_B_RANK_FREE_SIGN_PRESSURE,
                    "vote_law": "rank_free_sign_pressure",
                    "tie_policy_id": TIE_POLICY_DETERMINISTIC_HASH_MATCHED,
                    "required": True,
                },
            ),
            "exactly A0/A1/C",
        ),
        (
            lambda packet: packet["arms"].append(
                {
                    "arm_id": ARM_INVERTED_SIGN_PRESSURE,
                    "vote_law": "inverted_rank_free_sign_pressure",
                    "tie_policy_id": TIE_POLICY_DETERMINISTIC_HASH_MATCHED,
                    "required": False,
                },
            ),
            "exactly A0/A1/C",
        ),
        (
            lambda packet: packet["commands"].append({**packet["commands"][0]}),
            "exactly 9 commands",
        ),
        (
            lambda packet: packet["commands"][0].update({"mode": "rank_signal_300"}),
            "rank_signal_150 only",
        ),
        (
            lambda packet: packet["commands"][0]["argv"].extend(["--support-order-seed", "17"]),
            "original trajectory argv must omit",
        ),
        (
            lambda packet: packet["commands"][3]["argv"].remove("--support-order-seed"),
            "seeded trajectory argv must include",
        ),
        (
            lambda packet: packet["commands"][6].update({"support_order_seed": 44}),
            "null, 29, 43",
        ),
        (
            lambda packet: packet["support_order_proof_contract"].update(
                {"fixed_preregistered_new_seed": 44},
            ),
            "seed43",
        ),
        (
            lambda packet: packet["support_order_proof_contract"].update(
                {"post_hoc_seed_selection_allowed": True},
            ),
            "post-hoc",
        ),
        (
            lambda packet: packet["context_only_prior_receipts"][0].update(
                {"classifier_evidence": True},
            ),
            "classifier evidence",
        ),
        (
            lambda packet: packet.update({"context_only_prior_receipts": []}),
            "exactly Step-4 and Step-5 context",
        ),
        (
            lambda packet: packet.pop("context_only_prior_receipts"),
            "exactly Step-4 and Step-5 context",
        ),
        (
            lambda packet: packet["context_only_prior_receipts"][0].update(
                {"label": "step4_reused_classifier_evidence"},
            ),
            "labels must be Step-4 and Step-5",
        ),
        (
            lambda packet: packet["mass_confound_rule"].update(
                {"compares": ["C_vs_A0", "C_vs_B"]},
            ),
            "A0 vs A1 and A0 vs C",
        ),
        (
            lambda packet: packet["order_averaged_stability_rule"].update(
                {"primary_evidence": "pooled_loss"},
            ),
            "seed-level dominance",
        ),
        (
            lambda packet: packet["order_averaged_stability_rule"].update(
                {"pooled_loss_cannot_override_seed_level_instability": False},
            ),
            "pooled loss",
        ),
        (
            lambda packet: packet.update({"ready_for_main_science": True}),
            "ready_for_main_science",
        ),
        (
            lambda packet: packet.update({"full_sub2_claim": True}),
            "full_sub2",
        ),
        (
            lambda packet: packet.update({"qacc_kernelized": True}),
            "qacc_kernelized",
        ),
        (
            lambda packet: packet.update({"raw_per_proposal_arrays_included": True}),
            "raw per-proposal arrays",
        ),
    ],
)
def test_step6_validator_rejects_scope_drift_and_reuse_evidence(mutation, error):
    packet = build_order_averaged_a0_component_decomposition_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-step6",
    )
    mutation(packet)

    with pytest.raises(ValueError, match=error):
        validate_order_averaged_a0_component_decomposition_packet(packet)


def test_oracle_screen_packet_declares_three_arms_and_hard_non_persistence():
    packet = build_candidate_set_viability_oracle_screen_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
    )

    validate_candidate_set_viability_oracle_screen_packet(packet)
    assert packet["packet_kind"] == ORACLE_SCREEN_PACKET_KIND
    assert packet["diagnostic_class"] == "pre_full_stack_diagnostic"
    assert packet["author_only"] is True
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["launch_gate_id"] is None
    assert packet["pt_mutated"] is False
    assert packet["checkpoint_written"] is False
    assert packet["readiness_claim"] is False
    assert packet["full_sub2_claim"] is False
    assert packet["ready_for_main_science"] is False
    assert packet["carrier_claim"] is False
    assert packet["optimizer_credit_state_row_flip"] is False
    assert packet["optimizer_credit_state_science_dependent"] is True
    assert packet["same_candidate_set_required"] is True

    arms = {arm["arm_id"]: arm for arm in packet["arms"]}
    assert set(arms) == {
        ORACLE_ARM_CURRENT_CREDIT_RANK_BUCKET_CURRENT_ORDER,
        ORACLE_ARM_DETERMINISTIC_HASH_SAME_VOTES,
        ORACLE_ARM_DIAGNOSTIC_LOCAL_LOSS_DELTA,
    }
    assert {arm["candidate_set"] for arm in arms.values()} == {
        "same_projected_move_candidate_set",
    }
    assert arms[ORACLE_ARM_DIAGNOSTIC_LOCAL_LOSS_DELTA]["vote_source"] == (
        "diagnostic_local_loss_delta"
    )
    assert arms[ORACLE_ARM_DIAGNOSTIC_LOCAL_LOSS_DELTA]["q_persisted"] is False
    assert arms[ORACLE_ARM_DIAGNOSTIC_LOCAL_LOSS_DELTA]["learner_teacher_promotion"] is False
    assert arms[ORACLE_ARM_DIAGNOSTIC_LOCAL_LOSS_DELTA]["checkpoint_promotional"] is False

    non_persist = packet["oracle_non_persistence_contract"]
    assert non_persist["q_persist_allowed"] is False
    assert non_persist["oracle_state_survives_into_learner"] is False
    assert non_persist["learner_teacher_promotion_allowed"] is False
    assert non_persist["pt_writes_allowed"] is False

    budget = packet["oracle_feasibility_budget"]
    assert budget["probe_required_before_full_screen"] is True
    assert budget["budget_present"] is True
    assert budget["allowed_max_sampled_candidates"] == list(
        ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES
    )
    assert budget["max_sampled_candidates"] == 8
    assert budget["max_seconds_by_budget"] == {
        str(candidate_budget): seconds
        for candidate_budget, seconds in ORACLE_SCREEN_MAX_SECONDS_BY_BUDGET.items()
    }
    assert budget["max_seconds"] == oracle_screen_budget_max_seconds(8)
    assert budget["classify_branch_on_missing_overrun_or_unsafe"] == (
        BRANCH_ORACLE_INFEASIBLE_OR_TOO_EXPENSIVE
    )

    summary = packet["compact_summary_schema"]
    assert summary["compact_summary_only"] is True
    assert set(summary["allowed_fields"]) == {
        "candidate_count",
        "sampled_candidate_count",
        "top_k",
        "sign_concordance",
        "credit_rank_deciles",
        "local_loss_delta_deciles",
        "paired_loss_branch_fields",
        "wider_screen_interpretation_inputs",
    }
    assert summary["raw_per_proposal_arrays"] is False


def test_oracle_screen_packet_pins_seed_order_contract_and_classifier_branches():
    packet = build_candidate_set_viability_oracle_screen_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
    )

    seed_contract = packet["seed_order_contract"]
    assert seed_contract["contrast_support_order_seeds"] == list(ORACLE_SCREEN_CONTRAST_SEEDS)
    assert seed_contract["contrast_seed_roles"]["seed43"] == "A0-bad contrast"
    assert seed_contract["contrast_seed_roles"]["seed29"] == "A0-good contrast"
    assert seed_contract["n20_screen_rows"] == 20
    assert seed_contract["n20_screen_is_launch_gated"] is True
    promotion = seed_contract["promotion_condition"]
    assert promotion["promote_to_n50_x_3_orderings"] is True
    assert promotion["promotion_rows"] == 50
    assert promotion["support_order_seeds"] == list(ORACLE_SCREEN_PROMOTION_ORDER_SEEDS)
    assert promotion["only_if_non_null"] is True
    assert promotion["only_if_not_artifact_confounded"] is True
    assert promotion["post_hoc_seed_selection_allowed"] is False

    classifier = packet["classifier_contract"]
    assert classifier["exactly_one_branch"] is True
    assert set(classifier["allowed_branches"]) == set(ORACLE_SCREEN_BRANCHES)

    interpretation = packet["wider_screen_interpretation_contract"]
    assert interpretation["runtime_branch_classification_semantics_frozen"] is True
    assert interpretation["max_sampled_candidates"] == 8
    assert interpretation["tier_max_seconds"] == oracle_screen_budget_max_seconds(8)
    assert interpretation["positive_interpretation_verdicts"] == list(
        ORACLE_WIDER_SCREEN_INTERPRETATION_VERDICTS
    )
    assert interpretation["negative_low_level_passthrough"] == [
        BRANCH_CANDIDATE_GENERATION_BAD_OR_NO_LOCAL_SIGNAL,
        BRANCH_ORACLE_INFEASIBLE_OR_TOO_EXPENSIVE,
    ]
    ok_band = interpretation["ranking_effectively_ok"]
    assert ok_band["oracle_best_current_rank_position_lt_rule"]["position_source"] == (
        "oracle_best_current_sampled_rank_position"
    )
    assert ok_band["oracle_best_current_rank_position_lt_rule"]["absolute_floor_positions"] == 5
    assert ok_band["oracle_best_current_rank_position_lt_examples"] == {
        str(candidate_budget): oracle_screen_effectively_ok_rank_position_exclusive_bound(
            candidate_budget
        )
        for candidate_budget in ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES
    }
    assert ok_band["current_vs_oracle_top1_gap_ratio_max_inclusive"] == 0.25
    bad_band = interpretation["credit_ranking_bad"]
    assert bad_band["rank_fraction_source"] == "oracle_best_current_sampled_rank_position"
    assert bad_band["oracle_best_current_rank_fraction_gt"] == 0.25
    assert bad_band["current_vs_oracle_top1_gap_ratio_gt"] == 0.50
    assert interpretation["next_branch_by_interpretation"] == {
        ORACLE_WIDER_SCREEN_VERDICT_RANKING_EFFECTIVELY_OK: (
            "ranking_not_the_bottleneck__reopen_scheduler_cap_backlog_multi_step"
        ),
        ORACLE_WIDER_SCREEN_VERDICT_RANKING_SUBOPTIMAL: (
            "credit_magnitude_or_rank_bin_calibration"
        ),
        ORACLE_WIDER_SCREEN_VERDICT_CREDIT_RANKING_BAD: (
            "update_law_or_credit_ranking_pivot"
        ),
    }


@pytest.mark.parametrize("budget", [32, 64])
def test_oracle_screen_launch_bundle_embeds_afbe598_contract_and_fixed_two_seed_commands(
    budget: int,
):
    packet = build_candidate_set_viability_oracle_screen_launch_bundle(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-oracle-screen",
        max_sampled_candidates=budget,
    )

    validate_candidate_set_viability_oracle_screen_launch_bundle(packet)
    assert packet["packet_kind"] == ORACLE_SCREEN_LAUNCH_BUNDLE_PACKET_KIND
    assert packet["science_contract_commit_sha"] == ORACLE_SCREEN_SCIENCE_CONTRACT_COMMIT_SHA
    assert packet["screen_rows"] == 20
    assert packet["curriculum_seed"] == 17
    assert packet["support_order_seeds"] == list(ORACLE_SCREEN_CONTRAST_SEEDS)
    assert packet["same_candidate_set_required"] is True
    assert packet["science_contract"]["packet_kind"] == ORACLE_SCREEN_PACKET_KIND
    assert packet["science_contract"]["parent_path"] == "parent.pt"
    assert packet["science_contract"]["parent_sha256"] == "abc123"
    assert packet["oracle_feasibility_budget"]["max_sampled_candidates"] == budget
    assert packet["oracle_feasibility_budget"]["max_seconds"] == oracle_screen_budget_max_seconds(
        budget
    )
    assert packet["wider_screen_interpretation_contract"]["max_sampled_candidates"] == budget
    assert len(packet["commands"]) == 2
    assert {command["support_order_seed"] for command in packet["commands"]} == {43, 29}
    assert {command["seed_label"] for command in packet["commands"]} == {"seed43", "seed29"}
    assert all(command["oracle_screen_mode"] == "candidate_set_viability" for command in packet["commands"])
    assert all(command["batch_size"] == 20 for command in packet["commands"])
    assert all(command["steps_requested"] == 1 for command in packet["commands"])
    assert all(command["max_sampled_candidates"] == budget for command in packet["commands"])
    assert all(
        command["oracle_max_seconds"] == oracle_screen_budget_max_seconds(budget)
        for command in packet["commands"]
    )
    assert all("--oracle-screen-mode" in command["argv"] for command in packet["commands"])
    assert all(
        "--oracle-screen-max-sampled-candidates" in command["argv"]
        for command in packet["commands"]
    )
    assert all(
        command["argv"][
            command["argv"].index("--oracle-screen-max-sampled-candidates") + 1
        ]
        == str(budget)
        for command in packet["commands"]
    )
    assert all("--science-arm" not in command["argv"] for command in packet["commands"])


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (
            {
                "oracle_feasible": False,
                "candidate_set_contains_ce_improving_move": True,
            },
            BRANCH_ORACLE_INFEASIBLE_OR_TOO_EXPENSIVE,
        ),
        (
            {
                "oracle_feasible": True,
                "candidate_set_contains_ce_improving_move": False,
            },
            BRANCH_CANDIDATE_GENERATION_BAD_OR_NO_LOCAL_SIGNAL,
        ),
        (
            {
                "oracle_feasible": True,
                "candidate_set_contains_ce_improving_move": True,
                "current_credit_rank_recovers_improvement": False,
                "deterministic_hash_recovers_improvement": True,
            },
            BRANCH_SCHEDULER_ONLY_ORDER_SENSITIVE,
        ),
        (
            {
                "oracle_feasible": True,
                "candidate_set_contains_ce_improving_move": True,
                "current_credit_rank_recovers_improvement": False,
                "credit_sign_concordance_positive": True,
            },
            BRANCH_CREDIT_MAGNITUDE_BAD_SIGN_USABLE,
        ),
        (
            {
                "oracle_feasible": True,
                "candidate_set_contains_ce_improving_move": True,
                "current_credit_rank_recovers_improvement": False,
                "oracle_advantage_over_current": True,
            },
            BRANCH_CANDIDATE_SET_VIABLE_CREDIT_RANKING_BAD,
        ),
    ],
)
def test_oracle_screen_classifier_returns_exactly_one_branch(kwargs, expected):
    assert classify_candidate_set_viability_oracle_screen(**kwargs) == expected
    assert expected in ORACLE_SCREEN_BRANCHES


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda packet: packet.update({"readiness_claim": True}),
            "readiness_claim",
        ),
        (
            lambda packet: packet.update({"full_sub2_claim": True}),
            "full_sub2_claim",
        ),
        (
            lambda packet: packet.update({"carrier_claim": True}),
            "carrier claim",
        ),
        (
            lambda packet: packet.update({"qacc_kernelized": True}),
            "qacc_kernelized",
        ),
        (
            lambda packet: packet["arms"][2].update({"q_persisted": True}),
            "packet.arms.2.q_persisted",
        ),
        (
            lambda packet: packet.update({"q_persisted": True}),
            "packet.q_persisted",
        ),
        (
            lambda packet: packet.update(
                {"nested_oracle_probe": {"q_persisted": True}},
            ),
            "packet.nested_oracle_probe.q_persisted",
        ),
        (
            lambda packet: packet["arms"][2].update({"learner_teacher_promotion": True}),
            "packet.arms.2.learner_teacher_promotion",
        ),
        (
            lambda packet: packet.update({"learner_teacher_promotion": True}),
            "packet.learner_teacher_promotion",
        ),
        (
            lambda packet: packet.update(
                {"diagnostic_oracle": {"learner_teacher_promotion": True}},
            ),
            "packet.diagnostic_oracle.learner_teacher_promotion",
        ),
        (
            lambda packet: packet.update({"checkpoint_promotional": True}),
            "packet.checkpoint_promotional",
        ),
        (
            lambda packet: packet.update(
                {"nested_oracle_probe": {"checkpoint_promotion_claim": True}},
            ),
            "packet.nested_oracle_probe.checkpoint_promotion_claim",
        ),
        (
            lambda packet: packet["oracle_non_persistence_contract"].update(
                {"oracle_state_survives_into_learner": True},
            ),
            "packet.oracle_non_persistence_contract.oracle_state_survives_into_learner",
        ),
        (
            lambda packet: packet["oracle_feasibility_budget"].pop("max_seconds"),
            "missing required fields",
        ),
        (
            lambda packet: packet["oracle_feasibility_budget"].update({"budget_present": False}),
            "budget must be present",
        ),
        (
            lambda packet: packet["oracle_feasibility_budget"].update(
                {"max_sampled_candidates": 16}
            ),
            "one of \\{8,32,64\\}",
        ),
        (
            lambda packet: packet["wider_screen_interpretation_contract"].update(
                {"max_sampled_candidates": 16}
            ),
            "one of \\{8,32,64\\}",
        ),
        (
            lambda packet: packet["compact_summary_schema"].update(
                {"raw_local_loss_deltas": True},
            ),
            "raw proposal arrays",
        ),
        (
            lambda packet: packet.update({"raw_per_proposal_arrays_included": True}),
            "raw per-proposal arrays",
        ),
        (
            lambda packet: packet["artifact_policy"].update(
                {"oracle_artifact_path": "/tmp/oracle.pt"},
            ),
            ".pt artifacts",
        ),
        (
            lambda packet: packet.update({"oracle_artifact_path": "/tmp/oracle.pt"}),
            "packet.oracle_artifact_path",
        ),
        (
            lambda packet: packet.update(
                {"extra_oracle_artifacts": {"oracle_artifact_path": "/tmp/oracle.pt"}},
            ),
            "packet.extra_oracle_artifacts.oracle_artifact_path",
        ),
        (
            lambda packet: packet["classifier_contract"].update({"allowed_branches": []}),
            "allowed branches",
        ),
        (
            lambda packet: packet["seed_order_contract"].update(
                {"contrast_support_order_seeds": [29, 43]},
            ),
            "contrast seeds",
        ),
        (
            lambda packet: packet["wider_screen_interpretation_contract"][
                "ranking_effectively_ok"
            ].update({"current_vs_oracle_top1_gap_ratio_max_inclusive": 0.2}),
            "gap-ratio ceiling",
        ),
    ],
)
def test_oracle_screen_validator_rejects_persistence_and_scope_drift(mutation, error):
    packet = build_candidate_set_viability_oracle_screen_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
    )
    mutation(packet)

    with pytest.raises(ValueError, match=error):
        validate_candidate_set_viability_oracle_screen_packet(packet)


def _remove_flag_with_value(argv: list[str], flag: str) -> None:
    index = argv.index(flag)
    del argv[index: index + 2]


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda packet: _remove_flag_with_value(
                packet["commands"][0]["argv"],
                "--oracle-screen-mode",
            ),
            "missing required probe launch arguments",
        ),
        (
            lambda packet: packet["commands"][0]["argv"].__setitem__(
                packet["commands"][0]["argv"].index("--oracle-screen-mode") + 1,
                "wrong_oracle_mode",
            ),
            "--oracle-screen-mode must be 'candidate_set_viability'",
        ),
        (
            lambda packet: packet["commands"][0]["argv"].extend(["--science-arm", ARM_A0_RANK_BUCKET_CURRENT]),
            "--oracle-screen-mode",
        ),
        (
            lambda packet: packet.update({"science_contract_commit_sha": "deadbeef"}),
            "afbe598",
        ),
        (
            lambda packet: packet["commands"][0].update({"batch_size": 19}),
            "batch_size must equal N=20",
        ),
        (
            lambda packet: packet["commands"][0].update({"max_sampled_candidates": 16}),
            "one of \\{8,32,64\\}",
        ),
    ],
)
def test_oracle_screen_launch_bundle_validator_rejects_scope_drift(mutation, error):
    packet = build_candidate_set_viability_oracle_screen_launch_bundle(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-oracle-screen",
    )
    mutation(packet)

    with pytest.raises(ValueError, match=error):
        validate_candidate_set_viability_oracle_screen_launch_bundle(packet)


def test_credit_ranking_pivot_packet_declares_compact_stage_a_and_bounded_stage_b():
    packet = build_credit_ranking_pivot_measurement_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
    )

    validate_credit_ranking_pivot_measurement_packet(packet)
    assert packet["packet_kind"] == CREDIT_RANKING_PIVOT_MEASUREMENT_PACKET_KIND
    assert packet["diagnostic_class"] == "pre_full_stack_diagnostic"
    assert packet["author_only"] is True
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["pt_mutated"] is False
    assert packet["same_candidate_set_required"] is True
    assert packet["oracle_feasibility_budget"]["max_sampled_candidates"] == 32
    assert packet["oracle_feasibility_budget"]["max_seconds"] == oracle_screen_budget_max_seconds(
        32
    )
    summary = packet["compact_summary_schema"]
    assert summary["compact_summary_only"] is True
    assert set(summary["allowed_fields"]) == {
        "candidate_count",
        "sampled_candidate_count",
        "sampled_candidate_table",
        "score_family_metrics",
        "stage_a_null_guard",
        "tie_band_ambiguity",
        "local_apply_magnitude_smoke",
        "telemetry",
    }
    contract = packet["measurement_contract"]
    assert contract["score_family"]["primary"] == "S_vote_margin"
    assert contract["score_family"]["hash_control_role"] == "null_distribution_only"
    assert contract["stage_a"]["non_predictive_branch_label"] == (
        BRANCH_PREREGISTERED_CHEAP_LEARNER_FEATURE_FAMILY_CANNOT_PREDICT_REGRET
    )
    assert contract["stage_a"]["oracle_best_sampled_rank_position_poor_fraction"] == 0.25
    assert contract["stage_a"]["oracle_best_sampled_rank_position_poor_threshold_rule"] == (
        "ceil(fraction * sampled_candidate_count)"
    )
    assert contract["stage_a"]["predictive_seed_label"] == "primary_score_predictive_for_local_regret"
    assert contract["tie_band_ambiguity_guard"]["ambiguous_if_regret_spread_ratio_gt"] == 0.25
    assert contract["tie_band_ambiguity_guard"]["ambiguous_branch_label"] == (
        BRANCH_MEASUREMENT_AMBIGUOUS_TIE_BAND_ALIASING
    )
    assert contract["stage_b_local_apply_magnitude_smoke"][
        "current_spec_is_non_definitive_without_live_full_cap"
    ] is True
    assert contract["stage_b_local_apply_magnitude_smoke"][
        "definitive_b_requires_follow_on"
    ] is True
    assert contract["allowed_seed_local_labels"] == [
        BRANCH_PREREGISTERED_CHEAP_LEARNER_FEATURE_FAMILY_CANNOT_PREDICT_REGRET,
        "primary_score_predictive_for_local_regret",
        BRANCH_MEASUREMENT_AMBIGUOUS_TIE_BAND_ALIASING,
        BRANCH_MEASUREMENT_AMBIGUOUS_NO_BRANCH,
    ]


def test_credit_ranking_pivot_launch_bundle_pins_budget32_and_fixed_two_seed_commands():
    packet = build_credit_ranking_pivot_measurement_launch_bundle(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-credit-ranking-pivot",
    )

    validate_credit_ranking_pivot_measurement_launch_bundle(packet)
    assert packet["packet_kind"] == CREDIT_RANKING_PIVOT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND
    assert packet["screen_rows"] == 20
    assert packet["curriculum_seed"] == 17
    assert packet["support_order_seeds"] == list(ORACLE_SCREEN_CONTRAST_SEEDS)
    assert packet["same_candidate_set_required"] is True
    assert packet["oracle_feasibility_budget"]["max_sampled_candidates"] == 32
    assert packet["oracle_feasibility_budget"]["max_seconds"] == oracle_screen_budget_max_seconds(
        32
    )
    assert packet["science_contract"]["packet_kind"] == CREDIT_RANKING_PIVOT_MEASUREMENT_PACKET_KIND
    assert packet["measurement_contract"] == packet["science_contract"]["measurement_contract"]
    assert packet["terminal_criteria"]["branch_classifier"] == [
        BRANCH_PREREGISTERED_CHEAP_LEARNER_FEATURE_FAMILY_CANNOT_PREDICT_REGRET,
        "primary_score_predictive_for_local_regret",
        BRANCH_MEASUREMENT_AMBIGUOUS_TIE_BAND_ALIASING,
        BRANCH_MEASUREMENT_AMBIGUOUS_NO_BRANCH,
    ]
    assert len(packet["commands"]) == 2
    assert {command["support_order_seed"] for command in packet["commands"]} == {43, 29}
    assert all(
        command["oracle_screen_mode"] == "credit_ranking_pivot_measurement"
        for command in packet["commands"]
    )
    assert all(command["max_sampled_candidates"] == 32 for command in packet["commands"])
    assert all(
        command["oracle_max_seconds"] == oracle_screen_budget_max_seconds(32)
        for command in packet["commands"]
    )


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda packet: packet.update({"same_candidate_set_required": False}),
            "same candidate set",
        ),
        (
            lambda packet: packet["oracle_feasibility_budget"].update(
                {"max_sampled_candidates": 8}
            ),
            "budget 32",
        ),
        (
            lambda packet: packet["compact_summary_schema"].update(
                {"raw_candidate_scores": True}
            ),
            "raw proposal arrays",
        ),
        (
            lambda packet: packet["artifact_policy"].update(
                {"compact_json_ndjson_only": False}
            ),
            "compact JSON/NDJSON",
        ),
        (
            lambda packet: packet["measurement_contract"][
                "stage_b_local_apply_magnitude_smoke"
            ].update({"current_spec_is_non_definitive_without_live_full_cap": False}),
            "non-definitive without live full cap",
        ),
        (
            lambda packet: packet["measurement_contract"][
                "stage_b_local_apply_magnitude_smoke"
            ].update({"definitive_b_requires_follow_on": False}),
            "require follow-on for definitive b",
        ),
    ],
)
def test_credit_ranking_pivot_packet_validator_rejects_fail_closed_scope_drift(
    mutation,
    error,
):
    packet = build_credit_ranking_pivot_measurement_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
    )
    mutation(packet)

    with pytest.raises(ValueError, match=error):
        validate_credit_ranking_pivot_measurement_packet(packet)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda packet: packet["commands"][0].update({"support_order_seed": 17}),
            "contrast seeds",
        ),
        (
            lambda packet: packet["commands"][0].update({"max_sampled_candidates": 64}),
            "must be 32",
        ),
        (
            lambda packet: packet["oracle_feasibility_budget"].update(
                {"max_sampled_candidates": 64}
            ),
            "budget 32",
        ),
        (
            lambda packet: packet["commands"][0]["argv"].__setitem__(
                packet["commands"][0]["argv"].index("--oracle-screen-mode") + 1,
                "candidate_set_viability",
            ),
            "credit_ranking_pivot_measurement",
        ),
    ],
)
def test_credit_ranking_pivot_launch_bundle_validator_rejects_seed_and_budget_drift(
    mutation,
    error,
):
    packet = build_credit_ranking_pivot_measurement_launch_bundle(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-credit-ranking-pivot",
    )
    mutation(packet)

    with pytest.raises(ValueError, match=error):
        validate_credit_ranking_pivot_measurement_launch_bundle(packet)


def test_activation_credit_packet_declares_device_resident_within_band_contract():
    packet = build_activation_credit_measurement_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
    )

    validate_activation_credit_measurement_packet(packet)
    assert packet["packet_kind"] == ACTIVATION_CREDIT_MEASUREMENT_PACKET_KIND
    assert packet["diagnostic_class"] == "pre_full_stack_diagnostic"
    assert packet["author_only"] is True
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["pt_mutated"] is False
    assert packet["eligible_scope"] == ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE
    assert packet["same_candidate_set_required"] is True
    assert packet["oracle_feasibility_budget"]["max_sampled_candidates"] == 32
    assert packet["oracle_feasibility_budget"]["max_seconds"] == oracle_screen_budget_max_seconds(
        32
    )
    assert packet["scale_smoke_required_before_full_eval"] is True
    assert packet["scale_smoke_launch_bundle_packet_kind"] == (
        ACTIVATION_CREDIT_SCALE_SMOKE_LAUNCH_BUNDLE_PACKET_KIND
    )
    assert packet["fresh_confirmation_seed_required_for_persistent_followup"] == (
        ACTIVATION_CREDIT_FRESH_CONFIRMATION_SEED
    )
    summary = packet["compact_summary_schema"]
    assert summary["compact_summary_only"] is True
    assert set(summary["allowed_fields"]) == {
        "candidate_count",
        "sampled_candidate_count",
        "sampled_candidate_table",
        "target_tie_band",
        "family_metrics",
        "telemetry",
    }
    contract = packet["measurement_contract"]
    assert contract["required_eligible_scope"] == ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE
    assert contract["activation_credit_source"]["capture_device_mode"] == "device_resident"
    assert contract["activation_credit_source"]["grad_proxy_compute_mode"] == (
        "candidate_only_gather"
    )
    assert contract["activation_credit_source"]["diag_fisher_reuses_grad_proxy_captures"] is True
    assert contract["activation_credit_source"]["second_backward_forbidden"] is True
    assert contract["activation_credit_source"]["no_extra_response_label_mask"] is True
    assert contract["activation_credit_source"]["policy_facing_fields"] == [
        ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_FIELD,
        ACTIVATION_CREDIT_SNR_Q5_FIELD,
        ACTIVATION_CREDIT_DIAG_FISHER_Q5_FIELD,
    ]
    assert contract["feature_construction"]["candidate_delta_weight_effective_weight_space"] is True
    assert contract["feature_construction"]["diag_fisher_surrogate_kind"] == (
        "empirical_fisher_gauss_newton_diagonal"
    )
    assert contract["feature_construction"]["second_order_snr_eps"] == (
        ACTIVATION_CREDIT_SECOND_ORDER_SNR_EPS
    )
    assert contract["feature_construction"]["q5_bin_count"] == (
        ACTIVATION_CREDIT_MAGNITUDE_Q5_BIN_COUNT
    )
    assert contract["feature_construction"]["q5_min_bucket_size"] == (
        ACTIVATION_CREDIT_MAGNITUDE_Q5_MIN_BUCKET_SIZE
    )
    assert contract["family_discriminator"]["primary"] == ACTIVATION_CREDIT_PRIMARY_FAMILY_ID
    assert contract["family_discriminator"]["fields_by_family_id"][
        ACTIVATION_CREDIT_PRIMARY_FAMILY_ID
    ] == [ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_FIELD]
    assert contract["family_discriminator"]["fields_by_family_id"][
        ACTIVATION_CREDIT_SNR_Q5_ABLATION_FAMILY_ID
    ] == [ACTIVATION_CREDIT_SNR_Q5_FIELD]
    assert contract["family_discriminator"]["fields_by_family_id"][
        ACTIVATION_CREDIT_DIAG_FISHER_Q5_ABLATION_FAMILY_ID
    ] == [ACTIVATION_CREDIT_DIAG_FISHER_Q5_FIELD]
    assert contract["scale_smoke_gate"]["required_max_sampled_candidates"] == 8
    assert contract["scale_smoke_gate"]["required_batch_size"] == 4
    assert contract["fresh_confirmation_gate"][
        "required_seed_before_persistent_followup"
    ] == ACTIVATION_CREDIT_FRESH_CONFIRMATION_SEED
    assert contract["allowed_seed_local_labels"] == list(ACTIVATION_CREDIT_BRANCHES)
    assert contract["fragmentation_audit"]["candidate_delta_weight_support_required"] is True
    assert contract["fragmentation_audit"]["q5_primary_prefix"] == "taylor_benefit_q5"
    assert contract["fragmentation_audit"]["q5_report_only_prefixes"] == [
        "snr_q5",
        "diagfisher_q5",
    ]
    assert contract["fragmentation_audit"]["q5_min_bucket_candidate_count_required"] == (
        ACTIVATION_CREDIT_MAGNITUDE_Q5_MIN_BUCKET_SIZE
    )
    assert contract["fragmentation_audit"]["q5_singleton_buckets_forbidden"] is True
    assert contract["fragmentation_audit"]["q5_ties_force_ambiguous"] is True


def test_activation_credit_launch_bundle_pins_budget32_and_fixed_two_seed_commands():
    packet = build_activation_credit_measurement_launch_bundle(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-activation-credit",
    )

    validate_activation_credit_measurement_launch_bundle(packet)
    assert packet["packet_kind"] == ACTIVATION_CREDIT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND
    assert packet["eligible_scope"] == ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE
    assert packet["screen_rows"] == 20
    assert packet["curriculum_seed"] == 17
    assert packet["support_order_seeds"] == list(ORACLE_SCREEN_CONTRAST_SEEDS)
    assert packet["same_candidate_set_required"] is True
    assert packet["scale_smoke_required_before_full_eval"] is True
    assert packet["oracle_feasibility_budget"]["max_sampled_candidates"] == 32
    assert packet["oracle_feasibility_budget"]["max_seconds"] == oracle_screen_budget_max_seconds(
        32
    )
    assert packet["science_contract"]["packet_kind"] == ACTIVATION_CREDIT_MEASUREMENT_PACKET_KIND
    assert packet["measurement_contract"] == packet["science_contract"]["measurement_contract"]
    assert packet["measurement_contract"]["within_band_decision"]["predictive_family_id"] == (
        ACTIVATION_CREDIT_PRIMARY_FAMILY_ID
    )
    assert packet["measurement_contract"]["feature_construction"]["second_order_snr_eps"] == (
        ACTIVATION_CREDIT_SECOND_ORDER_SNR_EPS
    )
    assert packet["terminal_criteria"]["branch_classifier"] == [
        BRANCH_ACTIVATION_CREDIT_CANDIDATE_SIGNAL,
        BRANCH_ACTIVATION_CREDIT_MISSING_SIGNAL_DEEPER_THAN_FIRST_ORDER_CREDIT_STORAGE,
        BRANCH_ACTIVATION_CREDIT_AMBIGUOUS_NO_BRANCH,
    ]
    assert packet["terminal_criteria"]["required_eligible_scope"] == (
        ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE
    )
    assert packet["terminal_criteria"][
        "fresh_confirmation_seed_required_for_persistent_followup"
    ] == ACTIVATION_CREDIT_FRESH_CONFIRMATION_SEED
    assert packet["terminal_criteria"]["topology_control_positive_forces_ambiguous"] is True
    assert {
        "control_parity_gate",
        "prior_null_setup_gate",
        "verdict_rule",
    }.isdisjoint(packet["terminal_criteria"])
    assert len(packet["commands"]) == 2
    assert {command["support_order_seed"] for command in packet["commands"]} == {43, 29}
    assert all(
        command["oracle_screen_mode"] == "activation_credit_measurement"
        for command in packet["commands"]
    )
    assert all(command["max_sampled_candidates"] == 32 for command in packet["commands"])
    assert all(command["batch_size"] == 20 for command in packet["commands"])
    assert all(
        command["eligible_scope"] == ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE
        for command in packet["commands"]
    )
    assert all(
        "--eligible-scope" in command["argv"]
        and command["argv"][command["argv"].index("--eligible-scope") + 1]
        == ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE
        for command in packet["commands"]
    )
    assert all(
        command["env"][ACTIVATION_CREDIT_STDOUT_PATH_ENV] == command["stdout_path"]
        and command["env"][ACTIVATION_CREDIT_STDERR_PATH_ENV] == command["stderr_path"]
        for command in packet["commands"]
    )


def test_activation_credit_scale_smoke_launch_bundle_pins_budget8_and_batch4_commands():
    packet = build_activation_credit_scale_smoke_launch_bundle(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-activation-credit-smoke",
    )

    validate_activation_credit_scale_smoke_launch_bundle(packet)
    assert packet["packet_kind"] == ACTIVATION_CREDIT_SCALE_SMOKE_LAUNCH_BUNDLE_PACKET_KIND
    assert packet["eligible_scope"] == ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE
    assert packet["screen_rows"] == ACTIVATION_CREDIT_SMOKE_BATCH_SIZE
    assert packet["curriculum_seed"] == 17
    assert packet["support_order_seeds"] == list(ORACLE_SCREEN_CONTRAST_SEEDS)
    assert packet["same_candidate_set_required"] is True
    assert packet["oracle_feasibility_budget"]["max_sampled_candidates"] == (
        ACTIVATION_CREDIT_SMOKE_MAX_SAMPLED_CANDIDATES
    )
    assert packet["oracle_feasibility_budget"]["max_seconds"] == oracle_screen_budget_max_seconds(
        ACTIVATION_CREDIT_SMOKE_MAX_SAMPLED_CANDIDATES
    )
    assert set(packet["compact_summary_schema"]["allowed_fields"]) == {
        "target_tie_band_id",
        "target_band_candidate_count",
        "grad_proxy_candidate_count",
        "magnitude_bin_threshold",
        "magnitude_bin_histogram",
        "magnitude_bin_degenerate",
        "singleton_magnitude_source_count",
        "sampled_target_band_rows",
    }
    assert packet["scale_smoke_contract"]["required_batch_size"] == ACTIVATION_CREDIT_SMOKE_BATCH_SIZE
    assert packet["terminal_criteria"]["branch_classifier"] is None
    assert packet["terminal_criteria"]["required_max_sampled_candidates"] == (
        ACTIVATION_CREDIT_SMOKE_MAX_SAMPLED_CANDIDATES
    )
    assert packet["terminal_criteria"]["required_batch_size"] == ACTIVATION_CREDIT_SMOKE_BATCH_SIZE
    assert {
        "control_parity_gate",
        "prior_null_setup_gate",
        "verdict_rule",
    }.isdisjoint(packet["terminal_criteria"])
    occupancy = packet["terminal_criteria"]["occupancy_outcome_contract"]
    assert occupancy == packet["scale_smoke_contract"]["occupancy_outcome_contract"]
    assert occupancy["per_seed_receipt_fields_required"] == [
        "target_band_candidate_count",
        "grad_proxy_candidate_count",
    ]
    assert occupancy["pass_requires_any_seed_positive_fields"] == [
        "target_band_candidate_count",
        "grad_proxy_candidate_count",
    ]
    assert occupancy["per_seed_target_band_zero_label"] == "occupancy_miss"
    assert occupancy["all_seeds_target_band_zero_outcome"] == (
        "inconclusive_on_gather_timing_only"
    )
    assert occupancy["all_seeds_target_band_zero_reprobe_budgets"] == [12, 16]
    assert occupancy["target_band_positive_grad_proxy_zero_outcome"] == (
        "smoke_failure_repair_signal"
    )
    assert len(packet["commands"]) == 2
    assert {command["support_order_seed"] for command in packet["commands"]} == {43, 29}
    assert all(
        command["oracle_screen_mode"] == "activation_credit_scale_smoke"
        for command in packet["commands"]
    )
    assert all(
        command["max_sampled_candidates"] == ACTIVATION_CREDIT_SMOKE_MAX_SAMPLED_CANDIDATES
        for command in packet["commands"]
    )
    assert all(command["batch_size"] == ACTIVATION_CREDIT_SMOKE_BATCH_SIZE for command in packet["commands"])
    assert all(
        "--eligible-scope" in command["argv"]
        and command["argv"][command["argv"].index("--eligible-scope") + 1]
        == ACTIVATION_CREDIT_REQUIRED_ELIGIBLE_SCOPE
        for command in packet["commands"]
    )
    assert all(
        command["env"][ACTIVATION_CREDIT_STDOUT_PATH_ENV] == command["stdout_path"]
        and command["env"][ACTIVATION_CREDIT_STDERR_PATH_ENV] == command["stderr_path"]
        for command in packet["commands"]
    )


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda packet: packet["measurement_contract"]["family_discriminator"][
                "fields_by_family_id"
            ].update({ACTIVATION_CREDIT_PRIMARY_FAMILY_ID: [
                ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_FIELD,
                ACTIVATION_CREDIT_SNR_Q5_FIELD,
            ]}),
            "second-order q5 primary family drifted",
        ),
        (
            lambda packet: packet["measurement_contract"]["fresh_confirmation_gate"].update(
                {"required_seed_before_persistent_followup": 43}
            ),
            "fresh confirmation seed",
        ),
        (
            lambda packet: packet.update({"eligible_scope": "all-bitlinear"}),
            "eligible_scope=first-bitlinear",
        ),
    ],
)
def test_activation_credit_packet_validator_rejects_primary_family_seed_and_scope_drift(
    mutation,
    error,
):
    packet = build_activation_credit_measurement_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
    )
    mutation(packet)

    with pytest.raises(ValueError, match=error):
        validate_activation_credit_measurement_packet(packet)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda packet: packet["measurement_contract"]["feature_construction"].update(
                {"q5_min_bucket_size": 4}
            ),
            "q5 min bucket size drifted",
        ),
        (
            lambda packet: packet["measurement_contract"]["fragmentation_audit"].update(
                {"q5_ties_force_ambiguous": False}
            ),
            "q5 tie guard drifted",
        ),
        (
            lambda packet: packet["measurement_contract"]["feature_construction"].update(
                {"second_order_snr_eps": 1e-8}
            ),
            "second-order snr eps drifted",
        ),
    ],
)
def test_activation_credit_packet_validator_rejects_q5_guard_drift(
    mutation,
    error,
):
    packet = build_activation_credit_measurement_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
    )
    mutation(packet)

    with pytest.raises(ValueError, match=error):
        validate_activation_credit_measurement_packet(packet)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda packet: packet["commands"][0].update({"max_sampled_candidates": 64}),
            "must pin budget 32|max_sampled_candidates drifted",
        ),
        (
            lambda packet: packet["commands"][0].update({"batch_size": 8}),
            "batch_size drifted",
        ),
        (
            lambda packet: packet["commands"][0]["argv"].__setitem__(
                packet["commands"][0]["argv"].index("--eligible-scope") + 1,
                "all-bitlinear",
            ),
            "eligible-scope|first-bitlinear",
        ),
    ],
)
def test_activation_credit_launch_bundle_validator_rejects_budget_batch_and_scope_drift(
    mutation,
    error,
):
    packet = build_activation_credit_measurement_launch_bundle(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-activation-credit",
    )
    mutation(packet)

    with pytest.raises(ValueError, match=error):
        validate_activation_credit_measurement_launch_bundle(packet)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda packet: packet["terminal_criteria"].update({"control_parity_gate": {}}),
            "must not contain control_parity_gate",
        ),
        (
            lambda packet: packet["terminal_criteria"].update({"prior_null_setup_gate": {}}),
            "must not contain prior_null_setup_gate",
        ),
        (
            lambda packet: packet["terminal_criteria"].update({"verdict_rule": {}}),
            "must not contain verdict_rule",
        ),
        (
            lambda packet: packet["commands"][0]["env"].update(
                {ACTIVATION_CREDIT_STDOUT_PATH_ENV: "/tmp/drifted-stdout.ndjson"}
            ),
            "env stdout path must match stdout_path",
        ),
        (
            lambda packet: packet["commands"][0]["env"].update(
                {ACTIVATION_CREDIT_STDERR_PATH_ENV: "/tmp/drifted-stderr.log"}
            ),
            "env stderr path must match stderr_path",
        ),
    ],
)
def test_activation_credit_launch_bundle_validator_rejects_stale_terminal_keys_and_log_path_drift(
    mutation,
    error,
):
    packet = build_activation_credit_measurement_launch_bundle(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-activation-credit",
    )
    mutation(packet)

    with pytest.raises(ValueError, match=error):
        validate_activation_credit_measurement_launch_bundle(packet)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda packet: packet["commands"][0].update({"max_sampled_candidates": 32}),
            "must pin budget 8|max_sampled_candidates drifted",
        ),
        (
            lambda packet: packet["commands"][0].update({"batch_size": 8}),
            "batch size 4|batch_size drifted",
        ),
        (
            lambda packet: packet["terminal_criteria"].update(
                {"branch_classifier": list(ACTIVATION_CREDIT_BRANCHES)}
            ),
            "branch_classifier must stay null",
        ),
    ],
)
def test_activation_credit_smoke_launch_bundle_validator_rejects_budget_batch_and_branch_drift(
    mutation,
    error,
):
    packet = build_activation_credit_scale_smoke_launch_bundle(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-activation-credit-smoke",
    )
    mutation(packet)

    with pytest.raises(ValueError, match=error):
        validate_activation_credit_scale_smoke_launch_bundle(packet)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda packet: packet["terminal_criteria"].update({"control_parity_gate": {}}),
            "must not contain control_parity_gate",
        ),
        (
            lambda packet: packet["terminal_criteria"].update({"prior_null_setup_gate": {}}),
            "must not contain prior_null_setup_gate",
        ),
        (
            lambda packet: packet["terminal_criteria"].update({"verdict_rule": {}}),
            "must not contain verdict_rule",
        ),
        (
            lambda packet: packet["terminal_criteria"]["occupancy_outcome_contract"].update(
                {"per_seed_target_band_zero_label": "soft_fail"}
            ),
            "occupancy_miss label drifted",
        ),
        (
            lambda packet: packet["terminal_criteria"]["occupancy_outcome_contract"].update(
                {"all_seeds_target_band_zero_reprobe_budgets": [32]}
            ),
            "re-smoke budgets \\[12, 16\\]",
        ),
    ],
)
def test_activation_credit_smoke_launch_bundle_validator_rejects_stale_terminal_verdict_keys_by_name_and_occupancy_drift(
    mutation,
    error,
):
    packet = build_activation_credit_scale_smoke_launch_bundle(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-activation-credit-smoke",
    )
    mutation(packet)

    with pytest.raises(ValueError, match=error):
        validate_activation_credit_scale_smoke_launch_bundle(packet)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda packet: packet["commands"][0].update(
                {
                    "env": {
                        "HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE": "1",
                        "HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH": "1",
                    }
                }
            ),
            "env stdout path must match stdout_path",
        ),
        (
            lambda packet: packet["commands"][0]["env"].update(
                {ACTIVATION_CREDIT_STDERR_PATH_ENV: "/tmp/drifted-stderr.log"}
            ),
            "env stderr path must match stderr_path",
        ),
    ],
)
def test_activation_credit_smoke_launch_bundle_validator_rejects_missing_or_drifted_log_path_env(
    mutation,
    error,
):
    packet = build_activation_credit_scale_smoke_launch_bundle(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-activation-credit-smoke",
    )
    mutation(packet)

    with pytest.raises(ValueError, match=error):
        validate_activation_credit_scale_smoke_launch_bundle(packet)


def test_within_tie_band_packet_declares_one_sided_null_guards_and_fragmentation_audit():
    packet = build_within_tie_band_discriminator_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
    )

    validate_within_tie_band_discriminator_packet(packet)
    assert packet["packet_kind"] == WITHIN_TIE_BAND_DISCRIMINATOR_PACKET_KIND
    assert packet["diagnostic_class"] == "pre_full_stack_diagnostic"
    assert packet["author_only"] is True
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["pt_mutated"] is False
    assert packet["same_candidate_set_required"] is True
    assert packet["oracle_feasibility_budget"]["max_sampled_candidates"] == 32
    assert packet["oracle_feasibility_budget"]["max_seconds"] == oracle_screen_budget_max_seconds(
        32
    )
    summary = packet["compact_summary_schema"]
    assert summary["compact_summary_only"] is True
    assert set(summary["allowed_fields"]) == {
        "candidate_count",
        "sampled_candidate_count",
        "sampled_candidate_table",
        "target_tie_band",
        "family_metrics",
        "telemetry",
    }
    contract = packet["measurement_contract"]
    assert contract["target_tie_band_id"] == "voteabs=4|marginabs=4"
    assert contract["family_discriminator"]["primary"] == WITHIN_TIE_BAND_PRIMARY_FAMILY_ID
    null_contract = contract["family_discriminator"]["null_distribution"]
    assert null_contract["smaller_bucket_fraction_guard_field"] == (
        "matched_hash_null_fraction_gte_observed_bucket_fraction"
    )
    assert null_contract["larger_regret_capture_guard_field"] == (
        "matched_hash_null_fraction_lte_observed_regret_capture_ratio"
    )
    decision = contract["within_band_decision"]
    assert decision["predictive_branch_label"] == (
        BRANCH_WITHIN_TIE_BAND_LEARNER_FEATURES_SEPARATE_REGRET
    )
    assert decision["fail_closed_branch_label"] == BRANCH_WITHIN_TIE_BAND_NEEDS_NEW_LEARNER_STATE
    assert decision["ambiguous_branch_label"] == BRANCH_WITHIN_TIE_BAND_AMBIGUOUS_NO_BRANCH
    assert contract["fragmentation_audit"]["bucket_cardinality_histogram_required"] is True
    assert contract["fragmentation_audit"]["singleton_bucket_count_required"] is True


def test_within_tie_band_launch_bundle_pins_budget32_and_fixed_two_seed_commands():
    packet = build_within_tie_band_discriminator_launch_bundle(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-within-tie-band",
    )

    validate_within_tie_band_discriminator_launch_bundle(packet)
    assert packet["packet_kind"] == WITHIN_TIE_BAND_DISCRIMINATOR_LAUNCH_BUNDLE_PACKET_KIND
    assert packet["screen_rows"] == 20
    assert packet["curriculum_seed"] == 17
    assert packet["support_order_seeds"] == list(ORACLE_SCREEN_CONTRAST_SEEDS)
    assert packet["same_candidate_set_required"] is True
    assert packet["oracle_feasibility_budget"]["max_sampled_candidates"] == 32
    assert packet["oracle_feasibility_budget"]["max_seconds"] == oracle_screen_budget_max_seconds(
        32
    )
    assert packet["science_contract"]["packet_kind"] == WITHIN_TIE_BAND_DISCRIMINATOR_PACKET_KIND
    assert packet["measurement_contract"] == packet["science_contract"]["measurement_contract"]
    assert packet["terminal_criteria"]["branch_classifier"] == [
        BRANCH_WITHIN_TIE_BAND_LEARNER_FEATURES_SEPARATE_REGRET,
        BRANCH_WITHIN_TIE_BAND_NEEDS_NEW_LEARNER_STATE,
        BRANCH_WITHIN_TIE_BAND_AMBIGUOUS_NO_BRANCH,
    ]
    assert len(packet["commands"]) == 2
    assert {command["support_order_seed"] for command in packet["commands"]} == {43, 29}
    assert all(
        command["oracle_screen_mode"] == "within_tie_band_discriminator"
        for command in packet["commands"]
    )
    assert all(command["max_sampled_candidates"] == 32 for command in packet["commands"])


def test_b2b_sequential_packet_author_scaffold_is_non_executing():
    packet = build_b2b_sequential_within_tie_band_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
    )

    validate_b2b_sequential_within_tie_band_packet(packet)
    assert packet["packet_kind"] == B2B_SEQUENTIAL_WITHIN_TIE_BAND_PACKET_KIND
    assert packet["author_only"] is True
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["pt_mutated"] is False
    contract = packet["measurement_contract"]
    assert contract["capture_side"] == "pre_update_same_vote"
    assert contract["candidate_apply_policy"] == "full_vote_planned_candidate_force_apply_v1"
    assert contract["cross_comparable_to_single_step_oracle_screen"] is False
    assert contract["estimand_non_comparable_to_single_step_sparse_singleton_oracle"] is True
    assert contract["min_steps_for_verdict"] == B2B_SEQUENTIAL_STEPS_FOR_VERDICT
    assert contract["pre_full_stack_diagnostic_only"] is True
    assert contract["runtime_readiness_claim"] is False
    assert contract["training_or_acquisition_claim"] is False
    assert contract["full_sub2_claim"] is False


def test_b2b_sequential_launch_bundle_pins_fifty_step_capture_commands():
    packet = build_b2b_sequential_within_tie_band_launch_bundle(
        parent_path="parent.pt",
        parent_sha256="abc123",
        repo_root="/repo",
        run_root="/tmp/hrm158-b2b-sequential",
    )

    validate_b2b_sequential_within_tie_band_launch_bundle(packet)
    assert packet["packet_kind"] == B2B_SEQUENTIAL_WITHIN_TIE_BAND_LAUNCH_BUNDLE_PACKET_KIND
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert len(packet["commands"]) == 2
    assert {command["support_order_seed"] for command in packet["commands"]} == {43, 29}
    assert all(command["b2b_sequential_capture"] for command in packet["commands"])
    assert all(
        command["steps_requested"] == B2B_SEQUENTIAL_STEPS_FOR_VERDICT
        for command in packet["commands"]
    )
    assert all(
        "--b2b-sequential-within-tie-band-capture" in command["argv"]
        for command in packet["commands"]
    )
    assert all(
        "--oracle-screen-mode" not in command["argv"] for command in packet["commands"]
    )


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda packet: packet["measurement_contract"]["family_discriminator"][
                "null_distribution"
            ].update(
                {
                    "smaller_bucket_fraction_guard_field": (
                        "matched_hash_null_fraction_lte_observed_bucket_fraction"
                    )
                }
            ),
            "smaller-bucket guard field",
        ),
        (
            lambda packet: packet["measurement_contract"]["family_discriminator"][
                "null_distribution"
            ].update(
                {
                    "larger_regret_capture_guard_field": (
                        "matched_hash_null_fraction_gte_observed_regret_capture_ratio"
                    )
                }
            ),
            "regret-capture guard field",
        ),
    ],
)
def test_within_tie_band_packet_validator_rejects_reversed_one_sided_null_guards(
    mutation,
    error,
):
    packet = build_within_tie_band_discriminator_packet(
        parent_path="parent.pt",
        parent_sha256="abc123",
    )
    mutation(packet)

    with pytest.raises(ValueError, match=error):
        validate_within_tie_band_discriminator_packet(packet)


def test_packet_script_writes_compact_launch_packet_with_null_gate(tmp_path: Path, capsys):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"read-only parent bytes")
    parent_sha = hashlib.sha256(b"read-only parent bytes").hexdigest()
    out = tmp_path / "packet.json"

    exit_code = packet_main(
        [
            "--parent",
            str(parent),
            "--parent-sha256",
            parent_sha,
            "--json-out",
            str(out),
            "--mode",
            SCIENCE_MODE_PRETERMINAL_SCREEN,
        ],
    )

    assert exit_code == 0
    packet = json.loads(out.read_text(encoding="utf-8"))
    validate_optimizer_update_law_science_packet(packet)
    assert packet["launch_gate_id"] is None
    assert packet["parent_hash_basis"] == "read_only_parent_file_sha256"
    assert packet["dry_run_packet_written"] is True
    assert packet["gpu_launch_command_authorized"] is False
    assert json.loads(capsys.readouterr().out)["launch_gate_id"] is None


def test_packet_script_writes_step2_author_only_launch_bundle(tmp_path: Path, capsys):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"read-only parent bytes")
    parent_sha = hashlib.sha256(b"read-only parent bytes").hexdigest()
    out = tmp_path / "step2-launch-bundle.json"
    run_root = tmp_path / "run"

    exit_code = packet_main(
        [
            "--packet-kind",
            STEP2_LAUNCH_BUNDLE_PACKET_KIND,
            "--parent",
            str(parent),
            "--parent-sha256",
            parent_sha,
            "--json-out",
            str(out),
            "--run-root",
            str(run_root),
        ],
    )

    assert exit_code == 0
    packet = json.loads(out.read_text(encoding="utf-8"))
    validate_optimizer_update_law_launch_bundle(packet)
    assert packet["packet_kind"] == STEP2_LAUNCH_BUNDLE_PACKET_KIND
    assert packet["launch_gate_id"] is None
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["pt_mutated"] is False
    assert packet["parent_hash_basis"] == "read_only_parent_file_sha256"
    assert packet["dry_run_packet_written"] is True
    assert packet["gpu_launch_command_authorized"] is False
    assert packet["step2_launch_gate_required"] is True
    assert len(packet["commands"]) == 8
    assert json.loads(capsys.readouterr().out)["packet_kind"] == STEP2_LAUNCH_BUNDLE_PACKET_KIND


def test_packet_script_writes_step3_author_only_measurement_power_packet(tmp_path: Path, capsys):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"read-only parent bytes")
    parent_sha = hashlib.sha256(b"read-only parent bytes").hexdigest()
    out = tmp_path / "step3-packet.json"
    run_root = tmp_path / "run"

    exit_code = packet_main(
        [
            "--packet-kind",
            STEP3_MEASUREMENT_POWER_TRUST_REGION_PACKET_KIND,
            "--parent",
            str(parent),
            "--parent-sha256",
            parent_sha,
            "--json-out",
            str(out),
            "--run-root",
            str(run_root),
        ],
    )

    assert exit_code == 0
    packet = json.loads(out.read_text(encoding="utf-8"))
    validate_measurement_power_then_trust_region_packet(packet)
    assert packet["packet_kind"] == STEP3_MEASUREMENT_POWER_TRUST_REGION_PACKET_KIND
    assert packet["launch_gate_id"] is None
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["pt_mutated"] is False
    assert packet["parent_hash_basis"] == "read_only_parent_file_sha256"
    assert packet["dry_run_packet_written"] is True
    assert packet["gpu_launch_command_authorized"] is False
    assert packet["step3_launch_gate_required"] is True
    assert len(packet["commands"]) == 18
    assert json.loads(capsys.readouterr().out)["packet_kind"] == (
        STEP3_MEASUREMENT_POWER_TRUST_REGION_PACKET_KIND
    )


def test_packet_script_writes_step4_author_only_rank_signal_packet(tmp_path: Path, capsys):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"read-only parent bytes")
    parent_sha = hashlib.sha256(b"read-only parent bytes").hexdigest()
    out = tmp_path / "step4-packet.json"
    run_root = tmp_path / "run"

    exit_code = packet_main(
        [
            "--packet-kind",
            STEP4_POWERED_RANK_SIGNAL_DECOMPOSITION_PACKET_KIND,
            "--parent",
            str(parent),
            "--parent-sha256",
            parent_sha,
            "--json-out",
            str(out),
            "--run-root",
            str(run_root),
        ],
    )

    assert exit_code == 0
    packet = json.loads(out.read_text(encoding="utf-8"))
    validate_powered_rank_signal_decomposition_packet(packet)
    assert packet["packet_kind"] == STEP4_POWERED_RANK_SIGNAL_DECOMPOSITION_PACKET_KIND
    assert packet["launch_gate_id"] is None
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["pt_mutated"] is False
    assert packet["parent_hash_basis"] == "read_only_parent_file_sha256"
    assert packet["dry_run_packet_written"] is True
    assert packet["gpu_launch_command_authorized"] is False
    assert packet["step4_launch_gate_required"] is True
    assert len(packet["commands"]) == 10
    assert json.loads(capsys.readouterr().out)["packet_kind"] == (
        STEP4_POWERED_RANK_SIGNAL_DECOMPOSITION_PACKET_KIND
    )


def test_packet_script_writes_step5_support_order_packet(tmp_path: Path, capsys):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"read-only parent bytes")
    parent_sha = hashlib.sha256(b"read-only parent bytes").hexdigest()
    out = tmp_path / "step5-packet.json"
    run_root = tmp_path / "run"

    exit_code = packet_main(
        [
            "--packet-kind",
            STEP5_SUPPORT_ORDER_TRAJECTORY_ROBUSTNESS_PACKET_KIND,
            "--parent",
            str(parent),
            "--parent-sha256",
            parent_sha,
            "--json-out",
            str(out),
            "--run-root",
            str(run_root),
        ],
    )

    assert exit_code == 0
    packet = json.loads(out.read_text(encoding="utf-8"))
    validate_support_order_trajectory_robustness_packet(packet)
    assert packet["packet_kind"] == STEP5_SUPPORT_ORDER_TRAJECTORY_ROBUSTNESS_PACKET_KIND
    assert packet["launch_gate_id"] is None
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["pt_mutated"] is False
    assert packet["parent_hash_basis"] == "read_only_parent_file_sha256"
    assert packet["dry_run_packet_written"] is True
    assert packet["gpu_launch_command_authorized"] is False
    assert packet["step5_launch_gate_required"] is True
    assert len(packet["commands"]) == 4
    assert {
        command["argv"][command["argv"].index("--support-order-seed") + 1]
        for command in packet["commands"]
    } == {"29"}
    assert json.loads(capsys.readouterr().out)["packet_kind"] == (
        STEP5_SUPPORT_ORDER_TRAJECTORY_ROBUSTNESS_PACKET_KIND
    )


def test_packet_script_writes_step6_order_averaged_packet(tmp_path: Path, capsys):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"read-only parent bytes")
    parent_sha = hashlib.sha256(b"read-only parent bytes").hexdigest()
    out = tmp_path / "step6-packet.json"
    run_root = tmp_path / "run"

    exit_code = packet_main(
        [
            "--packet-kind",
            STEP6_ORDER_AVERAGED_A0_COMPONENT_DECOMPOSITION_PACKET_KIND,
            "--parent",
            str(parent),
            "--parent-sha256",
            parent_sha,
            "--json-out",
            str(out),
            "--run-root",
            str(run_root),
        ],
    )

    assert exit_code == 0
    packet = json.loads(out.read_text(encoding="utf-8"))
    validate_order_averaged_a0_component_decomposition_packet(packet)
    assert packet["packet_kind"] == STEP6_ORDER_AVERAGED_A0_COMPONENT_DECOMPOSITION_PACKET_KIND
    assert packet["launch_gate_id"] is None
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["pt_mutated"] is False
    assert packet["parent_hash_basis"] == "read_only_parent_file_sha256"
    assert packet["dry_run_packet_written"] is True
    assert packet["gpu_launch_command_authorized"] is False
    assert packet["step6_launch_gate_required"] is True
    assert len(packet["commands"]) == 9
    original_commands = [
        command for command in packet["commands"]
        if command["support_order_seed"] is None
    ]
    seeded_commands = [
        command for command in packet["commands"]
        if command["support_order_seed"] in {29, 43}
    ]
    assert len(original_commands) == 3
    assert len(seeded_commands) == 6
    assert all("--support-order-seed" not in command["argv"] for command in original_commands)
    assert {
        command["argv"][command["argv"].index("--support-order-seed") + 1]
        for command in seeded_commands
    } == {"29", "43"}
    assert json.loads(capsys.readouterr().out)["packet_kind"] == (
        STEP6_ORDER_AVERAGED_A0_COMPONENT_DECOMPOSITION_PACKET_KIND
    )


def test_packet_script_writes_oracle_screen_author_packet(tmp_path: Path, capsys):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"read-only parent bytes")
    parent_sha = hashlib.sha256(b"read-only parent bytes").hexdigest()
    out = tmp_path / "oracle-screen-packet.json"

    exit_code = packet_main(
        [
            "--packet-kind",
            ORACLE_SCREEN_PACKET_KIND,
            "--parent",
            str(parent),
            "--parent-sha256",
            parent_sha,
            "--json-out",
            str(out),
            "--oracle-screen-max-sampled-candidates",
            "32",
        ],
    )

    assert exit_code == 0
    packet = json.loads(out.read_text(encoding="utf-8"))
    validate_candidate_set_viability_oracle_screen_packet(packet)
    assert packet["packet_kind"] == ORACLE_SCREEN_PACKET_KIND
    assert packet["launch_gate_id"] is None
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["pt_mutated"] is False
    assert packet["parent_hash_basis"] == "read_only_parent_file_sha256"
    assert packet["dry_run_packet_written"] is True
    assert packet["gpu_launch_command_authorized"] is False
    assert packet["oracle_screen_launch_gate_required"] is True
    assert packet["oracle_feasibility_budget"]["max_sampled_candidates"] == 32
    assert packet["oracle_feasibility_budget"]["max_seconds"] == oracle_screen_budget_max_seconds(
        32
    )
    assert packet["wider_screen_interpretation_contract"]["max_sampled_candidates"] == 32
    assert json.loads(capsys.readouterr().out)["packet_kind"] == ORACLE_SCREEN_PACKET_KIND


@pytest.mark.parametrize("budget", [32, 64])
def test_packet_script_writes_oracle_screen_launch_bundle(
    tmp_path: Path,
    capsys,
    budget: int,
):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"read-only parent bytes")
    parent_sha = hashlib.sha256(b"read-only parent bytes").hexdigest()
    out = tmp_path / f"oracle-screen-launch-bundle-{budget}.json"
    run_root = tmp_path / "run"

    exit_code = packet_main(
        [
            "--packet-kind",
            ORACLE_SCREEN_LAUNCH_BUNDLE_PACKET_KIND,
            "--parent",
            str(parent),
            "--parent-sha256",
            parent_sha,
            "--json-out",
            str(out),
            "--run-root",
            str(run_root),
            "--oracle-screen-max-sampled-candidates",
            str(budget),
        ],
    )

    assert exit_code == 0
    packet = json.loads(out.read_text(encoding="utf-8"))
    validate_candidate_set_viability_oracle_screen_launch_bundle(packet)
    assert packet["packet_kind"] == ORACLE_SCREEN_LAUNCH_BUNDLE_PACKET_KIND
    assert packet["launch_gate_id"] is None
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["pt_mutated"] is False
    assert packet["parent_hash_basis"] == "read_only_parent_file_sha256"
    assert packet["dry_run_packet_written"] is True
    assert packet["gpu_launch_command_authorized"] is False
    assert packet["oracle_screen_launch_gate_required"] is True
    assert packet["oracle_feasibility_budget"]["max_sampled_candidates"] == budget
    assert packet["oracle_feasibility_budget"]["max_seconds"] == oracle_screen_budget_max_seconds(
        budget
    )
    assert packet["wider_screen_interpretation_contract"]["max_sampled_candidates"] == budget
    assert len(packet["commands"]) == 2
    assert {command["argv"][command["argv"].index("--support-order-seed") + 1] for command in packet["commands"]} == {
        "29",
        "43",
    }
    assert all(
        command["argv"][
            command["argv"].index("--oracle-screen-max-sampled-candidates") + 1
        ]
        == str(budget)
        for command in packet["commands"]
    )
    assert all("--science-arm" not in command["argv"] for command in packet["commands"])
    assert json.loads(capsys.readouterr().out)["packet_kind"] == (
        ORACLE_SCREEN_LAUNCH_BUNDLE_PACKET_KIND
    )


def test_packet_script_writes_credit_ranking_pivot_author_packet(
    tmp_path: Path,
    capsys,
):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"read-only parent bytes")
    parent_sha = hashlib.sha256(b"read-only parent bytes").hexdigest()
    out = tmp_path / "credit-ranking-pivot-packet.json"

    exit_code = packet_main(
        [
            "--packet-kind",
            CREDIT_RANKING_PIVOT_MEASUREMENT_PACKET_KIND,
            "--parent",
            str(parent),
            "--parent-sha256",
            parent_sha,
            "--json-out",
            str(out),
        ],
    )

    assert exit_code == 0
    packet = json.loads(out.read_text(encoding="utf-8"))
    validate_credit_ranking_pivot_measurement_packet(packet)
    assert packet["packet_kind"] == CREDIT_RANKING_PIVOT_MEASUREMENT_PACKET_KIND
    assert packet["launch_gate_id"] is None
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["pt_mutated"] is False
    assert packet["parent_hash_basis"] == "read_only_parent_file_sha256"
    assert packet["dry_run_packet_written"] is True
    assert packet["gpu_launch_command_authorized"] is False
    assert packet["oracle_screen_launch_gate_required"] is True
    assert packet["oracle_feasibility_budget"]["max_sampled_candidates"] == 32
    assert json.loads(capsys.readouterr().out)["packet_kind"] == (
        CREDIT_RANKING_PIVOT_MEASUREMENT_PACKET_KIND
    )


def test_packet_script_writes_credit_ranking_pivot_launch_bundle(
    tmp_path: Path,
    capsys,
):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"read-only parent bytes")
    parent_sha = hashlib.sha256(b"read-only parent bytes").hexdigest()
    out = tmp_path / "credit-ranking-pivot-launch-bundle.json"
    run_root = tmp_path / "run"

    exit_code = packet_main(
        [
            "--packet-kind",
            CREDIT_RANKING_PIVOT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND,
            "--parent",
            str(parent),
            "--parent-sha256",
            parent_sha,
            "--json-out",
            str(out),
            "--run-root",
            str(run_root),
        ],
    )

    assert exit_code == 0
    packet = json.loads(out.read_text(encoding="utf-8"))
    validate_credit_ranking_pivot_measurement_launch_bundle(packet)
    assert packet["packet_kind"] == CREDIT_RANKING_PIVOT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND
    assert packet["launch_gate_id"] is None
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["pt_mutated"] is False
    assert packet["parent_hash_basis"] == "read_only_parent_file_sha256"
    assert packet["dry_run_packet_written"] is True
    assert packet["gpu_launch_command_authorized"] is False
    assert packet["oracle_screen_launch_gate_required"] is True
    assert packet["oracle_feasibility_budget"]["max_sampled_candidates"] == 32
    assert len(packet["commands"]) == 2
    assert {
        command["argv"][command["argv"].index("--support-order-seed") + 1]
        for command in packet["commands"]
    } == {"29", "43"}
    assert json.loads(capsys.readouterr().out)["packet_kind"] == (
        CREDIT_RANKING_PIVOT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND
    )


def test_packet_script_writes_activation_credit_author_packet(
    tmp_path: Path,
    capsys,
):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"read-only parent bytes")
    parent_sha = hashlib.sha256(b"read-only parent bytes").hexdigest()
    out = tmp_path / "activation-credit-packet.json"

    exit_code = packet_main(
        [
            "--packet-kind",
            ACTIVATION_CREDIT_MEASUREMENT_PACKET_KIND,
            "--parent",
            str(parent),
            "--parent-sha256",
            parent_sha,
            "--json-out",
            str(out),
        ],
    )

    assert exit_code == 0
    packet = json.loads(out.read_text(encoding="utf-8"))
    validate_activation_credit_measurement_packet(packet)
    assert packet["packet_kind"] == ACTIVATION_CREDIT_MEASUREMENT_PACKET_KIND
    assert packet["launch_gate_id"] is None
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["pt_mutated"] is False
    assert packet["parent_hash_basis"] == "read_only_parent_file_sha256"
    assert packet["dry_run_packet_written"] is True
    assert packet["gpu_launch_command_authorized"] is False
    assert packet["oracle_screen_launch_gate_required"] is True
    assert packet["oracle_feasibility_budget"]["max_sampled_candidates"] == 32
    assert json.loads(capsys.readouterr().out)["packet_kind"] == (
        ACTIVATION_CREDIT_MEASUREMENT_PACKET_KIND
    )


def test_packet_script_writes_activation_credit_launch_bundle(
    tmp_path: Path,
    capsys,
):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"read-only parent bytes")
    parent_sha = hashlib.sha256(b"read-only parent bytes").hexdigest()
    out = tmp_path / "activation-credit-launch-bundle.json"
    run_root = tmp_path / "run"

    exit_code = packet_main(
        [
            "--packet-kind",
            ACTIVATION_CREDIT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND,
            "--parent",
            str(parent),
            "--parent-sha256",
            parent_sha,
            "--json-out",
            str(out),
            "--run-root",
            str(run_root),
        ],
    )

    assert exit_code == 0
    packet = json.loads(out.read_text(encoding="utf-8"))
    validate_activation_credit_measurement_launch_bundle(packet)
    assert packet["packet_kind"] == ACTIVATION_CREDIT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND
    assert packet["launch_gate_id"] is None
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["pt_mutated"] is False
    assert packet["parent_hash_basis"] == "read_only_parent_file_sha256"
    assert packet["dry_run_packet_written"] is True
    assert packet["gpu_launch_command_authorized"] is False
    assert packet["oracle_screen_launch_gate_required"] is True
    assert packet["oracle_feasibility_budget"]["max_sampled_candidates"] == 32
    assert len(packet["commands"]) == 2
    assert (
        packet["measurement_contract"]["family_discriminator"]["primary"]
        == ACTIVATION_CREDIT_PRIMARY_FAMILY_ID
    )
    assert {
        command["argv"][command["argv"].index("--support-order-seed") + 1]
        for command in packet["commands"]
    } == {"29", "43"}
    assert json.loads(capsys.readouterr().out)["packet_kind"] == (
        ACTIVATION_CREDIT_MEASUREMENT_LAUNCH_BUNDLE_PACKET_KIND
    )


def test_packet_script_writes_activation_credit_smoke_launch_bundle(
    tmp_path: Path,
    capsys,
):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"read-only parent bytes")
    parent_sha = hashlib.sha256(b"read-only parent bytes").hexdigest()
    out = tmp_path / "activation-credit-smoke-launch-bundle.json"
    run_root = tmp_path / "run"

    exit_code = packet_main(
        [
            "--packet-kind",
            ACTIVATION_CREDIT_SCALE_SMOKE_LAUNCH_BUNDLE_PACKET_KIND,
            "--parent",
            str(parent),
            "--parent-sha256",
            parent_sha,
            "--json-out",
            str(out),
            "--run-root",
            str(run_root),
        ],
    )

    assert exit_code == 0
    packet = json.loads(out.read_text(encoding="utf-8"))
    validate_activation_credit_scale_smoke_launch_bundle(packet)
    assert packet["packet_kind"] == ACTIVATION_CREDIT_SCALE_SMOKE_LAUNCH_BUNDLE_PACKET_KIND
    assert packet["launch_gate_id"] is None
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["pt_mutated"] is False
    assert packet["parent_hash_basis"] == "read_only_parent_file_sha256"
    assert packet["dry_run_packet_written"] is True
    assert packet["gpu_launch_command_authorized"] is False
    assert packet["oracle_screen_launch_gate_required"] is True
    assert packet["oracle_feasibility_budget"]["max_sampled_candidates"] == (
        ACTIVATION_CREDIT_SMOKE_MAX_SAMPLED_CANDIDATES
    )
    assert len(packet["commands"]) == 2
    assert all(
        command["env"][ACTIVATION_CREDIT_STDOUT_PATH_ENV] == command["stdout_path"]
        and command["env"][ACTIVATION_CREDIT_STDERR_PATH_ENV] == command["stderr_path"]
        for command in packet["commands"]
    )
    assert json.loads(capsys.readouterr().out)["packet_kind"] == (
        ACTIVATION_CREDIT_SCALE_SMOKE_LAUNCH_BUNDLE_PACKET_KIND
    )


def test_packet_script_writes_within_tie_band_author_packet(
    tmp_path: Path,
    capsys,
):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"read-only parent bytes")
    parent_sha = hashlib.sha256(b"read-only parent bytes").hexdigest()
    out = tmp_path / "within-tie-band-packet.json"

    exit_code = packet_main(
        [
            "--packet-kind",
            WITHIN_TIE_BAND_DISCRIMINATOR_PACKET_KIND,
            "--parent",
            str(parent),
            "--parent-sha256",
            parent_sha,
            "--json-out",
            str(out),
        ],
    )

    assert exit_code == 0
    packet = json.loads(out.read_text(encoding="utf-8"))
    validate_within_tie_band_discriminator_packet(packet)
    assert packet["packet_kind"] == WITHIN_TIE_BAND_DISCRIMINATOR_PACKET_KIND
    assert packet["launch_gate_id"] is None
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["pt_mutated"] is False
    assert packet["parent_hash_basis"] == "read_only_parent_file_sha256"
    assert packet["dry_run_packet_written"] is True
    assert packet["gpu_launch_command_authorized"] is False
    assert packet["oracle_screen_launch_gate_required"] is True
    assert packet["oracle_feasibility_budget"]["max_sampled_candidates"] == 32
    assert json.loads(capsys.readouterr().out)["packet_kind"] == (
        WITHIN_TIE_BAND_DISCRIMINATOR_PACKET_KIND
    )


def test_packet_script_writes_within_tie_band_launch_bundle(
    tmp_path: Path,
    capsys,
):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"read-only parent bytes")
    parent_sha = hashlib.sha256(b"read-only parent bytes").hexdigest()
    out = tmp_path / "within-tie-band-launch-bundle.json"
    run_root = tmp_path / "run"

    exit_code = packet_main(
        [
            "--packet-kind",
            WITHIN_TIE_BAND_DISCRIMINATOR_LAUNCH_BUNDLE_PACKET_KIND,
            "--parent",
            str(parent),
            "--parent-sha256",
            parent_sha,
            "--json-out",
            str(out),
            "--run-root",
            str(run_root),
        ],
    )

    assert exit_code == 0
    packet = json.loads(out.read_text(encoding="utf-8"))
    validate_within_tie_band_discriminator_launch_bundle(packet)
    assert packet["packet_kind"] == WITHIN_TIE_BAND_DISCRIMINATOR_LAUNCH_BUNDLE_PACKET_KIND
    assert packet["launch_gate_id"] is None
    assert packet["commands_executed"] is False
    assert packet["gpu_launched"] is False
    assert packet["pt_mutated"] is False
    assert packet["parent_hash_basis"] == "read_only_parent_file_sha256"
    assert packet["dry_run_packet_written"] is True
    assert packet["gpu_launch_command_authorized"] is False
    assert packet["oracle_screen_launch_gate_required"] is True
    assert packet["oracle_feasibility_budget"]["max_sampled_candidates"] == 32
    assert len(packet["commands"]) == 2
    assert {
        command["argv"][command["argv"].index("--support-order-seed") + 1]
        for command in packet["commands"]
    } == {"29", "43"}
    assert json.loads(capsys.readouterr().out)["packet_kind"] == (
        WITHIN_TIE_BAND_DISCRIMINATOR_LAUNCH_BUNDLE_PACKET_KIND
    )
