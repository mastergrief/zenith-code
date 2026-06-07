"""Step-1 optimizer/update-law science packet tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.optimizer_update_law_science import (
    ARM_A0_RANK_BUCKET_CURRENT,
    ARM_A1_RANK_BUCKET_ORDER_MATCHED,
    ARM_B_RANK_FREE_SIGN_PRESSURE,
    ARM_INVERTED_SIGN_PRESSURE,
    BRANCH_INSUFFICIENT_SEPARATION,
    BRANCH_PRIOR_NULL_SETUP_UNVERIFIED,
    BRANCH_RANK_FREE_POSITIVE,
    BRANCH_TIE_POLICY_OR_OVERUPDATE,
    CONTROL_PARITY_FRACTION_MAX,
    CONTROL_PARITY_FRACTION_MIN,
    FIXED_RANK_BUCKET_NON_TARGET_AUX,
    SCIENCE_MODE_BRANCH_VERDICT,
    SCIENCE_MODE_PRETERMINAL_SCREEN,
    STEP1_DRY_RUN_PACKET_KIND,
    STEP2_LAUNCH_BUNDLE_PACKET_KIND,
    TIE_POLICY_CURRENT_MARGIN_INDEX,
    TIE_POLICY_DETERMINISTIC_HASH_MATCHED,
    build_optimizer_update_law_launch_bundle,
    build_optimizer_update_law_science_packet,
    classify_optimizer_update_law_branch,
    validate_optimizer_update_law_launch_bundle,
    validate_optimizer_update_law_science_packet,
)
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
