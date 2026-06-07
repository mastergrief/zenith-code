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
    BRANCH_RANK_FREE_POSITIVE,
    BRANCH_TIE_POLICY_OR_OVERUPDATE,
    CONTROL_PARITY_FRACTION_MAX,
    CONTROL_PARITY_FRACTION_MIN,
    FIXED_RANK_BUCKET_NON_TARGET_AUX,
    SCIENCE_MODE_BRANCH_VERDICT,
    SCIENCE_MODE_PRETERMINAL_SCREEN,
    TIE_POLICY_CURRENT_MARGIN_INDEX,
    TIE_POLICY_DETERMINISTIC_HASH_MATCHED,
    build_optimizer_update_law_science_packet,
    classify_optimizer_update_law_branch,
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
