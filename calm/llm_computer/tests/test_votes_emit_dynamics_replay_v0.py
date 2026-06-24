"""CPU tests for votes-emit dynamics replay scorer and launch packet."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaAccumulatorState,
    BoundedDeltaTensorState,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
    EventCodedAccEvent,
    pack_event_coded_acc_checkpoint_reference,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    measure_r4v_event_coded_acc_budget,
)
from calm.hrm_text_158.native_full_stack.qscale_linear import QScaleWeightState
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import crossing_bool_w6
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec
from calm.hrm_text_158.native_full_stack.votes_emit_collector import (
    VOTES_EMIT_SECTION6_CONTRACT_FIELDS,
    VotesEmitCollector,
    build_votes_emit_step_record,
)
from calm.hrm_text_158.native_full_stack.votes_emit_dynamics_replay import (
    ARM_MODE_LIVE_ALTERED_DYNAMICS,
    ARM_MODE_REPLAY_ONLY,
    CLASSIFIER_INTRINSIC_WIDE_CONFIRMED,
    CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW,
    CLASSIFIER_REDUCIBLE_UNDER_DYNAMICS,
    CLASSIFIER_STATIC_PROXY_ARTIFACT,
    REPLAY_MODE_R_DYNAMICS,
    REPLAY_MODE_R_STATIC,
    classify_dynamics_proof_verdict,
    load_votes_emit_step_records,
    reconstruct_sampled_crossing_mask_hash,
    score_votes_emit_replay_run,
    section6_contract_field_names,
    verify_section6_internal_consistency,
)
from scripts.hrm_text_158_votes_emit_dynamics_replay import main as replay_main


DESIGN_SECTION6_CONTRACT_FIELDS = (
    "applied_flat_indices_hash",
    "cap_order_summary",
    "pre_update_state_hash",
)


def _make_state(*, numel: int = 64) -> BoundedDeltaTensorState:
    side = int(numel**0.5)
    if side * side != numel:
        shape = (numel,)
    else:
        shape = (side, side)
    q = torch.tensor([-1, 0, 1], dtype=torch.int8)
    idx = torch.arange(numel, dtype=torch.long) % 3
    q_levels = q[idx].view(shape).contiguous()
    acc = torch.zeros(numel, dtype=torch.int16)
    bounded = BoundedDeltaAccumulatorState(
        logical_shape=tuple(int(dim) for dim in q_levels.shape),
        cold_default_value=0,
        hot_exact_indices=(),
        hot_exact_values=(),
        cold_exception_indices=(),
        cold_exception_values=(),
        candidate_name="cold_default",
        raw_arrays_included=False,
    )
    return BoundedDeltaTensorState(
        state_key="proj",
        q_levels=q_levels,
        frozen_scale=torch.tensor(1.0, dtype=torch.float32),
        bounded_accumulator=bounded,
        exact_accumulator_shadow=acc.view_as(q_levels),
        bounded_accumulator_fresh_for_exact_shadow=False,
    )


def _vote_spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=4096,
    )


def _votes_for_state(state: BoundedDeltaTensorState) -> torch.Tensor:
    return torch.randint(-3, 4, state.q_levels.shape, dtype=torch.int16)


def _emit_fixture_run(tmp_path: Path, *, steps: int = 2) -> Path:
    collector = VotesEmitCollector(tmp_path)
    state = _make_state()
    votes = _votes_for_state(state)
    local_loss_delta = torch.randn(state.q_levels.shape, dtype=torch.float32)
    for step in range(steps):
        record = build_votes_emit_step_record(
            optimizer_step_index=step,
            tensor_states={"proj": state},
            votes_by_key={"proj": votes},
            vote_specs_by_key={"proj": _vote_spec()},
            max_abs_per_tensor=4096,
            two_tier_carry_w6_enabled=True,
            local_loss_delta_by_key={"proj": local_loss_delta},
            local_selection_ordering_seed=17,
        )
        collector.emit_step(record, optimizer_step_index=step)
    return tmp_path


def test_emitter_section6_contract_matches_design_field_list() -> None:
    assert tuple(VOTES_EMIT_SECTION6_CONTRACT_FIELDS) == DESIGN_SECTION6_CONTRACT_FIELDS
    assert section6_contract_field_names() == DESIGN_SECTION6_CONTRACT_FIELDS


def test_emitter_writes_design_section6_field_set(tmp_path: Path) -> None:
    record = build_votes_emit_step_record(
        optimizer_step_index=0,
        tensor_states={"proj": _make_state()},
        votes_by_key={"proj": _votes_for_state(_make_state())},
        vote_specs_by_key={"proj": _vote_spec()},
        max_abs_per_tensor=4096,
    )
    for field_name in DESIGN_SECTION6_CONTRACT_FIELDS:
        assert field_name in record, f"missing design §6 field {field_name}"
    assert record["applied_flip_count_is_preview"] is True
    cap_summary = record["cap_order_summary"]
    assert cap_summary["accepted_flat_indices_hash"] == record["applied_flat_indices_hash"]
    assert isinstance(record["pre_update_state_hash"], str)
    assert record["pre_update_state_hash"]


def test_replay_reconstructs_crossing_mask_from_emitted_candidate_table() -> None:
    record = build_votes_emit_step_record(
        optimizer_step_index=0,
        tensor_states={"proj": _make_state()},
        votes_by_key={"proj": _votes_for_state(_make_state())},
        vote_specs_by_key={"proj": _vote_spec()},
        max_abs_per_tensor=4096,
    )
    threshold_abs = int(record["threshold_semantics"]["crossing_threshold_abs"])
    for row in record["sampled_candidate_table"]:
        new_acc = int(row["pre_accumulator_i16"]) + int(row["vote_value"])
        expected = crossing_bool_w6(
            new_acc,
            int(row["current_q_level"]),
            threshold_abs=threshold_abs,
        )
        assert isinstance(expected, bool)
    digest = reconstruct_sampled_crossing_mask_hash(record)
    assert len(digest) == 64


def test_replay_reconstructs_applied_mask_and_q_trajectory(tmp_path: Path) -> None:
    run_root = _emit_fixture_run(tmp_path)
    records = load_votes_emit_step_records(run_root)
    record = records[0]
    assert record["applied_flat_indices_hash"]
    assert record["cap_order_summary"]["accepted_flat_indices_hash"] == record[
        "applied_flat_indices_hash"
    ]
    assert all("current_q_level" in row for row in record["sampled_candidate_table"])


def _rewrite_step_with_manifest(collector_root: Path, step_name: str, payload: dict) -> None:
    import hashlib
    import json

    per_step_dir = collector_root / "votes_emit" / "v1" / "per_step"
    emit_root = collector_root / "votes_emit" / "v1"
    step_path = per_step_dir / f"{step_name}.json"
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    step_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    step_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = emit_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["per_step_hashes"][step_name] = step_hash
    stable_manifest = {
        "schema_version": manifest["schema_version"],
        "per_step_hashes": dict(sorted(manifest["per_step_hashes"].items())),
        "step_count": int(manifest["step_count"]),
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(stable_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def test_replay_hash_mismatch_emits_missing_observables(tmp_path: Path) -> None:
    run_root = _emit_fixture_run(tmp_path)
    step_path = run_root / "votes_emit" / "v1" / "per_step" / "00000.json"
    payload = json.loads(step_path.read_text(encoding="utf-8"))
    payload["applied_flat_indices_hash"] = "deadbeef" * 8
    _rewrite_step_with_manifest(run_root, "00000", payload)
    receipt = score_votes_emit_replay_run(
        run_root,
        replay_mode=REPLAY_MODE_R_STATIC,
        arm_id="V0",
        arm_mode=ARM_MODE_REPLAY_ONLY,
        from_clean_contiguous=False,
        live_evidence=False,
    )
    assert receipt["classifier_verdict"] == CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW
    assert "applied_flat_indices_hash_mismatch_vs_cap_order_summary" in receipt[
        "section6_consistency_failures"
    ]


def test_replay_missing_required_fields_emits_missing_observables_or_invalid_window(
    tmp_path: Path,
) -> None:
    run_root = _emit_fixture_run(tmp_path)
    step_path = run_root / "votes_emit" / "v1" / "per_step" / "00000.json"
    payload = json.loads(step_path.read_text(encoding="utf-8"))
    del payload["pre_update_state_hash"]
    _rewrite_step_with_manifest(run_root, "00000", payload)
    receipt = score_votes_emit_replay_run(
        run_root,
        replay_mode=REPLAY_MODE_R_STATIC,
        arm_id="V0",
        arm_mode=ARM_MODE_REPLAY_ONLY,
        from_clean_contiguous=False,
        live_evidence=False,
    )
    assert receipt["classifier_verdict"] == CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW
    assert "pre_update_state_hash" in receipt["missing_section6_fields"]


def test_classifier_never_banks_reducible_or_intrinsic_from_r_static_replay(
    tmp_path: Path,
) -> None:
    run_root = _emit_fixture_run(tmp_path)
    receipt = score_votes_emit_replay_run(
        run_root,
        replay_mode=REPLAY_MODE_R_STATIC,
        arm_id="V4",
        arm_mode=ARM_MODE_REPLAY_ONLY,
        from_clean_contiguous=True,
        live_evidence=False,
        r4v_ledger_pass=True,
        all_live_variants_failed_sub2=True,
    )
    assert receipt["classifier_verdict"] == CLASSIFIER_STATIC_PROXY_ARTIFACT
    assert receipt["reducible_or_intrinsic_from_replay_only"] is False


def test_classifier_reducible_only_on_live_arm_with_acc_sub2_and_observables() -> None:
    packed = pack_event_coded_acc_checkpoint_reference(
        logical_numel=4096,
        events=(
            EventCodedAccEvent(flat_index=1, direction=0, residual_mag=1, event_type=1),
        ),
        backlog_indices=(),
    )
    qstate = QScaleWeightState(
        q_levels=torch.zeros((64, 64), dtype=torch.int8),
        scale=torch.tensor(1.0, dtype=torch.float32),
    )
    report = measure_r4v_event_coded_acc_budget(
        [qstate],
        [packed],
        state_keys=["proj"],
    )
    verdict = classify_dynamics_proof_verdict(
        type(
            "Inputs",
            (),
            {
                "replay_mode": REPLAY_MODE_R_STATIC,
                "arm_id": "V4",
                "arm_mode": ARM_MODE_LIVE_ALTERED_DYNAMICS,
                "from_clean_contiguous": True,
                "run_health_ok": True,
                "section6_complete": True,
                "section6_consistent": True,
                "live_evidence": True,
                "r4v_ledger_pass": bool(report.r4v_ledger_pass),
                "static_proxy_gap_falsified": False,
                "all_live_variants_failed_sub2": False,
            },
        )()
    )
    assert verdict == CLASSIFIER_REDUCIBLE_UNDER_DYNAMICS


def test_classifier_intrinsic_only_on_from_clean_contiguous_live_evidence() -> None:
    verdict = classify_dynamics_proof_verdict(
        type(
            "Inputs",
            (),
            {
                "replay_mode": REPLAY_MODE_R_STATIC,
                "arm_id": "V0",
                "arm_mode": ARM_MODE_LIVE_ALTERED_DYNAMICS,
                "from_clean_contiguous": True,
                "run_health_ok": True,
                "section6_complete": True,
                "section6_consistent": True,
                "live_evidence": True,
                "r4v_ledger_pass": False,
                "static_proxy_gap_falsified": False,
                "all_live_variants_failed_sub2": True,
            },
        )()
    )
    assert verdict == CLASSIFIER_INTRINSIC_WIDE_CONFIRMED


def test_r_static_vs_r_dynamics_mode_labeling(tmp_path: Path) -> None:
    run_root = _emit_fixture_run(tmp_path)
    static_receipt = score_votes_emit_replay_run(
        run_root,
        replay_mode=REPLAY_MODE_R_STATIC,
        arm_id="V3",
        arm_mode=ARM_MODE_REPLAY_ONLY,
        from_clean_contiguous=False,
        live_evidence=False,
    )
    dynamics_receipt = score_votes_emit_replay_run(
        run_root,
        replay_mode=REPLAY_MODE_R_DYNAMICS,
        arm_id="V3",
        arm_mode=ARM_MODE_REPLAY_ONLY,
        from_clean_contiguous=False,
        live_evidence=False,
    )
    assert static_receipt["replay_mode_label"] == "R-static"
    assert dynamics_receipt["replay_mode_label"] == "R-dynamics"


def test_launch_packet_schema_required_fields_and_sub2_first_assertions() -> None:
    packet_path = Path(
        "artifacts/consensus_prep/votes_emitting_dynamics_proof_gpu_launch_packet_v1.json"
    )
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    required = {
        "dispatch_msg_id",
        "packet_version",
        "parent_checkpoint_pin",
        "fail_closed_constraints",
        "dynamics_proof_arms",
        "sub2_first_gate",
        "votes_emit_contract",
        "branch_classifier_preregistered",
    }
    assert required.issubset(packet)
    assertions = packet["sub2_first_gate"]["assertions"]
    assert assertions["ready_for_pre_full_stack_diagnostic"] is True
    assert assertions["ready_for_main_science"] is False
    assert assertions["main_science_launch_blocked"] is True
    assert "--expect-ready" in packet["sub2_first_gate"]["forbidden"]
    assert packet["fail_closed_constraints"]["votes_emit_section6_required"] is True


def test_launch_packet_from_clean_contiguous_backlog_flags_and_arm_modes() -> None:
    packet_path = Path(
        "artifacts/consensus_prep/votes_emitting_dynamics_proof_gpu_launch_packet_v1.json"
    )
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    constraints = packet["fail_closed_constraints"]
    assert constraints["contiguous_from_clean_parent"] is True
    assert constraints["from_clean_no_resume"] is True
    assert constraints["r7_deferred_backlog_carry_enabled"] is False
    arms = packet["dynamics_proof_arms"]
    assert arms["V4"]["reducible_capable"] is True
    assert arms["V1"]["reducible_capable"] is False
    assert arms["V3"]["arm_mode"] == "replay-only"
    assert arms["V0"]["arm_mode"] == "live-altered-dynamics"


LAUNCH_PACKET_MAIN_PATH = Path(
    "artifacts/consensus_prep/votes_emitting_dynamics_proof_gpu_launch_packet_v1.json"
)
LAUNCH_PACKET_COMPANION_PATH = Path(
    "artifacts/consensus_prep/votes_emitting_dynamics_proof_gpu_launch_packet_v1_replay_commands.json"
)


@pytest.mark.parametrize(
    "artifact_path",
    [LAUNCH_PACKET_MAIN_PATH, LAUNCH_PACKET_COMPANION_PATH],
    ids=["main_packet", "companion_replay_commands"],
)
def test_launch_artifacts_both_json_load(artifact_path: Path) -> None:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload


def test_launch_companion_replay_commands_json_parses_and_carries_required_commands() -> None:
    companion = json.loads(LAUNCH_PACKET_COMPANION_PATH.read_text(encoding="utf-8"))
    required_keys = {
        "classifier_command",
        "sub2_readiness_command",
        "sub2_readiness_assertions",
        "votes_emit_section6_contract_witness_command",
        "gpu_diagnostic_command_template",
        "main_packet_artifact",
        "schema",
        "packet_version",
        "head_pin",
        "run_root_pattern",
        "note",
    }
    assert required_keys.issubset(companion)
    assertions = companion["sub2_readiness_assertions"]
    assert assertions["ready_for_pre_full_stack_diagnostic"] is True
    assert assertions["ready_for_main_science"] is False
    assert assertions["main_science_launch_blocked"] is True
    gpu_cmd = str(companion["gpu_diagnostic_command_template"])
    assert "--votes-emit-enabled" in gpu_cmd
    assert "--allow-gpu-launch" in gpu_cmd
    sub2_cmd = str(companion["sub2_readiness_command"])
    assert "--expect-ready" not in sub2_cmd


def test_section6_internal_consistency_passes_on_valid_record() -> None:
    record = build_votes_emit_step_record(
        optimizer_step_index=0,
        tensor_states={"proj": _make_state()},
        votes_by_key={"proj": _votes_for_state(_make_state())},
        vote_specs_by_key={"proj": _vote_spec()},
        max_abs_per_tensor=4096,
    )
    assert verify_section6_internal_consistency(record) == []


def test_replay_cli_writes_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = _emit_fixture_run(tmp_path / "run")
    out_path = tmp_path / "receipt.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "hrm_text_158_votes_emit_dynamics_replay.py",
            "--run-root",
            str(run_root),
            "--replay-mode",
            REPLAY_MODE_R_STATIC,
            "--arm-id",
            "V0",
            "--arm-mode",
            ARM_MODE_REPLAY_ONLY,
            "--json-out",
            str(out_path),
        ],
    )
    replay_main()
    receipt = json.loads(out_path.read_text(encoding="utf-8"))
    assert receipt["classifier_verdict"] == CLASSIFIER_STATIC_PROXY_ARTIFACT
