from __future__ import annotations

import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.accumulator_policy_shadow_screen import (
    ACCUMULATOR_POLICY_SHADOW_SCREEN_SCHEMA_VERSION_DYNAMICS,
    ARM_ACCUMULATOR_ONLY,
    ARM_ACCUMULATOR_PLUS_TRANSIENT,
    ARM_INT16_BASELINE,
    ARM_TRANSIENT_RESOLVER_ONLY,
    CLAIM_ALGORITHMIC_PROXY_NOT_PHYSICAL_SUB2,
    CLAIM_INT16_REFERENCE,
    CLAIM_SUB2,
    CLAIM_TRANSIENT_FP_DEBT,
    DEFAULT_PREREG_THRESHOLDS,
    FAIL_ACCUMULATOR_FIELDS_UNAVAILABLE,
    FAIL_MULTI_SOURCE_FUSION_REJECTED,
    FAIL_NO_REAL_CANDIDATE_TABLE,
    FAIL_NO_REAL_SEQUENTIAL_CAPTURE,
    FAIL_INSUFFICIENT_OPTIMIZER_STEPS,
    FAIL_MULTIPLE_SEQUENTIAL_TRACES_REJECTED_WITHOUT_ALIGNMENT_PROOF,
    FAIL_NON_MONOTONIC_STEP_INDEX,
    FAIL_MIXED_SOURCE_KIND,
    B2B_SEQUENTIAL_TRACE_SCHEMA,
    REAL_SEQUENTIAL_TRACE_PROOF_SIDE,
    build_real_sequential_trace_candidate_stream,
    FAIL_TRANSIENT_FIELDS_UNAVAILABLE,
    LABEL_ACCUMULATOR_IMPROVES_BUT_NOT_ENOUGH,
    LABEL_ACCUMULATOR_TRACKS_INT16_POLICY,
    LABEL_SCREEN_HARNESS_OR_GATE_FAIL,
    LABEL_STATIC_PROXY_NOT_PERSISTENT_DYNAMICS,
    LABEL_STATIC_TRANSIENT_PROXY_AVAILABLE,
    LABEL_TRANSIENT_CARRIES_SELECTION,
    PRE_FULL_STACK_DIAGNOSTIC_ONLY,
    REQUIRED_SHADOW_ARMS,
    REQUIRED_THRESHOLD_FIELDS,
    REAL_SEQUENTIAL_TRACE_REPLAY_RECEIPT_KIND,
    REAL_TABLE_REPLAY_RECEIPT_KIND,
    SOURCE_KIND_ACTIVATION_CREDIT_MEASUREMENT,
    SOURCE_KIND_WITHIN_TIE_BAND_DISCRIMINATOR,
    SYNTHETIC_DYNAMICS_RECEIPT_KIND,
    TRACE_TEMPORALITY_SEQUENTIAL_OPTIMIZER_STEPS,
    TRACE_TEMPORALITY_STATIC_SNAPSHOT,
    TRACKING_SCOPE_OPTIMIZER_STEP_TRAJECTORY,
    TRACKING_SCOPE_SNAPSHOT_SCREEN,
    _DynamicsArmShadowState,
    _dynamics_flip_policy_score,
    _row_only_flip_policy_score,
    _row_only_selection_at_step,
    _select_dynamics_candidate,
    build_synthetic_temporal_within_tie_band_stream,
    run_accumulator_policy_dynamics_screen,
    run_accumulator_policy_shadow_screen,
    run_real_sequential_trace_replay_from_paths,
    run_real_table_static_proxy_replay_from_paths,
)


def _write_receipt(path: Path, *, mode: str, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "hrm_text_158_c2p1_real_model_bounded_delta_probe/v0",
                "oracle_screen": {
                    "schema": f"hrm_text_158_{mode}_runtime/v0",
                    "mode": mode,
                    "compact_summary": {
                        "sampled_candidate_count": len(rows),
                        "candidate_count": len(rows),
                        "sampled_candidate_table": rows,
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _activation_row(candidate_id: str, rank: int, delta: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "state_key": "layer.weight",
        "flat_index": rank,
        "vote_value": 1,
        "current_margin_abs": 10 - rank,
        "current_rank_position": rank,
        "candidate_loss": 1.0 + delta,
        "local_loss_delta": delta,
        "taylor_benefit": float(10 - rank),
        "snr": float(5 - rank),
        "diag_fisher": float(rank + 1),
        "activation_feature_valid": True,
    }


def _within_tie_band_row(candidate_id: str, rank: int, delta: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "state_key": "layer.weight",
        "flat_index": rank,
        "vote_value": 1,
        "current_margin_abs": 10 - rank,
        "current_rank_position": rank,
        "tie_band_id": "target",
        "current_q_level": 0,
        "pre_accumulator_i16": rank,
        "new_acc_i32_signed": 20 - rank,
        "proposal_direction": 1,
        "threshold_residual_signed": rank - 2,
        "proximity_to_threshold": rank,
        "state_candidate_count": 3,
        "current_rank_quartile_within_state": 0,
        "flat_index_quartile": 0,
        "transition_class": "toward_threshold",
        "candidate_loss": 1.0 + delta,
        "local_loss_delta": delta,
    }


def _canonical_b2b_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        [
            {
                "candidate_id": str(row["candidate_id"]),
                "current_rank_position": int(row["current_rank_position"]),
                "local_loss_delta": float(row["local_loss_delta"]),
                "pre_accumulator_i16": int(row["pre_accumulator_i16"]),
                "new_acc_i32_signed": int(row["new_acc_i32_signed"]),
                "proximity_to_threshold": int(row["proximity_to_threshold"]),
            }
            for row in rows
        ],
        key=lambda row: str(row["candidate_id"]),
    )


def _write_b2b_sequential_trace(
    path: Path,
    *,
    steps: int,
    candidates_per_step: int = 4,
) -> Path:
    from calm.hrm_text_158.native_full_stack.accumulator_policy_shadow_screen import (
        _stable_hash16,
    )

    lines = [
        json.dumps({"schema": B2B_SEQUENTIAL_TRACE_SCHEMA}, sort_keys=True),
    ]
    for step_index in range(1, steps + 1):
        rows = [
            _within_tie_band_row(f"s{step_index:04d}:c{idx}", idx, -0.30 + 0.05 * idx)
            for idx in range(candidates_per_step)
        ]
        canonical = _canonical_b2b_rows(rows)
        lines.append(
            json.dumps(
                {
                    "source_kind": SOURCE_KIND_WITHIN_TIE_BAND_DISCRIMINATOR,
                    "optimizer_step_index": step_index,
                    "pre_update_state_hash": f"pre{step_index:04d}",
                    "source_table_hash": _stable_hash16(canonical),
                    "sampled_candidate_table": canonical,
                    "post_update_telemetry": {"q_changed_count": 1},
                },
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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
    assert readiness["ready_for_pre_full_stack_diagnostic"] is True
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


def test_real_activation_credit_replay_is_static_only_and_hashes_sources(
    tmp_path: Path,
) -> None:
    source_a = _write_receipt(
        tmp_path / "seed43.json",
        mode=SOURCE_KIND_ACTIVATION_CREDIT_MEASUREMENT,
        rows=[
            _activation_row("a", 0, -0.30),
            _activation_row("b", 1, -0.20),
        ],
    )
    source_b = _write_receipt(
        tmp_path / "seed29.json",
        mode=SOURCE_KIND_ACTIVATION_CREDIT_MEASUREMENT,
        rows=[
            _activation_row("c", 0, -0.25),
            _activation_row("d", 1, -0.10),
        ],
    )

    receipt = run_real_table_static_proxy_replay_from_paths(
        [source_a, source_b],
        stable_copy_dir=tmp_path / "stable",
    )

    assert receipt["receipt_kind"] == REAL_TABLE_REPLAY_RECEIPT_KIND
    assert receipt["trace_temporality"] == TRACE_TEMPORALITY_STATIC_SNAPSHOT
    assert receipt["tracking_scope"] == TRACKING_SCOPE_SNAPSHOT_SCREEN
    assert receipt["dynamics_verdict_allowed"] is False
    assert receipt["primary_label"] == LABEL_SCREEN_HARNESS_OR_GATE_FAIL
    assert receipt["primary_label"] != LABEL_ACCUMULATOR_TRACKS_INT16_POLICY
    assert receipt["static_proxy_label"] == LABEL_STATIC_TRANSIENT_PROXY_AVAILABLE
    assert LABEL_STATIC_PROXY_NOT_PERSISTENT_DYNAMICS in receipt["failure_reasons"]
    assert FAIL_ACCUMULATOR_FIELDS_UNAVAILABLE in receipt["failure_reasons"]
    assert "stream_steps" not in receipt
    replay = receipt["real_table_replay"]
    assert replay["source_kinds"] == [SOURCE_KIND_ACTIVATION_CREDIT_MEASUREMENT]
    assert replay["table_count"] == 2
    assert replay["aggregated_snapshot_steps"] == 2
    assert replay["arm_availability"][ARM_ACCUMULATOR_ONLY] is False
    assert replay["arm_availability"][ARM_TRANSIENT_RESOLVER_ONLY] is True
    assert set(receipt["candidate_stream_hashes_by_arm"].values()) == {
        receipt["candidate_stream_hash"]
    }
    assert receipt["aggregate_metrics"]["candidate_table_count"] == 2
    assert receipt["aggregate_metrics"]["aggregated_snapshot_steps"] == 2
    for record in replay["source_records"]:
        assert record["ephemeral_source"] is True
        assert record["source_hash"] == record["copied_hash"]
        assert record["copied_path"]


def test_real_replay_missing_table_fails_closed(tmp_path: Path) -> None:
    receipt = run_real_table_static_proxy_replay_from_paths(
        [tmp_path / "missing.json"],
        stable_copy_dir=tmp_path / "stable",
    )

    assert receipt["screen_harness_or_gate_fail"] is True
    assert receipt["primary_label"] == LABEL_SCREEN_HARNESS_OR_GATE_FAIL
    assert FAIL_NO_REAL_CANDIDATE_TABLE in receipt["failure_reasons"]
    assert receipt["real_table_replay"]["table_count"] == 0
    assert receipt["dynamics_verdict_allowed"] is False


def test_real_replay_missing_transient_fields_fails_closed(tmp_path: Path) -> None:
    source = _write_receipt(
        tmp_path / "within.json",
        mode=SOURCE_KIND_WITHIN_TIE_BAND_DISCRIMINATOR,
        rows=[
            _within_tie_band_row("a", 0, -0.30),
            _within_tie_band_row("b", 1, -0.20),
        ],
    )

    receipt = run_real_table_static_proxy_replay_from_paths(
        [source],
        stable_copy_dir=tmp_path / "stable",
    )

    assert receipt["screen_harness_or_gate_fail"] is True
    assert FAIL_TRANSIENT_FIELDS_UNAVAILABLE in receipt["failure_reasons"]
    assert receipt["real_table_replay"]["arm_availability"][ARM_ACCUMULATOR_ONLY] is True
    assert (
        receipt["real_table_replay"]["arm_availability"][ARM_TRANSIENT_RESOLVER_ONLY]
        is False
    )
    assert receipt["primary_label"] != LABEL_ACCUMULATOR_TRACKS_INT16_POLICY


def test_real_replay_missing_accumulator_fields_marks_arm_unavailable(
    tmp_path: Path,
) -> None:
    source = _write_receipt(
        tmp_path / "activation.json",
        mode=SOURCE_KIND_ACTIVATION_CREDIT_MEASUREMENT,
        rows=[
            _activation_row("a", 0, -0.30),
            _activation_row("b", 1, -0.20),
        ],
    )

    receipt = run_real_table_static_proxy_replay_from_paths(
        [source],
        stable_copy_dir=tmp_path / "stable",
    )

    assert FAIL_ACCUMULATOR_FIELDS_UNAVAILABLE in receipt["failure_reasons"]
    assert receipt["real_table_replay"]["arm_availability"][ARM_ACCUMULATOR_ONLY] is False
    assert receipt["static_proxy_label"] == LABEL_STATIC_TRANSIENT_PROXY_AVAILABLE
    assert receipt["dynamics_verdict_allowed"] is False


def test_real_replay_rejects_multi_mode_fusion_without_alignment_proof(
    tmp_path: Path,
) -> None:
    activation = _write_receipt(
        tmp_path / "activation.json",
        mode=SOURCE_KIND_ACTIVATION_CREDIT_MEASUREMENT,
        rows=[_activation_row("a", 0, -0.30)],
    )
    within = _write_receipt(
        tmp_path / "within.json",
        mode=SOURCE_KIND_WITHIN_TIE_BAND_DISCRIMINATOR,
        rows=[_within_tie_band_row("a", 0, -0.30)],
    )

    receipt = run_real_table_static_proxy_replay_from_paths(
        [activation, within],
        stable_copy_dir=tmp_path / "stable",
    )

    assert receipt["screen_harness_or_gate_fail"] is True
    assert FAIL_MULTI_SOURCE_FUSION_REJECTED in receipt["failure_reasons"]
    assert receipt["real_table_replay"]["table_count"] == 0
    assert (
        receipt["real_table_replay"][
            "multi_source_fusion_rejected_without_alignment_proof"
        ]
        is True
    )


def test_dynamics_synthetic_harness_never_emits_temporal_tracking_label() -> None:
    receipt = run_accumulator_policy_dynamics_screen(steps=50)

    assert receipt["schema_version"] == ACCUMULATOR_POLICY_SHADOW_SCREEN_SCHEMA_VERSION_DYNAMICS
    assert receipt["receipt_kind"] == SYNTHETIC_DYNAMICS_RECEIPT_KIND
    assert receipt["harness_validation_only"] is True
    assert receipt["trace_temporality"] == TRACE_TEMPORALITY_SEQUENTIAL_OPTIMIZER_STEPS
    assert receipt["tracking_scope"] == TRACKING_SCOPE_OPTIMIZER_STEP_TRAJECTORY
    assert receipt["dynamics_verdict_allowed"] is False
    assert receipt["primary_label"] != LABEL_ACCUMULATOR_TRACKS_INT16_POLICY
    assert receipt["diagnostic_contract"]["satisfied"] is True


def test_dynamics_state_carry_isolation_forces_shadow_not_row_only_pick() -> None:
    stream = build_synthetic_temporal_within_tie_band_stream(
        steps=50,
        candidates_per_step=2,
        mode="state_carry_isolation",
    )
    isolation_step = stream[-1]
    candidates = isolation_step["candidates"]
    carried_state = run_accumulator_policy_dynamics_screen(
        candidate_stream=stream[:-1],
        steps=49,
    )["arms"][ARM_ACCUMULATOR_ONLY]["final_shadow_state"]
    carried_i32 = int(carried_state["carried_accumulator_i32"])

    row_only_pick = _row_only_selection_at_step(candidates, rate_cap=1)[0]
    dynamics_pick = _select_dynamics_candidate(
        candidates,
        arm=ARM_ACCUMULATOR_ONLY,
        shadow_state=_DynamicsArmShadowState(carried_accumulator_i32=carried_i32),
        rate_cap=1,
    )[0]

    assert row_only_pick == "s0049:c0"
    assert dynamics_pick == "s0049:c1"
    assert _row_only_flip_policy_score(candidates[0]) > _row_only_flip_policy_score(
        candidates[1]
    )
    assert _dynamics_flip_policy_score(
        candidates[1],
        carried_accumulator_i32=carried_i32,
    ) > _dynamics_flip_policy_score(
        candidates[0],
        carried_accumulator_i32=carried_i32,
    )

    receipt = run_accumulator_policy_dynamics_screen(candidate_stream=stream, steps=50)
    assert receipt["arms"][ARM_ACCUMULATOR_ONLY]["final_shadow_state"][
        "carried_accumulator_i32"
    ] == 18


def test_dynamics_classifier_taxonomy_on_synthetic_modes() -> None:
    expected_by_mode = {
        "accumulator_tracks": LABEL_ACCUMULATOR_TRACKS_INT16_POLICY,
        "transient_carries": LABEL_TRANSIENT_CARRIES_SELECTION,
        "accumulator_null": LABEL_ACCUMULATOR_IMPROVES_BUT_NOT_ENOUGH,
    }
    observed: dict[str, str] = {}
    for mode, expected in expected_by_mode.items():
        receipt = run_accumulator_policy_dynamics_screen(
            steps=50,
            synthetic_mode=mode,
        )
        observed[mode] = receipt["raw_primary_label"]
        assert receipt["raw_primary_label"] == expected

    assert len(set(observed.values())) == 3
    transient = run_accumulator_policy_dynamics_screen(
        steps=50,
        synthetic_mode="transient_carries",
    )
    assert transient["aggregate_metrics"]["transient_only_advantage_vs_accumulator"] > (
        transient["thresholds"]["max_transient_only_advantage_allowed"]
    )
    tracks = run_accumulator_policy_dynamics_screen(
        steps=50,
        synthetic_mode="accumulator_tracks",
    )
    assert tracks["primary_label"] == LABEL_ACCUMULATOR_IMPROVES_BUT_NOT_ENOUGH


def test_real_sequential_replay_fail_closed_until_b2b_capture(tmp_path: Path) -> None:
    source = _write_receipt(
        tmp_path / "within.json",
        mode=SOURCE_KIND_WITHIN_TIE_BAND_DISCRIMINATOR,
        rows=[
            _within_tie_band_row("a", 0, -0.30),
            _within_tie_band_row("b", 1, -0.20),
        ],
    )

    receipt = run_real_sequential_trace_replay_from_paths(
        [source],
        stable_copy_dir=tmp_path / "stable",
    )

    assert receipt["receipt_kind"] == REAL_SEQUENTIAL_TRACE_REPLAY_RECEIPT_KIND
    assert receipt["dynamics_verdict_allowed"] is False
    assert receipt["primary_label"] == LABEL_SCREEN_HARNESS_OR_GATE_FAIL
    assert FAIL_NO_REAL_SEQUENTIAL_CAPTURE in receipt["failure_reasons"]
    assert receipt["real_sequential_replay"]["no_real_sequential_capture"] is True
    assert receipt["aggregate_metrics"]["optimizer_step_count"] == 0


def test_real_sequential_replay_accepts_valid_fifty_step_trace(tmp_path: Path) -> None:
    trace = _write_b2b_sequential_trace(tmp_path / "trace.ndjson", steps=50)

    receipt = run_real_sequential_trace_replay_from_paths(
        [trace],
        stable_copy_dir=tmp_path / "stable",
    )

    assert receipt["receipt_kind"] == REAL_SEQUENTIAL_TRACE_REPLAY_RECEIPT_KIND
    assert receipt["proof_side"] == REAL_SEQUENTIAL_TRACE_PROOF_SIDE
    assert receipt["real_sequential_replay"]["no_real_sequential_capture"] is False
    assert receipt["aggregate_metrics"]["optimizer_step_count"] == 50
    assert receipt["harness_validation_only"] is False


def test_real_sequential_replay_rejects_short_trace(tmp_path: Path) -> None:
    trace = _write_b2b_sequential_trace(tmp_path / "short.ndjson", steps=10)

    receipt = run_real_sequential_trace_replay_from_paths(
        [trace],
        stable_copy_dir=tmp_path / "stable",
    )

    assert receipt["dynamics_verdict_allowed"] is False
    assert FAIL_INSUFFICIENT_OPTIMIZER_STEPS in receipt["failure_reasons"]


def test_real_sequential_trace_parser_rejects_non_monotonic_index(tmp_path: Path) -> None:
    trace = _write_b2b_sequential_trace(tmp_path / "trace.ndjson", steps=3)
    lines = trace.read_text(encoding="utf-8").splitlines()
    lines[2], lines[3] = lines[3], lines[2]
    trace.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _stream, _metadata, failures = build_real_sequential_trace_candidate_stream([trace])

    assert _stream == ()
    assert FAIL_NON_MONOTONIC_STEP_INDEX in failures


def test_real_sequential_replay_rejects_stitched_independent_traces(
    tmp_path: Path,
) -> None:
    seed43 = _write_b2b_sequential_trace(tmp_path / "seed43.ndjson", steps=50)
    seed29 = _write_b2b_sequential_trace(tmp_path / "seed29.ndjson", steps=50)

    receipt = run_real_sequential_trace_replay_from_paths(
        [seed43, seed29],
        stable_copy_dir=tmp_path / "stable",
    )

    assert receipt["dynamics_verdict_allowed"] is False
    assert receipt["primary_label"] == LABEL_SCREEN_HARNESS_OR_GATE_FAIL
    assert (
        FAIL_MULTIPLE_SEQUENTIAL_TRACES_REJECTED_WITHOUT_ALIGNMENT_PROOF
        in receipt["failure_reasons"]
    )
    assert receipt["aggregate_metrics"]["optimizer_step_count"] == 0


def test_compact_within_tie_band_table_requires_accumulator_fields() -> None:
    from calm.hrm_text_158.native_full_stack.oracle_screen_runner import (
        build_compact_within_tie_band_sampled_table_rows,
        canonical_within_tie_band_rows_for_b2b_hash,
    )

    row = {
        **_within_tie_band_row("a", 0, -0.30),
        "abs_vote_value": 1,
        "flat_index": 0,
        "state_key": "layer.weight",
        "candidate_loss": 1.0,
        "regret_vs_target_tie_band_oracle_top1_local_loss_delta": None,
    }
    table = build_compact_within_tie_band_sampled_table_rows([row])
    canonical = canonical_within_tie_band_rows_for_b2b_hash(table)
    assert canonical[0]["pre_accumulator_i16"] == 0
    assert canonical[0]["new_acc_i32_signed"] == 20
    assert canonical[0]["proximity_to_threshold"] == 0

    incomplete = dict(row)
    del incomplete["pre_accumulator_i16"]
    with pytest.raises(KeyError):
        build_compact_within_tie_band_sampled_table_rows([incomplete])


def test_dynamics_contract_fail_closed_on_stream_drift() -> None:
    receipt = run_accumulator_policy_dynamics_screen(
        steps=50,
        candidate_stream_hash_overrides={ARM_ACCUMULATOR_ONLY: "drifted-stream"},
    )

    assert receipt["screen_harness_or_gate_fail"] is True
    assert receipt["primary_label"] == LABEL_SCREEN_HARNESS_OR_GATE_FAIL
    assert "same_candidate_stream" in receipt["failure_reasons"]
