"""Focused tests for the strict sub-2 native-birth scaffold report."""
from __future__ import annotations

from dataclasses import replace

import pytest

import calm.hrm_text_158.native_full_stack as native_full_stack
from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
    ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2,
)
from calm.hrm_text_158.native_full_stack.sub2_native_birth_scaffold import (
    ACQUISITION_GATE_DEFERRED,
    ACQUISITION_GATE_RESULT,
    ACQUISITION_GATE_RUNNING,
    ACQUISITION_GATE_UNBLOCKED_NOT_RUN,
    DENSE_TRANSIENT_CREDIT_ROLE_TRAINING_COMPUTE_CONTROL_ONLY,
    HYBRID_MOVEMENT_CONTRACT_SCOPE,
    HYBRID_MOVEMENT_METRIC_NAME,
    HYBRID_PERSISTENT_MODE_APPLIED_CROSSING_DIRECTION_PLUS_4BIT_RESIDUAL,
    HYBRID_SCOPE_DECISION_LOCKED_ANSWER,
    HYBRID_SCOPE_DECISION_LOCKED_OPTION,
    HYBRID_SCOPE_DECISION_SOURCE_MSG_ID,
    LEDGER_CLASS_LEQ2,
    LEDGER_CLASS_NOT_YET,
    RUNTIME_STATE_AUTHORITY_SUB2_PERSISTENT_HYBRID_DENSE_TRANSIENT_CREDIT,
    RUNTIME_STATE_AUTHORITY_SUB2_SCAFFOLD_ONLY,
    attach_strict_sub2_scoped_candidate_proof,
    build_hybrid_persistent_sidecar_ledger,
    build_strict_sub2_candidate_runtime_scaffold,
    build_strict_sub2_hybrid_runtime_movement_overlay,
    validate_hybrid_persistent_sidecar_ledger,
    validate_strict_sub2_candidate_runtime_scaffold_report,
    validate_strict_sub2_hybrid_runtime_movement_overlay,
)


def _eligible_shapes():
    return {
        "model.H_level.core.layers.0.attn.gqkv_proj": (128, 128),
        "model.H_level.core.layers.0.attn.o_proj": (128, 128),
    }


def _activation_ledger():
    return {
        "schema": "hrm158_activation_paid_bits_ledger/v0.group128_base3_fp32scale",
        "pass": True,
        "surface_count": 3,
        "budget_failure_count": 0,
        "budget_failure_surfaces": [],
        "surfaces": [
            {
                "name": "attn.gqkv.gate",
                "paid_bits_per_value": 1.85,
                "runtime_packable_sub2": True,
                "tail_or_non128_budget_fail": False,
            },
            {
                "name": "residual.post_attn",
                "paid_bits_per_value": 1.85,
                "runtime_packable_sub2": True,
                "tail_or_non128_budget_fail": False,
            },
            {
                "name": "recurrent.z_L_update",
                "paid_bits_per_value": 1.85,
                "runtime_packable_sub2": True,
                "tail_or_non128_budget_fail": False,
            },
        ],
    }


def _live_both_gate():
    return {
        "schema": "hrm158_tierb_live_both_gate_launch_ledger/v0.static_shape_contract",
        "pass": True,
        "not_covered": ["kv_cache.append_update"],
        "not_covered_reason": "slice_1 lacks native KV-cache append/update seam",
        "surface_count": 12,
        "required_surface_count": 12,
    }


def _hot_loop_residency():
    return {
        "schema": "hrm158_tierb_device_vs_hot_loop_residency/v0",
        "device_residency": {
            "model_forward_backward": "cuda",
            "activation_codec": "cuda",
            "qscale_leaf_gradients": "cuda",
        },
        "hot_loop_residency": {
            "model_forward_backward": "cuda",
            "qacc_update_over_64": "cpu_reference",
            "qacc_vote_selection": "cpu_reference",
            "qacc_apply_vote_step": "cpu_reference",
            "gradient_transfer": "cuda_to_cpu",
        },
        "qacc_kernelized": False,
        "qacc_kernelization_status": "not_kernelized_reference_cpu_path_correct_but_slow",
        "not_a_gpu_hot_path_claim": True,
        "next_perf_lever": "kernelize qacc vote selection/apply path after science smoke proves learning",
        "pass": True,
    }


def _build_report():
    return build_strict_sub2_candidate_runtime_scaffold(
        eligible_module_shapes=_eligible_shapes(),
        activation_paid_bits_ledger=_activation_ledger(),
        live_both_gate=_live_both_gate(),
        hot_loop_residency=_hot_loop_residency(),
    )


def _build_hybrid_overlay():
    return build_strict_sub2_hybrid_runtime_movement_overlay(
        logical_shapes=tuple(_eligible_shapes().values()),
        event_counts=(8, 8),
        persistent_mode=HYBRID_PERSISTENT_MODE_APPLIED_CROSSING_DIRECTION_PLUS_4BIT_RESIDUAL,
        residual_bits_per_event=4,
        persistent_dense_shadow_present=False,
        persistent_dense_shadow_bytes=0,
        local_update_law_label=ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2,
    )


def test_strict_sub2_scaffold_is_fail_closed_hybrid_boundary_only():
    report = _build_report()

    validate_strict_sub2_candidate_runtime_scaffold_report(report)

    assert (
        report.runtime_state_authority
        == RUNTIME_STATE_AUTHORITY_SUB2_PERSISTENT_HYBRID_DENSE_TRANSIENT_CREDIT
    )
    assert report.pass_report is True
    assert report.candidate_runtime_complete is False
    assert report.physical_persistent_target_pass is True
    assert report.physical_persistent_bpw < 2.0
    assert report.physical_persistent_interpretation
    assert report.persistent_sub2_hybrid_only is True
    assert report.dense_transient_credit_allowed is True
    assert (
        report.dense_transient_credit_role
        == DENSE_TRANSIENT_CREDIT_ROLE_TRAINING_COMPUTE_CONTROL_ONLY
    )
    assert report.dense_transient_credit_counted_in_physical_persistent_bpw is False
    assert report.transient_debt_present is True
    assert report.transient_debt_non_blocking is True
    assert report.transient_debt_row_names == (
        "activations_and_residual_runtime_packability",
        "attention_kv_append_update",
        "qacc_hot_loop_residency",
    )
    assert report.full_runtime_sub2_achieved is False
    assert report.native_transient_sub2_achieved is False
    assert report.fully_fp_free_achieved is False
    assert report.scope_decision_source_msg_id == HYBRID_SCOPE_DECISION_SOURCE_MSG_ID
    assert report.scope_decision_locked_option == HYBRID_SCOPE_DECISION_LOCKED_OPTION
    assert report.scope_decision_locked_answer == HYBRID_SCOPE_DECISION_LOCKED_ANSWER
    assert report.acquisition_science_status == ACQUISITION_GATE_UNBLOCKED_NOT_RUN
    assert report.acquisition_achieved is False
    assert report.acquisition_gate["status"] == ACQUISITION_GATE_UNBLOCKED_NOT_RUN

    persistent = {row.name: row for row in report.persistent_candidate_rows}
    assert persistent["q_storage"].classification == LEDGER_CLASS_LEQ2
    assert persistent["q_storage"].in_candidate_authority is True
    assert persistent["q_storage"].counted_in_physical_persistent_bpw is True
    assert persistent["accumulator_substitute"].classification == LEDGER_CLASS_NOT_YET
    assert persistent["accumulator_substitute"].blocker is True
    assert persistent["accumulator_substitute"].in_candidate_authority is False

    off_path = {row.name: row for row in report.off_path_control_rows}
    assert off_path["frozen_scales_fp32_metadata"].classification == LEDGER_CLASS_NOT_YET
    assert off_path["dense_int16_accumulator_control"].bits_per_weight == pytest.approx(16.0)

    adjacent = {row.name: row for row in report.adjacent_runtime_rows}
    assert adjacent["activations_and_residual_runtime_packability"].classification == LEDGER_CLASS_LEQ2
    assert adjacent["attention_kv_append_update"].classification == LEDGER_CLASS_NOT_YET
    assert adjacent["qacc_hot_loop_residency"].classification == LEDGER_CLASS_NOT_YET

    assert "justified_fp_exception" not in str(report.to_dict())
    assert "accumulator_substitute" in report.blocker_names
    assert "attention_kv_append_update" in report.blocker_names
    assert "qacc_hot_loop_residency" in report.blocker_names


def test_strict_sub2_scaffold_validator_rejects_blocked_candidate_authority_row():
    report = _build_report()
    q_storage = report.persistent_candidate_rows[0]
    bad_row = replace(
        q_storage,
        classification=LEDGER_CLASS_NOT_YET,
        in_candidate_authority=True,
        counted_in_physical_persistent_bpw=True,
        blocker=True,
    )
    bad_report = replace(
        report,
        persistent_candidate_rows=(bad_row,) + report.persistent_candidate_rows[1:],
    )

    with pytest.raises(ValueError, match="cannot be in candidate authority while blocked"):
        validate_strict_sub2_candidate_runtime_scaffold_report(bad_report)


def test_native_full_stack_public_exports_include_strict_sub2_scaffold_symbols():
    names = {
        "ACQUISITION_GATE_DEFERRED",
        "ACQUISITION_GATE_UNBLOCKED_NOT_RUN",
        "DENSE_TRANSIENT_CREDIT_ROLE_TRAINING_COMPUTE_CONTROL_ONLY",
        "HYBRID_SCOPE_DECISION_LOCKED_ANSWER",
        "HYBRID_SCOPE_DECISION_LOCKED_OPTION",
        "HYBRID_SCOPE_DECISION_SOURCE_MSG_ID",
        "LEDGER_CLASS_EXECUTABLE",
        "LEDGER_CLASS_LEQ2",
        "LEDGER_CLASS_NOT_YET",
        "RUNTIME_STATE_AUTHORITY_DENSE_CONTROL",
        "RUNTIME_STATE_AUTHORITY_SUB2_CANDIDATE_EXECUTABLE",
        "RUNTIME_STATE_AUTHORITY_SUB2_PERSISTENT_HYBRID_DENSE_TRANSIENT_CREDIT",
        "RUNTIME_STATE_AUTHORITY_SUB2_SCAFFOLD_ONLY",
        "StrictSub2CandidateRuntimeScaffoldReport",
        "StrictSub2ScaffoldRow",
        "attach_strict_sub2_scoped_candidate_proof",
        "build_strict_sub2_candidate_runtime_scaffold",
        "validate_strict_sub2_candidate_runtime_scaffold_report",
    }

    exported = set(native_full_stack.__all__)
    for name in names:
        assert hasattr(native_full_stack, name)
        assert name in exported


def test_scaffold_accepts_scoped_accumulator_local_vote_update_proof_without_full_promotion():
    report = _build_report()
    scoped = {
        "surface": "accumulator_substitute",
        "scoped_label": ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2,
        "terminal_classification": ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2,
        "pass": True,
        "runtime_state_authority_after": RUNTIME_STATE_AUTHORITY_SUB2_SCAFFOLD_ONLY,
        "candidate_dense_decode_used": False,
        "candidate_accumulator_transient_over2_used": False,
        "candidate_vote_transient_over2_used": False,
        "candidate_dense_vote_authority_used": False,
        "scoped_physical_budget_claim": "algorithmic_only_not_physical_sub2",
        "q_storage_physical_budget_covered_by_scoped_proof": False,
        "frozen_scale_physical_budget_covered_by_scoped_proof": False,
        "coverage_domain": {
            "no_global_cap": True,
            "sparse_vote_events_only": True,
        },
        "storage_projection": {
            "bounded_delta_acc_bits_per_weight": 10.0,
        },
    }

    promoted = attach_strict_sub2_scoped_candidate_proof(
        report,
        scoped_candidate_proof=scoped,
    )

    validate_strict_sub2_candidate_runtime_scaffold_report(promoted)
    assert (
        promoted.runtime_state_authority
        == RUNTIME_STATE_AUTHORITY_SUB2_PERSISTENT_HYBRID_DENSE_TRANSIENT_CREDIT
    )
    assert promoted.candidate_runtime_complete is False
    assert (
        promoted.scoped_candidate_proof["scoped_label"]
        == ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2
    )
    assert (
        promoted.scoped_candidate_proof["runtime_state_authority_after"]
        == promoted.runtime_state_authority
    )


def test_hybrid_sidecar_ledger_reports_explicit_4bit_codec_budget():
    ledger = build_hybrid_persistent_sidecar_ledger(
        logical_shapes=tuple(_eligible_shapes().values()),
        event_counts=(8, 8),
        persistent_mode=HYBRID_PERSISTENT_MODE_APPLIED_CROSSING_DIRECTION_PLUS_4BIT_RESIDUAL,
        residual_bits_per_event=4,
    )

    validate_hybrid_persistent_sidecar_ledger(ledger)

    assert ledger.persistent_mode == HYBRID_PERSISTENT_MODE_APPLIED_CROSSING_DIRECTION_PLUS_4BIT_RESIDUAL
    assert ledger.direction_bits_per_event == 1
    assert ledger.residual_bits_per_event == 4
    assert ledger.total_event_count == 16
    assert ledger.total_metadata_bits > 0
    assert ledger.index_bits_kind == "per_tensor_local"
    assert ledger.packet_count_bits_formula == "ceil(log2(numel + 1)) per tensor"
    assert ledger.row_only_lt2 is True
    assert ledger.inclusive_lt2 is True
    assert ledger.inclusive_bits_per_weight == pytest.approx(
        ledger.q_bits_per_weight
        + ledger.frozen_scale_bits_per_weight
        + ledger.sidecar_bits_per_weight
    )
    assert len(ledger.shape_breakdown) == 2
    assert all(int(row["index_bits"]) > 0 for row in ledger.shape_breakdown)
    assert all(int(row["packet_count_bits"]) > 0 for row in ledger.shape_breakdown)


def test_hybrid_runtime_overlay_is_fail_closed_and_non_overclaiming():
    report = _build_hybrid_overlay()

    validate_strict_sub2_hybrid_runtime_movement_overlay(report)

    assert report.runtime_state_authority == (
        RUNTIME_STATE_AUTHORITY_SUB2_PERSISTENT_HYBRID_DENSE_TRANSIENT_CREDIT
    )
    assert report.persistent_mode == HYBRID_PERSISTENT_MODE_APPLIED_CROSSING_DIRECTION_PLUS_4BIT_RESIDUAL
    assert report.pass_report is True
    assert report.persistent_dense_shadow_present is False
    assert report.persistent_dense_shadow_bytes == 0
    assert report.bounded_only_collapse is True
    assert report.local_update_law_reused is True
    assert report.second_update_law_required is False
    assert report.dense_transient_credit_allowed is True
    assert (
        report.dense_transient_credit_role
        == DENSE_TRANSIENT_CREDIT_ROLE_TRAINING_COMPUTE_CONTROL_ONLY
    )
    assert report.transient_debt_present is True
    assert report.persistent_sub2_hybrid_only is True
    assert report.full_runtime_sub2_achieved is False
    assert report.candidate_runtime_complete is False
    assert report.acquisition_science_status == ACQUISITION_GATE_UNBLOCKED_NOT_RUN
    assert report.acquisition_achieved is False
    assert report.movement_contract_scope == HYBRID_MOVEMENT_CONTRACT_SCOPE
    assert report.movement_metric_name == HYBRID_MOVEMENT_METRIC_NAME
    assert report.movement_metric_min_delta == 1
    assert report.q_changed_must_be_positive is True
    assert report.hard_fail_required_false is True
    assert report.persistent_authority_row_names == (
        "q_storage",
        "frozen_scales_fp32_metadata",
        "accumulator_sidecar",
    )
    assert "accumulator_substitute" in report.blocked_row_names
    assert "attention_kv_append_update" in report.blocked_row_names
    assert "qacc_hot_loop_residency" in report.blocked_row_names
    ledger = report.persistent_sidecar_ledger
    assert ledger["inclusive_lt2"] is True
    assert ledger["residual_bits_per_event"] == 4
    assert ledger["total_event_count"] == 16


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("persistent_dense_shadow_present", True, "cannot persist dense shadow state"),
        ("full_runtime_sub2_achieved", True, "cannot claim full_runtime_sub2_achieved"),
        ("transient_debt_present", False, "must disclose transient_debt_present=true"),
        ("candidate_runtime_complete", True, "cannot claim candidate_runtime_complete"),
    ],
)
def test_hybrid_runtime_overlay_validator_rejects_overclaim_and_missing_debt(field, value, match):
    report = _build_hybrid_overlay()
    bad_report = replace(report, **{field: value})

    with pytest.raises(ValueError, match=match):
        validate_strict_sub2_hybrid_runtime_movement_overlay(bad_report)


def test_scaffold_rejects_scoped_candidate_proof_with_dense_decode_or_wrong_null_taxonomy():
    report = _build_report()
    bad_dense = {
        "surface": "accumulator_substitute",
        "scoped_label": ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        "terminal_classification": ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        "pass": True,
        "runtime_state_authority_after": RUNTIME_STATE_AUTHORITY_SUB2_PERSISTENT_HYBRID_DENSE_TRANSIENT_CREDIT,
        "candidate_dense_decode_used": True,
        "candidate_accumulator_transient_over2_used": False,
        "candidate_vote_transient_over2_used": False,
        "candidate_dense_vote_authority_used": False,
        "scoped_physical_budget_claim": "physical_sub2_budgeted",
        "q_storage_physical_budget_covered_by_scoped_proof": False,
        "frozen_scale_physical_budget_covered_by_scoped_proof": False,
        "coverage_domain": {"no_global_cap": True},
        "storage_projection": {"bounded_delta_acc_bits_per_weight": 1.0},
    }
    bad_null = dict(bad_dense)
    bad_null.update(
        {
            "candidate_dense_decode_used": False,
            "pass": False,
            "scoped_label": None,
            "terminal_classification": "not_a_real_null",
        }
    )

    with pytest.raises(ValueError, match="dense decode"):
        attach_strict_sub2_scoped_candidate_proof(
            report,
            scoped_candidate_proof=bad_dense,
        )
    with pytest.raises(ValueError, match="intrinsic domain-gap null"):
        attach_strict_sub2_scoped_candidate_proof(
            report,
            scoped_candidate_proof=bad_null,
        )


def test_scaffold_rejects_positive_scoped_candidate_without_storage_projection_or_with_bpw_ge_2():
    report = _build_report()
    missing_projection = {
        "surface": "accumulator_substitute",
        "scoped_label": ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        "terminal_classification": ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        "pass": True,
        "runtime_state_authority_after": RUNTIME_STATE_AUTHORITY_SUB2_PERSISTENT_HYBRID_DENSE_TRANSIENT_CREDIT,
        "candidate_dense_decode_used": False,
        "candidate_accumulator_transient_over2_used": False,
        "candidate_vote_transient_over2_used": False,
        "candidate_dense_vote_authority_used": False,
        "scoped_physical_budget_claim": "physical_sub2_budgeted",
        "q_storage_physical_budget_covered_by_scoped_proof": False,
        "frozen_scale_physical_budget_covered_by_scoped_proof": False,
        "coverage_domain": {"no_global_cap": True},
    }
    bad_bpw = dict(missing_projection)
    bad_bpw["storage_projection"] = {"bounded_delta_acc_bits_per_weight": 2.0}

    with pytest.raises(ValueError, match="must disclose storage_projection"):
        attach_strict_sub2_scoped_candidate_proof(
            report,
            scoped_candidate_proof=missing_projection,
        )
    with pytest.raises(ValueError, match="storage_projection bounded_delta_acc_bits_per_weight < 2"):
        attach_strict_sub2_scoped_candidate_proof(
            report,
            scoped_candidate_proof=bad_bpw,
        )


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("full_runtime_sub2_achieved", True, "full_runtime_sub2_achieved"),
        ("native_transient_sub2_achieved", True, "native_transient_sub2_achieved"),
        ("fully_fp_free_achieved", True, "fully_fp_free_achieved"),
        (
            "dense_transient_credit_counted_in_physical_persistent_bpw",
            True,
            "cannot count dense transient credit",
        ),
        ("candidate_runtime_complete", True, "candidate_runtime_complete=false"),
        ("acquisition_achieved", True, "acquisition_achieved=true"),
    ],
)
def test_hybrid_validator_rejects_overclaim_flags(field_name: str, value, match: str):
    report = _build_report()
    bad_report = replace(report, **{field_name: value})

    with pytest.raises(ValueError, match=match):
        validate_strict_sub2_candidate_runtime_scaffold_report(bad_report)


def test_hybrid_validator_rejects_missing_transient_debt_rows_or_scope_provenance():
    report = _build_report()

    with pytest.raises(ValueError, match="transient_debt_row_names"):
        validate_strict_sub2_candidate_runtime_scaffold_report(
            replace(report, transient_debt_row_names=())
        )
    with pytest.raises(ValueError, match="scope decision source msg id"):
        validate_strict_sub2_candidate_runtime_scaffold_report(
            replace(report, scope_decision_source_msg_id="")
        )


def test_hybrid_validator_rejects_off_path_row_counted_in_persistent_bpw():
    report = _build_report()
    off_path = report.off_path_control_rows[0]
    bad_off_path = replace(
        off_path,
        in_candidate_authority=True,
        counted_in_physical_persistent_bpw=True,
        bits_per_weight=1.0,
        blocker=False,
    )
    bad_report = replace(
        report,
        off_path_control_rows=(bad_off_path,) + report.off_path_control_rows[1:],
    )

    with pytest.raises(ValueError, match="off-path or adjacent runtime sections"):
        validate_strict_sub2_candidate_runtime_scaffold_report(bad_report)


def test_hybrid_validator_rejects_acquisition_claims_or_deferred_status_regression():
    report = _build_report()

    with pytest.raises(ValueError, match="acquisition_science_status=unblocked_not_run"):
        validate_strict_sub2_candidate_runtime_scaffold_report(
            replace(report, acquisition_science_status=ACQUISITION_GATE_DEFERRED)
        )
    with pytest.raises(ValueError, match="acquisition gate at unblocked_not_run"):
        validate_strict_sub2_candidate_runtime_scaffold_report(
            replace(report, acquisition_gate={**report.acquisition_gate, "status": ACQUISITION_GATE_DEFERRED})
        )
