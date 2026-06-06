"""Focused tests for the strict sub-2 native-birth scaffold report."""
from __future__ import annotations

from dataclasses import replace

import pytest

import calm.hrm_text_158.native_full_stack as native_full_stack
from calm.hrm_text_158.native_full_stack.sub2_native_birth_scaffold import (
    ACQUISITION_GATE_DEFERRED,
    LEDGER_CLASS_LEQ2,
    LEDGER_CLASS_NOT_YET,
    RUNTIME_STATE_AUTHORITY_SUB2_SCAFFOLD_ONLY,
    build_strict_sub2_candidate_runtime_scaffold,
    validate_strict_sub2_candidate_runtime_scaffold_report,
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


def test_strict_sub2_scaffold_is_fail_closed_and_scaffold_only():
    report = build_strict_sub2_candidate_runtime_scaffold(
        eligible_module_shapes=_eligible_shapes(),
        activation_paid_bits_ledger=_activation_ledger(),
        live_both_gate=_live_both_gate(),
        hot_loop_residency=_hot_loop_residency(),
    )

    validate_strict_sub2_candidate_runtime_scaffold_report(report)

    assert report.runtime_state_authority == RUNTIME_STATE_AUTHORITY_SUB2_SCAFFOLD_ONLY
    assert report.pass_report is True
    assert report.candidate_runtime_complete is False
    assert report.physical_persistent_target_pass is True
    assert report.physical_persistent_bpw < 2.0
    assert report.acquisition_gate["status"] == ACQUISITION_GATE_DEFERRED

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
    report = build_strict_sub2_candidate_runtime_scaffold(
        eligible_module_shapes=_eligible_shapes(),
        activation_paid_bits_ledger=_activation_ledger(),
        live_both_gate=_live_both_gate(),
        hot_loop_residency=_hot_loop_residency(),
    )
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
        "LEDGER_CLASS_EXECUTABLE",
        "LEDGER_CLASS_LEQ2",
        "LEDGER_CLASS_NOT_YET",
        "RUNTIME_STATE_AUTHORITY_DENSE_CONTROL",
        "RUNTIME_STATE_AUTHORITY_SUB2_CANDIDATE_EXECUTABLE",
        "RUNTIME_STATE_AUTHORITY_SUB2_SCAFFOLD_ONLY",
        "STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_LABEL",
        "STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_SCHEMA_VERSION",
        "STRICT_SUB2_CANDIDATE_RUNTIME_TARGET_NAME",
        "StrictSub2CandidateRuntimeScaffoldReport",
        "StrictSub2ScaffoldRow",
        "build_strict_sub2_candidate_runtime_scaffold",
        "validate_strict_sub2_candidate_runtime_scaffold_report",
    }

    exported = set(native_full_stack.__all__)
    for name in names:
        assert hasattr(native_full_stack, name)
        assert name in exported
