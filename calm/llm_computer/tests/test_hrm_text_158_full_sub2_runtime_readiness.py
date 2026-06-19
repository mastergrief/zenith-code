"""Focused tests for the full-sub2 runtime readiness gate."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import hashlib
import subprocess
import sys

import pytest

import calm.hrm_text_158.native_full_stack as native_full_stack
from calm.hrm_text_158.native_full_stack.full_sub2_runtime_readiness import (
    FIXTURE_CURRENT_REPO,
    FIXTURE_GATED_SUB2_CHECKPOINT_PATH,
    FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ACTIVATION_RESIDUALS_BLOCKED,
    FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ATTENTION_KV_BLOCKED,
    FIXTURE_GATED_SUB2_CHECKPOINT_PATH_BACKWARD_RECOMPUTE,
    FIXTURE_GATED_SUB2_CHECKPOINT_PATH_NATIVE_KERNELIZED_HOT_PATH_BLOCKED,
    FIXTURE_GATED_SUB2_CHECKPOINT_PATH_OPTIMIZER_CREDIT_STATE_BLOCKED,
    FIXTURE_LIVE_P1_AUTHORITY_CONVERSION,
    FIXTURE_MAIN_READY,
    FIXTURE_MISSING_ACTIVATIONS,
    FIXTURE_MISSING_ATTENTION,
    FIXTURE_MISSING_BACKWARD,
    FIXTURE_PRE_FULL_STACK_DIAGNOSTIC,
    FIXTURE_STEP2A_CANDIDATE_PERSISTENT_CORE_ABSENCE,
    FIXTURE_TRANSIENT_FP_DEBT,
    FULL_SUB2_RUNTIME_CLASSIFICATIONS,
    FULL_SUB2_RUNTIME_REQUIRED_SURFACES,
    RUNTIME_CLASS_EXPLICIT_EXCEPTION,
    RUNTIME_CLASS_MISSING,
    RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
    RUNTIME_CLASS_SUB2,
    RUNTIME_CLASS_TRANSIENT_FP_DEBT,
    SURFACE_ACTIVATIONS_RESIDUALS,
    SURFACE_ATTENTION_KV_ATTENTION_BUFFERS,
    SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS,
    SURFACE_DENSE_INT16_PERSISTENT_ACCUMULATOR_ABSENCE,
    SURFACE_FP_EXCEPTIONS_LEDGER,
    SURFACE_NATIVE_KERNELIZED_HOT_PATH,
    SURFACE_OPTIMIZER_CREDIT_STATE,
    SURFACE_PERSISTENT_QACC_AUTHORITY,
    SURFACE_Q_SIDECAR_VOTE_CARRIER,
    FullSub2RuntimeSurfaceReceipt,
    apply_live_p1_conversion_surface_overrides,
    apply_live_r1_backward_wiring_surface_overrides,
    apply_live_activation_residuals_surface_overrides,
    build_full_sub2_runtime_ready_for_science,
    current_repo_scaffold_surfaces,
    fixture_full_sub2_runtime_ready_for_science,
    gated_sub2_checkpoint_path_activation_residuals_blocked_surfaces,
    gated_sub2_checkpoint_path_attention_kv_blocked_surfaces,
    gated_sub2_checkpoint_path_backward_recompute_surfaces,
    gated_sub2_checkpoint_path_native_kernelized_hot_path_blocked_surfaces,
    gated_sub2_checkpoint_path_optimizer_credit_state_blocked_surfaces,
    gated_sub2_checkpoint_path_surfaces,
    live_p1_authority_conversion_surfaces,
    live_r1_backward_wiring_surfaces,
    main_ready_fixture_surfaces,
    validate_full_sub2_runtime_ready_for_science_receipt,
)
from calm.hrm_text_158.native_full_stack.activation_relief import (
    AUTHORIZED_R1_L_SURFACE_TUPLE,
    PROOF_KIND_LAUNCH_RUNTIME_VALIDATION,
    R1_CPU_BASE_COMMIT_SHA,
    BackwardRecomputeSavedTensorReceipt,
    LaunchRuntimeBackwardValidationReceipt,
    build_backward_recompute_saved_tensor_receipt,
    build_launch_runtime_backward_validation_receipt,
    build_trainer_backward_wiring_proof_receipt,
)
from calm.llm_computer.tests.test_hrm_text_158_activation_relief import (
    _saved_tensor_proof,
)
from calm.llm_computer.tests.test_hrm_text_158_trainer_sub2_authority_live_checkpoint import (
    _mint_live_conversion_receipt,
)


SCRIPT = Path("scripts/hrm_text_158_full_sub2_runtime_readiness.py")
GATED_SUB2_CHECKPOINT_PATH_REASON = (
    "gated default-off sidecar checkpoint path only; default runtime not sub2"
)
GATED_LOSSLESS_RECOMPUTE_REASON = (
    "gated default-off lossless recompute path only; default runtime not sub2"
)


def _main_ready_receipt():
    return fixture_full_sub2_runtime_ready_for_science(FIXTURE_MAIN_READY)


def _replace_surface(
    surface_id: str,
    classification: str,
    *,
    reason: str = "test replacement reason",
    source_anchor: str = "test_hrm_text_158_full_sub2_runtime_readiness.py:fixture",
    proof_artifact_or_test: str = "test_hrm_text_158_full_sub2_runtime_readiness.py",
    sunset_condition: str = "",
    diagnostic_exception_reason: str = "",
    why_cheaper_than_full_stack_first: str = "",
    diagnostic_exclusion_reason: str = "",
):
    out = []
    for surface in main_ready_fixture_surfaces():
        if surface.surface_id == surface_id:
            out.append(
                FullSub2RuntimeSurfaceReceipt(
                    surface_id=surface_id,
                    classification=classification,
                    reason=reason,
                    source_anchor=source_anchor,
                    proof_artifact_or_test=proof_artifact_or_test,
                    sunset_condition=sunset_condition,
                    diagnostic_exception_reason=diagnostic_exception_reason,
                    why_cheaper_than_full_stack_first=why_cheaper_than_full_stack_first,
                    diagnostic_exclusion_reason=diagnostic_exclusion_reason,
                )
            )
        else:
            out.append(surface)
    return tuple(out)


def test_main_ready_fixture_allows_only_justified_explicit_exception():
    receipt = _main_ready_receipt()

    validate_full_sub2_runtime_ready_for_science_receipt(receipt)
    assert receipt.ready_for_main_science is True
    assert receipt.main_science_launch_blocked is False
    assert receipt.ready_for_pre_full_stack_diagnostic is False
    assert receipt.sub2_surface_count == len(FULL_SUB2_RUNTIME_REQUIRED_SURFACES) - 1
    assert receipt.counts_by_class[RUNTIME_CLASS_SUB2] == receipt.sub2_surface_count
    assert receipt.counts_by_class[RUNTIME_CLASS_EXPLICIT_EXCEPTION] == 1
    assert receipt.explicit_exception_surface_names == (SURFACE_FP_EXCEPTIONS_LEDGER,)
    assert SURFACE_FP_EXCEPTIONS_LEDGER not in receipt.surface_names_by_class[RUNTIME_CLASS_SUB2]


@pytest.mark.parametrize(
    ("fixture_name", "surface_id"),
    (
        (FIXTURE_MISSING_ACTIVATIONS, SURFACE_ACTIVATIONS_RESIDUALS),
        (FIXTURE_MISSING_ATTENTION, SURFACE_ATTENTION_KV_ATTENTION_BUFFERS),
        (FIXTURE_MISSING_BACKWARD, SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS),
    ),
)
def test_missing_required_surfaces_fail_closed(fixture_name, surface_id):
    receipt = fixture_full_sub2_runtime_ready_for_science(fixture_name)

    assert receipt.ready_for_main_science is False
    assert receipt.main_science_launch_blocked is True
    assert receipt.ready_for_pre_full_stack_diagnostic is False
    assert receipt.missing_surface_names == (surface_id,)
    assert receipt.counts_by_class[RUNTIME_CLASS_MISSING] == 1
    assert surface_id in receipt.blocker_surface_names


def test_pre_full_stack_diagnostic_allows_only_diagnostic_readiness():
    receipt = fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_PRE_FULL_STACK_DIAGNOSTIC
    )

    assert receipt.ready_for_main_science is False
    assert receipt.main_science_launch_blocked is True
    assert receipt.ready_for_pre_full_stack_diagnostic is True
    assert receipt.pre_full_stack_diagnostic_surface_names == (
        SURFACE_ACTIVATIONS_RESIDUALS,
    )
    assert receipt.counts_by_class[RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC] == 1


def test_transient_fp_debt_blocks_main_science_without_becoming_exception():
    receipt = fixture_full_sub2_runtime_ready_for_science(FIXTURE_TRANSIENT_FP_DEBT)

    assert receipt.ready_for_main_science is False
    assert receipt.main_science_launch_blocked is True
    assert receipt.ready_for_pre_full_stack_diagnostic is True
    assert receipt.counts_by_class[RUNTIME_CLASS_TRANSIENT_FP_DEBT] == 1
    assert receipt.counts_by_class[RUNTIME_CLASS_EXPLICIT_EXCEPTION] == 1
    assert receipt.transient_fp_debt_surface_names
    assert receipt.transient_fp_debt_surface_names != receipt.explicit_exception_surface_names


def test_step2a_candidate_fixture_keeps_live_rows_non_sub2():
    receipt = fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_STEP2A_CANDIDATE_PERSISTENT_CORE_ABSENCE
    )
    classes = {surface.surface_id: surface.classification for surface in receipt.surfaces}

    assert receipt.ready_for_main_science is False
    assert receipt.main_science_launch_blocked is True
    assert classes[SURFACE_PERSISTENT_QACC_AUTHORITY] == RUNTIME_CLASS_TRANSIENT_FP_DEBT
    assert (
        classes[SURFACE_DENSE_INT16_PERSISTENT_ACCUMULATOR_ABSENCE]
        == RUNTIME_CLASS_MISSING
    )
    assert classes[SURFACE_Q_SIDECAR_VOTE_CARRIER] == RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC
    assert SURFACE_PERSISTENT_QACC_AUTHORITY in receipt.blocker_surface_names
    assert SURFACE_DENSE_INT16_PERSISTENT_ACCUMULATOR_ABSENCE in receipt.blocker_surface_names
    assert SURFACE_Q_SIDECAR_VOTE_CARRIER in receipt.blocker_surface_names


def test_gated_sub2_checkpoint_path_fixture_flips_only_persistent_core_rows():
    current = fixture_full_sub2_runtime_ready_for_science(FIXTURE_CURRENT_REPO)
    gated = fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_GATED_SUB2_CHECKPOINT_PATH
    )
    validate_full_sub2_runtime_ready_for_science_receipt(gated)
    current_classes = {
        surface.surface_id: surface.classification for surface in current.surfaces
    }
    gated_classes = {
        surface.surface_id: surface.classification for surface in gated.surfaces
    }
    gated_reasons = {surface.surface_id: surface.reason for surface in gated.surfaces}
    flipped_surface_ids = {
        SURFACE_PERSISTENT_QACC_AUTHORITY,
        SURFACE_DENSE_INT16_PERSISTENT_ACCUMULATOR_ABSENCE,
        SURFACE_Q_SIDECAR_VOTE_CARRIER,
    }
    changed_surface_ids = {
        surface_id
        for surface_id, current_class in current_classes.items()
        if gated_classes[surface_id] != current_class
    }

    assert current_classes[SURFACE_PERSISTENT_QACC_AUTHORITY] != RUNTIME_CLASS_SUB2
    assert (
        current_classes[SURFACE_DENSE_INT16_PERSISTENT_ACCUMULATOR_ABSENCE]
        != RUNTIME_CLASS_SUB2
    )
    assert current_classes[SURFACE_Q_SIDECAR_VOTE_CARRIER] != RUNTIME_CLASS_SUB2
    assert changed_surface_ids == flipped_surface_ids
    for surface_id in flipped_surface_ids:
        assert gated_classes[surface_id] == RUNTIME_CLASS_SUB2
        assert GATED_SUB2_CHECKPOINT_PATH_REASON in gated_reasons[surface_id]
        assert "9600c36" in gated_reasons[surface_id]
    assert gated.ready_for_main_science is False
    assert gated.main_science_launch_blocked is True
    assert {
        SURFACE_OPTIMIZER_CREDIT_STATE,
        SURFACE_ACTIVATIONS_RESIDUALS,
        SURFACE_ATTENTION_KV_ATTENTION_BUFFERS,
        SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS,
        SURFACE_NATIVE_KERNELIZED_HOT_PATH,
    }.issubset(set(gated.blocker_surface_names))


def test_gated_backward_recompute_fixture_flips_only_backward_saved_tensors():
    current = fixture_full_sub2_runtime_ready_for_science(FIXTURE_CURRENT_REPO)
    gated = fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_GATED_SUB2_CHECKPOINT_PATH
    )
    recompute = fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_GATED_SUB2_CHECKPOINT_PATH_BACKWARD_RECOMPUTE
    )
    current_classes = {
        surface.surface_id: surface.classification for surface in current.surfaces
    }
    gated_classes = {
        surface.surface_id: surface.classification for surface in gated.surfaces
    }
    recompute_classes = {
        surface.surface_id: surface.classification for surface in recompute.surfaces
    }
    recompute_reasons = {
        surface.surface_id: surface.reason for surface in recompute.surfaces
    }
    changed_surface_ids = {
        surface_id
        for surface_id, gated_class in gated_classes.items()
        if recompute_classes[surface_id] != gated_class
    }

    assert current_classes[SURFACE_ACTIVATIONS_RESIDUALS] != RUNTIME_CLASS_SUB2
    assert current_classes[SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS] == RUNTIME_CLASS_MISSING
    assert gated_classes[SURFACE_ACTIVATIONS_RESIDUALS] != RUNTIME_CLASS_SUB2
    assert gated_classes[SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS] == RUNTIME_CLASS_MISSING
    assert changed_surface_ids == {SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS}
    assert recompute_classes[SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS] == RUNTIME_CLASS_SUB2
    assert recompute_classes[SURFACE_ACTIVATIONS_RESIDUALS] == RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC
    assert (
        GATED_LOSSLESS_RECOMPUTE_REASON
        in recompute_reasons[SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS]
    )
    assert "boundary z_H/z_L inputs remain accounted" in recompute_reasons[
        SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS
    ]
    assert recompute.ready_for_main_science is False
    assert recompute.main_science_launch_blocked is True
    assert recompute.ready_for_pre_full_stack_diagnostic is True
    assert SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS not in recompute.blocker_surface_names
    assert {
        SURFACE_ACTIVATIONS_RESIDUALS,
        SURFACE_ATTENTION_KV_ATTENTION_BUFFERS,
        SURFACE_OPTIMIZER_CREDIT_STATE,
        SURFACE_NATIVE_KERNELIZED_HOT_PATH,
    }.issubset(set(recompute.blocker_surface_names))


def test_gated_activation_residuals_blocked_fixture_updates_only_activation_reason_and_proof():
    current = fixture_full_sub2_runtime_ready_for_science(FIXTURE_CURRENT_REPO)
    gated = fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_GATED_SUB2_CHECKPOINT_PATH
    )
    recompute = fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_GATED_SUB2_CHECKPOINT_PATH_BACKWARD_RECOMPUTE
    )
    blocked = fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ACTIVATION_RESIDUALS_BLOCKED
    )
    validate_full_sub2_runtime_ready_for_science_receipt(blocked)

    current_classes = {
        surface.surface_id: surface.classification for surface in current.surfaces
    }
    gated_classes = {
        surface.surface_id: surface.classification for surface in gated.surfaces
    }
    recompute_classes = {
        surface.surface_id: surface.classification for surface in recompute.surfaces
    }
    blocked_classes = {
        surface.surface_id: surface.classification for surface in blocked.surfaces
    }
    recompute_surfaces = {
        surface.surface_id: surface.to_dict() for surface in recompute.surfaces
    }
    blocked_surfaces = {
        surface.surface_id: surface.to_dict() for surface in blocked.surfaces
    }

    assert current_classes[SURFACE_ACTIVATIONS_RESIDUALS] != RUNTIME_CLASS_SUB2
    assert gated_classes[SURFACE_ACTIVATIONS_RESIDUALS] != RUNTIME_CLASS_SUB2
    assert recompute_classes[SURFACE_ACTIVATIONS_RESIDUALS] != RUNTIME_CLASS_SUB2
    assert blocked_classes == recompute_classes
    assert blocked_classes[SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS] == RUNTIME_CLASS_SUB2
    assert (
        blocked_classes[SURFACE_ACTIVATIONS_RESIDUALS]
        == RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC
    )

    changed_surface_ids = {
        surface_id
        for surface_id, recompute_surface in recompute_surfaces.items()
        if blocked_surfaces[surface_id] != recompute_surface
    }
    assert changed_surface_ids == {SURFACE_ACTIVATIONS_RESIDUALS}
    activation_recompute = recompute_surfaces[SURFACE_ACTIVATIONS_RESIDUALS]
    activation_blocked = blocked_surfaces[SURFACE_ACTIVATIONS_RESIDUALS]
    changed_activation_fields = {
        field
        for field, recompute_value in activation_recompute.items()
        if activation_blocked[field] != recompute_value
    }
    assert changed_activation_fields == {"reason", "proof_artifact_or_test"}
    assert "fail-closed activation/residual live-tensor harness" in activation_blocked["reason"]
    assert "zL_init" in activation_blocked["reason"]
    assert "non_eligible_hrm_tensors" in activation_blocked["reason"]
    assert (
        "test_activation_residuals_fail_closed_receipt_enumerates_live_tensor_families_without_flip"
        in activation_blocked["proof_artifact_or_test"]
    )
    assert blocked.ready_for_main_science is False
    assert blocked.main_science_launch_blocked is True
    assert blocked.ready_for_pre_full_stack_diagnostic is True
    assert {
        SURFACE_ACTIVATIONS_RESIDUALS,
        SURFACE_ATTENTION_KV_ATTENTION_BUFFERS,
        SURFACE_OPTIMIZER_CREDIT_STATE,
        SURFACE_NATIVE_KERNELIZED_HOT_PATH,
    }.issubset(set(blocked.blocker_surface_names))
    assert SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS not in blocked.blocker_surface_names


def test_gated_attention_kv_blocked_fixture_updates_only_attention_reason_and_proof():
    current = fixture_full_sub2_runtime_ready_for_science(FIXTURE_CURRENT_REPO)
    gated = fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_GATED_SUB2_CHECKPOINT_PATH
    )
    recompute = fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_GATED_SUB2_CHECKPOINT_PATH_BACKWARD_RECOMPUTE
    )
    activation_blocked = fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ACTIVATION_RESIDUALS_BLOCKED
    )
    attention_blocked = fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ATTENTION_KV_BLOCKED
    )
    validate_full_sub2_runtime_ready_for_science_receipt(attention_blocked)

    current_classes = {
        surface.surface_id: surface.classification for surface in current.surfaces
    }
    gated_classes = {
        surface.surface_id: surface.classification for surface in gated.surfaces
    }
    recompute_classes = {
        surface.surface_id: surface.classification for surface in recompute.surfaces
    }
    activation_blocked_classes = {
        surface.surface_id: surface.classification
        for surface in activation_blocked.surfaces
    }
    attention_blocked_classes = {
        surface.surface_id: surface.classification
        for surface in attention_blocked.surfaces
    }
    activation_blocked_surfaces = {
        surface.surface_id: surface.to_dict()
        for surface in activation_blocked.surfaces
    }
    attention_blocked_surfaces = {
        surface.surface_id: surface.to_dict()
        for surface in attention_blocked.surfaces
    }

    assert current_classes[SURFACE_ATTENTION_KV_ATTENTION_BUFFERS] != RUNTIME_CLASS_SUB2
    assert gated_classes[SURFACE_ATTENTION_KV_ATTENTION_BUFFERS] != RUNTIME_CLASS_SUB2
    assert recompute_classes[SURFACE_ATTENTION_KV_ATTENTION_BUFFERS] != RUNTIME_CLASS_SUB2
    assert attention_blocked_classes == activation_blocked_classes
    assert (
        attention_blocked_classes[SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS]
        == RUNTIME_CLASS_SUB2
    )
    assert (
        attention_blocked_classes[SURFACE_ACTIVATIONS_RESIDUALS]
        == RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC
    )
    assert (
        attention_blocked_classes[SURFACE_ATTENTION_KV_ATTENTION_BUFFERS]
        == RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC
    )

    changed_surface_ids = {
        surface_id
        for surface_id, activation_surface in activation_blocked_surfaces.items()
        if attention_blocked_surfaces[surface_id] != activation_surface
    }
    assert changed_surface_ids == {SURFACE_ATTENTION_KV_ATTENTION_BUFFERS}
    attention_before = activation_blocked_surfaces[SURFACE_ATTENTION_KV_ATTENTION_BUFFERS]
    attention_after = attention_blocked_surfaces[SURFACE_ATTENTION_KV_ATTENTION_BUFFERS]
    changed_attention_fields = {
        field
        for field, before_value in attention_before.items()
        if attention_after[field] != before_value
    }
    assert changed_attention_fields == {"reason", "proof_artifact_or_test"}
    assert "fail-closed attention/KV live-tensor harness" in attention_after["reason"]
    assert "q/k/v seam observations" in attention_after["reason"]
    assert "PrefixLM mask" in attention_after["reason"]
    assert "GQA repeat" in attention_after["reason"]
    assert "SDPA workspace" in attention_after["reason"]
    assert "runtime KVCache" in attention_after["reason"]
    assert (
        "test_attention_kv_fail_closed_receipt_enumerates_qkv_allowlist_without_flip"
        in attention_after["proof_artifact_or_test"]
    )
    assert attention_blocked.ready_for_main_science is False
    assert attention_blocked.main_science_launch_blocked is True
    assert attention_blocked.ready_for_pre_full_stack_diagnostic is True
    assert set(attention_blocked.blocker_surface_names) == {
        SURFACE_ACTIVATIONS_RESIDUALS,
        SURFACE_ATTENTION_KV_ATTENTION_BUFFERS,
        SURFACE_OPTIMIZER_CREDIT_STATE,
        SURFACE_NATIVE_KERNELIZED_HOT_PATH,
    }
    assert SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS not in (
        attention_blocked.blocker_surface_names
    )


def test_explicit_exception_requires_fail_closed_fields():
    surfaces = _replace_surface(
        SURFACE_FP_EXCEPTIONS_LEDGER,
        RUNTIME_CLASS_EXPLICIT_EXCEPTION,
        sunset_condition="",
    )

    with pytest.raises(ValueError, match="sunset_condition"):
        build_full_sub2_runtime_ready_for_science(surfaces)


def test_unknown_surface_id_fails_validation():
    surfaces = list(main_ready_fixture_surfaces())
    surfaces[0] = FullSub2RuntimeSurfaceReceipt(
        surface_id="activations_residual_typo",
        classification=RUNTIME_CLASS_SUB2,
        reason="typo should not be accepted as a required surface",
        source_anchor="test",
        proof_artifact_or_test="test",
    )

    with pytest.raises(ValueError):
        build_full_sub2_runtime_ready_for_science(tuple(surfaces))


def test_duplicate_surface_id_fails_validation():
    surfaces = list(main_ready_fixture_surfaces())
    surfaces[1] = surfaces[0]

    with pytest.raises(ValueError, match="duplicate"):
        build_full_sub2_runtime_ready_for_science(tuple(surfaces))


def test_export_api_smoke():
    assert native_full_stack.FULL_SUB2_RUNTIME_CLASSIFICATIONS == FULL_SUB2_RUNTIME_CLASSIFICATIONS
    assert (
        native_full_stack.FIXTURE_GATED_SUB2_CHECKPOINT_PATH
        == FIXTURE_GATED_SUB2_CHECKPOINT_PATH
    )
    assert (
        native_full_stack.FIXTURE_GATED_SUB2_CHECKPOINT_PATH_BACKWARD_RECOMPUTE
        == FIXTURE_GATED_SUB2_CHECKPOINT_PATH_BACKWARD_RECOMPUTE
    )
    assert (
        native_full_stack.FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ACTIVATION_RESIDUALS_BLOCKED
        == FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ACTIVATION_RESIDUALS_BLOCKED
    )
    assert (
        native_full_stack.FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ATTENTION_KV_BLOCKED
        == FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ATTENTION_KV_BLOCKED
    )
    assert (
        native_full_stack.FIXTURE_GATED_SUB2_CHECKPOINT_PATH_OPTIMIZER_CREDIT_STATE_BLOCKED
        == FIXTURE_GATED_SUB2_CHECKPOINT_PATH_OPTIMIZER_CREDIT_STATE_BLOCKED
    )
    assert (
        native_full_stack.FIXTURE_GATED_SUB2_CHECKPOINT_PATH_NATIVE_KERNELIZED_HOT_PATH_BLOCKED
        == FIXTURE_GATED_SUB2_CHECKPOINT_PATH_NATIVE_KERNELIZED_HOT_PATH_BLOCKED
    )
    assert native_full_stack.RUNTIME_CLASS_TRANSIENT_FP_DEBT == RUNTIME_CLASS_TRANSIENT_FP_DEBT
    assert "FIXTURE_GATED_SUB2_CHECKPOINT_PATH" in native_full_stack.__all__
    assert "FIXTURE_GATED_SUB2_CHECKPOINT_PATH_BACKWARD_RECOMPUTE" in native_full_stack.__all__
    assert (
        "FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ACTIVATION_RESIDUALS_BLOCKED"
        in native_full_stack.__all__
    )
    assert (
        "FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ATTENTION_KV_BLOCKED"
        in native_full_stack.__all__
    )
    assert (
        "FIXTURE_GATED_SUB2_CHECKPOINT_PATH_OPTIMIZER_CREDIT_STATE_BLOCKED"
        in native_full_stack.__all__
    )
    assert (
        "FIXTURE_GATED_SUB2_CHECKPOINT_PATH_NATIVE_KERNELIZED_HOT_PATH_BLOCKED"
        in native_full_stack.__all__
    )
    assert "fixture_full_sub2_runtime_ready_for_science" in native_full_stack.__all__
    assert "gated_sub2_checkpoint_path_backward_recompute_surfaces" in native_full_stack.__all__
    assert (
        "gated_sub2_checkpoint_path_activation_residuals_blocked_surfaces"
        in native_full_stack.__all__
    )
    assert (
        "gated_sub2_checkpoint_path_attention_kv_blocked_surfaces"
        in native_full_stack.__all__
    )
    assert (
        "gated_sub2_checkpoint_path_optimizer_credit_state_blocked_surfaces"
        in native_full_stack.__all__
    )
    assert "gated_sub2_checkpoint_path_surfaces" in native_full_stack.__all__
    receipt = native_full_stack.fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_PRE_FULL_STACK_DIAGNOSTIC
    )
    assert receipt.ready_for_pre_full_stack_diagnostic is True
    gated_surfaces = native_full_stack.gated_sub2_checkpoint_path_surfaces()
    assert gated_surfaces == gated_sub2_checkpoint_path_surfaces()
    recompute_surfaces = native_full_stack.gated_sub2_checkpoint_path_backward_recompute_surfaces()
    assert recompute_surfaces == gated_sub2_checkpoint_path_backward_recompute_surfaces()
    blocked_surfaces = (
        native_full_stack.gated_sub2_checkpoint_path_activation_residuals_blocked_surfaces()
    )
    assert blocked_surfaces == gated_sub2_checkpoint_path_activation_residuals_blocked_surfaces()
    attention_blocked_surfaces = (
        native_full_stack.gated_sub2_checkpoint_path_attention_kv_blocked_surfaces()
    )
    assert attention_blocked_surfaces == gated_sub2_checkpoint_path_attention_kv_blocked_surfaces()
    optimizer_blocked_surfaces = (
        native_full_stack.gated_sub2_checkpoint_path_optimizer_credit_state_blocked_surfaces()
    )
    assert (
        optimizer_blocked_surfaces
        == gated_sub2_checkpoint_path_optimizer_credit_state_blocked_surfaces()
    )
    native_blocked_surfaces = (
        native_full_stack.gated_sub2_checkpoint_path_native_kernelized_hot_path_blocked_surfaces()
    )
    assert (
        native_blocked_surfaces
        == gated_sub2_checkpoint_path_native_kernelized_hot_path_blocked_surfaces()
    )


@pytest.mark.parametrize(
    ("fixture_name", "surface_id"),
    (
        (FIXTURE_MISSING_ACTIVATIONS, SURFACE_ACTIVATIONS_RESIDUALS),
        (FIXTURE_MISSING_ATTENTION, SURFACE_ATTENTION_KV_ATTENTION_BUFFERS),
        (FIXTURE_MISSING_BACKWARD, SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS),
    ),
)
def test_cli_expect_ready_negative_writes_json(tmp_path, fixture_name, surface_id):
    json_out = tmp_path / f"{fixture_name}.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fixture",
            fixture_name,
            "--json-out",
            str(json_out),
            "--expect-ready",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode != 0
    assert json_out.exists()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["ready_for_main_science"] is False
    assert payload["main_science_launch_blocked"] is True
    assert surface_id in payload["missing_surface_names"]
    assert surface_id in payload["blocker_surface_names"]


def test_cli_gated_sub2_checkpoint_path_expect_ready_still_blocks(tmp_path):
    json_out = tmp_path / "gated_sub2_checkpoint_path.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fixture",
            FIXTURE_GATED_SUB2_CHECKPOINT_PATH,
            "--json-out",
            str(json_out),
            "--expect-ready",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 2
    assert json_out.exists()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    surfaces = {surface["surface_id"]: surface for surface in payload["surfaces"]}
    assert payload["ready_for_main_science"] is False
    assert payload["main_science_launch_blocked"] is True
    for surface_id in (
        SURFACE_PERSISTENT_QACC_AUTHORITY,
        SURFACE_DENSE_INT16_PERSISTENT_ACCUMULATOR_ABSENCE,
        SURFACE_Q_SIDECAR_VOTE_CARRIER,
    ):
        assert surfaces[surface_id]["classification"] == RUNTIME_CLASS_SUB2
        assert GATED_SUB2_CHECKPOINT_PATH_REASON in surfaces[surface_id]["reason"]
    assert {
        SURFACE_OPTIMIZER_CREDIT_STATE,
        SURFACE_ACTIVATIONS_RESIDUALS,
        SURFACE_ATTENTION_KV_ATTENTION_BUFFERS,
        SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS,
        SURFACE_NATIVE_KERNELIZED_HOT_PATH,
    }.issubset(set(payload["blocker_surface_names"]))


def test_cli_gated_backward_recompute_expect_ready_still_blocks_with_diagnostic_ready(tmp_path):
    json_out = tmp_path / "gated_sub2_checkpoint_path_backward_recompute.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fixture",
            FIXTURE_GATED_SUB2_CHECKPOINT_PATH_BACKWARD_RECOMPUTE,
            "--json-out",
            str(json_out),
            "--expect-ready",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 2
    assert json_out.exists()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    surfaces = {surface["surface_id"]: surface for surface in payload["surfaces"]}
    assert payload["ready_for_main_science"] is False
    assert payload["main_science_launch_blocked"] is True
    assert payload["ready_for_pre_full_stack_diagnostic"] is True
    assert (
        surfaces[SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS]["classification"]
        == RUNTIME_CLASS_SUB2
    )
    assert (
        surfaces[SURFACE_ACTIVATIONS_RESIDUALS]["classification"]
        == RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC
    )
    assert (
        GATED_LOSSLESS_RECOMPUTE_REASON
        in surfaces[SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS]["reason"]
    )
    assert {
        SURFACE_ACTIVATIONS_RESIDUALS,
        SURFACE_ATTENTION_KV_ATTENTION_BUFFERS,
        SURFACE_OPTIMIZER_CREDIT_STATE,
        SURFACE_NATIVE_KERNELIZED_HOT_PATH,
    }.issubset(set(payload["blocker_surface_names"]))
    assert SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS not in payload["blocker_surface_names"]


def test_cli_gated_activation_residuals_blocked_expect_ready_emits_blocker_reason(tmp_path):
    json_out = tmp_path / "gated_sub2_checkpoint_path_activation_residuals_blocked.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fixture",
            FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ACTIVATION_RESIDUALS_BLOCKED,
            "--json-out",
            str(json_out),
            "--expect-ready",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 2
    assert json_out.exists()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    surfaces = {surface["surface_id"]: surface for surface in payload["surfaces"]}
    activation = surfaces[SURFACE_ACTIVATIONS_RESIDUALS]
    assert payload["ready_for_main_science"] is False
    assert payload["main_science_launch_blocked"] is True
    assert payload["ready_for_pre_full_stack_diagnostic"] is True
    assert activation["classification"] == RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC
    assert "fail-closed activation/residual live-tensor harness" in activation["reason"]
    assert "zL_init" in activation["reason"]
    assert SURFACE_ACTIVATIONS_RESIDUALS in payload["blocker_surface_names"]
    assert SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS not in payload["blocker_surface_names"]
    assert "activations_residuals" in result.stdout


def test_cli_gated_attention_kv_blocked_expect_ready_emits_blocker_reason(tmp_path):
    json_out = tmp_path / "gated_sub2_checkpoint_path_attention_kv_blocked.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fixture",
            FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ATTENTION_KV_BLOCKED,
            "--json-out",
            str(json_out),
            "--expect-ready",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 2
    assert json_out.exists()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    surfaces = {surface["surface_id"]: surface for surface in payload["surfaces"]}
    attention = surfaces[SURFACE_ATTENTION_KV_ATTENTION_BUFFERS]
    assert payload["ready_for_main_science"] is False
    assert payload["main_science_launch_blocked"] is True
    assert payload["ready_for_pre_full_stack_diagnostic"] is True
    assert attention["classification"] == RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC
    assert "fail-closed attention/KV live-tensor harness" in attention["reason"]
    assert "q/k/v seam observations" in attention["reason"]
    assert "runtime KVCache" in attention["reason"]
    assert set(payload["blocker_surface_names"]) == {
        SURFACE_ACTIVATIONS_RESIDUALS,
        SURFACE_ATTENTION_KV_ATTENTION_BUFFERS,
        SURFACE_OPTIMIZER_CREDIT_STATE,
        SURFACE_NATIVE_KERNELIZED_HOT_PATH,
    }
    assert SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS not in payload["blocker_surface_names"]
    assert "attention_kv_attention_buffers" in result.stdout


def test_cli_gated_optimizer_credit_state_blocked_expect_ready_emits_blocker_reason(tmp_path):
    json_out = tmp_path / "gated_sub2_checkpoint_path_optimizer_credit_state_blocked.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fixture",
            FIXTURE_GATED_SUB2_CHECKPOINT_PATH_OPTIMIZER_CREDIT_STATE_BLOCKED,
            "--json-out",
            str(json_out),
            "--expect-ready",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 2
    assert json_out.exists()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    surfaces = {surface["surface_id"]: surface for surface in payload["surfaces"]}
    optimizer = surfaces[SURFACE_OPTIMIZER_CREDIT_STATE]
    assert payload["ready_for_main_science"] is False
    assert payload["main_science_launch_blocked"] is True
    assert payload["ready_for_pre_full_stack_diagnostic"] is True
    assert optimizer["classification"] == RUNTIME_CLASS_TRANSIENT_FP_DEBT
    assert "fail-closed optimizer/credit-state harness" in optimizer["reason"]
    assert "weighted_grad" in optimizer["reason"]
    assert "credit_capture_tensors" in optimizer["reason"]
    assert set(payload["blocker_surface_names"]) == {
        SURFACE_ACTIVATIONS_RESIDUALS,
        SURFACE_ATTENTION_KV_ATTENTION_BUFFERS,
        SURFACE_OPTIMIZER_CREDIT_STATE,
        SURFACE_NATIVE_KERNELIZED_HOT_PATH,
    }
    assert SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS not in payload["blocker_surface_names"]
    assert "optimizer_credit_state" in result.stdout


def test_cli_gated_native_kernelized_hot_path_blocked_expect_ready_emits_blocker_reason(tmp_path):
    json_out = tmp_path / "gated_sub2_checkpoint_path_native_kernelized_hot_path_blocked.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fixture",
            FIXTURE_GATED_SUB2_CHECKPOINT_PATH_NATIVE_KERNELIZED_HOT_PATH_BLOCKED,
            "--json-out",
            str(json_out),
            "--expect-ready",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 2
    assert json_out.exists()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    surfaces = {surface["surface_id"]: surface for surface in payload["surfaces"]}
    native = surfaces[SURFACE_NATIVE_KERNELIZED_HOT_PATH]
    assert payload["ready_for_main_science"] is False
    assert payload["main_science_launch_blocked"] is True
    assert payload["ready_for_pre_full_stack_diagnostic"] is True
    assert native["classification"] == RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC
    assert "fail-closed native kernelized hot-path harness" in native["reason"]
    assert "qacc_kernelized=false" in native["reason"]
    assert "device=cuda" in native["reason"]
    assert "CPU row materialization" in native["reason"]
    assert set(payload["blocker_surface_names"]) == {
        SURFACE_ACTIVATIONS_RESIDUALS,
        SURFACE_ATTENTION_KV_ATTENTION_BUFFERS,
        SURFACE_OPTIMIZER_CREDIT_STATE,
        SURFACE_NATIVE_KERNELIZED_HOT_PATH,
    }
    assert SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS not in payload["blocker_surface_names"]
    assert "native_kernelized_hot_path" in result.stdout


def test_gated_optimizer_credit_state_blocked_fixture_updates_only_optimizer_debt_fields():
    attention_blocked = fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_GATED_SUB2_CHECKPOINT_PATH_ATTENTION_KV_BLOCKED
    )
    optimizer_blocked = fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_GATED_SUB2_CHECKPOINT_PATH_OPTIMIZER_CREDIT_STATE_BLOCKED
    )
    validate_full_sub2_runtime_ready_for_science_receipt(optimizer_blocked)

    attention_classes = {
        surface.surface_id: surface.classification
        for surface in attention_blocked.surfaces
    }
    optimizer_classes = {
        surface.surface_id: surface.classification
        for surface in optimizer_blocked.surfaces
    }
    attention_surfaces = {
        surface.surface_id: surface.to_dict()
        for surface in attention_blocked.surfaces
    }
    optimizer_surfaces = {
        surface.surface_id: surface.to_dict()
        for surface in optimizer_blocked.surfaces
    }

    assert optimizer_classes == attention_classes
    assert (
        optimizer_classes[SURFACE_OPTIMIZER_CREDIT_STATE]
        == RUNTIME_CLASS_TRANSIENT_FP_DEBT
    )
    assert (
        optimizer_classes[SURFACE_ATTENTION_KV_ATTENTION_BUFFERS]
        == RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC
    )
    assert (
        optimizer_classes[SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS]
        == RUNTIME_CLASS_SUB2
    )

    changed_surface_ids = {
        surface_id
        for surface_id, attention_surface in attention_surfaces.items()
        if optimizer_surfaces[surface_id] != attention_surface
    }
    assert changed_surface_ids == {SURFACE_OPTIMIZER_CREDIT_STATE}
    optimizer_before = attention_surfaces[SURFACE_OPTIMIZER_CREDIT_STATE]
    optimizer_after = optimizer_surfaces[SURFACE_OPTIMIZER_CREDIT_STATE]
    changed_optimizer_fields = {
        field
        for field, before_value in optimizer_before.items()
        if optimizer_after[field] != before_value
    }
    assert changed_optimizer_fields == {
        "reason",
        "proof_artifact_or_test",
        "source_anchor",
    }
    assert "fail-closed optimizer/credit-state harness" in optimizer_after["reason"]
    assert "weighted_grad" in optimizer_after["reason"]
    assert "credit" in optimizer_after["reason"]
    assert "projected_moves" in optimizer_after["reason"]
    assert "dense_rank_votes" in optimizer_after["reason"]
    assert "credit_capture_tensors" in optimizer_after["reason"]
    assert "optimizer_credit_state.py" in optimizer_after["source_anchor"]
    assert (
        "test_optimizer_credit_state_fail_closed_receipt_enumerates_dense_debt_without_flip"
        in optimizer_after["proof_artifact_or_test"]
    )
    assert optimizer_blocked.ready_for_main_science is False
    assert optimizer_blocked.main_science_launch_blocked is True
    assert optimizer_blocked.ready_for_pre_full_stack_diagnostic is True
    assert set(optimizer_blocked.blocker_surface_names) == {
        SURFACE_ACTIVATIONS_RESIDUALS,
        SURFACE_ATTENTION_KV_ATTENTION_BUFFERS,
        SURFACE_OPTIMIZER_CREDIT_STATE,
        SURFACE_NATIVE_KERNELIZED_HOT_PATH,
    }
    assert SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS not in (
        optimizer_blocked.blocker_surface_names
    )


def test_gated_native_kernelized_hot_path_blocked_fixture_updates_only_native_fields():
    optimizer_blocked = fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_GATED_SUB2_CHECKPOINT_PATH_OPTIMIZER_CREDIT_STATE_BLOCKED
    )
    native_blocked = fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_GATED_SUB2_CHECKPOINT_PATH_NATIVE_KERNELIZED_HOT_PATH_BLOCKED
    )
    validate_full_sub2_runtime_ready_for_science_receipt(native_blocked)

    optimizer_classes = {
        surface.surface_id: surface.classification
        for surface in optimizer_blocked.surfaces
    }
    native_classes = {
        surface.surface_id: surface.classification
        for surface in native_blocked.surfaces
    }
    optimizer_surfaces = {
        surface.surface_id: surface.to_dict()
        for surface in optimizer_blocked.surfaces
    }
    native_surfaces = {
        surface.surface_id: surface.to_dict()
        for surface in native_blocked.surfaces
    }

    assert native_classes == optimizer_classes
    assert (
        native_classes[SURFACE_NATIVE_KERNELIZED_HOT_PATH]
        == RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC
    )
    assert (
        native_classes[SURFACE_OPTIMIZER_CREDIT_STATE]
        == RUNTIME_CLASS_TRANSIENT_FP_DEBT
    )
    assert (
        native_classes[SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS]
        == RUNTIME_CLASS_SUB2
    )

    changed_surface_ids = {
        surface_id
        for surface_id, optimizer_surface in optimizer_surfaces.items()
        if native_surfaces[surface_id] != optimizer_surface
    }
    assert changed_surface_ids == {SURFACE_NATIVE_KERNELIZED_HOT_PATH}
    native_before = optimizer_surfaces[SURFACE_NATIVE_KERNELIZED_HOT_PATH]
    native_after = native_surfaces[SURFACE_NATIVE_KERNELIZED_HOT_PATH]
    changed_native_fields = {
        field
        for field, before_value in native_before.items()
        if native_after[field] != before_value
    }
    assert changed_native_fields == {
        "reason",
        "proof_artifact_or_test",
        "source_anchor",
    }
    assert "fail-closed native kernelized hot-path harness" in native_after["reason"]
    assert "qacc_kernelized=false" in native_after["reason"]
    assert "Triton preplan" in native_after["reason"]
    assert "final-row torch-CUDA reference" in native_after["reason"]
    assert "MARGIN-only/default-off reference" in native_after["reason"]
    assert "native custom kernel speed claim" in native_after["reason"]
    assert "device=cuda" in native_after["reason"]
    assert "native_kernelized_hot_path.py" in native_after["source_anchor"]
    assert (
        "test_native_kernelized_hot_path_receipt_enumerates_current_blockers_without_flip"
        in native_after["proof_artifact_or_test"]
    )
    assert native_blocked.ready_for_main_science is False
    assert native_blocked.main_science_launch_blocked is True
    assert native_blocked.ready_for_pre_full_stack_diagnostic is True
    assert set(native_blocked.blocker_surface_names) == {
        SURFACE_ACTIVATIONS_RESIDUALS,
        SURFACE_ATTENTION_KV_ATTENTION_BUFFERS,
        SURFACE_OPTIMIZER_CREDIT_STATE,
        SURFACE_NATIVE_KERNELIZED_HOT_PATH,
    }
    assert SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS not in (
        native_blocked.blocker_surface_names
    )


def test_readiness_classes_are_exact_five_class_prereg():
    assert FULL_SUB2_RUNTIME_CLASSIFICATIONS == (
        RUNTIME_CLASS_SUB2,
        RUNTIME_CLASS_EXPLICIT_EXCEPTION,
        RUNTIME_CLASS_TRANSIENT_FP_DEBT,
        RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
        RUNTIME_CLASS_MISSING,
    )


def test_live_p1_authority_conversion_flips_exactly_authorized_rows():
    base = current_repo_scaffold_surfaces()
    base_by_id = {surface.surface_id: surface for surface in base}
    receipt = _mint_live_conversion_receipt()
    readiness = live_p1_authority_conversion_surfaces(receipt)
    authorized = set(receipt.readiness_row_flip_authorized_surface_names)
    assert readiness.sub2_surface_count == len(authorized)
    for surface in readiness.surfaces:
        if surface.surface_id in authorized:
            assert surface.classification == RUNTIME_CLASS_SUB2
        else:
            assert surface.classification == base_by_id[surface.surface_id].classification


def test_live_p1_authority_conversion_keeps_main_and_diag_false():
    receipt = _mint_live_conversion_receipt()
    readiness = live_p1_authority_conversion_surfaces(receipt)
    assert readiness.ready_for_main_science is False
    assert readiness.ready_for_pre_full_stack_diagnostic is False
    assert receipt.ready_for_main_science is False
    assert receipt.ready_for_pre_full_stack_diagnostic is False


def test_gated_fixture_does_not_satisfy_live_applier():
    receipt = _mint_live_conversion_receipt()
    gated_surfaces = gated_sub2_checkpoint_path_surfaces()
    live_surfaces = apply_live_p1_conversion_surface_overrides(receipt)
    q_sidecar_live = next(
        surface
        for surface in live_surfaces
        if surface.surface_id == SURFACE_Q_SIDECAR_VOTE_CARRIER
    )
    q_sidecar_gated = next(
        surface
        for surface in gated_surfaces
        if surface.surface_id == SURFACE_Q_SIDECAR_VOTE_CARRIER
    )
    assert receipt.p1_envelope_sha256 in q_sidecar_live.reason
    assert receipt.p1_envelope_sha256 not in q_sidecar_gated.reason
    assert GATED_SUB2_CHECKPOINT_PATH_REASON in q_sidecar_gated.reason


def test_live_p1_applier_without_validated_receipt_raises():
    with pytest.raises(ValueError):
        apply_live_p1_conversion_surface_overrides(
            replace(_mint_live_conversion_receipt(), pass_receipt=False)
        )


def test_live_p1_fixture_name_requires_explicit_receipt_json():
    with pytest.raises(ValueError, match="explicit validated receipt JSON"):
        fixture_full_sub2_runtime_ready_for_science(FIXTURE_LIVE_P1_AUTHORITY_CONVERSION)


def test_live_p1_two_row_fallback_keeps_q_sidecar_blocked():
    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
        AUTHORIZED_P1B_SURFACE_TUPLE_2ROW,
        validate_trainer_sub2_authority_live_conversion_receipt,
    )

    minted = _mint_live_conversion_receipt()
    two_row = replace(
        minted,
        readiness_row_flip_authorized_surface_names=AUTHORIZED_P1B_SURFACE_TUPLE_2ROW,
        q_sidecar_vote_carrier_deferred=True,
        q_sidecar_deferred_reason="test 2-row fallback: q_changed_count=0",
        q_changed_count=0,
        post_resume_update_mutated=False,
    )
    validate_trainer_sub2_authority_live_conversion_receipt(two_row)
    readiness = live_p1_authority_conversion_surfaces(two_row)
    assert readiness.sub2_surface_count == 2
    q_sidecar = next(
        surface
        for surface in readiness.surfaces
        if surface.surface_id == SURFACE_Q_SIDECAR_VOTE_CARRIER
    )
    assert q_sidecar.classification != RUNTIME_CLASS_SUB2
    assert two_row.q_sidecar_deferred_reason


def _post_p1_base_surfaces():
    return apply_live_p1_conversion_surface_overrides(_mint_live_conversion_receipt())


def _mint_cpu_wiring_receipt():
    main_proof = {
        "baseline_saved_tensor_count": 20,
        "recompute_saved_tensor_count": 15,
        "internal_payload_tensor_count": 0,
        "recompute_checkpoint_fired": True,
    }
    return build_trainer_backward_wiring_proof_receipt(
        source_commit_sha="cpu-wiring-test",
        proof_command_argv=("pytest",),
        H_cycles=2,
        L_cycles=3,
        bp_steps=5,
        main_path_proof=main_proof,
        retained_side_in_scope=False,
        retained_side_skip_reason="test-only main path",
    )


def _mint_valid_launch_receipt() -> LaunchRuntimeBackwardValidationReceipt:
    launch_source = R1_CPU_BASE_COMMIT_SHA
    manifest = {
        "r1_cpu_base_commit_sha": R1_CPU_BASE_COMMIT_SHA,
        "launch_source_commit_sha": launch_source,
        "archive_created_at_utc": "2026-06-15T00:00:00Z",
        "archive_method": "git_archive_HEAD",
    }
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": ".",
        "R1L_LAUNCH_RECEIPT_JSON": "/tmp/run/receipts/r1l_launch_runtime_receipt.json",
        "R1L_LAUNCH_LOG": "/tmp/run/logs/r1l_launch.log",
        "R1L_W6_PARENT_PATH": "/tmp/run/artifacts/w6_parent_readonly.pt",
    }
    return build_launch_runtime_backward_validation_receipt(
        launch_source_commit_sha=launch_source,
        launch_manifest_embedded=manifest,
        proof_env_embedded=env,
        proof_command_argv=("pytest", "launch"),
        clean_run_dir_sha256="a" * 64,
        w6_parent_path="/tmp/run/artifacts/w6_parent_readonly.pt",
        gpu_name="synthetic-gpu",
        gpu_uuid="gpu-uuid-test",
        driver_version="550.00",
        cuda_version="12.4",
        torch_version="2.5.0",
        model_config_digest_sha256="b" * 64,
        proof_batch_digest_sha256="c" * 64,
        retained_support_digest_sha256="d" * 64,
        main_baseline_saved_tensor_count=20,
        main_recompute_saved_tensor_count=15,
        main_saved_tensor_payload_bytes_baseline=1000,
        main_saved_tensor_payload_bytes_recompute=800,
        retained_side_in_scope=True,
        retained_side_baseline_saved_tensor_count=18,
        retained_side_recompute_saved_tensor_count=14,
        retained_saved_tensor_payload_bytes_delta=400,
        paired_run_count=3,
        cuda_peak_allocated_bytes_baseline_median=64 * 1024 * 1024,
        cuda_peak_allocated_bytes_recompute_median=56 * 1024 * 1024,
        cuda_peak_reserved_bytes_delta_median=0,
        log_artifact_sha256=hashlib.sha256(b"r1l launch log bytes").hexdigest(),
    )


def test_live_r1_applier_rejects_fixture_backward_recompute_receipt():
    fixture_receipt = build_backward_recompute_saved_tensor_receipt(
        H_cycles=2,
        L_cycles=3,
        bp_steps=5,
        saved_tensor_proof=_saved_tensor_proof(bp_steps=5),
    )
    assert isinstance(fixture_receipt, BackwardRecomputeSavedTensorReceipt)
    with pytest.raises(ValueError, match="fixture backward recompute receipt"):
        apply_live_r1_backward_wiring_surface_overrides(
            fixture_receipt,
            base_surfaces=_post_p1_base_surfaces(),
        )


def test_live_r1_applier_rejects_cpu_wiring_receipt():
    with pytest.raises(ValueError, match="CPU production autograd wiring"):
        apply_live_r1_backward_wiring_surface_overrides(
            _mint_cpu_wiring_receipt(),
            base_surfaces=_post_p1_base_surfaces(),
        )


def test_live_r1_applier_rejects_forged_launch_runtime_receipt():
    forged = replace(
        _mint_valid_launch_receipt(),
        launch_runtime_validation_pass=False,
    )
    with pytest.raises(ValueError, match="launch_runtime_validation_pass"):
        apply_live_r1_backward_wiring_surface_overrides(
            forged,
            base_surfaces=_post_p1_base_surfaces(),
        )


def test_live_r1_launch_runtime_validation_flips_exactly_backward_row():
    base = _post_p1_base_surfaces()
    base_by_id = {surface.surface_id: surface for surface in base}
    receipt = _mint_valid_launch_receipt()
    readiness = live_r1_backward_wiring_surfaces(
        receipt,
        base_surfaces=base,
    )
    assert readiness.ready_for_main_science is False
    assert readiness.ready_for_pre_full_stack_diagnostic is True
    assert readiness.sub2_surface_count == 4
    changed = {
        surface.surface_id
        for surface in readiness.surfaces
        if surface.classification != base_by_id[surface.surface_id].classification
    }
    assert changed == {SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS}
    backward = next(
        surface
        for surface in readiness.surfaces
        if surface.surface_id == SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS
    )
    assert backward.classification == RUNTIME_CLASS_SUB2


def test_current_repo_scaffold_unchanged_by_cpu_wiring_receipt():
    current = fixture_full_sub2_runtime_ready_for_science(FIXTURE_CURRENT_REPO)
    assert current.ready_for_pre_full_stack_diagnostic is False
    backward = next(
        surface
        for surface in current.surfaces
        if surface.surface_id == SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS
    )
    assert backward.classification == RUNTIME_CLASS_MISSING
    with pytest.raises(ValueError, match="CPU production autograd wiring"):
        apply_live_r1_backward_wiring_surface_overrides(_mint_cpu_wiring_receipt())


def _mint_r2a_cpu_seam_receipt():
    from calm.hrm_text_158.native_full_stack.activation_relief import (
        build_trainer_activation_residuals_seam_proof_receipt,
    )
    from calm.llm_computer.tests.test_hrm_text_158_activation_relief import (
        _activation_residual_live_tensor_proof,
    )

    events, zL_init_observation = _activation_residual_live_tensor_proof()
    return build_trainer_activation_residuals_seam_proof_receipt(
        source_commit_sha="abc123",
        proof_command_argv=("test",),
        seam_events=events,
        zL_init_observation=zL_init_observation,
    )


def test_live_r2a_applier_rejects_fixture_fail_closed_receipt():
    from calm.hrm_text_158.native_full_stack.activation_relief import (
        build_activation_residuals_fail_closed_receipt,
    )
    from calm.llm_computer.tests.test_hrm_text_158_activation_relief import (
        _activation_residual_live_tensor_proof,
    )

    events, zL_init_observation = _activation_residual_live_tensor_proof()
    fixture_receipt = build_activation_residuals_fail_closed_receipt(
        seam_events=events,
        zL_init_observation=zL_init_observation,
    )
    with pytest.raises(ValueError, match="fixture activation/residual fail-closed"):
        apply_live_activation_residuals_surface_overrides(fixture_receipt)


def test_live_r2a_applier_rejects_cpu_seam_receipt():
    with pytest.raises(ValueError, match="CPU production seam observation"):
        apply_live_activation_residuals_surface_overrides(_mint_r2a_cpu_seam_receipt())


def _mint_r2a_cpu_m1_lossless_equiv_receipt():
    from calm.hrm_text_158.native_full_stack.activation_residuals_m1_remat import (
        build_trainer_activation_residuals_lossless_equivalence_receipt,
    )
    from calm.llm_computer.tests.test_hrm_text_158_trainer_activation_residuals_m1_equiv import (
        _run_hrm_with_codec,
    )

    codec = _run_hrm_with_codec()
    from calm.hrm_text_158.native_full_stack.activation_relief import (
        zL_init_observation_from_hrm_module,
    )
    from calm.hrm_text_158.hrm import HierarchicalReasoningModel
    from calm.llm_computer.tests.test_hrm_text_158_activation_relief import _tiny_config

    hrm = HierarchicalReasoningModel(_tiny_config())
    return build_trainer_activation_residuals_lossless_equivalence_receipt(
        source_commit_sha="abc123",
        proof_command_argv=("test",),
        seam_events=codec.seam_events,
        zL_init_observation=zL_init_observation_from_hrm_module(hrm),
        telemetry=codec.telemetry(),
        main_path_proven=True,
        main_autograd_path_differs_from_baseline=True,
    )


def test_live_r2a_applier_rejects_cpu_lossless_equiv_receipt():
    with pytest.raises(ValueError, match="CPU lossless equivalence receipt"):
        apply_live_activation_residuals_surface_overrides(
            _mint_r2a_cpu_m1_lossless_equiv_receipt()
        )
