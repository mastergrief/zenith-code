"""Focused tests for the full-sub2 runtime readiness gate."""
from __future__ import annotations

import json
from pathlib import Path
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
    build_full_sub2_runtime_ready_for_science,
    fixture_full_sub2_runtime_ready_for_science,
    gated_sub2_checkpoint_path_activation_residuals_blocked_surfaces,
    gated_sub2_checkpoint_path_attention_kv_blocked_surfaces,
    gated_sub2_checkpoint_path_backward_recompute_surfaces,
    gated_sub2_checkpoint_path_surfaces,
    main_ready_fixture_surfaces,
    validate_full_sub2_runtime_ready_for_science_receipt,
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


def test_readiness_classes_are_exact_five_class_prereg():
    assert FULL_SUB2_RUNTIME_CLASSIFICATIONS == (
        RUNTIME_CLASS_SUB2,
        RUNTIME_CLASS_EXPLICIT_EXCEPTION,
        RUNTIME_CLASS_TRANSIENT_FP_DEBT,
        RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
        RUNTIME_CLASS_MISSING,
    )
