"""Focused tests for the full-sub2 runtime readiness gate."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

import calm.hrm_text_158.native_full_stack as native_full_stack
from calm.hrm_text_158.native_full_stack.full_sub2_runtime_readiness import (
    FIXTURE_MAIN_READY,
    FIXTURE_MISSING_ACTIVATIONS,
    FIXTURE_MISSING_ATTENTION,
    FIXTURE_MISSING_BACKWARD,
    FIXTURE_PRE_FULL_STACK_DIAGNOSTIC,
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
    SURFACE_FP_EXCEPTIONS_LEDGER,
    FullSub2RuntimeSurfaceReceipt,
    build_full_sub2_runtime_ready_for_science,
    fixture_full_sub2_runtime_ready_for_science,
    main_ready_fixture_surfaces,
    validate_full_sub2_runtime_ready_for_science_receipt,
)


SCRIPT = Path("scripts/hrm_text_158_full_sub2_runtime_readiness.py")


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
    assert native_full_stack.RUNTIME_CLASS_TRANSIENT_FP_DEBT == RUNTIME_CLASS_TRANSIENT_FP_DEBT
    assert "fixture_full_sub2_runtime_ready_for_science" in native_full_stack.__all__
    receipt = native_full_stack.fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_PRE_FULL_STACK_DIAGNOSTIC
    )
    assert receipt.ready_for_pre_full_stack_diagnostic is True


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


def test_readiness_classes_are_exact_five_class_prereg():
    assert FULL_SUB2_RUNTIME_CLASSIFICATIONS == (
        RUNTIME_CLASS_SUB2,
        RUNTIME_CLASS_EXPLICIT_EXCEPTION,
        RUNTIME_CLASS_TRANSIENT_FP_DEBT,
        RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
        RUNTIME_CLASS_MISSING,
    )

