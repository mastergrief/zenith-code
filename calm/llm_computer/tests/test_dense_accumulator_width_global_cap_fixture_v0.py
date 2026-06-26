"""O5 global-cap fixture tests for dense accumulator width parity screen."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from calm.hrm_text_158.native_full_stack.dense_accumulator_width_parity_screen import (
    CLASSIFIER_C3_MISSING_OBSERVABLES,
    MANDATORY_WIDTH_GRID,
    O5FixtureResult,
    build_o5_fixture_result,
    measure_o5_global_cap_surface_for_width,
    run_width_parity_screen,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import apply_global_rate_cap_reference


def test_o5_fixture_uses_w_specific_plans_not_density_module() -> None:
    """W-variant surfaces must come from width-scoped clip specs + apply_global_rate_cap_reference."""

    surface16 = measure_o5_global_cap_surface_for_width(16)
    surface5 = measure_o5_global_cap_surface_for_width(5)
    assert surface16.width == 16
    assert surface5.width == 5
    assert surface16.global_cap_row_count > 0
    assert surface5.global_cap_row_count > 0


def test_o5_fixture_observed_with_w_specific_cap_inputs() -> None:
    result = build_o5_fixture_result(widths=MANDATORY_WIDTH_GRID)
    assert result.observed is True
    assert result.reason == "w_specific_plan_global_cap_fixture"
    assert len(result.surfaces_by_width) == len(MANDATORY_WIDTH_GRID)
    ref = next(surface for surface in result.surfaces_by_width if surface.width == 16)
    w8 = next(surface for surface in result.surfaces_by_width if surface.width == 8)
    assert w8.accepted_identities == ref.accepted_identities
    assert w8.deferred_identities == ref.deferred_identities


def test_o5_missing_observables_when_fixture_construction_fails() -> None:
    with patch(
        "calm.hrm_text_158.native_full_stack.dense_accumulator_width_parity_screen.measure_o5_global_cap_surface_for_width",
        side_effect=RuntimeError("cannot build w-specific plan"),
    ):
        result = build_o5_fixture_result(widths=(16, 8))
    assert result.observed is False
    assert "o5_fixture_construction_failed" in result.reason


def test_screen_missing_o5_does_not_claim_full_contract_pass() -> None:
    with patch(
        "calm.hrm_text_158.native_full_stack.dense_accumulator_width_parity_screen.build_o5_fixture_result",
        return_value=O5FixtureResult(
            observed=False,
            reason="fixture_not_constructible",
            reference_width=16,
            surfaces_by_width=(),
            o5_drift_vs_reference=(),
        ),
    ):
        screen = run_width_parity_screen()
    assert screen.classifier == CLASSIFIER_C3_MISSING_OBSERVABLES


def test_apply_global_rate_cap_reference_is_real_import_surface() -> None:
    assert callable(apply_global_rate_cap_reference)
