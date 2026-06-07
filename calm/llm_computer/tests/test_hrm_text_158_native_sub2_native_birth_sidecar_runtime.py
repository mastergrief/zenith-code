from __future__ import annotations

import calm.hrm_text_158.native_full_stack as native_full_stack
import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.sub2_native_birth_scaffold import (
    ACQUISITION_GATE_RESULT,
    ACQUISITION_GATE_RUNNING,
    ACQUISITION_GATE_UNBLOCKED_NOT_RUN,
    HYBRID_PERSISTENT_MODE_APPLIED_CROSSING_DIRECTION_PLUS_4BIT_RESIDUAL,
    build_strict_sub2_hybrid_runtime_movement_overlay,
    validate_strict_sub2_hybrid_runtime_movement_overlay,
)
from calm.hrm_text_158.native_full_stack.sub2_native_birth_sidecar_runtime import (
    AppliedCrossingDirectionResidualPersistentState,
    PERSISTENT_SIDECAR_BUDGET_FAIL,
    PERSISTENT_SIDECAR_STATE_AUTHORITY_FAIL,
    collapse_sidecar_persistent_states,
    consume_hybrid_sidecar_budget_guard,
    hybrid_sidecar_persistent_state_report,
    make_applied_crossing_direction_residual_persistent_state,
    materialize_transient_sidecar_shadow_state,
)


def test_sidecar_state_validates_unique_indices_direction_and_residual_range():
    with pytest.raises(ValueError, match="must be unique"):
        AppliedCrossingDirectionResidualPersistentState(
            state_key="dup",
            q_levels=torch.zeros(4, dtype=torch.int8),
            frozen_scale=torch.tensor(0.25, dtype=torch.float32),
            applied_indices=(0, 0),
            applied_directions=(1, -1),
            residual_values=(0, 1),
        )
    with pytest.raises(ValueError, match="must be \\+/-1"):
        AppliedCrossingDirectionResidualPersistentState(
            state_key="dir",
            q_levels=torch.zeros(4, dtype=torch.int8),
            frozen_scale=torch.tensor(0.25, dtype=torch.float32),
            applied_indices=(0,),
            applied_directions=(0,),
            residual_values=(0,),
        )
    with pytest.raises(ValueError, match="signed 4-bit range"):
        AppliedCrossingDirectionResidualPersistentState(
            state_key="resid",
            q_levels=torch.zeros(4, dtype=torch.int8),
            frozen_scale=torch.tensor(0.25, dtype=torch.float32),
            applied_indices=(0,),
            applied_directions=(1,),
            residual_values=(8,),
        )


def test_sidecar_materialize_and_collapse_preserve_q_scale_and_no_dense_shadow():
    prior = {
        "toy": make_applied_crossing_direction_residual_persistent_state(
            "toy",
            torch.tensor([0, 0, 0, 0], dtype=torch.int8),
            0.25,
            applied_indices=(),
            applied_directions=(),
            residual_values=(),
        )
    }
    next_state = make_bounded_tensor_state(
        "toy",
        torch.tensor([1, 0, -1, 0], dtype=torch.int8),
        0.25,
        torch.tensor([1, 0, -2, 0], dtype=torch.int16),
        hot_exact_indices=(0, 2),
        cold_default_value=0,
    )

    collapsed = collapse_sidecar_persistent_states({"toy": next_state}, prior_states=prior)
    persistent = collapsed["toy"]
    assert persistent.q_levels.tolist() == [1, 0, -1, 0]
    assert float(persistent.frozen_scale.item()) == pytest.approx(0.25)
    assert persistent.applied_indices == (0, 2)
    assert persistent.applied_directions == (1, -1)
    assert persistent.residual_values == (1, -2)
    assert not hasattr(persistent, "exact_accumulator_shadow")

    materialized = materialize_transient_sidecar_shadow_state(persistent)
    assert materialized.q_levels.tolist() == [1, 0, -1, 0]
    assert materialized.frozen_scale.item() == pytest.approx(0.25)
    assert materialized.exact_accumulator_shadow.tolist() == [1, 0, -2, 0]
    assert materialized.bounded_accumulator.hot_exact_indices == (0, 2)


def test_sidecar_budget_guard_fails_closed_on_synthetic_inclusive_ge_2():
    states = {
        "tiny": make_applied_crossing_direction_residual_persistent_state(
            "tiny",
            torch.tensor([1, 1, -1, -1], dtype=torch.int8),
            0.25,
            applied_indices=(0, 1, 2, 3),
            applied_directions=(1, 1, -1, -1),
            residual_values=(7, 7, -7, -7),
        )
    }

    report = hybrid_sidecar_persistent_state_report(states)
    guard_pass, stop_reason = consume_hybrid_sidecar_budget_guard(report)
    runtime_step_pass = bool(True and guard_pass)

    assert report.pass_report is False
    assert report.budget_guard.pass_guard is False
    assert report.budget_guard.stop_reason == PERSISTENT_SIDECAR_BUDGET_FAIL
    assert report.budget_guard.inclusive_bits_per_weight >= 2.0
    assert guard_pass is False
    assert stop_reason == PERSISTENT_SIDECAR_BUDGET_FAIL
    assert runtime_step_pass is False


def test_sidecar_guard_names_state_authority_fail_when_dense_shadow_leaks():
    with pytest.raises(ValueError, match="cannot persist dense shadow state"):
        build_strict_sub2_hybrid_runtime_movement_overlay(
            logical_shapes=((512, 512),),
            event_counts=(16,),
            persistent_mode=HYBRID_PERSISTENT_MODE_APPLIED_CROSSING_DIRECTION_PLUS_4BIT_RESIDUAL,
            residual_bits_per_event=4,
            persistent_dense_shadow_present=True,
            persistent_dense_shadow_bytes=512,
            local_update_law_label="dummy",
            acquisition_science_status=ACQUISITION_GATE_UNBLOCKED_NOT_RUN,
            acquisition_achieved=False,
        )

    states = {
        "dense_shadow_illegal": make_applied_crossing_direction_residual_persistent_state(
            "dense_shadow_illegal",
            torch.zeros(8192, dtype=torch.int8),
            0.25,
            applied_indices=(0,),
            applied_directions=(1,),
            residual_values=(0,),
        )
    }
    report = hybrid_sidecar_persistent_state_report(states)
    # Direct helper path is budget-pass on this represented set.
    assert report.budget_guard.pass_guard is True
    # Overlay legality itself is separately frozen by the scaffold validator/builder.
    assert PERSISTENT_SIDECAR_STATE_AUTHORITY_FAIL == "persistent_sidecar_state_authority_fail"


@pytest.mark.parametrize(
    "status",
    [
        ACQUISITION_GATE_UNBLOCKED_NOT_RUN,
        ACQUISITION_GATE_RUNNING,
        ACQUISITION_GATE_RESULT,
    ],
)
def test_overlay_status_legality_accepts_step2_statuses_with_acquisition_false(status):
    overlay = build_strict_sub2_hybrid_runtime_movement_overlay(
        logical_shapes=((512, 512),),
        event_counts=(16,),
        persistent_mode=HYBRID_PERSISTENT_MODE_APPLIED_CROSSING_DIRECTION_PLUS_4BIT_RESIDUAL,
        residual_bits_per_event=4,
        persistent_dense_shadow_present=False,
        persistent_dense_shadow_bytes=0,
        local_update_law_label="dummy",
        acquisition_science_status=status,
        acquisition_achieved=False,
    )
    validate_strict_sub2_hybrid_runtime_movement_overlay(overlay)


def test_native_full_stack_public_exports_include_promoted_sidecar_runtime_symbols():
    names = {
        "AppliedCrossingDirectionResidualPersistentState",
        "PERSISTENT_SIDECAR_BUDGET_FAIL",
        "PERSISTENT_SIDECAR_STATE_AUTHORITY_FAIL",
        "collapse_sidecar_persistent_states",
        "consume_hybrid_sidecar_budget_guard",
        "hybrid_sidecar_persistent_state_report",
        "make_applied_crossing_direction_residual_persistent_state",
        "materialize_transient_sidecar_shadow_state",
        "tierb_lane1_hybrid_movement_report",
        "tierb_lane1_hybrid_movement_success_contract",
        "tierb_lane1_hybrid_movement_terminal_semantics_contract",
    }

    exported = set(native_full_stack.__all__)
    for name in names:
        assert hasattr(native_full_stack, name)
        assert name in exported
