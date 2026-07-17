"""CPU-static isolation + precedence tests (PLAN v6 D1)."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import BoundedDeltaAccumulatorState
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import BoundedDeltaTensorState
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_integrity_proofs import (
    ASYMMETRY,
    INTEGRITY,
    IntegrityProofError,
    NULL_OR_HARMFUL,
    PRESENT,
    assert_zero_cross_arm_storage_overlap,
    canonical_result_forbidden,
    hash_arm_state_manifest,
    reject_shallow_shared_storage_fork,
    terminal_precedence_classify,
    untouched_sentinel_report,
    within_state_alias_topology,
)

MOD = (
    Path(__file__).resolve().parents[2]
    / "hrm_text_158/native_full_stack/signed_utility_fixed_state_integrity_proofs.py"
)


def test_loc_budget():
    assert sum(1 for _ in MOD.open()) <= 280


def _state(q: torch.Tensor, acc: torch.Tensor):
    return {"q_levels": q, "exact_accumulator_shadow": acc, "frozen_scale": torch.tensor(1.0)}


def _bounded_state(**over) -> BoundedDeltaTensorState:
    q = torch.zeros(2, dtype=torch.int8)
    shadow = torch.zeros(2, dtype=torch.int16)
    acc = BoundedDeltaAccumulatorState(
        logical_shape=(2,),
        cold_default_value=0,
        hot_exact_indices=(),
        hot_exact_values=(),
    )
    kwargs = dict(
        state_key="toy",
        q_levels=q,
        frozen_scale=torch.tensor(1.0),
        bounded_accumulator=acc,
        exact_accumulator_shadow=shadow,
        bounded_accumulator_fresh_for_exact_shadow=True,
        event_coded_live_carrier=None,
    )
    kwargs.update(over)
    return BoundedDeltaTensorState(**kwargs)


def test_deep_fork_passes_and_records_within_state_aliases():
    q = torch.zeros(8, dtype=torch.int8)
    acc = torch.zeros(8, dtype=torch.int16)
    # intentional within-state alias: view shares storage with q
    st = {"q_levels": q, "q_view": q.view(-1), "exact_accumulator_shadow": acc, "frozen_scale": torch.tensor(1.0)}
    topo = within_state_alias_topology(st)
    assert topo["shared_storage_groups"]
    a = _state(q.clone(), acc.clone())
    b = _state(q.clone(), acc.clone())
    report = assert_zero_cross_arm_storage_overlap({"prod": a, "inv": b})
    assert report["ok"] is True


def test_shallow_copy_shared_storage_rejected():
    q = torch.zeros(4, dtype=torch.int8)
    acc = torch.zeros(4, dtype=torch.int16)
    original = _state(q, acc)
    shallow = {"q_levels": q, "exact_accumulator_shadow": acc, "frozen_scale": original["frozen_scale"]}
    with pytest.raises(IntegrityProofError, match="cross_arm_storage_overlap"):
        reject_shallow_shared_storage_fork(original, shallow)


def test_clone_with_shared_q_storage_rejected_across_arms():
    q = torch.zeros(4, dtype=torch.int8)
    a = _state(q, torch.zeros(4, dtype=torch.int16))
    b = _state(q, torch.zeros(4, dtype=torch.int16))  # shared q storage
    with pytest.raises(IntegrityProofError, match="cross_arm_storage_overlap"):
        assert_zero_cross_arm_storage_overlap({"prod": a, "noop": b})


def test_untouched_sentinel_drift_and_hash_stable():
    a = _state(torch.zeros(2, dtype=torch.int8), torch.zeros(2, dtype=torch.int16))
    h0 = hash_arm_state_manifest(a)
    before = {"noop": h0, "parent": "p" * 64}
    after_ok = {"noop": h0, "parent": "p" * 64}
    assert untouched_sentinel_report(before=before, after=after_ok, required_unchanged=["noop", "parent"])["ok"]
    a["q_levels"][0] = 1
    h1 = hash_arm_state_manifest(a)
    with pytest.raises(IntegrityProofError, match="untouched_sentinel_drift"):
        untouched_sentinel_report(
            before=before, after={"noop": h1, "parent": "p" * 64}, required_unchanged=["noop", "parent"]
        )


def test_bounded_state_non_tensor_metadata_changes_hash():
    st = _bounded_state()
    h0 = hash_arm_state_manifest(st)
    st_fresh = dataclasses.replace(
        st,
        bounded_accumulator_fresh_for_exact_shadow=False,
        bounded_accumulator_rebuild_cold_default_value=3,
    )
    assert hash_arm_state_manifest(st_fresh) != h0
    st_key = dataclasses.replace(st, state_key="other")
    assert hash_arm_state_manifest(st_key) != h0
    st_acc = dataclasses.replace(
        st,
        bounded_accumulator=BoundedDeltaAccumulatorState(
            logical_shape=(2,),
            cold_default_value=0,
            hot_exact_indices=(0,),
            hot_exact_values=(1,),
            cold_exception_indices=(1,),
            cold_exception_values=(2,),
        ),
    )
    assert hash_arm_state_manifest(st_acc) != h0
    st_rebuild = dataclasses.replace(
        st,
        bounded_accumulator_rebuild_hot_exact_indices=(0,),
        bounded_accumulator_rebuild_cold_default_value=7,
    )
    assert hash_arm_state_manifest(st_rebuild) != h0


def test_precedence_integrity_beats_asymmetry_when_both_present():
    assert (
        terminal_precedence_classify(
            integrity_failure=True, asymmetry_failure=True, empty_applied=False, science_classifier=PRESENT
        )
        == INTEGRITY
    )


def test_empty_applied_is_integrity_not_null():
    assert (
        terminal_precedence_classify(
            integrity_failure=False, asymmetry_failure=False, empty_applied=True, science_classifier=NULL_OR_HARMFUL
        )
        == INTEGRITY
    )


def test_science_only_after_integrity_and_asymmetry_clear():
    assert (
        terminal_precedence_classify(
            integrity_failure=False, asymmetry_failure=False, empty_applied=False, science_classifier=PRESENT
        )
        == PRESENT
    )
    assert (
        terminal_precedence_classify(
            integrity_failure=False, asymmetry_failure=True, empty_applied=False, science_classifier=PRESENT
        )
        == ASYMMETRY
    )


def test_canonical_result_forbidden_classes():
    assert canonical_result_forbidden("preflight_execution_receipt")
    assert canonical_result_forbidden("timeout_exit_124")
    assert canonical_result_forbidden("OOM")
    assert canonical_result_forbidden("crash_before_terminal_classifier")
    assert not canonical_result_forbidden("science_complete")
