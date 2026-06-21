"""B2-5c Step-1b-(1) candidate→global-cap production seam tests (CPU-only)."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    _tensor_sha256,
    encode_budget_capped_hybrid_reference,
    execute_direct_bounded_local_vote_update_candidate,
)
from calm.hrm_text_158.native_full_stack.candidate_global_cap_bridge_reference import (
    BridgeFixtureSpec,
    assert_structural_candidate_global_cap_reject,
    compose_candidate_global_cap_bridge_reference,
    run_bridge_fixture,
    _build_state,
    _vote_spec,
)
from calm.hrm_text_158.native_full_stack.candidate_global_cap_production_seam import (
    PRODUCTION_SEAM_HARD_FALSE_FIELDS,
    PRODUCTION_SEAM_NON_CLAIMS,
    CandidateGlobalCapSeamEntry,
    apply_candidate_global_cap_production_seam,
    apply_candidate_global_cap_production_seam_single,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapSpec,
    apply_global_rate_cap_reference,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    LOCAL_SELECTION_ORDER_DETERMINISTIC_HASH_MATCHED,
)

GUARD_EXACT_SUBSTRING = (
    "candidate_mode local vote-update proof does not cover global cap"
)


def _row_identities(result, state_key: str) -> tuple[list[int], list[int]]:
    accepted = [
        int(row.flat_index)
        for row in result.accepted_rows
        if row.state_key == state_key
    ]
    deferred = [
        int(row.flat_index)
        for row in result.deferred_rows
        if row.state_key == state_key
    ]
    return accepted, deferred


def _global_accepted_identities(result) -> set[tuple[str, int]]:
    return {(row.state_key, int(row.flat_index)) for row in result.accepted_rows}


def _build_candidate_entry(spec: BridgeFixtureSpec) -> CandidateGlobalCapSeamEntry:
    state = _build_state(
        spec.numel,
        acc_overrides=spec.acc_overrides,
        q_overrides=spec.q_overrides,
    )
    vote_spec = _vote_spec(max_abs_per_tensor=spec.max_abs_per_tensor)
    bounded = encode_budget_capped_hybrid_reference(
        state,
        hot_exact_indices=spec.hot_exact_indices,
        cold_default_value=0,
    )
    candidate_result = execute_direct_bounded_local_vote_update_candidate(
        state_key=spec.state_key,
        q_levels=state.q_levels,
        bounded_accumulator=bounded,
        sparse_vote_events=spec.sparse_votes,
        vote_spec=vote_spec,
    )
    return CandidateGlobalCapSeamEntry(
        prior_state=state,
        candidate_result=candidate_result,
        vote_spec=vote_spec,
    )


def _build_two_tensor_global_entries() -> tuple[
    dict[str, CandidateGlobalCapSeamEntry],
    GlobalRateCapSpec,
    GlobalRateCapSpec,
]:
    entry_a = _build_candidate_entry(
        BridgeFixtureSpec(
            fixture_name="F_TWO_TENSOR_GLOBAL_A",
            fixture_role="representative_consumer",
            state_key="A",
            numel=8,
            acc_overrides={0: 9, 1: -9, 2: 9, 3: -9},
            sparse_votes={0: 2, 1: -2, 2: 2, 3: -2},
            hot_exact_indices=(0, 1, 2, 3),
            cap=2,
            max_abs_per_tensor=8,
        ),
    )
    entry_b = _build_candidate_entry(
        BridgeFixtureSpec(
            fixture_name="F_TWO_TENSOR_GLOBAL_B",
            fixture_role="representative_consumer",
            state_key="B",
            numel=8,
            acc_overrides={0: 16, 1: -16, 2: 16, 3: -16},
            sparse_votes={0: 2, 1: -2, 2: 2, 3: -2},
            hot_exact_indices=(0, 1, 2, 3),
            cap=2,
            max_abs_per_tensor=8,
        ),
    )
    global_cap_spec = GlobalRateCapSpec(
        cap=2,
        step=1,
        ordering_seed=0,
        mutate_outputs=False,
    )
    per_state_cap_spec = GlobalRateCapSpec(
        cap=2,
        step=1,
        ordering_seed=0,
        mutate_outputs=False,
    )
    return {"A": entry_a, "B": entry_b}, global_cap_spec, per_state_cap_spec


def test_trainer_guard_unchanged_exact_error():
    try:
        assert_structural_candidate_global_cap_reject()
    except AssertionError as exc:
        raise AssertionError(
            f"expected structural global-cap rejection with guard substring, got {exc!r}",
        ) from exc
    repo_root = Path(__file__).resolve().parents[2]
    trainer_path = (
        repo_root / "hrm_text_158" / "native_full_stack" / "bounded_delta_learner.py"
    )
    source = trainer_path.read_text(encoding="utf-8")
    assert GUARD_EXACT_SUBSTRING in source


def test_seam_not_imported_by_bounded_delta_learner():
    repo_root = Path(__file__).resolve().parents[2]
    trainer_path = (
        repo_root / "hrm_text_158" / "native_full_stack" / "bounded_delta_learner.py"
    )
    source = trainer_path.read_text(encoding="utf-8")
    assert "candidate_global_cap_production_seam" not in source
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    }
    assert "candidate_global_cap_production_seam" not in imported


def test_seam_admission_rejects_replay_pc_deferred_front_c_alternate_ordering():
    entry = _build_candidate_entry(
        BridgeFixtureSpec(
            fixture_name="F_ADMISSION_PROBE",
            fixture_role="representative_consumer",
            state_key="F_ADMISSION_PROBE",
            numel=4,
            acc_overrides={0: 9, 2: -9},
            sparse_votes={0: 2, 2: -2},
            hot_exact_indices=(0, 2),
            cap=10,
            max_abs_per_tensor=4,
        ),
    )
    cap_spec = GlobalRateCapSpec(cap=1, step=1, ordering_seed=0, mutate_outputs=False)
    base_kwargs = {
        "entries": {"F_ADMISSION_PROBE": entry},
        "global_cap_spec": cap_spec,
    }
    with pytest.raises(ValueError, match="deferred backlog"):
        apply_candidate_global_cap_production_seam(
            **base_kwargs,
            deferred_backlog={"F_ADMISSION_PROBE": {}},
        )
    with pytest.raises(ValueError, match="replay/pc auxiliary paths"):
        apply_candidate_global_cap_production_seam(
            **base_kwargs,
            replay_ce_veto_votes_by_key={"F_ADMISSION_PROBE": torch.zeros(4, dtype=torch.int16)},
        )
    with pytest.raises(ValueError, match="replay/pc auxiliary paths"):
        apply_candidate_global_cap_production_seam(
            **base_kwargs,
            pc_aux_votes_by_key={"F_ADMISSION_PROBE": torch.zeros(4, dtype=torch.int16)},
        )
    with pytest.raises(ValueError, match="front_c live identity observation"):
        apply_candidate_global_cap_production_seam(
            **base_kwargs,
            front_c_identity_observer=object(),
        )
    with pytest.raises(ValueError, match="alternate local ordering"):
        apply_candidate_global_cap_production_seam(
            **base_kwargs,
            local_selection_ordering_mode=LOCAL_SELECTION_ORDER_DETERMINISTIC_HASH_MATCHED,
        )


def test_seam_mutating_mode_applies_cap():
    spec = BridgeFixtureSpec(
        fixture_name="F_MUTATING",
        fixture_role="representative_consumer",
        state_key="F_MUTATING",
        numel=8,
        acc_overrides={0: 9, 1: -9, 2: 9, 3: -9},
        sparse_votes={0: 2, 1: -2, 2: 2, 3: -2},
        hot_exact_indices=(0, 1, 2, 3),
        cap=2,
        max_abs_per_tensor=8,
    )
    entry = _build_candidate_entry(spec)
    cap_spec = GlobalRateCapSpec(
        cap=2,
        step=1,
        ordering_seed=0,
        mutate_outputs=True,
    )
    seam_result = apply_candidate_global_cap_production_seam_single(
        spec.state_key,
        entry,
        cap_spec,
    )
    accepted, deferred = _row_identities(seam_result.cap_result, spec.state_key)
    q_out, acc_out, _ = seam_result.q_acc_by_key[spec.state_key]
    prior_q = entry.prior_state.q_levels

    bridge_plan = seam_result.cap_inputs[0].plan
    plan_new_acc = bridge_plan.new_acc_i32.flatten()
    for index in accepted:
        direction = int(
            entry.candidate_result.next_q_levels.flatten()[index].item()
            - prior_q.flatten()[index].item(),
        )
        assert direction in (-1, 1)
        assert int(q_out.flatten()[index].item()) == int(prior_q.flatten()[index].item()) + direction
        assert int(q_out.flatten()[index].item()) != int(prior_q.flatten()[index].item())
    for index in deferred:
        assert int(q_out.flatten()[index].item()) == int(prior_q.flatten()[index].item())
        assert int(acc_out.flatten()[index].item()) == int(plan_new_acc[index].item())
    assert len(accepted) == 2
    assert len(deferred) == 2


def test_seam_multi_tensor_global_cap_selection():
    entries, global_cap_spec, per_state_cap_spec = _build_two_tensor_global_entries()
    seam_result = apply_candidate_global_cap_production_seam(entries, global_cap_spec)
    assert len(seam_result.cap_inputs) == 2
    assert seam_result.summary["production_seam_cap_input_count"] == 2

    assert seam_result.magnitude_regime_by_key["A"] == "no_clip_exact_add_back"
    assert seam_result.magnitude_regime_by_key["B"] == "no_clip_exact_add_back"

    global_accepted = _global_accepted_identities(seam_result.cap_result)
    assert len(global_accepted) == 2

    per_state_accepted: set[tuple[str, int]] = set()
    cap_inputs_by_key = {item.state_key: item for item in seam_result.cap_inputs}
    for state_key in entries:
        cap_input = cap_inputs_by_key[state_key]
        per_state_result = apply_global_rate_cap_reference(
            [cap_input],
            per_state_cap_spec,
        )
        per_state_accepted.update(_global_accepted_identities(per_state_result))

    assert len(per_state_accepted) == 4
    assert global_accepted != per_state_accepted
    assert all(key[0] == "B" for key in global_accepted)


def test_seam_prior_state_inputs_unmutated():
    entries, global_cap_spec, _ = _build_two_tensor_global_entries()
    q_before = {
        key: _tensor_sha256(entry.prior_state.q_levels) for key, entry in entries.items()
    }
    acc_before = {
        key: _tensor_sha256(entry.prior_state.accumulators)
        for key, entry in entries.items()
    }
    seam_result = apply_candidate_global_cap_production_seam(
        entries,
        GlobalRateCapSpec(
            cap=2,
            step=1,
            ordering_seed=0,
            mutate_outputs=True,
        ),
    )
    for key in entries:
        assert _tensor_sha256(entries[key].prior_state.q_levels) == q_before[key]
        assert _tensor_sha256(entries[key].prior_state.accumulators) == acc_before[key]
        assert seam_result.prior_state_q_sha256_by_key[key] == q_before[key]
        assert seam_result.prior_state_acc_sha256_by_key[key] == acc_before[key]


def test_seam_single_state_regression_matches_step1a_compose():
    spec = BridgeFixtureSpec(
        fixture_name="F_BRIDGE_MINIMAL",
        fixture_role="representative_consumer",
        state_key="F_BRIDGE_MINIMAL",
        numel=4,
        acc_overrides={0: 9, 2: -9},
        sparse_votes={0: 2, 2: -2},
        hot_exact_indices=(0, 2),
        cap=10,
        max_abs_per_tensor=4,
    )
    entry = _build_candidate_entry(spec)
    cap_spec = GlobalRateCapSpec(
        cap=int(spec.cap),
        step=1,
        ordering_seed=0,
        mutate_outputs=False,
    )
    seam_result = apply_candidate_global_cap_production_seam_single(
        spec.state_key,
        entry,
        cap_spec,
    )
    state = entry.prior_state
    bounded = encode_budget_capped_hybrid_reference(
        state,
        hot_exact_indices=spec.hot_exact_indices,
        cold_default_value=0,
    )
    _, compose_result, _ = compose_candidate_global_cap_bridge_reference(
        state=state,
        sparse_votes=spec.sparse_votes,
        vote_spec=entry.vote_spec,
        cap_spec=cap_spec,
        state_key=spec.state_key,
        bounded=bounded,
        candidate_result=entry.candidate_result,
    )
    seam_accepted, seam_deferred = _row_identities(
        seam_result.cap_result,
        spec.state_key,
    )
    compose_accepted, compose_deferred = _row_identities(
        compose_result,
        spec.state_key,
    )
    assert seam_accepted == compose_accepted
    assert seam_deferred == compose_deferred


def test_hard_false_non_claims_and_structural_guard():
    measurement = run_bridge_fixture(
        BridgeFixtureSpec(
            fixture_name="F_CLIP_BOUNDARY",
            fixture_role="classifier_negative",
            state_key="F_CLIP_BOUNDARY",
            numel=4,
            acc_overrides={0: 18, 2: -18},
            sparse_votes={0: 2, 2: -2},
            hot_exact_indices=(0, 2),
            cap=10,
            max_abs_per_tensor=4,
        ),
    )
    assert measurement.add_back_clip_boundary_reconciliation is True

    entry = _build_candidate_entry(
        BridgeFixtureSpec(
            fixture_name="F_CLIP_BOUNDARY",
            fixture_role="classifier_negative",
            state_key="F_CLIP_BOUNDARY",
            numel=4,
            acc_overrides={0: 18, 2: -18},
            sparse_votes={0: 2, 2: -2},
            hot_exact_indices=(0, 2),
            cap=10,
            max_abs_per_tensor=4,
        ),
    )
    seam_result = apply_candidate_global_cap_production_seam_single(
        "F_CLIP_BOUNDARY",
        entry,
        GlobalRateCapSpec(cap=10, step=1, ordering_seed=0, mutate_outputs=False),
    )
    assert (
        seam_result.magnitude_regime_by_key["F_CLIP_BOUNDARY"]
        == "clip_boundary_reconciliation"
    )
    assert seam_result.composition_path_exists is False
    for field in PRODUCTION_SEAM_HARD_FALSE_FIELDS:
        assert getattr(seam_result, field) is False
    for claim in PRODUCTION_SEAM_NON_CLAIMS:
        assert claim in seam_result.non_claims

    repo_root = Path(__file__).resolve().parents[2]
    module_dir = repo_root / "hrm_text_158" / "native_full_stack"
    seam_source = (module_dir / "candidate_global_cap_production_seam.py").read_text(
        encoding="utf-8",
    )
    assert "native_dispatch" not in seam_source
    assert_structural_candidate_global_cap_reject()
