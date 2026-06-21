"""B2-5c Step-1b-(3a) candidate global-cap trainer wiring tests (CPU-only)."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
    _tensor_sha256,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    apply_bounded_delta_vote_step,
    make_bounded_tensor_state,
    tensor_sha256,
)
from calm.hrm_text_158.native_full_stack.candidate_global_cap_bridge_reference import (
    BridgeFixtureSpec,
    _build_state,
    _vote_spec,
)
from calm.hrm_text_158.native_full_stack.candidate_global_cap_production_seam import (
    CandidateGlobalCapSeamEntry,
    apply_candidate_global_cap_production_seam,
)
from calm.hrm_text_158.native_full_stack.candidate_global_cap_trainer_wiring_receipt import (
    CANDIDATE_GLOBAL_CAP_TRAINER_WIRING_HARD_FALSE_FIELDS,
    CANDIDATE_GLOBAL_CAP_TRAINER_WIRING_NON_CLAIMS,
    build_candidate_global_cap_trainer_wiring_receipt,
    validate_candidate_global_cap_trainer_wiring_receipt,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import GlobalRateCapSpec
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec

GUARD_EXACT_SUBSTRING = (
    "candidate_mode local vote-update proof does not cover global cap"
)

SEAM_MODULE_SHA256 = (
    "162aaacbb29a21e7aa1152a9a3d610f2d2c64e987ae225bcb615c94fb084889b"
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


def _build_bridge_fixture_spec(state_key: str) -> BridgeFixtureSpec:
    if state_key == "A":
        return BridgeFixtureSpec(
            fixture_name="F_TWO_TENSOR_GLOBAL_A",
            fixture_role="representative_consumer",
            state_key="A",
            numel=8,
            acc_overrides={0: 9, 1: -9, 2: 9, 3: -9},
            sparse_votes={0: 2, 1: -2, 2: 2, 3: -2},
            hot_exact_indices=(0, 1, 2, 3),
            cap=2,
            max_abs_per_tensor=8,
        )
    if state_key == "B":
        return BridgeFixtureSpec(
            fixture_name="F_TWO_TENSOR_GLOBAL_B",
            fixture_role="representative_consumer",
            state_key="B",
            numel=8,
            acc_overrides={0: 16, 1: -16, 2: 16, 3: -16},
            sparse_votes={0: 2, 1: -2, 2: 2, 3: -2},
            hot_exact_indices=(0, 1, 2, 3),
            cap=2,
            max_abs_per_tensor=8,
        )
    raise ValueError(f"unsupported state_key {state_key!r}")


def _build_two_tensor_trainer_inputs() -> tuple[
    dict[str, object],
    dict[str, VoteUpdateSpec],
    dict[str, dict[int, int]],
    GlobalRateCapSpec,
]:
    tensor_states = {}
    vote_specs = {}
    sparse_events = {}
    for state_key in ("A", "B"):
        spec = _build_bridge_fixture_spec(state_key)
        vu = _build_state(
            spec.numel,
            acc_overrides=spec.acc_overrides,
            q_overrides=spec.q_overrides,
        )
        tensor_states[state_key] = make_bounded_tensor_state(
            state_key,
            vu.q_levels,
            0.5,
            vu.accumulators,
            hot_exact_indices=spec.hot_exact_indices,
        )
        vote_specs[state_key] = _vote_spec(max_abs_per_tensor=spec.max_abs_per_tensor)
        sparse_events[state_key] = dict(spec.sparse_votes)
    global_cap_spec = GlobalRateCapSpec(
        cap=2,
        step=1,
        ordering_seed=0,
        mutate_outputs=False,
    )
    return tensor_states, vote_specs, sparse_events, global_cap_spec


def _candidate_sparse_kwargs(
    tensor_states,
    vote_specs,
    sparse_events,
    *,
    global_cap_spec=None,
    seam_enabled: bool = False,
    **overrides,
):
    base = dict(
        tensor_states=tensor_states,
        votes_by_key=None,
        vote_specs_by_key=vote_specs,
        candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        candidate_sparse_vote_events_by_key=sparse_events,
        candidate_oracle_control_enabled=False,
        sparse_vote_authority_only=True,
        global_cap_spec=global_cap_spec,
        candidate_global_cap_production_seam_enabled=seam_enabled,
    )
    base.update(overrides)
    return base


def _artifacts_sha256(artifacts) -> str:
    digest = hashlib.sha256()
    digest.update(str(artifacts.applied_indices).encode("utf-8"))
    digest.update(_tensor_sha256(artifacts.restored_support).encode("utf-8"))
    digest.update(str(artifacts.proof.get("applied_row_identities_sha256", "")).encode("utf-8"))
    return digest.hexdigest()


def _cap_inputs_sha256(cap_inputs) -> str:
    digest = hashlib.sha256()
    for item in cap_inputs:
        digest.update(item.state_key.encode("utf-8"))
        digest.update(_tensor_sha256(item.plan.new_acc_i32).encode("utf-8"))
        digest.update(_tensor_sha256(item.plan.applied_indices).encode("utf-8"))
    return digest.hexdigest()


def _vote_update_state_sha256(vu_state) -> tuple[str, str]:
    return _tensor_sha256(vu_state.q_levels), _tensor_sha256(vu_state.accumulators)


_CAP_SUMMARY_KEYS: tuple[str, ...] = (
    "pre_cap_demand_sha256",
    "exact_shadow_full_demand_sha256",
    "exact_shadow_accepted_sha256",
    "exact_shadow_deferred_sha256",
    "global_pre_cap_would_apply_count",
    "global_rate_cap_accepted_count",
    "global_rate_cap_deferred_count",
    "accepted_fresh_count",
    "accepted_from_prior_deferred_count",
)


def _cap_summary_subset(summary: dict) -> dict:
    return {key: summary[key] for key in _CAP_SUMMARY_KEYS if key in summary}


def _install_seam_capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    import calm.hrm_text_158.native_full_stack.candidate_global_cap_production_seam as seam_module

    captured: dict[str, object] = {}
    real_apply = seam_module.apply_candidate_global_cap_production_seam

    def _capturing_apply(entries, global_cap_spec, **kwargs):
        result = real_apply(entries, global_cap_spec, **kwargs)
        captured["entries"] = dict(entries)
        captured["cap_inputs"] = result.cap_inputs
        captured["artifacts_by_key"] = dict(result.artifacts_by_key)
        captured["cap_result"] = result.cap_result
        return result

    monkeypatch.setattr(
        seam_module,
        "apply_candidate_global_cap_production_seam",
        _capturing_apply,
    )
    return captured


def _direct_seam_from_trainer_inputs(tensor_states, vote_specs, sparse_events, global_cap_spec):
    from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
        execute_direct_bounded_local_vote_update_candidate,
        encode_budget_capped_hybrid_reference,
    )

    entries: dict[str, CandidateGlobalCapSeamEntry] = {}
    for state_key, prior_tensor in sorted(tensor_states.items()):
        spec = _build_bridge_fixture_spec(state_key)
        vu = prior_tensor.vote_update_state()
        bounded = encode_budget_capped_hybrid_reference(
            vu,
            hot_exact_indices=spec.hot_exact_indices,
            cold_default_value=0,
        )
        candidate_result = execute_direct_bounded_local_vote_update_candidate(
            state_key=state_key,
            q_levels=prior_tensor.q_levels,
            bounded_accumulator=bounded,
            sparse_vote_events=sparse_events[state_key],
            vote_spec=vote_specs[state_key],
        )
        entries[state_key] = CandidateGlobalCapSeamEntry(
            prior_state=vu,
            candidate_result=candidate_result,
            vote_spec=vote_specs[state_key],
        )
    return apply_candidate_global_cap_production_seam(entries, global_cap_spec)


def test_default_off_exact_guard_substring():
    tensor_states, vote_specs, sparse_events, global_cap_spec = _build_two_tensor_trainer_inputs()
    with pytest.raises(ValueError, match=GUARD_EXACT_SUBSTRING):
        apply_bounded_delta_vote_step(
            **_candidate_sparse_kwargs(
                tensor_states,
                vote_specs,
                sparse_events,
                global_cap_spec=global_cap_spec,
                seam_enabled=False,
            ),
        )


def test_flag_kw_default_false_when_omitted():
    params = inspect.signature(apply_bounded_delta_vote_step).parameters
    assert "candidate_global_cap_production_seam_enabled" in params
    assert params["candidate_global_cap_production_seam_enabled"].default is False


def test_flag_absent_global_cap_none_byte_identical_candidate_path():
    tensor_states, vote_specs, sparse_events, _ = _build_two_tensor_trainer_inputs()
    kwargs = _candidate_sparse_kwargs(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec=None,
    )
    baseline = apply_bounded_delta_vote_step(**kwargs)
    with_kw = apply_bounded_delta_vote_step(
        **{**kwargs, "candidate_global_cap_production_seam_enabled": False},
    )
    assert baseline.global_summary == with_kw.global_summary
    assert baseline.deferred_backlog == with_kw.deferred_backlog
    for key in baseline.tensor_states:
        assert tensor_sha256(baseline.tensor_states[key].q_levels) == tensor_sha256(
            with_kw.tensor_states[key].q_levels,
        )
        assert baseline.tensor_stats[key]["q_sha256_after"] == with_kw.tensor_stats[key][
            "q_sha256_after"
        ]


def test_flag_on_q_acc_and_identities_match_direct_seam():
    tensor_states, vote_specs, sparse_events, global_cap_spec = _build_two_tensor_trainer_inputs()
    wired = apply_bounded_delta_vote_step(
        **_candidate_sparse_kwargs(
            tensor_states,
            vote_specs,
            sparse_events,
            global_cap_spec=global_cap_spec,
            seam_enabled=True,
        ),
    )
    direct = _direct_seam_from_trainer_inputs(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    direct_summary = direct.cap_result.step_summary
    for state_key in ("A", "B"):
        wired_q, wired_acc, _ = (
            wired.tensor_states[state_key].q_levels,
            wired.tensor_states[state_key].exact_accumulator_shadow,
            wired.tensor_stats[state_key],
        )
        seam_q, seam_acc, _ = direct.q_acc_by_key[state_key]
        assert tensor_sha256(wired_q) == tensor_sha256(seam_q)
        assert tensor_sha256(wired_acc) == tensor_sha256(seam_acc)
    assert _cap_summary_subset(wired.global_summary) == _cap_summary_subset(direct_summary)
    assert wired.global_summary["exact_shadow_accepted_sha256"] == direct_summary[
        "exact_shadow_accepted_sha256"
    ]
    assert wired.global_summary["exact_shadow_deferred_sha256"] == direct_summary[
        "exact_shadow_deferred_sha256"
    ]
    assert wired.global_summary["pre_cap_demand_sha256"] == direct_summary[
        "pre_cap_demand_sha256"
    ]
    assert wired.global_summary["global_rate_cap_accepted_count"] == direct_summary[
        "global_rate_cap_accepted_count"
    ]
    assert wired.global_summary["global_rate_cap_deferred_count"] == direct_summary[
        "global_rate_cap_deferred_count"
    ]
    assert _global_accepted_identities(direct.cap_result) == {
        (row.state_key, int(row.flat_index)) for row in direct.cap_result.accepted_rows
    }


def test_flag_on_deferred_backlog_matches_seam_cap_result():
    tensor_states, vote_specs, sparse_events, global_cap_spec = _build_two_tensor_trainer_inputs()
    wired = apply_bounded_delta_vote_step(
        **_candidate_sparse_kwargs(
            tensor_states,
            vote_specs,
            sparse_events,
            global_cap_spec=global_cap_spec,
            seam_enabled=True,
        ),
    )
    direct = _direct_seam_from_trainer_inputs(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    assert wired.deferred_backlog == direct.cap_result.deferred_backlog


def test_flag_on_global_summary_carries_cap_summary():
    tensor_states, vote_specs, sparse_events, global_cap_spec = _build_two_tensor_trainer_inputs()
    wired = apply_bounded_delta_vote_step(
        **_candidate_sparse_kwargs(
            tensor_states,
            vote_specs,
            sparse_events,
            global_cap_spec=global_cap_spec,
            seam_enabled=True,
        ),
    )
    direct = _direct_seam_from_trainer_inputs(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    direct_summary = direct.cap_result.step_summary
    assert wired.global_summary.get("global_rate_cap_enabled") is True
    assert wired.global_summary.get("candidate_global_cap_production_seam_enabled") is True
    assert _cap_summary_subset(wired.global_summary) == _cap_summary_subset(direct_summary)
    ordering = wired.global_summary.get("global_rate_cap_ordering_summary") or {}
    direct_ordering = direct_summary.get("global_rate_cap_ordering_summary") or {}
    assert ordering.get("global_indices_sha256") == direct_ordering.get("global_indices_sha256")
    assert ordering.get("selected_count") == direct_ordering.get("selected_count")
    assert ordering.get("deferred_count") == direct_ordering.get("deferred_count")


def test_flag_on_tensor_stats_from_seam_q_acc_not_candidate_local():
    tensor_states, vote_specs, sparse_events, global_cap_spec = _build_two_tensor_trainer_inputs()
    wired = apply_bounded_delta_vote_step(
        **_candidate_sparse_kwargs(
            tensor_states,
            vote_specs,
            sparse_events,
            global_cap_spec=global_cap_spec,
            seam_enabled=True,
        ),
    )
    direct = _direct_seam_from_trainer_inputs(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    local_only = apply_bounded_delta_vote_step(
        **_candidate_sparse_kwargs(tensor_states, vote_specs, sparse_events),
    )
    for state_key in ("A", "B"):
        seam_q, seam_acc, _ = direct.q_acc_by_key[state_key]
        assert wired.tensor_stats[state_key]["q_sha256_after"] == tensor_sha256(seam_q)
        assert wired.tensor_stats[state_key]["exact_accumulator_shadow_sha256_after"] == (
            tensor_sha256(seam_acc)
        )
        assert wired.tensor_stats[state_key]["q_sha256_after"] != local_only.tensor_stats[state_key][
            "q_sha256_after"
        ]


def test_wired_cap_inputs_and_artifacts_match_direct_seam(monkeypatch: pytest.MonkeyPatch):
    tensor_states, vote_specs, sparse_events, global_cap_spec = _build_two_tensor_trainer_inputs()
    original_prior_sha_by_key = {
        state_key: _vote_update_state_sha256(tensor_states[state_key].vote_update_state())
        for state_key in ("A", "B")
    }
    captured = _install_seam_capture(monkeypatch)
    wired = apply_bounded_delta_vote_step(
        **_candidate_sparse_kwargs(
            tensor_states,
            vote_specs,
            sparse_events,
            global_cap_spec=global_cap_spec,
            seam_enabled=True,
        ),
    )
    assert captured, "wired trainer must invoke apply_candidate_global_cap_production_seam"
    direct = _direct_seam_from_trainer_inputs(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    captured_entries = captured["entries"]
    captured_cap_inputs = captured["cap_inputs"]
    captured_artifacts = captured["artifacts_by_key"]
    for state_key in ("A", "B"):
        entry = captured_entries[state_key]
        q_sha, acc_sha = _vote_update_state_sha256(entry.prior_state)
        assert (q_sha, acc_sha) == original_prior_sha_by_key[state_key]
        assert _artifacts_sha256(captured_artifacts[state_key]) == _artifacts_sha256(
            direct.artifacts_by_key[state_key],
        )
    assert _cap_inputs_sha256(captured_cap_inputs) == _cap_inputs_sha256(direct.cap_inputs)
    assert wired.deferred_backlog == direct.cap_result.deferred_backlog


def test_hard_false_non_claims_receipt():
    receipt = build_candidate_global_cap_trainer_wiring_receipt(
        composition_path_exists_in_code=True,
        composition_path_default_active=False,
        trainer_candidate_global_cap_composition_active=False,
        flag_default_off_guard_preserved=True,
        seam_module_byte_frozen=True,
    )
    validate_candidate_global_cap_trainer_wiring_receipt(receipt)
    assert receipt.composition_path_exists_in_code is True
    assert receipt.composition_path_default_active is False
    assert receipt.non_claims == CANDIDATE_GLOBAL_CAP_TRAINER_WIRING_NON_CLAIMS
    for field_name in CANDIDATE_GLOBAL_CAP_TRAINER_WIRING_HARD_FALSE_FIELDS:
        assert getattr(receipt, field_name) is False


def test_seam_module_byte_frozen_import_only():
    repo_root = Path(__file__).resolve().parents[2]
    seam_path = (
        repo_root
        / "hrm_text_158"
        / "native_full_stack"
        / "candidate_global_cap_production_seam.py"
    )
    import hashlib as _hashlib

    actual = _hashlib.sha256(seam_path.read_bytes()).hexdigest()
    assert actual == SEAM_MODULE_SHA256


@pytest.mark.skip(
    reason=(
        "B-lite shape compat: "
        "test_hrm_text_158_candidate_global_cap_b_lite_native_shape_compat_gpu.py (Step-1b-3b)"
    ),
)
def test_b_lite_native_shape_compat_deferred_to_3b_gpu():
    """Placeholder: native vs CPU selection on seam cap_inputs runs in 3b only."""
    pytest.skip("3b GPU lane required")


@pytest.mark.skip(
    reason=(
        "Wired-trainer GPU e2e: "
        "test_hrm_text_158_candidate_global_cap_wired_trainer_e2e_gpu.py (Step-1b-3c)"
    ),
)
def test_wired_trainer_e2e_deferred_to_3c_gpu():
    """Placeholder: flag-on wired-trainer GPU e2e parity runs in 3c only."""
    pytest.skip("3c GPU lane required")
