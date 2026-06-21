"""B2-5c/R4 wired MIXED-REGIME GPU cap-apply mutation equivalence vs CPU seam reference.

Equivalence/compat-only on F_MIXED_REGIME_GLOBAL: GPU-routed global-cap APPLY within
the flag-ON wired trainer path == CPU seam reference (q/acc/backlog/cap-summary) in
one heterogeneous step across >=3 state keys where accepted rows span BOTH
clip_boundary_reconciliation AND no_clip_exact_add_back regimes.

Uses env-gated CUDA REFERENCE q/acc apply on seam-built cap_inputs — NOT native
q/acc Triton proof. NOT full-trainer-on-GPU / native-candidate-GPU / readiness /
acquisition / selection_parity_pass / optimizer_credit_state /
global_cap_margin_only mint.

GPU execution is a separate +1 on hrm_text_158_gpu0 after the R4 diff gate.
"""
from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

import calm.hrm_text_158.native_full_stack.candidate_global_cap_production_seam as seam_module
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
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapOrderingMode,
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    select_global_rate_cap_rows,
    tensor_offsets_for_vote_update_states,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import RUN_GPU_GLOBAL_RATE_CAP_ENV
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_dispatch import (
    RUN_GPU_GLOBAL_RATE_CAP_NATIVE_SELECTION_ENV,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_packed_key_scaffold import (
    _TRITON_AVAILABLE,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    RUN_GPU_Q_ACC_APPLY_ENV,
    VoteUpdateSpec,
)
from calm.llm_computer.tests.test_hrm_text_158_candidate_global_cap_trainer_wiring import (
    _CAP_SUMMARY_KEYS,
    _candidate_sparse_kwargs,
    _cap_summary_subset,
)
from calm.llm_computer.tests.test_hrm_text_158_candidate_global_cap_wired_trainer_e2e_gpu import (
    WIRED_E2E_CAPTURED,
    WIRED_E2E_TELEMETRY,
    _assert_cap_inputs_identity_match,
    _inputs_on_cuda,
    _install_gpu_cap_route_patch,
    _mirror_cap_inputs_to_cuda,
    _rebuild_independent_seam_cap_inputs,
    _require_wired_e2e_gpu_lane_or_skip,
    _restore_cpu_cap_route_patch,
)

MIXED_REGIME_STATE_KEYS: tuple[str, ...] = ("A", "B", "C")
MIXED_REGIME_THRESHOLD = 10
MIXED_REGIME_CLIP_EDGE = 9

PINNED_ACCEPTED: frozenset[tuple[str, int]] = frozenset({("A", 0), ("B", 0)})
PINNED_DEFERRED_BY_KEY: dict[str, set[int]] = {
    "A": set(),
    "B": set(),
    "C": {0},
}
PINNED_ORDERED_ROWS: tuple[tuple[str, int, int, int], ...] = (
    ("A", 0, 0, 19),
    ("B", 0, 8, 18),
    ("C", 0, 16, 16),
)
PINNED_CANDIDATE_UNIVERSE: dict[str, dict[int, int]] = {
    "A": {0: 19},
    "B": {0: 18},
    "C": {0: 16},
}
PINNED_SUB_THRESHOLD_SUPPORTS: dict[str, dict[int, int]] = {
    "A": {1: 9, 2: -9, 3: 9},
    "B": {1: 8, 2: -8, 3: 8},
    "C": {1: 7, 2: -7, 3: 7},
}
PINNED_MAGNITUDE_REGIME_BY_KEY: dict[str, str] = {
    "A": "clip_boundary_reconciliation",
    "B": "no_clip_exact_add_back",
    "C": "no_clip_exact_add_back",
}
PINNED_ACCEPTED_REGIME_BY_KEY: dict[str, str] = {
    "A": "clip_boundary_reconciliation",
    "B": "no_clip_exact_add_back",
}

MIXED_REGIME_E2E_PROOF: dict[str, bool] = {
    "gpu_mixed_regime_wired_trainer_parity_proven": False,
}

MIXED_REGIME_E2E_NON_CLAIMS: tuple[str, ...] = (
    "B2-5c/R4 is wired mixed-regime cap-apply mutation equivalence — GPU-routed global-cap APPLY within wired trainer path == CPU seam reference in one heterogeneous step",
    "B2-5c/R4 exercises accepted rows spanning >=2 tensors AND both clip_boundary_reconciliation and no_clip_exact_add_back regime classes",
    "B2-5c/R4 candidate+bridge are CPU reference computations (frozen); only seam cap_inputs mirror to CUDA",
    "B2-5c/R4 uses env-gated CUDA REFERENCE q/acc apply, NOT native q/acc Triton proof",
    "B2-5c/R4 does NOT mint selection_parity_pass",
    "B2-5c/R4 does NOT flip readiness / acquisition / training-success rows",
    "B2-5c/R4 does NOT flip optimizer_credit_state",
    "B2-5c/R4 does NOT flip global_cap_margin_only_reference",
    "B2-5c/R4 does NOT claim deferred backlog carry, whole-trainer GPU routing, native candidate GPU routing, or scale/production-size generality",
)

_FORBIDDEN_MINT_TERMS: tuple[str, ...] = (
    "selection_parity_pass=True",
    "readiness_flip_authorized",
    "optimizer_credit_state_sub2_claim",
    "global_cap_margin_only_reference_flipped",
    "training_success",
    "HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY_NATIVE",
    "CUDA-mirrored trainer inputs",
    "full-trainer-on-GPU",
    "native-candidate-GPU",
    "backlog-carry",
)


def _reset_mixed_regime_e2e_proof() -> None:
    MIXED_REGIME_E2E_PROOF["gpu_mixed_regime_wired_trainer_parity_proven"] = False


def _reset_wired_telemetry() -> None:
    for key in WIRED_E2E_TELEMETRY:
        WIRED_E2E_TELEMETRY[key] = 0
    WIRED_E2E_CAPTURED["wrapper_cap_inputs_cpu"] = None


def _build_mixed_regime_bridge_spec(state_key: str) -> BridgeFixtureSpec:
    if state_key == "A":
        return BridgeFixtureSpec(
            fixture_name="F_MIXED_REGIME_GLOBAL_A",
            fixture_role="representative_consumer",
            state_key="A",
            numel=8,
            acc_overrides={0: 17, 1: 8, 2: -8, 3: 8},
            sparse_votes={0: 2, 1: 1, 2: -1, 3: 1},
            hot_exact_indices=(0, 1, 2, 3),
            cap=2,
            max_abs_per_tensor=8,
        )
    if state_key == "B":
        return BridgeFixtureSpec(
            fixture_name="F_MIXED_REGIME_GLOBAL_B",
            fixture_role="representative_consumer",
            state_key="B",
            numel=8,
            acc_overrides={0: 16, 1: 7, 2: -7, 3: 7},
            sparse_votes={0: 2, 1: 1, 2: -1, 3: 1},
            hot_exact_indices=(0, 1, 2, 3),
            cap=2,
            max_abs_per_tensor=8,
        )
    if state_key == "C":
        return BridgeFixtureSpec(
            fixture_name="F_MIXED_REGIME_GLOBAL_C",
            fixture_role="representative_consumer",
            state_key="C",
            numel=8,
            acc_overrides={0: 14, 1: 6, 2: -6, 3: 6},
            sparse_votes={0: 2, 1: 1, 2: -1, 3: 1},
            hot_exact_indices=(0, 1, 2, 3),
            cap=2,
            max_abs_per_tensor=8,
        )
    raise ValueError(f"unsupported state_key {state_key!r}")


def _build_mixed_regime_trainer_inputs() -> tuple[
    dict[str, object],
    dict[str, VoteUpdateSpec],
    dict[str, dict[int, int]],
    GlobalRateCapSpec,
]:
    tensor_states = {}
    vote_specs = {}
    sparse_events = {}
    for state_key in MIXED_REGIME_STATE_KEYS:
        spec = _build_mixed_regime_bridge_spec(state_key)
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
        mutate_outputs=True,
    )
    return tensor_states, vote_specs, sparse_events, global_cap_spec


def _recompute_crossing_universe(
    *,
    acc_overrides: dict[int, int],
    sparse_votes: dict[int, int],
    q_overrides: dict[int, int] | None = None,
    numel: int = 8,
) -> dict[int, int]:
    state = _build_state(
        numel,
        acc_overrides=acc_overrides,
        q_overrides=q_overrides or {},
    )
    q_flat = state.q_levels.flatten()
    acc_flat = state.accumulators.flatten()
    threshold = MIXED_REGIME_THRESHOLD
    crossings: dict[int, int] = {}
    for index, vote in sparse_votes.items():
        support = int(acc_flat[int(index)].item()) + int(vote)
        q_value = int(q_flat[int(index)].item())
        if (support >= threshold and q_value < 1) or (
            support <= -threshold and q_value > -1
        ):
            crossings[int(index)] = abs(support)
    return crossings


def _assert_no_straggler_crossings(
    *,
    acc_overrides: dict[int, int],
    sparse_votes: dict[int, int],
    sub_threshold_supports: dict[int, int],
    state_key: str,
) -> None:
    state = _build_state(8, acc_overrides=acc_overrides)
    q_flat = state.q_levels.flatten()
    acc_flat = state.accumulators.flatten()
    threshold = MIXED_REGIME_THRESHOLD
    for index, expected_support in sub_threshold_supports.items():
        vote = int(sparse_votes[int(index)])
        support = int(acc_flat[int(index)].item()) + vote
        assert support == int(expected_support), (
            f"{state_key}[{index}] support mismatch: {support} != {expected_support}"
        )
        q_value = int(q_flat[int(index)].item())
        crosses = (support >= threshold and q_value < 1) or (
            support <= -threshold and q_value > -1
        )
        assert not crosses, f"{state_key}[{index}] unexpectedly crosses with support {support}"


def _q_flip_identities(
    prior_q: torch.Tensor,
    post_q: torch.Tensor,
    state_key: str,
) -> set[tuple[str, int]]:
    prior_flat = prior_q.flatten()
    post_flat = post_q.flatten()
    return {
        (state_key, int(index))
        for index in range(int(prior_flat.numel()))
        if int(prior_flat[index].item()) != int(post_flat[index].item())
    }


def _assert_mixed_regime_fixture_semantics(
    tensor_states,
    vote_specs,
    sparse_events,
    global_cap_spec: GlobalRateCapSpec,
) -> None:
    assert global_cap_spec.mutate_outputs is True
    assert global_cap_spec.normalized_ordering_mode == GlobalRateCapOrderingMode.MARGIN
    assert int(global_cap_spec.cap) == 2

    for state_key in MIXED_REGIME_STATE_KEYS:
        spec = _build_mixed_regime_bridge_spec(state_key)
        crossings = _recompute_crossing_universe(
            acc_overrides=spec.acc_overrides,
            sparse_votes=spec.sparse_votes,
            q_overrides=spec.q_overrides,
            numel=spec.numel,
        )
        assert crossings == PINNED_CANDIDATE_UNIVERSE[state_key]
        assert len(crossings) >= 1
        _assert_no_straggler_crossings(
            acc_overrides=spec.acc_overrides,
            sparse_votes=spec.sparse_votes,
            sub_threshold_supports=PINNED_SUB_THRESHOLD_SUPPORTS[state_key],
            state_key=state_key,
        )

    cap_inputs = _rebuild_independent_seam_cap_inputs(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    assert len(cap_inputs) == 3
    for state_key in MIXED_REGIME_STATE_KEYS:
        item = next(entry for entry in cap_inputs if entry.state_key == state_key)
        expected_indices = set(PINNED_CANDIDATE_UNIVERSE[state_key])
        applied = {int(idx) for idx in item.plan.applied_indices.tolist()}
        assert applied == expected_indices

    offsets = tensor_offsets_for_vote_update_states(cap_inputs)
    assert offsets == {"A": 0, "B": 8, "C": 16}

    oracle_rows, accepted_rows, deferred_rows = select_global_rate_cap_rows(
        cap_inputs,
        global_cap_spec,
        tensor_offsets=offsets,
    )
    assert len(oracle_rows) == 3
    ordered_preview = [
        (row.state_key, int(row.flat_index), int(row.global_flat_index), int(row.abs_new_acc))
        for row in oracle_rows
    ]
    assert tuple(ordered_preview) == PINNED_ORDERED_ROWS

    accepted_identities = {(row.state_key, int(row.flat_index)) for row in accepted_rows}
    deferred_identities = {(row.state_key, int(row.flat_index)) for row in deferred_rows}
    assert accepted_identities == set(PINNED_ACCEPTED)
    assert deferred_identities == {("C", 0)}
    accepted_tensors = {state_key for state_key, _ in accepted_identities}
    assert len(accepted_tensors) >= 2
    accepted_regime_classes = {
        PINNED_ACCEPTED_REGIME_BY_KEY[state_key] for state_key in accepted_tensors
    }
    assert accepted_regime_classes == {
        "clip_boundary_reconciliation",
        "no_clip_exact_add_back",
    }

    deferred_by_key: dict[str, set[int]] = {key: set() for key in MIXED_REGIME_STATE_KEYS}
    for state_key, flat_index in deferred_identities:
        deferred_by_key[state_key].add(int(flat_index))
    assert deferred_by_key == PINNED_DEFERRED_BY_KEY

    from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
        encode_budget_capped_hybrid_reference,
        execute_direct_bounded_local_vote_update_candidate,
    )

    entries: dict[str, CandidateGlobalCapSeamEntry] = {}
    for state_key in MIXED_REGIME_STATE_KEYS:
        prior_tensor = tensor_states[state_key]
        prior_vu = prior_tensor.vote_update_state()
        bounded = encode_budget_capped_hybrid_reference(
            prior_vu,
            hot_exact_indices=prior_tensor.rebuild_hot_exact_indices(),
            cold_default_value=prior_tensor.rebuild_cold_default_value(),
        )
        candidate_result = execute_direct_bounded_local_vote_update_candidate(
            state_key=state_key,
            q_levels=prior_vu.q_levels,
            bounded_accumulator=bounded,
            sparse_vote_events=sparse_events[state_key],
            vote_spec=vote_specs[state_key],
        )
        entries[state_key] = CandidateGlobalCapSeamEntry(
            prior_state=prior_vu,
            candidate_result=candidate_result,
            vote_spec=vote_specs[state_key],
        )
    seam_result = apply_candidate_global_cap_production_seam(entries, global_cap_spec)
    assert seam_result.magnitude_regime_by_key == PINNED_MAGNITUDE_REGIME_BY_KEY


def _assert_mixed_regime_mutation_non_vacuity(
    step_result,
    *,
    prior_tensors: dict[str, object],
    cap_inputs: list[GlobalRateCapTensorInput],
    label: str,
) -> set[tuple[str, int]]:
    assert step_result.global_summary.get("ternary_mutation_enabled") is True
    summary = step_result.global_summary
    assert int(summary["global_rate_cap_applied_count"]) == 2
    assert int(summary["global_rate_cap_accepted_count"]) == 2
    assert int(summary["q_changed_count"]) == 2
    assert int(summary["global_rate_cap_deferred_count"]) == 1

    for state_key in MIXED_REGIME_STATE_KEYS:
        deferred_keys = set(step_result.deferred_backlog.get(state_key, {}))
        assert deferred_keys == PINNED_DEFERRED_BY_KEY[state_key]

    accepted_identities: set[tuple[str, int]] = set()
    for state_key in MIXED_REGIME_STATE_KEYS:
        prior_tensor = prior_tensors[state_key]
        post_state = step_result.tensor_states[state_key]
        q_flips = _q_flip_identities(prior_tensor.q_levels, post_state.q_levels, state_key)
        accepted_identities.update(q_flips)

    assert accepted_identities == set(PINNED_ACCEPTED)

    cap_by_key = {item.state_key: item for item in cap_inputs}
    for state_key in ("A", "B"):
        prior_tensor = prior_tensors[state_key]
        post_state = step_result.tensor_states[state_key]
        prior_q = prior_tensor.q_levels.flatten()
        prior_acc = prior_tensor.vote_update_state().accumulators.flatten()
        post_q = post_state.q_levels.flatten()
        post_acc = post_state.exact_accumulator_shadow.flatten()

        if state_key == "A":
            assert int(prior_q[0].item()) == 0
            assert int(post_q[0].item()) == 1
            assert int(prior_acc[0].item()) == 17
            assert int(post_acc[0].item()) == 9
            assert abs(int(post_acc[0].item())) <= MIXED_REGIME_CLIP_EDGE
        elif state_key == "B":
            assert int(prior_q[0].item()) == 0
            assert int(post_q[0].item()) == 1
            assert int(prior_acc[0].item()) == 16
            assert int(post_acc[0].item()) == 8

    prior_c = prior_tensors["C"]
    post_c = step_result.tensor_states["C"]
    prior_q_c = prior_c.q_levels.flatten()
    prior_acc_c = prior_c.vote_update_state().accumulators.flatten()
    post_q_c = post_c.q_levels.flatten()
    post_acc_c = post_c.exact_accumulator_shadow.flatten()
    plan_new_acc_c = cap_by_key["C"].plan.new_acc_i32.flatten()
    assert int(prior_q_c[0].item()) == 0
    assert int(post_q_c[0].item()) == 0
    assert int(prior_acc_c[0].item()) == 14
    assert int(post_acc_c[0].item()) == 16
    assert int(plan_new_acc_c[0].item()) == 16

    assert label in {"cpu_oracle", "gpu_wired"}
    return accepted_identities


def _assert_mixed_regime_wired_parity(cpu_oracle, gpu_wired) -> None:
    for state_key in MIXED_REGIME_STATE_KEYS:
        assert tensor_sha256(gpu_wired.tensor_states[state_key].q_levels) == tensor_sha256(
            cpu_oracle.tensor_states[state_key].q_levels,
        )
        assert tensor_sha256(
            gpu_wired.tensor_states[state_key].exact_accumulator_shadow,
        ) == tensor_sha256(cpu_oracle.tensor_states[state_key].exact_accumulator_shadow)
        assert gpu_wired.tensor_stats[state_key]["q_sha256_after"] == cpu_oracle.tensor_stats[
            state_key
        ]["q_sha256_after"]
        assert gpu_wired.tensor_stats[state_key][
            "exact_accumulator_shadow_sha256_after"
        ] == cpu_oracle.tensor_stats[state_key]["exact_accumulator_shadow_sha256_after"]
    assert gpu_wired.deferred_backlog == cpu_oracle.deferred_backlog
    assert _cap_summary_subset(gpu_wired.global_summary) == _cap_summary_subset(
        cpu_oracle.global_summary,
    )
    for key in _CAP_SUMMARY_KEYS:
        if key in cpu_oracle.global_summary:
            assert gpu_wired.global_summary[key] == cpu_oracle.global_summary[key]
    assert gpu_wired.global_summary.get("selection_parity_pass") is not True
    assert gpu_wired.global_summary.get("readiness_flip_authorized") is not True


def _mixed_regime_gpu_e2e_body() -> None:
    _require_wired_e2e_gpu_lane_or_skip()
    _reset_wired_telemetry()
    _reset_mixed_regime_e2e_proof()
    tensor_states, vote_specs, sparse_events, global_cap_spec = _build_mixed_regime_trainer_inputs()
    assert global_cap_spec.mutate_outputs is True
    _assert_mixed_regime_fixture_semantics(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    prior_tensors = {key: copy.deepcopy(tensor_states[key]) for key in MIXED_REGIME_STATE_KEYS}
    cap_inputs = _rebuild_independent_seam_cap_inputs(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    cpu_oracle = apply_bounded_delta_vote_step(
        **_candidate_sparse_kwargs(
            copy.deepcopy(tensor_states),
            vote_specs,
            sparse_events,
            global_cap_spec=global_cap_spec,
            seam_enabled=True,
        ),
    )
    cpu_accepted = _assert_mixed_regime_mutation_non_vacuity(
        cpu_oracle,
        prior_tensors=prior_tensors,
        cap_inputs=cap_inputs,
        label="cpu_oracle",
    )

    _install_gpu_cap_route_patch()
    try:
        gpu_wired = apply_bounded_delta_vote_step(
            **_candidate_sparse_kwargs(
                copy.deepcopy(tensor_states),
                vote_specs,
                sparse_events,
                global_cap_spec=global_cap_spec,
                seam_enabled=True,
            ),
        )
    finally:
        _restore_cpu_cap_route_patch()

    assert WIRED_E2E_TELEMETRY["seam_cap_reference_wrapper_calls"] == 1
    assert WIRED_E2E_TELEMETRY["cuda_inputs_observed"] == 1
    assert WIRED_E2E_TELEMETRY["gpu_apply_calls"] == 1
    assert WIRED_E2E_TELEMETRY["cpu_fallback_count"] == 0

    captured = WIRED_E2E_CAPTURED["wrapper_cap_inputs_cpu"]
    assert captured is not None
    rebuilt = _rebuild_independent_seam_cap_inputs(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    _assert_cap_inputs_identity_match(captured, rebuilt)

    gpu_accepted = _assert_mixed_regime_mutation_non_vacuity(
        gpu_wired,
        prior_tensors=prior_tensors,
        cap_inputs=cap_inputs,
        label="gpu_wired",
    )
    assert cpu_accepted == gpu_accepted

    provenance = gpu_wired.global_summary.get("wired_e2e_adapter_provenance") or {}
    assert provenance.get("summary_source") == "gpu_selection_tensors"
    assert provenance.get("global_cap_gpu_native") is True
    assert provenance.get("cpu_fallback_count") == 0
    _assert_mixed_regime_wired_parity(cpu_oracle, gpu_wired)
    MIXED_REGIME_E2E_PROOF["gpu_mixed_regime_wired_trainer_parity_proven"] = True


def _mixed_regime_skip_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda_available: bool,
    triton_available: bool,
    lane_env_set: bool,
) -> None:
    import _pytest.outcomes as outcomes

    if lane_env_set:
        monkeypatch.setenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, "1")
        monkeypatch.setenv(RUN_GPU_GLOBAL_RATE_CAP_NATIVE_SELECTION_ENV, "1")
        monkeypatch.setenv(RUN_GPU_Q_ACC_APPLY_ENV, "1")
    else:
        monkeypatch.delenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, raising=False)
        monkeypatch.delenv(RUN_GPU_GLOBAL_RATE_CAP_NATIVE_SELECTION_ENV, raising=False)
        monkeypatch.delenv(RUN_GPU_Q_ACC_APPLY_ENV, raising=False)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(
        "calm.llm_computer.tests.test_hrm_text_158_candidate_global_cap_wired_trainer_mixed_regime_e2e_gpu._TRITON_AVAILABLE",
        triton_available,
    )

    sentinel_calls = {"count": 0}

    def _sentinel(*args: Any, **kwargs: Any) -> None:
        sentinel_calls["count"] += 1
        raise AssertionError(
            "seam_module.apply_global_rate_cap_reference must not run on CPU/skip path",
        )

    monkeypatch.setattr(seam_module, "apply_global_rate_cap_reference", _sentinel)
    _reset_mixed_regime_e2e_proof()
    _reset_wired_telemetry()

    with pytest.raises(outcomes.Skipped):
        _mixed_regime_gpu_e2e_body()

    assert sentinel_calls["count"] == 0
    assert WIRED_E2E_TELEMETRY["seam_cap_reference_wrapper_calls"] == 0
    assert MIXED_REGIME_E2E_PROOF["gpu_mixed_regime_wired_trainer_parity_proven"] is False


def test_mixed_regime_e2e_non_claims_frozen() -> None:
    joined = "\n".join(MIXED_REGIME_E2E_NON_CLAIMS)
    for term in _FORBIDDEN_MINT_TERMS:
        assert term not in joined
    assert "mixed-regime" in MIXED_REGIME_E2E_NON_CLAIMS[0]
    assert "both clip_boundary_reconciliation and no_clip_exact_add_back" in (
        MIXED_REGIME_E2E_NON_CLAIMS[1]
    )


def test_mixed_regime_fixture_semantics_and_mutation_non_vacuity_cpu() -> None:
    _reset_mixed_regime_e2e_proof()
    tensor_states, vote_specs, sparse_events, global_cap_spec = _build_mixed_regime_trainer_inputs()
    assert global_cap_spec.mutate_outputs is True
    _assert_mixed_regime_fixture_semantics(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    prior_tensors = {key: copy.deepcopy(tensor_states[key]) for key in MIXED_REGIME_STATE_KEYS}
    cap_inputs = _rebuild_independent_seam_cap_inputs(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    cpu_oracle = apply_bounded_delta_vote_step(
        **_candidate_sparse_kwargs(
            copy.deepcopy(tensor_states),
            vote_specs,
            sparse_events,
            global_cap_spec=global_cap_spec,
            seam_enabled=True,
        ),
    )
    _assert_mixed_regime_mutation_non_vacuity(
        cpu_oracle,
        prior_tensors=prior_tensors,
        cap_inputs=cap_inputs,
        label="cpu_oracle",
    )
    assert "global_rate_cap_applied_count" in cpu_oracle.tensor_stats["A"]
    assert MIXED_REGIME_E2E_PROOF["gpu_mixed_regime_wired_trainer_parity_proven"] is False


def test_mixed_regime_cpu_proves_gpu_body_unexecuted_cuda_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mixed_regime_skip_wrapper(
        monkeypatch,
        cuda_available=False,
        triton_available=False,
        lane_env_set=False,
    )


def test_mixed_regime_cpu_proves_gpu_body_unexecuted_lane_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mixed_regime_skip_wrapper(
        monkeypatch,
        cuda_available=True,
        triton_available=True,
        lane_env_set=False,
    )


def test_mixed_regime_cap_inputs_mirror_cuda_audit() -> None:
    _require_wired_e2e_gpu_lane_or_skip()
    tensor_states, vote_specs, sparse_events, global_cap_spec = _build_mixed_regime_trainer_inputs()
    cap_inputs_cpu = _rebuild_independent_seam_cap_inputs(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    cap_inputs_cuda = _mirror_cap_inputs_to_cuda(cap_inputs_cpu)
    assert _inputs_on_cuda(cap_inputs_cuda)
    for item in cap_inputs_cuda:
        tensors = (
            item.state.q_levels,
            item.state.accumulators,
            item.plan.q_i16,
            item.plan.new_acc_i32,
            item.plan.applied_indices,
            item.plan.applied_directions,
            item.plan.applied_thresholds,
        )
        assert len(tensors) == 7
        for tensor in tensors:
            assert tensor.device.type == "cuda"


def test_mixed_regime_f_mixed_regime_global_flag_on_parity() -> None:
    _reset_mixed_regime_e2e_proof()
    _mixed_regime_gpu_e2e_body()
    assert MIXED_REGIME_E2E_PROOF["gpu_mixed_regime_wired_trainer_parity_proven"] is True
