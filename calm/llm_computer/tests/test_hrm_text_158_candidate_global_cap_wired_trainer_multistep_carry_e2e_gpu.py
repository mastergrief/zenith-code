"""B2-5c/R6 wired MULTI-STEP q/acc CARRY GPU cap-apply mutation equivalence vs CPU seam reference.

Equivalence/compat-only on F_MULTISTEP_CARRY_GLOBAL: GPU-routed global-cap APPLY within
the flag-ON wired trainer path across 3 sequential steps (prior step mutated tensor_states
→ next input after shadow-sync rebuild) == CPU seam reference trajectory (q/acc/backlog/
cap-summary per step), per-step C1-resolved cap (non-saturating), MARGIN-only.

Uses env-gated CUDA REFERENCE q/acc apply on seam-built cap_inputs — NOT native
q/acc Triton proof. NOT full-trainer-on-GPU / native-candidate-GPU / readiness /
acquisition / selection_parity_pass / optimizer_credit_state /
global_cap_margin_only mint.

GPU execution is a separate +1 on hrm_text_158_gpu0 after the R6 diff gate.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

import pytest
import torch

import calm.hrm_text_158.native_full_stack.candidate_global_cap_production_seam as seam_module
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
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
    C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
    CAP_ORDERING_HASH_SEED,
    GlobalRateCapOrderingMode,
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    resolve_named_global_cap_spec,
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

MULTISTEP_CARRY_STATE_KEYS: tuple[str, ...] = ("A", "B")
MULTISTEP_CARRY_NUMEL = 8
MULTISTEP_CARRY_THRESHOLD = 10
MULTISTEP_CARRY_CLIP_EDGE = 9
MULTISTEP_CARRY_STEPS: tuple[int, ...] = (1, 2, 3)

PINNED_ACCEPTED: frozenset[tuple[str, int]] = frozenset({("A", 0), ("B", 0)})

PINNED_CAPS_BY_STEP: dict[int, int] = {1: 512, 2: 512, 3: 256}
PINNED_DEMAND_BY_STEP: dict[int, int] = {1: 2, 2: 2, 3: 2}

PINNED_SPARSE_VOTES_BY_STEP: dict[int, dict[str, dict[int, int]]] = {
    1: {"A": {0: 2}, "B": {0: 2}},
    2: {"A": {0: -20}, "B": {0: -18}},
    3: {"A": {0: 12}, "B": {0: 10}},
}

PINNED_CANDIDATE_UNIVERSE_BY_STEP: dict[int, dict[str, dict[int, int]]] = {
    1: {"A": {0: 19}, "B": {0: 18}},
    2: {"A": {0: 11}, "B": {0: 10}},
    3: {"A": {0: 11}, "B": {0: 10}},
}

PINNED_FRESH_STEP2_UNIVERSE: dict[str, dict[int, int]] = {"A": {}, "B": {}}

PINNED_ORDERED_ROWS_BY_STEP: dict[int, tuple[tuple[str, int, int, int], ...]] = {
    1: (("A", 0, 0, 19), ("B", 0, 8, 18)),
    2: (("A", 0, 0, 11), ("B", 0, 8, 10)),
    3: (("A", 0, 0, 11), ("B", 0, 8, 10)),
}

PINNED_POST_ROW0_BY_STEP: dict[int, dict[str, tuple[int, int]]] = {
    1: {"A": (1, 9), "B": (1, 8)},
    2: {"A": (0, -1), "B": (0, 0)},
    3: {"A": (1, 1), "B": (1, 0)},
}

PINNED_INITIAL_ACC_ROW0: dict[str, int] = {"A": 17, "B": 16}
PINNED_SUB_THRESHOLD_SUPPORTS: dict[str, dict[int, int]] = {
    "A": {1: 8, 2: -8, 3: 8},
    "B": {1: 8, 2: -8, 3: 8},
}

PINNED_MAGNITUDE_REGIME_STEP1: dict[str, str] = {
    "A": "clip_boundary_reconciliation",
    "B": "no_clip_exact_add_back",
}
PINNED_MAGNITUDE_REGIME_STEPS_2_3: dict[str, str] = {
    "A": "no_clip_exact_add_back",
    "B": "no_clip_exact_add_back",
}

MULTISTEP_CARRY_E2E_PROOF: dict[str, bool] = {
    "gpu_multistep_carry_wired_trainer_parity_proven": False,
}

MULTISTEP_CARRY_E2E_NON_CLAIMS: tuple[str, ...] = (
    "B2-5c/R6 is wired 3-step q/acc carry cap-apply mutation equivalence — GPU-CARRIED trajectory == CPU seam reference per step",
    "B2-5c/R6 exercises acc-residual carry across sequential wired steps with per-step C1 non-saturating caps",
    "B2-5c/R6 candidate+bridge are CPU reference computations (frozen); only seam cap_inputs mirror to CUDA",
    "B2-5c/R6 uses env-gated CUDA REFERENCE q/acc apply, NOT native q/acc Triton proof",
    "B2-5c/R6 does NOT mint selection_parity_pass",
    "B2-5c/R6 does NOT flip readiness / acquisition / training-success rows",
    "B2-5c/R6 does NOT flip optimizer_credit_state",
    "B2-5c/R6 does NOT flip global_cap_margin_only_reference",
    "B2-5c/R6 does NOT claim saturating C1 cap-transition-at-scale, R5 scale reuse, deferred backlog carry, non-MARGIN, native candidate GPU routing, whole-trainer GPU routing, guard-flip/default-on, or production generality beyond the pinned fixture",
    "B2-5c/R6 with_fresh_bounded_accumulator between steps is harness shadow-sync only, not a prod seam change",
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


def _reset_multistep_carry_e2e_proof() -> None:
    MULTISTEP_CARRY_E2E_PROOF["gpu_multistep_carry_wired_trainer_parity_proven"] = False


def _reset_wired_telemetry() -> None:
    for key in WIRED_E2E_TELEMETRY:
        WIRED_E2E_TELEMETRY[key] = 0
    WIRED_E2E_CAPTURED["wrapper_cap_inputs_cpu"] = None


def _build_multistep_carry_bridge_spec(state_key: str) -> BridgeFixtureSpec:
    acc0 = PINNED_INITIAL_ACC_ROW0[state_key]
    return BridgeFixtureSpec(
        fixture_name=f"F_MULTISTEP_CARRY_GLOBAL_{state_key}",
        fixture_role="representative_consumer",
        state_key=state_key,
        numel=MULTISTEP_CARRY_NUMEL,
        acc_overrides={0: acc0, 1: 8, 2: -8, 3: 8},
        sparse_votes={},
        hot_exact_indices=(0, 1, 2, 3),
        cap=PINNED_CAPS_BY_STEP[3],
        max_abs_per_tensor=8,
    )


def _build_multistep_carry_trainer_inputs() -> tuple[
    dict[str, BoundedDeltaTensorState],
    dict[str, VoteUpdateSpec],
]:
    tensor_states: dict[str, BoundedDeltaTensorState] = {}
    vote_specs: dict[str, VoteUpdateSpec] = {}
    for state_key in MULTISTEP_CARRY_STATE_KEYS:
        spec = _build_multistep_carry_bridge_spec(state_key)
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
    return tensor_states, vote_specs


def _sparse_events_for_step(step: int) -> dict[str, dict[int, int]]:
    step_votes = PINNED_SPARSE_VOTES_BY_STEP[int(step)]
    return {state_key: dict(step_votes.get(state_key, {})) for state_key in MULTISTEP_CARRY_STATE_KEYS}


def _resolve_step_global_cap_spec(step: int) -> GlobalRateCapSpec:
    resolved = resolve_named_global_cap_spec(
        C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
        step=int(step),
    )
    assert resolved is not None
    assert int(resolved.cap) == PINNED_CAPS_BY_STEP[int(step)]
    assert int(resolved.step) == int(step)
    assert resolved.normalized_ordering_mode == GlobalRateCapOrderingMode.MARGIN
    assert resolved.mutate_outputs is True
    assert int(resolved.ordering_seed) == CAP_ORDERING_HASH_SEED
    return resolved


def _assert_exact_shadow_present(tensor_states: Mapping[str, BoundedDeltaTensorState]) -> None:
    for state_key in MULTISTEP_CARRY_STATE_KEYS:
        state = tensor_states[state_key]
        assert state.exact_accumulator_shadow is not None, (
            f"{state_key} missing exact_accumulator_shadow before fresh rebuild"
        )


def _prepare_fresh_carried_inputs(
    tensor_states: Mapping[str, BoundedDeltaTensorState],
) -> dict[str, BoundedDeltaTensorState]:
    _assert_exact_shadow_present(tensor_states)
    return {
        state_key: tensor_states[state_key].with_fresh_bounded_accumulator()
        for state_key in MULTISTEP_CARRY_STATE_KEYS
    }


def _state_chain_fingerprint(tensor_states: Mapping[str, BoundedDeltaTensorState]) -> dict[str, str]:
    return {
        state_key: (
            f"{tensor_sha256(tensor_states[state_key].q_levels)}:"
            f"{tensor_sha256(tensor_states[state_key].exact_accumulator_shadow)}"
        )
        for state_key in MULTISTEP_CARRY_STATE_KEYS
    }


def _recompute_crossing_universe(
    *,
    acc_overrides: dict[int, int],
    sparse_votes: dict[int, int],
    q_overrides: dict[int, int] | None = None,
    numel: int = MULTISTEP_CARRY_NUMEL,
) -> dict[int, int]:
    state = _build_state(
        numel,
        acc_overrides=acc_overrides,
        q_overrides=q_overrides or {},
    )
    q_flat = state.q_levels.flatten()
    acc_flat = state.accumulators.flatten()
    threshold = MULTISTEP_CARRY_THRESHOLD
    crossings: dict[int, int] = {}
    for index, vote in sparse_votes.items():
        support = int(acc_flat[int(index)].item()) + int(vote)
        q_value = int(q_flat[int(index)].item())
        if (support >= threshold and q_value < 1) or (
            support <= -threshold and q_value > -1
        ):
            crossings[int(index)] = abs(support)
    return crossings


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


def _assert_no_straggler_crossings(
    *,
    acc_overrides: dict[int, int],
    sparse_votes: dict[int, int],
    q_overrides: dict[int, int] | None,
    sub_threshold_supports: dict[int, int],
    state_key: str,
) -> None:
    state = _build_state(
        MULTISTEP_CARRY_NUMEL,
        acc_overrides=acc_overrides,
        q_overrides=q_overrides or {},
    )
    q_flat = state.q_levels.flatten()
    acc_flat = state.accumulators.flatten()
    threshold = MULTISTEP_CARRY_THRESHOLD
    for index, expected_support in sub_threshold_supports.items():
        vote = int(sparse_votes.get(int(index), 0))
        support = int(acc_flat[int(index)].item()) + vote
        assert support == int(expected_support), (
            f"{state_key}[{index}] support mismatch: {support} != {expected_support}"
        )
        q_value = int(q_flat[int(index)].item())
        crosses = (support >= threshold and q_value < 1) or (
            support <= -threshold and q_value > -1
        )
        assert not crosses, f"{state_key}[{index}] unexpectedly crosses with support {support}"


def _run_bounded_delta_step(
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    vote_specs: Mapping[str, VoteUpdateSpec],
    sparse_events: Mapping[str, dict[int, int]],
    global_cap_spec: GlobalRateCapSpec,
) -> Any:
    return apply_bounded_delta_vote_step(
        **_candidate_sparse_kwargs(
            copy.deepcopy(tensor_states),
            vote_specs,
            sparse_events,
            global_cap_spec=global_cap_spec,
            seam_enabled=True,
        ),
    )


def _assert_step_cap_contract(step: int, global_cap_spec: GlobalRateCapSpec) -> None:
    assert int(global_cap_spec.cap) == PINNED_CAPS_BY_STEP[int(step)]
    demand = PINNED_DEMAND_BY_STEP[int(step)]
    assert demand < int(global_cap_spec.cap)


def _assert_step_fixture_semantics(
    *,
    step: int,
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    vote_specs: Mapping[str, VoteUpdateSpec],
    sparse_events: Mapping[str, dict[int, int]],
    global_cap_spec: GlobalRateCapSpec,
) -> list[GlobalRateCapTensorInput]:
    _assert_step_cap_contract(step, global_cap_spec)
    pinned_universe = PINNED_CANDIDATE_UNIVERSE_BY_STEP[int(step)]

    for state_key in MULTISTEP_CARRY_STATE_KEYS:
        prior = tensor_states[state_key]
        prior_vu = prior.vote_update_state()
        acc_overrides = {
            int(index): int(prior_vu.accumulators.flatten()[int(index)].item())
            for index in range(4)
        }
        q_overrides = {
            0: int(prior_vu.q_levels.flatten()[0].item()),
        }
        crossings = _recompute_crossing_universe(
            acc_overrides=acc_overrides,
            sparse_votes=sparse_events[state_key],
            q_overrides=q_overrides,
        )
        assert crossings == pinned_universe[state_key]
        _assert_no_straggler_crossings(
            acc_overrides=acc_overrides,
            sparse_votes=sparse_events[state_key],
            q_overrides=q_overrides,
            sub_threshold_supports=PINNED_SUB_THRESHOLD_SUPPORTS[state_key],
            state_key=state_key,
        )

    cap_inputs = _rebuild_independent_seam_cap_inputs(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    assert len(cap_inputs) == len(MULTISTEP_CARRY_STATE_KEYS)
    offsets = tensor_offsets_for_vote_update_states(cap_inputs)
    assert offsets == {"A": 0, "B": 8}

    oracle_rows, accepted_rows, deferred_rows = select_global_rate_cap_rows(
        cap_inputs,
        global_cap_spec,
        tensor_offsets=offsets,
    )
    ordered_preview = [
        (row.state_key, int(row.flat_index), int(row.global_flat_index), int(row.abs_new_acc))
        for row in oracle_rows
    ]
    assert tuple(ordered_preview) == PINNED_ORDERED_ROWS_BY_STEP[int(step)]
    accepted_identities = {(row.state_key, int(row.flat_index)) for row in accepted_rows}
    deferred_identities = {(row.state_key, int(row.flat_index)) for row in deferred_rows}
    assert accepted_identities == set(PINNED_ACCEPTED)
    assert deferred_identities == set()

    from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
        encode_budget_capped_hybrid_reference,
        execute_direct_bounded_local_vote_update_candidate,
    )

    entries: dict[str, CandidateGlobalCapSeamEntry] = {}
    for state_key in MULTISTEP_CARRY_STATE_KEYS:
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
    expected_regime = (
        PINNED_MAGNITUDE_REGIME_STEP1
        if int(step) == 1
        else PINNED_MAGNITUDE_REGIME_STEPS_2_3
    )
    assert seam_result.magnitude_regime_by_key == expected_regime
    return cap_inputs


def _assert_step_mutation_non_vacuity(
    step_result: Any,
    *,
    step: int,
    prior_tensors: Mapping[str, BoundedDeltaTensorState],
    cap_inputs: list[GlobalRateCapTensorInput],
    label: str,
) -> set[tuple[str, int]]:
    summary = step_result.global_summary
    demand = PINNED_DEMAND_BY_STEP[int(step)]
    assert int(summary["global_pre_cap_would_apply_count"]) == demand
    assert int(summary["global_rate_cap_applied_count"]) == demand
    assert int(summary["global_rate_cap_deferred_count"]) == 0
    assert int(summary["q_changed_count"]) == demand
    assert step_result.global_summary.get("ternary_mutation_enabled") is True

    for state_key in MULTISTEP_CARRY_STATE_KEYS:
        assert step_result.deferred_backlog.get(state_key, {}) == {}

    accepted_identities: set[tuple[str, int]] = set()
    for state_key in MULTISTEP_CARRY_STATE_KEYS:
        prior_tensor = prior_tensors[state_key]
        post_state = step_result.tensor_states[state_key]
        q_flips = _q_flip_identities(prior_tensor.q_levels, post_state.q_levels, state_key)
        accepted_identities.update(q_flips)
    assert accepted_identities == set(PINNED_ACCEPTED)

    expected_post = PINNED_POST_ROW0_BY_STEP[int(step)]
    for state_key in MULTISTEP_CARRY_STATE_KEYS:
        post_state = step_result.tensor_states[state_key]
        post_q = int(post_state.q_levels.flatten()[0].item())
        post_acc = int(post_state.exact_accumulator_shadow.flatten()[0].item())
        expected_q, expected_acc = expected_post[state_key]
        assert post_q == expected_q
        assert post_acc == expected_acc

    cap_by_key = {item.state_key: item for item in cap_inputs}
    assert len(cap_by_key) == len(MULTISTEP_CARRY_STATE_KEYS)
    assert label in {"cpu_oracle", "gpu_wired"}
    return accepted_identities


def _assert_step_wired_parity(cpu_oracle: Any, gpu_wired: Any) -> None:
    for state_key in MULTISTEP_CARRY_STATE_KEYS:
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


def _assert_counterfactual_freshness(
    *,
    step1_cpu_result: Any,
    initial_states: Mapping[str, BoundedDeltaTensorState],
) -> None:
    step2_votes = PINNED_SPARSE_VOTES_BY_STEP[2]
    carried_universe: dict[str, dict[int, int]] = {}
    for state_key in MULTISTEP_CARRY_STATE_KEYS:
        post_state = step1_cpu_result.tensor_states[state_key]
        post_vu = post_state.vote_update_state()
        acc_overrides = {
            int(index): int(post_vu.accumulators.flatten()[int(index)].item())
            for index in range(4)
        }
        q_overrides = {0: int(post_vu.q_levels.flatten()[0].item())}
        carried_universe[state_key] = _recompute_crossing_universe(
            acc_overrides=acc_overrides,
            sparse_votes=step2_votes[state_key],
            q_overrides=q_overrides,
        )

    fresh_universe: dict[str, dict[int, int]] = {}
    for state_key in MULTISTEP_CARRY_STATE_KEYS:
        spec = _build_multistep_carry_bridge_spec(state_key)
        fresh_universe[state_key] = _recompute_crossing_universe(
            acc_overrides=spec.acc_overrides,
            sparse_votes=step2_votes[state_key],
            q_overrides={0: 0},
        )

    assert fresh_universe == PINNED_FRESH_STEP2_UNIVERSE
    assert carried_universe == PINNED_CANDIDATE_UNIVERSE_BY_STEP[2]
    assert fresh_universe != carried_universe

    q_sat_only = _recompute_crossing_universe(
        acc_overrides={0: PINNED_INITIAL_ACC_ROW0["A"], 1: 8, 2: -8, 3: 8},
        sparse_votes=step2_votes["A"],
        q_overrides={0: 1},
    )
    assert q_sat_only == {}

    _assert_exact_shadow_present(initial_states)


def _run_cpu_reference_trajectory(
    initial_states: Mapping[str, BoundedDeltaTensorState],
    vote_specs: Mapping[str, VoteUpdateSpec],
) -> tuple[list[Any], list[dict[str, str]]]:
    cpu_states = copy.deepcopy(initial_states)
    cpu_traj: list[Any] = []
    input_fingerprints: list[dict[str, str]] = []
    _assert_exact_shadow_present(cpu_states)
    input_fingerprints.append(_state_chain_fingerprint(cpu_states))

    for step in MULTISTEP_CARRY_STEPS:
        if step > 1:
            cpu_states = _prepare_fresh_carried_inputs(cpu_states)
            input_fingerprints.append(_state_chain_fingerprint(cpu_states))
        sparse_events = _sparse_events_for_step(step)
        global_cap_spec = _resolve_step_global_cap_spec(step)
        prior_tensors = {key: copy.deepcopy(cpu_states[key]) for key in MULTISTEP_CARRY_STATE_KEYS}
        cap_inputs = _assert_step_fixture_semantics(
            step=step,
            tensor_states=cpu_states,
            vote_specs=vote_specs,
            sparse_events=sparse_events,
            global_cap_spec=global_cap_spec,
        )
        cpu_out = _run_bounded_delta_step(cpu_states, vote_specs, sparse_events, global_cap_spec)
        _assert_step_mutation_non_vacuity(
            cpu_out,
            step=step,
            prior_tensors=prior_tensors,
            cap_inputs=cap_inputs,
            label="cpu_oracle",
        )
        cpu_traj.append(cpu_out)
        cpu_states = cpu_out.tensor_states

    return cpu_traj, input_fingerprints


def _multistep_carry_gpu_e2e_body() -> None:
    _require_wired_e2e_gpu_lane_or_skip()
    _reset_wired_telemetry()
    _reset_multistep_carry_e2e_proof()

    initial_states, vote_specs = _build_multistep_carry_trainer_inputs()
    _assert_exact_shadow_present(initial_states)

    cpu_traj, cpu_input_fingerprints = _run_cpu_reference_trajectory(initial_states, vote_specs)
    _assert_counterfactual_freshness(
        step1_cpu_result=cpu_traj[0],
        initial_states=initial_states,
    )

    gpu_states = copy.deepcopy(initial_states)
    gpu_input_fingerprints: list[dict[str, str]] = []
    gpu_parent_out_fingerprints: list[dict[str, str] | None] = [None]
    _assert_exact_shadow_present(gpu_states)
    gpu_input_fingerprints.append(_state_chain_fingerprint(gpu_states))

    _install_gpu_cap_route_patch()
    try:
        for step in MULTISTEP_CARRY_STEPS:
            if step > 1:
                assert gpu_parent_out_fingerprints[-1] is not None
                fresh_from_gpu_parent = _state_chain_fingerprint(
                    _prepare_fresh_carried_inputs(
                        {
                            state_key: gpu_states[state_key]
                            for state_key in MULTISTEP_CARRY_STATE_KEYS
                        }
                    ),
                )
                assert fresh_from_gpu_parent == gpu_parent_out_fingerprints[-1]
                gpu_states = _prepare_fresh_carried_inputs(gpu_states)
                gpu_input_fingerprints.append(_state_chain_fingerprint(gpu_states))
                assert gpu_input_fingerprints[-1] == cpu_input_fingerprints[step - 1]

            sparse_events = _sparse_events_for_step(step)
            global_cap_spec = _resolve_step_global_cap_spec(step)
            prior_tensors = {
                key: copy.deepcopy(gpu_states[key]) for key in MULTISTEP_CARRY_STATE_KEYS
            }
            cap_inputs = _assert_step_fixture_semantics(
                step=step,
                tensor_states=gpu_states,
                vote_specs=vote_specs,
                sparse_events=sparse_events,
                global_cap_spec=global_cap_spec,
            )
            gpu_out = _run_bounded_delta_step(
                gpu_states,
                vote_specs,
                sparse_events,
                global_cap_spec,
            )

            cpu_oracle = cpu_traj[step - 1]
            _assert_step_wired_parity(cpu_oracle, gpu_out)
            _assert_step_mutation_non_vacuity(
                gpu_out,
                step=step,
                prior_tensors=prior_tensors,
                cap_inputs=cap_inputs,
                label="gpu_wired",
            )

            assert WIRED_E2E_TELEMETRY["seam_cap_reference_wrapper_calls"] == int(step)
            assert WIRED_E2E_TELEMETRY["cuda_inputs_observed"] == int(step)
            assert WIRED_E2E_TELEMETRY["gpu_apply_calls"] == int(step)
            assert WIRED_E2E_TELEMETRY["cpu_fallback_count"] == 0

            captured = WIRED_E2E_CAPTURED["wrapper_cap_inputs_cpu"]
            assert captured is not None
            rebuilt = _rebuild_independent_seam_cap_inputs(
                gpu_states,
                vote_specs,
                sparse_events,
                global_cap_spec,
            )
            _assert_cap_inputs_identity_match(captured, rebuilt)

            gpu_parent_out_fingerprints.append(_state_chain_fingerprint(gpu_out.tensor_states))
            gpu_states = gpu_out.tensor_states
    finally:
        _restore_cpu_cap_route_patch()

    provenance = gpu_out.global_summary.get("wired_e2e_adapter_provenance") or {}
    assert provenance.get("summary_source") == "gpu_selection_tensors"
    assert provenance.get("global_cap_gpu_native") is True
    assert provenance.get("cpu_fallback_count") == 0
    MULTISTEP_CARRY_E2E_PROOF["gpu_multistep_carry_wired_trainer_parity_proven"] = True


def _multistep_carry_skip_wrapper(
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
        "calm.llm_computer.tests.test_hrm_text_158_candidate_global_cap_wired_trainer_multistep_carry_e2e_gpu._TRITON_AVAILABLE",
        triton_available,
    )

    sentinel_calls = {"count": 0}

    def _sentinel(*args: Any, **kwargs: Any) -> None:
        sentinel_calls["count"] += 1
        raise AssertionError(
            "seam_module.apply_global_rate_cap_reference must not run on CPU/skip path",
        )

    monkeypatch.setattr(seam_module, "apply_global_rate_cap_reference", _sentinel)
    _reset_multistep_carry_e2e_proof()
    _reset_wired_telemetry()

    with pytest.raises(outcomes.Skipped):
        _multistep_carry_gpu_e2e_body()

    assert sentinel_calls["count"] == 0
    assert WIRED_E2E_TELEMETRY["seam_cap_reference_wrapper_calls"] == 0
    assert MULTISTEP_CARRY_E2E_PROOF["gpu_multistep_carry_wired_trainer_parity_proven"] is False


def test_multistep_carry_e2e_non_claims_frozen() -> None:
    joined = "\n".join(MULTISTEP_CARRY_E2E_NON_CLAIMS)
    for term in _FORBIDDEN_MINT_TERMS:
        assert term not in joined
    assert "3-step q/acc carry" in MULTISTEP_CARRY_E2E_NON_CLAIMS[0]
    assert "acc-residual carry" in MULTISTEP_CARRY_E2E_NON_CLAIMS[1]


def test_multistep_carry_counterfactual_freshness_cpu() -> None:
    _reset_multistep_carry_e2e_proof()
    initial_states, vote_specs = _build_multistep_carry_trainer_inputs()
    cpu_traj, _ = _run_cpu_reference_trajectory(initial_states, vote_specs)
    _assert_counterfactual_freshness(step1_cpu_result=cpu_traj[0], initial_states=initial_states)
    assert MULTISTEP_CARRY_E2E_PROOF["gpu_multistep_carry_wired_trainer_parity_proven"] is False


def test_multistep_carry_fixture_semantics_and_cpu_trajectory() -> None:
    _reset_multistep_carry_e2e_proof()
    initial_states, vote_specs = _build_multistep_carry_trainer_inputs()
    cpu_traj, _ = _run_cpu_reference_trajectory(initial_states, vote_specs)
    assert len(cpu_traj) == 3
    terminal = cpu_traj[-1]
    assert int(terminal.tensor_states["A"].q_levels.flatten()[0].item()) == 1
    assert int(terminal.tensor_states["A"].exact_accumulator_shadow.flatten()[0].item()) == 1
    assert int(terminal.tensor_states["B"].q_levels.flatten()[0].item()) == 1
    assert int(terminal.tensor_states["B"].exact_accumulator_shadow.flatten()[0].item()) == 0
    assert "global_rate_cap_applied_count" in terminal.tensor_stats["A"]
    assert MULTISTEP_CARRY_E2E_PROOF["gpu_multistep_carry_wired_trainer_parity_proven"] is False


def test_multistep_carry_cpu_proves_gpu_body_unexecuted_cuda_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _multistep_carry_skip_wrapper(
        monkeypatch,
        cuda_available=False,
        triton_available=False,
        lane_env_set=False,
    )


def test_multistep_carry_cpu_proves_gpu_body_unexecuted_lane_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _multistep_carry_skip_wrapper(
        monkeypatch,
        cuda_available=True,
        triton_available=True,
        lane_env_set=False,
    )


def test_multistep_carry_cap_inputs_mirror_cuda_audit() -> None:
    _require_wired_e2e_gpu_lane_or_skip()
    initial_states, vote_specs = _build_multistep_carry_trainer_inputs()
    sparse_events = _sparse_events_for_step(1)
    global_cap_spec = _resolve_step_global_cap_spec(1)
    cap_inputs_cpu = _rebuild_independent_seam_cap_inputs(
        initial_states,
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


def test_multistep_carry_f_multistep_carry_global_flag_on_parity() -> None:
    _reset_multistep_carry_e2e_proof()
    _multistep_carry_gpu_e2e_body()
    assert MULTISTEP_CARRY_E2E_PROOF["gpu_multistep_carry_wired_trainer_parity_proven"] is True
