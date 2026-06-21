"""B2-5c/R1 clip-boundary regime GPU cap-apply parity vs CPU seam reference.

Clip-boundary regime equivalence on F_CLIP_BOUNDARY_WIRED: GPU-routed global-cap
APPLY within the flag-ON wired trainer path == CPU seam reference (q/acc/backlog/
cap-summary + clamp counters) at accumulator clip-boundary, with non-vacuous
mutate_outputs=True accepted-row mutation.

Per-tensor stats schema divergence (documented, NOT a compat claim): CPU reference
per-tensor stats emit ``global_rate_cap_applied_count`` and plain
``global_rate_cap_accepted_indices`` (global_rate_cap.py:765-784). GPU per-tensor
stats omit those keys and instead emit ``q_changed_count``,
``global_rate_cap_accepted_count``, and global_indices_sha fields
(global_rate_cap_gpu.py:848-887). R1 non-vacuity uses path-neutral
``BoundedDeltaLearnerStepResult`` surfaces only (flattened q-flip set,
``deferred_backlog`` exact keys, ``global_summary`` counts, tensor state values).

Fixture index 5 is the Stage A accumulator-clip witness (126+6→127 via q[5]=1
blocking threshold crossing). The only global-cap deferred row is index 2.

Uses env-gated CUDA REFERENCE q/acc apply on seam-built cap_inputs — NOT native
q/acc Triton proof. NOT full-trainer-on-GPU / native-candidate-GPU / readiness /
acquisition / selection_parity_pass / optimizer_credit_state /
global_cap_margin_only mint.

GPU execution is a separate +1 on hrm_text_158_gpu0 after the R1 diff gate.
"""
from __future__ import annotations

import copy
import hashlib
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

CLIP_BOUNDARY_STATE_KEY = "CLIP"
PINNED_ACCEPTED = (CLIP_BOUNDARY_STATE_KEY, 0)
PINNED_DEFERRED = (CLIP_BOUNDARY_STATE_KEY, 2)
CLIP_BOUNDARY_THRESHOLD = 10
CLIP_BOUNDARY_ACCUMULATOR_CLIP_MIN = -127
CLIP_BOUNDARY_ACCUMULATOR_CLIP_MAX = 127

CLIP_BOUNDARY_E2E_PROOF: dict[str, bool] = {
    "gpu_clip_boundary_wired_trainer_parity_proven": False,
}

CLIP_BOUNDARY_E2E_NON_CLAIMS: tuple[str, ...] = (
    "B2-5c/R1 is clip-boundary regime equivalence — GPU-routed global-cap APPLY within wired trainer path == CPU seam reference",
    "B2-5c/R1 exercises non-vacuous mutate_outputs=True accepted-row mutation (not 3c passthrough)",
    "B2-5c/R1 candidate+bridge are CPU reference computations (frozen); only seam cap_inputs mirror to CUDA",
    "B2-5c/R1 uses env-gated CUDA REFERENCE q/acc apply, NOT native q/acc Triton proof",
    "B2-5c/R1 does NOT mint selection_parity_pass",
    "B2-5c/R1 does NOT flip readiness / acquisition / training-success rows",
    "B2-5c/R1 does NOT flip optimizer_credit_state",
    "B2-5c/R1 does NOT flip global_cap_margin_only_reference",
    "B2-5c/R1 does NOT claim production guard changes, default on, whole-trainer GPU routing, or native candidate GPU routing",
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
    "guard-flip",
    "default-on",
)


def _reset_clip_boundary_e2e_proof() -> None:
    CLIP_BOUNDARY_E2E_PROOF["gpu_clip_boundary_wired_trainer_parity_proven"] = False


def _reset_wired_telemetry() -> None:
    for key in WIRED_E2E_TELEMETRY:
        WIRED_E2E_TELEMETRY[key] = 0
    WIRED_E2E_CAPTURED["wrapper_cap_inputs_cpu"] = None


def _clip_i16(value: int, clip_min: int, clip_max: int) -> int:
    return int(max(int(clip_min), min(int(clip_max), int(value))))


def _identity_sha256(identities: set[tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for state_key, flat_index in sorted(identities):
        digest.update(state_key.encode("utf-8"))
        digest.update(b":")
        digest.update(str(int(flat_index)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _build_clip_boundary_trainer_inputs() -> tuple[
    dict[str, object],
    dict[str, VoteUpdateSpec],
    dict[str, dict[int, int]],
    GlobalRateCapSpec,
]:
    acc_overrides = {0: 18, 2: -18, 5: 126}
    q_overrides = {5: 1}
    sparse_votes = {0: 2, 2: -2, 5: 6}
    hot_exact_indices = (0, 2)
    vote_state = _build_state(
        8,
        acc_overrides=acc_overrides,
        q_overrides=q_overrides,
    )
    vote_spec = _vote_spec(max_abs_per_tensor=8)
    tensor_state = make_bounded_tensor_state(
        CLIP_BOUNDARY_STATE_KEY,
        vote_state.q_levels,
        0.5,
        vote_state.accumulators,
        hot_exact_indices=hot_exact_indices,
    )
    global_cap_spec = GlobalRateCapSpec(
        cap=1,
        step=1,
        ordering_seed=0,
        mutate_outputs=True,
    )
    return (
        {CLIP_BOUNDARY_STATE_KEY: tensor_state},
        {CLIP_BOUNDARY_STATE_KEY: vote_spec},
        {CLIP_BOUNDARY_STATE_KEY: dict(sparse_votes)},
        global_cap_spec,
    )


def _candidate_stage_residual_clamp(
    *,
    support_after_vote: int,
    direction: int,
    threshold: int,
) -> tuple[int, int, bool]:
    pre_residual = int(support_after_vote) - (int(direction) * int(threshold))
    post_residual = _clip_i16(
        pre_residual,
        -int(threshold) + 1,
        int(threshold) - 1,
    )
    return pre_residual, post_residual, pre_residual != post_residual


def _compute_clip_boundary_telemetry(
    seam_result,
    cap_inputs: list[GlobalRateCapTensorInput],
) -> dict[str, Any]:
    artifacts = seam_result.artifacts_by_key[CLIP_BOUNDARY_STATE_KEY]
    vote_spec = _vote_spec(max_abs_per_tensor=8)
    clip_min = int(vote_spec.accumulator_clip_min)
    clip_max = int(vote_spec.accumulator_clip_max)
    threshold = int(artifacts.threshold)

    accumulator_clip_hit_rows: list[int] = []
    acc_before = _build_state(
        8,
        acc_overrides={0: 18, 2: -18, 5: 126},
        q_overrides={5: 1},
    ).accumulators
    sparse_votes = {0: 2, 2: -2, 5: 6}
    for index, vote in sparse_votes.items():
        prior = int(acc_before[int(index)].item())
        decayed = prior
        pre_support = int(decayed) + int(vote)
        post_support = _clip_i16(pre_support, clip_min, clip_max)
        if pre_support != post_support:
            accumulator_clip_hit_rows.append(int(index))

    residual_clip_hit_rows: dict[int, dict[str, int | bool]] = {}
    for index in artifacts.applied_indices:
        idx = int(index)
        direction = int(artifacts.applied_directions[idx])
        vote = int(sparse_votes.get(idx, 0))
        prior_acc = int(acc_before[int(idx)].item())
        support_after_vote = _clip_i16(
            prior_acc + vote,
            clip_min,
            clip_max,
        )
        pre_residual, post_residual, hit = _candidate_stage_residual_clamp(
            support_after_vote=support_after_vote,
            direction=direction,
            threshold=threshold,
        )
        residual_clip_hit_rows[idx] = {
            "pre_residual": pre_residual,
            "post_residual": post_residual,
            "clamp_hit": hit,
        }

    cap_item = next(item for item in cap_inputs if item.state_key == CLIP_BOUNDARY_STATE_KEY)
    restored = cap_item.plan.new_acc_i32.flatten()
    apply_reclamp_rows: dict[int, dict[str, int | bool]] = {}
    for index in artifacts.applied_indices:
        idx = int(index)
        direction = int(artifacts.applied_directions[idx])
        pre_apply = int(restored[idx].item()) - (direction * threshold)
        post_apply = _clip_i16(pre_apply, -threshold + 1, threshold - 1)
        apply_reclamp_rows[idx] = {
            "pre_apply_residual": pre_apply,
            "post_apply_residual": post_apply,
            "clamp_hit": pre_apply != post_apply,
        }

    proof = artifacts.proof
    return {
        "accumulator_clip_hit_rows": sorted(accumulator_clip_hit_rows),
        "residual_clip_hit_rows": residual_clip_hit_rows,
        "residual_after_threshold_sha256": str(proof["residual_after_threshold_sha256"]),
        "magnitude_regime": seam_result.magnitude_regime_by_key[CLIP_BOUNDARY_STATE_KEY],
        "apply_residual_reclamp_idempotent": all(
            not bool(row["clamp_hit"]) for row in apply_reclamp_rows.values()
        ),
        "apply_residual_reclamp_rows": apply_reclamp_rows,
        "pinned_accepted_candidate_clamp": residual_clip_hit_rows.get(PINNED_ACCEPTED[1]),
        "pinned_deferred_candidate_clamp": residual_clip_hit_rows.get(PINNED_DEFERRED[1]),
    }


def _mutation_hit_identities(
    *,
    prior_q: torch.Tensor,
    prior_acc: torch.Tensor,
    post_q: torch.Tensor,
    post_acc: torch.Tensor,
    state_key: str,
) -> set[tuple[str, int]]:
    hits: set[tuple[str, int]] = set()
    for index in range(int(prior_q.numel())):
        if int(prior_q[index].item()) != int(post_q[index].item()) or int(
            prior_acc[index].item(),
        ) != int(post_acc[index].item()):
            hits.add((state_key, int(index)))
    return hits


def _accepted_clamp_mutation_identity_sha256(
    accepted_identities: set[tuple[str, int]],
    mutation_hit_identities: set[tuple[str, int]],
) -> str:
    return _identity_sha256(accepted_identities & mutation_hit_identities)


def _assert_clip_boundary_fixture_semantics(
    tensor_states,
    vote_specs,
    sparse_events,
    global_cap_spec: GlobalRateCapSpec,
) -> dict[str, Any]:
    assert global_cap_spec.mutate_outputs is True
    assert global_cap_spec.normalized_ordering_mode == GlobalRateCapOrderingMode.MARGIN

    cap_inputs = _rebuild_independent_seam_cap_inputs(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    assert len(cap_inputs) == 1
    cap_item = cap_inputs[0]
    assert int(cap_item.plan.applied_indices.numel()) == 2

    offsets = tensor_offsets_for_vote_update_states(cap_inputs)
    oracle_rows, accepted_rows, deferred_rows = select_global_rate_cap_rows(
        cap_inputs,
        global_cap_spec,
        tensor_offsets=offsets,
    )
    assert len(oracle_rows) == 2
    accepted_identities = {(row.state_key, int(row.flat_index)) for row in accepted_rows}
    deferred_identities = {(row.state_key, int(row.flat_index)) for row in deferred_rows}
    assert accepted_identities == {PINNED_ACCEPTED}
    assert deferred_identities == {PINNED_DEFERRED}

    from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
        encode_budget_capped_hybrid_reference,
        execute_direct_bounded_local_vote_update_candidate,
    )

    prior_tensor = tensor_states[CLIP_BOUNDARY_STATE_KEY]
    prior_vu = prior_tensor.vote_update_state()
    bounded = encode_budget_capped_hybrid_reference(
        prior_vu,
        hot_exact_indices=prior_tensor.rebuild_hot_exact_indices(),
        cold_default_value=prior_tensor.rebuild_cold_default_value(),
    )
    candidate_result = execute_direct_bounded_local_vote_update_candidate(
        state_key=CLIP_BOUNDARY_STATE_KEY,
        q_levels=prior_vu.q_levels,
        bounded_accumulator=bounded,
        sparse_vote_events=sparse_events[CLIP_BOUNDARY_STATE_KEY],
        vote_spec=vote_specs[CLIP_BOUNDARY_STATE_KEY],
    )
    seam_result = apply_candidate_global_cap_production_seam(
        {
            CLIP_BOUNDARY_STATE_KEY: CandidateGlobalCapSeamEntry(
                prior_state=prior_vu,
                candidate_result=candidate_result,
                vote_spec=vote_specs[CLIP_BOUNDARY_STATE_KEY],
            ),
        },
        global_cap_spec,
    )
    telemetry = _compute_clip_boundary_telemetry(seam_result, cap_inputs)
    assert telemetry["magnitude_regime"] == "clip_boundary_reconciliation"
    # Index 5: Stage A accumulator-clip witness (126+6→127); not a global-cap row.
    assert 5 in telemetry["accumulator_clip_hit_rows"]
    assert telemetry["pinned_accepted_candidate_clamp"] == {
        "pre_residual": 10,
        "post_residual": 9,
        "clamp_hit": True,
    }
    assert telemetry["pinned_deferred_candidate_clamp"] == {
        "pre_residual": -10,
        "post_residual": -9,
        "clamp_hit": True,
    }
    assert telemetry["apply_residual_reclamp_idempotent"] is True
    return telemetry


def _q_flip_indices(prior_q: torch.Tensor, post_q: torch.Tensor) -> set[int]:
    prior_flat = prior_q.flatten()
    post_flat = post_q.flatten()
    return {
        int(index)
        for index in range(int(prior_flat.numel()))
        if int(prior_flat[index].item()) != int(post_flat[index].item())
    }


def _assert_clip_boundary_mutation_non_vacuity(
    step_result,
    *,
    prior_tensor,
    pinned_accepted: tuple[str, int],
    pinned_deferred: tuple[str, int],
    cap_inputs: list[GlobalRateCapTensorInput],
    label: str,
) -> str:
    assert step_result.global_summary.get("ternary_mutation_enabled") is True
    summary = step_result.global_summary
    assert int(summary["global_rate_cap_applied_count"]) == 1
    assert int(summary["global_rate_cap_accepted_count"]) == 1
    assert int(summary["q_changed_count"]) == 1
    assert int(summary["global_rate_cap_deferred_count"]) == 1

    deferred_backlog_keys = set(
        step_result.deferred_backlog.get(CLIP_BOUNDARY_STATE_KEY, {}),
    )
    assert deferred_backlog_keys == {pinned_deferred[1]}

    post_state = step_result.tensor_states[CLIP_BOUNDARY_STATE_KEY]
    prior_q = prior_tensor.q_levels
    prior_acc = prior_tensor.vote_update_state().accumulators
    post_q = post_state.q_levels
    post_acc = post_state.exact_accumulator_shadow
    assert post_acc is not None

    q_flip_indices = _q_flip_indices(prior_q, post_q)
    assert q_flip_indices == {pinned_accepted[1]}
    accepted_identities = {
        (CLIP_BOUNDARY_STATE_KEY, int(index)) for index in q_flip_indices
    }

    prior_q_flat = prior_q.flatten()
    prior_acc_flat = prior_acc.flatten()
    post_q_flat = post_q.flatten()
    post_acc_flat = post_acc.flatten()
    mutation_hits = _mutation_hit_identities(
        prior_q=prior_q_flat,
        prior_acc=prior_acc_flat,
        post_q=post_q_flat,
        post_acc=post_acc_flat,
        state_key=CLIP_BOUNDARY_STATE_KEY,
    )
    assert pinned_accepted in mutation_hits
    assert accepted_identities & mutation_hits

    accepted_idx = pinned_accepted[1]
    deferred_idx = pinned_deferred[1]
    assert int(post_q_flat[accepted_idx].item()) == 1
    assert int(prior_q_flat[accepted_idx].item()) == 0
    assert int(post_acc_flat[accepted_idx].item()) == 9
    assert int(prior_acc_flat[accepted_idx].item()) == 18

    cap_item = next(item for item in cap_inputs if item.state_key == CLIP_BOUNDARY_STATE_KEY)
    plan_new_acc = cap_item.plan.new_acc_i32.flatten()
    assert int(post_q_flat[deferred_idx].item()) == int(prior_q_flat[deferred_idx].item())
    assert int(post_acc_flat[deferred_idx].item()) == int(plan_new_acc[deferred_idx].item())
    assert int(post_q_flat[deferred_idx].item()) == 0
    assert int(post_acc_flat[deferred_idx].item()) == -19

    identity_sha = _accepted_clamp_mutation_identity_sha256(
        accepted_identities,
        mutation_hits,
    )
    assert identity_sha
    assert label in {"cpu_oracle", "gpu_wired"}
    return identity_sha


def _assert_clip_boundary_regime_parity(
    cpu_telemetry: dict[str, Any],
    gpu_telemetry: dict[str, Any],
) -> None:
    keys = (
        "accumulator_clip_hit_rows",
        "residual_clip_hit_rows",
        "residual_after_threshold_sha256",
        "magnitude_regime",
        "apply_residual_reclamp_idempotent",
        "pinned_accepted_candidate_clamp",
        "pinned_deferred_candidate_clamp",
    )
    for key in keys:
        assert cpu_telemetry[key] == gpu_telemetry[key]


def _assert_clip_boundary_wired_parity(cpu_oracle, gpu_wired) -> None:
    state_key = CLIP_BOUNDARY_STATE_KEY
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


def _clip_boundary_gpu_e2e_body() -> None:
    _require_wired_e2e_gpu_lane_or_skip()
    _reset_wired_telemetry()
    _reset_clip_boundary_e2e_proof()
    tensor_states, vote_specs, sparse_events, global_cap_spec = _build_clip_boundary_trainer_inputs()
    assert global_cap_spec.mutate_outputs is True
    cpu_telemetry = _assert_clip_boundary_fixture_semantics(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    prior_tensor = copy.deepcopy(tensor_states[CLIP_BOUNDARY_STATE_KEY])
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
    cpu_identity_sha = _assert_clip_boundary_mutation_non_vacuity(
        cpu_oracle,
        prior_tensor=prior_tensor,
        pinned_accepted=PINNED_ACCEPTED,
        pinned_deferred=PINNED_DEFERRED,
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

    gpu_identity_sha = _assert_clip_boundary_mutation_non_vacuity(
        gpu_wired,
        prior_tensor=prior_tensor,
        pinned_accepted=PINNED_ACCEPTED,
        pinned_deferred=PINNED_DEFERRED,
        cap_inputs=cap_inputs,
        label="gpu_wired",
    )
    assert cpu_identity_sha == gpu_identity_sha

    cpu_proof = cpu_oracle.global_summary["candidate_local_update_proof_by_key"][
        CLIP_BOUNDARY_STATE_KEY
    ]
    gpu_proof = gpu_wired.global_summary["candidate_local_update_proof_by_key"][
        CLIP_BOUNDARY_STATE_KEY
    ]
    assert (
        cpu_proof["residual_after_threshold_sha256"]
        == gpu_proof["residual_after_threshold_sha256"]
    )
    gpu_telemetry = dict(cpu_telemetry)
    gpu_telemetry["residual_after_threshold_sha256"] = str(
        gpu_proof["residual_after_threshold_sha256"],
    )
    _assert_clip_boundary_regime_parity(cpu_telemetry, gpu_telemetry)

    provenance = gpu_wired.global_summary.get("wired_e2e_adapter_provenance") or {}
    assert provenance.get("summary_source") == "gpu_selection_tensors"
    assert provenance.get("global_cap_gpu_native") is True
    assert provenance.get("cpu_fallback_count") == 0
    _assert_clip_boundary_wired_parity(cpu_oracle, gpu_wired)
    CLIP_BOUNDARY_E2E_PROOF["gpu_clip_boundary_wired_trainer_parity_proven"] = True


def _clip_boundary_skip_wrapper(
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
        "calm.llm_computer.tests.test_hrm_text_158_candidate_global_cap_wired_trainer_clip_boundary_e2e_gpu._TRITON_AVAILABLE",
        triton_available,
    )

    sentinel_calls = {"count": 0}

    def _sentinel(*args, **kwargs):
        sentinel_calls["count"] += 1
        raise AssertionError(
            "seam_module.apply_global_rate_cap_reference must not run on CPU/skip path",
        )

    monkeypatch.setattr(seam_module, "apply_global_rate_cap_reference", _sentinel)
    _reset_clip_boundary_e2e_proof()
    _reset_wired_telemetry()

    with pytest.raises(outcomes.Skipped):
        _clip_boundary_gpu_e2e_body()

    assert sentinel_calls["count"] == 0
    assert WIRED_E2E_TELEMETRY["seam_cap_reference_wrapper_calls"] == 0
    assert CLIP_BOUNDARY_E2E_PROOF["gpu_clip_boundary_wired_trainer_parity_proven"] is False


def test_clip_boundary_e2e_non_claims_frozen() -> None:
    joined = "\n".join(CLIP_BOUNDARY_E2E_NON_CLAIMS)
    for term in _FORBIDDEN_MINT_TERMS:
        assert term not in joined
    assert "clip-boundary regime equivalence" in CLIP_BOUNDARY_E2E_NON_CLAIMS[0]
    assert "non-vacuous mutate_outputs=True" in CLIP_BOUNDARY_E2E_NON_CLAIMS[1]


def test_clip_boundary_fixture_semantics_and_mutation_non_vacuity_cpu() -> None:
    _reset_clip_boundary_e2e_proof()
    tensor_states, vote_specs, sparse_events, global_cap_spec = _build_clip_boundary_trainer_inputs()
    assert global_cap_spec.mutate_outputs is True
    telemetry = _assert_clip_boundary_fixture_semantics(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    assert telemetry["apply_residual_reclamp_idempotent"] is True
    prior_tensor = copy.deepcopy(tensor_states[CLIP_BOUNDARY_STATE_KEY])
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
    _assert_clip_boundary_mutation_non_vacuity(
        cpu_oracle,
        prior_tensor=prior_tensor,
        pinned_accepted=PINNED_ACCEPTED,
        pinned_deferred=PINNED_DEFERRED,
        cap_inputs=cap_inputs,
        label="cpu_oracle",
    )
    assert "global_rate_cap_applied_count" in cpu_oracle.tensor_stats[CLIP_BOUNDARY_STATE_KEY]
    assert CLIP_BOUNDARY_E2E_PROOF["gpu_clip_boundary_wired_trainer_parity_proven"] is False


def test_clip_boundary_cpu_proves_gpu_body_unexecuted_cuda_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clip_boundary_skip_wrapper(
        monkeypatch,
        cuda_available=False,
        triton_available=False,
        lane_env_set=False,
    )


def test_clip_boundary_cpu_proves_gpu_body_unexecuted_lane_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clip_boundary_skip_wrapper(
        monkeypatch,
        cuda_available=True,
        triton_available=True,
        lane_env_set=False,
    )


def test_clip_boundary_cap_inputs_mirror_cuda_audit() -> None:
    _require_wired_e2e_gpu_lane_or_skip()
    tensor_states, vote_specs, sparse_events, global_cap_spec = _build_clip_boundary_trainer_inputs()
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


def test_clip_boundary_wired_flag_on_parity() -> None:
    _reset_clip_boundary_e2e_proof()
    _clip_boundary_gpu_e2e_body()
    assert CLIP_BOUNDARY_E2E_PROOF["gpu_clip_boundary_wired_trainer_parity_proven"] is True
