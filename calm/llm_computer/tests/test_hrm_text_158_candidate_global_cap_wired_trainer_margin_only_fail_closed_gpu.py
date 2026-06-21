"""B2-5c/R7 wired-path MARGIN-only fail-closed on the GPU-routed trainer path.

Narrow claim: the GPU-routed global-cap APPLY within the flag-ON wired trainer path
raises before global-cap q/acc mutation when global_cap_spec.ordering_mode is
non-MARGIN; paired MARGIN control succeeds on the same fixture.

Path trace (GPU patch + lane env):
  apply_bounded_delta_vote_step(seam_enabled=True)
    -> apply_candidate_global_cap_production_seam (seam:192)
    -> patched apply_global_rate_cap_reference (_gpu_routing_cap_reference)
    -> apply_global_rate_cap_torch_cuda_reference_under_margin
    -> _validate_margin_only_spec (grc_gpu:719) FIRST — before _device_row_tensors
       / apply_cap_row_mutation_with_device_rows.

Prior art (low-level SELECT only, empty inputs — NOT duplicated here):
  test_hrm_text_158_native_global_rate_cap_gpu.py:163

CPU no-patch path honors non-MARGIN (out-of-scope residual); R7 MUST use GPU
patch + lane env or the test would false-pass.

NOT: readiness / global_cap_margin_only / selection_parity_pass /
optimizer_credit_state / native-qacc-Triton / full-trainer / native-candidate-GPU /
guard-flip / default-on / R5 scale / backlog-carry / R8 fallback-activation.
"""
from __future__ import annotations

import copy

import pytest
import torch

import calm.hrm_text_158.native_full_stack.candidate_global_cap_production_seam as seam_module
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    apply_bounded_delta_vote_step,
    tensor_sha256,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapOrderingMode,
    GlobalRateCapSpec,
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
from calm.hrm_text_158.native_full_stack.vote_update import RUN_GPU_Q_ACC_APPLY_ENV
from calm.llm_computer.tests.test_hrm_text_158_candidate_global_cap_trainer_wiring import (
    _build_two_tensor_trainer_inputs,
    _candidate_sparse_kwargs,
)
from calm.llm_computer.tests.test_hrm_text_158_candidate_global_cap_wired_trainer_e2e_gpu import (
    WIRED_E2E_TELEMETRY,
    _install_gpu_cap_route_patch,
    _rebuild_independent_seam_cap_inputs,
    _require_wired_e2e_gpu_lane_or_skip,
    _restore_cpu_cap_route_patch,
    _reset_wired_e2e_telemetry,
)

# F_TWO_TENSOR_GLOBAL MARGIN cap=2 mutate_outputs=True — grounded via
# select_global_rate_cap_rows on seam cap_inputs (B wins abs_new_acc=18 rows).
PINNED_MARGIN_CONTROL_ACCEPTED: frozenset[tuple[str, int]] = frozenset({("B", 0), ("B", 1)})

MARGIN_ONLY_FAIL_CLOSED_PROOF: dict[str, bool] = {
    "wired_path_margin_only_fail_closed_proven": False,
}

MARGIN_ONLY_NON_CLAIMS: tuple[str, ...] = (
    "B2-5c/R7 proves wired-path MARGIN-only fail-closed on GPU-routed global-cap APPLY",
    "B2-5c/R7 paired MARGIN control succeeds on F_TWO_TENSOR_GLOBAL under GPU patch + lane env",
    "B2-5c/R7 uses env-gated CUDA REFERENCE q/acc apply on seam-built cap_inputs",
    "B2-5c/R7 does NOT mint selection_parity_pass",
    "B2-5c/R7 does NOT flip readiness / acquisition / training-success rows",
    "B2-5c/R7 does NOT flip optimizer_credit_state",
    "B2-5c/R7 does NOT flip global_cap_margin_only_reference",
    "B2-5c/R7 does NOT prove guard-flip / default-on / R8 fallback routing",
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
)


def _reset_margin_only_proof() -> None:
    MARGIN_ONLY_FAIL_CLOSED_PROOF["wired_path_margin_only_fail_closed_proven"] = False


def _non_margin_spec(ordering_mode: GlobalRateCapOrderingMode) -> GlobalRateCapSpec:
    return GlobalRateCapSpec(
        cap=2,
        step=1,
        ordering_mode=ordering_mode,
        mutate_outputs=True,
    )


def _margin_control_spec() -> GlobalRateCapSpec:
    return GlobalRateCapSpec(
        cap=2,
        step=1,
        ordering_mode=GlobalRateCapOrderingMode.MARGIN,
        mutate_outputs=True,
    )


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


def _margin_control_oracle_accepted_identities(
    tensor_states,
    vote_specs,
    sparse_events,
    global_cap_spec: GlobalRateCapSpec,
) -> set[tuple[str, int]]:
    cap_inputs = _rebuild_independent_seam_cap_inputs(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    offsets = tensor_offsets_for_vote_update_states(cap_inputs)
    _, accepted_rows, _ = select_global_rate_cap_rows(
        cap_inputs,
        global_cap_spec,
        tensor_offsets=offsets,
    )
    return {(row.state_key, int(row.flat_index)) for row in accepted_rows}


def _input_tensor_shas(tensor_states: dict) -> dict[str, tuple[str, str]]:
    return {
        state_key: (
            tensor_sha256(tensor_states[state_key].q_levels),
            tensor_sha256(tensor_states[state_key].exact_accumulator_shadow),
        )
        for state_key in sorted(tensor_states)
    }


def _assert_input_tensor_shas_unchanged(
    tensor_states: dict,
    before: dict[str, tuple[str, str]],
) -> None:
    """Secondary sanity only — apply_bounded_delta_vote_step deep-copies input."""
    after = _input_tensor_shas(tensor_states)
    assert after == before


def _run_wired_refused_path(ordering_mode: GlobalRateCapOrderingMode) -> None:
    _require_wired_e2e_gpu_lane_or_skip()
    _reset_margin_only_proof()
    _reset_wired_e2e_telemetry()

    tensor_states, vote_specs, sparse_events, _ = _build_two_tensor_trainer_inputs()
    global_cap_spec = _non_margin_spec(ordering_mode)
    input_shas = _input_tensor_shas(tensor_states)

    _install_gpu_cap_route_patch()
    try:
        with pytest.raises(NotImplementedError, match="MARGIN ordering only"):
            apply_bounded_delta_vote_step(
                **_candidate_sparse_kwargs(
                    tensor_states,
                    vote_specs,
                    sparse_events,
                    global_cap_spec=global_cap_spec,
                    seam_enabled=True,
                ),
            )
    finally:
        _restore_cpu_cap_route_patch()

    assert MARGIN_ONLY_FAIL_CLOSED_PROOF["wired_path_margin_only_fail_closed_proven"] is False
    assert WIRED_E2E_TELEMETRY["cpu_fallback_count"] == 0
    _assert_input_tensor_shas_unchanged(tensor_states, input_shas)
    # Wrapper may increment telemetry before validate raises (R6 lesson) — not asserted.


def _run_wired_margin_control_body() -> None:
    _require_wired_e2e_gpu_lane_or_skip()
    _reset_margin_only_proof()
    _reset_wired_e2e_telemetry()

    tensor_states, vote_specs, sparse_events, _ = _build_two_tensor_trainer_inputs()
    global_cap_spec = _margin_control_spec()
    prior_tensors = {key: copy.deepcopy(tensor_states[key]) for key in ("A", "B")}

    _install_gpu_cap_route_patch()
    try:
        gpu_wired = apply_bounded_delta_vote_step(
            **_candidate_sparse_kwargs(
                tensor_states,
                vote_specs,
                sparse_events,
                global_cap_spec=global_cap_spec,
                seam_enabled=True,
            ),
        )
    finally:
        _restore_cpu_cap_route_patch()

    assert WIRED_E2E_TELEMETRY["cpu_fallback_count"] == 0
    assert gpu_wired.global_summary.get("global_rate_cap_enabled") is True
    assert int(gpu_wired.global_summary.get("global_rate_cap_applied_count", 0)) >= 1
    q_changed_count = int(gpu_wired.global_summary.get("q_changed_count", 0))
    assert q_changed_count >= 1

    expected_accepted = _margin_control_oracle_accepted_identities(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    assert expected_accepted == set(PINNED_MARGIN_CONTROL_ACCEPTED)
    q_flip_identities: set[tuple[str, int]] = set()
    for state_key in ("A", "B"):
        q_flip_identities.update(
            _q_flip_identities(
                prior_tensors[state_key].q_levels,
                gpu_wired.tensor_states[state_key].q_levels,
                state_key,
            ),
        )
    assert q_flip_identities == expected_accepted
    assert len(q_flip_identities) == q_changed_count
    assert gpu_wired.global_summary.get("selection_parity_pass") is not True
    MARGIN_ONLY_FAIL_CLOSED_PROOF["wired_path_margin_only_fail_closed_proven"] = True


def _install_seam_cap_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    sentinel_calls: dict[str, int],
) -> None:
    def _sentinel(*args, **kwargs):
        sentinel_calls["count"] += 1
        raise AssertionError(
            "seam_module.apply_global_rate_cap_reference must not run on CPU/skip path",
        )

    monkeypatch.setattr(seam_module, "apply_global_rate_cap_reference", _sentinel)


def _margin_only_skip_wrapper(
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
        "calm.llm_computer.tests.test_hrm_text_158_candidate_global_cap_wired_trainer_e2e_gpu._TRITON_AVAILABLE",
        triton_available,
    )

    sentinel_calls = {"count": 0}
    _install_seam_cap_sentinel(monkeypatch, sentinel_calls)
    _reset_margin_only_proof()
    _reset_wired_e2e_telemetry()

    with pytest.raises(outcomes.Skipped):
        _run_wired_margin_control_body()

    assert sentinel_calls["count"] == 0
    assert WIRED_E2E_TELEMETRY["seam_cap_reference_wrapper_calls"] == 0
    assert MARGIN_ONLY_FAIL_CLOSED_PROOF["wired_path_margin_only_fail_closed_proven"] is False


def test_margin_only_fixture_oracle_accepted_identities_cpu() -> None:
    tensor_states, vote_specs, sparse_events, _ = _build_two_tensor_trainer_inputs()
    global_cap_spec = _margin_control_spec()
    accepted = _margin_control_oracle_accepted_identities(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    assert accepted == set(PINNED_MARGIN_CONTROL_ACCEPTED)
    assert accepted == {("B", 0), ("B", 1)}
    assert MARGIN_ONLY_FAIL_CLOSED_PROOF["wired_path_margin_only_fail_closed_proven"] is False


def test_margin_only_non_claims_frozen() -> None:
    joined = "\n".join(MARGIN_ONLY_NON_CLAIMS)
    for term in _FORBIDDEN_MINT_TERMS:
        assert term not in joined
    assert "wired-path MARGIN-only fail-closed" in MARGIN_ONLY_NON_CLAIMS[0]
    assert "prior art" not in joined.lower()


def test_margin_only_cpu_proves_gpu_body_unexecuted_cuda_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _margin_only_skip_wrapper(
        monkeypatch,
        cuda_available=False,
        triton_available=False,
        lane_env_set=False,
    )


def test_margin_only_cpu_proves_gpu_body_unexecuted_lane_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _margin_only_skip_wrapper(
        monkeypatch,
        cuda_available=True,
        triton_available=True,
        lane_env_set=False,
    )


def test_margin_only_wired_path_rejects_hash_shuffle_gpu() -> None:
    _run_wired_refused_path(GlobalRateCapOrderingMode.HASH_SHUFFLE)


def test_margin_only_wired_path_rejects_round_robin_gpu() -> None:
    _run_wired_refused_path(GlobalRateCapOrderingMode.ROUND_ROBIN)


def test_margin_only_wired_path_margin_control_succeeds_gpu() -> None:
    _run_wired_margin_control_body()
