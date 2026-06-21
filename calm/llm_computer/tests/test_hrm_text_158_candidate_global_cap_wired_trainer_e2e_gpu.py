"""B2-5c Step-1b-(3c) wired GPU trainer END-TO-END parity vs CPU seam reference.

Equivalence/compat-only on F_TWO_TENSOR_GLOBAL: flag-ON apply_bounded_delta_vote_step
with CUDA-mirrored inputs equals CPU wired/seam oracle (q/acc, backlog, cap summary).

Uses env-gated CUDA REFERENCE q/acc apply — NOT native q/acc Triton proof.
NOT readiness / acquisition / training-success / selection_parity_pass /
optimizer_credit_state / global_cap_margin_only mint.

GPU execution is a separate +1 on hrm_text_158_gpu0 after the 3c diff gate.
"""
from __future__ import annotations

import copy
import os
from dataclasses import replace

import pytest
import torch

import calm.hrm_text_158.native_full_stack.candidate_global_cap_production_seam as seam_module
import calm.hrm_text_158.native_full_stack.global_rate_cap as grc_module
import calm.hrm_text_158.native_full_stack.global_rate_cap_gpu as gpu_mod
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
    apply_bounded_delta_vote_step,
    tensor_sha256,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapOrderingMode,
    GlobalRateCapRow,
    GlobalRateCapResult,
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
    VoteUpdatePlan,
    VoteUpdateState,
)
from calm.llm_computer.tests.test_hrm_text_158_candidate_global_cap_trainer_wiring import (
    _CAP_SUMMARY_KEYS,
    _build_two_tensor_trainer_inputs,
    _candidate_sparse_kwargs,
    _cap_summary_subset,
)

WIRED_E2E_PROOF: dict[str, bool] = {
    "gpu_wired_trainer_parity_proven": False,
}

WIRED_E2E_TELEMETRY: dict[str, int] = {
    "seam_cap_reference_wrapper_calls": 0,
    "gpu_apply_calls": 0,
    "cpu_fallback_count": 0,
    "cuda_inputs_observed": 0,
}

WIRED_E2E_NON_CLAIMS: tuple[str, ...] = (
    "B2-5c Step-1b-(3c) is wired GPU trainer step == CPU seam reference (equivalence/compat-only)",
    "B2-5c Step-1b-(3c) uses env-gated CUDA REFERENCE q/acc apply, NOT native q/acc Triton proof",
    "B2-5c Step-1b-(3c) does NOT mint selection_parity_pass",
    "B2-5c Step-1b-(3c) does NOT flip readiness / acquisition / training-success rows",
    "B2-5c Step-1b-(3c) does NOT flip optimizer_credit_state",
    "B2-5c Step-1b-(3c) does NOT flip global_cap_margin_only_reference",
)

_FORBIDDEN_MINT_TERMS: tuple[str, ...] = (
    "selection_parity_pass=True",
    "readiness_flip_authorized",
    "optimizer_credit_state_sub2_claim",
    "global_cap_margin_only_reference_flipped",
    "training_success",
    "HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY_NATIVE",
)


def _reset_wired_e2e_proof() -> None:
    WIRED_E2E_PROOF["gpu_wired_trainer_parity_proven"] = False


def _reset_wired_e2e_telemetry() -> None:
    for key in WIRED_E2E_TELEMETRY:
        WIRED_E2E_TELEMETRY[key] = 0


def _require_wired_e2e_gpu_lane_or_skip() -> None:
    if not torch.cuda.is_available():
        pytest.skip("wired e2e: CUDA absent")
    if not _TRITON_AVAILABLE:
        pytest.skip("wired e2e: Triton absent")
    if os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_ENV) != "1":
        pytest.skip(f"wired e2e: needs {RUN_GPU_GLOBAL_RATE_CAP_ENV}=1 in gpu:0 lane")
    if os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_NATIVE_SELECTION_ENV) != "1":
        pytest.skip(
            f"wired e2e: needs {RUN_GPU_GLOBAL_RATE_CAP_NATIVE_SELECTION_ENV}=1",
        )
    if os.environ.get(RUN_GPU_Q_ACC_APPLY_ENV) != "1":
        pytest.skip(f"wired e2e: needs {RUN_GPU_Q_ACC_APPLY_ENV}=1 in gpu:0 lane")


def _inputs_on_cuda(inputs: list[GlobalRateCapTensorInput]) -> bool:
    for item in inputs:
        tensors = (
            item.state.q_levels,
            item.state.accumulators,
            item.plan.q_i16,
            item.plan.new_acc_i32,
            item.plan.applied_indices,
            item.plan.applied_directions,
            item.plan.applied_thresholds,
        )
        if any(tensor.device.type != "cuda" for tensor in tensors):
            return False
    return True


def _tuple_to_global_rate_cap_row(
    row_tuple: tuple[str, int, int, int, int, int],
) -> GlobalRateCapRow:
    state_key, flat_index, local_pos, global_flat_index, abs_new_acc, threshold_abs = row_tuple
    return GlobalRateCapRow(
        state_key=state_key,
        flat_index=int(flat_index),
        local_pos=int(local_pos),
        global_flat_index=int(global_flat_index),
        abs_new_acc=int(abs_new_acc),
        threshold_abs=int(threshold_abs),
        margin_abs_over_threshold=int(abs_new_acc) - int(threshold_abs),
    )


def _adapt_gpu_apply_to_global_rate_cap_result(
    gpu_apply: gpu_mod.DeviceGlobalRateCapApplyResult,
    spec: GlobalRateCapSpec,
    *,
    tie_rule_mode: str,
    contract_name: str | None,
) -> GlobalRateCapResult:
    selection = gpu_apply.selection
    rows = [_tuple_to_global_rate_cap_row(t) for t in selection.ordered_rows_as_tuples()]
    accepted_rows = [_tuple_to_global_rate_cap_row(t) for t in selection.accepted_rows_as_tuples()]
    deferred_rows = [_tuple_to_global_rate_cap_row(t) for t in selection.deferred_rows_as_tuples()]
    tie_mode = grc_module.validate_global_tie_rule_mode(tie_rule_mode)
    shadow_summary = {
        "pre_cap_demand_sha256": grc_module._row_global_index_sha(rows),
        "global_tie_rule_mode": tie_mode,
        "exact_shadow_full_demand_sha256": grc_module._row_global_index_sha(rows),
        "exact_shadow_accepted_sha256": grc_module._row_global_index_sha(accepted_rows),
        "exact_shadow_deferred_sha256": grc_module._row_global_index_sha(deferred_rows),
        "mixed_class_count": 0,
        "mixed_class_row_count": 0,
        "max_mixed_class_cardinality": 0,
        "dropped_mass_count": 0,
        "dropped_mass_identities_sha256": grc_module._identity_sha256(set()),
        "drop_exercised": False,
    }
    if contract_name is not None:
        shadow_summary["global_rate_cap_contract_name"] = str(contract_name)

    backlog = copy.deepcopy(selection.deferred_backlog)
    accepted_count = len(accepted_rows)
    deferred_count = len(deferred_rows)
    accepted_from_prior_deferred = int(
        selection.stats.get("accepted_from_prior_deferred_count", 0),
    )
    total_q_changed = int(gpu_apply.stats.get("q_changed_count", 0))
    age_summary = grc_module._deferred_age_summary(backlog, step=spec.step)
    adapter_provenance = {
        "summary_source": "gpu_selection_tensors",
        "q_acc_source": "gpu_apply_tensor_results",
        "backend": str(selection.backend),
        "global_cap_gpu_native": True,
        "cpu_fallback_count": 0,
    }
    step_summary = {
        "global_rate_cap_enabled": True,
        "global_rate_cap_cap": int(spec.cap),
        "global_rate_cap_ordering_mode": spec.normalized_ordering_mode.value,
        "global_rate_cap_ordering_seed": int(spec.ordering_seed),
        "global_rate_cap_ordering_summary": {
            "schema_version": "global_rate_cap_ordering/v1",
            "mode": spec.normalized_ordering_mode.value,
            "default_margin_behavior_equivalent": (
                spec.normalized_ordering_mode == GlobalRateCapOrderingMode.MARGIN
            ),
            "seed": int(spec.ordering_seed),
            "global_step": int(spec.step),
            "order_key": "highest_abs_new_acc_then_lower_global_flat_index",
            "full_demand_count": len(rows),
            "selected_count": accepted_count,
            "deferred_count": deferred_count,
            "global_indices_sha256": {
                "full_demand": grc_module._row_global_index_sha(rows),
                "cap_selected": grc_module._row_global_index_sha(accepted_rows),
                "cap_deferred": grc_module._row_global_index_sha(deferred_rows),
            },
        },
        "functional_veto_policy": grc_module.DEFERRED_NON_SCOPE,
        "bad_pressure_drain_policy": grc_module.DEFERRED_NON_SCOPE,
        "global_cap_gpu_native": True,
        "wired_e2e_adapter_provenance": adapter_provenance,
        "ternary_mutation_enabled": bool(spec.mutate_outputs),
        "ternary_mutation_frozen": not bool(spec.mutate_outputs),
        "global_pre_cap_would_apply_count": len(rows),
        "global_rate_cap_accepted_count": accepted_count,
        "global_rate_cap_applied_count": accepted_count if spec.mutate_outputs else 0,
        "global_rate_cap_deferred_count": deferred_count,
        "global_rate_cap_saturated": len(rows) > int(spec.cap),
        "global_rate_cap_fill_ratio": grc_module._safe_ratio(accepted_count, int(spec.cap)),
        "global_deferred_ratio": grc_module._safe_ratio(deferred_count, len(rows)),
        "accepted_from_prior_deferred_count": accepted_from_prior_deferred,
        "accepted_fresh_count": accepted_count - accepted_from_prior_deferred,
        "q_changed_count": total_q_changed,
        **shadow_summary,
        **age_summary,
    }
    return GlobalRateCapResult(
        tensor_results=list(gpu_apply.tensor_results),
        step_summary=step_summary,
        rows=rows,
        accepted_rows=accepted_rows,
        deferred_rows=deferred_rows,
        deferred_backlog=backlog,
    )


def _mirror_plan_to_device(plan: VoteUpdatePlan, device: torch.device) -> VoteUpdatePlan:
    return VoteUpdatePlan(
        q_i16=plan.q_i16.to(device),
        new_acc_i32=plan.new_acc_i32.to(device),
        candidate_indices=plan.candidate_indices.to(device),
        pre_veto_selected_indices=plan.pre_veto_selected_indices.to(device),
        applied_indices=plan.applied_indices.to(device),
        applied_directions=plan.applied_directions.to(device),
        applied_thresholds=plan.applied_thresholds.to(device),
        replay_ce_veto_indices=plan.replay_ce_veto_indices.to(device),
        replay_veto_directions=plan.replay_veto_directions.to(device),
        replay_veto_thresholds=plan.replay_veto_thresholds.to(device),
        pc_aux_negative_indices=plan.pc_aux_negative_indices.to(device),
        pc_aux_veto_indices=plan.pc_aux_veto_indices.to(device),
        stats=plan.stats,
    )


def _mirror_cap_inputs_to_cuda(
    inputs: list[GlobalRateCapTensorInput],
) -> list[GlobalRateCapTensorInput]:
    device = torch.device("cuda")
    mirrored: list[GlobalRateCapTensorInput] = []
    for item in inputs:
        mirrored.append(
            GlobalRateCapTensorInput(
                state_key=item.state_key,
                state=VoteUpdateState(
                    q_levels=item.state.q_levels.to(device),
                    accumulators=item.state.accumulators.to(device),
                ),
                plan=_mirror_plan_to_device(item.plan, device),
                vote_inputs=None,
            ),
        )
    return mirrored


def _gpu_routing_cap_reference(
    inputs: list[GlobalRateCapTensorInput],
    spec: GlobalRateCapSpec,
    *,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None = None,
    tensor_offsets: dict[str, int] | None = None,
    tie_rule_mode: str = grc_module.EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    contract_name: str | None = None,
) -> GlobalRateCapResult:
    WIRED_E2E_TELEMETRY["seam_cap_reference_wrapper_calls"] += 1
    routed_inputs = inputs
    if any(item.state.q_levels.device.type == "cuda" for item in inputs):
        routed_inputs = _mirror_cap_inputs_to_cuda(inputs)
    if _inputs_on_cuda(routed_inputs):
        WIRED_E2E_TELEMETRY["cuda_inputs_observed"] += 1
        WIRED_E2E_TELEMETRY["gpu_apply_calls"] += 1
        gpu_apply = gpu_mod.apply_global_rate_cap_torch_cuda_reference_under_margin(
            routed_inputs,
            spec,
            tensor_offsets=tensor_offsets,
            deferred_backlog=deferred_backlog,
        )
        return _adapt_gpu_apply_to_global_rate_cap_result(
            gpu_apply,
            spec,
            tie_rule_mode=tie_rule_mode,
            contract_name=contract_name,
        )
    WIRED_E2E_TELEMETRY["cpu_fallback_count"] += 1
    return seam_module._wired_e2e_real_apply_global_rate_cap_reference(
        inputs,
        spec,
        deferred_backlog=deferred_backlog,
        tensor_offsets=tensor_offsets,
        tie_rule_mode=tie_rule_mode,
        contract_name=contract_name,
    )


def _install_gpu_cap_route_patch() -> None:
    if not hasattr(seam_module, "_wired_e2e_real_apply_global_rate_cap_reference"):
        seam_module._wired_e2e_real_apply_global_rate_cap_reference = (
            seam_module.apply_global_rate_cap_reference
        )
    seam_module.apply_global_rate_cap_reference = _gpu_routing_cap_reference


def _restore_cpu_cap_route_patch() -> None:
    if hasattr(seam_module, "_wired_e2e_real_apply_global_rate_cap_reference"):
        seam_module.apply_global_rate_cap_reference = (
            seam_module._wired_e2e_real_apply_global_rate_cap_reference
        )


def _mirror_tensor_states_to_cuda(
    tensor_states: dict[str, BoundedDeltaTensorState],
) -> dict[str, BoundedDeltaTensorState]:
    device = torch.device("cuda")
    mirrored: dict[str, BoundedDeltaTensorState] = {}
    for state_key, state in tensor_states.items():
        shadow = state.exact_accumulator_shadow
        mirrored[state_key] = replace(
            state,
            q_levels=state.q_levels.to(device),
            frozen_scale=state.frozen_scale.to(device),
            exact_accumulator_shadow=shadow.to(device) if shadow is not None else None,
        )
    return mirrored


def _assert_trainer_states_on_cuda(tensor_states: dict[str, BoundedDeltaTensorState]) -> None:
    for state in tensor_states.values():
        assert state.q_levels.device.type == "cuda"
        assert state.frozen_scale.device.type == "cuda"
        if state.exact_accumulator_shadow is not None:
            assert state.exact_accumulator_shadow.device.type == "cuda"


def _assert_fixture_semantics_from_trainer_inputs(
    tensor_states: dict[str, BoundedDeltaTensorState],
    vote_specs,
    sparse_events,
    global_cap_spec: GlobalRateCapSpec,
) -> None:
    from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
        encode_budget_capped_hybrid_reference,
        execute_direct_bounded_local_vote_update_candidate,
    )
    from calm.hrm_text_158.native_full_stack.candidate_global_cap_production_seam import (
        CandidateGlobalCapSeamEntry,
    )

    entries: dict[str, CandidateGlobalCapSeamEntry] = {}
    for state_key, prior_tensor in sorted(tensor_states.items()):
        vu = prior_tensor.vote_update_state()
        bounded = encode_budget_capped_hybrid_reference(
            vu,
            hot_exact_indices=prior_tensor.rebuild_hot_exact_indices(),
            cold_default_value=prior_tensor.rebuild_cold_default_value(),
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
    cap_inputs = []
    seam = seam_module.apply_candidate_global_cap_production_seam(entries, global_cap_spec)
    cap_inputs = list(seam.cap_inputs)
    assert global_cap_spec.normalized_ordering_mode == GlobalRateCapOrderingMode.MARGIN
    offsets = tensor_offsets_for_vote_update_states(cap_inputs)
    oracle_rows, _, _ = select_global_rate_cap_rows(
        cap_inputs,
        global_cap_spec,
        tensor_offsets=offsets,
    )
    assert len(oracle_rows) == 8
    for item in cap_inputs:
        assert int(item.plan.applied_indices.numel()) == 4
    b_entry = next(item for item in cap_inputs if item.state_key == "B")
    b_acc = b_entry.state.accumulators.to(torch.int64)
    assert int(b_acc[0].item()) == 16
    assert int(b_acc[1].item()) == -16


def _assert_wired_e2e_parity(cpu_oracle, gpu_wired) -> None:
    for state_key in ("A", "B"):
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


def _wired_trainer_gpu_e2e_body() -> None:
    _require_wired_e2e_gpu_lane_or_skip()
    _reset_wired_e2e_telemetry()
    tensor_states, vote_specs, sparse_events, global_cap_spec = _build_two_tensor_trainer_inputs()
    _assert_fixture_semantics_from_trainer_inputs(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    cpu_oracle = apply_bounded_delta_vote_step(
        **_candidate_sparse_kwargs(
            tensor_states,
            vote_specs,
            sparse_events,
            global_cap_spec=global_cap_spec,
            seam_enabled=True,
        ),
    )
    cuda_states = _mirror_tensor_states_to_cuda(tensor_states)
    _assert_trainer_states_on_cuda(cuda_states)
    _install_gpu_cap_route_patch()
    try:
        gpu_wired = apply_bounded_delta_vote_step(
            **_candidate_sparse_kwargs(
                cuda_states,
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
    provenance = gpu_wired.global_summary.get("wired_e2e_adapter_provenance") or {}
    assert provenance.get("summary_source") == "gpu_selection_tensors"
    assert provenance.get("global_cap_gpu_native") is True
    assert provenance.get("cpu_fallback_count") == 0
    _assert_wired_e2e_parity(cpu_oracle, gpu_wired)
    WIRED_E2E_PROOF["gpu_wired_trainer_parity_proven"] = True


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


def _assert_gpu_body_skips_without_seam_call(
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
    _reset_wired_e2e_proof()
    _reset_wired_e2e_telemetry()

    with pytest.raises(outcomes.Skipped):
        _wired_trainer_gpu_e2e_body()

    assert sentinel_calls["count"] == 0
    assert WIRED_E2E_TELEMETRY["seam_cap_reference_wrapper_calls"] == 0
    assert WIRED_E2E_PROOF["gpu_wired_trainer_parity_proven"] is False


def test_wired_e2e_non_claims_frozen() -> None:
    joined = "\n".join(WIRED_E2E_NON_CLAIMS)
    for term in _FORBIDDEN_MINT_TERMS:
        assert term not in joined
    assert "equivalence/compat-only" in WIRED_E2E_NON_CLAIMS[0]
    assert "CUDA REFERENCE q/acc apply" in WIRED_E2E_NON_CLAIMS[1]


def test_wired_e2e_fixture_semantics_cpu() -> None:
    tensor_states, vote_specs, sparse_events, global_cap_spec = _build_two_tensor_trainer_inputs()
    _assert_fixture_semantics_from_trainer_inputs(
        tensor_states,
        vote_specs,
        sparse_events,
        global_cap_spec,
    )
    assert WIRED_E2E_PROOF["gpu_wired_trainer_parity_proven"] is False


def test_wired_e2e_cpu_proves_gpu_body_unexecuted_cuda_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_gpu_body_skips_without_seam_call(
        monkeypatch,
        cuda_available=False,
        triton_available=False,
        lane_env_set=False,
    )


def test_wired_e2e_cpu_proves_gpu_body_unexecuted_lane_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_gpu_body_skips_without_seam_call(
        monkeypatch,
        cuda_available=True,
        triton_available=True,
        lane_env_set=False,
    )


def test_wired_e2e_device_mirror_cuda_audit() -> None:
    _require_wired_e2e_gpu_lane_or_skip()
    tensor_states, _, _, _ = _build_two_tensor_trainer_inputs()
    cuda_states = _mirror_tensor_states_to_cuda(tensor_states)
    _assert_trainer_states_on_cuda(cuda_states)


def test_wired_e2e_f_two_tensor_global_flag_on_parity() -> None:
    _reset_wired_e2e_proof()
    _wired_trainer_gpu_e2e_body()
    assert WIRED_E2E_PROOF["gpu_wired_trainer_parity_proven"] is True
