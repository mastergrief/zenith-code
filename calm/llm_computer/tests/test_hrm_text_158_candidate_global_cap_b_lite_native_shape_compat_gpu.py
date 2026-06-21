"""B2-5c Step-1b-(3b) B-lite native-selection SHAPE-COMPATIBILITY on seam cap_inputs.

Compat-only: proves F_TWO_TENSOR_GLOBAL seam cap_inputs feed native MARGIN selection
on CUDA with CPU-oracle identity + ordering parity. NOT selection_parity_pass /
readiness / training-success / optimizer_credit_state / global_cap_margin_only.

CPU boxes: GPU body skips cleanly (no false pass). GPU execution is a separate +1
on hrm_text_158_gpu0 after the 3b diff gate.
"""
from __future__ import annotations

import os

import pytest
import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapOrderingMode,
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    select_global_rate_cap_rows,
    tensor_offsets_for_vote_update_states,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import RUN_GPU_GLOBAL_RATE_CAP_ENV
import calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_dispatch as dispatch_mod
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_dispatch import (
    RUN_GPU_GLOBAL_RATE_CAP_NATIVE_SELECTION_ENV,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_packed_key_scaffold import (
    _TRITON_AVAILABLE,
)
from calm.hrm_text_158.native_full_stack.candidate_global_cap_production_seam import (
    apply_candidate_global_cap_production_seam,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdatePlan, VoteUpdateState
from calm.llm_computer.tests.test_hrm_text_158_candidate_global_cap_production_seam import (
    _build_two_tensor_global_entries,
)

B_LITE_COMPAT_PROOF: dict[str, bool] = {
    "b_lite_native_shape_compat_proven": False,
}

B_LITE_NON_CLAIMS: tuple[str, ...] = (
    "B2-5c Step-1b-(3b) is native-selection SHAPE-COMPATIBILITY on seam cap_inputs only",
    "B2-5c Step-1b-(3b) does NOT mint selection_parity_pass",
    "B2-5c Step-1b-(3b) does NOT flip readiness / acquisition / training-success rows",
    "B2-5c Step-1b-(3b) does NOT flip optimizer_credit_state",
    "B2-5c Step-1b-(3b) does NOT flip global_cap_margin_only_reference",
    "B2-5c Step-1b-(3b) does NOT wire native selector into the trainer loop",
)

_FORBIDDEN_MINT_TERMS: tuple[str, ...] = (
    "selection_parity_pass=True",
    "readiness_flip_authorized",
    "optimizer_credit_state_sub2_claim",
    "global_cap_margin_only_reference_flipped",
    "training_success",
)


def _reset_b_lite_proof() -> None:
    B_LITE_COMPAT_PROOF["b_lite_native_shape_compat_proven"] = False


def _build_seam_cap_inputs() -> tuple[list[GlobalRateCapTensorInput], GlobalRateCapSpec]:
    entries, global_cap_spec, _ = _build_two_tensor_global_entries()
    seam = apply_candidate_global_cap_production_seam(entries, global_cap_spec)
    return list(seam.cap_inputs), global_cap_spec


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
    *,
    device: torch.device | None = None,
) -> list[GlobalRateCapTensorInput]:
    cuda = device or torch.device("cuda")
    mirrored: list[GlobalRateCapTensorInput] = []
    for item in inputs:
        mirrored.append(
            GlobalRateCapTensorInput(
                state_key=item.state_key,
                state=VoteUpdateState(
                    q_levels=item.state.q_levels.to(cuda),
                    accumulators=item.state.accumulators.to(cuda),
                ),
                plan=_mirror_plan_to_device(item.plan, cuda),
                vote_inputs=None,
            ),
        )
    return mirrored


def _require_b_lite_gpu_lane_or_skip() -> None:
    if not torch.cuda.is_available():
        pytest.skip("B-lite shape compat: CUDA absent")
    if not _TRITON_AVAILABLE:
        pytest.skip("B-lite shape compat: Triton absent")
    if os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_ENV) != "1":
        pytest.skip(f"B-lite shape compat: needs {RUN_GPU_GLOBAL_RATE_CAP_ENV}=1 in gpu:0 lane")
    if os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_NATIVE_SELECTION_ENV) != "1":
        pytest.skip(
            f"B-lite shape compat: needs {RUN_GPU_GLOBAL_RATE_CAP_NATIVE_SELECTION_ENV}=1",
        )


def _assert_fixture_semantics(
    cap_inputs: list[GlobalRateCapTensorInput],
    spec: GlobalRateCapSpec,
) -> None:
    assert spec.normalized_ordering_mode == GlobalRateCapOrderingMode.MARGIN
    offsets = tensor_offsets_for_vote_update_states(cap_inputs)
    oracle_rows, _, _ = select_global_rate_cap_rows(
        cap_inputs,
        spec,
        tensor_offsets=offsets,
    )
    assert len(oracle_rows) == 8
    assert {item.state_key for item in cap_inputs} == {"A", "B"}
    for item in cap_inputs:
        assert int(item.plan.applied_indices.numel()) == 4
    b_entry = next(item for item in cap_inputs if item.state_key == "B")
    b_acc = b_entry.state.accumulators.to(torch.int64)
    assert int(b_acc[0].item()) == 16
    assert int(b_acc[1].item()) == -16


def _b_lite_native_shape_compat_gpu_body() -> None:
    _require_b_lite_gpu_lane_or_skip()
    cap_inputs_cpu, spec = _build_seam_cap_inputs()
    _assert_fixture_semantics(cap_inputs_cpu, spec)
    cap_inputs_cuda = _mirror_cap_inputs_to_cuda(cap_inputs_cpu)
    device = dispatch_mod._common_device(cap_inputs_cuda)
    assert device.type == "cuda"

    offsets = tensor_offsets_for_vote_update_states(cap_inputs_cuda)
    oracle_rows, accepted_rows, deferred_rows = select_global_rate_cap_rows(
        cap_inputs_cpu,
        spec,
        tensor_offsets=offsets,
    )
    selection, receipt = dispatch_mod.select_global_rate_cap_rows_margin_native(
        cap_inputs_cuda,
        spec,
        tensor_offsets=offsets,
    )

    assert receipt.selection_parity_pass is False
    assert receipt.parity_proof is None
    assert receipt.row_count == 8
    assert receipt.single_block_regime is True

    oracle_global = [int(row.global_flat_index) for row in oracle_rows]
    native_global = selection.row_global_flat_indices.detach().cpu().tolist()
    assert native_global == oracle_global

    accepted_oracle = {(row.state_key, int(row.flat_index)) for row in accepted_rows}
    deferred_oracle = {(row.state_key, int(row.flat_index)) for row in deferred_rows}
    accepted_native = {
        (oracle_rows[int(pos)].state_key, int(oracle_rows[int(pos)].flat_index))
        for pos in selection.accepted_positions.detach().cpu().tolist()
    }
    deferred_native = {
        (oracle_rows[int(pos)].state_key, int(oracle_rows[int(pos)].flat_index))
        for pos in selection.deferred_positions.detach().cpu().tolist()
    }
    assert accepted_native == accepted_oracle
    assert deferred_native == deferred_oracle

    B_LITE_COMPAT_PROOF["b_lite_native_shape_compat_proven"] = True


def test_b_lite_non_claims_frozen() -> None:
    for term in _FORBIDDEN_MINT_TERMS:
        joined = "\n".join(B_LITE_NON_CLAIMS)
        assert term not in joined
    assert "SHAPE-COMPATIBILITY" in B_LITE_NON_CLAIMS[0]


def test_b_lite_f_two_tensor_global_fixture_semantics_cpu() -> None:
    cap_inputs, spec = _build_seam_cap_inputs()
    _assert_fixture_semantics(cap_inputs, spec)
    assert B_LITE_COMPAT_PROOF["b_lite_native_shape_compat_proven"] is False


def _install_native_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    native_calls: dict[str, int],
) -> None:
    def _sentinel(*args, **kwargs):
        native_calls["count"] += 1
        raise AssertionError(
            "select_global_rate_cap_rows_margin_native must not run on CPU/skip path",
        )

    monkeypatch.setattr(
        dispatch_mod,
        "select_global_rate_cap_rows_margin_native",
        _sentinel,
    )


def _assert_gpu_body_skips_without_native_call(
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
    else:
        monkeypatch.delenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, raising=False)
        monkeypatch.delenv(RUN_GPU_GLOBAL_RATE_CAP_NATIVE_SELECTION_ENV, raising=False)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(
        "calm.llm_computer.tests.test_hrm_text_158_candidate_global_cap_b_lite_native_shape_compat_gpu._TRITON_AVAILABLE",
        triton_available,
    )

    native_calls = {"count": 0}
    _install_native_sentinel(monkeypatch, native_calls)
    _reset_b_lite_proof()

    with pytest.raises(outcomes.Skipped):
        _b_lite_native_shape_compat_gpu_body()

    assert native_calls["count"] == 0
    assert B_LITE_COMPAT_PROOF["b_lite_native_shape_compat_proven"] is False


def test_b_lite_cpu_proves_gpu_body_unexecuted_cuda_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_gpu_body_skips_without_native_call(
        monkeypatch,
        cuda_available=False,
        triton_available=False,
        lane_env_set=False,
    )


def test_b_lite_cpu_proves_gpu_body_unexecuted_lane_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_gpu_body_skips_without_native_call(
        monkeypatch,
        cuda_available=True,
        triton_available=True,
        lane_env_set=False,
    )


def test_b_lite_device_mirror_common_device_cuda() -> None:
    _require_b_lite_gpu_lane_or_skip()
    cap_inputs_cpu, spec = _build_seam_cap_inputs()
    cap_inputs_cuda = _mirror_cap_inputs_to_cuda(cap_inputs_cpu)
    device = dispatch_mod._common_device(cap_inputs_cuda)
    assert device.type == "cuda"
    del spec


def test_b_lite_f_two_tensor_global_native_shape_compat() -> None:
    _reset_b_lite_proof()
    _b_lite_native_shape_compat_gpu_body()
    assert B_LITE_COMPAT_PROOF["b_lite_native_shape_compat_proven"] is True
