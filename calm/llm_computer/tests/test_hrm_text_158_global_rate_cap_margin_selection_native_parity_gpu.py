"""B2-5a′ Stage-2 native MARGIN-selection GPU parity (fail-not-skip contract).

SOLE legitimate minter of selection_parity_pass=True on GPU after exact parity.
On CPU boxes without CUDA lane env: fail-not-skip when capacity exists.
Structural CPU tests: dispatch defer/refusal + sentinel guards without GPU launch.
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
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import RUN_GPU_GLOBAL_RATE_CAP_ENV
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_dispatch import (
    RUN_GPU_GLOBAL_RATE_CAP_NATIVE_SELECTION_ENV,
    select_global_rate_cap_rows_margin_native,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_packed_key_scaffold import (
    LEGACY_RUN_GPU_GLOBAL_RATE_CAP_NATIVE_ENV,
    _TRITON_AVAILABLE,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_parity_receipt import (
    NativeSelectionParityProof,
    apply_native_selection_parity_proof,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_step0_budget import (
    build_upper_bound_fixture_inputs,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_triton_kernel import (
    INT64_MAX,
    evaluate_padding_headroom,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
)

_NATIVE_ENV_OK = (
    os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_ENV) == "1"
    and os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_NATIVE_SELECTION_ENV) == "1"
)


def _spec(**kwargs) -> VoteUpdateSpec:
    base = dict(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=1,
        max_abs_per_tensor=64,
        fraction_per_tensor=1.0,
    )
    base.update(kwargs)
    return VoteUpdateSpec(**base)


def _tensor_input(state_key, q, acc, votes, *, device) -> GlobalRateCapTensorInput:
    state = VoteUpdateState(
        q_levels=torch.as_tensor(q, dtype=torch.int8, device=device),
        accumulators=torch.as_tensor(acc, dtype=torch.int16, device=device),
    )
    return GlobalRateCapTensorInput(
        state_key=state_key,
        state=state,
        plan=plan_integer_vote_update_reference(
            state,
            VoteUpdateInputs(votes=torch.as_tensor(votes, dtype=torch.int16, device=device)),
            _spec(),
        ),
    )


def _require_gpu_lane_or_fail() -> None:
    cuda_ok = torch.cuda.is_available()
    if not cuda_ok:
        pytest.skip("GPU parity body requires CUDA; structural CPU tests still run")
    if not _TRITON_AVAILABLE:
        pytest.skip("GPU parity body requires Triton")
    if not _NATIVE_ENV_OK:
        pytest.fail(
            f"CUDA+Triton present but env unset: need {RUN_GPU_GLOBAL_RATE_CAP_ENV}=1 and "
            f"{RUN_GPU_GLOBAL_RATE_CAP_NATIVE_SELECTION_ENV}=1 (fail-not-skip)"
        )


def test_legacy_native_env_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, "1")
    monkeypatch.setenv(RUN_GPU_GLOBAL_RATE_CAP_NATIVE_SELECTION_ENV, "1")
    monkeypatch.setenv(LEGACY_RUN_GPU_GLOBAL_RATE_CAP_NATIVE_ENV, "1")
    device = torch.device("cpu")
    inputs = [
        _tensor_input("a", [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 30, 30], device=device),
    ]
    spec = GlobalRateCapSpec(cap=2, step=1, ordering_mode=GlobalRateCapOrderingMode.MARGIN)
    with pytest.raises(RuntimeError, match="fail-closed"):
        select_global_rate_cap_rows_margin_native(inputs, spec)


def test_row_count_gt_block_refusal_cpu(monkeypatch) -> None:
    monkeypatch.setenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, "1")
    monkeypatch.setenv(RUN_GPU_GLOBAL_RATE_CAP_NATIVE_SELECTION_ENV, "1")
    inputs = build_upper_bound_fixture_inputs(numel=4096, max_abs_per_tensor=256, num_states=8)
    spec = GlobalRateCapSpec(cap=512, step=1, ordering_mode=GlobalRateCapOrderingMode.MARGIN)
    _, receipt = select_global_rate_cap_rows_margin_native(inputs, spec)
    assert receipt.multiblock_deferred is True
    assert receipt.selection_parity_pass is False
    assert receipt.row_count > 1024


def test_sentinel_case_b_budget_infeasible() -> None:
    headroom = evaluate_padding_headroom(host_max_full_key=INT64_MAX, full_pack_bits=63)
    assert headroom["budget_infeasible"] is True


def test_gpu_parity_cross_tie_fail_not_skip_contract() -> None:
    _require_gpu_lane_or_fail()
    device = torch.device("cuda")
    inputs = [
        _tensor_input("a", [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 30, 30], device=device),
        _tensor_input("b", [0, 0, 0, 0], [0, 0, 0, 0], [30, 30, 0, 0], device=device),
    ]
    spec = GlobalRateCapSpec(cap=3, step=1, ordering_mode=GlobalRateCapOrderingMode.MARGIN)
    oracle_rows, accepted_rows, deferred_rows = select_global_rate_cap_rows(inputs, spec)
    selection, receipt = select_global_rate_cap_rows_margin_native(inputs, spec)
    assert receipt.selection_parity_pass is False
    assert receipt.parity_proof is None
    oracle_global = [r.global_flat_index for r in oracle_rows]
    native_global = selection.row_global_flat_indices.detach().cpu().tolist()
    assert native_global == oracle_global
    accepted_oracle = {r.global_flat_index for r in accepted_rows}
    deferred_oracle = {r.global_flat_index for r in deferred_rows}
    accepted_native = set(
        selection.row_global_flat_indices[selection.accepted_positions].detach().cpu().tolist()
    )
    deferred_native = set(
        selection.row_global_flat_indices[selection.deferred_positions].detach().cpu().tolist()
        if selection.deferred_positions.numel() > 0
        else []
    )
    ordered_global_indices_match = native_global == oracle_global
    accepted_positions_match = accepted_native == accepted_oracle
    deferred_positions_match = deferred_native == deferred_oracle
    parity_ok = (
        ordered_global_indices_match
        and accepted_positions_match
        and deferred_positions_match
    )
    proof = NativeSelectionParityProof(
        parity_ok=parity_ok,
        ordered_global_indices_match=ordered_global_indices_match,
        accepted_positions_match=accepted_positions_match,
        deferred_positions_match=deferred_positions_match,
    )
    assert receipt.token is not None
    receipt = apply_native_selection_parity_proof(
        receipt, parity_proof=proof, token=receipt.token
    )
    assert receipt.selection_parity_pass is True
    assert receipt.single_block_regime is True
    assert receipt.native_path_audit_pass is True


def test_gpu_module_unexecuted_for_pass_on_cpu_without_lane() -> None:
    if torch.cuda.is_available() and _TRITON_AVAILABLE:
        pytest.skip(
            "CUDA+Triton box without lane env: fail-not-skip covered by "
            "test_gpu_parity_cross_tie_fail_not_skip_contract; GPU parity UNEXECUTED-for-pass"
        )
