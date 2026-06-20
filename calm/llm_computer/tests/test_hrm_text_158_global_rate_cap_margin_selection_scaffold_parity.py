"""B2-5a CPU scaffold MARGIN-selection ordering parity (11 cases).

Prior native-named GPU parity proof removed/fail-closed on B2-5a null.  This
module compares the NON-NATIVE CPU scaffold ordering against the CPU oracle.
Positive ordering cases require a granted gpu:0 lane + scaffold env on CUDA
boxes (fail-not-skip when capacity exists but env unset).  Structural rejection
cases run on CPU without lane env.
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
from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import (
    DeviceGlobalRateCapSelectionResult,
    RUN_GPU_GLOBAL_RATE_CAP_ENV,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_packed_key_scaffold import (
    GlobalRateCapMarginSelectionFeasibilityNull,
    RUN_GPU_GLOBAL_RATE_CAP_SCAFFOLD_ENV,
    select_global_rate_cap_rows_margin_scaffold,
    _TRITON_AVAILABLE,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
)

_PARITY_ENV_OK = (
    os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_ENV) == "1"
    and os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_SCAFFOLD_ENV) == "1"
)


def _require_gpu_lane_or_fail():
    """Guard for the positive-parity body.  Fail-not-skip means: when the
    box HAS CUDA + Triton capacity but the env lane is unset, we pytest.fail
    (no silent skip laundering a missing pass into green).  When the box
    genuinely lacks CUDA / Triton (this CPU box), we skip structural+positive
    parity and rely on the dedicated fail-not-skip contract test below."""
    cuda_ok = torch.cuda.is_available()
    triton_ok = _TRITON_AVAILABLE
    if not cuda_ok:
        pytest.skip(
            "B2-5a positive-parity body requires a CUDA box; no CUDA here. "
            "Structural + rejection cases run on CPU."
        )
    if not triton_ok:
        pytest.skip(
            "B2-5a positive-parity body requires Triton; not installed here."
        )
    # Fail-not-skip only when the env was ARMED for a scaffold launch but the
    # scaffold gate is unset on a CUDA-capable box — that is the laundering case.
    if cuda_ok and triton_ok and not _PARITY_ENV_OK:
        pytest.fail(
            f"CUDA + Triton present but env unset: {RUN_GPU_GLOBAL_RATE_CAP_ENV}=1 "
            f"AND {RUN_GPU_GLOBAL_RATE_CAP_SCAFFOLD_ENV}=1 are required for scaffold routing. "
            "Fail-not-skip: no silent fallback when capacity exists (NOT a native pass)."
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


def _state(q, acc, *, device) -> VoteUpdateState:
    return VoteUpdateState(
        q_levels=torch.as_tensor(q, dtype=torch.int8, device=device),
        accumulators=torch.as_tensor(acc, dtype=torch.int16, device=device),
    )


def _inputs(votes, *, device, **kwargs) -> VoteUpdateInputs:
    converted = {}
    for name, value in kwargs.items():
        converted[name] = (
            None
            if value is None
            else torch.as_tensor(
                value,
                dtype=torch.int8 if name.endswith("moves") else torch.int16,
                device=device,
            )
        )
    return VoteUpdateInputs(
        votes=torch.as_tensor(votes, dtype=torch.int16, device=device), **converted
    )


def _tensor_input(state_key, q, acc, votes, *, device, **vote_kwargs) -> GlobalRateCapTensorInput:
    state = _state(q, acc, device=device)
    return GlobalRateCapTensorInput(
        state_key=state_key,
        state=state,
        plan=plan_integer_vote_update_reference(
            state, _inputs(votes, device=device, **vote_kwargs), _spec()
        ),
    )


def _byte_hash(tensor: torch.Tensor) -> bytes:
    return (
        tensor.detach().cpu().contiguous().view(-1).to(torch.int64).numpy().tobytes()
    )


_CASES = {
    "cap_zero": lambda d, c: (
        [
            _tensor_input("a", [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 30, 30], device=d),
            _tensor_input("b", [0, 0, 0, 0], [0, 0, 0, 0], [30, 30, 0, 0], device=d),
        ],
        GlobalRateCapSpec(cap=0, step=1, ordering_mode=GlobalRateCapOrderingMode.MARGIN),
        None,
    ),
    "cap_saturated": lambda d, c: (
        [
            _tensor_input("a", [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 30, 30], device=d),
            _tensor_input("b", [0, 0, 0, 0], [0, 0, 0, 0], [30, 30, 0, 0], device=d),
        ],
        GlobalRateCapSpec(cap=1, step=1, ordering_mode=GlobalRateCapOrderingMode.MARGIN),
        None,
    ),
    "cap_equals_rows": lambda d, c: (
        [
            _tensor_input("a", [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 30, 30], device=d),
            _tensor_input("b", [0, 0, 0, 0], [0, 0, 0, 0], [30, 30, 0, 0], device=d),
        ],
        GlobalRateCapSpec(cap=4, step=1, ordering_mode=GlobalRateCapOrderingMode.MARGIN),
        None,
    ),
    "cap_exceeds_rows": lambda d, c: (
        [
            _tensor_input("a", [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 30, 30], device=d),
            _tensor_input("b", [0, 0, 0, 0], [0, 0, 0, 0], [30, 30, 0, 0], device=d),
        ],
        GlobalRateCapSpec(cap=6, step=1, ordering_mode=GlobalRateCapOrderingMode.MARGIN),
        None,
    ),
    "empty_one_state": lambda d, c: (
        [
            _tensor_input("empty", [0, 0], [0, 0], [0, 0], device=d),
            _tensor_input("nonempty", [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 30, 30], device=d),
        ],
        GlobalRateCapSpec(cap=2, step=1, ordering_mode=GlobalRateCapOrderingMode.MARGIN),
        None,
    ),
    "empty_all_states": lambda d, c: (
        [
            _tensor_input("a", [0, 0], [0, 0], [0, 0], device=d),
            _tensor_input("b", [0, 0], [0, 0], [0, 0], device=d),
        ],
        GlobalRateCapSpec(cap=2, step=1, ordering_mode=GlobalRateCapOrderingMode.MARGIN),
        None,
    ),
    "cross_state_abs_tie": lambda d, c: (
        [
            _tensor_input("proj_in", [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 30, 30], device=d),
            _tensor_input("proj_out", [0, 0, 0, 0], [0, 0, 0, 0], [30, 30, 0, 0], device=d),
        ],
        GlobalRateCapSpec(cap=3, step=1, ordering_mode=GlobalRateCapOrderingMode.MARGIN),
        None,
    ),
    "same_state_abs_tie": lambda d, c: (
        [
            _tensor_input("a", [0, 0, 0, 0], [0, 0, 0, 0], [10, 10, 10, 10], device=d),
        ],
        GlobalRateCapSpec(cap=2, step=1, ordering_mode=GlobalRateCapOrderingMode.MARGIN),
        None,
    ),
    "non_contiguous_non_overlapping": lambda d, c: (
        [
            _tensor_input("proj_in", [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 30, 30], device=d),
            _tensor_input("proj_out", [0, 0, 0, 0], [0, 0, 0, 0], [30, 30, 0, 0], device=d),
        ],
        GlobalRateCapSpec(cap=3, step=1, ordering_mode=GlobalRateCapOrderingMode.MARGIN),
        {"proj_in": 0, "proj_out": 10},
    ),
    "negative_offset_reject": lambda d, c: (
        [
            _tensor_input("proj_in", [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 30, 30], device=d),
            _tensor_input("proj_out", [0, 0, 0, 0], [0, 0, 0, 0], [30, 30, 0, 0], device=d),
        ],
        GlobalRateCapSpec(cap=3, step=1, ordering_mode=GlobalRateCapOrderingMode.MARGIN),
        {"proj_in": 0, "proj_out": -5},
    ),
    "overlapping_custom_offsets": lambda d, c: (
        [
            _tensor_input("a", [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 30, 30], device=d),
            _tensor_input("b", [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 30, 30], device=d),
        ],
        GlobalRateCapSpec(cap=3, step=1, ordering_mode=GlobalRateCapOrderingMode.MARGIN),
        {"a": 0, "b": 0},
    ),
}


@pytest.mark.parametrize("case_name", sorted(_CASES.keys()))
def test_scaffold_margin_selection_ordering_parity(case_name):
    # Positive ordering body: requires granted gpu:0 lane + scaffold env on CUDA
    # boxes; fail-not-skip when capacity exists but env unset.  Rejection cases
    # (neg-offset, overlap, empty_all_states) run on CPU without lane env.
    is_structural_case = case_name in (
        "negative_offset_reject",
        "overlapping_custom_offsets",
        "empty_all_states",
    )
    if not is_structural_case:
        _require_gpu_lane_or_fail()
    # CPU-capable device for the structural cases (no env arming needed).
    device = (
        torch.device("cuda")
        if is_structural_case and torch.cuda.is_available()
        else torch.device("cpu")
    )
    inputs, spec, offsets = _CASES[case_name](device, None)

    if case_name == "negative_offset_reject":
        with pytest.raises(ValueError, match=r"must be >= 0"):
            select_global_rate_cap_rows_margin_scaffold(
                inputs, spec, tensor_offsets=offsets
            )
        return

    if case_name == "overlapping_custom_offsets":
        with pytest.raises(ValueError, match=r"duplicate"):
            select_global_rate_cap_rows_margin_scaffold(
                inputs, spec, tensor_offsets=offsets
            )
        return

    ordered, accepted_rows, deferred_rows = select_global_rate_cap_rows(
        inputs, spec, tensor_offsets=offsets
    )
    selection, receipt = select_global_rate_cap_rows_margin_scaffold(
        inputs, spec, tensor_offsets=offsets
    )
    # empty_all_states early-return path
    if case_name == "empty_all_states":
        assert receipt.empty_branch_taken is True
        assert receipt.row_count == 0
        assert receipt.observed_max_abs_observed == -1
        assert receipt.selection_parity_pass is False
        assert len(ordered) == 0 and len(accepted_rows) == 0 and len(deferred_rows) == 0
        return

    oracle_ordered = [r.global_flat_index for r in ordered]
    oracle_accepted = [r.global_flat_index for r in accepted_rows]
    oracle_deferred = [r.global_flat_index for r in deferred_rows]
    scaffold_ordered = selection.row_global_flat_indices.tolist()
    scaffold_accepted = selection.row_global_flat_indices[selection.accepted_positions].tolist()
    scaffold_deferred = selection.row_global_flat_indices[selection.deferred_positions].tolist()
    assert oracle_ordered == scaffold_ordered
    assert oracle_accepted == scaffold_accepted
    assert oracle_deferred == scaffold_deferred
    # byte-exact hashes on accepted/deferred row tensors
    assert (
        _byte_hash(selection.row_global_flat_indices[selection.accepted_positions])
        == _byte_hash(torch.as_tensor(oracle_accepted, dtype=torch.int64, device=device))
    )
    assert receipt.negative_offset_reject_evidence is True
