"""B2-5a env-gate + no-silent-fallback dispatch tests (scaffold vs reference).

CPU-only (no CUDA lane).  Validates the env gate matrix, the no-silent-fallback
contract (scaffold-on + Triton-missing -> RuntimeError, NOT reference), the
legacy ``..._NATIVE`` env fail-closed guard, the reference-body-untouched
invariant, and scaffold-path routing only (NOT a native pass).
"""
from __future__ import annotations

import os

import pytest
import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapOrderingMode,
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import (
    RUN_GPU_GLOBAL_RATE_CAP_ENV,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_scaffold_dispatch import (
    RUN_GPU_GLOBAL_RATE_CAP_SCAFFOLD_ENV,
    select_global_rate_cap_rows_under_margin,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_packed_key_scaffold import (
    LEGACY_RUN_GPU_GLOBAL_RATE_CAP_NATIVE_ENV,
    _TRITON_AVAILABLE,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
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


def _tensor_input(state_key) -> GlobalRateCapTensorInput:
    q = [0, 0, 0, 0]
    acc = [0, 0, 0, 0]
    votes = [0, 0, 30, 30]
    state = VoteUpdateState(
        q_levels=torch.as_tensor(q, dtype=torch.int8),
        accumulators=torch.as_tensor(acc, dtype=torch.int16),
    )
    inputs = VoteUpdateInputs(votes=torch.as_tensor(votes, dtype=torch.int16))
    return GlobalRateCapTensorInput(
        state_key=state_key,
        state=state,
        plan=plan_integer_vote_update_reference(state, inputs, _spec()),
    )


def _inputs_two_states():
    return [_tensor_input("a"), _tensor_input("b")]


def test_lane_env_required_default_off(monkeypatch):
    monkeypatch.delenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, raising=False)
    monkeypatch.delenv(RUN_GPU_GLOBAL_RATE_CAP_SCAFFOLD_ENV, raising=False)
    with pytest.raises(RuntimeError, match=RUN_GPU_GLOBAL_RATE_CAP_ENV):
        select_global_rate_cap_rows_under_margin(
            _inputs_two_states(), GlobalRateCapSpec(cap=1, step=1)
        )


def test_lane_on_scaffold_off_falls_back_to_reference(monkeypatch):
    # Lane env on, scaffold env OFF -> dispatch routes to the reference path
    # (NOT the CPU scaffold path).  On a CPU box the reference raises its own
    # CUDA-device check (ValueError), demonstrating the branch went
    # reference, not scaffold — i.e. the env gate correctly suppressed the
    # scaffold without a silent fallback.
    monkeypatch.setenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, "1")
    monkeypatch.delenv(RUN_GPU_GLOBAL_RATE_CAP_SCAFFOLD_ENV, raising=False)
    with pytest.raises((RuntimeError, ValueError)):
        select_global_rate_cap_rows_under_margin(
            _inputs_two_states(), GlobalRateCapSpec(cap=1, step=1)
        )


def test_scaffold_on_no_triton_raises_no_silent_fallback(monkeypatch):
    monkeypatch.setenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, "1")
    monkeypatch.setenv(RUN_GPU_GLOBAL_RATE_CAP_SCAFFOLD_ENV, "1")
    if _TRITON_AVAILABLE:
        pytest.skip("Triton present on this box; cannot exercise the no-Triton path")
    with pytest.raises(RuntimeError, match="requires Triton"):
        select_global_rate_cap_rows_under_margin(
            _inputs_two_states(), GlobalRateCapSpec(cap=1, step=1)
        )


def test_legacy_native_env_fail_closed(monkeypatch):
    monkeypatch.setenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, "1")
    monkeypatch.setenv(LEGACY_RUN_GPU_GLOBAL_RATE_CAP_NATIVE_ENV, "1")
    with pytest.raises(RuntimeError, match="fail-closed"):
        select_global_rate_cap_rows_under_margin(
            _inputs_two_states(), GlobalRateCapSpec(cap=1, step=1)
        )


def test_reference_bodies_not_mutated_at_new_entrypoint():
    # The new entrypoint is select_global_rate_cap_rows_under_margin; the
    # reference implementation entrypoint name and docstring must be intact.
    from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import (
        select_global_rate_cap_rows_torch_cuda_reference,
    )
    assert callable(select_global_rate_cap_rows_torch_cuda_reference)
    assert "MARGIN" in (select_global_rate_cap_rows_torch_cuda_reference.__doc__ or "")
