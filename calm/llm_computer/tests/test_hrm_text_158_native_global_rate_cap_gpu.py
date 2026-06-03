"""Phase-1 L2-A MARGIN-only GPU global-rate-cap reference tests."""
from __future__ import annotations

import os
from pathlib import Path
import time

import pytest
import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapOrderingMode,
    GlobalRateCapRow,
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    apply_global_rate_cap_reference,
    select_global_rate_cap_rows,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import (
    DEFAULT_GLOBAL_RATE_CAP_GPU_ARTIFACT_PATH,
    GLOBAL_RATE_CAP_GPU_ARTIFACT_ENV,
    GLOBAL_RATE_CAP_TORCH_CUDA_REFERENCE_SCOPE,
    RUN_GPU_GLOBAL_RATE_CAP_ENV,
    apply_global_rate_cap_torch_cuda_reference_under_margin,
    select_global_rate_cap_rows_torch_cuda_reference,
    write_global_rate_cap_gpu_receipt_artifact,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    RUN_GPU_Q_ACC_APPLY_ENV,
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
)


GPU_GLOBAL_RATE_CAP_SELECTION = pytest.mark.skipif(
    os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_ENV) != "1" or not torch.cuda.is_available(),
    reason=(
        "global cap CUDA selection receipt deferred; set "
        "HRM_TEXT_158_RUN_GPU_GLOBAL_RATE_CAP=1 only inside a granted gpu:0 lane"
    ),
)
GPU_GLOBAL_RATE_CAP_APPLY = pytest.mark.skipif(
    os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_ENV) != "1"
    or os.environ.get(RUN_GPU_Q_ACC_APPLY_ENV) != "1"
    or not torch.cuda.is_available(),
    reason=(
        "global cap CUDA apply-chain receipt deferred; set global-cap and q/acc "
        "env gates only inside a granted gpu:0 lane"
    ),
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


def _state(
    q: list[int] | torch.Tensor,
    acc: list[int] | torch.Tensor,
    *,
    device: str,
) -> VoteUpdateState:
    return VoteUpdateState(
        q_levels=torch.as_tensor(q, dtype=torch.int8, device=device),
        accumulators=torch.as_tensor(acc, dtype=torch.int16, device=device),
    )


def _inputs(
    votes: list[int] | torch.Tensor,
    *,
    device: str,
    **kwargs,
) -> VoteUpdateInputs:
    converted = {}
    for name, value in kwargs.items():
        if value is None:
            converted[name] = None
        elif name.endswith("moves"):
            converted[name] = torch.as_tensor(value, dtype=torch.int8, device=device)
        else:
            converted[name] = torch.as_tensor(value, dtype=torch.int16, device=device)
    return VoteUpdateInputs(
        votes=torch.as_tensor(votes, dtype=torch.int16, device=device),
        **converted,
    )


def _tensor_input(
    state_key: str,
    q: list[int] | torch.Tensor,
    acc: list[int] | torch.Tensor,
    votes: list[int] | torch.Tensor,
    *,
    device: str,
    spec: VoteUpdateSpec | None = None,
    **vote_kwargs,
) -> GlobalRateCapTensorInput:
    state = _state(q, acc, device=device)
    inputs = _inputs(votes, device=device, **vote_kwargs)
    plan = plan_integer_vote_update_reference(state, inputs, spec or _spec())
    return GlobalRateCapTensorInput(state_key=state_key, state=state, plan=plan)


def _cross_state_tie_inputs(device: str) -> list[GlobalRateCapTensorInput]:
    return [
        _tensor_input(
            "proj_in",
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 30, 30],
            device=device,
        ),
        _tensor_input(
            "proj_out",
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [30, 30, 0, 0],
            device=device,
        ),
    ]


def _row_tuple(row: GlobalRateCapRow) -> tuple[str, int, int, int, int, int]:
    return (
        row.state_key,
        int(row.flat_index),
        int(row.local_pos),
        int(row.global_flat_index),
        int(row.abs_new_acc),
        int(row.threshold_abs),
    )


def test_gpu_global_cap_default_off_before_lane(monkeypatch):
    monkeypatch.delenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, raising=False)
    with pytest.raises(RuntimeError, match=RUN_GPU_GLOBAL_RATE_CAP_ENV):
        select_global_rate_cap_rows_torch_cuda_reference(
            [],
            GlobalRateCapSpec(cap=1, step=1),
        )


def test_gpu_global_cap_rejects_unsupported_ordering_modes_without_policy_science(monkeypatch):
    monkeypatch.setenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, "1")
    for mode in (
        GlobalRateCapOrderingMode.HASH_SHUFFLE,
        GlobalRateCapOrderingMode.ROUND_ROBIN,
    ):
        with pytest.raises(NotImplementedError, match="MARGIN ordering only"):
            select_global_rate_cap_rows_torch_cuda_reference(
                [],
                GlobalRateCapSpec(cap=1, step=1, ordering_mode=mode),
            )


@GPU_GLOBAL_RATE_CAP_SELECTION
def test_cuda_margin_selection_matches_ordered_cpu_rows_cross_state_tie():
    spec = GlobalRateCapSpec(cap=3, step=7, ordering_mode=GlobalRateCapOrderingMode.MARGIN)
    cpu_inputs = _cross_state_tie_inputs("cpu")
    cuda_inputs = _cross_state_tie_inputs("cuda")
    cpu_rows, cpu_accepted, cpu_deferred = select_global_rate_cap_rows(cpu_inputs, spec)

    result = select_global_rate_cap_rows_torch_cuda_reference(cuda_inputs, spec)

    assert result.scope == GLOBAL_RATE_CAP_TORCH_CUDA_REFERENCE_SCOPE
    assert result.backend == "cuda"
    assert result.tensor_offsets == {"proj_in": 0, "proj_out": 4}
    assert result.ordered_rows_as_tuples() == tuple(_row_tuple(row) for row in cpu_rows)
    assert result.accepted_rows_as_tuples() == tuple(_row_tuple(row) for row in cpu_accepted)
    assert result.deferred_rows_as_tuples() == tuple(_row_tuple(row) for row in cpu_deferred)
    assert [row.global_flat_index for row in cpu_accepted] == [2, 3, 4]
    assert [row.global_flat_index for row in cpu_deferred] == [5]
    assert result.rows_by_state["proj_in"].accepted_indices.detach().cpu().tolist() == [2, 3]
    assert result.rows_by_state["proj_out"].accepted_indices.detach().cpu().tolist() == [0]
    assert result.rows_by_state["proj_out"].deferred_indices.detach().cpu().tolist() == [1]
    assert result.stats["lexicographic_stable_sort"] is True
    assert result.stats["device_row_tensors_emitted_before_cpu_telemetry"] is True
    assert result.stats["python_row_lists_materialized_before_q_acc_apply"] is False
    assert result.stats["policy_modes_rejected"] == ["hash_shuffle", "round_robin"]


@GPU_GLOBAL_RATE_CAP_APPLY
def test_cuda_apply_chain_matches_cpu_oracle_and_writes_compact_artifact(tmp_path):
    spec = GlobalRateCapSpec(cap=3, step=7, ordering_mode=GlobalRateCapOrderingMode.MARGIN)
    cpu_inputs = _cross_state_tie_inputs("cpu")
    cuda_inputs = _cross_state_tie_inputs("cuda")
    cpu_result = apply_global_rate_cap_reference(cpu_inputs, spec)

    torch.cuda.reset_peak_memory_stats()
    warm = apply_global_rate_cap_torch_cuda_reference_under_margin(cuda_inputs, spec)
    torch.cuda.synchronize()
    assert warm.selection.accepted_rows_as_tuples() == tuple(
        _row_tuple(row) for row in cpu_result.accepted_rows
    )

    start = time.perf_counter()
    result = apply_global_rate_cap_torch_cuda_reference_under_margin(cuda_inputs, spec)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - start) * 1000.0
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    cpu_by_key = {tensor.state_key: tensor for tensor in cpu_result.tensor_results}
    for tensor in result.tensor_results:
        cpu_tensor = cpu_by_key[tensor.state_key]
        assert tensor.q_levels.detach().cpu().tolist() == cpu_tensor.q_levels.tolist()
        assert tensor.accumulators.detach().cpu().tolist() == cpu_tensor.accumulators.tolist()
        assert tensor.stats["accepted_row_source"] == "device_selection_result.rows_by_state"
        assert tensor.stats["python_row_lists_materialized_before_q_acc_apply"] is False

    assert result.stats["accepted_row_source"] == "device_selection_result.rows_by_state"
    assert result.stats["python_row_lists_materialized_before_q_acc_apply"] is False
    assert result.stats["global_rate_cap_accepted_count"] == 3
    assert result.stats["global_rate_cap_deferred_count"] == 1
    assert result.stats["deferred_backlog_size"] == 1
    assert result.stats["q_changed_count"] == cpu_result.step_summary["q_changed_count"]
    assert wall_ms > 0.0
    assert peak_allocated > 0
    assert peak_reserved >= peak_allocated

    artifact_target = Path(
        os.environ.get(GLOBAL_RATE_CAP_GPU_ARTIFACT_ENV, str(tmp_path / "global_rate_cap_gpu.json"))
    )
    payload = result.compact_artifact_payload()
    payload["receipt"] = {
        "label": GLOBAL_RATE_CAP_TORCH_CUDA_REFERENCE_SCOPE,
        "default_artifact_path": DEFAULT_GLOBAL_RATE_CAP_GPU_ARTIFACT_PATH,
        "wall_ms": wall_ms,
        "peak_allocated": int(peak_allocated),
        "peak_reserved": int(peak_reserved),
        "parity_cpu_q_acc": True,
        "raw_tensor_fields_omitted": True,
    }
    written = write_global_rate_cap_gpu_receipt_artifact(payload, artifact_target)
    artifact_text = written.read_text(encoding="utf-8")
    assert written.stat().st_size < 50_000
    assert "q_levels" not in artifact_text
    assert "accumulators" not in artifact_text
    print(
        "global_rate_cap_torch_cuda_reference_margin_receipt "
        f"accepted={result.stats['global_rate_cap_accepted_count']} "
        f"deferred={result.stats['global_rate_cap_deferred_count']} "
        f"wall_ms={wall_ms:.4f} peak_allocated={peak_allocated} "
        f"peak_reserved={peak_reserved} artifact={written} "
        f"scope={GLOBAL_RATE_CAP_TORCH_CUDA_REFERENCE_SCOPE}"
    )
