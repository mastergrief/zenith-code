"""GPU parity tests for BR-3C-H credit-axis kernel (H.2 gated — skip without CUDA)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    authoritative_forward_context,
    default_dry_run_rank_vote_spec,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.integer_credit_axis_gpu_kernel import (
    _launch_s1_attribution_triton,
    _boundary_quantize_captures,
    credit_axis_kernelized_sparse_pipeline_cuda,
    credit_axis_kernelized_sparse_pipeline_cuda_torch_sort_s4,
)
from calm.hrm_text_158.native_full_stack.integer_credit_axis_gpu_receipt import (
    RUN_GPU_CREDIT_AXIS_KERNEL_ENV,
    build_cpu_oracle_payload_hashes_for_gpu_parity,
    cpu_oracle_payload_hashes_from_gpu_parity,
    pipeline_result_to_live_gpu_tensors,
)
from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
    INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V1,
    streaming_sparse_attribution_from_captures,
)
from calm.hrm_text_158.native_full_stack.integer_native_optimizer_credit_path_design import (
    prove_integer_credit_axis_integration,
)
from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (
    canonical_rank_vote_spec,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU parity tests require CUDA (H.2 launch packet)",
)


class _Tiny(torch.nn.Module):
    def __init__(self, in_features: int = 3, out_features: int = 2) -> None:
        super().__init__()
        self.proj = BitLinear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _integration_and_bins():
    torch.manual_seed(158)
    model = _Tiny()
    with torch.no_grad():
        model.proj.weight.zero_()
    eligible = {"proj": model.proj}
    q = torch.zeros_like(model.proj.weight.detach(), dtype=torch.int8)
    state = make_bounded_tensor_state(
        "proj",
        q,
        torch.tensor(1.0, dtype=torch.float32),
        hot_exact_indices=tuple(range(int(q.numel()))),
    )
    x = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)
    target = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    model.zero_grad(set_to_none=True)
    with authoritative_forward_context(
        eligible,
        {"proj": state},
        device="cpu",
        requires_grad=True,
    ) as handle:
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
    capture = handle.captures["proj"]
    weight_shape = tuple(int(dim) for dim in state.q_levels.shape)
    q_levels_flat = state.q_levels.reshape(-1)
    integration = prove_integer_credit_axis_integration(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=weight_shape,
        q_levels_flat=q_levels_flat,
        rank_spec=default_dry_run_rank_vote_spec(),
        comparable_set_id="br3c_h1b_gpu_parity",
        reference_oracle_run_id="oracle_br3c_h1b",
        candidate_run_id="candidate_br3c_h1b",
    )
    bins = canonical_rank_vote_spec(default_dry_run_rank_vote_spec())
    return integration, capture, weight_shape, q_levels_flat, bins


def _bxs_captures(*, batch: int = 2, seq: int = 3, in_features: int = 8, out_features: int = 4):
    torch.manual_seed(1581)
    inputs = (torch.randn(batch, seq, in_features),)
    grad_outputs = (torch.randn(batch, seq, out_features),)
    return inputs, grad_outputs, (out_features, in_features)


def test_s1_triton_launch_matches_cpu_einsum_bxS(monkeypatch) -> None:
    monkeypatch.setenv(RUN_GPU_CREDIT_AXIS_KERNEL_ENV, "1")
    inputs, grad_outputs, shape = _bxs_captures(batch=2, seq=3)
    out_features, in_features = shape
    input_q15_list, grad_q16_list, _ = _boundary_quantize_captures(
        inputs,
        grad_outputs,
        device=torch.device("cuda"),
    )
    gpu_flat, gpu_attr, s1_native = _launch_s1_attribution_triton(
        input_q15_list,
        grad_q16_list,
        out_features=out_features,
        in_features=in_features,
        law_id=INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V1,
    )
    assert s1_native is True
    cpu_events, _ = streaming_sparse_attribution_from_captures(
        inputs,
        grad_outputs,
        weight_shape=shape,
    )
    assert torch.equal(gpu_flat.cpu(), cpu_events.flat_indices)
    assert torch.equal(gpu_attr.cpu(), cpu_events.attribution_q31)


def test_default_path_sets_all_stage_native_true(monkeypatch) -> None:
    integration, capture, weight_shape, q_levels_flat, bins = _integration_and_bins()
    monkeypatch.setenv(RUN_GPU_CREDIT_AXIS_KERNEL_ENV, "1")
    result = credit_axis_kernelized_sparse_pipeline_cuda(
        capture_inputs=capture["inputs"],
        capture_grad_outputs=capture["grad_outputs"],
        weight_shape=weight_shape,
        q_levels_flat=q_levels_flat,
        rank_bin_spec_canonical=bins,
        credit_law_id="credit_neg_attribution_q31_v1",
    )
    assert result.stage_native_evidence is not None
    assert result.stage_native_evidence.whole_pipeline_native is True
    assert result.torch_cuda_reference_only is False


def test_gpu_pipeline_native_s4_sets_torch_cuda_reference_only_false(monkeypatch) -> None:
    integration, capture, weight_shape, q_levels_flat, bins = _integration_and_bins()
    monkeypatch.setenv(RUN_GPU_CREDIT_AXIS_KERNEL_ENV, "1")
    result = credit_axis_kernelized_sparse_pipeline_cuda(
        capture_inputs=capture["inputs"],
        capture_grad_outputs=capture["grad_outputs"],
        weight_shape=weight_shape,
        q_levels_flat=q_levels_flat,
        rank_bin_spec_canonical=bins,
        credit_law_id="credit_neg_attribution_q31_v1",
    )
    assert result.torch_cuda_reference_only is False


def test_torch_sort_s4_path_sets_reference_only(monkeypatch) -> None:
    integration, capture, weight_shape, q_levels_flat, bins = _integration_and_bins()
    monkeypatch.setenv(RUN_GPU_CREDIT_AXIS_KERNEL_ENV, "1")
    result = credit_axis_kernelized_sparse_pipeline_cuda_torch_sort_s4(
        capture_inputs=capture["inputs"],
        capture_grad_outputs=capture["grad_outputs"],
        weight_shape=weight_shape,
        q_levels_flat=q_levels_flat,
        rank_bin_spec_canonical=bins,
        credit_law_id="credit_neg_attribution_q31_v1",
    )
    assert result.torch_cuda_reference_only is True
    assert result.stage_native_evidence is not None
    assert result.stage_native_evidence.s4_native is False


def test_gpu_five_key_oracle_payload_hashes_match_cpu(monkeypatch) -> None:
    integration, capture, weight_shape, q_levels_flat, bins = _integration_and_bins()
    monkeypatch.setenv(RUN_GPU_CREDIT_AXIS_KERNEL_ENV, "1")
    result = credit_axis_kernelized_sparse_pipeline_cuda(
        capture_inputs=capture["inputs"],
        capture_grad_outputs=capture["grad_outputs"],
        weight_shape=weight_shape,
        q_levels_flat=q_levels_flat,
        rank_bin_spec_canonical=bins,
        credit_law_id="credit_neg_attribution_q31_v1",
    )
    oracle_5 = build_cpu_oracle_payload_hashes_for_gpu_parity(
        integration_receipt=integration,
        credit_q31=integration.bound_credit_q31,
        projected_moves=integration.bound_projected_moves,
        projected_move_indices=integration.bound_projected_move_indices,
        rank_bin_spec_canonical=bins,
    )
    live = pipeline_result_to_live_gpu_tensors(result)
    from calm.hrm_text_158.native_full_stack.integer_credit_axis_gpu_receipt import (
        _recompute_gpu_payload_hashes_from_live_tensors,
    )

    gpu_hashes = _recompute_gpu_payload_hashes_from_live_tensors(live)
    cpu_hashes = cpu_oracle_payload_hashes_from_gpu_parity(oracle_5)
    for key in cpu_hashes:
        assert gpu_hashes[key] == cpu_hashes[key], key


def test_gpu_five_key_oracle_bxS_capture(monkeypatch) -> None:
    torch.manual_seed(1599)
    inputs, grad_outputs, shape = _bxs_captures(batch=2, seq=3, in_features=4, out_features=2)
    q_levels_flat = torch.zeros(shape[0] * shape[1], dtype=torch.int8)
    bins = canonical_rank_vote_spec(default_dry_run_rank_vote_spec())
    integration = prove_integer_credit_axis_integration(
        inputs,
        grad_outputs,
        weight_shape=shape,
        q_levels_flat=q_levels_flat,
        rank_spec=default_dry_run_rank_vote_spec(),
        comparable_set_id="br3c_h1b_bxs",
        reference_oracle_run_id="oracle_bxs",
        candidate_run_id="candidate_bxs",
    )
    monkeypatch.setenv(RUN_GPU_CREDIT_AXIS_KERNEL_ENV, "1")
    result = credit_axis_kernelized_sparse_pipeline_cuda(
        capture_inputs=inputs,
        capture_grad_outputs=grad_outputs,
        weight_shape=shape,
        q_levels_flat=q_levels_flat,
        rank_bin_spec_canonical=bins,
        credit_law_id="credit_neg_attribution_q31_v1",
    )
    assert result.stage_native_evidence is not None
    assert result.stage_native_evidence.s1_native is True
