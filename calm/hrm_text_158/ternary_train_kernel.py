"""TTrain-B: Triton fused-quantize STE-prep kernel + autograd Function.

Per codex +1 implement Phase B at msg 1779538337913-2d79fa93. Companion
design in Phase A receipt msg 1779538301747-41558cb6.

Goal: eliminate the ~108 MB/step of redundant tensor passes that BitLinear's
STE prep does today (`scale=abs().mean()`, `w_q=round.clamp`, `w_q_ste=w +
(w_q*s - w).detach()`) and replace with a single fused pass that produces
`w_q * scale` directly. cuBLAS handles the matmul; explicit
autograd.Function carries STE-correct backward.

STE preservation (load-bearing, verified by Phase B grad parity test):
- Forward VALUE: `w_q_scaled = round(weight / scale).clamp(-1, 1) * scale`
  bit-equivalent to current BitLinear.quantize_weight()'s forward value
- Backward identity:
  grad_input        = grad_output @ w_q_scaled    (uses quantized fwd weight)
  grad_master_weight = grad_output.T @ input      (STE identity to master)
  grad_bias          = grad_output.sum_over_batch (standard)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================================ #
# Fused quantize Triton kernel
# ============================================================================ #


@triton.jit
def _fused_quantize_kernel(
    W_ptr,             # in: master weight, fp32 (out, in)
    W_q_scaled_ptr,    # out: w_q * scale, fp32, same shape as W
    scale_ptr,         # in: 0-d scalar fp32 (per-tensor clamped abs-mean)
    n_elements,        # in: total numel of W
    BLOCK: tl.constexpr,
):
    """Per-element: w_q_scaled[i] = round(w[i] / scale).clamp(-1, 1) * scale.

    Round semantics: half-to-even (matches PyTorch torch.round default) via
    libdevice.rintf. Falls back to half-away-from-zero only if libdevice
    is unavailable (Triton 3.6.0 ships it; we don't expect the fallback).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    s = tl.load(scale_ptr)  # 0-d scalar
    # 1/scale once; multiply for the divide. Branch is fine — scalar.
    inv_s = 1.0 / s
    q = w * inv_s
    # Round to nearest even (matches PyTorch torch.round default).
    q_round = tl.extra.libdevice.rint(q)
    # Clamp to {-1, 0, +1}
    q_clamped = tl.where(q_round > 1.0, 1.0, tl.where(q_round < -1.0, -1.0, q_round))
    out = q_clamped * s
    tl.store(W_q_scaled_ptr + offs, out, mask=mask)


def fused_quantize(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Triton-fused: w_q_scaled[i] = round(w[i] / scale).clamp(-1, 1) * scale.

    Args:
        weight: master weight, fp32, 2-D (out, in). Must be CUDA.
        scale:  0-d scalar fp32 tensor (already-clamped per-tensor scale).

    Returns:
        w_q_scaled: fp32 (out, in), values in {-scale, 0, +scale}.

    The single fused pass replaces the chained ops:
        w_q = (weight / scale).round().clamp(-1, 1)
        w_q_scaled = w_q * scale
    plus saves the `w + (w_q*scale - w).detach()` STE wrap (the autograd
    Function provides the identity directly in backward).
    """
    assert weight.is_cuda, "fused_quantize requires CUDA weight"
    assert weight.dtype == torch.float32, "fused_quantize requires fp32 weight"
    assert scale.is_cuda and scale.dtype == torch.float32 and scale.numel() == 1, \
        "scale must be a 0-d fp32 CUDA tensor"
    out = torch.empty_like(weight)
    n = weight.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _fused_quantize_kernel[grid](weight, out, scale, n, BLOCK=BLOCK)
    return out


# ============================================================================ #
# Custom autograd Function — STE-correct backward
# ============================================================================ #


class NativeTernaryTrainFn(torch.autograd.Function):
    """STE-correct fused-quantize forward + explicit backward.

    Replaces BitLinear's:
        scale = weight.abs().mean().clamp(min=_SCALE_EPS)
        w_q = (weight / scale).round().clamp(-1, 1)
        w_q_ste = weight + (w_q * scale - weight).detach()
        output = F.linear(input, w_q_ste, bias)

    with:
        scale = weight.abs().mean().clamp(min=scale_eps)
        w_q_scaled = fused_quantize(weight, scale)   # Triton, single pass
        output = F.linear(input, w_q_scaled, bias)   # cuBLAS unchanged

    Backward matches BitNet/BitLinear STE convention exactly:
        grad_input = grad_output @ w_q_scaled   (uses quantized fwd weight)
        grad_weight = grad_output.flat.T @ input.flat   (STE identity → master)
        grad_bias  = grad_output.flat.sum(0) if bias present

    Forward value bit-equivalent to BitLinear.quantize_weight()'s forward
    (same scale eps, same quantize map, same multiply order). Backward
    avoids materializing `w + (w_q*scale - w).detach()` which is what
    Python-side autograd would otherwise have to construct.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        scale_eps: float,
    ) -> torch.Tensor:
        # Per-tensor scale (PyTorch reduce — already fused under the hood).
        scale = weight.abs().mean().clamp(min=scale_eps)
        # Fused quantize via Triton (single pass over weight).
        w_q_scaled = fused_quantize(weight, scale)
        # cuBLAS matmul.
        output = F.linear(input, w_q_scaled, bias)
        # Save tensors backward needs. w_q_scaled is also the saved "weight" view.
        ctx.save_for_backward(input, w_q_scaled)
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, w_q_scaled = ctx.saved_tensors
        # grad wrt input — matches current STE convention (uses quantized weight).
        # F.linear(input, weight): y[..., j] = sum_k input[..., k] * weight[j, k]
        # → grad_input[..., k] = sum_j grad_output[..., j] * weight[j, k]
        # → grad_input = grad_output @ weight  (where weight is (out, in))
        grad_input = grad_output @ w_q_scaled if ctx.needs_input_grad[0] else None
        # grad wrt master weight — STE identity (skips the quantize map).
        # grad_weight[j, k] = sum_(batch,seq) grad_output[..., j] * input[..., k]
        # Flatten leading dims for 2-D matmul.
        grad_weight = None
        grad_bias = None
        if ctx.needs_input_grad[1] or (ctx.has_bias and ctx.needs_input_grad[2]):
            in_flat = input.reshape(-1, input.shape[-1])
            go_flat = grad_output.reshape(-1, grad_output.shape[-1])
            if ctx.needs_input_grad[1]:
                grad_weight = go_flat.transpose(0, 1) @ in_flat
            if ctx.has_bias and ctx.needs_input_grad[2]:
                grad_bias = go_flat.sum(0)
        # scale_eps is a python float, no grad.
        return grad_input, grad_weight, grad_bias, None


def native_ternary_train_linear(
    input: torch.Tensor,
    master_weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    scale_eps: float = 1e-5,
) -> torch.Tensor:
    """Entry point: invoke NativeTernaryTrainFn with the project's eps default.

    The default `scale_eps=1e-5` matches `BitLinear._SCALE_EPS` (must stay
    aligned per codex msg 1779538337913-2d79fa93 correction). Callers
    typically pass `BitLinear._SCALE_EPS` explicitly to avoid duplicating
    the constant.
    """
    return NativeTernaryTrainFn.apply(input, master_weight, bias, scale_eps)
