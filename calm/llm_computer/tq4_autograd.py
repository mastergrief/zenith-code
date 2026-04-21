"""tq4 autograd wrapper — Triton forward + materialized-W backward.

Enables training-mode use of the fast Triton tq4 kernel. The raw
Triton kernel (`tq4_linear_triton`) uses `tl.store` to write outputs
and does not carry an autograd graph; calling it on a grad-requiring
input silently breaks backward.

This module provides `Tq4TritonAutogradFunction`, a subclass of
`torch.autograd.Function` that:

  forward  — runs `tq4_linear_triton` (fast, no materialized W)
  backward — rematerializes W_math from (qs, d, centroids) on demand,
             computes grad_x_rot = grad_y @ W_math, then un-rotates
             by Pi (grad_x = grad_x_rot @ Pi).

Weights are frozen so only x carries gradient; qs/d/pi/centroids pass
through as None.

Installation pattern (monkey-patch MmapTq4Linear.__call__ for the
training session, restore in finally):

    from calm.llm_computer.tq4_autograd import (
        install_tq4_autograd, restore_tq4_autograd,
    )
    install_tq4_autograd()
    try:
        train(...)
    finally:
        restore_tq4_autograd()

Math equivalence with the existing PyTorch fast path:

  y = (x @ Pi.T) @ W_math.T
  grad_y has shape (..., out_features). Autograd wants grad_x shape
  matching x = (..., in_features). The chain rule gives:

    grad_x_rot = grad_y @ W_math        # (..., in_features)
    grad_x     = grad_x_rot @ Pi        # undo Pi.T from forward

  `@ Pi` recovers the un-rotated gradient because Pi is orthogonal
  (Pi.T @ Pi = I), so `(x @ Pi.T) @ Pi = x`.
"""

from __future__ import annotations

import torch


class Tq4TritonAutogradFunction(torch.autograd.Function):
    """Triton tq4 linear with custom backward.

    Forward streams tq4 bytes through the Triton matvec/matmul kernel
    without materializing W. Backward materializes W from
    (centroids[codes] * d) and does a standard matmul + Pi unrotation.

    `@custom_fwd`/`@custom_bwd` make the function autocast-aware:
    under `torch.amp.autocast(bfloat16)`, `x` arrives as bf16; we
    cast it to fp32 for the Triton kernel (which uses fp32
    internally), and the returned grad is cast back to x's original
    dtype automatically by the decorator.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, x, qs, d, pi, centroids, out_features, in_features):
        from calm.llm_computer.tq4_triton import tq4_linear_triton

        y = tq4_linear_triton(
            x, qs, d, pi, centroids, out_features, in_features
        )
        ctx.save_for_backward(qs, d, pi, centroids)
        ctx.out_features = out_features
        ctx.in_features = in_features
        return y

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_y):
        """Uses Triton backward kernel to match Triton forward's exact
        reduction order — otherwise 6e-5 per-linear forward/backward
        mismatch compounds to 50° gradient direction error per Gemma
        layer and training diverges. Using PyTorch-materialized W for
        backward was measured at cosine=0.60 per layer (see
        tq4_triton.py _tq4_backward_kernel docstring)."""
        from calm.llm_computer.tq4_triton import tq4_backward_triton

        qs, d, pi, centroids = ctx.saved_tensors
        in_f = ctx.in_features
        out_f = ctx.out_features

        grad_x_rot = tq4_backward_triton(
            grad_y, qs, d, centroids, out_f, in_f
        )

        # Undo Pi.T rotation: x_rot = x @ Pi.T  ⇒  grad_x = grad_x_rot @ Pi
        *batch, _ = grad_x_rot.shape
        bpr = in_f // 256
        grad_x = (grad_x_rot.reshape(*batch, bpr, 256) @ pi).reshape(
            *batch, in_f
        )

        return grad_x, None, None, None, None, None, None


def _autograd_call(self, x):
    """Replacement for `MmapTq4Linear.__call__` that routes through
    `Tq4TritonAutogradFunction.apply(...)`. Same dispatch contract as
    the original (requires GPU preload of qs/d/pi/centroids)."""
    from calm.llm_computer.gemma_substrate import MmapTq4Linear

    assert self._gpu_qs is not None, (
        "Tq4TritonAutograd requires GPU-preloaded tq4 weights"
    )
    assert MmapTq4Linear._shared_pi is not None, (
        "Tq4TritonAutograd requires MmapTq4Linear._shared_pi initialized"
    )
    return Tq4TritonAutogradFunction.apply(
        x,
        self._gpu_qs,
        self._gpu_d,
        MmapTq4Linear._shared_pi,
        MmapTq4Linear._shared_centroids,
        self.out_features,
        self.in_features,
    )


def install_tq4_autograd():
    """Monkey-patch MmapTq4Linear.__call__ to the autograd-safe Triton
    wrapper. Idempotent; stores the original for restore."""
    from calm.llm_computer.gemma_substrate import MmapTq4Linear

    if getattr(MmapTq4Linear, "_orig_call", None) is None:
        MmapTq4Linear._orig_call = MmapTq4Linear.__call__
    MmapTq4Linear.__call__ = _autograd_call


def restore_tq4_autograd():
    """Restore the original `MmapTq4Linear.__call__` (the dispatch
    that respects `_use_triton`)."""
    from calm.llm_computer.gemma_substrate import MmapTq4Linear

    orig = getattr(MmapTq4Linear, "_orig_call", None)
    if orig is not None:
        MmapTq4Linear.__call__ = orig
        MmapTq4Linear._orig_call = None


if __name__ == "__main__":
    # Self-test: check forward matches the existing PyTorch fast path on
    # a synthetic GPU tq4 linear, and backward produces sane shapes.
    import torch as _t

    from calm.llm_computer.gemma_substrate import (
        MmapTq4Linear, _tq4_linear_kernel, enable_triton_tq4,
    )
    from calm.llm_computer.tq4_pi_loader import load_pi_and_centroids

    enable_triton_tq4(False)  # PyTorch path for comparison
    device = "cuda" if _t.cuda.is_available() else "cpu"
    assert device == "cuda", "autograd Triton self-test needs CUDA"

    # Load Pi + centroids
    pi, centroids = load_pi_and_centroids(device=device)
    MmapTq4Linear._shared_pi = pi
    MmapTq4Linear._shared_centroids = centroids

    in_f, out_f = 2560, 16384
    n_blocks = out_f * (in_f // 256)

    # Random tq4 state
    qs = _t.randint(0, 256, (n_blocks, 128), dtype=_t.uint8, device=device)
    d = _t.rand(n_blocks, device=device) * 0.1

    x = _t.randn(1, 8, in_f, device=device, requires_grad=True)

    # Reference: PyTorch fast path, captures grad via F.linear
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = _tq4_linear_kernel(x_ref, qs, d, out_f, in_f, pi, centroids)
    y_ref.sum().backward()

    # Ours: Triton forward + manual backward
    y_ours = Tq4TritonAutogradFunction.apply(
        x, qs, d, pi, centroids, out_f, in_f
    )
    y_ours.sum().backward()

    # Compare y and grad_x
    y_max_diff = (y_ours - y_ref).abs().max().item()
    grad_max_diff = (x.grad - x_ref.grad).abs().max().item()
    print(f"y max diff:    {y_max_diff:.6e}")
    print(f"grad max diff: {grad_max_diff:.6e}")
    assert y_max_diff < 1e-3, "forward mismatch"
    assert grad_max_diff < 1e-3, "backward mismatch"
    print("Tq4TritonAutogradFunction self-test PASS")
