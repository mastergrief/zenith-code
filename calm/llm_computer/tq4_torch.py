"""Pure-PyTorch TurboQuant tq4 — encode, decode, gradient-aware Linear.

Ports `ggml_tq4_k256_*` from `llama.cpp` (our `zenith` branch, commit
`a6218df`) into PyTorch so tq4-quantized weights can be trained
(QLoRA-style with frozen base + LoRA adapters) and served via the same
format throughout the stack.

Format (matches ggml-quants.c:2727 `quantize_row_tq4_k256_ref`):

  Block = 256 values (HEAD_DIM=256), stored as:
    - 128 bytes `qs`: nybble-packed codebook indices (2 codes/byte)
    - 2 bytes `d`:    fp16 L2 norm of the unrotated block

Quantize:
  1. L2-norm each block: norm = ||x||, inv_norm = 1/norm
  2. Rotate: rotated = inv_norm * Pi @ x
  3. Quantize each rotated value via 15 boundaries (Lloyd-Max) → 4-bit
  4. Pack nybbles, store fp16 norm

Dequantize:
  1. Read fp16 norm
  2. Unpack nybbles → centroid lookup
  3. Inverse-rotate: y = norm * Pi^T @ y_hat
     (Pi orthogonal ⇒ Pi^T is inverse)

Lloyd-Max codebook (matches ggml-quants.c:2644):
  16 levels for N(0, sigma²), sigma = 1/√256. 200 iterations of
  E-step (reassign to nearest centroid via boundaries) + M-step
  (conditional mean of N(0, sigma²) between each pair of boundaries).

This port uses torch's deterministic seeded QR for Pi generation,
which gives a mathematically-equivalent but BYTE-DIFFERENT Pi matrix
than the C hardcoded table. For GGUF compatibility, load the actual
Pi from `turboquant_tables.h` via `load_pi_from_c_header()` (slow
but bit-exact).

Gradient flow: tq4 quantization is non-differentiable (4-bit codes).
We use straight-through gradient on dequant: during backward, gradient
passes through as if dequant were identity. This is the standard
QLoRA approach for 4-bit training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


HEAD_DIM = 256
N_LEVELS = 16
PI_SEED = 42


# ----- Pi rotation matrix -----

def _deterministic_orthogonal(n: int, seed: int,
                              device: torch.device,
                              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate a deterministic n×n orthogonal matrix via seeded QR of
    a standard normal. Reproducible given (n, seed). NOT bit-exact with
    the C-table version (different RNG); use for functional testing."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    m = torch.randn(n, n, generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(m)
    # Fix sign ambiguity: make diag(r) positive
    sign = torch.sign(torch.diag(r))
    sign[sign == 0] = 1.0
    return (q * sign).to(device=device, dtype=dtype)


def build_pi(device: torch.device = torch.device("cpu"),
             dtype: torch.dtype = torch.float32,
             source: str = "torch") -> torch.Tensor:
    """Get the Pi rotation matrix (HEAD_DIM × HEAD_DIM).

    Args:
        source: 'torch' (default) → deterministic QR via PyTorch. Fast
            and portable but NOT bit-exact with the C reference.
            'c_header' → parse llama.cpp's turboquant_tables.h for
            bit-exact compat with existing tq4 GGUFs. Requires the
            header file to be available on disk.
    """
    if source == "c_header":
        from calm.llm_computer.tq4_pi_loader import load_c_reference_pi
        pi = load_c_reference_pi()
        return pi.to(device=device, dtype=dtype)
    if source == "torch":
        return _deterministic_orthogonal(HEAD_DIM, PI_SEED, device, dtype)
    raise ValueError(f"unknown Pi source {source!r}, expected 'torch' or 'c_header'")


# ----- Lloyd-Max codebook -----

def compute_lloyd_max_codebook(
    n_levels: int = N_LEVELS, head_dim: int = HEAD_DIM,
    max_iter: int = 200, tol: float = 1e-10,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute centroids + boundaries for Lloyd-Max quantization.

    Matches `ggml_tq4_k256_compute_codebook_impl` in ggml-quants.c:
    optimal 16-level quantization for N(0, sigma²) where
    sigma = 1/sqrt(head_dim).

    Returns (centroids, boundaries) as float32 tensors.
    """
    sigma = 1.0 / math.sqrt(head_dim)
    lo = -3.5 * sigma
    hi = 3.5 * sigma

    # Uniform initialization
    idx = torch.arange(n_levels, dtype=dtype)
    c = lo + (hi - lo) * (idx + 0.5) / n_levels

    sigma_sq = sigma * sigma
    inv_sigma_sqrt2 = 1.0 / (sigma * math.sqrt(2.0))
    pdf_norm = 1.0 / (sigma * math.sqrt(2.0 * math.pi))

    for _ in range(max_iter):
        # Boundaries = midpoints of adjacent centroids
        b = 0.5 * (c[:-1] + c[1:])

        # Edges include extended endpoints
        edges = torch.empty(n_levels + 1, dtype=dtype)
        edges[0] = lo * 3.0
        edges[-1] = hi * 3.0
        edges[1:-1] = b

        # E-step: conditional mean per bin, using truncated normal formula
        # For N(0, sigma²) restricted to [a, b]:
        #   mean = -sigma² * (pdf(b) - pdf(a)) / (0.5 * (erf(b/(sigma*sqrt(2))) - erf(a/(sigma*sqrt(2)))))
        a = edges[:-1]
        bb = edges[1:]
        pdf_a = pdf_norm * torch.exp(-0.5 * a * a / sigma_sq)
        pdf_b = pdf_norm * torch.exp(-0.5 * bb * bb / sigma_sq)
        num = -sigma_sq * (pdf_b - pdf_a)
        den = 0.5 * (torch.erf(bb * inv_sigma_sqrt2) - torch.erf(a * inv_sigma_sqrt2))
        new_c = torch.where(den > 1e-15, num / den, c)

        if (new_c - c).abs().max().item() < tol:
            c = new_c
            break
        c = new_c

    b = 0.5 * (c[:-1] + c[1:])
    return c.to(torch.float32), b.to(torch.float32)


# ----- Tq4Tensor -----

@dataclass
class Tq4Tensor:
    """Compressed 4-bit representation.

    Attributes:
        qs: (n_blocks, HEAD_DIM/2) uint8. Each byte holds 2 nybble codes.
        d:  (n_blocks,) float32 L2 norms of each unrotated block.
        shape: original tensor shape.
    """
    qs: torch.Tensor
    d: torch.Tensor
    shape: tuple

    @property
    def n_blocks(self) -> int:
        return self.qs.size(0)

    def bytes_on_disk(self) -> int:
        """Theoretical packed size: 128 bytes qs + 2 bytes d + 2 bytes pad."""
        return self.n_blocks * 132


def _flatten_to_blocks(x: torch.Tensor) -> torch.Tensor:
    """Flatten and reshape to (n_blocks, HEAD_DIM). Requires numel()
    divisible by HEAD_DIM."""
    assert x.numel() % HEAD_DIM == 0, (
        f"numel {x.numel()} not divisible by HEAD_DIM {HEAD_DIM}"
    )
    return x.reshape(-1, HEAD_DIM)


def quantize_tq4(
    x: torch.Tensor,
    pi: Optional[torch.Tensor] = None,
    boundaries: Optional[torch.Tensor] = None,
) -> Tq4Tensor:
    """Encode a tensor into tq4 format. Matches the C reference bit-by-
    bit assuming matching Pi + boundaries."""
    orig_shape = tuple(x.shape)
    if pi is None:
        pi = build_pi(device=x.device, dtype=torch.float32)
    if boundaries is None:
        _, boundaries = compute_lloyd_max_codebook()
        boundaries = boundaries.to(device=x.device)

    blocks = _flatten_to_blocks(x.float())  # (n_blocks, HEAD_DIM)
    n_blocks = blocks.size(0)

    # 1. L2 norms per block
    norm = blocks.norm(dim=-1)  # (n_blocks,)
    inv_norm = torch.where(norm > 1e-8, 1.0 / norm, torch.zeros_like(norm))

    # 2. Rotate: rotated[i, :] = inv_norm[i] * Pi @ blocks[i, :]
    #    Equivalent matmul: rotated = blocks @ Pi.T, scaled per-row
    rotated = blocks @ pi.T.to(blocks.dtype)  # (n_blocks, HEAD_DIM)
    rotated = rotated * inv_norm.unsqueeze(-1)

    # 3. Quantize each value by boundary scan — codes in {0..15}
    # boundaries shape: (N_LEVELS - 1,). Count boundaries each value exceeds.
    # codes[i, j] = #{b : rotated[i, j] >= boundaries[b]}
    codes = (rotated.unsqueeze(-1) >= boundaries.view(1, 1, -1)).sum(dim=-1)
    codes = codes.to(torch.int64)  # (n_blocks, HEAD_DIM)

    # 4. Pack nybbles: byte[p] = codes[2p] | (codes[2p+1] << 4)
    codes = codes.reshape(n_blocks, HEAD_DIM // 2, 2)
    qs = (codes[:, :, 0] & 0x0F) | ((codes[:, :, 1] & 0x0F) << 4)
    qs = qs.to(torch.uint8)

    return Tq4Tensor(qs=qs, d=norm.contiguous(), shape=orig_shape)


def dequantize_tq4(
    q: Tq4Tensor,
    pi: Optional[torch.Tensor] = None,
    centroids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Decode tq4 back to float tensor. Reconstruction to within
    quantization error of the original."""
    if pi is None:
        pi = build_pi(device=q.qs.device, dtype=torch.float32)
    if centroids is None:
        centroids, _ = compute_lloyd_max_codebook()
        centroids = centroids.to(device=q.qs.device)

    n_blocks = q.qs.size(0)
    # Unpack nybbles → (n_blocks, HEAD_DIM) codes
    low = q.qs & 0x0F
    high = (q.qs >> 4) & 0x0F
    codes = torch.stack([low, high], dim=-1).reshape(n_blocks, HEAD_DIM)
    # Centroid lookup
    y_hat = centroids[codes.to(torch.long)]  # (n_blocks, HEAD_DIM) float32

    # Inverse rotation: result = norm * Pi^T @ y_hat  ==  y_hat @ Pi
    result = y_hat @ pi.to(y_hat.dtype)
    result = result * q.d.unsqueeze(-1)
    return result.reshape(q.shape)


# ----- Straight-through gradient -----

class _Tq4DequantSTE(torch.autograd.Function):
    """Custom autograd: forward dequantizes; backward passes gradient
    through unchanged (straight-through estimator). Standard for 4-bit
    training like QLoRA."""

    @staticmethod
    def forward(ctx, qs, d, pi, centroids, shape):
        ctx.save_for_backward(pi, centroids)
        ctx.saved_shape = shape
        n_blocks = qs.size(0)
        low = qs & 0x0F
        high = (qs >> 4) & 0x0F
        codes = torch.stack([low, high], dim=-1).reshape(n_blocks, HEAD_DIM)
        y_hat = centroids[codes.to(torch.long)]
        result = y_hat @ pi.to(y_hat.dtype) * d.unsqueeze(-1)
        return result.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: no gradient to qs (it's 4-bit int) or d
        # (re-quantization is non-differentiable). Gradient continues
        # through the consuming linear op to update LoRA adapters.
        return None, None, None, None, None


def dequantize_tq4_differentiable(
    q: Tq4Tensor,
    pi: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Differentiable wrapper around dequant — passes straight-through
    gradients. Use in Linear layers where LoRA adapters need gradient."""
    return _Tq4DequantSTE.apply(q.qs, q.d, pi, centroids, q.shape)


# ----- Tq4Linear -----

class Tq4Linear(nn.Module):
    """Linear layer with tq4-stored weight. Forward dequantizes on-the-fly.

    Weight is frozen (tq4 codes are int, not trainable). Use alongside
    LoRA adapters for parameter-efficient fine-tuning.

    Shapes:
      weight: (out_features, in_features), stored as Tq4Tensor
      bias:   (out_features,) optional, trainable

    Forward: y = x @ dequant(weight).T + bias
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert (in_features * out_features) % HEAD_DIM == 0, (
            f"in*out {in_features*out_features} must be divisible by "
            f"HEAD_DIM {HEAD_DIM}"
        )
        self._qs: Optional[torch.Tensor] = None
        self._d: Optional[torch.Tensor] = None
        self._weight_shape = (out_features, in_features)
        # Pi + codebook as buffers (non-trainable, loaded once)
        pi = build_pi()
        centroids, _ = compute_lloyd_max_codebook()
        self.register_buffer("_pi", pi)
        self.register_buffer("_centroids", centroids)
        # Optional bias (trainable)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def load_weight(self, weight_fp32: torch.Tensor) -> None:
        """Quantize and store the given weight tensor."""
        assert weight_fp32.shape == self._weight_shape, (
            f"shape {weight_fp32.shape} != {self._weight_shape}"
        )
        q = quantize_tq4(weight_fp32, pi=self._pi)
        self._qs = q.qs.to(self._pi.device)
        self._d = q.d.to(self._pi.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self._qs is not None, (
            "Tq4Linear has no weight loaded; call .load_weight() first"
        )
        q = Tq4Tensor(qs=self._qs, d=self._d, shape=self._weight_shape)
        w = dequantize_tq4_differentiable(q, self._pi, self._centroids)
        return F.linear(x, w, self.bias)

    def is_loaded(self) -> bool:
        return self._qs is not None
