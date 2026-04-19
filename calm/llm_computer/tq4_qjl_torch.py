"""TurboQuant inner-product-optimal variant — Q_mse(3 bits) + QJL(1 bit).

Implements Algorithm 2 from the TurboQuant paper (`§3.2 — Inner-Product-
Optimal TurboQuant`):

  z      = Pi @ x_unit              # rotate (Pi orthogonal, seed=42)
  z_hat  = Q_mse(z; b-1=3 bits)     # Lloyd-Max codebook for N(0, 1/d), 8 levels
  r      = z - z_hat                 # per-coord residual
  signs  = sign(S · r)               # 1-bit JL sketch, S ∈ R^(d×d) Gaussian
  store  = (codes_3bit, signs, d_mse=‖x‖, d_qjl=‖r‖)

Decode-time inner-product estimator (Algorithm 2's `<x_q, y>`):

  y_rot       = Pi @ y
  mse_term    = Σ_i centroids[codes_i] · y_rot_i
  qjl_term    = (sqrt(π/2) · d_qjl / d) · <signs, S · y_rot>   ← S·y stays real
  <x_q, y>    ≈ d_mse · (mse_term + qjl_term)

The estimator coefficient comes from Sheppard's identity for two
correlated Gaussians: E[sign(g·r)·(g·y_rot)] = sqrt(2/π)·<r,y_rot>/‖r‖
for g ~ N(0, I_d). Summing m=d row contributions and inverting gives
<r,y_rot> = ‖r‖·sqrt(π/2)/d · <sign(S·r), S·y_rot>. Scaling by d_mse
recovers the un-normalized <x, y>.

Unbiased: E[<x_q, y>] = <x, y> for any y when S is i.i.d. Gaussian.

Block format (132 bytes total — matches tq4_k256 for memory parity):

  uint8  qs_3bit[96]    — 256 codes × 3 bits, packed 8 codes per 3 bytes
  uint8  qjl_signs[32]  — 256 sign bits, 8 per byte
  fp16   d_mse          — block L2 norm
  fp16   d_qjl          — residual L2 norm

Compared with `tq4_torch.py` (Q_mse-only, 4-bit) at equal bpw:
  - Lower per-coord precision (3 bits vs 4) → larger residual
  - QJL term recovers inner-product accuracy in expectation
  - V cache should still use Q_mse-only (V is consumed by linear weighted
    sum, not inner products); only K cache benefits from this variant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from calm.llm_computer.tq4_torch import (
    HEAD_DIM, build_pi, _deterministic_orthogonal,
)


N_LEVELS_QJL = 8       # 3-bit Q_mse stage
JL_SEED = 137           # different from PI_SEED=42 to avoid degeneracy


# ============================================================================
# Lloyd-Max codebook for the 3-bit Q_mse stage
# ============================================================================

def compute_lloyd_max_codebook_3bit(
    head_dim: int = HEAD_DIM,
    max_iter: int = 200, tol: float = 1e-10,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """8-level Lloyd-Max codebook for N(0, 1/head_dim). Mirrors
    `compute_lloyd_max_codebook` in tq4_torch but with N_LEVELS=8."""
    sigma = 1.0 / math.sqrt(head_dim)
    lo = -3.5 * sigma
    hi = 3.5 * sigma

    idx = torch.arange(N_LEVELS_QJL, dtype=dtype)
    c = lo + (hi - lo) * (idx + 0.5) / N_LEVELS_QJL

    sigma_sq = sigma * sigma
    inv_sigma_sqrt2 = 1.0 / (sigma * math.sqrt(2.0))
    pdf_norm = 1.0 / (sigma * math.sqrt(2.0 * math.pi))

    for _ in range(max_iter):
        b = 0.5 * (c[:-1] + c[1:])
        edges = torch.empty(N_LEVELS_QJL + 1, dtype=dtype)
        edges[0] = lo * 3.0
        edges[-1] = hi * 3.0
        edges[1:-1] = b
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


def build_jl_matrix(
    d: int = HEAD_DIM, seed: int = JL_SEED,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Standard Gaussian JL matrix S ∈ R^(d × d). Deterministic given seed.
    Each entry ~ N(0, 1). Used for both encode (sign(S·r)) and decode-time
    inner-product estimator (sign(S·y_rot))."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.randn(d, d, generator=gen, dtype=torch.float32)
    return s.to(device=device, dtype=dtype)


# ============================================================================
# 3-bit packing / unpacking (8 codes per 3 bytes)
# ============================================================================

def _pack_3bit(codes: torch.Tensor) -> torch.Tensor:
    """codes: (N_blocks, 256) uint8 in 0..7. Returns (N_blocks, 96) uint8.

    Packing layout: code i occupies bits [3*i : 3*i+3] of a 24-bit uint
    formed from 3 little-endian bytes. 8 codes per group → 3 bytes.
    """
    assert codes.dtype == torch.uint8
    assert codes.shape[-1] == 256
    n_blocks = codes.shape[0]
    groups = codes.reshape(n_blocks, 32, 8).long()  # (N, 32, 8)
    shifts = (torch.arange(8, device=codes.device) * 3).reshape(1, 1, 8)
    packed_24 = (groups << shifts).sum(dim=-1)  # (N, 32) uint24 in uint64
    b0 = (packed_24 & 0xFF).to(torch.uint8)
    b1 = ((packed_24 >> 8) & 0xFF).to(torch.uint8)
    b2 = ((packed_24 >> 16) & 0xFF).to(torch.uint8)
    return torch.stack([b0, b1, b2], dim=-1).reshape(n_blocks, 96)


def _unpack_3bit(qs_3bit: torch.Tensor) -> torch.Tensor:
    """Inverse of _pack_3bit. (N_blocks, 96) uint8 → (N_blocks, 256) uint8."""
    n_blocks = qs_3bit.shape[0]
    qs = qs_3bit.reshape(n_blocks, 32, 3).long()
    packed_24 = qs[..., 0] | (qs[..., 1] << 8) | (qs[..., 2] << 16)  # (N, 32)
    shifts = (torch.arange(8, device=qs_3bit.device) * 3).reshape(1, 1, 8)
    codes = (packed_24.unsqueeze(-1) >> shifts) & 0x7  # (N, 32, 8)
    return codes.reshape(n_blocks, 256).to(torch.uint8)


def _pack_signs(signs: torch.Tensor) -> torch.Tensor:
    """signs: (N_blocks, 256) values in {-1, +1} or {0, 1}.
    Returns (N_blocks, 32) uint8 with 8 sign bits per byte."""
    n_blocks = signs.shape[0]
    bits = (signs > 0).to(torch.uint8).reshape(n_blocks, 32, 8).long()
    shifts = torch.arange(8, device=signs.device).reshape(1, 1, 8)
    packed = (bits << shifts).sum(dim=-1).to(torch.uint8)  # (N, 32)
    return packed


def _unpack_signs(qjl_signs: torch.Tensor) -> torch.Tensor:
    """Inverse of _pack_signs. (N_blocks, 32) uint8 → (N_blocks, 256)
    fp32 in {-1, +1}."""
    n_blocks = qjl_signs.shape[0]
    sb = qjl_signs.reshape(n_blocks, 32).long()
    shifts = torch.arange(8, device=qjl_signs.device).reshape(1, 1, 8)
    bits = (sb.unsqueeze(-1) >> shifts) & 0x1  # (N, 32, 8)
    return (bits.reshape(n_blocks, 256).float() * 2.0 - 1.0)


# ============================================================================
# Tq4QjlTensor + encode + decode
# ============================================================================

@dataclass
class Tq4QjlTensor:
    qs_3bit: torch.Tensor    # (n_blocks, 96) uint8
    qjl_signs: torch.Tensor  # (n_blocks, 32) uint8
    d_mse: torch.Tensor      # (n_blocks,) fp16
    d_qjl: torch.Tensor      # (n_blocks,) fp16
    shape: tuple

    @property
    def n_blocks(self) -> int:
        return self.qs_3bit.shape[0]

    def bytes_on_disk(self) -> int:
        """Theoretical packed size: 96 + 32 + 2 + 2 = 132 bytes per block.
        Matches tq4_k256 for memory parity."""
        return self.n_blocks * 132


def quantize_tq4_qjl(
    x: torch.Tensor,
    pi: Optional[torch.Tensor] = None,
    boundaries_3bit: Optional[torch.Tensor] = None,
    centroids_3bit: Optional[torch.Tensor] = None,
    jl: Optional[torch.Tensor] = None,
) -> Tq4QjlTensor:
    """Encode via Algorithm 2 from TurboQuant §3.2.

    Args:
      x: arbitrary-shape tensor with numel divisible by HEAD_DIM=256.
      pi: (256, 256) orthogonal rotation. Default: build_pi seed=42.
      boundaries_3bit / centroids_3bit: 8-level Lloyd-Max codebook.
      jl: (256, 256) Gaussian JL matrix. Default: build_jl_matrix seed=137.

    Returns:
      Tq4QjlTensor with packed 3-bit codes + sign bits + (d_mse, d_qjl).
    """
    orig_shape = tuple(x.shape)
    if pi is None:
        pi = build_pi(device=x.device, dtype=torch.float32)
    if boundaries_3bit is None or centroids_3bit is None:
        c, b = compute_lloyd_max_codebook_3bit()
        if centroids_3bit is None:
            centroids_3bit = c.to(x.device)
        if boundaries_3bit is None:
            boundaries_3bit = b.to(x.device)
    if jl is None:
        jl = build_jl_matrix(device=x.device)

    blocks = x.float().reshape(-1, HEAD_DIM)
    n_blocks = blocks.shape[0]

    # 1. Per-block L2 normalize
    d_mse = blocks.norm(dim=-1)  # (n_blocks,)
    inv_d_mse = torch.where(d_mse > 1e-8, 1.0 / d_mse, torch.zeros_like(d_mse))

    # 2. Rotate
    z = blocks @ pi.T  # (n_blocks, 256)
    z = z * inv_d_mse.unsqueeze(-1)  # unit-norm rotated

    # 3. Q_mse at 3 bits (8 levels)
    codes = (z.unsqueeze(-1) >= boundaries_3bit.view(1, 1, -1)).sum(dim=-1)
    codes = codes.to(torch.uint8)  # (n_blocks, 256) in 0..7
    z_hat = centroids_3bit[codes.long()]  # (n_blocks, 256)

    # 4. Residual
    r = z - z_hat  # (n_blocks, 256)
    d_qjl = r.norm(dim=-1)  # (n_blocks,)

    # 5. QJL stage: signs = sign(S · r)
    s_r = r @ jl.T  # (n_blocks, 256)
    signs = (s_r > 0).to(torch.float32) * 2.0 - 1.0

    # Pack
    qs_3bit = _pack_3bit(codes)
    qjl_signs = _pack_signs(signs)
    return Tq4QjlTensor(
        qs_3bit=qs_3bit,
        qjl_signs=qjl_signs,
        d_mse=d_mse.to(torch.float16),
        d_qjl=d_qjl.to(torch.float16),
        shape=orig_shape,
    )


def dequantize_tq4_qjl_mse_only(
    q: Tq4QjlTensor,
    pi: Optional[torch.Tensor] = None,
    centroids_3bit: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Best-effort reconstruction using ONLY the MSE stage (no QJL).
    Useful for debugging the per-element reconstruction error of the
    3-bit stage. For production use, prefer `qjl_inner_product` which
    uses both stages and is the unbiased estimator."""
    if pi is None:
        pi = build_pi(device=q.qs_3bit.device, dtype=torch.float32)
    if centroids_3bit is None:
        c, _ = compute_lloyd_max_codebook_3bit()
        centroids_3bit = c.to(q.qs_3bit.device)

    codes = _unpack_3bit(q.qs_3bit)  # (n_blocks, 256)
    z_hat = centroids_3bit[codes.long()]
    # Inverse rotation: y_hat @ Pi  (Pi orthogonal ⇒ y_hat @ Pi un-rotates
    # vectors that were stored as Pi.T-rotated, matching tq4_torch convention).
    result = (z_hat @ pi) * q.d_mse.float().unsqueeze(-1)
    return result.reshape(q.shape)


def qjl_inner_product(
    q: Tq4QjlTensor,
    y: torch.Tensor,
    pi: Optional[torch.Tensor] = None,
    centroids_3bit: Optional[torch.Tensor] = None,
    jl: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unbiased inner-product estimator <x_q, y> per Algorithm 2.

    Args:
      q: Tq4QjlTensor encoding x with shape (..., HEAD_DIM).
      y: tensor with last dim HEAD_DIM. Inner product is per-block.

    Returns:
      Tensor with shape `x.shape[:-1] broadcast against y.shape[:-1]`.
      For y shape (HEAD_DIM,): returns (n_blocks,).
      For y shape (M, HEAD_DIM): returns (M, n_blocks).
    """
    if pi is None:
        pi = build_pi(device=q.qs_3bit.device, dtype=torch.float32)
    if centroids_3bit is None:
        c, _ = compute_lloyd_max_codebook_3bit()
        centroids_3bit = c.to(q.qs_3bit.device)
    if jl is None:
        jl = build_jl_matrix(device=q.qs_3bit.device)

    codes = _unpack_3bit(q.qs_3bit)        # (n_blocks, 256)
    z_hat = centroids_3bit[codes.long()]    # (n_blocks, 256)
    signs = _unpack_signs(q.qjl_signs)      # (n_blocks, 256) {-1, +1}

    # Pre-rotate y once.
    y2 = y.reshape(-1, HEAD_DIM) if y.dim() > 1 else y.unsqueeze(0)
    y_rot = y2 @ pi.T                       # (M, 256)

    # MSE term: <z_hat, y_rot> per (block, m)
    mse_term = z_hat @ y_rot.T              # (n_blocks, M)

    # QJL term: <signs, S·y_rot> per (block, m), scaled.
    # S·y_rot stays REAL — Sheppard's identity needs the unsigned values
    # on the y side. Signing y collapses signal and biases the estimator
    # (the version that sign-collapsed both sides was off by ~200%).
    sy = y_rot @ jl.T                       # (M, 256) real
    sign_dot = signs @ sy.T                 # (n_blocks, M)
    # Estimator coefficient: sqrt(π/2) · d_qjl / d (linear in d_qjl, not
    # squared — d_qjl² would re-scale by an extra ‖r‖ that doesn't appear
    # in the paper's identity).
    coef_per_block = (math.sqrt(math.pi / 2.0)
                      * q.d_qjl.float()
                      / HEAD_DIM)
    qjl_term = coef_per_block.unsqueeze(-1) * sign_dot  # (n_blocks, M)

    score = q.d_mse.float().unsqueeze(-1) * (mse_term + qjl_term)  # (n_blocks, M)

    if y.dim() == 1:
        return score.squeeze(-1)
    return score.T  # (M, n_blocks)
