"""RoPE — rotary positional embedding for Gemma-compatible streams.

Adds RoPE to the substrate for any stream hosting external-model
weights (Gemma, Llama) that expect rotary position. Compiled-card
streams (d_head=2) don't use RoPE.

Reference: Su et al. 2021 — Roformer. For Gemma specifically: standard
RoPE with base=10000 (or larger for Gemma 4 with extended context);
half-dim rotation (rotate pairs of channels).
"""

from __future__ import annotations

import math

import torch


def build_rope_cache(
    head_dim: int, max_len: int, base: float = 10000.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) of shape (max_len, head_dim).

    Inverse frequencies: inv_freq[i] = 1 / base^(2i / head_dim) for
    i in [0, head_dim/2). Applied to paired channels.
    """
    assert head_dim % 2 == 0, f"head_dim {head_dim} must be even for RoPE"
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device,
                                             dtype=torch.float32) / half))
    pos = torch.arange(max_len, device=device, dtype=torch.float32)
    # (max_len, half)
    freqs = torch.outer(pos, inv_freq)
    # Full (max_len, head_dim) by duplicating each freq
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos.to(dtype), sin.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dim into the first."""
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply RoPE to x with shape (..., seq_len, head_dim).

    Args:
        x: input tensor, last two dims = (seq_len, head_dim).
        cos, sin: from build_rope_cache, shape (max_len, head_dim).
        positions: optional (seq_len,) int tensor selecting positions;
            if None, uses arange(seq_len).

    Returns:
        Rotated tensor, same shape as x.
    """
    seq_len = x.size(-2)
    if positions is None:
        positions = torch.arange(seq_len, device=x.device)
    # Pick (cos, sin) at those positions → (seq_len, head_dim)
    c = cos[positions]
    s = sin[positions]
    # Broadcast over leading dims
    while c.dim() < x.dim():
        c = c.unsqueeze(0)
        s = s.unsqueeze(0)
    return x * c + rotate_half(x) * s
