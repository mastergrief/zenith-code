"""HRM-Text-1.58 atomic layers.

Source: sapientinc/HRM-Text SHA 056c4ec, `models/layers.py` + `models/common.py`.

Deviations recorded in RESEARCH/HRM-Text-1.58/01_DEVIATIONS.md:
- D1.2: `F.scaled_dot_product_attention` + constructed PrefixLM mask
  replaces flash-attn `flash_attn_varlen_prefixlm` /
  `flash_attn_with_kvcache`. Kept the mask semantics identical.
- D1.6: PrefixLM mask builder is inline (build_prefix_lm_mask).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# --------------------------------------------------------------------------- #
# Init helpers (port of models/common.py + models/layers.py:19-20)
# --------------------------------------------------------------------------- #

def find_multiple(a: int, b: int) -> int:
    """Round a up to next multiple of b. Port of `models/layers.py:19-20`."""
    return (-(a // -b)) * b


def trunc_normal_init_(tensor: Tensor, std: float = 1.0) -> Tensor:
    """Fast approximate truncated normal init.

    Port of `sapientinc/HRM-Text/models/common.py:10-13` (verbatim).
    """
    return tensor.normal_().fmod_(3.0).mul_(1.014762601732121 * std)


# --------------------------------------------------------------------------- #
# Linear init (port of models/layers.py:61-85)
# --------------------------------------------------------------------------- #

class LinearInit(nn.Module):
    """Linear with truncated-normal init.

    Port of `sapientinc/HRM-Text/models/layers.py:61-85`.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        batch_out_features: Sequence[int] = (),
        init_std: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        # Truncated LeCun normal init
        if init_std is None:
            init_std = 1.0 / (in_features ** 0.5)
        self.weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty((math.prod(batch_out_features) * out_features, in_features), **kwargs),
                std=init_std,
            )
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros((math.prod(batch_out_features) * out_features,), **kwargs))

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)


# --------------------------------------------------------------------------- #
# Scaled embedding init (port of models/layers.py:88-102)
# --------------------------------------------------------------------------- #

class ScaledEmbeddingInit(nn.Module):
    """Embedding with scale factor = 1/init_std, applied multiplicatively.

    Port of `sapientinc/HRM-Text/models/layers.py:88-102`.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, **kwargs) -> None:
        super().__init__()
        self.scale = 1.0 / init_std
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim), **kwargs), std=init_std)
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.scale * F.embedding(input, self.embedding_weight)


# --------------------------------------------------------------------------- #
# RoPE (port of models/layers.py:23-58)
# --------------------------------------------------------------------------- #

CosSin = Tuple[Tensor, Tensor]


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims. Port of `models/layers.py:23-27`."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: Tensor, cos_sin: CosSin) -> Tensor:
    """Apply RoPE to x. Port of `models/layers.py:30-38`.

    x: [..., seq_len, num_heads, head_dim]
    cos, sin: [seq_len, head_dim] OR [..., seq_len, head_dim]
    """
    cos, sin = cos_sin
    return ((x * cos.unsqueeze(-2)) + (rotate_half(x) * sin.unsqueeze(-2))).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Buffered cos/sin lookup. Port of `models/layers.py:41-58`."""
    def __init__(self, dim: int, max_seq_len: int, base: float, **kwargs) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, **kwargs) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32, **kwargs)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self, position_ids: Optional[Tensor]) -> CosSin:
        if position_ids is not None:
            return self.cos_cached[position_ids], self.sin_cached[position_ids]
        return self.cos_cached, self.sin_cached


# --------------------------------------------------------------------------- #
# PrefixLM mask builder (D1.6 — replaces flash_attn_varlen_prefixlm)
# --------------------------------------------------------------------------- #

def build_prefix_lm_mask(
    seq_len: int,
    sep_positions: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bool,
) -> Tensor:
    """Build PrefixLM attention mask matching upstream
    flash_attn_varlen_prefixlm semantics.

    Spec (matches `attn_type='prefixlm'` in `models/layers.py:142-151`):
    - Positions 0..sep_position (inclusive) are bidirectional prefix
    - Positions > sep_position attend prefix + their own causal history
    - Diagonal always True (token attends itself)

    Args:
        seq_len: S
        sep_positions: (B,) long, per-row sep token position (0 ≤ sep < S)
        device, dtype: target

    Returns:
        mask: (B, S, S) bool. mask[b, q, k] = True iff query q in batch b
              can attend key k. Pass as additive negative-inf mask to
              SDPA via `attn_mask = mask` (boolean True = allow).
    """
    assert sep_positions.dim() == 1
    B = sep_positions.shape[0]
    # idx[q, k]: q on dim 1, k on dim 2 after broadcasting
    q_idx = torch.arange(seq_len, device=device).view(1, seq_len, 1)  # (1, S, 1)
    k_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len)  # (1, 1, S)
    sep = sep_positions.view(B, 1, 1)

    # Causal lower-triangular for q >= k
    causal = (k_idx <= q_idx)
    # Allow attending to prefix (k <= sep) from any query
    prefix_keys = (k_idx <= sep)
    # Query is in prefix: bidirectional (any key)
    q_in_prefix = (q_idx <= sep)
    # Final mask: q_in_prefix → allow (k <= sep ∪ k <= q_idx is moot for in-prefix; allow all prefix keys)
    # q_in_suffix → allow prefix keys (k <= sep) + causal history (k <= q)
    mask = torch.where(
        q_in_prefix,
        prefix_keys,                       # in-prefix query: only see prefix (bidirectional)
        prefix_keys | causal,              # suffix query: see prefix + own causal history
    )
    return mask.to(dtype)


# --------------------------------------------------------------------------- #
# Attention (port of models/layers.py:116-155)
# --------------------------------------------------------------------------- #

class Attention(nn.Module):
    """Fused gqkv attention with sigmoid-gated output.

    Port of `sapientinc/HRM-Text/models/layers.py:116-155`.

    Deviation D1.2 (RESEARCH/HRM-Text-1.58/01_DEVIATIONS.md):
    Replace `flash_attn_varlen_prefixlm` + `flash_attn_with_kvcache`
    with `F.scaled_dot_product_attention` + constructed PrefixLM mask
    via `build_prefix_lm_mask`. Mask semantics preserved.

    Deviation D2.1 (Phase 2): when use_ternary_bulk=True, gqkv_proj +
    o_proj use BitLinear (ternary master+STE) instead of LinearInit.
    """
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        attn_type: str,
        init_std_in: Optional[float] = None,
        init_std_out: Optional[float] = None,
        use_ternary_bulk: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.attn_type = attn_type
        # Ternary or FP/BF16 master? Per D2.1 bounded scope: gqkv_proj + o_proj only.
        # Import inside __init__ to avoid circular import at module load
        # (bit_linear imports trunc_normal_init_ from layers).
        if use_ternary_bulk:
            from calm.hrm_text_158.bit_linear import BitLinear
            LinearImpl = BitLinear
        else:
            LinearImpl = LinearInit
        # Fused gqkv: layout = (gate, query, key, value) along the head axis
        # Total: 2*num_heads (gate+query) + 2*num_key_value_heads (key+value)
        self.gqkv_proj = LinearImpl(
            hidden_size,
            self.head_dim,
            batch_out_features=(2 * self.num_heads + 2 * self.num_key_value_heads,),
            bias=False,
            init_std=init_std_in,
            **kwargs,
        )
        self.o_proj = LinearImpl(head_dim * num_heads, hidden_size, bias=False, init_std=init_std_out, **kwargs)

    def forward(
        self,
        hidden_states: Tensor,
        cos_sin: Optional[CosSin] = None,
        sep_positions: Optional[Tensor] = None,
        **seq_info,
    ) -> Tensor:
        # gqkv: [..., S, hidden_size] -> [..., S, (2*h+2*kvh)*head_dim]
        B, S, _ = hidden_states.shape
        gqkv = self.gqkv_proj(hidden_states)
        # Split into heads dimension. Port of `layers.py:134`:
        #   gqkv = rearrange(gqkv, "... (h hd) -> ... h hd", h=2h+2kvh)
        total_heads = 2 * self.num_heads + 2 * self.num_key_value_heads
        gqkv = gqkv.view(B, S, total_heads, self.head_dim)
        # Split order: (num_heads, num_heads, num_key_value_heads, num_key_value_heads)
        gate, query, key, value = gqkv.split(
            (self.num_heads, self.num_heads, self.num_key_value_heads, self.num_key_value_heads),
            dim=-2,
        )
        # query, key, value: [B, S, num_heads, head_dim]
        if cos_sin is not None:
            query = apply_rotary_pos_emb(query, cos_sin)
            key = apply_rotary_pos_emb(key, cos_sin)
        # SDPA expects (B, num_heads, S, head_dim)
        q_t = query.transpose(1, 2)
        k_t = key.transpose(1, 2)
        v_t = value.transpose(1, 2)
        # Build mask
        if self.attn_type == "prefixlm":
            assert sep_positions is not None, (
                "Attention requires sep_positions when attn_type='prefixlm'"
            )
            # mask shape: (B, S, S). Expand to (B, 1, S, S) for SDPA.
            mask_2d = build_prefix_lm_mask(S, sep_positions, device=hidden_states.device, dtype=torch.bool)
            attn_mask = mask_2d.unsqueeze(1)  # (B, 1, S, S)
            is_causal = False
        elif self.attn_type == "causal":
            attn_mask = None
            is_causal = True
        else:
            raise NotImplementedError(f"attn_type={self.attn_type!r}")
        # Need to handle GQA (num_kv < num_heads) — repeat KV heads if needed
        if self.num_key_value_heads != self.num_heads:
            kv_repeat = self.num_heads // self.num_key_value_heads
            k_t = k_t.repeat_interleave(kv_repeat, dim=1)
            v_t = v_t.repeat_interleave(kv_repeat, dim=1)
        attn_out = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
        # attn_out: (B, num_heads, S, head_dim) -> (B, S, num_heads, head_dim)
        attn_out = attn_out.transpose(1, 2)
        # Sigmoid-gated output: port of `layers.py:154`
        attn_out = (torch.sigmoid(gate) * attn_out).reshape(B, S, self.num_heads * self.head_dim)
        return self.o_proj(attn_out)


# --------------------------------------------------------------------------- #
# SwiGLU (port of models/layers.py:158-168)
# --------------------------------------------------------------------------- #

class SwiGLU(nn.Module):
    """Fused gate+up SwiGLU MLP.

    Port of `sapientinc/HRM-Text/models/layers.py:158-168`.

    Deviation D2.1 (Phase 2): when use_ternary_bulk=True, gate_up_proj +
    down_proj use BitLinear (ternary master+STE) instead of LinearInit.
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        init_std_in: Optional[float] = None,
        init_std_out: Optional[float] = None,
        use_ternary_bulk: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Per D2.1 bounded scope: gate_up_proj + down_proj.
        if use_ternary_bulk:
            from calm.hrm_text_158.bit_linear import BitLinear
            LinearImpl = BitLinear
        else:
            LinearImpl = LinearInit
        # Fused (gate, up) projection: output dim = 2 * intermediate_size
        self.gate_up_proj = LinearImpl(
            hidden_size,
            intermediate_size,
            batch_out_features=(2,),
            bias=False,
            init_std=init_std_in,
            **kwargs,
        )
        self.down_proj = LinearImpl(
            intermediate_size,
            hidden_size,
            bias=False,
            init_std=init_std_out,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
