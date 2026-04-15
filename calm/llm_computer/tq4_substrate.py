"""Tq4-storage variant of GroupedSmall2DTransformer.

Same architecture as GroupedSmall2DTransformer but every Linear is a
Tq4LinearGGMLOriented storing weights in tq4 (4.125 bits/element +
block structure). At Gemma 4 E4B scale (d_model=4096, 42 layers) this
brings substrate memory from ~34GB FP32 down to ~4.4GB tq4 — fits in
8GB VRAM with room for activations.

Forward pass: each layer dequantizes its tq4 weights on-the-fly (via
Tq4LinearGGMLOriented's straight-through dequant), runs attention + FFN,
residuals stay in FP32. Training through STE works the same as
regular Tq4Linear.

Weight storage convention: (in, out) matching GGUF. Lets us byte-level
install Gemma bytes without re-quantization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from calm.llm_computer.grouped_attention import (
    grouped_attention, grouped_attention_single_head_mode,
)
from calm.llm_computer.grouped_small2d import GroupedSmall2DConfig
from calm.llm_computer.tq4_byte_install import (
    Tq4LinearGGMLOriented, pad_tq4_tensor_rows_and_cols,
)
from calm.llm_computer.tq4_torch import HEAD_DIM, Tq4Tensor


class Tq4GroupedSmall2DTransformer(nn.Module):
    """tq4-storage GroupedSmall2DTransformer.

    Mirrors the parent's forward pass but stores all Linear weights as
    Tq4LinearGGMLOriented. Tokens and positions still use FP32
    embeddings (small tensors; quantizing them hurts).

    Notes on shape discipline:
      - W_qkv stacks Q, K, V as a single Linear(in=D, out=3*D). Byte-
        install for Q, K, V separately uses `install_qkv_from_parts`.
      - W_out: Linear(in=D, out=D)
      - ff_in stacks gate and up: Linear(in=D, out=2*d_ffn)
      - ff_out: Linear(in=d_ffn, out=D)
      - head: Linear(in=D, out=vocab)
    """

    def __init__(self, config: GroupedSmall2DConfig):
        super().__init__()
        self.config = config
        cfg = config
        # Assert block alignment — tq4 requires everything be multiples of 256
        for name, val in [
            ("d_model", cfg.d_model),
            ("d_ffn", cfg.d_ffn),
            ("3*d_model", 3 * cfg.d_model),
            ("2*d_ffn", 2 * cfg.d_ffn),
            ("vocab_size", cfg.vocab_size),
        ]:
            assert val % HEAD_DIM == 0, (
                f"{name}={val} not divisible by HEAD_DIM={HEAD_DIM}"
            )

        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)

        self.W_qkv = nn.ModuleList([
            Tq4LinearGGMLOriented(cfg.d_model, 3 * cfg.d_model)
            for _ in range(cfg.n_layers)
        ])
        self.W_out = nn.ModuleList([
            Tq4LinearGGMLOriented(cfg.d_model, cfg.d_model)
            for _ in range(cfg.n_layers)
        ])
        self.ff_in = nn.ModuleList([
            Tq4LinearGGMLOriented(cfg.d_model, 2 * cfg.d_ffn)
            for _ in range(cfg.n_layers)
        ])
        self.ff_out = nn.ModuleList([
            Tq4LinearGGMLOriented(cfg.d_ffn, cfg.d_model)
            for _ in range(cfg.n_layers)
        ])
        self.head = Tq4LinearGGMLOriented(cfg.d_model, cfg.vocab_size)

    def initialize_all_zero_tq4(self) -> None:
        """Install zero-block weights into every Tq4LinearGGMLOriented so
        forward pass runs (returns all-zero logits until real weights
        are installed).
        """
        def zero_q(in_features: int, out_features: int) -> Tq4Tensor:
            total = in_features * out_features
            n_blocks = total // HEAD_DIM
            qs = torch.zeros(n_blocks, 128, dtype=torch.uint8)
            d = torch.zeros(n_blocks, dtype=torch.float32)
            return Tq4Tensor(qs=qs, d=d, shape=(in_features, out_features))

        for layer in self.W_qkv:
            layer.install_tq4(zero_q(layer.in_features, layer.out_features))
        for layer in self.W_out:
            layer.install_tq4(zero_q(layer.in_features, layer.out_features))
        for layer in self.ff_in:
            layer.install_tq4(zero_q(layer.in_features, layer.out_features))
        for layer in self.ff_out:
            layer.install_tq4(zero_q(layer.in_features, layer.out_features))
        self.head.install_tq4(
            zero_q(self.head.in_features, self.head.out_features),
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        B, S = idx.shape
        pos_idx = torch.arange(S, device=idx.device)
        x = self.tok(idx) + self.pos(pos_idx)
        mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1,
        )

        for layer in range(cfg.n_layers):
            qkv = self.W_qkv[layer](x)
            qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            q_bh = q.transpose(1, 2)
            k_bh = k.transpose(1, 2)
            v_bh = v.transpose(1, 2)

            mode = cfg.layer_modes[layer]
            if mode == "grouped":
                attn = grouped_attention(
                    q_bh, k_bh, v_bh,
                    n_groups=cfg.layer_n_groups[layer],
                    group_size=cfg.layer_group_sizes[layer],
                    mask=mask,
                )
            else:
                attn = grouped_attention_single_head_mode(
                    q_bh, k_bh, v_bh, mask=mask, scale=1.0,
                )
            attn = attn.reshape(B, S, cfg.d_model)
            x = x + self.W_out[layer](attn)
            gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
            x = x + self.ff_out[layer](F.relu(gate) * val)

        return self.head(x)


def install_qkv_from_parts(
    target: Tq4LinearGGMLOriented,
    q_tq4: Tq4Tensor,
    k_tq4: Tq4Tensor,
    v_tq4: Tq4Tensor,
    substrate_d_model: int,
) -> None:
    """Byte-install separate Q, K, V tq4 tensors into a stacked W_qkv.

    W_qkv stores weight in (in=D_s, out=3*D_s) orientation. Q occupies
    output columns [0, D_s); K occupies [D_s, 2*D_s); V occupies
    [2*D_s, 3*D_s). Each region is zero-padded from its source Gemma
    shape up to D_s.

    Gemma provides:
      q_tq4: shape (gemma_d_model, gemma_q_out)   — e.g. (2560, 2048)
      k_tq4: shape (gemma_d_model, gemma_kv_out)  — e.g. (2560, 512)
      v_tq4: shape (gemma_d_model, gemma_kv_out)  — e.g. (2560, 512)

    Targets:
      W_qkv columns [0, D_s)       = Q region   ← padded q_tq4
      W_qkv columns [D_s, 2*D_s)    = K region   ← padded k_tq4
      W_qkv columns [2*D_s, 3*D_s)  = V region   ← padded v_tq4

    Since W_qkv stores (D_s, 3*D_s) in row-major (in, out) block order,
    the three regions correspond to three contiguous column ranges per
    row. We build the assembled tq4 byte buffer row-by-row:

      row i (i < gemma_d_model):
        Q blocks (padded to D_s/HEAD_DIM blocks) ||
        K blocks (padded) || V blocks (padded)
      row i (gemma_d_model <= i < D_s):
        all zero blocks across 3*D_s
    """
    D_s = substrate_d_model
    gemma_d_model, gemma_q_out = q_tq4.shape
    _, gemma_kv_out = k_tq4.shape
    assert v_tq4.shape == k_tq4.shape, "K and V shapes must match"
    assert q_tq4.shape[0] == gemma_d_model
    assert gemma_d_model <= D_s
    assert gemma_q_out <= D_s and gemma_kv_out <= D_s

    # Block counts per row for each region
    bpr_q_src = gemma_q_out // HEAD_DIM    # blocks per Gemma Q row
    bpr_kv_src = gemma_kv_out // HEAD_DIM
    bpr_target = D_s // HEAD_DIM            # blocks per substrate region row
    bpr_full_row = 3 * bpr_target           # blocks per substrate W_qkv row

    total_blocks = D_s * bpr_full_row
    new_qs = torch.zeros(total_blocks, 128, dtype=torch.uint8)
    new_d = torch.zeros(total_blocks, dtype=torch.float32)

    for row in range(gemma_d_model):
        # Q region: substrate_blocks[row*bpr_full_row : row*bpr_full_row+bpr_target]
        q_src_start = row * bpr_q_src
        q_tgt_start = row * bpr_full_row
        new_qs[q_tgt_start : q_tgt_start + bpr_q_src] = (
            q_tq4.qs[q_src_start : q_src_start + bpr_q_src]
        )
        new_d[q_tgt_start : q_tgt_start + bpr_q_src] = (
            q_tq4.d[q_src_start : q_src_start + bpr_q_src]
        )
        # K region: offset by bpr_target
        k_src_start = row * bpr_kv_src
        k_tgt_start = q_tgt_start + bpr_target
        new_qs[k_tgt_start : k_tgt_start + bpr_kv_src] = (
            k_tq4.qs[k_src_start : k_src_start + bpr_kv_src]
        )
        new_d[k_tgt_start : k_tgt_start + bpr_kv_src] = (
            k_tq4.d[k_src_start : k_src_start + bpr_kv_src]
        )
        # V region: offset by 2 * bpr_target
        v_src_start = row * bpr_kv_src
        v_tgt_start = q_tgt_start + 2 * bpr_target
        new_qs[v_tgt_start : v_tgt_start + bpr_kv_src] = (
            v_tq4.qs[v_src_start : v_src_start + bpr_kv_src]
        )
        new_d[v_tgt_start : v_tgt_start + bpr_kv_src] = (
            v_tq4.d[v_src_start : v_src_start + bpr_kv_src]
        )
        # Remaining blocks in the row (padding regions) stay zero

    assembled = Tq4Tensor(
        qs=new_qs, d=new_d, shape=(D_s, 3 * D_s),
    )
    target.install_tq4(assembled)


def install_ffn_in_from_parts(
    target: Tq4LinearGGMLOriented,
    gate_tq4: Tq4Tensor,
    up_tq4: Tq4Tensor,
    substrate_d_model: int,
    substrate_d_ffn: int,
) -> None:
    """Byte-install gate and up projections into stacked ff_in.

    ff_in stores (in=D_s, out=2*d_ffn_s) in (in, out) orientation.
    Gate occupies [0, d_ffn_s); up occupies [d_ffn_s, 2*d_ffn_s).
    """
    D_s = substrate_d_model
    D_ffn_s = substrate_d_ffn
    gemma_d_model, gemma_d_ffn = gate_tq4.shape
    assert up_tq4.shape == gate_tq4.shape
    assert gemma_d_model <= D_s
    assert gemma_d_ffn <= D_ffn_s

    bpr_src = gemma_d_ffn // HEAD_DIM
    bpr_target = D_ffn_s // HEAD_DIM
    bpr_full_row = 2 * bpr_target
    total_blocks = D_s * bpr_full_row

    new_qs = torch.zeros(total_blocks, 128, dtype=torch.uint8)
    new_d = torch.zeros(total_blocks, dtype=torch.float32)

    for row in range(gemma_d_model):
        row_tgt = row * bpr_full_row
        # Gate region
        gate_src = row * bpr_src
        new_qs[row_tgt : row_tgt + bpr_src] = gate_tq4.qs[gate_src : gate_src + bpr_src]
        new_d[row_tgt : row_tgt + bpr_src] = gate_tq4.d[gate_src : gate_src + bpr_src]
        # Up region
        up_src = row * bpr_src
        up_tgt = row_tgt + bpr_target
        new_qs[up_tgt : up_tgt + bpr_src] = up_tq4.qs[up_src : up_src + bpr_src]
        new_d[up_tgt : up_tgt + bpr_src] = up_tq4.d[up_src : up_src + bpr_src]

    assembled = Tq4Tensor(qs=new_qs, d=new_d, shape=(D_s, 2 * D_ffn_s))
    target.install_tq4(assembled)


def install_simple_tq4_corner(
    target: Tq4LinearGGMLOriented,
    src_tq4: Tq4Tensor,
    target_in: int,
    target_out: int,
) -> None:
    """Install a src_tq4 tensor into the top-left corner of target.

    Used for W_out and ff_out where Gemma's weight is smaller than the
    substrate's Linear but doesn't need stacking (unlike Q/K/V).
    """
    padded = pad_tq4_tensor_rows_and_cols(
        src_tq4, target_n_rows=target_in, target_n_cols=target_out,
    )
    target.install_tq4(padded)
