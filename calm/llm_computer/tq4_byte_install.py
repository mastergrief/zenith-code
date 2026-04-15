"""Byte-level install of Gemma tq4 weights into substrate Tq4Linear.

Copies tq4 block bytes DIRECTLY from GGUF into substrate storage, with
zero blocks filling padding regions. Preserves Gemma's exact tq4
numerics — no dequant, no re-quant, no 1% drift per weight.

Key insight: tq4 blocks are 256 contiguous elements. If we pad along
the LAST dimension AND original row length is a multiple of 256 AND
substrate row length is also a multiple of 256, we can copy block-
aligned slabs with zero blocks filling the gaps.

Transpose handling: GGUF stores (in, out); our Tq4Linear stores (out, in).
A transpose does NOT preserve block structure. Two options:

  (a) Use a ggml-orientation variant of Tq4Linear that matches GGUF's
      layout; forward does `x @ W` instead of `F.linear(x, W)`
  (b) Accept the dequant+re-quant 1% drift for the sake of uniform
      Tq4Linear convention

This module implements (a) via `Tq4LinearGGMLOriented`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from calm.llm_computer.tq4_torch import (
    HEAD_DIM, Tq4Tensor, build_pi, compute_lloyd_max_codebook,
    dequantize_tq4_differentiable,
)


def build_zero_tq4_block_bytes(n_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct n_blocks worth of zero tq4 blocks.

    A zero block: d=0 (fp16 zero) → regardless of qs, dequant yields 0
    because y_final = d * Pi^T @ centroids[codes] = 0 * (...) = 0.

    Returns (qs, d) tensors of shapes (n_blocks, 128) and (n_blocks,).
    """
    qs = torch.zeros(n_blocks, 128, dtype=torch.uint8)
    d = torch.zeros(n_blocks, dtype=torch.float32)
    return qs, d


def pad_tq4_tensor_columns(
    src: Tq4Tensor,
    target_n_cols: int,
) -> Tq4Tensor:
    """Pad a tq4 tensor's LAST dimension from src_cols to target_n_cols.

    Requires:
      - src.shape = (rows, src_cols)
      - src_cols divisible by HEAD_DIM=256 (so each row is an integer
        number of blocks)
      - target_n_cols divisible by HEAD_DIM

    Operation: for each source row, append (target_n_cols - src_cols) / 256
    zero blocks after its existing blocks.

    Returns: Tq4Tensor with shape (rows, target_n_cols) where original
    bytes fill [:, :src_cols] and zero blocks fill [:, src_cols:].
    """
    rows, src_cols = src.shape
    assert src_cols % HEAD_DIM == 0, (
        f"src_cols {src_cols} not divisible by HEAD_DIM {HEAD_DIM}"
    )
    assert target_n_cols % HEAD_DIM == 0, (
        f"target_n_cols {target_n_cols} not divisible by HEAD_DIM {HEAD_DIM}"
    )
    assert target_n_cols >= src_cols
    assert src.qs.shape[0] == rows * (src_cols // HEAD_DIM)

    blocks_per_src_row = src_cols // HEAD_DIM
    blocks_per_tgt_row = target_n_cols // HEAD_DIM
    pad_blocks_per_row = blocks_per_tgt_row - blocks_per_src_row
    total_tgt_blocks = rows * blocks_per_tgt_row

    new_qs = torch.zeros(total_tgt_blocks, 128, dtype=torch.uint8)
    new_d = torch.zeros(total_tgt_blocks, dtype=torch.float32)

    for row in range(rows):
        src_start = row * blocks_per_src_row
        tgt_start = row * blocks_per_tgt_row
        # Copy src row's blocks into first blocks_per_src_row of tgt row
        new_qs[tgt_start : tgt_start + blocks_per_src_row] = \
            src.qs[src_start : src_start + blocks_per_src_row]
        new_d[tgt_start : tgt_start + blocks_per_src_row] = \
            src.d[src_start : src_start + blocks_per_src_row]
        # Remaining blocks in tgt row stay zero (already zero-init)

    return Tq4Tensor(qs=new_qs, d=new_d, shape=(rows, target_n_cols))


def pad_tq4_tensor_rows(
    src: Tq4Tensor,
    target_n_rows: int,
) -> Tq4Tensor:
    """Pad a tq4 tensor with extra rows at the bottom (all zero blocks).

    This is cheap: just append (target_n_rows - src_rows) * blocks_per_row
    zero blocks to the end.
    """
    src_rows, cols = src.shape
    assert cols % HEAD_DIM == 0
    assert target_n_rows >= src_rows

    blocks_per_row = cols // HEAD_DIM
    extra_rows = target_n_rows - src_rows
    extra_blocks = extra_rows * blocks_per_row
    zero_qs, zero_d = build_zero_tq4_block_bytes(extra_blocks)

    return Tq4Tensor(
        qs=torch.cat([src.qs, zero_qs], dim=0),
        d=torch.cat([src.d, zero_d], dim=0),
        shape=(target_n_rows, cols),
    )


def pad_tq4_tensor_rows_and_cols(
    src: Tq4Tensor,
    target_n_rows: int,
    target_n_cols: int,
) -> Tq4Tensor:
    """Pad both dimensions. Column-pad first (per-row), then row-pad."""
    col_padded = pad_tq4_tensor_columns(src, target_n_cols)
    return pad_tq4_tensor_rows(col_padded, target_n_rows)


class Tq4LinearGGMLOriented(nn.Module):
    """Tq4Linear variant storing weight in GGUF's (in, out) orientation.

    Standard PyTorch Linear forward: y = x @ W.T where W is (out, in).
    This variant stores W as (in, out) matching GGUF, and forward does
    y = x @ W directly. Byte-compatible with GGUF tq4 tensors.

    Blocks are 256 contiguous elements taken row-major from the (in, out)
    shape. For Gemma's attn_q at (2560, 2048): row = input channel,
    col = output channel, blocks scan across the output dim per input.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert (in_features * out_features) % HEAD_DIM == 0, (
            f"in*out must be divisible by {HEAD_DIM}"
        )
        assert out_features % HEAD_DIM == 0, (
            "out_features must be divisible by HEAD_DIM so each row of the "
            "(in, out) weight is an integer number of tq4 blocks"
        )
        self._qs: Optional[torch.Tensor] = None
        self._d: Optional[torch.Tensor] = None
        self._weight_shape = (in_features, out_features)
        pi = build_pi(source="c_header")
        centroids, _ = compute_lloyd_max_codebook()
        self.register_buffer("_pi", pi)
        self.register_buffer("_centroids", centroids)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def install_tq4(self, q: Tq4Tensor) -> None:
        """Install pre-quantized tq4 bytes. No re-quantization."""
        assert q.shape == self._weight_shape, (
            f"tq4 shape {q.shape} != expected {self._weight_shape}"
        )
        expected_blocks = (self.in_features * self.out_features) // HEAD_DIM
        assert q.qs.shape == (expected_blocks, 128)
        self._qs = q.qs.clone()
        self._d = q.d.clone()

    def is_loaded(self) -> bool:
        return self._qs is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self._qs is not None, "call install_tq4() first"
        q = Tq4Tensor(qs=self._qs, d=self._d, shape=self._weight_shape)
        w = dequantize_tq4_differentiable(q, self._pi, self._centroids)
        # GGML orientation: (in, out). Forward is x @ W (not x @ W.T)
        out = x @ w
        if self.bias is not None:
            out = out + self.bias
        return out
