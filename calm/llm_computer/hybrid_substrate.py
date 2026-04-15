"""Hybrid substrate — per-layer linear type (FP32 or Tq4) in one model.

Round 11 finding: tq4 quantization destroys compiled-card correctness.
Compiled card weights are discrete (mostly 0, sparse ±1 / ±16 coefficients)
and the Lloyd-Max codebook — tuned for Gaussian LM weights — crushes
them to noise. Measurement: dispatched_v4 791/791 → 70/791 through a
tq4 roundtrip.

Fix: keep compiled-card layers in FP32; quantize Gemma-style trained
layers to tq4 as usual. Both coexist in one `nn.Module` with per-layer
linear-type dispatch. One `.pt`, two weight encodings.

Implementation:
  - `FP32LinearGGMLOriented` — nn.Linear-analog in GGUF's (in, out)
    orientation, forward is `x @ W` (matching Tq4LinearGGMLOriented's
    semantics so a layer can be either type and the forward loop is
    identical).
  - `HybridGroupedSmall2DTransformer` — per-layer selection of
    FP32LinearGGMLOriented or Tq4LinearGGMLOriented. Config carries a
    tuple of layer types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from calm.llm_computer.grouped_attention import grouped_attention_single_head_mode
from calm.llm_computer.grouped_small2d import GroupedSmall2DConfig
from calm.llm_computer.tq4_byte_install import Tq4LinearGGMLOriented
from calm.llm_computer.tq4_torch import HEAD_DIM


class FP32LinearGGMLOriented(nn.Module):
    """FP32 linear with GGUF weight orientation — `y = x @ W`, W: (in, out).

    Interface-compatible with `Tq4LinearGGMLOriented` so a substrate can
    mix-and-match linear types per layer. Compiled-card layers use this;
    Gemma layers use the tq4 variant.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight

    def is_loaded(self) -> bool:
        return True  # FP32 is always "loaded" (zero by default)


@dataclass
class HybridGroupedSmall2DConfig(GroupedSmall2DConfig):
    """Extends GroupedSmall2DConfig with per-layer linear type.

    `layer_linear_types`: tuple of "fp32" | "tq4" per layer. None → all fp32.
    """
    layer_linear_types: Optional[tuple[str, ...]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.layer_linear_types is None:
            object.__setattr__(self, "layer_linear_types",
                               tuple(["fp32"] * self.n_layers))
        assert len(self.layer_linear_types) == self.n_layers
        for i, t in enumerate(self.layer_linear_types):
            assert t in ("fp32", "tq4"), (
                f"layer {i}: unknown linear type {t!r}"
            )


def _make_linear(kind: str, in_features: int, out_features: int) -> nn.Module:
    if kind == "fp32":
        return FP32LinearGGMLOriented(in_features, out_features)
    if kind == "tq4":
        # Tq4 requires dimensional alignment — caller's responsibility.
        return Tq4LinearGGMLOriented(in_features, out_features)
    raise ValueError(f"unknown linear kind: {kind}")


class HybridGroupedSmall2DTransformer(nn.Module):
    """Small2DTransformer with per-layer linear type dispatch.

    Every `Linear` in the model is either FP32LinearGGMLOriented or
    Tq4LinearGGMLOriented based on `config.layer_linear_types`. The
    forward pass is identical to `GroupedSmall2DTransformer` because
    both linear types share the `y = x @ W` contract.

    tq4 layers must have block-aligned dims. FP32 layers have no
    alignment constraint (but the whole substrate must satisfy the
    config-level alignment).
    """

    def __init__(self, config: HybridGroupedSmall2DConfig):
        super().__init__()
        self.config = config
        self._hybrid_config = config
        cfg = config
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)

        types = cfg.layer_linear_types
        self.W_qkv = nn.ModuleList([
            _make_linear(types[i], cfg.d_model, 3 * cfg.d_model)
            for i in range(cfg.n_layers)
        ])
        self.W_out = nn.ModuleList([
            _make_linear(types[i], cfg.d_model, cfg.d_model)
            for i in range(cfg.n_layers)
        ])
        self.ff_in = nn.ModuleList([
            _make_linear(types[i], cfg.d_model, 2 * cfg.d_ffn)
            for i in range(cfg.n_layers)
        ])
        self.ff_out = nn.ModuleList([
            _make_linear(types[i], cfg.d_ffn, cfg.d_model)
            for i in range(cfg.n_layers)
        ])
        # Head always FP32 (no quantize reason — we want exact logits).
        self.head = FP32LinearGGMLOriented(cfg.d_model, cfg.vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        cfg = self._hybrid_config
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
            # Single-head mode only (compiled cards + Gemma-stand-in are
            # both compatible; full grouped-Gemma is deferred).
            attn = grouped_attention_single_head_mode(
                q_bh, k_bh, v_bh, mask=mask, scale=1.0,
                hard_max=cfg.layer_hard_max[layer],
            )
            attn = attn.reshape(B, S, cfg.d_model)
            x = x + self.W_out[layer](attn)
            gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
            x = x + self.ff_out[layer](F.relu(gate) * val)
        return self.head(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def initialize_tq4_layers_to_zero(self) -> None:
        """Install zero tq4 blocks on every tq4 layer so forward runs.
        (FP32 layers start as zero nn.Parameter, no init needed.)"""
        from calm.llm_computer.tq4_torch import Tq4Tensor
        cfg = self._hybrid_config
        for l in range(cfg.n_layers):
            if cfg.layer_linear_types[l] != "tq4":
                continue
            for linear in (self.W_qkv[l], self.W_out[l], self.ff_in[l],
                           self.ff_out[l]):
                in_f = linear.in_features
                out_f = linear.out_features
                total = in_f * out_f
                n_blocks = total // HEAD_DIM
                qs = torch.zeros(n_blocks, 128, dtype=torch.uint8)
                d = torch.zeros(n_blocks, dtype=torch.float32)
                linear.install_tq4(
                    Tq4Tensor(qs=qs, d=d, shape=(in_f, out_f))
                )


def install_compiled_card_hybrid(
    substrate: HybridGroupedSmall2DTransformer,
    card,
    ch_off: int,
    sh_off: int,
    ffn_off: int,
    tok_off: int,
    layer_off: int,
) -> None:
    """Install a compiled card into FP32 layers of the hybrid substrate.

    The card's standard PyTorch Linear weights (out, in) are TRANSPOSED
    into the substrate's GGML-oriented (in, out) slots. Requires all
    target layers to be "fp32" — installing into tq4 would require
    quantization and incur the compiled-card correctness loss that
    Round 11 diagnosed.
    """
    cfg = substrate.config
    c_cfg = card.config
    D_s = cfg.d_model
    F_s = cfg.d_ffn
    D_c = c_cfg.d_model
    H_c = c_cfg.n_heads
    F_c = c_cfg.d_ffn
    V_c = c_cfg.vocab_size

    for l in range(c_cfg.n_layers):
        s_l = layer_off + l
        assert cfg.layer_linear_types[s_l] == "fp32", (
            f"card layer {l} → substrate layer {s_l} is "
            f"{cfg.layer_linear_types[s_l]}, must be 'fp32'"
        )

    with torch.no_grad():
        substrate.tok.weight[
            tok_off : tok_off + V_c, ch_off : ch_off + D_c,
        ] = card.tok.weight
        pos_rows = min(c_cfg.max_len, cfg.max_len)
        substrate.pos.weight[
            :pos_rows, ch_off : ch_off + D_c,
        ] = card.pos.weight[:pos_rows]

        for l in range(c_cfg.n_layers):
            s_l = layer_off + l
            # Transpose card's (out=3*D_c, in=D_c) → (in=D_c, out=3*D_c)
            qkv_c = card.W_qkv[l].weight.T.contiguous()
            qkv_s = substrate.W_qkv[s_l].weight  # (D_s, 3*D_s)
            # Q cols: [2*sh_off, 2*sh_off + D_c) in first D_s cols of substrate.
            # K cols: [D_s + 2*sh_off, ...]; V at [2*D_s + 2*sh_off, ...].
            qkv_s[ch_off : ch_off + D_c,
                  2 * sh_off : 2 * sh_off + D_c] = qkv_c[:, 0:D_c]
            qkv_s[ch_off : ch_off + D_c,
                  D_s + 2 * sh_off : D_s + 2 * sh_off + D_c] = \
                qkv_c[:, D_c : 2 * D_c]
            qkv_s[ch_off : ch_off + D_c,
                  2 * D_s + 2 * sh_off : 2 * D_s + 2 * sh_off + D_c] = \
                qkv_c[:, 2 * D_c : 3 * D_c]

            # W_out: card (out=D_c, in=D_c) → substrate (in=D_s, out=D_s)
            # Transposed: (in=D_c, out=D_c) → placed at rows [2*sh_off, +D_c),
            # cols [ch_off, +D_c).
            out_c = card.W_out[l].weight.T.contiguous()
            substrate.W_out[s_l].weight[
                2 * sh_off : 2 * sh_off + D_c,
                ch_off : ch_off + D_c,
            ] = out_c

            # ff_in: card (out=2*F_c, in=D_c) → substrate (in=D_s, out=2*F_s)
            # Card gate rows [0, F_c), val rows [F_c, 2*F_c) (post-chunk).
            # Substrate gate cols [0, F_s), val cols [F_s, 2*F_s).
            ff_in_c = card.ff_in[l].weight.T.contiguous()  # (D_c, 2*F_c)
            ff_in_s = substrate.ff_in[s_l].weight           # (D_s, 2*F_s)
            # Card gate cols [0, F_c) → substrate gate cols [ffn_off, ffn_off+F_c)
            ff_in_s[ch_off : ch_off + D_c,
                    ffn_off : ffn_off + F_c] = ff_in_c[:, 0:F_c]
            # Card val cols [F_c, 2*F_c) → substrate val cols
            # [F_s + ffn_off, F_s + ffn_off + F_c).
            ff_in_s[ch_off : ch_off + D_c,
                    F_s + ffn_off : F_s + ffn_off + F_c] = ff_in_c[:, F_c:]

            # ff_out: card (out=D_c, in=F_c) → substrate (in=F_s, out=D_s)
            ff_out_c = card.ff_out[l].weight.T.contiguous()  # (F_c, D_c)
            substrate.ff_out[s_l].weight[
                ffn_off : ffn_off + F_c,
                ch_off : ch_off + D_c,
            ] = ff_out_c

        # Head is FP32, GGML-oriented (in=D_s, out=V_s). Card head is
        # (V_c, D_c). Transpose to (D_c, V_c) → place at rows [ch_off, +D_c),
        # cols [tok_off, +V_c).
        head_c = card.head.weight.T.contiguous()
        substrate.head.weight[
            ch_off : ch_off + D_c,
            tok_off : tok_off + V_c,
        ] = head_c
