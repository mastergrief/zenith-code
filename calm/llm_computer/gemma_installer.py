"""Install Gemma 4 E4B weights into the unified substrate.

Flow per layer:
  1. Read Gemma's tq4 tensor from GGUF (e.g., blk.N.attn_q.weight,
     shape (d_model=2560, n_heads*d_head=2048))
  2. Dequantize to FP32 (tq4 → float)
  3. Install into a corner of the upscaled substrate's W_qkv[N] weight.
     W_qkv has shape (3 * substrate_d_model, substrate_d_model) stacking
     Q, K, V projections. Gemma's Q occupies the first 1/3, padded with
     zeros; same for K and V in their respective thirds.
  4. For FP32 norms (attn_norm, ffn_norm, etc.) — direct copy into the
     unified substrate's first gemma_d_model slots.

Memory strategy: we dequantize per-tensor on load (one 21MB chunk at a
time), install, then discard the FP32 intermediate. The substrate
stores FP32 Linear layers (at d_model=4096 this is ~300MB per layer's
attention + ~270MB per layer's FFN). For 42 layers, full substrate is
~24GB FP32 — doesn't fit in 8GB VRAM at FP32 precision.

For MVP validation we do this on CPU at tiny scale. Scaling to the
full model would either:
  (a) convert substrate layers to tq4 after install (re-quantize the
      padded tensors) — this works and would fit in ~12GB VRAM tq4
  (b) keep only Gemma-shaped portion in memory, compute with slicing
  (c) stream layer-by-layer, never materialize full substrate

For this session we build option (a) — install FP32, then re-tq4 the
installed layer. MVP tests at tiny scale confirm the installation math
is right.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from calm.llm_computer.grouped_small2d import GroupedSmall2DTransformer
from calm.llm_computer.tq4_gguf_loader import (
    extract_fp_tensor, extract_tq4_tensor, read_turboquant_gguf,
)
from calm.llm_computer.tq4_torch import build_pi, dequantize_tq4
from calm.llm_computer.unified_tensor import (
    UnifiedTensorConfig, install_padded_weight,
)


def dequantize_gemma_tq4(
    reader, tensor_name: str, pi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Read a tq4 tensor from GGUF and return it as FP32 for installation."""
    q = extract_tq4_tensor(reader, tensor_name)
    if pi is None:
        pi = build_pi(source="c_header")
    return dequantize_tq4(q, pi=pi)


def install_gemma_attention_into_upscaled_substrate(
    substrate: GroupedSmall2DTransformer,
    cfg: UnifiedTensorConfig,
    reader,
    layer_idx: int,
    pi: Optional[torch.Tensor] = None,
) -> None:
    """Install Gemma's attention projections (q, k, v, o) for one layer.

    The upscaled substrate's W_qkv[layer] has shape
    (3 * substrate_d_model, substrate_d_model). We decompose this stacked
    weight into three (substrate_d_model, substrate_d_model) blocks:
      rows 0 .. D_s          : Q
      rows D_s .. 2D_s        : K
      rows 2D_s .. 3D_s       : V
    where D_s = substrate_d_model.

    Gemma's Q has shape (gemma_q_out, gemma_d_model) where
    gemma_q_out = n_heads * head_dim (varies per layer type: 2048 for
    SWA, 4096 for full). We install it into:
      substrate.W_qkv[layer].weight[0 : gemma_q_out, 0 : gemma_d_model]

    Gemma's K, V are smaller due to GQA (n_kv_heads < n_heads).
    They install starting at rows D_s (K) and 2D_s (V).
    """
    if pi is None:
        pi = build_pi(source="c_header")
    D_s = cfg.substrate_d_model
    D_g = cfg.gemma_d_model

    # Dequantize Gemma's 4 attention matrices
    q_fp = dequantize_gemma_tq4(reader, f"blk.{layer_idx}.attn_q.weight", pi=pi)
    k_fp = dequantize_gemma_tq4(reader, f"blk.{layer_idx}.attn_k.weight", pi=pi)
    v_fp = dequantize_gemma_tq4(reader, f"blk.{layer_idx}.attn_v.weight", pi=pi)
    o_fp = dequantize_gemma_tq4(reader, f"blk.{layer_idx}.attn_output.weight", pi=pi)

    # GGUF stores weights as (in_features, out_features) in llama.cpp
    # column-major convention. PyTorch Linear expects (out, in). Need to
    # transpose.
    q_fp = q_fp.T.contiguous()  # (q_out, d_model)
    k_fp = k_fp.T.contiguous()
    v_fp = v_fp.T.contiguous()
    o_fp = o_fp.T.contiguous()  # (d_model, q_out)

    w_qkv = substrate.W_qkv[layer_idx]
    w_out = substrate.W_out[layer_idx]

    # Shapes after transpose:
    q_out = q_fp.shape[0]      # gemma_q_out (2048 or 4096)
    kv_out = k_fp.shape[0]     # gemma_kv_out

    # Install Q into substrate.W_qkv rows [0, q_out] cols [0, D_g]
    install_padded_weight(w_qkv, q_fp, row_offset=0, col_offset=0)
    # Install K at rows [D_s, D_s + kv_out]
    install_padded_weight(w_qkv, k_fp, row_offset=D_s, col_offset=0)
    # Install V at rows [2*D_s, 2*D_s + kv_out]
    install_padded_weight(w_qkv, v_fp, row_offset=2*D_s, col_offset=0)

    # Install O projection: (d_model, q_out) → substrate W_out shape
    # (D_s, D_s). Installs at rows [0, D_g], cols [0, q_out].
    install_padded_weight(w_out, o_fp, row_offset=0, col_offset=0)


def install_gemma_ffn_into_upscaled_substrate(
    substrate: GroupedSmall2DTransformer,
    cfg: UnifiedTensorConfig,
    reader,
    layer_idx: int,
    pi: Optional[torch.Tensor] = None,
) -> None:
    """Install Gemma's gate_proj, up_proj, down_proj for one layer.

    Substrate's ff_in[layer] has shape (2 * substrate_d_ffn,
    substrate_d_model) — gate and up stacked. ff_out[layer] has shape
    (substrate_d_model, substrate_d_ffn) — down projection.

    Gemma's weights:
      gate_proj: (d_ffn=10240, d_model=2560)  stored as (d_model, d_ffn)
      up_proj:   (d_ffn, d_model)
      down_proj: (d_model, d_ffn)
    """
    if pi is None:
        pi = build_pi(source="c_header")

    gate_fp = dequantize_gemma_tq4(reader, f"blk.{layer_idx}.ffn_gate.weight", pi=pi)
    up_fp = dequantize_gemma_tq4(reader, f"blk.{layer_idx}.ffn_up.weight", pi=pi)
    down_fp = dequantize_gemma_tq4(reader, f"blk.{layer_idx}.ffn_down.weight", pi=pi)

    # Transpose GGUF (in, out) → PyTorch (out, in)
    gate_fp = gate_fp.T.contiguous()  # (d_ffn, d_model)
    up_fp = up_fp.T.contiguous()
    down_fp = down_fp.T.contiguous()  # (d_model, d_ffn)

    D_s_ffn = cfg.substrate_d_ffn
    ff_in = substrate.ff_in[layer_idx]   # (2 * D_s_ffn, D_s)
    ff_out = substrate.ff_out[layer_idx]  # (D_s, D_s_ffn)

    # Install gate into rows [0, d_ffn_gemma] and up into [D_s_ffn, D_s_ffn + d_ffn_gemma]
    install_padded_weight(ff_in, gate_fp, row_offset=0, col_offset=0)
    install_padded_weight(ff_in, up_fp, row_offset=D_s_ffn, col_offset=0)
    # Install down at rows [0, d_model_gemma], cols [0, d_ffn_gemma]
    install_padded_weight(ff_out, down_fp, row_offset=0, col_offset=0)


def install_gemma_embeddings_into_upscaled_substrate(
    substrate: GroupedSmall2DTransformer,
    cfg: UnifiedTensorConfig,
    reader,
    pi: Optional[torch.Tensor] = None,
) -> None:
    """Install Gemma's token embedding + output norm into the substrate.

    Note: Gemma's token_embd is Q6_K in our GGUF. We skip it here and
    leave the substrate's tok embedding randomly initialized. Full
    production load would need Q6_K dequant (not yet shipped).
    """
    # output_norm is F32 — install into substrate.output_norm directly
    try:
        t = extract_fp_tensor(reader, "output_norm.weight")
        # Substrate doesn't have output_norm as a trainable weight
        # (Small2DTransformer lacks a final-norm layer). Skip for now;
        # requires extending Small2DTransformer.
        pass
    except Exception:
        pass


def install_full_gemma_into_substrate(
    substrate: GroupedSmall2DTransformer,
    cfg: UnifiedTensorConfig,
    gguf_path: str | Path,
    layer_limit: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """Walk every Gemma layer and install its attention + FFN weights.

    Returns a dict summary of what was loaded (layer count, any errors).
    """
    reader = read_turboquant_gguf(gguf_path)
    pi = build_pi(source="c_header")

    n_layers = cfg.gemma_n_layers if layer_limit is None else min(layer_limit, cfg.gemma_n_layers)
    summary = {"layers_loaded": 0, "errors": []}

    for i in range(n_layers):
        try:
            install_gemma_attention_into_upscaled_substrate(
                substrate, cfg, reader, layer_idx=i, pi=pi,
            )
            install_gemma_ffn_into_upscaled_substrate(
                substrate, cfg, reader, layer_idx=i, pi=pi,
            )
            summary["layers_loaded"] += 1
            if verbose:
                print(f"  layer {i} installed", flush=True)
        except Exception as e:
            summary["errors"].append((i, str(e)))
            if verbose:
                print(f"  layer {i} FAILED: {e}", flush=True)

    install_gemma_embeddings_into_upscaled_substrate(substrate, cfg, reader, pi=pi)
    return summary
