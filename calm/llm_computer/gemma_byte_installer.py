"""End-to-end byte-level installer: Gemma GGUF → Tq4GroupedSmall2DTransformer.

Walks every Gemma layer, extracts tq4 tensors from the GGUF, and
installs them byte-level into the substrate's Tq4LinearGGMLOriented
weights. No dequantization, no re-quantization.

Memory-wise this is efficient: each tq4 tensor is ~21 KB / 256 elements
(vs FP32's 1 MB / 256 elements). A full 42-layer Gemma substrate fits
in ~4.4 GB on disk vs 34 GB FP32.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from calm.llm_computer.tq4_gguf_loader import (
    extract_tq4_tensor, read_turboquant_gguf,
)
from calm.llm_computer.tq4_substrate import (
    Tq4GroupedSmall2DTransformer,
    install_ffn_in_from_parts,
    install_qkv_from_parts,
    install_simple_tq4_corner,
)
from calm.llm_computer.unified_tensor import UnifiedTensorConfig


def install_gemma_layer_bytes(
    substrate: Tq4GroupedSmall2DTransformer,
    cfg: UnifiedTensorConfig,
    reader,
    layer_idx: int,
) -> None:
    """Install one Gemma layer's attention + FFN weights byte-level
    into the tq4 substrate. No dequant.

    For layer_idx in gemma_full_layer_indices, Gemma uses head_dim=512
    (full attention). Otherwise head_dim=256 (SWA). Both fit because
    we byte-install Q/K/V into a substrate W_qkv shaped (D_s, 3*D_s).
    """
    D_s = cfg.substrate_d_model
    D_ffn_s = cfg.substrate_d_ffn

    # Extract all 7 tq4 tensors for this layer
    q = extract_tq4_tensor(reader, f"blk.{layer_idx}.attn_q.weight")
    k = extract_tq4_tensor(reader, f"blk.{layer_idx}.attn_k.weight")
    v = extract_tq4_tensor(reader, f"blk.{layer_idx}.attn_v.weight")
    o = extract_tq4_tensor(reader, f"blk.{layer_idx}.attn_output.weight")
    gate = extract_tq4_tensor(reader, f"blk.{layer_idx}.ffn_gate.weight")
    up = extract_tq4_tensor(reader, f"blk.{layer_idx}.ffn_up.weight")
    down = extract_tq4_tensor(reader, f"blk.{layer_idx}.ffn_down.weight")

    # Install W_qkv: stacked Q|K|V
    install_qkv_from_parts(
        substrate.W_qkv[layer_idx], q, k, v,
        substrate_d_model=D_s,
    )

    # Install W_out: attn_output. Gemma's shape is (n_heads * head_dim, d_model)
    # = (q_out, d_model). In GGUF (in, out) convention: (in=q_out, out=d_model).
    # Substrate W_out is (D_s, D_s) (in=D_s, out=D_s). We install at corner.
    install_simple_tq4_corner(
        substrate.W_out[layer_idx], o,
        target_in=D_s, target_out=D_s,
    )

    # Install ff_in: stacked gate|up. Both have GGUF shape (d_model, d_ffn).
    install_ffn_in_from_parts(
        substrate.ff_in[layer_idx], gate, up,
        substrate_d_model=D_s, substrate_d_ffn=D_ffn_s,
    )

    # Install ff_out: down_proj. GGUF shape (d_ffn, d_model).
    # Substrate ff_out is (D_ffn_s, D_s). Corner install.
    install_simple_tq4_corner(
        substrate.ff_out[layer_idx], down,
        target_in=D_ffn_s, target_out=D_s,
    )


def install_full_gemma_bytes(
    substrate: Tq4GroupedSmall2DTransformer,
    cfg: UnifiedTensorConfig,
    gguf_path: str | Path,
    layer_limit: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """Walk every Gemma layer, byte-install attention + FFN into the
    tq4 substrate. Returns a summary dict of loaded layers + errors."""
    reader = read_turboquant_gguf(gguf_path)
    n_layers = cfg.gemma_n_layers if layer_limit is None else min(
        layer_limit, cfg.gemma_n_layers,
    )
    summary = {"layers_loaded": 0, "errors": []}

    # First: initialize all zero blocks in every layer so the forward
    # pass can run even with partial installation
    substrate.initialize_all_zero_tq4()

    for i in range(n_layers):
        try:
            install_gemma_layer_bytes(substrate, cfg, reader, layer_idx=i)
            summary["layers_loaded"] += 1
            if verbose:
                print(f"  layer {i} byte-installed", flush=True)
        except Exception as e:
            summary["errors"].append((i, str(e)))
            if verbose:
                print(f"  layer {i} FAILED: {e}", flush=True)
    return summary
