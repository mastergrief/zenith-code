"""Install SubstrateHRM weights into the unified substrate's reserved
sub-head and channel slots.

SubstrateHRM is a decoder-only Small2DTransformer with d_model=32,
n_heads=16, trained to emit `<problem> =` structure from NL inputs.

Installation: since SubstrateHRM is substrate-native (it's also a
Small2DTransformer), the weights are directly compatible — we just
position them in the reserved sub-head range of the unified substrate.

For each layer of the HRM checkpoint, install:
  - W_qkv rows corresponding to HRM's sub-head range (e.g., 1024..1039
    for math HRM in the upscaled substrate)
  - W_out rows corresponding to HRM's output channel range (e.g.,
    2560..2591 for math HRM's residual slice)
  - ff_in / ff_out into reserved FFN neurons
  - tok / pos: embedded in the HRM's channel range

Important: HRM's d_model=32 is much smaller than the substrate's 4096.
HRM's Q/K/V projections go from 32 input channels (its reserved slice)
into 32-channel output (16 sub-heads × 2). So installation is a
corner-patch into specific rows and columns of the larger substrate
weights.

This MVP wires up the mechanics; a full install walks all layers of
an HRM checkpoint and places each one correctly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from calm.llm_computer.grouped_small2d import GroupedSmall2DTransformer
from calm.llm_computer.model import Small2DConfig, Small2DTransformer
from calm.llm_computer.unified_tensor import UnifiedTensorConfig


def install_hrm_into_substrate(
    substrate: GroupedSmall2DTransformer,
    cfg: UnifiedTensorConfig,
    hrm: Small2DTransformer,
    hrm_name: str,
    layer_idx: int,
) -> None:
    """Install one HRM specialist's layer `layer_idx` weights into the
    unified substrate at its reserved channels + sub-heads.

    Args:
        substrate: the unified GroupedSmall2DTransformer
        cfg: UnifiedTensorConfig with HRM partition info
        hrm: SubstrateHRM Small2DTransformer checkpoint
        hrm_name: which specialist ("math", "nl", etc.)
        layer_idx: which substrate layer receives this HRM layer

    The HRM has its own n_layers (typically 2-5). We install the HRM's
    layer_N into the substrate's layer_N for each N in range(n_layers).
    This function handles just one layer; full install calls it per-layer.
    """
    if hrm_name not in cfg.hrm_channels:
        raise KeyError(f"HRM '{hrm_name}' not in config.hrm_specialists")

    # HRM's residual slice in the substrate
    ch_lo, ch_hi = cfg.hrm_channels[hrm_name]
    sh_lo, sh_hi = cfg.hrm_sub_heads[hrm_name]
    hrm_d_model = ch_hi - ch_lo         # e.g., 32
    hrm_n_heads = sh_hi - sh_lo          # e.g., 16
    assert hrm_d_model == hrm.config.d_model, (
        f"HRM d_model {hrm.config.d_model} != reserved {hrm_d_model}"
    )
    assert hrm_n_heads == hrm.config.n_heads, (
        f"HRM n_heads {hrm.config.n_heads} != reserved {hrm_n_heads}"
    )

    # HRM's W_qkv[layer] has shape (3 * hrm_d_model, hrm_d_model)
    # The substrate's W_qkv[layer] has shape (3 * D_s, D_s)
    # We install:
    #   HRM's Q rows [0, hrm_d_model] →
    #     substrate rows [2 * sh_lo, 2 * sh_lo + hrm_d_model] (Q segment)
    # Wait — the substrate W_qkv stacks Q, K, V as:
    #   Q: rows [0, D_s]
    #   K: rows [D_s, 2*D_s]
    #   V: rows [2*D_s, 3*D_s]
    # Within each segment, sub-head i occupies rows [2i, 2i+2] (d_head=2).
    # So HRM's sub-head range [sh_lo, sh_hi] maps to substrate rows
    # [2*sh_lo, 2*sh_hi] within each Q/K/V segment.
    # Columns are channels: HRM's inputs are its reserved channels [ch_lo, ch_hi]
    D_s = cfg.substrate_d_model

    with torch.no_grad():
        # Q install
        substrate.W_qkv[layer_idx].weight[
            2*sh_lo : 2*sh_hi, ch_lo:ch_hi,
        ] = hrm.W_qkv[layer_idx].weight[0:hrm_d_model, :]
        # K install
        substrate.W_qkv[layer_idx].weight[
            D_s + 2*sh_lo : D_s + 2*sh_hi, ch_lo:ch_hi,
        ] = hrm.W_qkv[layer_idx].weight[hrm_d_model:2*hrm_d_model, :]
        # V install
        substrate.W_qkv[layer_idx].weight[
            2*D_s + 2*sh_lo : 2*D_s + 2*sh_hi, ch_lo:ch_hi,
        ] = hrm.W_qkv[layer_idx].weight[2*hrm_d_model:3*hrm_d_model, :]

        # W_out: HRM's W_out is (hrm_d_model, hrm_d_model). The substrate's
        # W_out is (D_s, D_s). HRM writes back into its reserved channels
        # via rows [ch_lo, ch_hi], reading from sub-head outputs [2sh_lo, 2sh_hi].
        substrate.W_out[layer_idx].weight[
            ch_lo:ch_hi, 2*sh_lo:2*sh_hi,
        ] = hrm.W_out[layer_idx].weight

        # FFN: HRM has its own d_ffn. For MVP we allocate HRM's FFN neurons
        # at the end of the substrate's ff region, beyond Gemma's d_ffn use.
        # Deferred: FFN install needs a careful allocation scheme; skipping
        # for this MVP so we can test the attention path first.


def install_hrm_full(
    substrate: GroupedSmall2DTransformer,
    cfg: UnifiedTensorConfig,
    hrm: Small2DTransformer,
    hrm_name: str,
) -> None:
    """Install all layers of an HRM specialist into the substrate.

    Maps HRM layer i → substrate layer i. The substrate must have at
    least as many layers as the HRM. HRMs typically have 2-5 layers,
    much less than substrate's 42.
    """
    if hrm.config.n_layers > substrate.config.n_layers:
        raise ValueError(
            f"HRM has more layers ({hrm.config.n_layers}) than substrate "
            f"({substrate.config.n_layers})"
        )
    for i in range(hrm.config.n_layers):
        install_hrm_into_substrate(substrate, cfg, hrm, hrm_name, layer_idx=i)


def build_tiny_hrm_for_testing(
    d_model: int, n_heads: int, n_layers: int = 2,
) -> Small2DTransformer:
    """Create a tiny SubstrateHRM-shaped model with random weights for
    testing the installer without needing real checkpoints."""
    cfg = Small2DConfig(
        vocab_size=32, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=64, max_len=16, use_hard_max=False,
    )
    m = Small2DTransformer(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.1)
    return m
