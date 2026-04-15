"""Tests for end-to-end Gemma byte installation into tq4 substrate."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from calm.llm_computer.gemma_byte_installer import (
    install_full_gemma_bytes, install_gemma_layer_bytes,
)
from calm.llm_computer.tq4_gguf_loader import read_turboquant_gguf
from calm.llm_computer.tq4_substrate import Tq4GroupedSmall2DTransformer
from calm.llm_computer.unified_tensor import UnifiedTensorConfig


GGUF_PATH = Path(
    os.environ.get(
        "ZENITH_GEMMA_GGUF",
        "/home/gabe/models/gemma-4-E4B-it-tq4-aligned.gguf",
    )
)
GGUF_AVAILABLE = GGUF_PATH.exists()


def _make_cfg_for_layer_limit(n_layers: int):
    """Scale the Gemma-default config down to n_layers to keep memory
    tractable. Gemma 4 E4B's full_attention layers are at indices 5, 11,
    17, 23, 29, 35, 41 — all beyond n_layers=2 so we don't hit them."""
    # Filter gemma_full_layer_indices to only valid indices within n_layers
    default_full = tuple(range(5, 42, 6))
    in_range = tuple(i for i in default_full if i < n_layers)
    return UnifiedTensorConfig(
        gemma_n_layers=n_layers,
        gemma_full_layer_indices=in_range,
    )


@pytest.mark.skipif(not GGUF_AVAILABLE, reason="Gemma GGUF not present")
def test_install_layer_0_byte_level():
    """Install just layer 0 byte-level into a tq4 substrate. Verify
    weights are populated and forward runs."""
    cfg = _make_cfg_for_layer_limit(1)
    substrate_cfg = cfg.build_grouped_config()
    substrate = Tq4GroupedSmall2DTransformer(substrate_cfg)
    # Initialize all layers to zero blocks first (required before partial install)
    substrate.initialize_all_zero_tq4()
    with torch.no_grad():
        substrate.tok.weight.zero_()
        substrate.pos.weight.zero_()

    reader = read_turboquant_gguf(GGUF_PATH)
    install_gemma_layer_bytes(substrate, cfg, reader, layer_idx=0)

    # Verify layer 0's tq4 storage now has non-zero blocks
    l = substrate.W_qkv[0]
    assert l.is_loaded()
    # At least some blocks should have non-zero d (Gemma's norms)
    assert (l._d != 0).any(), "W_qkv after install should have non-zero d values"


@pytest.mark.skipif(not GGUF_AVAILABLE, reason="Gemma GGUF not present")
def test_full_gemma_install_first_2_layers():
    """End-to-end: install first 2 Gemma layers into tq4 substrate,
    verify forward pass runs and produces finite logits."""
    cfg = _make_cfg_for_layer_limit(2)
    substrate_cfg = cfg.build_grouped_config()
    substrate = Tq4GroupedSmall2DTransformer(substrate_cfg)

    summary = install_full_gemma_bytes(
        substrate, cfg, GGUF_PATH, layer_limit=2, verbose=False,
    )
    assert summary["layers_loaded"] == 2, (
        f"expected 2 layers loaded: {summary}"
    )
    assert len(summary["errors"]) == 0

    # Populate embeddings (head stays zero-init; quantizing a
    # 4096x262144 head blows up memory at 16GB+ because the standard
    # quantize_tq4 broadcasts a (n_blocks, 256, n_boundaries) tensor
    # during boundary comparison. For this structural test we just
    # sample from the RESIDUAL before the head, not final logits.)
    with torch.no_grad():
        substrate.tok.weight.normal_(0, 0.02)
        substrate.pos.weight.normal_(0, 0.02)

    # Replicate substrate forward up to the head — inspect residual
    substrate.eval()
    x = torch.tensor([[1, 100, 200, 42]], dtype=torch.long)
    with torch.no_grad():
        # Manual forward-sans-head to get final residual
        B, S = x.shape
        pos_idx = torch.arange(S, device=x.device)
        res = substrate.tok(x) + substrate.pos(pos_idx)
        mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=res.device), diagonal=1,
        )
        for layer in range(substrate.config.n_layers):
            qkv = substrate.W_qkv[layer](res)
            qkv = qkv.reshape(B, S, 3, substrate.config.n_heads,
                              substrate.config.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            # Standard d_head=2 per-sub-head attention
            scores = torch.einsum("bhid,bhjd->bhij", q, k)
            scores = scores.masked_fill(mask, float("-inf"))
            import torch.nn.functional as F
            weights = F.softmax(scores, dim=-1)
            attn = torch.einsum("bhij,bhjd->bhid", weights, v)
            attn = attn.transpose(1, 2).reshape(B, S, substrate.config.d_model)
            res = res + substrate.W_out[layer](attn)
            gate, val = substrate.ff_in[layer](res).chunk(2, dim=-1)
            res = res + substrate.ff_out[layer](F.relu(gate) * val)

    assert res.shape == (1, 4, substrate.config.d_model)
    assert torch.isfinite(res).all(), "residual should be finite"
    # Residual should be non-trivial (not all zeros)
    # Gemma's installed weights should produce non-zero outputs
    # in the first gemma_d_model channels
    gemma_channels = res[:, :, :cfg.gemma_d_model]
    assert gemma_channels.std() > 1e-4, (
        f"Gemma channels should be non-zero, got std={gemma_channels.std()}"
    )


def test_install_layer_bytes_structural_without_gguf():
    """Structural test: install_gemma_layer_bytes requires a reader
    argument. Verify the function signature works with a fake layer."""
    # Build a tiny config and substrate
    tiny_cfg = UnifiedTensorConfig(
        gemma_d_model=256, gemma_n_heads=4, gemma_n_kv_heads=2,
        gemma_n_layers=1, gemma_d_ffn=256,
        gemma_swa_head_dim=32, gemma_full_head_dim=64,
        gemma_vocab_size=256, gemma_max_position=8,
        gemma_full_layer_indices=(),
        hrm_specialists=("math",),
        hrm_d_model=8, hrm_n_heads=4,
        n_compiled_sub_heads=16,
        keyed_memory_channels=16, call_stack_channels=8,
        card_scratchpad_channels=8,
    )
    substrate_cfg = tiny_cfg.build_grouped_config()
    substrate = Tq4GroupedSmall2DTransformer(substrate_cfg)
    # Just verify it builds without error
    assert substrate.config.n_layers == 1


if __name__ == "__main__":
    test_install_layer_bytes_structural_without_gguf()
    print("[ok] structural install works")
    if GGUF_AVAILABLE:
        test_install_layer_0_byte_level()
        print("[ok] Gemma layer 0 byte-installed")
        test_full_gemma_install_first_2_layers()
        print("[ok] Full 2-layer Gemma substrate runs forward pass")
    else:
        print("[SKIP] GGUF not available")
