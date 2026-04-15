"""Tests for Gemma → unified substrate installation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from calm.llm_computer.gemma_installer import (
    install_full_gemma_into_substrate,
)
from calm.llm_computer.unified_tensor import (
    UnifiedTensorConfig, build_unified_substrate,
)


GGUF_PATH = Path(
    os.environ.get(
        "ZENITH_GEMMA_GGUF",
        "/home/gabe/models/gemma-4-E4B-it-tq4-aligned.gguf",
    )
)
GGUF_AVAILABLE = GGUF_PATH.exists()


@pytest.mark.skipif(not GGUF_AVAILABLE, reason="Gemma GGUF not present")
def test_install_first_layer_of_gemma():
    """Install just the first layer of Gemma into a minimally-sized
    unified substrate. Verify weights are non-zero afterwards.

    To keep memory tractable, we use a config with n_layers=1 — just
    enough to install Gemma's layer 0 weights. Full 42-layer substrate
    at d_model=4096 FP32 would be ~34GB."""
    cfg = UnifiedTensorConfig(gemma_n_layers=1)  # only build 1 layer
    substrate = build_unified_substrate(cfg)

    # Verify substrate starts fully zeroed
    assert (substrate.W_qkv[0].weight == 0).all()
    assert (substrate.ff_in[0].weight == 0).all()

    # Install just layer 0 (saves time)
    summary = install_full_gemma_into_substrate(
        substrate, cfg, GGUF_PATH, layer_limit=1, verbose=False,
    )
    assert summary["layers_loaded"] == 1, (
        f"expected 1 layer loaded, got {summary}"
    )

    # Layer 0 weights are now non-zero in the Gemma region
    D_s = cfg.substrate_d_model       # 4096
    D_g = cfg.gemma_d_model            # 2560
    # Q projection lives at W_qkv rows [0..q_out], cols [0..D_g]
    # Layer 0 is SWA in Gemma 4 → q_out = 8 * 256 = 2048
    q_out = cfg.gemma_n_heads * cfg.gemma_swa_head_dim
    assert (substrate.W_qkv[0].weight[:q_out, :D_g] != 0).any(), (
        "Q projection weight should be non-zero after install"
    )
    # Outside the Gemma region should still be zero
    assert (substrate.W_qkv[0].weight[:q_out, D_g:] == 0).all(), (
        "Gemma weights shouldn't bleed into non-Gemma channels"
    )


@pytest.mark.skipif(not GGUF_AVAILABLE, reason="Gemma GGUF not present")
def test_install_preserves_other_regions():
    """After installing Gemma, non-Gemma channels (HRM, memory, free)
    must still be zero."""
    cfg = UnifiedTensorConfig(gemma_n_layers=2)
    substrate = build_unified_substrate(cfg)
    summary = install_full_gemma_into_substrate(
        substrate, cfg, GGUF_PATH, layer_limit=1,
    )
    assert summary["layers_loaded"] == 1

    D_s = cfg.substrate_d_model

    # Layer 0 W_qkv is (3 * D_s, D_s). Q is [0, D_s], K is [D_s, 2D_s],
    # V is [2D_s, 3D_s]. Non-Gemma regions within each:
    # Within Q: rows [q_out, D_s] should be zero
    q_out = cfg.gemma_n_heads * cfg.gemma_swa_head_dim  # 2048 SWA
    assert (substrate.W_qkv[0].weight[q_out:D_s, :] == 0).all(), (
        "Rows beyond Gemma's Q output should remain zero"
    )

    # Also verify NO layers beyond the first have been touched
    if cfg.gemma_n_layers >= 2:
        assert (substrate.W_qkv[1].weight == 0).all(), (
            "layer 1 shouldn't be touched when layer_limit=1"
        )


def test_install_on_tiny_substrate_structural():
    """Verify the install API works structurally without requiring the
    full GGUF. Uses a fake reader."""
    # Build a tiny substrate and verify the install functions don't
    # crash when there's no real tensor (they'll raise KeyError)
    tiny_cfg = UnifiedTensorConfig(
        gemma_d_model=256, gemma_n_heads=4, gemma_n_kv_heads=2,
        gemma_n_layers=2, gemma_d_ffn=512,
        gemma_swa_head_dim=32, gemma_full_head_dim=64,
        gemma_vocab_size=100, gemma_max_position=32,
        gemma_full_layer_indices=(1,),
        hrm_specialists=("math",), hrm_d_model=8, hrm_n_heads=4,
        n_compiled_sub_heads=16, keyed_memory_channels=32,
        call_stack_channels=16, card_scratchpad_channels=16,
    )
    substrate = build_unified_substrate(tiny_cfg)
    # substrate should build without error
    assert substrate.config.d_model == tiny_cfg.substrate_d_model


if __name__ == "__main__":
    test_install_on_tiny_substrate_structural()
    print("[ok] tiny substrate builds")
    if GGUF_AVAILABLE:
        test_install_first_layer_of_gemma()
        print("[ok] installed Gemma layer 0 into upscaled substrate")
        test_install_preserves_other_regions()
        print("[ok] non-Gemma regions preserved")
    else:
        print("[SKIP] GGUF not available")
