"""Tests for Gemma 4 E4B stream + heterogeneous config + GGUF load."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from calm.llm_computer.gemma4_config import (
    Gemma4Config, Gemma4LayerConfig, gemma4_e4b_config,
)
from calm.llm_computer.gemma4_stream import Gemma4Layer, Gemma4Stream


GGUF_PATH = Path(
    os.environ.get(
        "ZENITH_GEMMA_GGUF",
        "/home/gabe/models/gemma-4-E4B-it-tq4-aligned.gguf",
    )
)
GGUF_AVAILABLE = GGUF_PATH.exists()


# ----- Gemma4Config -----

def test_full_attention_layers_default():
    cfg = gemma4_e4b_config()
    # Every 6th layer starting at 5: 5, 11, 17, 23, 29, 35, 41
    assert cfg.full_attention_layers == (5, 11, 17, 23, 29, 35, 41)


def test_layer_config_swa():
    cfg = gemma4_e4b_config()
    lc = cfg.layer_config(0)
    assert isinstance(lc, Gemma4LayerConfig)
    assert not lc.is_full_attention
    assert lc.head_dim == 256
    assert lc.rope_freq_base == 10000.0
    assert lc.sliding_window == 512
    assert lc.attention_type == "swa"


def test_layer_config_full():
    cfg = gemma4_e4b_config()
    lc = cfg.layer_config(5)
    assert lc.is_full_attention
    assert lc.head_dim == 512
    assert lc.rope_freq_base == 1_000_000.0
    assert lc.sliding_window is None
    assert lc.attention_type == "full"


def test_projection_sizes():
    cfg = gemma4_e4b_config()
    # SWA layer 0: n_heads=8 * head_dim=256 = 2048
    assert cfg.q_proj_out(0) == 2048
    # n_kv_heads=2 * 256 = 512
    assert cfg.kv_proj_out(0) == 512
    # Full layer 5: 8 * 512 = 4096
    assert cfg.q_proj_out(5) == 4096
    assert cfg.kv_proj_out(5) == 1024


def test_all_layer_configs_length():
    cfg = gemma4_e4b_config()
    all_lc = cfg.all_layer_configs()
    assert len(all_lc) == cfg.n_layers
    assert sum(1 for lc in all_lc if lc.is_full_attention) == 7


# ----- Gemma4Layer -----

def test_gemma4_layer_swa_shapes():
    cfg = gemma4_e4b_config()
    layer = Gemma4Layer(cfg, layer_idx=0)
    # q_proj: d_model → n_heads * head_dim = 2560 → 2048
    assert layer.q_proj.in_features == 2560
    assert layer.q_proj.out_features == 2048
    # k_proj: 2560 → 2 * 256 = 512
    assert layer.k_proj.out_features == 512


def test_gemma4_layer_full_shapes():
    cfg = gemma4_e4b_config()
    layer = Gemma4Layer(cfg, layer_idx=5)
    assert layer.q_proj.out_features == 4096
    assert layer.k_proj.out_features == 1024


def test_gemma4_layer_has_gemma4_specific_norms():
    cfg = gemma4_e4b_config()
    layer = Gemma4Layer(cfg, layer_idx=0)
    # Per-head norms specific to Gemma 4
    assert layer.attn_q_norm.shape == (256,)
    assert layer.attn_k_norm.shape == (256,)
    # Scalar output scale
    assert layer.layer_output_scale.shape == (1,)
    # Post-attention norm
    assert layer.post_attn_norm.shape == (cfg.d_model,)


# ----- Gemma4Stream (without GGUF load) -----

def test_gemma4_stream_constructs():
    """Construct stream (randomized weights) — just verify shapes."""
    cfg = gemma4_e4b_config()
    # Smaller test config to keep memory manageable
    tiny = Gemma4Config(
        d_model=256, n_heads=2, n_kv_heads=1, n_layers=4,
        d_ffn=512, vocab_size=512, max_position=128,
        swa_head_dim=64, full_head_dim=128,
        swa_rope_dim_count=64, full_rope_dim_count=128,
        full_attention_layers=(3,),
        per_layer_embed_dim=32,
    )
    stream = Gemma4Stream(tiny)
    assert len(stream.layers) == 4
    # RoPE caches exist for both SWA and full
    assert stream.swa_rope_cos.shape == (128, 64)
    assert stream.full_rope_cos.shape == (128, 128)


# ----- GGUF-backed tests (require real file) -----

@pytest.mark.skipif(not GGUF_AVAILABLE, reason="Gemma 4 GGUF not present")
def test_derive_config_from_real_gguf():
    from calm.llm_computer.gemma4_config import derive_config_from_gguf
    from calm.llm_computer.tq4_gguf_loader import read_turboquant_gguf
    reader = read_turboquant_gguf(GGUF_PATH)
    cfg = derive_config_from_gguf(reader)
    assert cfg.n_layers == 42
    assert cfg.d_model == 2560
    assert cfg.n_heads == 8
    assert cfg.n_kv_heads == 2
    assert cfg.swa_head_dim == 256
    assert cfg.full_head_dim == 512
    assert cfg.full_attention_layers == (5, 11, 17, 23, 29, 35, 41)


@pytest.mark.skipif(not GGUF_AVAILABLE, reason="Gemma 4 GGUF not present")
def test_load_gemma4_stream_from_gguf_structural():
    """Full load test — verifies the loader walks the GGUF without errors.
    Numerical correctness against llama.cpp is a separate step (see
    scripts/validate_gemma4_vs_llamacpp.py).
    """
    from calm.llm_computer.gemma4_stream import load_gemma4_stream_from_gguf
    stream = load_gemma4_stream_from_gguf(str(GGUF_PATH), device="cpu")
    # Verify every layer's tq4 projections are loaded
    for i, layer in enumerate(stream.layers):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj", "inp_gate"):
            m = getattr(layer, name)
            assert m.is_loaded(), f"layer {i} {name} not loaded"


if __name__ == "__main__":
    test_full_attention_layers_default()
    print("[ok] full_attention_layers = (5, 11, 17, 23, 29, 35, 41)")
    test_layer_config_swa()
    print("[ok] SWA layer config")
    test_layer_config_full()
    print("[ok] full attention layer config")
    test_projection_sizes()
    print("[ok] projection sizes per layer type")
    test_all_layer_configs_length()
    print("[ok] all_layer_configs")
    test_gemma4_layer_swa_shapes()
    print("[ok] Gemma4Layer SWA shapes")
    test_gemma4_layer_full_shapes()
    print("[ok] Gemma4Layer full shapes")
    test_gemma4_layer_has_gemma4_specific_norms()
    print("[ok] Gemma4Layer has Gemma 4 specific norms")
    test_gemma4_stream_constructs()
    print("[ok] Gemma4Stream constructs (tiny config)")
    if GGUF_AVAILABLE:
        test_derive_config_from_real_gguf()
        print("[ok] derive config from real Gemma 4 GGUF")
        test_load_gemma4_stream_from_gguf_structural()
        print("[ok] full Gemma 4 stream loads from GGUF structurally")
    else:
        print("[SKIP] real GGUF not available")
