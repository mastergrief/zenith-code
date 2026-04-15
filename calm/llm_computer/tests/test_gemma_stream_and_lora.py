"""Tests for Gemma stream + LoRA + HF loader integration.

Uses TINY synthetic config (not full 2B) so tests run in seconds.
Real Gemma E2B loading is a downstream activity that requires HF
weights and bit-exact debugging; this validates the PLUMBING.
"""

from __future__ import annotations

import pytest
import torch

from calm.llm_computer.gemma_stream import GemmaStream, gemma_attention, gemma_ffn
from calm.llm_computer.hf_gemma_loader import (
    GemmaConfig, GemmaLayer, GemmaLayerWeights, freeze_gemma_base,
    gemma_tensor_names, validate_gemma_weight_shapes,
)
from calm.llm_computer.lora import (
    LoRAAdapter, LoRATq4Linear, merge_lora_into_base,
)
from calm.llm_computer.tq4_torch import HEAD_DIM, Tq4Linear


def _tiny_cfg():
    """Tiny Gemma-shaped config for unit tests. head_dim must divide
    cleanly into HEAD_DIM=256 for tq4."""
    return GemmaConfig(
        d_model=256, n_heads=4, n_kv_heads=2, head_dim=64,
        n_layers=2, d_ffn=512, vocab_size=128, max_position=8,
    )


# ----- LoRA -----

def test_lora_adapter_zero_init_b():
    """At init, B=0 so adapter output = 0 (no disruption)."""
    adapter = LoRAAdapter(in_features=32, out_features=32, rank=4)
    x = torch.randn(2, 3, 32)
    out = adapter(x)
    assert torch.allclose(out, torch.zeros_like(out))


def test_lora_adapter_produces_output_after_training():
    adapter = LoRAAdapter(in_features=32, out_features=32, rank=4, alpha=8)
    # Set B non-zero to simulate training
    with torch.no_grad():
        adapter.B.normal_(0, 0.1)
    x = torch.randn(2, 3, 32)
    out = adapter(x)
    assert not torch.allclose(out, torch.zeros_like(out))
    assert out.shape == (2, 3, 32)


def test_lora_wrap_base_frozen():
    base = Tq4Linear(256, 256)
    base.load_weight(torch.randn(256, 256) * 0.05)
    wrapped = LoRATq4Linear(base, rank=4, alpha=8)
    # Base parameters should be frozen
    for p in wrapped.base.parameters():
        assert not p.requires_grad
    # Adapter parameters trainable
    for p in wrapped.adapter.parameters():
        assert p.requires_grad


def test_lora_initial_output_equals_base():
    base = Tq4Linear(256, 256)
    base.load_weight(torch.randn(256, 256) * 0.05)
    wrapped = LoRATq4Linear(base, rank=4)
    x = torch.randn(1, 4, 256)
    base_out = base(x)
    wrapped_out = wrapped(x)
    # B is zero at init → LoRA output is zero, wrapper output = base
    assert torch.allclose(base_out, wrapped_out, atol=1e-5)


def test_lora_trainable_params_reasonable():
    base = Tq4Linear(256, 256)
    base.load_weight(torch.randn(256, 256) * 0.05)
    wrapped = LoRATq4Linear(base, rank=8, alpha=16)
    # A: (8, 256) = 2048, B: (256, 8) = 2048 → 4096 trainable
    assert wrapped.trainable_params() == 2048 + 2048


def test_merge_lora_into_base_zeros_b():
    base = Tq4Linear(256, 256)
    base.load_weight(torch.randn(256, 256) * 0.05)
    wrapped = LoRATq4Linear(base, rank=4)
    # Simulate training
    with torch.no_grad():
        wrapped.adapter.B.normal_(0, 0.05)
    merge_lora_into_base(wrapped)
    # After merge, B should be zeroed
    assert torch.allclose(wrapped.adapter.B, torch.zeros_like(wrapped.adapter.B))


# ----- HF loader -----

def test_gemma_tensor_names_cover_expected_keys():
    names = gemma_tensor_names(n_layers=2)
    assert "embed_tokens" in names
    assert "final_norm" in names
    for i in range(2):
        for suffix in ("q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj",
                       "input_norm", "post_attn_norm"):
            key = f"layer_{i}.{suffix}"
            assert key in names
            assert names[key].startswith(f"model.layers.{i}.")


def test_validate_shapes_accepts_valid():
    cfg = _tiny_cfg()
    w = GemmaLayerWeights(
        q_proj=torch.randn(cfg.q_proj_out, cfg.d_model),
        k_proj=torch.randn(cfg.kv_proj_out, cfg.d_model),
        v_proj=torch.randn(cfg.kv_proj_out, cfg.d_model),
        o_proj=torch.randn(cfg.d_model, cfg.q_proj_out),
        gate_proj=torch.randn(cfg.d_ffn, cfg.d_model),
        up_proj=torch.randn(cfg.d_ffn, cfg.d_model),
        down_proj=torch.randn(cfg.d_model, cfg.d_ffn),
        input_norm=torch.ones(cfg.d_model),
        post_attn_norm=torch.ones(cfg.d_model),
    )
    validate_gemma_weight_shapes(w, cfg)  # should not raise


def test_validate_shapes_rejects_bad():
    cfg = _tiny_cfg()
    w = GemmaLayerWeights(
        q_proj=torch.randn(100, 100),  # wrong shape
        k_proj=torch.randn(cfg.kv_proj_out, cfg.d_model),
        v_proj=torch.randn(cfg.kv_proj_out, cfg.d_model),
        o_proj=torch.randn(cfg.d_model, cfg.q_proj_out),
        gate_proj=torch.randn(cfg.d_ffn, cfg.d_model),
        up_proj=torch.randn(cfg.d_ffn, cfg.d_model),
        down_proj=torch.randn(cfg.d_model, cfg.d_ffn),
        input_norm=torch.ones(cfg.d_model),
        post_attn_norm=torch.ones(cfg.d_model),
    )
    with pytest.raises(AssertionError, match="q_proj"):
        validate_gemma_weight_shapes(w, cfg)


def test_gemma_layer_loads_weights():
    cfg = _tiny_cfg()
    layer = GemmaLayer(cfg)
    w = GemmaLayerWeights(
        q_proj=torch.randn(cfg.q_proj_out, cfg.d_model) * 0.02,
        k_proj=torch.randn(cfg.kv_proj_out, cfg.d_model) * 0.02,
        v_proj=torch.randn(cfg.kv_proj_out, cfg.d_model) * 0.02,
        o_proj=torch.randn(cfg.d_model, cfg.q_proj_out) * 0.02,
        gate_proj=torch.randn(cfg.d_ffn, cfg.d_model) * 0.02,
        up_proj=torch.randn(cfg.d_ffn, cfg.d_model) * 0.02,
        down_proj=torch.randn(cfg.d_model, cfg.d_ffn) * 0.02,
        input_norm=torch.ones(cfg.d_model),
        post_attn_norm=torch.ones(cfg.d_model),
    )
    layer.load_weights(w)
    assert layer.q_proj.is_loaded()
    assert layer.k_proj.is_loaded()
    assert layer.down_proj.is_loaded()


def test_freeze_gemma_base():
    cfg = _tiny_cfg()
    layer = GemmaLayer(cfg)
    n = freeze_gemma_base(layer)
    assert n > 0
    for p in layer.parameters():
        assert not p.requires_grad


# ----- Full stream -----

def test_gemma_stream_constructs():
    cfg = _tiny_cfg()
    stream = GemmaStream(cfg)
    # Should have 2 layers
    assert len(stream.layers) == cfg.n_layers
    # RoPE cache should exist
    assert stream.rope_cos.shape == (cfg.max_position, cfg.head_dim)


def _load_random_stream(cfg):
    """Populate a stream with random weights for forward testing."""
    stream = GemmaStream(cfg)
    with torch.no_grad():
        stream.embed.weight.normal_(0, 0.02)
        stream.final_norm.weight.fill_(1.0)
        for layer in stream.layers:
            w = GemmaLayerWeights(
                q_proj=torch.randn(cfg.q_proj_out, cfg.d_model) * 0.02,
                k_proj=torch.randn(cfg.kv_proj_out, cfg.d_model) * 0.02,
                v_proj=torch.randn(cfg.kv_proj_out, cfg.d_model) * 0.02,
                o_proj=torch.randn(cfg.d_model, cfg.q_proj_out) * 0.02,
                gate_proj=torch.randn(cfg.d_ffn, cfg.d_model) * 0.02,
                up_proj=torch.randn(cfg.d_ffn, cfg.d_model) * 0.02,
                down_proj=torch.randn(cfg.d_model, cfg.d_ffn) * 0.02,
                input_norm=torch.ones(cfg.d_model),
                post_attn_norm=torch.ones(cfg.d_model),
            )
            layer.load_weights(w)
    return stream


def test_gemma_stream_forward_shape():
    cfg = _tiny_cfg()
    stream = _load_random_stream(cfg)
    stream.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4), dtype=torch.long)
    with torch.no_grad():
        logits = stream(input_ids)
    assert logits.shape == (1, 4, cfg.vocab_size)
    # No NaN/inf
    assert torch.isfinite(logits).all()


def test_gemma_stream_lora_training_loop():
    """End-to-end: load stream, enable LoRA, freeze base, train
    LoRA adapters on a dummy loss. Base weights must stay untouched."""
    cfg = _tiny_cfg()
    stream = _load_random_stream(cfg)
    stream.enable_lora(rank=4, alpha=8, targets=["q_proj", "v_proj"])
    stream.freeze_base()

    # Snapshot base weights
    from calm.llm_computer.lora import LoRATq4Linear
    snap_qs = stream.layers[0].q_proj.base._qs.clone()
    snap_d = stream.layers[0].q_proj.base._d.clone()

    # Train LoRA adapters
    trainable = [p for p in stream.parameters() if p.requires_grad]
    assert len(trainable) > 0, "no trainable params after freeze_base"
    opt = torch.optim.Adam(trainable, lr=1e-3)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 4), dtype=torch.long)
    target = torch.randint(0, cfg.vocab_size, (2, 4), dtype=torch.long)
    stream.train()
    for _ in range(3):
        opt.zero_grad()
        logits = stream(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), target.reshape(-1),
        )
        loss.backward()
        opt.step()

    # Base weight codes must be bit-identical
    assert torch.equal(stream.layers[0].q_proj.base._qs, snap_qs)
    assert torch.equal(stream.layers[0].q_proj.base._d, snap_d)


def test_gemma_attention_shape():
    cfg = _tiny_cfg()
    layer = GemmaLayer(cfg)
    # Load random weights
    for name, out in [("q_proj", cfg.q_proj_out), ("k_proj", cfg.kv_proj_out),
                       ("v_proj", cfg.kv_proj_out), ("o_proj", cfg.d_model)]:
        getattr(layer, name).load_weight(
            torch.randn(out, getattr(layer, name).in_features) * 0.02,
        )
    from calm.llm_computer.rope import build_rope_cache
    cos, sin = build_rope_cache(cfg.head_dim, cfg.max_position)
    x = torch.randn(1, 4, cfg.d_model)
    out = gemma_attention(
        x, layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj,
        cfg, cos, sin,
    )
    assert out.shape == x.shape


def test_gemma_ffn_shape():
    cfg = _tiny_cfg()
    layer = GemmaLayer(cfg)
    layer.gate_proj.load_weight(torch.randn(cfg.d_ffn, cfg.d_model) * 0.02)
    layer.up_proj.load_weight(torch.randn(cfg.d_ffn, cfg.d_model) * 0.02)
    layer.down_proj.load_weight(torch.randn(cfg.d_model, cfg.d_ffn) * 0.02)
    x = torch.randn(1, 4, cfg.d_model)
    out = gemma_ffn(x, layer.gate_proj, layer.up_proj, layer.down_proj)
    assert out.shape == x.shape


if __name__ == "__main__":
    test_lora_adapter_zero_init_b()
    print("[ok] LoRA B=0 init → adapter output=0")
    test_lora_adapter_produces_output_after_training()
    print("[ok] LoRA adapter forward")
    test_lora_wrap_base_frozen()
    print("[ok] LoRA wrap freezes base")
    test_lora_initial_output_equals_base()
    print("[ok] LoRA initial output = base")
    test_lora_trainable_params_reasonable()
    print("[ok] LoRA param count")
    test_merge_lora_into_base_zeros_b()
    print("[ok] merge_lora zeros B")
    test_gemma_tensor_names_cover_expected_keys()
    print("[ok] tensor names cover expected keys")
    test_validate_shapes_accepts_valid()
    print("[ok] validate accepts valid")
    test_validate_shapes_rejects_bad()
    print("[ok] validate rejects bad shapes")
    test_gemma_layer_loads_weights()
    print("[ok] Gemma layer loads weights")
    test_freeze_gemma_base()
    print("[ok] freeze gemma base")
    test_gemma_stream_constructs()
    print("[ok] Gemma stream constructs")
    test_gemma_stream_forward_shape()
    print("[ok] Gemma stream forward shape")
    test_gemma_stream_lora_training_loop()
    print("[ok] end-to-end LoRA training")
    test_gemma_attention_shape()
    print("[ok] Gemma attention shape")
    test_gemma_ffn_shape()
    print("[ok] Gemma FFN shape")
