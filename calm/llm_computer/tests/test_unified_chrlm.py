"""Tests for unified CHRLM infrastructure — compiled + trained in one tensor.

Demonstrates the fusion pattern at adder_tiny scale:
  - Install compiled adder_tiny into layer 1 of a 2-layer substrate
  - Freeze layer 1 + embeddings + head (the compiled portions)
  - Layer 0 remains trainable
  - Compiled adder still passes 16/16 exhaustive before AND after trainable
    layer gets noisy weights (simulates start of training)
"""

from __future__ import annotations

import torch

from calm.llm_computer.unified_chrlm import (
    UnifiedCHRLMConfig,
    build_unified_chrlm,
    install_compiled_program,
    freeze_layer_params,
    freeze_embeddings_and_head,
    trainable_param_count,
    verify_compiled_preserved,
)
# Reuse the compiled-adder installer from fusion-mvp experiment — it
# builds a compiled adder that slots into a 2-layer model.
from scripts.experiment_fast_weights_fusion import (
    build_adder_tiny_small2d, exhaustive_adder,
)


def _build_compiled_adder_at_layer_1():
    """Source program: adder compiled into layer 1 of a 2-layer model."""
    return build_adder_tiny_small2d(target_layer=1, n_layers=2)


def test_empty_unified_chrlm_is_zero():
    """build_unified_chrlm returns an all-zero model."""
    cfg = UnifiedCHRLMConfig(
        vocab_size=8, d_model=10, n_heads=5, n_layers=2, d_ffn=14,
        max_len=4, use_hard_max=True,
    )
    model = build_unified_chrlm(cfg)
    for p in model.parameters():
        assert (p == 0).all(), "unified CHRLM should start zeroed"


def test_load_compiled_adder_via_state_dict():
    """Use state_dict transfer (cleanest path for a single-program model):
    the compiled adder IS a substrate tensor, so its state_dict IS the
    unified CHRLM's state_dict at matching dimensions."""
    src = _build_compiled_adder_at_layer_1()
    # Build an identical-shape empty unified model
    cfg = UnifiedCHRLMConfig(
        vocab_size=src.config.vocab_size,
        d_model=src.config.d_model,
        n_heads=src.config.n_heads,
        n_layers=src.config.n_layers,
        d_ffn=src.config.d_ffn,
        max_len=src.config.max_len,
        use_hard_max=src.config.use_hard_max,
    )
    unified = build_unified_chrlm(cfg)
    unified.load_state_dict(src.state_dict())
    acc = exhaustive_adder(unified)
    assert acc == 16, f"compiled adder should pass 16/16 after transfer, got {acc}"


def test_freeze_layer_params():
    """freeze_layer_params sets requires_grad=False on all layer tensors."""
    cfg = UnifiedCHRLMConfig(
        vocab_size=8, d_model=10, n_heads=5, n_layers=2, d_ffn=14,
        max_len=4, use_hard_max=True,
    )
    model = build_unified_chrlm(cfg)
    total_trainable = trainable_param_count(model)
    n_frozen = freeze_layer_params(model, layer_idx=1)
    after_trainable = trainable_param_count(model)
    assert n_frozen > 0
    assert after_trainable == total_trainable - n_frozen


def test_freeze_embeddings_and_head():
    """Embeddings + LM head protection."""
    cfg = UnifiedCHRLMConfig(
        vocab_size=8, d_model=10, n_heads=5, n_layers=2, d_ffn=14,
        max_len=4, use_hard_max=True,
    )
    model = build_unified_chrlm(cfg)
    n_frozen = freeze_embeddings_and_head(model)
    # Token embed (8 * 10) + pos embed (4 * 10) + head (10 * 8) = 80 + 40 + 80 = 200
    assert n_frozen == 80 + 40 + 80


def test_fusion_compiled_survives_trainable_layer_noise():
    """Load compiled adder into a 2-layer model, freeze everything
    compiled, add training-like noise to the trainable layer 0, confirm
    compiled layer 1 still passes 16/16 exhaustive."""
    src = _build_compiled_adder_at_layer_1()
    cfg = UnifiedCHRLMConfig(
        vocab_size=src.config.vocab_size,
        d_model=src.config.d_model,
        n_heads=src.config.n_heads,
        n_layers=src.config.n_layers,
        d_ffn=src.config.d_ffn,
        max_len=src.config.max_len,
        use_hard_max=src.config.use_hard_max,
    )
    unified = build_unified_chrlm(cfg)
    unified.load_state_dict(src.state_dict())

    # Freeze layer 1 (compiled) + embeddings + head. Layer 0 stays
    # trainable — this is how the unified CHRLM training would be set up.
    freeze_layer_params(unified, layer_idx=1)
    freeze_embeddings_and_head(unified)

    # Baseline: compiled adder works before we touch layer 0
    assert exhaustive_adder(unified) == 16

    # Simulate a training step's worth of noise on layer 0 (the trainable
    # region). Use small sigma so it's realistic: gradient updates are
    # typically ~1e-3 to 1e-2 per step.
    with torch.no_grad():
        for p in unified.W_qkv[0].parameters():
            p.add_(torch.randn_like(p) * 1e-4)
        for p in unified.ff_in[0].parameters():
            p.add_(torch.randn_like(p) * 1e-4)

    # Compiled adder (layer 1) must still hit 16/16. Its weights are
    # frozen, and layer-0 activations feed into layer 1 but the adder
    # compile uses self-written channels that trainable noise doesn't
    # typically overwrite.
    post = exhaustive_adder(unified)
    assert post == 16, (
        f"compiled adder regressed after trainable-layer noise: {post}/16"
    )


def test_trainable_count_shrinks_when_compiled_portions_frozen():
    """After freezing compiled portions, trainable count should equal
    the non-compiled layers' worth of params."""
    cfg = UnifiedCHRLMConfig(
        vocab_size=8, d_model=10, n_heads=5, n_layers=2, d_ffn=14,
        max_len=4, use_hard_max=True,
    )
    model = build_unified_chrlm(cfg)
    before = trainable_param_count(model)

    # Freeze compiled portions
    freeze_layer_params(model, layer_idx=1)
    freeze_embeddings_and_head(model)
    after = trainable_param_count(model)

    # Should be strictly smaller.
    assert after < before
    # Layer 0 (4 linear tensors) should still be trainable.
    layer0_count = sum(
        p.numel()
        for mod in (model.W_qkv[0], model.W_out[0],
                    model.ff_in[0], model.ff_out[0])
        for p in mod.parameters()
    )
    assert after == layer0_count


if __name__ == "__main__":
    test_empty_unified_chrlm_is_zero()
    print("[ok] empty unified CHRLM starts zeroed")
    test_load_compiled_adder_via_state_dict()
    print("[ok] compiled adder loads into unified substrate via state_dict")
    test_freeze_layer_params()
    print("[ok] freeze_layer_params reduces trainable count")
    test_freeze_embeddings_and_head()
    print("[ok] freeze_embeddings_and_head covers tok/pos/head")
    test_fusion_compiled_survives_trainable_layer_noise()
    print("[ok] compiled adder survives noise on trainable layer 0")
    test_trainable_count_shrinks_when_compiled_portions_frozen()
    print("[ok] trainable count matches unfrozen-layer params only")
