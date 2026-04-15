"""Tests for channel masking — gradient hooks that block a trainable
layer from writing to compiled programs' output channels."""

from __future__ import annotations

import torch

from calm.llm_computer.channel_masking import (
    compiled_output_channels_adder_tiny, protect_residual_channels,
)
from calm.llm_computer.unified_chrlm import (
    UnifiedCHRLMConfig, build_unified_chrlm,
    freeze_embeddings_and_head, freeze_layer_params,
)
from scripts.experiment_fast_weights_fusion import (
    build_adder_tiny_small2d, exhaustive_adder,
)


def _make_unified_with_adder():
    """2-layer unified model: layer 0 = compiled adder, layer 1 = trainable."""
    cfg = UnifiedCHRLMConfig(
        vocab_size=8, d_model=10, n_heads=5, n_layers=2, d_ffn=14,
        max_len=4, use_hard_max=True,
        compiled_layers=(0,),
    )
    model = build_unified_chrlm(cfg)
    src = build_adder_tiny_small2d(target_layer=0, n_layers=2)
    model.load_state_dict(src.state_dict())
    # Small random init on layer 1 so there's something to protect
    with torch.no_grad():
        for lin in (model.W_qkv[1], model.W_out[1],
                    model.ff_in[1], model.ff_out[1]):
            lin.weight.normal_(0.0, 0.02)
    freeze_layer_params(model, layer_idx=0)
    freeze_embeddings_and_head(model)
    return model


def test_protected_rows_zero_at_init():
    """After protection, protected rows of W_out and ff_out must be zero."""
    model = _make_unified_with_adder()
    protected = (3, 4, 5)
    protect_residual_channels(model, layer_idx=1, protected_channels=protected)
    for c in protected:
        assert (model.W_out[1].weight[c, :] == 0).all(), (
            f"W_out row {c} should be zero after protection"
        )
        assert (model.ff_out[1].weight[c, :] == 0).all(), (
            f"ff_out row {c} should be zero after protection"
        )


def test_unprotected_rows_untouched_at_init():
    """Non-protected rows must retain their init values."""
    model = _make_unified_with_adder()
    original_w = model.W_out[1].weight.clone()
    original_ff = model.ff_out[1].weight.clone()
    protected = (3, 4, 5)
    protect_residual_channels(model, layer_idx=1, protected_channels=protected)
    for c in range(model.config.d_model):
        if c in protected:
            continue
        assert torch.equal(model.W_out[1].weight[c, :], original_w[c, :]), (
            f"W_out row {c} should be untouched"
        )
        assert torch.equal(model.ff_out[1].weight[c, :], original_ff[c, :]), (
            f"ff_out row {c} should be untouched"
        )


def test_gradient_hook_zeros_protected_rows():
    """After a backward pass, gradient on protected rows must be zero."""
    model = _make_unified_with_adder()
    protected = (3, 4, 5)
    protect_residual_channels(model, layer_idx=1, protected_channels=protected)
    # Forward + backward with any loss
    x = torch.tensor([[1, 2]], dtype=torch.long)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    for c in protected:
        assert (model.W_out[1].weight.grad[c, :] == 0).all(), (
            f"W_out row {c} gradient should be zero after hook"
        )
        assert (model.ff_out[1].weight.grad[c, :] == 0).all(), (
            f"ff_out row {c} gradient should be zero after hook"
        )


def test_hook_preserves_unprotected_gradient():
    """Synthetic check: inject a gradient of all ones via `(w * detached
    ones).sum()` — derivative w.r.t. w is detached ones. The hook should
    zero protected rows and leave unprotected rows as ones."""
    model = _make_unified_with_adder()
    protected = (3, 4, 5)
    protect_residual_channels(model, layer_idx=1, protected_channels=protected)

    w = model.W_out[1].weight
    w.grad = None
    # derivative of (w * fake).sum() w.r.t. w is fake
    fake = torch.ones_like(w).detach()
    (w * fake).sum().backward()

    assert w.grad is not None
    for c in protected:
        assert (w.grad[c, :] == 0).all(), f"protected row {c} grad not zeroed"
    unprotected = [c for c in range(model.config.d_model) if c not in protected]
    for c in unprotected:
        assert (w.grad[c, :] == 1).all(), (
            f"unprotected row {c} grad should be ones, got {w.grad[c, :]}"
        )


def test_adder_survives_aggressive_training_with_mask():
    """The real gate: with channel mask, adder survives aggressive
    training that breaks it without the mask."""
    model = _make_unified_with_adder()
    protect_residual_channels(
        model, layer_idx=1,
        protected_channels=compiled_output_channels_adder_tiny(),
    )
    assert exhaustive_adder(model) == 16, "pre-training adder must pass"

    # Aggressive: lr=1e-3, 100 steps on adder task
    import torch.nn.functional as F
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-3)
    xs = torch.tensor([[a, b] for a in range(4) for b in range(4)], dtype=torch.long)
    ys = torch.tensor([a + b for a in range(4) for b in range(4)], dtype=torch.long)
    rng = torch.Generator().manual_seed(0)
    model.train()
    for _ in range(100):
        idx = torch.randint(0, 16, (8,), generator=rng)
        logits = model(xs[idx])
        loss = F.cross_entropy(logits[:, 1, :], ys[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    assert exhaustive_adder(model) == 16, (
        "adder should survive 100 aggressive-lr steps under channel mask"
    )


def test_out_of_range_channel_raises():
    model = _make_unified_with_adder()
    try:
        protect_residual_channels(model, layer_idx=1, protected_channels=[99])
    except IndexError:
        return
    raise AssertionError("out-of-range channel should raise IndexError")


def test_removable_handles_teardown():
    """Handles returned must be RemovableHandles that can be .remove()'d."""
    model = _make_unified_with_adder()
    handles = protect_residual_channels(
        model, layer_idx=1, protected_channels=(3, 4, 5),
    )
    assert len(handles) == 2  # one per weight (W_out, ff_out)
    for h in handles:
        h.remove()  # should not raise


if __name__ == "__main__":
    test_protected_rows_zero_at_init()
    print("[ok] protected rows zero at init")
    test_unprotected_rows_untouched_at_init()
    print("[ok] unprotected rows preserved at init")
    test_gradient_hook_zeros_protected_rows()
    print("[ok] gradient hook zeros protected rows on backward")
    test_hook_preserves_unprotected_gradient()
    print("[ok] hook preserves gradient on unprotected rows")
    test_adder_survives_aggressive_training_with_mask()
    print("[ok] adder survives aggressive training with channel mask")
    test_out_of_range_channel_raises()
    print("[ok] out-of-range channel raises IndexError")
    test_removable_handles_teardown()
    print("[ok] handles are removable")
