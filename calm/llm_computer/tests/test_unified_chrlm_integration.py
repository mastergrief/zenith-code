"""Integration tests: unified_chrlm with L1 (registry) and L2 (multi-stream)."""

from __future__ import annotations

import pytest
import torch

from calm.llm_computer.channel_registry import (
    AllocationError, MultiStreamChannelRegistry,
    adder_tiny_allocation, ChannelRegistry,
)
from calm.llm_computer.multi_stream import (
    MultiStreamConfig, MultiStreamTransformer, StreamSpec,
    build_empty_multistream,
)
from calm.llm_computer.unified_chrlm import (
    UnifiedCHRLMConfig, build_unified_chrlm,
    install_compiled_in_stream, install_compiled_with_registry,
    freeze_stream_layer, freeze_stream_embeddings,
)
from scripts.experiment_fast_weights_fusion import (
    build_adder_tiny_small2d, exhaustive_adder,
)


def _build_adder_src():
    return build_adder_tiny_small2d(target_layer=0, n_layers=1)


def test_install_with_registry_registers_allocations():
    cfg = UnifiedCHRLMConfig(
        vocab_size=8, d_model=10, n_heads=5, n_layers=1, d_ffn=14,
        max_len=4, use_hard_max=True,
    )
    model = build_unified_chrlm(cfg)
    registry = ChannelRegistry(d_model=10)

    install_compiled_with_registry(
        model, program_builder=_build_adder_src,
        target_layer=0, registry=registry,
        allocations=adder_tiny_allocation(),
    )
    # All adder channels 0..9 should be registered now
    assert registry.all_allocated() == frozenset(range(10))
    # And the compiled adder still works
    assert exhaustive_adder(model) == 16


def test_install_with_registry_catches_conflicts():
    cfg = UnifiedCHRLMConfig(
        vocab_size=8, d_model=10, n_heads=5, n_layers=1, d_ffn=14,
        max_len=4, use_hard_max=True,
    )
    model = build_unified_chrlm(cfg)
    registry = ChannelRegistry(d_model=10)
    # Pre-allocate a channel the adder wants
    registry.allocate("squatter", channels=[5], ch_type="other")
    with pytest.raises(AllocationError, match="conflict"):
        install_compiled_with_registry(
            model, program_builder=_build_adder_src,
            target_layer=0, registry=registry,
            allocations=adder_tiny_allocation(),
        )


def test_install_in_stream_places_adder_on_named_stream():
    cfg = MultiStreamConfig(
        streams=(
            StreamSpec("math", d_model=10, n_heads=5, d_ffn=14),
            StreamSpec("lm", d_model=32, n_heads=16, d_ffn=64),
        ),
        n_layers=1, vocab_size=8, max_len=4, use_hard_max=True,
    )
    ms = build_empty_multistream(cfg)
    install_compiled_in_stream(
        ms, program_builder=_build_adder_src,
        stream_name="math", target_layer=0,
    )
    # Math stream params should be non-zero (adder loaded)
    math_nonzero = any(
        (p != 0).any().item()
        for p in ms.streams["math"].parameters()
    )
    assert math_nonzero
    # LM stream params should still be all zero
    for p in ms.streams["lm"].parameters():
        assert (p == 0).all()
    # Head should have non-zero entries only in the math stream's offset
    # (first 10 columns = math, next 32 = lm)
    assert (ms.head.weight[:, 0:10] != 0).any()
    assert (ms.head.weight[:, 10:42] == 0).all()


def test_adder_works_in_multistream_via_head_offset():
    """Install adder on math stream; the shared head's math offset
    should decode a+b correctly even with lm stream empty."""
    cfg = MultiStreamConfig(
        streams=(
            StreamSpec("math", d_model=10, n_heads=5, d_ffn=14),
            StreamSpec("lm", d_model=32, n_heads=16, d_ffn=64),
        ),
        n_layers=1, vocab_size=8, max_len=4, use_hard_max=True,
    )
    ms = build_empty_multistream(cfg)
    install_compiled_in_stream(
        ms, program_builder=_build_adder_src,
        stream_name="math", target_layer=0,
    )
    # Exhaustive check: for all 16 (a,b), argmax over vocab should
    # equal a+b at position 1 (adder's output position).
    ok = 0
    for a in range(4):
        for b in range(4):
            x = torch.tensor([[a, b]], dtype=torch.long)
            with torch.no_grad():
                logits = ms(x)[0, 1, :7]  # restrict to adder vocab 0..6
            if int(logits.argmax().item()) == a + b:
                ok += 1
    assert ok == 16, f"adder in multi-stream: {ok}/16"


def test_install_in_stream_with_registry():
    cfg = MultiStreamConfig(
        streams=(
            StreamSpec("math", d_model=10, n_heads=5, d_ffn=14),
            StreamSpec("lm", d_model=32, n_heads=16, d_ffn=64),
        ),
        n_layers=1, vocab_size=8, max_len=4, use_hard_max=True,
    )
    ms = build_empty_multistream(cfg)
    regs = MultiStreamChannelRegistry.from_config(cfg)
    install_compiled_in_stream(
        ms, program_builder=_build_adder_src,
        stream_name="math", target_layer=0,
        registry=regs, allocations=adder_tiny_allocation(),
    )
    # Only math stream's registry should have allocations
    assert regs.for_stream("math").all_allocated() == frozenset(range(10))
    assert regs.for_stream("lm").all_allocated() == frozenset()


def test_freeze_stream_layer_and_embeddings():
    cfg = MultiStreamConfig(
        streams=(
            StreamSpec("math", d_model=10, n_heads=5, d_ffn=14),
            StreamSpec("lm", d_model=32, n_heads=16, d_ffn=64),
        ),
        n_layers=2, vocab_size=8, max_len=4,
    )
    ms = MultiStreamTransformer(cfg)
    n_layer = freeze_stream_layer(ms, "math", layer_idx=0)
    n_emb = freeze_stream_embeddings(ms, "math")
    assert n_layer > 0 and n_emb > 0
    # Math layer 0 + embeddings frozen
    for linear in (ms.streams["math"].W_qkv[0], ms.streams["math"].W_out[0],
                   ms.streams["math"].ff_in[0], ms.streams["math"].ff_out[0]):
        for p in linear.parameters():
            assert not p.requires_grad
    for p in ms.streams["math"].tok.parameters():
        assert not p.requires_grad
    # Math layer 1 still trainable
    for p in ms.streams["math"].W_qkv[1].parameters():
        assert p.requires_grad
    # LM stream untouched
    for p in ms.streams["lm"].parameters():
        assert p.requires_grad


def test_multistream_isolation_no_channel_mask_needed():
    """Key hypothesis: with multi-stream, training lm doesn't
    corrupt adder on math — no gradient hooks needed."""
    cfg = MultiStreamConfig(
        streams=(
            StreamSpec("math", d_model=10, n_heads=5, d_ffn=14),
            StreamSpec("lm", d_model=32, n_heads=16, d_ffn=64),
        ),
        n_layers=1, vocab_size=16, max_len=4, use_hard_max=True,
    )
    ms = build_empty_multistream(cfg)
    install_compiled_in_stream(
        ms, program_builder=_build_adder_src,
        stream_name="math", target_layer=0,
    )
    # Snapshot math stream + the head's math-offset columns
    math_snapshot = {
        name: p.detach().clone()
        for name, p in ms.streams["math"].named_parameters()
    }
    head_math_slice = ms.head.weight[:, :10].detach().clone()

    # Freeze math + head; train lm stream only
    freeze_stream_layer(ms, "math", layer_idx=0)
    freeze_stream_embeddings(ms, "math")
    for p in ms.head.parameters():
        p.requires_grad = False  # freeze head entirely for this test
    # Randomize lm stream so it has gradients to follow
    with torch.no_grad():
        for p in ms.streams["lm"].parameters():
            p.normal_(0, 0.1)

    trainable = [p for p in ms.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-2)
    xs = torch.randint(0, 8, (16, 4), dtype=torch.long)
    ys = torch.randint(0, 8, (16,), dtype=torch.long)
    ms.train()
    for _ in range(20):
        logits = ms(xs)[:, -1, :]
        loss = torch.nn.functional.cross_entropy(logits, ys)
        opt.zero_grad(); loss.backward(); opt.step()

    # Math stream params bit-identical
    for name, p in ms.streams["math"].named_parameters():
        assert torch.equal(p, math_snapshot[name]), (
            f"math.{name} changed during lm-only training"
        )
    # Head's math slice unchanged (whole head was frozen)
    assert torch.equal(ms.head.weight[:, :10], head_math_slice)

    # Adder still works!
    ok = 0
    for a in range(4):
        for b in range(4):
            x = torch.tensor([[a, b]], dtype=torch.long)
            with torch.no_grad():
                logits = ms(x)[0, 1, :7]
            if int(logits.argmax().item()) == a + b:
                ok += 1
    assert ok == 16, f"adder broken after lm training: {ok}/16"


if __name__ == "__main__":
    test_install_with_registry_registers_allocations()
    print("[ok] install_compiled_with_registry records allocations")
    test_install_with_registry_catches_conflicts()
    print("[ok] install_compiled_with_registry catches conflicts")
    test_install_in_stream_places_adder_on_named_stream()
    print("[ok] install_in_stream places adder on correct stream")
    test_adder_works_in_multistream_via_head_offset()
    print("[ok] adder decodes correctly via head offset")
    test_install_in_stream_with_registry()
    print("[ok] install_in_stream with registry")
    test_freeze_stream_layer_and_embeddings()
    print("[ok] freeze stream layer and embeddings")
    test_multistream_isolation_no_channel_mask_needed()
    print("[ok] multi-stream isolation — no hooks needed")
