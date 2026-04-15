"""Tests for shared primitive pool."""

from __future__ import annotations

import pytest
import torch

from calm.llm_computer.primitive_pool import (
    AttentionHeadPool, FFNNeuronPool,
    SharedPrimitiveRegistry, CardPrimitiveSpec,
)


def test_attention_pool_shapes():
    pool = AttentionHeadPool(n_primitives=4, d_model=8, d_head=2)
    x = torch.randn(2, 4, 8)
    out = pool.forward_head(0, x)
    assert out.shape == (2, 4, 8)


def test_two_cards_share_head_get_same_output():
    """Shared head must produce identical output for cards that
    reference it — proof that sharing is semantic, not just indexing."""
    pool = AttentionHeadPool(n_primitives=3, d_model=8, d_head=2)
    with torch.no_grad():
        for p in pool.parameters():
            p.normal_(0, 0.1)
    x = torch.randn(1, 3, 8)
    # Two cards both use head 1
    card_a_out = pool.forward_head(1, x)
    card_b_out = pool.forward_head(1, x)
    assert torch.equal(card_a_out, card_b_out), (
        "same head must produce same output"
    )


def test_two_cards_using_different_heads_differ():
    """Sanity: different heads give different outputs (no collapse)."""
    pool = AttentionHeadPool(n_primitives=3, d_model=8, d_head=2)
    with torch.no_grad():
        for p in pool.parameters():
            p.normal_(0, 0.1)
    x = torch.randn(1, 3, 8)
    out_0 = pool.forward_head(0, x)
    out_1 = pool.forward_head(1, x)
    assert not torch.allclose(out_0, out_1)


def test_forward_multi_sums_heads():
    pool = AttentionHeadPool(n_primitives=4, d_model=8, d_head=2)
    with torch.no_grad():
        for p in pool.parameters():
            p.normal_(0, 0.1)
    x = torch.randn(1, 3, 8)
    single = pool.forward_head(0, x)
    double = pool.forward_multi([0, 0], x)
    assert torch.allclose(double, single * 2, atol=1e-5)


def test_d_head_not_2_rejected():
    with pytest.raises(AssertionError, match="d_head=2"):
        AttentionHeadPool(n_primitives=2, d_model=8, d_head=4)


def test_ffn_pool_forward_shape():
    pool = FFNNeuronPool(n_primitives=4, d_model=8)
    with torch.no_grad():
        for p in pool.parameters():
            p.normal_(0, 0.1)
    x = torch.randn(2, 3, 8)
    out = pool.forward_neuron(0, x)
    assert out.shape == (2, 3, 8)


def test_ffn_pool_writes_only_to_output_channel():
    """Neuron i writes to channel (i % d_model), others stay zero.
    Force non-zero activation by using positive gate/val weights
    and positive input (so ReLU(gate) > 0 guaranteed)."""
    pool = FFNNeuronPool(n_primitives=4, d_model=8)
    with torch.no_grad():
        pool.gate_w.fill_(0.5)  # positive gate weights
        pool.val_w.fill_(0.5)
        pool.coef.fill_(1.0)
    x = torch.ones(1, 2, 8)  # positive input → gate = 0.5 * 8 = 4 > 0
    out = pool.forward_neuron(3, x)
    # Expected output channel: 3 % 8 = 3
    for ch in range(8):
        if ch == 3:
            assert (out[:, :, ch] != 0).any(), (
                f"neuron 3 should have non-zero output on channel 3"
            )
        else:
            assert (out[:, :, ch] == 0).all()


def test_ffn_pool_shared_neuron_same_output():
    pool = FFNNeuronPool(n_primitives=4, d_model=8)
    with torch.no_grad():
        for p in pool.parameters():
            p.normal_(0, 0.1)
    x = torch.randn(1, 2, 8)
    a = pool.forward_neuron(2, x)
    b = pool.forward_neuron(2, x)
    assert torch.equal(a, b)


def test_registry_allows_sharing():
    reg = SharedPrimitiveRegistry(n_heads=4, n_neurons=8)
    reg.register("cardA", head_indices=[0, 1], neuron_indices=[0, 1])
    reg.register("cardB", head_indices=[0, 2], neuron_indices=[1, 2])
    # Head 0 shared by A and B
    sharers = reg.cards_sharing_head(0)
    assert set(sharers) == {"cardA", "cardB"}
    # Neuron 1 shared by A and B
    assert set(reg.cards_sharing_neuron(1)) == {"cardA", "cardB"}


def test_registry_rejects_duplicate_card():
    reg = SharedPrimitiveRegistry(n_heads=4, n_neurons=8)
    reg.register("cardA", head_indices=[0], neuron_indices=[0])
    with pytest.raises(ValueError, match="already registered"):
        reg.register("cardA", head_indices=[1], neuron_indices=[1])


def test_registry_rejects_out_of_range():
    reg = SharedPrimitiveRegistry(n_heads=4, n_neurons=8)
    with pytest.raises(IndexError, match="head"):
        reg.register("cardA", head_indices=[4], neuron_indices=[0])
    with pytest.raises(IndexError, match="neuron"):
        reg.register("cardB", head_indices=[0], neuron_indices=[10])


def test_sharing_stats_counts_reuse():
    reg = SharedPrimitiveRegistry(n_heads=8, n_neurons=8)
    reg.register("cardA", head_indices=[0, 1, 2], neuron_indices=[0, 1])
    reg.register("cardB", head_indices=[1, 2], neuron_indices=[0])  # shares 1, 2, neuron 0
    stats = reg.sharing_stats()
    # Unique heads referenced: {0, 1, 2} = 3
    assert stats["unique_heads_used"] == 3
    # Total refs: 3 + 2 = 5; sharing = 5 - 3 = 2
    assert stats["total_head_refs"] == 5
    assert stats["sharing_heads"] == 2
    # Neurons: unique {0, 1} = 2, total 2+1=3, sharing 1
    assert stats["unique_neurons_used"] == 2
    assert stats["sharing_neurons"] == 1


if __name__ == "__main__":
    test_attention_pool_shapes()
    print("[ok] AttentionHeadPool shapes")
    test_two_cards_share_head_get_same_output()
    print("[ok] shared head = same output")
    test_two_cards_using_different_heads_differ()
    print("[ok] different heads differ")
    test_forward_multi_sums_heads()
    print("[ok] forward_multi sums contributions")
    test_d_head_not_2_rejected()
    print("[ok] d_head!=2 rejected")
    test_ffn_pool_forward_shape()
    print("[ok] FFNNeuronPool shapes")
    test_ffn_pool_writes_only_to_output_channel()
    print("[ok] neuron writes to specific channel")
    test_ffn_pool_shared_neuron_same_output()
    print("[ok] shared neuron same output")
    test_registry_allows_sharing()
    print("[ok] registry tracks sharing")
    test_registry_rejects_duplicate_card()
    print("[ok] duplicate card rejected")
    test_registry_rejects_out_of_range()
    print("[ok] out-of-range rejected")
    test_sharing_stats_counts_reuse()
    print("[ok] sharing_stats accurate")
