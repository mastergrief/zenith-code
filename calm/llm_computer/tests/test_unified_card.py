"""Tests for UnifiedCHRLMCard — Gemma + compiled cards + trained in one .pt."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from calm.llm_computer.unified_card import (
    CrossStreamJoin, UnifiedCHRLMCard, add_forward_residual_to_gemma4,
)


class _DummyStream(nn.Module):
    """Minimal stream for unit tests."""

    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, vocab)
        self.d_model = d_model

    def forward_residual(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        return self.proj(x)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_residual(input_ids))


class _DummyStream2(nn.Module):
    """Stream with different d_model — like a compiled card."""

    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.head = nn.Linear(d_model, vocab)
        self.d_model = d_model

    def forward_residual(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_residual(input_ids))


# ----- UnifiedCHRLMCard basics -----

def test_unified_card_holds_multiple_streams():
    streams = {
        "big":   _DummyStream(d_model=32, vocab=100),
        "small": _DummyStream2(d_model=8, vocab=20),
    }
    card = UnifiedCHRLMCard(streams)
    assert set(card.stream_names()) == {"big", "small"}
    # Params from both streams counted
    assert card.total_param_count() > 0
    per = card.per_stream_param_counts()
    assert "big" in per and "small" in per


def test_forward_via_named_stream():
    streams = {
        "big": _DummyStream(d_model=16, vocab=50),
        "small": _DummyStream2(d_model=4, vocab=10),
    }
    card = UnifiedCHRLMCard(streams)
    input_ids = torch.randint(0, 10, (1, 3))
    # Default: use first by name
    out = card.forward(input_ids, stream="small")
    assert out.shape == (1, 3, 10)


def test_forward_all_returns_per_stream_residuals():
    streams = {
        "s1": _DummyStream(d_model=16, vocab=50),
        "s2": _DummyStream2(d_model=8, vocab=50),
    }
    card = UnifiedCHRLMCard(streams)
    input_ids = torch.randint(0, 50, (1, 4))
    outs = card.forward_all(input_ids)
    assert "s1" in outs and "s2" in outs
    assert outs["s1"].shape == (1, 4, 16)  # s1's d_model
    assert outs["s2"].shape == (1, 4, 8)    # s2's d_model


def test_save_and_load_roundtrip(tmp_path):
    torch.manual_seed(7)
    streams = {
        "big": _DummyStream(d_model=32, vocab=100),
        "small": _DummyStream2(d_model=8, vocab=20),
    }
    # Randomize params to verify state_dict actually saves them
    with torch.no_grad():
        for p in nn.Sequential(*streams.values()).parameters():
            p.normal_(0, 0.1)
    card = UnifiedCHRLMCard(streams)
    save_path = tmp_path / "card.pt"
    card.save(save_path)
    assert save_path.exists()

    # Rebuild: provide builders
    builders = {
        "big":   lambda: _DummyStream(d_model=32, vocab=100),
        "small": lambda: _DummyStream2(d_model=8, vocab=20),
    }
    card2 = UnifiedCHRLMCard.load(save_path, builders)

    # Params must match bit-for-bit
    for name in card.streams:
        for p1, p2 in zip(
            card.streams[name].parameters(),
            card2.streams[name].parameters(),
        ):
            assert torch.equal(p1, p2)


def test_load_rejects_wrong_builders(tmp_path):
    card = UnifiedCHRLMCard({"a": _DummyStream(8, 20)})
    path = tmp_path / "c.pt"
    card.save(path)
    # Builder has wrong name
    with pytest.raises(ValueError, match="stream name mismatch"):
        UnifiedCHRLMCard.load(path, {"b": lambda: _DummyStream(8, 20)})


# ----- CrossStreamJoin -----

def test_cross_stream_join_shape():
    join = CrossStreamJoin(from_d=32, to_d=64, rank=4)
    x = torch.randn(2, 5, 32)
    out = join(x)
    assert out.shape == (2, 5, 64)


def test_cross_stream_join_zero_init_b_produces_zero():
    """At init, B=0 so join output should be exactly zero
    (adapter-compatible behavior)."""
    join = CrossStreamJoin(from_d=32, to_d=64, rank=4)
    x = torch.randn(2, 5, 32)
    out = join(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_cross_stream_join_trainable():
    join = CrossStreamJoin(from_d=16, to_d=8, rank=2)
    x = torch.randn(1, 3, 16)
    target = torch.randn(1, 3, 8)
    opt = torch.optim.Adam(join.parameters(), lr=1e-2)
    for _ in range(10):
        opt.zero_grad()
        y = join(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()
        opt.step()
    # After training, B should have non-zero values
    assert (join.B != 0).any()


# ----- Gemma4 forward_residual patch -----

class _FakeGemma4(nn.Module):
    """Minimal stand-in for Gemma4Stream to test the patch without loading 2.5B params."""

    def __init__(self):
        super().__init__()
        from calm.llm_computer.gemma4_config import Gemma4Config
        self.cfg = Gemma4Config(
            d_model=16, n_heads=2, n_kv_heads=1, n_layers=2,
            d_ffn=32, vocab_size=32, max_position=8,
            swa_head_dim=8, full_head_dim=16,
            swa_rope_dim_count=8, full_rope_dim_count=16,
            full_attention_layers=(1,),
            per_layer_embed_dim=4,
        )
        self.token_embd = nn.Embedding(32, 16)
        self.per_layer_token_embd = nn.Embedding(32, 4 * 2)
        self.per_layer_proj_norm = nn.Parameter(torch.ones(4))
        self.per_layer_proj = nn.Parameter(torch.zeros(16, 4 * 2))
        # Must have layers attribute with forward-compatible sigs
        self.layers = nn.ModuleList()
        self.output_norm = nn.Parameter(torch.ones(16))

    def _rms_apply(self, x, weight):
        rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        return weight * x / rms

    def _forward_layer(self, x, per_layer_embed, layer, positions):
        return x  # no-op for test


def test_add_forward_residual_patch():
    stream = _FakeGemma4()
    add_forward_residual_to_gemma4(stream)
    assert hasattr(stream, "forward_residual")
    input_ids = torch.tensor([[0, 1, 2]])
    out = stream.forward_residual(input_ids)
    # Should return (B, S, d_model) — not logits
    assert out.shape == (1, 3, 16)


if __name__ == "__main__":
    test_unified_card_holds_multiple_streams()
    print("[ok] unified card holds multiple streams")
    test_forward_via_named_stream()
    print("[ok] forward via named stream")
    test_forward_all_returns_per_stream_residuals()
    print("[ok] forward_all returns per-stream residuals")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path
        test_save_and_load_roundtrip(Path(d))
    print("[ok] save/load roundtrip preserves params")
    with tempfile.TemporaryDirectory() as d:
        test_load_rejects_wrong_builders(Path(d))
    print("[ok] load rejects wrong builders")
    test_cross_stream_join_shape()
    print("[ok] cross-stream join shape")
    test_cross_stream_join_zero_init_b_produces_zero()
    print("[ok] cross-stream join zero B = zero output")
    test_cross_stream_join_trainable()
    print("[ok] cross-stream join trainable")
    test_add_forward_residual_patch()
    print("[ok] gemma4 forward_residual patch")
