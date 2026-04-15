"""Tests for multi-stream residual architecture (Level 2 bus redesign)."""

from __future__ import annotations

import pytest
import torch

from calm.llm_computer.multi_stream import (
    MultiStreamConfig, MultiStreamTransformer, StreamSpec,
    build_empty_multistream, freeze_head, freeze_stream,
    trainable_param_count,
)


def _two_stream_cfg():
    return MultiStreamConfig(
        streams=(
            StreamSpec(name="math", d_model=10, n_heads=5, d_ffn=14),
            StreamSpec(name="lm",   d_model=32, n_heads=16, d_ffn=64),
        ),
        n_layers=2, vocab_size=16, max_len=4, use_hard_max=False,
    )


def test_d_head_invariant():
    # d_model=6 / n_heads=3 → d_head=2 ok
    MultiStreamConfig(
        streams=(StreamSpec("a", 6, 3, 4),),
        n_layers=1, vocab_size=4, max_len=2,
    )
    # d_model=8 / n_heads=4 → d_head=2 ok
    MultiStreamConfig(
        streams=(StreamSpec("a", 8, 4, 4),),
        n_layers=1, vocab_size=4, max_len=2,
    )
    # d_model=6 / n_heads=2 → d_head=3, must fail
    with pytest.raises(AssertionError, match="d_head"):
        MultiStreamConfig(
            streams=(StreamSpec("a", 6, 2, 4),),
            n_layers=1, vocab_size=4, max_len=2,
        )


def test_duplicate_stream_names_rejected():
    with pytest.raises(AssertionError, match="duplicate"):
        MultiStreamConfig(
            streams=(
                StreamSpec("a", 10, 5, 14),
                StreamSpec("a", 32, 16, 64),
            ),
            n_layers=1, vocab_size=4, max_len=2,
        )


def test_forward_shape():
    cfg = _two_stream_cfg()
    m = MultiStreamTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, cfg.max_len), dtype=torch.long)
    logits = m(x)
    assert logits.shape == (3, cfg.max_len, cfg.vocab_size)


def test_param_count_is_stream_sum_plus_head():
    cfg = _two_stream_cfg()
    m = MultiStreamTransformer(cfg)
    head_params = cfg.total_d * cfg.vocab_size  # Linear with bias=False
    stream_params = {
        s.name: sum(p.numel() for p in m.streams[s.name].parameters())
        for s in cfg.streams
    }
    assert m.param_count() == sum(stream_params.values()) + head_params


def test_build_empty_zeros_all_params():
    cfg = _two_stream_cfg()
    m = build_empty_multistream(cfg)
    for p in m.parameters():
        assert (p == 0).all()


def test_freeze_stream_shrinks_trainable():
    cfg = _two_stream_cfg()
    m = MultiStreamTransformer(cfg)
    before = trainable_param_count(m)
    frozen = freeze_stream(m, "math")
    after = trainable_param_count(m)
    assert frozen > 0
    assert after == before - frozen
    # All "math" stream params should have requires_grad=False
    for p in m.streams["math"].parameters():
        assert not p.requires_grad
    # All "lm" stream params still trainable
    for p in m.streams["lm"].parameters():
        assert p.requires_grad


def test_freeze_head():
    cfg = _two_stream_cfg()
    m = MultiStreamTransformer(cfg)
    n = freeze_head(m)
    assert n == cfg.total_d * cfg.vocab_size  # bias=False
    for p in m.head.parameters():
        assert not p.requires_grad


def test_physical_isolation_between_streams():
    """If we train only stream B, stream A's parameters must be bit-
    identical before and after. No gradient flows to frozen stream
    because streams don't share any parameter."""
    cfg = _two_stream_cfg()
    torch.manual_seed(0)
    m = MultiStreamTransformer(cfg)
    # Small random init so both streams have non-zero parameters
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.02)

    # Snapshot math stream parameters
    math_snapshot = {
        name: p.detach().clone()
        for name, p in m.streams["math"].named_parameters()
    }

    # Freeze math stream; train lm stream + head for a few steps
    freeze_stream(m, "math")
    trainable = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-2)
    rng = torch.Generator().manual_seed(0)
    xs = torch.randint(0, cfg.vocab_size, (16, cfg.max_len),
                       dtype=torch.long, generator=rng)
    ys = torch.randint(0, cfg.vocab_size, (16,),
                       dtype=torch.long, generator=rng)
    m.train()
    for _ in range(20):
        logits = m(xs)[:, -1, :]
        loss = torch.nn.functional.cross_entropy(logits, ys)
        opt.zero_grad(); loss.backward(); opt.step()

    # Math stream params must be unchanged bit-for-bit
    for name, p in m.streams["math"].named_parameters():
        assert torch.equal(p, math_snapshot[name]), (
            f"math.{name} changed during lm-only training"
        )


def test_save_reload_round_trip():
    cfg = _two_stream_cfg()
    torch.manual_seed(42)
    m = MultiStreamTransformer(cfg)
    # Randomize
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.1)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.max_len), dtype=torch.long)
    before = m(x)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(m.state_dict(), f.name)
        path = f.name

    m2 = MultiStreamTransformer(cfg)
    m2.load_state_dict(torch.load(path, weights_only=True))
    m2.eval()
    after = m2(x)
    assert torch.equal(before, after), "forward output differs after reload"


def test_single_stream_multistream_parity_shape():
    """A single-stream MultiStreamTransformer produces logits of the
    expected shape and doesn't crash. Not a bit-match to Small2DTransformer
    (different parameter layout) but structurally equivalent."""
    cfg = MultiStreamConfig(
        streams=(StreamSpec("only", d_model=10, n_heads=5, d_ffn=14),),
        n_layers=2, vocab_size=8, max_len=4,
    )
    m = MultiStreamTransformer(cfg)
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    out = m(x)
    assert out.shape == (1, 4, 8)


if __name__ == "__main__":
    test_d_head_invariant()
    print("[ok] d_head=2 invariant enforced")
    test_duplicate_stream_names_rejected()
    print("[ok] duplicate stream names rejected")
    test_forward_shape()
    print("[ok] forward shape correct")
    test_param_count_is_stream_sum_plus_head()
    print("[ok] param count = streams + head")
    test_build_empty_zeros_all_params()
    print("[ok] build_empty zeros all params")
    test_freeze_stream_shrinks_trainable()
    print("[ok] freeze_stream reduces trainable count")
    test_freeze_head()
    print("[ok] freeze_head works")
    test_physical_isolation_between_streams()
    print("[ok] frozen stream unchanged during other stream's training")
    test_save_reload_round_trip()
    print("[ok] save/reload preserves logits")
    test_single_stream_multistream_parity_shape()
    print("[ok] single-stream multistream shapes out correctly")
