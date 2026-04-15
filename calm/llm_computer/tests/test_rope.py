"""Tests for RoPE."""

from __future__ import annotations

import torch

from calm.llm_computer.rope import (
    apply_rope, build_rope_cache, rotate_half,
)


def test_rope_cache_shapes():
    cos, sin = build_rope_cache(head_dim=8, max_len=16)
    assert cos.shape == (16, 8)
    assert sin.shape == (16, 8)


def test_rope_cache_position_0_is_identity():
    """At position 0, cos=1 and sin=0 everywhere → rotation = identity."""
    cos, sin = build_rope_cache(head_dim=8, max_len=4)
    assert torch.allclose(cos[0], torch.ones_like(cos[0]))
    assert torch.allclose(sin[0], torch.zeros_like(sin[0]))


def test_rotate_half():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # half=2
    # Expected: [-x[2:], x[:2]] = [-3, -4, 1, 2]
    out = rotate_half(x)
    assert torch.equal(out, torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_apply_rope_position_0_identity():
    """At position 0, rotation should be identity."""
    cos, sin = build_rope_cache(head_dim=8, max_len=4)
    x = torch.randn(1, 1, 8)  # (batch, seq=1, head_dim)
    out = apply_rope(x, cos, sin)
    assert torch.allclose(out, x, atol=1e-6)


def test_apply_rope_preserves_norm():
    """RoPE is a rotation, so L2 norm of each (seq position) is
    preserved."""
    cos, sin = build_rope_cache(head_dim=16, max_len=8)
    x = torch.randn(2, 8, 16)
    y = apply_rope(x, cos, sin)
    x_norm = x.norm(dim=-1)
    y_norm = y.norm(dim=-1)
    assert torch.allclose(x_norm, y_norm, atol=1e-5)


def test_apply_rope_different_positions_different_outputs():
    cos, sin = build_rope_cache(head_dim=8, max_len=8)
    x = torch.ones(1, 8, 8)  # same input at every position
    y = apply_rope(x, cos, sin)
    # Position 0 should equal input; others should differ
    assert torch.allclose(y[0, 0], x[0, 0])
    # Position 1..7 must differ from position 0 (rotation is non-trivial)
    for p in range(1, 8):
        assert not torch.allclose(y[0, p], y[0, 0])


def test_apply_rope_with_explicit_positions():
    cos, sin = build_rope_cache(head_dim=8, max_len=16)
    x = torch.randn(1, 3, 8)
    positions = torch.tensor([5, 7, 9])
    y = apply_rope(x, cos, sin, positions=positions)
    assert y.shape == x.shape


def test_rope_cache_different_base():
    """Different base values should produce different caches."""
    cos1, _ = build_rope_cache(head_dim=8, max_len=4, base=10000.0)
    cos2, _ = build_rope_cache(head_dim=8, max_len=4, base=500000.0)
    assert not torch.allclose(cos1, cos2)


def test_odd_head_dim_rejected():
    try:
        build_rope_cache(head_dim=7, max_len=4)
    except AssertionError:
        return
    raise AssertionError("odd head_dim should be rejected")


if __name__ == "__main__":
    test_rope_cache_shapes()
    print("[ok] rope cache shapes")
    test_rope_cache_position_0_is_identity()
    print("[ok] position 0 cos=1 sin=0")
    test_rotate_half()
    print("[ok] rotate_half")
    test_apply_rope_position_0_identity()
    print("[ok] apply at position 0 = identity")
    test_apply_rope_preserves_norm()
    print("[ok] rope preserves norm")
    test_apply_rope_different_positions_different_outputs()
    print("[ok] different positions rotate differently")
    test_apply_rope_with_explicit_positions()
    print("[ok] explicit positions")
    test_rope_cache_different_base()
    print("[ok] different base differs")
    test_odd_head_dim_rejected()
    print("[ok] odd head_dim rejected")
