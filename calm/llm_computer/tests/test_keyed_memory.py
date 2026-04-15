"""Tests for keyed residual memory (Level 3 bus redesign).

Gates:
  - parabolic_key math is correct
  - write + hard_read round-trips exactly
  - write + soft_read round-trips approximately (softmax noise)
  - multiple keys coexist in the same residual stream
  - read-by-name symbolic interface works end-to-end
  - gradients flow through soft read (for trainable value cards)
  - KeyRegistry registers / recovers names
  - invalid key_id / channel configs caught at validate time
"""

from __future__ import annotations

import pytest
import torch

from calm.llm_computer.keyed_memory import (
    KeyedMemoryConfig, KeyRegistry, parabolic_key,
    read_by_key_attention, read_by_key_hard, read_by_name,
    write_by_name, write_keyed_slot,
)


def _simple_cfg():
    # d_model=8: key channels 0-1, value channels 2-5
    return KeyedMemoryConfig(
        key_channels=slice(0, 2),
        value_channels=slice(2, 6),
        max_key_id=16,
    )


def test_parabolic_key_math():
    # key_id 0 → (0, 0)
    assert parabolic_key(0, slice(0, 2)) == (0.0, 0.0)
    # key_id 3 → (6, -9)
    assert parabolic_key(3, slice(0, 2)) == (6.0, -9.0)
    # key_id 5 → (10, -25)
    assert parabolic_key(5, slice(0, 2)) == (10.0, -25.0)


def test_parabolic_key_requires_2_channels():
    with pytest.raises(AssertionError, match="2-channel"):
        parabolic_key(1, slice(0, 3))


def test_write_and_hard_read_roundtrip():
    cfg = _simple_cfg()
    # 1 batch, 4 positions, d_model=8
    x = torch.zeros(1, 4, 8)
    v = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    x = write_keyed_slot(x, position=2, key_id=5, value=v, cfg=cfg)
    # Read back via hard attention
    out = read_by_key_hard(x, query_key_id=5, cfg=cfg)
    assert torch.allclose(out, v, atol=1e-6)


def test_write_and_soft_read_approximately_matches():
    cfg = _simple_cfg()
    x = torch.zeros(2, 6, 8)
    v = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    x = write_keyed_slot(x, position=3, key_id=7, value=v, cfg=cfg)
    out = read_by_key_attention(x, query_key_id=7, cfg=cfg)
    # Soft attention has some mass on other positions where key_channels=0
    # but should be dominated by the correct position.
    # Expected error is relative to other positions' values (zero here).
    assert torch.allclose(out, v, atol=1.0), (
        f"soft read drift too large: got {out}, expected ~{v}"
    )


def test_multiple_keys_coexist():
    cfg = _simple_cfg()
    x = torch.zeros(1, 8, 8)
    v1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    v2 = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
    v3 = torch.tensor([[9.0, 10.0, 11.0, 12.0]])
    x = write_keyed_slot(x, position=0, key_id=2, value=v1, cfg=cfg)
    x = write_keyed_slot(x, position=3, key_id=4, value=v2, cfg=cfg)
    x = write_keyed_slot(x, position=5, key_id=7, value=v3, cfg=cfg)

    assert torch.allclose(read_by_key_hard(x, 2, cfg), v1)
    assert torch.allclose(read_by_key_hard(x, 4, cfg), v2)
    assert torch.allclose(read_by_key_hard(x, 7, cfg), v3)


def test_read_missing_key_is_zero_or_near_zero():
    """Reading an unwritten key should return near-zero (no position
    was written, so keys and values are all zero)."""
    cfg = _simple_cfg()
    x = torch.zeros(1, 4, 8)
    out = read_by_key_hard(x, query_key_id=3, cfg=cfg)
    assert torch.allclose(out, torch.zeros_like(out))


def test_symbolic_write_and_read_by_name():
    cfg = _simple_cfg()
    reg = KeyRegistry()
    x = torch.zeros(1, 4, 8)
    v = torch.tensor([[1.5, 2.5, 3.5, 4.5]])
    x = write_by_name(x, position=1, key_name="sum", value=v,
                     registry=reg, cfg=cfg)
    assert "sum" in reg
    out = read_by_name(x, "sum", registry=reg, cfg=cfg, hard=True)
    assert torch.allclose(out, v)


def test_key_registry_roundtrips_names():
    reg = KeyRegistry()
    id_a = reg.register("alpha")
    id_b = reg.register("beta")
    id_a_again = reg.register("alpha")  # idempotent
    assert id_a == id_a_again
    assert id_a != id_b
    assert reg.name_of(id_a) == "alpha"
    assert reg.name_of(id_b) == "beta"
    assert set(reg.names()) == {"alpha", "beta"}


def test_key_registry_respects_max():
    # max_key_id is exclusive; IDs start at 1. So max_key_id=3 gives IDs {1, 2}.
    reg = KeyRegistry(max_key_id=3)
    reg.register("a")
    reg.register("b")
    with pytest.raises(ValueError, match="full"):
        reg.register("c")


def test_config_validates_channel_layout():
    # Overlapping key/value channels must fail
    with pytest.raises(AssertionError, match="overlap"):
        KeyedMemoryConfig(
            key_channels=slice(0, 2),
            value_channels=slice(1, 5),  # overlaps key[1]
        ).validate(d_model=8)
    # Key region must be exactly 2 channels
    with pytest.raises(AssertionError, match="exactly 2 channels"):
        KeyedMemoryConfig(
            key_channels=slice(0, 3),
            value_channels=slice(3, 6),
        ).validate(d_model=8)


def test_gradient_flows_through_soft_read():
    """Soft read is differentiable — value cards can be trainable."""
    cfg = _simple_cfg()
    x = torch.zeros(1, 4, 8, requires_grad=False)
    v = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    x = write_keyed_slot(x, position=2, key_id=3, value=v, cfg=cfg)
    out = read_by_key_attention(x, query_key_id=3, cfg=cfg)
    out.sum().backward()
    assert v.grad is not None
    # Gradient on the written value should be non-zero
    assert (v.grad != 0).any()


def test_invalid_key_id_raises():
    cfg = _simple_cfg()  # max_key_id=16
    x = torch.zeros(1, 4, 8)
    v = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    with pytest.raises(AssertionError):
        write_keyed_slot(x, position=0, key_id=20, value=v, cfg=cfg)


if __name__ == "__main__":
    test_parabolic_key_math()
    print("[ok] parabolic key math")
    test_parabolic_key_requires_2_channels()
    print("[ok] key region must be 2 channels")
    test_write_and_hard_read_roundtrip()
    print("[ok] write/hard_read round-trip exact")
    test_write_and_soft_read_approximately_matches()
    print("[ok] write/soft_read round-trip approximate")
    test_multiple_keys_coexist()
    print("[ok] multiple keys coexist")
    test_read_missing_key_is_zero_or_near_zero()
    print("[ok] missing key returns zero")
    test_symbolic_write_and_read_by_name()
    print("[ok] symbolic name interface works")
    test_key_registry_roundtrips_names()
    print("[ok] registry round-trips names")
    test_key_registry_respects_max()
    print("[ok] registry enforces max_key_id")
    test_config_validates_channel_layout()
    print("[ok] config validates layout")
    test_gradient_flows_through_soft_read()
    print("[ok] gradient flows through soft read")
    test_invalid_key_id_raises()
    print("[ok] invalid key_id raises")
