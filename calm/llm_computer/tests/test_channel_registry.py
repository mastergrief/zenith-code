"""Tests for channel registry (Level 1 residual bus redesign)."""

from __future__ import annotations

import pytest

from calm.llm_computer.channel_registry import (
    AllocationError, ChannelAllocation, ChannelRegistry,
    adder_tiny_allocation, register_adder_tiny,
)


def test_registry_accepts_non_overlapping_allocations():
    r = ChannelRegistry(d_model=16)
    r.allocate("cardA", channels=range(0, 4), ch_type="int_scalar",
               purpose="card A's integers")
    r.allocate("cardB", channels=range(4, 8), ch_type="text",
               purpose="card B's text")
    r.allocate("cardC", channels=[12, 14], ch_type="attention_key")
    assert len(r.cards()) == 3
    assert r.channels_for("cardA") == frozenset({0, 1, 2, 3})
    assert r.channels_for("cardB") == frozenset({4, 5, 6, 7})
    assert r.channels_for("cardC") == frozenset({12, 14})


def test_registry_rejects_overlap():
    r = ChannelRegistry(d_model=16)
    r.allocate("cardA", channels=range(0, 4), ch_type="int_scalar")
    with pytest.raises(AllocationError, match="channel 3 conflict"):
        r.allocate("cardB", channels=range(3, 6), ch_type="text")
    # cardB should NOT be registered despite partial success
    assert r.get_card("cardB") is None


def test_registry_rejects_duplicate_card_name():
    r = ChannelRegistry(d_model=16)
    r.allocate("cardA", channels=range(0, 4), ch_type="int_scalar")
    with pytest.raises(AllocationError, match="already registered"):
        r.allocate("cardA", channels=range(4, 8), ch_type="text")


def test_registry_rejects_out_of_range_channel():
    r = ChannelRegistry(d_model=8)
    with pytest.raises(AllocationError, match="out of range"):
        r.allocate("cardA", channels=[5, 10], ch_type="int_scalar")


def test_get_owner_and_free_channels():
    r = ChannelRegistry(d_model=16)
    r.allocate("cardA", channels=[0, 1, 5], ch_type="int_scalar")
    owner = r.get_owner(0)
    assert owner is not None and owner.card_name == "cardA"
    assert r.get_owner(2) is None
    assert r.free_channels() == frozenset(range(16)) - {0, 1, 5}


def test_validate_coverage_raises_on_missing():
    r = ChannelRegistry(d_model=16)
    r.allocate("cardA", channels=[0, 1, 2], ch_type="int_scalar")
    r.validate_coverage({0, 1})  # passes
    with pytest.raises(AllocationError, match=r"\[3\]"):
        r.validate_coverage({0, 3})


def test_adder_tiny_allocation_covers_channels_0_to_9():
    allocs = adder_tiny_allocation()
    all_channels = frozenset()
    for a in allocs:
        assert all_channels.isdisjoint(a.channels), (
            f"adder_tiny allocations overlap: {a}"
        )
        all_channels |= a.channels
    assert all_channels == frozenset(range(10))


def test_register_adder_tiny_into_registry():
    r = ChannelRegistry(d_model=16)
    register_adder_tiny(r)
    # Adder owns 0..9; 10..15 still free
    assert r.all_allocated() == frozenset(range(10))
    assert r.free_channels() == frozenset(range(10, 16))
    # Can still add echo card on free channels
    r.allocate("echo", channels=range(10, 16), ch_type="text",
               purpose="echo output channels for vocab 8-15")
    # And echo conflicts if we try to overlap with adder's step funcs
    with pytest.raises(AllocationError, match="channel 5 conflict"):
        r.allocate("bad", channels=[5, 6], ch_type="anything")


def test_channel_allocation_is_frozen_dataclass():
    alloc = ChannelAllocation(
        card_name="x",
        channels=frozenset({1, 2}),
        ch_type="t",
    )
    # frozen=True means we can't mutate fields
    with pytest.raises(Exception):
        alloc.ch_type = "other"  # type: ignore[misc]


def test_describe_returns_readable_string():
    r = ChannelRegistry(d_model=4)
    r.allocate("cardA", channels=[0, 2], ch_type="int_scalar", purpose="A's ints")
    out = r.describe()
    assert "cardA" in out
    assert "int_scalar" in out
    assert "<free>" in out  # for channels 1, 3


if __name__ == "__main__":
    test_registry_accepts_non_overlapping_allocations()
    print("[ok] non-overlapping allocations")
    test_registry_rejects_overlap()
    print("[ok] rejects overlap")
    test_registry_rejects_duplicate_card_name()
    print("[ok] rejects duplicate card name")
    test_registry_rejects_out_of_range_channel()
    print("[ok] rejects out-of-range channel")
    test_get_owner_and_free_channels()
    print("[ok] get_owner and free_channels")
    test_validate_coverage_raises_on_missing()
    print("[ok] validate_coverage")
    test_adder_tiny_allocation_covers_channels_0_to_9()
    print("[ok] adder_tiny allocation covers 0..9")
    test_register_adder_tiny_into_registry()
    print("[ok] register_adder_tiny works in registry")
    test_channel_allocation_is_frozen_dataclass()
    print("[ok] ChannelAllocation is frozen")
    test_describe_returns_readable_string()
    print("[ok] describe is readable")
