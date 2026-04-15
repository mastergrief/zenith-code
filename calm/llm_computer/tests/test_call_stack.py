"""Tests for call-stack residual region."""

from __future__ import annotations

import pytest
import torch

from calm.llm_computer.call_stack import (
    CallStackConfig, depth, get_tos, peek, pop, push, set_tos,
)


def _cfg():
    # d_model=16: stack region = channels 0-11 (3 frames of 4), TOS = channel 15
    return CallStackConfig(
        stack_channels=slice(0, 12),
        tos_channel=15,
        frame_size=4,
        max_depth=3,
    )


def test_validate_rejects_overlap_and_wrong_size():
    # Stack size doesn't match max_depth * frame_size
    bad = CallStackConfig(
        stack_channels=slice(0, 10), tos_channel=15,
        frame_size=4, max_depth=3,
    )
    with pytest.raises(AssertionError, match="width"):
        bad.validate(16)
    # TOS inside stack region
    bad2 = CallStackConfig(
        stack_channels=slice(0, 12), tos_channel=5,
        frame_size=4, max_depth=3,
    )
    with pytest.raises(AssertionError, match="inside stack region"):
        bad2.validate(16)


def test_push_and_pop_roundtrip():
    cfg = _cfg()
    x = torch.zeros(1, 2, 16)
    frame = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x = push(x, frame, cfg)
    assert get_tos(x, cfg) == 1
    x_after, popped = pop(x, cfg)
    assert torch.allclose(popped, frame.unsqueeze(0))
    assert get_tos(x_after, cfg) == 0


def test_multiple_pushes_advance_tos():
    cfg = _cfg()
    x = torch.zeros(1, 2, 16)
    for i in range(3):
        frame = torch.full((4,), float(i + 1))
        x = push(x, frame, cfg)
        assert get_tos(x, cfg) == i + 1


def test_stack_overflow_raises():
    cfg = _cfg()
    x = torch.zeros(1, 2, 16)
    for _ in range(3):
        x = push(x, torch.ones(4), cfg)
    with pytest.raises(IndexError, match="overflow"):
        push(x, torch.ones(4), cfg)


def test_stack_underflow_raises():
    cfg = _cfg()
    x = torch.zeros(1, 2, 16)
    with pytest.raises(IndexError, match="underflow"):
        pop(x, cfg)


def test_pop_returns_most_recent():
    cfg = _cfg()
    x = torch.zeros(1, 2, 16)
    x = push(x, torch.tensor([1.0, 0, 0, 0]), cfg)
    x = push(x, torch.tensor([2.0, 0, 0, 0]), cfg)
    x = push(x, torch.tensor([3.0, 0, 0, 0]), cfg)
    # LIFO: pop returns 3 first
    x, a = pop(x, cfg)
    assert a[0, 0].item() == 3.0
    x, b = pop(x, cfg)
    assert b[0, 0].item() == 2.0
    x, c = pop(x, cfg)
    assert c[0, 0].item() == 1.0


def test_peek_no_mutation():
    cfg = _cfg()
    x = torch.zeros(1, 2, 16)
    x = push(x, torch.tensor([7.0, 8.0, 9.0, 10.0]), cfg)
    before_tos = get_tos(x, cfg)
    peeked = peek(x, cfg)
    assert torch.allclose(peeked, torch.tensor([[7.0, 8.0, 9.0, 10.0]]))
    assert get_tos(x, cfg) == before_tos  # unchanged


def test_peek_with_offset():
    cfg = _cfg()
    x = torch.zeros(1, 2, 16)
    x = push(x, torch.tensor([1.0, 0, 0, 0]), cfg)  # depth 0
    x = push(x, torch.tensor([2.0, 0, 0, 0]), cfg)  # depth 1 (TOS = 2)
    # peek offset=0 → top (depth 1) = [2,0,0,0]
    # peek offset=1 → below (depth 0) = [1,0,0,0]
    assert peek(x, cfg, offset=0)[0, 0].item() == 2.0
    assert peek(x, cfg, offset=1)[0, 0].item() == 1.0


def test_peek_past_bottom_raises():
    cfg = _cfg()
    x = torch.zeros(1, 2, 16)
    x = push(x, torch.ones(4), cfg)
    with pytest.raises(IndexError, match="past stack bottom"):
        peek(x, cfg, offset=5)


def test_set_tos_clamps_range():
    cfg = _cfg()
    x = torch.zeros(1, 2, 16)
    with pytest.raises(AssertionError, match="out of range"):
        set_tos(x, cfg, new_tos=5)  # > max_depth=3


def test_depth_utility():
    cfg = _cfg()
    x = torch.zeros(1, 2, 16)
    assert depth(x, cfg) == 0
    x = push(x, torch.ones(4), cfg)
    x = push(x, torch.ones(4), cfg)
    assert depth(x, cfg) == 2


def test_pop_clears_slot():
    """After pop, the slot should be zeroed so future pushes don't
    leak old data."""
    cfg = _cfg()
    x = torch.zeros(1, 2, 16)
    x = push(x, torch.tensor([5.0, 6.0, 7.0, 8.0]), cfg)
    x, _ = pop(x, cfg)
    # Check the region that was just popped is zero
    assert (x[:, 0, 0:4] == 0).all()


def test_recursive_simulation():
    """Simulate a recursive call: push parent context, do "work",
    pop back. Working memory preserves parent state."""
    cfg = _cfg()
    x = torch.zeros(1, 2, 16)
    # Push parent context [a=3, b=5]
    x = push(x, torch.tensor([3.0, 5.0, 0, 0]), cfg)
    # "Recursive call" pushes a sub-context [c=7, d=11]
    x = push(x, torch.tensor([7.0, 11.0, 0, 0]), cfg)
    # Do some "work" with the sub-context (peek + compute)
    sub = peek(x, cfg)
    assert sub[0, 0].item() == 7.0
    assert sub[0, 1].item() == 11.0
    # Pop sub-frame; parent should be exposed again
    x, sub_popped = pop(x, cfg)
    parent = peek(x, cfg)
    assert parent[0, 0].item() == 3.0
    assert parent[0, 1].item() == 5.0


if __name__ == "__main__":
    test_validate_rejects_overlap_and_wrong_size()
    print("[ok] validate catches wrong size and overlap")
    test_push_and_pop_roundtrip()
    print("[ok] push / pop round-trip")
    test_multiple_pushes_advance_tos()
    print("[ok] multi-push advances TOS")
    test_stack_overflow_raises()
    print("[ok] overflow raises")
    test_stack_underflow_raises()
    print("[ok] underflow raises")
    test_pop_returns_most_recent()
    print("[ok] LIFO order preserved")
    test_peek_no_mutation()
    print("[ok] peek doesn't mutate")
    test_peek_with_offset()
    print("[ok] peek with offset")
    test_peek_past_bottom_raises()
    print("[ok] peek past bottom raises")
    test_set_tos_clamps_range()
    print("[ok] set_tos range-checked")
    test_depth_utility()
    print("[ok] depth() utility")
    test_pop_clears_slot()
    print("[ok] pop clears slot")
    test_recursive_simulation()
    print("[ok] recursive simulation works")
