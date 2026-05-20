"""Slice 9 regression tests: residual H/L carry (Tier B).

LAST Tier B slice. Per co_lead audit msg 1779304303629: explicit
`carry=None, return_carry=False` kwargs (NO mutable model.carry state).
**Residual** H/L carry — `_delta_layer_stack` zeroes DeltaNet fast-
weight S_state per call, so carry only preserves z_H/z_L RESIDUALS
between forwards, NOT streaming DeltaNet cache. Gated by config flag
`use_carry=True` (opt-in by construction so cached-decode blocklist
can refuse such models cleanly).

GPU-only.
"""
from __future__ import annotations

import pytest
import torch

from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta


DEVICE = "cuda"
SEP_ID = 3

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Slice 9 tests are GPU-only per user direction; CUDA unavailable",
)


def _make_dt(use_carry: bool = False, h_cycles: int = 2, n_iter: int = 2,
             seed: int = 42):
    torch.manual_seed(seed)
    return build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=SEP_ID,
        use_chunkwise=False,
        n_iterations=n_iter, h_cycles=h_cycles,
        use_carry=use_carry,
    ).to(DEVICE)


# ===== Section A: use_carry gate =====

def test_carry_kwarg_without_use_carry_flag_raises():
    """Calling forward with carry kwargs on a model built with
    use_carry=False raises ValueError citing the missing flag."""
    m = _make_dt(use_carry=False, h_cycles=2, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with pytest.raises(ValueError, match=r"use_carry=True"):
        with torch.no_grad():
            m(idx, return_carry=True)

    fake_carry = torch.zeros(1, 6, 8, device=DEVICE)
    with pytest.raises(ValueError, match=r"use_carry=True"):
        with torch.no_grad():
            m(idx, carry=fake_carry)


def test_default_forward_unchanged_when_use_carry_on():
    """use_carry=True alone (no carry kwarg) → forward output bit-
    identical to use_carry=False. The flag is declarative; without
    runtime kwargs it doesn't perturb computation."""
    m_off = _make_dt(use_carry=False, h_cycles=2, n_iter=2, seed=42)
    m_on = _make_dt(use_carry=True, h_cycles=2, n_iter=2, seed=42)
    m_off.eval(); m_on.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        a = m_off(idx)
        b = m_on(idx)
    assert torch.equal(a, b)


# ===== Section B: return_carry returns a tensor =====

def test_return_carry_true_returns_tuple_with_carry():
    """forward(idx, return_carry=True) returns (logits, carry) tuple."""
    m = _make_dt(use_carry=True, h_cycles=2, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        result = m(idx, return_carry=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    logits, carry = result
    assert carry.shape == (1, 6, 8)  # (B, S, d_model)
    assert torch.isfinite(carry).all()


def test_return_carry_carry_is_detached():
    """The returned carry tensor must be detached — no gradient flows
    back through it into a prior forward's computation graph."""
    m = _make_dt(use_carry=True, h_cycles=2, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        _, carry = m(idx, return_carry=True)
    assert not carry.requires_grad
    assert carry.grad_fn is None


# ===== Section C: carry-in changes output =====

def test_passing_carry_changes_output():
    """Passing a non-default carry as the H-loop initial state produces
    different output than carry=None (the default which uses tok+pos
    embedding)."""
    m = _make_dt(use_carry=True, h_cycles=2, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    # Get a carry from a prior forward
    with torch.no_grad():
        _, carry_out = m(idx, return_carry=True)
    # Now feed it back as initial state
    with torch.no_grad():
        out_with_carry = m(idx, carry=carry_out)
        out_without_carry = m(idx)
    # Result with carry-as-initial-state must differ from default
    assert not torch.allclose(out_with_carry, out_without_carry)


# ===== Section D: shape validation =====

def test_carry_shape_mismatch_raises():
    """Passing a mismatched-shape carry tensor raises ValueError citing
    the expected shape."""
    m = _make_dt(use_carry=True, h_cycles=2, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    # idx is (1, 6) → expected carry shape (1, 6, 8). Pass wrong shape.
    bad_carry = torch.zeros(1, 5, 8, device=DEVICE)  # wrong seq length
    with pytest.raises(ValueError, match=r"carry shape"):
        with torch.no_grad():
            m(idx, carry=bad_carry)


# ===== Section E: composes with return_per_iter =====

def test_return_per_iter_plus_return_carry_returns_three_tuple():
    """Both kwargs on → tuple of (logits, per_iter_list, carry)."""
    m = _make_dt(use_carry=True, h_cycles=2, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        result = m(idx, return_per_iter=True, return_carry=True)
    assert isinstance(result, tuple)
    assert len(result) == 3
    logits, per_iter, carry = result
    assert isinstance(per_iter, list)
    assert len(per_iter) == 2  # h_cycles
    assert carry.shape == (1, 6, 8)


# ===== Section F: cached-decode blocklist =====

def test_cached_decode_blocks_use_carry():
    """use_carry on config → cached decode raises NotImplementedError."""
    m = _make_dt(use_carry=True, h_cycles=1, n_iter=1, seed=42)
    m.eval()
    prefix = torch.tensor([[5, 7, SEP_ID, 9, 11]], device=DEVICE)
    with pytest.raises(NotImplementedError, match=r"use_carry"):
        m.decode_greedy_cached(prefix, max_gen=2, eos_token=None)


# ===== Section G: h_cycles=1 carry =====

def test_carry_at_h_cycles_1_works():
    """At h_cycles=1 (early-return path) carry must also work — it's
    the single H cycle's initial state."""
    m = _make_dt(use_carry=True, h_cycles=1, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        _, carry_out = m(idx, return_carry=True)
        out_resumed = m(idx, carry=carry_out)
        out_default = m(idx)
    assert carry_out.shape == (1, 6, 8)
    assert not torch.allclose(out_resumed, out_default)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA unavailable, skipping (GPU-only)")
        raise SystemExit(0)
    for fn_name in [
        "test_carry_kwarg_without_use_carry_flag_raises",
        "test_default_forward_unchanged_when_use_carry_on",
        "test_return_carry_true_returns_tuple_with_carry",
        "test_return_carry_carry_is_detached",
        "test_passing_carry_changes_output",
        "test_carry_shape_mismatch_raises",
        "test_return_per_iter_plus_return_carry_returns_three_tuple",
        "test_cached_decode_blocks_use_carry",
        "test_carry_at_h_cycles_1_works",
    ]:
        globals()[fn_name]()
    print("All Slice 9 tests PASSED")
