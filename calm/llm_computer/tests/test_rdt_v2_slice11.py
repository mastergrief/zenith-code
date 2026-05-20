"""Slice 11 regression tests: deep supervision (Tier B).

Adds `forward(idx, return_per_iter=False)` kwarg. When True, returns
`(final_log_probs, per_iter_log_probs)` where `per_iter_log_probs` is
a list of length `h_cycles` containing log-probs computed by passing
each per-H-cycle z_H through the head + copy mechanism.

Per co_lead audit msg 1779304303629: must precede S10b ACT training
(ACT loss reuses the same per-iter collection).

Default `return_per_iter=False` preserves prior slice call-sites
bit-equivalently.

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
    reason="Slice 11 tests are GPU-only per user direction; CUDA unavailable",
)


def _make_dt(h_cycles: int = 2, n_iter: int = 2,
             use_inject: bool = False, use_z: bool = False,
             use_h_stack: bool = False, use_h_rmsnorm: bool = False,
             seed: int = 42):
    torch.manual_seed(seed)
    return build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=SEP_ID,
        use_chunkwise=False,
        n_iterations=n_iter, h_cycles=h_cycles,
        use_input_injection=use_inject, use_z_init=use_z,
        use_h_layer_stack=use_h_stack,
        use_h_rmsnorm=use_h_rmsnorm,
    ).to(DEVICE)


# ===== Section A: Default kwarg + backward-compat =====

def test_return_per_iter_default_false_matches_no_kwarg():
    """Calling forward(idx) (no kwarg) vs forward(idx, return_per_iter=False)
    must return bit-identical log-probs."""
    m = _make_dt(h_cycles=2, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        a = m(idx)
        b = m(idx, return_per_iter=False)
    assert torch.equal(a, b)


def test_return_per_iter_false_returns_single_tensor():
    """Default kwarg returns a single tensor, not a tuple. Backward-compat
    with every prior slice's `out = model(idx)` pattern."""
    m = _make_dt(seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out = m(idx)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 6, 20)


# ===== Section B: return_per_iter=True returns tuple =====

def test_return_per_iter_true_returns_tuple():
    """With kwarg on, returns (final_log_probs, per_iter_log_probs_list)."""
    m = _make_dt(h_cycles=2, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out = m(idx, return_per_iter=True)
    assert isinstance(out, tuple)
    assert len(out) == 2
    final, per_iter = out
    assert isinstance(final, torch.Tensor)
    assert isinstance(per_iter, list)


def test_per_iter_list_has_h_cycles_entries():
    """Per-iter list length matches h_cycles."""
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    for h in (1, 2, 3):
        m = _make_dt(h_cycles=h, n_iter=2, seed=42)
        m.eval()
        with torch.no_grad():
            _, per_iter = m(idx, return_per_iter=True)
        assert len(per_iter) == h, f"expected {h} per-iter entries, got {len(per_iter)}"


def test_per_iter_entries_have_correct_shape():
    """Each per-iter entry is log-probs shaped (B, S, vocab)."""
    m = _make_dt(h_cycles=2, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        _, per_iter = m(idx, return_per_iter=True)
    for lp in per_iter:
        assert lp.shape == (1, 6, 20)
        assert torch.isfinite(lp).all()


# ===== Section C: Final entry matches non-per-iter forward =====

def test_final_log_probs_matches_baseline():
    """The (return_per_iter=True) tuple's first element must equal the
    (return_per_iter=False) forward output bit-identically. Confirms
    deep supervision is purely additive collection — doesn't perturb
    the final computation."""
    m = _make_dt(h_cycles=2, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out_baseline = m(idx, return_per_iter=False)
        final, _ = m(idx, return_per_iter=True)
    assert torch.equal(out_baseline, final)


def test_last_per_iter_entry_matches_final():
    """The LAST entry in per_iter_log_probs corresponds to z_H at the
    end of the last H cycle — which is exactly what `_forward_backbone`
    returns as `final_x`. So per_iter[-1] should equal final log-probs."""
    m = _make_dt(h_cycles=2, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        final, per_iter = m(idx, return_per_iter=True)
    # Allow numerical epsilon — both come from `_compute_log_probs`
    # with identical x, but copy mechanism scatter ops can have
    # nondeterministic last-bit floats. Realistically equal.
    assert torch.allclose(final, per_iter[-1], atol=1e-6, rtol=1e-6)


# ===== Section D: H cycles differ → per-iter entries differ =====

def test_per_iter_entries_differ_at_h_gt_1():
    """At h_cycles > 1 with input_injection (so per-cycle z_H states
    diverge meaningfully), the per-iter log-probs entries should NOT
    all be identical — each represents a different point in the
    H-cycle progression."""
    m = _make_dt(h_cycles=3, n_iter=2, use_inject=True, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        _, per_iter = m(idx, return_per_iter=True)
    assert len(per_iter) == 3
    # At least one pair must differ — z_H states evolve across H cycles.
    assert not torch.allclose(per_iter[0], per_iter[-1])


# ===== Section E: Gradient flows through per-iter outputs =====

def test_per_iter_gradient_flows():
    """Deep supervision use case: trainer computes per-iter loss and
    sums them. Backward through that summed loss MUST produce gradient
    on shared weights (W_qkv etc.)."""
    m = _make_dt(h_cycles=2, n_iter=2, seed=42)
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    _, per_iter = m(idx, return_per_iter=True)
    # Simulate a trainer's deep-supervision loss
    loss = sum(lp.sum() for lp in per_iter) / len(per_iter)
    loss.backward()
    g = m.W_qkv[0].weight.grad
    assert g is not None
    assert torch.isfinite(g).all()
    assert g.abs().max().item() > 0


def test_per_iter_gradient_differs_from_final_only():
    """Per-iter loss has DIFFERENT gradient than final-only loss
    (more terms contribute). At h_cycles > 1 the gradients must differ."""
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)

    # Final-only loss
    m_final = _make_dt(h_cycles=2, n_iter=2, seed=42)
    final_only = m_final(idx)
    final_only.sum().backward()
    g_final_only = m_final.W_qkv[0].weight.grad.clone()

    # Deep-supervision loss
    m_deep = _make_dt(h_cycles=2, n_iter=2, seed=42)
    _, per_iter = m_deep(idx, return_per_iter=True)
    deep_loss = sum(lp.sum() for lp in per_iter)
    deep_loss.backward()
    g_deep = m_deep.W_qkv[0].weight.grad.clone()

    assert torch.isfinite(g_final_only).all()
    assert torch.isfinite(g_deep).all()
    assert not torch.allclose(g_final_only, g_deep)


# ===== Section F: Diagnostic exposures unchanged by per-iter =====

def test_last_p_copy_set_from_final_only():
    """`self.last_p_copy` must reflect the FINAL pass through copy
    mechanism, NOT a per-iter intermediate. Trainer-side diagnostics
    (eval_dt_checkpoint reads last_p_copy) should see the same value
    whether deep supervision is on or off."""
    m_a = _make_dt(h_cycles=2, n_iter=2, seed=42)
    m_a.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        m_a(idx, return_per_iter=False)
    p_copy_baseline = m_a.last_p_copy.clone()

    m_b = _make_dt(h_cycles=2, n_iter=2, seed=42)
    m_b.eval()
    with torch.no_grad():
        m_b(idx, return_per_iter=True)
    p_copy_with_per_iter = m_b.last_p_copy.clone()

    # Both came from the FINAL z_H pass through the copy mechanism,
    # not from per-iter intermediates (expose_diagnostics=False on those).
    assert torch.equal(p_copy_baseline, p_copy_with_per_iter)


# ===== Section G: h_cycles=1 degenerate case =====

def test_h_cycles_1_per_iter_has_one_entry():
    """At h_cycles=1 the flat path runs L loop once. Per-iter list
    contains exactly one entry (the final residual), which equals
    final log-probs."""
    m = _make_dt(h_cycles=1, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        final, per_iter = m(idx, return_per_iter=True)
    assert len(per_iter) == 1
    assert torch.allclose(per_iter[0], final, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA unavailable, skipping (GPU-only)")
        raise SystemExit(0)
    for fn_name in [
        "test_return_per_iter_default_false_matches_no_kwarg",
        "test_return_per_iter_false_returns_single_tensor",
        "test_return_per_iter_true_returns_tuple",
        "test_per_iter_list_has_h_cycles_entries",
        "test_per_iter_entries_have_correct_shape",
        "test_final_log_probs_matches_baseline",
        "test_last_per_iter_entry_matches_final",
        "test_per_iter_entries_differ_at_h_gt_1",
        "test_per_iter_gradient_flows",
        "test_per_iter_gradient_differs_from_final_only",
        "test_last_p_copy_set_from_final_only",
        "test_h_cycles_1_per_iter_has_one_entry",
    ]:
        globals()[fn_name]()
    print("All Slice 11 tests PASSED")
