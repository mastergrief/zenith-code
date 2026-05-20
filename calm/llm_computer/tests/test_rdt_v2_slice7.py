"""Slice 7 regression tests: global bp_warmup grain (Tier B).

Generalizes Slice 3 bp_warmup from L-grain (per L-loop uniform) to
'global' grain (flattened h_cycles × n_iters). Only meaningful when
`h_cycles > 1` (Slice 8 dependency in board task #28). When the whole
H cycle would be detached under 'global', the H stack call between
cycles also runs under no_grad.

Per co_lead audit msg 1779304303629: keep L-grain default (Slice 3
backward behavior preserved); add 'global' as opt-in via
`model.bp_warmup_ctx(k, grain='global')`.

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
    reason="Slice 7 tests are GPU-only per user direction; CUDA unavailable",
)


def _make_dt(h_cycles: int = 2, n_iter: int = 3, use_inject: bool = False,
             use_h_stack: bool = False, seed: int = 42):
    torch.manual_seed(seed)
    return build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=SEP_ID,
        use_chunkwise=False,
        n_iterations=n_iter, h_cycles=h_cycles,
        use_input_injection=use_inject,
        use_h_layer_stack=use_h_stack,
    ).to(DEVICE)


# ===== Section A: Default + grain validation =====

def test_bp_warmup_grain_default_is_l():
    """`_bp_warmup_grain` defaults to 'l' (Slice 3 backward behavior)."""
    m = _make_dt(seed=42)
    assert m._bp_warmup_grain == "l"


def test_bp_warmup_grain_validation_rejects_unknown():
    """`bp_warmup_ctx(k, grain=...)` rejects values outside {'l', 'global'}."""
    m = _make_dt(seed=42)
    with pytest.raises(ValueError, match=r"grain must be"):
        with m.bp_warmup_ctx(1, grain="batch"):
            pass


def test_bp_warmup_ctx_restores_grain_on_exit():
    """Grain reverts to 'l' (or prior value) after context exit."""
    m = _make_dt(seed=42)
    assert m._bp_warmup_grain == "l"
    with m.bp_warmup_ctx(1, grain="global"):
        assert m._bp_warmup_grain == "global"
        assert m._bp_warmup_active_iters == 1
    assert m._bp_warmup_grain == "l"
    assert m._bp_warmup_active_iters is None


# ===== Section B: 'l' grain bit-equivalent to Slice 3 =====

def test_bp_warmup_l_grain_explicit_matches_default():
    """Explicitly passing grain='l' (Slice 7 opt-in) must match the
    no-grain-arg call (Slice 3 default behavior). Bit-equivalence."""
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)

    m_a = _make_dt(h_cycles=2, n_iter=2, use_inject=False, seed=42)
    with m_a.bp_warmup_ctx(1):
        out = m_a(idx)
        out.sum().backward()
    g_default = m_a.W_qkv[0].weight.grad.clone()

    m_b = _make_dt(h_cycles=2, n_iter=2, use_inject=False, seed=42)
    with m_b.bp_warmup_ctx(1, grain="l"):
        out = m_b(idx)
        out.sum().backward()
    g_explicit_l = m_b.W_qkv[0].weight.grad.clone()

    assert torch.equal(g_default, g_explicit_l)


# ===== Section C: 'global' grain produces different gradient =====

def test_bp_warmup_global_differs_from_l_at_h_gt_1():
    """At h_cycles > 1, the distinction between 'l' and 'global' grains
    surfaces when `bp_active == n_iters` (NOT bp_active=1 — see comment
    block). At bp_active=n_iters:

    - 'l' grain → detach_until = 0 per L loop → full L grad in EVERY
      H cycle → gradient flows ACROSS H cycles (no detach severs
      cross-H carry).
    - 'global' grain → per_h_detach_until = [n_iters, 0] for h=2 →
      cycle 0 fully detached, cycle 1 full grad → gradient flows
      through ONLY cycle 1's L iters.

    Different number of contributing L iters → different W_qkv grad.

    Why bp_active=1 doesn't distinguish: at bp_active=1, L-grain has
    detach_until = n_iters - 1 > 0, so iter 0's `x_l.detach()` line
    in `_run_l_loop` already severs cross-H gradient incidentally.
    Both grains converge to "only last L iter of last H cycle". The
    distinction surfaces only when L-grain has NO L-loop detach
    (detach_until == 0)."""
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)

    m_l = _make_dt(h_cycles=2, n_iter=2, use_inject=False, seed=42)
    with m_l.bp_warmup_ctx(2, grain="l"):  # bp_active = n_iters
        out = m_l(idx)
        out.sum().backward()
    g_l = m_l.W_qkv[0].weight.grad.clone()

    m_g = _make_dt(h_cycles=2, n_iter=2, use_inject=False, seed=42)
    with m_g.bp_warmup_ctx(2, grain="global"):
        out = m_g(idx)
        out.sum().backward()
    g_global = m_g.W_qkv[0].weight.grad.clone()

    assert torch.isfinite(g_l).all() and torch.isfinite(g_global).all()
    assert not torch.allclose(g_l, g_global)


def test_bp_warmup_global_h_cycles_1_collapses_to_l():
    """At h_cycles=1 the two grains MUST collapse — there's only one H
    cycle so 'global' and 'l' express the same constraint."""
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)

    m_l = _make_dt(h_cycles=1, n_iter=3, use_inject=False, seed=42)
    with m_l.bp_warmup_ctx(1, grain="l"):
        out = m_l(idx)
        out.sum().backward()
    g_l = m_l.W_qkv[0].weight.grad.clone()

    m_g = _make_dt(h_cycles=1, n_iter=3, use_inject=False, seed=42)
    with m_g.bp_warmup_ctx(1, grain="global"):
        out = m_g(idx)
        out.sum().backward()
    g_global = m_g.W_qkv[0].weight.grad.clone()

    assert torch.equal(g_l, g_global)


# ===== Section D: 'global' grain with H bank (Slice 8 dependency) =====

def test_bp_warmup_global_with_h_stack_only_last_h_cycle_gets_h_grad():
    """With use_h_layer_stack=True + bp_active=n_iters:
    - 'l' grain: H stack call BOTH between cycles AND after last cycle
      contribute gradient to H bank (h_detach_set is empty).
    - 'global' grain: H stack call between cycle 0 and 1 runs under
      no_grad (h_detach_set = {0}), so only the H stack call after
      cycle 1 contributes gradient.

    H bank gradient SHAPES must differ across the two grains."""
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)

    m_l = _make_dt(h_cycles=2, n_iter=2, use_h_stack=True,
                   use_inject=False, seed=42)
    with m_l.bp_warmup_ctx(2, grain="l"):  # n_iters → full L grad per cycle
        out = m_l(idx)
        out.sum().backward()
    g_h_l = m_l.H_W_qkv[0].weight.grad.clone()

    m_g = _make_dt(h_cycles=2, n_iter=2, use_h_stack=True,
                   use_inject=False, seed=42)
    with m_g.bp_warmup_ctx(2, grain="global"):
        out = m_g(idx)
        out.sum().backward()
    g_h_global = m_g.H_W_qkv[0].weight.grad.clone()

    assert torch.isfinite(g_h_l).all()
    assert torch.isfinite(g_h_global).all()
    assert g_h_l.abs().max().item() > 0
    assert g_h_global.abs().max().item() > 0
    # l-grain has BOTH H stack calls contributing; global has only the
    # last → different gradient shape.
    assert not torch.allclose(g_h_l, g_h_global)


# ===== Section E: 'global' grain with bp_active >= total → full grad =====

def test_bp_warmup_global_bp_active_above_total_full_grad():
    """When bp_active >= h_cycles * n_iters, 'global' grain collapses to
    full gradient flow (same as bp_active=None)."""
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    # Use a fixed seed-pinned sample for full reproducibility
    m_a = _make_dt(h_cycles=2, n_iter=2, use_inject=False, seed=42)
    # No ctx → full grad
    out = m_a(idx)
    out.sum().backward()
    g_full = m_a.W_qkv[0].weight.grad.clone()

    m_b = _make_dt(h_cycles=2, n_iter=2, use_inject=False, seed=42)
    # bp_active=4 == h_cycles*n_iters → full grad
    with m_b.bp_warmup_ctx(4, grain="global"):
        out = m_b(idx)
        out.sum().backward()
    g_full_via_ctx = m_b.W_qkv[0].weight.grad.clone()

    assert torch.equal(g_full, g_full_via_ctx)


# ===== Section F: Forward bit-equivalence (bp_warmup is backward-only) =====

def test_bp_warmup_global_forward_unchanged():
    """bp_warmup is a BACKWARD-side concern only — forward output must
    be bit-identical with/without the ctx (same as Slice 3 invariant)."""
    m = _make_dt(h_cycles=2, n_iter=2, use_inject=False, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out_no_ctx = m(idx)
    with torch.no_grad():
        with m.bp_warmup_ctx(1, grain="global"):
            out_global = m(idx)
    assert torch.equal(out_no_ctx, out_global)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA unavailable, skipping (GPU-only)")
        raise SystemExit(0)
    for fn_name in [
        "test_bp_warmup_grain_default_is_l",
        "test_bp_warmup_grain_validation_rejects_unknown",
        "test_bp_warmup_ctx_restores_grain_on_exit",
        "test_bp_warmup_l_grain_explicit_matches_default",
        "test_bp_warmup_global_differs_from_l_at_h_gt_1",
        "test_bp_warmup_global_h_cycles_1_collapses_to_l",
        "test_bp_warmup_global_with_h_stack_only_last_h_cycle_gets_h_grad",
        "test_bp_warmup_global_bp_active_above_total_full_grad",
        "test_bp_warmup_global_forward_unchanged",
    ]:
        globals()[fn_name]()
    print("All Slice 7 tests PASSED")
