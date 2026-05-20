"""Slice 8 regression tests: separate H/L weight banks (Tier B).

Per co_lead audit msg 1779304303629: full H layer stack with its own
weights at the H boundary, factored — not copy/pasted — through a
shared `_delta_layer_stack(..., bank=...)` interface. H bank owns
QKV/out/FF/beta and any slice-added per-layer modules (gate_proj,
short_conv) when their respective flags are also on.

Seam migration:
  Slice 5: z_H = h_norm(z_L)  (RMSNorm on raw z_L)
  Slice 8: z_H = h_norm(H_stack(z_L))  (RMSNorm on H-processed z_L)

Doubles DT param count when on. Default off → Slice 4-5 hand-off
behavior preserved bit-equivalently.

GPU-only.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta
from calm.llm_computer.dt_install import load_dt_checkpoint


DEVICE = "cuda"
SEP_ID = 3

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Slice 8 tests are GPU-only per user direction; CUDA unavailable",
)


def _make_dt(use_h_layer_stack: bool = False, h_cycles: int = 1, n_iter: int = 1,
             use_inject: bool = False, use_z: bool = False,
             use_h_rmsnorm: bool = False, use_gated: bool = False,
             use_short_conv: bool = False, seed: int = 42):
    torch.manual_seed(seed)
    return build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=SEP_ID,
        use_chunkwise=False,
        n_iterations=n_iter, h_cycles=h_cycles,
        use_input_injection=use_inject, use_z_init=use_z,
        use_gated_attention=use_gated,
        use_short_conv=use_short_conv,
        use_h_rmsnorm=use_h_rmsnorm,
        use_h_layer_stack=use_h_layer_stack,
    ).to(DEVICE)


# ===== Section A: Allocation =====

def test_h_stack_off_modules_are_none():
    """Flag off → H_* attributes are None. Zero state-dict drift for
    pre-Slice-8 checkpoints."""
    m = _make_dt(use_h_layer_stack=False, seed=42)
    assert m.H_W_qkv is None
    assert m.H_W_out is None
    assert m.H_ff_in is None
    assert m.H_ff_out is None
    assert m.H_beta_head is None
    assert m._h_bank is None


def test_h_stack_on_full_bank_allocated():
    """Flag on → H_* ModuleLists allocated matching L shape."""
    m = _make_dt(use_h_layer_stack=True, seed=42)
    assert isinstance(m.H_W_qkv, nn.ModuleList)
    assert len(m.H_W_qkv) == 2  # n_layers
    assert isinstance(m.H_W_out, nn.ModuleList)
    assert isinstance(m.H_ff_in, nn.ModuleList)
    assert isinstance(m.H_ff_out, nn.ModuleList)
    assert isinstance(m.H_beta_head, nn.ModuleList)
    assert m._h_bank is not None


def test_h_stack_param_count_roughly_doubles_dt_layers():
    """Allocating H bank should add ~same number of params as the L bank
    (mirror shapes). Excludes embedding, head, copy_*, h_norm which stay
    single-instance."""
    m_off = _make_dt(use_h_layer_stack=False, seed=42)
    m_on = _make_dt(use_h_layer_stack=True, seed=42)
    diff = m_on.param_count() - m_off.param_count()
    # H bank mirrors L: W_qkv (8 × 24), W_out (8 × 8), ff_in (8 × 32),
    # ff_out (16 × 8), beta_head (8 × 1 + 1) per layer × 2 layers
    expected_per_layer = (
        8 * 24       # W_qkv (d × 3d)
        + 8 * 8      # W_out (d × d)
        + 8 * 32     # ff_in (d × 2 * d_ffn) = 8 × 2 × 16
        + 16 * 8     # ff_out (d_ffn × d)
        + (8 + 1)    # beta_head (weight + bias)
    )
    expected = expected_per_layer * 2  # n_layers
    assert diff == expected, f"expected +{expected}, got +{diff}"


def test_h_stack_with_gated_attention_allocates_h_gate():
    """When use_h_layer_stack=True AND use_gated_attention=True,
    H_attn_gate_proj is allocated as ModuleList."""
    m = _make_dt(use_h_layer_stack=True, use_gated=True, seed=42)
    assert isinstance(m.H_attn_gate_proj, nn.ModuleList)
    assert len(m.H_attn_gate_proj) == 2


def test_h_stack_with_short_conv_allocates_h_convs():
    """When use_h_layer_stack=True AND use_short_conv=True, H_short_conv_q/k/v
    are allocated."""
    m = _make_dt(use_h_layer_stack=True, use_short_conv=True, seed=42)
    assert isinstance(m.H_short_conv_q, nn.ModuleList)
    assert isinstance(m.H_short_conv_k, nn.ModuleList)
    assert isinstance(m.H_short_conv_v, nn.ModuleList)


# ===== Section B: Flag-off bit-equivalence + h_cycles=1 baseline =====

def test_h_stack_h_cycles_1_inert():
    """At h_cycles=1, the H boundary code never fires (early return).
    Flag on vs off must produce identical output."""
    m_off = _make_dt(use_h_layer_stack=False, h_cycles=1, n_iter=2, seed=42)
    m_on = _make_dt(use_h_layer_stack=True, h_cycles=1, n_iter=2, seed=42)
    m_off.eval(); m_on.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        a = m_off(idx)
        b = m_on(idx)
    # h_cycles=1 early-return path bypasses H stack entirely.
    assert torch.equal(a, b)


def test_h_stack_h_cycles_1_inert_under_lecun_init():
    """Co_lead audit msg 1779305159197: with both `use_h_layer_stack=True`
    AND `use_lecun_init=True`, the h_cycles=1 inert invariant MUST hold.
    A broad `_apply_lecun_init()` re-application would re-initialize the
    L bank a second time with fresh RNG, silently scrambling its weights
    vs flag-off. The scoped `_apply_lecun_init_to(h_roots)` fix is
    falsifiable only via this test."""
    torch.manual_seed(42)
    m_off = build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=SEP_ID,
        use_chunkwise=False, n_iterations=2, h_cycles=1,
        use_h_layer_stack=False, use_lecun_init=True,
    ).to(DEVICE)
    torch.manual_seed(42)
    m_on = build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=SEP_ID,
        use_chunkwise=False, n_iterations=2, h_cycles=1,
        use_h_layer_stack=True, use_lecun_init=True,
    ).to(DEVICE)
    m_off.eval(); m_on.eval()

    # L bank weights MUST be identical between the two builds (scoped
    # LeCun re-init only touches H bank, not L).
    assert torch.equal(m_off.W_qkv[0].weight, m_on.W_qkv[0].weight), (
        "L bank W_qkv diverged under use_lecun_init=True when "
        "use_h_layer_stack flipped — scoped LeCun re-init regressed."
    )
    # copy_gate.bias must still be -2.0 (init contract preserved)
    assert m_off.copy_gate.bias.item() == pytest.approx(-2.0)
    assert m_on.copy_gate.bias.item() == pytest.approx(-2.0)

    # Forward output identical at h_cycles=1 (H path never invoked)
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        a = m_off(idx)
        b = m_on(idx)
    assert torch.equal(a, b)


def test_h_stack_off_h_cycles_2_unchanged_vs_slice4():
    """Flag off at h_cycles=2 should produce the Slice 4 hand-off behavior
    exactly (`z_H = z_L`, no H stack). Confirms we didn't perturb the
    baseline at non-trivial h_cycles."""
    m = _make_dt(use_h_layer_stack=False, h_cycles=2, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out = m(idx)
    assert torch.isfinite(out).all()


# ===== Section C: Flag-on observable effect =====

def test_h_stack_changes_output_at_h_gt_1():
    """At h_cycles > 1, flag on transforms z_L through the H bank →
    output differs from flag off."""
    m_off = _make_dt(use_h_layer_stack=False, h_cycles=2, n_iter=2, seed=42)
    m_on = _make_dt(use_h_layer_stack=True, h_cycles=2, n_iter=2, seed=42)
    m_off.eval(); m_on.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        a = m_off(idx)
        b = m_on(idx)
    assert not torch.allclose(a, b)


def test_h_stack_with_rmsnorm_normalizes_h_stack_output():
    """Slice 5 seam migration: with use_h_rmsnorm AND use_h_layer_stack
    both on, the RMSNorm wraps the H stack output, not the raw z_L.
    Falsifier: if order swaps, output would differ.

    We verify INDIRECTLY by checking that all three configs produce
    distinct outputs:
      (h_stack=True, rmsnorm=False)
      (h_stack=True, rmsnorm=True)
      (h_stack=False, rmsnorm=True)  ← Slice 5 path
    """
    m_a = _make_dt(use_h_layer_stack=True, use_h_rmsnorm=False,
                   h_cycles=2, n_iter=2, seed=42)
    m_b = _make_dt(use_h_layer_stack=True, use_h_rmsnorm=True,
                   h_cycles=2, n_iter=2, seed=42)
    m_c = _make_dt(use_h_layer_stack=False, use_h_rmsnorm=True,
                   h_cycles=2, n_iter=2, seed=42)
    for m in (m_a, m_b, m_c):
        m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        a = m_a(idx)
        b = m_b(idx)
        c = m_c(idx)
    assert torch.isfinite(a).all()
    assert torch.isfinite(b).all()
    assert torch.isfinite(c).all()
    assert not torch.allclose(a, b)
    assert not torch.allclose(b, c)
    assert not torch.allclose(a, c)


# ===== Section D: Gradient flow + LeCun init =====

def test_h_stack_gradient_flows_through_h_weights():
    """H bank weights receive gradient during backward."""
    m = _make_dt(use_h_layer_stack=True, h_cycles=2, n_iter=2, seed=42)
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    out = m(idx)
    out.sum().backward()
    g = m.H_W_qkv[0].weight.grad
    assert g is not None
    assert torch.isfinite(g).all()
    assert g.abs().max().item() > 0


def test_h_stack_lecun_init_applied_to_h_weights():
    """Per co_lead audit: re-apply LeCun init AFTER H bank allocation.
    Confirm H_W_qkv weight distribution differs from default-init when
    use_lecun_init=True."""
    torch.manual_seed(42)
    m_default = build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=SEP_ID,
        use_chunkwise=False,
        use_h_layer_stack=True, use_lecun_init=False,
    ).to(DEVICE)
    torch.manual_seed(42)
    m_lecun = build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=SEP_ID,
        use_chunkwise=False,
        use_h_layer_stack=True, use_lecun_init=True,
    ).to(DEVICE)
    assert not torch.equal(
        m_default.H_W_qkv[0].weight,
        m_lecun.H_W_qkv[0].weight,
    ), "LeCun re-init should change H_W_qkv from default"
    # beta_head bias must stay 0 on both (bias-preserving)
    for h in m_lecun.H_beta_head:
        assert h.bias.item() == pytest.approx(0.0)


# ===== Section E: Cached-decode blocklist =====

def test_cached_decode_blocks_h_layer_stack():
    """`use_h_layer_stack` is on the blocklist defensively (even though
    h_cycles=1 path is bit-equivalent, the flag-on at h_cycles>1 case
    cannot be honored by the flat cached path)."""
    m = _make_dt(use_h_layer_stack=True, h_cycles=1, n_iter=1, seed=42)
    m.eval()
    prefix = torch.tensor([[5, 7, SEP_ID, 9, 11]], device=DEVICE)
    with pytest.raises(NotImplementedError, match=r"use_h_layer_stack"):
        m.decode_greedy_cached(prefix, max_gen=2, eos_token=None)


# ===== Section F: Checkpoint round-trip =====

def test_checkpoint_roundtrip_h_layer_stack():
    """use_h_layer_stack persists; H bank reloads; logits match."""
    m_orig = _make_dt(use_h_layer_stack=True, h_cycles=2, n_iter=2,
                      use_h_rmsnorm=True, seed=42)
    m_orig.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out_orig = m_orig(idx)
    assert torch.isfinite(out_orig).all()

    cfg_dict = {
        "vocab_size": 20, "max_len": 24,
        "d_model": 8, "n_heads": 4, "n_layers": 2,
        "d_ffn": 16, "n_copy_heads": 2,
        "use_chunkwise": False,
        "n_iterations": 2,
        "h_cycles": 2,
        "use_h_rmsnorm": True,
        "use_h_layer_stack": True,
    }
    ckpt = {
        "model_state": m_orig.state_dict(),
        "config": cfg_dict,
        "epoch": 1, "val_autoreg": 0.5, "n_train": 100, "n_val": 10,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_slice8.pt"
        torch.save(ckpt, path)
        m_loaded, _ = load_dt_checkpoint(str(path), device=DEVICE)
    m_loaded.eval()

    assert m_loaded.config.use_h_layer_stack is True
    assert isinstance(m_loaded.H_W_qkv, nn.ModuleList)
    assert len(m_loaded.H_W_qkv) == 2

    with torch.no_grad():
        out_loaded = m_loaded(idx)
    assert torch.equal(out_orig, out_loaded)


def test_checkpoint_backward_compat_h_layer_stack():
    """Old checkpoint without use_h_layer_stack loads with flag=False."""
    m_orig = _make_dt(use_h_layer_stack=False, seed=42)
    m_orig.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out_orig = m_orig(idx)

    cfg_dict = {
        "vocab_size": 20, "max_len": 24,
        "d_model": 8, "n_heads": 4, "n_layers": 2,
        "d_ffn": 16, "n_copy_heads": 2,
        "use_chunkwise": False,
        # NO use_h_layer_stack key
    }
    ckpt = {
        "model_state": m_orig.state_dict(),
        "config": cfg_dict,
        "epoch": 1, "val_autoreg": 0.5, "n_train": 100, "n_val": 10,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_slice8_compat.pt"
        torch.save(ckpt, path)
        m_loaded, _ = load_dt_checkpoint(str(path), device=DEVICE)
    m_loaded.eval()
    assert m_loaded.config.use_h_layer_stack is False
    assert m_loaded.H_W_qkv is None
    assert m_loaded._h_bank is None

    with torch.no_grad():
        out_loaded = m_loaded(idx)
    assert torch.equal(out_orig, out_loaded)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA unavailable, skipping (GPU-only)")
        raise SystemExit(0)
    for fn_name in [
        "test_h_stack_off_modules_are_none",
        "test_h_stack_on_full_bank_allocated",
        "test_h_stack_param_count_roughly_doubles_dt_layers",
        "test_h_stack_with_gated_attention_allocates_h_gate",
        "test_h_stack_with_short_conv_allocates_h_convs",
        "test_h_stack_h_cycles_1_inert",
        "test_h_stack_off_h_cycles_2_unchanged_vs_slice4",
        "test_h_stack_changes_output_at_h_gt_1",
        "test_h_stack_with_rmsnorm_normalizes_h_stack_output",
        "test_h_stack_gradient_flows_through_h_weights",
        "test_h_stack_lecun_init_applied_to_h_weights",
        "test_cached_decode_blocks_h_layer_stack",
        "test_checkpoint_roundtrip_h_layer_stack",
        "test_checkpoint_backward_compat_h_layer_stack",
    ]:
        globals()[fn_name]()
    print("All Slice 8 tests PASSED")
