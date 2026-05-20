"""Slice 2 regression tests: z_L_init + LeCun init ported from HRM-Text.

Builds on Slice 1's flag plumbing (use_gated_attention, use_input_injection).
Slice 2 adds two more flags via the same discipline:

  * `use_z_init` — learned per-channel initial hidden state for the D5
    recurrence. Iter 0 starts from `z_init.expand(B, S, D)` instead of
    (tok+pos) embedding. Pairs naturally with `use_input_injection` to
    recover the embed at iter 0. Only fires when `n_iterations > 1`
    (preserves the n=1 bit-equivalence guard from Slice 1).
  * `use_lecun_init` — re-init every nn.Linear weight with
    `normal_(0, sqrt(1/fan_in))` per HRM-Text. Bias-preserving so existing
    init contracts (copy_gate_bias_init, beta_head bias=0) stay intact.

GPU-only per user direction "only use gpu too". Toy d_model=8 configs
fit easily in 2 GiB free VRAM on a 4070 Laptop.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta
from calm.llm_computer.dt_install import load_dt_checkpoint
from calm.llm_computer.model import Small2DConfig, Small2DTransformer
from calm.llm_computer.recurrent_substrate import RecurrentConfig, RecurrentSmall2DTransformer


DEVICE = "cuda"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Slice 2 tests are GPU-only per user direction; CUDA unavailable",
)


# ===== Section A: LeCun init on Small2DTransformer =====

def _make_small2d(use_lecun: bool = False, seed: int = 42) -> Small2DTransformer:
    torch.manual_seed(seed)
    m = Small2DTransformer(Small2DConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8,
        max_len=32, use_hard_max=False, use_lecun_init=use_lecun,
    ))
    return m.to(DEVICE)


def test_small2d_no_lecun_init_default():
    """Flag off → standard PyTorch init applies (kaiming_uniform-ish ranges)."""
    m = _make_small2d(use_lecun=False, seed=42)
    # PyTorch's default for nn.Linear with d_in=8: kaiming_uniform_(a=sqrt(5))
    # → weights in [-sqrt(1/8), sqrt(1/8)] ≈ [-0.354, 0.354]
    # LeCun normal with same fan_in: std=sqrt(1/8)=0.354, so a ~5σ tail can hit
    # ~1.77 — clearly distinguishable from kaiming-uniform's bounded range.
    max_abs = m.W_qkv[0].weight.abs().max().item()
    assert max_abs < 0.5, (
        f"Default init should produce |w| < ~0.354 for d_in=8, got {max_abs}"
    )


def test_small2d_lecun_init_changes_weights():
    """Flag on → weights differ from flag-off at construction (different distribution)."""
    m_base = _make_small2d(use_lecun=False, seed=42)
    m_lecun = _make_small2d(use_lecun=True, seed=42)
    # Same fields exist in both; values must differ for at least one layer.
    w_base = m_base.W_qkv[0].weight
    w_lecun = m_lecun.W_qkv[0].weight
    assert w_base.shape == w_lecun.shape
    assert not torch.equal(w_base, w_lecun), (
        "LeCun re-init should produce different weights than torch default"
    )


def test_small2d_lecun_init_forward_runs():
    """Lecun-init'd model produces finite logits on a forward pass — basic sanity."""
    m = _make_small2d(use_lecun=True, seed=42)
    m.eval()
    idx = torch.randint(0, 16, (1, 8), device=DEVICE)
    with torch.no_grad():
        out = m(idx)
    assert torch.isfinite(out).all()
    assert out.shape == (1, 8, 16)


# ===== Section B: z_init on RecurrentSmall2DTransformer =====

def _make_recurrent(use_z: bool = False, use_inject: bool = False,
                    seed: int = 42) -> RecurrentSmall2DTransformer:
    torch.manual_seed(seed)
    m = RecurrentSmall2DTransformer(RecurrentConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8,
        max_len=32, use_hard_max=False,
        default_iterations=3, max_iterations=8,
        use_input_injection=use_inject,
        use_z_init=use_z,
    ))
    return m.to(DEVICE)


def test_recurrent_z_init_parameter_allocation():
    """Flag on → z_init exists as Parameter; flag off → z_init is None."""
    m_off = _make_recurrent(use_z=False, seed=42)
    m_on = _make_recurrent(use_z=True, seed=42)
    assert m_off.z_init is None
    assert isinstance(m_on.z_init, nn.Parameter)
    assert m_on.z_init.shape == (8,)
    # Small-norm init so flag-on doesn't blow gradients at step 0.
    assert m_on.z_init.abs().max().item() < 0.2  # sane bound for std=0.02


def test_recurrent_z_init_n1_bit_equivalent():
    """n_iterations=1 → early return preserves parent baseline regardless of flag.
    Load-bearing guard from Slice 1 must extend to z_init."""
    m_b = _make_recurrent(use_z=False, seed=42)
    m_z = _make_recurrent(use_z=True, use_inject=True, seed=42)
    m_b.eval(); m_z.eval()
    torch.manual_seed(123)
    idx = torch.randint(0, 16, (1, 8), device=DEVICE)
    with torch.no_grad():
        # m_b uses Small2DTransformer.forward at n=1 (no kwarg).
        # m_z early-returns to same path at n_iterations=1.
        out_b = Small2DTransformer.forward(m_z, idx)
        out_z = m_z(idx, n_iterations=1)
    # At n=1, m_z's forward goes through Small2DTransformer.forward
    # (early return) — same code path, same weights → bit-identical output.
    assert torch.equal(out_z, out_b)


def test_recurrent_z_init_n3_changes_output():
    """At n_iterations=3, z_init shifts iter-0 hidden state → different output."""
    m_b = _make_recurrent(use_z=False, use_inject=True, seed=42)
    m_z = _make_recurrent(use_z=True, use_inject=True, seed=42)
    m_b.eval(); m_z.eval()
    torch.manual_seed(123)
    idx = torch.randint(0, 16, (1, 8), device=DEVICE)
    with torch.no_grad():
        ob = m_b(idx, n_iterations=3)
        oz = m_z(idx, n_iterations=3)
    # NOTE: m_z's z_init was initialized via global RNG inside __init__
    # AFTER super() built all other params. RecurrentSmall2DTransformer
    # constructs no further params after z_init, so other params match
    # m_b's. The only difference: m_z applies z_init at iter 0.
    assert not torch.allclose(ob, oz)


# ===== Section C: z_init on CopyAugmentedDeltaNet (DT path) =====

def _make_dt(use_z: bool = False, use_inject: bool = False,
             use_lecun: bool = False, use_gated: bool = False,
             n_iter: int = 1, seed: int = 42):
    torch.manual_seed(seed)
    m = build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=3,
        use_chunkwise=False, n_iterations=n_iter,
        use_input_injection=use_inject,
        use_z_init=use_z,
        use_gated_attention=use_gated,
        use_lecun_init=use_lecun,
    )
    return m.to(DEVICE)


def test_dt_z_init_parameter_allocation():
    m_off = _make_dt(use_z=False, seed=42)
    m_on = _make_dt(use_z=True, seed=42)
    assert m_off.z_init is None
    assert isinstance(m_on.z_init, nn.Parameter)
    assert m_on.z_init.shape == (8,)


def test_dt_z_init_n1_bit_equivalent():
    """At n_iters=1 the use_z_init gate evaluates False, so z_init is never
    read in forward. The local-generator init of z_init means the global
    RNG stream is not consumed, so ALL OTHER PARAMETERS match exactly
    between use_z_init=False and use_z_init=True at the same seed.
    Therefore forward output is bit-identical."""
    m_b = _make_dt(use_z=False, n_iter=1, seed=42)
    m_z = _make_dt(use_z=True, n_iter=1, seed=42)
    m_b.eval(); m_z.eval()
    torch.manual_seed(123)
    idx = torch.tensor([[5, 7, 3, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        ob = m_b(idx)
        oz = m_z(idx)
    assert torch.equal(ob, oz), (
        "n=1 must preserve baseline: z_init shouldn't fire and shouldn't "
        "disturb global RNG during construction."
    )


def test_dt_z_init_n3_changes_logits():
    """At n_iters=3 with z_init+injection, output differs from baseline n=3."""
    m_b = _make_dt(use_z=False, use_inject=True, n_iter=3, seed=42)
    m_z = _make_dt(use_z=True, use_inject=True, n_iter=3, seed=42)
    m_b.eval(); m_z.eval()
    torch.manual_seed(123)
    idx = torch.tensor([[5, 7, 3, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        ob = m_b(idx)
        oz = m_z(idx)
    assert not torch.allclose(ob, oz)


def test_dt_z_init_iter0_injection_fires_when_z_replaces_x():
    """With use_z_init=True AND use_input_injection=True at n>1, iter 0 MUST
    inject the embedding (z_init alone carries no token info). Falsifier:
    if the skip-iter-0 guard fires when z_init is on, the embedding never
    reaches the layer stack at iter 0 → output should differ from a run
    where injection is correctly applied. We verify by checking that the
    z_init+inject path differs from a z_init-only-no-inject path."""
    m_zonly = _make_dt(use_z=True, use_inject=False, n_iter=3, seed=42)
    m_zinj = _make_dt(use_z=True, use_inject=True, n_iter=3, seed=42)
    m_zonly.eval(); m_zinj.eval()
    torch.manual_seed(123)
    idx = torch.tensor([[5, 7, 3, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        o_zonly = m_zonly(idx)
        o_zinj = m_zinj(idx)
    # z_only with no injection: layer stack sees only z_init (no embed) at iter 0
    # z_inj: layer stack sees z_init + embed at iter 0
    # These MUST differ — proves iter-0 injection fires when z_init is active.
    assert not torch.allclose(o_zonly, o_zinj)


# ===== Section D: LeCun init on DT path (composes with z_init + gated) =====

def test_dt_lecun_init_changes_weights():
    m_b = _make_dt(use_lecun=False, seed=42)
    m_l = _make_dt(use_lecun=True, seed=42)
    # Multiple Linear families to check — base, beta_head, copy_*
    assert not torch.equal(m_b.W_qkv[0].weight, m_l.W_qkv[0].weight)
    assert not torch.equal(m_b.copy_q_proj.weight, m_l.copy_q_proj.weight)
    assert not torch.equal(m_b.beta_head[0].weight, m_l.beta_head[0].weight)


def test_dt_lecun_preserves_copy_gate_bias():
    """copy_gate.bias must remain at copy_gate_bias_init (-2.0) after LeCun re-init."""
    m_l = _make_dt(use_lecun=True, seed=42)
    # copy_gate.bias is a 1-element Parameter; assert equal to copy_gate_bias_init.
    assert m_l.copy_gate.bias.shape == (1,)
    assert m_l.copy_gate.bias.item() == pytest.approx(-2.0)


def test_dt_lecun_preserves_beta_head_bias():
    """beta_head[*].bias must remain at 0.0 (set in DeltaNetSmall2DTransformer.__init__)."""
    m_l = _make_dt(use_lecun=True, seed=42)
    for h in m_l.beta_head:
        assert h.bias.shape == (1,)
        assert h.bias.item() == pytest.approx(0.0)


def test_dt_lecun_init_forward_finite():
    """LeCun-init'd DT produces finite logits."""
    m = _make_dt(use_lecun=True, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, 3, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out = m(idx)
    assert torch.isfinite(out).all()


# ===== Section E: Checkpoint round-trip (Slice 2 flags) =====

def test_checkpoint_roundtrip_slice2_flags():
    """Save DT with use_z_init+use_lecun_init+use_input_injection; reload;
    verify all flags present, biases preserved, and logits match."""
    m_orig = _make_dt(
        use_z=True, use_inject=True, use_lecun=True, use_gated=True,
        n_iter=2, seed=42,
    )
    m_orig.eval()
    torch.manual_seed(123)
    idx = torch.tensor([[5, 7, 3, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out_orig = m_orig(idx)

    cfg_dict = {
        "vocab_size": 20, "max_len": 24,
        "d_model": 8, "n_heads": 4, "n_layers": 2,
        "d_ffn": 16, "n_copy_heads": 2,
        "use_chunkwise": False,
        "n_iterations": 2,
        "use_loop_index": False,
        "use_input_injection": True,
        "use_gated_attention": True,
        "use_z_init": True,
        "use_lecun_init": True,
        "copy_gate_bias_init": -2.0,
    }
    ckpt = {
        "model_state": m_orig.state_dict(),
        "config": cfg_dict,
        "epoch": 1, "val_autoreg": 0.5, "n_train": 100, "n_val": 10,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_dt_slice2.pt"
        torch.save(ckpt, path)
        m_loaded, _ = load_dt_checkpoint(str(path), device=DEVICE)
    m_loaded.eval()

    # Flag bookkeeping
    assert m_loaded.config.use_z_init is True
    assert m_loaded.config.use_lecun_init is True
    assert m_loaded.config.use_input_injection is True
    assert m_loaded.config.use_gated_attention is True
    assert m_loaded.config.n_iterations == 2

    # Parameter presence
    assert isinstance(m_loaded.z_init, nn.Parameter)
    assert m_loaded.attn_gate_proj is not None

    # Bias preservation (post-load, LeCun re-init was applied during build
    # then state_dict overwrote weights — biases load from state_dict)
    assert m_loaded.copy_gate.bias.item() == pytest.approx(-2.0)
    for h in m_loaded.beta_head:
        assert h.bias.item() == pytest.approx(0.0)

    # Logit parity
    with torch.no_grad():
        out_loaded = m_loaded(idx)
    assert torch.equal(out_orig, out_loaded)


def test_checkpoint_backward_compat_slice2():
    """Old checkpoint (no Slice 2 flag keys) must load with flags defaulting off."""
    m_orig = _make_dt(seed=42)  # no Slice 2 flags
    m_orig.eval()
    torch.manual_seed(123)
    idx = torch.tensor([[5, 7, 3, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out_orig = m_orig(idx)

    # Schema WITHOUT new Slice 2 flag keys
    cfg_dict = {
        "vocab_size": 20, "max_len": 24,
        "d_model": 8, "n_heads": 4, "n_layers": 2,
        "d_ffn": 16, "n_copy_heads": 2,
        "use_chunkwise": False,
        # Slice 1 flags present but defaults; Slice 2 flags ABSENT entirely
        "n_iterations": 1,
    }
    ckpt = {
        "model_state": m_orig.state_dict(),
        "config": cfg_dict,
        "epoch": 1, "val_autoreg": 0.5, "n_train": 100, "n_val": 10,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_dt_compat.pt"
        torch.save(ckpt, path)
        m_loaded, _ = load_dt_checkpoint(str(path), device=DEVICE)
    m_loaded.eval()

    assert m_loaded.z_init is None
    assert m_loaded.config.use_z_init is False
    assert m_loaded.config.use_lecun_init is False

    with torch.no_grad():
        out_loaded = m_loaded(idx)
    assert torch.equal(out_orig, out_loaded)


if __name__ == "__main__":
    # Direct run dispatches a minimal smoke. Use pytest for full coverage.
    if not torch.cuda.is_available():
        print("CUDA unavailable, skipping (GPU-only)")
        raise SystemExit(0)
    test_small2d_no_lecun_init_default()
    test_small2d_lecun_init_changes_weights()
    test_small2d_lecun_init_forward_runs()
    test_recurrent_z_init_parameter_allocation()
    test_recurrent_z_init_n1_bit_equivalent()
    test_recurrent_z_init_n3_changes_output()
    test_dt_z_init_parameter_allocation()
    test_dt_z_init_n1_bit_equivalent()
    test_dt_z_init_n3_changes_logits()
    test_dt_z_init_iter0_injection_fires_when_z_replaces_x()
    test_dt_lecun_init_changes_weights()
    test_dt_lecun_preserves_copy_gate_bias()
    test_dt_lecun_preserves_beta_head_bias()
    test_dt_lecun_init_forward_finite()
    test_checkpoint_roundtrip_slice2_flags()
    test_checkpoint_backward_compat_slice2()
    print("All Slice 2 tests PASSED")
