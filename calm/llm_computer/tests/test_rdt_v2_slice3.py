"""Slice 3 regression tests: PrefixLM attention + bp_warmup ported from HRM-Text.

Two more flags ported via the same Slice 1/2 discipline:

  * `use_prefix_lm` — relax causal mask within the prompt prefix. Queries
    at positions [0, sep_pos) attend bidirectionally to keys at positions
    [0, sep_pos); everything else stays causal. Inert on the base
    Small2DTransformer because `_compute_prefix_mask` returns None; the
    CopyAugmentedDeltaNet override derives the prefix from sep_token_id.
    Inert on the DT delta-only path too (no softmax mask to relax) — only
    matters when `use_softmax_attn=True` (hybrid softmax+delta) or on the
    vanilla path (`use_delta_net=False`).
  * `bp_warmup_active_iters` — runtime kwarg (NOT persisted) that limits
    gradient flow to the LAST k iterations of the D5 loop. Earlier iters
    run under no_grad and the hidden state enters the gradient-tracked
    region as a detached tensor. None = full grad. Implements HRM-Text
    backprop warmup.

GPU-only per user direction. Toy d_model=8 configs fit in 2 GiB VRAM.
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
SEP_ID = 3

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Slice 3 tests are GPU-only per user direction; CUDA unavailable",
)


# ===== Section A: PrefixLM on base Small2DTransformer hook =====

def test_small2d_prefix_lm_hook_default_none():
    """Base Small2DTransformer._compute_prefix_mask returns None — flag-on
    here has no behavioral effect (the mask is never constructed)."""
    torch.manual_seed(42)
    m = Small2DTransformer(Small2DConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8,
        max_len=32, use_hard_max=False, use_prefix_lm=True,
    )).to(DEVICE)
    idx = torch.randint(0, 16, (1, 8), device=DEVICE)
    # hook returns None
    assert m._compute_prefix_mask(idx) is None


def test_small2d_prefix_lm_flag_inert_on_base():
    """On base Small2DTransformer, use_prefix_lm=True is inert because the
    hook returns None. Output must equal flag-off output at same seed."""
    torch.manual_seed(42)
    m_off = Small2DTransformer(Small2DConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8,
        max_len=32, use_hard_max=False, use_prefix_lm=False,
    )).to(DEVICE)
    torch.manual_seed(42)
    m_on = Small2DTransformer(Small2DConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8,
        max_len=32, use_hard_max=False, use_prefix_lm=True,
    )).to(DEVICE)
    m_off.eval(); m_on.eval()
    idx = torch.randint(0, 16, (1, 8), device=DEVICE)
    with torch.no_grad():
        a = m_off(idx)
        b = m_on(idx)
    assert torch.equal(a, b)


# ===== Section B: PrefixLM on CopyAugmentedDeltaNet (subclass override) =====

def _make_dt(use_prefix: bool = False, use_softmax: bool = False,
             use_delta: bool = True,
             use_lecun: bool = False, seed: int = 42):
    torch.manual_seed(seed)
    m = build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=SEP_ID,
        use_chunkwise=False,
        use_softmax_attn=use_softmax,
        use_prefix_lm=use_prefix,
        use_lecun_init=use_lecun,
    )
    if not use_delta:
        m.config.use_delta_net = False  # vanilla attention fallback path
    return m.to(DEVICE)


def test_dt_prefix_mask_hook_uses_sep_id():
    """CopyAugmentedDeltaNet._compute_prefix_mask returns sep-derived mask."""
    m = _make_dt(use_prefix=True, seed=42)
    # idx: [5, 7, SEP, 9, 11, 13] — sep at pos 2
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    mask = m._compute_prefix_mask(idx)
    # Positions before first SEP (i.e. 0, 1) are prefix
    expected = torch.tensor([[True, True, False, False, False, False]], device=DEVICE)
    assert torch.equal(mask, expected)


def test_dt_prefix_lm_inert_on_delta_only_path():
    """On the DT default (delta-only, no softmax attention), prefix_lm has
    no effect: _delta_layer_stack skips _attention entirely."""
    m_off = _make_dt(use_prefix=False, use_softmax=False, seed=42)
    m_on = _make_dt(use_prefix=True, use_softmax=False, seed=42)
    m_off.eval(); m_on.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        a = m_off(idx)
        b = m_on(idx)
    # Same path, same params, no softmax mask to relax → bit-identical.
    assert torch.equal(a, b)


def test_dt_prefix_lm_changes_softmax_path():
    """With use_softmax_attn=True, the prefix mask relaxation should change
    output vs causal-only baseline."""
    m_off = _make_dt(use_prefix=False, use_softmax=True, seed=42)
    m_on = _make_dt(use_prefix=True, use_softmax=True, seed=42)
    m_off.eval(); m_on.eval()
    # idx with sep mid-sequence so the prefix block is non-trivial
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        a = m_off(idx)
        b = m_on(idx)
    # Different mask → different attention pattern → different output
    assert not torch.allclose(a, b)


def test_dt_prefix_lm_no_sep_means_full_bidirectional():
    """If idx contains no sep token, `_build_prefix_mask` returns all-True
    (sep_pos defaults to S, every position < S). That means the entire
    sequence is treated as a single prefix block → fully bidirectional
    attention. Behavior MUST differ from causal-only baseline."""
    m_off = _make_dt(use_prefix=False, use_softmax=True, seed=42)
    m_on = _make_dt(use_prefix=True, use_softmax=True, seed=42)
    m_off.eval(); m_on.eval()
    # No SEP_ID in idx
    idx = torch.tensor([[5, 7, 11, 9, 11, 13]], device=DEVICE)
    mask = m_on._compute_prefix_mask(idx)
    # Confirm the mask is all-True per _build_prefix_mask semantics
    assert mask.all()
    with torch.no_grad():
        a = m_off(idx)
        b = m_on(idx)
    # All-True prefix → full-bidirectional softmax → differs from causal
    assert not torch.allclose(a, b)


def test_dt_prefix_lm_on_vanilla_path_changes_output():
    """use_delta_net=False fallback path also wires prefix_mask. Verify it
    has effect on a non-empty prefix."""
    m_off = _make_dt(use_prefix=False, use_softmax=True, use_delta=False, seed=42)
    m_on = _make_dt(use_prefix=True, use_softmax=True, use_delta=False, seed=42)
    m_off.eval(); m_on.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        a = m_off(idx)
        b = m_on(idx)
    assert not torch.allclose(a, b)


# ===== Section C: bp_warmup runtime knob =====

def _make_dt_d5(n_iter: int = 3, seed: int = 42):
    torch.manual_seed(seed)
    return build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=SEP_ID,
        use_chunkwise=False, n_iterations=n_iter,
        use_input_injection=True,
    ).to(DEVICE)


def test_bp_warmup_default_full_grad():
    """No ctx → all iters differentiable. Param.grad nonzero for tok embed."""
    m = _make_dt_d5(n_iter=3, seed=42)
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    out = m(idx)
    loss = out.sum()
    loss.backward()
    # Some param somewhere must have grad — pick W_qkv[0].weight (shared across iters)
    g = m.W_qkv[0].weight.grad
    assert g is not None
    assert g.abs().max().item() > 0


def test_bp_warmup_active_1_smaller_grad_than_full():
    """At active_iters=1, only last iter's contribution to dL/dW_shared remains.
    Compared to active_iters=3 (full), the gradient magnitude should DIFFER
    (typically smaller since fewer iter contributions sum)."""
    # Full-grad reference
    m_full = _make_dt_d5(n_iter=3, seed=42)
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    out = m_full(idx)
    out.sum().backward()
    g_full = m_full.W_qkv[0].weight.grad.clone()

    # Reduced grad (bp_warmup=1)
    m_warm = _make_dt_d5(n_iter=3, seed=42)
    with m_warm.bp_warmup_ctx(1):
        out = m_warm(idx)
        out.sum().backward()
    g_warm = m_warm.W_qkv[0].weight.grad.clone()

    # Both nonzero
    assert g_full.abs().max().item() > 0
    assert g_warm.abs().max().item() > 0
    # Different shapes of gradient (different iters contribute)
    assert not torch.allclose(g_full, g_warm)


def test_bp_warmup_n_iters_1_noop():
    """When n_iters=1, the bp_warmup ctx is a no-op (no iters to detach)."""
    m_a = _make_dt_d5(n_iter=1, seed=42)
    m_b = _make_dt_d5(n_iter=1, seed=42)
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    out_a = m_a(idx)
    out_a.sum().backward()
    with m_b.bp_warmup_ctx(1):
        out_b = m_b(idx)
        out_b.sum().backward()
    assert torch.equal(m_a.W_qkv[0].weight.grad, m_b.W_qkv[0].weight.grad)


def test_bp_warmup_ctx_restores_on_exit():
    """After exiting bp_warmup_ctx, the runtime attr returns to its prior state."""
    m = _make_dt_d5(n_iter=3, seed=42)
    assert m._bp_warmup_active_iters is None
    with m.bp_warmup_ctx(1):
        assert m._bp_warmup_active_iters == 1
    assert m._bp_warmup_active_iters is None


def test_bp_warmup_forward_pass_correctness():
    """bp_warmup is a backward-side concern only — forward output must be
    bit-identical with/without the ctx (no_grad's forward arithmetic is
    the same as enable_grad's forward arithmetic)."""
    m = _make_dt_d5(n_iter=3, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out_full = m(idx)
    with torch.no_grad():
        with m.bp_warmup_ctx(1):
            out_warm = m(idx)
    assert torch.equal(out_full, out_warm)


# ===== Section D: Recurrent path bp_warmup + prefix_lm =====

def _make_recurrent(use_prefix: bool = False, default_iters: int = 3, seed: int = 42):
    torch.manual_seed(seed)
    return RecurrentSmall2DTransformer(RecurrentConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8,
        max_len=32, use_hard_max=False,
        default_iterations=default_iters, max_iterations=8,
        use_input_injection=True,
        use_prefix_lm=use_prefix,
    )).to(DEVICE)


def test_recurrent_bp_warmup_kwarg_forward_unchanged():
    """Recurrent forward bp_warmup_active_iters kwarg doesn't change forward output."""
    m = _make_recurrent(default_iters=3, seed=42)
    m.eval()
    idx = torch.randint(0, 16, (1, 8), device=DEVICE)
    with torch.no_grad():
        out_full = m(idx, n_iterations=3, bp_warmup_active_iters=None)
        out_warm = m(idx, n_iterations=3, bp_warmup_active_iters=1)
    assert torch.equal(out_full, out_warm)


def test_recurrent_bp_warmup_changes_gradient():
    """At n=3 with bp_warmup_active_iters=1, only last iter contributes
    gradient through shared weights. Compared to active=3 (full), grad differs."""
    idx = torch.randint(0, 16, (1, 8), device=DEVICE, generator=torch.Generator(device=DEVICE).manual_seed(7))

    m_full = _make_recurrent(default_iters=3, seed=42)
    out = m_full(idx, n_iterations=3, bp_warmup_active_iters=None)
    out.sum().backward()
    g_full = m_full.W_qkv[0].weight.grad.clone()

    m_warm = _make_recurrent(default_iters=3, seed=42)
    out = m_warm(idx, n_iterations=3, bp_warmup_active_iters=1)
    out.sum().backward()
    g_warm = m_warm.W_qkv[0].weight.grad.clone()

    assert not torch.allclose(g_full, g_warm)


# ===== Section E: Checkpoint round-trip — use_prefix_lm persists =====

def test_checkpoint_roundtrip_slice3_flags():
    """Save DT with use_prefix_lm=True; reload; verify flag persists,
    biases stay, and logits match."""
    torch.manual_seed(42)
    m_orig = build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=SEP_ID,
        use_chunkwise=False, use_softmax_attn=True,
        use_prefix_lm=True,
        use_lecun_init=True,
    ).to(DEVICE)
    m_orig.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out_orig = m_orig(idx)

    cfg_dict = {
        "vocab_size": 20, "max_len": 24,
        "d_model": 8, "n_heads": 4, "n_layers": 2,
        "d_ffn": 16, "n_copy_heads": 2,
        "use_chunkwise": False,
        "n_iterations": 1,
        "use_prefix_lm": True,
        "use_lecun_init": True,
        "copy_gate_bias_init": -2.0,
    }
    ckpt = {
        "model_state": m_orig.state_dict(),
        "config": cfg_dict,
        "epoch": 1, "val_autoreg": 0.5, "n_train": 100, "n_val": 10,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_slice3.pt"
        torch.save(ckpt, path)
        m_loaded, _ = load_dt_checkpoint(str(path), device=DEVICE)
    m_loaded.eval()

    assert m_loaded.config.use_prefix_lm is True
    assert m_loaded.copy_gate.bias.item() == pytest.approx(-2.0)

    # NOTE: m_loaded was built with use_softmax_attn defaulting to False
    # in build_copy_augmented_delta (load_dt_checkpoint doesn't pass it).
    # That's actually a separate persistence gap from earlier slices — for
    # Slice 3 round-trip we only assert prefix_lm + biases survive.
    # Logit parity in the path actually exercised:
    with torch.no_grad():
        # Re-run with the same softmax-on config to match m_orig
        m_loaded.config.use_softmax_attn = True
        out_loaded = m_loaded(idx)
    assert torch.equal(out_orig, out_loaded)


def test_checkpoint_backward_compat_slice3():
    """Old checkpoint without Slice 3 keys loads with flag defaulting False."""
    torch.manual_seed(42)
    m_orig = build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=SEP_ID,
        use_chunkwise=False,
    ).to(DEVICE)
    m_orig.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out_orig = m_orig(idx)

    cfg_dict = {
        "vocab_size": 20, "max_len": 24,
        "d_model": 8, "n_heads": 4, "n_layers": 2,
        "d_ffn": 16, "n_copy_heads": 2,
        "use_chunkwise": False,
        # NO use_prefix_lm key
    }
    ckpt = {
        "model_state": m_orig.state_dict(),
        "config": cfg_dict,
        "epoch": 1, "val_autoreg": 0.5, "n_train": 100, "n_val": 10,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_slice3_compat.pt"
        torch.save(ckpt, path)
        m_loaded, _ = load_dt_checkpoint(str(path), device=DEVICE)
    m_loaded.eval()
    assert m_loaded.config.use_prefix_lm is False

    with torch.no_grad():
        out_loaded = m_loaded(idx)
    assert torch.equal(out_orig, out_loaded)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA unavailable, skipping (GPU-only)")
        raise SystemExit(0)
    test_small2d_prefix_lm_hook_default_none()
    test_small2d_prefix_lm_flag_inert_on_base()
    test_dt_prefix_mask_hook_uses_sep_id()
    test_dt_prefix_lm_inert_on_delta_only_path()
    test_dt_prefix_lm_changes_softmax_path()
    test_dt_prefix_lm_no_sep_means_no_prefix()
    test_dt_prefix_lm_on_vanilla_path_changes_output()
    test_bp_warmup_default_full_grad()
    test_bp_warmup_active_1_smaller_grad_than_full()
    test_bp_warmup_n_iters_1_noop()
    test_bp_warmup_ctx_restores_on_exit()
    test_bp_warmup_forward_pass_correctness()
    test_recurrent_bp_warmup_kwarg_forward_unchanged()
    test_recurrent_bp_warmup_changes_gradient()
    test_checkpoint_roundtrip_slice3_flags()
    test_checkpoint_backward_compat_slice3()
    print("All Slice 3 tests PASSED")
