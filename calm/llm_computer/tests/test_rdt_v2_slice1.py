"""Slice 1 regression tests: HRM-Text input injection + gated attention ported to DT.

Per codex_co_lead audit msgs 1779297967921 + 1779298145753, the load-bearing
seams for RDT are:
  1. `Small2DTransformer` (non-DT cards): use_gated_attention flag on _attention
  2. `RecurrentSmall2DTransformer` (D5 cards): use_input_injection on forward
  3. `DeltaNetSmall2DTransformer._forward_backbone`: use_input_injection on D5 loop
  4. `DeltaNetSmall2DTransformer._delta_layer_stack`: use_gated_attention on delta_out
  5. `CopyAugmentedDeltaNet.decode_greedy_cached`: gate parity for cached decode
  6. `build_copy_augmented_delta` + `load_dt_checkpoint`: flag persistence round-trip

All flags default off; bit-equivalence preserved unless flag explicitly set.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from calm.llm_computer.copy_augmented_delta import (
    CopyAugmentedDeltaConfig,
    CopyAugmentedDeltaNet,
    build_copy_augmented_delta,
)
from calm.llm_computer.dt_install import load_dt_checkpoint
from calm.llm_computer.model import Small2DConfig, Small2DTransformer
from calm.llm_computer.recurrent_substrate import RecurrentConfig, RecurrentSmall2DTransformer


# ===== Section A: Small2DTransformer (non-DT) flag plumbing =====

def _make_small2d(use_gated: bool = False, seed: int = 42) -> Small2DTransformer:
    torch.manual_seed(seed)
    return Small2DTransformer(Small2DConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8,
        max_len=32, use_hard_max=False, use_gated_attention=use_gated,
    ))


def test_small2d_no_gate_deterministic():
    m = _make_small2d(use_gated=False, seed=42)
    m.eval()
    torch.manual_seed(123)
    idx = torch.randint(0, 16, (1, 8))
    with torch.no_grad():
        out1 = m(idx)
        out2 = m(idx)
    assert torch.equal(out1, out2)
    assert m.attn_gate_proj is None


def test_small2d_gated_changes_output():
    m_baseline = _make_small2d(use_gated=False, seed=42)
    m_gated = _make_small2d(use_gated=True, seed=42)
    m_baseline.eval(); m_gated.eval()
    torch.manual_seed(123)
    idx = torch.randint(0, 16, (1, 8))
    with torch.no_grad():
        ob = m_baseline(idx)
        og = m_gated(idx)
    assert m_gated.attn_gate_proj is not None
    assert len(m_gated.attn_gate_proj) == 2
    assert not torch.allclose(ob, og)


def test_small2d_gated_extra_params():
    m_b = _make_small2d(use_gated=False, seed=42)
    m_g = _make_small2d(use_gated=True, seed=42)
    diff = m_g.param_count() - m_b.param_count()
    assert diff == 2 * (8 * 8)  # n_layers=2, d_model^2 per gate proj


# ===== Section B: RecurrentSmall2DTransformer (non-DT D5) flag plumbing =====

def _make_recurrent(use_inject: bool = False, seed: int = 42) -> RecurrentSmall2DTransformer:
    torch.manual_seed(seed)
    return RecurrentSmall2DTransformer(RecurrentConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8,
        max_len=32, use_hard_max=False,
        default_iterations=3, max_iterations=8,
        use_input_injection=use_inject,
    ))


def test_recurrent_n1_bit_equivalent():
    """n_iterations=1 must match parent regardless of injection flag."""
    m_inject = _make_recurrent(use_inject=True, seed=42)
    m_inject.eval()
    torch.manual_seed(123)
    idx = torch.randint(0, 16, (1, 8))
    with torch.no_grad():
        out_n1 = m_inject(idx, n_iterations=1)
        out_parent = Small2DTransformer.forward(m_inject, idx)
    assert torch.equal(out_n1, out_parent)


def test_recurrent_inject_changes_n3():
    m_b = _make_recurrent(use_inject=False, seed=42)
    m_i = _make_recurrent(use_inject=True, seed=42)
    m_b.eval(); m_i.eval()
    torch.manual_seed(123)
    idx = torch.randint(0, 16, (1, 8))
    with torch.no_grad():
        ob = m_b(idx, n_iterations=3)
        oi = m_i(idx, n_iterations=3)
    assert not torch.allclose(ob, oi)


# ===== Section C: DT-path (CopyAugmentedDeltaNet) flag plumbing =====

def _make_dt(use_inject: bool = False, use_gated: bool = False, n_iter: int = 1, seed: int = 42) -> CopyAugmentedDeltaNet:
    torch.manual_seed(seed)
    return build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=3,
        use_chunkwise=False, n_iterations=n_iter,
        use_input_injection=use_inject,
        use_gated_attention=use_gated,
    )


def test_dt_flags_off_deterministic():
    """All flags off → forward is deterministic + has no gate params."""
    m = _make_dt(seed=42)
    m.eval()
    torch.manual_seed(123)
    # Need at least one sep token for copy mechanism to work
    idx = torch.tensor([[5, 7, 3, 9, 11, 13]])  # sep at pos 2
    with torch.no_grad():
        out1 = m(idx)
        out2 = m(idx)
    assert torch.equal(out1, out2)
    assert m.attn_gate_proj is None


def test_dt_gated_changes_logits():
    """use_gated_attention=True on DT path applies gate to delta_out → different logits."""
    m_b = _make_dt(use_gated=False, seed=42)
    m_g = _make_dt(use_gated=True, seed=42)
    m_b.eval(); m_g.eval()
    torch.manual_seed(123)
    idx = torch.tensor([[5, 7, 3, 9, 11, 13]])
    with torch.no_grad():
        ob = m_b(idx)
        og = m_g(idx)
    assert m_g.attn_gate_proj is not None
    assert not torch.allclose(ob, og)


def test_dt_input_injection_n1_bit_equivalent():
    """use_input_injection=True at n_iterations=1 must match baseline exactly.

    Per co_lead audit msg 1779298145753: 'use_input_injection=True, n_iterations=1
    should match baseline exactly if you skip iteration 0.' This protects the
    cached-decode guard shape — single-iteration cards are always safe.
    """
    m_b = _make_dt(use_inject=False, n_iter=1, seed=42)
    m_i = _make_dt(use_inject=True, n_iter=1, seed=42)
    m_b.eval(); m_i.eval()
    torch.manual_seed(123)
    idx = torch.tensor([[5, 7, 3, 9, 11, 13]])
    with torch.no_grad():
        ob = m_b(idx)
        oi = m_i(idx)
    # When n_iter=1, the iteration-0 skip means injection never fires
    assert torch.equal(ob, oi)


def test_dt_input_injection_n3_changes_logits():
    """At n_iterations>1, use_input_injection=True must produce different logits."""
    m_b = _make_dt(use_inject=False, n_iter=3, seed=42)
    m_i = _make_dt(use_inject=True, n_iter=3, seed=42)
    m_b.eval(); m_i.eval()
    torch.manual_seed(123)
    idx = torch.tensor([[5, 7, 3, 9, 11, 13]])
    with torch.no_grad():
        ob = m_b(idx)
        oi = m_i(idx)
    assert not torch.allclose(ob, oi)


def test_dt_cached_decode_parity_no_flags():
    """Without flags, decode_greedy_cached's first emitted token must match the
    full-forward argmax at the last prefix position. Regression guard against
    any cached-decode refactor breaking the existing baseline."""
    m = _make_dt(seed=42)
    m.eval()
    torch.manual_seed(123)
    prefix = torch.tensor([[5, 7, 3, 9, 11]])  # short prompt with sep
    with torch.no_grad():
        # Cached decode emits its first token via _predict_next_token at last prefix pos
        gen = m.decode_greedy_cached(prefix, max_gen=1, eos_token=None)
        # Compare to the manual single-step path:
        # Full forward → log_probs at last position → argmax
        log_probs = m(prefix)  # (B, S, V)
        manual_argmax = log_probs[0, -1].argmax().item()
    # decode_greedy_cached's first emitted token should match
    assert gen.shape[1] == 1
    assert int(gen[0, 0].item()) == manual_argmax


def test_dt_cached_decode_parity_gated():
    """With use_gated_attention=True, cached decode and full forward must
    still agree on first emitted token. This is the load-bearing parity
    co_lead specified — without it, training-time forward and product-path
    cached decode silently diverge.
    """
    m = _make_dt(use_gated=True, seed=42)
    m.eval()
    torch.manual_seed(123)
    prefix = torch.tensor([[5, 7, 3, 9, 11]])
    with torch.no_grad():
        gen = m.decode_greedy_cached(prefix, max_gen=1, eos_token=None)
        log_probs = m(prefix)
        manual_argmax = log_probs[0, -1].argmax().item()
    assert int(gen[0, 0].item()) == manual_argmax


# ===== Section D: Checkpoint round-trip persistence =====

def test_checkpoint_roundtrip_flags():
    """Save a DT with use_gated_attention=True; load via load_dt_checkpoint;
    verify the reloaded model has the same architecture AND produces same logits.

    Falsifier (per co_lead): if build_copy_augmented_delta doesn't accept the
    new flags, or load_dt_checkpoint doesn't read them from cfg, the reloaded
    model would silently lose the gate parameters and produce different output.
    """
    m_orig = _make_dt(use_gated=True, use_inject=True, n_iter=2, seed=42)
    m_orig.eval()

    torch.manual_seed(123)
    idx = torch.tensor([[5, 7, 3, 9, 11, 13]])
    with torch.no_grad():
        out_orig = m_orig(idx)

    # Save in the schema the training scripts use
    cfg_dict = {
        "vocab_size": 20, "max_len": 24,
        "d_model": 8, "n_heads": 4, "n_layers": 2,
        "d_ffn": 16, "n_copy_heads": 2,
        "use_chunkwise": False,
        "n_iterations": 2,
        "use_loop_index": False,
        "use_input_injection": True,
        "use_gated_attention": True,
        "copy_gate_bias_init": -2.0,
    }
    ckpt = {
        "model_state": m_orig.state_dict(),
        "config": cfg_dict,
        "epoch": 1, "val_autoreg": 0.5, "n_train": 100, "n_val": 10,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_dt.pt"
        torch.save(ckpt, path)
        m_loaded, _ = load_dt_checkpoint(str(path), device="cpu")
    m_loaded.eval()

    # Same gate params present
    assert m_loaded.attn_gate_proj is not None
    assert len(m_loaded.attn_gate_proj) == 2

    # Same config flags
    assert m_loaded.config.use_gated_attention is True
    assert m_loaded.config.use_input_injection is True
    assert m_loaded.config.n_iterations == 2

    # Same logits
    with torch.no_grad():
        out_loaded = m_loaded(idx)
    assert torch.equal(out_orig, out_loaded), "reloaded model logits diverged from original"


def test_checkpoint_roundtrip_backward_compat():
    """Old checkpoint (no new flags in config) must still load — flags default off."""
    m_orig = _make_dt(seed=42)  # no new flags
    m_orig.eval()
    torch.manual_seed(123)
    idx = torch.tensor([[5, 7, 3, 9, 11, 13]])
    with torch.no_grad():
        out_orig = m_orig(idx)

    # Schema without new flags — simulating an old checkpoint
    cfg_dict = {
        "vocab_size": 20, "max_len": 24,
        "d_model": 8, "n_heads": 4, "n_layers": 2,
        "d_ffn": 16, "n_copy_heads": 2,
        "use_chunkwise": False,
        # NO new flags
    }
    ckpt = {
        "model_state": m_orig.state_dict(),
        "config": cfg_dict,
        "epoch": 1, "val_autoreg": 0.5, "n_train": 100, "n_val": 10,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_dt.pt"
        torch.save(ckpt, path)
        m_loaded, _ = load_dt_checkpoint(str(path), device="cpu")
    m_loaded.eval()
    assert m_loaded.attn_gate_proj is None
    assert m_loaded.config.use_input_injection is False
    assert m_loaded.config.n_iterations == 1

    with torch.no_grad():
        out_loaded = m_loaded(idx)
    assert torch.equal(out_orig, out_loaded)


if __name__ == "__main__":
    test_small2d_no_gate_deterministic()
    test_small2d_gated_changes_output()
    test_small2d_gated_extra_params()
    test_recurrent_n1_bit_equivalent()
    test_recurrent_inject_changes_n3()
    test_dt_flags_off_deterministic()
    test_dt_gated_changes_logits()
    test_dt_input_injection_n1_bit_equivalent()
    test_dt_input_injection_n3_changes_logits()
    test_dt_cached_decode_parity_no_flags()
    test_dt_cached_decode_parity_gated()
    test_checkpoint_roundtrip_flags()
    test_checkpoint_roundtrip_backward_compat()
    print("All Slice 1 tests PASSED")
