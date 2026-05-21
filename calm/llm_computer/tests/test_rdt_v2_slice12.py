"""Slice 12 regression tests: per-layer Pre-RMSNorm flag (L-stack stability).

Adds `DeltaNetConfig.use_pre_rmsnorm: bool = False` (default off → bit-
equivalent to Slice 1-11). When on, allocates two RMSNorm modules per
layer in L bank (and H bank when `use_h_layer_stack=True`); applied
inside `_delta_layer_stack` BEFORE the sequence-mixer (QKV projection)
AND BEFORE the FFN sublayer — full pre-norm block pattern (HRM-Text /
Llama / GPT-NeoX canonical).

Fixes S2 NaN root cause: without per-layer norm, the L-stack residual
accumulates `x = x + W_out(...) + ff_out(...)` per layer; at the Core-
H/L flag bundle (`use_input_injection + n_iter=2 + h_cycles=2 +
h_layer_stack`), magnitudes saturate fp32 at `ff_out.3` (~1.76e35) per
the `/tmp/diagnose_s2_nan.py` receipt `1779353960822-1bade0d5`.

Per co_lead audit `1779354358961-1aff6d0c`:
- A1: `use_pre_rmsnorm` must be on the cached-decode blocklist
  (`copy_augmented_delta.py:333-376`) — flat prefill path cannot honor
  the pre-norm; training-time `forward` would silently diverge from
  product-path `decode_greedy_cached` otherwise.
- A2: NaN falsifier uses a deterministic GSM8k-like fixture with real
  vocab/SEP contract (`Gsm8kTokenizer.from_corpus` + `encode_example`),
  NOT unconstrained `torch.randint`. If the falsifier no longer NaNs
  under the fixture, stop and report rather than weakening to vibes-
  only finite tests.

GPU-only.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from calm.llm_computer.copy_augmented_delta import (
    build_copy_augmented_delta,
)
from calm.llm_computer.gsm8k_tokenizer import Gsm8kTokenizer


DEVICE = "cuda"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Slice 12 tests are GPU-only per user direction; CUDA unavailable",
)


# ===== Fixtures =====

_SYNTH_CORPUS = [
    # Mini synthetic corpus to seed the Gsm8kTokenizer with real-shape
    # vocab (digits, letters, punctuation, $, %, etc.). Mirrors the
    # distribution we'd hit on actual GSM8k. Tokenizer built from this
    # gives a real vocab/SEP contract without an HF / datasets dependency.
    {"question": "Alice has 3 apples. Bob gives her 7 more. How many?",
     "expected": "10"},
    {"question": "John buys 5 books at $12 each. Total cost?",
     "expected": "60"},
    {"question": "A train travels 60 mph for 2.5 hours. Distance?",
     "expected": "150"},
    {"question": "What is 17 times 23?", "expected": "391"},
    {"question": "Sara saves $25 weekly. After 8 weeks, how much?",
     "expected": "200"},
    {"question": "If x + 4 = 12, what is x?", "expected": "8"},
    {"question": "A box contains 144 items. 1/3 sold. Remaining?",
     "expected": "96"},
]


def _make_tokenizer() -> Gsm8kTokenizer:
    """Build a Gsm8kTokenizer from the synthetic corpus above."""
    return Gsm8kTokenizer.from_corpus(_SYNTH_CORPUS)


def _make_failed_shape_ids(tok: Gsm8kTokenizer, target_length: int = 160,
                           seed: int = 42) -> torch.Tensor:
    """Deterministic GSM8k-like ids tensor at the EXACT length and
    shape that triggered the S2 NaN (~160 tokens, real SEP contract).
    Uses repeated encoding of synthetic rows to reach target length.
    Per co_lead audit `1779354358961-1aff6d0c` amendment A2 — falsifier
    must use real vocab/SEP, not unconstrained torch.randint.
    """
    torch.manual_seed(seed)
    # Encode the canonical 17*23 row (matches the project's smoke test).
    base_ids, _sep_pos = tok.encode_example("What is 17 times 23?", "391")
    # If shorter than target, repeat-pad with the same row's tail so the
    # SEP contract appears at exactly one position (matches the trainer
    # contract: BOS + question + SEP + answer + EOS).
    ids = list(base_ids)
    if len(ids) < target_length:
        # Pad with the EOS token to reach target length deterministically.
        ids.extend([tok.eos_id] * (target_length - len(ids)))
    ids = ids[:target_length]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)


def _make_core_hl_model(use_pre_rmsnorm: bool, seed: int = 42, *,
                        vocab_size: int) -> nn.Module:
    """Build a CopyAugmentedDeltaNet at the EXACT S2 Core-H/L config
    that triggered the NaN, varying only `use_pre_rmsnorm`. d_model=64,
    n_layers=4, full flag bundle on (matches scripts/train_dt_gsm8k.py
    locked S2 config per resume-pack `1779315087054-19a47385`)."""
    torch.manual_seed(seed)
    return build_copy_augmented_delta(
        vocab_size=vocab_size, d_model=64, n_heads=32, n_layers=4,
        d_ffn=128, max_len=512, n_copy_heads=4,
        sep_token_id=3,
        use_chunkwise=True,
        n_iterations=2,
        h_cycles=2,
        use_loop_index=True,
        use_input_injection=True,
        use_gated_attention=True,
        use_z_init=True,
        use_lecun_init=True,
        use_short_conv=True,
        use_h_rmsnorm=True,
        use_h_layer_stack=True,
        use_pre_rmsnorm=use_pre_rmsnorm,
    ).to(DEVICE)


# ===== Section A: Flag plumbing + allocation =====

def test_slice12_flag_default_off_no_alloc():
    """Default `use_pre_rmsnorm=False` → no pre_mixer_norm/pre_ffn_norm
    modules allocated. Slice 1-11 bit-equivalence baseline."""
    torch.manual_seed(42)
    m = build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=3,
        use_chunkwise=False,
    ).to(DEVICE)
    assert m.pre_mixer_norm is None
    assert m.pre_ffn_norm is None
    assert m.H_pre_mixer_norm is None
    assert m.H_pre_ffn_norm is None


def test_slice12_flag_on_allocates_l_bank_norms():
    """Flag on → per-layer RMSNorm modules in L bank (count = n_layers
    for each of pre_mixer / pre_ffn). H bank stays None when
    `use_h_layer_stack=False`."""
    torch.manual_seed(42)
    n_layers = 4
    m = build_copy_augmented_delta(
        vocab_size=20, d_model=64, n_heads=32, n_layers=n_layers,
        d_ffn=128, max_len=24, n_copy_heads=2, sep_token_id=3,
        use_chunkwise=False,
        use_pre_rmsnorm=True,
    ).to(DEVICE)
    assert m.pre_mixer_norm is not None
    assert m.pre_ffn_norm is not None
    assert len(m.pre_mixer_norm) == n_layers
    assert len(m.pre_ffn_norm) == n_layers
    for norm in m.pre_mixer_norm:
        assert isinstance(norm, nn.RMSNorm)
    for norm in m.pre_ffn_norm:
        assert isinstance(norm, nn.RMSNorm)
    # H bank not allocated when h_layer_stack=False
    assert m.H_pre_mixer_norm is None
    assert m.H_pre_ffn_norm is None


def test_slice12_h_bank_symmetric_allocation():
    """Flag on + `use_h_layer_stack=True` → BOTH L and H banks have
    norms allocated. Symmetric per co_lead `1779353989271`."""
    torch.manual_seed(42)
    n_layers = 4
    m = build_copy_augmented_delta(
        vocab_size=20, d_model=64, n_heads=32, n_layers=n_layers,
        d_ffn=128, max_len=24, n_copy_heads=2, sep_token_id=3,
        use_chunkwise=False,
        h_cycles=2,  # required for h_layer_stack to be meaningful
        use_h_layer_stack=True,
        use_pre_rmsnorm=True,
    ).to(DEVICE)
    assert m.pre_mixer_norm is not None
    assert m.pre_ffn_norm is not None
    assert m.H_pre_mixer_norm is not None
    assert m.H_pre_ffn_norm is not None
    assert len(m.H_pre_mixer_norm) == n_layers
    assert len(m.H_pre_ffn_norm) == n_layers
    for norm in m.H_pre_mixer_norm:
        assert isinstance(norm, nn.RMSNorm)


# ===== Section B: NaN falsifier (Core-H/L composition) =====

def test_slice12_falsifier_flag_off_nans_under_real_fixture():
    """A2 (amendment): deterministic GSM8k-like fixture with real
    vocab/SEP contract. Flag OFF on the EXACT Core-H/L config that
    triggered the S2 NaN — assert non-finite output to PROVE this
    test would catch a regression of the Slice 12 fix.

    If this assertion fails (i.e. flag-off no longer NaNs under the
    deterministic fixture), STOP and report rather than weakening the
    test into a vibes-only finite check. Per co_lead `1779354358961`.
    """
    tok = _make_tokenizer()
    ids = _make_failed_shape_ids(tok, target_length=160, seed=42)

    m = _make_core_hl_model(use_pre_rmsnorm=False, seed=42,
                            vocab_size=tok.vocab_size)
    m.eval()
    with torch.no_grad():
        final_log_probs, per_iter_list = m(ids, return_per_iter=True)

    has_nan_final = bool(torch.isnan(final_log_probs).any().item())
    has_nan_per_iter = any(
        bool(torch.isnan(lp).any().item()) for lp in per_iter_list
    )
    assert has_nan_final or has_nan_per_iter, (
        "Slice 12 FALSIFIER FAILED: flag-off Core-H/L config should NaN "
        "on the deterministic GSM8k-like fixture (160 tokens, real SEP "
        "contract). If you see this, the underlying NaN mode has been "
        "fixed by something else OR the fixture no longer hits the "
        "failure regime — STOP and report rather than weakening this "
        "assertion. Falsifier per co_lead audit `1779354358961-1aff6d0c`."
    )


def test_slice12_pre_rmsnorm_fixes_core_hl_nan():
    """A2 happy path: SAME fixture + SAME Core-H/L config + flag ON
    → final + ALL per-iter outputs finite. This is the actual fix
    landing — must pass."""
    tok = _make_tokenizer()
    ids = _make_failed_shape_ids(tok, target_length=160, seed=42)

    m = _make_core_hl_model(use_pre_rmsnorm=True, seed=42,
                            vocab_size=tok.vocab_size)
    m.eval()
    with torch.no_grad():
        final_log_probs, per_iter_list = m(ids, return_per_iter=True)

    assert torch.isfinite(final_log_probs).all(), (
        f"final_log_probs has non-finite values; max abs = "
        f"{float(final_log_probs.abs().max()):.3e}"
    )
    for i, lp in enumerate(per_iter_list):
        assert torch.isfinite(lp).all(), (
            f"per_iter[{i}] has non-finite values; max abs = "
            f"{float(lp.abs().max()):.3e}"
        )


# ===== Section C: Cached-decode blocklist (amendment A1) =====

def test_slice12_cached_decode_blocked_with_flag_on():
    """A1: `decode_greedy_cached` must refuse a model built with
    `use_pre_rmsnorm=True`. Cached prefill walks flat layers and
    cannot honor the pre-norm path — silent divergence between
    training-time `forward` and product-path `decode_greedy_cached`
    is the failure mode we're blocking.

    Per co_lead audit `1779354358961-1aff6d0c` amendment A1.
    """
    torch.manual_seed(42)
    m = build_copy_augmented_delta(
        vocab_size=20, d_model=64, n_heads=32, n_layers=2,
        d_ffn=128, max_len=24, n_copy_heads=2, sep_token_id=3,
        use_chunkwise=False,
        use_pre_rmsnorm=True,
    ).to(DEVICE)
    # SEP at position 5 so prefix_ids has the right contract.
    prefix_ids = torch.tensor([[1, 2, 4, 5, 6, 3, 7, 8]],
                              dtype=torch.long, device=DEVICE)
    with pytest.raises(NotImplementedError) as exc_info:
        m.decode_greedy_cached(prefix_ids, max_gen=4, eos_token=0)
    msg = str(exc_info.value)
    assert "use_pre_rmsnorm" in msg, (
        f"NotImplementedError must mention `use_pre_rmsnorm`; got: {msg}"
    )


def test_slice12_cached_decode_runs_with_flag_off():
    """A1 sanity: cached decode still works when `use_pre_rmsnorm=False`
    (only this flag is the gate; other flags must remain unblocked at
    their default values)."""
    torch.manual_seed(42)
    m = build_copy_augmented_delta(
        vocab_size=20, d_model=64, n_heads=32, n_layers=2,
        d_ffn=128, max_len=24, n_copy_heads=2, sep_token_id=3,
        use_chunkwise=False,
        # Explicitly off:
        use_pre_rmsnorm=False,
    ).to(DEVICE)
    prefix_ids = torch.tensor([[1, 2, 4, 5, 6, 3, 7, 8]],
                              dtype=torch.long, device=DEVICE)
    m.eval()
    out_ids = m.decode_greedy_cached(prefix_ids, max_gen=4, eos_token=0)
    # Sanity: returns a tensor of generated ids
    assert isinstance(out_ids, torch.Tensor)
    assert out_ids.numel() >= 0  # may early-EOS; just must not raise


# ===== Section D: Checkpoint round-trip =====

def test_slice12_checkpoint_roundtrip_flag_on():
    """Save state_dict from a flag-on model, build a fresh flag-on model
    with the same config, load_state_dict(strict=True) — verify all
    RMSNorm weights round-trip and forward output matches bit-exactly."""
    torch.manual_seed(42)
    common_kwargs = dict(
        vocab_size=20, d_model=64, n_heads=32, n_layers=2,
        d_ffn=128, max_len=24, n_copy_heads=2, sep_token_id=3,
        use_chunkwise=False,
        use_pre_rmsnorm=True,
    )
    m1 = build_copy_augmented_delta(**common_kwargs).to(DEVICE)
    # Stash forward output before save
    ids = torch.tensor([[1, 2, 4, 5, 6, 3, 7, 8]],
                       dtype=torch.long, device=DEVICE)
    m1.eval()
    with torch.no_grad():
        out1 = m1(ids)
    # Save / load
    buf = io.BytesIO()
    torch.save(m1.state_dict(), buf)
    buf.seek(0)
    state = torch.load(buf, weights_only=True)
    m2 = build_copy_augmented_delta(**common_kwargs).to(DEVICE)
    m2.load_state_dict(state, strict=True)
    m2.eval()
    with torch.no_grad():
        out2 = m2(ids)
    assert torch.equal(out1, out2), (
        "Round-trip forward should be bit-exact after load_state_dict; "
        "RMSNorm weights may not have persisted correctly."
    )


# ===== Section E: RNG-isolation falsifier (Slice 8 / 10a pattern) =====

def test_slice12_rng_isolation_downstream_seed_stable():
    """Flag-on allocation must save/restore global RNG state so
    downstream subclass-built params (CopyAugmentedDeltaNet's copy_gate,
    copy_q_proj, copy_k_proj built AFTER super().__init__() returns) get
    the SAME seeded values whether the flag is on or off. Same pattern
    as Slice 2 z_init, Slice 8 H bank, Slice 10a halt_head.

    At the bit-equivalent base config (h_cycles=1, n_iter=1,
    h_layer_stack=False — all RDT-v2 flags off EXCEPT use_pre_rmsnorm),
    copy_q_proj weights must MATCH between flag-on and flag-off builds
    at the same seed."""
    common_kwargs = dict(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=3,
        use_chunkwise=False,
    )
    seed = 7

    torch.manual_seed(seed)
    m_off = build_copy_augmented_delta(use_pre_rmsnorm=False,
                                       **common_kwargs).to(DEVICE)

    torch.manual_seed(seed)
    m_on = build_copy_augmented_delta(use_pre_rmsnorm=True,
                                      **common_kwargs).to(DEVICE)

    # copy_q_proj / copy_k_proj / copy_gate are built AFTER the
    # RMSNorm allocation. If RNG isn't isolated, their weights would
    # silently diverge between flag-on and flag-off at the same seed.
    assert torch.equal(m_off.copy_q_proj.weight, m_on.copy_q_proj.weight)
    assert torch.equal(m_off.copy_k_proj.weight, m_on.copy_k_proj.weight)
    assert torch.equal(m_off.copy_gate.weight, m_on.copy_gate.weight)
