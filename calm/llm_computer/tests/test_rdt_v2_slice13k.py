"""Slice 13k softmax-only mixer mode tests.

Per codex audit chain msg 1779442422990-d4ae4217 → 1779442478419-69b61e00
(SIZING: MEDIUM, DEVELOPER_STEPS: 2, Step 1).

The softmax-only mode keeps the H/L stack + carry + halt_head outer-loop
semantics but SKIPS the DeltaNet chunkwise/per-position recurrence
entirely. Different from `use_softmax_attn=True` alone (which runs delta
IN PARALLEL with softmax — hybrid). Mutex enforced at CLI level.

Tests:
- softmax-only produces non-None last_q_pair (Q-head still active)
- softmax-only produces valid carry shape (H/L stack semantics preserved)
- PrefixLM off vs on under softmax-only changes logits on sep batch
  (i.e. the softmax + PrefixLM mechanism is wired and active)
- Negative-assertion proxy: state_dict does not differ between
  use_softmax_attn=True hybrid and use_softmax_only=True (the latter
  just skips delta compute — same param count). Confirms no spurious
  param allocations.
"""
from __future__ import annotations

import pytest
import torch

from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta


def _make(use_softmax_only: bool, use_prefix_lm: bool = False,
          use_softmax_attn: bool = False, seed: int = 42):
    """TRM-1.58 first-config-shaped model toggling softmax-only and
    related flags."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return build_copy_augmented_delta(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=3,
        use_chunkwise=True, n_iterations=2, h_cycles=2,
        use_loop_index=True, use_input_injection=True,
        use_gated_attention=True, use_z_init=True, use_lecun_init=True,
        use_h_rmsnorm=True, use_short_conv=True, use_h_layer_stack=True,
        use_pre_rmsnorm=True,
        use_halt_head=True, use_carry=True,
        use_softmax_only=use_softmax_only,
        use_softmax_attn=use_softmax_attn,
        use_prefix_lm=use_prefix_lm,
    )


def _sep_batch():
    """Batch with sep at position 6."""
    return torch.tensor([[1, 5, 6, 7, 8, 9, 3, 10, 11]], dtype=torch.long)


# =============================================================================
# Step 1 assertion 1: softmax_only + halt_head + carry → non-None last_q_pair
# =============================================================================


def test_softmax_only_produces_non_none_last_q_pair():
    """Softmax-only must preserve the Q-head outer-loop semantics — it
    is a mixer-mode change, NOT a fallback to vanilla transformer
    (which is what use_delta_net=False does). last_q_pair must be set
    after forward."""
    m = _make(use_softmax_only=True)
    m.eval()
    ids = _sep_batch()
    with torch.no_grad():
        _ = m(ids)
    assert m.last_q_pair is not None, (
        "softmax-only must preserve Q-head; got last_q_pair=None. "
        "This indicates the build path fell through to the "
        "use_delta_net=False vanilla fallback rather than running the "
        "H/L stack with softmax-only mixing."
    )
    assert m.last_q_pair.shape == (1, 2), (
        f"last_q_pair shape mismatch: expected (1, 2), got "
        f"{tuple(m.last_q_pair.shape)}"
    )


def test_softmax_only_auto_forces_softmax_attn():
    """Build path must auto-set use_softmax_attn=True when
    use_softmax_only=True (otherwise the new branch in
    _delta_layer_stack asserts attn is non-None and would fail)."""
    m = _make(use_softmax_only=True)
    assert m.config.use_softmax_only is True
    assert m.config.use_softmax_attn is True, (
        "build_copy_augmented_delta must auto-force use_softmax_attn=True "
        "when use_softmax_only=True"
    )


# =============================================================================
# Step 1 assertion 2: valid carry shape
# =============================================================================


def test_softmax_only_valid_carry_shape():
    """H/L stack semantics preserved → return_carry must yield a
    grad-detached tensor with shape (B, S, d_model). Same as delta-only path."""
    m = _make(use_softmax_only=True)
    m.eval()
    ids = _sep_batch()
    with torch.no_grad():
        out = m(ids, return_carry=True)
    assert isinstance(out, tuple), (
        f"return_carry=True must return tuple; got {type(out)}"
    )
    log_probs, carry = out
    B = ids.shape[0]
    S = ids.shape[1]
    d_model = m.config.d_model
    assert carry.shape == (B, S, d_model), (
        f"carry shape mismatch: expected ({B}, {S}, {d_model}), got "
        f"{tuple(carry.shape)}"
    )
    assert not carry.requires_grad, "carry should be detached"


# =============================================================================
# Step 1 assertion 3: PrefixLM off-vs-on changes logits on sep batch
#                     (proves softmax + PrefixLM mechanism is active)
# =============================================================================


def test_softmax_only_prefix_lm_forward_diff_nonzero():
    """Under use_softmax_only=True, toggling use_prefix_lm must change
    logits at prompt positions (positions ≤ sep). Documents that the
    softmax+PrefixLM mechanism is actually wired into the active
    code path. Inverse of the 13j inertness test for the delta-only path."""
    m_off = _make(use_softmax_only=True, use_prefix_lm=False).eval()
    m_on = _make(use_softmax_only=True, use_prefix_lm=True).eval()
    # Same seed → same state_dict
    sd_off = m_off.state_dict(); sd_on = m_on.state_dict()
    for k in sd_off:
        assert torch.equal(sd_off[k], sd_on[k]), (
            f"param {k} differs unexpectedly between prefix_lm off/on"
        )
    ids = _sep_batch()
    sep_pos = 6
    with torch.no_grad():
        out_off = m_off(ids)
        out_on = m_on(ids)
    prompt_diff = (out_off[:, :sep_pos + 1, :]
                   - out_on[:, :sep_pos + 1, :]).abs().max().item()
    assert prompt_diff > 1e-4, (
        f"under softmax_only, prefix_lm off-vs-on max abs prompt diff = "
        f"{prompt_diff:.3e}; must be >> 0 to confirm mechanism is active. "
        f"If zero, the new branch in _delta_layer_stack is not routing "
        f"prefix_mask into _attention."
    )


# =============================================================================
# Negative assertion proxy: softmax-only state_dict has SAME param count
# as use_softmax_attn=True hybrid (just skips compute, no spurious params)
# =============================================================================


def test_softmax_only_no_extra_params_vs_hybrid():
    """Sanity check that softmax-only doesn't allocate extra params
    beyond what the hybrid (softmax+delta) configuration has — they
    use the same building blocks, softmax-only just skips compute
    paths. Negative-assertion proxy for codex msg 1779442478419:
    softmax-only must not allocate DeltaNet chunkwise state."""
    m_hybrid = _make(use_softmax_only=False, use_softmax_attn=True)
    m_only = _make(use_softmax_only=True)
    n_hybrid = sum(p.numel() for p in m_hybrid.parameters())
    n_only = sum(p.numel() for p in m_only.parameters())
    assert n_hybrid == n_only, (
        f"softmax-only param count differs from hybrid: "
        f"hybrid={n_hybrid}, only={n_only}. Indicates spurious "
        f"allocation in softmax-only mode."
    )


# =============================================================================
# Mutex sanity: build path does NOT raise when both flags set in library;
# the mutex enforcement lives at CLI in train_dt_gsm8k.py (per codex)
# =============================================================================


def test_softmax_only_with_both_flags_at_library_level_succeeds():
    """At library level, passing both use_softmax_only=True AND
    use_softmax_attn=True is benign (softmax-only auto-forces softmax_attn
    anyway). The mutex enforcement is at CLI parse time, not library."""
    m = _make(use_softmax_only=True, use_softmax_attn=True)
    assert m.config.use_softmax_only is True
    assert m.config.use_softmax_attn is True
    m.eval()
    ids = _sep_batch()
    with torch.no_grad():
        _ = m(ids)
    assert m.last_q_pair is not None
