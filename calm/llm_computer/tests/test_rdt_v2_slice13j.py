"""Slice 13j inertness regression test (documentation/protection only).

Per codex audit chain msg 1779398485126 -> 1779432805671 -> 1779432871832:
The `--use-prefix-lm` flag is wired through trainer + config + metadata
but is functionally INERT on the active DeltaNet/chunkwise code path.
Evidence: `delta_rule.py:1027-1034` computes `prefix_mask`,
`:1265-1286` only forwards it into `_attention` when
`cfg.use_softmax_attn=True`. With `use_softmax_attn=False` (DeltaNet
canonical), the mask has nowhere to go.

These tests prevent future Claude from re-litigating the inertness
and ensure that any future real 13j implementation must FIRST update
these tests (the parity assertion would break if `use_prefix_lm`
actually changed behavior, which is the gate for real-impl status).

When a real 13j PrefixLM-for-DeltaNet implementation lands:
- The "delta-only, prefix_lm off-vs-on bit-identical" assertion MUST
  fail (max_abs_logit_diff > 0 at prompt positions)
- This file's tests should be updated to assert the OPPOSITE
  property — that with delta-only + prefix-lm-impl, prompt-position
  logits actually change.
"""
from __future__ import annotations

import torch

from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta


def _make(use_prefix_lm: bool, use_softmax_attn: bool = False,
          seed: int = 42):
    """Build two-config-identical-except-flag model under TRM-1.58 first
    config (chunkwise DeltaNet)."""
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
        use_prefix_lm=use_prefix_lm,
        use_softmax_attn=use_softmax_attn,
    )


def _sep_batch():
    """Batch with sep at position 6 (5 prompt tokens + sep + 2 response tokens).

    Shape: (B=1, S=9).
    """
    return torch.tensor([[1, 5, 6, 7, 8, 9, 3, 10, 11]], dtype=torch.long)


def test_use_prefix_lm_state_dict_identical_under_delta_only():
    """With chunkwise DeltaNet (use_softmax_attn=False), toggling
    use_prefix_lm produces bit-identical state_dict — no extra params
    are allocated for the prefix-LM mechanism since it has no consumer."""
    m_off = _make(use_prefix_lm=False)
    m_on = _make(use_prefix_lm=True)
    sd_off = m_off.state_dict()
    sd_on = m_on.state_dict()
    assert set(sd_off.keys()) == set(sd_on.keys()), (
        f"state_dict key mismatch: off={set(sd_off.keys())-set(sd_on.keys())}, "
        f"on={set(sd_on.keys())-set(sd_off.keys())}"
    )
    for k in sd_off:
        assert torch.equal(sd_off[k], sd_on[k]), (
            f"param {k} differs between use_prefix_lm off/on; "
            f"on chunkwise DeltaNet path no params should differ."
        )


def test_use_prefix_lm_logits_identical_under_delta_only():
    """Forward pass with toggled `use_prefix_lm` must produce
    bit-identical logits across ALL token positions (prompt AND response)
    under chunkwise DeltaNet. Any non-zero diff is evidence the flag
    has been wired into the active code path — which is what the real
    13j slice will eventually do; this test then must be updated to
    assert the OPPOSITE.
    """
    m_off = _make(use_prefix_lm=False).eval()
    m_on = _make(use_prefix_lm=True).eval()
    ids = _sep_batch()
    with torch.no_grad():
        out_off = m_off(ids)
        out_on = m_on(ids)
    assert out_off.shape == out_on.shape, (
        f"logits shape mismatch: off={tuple(out_off.shape)} "
        f"on={tuple(out_on.shape)}"
    )
    diff = (out_off - out_on).abs().max().item()
    assert diff == 0.0, (
        f"use_prefix_lm changed logits under chunkwise DeltaNet "
        f"(max_abs_diff={diff:.3e}); if intentional, this test must be "
        f"updated to assert non-zero diff at prompt positions."
    )


def test_use_prefix_lm_prompt_positions_identical_under_delta_only():
    """Prompt-position-specific assertion: the positions ≤ sep (the
    bidirectionally-visible portion in a real PrefixLM implementation)
    must produce IDENTICAL logits under the inert delta-only path.
    Separate assertion from the all-positions test so future debugging
    pinpoints which positions diverged when real-impl lands."""
    m_off = _make(use_prefix_lm=False).eval()
    m_on = _make(use_prefix_lm=True).eval()
    ids = _sep_batch()
    sep_pos = 6
    with torch.no_grad():
        out_off = m_off(ids)
        out_on = m_on(ids)
    prompt_diff = (out_off[:, :sep_pos + 1, :] - out_on[:, :sep_pos + 1, :]
                   ).abs().max().item()
    response_diff = (out_off[:, sep_pos + 1:, :] - out_on[:, sep_pos + 1:, :]
                     ).abs().max().item()
    assert prompt_diff == 0.0, (
        f"prompt-position logit diff under delta-only: {prompt_diff:.3e}; "
        f"use_prefix_lm must be inert here until real 13j lands."
    )
    assert response_diff == 0.0, (
        f"response-position logit diff under delta-only: {response_diff:.3e}; "
        f"use_prefix_lm must be inert here until real 13j lands."
    )
