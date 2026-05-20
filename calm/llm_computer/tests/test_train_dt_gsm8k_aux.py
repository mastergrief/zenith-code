"""S0c-aux gates per codex audit `1779314708107`.

Three required assertions:
1. `_masked_shifted_nll` applies the target mask, NOT prompt tokens.
2. aux_weight=0 is bit-equivalent to the helper's final-only result.
3. aux_weight>0 changes the loss tensor (and gradient) vs final-only in
   an h_cycles>1 setup where per-iter entries provably differ.
"""
from __future__ import annotations

import torch

from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta
from scripts.train_dt_gsm8k import _masked_shifted_nll


SEP_ID = 3


def _tiny_model(**flag_overrides):
    """Builds a tiny CPU DT with chunkwise off (slice11 tests run without
    chunkwise; matches the per-iter divergence path).
    """
    kw = dict(
        vocab_size=82, d_model=64, n_heads=32, n_layers=2,
        d_ffn=64, max_len=32, n_copy_heads=4,
        sep_token_id=SEP_ID, use_chunkwise=False,
    )
    kw.update(flag_overrides)
    return build_copy_augmented_delta(**kw)


def _toy_batch(B: int = 2, L: int = 8, sep_at: int = 4):
    """Two sequences of length L with <sep> at position `sep_at`.

    Mask is True for positions [sep_at .. L-2] inclusive (the target-side
    positions whose NEXT-token prediction the loss should score).
    """
    torch.manual_seed(0)
    ids = torch.randint(low=5, high=80, size=(B, L), dtype=torch.long)
    ids[:, sep_at] = SEP_ID
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, sep_at:L - 1] = True
    return ids, mask


def test_masked_shifted_nll_applies_target_mask_not_prompt():
    """If we corrupt log-probs at PROMPT positions, the helper's output
    must not change — proves mask zeros prompt-side contribution.
    """
    torch.manual_seed(1)
    m = _tiny_model(h_cycles=1)
    m.eval()
    ids, mask = _toy_batch(sep_at=4)

    with torch.no_grad():
        lp_clean = m(ids).clone()
    nll_clean = _masked_shifted_nll(lp_clean, ids, mask)

    # Corrupt prompt-side log-probs (positions 0 .. sep_at-1 inclusive).
    # These positions predict tokens at indices 1 .. sep_at — all of which
    # the mask zeroes out.
    lp_corrupt = lp_clean.clone()
    sep_at = 4
    lp_corrupt[:, :sep_at] = torch.randn_like(lp_corrupt[:, :sep_at]) * 100
    nll_corrupt = _masked_shifted_nll(lp_corrupt, ids, mask)

    assert torch.allclose(nll_clean, nll_corrupt, atol=1e-6), (
        f"helper leaked prompt-position loss: clean={nll_clean.item():.6f} "
        f"corrupt={nll_corrupt.item():.6f}"
    )


def test_aux_weight_zero_is_bit_equivalent_to_final_only():
    """The S0c-aux path must preserve baseline numerics at aux_weight=0.
    Compute `_masked_shifted_nll(m(ids), ids, mask)` directly and assert
    it equals `final_NLL + 0 * aux_NLL` — i.e., the train loop branch on
    `aux_weight > 0.0` is the only switch.
    """
    torch.manual_seed(2)
    m = _tiny_model(h_cycles=2, n_iterations=2, use_input_injection=True)
    m.eval()
    ids, mask = _toy_batch(sep_at=4)

    with torch.no_grad():
        baseline = _masked_shifted_nll(m(ids), ids, mask)
        final_lp, per_iter = m(ids, return_per_iter=True)
        final_nll = _masked_shifted_nll(final_lp, ids, mask)
        # aux_weight=0 contribution should vanish even if per-iter NLLs
        # are nonzero — equivalent to skipping the aux path entirely.
        aux_nll = sum(_masked_shifted_nll(lp, ids, mask)
                      for lp in per_iter) / len(per_iter)
        loss_aux_zero = final_nll + 0.0 * aux_nll

    # Final equals baseline (same forward path, m(ids) and m(ids)[0] both
    # land at _compute_log_probs(final_x, idx)).
    assert torch.allclose(baseline, final_nll, atol=1e-6)
    assert torch.allclose(loss_aux_zero, baseline, atol=1e-6), (
        f"aux=0 path drifted from baseline: aux0={loss_aux_zero.item():.6f} "
        f"baseline={baseline.item():.6f}"
    )


def test_aux_weight_positive_changes_loss_and_gradient_at_h_cycles_gt_1():
    """At h_cycles>1 + use_input_injection, per-iter entries differ from
    `final` (slice11 test :147-159 proves this). So aux>0 must shift
    both the loss scalar AND the gradient on a shared weight (W_qkv[0])
    vs the final-only path.
    """
    torch.manual_seed(3)
    ids, mask = _toy_batch(sep_at=4)

    # Final-only path.
    m_final = _tiny_model(h_cycles=2, n_iterations=2, use_input_injection=True)
    loss_final = _masked_shifted_nll(m_final(ids), ids, mask)
    loss_final.backward()
    g_final = m_final.W_qkv[0].weight.grad.clone()

    # Aux>0 path: same seed for fair grad comparison.
    torch.manual_seed(3)
    m_aux = _tiny_model(h_cycles=2, n_iterations=2, use_input_injection=True)
    final_lp_a, per_iter_a = m_aux(ids, return_per_iter=True)
    final_nll_a = _masked_shifted_nll(final_lp_a, ids, mask)
    aux_nll_a = sum(_masked_shifted_nll(lp, ids, mask)
                    for lp in per_iter_a) / len(per_iter_a)
    loss_aux = final_nll_a + 0.5 * aux_nll_a
    loss_aux.backward()
    g_aux = m_aux.W_qkv[0].weight.grad.clone()

    # Loss scalar differs.
    assert not torch.allclose(loss_final, loss_aux), (
        f"aux>0 did not change loss: final={loss_final.item():.6f} "
        f"aux={loss_aux.item():.6f}"
    )
    # Gradient on shared weight differs.
    assert torch.isfinite(g_final).all() and torch.isfinite(g_aux).all()
    assert not torch.allclose(g_final, g_aux), (
        "aux>0 did not change gradient on W_qkv[0]; per-iter loss is not "
        "propagating through the backbone"
    )
