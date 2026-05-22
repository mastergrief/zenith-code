"""HRM-Text-1.58 Phase 1 smoke tests.

Per task #51 + codex msg 1779451257744 Phase 1 implement +1 with 5 guardrails:

1. Forward shape + param count sanity
2. Label-mask test tied to Gsm8kTokenizer (prefix/BOS/SEP=ignore, only >sep trains)
3. PrefixLM mask SEMANTIC assertions on at least 2 rows with different sep_pos
4. bp_steps audit at bp_steps=2 AND bp_steps=5, matching upstream formula
5. Finite fwd/bwd/opt.step on a single GSM8k-shaped batch

All tests run CPU-first (so they pass without a GPU); CUDA-only tests
fall back to skip.
"""
from __future__ import annotations

from typing import List

import math
import pytest
import torch
import torch.nn.functional as F

from calm.hrm_text_158 import (
    HierarchicalReasoningModel,
    HierarchicalReasoningModelConfig,
    LMHead,
    LMHeadConfig,
    build_prefix_lm_mask,
)
from calm.hrm_text_158.lm_head import IGNORE_LABEL_ID


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #

def _tiny_config(**overrides) -> HierarchicalReasoningModelConfig:
    """Minimal valid HRM config for smoke tests. Tier A scale (D1.1)."""
    base = dict(
        max_seq_len=32,
        n_layers=4,       # half_layers=True → 2 H + 2 L
        hidden_size=64,   # NOTE: head_dim = 64/2 = 32, NOT 128 (smoke-tiny)
        num_heads=2,
        expansion=4,
        H_cycles=2,
        L_cycles=3,
        half_layers=True,
        bp_warmup_ratio=0.2,
        bp_min_steps=2,
        bp_max_steps=5,
    )
    base.update(overrides)
    return HierarchicalReasoningModelConfig(**base)


def _make_model(vocab_size: int = 98, seed: int = 42) -> LMHead:
    """Build full LMHead(HRM(...)) with the tiny config."""
    torch.manual_seed(seed)
    hrm = HierarchicalReasoningModel(_tiny_config())
    return LMHead(hrm, LMHeadConfig(vocab_size=vocab_size))


# ----------------------------------------------------------------------------- #
# Guardrail 1: forward shape + param count sanity
# ----------------------------------------------------------------------------- #

def test_forward_shape_smoke() -> None:
    """Forward pass on B=2, S=16 produces logits of shape (B, S, vocab)."""
    m = _make_model(vocab_size=98)
    m.eval()
    B, S = 2, 16
    ids = torch.randint(0, 98, (B, S), dtype=torch.long)
    sep = torch.tensor([4, 6], dtype=torch.long)
    pos = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, -1)
    with torch.no_grad():
        carry, logits = m(None, {"inputs": ids, "sep_positions": sep, "position_ids": pos})
    assert logits.shape == (B, S, 98), f"unexpected logits shape: {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), "logits contain non-finite values"


def test_param_count_tiny_config_under_1m() -> None:
    """Tier A tiny smoke config should land in ~100K-500K param range.
    Reality check that we didn't accidentally instantiate the 1B."""
    m = _make_model(vocab_size=98)
    total = sum(p.numel() for p in m.parameters())
    assert 50_000 < total < 1_500_000, (
        f"tiny-config param count {total:,} out of expected range [50K, 1.5M]. "
        f"Check that hidden_size=64, n_layers=4 (half) honored."
    )


# ----------------------------------------------------------------------------- #
# Guardrail 2: PrefixLM mask SEMANTIC test (>= 2 rows, different sep_pos)
# ----------------------------------------------------------------------------- #

def test_prefix_lm_mask_semantics_multi_row() -> None:
    """Build prefix_lm mask for B=3 with sep at positions {3, 5, 7}.

    Spec (from RESEARCH/HRM-Text-1.58/00_ARCHITECTURE.md):
    - Positions 0..sep (inclusive) are BIDIRECTIONAL prefix
    - Positions > sep attend prefix + own causal history
    - Diagonal always True
    """
    S = 10
    sep_positions = torch.tensor([3, 5, 7], dtype=torch.long)
    mask = build_prefix_lm_mask(S, sep_positions, device=torch.device("cpu"), dtype=torch.bool)
    assert mask.shape == (3, S, S), f"expected (3, {S}, {S}), got {tuple(mask.shape)}"

    # Diagonal: True everywhere
    for b in range(3):
        diag = torch.diag(mask[b])
        assert diag.all(), f"diagonal not all True for batch {b}"

    # Row b=0, sep=3: in-prefix queries (q ∈ {0..3}) attend ONLY prefix keys
    # (k <= sep=3). This is bidirectional within the prefix but does NOT
    # extend visibility into the suffix — those positions are masked out.
    for q in range(0, 4):  # q in prefix
        for k in range(S):
            # In-prefix query attends only prefix keys (k <= sep=3)
            expected = k <= 3
            assert mask[0, q, k].item() == expected, (
                f"b=0 sep=3 q={q} k={k}: expected {expected}, got {mask[0, q, k].item()}. "
                f"In-prefix query should see only prefix keys (k<=sep)."
            )

    # Row b=0, suffix queries q=4..9: should attend prefix keys (0..3) + causal history (k<=q)
    for q in range(4, S):
        for k in range(S):
            in_prefix_key = k <= 3
            causal_key = k <= q
            expected = in_prefix_key or causal_key
            assert mask[0, q, k].item() == expected, (
                f"b=0 sep=3 q={q} k={k}: expected {expected}, got {mask[0, q, k].item()}"
            )

    # Row b=2, sep=7: only positions 8,9 are post-sep
    # Pre-sep queries (0..7) see only prefix (0..7)
    for q in range(0, 8):
        for k in range(S):
            expected = k <= 7
            assert mask[2, q, k].item() == expected, (
                f"b=2 sep=7 q={q} k={k}: expected {expected}, got {mask[2, q, k].item()}"
            )

    # Row b=2, q=9 (suffix): see prefix 0..7 + own history k<=9 → all True
    assert mask[2, 9, :].all(), "b=2 sep=7 q=9 should attend all keys"


# ----------------------------------------------------------------------------- #
# Guardrail 3: bp_steps audit at bp_steps=2 + bp_steps=5
# ----------------------------------------------------------------------------- #

def test_bp_steps_audit_bp_steps_2_and_5() -> None:
    """Verify grad-enabled mask matches upstream
    sapientinc/HRM-Text/models/baselines/hrm_nocarry_bp_warmup.py:78-89.

    For H_cycles=2, L_cycles=3:
    - bp_steps=2: H_bp_steps=min(2, 1)=1, L_bp_steps=1
        L grad iff k >= 6-1=5 → only k=5
        H grad iff i >= 2-1=1 → only i=1 (last H)
    - bp_steps=5: H_bp_steps=min(2, 4)=2, L_bp_steps=3
        L grad iff k >= 6-3=3 → k in {3,4,5}
        H grad iff i >= 2-2=0 → both H (i in {0,1})
    """
    cfg = _tiny_config()
    hrm = HierarchicalReasoningModel(cfg)

    # Instrument: wrap L_level + H_level to capture is_grad_enabled() at each call
    grad_log: List[tuple] = []

    orig_l_fwd = hrm.L_level.forward
    orig_h_fwd = hrm.H_level.forward

    l_call_counter = [0]
    h_call_counter = [0]

    def l_wrapper(hidden_states, input_injection, **kwargs):
        grad_log.append(("L", l_call_counter[0], torch.is_grad_enabled()))
        l_call_counter[0] += 1
        return orig_l_fwd(hidden_states, input_injection, **kwargs)

    def h_wrapper(hidden_states, input_injection, **kwargs):
        grad_log.append(("H", h_call_counter[0], torch.is_grad_enabled()))
        h_call_counter[0] += 1
        return orig_h_fwd(hidden_states, input_injection, **kwargs)

    hrm.L_level.forward = l_wrapper  # type: ignore
    hrm.H_level.forward = h_wrapper  # type: ignore

    B, S = 1, 8
    x = torch.randn(B, S, cfg.hidden_size)
    sep = torch.tensor([3], dtype=torch.long)
    pos = torch.arange(S, dtype=torch.long).unsqueeze(0)

    # bp_steps=2 audit
    grad_log.clear()
    l_call_counter[0] = 0
    h_call_counter[0] = 0
    hrm(None, x, bp_steps=2, sep_positions=sep, position_ids=pos)
    # Expect 6 L calls, 2 H calls in interleaved order: 3 L + 1 H + 3 L + 1 H
    # bp_steps=2: only k=5 has L grad, only i=1 has H grad
    l_grad = [entry[2] for entry in grad_log if entry[0] == "L"]
    h_grad = [entry[2] for entry in grad_log if entry[0] == "H"]
    assert len(l_grad) == 6, f"expected 6 L calls, got {len(l_grad)}"
    assert len(h_grad) == 2, f"expected 2 H calls, got {len(h_grad)}"
    expected_l_bp2 = [False, False, False, False, False, True]   # only k=5
    expected_h_bp2 = [False, True]                                # only i=1
    assert l_grad == expected_l_bp2, f"bp_steps=2 L grad pattern wrong: {l_grad}, expected {expected_l_bp2}"
    assert h_grad == expected_h_bp2, f"bp_steps=2 H grad pattern wrong: {h_grad}, expected {expected_h_bp2}"

    # bp_steps=5 audit
    grad_log.clear()
    l_call_counter[0] = 0
    h_call_counter[0] = 0
    hrm(None, x, bp_steps=5, sep_positions=sep, position_ids=pos)
    l_grad = [entry[2] for entry in grad_log if entry[0] == "L"]
    h_grad = [entry[2] for entry in grad_log if entry[0] == "H"]
    expected_l_bp5 = [False, False, False, True, True, True]   # last 3 (k=3,4,5)
    expected_h_bp5 = [True, True]                              # both (i=0,1)
    assert l_grad == expected_l_bp5, f"bp_steps=5 L grad pattern wrong: {l_grad}, expected {expected_l_bp5}"
    assert h_grad == expected_h_bp5, f"bp_steps=5 H grad pattern wrong: {h_grad}, expected {expected_h_bp5}"


# ----------------------------------------------------------------------------- #
# Guardrail 4: label mask tied to Gsm8kTokenizer
# ----------------------------------------------------------------------------- #

def test_label_mask_tokenizer_tied() -> None:
    """Tokenizer-tied label-mask test per codex msg 1779451812361 gate 2.

    Source-faithful PrefixLM training is left-shifted (upstream
    `sapientinc/HRM-Text/dataset_new.py:102-108`). For our Gsm8kTokenizer
    `encode_example(...)` which produces
        ids = [BOS, q_0, ..., q_{n-1}, SEP, t_0, ..., t_{k-1}, EOS]
        sep_pos = position of SEP
    the LM contract is:
        inputs = ids[:-1]                        # drop EOS, length L-1
        labels[:sep_pos] = IGNORE_LABEL_ID       # prompt-internal predictions ignored
        labels[sep_pos:] = ids[sep_pos+1:]       # SEP predicts t_0, t_i predicts t_{i+1}, t_{k-1} predicts EOS

    Valid count == len(target_digits) + 1 (k target chars + EOS).
    This prevents the model from being trained to "copy itself" within
    the target span (label at sep_pos+i predicts ids[sep_pos+i+1], not
    ids[sep_pos+i]).
    """
    from calm.llm_computer.gsm8k_tokenizer import Gsm8kTokenizer

    # Build a tiny corpus to seed the tokenizer.
    tiny_corpus = [
        {"question": "what is 17 times 23?", "expected": 391},
        {"question": "compute 12 plus 30", "expected": 42},
        {"question": "find 5 minus 1", "expected": 4},
    ]
    tok = Gsm8kTokenizer.from_corpus(tiny_corpus)
    assert tok.vocab_size < 98  # tiny corpus → smaller vocab; ok

    # encode_example for both rows
    ids_a_list, sep_a = tok.encode_example("what is 17 times 23?", 391)
    ids_b_list, sep_b = tok.encode_example("compute 12 plus 30", 42)
    # Pad to common length L
    L = max(len(ids_a_list), len(ids_b_list))
    pad_id = tok.pad_id
    ids_a = ids_a_list + [pad_id] * (L - len(ids_a_list))
    ids_b = ids_b_list + [pad_id] * (L - len(ids_b_list))

    ids_full = torch.tensor([ids_a, ids_b], dtype=torch.long)
    B = 2
    # Source-faithful shift contract
    inputs = ids_full[:, :-1].contiguous()        # (B, L-1)
    labels = torch.full_like(inputs, IGNORE_LABEL_ID)
    for b, sep_pos in enumerate([sep_a, sep_b]):
        labels[b, sep_pos:] = ids_full[b, sep_pos + 1 :]
        # Also ignore padding labels (after EOS, the padding section)
        # Find EOS in the inputs at-or-after sep_pos
        eos_id = tok.eos_id
        # ids_full[b] = [..., sep@sep_pos, t_0..t_{k-1}, eos, pad...]
        # labels[b, sep_pos + k] = eos_id is the LAST valid label
        # labels[b, sep_pos + k + 1 :] = pad (predict-from-eos+0); ignore them
        # Find eos position in input ids
        for i in range(sep_pos + 1, ids_full.shape[1]):
            if ids_full[b, i].item() == eos_id:
                # labels[b, i-1] = eos (last valid); labels[b, i:] = ignore
                if i < labels.shape[1]:
                    labels[b, i:] = IGNORE_LABEL_ID
                break

    # Build position_ids + sep_positions for the input side
    pos = torch.arange(inputs.shape[1], dtype=torch.long).unsqueeze(0).expand(B, -1)
    sep_t = torch.tensor([sep_a, sep_b], dtype=torch.long)

    # Build model using THIS tokenizer's vocab
    torch.manual_seed(42)
    hrm = HierarchicalReasoningModel(_tiny_config(max_seq_len=L))
    m = LMHead(hrm, LMHeadConfig(vocab_size=tok.vocab_size))

    new_carry, loss, metrics = m(
        None,
        {"inputs": inputs, "sep_positions": sep_t, "position_ids": pos, "labels": labels},
    )

    # Expected valid count per row: len(target_chars) + 1 (EOS).
    # For target=391: 3 chars + 1 EOS = 4. For target=42: 2 chars + 1 EOS = 3.
    # Wait — labels[sep_pos] = ids[sep_pos+1] = first target char.
    #        labels[sep_pos+k-1] = ids[sep_pos+k] = last target char.
    #        labels[sep_pos+k] = ids[sep_pos+k+1] = EOS.
    # So valid count = k + 1 (k target chars predicted at positions sep_pos..sep_pos+k-1,
    # plus EOS predicted at position sep_pos+k).
    expected_a = len("391") + 1   # 4
    expected_b = len("42") + 1    # 3
    expected_total = expected_a + expected_b   # 7

    loss_sum, loss_counts = metrics["loss"]
    assert loss_counts.item() == expected_total, (
        f"loss denominator must equal count of valid (non-IGNORE) labels: "
        f"expected {expected_total} = ({expected_a} + {expected_b}), got {loss_counts.item()}"
    )
    accuracy_count, accuracy_total = metrics["accuracy"]
    assert accuracy_total.item() == expected_total, (
        f"accuracy denominator: expected {expected_total}, got {accuracy_total.item()}"
    )
    assert torch.isfinite(loss), f"loss not finite: {loss}"

    # Anti-self-copy check: label at position sep_pos in row 0 (input is SEP)
    # predicts ids_full[0, sep_pos+1] = first target char. NOT the SEP itself.
    assert labels[0, sep_a].item() == ids_full[0, sep_a + 1].item(), (
        f"label at sep_pos must point to first target char (left-shifted), "
        f"not to SEP itself (which would be self-copy)."
    )
    # And label at sep_pos - 1 (last prompt char) is IGNORE
    if sep_a > 0:
        assert labels[0, sep_a - 1].item() == IGNORE_LABEL_ID


# ----------------------------------------------------------------------------- #
# Guardrail 5: finite fwd + bwd + opt.step
# ----------------------------------------------------------------------------- #

def test_full_step_finite() -> None:
    """Full train step on Gsm8k-shaped batch produces finite loss + finite
    grads + finite param updates after opt.step."""
    m = _make_model(vocab_size=98)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

    B, S = 2, 16
    ids = torch.randint(0, 98, (B, S), dtype=torch.long)
    sep = torch.tensor([5, 7], dtype=torch.long)
    pos = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, -1)
    labels = torch.full((B, S), IGNORE_LABEL_ID, dtype=torch.long)
    labels[0, sep[0] + 1 : 11] = torch.randint(1, 98, (5,))
    labels[1, sep[1] + 1 : 13] = torch.randint(1, 98, (5,))

    # Snapshot original params (one tensor)
    orig_param = next(m.parameters()).detach().clone()

    new_carry, loss, metrics = m(
        None,
        {"inputs": ids, "sep_positions": sep, "position_ids": pos, "labels": labels},
        bp_steps=5,
    )
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    opt.zero_grad()
    loss.backward()
    # Check grads finite
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"grad non-finite on {n}"
    grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
    assert torch.isfinite(grad_norm), f"grad_norm not finite: {grad_norm}"
    opt.step()
    # Confirm at least one param actually moved
    new_param = next(m.parameters()).detach()
    assert not torch.allclose(orig_param, new_param), "params did not move after opt.step"
    # And remain finite
    for n, p in m.named_parameters():
        assert torch.isfinite(p).all(), f"param non-finite after step: {n}"


# ----------------------------------------------------------------------------- #
# Bonus: compute_train_extra_args returns linearly-ramped bp_steps
# ----------------------------------------------------------------------------- #

def test_lmhead_delegates_compute_train_extra_args() -> None:
    """Codex msg 1779451812361 gate 1: LMHead must delegate
    `compute_train_extra_args` from the wrapped model so the trainer
    can pull the bp_steps schedule through the wrapper."""
    m = _make_model(vocab_size=98)
    # Method must exist + return the same dict as the inner HRM.
    assert hasattr(m, "compute_train_extra_args")
    out_wrapper = m.compute_train_extra_args(50, 1000)
    out_inner = m.model.compute_train_extra_args(50, 1000)
    assert out_wrapper == out_inner
    assert "bp_steps" in out_wrapper
    # Sanity at step 0 + after warmup
    assert m.compute_train_extra_args(0, 1000)["bp_steps"] == 2
    assert m.compute_train_extra_args(500, 1000)["bp_steps"] == 5


def test_tier_a_real_config_forward() -> None:
    """Codex msg 1779451812361 gate 3: explicit test at the real Tier A
    config (hidden_size=256, num_heads=2, head_dim=128) — the
    D1.1-preserved invariant. Confirms param count is in expected range
    AND forward produces finite logits at B=1, S=8."""
    torch.manual_seed(42)
    tier_a_cfg = HierarchicalReasoningModelConfig(
        max_seq_len=256,
        n_layers=4,        # half_layers=True → 2 per H/L
        hidden_size=256,   # NOT 64 — real Tier A
        num_heads=2,       # head_dim = 256/2 = 128 (preserved per D1.1)
        expansion=4,
        H_cycles=2,
        L_cycles=3,
        half_layers=True,
    )
    hrm = HierarchicalReasoningModel(tier_a_cfg)
    m = LMHead(hrm, LMHeadConfig(vocab_size=98))
    total = sum(p.numel() for p in m.parameters())
    # Expected 3.72M; allow ±5% room
    assert 3_500_000 < total < 4_000_000, (
        f"Tier A param count {total:,} out of expected ~3.72M range. "
        f"Check hidden=256, n_layers=4 (split), heads=2, head_dim=128."
    )
    # head_dim invariant
    block = m.model.H_level.core.layers[0]
    assert block.attn.head_dim == 128, (
        f"D1.1 invariant violated: head_dim={block.attn.head_dim}, expected 128"
    )
    # Tiny forward
    m.eval()
    ids = torch.randint(0, 98, (1, 8), dtype=torch.long)
    sep = torch.tensor([3], dtype=torch.long)
    pos = torch.arange(8, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        carry, logits = m(None, {"inputs": ids, "sep_positions": sep, "position_ids": pos})
    assert logits.shape == (1, 8, 98)
    assert torch.isfinite(logits).all()


def test_bp_steps_warmup_ramp() -> None:
    """compute_train_extra_args ramps bp_steps from bp_min_steps to bp_max_steps
    over bp_warmup_ratio * total_steps. After warmup, stays at bp_max_steps."""
    cfg = _tiny_config(bp_warmup_ratio=0.2, bp_min_steps=2, bp_max_steps=5)
    hrm = HierarchicalReasoningModel(cfg)
    total = 1000
    # Step 0: bp_steps should be bp_min_steps=2
    assert hrm.compute_train_extra_args(0, total)["bp_steps"] == 2
    # Step at end of warmup (200): bp_max_steps
    assert hrm.compute_train_extra_args(200, total)["bp_steps"] == 5
    # Mid-warmup (step 100, half): bp_min + (0.5 * (bp_max - bp_min)) = 2 + 1 = 3 (int trunc)
    assert hrm.compute_train_extra_args(100, total)["bp_steps"] == 3
    # After warmup: still bp_max
    assert hrm.compute_train_extra_args(500, total)["bp_steps"] == 5
    # bp_warmup_ratio=0 → bp_max_steps from step 0
    cfg2 = _tiny_config(bp_warmup_ratio=0.0, bp_min_steps=2, bp_max_steps=5)
    hrm2 = HierarchicalReasoningModel(cfg2)
    assert hrm2.compute_train_extra_args(0, total)["bp_steps"] == 5
