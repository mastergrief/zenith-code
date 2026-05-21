"""Slice 13h source-faithful HRM-Text ACT-greedy inference decoder.

Tests for `scripts/train_dt_gsm8k.autoreg_decode_integer_hrm` and the
companion `autoreg_eval_hrm`. Per codex audit chain
1779391729056 -> 1779391827108 -> 1779391870772 -> 1779391876372:

- Halt-at-seg1 path equals old single-forward decoder when the model's
  Q_halt > Q_continue every segment (validates "old behavior is a
  special case" of the new behavior).
- Forced-min-segments path runs multiple segments + records correct
  segment count.
- `act_inference=True` continues to raise until properly delegated
  through the new outer-loop helper (deferred per Slice 13f.2).
- Outer-loop emission matches per-segment final-position logits
  (no spurious per-segment leakage).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta
from scripts.train_dt_gsm8k import (
    autoreg_decode_integer,
    autoreg_decode_integer_hrm,
)


class _StubTokenizer:
    """Minimal tokenizer matching the interface used by the decoders."""

    def __init__(self, vocab_size: int = 16, sep_id: int = 3,
                 bos_id: int = 1, eos_id: int = 2):
        self.vocab_size = vocab_size
        self.sep_id = sep_id
        self.bos_id = bos_id
        self.eos_id = eos_id

    def encode(self, text: str) -> list[int]:
        # Treat each char as id (offset 4 to avoid specials).
        return [4 + (ord(c) % (self.vocab_size - 4)) for c in text]

    def decode(self, ids: list[int], stop_at_eos: bool = True) -> str:
        out = []
        for i in ids:
            if stop_at_eos and i == self.eos_id:
                break
            out.append(chr(i + 32))
        return "".join(out)


def _make_model(*, use_carry=True, use_halt_head=True,
                h_cycles=2, n_iter=2,
                d_model=8, n_heads=4, n_layers=2, vocab_size=16,
                max_len=24, d_ffn=16, seed=42):
    torch.manual_seed(seed)
    return build_copy_augmented_delta(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        n_copy_heads=2, sep_token_id=3,
        use_chunkwise=True,
        n_iterations=n_iter,
        h_cycles=h_cycles,
        use_loop_index=True, use_input_injection=True,
        use_gated_attention=True, use_z_init=True, use_lecun_init=True,
        use_h_rmsnorm=True, use_short_conv=True, use_h_layer_stack=True,
        use_pre_rmsnorm=True,
        use_halt_head=use_halt_head,
        use_carry=use_carry,
    )


def _force_halt_at_seg1(model):
    """Patch halt_head so Q_halt = 1 and Q_continue = 0 for every input,
    making the HRM decoder ALWAYS halt at segment 1 (matches old decoder)."""
    with torch.no_grad():
        # halt_head: Linear(d_model, 2) followed by sigmoid in last_q_pair.
        # Saturate Q_halt high (+10 -> sigmoid ~1.0) and Q_continue low (-10
        # -> sigmoid ~0.0) for any input by zeroing weights and biasing.
        model.halt_head.weight.zero_()
        model.halt_head.bias[0] = 10.0   # Q_halt
        model.halt_head.bias[1] = -10.0  # Q_continue


def _force_continue_to_max(model):
    """Patch halt_head so the head ALWAYS prefers continue; halt only fires
    via the forced seg==m_max guard."""
    with torch.no_grad():
        model.halt_head.weight.zero_()
        model.halt_head.bias[0] = -10.0
        model.halt_head.bias[1] = 10.0


# ============================================================================
# Test (a): halt-at-seg1 path bit-equivalent to old decoder
# ============================================================================


def test_halt_at_seg1_bit_equivalent_to_old_decoder():
    """When Q_halt > Q_continue at seg 1, HRM decoder must emit the same
    tokens as the legacy single-forward decoder. Bit-equivalence is the
    'old behavior is a special case of new behavior' guarantee."""
    m = _make_model()
    _force_halt_at_seg1(m)
    tok = _StubTokenizer(vocab_size=16, sep_id=3, bos_id=1, eos_id=2)
    question = "abc"
    device = "cpu"

    old_str = autoreg_decode_integer(m, tok, question, max_new=8, device=device)
    new = autoreg_decode_integer_hrm(
        m, tok, question, m_max=4, max_new=8,
        eval_min_segments=1, device=device,
    )
    assert new["decoded"] == old_str, (
        f"HRM halt-at-seg1 must equal legacy decoder: "
        f"old={old_str!r}  new={new['decoded']!r}"
    )
    # All tokens emitted at segment 1
    assert all(s == 1 for s in new["segs_per_token"]), (
        f"expected every token emitted at seg 1; got {new['segs_per_token']}"
    )
    # Halt histogram concentrated at index 0
    assert new["halt_histogram"][0] == len(new["segs_per_token"])
    assert all(h == 0 for h in new["halt_histogram"][1:])


# ============================================================================
# Test (b): forced-min-segments path runs multiple segments
# ============================================================================


def test_forced_min_segments_runs_multiple_segments():
    """With eval_min_segments=m_max, every token must walk all M_max
    segments before halting (forced-deep eval ablation)."""
    m = _make_model()
    _force_halt_at_seg1(m)  # head says halt, but min_segments forces deeper
    tok = _StubTokenizer()
    M_MAX = 4
    new = autoreg_decode_integer_hrm(
        m, tok, "abc", m_max=M_MAX, max_new=4,
        eval_min_segments=M_MAX, device="cpu",
    )
    # Every emitted token used exactly M_max segments
    assert all(s == M_MAX for s in new["segs_per_token"]), (
        f"expected every token at seg {M_MAX}; got {new['segs_per_token']}"
    )
    # Halt histogram concentrated at index M_max-1
    n_tokens = len(new["segs_per_token"])
    assert new["halt_histogram"][M_MAX - 1] == n_tokens
    assert sum(new["halt_histogram"][: M_MAX - 1]) == 0


def test_continue_always_path_hits_forced_max():
    """When head ALWAYS prefers continue (Q_continue > Q_halt), the only
    halt path is forced seg == m_max. Histogram concentrated at last bin."""
    m = _make_model()
    _force_continue_to_max(m)
    tok = _StubTokenizer()
    M_MAX = 4
    new = autoreg_decode_integer_hrm(
        m, tok, "abc", m_max=M_MAX, max_new=4,
        eval_min_segments=1, device="cpu",
    )
    assert all(s == M_MAX for s in new["segs_per_token"]), (
        f"continue-always must reach forced max each token; "
        f"got {new['segs_per_token']}"
    )
    assert new["halt_histogram"][M_MAX - 1] == len(new["segs_per_token"])


# ============================================================================
# Test (c): act_inference=True remains raising
# ============================================================================


def test_act_inference_true_still_raises():
    """Slice 13f.2 deferred act_inference on halt-head models; 13h adds
    the source-faithful helper externally but does NOT yet re-enable the
    in-forward act_inference path. Calling act_inference=True on a
    halt-head model must still raise."""
    m = _make_model()
    m.eval()
    ids = torch.randint(4, 16, (1, 8), dtype=torch.long)
    ids[0, 3] = 3  # sep
    with pytest.raises(ValueError, match="DEFERRED|deferred"):
        with torch.no_grad():
            _ = m(ids, act_inference=True)


# ============================================================================
# Test (d): outer-loop emission uses halted-segment final-pos logits
# ============================================================================


def test_emission_uses_halted_segment_final_pos_logits():
    """Hand-roll the segment loop and verify the new decoder's first
    emitted token equals the argmax of the halted segment's final-position
    logits — not, e.g., the segment 1 logits when halting at segment 2."""
    m = _make_model()
    _force_continue_to_max(m)  # always go to forced max => seg = M_max
    tok = _StubTokenizer()
    M_MAX = 4

    # Hand-roll the segment loop on the initial prefix.
    q_ids = tok.encode("abc")
    prefix = [tok.bos_id] + q_ids + [tok.sep_id]
    ids = torch.tensor([prefix], dtype=torch.long)
    m.eval()
    carry = None
    final_log_probs = None
    with torch.no_grad():
        for _ in range(M_MAX):
            out = m(ids, carry=carry, return_carry=True)
            log_probs, carry_m = out
            final_log_probs = log_probs
            carry = carry_m.detach() if carry_m is not None else None
    expected_first_tok = int(final_log_probs[0, -1].argmax().item())

    # Run the production decoder and recover its first emitted token from
    # the decoded string vs prefix.
    new = autoreg_decode_integer_hrm(
        m, tok, "abc", m_max=M_MAX, max_new=1,
        eval_min_segments=1, device="cpu",
    )
    # If max_new=1 and the model emits at least one token, decoded is a
    # length-1 string (or empty if first token is eos). Re-encode to compare.
    if new["decoded"]:
        actual_first_tok = ord(new["decoded"][0]) - 32
    else:
        actual_first_tok = tok.eos_id
    assert actual_first_tok == expected_first_tok, (
        f"new decoder must emit halted-segment final-pos argmax; "
        f"got {actual_first_tok}, expected {expected_first_tok}"
    )
