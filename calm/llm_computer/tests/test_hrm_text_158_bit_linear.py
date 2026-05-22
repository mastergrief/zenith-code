"""HRM-Text-1.58 Phase 2 Slice 1: BitLinear tests.

Per task #51, codex msg 1779457170889 Phase 2 Slice 1 +1.

Step 1 assertions (all 4 codex categories):
- [types]: BitLinear quantizes to {-1, 0, +1}; finite scale
- [tests]: STE backward sends nonzero gradient to FP master weight;
  state_dict round-trip preserves master weights (NOT quantized values);
  D2.2 scope -- lm_head/embd/norms/zL_init stay FP when use_ternary_bulk=True
- [runtime]: 1-step trainer-shaped forward with ternary enabled is finite,
  emits same metrics schema as FP
- [data]: no dataset/probe denominator changes (verified by structural test
  of LMHead metrics dict shape)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from calm.hrm_text_158 import (
    BitLinear,
    HierarchicalReasoningModel,
    HierarchicalReasoningModelConfig,
    LMHead,
    LMHeadConfig,
    LinearInit,
    ScaledEmbeddingInit,
)
from calm.hrm_text_158.lm_head import IGNORE_LABEL_ID


# ----------------------------------------------------------------------------- #
# [types]: BitLinear quantizes to {-1, 0, +1} with finite scale
# ----------------------------------------------------------------------------- #

def test_bit_linear_quantizes_to_ternary_levels() -> None:
    """Codex Step 1 [types]: get_ternary_levels() must return only {-1, 0, +1}."""
    torch.manual_seed(42)
    bl = BitLinear(in_features=64, out_features=32, bias=False)
    levels = bl.get_ternary_levels()
    unique = torch.unique(levels)
    # Must be a subset of {-1, 0, +1}
    allowed = torch.tensor([-1.0, 0.0, 1.0])
    for v in unique:
        assert any(abs(v.item() - a) < 1e-6 for a in allowed.tolist()), (
            f"ternary level {v.item()} not in {{-1, 0, +1}}; full set: {unique.tolist()}"
        )


def test_bit_linear_scale_finite_and_positive() -> None:
    """Codex Step 1 [types]: scale (per-tensor absmean) must be finite + positive."""
    torch.manual_seed(42)
    bl = BitLinear(in_features=64, out_features=32, bias=False)
    _, scale = bl.quantize_weight()
    assert torch.isfinite(scale).item()
    assert scale.item() > 0


def test_bit_linear_zero_weight_uses_eps_floor() -> None:
    """Defensive: if all weights are zero, scale falls back to _SCALE_EPS, no NaN."""
    bl = BitLinear(in_features=64, out_features=32, bias=False)
    with torch.no_grad():
        bl.weight.zero_()
    w_q, scale = bl.quantize_weight()
    assert torch.isfinite(w_q).all()
    assert torch.isfinite(scale).item()
    # fp32 representation of 1e-5 is not exact; use approx
    assert scale.item() == pytest.approx(BitLinear._SCALE_EPS, rel=1e-5)


# ----------------------------------------------------------------------------- #
# [tests]: STE backward + state_dict preservation
# ----------------------------------------------------------------------------- #

def test_bit_linear_ste_backward_nonzero_grad() -> None:
    """Codex Step 1 [tests]: STE sends nonzero gradient to FP master.

    Backward through w_q*scale must hit `self.weight.grad` via the
    identity STE trick: w + sg(w_q*scale - w).
    """
    torch.manual_seed(42)
    bl = BitLinear(in_features=32, out_features=16, bias=False)
    x = torch.randn(4, 32, requires_grad=False)
    y = bl(x)
    loss = y.sum()
    loss.backward()
    assert bl.weight.grad is not None, "no grad on weight after backward"
    assert torch.isfinite(bl.weight.grad).all()
    # At least SOME weights must have nonzero grad (the linear chain
    # rule applied through STE gives nonzero gradient on any weight
    # whose input dim is active).
    assert (bl.weight.grad.abs() > 0).any(), "STE produced all-zero gradient"


def test_bit_linear_forward_uses_quantized_value() -> None:
    """Forward output must equal F.linear with the (quantized*scale) values,
    NOT the master FP weights. Verify by computing both paths."""
    torch.manual_seed(42)
    bl = BitLinear(in_features=16, out_features=8, bias=False)
    x = torch.randn(3, 16)
    y_bit = bl(x)
    # Reproduce forward path manually:
    w_q, scale = bl.quantize_weight()
    y_manual = F.linear(x, w_q)
    assert torch.allclose(y_bit, y_manual, atol=1e-6)
    # Different from raw FP linear (since quantized weights differ from FP master)
    y_fp = F.linear(x, bl.weight)
    diff = (y_bit - y_fp).abs().max().item()
    assert diff > 1e-3, (
        f"BitLinear forward should differ from raw FP linear (got max diff {diff:.6e}); "
        f"if 0, quantization is no-op."
    )


def test_bit_linear_state_dict_preserves_master_weight() -> None:
    """Codex Step 1 [tests]: state_dict save/load preserves the FP/BF16
    MASTER weight (not the quantized snapshot). Roundtrip-load and
    confirm bl_b.weight == bl_a.weight."""
    torch.manual_seed(42)
    bl_a = BitLinear(in_features=32, out_features=16, bias=True)
    sd = bl_a.state_dict()
    assert "weight" in sd
    # Bias only if requested
    assert "bias" in sd
    # Master weights are FP — should match the saved tensor exactly
    assert torch.equal(sd["weight"], bl_a.weight)

    bl_b = BitLinear(in_features=32, out_features=16, bias=True)
    # Different init → different weights initially
    assert not torch.equal(bl_a.weight, bl_b.weight)
    bl_b.load_state_dict(sd)
    # After load: same master weight
    assert torch.equal(bl_a.weight, bl_b.weight)
    # And forward outputs match (since master weights match)
    x = torch.randn(2, 32)
    assert torch.allclose(bl_a(x), bl_b(x), atol=1e-6)


# ----------------------------------------------------------------------------- #
# D2.2 scope: lm_head / embed_tokens / norms / zL_init stay FP
# ----------------------------------------------------------------------------- #

def test_d22_scope_lm_head_stays_fp_when_ternary_bulk_on() -> None:
    """Codex Step 1 [tests]: D2.2 bounded scope. When use_ternary_bulk=True,
    only attention.gqkv_proj/o_proj + mlp.gate_up_proj/down_proj become
    BitLinear. lm_head, embed_tokens, zL_init stay FP."""
    cfg = HierarchicalReasoningModelConfig(
        max_seq_len=32, n_layers=2, hidden_size=32, num_heads=2,
        expansion=4, H_cycles=2, L_cycles=2, half_layers=True,
        use_ternary_bulk=True,
    )
    hrm = HierarchicalReasoningModel(cfg)
    m = LMHead(hrm, LMHeadConfig(vocab_size=98))

    # lm_head: LinearInit (NOT BitLinear) per D2.2
    assert isinstance(m.lm_head, LinearInit), (
        f"lm_head must be LinearInit (FP) per D2.2; got {type(m.lm_head).__name__}"
    )
    assert not isinstance(m.lm_head, BitLinear)
    # embed_tokens: ScaledEmbeddingInit per D2.2
    assert isinstance(m.embed_tokens, ScaledEmbeddingInit)
    # zL_init: nn.Buffer (NOT a Parameter, NOT BitLinear-wrapped) per D2.2
    assert isinstance(hrm.zL_init, torch.Tensor)
    # Confirm zL_init is registered as buffer (not parameter)
    assert "zL_init" in dict(hrm.named_buffers())

    # All attention.gqkv_proj/o_proj + mlp.gate_up_proj/down_proj in BOTH H_level + L_level
    # transformer blocks MUST be BitLinear per D2.1
    bitlinear_count = 0
    linearinit_count = 0
    for name, mod in m.named_modules():
        # The 4 bulk projections live inside transformer blocks
        leaf_name = name.split(".")[-1] if name else ""
        if leaf_name in ("gqkv_proj", "o_proj", "gate_up_proj", "down_proj"):
            assert isinstance(mod, BitLinear), (
                f"{name} must be BitLinear when use_ternary_bulk=True; "
                f"got {type(mod).__name__}"
            )
            bitlinear_count += 1
        elif leaf_name == "lm_head":
            assert isinstance(mod, LinearInit) and not isinstance(mod, BitLinear)
            linearinit_count += 1
    # n_layers=2 + half_layers=True → effective_n_layers=1 per H/L bank
    # 1 H block + 1 L block = 2 transformer blocks total
    # Each block has: attn (gqkv_proj + o_proj) + mlp (gate_up_proj + down_proj) = 4 projections
    # → 8 BitLinear instances expected
    assert bitlinear_count == 8, f"expected 8 BitLinear instances; got {bitlinear_count}"


def test_d22_scope_fp_baseline_has_no_bitlinear() -> None:
    """Negative-assertion: when use_ternary_bulk=False (FP baseline), NO
    BitLinear instances exist in the model."""
    cfg = HierarchicalReasoningModelConfig(
        max_seq_len=32, n_layers=2, hidden_size=32, num_heads=2,
        expansion=4, H_cycles=2, L_cycles=2, half_layers=True,
        use_ternary_bulk=False,
    )
    hrm = HierarchicalReasoningModel(cfg)
    m = LMHead(hrm, LMHeadConfig(vocab_size=98))
    for name, mod in m.named_modules():
        assert not isinstance(mod, BitLinear), (
            f"FP baseline must have NO BitLinear; found {name} of type {type(mod).__name__}"
        )


# ----------------------------------------------------------------------------- #
# [runtime]: full 1-step training cycle with ternary enabled is finite
# ----------------------------------------------------------------------------- #

def test_full_step_finite_with_ternary_bulk() -> None:
    """Codex Step 1 [runtime]: one tiny train step with ternary enabled
    must be finite end-to-end (loss + grad + opt.step + final params)."""
    cfg = HierarchicalReasoningModelConfig(
        max_seq_len=32, n_layers=2, hidden_size=32, num_heads=2,
        expansion=4, H_cycles=2, L_cycles=2, half_layers=True,
        use_ternary_bulk=True,
    )
    torch.manual_seed(42)
    hrm = HierarchicalReasoningModel(cfg)
    m = LMHead(hrm, LMHeadConfig(vocab_size=98))
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

    B, S = 2, 16
    ids = torch.randint(0, 98, (B, S), dtype=torch.long)
    sep = torch.tensor([5, 7], dtype=torch.long)
    pos = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, -1)
    labels = torch.full((B, S), IGNORE_LABEL_ID, dtype=torch.long)
    labels[0, 6:11] = torch.randint(1, 98, (5,))
    labels[1, 8:13] = torch.randint(1, 98, (5,))

    orig_param = next(m.parameters()).detach().clone()
    new_carry, loss, metrics = m(
        None,
        {"inputs": ids, "sep_positions": sep, "position_ids": pos, "labels": labels},
        bp_steps=2,
    )
    assert torch.isfinite(loss), f"ternary loss not finite: {loss}"
    opt.zero_grad()
    loss.backward()
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad on {n}"
    grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
    assert torch.isfinite(grad_norm)
    opt.step()
    new_param = next(m.parameters()).detach()
    # Master weights must move (gradient flowed through STE)
    assert not torch.allclose(orig_param, new_param), (
        "master params did not move after opt.step (STE backward likely broken)"
    )
    for n, p in m.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite param after step: {n}"


def test_ternary_metrics_schema_matches_fp() -> None:
    """Codex Step 1 [data]: ternary path emits the same metrics schema as FP.

    Critical for the Step 2 A/B — receipt comparison requires identical
    metrics field names + shapes."""
    torch.manual_seed(42)

    def _build(use_ternary):
        cfg = HierarchicalReasoningModelConfig(
            max_seq_len=32, n_layers=2, hidden_size=32, num_heads=2,
            expansion=4, H_cycles=2, L_cycles=2, half_layers=True,
            use_ternary_bulk=use_ternary,
        )
        hrm = HierarchicalReasoningModel(cfg)
        return LMHead(hrm, LMHeadConfig(vocab_size=98))

    m_fp = _build(False).eval()
    m_tern = _build(True).eval()

    B, S = 2, 16
    ids = torch.randint(0, 98, (B, S), dtype=torch.long)
    sep = torch.tensor([5, 7], dtype=torch.long)
    pos = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, -1)
    labels = torch.full((B, S), IGNORE_LABEL_ID, dtype=torch.long)
    labels[0, 6:11] = torch.randint(1, 98, (5,))
    labels[1, 8:13] = torch.randint(1, 98, (5,))
    batch = {"inputs": ids, "sep_positions": sep, "position_ids": pos, "labels": labels}

    with torch.no_grad():
        _, _, m_fp_metrics = m_fp(None, batch)
        _, _, m_tern_metrics = m_tern(None, batch)

    # Same metric keys
    assert set(m_fp_metrics.keys()) == set(m_tern_metrics.keys())
    # Same per-key tuple shape
    for k in m_fp_metrics:
        assert len(m_fp_metrics[k]) == len(m_tern_metrics[k]), (
            f"metric {k} tuple length differs"
        )


# ----------------------------------------------------------------------------- #
# state_dict round-trip preserves master weights for FULL HRM-text-158 model
# (load -> forward -> same output as before save)
# ----------------------------------------------------------------------------- #

def test_probe_reconstruction_respects_ckpt_use_ternary_bulk(tmp_path) -> None:
    """Codex msg 1779457628632 (load-bearing): probe.reconstruct must
    pass use_ternary_bulk through from the saved ckpt config blob.

    Without this, a ternary-trained ckpt reconstructs as an FP LinearInit
    model (state_dict keys match because BitLinear + LinearInit both use
    `weight`/`bias`), and inference runs FP linear over the master weights
    -- silently wrong probe results, false A/B.

    This test simulates the save→reconstruct round-trip and asserts the
    reconstructed model has BitLinear in the 4 bulk projection sites
    when the ckpt blob carries use_ternary_bulk=True.
    """
    import sys
    sys.path.insert(0, ".")
    from scripts.probe_hrm_text_158 import _build_model_from_ckpt
    from calm.llm_computer.gsm8k_tokenizer import Gsm8kTokenizer

    torch.manual_seed(42)
    # Build small ternary model + save its state_dict + config blob
    cfg = HierarchicalReasoningModelConfig(
        max_seq_len=32, n_layers=2, hidden_size=32, num_heads=2,
        expansion=4, H_cycles=2, L_cycles=2, half_layers=True,
        use_ternary_bulk=True,
    )
    hrm = HierarchicalReasoningModel(cfg)
    tok = Gsm8kTokenizer.from_corpus([
        {"question": "what is 1 plus 1?", "expected": 2},
        {"question": "what is 17 times 23?", "expected": 391},
    ])
    m_src = LMHead(hrm, LMHeadConfig(vocab_size=tok.vocab_size))

    # Build the ckpt blob with the SAME shape as the trainer writes
    ckpt = {
        "model_state": m_src.state_dict(),
        "config": {
            "vocab_size": tok.vocab_size,
            "gsm8k_char_vocab": tok.vocab_as_list(),
            "gsm8k_normalizer_version": tok.normalizer_version,
            "max_seq_len": cfg.max_seq_len,
            "n_layers": cfg.n_layers,
            "hidden_size": cfg.hidden_size,
            "num_heads": cfg.num_heads,
            "expansion": cfg.expansion,
            "H_cycles": cfg.H_cycles,
            "L_cycles": cfg.L_cycles,
            "half_layers": cfg.half_layers,
            "bp_warmup_ratio": cfg.bp_warmup_ratio,
            "bp_min_steps": cfg.bp_min_steps,
            "bp_max_steps": cfg.bp_max_steps,
            "norm_type": cfg.norm_type,
            "norm_eps": cfg.norm_eps,
            "rope_theta": cfg.rope_theta,
            "attn_type": cfg.attn_type,
            "init_type": cfg.init_type,
            "pos_emb_type": cfg.pos_emb_type,
            "use_ternary_bulk": True,  # CRITICAL: ckpt carries the flag
        },
    }

    # Reconstruct via probe's loader
    m_recon, tok_recon = _build_model_from_ckpt(ckpt, device="cpu")

    # Verify ALL bulk-projection sites are BitLinear (not FP LinearInit)
    bitlinear_count = 0
    for name, mod in m_recon.named_modules():
        leaf = name.split(".")[-1] if name else ""
        if leaf in ("gqkv_proj", "o_proj", "gate_up_proj", "down_proj"):
            assert isinstance(mod, BitLinear), (
                f"PROBE RECONSTRUCTION BUG: {name} should be BitLinear "
                f"when ckpt['config']['use_ternary_bulk']=True, "
                f"got {type(mod).__name__}. Codex msg 1779457628632."
            )
            bitlinear_count += 1
    # 2 transformer blocks × 4 projections = 8 BitLinear
    assert bitlinear_count == 8

    # And lm_head MUST remain FP LinearInit per D2.2
    assert isinstance(m_recon.lm_head, LinearInit) and not isinstance(m_recon.lm_head, BitLinear)


def test_probe_reconstruction_falls_back_to_fp_when_flag_absent() -> None:
    """Backwards-compat: ckpts saved BEFORE Phase 2 won't have
    `use_ternary_bulk` in their config blob. Default to False (FP).
    Important for loading the FP Tier A / Tier B ckpts already on disk."""
    import sys
    sys.path.insert(0, ".")
    from scripts.probe_hrm_text_158 import _build_model_from_ckpt
    from calm.llm_computer.gsm8k_tokenizer import Gsm8kTokenizer

    torch.manual_seed(42)
    cfg = HierarchicalReasoningModelConfig(
        max_seq_len=32, n_layers=2, hidden_size=32, num_heads=2,
        expansion=4, H_cycles=2, L_cycles=2, half_layers=True,
        use_ternary_bulk=False,
    )
    hrm = HierarchicalReasoningModel(cfg)
    tok = Gsm8kTokenizer.from_corpus([
        {"question": "what is 1 plus 1?", "expected": 2},
    ])
    m_src = LMHead(hrm, LMHeadConfig(vocab_size=tok.vocab_size))

    # Ckpt blob WITHOUT use_ternary_bulk key (simulates pre-Phase-2 ckpt)
    ckpt = {
        "model_state": m_src.state_dict(),
        "config": {
            "vocab_size": tok.vocab_size,
            "gsm8k_char_vocab": tok.vocab_as_list(),
            "gsm8k_normalizer_version": tok.normalizer_version,
            "max_seq_len": cfg.max_seq_len,
            "n_layers": cfg.n_layers,
            "hidden_size": cfg.hidden_size,
            "num_heads": cfg.num_heads,
            "expansion": cfg.expansion,
            "H_cycles": cfg.H_cycles,
            "L_cycles": cfg.L_cycles,
            "half_layers": cfg.half_layers,
            "bp_warmup_ratio": cfg.bp_warmup_ratio,
            "bp_min_steps": cfg.bp_min_steps,
            "bp_max_steps": cfg.bp_max_steps,
            "norm_type": cfg.norm_type,
            "norm_eps": cfg.norm_eps,
            "rope_theta": cfg.rope_theta,
            "attn_type": cfg.attn_type,
            "init_type": cfg.init_type,
            "pos_emb_type": cfg.pos_emb_type,
            # use_ternary_bulk key OMITTED -- pre-Phase-2 ckpt shape
        },
    }
    m_recon, _ = _build_model_from_ckpt(ckpt, device="cpu")
    # Reconstructed model should be FP (no BitLinear instances anywhere)
    for name, mod in m_recon.named_modules():
        assert not isinstance(mod, BitLinear), (
            f"Pre-Phase-2 ckpt missing use_ternary_bulk should default to FP; "
            f"got {name} as BitLinear"
        )


def test_full_model_ternary_state_dict_round_trip() -> None:
    """End-to-end save/load preserves FP master weights, NOT quantized snapshot."""
    torch.manual_seed(42)
    cfg = HierarchicalReasoningModelConfig(
        max_seq_len=32, n_layers=2, hidden_size=32, num_heads=2,
        expansion=4, H_cycles=2, L_cycles=2, half_layers=True,
        use_ternary_bulk=True,
    )
    m_a = LMHead(HierarchicalReasoningModel(cfg), LMHeadConfig(vocab_size=98)).eval()

    B, S = 1, 8
    ids = torch.randint(0, 98, (B, S), dtype=torch.long)
    sep = torch.tensor([3], dtype=torch.long)
    pos = torch.arange(S, dtype=torch.long).unsqueeze(0)
    batch = {"inputs": ids, "sep_positions": sep, "position_ids": pos}

    with torch.no_grad():
        _, logits_a = m_a(None, batch)

    sd = m_a.state_dict()

    # Build fresh model + load
    torch.manual_seed(99)  # different seed → different init
    m_b = LMHead(HierarchicalReasoningModel(cfg), LMHeadConfig(vocab_size=98)).eval()
    res = m_b.load_state_dict(sd, strict=True)
    assert len(res.missing_keys) == 0
    assert len(res.unexpected_keys) == 0

    with torch.no_grad():
        _, logits_b = m_b(None, batch)
    assert torch.allclose(logits_a, logits_b, atol=1e-5)
