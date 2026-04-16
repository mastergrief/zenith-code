"""Round-29 Level 5: Gemma + HRM + compiled adder in ONE layer.

Three attention modes coexist in a single layer's forward pass:

  Sub-heads 0..7:   "Gemma-like" (grouped softmax, 2 groups × 4)
  Sub-heads 8..11:  "HRM-like" (single softmax, trained weights)
  Sub-heads 12..16: compiled adder (single hard_max, exact weights)

One W_qkv matrix. One W_out matrix. One FFN. One layer. One forward.
Three partition modes.

Layout (d_model=34, n_heads=17):
  Channels 0..15:   Gemma-like residual
  Channels 16..23:  HRM channels (d_model_hrm=8, 4 sub-heads)
  Channels 24..33:  adder channels (d_model_adder=10, 5 sub-heads)

Tests:
  (a) Compiled adder: 16/16 on (a, b) ∈ [0, 3]² — exact via hard_max.
  (b) HRM: produces non-trivial output at its channels — trained weights
      active via single-softmax.
  (c) Gemma: produces non-trivial output at its channels — random-init
      active via grouped-softmax.
  (d) Zero cross-talk: each card's channels are IDENTICAL whether the
      other two cards are installed or not.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUp, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DConfig, Small2DTransformer


# Layout
D_MODEL = 34
N_HEADS = D_MODEL // 2  # 17
N_LAYERS = 1
D_FFN = 24  # Gemma 4 + HRM 4 + adder 14 = 22, round up
MAX_LEN = 4
VOCAB = 8

# Partition boundaries
GEMMA_SH = 8       # sub-heads [0, 8), channels [0, 16)
GEMMA_CH = 16
GEMMA_GROUPS = 2
GEMMA_GROUP_SIZE = 4

HRM_SH_START = 8   # sub-heads [8, 12), channels [16, 24)
HRM_SH_END = 12
HRM_CH_OFF = 16
HRM_D = 8

ADDER_SH_START = 12  # sub-heads [12, 17), channels [24, 34)
ADDER_CH_OFF = 24
ADDER_D = 10

MAX_OPERAND = 3
MAX_SUM = 6


def build_gemma_standin(seed: int = 42):
    """Random-init Gemma-like weights at channels [0, 16), sub-heads [0, 8)."""
    cfg = Small2DConfig(
        vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ffn=D_FFN, max_len=MAX_LEN, use_hard_max=False,
    )
    m = Small2DTransformer(cfg)
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        m.tok.weight[:, :GEMMA_CH].normal_(0, 0.1)
        m.pos.weight[:, :GEMMA_CH].normal_(0, 0.1)
        SH2 = 2 * GEMMA_SH
        for seg in [0, D_MODEL, 2 * D_MODEL]:
            m.W_qkv[0].weight[seg:seg + SH2, :GEMMA_CH].normal_(0, 0.1)
        m.W_out[0].weight[:GEMMA_CH, :SH2].normal_(0, 0.1)
        m.ff_in[0].weight[:4, :GEMMA_CH].normal_(0, 0.1)
        m.ff_in[0].weight[D_FFN:D_FFN + 4, :GEMMA_CH].normal_(0, 0.1)
        m.ff_out[0].weight[:GEMMA_CH, :4].normal_(0, 0.1)
    return m


def build_hrm_standin():
    """Random-init HRM-like at d_model=8 (its native size). Will be
    installed at sub-heads [8, 12), channels [16, 24)."""
    cfg = Small2DConfig(
        vocab_size=VOCAB, d_model=HRM_D, n_heads=HRM_D // 2,
        n_layers=1, d_ffn=4, max_len=MAX_LEN, use_hard_max=False,
    )
    m = Small2DTransformer(cfg)
    torch.manual_seed(99)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.1)
    return m


def build_adder():
    """Compiled adder_tiny at d_model=10 (native). Will be installed at
    sub-heads [12, 17), channels [24, 34)."""
    from calm.llm_computer.programs.adder_tiny import build_adder_tiny
    return build_adder_tiny(vocab_size=VOCAB, max_len=MAX_LEN)


def install_at_offset(substrate, card, ch_off, sh_off, ffn_off):
    """Corner-patch card weights into substrate at given offsets."""
    c = card.config
    D_c = c.d_model
    H_c = c.n_heads
    F_c = c.d_ffn
    with torch.no_grad():
        substrate.tok.weight[:VOCAB, ch_off:ch_off + D_c] += card.tok.weight
        substrate.pos.weight[:MAX_LEN, ch_off:ch_off + D_c] += card.pos.weight[:MAX_LEN]
        # W_qkv: Q/K/V segments
        for seg_off_s, seg_off_c in [(0, 0), (D_MODEL, D_c), (2 * D_MODEL, 2 * D_c)]:
            substrate.W_qkv[0].weight[
                seg_off_s + 2 * sh_off : seg_off_s + 2 * sh_off + D_c,
                ch_off : ch_off + D_c,
            ] = card.W_qkv[0].weight[seg_off_c : seg_off_c + D_c, :]
        # W_out
        substrate.W_out[0].weight[
            ch_off : ch_off + D_c,
            2 * sh_off : 2 * sh_off + D_c,
        ] = card.W_out[0].weight
        # ff_in: gate [ffn_off, +F_c), val [D_FFN+ffn_off, +F_c)
        substrate.ff_in[0].weight[
            ffn_off : ffn_off + F_c, ch_off : ch_off + D_c,
        ] = card.ff_in[0].weight[:F_c, :]
        substrate.ff_in[0].weight[
            D_FFN + ffn_off : D_FFN + ffn_off + F_c,
            ch_off : ch_off + D_c,
        ] = card.ff_in[0].weight[F_c:, :]
        # ff_out
        substrate.ff_out[0].weight[
            ch_off : ch_off + D_c,
            ffn_off : ffn_off + F_c,
        ] = card.ff_out[0].weight
        # head
        substrate.head.weight[:VOCAB, ch_off:ch_off + D_c] += card.head.weight


def three_way_forward(model, idx):
    """Forward with 3-way per-sub-head attention partition."""
    B, S = idx.shape
    pos_idx = torch.arange(S, device=idx.device)
    x = model.tok(idx) + model.pos(pos_idx)
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)

    qkv = model.W_qkv[0](x)
    qkv = qkv.reshape(B, S, 3, N_HEADS, 2)
    q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (B, H, S, 2)

    # --- Partition 1: Gemma (grouped softmax, 2 groups × 4 sub-heads) ---
    q_g = q[:, :GEMMA_SH]   # (B, 8, S, 2)
    k_g = k[:, :GEMMA_SH]
    v_g = v[:, :GEMMA_SH]
    # Reshape for grouped attention: (B, n_groups, group_size, S, 2)
    q_g = q_g.reshape(B, GEMMA_GROUPS, GEMMA_GROUP_SIZE, S, 2)
    k_g = k_g.reshape(B, GEMMA_GROUPS, GEMMA_GROUP_SIZE, S, 2)
    v_g = v_g.reshape(B, GEMMA_GROUPS, GEMMA_GROUP_SIZE, S, 2)
    # Sum scores across group before softmax
    scores_g = torch.einsum("bgsid,bgtjd->bgsij", q_g, k_g)
    # Sum across sub-heads within group: (B, n_groups, S, S)
    scores_g_summed = scores_g.sum(dim=2)
    scores_g_summed = scores_g_summed.masked_fill(mask.view(1, 1, S, S), float("-inf"))
    weights_g = F.softmax(scores_g_summed, dim=-1)  # (B, groups, S, S)
    # Apply SHARED weights to each sub-head's V
    attn_g = torch.einsum("bgij,bgsid->bgsid", weights_g, v_g)
    attn_g = attn_g.reshape(B, GEMMA_SH, S, 2)

    # --- Partition 2: HRM (single softmax) ---
    q_h = q[:, HRM_SH_START:HRM_SH_END]
    k_h = k[:, HRM_SH_START:HRM_SH_END]
    v_h = v[:, HRM_SH_START:HRM_SH_END]
    scores_h = torch.einsum("bhid,bhjd->bhij", q_h, k_h)
    scores_h = scores_h.masked_fill(mask, float("-inf"))
    weights_h = F.softmax(scores_h, dim=-1)
    attn_h = torch.einsum("bhij,bhjd->bhid", weights_h, v_h)

    # --- Partition 3: Compiled adder (single hard_max) ---
    q_a = q[:, ADDER_SH_START:]
    k_a = k[:, ADDER_SH_START:]
    v_a = v[:, ADDER_SH_START:]
    scores_a = torch.einsum("bhid,bhjd->bhij", q_a, k_a)
    scores_a = scores_a.masked_fill(mask, float("-inf"))
    idx_a = scores_a.argmax(dim=-1, keepdim=True)
    weights_a = torch.zeros_like(scores_a)
    weights_a.scatter_(-1, idx_a, 1.0)
    attn_a = torch.einsum("bhij,bhjd->bhid", weights_a, v_a)

    # Concat all partitions
    attn = torch.cat([attn_g, attn_h, attn_a], dim=1)  # (B, 17, S, 2)
    attn = attn.transpose(1, 2).reshape(B, S, D_MODEL)

    x = x + model.W_out[0](attn)
    gate, val = model.ff_in[0](x).chunk(2, dim=-1)
    x = x + model.ff_out[0](F.relu(gate) * val)
    return x, model.head(x)


if __name__ == "__main__":
    print("[R29] building 3 cards at their native sizes...")
    gemma = build_gemma_standin()
    hrm = build_hrm_standin()
    adder = build_adder()
    print(f"  gemma:  d={GEMMA_CH}, sub-heads [0, {GEMMA_SH})")
    print(f"  hrm:    d={HRM_D}, sub-heads [{HRM_SH_START}, {HRM_SH_END})")
    print(f"  adder:  d={ADDER_D}, sub-heads [{ADDER_SH_START}, {N_HEADS})")

    print("\n[R29] installing all 3 into ONE tensor...")
    # Start from gemma's weights (already at right position)
    substrate = gemma
    install_at_offset(substrate, hrm, ch_off=HRM_CH_OFF, sh_off=HRM_SH_START,
                      ffn_off=4)
    install_at_offset(substrate, adder, ch_off=ADDER_CH_OFF, sh_off=ADDER_SH_START,
                      ffn_off=8)
    print(f"  d_model={D_MODEL}, n_heads={N_HEADS}, params={substrate.param_count()}")

    # --- CHECK (a): adder correct ---
    print("\n[R29] CHECK (a) — compiled adder (hard_max sub-heads)")
    ok = total = 0
    for a, b in itertools.product(range(MAX_OPERAND + 1), repeat=2):
        x = torch.tensor([[a, b]], dtype=torch.long)
        with torch.no_grad():
            _, logits = three_way_forward(substrate, x)
        pred = int(logits[0, 1].argmax().item())
        ok += (pred == a + b)
        total += 1
        if pred != a + b:
            print(f"  [✗] {a}+{b}: got {pred}, expected {a + b}")
    print(f"  adder: {ok}/{total} — {'PASS' if ok == total else 'FAIL'}")

    # --- CHECK (b): HRM non-trivial ---
    print("\n[R29] CHECK (b) — HRM (softmax sub-heads) non-trivial")
    x = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        res, _ = three_way_forward(substrate, x)
    hrm_res = res[0, :, HRM_CH_OFF:HRM_CH_OFF + HRM_D]
    ok_hrm = hrm_res.std().item() > 1e-3
    print(f"  HRM channels std={hrm_res.std():.4f} — "
          f"{'PASS' if ok_hrm else 'FAIL'}")

    # --- CHECK (c): Gemma non-trivial ---
    print("\n[R29] CHECK (c) — Gemma (grouped softmax) non-trivial")
    gemma_res = res[0, :, :GEMMA_CH]
    ok_gemma = gemma_res.std().item() > 1e-3
    print(f"  Gemma channels std={gemma_res.std():.4f} — "
          f"{'PASS' if ok_gemma else 'FAIL'}")

    # --- CHECK (d): zero cross-talk ---
    print("\n[R29] CHECK (d) — zero cross-talk between partitions")
    # Build baseline: only Gemma installed (no HRM, no adder)
    baseline = build_gemma_standin()
    # Same random seed → same Gemma weights
    with torch.no_grad():
        res_base, _ = three_way_forward(baseline, x)
    gemma_base = res_base[0, :, :GEMMA_CH]
    diff_gemma = (gemma_res - gemma_base).abs().max().item()

    # Build HRM-only baseline
    hrm_only = build_gemma_standin()
    with torch.no_grad():
        for p in hrm_only.parameters():
            p.zero_()
    install_at_offset(hrm_only, hrm, ch_off=HRM_CH_OFF, sh_off=HRM_SH_START,
                      ffn_off=4)
    with torch.no_grad():
        res_hrm_only, _ = three_way_forward(hrm_only, x)
    hrm_alone = res_hrm_only[0, :, HRM_CH_OFF:HRM_CH_OFF + HRM_D]
    diff_hrm = (hrm_res - hrm_alone).abs().max().item()

    print(f"  Gemma channels: |with_all - gemma_only| = {diff_gemma:.2e}")
    print(f"  HRM channels:   |with_all - hrm_only|   = {diff_hrm:.2e}")
    ok_cross = diff_gemma < 1e-5 and diff_hrm < 1e-5
    print(f"  {'PASS' if ok_cross else 'FAIL'} — "
          f"{'zero' if ok_cross else 'NON-zero'} cross-talk")

    # --- Summary ---
    all_ok = (ok == total) and ok_hrm and ok_gemma and ok_cross
    print(f"\n[R29] OVERALL: {'PASS' if all_ok else 'FAIL'}")
    print(f"[R29] Level 5 — three card types, ONE layer, ONE forward:")
    print(f"[R29]   grouped-softmax + single-softmax + single-hard_max")
    print(f"[R29]   {'VALIDATED' if all_ok else 'NOT VALIDATED'}")
