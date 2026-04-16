"""Round-23: Level 4 — compiled adder inside REAL Gemma's attention layer.

Loads real Gemma 4 E4B layer 0 from GGUF (tq4 → FP32 dequant), builds a
d_model=4096 forward pass where Gemma's 1024 sub-heads run softmax AND
a compiled adder's 5 sub-heads run hard_max — in ONE forward, ONE layer,
ONE weight tensor.

Layout (d_model=4096 substrate at d_head=2):
  Sub-heads 0..1023:    real Gemma Q (grouped softmax, 8 groups × 128)
  Sub-heads 1024..1028: compiled adder_tiny (single hard_max)
  Sub-heads 1029..2047: free (zero)

  KV sub-heads 0..255:  real Gemma K/V (GQA, 2 heads × 128)
  KV sub-heads 256..260: compiled adder K/V
  KV sub-heads 261..2047: free

  Channels 0..2559:     Gemma residual
  Channels 2560..2569:  adder channels (own, bias, copy_a, steps)
  Channels 2570..4095:  free

Memory: ~700 MB for one-layer FP32 substrate + Gemma dequant. Fits easily.

Tests:
  (a) Compiled adder at sub-heads 1024..1028 computes correct a+b at
      its output channels [2560+] — 16/16 on (a, b) ∈ [0, 3]².
  (b) Gemma's residual at channels [0, 2560) is IDENTICAL whether adder
      sub-heads are present or not — zero disturbance.
"""

from __future__ import annotations

import itertools
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from calm.llm_computer.tq4_gguf_loader import read_turboquant_gguf, extract_tq4_tensor
from calm.llm_computer.tq4_torch import dequantize_tq4, build_pi
from calm.llm_computer.programs.adder_tiny import build_adder_tiny


GGUF_PATH = Path(os.environ.get(
    "ZENITH_GEMMA_GGUF",
    "/home/gabe/models/gemma-4-E4B-it-tq4-aligned.gguf",
))

# Substrate dimensions
D_MODEL = 4096
D_GEMMA = 2560
GEMMA_Q_DIM = 2048       # 8 heads × 256 head_dim
GEMMA_KV_DIM = 512        # 2 kv heads × 256 head_dim
GEMMA_Q_SUB_HEADS = 1024  # 2048 / 2
GEMMA_KV_SUB_HEADS = 256  # 512 / 2
N_SUB_HEADS = D_MODEL // 2  # 2048

# Adder placement in free space
ADDER_CH_OFF = D_GEMMA     # 2560
ADDER_SH_OFF = GEMMA_Q_SUB_HEADS  # 1024
ADDER_KV_SH_OFF = GEMMA_KV_SUB_HEADS  # 256

VOCAB = 8
MAX_OPERAND = 3


def load_gemma_layer0_fp32(gguf_path):
    """Load Gemma layer 0 Q/K/V/O as FP32, transposed to PyTorch convention."""
    reader = read_turboquant_gguf(gguf_path)
    pi = build_pi(source="c_header")
    names = ["attn_q", "attn_k", "attn_v", "attn_output"]
    results = {}
    for n in names:
        t = extract_tq4_tensor(reader, f"blk.0.{n}.weight")
        fp = dequantize_tq4(t, pi=pi)
        results[n] = fp.T.contiguous()  # (out, in) PyTorch convention
    return results


def build_substrate_weights(gemma_weights, adder):
    """Build combined W_qkv and W_out tensors with Gemma at [0, 1024)
    sub-heads and adder at [1024, 1029) sub-heads."""
    # Substrate W_qkv shape: (3*D_MODEL, D_MODEL). Stores Q|K|V stacked.
    W_qkv = torch.zeros(3 * D_MODEL, D_MODEL)
    W_out = torch.zeros(D_MODEL, D_MODEL)

    # Install Gemma Q at rows [0, GEMMA_Q_DIM), cols [0, D_GEMMA)
    W_qkv[:GEMMA_Q_DIM, :D_GEMMA] = gemma_weights["attn_q"]
    # Gemma K at rows [D_MODEL, D_MODEL + GEMMA_KV_DIM)
    W_qkv[D_MODEL:D_MODEL + GEMMA_KV_DIM, :D_GEMMA] = gemma_weights["attn_k"]
    # Gemma V at rows [2*D_MODEL, 2*D_MODEL + GEMMA_KV_DIM)
    W_qkv[2 * D_MODEL:2 * D_MODEL + GEMMA_KV_DIM, :D_GEMMA] = gemma_weights["attn_v"]
    # Gemma O: (D_GEMMA, GEMMA_Q_DIM) → substrate rows [0, D_GEMMA), cols [0, GEMMA_Q_DIM)
    W_out[:D_GEMMA, :GEMMA_Q_DIM] = gemma_weights["attn_output"]

    # Install adder at free sub-heads
    # Adder has d_model=10 (5 sub-heads). Its W_qkv is (30, 10).
    # Q rows [0, 10) → substrate Q rows [2*ADDER_SH_OFF, 2*ADDER_SH_OFF+10)
    # K rows [10, 20) → substrate K rows [D_MODEL + 2*ADDER_KV_SH_OFF, ...]
    # V rows [20, 30) → substrate V rows [2*D_MODEL + 2*ADDER_KV_SH_OFF, ...]
    # All at cols [ADDER_CH_OFF, ADDER_CH_OFF + 10)
    a_qkv = adder.W_qkv[0].weight.data  # (30, 10) for 1-layer adder
    a_d = adder.config.d_model  # 10

    q_start = 2 * ADDER_SH_OFF
    W_qkv[q_start:q_start + a_d,
          ADDER_CH_OFF:ADDER_CH_OFF + a_d] = a_qkv[:a_d]

    k_start = D_MODEL + 2 * ADDER_KV_SH_OFF
    W_qkv[k_start:k_start + a_d,
          ADDER_CH_OFF:ADDER_CH_OFF + a_d] = a_qkv[a_d:2 * a_d]

    v_start = 2 * D_MODEL + 2 * ADDER_KV_SH_OFF
    W_qkv[v_start:v_start + a_d,
          ADDER_CH_OFF:ADDER_CH_OFF + a_d] = a_qkv[2 * a_d:3 * a_d]

    # Adder O: (10, 10) → substrate rows [ADDER_CH_OFF, +10), cols [2*ADDER_SH_OFF, +10)
    a_out = adder.W_out[0].weight.data  # (10, 10)
    W_out[ADDER_CH_OFF:ADDER_CH_OFF + a_d,
         2 * ADDER_SH_OFF:2 * ADDER_SH_OFF + a_d] = a_out

    return W_qkv, W_out


def build_substrate_ffn(adder):
    """Build combined ff_in / ff_out. Gemma FFN is skipped for this
    one-layer probe (would need Gemma's gate/up/down weights and a GLU
    FFN — out of scope for the attention-partition test). Only the
    adder's FFN neurons are populated at their offset."""
    # Use d_ffn large enough for adder. Gemma's d_ffn = 10240; use that.
    D_FFN = 10240
    ff_in = torch.zeros(2 * D_FFN, D_MODEL)
    ff_out = torch.zeros(D_MODEL, D_FFN)

    a_d = adder.config.d_model
    a_ffn = adder.config.d_ffn
    a_ff_in = adder.ff_in[0].weight.data   # (2*a_ffn, a_d)
    a_ff_out = adder.ff_out[0].weight.data  # (a_d, a_ffn)

    # Gate neurons at [0, a_ffn) of adder → substrate gate [0, a_ffn),
    # cols [ADDER_CH_OFF, +a_d)
    ff_in[:a_ffn, ADDER_CH_OFF:ADDER_CH_OFF + a_d] = a_ff_in[:a_ffn]
    # Val neurons at [D_FFN, D_FFN + a_ffn)
    ff_in[D_FFN:D_FFN + a_ffn,
         ADDER_CH_OFF:ADDER_CH_OFF + a_d] = a_ff_in[a_ffn:]
    # ff_out
    ff_out[ADDER_CH_OFF:ADDER_CH_OFF + a_d, :a_ffn] = a_ff_out

    return ff_in, ff_out, D_FFN


def partitioned_forward_one_layer(
    tok_emb, pos_emb, W_qkv, W_out, ff_in, ff_out, d_ffn, head_w, idx,
):
    """One-layer forward with per-sub-head attention partition."""
    B, S = idx.shape
    pos_idx = torch.arange(S)
    x = tok_emb[idx] + pos_emb[:S]
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)

    # QKV projection
    qkv = x @ W_qkv.T  # (B, S, 3*D_MODEL)
    q_all = qkv[:, :, :D_MODEL]
    k_all = qkv[:, :, D_MODEL:2 * D_MODEL]
    v_all = qkv[:, :, 2 * D_MODEL:]

    # Reshape to sub-heads
    q = q_all.reshape(B, S, N_SUB_HEADS, 2)
    k = k_all.reshape(B, S, N_SUB_HEADS, 2)
    v = v_all.reshape(B, S, N_SUB_HEADS, 2)

    # --- Gemma partition (sub-heads 0..1023): softmax ---
    q_g = q[:, :, :GEMMA_Q_SUB_HEADS].transpose(1, 2)  # (B, 1024, S, 2)
    # GQA: K/V at [0, 256), repeat for groups
    k_g = k[:, :, :GEMMA_KV_SUB_HEADS].transpose(1, 2)  # (B, 256, S, 2)
    v_g = v[:, :, :GEMMA_KV_SUB_HEADS].transpose(1, 2)
    # Repeat K/V for 4:1 GQA ratio
    k_g = k_g.repeat_interleave(4, dim=1)  # (B, 1024, S, 2)
    v_g = v_g.repeat_interleave(4, dim=1)
    scores_g = torch.einsum("bhid,bhjd->bhij", q_g, k_g)
    scores_g = scores_g.masked_fill(mask, float("-inf"))
    weights_g = F.softmax(scores_g, dim=-1)
    attn_g = torch.einsum("bhij,bhjd->bhid", weights_g, v_g)

    # --- Adder partition (sub-heads 1024..1028): hard_max ---
    adder_n = 5  # adder_tiny has 5 sub-heads
    q_a = q[:, :, ADDER_SH_OFF:ADDER_SH_OFF + adder_n].transpose(1, 2)
    k_a = k[:, :, ADDER_KV_SH_OFF:ADDER_KV_SH_OFF + adder_n].transpose(1, 2)
    v_a = v[:, :, ADDER_KV_SH_OFF:ADDER_KV_SH_OFF + adder_n].transpose(1, 2)
    scores_a = torch.einsum("bhid,bhjd->bhij", q_a, k_a)
    scores_a = scores_a.masked_fill(mask, float("-inf"))
    idx_a = scores_a.argmax(dim=-1, keepdim=True)
    weights_a = torch.zeros_like(scores_a)
    weights_a.scatter_(-1, idx_a, 1.0)
    attn_a = torch.einsum("bhij,bhjd->bhid", weights_a, v_a)

    # --- Free sub-heads (1029..2047): zero output ---
    # Their Q/K/V are zero → attention output is zero. Just pad.
    n_free = N_SUB_HEADS - GEMMA_Q_SUB_HEADS - adder_n
    attn_free = torch.zeros(B, n_free, S, 2)

    # Concat in head order: Gemma, adder, free
    attn = torch.cat([attn_g, attn_a, attn_free], dim=1)  # (B, 2048, S, 2)
    attn = attn.transpose(1, 2).reshape(B, S, D_MODEL)

    # W_out
    x = x + (attn @ W_out.T)

    # FFN (gate-val with ReLU)
    ffn_out = x @ ff_in.T  # (B, S, 2*d_ffn)
    gate, val = ffn_out[:, :, :d_ffn], ffn_out[:, :, d_ffn:]
    hidden = F.relu(gate) * val
    x = x + (hidden @ ff_out.T)

    # Head
    logits = x @ head_w.T
    return x, logits


def main():
    if not GGUF_PATH.exists():
        print(f"GGUF not found: {GGUF_PATH}")
        return

    t0 = time.time()
    print("[L4] loading real Gemma layer 0 from GGUF...")
    gemma_w = load_gemma_layer0_fp32(GGUF_PATH)
    print(f"  Q: {tuple(gemma_w['attn_q'].shape)}, "
          f"K: {tuple(gemma_w['attn_k'].shape)}")

    print("[L4] building compiled adder_tiny...")
    adder = build_adder_tiny(vocab_size=VOCAB, max_len=4)
    print(f"  adder d_model={adder.config.d_model}, n_heads={adder.config.n_heads}")

    print("[L4] building substrate weights (Gemma + adder at free sub-heads)...")
    W_qkv, W_out = build_substrate_weights(gemma_w, adder)
    ff_in, ff_out, d_ffn = build_substrate_ffn(adder)
    del gemma_w  # free ~40 MB

    # Embeddings: random tok (no Q6_K for this test), random pos
    torch.manual_seed(42)
    tok_emb = torch.zeros(VOCAB, D_MODEL)
    pos_emb = torch.zeros(4, D_MODEL)
    # Adder's tok embed at its channels
    a_d = adder.config.d_model
    tok_emb[:, ADDER_CH_OFF:ADDER_CH_OFF + a_d] = adder.tok.weight.data
    pos_emb[:, ADDER_CH_OFF:ADDER_CH_OFF + a_d] = adder.pos.weight.data[:4]
    # Gemma channels: small random (simulating Gemma tok_embd)
    tok_emb[:, :D_GEMMA].normal_(0, 0.02)
    pos_emb[:, :D_GEMMA].normal_(0, 0.02)

    # Head: adder's head at its channels
    head_w = torch.zeros(VOCAB, D_MODEL)
    head_w[:, ADDER_CH_OFF:ADDER_CH_OFF + a_d] = adder.head.weight.data

    print(f"  substrate d_model={D_MODEL}, sub-heads: "
          f"Gemma [0, {GEMMA_Q_SUB_HEADS}), "
          f"adder [{ADDER_SH_OFF}, {ADDER_SH_OFF + adder.config.n_heads}), "
          f"free [{ADDER_SH_OFF + adder.config.n_heads}, {N_SUB_HEADS})")
    print(f"  total weight memory: "
          f"{(W_qkv.numel() + W_out.numel() + ff_in.numel() + ff_out.numel()) * 4 / 1e6:.0f} MB")

    # --- CHECK (a): adder computes correctly ---
    print(f"\n[L4] CHECK (a) — compiled adder inside real Gemma layer")
    ok = 0
    total = 0
    with torch.no_grad():
        for a, b in itertools.product(range(MAX_OPERAND + 1), repeat=2):
            x = torch.tensor([[a, b]])
            _, logits = partitioned_forward_one_layer(
                tok_emb, pos_emb, W_qkv, W_out, ff_in, ff_out, d_ffn, head_w, x,
            )
            pred = int(logits[0, 1].argmax().item())
            expected = a + b
            if pred == expected:
                ok += 1
            else:
                print(f"  [✗] {a} + {b}: got {pred}, expected {expected}")
            total += 1
    print(f"  adder: {ok}/{total} — {'PASS' if ok == total else 'FAIL'}")

    # --- CHECK (b): Gemma residual undisturbed ---
    print(f"\n[L4] CHECK (b) — Gemma residual undisturbed by compiled sub-heads")
    # Baseline: same forward but adder sub-heads zeroed in W_qkv/W_out
    W_qkv_base = W_qkv.clone()
    W_out_base = W_out.clone()
    # Zero adder Q/K/V/O
    a_d_model = adder.config.d_model
    q_s = 2 * ADDER_SH_OFF
    W_qkv_base[q_s:q_s + a_d_model, ADDER_CH_OFF:ADDER_CH_OFF + a_d_model] = 0
    k_s = D_MODEL + 2 * ADDER_KV_SH_OFF
    W_qkv_base[k_s:k_s + a_d_model, ADDER_CH_OFF:ADDER_CH_OFF + a_d_model] = 0
    v_s = 2 * D_MODEL + 2 * ADDER_KV_SH_OFF
    W_qkv_base[v_s:v_s + a_d_model, ADDER_CH_OFF:ADDER_CH_OFF + a_d_model] = 0
    W_out_base[ADDER_CH_OFF:ADDER_CH_OFF + a_d_model,
              2 * ADDER_SH_OFF:2 * ADDER_SH_OFF + a_d_model] = 0
    # Zero adder FFN
    ff_in_base = ff_in.clone()
    ff_out_base = ff_out.clone()
    ff_in_base[:, ADDER_CH_OFF:ADDER_CH_OFF + a_d_model] = 0
    ff_out_base[ADDER_CH_OFF:ADDER_CH_OFF + a_d_model, :] = 0

    x_probe = torch.tensor([[0, 1]])
    with torch.no_grad():
        res_with, _ = partitioned_forward_one_layer(
            tok_emb, pos_emb, W_qkv, W_out, ff_in, ff_out, d_ffn, head_w, x_probe,
        )
        res_without, _ = partitioned_forward_one_layer(
            tok_emb, pos_emb, W_qkv_base, W_out_base, ff_in_base, ff_out_base,
            d_ffn, head_w, x_probe,
        )

    gemma_diff = (res_with[:, :, :D_GEMMA] - res_without[:, :, :D_GEMMA]).abs().max().item()
    ok_b = gemma_diff < 1e-5
    print(f"  max |Gemma_channels_with_adder - Gemma_channels_without| = {gemma_diff:.2e}")
    print(f"  {'PASS' if ok_b else 'FAIL'}")

    all_ok = (ok == total) and ok_b
    t = time.time() - t0
    print(f"\n[L4] OVERALL: {'PASS' if all_ok else 'FAIL'}  ({t:.1f}s)")
    print("[L4] Level 4 — compiled program inside REAL Gemma's attention:")
    print(f"[L4]   {'VALIDATED' if all_ok else 'NOT VALIDATED'}")
    if all_ok:
        print(f"[L4]   Gemma's {GEMMA_Q_SUB_HEADS} sub-heads (real tq4 bytes) "
              f"+ {adder.config.n_heads} compiled sub-heads")
        print(f"[L4]   = one tensor, one layer, one forward, zero cross-talk")


if __name__ == "__main__":
    main()
