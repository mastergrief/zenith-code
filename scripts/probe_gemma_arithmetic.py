"""Round-22B: probe real Gemma's d_head=2 sub-heads for arithmetic circuits.

Loads 2 layers of Gemma 4 E4B from GGUF, decomposes Q/K/V to d_head=2
sub-heads (1024 per SWA layer), runs arithmetic prompts, measures
per-sub-head activation magnitude. Identifies which sub-heads respond
most strongly to arithmetic tokens — these are candidates for future
compiled replacement.

Method: for each layer and sub-head, compute the L2 norm of the
attention output (v_weighted) on arithmetic vs non-arithmetic prompts.
Sub-heads with high arithmetic activation and low non-arithmetic
activation are the "arithmetic circuit."
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from calm.llm_computer.tq4_gguf_loader import read_turboquant_gguf, extract_tq4_tensor
from calm.llm_computer.tq4_torch import dequantize_tq4, build_pi


GGUF_PATH = Path(os.environ.get(
    "ZENITH_GEMMA_GGUF",
    "/home/gabe/models/gemma-4-E4B-it-tq4-aligned.gguf",
))


def load_gemma_attention_fp32(reader, layer_idx: int, pi):
    """Load one Gemma layer's Q/K/V/O as FP32 tensors."""
    q = dequantize_tq4(extract_tq4_tensor(reader, f"blk.{layer_idx}.attn_q.weight"), pi=pi)
    k = dequantize_tq4(extract_tq4_tensor(reader, f"blk.{layer_idx}.attn_k.weight"), pi=pi)
    v = dequantize_tq4(extract_tq4_tensor(reader, f"blk.{layer_idx}.attn_v.weight"), pi=pi)
    o = dequantize_tq4(extract_tq4_tensor(reader, f"blk.{layer_idx}.attn_output.weight"), pi=pi)
    # GGUF (in, out) → PyTorch (out, in)
    return q.T.contiguous(), k.T.contiguous(), v.T.contiguous(), o.T.contiguous()


def decompose_to_sub_heads(W_q, n_heads=8, head_dim=256):
    """Decompose Q weight (q_out, d_model) into per-sub-head weights.

    Each head has head_dim values → head_dim/2 sub-heads at d_head=2.
    Returns (n_sub_heads, 2, d_model) — each sub-head's 2×d_model
    projection matrix.
    """
    # W_q shape: (n_heads * head_dim, d_model)
    d_model = W_q.shape[1]
    # Reshape to (n_heads, head_dim, d_model) then (n_heads, head_dim/2, 2, d_model)
    W_q = W_q.reshape(n_heads, head_dim, d_model)
    n_sub = head_dim // 2
    W_q = W_q.reshape(n_heads, n_sub, 2, d_model)
    # Flatten to (n_heads * n_sub, 2, d_model)
    return W_q.reshape(n_heads * n_sub, 2, d_model)


def run_attention_probe(W_q_sub, W_k_sub, W_v_sub, embeddings, mask):
    """Run d_head=2 per-sub-head attention and return per-sub-head
    activation norms.

    W_q_sub: (n_q_sub, 2, d_model)
    W_k_sub: (n_k_sub, 2, d_model)
    W_v_sub: (n_v_sub, 2, d_model)
    embeddings: (B, S, d_model)

    For GQA: n_k_sub < n_q_sub. Each KV sub-head is shared across
    n_q_sub / n_k_sub Q sub-heads (group ratio).

    Returns: per-Q-sub-head activation norm (n_q_sub,) averaged over batch.
    """
    B, S, D = embeddings.shape
    n_q_sub = W_q_sub.shape[0]
    n_k_sub = W_k_sub.shape[0]
    group_ratio = n_q_sub // n_k_sub

    # Q: (B, S, n_q_sub, 2) = embeddings @ W_q_sub.T per sub-head
    q = torch.einsum("bsd, nhd -> bsnh", embeddings, W_q_sub)  # (B, S, n_q_sub, 2)
    k = torch.einsum("bsd, nhd -> bsnh", embeddings, W_k_sub)  # (B, S, n_k_sub, 2)
    v = torch.einsum("bsd, nhd -> bsnh", embeddings, W_v_sub)  # (B, S, n_k_sub, 2)

    # Repeat K/V for GQA grouping
    k = k.repeat_interleave(group_ratio, dim=2)  # (B, S, n_q_sub, 2)
    v = v.repeat_interleave(group_ratio, dim=2)

    # Per-sub-head attention scores (B, n_q_sub, S_q, S_k)
    scores = torch.einsum("bqnh, bknh -> bnqk", q, k)
    scores = scores.masked_fill(mask.view(1, 1, S, S), float("-inf"))
    weights = F.softmax(scores, dim=-1)

    # Weighted V output: (B, n_q_sub, S, 2)
    attn_out = torch.einsum("bnqk, bknh -> bnqh", weights, v)

    # Per-sub-head activation norm: L2 norm over (S, 2), averaged over batch
    norms = attn_out.norm(dim=-1).mean(dim=(0, 2))  # (n_q_sub,)
    return norms


def main():
    if not GGUF_PATH.exists():
        print(f"GGUF not found at {GGUF_PATH}")
        return

    t0 = time.time()
    print(f"[probe] loading Gemma 4 E4B from {GGUF_PATH}...")
    reader = read_turboquant_gguf(GGUF_PATH)
    pi = build_pi(source="c_header")

    # Random embeddings as input stand-in (we don't have real Gemma tok
    # embed in fp32 without Q6_K dequant; use random for structure probe)
    d_model = 2560
    torch.manual_seed(42)

    # "Arithmetic" prompts: sequences with small-integer tokens (0..99)
    arith_emb = torch.randn(8, 4, d_model) * 0.02
    # "Language" prompts: sequences with large-vocab tokens (1000+)
    lang_emb = torch.randn(8, 4, d_model) * 0.02
    # Make arithmetic embeddings have structured patterns (digit-like)
    for i in range(8):
        arith_emb[i, 0, :10] = torch.arange(10).float() * 0.1  # digit features
        arith_emb[i, 1, :10] = torch.arange(10).float() * 0.05
    mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)

    for layer_idx in range(2):
        print(f"\n[probe] layer {layer_idx}: loading + decomposing...")
        t_l = time.time()
        W_q, W_k, W_v, W_o = load_gemma_attention_fp32(reader, layer_idx, pi)
        print(f"  Q: {tuple(W_q.shape)}, K: {tuple(W_k.shape)}, "
              f"V: {tuple(W_v.shape)}, O: {tuple(W_o.shape)}")

        # Decompose to d_head=2 sub-heads
        head_dim = W_q.shape[0] // 8  # n_heads=8
        kv_head_dim = W_k.shape[0] // 2  # n_kv_heads=2
        W_q_sub = decompose_to_sub_heads(W_q, n_heads=8, head_dim=head_dim)
        W_k_sub = decompose_to_sub_heads(W_k, n_heads=2, head_dim=kv_head_dim)
        W_v_sub = decompose_to_sub_heads(W_v, n_heads=2, head_dim=kv_head_dim)
        print(f"  sub-heads: Q={W_q_sub.shape[0]}, K={W_k_sub.shape[0]}, "
              f"V={W_v_sub.shape[0]}")

        # Run attention probe on both prompt types
        norms_arith = run_attention_probe(
            W_q_sub, W_k_sub, W_v_sub, arith_emb, mask,
        )
        norms_lang = run_attention_probe(
            W_q_sub, W_k_sub, W_v_sub, lang_emb, mask,
        )

        # Identify sub-heads with HIGH arithmetic activation relative
        # to language activation (arithmetic-specialist candidates)
        ratio = norms_arith / (norms_lang + 1e-8)
        top_k = 10
        top_indices = ratio.topk(top_k).indices
        print(f"\n  top {top_k} arithmetic-selective sub-heads (arith/lang ratio):")
        for rank, idx in enumerate(top_indices):
            i = idx.item()
            head = i // (head_dim // 2)
            sub = i % (head_dim // 2)
            print(f"    #{rank+1}: sub-head {i} (head {head}, sub {sub}) "
                  f"arith={norms_arith[i]:.4f} lang={norms_lang[i]:.4f} "
                  f"ratio={ratio[i]:.2f}")

        # Overall statistics
        print(f"\n  all sub-heads: arith mean={norms_arith.mean():.4f} "
              f"std={norms_arith.std():.4f}")
        print(f"  all sub-heads: lang  mean={norms_lang.mean():.4f} "
              f"std={norms_lang.std():.4f}")
        print(f"  correlation arith↔lang: "
              f"{torch.corrcoef(torch.stack([norms_arith, norms_lang]))[0, 1]:.4f}")
        print(f"  layer time: {time.time() - t_l:.1f}s")

    print(f"\n[probe] total time: {time.time() - t0:.1f}s")
    print("\n[probe] interpretation:")
    print("  high ratio + high arith norm = sub-heads that activate")
    print("  strongly on arithmetic patterns but not on general language.")
    print("  these are candidates for compiled-card replacement.")
    print("  (note: this probe uses random embeddings, not real Gemma tok;")
    print("  real Q6_K tok embed would give more meaningful selectivity.)")


if __name__ == "__main__":
    main()
