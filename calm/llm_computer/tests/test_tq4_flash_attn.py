"""Unit tests for fused tq4 flash-attn decode kernel.

Synthetic K/V data — no Gemma required. Compares fused_tq4_flash_attn_decode
against a fp32 reference (softmax(Q @ K.T) @ V) on tq4-roundtripped K/V
to isolate kernel error from quantization noise.

Tolerance: cosine ≥ 0.998 per head. Cosine 0.999+ achievable on most
configs but tq4 Lloyd-Max boundary effects can push borderline cases
slightly lower at small N.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_heads_q,n_heads_kv,d_head,N", [
    (4, 4, 256, 16),     # MHA, short
    (4, 4, 256, 256),    # MHA, medium
    (8, 4, 256, 64),     # GQA 2x
    (8, 2, 256, 128),    # GQA 4x (Gemma E4B SWA)
    (4, 4, 256, 1024),   # longer
])
def test_fused_tq4_flash_attn_decode_matches_fp32(
        n_heads_q, n_heads_kv, d_head, N):
    from calm.llm_computer.tq4_flash_attn import fused_tq4_flash_attn_decode
    from calm.llm_computer.tq4_torch import (
        Tq4Tensor, build_pi, compute_lloyd_max_codebook,
        dequantize_tq4, quantize_tq4,
    )

    torch.manual_seed(0)
    device = "cuda"
    pi = build_pi(device=device)
    centroids, boundaries = compute_lloyd_max_codebook()
    centroids = centroids.to(device)
    boundaries = boundaries.to(device)

    # Random K/V/Q. Scale K/V down so softmax has a meaningful spread
    # (giant scores → degenerate softmax → trivial reference).
    k = torch.randn(n_heads_kv, N, d_head, device=device) * 0.1
    v = torch.randn(n_heads_kv, N, d_head, device=device) * 0.1
    q = torch.randn(n_heads_q, d_head, device=device) * 0.1

    # Quantize K and V per-head into tq4 head-major layout.
    bpr = d_head // 256
    k_qs = torch.empty(n_heads_kv, N * bpr, 128,
                       dtype=torch.uint8, device=device)
    k_d = torch.empty(n_heads_kv, N * bpr, dtype=torch.float32, device=device)
    v_qs = torch.empty_like(k_qs)
    v_d = torch.empty_like(k_d)
    # Reference uses tq4-roundtripped K/V to isolate kernel error from
    # quantization noise.
    k_dq = torch.empty_like(k)
    v_dq = torch.empty_like(v)
    for h in range(n_heads_kv):
        kq = quantize_tq4(k[h].reshape(-1), pi=pi, boundaries=boundaries)
        vq = quantize_tq4(v[h].reshape(-1), pi=pi, boundaries=boundaries)
        k_qs[h] = kq.qs
        k_d[h] = kq.d
        v_qs[h] = vq.qs
        v_d[h] = vq.d
        k_dq[h] = dequantize_tq4(kq, pi=pi, centroids=centroids).reshape(N, d_head)
        v_dq[h] = dequantize_tq4(vq, pi=pi, centroids=centroids).reshape(N, d_head)

    # Pre-rotate Q outside kernel (caller responsibility).
    q_rot = q @ pi.T

    # Causal-style mask: fully unmasked for this test (decode = attend to all).
    attn_mask = torch.zeros(N, device=device, dtype=torch.float32)

    # Fused kernel result.
    out_fused = fused_tq4_flash_attn_decode(
        q_rot, k_qs, k_d, v_qs, v_d, centroids, pi, attn_mask)

    # Reference: standard fp32 attention on tq4-roundtripped K/V.
    gqa_repeat = n_heads_q // n_heads_kv
    k_full = k_dq.repeat_interleave(gqa_repeat, dim=0)  # (n_heads_q, N, d_head)
    v_full = v_dq.repeat_interleave(gqa_repeat, dim=0)
    scores_ref = torch.einsum("hd,hnd->hn", q, k_full)
    weights_ref = torch.softmax(scores_ref, dim=-1)
    out_ref = torch.einsum("hn,hnd->hd", weights_ref, v_full)

    # Per-head cosine.
    cosines = []
    for h in range(n_heads_q):
        cos = F.cosine_similarity(out_fused[h], out_ref[h], dim=0).item()
        cosines.append(cos)

    min_cos = min(cosines)
    print(f"\n  N={N} GQA={gqa_repeat}x cos min={min_cos:.5f} "
          f"mean={sum(cosines)/len(cosines):.5f}")
    assert min_cos >= 0.998, (
        f"per-head cosine fell to {min_cos:.5f} < 0.998; "
        f"all: {[f'{c:.4f}' for c in cosines]}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_tq4_flash_attn_decode_with_mask():
    """Causal mask: only first N/2 positions attended. Reference must
    match exactly the unmasked-prefix attention."""
    from calm.llm_computer.tq4_flash_attn import fused_tq4_flash_attn_decode
    from calm.llm_computer.tq4_torch import (
        build_pi, compute_lloyd_max_codebook, dequantize_tq4, quantize_tq4,
    )

    torch.manual_seed(1)
    device = "cuda"
    pi = build_pi(device=device)
    centroids, boundaries = compute_lloyd_max_codebook()
    centroids = centroids.to(device)
    boundaries = boundaries.to(device)

    n_heads_q, n_heads_kv, d_head, N = 4, 4, 256, 64
    k = torch.randn(n_heads_kv, N, d_head, device=device) * 0.1
    v = torch.randn(n_heads_kv, N, d_head, device=device) * 0.1
    q = torch.randn(n_heads_q, d_head, device=device) * 0.1

    k_qs = torch.empty(n_heads_kv, N, 128, dtype=torch.uint8, device=device)
    k_d = torch.empty(n_heads_kv, N, dtype=torch.float32, device=device)
    v_qs = torch.empty_like(k_qs)
    v_d = torch.empty_like(k_d)
    k_dq = torch.empty_like(k)
    v_dq = torch.empty_like(v)
    for h in range(n_heads_kv):
        kq = quantize_tq4(k[h].reshape(-1), pi=pi, boundaries=boundaries)
        vq = quantize_tq4(v[h].reshape(-1), pi=pi, boundaries=boundaries)
        k_qs[h], k_d[h] = kq.qs, kq.d
        v_qs[h], v_d[h] = vq.qs, vq.d
        k_dq[h] = dequantize_tq4(kq, pi=pi, centroids=centroids).reshape(N, d_head)
        v_dq[h] = dequantize_tq4(vq, pi=pi, centroids=centroids).reshape(N, d_head)

    q_rot = q @ pi.T

    # Mask second half: -inf
    attn_mask = torch.zeros(N, device=device, dtype=torch.float32)
    attn_mask[N // 2:] = float("-inf")

    out_fused = fused_tq4_flash_attn_decode(
        q_rot, k_qs, k_d, v_qs, v_d, centroids, pi, attn_mask)

    # Reference: attend only to first N/2 positions.
    k_prefix = k_dq[:, :N // 2, :]
    v_prefix = v_dq[:, :N // 2, :]
    scores_ref = torch.einsum("hd,hnd->hn", q, k_prefix)
    weights_ref = torch.softmax(scores_ref, dim=-1)
    out_ref = torch.einsum("hn,hnd->hd", weights_ref, v_prefix)

    for h in range(n_heads_q):
        cos = F.cosine_similarity(out_fused[h], out_ref[h], dim=0).item()
        assert cos >= 0.998, f"head {h} cos={cos:.5f}"


@pytest.mark.parametrize("d_head", [256])
def test_tq4_weighted_v_smoke(d_head):
    """CPU import smoke: kernel module loads, wrapper signature is correct.
    Actual kernel exec needs CUDA; the Triton @jit shouldn't error at import."""
    from calm.llm_computer.tq4_flash_attn import (
        fused_tq4_flash_attn_decode, tq4_weighted_v,
    )
    assert callable(fused_tq4_flash_attn_decode)
    assert callable(tq4_weighted_v)
