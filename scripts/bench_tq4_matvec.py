"""Microbench for the tq4 matvec Triton kernel.

Runs `tq4_matvec_triton` on the canonical Gemma 4 E4B attn shape and
reports throughput. Used as the baseline for kernel optimization
rounds (R53.29+ porting TurboQuant's shared-mem LUT + fp16 activation
+ vector-load techniques).

Run: python3 scripts/bench_tq4_matvec.py
"""

from __future__ import annotations

import time

import torch

from calm.llm_computer.tq4_torch import (
    build_pi, compute_lloyd_max_codebook, quantize_tq4,
)
from calm.llm_computer.tq4_triton import tq4_matvec_triton


def bench_shape(in_features: int, out_features: int,
                iters: int = 500, warmup: int = 50) -> dict:
    device = "cuda"
    torch.manual_seed(0)
    # Random weights — dequant quality unimportant for perf bench
    W = torch.randn(out_features, in_features, device=device, dtype=torch.float32) * 0.05
    pi = build_pi(device=device, source="torch")
    _, boundaries = compute_lloyd_max_codebook()
    boundaries = boundaries.to(device)
    centroids, _ = compute_lloyd_max_codebook()
    centroids = centroids.to(device)

    # Quantize row-by-row (tq4 quantize is 1D over a flat block multiple)
    qs_rows = []
    d_rows = []
    for r in range(out_features):
        q = quantize_tq4(W[r], pi=pi, boundaries=boundaries)
        qs_rows.append(q.qs)  # (bpr, 128)
        d_rows.append(q.d)    # (bpr,)
    qs = torch.stack(qs_rows, dim=0).reshape(-1, 128).contiguous()
    d = torch.stack(d_rows, dim=0).reshape(-1).contiguous()

    # Input — Pi is 256×256 applied per-256-block
    x = torch.randn(in_features, device=device, dtype=torch.float32)
    bpr = in_features // 256
    x_rot = (x.reshape(bpr, 256) @ pi.T).reshape(in_features).contiguous()

    # Warmup
    for _ in range(warmup):
        y = tq4_matvec_triton(x_rot, qs, d, centroids, out_features, in_features)
    torch.cuda.synchronize()

    # Measure
    t0 = time.time()
    for _ in range(iters):
        y = tq4_matvec_triton(x_rot, qs, d, centroids, out_features, in_features)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    us_per_call = (elapsed / iters) * 1e6
    flops = 2 * out_features * in_features
    gflops = (flops * iters) / elapsed / 1e9
    bytes_read = qs.numel() + d.numel() * 4 + x_rot.numel() * 4
    gbps = (bytes_read * iters) / elapsed / 1e9
    return {
        "shape": (in_features, out_features),
        "us_per_call": us_per_call,
        "gflops": gflops,
        "gbps": gbps,
    }


def main():
    shapes = [
        # Gemma 4 E4B attn/ffn shapes
        (2560, 2048),   # attn_q (Q projection, 8 heads × 256)
        (2560, 512),    # attn_k / attn_v (2 kv heads × 256)
        (2048, 2560),   # attn_output
        (2560, 10240),  # ffn_gate / ffn_up
        (10240, 2560),  # ffn_down
    ]

    print(f"{'shape':<18} {'us/call':>10} {'GFLOP/s':>10} {'GB/s':>10}")
    print("-" * 52)
    results = []
    for ifeat, ofeat in shapes:
        r = bench_shape(ifeat, ofeat)
        results.append(r)
        print(f"{str(r['shape']):<18} {r['us_per_call']:>10.1f} "
              f"{r['gflops']:>10.1f} {r['gbps']:>10.1f}")
    # Overall "per forward step" proxy: sum times × n_layers × 2 attn + ...
    # Keep simple aggregate.
    total_us = sum(r["us_per_call"] for r in results)
    print("-" * 52)
    print(f"{'sum':<18} {total_us:>10.1f}")


if __name__ == "__main__":
    main()
