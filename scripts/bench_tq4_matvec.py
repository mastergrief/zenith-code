"""Microbench for the tq4 matvec Triton kernel — noise-stable version.

Runs `tq4_matvec_triton` on the canonical Gemma 4 E4B attn shape and
reports throughput. Uses CUDA events + heavy pre-warmup to stabilize
GPU clocks, takes median of N runs.

Used as the baseline for kernel optimization rounds (R53.29+ porting
TurboQuant's shared-mem LUT + fp16 activation + vector-load techniques).

Run: python3 scripts/bench_tq4_matvec.py
"""

from __future__ import annotations

import statistics
import time

import torch

from calm.llm_computer.tq4_torch import (
    build_pi, compute_lloyd_max_codebook, quantize_tq4,
)
from calm.llm_computer.tq4_triton import (
    tq4_matvec_triton_v1, tq4_matvec_triton_v2,
)


SHAPES = [
    (2560, 2048),   # attn_q
    (2560, 512),    # attn_k / attn_v
    (2048, 2560),   # attn_output
    (2560, 10240),  # ffn_gate / ffn_up
    (10240, 2560),  # ffn_down
]


def prepare_shape(in_features: int, out_features: int):
    device = "cuda"
    torch.manual_seed(0)
    W = torch.randn(out_features, in_features, device=device, dtype=torch.float32) * 0.05
    pi = build_pi(device=device, source="torch")
    centroids, boundaries = compute_lloyd_max_codebook()
    boundaries = boundaries.to(device)
    centroids = centroids.to(device)

    qs_rows = []
    d_rows = []
    for r in range(out_features):
        q = quantize_tq4(W[r], pi=pi, boundaries=boundaries)
        qs_rows.append(q.qs)
        d_rows.append(q.d)
    qs = torch.stack(qs_rows, dim=0).reshape(-1, 128).contiguous()
    d = torch.stack(d_rows, dim=0).reshape(-1).contiguous()

    x = torch.randn(in_features, device=device, dtype=torch.float32)
    bpr = in_features // 256
    x_rot = (x.reshape(bpr, 256) @ pi.T).reshape(in_features).contiguous()

    return {
        "x_rot": x_rot, "qs": qs, "d": d, "centroids": centroids,
        "in_features": in_features, "out_features": out_features,
    }


def time_shape_events(shape_data, kernel_fn, iters: int = 2000) -> float:
    """Returns us/call using CUDA events."""
    x_rot = shape_data["x_rot"]
    qs = shape_data["qs"]
    d = shape_data["d"]
    centroids = shape_data["centroids"]
    out_f = shape_data["out_features"]
    in_f = shape_data["in_features"]

    for _ in range(100):
        _ = kernel_fn(x_rot, qs, d, centroids, out_f, in_f)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = kernel_fn(x_rot, qs, d, centroids, out_f, in_f)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    return (ms * 1000.0) / iters


def heavy_warmup(seconds: float = 3.0) -> None:
    """Push GPU to steady-state clock before measuring."""
    device = "cuda"
    A = torch.randn(2048, 2048, device=device, dtype=torch.float16)
    B = torch.randn(2048, 2048, device=device, dtype=torch.float16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        C = A @ B
    torch.cuda.synchronize()


def measure(shape_data_dict: dict, kernel_fn=tq4_matvec_triton_v1,
            n_runs: int = 5, iters: int = 2000,
            label: str = "") -> dict[tuple, float]:
    results: dict[tuple, float] = {}
    for shape in SHAPES:
        times = []
        for _ in range(n_runs):
            times.append(time_shape_events(
                shape_data_dict[shape], kernel_fn, iters=iters))
        results[shape] = statistics.median(times)

    if label:
        print(f"\n=== {label} ===")
    print(f"{'shape':<18} {'us/call':>10} {'GFLOP/s':>10} {'GB/s':>10}")
    print("-" * 52)
    total_us = 0.0
    for shape in SHAPES:
        us = results[shape]
        in_f, out_f = shape
        flops = 2 * in_f * out_f
        gflops = flops / (us * 1e-6) / 1e9
        bytes_read = (out_f * (in_f // 256) * 128
                       + out_f * (in_f // 256) * 4
                       + in_f * 4)
        gbps = bytes_read / (us * 1e-6) / 1e9
        print(f"{str(shape):<18} {us:>10.1f} {gflops:>10.1f} {gbps:>10.1f}")
        total_us += us
    print("-" * 52)
    print(f"{'sum':<18} {total_us:>10.1f}")
    return results


def bench_ab(label_a: str = "baseline (tl.load gather)",
             label_b: str = "v2 (tl.gather from tile)") -> None:
    heavy_warmup(3.0)
    shape_data = {
        (ifeat, ofeat): prepare_shape(ifeat, ofeat)
        for ifeat, ofeat in SHAPES
    }
    # Run both variants interleaved (pair A/B per shape) to minimize
    # clock-drift bias between runs.
    res_a = measure(shape_data, kernel_fn=tq4_matvec_triton_v1, label=label_a)
    res_b = measure(shape_data, kernel_fn=tq4_matvec_triton_v2, label=label_b)

    print(f"\n=== delta ({label_b} vs {label_a}) ===")
    print(f"{'shape':<18} {'A us':>10} {'B us':>10} {'Δ%':>10}")
    print("-" * 52)
    sum_a = sum_b = 0.0
    for shape in SHAPES:
        a, b = res_a[shape], res_b[shape]
        pct = (b - a) / a * 100
        sum_a += a
        sum_b += b
        print(f"{str(shape):<18} {a:>10.1f} {b:>10.1f} {pct:>+9.1f}%")
    print("-" * 52)
    total_pct = (sum_b - sum_a) / sum_a * 100
    print(f"{'sum':<18} {sum_a:>10.1f} {sum_b:>10.1f} {total_pct:>+9.1f}%")


def main():
    bench_ab()


if __name__ == "__main__":
    main()
