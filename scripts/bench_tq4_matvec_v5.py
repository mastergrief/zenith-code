"""Bench v5 (int8 path) vs v2 (fp32 path, current default).

Both use pre-prepared activation (fair comparison — rotation and
quantization happen outside the measured kernel). In production,
x_rot is prepped once per layer and reused across Q/K/V/output or
gate/up, so the activation prep amortizes.

Uses the same heavy_warmup + CUDA events + median-of-5 methodology
as bench_tq4_matvec.py (mandatory per workflow.md §"GPU bench
discipline").
"""

from __future__ import annotations

import statistics
import time

import torch

from calm.llm_computer.tq4_torch import (
    build_pi, compute_lloyd_max_codebook, quantize_tq4,
)
from calm.llm_computer.tq4_triton import (
    tq4_matvec_triton_v2, tq4_matvec_triton_v5_prequant,
    _quantize_activation_q8, _prep_centroids_i8,
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

    qs_rows, d_rows = [], []
    for r in range(out_features):
        q = quantize_tq4(W[r], pi=pi, boundaries=boundaries)
        qs_rows.append(q.qs)
        d_rows.append(q.d)
    qs = torch.stack(qs_rows, dim=0).reshape(-1, 128).contiguous()
    d = torch.stack(d_rows, dim=0).reshape(-1).contiguous()

    x = torch.randn(in_features, device=device, dtype=torch.float32)
    bpr = in_features // 256
    x_rot = (x.reshape(bpr, 256) @ pi.T).reshape(in_features).contiguous()

    # v5 prep
    x_q8, x_scale = _quantize_activation_q8(x_rot, bpr)
    c_i8, c_rescale = _prep_centroids_i8(centroids)
    d_fused = (d * c_rescale).contiguous()

    return {
        "x_rot": x_rot,
        "qs": qs, "d": d, "centroids": centroids,
        "x_q8": x_q8, "x_scale": x_scale,
        "d_fused": d_fused, "c_i8": c_i8,
        "in_features": in_features, "out_features": out_features,
    }


def time_v2(shape_data, iters: int = 2000) -> float:
    x_rot = shape_data["x_rot"]
    qs = shape_data["qs"]
    d = shape_data["d"]
    centroids = shape_data["centroids"]
    out_f = shape_data["out_features"]
    in_f = shape_data["in_features"]

    for _ in range(100):
        _ = tq4_matvec_triton_v2(x_rot, qs, d, centroids, out_f, in_f)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = tq4_matvec_triton_v2(x_rot, qs, d, centroids, out_f, in_f)
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) * 1000.0) / iters


def time_v5(shape_data, iters: int = 2000) -> float:
    x_q8 = shape_data["x_q8"]
    x_scale = shape_data["x_scale"]
    qs = shape_data["qs"]
    d_fused = shape_data["d_fused"]
    c_i8 = shape_data["c_i8"]
    out_f = shape_data["out_features"]
    in_f = shape_data["in_features"]

    for _ in range(100):
        _ = tq4_matvec_triton_v5_prequant(
            x_q8, x_scale, qs, d_fused, c_i8, out_f, in_f)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = tq4_matvec_triton_v5_prequant(
            x_q8, x_scale, qs, d_fused, c_i8, out_f, in_f)
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) * 1000.0) / iters


def heavy_warmup(seconds: float = 3.0) -> None:
    device = "cuda"
    A = torch.randn(2048, 2048, device=device, dtype=torch.float16)
    B = torch.randn(2048, 2048, device=device, dtype=torch.float16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        _ = A @ B
    torch.cuda.synchronize()


def measure(shape_data_dict, time_fn, n_runs=5, iters=2000, label=""):
    results = {}
    for shape in SHAPES:
        times = [time_fn(shape_data_dict[shape], iters) for _ in range(n_runs)]
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
        bytes_read = out_f * (in_f // 256) * 128 + out_f * (in_f // 256) * 4 + in_f
        gbps = bytes_read / (us * 1e-6) / 1e9
        print(f"{str(shape):<18} {us:>10.1f} {gflops:>10.1f} {gbps:>10.1f}")
        total_us += us
    print("-" * 52)
    print(f"{'sum':<18} {total_us:>10.1f}")
    return results


def main():
    heavy_warmup(3.0)
    shape_data = {
        (ifeat, ofeat): prepare_shape(ifeat, ofeat)
        for ifeat, ofeat in SHAPES
    }

    # Warm up both paths' JIT
    for shape in SHAPES:
        _ = time_v2(shape_data[shape], iters=50)
        _ = time_v5(shape_data[shape], iters=50)
    torch.cuda.synchronize()

    res_v2 = measure(shape_data, time_v2, label="v2 (fp32 path — current default)")
    res_v5 = measure(shape_data, time_v5, label="v5 (int8 path, pre-quantized)")

    print("\n=== delta (v5 vs v2) ===")
    print(f"{'shape':<18} {'v2 us':>10} {'v5 us':>10} {'Δ%':>10}")
    print("-" * 52)
    sum_v2 = sum_v5 = 0.0
    for shape in SHAPES:
        a, b = res_v2[shape], res_v5[shape]
        pct = (b - a) / a * 100
        sum_v2 += a
        sum_v5 += b
        print(f"{str(shape):<18} {a:>10.1f} {b:>10.1f} {pct:>+9.1f}%")
    print("-" * 52)
    total_pct = (sum_v5 - sum_v2) / sum_v2 * 100
    print(f"{'sum':<18} {sum_v2:>10.1f} {sum_v5:>10.1f} {total_pct:>+9.1f}%")


if __name__ == "__main__":
    main()
