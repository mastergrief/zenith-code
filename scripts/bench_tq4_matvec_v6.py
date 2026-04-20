"""Bench v6 (int8 tl.dot tensor-core path) vs v2 (fp32, current default).

v6 engages Ada sm_89 int8 tensor cores via tl.dot(int8, int8, out=int32).
N=1 padded to N=16 (tensor core minimum), BLOCK_M ≥ 16 required.
Dispatcher falls back to v2 when BLOCK_M < 16 (attn_k/v shape).

Same heavy_warmup + CUDA events + median-of-5 methodology as
bench_tq4_matvec.py and bench_tq4_matvec_v5.py.
"""

from __future__ import annotations

import statistics
import time

import torch

from calm.llm_computer.tq4_torch import (
    build_pi, compute_lloyd_max_codebook, quantize_tq4,
)
from calm.llm_computer.tq4_triton import (
    tq4_matvec_triton_v2, tq4_matvec_triton_v6_prequant,
    _quantize_activation_q8, _prep_centroids_i8, _pick_block_m,
)


SHAPES = [
    (2560, 2048),   # attn_q, BLOCK_M=32 → tensor core
    (2560, 512),    # attn_k/v, BLOCK_M=4 → v2 fallback
    (2048, 2560),   # attn_output, BLOCK_M=32 → tensor core
    (2560, 10240),  # ffn_gate/up, BLOCK_M=64 → tensor core
    (10240, 2560),  # ffn_down, BLOCK_M=32 → tensor core
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

    x_q8, x_scale = _quantize_activation_q8(x_rot, bpr)
    c_i8, c_rescale = _prep_centroids_i8(centroids)
    d_fused = (d * c_rescale).contiguous()

    return {
        "x_rot": x_rot, "qs": qs, "d": d, "centroids": centroids,
        "x_q8": x_q8, "x_scale": x_scale,
        "d_fused": d_fused, "c_i8": c_i8,
        "in_features": in_features, "out_features": out_features,
    }


def time_v2(shape_data, iters: int = 2000) -> float:
    x_rot = shape_data["x_rot"]
    qs, d = shape_data["qs"], shape_data["d"]
    centroids = shape_data["centroids"]
    out_f = shape_data["out_features"]; in_f = shape_data["in_features"]

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


def time_v6(shape_data, iters: int = 2000) -> float:
    """Use v6_prequant if BLOCK_M≥16, else fall back to v2 for fair
    comparison (same fallback logic as the dispatcher)."""
    out_f = shape_data["out_features"]; in_f = shape_data["in_features"]
    if _pick_block_m(out_f) < 16:
        return time_v2(shape_data, iters)

    x_q8, x_scale = shape_data["x_q8"], shape_data["x_scale"]
    qs = shape_data["qs"]
    d_fused = shape_data["d_fused"]; c_i8 = shape_data["c_i8"]

    for _ in range(100):
        _ = tq4_matvec_triton_v6_prequant(
            x_q8, x_scale, qs, d_fused, c_i8, out_f, in_f)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = tq4_matvec_triton_v6_prequant(
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
    print(f"{'shape':<18} {'BLOCK_M':>8} {'us/call':>10}")
    print("-" * 40)
    total_us = 0.0
    for shape in SHAPES:
        us = results[shape]
        bm = _pick_block_m(shape[1])
        print(f"{str(shape):<18} {bm:>8} {us:>10.1f}")
        total_us += us
    print("-" * 40)
    print(f"{'sum':<18} {'':>8} {total_us:>10.1f}")
    return results


def main():
    heavy_warmup(3.0)
    shape_data = {
        (ifeat, ofeat): prepare_shape(ifeat, ofeat)
        for ifeat, ofeat in SHAPES
    }

    # JIT warm
    for shape in SHAPES:
        _ = time_v2(shape_data[shape], iters=50)
        _ = time_v6(shape_data[shape], iters=50)
    torch.cuda.synchronize()

    res_v2 = measure(shape_data, time_v2, label="v2 (fp32 path — current default)")
    res_v6 = measure(shape_data, time_v6, label="v6 (int8 tl.dot tensor-core, v2 fallback <16)")

    print("\n=== delta (v6 vs v2) ===")
    print(f"{'shape':<18} {'BLOCK_M':>8} {'v2 us':>10} {'v6 us':>10} {'Δ%':>10}")
    print("-" * 60)
    sum_v2 = sum_v6 = 0.0
    for shape in SHAPES:
        a, b = res_v2[shape], res_v6[shape]
        bm = _pick_block_m(shape[1])
        pct = (b - a) / a * 100
        sum_v2 += a; sum_v6 += b
        tag = "[tc]" if bm >= 16 else "[fb]"
        print(f"{str(shape):<18} {bm:>6}{tag} {a:>10.1f} {b:>10.1f} {pct:>+9.1f}%")
    print("-" * 60)
    total_pct = (sum_v6 - sum_v2) / sum_v2 * 100
    print(f"{'sum':<18} {'':>8} {sum_v2:>10.1f} {sum_v6:>10.1f} {total_pct:>+9.1f}%")


if __name__ == "__main__":
    main()
