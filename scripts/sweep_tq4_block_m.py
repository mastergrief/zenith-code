"""Sweep BLOCK_M for the tq4 v2 kernel across Gemma 4 E4B shapes."""

from __future__ import annotations

import statistics
import time

import torch

from calm.llm_computer.tq4_torch import (
    build_pi, compute_lloyd_max_codebook, quantize_tq4,
)
from calm.llm_computer.tq4_triton import _tq4_matvec_kernel_v2


SHAPES = [
    (2560, 2048),
    (2560, 512),
    (2048, 2560),
    (2560, 10240),
    (10240, 2560),
]

BLOCK_M_CANDIDATES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def prepare_shape(in_f: int, out_f: int) -> dict:
    device = "cuda"
    torch.manual_seed(0)
    W = torch.randn(out_f, in_f, device=device, dtype=torch.float32) * 0.05
    pi = build_pi(device=device, source="torch")
    centroids, boundaries = compute_lloyd_max_codebook()
    boundaries = boundaries.to(device)
    centroids = centroids.to(device)
    qs_rows, d_rows = [], []
    for r in range(out_f):
        q = quantize_tq4(W[r], pi=pi, boundaries=boundaries)
        qs_rows.append(q.qs)
        d_rows.append(q.d)
    qs = torch.stack(qs_rows, dim=0).reshape(-1, 128).contiguous()
    d = torch.stack(d_rows, dim=0).reshape(-1).contiguous()
    x = torch.randn(in_f, device=device, dtype=torch.float32)
    bpr = in_f // 256
    x_rot = (x.reshape(bpr, 256) @ pi.T).reshape(in_f).contiguous()
    return {
        "x_rot": x_rot, "qs": qs, "d": d, "centroids": centroids,
        "in_features": in_f, "out_features": out_f,
    }


def call_v2(data, block_m: int):
    x_rot = data["x_rot"]; qs = data["qs"]; d = data["d"]
    centroids = data["centroids"]
    out_f = data["out_features"]; in_f = data["in_features"]
    bpr = in_f // 256
    y = torch.empty(out_f, device=x_rot.device, dtype=torch.float32)
    grid = ((out_f + block_m - 1) // block_m,)
    _tq4_matvec_kernel_v2[grid](
        x_rot, qs.view(-1), d, centroids, y,
        in_f, out_f, BPR=bpr, BLOCK_HALF=128, BLOCK_M=block_m,
        num_warps=4,
    )
    return y


def time_shape(data, block_m: int, iters: int = 1000) -> float:
    try:
        for _ in range(50):
            _ = call_v2(data, block_m)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _ = call_v2(data, block_m)
        end.record()
        torch.cuda.synchronize()
        return (start.elapsed_time(end) * 1000) / iters
    except Exception as e:
        return float("inf")


def heavy_warmup(seconds: float = 3.0):
    A = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    B = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        C = A @ B
    torch.cuda.synchronize()


def main():
    heavy_warmup(3.0)
    shape_data = {s: prepare_shape(*s) for s in SHAPES}

    results = {}
    for shape in SHAPES:
        data = shape_data[shape]
        print(f"\n=== shape {shape} (BLOCK_M sweep) ===")
        row = {}
        for bm in BLOCK_M_CANDIDATES:
            if bm > shape[1]:
                continue
            times = [time_shape(data, bm) for _ in range(3)]
            med = statistics.median(times)
            row[bm] = med
            print(f"  BLOCK_M={bm:>4}: {med:>7.1f} us")
        best_bm = min(row, key=row.get)
        print(f"  BEST: BLOCK_M={best_bm} @ {row[best_bm]:.1f} us")
        results[shape] = (best_bm, row[best_bm])

    print("\n=== Summary: best BLOCK_M per shape ===")
    for shape, (bm, us) in results.items():
        print(f"  {shape}: BLOCK_M={bm} @ {us:.1f} us")


if __name__ == "__main__":
    main()
