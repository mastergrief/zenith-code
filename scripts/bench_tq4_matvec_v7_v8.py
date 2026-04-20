"""Bench v7 (widen-K-by-2) and v8 (acc reuse) vs v2 and v6.

Both v7 and v8 use LOSSY per-block scale approximations (pair-mean
for v7, full-block mean for v8). Correctness check first — if cosine
< 0.95, declare "can't fix without precision rework" and skip bench.
If cosine OK, bench to see if structural change wins perf.

Same heavy_warmup + CUDA events + median-of-5 methodology.
"""

from __future__ import annotations

import statistics
import time

import torch

from calm.llm_computer.tq4_torch import (
    build_pi, compute_lloyd_max_codebook, quantize_tq4,
)
from calm.llm_computer.tq4_triton import (
    tq4_matvec_triton_v2,
    tq4_matvec_triton_v6_prequant,
    tq4_matvec_triton_v7_prequant,
    tq4_matvec_triton_v8_prequant,
    _quantize_activation_q8, _prep_centroids_i8, _pick_block_m,
)


SHAPES = [
    (2560, 2048),
    (2560, 512),
    (2048, 2560),
    (2560, 10240),
    (10240, 2560),
]


def prepare_shape(in_f, out_f):
    device = "cuda"
    torch.manual_seed(0)
    W = torch.randn(out_f, in_f, device=device, dtype=torch.float32) * 0.05
    pi = build_pi(device=device, source="torch")
    centroids, boundaries = compute_lloyd_max_codebook()
    boundaries = boundaries.to(device); centroids = centroids.to(device)
    qs_rows, d_rows = [], []
    for r in range(out_f):
        q = quantize_tq4(W[r], pi=pi, boundaries=boundaries)
        qs_rows.append(q.qs); d_rows.append(q.d)
    qs = torch.stack(qs_rows).reshape(-1, 128).contiguous()
    d = torch.stack(d_rows).reshape(-1).contiguous()
    x = torch.randn(in_f, device=device)
    bpr = in_f // 256
    x_rot = (x.reshape(bpr, 256) @ pi.T).reshape(in_f).contiguous()
    x_q8, x_scale = _quantize_activation_q8(x_rot, bpr)
    c_i8, c_rescale = _prep_centroids_i8(centroids)
    d_fused = (d * c_rescale).contiguous()
    return {"x_rot": x_rot, "qs": qs, "d": d, "centroids": centroids,
            "x_q8": x_q8, "x_scale": x_scale,
            "d_fused": d_fused, "c_i8": c_i8,
            "in_features": in_f, "out_features": out_f}


def check_correctness(sd):
    out_f = sd["out_features"]; in_f = sd["in_features"]
    y_ref = tq4_matvec_triton_v2(sd["x_rot"], sd["qs"], sd["d"],
                                   sd["centroids"], out_f, in_f)
    results = {}
    for name, fn in [
        ("v6", lambda: tq4_matvec_triton_v6_prequant(
            sd["x_q8"], sd["x_scale"], sd["qs"],
            sd["d_fused"], sd["c_i8"], out_f, in_f)),
        ("v7", lambda: tq4_matvec_triton_v7_prequant(
            sd["x_q8"], sd["x_scale"], sd["qs"],
            sd["d_fused"], sd["c_i8"], out_f, in_f)),
        ("v8", lambda: tq4_matvec_triton_v8_prequant(
            sd["x_q8"], sd["x_scale"], sd["qs"],
            sd["d_fused"], sd["c_i8"], out_f, in_f)),
    ]:
        try:
            y = fn()
            cos = torch.nn.functional.cosine_similarity(
                y_ref.unsqueeze(0), y.unsqueeze(0)).item()
            rel = (y_ref - y).abs().max().item() / (y_ref.abs().max().item() + 1e-9)
            results[name] = (cos, rel)
        except Exception as e:
            results[name] = (None, str(e)[:80])
    return results


def time_fn(fn, iters=2000):
    for _ in range(100):
        _ = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = fn()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) * 1000.0) / iters


def heavy_warmup(seconds=3.0):
    A = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    B = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        _ = A @ B
    torch.cuda.synchronize()


def main():
    print("=" * 70)
    print("CORRECTNESS CHECK (vs v2)")
    print("=" * 70)
    print(f"{'shape':<18} {'v6 cos':>8} {'v6 rel':>8} "
          f"{'v7 cos':>8} {'v7 rel':>8} {'v8 cos':>8} {'v8 rel':>8}")
    shape_data = {}
    for shape in SHAPES:
        sd = prepare_shape(*shape)
        shape_data[shape] = sd
        if _pick_block_m(shape[1]) < 16:
            print(f"{str(shape):<18} {'-':>8} {'-':>8} {'-':>8} {'-':>8} "
                  f"{'-':>8} {'-':>8}  [BLOCK_M<16, skip]")
            continue
        res = check_correctness(sd)
        row = f"{str(shape):<18}"
        for name in ("v6", "v7", "v8"):
            cos, rel = res[name]
            if cos is None:
                row += f" {'ERR':>8} {rel[:6]:>8}"
            else:
                row += f" {cos:>8.5f} {rel:>7.2%}"
        print(row)

    # Bench only if correctness is "not totally broken"
    print("\n" + "=" * 70)
    print("PERF (BLOCK_M≥16 shapes only)")
    print("=" * 70)
    heavy_warmup(3.0)
    # JIT warm
    for shape in SHAPES:
        if _pick_block_m(shape[1]) < 16:
            continue
        sd = shape_data[shape]
        _ = time_fn(lambda: tq4_matvec_triton_v2(sd["x_rot"], sd["qs"],
                    sd["d"], sd["centroids"], shape[1], shape[0]), iters=30)
        _ = time_fn(lambda: tq4_matvec_triton_v6_prequant(
            sd["x_q8"], sd["x_scale"], sd["qs"], sd["d_fused"], sd["c_i8"],
            shape[1], shape[0]), iters=30)
        _ = time_fn(lambda: tq4_matvec_triton_v7_prequant(
            sd["x_q8"], sd["x_scale"], sd["qs"], sd["d_fused"], sd["c_i8"],
            shape[1], shape[0]), iters=30)
        _ = time_fn(lambda: tq4_matvec_triton_v8_prequant(
            sd["x_q8"], sd["x_scale"], sd["qs"], sd["d_fused"], sd["c_i8"],
            shape[1], shape[0]), iters=30)
    torch.cuda.synchronize()

    print(f"{'shape':<18} {'v2':>8} {'v6':>8} {'v7':>8} {'v8':>8} "
          f"{'v7vs2':>8} {'v8vs2':>8}")
    tv2 = tv6 = tv7 = tv8 = 0.0
    for shape in SHAPES:
        if _pick_block_m(shape[1]) < 16:
            continue
        sd = shape_data[shape]
        out_f, in_f = shape[1], shape[0]
        times = {
            "v2": statistics.median([time_fn(
                lambda: tq4_matvec_triton_v2(sd["x_rot"], sd["qs"], sd["d"],
                    sd["centroids"], out_f, in_f)) for _ in range(5)]),
            "v6": statistics.median([time_fn(
                lambda: tq4_matvec_triton_v6_prequant(
                    sd["x_q8"], sd["x_scale"], sd["qs"], sd["d_fused"],
                    sd["c_i8"], out_f, in_f)) for _ in range(5)]),
            "v7": statistics.median([time_fn(
                lambda: tq4_matvec_triton_v7_prequant(
                    sd["x_q8"], sd["x_scale"], sd["qs"], sd["d_fused"],
                    sd["c_i8"], out_f, in_f)) for _ in range(5)]),
            "v8": statistics.median([time_fn(
                lambda: tq4_matvec_triton_v8_prequant(
                    sd["x_q8"], sd["x_scale"], sd["qs"], sd["d_fused"],
                    sd["c_i8"], out_f, in_f)) for _ in range(5)]),
        }
        dv7 = (times["v7"] - times["v2"]) / times["v2"] * 100
        dv8 = (times["v8"] - times["v2"]) / times["v2"] * 100
        print(f"{str(shape):<18} {times['v2']:>7.1f} {times['v6']:>7.1f} "
              f"{times['v7']:>7.1f} {times['v8']:>7.1f} "
              f"{dv7:>+7.1f}% {dv8:>+7.1f}%")
        tv2 += times["v2"]; tv6 += times["v6"]
        tv7 += times["v7"]; tv8 += times["v8"]
    print("-" * 70)
    print(f"{'sum':<18} {tv2:>7.1f} {tv6:>7.1f} {tv7:>7.1f} {tv8:>7.1f} "
          f"{(tv7-tv2)/tv2*100:>+7.1f}% {(tv8-tv2)/tv2*100:>+7.1f}%")


if __name__ == "__main__":
    main()
