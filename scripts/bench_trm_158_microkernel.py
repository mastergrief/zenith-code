"""TRM-1.58 Gate B.3: ternary microkernel feasibility bench.

Compares three matmul paths per shape:
  1. cuBLAS BF16  (speed-of-light upper bound — what we'd run without ternary)
  2. PyTorch fake-quant TernaryLinear (no-kernel ternary baseline)
  3. Triton ternary kernel (the candidate)

Per workflow.md GPU bench discipline:
  - heavy_warmup (3s dense BF16 matmul to steady-state clock)
  - torch.cuda.Event timing (GPU-side timestamps)
  - Median of 5 × 2000 iters per shape
  - Same-process A/B paired per-shape
  - Correctness check (torch.allclose) BEFORE timing each variant

Shape grid (Gate B.3 minimum-viable scope):
  - TRM-1.58 first config (d_model=64, d_ffn=384, M=160 typical seq):
    W_qkv (192, 64), W_out (64, 64), ff_in (768, 64), ff_out (64, 384)
  - Scale-up (d_model=256, d_ffn=1024, M=160 — viability at scale):
    W_qkv (768, 256), W_out (256, 256), ff_in (2048, 256), ff_out (256, 1024)

Honest-reporting contract:
  - Report per-shape: cuBLAS, fake-quant, Triton wall-clock + memory
  - NO "kernel is faster" claim unless Triton beats BOTH baselines
  - Launch-bound at d_model=64 is an INFORMATIVE NULL, not a kernel failure
"""
from __future__ import annotations

import statistics
import sys
import time

import torch

from calm.llm_computer.ternary_linear import TernaryLinear
from calm.llm_computer.ternary_triton import (
    pack_ternary_2bit,
    quantize_to_ternary_indices,
    quantize_activation_int8_pertoken,
    ternary_matmul_triton,
    ternary_matmul_triton_v2,
    ternary_matmul_triton_v2_prequant,
)


N_ITERS = 2000
N_TRIALS = 5
WARMUP_SECONDS = 3.0


def heavy_warmup(seconds: float, device: str) -> None:
    """Dense BF16 matmul to drive GPU clock to steady state."""
    A = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
    B = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(100):
            _ = A @ B
        torch.cuda.synchronize()


def bench_one(fn, n_iters: int = N_ITERS, n_trials: int = N_TRIALS):
    """Median-of-trials of n_iters back-to-back invocations of fn().

    Returns: (median_us_per_iter, all_trials_us)
    """
    times = []
    for _ in range(n_trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            _ = fn()
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        times.append(elapsed_ms * 1000 / n_iters)
    return statistics.median(times), times


def cublas_bf16_matmul(x_bf16, w_bf16):
    """y = x @ w.T  in BF16 via cuBLAS."""
    return torch.nn.functional.linear(x_bf16, w_bf16)


def fake_quant_ternary(x_fp32, ternary_mod):
    """Pure-PyTorch TernaryLinear forward (fake-quant + STE)."""
    return ternary_mod(x_fp32)


def triton_ternary(x_fp32, w_packed, scale, in_features, out_features):
    """Triton ternary matmul kernel (v1, FP32 IEEE)."""
    return ternary_matmul_triton(x_fp32, w_packed, scale,
                                  in_features, out_features)


def triton_ternary_v2(x_fp32, w_packed, scale, in_features, out_features):
    """Triton ternary matmul kernel (v2, W1.58A8 INT8 tensor cores)."""
    return ternary_matmul_triton_v2(x_fp32, w_packed, scale,
                                     in_features, out_features)


def bench_shape(label: str, M: int, in_f: int, out_f: int, device: str = "cuda"):
    print(f"\n[bench] === {label}: M={M}  in={in_f}  out={out_f} ===")
    torch.manual_seed(42)
    x_fp32 = torch.randn(M, in_f, device=device, dtype=torch.float32)
    w_fp32 = torch.randn(out_f, in_f, device=device, dtype=torch.float32)

    # Variant 1: cuBLAS BF16
    x_bf16 = x_fp32.to(torch.bfloat16)
    w_bf16 = w_fp32.to(torch.bfloat16)
    # Correctness verify (BF16 has rel ~1e-2 error so just check shape + finite)
    y_cublas = cublas_bf16_matmul(x_bf16, w_bf16)
    assert y_cublas.shape == (M, out_f)
    assert torch.isfinite(y_cublas).all()

    # Variant 2: PyTorch fake-quant TernaryLinear
    ternary_mod = TernaryLinear(in_f, out_f, bias=False).to(device)
    with torch.no_grad():
        ternary_mod.weight.copy_(w_fp32)
    ternary_mod.eval()
    with torch.no_grad():
        y_fakeq = fake_quant_ternary(x_fp32, ternary_mod)
    assert y_fakeq.shape == (M, out_f)

    # Variant 3: Triton ternary v1 (FP32 IEEE)
    indices, scale = quantize_to_ternary_indices(w_fp32)
    w_packed = pack_ternary_2bit(indices)
    y_tri = triton_ternary(x_fp32, w_packed, scale, in_f, out_f)
    # Correctness gate vs fake-quant reference
    diff_tri_vs_fakeq = (y_tri - y_fakeq).abs().max().item()
    print(f"[bench]   correctness v1 vs fake-quant: max|diff|={diff_tri_vs_fakeq:.3e}")
    assert diff_tri_vs_fakeq < 1e-3, f"Triton v1 diverged at bench shape: {diff_tri_vs_fakeq}"

    # Variant 4: Triton ternary v2 (W1.58A8 INT8 tensor cores)
    y_v2 = triton_ternary_v2(x_fp32, w_packed, scale, in_f, out_f)
    # Correctness gate vs fake-quant reference (int8 activation has rel ~1/127
    # noise so threshold is looser than v1's fp32 IEEE; use rel error vs
    # output magnitude)
    diff_v2_vs_fakeq = (y_v2 - y_fakeq).abs().max().item()
    rel_v2 = diff_v2_vs_fakeq / (y_fakeq.abs().max().item() + 1e-8)
    print(f"[bench]   correctness v2 vs fake-quant: max|diff|={diff_v2_vs_fakeq:.3e}  "
          f"rel={rel_v2:.3e}")
    # int8 absmax activation expected to land within ~2-3% relative error
    assert rel_v2 < 0.05, f"Triton v2 diverged at bench shape (rel={rel_v2})"

    # Pre-quantize activation for v2_prequant (amortized real-pipeline cost
    # — in a real forward, x quantized once per layer-input, reused across
    # W_qkv/W_out/ff_in/ff_out projections)
    x_i8_pre, x_scale_pre = quantize_activation_int8_pertoken(x_fp32)
    x_i8_pre = x_i8_pre.contiguous()
    x_scale_pre = x_scale_pre.contiguous()

    # Bench
    t_cublas, _ = bench_one(lambda: cublas_bf16_matmul(x_bf16, w_bf16))
    with torch.no_grad():
        t_fakeq, _ = bench_one(lambda: fake_quant_ternary(x_fp32, ternary_mod))
    t_tri, _ = bench_one(lambda: triton_ternary(x_fp32, w_packed, scale,
                                                 in_f, out_f))
    t_v2, _ = bench_one(lambda: triton_ternary_v2(x_fp32, w_packed, scale,
                                                   in_f, out_f))
    t_v2_pre, _ = bench_one(lambda: ternary_matmul_triton_v2_prequant(
        x_i8_pre, x_scale_pre, w_packed, scale, in_f, out_f))

    # Memory footprint (weight side only, per-tensor)
    mem_bf16 = w_bf16.element_size() * w_bf16.numel()
    mem_fakeq = w_fp32.element_size() * w_fp32.numel()    # full FP master
    mem_packed = w_packed.element_size() * w_packed.numel()  # ternary forward
    # For fake-quant inference-only with no master grad, you could use the
    # quantized w_q tensor which is FP32 with ternary values — same size.
    # For deployment, ternary packed is the only relevant memory size.

    print(f"[bench]   timing (μs/iter, median of {N_TRIALS} × {N_ITERS} iters):")
    print(f"[bench]     cuBLAS BF16:           {t_cublas:8.2f} μs/iter")
    print(f"[bench]     PyTorch fake-quant:    {t_fakeq:8.2f} μs/iter  "
          f"({t_fakeq/t_cublas:.2f}× cuBLAS)")
    print(f"[bench]     Triton v1 (FP32 IEEE): {t_tri:8.2f} μs/iter  "
          f"({t_tri/t_cublas:.2f}× cuBLAS,  {t_tri/t_fakeq:.2f}× fake-quant)")
    print(f"[bench]     Triton v2 (W1.58A8):   {t_v2:8.2f} μs/iter  "
          f"({t_v2/t_cublas:.2f}× cuBLAS,  {t_v2/t_fakeq:.2f}× fake-quant,  "
          f"{t_v2/t_tri:.2f}× v1)")
    print(f"[bench]     Triton v2 (pre-quant): {t_v2_pre:8.2f} μs/iter  "
          f"({t_v2_pre/t_cublas:.2f}× cuBLAS,  {t_v2_pre/t_tri:.2f}× v1,  "
          f"{t_v2/t_v2_pre:.2f}× faster than v2-inline)")
    print(f"[bench]   weight memory:")
    print(f"[bench]     BF16:               {mem_bf16:8d} bytes")
    print(f"[bench]     fake-quant FP32:    {mem_fakeq:8d} bytes")
    print(f"[bench]     ternary packed:     {mem_packed:8d} bytes  "
          f"({mem_bf16/mem_packed:.1f}× smaller than BF16)")

    return {
        "label": label, "M": M, "in_f": in_f, "out_f": out_f,
        "t_cublas_us": t_cublas, "t_fakeq_us": t_fakeq,
        "t_tri_us": t_tri, "t_v2_us": t_v2, "t_v2_pre_us": t_v2_pre,
        "mem_bf16": mem_bf16, "mem_packed": mem_packed,
        "v1_diff_max": diff_tri_vs_fakeq, "v2_rel_diff": rel_v2,
    }


def main():
    assert torch.cuda.is_available(), "GPU required"
    device = "cuda"
    print(f"[bench] device={device}")
    print(f"[bench] heavy_warmup({WARMUP_SECONDS}s) ...")
    heavy_warmup(WARMUP_SECONDS, device)
    print("[bench] warmup done; starting bench loop")

    # TRM-1.58 first config shapes
    shapes_first_config = [
        ("W_qkv  (d=64, d_ffn=384)",  160, 64, 192),
        ("W_out  (d=64, d_ffn=384)",  160, 64, 64),
        ("ff_in  (d=64, d_ffn=384)",  160, 64, 768),
        ("ff_out (d=64, d_ffn=384)",  160, 384, 64),
    ]
    # Scale-up shapes (d_model=256, d_ffn=1024)
    shapes_scaleup = [
        ("W_qkv  (d=256, d_ffn=1024)", 160, 256, 768),
        ("W_out  (d=256, d_ffn=1024)", 160, 256, 256),
        ("ff_in  (d=256, d_ffn=1024)", 160, 256, 2048),
        ("ff_out (d=256, d_ffn=1024)", 160, 1024, 256),
    ]

    print("\n" + "="*70)
    print("[bench] TRM-1.58 first config (d_model=64, d_ffn=384)")
    print("="*70)
    results_first = [bench_shape(*shape) for shape in shapes_first_config]

    print("\n" + "="*70)
    print("[bench] Scale-up (d_model=256, d_ffn=1024)")
    print("="*70)
    results_scaleup = [bench_shape(*shape) for shape in shapes_scaleup]

    # Summary
    print("\n" + "="*70)
    print("[bench] SUMMARY")
    print("="*70)
    print(f"{'shape':40s}  {'cuBLAS':>7s}  {'fake-q':>7s}  {'v1':>7s}  "
          f"{'v2':>7s}  {'v2pre':>7s}  {'v2pre/cu':>9s}  {'mem×':>5s}")
    for r in results_first + results_scaleup:
        print(f"{r['label']:40s}  "
              f"{r['t_cublas_us']:7.2f}  {r['t_fakeq_us']:7.2f}  "
              f"{r['t_tri_us']:7.2f}  {r['t_v2_us']:7.2f}  {r['t_v2_pre_us']:7.2f}  "
              f"{r['t_v2_pre_us']/r['t_cublas_us']:9.2f}  "
              f"{r['mem_bf16']/r['mem_packed']:5.1f}")

    # Honest 2-axis verdict per codex 1779377742036-0686905a
    print()
    def axis_report(results, axis_name, kernel_key):
        beats_cublas = [r for r in results if r[kernel_key] < r['t_cublas_us']]
        beats_fakeq = [r for r in results if r[kernel_key] < r['t_fakeq_us']]
        return beats_cublas, beats_fakeq

    for variant_label, kernel_key in [("v1 (FP32 IEEE)", "t_tri_us"),
                                       ("v2 (W1.58A8, inline quant)", "t_v2_us"),
                                       ("v2 (W1.58A8, pre-quant)", "t_v2_pre_us")]:
        print(f"[bench] {variant_label}:")
        for cfg_label, results in [("first config (d=64)", results_first),
                                     ("scale-up (d=256)", results_scaleup)]:
            beats_cublas, beats_fakeq = axis_report(results, cfg_label, kernel_key)
            verdict = (
                "full speed claim" if beats_cublas and beats_fakeq
                else "memory + dispatch win" if beats_fakeq
                else "speed null"
            )
            print(f"[bench]   {cfg_label}: {len(beats_cublas)}/{len(results)} beat cuBLAS, "
                  f"{len(beats_fakeq)}/{len(results)} beat fake-quant → {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
