"""Microbench: attn_kv_fused vs sequential attn_k(x), attn_v(x).

End-to-end decode bench (bench_decode_paths.py) is unreliable under
concurrent CPU load (rustc, codex tests). This microbench compares
exactly ONE kernel invocation against the pair it replaces, using
cudaEvent timing and batched median-of-N for stability.

What it measures:
  t_sep  = time(attn_k(x)) + time(attn_v(x))
  t_fuse = time(attn_kv_fused(x))

Expected: t_fuse < t_sep because one kernel launch + shared x_rot read.
Run via daemon (m, tok pre-bound):
  bin/gemma-run scripts/bench_attn_kv_fused.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import statistics

import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


N_ITERS = 200              # iterations per timing
N_RUNS = 5                 # outer runs for median
LAYERS_TO_BENCH = [0, 5, 23]  # SWA layer, global layer (d_head=512)


def heavy_warmup(secs: float = 2.0) -> None:
    t_end = time.time() + secs
    a = torch.randn(2048, 2048, dtype=torch.float16, device="cuda")
    b = torch.randn(2048, 2048, dtype=torch.float16, device="cuda")
    while time.time() < t_end:
        _ = a @ b
    torch.cuda.synchronize()


def time_op(fn, n_iters: int = N_ITERS) -> float:
    """Return seconds for n_iters invocations (cudaEvent timing)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(n_iters):
        result = fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    return ms / 1000.0 / n_iters  # per-iter seconds


def main() -> None:
    if "m" not in globals() or "tok" not in globals():
        print("ERROR: run via bin/gemma-run")
        return

    m_ref = globals()["m"]
    cfg = m_ref.config

    print("=" * 72, flush=True)
    print("attn_kv_fused microbench", flush=True)
    print(f"iters={N_ITERS}, runs={N_RUNS}, layers={LAYERS_TO_BENCH}", flush=True)
    print("=" * 72, flush=True)

    heavy_warmup(2.0)
    print("[warmup done]", flush=True)

    total_sep_us = 0.0
    total_fuse_us = 0.0

    for layer_idx in LAYERS_TO_BENCH:
        layer = m_ref.layers[layer_idx]

        # Build a representative input: (1, 1, d_model) fp32 on cuda
        # matching the _forward_layer single-token decode shape.
        x = torch.randn(1, 1, cfg.d_model, dtype=torch.float32, device="cuda")

        # Correctness check first (cheap)
        torch.cuda.synchronize()
        with torch.no_grad():
            k1 = layer.attn_k(x)
            v1 = layer.attn_v(x)
            k2, v2 = layer.attn_kv_fused(x)
        diff_k = (k1 - k2).abs().max().item()
        diff_v = (v1 - v2).abs().max().item()
        print(f"\nlayer {layer_idx} (d_head_k out={layer.attn_k.out_features}):",
              flush=True)
        print(f"  correctness: max|Δk|={diff_k:.2e}, max|Δv|={diff_v:.2e}",
              flush=True)
        assert diff_k < 1e-4, f"k mismatch {diff_k}"
        assert diff_v < 1e-4, f"v mismatch {diff_v}"

        # Warm kernels once (JIT compile)
        with torch.no_grad():
            _ = layer.attn_k(x)
            _ = layer.attn_v(x)
            _ = layer.attn_kv_fused(x)

        sep_times = []
        fuse_times = []
        for r in range(N_RUNS):
            with torch.no_grad():
                def sep():
                    return layer.attn_k(x), layer.attn_v(x)
                def fuse():
                    return layer.attn_kv_fused(x)
                t_sep = time_op(sep)
                t_fuse = time_op(fuse)
            sep_times.append(t_sep * 1e6)   # μs
            fuse_times.append(t_fuse * 1e6)

        m_sep = statistics.median(sep_times)
        m_fuse = statistics.median(fuse_times)
        speedup = m_sep / m_fuse if m_fuse > 0 else 0
        print(f"  sep   : {m_sep:7.2f} μs  (runs: "
              f"{', '.join(f'{t:.1f}' for t in sep_times)})", flush=True)
        print(f"  fuse  : {m_fuse:7.2f} μs  (runs: "
              f"{', '.join(f'{t:.1f}' for t in fuse_times)})", flush=True)
        print(f"  speedup: {speedup:.3f}×  ({((speedup-1)*100):+.1f}%)",
              flush=True)

        total_sep_us += m_sep
        total_fuse_us += m_fuse

    print("\n" + "=" * 72, flush=True)
    print("AGGREGATE (all benched layers)", flush=True)
    print("=" * 72, flush=True)
    total_speedup = total_sep_us / total_fuse_us if total_fuse_us > 0 else 0
    print(f"  total sep   : {total_sep_us:7.2f} μs", flush=True)
    print(f"  total fuse  : {total_fuse_us:7.2f} μs", flush=True)
    print(f"  total saved : {total_sep_us - total_fuse_us:7.2f} μs "
          f"({total_speedup:.3f}×, {((total_speedup-1)*100):+.1f}%)",
          flush=True)

    # Estimate end-to-end impact: ~24 own-KV layers in Gemma E4B
    # (42 total - 18 shared-KV readers). If the layer-avg savings
    # generalize, decode-step savings = avg_saving × 24.
    own_kv_layers = 0
    for il in range(cfg.n_layers):
        kv_src = cfg.kv_source_layer(il, is_swa=True)
        if kv_src == il:
            own_kv_layers += 1
    avg_saving = (total_sep_us - total_fuse_us) / len(LAYERS_TO_BENCH)
    est_step_saving_ms = avg_saving * own_kv_layers / 1000
    print(f"\n  est. decode-step savings: {avg_saving:.2f} μs/layer × "
          f"{own_kv_layers} own-KV layers = {est_step_saving_ms:.2f} ms/step",
          flush=True)
    print(f"  at 40 ms/step baseline → {est_step_saving_ms/40*100:.1f}% lift",
          flush=True)

    print("\n[r53.bench_attn_kv] DONE", flush=True)


if __name__ == "__daemon__":
    main()
elif __name__ == "__main__":
    print("ERROR: run via bin/gemma-run")
    sys.exit(1)
