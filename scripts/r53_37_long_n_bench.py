"""R53.37 — Long-N decode bench: fp16 KV vs tq4 memo vs tq4 fused
flash-attn at N ∈ {8192, 16384}.

Purpose: extend the 2026-04-20 r53_phase2_bench.py curve past the
current runtime N-gate (`128 < cached_kv_len < 2048`) to find or
firmly rule out an asymptotic crossover where the fused flash-attn
kernel catches up to or overtakes the Phase 1 memoized-dequant path.

Prediction from R53.34 + arch first principles:
- Fused kernel has fixed per-Q-head × 42-layer launch overhead
  (336 Triton launches per decode step).
- Memo path materializes KV once on insert; reuses one cuBLAS matmul
  per step. cuBLAS is near-peak on (1, 2560) @ (N, 2560) and scales
  linearly with N.
- Expectation: memo continues to dominate at N ≥ 4K; crossover
  unlikely unless tile scheduling / per-head fusion improves.

GPU discipline (.claude/rules/workflow.md §"GPU bench discipline"):
- heavy_warmup(3.0s) before any timing
- torch.cuda.Event(enable_timing=True) — not time.time()
- correctness check vs fp16 baseline before timing
- median-of-5 per config (configurable via N_RUNS)
- same-process A/B, paired per-N (GPU clock doesn't cool between)

Cost estimate (RTX 4070 Laptop, Gemma E4B tq4):
- fp16 at N=8192: 8192 / ~7 tok/s ≈ 20 min per run
- memo at N=8192: 8192 / ~6 tok/s ≈ 23 min per run
- fused at N=8192: 8192 / ~6 tok/s ≈ 23 min per run
- N=8192 total: 66 min × N_RUNS
- N=16384 total: 132 min × N_RUNS

With N_RUNS=1 and [8192, 16384]: ~3.3 hours.
With N_RUNS=3 and [8192]: ~3.3 hours.

Default: N=[8192] × 1 run for directional check. Adjust
LONG_N_CONFIGS and N_RUNS at the top of this file for wider sweep.

Run via daemon (m, tok pre-bound):
  bin/gemma-run scripts/r53_37_long_n_bench.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---- Bench config ----
# Start small — single directional run at 8K. Bump to [8192, 16384]
# and/or N_RUNS=3 for a proper median bench when session budget allows.
LONG_N_CONFIGS = [
    (8192,  "n_decode=8192  — extending past 2048 gate"),
    # (16384, "n_decode=16384 — asymptotic regime"),   # ~2.2h per path
]
N_RUNS = 1          # median-of-N_RUNS per config
HEAVY_WARMUP_SEC = 3.0
PROMPT = "The quick brown fox jumps over the lazy dog. " * 4   # ~40 tok


def _heavy_warmup(duration_sec: float):
    """Run dense fp16 matmuls at steady-state clock for `duration_sec`.
    Ensures GPU is at sustained frequency before timing."""
    print(f"[bench] heavy_warmup({duration_sec}s)...", flush=True)
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.time()
    while time.time() - t0 < duration_sec:
        c = a @ b
        a = c * 0.5 + a * 0.5
    torch.cuda.synchronize()
    del a, b, c


def _time_decode_cuda_events(m_ref, tok_ref, cache_factory, fused_on,
                             prompt: str, n_decode: int) -> float:
    """Time a single prefill + n_decode decode path using cuda.Events.
    Returns decode-only seconds (prefill excluded). Caller repeats
    for median."""
    from calm.llm_computer.gemma_substrate import enable_fused_flash_attn
    enable_fused_flash_attn(fused_on)

    ids = tok_ref.encode(prompt)
    cache = cache_factory(m_ref, len(ids) + n_decode + 8)

    # Prefill (not timed)
    with torch.no_grad():
        logits = m_ref.forward(torch.tensor([ids]), device="cuda",
                               kv_cache=cache, start_pos=0)
    torch.cuda.synchronize()
    cur_id = int(logits[0, -1].argmax().item())

    # Timed decode
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for step in range(n_decode):
        with torch.no_grad():
            logits = m_ref.forward(torch.tensor([[cur_id]]), device="cuda",
                                   kv_cache=cache,
                                   start_pos=len(ids) + step)
        cur_id = int(logits[0, -1].argmax().item())
    end.record()
    end.synchronize()
    decode_ms = start.elapsed_time(end)
    return decode_ms / 1000.0


def _correctness_check(m_ref, tok_ref, cache_factory):
    """Verify fused vs memo produce same first-token argmax on a short
    prompt. Single-sample sanity; not exhaustive."""
    from calm.llm_computer.gemma_substrate import (
        KVCache, KVCacheTq4, enable_fused_flash_attn,
    )
    ids = tok_ref.encode("What is 17 times 23?")
    # fp16 reference
    enable_fused_flash_attn(True)
    cache = KVCache(m_ref.config.n_layers, device="cuda")
    with torch.no_grad():
        l_ref = m_ref.forward(torch.tensor([ids]), device="cuda",
                              kv_cache=cache, start_pos=0)
    ref_argmax = int(l_ref[0, -1].argmax())
    # tq4 fused
    enable_fused_flash_attn(True)
    cache = KVCacheTq4(m_ref, max_len=64, device="cuda")
    with torch.no_grad():
        l_fused = m_ref.forward(torch.tensor([ids]), device="cuda",
                                kv_cache=cache, start_pos=0)
    fused_argmax = int(l_fused[0, -1].argmax())
    # tq4 memo
    enable_fused_flash_attn(False)
    cache = KVCacheTq4(m_ref, max_len=64, device="cuda")
    with torch.no_grad():
        l_memo = m_ref.forward(torch.tensor([ids]), device="cuda",
                               kv_cache=cache, start_pos=0)
    memo_argmax = int(l_memo[0, -1].argmax())
    print(f"[bench] correctness: fp16={ref_argmax} "
          f"tq4-fused={fused_argmax} tq4-memo={memo_argmax}", flush=True)
    # Don't hard-assert — tq4 is lossy, may diverge on edge tokens
    return ref_argmax == fused_argmax == memo_argmax


def run_bench():
    from calm.llm_computer.gemma_substrate import (
        KVCache, KVCacheTq4, enable_triton_tq4,
    )

    # m, tok are daemon globals
    global m, tok

    enable_triton_tq4(True)   # weight kernels always on

    # Correctness sanity (single sample, not exhaustive)
    ok = _correctness_check(m, tok)
    print(f"[bench] correctness sanity: argmax match = {ok}", flush=True)

    _heavy_warmup(HEAVY_WARMUP_SEC)

    all_results = []
    for n_decode, label in LONG_N_CONFIGS:
        print(f"\n[bench] === {label} ===", flush=True)

        fp16_runs, memo_runs, fused_runs = [], [], []
        for run in range(N_RUNS):
            print(f"[bench] run {run+1}/{N_RUNS}", flush=True)

            t = _time_decode_cuda_events(
                m, tok,
                lambda mr, n: KVCache(mr.config.n_layers, device="cuda"),
                fused_on=True, prompt=PROMPT, n_decode=n_decode)
            fp16_tps = n_decode / t
            fp16_runs.append(fp16_tps)
            print(f"  fp16 KV:          {t:7.2f}s  {fp16_tps:6.2f} tok/s",
                  flush=True)

            t = _time_decode_cuda_events(
                m, tok,
                lambda mr, n: KVCacheTq4(mr, max_len=n, device="cuda"),
                fused_on=False, prompt=PROMPT, n_decode=n_decode)
            memo_tps = n_decode / t
            memo_runs.append(memo_tps)
            print(f"  tq4 memo:         {t:7.2f}s  {memo_tps:6.2f} tok/s",
                  flush=True)

            t = _time_decode_cuda_events(
                m, tok,
                lambda mr, n: KVCacheTq4(mr, max_len=n, device="cuda"),
                fused_on=True, prompt=PROMPT, n_decode=n_decode)
            fused_tps = n_decode / t
            fused_runs.append(fused_tps)
            print(f"  tq4 fused:        {t:7.2f}s  {fused_tps:6.2f} tok/s",
                  flush=True)

        # Median across runs
        def _med(xs): return statistics.median(xs) if xs else 0.0
        fp16_m = _med(fp16_runs)
        memo_m = _med(memo_runs)
        fused_m = _med(fused_runs)
        all_results.append((n_decode, fp16_m, memo_m, fused_m))

        speedup = fused_m / memo_m if memo_m else 0.0
        vs_fp16 = fused_m / fp16_m if fp16_m else 0.0
        verdict = "FUSED WINS" if speedup > 1.0 else "MEMO WINS"
        print(f"[bench] N={n_decode}  median tok/s: "
              f"fp16={fp16_m:.2f} memo={memo_m:.2f} fused={fused_m:.2f}",
              flush=True)
        print(f"[bench] fused/memo={speedup:.2f}×  fused/fp16={vs_fp16:.1%}  "
              f"→ {verdict}", flush=True)

    # Summary
    print("\n[bench] SUMMARY (median-of-{})".format(N_RUNS), flush=True)
    print(f"{'N':>6}  {'fp16':>8}  {'memo':>8}  {'fused':>8}  "
          f"{'fused/memo':>11}  {'% fp16':>7}  verdict", flush=True)
    for n_decode, fp16_m, memo_m, fused_m in all_results:
        sp = fused_m / memo_m if memo_m else 0.0
        fr = fused_m / fp16_m if fp16_m else 0.0
        verdict = "FUSED" if sp > 1.0 else "MEMO"
        print(f"{n_decode:>6}  {fp16_m:>7.2f}  {memo_m:>7.2f}  "
              f"{fused_m:>7.2f}  {sp:>10.2f}×  {fr*100:>6.1f}%  {verdict}",
              flush=True)

    print("[bench] DONE", flush=True)


run_bench()
