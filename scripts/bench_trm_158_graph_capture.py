"""Slice 15: CUDA graph capture bench for full TRM-1.58 forward pass.

Compares four full-forward variants at the TRM-1.58 first config
(d_model=64, d_ffn=384, n_iter=4, h_cycles=2 — 10 effective stack passes):

  1. FP RDT-v2 forward (use_ternary_bulk=False, BF16/FP32 native)
  2. FP RDT-v2 forward, graph-captured
  3. TRM-1.58 forward (use_ternary_bulk=True, PyTorch fake-quant ternary)
  4. TRM-1.58 forward, graph-captured

Goal: characterize whether CUDA graph capture amortizes the launch-
overhead bottleneck diagnosed in Slice 14a/b. If the remaining
TRM-1.58-vs-cuBLAS gap is dominated by launch overhead, graph capture
should close it. If matmul-itself dominates, graph capture won't help.

Speed-claim gate per locked Gate B contract:
  - TRM-1.58 (3 or 4) must beat FP RDT-v2 (1 or 2) at full-forward
    wall-clock to claim "TRM-1.58 is faster than FP RDT-v2."
  - Otherwise: report honest null + 8× memory + 1.6× fusion as the
    final TRM-1.58 efficiency receipt.

Per workflow.md GPU bench discipline: heavy_warmup, torch.cuda.Event,
median of 5 × N iters.
"""
from __future__ import annotations

import statistics
import sys
import time

import torch

from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta
from calm.llm_computer.gsm8k_tokenizer import Gsm8kTokenizer


# Same fixture as Gate A falsifier
SYNTH_CORPUS = [
    {"question": "Alice has 3 apples. Bob gives her 7 more. How many?", "expected": "10"},
    {"question": "John buys 5 books at $12 each. Total cost?", "expected": "60"},
    {"question": "A train travels 60 mph for 2.5 hours. Distance?", "expected": "150"},
    {"question": "What is 17 times 23?", "expected": "391"},
    {"question": "Sara saves $25 weekly. After 8 weeks, how much?", "expected": "200"},
]

N_ITERS = 100      # full forward is much more expensive than a single matmul
N_TRIALS = 5
WARMUP_SECONDS = 3.0


def heavy_warmup(seconds: float, device: str) -> None:
    A = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
    B = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(100):
            _ = A @ B
        torch.cuda.synchronize()


def bench_fn(fn, n_iters: int = N_ITERS, n_trials: int = N_TRIALS):
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
        times.append(elapsed_ms * 1000 / n_iters)  # μs per call
    return statistics.median(times)


def build_model(tok, use_ternary_bulk: bool):
    torch.manual_seed(42)
    return build_copy_augmented_delta(
        vocab_size=tok.vocab_size,
        d_model=64, n_heads=32, n_layers=4, d_ffn=384, max_len=512,
        n_copy_heads=4, sep_token_id=tok.sep_id,
        use_chunkwise=True,
        n_iterations=4, h_cycles=2,
        use_loop_index=True, use_input_injection=True,
        use_gated_attention=True, use_z_init=True, use_lecun_init=True,
        use_h_rmsnorm=True, use_short_conv=True, use_h_layer_stack=True,
        use_pre_rmsnorm=True,
        use_ternary_bulk=use_ternary_bulk,
    ).cuda()


def main():
    assert torch.cuda.is_available(), "GPU required"
    device = "cuda"
    print(f"[graph-bench] device={device}")

    tok = Gsm8kTokenizer.from_corpus(SYNTH_CORPUS)

    # Build both models
    print("[graph-bench] building FP RDT-v2 (use_ternary_bulk=False)...")
    m_fp = build_model(tok, use_ternary_bulk=False)
    m_fp.eval()
    print("[graph-bench] building TRM-1.58 (use_ternary_bulk=True)...")
    m_trm = build_model(tok, use_ternary_bulk=True)
    m_trm.eval()

    n_params_fp = sum(p.numel() for p in m_fp.parameters())
    n_params_trm = sum(p.numel() for p in m_trm.parameters())
    print(f"[graph-bench]   FP params:   {n_params_fp:,}")
    print(f"[graph-bench]   TRM params:  {n_params_trm:,}")
    print(f"[graph-bench]   param delta: {n_params_trm - n_params_fp} "
          f"(expected 0 — TernaryLinear has same param count as nn.Linear)")

    # Encode fixed input (graphs require fixed shapes)
    base_ids, _ = tok.encode_example("What is 17 times 23?", "391")
    ids = list(base_ids)
    if len(ids) < 160:
        ids.extend([tok.eos_id] * (160 - len(ids)))
    ids = ids[:160]
    ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    print(f"[graph-bench] input ids.shape={tuple(ids_t.shape)}")

    print(f"[graph-bench] heavy_warmup({WARMUP_SECONDS}s)...")
    heavy_warmup(WARMUP_SECONDS, device)

    # ---------- Bench 1: FP RDT-v2 forward (no graph) ----------
    print("\n[graph-bench] === bench 1: FP RDT-v2 forward (no graph) ===")
    with torch.no_grad():
        # warm up to avoid first-call compile / autotune costs
        for _ in range(3):
            _ = m_fp(ids_t)
        torch.cuda.synchronize()
        t_fp_nograph = bench_fn(lambda: m_fp(ids_t))
    print(f"[graph-bench]   FP no-graph: {t_fp_nograph:8.2f} μs/forward "
          f"(median of {N_TRIALS} × {N_ITERS})")

    # ---------- Bench 2: FP RDT-v2 forward (graph-captured) ----------
    print("\n[graph-bench] === bench 2: FP RDT-v2 forward (graph-captured) ===")
    try:
        # Per PyTorch docs: separate stream for graph capture, warmup on that stream first
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(3):
                    _ = m_fp(ids_t)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Capture
        g_fp = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g_fp):
            out_fp_g = m_fp(ids_t)
        print(f"[graph-bench]   FP graph captured. Output finite: "
              f"{bool(torch.isfinite(out_fp_g).all())}")

        # Replay
        def replay_fp():
            g_fp.replay()
            return out_fp_g
        t_fp_graph = bench_fn(replay_fp)
        print(f"[graph-bench]   FP graph:    {t_fp_graph:8.2f} μs/forward "
              f"({t_fp_nograph/t_fp_graph:.2f}× speedup from capture)")
    except Exception as e:
        print(f"[graph-bench]   FP graph CAPTURE FAILED: {e}")
        t_fp_graph = None

    # ---------- Bench 3: TRM-1.58 forward (no graph) ----------
    print("\n[graph-bench] === bench 3: TRM-1.58 forward (no graph) ===")
    with torch.no_grad():
        for _ in range(3):
            _ = m_trm(ids_t)
        torch.cuda.synchronize()
        t_trm_nograph = bench_fn(lambda: m_trm(ids_t))
    print(f"[graph-bench]   TRM no-graph: {t_trm_nograph:8.2f} μs/forward "
          f"(median of {N_TRIALS} × {N_ITERS})")

    # ---------- Bench 4: TRM-1.58 forward (graph-captured) ----------
    print("\n[graph-bench] === bench 4: TRM-1.58 forward (graph-captured) ===")
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(3):
                    _ = m_trm(ids_t)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g_trm = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g_trm):
            out_trm_g = m_trm(ids_t)
        print(f"[graph-bench]   TRM graph captured. Output finite: "
              f"{bool(torch.isfinite(out_trm_g).all())}")

        def replay_trm():
            g_trm.replay()
            return out_trm_g
        t_trm_graph = bench_fn(replay_trm)
        print(f"[graph-bench]   TRM graph:    {t_trm_graph:8.2f} μs/forward "
              f"({t_trm_nograph/t_trm_graph:.2f}× speedup from capture)")
    except Exception as e:
        print(f"[graph-bench]   TRM graph CAPTURE FAILED: {e}")
        t_trm_graph = None

    # ---------- Summary ----------
    print("\n" + "="*70)
    print("[graph-bench] SUMMARY (TRM-1.58 first config: d_model=64, d_ffn=384,")
    print("[graph-bench]                                 n_iter=4, h_cycles=2)")
    print("="*70)
    print(f"{'variant':40s}  {'μs/forward':>12s}  {'vs FP no-graph':>15s}")
    print(f"{'FP RDT-v2 (no graph)':40s}  {t_fp_nograph:12.2f}  {1.0:15.2f}×")
    if t_fp_graph is not None:
        print(f"{'FP RDT-v2 (graph-captured)':40s}  {t_fp_graph:12.2f}  "
              f"{t_fp_nograph/t_fp_graph:15.2f}×")
    print(f"{'TRM-1.58 (no graph)':40s}  {t_trm_nograph:12.2f}  "
          f"{t_fp_nograph/t_trm_nograph:15.2f}×")
    if t_trm_graph is not None:
        print(f"{'TRM-1.58 (graph-captured)':40s}  {t_trm_graph:12.2f}  "
              f"{t_fp_nograph/t_trm_graph:15.2f}×")

    print()
    print("[graph-bench] Speed-claim gate (TRM-1.58 must beat FP RDT-v2):")
    if t_trm_graph is not None and t_fp_graph is not None:
        if t_trm_graph < t_fp_graph:
            print(f"[graph-bench]   ✓ TRM-1.58 graph ({t_trm_graph:.1f} μs) "
                  f"BEATS FP graph ({t_fp_graph:.1f} μs) — speed claim CLEARS at this scale")
        elif t_trm_graph < t_fp_nograph:
            print(f"[graph-bench]   ◐ TRM-1.58 graph ({t_trm_graph:.1f} μs) beats "
                  f"FP no-graph ({t_fp_nograph:.1f} μs) but trails FP graph "
                  f"({t_fp_graph:.1f} μs) — partial speed win")
        else:
            print(f"[graph-bench]   ✗ TRM-1.58 graph ({t_trm_graph:.1f} μs) "
                  f"TRAILS FP graph ({t_fp_graph:.1f} μs) — no speed claim "
                  f"at d_model=64")
    elif t_trm_graph is not None:
        if t_trm_graph < t_fp_nograph:
            print(f"[graph-bench]   ◐ TRM-1.58 graph beats FP no-graph (FP graph "
                  f"unavailable) — partial speed signal")
        else:
            print(f"[graph-bench]   ✗ TRM-1.58 graph trails FP no-graph "
                  f"(FP graph unavailable) — no speed claim")
    else:
        print(f"[graph-bench]   ?  Graph capture failed; comparing no-graph only")
        if t_trm_nograph < t_fp_nograph:
            print(f"[graph-bench]     TRM no-graph ({t_trm_nograph:.1f} μs) beats "
                  f"FP no-graph ({t_fp_nograph:.1f} μs)")
        else:
            print(f"[graph-bench]     TRM no-graph trails FP no-graph")

    return 0


if __name__ == "__main__":
    sys.exit(main())
