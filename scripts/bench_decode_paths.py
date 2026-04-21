"""bench_decode_paths — measure end-to-end decode tok/s across
current substrate paths.

Four paths compared:

  (A) generate(use_tq4_kv=False)           — tq4 weights + fp16 KV, no CUDA Graphs
  (B) generate(use_tq4_kv=True)            — tq4 weights + tq4 KV, no CUDA Graphs
  (C) generate_with_graph(max_len=...)     — tq4 weights + fp16 KVCacheStatic + CUDA Graphs
  (D) generate_with_graph_tq4(max_len=...) — tq4 weights + tq4 KVCacheTq4Static + CUDA Graphs (NEW)

Goal: establish baseline tok/s at a realistic decode length (256 tok)
from a fixed prompt. The numbers guide the decode-speedup work:
the gap between (A|B) and (C) tells us how much CUDA Graphs would
lift the tq4 KV path once KVCacheTq4 becomes graph-safe.

GPU bench discipline per workflow.md:
- heavy_warmup
- torch.cuda.Event timing
- median of 3 runs
- correctness check against path A

Run via daemon (m, tok pre-bound):
  bin/gemma-run scripts/bench_decode_paths.py
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


# Bench config
PROMPT = ("Write a Python function that sorts a list of integers "
          "using quicksort. Include type hints.\n\n")
DECODE_TOKENS = 256            # short decode — we care about steady-state tok/s
N_RUNS = 3
MAX_LEN_GRAPH = 1024           # KVCacheStatic buffer size for graph path


def heavy_warmup(secs: float = 3.0) -> None:
    """Spin dense fp16 matmuls to pin GPU to steady-state clock."""
    t_end = time.time() + secs
    a = torch.randn(2048, 2048, dtype=torch.float16, device="cuda")
    b = torch.randn(2048, 2048, dtype=torch.float16, device="cuda")
    while time.time() < t_end:
        c = a @ b
    torch.cuda.synchronize()


def time_decode(fn) -> float:
    """Returns seconds for a single decode call via cudaEvent.
    Assumes fn is a no-arg callable that runs a full generate()."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    return ms / 1000.0, result


def bench_path(name: str, fn, n_runs: int = N_RUNS,
               expected_tokens: int | None = None) -> dict:
    """Run fn n_runs times, report median tok/s + notes."""
    secs_list = []
    token_count = None
    for i in range(n_runs):
        secs, out = time_decode(fn)
        toks = len(out["token_ids"])
        if token_count is None:
            token_count = toks
        secs_list.append(secs)
        print(f"  [{name} run {i+1}] {toks} tok in {secs:.2f}s "
              f"= {toks/secs:.2f} tok/s", flush=True)
    median_s = statistics.median(secs_list)
    return {
        "name": name,
        "median_s": median_s,
        "tokens": token_count,
        "tok_per_s": token_count / median_s if token_count else 0,
        "all_secs": secs_list,
    }


def main() -> None:
    if "m" not in globals() or "tok" not in globals():
        print("ERROR: run via bin/gemma-run (needs m, tok pre-bound)")
        return

    m_ref = globals()["m"]
    tok_ref = globals()["tok"]

    print("=" * 72, flush=True)
    print("bench_decode_paths — baseline decode throughput", flush=True)
    print("=" * 72, flush=True)
    print(f"prompt: {PROMPT.strip()[:60]}...", flush=True)
    print(f"decode: {DECODE_TOKENS} tokens, n_runs={N_RUNS}", flush=True)
    print("=" * 72, flush=True)

    prompt_ids = tok_ref.encode(PROMPT)
    print(f"prompt tokens: {len(prompt_ids)}", flush=True)

    heavy_warmup(3.0)
    print("[warmup done]", flush=True)

    # Pre-run once to JIT-compile Triton kernels + warm caches
    print("\n[jit warmup] generate(use_tq4_kv=False) 64 tok...", flush=True)
    _ = m_ref.generate(PROMPT, tok_ref, max_tokens=64, device="cuda",
                       stop_on_eos=False, use_tq4_kv=False)

    print("[jit warmup] generate(use_tq4_kv=True) 64 tok...", flush=True)
    _ = m_ref.generate(PROMPT, tok_ref, max_tokens=64, device="cuda",
                       stop_on_eos=False, use_tq4_kv=True)

    print("[jit warmup] generate_with_graph 64 tok...", flush=True)
    _ = m_ref.generate_with_graph(PROMPT, tok_ref, max_tokens=64,
                                   device="cuda", stop_on_eos=False,
                                   max_len=MAX_LEN_GRAPH)

    print("[jit warmup] generate_with_graph_tq4 64 tok...", flush=True)
    _ = m_ref.generate_with_graph_tq4(PROMPT, tok_ref, max_tokens=64,
                                       device="cuda", stop_on_eos=False,
                                       max_len=MAX_LEN_GRAPH)

    print("\n[bench path A] generate() fp16 KV, no graphs", flush=True)
    path_a = bench_path(
        "A-fp16-KV",
        lambda: m_ref.generate(PROMPT, tok_ref, max_tokens=DECODE_TOKENS,
                               device="cuda", stop_on_eos=False,
                               use_tq4_kv=False),
    )

    print("\n[bench path B] generate() tq4 KV, no graphs", flush=True)
    path_b = bench_path(
        "B-tq4-KV",
        lambda: m_ref.generate(PROMPT, tok_ref, max_tokens=DECODE_TOKENS,
                               device="cuda", stop_on_eos=False,
                               use_tq4_kv=True),
    )

    print("\n[bench path C] generate_with_graph() fp16 static + graphs",
          flush=True)
    path_c = bench_path(
        "C-graph-fp16",
        lambda: m_ref.generate_with_graph(
            PROMPT, tok_ref, max_tokens=DECODE_TOKENS, device="cuda",
            stop_on_eos=False, max_len=MAX_LEN_GRAPH),
    )

    print("\n[bench path D] generate_with_graph_tq4() tq4 static + graphs",
          flush=True)
    path_d = bench_path(
        "D-graph-tq4",
        lambda: m_ref.generate_with_graph_tq4(
            PROMPT, tok_ref, max_tokens=DECODE_TOKENS, device="cuda",
            stop_on_eos=False, max_len=MAX_LEN_GRAPH),
    )

    # ---- Report ----
    print("\n" + "=" * 72, flush=True)
    print("RESULTS", flush=True)
    print("=" * 72, flush=True)
    header = f"{'path':<18} {'tokens':>7} {'median_s':>10} {'tok/s':>8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in (path_a, path_b, path_c, path_d):
        print(f"{r['name']:<18} {r['tokens']:>7d} "
              f"{r['median_s']:>10.2f} {r['tok_per_s']:>8.2f}",
              flush=True)

    print("\nratios:", flush=True)
    print(f"  B/A = {path_b['tok_per_s']/path_a['tok_per_s']:.3f}  "
          f"(tq4 KV overhead vs fp16 KV, no graphs)", flush=True)
    print(f"  C/A = {path_c['tok_per_s']/path_a['tok_per_s']:.3f}  "
          f"(CUDA Graphs lift for fp16 KV)", flush=True)
    print(f"  C/B = {path_c['tok_per_s']/path_b['tok_per_s']:.3f}  "
          f"(graphs+fp16 vs no-graphs+tq4)", flush=True)
    print(f"  D/B = {path_d['tok_per_s']/path_b['tok_per_s']:.3f}  "
          f"(GRAPHS LIFT FOR TQ4 KV — main result)", flush=True)
    print(f"  D/C = {path_d['tok_per_s']/path_c['tok_per_s']:.3f}  "
          f"(tq4+graphs vs fp16+graphs — cost of tq4 dequant)", flush=True)

    llama_target = 42.0
    print(f"\nllama.cpp target (~{llama_target} tok/s on same GGUF):",
          flush=True)
    for r in (path_a, path_b, path_c, path_d):
        gap = r['tok_per_s'] / llama_target * 100
        print(f"  {r['name']:<18} {r['tok_per_s']:>6.2f} tok/s "
              f"= {gap:5.1f}% of llama", flush=True)
    print("\n[r53.bench_decode] DONE", flush=True)


if __name__ == "__daemon__":
    main()
elif __name__ == "__main__":
    print("ERROR: run via bin/gemma-run (this script needs m, tok from daemon)")
    sys.exit(1)
