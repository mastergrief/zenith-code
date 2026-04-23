# TurboQuant — Historical receipts (Round 11 hybrid finding, R53.29-R53.34 kernel arc, 2026-04-20 bench re-run, tq4_qjl research artifact)

Per-kernel bench tables, matvec-variant A/B results, fused
flash-attention full perf curve, CUDA-vs-Triton lesson-transfer
analysis. Current rules: `.claude/rules/turboquant.md`. This file
exists for archaeology — "which kernel variant shipped when", "why
the shipped-policy N-gate is 128..2048", "what CUDA tricks didn't
transfer to Triton".

## Round 11 hybrid-substrate origin

tq4 quantization DESTROYS compiled card weights. Lloyd-Max codebook
is tuned for Gaussian LM weights; compiled cards have discrete
integer coefs (±1, ±16). Measured: dispatched_v4 791/791 → 70/791
(91% loss) through tq4 roundtrip. Fix was `HybridGroupedSmall2DTransformer`
with per-layer linear type — tq4 for Gemma, FP32 for compiled cards.
Validated real Gemma 4 E4B attn_q.weight (2560×2048, 20480 tq4 blocks)
padded to (4096, 4096) → dequant → top-left matches original to
atol=1e-5.

## Matvec kernel variants (R53.29-R53.32)

`tq4_matvec_triton` went through four variants. Only v2 ships in
production; others live in tree as null receipts.

| Variant | Technique | Δ vs v1 | Status |
|---|---|---|---|
| v1 | Global-memory gather of centroids via `tl.load(ptr + nybble)` | baseline | Kept for A/B (`tq4_matvec_triton_v1`) |
| **v2** | **Shared-mem LUT: load centroids into program-local `(16,)` tile, use `tl.gather(tile, idx)`** | **-5 to -10% aggregate** | **Production default** (`cbb8073`) |
| v3 | fp16 x_rot activation (halve BW, upcast in dot) | +0.2 / +8.7% | Null (`cfa584f`) |
| v4 | uint32 packed qs loads (32 loads vs 128 per block) | +9.8 / +16.4% | Null (`cfa584f`) |

### Lessons transferred vs not transferred

From TurboQuant CUDA commit `51481c3` (+89% / +45% / etc. on RTX 5090
Blackwell):
- **Transferred**: shared-mem centroid LUT (-7% aggregate in Triton).
  The Triton compiler places the `(16,)` tile in registers/SMEM
  rather than the global-memory gather path.
- **Did NOT transfer**: fp16 activations, uint32 vectorized loads.
  Triton already auto-coalesces on Ada L1 — the hand-tuned CUDA
  techniques overlap with what the Triton backend emits; adding them
  manually introduces `tl.join` / reshape overhead that exceeds the
  BW savings. BLOCK_M sweep (R53.32) — current `_pick_block_m`
  heuristic holds.

## End-to-end decode bench (2026-04-21 clean, median-of-5)

Four paths A/B/C/D per `scripts/bench_decode_paths.py`:

| path | config | tok/s | % llama |
|---|---|---:|---:|
| A | fp16 KV, no graphs | 7.14 | 17% |
| B | tq4 KV, no graphs | 5.56 | 13% |
| C | fp16 KV + CUDA Graphs | 33.35 | 79% |
| D | tq4 KV + CUDA Graphs (`bdf67ee`) | **25.02** | **60%** |

llama.cpp baseline on the same GGUF is ~42 tok/s. Historical
"42 tok/s / 90% llama" claim from session 32 is unreproducible in
current bench — hardware/driver state dependent; reserve for matching
conditions or rebench.

## Fused k+v projection microbench (Track A, 2026-04-21)

`GemmaLayer.attn_kv_fused` via `tq4_linear_dual_triton`. Commits
`da382d7` (shipped) + `f59ae73` (microbench validation).

Microbench (median-of-5 × 200 iters, cudaEvent timing):

| Layer | d_head | sep μs | fuse μs | speedup |
|---:|---:|---:|---:|---:|
| 0 (SWA) | 256 (bpr=1) | 202.96 | 134.38 | **1.51×** |
| 5 (GLB) | 512 (bpr=2) | 177.45 | 102.15 | **1.74×** |
| 23 (GLB) | 512 (bpr=2) | 179.09 | 102.70 | **1.74×** |
| **aggregate** | | **559.50** | **339.24** | **1.65×** (+64.9%) |

Correctness: max \|Δ\| ≤ 1e-6 (FP noise). Per-step save: 73.42 μs × 24
own-KV layers = 1.76 ms/step → **+4.4% e2e projected**. End-to-end
validation unverified — rustc + codex_tui CPU contention at session
end contaminated the D-path bench (21.01 vs clean-baseline 25.02).
Rebench in idle environment pending.

## R53.34 fused flash-attn perf curve (2026-04-20 bench re-run + R14 long-N)

Bench scripts: `scripts/r53_phase2_bench.py` + `scripts/r53_37_long_n_bench.py`.
Initial single-run R53.34 read said "8-10% slower at all N≤1024" and
shipped `_use_fused_flash_attn=False`. A clean re-run showed the curve
is **non-monotonic** — fused has a mid-range sweet spot. R14 (same
session) extended to N=8192 with proper GPU discipline (heavy_warmup
3s + cuda.Event + correctness sanity):

| N | fp16 KV | tq4 memo | tq4 fused | fused/memo | fused/fp16 |
|---:|---:|---:|---:|---:|---:|
| 64 | 6.99 | 4.88 | 4.00 | 0.82× | 57.1% |
| 256 | 6.67 | 5.63 | **6.40** | **1.14×** | **95.9%** |
| 1024 | 7.83 | 6.08 | **6.43** | **1.06×** | 82.1% |
| 4096 | 7.21 | 6.09 | 5.65 | 0.93× | 78.4% |
| 8192 | 5.32 | 4.64 | 4.41 | 0.95× | 83.0% |

R14 confirms the first-principles prediction: memo continues to
dominate fused past N=4K. No asymptotic crossover observed — fused
kernel's fixed 336 per-step Triton launches (42 layers × 8 Q heads)
vs memo's single cuBLAS matmul scaling linearly with N. The runtime
N-gate `128 < cached_kv_len < 2048` is confirmed optimal. Note fp16
itself degrades 26% going 4K→8K (cache memory pressure from
4096-token KV at fp16 ≈ 840 MB); memo/fused degrade in step.

### Two distinct regimes, two distinct bottlenecks

- **Small N (≤128)**: launch overhead dominates. Fused issues 336
  per-Q-head kernel launches per decode step (42 layers × 8 heads);
  at N=64 total step work is ~20-50 ms and the ~1 ms launch overhead
  is a meaningful fraction.
- **Large N (≥4K)**: cuBLAS-on-memoized-fp16 beats Triton streaming.
  Memo amortizes dequant over all subsequent steps (materialize once
  on insertion, reuse via one cuBLAS matmul per layer); cuBLAS is
  near-peak on `(1, 2560) @ (N, 2560)`, fused's per-Q-head Triton
  tiles aren't.

### Caveats (methodology)

Bench is single-run per config (not median-of-5 per `workflow.md`
§"GPU bench discipline"); direction is reliable, magnitudes soft.
Gate thresholds chosen with ~2× safety margin for driver/clock
variance.

### Unlock potential at short N

Would need (a) one Triton kernel spanning all Q heads (remove Python
loop → 1 launch/layer instead of 8), (b) parallel-over-N V kernel
via TILE_N blocking. Both non-trivial; not pursued because the gated
default already captures the measured win.

## Correctness validation

7/7 unit tests cosine=1.0 vs fp32 ref at N∈{16,64,128,256,1024};
real-Gemma ablation Δmean=0.0 argmax=+0.

## tq4_qjl_k256 research artifact — ruled out as default KV

Paper: TurboQuant §3.2. Q_prod = Q_mse(3 bits) + QJL(1-bit residual).
Theoretically unbiased inner-product estimator at the same 4 bpw as
tq4_k256.

Block layout (132 bytes total, matches tq4_k256):
```
struct tq4_qjl_block {
    uint8_t qs_3bit[96];    // 256 codes × 3 bits, packed 8-per-3-bytes
    uint8_t qjl_signs[32];  // 256 QJL sign bits, 8-per-byte
    ggml_half d_mse;        // block L2 norm (MSE stage scale)
    ggml_half d_qjl;        // residual L2 norm (QJL estimator scale)
};
```

Encode: `x → Pi @ x_unit → Q_mse(3 bits) → residual r → sign(S @ r)`
where S is d×d standard-Gaussian JL (seed=137 — distinct from Pi's
seed=42 to avoid degeneracy).

Decode estimator:
```
score ≈ d_mse · (<centroids[codes], Pi @ y>
              + (sqrt(π/2) · d_qjl / d) · <signs, S · Pi @ y>)
```

**Unbiasedness validated**: `test_tq4_qjl_torch.py::test_qjl_unbiasedness_empirical`
— 1.15σ from truth on n=1000 JL-realization samples.

**Deployment verdict**: do NOT use as default KV cache encoding.
Empirical measurement (`test_tq4_qjl_flash_attn.py`) shows QJL has
worse attention-output cosine than plain tq4 at every context tested
(N=16→4096, Δ=-0.02 to -0.17). Softmax non-linearly amplifies QJL's
per-realization variance more than it amplifies MSE-only's small
structural bias. Paper's distortion claim is about expected inner
product MSE, not softmax preservation.

Artifact kept in tree (`calm/llm_computer/tq4_qjl_torch.py`,
`tq4_flash_attn.fused_tq4_qjl_flash_attn_decode`) for future use
cases where unbiased `<x, y>` matters more than softmax —
nearest-neighbor lookup, hash retrieval, cosine-similarity ranking.

Full ruled-out entry: `MEMORY/atlas/tracing_roadmap_part_1.md`
§"TurboQuant Q_prod for KV cache".

## Cross-refs

- Current rules: `.claude/rules/turboquant.md`
- Substrate install pattern: `.claude/rules/Substrate.md`
- Gemma substrate loader integration: `.claude/rules/architecture.md`
- R53 full tracing-arc receipts: `MEMORY/atlas/tracing_roadmap_part_1.md`
