# TurboQuant + Quantization Rules

## TurboQuant Types

| Type | bpw | Block | Levels | Use |
|------|-----|-------|--------|-----|
| tq3_k256 | 3.06 | 98 B | 8 | KV cache only |
| tq3_k512 | 3.03 | 194 B | 8 | KV cache (head_dim=512) |
| tq4_k256 | 4.125 | 132 B | 16 | **Weights + KV cache** |
| tq4_qjl_k256 | 4.125 | 132 B | 8 + 1-bit | Research artifact — see below |

tq4 is the recommended type. 132-byte blocks (128 qs + 2 d + 2 pad)
for 4-byte aligned CUDA loads. Pi rotation (seed=42, 256×256 orthogonal),
16-level Lloyd-Max codebook for N(0, 1/√256).

### tq4_qjl_k256 — inner-product-optimal variant (research only)

Paper: TurboQuant §3.2. Q_prod = Q_mse(3 bits) + QJL(1-bit residual).
Theoretically unbiased inner-product estimator at the same 4 bpw as
tq4_k256. Block layout (132 bytes total, matches tq4_k256):

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

**Deployment verdict: do NOT use as default KV cache encoding.**
Empirical measurement (`test_tq4_qjl_flash_attn.py`) shows QJL has
worse attention-output cosine than plain tq4 at every context tested
(N=16→4096, Δ=-0.02 to -0.17). Softmax non-linearly amplifies QJL's
per-realization variance more than it amplifies MSE-only's small
structural bias. Paper's distortion claim is about expected inner
product MSE, not softmax preservation. Full ruled-out entry:
`.claude/rules/tracing_roadmap.md` §"TurboQuant Q_prod for KV cache".

Artifact kept in tree (`calm/llm_computer/tq4_qjl_torch.py`,
`tq4_flash_attn.fused_tq4_qjl_flash_attn_decode`) for future use cases
where unbiased `<x, y>` matters more than softmax — nearest-neighbor
lookup, hash retrieval, cosine-similarity ranking.

## tq4 Block Format

```
struct tq4_block {       // 132 bytes, 256 elements
    uint8_t qs[128];     // 2 nybble codes per byte
    ggml_half d;         // L2 norm (block scale)
    uint8_t pad[2];      // alignment padding
};
```

Quantize: `x → normalize → rotate by Pi → quantize to 16 levels → pack nybbles`
Dequant: `unpack → lookup centroids → inverse rotate by Pi.T → scale by d`

Python: `calm/llm_computer/tq4_torch.py` (quantize_tq4, dequantize_tq4, Tq4Tensor)
C reference Pi loader: `calm/llm_computer/tq4_pi_loader.py` (bit-exact from turboquant_tables.h)

## Q6_K Format (for Gemma token_embd)

```
struct block_q6_K {      // 210 bytes, 256 elements
    uint8_t ql[128];     // quants, low 4 bits
    uint8_t qh[64];      // quants, high 2 bits
    int8_t  scales[16];  // per-sub-block scales (signed)
    ggml_half d;         // super-block scale
};
```

Dequant per element: `value = d * scales[sub_block] * ((ql_4bit | (qh_2bit << 4)) - 32)`

Python: `calm/llm_computer/q6k_dequant.py` — vectorized PyTorch port of
llama.cpp's `dequantize_row_q6_K`. Loads Gemma's token_embd (262144 ×
2560) in ~110s on CPU. Stats match: std 0.024, range [-0.54, 0.55].

## Hybrid Substrate: FP32 + tq4 Per-Layer

**Critical finding (Round 11)**: tq4 quantization DESTROYS compiled card
weights. Lloyd-Max codebook is tuned for Gaussian LM weights; compiled
cards have discrete integer coefs (±1, ±16). Measured: dispatched_v4
791/791 → 70/791 (91% loss) through tq4 roundtrip.

**Fix**: `HybridGroupedSmall2DTransformer` with per-layer linear type:
- **tq4 layers** (`Tq4LinearGGMLOriented`): Gemma's attention + FFN
- **FP32 layers** (`FP32LinearGGMLOriented`): compiled cards + HRMs

Both share `y = x @ W` GGML convention. Forward loop unchanged.
Compiled card accuracy preserved at 791/791 through hybrid.

**NEVER quantize compiled card weights to tq4.** Use hybrid per-layer
dispatch instead.

## GGUF Loader

`calm/llm_computer/tq4_gguf_loader.py`:
- Monkey-patches `gguf` library for custom TurboQuant types (TQ3=42, TQ4=44)
- `extract_tq4_tensor(reader, name)` → Tq4Tensor
- `extract_fp_tensor(reader, name)` → torch.Tensor (F16/F32)
- Q6_K: use `q6k_dequant.extract_q6_k_tensor` instead

## Byte-Level Install (Zero Drift)

`Tq4LinearGGMLOriented` stores weight in GGUF's (in, out) orientation.
Forward: `y = x @ W` (not `x @ W.T`). Byte-compatible with GGUF.

`pad_tq4_tensor_rows_and_cols()` extends with zero blocks (qs=0, d=0 →
dequant exactly to zero). Preserves all original bytes bit-for-bit.

Validated: real Gemma 4 E4B attn_q.weight (2560×2048, 20480 tq4 blocks)
padded to (4096, 4096) → dequant → top-left matches original to atol=1e-5.

## VRAM Budget (RTX 4070, 8 GB)

| Config | Weights | KV cache | Total |
|--------|---------|----------|-------|
| tq4 + tq4 KV (production) | ~5.0 GB | ~2.0 GB | ~7.0 GB |
| Q5_K_M + f16 KV | 5.48 GB | ~4.0 GB | ~9.5 GB (OOM) |
| Hybrid substrate (2 tq4 + 2 fp32) | varies | N/A | ~50 MB - 10 GB |
| **Prod Gemma substrate (Triton stack)** | **~3.5 GB tq4 + Q6_K** | tq4/FP16 | **~5.0 GB baseline** |

Substrate baseline of ~5.0 GB leaves ~3 GB headroom for FP32 hosting layers
(see `.claude/MEMORY/substrate_registry.md` for budget table) plus
activations + KV cache.

## Triton Kernels (`calm/llm_computer/tq4_triton.py`)

The hot path. PyTorch dequant materialized full FP32 W per call (~26M
elements for ffn_up) — bandwidth-bound at ~6.8 ms per linear, ~3 sec
per token across 378 calls. Triton kernels stream tq4 bytes directly
into the dot product, never materializing W.

Math: `y = x @ W` where `W = (centroids[codes] @ Pi) * d`. Pi is
orthogonal so `y = (x @ Pi.T) @ (centroids[codes] * d)`. Kernels take
pre-rotated `x_rot` and the un-rotated centroid weights. Bit-equivalent
to the PyTorch path (max abs diff ~6e-8).

| Kernel | Use | Speedup |
|---|---|---|
| `tq4_matvec_triton` | single x vector (decode path); dispatches v2 | 5-17× per linear |
| `tq4_matmul_triton` | batched x (S>1, prefill path) — 2D grid `(out_tiles, n_seq)` | 1 launch instead of S |
| `tq4_linear_dual_triton` | gate+up share x — fused dual kernel | half the Python overhead |
| `q6k_matvec_triton` | output head (262K vocab Q6_K dequant + matmul) | 125× vs chunked |
| `q6k_lookup_triton` | single-token Q6_K embedding lookup | minor; saves Python ops |

### Matvec kernel variants (R53.29-R53.32)

`tq4_matvec_triton` went through four variants this session. Only
v2 ships in production; others live in tree as null receipts.

| Variant | Technique | Δ vs v1 | Status |
|---|---|---|---|
| v1 | Global-memory gather of centroids via `tl.load(ptr + nybble)` | baseline | Kept for A/B (`tq4_matvec_triton_v1`) |
| **v2** | **Shared-mem LUT: load centroids into program-local `(16,)` tile, use `tl.gather(tile, idx)`** | **-5 to -10% aggregate** | **Production default** (cbb8073) |
| v3 | fp16 x_rot activation (halve BW, upcast in dot) | +0.2/+8.7% | Null (cfa584f) |
| v4 | uint32 packed qs loads (32 loads vs 128 per block) | +9.8/+16.4% | Null (cfa584f) |

**Lessons transferred vs not transferred** from TurboQuant CUDA
commit 51481c3 (+89% / +45% / etc. on RTX 5090 Blackwell):
- **Transferred**: shared-mem centroid LUT (-7% aggregate in Triton).
  The Triton compiler places the `(16,)` tile in registers/SMEM
  rather than the global-memory gather path.
- **Did NOT transfer**: fp16 activations, uint32 vectorized loads.
  Triton already auto-coalesces on Ada L1 — the hand-tuned CUDA
  techniques overlap with what the Triton backend emits; adding
  them manually introduces `tl.join`/reshape overhead that exceeds
  the BW savings. BLOCK_M sweep (R53.32) — current
  `_pick_block_m` heuristic holds.

`_pick_block_m(out_features)` heuristic: BLOCK_M=64 for `out >= 4096`,
32 for 2048, 16 for 1024, 4 for 512, 1 otherwise. Dual kernel caps
BLOCK_M at 32 (register pressure with two weight matrices).

Toggle via `enable_triton_tq4(True)` (module-level) or `--triton`
CLI flag on `gemma_substrate.py`. End-to-end on RTX 4070M with
gemma-4-E4B-it-tq4 (2026-04-21 clean bench, median-of-5, four
paths A/B/C/D per `scripts/bench_decode_paths.py`):

| path | config | tok/s | % llama |
|---|---|---:|---:|
| A | fp16 KV, no graphs | 7.14 | 17% |
| B | tq4 KV, no graphs | 5.56 | 13% |
| C | fp16 KV + CUDA Graphs | 33.35 | 79% |
| D | tq4 KV + CUDA Graphs (`bdf67ee`) | **25.02** | **60%** |

llama.cpp baseline on the same GGUF is ~42 tok/s. Historical
"42 tok/s / 90% llama" claim from session 32 is unreproducible
in current bench — hardware/driver state dependent; reserve for
matching conditions or rebench. See `.claude/rules/architecture.md`
"Gemma substrate loader" for the full perf chain.

## Graph-captured tq4 KV decode (Track A, 2026-04-21)

The D-path (tq4+graphs, 25.02 tok/s) is enabled by two additions
to `gemma_substrate.py`:

1. **`KVCacheTq4Static`** (graph-safe) — shared `pos_t: (n_layers,)`
   long tensor, `valid_mask_all: (n_layers, max_len)` bool,
   per-layer `_bpr_offsets`. Writes via `index_copy_` (graph-safe)
   at `_bpr_offsets + pos_t*bpr`; attention reads the full
   pre-allocated `max_len` with an additive valid-mask — no
   Python-int slicing inside the graph.
2. **`generate_with_graph_tq4()`** — prefill on dynamic
   `KVCacheTq4`, byte-copy transfer into static, 3-iter side-stream
   warmup, capture one-step graph, replay `max_tokens-1` times.
   Warmup-slot invariant: warmup writes same bytes graph will write
   at prefill_len (else capture would bake a wrong dependency).

Pi bpr=2 extension in `fused_tq4_flash_attn_decode` — global
layers (d_head=512) are now supported via `q_2d.reshape(H, bpr,
256) @ pi` reshape around the Pi-unrotate.

Static-path fused flash-attn is **always-on** for graph capture —
bypasses the dynamic `128 < layer_pos < 2048` bench gate the
non-static path uses. A bpr=2 regression would therefore only
surface under graph decode.

## Fused k+v tq4 projection (Track A, 2026-04-21)

`GemmaLayer.attn_kv_fused` routes K and V projections through
**one Triton launch** (`tq4_linear_dual_triton`, same primitive
behind gate+up FFN fusion). Commits `da382d7` (shipped) +
`f59ae73` (microbench validation).

Microbench (median-of-5 × 200 iters, cudaEvent timing):

| Layer | d_head | sep μs | fuse μs | speedup |
|---:|---:|---:|---:|---:|
| 0 (SWA) | 256 (bpr=1) | 202.96 | 134.38 | **1.51×** |
| 5 (GLB) | 512 (bpr=2) | 177.45 | 102.15 | **1.74×** |
| 23 (GLB) | 512 (bpr=2) | 179.09 | 102.70 | **1.74×** |
| **aggregate** | | **559.50** | **339.24** | **1.65×** (+64.9%) |

Correctness: max \|Δ\| ≤ 1e-6 (FP noise). Per-step save:
73.42 μs × 24 own-KV layers = 1.76 ms/step → **+4.4% e2e
projected**. End-to-end validation is **unverified** — rustc +
codex_tui CPU contention at session end contaminated the D-path
bench (21.01 vs clean-baseline 25.02). Rebench in idle
environment pending.

Invariant: fallback to separate `ak(x), av(x)` when `_gpu_qs`
unset, Pi buffer unset, or `_use_triton=False`.

## Fused flash-attention decode with tq4 K/V (R53.34)

File: `calm/llm_computer/tq4_flash_attn.py`. Entry point:
`fused_tq4_flash_attn_decode(q_rot, k_qs, k_d, v_qs, v_d, centroids,
pi, attn_mask, softcap=0.0)`. Decode-only (S_q=1) MVP.

**Why it exists**: pre-R53.34 `KVCacheTq4.update()` dequanted the
full cached sequence on every call. Per decode step this is O(N)
work; across a 500-token decode that's O(N²) dequant across 42
layers = ~10M block dequants per generation. The fused kernel
bypasses the dequant materialization entirely.

**Storage contract** (head-major — required by the fused kernel,
differs from the interleaved token-major storage of the pre-R53.34
`KVCacheTq4`):

```
K.qs: (n_heads_kv, N * bpr, 128) uint8 — contiguous per head
K.d:  (n_heads_kv, N * bpr) fp32
V.qs: same layout
V.d:  same layout
```

where `bpr = d_head // 256` (= 1 for Gemma E4B d_head=256). For
d_head=256 each position is exactly one tq4 block per head.

**Math** (Pi orthogonal cancels in inner products):
```
scores[h, n] = (Pi @ Q[h]) · (Pi @ K[kv_h, n])       = Q · K
            = q_rot[h] @ k_dequant_rotated[kv_h, n]
out[h]       = sum_n softmax(scores[h])[n] * V[kv_h, n]
            = Pi.T @ (sum_n p[n] * v_dequant_rotated[kv_h, n])
```

So K-side scoring runs against the rotated K codes directly (skip
per-block Pi.T inside the kernel — saves 1 matmul per block), and
V-side weighted sum is post-rotated once per head outside the kernel.

**Two kernels**:
- **K side**: reuses existing `tq4_matvec_triton` (K layout
  (N, d_head) matches the kernel's `(out_features=N,
  in_features=d_head)` contract). One call per Q head.
- **V side**: new `_tq4_weighted_v_kernel` with `grid=(n_heads_q,)`
  — one program per Q head, streams N tq4 V blocks, accumulates
  fp32 `(D_HEAD,)` in registers. Output in Pi-rotated domain;
  caller applies Pi.T once per head (cheap D_HEAD×D_HEAD matmul).

**Wiring**: `generate(use_tq4_kv=True)` allocates `KVCacheTq4`
sized to `len(prompt) + max_tokens`. Gated by
`enable_fused_flash_attn()` flag, with the fused kernel used for
`KVCacheTq4 AND S==1 AND d_head==256 AND not partitions`. Global
layers (d_head=512) fall back to the Phase 1 memoized dequant path.

**Correctness**: 7/7 unit tests cosine=1.0 vs fp32 ref at
N∈{16,64,128,256,1024}; real-Gemma ablation Δmean=0.0 argmax=+0.

**Perf (2026-04-20 bench re-run, `scripts/r53_phase2_bench.py`
+ R14 long-N via `scripts/r53_37_long_n_bench.py`)**:
the initial R53.34 single-run read said "8-10% slower at all
N≤1024" and shipped `_use_fused_flash_attn=False`. A clean re-run
showed the curve is **non-monotonic** — fused has a mid-range
sweet spot. R14 (same session) extended to N=8192 with proper GPU
discipline (heavy_warmup 3s + cuda.Event + correctness sanity):

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
N-gate `128 < cached_kv_len < 2048` is confirmed optimal; no change
needed. Note fp16 itself degrades 26% going 4K→8K (cache memory
pressure from 4096-token KV at fp16 ≈ 840 MB); memo/fused degrade
in step.

**Two distinct regimes, two distinct bottlenecks:**
- Small N (≤128): launch overhead dominates. Fused issues 336
  per-Q-head kernel launches per decode step (42 layers × 8 heads);
  at N=64 total step work is ~20-50 ms and the ~1 ms launch
  overhead is a meaningful fraction.
- Large N (≥4K): cuBLAS-on-memoized-fp16 beats Triton streaming.
  Memo amortizes dequant over all subsequent steps (materialize
  once on insertion, reuse via one cuBLAS matmul per layer);
  cuBLAS is near-peak on `(1, 2560) @ (N, 2560)`, fused's
  per-Q-head Triton tiles aren't.

**Shipped policy (as of 2026-04-20)**: `_use_fused_flash_attn=True`
default, with runtime conditional in `_forward_layer` gating on
`128 < kv_cache.layer_pos[kv_src] < 2048`. Inside the gate fused
runs; outside it falls back to Phase 1 memo. Caveats: bench is
single-run per config (not median-of-5 per `workflow.md` §"GPU
bench discipline"); direction is reliable, magnitudes soft. Gate
thresholds chosen with ~2× safety margin for driver/clock variance.
Disable everything via `enable_fused_flash_attn(False)`.

**Realistic workload coverage**:
- Chat turns (50-500 decode tok): entirely in the gate → captures
  6-14% decode speedup vs memo-only.
- Short eval problems: mostly in the gate.
- Long R53 eval (AdaptiveBudget up to 16K): first ~2K steps in
  gate, then falls back to memo for the asymptotic regime → no
  regression on long-decode workloads.

**Unlock potential at short N would need**: (a) one Triton kernel
spanning all Q heads (remove Python loop → 1 launch/layer instead
of 8), (b) parallel-over-N V kernel via TILE_N blocking. Both
non-trivial; not pursued because the gated default already
captures the measured win.

See `tracing_roadmap.md` ruled-out row (Round 53.34) for full
A/B receipt. Adjacent null: TurboQuant Q_prod (3-bit Q_mse +
1-bit QJL encoding) — implemented in `tq4_qjl_torch.py`, proven
unbiased inner-product estimator, but empirical attention-output
cosine WORSE than tq4 Q_mse alone at every N tested. Kept in
tree (`fused_tq4_qjl_flash_attn_decode`) for future nearest-
neighbor / retrieval use cases where unbiased <x,y> matters more
than softmax output.

## Quantization Commands

```bash
# tq4 quantization
setsid ~/llama.cpp/build/bin/llama-quantize \
  model-F16.gguf model-tq4.gguf TQ4_K256 4 < /dev/null > /tmp/q.log 2>&1 &
disown -a

# ALWAYS: setsid + disown (survive CC crashes), 4 threads, < /dev/null
# ALWAYS: verify output: xxd model.gguf | head -1 (must show 'GGUF')
```
