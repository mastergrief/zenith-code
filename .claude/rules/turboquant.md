---
paths:
  - "calm/llm_computer/tq4*.py"
  - "calm/llm_computer/q6k_dequant.py"
  - "calm/llm_computer/tests/test_tq4*.py"
  - "calm/llm_computer/tests/test_kvcache_tq4*.py"
  - "scripts/bench_tq4*.py"
  - "scripts/test_tq4*.py"
  - "scripts/sweep_tq4_block_m.py"
  - "scripts/llama_cpp_patches/**"
  - "RESEARCH/TQ/**"
---

# TurboQuant + Quantization Rules

> Historical receipts (per-kernel bench tables, matvec variant A/B,
> fused flash-attn full perf curve, CUDA-vs-Triton lesson-transfer
> analysis): see `MEMORY/atlas/turboquant_arc.md`.

## TurboQuant Types

| Type | bpw | Block | Levels | Use |
|------|-----|-------|--------|-----|
| tq3_k256 | 3.06 | 98 B | 8 | KV cache only |
| tq3_k512 | 3.03 | 194 B | 8 | KV cache (head_dim=512) |
| tq4_k256 | 4.125 | 132 B | 16 | **Weights + KV cache** |
| tq4_qjl_k256 | 4.125 | 132 B | 8 + 1-bit | Research artifact (NN lookup / cosine ranking use cases — NOT default) |

tq4 is the recommended type. 132-byte blocks (128 qs + 2 d + 2 pad)
for 4-byte aligned CUDA loads. Pi rotation (seed=42, 256×256 orthogonal),
16-level Lloyd-Max codebook for N(0, 1/√256).

**Rule**: use tq4_k256 (Q_mse-only) as default KV encoding. Do NOT
substitute tq4_qjl_k256 even though both are 4 bpw — QJL has worse
attention-output cosine than plain tq4 at every measured context
because softmax amplifies per-realization variance. QJL is only
correct when expected inner-product MSE matters and softmax doesn't
(e.g. hash retrieval, cosine-similarity ranking).

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

Python: `calm/llm_computer/tq4_torch.py` (`quantize_tq4`, `dequantize_tq4`,
`Tq4Tensor`). C reference Pi loader: `calm/llm_computer/tq4_pi_loader.py`
(bit-exact from `turboquant_tables.h`).

## Q6_K Format (token_embd)

```
struct block_q6_K {      // 210 bytes, 256 elements
    uint8_t ql[128];     // quants, low 4 bits
    uint8_t qh[64];      // quants, high 2 bits
    int8_t  scales[16];  // per-sub-block scales (signed)
    ggml_half d;         // super-block scale
};
```

Dequant per element: `value = d * scales[sub_block] * ((ql_4bit | (qh_2bit << 4)) - 32)`

Python: `calm/llm_computer/q6k_dequant.py` — vectorized PyTorch port
of llama.cpp's `dequantize_row_q6_K`.

## Hybrid Substrate: FP32 + tq4 per-layer

**Critical rule**: tq4 quantization DESTROYS compiled card weights.
Lloyd-Max codebook is tuned for Gaussian LM weights; compiled cards
have discrete integer coefs (±1, ±16). Measured loss is catastrophic
through tq4 roundtrip.

**Fix**: `HybridGroupedSmall2DTransformer` with per-layer linear type:
- **tq4 layers** (`Tq4LinearGGMLOriented`): the base model's attention + FFN
- **FP32 layers** (`FP32LinearGGMLOriented`): compiled cards + HRMs

Both share `y = x @ W` GGML convention. Forward loop unchanged.
Compiled card accuracy preserved through hybrid.

**NEVER quantize compiled card weights to tq4.** Use hybrid per-layer
dispatch instead.

## GGUF Loader

`calm/llm_computer/tq4_gguf_loader.py`:
- Monkey-patches `gguf` library for custom TurboQuant types (TQ3=42, TQ4=44)
- `extract_tq4_tensor(reader, name)` → `Tq4Tensor`
- `extract_fp_tensor(reader, name)` → `torch.Tensor` (F16/F32)
- Q6_K: use `q6k_dequant.extract_q6_k_tensor` instead

## Byte-Level Install (Zero Drift)

`Tq4LinearGGMLOriented` stores weight in GGUF's (in, out) orientation.
Forward: `y = x @ W` (not `x @ W.T`). Byte-compatible with GGUF.

`pad_tq4_tensor_rows_and_cols()` extends with zero blocks (qs=0, d=0
→ dequant exactly to zero). Preserves all original bytes bit-for-bit.

## VRAM Budget (RTX 4070, 8 GB)

| Config | Weights | KV cache | Total |
|--------|---------|----------|-------|
| tq4 + tq4 KV (production) | ~5.0 GB | ~2.0 GB | ~7.0 GB |
| Q5_K_M + f16 KV | 5.48 GB | ~4.0 GB | ~9.5 GB (OOM) |
| Hybrid substrate (2 tq4 + 2 fp32) | varies | N/A | ~50 MB - 10 GB |
| **Prod substrate (Triton stack)** | **~3.5 GB tq4 + Q6_K** | tq4/FP16 | **~5.0 GB baseline** |

Substrate baseline leaves ~3 GB headroom for FP32 hosting layers (see
`MEMORY/substrate_registry.md` for per-card budget) plus activations +
KV cache.

## Triton Kernels (`calm/llm_computer/tq4_triton.py`)

The hot path. PyTorch dequant materialized full FP32 W per call —
bandwidth-bound. Triton kernels stream tq4 bytes directly into the
dot product, never materializing W.

Math: `y = x @ W` where `W = (centroids[codes] @ Pi) * d`. Pi is
orthogonal so `y = (x @ Pi.T) @ (centroids[codes] * d)`. Kernels take
pre-rotated `x_rot` and un-rotated centroid weights. Bit-equivalent to
PyTorch path (max abs diff ~6e-8).

| Kernel | Use |
|---|---|
| `tq4_matvec_triton` | single x vector (decode path); dispatches v2 shared-mem LUT |
| `tq4_matmul_triton` | batched x (S>1, prefill path) — 2D grid `(out_tiles, n_seq)`, 1 launch instead of S |
| `tq4_linear_dual_triton` | gate+up share x — fused dual kernel (half the Python overhead) |
| `q6k_matvec_triton` | output head (262K vocab Q6_K dequant + matmul) |
| `q6k_lookup_triton` | single-token Q6_K embedding lookup |

**Matvec production variant**: v2 (shared-memory LUT via `tl.gather`
on a program-local `(16,)` tile of centroids). Other variants (fp16
activation, uint32 packed qs loads) kept in tree as A/B baselines
but nulled — Triton auto-coalesces on Ada L1 and hand-tuned CUDA
techniques don't transfer cleanly.

**`_pick_block_m(out_features)` heuristic**: BLOCK_M=64 for
`out >= 4096`, 32 for 2048, 16 for 1024, 4 for 512, 1 otherwise.
Dual kernel caps BLOCK_M at 32 (register pressure with two weight
matrices).

**Toggle**: `enable_triton_tq4(True)` (module-level) or `--triton`
CLI flag on `gemma_substrate.py`.

## Graph-captured tq4 KV decode

Enabled by two additions to `gemma_substrate.py`:

1. **`KVCacheTq4Static`** (graph-safe) — shared `pos_t: (n_layers,)`
   long tensor, `valid_mask_all: (n_layers, max_len)` bool, per-layer
   `_bpr_offsets`. Writes via `index_copy_` (graph-safe) at
   `_bpr_offsets + pos_t*bpr`; attention reads full pre-allocated
   `max_len` with additive valid-mask — no Python-int slicing inside
   the graph.
2. **`generate_with_graph_tq4()`** — prefill on dynamic `KVCacheTq4`,
   byte-copy transfer into static, 3-iter side-stream warmup, capture
   one-step graph, replay `max_tokens-1` times.
   **Warmup-slot invariant**: warmup writes same bytes graph will
   write at prefill_len (else capture bakes a wrong dependency).

Pi bpr=2 extension in `fused_tq4_flash_attn_decode` supports global
layers (d_head=512) via `q_2d.reshape(H, bpr, 256) @ pi` reshape.

Static-path fused flash-attn is **always-on** for graph capture —
bypasses the dynamic N-gate the non-static path uses. A bpr=2
regression would therefore only surface under graph decode.

## Fused k+v tq4 projection

`GemmaLayer.attn_kv_fused` routes K and V projections through **one
Triton launch** (`tq4_linear_dual_triton`, same primitive behind
gate+up FFN fusion). Falls back to separate `ak(x), av(x)` when
`_gpu_qs` unset, Pi buffer unset, or `_use_triton=False`.

## Fused flash-attention decode with tq4 K/V

File: `calm/llm_computer/tq4_flash_attn.py`. Entry:
`fused_tq4_flash_attn_decode(q_rot, k_qs, k_d, v_qs, v_d, centroids,
pi, attn_mask, softcap=0.0)`. Decode-only (S_q=1).

**Why it exists**: dynamic-KV decode dequanted the full cached
sequence on every call — O(N) per step, O(N²) across full decode.
Fused kernel bypasses the materialization.

**Storage contract** (head-major):
```
K.qs: (n_heads_kv, N * bpr, 128) uint8 — contiguous per head
K.d:  (n_heads_kv, N * bpr) fp32
V.qs: same layout
V.d:  same layout
```
`bpr = d_head // 256` (= 1 at d_head=256).

**Math** (Pi orthogonal cancels in inner products):
```
scores[h, n] = (Pi @ Q[h]) · (Pi @ K[kv_h, n])       = Q · K
            = q_rot[h] @ k_dequant_rotated[kv_h, n]
out[h]       = sum_n softmax(scores[h])[n] * V[kv_h, n]
            = Pi.T @ (sum_n p[n] * v_dequant_rotated[kv_h, n])
```

K-side scoring runs against rotated K codes directly (skip per-block
Pi.T inside kernel — saves 1 matmul/block). V-side weighted sum is
post-rotated once per head outside the kernel.

**Two kernels**:
- **K side**: reuses `tq4_matvec_triton` (K layout `(N, d_head)`
  matches the kernel's `(out_features=N, in_features=d_head)`
  contract). One call per Q head.
- **V side**: new `_tq4_weighted_v_kernel` with `grid=(n_heads_q,)`
  — one program per Q head, streams N tq4 V blocks, accumulates fp32
  `(D_HEAD,)` in registers. Output Pi-rotated; caller applies Pi.T
  once per head.

**Wiring**: `generate(use_tq4_kv=True)` allocates `KVCacheTq4` sized
to `len(prompt) + max_tokens`. Gated by `enable_fused_flash_attn()`
flag, with fused kernel used for `KVCacheTq4 AND S==1 AND d_head==256
AND not partitions`. Global layers (d_head=512) fall back to Phase 1
memoized dequant path.

**Shipped policy**: `_use_fused_flash_attn=True` default, with
runtime conditional in `_forward_layer` gating on
`128 < kv_cache.layer_pos[kv_src] < 2048`. Inside the gate fused
runs; outside it falls back to memo. Rationale: launch overhead
dominates at small N (≤128), cuBLAS-on-memoized-fp16 dominates at
large N (≥4K), fused wins the middle band.

**Realistic workload coverage**:
- Chat turns (50-500 decode tok): entirely in gate → captures decode
  speedup vs memo-only.
- Short eval problems: mostly in gate.
- Long eval (AdaptiveBudget up to 16K): first ~2K steps in gate,
  then falls back to memo for asymptotic regime → no regression on
  long-decode workloads.

Disable via `enable_fused_flash_attn(False)`.

## Quantization Commands

```bash
# In a dedicated shell/session, run quantization as the foreground process
# and write its output to a log.
~/llama.cpp/build/bin/llama-quantize \
  model-F16.gguf model-tq4.gguf TQ4_K256 4 > /tmp/q.log 2>&1

# Then arm Monitor on the log.
Monitor(command="bin/watch-wrap --log /tmp/q.log --heartbeat 180 --error 'Traceback|Error|Killed|OOM|FAILED|assert' --progress 'quant|tensor|write|%' --success 'done|complete|finished' --stop-on 'done|complete|finished' --replay 20")

# ALWAYS: foreground shell/session + Monitor; 4 threads.
# ALWAYS: verify output: xxd model.gguf | head -1 (must show 'GGUF')
```

## Related rules

- `Substrate.md` — hybrid per-layer install pattern
- `architecture.md` — substrate loader integration
- `MEMORY/atlas/turboquant_arc.md` — per-kernel bench receipts + CUDA-vs-Triton lesson-transfer analysis
