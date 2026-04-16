# TurboQuant + Quantization Rules

## TurboQuant Types

| Type | bpw | Block | Levels | Use |
|------|-----|-------|--------|-----|
| tq3_k256 | 3.06 | 98 B | 8 | KV cache only |
| tq3_k512 | 3.03 | 194 B | 8 | KV cache (head_dim=512) |
| tq4_k256 | 4.125 | 132 B | 16 | **Weights + KV cache** |

tq4 is the recommended type. 132-byte blocks (128 qs + 2 d + 2 pad)
for 4-byte aligned CUDA loads. Pi rotation (seed=42, 256×256 orthogonal),
16-level Lloyd-Max codebook for N(0, 1/√256).

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

## Quantization Commands

```bash
# tq4 quantization
setsid ~/llama.cpp/build/bin/llama-quantize \
  model-F16.gguf model-tq4.gguf TQ4_K256 4 < /dev/null > /tmp/q.log 2>&1 &
disown -a

# ALWAYS: setsid + disown (survive CC crashes), 4 threads, < /dev/null
# ALWAYS: verify output: xxd model.gguf | head -1 (must show 'GGUF')
```
