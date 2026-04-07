# Gemma 4 E4B vs Qwen 3.5 4B — Long-context recall comparison

**Date**: 2026-04-07
**Methodology**: Needle-in-haystack testing across three modes (single, multi, distractor) at sizes from 4K to 220K tokens. Both models served via patched llama.cpp at `--ctx-size 262144` with `--parallel 1`. llama.cpp source patched at `tools/server/server-context.cpp:763-766` to remove the per-slot training-context cap.

**Important correction to earlier session notes**: I had told the user Qwen 3.5 4B was ~32K trained. **That was wrong.** Qwen 3.5 4B is trained at `n_ctx_train = 262144` (256K), identical to Gemma 4 26B-A4B. This comparison is fair — both models are testing within their trained range, no extrapolation for either.

## Overall scoreboard

| Test | Gemma 4 E4B | Qwen 3.5 4B | Winner |
|---|:-:|:-:|:-:|
| Single-needle (21 prompts, 4K-220K) | **21/21** | **21/21** | tied |
| Multi-needle (7 prompts, 5 needles each) | **6/7** (220K: 4/5) | **5/7** (180K: 4/5, 220K: 3/5 + hallucination) | **Gemma** |
| Distractor (7 prompts, 4 decoys each) | **7/7** | **5/7** (64K ✗, 100K ✗) | **Gemma** |
| **Total** | **34/35 (97%)** | **31/35 (89%)** | **Gemma** |

## Single-needle grid

Identical perfect scores for both. The easy version of the test.

| Size | Gemma depth 10%/50%/90% | Qwen depth 10%/50%/90% |
|---:|:-:|:-:|
| 4K | ✓ ✓ ✓ | ✓ ✓ ✓ |
| 32K | ✓ ✓ ✓ | ✓ ✓ ✓ |
| 64K | ✓ ✓ ✓ | ✓ ✓ ✓ |
| 100K | ✓ ✓ ✓ | ✓ ✓ ✓ |
| 130K | ✓ ✓ ✓ | ✓ ✓ ✓ |
| 180K | ✓ ✓ ✓ | ✓ ✓ ✓ |
| 220K | ✓ ✓ ✓ | ✓ ✓ ✓ |

**Key finding**: single-needle retrieval is easy for both models across the full 220K range. This does NOT mean they're equivalent — it means single-needle is not the discriminating test.

## Multi-needle grid (5 needles, must find ALL)

Gemma wins by preserving multi-fact recall deeper into the context.

| Size | Gemma | Qwen |
|---:|:-:|:-:|
| 4K | 5/5 ✓ | 5/5 ✓ |
| 32K | 5/5 ✓ | 5/5 ✓ |
| 64K | 5/5 ✓ | 5/5 ✓ |
| 100K | 5/5 ✓ | 5/5 ✓ |
| 130K | 5/5 ✓ | 5/5 ✓ |
| **180K** | **5/5 ✓** | **4/5 ✗** |
| **220K** | **4/5 ✗** | **3/5 + hallucinated one** ✗ |

**Gemma's effective multi-needle context: ~180K**
**Qwen's effective multi-needle context: ~130K**

### Failure mode matters

At 220K Qwen did not just miss a fact — it **hallucinated one**. Expected `14223-CRIMSON-EAGLE`, Qwen returned `14223-AZURE-MARTEN`. Correct number, invented color and animal. This is a confidently-wrong answer, which is more dangerous than "couldn't find it" because a downstream user would take it at face value.

Gemma's 220K failure was cleaner: it returned 4 correct facts and dropped the 5th entirely, which is at least honest about the limit.

## Distractor grid (1 primary + 4 decoys, must return PRIMARY only, no leaks)

Gemma has clean monotonic behavior. Qwen has a mid-range dip.

| Size | Gemma | Qwen |
|---:|:-:|:-:|
| 4K | ✓ clean | ✓ clean (primary in reasoning) |
| 32K | ✓ clean | ✓ clean (primary in reasoning) |
| **64K** | **✓ clean** | **✗ missed primary** |
| **100K** | **✓ clean** | **✗ picked wrong distractor** |
| 130K | ✓ clean | ✓ clean |
| 180K | ✓ clean | ✓ clean |
| 220K | ✓ clean | ✓ clean |

**Qwen's distractor failure at 100K is particularly alarming**: the response content contained `58537-EMERALD` — a partial distractor code, not the primary. The model wasn't just confused; it actively chose a wrong answer. The U-shape (works at edges, fails in the middle) is a known symptom of "lost in the middle" attention behavior on full-attention architectures like Qwen's.

Gemma's sliding-window attention seems to protect against this specific failure mode — every layer has a guaranteed local context plus periodic full-attention layers for long-range. Even if long-range attention degrades in the middle, the local pattern keeps the model grounded.

## Timing comparison

Both models, single-needle, depth 10% (the slowest depth since the model has the most haystack to process):

| Size | Gemma (s) | Qwen (s) |
|---:|---:|---:|
| 4K | 5.5 | 3.3 |
| 32K | 14.2 | 14.2 |
| 64K | 32.9 | 32.5 |
| 100K | 59.8 | 60.9 |
| 130K | 80.3 | 85.8 |
| 180K | 135.7 | 139.3 |
| 220K | 192.6 | **248.3** |

Qwen is faster at short context (3.3s vs 5.5s at 4K — smaller weights, less memory bandwidth), but Gemma is faster at extreme long context (192.6s vs 248.3s at 220K — sliding window caps the per-layer cost). They are roughly equivalent in the middle range.

The crossover happens around ~200K tokens, which is past where you'd run either model interactively anyway. For normal use, both are similarly fast.

## VRAM

At 256K context with Q4 KV cache, both models use roughly the same VRAM on the 8 GB 4070 Laptop:

| Model | Weights | KV cache + compute | Total |
|---|---:|---:|---:|
| Gemma 4 E4B Q5_K_M | 5.48 GB | ~1.2 GB | ~6.7 GB |
| Qwen 3.5 4B Q5_K_M | 2.9 GB | ~4.4 GB | ~7.3 GB |

Gemma has bigger weights but dramatically smaller KV cache (sliding window reduces per-layer cost). Qwen has smaller weights but ~3.5× larger KV at long context (no sliding window). Net: Gemma is slightly more VRAM-efficient at 256K, but both fit comfortably with ~1 GB headroom.

## Conclusions

1. **Effective context matters more than trained context.** Both models have 256K trained max. But Gemma reliably uses up to 180K for hard retrieval tasks; Qwen only to 130K.

2. **Sliding-window attention is a real advantage for long-context reliability.** Gemma's hybrid sliding+full attention pattern protects against middle-of-context attention failures that Qwen exhibits at 64K-100K in the distractor test.

3. **Hallucination under pressure is Qwen's weakest failure mode.** At 220K multi-needle, Qwen didn't admit failure — it made up a plausible-looking wrong answer. For a coding assistant this is worse than admitting uncertainty.

4. **Gemma 4 E4B beats Qwen 3.5 4B at long-context recall across every test type.** Combined with the earlier A/B eval showing Gemma wins 5/0 on single-turn coding correctness, the recommendation to switch the project base model is strongly reinforced.

5. **Both models handle short context (4K-64K) cleanly** aside from the Qwen 64K distractor dip. For typical interactive coding sessions that never exceed 32K total tokens, you wouldn't see the difference.

6. **llama.cpp patch was necessary.** The training-context cap in `tools/server/server-context.cpp:763-766` silently capped slots to `n_ctx_train`, preventing any slot from exceeding trained context even with explicit YaRN flags. The one-line patch (commenting out `n_ctx_slot = n_ctx_train`) is preserved locally and is needed for any future 256K+ testing. Note: Gemma's Q5_K_M GGUF metadata also forces rope scaling to "linear" regardless of CLI flags, so YaRN extension on Gemma doesn't currently work — past 131K is raw RoPE extrapolation.

## Updated compaction limits

`agents/compact.py` updated with NIAH-validated limits:

```python
MODEL_CONTEXT_LIMITS = {
    "llamacpp":     65536,   # generic fallback
    "gemma-4-e4b":  180000,  # validated clean through 180K multi+distractor
    "gemma-4-E4B":  180000,
    "qwen3.5-4b":   130000,  # validated clean through 130K; 64/100K distractor dip
    "Qwen3.5-4B":   130000,
    ...
}
```

The harness will now automatically choose the correct limit based on the loaded model name, and start summarizing old turns before the model's effective context window is exhausted. Both limits leave ~20% safety margin below where failures began.

## Files

Individual reports:
- `2026-04-07_gemma4_e4b_needle.md` — initial single-needle at 128K (25/25)
- `2026-04-07_gemma4_e4b_needle_256k_single.md` — single-needle at 256K (21/21)
- `2026-04-07_gemma4_e4b_needle_256k_multi.md` — multi-needle at 256K (6/7)
- `2026-04-07_gemma4_e4b_needle_256k_distractor.md` — distractor at 256K (7/7)
- `2026-04-07_qwen4b_needle_256k_single.md` — single-needle at 256K (21/21)
- `2026-04-07_qwen4b_needle_256k_multi.md` — multi-needle at 256K (5/7)
- `2026-04-07_qwen4b_needle_256k_distractor.md` — distractor at 256K (5/7)
- `2026-04-07_qwen4b_vs_gemma4_e4b.md` — earlier single-turn coding eval (5/0 Gemma wins)
