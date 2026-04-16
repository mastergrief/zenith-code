# Session Handoff — 2026-04-16 (Session 31)

Branch: `feature/multi-agent-qwen`
Prior handoff (session 30): ended at `1a7da0e`.
This session: **14 commits, ~5,440 lines added**, invented the Pointer
Transducer architecture, validated across 7 domains, built substrate-
native Gemma inference from GGUF, and established the output-language
family principle for domain scaling.

## Goal

Fix HRM data distribution (0% on single-digit operands), generalize
across domains, then build substrate-native inference that replaces
llama-server. Evolved into the most productive session to date:
new architecture (pointer-copy), new scaling principle (output-language
families), new domains (creative writing, reasoning), and a full
PyTorch Gemma loader from GGUF.

## Completed (14 commits)

### Pointer Transducer architecture (`calm/llm_computer/copy_augmented.py`)

**The key invention.** `CopyAugmentedTransformer` subclasses `Small2DTransformer`,
adds learned copy gate + pointer attention. 1,089 extra params (0.6%).
At each decode step: `p_copy * P_copy + (1-p_copy) * P_gen`. Digits →
copy from input, operators → generate from vocabulary.

- Copy gate bias initialized at -2.0 (starts preferring generation)
- Forward returns **log-probs** (not logits) — use `F.nll_loss`
- Substrate-native: same d_head=2 invariant, same `.pt` format

### Cross-domain PT validation (7 checkpoints)

| Checkpoint | Domain | Val autoreg | Held-out | Time |
|---|---|---|---|---|
| `copy_augmented_hrm_best.pt` | NL math | 100% | 200/200 | 38s |
| `copy_word_best.pt` | Word problems | 98% | 96/100 | 248s |
| `copy_gsm_best.pt` | GSM-style | 100% | 95/100 | 491s |
| `copy_funcall_best.pt` | Funcall reasoning | 86% | 171/200 | 611s |
| `copy_logic_best.pt` | Logic reasoning | 86% | 88/100 | 910s |
| `copy_writing_best.pt` | Creative writing | 96% | 97/100 | 255s |
| `copy_reasoning_best.pt` | Combined (9 cats) | 74% | — | plateau |

### Output-language family principle

Combined 9-category reasoning model plateaued at 74%. Diagnosis: two
structurally different output languages competing. Split by family:

- **Funcall family** (`fn(args)`): percentage 100%, ratio 100%, seq_cost 83%
- **Logic family** (`a > b and`): arithmetic 97%, syllogism 92%, compare 81%
- **Routing family** (function names): writing 97%, 11 categories

**Rule: one PT per output-language family, not per domain.** ~3-5 PTs
cover 30+ domains. Adding a domain within a family = data-only.

### Ceilings broken

| Old ceiling | New | How |
|---|---|---|
| Single-digit 0% | **100%** | Balanced data distribution |
| 3-digit 68% | **100%** | Copy mechanism |
| GSM 93% (28/30) | **95%** | Copy mechanism |
| Syllogism 36% | **92%** | Output-family split |

### CALM backends (4 new)

- `calm/backends/reasoning_ops.py` — 11 functions, 14 NL patterns
- `calm/backends/reasoning_kb.py` — 7 functions (syllogisms, fallacies)
- `calm/backends/writing_ops.py` — 14 functions (syllables, rhyme, meter, readability)
- `calm/backends/writing_kb.py` — 130 entries across 7 categories (40 poetry forms, 30 rhetoric devices, 20 meter patterns, 12 archetypes, 10 genres, 8 narrative structures, 10 writing rules)

### Infrastructure

- `calm/llm_computer/grammar_decode.py` — inference-time grammar mask + EOS boost. Null result on word problems (0 fixes, 0 regressions). Infrastructure shipped for future use.
- `calm/llm_computer/substrate_server.py` — OpenAI-compatible API server with keyword-based PT routing, CALM precompute fallback, llama-server proxy. 3/8 on free-form queries (PT template diversity gap).
- `.claude/commands/domain.md` — `/domain` slash command for guided domain addition (7 steps with AskUserQuestion).
- `calm/hrm/data.py` — VOCAB_SIZE 80 → 82 (added `><`)
- `calm/hrm/{reasoning,writing}_data.py` — data generators with balanced sampling

### Substrate-native Gemma loader (`calm/llm_computer/gemma_substrate.py`)

Full Gemma 4 E4B (42 layers, 720 tensors, 262K vocab) loaded from GGUF:

- **Mmap zero-copy**: 1.9 GB RSS during loading (was OOM at 28 GB)
- **GPU preload**: all tq4 bytes on GPU (2.07 GB) + Q6_K embeddings (1.1 GB) = 5.07 GB
- **GPU dequant**: Pi + centroids cached on GPU, zero CPU→GPU transfer during inference
- **KV cache**: FP16, sliding window, per-layer (tq4 KV planned for 512K)
- **Chunked output head**: Q6_K dequant in 16K-row chunks, 160 MB peak
- **GemmaTokenizer** (`calm/llm_computer/synth/gemma_tokenizer.py`): 262K vocab from GGUF
- **Architecture**: GQA (8Q/2KV), per-layer head dim (SWA 256 / global 512), proportional RoPE with freq_factors, V normalization, post-attn/FFN norm before residual, per-layer embedding injection, layer output scale, logit softcapping (tanh/30)
- **Performance**: 0.54-0.62 tok/s, 2.2s prefill for 6 tokens

**Status: output incoherent.** All architecture features implemented per
`llama.cpp/src/models/gemma4-iswa.cpp` reference, but text output is
garbage (random multilingual tokens). Systematic per-layer debugging
needed — see "Next Steps" below.

## In Progress

### Gemma substrate coherence debugging

The inference pipeline runs end-to-end but produces wrong tokens. The
remaining issue is a weight dequant or computation mismatch vs llama.cpp.
The pre-transpose output (v6) showed English programming terms ("useState",
"Double") — LESS random than post-transpose (v7). Reverted to original
orientation.

Candidates:
1. Weight reshape (in, out) vs (out, in) — original is closer but still wrong
2. Per-layer embedding data layout (n_layers × d_per_layer vs d_per_layer × n_layers)
3. Per-layer model_proj FP16 loading orientation
4. RoPE freq_factors application (multiply vs divide)
5. Attention implementation detail (scaling, masking, head ordering)

## Next Steps (priority order)

### 1. Debug Gemma substrate output (highest priority)

**Approach**: systematic per-layer comparison against llama-server.

```bash
# Step 1: Get llama-server intermediate values
# Use llama-server's debug logging to dump layer activations for
# the prompt "The capital of France is" (6 tokens)

# Step 2: In gemma_substrate.py, dump h after each layer
# Compare: which layer first diverges from llama-server?

# Step 3: Within that layer, compare Q/K/V/attn_out/FFN values
# The first divergent value points to the bug
```

The bug is likely in one of:
- `MmapTq4Linear.dequant()` reshape/orientation
- `GpuQ6KEmbedding.__getitem__()` row lookup
- `_forward_layer()` per-layer embedding injection
- `_apply_rope()` with freq_factors

### 2. tq4 KV cache (for 512K context)

Once output is coherent, swap FP16 → tq4 in `KVCache.update()`:
```python
self.k_cache[layer_idx] = quantize_tq4(k_new)
k_full = dequantize_tq4(self.k_cache[layer_idx])
```
This enables 512K context (1.8 GB KV vs 7.3 GB FP16).

### 3. Wire substrate server to Gemma substrate

Replace individual PT inference in `substrate_server.py` with
`GemmaSubstrate` forward pass. One server, one model, one forward.

### 4. Install PTs into Gemma substrate

Use `install_compiled_card_hybrid` to put PT checkpoints at reserved
sub-head offsets in the loaded Gemma model. Prove PTs fire alongside
Gemma in one forward pass.

### 5. torch.compile for performance

Once correctness is proven, `torch.compile` the forward pass for
2-5x speedup (fuses dequant + matmul kernels).

## Key Context

### Corrected architecture principle

**"Model understands, transducers structure, cards compute, engine
verifies."** No single component reasons — the pipeline produces
reasoned answers through composition.

### Accuracy priority order (session 31 finding)

```
1. Data distribution — every valid input region covered? (free)
2. Mechanism — right operation for the task? (cheap)
3. Output-family split — too many output languages? (moderate)
4. Capacity — model genuinely too small? (expensive, last resort)
```

### Failed approaches (don't retry)

1. **50/50 small/large operand split** → overcorrected (mid collapsed to 2%). Use 3-bucket.
2. **Grammar-constrained decoding** → null on word problems. Failures are semantic not syntactic.
3. **Combined 9-category reasoning model** → plateaued at 74%. Split by output family.
4. **Weight transpose (out, in) for GGML** → made output WORSE. Original (in, out) reshape is closer to correct.
5. **Full FP32 embedding cache on GPU** → OOM (2.56 GB + 2 GB weights). Use chunked Q6K dequant.
6. **Full embedding dequant on GPU at once** → OOM. Use chunked 16K-row batches.
7. **CPU→GPU transfer per-layer** → too slow. Preload tq4 bytes to GPU, dequant on GPU.

### Environment state

- Branch `feature/multi-agent-qwen`, ~242 commits ahead of origin.
- 7 copy-augmented PT checkpoints in `calm/hrm/checkpoints/copy_*_best.pt`
- `substrate_hrm_nl_best.pt` retrained with balanced data (94% autoreg)
- llama-server NOT running. GPU free.
- 281 tests passing.
- VOCAB_SIZE = 82 (was 80, added `><`).
- Gemma substrate loads at 5.07 GB GPU, 0.54 tok/s, output incoherent.
- RTX 4070 (8 GB VRAM) + 32 GB DDR5.
- Python 3.13.7, PyTorch 2.10.0+cu128.

### User's key insights

- "It's an actual brain with sub-regions for different tasks" — substrate = specialized sub-head regions, shared wiring (channels), one forward pass
- "Model understands, transducers structure, cards compute, engine verifies" — corrected principle
- "One PT per output-language family, not per domain" — the scaling insight
- "You just expand data and substrate gets better? Like coding billions of params into an LLM?" — knowledge DB as alternative to training
- "Why aren't we using tq4?" — tq4 everywhere: weights, KV cache, same memory efficiency as llama-server
- "Why can't we build our own version of llama?" — substrate-native inference from PyTorch

## First action on resume

1. Read this handoff.
2. Start llama-server with the same GGUF, send "The capital of France is", capture the first generated token. If it's "Paris" → llama-server is the reference.
3. In `gemma_substrate.py`, add debug dumps: after embedding lookup, after layer 0, after layer 1. Compare against llama-server's internal values (use `/slots` API or add logging).
4. Find the first divergent value → that's the bug.
5. Fix → coherent text → commit → wire into substrate_server → ship.
