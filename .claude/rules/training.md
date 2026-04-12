# Training Rules

## VRAM Budget

### Local (8 GB RTX 4070 Laptop)
- Qwen 3 0.6B QLoRA: ~2.5 GB — fits with batch=4, packing=true
- Qwen 3.5 0.8B QLoRA: ~4 GB — requires batch=1, packing=false, seq_len=1024
- Qwen 3.5 4B QLoRA: **OOM on 8 GB, 16 GB, and 15 GB** — 248K vocab CE loss is too large
- Always stop Ollama before training: `ollama stop <model>` or verify `ollama ps` is empty

### Cloud (Colab Pro A100 40GB)
- Qwen 3.5 4B QLoRA: fits with batch=1, seq_len=1024, packing=false
- Use `agents/distill/train_4b_colab.ipynb` — **this is the active training path**. `train_4b_cloud.py` exists for RunPod/Lambda but is not currently used (RunPod is set up but the user runs training on Colab Pro).
- Cost: ~$0.50-1.00 per training run (~30-40 min on A100)

## Priority Order
- **Data quality > data quantity > model size > training tricks.** One hour writing 20 high-quality examples beats hours of hyperparameter tuning.
- **Model size matters for correctness.** 0.8B learned `<think>` format but gave wrong answers. 4B confirmed: 3/5 eval PASS with `enable_thinking: true`.
- Each example should demonstrate the *reasoning process* (`<think>` block), not just the answer
- Match the training domain to the task: coding data for coding models, routing data for routing models

## Training Best Practices
- Always use `train_on_responses_only` — masks instruction/prompt tokens so loss is only computed on the model's generated responses
- ~1,300 curated examples with 3 epochs works well (confirmed: 0.8B run 2, loss 1.106)
- `nohurry/Opus-4.6-Reasoning-3000x-filtered` is the best-filtered Claude reasoning dataset on HuggingFace
- 3 epochs on curated, diverse data — 1 epoch underfits, more epochs on small datasets = memorization
- Filter training data before use: `python -m agents.distill.filter_reasoning --merge`
- Filter aggressively — removing bad data improves results more than adding mediocre data

## Dataset Quality
- Claude-authored/hand-written data >> 9B-generated data (higher quality, more consistent)
- HuggingFace datasets (nohurry, TeichAI, Crownelius) provide Claude Opus reasoning traces — filtered to 832 examples
- Hand-written data committed to repo: `coding_reasoning_claude.jsonl` (507 examples), `orchestrator_claude.jsonl` (121 examples)
- Merged training file: `claude_reasoning.jsonl` (1,339 examples = 832 HF + 507 hand-written)
- 2026-04-07 addition: +19 hand-written examples (11 React + 8 security) via `scripts/generate_react_security_examples.py`, targeting gaps identified in the initial Qwen 4B eval (React hook cleanup, SSRF, CSP, IDOR, password reset, etc.)
- Filter pipeline: tiered keyword matching (1 strong keyword + 2 general, OR 5+ general, OR code blocks), dedup by first 60 chars, think-block minimum lengths
- Filter out: hallucinated facts, non-technical content, NLP benchmark patterns, junk (<200 char responses)
- Per-domain specialist datasets remain small (python 25, typescript 39, rust 53) and are 9B-generated — need hand-written upgrade before specialist training. The "React/security weak spots" framing in older notes referred to `coding_reasoning_claude.jsonl` gaps; those are now filled.

## Known Issues
- **Qwen 3.5 248K vocab**: fused cross-entropy loss OOMs on anything under 40GB VRAM for 4B, under 8GB for 0.8B
  - 0.8B fix: batch_size=1, max_seq_length=1024, packing=False
  - 4B fix: use cloud GPU (Colab A100)
- Unsloth compiled cache stored at `unsloth_compiled_cache/` in project root (gitignored)
- Git Bash mangles WSL paths with parentheses in PATH — use `wsl -e bash -c` or write scripts to `/tmp/`
- **WSL Windows-binary stdin consumption**: any `*.exe` called from a bash script (e.g. `tasklist.exe`, `cmd.exe`, `winget.exe`) consumes parent stdin via WSL's interop shim, even if the binary doesn't intentionally read it. Always pass `< /dev/null` to Windows binaries in scripts that may receive piped stdin. See `bin/zenith` (search for `tasklist.exe`) for the canonical fix and the inline comment explaining why. This bit `printf "..." | zenith` invocation hard during the 2026-04-07 harness debugging — the harness's first `input()` call got `EOFError` immediately because `tasklist.exe` had already drained the pipe.
- **llama.cpp slot context cap** (session 2026-04-07): `tools/server/server-context.cpp:763-766` silently caps each slot's context to `n_ctx_train`, so `--ctx-size` past the trained max is invisibly truncated. Patched locally to comment out `n_ctx_slot = n_ctx_train`. Not upstreamed — re-apply after `git pull` on llama.cpp. Required for 256K-range NIAH testing on Gemma 4 E4B (trained at 128K).
- **Gemma 4 GGUF rope-scaling metadata override** (session 2026-04-07): Gemma 4 E4B's GGUF bakes in `rope scaling = linear`; the `--rope-scaling yarn` CLI flag is silently ignored. Past the trained context is raw RoPE extrapolation (no scaling). Works empirically up to ~200K on single-needle but multi-needle drops to 4/5 at 220K.
- **llama-server `--parallel` default** (session 2026-04-07): defaults to 4 slots; each slot gets `ctx_size / parallel`. For the harness's single-user workflow always pass `--parallel 1` to get the full `ctx_size` in one slot. `bin/zenith` passes this since commit `4644051`; manual `llama-server` invocations still need it.
- No Thunderbolt/eGPU on Acer Nitro AN17-42 — cloud GPUs required for 4B+ training
- RunPod SSH proxy unreliable from WSL2 — use web terminal or Colab instead

## Quantization (llama-quantize)

### TurboQuant Types
| Type | bpw | Block | Levels | Packing | Use |
|------|-----|-------|--------|---------|-----|
| tq3_k256 | 3.06 | 98 B | 8 | 3-bit groups (8 codes/3 bytes) | KV cache only |
| tq3_k512 | 3.03 | 194 B | 8 | 3-bit groups | KV cache (head_dim=512 layers) |
| tq4_k256 | 4.125 | 132 B | 16 | nybble (2 codes/byte) + 2 B trailing pad | **Weights + KV cache** |

- **tq4 is the recommended TurboQuant type** — 4-bit enables the rotated-domain FA speedup (2.2x V dequant) that tq3 cannot use (3-bit re-quantization cascades FP precision diffs across layers, see fattn-vec.cuh comment block)
- All tq*_k256 types share the same Pi rotation matrix (seed=42, 256×256 orthogonal)
- Codebooks are 16-level (tq4) or 8-level (tq3) Lloyd-Max for N(0, 1/√256), computed at runtime
- **tq4 block is padded to 132 bytes (session 16, 2026-04-11).** Was 130 before (2-byte `ggml_half d` + 128-byte `qs`). The 2-byte trailing pad makes every block 4-byte aligned so CUDA mmvq / fattn can issue aligned uint32 loads on `qs`. **Breaks compatibility with pre-132-byte tq4 GGUFs — re-quantize.**

### Quantization Commands
```bash
# f16 → tq4 (weights + KV cache compatible)
llama-quantize model-F16.gguf model-tq4.gguf TQ4_K256

# ALWAYS detach long quantize jobs to survive CC/WSL crashes:
setsid ~/llama.cpp/build/bin/llama-quantize \
  ~/models/model-F16.gguf ~/models/model-tq4.gguf \
  TQ4_K256 4 < /dev/null > /tmp/quantize.log 2>&1 &
disown -a
# Monitor: tail -f /tmp/quantize.log
# Verify: xxd model-tq4.gguf | head -1  (must show 'GGUF')

# Per-tensor selective (e.g. FFN only, keep attention as Q5_K_M)
llama-quantize --allow-requantize \
  --tensor-type "ffn_gate_up.*=tq4_k256" \
  --tensor-type "ffn_down.*=tq4_k256" \
  model-F16.gguf model-hybrid.gguf Q5_K_M
```

### WSL2 OOM During Quantization
- `llama-quantize` mmaps the source GGUF (~15 GB for E4B f16) and needs working RAM for the Pi rotation buffers. On 28 GB WSL2 this can OOM and **crash the entire WSL VM**.
- Fix: use 1 thread to reduce peak RAM: append `1` as last arg to `llama-quantize`
- Fix: `disown` the process so CC crashes don't kill a running quantize
- Fix: for very large models (26B+), `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` before quantizing to free page cache
- The corrupt-GGUF symptom is `invalid magic characters: '????'` — delete and re-quantize

### Quantize Process Management (MANDATORY)
- **ALWAYS use `setsid` + `disown`** for any quantize that takes >30s. Without this, a CC crash or WSL OOM kills the child process mid-write, corrupting the output GGUF.
- **ALWAYS use 4 threads** (append `4` as last arg). Single-thread quantize of E4B takes 10+ min vs ~6 min with 4 threads. The extra threads don't meaningfully increase RAM — the mmap dominates.
- **ALWAYS redirect stdin** from `/dev/null` (`< /dev/null`) to prevent WSL interop stdin consumption.
- **NEVER run quantize while building llama.cpp** — stacked nvcc + quantize can push WSL past 28 GB and crash the VM.
- **ALWAYS verify** the output GGUF header after quantize: `xxd model.gguf | head -1` must show `4747 5546` (`GGUF`). All-zeros means corrupt — delete and retry.

### Serving with TurboQuant KV Cache
```bash
# tq4 KV cache (recommended — enables rotated-domain FA speedup)
llama-server -m model.gguf --cache-type-k tq4_k256 --cache-type-v tq4_k256

# tq3 KV cache (smaller but no rotated-domain speedup)
llama-server -m model.gguf --cache-type-k tq3_k256 --cache-type-v tq3_k256

# Short aliases
--cache-type-k tq4 --cache-type-v tq4
--cache-type-k tq3 --cache-type-v tq3
```

### VRAM Budget (256K context, Q4 KV cache quantization, RTX 4070 8 GB)
| Config | Weights | KV cache | Total |
|--------|---------|----------|-------|
| Q5_K_M + f16 KV | 2.9 GB | ~4.0 GB | ~6.9 GB |
| Q5_K_M + tq3 KV | 2.9 GB | ~1.5 GB | ~4.4 GB |
| tq4 + tq4 KV | ~3.7 GB | ~2.0 GB | ~5.7 GB |
| tq3 + tq3 KV | ~2.2 GB | ~1.5 GB | ~3.7 GB |

## Export & Serving
- llama.cpp built at `~/llama.cpp/build/bin/` with CUDA support (RTX 4070), local patch at `tools/server/server-context.cpp:763-766` (see Known Issues)
- `llama-quantize`: convert FP16 safetensors → GGUF quantized — **Q5_K_M is the default across all model sizes** (best quality/size tradeoff per research). Only drop to Q4_K_M when VRAM forces it.
- `llama-server`: serve with OpenAI-compatible API, KV cache quantization, `enable_thinking` support
- 4B reasoning base GGUF at `~/models/Qwen3.5-4B.Q5_K_M.gguf` (2.9 GB, serving at **256K context**, ~7.3 GB VRAM with Q4 KV)
- Alternative: Gemma 4 E4B GGUF at `~/models/gemma-4-E4B-it-Q5_K_M.gguf` (5.48 GB, stock, ~6.7 GB VRAM at 256K thanks to sliding-window attention). Validated 2026-04-07, beats fine-tuned Qwen on coding eval.
- **tq4 serving GGUF** at `~/models/gemma-4-E4B-it-tq4-aligned.gguf` (5.0 GB, 132-byte-block layout, session 16). The old `~/models/gemma-4-E4B-it-tq4.gguf` (4.7 GB, 130-byte blocks) is **incompatible** with post-session-16 llama.cpp and should be ignored or deleted.
- Launch via `zenith` command (auto-starts llama-server at `ZENITH_CTX=262144`) or manually with `--ctx-size 262144 --parallel 1 --cache-type-k q4_0 --cache-type-v q4_0 -ngl 999`
- For Ollama: Modelfiles in `models/`, use `ollama create`
- `-ngl 999` forces all layers to GPU
