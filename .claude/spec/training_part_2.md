Training Part 2

### Empirics
- **SubstrateLM MVP** (`scripts/train_substrate_lm.py`): 1.25M params,
  1.5K Claude examples, 3 epochs, batch 16, 13 min on CPU → ppl
  4096→424. Format learned (`<think>`, numbered lists, code markers);
  content coherence needs scale (100M-500M for useful prose).
  Checkpoint at `calm/llm_computer/checkpoints/substrate_lm_mvp.pt`.
- **Training corpus**: reuse existing `agents/distill/data/claude_reasoning.jsonl`
  (910 examples) + `coding_reasoning_claude.jsonl` (547 examples).
  Claude-authored > Gemma-generated at this scale (quality dominates
  at small sample counts).

### Hybrid (SubstrateHRLM) training lessons

- **v1 result** (`scripts/train_hybrid_substrate.py`): LM mode improved
  by 8.7% (cross-task transfer from HRM's grounded data). HRM mode
  collapsed to 0% correct (78% parseable, 0% structural match).
- **Diagnosis**: token-count imbalance (LM ~500 tok/ex vs HRM ~30
  tok/ex → 16:1 LM gradient dominance) + single template family from
  `nl_data.py` only (~15 phrasings).
- **v2 fixes** (`scripts/train_substrate_hrmlm_v2.py`): oversample HRM
  examples (3-4×), mode-loss weighting (HRM mode ×4), pool template
  variety (nl_data + word_data + gsm_data + multi20_data → ~50+
  phrasings). Applies the session-26 multi20 lesson: **variety beats
  capacity for structure tasks**.
- **v2 architecture**: D3 mixed geometry per layer (e.g.
  `["euclidean", "hyperbolic", "lattice", "euclidean"]`). D5 per-mode
  iteration count (HRM at 2-3 iterations, LM at 1). D2 trace emission
  available for eval inspection.

### Cross-task transfer measurement (1+1=N)

When training a hybrid card, measure each mode's accuracy separately
against single-task baselines:

- **1+1=2** → modes coexist (no degradation). Acceptable PASS.
- **1+1>2** → cross-task transfer (one mode improves the other).
  Strongest PASS; evidence for shared representations.
- **1+1<2** → modes interfere. Investigate token ratios, loss
  weighting, mode-token strength.

v1 result: 1+1=2.09 on LM (positive transfer from HRM data) and
1+1=0 on HRM (collapse). Useful diagnostic per-mode — one signal alone
would have missed the asymmetry.

### Per-mode iteration during training (D5)

- HRM-mode examples in hybrid training run with `n_iterations=2-3` (more
  thinking on structure tasks).
- LM-mode examples run with `n_iterations=1` (one pass for fluent
  generation).
- At inference, `n_iterations` is dispatched per mode token.
- Training cost: ~1.5× per-step time when batch contains HRM examples.

### GPU vs CPU for substrate training

All substrate training scripts accept `--device auto` (default cuda if
available, else cpu).

**Stay on CPU**: model <500K params AND seq <128 tok AND pure-Euclidean
AND no D5. SubstrateLM MVP at 1.25M params trained in 13 min CPU.

**Move to GPU**: model >2M params, seq >256, D3 mixed geometry (hyperbolic
`acosh` doesn't vectorize well on CPU), D5 recurrence (serial kernel
launches), or any combination — effects compound.

**Observed** (v2 SubstrateHRLM): CPU 28s/step projecting 12h; GPU RTX 4070
same config 4.8s/step → 2h. Only 6× speedup (not 10-20×) because D5 serial
Python loop bottlenecks.

**GPU prerequisite**: Gemma must NOT be in VRAM (8 GB ceiling too tight
for Gemma + training). `pkill llama-server` before launching; verify CUDA
via `python3 -c "import torch; print(torch.cuda.is_available())"`.

### Safer-config for noisy-grad training

R52.2 canonical instance: (batch=1, lr=1e-3, grad_clip=1.0) diverged at
step 75 on a loss=30.2 outlier — Adam momentum poisoned, EMA climbed 2.23 →
3.93 over 20 steps. Restarted with (batch=4, lr=3e-4, grad_clip=0.1,
warmup=200) — converged cleanly to val 1.21 over 1000 steps.

When batch is small, prompts are mixed-loss, and you're on Adam/AdamW: use
**batch ≥ 4**, **grad_clip ≤ 0.1**, **lr ≤ 3e-4**, **warmup ≥ 200**.
**Diagnose Adam momentum poisoning** by EMA: if loss spikes and EMA climbs
20+ steps without recovery, optimizer state is corrupted — kill and
restart; continued training won't recover.

### Scheduled Sampling for SubstrateHRM (session 30)

**The training trick that tripled autoregressive accuracy: 33% → 90%.**

The old SubstrateHRM checkpoint reported 99.1% val_acc but that was
teacher-forced. Greedy autoregressive decode hit only 33% — the model
never learned to recover from its own errors.

Fix: **scheduled sampling**. During training, randomly replace teacher-
forced tokens with the model's own predictions. The ratio decays linearly:

```
tf_ratio: 1.0 (pure teacher) → 0.3 (70% self-generated) over 500 epochs
```

Implementation (`scripts/train_substrate_hrm.py`):
- 2-pass per batch: (1) no-grad forward to get predictions, (2) vectorized
  swap via `torch.where(swap_mask, shifted_preds, x)`, (3) gradient forward
  on modified input.
- **Vectorize the swap** — a Python loop over positions is CPU-bound at 95%.
  Use `torch.where` for the full batch in one op.
- **Autoreg eval** (`_autoreg_eval`): greedy-decodes the expression from the
  NL prompt and checks exact match. This is the gate metric. Save checkpoint
  on best autoreg, not teacher-forced accuracy.

Result on RTX 4070 (180K params, 879 seconds):

```
Epoch   1:   4% autoreg  (baseline, tf_ratio=1.00)
Epoch  50:  84%           (scheduled sampling kicks in)
Epoch 100:  88%           (broke through first plateau)
Epoch 150:  90%           (peak, checkpoint saved)
Epoch 500:  90%           (held through tf_ratio=0.30)
```

**Data distribution matters**: 90% is on the training distribution
(operands [1, 999]). Single-digit operands ([1, 9]) are underrepresented
(~1% of random draws) → 25% on small-operand test. Fix: include explicit
small-operand examples in training data. Another 15 min run.

**Rule: always use `--scheduled-sampling` (default ON) when training
SubstrateHRM.** The cost is ~2× per epoch (two forwards). The benefit
is the difference between 33% and 90% on the metric that matters.

### Compiled-vs-trained card distinction

- **Compiled cards** (gate-graph IR → weights) — no training, exact,
  programmatic. Lives in `calm/llm_computer/programs/`.
- **Trained cards** (substrate-native or HRMSeq2Seq) — SGD,
  statistical. Lives in `calm/llm_computer/checkpoints/` or
  `calm/hrm/checkpoints/`.
- **Future: distill-to-compiled** — detect stable learned patterns and
  replace with exact compiled equivalents. Novel research direction;
  no other architecture supports this because no other architecture
  has compiled-program slots in the same weight tensor as trained ones.

## Dataset Quality
- Claude-authored/hand-written data >> 9B-generated data (higher quality, more consistent)
- HuggingFace datasets (nohurry, TeichAI, Crownelius) provide Claude Opus reasoning traces — filtered to 832 examples
- Hand-written data committed to repo: `coding_reasoning_claude.jsonl` (507 examples), `orchestrator_claude.jsonl` (121 examples)
- Merged training file: `claude_reasoning.jsonl` (910 examples, re-filtered pass over the HF + hand-written sources; the earlier 1,339 count was a pre-filter merge)
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

### Substrate eval defaults (R53.28 + R53.34 + centralized)

**Canonical module**: `calm/llm_computer/eval_defaults.py` (shipped
in `805e539` Track A). Four constants govern every substrate eval;
change them in one place and all R-series scripts pick them up on
next run.

```python
from calm.llm_computer.eval_defaults import (
    EVAL_CTX_SIZE,       # 32768 — pre-allocated tq4 KV ceiling
    EVAL_MAX_TOKENS,     # 16384 — AdaptiveBudget output clamp
    ITERATION_N,         # 5 — fast-iteration problem count
    FINAL_N,             # 20 — commit-baseline problem count
    resolve_problem_window, # rotation helper: (n, skip) from /tmp/substrate_eval_rotation.json
    get_adaptive_budget, # per-prompt budget + estimate metadata
)

n, skip = resolve_problem_window()  # default (5, 0); FINAL=True sets (20, 0)
budget, est = get_adaptive_budget(prompt)
out = m.generate(prompt, tok, max_tokens=budget,
                 use_tq4_kv=True, kv_max_len=EVAL_CTX_SIZE)
```

`bin/mbpp-rotate N` writes the window state (window=N, final=False);
`MBPP_FINAL=1` env toggles final mode per-script (e.g. `MBPP_N = FINAL_N
if os.environ.get("MBPP_FINAL")=="1" else ITERATION_N`).

- `EVAL_CTX_SIZE=32768` pre-allocates `KVCacheTq4` regardless of
  prompt length. tq4 storage is ~3.6× smaller than fp16 so 32K costs
  only ~700 MB added VRAM on top of the ~5 GB substrate baseline.
  Leaves headroom under the 8 GB ceiling for 1-2 FP32 host layers.
- `EVAL_MAX_TOKENS=16384` is the AdaptiveBudget clamp. Tiered
  (trivial 2K / easy 4K / medium 8K / hard 16K / deep 32K) but
  always clamped here. Gemma 4 E4B trains at 131K ctx; any budget
  < 4K truncates real coding problems mid-function (receipt: R53.25
  lifted `log_level_counts` 0/0 → 6/6 purely from 400 → 900 tok bump).
- `generate(use_tq4_kv=True)` routes through real-tq4 `KVCacheTq4`
  (multi-token prefill S≥1). Phase 2 fused flash-attn kernel ships
  default-on (`_use_fused_flash_attn=True`) with runtime N-gate
  `128 < cached_kv_len < 2048` — the measured winning band. Outside
  the band (small prompts + first ~128 decode steps, or decode past
  2048) it falls back to Phase 1 memoized dequant. Long R53 eval
  with AdaptiveBudget up to 16K uses memo past 2048 — no regression.
  See `tq4_flash_attn.py`, `MEMORY/atlas/tracing_arc_part_2.md` Round 53.34, and
  `turboquant.md` for the bench table.
- **Exception**: `scripts/r51_eval_dual_gate.py` and
  `scripts/r52_eval_dual_gate.py` use `K_TOKENS=12` as a measurement
  design (prefix-match on teacher-vs-student first-K tokens), not an
  eval budget. Don't migrate those to EVAL_MAX_TOKENS — K is the
  metric, not a cap.

## Export & Serving

Canonical serving docs in `.claude/CLAUDE.md` §"Serving Architecture".
Quantization-specific details in `.claude/spec/turboquant.md`. Default:
Q5_K_M weights + Q4/tq4 KV; `Q5_K_M` over Q4 unless VRAM forces; for
Gemma use tq4 + tq4-KV at 512K context.

Training-output path: merge LoRA → GGUF via `llama-quantize` → serve
via `llama-server` (OpenAI-compatible API) or `ollama create` from a
Modelfile in `models/`.
