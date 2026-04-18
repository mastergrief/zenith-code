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

### Substrate FP32 hosting layers (RTX 4070, 8 GB)

For installing cards INTO Gemma's attention via
`install_card_in_attention` (see `Substrate.md`), the host layer must
be FP32 (tq4 quant noise destroys compiled coefs — Round 11). One-time
conversion via `GemmaSubstrate.convert_layer_to_fp32(layer_idx)`.

- per FP32 SWA layer: ~330 MB
- per FP32 global layer (5, 11, 17, 23, 29, 35, 41): ~600 MB
- Substrate baseline (Triton + tq4 + Q6_K): ~5.0 GB
- Practical: **5-7 hosting layers** before bumping the 8 GB ceiling

Track allocations in `.claude/MEMORY/substrate_registry.md` so two
domains don't reserve overlapping channels/sub-heads in the same host.

## Priority Order
- **Backend coverage > data quality > data quantity > model size > training tricks.**
  Adding a compute backend is instant, free, and deterministic. Training is
  expensive, slow, and probabilistic. Build a backend first; only train when
  the domain CAN'T be computed (style, creativity, judgment).
- Auto-CALM with 9 backends scores **100% on 40-problem math benchmark**
  without any fine-tuning. Stock Gemma 4 E4B + modular compute = frontier
  accuracy on computable domains.
- **When training IS needed**: data quality > quantity > model size.
  One hour writing 20 high-quality examples beats hours of tuning.
- Each example should demonstrate the *reasoning process* (`<think>` block), not just the answer
- Match the training domain to the task: coding data for coding models, routing data for routing models

## Auto-Training Data (from Auto-CALM corrections)
- Every Auto-CALM correction generates a labeled training example automatically
- Sub-collectors: `MathCollector`, `BoolCollector`, `CodeCollector`
- Output: `.calm_training/auto/{math,bool,code}.jsonl` — distillation-compatible
- Merge with: `AutoTrainingCollector().export_merged()`
- Virtuous cycle: model errors → corrections → training data → (optional) fine-tune → fewer errors
- **This is the primary training data source going forward** — zero manual labeling
- For substrate-native cards, this feeds continuous self-distill cycles
  into the trained portions (pattern proven by session-27
  `self_distill_synth.py` — teach once, library accretes, fine-tune
  folds library back into weights). The same pattern generalizes to
  SubstrateLM / SubstrateHRLM weight updates in the Brain + Cards
  composition model.

## Training Best Practices
- Always use `train_on_responses_only` — masks instruction/prompt tokens so loss is only computed on the model's generated responses
- ~1,300 curated examples with 3 epochs works well (confirmed: 0.8B run 2, loss 1.106)
- `nohurry/Opus-4.6-Reasoning-3000x-filtered` is the best-filtered Claude reasoning dataset on HuggingFace
- 3 epochs on curated, diverse data — 1 epoch underfits, more epochs on small datasets = memorization
- Filter training data before use: `python -m agents.distill.filter_reasoning --merge`
- Filter aggressively — removing bad data improves results more than adding mediocre data

## HRM Training (CRLM workflow)

For HRMs in the HRM-thinking + LLM-Compute split (`calm/hrm/` → `calm/llm_computer/`), the training rules differ from the distillation pipeline above:

### Don't make the model memorize values

The single biggest training improvement in session 25: **stop asking HRM to compute values**. Use `--structure-only` in `calm/hrm/train.py`. Decoder target becomes `problem + = + <eos>`. The LLM-Computer interpreter handles every value via `safe_eval`. Result: 245K → 48K params, 15 min → 145 sec training, 43% → 96.7% full-expression accuracy.

### Sweet-spot config (all production domains, session 26)

```bash
# Math (3-digit operands)
PYTHONPATH=. python3 -m calm.hrm.train --seq2seq --structure-only \
  --hidden 32 --num-heads 4 --l-layers 1 --h-layers 1 --dec-layers 1 \
  --epochs 500 --lr 1e-3 --problems 2000 --batch-size 128 \
  --max-enc 32 --max-dec 32

# NL templates, word problems, GSM-style, multi-task (pooled) —
# standalone trainers at scripts/train_hrm_{nl,word,gsm,multi}.py.
# Each uses the same 48K architecture; only --max-enc changes per domain
# (48 / 80 / 128 / 128 respectively).
```

All 5 production checkpoints: 48,864 params. Best-val-acc selection lands between epoch 100-300 in practice. Training time: ~145s (math) to ~800s (GSM / multi-task) on RTX 4070.

### Rule: ALWAYS `--epochs 500`, never `--epochs 100`

Observed 4 times across session 26: cosine LR schedule over only 100 epochs under-fits on any NL domain. The LR decays to ~0 before the model has converged on digit-copy precision. Pattern:

| Domain | 100ep result (full-expression) | 500ep result | Delta |
|---|---:|---:|---:|
| Math 3-digit | 26.7% | 100% | +73pp |
| NL templates | 96.7% | 97% | baseline |
| Word problems | 100% (killed early) | — | — |
| GSM-style | 83.3% | 93.3% | +10pp |

Set `--epochs 500` generously. Rely on `best_val_acc` checkpoint selection to pick the right moment. Don't kill early unless monitored logs confirm convergence.

### CRLM scaling-law empirics (session 26, 48K params each)

| Input language | Max chars | Per-token | Full / structural | Note |
|---|---:|---:|---:|---|
| Math expression echo (3-digit) | ~20 | 100% | 30/30 | Round 1e sweet spot |
| NL templates ("what is X plus Y?") | ~30 | 99.8% | 29/30 | Translation |
| Word problems (names + pronouns) | 78 | 99.7% | 30/30 | Anaphora 2-3 sentences |
| GSM-style (subordinate clauses) | 104 | 99.6% | 28/30 | **First ceiling** — digit transpositions |
| Multi-task (all four pooled) | 104 | 100% | TBD per-domain | Vector 2 phase 1 |

The GSM shortfall was digit transposition — **fixed by copy mechanism in session 31** (93%→95% held-out). See PT training below.

### Pointer Transducer training (session 31, replaces HRM for new work)

**Architecture**: `CopyAugmentedTransformer` (`calm/llm_computer/copy_augmented.py`). Decoder-only `Small2DTransformer` + learned copy gate + pointer attention. 1,089 extra params. Forward returns **log-probs** — use `F.nll_loss`, not CE.

**Cross-domain results (all ~185K params, `--epochs 500`, scheduled sampling):**

| Domain | Val autoreg | Held-out | Training time | Max input |
|---|---|---|---|---|
| NL math | 100% | 200/200 | 38s | 30 chars |
| Word problems | 98% | 96/100 | 248s | 78 chars |
| GSM-style | 100% | 95/100 | 491s | 104 chars |
| Funcall reasoning | 86% | 171/200 | 611s | 88 chars |
| Logic reasoning | 86% | 88/100 | 910s | 121 chars |
| Creative writing | 96% | 97/100 | 255s | 65 chars |

**Key training rules for PT:**
- **Balanced `_sample_operand()`**: uniform across digit-length buckets [1-9]/[10-99]/[100+]. Without this, small operands get 0%.
- **`max_len` ≥ max_prefix + max_expression + decode_headroom**: positional embeddings cap sequence length. CUDA assert if autoreg exceeds it.
- **One PT per output-language family**: function-call, infix arithmetic, boolean logic. Combined model plateaus at 74%; split recovers 86-88%.
- **Autoreg eval is the gate**: teacher-forced val_acc is misleading (99.6% while autoreg is 74%). Always use `_autoreg_eval`.
- **Copy gate bias = -2.0**: initializes toward generation, learns to copy.
- **VOCAB_SIZE = 82** (added `><` in session 31). Old checkpoints use 80 and load fine.

**Remaining ceiling**: 3+ operand copy accuracy (68-83%). Copy attention over prefix gets noisy with 3+ numbers. Known fix: two-stage decode via D5 recurrence.

**Knowledge DB as alternative to training** (session 31 finding): for domains where the value is knowing rules, not computing values (creative writing, legal, style), a knowledge backend (`*_kb.py`) with dictionary entries is more effective than training. Each entry = 3 ReGLU neurons when compiled into substrate. The writing KB has 130 entries across 7 categories — equivalent information to billions of gradient updates but instant, exact, and inspectable.

**Accuracy priority order (session 31 finding):**
1. Data distribution — every valid input region covered? (free)
2. Mechanism — right operation? e.g. copy vs generate (cheap)
3. Output-family split — too many output languages in one model? (moderate)
4. Capacity — model genuinely too small? (expensive, last resort)

### Legacy HRM training notes

Per-token val_acc is inflated by trivial-copy tokens. **The gate that matters is full-expression accuracy**, not the trainer's reported number.

Scratchpad-with-intermediate-values forces memorization that small models can't deliver. **Stop asking the model to compute; let it emit structure and route values to the substrate.**

**HRM size scales with input-language complexity, NOT problem-difficulty.** PT size scales with **output-language complexity** — the number of structurally distinct expression formats. Both scale independently of value range.

### Pitfalls observed (sessions 25-26)

- **Cosine LR scheduled to 0 too early kills learning.** See the `--epochs 500` rule above — this is now elevated from pitfall to mandatory default on any NL-input HRM.
- **Smoke cases failing at 3-digit numbers** are out-of-distribution. `MathDataGenerator`'s `_arithmetic_simple` was capped at `randint(1, 99)`; bumped to 999 in session 26 step 1. Any new domain: match the operand range to the smoke cases you care about.
- **Per-token > 90% but structural < 50%** means the model nailed structure tokens (operators, parens, function names) but mis-copied digits. Use `_hrm_raw_emit()` from eval scripts to inspect raw output — if structure is good but digits drift, add data OR scale capacity. The `--verified` mode (LLM-Computer parses the input directly) masks this class of failure when full-expression is the gate but reveals it when structural-match is the gate.
- **Per-domain `max_enc` is load-bearing.** Math fits in 32, NL in 48, word problems in 80, GSM in 128, multi-task in 128 (max of components). Undershoot `max_enc` and the sentence is truncated silently mid-operand; overshoot and you waste compute. The canonical trainer scripts (`scripts/train_hrm_{nl,word,gsm,multi}.py`) document the correct bound per domain.

## Substrate-Native Training (SubstrateLM / SubstrateHRLM)

Different from HRMSeq2Seq training above. Substrate-native cards are
decoder-only `Small2DTransformer` instances trained on chat-formatted
text. They become substrate-compliant `.pt` files that compose with
compiled programs and HRM specialists at runtime.

**Compiled cards from `programs/` need NO retraining to install into
prod Gemma**: `install_card_in_attention(card, layer, sub_head_off,
ch_off, d_card, mode='hard_max')` writes the card's existing
gate-graph IR weights into Gemma's FP32-converted `attn_q/k/v/output`
at the reserved slot. Per-sub-head dispatch handles the different
attention mode. End-to-end auto-upgrade loop: detect → log → compile
recall card → install → persist. Demo:
`scripts/gemma_learning_loop_demo.py` (5/5 wrong → 5/5 correct after
loop closes). See `Substrate.md` for install patterns.

### Config (MVP scale, proven)

- `d_model=96-192`, `n_heads=d_model/2` (d_head=2 invariant),
  `n_layers=3-5`, `d_ffn=2-4× d_model`, `max_len=256-512`.
- BPE tokenizer via `tokenizers` lib, `vocab=4096-16384`. ByteLevel
  pre-tokenizer AND ByteLevel decoder (forgetting the decoder breaks
  sample readability — leaves raw BPE markers `Ġ`, `Ċ`). Trained on the
  corpus directly.
- Special tokens: `<|sys|>`, `<|user|>`, `<|asst|>`, `<|eos|>`, `<|pad|>`
  + mode prefixes `<|lm|>`, `<|hrm|>` for hybrids.
- Loss masking: `train_on_responses_only` — only assistant tokens get
  gradient. For HRM-mode examples in hybrids, the mask is the expression
  text after `=`.

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

Per `.claude/rules/workflow.md`:
- CPU is fine for tiny models (<500K params), short sequences, pure-
  Euclidean attention.
- GPU becomes necessary when D3 (hyperbolic `acosh`) + D5 (per-iteration
  Python loop) compound per-step cost. v2 observed: CPU 28s/step → GPU
  4.8s/step (only 6× speedup because D5 launches kernels serially).
- Prerequisite: Gemma must NOT be in VRAM. Kill `llama-server` before
  launching GPU training.
- All substrate scripts accept `--device auto` (defaults to cuda if
  available).

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

## Export & Serving
- llama.cpp built at `~/llama.cpp/build/bin/` with CUDA support (RTX 4070), local patch at `tools/server/server-context.cpp:763-766` (see Known Issues)
- `llama-quantize`: convert FP16 safetensors → GGUF quantized — **Q5_K_M is the default across all model sizes** (best quality/size tradeoff per research). Only drop to Q4_K_M when VRAM forces it.
- `llama-server`: serve with OpenAI-compatible API, KV cache quantization, `enable_thinking` support
- 4B reasoning base GGUF at `~/models/Qwen3.5-4B.Q5_K_M.gguf` (2.9 GB, serving at **256K context**, ~7.3 GB VRAM with Q4 KV)
- Alternative: Gemma 4 E4B GGUF at `~/models/gemma-4-E4B-it-Q5_K_M.gguf` (5.48 GB, stock, ~6.7 GB VRAM at 256K thanks to sliding-window attention). Validated 2026-04-07, beats fine-tuned Qwen on coding eval.
- **tq4 serving GGUF** at `~/models/gemma-4-E4B-it-tq4-aligned.gguf` (5.0 GB, 132-byte-block layout, session 16). The old `~/models/gemma-4-E4B-it-tq4.gguf` (4.7 GB, 130-byte blocks) is **incompatible** with post-session-16 llama.cpp and should be ignored or deleted.
- Launch via `zenith` command (auto-starts llama-server at `ZENITH_CTX=524288`) or manually with `--ctx-size 524288 --parallel 1 --cache-type-k tq4_k256 --cache-type-v tq4_k256 -ngl 999`
- For Ollama: Modelfiles in `models/`, use `ollama create`
- `-ngl 999` forces all layers to GPU
