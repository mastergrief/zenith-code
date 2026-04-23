# Training Rules

> Historical receipts (session-25/26/27/30/31 training-arc anecdotes,
> R-numbered flag tables, HRM → PT evolution receipts, SubstrateLM
> MVP empirics, SubstrateHRLM v1/v2 hybrid receipts, dataset-addition
> specifics): see `MEMORY/atlas/training_part_1.md` +
> `MEMORY/atlas/training_part_2.md`. Quantization details (tq4/tq3
> block format, kernels): `rules/turboquant.md`.

## VRAM Budget

### Local (8 GB RTX 4070 Laptop)

- Qwen 3 0.6B QLoRA: ~2.5 GB — fits with batch=4, packing=true
- Qwen 3.5 0.8B QLoRA: ~4 GB — requires batch=1, packing=false, seq_len=1024
- Qwen 3.5 4B QLoRA: **OOM on 8 GB, 16 GB, 15 GB** — 248K vocab CE loss is too large
- Always stop Ollama before training: `ollama stop <model>` or verify `ollama ps` is empty

### Cloud (Colab Pro A100 40GB)

- Qwen 3.5 4B QLoRA: fits with batch=1, seq_len=1024, packing=false
- Use `agents/distill/train_4b_colab.ipynb` (active training path).
  `train_4b_cloud.py` exists for RunPod/Lambda but is not currently used.
- Cost: ~$0.50-1.00 per training run (~30-40 min on A100)

### Substrate FP32 hosting layers (RTX 4070, 8 GB)

For installing cards INTO Gemma's attention via
`install_card_in_attention` (see `Substrate.md`), the host layer must
be FP32 (tq4 quant noise destroys compiled coefs). One-time conversion
via `GemmaSubstrate.convert_layer_to_fp32(layer_idx)`.

- per FP32 SWA layer: ~330 MB
- per FP32 global layer (5, 11, 17, 23, 29, 35, 41): ~600 MB
- Substrate baseline (Triton + tq4 + Q6_K): ~5.0 GB
- Practical: **5-7 hosting layers** before bumping 8 GB ceiling

Track allocations in `.claude/MEMORY/substrate_registry.md` so two
domains don't reserve overlapping channels/sub-heads in the same host.

## Priority Order

**Backend coverage > data quality > data quantity > model size > training tricks.**

- Adding a compute backend is instant, free, and deterministic.
  Training is expensive, slow, and probabilistic. Build a backend
  first; only train when the domain CAN'T be computed (style,
  creativity, judgment).
- Auto-CALM with 9 backends scores 100% on the 40-problem math
  benchmark without any fine-tuning. Stock Gemma 4 E4B + modular
  compute = frontier accuracy on computable domains.
- **When training IS needed**: data quality > quantity > model size.
  One hour writing 20 high-quality examples beats hours of tuning.
- Each example should demonstrate the *reasoning process* (`<think>`
  block), not just the answer.
- Match the training domain to the task: coding data for coding
  models, routing data for routing models.

## Auto-Training Data (from Auto-CALM corrections)

- Every Auto-CALM correction generates a labeled training example
  automatically
- Sub-collectors: `MathCollector`, `BoolCollector`, `CodeCollector`
- Output: `.calm_training/auto/{math,bool,code}.jsonl` (distillation-compatible)
- Merge with: `AutoTrainingCollector().export_merged()`
- Virtuous cycle: model errors → corrections → training data →
  (optional) fine-tune → fewer errors
- **Primary training data source going forward** — zero manual labeling
- For substrate-native cards, feeds continuous self-distill cycles
  (teach once, library accretes, fine-tune folds library back into
  weights).

## Training Best Practices

- Always use `train_on_responses_only` — masks instruction/prompt
  tokens so loss is only computed on the model's generated responses
- ~1,300 curated examples with 3 epochs works well
- `nohurry/Opus-4.6-Reasoning-3000x-filtered` is the best-filtered
  Claude reasoning dataset on HuggingFace
- 3 epochs on curated, diverse data — 1 epoch underfits, more epochs
  on small datasets = memorization
- Filter training data before use:
  `python -m agents.distill.filter_reasoning --merge`
- Filter aggressively — removing bad data improves results more than
  adding mediocre data

## HRM Training (CRLM workflow)

For HRMs in the HRM-thinking + LLM-Compute split (`calm/hrm/` →
`calm/llm_computer/`), training rules differ from the distillation
pipeline above.

### Don't make the model memorize values

**Stop asking HRM to compute values.** Use `--structure-only` in
`calm/hrm/train.py`. Decoder target becomes `problem + = + <eos>`.
The LLM-Computer interpreter handles every value via `safe_eval`.
Result: 245K → 48K params, 15 min → 145 sec training,
43% → 96.7% full-expression accuracy.

### Sweet-spot config (all production domains)

```bash
# Math (3-digit operands)
PYTHONPATH=. python3 -m calm.hrm.train --seq2seq --structure-only \
  --hidden 32 --num-heads 4 --l-layers 1 --h-layers 1 --dec-layers 1 \
  --epochs 500 --lr 1e-3 --problems 2000 --batch-size 128 \
  --max-enc 32 --max-dec 32

# NL templates, word problems, GSM-style, multi-task (pooled):
# standalone trainers at scripts/train_hrm_{nl,word,gsm,multi}.py.
# Same 48K architecture; only --max-enc changes per domain
# (48 / 80 / 128 / 128 respectively).
```

All 5 production checkpoints: 48,864 params. Best-val-acc selection
lands between epoch 100-300. Training time: ~145s (math) to ~800s
(GSM / multi-task) on RTX 4070.

### Rule: ALWAYS `--epochs 500`, never `--epochs 100`

Cosine LR schedule over only 100 epochs under-fits on any NL domain.
The LR decays to ~0 before the model converges on digit-copy
precision. Set `--epochs 500` generously; rely on `best_val_acc`
checkpoint selection to pick the right moment.

## Pointer Transducer (PT) training — replaces HRM for new work

**Architecture**: `CopyAugmentedTransformer`
(`calm/llm_computer/copy_augmented.py`). Decoder-only
`Small2DTransformer` + learned copy gate + pointer attention. 1,089
extra params. Forward returns **log-probs** — use `F.nll_loss`, not CE.

**Key training rules for PT:**

- **Balanced `_sample_operand()`**: uniform across digit-length
  buckets [1-9] / [10-99] / [100+]. Without this, small operands get 0%.
- **`max_len` ≥ max_prefix + max_expression + decode_headroom**:
  positional embeddings cap sequence length. CUDA assert if autoreg
  exceeds it.
- **One PT per output-language family**: function-call, infix
  arithmetic, boolean logic. Combined model plateaus at 74%; split
  recovers 86-88%.
- **Autoreg eval is the gate**: teacher-forced val_acc is misleading
  (99.6% while autoreg is 74%). Always use `_autoreg_eval` — *on
  raw/unaugmented val*. Split BEFORE aug; paraphrase-augmented val
  contains variants of train problems and autoreg becomes memorization.
- **Copy gate bias = -2.0** (stable on retrieval / NL-math).
  Code-skeleton regime needs `-1.0` + aux copy-loss
  (`--copy-aux-weight 0.5`) — without aux, gate collapses to ~0.018
  and model becomes gen-only.
- **VOCAB_SIZE = 82** (added `><` token).

**Remaining ceiling**: 3+ operand copy accuracy (68-83%). Copy
attention over prefix gets noisy with 3+ numbers. Known fix:
two-stage decode via D5 recurrence.

**Knowledge DB as alternative to training**: for domains where the
value is knowing rules, not computing values (creative writing,
legal, style), a knowledge backend (`*_kb.py`) with dictionary
entries is more effective than training. Each entry = 3 ReGLU
neurons when compiled into substrate.

**Accuracy priority order:**
1. Data distribution — every valid input region covered? (free)
2. Mechanism — right operation? e.g. copy vs generate (cheap)
3. Output-family split — too many output languages in one model?
   (moderate)
4. Capacity — model genuinely too small? (expensive, last resort)

## Legacy HRM training notes

Per-token val_acc is inflated by trivial-copy tokens. **The gate
that matters is full-expression accuracy**, not the trainer's
reported number.

Scratchpad-with-intermediate-values forces memorization that small
models can't deliver. **Stop asking the model to compute; let it
emit structure and route values to the substrate.**

**HRM size scales with input-language complexity, NOT problem-
difficulty.** PT size scales with **output-language complexity** —
the number of structurally distinct expression formats. Both scale
independently of value range.

## DT (PT+Delta) defaults

New trained cards default to DT (`CopyAugmentedDeltaNet`) for
retrieval + structure-extraction regimes. Held-out parity on copy-
dominant structure tasks, large gains on retrieval-shaped tasks,
faster training convergence. Full recipe + install: `delta_rule.md`.

**Default config for MQAR/NL-math cards**:
`use_chunkwise=True, n_delta_heads=1, n_iterations=1, chunk_size=32`.

**Training recipe differs from plain PT**: `F.nll_loss` (not
`F.cross_entropy`) because forward returns log-probs; chunkwise
gives 3-7× per-epoch speedup; data budget scales with N per the
MQAR curve: **"+5 on N needs 2× training data"** (2K/N → 5K/N →
10K/N for N=5-10 / 15 / 20).

**Code-skeleton DT recipe** differs from MQAR/NL — see `delta_rule.md`
§"Code-skeleton recipe" for canonical flags (aux copy-loss, split-
before-aug, EMA 0.995, sqrt_inverse sampler).

## Pitfalls observed

- **Cosine LR scheduled to 0 too early kills learning.** Mandatory
  `--epochs 500` default on any NL-input HRM.
- **Smoke cases failing at 3-digit numbers** are out-of-distribution.
  Match the operand range to smoke cases you care about (e.g. bump
  `_arithmetic_simple` from `randint(1, 99)` to `randint(1, 999)`).
- **Per-token > 90% but structural < 50%** means the model nailed
  structure tokens (operators, parens, function names) but
  mis-copied digits. Use `_hrm_raw_emit()` from eval scripts to
  inspect raw output — add data OR scale capacity.
- **Per-domain `max_enc` is load-bearing.** Math fits in 32, NL in
  48, word problems in 80, GSM in 128, multi-task in 128. Undershoot
  and the sentence truncates silently mid-operand.
- **Triton-autograd works in isolation but fails against
  PyTorch-captured teacher targets.** Triton's different FP32
  reduction order produces ~6e-5 forward drift vs PyTorch `F.linear`;
  compounds through Gemma nonlinearities. **Rule**: if using Triton
  autograd, re-capture teacher targets through the same Triton path.

## Substrate-Native Training (SubstrateLM / SubstrateHRLM)

Different from HRMSeq2Seq. Substrate-native cards are decoder-only
`Small2DTransformer` instances trained on chat-formatted text, then
compose with compiled programs and HRM specialists at runtime.

**Compiled cards need NO retraining to install into prod Gemma**:
`install_card_in_attention` writes existing gate-graph IR weights
into FP32-converted `attn_q/k/v/output`. End-to-end auto-upgrade
loop: detect → log → compile recall card → install → persist. Demo:
`scripts/gemma_learning_loop_demo.py`.

### Config (MVP scale, proven)

- `d_model=96-192`, `n_heads=d_model/2` (d_head=2 invariant),
  `n_layers=3-5`, `d_ffn=2-4× d_model`, `max_len=256-512`.
- BPE tokenizer via `tokenizers` lib, `vocab=4096-16384`. ByteLevel
  pre-tokenizer AND ByteLevel decoder (forgetting the decoder breaks
  sample readability).
- Special tokens: `<|sys|>`, `<|user|>`, `<|asst|>`, `<|eos|>`,
  `<|pad|>` + mode prefixes `<|lm|>`, `<|hrm|>` for hybrids.
- Loss masking: `train_on_responses_only` — only assistant tokens
  get gradient. For HRM-mode examples in hybrids, the mask is the
  expression text after `=`.

### Cross-task transfer measurement (1+1=N)

When training a hybrid card, measure each mode's accuracy separately
against single-task baselines:

- **1+1=2** → modes coexist (no degradation). Acceptable PASS.
- **1+1>2** → cross-task transfer (one mode improves the other).
  Strongest PASS; evidence for shared representations.
- **1+1<2** → modes interfere. Investigate token ratios, loss
  weighting, mode-token strength.

### Per-mode iteration during training (D5)

- HRM-mode examples in hybrid training run with `n_iterations=2-3`
  (more thinking on structure tasks).
- LM-mode examples run with `n_iterations=1` (one pass for fluent
  generation).
- At inference, `n_iterations` is dispatched per mode token.
- Training cost: ~1.5× per-step time when batch contains HRM examples.

### GPU vs CPU decision rule

All substrate training scripts accept `--device auto`.

- **Stay on CPU**: model <500K params AND seq <128 tok AND
  pure-Euclidean AND no D5.
- **Move to GPU**: model >2M params, seq >256, D3 mixed geometry
  (hyperbolic `acosh` doesn't vectorize on CPU), D5 recurrence
  (serial kernel launches), or any combination — effects compound.
- Observed GPU speedup: 6× (not 10-20×) because D5 serial Python
  loop bottlenecks.

**GPU prerequisite**: Gemma must NOT be in VRAM. `pkill llama-server`
before launching; verify via `python3 -c "import torch; print(torch.cuda.is_available())"`.

### Safer-config for noisy-grad training

When batch is small, prompts are mixed-loss, and you're on Adam/AdamW:

- **batch ≥ 4**, **grad_clip ≤ 0.1**, **lr ≤ 3e-4**, **warmup ≥ 200**.
- **Diagnose Adam momentum poisoning by EMA**: if loss spikes and
  EMA climbs 20+ steps without recovery, optimizer state is
  corrupted — kill and restart; continued training won't recover.

### Scheduled Sampling for SubstrateHRM

**The training trick that tripled autoregressive accuracy** (typical:
33% → 90%). Without it, the model reports high val_acc under
teacher-forcing but can't recover from its own decode errors.

**Fix**: during training, randomly replace teacher-forced tokens
with the model's own predictions. Ratio decays linearly:
`tf_ratio: 1.0 (pure teacher) → 0.3 (70% self-generated)` over 500
epochs.

Implementation (`scripts/train_substrate_hrm.py`):
- 2-pass per batch: (1) no-grad forward to get predictions,
  (2) vectorized swap via `torch.where(swap_mask, shifted_preds, x)`,
  (3) gradient forward on modified input.
- **Vectorize the swap** — Python loop over positions is CPU-bound
  at 95%. Use `torch.where` for the full batch in one op.
- **Autoreg eval** (`_autoreg_eval`): greedy-decodes from NL prompt,
  checks exact match. This is the gate metric. Save checkpoint on
  best autoreg, not teacher-forced accuracy.

**Data distribution matters**: if operand range is `[1, 999]`,
single-digit operands ([1, 9]) are underrepresented in random
draws → 25% on small-operand test. Include explicit small-operand
examples in training data.

**Rule**: always use `--scheduled-sampling` (default ON) when
training SubstrateHRM. Cost is ~2× per epoch (two forwards); benefit
is the difference between 33% and 90% on the metric that matters.

### Compiled-vs-trained card distinction

- **Compiled cards** (gate-graph IR → weights) — no training, exact,
  programmatic. Lives in `calm/llm_computer/programs/`.
- **Trained cards** (substrate-native or HRMSeq2Seq) — SGD,
  statistical. Lives in `calm/llm_computer/checkpoints/` or
  `calm/hrm/checkpoints/`.
- **Future: distill-to-compiled** — detect stable learned patterns
  and replace with exact compiled equivalents.

## Dataset Quality

- Claude-authored/hand-written data >> 9B-generated data (higher
  quality, more consistent)
- HuggingFace datasets (nohurry, TeichAI, Crownelius) provide Claude
  Opus reasoning traces — filter aggressively
- Hand-written data committed to repo: `coding_reasoning_claude.jsonl`,
  `orchestrator_claude.jsonl`
- Filter pipeline: tiered keyword matching (1 strong keyword + 2
  general, OR 5+ general, OR code blocks), dedup by first 60 chars,
  think-block minimum lengths
- Filter out: hallucinated facts, non-technical content, NLP
  benchmark patterns, junk (<200 char responses)

## Substrate eval defaults

**Canonical module**: `calm/llm_computer/eval_defaults.py`. Four
constants govern every substrate eval:

```python
from calm.llm_computer.eval_defaults import (
    EVAL_CTX_SIZE,       # 32768 — pre-allocated tq4 KV ceiling
    EVAL_MAX_TOKENS,     # 16384 — AdaptiveBudget output clamp
    ITERATION_N,         # 5 — fast-iteration problem count
    FINAL_N,             # 20 — commit-baseline problem count
    resolve_problem_window,
    get_adaptive_budget,
)

n, skip = resolve_problem_window()
budget, est = get_adaptive_budget(prompt)
out = m.generate(prompt, tok, max_tokens=budget,
                 use_tq4_kv=True, kv_max_len=EVAL_CTX_SIZE)
```

- `EVAL_CTX_SIZE=32768` pre-allocates `KVCacheTq4`. tq4 is ~3.6×
  smaller than fp16 so 32K costs ~700 MB on top of ~5 GB substrate
  baseline. Leaves headroom for 1-2 FP32 host layers.
- `EVAL_MAX_TOKENS=16384` is the AdaptiveBudget clamp. Tiered
  (trivial 2K / easy 4K / medium 8K / hard 16K / deep 32K) but
  always clamped. Any budget < 4K truncates real coding problems
  mid-function.
- `bin/mbpp-rotate N` writes window state. `MBPP_FINAL=1` env
  toggles final mode per-script.
- **Exception**: dual-gate eval scripts use `K_TOKENS=12` as a
  measurement design (prefix-match on teacher-vs-student first-K
  tokens), not an eval budget. Don't migrate to EVAL_MAX_TOKENS.

## Known Issues

- **Qwen 3.5 248K vocab**: fused cross-entropy loss OOMs on anything
  under 40GB VRAM for 4B, under 8GB for 0.8B.
  - 0.8B fix: batch_size=1, max_seq_length=1024, packing=False
  - 4B fix: use cloud GPU (Colab A100)
- Unsloth compiled cache stored at `unsloth_compiled_cache/` in
  project root (gitignored)
- Git Bash mangles WSL paths with parentheses in PATH — use
  `wsl -e bash -c` or write scripts to `/tmp/`
- **WSL Windows-binary stdin consumption**: any `*.exe` called from
  a bash script consumes parent stdin via WSL's interop shim. Always
  pass `< /dev/null` to Windows binaries in scripts that may receive
  piped stdin.
- **llama.cpp slot context cap**: `tools/server/server-context.cpp`
  silently caps each slot's context to `n_ctx_train`. Patched locally
  to comment out `n_ctx_slot = n_ctx_train`. Re-apply after `git pull`.
  Required for 256K-range NIAH testing on Gemma 4 E4B.
- **Gemma 4 GGUF rope-scaling metadata override**: Gemma 4 E4B's GGUF
  bakes in `rope scaling = linear`; `--rope-scaling yarn` CLI flag
  is silently ignored.
- **llama-server `--parallel` default**: defaults to 4 slots; each
  slot gets `ctx_size / parallel`. For the harness's single-user
  workflow always pass `--parallel 1`.
- No Thunderbolt/eGPU on Acer Nitro AN17-42 — cloud GPUs required
  for 4B+ training.
- RunPod SSH proxy unreliable from WSL2 — use web terminal or Colab.

## Quantization

See `.claude/rules/turboquant.md` for full tq4/tq3 kernel details and
block format. Training-output pipeline:

```bash
# f16 → tq4 (weights + KV cache compatible)
llama-quantize model-F16.gguf model-tq4.gguf TQ4_K256

# ALWAYS detach long quantize jobs:
setsid ~/llama.cpp/build/bin/llama-quantize \
  ~/models/model-F16.gguf ~/models/model-tq4.gguf \
  TQ4_K256 4 < /dev/null > /tmp/quantize.log 2>&1 &
disown -a
# Verify: xxd model-tq4.gguf | head -1  (must show 'GGUF')
```

### Quantize process management (MANDATORY)

- **ALWAYS `setsid` + `disown`** for any quantize >30s. Without
  this, a CC crash or WSL OOM kills the child process mid-write,
  corrupting the output.
- **ALWAYS 4 threads**. Single-thread E4B quantize takes 10+ min vs
  ~6 min with 4 threads. Extra threads don't meaningfully increase
  RAM (mmap dominates).
- **ALWAYS redirect stdin** from `/dev/null` to prevent WSL interop
  stdin consumption.
- **NEVER run quantize while building llama.cpp** — stacked nvcc +
  quantize can push WSL past 28 GB and crash the VM.
- **ALWAYS verify** the output GGUF header: `xxd model.gguf | head -1`
  must show `4747 5546` (`GGUF`). All-zeros = corrupt; delete and retry.

### WSL2 OOM during quantization

- `llama-quantize` mmaps the source GGUF (~15 GB for E4B f16) and
  needs working RAM for Pi rotation buffers. On 28 GB WSL2 this can
  OOM and crash the entire WSL VM.
- Fix: `disown` the process so CC crashes don't kill it
- For very large models (26B+), `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`
  before quantizing
- Corrupt-GGUF symptom: `invalid magic characters: '????'` — delete
  and re-quantize.

## Export & Serving

Canonical serving docs in `.claude/CLAUDE.md` §"Serving Architecture".
Quantization-specific details in `rules/turboquant.md`.

Training-output path: merge LoRA → GGUF via `llama-quantize` → serve
via `llama-server` (OpenAI-compatible API) or `ollama create` from a
Modelfile in `models/`.
