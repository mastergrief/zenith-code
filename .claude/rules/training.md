# Training Rules

Training part 1

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

### HRM checkpoint inventory (`calm/hrm/checkpoints/`)

**Production HRMSeq2Seq (48K params, `--structure-only`):**
- `math_structure_best.pt` — 3-digit math (session 26 sweet spot)
- `nl_math_structure_best.pt` — NL templates
- `word_problem_best.pt` — word problems with names/pronouns
- `gsm_best.pt` — GSM-style narratives
- `multi_task_best.pt` — all four domains pooled

**Production Pointer Transducer (~185K params, session 31):**
- `copy_augmented_hrm_best.pt` — NL math (infix family)
- `copy_word_best.pt` — word problems
- `copy_gsm_best.pt` — GSM-style
- `copy_funcall_best.pt` — function-call family (percentage, ratio)
- `copy_logic_best.pt` — boolean logic family
- `copy_reasoning_best.pt`, `copy_writing_best.pt` — bonus domains

**PT+Delta (default for new cards post-R-delta-20, 2026-04-21):**
- `copy_augmented_delta_best.pt` — NL math via DeltaNet backbone
  (R-delta-6a, 100% val autoreg ep15). 183,877 params.
- `copy_augmented_delta_mqar_best.pt` — deployable MQAR card
  (R-delta-21, 100% held-out on N=5/10/15). Trained by
  `scripts/train_pt_delta_mqar.py` (5K/N × N=[5,10,15] × 20 ep,
  chunkwise + scheduled sampling, ~2 min wall).

Default config for new domain cards (commit `63a49fc`):
`use_chunkwise=True, n_delta_heads=1, n_iterations=1, chunk_size=32`.
Full rule: `.claude/rules/delta_rule.md`.

Training recipe differs from plain PT: `F.nll_loss` (not
`F.cross_entropy`) because forward returns log-probs; chunkwise
gives 3-7× per-epoch speedup; data budget scales with N per the
MQAR curve: +5 on N needs 2× training data (2K/N → 5K/N → 10K/N
for N=5-10, 15, 20 respectively).

**Substrate-native (180K params, session 30):**
- `substrate_hrm_nl_best.pt` — first HRM on Small2DTransformer substrate, 90% autoregressive on NL math (scheduled sampling)

**Historical / experimental (kept for eval comparison, not
production):**
- `math_hrm_best.pt` — legacy encoder-decoder (pre-structure-only, round 1a-1d)
- `math_scratchpad_best.pt` — scratchpad-with-intermediate-values variant (session 25 negative result — memorization trap)
- `math_seq2seq_best.pt` — seq2seq baseline, round 1a
- `math_structure_2digit.pt.bak` — backup, 2-digit operand variant
- `multi10_best.pt`, `multi20_best.pt` — multi-task curriculum iterations (session 26, 10 and 20 template variants)
- `meta_best.pt` — meta-reasoning experiment, session 26
- `router_best.pt` — keyword-routing network for `substrate_server.py` (~38 KB)
- `synth_familyA_best.pt`, `synth_familyA_distilled.pt` — session 27 self-distill experiment (teach once → library accretes → fold back via fine-tune)

If you're evaluating a checkpoint whose purpose isn't obvious from
the filename, read the git log for the script that created it
(`scripts/train_hrm_*.py` or `scripts/train_copy_*.py` or
`scripts/train_substrate_*.py`).

### Pitfalls observed (sessions 25-26)

- **Cosine LR scheduled to 0 too early kills learning.** See the `--epochs 500` rule above — this is now elevated from pitfall to mandatory default on any NL-input HRM.
- **Smoke cases failing at 3-digit numbers** are out-of-distribution. `MathDataGenerator`'s `_arithmetic_simple` was capped at `randint(1, 99)`; bumped to 999 in session 26 step 1. Any new domain: match the operand range to the smoke cases you care about.
- **Per-token > 90% but structural < 50%** means the model nailed structure tokens (operators, parens, function names) but mis-copied digits. Use `_hrm_raw_emit()` from eval scripts to inspect raw output — if structure is good but digits drift, add data OR scale capacity. The `--verified` mode (LLM-Computer parses the input directly) masks this class of failure when full-expression is the gate but reveals it when structural-match is the gate.
- **Per-domain `max_enc` is load-bearing.** Math fits in 32, NL in 48, word problems in 80, GSM in 128, multi-task in 128 (max of components). Undershoot `max_enc` and the sentence is truncated silently mid-operand; overshoot and you waste compute. The canonical trainer scripts (`scripts/train_hrm_{nl,word,gsm,multi}.py`) document the correct bound per domain.
- **Triton-autograd works in isolation but fails against PyTorch-captured teacher targets (R52.1c).** `Tq4TritonAutogradFunction` passed `gradcheck` and cosine=1.0 on 17-linear chain, BUT training diverged: Triton's different FP32 reduction order produces ~6e-5 forward drift vs PyTorch `F.linear`; compounds through Gemma nonlinearities. **Rule**: if using Triton autograd, re-capture teacher targets through the same Triton path — don't mix. Kernels kept at `tq4_autograd.py` + `tq4_triton.py::tq4_backward_triton`.

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

