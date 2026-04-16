# Session Handoff — 2026-04-16 (Session 31)

Branch: `feature/multi-agent-qwen`
Prior handoff (session 30): ended at `1a7da0e` (unified tensor docs).
This session: **5 commits, ~2,680 lines added**, invented the Pointer
Transducer (copy-augmented `Small2DTransformer`), validated across 4
domains, established the output-language-family split principle, and
built the `/domain` command for repeatable domain addition.

## Goal

Fix HRM data distribution gap (0% on single-digit operands, 90% ceiling
overall) and generalize the fix across all domains. Session evolved into
a deeper architectural shift: replacing HRM-style generation with
pointer-copy transduction, splitting domains by output-language family,
and building the repeatable workflow for adding domains to the substrate.

## Completed (5 commits)

### Pointer-copy mechanism (`CopyAugmentedTransformer`)

**The key invention of this session.** Added a learned copy gate +
pointer attention to `Small2DTransformer`. 1,089 extra params (0.6%).
At each decode step the model chooses: generate from vocabulary OR copy
from an input position. Digits get copied exactly; operators get
generated.

File: `calm/llm_computer/copy_augmented.py`

- `CopyAugmentedConfig` — extends `Small2DConfig` with `n_copy_heads`, `sep_token_id`
- `CopyAugmentedTransformer` — subclasses `Small2DTransformer`, adds copy gate + pointer attention
- `build_copy_augmented_hrm()` — factory matching existing `build_substrate_hrm()` signature
- Forward returns log-probs (not raw logits) — use NLL loss, not CE

### Cross-domain validation

| Domain | Commit | Max input | Val autoreg | Held-out | Training time |
|---|---|---|---|---|---|
| NL math | `25ab154` | 30 chars | 100% | 200/200 | 38s |
| Word problems | `689076f` | 78 chars | 98% | 96/100 | 248s |
| GSM-style | `95e0a61` | 104 chars | 100% | 95/100 | 491s |
| Funcall reasoning | `608db13` | 88 chars | 86% | 171/200 | 611s |
| Logic reasoning | `608db13` | 121 chars | 86% | 88/100 | 910s |

### Old ceilings broken

| Ceiling | Old | New | How |
|---|---|---|---|
| Single-digit operands | 0% | **100%** | Balanced data distribution |
| 3-digit operands | 68% | **100%** | Copy mechanism |
| GSM 28/30 (93%) | 93% | **95%** | Copy mechanism |
| Syllogism | 36% | **92%** | Output-family split |

### Output-language family split principle

Combined 9-category model plateaued at 74%. Diagnosis: two structurally
different output languages (function-call syntax vs infix operators) in
one model. Split into:

- **Funcall PT** — `percentage(x, y)`, `sequence_cost(...)`, `multi_max(...)`, `ratio_simplify(...)` → 86% overall (100% on 2-arg)
- **Logic PT** — `x > y`, `x > y and y > z`, `z if x > y else w`, `x + y - z` → 88% held-out

**Principle: one PT per output-language family, not per domain.** ~3-5
PTs cover 30+ domains. Adding a domain within an existing family is a
data-only operation (write templates, retrain).

### Balanced data distribution

All data generators (`nl_data.py`, `word_data.py`, `gsm_data.py`,
`reasoning_data.py`) now use `_sample_operand()` with uniform digit-
length bucketing: equal probability across [1-9], [10-99], [100+].

### Grammar-constrained decoding (null result)

`calm/llm_computer/grammar_decode.py` — inference-time mask for valid
math expressions + EOS boosting. **Null result**: 96/100 → 96/100 on
word problems (0 fixes, 0 regressions). Failures are semantic (model
errors), not syntactic. Infrastructure shipped for future use.

### Vocab expansion

`calm/hrm/data.py`: `_CHARS` expanded with `><`. VOCAB_SIZE 80 → 82.
Existing checkpoints (saved with `vocab_size=80` in config) unaffected.

### CALM reasoning backends

- `calm/backends/reasoning_ops.py` — 11 functions: `chained_eval`, `compare`, `conditional_eval`, `sequence_cost`, `syllogism_check`, `multi_max`, `multi_min`, `percentage`, `ratio_simplify`, `ratio_decimal`, `step_by_step` + 14 NL patterns
- `calm/backends/reasoning_kb.py` — 7 functions: syllogism forms (10 Aristotelian), logical fallacies (6), transitive/non-transitive relations + 5 NL patterns

### `/domain` slash command

`.claude/commands/domain.md` — 7-step guided workflow for adding
domains to the CRLM stack. Uses `AskUserQuestion` at every decision
point. Steps: scope → CALM backend → compiled card → templates → train
PT → evaluate → install.

## In Progress

### Nothing running or half-shipped

All training runs killed at convergence. GPU free. 281 tests pass
(up from 250 — new backends auto-registered).

## Next Steps (priority order)

### 1. Two-stage decode via D5 recurrence (highest leverage)

The remaining accuracy ceiling across all domains is 3+ operand copy
errors (68-83% on 3-4 arg expressions). Two-stage decode fixes this:
iteration 1 emits structure skeleton, iteration 2 fills operand slots
independently. Implementation:

- Add `n_iterations=2` to `CopyAugmentedTransformer.forward()` via
  `RecurrentConfig` from `calm/llm_computer/recurrent_substrate.py`
- During iteration 1: copy gate stays low (generate structure)
- During iteration 2: copy gate goes high (fill slots by pointing)
- Retrain with `n_iterations=2` — same data, same params, 2x forward time
- Expected: 3+ operand accuracy from 68-83% → 95%+

### 2. Multi-domain training within a family

Train one funcall PT on pooled data from multiple domains (percentage +
temperature + chemistry + geometry). Test for cross-domain transfer
(does adding temperature data improve percentage accuracy?). If yes,
one PT truly serves N domains.

### 3. Wire auto-upgrade into zenith harness

`AutoUpgradeEngine` exists but isn't called from `agents/harness.py`.
Integration: init (load .pt), query (verify via CALM), exit (compile
corrections + save). ~5 lines at 3 call sites.

### 4. Install PTs into unified substrate

Script that takes N PT checkpoints and `install_compiled_card_hybrid`
into the unified tensor at reserved sub-head offsets. Prove 2+ PTs
coexist with Gemma + compiled cards in one forward pass.

### 5. Update CLAUDE.md and rules

Session 31 discoveries not yet reflected in docs:
- Copy-augmented architecture (new module, new training pattern)
- Output-language family principle
- `/domain` command
- Vocab expansion to 82
- Balanced data distribution as a rule
- Corrected principle: "Model understands, transducer structures, cards compute, engine verifies"

## Key Context

### Critical design decisions

1. **Pointer-copy is additive** — `CopyAugmentedTransformer` subclasses
   `Small2DTransformer`. The copy gate, copy Q/K projections are the
   only additions. Removing them recovers base behavior.

2. **Copy gate bias initialized at -2.0** — model starts by preferring
   generation (existing behavior) and learns to copy. Without this,
   early training is unstable.

3. **Forward returns log-probs, not logits** — because the copy
   distribution is a probability (from scatter_add of attention
   weights), not logits. Use NLL loss, not CE. Every training script
   uses `F.nll_loss`.

4. **`max_len` must exceed max_prefix + max_gen** — the positional
   embeddings cap sequence length. During autoreg eval, the sequence
   grows beyond training length. CUDA assert if exceeded. The autoreg
   eval function now caps `gen_budget = min(max_gen, pos_limit - len(ids) - 1)`.

5. **Output-language family split** — one PT per output syntax family,
   not per domain. Function-call (`fn(args)`) and logic (`a > b and`)
   are different output languages. Forcing both into one 185K model
   plateaus at 74%. Splitting recovers 86-88%.

### Failed approaches (don't retry)

1. **Grammar-constrained decoding for word problems** → null result.
   Failures are semantic (model generates valid but wrong expressions),
   not syntactic. Grammar mask + EOS boost traded 2 fixes for 3
   regressions (truncated multi-digit numbers). Space-only EOS boost
   was safe but didn't trigger.

2. **50/50 small/large operand split** → overcorrected. Small 0→100%
   but mid collapsed to 2%. Fix: 3-bucket uniform digit-length split.

3. **Combined 9-category reasoning model** → plateaued at 74%.
   Diagnosis: two output languages competing. Fix: split by family.

4. **`>` and `<` not in char vocab** → CUDA assert during training.
   Expression `5 > 4` tokenizes as `5 4` (operator dropped), training
   on garbage targets. Fix: add `><` to `_CHARS` in `data.py`.

5. **max_len=160 with 138-token prefixes** → CUDA assert during
   autoreg eval (prefix + generated tokens exceed positional embeddings).
   Fix: increase max_len to 208, cap gen_budget in eval.

### Environment state

- Branch `feature/multi-agent-qwen`, ~228 commits ahead of origin.
- 6 new copy-augmented checkpoints in `calm/hrm/checkpoints/copy_*_best.pt`
- `substrate_hrm_nl_best.pt` retrained with balanced data (94% autoreg)
- llama-server NOT running. GPU free.
- 281 tests passing (up from 250 — new CALM backends).
- VOCAB_SIZE = 82 (was 80, added `><`).
- RTX 4070 (8 GB VRAM) + 32 GB DDR5.
- Python 3.13.7, PyTorch 2.10.0+cu128.

### User's key insights (important for tone and direction)

- "So it's an actual brain with sub-regions for different tasks" — the substrate is literally a brain architecture: specialized regions (sub-heads), shared wiring (channels), one forward pass
- "Model understands, transducer structures, cards compute, engine verifies" — corrected principle replacing "model reasons"
- "One PT per output-language family, not per domain" — the scaling insight
- "Balanced data distribution vs capacity" — data coverage and mechanism fit before scaling capacity
- "The system reasons through composition" — no single component thinks, the pipeline produces reasoned answers

### Gotchas for next session

1. **Copy model returns log-probs** — use `F.nll_loss`, not `F.cross_entropy`. Every `train_copy_*.py` script does this correctly. Don't mix with base `Small2DTransformer` which returns raw logits.

2. **VOCAB_SIZE is now 82** — new models must use 82. Old checkpoints saved `vocab_size=80` in their config dict and load fine (the builder uses the saved config, not the global VOCAB_SIZE).

3. **max_len must account for autoreg** — `max_len >= max_prefix_len + max_expression_len + decode_headroom`. Prefix can be 138 tokens for reasoning templates. The autoreg eval caps gen_budget to stay within positional embedding range.

4. **3+ operand copy errors** — consistent across all domains. The copy attention over the prefix gets noisy with 3+ numbers to locate. Known fix: two-stage decode via D5. Don't try to fix with more data or capacity — it's a mechanism problem.

5. **Balanced `_sample_operand()`** — every data generator must use it. Uniform sampling from `[1, max_val]` underrepresents small digits (~1% of draws). The 3-bucket approach gives 33/33/33 across digit lengths.

## First action on resume

1. Read this handoff.
2. Update CLAUDE.md and rules with session 31 discoveries (copy mechanism, output-family split, `/domain` command, vocab 82, corrected architecture principle).
3. Prototype two-stage decode: add `n_iterations=2` to `CopyAugmentedTransformer`, retrain funcall PT, measure 3+ operand accuracy.
4. If two-stage works: retrain all domains with it, then install multiple PTs into the unified substrate tensor.
