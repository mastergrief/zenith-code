# Training Rules

> Historical receipts (session-25/26/27/30/31 training-arc anecdotes,
> R-numbered flag tables, HRM → PT evolution receipts, SubstrateLM
> MVP empirics, SubstrateHRLM v1/v2 hybrid receipts, dataset-addition
> specifics): see `MEMORY/atlas/training_part_1.md` +
> `MEMORY/atlas/training_part_2.md`. Quantization details (tq4/tq3
> block format, kernels): `rules/turboquant.md`.

## HRM-Text-1.58 Fork: Progressive Checkpoint Curriculum

**The active training lane.** The fork target is **`hrm-158-base`**, a
robust all-rounder native HRM-Text-1.58 checkpoint. PT/DT/Substrate
guidance below remains as **legacy/adjacent** reference for retrieval /
structure-extraction lanes; native HRM-Text-1.58 is now the primary
training lane for `hrm-158-base`.

**Canonical workflow: see `rules/hrm-158.md`.** The bank gate (acquire ≥90% /
retain ≥90%), the **auditable-full-density-default + slow-safe-learning** slice
recipe (bounded stair-step is the FALLBACK after a classified collision /
oversized support), retention mechanisms (replay + parent consistency + broad
retained supports + close-sibling protection), and failure-mode classification
all live there. This section keeps the model specifics + literal operational
command invocations; `hrm-158.md` is canonical for the policy.

### Model + tokenizer

- ~29.6M params, Tier B config (`hidden=512 n_layers=8 num_heads=4
  expansion=4 H_cycles=2 L_cycles=3 max_len=384`).
- Fixed broad byte tokenizer (`vocab=260`, normalizer `byte_utf8_v1`)
  across math / language / code — DO NOT swap mid-curriculum.
- Ternary bulk linears (BitLinear on q/k/v/o/gate/up/down; `lm_head`,
  `embd`, norms stay FP per the D2.2 contract).
- Native ternary training kernel (Triton fused-quantize STE-correct
  backward) enabled via `--use-native-ternary-train`.
- Fast probe path: cached ternary inference + KV decode cache + batched
  probe eval (single forward over equal-prefix groups, batch size 32).

### Loss contract

Response-only loss: prompt/instruction tokens are masked; loss is only
on generated response tokens.

### Progressive checkpoint curriculum (the loop)

1. **Start from latest banked checkpoint** (the current chain head).
2. **Train one auditable finite-support capability slice** (a curriculum rung,
   full-density when small enough to audit completely; bounded fallback
   otherwise — e.g. K=N addition under a carry-stratified partition, or a
   paraphrase block).
3. **Replay important prior rungs** in the same training run; do NOT
   train new-data-only. Fragile slices default to `--replay-ratio 0.80`
   (80% replay, 20% new rung data); simple non-fragile rungs may use
   `0.65`.
4. **Keep tokenizer broad and fixed** across math / language / code.
5. **Promote a checkpoint only after** sampled probes + A0 exhaustive
   finite-support audit + explicit watch rows prove acquisition AND clear
   **under the named gate semantics** (true priors at the ≥90% retain bar;
   close siblings reported, blocking bank only on a broad parent-relative
   cluster when that is the declared semantics). **Bank the earliest checkpoint
   that clears all hard gates — the final checkpoint has no privilege.**
6. **If failures appear, classify before changing recipe**: train-set
   miss / held-set generalization residual / parent-relative cluster /
   signal-starvation. Each class has a different repair shape.

### Default recipe — slow-safe learning (gabe-locked)

**Slow-safe learning is the default method, not just for fragile slices.** The
banked full-density recipe is full-density support PLUS slow-safe learning (one
atom — full-density without slow-safe is not the method). Lower update pressure
is the retention knob; higher lr migrates digit/template clusters into prior
rungs. `rules/hrm-158.md` §"Recipe band" is canonical; the band: LR ~`5e-5`,
replay ~`0.80`, ≤1500-step window, pc on acquired priors, **no knob escalation
on a miss**. The producer/consumer audit watcher
(`scripts/parallel_audit_watcher.py`) is **required** (must prove OVERLAP per
save step — only OVERLAP-clean saves are bank-eligible; else
SERIAL_FALLBACK/MISSED unless explicitly waived). **Pre-launch: verify box
code-currency** (probe/watcher/rung files synced), not just reachability.

```
--curriculum-rung <rung>
--use-broad-tokenizer
--load-from <banked-chain-head>.pt
--replay-rungs <comma-separated prior rungs>
--replay-ratio 0.80
--lr 5e-5
--curriculum-n-train 12000
--curriculum-seed 17
--parent-consistency-weight 1.0 --parent-consistency-temp 1.0
--retained-support L0b:1.0 --retained-support math_a0:1.0
--retention-anchor-set math_fragile_v1 --retention-anchor-repeat 3
--use-ternary-bulk --use-native-ternary-train
--save-at-step 250 --save-at-step 500 --save-at-step 750 \
--save-at-step 1000 --save-at-step 1250 --save-at-step 1500
```

**Non-fragile exception (evidence-gated)**: a simple math rung that
proves it tolerates more pressure may use faster lr (e.g. `--lr 5e-4
--replay-ratio 0.65`, saves 500/750/1000). Default stays slow-safe
until a rung earns the exception.

For targeted-repair runs that need to replay over positionally-future
rungs, add `--allow-future-replay`.

### Validation

- **A0 exhaustive finite-support audit**: every row of every active
  rung's support decoded via the faststack path; strict-exact match
  AND parsed-correct reported separately:

  ```
  PYTHONPATH=. python3 scripts/probe_hrm_text_158.py \
    --ckpt-path calm/hrm/checkpoints/<run>_final.pt \
    --exhaustive-finite-supports \
    --watch-rows-json /tmp/<run>_watch_rows.json \
    --audit-output-json /tmp/<run>_audit.json \
    --use-cached-ternary-infer --use-kv-cache-decode \
    --use-batched-probe-eval --probe-batch-size 32
  ```
- **Sampled probe** for trend tracking: same script with
  `--curriculum-rungs <comma-separated rung list>` (per-rung 50
  samples).
- **Watch rows**: explicit list of edge / boundary / known-residual
  rows passed via `--watch-rows-json`. Each is `{key, question,
  expected}`; results reported as `exact_ok`/`parsed_ok`.
- **Keyed one_digit audit**: 9-row exhaustive per audit-eligible
  rung via `ONE_DIGIT_AUDIT_REGISTRY` in
  `scripts/probe_hrm_text_158.py`. Every audit-eligible rung present
  in the probe gets a keyed audit (no silent retention drops).

### Failure-mode classification

| Class | Signal | Repair |
|---|---|---|
| **Train-set miss** | row IS in train, model decodes wrong | targeted-repair pass; strict singleton, no curriculum redesign |
| **Held-set generalization residual** | row is NOT in train, model decodes wrong | curriculum design choice; not a defect per se — track, do NOT redesign partitions to "fix" |
| **Parent-relative cluster** | 3+ same-surface holes appear in a prior rung that the parent had clean | revert / re-recipe; cluster-swap detected |
| **Signal-starvation** | per-prior signal too thin under heavy replay | bump corpus size (`--curriculum-n-train`) before changing other knobs |

### Artifacts policy

`.pt` checkpoint files are **runtime/research outputs, not repo
content**. Commit code/tooling/docs/manifest receipts. Chain-head
provenance lives on the ai-room board (msg ID + path + recipe + A0
result + known residuals). Storage decisions are separate from this
training-rule contract.

### Throughput

Cached/batched probe path is the default; native ternary train is
preferred when available. Both consistently outperform their FP /
non-cached baselines on this stack; see `MEMORY/atlas/training_*.md`
for per-round measurements.

### Strategic arc

`hrm-158-base` as a robust all-rounder via math-first progressive
reasoning blocks, then interleaved structured language / instruction
blocks with math replay, then code/tool-use blocks. **Specialists and
HRM-1.58-MoE branch from robust base checkpoints, not from weak narrow
experts.**

---

# Legacy / adjacent training lanes (pointers only)

> Legacy/adjacent, NOT the active lane — use only if Gabe explicitly reopens a
> lane. Full recipes / commands / metrics / receipts live in the dedicated
> files below and in `MEMORY/atlas/training_part_1.md` + `_part_2.md`, never in
> this rule.

- **VRAM budget + Substrate FP32 card hosting** → `environment.md`, `Substrate.md`
- **PT / DT recipes** (incl. code-skeleton) → `delta_rule.md`
- **Distillation / QLoRA** base + specialists → `distillation.md`
- **Substrate-native training** (SubstrateLM/HRM, scheduled sampling, cross-task transfer) → `architecture.md`, `MEMORY/atlas/training_part_2.md`
- **CRLM HRM** (structure-only recipe, sweet-spot configs, epoch-budget rule) → `MEMORY/atlas/training_part_1.md`
- **Quantization** (tq4/tq3 commands + block format) → `turboquant.md`
- **Export & serving** → `environment.md`, `.codex/AGENTS.md` §"Serving Architecture"
- **Backend-first priority + Auto-CALM training-data cycle** → `calm.md`
- **Substrate eval defaults** → `workflow.md`, `calm/llm_computer/eval_defaults.py`
