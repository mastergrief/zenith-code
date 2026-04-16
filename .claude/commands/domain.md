# /domain — Add a New Domain to the Unified Single Tensor

**Input**: [$ARGUMENTS] — Domain name and/or description (e.g. "temperature conversion", "chemistry molecular weights", "music theory intervals")

**Purpose**: Guided workflow to add a complete domain to the CRLM stack — facade with compiled ops, NL templates, trained pointer-copy transducer, and installation into the unified substrate. Uses `AskUserQuestion` at every decision point.

**Working policy**: No subagents. Direct Edit/Write/Read/Grep/Bash. Hypothesis-build-test-iterate per `.claude/rules/workflow.md`.

---

## STEP 1 — SCOPE THE DOMAIN

If `$ARGUMENTS` provided, use as starting context. Otherwise ask.

`AskUserQuestion`: **What domain are you building?**
- Gather: domain name, what it computes, example queries a user would ask
- Clarify: is this a compute domain (deterministic functions) or knowledge domain (factual lookups) or both?

Then `AskUserQuestion`: **What operations does this domain need?**
- Present 2-4 candidate operation sets based on the domain description
- Each option should list concrete functions (e.g. "celsius_to_fahrenheit, kelvin_to_celsius, fahrenheit_to_kelvin")
- User picks or customizes

Then `AskUserQuestion`: **Does this domain need a CALM backend too?**
- Option A: "Yes — write `calm/backends/<domain>_ops.py` alongside the compiled card (Recommended)"
- Option B: "No — compiled card only, skip CALM backend"
- Option C: "Knowledge backend — write `calm/backends/<domain>_kb.py` for factual lookups"

---

## STEP 2 — BUILD THE CALM BACKEND (if selected)

Write `calm/backends/<domain>_ops.py` or `<domain>_kb.py`:
- Export `<DOMAIN>_FUNCTIONS` dict mapping function names to pure functions
- Export `<DOMAIN>_NL_PATTERNS` list for precompute matching
- Add `_DATA_VERSION` for knowledge backends

**Verify**: `PYTHONPATH=. python3 -c "from calm.backends.<domain>_ops import *; print(len(<DOMAIN>_FUNCTIONS), 'functions')"` 

Run: `PYTHONPATH=. python3 -m pytest calm/tests/ -x -q --tb=short` — confirm no import breaks.

---

## STEP 3 — BUILD THE COMPILED CARD

`AskUserQuestion`: **How should the compiled card work?**
- Option A: "Gate-graph IR — declarative, auto-scheduled (Recommended)"
- Option B: "Hand-wired Small2DTransformer weights"
- Option C: "Skip compiled card — CALM backend handles computation"

If gate-graph IR selected:
1. Write `calm/llm_computer/programs/<domain>_card.py`
2. Define operations as `CompiledOp` with imports/exports
3. Build via `program_builder.py` facade
4. **Verify exhaustive**: run all valid inputs, report accuracy

If hand-wired:
1. Write the program following existing patterns in `calm/llm_computer/programs/`
2. Exhaustive test with expected accuracy target

Show the user the accuracy result and ask to proceed.

---

## STEP 4 — WRITE NL TEMPLATES

`AskUserQuestion`: **How many NL template phrasings do you want?**
- Option A: "Starter set (~10-15 templates) — fast, good for validation (Recommended)"
- Option B: "Full set (~30-50 templates) — better generalization, takes longer to write"
- Option C: "Let me provide example queries and you generate templates"

Write `calm/hrm/<domain>_data.py`:
- Follow the pattern of `nl_data.py` / `word_data.py` / `gsm_data.py`
- Dataclass: `<Domain>Problem(problem: str, expression: str, answer: str)`
- Generator class with `_sample_operand()` for balanced digit-length coverage
- Templates: `(nl_template, expression_template, operand_ranges)`

`AskUserQuestion`: **Review the templates — any phrasings to add or change?**
- Show 5-10 sample generated problems
- User can approve, request additions, or provide custom templates

---

## STEP 5 — TRAIN THE POINTER-COPY TRANSDUCER

Write `scripts/train_copy_<domain>.py`:
- Reuse `CopyAugmentedTransformer` from `calm/llm_computer/copy_augmented.py`
- Dataset class with loss mask on expression tokens only
- Scheduled sampling (tf_ratio 1.0 → 0.3)
- Autoreg eval as gate metric
- Checkpoint at `calm/hrm/checkpoints/copy_<domain>_best.pt`

`AskUserQuestion`: **Training configuration?**
- Option A: "Default (5000 problems, 500 epochs, d_model=64) — proven on 3 domains (Recommended)"
- Option B: "Larger (10000 problems, 500 epochs, d_model=96) — for complex templates"
- Option C: "Quick test (2000 problems, 200 epochs) — fast validation before full run"

Determine max_len from template lengths + expression lengths + overhead.

Launch training detached:
```bash
setsid env PYTHONPATH=. python3 -u scripts/train_copy_<domain>.py \
  --epochs <N> --problems <N> --device auto --eval-every 10 \
  < /dev/null > /tmp/copy_<domain>_train.log 2>&1 &
disown -a
```

Monitor with: `tail -f /tmp/copy_<domain>_train.log | grep -E --line-buffered "copy-<domain>|Traceback|Error|Killed|OOM"`

Watch for convergence using the plateau rule (3 evals at same accuracy = done).

---

## STEP 6 — EVALUATE

Once training converges, run per-bucket eval:
- Small operands [1-9]
- Mid operands [10-99]  
- Large operands [100+] (if applicable)
- Full range
- Held-out test set (different seed, 100 problems)

Show results table to user.

`AskUserQuestion`: **Results — ship or iterate?**
- Option A: "Ship — accuracy is good enough for the stack (Recommended if >95%)"
- Option B: "Iterate — add more templates and retrain"
- Option C: "Scale up — increase model capacity (d_model, n_layers)"
- Option D: "Abort — domain isn't working, revisit approach"

If iterate: loop back to Step 4 (templates) or Step 5 (training config).

---

## STEP 7 — INSTALL INTO PROD GEMMA SUBSTRATE

Install paths into `GemmaSubstrate` (`calm/llm_computer/gemma_substrate.py`,
gemma-4-E4B-it-tq4). Two patterns coexist; pick one per card:

- **`install_card_in_attention(...)`** — card weights live INSIDE Gemma's
  `attn_q/k/v/output` tensors at a (sub-head, channel) rectangle.
  Requires `convert_layer_to_fp32(layer_idx)` first. Use for compiled
  cards (`mode='hard_max'`) and lookup cards (`mode='softmax'`). One
  hosting FP32 layer can host many cards. Card weights ship in the .pt.
- **`CardSlot(...).attach(m, preserve=True)`** — card runs as a separate
  Module forward; output added to residual at the install layer; subsequent
  layers' attn/ffn/proj contributions to the reserved channels are masked.
  Use for PTs (copy-augmented attention not reducible to a sub-head mode)
  and any card with a custom forward.

### 7.1 — Allocate from the registry

Read `.claude/MEMORY/substrate_registry.md` (create if missing). Each
domain reserves:

- `host_layer` (int) — the Gemma layer hosting this domain's cards
- `ch_in` (lo, hi) — residual input channels for the card
- `ch_out` (lo, hi) — residual output channels (often == ch_in)
- `sub_head_offset` (int) — start sub-head within head 0 of host_layer
- `vocab_mapping` (dict) — card_token_id → Gemma BPE token (for verify hook)

Channel rules:
- Reserve from the high end of d_model=2560 (start at 2560 - sum_of_d_card)
- Two domains MUST NOT overlap channel ranges or sub-head ranges in the
  same host_layer
- d_card per card must be a multiple of 2 (d_head=2 invariant)

VRAM budget (8 GB RTX 4070):
- Substrate baseline: ~5.0 GB (Gemma tq4 + Q6_K + Triton kernels)
- Per FP32 SWA layer conversion: ~330 MB
- Per FP32 global layer (5, 11, 17, 23, 29, 35, 41): ~600 MB
- Practical max: 5-7 FP32 layers → 5-7 hosting slots for in-attention

`AskUserQuestion`: **Pick the install pattern**
- Option A: "in-attention (hard_max for compiled, softmax for HRM-style)
  — card ships in .pt (Recommended for compiled cards)"
- Option B: "CardSlot — card stays as separate Module (Required for PTs)"
- Option C: "Both — compiled card in-attention + PT via CardSlot at the
  same host_layer (full CRLM pipeline)"

### 7.2 — Write the install script

Create `scripts/install_<domain>_in_gemma.py`. Skeleton for a compiled
card (in-attention, hard_max):

```python
from calm.llm_computer.gemma_substrate import (
    GemmaSubstrate, enable_triton_tq4, VerificationHook)
from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer
from calm.llm_computer.programs.<domain>_card import build_<domain>

enable_triton_tq4(True)
m = GemmaSubstrate.from_gguf(
    "/home/gabe/models/gemma-4-E4B-it-tq4-aligned.gguf", max_len=512)
m.preload_gpu("cuda")
m.warmup(seq_lens=(1, 6))

card = build_<domain>().cuda().eval()
HOST_LAYER, CH_OFF, D_CARD, SH_OFF = 30, 2552, 8, 0    # from registry
m.convert_layer_to_fp32(HOST_LAYER)
info = m.install_card_in_attention(
    card=card, layer_idx=HOST_LAYER, sub_head_offset=SH_OFF,
    ch_off=CH_OFF, d_card=D_CARD, mode="hard_max")

# Optional: verification hook to bias Gemma logits on card hit
tok = GemmaTokenizer.from_gguf(
    "/home/gabe/models/gemma-4-E4B-it-tq4-aligned.gguf")
vocab_map = {...}                  # card_token → Gemma BPE token
# (For PT: use CardSlot pattern — see scripts/gemma_learning_loop_demo.py)
```

For PT install (CardSlot pattern), copy from `gemma_learning_loop_demo.py`
or `gemma_substrate.py` docstrings — `card_input_fn` discretizes residual
to char tokens, `output_fn` writes logits back to reserved channels.

### 7.3 — Verify install end-to-end

Hook the card's forward, run a known prompt, confirm:

1. Card receives the expected input (residual slice or fixed input)
2. Card produces the expected output (compare to standalone forward)
3. Gemma's logits diverge from no-install baseline (proves install fires)
4. For verify hook: argmax flips to the verified token on math prompts

Save the registry entry (`.claude/MEMORY/substrate_registry.md`) with:
- domain name, host_layer, ch ranges, sub_head_offset, install mode,
  vocab_mapping, install date, max abs diff vs baseline

### 7.4 — Persist

Two persistence paths:

- **In-attention install**: `torch.save(m, "substrate.pt")` ships card
  weights baked into Gemma's `attn_q/k/v/output`. Next session: load
  the .pt, no re-install needed.
- **CardSlot install**: card weights stay separate. Save the card's
  state_dict alongside; on load, rebuild CardSlot + .attach(m, preserve=True).
- **Learning loop** (`KnowledgeStore`): save corrections JSON; on load,
  `load_corrections()` + `build_recall_model()` + `CardSlot.attach()`
  reinstates the recall card. See `scripts/gemma_learning_loop_demo.py`.

`AskUserQuestion`: **Commit this domain?**
- Option A: "Yes — commit all files + registry (Recommended)"
- Option B: "Yes — commit but don't update registry yet"
- Option C: "No — review first"

If committing, stage:
- `calm/backends/<domain>_ops.py` (if created)
- `calm/hrm/<domain>_data.py`
- `scripts/train_copy_<domain>.py`
- `scripts/install_<domain>_in_gemma.py`
- `calm/hrm/checkpoints/copy_<domain>_best.pt`
- `calm/llm_computer/programs/<domain>_card.py` (if created)
- `.claude/MEMORY/substrate_registry.md`

Commit with: card params, install mode, channel/sub-head allocation,
verify diff vs baseline.

---

## CONVENTIONS

- All NL data generators must use `_sample_operand()` for balanced digit-length coverage
- All training scripts use `CopyAugmentedTransformer` (pointer-copy) not base `Small2DTransformer`
- Gate metric is autoreg accuracy, NOT teacher-forced val_acc
- Checkpoint saved on best autoreg, not best val_acc or lowest loss
- `--epochs 500` minimum (cosine LR under-fits at 100)
- Scheduled sampling ON by default (tf_ratio 1.0 → 0.3)
- One round per commit with measurement table
- **Substrate install**: every domain MUST have an entry in
  `.claude/MEMORY/substrate_registry.md` before commit. Channel and
  sub-head ranges are first-come-first-serve; check for collisions
  before allocating.
