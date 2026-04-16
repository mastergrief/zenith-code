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

## STEP 7 — INSTALL INTO SUBSTRATE (future)

**Note**: Full substrate installation requires allocating sub-head slots and channel ranges. This step documents what needs to happen but doesn't execute the install unless the unified tensor is loaded.

Document the installation plan:
- Channel range for this domain's I/O
- Sub-head range for PT + compiled card
- FFN neuron range
- Layer offset

`AskUserQuestion`: **Commit this domain?**
- Option A: "Yes — commit all files (Recommended)"
- Option B: "Yes — commit but don't update CLAUDE.md yet"
- Option C: "No — I want to review first"

If committing, stage:
- `calm/backends/<domain>_ops.py` (if created)
- `calm/hrm/<domain>_data.py`
- `scripts/train_copy_<domain>.py`
- `calm/hrm/checkpoints/copy_<domain>_best.pt`
- `calm/llm_computer/programs/<domain>_card.py` (if created)

Commit with before/after accuracy table in message body.

---

## CONVENTIONS

- All NL data generators must use `_sample_operand()` for balanced digit-length coverage
- All training scripts use `CopyAugmentedTransformer` (pointer-copy) not base `Small2DTransformer`
- Gate metric is autoreg accuracy, NOT teacher-forced val_acc
- Checkpoint saved on best autoreg, not best val_acc or lowest loss
- `--epochs 500` minimum (cosine LR under-fits at 100)
- Scheduled sampling ON by default (tf_ratio 1.0 → 0.3)
- One round per commit with measurement table
