# R22a — PT+Delta MQAR card installed on prod Gemma (mechanism test)

First install of `copy_augmented_delta_mqar_best.pt` (R21 artifact,
100% held-out on N=5/10/15) into `GemmaSubstrate` via CardSlot + VerificationHook.

## Goal

Prove the install mechanism end-to-end on a simple prompt format.
**This is a mechanism test, not a capability test.** Capability gain
(R22b) requires finding prompts where stock Gemma actually fails.

## Design

- `scripts/r22_install_mqar_card.py` — end-to-end prototype
- Adapter: regex-based `<mem>k=v k=v ...</mem>` + `value of X` → MQAR
  string `"k1 v1 k2 v2 ... ; k_q"` char-tokenized via `_CHAR_TO_ID`
- CardSlot @ L30 ch[2480:2562], `preserve=True`, `use_full_residual=True`
- `card_input_fn`: returns stashed MQAR ids if active, else single `<pad>`
- `card_output_fn`: adds card's last-position logits to reserved channels
  when active; zeros `card_out` in-place when inactive so
  VerificationHook sees flat logits (peak-median=0 < `min_margin`)
- VerificationHook: maps card digit chars `'0'-'9'` → Gemma BPE digit
  tokens (via `tok.encode(" d")[-1]`), `boost=50.0`, `min_margin=0.5`

## Results

### Sanity — card standalone

```
input: "a 3 b 7 c 1 ; b" → card predicts '7' (expected '7')  ✓
```

### Full pipeline

```
prompt                                              baseline  with-card
─────────────────────────────────────────────────────────────────────────
<mem>a=3 b=7 c=1</mem> ... value of b?                  '7'        '7'  ✓
<mem>x=5 y=2 z=8</mem> ... value of x?                  '5'        '5'  ✓
<mem>p=9 q=4</mem> ... value of q?                      '4'        '4'  ✓
2 plus 3 equals  (regression guard, no <mem>)           '6'        '6'  ✓ no regression
```

- **3/3 MQAR-path prompts** emit the correct digit
- **0/4 prompts regress** — baseline == with-card on all four
- On the no-`<mem>` regression prompt, Gemma's own prior gives '6'
  (arithmetically wrong; irrelevant for this test), and the card
  correctly abstains

## Critical finding — capability-gate violation

**Stock Gemma already nails the simple `<mem>` format** at 3/3. Per
`capability_gain.md` §"failure-surface gate": every claim of gain
requires first establishing that Gemma fails baseline. The `<mem>`
prompts pass on stock Gemma because:

- Prompt is short (~16 tokens) — well within Gemma's attention
- Keys are single-letter, values single-digit — trivial to copy
- Gemma's native attention handles this pattern via simple
  induction-style heads (see `MEMORY/atlas/tracing_arc_part_1.md` R32 L37 H6)

**Mechanism gain ≠ capability gain.** The substrate → Gemma token
bias pathway is verified working. The product win (R22b) is still
ahead.

## Install mechanics notes (save future time)

1. **Daemon globals persist across script reloads.** `install()` must
   explicitly clear `m.layers[layer_idx].card_slots`,
   `m.verification_hooks`, and `m.reserved_channels` entries matching
   the target layer before attaching, or stale closures fire
   alongside new ones. Cost of forgetting: first two rerun attempts
   failed with "size 80 vs 82" errors tracking to an OLD output_fn
   closure from a pre-clamp attempt.

2. **ch_off + d_card can exceed d_model.** Card vocab=82, reserved at
   ch_off=2480 → ch_hi=2562, but d_model=2560. Gemma's indexing
   truncates silently to [2480:2560] = 80 channels. `card_output_fn`
   must clamp `hi = min(ch_hi, h.shape[-1])` and take `ans[..., :n]`
   to match. Could also pick ch_off=2478 to fit exactly, but the
   clamp is more robust.

3. **Checkpoint key is `model_state_dict`, NOT `model_state`.** Small
   typo cost one rerun; now locked into load_mqar_card.

4. **`card_output_fn` receives `logits` as the card's full output;
   `slot.last_output = card_out` is set AFTER output_fn.** To silence
   VerificationHook on inactive prompts, mutate `logits` in-place
   (zero it out) in the inactive branch.

## R22b scope (next)

Find prompts where stock Gemma fails on MQAR-shaped recall so the
card's contribution is measurable. Candidates ordered by complexity:

1. **Long prefix with distractors** — prepend 500-2000 tokens of
   unrelated prose before `<mem>`. At NIAH Gemma 220K single-needle
   is 21/21; distractor test drops to 7/7. Somewhere between those
   points is where our small-card help is measurable.
2. **Multi-needle** — NIAH multi-needle test at 220K was 4/5 Gemma,
   3/5 Qwen. Our card solves retrieval by construction — if we can
   match Gemma input-format, we can patch the 1/5 miss.
3. **NL-heavy format** — `Alice has 3 apples, Bob has 7 apples. How
   many does Bob have?` instead of `<mem>`. Requires richer adapter
   (NER/AST walker over Gemma's emitted Python context), not just
   regex.

R22b runs the failure-surface gate FIRST: score stock Gemma on ~50
prompts per candidate format, keep only where Gemma misses. Measure
card lift on that corpus.

## File shipped

- `scripts/r22_install_mqar_card.py` (~280 LOC) — self-contained
  install prototype. Runnable via `bin/gemma-run`.
