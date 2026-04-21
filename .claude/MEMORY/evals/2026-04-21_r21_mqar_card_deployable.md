# R21 — Deployable PT+Delta MQAR card (2026-04-21)

Produces `copy_augmented_delta_mqar_best.pt` — the trained PT+Delta
card that solves MQAR across N=5-15 at 100%. This is the artifact
that will install on Gemma via CardSlot in R22.

## Training

  script:       scripts/train_pt_delta_mqar.py
  task:         mqar (3 N values pooled)
  data:         5000 problems per N × 3 N values = 15K train
                100 problems per N × 3 = 300 val
  N values:     [5, 10, 15]
  config:       d_model=64, n_heads=32, n_layers=4, d_ffn=128,
                max_len=128, n_copy_heads=4
                chunkwise=True, n_delta_heads=1, n_iterations=1
  training:     AdamW lr=1e-3, cosine schedule, scheduled sampling
                tf=1.0→0.3 over 20 epochs, batch=64
  params:       183,877

## Results

Training trajectory (evals every 2 epochs):

  ep 2: loss 0.85, overall 29%  (N5=36% N10=26% N15=24%)
  ep 4: loss 0.05, overall 99%  (N5=100% N10=100% N15=97%)
  ep 6: loss 0.01, overall 100% (N5=100% N10=100% N15=100%)

Best saved at ep6. Training wall time: ~150s total (2 min).

## Held-out eval (fresh seed, never used in training)

  N          100 problems           cached decode time
  ---       -----------              ------------------
  N=5       100/100 (100%)           2.0s
  N=10      100/100 (100%)           1.7s
  N=15      100/100 (100%)           1.9s
  N=20      0/100 (OOD — not trained)  4.7s

**N=20 drops to 0% because the card wasn't trained at N=20.**
Expected. To cover N=20 would need a ~10K/N training pass per
R14-b's finding that "+5 on N needs 2× data". Not blocking for R22
— the card covers N=5-15 which is the commercial sweet spot
(function-body binding tracking, dict-key lookup, NER fact
retrieval in long NL contexts).

## Card properties (R22 install spec)

  File:            calm/hrm/checkpoints/copy_augmented_delta_mqar_best.pt
  Size:            748 KB
  Params:          183,877
  Architecture:    CopyAugmentedDeltaNet (PT+Delta)
  Vocab:           82 chars (calm/hrm/data.py _CHAR_TO_ID)
  Input format:    "k1 v1 k2 v2 ... kN vN <sep> query_key" → "value"
  Capability:      MQAR N=5-15 at 100% held-out
  Inference:       cached decode 1.18× plain PT overhead (R20b)

## R22 install plan (deferred to next session)

1. **Load card**: `torch.load('.../copy_augmented_delta_mqar_best.pt')`,
   wrap in CopyAugmentedDeltaNet + restore state dict.
2. **Input adapter**: write `gemma_residual_to_mqar_input(h, prefix_ids)`
   that extracts (k, v) pairs from Gemma's NL context and tokenizes to
   the card's 82-char vocab. Most naive version: regex over `key =
   value` assignments in Gemma's prefix text; hash keys/values to
   single chars.
3. **Install via CardSlot**: `CardSlot(layer_idx=30, card=card,
   d_card=..., card_input_fn=adapter, output_fn=writer).attach(gemma,
   preserve=True)`. Writer maps card output argmax → Gemma BPE
   digit/letter bias via VerificationHook.
4. **Measurement**: run NIAH-style multi-needle prompts on Gemma +
   card vs Gemma alone. Target prompts: "Alice is 30. Bob is 25. ...
   What is Alice's age?" at N=5-15 facts. Gemma baseline multi-needle
   degrades at N≥5 (from our 2026-04-07 NIAH eval); card should
   preserve 100%.
5. **Regression guard**: run prompts with NO (k, v) structure (e.g.
   "Write a haiku about cats") — card's hash-match gate should miss,
   output unchanged from pure-Gemma baseline. Preservation test.

## Why this scopes naturally to R22

The trained card is ready; the install is the remaining engineering.
The design (CardSlot pattern from session 32 PT install) is proven;
the adapter (NL → 82-char-vocab MQAR input) is the novel piece and
the right place for design effort. Would be a half-day next session.

## Related

- R13-R14-b: MQAR data-scaling curve
- R17: chunkwise DeltaNet (training speedup)
- R20: PT+Delta as default card architecture
- R20b: cached decode for inference parity
- `calm/llm_computer/gemma_substrate.py:CardSlot` — install infrastructure
  already exists from session 32

## Raw log

- `/tmp/r21_train_mqar_card.log`
