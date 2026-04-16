# Substrate Install Registry

Source of truth for what's installed in the prod Gemma substrate
(`gemma-4-E4B-it-tq4-aligned.gguf` via `GemmaSubstrate`). Every domain
added through `/domain` MUST land here before commit. First-come-
first-serve allocation; check for collisions before reserving.

## Architecture invariants

- Gemma 4 E4B: 42 layers (35 SWA + 7 global), d_model=2560
- SWA layers: 8 heads × 128 sub-heads of d_head=2 = 1024 sub-heads/layer
- Global layers (5, 11, 17, 23, 29, 35, 41): 8 heads × 256 sub-heads = 2048
- Layers 0-23 own their KV; layers 24-41 reuse (SWA → 22, global → 23)
- Free residual range: bounded by VRAM, not architecture (every reserved
  channel costs Gemma some signal — preservation masking zeros Gemma's
  contribution to those channels in subsequent layers)

## VRAM budget (RTX 4070, 8 GB)

- Substrate baseline (Gemma tq4 + Q6_K + Triton + Pi cache): ~5.0 GB
- Per FP32 SWA layer conversion: ~330 MB
- Per FP32 global layer conversion: ~600 MB
- Activations + KV at 1K context: ~0.5 GB
- Practical max: 5-7 FP32 hosting layers

## Channel allocation (start at high end of d_model=2560)

| ch lo | ch hi | width | host_layer | sub_head | mode | domain | install date | max diff vs base |
|-------|-------|-------|------------|----------|------|--------|--------------|------------------|
| _none_ yet                                                                                          |

## Reserved-by-convention demo channels (transient, not committed)

These were used in interactive testing this session, will be reclaimed:

| ch lo | ch hi | what |
|-------|-------|------|
| 2400  | 2480  | PT (`copy_augmented_hrm`) — `gemma_substrate.py` chained demo |
| 2480  | 2488  | `adder_tiny` — chained CRLM demo |
| 2480  | 2558  | `KnowledgeStore` recall card — `gemma_learning_loop_demo.py` |
| 2544  | 2552  | `threshold` — CardSlot install demo (commit 7072713) |
| 2552  | 2560  | `add_one` — CardSlot install demo (commit 7072713) |

## Verification hook vocab mappings (Gemma BPE token IDs)

Single-digit Gemma BPE tokens (262K vocab):

| digit | Gemma token id | id_to_token |
|-------|----------------|-------------|
| 0     | 236771         | `'0'`       |
| 1     | 236770         | `'1'`       |
| 2     | 236778         | `'2'`       |
| 3     | 236800         | `'3'`       |
| 4     | 236812         | `'4'`       |
| 5     | 236810         | `'5'`       |
| 6     | 236825         | `'6'`       |
| 7     | 236832         | `'7'`       |
| 8     | 236828         | `'8'`       |
| 9     | 236819         | `'9'`       |

## Install patterns

- **In-attention** (`install_card_in_attention`): card weights ship in
  the .pt; requires `convert_layer_to_fp32(host_layer)` first; supports
  `mode='hard_max'` (compiled), `mode='softmax'` (HRM-style), or
  `mode='grouped'` (defaults to Gemma's pipeline).
- **CardSlot** (`CardSlot(...).attach`): card runs as a separate Module
  forward; preservation masking on by default; required for PTs and
  cards with custom forwards.
- **VerificationHook** (`VerificationHook(slot, vocab_mapping, boost)`):
  optional, biases Gemma's logits at the head with the card's argmax
  through a `card_token → gemma_token` mapping.
- **Learning loop** (`KnowledgeStore`): JSON corrections persist
  alongside the .pt; on load, `build_recall_model()` + CardSlot install
  reinstates the recall card.
