# Substrate — Unified Single Tensor Architecture

## Core Thesis

**The model IS the substrate. The substrate IS the model.** Gemma's
own attention layers host compiled programs, trained HRM specialists,
and persistent knowledge — all in ONE weight tensor, ONE forward pass,
ZERO cross-talk. Adding a compiled program to Gemma is a weight edit,
not retraining.

## d_head=2 Decomposition (the architectural foundation)

`d_head=256` attention = 128 × `d_head=2` sub-heads with scores SUMMED
before softmax (proven exact to float32, commit `243b4ab`). Every
transformer attention head IS a collection of tiny analytically-tractable
units. You can surgically modify individual sub-heads while leaving
adjacent ones untouched.

At Gemma 4 E4B scale:
- 8 heads × 128 sub-heads = **1024 d_head=2 sub-heads** per SWA layer
- Gemma uses 1024 sub-heads → **1024 FREE** per SWA layer
- 35 SWA layers → **35,840 free sub-head slots** for programs
- 1536 free channels [2560, 4096) for card I/O

## Per-Sub-Head Attention Partition (Level 5)

Three attention modes coexist in ONE layer (proven Round 29):

```
Sub-heads 0..1023:    Gemma      (grouped softmax)
Sub-heads 1024..1055: HRM        (single softmax)
Sub-heads 1056..1060: Compiled   (single hard_max)
Sub-heads 1061..2047: FREE
```

Forward splits Q/K/V by sub-head range, applies each mode's attention,
concats output. FFN is shared; each domain's neurons fire only on its
channels. Cross-talk: **0.00e+00** (Rounds 22A, 29).

**Now real on prod Gemma 4 E4B** (`GemmaSubstrate`, session 32). Set
via `GemmaSubstrate.attention_partition[layer_idx]` — list of
`(sh_lo, sh_hi, mode)` entries. `_forward_layer` dispatches: zeros
card sub-heads' Q for the standard sum-then-softmax, computes Gemma's
attention without their contribution, then computes per-sub-head
attention for each card range with its own mode (`'softmax'` or
`'hard_max'`), reshape to `(B, H, S, n_sub, 2)`, einsum scores per
sub-head, mask, softmax/hard_max, einsum back with V slices. Layers
without partitions stay on the fast path (no perf cost).
`install_card_in_attention(..., mode='hard_max')` registers the
partition automatically.

## Channel Allocation Protocol

```
Channels 0..2559:     Gemma residual (pretrained, frozen)
Channels 2560..2623:  HRM I/O (per specialist: 32 channels)
Channels 2624..3205:  Compiled card I/O (dispatched_v4: 582 ch)
Channels 3206..3225:  Knowledge DB (recall channels)
Channels 3226..4095:  FREE (870 channels for future domains)
```

Each domain owns a disjoint channel rectangle. Writes are additive via
W_out (each domain's W_out rows are nonzero only in its channel range).
No domain can corrupt another's residual.

## Hybrid Per-Layer Linear Type

**Round 11 finding**: tq4 quantization destroys compiled card weights
(791/791 → 70/791, 91% loss). Lloyd-Max codebook is tuned for Gaussian
LM weights; compiled cards have discrete ±1/±16 coefs.

Two implementations:

- **`HybridGroupedSmall2DTransformer`** (substrate-native demo, GGML orientation)
  - `Tq4LinearGGMLOriented` for Gemma layers (byte-preserving)
  - `FP32LinearGGMLOriented` for compiled/HRM layers (exact)
- **`GemmaSubstrate.convert_layer_to_fp32(layer_idx)`** (prod Gemma, session 32)
  - Replaces `MmapTq4Linear` with `FP32GemmaLinear` for the chosen
    layer. One-time dequant via the tq4 path; result lossless to
    numerical noise (max abs diff ~2e-5 vs original tq4 forward).
  - Required before `install_card_in_attention` (surgical edits to
    tq4 weights would re-quantize and lose compiled coefs).
  - Cost: ~330 MB SWA, ~600 MB global per layer; budget for 5-7
    hosting layers on 8 GB.

Files: `calm/llm_computer/hybrid_substrate.py`,
`calm/llm_computer/gemma_substrate.py`.

## Card Installation

Two install modes on prod `GemmaSubstrate` (session 32). They are
architecturally distinct, with different perf / VRAM / capability
profiles. Pick per card.

### Mode tradeoffs

| Aspect | In-attention | CardSlot (residual-additive) |
|---|---|---|
| Card compute | Inside Gemma's `attn_q/k/v/output` matmul | Separate Module, appended after layer |
| Upgrades attention directly | Yes (same kernel) | No (runs alongside) |
| Sub-head budget | Consumes reserved sub-heads | None |
| FP32 cost | ~330 MB SWA / ~600 MB global per host | Zero |
| Card types | Pure-attention only today | Any `nn.Module` incl. PT, FFN cards |
| Custom attention (PT copy gate) | Impossible | Required |
| Perf | Zero overhead | Small extra matmul per slot |

**Known limit**: `install_card_in_attention` writes `attn_q/k/v/output`
only. Cards with ReGLU/FFN (`adder_tiny`, `gcd`, `reasoning_engine`)
need an FFN migration (not yet shipped) before they can be fully
in-attention. Pure-attention cards (`add_one`, `threshold`, `copy_past`,
`retrieve_by_index`) work today.

**Facade packaging**: `calm/llm_computer/facades/math_addition.py`
(`MathAdditionFacade`, Round 8) is the first reusable domain class.
New domains subclass the pattern and get install/detach/set_prompt
+ save/load + detach reversibility. Prefer facades over one-off
install scripts.

**1. In-attention** — card weights live INSIDE `attn_q/k/v/output`:

```python
m.convert_layer_to_fp32(host_layer)           # one-time per host
m.install_card_in_attention(
    card=card, layer_idx=host_layer,
    sub_head_offset=0, ch_off=2552, d_card=8,
    mode='hard_max',                          # or 'softmax', 'grouped'
)
```

Surgical writes:
```
attn_q[ch_off:ch_off+d_card, sh_lo:sh_hi]    = card.W_qkv[Q].T
attn_k[ch_off:ch_off+d_card, sh_lo:sh_hi]    = card.W_qkv[K].T
attn_v[ch_off:ch_off+d_card, sh_lo:sh_hi]    = card.W_qkv[V].T
attn_output[sh_lo:sh_hi, ch_off:ch_off+d_card] = card.W_out.T
```
Other rows zeroed so card sub-head's input comes ONLY from reserved
channels. `mode='hard_max'`/`'softmax'` registers a per-sub-head
partition entry so the card's attention runs in its own mode (not
Gemma's grouped softmax). Card weights ship in the .pt.

**2. Residual-additive (CardSlot)** — card runs as a separate Module:

```python
slot = CardSlot(layer_idx=30, ch_off=2480, card=pt, d_card=80,
                card_input_fn=adapter, use_full_residual=True,
                output_fn=writer)
slot.attach(m, preserve=True)                 # masks subsequent layers
```

`preserve=True` registers the channel range so subsequent layers'
attn / ffn / per-layer-embed contributions to those channels are
zeroed at runtime — card output flows through to output_norm intact.
Used for PTs (copy-augmented attention can't reduce to a sub-head
mode) and prototyping.

**3. Hub-first forced-attention (HubInjectionCard)** — R44 facade
form of R43's causal-validation intervention. For shared hub heads
(L23 H1/H4, serving arithmetic + SV agreement + comparison +
counting + multi-step composition), install the intervention as a
runtime-dispatched facade: detect the natural top-position via
live Q/K, force one-hot attention, no per-task hand-dispatch.

```python
from calm.llm_computer.facades.hub_l23 import HubInjectionCard
card = HubInjectionCard(layer_idx=23, heads=[1, 4])
card.install(m)                              # hooks L23 attention
# now arithmetic + SV + comparison + counting + multi-step
# all benefit from the same install — 5-for-1 ROI
```

Bit-identical to R43's inline intervention (R44 measurement).
`generate()` path verified 5×12 decode tokens (R45) — compatible
with autoregressive generation, not just single-token prediction.
Use for hub heads with validated cross-task causal effect. Facade
file: `calm/llm_computer/facades/hub_l23.py`.

**4. VerificationHook** — close the loop card → Gemma logits:

```python
m.verification_hooks.append(
    VerificationHook(slot, vocab_mapping=DIGIT_TO_GEMMA, boost=50.0))
```

Reads `slot.last_output`, picks argmax, biases the corresponding
Gemma BPE token logit by `boost`. Runs after head + softcapping. On
the math benchmark this overrode Gemma's "Two plus three equals
**six**" with the verified `'5'`.

Files: `calm/llm_computer/gemma_substrate.py` (prod Gemma),
`calm/llm_computer/card_installer.py`,
`calm/llm_computer/facades/hub_l23.py` (HubInjectionCard),
`calm/llm_computer/hybrid_substrate.py` (demo substrate).

## Facade / Import System (Program Builder)

A module system for compiled neural programs:

```python
stdlib = StdLib(exports={"a": 3, "b": 4, "bias": 1})
adder = CompiledOp(imports={"x": "a", "y": "b"}, exports="sum")
model = build_program(stdlib, [adder, ...], head)
```

- **StdLib**: layer-0 facade with tok/pos/copy primitives
- **CompiledOp**: declares imports (channel names), gate/val formulas, exports
- **Linker** (`build_program`): resolves imports to channel numbers, auto-schedules
  layers (topological sort), compiles to one Small2DTransformer
- Bad imports → `KeyError` at build time, not runtime

File: `calm/llm_computer/program_builder.py`

## Inter-Slot Composition

Two separately-compiled cards compose via a shared residual channel:

```
Card A (layer 0): writes a+b → channel 9
Card B (layer 1): reads channel 9 → outputs indicator
merge_cards(A, B) → one model, one forward, composition works
```

Cards don't know each other during compilation. They agree on channel
numbers (the "interface"). The substrate is the wiring plane. Proven
64/64 exhaustive (Round 21).

File: `calm/llm_computer/programs/composed_sum_threshold.py`

## Depth-Compounding

One sub-head across multiple layers = compound computation pipeline:

```
Layer 0: a+b → CH_SUM           (adder)
Layer 1: SUM×2 → CH_DOUBLE      (doubler, reads layer 0's output)
Layer 2: DOUBLE≥8 → indicator   (classifier, reads layer 1's output)
```

Channels are registers. Layers are instructions. Sub-heads are threads.
Gemma's 42 layers × 1024 free sub-heads = **43,008 compute slots**.

File: `calm/llm_computer/programs/depth_compound.py`

## Cross-Card Gating

`dispatched_v4`: opcode thresholds shifted by +1 so valid ops are
[1, N_OPS]. Token 0 = "not my input" → ALL card slots output exactly
zero. Cross-card gating with zero extra layers.

File: `calm/llm_computer/programs/dispatched_v4.py`

## Persistent Knowledge

Corrections compiled into weights as step-function indicators:
`indicator(x == k) = ReLU(x-k+1) - 2·ReLU(x-k) + ReLU(x-k-1)`.
3 ReGLU neurons per fact. Cross-session via save/reload of `.pt`.

File: `calm/llm_computer/persistent_knowledge.py`

## Auto-Upgrade Loop

```
User queries → CALM verifies → corrections logged → compile into weights → save .pt
Next session → load → errors permanently fixed → zero retraining
```

`AutoUpgradeEngine` connects all pieces (substrate-native demo).
Proven: 0/8 → 8/8 → 11/11 across 3 sessions.

**On prod Gemma** (`scripts/gemma_learning_loop_demo.py`, session 32):
the same loop end-to-end on Gemma 4 E4B. 5 wrong addition prompts →
log corrections to `KnowledgeStore` → `build_recall_model()` produces
a 4,304-param `Small2DTransformer` recall card (3 ReGLU per fact,
step-function dispatch) → `CardSlot.attach` + `VerificationHook` →
5/5 correct. JSON persistence round-trips bit-identical recall.

Files: `calm/llm_computer/auto_upgrade.py`,
`calm/llm_computer/persistent_knowledge.py`,
`scripts/gemma_learning_loop_demo.py`.

## Level Cascade (all validated)

```
Level 1: Cards compose via shared channels              ✓ Round 21
Level 2: HRM + card coexist in substrate                ✓ Round 9
Level 3: Card inside Gemma-like attention (demo)        ✓ Round 22A
Level 4: Card inside REAL Gemma attention (GGUF bytes)  ✓ Round 23
Level 5: Gemma + HRM + compiled ALL in ONE layer        ✓ Round 29 (demo)
                                                         ✓ session 32 (prod Gemma)
Level 6: Learning loop closed end-to-end on prod Gemma  ✓ session 32
         (detect → log → compile → install → persist)
```

## GPU Scaling

| Hardware | Capacity | Params |
|---|---|---|
| RTX 4070 (8 GB) | 50 HRM slots + card | 889M, 68.6× speedup, 3.56 GB VRAM |
| RTX 4090 (24 GB) | ~100 domains | ~3B |
| A100 (80 GB) | ~500 domains, team substrate | ~10B |

## Key Files

Substrate-native (sessions 26-30):

| Module | Purpose |
|---|---|
| `card_installer.py` | CardSlot + install_compiled_card (substrate-native) |
| `hybrid_substrate.py` | HybridGroupedSmall2DTransformer + FP32/tq4 per-layer |
| `tied_embedding.py` | tie_head_to_tok for Gemma-style weight sharing |
| `q6k_dequant.py` | Q6_K → FP32 for Gemma token_embd |
| `substrate_compute.py` | SubstrateComputer (text → card → answer) |
| `unified_substrate_compute.py` | UnifiedSubstrateComputer (HRM + card + Gemma) |
| `auto_upgrade.py` | AutoUpgradeEngine (CALM → compile → persist) |
| `persistent_knowledge.py` | KnowledgeStore (corrections → weights) |
| `program_builder.py` | StdLib + CompiledOp + build_program linker |

Prod Gemma (session 32):

| Module | Purpose |
|---|---|
| `gemma_substrate.py` | `GemmaSubstrate` (full Gemma 4 E4B from GGUF), `MmapTq4Linear`, `FP32GemmaLinear`, `GpuQ6KEmbedding`, `KVCache` / `KVCacheStatic` / `KVCacheTq4` (R53.28 — multi-token prefill S≥1, per-layer position tracking, `trim_swa_storage` via direct byte-copy), `CardSlot`, `VerificationHook`, `convert_layer_to_fp32`, `install_card_in_attention`, `attention_partition`, `generate(use_tq4_kv=True)` dispatch, `generate_with_graph`, `warmup` |
| `tq4_triton.py` | Fused dequant+matvec/matmul Triton kernels for tq4 (5-17×) and Q6_K (125×); dual gate+up kernel; per-shape BLOCK_M heuristic; v2 matvec shared-mem LUT default (R53.29, -7% aggregate) |
| `tq4_flash_attn.py` | R53.34 fused flash-attention decode with tq4 K/V. `fused_tq4_flash_attn_decode(q_rot, k_qs, k_d, v_qs, v_d, centroids, pi, attn_mask)`. Head-major storage contract `(n_heads_kv, N*bpr, 128)`. K-side reuses `tq4_matvec_triton`; V-side `_tq4_weighted_v_kernel` grid=(n_heads_q,) accumulates fp32 (D_HEAD,) per head, Pi.T applied outside. Parity validated: mean cosine ≥ 0.99 vs fp16 KVCache, argmax preservation ≥ 14/16. Default-on (`_use_fused_flash_attn=True`) with runtime N-gate `128 < cached_kv_len < 2048` per 2026-04-20 re-bench (+6 to +14% in band). SWA layers fused; global layers (d_head=512) fall back to memoized dequant. Full spec: `.claude/rules/turboquant.md` §"Fused flash-attention decode". |
| `scripts/gemma_learning_loop_demo.py` | End-to-end detect → log → compile → install → persist (5/5 wrong → 5/5 correct) |
| `.claude/MEMORY/substrate_registry.md` | Source of truth for installed-domain channel/sub-head allocation; BPE digit token mappings; install pattern reference |

| Program | What it proves |
|---|---|
| `compiled_router.py` | ADD/MUL opcode dispatch (Round 1) |
| `dispatched_v2.py` | 5-op internal gating (Round 4) |
| `dispatched_v3.py` | 9 ops scaled (Round 8) |
| `dispatched_v4.py` | Cross-card gating via opcode shift (Round 10) |
| `composed_sum_threshold.py` | Inter-slot composition (Round 21) |
| `depth_compound.py` | 3-stage depth pipeline (Round 24) |
| `reasoning_engine.py` | Comparison + logic + transitivity (Round 30) |
| `compiled_in_gemma.py` | Card inside Gemma layer (Round 22A) |
| `three_in_one_layer.py` | Level 5: 3 modes one layer (Round 29) |
