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

Fix: `HybridGroupedSmall2DTransformer` with per-layer linear type:
- `Tq4LinearGGMLOriented` for Gemma layers (byte-preserving)
- `FP32LinearGGMLOriented` for compiled/HRM layers (exact)
- Both share `y = x @ W` GGML-oriented contract

File: `calm/llm_computer/hybrid_substrate.py`

## Card Installation

`CardSlot(ch_off, sh_off, ffn_off, tok_off, layer_off)` specifies
where a card goes in the substrate. The installer corner-patches
weights into the reserved rectangle.

Two installers:
- `install_compiled_card` (for `GroupedSmall2DTransformer`, PyTorch orientation)
- `install_compiled_card_hybrid` (for `HybridGroupedSmall2DTransformer`, GGML orientation — transposes automatically)

File: `calm/llm_computer/card_installer.py`, `hybrid_substrate.py`

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

`AutoUpgradeEngine` connects all pieces. The user makes the system
smarter by using it. Proven: 0/8 → 8/8 → 11/11 across 3 sessions.

File: `calm/llm_computer/auto_upgrade.py`

## Level Cascade (all validated)

```
Level 1: Cards compose via shared channels              ✓ Round 21
Level 2: HRM + card coexist in substrate                ✓ Round 9
Level 3: Card inside Gemma-like attention (demo)         ✓ Round 22A
Level 4: Card inside REAL Gemma attention (GGUF bytes)   ✓ Round 23
Level 5: Gemma + HRM + compiled ALL in ONE layer         ✓ Round 29
```

## GPU Scaling

| Hardware | Capacity | Params |
|---|---|---|
| RTX 4070 (8 GB) | 50 HRM slots + card | 889M, 68.6× speedup, 3.56 GB VRAM |
| RTX 4090 (24 GB) | ~100 domains | ~3B |
| A100 (80 GB) | ~500 domains, team substrate | ~10B |

## Key Files (session 30)

| Module | Purpose |
|---|---|
| `card_installer.py` | CardSlot + install_compiled_card |
| `hybrid_substrate.py` | HybridGroupedSmall2DTransformer + FP32/tq4 per-layer |
| `tied_embedding.py` | tie_head_to_tok for Gemma-style weight sharing |
| `q6k_dequant.py` | Q6_K → FP32 for Gemma token_embd |
| `substrate_compute.py` | SubstrateComputer (text → card → answer) |
| `unified_substrate_compute.py` | UnifiedSubstrateComputer (HRM + card + Gemma) |
| `auto_upgrade.py` | AutoUpgradeEngine (CALM → compile → persist) |
| `persistent_knowledge.py` | KnowledgeStore (corrections → weights) |
| `program_builder.py` | StdLib + CompiledOp + build_program linker |

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
