# Substrate — Historical receipts (Rounds 1-30 level cascade, Round 11 hybrid finding, sessions 26-32 arc, R22 install calibration)

Per-level validation receipts, session-32 prod Gemma port detail,
Round 11 hybrid-finding origin, R22 retrieval-card install calibration
arc, and R-delta-20 DT defaults-shipping receipt. Current install
patterns: `.claude/rules/Substrate.md`. This file exists for
archaeology — "which round proved which level", "what commit
introduced which install pattern", "why the R22 thresholds are
what they are".

## Round 11 hybrid-substrate origin

tq4 quantization DESTROYS compiled card weights (measured: dispatched_v4
791/791 → 70/791, 91% loss through tq4 roundtrip). Lloyd-Max codebook
is tuned for Gaussian LM weights; compiled cards have discrete ±1/±16
coefs.

Fix: `HybridGroupedSmall2DTransformer` with per-layer linear type.
Validated real Gemma 4 E4B attn_q.weight (2560×2048, 20480 tq4 blocks)
padded to (4096, 4096) → dequant → top-left matches original to
atol=1e-5. Current rule in `Substrate.md` §"Hybrid Per-Layer Linear
Type"; full bench in `MEMORY/atlas/turboquant_arc.md`.

## d_head=2 decomposition provenance

`d_head=256` attention = 128 × `d_head=2` sub-heads with scores
SUMMED before softmax — proven exact to float32 in commit `243b4ab`.
The architectural foundation for per-sub-head attention partition
(Level 5).

## Level cascade (all validated)

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

## Session 32 prod Gemma port

Established in session 32: `GemmaSubstrate` (`calm/llm_computer/gemma_substrate.py`)
is the full Gemma 4 E4B from GGUF in PyTorch. Per-sub-head attention
partition ported from Level-5 demo to prod; three attention modes
coexist in one Gemma layer verified (non-zero distinct diffs). Residual-
additive `CardSlot` pattern for cards with custom forwards.

Auto-upgrade loop (`scripts/gemma_learning_loop_demo.py`):
5 wrong addition prompts → log corrections to `KnowledgeStore` →
`build_recall_model()` produces a 4,304-param `Small2DTransformer`
recall card (3 ReGLU per fact, step-function dispatch) →
`CardSlot.attach` + `VerificationHook` → 5/5 correct. JSON persistence
round-trips bit-identical recall.

Earlier auto-upgrade substrate-native demo: 0/8 → 8/8 → 11/11 across
3 sessions. Proves the loop closes before the prod port.

## Compiled-program validation receipts

| Program | Proof round | Capability |
|---|---|---|
| `compiled_router.py` | Round 1 | ADD/MUL opcode dispatch |
| `dispatched_v2.py` | Round 4 | 5-op internal gating |
| `dispatched_v3.py` | Round 8 | 9 ops scaled |
| `dispatched_v4.py` | Round 10 | Cross-card gating via opcode shift |
| `composed_sum_threshold.py` | Round 21 | Inter-slot composition (64/64 exhaustive) |
| `depth_compound.py` | Round 24 | 3-stage depth pipeline |
| `reasoning_engine.py` | Round 30 | Comparison + logic + transitivity (512/512) |
| `compiled_in_gemma.py` | Round 22A | Card inside Gemma layer |
| `three_in_one_layer.py` | Round 29 | Level 5: 3 modes one layer |

Each proves its level of the cascade above.

## MathAdditionFacade origin

`calm/llm_computer/facades/math_addition.py` (`MathAdditionFacade`,
Round 8) is the first reusable domain class. New domains subclass
the pattern and get install/detach/set_prompt + save/load + detach
reversibility. Established "prefer facades over one-off install
scripts" as the canonical pattern.

## R22 retrieval-card install calibration arc

MQAR card installed on prod Gemma via `CardSlot` + `VerificationHook`
+ adapter. 7-round debug arc + R22e diagnostic shipped at
`min_margin=22.0` (+9/60 on 2026-04-21, commit `73df738`). R22f
(2026-04-22, commit `9691e06`) recalibrated the threshold to **14.5**
after diagnosing the flat N=10 cells as gate-silence, not card failure.

### Why 14.5, not 22.0 (R22f)

R22f sweep showed N=5 card margins cluster at p50=23.3 (above 22.0
threshold); N=10 p50=20.83 p5=15.21; N=15 p50=18.63 p5=16.39.
Threshold=22.0 was N=5-calibrated and over-gated N≥10 despite
standalone card being 100% correct (20/20 each) on those Ns.
Threshold=14.5 sits below observed p5 across all Ns and preserves
zero-regression invariant.

### Result at 14.5 (commit `9691e06`)

```
baseline:  42/60  (70.0%)
with card: 60/60  (100%)    Δ=+18 absolute, 43% relative
hook fired: 59/60
WINS: 18    REGR: 0
```

Per-cell at 14.5: all six cells 10/10. R22d rerun (commit `c3cc73f`,
all-keys-per-mem-block corpus) independently confirmed 42/60 → 60/60
at the same threshold.

### preserve=True side effect (R22b r6/r7, commit `7db6eb9`)

The `preserve=True` legacy mode pins the channel range even when the
card writes NOTHING. At rest (card silent), channels
`[ch_off:ch_off+d_card]` carry whatever the host layer wrote there
rather than being freely overwritten by downstream layers, which
subtly shifts Gemma's head projection. One measurable regression
(`q=v margin=0.00`) disappeared when switching to `preserve=False`.
Established the R22 default as `preserve=False`.

### Margin-gate alignment (commit `e169d6d`)

`card_output_fn` must itself skip the residual write when card's
(peak − median) margin is below `write_margin`. Without this gate,
low-confidence card output still writes, and the write propagates
through the head projection even when `VerificationHook.min_margin`
silences the logit bias. Established `write_margin == min_margin`
symmetry rule.

## DT default shipping (R-delta-20, commit `63a49fc`, 2026-04-21)

New trained cards default to DT (`CopyAugmentedDeltaNet`) rather
than plain `CopyAugmentedTransformer` for retrieval + structure-
extraction regimes. Held-out parity on copy-dominant structure
tasks (NL math 99.5% both), +21-84pp on retrieval-shaped tasks
(MQAR N=5-20), 3-10× faster training convergence, 1.18× inference
overhead via `decode_greedy_cached` (commit `e6f2d5c`).

Deployable card: `copy_augmented_delta_mqar_best.pt` (748 KB, 183K
params, MQAR N=5-15 @ 100% held-out). Full arc in
`MEMORY/atlas/delta_rule_arc.md`.

## Code-skeleton DT open arc (2026-04-22)

`dt_code_skel_v13_ep16_0193.pt` at 0.193 honest val on 520 held-out
problems. NOT install-viable — threshold is ≥ 0.40 honest val
before wiring to Gemma. Recipe differs from MQAR/NL defaults
(requires R26 aux copy-loss + R27 split-before-aug + gate init -1.0
+ EMA 0.995). See `delta_rule.md` §"Code-skeleton recipe" for current
canonical flags; full arc in `MEMORY/atlas/delta_rule_arc.md`.

## KVCacheTq4 multi-token prefill (R53.28)

`calm/llm_computer/gemma_substrate.py:KVCacheTq4` supports multi-token
prefill S≥1, per-layer position tracking, `trim_swa_storage` via
direct byte-copy. R53.28 receipt in `MEMORY/atlas/tracing_roadmap_part_1.md`.

## Fused flash-attention integration (R53.34)

`calm/llm_computer/tq4_flash_attn.py:fused_tq4_flash_attn_decode`
integrated via `_use_fused_flash_attn=True` default with runtime
N-gate `128 < cached_kv_len < 2048` per 2026-04-20 re-bench. SWA
layers fused; global layers (d_head=512) fall back to memoized
dequant. Full bench receipt + policy rationale:
`MEMORY/atlas/turboquant_arc.md`.

## Cross-refs

- Current install patterns: `.claude/rules/Substrate.md`
- DT install full arc: `MEMORY/atlas/delta_rule_arc.md`
- tq4 kernel + flash-attn receipts: `MEMORY/atlas/turboquant_arc.md`
- Tracing-arc validation (R28/R42/R43 etc.): `MEMORY/atlas/tracing_roadmap_part_1.md`
- Auto-upgrade CALM integration: `.claude/rules/calm_part_2.md` §"Auto-Upgrade Loop"
