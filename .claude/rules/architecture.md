# Architecture Rules

## Agent System
- Dual backend: Ollama (`localhost:11434/api/chat`) and llama.cpp (`localhost:8080/v1/chat/completions`)
- Auto-detection: `detect_backend()` checks llama.cpp first, falls back to Ollama
- Use `urllib.request` (stdlib only) — no `requests` dependency in core agents
- Streaming: Ollama uses NDJSON lines, llama.cpp uses SSE (OpenAI-compatible)
- Thinking mode: llama.cpp supports `enable_thinking: true`, streams `reasoning_content` separately via `thinking_start`/`thinking_token`/`thinking_end` events
- Connection retry: 3 attempts with exponential backoff (1s/2s/4s) on both backends
- System prompt builder: auto-discovers CLAUDE.md, adds cwd/date/tools, caps at 2000 chars
- Effort mode: `low` (3072 tokens, concise), `medium` (6144, default), `max` (32768, deep thinking)
- llama.cpp sampling: `temperature=0.7, frequency_penalty=0.8, presence_penalty=0.3, max_tokens=effort-dependent`. Same params on both streaming and non-streaming paths to keep behavior consistent.
- Output dedup (storage layer): `_is_repeating()` catches tail-window repeats (>200 chars), `_find_halved_duplicate()` catches `A+A` patterns with any/no separator (commit `3cf1a69`), `_dedup_blocks()` removes duplicate paragraphs post-generation. **Note**: visible response duplication is almost always a *display* bug (e.g. the harness double-print fixed in `c11232a`), not model looping — check stored session state first before assuming the model is repeating.
- Agent history is append-only within a session. Use `agent.reset()` to clear
- Auto-compaction: when history exceeds `max_context_tokens`, old messages are summarized into a system message. `max_context_tokens` is computed by `Harness._compute_compact_threshold()` as `min(per-GGUF limit from MODEL_CONTEXT_LIMITS, int(ctx_size * 0.89))`. The 89% safe-ctx margin leaves headroom for the next turn's response — at default 256K ctx, headroom is 29184 tokens, which is BELOW max-effort `max_tokens` (32768). Max-effort responses can soft-truncate by ~3.5K right at the threshold; after compaction fires, full 32K is available again.
- Tool calling loop: up to `max_tool_rounds` iterations (default 10). Works on both Ollama and llama.cpp (SSE delta assembly for tool calls)
- Coordinator uses JSON protocol: `{"delegate": "name", "task": "..."}` or `{"final": "answer"}`
- SpecialistCoordinator auto-selects between **hot-swap mode** (llama.cpp + specialist GGUFs discovered on disk via `discover_specialist_models()`) and **Ollama multi-model mode** (per-agent distinct Ollama model names); falls back to single base model if neither is available

## Substrate Pattern

**The model IS the substrate.** Session 30 validated through Level 5
on the substrate-native demo (`HybridGroupedSmall2DTransformer`).
**Session 32** ported the full pattern to prod Gemma 4 E4B
(`GemmaSubstrate`): `convert_layer_to_fp32` + `install_card_in_attention`
+ per-sub-head dispatch via `attention_partition` — three attention
modes coexist in one Gemma layer with verified non-zero distinct
diffs. Plus the residual-additive `CardSlot` pattern for cards with
custom forwards (PTs).

Full spec: `.claude/rules/Substrate.md`

- **Substrate** = `Small2DTransformer` + `d_head=2` invariant + channel
  allocation + gate-graph IR + per-sub-head attention partition (3 modes:
  grouped-softmax, single-softmax, single-hard_max). The model's weight
  tensor IS the substrate; programs are installed into free sub-head slots.
- **Card** = compiled (gate-graph IR → weights, exact) or trained (SGD).
  Installed into the substrate via `install_compiled_card` at reserved
  channel/sub-head/FFN/vocab/layer rectangles.
- **Domain** = a facade with imports/exports (StdLib + CompiledOps) hosting
  HRM + compiled ops + knowledge facts for one knowledge area.
- **Unified substrate** = Gemma (tq4 layers, softmax) + N domain facades
  (FP32 layers, mixed softmax/hard_max) + knowledge DB. One `.pt` file.

Key session-30 results:
- **Level 5 validated**: 3 attention modes in one layer, zero cross-talk
- **Real Gemma bytes**: 2 layers byte-installed from GGUF + card, one tensor
- **GPU**: 68× speedup at 889M params on RTX 4070
- **Auto-upgrade**: CALM → compile → persist, self-improving across sessions
- **HRM 90% autoreg**: scheduled sampling, 15 min on RTX 4070
- **Compiled reasoning**: comparison, logic, transitivity — exact

Capacity: 1024 free sub-heads × 35 SWA layers = 35,840 compute slots.
~32 sub-heads per domain → **30 domains** on RTX 4070 (8 GB).

### Facade / Import System

```python
stdlib = StdLib(exports={"a": 3, "bias": 1})
adder = CompiledOp(imports={"x": "a"}, exports="sum")
model = build_program(stdlib, [adder, ...], head)
```

Linker resolves imports to channels, auto-schedules layers. Bad imports
caught at build time. File: `calm/llm_computer/program_builder.py`.

### Persistent Knowledge DB

Corrections compiled as step-function indicators (3 ReGLU per fact).
Cross-session via save/reload. Auto-upgrade loop: CALM catches error →
compile into weights → persist. 0/8 → 11/11 across 3 sessions.
Files: `persistent_knowledge.py`, `auto_upgrade.py`.

## Modular Compute Architecture (CALM)

**Model reasons, backends compute, engine verifies.** Adding a backend is
equivalent to training — the model gets smarter at that domain instantly.

- **Auto-CALM** is the default. Model writes naturally, engine verifies claims,
  pre-computes answers, fixes code from NL descriptions. 100% on 40-problem benchmark.
- **Explicit CALM** (`<calm>` blocks) is the power-user path. 85-98% benchmark.
- **Backends** are modular Python files in `calm/backends/`. Each exports a `*_FUNCTIONS`
  dict registered in `expression.py` via try/import. Missing backends degrade gracefully.
- **116 backends, 1002 verified functions, 550 NL patterns**: compute (79 `*_ops.py`)
  + knowledge (10 `*_kb.py`). Full spec: `.claude/rules/calm.md`
- **39 cognitive modules** in 5 layers: verification, reasoning, quality, meta, planning.
  Auto-routed by `calm/router.py` (33-70ms overhead). Full spec: `.claude/rules/calm.md`
- **Engine V2** (`calm/engine_v2.py`): 7-phase pipeline with self-healing quality loop,
  adaptive thinking budget (2K→32K), cross-turn state, module learning.
- To add a domain: write `calm/backends/X_ops.py` (compute) or `X_kb.py` (knowledge)
  → export `X_FUNCTIONS` dict + optional `X_NL_PATTERNS` list → done
  (auto-discovery registers both, zero other files to edit)

## Pointer Transducer + LLM-Computer Architecture

The CRLM thesis: **partition intelligence into structure (learned, modest scale) + values (compiled, exact)**. Pointer Transducers extract problem structure from NL via copy-augmented attention; LLM-Computer recomputes every value via a deterministic interpreter backed by the CALM function registry.

### Pointer Transducer (`calm/llm_computer/copy_augmented.py`, session 31)

- `CopyAugmentedTransformer`: subclasses `Small2DTransformer`, adds learned copy gate (1 linear → sigmoid) + pointer attention (dedicated copy Q/K projections). 1,089 extra params (0.6%). At each decode step: `p_copy * P_copy + (1-p_copy) * P_gen`. Digits → copy from input, operators → generate from vocab.
- **Forward returns log-probs** (not logits). Use `F.nll_loss`, not `F.cross_entropy`. The copy distribution is a probability (scatter_add of attention weights), not logits.
- **Copy gate bias initialized at -2.0** — model starts preferring generation, learns to copy. Without this, early training is unstable.
- **`max_len` must exceed prefix + decode budget** — positional embeddings cap sequence length. Autoreg eval caps `gen_budget = min(max_gen, pos_limit - len(ids) - 1)`.
- **One PT per output-language family**: function-call (`fn(args)`), infix arithmetic (`a + b`), boolean logic (`a > b and`). ~3-5 families cover 30+ domains. Adding a domain within a family = data-only.
- **Training**: scheduled sampling (tf_ratio 1.0→0.3), autoreg eval as gate metric, `--epochs 500`, balanced `_sample_operand()` in all data generators. Scripts: `scripts/train_copy_*.py`.
- **Checkpoints**: `calm/hrm/checkpoints/copy_*_best.pt` (NL math 100%, word 98%, GSM 100%, funcall 86%, logic 86%).
- **Remaining ceiling**: 3+ operand copy accuracy (68-83%). Fix: two-stage decode via D5 recurrence (skeleton → slot fill).
- **Prod Gemma install (session 32)**: PTs install via `CardSlot(layer_idx, ch_off, pt, d_card=80, card_input_fn=adapter, use_full_residual=True, output_fn=writer).attach(m, preserve=True)`. PT's copy-augmented attention can't reduce to a sub-head mode, so CardSlot (separate forward + additive residual write + preservation masking) is the right pattern. Chained CRLM proven inside one Gemma forward: `copy_augmented_hrm` PT writes structure log_probs at ch[2400:2480] → adder_tiny reads those channels via `card_input_fn`, computes the answer, writes at ch[2480:2488] → `VerificationHook` biases Gemma's BPE digit logit. See commit `f5455f6`.

### Legacy HRM (`calm/hrm/model.py`, sessions 24-30)

`HRMSeq2Seq`: encoder-decoder with nested L/H recurrence, 48K params, `--structure-only` mode. Superseded by PT for all new work. 5 checkpoints still functional for eval comparison. Peak: 90% autoreg (session 30, scheduled sampling).

### Accuracy priority order (session 31 finding)

```
Accuracy stuck? Check:
1. Data distribution — every valid input region covered? (free)
2. Mechanism — right operation for the task? (cheap, e.g. copy vs generate)
3. Output-family split — one model handling too many output languages? (moderate)
4. Capacity — model genuinely too small? (expensive, last resort)
```

Session 31 never needed step 4. Steps 1-3 took 0%→100% (data), 68%→100% (mechanism), 74%→88% (split).

### LLM-Computer (`calm/llm_computer/`)

Implementation of Percepta's March 2026 research (RESEARCH/01-03):

- `Small2DTransformer` (`model.py`): vanilla PyTorch, `d_head=2`, optional `use_hard_max=True`. Standard `nn.MultiheadAttention` + gated ReLU FFN + causal mask + learned positional embeddings. Weights are compiled source code, not statistical summary.
- `HullKVCache` (`hull_cache.py`): online 2D convex hull via Andrew's monotone chain. **108× speedup** vs linear scan at N=2K (`tests/test_hull_cache.py`). Parity with batched hard-max attention validated against compiled programs (`tests/test_hull_cache_attention.py`). Not yet wired into `Small2DTransformer.forward()` — perf path for long sequences, our programs use S ≤ 5 where linear scan wins.
- Gate-graph IR (`gate_graph.py`) — compute + hardware families:
  - **Compute** (interpreter walks): `Const`, `BinOp`, `Delegate`, `Result`.
  - **Hardware** (compiler walks): `TokenEmbed` (per-token entries), `PosEmbed` (per-position entries), `LookUp` (copy-from-pos-0 attention), `LookUpExact` (parabolic-key `k_j = (2j, -j²)` with per-channel coefficients — `pos_key0_coef=2.0` on a scalar key channel enables semantic-keyed retrieval without a precomputed `2·key` table), `ReGLU` (one FFN neuron: `out += coef · val · ReLU(gate)`), `LinearHead`, `TokenInput`/`TokenOutput` (legacy Layer-1 shorthand).
- Declarative compiler (`compile.py`): `compile_program(graph, d_model, n_heads, n_layers, d_ffn, max_len, vocab_size)` zeroes every weight, walks hardware nodes, populates tok/pos/QKV/out/ffn/head tensors per-node. Per-layer head and neuron counters allocate sequentially. `d_head == 2` enforced by assert.
- Greedy auto-scheduler (`schedule.py`): `auto_schedule(graph)` assigns each `LookUp`/`LookUpExact`/`ReGLU` node its minimum valid `(layer, phase)` based on channel availability. Phase 0 = layer 0 attn, phase 1 = layer 0 FFN, etc. Callers no longer hand-pick layers. MILP scheduling (RESEARCH/03 §6) deferred until programs hit ~30+ gates with real slot pressure.
- Parser (`parse.py`): `parse_expression()` via Python `ast.parse` → `GateGraph`. `extract_problem_from_trace()` extracts the pre-`=` segment from HRM scratchpad and strips `<call>` markers.
- Interpreter (`interpret.py`): topo-walks compute nodes; `Delegate` routes through `safe_eval` (full 1002-function backend registry).
- **9 compiled programs in `programs/`:**
  - Primitives: `add_one` (1,280 params), `copy_past` (2,560), `increment_counter` (2,176), `threshold` (216). Each has an `*_ir.py` IR-compiled counterpart; 3 of 4 bit-match the hand-wired version (`copy_past` differs only in head packing; behavior identical).
  - Composition: `adder_tiny` (1,020 params, 1-digit sum via LookUp + 14 ReGLU step functions, 16/16 exhaustive), `adder` (486,012 params, 2-digit sum, **10,000/10,000 exhaustive** in 0.38s).
  - Memory: `retrieve_by_index` (1,164 params, position-indexed parabolic-key retrieval, 256/256 exhaustive), `retrieve_threshold` (590 params, same-layer attn+FFN composition, 256/256), `read_by_key` (1,410 params, semantic KV via ReGLU key-squaring + coefficient-parametrized `LookUpExact`, 96/96 = 4! perms × 4 queries).
- **ReGLU key-squaring trick** (enables semantic-keyed lookup): `-k² = -k · ReLU(k)` for non-negative integer `k`. One ReGLU neuron in layer-0 FFN writes `-k²` to a residual channel; a later layer's `LookUpExact` reads it as `pos_key1` with `pos_key0_coef=2.0` on the raw key channel. This lifts a scalar key into the `(2k, -k²)` parabolic form exactly.
- **Grammar-constrained decoding** (`grammar_decode.py`): inference-time mask for valid math expressions + EOS boosting. Null result on current models but infrastructure shipped.
- **Substrate server** (`substrate_server.py`): OpenAI-compatible API serving PTs + CALM precompute. Keyword-based routing across 7 PT domains. Optional llama-server fallback for general language.
- **Gemma substrate loader** (`gemma_substrate.py`, sessions 31-32): full Gemma 4 E4B from GGUF in PyTorch. `MmapTq4Linear` (GPU-preloaded tq4 bytes, dequant on GPU), `FP32GemmaLinear` (drop-in replacement for hosting in-attention card installs), `GpuQ6KEmbedding` (Q6_K components on GPU), `KVCache` / `KVCacheStatic` (CUDA-Graph-friendly fixed buffers) / `KVCacheTq4` (real tq4 storage, 4.4× memory), `GemmaTokenizer` (262K vocab from GGUF). Architecture: 42 layers, GQA 8Q/2KV, per-layer head dim, proportional RoPE, per-layer embedding injection. **42 tok/s steady decode** (160× over baseline, 90% of llama.cpp), 5.07 GB GPU. Triton fused dequant kernels (`tq4_triton.py`, see `turboquant.md`); CUDA Graph capture (`generate_with_graph`); in-attention card install (`install_card_in_attention`, `convert_layer_to_fp32`); per-sub-head attention dispatch (`attention_partition`, three modes coexist in one layer); residual-additive install (`CardSlot.attach(preserve=True)`); verification feedback (`VerificationHook`); learning loop (`KnowledgeStore` + `CardSlot` — see `scripts/gemma_learning_loop_demo.py`). Domain registry: `.claude/MEMORY/substrate_registry.md`. Full spec: `.claude/rules/Substrate.md`.

### Substrate Extensions (D2/D3/D5 + Fast Weights)

Four opt-in substrate primitives added this session. All are additive —
defaults preserve base `Small2DTransformer` behavior bitwise, so the 15
compiled programs and existing checkpoints (including `substrate_hrm_nl_best.pt`)
work unchanged.

- **D2 computation traces** (`computation_trace.py`) —
  `TracedSmall2DTransformer` emits a `ComputationTrace` alongside logits
  when `forward(idx, trace=...)` is called. Trace captures per-layer
  attention weights + argmax, FFN active neuron count, peak activation,
  optional fast-weight norm, geometry name. Foundation for
  self-introspection and targeted online learning.
- **D3 mixed geometry** (`mixed_geometry.py`) — per-layer
  `layer_geometries` config. Five score functions: `euclidean` (dot
  product, default), `hyperbolic` (Poincaré disk distance), `spherical`
  (cosine similarity), `toroidal` (wrapped Euclidean), `lattice` (snap-
  to-integer). At `d_head=2` these are uniquely accessible (closed-form
  2D geometric operations). `MixedGeometrySmall2DTransformer` dispatches
  per layer; `layer_geometries=None` falls back to parent behavior.
- **D5 recurrent substrate** (`recurrent_substrate.py`) —
  `n_iterations` kwarg iterates the same layers on the residual stream
  within one forward pass. HRM-style L/H, Universal Transformer pattern.
  Weights shared across iterations — more thinking without parameter
  cost. `RecurrentConfig.max_iterations` clamps runaway requests.
- **Combined** (`combined_substrate.py`) —
  `CombinedSmall2DTransformer` bundles D2+D3+D5 for hybrid training
  (used by `scripts/train_substrate_hrmlm_v2.py`).
- **Fast weights** (`fast_weights.py`) —
  `FastWeightSmall2DTransformer` Schlag-style asymmetric Hebbian writes
  at inference: `W_fast_t = λ·W_fast_{t-1} + η·outer(v_t, k_t)/d_model`,
  read via `W_fast @ q_t`. Runtime weight addition, no gradient descent.
  Round 1 result: **99.1% on held-out 3-pair associative recall at
  d_head=2** (vs vanilla 35.3% — the mechanism works at this narrow head
  dimension, a novel empirical result with no prior literature). Round 2
  (fusion): fast weights stay silent when projections are silent — no
  interference with compiled programs. Rounds 3 (d_model scaling)
  and 4 (delta rule + write gate) nulls diagnosed the n=10 ceiling
  as structural interference (cross-key leakage), not capacity. Optional
  `use_delta_rule` and `use_write_gate` config flags preserved for
  ablation. Fourteen tests in `tests/test_substrate_extensions.py`.

### Substrate-Compliant Card Types

All cards are `.pt` files following the `Small2DTransformer` architecture.
Types:

- **Compiled programs** (24 in `programs/`) — gate-graph IR → weights,
  no training, exact. Original 15: `adder` (10K/10K), `gcd` (256/256),
  `factorial` (9/9), `is_prime` (99/99), `dispatched` (279/279),
  `countdown`, `isa`, `adder_tiny`, `add_one`, `copy_past`,
  `increment_counter`, `threshold`, `retrieve_by_index`,
  `retrieve_threshold`, `read_by_key`. Session 30 additions:
  `compiled_router` (ADD/MUL dispatch), `dispatched_v2` (5 ops),
  `dispatched_v3` (9 ops), `dispatched_v4` (5 ops + cross-card gating),
  `composed_sum_threshold` (inter-slot composition),
  `depth_compound` (3-stage pipeline), `reasoning_engine` (comparison
  + logic + transitivity), `compiled_in_gemma` (inside Gemma layer),
  `three_in_one_layer` (Level 5: 3 modes one layer).
- **HRM specialists** (5 in `calm/hrm/checkpoints/`) — `HRMSeq2Seq`
  architecture (NOT on the substrate). Separate file format; migration
  to substrate-native is proven via `substrate_hrm_nl_best.pt` (180K
  params, 99.1% on NL templates). Five specialists kept at 48K params
  via `--structure-only`: math, nl, word, gsm, meta (56% OOD, capacity-
  bound).
- **SubstrateLM** (`substrate_lm_mvp.pt`, 1.25M params) — decoder-only
  `Small2DTransformer` trained on Claude reasoning corpus. BPE tokenizer,
  chat formatter. MVP demonstrates substrate hosts LM behavior
  (ppl 4096→424 in 13 min CPU). Format acquired (<think>, numbered
  lists, code markers); content coherence requires scale (100M-500M for
  useful prose).
- **SubstrateHRM** (`substrate_hrm_nl_best.pt`, 180K params) — decoder-
  only `Small2DTransformer` trained on NL→math structure with scheduled
  sampling. **90% autoregressive accuracy** (the metric that matters).
  Teacher-forced val_acc 99.0%. Trained in 879s on RTX 4070. Installs
  into the unified substrate at reserved sub-heads. Bit-identical to
  standalone when embedded in substrate (proven Round 9: 0.00e+00 diff).
- **SubstrateHRLM** (`substrate_hybrid_mvp.pt`, 1.25M; v2 in flight) —
  hybrid LM+HRM trained jointly with mode prefixes. v1 PARTIAL result:
  LM +8.7% cross-task transfer, HRM mode collapsed to 0% (token-count
  imbalance + single template family). v2 fixes curriculum (multi20
  template variety, oversampling, mode-loss weighting) and adds
  D3/D5 extensions.
- **Future domain brains** (`CHRLM-Coding`, `CHRLM-Math`, `CHRLM-Legal`,
  etc.) — same substrate, different card configurations per domain.

### Brain + Cards Composition Model

CHRLM-General **brain** handles NL + planning + reasoning + routing. It
dispatches to **cards** rather than implementing card capabilities itself
— don't make the brain do arithmetic when compiled `adder` is available.

- Thin brain (~100M-500M params target) + thick toolset (15+ compiled
  cards + 5-7 HRM specialists + 116 CALM backends).
- Brain's corpus = conversation + reasoning (existing) + planning
  examples (decomposition trees) + routing examples (card selection).
- Fractal composition: domain CHRLMs (Coding, Math, etc.) each use the
  same substrate, each is a brain + its own toolset. Top-level brain
  routes between domain brains.
- Composition is runtime via shared protocols, not compile-time via
  shared tensors. Each card is its own `.pt` file loaded on demand.

### Convergence pipeline

```
HRM emits structure → parse to GateGraph → interpret with safe_eval
                                                       ↓
                                              analytically-correct answer
```

Eval mode runs this path per-domain: `scripts/eval_hrm_math.py --verified`, `eval_hrm_nl.py`, `eval_hrm_word.py`, `eval_hrm_gsm.py`, `eval_hrm_multi.py`. CRLM scaling-law empirics across 4 production checkpoints (all 48K params, all `--structure-only`):

| Input language | Max chars | Per-token | Full-expression / structural |
|---|---:|---:|---:|
| Math expression echo (3-digit) | ~20 | 100% | 30/30, smoke 5/5 |
| NL templates ("what is X plus Y?") | ~30 | 99.8% | 29/30, smoke 5/5 |
| Word problems (names, pronouns, multi-step) | 78 | 99.7% | 30/30, smoke 5/5 |
| GSM-style (subordinate clauses, 3-4 terms) | 104 | 99.6% | 28/30 — **first observed ceiling** |

Multi-task HRM (`calm/hrm/checkpoints/multi_task_best.pt`) pools all four domains into one 48K model: 100% per-token val_acc; per-domain eval via `scripts/eval_hrm_multi.py`. This is Vector 2 phase 1.

Future direction: replace the Python interpreter with a compiled `Small2DTransformer` per query (the paper's Futamura projection). Same IR, different execution substrate. Scoped into four phases in `.claude/MEMORY/CRLM_SPEC.md` §H.

## File Organization
- `agents/` — core harness code (15 files, ~4,400 LOC). No ML dependencies. Must work on Windows + WSL2 with Python 3.11+
- `agents/distill/` — training pipeline (10 Python files + 1 notebook). ML dependencies (torch, unsloth, transformers) only required here. **Secondary to backends** — only needed for domains that can't be computed.
- `calm/` — CALM engine + Auto-CALM + modular backends + cognitive intelligence layer (~194 files, ~37,400 LOC, 250 tests). Engine V2 pipeline with 116 backends, 39 cognitive modules, adaptive thinking, self-healing, factual cross-check, module learning feedback loop. Dependencies: `wasmtime` (optional, for wasm backend). Full spec: `.claude/rules/calm.md`
- `calm/hrm/` — HRM (Hierarchical Reasoning Model) encoder-decoder. Core: `model.py` (HRM, HRMSeq2Seq, HRMEncoder, HRMDecoder), `train.py`/`inference.py`. Per-domain data generators: `data.py` (math), `nl_data.py` (NL templates), `word_data.py` (word problems), `gsm_data.py` (GSM-style narratives), `multi_data.py` (pooled). 5 production checkpoints at `calm/hrm/checkpoints/*_best.pt`, all 48K params via `--structure-only`. Dedicated tests: `calm/hrm/tests/`.
- `calm/llm_computer/` — substrate core. `Small2DTransformer` (`model.py`), `HullKVCache` (`hull_cache.py`), gate-graph IR (`gate_graph.py`), declarative compiler (`compile.py`), greedy auto-scheduler (`schedule.py`), parser/interpreter (`parse.py` + `interpret.py`), 15 compiled programs in `programs/`. **Substrate extensions**: `fast_weights.py` (D1 runtime Hebbian writes), `computation_trace.py` (D2 traces), `mixed_geometry.py` (D3 per-layer geometries), `recurrent_substrate.py` (D5 iteration budget), `combined_substrate.py` (D2+D3+D5 bundle), `substrate_lm.py` (BPE + training pipeline for substrate-native cards). **Prod Gemma stack (session 32)**: `gemma_substrate.py` (the loaded model + install API + CardSlot/VerificationHook + KVCache variants), `tq4_triton.py` (fused dequant Triton kernels for tq4 + Q6_K). Trained substrate cards live in `checkpoints/` (substrate_lm_mvp, substrate_hybrid_mvp, substrate_hrlm_v2, substrate_hrm_nl_best, synth_familyA variants). 78+ tests in `tests/`.
- `models/` — Ollama Modelfiles (3 files: qwen9b-fast, qwen4b-fast, reasoning-base)
- `bin/zenith` — launcher script: auto-starts llama.cpp, `--gguf PATH` first-arg flag, configurable via `ZENITH_*` env vars. Does NOT `cd` into repo.
- `scripts/` — dev tooling. CHRLM + fast-weights + substrate-native training: `experiment_fast_weights{.py, _fusion.py, _scaling.py, _round4.py}` (Rounds 1-4), `train_substrate_lm.py` (SubstrateLM MVP), `train_hybrid_substrate.py` (SubstrateHRLM v1), `train_substrate_hrmlm_v2.py` (v2 with D3/D5 + curriculum fixes + `--device auto` GPU support), `chat_substrate_lm.py` (MVP REPL), `unified_chat.py` (CHRLM + Gemma single-conversation REPL with routing gate). Legacy: needle_test, eval_base_models, smoke_test_harness, test_model_swap, generate_react_security_examples, setup_training.
- `.claude/MEMORY/evals/` — NIAH and A/B eval reports (authoritative for `compact.py:MODEL_CONTEXT_LIMITS`)
- `rust/` — upstream claw-code Rust port (9 crates + workspace, separate build system)
- `src/` — upstream claw-code Python port (reference, not actively developed)

## Tools
- 20 tools: `bash`, `read_file`, `write_file`, `edit_file`, `grep`, `list_files`, `list_directory`, `Agent`, `AgentCreate`, `AgentMessage`, `AgentGet`, `AgentList`, `AgentTerminate`, `Sleep`, `WebFetch`, `WebSearch`, `AskUserQuestion`, `TodoWrite`, `TodoRead`, `MultiEdit`
- Permission check via `check_permission(tool, args, mode)` runs before every tool dispatch
- 3 permission modes: `READ_ONLY`, `WORKSPACE_WRITE` (default), `FULL_ACCESS`
- Bash classification: `classify_bash()` returns `SAFE`/`WRITE`/`DESTRUCTIVE`/`BLOCKED` with git subcommand awareness
- User confirmation: `WRITE`/`DESTRUCTIVE` bash and file writes prompt `[y/N]` in WORKSPACE_WRITE mode
- System paths blocked: `/etc/`, `/usr/`, `/var/`, `/boot/`, `/sys/`, `/proc/`, `~/.ssh/`
- Safety limits: 30s bash timeout, 500-line default read_file limit (with offset/limit params), binary detection (NUL in first 8KB), 100 file search limit, 50 match truncation
- `edit_file` validates: old_string must exist, must differ from new_string, must be unique (unless replace_all=True). Returns context preview (lines around edit).
- New tools must be added to `TOOL_DEFINITIONS`, `execute_tool()` dispatcher, and harness `_on_event` display
- Tool output is always a string. Errors return `"Error: {message}"`. Denied returns `"Denied by user: {reason}"`. Blocked returns `"Blocked: {reason}"`

## Production Features
- **Permissions** (`permissions.py`): `PermissionMode` enum + `BashRisk` enum with `classify_bash()` and `check_permission()`
- **Compaction** (`compact.py`): summarizes old messages, preserves last 4 verbatim. **Per-GGUF** context limits in `MODEL_CONTEXT_LIMITS` (Gemma 4 E4B 200K, Qwen 3.5 4B 130K, llama.cpp generic fallback 65K). Values are NIAH-validated against `.claude/MEMORY/evals/2026-04-07_summary_needle_comparison.md` — don't change them without re-running `scripts/needle_test.py`. Summary compression (1200 chars, 24 lines max). Env override: `ZENITH_AUTO_COMPACT_TOKENS`
- **Config** (`config.py`): loads `.zenithrc`/`zenith.json`, explicit `ENV_VARS` registry mapping config keys → `ZENITH_*` names. `ctx_size` default 524288
- **History** (`history.py`): `HistoryLog` with timestamped events, rendered via `/history`
- **Sessions** (`session.py`): save/load to `.zenith_sessions/`, JSON format. Auto-save on exit, `/resume` for latest
- **Hot-swap** (`model_swap.py`): `LlamaServerManager` orchestrates llama-server subprocess lifecycle. Adopts externally-started servers via `/props` + `/proc/net/tcp` PID lookup. `swap(target)` is a no-op when the target path is already loaded (uses `Path.resolve()` for comparison, so symlinks collapse — hard links are needed to force a real kill+restart for testing). Integration tested in `scripts/test_model_swap.py`.
- **Streaming**: dual backend streaming with thinking display. Readline integration with `~/.zenith_history`
- **`_streaming_text` flag invariant** (`harness.py`): tracks whether we're inside an open green ANSI block during a streamed response. **Do NOT reset it in the `response` event handler** — the main loop checks it to decide whether to re-print the response. Resetting in the handler causes the main loop's "non-streamed" branch to fire and double-print every streamed response (the bug fixed in commit `c11232a`). The handler may print `{RESET}` to close the color, but only the main loop should set `_streaming_text = False`.
- **Agent context limit lookup invariant** (`agent.py:~174`, session 2026-04-07): when `backend == "llamacpp"`, `Agent.__init__` must call `detect_llamacpp_model()` (which queries `/props` for the loaded GGUF path) and pass that to `detect_context_limit()`. **Do NOT pass the literal string `"llamacpp"`** — that would always match the generic 65K fallback and skip per-GGUF lookups in `MODEL_CONTEXT_LIMITS`. The previous wiring had this bug; every session silently got 65K regardless of loaded model. The `if max_context_tokens is not None` branch is explicit so an explicit caller override takes precedence over auto-detection.
- **Harness loaded-model cache invariant** (`harness.py`, session 2026-04-07): `Harness.__init__` queries `/props` once and caches `self._loaded_llamacpp_model`. The `/swap` command handler AND the `/backend llamacpp` handler **must** refresh this cache AND call `_compute_compact_threshold()` to update `max_context_tokens` on every agent in `self.agents`. Forgetting to refresh leaves agents compacting on the OLD model's limit (e.g., swap to Qwen after Gemma still uses Gemma's threshold). Both handlers currently do this — keep them in sync if you add a third path that swaps models.
- **89% safe-ctx compaction margin** (`harness.py:_compute_compact_threshold`, raised from 85% in session 2026-04-08): the compaction threshold is `min(per-GGUF model limit, int(ctx_size * 0.89))`. At default 256K ctx the binding constraint is the Gemma model entry (232960 = 227.5K), giving 29184 tokens of headroom. **This is BELOW `EFFORT_LEVELS["max"]["max_tokens"]` (32768)** — by user choice. Max-effort responses can soft-truncate by ~3.5K when conversation sits right at the threshold; the next turn compacts and full 32K is available again. Smaller `ZENITH_CTX` values still bind via `safe_ctx` (e.g. 131072 → safe_ctx 116654 → caps below model limit). If you raise `max_tokens` further, raise the safe-ctx multiplier or accept more truncation.

## Serving Architecture
- **llama.cpp (primary)**: Gemma 4 E4B tq4 or Q5_K_M at **512K context** with tq4 or Q4 KV cache (~5-7 GB VRAM, pre-allocated)
- **Production GGUF**: `~/models/gemma-4-E4B-it-tq4-aligned.gguf` (5.0 GB, tq4, 132-byte blocks). **Alternative**: `gemma-4-E4B-it-Q5_K_M.gguf` (5.48 GB), `Qwen3.5-4B.Q5_K_M.gguf` (2.9 GB). Hot-swap via `/swap` or `ZENITH_MODEL`.
- llama-server binary at `~/llama.cpp/build/bin/`, **branch `zenith`** with TurboQuant fusion + OP_TIMING
- **TurboQuant tq4 KV**: `--cache-type-k tq4_k256 --cache-type-v tq4_k256`. 4.125 bpw, 16-level Lloyd-Max, Pi rotation (seed=42). 132-byte blocks for 4-byte aligned CUDA loads (session 16 alignment fix). **Old 130-byte GGUFs incompatible.**
- **llama-server `--parallel 1` requirement** (session 2026-04-07): without it, llama-server defaults to 4 slots and splits `--ctx-size` across them, so each slot gets only `ctx_size / 4`. For single-user workflow (the harness is always single-user) pass `--parallel 1`. `bin/zenith` passes this since commit `4644051` — manual `llama-server` invocations must pass it too.
- **Gemma 4 GGUF rope-scaling metadata override** (session 2026-04-07): Gemma 4 E4B's GGUF metadata forces `rope scaling = linear` in llama.cpp; the `--rope-scaling yarn` CLI flag is silently ignored. Extrapolating past `n_ctx_train=131072` uses raw linear RoPE extrapolation, not YaRN. Works empirically up to ~200K on single-needle (21/21 PASS at 220K), but multi-needle degrades to 4/5 at 220K. See `2026-04-07_gemma4_e4b_needle_256k_multi.md`.
- **llama.cpp slot context cap patch** (session 2026-04-07, outside repo): the unpatched `tools/server/server-context.cpp:763-766` hardcodes per-slot context to `n_ctx_train`, silently capping `--ctx-size` at the trained max. For NIAH testing past 128K on Gemma 4 E4B we patched the cap out locally. The patch is not upstreamed; re-apply after any `git pull` on llama.cpp source.
- **Hot-swap IMPLEMENTED** (session 2026-04-07): `agents/model_swap.py:LlamaServerManager` kills + restarts llama-server for each swap (~5–15s depending on disk page-cache warmth). Used by the `/swap` harness command and by `SpecialistCoordinator` when specialist GGUFs are discovered on disk. Specialist GGUFs don't exist yet, so hot-swap orchestration is dormant until specialists are trained.
- **Ollama (fallback)**: stock models, quick testing. Current pulled set: `qwen3.5:4b`, `qwen3.5:9b`, `qwen3:0.6b/4b/8b`, plus custom `qwen4b-fast:latest`, `qwen9b-fast:latest`, `reasoning-base:latest` (verify with `curl -s localhost:11434/api/tags`)

## Why Q4 KV Cache (Mandatory, Not Optional)
- KV cache scales as `2 (K+V) × num_layers × num_kv_heads × head_dim × ctx × dtype_bytes`. For 4B at 64K in FP16 the cache alone is 8–15 GB — over our 8 GB VRAM budget before the model even loads.
- `q4_0` stores ~4.5 bits/element vs FP16's 16 (~3.5–4× shrink). Brings the cache to ~3.0–3.5 GB, leaving room for weights (~2.9 GB) + compute buffers within 8 GB.
- Without `--cache-type-k q4_0 --cache-type-v q4_0` we cap at ~8K context — too small for real coding work where files and conversations routinely exceed that.
- Quality cost is near-zero: KV values are runtime activations (not trained parameters), and the attention softmax is contractive — small numerical noise averages out across many tokens. llama.cpp community testing confirms Q4 KV is nearly indistinguishable from FP16 for inference.
- K is slightly more sensitive than V; we use Q4 for both as the most aggressive safe setting. Fall back to K=Q8/V=Q4 only if quality regressions appear.
- Cache is **pre-allocated at server startup**, not on demand. Even a 100-token prompt locks the full ~6.3 GB. Trade: no surprise mid-session OOMs, ceiling known immediately at startup.
- Specialists must use the same `--cache-type-k q4_0 --cache-type-v q4_0` when served — none fit at 64K without it. Don't drop these flags for "experimental" specialist serving.

## Training Pipeline
- Stage 1 (reasoning base): 0.8B local or 4B cloud. 4B trained and serving.
- Stage 2 (specialists): run on top of 4B reasoning base. Not yet trained.
- Both stages use `train_on_responses_only` — masks instruction tokens
- Training data format: JSONL with `{"messages": [system, user, assistant]}`, assistant starts with `<think>`
- 4B training requires cloud GPU (Colab A100 40GB+). 0.8B fits locally.
- Export pipeline: merge LoRA → GGUF (llama.cpp) → serve via llama-server or Ollama
