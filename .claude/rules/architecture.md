# Architecture Rules

> Historical receipts (substrate-port session detail, DT/PT+Delta
> commits, retrieval-card install arc cross-ref, decode-perf bench
> numbers, production-feature commit-cited bug fixes, tq4 alignment
> session fix): see `MEMORY/atlas/architecture_arc.md`.

## Agent System

- Dual backend: Ollama (`localhost:11434/api/chat`) and llama.cpp
  (`localhost:8080/v1/chat/completions`)
- Auto-detection: `detect_backend()` checks llama.cpp first, falls
  back to Ollama
- Use `urllib.request` (stdlib only) — no `requests` dependency in
  core agents
- Streaming: Ollama uses NDJSON lines, llama.cpp uses SSE
  (OpenAI-compatible)
- Thinking mode: llama.cpp supports `enable_thinking: true`, streams
  `reasoning_content` separately via
  `thinking_start`/`thinking_token`/`thinking_end` events
- Connection retry: 3 attempts with exponential backoff (1s/2s/4s)
  on both backends
- System prompt builder: auto-discovers CLAUDE.md, adds cwd/date/tools,
  caps at 2000 chars
- Effort mode: `low` (3072 tokens, concise), `medium` (6144, default),
  `max` (32768, deep thinking)
- llama.cpp sampling: `temperature=0.7, frequency_penalty=0.8,
  presence_penalty=0.3, max_tokens=effort-dependent`. Same params on
  streaming and non-streaming paths.
- Output dedup (storage layer): `_is_repeating()` catches tail-window
  repeats (>200 chars), `_find_halved_duplicate()` catches `A+A`
  patterns, `_dedup_blocks()` removes duplicate paragraphs
  post-generation. **Note**: visible response duplication is almost
  always a *display* bug, not model looping — check stored session
  state first.
- Agent history is append-only within a session. Use `agent.reset()`
  to clear.
- Auto-compaction: when history exceeds `max_context_tokens`, old
  messages are summarized into a system message.
  `max_context_tokens = min(per-GGUF limit from MODEL_CONTEXT_LIMITS,
  int(ctx_size * 0.89))`. The 89% safe-ctx margin leaves headroom for
  the next response.
- Tool calling loop: up to `max_tool_rounds` iterations (default 10).
  Works on both backends (SSE delta assembly for tool calls).
- Coordinator JSON protocol: `{"delegate": "name", "task": "..."}`
  or `{"final": "answer"}`.
- SpecialistCoordinator auto-selects between **hot-swap mode**
  (llama.cpp + specialist GGUFs discovered on disk via
  `discover_specialist_models()`) and **Ollama multi-model mode**
  (per-agent distinct Ollama model names); falls back to single base
  model if neither is available.

## Substrate Pattern

**The model IS the substrate.** Validated through Level 5 on the
substrate-native demo (`HybridGroupedSmall2DTransformer`), then ported
to prod Gemma 4 E4B (`GemmaSubstrate`): `convert_layer_to_fp32` +
`install_card_in_attention` + per-sub-head dispatch via
`attention_partition`. Three attention modes coexist in one Gemma
layer with verified non-zero distinct diffs. Plus residual-additive
`CardSlot` pattern for cards with custom forwards (PTs).

Full spec: `Substrate.md`.

- **Substrate** = `Small2DTransformer` + `d_head=2` invariant +
  channel allocation + gate-graph IR + per-sub-head attention
  partition (3 modes: grouped-softmax, single-softmax,
  single-hard_max). The model's weight tensor IS the substrate.
- **Card** = compiled (gate-graph IR → weights, exact) or trained
  (SGD). Installed into the substrate via `install_compiled_card` at
  reserved channel/sub-head/FFN/vocab/layer rectangles.
- **Domain** = a facade with imports/exports (StdLib + CompiledOps)
  hosting HRM + compiled ops + knowledge facts for one knowledge area.
- **Unified substrate** = Gemma (tq4 layers, softmax) + N domain
  facades (FP32 layers, mixed softmax/hard_max) + knowledge DB. One
  `.pt` file.

Capacity: 1024 free sub-heads × 35 SWA layers = 35,840 compute slots.
~32 sub-heads per domain → **30 domains** on RTX 4070 (8 GB).

### Facade / Import System

```python
stdlib = StdLib(exports={"a": 3, "bias": 1})
adder = CompiledOp(imports={"x": "a"}, exports="sum")
model = build_program(stdlib, [adder, ...], head)
```

Linker resolves imports to channels, auto-schedules layers. Bad
imports caught at build time. File: `calm/llm_computer/program_builder.py`.

### Persistent Knowledge DB

Corrections compiled as step-function indicators (3 ReGLU per fact).
Cross-session via save/reload. Auto-upgrade loop: CALM catches error
→ compile into weights → persist. Files: `persistent_knowledge.py`,
`auto_upgrade.py`.

## Modular Compute Architecture (CALM)

**Model reasons, backends compute, engine verifies.** Adding a
backend is equivalent to training — the model gets smarter at that
domain instantly.

- **Auto-CALM** is the default. Model writes naturally, engine
  verifies claims, pre-computes answers, fixes code from NL
  descriptions. 100% on 40-problem benchmark.
- **Explicit CALM** (`<calm>` blocks) is the power-user path.
- **Backends** are modular Python files in `calm/backends/`. Each
  exports a `*_FUNCTIONS` dict registered in `expression.py` via
  try/import. Missing backends degrade gracefully.
- **120 backends, 1002 verified functions, 550 NL patterns**:
  compute (81 `*_ops.py`) + knowledge (39 `*_kb.py`). Full spec:
  `calm.md`.
- **39 cognitive modules** in 5 layers: verification, reasoning,
  quality, meta, planning. Auto-routed by `calm/router.py`. Full
  spec: `calm.md`.
- **Engine V2** (`calm/engine_v2.py`): 7-phase pipeline with
  self-healing quality loop, adaptive thinking budget (2K→32K),
  cross-turn state, module learning.
- To add a domain: write `calm/backends/X_ops.py` (compute) or
  `X_kb.py` (knowledge) → export `X_FUNCTIONS` dict + optional
  `X_NL_PATTERNS` list → done (auto-discovery registers both, zero
  other files to edit).

## Pointer Transducer + LLM-Computer Architecture

The CRLM thesis: **partition intelligence into structure (learned,
modest scale) + values (compiled, exact)**. Pointer Transducers
extract problem structure from NL via copy-augmented attention;
LLM-Computer recomputes every value via a deterministic interpreter
backed by the CALM function registry.

### Pointer Transducer (`calm/llm_computer/copy_augmented.py`)

`CopyAugmentedTransformer`: subclasses `Small2DTransformer`, adds
learned copy gate (1 linear → sigmoid) + pointer attention (dedicated
copy Q/K projections). 1,089 extra params (0.6%). At each decode
step: `p_copy * P_copy + (1-p_copy) * P_gen`. Digits → copy from
input, operators → generate from vocab.

**DT (Delta-Transducer) / `CopyAugmentedDeltaNet`** is the default
trained-card architecture **for retrieval + structure-extraction
regimes**. Copy gate + DeltaNet Householder fast-weight backbone.
Matches plain PT on structure tasks + wins large margins on retrieval
tasks (MQAR with high N). Chunkwise UT-transform training (3-7×
speedup) + cached `decode_greedy_cached` (~1.18× plain-PT inference).
Plain PT stays as ablation baseline. Full spec: `delta_rule.md`.
**Code-skeleton DT** is an open arc — recipe differs from retrieval
defaults. See `delta_rule.md` §"Code-skeleton recipe".

**Forward returns log-probs** (not logits). Use `F.nll_loss`, not
`F.cross_entropy`. The copy distribution is a probability
(scatter_add of attention weights), not logits.

**Training rules** (full spec in `training.md`):
- Copy gate bias initialized at -2.0 (model starts preferring
  generation, learns to copy).
- `max_len` must exceed prefix + decode budget — positional
  embeddings cap sequence length.
- One PT per output-language family.
- Scheduled sampling (tf_ratio 1.0→0.3), autoreg eval as gate metric.

**Prod Gemma install**: retrieval PTs install via
`CardSlot(layer_idx, ch_off, pt, d_card=80, card_input_fn=adapter,
use_full_residual=True, output_fn=writer).attach(m, preserve=False)`
with aligned `write_margin=min_margin=T` (4-gate default — see
`delta_rule.md` for canonical threshold + per-N calibration rule).
PT's copy-augmented attention can't reduce to a sub-head mode, so
CardSlot (separate forward + additive residual write + optional
preservation masking) is the right pattern. Chained CRLM proven
inside one Gemma forward: PT writes structure log_probs at reserved
channels → compute card reads those via `card_input_fn`, computes
the answer, writes at output channels → `VerificationHook` biases
Gemma's BPE digit logit.

### Legacy HRM (`calm/hrm/model.py`)

`HRMSeq2Seq`: encoder-decoder with nested L/H recurrence, 48K params,
`--structure-only` mode. Superseded by PT for new work. 5 checkpoints
still functional for eval comparison. Peak: 90% autoreg with
scheduled sampling.

### Accuracy priority order

```
Accuracy stuck? Check:
1. Data distribution — every valid input region covered? (free)
2. Mechanism — right operation for the task? (cheap, e.g. copy vs generate)
3. Output-family split — one model handling too many output languages? (moderate)
4. Capacity — model genuinely too small? (expensive, last resort)
```

### LLM-Computer (`calm/llm_computer/`)

Implementation of Percepta's LLM-Computer research (see
`RESEARCH/LLM-COMPUTER/`):

- `Small2DTransformer` (`model.py`): vanilla PyTorch, `d_head=2`,
  optional `use_hard_max=True`. Standard `nn.MultiheadAttention` +
  gated ReLU FFN + causal mask + learned positional embeddings.
  Weights are compiled source code, not statistical summary.
- `HullKVCache` (`hull_cache.py`): online 2D convex hull via
  Andrew's monotone chain. **108× speedup** vs linear scan at N=2K.
  Parity with batched hard-max attention validated against compiled
  programs. Not yet wired into `Small2DTransformer.forward()` —
  perf path for long sequences; programs use S ≤ 5 where linear
  scan wins.
- Gate-graph IR (`gate_graph.py`) — compute + hardware families:
  - **Compute** (interpreter walks): `Const`, `BinOp`, `Delegate`,
    `Result`.
  - **Hardware** (compiler walks): `TokenEmbed`, `PosEmbed`, `LookUp`,
    `LookUpExact` (parabolic-key `k_j = (2j, -j²)` with per-channel
    coefficients), `ReGLU`, `LinearHead`, `TokenInput`/`TokenOutput`.
- Declarative compiler (`compile.py`):
  `compile_program(graph, d_model, n_heads, n_layers, d_ffn, max_len,
  vocab_size)` zeroes every weight, walks hardware nodes, populates
  tok/pos/QKV/out/ffn/head tensors per-node. `d_head == 2` enforced
  by assert.
- Greedy auto-scheduler (`schedule.py`): `auto_schedule(graph)` assigns
  each node its minimum valid `(layer, phase)` based on channel
  availability.
- Parser (`parse.py`): `parse_expression()` via Python `ast.parse`
  → `GateGraph`.
- Interpreter (`interpret.py`): topo-walks compute nodes; `Delegate`
  routes through `safe_eval`.
- **Compiled programs in `programs/`** (24 total — current
  inventory in `Substrate.md` §"Key Files"):
  - Primitives: `add_one`, `copy_past`, `increment_counter`, `threshold`.
  - Composition: `adder_tiny`, `adder` (10K/10K exhaustive).
  - Memory: `retrieve_by_index`, `retrieve_threshold`, `read_by_key`.
- **ReGLU key-squaring trick** (enables semantic-keyed lookup):
  `-k² = -k · ReLU(k)` for non-negative integer `k`. One ReGLU neuron
  in layer-0 FFN writes `-k²` to a residual channel; a later layer's
  `LookUpExact` reads it as `pos_key1` with `pos_key0_coef=2.0` on
  the raw key channel.
- **Grammar-constrained decoding** (`grammar_decode.py`):
  inference-time mask for valid math expressions + EOS boosting.
- **Substrate server** (`substrate_server.py`): OpenAI-compatible
  API serving PTs + CALM precompute. Keyword-based routing across
  PT domains. Optional llama-server fallback for general language.
- **Gemma substrate loader** (`gemma_substrate.py`): full Gemma 4
  E4B from GGUF in PyTorch. Architecture: 42 layers, GQA 8Q/2KV,
  per-layer head dim, proportional RoPE, per-layer embedding
  injection. Components: `MmapTq4Linear`, `FP32GemmaLinear`,
  `GpuQ6KEmbedding`, `KVCache` / `KVCacheStatic` / `KVCacheTq4`,
  `GemmaTokenizer`. Triton fused dequant kernels (`tq4_triton.py`)
  + fused flash-attention decode (`tq4_flash_attn.py`). Full spec:
  `turboquant.md` for kernels, `Substrate.md` for install API.

### Substrate Extensions (D2/D3/D5 + Fast Weights)

Four opt-in substrate primitives. Additive — defaults preserve base
`Small2DTransformer` behavior bitwise.

- **D2 computation traces** (`computation_trace.py`) —
  `TracedSmall2DTransformer` emits a `ComputationTrace` alongside
  logits. Captures per-layer attention weights + argmax, FFN active
  neuron count, peak activation, fast-weight norm, geometry name.
- **D3 mixed geometry** (`mixed_geometry.py`) — per-layer
  `layer_geometries` config. Five score functions: `euclidean`,
  `hyperbolic`, `spherical`, `toroidal`, `lattice`. At `d_head=2`
  these are uniquely accessible (closed-form 2D geometric operations).
- **D5 recurrent substrate** (`recurrent_substrate.py`) —
  `n_iterations` kwarg iterates the same layers on the residual
  stream within one forward pass. HRM-style L/H, Universal Transformer
  pattern. Weights shared across iterations.
- **Combined** (`combined_substrate.py`) — `CombinedSmall2DTransformer`
  bundles D2+D3+D5 for hybrid training.
- **Fast weights** (`fast_weights.py`) — `FastWeightSmall2DTransformer`
  Schlag-style asymmetric Hebbian writes at inference:
  `W_fast_t = λ·W_fast_{t-1} + η·outer(v_t, k_t)/d_model`, read via
  `W_fast @ q_t`. Runtime weight addition, no gradient descent.
  Optional `use_delta_rule` and `use_write_gate` config flags
  preserved for ablation.

### Substrate-Compliant Card Types

All cards are `.pt` files following `Small2DTransformer` architecture.

- **Compiled programs** — gate-graph IR → weights, no training,
  exact. Lives in `calm/llm_computer/programs/`. Full inventory in
  `Substrate.md` §"Key Files".
- **HRM specialists** — `HRMSeq2Seq` architecture (NOT on the
  substrate). Five specialists at 48K params via `--structure-only`.
- **SubstrateLM** — decoder-only `Small2DTransformer` trained on
  Claude reasoning corpus. BPE tokenizer, chat formatter.
- **SubstrateHRM** — decoder-only `Small2DTransformer` trained on
  NL→math structure with scheduled sampling.
- **SubstrateHRLM** — hybrid LM+HRM trained jointly with mode
  prefixes.
- **Future domain brains** (`CHRLM-Coding`, `CHRLM-Math`,
  `CHRLM-Legal`, etc.) — same substrate, different card configurations.

### Brain + Cards Composition Model

CHRLM-General **brain** handles NL + planning + reasoning + routing.
Dispatches to **cards** rather than implementing card capabilities
itself.

- Thin brain (~100M-500M params target) + thick toolset (compiled
  cards + HRM specialists + CALM backends).
- Brain's corpus = conversation + reasoning + planning examples
  (decomposition trees) + routing examples (card selection).
- Fractal composition: domain CHRLMs each use the same substrate,
  each is a brain + its own toolset. Top-level brain routes between
  domain brains.
- Composition is runtime via shared protocols, not compile-time via
  shared tensors. Each card is its own `.pt` file loaded on demand.

### Convergence pipeline

```
HRM emits structure → parse to GateGraph → interpret with safe_eval
                                                       ↓
                                              analytically-correct answer
```

Eval mode runs this path per-domain: `scripts/eval_hrm_math.py
--verified`, `eval_hrm_nl.py`, etc.

Future direction: replace the Python interpreter with a compiled
`Small2DTransformer` per query (the paper's Futamura projection).
Same IR, different execution substrate.

## File Organization

- `agents/` — core harness code (15 files, ~4,423 LOC). No ML
  dependencies. Must work on Windows + WSL2 with Python 3.11+.
- `agents/distill/` — training pipeline. ML dependencies (torch,
  unsloth, transformers) only required here. **Secondary to backends**
  — only needed for domains that can't be computed.
- `calm/` — CALM engine + Auto-CALM + modular backends + cognitive
  intelligence layer. Engine V2 pipeline with 120 backends, 39
  cognitive modules. Dependencies: `wasmtime` (optional, for wasm
  backend). Full spec: `calm.md`.
- `calm/hrm/` — HRM encoder-decoder. Per-domain data generators:
  `data.py` (math), `nl_data.py`, `word_data.py`, `gsm_data.py`,
  `multi_data.py`. 5 production checkpoints at
  `calm/hrm/checkpoints/*_best.pt`, all 48K params via
  `--structure-only`. Tests: `calm/hrm/tests/`.
- `calm/llm_computer/` — substrate core. `Small2DTransformer`,
  `HullKVCache`, gate-graph IR, declarative compiler, auto-scheduler,
  parser/interpreter, compiled programs in `programs/`. Substrate
  extensions: `fast_weights.py`, `computation_trace.py`,
  `mixed_geometry.py`, `recurrent_substrate.py`, `combined_substrate.py`,
  `substrate_lm.py`. Prod Gemma stack: `gemma_substrate.py`,
  `tq4_triton.py`, `tq4_flash_attn.py`. Trained substrate cards in
  `checkpoints/`. Tests in `tests/`.
- `models/` — Ollama Modelfiles.
- `bin/zenith` — launcher script: auto-starts llama.cpp, `--gguf
  PATH` first-arg flag, configurable via `ZENITH_*` env vars.
- `scripts/` — dev tooling. Training, eval, bench scripts.
- `.claude/MEMORY/evals/` — NIAH and A/B eval reports (authoritative
  for `compact.py:MODEL_CONTEXT_LIMITS`).
- `rust/` — upstream claw-code Rust port (9 crates + workspace,
  separate build).
- `src/` — upstream claw-code Python port (reference, not actively
  developed).

## Tools

- 20 tools: `bash`, `read_file`, `write_file`, `edit_file`, `grep`,
  `list_files`, `list_directory`, `Agent`, `AgentCreate`, `AgentMessage`,
  `AgentGet`, `AgentList`, `AgentTerminate`, `Sleep`, `WebFetch`,
  `WebSearch`, `AskUserQuestion`, `TodoWrite`, `TodoRead`, `MultiEdit`
- Permission check via `check_permission(tool, args, mode)` runs
  before every tool dispatch
- 3 permission modes: `READ_ONLY`, `WORKSPACE_WRITE` (default),
  `FULL_ACCESS`
- Bash classification: `classify_bash()` returns
  `SAFE`/`WRITE`/`DESTRUCTIVE`/`BLOCKED` with git subcommand awareness
- User confirmation: `WRITE`/`DESTRUCTIVE` bash and file writes
  prompt `[y/N]` in WORKSPACE_WRITE mode
- System paths blocked: `/etc/`, `/usr/`, `/var/`, `/boot/`, `/sys/`,
  `/proc/`, `~/.ssh/`
- Safety limits: 30s bash timeout, 500-line default read_file limit
  (with offset/limit params), binary detection, 100 file search
  limit, 50 match truncation
- `edit_file` validates: old_string must exist, must differ from
  new_string, must be unique (unless replace_all=True). Returns
  context preview.
- New tools must be added to `TOOL_DEFINITIONS`, `execute_tool()`
  dispatcher, and harness `_on_event` display
- Tool output is always a string. Errors return `"Error: {message}"`.
  Denied returns `"Denied by user: {reason}"`. Blocked returns
  `"Blocked: {reason}"`.

## Production Features (current invariants)

- **Permissions** (`permissions.py`): `PermissionMode` enum +
  `BashRisk` enum with `classify_bash()` and `check_permission()`.
- **Compaction** (`compact.py`): summarizes old messages, preserves
  last 4 verbatim. **Per-GGUF** context limits in
  `MODEL_CONTEXT_LIMITS` (Gemma 4 E4B 200K, Qwen 3.5 4B 130K,
  llama.cpp generic fallback 65K). Values are NIAH-validated — don't
  change without re-running `scripts/needle_test.py`. Summary
  compression (1200 chars, 24 lines max). Env override:
  `ZENITH_AUTO_COMPACT_TOKENS`.
- **Config** (`config.py`): loads `.zenithrc`/`zenith.json`, explicit
  `ENV_VARS` registry mapping config keys → `ZENITH_*` names.
  `ctx_size` default 524288.
- **History** (`history.py`): `HistoryLog` with timestamped events,
  rendered via `/history`.
- **Sessions** (`session.py`): save/load to `.zenith_sessions/`,
  JSON format. Auto-save on exit, `/resume` for latest.
- **Hot-swap** (`model_swap.py`): `LlamaServerManager` orchestrates
  llama-server subprocess lifecycle. Adopts externally-started
  servers via `/props` + `/proc/net/tcp` PID lookup. `swap(target)`
  is a no-op when the target path is already loaded (uses
  `Path.resolve()` for comparison, so symlinks collapse — hard links
  needed to force kill+restart for testing).
- **Streaming**: dual backend streaming with thinking display.
  Readline integration with `~/.zenith_history`.

### Load-bearing invariants (don't violate)

- **`_streaming_text` flag** (`harness.py`): tracks whether we're
  inside an open green ANSI block during a streamed response.
  **Do NOT reset it in the `response` event handler** — the main
  loop checks it to decide whether to re-print the response.
  Resetting in the handler causes the main loop's "non-streamed"
  branch to fire and double-print every streamed response.
- **Agent context limit lookup** (`agent.py`): when
  `backend == "llamacpp"`, `Agent.__init__` must call
  `detect_llamacpp_model()` (which queries `/props` for the loaded
  GGUF path) and pass that to `detect_context_limit()`. **Do NOT
  pass the literal string `"llamacpp"`** — that always matches the
  generic 65K fallback and skips per-GGUF lookups in
  `MODEL_CONTEXT_LIMITS`. The `if max_context_tokens is not None`
  branch ensures explicit caller override takes precedence.
- **Harness loaded-model cache** (`harness.py`): `Harness.__init__`
  queries `/props` once and caches `self._loaded_llamacpp_model`. The
  `/swap` command handler AND the `/backend llamacpp` handler **must**
  refresh this cache AND call `_compute_compact_threshold()` to
  update `max_context_tokens` on every agent. Forgetting leaves
  agents compacting on the OLD model's limit.
- **89% safe-ctx compaction margin**
  (`harness.py:_compute_compact_threshold`): the threshold is
  `min(per-GGUF model limit, int(ctx_size * 0.89))`. At default
  256K ctx the binding constraint is the Gemma model entry
  (232960 = 227.5K), giving 29184 tokens of headroom. **This is BELOW
  `EFFORT_LEVELS["max"]["max_tokens"]` (32768)** by user choice —
  max-effort responses can soft-truncate by ~3.5K when conversation
  sits right at the threshold.

## Serving Architecture

- **llama.cpp (primary)**: Gemma 4 E4B tq4 or Q5_K_M at **512K
  context** with tq4 or Q4 KV cache (~5-7 GB VRAM, pre-allocated)
- **Production GGUF**: `~/models/gemma-4-E4B-it-tq4-aligned.gguf`
  (5.0 GB, tq4, 132-byte blocks). Alternatives:
  `gemma-4-E4B-it-Q5_K_M.gguf`, `Qwen3.5-4B.Q5_K_M.gguf`. Hot-swap
  via `/swap` or `ZENITH_MODEL`.
- llama-server binary at `~/llama.cpp/build/bin/`, **branch `zenith`**
  with TurboQuant fusion + OP_TIMING.
- **TurboQuant tq4 KV**: `--cache-type-k tq4_k256 --cache-type-v tq4_k256`.
  4.125 bpw, 16-level Lloyd-Max, Pi rotation (seed=42). 132-byte
  blocks for 4-byte aligned CUDA loads. **Old 130-byte GGUFs
  incompatible.**
- **llama-server `--parallel 1` requirement**: without it,
  llama-server defaults to 4 slots and splits `--ctx-size` across
  them, so each slot gets only `ctx_size / 4`. For single-user
  workflow always pass `--parallel 1`. `bin/zenith` passes this;
  manual `llama-server` invocations must too.
- **Gemma 4 GGUF rope-scaling metadata override**: Gemma 4 E4B's
  GGUF metadata forces `rope scaling = linear`; `--rope-scaling yarn`
  CLI flag is silently ignored. Past trained context is raw RoPE
  extrapolation. Works empirically up to ~200K on single-needle but
  multi-needle degrades at 220K.
- **llama.cpp slot context cap patch** (outside repo): unpatched
  `tools/server/server-context.cpp` hardcodes per-slot context to
  `n_ctx_train`, silently capping `--ctx-size`. Patch comments out
  the cap; not upstreamed; re-apply after any `git pull` on
  llama.cpp source.
- **Hot-swap**: `agents/model_swap.py:LlamaServerManager` kills +
  restarts llama-server for each swap (~5-15s depending on disk
  page-cache warmth). Used by `/swap` harness command and
  `SpecialistCoordinator` when specialist GGUFs discovered on disk.
- **Ollama (fallback)**: stock models, quick testing. Pulled set:
  `qwen3.5:4b`, `qwen3.5:9b`, `qwen3:0.6b/4b/8b`, plus custom
  `qwen4b-fast`, `qwen9b-fast`, `reasoning-base` (verify with
  `curl -s localhost:11434/api/tags`).

## Why Q4 KV Cache (Mandatory, Not Optional)

- KV cache scales as `2 (K+V) × num_layers × num_kv_heads × head_dim
  × ctx × dtype_bytes`. For 4B at 64K in FP16 the cache alone is
  8-15 GB — over our 8 GB VRAM budget before the model even loads.
- `q4_0` stores ~4.5 bits/element vs FP16's 16 (~3.5-4× shrink).
  Brings the cache to ~3.0-3.5 GB, leaving room for weights
  (~2.9 GB) + compute buffers within 8 GB.
- Without `--cache-type-k q4_0 --cache-type-v q4_0` we cap at ~8K
  context — too small for real coding work.
- Quality cost is near-zero: KV values are runtime activations (not
  trained parameters), and the attention softmax is contractive —
  small numerical noise averages out. llama.cpp community testing
  confirms Q4 KV is nearly indistinguishable from FP16 for inference.
- K is slightly more sensitive than V; we use Q4 for both as the
  most aggressive safe setting.
- Cache is **pre-allocated at server startup**, not on demand. Even
  a 100-token prompt locks the full ~6.3 GB. No surprise mid-session
  OOMs, ceiling known immediately at startup.
- Specialists must use the same `--cache-type-k q4_0 --cache-type-v q4_0`
  when served — none fit at 64K without it.

## Training Pipeline

- Stage 1 (reasoning base): 0.8B local or 4B cloud. 4B trained and
  serving.
- Stage 2 (specialists): run on top of 4B reasoning base. Not yet
  trained.
- Both stages use `train_on_responses_only` — masks instruction
  tokens.
- Training data format: JSONL with
  `{"messages": [system, user, assistant]}`, assistant starts with
  `<think>`.
- 4B training requires cloud GPU (Colab A100 40GB+). 0.8B fits
  locally.
- Export pipeline: merge LoRA → GGUF (llama.cpp) → serve via
  llama-server or Ollama.
