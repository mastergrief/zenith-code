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

## Modular Compute Architecture (CALM)

**Model reasons, backends compute, engine verifies.** Adding a backend is
equivalent to training — the model gets smarter at that domain instantly.

- **Auto-CALM** is the default. Model writes naturally, engine verifies claims,
  pre-computes answers, fixes code from NL descriptions. 100% on 40-problem benchmark.
- **Explicit CALM** (`<calm>` blocks) is the power-user path. 85-98% benchmark.
- **Backends** are modular Python files in `calm/backends/`. Each exports a `*_FUNCTIONS`
  dict registered in `expression.py` via try/import. Missing backends degrade gracefully.
- **9 backends, 70+ verified functions**: math, strings, wasm, code, security, dates,
  units, statistics, algorithms. Full spec: `.claude/rules/calm.md`
- To add a domain: write `calm/backends/X_ops.py` → export dict → register in `expression.py`
  → (optional) add precompute patterns in `auto_calm.py`

## File Organization
- `agents/` — core harness code (15 files, ~4,400 LOC). No ML dependencies. Must work on Windows + WSL2 with Python 3.11+
- `agents/distill/` — training pipeline (10 Python files + 1 notebook). ML dependencies (torch, unsloth, transformers) only required here. **Secondary to backends** — only needed for domains that can't be computed.
- `calm/` — CALM engine + Auto-CALM + modular backends (35+ files, ~9,500 LOC, 250 tests). Dependencies: `wasmtime` (optional, for wasm backend). Full spec: `.claude/rules/calm.md`
- `models/` — Ollama Modelfiles (3 files: qwen9b-fast, qwen4b-fast, reasoning-base)
- `bin/zenith` — launcher script: auto-starts llama.cpp, `--gguf PATH` first-arg flag, configurable via `ZENITH_*` env vars. Does NOT `cd` into repo.
- `scripts/` — dev tooling (needle_test, eval_base_models, smoke_test_harness, test_model_swap, generate_react_security_examples, setup_training)
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
- **Config** (`config.py`): loads `.zenithrc`/`zenith.json`, explicit `ENV_VARS` registry mapping config keys → `ZENITH_*` names. `ctx_size` default 262144
- **History** (`history.py`): `HistoryLog` with timestamped events, rendered via `/history`
- **Sessions** (`session.py`): save/load to `.zenith_sessions/`, JSON format. Auto-save on exit, `/resume` for latest
- **Hot-swap** (`model_swap.py`): `LlamaServerManager` orchestrates llama-server subprocess lifecycle. Adopts externally-started servers via `/props` + `/proc/net/tcp` PID lookup. `swap(target)` is a no-op when the target path is already loaded (uses `Path.resolve()` for comparison, so symlinks collapse — hard links are needed to force a real kill+restart for testing). Integration tested in `scripts/test_model_swap.py`.
- **Streaming**: dual backend streaming with thinking display. Readline integration with `~/.zenith_history`
- **`_streaming_text` flag invariant** (`harness.py`): tracks whether we're inside an open green ANSI block during a streamed response. **Do NOT reset it in the `response` event handler** — the main loop checks it to decide whether to re-print the response. Resetting in the handler causes the main loop's "non-streamed" branch to fire and double-print every streamed response (the bug fixed in commit `c11232a`). The handler may print `{RESET}` to close the color, but only the main loop should set `_streaming_text = False`.
- **Agent context limit lookup invariant** (`agent.py:~174`, session 2026-04-07): when `backend == "llamacpp"`, `Agent.__init__` must call `detect_llamacpp_model()` (which queries `/props` for the loaded GGUF path) and pass that to `detect_context_limit()`. **Do NOT pass the literal string `"llamacpp"`** — that would always match the generic 65K fallback and skip per-GGUF lookups in `MODEL_CONTEXT_LIMITS`. The previous wiring had this bug; every session silently got 65K regardless of loaded model. The `if max_context_tokens is not None` branch is explicit so an explicit caller override takes precedence over auto-detection.
- **Harness loaded-model cache invariant** (`harness.py`, session 2026-04-07): `Harness.__init__` queries `/props` once and caches `self._loaded_llamacpp_model`. The `/swap` command handler AND the `/backend llamacpp` handler **must** refresh this cache AND call `_compute_compact_threshold()` to update `max_context_tokens` on every agent in `self.agents`. Forgetting to refresh leaves agents compacting on the OLD model's limit (e.g., swap to Qwen after Gemma still uses Gemma's threshold). Both handlers currently do this — keep them in sync if you add a third path that swaps models.
- **89% safe-ctx compaction margin** (`harness.py:_compute_compact_threshold`, raised from 85% in session 2026-04-08): the compaction threshold is `min(per-GGUF model limit, int(ctx_size * 0.89))`. At default 256K ctx the binding constraint is the Gemma model entry (232960 = 227.5K), giving 29184 tokens of headroom. **This is BELOW `EFFORT_LEVELS["max"]["max_tokens"]` (32768)** — by user choice. Max-effort responses can soft-truncate by ~3.5K when conversation sits right at the threshold; the next turn compacts and full 32K is available again. Smaller `ZENITH_CTX` values still bind via `safe_ctx` (e.g. 131072 → safe_ctx 116654 → caps below model limit). If you raise `max_tokens` further, raise the safe-ctx multiplier or accept more truncation.

## Serving Architecture
- **llama.cpp (primary)**: Gemma 4 E4B tq4 or Q5_K_M at **256K context** with tq4 or Q4 KV cache (~5-7 GB VRAM, pre-allocated)
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
