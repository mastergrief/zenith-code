---
paths:
  - "agents/**"
  - "bin/**"
  - "rust/**"
---

# Architecture Rules

> Historical receipts (substrate-port session detail, DT/PT+Delta
> commits, retrieval-card install arc cross-ref, decode-perf bench
> numbers, production-feature commit-cited bug fixes, tq4 alignment
> session fix): see `MEMORY/atlas/architecture_arc.md`.

This file covers **current invariants of the Python agent harness**
(`agents/`) and the **cross-cutting file organization** of the repo.
Subsystem specs live in their own rule files — cross-refs below.

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

## Subsystem cross-refs

Detail for every subsystem lives in its own rule. This file summarizes
interfaces, not internals.

- **Substrate + cards** → `Substrate.md` (legacy/adjacent unless reopened)
- **CALM** → `calm.md`
- **HRM-Text-1.58** → `hrm-158.md` (checkpoints in `claw-code-hrm-text-158`)
- **PT / DT** → `delta_rule.md` + `training.md` (legacy/adjacent)
- **tq4 / flash-attn** → `turboquant.md`
- **Serving + VRAM** → `environment.md`
- **NIAH context limits** → `niah_validation.md`

Substrate stack, extensions, card typology, Gemma install API, CRLM pipeline:
`Substrate.md` + `calm.md`.

## File Organization

- `agents/` — core harness. No ML deps. Windows + WSL2, Python 3.11+.
- `agents/distill/` — training pipeline (ML deps here only). Secondary.
- `calm/` — CALM engine (live infrastructure). Spec: `calm.md`.
- `calm/hrm/` + `calm/llm_computer/` — legacy/adjacent in this repo;
  active HRM-Text-1.58 training + checkpoints live in
  `claw-code-hrm-text-158`.
- `models/` — Ollama Modelfiles.
- `bin/zenith` — launcher: llama.cpp auto-start, `--gguf PATH`, `ZENITH_*`.
- `scripts/` — dev tooling. Per-subsystem rules for path-scoped detail.
- `.claude/MEMORY/evals/` — NIAH/A/B reports (`compact.py:MODEL_CONTEXT_LIMITS`).
- `rust/` — upstream Rust port (separate build).
- `src/` — upstream Python port (reference, not active).

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
  new_string, must be unique (unless replace_all=True)
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
  llama.cpp generic fallback 65K). NIAH-validated — don't change
  without re-running `scripts/needle_test.py`. Summary compression
  (1200 chars, 24 lines max). Env override: `ZENITH_AUTO_COMPACT_TOKENS`.
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
  `Path.resolve()` for comparison — symlinks collapse; hard links
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

## Serving + VRAM + training

Full specs live in dedicated rule files:

- **Serving architecture, GGUF paths, `--parallel 1` requirement,
  llama.cpp slot-context patch, Ollama fallback set** → `environment.md`
  §"Serving Architecture"
- **Why Q4/tq4 KV is mandatory** (KV-cache VRAM math, contractive
  softmax quality argument, pre-allocation behavior) → `environment.md`
  §"Why Q4 KV Cache" + `turboquant.md`
- **Training pipeline** (Stage 1 reasoning base, Stage 2 specialists,
  `train_on_responses_only`, JSONL schema, export → GGUF) →
  `training.md` + `distillation.md`; active HRM-Text-1.58 curriculum →
  `hrm-158.md`
