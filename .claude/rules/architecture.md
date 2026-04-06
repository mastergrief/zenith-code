# Architecture Rules

## Agent System
- Dual backend: Ollama (`localhost:11434/api/chat`) and llama.cpp (`localhost:8080/v1/chat/completions`)
- Auto-detection: `detect_backend()` checks llama.cpp first, falls back to Ollama
- Use `urllib.request` (stdlib only) — no `requests` dependency in core agents
- Streaming: Ollama uses NDJSON lines, llama.cpp uses SSE (OpenAI-compatible)
- Thinking mode: llama.cpp supports `enable_thinking: true`, streams `reasoning_content` separately via `thinking_start`/`thinking_token`/`thinking_end` events
- Connection retry: 3 attempts with exponential backoff (1s/2s/4s) on both backends
- System prompt builder: auto-discovers CLAUDE.md, adds cwd/date/tools, caps at 2000 chars
- Effort mode: `low` (1024 tokens, concise), `medium` (2048, default), `max` (8192, deep thinking)
- Output dedup: `_is_repeating()` detects streaming loops (100-char window), `_dedup_blocks()` removes duplicate paragraphs post-generation
- Agent history is append-only within a session. Use `agent.reset()` to clear
- Auto-compaction: when history exceeds `max_context_tokens`, old messages are summarized into a system message
- Tool calling loop: up to `max_tool_rounds` iterations (default 10). Works on both Ollama and llama.cpp (SSE delta assembly for tool calls)
- Coordinator uses JSON protocol: `{"delegate": "name", "task": "..."}` or `{"final": "answer"}`

## File Organization
- `agents/` — core harness code (13 files, ~2,000 lines). No ML dependencies. Must work on Windows + WSL2 with Python 3.11+
- `agents/distill/` — training pipeline (10 Python files + 1 notebook). ML dependencies (torch, unsloth, transformers) only required here
- `models/` — Ollama Modelfiles (3 files: qwen9b-fast, qwen4b-fast, reasoning-base)
- `bin/claw` — launcher script: auto-starts llama.cpp, configurable via `CLAW_*` env vars
- `rust/` — upstream claw-code Rust port (9 crates + workspace, separate build system)
- `src/` — upstream claw-code Python port (reference, not actively developed)

## Tools
- 6 tools: `bash`, `read_file`, `write_file`, `edit_file`, `grep`, `list_files`
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
- **Compaction** (`compact.py`): summarizes old messages, preserves last 4 verbatim. Per-model context limits (64K for llama.cpp). Summary compression (1200 chars, 24 lines max). Env override: `CLAW_AUTO_COMPACT_TOKENS`
- **Config** (`config.py`): loads `.clawrc`/`claw.json`, `CLAW_*` env var overrides
- **History** (`history.py`): `HistoryLog` with timestamped events, rendered via `/history`
- **Sessions** (`session.py`): save/load to `.claw_sessions/`, JSON format. Auto-save on exit, `/resume` for latest
- **Streaming**: dual backend streaming with thinking display. Readline integration with `~/.claw_history`

## Serving Architecture
- **llama.cpp (primary)**: 4B Q5_K_M at 64K context with Q4 KV cache (~6.3GB VRAM, pre-allocated)
- GGUF at `~/models/Qwen3.5-4B.Q5_K_M.gguf`, built with CUDA at `~/llama.cpp/build/bin/`
- Hot-swap planned: specialists swap onto same GPU (5-10s swap, full GPU each)
- **Ollama (fallback)**: stock models, quick testing. `qwen3.5:0.8b` available

## Training Pipeline
- Stage 1 (reasoning base): 0.8B local or 4B cloud. 4B trained and serving.
- Stage 2 (specialists): run on top of 4B reasoning base. Not yet trained.
- Both stages use `train_on_responses_only` — masks instruction tokens
- Training data format: JSONL with `{"messages": [system, user, assistant]}`, assistant starts with `<think>`
- 4B training requires cloud GPU (Colab A100 40GB+). 0.8B fits locally.
- Export pipeline: merge LoRA → GGUF (llama.cpp) → serve via llama-server or Ollama
