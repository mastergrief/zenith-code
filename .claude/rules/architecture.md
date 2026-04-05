# Architecture Rules

## Agent System
- All agents communicate with Ollama via HTTP POST to `localhost:11434/api/chat`
- Use `urllib.request` (stdlib only) for Ollama communication — no `requests` dependency in core agents
- Streaming supported: `_call_ollama_stream()` reads NDJSON lines, yields tokens via `on_event("token", ...)`
- Agent history is append-only within a session. Use `agent.reset()` to clear
- Auto-compaction: when history exceeds `max_context_tokens`, old messages are summarized into a system message
- Tool calling loop: agent calls tools → gets results → calls Ollama again → up to `max_tool_rounds` iterations (default 10)
- Coordinator uses JSON protocol: `{"delegate": "name", "task": "..."}` or `{"final": "answer"}`

## File Organization
- `agents/` — core harness code (12 files, ~1,400 lines). No ML dependencies. Must work on Windows + WSL2 with just Python 3.11+
- `agents/distill/` — training pipeline (10 files). ML dependencies (torch, unsloth, transformers) only required here
- `models/` — Ollama Modelfiles
- `rust/` — upstream claw-code Rust port (9 crates, separate build system)
- `src/` — upstream claw-code Python port (reference, not actively developed)

## Tools
- 6 tools: `bash`, `read_file`, `write_file`, `edit_file`, `grep`, `list_files`
- Permission check runs before every tool dispatch via `DEFAULT_PERMISSIONS.blocks(name, args)`
- Destructive bash commands (rm -rf /, shred, fork bombs) are blocked
- Writes to system paths (/etc/, /usr/, ~/.ssh/) are blocked
- Tools have safety limits: 30s bash timeout, 500 line file truncation, 100 file search limit, 50 match truncation
- New tools must be added to `TOOL_DEFINITIONS` (Ollama schema), `execute_tool()` dispatcher, and harness `_on_event` display in `tools.py` and `harness.py`
- `edit_file` validates: old_string must exist, must differ from new_string, must be unique (unless replace_all=True)
- Tool output is always a string. Errors return `"Error: {message}"`. Blocked tools return `"Blocked: {reason}"`

## Production Features
- **Permissions** (`permissions.py`): `ToolPermissions` dataclass with deny-list for tool names, bash patterns, and write paths
- **Compaction** (`compact.py`): summarizes old messages when history exceeds model context limit. Preserves last 4 messages verbatim
- **History** (`history.py`): `HistoryLog` with timestamped events, rendered via `/history` command
- **Sessions** (`session.py`): save/load to `.claw_sessions/` directory, JSON format with agent name, model, history, timestamp
- **Streaming**: `Agent._call_ollama_stream()` yields tokens, harness prints them inline

## Training Pipeline
- Stage 1 (reasoning base) runs once, creates a smarter base model
- Stage 2 (specialists) run on top of the reasoning base
- Both stages use `train_on_responses_only` — masks instruction tokens so loss is only on model responses
- Training data format: JSONL with `{"messages": [{"role": "...", "content": "..."}]}`
- `filter_reasoning.py` filters junk/hallucinations and merges hand-written data with HuggingFace data
- All training runs in WSL2. Stop Ollama before training to free VRAM
- Export pipeline: merge LoRA → GGUF (llama.cpp) → Ollama Modelfile → `ollama create`
