# Architecture Rules

## Agent System
- All agents communicate with Ollama via HTTP POST to `localhost:11434/api/chat`
- Use `urllib.request` (stdlib only) for Ollama communication — no `requests` dependency in core agents
- Agent history is append-only within a session. Use `agent.reset()` to clear
- Tool calling loop: agent calls tools → gets results → calls Ollama again → up to `max_tool_rounds` iterations
- Coordinator uses JSON protocol: `{"delegate": "name", "task": "..."}` or `{"final": "answer"}`

## File Organization
- `agents/` — core harness code. No ML dependencies. Must work on Windows + WSL2 with just Python 3.11+
- `agents/distill/` — training pipeline. ML dependencies (torch, unsloth, transformers) only required here
- `models/` — Ollama Modelfiles
- `rust/` — upstream claw-code Rust port (separate build system)
- `src/` — upstream claw-code Python port (reference, not actively developed)

## Tools
- Tools have safety limits: 30s bash timeout, 500 line file truncation, 100 file search limit, 50 match truncation
- New tools must be added to both `TOOL_DEFINITIONS` (Ollama schema) and `execute_tool()` dispatcher in `tools.py`
- Tool output is always a string. Errors return `"Error: {message}"`

## Training Pipeline
- Stage 1 (reasoning base) runs once, creates a smarter base model
- Stage 2 (specialists) run on top of the reasoning base
- Training data format: JSONL with `{"messages": [{"role": "...", "content": "..."}]}`
- All training runs in WSL2. Stop Ollama before training to free VRAM
- Export pipeline: merge LoRA → GGUF (llama.cpp) → Ollama Modelfile → `ollama create`
