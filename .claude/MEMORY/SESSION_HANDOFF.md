# Session Handoff — 2026-04-04

## Goal
Build a local multi-agent coding assistant powered by Qwen models via Ollama, with a distillation pipeline to create a swarm of specialist 0.6-0.8B models from larger teachers (9B + Claude Opus reasoning data).

## Completed

### Ollama Setup
- Installed Ollama on Windows, pulled models: qwen3:0.6b, qwen3:4b, qwen3:8b, qwen3.5:4b, qwen3.5:9b, qwen3.5:0.8b
- Created optimized Modelfiles: `qwen9b-fast` (100% GPU, 2K ctx), `qwen4b-fast` (100% GPU, 8K ctx)
- Confirmed WSL2 Ubuntu 24.04 can reach Ollama on localhost:11434

### GitHub Repo
- Forked `ultraworkers/claw-code` → `mastergrief/claw-code`
- Branch: `feature/multi-agent-qwen`
- **Nothing committed yet** — all work is unstaged

### Python Agent Harness (`agents/`)
- `agent.py` — Agent class with Ollama chat + tool calling loop
- `coordinator.py` — Coordinator delegates via JSON protocol
- `swarm.py` — Parallel agent execution (ThreadPoolExecutor)
- `tools.py` — 5 tools: bash, read_file, write_file, grep, list_files
- `harness.py` — Terminal REPL with /agents, /switch, /team, /solo, /spawn, /model, /distill commands
- `specialist_coordinator.py` — SpecialistCoordinator with auto-detection of specialist models
- `example.py` — Working demos (tested, all pass)
- **All tested and working** — tool calling, swarm broadcast, coordinator delegation confirmed

### Rust Ollama Provider (partial)
- Added `Ollama` variant to `ProviderKind` enum in `rust/crates/api/src/providers/mod.rs`
- Added `ollama()` config + `ollama_running()` check in `openai_compat.rs`
- Registered qwen models in MODEL_REGISTRY
- **NOT BUILT** — needs MSVC Build Tools (installed rustc 1.94.1 but no linker)

### Distillation Pipeline (`agents/distill/`)
- `config.py` — 6 domains defined (orchestrator, typescript, python, rust, devops, reviewer)
- `generate.py` — DatasetGenerator using 9B teacher (working but slow)
- `train_base.py` — Stage 1: reasoning base from Claude data
- `train.py` — Stage 2: specialist fine-tuning (auto-detects reasoning base)
- `export.py` — GGUF conversion + Ollama registration
- `validate.py` — A/B comparison using 9B as judge
- `fetch_datasets.py` — Downloads HuggingFace datasets (TeichAI + Crownelius)
- `seeds/` — 50-100 seed prompts per domain (all 6 written)

### Training Data
- `orchestrator.jsonl` — 130 routing examples (Claude-authored, high quality)
- `claude_reasoning.jsonl` — 3,047 Claude Opus reasoning traces (from HuggingFace)
- `python.jsonl` — 10 examples (subagent still writing, slow due to large code responses)
- `typescript.jsonl` — 11 examples (subagent still writing)
- `rust.jsonl` — 11 examples (deprioritized for now)

### Training Runs
- **Orchestrator on Qwen 3 0.6B**: SUCCESS — 57 seconds, loss 2.16→0.50, merged to `agents/distill/merged/orchestrator/`
- **Reasoning base on Qwen 3.5 0.8B**: IN PROGRESS — step 5/191, ~1.5 hours remaining

### Project Docs
- `.claude/CLAUDE.md` — full project reference
- `.claude/commands/update.md` — `/update` command ported from mercury
- `.claude/commands/handoff.md` — `/handoff` command ported from mercury
- `.claude/rules/architecture.md` — agent system + file org rules
- `.claude/rules/training.md` — VRAM budget + known issues

## In Progress

### Reasoning Base Training (Stage 1)
- Running on Qwen 3.5 0.8B with 3,047 Claude reasoning examples
- Settings: batch=1, grad_accum=16, seq_len=1024, packing=false, 1 epoch
- **OOM was the main battle** — Qwen 3.5's 248K vocab makes fused CE loss hungry
- Fix: batch=1, seq_len=1024, packing=false, stop Ollama before training
- ETA: ~1.5 hours from session start

### Training Data Generation (subagents)
- Python and TypeScript subagents writing JSONL but slow (large code responses)
- Rust/DevOps/Reviewer deprioritized — focus on orchestrator + python + typescript

## Next Steps

1. **Wait for reasoning base training to complete** — check `agents/distill/merged/reasoning_base/`
2. **Train orchestrator on reasoning base** — retrain using the 0.8B reasoning base instead of vanilla 0.6B
3. **Complete Python + TypeScript training data** — if subagents didn't finish, generate manually or use `generate.py` with 9B
4. **Train Python + TypeScript specialists** (Stage 2) on top of reasoning base
5. **Export specialists to Ollama** — `python -m agents.distill.export --domain orchestrator`
6. **Test end-to-end** — launch harness with `/distill on` and `/team`, verify routing + specialist responses
7. **Install MSVC Build Tools** — to build Rust claw-code: `winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"`
8. **Commit and push** — all work is uncommitted on `feature/multi-agent-qwen`
9. **Integrate claw-code src/ features** — session persistence, permissions, cost tracking, transcript compaction (see session notes)

## Key Context

### Failed Approaches
- **Qwen 3.5 0.8B with batch=4, packing=true**: OOM — fused CE loss eats all 8GB VRAM
- **Qwen 3.5 0.8B with batch=2, packing=false, seq=2048**: Still OOM
- **Qwen 3.5 0.8B with batch=1, packing=false, seq=1024**: WORKS (current)
- **Qwen 3 0.6B with batch=4, packing=true, 3K examples**: OOM on reasoning data (worked on 130 orchestrator examples)
- **WSL2 bash -c with PATH expansion**: `Program Files (x86)` breaks bash. Use `wsl -e bash -c` or write script files
- **winget from Git Bash**: Produces no output. Use PowerShell or `cmd.exe /c` for winget

### Hardware State
- GPU: RTX 4070 Laptop GPU, 8 GB VRAM, CUDA 8.9
- Ollama: should be stopped during training, restart after
- WSL2: Ubuntu 24.04, Python 3.13, torch 2.10.0+cu128, unsloth 2026.4.2
- Rust: 1.94.1 installed but no MSVC linker

### Discovery: TeichAI Distilled Model
- `TeichAI/Qwen3.5-4B-Claude-Opus-Reasoning-Distill` on HuggingFace
- Proves: ~4K examples, single epoch, full fine-tune on Qwen 3.5 4B works
- Their datasets used in our pipeline: TeichAI/Claude-Opus-4.6-Reasoning-887x + Crownelius/Opus-4.6-Reasoning-2100x-formatted
- Could be used as a smarter 4B base model instead of stock Qwen 3.5 4B

### Claw-Code Features Worth Porting
From `src/`: session_store.py (36L), permissions.py (21L), history.py (23L), cost_tracker.py (14L), transcript.py (24L) — all small, ready to integrate

## Files in Project

```
agents/
  __init__.py                    — exports Agent, Coordinator, Swarm, SpecialistCoordinator
  agent.py                       — base Agent class with Ollama + tool calling
  coordinator.py                 — task delegation via JSON protocol
  swarm.py                       — parallel agent execution
  tools.py                       — 5 tools (bash, read, write, grep, list_files)
  harness.py                     — terminal REPL with slash commands
  specialist_coordinator.py      — routes to fine-tuned specialist models
  example.py                     — demo scripts
  distill/
    __init__.py                  — package marker
    config.py                    — domains, model names, QLoRA params
    generate.py                  — teacher→JSONL data generator
    train_base.py                — Stage 1: reasoning base training
    train.py                     — Stage 2: specialist training
    export.py                    — GGUF conversion + Ollama registration
    validate.py                  — A/B specialist vs base comparison
    fetch_datasets.py            — HuggingFace dataset downloader
    seeds/                       — 50-100 seed prompts per domain (6 files)
    data/                        — JSONL training data (gitignored)
    checkpoints/                 — LoRA adapters (gitignored)
    merged/                      — merged models (gitignored)
models/
  Modelfile.qwen9b-fast          — optimized 9B (100% GPU, 2K ctx)
  Modelfile.qwen4b-fast          �� optimized 4B (100% GPU, 8K ctx)
.claude/
  CLAUDE.md                      — project reference doc
  commands/update.md             — /update command
  commands/handoff.md            — /handoff command
  rules/architecture.md          — agent + file org rules
  rules/training.md              — VRAM + training rules
  MEMORY/SESSION_HANDOFF.md      — this file
```
