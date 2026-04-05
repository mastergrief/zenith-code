# Claw Code — Multi-Agent Harness + Specialist Distillation

Fork of [ultraworkers/claw-code](https://github.com/ultraworkers/claw-code) with a Python multi-agent harness and distillation pipeline for creating specialist 0.6-0.8B models from larger teachers.

## Architecture

Two systems coexist:
1. **Python agent harness** (`agents/`) — working terminal coding assistant with streaming, permissions, sessions, and context compaction
2. **Rust claw-code port** (`rust/`) — upstream claw-code, needs MSVC Build Tools to compile on Windows

## Python Agent Harness (`agents/`, ~1,400 lines across 12 files)

### Core Files
- `agent.py` — Base `Agent` class, Ollama chat with streaming + tool calling loop (max 10 rounds), auto-compaction
- `coordinator.py` — `Coordinator` delegates tasks via JSON `{"delegate": "name", "task": "..."}` protocol
- `swarm.py` — `Swarm` runs agents in parallel via `ThreadPoolExecutor`
- `tools.py` — 6 tools: `bash`, `read_file`, `write_file`, `edit_file`, `grep`, `list_files` (with safety limits + permission checks)
- `harness.py` — Terminal REPL with colored output, streaming tokens, and slash commands
- `specialist_coordinator.py` — `SpecialistCoordinator` routes to domain-specific fine-tuned models
- `example.py` — Demo scripts for Coordinator + Swarm patterns

### Production Features
- `permissions.py` — Tool deny-list blocking destructive bash commands and system path writes
- `compact.py` — Transcript compaction: auto-summarize old messages when approaching context limit
- `history.py` — Timestamped audit log for tool calls, responses, errors, commands
- `session.py` — Save/load agent conversations to `.claw_sessions/` as JSON

### Harness Commands
| Command | Action |
|---------|--------|
| `/help` | Show commands |
| `/agents` | List active agents |
| `/switch <name>` | Switch to specific agent |
| `/team` | Enable coordinator mode |
| `/solo` | Single agent mode |
| `/spawn <name> <role>` | Create new agent |
| `/reset` | Clear all histories |
| `/cd <path>` | Change working directory |
| `/model <name>` | Switch Ollama model |
| `/history` | Show session event log |
| `/save` | Save active agent's session |
| `/sessions` | List saved sessions |
| `/load <id>` | Load a saved session |
| `/distill status` | Show available specialist models |
| `/distill on/off` | Toggle specialist routing |
| `/exit` | Quit |

### Running the Harness
```bash
cd ~/Projects/claw-code
PYTHONUTF8=1 PYTHONPATH=. python3 agents/harness.py --model qwen9b-fast
```
WSL2 can access Ollama on Windows via `localhost:11434`.

## Distillation Pipeline (`agents/distill/`)

Two-stage approach to create specialist 0.6-0.8B models:
1. **Stage 1**: Train a reasoning base from Claude Opus reasoning data (2,875 filtered examples)
2. **Stage 2**: Fine-tune specialists on domain-specific data on top of the reasoning base

### Pipeline Scripts
- `config.py` — Domains, model names, QLoRA params, paths
- `generate.py` — Teacher (9B) generates JSONL training data. CLI: `python -m agents.distill.generate --domain python`
- `train_base.py` — Stage 1: reasoning base from Claude data. Uses Qwen 3.5 0.8B + QLoRA + `train_on_responses_only`
- `train.py` — Stage 2: domain specialist training. Auto-uses reasoning base if available. Uses `train_on_responses_only`
- `export.py` — Convert merged model → GGUF → Ollama Modelfile → `ollama create`
- `validate.py` — A/B compare specialist vs base using 9B as judge
- `fetch_datasets.py` — Download Claude reasoning datasets from HuggingFace (nohurry, TeichAI, Crownelius)
- `filter_reasoning.py` — Filter junk/hallucinations from reasoning data, merge with hand-written examples

### Specialist Domains
| Domain | Ollama Name | Focus |
|--------|-------------|-------|
| orchestrator | specialist-orchestrator | Task routing/classification |
| typescript | specialist-ts | React, Node, TS, Next.js |
| python | specialist-py | FastAPI, Django, pytest |
| rust | specialist-rust | Ownership, tokio, serde |
| devops | specialist-devops | Docker, K8s, Terraform |
| reviewer | specialist-reviewer | Security, bugs, perf |

### Training Data (`agents/distill/data/`, gitignored except hand-written files)
- `claude_reasoning.jsonl` — 2,875 filtered+curated reasoning examples (from HuggingFace + hand-written)
- `coding_reasoning_claude.jsonl` — 90 hand-written coding reasoning examples (committed)
- `orchestrator.jsonl` — 252 routing examples (130 original + 121 Claude-authored)
- `orchestrator_claude.jsonl` — 121 Claude-authored routing examples (committed)
- `python.jsonl` — 25 examples (9B-generated)
- `typescript.jsonl` — 39 examples (9B-generated)
- `rust.jsonl` — 53 examples (9B-generated)
- `seeds/` — 50-100 seed prompts per domain for generation

### Training Commands (WSL2)
```bash
# Stage 1: Reasoning base
PYTHONPATH=. python3 -m agents.distill.train_base

# Stage 2: Specialist
PYTHONPATH=. python3 -m agents.distill.train --domain orchestrator

# Filter + merge reasoning data
PYTHONPATH=. python3 -m agents.distill.filter_reasoning        # filter only
PYTHONPATH=. python3 -m agents.distill.filter_reasoning --merge # filter + merge hand-written
```

## Ollama Models

Available models (run `ollama list` to verify):
- `qwen9b-fast` — Qwen 3.5 9B, 2048 ctx, 100% GPU, tuned params
- `qwen4b-fast` — Qwen 3.5 4B, 8192 ctx, 100% GPU
- `qwen3.5:9b`, `qwen3.5:4b` — stock Qwen 3.5 models
- `qwen3:0.6b`, `qwen3:4b`, `qwen3:8b` — stock Qwen 3 models

Custom Modelfiles in `models/`:
- `Modelfile.qwen9b-fast` — optimized 9B
- `Modelfile.qwen4b-fast` — optimized 4B

## Hardware

- Laptop: Acer Nitro AN17-42
- GPU: NVIDIA RTX 4070 Laptop GPU (8 GB VRAM) — no Thunderbolt/eGPU support
- iGPU: AMD Radeon 780M (display only, no CUDA)
- RAM: 32 GB
- WSL2: Ubuntu 24.04 with GPU access
- Training: Unsloth 2026.4.2 + PyTorch 2.10.0

## Key Constraints

- **8 GB VRAM**: Qwen 3.5 9B only fits at 2K context (100% GPU). 4B fits at 8K. 0.8B fits at 32-64K.
- **Qwen 3.5 0.8B**: Large vocab (248K tokens) causes OOM with fused CE loss — use batch=1, seq_len=1024, packing=false
- **Git Bash + WSL2**: Paths with `Program Files (x86)` break bash `-c` commands. Use `-e` or write scripts to `/tmp/`
- **Ollama during training**: Stop all Ollama models before training to free VRAM (`ollama stop <model>`)
- **No eGPU**: Laptop has USB-C 3.2 but no Thunderbolt/USB4. Cloud GPUs are the path to bigger models.

## Branch

`feature/multi-agent-qwen` on `mastergrief/claw-code` (forked from `ultraworkers/claw-code`)
