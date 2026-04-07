# Claw Code — Multi-Agent Harness + Specialist Distillation

Fork of [ultraworkers/claw-code](https://github.com/ultraworkers/claw-code) with a Python multi-agent harness and distillation pipeline for fine-tuning Qwen 3.5 4B specialists from curated training data.

## Orchestrator Role & Tool Restrictions
Full spec: `.claude/rules/orchestration.md`
- **You are a DISPATCHER, not a worker** — spawn agents for all investigation, editing, and testing
- Pre-tool checkpoint: Investigation → `explorer`, Code modification → `developer`, Harness testing → `harness-tester`, Training data → `trainer`
- If 3+ sequential subagents needed → use team instead. Teams: `shutdown_request` → wait → `TeamDelete()`

## VDD Protocol (Validation Driven Development)
Full spec: `.claude/rules/vdd.md`

- `/VDD` — single team `vdd`, 5-6 teammates across 3 phases, full discovery → develop → validate
- One `TeamCreate` at start, one `TeamDelete` at end — teammates spawned as phases progress
- Cross-phase consultation via DMs (explorer available throughout, developer stays alive for self-healing)
- Self-healing: harness-tester ↔ developer direct messaging, no team respawn
- Gates: VERIFY-ON-DISK (`git diff`), PRE-VALIDATE (imports + cargo check)
- Plan approval (`mode: "plan"`) for MEDIUM/LARGE developer steps
- Three-layer verification: code review + harness integration test + static analysis
- Graceful shutdown: `shutdown_request` all → wait ~5s → `TeamDelete()`

---

## BLOCKING VIOLATIONS
| Violation | Why It's Blocking |
|-----------|-------------------|
| Using `Edit`/`Write` directly for code | Pollutes context, use `developer` agent |
| Multi-file `Grep`/`Read` investigation | Pollutes context, use `explorer` agent |
| Direct harness testing | Pollutes context, use `harness-tester` agent |
| Direct training data generation | Pollutes context, use `trainer` agent |
| Self-investigation after test failure | VDD violation — spawn `explorer` with `Task` tool |
| Skipping DISCOVERY phase | Issues discovered too late |
| Running subagents in background | Loses results, always foreground |
| Skipping static checks after mutations | Errors compound |
| Accepting planner deliverable without `TaskList` verification | Planner may assemble matrix before harness-tester finishes. Run `TaskList` → ALL tasks `completed` before accepting. |

**IMPORTANT — SUBAGENT DIRECTIVE**
- PARALLEL subagents: ALL `Agent` invocations MUST be in a single `<function_calls>` block with ZERO text between `</invoke>` and the next `<invoke>`. Any text output between calls forces a round-trip, serializing them.
- Plan ALL agent prompts BEFORE emitting the function_calls block — never start writing tool calls until every prompt is ready.
- Model `Opus` used for all subagents at all times.

---

## Config `.claude/` Editing Directive

When editing `.claude/` configs (agents, CLAUDE.md, commands, rules etc):
- **Preserve structure** → Match existing formatting (bullets, sections, headers)
- **Match tone** → Imperative, terse, no fluff (e.g., "Do X" not "You should consider doing X")
- **Add value** → Every word must serve purpose (examples only if essential)
- **No verbosity** → 500 lines is hard limit, 250-500 is sweet spot. Be concise without losing context.
- **Maintain style & patterns** → Use existing conventions
- **No duplication** → Don't repeat information already present elsewhere
- **Verify integration** → New content must flow naturally with surrounding text

---

## Architecture

Two systems coexist:
1. **Python agent harness** (`agents/`) — terminal coding assistant with dual backend (Ollama + llama.cpp), 3-level permissions, thinking mode, sessions, compaction, and effort control
2. **Rust claw-code port** (`rust/`) — upstream claw-code, 9 crates, separate build system

Serving: 4B reasoning base via llama.cpp at 64K context with Q4 KV cache (~6.3GB VRAM). Specialists planned as hot-swap on same GPU.

## Python Agent Harness (`agents/`, ~2,000 lines across 13 files)

### Core Files
- `agent.py` (449 lines) — `Agent` class with dual backend (Ollama + llama.cpp), streaming, thinking mode, tool calling loop (max 10 rounds), connection retry with backoff, system prompt builder, effort mode, output dedup
- `coordinator.py` (76 lines) — `Coordinator` delegates tasks via JSON `{"delegate": "name", "task": "..."}` protocol
- `swarm.py` (55 lines) — `Swarm` runs agents in parallel via `ThreadPoolExecutor`
- `tools.py` (312 lines) — 6 tools: `bash`, `read_file`, `write_file`, `edit_file`, `grep`, `list_files`. read_file supports offset/limit windowing + binary detection. edit_file shows context preview.
- `harness.py` (536 lines) — Terminal REPL with colored output, streaming tokens, thinking display, user confirmation prompts, readline history, 20 slash commands
- `specialist_coordinator.py` (76 lines) — `SpecialistCoordinator` routes to domain-specific fine-tuned models
- `example.py` (43 lines) — Demo scripts for Coordinator + Swarm patterns

### Production Features
- `permissions.py` (133 lines) — 3 permission modes (READ_ONLY/WORKSPACE_WRITE/FULL_ACCESS), 4-level bash classification (SAFE/WRITE/DESTRUCTIVE/BLOCKED), git subcommand awareness, write redirect detection, system path blocking
- `compact.py` (215 lines) — Auto-compaction with per-model context limits (64K for llama.cpp), summary compression, env var override (`CLAW_AUTO_COMPACT_TOKENS`)
- `config.py` (30 lines) — Config loader for `.clawrc`/`claw.json` with `CLAW_*` env var overrides
- `history.py` (34 lines) — Timestamped audit log for tool calls, responses, errors, commands
- `session.py` (51 lines) — Save/load agent conversations to `.claw_sessions/` as JSON, auto-save on exit

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
| `/model <name>` | Switch model for all agents |
| `/backend [ollama\|llamacpp]` | Show/switch backend |
| `/effort [low\|medium\|max]` | Show/set reasoning effort |
| `/history` | Show session event log |
| `/save` | Save active agent's session |
| `/sessions` | List saved sessions |
| `/load <id>` | Load a saved session |
| `/resume` | Load most recent session |
| `/distill status` | Show available specialist models |
| `/distill on/off` | Toggle specialist routing |
| `/exit` | Quit |

### Running the Harness
```bash
# Preferred: auto-starts llama.cpp, serves 4B at 64K context
claw

# Manual: specify backend and model
PYTHONUTF8=1 PYTHONPATH=. python3 agents/harness.py --backend llamacpp
PYTHONUTF8=1 PYTHONPATH=. python3 agents/harness.py --model qwen3.5:0.8b --backend ollama

# Programmatic / smoke-test invocation — pipe prompts via stdin, capture to log
printf "what is 2+2?\n/exit\n" | claw --effort max > /tmp/claw.log 2>&1

# CLI flags: --model, --backend, --effort, --resume, --permission-mode, --cd
```
`bin/claw` launcher: auto-starts llama.cpp if not running, waits for health, passes `--backend llamacpp`. Configurable via `CLAW_MODEL`, `CLAW_PORT`, `CLAW_CTX` env vars. The stdin pipe form works in any environment (TTY or non-TTY) because the harness uses plain `input()`; redirect output to a file to keep model token spam out of your terminal/context.

## Distillation Pipeline (`agents/distill/`, 10 Python files)

### Current Status
- **0.8B reasoning base**: trained (3 epochs, loss 1.106), eval: format learned but substance wrong — model too small
- **4B reasoning base**: trained on Colab A100 (1,320 examples, 3 epochs), exported to GGUF Q5_K_M, serving via llama.cpp at 64K context. Eval: 3/5 PASS with thinking enabled (race condition, OOMKilled, architecture pass; React and security partial)
- **Specialists**: not yet trained — 4B base ready, need targeted training data expansion (React/frontend, security)

### Pipeline Scripts
- `config.py` — Domains, model names, QLoRA params, paths
- `generate.py` — Teacher (9B) generates JSONL training data. CLI: `python -m agents.distill.generate --domain python`
- `train_base.py` — Stage 1: 0.8B reasoning base (local). Qwen 3.5 0.8B + QLoRA + `train_on_responses_only`, 3 epochs
- `train_4b_cloud.py` — Stage 1: 4B reasoning base (cloud). Qwen 3.5 4B + QLoRA, requires 40GB+ VRAM (A100)
- `train_4b_colab.ipynb` — Colab notebook for 4B training
- `train.py` — Stage 2: domain specialist training. Auto-uses reasoning base if available
- `export.py` — Convert merged model → GGUF → Ollama Modelfile → `ollama create`
- `validate.py` — A/B compare specialist vs base using 9B as judge
- `fetch_datasets.py` — Download Claude reasoning datasets from HuggingFace (nohurry, TeichAI, Crownelius)
- `filter_reasoning.py` — Tiered keyword filtering + dedup + merge hand-written with HuggingFace data

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
- `claude_reasoning.jsonl` — 1,320 merged examples (832 filtered HuggingFace + 488 hand-written)
- `coding_reasoning_claude.jsonl` — 488 hand-written coding reasoning examples (committed)
- `claude_reasoning_filtered.jsonl` — 832 filtered HuggingFace examples (intermediate)
- `claude_reasoning_prefilter.jsonl` — backup of pre-filter merged data
- `orchestrator.jsonl` — 252 routing examples (130 original + 121 Claude-authored)
- `orchestrator_claude.jsonl` — 121 Claude-authored routing examples (committed)
- `python.jsonl` — 25 examples (9B-generated)
- `typescript.jsonl` — 39 examples (9B-generated)
- `rust.jsonl` — 53 examples (9B-generated)

### Training Commands
```bash
# Stage 1: 0.8B reasoning base (local, 8GB VRAM)
PYTHONPATH=. python3 -m agents.distill.train_base

# Stage 1: 4B reasoning base (cloud, 40GB+ VRAM)
# Use train_4b_colab.ipynb on Google Colab with A100
# Or: python3 train_4b_cloud.py on RunPod/Lambda

# Stage 2: Specialist (on top of reasoning base)
PYTHONPATH=. python3 -m agents.distill.train --domain orchestrator

# Filter + merge reasoning data
PYTHONPATH=. python3 -m agents.distill.filter_reasoning        # filter only
PYTHONPATH=. python3 -m agents.distill.filter_reasoning --merge # filter + merge hand-written
```

## Training Philosophy

**Data quality > data quantity > model size > training tricks.**

1. Write high-quality examples — one good Claude-authored example teaches more than ten 9B-generated ones
2. Train on responses only — don't waste gradients learning to predict prompts
3. Match domain to task — coding reasoning data for coding models, routing data for routing models
4. Filter aggressively — removing bad data improves results more than adding mediocre data
5. 3 epochs on curated data — diverse enough (1,320 unique topics) to avoid memorization; 1 epoch underfits

## Serving Architecture

**llama.cpp (primary)** — 4B reasoning base, full GPU:
- Model: `~/models/Qwen3.5-4B.Q5_K_M.gguf` (2.9GB, fine-tuned)
- Context: 64K tokens with Q4 KV cache (~6.3GB total VRAM, pre-allocated)
- Thinking: `enable_thinking: true` by default, reasoning in separate `reasoning_content` field
- Launch: `claw` command auto-starts, or manually: `llama-server -m model.gguf --ctx-size 65536 --cache-type-k q4_0 --cache-type-v q4_0 -ngl 999 --port 8080`
- Specialists: planned hot-swap on same GPU (5-10s swap time)

**Ollama (fallback)** — stock models, quick testing:
- Pulled models: `qwen3.5:0.8b`, `qwen3.5:4b`, `qwen3.5:9b`, `qwen3:0.6b`, `qwen3:4b`, `qwen3:8b`, plus custom Modelfiles `qwen4b-fast`, `qwen9b-fast`, `reasoning-base` (verify current with `curl -s localhost:11434/api/tags`)
- Custom Modelfiles in `models/`: `Modelfile.qwen9b-fast` (2048 ctx), `Modelfile.qwen4b-fast` (8192 ctx), `Modelfile.reasoning-base` (32K ctx)
- Kill Windows Ollama to free VRAM: `taskkill /IM ollama.exe /F`

## Local Tools

- **llama.cpp**: built at `~/llama.cpp/build/bin/` with CUDA support (RTX 4070)
  - `llama-quantize` — convert FP16 safetensors to GGUF quantized formats
  - `llama-server` — serve models with OpenAI-compatible API, KV cache quantization, thinking mode
- **Unsloth**: 2026.4.2 + PyTorch 2.10.0+cu128 (for local 0.8B training)
- **Serena MCP**: installed at `/home/gabe/serena-fork`, configured for this project

## Cloud Accounts

- **RunPod**: API key in `.env.local`, `runpodctl` installed, MCP server configured in `~/.claude.json`
- **Google Colab Pro**: 100 compute units, Colab MCP configured in `~/.claude.json`

## Hardware

- Laptop: Acer Nitro AN17-42
- GPU: NVIDIA RTX 4070 Laptop GPU (8 GB VRAM) — no Thunderbolt/eGPU support
- iGPU: AMD Radeon 780M (display only, no CUDA)
- RAM: 32 GB DDR5 5600MHz
- WSL2: Ubuntu 24.04 with GPU access
- Training (local): Unsloth 2026.4.2 + PyTorch 2.10.0+cu128
- Training (cloud): Google Colab Pro A100 (40GB)

## Key Constraints

- **8 GB VRAM**: 4B Q5 fits at 64K context with Q4 KV cache (llama.cpp). 9B Q4 fits at 2K (Ollama). 0.8B FP16 fits at 32K.
- **Qwen 3.5 4B QLoRA**: 248K vocab CE loss OOMs on anything under 40GB VRAM. Must use cloud GPU (Colab A100).
- **Qwen 3.5 0.8B QLoRA**: fits locally at batch=1, seq_len=1024, packing=false
- **WSL2 + Windows Ollama**: both can run Ollama. Don't run both simultaneously — VRAM conflict. Prefer WSL-native.
- **No eGPU**: laptop has USB-C 3.2 but no Thunderbolt/USB4. Cloud GPUs for 4B+ training.

## Branch

`feature/multi-agent-qwen` on `mastergrief/claw-code` (forked from `ultraworkers/claw-code`)
