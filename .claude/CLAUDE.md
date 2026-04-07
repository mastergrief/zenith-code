# Zenith Code — Multi-Agent Harness + Specialist Distillation

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
1. **Python agent harness** (`agents/`) — terminal coding assistant with dual backend (Ollama + llama.cpp), 3-level permissions, thinking mode, sessions, compaction, effort control, and llama.cpp hot-swap
2. **Rust claw-code port** (`rust/`) — upstream claw-code, 9 crates, separate build system

Serving: either Qwen 3.5 4B or Gemma 4 E4B via llama.cpp at **256K context** (`ZENITH_CTX=262144` default) with Q4 KV cache (~6.7–7.3 GB VRAM). Harness auto-computes compaction threshold as `min(per-GGUF NIAH-validated limit, int(ctx_size * 0.85))`. Hot-swap between bases is implemented via `agents/model_swap.py`.

## Python Agent Harness (`agents/`, ~2,870 lines across 14 files)

### Core Files
- `agent.py` (520 lines) — `Agent` class with dual backend (Ollama + llama.cpp), streaming, thinking mode, tool calling loop (max 10 rounds), connection retry with backoff, system prompt builder, effort mode, output dedup, `detect_llamacpp_model()` helper that queries `/props` for the loaded GGUF path
- `coordinator.py` (76 lines) — `Coordinator` delegates tasks via JSON `{"delegate": "name", "task": "..."}` protocol
- `swarm.py` (55 lines) — `Swarm` runs agents in parallel via `ThreadPoolExecutor`
- `tools.py` (312 lines) — 6 tools: `bash`, `read_file`, `write_file`, `edit_file`, `grep`, `list_files`. read_file supports offset/limit windowing + binary detection. edit_file shows context preview.
- `harness.py` (664 lines) — Terminal REPL with colored output, streaming tokens, thinking display, user confirmation prompts, readline history, 17 slash commands including `/swap` for hot-swapping GGUFs; caches `_loaded_llamacpp_model` via `/props` query at init and recomputes compaction threshold on `/swap` or `/backend`
- `specialist_coordinator.py` (245 lines) — `SpecialistCoordinator` auto-selects between hot-swap mode (llama.cpp + specialist GGUFs on disk via `discover_specialist_models()`) and Ollama multi-model mode; falls back to base model if neither available
- `model_swap.py` (407 lines) — `LlamaServerManager` orchestrates llama-server subprocess lifecycle (start/stop/swap), adopts externally-started servers via `/props`, finds the listening PID via `/proc/net/tcp` for servers it doesn't own. `discover_specialist_models()` scans `~/models/` for domain-named GGUFs
- `example.py` (43 lines) — Demo scripts for Coordinator + Swarm patterns

### Production Features
- `permissions.py` (133 lines) — 3 permission modes (READ_ONLY/WORKSPACE_WRITE/FULL_ACCESS), 4-level bash classification (SAFE/WRITE/DESTRUCTIVE/BLOCKED), git subcommand awareness, write redirect detection, system path blocking
- `compact.py` (244 lines) — Auto-compaction with per-GGUF context limits (Gemma 4 E4B 200K, Qwen 3.5 4B 130K, llama.cpp fallback 65K — NIAH-validated, see `.claude/MEMORY/evals/2026-04-07_summary_needle_comparison.md`), summary compression, env var override (`ZENITH_AUTO_COMPACT_TOKENS`)
- `config.py` (61 lines) — Config loader for `.zenithrc`/`zenith.json` with explicit `ZENITH_*` env var registry (`ZENITH_MODEL`, `ZENITH_BACKEND`, `ZENITH_CTX`, `ZENITH_AUTO_COMPACT_TOKENS`, `ZENITH_PERMISSION_MODE`, `ZENITH_EFFORT`); `ctx_size` default is 262144
- `history.py` (34 lines) — Timestamped audit log for tool calls, responses, errors, commands
- `session.py` (51 lines) — Save/load agent conversations to `.zenith_sessions/` as JSON, auto-save on exit

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
| `/backend [ollama\|llamacpp]` | Show/switch backend (re-detects loaded GGUF + recomputes compaction threshold) |
| `/swap [name]` | Hot-swap the loaded GGUF via `LlamaServerManager` (substring match against `~/models/*.gguf`); no arg shows current + lists available |
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
# Preferred: auto-starts llama.cpp with default ~/models/Qwen3.5-4B.Q5_K_M.gguf at 256K
zenith

# Pick a specific GGUF at launch time (new --gguf launcher flag)
zenith --gguf ~/models/gemma-4-E4B-it-Q5_K_M.gguf

# Or set the env var once in your shell rc
ZENITH_MODEL=~/models/gemma-4-E4B-it-Q5_K_M.gguf zenith

# Hot-swap from inside an active session (no restart needed)
> /swap gemma       # substring match in ~/models/*.gguf
> /swap qwen        # swap back

# Manual: specify backend and model (bypasses bin/zenith)
PYTHONUTF8=1 PYTHONPATH=. python3 agents/harness.py --backend llamacpp
PYTHONUTF8=1 PYTHONPATH=. python3 agents/harness.py --model qwen3.5:4b --backend ollama

# Programmatic / smoke-test invocation — pipe prompts via stdin, capture to log
printf "what is 2+2?\n/exit\n" | zenith --effort max > /tmp/zenith.log 2>&1

# Override context (smaller if VRAM constrained, or for faster cold start)
ZENITH_CTX=65536 zenith

# CLI flags: --model, --backend, --ctx-size, --effort, --resume, --permission-mode, --cd
```
`bin/zenith` launcher: auto-starts llama.cpp if not running, waits for health, passes `--backend llamacpp`. Default `ZENITH_CTX=262144` (256K). Configurable via `ZENITH_MODEL`, `ZENITH_PORT`, `ZENITH_CTX`, `ZENITH_LLAMA_SERVER` env vars, plus the `--gguf PATH` CLI flag (must be first arg). The stdin pipe form works in any environment (TTY or non-TTY) because the harness uses plain `input()`; redirect output to a file to keep model token spam out of your terminal/context. `bin/zenith` does NOT `cd` into the repo root before exec'ing the harness — this keeps `.zenithrc` lookup and CLAUDE.md auto-discovery honoring the user's actual cwd.

## Distillation Pipeline (`agents/distill/`, 10 Python files)

### Current Status
- **0.8B reasoning base**: trained (3 epochs, loss 1.106), eval: format learned but substance wrong — model too small
- **4B reasoning base (Qwen)**: trained on Colab A100 (1,339 examples after 2026-04-07 React/security expansion, 3 epochs), exported to GGUF Q5_K_M, serving via llama.cpp. Earlier eval: 3/5 PASS with thinking enabled (race condition, OOMKilled, architecture pass; React and security partial). **Subsequent 5-prompt A/B vs stock Gemma 4 E4B (2026-04-07) scored fine-tuned Qwen 0/5 — the React and security failures were correctness bugs (hallucinated Node.js `beforeOOM` API, broken Postgres `FOR UPDATE SKIP LOCKED` queue, regex-on-hostname SSRF check). See `.claude/MEMORY/evals/2026-04-07_qwen4b_vs_gemma4_e4b.md`.**
- **Gemma 4 E4B (stock)**: validated as alternative base 2026-04-07. Beats fine-tuned Qwen 5/0 on the same coding eval without any fine-tuning. NIAH effective context: 200K (vs Qwen's 130K). Multimodal (vision projector available). GGUF on disk at `~/models/gemma-4-E4B-it-Q5_K_M.gguf` (5.48 GB). Not yet fine-tuned on the distillation dataset.
- **Hot-swap infrastructure**: IMPLEMENTED (`agents/model_swap.py` + `SpecialistCoordinator` hot-swap mode). `LlamaServerManager` handles kill+restart swap cycles (~5–15s depending on model size). Swap cost on a warm page cache is mostly PCIe transfer time.
- **Specialists**: not yet trained — both base models are ready, specialist GGUFs don't exist on disk yet. When they do, `SpecialistCoordinator` auto-detects and switches to hot-swap mode.

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
- `claude_reasoning.jsonl` — 1,339 merged examples (832 filtered HuggingFace + 507 hand-written)
- `coding_reasoning_claude.jsonl` — 507 hand-written coding reasoning examples (committed). +19 added 2026-04-07 via `scripts/generate_react_security_examples.py` (11 React + 8 security, targeting gaps in earlier Qwen eval)
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

**llama.cpp (primary)** — either 4B base via full GPU:
- Default model: `~/models/Qwen3.5-4B.Q5_K_M.gguf` (2.9 GB, fine-tuned)
- Alternative model: `~/models/gemma-4-E4B-it-Q5_K_M.gguf` (5.48 GB, stock; selectable via `--gguf` or `ZENITH_MODEL`)
- Context: **256K tokens** with Q4 KV cache (~6.7 GB for Gemma E4B, ~7.3 GB for Qwen 4B — sliding-window attention on Gemma makes its KV cache dramatically smaller). Pre-allocated at startup.
- Thinking: `enable_thinking: true` by default, reasoning in separate `reasoning_content` field
- Launch: `zenith` command auto-starts, or manually: `llama-server -m model.gguf --ctx-size 262144 --parallel 1 --cache-type-k q4_0 --cache-type-v q4_0 -ngl 999 --port 8080`
- `--parallel 1` is required — without it, llama-server splits `ctx_size` across 4 default slots, so each slot only gets `ctx_size / 4`
- **Hot-swap: IMPLEMENTED** via `agents/model_swap.py:LlamaServerManager`. Swap cycles are ~5–15s depending on disk page-cache warmth. `/swap` command uses it directly; `SpecialistCoordinator` uses it for domain routing when specialist GGUFs exist on disk.
- Both Qwen 3.5 4B and Gemma 4 E4B are trained at 256K native context (earlier notes had Qwen at 32K — that was wrong).

**Ollama (fallback)** — stock models, quick testing:
- Pulled models (verified via `curl -s localhost:11434/api/tags`): `qwen3.5:4b`, `qwen3.5:9b`, `qwen3:0.6b`, `qwen3:4b`, `qwen3:8b`, plus custom Modelfiles `qwen4b-fast:latest`, `qwen9b-fast:latest`, `reasoning-base:latest`
- Custom Modelfiles in `models/`: `Modelfile.qwen9b-fast` (2048 ctx), `Modelfile.qwen4b-fast` (8192 ctx), `Modelfile.reasoning-base` (32K ctx)
- Kill Windows Ollama to free VRAM: `taskkill /IM ollama.exe /F`

## Local Tools

- **llama.cpp**: built at `~/llama.cpp/build/bin/` with CUDA support (RTX 4070)
  - `llama-quantize` — convert FP16 safetensors to GGUF quantized formats
  - `llama-server` — serve models with OpenAI-compatible API, KV cache quantization, thinking mode
  - **Local patch** at `tools/server/server-context.cpp:763-766` — one-line edit comments out `n_ctx_slot = n_ctx_train` to remove the per-slot training-context cap, enabling `--ctx-size` past `n_ctx_train` for extrapolation testing. Not upstreamed. Re-apply after any `git pull` on the llama.cpp source. See `.claude/MEMORY/evals/2026-04-07_summary_needle_comparison.md` for the context.
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

- **8 GB VRAM**: both 4B-class bases fit at **256K context** with Q4 KV cache (llama.cpp) — Qwen 3.5 4B Q5 uses ~7.3 GB, Gemma 4 E4B Q5 uses ~6.7 GB (sliding-window attention makes Gemma's KV cache dramatically smaller at long context). 9B Q4 fits at 2K (Ollama). 0.8B FP16 fits at 32K.
- **Both 4B bases are trained at 256K context** (Qwen 3.5 4B and Gemma 4 E4B, verified via GGUF metadata `n_ctx_train=262144`). Earlier notes had Qwen at 32K — that was wrong.
- **Qwen 3.5 4B QLoRA**: 248K vocab CE loss OOMs on anything under 40GB VRAM. Must use cloud GPU (Colab A100).
- **Qwen 3.5 0.8B QLoRA**: fits locally at batch=1, seq_len=1024, packing=false
- **WSL2 + Windows Ollama**: both can run Ollama. Don't run both simultaneously — VRAM conflict. Prefer WSL-native.
- **No eGPU**: laptop has USB-C 3.2 but no Thunderbolt/USB4. Cloud GPUs for 4B+ training.

## Needle-in-Haystack Validation (2026-04-07)

Effective context for both base models was measured via single-needle, multi-needle, and distractor NIAH tests at 4K–220K haystack sizes. Full reports in `.claude/MEMORY/evals/2026-04-07_*_needle_256k_*.md`, summary in `2026-04-07_summary_needle_comparison.md`.

| Test type | Gemma 4 E4B | Qwen 3.5 4B |
|---|---:|---:|
| Single-needle (21 prompts, 4K–220K) | **21/21** | **21/21** |
| Multi-needle (7 prompts, 5 needles each) | **6/7** (fails 220K 4/5) | **5/7** (fails 180K 4/5, 220K 3/5 + hallucination) |
| Distractor (7 prompts, 4 decoys each) | **7/7** | **5/7** (U-shape dip at 64K/100K — picks wrong decoy) |
| **Total** | **34/35** | **31/35** |

Key findings:
- Both models handle single-needle cleanly through 220K (neither degrades on the easy test).
- Gemma's sliding-window attention protects against the "lost in the middle" failure mode Qwen exhibits at 64K/100K on the distractor test.
- Qwen's worst failure is **hallucination under pressure** (returns `14223-AZURE-MARTEN` when expected is `14223-CRIMSON-EAGLE` — correct number, invented suffix). Gemma's failures are silent omissions, which is a safer mode.
- `agents/compact.py:MODEL_CONTEXT_LIMITS` is the authoritative source of NIAH-validated values (Gemma 200K = 10% safety below the first failure point at 220K; Qwen 130K = safely below the first failure point at 180K).
- **The 256K tests required a local llama.cpp patch** to remove the per-slot training-context cap — see Local Tools above.

## Branch

`feature/multi-agent-qwen` on `mastergrief/zenith-code` (forked from `ultraworkers/claw-code`; renamed from `mastergrief/claw-code` 2026-04-07)
