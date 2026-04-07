# Session Handoff — 2026-04-06 (Session 4)

## Goal
Build enterprise-level command infrastructure, install Serena MCP, download and serve the 4B model via llama.cpp with CUDA, evaluate it, implement full harness parity with original claw-code, and update all documentation.

## Completed

### Enterprise Command Infrastructure (commit `2910d90`)
Built from scratch, modeled on zenith-fitness patterns (read DISCOVER-DEEP, DISCOVER-BACKEND, SPEC, VDD-FULL from zenith for reference).

**6 custom agents** in `.claude/agents/`:
- `explorer.md` (purple) — read-only codebase search, Serena-assisted
- `planner.md` (yellow) — synthesis gate, devil's advocate, cross-challenge
- `developer.md` (blue) — harness development specialist, knows all 13 files
- `trainer.md` (teal) — training data writing, JSONL format, quality standards
- `reviewer.md` (green) — code review against plan, read-only
- `harness-tester.md` (orange) — live harness testing, tool calling verification

**7 commands** in `.claude/commands/`:
- `DISCOVER.md` — single explorer, quick investigation
- `DISCOVER-DEEP.md` — 4-agent team (explorer + trainer + harness-tester + planner), cross-challenge, Solutions Matrix
- `SPEC.md` — post-discovery spec writing to Serena memory
- `VDD.md` — full discover → develop → validate lifecycle, 3 phases, self-healing
- `TRAIN-DATA.md` — training data generation/validation
- `EVAL.md` — model evaluation against 5 standard prompts
- `handoff.md` / `update.md` — built-in session management

**4 rules** in `.claude/rules/`:
- `orchestration.md` — dispatcher role, tool restrictions, synthesis gate enforcement, failure recovery
- `vdd.md` — single-team protocol, gates, cleanup
- `architecture.md` — full rewrite reflecting current codebase
- `training.md` — updated with 4B eval results

### Serena MCP Installation
- Configured in `~/.claude.json` under `projects["/mnt/c/Users/gabes/projects/claw-code"]`
- Uses `/home/gabe/serena-fork` with `uv run`
- NOT in `.claude.json` (project root) — that's for project settings, MCP goes in global config under project key
- Parity spec written to Serena memory: `specs/HARNESS_PARITY_SPEC/00_INDEX` through `03_TESTING`

### 4B Model Download and Serving
- Trained on Colab A100 (session 3), downloaded GGUF Q5_K_M from Colab browser download
- Located at `~/models/Qwen3.5-4B.Q5_K_M.gguf` (2.9GB)
- **llama.cpp rebuilt with CUDA**: `sudo apt install nvidia-cuda-toolkit` + `cmake -B build -DGGML_CUDA=ON`
- Serving: 64K context, Q4 KV cache, all 33 layers on GPU, ~6.3GB VRAM (pre-allocated)
- Previous CPU-only build had no GPU — key error: `warning: no usable GPU found`

### 4B Evaluation (commit after `9048406`)
**Without thinking**: 2/5 PASS (race condition + architecture), 3 PARTIAL. No `<think>` blocks produced.
**With `enable_thinking: true`**: 3/5 PASS (race condition, OOMKilled, architecture), 2 PARTIAL (React re-renders, file uploads). Genuine reasoning in `reasoning_content` field.
- **Key discovery**: Qwen 3.5 chat template uses `enable_thinking` parameter, NOT system prompt instructions
- Thinking mode fixes OOMKilled (was hallucinating tools without it)
- Weak spots: React/frontend (wrong useEffect claim), security (extension-only validation, no virus scan)

### llama.cpp Backend Integration (commit `9048406`)
- `agents/agent.py`: `detect_backend()` auto-detects llama.cpp (preferred) → Ollama fallback
- `_call_llamacpp()` / `_call_llamacpp_stream()`: OpenAI-compatible API, SSE streaming
- Thinking events: `thinking_start` → `thinking_token` → `thinking_end` → regular `token`
- `enable_thinking: true` always on by default
- `agents/compact.py`: `llamacpp: 65536` context limit
- `agents/harness.py`: `/backend` command, `--backend` CLI arg, thinking display

### `claw` Launcher (commit `9048406`)
- `bin/claw`: auto-starts llama.cpp if not running, waits for health (60s), launches harness
- Symlinked to `~/.local/bin/claw` — run from anywhere
- Env vars: `CLAW_MODEL`, `CLAW_PORT`, `CLAW_CTX`, `CLAW_LLAMA_SERVER`
- Bug fixed: `SCRIPT_DIR` used `readlink -f` to resolve symlink to repo root

### Harness Parity Implementation (commit `35287c4`)
Full parity spec implemented (5 steps from Serena memory `specs/HARNESS_PARITY_SPEC`):

1. **Bash validation pipeline** (`permissions.py`): `PermissionMode` enum (READ_ONLY/WORKSPACE_WRITE/FULL_ACCESS), `BashRisk` enum (SAFE/WRITE/DESTRUCTIVE/BLOCKED), `classify_bash()` with 33 safe commands, git subcommand awareness (9 safe, 8 write, 7 destructive), write redirect detection, path traversal detection
2. **User confirmation** (`harness.py`): `_confirm()` prompts `[y/N]` for WRITE/DESTRUCTIVE ops, handles mid-stream cleanup
3. **read_file windowing** (`tools.py`): `offset`/`limit` params, binary detection (NUL in first 8KB), edit context preview (lines around edit)
4. **Connection retry** (`agent.py`): `_request_with_retry()` with 3 attempts, 1s/2s/4s backoff
5. **Session + config + readline** (`harness.py`, `compact.py`, `config.py`): auto-save on exit, `/resume`, `.clawrc` loader, readline history at `~/.claw_history`

**Beyond spec** (also in `35287c4`):
- Effort mode: `--effort low/medium/max` controlling `max_tokens` and thinking depth
- llama.cpp tool calling fix: tools in payload, SSE delta assembly for `tool_calls`, `tool_call_id`
- Readline ANSI fix: `\001`/`\002` wrapping preventing prompt overwrite

### Output Repetition Detection (commit `d92f7bc`)
- `_is_repeating()`: streaming detection, breaks SSE loop if last 100 chars match earlier content
- `_dedup_blocks()`: post-generation cleanup, removes duplicate paragraphs and half-text mirrors
- Both only activate on output >40 chars to avoid false positives

### Documentation Update (commit `67deedd`)
- CLAUDE.md: 13 files / 2,033 lines, 20 commands, dual backend, all new features
- architecture.md: full rewrite
- training.md: 4B eval results, CUDA build, serving path

### DISCOVER-DEEP Gap Analysis
Ran full team-based discovery (explorer-original + explorer-harness + planner) mapping original claw-code vs our harness. Key finding: ~40% feature parity, gaps are asymmetric — strong on multi-agent/streaming, weak on safety/UX. Solutions Matrix produced with P0/P1/P2 prioritization.

## In Progress

### Uncommitted Changes
- `.claude/MEMORY/SESSION_HANDOFF.md` — this file
- `.claw_sessions/` — auto-saved session (untracked, gitignored)
- `.serena/` — Serena project config (untracked)
- `.env.local` — API keys (untracked, gitignored)

### `config.py` Not Wired Up
`agents/config.py` defines `load_config()` but it's never called — harness uses argparse directly. Should be integrated or removed.

## Next Steps

1. **Wire up `config.py`** — integrate `load_config()` into harness startup, or remove dead code
2. **Train specialists** — 4B reasoning base is ready, need:
   - Expand React/frontend training data (4B eval weak spot)
   - Expand security training data (4B eval weak spot)
   - Train orchestrator specialist on 252 routing examples
   - Train domain specialists (python, typescript, rust)
3. **Hot-swap implementation** — serve specialists via llama.cpp, swap on delegation
4. **Push to remote** — 5 unpushed commits on `feature/multi-agent-qwen`
5. **Test harness end-to-end** — run `claw`, test all 20 commands, verify tool calling + thinking + permissions + session management work together
6. **Integrate Serena tools into harness agents** — explorer/reviewer agents reference Serena but the harness's own agents don't use it

## Key Context

### What Failed
- **Colab browser download for large files**: 9GB tar.gz hung indefinitely. Google Drive mount failed from MCP (needs browser auth). Solution: download GGUF directly (3GB) via `files.download()`
- **llama.cpp CPU-only build**: first build had no CUDA toolkit installed. `warning: no usable GPU found`. Fix: `sudo apt install nvidia-cuda-toolkit` + rebuild with `-DGGML_CUDA=ON`
- **Thinking mode via system prompt**: tried `"Think step by step in <think> blocks"` and `/think` toggle — neither worked. Qwen 3.5 uses `enable_thinking` parameter in the API request, handled by the chat template
- **Custom agents not available as subagent_types**: `.claude/agents/*.md` files define custom agents but they can't be used as `subagent_type` in Agent tool calls. Must use built-in types (general-purpose, Explore, Plan) and inject the custom prompt manually.

### Architecture Decisions
- **Auto-detect backend, prefer llama.cpp**: `detect_backend()` checks `:8080/health` first. If llama.cpp is up, use it. Ollama is fallback only.
- **Suppress thinking tokens**: show "thinking..." indicator but don't stream reasoning text. Cleaner output, less noise for user.
- **Pre-allocated KV cache**: 6.3GB VRAM is the ceiling regardless of conversation length. No surprises mid-session.
- **Effort mode controls max_tokens**: low=1024, medium=2048, max=8192. Also prepends effort-specific system prompt prefix.
- **Frequency penalty 0.5**: llama.cpp backend uses this to reduce repetition alongside the dedup functions.
- **Serena MCP config location**: goes in `~/.claude.json` under `projects[path].mcpServers`, NOT in the repo's `.claude.json`. The repo file is for project settings (permissions, etc).

### MCP Servers Configured
- **Serena** (`~/.claude.json` project key): `uv run --directory /home/gabe/serena-fork serena start-mcp-server --context ide-assistant --project /mnt/c/Users/gabes/projects/claw-code`
- **Chrome DevTools** (`~/.claude.json` global): custom fork at `~/chrome-devtools-mcp-fork`
- **RunPod** (`~/.claude.json` global): pod management
- **Colab** (`~/.claude.json` global): notebook editing (limited — can't do browser auth or file uploads)

### Claude Code Built-in Agent Prompts (extracted from binary v2.1.92)
Extracted all 5 built-in agents from the ELF binary:
1. **general-purpose**: model inherited, all tools, broad research
2. **statusline-setup**: model sonnet, Read+Edit only, status line config
3. **claude-code-guide**: model haiku, WebFetch+WebSearch, docs lookup
4. **Explore**: model haiku, read-only, no CLAUDE.md, feature-flagged (`tengu_amber_stoat`)
5. **Plan**: model inherited, read-only, no CLAUDE.md, feature-flagged

## Files

```
agents/                          — 13 files, ~2,033 lines
  agent.py (449)                 — Agent class, dual backend, thinking, retry, effort, dedup
  tools.py (312)                 — 6 tools with windowing, binary detect, edit preview
  harness.py (536)               — REPL, 20 commands, confirmation, readline, auto-save
  permissions.py (133)           — 3 permission modes, 4-level bash classification
  compact.py (215)               — context compaction, per-model limits, summary compression
  config.py (30)                 — .clawrc loader (NOT YET WIRED UP)
  coordinator.py (76)            — JSON delegation protocol
  swarm.py (55)                  — parallel agent execution
  specialist_coordinator.py (76) — domain-specific routing
  session.py (51)                — save/load/list sessions
  history.py (34)                — timestamped event log
  example.py (43)                — demo scripts
  __init__.py (23)               — re-exports

agents/distill/                  — 10 files, 1,770 lines
  data/                          — 9 JSONL files, 6,070 lines total
    coding_reasoning_claude.jsonl (488)  — hand-written (committed)
    claude_reasoning.jsonl (1,320)       — merged training data (gitignored)
    orchestrator.jsonl (252)             — routing examples

models/                          — 3 Modelfiles (qwen9b-fast, qwen4b-fast, reasoning-base)
bin/claw                         — launcher script, auto-starts llama.cpp
~/models/Qwen3.5-4B.Q5_K_M.gguf — 2.9GB fine-tuned 4B (serving via llama.cpp)
~/llama.cpp/build/bin/           — CUDA build (llama-server, llama-quantize)

.claude/agents/                  — 6 custom agents
.claude/commands/                — 8 commands (DISCOVER, DISCOVER-DEEP, SPEC, VDD, TRAIN-DATA, EVAL, handoff, update)
.claude/rules/                   — 4 rules (architecture, orchestration, training, vdd)
.serena/memories/specs/          — HARNESS_PARITY_SPEC (4 parts: INDEX, ARCHITECTURE, IMPLEMENTATION, TESTING)
```
