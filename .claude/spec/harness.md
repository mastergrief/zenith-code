# Python Agent Harness — Commands and Launch

User-facing surface of the `agents/` harness (`zenith` CLI). Internals,
tool definitions, and streaming invariants live in
`.claude/spec/architecture.md` §"Agent System" + §"File Organization".

## Harness Commands

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

## Running the Harness

```bash
# Preferred: auto-starts llama.cpp with default ~/models/gemma-4-E4B-it-tq4-aligned.gguf at 512K
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

`bin/zenith` launcher: auto-starts llama.cpp if not running, waits for
health, passes `--backend llamacpp`. Default `ZENITH_CTX=524288` (512K).
Configurable via `ZENITH_MODEL`, `ZENITH_PORT`, `ZENITH_CTX`,
`ZENITH_LLAMA_SERVER` env vars, plus the `--gguf PATH` CLI flag (must
be first arg). The stdin pipe form works in any environment (TTY or
non-TTY) because the harness uses plain `input()`; redirect output to
a file to keep model token spam out of your terminal/context.
`bin/zenith` does NOT `cd` into the repo root before exec'ing the
harness — this keeps `.zenithrc` lookup and CLAUDE.md auto-discovery
honoring the user's actual cwd.

## Related rules

- `architecture.md` §"Agent System" — streaming, tool-call loop, permissions, compaction invariants
- `environment.md` §"Serving Architecture" — llama.cpp flags + GGUF paths the launcher expects
- `CLAUDE.md` — top-level index
