---
name: Training status
description: Current state of reasoning base and specialist training runs
type: project
---

Reasoning base training on Qwen 3.5 0.8B did not complete (session ended 2026-04-04). Needs to be restarted from scratch.

**Why:** Session ran out of time during the ~1.5hr training run.

**How to apply:** On next session, restart Stage 1 reasoning base training before proceeding to Stage 2 specialists. Remember to stop Ollama first (`ollama stop` all models) to free VRAM.
