# VDD Protocol — Quick Reference

**One team, phased spawning, zero team churn.**
- One `TeamCreate` at start, one `TeamDelete` at end
- Teammates spawned as phases progress, stay alive for cross-phase consultation
- Self-healing via direct messaging (harness-tester ↔ developer), no team respawn
- Task board spans entire lifecycle

## VDD Flow

```
TeamCreate("vdd"), Phase 1 tasks (6 with deps)
Spawn explorer + trainer + planner

Phase 1a: parallel investigation (explorer + trainer) → both report to planner
Phase 1b: planner cross-challenges with full visibility (devil's advocate)
Phase 1c: investigators respond to planner's targeted challenges
Phase 1d: planner synthesizes implementation plan → shutdown planner + trainer, keep explorer
Phase 2: spawn developer (mode: plan for MEDIUM/LARGE) → DMs explorer → implements → keep alive
Phase 3: spawn reviewer + harness-tester
  reviewer DMs explorer (regression checks) + harness-tester (additional scenarios)
  harness-tester DMs developer on failure (self-healing)
  All PASS → shutdown all → TeamDelete
```

## Gates

- **VERIFY-ON-DISK**: `git diff --stat` after each developer step (max 2 retries)
- **PRE-VALIDATE**: Python imports pass + Rust compiles (if changed) before Phase 3
- **Static checks**: `cargo clippy`, `cargo fmt`, `python3 -c "import agents.harness"` blocking between steps

## Cleanup

`shutdown_request` all → wait ~5s for idle → `TeamDelete()` → retry once if fails

## Rules

- Always run discovery — never skip, even with prior plans or specs
- Self-healing via DMs, not team respawn. Developer stays alive through validation.
- Plan approval (`mode: "plan"`) for MEDIUM/LARGE developer steps
- Static checks blocking between every developer step
- Three-layer verification: code review + harness integration test + static analysis
- Graceful shutdown: wait ~5s before TeamDelete
- Never run subagents in background
- ALL teammate spawns in single `<function_calls>` block
- Max 2 self-healing cycles per issue. Escalate to explorer, then to user.
- **Step 0 output injected into every spawn prompt** — teammates don't inherit conversation history
