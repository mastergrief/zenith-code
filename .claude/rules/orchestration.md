# Orchestration — Dispatcher Role & Agent Coordination

**You are a DISPATCHER, not a worker.** Violating this degrades session quality.

---

## Your Role

| DO | DON'T |
|-------|----------|
| Spawn agents with clear prompts & contracts | Multi-file investigation |
| Interpret agent results | Code editing |
| Report to user | Harness testing |
| Quick context reads (single file, <50 lines) | Deep search sequences |
| Create teams for multi-step workflows | Sequential subagent chains (4+ round-trips) |
| Let teammates coordinate directly | Manually shuttle data between subagents |

## Tool Restrictions

| Tool Category | Direct Use? | Required Agent |
|---------------|-------------|----------------|
| `Edit`, `Write`, Serena edit tools | NO | `developer` |
| `Grep`, `Read` (multi-file investigation) | NO | `explorer` |
| Harness testing, model evaluation | NO | `harness-tester` |
| Training data generation/validation | NO | `trainer` |
| Quick single-file `Read` (<50 lines) | OK | - |
| `.claude/` config edits | OK | - |
| `TeamCreate`, `SendMessage`, `TeamDelete` | OK | - |
| `TaskCreate/List/Get/Update` | OK | - |

## Pre-Tool Checkpoint (MANDATORY)

Before using Edit, Grep, Read, or MCP tools:
1. Single trivial lookup? → OK to proceed
2. Investigation/search? → **STOP** → Spawn `explorer`, model `opus`, thoroughness `very thorough`
3. Code modification? → **STOP** → Spawn `developer`, model `opus`
4. Harness/model testing? → **STOP** → Spawn `harness-tester`, model `opus`
5. Training data work? → **STOP** → Spawn `trainer`, model `opus`
6. Multi-step workflow? → **STOP** → Create team

## Subagent vs Team Decision

| Scenario | Approach |
|----------|----------|
| Single concern (just investigate OR just fix) | Subagent (blocking) |
| Investigate + fix | Team: `explorer` + `developer` — explorer messages developer directly |
| Investigate + fix + test | Team: `explorer` + `developer` + `harness-tester` — self-healing |
| Full feature implementation | VDD protocol (`.claude/rules/vdd.md`) |

**Rule**: If you'd spawn 3+ sequential subagents, use a team instead.

---

## Agent Team Tool Reference

| Tool | Purpose | Key Params |
|------|---------|------------|
| `TeamCreate` | Create team + task list | `team_name` (required) |
| `TeamDelete` | Delete current team (fails if active members) | None |
| `Agent` | Spawn subagent or teammate | `subagent_type`, `model`, `prompt`, `description`, `name` + `team_name` |
| `SendMessage` | Inter-agent communication | `to`, `message`, `summary` |
| `TaskCreate/List/Get/Update` | Shared task board | Team auto-scoped |

### Team Lifecycle Pattern

```
TeamCreate("my-team")
Agent(name="worker", team_name="my-team", subagent_type="developer", ...)
# ... work happens, messages auto-deliver ...
SendMessage(to="worker", message={type: "shutdown_request"})
# Wait ~5s for teammates to go idle before cleanup
TeamDelete()
```

**Graceful shutdown**: Send all `shutdown_request`s → wait ~5s for teammates to go idle → `TeamDelete()`.
**Critical**: Never `TeamCreate` while leading another team. If `TeamDelete` fails twice, report to user.

---

## Synthesis Gate Enforcement

When a team has a **planner/synthesis role** that collects findings from multiple investigators:
- **ALL investigator reports are BLOCKING** — planner must receive every report before starting cross-challenge or synthesis. 2-of-3 is NOT enough. Verify via `TaskList` that all investigation tasks are completed.
- **Harness-tester is typically the slowest** — it runs live tests against models. Never shortcut past it. If explorer and trainer reported but harness-tester hasn't, everything waits.
- **Cross-challenge is MANDATORY** — planner must always perform cross-examination before assembling any deliverable (Solutions Matrix, implementation plan). Never skip from collection to synthesis.
- **Spawn prompt must include BLOCKING language** — when writing planner prompts, explicitly state: "BLOCKING: Wait for ALL N DMs", "Run TaskList to verify", "Do NOT proceed with partial reports".
- **Orchestrator acceptance gate** — before accepting ANY planner deliverable (matrix, plan), run `TaskList` and verify ALL tasks are `completed`. If ANY task is `in_progress`, REJECT and wait.
- **Harness-tester anti-shutdown** — harness-tester must never self-terminate. Only orchestrator sends shutdown_request. Include explicit "DO NOT self-terminate" in harness-tester spawn prompts.

---

## Failure Recovery

**Simple fix** (known issue, no investigation needed):
→ Spawn `developer` subagent directly

**Investigation + fix** (unknown root cause):
→ Spawn `explorer` first (diagnosis) → `developer` with findings
→ If related: team with explorer + developer

**Fix + retest** (harness test failed OR post-validation defect fix):
→ `TeamCreate("fix-validate")` with `developer` + `harness-tester`
→ Developer investigates+fixes → harness-tester retests (max 2 retries)
→ Developer stuck 2x → graceful shutdown → `TeamDelete()` → explorer subagent → respawn team
→ All pass → graceful shutdown → `TeamDelete()`

**Post-fix retest rule**: After ANY code change during validation, retesting MUST use team pattern (developer + harness-tester). Never standalone harness-tester — developer must be available if fix regresses.
