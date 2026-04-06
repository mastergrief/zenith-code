# VDD — Single-Team Full Validation-Driven Development

**Input**: [$ARGUMENTS] - Feature/bug description, or following `/DISCOVER`, `/SPEC` output

**Single team, phased spawning, zero team churn.** Full discovery-plan → implementation → validation with cross-phase consultation and built-in self-healing.

**When to use:**
- **/VDD**: Cross-module, uncertain scope, complex features, needs runtime verification
- **/DISCOVER + manual dev**: Scope is clear, just want investigation then implement yourself
- **Developer subagent directly**: Fix describable in one sentence

**Strategy**: `TeamCreate("vdd")` → phased teammate spawning → task-board across all phases → cross-phase DMs → self-healing → `TeamDelete()`

---

## Team Lifecycle

```
TeamCreate("vdd")

Phase 1: spawn explorer + trainer + planner
  → parallel investigation (explorer + trainer)
  → both report to planner → planner cross-challenges → investigators respond
  → planner synthesizes implementation plan
  → shutdown planner + trainer → keep explorer

Phase 2: spawn developer into same team
  → developer DMs explorer for clarification → implementation → keep developer alive

Phase 3: spawn reviewer + harness-tester into same team
  → reviewer DMs explorer → harness-tester ↔ developer self-healing → all pass

shutdown all → TeamDelete("vdd")
```

**Peak active teammates**: 4 (Phase 3: explorer + developer + reviewer + harness-tester)

---

## Phase 1: DISCOVER-PLAN

### Step 0: Conceptual Analysis (orchestrator, before team creation)
Think about the request:
- What domains are affected? (harness, distill, rust, models)
- Identify search terms, target files, function names
- Determine if trainer investigation is needed (skip if purely harness/rust change)
- This is conceptual only — no codebase searching

**Critical**: Step 0 output MUST be injected into every teammate spawn prompt.

### Team & Task Creation
```
TeamCreate("vdd")

# Phase 1a — Parallel Investigation
Task 1: "Code & Architecture Analysis"                  [no deps] — explorer
Task 2: "Training Data & Pipeline Analysis"             [no deps] — trainer

# Phase 1b — Planner Cross-Challenge (gates on ALL investigation)
Task 3: "Cross-challenge & contradiction resolution"    [depends: 1, 2] — planner

# Phase 1c — Targeted Responses (gates on planner challenge)
Task 4: "Explorer targeted response"                    [depends: 3] — explorer
Task 5: "Trainer targeted response"                     [depends: 3] — trainer

# Phase 1d — Implementation Plan (gates on all responses)
Task 6: "Implementation plan synthesis"                 [depends: 4, 5] — planner
```

Spawn 3 teammates in parallel (single function_calls block).

### Teammate: EXPLORER

`subagent_type: explorer`, `model: opus`, `team_name: vdd`, `name: explorer`

> Perform VERY THOROUGH exploration for: $ARGUMENTS
>
> **Context from orchestrator**: [inject Step 0 output]
>
> MISSION: Unified analysis — code patterns, file dependencies, implementation approaches, architecture & regression risk. Single pass, full cross-referencing.
>
> Use Serena tools: `get_symbols_overview`, `trace_dependencies`, `find_referencing_symbols`, `search_for_pattern`
>
> CODE PATTERNS:
> 1. Find ALL files related to this feature/area
> 2. Map existing patterns (Agent class, tool dispatch, Ollama/llama.cpp communication)
> 3. Document data pipeline: user input → agent → model → tool call → result → response
> 4. Identify reusable code and existing abstractions
>
> DEPENDENCIES:
> 5. Create import/dependency graph (`trace_dependencies` for transitive)
> 6. Identify shared utilities and their consumers
> 7. Flag files approaching size thresholds (>400 concern, >800 must-split)
>
> ARCHITECTURE & REGRESSION:
> 8. Map how this feature integrates with the broader system
> 9. Find ALL consumers of code that would be modified
> 10. List features that could break (ASCII tree to visualize)
> 11. Document API contracts (Agent methods, tool signatures, Coordinator protocol)
> 12. Flag recent git changes that might conflict
>
> ### Reporting
> DM `planner` with structured findings. Then wait for cross-challenge DMs (Task 4).
> **Stay alive through all phases** — developer and reviewer will consult you.

### Teammate: TRAINER

`subagent_type: trainer`, `model: opus`, `team_name: vdd`, `name: trainer`

> Analyze training data and pipeline for: $ARGUMENTS
>
> **Context from orchestrator**: [inject Step 0 output]
>
> TRAINING DATA:
> 1. Identify affected training data files
> 2. Validate JSONL format and example quality
> 3. Check domain coverage and gaps
> 4. Verify filter pipeline handles changes
> 5. Document data prerequisites for any training runs
>
> PIPELINE INTEGRITY:
> 6. Verify config.py domain definitions
> 7. Check training scripts reference correct paths
> 8. Verify export pipeline (GGUF conversion, Ollama Modelfile)
>
> ### Reporting
> DM `planner` with structured findings. Then wait for cross-challenge DMs (Task 5).
> Stay alive through Phase 2 — developer may DM with training data questions.

### Teammate: PLANNER

`subagent_type: planner`, `model: opus`, `team_name: vdd`, `name: planner`

> You are the planner in a single-team VDD workflow for: $ARGUMENTS
>
> **Context from orchestrator**: [inject Step 0 output]
>
> ### Phase 1b: Collect & Cross-Challenge (Task 3)
> Wait for 2 DMs (explorer + trainer). Cross-challenge:
> 1. Identify contradictions between code and training data findings
> 2. Identify gaps neither covered
> 3. Design targeted challenges with specific file paths and function names
> DM each investigator with challenges.
>
> ### Phase 1d: Implementation Plan Synthesis (Task 6)
> After both respond to challenges, synthesize plan.
>
> **TESTABLE ASSERTIONS**: For each developer step:
> ```
> STEP N ASSERTIONS:
> - [code]: "import agents.harness succeeds with no errors"
> - [rust]: "cargo clippy passes with no warnings"
> - [data]: "JSONL has N examples, all valid format"
> - [runtime]: "harness responds to test prompt within 30s"
> ```
>
> **HARNESS TEST SCENARIOS**: For each step with runtime impact:
> ```
> HARNESS_SCENARIOS:
> - Scenario: <name>
>   Setup: <model needed, environment state>
>   Action: <what to send/do>
>   Verify: <expected output/behavior>
>   Fallback: <what to do if model not running>
> ```
>
> **DEVELOPER SIZING**:
> - **SMALL** (1 developer): Single concern, ≤3 files, no training data changes. → Single task.
> - **MEDIUM** (2 sequential): Multiple concerns, 4-8 files. → 2 ordered steps.
> - **LARGE** (3+): Cross-cutting, 8+ files, multiple domains. → 3+ ordered steps.
>
> Emit `SIZING: SMALL | MEDIUM | LARGE` + `DEVELOPER_STEPS: <N>` at top.
> Mark complete and message orchestrator with full plan.

### Phase 1 Gate

Orchestrator waits for planner's plan (Task 6). Task dependencies ensure cross-challenge completes first.

### Phase 1 Selective Shutdown

```
shutdown_request planner                    # Plan delivered
shutdown_request trainer                    # Default — or keep for training-heavy tasks
# explorer stays alive through all phases
```

---

## Phase 2: DEVELOP

Orchestrator adds dev tasks based on planner's SIZING:

**SMALL**: `Task 7: "Implementation: [full scope]" [depends: 6]`
**MEDIUM**: `Task 7: "Step 1/2: [scope]" [depends: 6]`, `Task 8: "Step 2/2: [scope]" [depends: 7]`
**LARGE**: Tasks 7..N, each depending on the previous.

### Developer Spawning

**SMALL** (1 developer, no plan approval):
`subagent_type: developer`, `model: opus`, `team_name: vdd`, `name: developer`

**MEDIUM/LARGE** (sequential, with plan approval):
Step 1: `subagent_type: developer`, `model: opus`, `mode: plan`, `team_name: vdd`, `name: developer`
Step 2+: Shutdown previous developer → spawn fresh with prior step context + `mode: plan`.
**Keep the LAST developer alive** — freshest implementation context for Phase 3 self-healing.

### Developer Prompt

> You are the developer in a single-team VDD workflow for: $ARGUMENTS
>
> [Plan from planner with testable assertions and harness test scenarios]
>
> You have teammates available for consultation:
> - DM `explorer` for: blast radius questions, pattern clarification, "is X used elsewhere?"
> - DM `trainer` for: training data format questions, domain coverage (if still alive)
>
> WORKFLOW:
> 1. Review plan and testable assertions for your step
> 2. Implement changes using Serena symbolic tools where appropriate
> 3. Run static checks:
>    - Python: `PYTHONPATH=. python3 -c "import agents.harness"` 
>    - Rust: `cd rust && cargo clippy --workspace --all-targets -- -D warnings`
> 4. Verify testable assertions from plan
> 5. Mark your task complete
>
> **Stay alive through Phase 3** — reviewer or harness-tester may report issues.
>
> SELF-HEALING (triggered by Phase 3 failure messages):
> 1. Read failure details from reviewer or harness-tester
> 2. If root cause unclear → DM `explorer` for investigation help
> 3. Fix → static checks → message the reporter to retest
> 4. Max 2 fix attempts per issue. Still failing → message orchestrator.

### VERIFY-ON-DISK Gate (after each developer step)

Orchestrator runs `git diff --stat`:
- Files changed → proceed
- No changes → message developer to retry (max 2 attempts, then report to user)

### PRE-VALIDATE Gate (after all dev steps complete)

1. Python imports pass: `PYTHONPATH=. python3 -c "import agents.harness; import agents.agent; print('OK')"`
2. Rust compiles (if changed): `cd rust && cargo check --workspace`
3. Both pass → proceed to Phase 3. Either fails → fix before spawning Phase 3.

---

## Phase 3: VALIDATE

Orchestrator adds validation tasks:
```
Task N+1: "Code Review"              [depends: last dev task]
Task N+2: "Harness Integration Test" [depends: last dev task]
```

Spawn 2 teammates in parallel (single function_calls block).

### Teammate: REVIEWER

`subagent_type: reviewer`, `model: opus`, `team_name: vdd`, `name: reviewer`

> Perform VERY THOROUGH code review for: $ARGUMENTS
>
> You are the review gate. Verify correctness against the plan — NOT stylistic improvements.
>
> INPUTS: Plan from Phase 1 (with testable assertions), discovery findings.
>
> You have teammates available:
> - DM `explorer` for: regression risk checks, dependency questions
> - DM `harness-tester` if review findings suggest specific runtime checks
>
> REVIEW CHECKLIST:
> 1. Assertions: verify each step's testable assertions
> 2. Regression risks: check flagged risks — addressed or avoided?
> 3. API contracts: modified functions maintain contracts?
> 4. Logic correctness: logic matches plan's intent?
> 5. Edge cases: error paths, boundary conditions handled?
> 6. Convention compliance: stdlib-only in agents/, JSONL format in distill/
> 7. Static checks: cargo clippy, python imports
>
> OUTPUT: `PASS`, `PASS_WITH_DEFECTS`, or `FAIL` with specific issues.
> On FAIL: message `developer` directly. On PASS: message orchestrator.

### Teammate: HARNESS-TESTER

`subagent_type: harness-tester`, `model: opus`, `team_name: vdd`, `name: harness-tester`

> You are the harness integration tester in a VDD workflow.
>
> Wait for your task to unblock (developer must complete). Execute test scenarios from the plan.
>
> **PHILOSOPHY**:
> - Run it, don't read it — code review catches logic errors, you catch runtime failures
> - Evidence-based — every assertion backed by actual command output
> - "Couldn't test" = FAIL, not PASS. If model isn't running, that's a failure.
>
> **SEQUENCE** (SETUP → ACT → VERIFY):
>
> | Phase | Steps | Gate |
> |-------|-------|------|
> | **SETUP** | Check Ollama/llama.cpp running, correct model loaded | Environment ready |
> | **ACT** | Execute harness test scenarios from plan | Actions executed |
> | **VERIFY** | Capture output, compare against expected behavior | Assertions pass |
>
> **ESCALATION** (only on VERIFY failure):
> 1. Capture exact error output
> 2. Check if it's environment issue (model not running) vs code issue
> 3. If code issue → message `developer` with failure details
>
> **ON FAILURE**: Message `developer` directly — they're still alive with full context.
> Include: what failed, exact output, reproduction steps.
> Developer will fix and tell you to retest. Max 2 retry cycles.
> After 2 retries still failing → message orchestrator for escalation.
>
> **ON PASS**: Message orchestrator with: summary of tests passed, output evidence.

---

## Phase 3 Flow

```
reviewer + harness-tester run in parallel

Review PASS + Harness PASS:
  → shutdown all → TeamDelete → DONE

Review PASS_WITH_DEFECTS + Harness PASS:
  → reviewer messages developer with defects
  → developer fixes → static checks → messages harness-tester to retest
  → harness PASS → DONE

Review FAIL:
  → reviewer messages developer with issues
  → developer fixes → static checks → repeat until PASS

Harness FAIL:
  → harness-tester messages developer directly
  → developer fixes → static checks → messages harness-tester to retest
  → max 2 fix cycles

Harness STUCK (2 fix cycles failed):
  1. Harness-tester messages orchestrator
  2. Orchestrator messages explorer: "investigate: [failure details]"
  3. Explorer analyzes → messages developer with diagnosis
  4. Developer fixes → harness retests
  5. Still failing → shutdown all → TeamDelete → report to user
```

---

## Cleanup

```
shutdown_request to all remaining teammates
→ wait ~5s for idle
→ TeamDelete()
If TeamDelete fails → retry once
```

---

## Orchestration Summary

```
TeamCreate("vdd")
Create Phase 1 tasks (6 tasks with dependencies)
Spawn explorer + trainer + planner (single function_calls block)

→ Phase 1a: parallel investigation (explorer + trainer)
  → both report to planner
→ Phase 1b: planner cross-challenges with full visibility
→ Phase 1c: investigators respond to planner's challenges
→ Phase 1d: planner synthesizes implementation plan
  → shutdown planner + trainer, keep explorer
  → add dev tasks based on planner SIZING

→ Spawn developer (mode: plan for MEDIUM/LARGE)
  → developer DMs explorer for clarification as needed
  → VERIFY-ON-DISK gate after each step
  → MEDIUM/LARGE: shutdown dev, spawn next with prior context, keep last dev alive
  → PRE-VALIDATE gate after all steps complete

→ Add validation tasks
→ Spawn reviewer + harness-tester (single function_calls block)
  → parallel: reviewer checks code, harness-tester tests runtime
  → reviewer DMs explorer for regression checks
  → harness-tester DMs developer on failure (self-healing)
  → reviewer DMs harness-tester with additional checks

→ All PASS → shutdown all → TeamDelete → DONE
```

---

## Rules

- Single team `vdd` — one TeamCreate at start, one TeamDelete at end. Zero team churn.
- Phased spawning: teammates join the team as their phase begins
- Task board spans entire lifecycle — Phase 1 tasks → dev tasks → validation tasks added progressively
- **Planner is the cross-challenge gate in Phase 1**: investigators report to planner, NOT to each other
- Cross-phase consultation via DMs: explorer available throughout, developer available in Phase 3
- Self-healing via direct messaging (harness-tester ↔ developer, reviewer → developer), no team respawn
- Selective shutdown: planner + trainer after Phase 1, last developer stays through Phase 3
- Plan approval (`mode: "plan"`) for MEDIUM/LARGE developer steps
- DEVELOPER_STEPS sizing from planner: SMALL (1), MEDIUM (2), LARGE (3+)
- Static checks are blocking after every developer edit
- VERIFY-ON-DISK gate after each developer step
- PRE-VALIDATE gate before Phase 3
- Three-layer verification: code review + harness integration test + static analysis
- Graceful shutdown: shutdown_request all → wait ~5s → TeamDelete
- Never run subagents in background
- Max 2 self-healing cycles per issue. Escalate to explorer, then to user.
- Always run Phase 1 — never skip, even with prior plans or specs
- **Step 0 output injected into every spawn prompt**
