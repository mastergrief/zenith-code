# DISCOVER-DEEP — Team-Based Discovery & Solutions Matrix

**Input**: [$ARGUMENTS] - Feature/bug description, query, or area to investigate

**Team-based discovery**: 3 parallel investigators (explorer + trainer + harness-tester) report to a planner who acts as synthesis gate and devil's advocate. Planner collects all findings before any cross-challenge begins.

**When to use:**
- **/DISCOVER-DEEP**: Complex investigation, cross-module, harness + training + runtime concerns, uncertain scope
- **/DISCOVER**: Quick investigation, narrow scope, single concern area

**Strategy**: `TeamCreate("discover")` → 3 investigators in parallel → all report to planner → planner drives cross-challenge with full visibility → planner assembles Solutions Matrix → `TeamDelete()`

---

## Setup

### Step 0: Conceptual Analysis (orchestrator, before team creation)
Think about the request:
- What exactly is being investigated (harness, distillation pipeline, rust, serving)?
- Identify search terms: function names, class names, tool names, config keys, model names
- Determine relevant domains: harness (`agents/`), distillation (`agents/distill/`), rust (`rust/`), models (`models/`)
- Which investigators are relevant? (skip harness-tester if purely code/data investigation)
- This is conceptual only — no codebase searching

**Critical**: Step 0 output MUST be injected into every teammate spawn prompt. Teammates don't inherit conversation history — without this context they re-derive what you already figured out. Include: search terms, target directories, domain objects.

### Team Creation
```
TeamCreate("discover")
Create tasks (granular, with dependency gates):

  # Phase 1 — Parallel Investigation
  Task 1:  "Code patterns & implementation mapping"         [no deps] — explorer
  Task 2:  "Architecture & regression risk analysis"        [no deps] — explorer
  Task 3:  "Training data & pipeline analysis"              [no deps] — trainer
  Task 4:  "Data quality & format validation"               [no deps] — trainer
  Task 5:  "Harness smoke test & environment check"         [no deps] — harness-tester
  Task 6:  "Tool calling & integration verification"        [no deps] — harness-tester

  # Phase 2 — Planner Cross-Challenge (gates on ALL Phase 1)
  Task 7:  "Synthesis & cross-challenge design"             [depends: 1,2,3,4,5,6] — planner

  # Phase 3 — Targeted Responses (gates on planner challenge)
  Task 8:  "Explorer targeted response"                     [depends: 7] — explorer
  Task 9:  "Trainer targeted response"                      [depends: 7] — trainer
  Task 10: "Harness-tester targeted probing"                [depends: 7] — harness-tester

  # Phase 4 — Solutions Matrix (gates on all responses)
  Task 11: "Solutions Matrix assembly"                      [depends: 8,9,10] — planner

Spawn all 4 teammates in parallel (single function_calls block)
```

Tasks 1-6 start immediately (Phase 1). Cross-challenge (7) gates on all investigation completing. Targeted responses (8-10) gate on planner's challenges. Solutions Matrix (11) gates on all responses.

---

## Teammate 1: EXPLORER — Code Patterns, Architecture & Regression Risk

`subagent_type: explorer`, `model: opus`, `team_name: discover`, `name: explorer`

> Perform VERY THOROUGH code exploration for: $ARGUMENTS
>
> **Context from orchestrator**: [inject Step 0 output — search terms, directories, domain objects]
>
> You are the code & architecture analyst in a discovery team. Focus on Concerns A (code patterns) and B (architecture/regression).
>
> ### Git Diff Scoping (run first)
> ```bash
> git log --oneline -15 -- <relevant paths>
> git diff --stat HEAD~10 -- <relevant paths>
> ```
>
> ### Serena-Assisted Analysis
> - `get_symbols_overview` for file structure mapping
> - `trace_dependencies` for transitive import graphs
> - `find_referencing_symbols` for consumer tracking
> - `search_for_pattern` for cross-module pattern finding
>
> ### 4-Phase Search: DISCOVER → LOCATE → UNDERSTAND → VALIDATE
>
> **CONCERN A: Code Patterns & Implementation** (Task 1)
> 1. Find ALL files related to the feature/area
> 2. Map existing patterns (Agent class hierarchy, tool dispatch, Ollama communication)
> 3. Document data pipeline: user input → agent → Ollama → tool call → result → response
> 4. Identify reusable code and existing abstractions
> 5. Map import/dependency graph (`trace_dependencies` for transitive incoming + outgoing)
> 6. Identify shared utilities and their consumers
> 7. Flag files approaching size thresholds (>400 concern, >800 must-split)
> 8. Identify entry points for changes, symbols needing modification vs creation
> 9. Document validation, error handling, edge case patterns
> 10. Note TODO/FIXME comments in the area
>
> **CONCERN B: Architecture & Regression Risk** (Task 2)
> 1. Map how feature integrates with broader system (harness ↔ agents ↔ Ollama/llama.cpp)
> 2. Identify module boundaries crossed
> 3. Document API contracts (Agent methods, tool signatures, Coordinator protocol)
> 4. Check convention compliance (stdlib-only in agents/, JSONL format in distill/)
> 5. **Transitive blast radius**: `trace_dependencies` on each modified symbol
> 6. Map shared state (agent history, permissions, session state) affected
> 7. Identify cross-feature data flows
> 8. List features that could break (ASCII tree to visualize)
> 9. Document assumptions other code makes about this area
> 10. Flag recent changes that might conflict
>
> ### Reporting (after Tasks 1+2 complete)
> DM `planner` with your full structured findings. Include:
> - File list with line counts
> - Dependency graph (ASCII trees from trace_dependencies)
> - Pattern inventory
> - Entry points for changes
> - Architecture impact assessment
> - Consumer list
> - Regression risk matrix
> - API contracts
>
> Then **wait** for planner's cross-challenge DMs (Task 8). Respond to each challenge with targeted investigation and DM findings back to `planner`.
> Mark sub-tasks complete as you finish each.

---

## Teammate 2: TRAINER — Training Data & Pipeline Analysis

`subagent_type: trainer`, `model: opus`, `team_name: discover`, `name: trainer`

> Analyze training data and distillation pipeline for: $ARGUMENTS
>
> **Context from orchestrator**: [inject Step 0 output — search terms, directories, domain objects]
>
> You are the training data and ML pipeline analyst in a discovery team. Focus on Concern C (data quality & pipeline).
>
> **CONCERN C: Training Data & Pipeline** (Tasks 3+4)
> 1. Identify ALL training data files relevant to the feature
> 2. Validate JSONL format: 3 messages (system, user, assistant), assistant starts with `<think>`
> 3. Check for duplicate examples (first 60 chars dedup)
> 4. Verify think-block quality (minimum length, genuine reasoning, not just restating)
> 5. Check for hallucinated facts, wrong technical claims, made-up libraries
> 6. Count examples per domain and category — identify gaps
> 7. Verify filter pipeline (`filter_reasoning.py`) handles edge cases
> 8. Check config.py domain definitions match actual data files
> 9. Verify training scripts reference correct data paths
> 10. If specialist data involved: verify domain relevance (rust examples in rust.jsonl, not python)
> 11. Check backward-compat: do new examples work with existing filter pipeline?
> 12. Document data prerequisites for any training runs
>
> ### Reporting (after Tasks 3+4 complete)
> DM `planner` with your full structured findings. Include:
> - Data file inventory (file, count, quality assessment)
> - Format validation results
> - Duplicate/quality issues found
> - Pipeline integrity check
> - Domain coverage gaps
> - Recommendations
>
> Then **wait** for planner's cross-challenge DMs (Task 9). Respond to each challenge with targeted investigation and DM findings back to `planner`.
> Mark sub-tasks complete as you finish each.

---

## Teammate 3: HARNESS-TESTER — Runtime & Integration Verification

`subagent_type: harness-tester`, `model: opus`, `team_name: discover`, `name: harness-tester`

> Test the live harness for: $ARGUMENTS
>
> **Context from orchestrator**: [inject Step 0 output — search terms, relevant features to test]
>
> You are the runtime tester in a discovery team. Focus on Concern D (live harness behavior).
>
> ### Phase A: General Observation (Tasks 5+6 — start immediately)
>
> **1. Environment Check** (Task 5)
> - Is Ollama running? What models available?
> - Is llama.cpp running? What port?
> - Python version, PYTHONPATH correct?
> - Any port conflicts?
>
> **2. Integration Verification** (Task 6)
> - Start harness, verify banner displays
> - Send basic prompt, verify response
> - Test tool calling round-trip (read_file, bash, grep)
> - Test permission blocking (destructive commands)
> - Check streaming works (tokens appear incrementally)
> - Check error handling (invalid model, connection refused)
>
> **3. Deep Observation (if idle before planner DMs arrive)**
> Don't wait idle. If Phase A completes before cross-challenge arrives:
> - Test coordinator mode (team delegation)
> - Test session save/load
> - Test compaction (if small context model available)
> - Test specialist routing (if specialists configured)
>
> ### Reporting (after Tasks 5+6 complete)
> DM `planner` with your full structured findings:
> 1. Environment status
> 2. Harness startup success/failure
> 3. Tool calling: which tools work, which don't
> 4. Permission enforcement: working or bypassed
> 5. Streaming: working or broken
> 6. Error handling: graceful or crashes
> 7. Any unexpected behavior
>
> ### Phase B: Targeted Probing (Task 10 — after planner DMs)
> Planner will DM you with specific things to test based on code + data analysis:
> - Edge cases from code analysis
> - Specific tool behaviors that changed
> - Specialist routing scenarios
> - Compaction edge cases
>
> Execute each targeted probe. Capture output and report findings.
> DM `planner` with results. Mark sub-tasks complete as you finish each.
>
> ### CRITICAL RULES
> - **DO NOT self-terminate.** Only the orchestrator sends shutdown requests.
> - **DO NOT write reports to files.** Reports go via DM to planner only.
> - **Stay alive through ALL phases** — Phase A (observation), idle deep-observation, Phase B (targeted probing).
> - Use `timeout` to prevent hanging tests (max 30s per test).

---

## Teammate 4: PLANNER — Synthesis Gate & Devil's Advocate

`subagent_type: planner`, `model: opus`, `team_name: discover`, `name: planner`

> You are the synthesis gate and devil's advocate in a discovery team for: $ARGUMENTS
>
> **Context from orchestrator**: [inject Step 0 output — search terms, directories, domain objects]
>
> ### Your Role
> You do NOT investigate the codebase directly. You receive findings from 3 investigators (explorer, trainer, harness-tester), identify contradictions and gaps, drive targeted cross-challenge, then assemble the final Solutions Matrix.
>
> ### Phase 2: Collect & Cross-Challenge (Task 7)
>
> **BLOCKING: Wait for ALL 3 DMs before proceeding.** Do NOT start cross-challenge or synthesis until every investigator has reported.
>
> Required reports (one from each):
> - `explorer`: file list, dependency graph, patterns, architecture impact, regression risks
> - `trainer`: data file inventory, format validation, pipeline integrity, domain coverage
> - `harness-tester`: environment status, tool calling results, streaming, error handling
>
> **Verification procedure (MANDATORY before ANY cross-challenge work):**
> 1. Run `TaskList`
> 2. Confirm Tasks 1-6 are ALL status `completed` — if ANY is `in_progress` or `pending`, STOP and WAIT
> 3. Confirm you have received 3 DMs (one per investigator) — if a task shows completed but no DM received, DM that investigator requesting findings
> 4. **Harness-tester is typically the SLOWEST investigator.** If explorer and trainer reported but harness-tester hasn't, DO NOT proceed. Wait. 2-of-3 is NOT enough.
>
> **Once ALL 3 reports are confirmed received, cross-challenge:**
>
> 1. **Identify contradictions** — where domains disagree:
>    - Explorer says tool X is registered; harness-tester says it throws "unknown tool"
>    - Explorer says streaming callback changed; harness-tester says streaming still works
>    - Trainer says training data covers domain Y; explorer says specialist router doesn't handle Y
>    - Explorer says permission check exists; harness-tester says destructive command wasn't blocked
>
> 2. **Identify gaps** — what no investigator covered:
>    - Code paths found by explorer but not tested by harness-tester
>    - Training domains not validated against specialist routing code
>    - Error handling paths not triggered during testing
>    - Config changes that affect both harness and training pipeline
>
> 3. **Design targeted challenges** — specific, actionable, with context:
>    - **→ explorer**: "harness-tester found tool X fails at runtime — trace the dispatch path in tools.py and find where it breaks"
>    - **→ trainer**: "explorer found specialist routing expects domain Y but training data file for Y is empty — verify data exists and check config.py"
>    - **→ harness-tester**: "explorer found streaming callback changed at agent.py:55 — verify tokens still stream correctly with a test prompt"
>
> DM each investigator with their targeted challenges. Be specific — include file paths, function names, line numbers from their reports.
>
> ### Phase 4: Solutions Matrix Assembly (Task 11)
>
> **BLOCKING: Do NOT assemble the matrix until ALL 3 targeted responses (Tasks 8-10) are received.**
> **Verification procedure (MANDATORY before matrix assembly):**
> 1. Run `TaskList`
> 2. Confirm Tasks 8, 9, AND 10 are ALL status `completed`
> 3. If ANY task is not `completed`, DO NOT assemble. DM the incomplete investigator to check status.
> 4. Only after ALL 3 are `completed` AND you have received 3 response DMs → proceed.
>
> After all 3 investigators respond to challenges, assemble:
>
> ```
> ## Solutions Matrix: [$ARGUMENTS]
>
> ### Executive Summary
> [2-3 sentences from combined findings]
>
> ### Affected Surface
> [From explorer — file list, key symbols, line counts]
>
> ### Training Data Status
> [From trainer — data inventory, quality issues, domain coverage]
>
> ### Runtime & Integration Findings
> [From harness-tester — what works, what's broken, environment status]
>
> ### Cross-Challenge Findings
> [Contradictions found and how they were resolved. Gaps identified and filled.
>  What changed between initial reports and post-challenge responses]
>
> ### Dependency Graph
> [ASCII trees from explorer's trace_dependencies]
>
> ### Solution Options
> [Synthesized from all findings — concrete implementation paths]
>
> ### Implementation Sequence
> [Ordered steps with dependencies]
>
> ### Regression Risks
> [Combined: explorer blast radius + trainer data integrity + harness-tester runtime issues]
>
> ### Open Questions
> [Ambiguities from any domain that couldn't be resolved]
> ```
>
> DM orchestrator with the complete Solutions Matrix. Mark Task 11 complete.

---

## Orchestration Flow

```
Step 0: Conceptual analysis (orchestrator) → extract search terms, dirs, domain objects
TeamCreate("discover") → 11 tasks (with dependency gates) → spawn 4 teammates

Phase 1 — Parallel investigation (no dependencies):
  explorer:        code patterns (T1) + architecture risk (T2)            → DMs planner
  trainer:         training data analysis (T3) + quality validation (T4)  → DMs planner
  harness-tester:  environment check (T5) + integration test (T6)        → DMs planner
  planner:         waits for all 3 reports

Phase 2 — Planner cross-challenge (gates on T1-T6):
  planner: receives all 3 reports, identifies contradictions + gaps (T7)
  planner: DMs each investigator with targeted challenges

Phase 3 — Targeted responses (gates on T7):
  explorer:        responds to planner's code challenges (T8)
  trainer:         responds to planner's data challenges (T9)
  harness-tester:  executes planner's targeted probes (T10)
  All respond back to planner

Phase 4 — Solutions Matrix (gates on T8-T10):
  planner: assembles Solutions Matrix from all findings (T11)
  planner: DMs orchestrator with complete matrix

Stuck teammate fallback:
  If any investigator hasn't reported within ~3 min → planner DMs to check status
  If planner hasn't received all reports → orchestrator DMs planner to check
  If teammate is stuck → planner redirects approach or asks orchestrator for guidance

shutdown all → wait ~5s → TeamDelete
```

---

## Cleanup

```
shutdown_request to explorer, trainer, harness-tester, planner
→ wait ~5s for idle
→ TeamDelete()
```

---

## Rules

- Single team `discover` — one TeamCreate, one TeamDelete
- 4 teammates spawned in parallel (single `<function_calls>` block)
- Granular tasks (11 total) with dependency gates — not 1 task per teammate
- **Step 0 output injected into every spawn prompt** — teammates don't inherit conversation history
- **Planner is the synthesis gate**: all investigators report to planner, NOT to each other. No peer-to-peer DMs — planner drives all cross-communication with full cross-domain visibility
- **BLOCKING COLLECTION**: Planner MUST receive ALL 3 investigator reports before starting cross-challenge. Planner MUST receive ALL 3 targeted responses before assembling Solutions Matrix. Verify via `TaskList` — do NOT proceed with partial reports. 2-of-3 is NOT enough.
- **Cross-challenge is MANDATORY**: Planner must ALWAYS perform cross-challenge (Phase 2) before Solutions Matrix (Phase 4). Never skip straight from collection to synthesis.
- **Devil's advocate**: planner identifies contradictions between domains that individual investigators can't see (code says X, data shows Y, harness does Z)
- Harness-tester does uninformed observation first, deep observation if idle, then informed probing from planner
- Solutions Matrix owned by planner (Task 11), not orchestrator
- Orchestrator reads planner's final output and reports to user
- No code changes — analysis and recommendations only
- Serena 4-phase compliance: explorer must complete DISCOVER → LOCATE → UNDERSTAND → VALIDATE
- **Harness-tester gate**: harness-tester may be slowest. NEVER skip or proceed without its findings.
- **Harness-tester anti-shutdown**: harness-tester must NEVER self-terminate. Only orchestrator sends shutdown_request.
- **Stuck fallback**: planner DMs investigator if no report within ~3 min; orchestrator DMs planner if matrix overdue
- Graceful shutdown: wait ~5s → TeamDelete
- Never run subagents in background
