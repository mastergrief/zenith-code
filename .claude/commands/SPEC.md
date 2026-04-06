# SPEC — Post-Discovery Specification & Memory Persist

**Input**: [$ARGUMENTS] - Optional refinements or direction. Assumes `/DISCOVER` or `/DISCOVER-DEEP` Solutions Matrix is already in conversation context.

**Post-discovery protocol**: Consumes existing Solutions Matrix from conversation, drives collaborative decision closure, writes implementation-ready spec to Serena memory. No redundant codebase search — discovery is already done.

**IMPORTANT**: Parent-only. Never `Edit` or `Write` code. Never spawn subagents. Spec authoring & persistence only.

---

## STEP 1 — VALIDATE DISCOVERY CONTEXT

Verify Solutions Matrix is present in conversation:
- Executive Summary, Affected Surface, Solution Options, Implementation Sequence, Regression Risks
- If missing/incomplete → tell user to run `/DISCOVER` or `/DISCOVER-DEEP` first (do NOT fallback discover)
- If present → extract: recommended approach, key files, open questions, sizing

---

## STEP 2 — COLLABORATIVE DECISION CLOSURE

Using Solutions Matrix as foundation:
- Present recommended approach with rationale grounded in discovery evidence
- Surface ALL open questions from Solutions Matrix — close every one
- `AskUserQuestion` for decisions on: scope boundaries, approach trade-offs, domain priorities
- Push back with alternatives if you see issues with proposed direction
- Flag any new concerns discovered during discussion
- Iterate until ALL decisions are closed — zero ambiguity remaining

**Decision closure checklist:**
- [ ] Scope: what's in, what's explicitly out
- [ ] Approach: which solution option (or hybrid)
- [ ] Domain: which domains affected (harness, distill, rust, models)
- [ ] Edge cases: how to handle identified regression risks
- [ ] Testing: what the harness-tester should verify after implementation
- [ ] Sizing confirmed: S/M/L and developer step count

---

## STEP 3 — WRITE SPEC TO SERENA MEMORY

Once Step 2 reaches consensus, persist as Serena memory.

### 3a. Naming & Structure
- Derive `FEATURE_NAME` from the agreed feature (e.g., `LLAMA_CPP_BACKEND_SPEC`, `SPECIALIST_ROUTING_SPEC`)
- Use `mcp__serena__write_memory` with structured content
- Each memory is self-contained — a fresh context window reads it cold

### 3b. Required Sections

| Section | Content |
|---------|---------|
| **Overview** | Status (`READY FOR IMPLEMENTATION`), design decisions with rationale, affected domains |
| **Architecture** | Data flow diagram (ASCII), module map with file:line refs, regression risks table, sizing + developer step count |
| **Implementation** | Ordered developer steps (Step 1 of N: [scope]), per-step file list with exact changes (code snippets), per-step testable assertions |
| **Testing** | Harness test scenarios (environment → action → verify), test model to use, expected behavior, edge cases, regression scenarios |

### 3c. Situational Sections (include only what applies)

| Section | When to Include | Content |
|---------|----------------|---------|
| Training Data | Training pipeline changes | Data format changes, filter updates, new domain data needed |
| Model Serving | llama.cpp/Ollama changes | Config changes, VRAM budget, context size, quantization |
| Rust Workspace | Rust crate changes | Crate dependency map, cargo commands, test plan |

### 3d. Writing Rules
- **Concise** — dense reference docs, not prose. A developer reads this and codes.
- **Code snippets** for key changes — show the actual fix, not just describe it
- **ASCII diagrams** for data flow and component relationships
- **Testable assertions** — mechanically verifiable ("tool returns string starting with 'Error:'" not "handles errors well")
- **Decisions include rationale** — "llama.cpp over Ollama because 64K context with Q4 KV cache"
- **Sizing + developer steps** — carries from Solutions Matrix, confirmed in Step 2

---

## EXECUTION FLOW

```
┌─────────────────────────────────────────────────────────┐
│ STEP 1: VALIDATE DISCOVERY CONTEXT                       │
│ └── Verify Solutions Matrix present, extract key facts   │
├─────────────────────────────────────────────────────────┤
│ STEP 2: COLLABORATIVE DECISION CLOSURE                   │
│ ├── Present recommended approach                         │
│ ├── AskUserQuestion to close open decisions              │
│ ├── Iterate until zero ambiguity                         │
│ └── Confirm scope, approach, domains, edge cases, sizing │
├─────────────────────────────────────────────────────────┤
│ STEP 3: WRITE SPEC TO SERENA MEMORY                      │
│ ├── Overview: status, decisions, affected domains        │
│ ├── Architecture: flow, risks, sizing                    │
│ ├── Implementation: steps, code snippets, assertions     │
│ ├── Testing: harness scenarios, edge cases, regression   │
│ └── Situational sections as needed                       │
└─────────────────────────────────────────────────────────┘
```

---

## CRITICAL RULES

1. **Post-discovery only** — do NOT rerun codebase search. Solutions Matrix is the input.
2. **All decisions closed** — zero open questions in final spec. Every ambiguity resolved in Step 2.
3. **Self-contained memory** — a fresh context window reads the spec cold and implements. No conversation context needed.
4. **Code snippets mandatory** — show the actual change, not just "modify this function"
5. **Testable assertions mandatory** — every spec section has mechanically verifiable criteria
6. **No code changes** — spec writing only. Implementation happens in a separate `/VDD` run.
7. **Summary table** — always present to user after writing: what was persisted and where
8. **Close questions** — always use `AskUserQuestion` to close open questions IF any
