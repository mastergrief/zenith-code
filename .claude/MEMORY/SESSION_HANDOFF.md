# Session Handoff — 2026-04-12 (Session 20)

## Goal

Build the complete CALM (Compute-Augmented Language Model) system
from scratch in one session. Started with phase 2a (stack VM from
session 19) and built everything through to a streaming execution
engine with 6 backend modules, 4-lane TMR verification, and an
automated 40-problem benchmark.

User's vision: **"deterministic brain on top of a probabilistic
nervous system"** — the LLM sequences, CPU computes, TMR verifies,
injection feeds back. A 4B model augmented with verified compute
modules that give it frontier-level capabilities on specific tasks.

**Next session target: Auto-CALM** (user's explicit pivot from
LoRA/training to pure infrastructure). Make CALM invisible — the
model writes naturally, the engine intercepts, verifies, and corrects
every computational claim and code edit without the model knowing
CALM exists. Three phases:
1. **Claim verification** — extract `X = Y` from output, verify, correct
2. **Computation extraction** — detect "let me compute X", evaluate, inject
3. **Intent-to-edit** — detect "change line 13 to...", generate code, apply, test
Hypothesis-test-iterate workflow throughout.

## Completed

### 8 commits this session (all on `feature/multi-agent-qwen`)

```
56be081 calm: streaming engine (v0.2) + sandbox file write
cf859c5 calm: wire security ops into expression evaluator + tune defaults
3db9d03 calm: security backend — advanced checks + multi-line SQL detection
11d2f95 calm: add security audit backend — OWASP Top 10 detection
71eab4e calm: add code backend module + expression evaluator dotted-name support
ce4e524 docs: update CLAUDE.md + rules for CALM, TurboQuant, llama.cpp zenith
dd913e6 calm: phase 2b-e — CALM execution engine + 4-lane TMR + 98% benchmark
d2a8ad8 calm: phase 2a — CALM v0.1 stack machine (29/29 tests pass) [session 19]
```

### Architecture: the MoE Pattern

The CALM engine is a Mixture of Experts where:
- **Router** = the LLM (decides what to compute)
- **Meta module** = engine + interceptor + 4-tier parser + verifier
- **Expert modules** = backends (math, string, wasm, code, security)

```
User prompt → [Planning turn: thinking 32K]
  ↓
[Streaming execution: SSE token-by-token, thinking on every turn]
  ↓
Model emits tokens → <calm> or <|tool_call> detected
  ↓
Accumulate until </calm> or <channel|>
  ↓
4-tier parse: NL → Stack VM → Expression → Sandbox
  ↓
Execute on CPU, 4-lane TMR verify
  ↓
Inject: [engine: stack=[391], output=[...]]
  ↓
Multi-turn continuation (assistant + user messages)
  ↓
Model thinks again (32K budget) → next block or final answer
```

### 6 Backend Modules (60+ operations)

| Module | Functions | LOC | Status |
|---|---|---|---|
| `math_ops.py` | 9 (sqrt, pow, is_prime, gcd, factorize, etc.) | 134 | Proven, 4-lane TMR |
| `string_ops.py` | 7 (len, upper, lower, contains, regex_match, etc.) | 92 | Proven |
| `wasm_ops.py` | 17 (int/float arithmetic via WebAssembly) | 174 | Proven, cross-checked vs math_ops |
| `code_ops.py` | 16 (read, write, edit, syntax_check, test, lint, search, etc.) | 310 | Proven read/test/analyze. Write limited. |
| `security_ops.py` | 8 (audit, sql_injection, xss, secrets, timing, ssrf, etc.) | 327 | Proven: 11/11 findings on subtle vuln file |
| `calm_math.wat` | WAT source for wasm backend | 60 | Compiled via wasmtime |

### Key Results

**Benchmark (40 problems):** 85-98% across runs. 100% on number
theory, reasoning, multi-step, algebra. Nondeterminism in whether
the model uses CALM vs answering from memory accounts for the range.

**Security audit:** 11 vulnerabilities found in a production-realistic
Flask app with subtle bugs (SSRF bypass, timing attack, format string
injection, multi-line SQL injection). 0 false positives.

**Code analysis:** Model reads our own codebase via CALM — correctly
reports 10 functions in interceptor.py, 479 lines, runs 29 tests.

**Streaming engine:** Thinking works on every turn — 5,471 chars of
thinking across 7 iterations (vs 400 chars on turn 0 only in v0.1).

**Generalization:** 5/5 novel problems (string counting, constraint
solving, financial math, algorithm tracing, logic puzzles) all
correct without domain-specific training.

### Key Decisions (with reasoning)

1. **CPU for compute, not GPU** — user proposed dual-lane KV cache.
   f16 KV at 256K = 8.9GB/head, far exceeding VRAM. CALM modules
   run on CPU + system RAM at zero GPU cost.

2. **4-tier parse pipeline** — the model writes Python, NL, expressions,
   or stack code unpredictably. Instead of forcing one format, try all
   four. Eliminated ~90% of "unknown instruction" errors.

3. **Post-verify over forced CALM** — nudging the model to use CALM
   dropped benchmark from 85% to 72%. Post-verify (let model answer,
   check independently, correct if wrong) raised it to 98%.

4. **Streaming engine (v0.2)** — `stop=["</calm>"]` fires during
   thinking, killing the response. SSE streaming detects `</calm>` in
   the token stream without stopping, so thinking works on every turn.

5. **Non-strict mode** — mismatches are training signal, not errors.
   The model never gets blocked by wrong predictions. VM is always
   authoritative.

6. **Auto-aliasing** — `register_backend("math.sqrt", fn)` auto-creates
   `"sqrt"` alias. `pop` → `drop`, `print` → `emit`. Model writes
   naturally, engine resolves.

### Benchmark Progression

| Change | Score |
|---|---|
| First live tests | 6/8 manual |
| `<pending>` eliminates predictions | 8/8 manual |
| First benchmark (40 problems) | 34/40 (85%) |
| + forced CALM nudge | 29/40 (72%) ← backfired, reverted |
| + post-verify + keyword fix | 36/40 (90%) |
| + empty retry + force on unverifiable | 39/40 (98%) |
| Typical run (nondeterminism) | 34-36/40 (85-90%) |

## In Progress

### The File Write Problem (the #1 blocker)

The 4B model can **read, analyze, test, and diagnose bugs perfectly**
through CALM. It correctly identifies divide-by-zero and input
validation bugs, plans the fix in 3,857 chars of thinking, but
**cannot emit multi-line Python inside `<calm>` blocks**.

The interceptor processes line-by-line. Multi-line strings
(triple-quoted, `\n` in strings) get split across lines and each
line fails. The sandbox block-level fallback helps but the model
can't format the `code_write(path, content)` call correctly because
the content has newlines and quotes that mess up the `<calm>` block
boundary detection.

**Proven**: sandbox CAN write files (tested manually: write fixed
calculator → 6/6 tests pass). The bottleneck is the model's ability
to emit the call, not the engine's ability to execute it.

## Next Steps — Auto-CALM (user's explicit direction)

### The Insight: Make CALM Invisible

The current architecture requires the model to emit `<calm>` blocks
with correct syntax. The 4B model can't reliably do this for code
edits (multi-line strings, quote escaping). **The fix: the model
doesn't need to know CALM exists.**

**Auto-CALM** = the engine intercepts the model's natural language
output, extracts computational claims and edit instructions, verifies
them, and silently corrects wrong answers. No `<calm>` tags, no
special syntax, no training.

### How Auto-CALM Works

```
Model writes naturally:
  "17 × 23 = 401, so the total is 401 + 100 = 501"
                           ↓
Engine scans output for computational claims:
  Claim 1: "17 × 23 = 401"  → engine computes 391 → WRONG
  Claim 2: "391 + 100 = 501" → engine computes 491 → WRONG (cascaded)
                           ↓
Engine rewrites:
  "17 × 23 = 391, so the total is 391 + 100 = 491"
                           ↓
Model sees corrected text in next turn (if multi-turn)
OR user sees corrected text directly (if single-turn)
```

### Auto-CALM for Code Edits

```
Model writes naturally:
  "The bug is on line 13 of calc.py. The divide function should
   check for zero: if b == 0, return an error message. Also line
   20 needs a try/except around the float() call."
                           ↓
Engine extracts structured edits:
  {file: "calc.py", edits: [
    {line: 13, intent: "add zero check before division"},
    {line: 20, intent: "wrap float() in try/except"}
  ]}
                           ↓
Engine generates the actual code (using the sandbox):
  Line 13: "    if b == 0: return 'Error: division by zero'"
  Line 20: wraps in try/except ValueError
                           ↓
Engine applies edits → code.syntax_check → code.test
                           ↓
If tests pass: inject "[verified: 2 edits applied, 6/6 tests pass]"
If tests fail: inject error, model retries with different description
```

### Three Layers of Auto-CALM

**Layer 1: Claim Verification (easiest, start here)**
- Regex-extract all `X = Y` and `X is Y` patterns from model output
- For numeric claims: evaluate X independently, check Y matches
- For boolean claims (is_prime, is_perfect): evaluate independently
- Wrong claims get silently corrected or flagged
- **Already partially implemented** as `_post_verify` in engine.py

**Layer 2: Computation Extraction**
- Detect when the model is doing math in natural language:
  "I need to multiply 17 by 23" → extract `17 * 23`
- Detect function-call-like descriptions:
  "Let me check if 391 is prime" → extract `is_prime(391)`
- Evaluate them before the model produces the answer
- Inject the correct result so the model doesn't need to guess
- **This replaces explicit `<calm>` blocks entirely.**

**Layer 3: Intent-to-Edit Translation (hardest, most impactful)**
- Detect when the model describes code changes in natural language:
  "Change the divide function to check for zero first"
  "Add error handling around the float conversion on line 20"
  "Replace the raw SQL query with parameterized query"
- Parse the intent into structured edits
- Generate the actual code changes (via sandbox)
- Apply → verify → test → report
- **This solves the file-write problem permanently** — the model
  describes what to change in English, the engine writes the code.

### Implementation Plan

**Phase 1: Claim Extraction + Verification (extend _post_verify)**
- Build `calm/auto_calm.py` with a `ClaimExtractor` class
- Regex patterns for `X = Y`, `X is Y`, `result is Y`, etc.
- Evaluate extracted expressions via `safe_eval` + sandbox
- Replace wrong claims in the output text
- Wire into both `engine.py` and `stream_engine.py`
- Measurement: run benchmark with auto-CALM, should hit 95%+
  because every wrong mental-math answer gets caught and fixed

**Phase 2: Computation Extraction (replace <calm> blocks)**
- Build `IntentExtractor` — detects "let me compute/check/find" patterns
- Extracts the computation as an expression
- Evaluates before the model produces the answer
- Injects the result into the model's context
- Measurement: model never needs to emit `<calm>` blocks, engine
  handles everything. Benchmark should match or exceed explicit CALM.

**Phase 3: Intent-to-Edit (the frontier unlock)**
- Build `EditExtractor` — detects code change descriptions
- Parses: file path, line numbers (from context), change description
- Uses the sandbox to generate the actual code:
  `run_python(f"generate_edit({file}, {line}, {description})")`
  where `generate_edit` is a template-based code generator
- Apply → syntax_check → test → report
- Measurement: model fixes the buggy calculator by describing the
  fix in natural language, engine applies and verifies.

### Why This Works Without Training

The model already does all the hard work:
- It **identifies the bug** correctly (proven: 3,857 chars of thinking)
- It **describes the fix** correctly in natural language
- It **knows which line** to change (reads the code, counts lines)

It just can't **format the fix as code inside a `<calm>` block**.

Auto-CALM removes that last requirement. The model describes the fix
in English. The engine converts English to code. The tests verify
the code. No CALM syntax knowledge needed. No training needed.

The `<calm>` block approach (v0.1/v0.2) remains available as an
explicit opt-in for power users or larger models that can format
the calls correctly. Auto-CALM is the transparent default that works
with any model size.

### Compatibility with LoRA (future)

If a CALM LoRA is trained later, it stacks cleanly:
- Auto-CALM handles what the model can't format
- LoRA teaches the model to format correctly over time
- As the model gets better at explicit `<calm>`, auto-CALM
  intervenes less (it only corrects wrong answers)
- Virtuous cycle: auto-CALM catches errors → training signal →
  LoRA learns → fewer errors → auto-CALM intervenes less

## Key Context

### Engine Defaults (as of this session)

| Setting | v0.1 (`engine.py`) | v0.2 (`stream_engine.py`) |
|---|---|---|
| Context | 512K (config.py) | 512K |
| Thinking budget | 32K | 32K |
| Max tokens/turn | 16K | 16K |
| Max iterations | 30 | 30 |
| Injection cap | 4000 chars | 4000 chars |
| Persist state | False | False |

### llama.cpp Interaction Gotchas

- `stop=["</calm>"]` fires during thinking — use SSE streaming instead
- Assistant prefill incompatible with `enable_thinking` — 400 error
- Gemma uses `<|tool_call>call:calm` / `<channel|>`, not `<calm>` / `</calm>`
- Model fabricates `[engine: ...]` lines — interceptor strips them
- Empty responses happen with large thinking budgets — engine retries (max 2)
- Multi-turn works: assistant + user messages for continuation

### The 4B Model's Strengths and Weaknesses

**Strong at (use CALM for these)**:
- Deciding what to compute (routing/sequencing)
- Reading and understanding code
- Planning multi-step approaches in thinking
- Using simple function calls: `gcd(391, 782)`, `is_prime(1009)`
- Interpreting injected results

**Weak at (this is what CALM compensates for)**:
- Multi-digit arithmetic (42*19 consistently wrong)
- Emitting multi-line code inside `<calm>` blocks
- Formatting complex function calls with escaped strings
- Knowing when to stop retrying (gets stuck in loops)

### TurboQuant / llama.cpp (unchanged from session 19)

- Production: `~/models/gemma-4-E4B-it-tq4-aligned.gguf` (5.0 GB)
- KV cache: `--cache-type-k tq4_k256 --cache-type-v tq4_k256`
- llama.cpp branch `zenith` at `a6218df` (3 commits: OP_TIMING, Gemma fusion fix, fused GLU kernel)
- ~45-48 tok/s at 8K context on RTX 4070 Laptop

## Files in Project (CALM subsystem, 7,644 LOC)

### Core Runtime
- `calm/engine.py` (527) — v0.1: hybrid thinking-plan + stop-mode
- `calm/stream_engine.py` (240) — v0.2: SSE streaming, thinking every turn
- `calm/interceptor.py` (479) — 4-tier parse + CALM block detection
- `calm/expression.py` (618) — AST-safe eval, 30+ functions, dotted names, code/security wrappers
- `calm/verifier.py` (560) — 4-lane TMR verification
- `calm/stack_vm.py` (522) — reference stack machine
- `calm/sandbox.py` (250) — subprocess Python isolation + code_write/read/edit
- `calm/nl_parser.py` (168) — NL → stack code translator
- `calm/grammar.py` (110) — GBNF grammar generator
- `calm/training.py` (94) — JSONL signal collector
- `calm/reasoning.py` (172) — structured chain tracker

### Backends
- `calm/backends/math_ops.py` (134) — 9 CPU math functions
- `calm/backends/string_ops.py` (92) — 7 string functions
- `calm/backends/wasm_ops.py` (174) — 17 wasm functions
- `calm/backends/code_ops.py` (310) — 16 code operations (read/write/test/lint/search)
- `calm/backends/security_ops.py` (327) — 8 security checks (OWASP Top 10 + advanced)
- `calm/backends/calm_math.wat` (60) — WebAssembly module

### Tests + Benchmarks
- `calm/tests/` (8 files, ~3000 LOC) — 214 tests
- `calm/benchmark.py` (227) — 40-problem automated eval
- `calm/live_test.py` (151) — grammar-constrained test
- `calm/meta_test.py` (203) — free-form meta-orchestration test

### Docs
- `.claude/rules/calm.md` — CALM subsystem rules
- `.claude/CLAUDE.md` — updated with CALM, TurboQuant, llama.cpp zenith

## Useful Commands

```bash
# Run all CALM tests (214 tests)
python3 -m pytest calm/tests/ -v

# Run benchmark (needs llama-server on :8080)
python3 -m calm.benchmark

# Streaming engine (v0.2, thinking every turn)
python3 -m calm.stream_engine "Find bugs in /tmp/app.py and fix them"

# v0.1 engine (stop-mode, more reliable for math)
python3 -m calm.engine "What is 17 * 23? Is it prime?"

# Security audit
python3 -c "from calm.expression import safe_eval; print(safe_eval('security.audit(\"/tmp/app.py\")'))"

# Start llama-server (tq4+tq4)
~/llama.cpp/build/bin/llama-server \
    -m ~/models/gemma-4-E4B-it-tq4-aligned.gguf \
    --ctx-size 8192 --parallel 1 \
    --cache-type-k tq4_k256 --cache-type-v tq4_k256 \
    -ngl 999 --port 8080
```
