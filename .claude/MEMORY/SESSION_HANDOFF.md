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

**Next session targets** (user's explicit request):
1. **Diff-based line edits** — the model says "change line 13 to X",
   the engine applies it. Solves the multi-line code_write problem.
2. **Structured edit protocol** — `{file, line, old, new}` JSON
   patches the model fills in, engine applies + verifies.
3. Continue with hypothesis-test-iterate workflow.

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

## Next Steps (user's explicit direction)

### 1. Diff-Based Line Edits (START HERE)

Instead of whole-file rewrites, the model specifies single-line changes:

```
<calm>
code.edit("/tmp/calc.py", 13, "    if b == 0: return 'Error: division by zero'")
code.edit("/tmp/calc.py", 14, "    return a / b")
</calm>
```

The expression evaluator handles `code.edit(path, line, content)` if
the content doesn't have quotes. For content with quotes, need to
handle escaping or use the sandbox.

**Approach**: extend `code.edit` to accept content via a simpler
encoding that avoids quote escaping issues. Options:
- Base64-encoded content: `code.edit(path, 13, b64"aWYgYiA9PSAw...")`
- JSON-patch format: `{"file": "...", "edits": [{"line": 13, "content": "..."}]}`
- Line-replacement by old→new: `code.replace(path, "return a / b", "if b == 0: return 'Error'\\n    return a / b")`

### 2. Structured Edit Protocol

Define a JSON patch format the model emits:
```json
{
  "file": "/tmp/calc.py",
  "edits": [
    {"line": 5, "action": "replace", "content": "    if b == 0: return 'Error'"},
    {"line": 6, "action": "insert", "content": "    return a / b"}
  ]
}
```

The engine parses this JSON (no quote escaping issues) and applies
the edits atomically. If syntax check fails after edits, auto-revert.

### 3. E2E Bug Fix Verification

Once edits work, the full loop:
1. `code.test` → find failures
2. `code.read` → see the code
3. Model thinks → plans fix
4. Structured edit → apply changes
5. `code.syntax_check` → verify valid Python
6. `code.test` → verify all pass
7. If fail → model reads error → tries again

### 4. Larger Model Testing

The 4B model's code generation is the constraint. Test with:
- Gemma 4 E4B at full size (not the 4B variant)
- Or run the CALM engine against a cloud API (Claude, GPT-4) to
  verify the architecture works with a frontier model

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
