# Session Handoff — 2026-04-12 (Session 21)

## Goal

Implement Auto-CALM — make CALM invisible, then scale it. Session 20
built explicit CALM (7,644 LOC, 98% benchmark) but the 4B model
can't emit multi-line code in `<calm>` blocks. Auto-CALM removes
that requirement — the model writes naturally, the engine verifies
everything on CPU.

User's vision evolved during the session: started with "implement
auto-calm" → grew to "build a nervous system for a brain" → "every
computable domain should have a backend" → "what if creative spark
could be computed too" → "is this a product?"

## Completed

### 20 commits this session, all on `feature/multi-agent-qwen`

Built the complete Auto-CALM system: 3 layers of verification,
30 modular backends, 251 verified compute functions, streaming
engine, self-learning, format-agnostic benchmark. Everything
hypothesis-test-iterated through Gemma 4 E4B tq4 at 512K context.

### Auto-CALM Architecture (calm/, ~14,700 LOC)

**Facade pattern** — `auto_calm.py` composes 6 focused modules:

| Module | LOC | Purpose |
|---|---|---|
| `auto_calm.py` | 324 | Facade: composes layers, CLI, re-exports |
| `verify.py` | 284 | Layer 1: claim extraction + verification |
| `precompute.py` | 346 | Layer 2: precomputation + system prompt |
| `intent_edit.py` | 356 | Layer 3: NL → template fix → test |
| `stream_auto.py` | 437 | Streaming verify + Gemma tool-call handler |
| `auto_learn.py` | 215 | Self-learning from corrections |
| `auto_training.py` | 337 | Training data generation |

**4-layer safety net** in streaming engine:
1. Precompute — inject verified facts before model responds
2. Stream verify — check claims at sentence boundaries as tokens arrive
3. Full-text verify — scan complete response post-generation
4. Prompt check — cross-check answer against expected value

**Self-learning loop** (closed):
```
model error → correction → pattern generalized → precompute next time
     ↑                                                        ↓
     └────── model uses precomputed fact ← no error ←─────────┘
```
Patterns stored in `calm/learned_patterns.jsonl` (committed to repo).

### 30 Modular Backends (calm/backends/, ~5,460 LOC)

Session 20 had 6 backends. This session added 24 more:

| # | Backend | Funcs | Added this session? |
|---|---|---|---|
| 1-5 | math, string, wasm, code, security | 57 | No (session 20) |
| 6-9 | date, convert, data, algo | 35 | No (session 20) |
| 10 | quality_ops | 7 | Yes — code complexity metrics |
| 11 | readability_ops | 5 | Yes — Flesch-Kincaid, jargon |
| 12 | regex_ops | 7 | Yes — pattern matching |
| 13 | json_ops | 7 | Yes — validate, path, diff |
| 14 | encoding_ops | 12 | Yes — base64, hex, hashing |
| 15 | git_ops | 7 | Yes — log, blame, status |
| 16 | network_ops | 9 | Yes — URL, IP, CIDR, HTTP |
| 17 | creative_ops | 9 | Yes — brainstorm, combine, novelty |
| 18 | impact_ops | 7 | Yes — call graph, blast radius |
| 19 | context_ops | 7 | Yes — git archaeology, code age |
| 20 | python_ops | 9 | Yes — builtin/method verification |
| 21 | math_extended_ops | 15 | Yes — matrices, modular arith, calculus |
| 22 | perf_ops | 6 | Yes — Big-O estimation |
| 23 | deps_ops | 6 | Yes — package analysis |
| 24 | refactor_ops | 4 | Yes — code smells |
| 25 | type_ops | 4 | Yes — annotation coverage |
| 26 | test_ops | 4 | Yes — test summary |
| 27 | doc_ops | 4 | Yes — docstring coverage |
| 28 | shell_ops | 7 | Yes — exit codes, dangerous cmds |
| 29 | semver_ops | 6 | Yes — version compare/satisfies |
| 30 | config_ops | 6 | Yes — YAML/TOML/INI/dotenv |

### Key Results

**Benchmark: 40/40 (100%)** — format-agnostic, prompt-independent.
Verified 3 times this session (with 9 backends, 16, and 30).

**Intent-to-edit: 10/10 + 13/13** on two unseen buggy files.
Template-based fixes (zero-check, try/except, bounds-check) +
LLM full-rewrite fallback + self-healing retry.

**Self-analysis**: Gemma analyzed its own code using quality_ops,
impact_ops, perf_ops — gave data-backed review citing specific
metrics (CC=13, coupling=7/100, 42.9% documented).

### Key Decisions

1. **Auto-discovery over manual registration** — precompute scans
   prompts for any registered function name. New backends work
   immediately with zero config in precompute.py.

2. **Compact system prompt** — listing all 251 functions caused
   the model to emit tool calls instead of NL. Fixed: show top 5
   per category (~739 tokens). Auto-discovery handles the rest.

3. **Format-agnostic benchmark** — keyword checker strips commas,
   LaTeX, markdown, backticks. Validates answers regardless of
   output format. Removed prompt dependency.

4. **Learned patterns committed to repo** — `calm/learned_patterns.jsonl`
   is the system's experience. Feeds precompute on CPU, not LoRA.
   Raw training data (`.calm_training/`) is gitignored.

5. **Precomputed facts capped at 5** — the learner generates
   combinatorial explosions (every N*M pair). Cap prevents context
   bloat and huge-integer crashes.

6. **Gemma tool-call handling** — model naturally emits
   `<|tool_call>call:module.func(args)`. Engine parses, executes,
   replaces with computed result. Works for all 30 backends.

### False Positive Log (claim verification)

| Issue | Fix |
|---|---|
| `391 = 17` from `391 = 17 × 20 + 51` | Require operator in LHS |
| RHS followed by operators | Negative lookahead |
| "if 391 is prime" (question, not claim) | Conditional word filter |
| Comma in numbers breaks LHS | Add `,` to digit class |
| "perfectly" matches "perfect" | Word boundary `\b` |
| Integer division "54÷7=7 R5" | Remainder context check |
| Single `*` stripped as markdown | Only strip `**` |
| Large integers crash str() | `_safe_str()` with try/except |

## Next Steps — Hypothesis, Test & Iterate

### Priority 1: More OP backends (user wants it loaded before using)

Build these next, each following hypothesis → measure → ship:

1. **SQL ops** — parse, validate, explain queries. Models write
   broken SQL constantly. AST parsing via `sqlparse` or manual.
   Test: have Gemma write queries and verify them.

2. **Cron ops** — parse cron expressions, compute next N runs,
   explain in English. Nobody can read `0 */4 * * 1-5`.
   Test: have Gemma interpret cron schedules.

3. **Bitwise ops** — masks, shifts, flags, bit counting. Pure
   computation the model can't do mentally.
   Test: have Gemma do bit manipulation problems.

4. **AST transform ops** — automated refactoring: rename symbol,
   extract function, inline variable. Actual code transforms,
   not suggestions. Test: refactor a messy file, verify syntax.

5. **Diff ops** — unified diff parsing, patch application,
   conflict detection. Test: parse a real git diff.

6. **Package ops** — npm/pip/cargo info from local cache.
   "What version of X do I have?" Test: query installed packages.

### Priority 2: Wire into harness

Once backends are "OP enough", integrate Auto-CALM into
`agents/harness.py` so every `zenith` response is verified.
The change is small — add `verify_and_correct()` as a
post-processor on every model response.

### Priority 3: Re-validate NIAH with tq4

The 200K compaction threshold in `compact.py:MODEL_CONTEXT_LIMITS`
was measured with Q5_K_M KV cache. tq4+tq4 KV may support
different effective context. Re-run `scripts/needle_test.py`
with the current tq4 serving config.

## Key Context

### Serving (unchanged)
- Gemma 4 E4B tq4 at 512K context, 32K thinking budget
- `~/models/gemma-4-E4B-it-tq4-aligned.gguf` (5.0 GB)
- `--cache-type-k tq4_k256 --cache-type-v tq4_k256 --parallel 1`
- ~47-50 tok/s, llama-server on port 8080
- Server crashes occasionally under heavy benchmark load — just restart

### System prompt sensitivity
- With 251 function names listed, model emits tool calls instead of NL
- Compact prompt (top 5 per category, ~739 tokens) fixes this
- Format-agnostic benchmark removes the dependency entirely
- The correctness doesn't depend on the prompt — only presentation does

### The learner generates too many precomputes
- `N * O` pattern matches every pair of numbers in a prompt
- Capped at 5 precomputed facts to prevent context bloat
- Large integers (factorial of large numbers) crash Python's str()
- `_safe_str()` truncates at 200 chars

### Architecture insight (from user)
- "Adding a backend is equivalent to training"
- "What if creative spark could be computed too?"
- "It's like giving a brain a nervous system"
- Percepta.ai is doing the opposite — putting computation inside
  transformer weights. Our approach: wrap the LLM in computers.
  Same result, ours is cheaper (zero training).

### Commercial direction
- `.claude/rules/commercial.md` documents product vision
- Currently R&D, not shipping. Commercial awareness is context.
- Key differentiators: tq4+tq4, Auto-CALM, self-learning, modular backends, fully local

## Files in Project

### Auto-CALM (new this session)
- `calm/auto_calm.py` (324) — facade: AutoCalmEngine, re-exports, CLI
- `calm/verify.py` (284) — Layer 1: claim extraction + verification
- `calm/precompute.py` (346) — Layer 2: precomputation + system prompt
- `calm/intent_edit.py` (356) — Layer 3: NL → template fix → verify
- `calm/stream_auto.py` (437) — streaming verify + tool-call handler
- `calm/auto_learn.py` (215) — self-learning from corrections
- `calm/auto_training.py` (337) — training data generation
- `calm/learned_patterns.jsonl` — self-learned error patterns (committed)
- `calm/backends/*.py` (~5,460) — 30 modular compute backends
- `calm/tests/test_auto_calm.py` (222) — 36 Auto-CALM tests

### Existing CALM (session 20)
- `calm/engine.py` (527) — explicit CALM v0.1
- `calm/stream_engine.py` (240) — explicit CALM v0.2
- `calm/interceptor.py` (479) — 4-tier parse
- `calm/expression.py` (780) — AST eval, 251 registered functions
- `calm/verifier.py` (560) — 4-lane TMR
- `calm/stack_vm.py` (522) — reference stack machine
- `calm/sandbox.py` (250) — subprocess isolation
- `calm/benchmark.py` (227) — 40-problem eval (format-agnostic)

## Useful Commands

```bash
# Auto-CALM (default)
python3 -m calm.auto_calm "What is fibonacci(30)? Is it prime?"

# Streaming Auto-CALM (4-layer safety net)
python3 -m calm.stream_auto "What is 347 * 289?"

# Intent-to-edit
python3 -c "from calm.auto_calm import IntentToEdit; IntentToEdit().fix('app.py', 'test_app.py', verbose=True)"

# Run all tests (250)
python3 -m pytest calm/tests/ -v

# Run 40-problem benchmark
python3 -u -c "
from calm.auto_calm import AutoCalmEngine
from calm.benchmark import PROBLEMS, _check_keywords
engine = AutoCalmEngine(thinking_budget=32768)
for prob in PROBLEMS:
    r = engine.run(prob.prompt, verbose=False)
    ok, _ = _check_keywords(r.response, prob.keywords)
    print(f'[{\"PASS\" if ok else \"FAIL\"}] #{prob.id} {prob.prompt[:50]}')
"

# Start llama-server (tq4+tq4, 512K)
~/llama.cpp/build/bin/llama-server \
    -m ~/models/gemma-4-E4B-it-tq4-aligned.gguf \
    --ctx-size 524288 --parallel 1 \
    --cache-type-k tq4_k256 --cache-type-v tq4_k256 \
    -ngl 999 --port 8080

# Check function count
python3 -c "from calm.expression import _FUNCTIONS; print(len(_FUNCTIONS))"
```
