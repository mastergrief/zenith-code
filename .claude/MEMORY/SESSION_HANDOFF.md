# Session Handoff — 2026-04-12 (Session 21)

## Goal

Implement Auto-CALM — make CALM invisible. Previous session (20) built
the explicit CALM system (7,644 LOC, 98% benchmark) but hit a wall:
the 4B model can't reliably emit multi-line code inside `<calm>` blocks.
Auto-CALM solves this by removing the requirement entirely — the model
writes naturally, the engine extracts/verifies/corrects every computational
claim without the model knowing CALM exists.

User directive: "implement auto-calm. hypothesis, test and iterate.
always run gemma with tq4 512k context, 32k think etc."

## Completed

### 3 commits this session (all on `feature/multi-agent-qwen`)

```
408cb64 auto-calm: boolean precomputation + tests (36 tests)
52c3978 auto-calm: precomputation injection (Layer 2 complete)
b1e8096 auto-calm: transparent compute verification (Layer 1+2)
```

### Result: 40/40 (100%) benchmark — up from 85-98% explicit CALM

| Mode | Score | Time | Notes |
|---|---|---|---|
| CALM v0.1 (explicit `<calm>`) | 85-98% | ~1000s | Model must format blocks |
| Auto-CALM (no precompute) | 35/40 (88%) | 1580s | Post-hoc only |
| **Auto-CALM + precompute** | **40/40 (100%)** | **1097s** | Pre-computed facts |

Per-category (all 100%): arithmetic 10/10, number_theory 10/10,
sequences 5/5, algebra 5/5, reasoning 5/5, multi_step 5/5.

### What was built: `calm/auto_calm.py` (814 LOC) + tests (222 LOC)

**Layer 1 — Claim Verification (`AutoCalm` class):**
- Regex extracts `expression = value` claims from model output
  - Arithmetic: `17 \times 23 = 391` (LaTeX + Unicode + plain)
  - Function calls: `factorial(10) = 3628800`
  - GCD/LCM: `GCD of 391 and 782 is 391`
  - Boolean: `391 is [not] prime`, `28 is a perfect number`, `X is divisible by Y`
- `_strip_formatting()` removes LaTeX (`\mathbf{}`, `\text{}`, `$`) and markdown (`**bold**`) before extraction
- `_is_conditional_match()` excludes question contexts: "if X is prime", "whether X is perfect", "check if X is divisible"
- `_is_integer_division_context()` detects "remainder" after division claims — treats `54 ÷ 7 = 7 remainder 5` as integer division
- Corrections applied from end-to-start to preserve span positions
- Boolean corrections flip `is`/`is not` instead of replacing with True/False

**Layer 2 — Precomputation + Prompt Verification (`AutoCalmEngine` class):**
- `_precompute(prompt)` extracts computations from the prompt BEFORE the model responds
  - Matches: "What is X?", "Compute X?", "Calculate X?", "Find X?"
  - NL patterns: fibonacci(N), factorial(N), collatz_length(N), nth_prime(N), next_prime(N), factorize(N), gcd(A,B), lcm(A,B)
  - Boolean: "Is X prime?", "Is X perfect?", "Is X divisible by Y?"
- Pre-computed values injected as `"Verified facts: X = Y"` in system prompt
- Model sees correct values upfront → uses them directly → no arithmetic errors
- `_verify_prompt_answer()` cross-checks the model's answer against pre-computed value
- Multi-turn correction: if prompt check fails, retries with correction message (max 1 retry)
- Falls back to appending `[Auto-CALM correction: ...]` note if retry also fails

### Key Decisions (with reasoning)

1. **Post-hoc over mid-stream** — can't inject mid-generation with non-streaming API. Verify after the fact, retry if wrong. Works because most claims are correct; we only correct the few wrong ones.

2. **Pre-computation is the biggest win** — injecting verified facts into the system prompt eliminates the model's arithmetic weakness entirely. fibonacci(30) goes from FAIL (model computes wrong) to PASS in 2.3s (model reads the fact). Collatz(27) goes from 165s (manual computation) to 2.3s.

3. **LaTeX stripping, not LaTeX parsing** — the model writes LaTeX math (`$17 \times 23 = 391$`, `\mathbf{100,283}`, `$$...$$`). Instead of teaching the regex to handle every LaTeX variant, strip formatting before extraction. Simpler, more robust.

4. **Don't strip single `*`** — markdown `**bold**` stripping is safe, but single `*italic*` conflicts with multiplication `17 * 23`. Only strip `**`.

5. **Conditional context filtering** — "if 391 is prime" is a question, not a claim. Regex lookbehinds don't work (engine matches mid-number). Solution: scan the 50 chars before each match for conditional words (if, whether, check, determine, test, verify).

6. **Integer division awareness** — "54 ÷ 7 = 7 remainder 5" is correct (integer division). Without this, Auto-CALM "corrects" to 7.714... which corrupts the model's long division work-showing.

7. **No cross-reference for "product is X"** — tried linking "The product is 35,818,953" back to the nearest `A × B` expression. Failed: picked wrong intermediate expression from LaTeX arrays. Deferred to prompt-level verification which is more reliable.

### False Positive Evolution (most instructive failures)

| Iteration | False Positive | Root Cause | Fix |
|---|---|---|---|
| 1 | `391 = 17` from `391 = 17 × 20 + 51` | Regex matched LHS without operator | Require operator in LHS |
| 2 | RHS followed by `\times` | Equation fragment, not standalone claim | Negative lookahead on operators |
| 3 | `391 is False` | Boolean correction replaced with value, not flipped assertion | Move boolean handler before generic handler |
| 4 | `To determine if 391 is prime` | Question context matched as assertion | Add conditional word filter |
| 5 | `100,283 \div 17` captured as `283 \div 17` | Comma in number broke LHS pattern | Add `,` to LHS digit class |
| 6 | `100,283 is perfectly divisible` | "perfectly" matched "perfect" pattern | Add `\b` after "perfect" |
| 7 | `54 \div 7 = 7 remainder 5` corrected to 7.714 | Integer division treated as float | Check "remainder" context |
| 8 | `check if the digit sum (42) is divisible by 9` | Conditional "if" too far from match | Widen prefix scan to 50 chars |
| 9 | `17 * 23 = 391 and 42 * 19 = 798` lost `*` | Markdown italic strip ate `*` operator | Only strip `**`, not `*` |

## In Progress

**Layer 3 (Intent-to-Edit) — not started.** The handoff from session 20
describes this as: detect code change descriptions in natural language,
generate actual code, apply + test. This would solve the "file write
problem" — the model describes fixes in English, the engine writes the code.

The model already demonstrates the key capability: it identifies bugs
correctly (3,857 chars of thinking in session 20), describes fixes in
natural language, knows which lines to change. It just can't format the
fix as code inside a `<calm>` block. Auto-CALM Layer 3 would bridge
that gap.

## Next Steps

1. **Wire Auto-CALM into the harness** — `agents/harness.py` should
   auto-verify claims when backend is llama.cpp. Add `/auto-calm` toggle
   command. When enabled, every model response passes through
   `AutoCalm.verify_and_correct()`.

2. **Layer 3: Intent-to-Edit** — the hardest and most impactful layer.
   Detect code change descriptions → generate code → apply → test.
   See session 20 handoff for detailed design. Key: the model describes
   "change the divide function to check for zero", the engine generates
   the actual `if b == 0: return 'Error'` and applies it.

3. **Streaming Auto-CALM** — currently uses non-streaming API (one-shot
   generate → verify). Could integrate with `stream_engine.py` to verify
   claims as tokens arrive, enabling mid-generation correction.

4. **Update CLAUDE.md and rules** — add Auto-CALM section to
   `.claude/CLAUDE.md` and `.claude/rules/calm.md`.

5. **Expand precompute patterns** — currently covers arithmetic,
   number theory functions, and boolean checks. Could add: string
   operations, date math, unit conversions, combinatorics.

## Key Context

### Engine Defaults

- **Gemma 4 E4B tq4** at 512K context, 32K thinking budget
- `~/models/gemma-4-E4B-it-tq4-aligned.gguf` (5.0 GB)
- `--cache-type-k tq4_k256 --cache-type-v tq4_k256 --parallel 1`
- ~47-50 tok/s at 8K context on RTX 4070 Laptop
- llama-server on port 8080

### What NOT to retry

- **Cross-reference "product is X"** — tried linking backward to nearest
  `A × B`. Picks wrong expression from LaTeX arrays (intermediate steps
  vs main expression). Prompt-level verification is more reliable.
- **Single `*` stripping** — conflicts with multiplication operator.
- **Regex lookbehinds for conditional context** — regex engine starts
  matching mid-number, so lookbehind sees wrong prefix.

### Architecture Note

Auto-CALM and explicit CALM coexist cleanly:
- `engine.py` / `stream_engine.py` — explicit `<calm>` blocks (v0.1/v0.2)
- `auto_calm.py` — transparent verification (no `<calm>` needed)
- Both use the same `expression.py:safe_eval()` for computation
- Both use the same `verifier.py` for TMR when applicable
- A future unified engine could use Auto-CALM as default and fall back
  to explicit CALM when the model emits `<calm>` blocks voluntarily

## Files in Project (CALM subsystem, ~8,680 LOC total)

### Auto-CALM (new this session)
- `calm/auto_calm.py` (814) — `AutoCalm` (claim extraction + verification), `AutoCalmEngine` (precompute + multi-turn correction)
- `calm/tests/test_auto_calm.py` (222) — 36 tests: extraction, boolean claims, conditional filtering, verification, correction, precompute

### Core Runtime (from session 20)
- `calm/engine.py` (527) — v0.1: hybrid thinking-plan + stop-mode
- `calm/stream_engine.py` (240) — v0.2: SSE streaming, thinking every turn
- `calm/interceptor.py` (479) — 4-tier parse + CALM block detection
- `calm/expression.py` (618) — AST-safe eval, 30+ functions
- `calm/verifier.py` (560) — 4-lane TMR verification
- `calm/stack_vm.py` (522) — reference stack machine
- `calm/sandbox.py` (250) — subprocess Python isolation
- `calm/nl_parser.py` (168) — NL → stack code translator
- `calm/grammar.py` (110) — GBNF grammar generator
- `calm/training.py` (94) — JSONL signal collector
- `calm/reasoning.py` (172) — structured chain tracker

### Backends
- `calm/backends/math_ops.py` (134) — 9 CPU math functions
- `calm/backends/string_ops.py` (92) — 7 string functions
- `calm/backends/wasm_ops.py` (174) — 17 wasm functions
- `calm/backends/code_ops.py` (310) — 16 code operations
- `calm/backends/security_ops.py` (327) — 8 security checks
- `calm/backends/calm_math.wat` (60) — WebAssembly module

### Tests + Benchmarks
- `calm/tests/` (9 files, ~3,200 LOC) — 242 tests (214 original + 28 auto-calm)
- `calm/benchmark.py` (227) — 40-problem automated eval

## Useful Commands

```bash
# Auto-CALM engine (precompute on by default)
python3 -m calm.auto_calm "What is 17 * 23? Is the result prime?"

# Run all tests (242 tests, ~4s)
python3 -m pytest calm/tests/ -v

# Run 40-problem benchmark with explicit CALM
python3 -m calm.benchmark

# Start llama-server (tq4+tq4, 512K)
~/llama.cpp/build/bin/llama-server \
    -m ~/models/gemma-4-E4B-it-tq4-aligned.gguf \
    --ctx-size 524288 --parallel 1 \
    --cache-type-k tq4_k256 --cache-type-v tq4_k256 \
    -ngl 999 --port 8080
```
