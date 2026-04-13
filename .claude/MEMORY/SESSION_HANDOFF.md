# Session Handoff — 2026-04-13 (Session 24)

## Goal

Scale CALM intelligence to 1000+ functions, fix cognitive module wiring,
sharpen quality scoring, add dynamic factual cross-checking, fix module
learning feedback loop, add hierarchical reasoning, and prototype HRM
(Hierarchical Reasoning Model) for latent-space reasoning. User directive:
"max the intelligence, scale the knowledge, hypothesis, build, test with
gemma, iterate."

## Completed

### 30 commits on `feature/multi-agent-qwen`

Session went from 65 backends/506 functions/19 broken modules to
116 backends/1002 functions/39 working modules, plus hierarchical routing,
dynamic factual cross-check, weighted quality scoring, module learning
feedback loop, and an HRM prototype.

### Backends Built (51 new, 65→116, 506→1002 functions)

**Compute backends (new `*_ops.py`):**
- `coordinate_ops` (9) — haversine, DMS, bearing, midpoint
- `statistics_ops` (16) — z-score, normal CDF/PDF, percentile, IQR, correlation
- `phonetics_ops` (6) — Soundex, Metaphone, NYSIIS, Levenshtein
- `linux_ops` (9) — chmod, umask, signals, process states, exit codes
- `encryption_ops` (7) — hash info, key sizes, password strength
- `time_ops` (12) — epoch conversion, business days, age, quarters
- `physics_ops` (22) — kinematics, forces, energy, electricity, waves
- `number_theory_ops` (14) — Euler totient, Catalan, partitions, Fibonacci check
- `logic_ops` (13) — truth tables, De Morgan's, set operations
- `graph_theory_ops` (15) — adjacency, BFS/DFS, components, cycle detection
- `calculus_ops` (11) — numerical derivative/integral, Taylor series
- `string_metrics_ops` (13) — Jaro-Winkler, Hamming, LCS, anagram/palindrome
- `set_ops` (15) — union/intersection/cartesian, Jaccard, power set
- `crypto_ops` (16) — MD5/SHA/HMAC hashing, base64, ROT13, Caesar cipher
- `boolean_ops` (17) — logic gates, adders, Gray code, parity
- `math_combinatorics_ops` (14) — derangements, Bell, multinomial, pigeonhole
- `math_trig_ops` (21) — sin/cos/tan/csc/sec/cot, law of cosines/sines
- `math_sequence_ops` (13) — arithmetic/geometric series, triangular/harmonic
- `math_number_ops` (22) — floor/ceil, lerp, power-of-2, geometric/harmonic mean
- `text_ops` (15) — word count, reading time, Flesch readability
- `format_ops` (17) — number/currency/bytes/duration/ordinal/roman formatting
- `validation_ops` (19) — email/URL/IP/UUID/credit card/ISBN validation, Luhn
- `calendar_ops` (14) — day of week, Easter, zodiac, week number
- `units_ops` (16) — 50+ unit conversions, BMI, Mach, dB↔ratio

**Knowledge backends (new `*_kb.py`):**
- `currency_kb` (8) — 155 ISO 4217 codes + symbols + decimals
- `measurement_kb` (6) — SI prefixes, base/derived units, conversions
- `sql_ref_kb` (6) — JOIN types, window functions, isolation levels
- `git_ref_kb` (6) — 25 commands, reset modes, merge vs rebase
- `music_kb` (9) — note frequencies, chords, scales, intervals
- `chemistry_kb` (7) — 36 molecules, functional groups, mole↔gram
- `networking_kb` (6) — OSI model, 15 protocols, TCP vs UDP, DNS records
- `data_structures_kb` (4) — 15 data structures with complexity
- `http_ref_kb` (6) — Cache-Control, CORS, security headers, content types
- `programming_kb` (6) — SOLID, paradigms, anti-patterns, principles
- `docker_kb` (5) — Dockerfile instructions, compose keys, best practices
- `regex_common_kb` (4) — 18 tested regex patterns
- `aws_kb` (4) — 24 AWS services with details
- `security_kb` (5) — OWASP Top 10, auth methods, vulnerability types
- `database_kb` (7) — ACID, CAP, normal forms, index types
- `testing_kb` (6) — 12 test types, patterns, coverage types
- `sorting_kb` (4) — 12 sorting algorithms with complexity
- `algorithms_kb` (7) — search, DP, greedy, graph, NP problems
- `encoding_ref_kb` (7) — 10 encodings, escape rules, BOM, line endings
- `api_patterns_kb` (9) — REST/GraphQL/gRPC, pagination, HTTP methods
- `cloud_patterns_kb` (7) — circuit breaker, saga, 12-factor, fallacies
- `compiler_kb` (7) — compilation stages, grammars, parsing, execution models
- `devops_kb` (7) — deployment strategies, CI/CD, SRE concepts
- `web_kb` (7) — semantic HTML, CSS layout/units, browser storage
- `type_system_kb` (6) — type systems, variance, common type patterns
- `concurrency_kb` (6) — concurrency models, sync primitives, async patterns
- `color_theory_kb` (16) — named colors, models, harmonies, WCAG contrast

### NL Pattern Coverage: 125→550, 54%→100%

Added NL patterns to all 32 backends that were missing them (session start),
then added patterns to all 51 new backends. Every backend now exports
`*_NL_PATTERNS`. Precompute can fire across all 116 backends.

### Cognitive Modules: 19 broken → 39 working

1. **Registered 17 missing modules** (commit `0eee704`): analogy, counterfactual,
   abstraction, creativity, evidence, compression, error_recovery, calibration,
   judgment, metacognition, goal_tracking, uncertainty, prerequisites, prioritize,
   constraints, conflict_resolution, provenance

2. **Fixed all interface mismatches** (commits `2116643`, `d13d51b`):
   - compression: `compress` not `analyze`
   - metacognition: 3 args not 1
   - judgment: `JudgmentEngine` not `JudgmentFramework`
   - prioritize: `Prioritizer.rank_from_text`
   - counterfactual: `analyze` not `generate`
   - error_recovery: 1 arg not 2
   - chain_verify: guard None wrong_steps
   - conflict_resolution: rewrote to detect textual tensions
   - 7 modules: fixed bound-method-as-string bug in summary extraction

3. **Built 3 new modules** (commits `04ae45a`, `a8bd024`):
   - `factual_check`: 48 static misconception patterns + 10 dynamic cross-check
     patterns that verify claims against backend functions at runtime
   - `confidence_check`: detects overconfidence (absolutes, false certainty)
   - `specificity`: detects generic platitudes ("use caching", "add indexes")

### Quality Scoring Reform (commit `4fee43a`)

Old: simple average where modules finding 0 issues scored 1.0. A nearly-all-wrong
response scored 92%.

New: weighted scoring — verification/planning modules that find issues get 3×
weight, quality/reasoning get 2×, meta gets 1.5×. Issue penalty increased from
0.15 to 0.20 per issue.

Result: bad response dropped from 92%→70%, good response 97%→93%. Gap: 23%.
Self-heal threshold (75%) now triggers on genuinely bad responses.

### Dynamic Factual Cross-Check (commit `054d477`)

`factual_check.py` now has two passes:
1. **Static**: 48 regex patterns for known misconceptions
2. **Dynamic**: 10 cross-check patterns that call backend functions at runtime
   (hash_output_length, which_layer, currency_decimals, molecular_weight,
   note_frequency, country_capital)

Example: response says "SHA-256 output is 32 bytes" → backend computes
`hash_output_length("sha256")` = 64 → flags the error.

### Module Learning Feedback Loop (commit `054d477`)

**Fixed**: `record_from_report()` was keying on raw summary strings like
`"precise (95%), 3 vague terms"` — unique every time, frequencies never
accumulated. Fixed: strip numbers/percentages from keys.

**Added**: prevention rules for confidence_check, specificity, factual_check.

**Verified**: 3 similar prompts → patterns accumulate → prevention suggestion
injected into next prompt's system prompt.

### Engine V2 — _raw_prompt Fix (commit `054d477`)

`_enrich_system_prompt` was calling `suggest_prompt_additions("")` because
`_raw_prompt` was never set in pre_analysis. Fixed: `pre_analysis["_raw_prompt"] = prompt`.

### Hierarchical Reasoning (commit `34ba479`)

New phase 2.7 in Engine V2: before generating, decompose multi-part prompts
into sub-problems and route computable ones to backends.

- `calm/hierarchical.py` (~180 lines): HierarchicalRouter, RoutingPlan, RoutedStep
- Backend routing: tries direct function calls (all_acid, ds_info, sort_info, etc.)
- Precompute matching: connects precomputed facts to sub-questions
- `calm/decompose.py`: multi-question prompts (2+ ?) now use structural
  decomposition instead of template matching

Gemma test: "Compare Redis vs PG? ACID properties? PG port?" → 3 sub-problems:
ACID answered by `all_acid()`, port by precompute, comparison sent to model.

### HRM Prototype (latent-space reasoning)

Built from scratch based on arxiv.org/abs/2506.21734 (Wang et al., 2025):

- `calm/hrm/model.py` (~200 lines): HRM model with nested L/H recurrent loops,
  RoPE, SwiGLU, RMSNorm. Standard PyTorch, no Flash Attention dependency.
- `calm/hrm/data.py` (~100 lines): generates math problems using CALM backends,
  character-level tokenization, masked answer format for latent prediction
- `calm/hrm/train.py` (~150 lines): training loop with cosine LR, masked loss,
  checkpointing
- `calm/hrm/inference.py` (~50 lines): load checkpoint, run inference on CPU

**Training results** (math domain, 101K params, RTX 4070):
- First attempt (next-token prediction): 99.9% val accuracy but inference broken
  — model learned to predict next token, not fill in blanks
- Second attempt (masked answer prediction): 45% val accuracy — correct formulation
  but needs more capacity/training
- Third attempt (128 hidden, 2 L-layers): **interrupted by user — not completed**

### Docs Updated

- `CLAUDE.md`: all counts updated to 116/1002/550/39
- `rules/architecture.md`: backend/function/module/LOC counts
- `rules/calm.md`: backend header, file map, Engine V2 pipeline, module table
- `rules/commercial.md`: backend count
- `rules/workflow.md`: CALM measurement patterns, multi-domain smoke test,
  CALM iteration pattern section

### Bug Fixes

- `precompute.py`: crash on None-template NL patterns (commit `a627334`)
- `linux_ops.py`: chmod_to_symbolic treating octal as decimal (commit `a9e9f59`)
- `chemistry_kb.py`: name-based molecule lookup (commit `fc8893d`)
- `decompose.py`: multi-question prompts losing sub-questions to template match

## In Progress

### HRM Training — Math Domain

The HRM model (`calm/hrm/`) is built and the training pipeline works.
The masked-answer format (model predicts answer in blank positions) is
correct but needs tuning:

- **101K params (hidden=64)**: 45% accuracy — too small for the task
- **~400K params (hidden=128, L=2)**: training was started but interrupted
- A checkpoint exists at `calm/hrm/checkpoints/math_hrm_best.pt` (from
  the 45% run — not useful for inference yet)

**What to try next**:
1. Hidden=128, L_layers=2, H_layers=2, epochs=2000, lr=5e-4
2. If still low: increase to hidden=256 (~3.4M params)
3. Consider: the masked prediction task may be too hard for this size.
   Alternative: autoregressive generation (the model predicts answer
   tokens one by one) — this worked at 99.9% with the first approach
   but the inference code needs to be autoregressive too.

### Uncommitted Changes

- `calm/learned_patterns.jsonl` — frequency bumps from Gemma testing
- `.claude/agents/` and `.claude/commands/` deletions (pre-existing)
- `rust/crates/api/src/{client.rs, providers/mod.rs}` — Ollama match arms
  (user deferred from session 22)

## Next Steps — Priority Order

### Priority 1: Fix HRM Training

The masked prediction approach (predict answer in blank positions) may be
wrong for this architecture. Two options:

**Option A**: Make inference autoregressive — the first training run (next-token
prediction) hit 99.9% accuracy. Fix the inference code to generate token by
token instead of predicting all at once.

**Option B**: Scale up the model for masked prediction — try hidden=128 or
hidden=256. The masked task is harder but more "latent" (true HRM style).

Decision point: does the user want true latent reasoning (masked, option B)
or practical accuracy first (autoregressive, option A)?

### Priority 2: Wire Engine V2 into Harness

Every `zenith` conversation should run through Engine V2 (pre-analysis,
precompute, cognitive routing, self-heal). Currently only works via
`python3 -m calm.engine_v2`. Changes needed in `agents/harness.py` and
`agents/agent.py`.

### Priority 3: More Domain HRM Models

Once math HRM works, train domain-specific models:
- Logic/constraints: ~650K params, 30 min training
- Code patterns: ~3.4M params, 1-2 hours
- Planning: ~3.4M params, 2-3 hours

### Priority 4: Cognitive Benchmark

Build a 40-problem benchmark for cognitive modules (like the math benchmark).
Known-bad responses with specific failure modes, measure detection rate.

### Priority 5: Push to Remote

76+ commits ahead of origin. User hasn't pushed yet.

## Key Context

### Architecture Insight: Five-Level Intelligence Stack

```
Level 0: Backends (1002 functions) — instant, verified, deterministic
Level 1: Precompute (550 NL patterns) — extract + inject before generation
Level 2: Hierarchical routing — decompose, route computable to L0/L1
Level 3: Model (Gemma 4B) — reasoning in token space
Level 4: Cognitive modules (39) — verify, score, self-heal after generation
Level 5: HRM (prototype) — latent-space reasoning for structured problems
```

### HRM Theory: Why It Fits (theoretical)

HRM fills the gap between backends and the LLM — problems that are
**structured enough to iterate on** but **too complex to compute directly**.

```
"What is 17 * 23?"       → Backend (deterministic, instant)
"Solve 6-constraint CSP" → HRM (latent iteration, ~1ms, 100K-3M params)
"Is this code buggy?"    → Gemma (token reasoning, ~30s, 4B params)
```

Backends can't do combinatorial search. The LLM approximates it in tokens
(slow, unreliable). HRM iterates in hidden state until constraints satisfy —
the L-module handles local computation, H-module guides the search, and
the outer loop refines until convergence. Same pattern as CALM's verify →
self-heal loop, but learned in a neural network instead of coded in Python.

Practical value: HRM at 100K-3M params runs in <1ms on CPU. Doesn't
compete with Gemma for VRAM. Each domain needs its own trained model, but
CALM backends generate the training data (verified problem/solution pairs).
The flywheel: backends generate data → HRM learns patterns → HRM handles
novel problems backends can't solve → CALM verifies HRM outputs.

In `hierarchical.py` routing:
```
sub-problem → computable?           → backend (L0)
           → structured reasoning?  → HRM (latent, L5)
           → open-ended?            → Gemma (tokens, L3)
```

### Performance Budget

The entire CALM pipeline (1002 functions, 550 patterns, 39 modules) adds
~172ms to inference. Model inference is 20-40 seconds. CALM overhead is
0.57% — effectively free. Backends don't cost performance, they BUY it
by preventing the model from wasting tokens on computable problems.

### Quality Discrimination

- Bad response (deliberate errors): 70% quality, 21 issues caught
- Good response (correct, specific): 93% quality, 4 minor issues
- Gap: 23% (was ~0% before weighted scoring)
- Self-heal threshold: 75% — bad responses now trigger correction loop

### Gemma Test Results (session 24)

9 test prompts covering: Rust, VRAM math, git hooks, quantization,
sliding window attention, Python async, CUDA shared memory, WSL2 memory,
llama-server flags. Key finding: llama-server `--parallel` question was
answered WRONG (model said "distributes across CPU cores" — actually
divides ctx_size into slots). This is the gap that project-specific KBs
would fix.

### Module Learning Normalization

Keys must be stripped of variable data before accumulating:
`"precise (95%), 3 vague terms"` → `"precise (), vague terms"`.
Without this, frequencies never reach threshold and suggestions never fire.

### Serving (unchanged from session 23)

- Gemma 4 E4B tq4 at 512K context, llama-server on port 8080
- `~/models/gemma-4-E4B-it-tq4-aligned.gguf` (5.0 GB)
- `--cache-type-k tq4_k256 --cache-type-v tq4_k256 --parallel 1`
- ~45-48 tok/s

## Useful Commands

```bash
# Check function count
python3 -c "from calm.expression import _FUNCTIONS; print(len(_FUNCTIONS))"

# Check NL pattern count
python3 -c "from calm.backends import NL_PATTERNS; print(len(NL_PATTERNS))"

# Check cognitive module count
python3 -c "from calm.router import CognitiveRouter; r=CognitiveRouter(); print(len(r._modules))"

# Run Engine V2 on a prompt
python3 -u -c "
from calm.engine_v2 import CalmEngineV2
engine = CalmEngineV2(thinking_budget=8192)
r = engine.run('Your prompt here', verbose=True)
print(r.response[:500])
print(r.summary())
"

# Test quality discrimination (bad vs good response)
python3 -c "
from calm.router import CognitiveRouter
router = CognitiveRouter()
router._max_modules = 50
r = router.analyze('prompt', 'bad response with always never everything', '')
print(f'Quality: {r.overall_quality:.0%}, Issues: {r.total_issues}')
"

# Train HRM (math domain)
python3 -m calm.hrm.train --epochs 2000 --hidden 128 --lr 5e-4

# Run all tests
python3 -m pytest calm/tests/ -v

# Start llama-server
setsid ~/llama.cpp/build/bin/llama-server \
  -m ~/models/gemma-4-E4B-it-tq4-aligned.gguf \
  --ctx-size 524288 --parallel 1 \
  --cache-type-k tq4_k256 --cache-type-v tq4_k256 \
  -ngl 999 --port 8080 < /dev/null > /tmp/llama-server.log 2>&1 &
```

## Session Stats

- **30 commits** this session
- **~37,400 LOC** total CALM engine (was ~26,300)
- **116 backends** (was 65), **1002 functions** (was 506)
- **550 NL patterns** (was 125), **100% coverage** (was 54%)
- **39 cognitive modules** (was 19 broken), **0 errors**
- **48 factual patterns** + **10 dynamic cross-check** (was 0)
- **Weighted quality scoring**: 23% gap bad vs good (was ~0%)
- **Self-heal confirmed working**: triggers at < 75%
- **Hierarchical routing**: decompose → route → compose (new)
- **HRM prototype**: 101K param model built, training pipeline working
- **250 tests pass** (no regressions)
