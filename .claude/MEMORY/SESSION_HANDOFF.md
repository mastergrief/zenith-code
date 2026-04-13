# Session Handoff — 2026-04-13 (Session 23)

## Goal

Scale CALM's intelligence by building more backends, cognitive modules,
and NL patterns. Then build an Engine V2 pipeline that integrates
everything into a self-healing quality loop. User directive: "max the
intelligence, scale the knowledge, and when we get to frontier parity
then we wire the engine into the harness."

## Completed

### 22 commits on `feature/multi-agent-qwen`

Session went from 36 backends/299 functions to 65 backends/506 functions,
plus 41 cognitive modules, Engine V2, cognitive router, adaptive thinking,
cross-turn state, and module learning. All from scratch this session.

### Backends Built (29 new, 299→506 functions)

**Compute backends (19 new `*_ops.py`):**
- `http_ops` (7) — status codes, methods, MIME types
- `uuid_ops` (8) — generate, validate, parse
- `csv_ops` (9) — parse, validate, column stats
- `markdown_ops` (7) — headers, TOC, code blocks, links
- `unicode_ops` (7) — codepoints, categories, confusables
- `color_ops` (9) — hex/RGB/HSL, WCAG contrast, complement
- `jwt_ops` (7) — decode header/payload, validate structure
- `timezone_ops` (7) — convert, UTC offset, DST
- `baseconv_ops` (9) — binary/octal/hex/arbitrary base
- `checksum_ops` (8) — Luhn, ISBN-10/13, EAN, UPC
- `bytesize_ops` (7) — human-readable, IEC vs SI
- `duration_ops` (7) — parse "2h30m", ISO 8601, convert
- `geometry_ops` (19) — circle, sphere, cone, trapezoid, distance
- `probability_ops` (11) — dice, coin, binomial, Bayes
- `roman_ops` (3) — Roman ↔ decimal, validation
- `financial_ops` (10) — compound interest, loan payments, NPV, ROI
- `ratio_ops` (9) — simplify fractions, percent change
- `cidr_ops` (8) — subnet mask, host count, IP-in-subnet
- `matrix_ops` (11) — determinant, multiply, transpose, dot/cross product

**Knowledge backends (10 new `*_kb.py`):**
- `country_kb` (8) — 195 countries: capitals, ISO, currencies
- `elements_kb` (9) — 118 elements: symbols, weights, electron config
- `constants_kb` (5) — CODATA 2018 physical constants
- `complexity_kb` (5) — sort/DS/graph Big-O
- `port_kb` (5) — 45 well-known ports
- `ascii_kb` (7) — control chars, escape sequences, CR vs LF
- `license_kb` (5) — 12 SPDX licenses, copyleft, compatibility
- `regex_ref_kb` (4) — 20 common patterns + syntax reference
- `error_code_kb` (4) — exit codes, POSIX errno, Unix signals
- `design_pattern_kb` (5) — 22 GoF + modern patterns

### Architecture Improvements

1. **Auto-discovery registry** (`calm/backends/__init__.py`):
   Scans `*_ops.py` + `*_kb.py`, registers `*_FUNCTIONS` + `*_NL_PATTERNS`.
   Adding a backend = write the file, done. expression.py: 936→657 LOC.

2. **Auto-collected NL patterns**: backends export `*_NL_PATTERNS` lists.
   Precompute iterates them. 120→125 patterns across 24 backends.

3. **`_kb.py` naming convention**: knowledge backends include
   `_DATA_VERSION` for staleness tracking.

4. **Bug fixes**:
   - `auto_learn.py`: `factorial(credit_card_number)` infinite hang → >10M guard
   - `verify.py`: base conversion claim patterns (Layer 1 catches hex/binary errors)
   - `precompute.py`: binary-aware patterns ("binary 10110011 to hex" → b3, not 9a443b)

### 41 Cognitive Modules (all new this session)

| Layer | Modules |
|-------|---------|
| **Verification** | chain_verify, consistency, logic, scope |
| **Reasoning** | decompose, causal, assumptions, analogy, temporal, counterfactual, hypothesis_gen |
| **Quality** | creativity, nuance, evidence, relevance, completeness, explanation, density, precision, compression, error_recovery |
| **Meta-cognitive** | calibration, judgment, metacognition, goal_tracking, abstraction, perspective, uncertainty, communication, prerequisites |
| **Planning** | prioritize, constraints, risk, disambiguation, provenance, conflict_resolution |

### Engine V2 (`calm/engine_v2.py`)

7-phase pipeline with self-healing:
1. PRE-ANALYZE: profile expertise, detect ambiguities, decompose, assess risks
2. ENRICH: inject pre-analysis into system prompt
3. ADAPTIVE BUDGET: trivial=2K, easy=4K, medium=8K, hard=16K, deep=32K
4. PRECOMPUTE: inject verified backend facts (500 functions + 125 NL patterns)
5. GENERATE: model responds with enriched context
6. VERIFY + COGNITIVE ROUTE: Auto-CALM claims + 41 modules (33-70ms)
7. SELF-HEAL: if quality < threshold, targeted correction → re-verify

### Supporting Systems

- **Cognitive Router** (`calm/router.py`): auto-selects relevant modules
  per prompt. Simple math → 6 modules. Architecture → 10+.
- **Adaptive Thinking** (`calm/adaptive.py`): 5 tiers based on complexity
  + precompute coverage. Precomputed answer → 2K (saves 4-16x).
- **Conversation State** (`calm/conversation.py`): cross-turn consistency,
  calibration, goal tracking, provenance.
- **Module Learning** (`calm/module_learning.py`): records recurring issues
  from router outputs → proactive prompt prevention.

### Gemma Test Results (8K thinking budget, session averages)

| Prompt Type | Quality | Corrections | Avg Time |
|-------------|---------|-------------|----------|
| Factual with precompute | 95-100% | 0 | 9-25s |
| Comparison (Redis vs PG) | 89-92% | 0 | 40-50s |
| Debugging scenario | 92% | 0 | 25-40s |
| Architecture design | 92-100% | 0 | 30-74s |
| Math (fractions) | 97% | 2 corrected | 25s |

Key finding: pre-analysis enrichment raises first-pass quality so much
that self-healing rarely triggers — prevention > correction.

## In Progress

### Uncommitted

- `calm/backends/matrix_ops.py` — written, not tested or committed (11 funcs:
  determinant, multiply, transpose, inverse, trace, dot/cross product)
- `calm/learned_patterns.jsonl` — frequency bumps from Gemma testing
- `rust/crates/api/src/{client.rs, providers/mod.rs}` — Ollama match arms
  from session 22 (user deferred full Ollama removal)

## Next Steps — Priority Order

### Priority 1: Keep Scaling (user directive)

More backends, modules, NL patterns. Build and test with Gemma. Ideas:

**Backends to build next:**
- `coordinate_ops` — lat/long, DMS conversion, haversine distance
- `statistics_ops` — z-score, percentile, normal distribution, chi-squared
- `phonetics_ops` — Soundex, Metaphone for fuzzy matching
- `currency_kb` — ISO 4217 codes, symbols, decimal places
- `measurement_kb` — SI prefixes, unit relationships
- `sql_ref_kb` — SQL syntax reference (JOIN types, window functions)
- `git_ref_kb` — git commands reference

**Cognitive modules to build:**
- `attention_allocation` — which part of a long prompt matters most
- `semantic_similarity` — measure how similar two texts are
- `transfer_learning` — cross-domain pattern recognition
- `socratic` — ask questions instead of answering (teaching mode)
- `context_awareness` — warn when conclusions depend on compacted info

**NL patterns to add:** 40+ backends still missing NL patterns. Run the
check: `python3 -c "..."` (see session for the one-liner).

### Priority 2: Wire Engine V2 into Harness

User directive: "when we get to frontier parity, wire the engine into
the harness." Changes needed:

- `agents/harness.py`: import Engine V2, wrap response generation
- `agents/agent.py`: pass response through router post-generation
- Add `/cognitive` command to show quality report
- Add `/quality` toggle to enable/disable cognitive analysis
- Wire adaptive thinking into effort modes

### Priority 3: Cognitive Benchmark

40-problem benchmark for cognitive modules (like the math benchmark):
- 10 scope/generalization tests (known overgeneralized responses)
- 10 disambiguation tests (ambiguous prompts with known interpretations)
- 10 completeness tests (multi-part questions)
- 10 explanation quality tests (circular, jargon-heavy)

### Priority 4: Update Session Handoff

Update CLAUDE.md backend count + cognitive module table after more
building. Currently accurate as of commit `bec0b9d`.

### Priority 5: Rust Cleanup

- Remove `ProviderKind::Ollama` entirely (user deferred)
- Fix warnings: unused imports in `session_control.rs`, dead code
- Run `cargo clippy --workspace --all-targets -- -D warnings`

## Key Context

### User Preferences (from memory)
- Works directly — no subagent dispatch (memory: feedback_no_agents.md)
- Hypothesis → test → iterate workflow for everything
- Test with Gemma, not just unit tests
- Concise communication, no fluff
- Defers non-critical work ("for another day")

### Architecture Insight: "Intelligence from Architecture"
The system produces 89-100% quality from a 4B model on 8GB VRAM by:
1. Backends compute what the model can't (500 functions, instant)
2. Pre-analysis tells the model things it can't see (expertise, risk, ambiguity)
3. 41 cognitive modules catch 41 specific failure modes (33-70ms)
4. Self-healing corrects what slips through (only when improvement is verified)
5. Module learning prevents recurring issues proactively

Key quote from the user: "do you think its possible to build knowledge
backends?" — YES. Knowledge backends (`_kb.py`) work identically to compute
backends. Same contract, same verification, same precompute. The engine
doesn't care if `f(x)` computes or looks up.

### Bugs Found & Fixed
- `factorial(credit_card_number)` — auto_learn tried to compute factorial
  of 16-digit CC numbers. Fixed with >10M guard in `auto_learn.py`.
- `to_hex(10110011)` vs `base_convert("10110011", 2, 16)` — precompute
  treated binary numbers as decimal. Fixed with binary-aware patterns.
- Python output buffering — even with `-u`, redirected stdout is fully
  buffered. Need `stdbuf -oL` + `flush=True`.
- Communication adapter misclassified experts — "amortized complexity"
  triggered beginner signals ("what is"). Fixed: expert vocab overrides
  beginner phrasing.

### Serving (unchanged)
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

# Find backends missing NL patterns
python3 -c "
import os
for f in sorted(os.listdir('calm/backends')):
    if f.endswith(('_ops.py','_kb.py')) and '_NL_PATTERNS' not in open(f'calm/backends/{f}').read():
        print(f'  {f}')
"

# Run Engine V2 on a prompt
python3 -u -c "
from calm.engine_v2 import CalmEngineV2
engine = CalmEngineV2(thinking_budget=32768)
r = engine.run('Your prompt here', verbose=True)
print(r.response[:500])
print(r.summary())
"

# Run cognitive router standalone
python3 -c "
from calm.router import CognitiveRouter
router = CognitiveRouter()
report = router.analyze('prompt', 'response')
print(report.summary())
"

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

- **22 commits** this session
- **~26,400 LOC** total CALM engine (was ~15,100)
- **65 backends** (was 36), **506 functions** (was 299)
- **125 NL patterns** (was 39)
- **41 cognitive modules** (was 0)
- **Engine V2**, cognitive router, adaptive thinking, conversation state, module learning — all new
- **250 tests pass**
