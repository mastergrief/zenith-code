# Code Reasoning DB — 8970-example substrate corpus + generator framework

R53 Phase 1 deliverable. Powers `CodeVerifierFacade` retrieval and is
training signal for the code PT (R53.5). Two parts:

1. **CodeExampleDB** — in-memory retrieval corpus, 8970 unique
   examples across 10 sources, deduped on problem hash
2. **DomainDataGenerator framework** — 9 concrete generators
   producing 222 verified (problem, solution, tests) examples
   with behavioral verification via sandbox

## DB composition (8970 unique after dedup)

Loaded in priority order — first occurrence of a problem wins dedup.
Quality/relevance decreases down the list; dedup ensures higher-
quality sources own their examples.

| Source | Count | File | Notes |
|---|---:|---|---|
| coding_reasoning_claude | 547 | `agents/distill/data/coding_reasoning_claude.jsonl` | hand-written Claude-authored |
| claude_reasoning | 910 | `claude_reasoning.jsonl` | merged + filtered HF |
| MBPP | 974 | `mbpp.jsonl` | all 4 splits, every example has test_list |
| HumanEvalPlus | 164 | `humanevalplus.jsonl` | canonical Python benchmark |
| BigCodeBench | 1140 | `bigcodebench.jsonl` | v0.1.4, multi-library coordination |
| CodeContests (Py3) | 2498 | `codecontests.jsonl` | Google DeepMind, filtered to Python3 solutions |
| Crownelius | 265 | `crownelius.jsonl` | Opus-reasoning, quality-filtered |
| TeichAI | 886 | `claude_reasoning_hf_raw.jsonl` | Claude Opus reasoning |
| Nohurry (code-only) | 106 | `nohurry_code.jsonl` | category=='code' filtered from 3000 |
| Multi-step synthetic | 48 | `multi_step_code.jsonl` | original template catalog |
| Generator output | 222 | `generated/*.jsonl` | 9 generators, all sandbox-verified |
| Language-specific | 117 | `python.jsonl` + `typescript.jsonl` + `rust.jsonl` | 9B-generated, lowest priority |
| Prefilter (long-tail) | residual | `claude_reasoning_prefilter.jsonl` | 2940 raw HF, residual after dedup |

Quality gate (`scripts/r53_fetch_corpora.py:passes_quality`):

- problem >= 20 chars
- solution >= 100 chars
- solution contains code markers (```, `def `, `class `, `import `,
  `function(...) {`)

Applied to HF fetches; hand-written + generator sources already clean.

### Index caches at `.cache/r53_code_db/`

- `tfidf.json` (36 MB) — BM25-ready TF-IDF index, 68,334 vocab
- `dense.pt` (46 MB) — fp16 dense vectors [8970, 2560]
- `dense.tq4.pt` (12 MB) — tq4-quantized dense vectors

## CodeExampleDB API

```python
from calm.llm_computer.facades.code_example_db import CodeExampleDB

db = CodeExampleDB.load_default()          # 8970 unique, dedup automatic
db.build_tfidf()                            # include_solution=True default
db.build_dense(m, tok)                      # needs daemon + GPU
db.save_indices(cache_dir, dense_quantize=True)
db.load_indices(cache_dir, prefer_tq4=False)

hits = db.retrieve("write is_prime", k=3, mode="hybrid",
                   dense_m=m, dense_tok=tok)
# Each hit: .example (CodeExample), .score (float)
```

See `retrieval.md` for retrieval-mode details.

## DomainDataGenerator framework

Base class at `calm/llm_computer/facades/data_generators/base.py`
with `VerifiedExample` dataclass:

```python
@dataclass
class VerifiedExample:
    problem: str
    signature: str
    solution: str
    test_cases: List[Tuple]
    reasoning: str                        # <think> block or synthesized
    algorithm: str
    complexity: str
    edge_cases: List[str]
    category: str
    generator_name: str
    skip_sandbox: bool = False
    metadata: Dict[str, Any] = ...
```

Three output sinks (each example can flow to all three):

- `to_messages_jsonl_record()` → messages-schema JSONL for DB ingest
- `to_pt_training_record()` → `{prompt, target}` chars for
  `CopyAugmentedTransformer` training (R53.5)
- `to_kb_entry()` → `(fn_name, callable)` for optional *_kb.py compile

`DomainDataGenerator.generate(n)` runs subclass's `generate_raw(n)`,
filters via base verification, returns up to n unique passing examples.

### Verification pipeline

```
raw example
  ↓
AST parse (always)           calm/backends/ast_ops.py
  ↓
function name check          must have expected def/class
  ↓
if skip_sandbox: accept (AST-only)      else:
  ↓
generate test harness        [test_cases] → try/except + print("PASS"/"FAIL")
  ↓
sandbox run_python (5 s timeout)        calm/sandbox.py
  ↓
require all PASS             reject on any FAIL or runtime error
  ↓
domain extra_verify()        subclass override for domain checks
  ↓
accept
```

`skip_sandbox=True` for problems using sandbox-blocked modules
(`urllib`, `threading`, `secrets`, `time.monotonic`-dependent).
Those ship AST-validated; correctness is by construction.

## Concrete generators (9 files, 222 verified examples)

| Generator | File | Examples | Coverage |
|---|---|---:|---|
| `algorithms` | algorithm_problems.py | 48 | 40 hand-curated templates: arithmetic, strings, algorithms, parsing |
| `param_math` | parameterized_math.py | 85 | Heavy sweeps: divisibility (39 k), power bases, GCD specializations, digit-base sums |
| `stdlib` | stdlib_usage.py | 21 | collections / itertools / functools / pathlib / bisect / enum / json idioms |
| `bug_fix` | bug_fix_pairs.py | 17 | canonical Python pitfalls: mutable default, late binding, float equality, bare except, dict mutation, etc. |
| `security` | security_patterns.py | 13 | OWASP-adjacent: SQLi, SSRF, path traversal, XSS, HMAC, rate limiting, API key format |
| `regex` | regex_patterns.py | 9 | email, IPv4, UUID v4, hex color, ISO date, semver, etc. |
| `data_structures` | data_structures.py | 9 | stack, queue, reverse, BST, heap top-k, trie, union-find, LRU, island count, cycle detect |
| `datetime_utils` | datetime_utils.py | 10 | days_between, leap_year, add_months with clamp, UTC offset, ISO duration parse, quarter |
| `functional` | functional_patterns.py | 10 | compose, pipe, memoize, once, curry, partition, flatten_iter, tap, zip_with, frequencies |

Registration via `register_generator(name, cls)` in each file's
module load. `__init__.py` eagerly imports to populate registry.

## Scripts

`scripts/r53_fetch_corpora.py` — HTTP fetch from HuggingFace
datasets-server API. Sources: MBPP, HumanEvalPlus (evalplus variant),
Nohurry (code category filter), Crownelius (retry with backoff),
CodeContests (Python3 language code 3 only), BigCodeBench (v0.1.4).
Each source has its own converter; quality gate applies uniformly.
Rate-limit handling with 5s/attempt backoff.

`scripts/r53_run_data_generators.py` — one-command rebuild. Walks
the generator registry, invokes each, writes `generated/<name>.jsonl`
+ `generated/pt_<name>.jsonl`. Rebuilds DB from `DEFAULT_CORPORA` +
generator output, runs `build_tfidf`. Daemon variant
(`scripts/r53_build_dense.py`) additionally runs `build_dense`.

Flags:
- `--n N` max per generator (default 1000, limited by hand-curated template count)
- `--skip-dense` CPU-only run
- `--dense-only` skip regeneration, just rebuild dense
- `--out DIR` cache directory (default `.cache/r53_code_db`)

## Format contamination finding (critical fix)

**Problem**: raw DB entries in `claude_reasoning*.jsonl` and generator
output contain `<think>` blocks + prose + fenced code. The original
`CodeExample.solution_preview` stripped only `<think>` but left prose
headers ("Here's my solution:" etc).

When prepended as "related past solutions" hints, Gemma imitated the
PROSE STYLE — emitted `<think>` blocks of its own or wrote explanatory
prose instead of clean code. My format-agnostic extractor then
returned nothing, scoring 0/0 on `r53_eval_complex.py`.

**Fix** (this session): `CodeExample.solution_preview` now:

1. Strips `<think>...</think>` blocks (greedy regex)
2. Strips trailing `**Verified test cases` / `**Sample I/O` /
   `**Unit tests` / `**Test harness` sections
3. Prefers first `\`\`\`python` fenced block if present
4. Falls back to slice-from-first `def `/`class `/`import `
5. Returns raw trimmed text as last resort

**Measured effect**: problem_2 (date_validation_chain) went from
hinted 0/0 (unextractable) → 12/12 (perfect) after the fix.

### Broader rule: retrieved content must match output target

When retrieval is injected as "here's a related solution", Gemma
imitates the FORMAT, not the content. So:

- For code generation: retrieved preview must be **code-only**
- For prose tasks: retrieved preview can be prose
- Never mix the two in the same preview window

The long-term fix is two-channel retrieval (code channel + reasoning-
trace channel) with independent gating. See `retrieval.md`
§"Two-channel design".

## Invariants

- **Dedup is on problem hash** (`CodeExample.key` = blake2b of
  problem text). Earlier-loaded sources win — hand-written > HF > 9B-
  generated ordering preserved in `DEFAULT_CORPORA`.
- **Sandbox skips non-deterministic** (`skip_sandbox=True`): urllib,
  secrets, threading, time-dependent. AST-only verification for those.
- **Generator outputs are idempotent** under same seed. `scripts/r53_run_data_generators.py`
  rebuilds deterministically — safe to rerun.
- **TF-IDF v2 format** is tuple-valued postings `(raw_tf, tfidf_w)` —
  v1 (scalar `tfidf_w`) is backward-load-compatible (raw_tf defaults
  to 1 for BM25 fallback scores).

## Related rules

- `retrieval.md` — retrieval algorithms on top of this DB
- `recursion.md` — card-level self-improvement using DB + CALM oracle
- `calm.md` — CALM backends that verify generator outputs
- `training.md` — PT training on `pt_*.jsonl` files (R53.5)
- `augmentation_thesis.md` §"Tier-1 preservation" — why retrieval
  must be gated
