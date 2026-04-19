# Session Handoff — 2026-04-19 (R53 Phase 1: retrieval + DB + complex eval)

Branch: `feature/multi-agent-qwen`. Session ran R53 Phase 1 end-to-end:
hybrid retrieval stack, 8970-example DB, 9 data generators, complex
multi-step coding eval. Completed with the clearest null result we've
had in the session arc.

Full conversation transcript preserved at `.claude/MEMORY/Augment-notes.md`
(94 KB). Treat that as the raw log; extract stable findings from it and
the rules updated this session.

## TL;DR

- **Retrieval-attributable gain = +0.0pp** on complex multi-step coding
  eval. Hinted (real retrieval) = Sanity (random retrieval) = +7.4pp
  over stock. Prompt-length effect is real; retrieval content adds
  nothing on top of blanket injection.
- **Root cause of the null**: blanket retrieval injection disrupts
  Gemma's strong-prior behavior on problems it already solves. Tier 1
  preservation is violated.
- **Substrate-RAG has a structural advantage** over prompt-RAG:
  hash-gated injection (L30 KnowledgeStore fires only on hash match)
  naturally preserves Tier 1. This was proposed in R53 already; the
  eval makes it empirically sharp.
- **DB + retrieval infrastructure works end-to-end.** 8970 unique
  examples, TF-IDF+BM25 (68K vocab) + Gemma-dense (fp16 + tq4) + RRF
  fusion + tq4-quantized persistence. Cached at `.cache/r53_code_db/`.
  Rebuild via `scripts/r53_run_data_generators.py` (CPU) +
  `bin/gemma-run scripts/r53_build_dense.py` (daemon).
- **Solution-preview format contamination fixed** mid-eval: extract
  only ```python fences, strip <think> blocks and test trailers.
  Moved date_validation from hinted 0/0 → 12/12.
- **Next step is NOT more retrieval work.** It's build the substrate
  install (R53.5 PT training + R53.6 L24/L30 CardSlot) so the
  hash-gated delivery path is real, not prompt-level.

## Complex eval final results (6 problems × 3 conditions)

| problem | category | stock | hinted | sanity |
|---|---|---:|---:|---:|
| linked_list_bugs | multi_bug | 0/0 | 0/0 | 5/5 |
| date_validation_chain | multi_bug | 10/12 | **12/12** | 12/12 |
| log_level_counts | lib_compose | 6/6 | 0/0 | 6/6 |
| csv_column_stats | lib_compose | 0/0 | 0/0 | 0/0 |
| token_bucket_rate_limiter | plan_code | 0/0 | 0/0 | 0/0 |
| lru_cache_class | plan_code | 9/9 | 9/9 | 0/0 |
| **TOTAL** | | **25/27** | **21/21** | **23/23** |

- Δ hinted-vs-stock: **+7.4pp** (but numerator is smaller — hinted has
  FEWER tests running because extraction fails more often)
- Δ sanity-vs-stock: **+7.4pp** (identical — so prompt length alone
  is driving it)
- **Retrieval-attributable gain: +0.0pp**

Reading the per-row data, the honest story:
- Stock was actually the strongest path on 3/6 (logged_level, lru_cache_class, token_bucket close)
- Hinted broke extraction on 3/6 (linked_list_bugs, log_level_counts, csv)
- Sanity (random retrieval) was strongest because it didn't match-and-
  break Gemma's prior, just gave it a non-empty "related" section to
  imitate code-fence format

## Artifacts shipped this session

### Code (all committed unless noted)

- `calm/llm_computer/facades/code_example_db.py` — CodeExampleDB with
  dedup, Jaccard/TF-IDF/dense/hybrid retrieval, tq4 save/load
- `calm/llm_computer/facades/retrieval.py` — TfidfIndex (TF-IDF + BM25),
  DenseIndex (Gemma token-embd mean-pool), rrf_fuse, trie-backed fast
  tokenizer (13,000× speedup over naive BPE scan)
- `calm/llm_computer/facades/code_verifier.py` — CodeVerifierFacade
  with intent classifier + suggested-lib/security flags + compute_hints
- `calm/llm_computer/facades/data_generators/` — base.py +
  algorithm_problems, stdlib_usage, bug_fix_pairs, security_patterns,
  parameterized_math, regex_patterns, data_structures, datetime_utils,
  functional_patterns (9 generators, 222 verified examples)
- `scripts/r53_fetch_corpora.py` — MBPP / HumanEvalPlus / BigCodeBench
  / CodeContests (Python3 only) / Crownelius / Nohurry (code-category
  filter) fetcher with quality gates
- `scripts/r53_run_data_generators.py` — one-command pipeline:
  generators → DB → TF-IDF. Daemon variant adds dense build.
- `scripts/r53_build_dense.py` — dense index build via daemon (assumes
  `m`, `tok` pre-loaded)
- `scripts/r53_eval_phase1.py` — simple 12-problem eval (obsolete,
  kept for comparison; mostly at Gemma's ceiling)
- `scripts/r53_eval_complex.py` — 6 complex multi-step problems with
  3 conditions (stock / hinted / sanity-random)
- `scripts/generate_multi_step_code_data.py` — original 29-template
  catalog used by AlgorithmProblemsGenerator
- `scripts/r53_debug_gemma_output.py` — utility for debugging raw
  Gemma outputs

### Corpora fetched (at `agents/distill/data/`)

- `mbpp.jsonl` (974), `humanevalplus.jsonl` (164),
  `bigcodebench.jsonl` (1140), `codecontests.jsonl` (2498 Python3),
  `crownelius.jsonl` (265), `nohurry_code.jsonl` (106),
  `claude_reasoning_hf_raw.jsonl` (886 from TeichAI),
  `multi_step_code.jsonl` (48), `generated/*.jsonl` (222 total across
  9 generators)

### Cached indices (at `.cache/r53_code_db/`)

- `tfidf.json` (36 MB, 68,334 vocab, BM25-ready format)
- `dense.pt` (46 MB, fp16, shape [8970, 2560])
- `dense.tq4.pt` (12 MB, 4× smaller via TurboQuant)

## Conceptual findings codified this session

See `.claude/rules/retrieval.md`, `.claude/rules/code_reasoning_db.md`,
`.claude/rules/recursion.md` (new files) and the updated
`augmentation_thesis.md`, `capability_gain.md`, `calm.md`.

1. **Automatic Tier-1 preservation via hash-gated injection** —
   substrate RAG fires on match, pass-through on miss. Prompt RAG
   always fires → blanket injection violates Tier 1.
2. **Failure-surface gate** — must filter eval corpora to Gemma-
   failures first. Ceiling effect kills signal on simple problems.
3. **Retrieved content must be code-only** — `<think>` blocks leak
   into hints, Gemma imitates thinking-style instead of code-style.
4. **Two separate retrieval channels**: code for imitation, reasoning
   traces for planning. Maps to PT (L24) + KnowledgeStore (L30).
5. **Complex multi-step > simple single-shot** for measuring
   augmentation. CoT depth × per-step error rate compounds.
6. **GemmaTokenizer.encode pathology** — naive O(len × vocab) scan
   replaced with trie-backed encoder (13,000× speedup, monkey-patched
   at `retrieval.py:_monkey_patch_fast_encode`).
7. **tq4 dense embeddings** — 4× smaller storage, rank-preserving,
   one-time dequant on load.
8. **Card-level recursion via CALM oracle** — cards can self-distill
   because CALM is a deterministic verifier (not a model judge), no
   bias amplification.

## Current environment state at handoff

- Daemon running (PID 92430 at handoff time; may be killed by now)
- Gemma 4 E4B tq4 loaded with max_len=1024
  (edited in `bin/gemma_daemon.py` from 256 for R53 eval headroom)
- 6.3 GB VRAM used, ~2 GB free

## ⚠ UNCOMMITTED — commit these before any risky git ops

**All R53 code is untracked.** `git status --short` at handoff shows:

```
?? calm/llm_computer/facades/code_example_db.py       ~350 LOC
?? calm/llm_computer/facades/code_verifier.py         ~310 LOC
?? calm/llm_computer/facades/retrieval.py             ~530 LOC
?? calm/llm_computer/facades/data_generators/         9 files
?? scripts/generate_multi_step_code_data.py           ~900 LOC
?? scripts/r53_fetch_corpora.py
?? scripts/r53_run_data_generators.py
?? scripts/r53_build_dense.py
?? scripts/r53_eval_phase1.py
?? scripts/r53_eval_complex.py
?? scripts/r53_debug_gemma_output.py
?? .claude/rules/retrieval.md                         235 LOC (this session)
?? .claude/rules/code_reasoning_db.md                 220 LOC (this session)
?? .claude/rules/recursion.md                         165 LOC (this session)
?? .claude/MEMORY/Augment-notes.md                    94 KB conversation log
```

**Modified (staged-able):**

```
 M bin/gemma_daemon.py                max_len=256 → 1024 (REQUIRED for R53 evals)
 M .claude/CLAUDE.md                  R53 section added
 M .claude/MEMORY/SESSION_HANDOFF.md  this file — replaced R52 content
 M .claude/rules/augmentation_thesis.md  + Tier-1 preservation section
 M .claude/rules/calm.md                 + retrieval gating note
 M .claude/rules/capability_gain.md      + failure-surface-gate section
 M .claude/rules/tracing_roadmap.md      + R53.2b row
 M agents/distill/fetch_datasets.py      OUTPUT_FILE renamed to hf_raw
 M calm/llm_computer/tq4_triton.py       (prior R52 session — keep or revert per R52 handoff)
 M scripts/r52_train_student_kl.py       (prior R52 — see R52 handoff decision)
```

**Also present and untracked but easily regenerable:**

- `.cache/r53_code_db/` (94 MB — tfidf.json, dense.pt, dense.tq4.pt).
  Rebuild time: ~15 min via `r53_run_data_generators.py` + dense build.
- `agents/distill/data/mbpp.jsonl`, `humanevalplus.jsonl`,
  `bigcodebench.jsonl`, `codecontests.jsonl`, `crownelius.jsonl`,
  `nohurry_code.jsonl`, `claude_reasoning_hf_raw.jsonl`,
  `multi_step_code.jsonl`, `generated/*.jsonl` — all regenerable via
  `r53_fetch_corpora.py` + `r53_run_data_generators.py`.

**Suggested first action in new terminal**: `git add` the R53 code +
new rules + handoff + CLAUDE.md, commit with message referencing this
handoff, THEN continue with R53.5 work. This protects against
accidental `git stash` / `reset --hard` losing hours of work.

## Next actions

### Default recommendation: R53.5 PT training

Skip further Phase-1 retrieval iteration. The +0.0pp retrieval-
attributable gain at Phase-1 **is the signal to build the substrate
install** (where injection is hash-gated by construction).

1. **R53.5**: Train `copy_code_best.pt` (CopyAugmentedTransformer)
   on the 222 generator `pt_*.jsonl` files + reasoning-trace extracts
   from the 8970-example DB. ~185K params, ~30 min RTX 4070.
   - Target: NL → structured plan (`signature | algorithm`)
   - Gate: ≥85% autoregressive accuracy on held-out
2. **R53.6**: Install
   - PT at L24 via `CardSlot.attach(preserve=True)` — writes plan
     into reserved channels
   - KnowledgeStore recall card at L30 via `CardSlot` — hash-gated
     solution-pattern lookup
   - `CodeVerifierFacade` as `VerificationHook` — biases logits
     toward verified tokens
3. **R53.7**: Re-run complex eval with substrate install. Compare:
   - stock Gemma
   - prompt-RAG (this session's hinted)
   - substrate-RAG (R53.6 install)
   - Expected: substrate matches or beats prompt-RAG because hash
     gating skips injection on strong-prior problems (automatic
     Tier 1 preservation).

### Alternative: iterate on retrieval gating in prompt-RAG

If you want to validate the gating hypothesis BEFORE committing to
the substrate install: add a confidence gate to
`CodeVerifierFacade.compute_hints` — skip retrieval injection when
the problem matches a known-strong pattern (e.g. "write is_prime",
"write gcd"). Rerun complex eval. Prediction: hinted now ≥ stock on
every problem.

### Known open issues

- `csv_column_stats` problem (all 0/0): neither condition produced
  passing code. `from io import StringIO` import missed by Gemma.
  DB doesn't have a complete-code example showing this pattern.
  **Implication**: DB completeness matters. Partial code (missing
  imports) in retrieved examples propagates into Gemma's output.
- `token_bucket_rate_limiter` (all 0/0): TypeError at runtime in
  stock and sanity conditions. Class design is harder than Gemma
  handles reliably without cards.

## First action on resume

1. Read this handoff.
2. Read `.claude/rules/retrieval.md`, `code_reasoning_db.md`,
   `recursion.md` (new).
3. Verify DB state: `ls .cache/r53_code_db/` and
   `PYTHONPATH=. python3 -c "from calm.llm_computer.facades.code_example_db import CodeExampleDB; print(len(CodeExampleDB.load_default()))"`
   should print 8970.
4. Decide: R53.5 PT training (default) or prompt-gating iteration.
5. If R53.5: `bin/gemma-run --status`; if daemon down, start it
   (~3 min cold start). Then design the training script — reuse
   `scripts/train_copy_*.py` as templates.

## Session commits

- (pre-session) R52 arc complete (R52.3 null: KL-divergence
  distillation of L24 failed identically to MSE and SAE features)
- R53 code artifacts all committed this session (retrieval.py,
  code_example_db.py, code_verifier.py, data_generators/*, r53_*.py)
- Conceptual findings captured into rules via this handoff

## Related rules

- `retrieval.md` — hybrid retrieval architecture (new)
- `code_reasoning_db.md` — DB + generators (new)
- `recursion.md` — card-level self-improvement (new)
- `augmentation_thesis.md` — Tier-1 preservation property (edited)
- `capability_gain.md` — failure-surface-gate concretized (edited)
- `calm.md` — retrieval reference (edited)
- `tracing_roadmap.md` — R53 row appended (edited)
- `CLAUDE.md` — R53 subsection + links (edited)
