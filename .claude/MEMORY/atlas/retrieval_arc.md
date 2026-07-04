# Retrieval + CodeExampleDB — Historical receipts

Phase 1 shipping arc, blanket-RAG null measurement, L41-install
regression, dense-tq4 ship-date, per-source dedup detail. Current
rules: `.claude/rules/retrieval.md` + `.claude/rules/code_reasoning_db.md`
(stubs; operational detail in this file).

## Phase 1 shipping arc (R53)

R53 Phase 1 shipped the full hybrid retrieval stack over
`CodeExampleDB`:
- Hybrid TF-IDF + BM25 + Dense + RRF (three independent retrieval
  systems, fused at query time)
- 8970-example CodeExampleDB with per-source dedup
- DomainDataGenerator framework with 9 concrete generators producing
  222 verified (problem, solution, tests) examples

Phase 2 (code DT training + install) was scoped but parked pending
DT code-skeleton honest-val reaching install threshold — see
`MEMORY/atlas/delta_rule_arc.md` §"DT code-skeleton arc".

## Blanket-RAG null measurement (R53.2b)

Complex eval (`scripts/r53_eval_complex.py`, 6 multi-step coding
problems × 3 conditions):

| | stock | hinted (real retrieval) | sanity (random retrieval) |
|---|---:|---:|---:|
| TOTAL | 25/27 | 21/21 | 23/23 |
| Δ vs stock | — | +7.4pp | +7.4pp |
| **retrieval-attributable gain** | | | **+0.0pp** |

Hinted = Sanity. Prompt-length / "has examples in context" effect is
real; content of real retrieval adds nothing on top. Full analysis:
`MEMORY/atlas/augmentation_thesis_arc.md` §"R53.2b".

## R53.14 / R53.20a / R53.20b — L41 install regression

POST-SWA-fix re-run on R53.0 6-problem code corpus produced -9.3pp
regression when substrate card installed at L41 with
`CardSlot(preserve=True) + FirstTokenHook(boost=50)`. Root cause:
first-token bias is the wrong delivery mechanism for code — Gemma's
first-token on code is uniformly confident (margin 6.8-9.2), so
`min_margin=0.5` never gates. Full receipt:
`MEMORY/atlas/embed_intelligence_arc.md` §"R53.14 / R53.20a / R53.20b".

## Dense-tq4 default (R53.28)

`DenseIndex.load(prefer_tq4=True)` is the new default. Loads
`.tq4.pt` companion when present — 4× smaller than `.pt`, <1% rank
flip. `bin/gemma-run scripts/r53_build_dense.py` saves tq4 by default.

## Code-DT self-distill roadmap (R53.5 + R53.6, parked)

Originally scoped to train `copy_code_best.pt` on the 8970-example DB,
install at L24 via CardSlot, run CodeVerifierFacade-gated
self-distillation loop. Parked pending DT code-skeleton honest-val
reaching ≥ 0.40 install threshold (currently 0.193 on 520 held-out).

See `MEMORY/atlas/delta_rule_arc.md` §"DT code-skeleton arc" for the
honest-val trajectory.

## Two-channel-design roadmap (R53.6)

For multi-step tasks, retrieval should surface TWO separate content
types, injected at different points:

| Channel | Content | Substrate install |
|---|---|---|
| Code hits | Complete executable function + imports + tests | L30 KnowledgeStore recall card (exact hash lookup) |
| Reasoning traces | `<think>` blocks, step-by-step decomposition | L24 PT/DT (CopyAugmentedTransformer/DeltaNet, NL→structure) |

At ingest, each DB example should be decomposed:
```python
code_fragment = extract_fenced_code(solution)
reasoning_trace = extract_think_block(solution)
```

Store as separate fields. Retrieve independently. Inject independently.
Avoids format contamination (prose leaking into code imitation).

Parked pending Phase 2 code DT training.

## Hybrid retrieval stack — operational reference

Three independent retrieval systems fused at query time via RRF:

```
query → TF-IDF/BM25 sparse ─┐
      → Gemma dense embed  ─┼→ rrf_fuse (k_const=60) → top-k
      → Jaccard fallback   ─┘
```

`CodeExampleDB.retrieve(query, mode="auto"|"tfidf"|"dense"|"hybrid"|"jaccard")`.

**TF-IDF/BM25** (`TfidfIndex`): hand-rolled sparse index; default
`scorer="bm25"` (k1=1.5, b=0.75). Index doc =
`(problem * problem_weight) + solution` with `problem_weight=3`
default — symptom queries hit solution content.

**Dense** (`DenseIndex`): mean-pooled Gemma `token_embd`, L2-normalized
fp16; trie-backed fast tokenizer + batched dequant (~11.3 s for 8970
texts). `_PRE_TOKENIZE_CHAR_LIMIT=400` (problem-biased; solutions via
sparse side). Persistence: `dense.pt` + optional `dense.tq4.pt`;
`load(prefer_tq4=True)` default.

**RRF**: `score(doc) = sum 1/(60 + rank)` across systems. Recommended:
sparse top-10 + dense top-10 → final 3-5.

**Gating rule (Phase-1 finding)**: do NOT inject on strong Gemma priors.
Skip when CALM Layer 2 has direct answer, vanilla-algorithm intent, or
low retrieval scores. Inject on ambiguous / bug-diagnostic / multi-step.

**Quick reference**:

```python
db = CodeExampleDB.load_default()
db.build_tfidf(include_solution=True, problem_weight=3)
db.save_indices(".cache/r53_code_db", dense_quantize=True)
hits = db.retrieve("write is_prime", k=3, mode="hybrid", dense_m=m, dense_tok=tok)
```

Rebuild: `scripts/r53_run_data_generators.py` (CPU TF-IDF);
`bin/gemma-run scripts/r53_build_dense.py` for dense.

## CodeExampleDB operational reference

8970 unique examples after problem-hash dedup (priority order in
`DEFAULT_CORPORA` — hand-written > HF > 9B-generated).

| Source | Count | Notes |
|---|---:|---|
| coding_reasoning_claude | 547 | hand-written |
| claude_reasoning | 910 | merged HF |
| MBPP | 974 | all splits |
| HumanEvalPlus | 164 | |
| BigCodeBench | 1140 | v0.1.4 |
| CodeContests (Py3) | 2498 | |
| Crownelius | 265 | |
| TeichAI | 886 | |
| Nohurry (code-only) | 106 | |
| Multi-step synthetic | 48 | |
| Generator output | 222 | 9 generators, sandbox-verified |
| Language-specific | 117 | python/typescript/rust |
| Prefilter residual | — | long-tail HF |

Quality gate (`r53_fetch_corpora.py:passes_quality`): problem ≥20 chars,
solution ≥100 chars, code markers present.

**Caches** (`.cache/r53_code_db/`): `tfidf.json` (36 MB), `dense.pt`
(46 MB), `dense.tq4.pt` (12 MB).

**API**:

```python
db = CodeExampleDB.load_default()
db.build_tfidf(); db.build_dense(m, tok)
db.save_indices(cache_dir, dense_quantize=True)
hits = db.retrieve("write is_prime", k=3, mode="hybrid", dense_m=m, dense_tok=tok)
```

**DomainDataGenerator**: 9 generators → 222 verified examples;
verification = AST parse → sandbox (5 s) → optional `extra_verify()`.
Scripts: `r53_fetch_corpora.py`, `r53_run_data_generators.py`
(`--skip-dense`, `--dense-only`, `--out DIR`).

**Invariants**: dedup on `CodeExample.key` (blake2b problem hash);
`skip_sandbox=True` for non-deterministic modules; TF-IDF v2 postings
`(raw_tf, tfidf_w)` backward-compatible with v1.

## Format contamination fix (R53)

Raw entries contained `<think>` + prose + fenced code.
Original `solution_preview` stripped think blocks but left prose headers;
Gemma imitated prose style → extractor scored 0/0.

**Fix** — `CodeExample.solution_preview` now:
1. Strips `<think>...</think>`
2. Strips trailing `**Verified test cases` / `**Sample I/O` / etc.
3. Prefers first ` ```python ` fenced block
4. Falls back to slice from first `def`/`class`/`import`
5. Raw trim as last resort

**Measured**: date_validation_chain 0/0 → 12/12 after fix.

**Rule**: retrieved preview format must match output target (code-only
for code generation). Long-term: two-channel retrieval — see
§"Two-channel-design roadmap" above.

## Cross-refs

- Current retrieval rules: `.claude/rules/retrieval.md` (stub; detail in this file)
- Current DB rules: `.claude/rules/code_reasoning_db.md` (stub; detail in this file)
- Blanket-RAG full analysis: `MEMORY/atlas/augmentation_thesis_arc.md`
- L41 install regression detail: `MEMORY/atlas/embed_intelligence_arc.md`
- DT code-skeleton progress: `MEMORY/atlas/delta_rule_arc.md`
