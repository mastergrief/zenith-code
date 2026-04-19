# Retrieval — Hybrid TF-IDF + BM25 + Dense + RRF

R53 Phase 1 shipped a full hybrid retrieval stack over `CodeExampleDB`.
This rule documents the architecture, API, and the invariants that
keep it correct. See `.claude/rules/code_reasoning_db.md` for the
DB side; this file is about how retrieval works on top of that DB.

## Architecture

Three independent retrieval systems, fused at query time via RRF:

```
query text
  ├─► TF-IDF sparse index  ─►  top-k_sparse ranked docs  ─┐
  ├─► Gemma-dense index    ─►  top-k_dense  ranked docs  ─┼─► rrf_fuse ─► top-k final
  └─► Jaccard (fallback)   ─►  top-k ranked docs         ─┘   (consensus-promoted)
```

At query time the mode is chosen by `CodeExampleDB.retrieve(query, mode="auto")`:

- `"auto"`: hybrid if both indices built, else tfidf-only if only tfidf,
  else dense-only, else jaccard
- `"tfidf"` / `"dense"` / `"hybrid"` / `"jaccard"`: explicit

## Component 1 — TF-IDF + BM25 (`retrieval.py:TfidfIndex`)

Hand-rolled sparse index, no sklearn dep. Stores:

- `_term_to_postings[term] -> {doc_idx: (raw_tf, tfidf_weight)}` —
  both representations stored so BM25 (uses raw tf) and cosine (uses
  tfidf weight) can use the same postings
- `_doc_lens[doc_idx]` — for BM25 length normalization
- `_avgdl` — average doc length
- `_idf[term]` — smoothed IDF: `log((N + 1) / (df + 1)) + 1`
- `_doc_norms[doc_idx]` — precomputed L2 for cosine

**Two scoring modes** selected at query time via `scorer=`:

- `scorer="bm25"` (default): Okapi BM25 with `k1=1.5`, `b=0.75`
  ```
  score(d, q) = sum_{t in q}
      idf(t) * tf(t,d) * (k1 + 1)
             / (tf(t,d) + k1 * (1 - b + b * |d|/avgdl))
  ```
- `scorer="tfidf"`: classic cosine over (1 + log(tf)) * idf vectors

BM25 is the default for unfamiliar queries (better length handling,
robust to long solution texts). TF-IDF cosine retained for
interpretability and lightweight cases.

### Indexing policy

`CodeExampleDB.build_tfidf(include_solution=True, problem_weight=3)`:

- When `include_solution=True` (default), the doc is `(problem * 3) + solution`
- Problem repeated to give ~3× weight — keeps user-intent as dominant
  signal while allowing symptom/symbol queries to hit on solution
  content
- Why: queries like "fix mutable default" should match bug-fix
  examples whose *solution* shows the fix pattern; indexing problem
  alone misses this

### Vocab stats on current DB (8970 examples)

- Problem-only indexing: ~27K vocab terms
- Problem + solution indexing: ~68K vocab terms (+152%)
- Build time: 0.8 s on CPU
- Persistence: `tfidf.json` (36 MB for 8970 docs)

### Serialization (v2)

`_term_to_postings` stored as `[[doc_idx, raw_tf, tfidf_weight], ...]`
triples. v1 format (pairs) is still loadable — raw_tf defaulted to 1
when upgrading (BM25 produces degraded but usable scores).

## Component 2 — Dense (`retrieval.py:DenseIndex`)

Substrate-native: mean-pooled Gemma `token_embd` over each problem's
token sequence, L2-normalized to a (d_model,) float16 vector.

### Encoding (`_encode_texts`)

Per-example encoding would be O(|vocab|) per call with the naive BPE
tokenizer — pathologically slow on 8970 long code-containing texts
(never completes in reasonable time). Two fixes compound:

1. **Trie-backed fast tokenizer** (`_monkey_patch_fast_encode`):
   builds a char-trie over the 262K-token GemmaTokenizer vocab once
   (~0.5 s), replaces `tok.encode` with an O(|text|) longest-prefix
   walk. ~13,000× speedup over the shipped `GemmaTokenizer.encode`.
   The shipped encoder uses naive `for token, tid in sorted_tokens:`
   scan per position — O(|text| × 262K). The trie version is O(|text|
   × max_token_len ≈ 20).
2. **Single batched dequant**: tokenize all N texts, form the union
   of unique token ids (~26K for 8970 texts), dequant that set ONCE
   from GpuQ6KEmbedding in one call, then CPU-side mean-pool per text
   from the fp32 lookup table. Avoids per-text GPU round-trips.

Result: **8970 texts encoded end-to-end in 11.3 s.**

### Truncation

`_PRE_TOKENIZE_CHAR_LIMIT = 400`: input truncated before tokenizing.
Problem statements are the retrieval-relevant signal; very long
code-fenced solutions would waste tokenize budget without adding
retrieval precision. Solutions are indexed by the SPARSE side via
`include_solution=True` — dense is intentionally problem-biased.

### tq4 persistence

`DenseIndex.save(path, quantize=True)` writes BOTH:

- `dense.pt` (fp16): shape (N, d_model), ~5 MB per 1K examples
- `dense.tq4.pt` (tq4): 4× smaller (TurboQuant 4-bit), ~1.3 MB per 1K

`DenseIndex.load(path, prefer_tq4=True)` dequants the tq4 file once
at load (~300 ms overhead) then query-time is identical to fp16.
Rank-preserving (Pi rotation + 16-level Lloyd-Max codebook); rank
flips observed < 1% on validation queries.

At 8970 examples: 46 MB fp16 vs 12 MB tq4. Practical threshold for
tq4 is ~50K+ examples, but we save both so the API is scale-ready.

## Component 3 — RRF (`retrieval.py:rrf_fuse`)

Reciprocal Rank Fusion merges multiple ranked lists using rank only
(not raw scores — so sparse cosine and dense cosine don't need
normalization):

```
score(doc) = sum over systems: 1 / (k_const + rank_in_that_system)
```

`k_const=60` per Cormack et al. 2009. Dampens the gap between rank 1
and rank 2 so single-system favorites don't dominate over cross-
system consensus. Robust; not sensitive to tuning.

Recommended k values on our 8970-example DB:

- sparse top-10 + dense top-10 → RRF → final 3-5
- Rationale: 10 each gives ~15 unique candidates after merge, RRF
  promotes docs appearing in BOTH; going to 20 dilutes signal, going
  to 3 misses paraphrase wins

## Retrieval-content policy (post-R53.2b fix)

The `solution_preview` fix documented in `code_reasoning_db.md` is
load-bearing: retrieved solutions injected into prompts must be
**code only**, not prose. `<think>` blocks and "Verified test cases:"
trailers are stripped at preview time (`CodeExample.solution_preview`).

Injection format:

```
Verified context (from local compute + example DB):
- problem_kind: <intent>
- likely imports: <libs>
- security concerns: <flags>
- verified: <arith precompute> = <value>
- related past solutions (for pattern reuse):
  [1] (0.77) problem: <problem[:120]>
      solution: <code-only preview[:max_example_chars]>
  [2] ...
```

### Gating rule (the Phase-1 finding)

Blanket retrieval injection violates Tier 1 preservation (see
`augmentation_thesis.md`). Do NOT inject retrieved examples on
problems where Gemma has a strong prior. Gate signals:

- **CALM Layer 2 precompute has a direct answer** → inject verified
  fact, SUPPRESS retrieved examples (answer is already exact)
- **Intent classifier detects "vanilla algorithm"** (`is_prime`, `gcd`,
  `fibonacci`, `binary_search`, etc.) → skip retrieval entirely
- **All top-k retrieval hits have low scores** (e.g. BM25 < 2.0,
  dense cosine < 0.5) → retrieval won't help, skip
- **Ambiguous / bug-diagnostic / multi-step composition** → inject
  (this is where retrieval helps)

This gating is what makes substrate-RAG structurally superior to
prompt-RAG: at L30, hash-lookup either MATCHES (inject fact) or
MISSES (pass-through, Gemma's native behavior). Automatic Tier 1
preservation with no gating logic needed. Prompt RAG requires
explicit gates; substrate RAG has them by construction.

## Two-channel design (future R53.6)

For multi-step tasks, retrieval should surface TWO separate types
of content, injected at different points:

| Channel | Content | Substrate install |
|---|---|---|
| **Code hits** | Complete executable function + imports + tests | L30 KnowledgeStore recall card (exact hash lookup) |
| **Reasoning traces** | `<think>` blocks, step-by-step decomposition | L24 PT (CopyAugmentedTransformer, NL→structure) |

At ingest, each DB example should be decomposed:

```python
code_fragment = extract_fenced_code(solution)
reasoning_trace = extract_think_block(solution)
```

Store as separate fields. Retrieve independently. Inject independently.
Avoids format contamination (prose leaking into code imitation).

## Quick reference

```python
from calm.llm_computer.facades.code_example_db import CodeExampleDB

db = CodeExampleDB.load_default()
db.build_tfidf(include_solution=True, problem_weight=3)
# (daemon) db.build_dense(m, tok)
db.save_indices(".cache/r53_code_db", dense_quantize=True)

hits = db.retrieve("write is_prime function", k=3, mode="hybrid",
                   dense_m=m, dense_tok=tok)
for h in hits:
    print(h.score, h.example.problem[:80])
```

Full pipeline rebuild: `PYTHONPATH=. python3 scripts/r53_run_data_generators.py`
(CPU path builds TF-IDF only; pass through `bin/gemma-run` with
`scripts/r53_build_dense.py` afterwards for dense).

## Ruled-out / refined directions

- **Substrate-RAG at L41 on code tasks (R53.14/20a/20b, post-SWA-fix)**
  — `-9.3pp` regression. Tier-1 preservation thesis holds in
  principle but the L41 install mechanism (CardSlot `preserve=True`
  + per-marker FirstTokenHook `boost=50`) disrupts HIT prompts:
  Gemma's first-token on code is confidently a fence/whitespace
  opener (margin 6.8-9.2), so `min_margin=0.5` never gates, hook
  always fires, forces "def"/"class" → code-without-fence →
  extractor fails. First-token bias is the wrong intervention for
  code. See `augmentation_thesis.md` §"R53.14/20a/20b" — refined
  thesis: Tier-1 holds at output-boundary `VerificationHook` with
  `min_margin` guard, NOT at residual-write CardSlot. Correct
  tier-2 target: post-generation AST walker.

- **`DenseIndex.load(prefer_tq4=True)` is the new default** (R53.28).
  Loads `.tq4.pt` companion when present (4× smaller than `.pt`,
  <1% rank flip). `bin/gemma-run scripts/r53_build_dense.py` saves
  tq4 by default.

## Related rules

- `code_reasoning_db.md` — what's in the DB + how it gets there
- `augmentation_thesis.md` §"Automatic Tier-1 preservation" — why
  gating matters
- `capability_gain.md` §"Failure-surface-gate" — why blanket eval
  misses the point
- `turboquant.md` — tq4 format + kernel details
- `commercial.md` — selective-intervention-as-product positioning
