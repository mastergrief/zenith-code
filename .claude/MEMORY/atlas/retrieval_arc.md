# Retrieval + CodeExampleDB — Historical receipts

Phase 1 shipping arc, blanket-RAG null measurement, L41-install
regression, dense-tq4 ship-date, per-source dedup detail. Current
rules: `.claude/rules/retrieval.md` + `.claude/rules/code_reasoning_db.md`.

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

## Cross-refs

- Current retrieval rules: `.claude/rules/retrieval.md`
- Current DB rules: `.claude/rules/code_reasoning_db.md`
- Blanket-RAG full analysis: `MEMORY/atlas/augmentation_thesis_arc.md`
- L41 install regression detail: `MEMORY/atlas/embed_intelligence_arc.md`
- DT code-skeleton progress: `MEMORY/atlas/delta_rule_arc.md`
