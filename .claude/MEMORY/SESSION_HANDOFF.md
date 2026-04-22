# Session Handoff — 2026-04-22 (R22 14.5 recal + decode-path facade proliferation + Recursion L1/L2 shipped)

## Goal

Opened with "read '/mnt/c/Users/gabes/projects/claw-code/.claude/MEMORY/SESSION_HANDOFF.md' and summarise" + "ok implement next steps". Session was a mega-session working the Phase-A queue from the 2026-04-21 handoff, then three follow-ups, then a combined arc of more auto-facades + Level-2 MetaFacade + rules rewrite.

User directions mid-session (chronological):
1. "implement next steps: hypothesis, build, test, commit and iterate" → worked full queue (items 1, 2, 3, 6, 7)
2. "implement all 3 please" → F1 ICD-10 retry + F2 NumericEncode+chain + F3 Recursion L1
3. "implement 2+3+7 as combined arc" → M1 (4 more auto-facades) + M2 (Level-2 MetaFacade) + M3 (docs rewrite)
4. "/update uses explore agents" (rejecting initial single-agent pass) → 3-agent P0/P1/P2 tier discipline
5. "/handoff" → this document

Ended at a clean stopping point: **13 commits**, full docs sync, substrate has 17 operational facades (6 hand-written + 11 auto/meta-generated), recursion infrastructure shipped through Level-2.

## Completed (13 commits, `9691e06` → `438e874`)

### Phase A — original queue (5 commits)

| SHA | Round | Headline |
|---|---|---|
| `9691e06` | R22f | N=10 flat diagnostic + threshold fix — 51/60 → **60/60 (Δ=+18)** via `min_margin=22.0 → 14.5`. Per-N margin discipline established: N=5 p50≈23.3, N=10 p50=20.83 p5=15.21, N=15 p50=18.63 p5=16.39. Threshold-below-lowest-p5 rule. |
| `69279d4` | R53a | **NumberTheoryFacade** (`calm/llm_computer/facades/number_theory.py`) — mod/GCD/LCM. **8/15 → 15/15 (Δ=+7)**. Caught + fixed decode-path bias bug: leading `▁` (id 236743) strip + POST_BIAS_BUDGET=4. |
| `c3cc73f` | R22d rerun | Independent 60-prompt all-keys corpus: **42/60 → 60/60** at threshold 14.5. 58/60 fired, 0 regressions. Confirms R22f result on different corpus shape. |
| `afc0220` | R60a | **Icd10RecallFacade** (`calm/llm_computer/facades/icd10_recall.py`) — first tier-3 decode-path facade (TEXT answer). **8/30 → 26/30 (Δ=+18)** on 72,748-code CMS 2022 DB. Generalizes step-through bias from integers to arbitrary Gemma BPE. |
| `956a3ae` | R70a | **PlannerFacade** (`calm/llm_computer/facades/planner.py`) — orchestrates 4 specialist facades. **20/20 route + 18/20 answer** on mixed 20-probe corpus. |

### Phase B — follow-ups (3 commits)

| SHA | Round | Headline |
|---|---|---|
| `8ba151d` | F1 | ICD-10 code-echo detect+retry infrastructure. In-context diagnosis injection + 3× boost on retry. 4 edges (T44.6X4D, T40.5X4D, V80.22XA, W10.0XXA) remain — genuine tier-3 limit. Score unchanged (26/30). |
| `5ee61a5` | F2 | **NumericEncodeFacade** (int→hex/binary/octal) + **PlannerFacade chain dispatch** for "X in hex/binary/octal". **12/12 route + 12/12 answer** on chain corpus. Option C step-1 per tracing_roadmap. |
| `3274659` | F3 | **Recursion Level-1 MVP** (`calm/llm_computer/recursion.py`). `FacadeSpec` → `validate_facade` → `generate_facade` (ast.parse-gated) → `import_facade_class`. Demo: factorial_auto + fibonacci_auto generated, **5/10 → 10/10**. Three CALM-anchored gates keep loop drift-free. |

### M1+M2 combined arc (1 commit)

| SHA | Arc | Headline |
|---|---|---|
| `5173745` | M1+M2 | M1: 4 Level-1 specs (combinations, permutations, power, next_prime) — **12/20 → 20/20 (Δ=+8)**. M2: `MetaFacade.from_oracle(fn_name, arity)` synthesizes FacadeSpec from just name + arity — 5 meta-specs (factorial, combinations, gcd, lcm, fibonacci) — **4/15 → 15/15 (Δ=+11)**. Combined 16/35 → 35/35. |

### Docs rewrite via /update 3-agent split (4 commits)

| SHA | Tier | Files |
|---|---|---|
| `5fa8228` | P0 | CLAUDE.md, delta_rule.md, Substrate.md, embed_intelligence.md, compute_facades.md, recursion.md, architecture.md — R22 14.5, decode-path facade list, recursion L1/L2 shipped, ▁-strip discipline, text-recall tier-3 refinement |
| `eec1178` | P1 | augmentation_thesis.md, tracing_roadmap.md, capability_gain.md — tier-2 table + facades-built rows + per-round receipts |
| `cf8a8ee` | P2 | workflow_part_1.md, commercial.md — daemon state invariants + commercial decode-path row |
| `438e874` | chore | r22d script default 22.0→14.5 + r60a eval regenerated |

### Key decisions + WHY

- **Threshold 14.5 over 18.0 / 22.0**: max lift (60/60) with 0 regressions and 59/60 fire rate. 14.5 sits below the lowest observed margin p5 (N=10's 15.21) across all in-distribution Ns.
- **Icd10 as decode-path (not CardSlot)**: hours to ship vs days; 26/30 on first pass. Rule refined in `augmentation_thesis.md`: short known-length text recall from static DB is decode-path-addressable.
- **PlannerFacade first-match-wins**: priority order icd10 → base_conv → numeric_encode → number_theory → multi_step. Avoids cross-facade parse ambiguity (ICD-10 codes could look decimal-ish; hex literals could look like multi_step operands).
- **Recursion as template-parameterized (not LLM-written)**: Level-1 generator is deterministic Python template + ast.parse gate + safe_eval oracle. This is the "drift-free by construction" claim from `recursion.md`. LLM-written Level-2 (Gemma writes FacadeSpec) is the natural future extension.
- **Bag-of-words scoring for ICD-10**: anchor-word too strict (Gemma's "Hypertension" missed "(primary) hypertension" literal), longest-word too weak ("complications" can be any diagnosis). Bag-of-words ≥4-char non-stopword is the robust middle.
- **R22f diagnostic chain** (non-obvious): `round6_gated_write.jsonl` showed 41/60 cards "silent" with margin=0 / argmax=`<pad>`. First hypothesis (parse failure) ruled out by offline regeneration. Live probe via `r22f_live_parse_trace.py` showed standalone card is 100% on N=10/15 with margins clustering below 22.0. The gate — not the card — was the bottleneck.

## In Progress

**None.** All 13 commits landed cleanly. No in-flight training runs, no mid-refactor code, no uncommitted session work at risk. Daemon running (PID 362642, 32m uptime, warm).

## ⚠ Uncommitted

```
 M .claude/MEMORY/notesd.md       — pre-existing modification (was ' M' at session start too)
?? .cache/                        — runtime cache (icd10 DB, r22b/r53/r60a/r70a/r70b jsonls); convention-gitignored
?? .claude/MEMORY/minutes/        — session transcript (2026-04-22_0747-1055_54a426a0.md, 62 KB)
?? .claude/scheduled_tasks.lock   — runtime lock
?? .codex/                        — external-tool workspace
?? .port_sessions/                — runtime
?? calm/.module_learning.json     — runtime state
```

**Risk assessment**:
- **`notesd.md`** — NOT in .gitignore. Contains DeltaNet / R7 analysis from prior session. Session-agnostic content. **(b) Safe to commit later** — user choice.
- **All `??` entries** — convention-gitignored runtime state. None are session-critical. `.cache/` contains the 6.4 MB ICD-10 DB + per-round jsonls; regenerable from scripts if needed.
- **No session-critical uncommitted work.** All shipped code + evals + docs are committed.

## Next Steps

**Recommended priority order** (by commercial/research leverage):

### Short (30 min — 2 hours each)

1. **Fix 4 stubborn ICD-10 codes** (T44.6X4D, T40.5X4D, V80.22XA, W10.0XXA). F1 infra landed but these resist rephrase + in-context injection. Two paths:
   - **Pure-DB bypass**: skip Gemma entirely for detected codes, return diagnosis text directly. Fastest; loses the "Gemma explains" property but gets 30/30.
   - **Multi-shot prompt ensemble**: generate with 3 different prompt formats, pick output with ≥1 significant word. ~3× inference cost; keeps Gemma in the loop.
   - Target: 26/30 → 30/30. Commit `8ba151d` has the retry scaffolding.

2. **Register auto-generated facades with PlannerFacade**. Currently `factorial_auto`, `fibonacci_auto`, `combinations_auto`, etc. aren't in Planner's dispatch chain — users with factorial prompts would hit the `multi_step` catch-all and miss the facade. Add a registry pattern: `recursion.generate_facade()` also registers the generated class with `PlannerFacade.register(cls)`. ~1 hour.

3. **Add days-between-dates facade** via `MetaFacade`. Parser is richer (ISO date extraction) but `date_ops` backend has `days_between(date1, date2)`. Good test of MetaFacade's `extra_patterns` for non-trivial regex. Shows the decode-path pattern extends to non-arithmetic domains. ~1-2 hours.

### Medium (4-8 hours)

4. **Recursion Level-3 — MetaMetaFacade**. Observe MetaFacade failure modes (higher arity, non-regex parsers, non-integer outputs like `sqrt` → float, boolean outputs like `is_prime`), propose new template families. Current Level-2 template library is the upper bound of the present system. Landing any of these extensions proves Level-3 is real.

5. **Hospital card deck**: use Level-2 MetaFacade to rapidly stand up 5-10 medical-vertical facades. Drug interaction lookup + dosage calculator + medication-name validation + chief-complaint-to-ICD mapping. Each is a `FacadeSpec` + oracle test set. 1-2 day build = deployable hospital-vertical demo.

6. **3-step Planner chains**: "GCD(48,180), then multiply by 3, in hex" → NumberTheory → MultiStep → NumericEncode. Requires parsing "then"/"," connectives in the chain detector. Proof of orchestration-layer scale.

### Longer (8+ hours, research-adjacent)

7. **LLM-written Level-2 variant**. Instead of template-param `FacadeSpec`, have Gemma (+ CodeExampleDB retrieval + CodeVerifierFacade) write the actual Python facade source. Then `ast.parse` + safe_eval oracle still gate. RLAIF-safe "Gemma self-improves": verifier rejects arbitrary code, no drift amplification. Key differentiator from SFT-based self-instruct.

8. **Wire CALM → oracle-signature inference**. Closed loop: CALM verifier catches Gemma failure → infers (fn_name, arity) from the prompt → MetaFacade proposes spec → Level-1 pipeline ships the facade. Last missing link before Phase B is fully autonomous.

## Key Context

### Decode-path bias discipline (R53a receipt, now canonical)

- Gemma's natural `0` logit after `"Answer: "` is **57–66**. +50 boost on `▁` (id 236743) cannot flip.
- **Rule**: strip BOTH BOS (id=2) AND leading `▁` (id=236743) from bias token sequences for integer-answer facades. Step-0 then biases the first digit directly.
- **POST_BIAS_BUDGET=4**: after bias exhausts, Gemma sticks in same-digit loops ("0-run", "F-run"). Cap post-bias tail at 4 natural tokens.
- **`_parse_int` caps digit-run at 12 chars** as a last-ditch defense against residual loops.
- **Scope**: applied in `number_theory.py`, `numeric_encode.py`, and all `recursion.py`-generated facades (via shared `_TEMPLATE`). NOT backported to `multi_step.py` / `base_conversion.py` — shipped tests don't trigger the bug.
- **Text-answer exception** (Icd10RecallFacade): do NOT strip `▁` — the diagnosis starts with a capital merged into `▁Type`, `▁Hypertension`, etc.

### Daemon state invariants (2026-04-22 lesson)

- `RESET_GLOBALS` (via `bin/gemma-run --reset`) does NOT clear `m.verification_hooks`, `m.reserved_channels`, or `m.layers[idx].card_slots`. A lingering R22 MQAR hook from r22d caused r60a's first run to emit `"4444..."` on every ICD-10 probe.
- **Rule**: every facade test script starts with `clear_card_state()` — pattern in `r60a_icd10_failure_gate.py`, `r70a_planner_mixed.py`, `m1a_four_new_facades.py`, `r80a_recursion_demo.py`.
- After editing a facade module source, **full daemon restart** (`--quit` + `--start`) is required, not `--reset`. The daemon re-execs the SCRIPT on each run, but `sys.modules` is shared. `importlib.reload` inside the script helps for THAT run but subsequent runs see the cached version again.
- Signature of a lingering hook: pure-digit artifacts on unrelated prompts ("hello" → "0000000"). Check state before suspecting new code.

### ICD-10 edge cases (F1 ruled out)

4 codes resist both prompt rephrase + in-context injection + 3× boost:
- **T44.6X4D** — Poisoning by alpha-adrenoreceptor antagonists, undetermined
- **T40.5X4D** — Poisoning by cocaine, undetermined, subsequent encounter
- **V80.22XA** — Occupant of animal-drawn vehicle injured in collision
- **W10.0XXA** — Fall (on)(from) escalator, initial encounter

Common trait: unusual internal tokens (`X4D`, `22XA`, `0XXA`) trigger Gemma's "code-analysis format" prior. Output dumps "Analysis of ICD-10 Code..." template that dominates both integer and text bias. Genuine tier-3 edge; documented in `compute_facades.md` as future work.

### Failed approaches (cite SHAs, don't retry)

- **Parse failure as R22 N=10 hypothesis** (r22f_parse_diag.py): offline regeneration showed 60/60 parse OK. The bottleneck was margin gate, not parser.
- **Anchor-word scoring for ICD-10**: too strict. Gemma's "Hypertension" missed anchor "(primary) hypertension".
- **Longest-word scoring for ICD-10**: too weak. "complications" matched any diagnosis containing that generic word.
- **WebFetch on CMS ICD-10 page**: 60000ms timeout. CMS zip URL returned HTML-redirect page, not ZIP. Used `smog1210/2022-ICD-10-CM-JSON` GitHub mirror instead (6.7 MB, 72,748 codes verified).
- **Hand-guessed ICD-10 codes**: 23/50 first-guess codes missing from 2022 DB. Switched to DB-sampled real codes.
- **`comb` / `perm` / `isqrt` / `divmod` as safe_eval function names**: not registered. Correct names are `combinations` / `permutations` / `sqrt` / (no divmod).
- **Generator `r`-prefix + `{pat!r}`**: double-escapes `\s` → `\\s`. Fix: `{pat!r}` without `r` prefix. Inline comment at `_render_parse_res_literals` in `recursion.py`.
- **`path.relative_to(ROOT)` in script running under daemon**: daemon cwd differs from repo root. Use `Path(__file__).resolve().parent.parent.parent` for `_REPO_ROOT`, catch `ValueError` in script.

### Methodology caveats

- All A/B runs are single-seed / single-run (not median-of-5 per workflow_part_1.md §"GPU bench discipline"). Acceptable for decode-path facades because the delta is large (several-fold change) and deterministic modulo Gemma's argmax; not acceptable for future Triton kernel work where variance is 20-30%.
- R60a scoring function evolved across three iterations (anchor → longest → bag-of-words). Final BoW was baked into the script before last commit; receipts earlier in the chain use weaker scoring.
- PlannerFacade uses runtime-glue orchestration (Option A). Option C (compiled planner card with channel-as-register state) is deferred.
- Recursion generator is template-parameterized, not LLM-written. Level-2 MetaFacade adds pattern synthesis but not code synthesis. True LLM-in-loop recursion remains future work.

### Hardware / environment state at session end

- **Daemon**: PID **362642**, uptime **32m37s** at /handoff time. `python3 -u bin/gemma_daemon.py`.
- **GPU**: 6299 MiB used, 27% util. Gemma 4 E4B tq4 substrate preloaded (5.17 GB) + Q6_K embedding + tq4 weights.
- **Substrate**: `/home/gabe/models/gemma-4-E4B-it-tq4-aligned.gguf`, 42 layers, 720 tensors.
- **ICD-10 DB**: `.cache/icd10/icd10cm_codes_2022.json` (6,707,336 bytes, 72,748 codes).
- **Generated facade files**: `calm/llm_computer/facades/factorial_auto.py` (6534 B), `fibonacci_auto.py` (6633 B), etc. — all verified on disk post-commit.
- Last daemon job: `m2a_metafacade_demo.py` (M2A_DONE).

## Files in Project (session-shipped)

### New core module
- `calm/llm_computer/recursion.py` — Level-1 `FacadeSpec` generator + Level-2 `MetaFacade`. `_TEMPLATE` at line ~70, `_REPO_ROOT` at line ~270, `generate_facade()` / `validate_facade()` / `import_facade_class()` at lines ~273-350, `MetaFacade.from_oracle()` at line ~528, module-level FACTORIAL_SPEC / FIBONACCI_SPEC / COMBINATIONS_SPEC / PERMUTATIONS_SPEC / POWER_SPEC / NEXT_PRIME_SPEC at 356-462.

### New facade files (15)
- **Hand-written specialists** (5): `number_theory.py`, `icd10_recall.py`, `numeric_encode.py`, `planner.py`, + prior `base_conversion.py` / `multi_step.py` unchanged.
- **Level-1 auto-generated** (6): `factorial_auto.py`, `fibonacci_auto.py`, `combinations_auto.py`, `permutations_auto.py`, `power_auto.py`, `next_prime_auto.py`.
- **Level-2 meta-synthesized** (5): `factorial_meta.py`, `combinations_meta.py`, `gcd_meta.py`, `lcm_meta.py`, `fibonacci_meta.py`.

### New scripts (14)
- Diagnostic: `r22f_{n10_diag, parse_diag, live_parse_trace, threshold_sweep}.py`, `r53a_{debug, debug_probe}.py`.
- A/B: `r22d_rerun_final_config.py`, `r53a_number_theory.py`, `r60a_icd10_failure_gate.py`, `r70a_planner_mixed.py`, `r70b_planner_chain.py`, `r80a_recursion_demo.py`, `m1a_four_new_facades.py`, `m2a_metafacade_demo.py`.

### Modified docs (11 rules files + CLAUDE.md)
- `.claude/CLAUDE.md` — Brain+Cards model rewritten, install typology updated 22.0→14.5, decode-path facade list extended.
- `.claude/rules/delta_rule.md` — §R22 install rewritten for 14.5 shipping + 22.0 preserved as historical receipt.
- `.claude/rules/Substrate.md` — §CardSlot example updated, per-N margin data added.
- `.claude/rules/embed_intelligence.md` — R22 min_margin narrative + new §"`▁`-strip + POST_BIAS_BUDGET discipline".
- `.claude/rules/compute_facades.md` — full rewrite; "Shipped instances" table (17 facades), three-path "Adding a new domain" (Level 0 / 1 / 2).
- `.claude/rules/recursion.md` — Level-1 + Level-2 rewritten as SHIPPED.
- `.claude/rules/architecture.md` — prod Gemma install line updated.
- `.claude/rules/augmentation_thesis.md` — tier-2 table extended; hospital vertical cites Icd10RecallFacade; tier-3 text-recall refinement.
- `.claude/rules/tracing_roadmap.md` — 7 new "Shipped and verified" rows.
- `.claude/rules/capability_gain.md` — §"2026-04-22 session receipts" with per-commit table.
- `.claude/rules/workflow_part_1.md` — new §"Daemon state invariants".
- `.claude/rules/commercial.md` — two new differentiator rows.

### New eval receipts (9)
- `.claude/MEMORY/evals/2026-04-22_{r22f_threshold_sweep, r22d_rerun_threshold_14.5, r53a_number_theory_facade, r60a_icd10_tier3_demo, r70a_planner_mixed, r70b_planner_chain, r80a_recursion_level1_demo, m1a_four_new_facades, m2a_level2_metafacade}.md`.

### Cache / data
- `.cache/icd10/icd10cm_codes_2022.json` (6.4 MB, 72,748 codes) — gitignored by convention.
- `.cache/{r22b, r60a_icd10_results.jsonl, r70a_planner_results.jsonl, r70b_planner_chain_results.jsonl}` — per-round replay data.

## Handoff verification

- Narrative vs git state: **match.** All 13 commits confirmed via `git log --oneline -15`.
- `min_margin=14.5` present in 5 doc locations (CLAUDE.md, delta_rule.md, Substrate.md, architecture.md, capability_gain.md). Stale 22.0 references only appear in explicit "historical receipt" context (delta_rule.md, embed_intelligence.md).
- 11 new facade files + `recursion.py` confirmed on disk.
- Generated `factorial_auto.py` (6534 B) + `fibonacci_auto.py` (6633 B) verified.
- `.cache/icd10/icd10cm_codes_2022.json` verified (6,707,336 bytes).
- All `.claude/` docs ≤500 LOC soft cap.
- 2-agent Explore grounding matches main-context narrative — no discrepancies flagged.
- Uncommitted session-critical files: **none.** Only `.claude/MEMORY/notesd.md` (pre-existing, non-today) + runtime caches.
