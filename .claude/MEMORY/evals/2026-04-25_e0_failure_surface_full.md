# E0 — Full Failure-Surface Scout (BigCodeBench N=100)

**Date**: 2026-04-25
**Spec**: `RESEARCH/UNIVERSAL_TRANSFORMERS/03_TESTING.md` §7 + §7a + §9
**Provenance**: claude+codex collab via ai-room board task `1777132726987-1d7fc6c6`
**Smoke baseline**: `.claude/MEMORY/evals/2026-04-25_e0_failure_surface_smoke.md`
**Runtime**: 73.75 min wall (4425s) for 100 generations + tests
**Decision**: **full-E0 gate clears with 2.5-3× headroom** (25 conservative / 30 liberal vs ≥10 threshold) → **UT-family arc unlocks for E1 (RDT card training)**

## What this extends from the smoke

The smoke (N=30) cleared its gate at 8 KEEP / 12 manual-review with one
critical post-run finding: 11 of 23 survivors were matplotlib
`can't start new thread` env failures, not capability failures. Codex
patched the runner (`5046981`) — RLIMIT_NPROC bump + MPLCONFIGDIR +
forced `matplotlib.use("Agg")` — and full E0 ran with the patched
classification baseline.

The smoke result then tested precondition (a) of the RDT thesis on
30 prompts. **Full E0 tests the same precondition at substrate-decision
scale (N=100)**: does a corpus of 30+ Gemma failures of
iteratively-refineable structured-output shape exist? If yes, E1
(RDT card training) becomes a justified follow-on commitment;
if no, UT-family arc parks.

## Setup (delta from smoke)

| Component | Smoke | Full E0 |
|---|---|---|
| Sample size | N=30 | **N=100** |
| Sample seed | 42 | 42 (same — first 30 of N=100 = smoke set, deterministic) |
| Runner | unpatched (smoke had matplotlib threading bug) | **patched (`5046981`)** — RLIMIT_NPROC removed, MPLCONFIGDIR + force-Agg added |
| Wall time | 23.7 min | 73.75 min |
| Filter-3 survivors | 23 | **45** |
| Manual filter-4 review | 12 (claude solo) | **45 (claude+codex split)** |

Everything else identical: T=0, max_tokens=6144, `gemma-4-E4B-it-tq4-aligned`,
`E0_PYTHON=~/.venvs/e0_bigcodebench/bin/python`, llama.cpp at `localhost:8080`.

## Raw runner outcomes

```
passed:           34
failed:           54
env_unsupported:  11
format_fail:       1   (BigCodeBench/664 — idiosyncratic, single occurrence at 6144 budget; not a plateau)
timeout:           0
```

Pass rate **34%** vs smoke's 13%. Two effects compound:

1. **Matplotlib partition shift** (codex's `5046981` patch effect): 5 of the 11 smoke matplotlib rows now correctly land as `solves_cleanly`/`partial` instead of `env_unsupported` or `fails_correctness`. Confirmed live at problems 7 (BigCodeBench/209), 14 (447), 20 (451), 21 (919), 27 (318) — exactly matching codex's pre-run replay receipt of 5 passed + 6 failed across the 11 matplotlib rows.
2. **Larger sample lands on more solvable corpus regions** — partial statistical effect, partial real coverage of easier BigCodeBench tasks.

## Analyzer partitioning (Phase 1c)

```
solves_cleanly:           43   (≥80% tests pass; drop, ceiling)
fails_correctness:        21   (tests run, ≤20% pass)
partial:                  24   (20-80% tests pass; keep)
environment_unsupported:  11
format_fails:              1

survivors_filters_1_3:    45   (fails_correctness + partial after filter 3)
manual_filter_4_required: 45
```

Note: `solves_cleanly=43` exceeds raw `passed=34` because 9 of the runner-`failed` rows had ≥80% test pass rate (just not 100%) and got reclassified as solves_cleanly by the analyzer. Correct partition behavior — matches the analyzer spec.

## Filter 4 — joint manual review (claude rows 1-22 + codex rows 23-45)

Split per `AI_ROOM_COLLAB.md` "parallel drafting on clean splits".
Each peer wrote KEEP/DROP/CAVEAT verdicts with one-line rationale to
`/tmp/e0_filter4_{claude,codex}.md`. Cross-review pass: zero
count-changing disagreements; each peer concurred fully on the other's
verdicts after reading.

### Per-row verdicts (consolidated 45 rows)

**claude rows 1-22** (BigCodeBench/51 → /440):

| Task | Verdict | Rationale |
|---|---|---|
| 51 | KEEP | KMeans got 0-row DataFrame ← Gemma's filter conditions wrong; multi-step refineable |
| 563 | KEEP | "OSError not raised" — error-handling logic refinement |
| 501 | DROP | recall: pandas `.to_excel()` engine selection for `.xls` (didn't know `xlwt` engine needed) |
| 457 | KEEP | off-by-some count (7 != 11), nested-list flatten + count refinement |
| 285 | DROP | recall: hallucinated `mechanize.Session` |
| 1116 | KEEP | statistical calc off (43.85 != 45.0) — formula/mode-tie refinement |
| 864 | DROP | recall: invented `PrettyDict` type |
| 65 | KEEP | "DataFrame are different" — multi-step DataFrame construction |
| 61 | DROP | recall: hallucinated matplotlib `Axes.get_win` |
| 191 | KEEP | pet-shop sim returned 1 customer instead of 100 — loop scope refinement |
| 1034 | KEEP | TypeError range vs list — algorithm shape right, type-conversion refineable |
| 407 | KEEP | "FileNotFoundError not raised" — error-handling-conditional refinement |
| 569 | DROP | recall: KeyError 'function_name' (output schema mismatch) |
| 326 | **CAVEAT** | empty-list-vs-bat-results — subprocess in deps; env-vs-capability ambiguous |
| 696 | KEEP | ValueError unpack-shape — return tuple shape wrong, refineable |
| 440 | KEEP | matrix shape mismatch (3,3) vs (48,1) — algebraic shape refinement |
| 189 | DROP | recall: URL parsing/validation rejected URL incorrectly |
| 704 | DROP | recall: tried `.columns` on list (type confusion) |
| 88 | DROP | recall: tried `.split()` on datetime (forgot `.strftime()`) |
| 1098 | KEEP | case-sensitivity off (`hello` != `Hello`) — algorithm refinement |
| 255 | KEEP | numpy array comparison failed — multi-step composition refinement |
| 775 | KEEP | empty dict vs full alphabet — counting/init algorithm refinement |

**codex rows 23-45** (BigCodeBench/1130 → /781):

| Task | Verdict | Rationale |
|---|---|---|
| 1130 | DROP | `Path.read` static pathlib API hallucination |
| 600 | DROP | `df['Word']` Series treatment when DataFrame expected — schema recall |
| 393 | DROP | `scipy.stats.probplot(dist=(mu, sigma))` static API misuse |
| 592 | KEEP | Multi-step file generation, fixed filename, return path; refineable |
| 727 | KEEP | Bag-of-words/vectorizer vocabulary choice fails edge cases; sklearn pipeline |
| 429 | DROP | `SelectKBest(metric=...)` nonexistent kwarg — API recall |
| 146 | **CAVEAT** | IP scan/ping loop structured but failure on subprocess/mock semantics |
| 350 | **CAVEAT** | File compression/move workflow structured but gzip/subprocess-sensitive |
| 946 | KEEP | Deterministic random DataFrame — RNG/ordering refinement |
| 777 | KEEP | Zip discovery/extraction — fixable traversal/extraction choices |
| 552 | DROP | matplotlib keyword typo `v` vs `va` — static API recall |
| 449 | KEEP | Standardize DataFrame + per-feature histograms; edge validation refinement |
| 114 | KEEP | Normalization pipeline; output shape + missing-key exception refinable |
| 469 | KEEP | Grade report + plot — DataFrame index/name/grade semantics refinement |
| 435 | KEEP | Employee DataFrame structure — dtype/validation refinable |
| 1022 | KEEP | CSV date-processing pipeline — filter/sort refinement |
| 810 | **CAVEAT** | Search/execute on Windows os.walk + subprocess — env/path caveat |
| 1103 | DROP | Subprocess execution required; observed failures dominated by `subprocess_blocked` — runner-env artifact |
| 1010 | **CAVEAT** | HTTP/PIL pipeline + mock requests — network/mock semantics caveat |
| 96 | KEEP | CSV word-count — `os.path.exists` precheck breaks mocked-open tests, refineable |
| 224 | KEEP | Generator/plot/FFT — variable-shadowing/indexing bug refinable |
| 130 | KEEP | Salt/hash pipeline — digest length and escape-hex refinement |
| 781 | DROP | Wrong output schema (`size_bytes` vs `size`) — schema recall |

### Final tally

```
KEEP (capability-relevant refineable):  25   (13 claude + 12 codex)
CAVEAT (subprocess/network/mock):        5   (326, 146, 350, 810, 1010)
DROP (recall/hallucination/env):        15   (8 claude + 7 codex)

Conservative survivor count (CAVEAT→DROP): 25
Liberal     survivor count (CAVEAT→KEEP): 30
```

CAVEAT rows are grouped per codex's framing rather than mixed into KEEP — they share a single boundary condition (subprocess/network/mock semantics that mix env restriction with capability) and would be evaluated together for any RDT card scope decision.

## Decision

**Full E0 gate (≥10 capability-relevant survivors per `03_TESTING.md` §7 step 5): CLEARS WITH HEADROOM.**

- Conservative: 25 vs gate 10 = **2.5×** above
- Liberal: 30 vs gate 10 = **3×** above

**UT-family arc unlocks for E1 (RDT Tier-3 card training).** Per spec `03_TESTING.md` §7, the next bet is:

1. Train a small RDT-shaped card (~500K-2M params, depth recurrence + ACT halting + loop-index embedding per `01_ARCHITECTURE.md`) on multi-hop reasoning corpus derived from the 25 KEEP rows + augmentation
2. Install via `CardSlot` at L24 (per `Substrate.md` §"Card Installation" + `augmentation_thesis.md` §"L24 deep-diffuse")
3. Apply `VerificationHook` margin-gating
4. A/B against Gemma-alone on a held-out subset of the corpus

Falsifier (per spec §9): card standalone fails ⇒ thesis null at this scale; OR card succeeds standalone but Gemma+card flat ⇒ install-mechanism null.

## Independent findings (hold regardless of E1 outcome)

1. **Stock Gemma 4 E4B has substantial multi-library reasoning failure surface** confirmed at scale: 54% raw failure rate (54/100); after partition + filter 4: 25-30 capability-relevant prompts in N=100 random sample. Extrapolating to BigCodeBench's 1140-task corpus: ~285-340 substrate-relevant capability gaps. Material target population for any Tier-3 reasoning intervention, not just RDT.

2. **The substrate's L24 deep-diffuse compositional gap** (per `augmentation_thesis.md`) is now empirically characterized: shape is *multi-library coordination + multi-step state-tracking + edge-condition handling*. Failure modes that DON'T fit (recall/API-hallucination/schema-mismatch) account for 33% of the survivors-before-filter-4 — handled by other Tier-3 mechanisms (knowledge backends, retrieval).

3. **Codex's containment-design discipline was load-bearing twice**: smoke matplotlib reclassification (+11 spurious capability failures recovered) and full-E0 environment-unsupported partition (11 honest env rows correctly partitioned). Without the env_unsupported partition, full E0 would have ~22 false survivors. **Reusable invariant** for any future substrate eval running generated code through tests.

4. **34% pass rate reframes the substrate's commercial story.** Phase C confirms Gemma is a competent baseline (1 in 3 BigCodeBench multi-library tasks just works). The substrate's narrative shifts from "make Gemma not-broken" to "extend a competent baseline to specific failure shapes." Sharper Tier-3 scope; cleaner commercial framing.

## Method receipts

- **Plateau-detection rule worked again**: monitor caught the smoke `format_fail` pattern at 7/30 problems. Full E0 had 1 idiosyncratic format_fail (BigCodeBench/664, single occurrence — no pattern to chase).
- **Workflow's MAX_TOKENS budget rule held**: 6144 prevented thinking-mode budget exhaustion across all 100 prompts.
- **Reproducibility validated**: same seed=42 sampling produced same first-30 task set as smoke; pass-pass / fail-fail matches on the smoke-overlap (with the 5 matplotlib partition-shifts as the only delta).

## Collab receipts (full E0 round)

3-message coordination + 4-message implementation collab on board task `1777132726987-1d7fc6c6`:

> *"first smoke quantifies unsupported rows instead of silently mutating the machine"* — codex msg `1777128569287-0d806ff2` (smoke setup, applied here)

> *"refinement-loop benefit requires structured output it can iteratively improve"* — claude lift from R19 (`r19_d5_refinement_null.md:57-65`), msg `1777120398328-fd67f769` (filter-4 criterion)

> *"5 passed / 6 failed (0 thread failures remain)"* — codex Phase B replay receipt at msg `1777133748926-fd06590f` (validated in live Phase C run at problems 7/14/20/21/27 = 5 confirmed passes from smoke matplotlib rows)

Filter-4 cross-review concur: msg `1777138527268-012c905b` (codex on claude's half) + msg `1777138539670-c480665a` (claude on codex's half).

## Files (commit landed at `5046981` for runner patch; this report new)

- `agents/distill/data/bigcodebench_raw.jsonl` (1140 rows, gitignored — regenerable via `scripts/r53_fetch_corpora.py --sources bigcodebench_raw`)
- `scripts/r19b_e0_failure_surface.py` (committed `d5ab31a`, claude)
- `scripts/r19b_e0_restricted_runner.py` (committed `d5ab31a`, codex; patched in `5046981`)
- `scripts/r19b_e0_analyze.py` (committed `d5ab31a`, codex)
- `/tmp/e0_full_n100_results.json` (raw run output, not committed)
- `/tmp/e0_full_n100_classified.json` (analyzer output, not committed)
- `/tmp/e0_filter4_claude.md` + `/tmp/e0_filter4_codex.md` (filter-4 verdict files, not committed)
- `.claude/MEMORY/evals/2026-04-25_e0_failure_surface_full.md` (this report, NEW)

## Next step

E1 (RDT Tier-3 card training) is the justified next experimental round. Scope per `RESEARCH/UNIVERSAL_TRANSFORMERS/03_TESTING.md` §7 + §9. Time estimate per spec: 2-4 weeks. Falsifier conditions specified in spec §9.

Standing by for user greenlight on E1 scope or different direction.
