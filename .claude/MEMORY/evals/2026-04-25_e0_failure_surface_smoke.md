# E0 — Failure-Surface Scout (BigCodeBench N=30 smoke)

**Date**: 2026-04-25
**Spec**: `RESEARCH/UNIVERSAL_TRANSFORMERS/03_TESTING.md` §7 + §7a + §9
**Provenance**: claude+codex collab via ai-room board task `1777128240830-bc475f75`
**Runtime**: 23.7 min wall (1424.9s) for 30 generations + tests
**Decision**: **smoke gate clears** (7-8 capability-relevant survivors vs ≥5 threshold) → **proceed to full E0 (N=100-200)** AFTER runner matplotlib/thread fix

## What this measures

Cheapest gate on whether the UT-family arc (`RESEARCH/UNIVERSAL_TRANSFORMERS/`)
is worth pursuing for this substrate. Tests precondition (a) of the
RDT thesis: does a corpus of stock-Gemma failures exist on
**iteratively-refineable structured output** that is NOT
decode-path-addressable? If yes, RDT-shaped Tier-3 card becomes a
justified bet (Candidate E1 in the spec). If no, UT-family arc parks.

Tests **only** precondition (a). Does NOT test whether card-scale
RDT actually works (b) or whether install delivers (c). Those are
weeks of follow-on work gated on this measurement.

## Setup

| Component | Value |
|---|---|
| Corpus | `agents/distill/data/bigcodebench_raw.jsonl` (1140 records, native fields preserved) |
| Sample size | N=30 (smoke) |
| Seed | 42 (reproducible) |
| Model | `gemma-4-E4B-it-tq4-aligned.gguf` via llama.cpp at `localhost:8080` |
| Sampling | T=0.0, max_tokens=6144 (matches harness `medium` effort) |
| Runner | `scripts/r19b_e0_restricted_runner.py` (codex, restricted subprocess + sanitized env + rlimits + socket monkeypatch + `MPLBACKEND=Agg`) |
| Test python | `~/.venvs/e0_bigcodebench/bin/python` (16-package BigCodeBench dep set) |
| Analyzer | `scripts/r19b_e0_analyze.py` (codex, partitions + filters 1-3) |

### Pipeline (5 files shipped)

1. **Phase 0** (claude) — `scripts/r53_fetch_corpora.py` += `convert_bigcodebench_raw` → `bigcodebench_raw.jsonl` keeping native `task_id` / `entry_point` / `libs` (parsed from Python-literal string) / `doc_struct` / **full untruncated `test`** (avoiding the existing converter's 1800-char truncation)
2. **Phase 1a** (codex) — `r19b_e0_restricted_runner.py` 489 lines, sanitized-env subprocess runner with all containment guards
3. **Phase 1b** (claude) — `r19b_e0_failure_surface.py` calls Gemma + extracts via `dt_install_eval.extract_code` + invokes Phase 1a runner
4. **Phase 1c** (codex) — `r19b_e0_analyze.py` partitions raw_results into `solves_cleanly / fails_correctness / partial / format_fails / environment_unsupported`, applies conservative filter 3 (decode-path-addressable)
5. **Phase 2** (joint) — manual filter 4 (iteratively-refineable structured output) review

### One mid-flight pivot (~7/30 in)

Plateau detection per `workflow.md` killed the first run at 4 consecutive `format_fail` outcomes. Diagnosis: `GEN_MAX_TOKENS=800` (lifted from `dt_install_eval` MBPP/HE+ default) was too low — Gemma 4 E4B emits `reasoning_content` (thinking) on complex BigCodeBench prompts and burned the full 800-token budget on reasoning before producing any `content`. Violated `architecture.md` MAX_TOKENS budget discipline rule (≥4K required). Fixed: `GEN_MAX_TOKENS=6144` matches harness `medium` effort default. Inline comment cites BigCodeBench/501 verification. Restart at same seed.

## Raw runner outcomes

```
passed:           4   (BigCodeBench/228, 54, 859, 689)
failed:          24
env_unsupported:  2   (BigCodeBench/13, 865)
format_fail:      0   (post-fix)
timeout:          0
```

## Analyzer partitioning (Phase 1c)

```
solves_cleanly:           5   (≥80% tests pass; drop, ceiling)
fails_correctness:       16   (tests run, all fail)
partial:                  7   (20-80% tests pass)
environment_unsupported:  2   (drop, env not capability)

survivors_filters_1_3:   23   (fails_correctness + partial after filter 3)
```

## Critical correction — matplotlib thread reclassification

11 of 23 "fails_correctness/partial" survivors share identical stderr `can't start new thread` and are matplotlib-plotting tasks. **Not capability failures** — runner restriction (RLIMIT_NPROC=16 too tight for matplotlib's Agg backend internals). Mechanically reclassify as `environment_unsupported: matplotlib_thread_limit`:

```
51, 457, 209, 65, 61, 447, 476, 1034, 451, 919, 318
```

**Corrected partitions:**

```
solves_cleanly:           5
environment_unsupported: 13   (2 original + 11 matplotlib)
fails_correctness:        5   (was 16, -11 matplotlib)
partial:                  7

filter-4 review pool:    12   (fails_correctness + partial, post-correction)
```

## Filter 4 — iteratively-refineable structured output (manual)

Per `r19_d5_refinement_null.md:57-65` — refinement-loop benefit requires structured output the model can iteratively improve. Drop recall / API-hallucination / single-token retrieval failures.

| # | Task | Failure shape | Filter 4 |
|---|---|---|---|
| 1 | BigCodeBench/563 | DLL load + multi-step file ops, partial 33% | **KEEP** — multi-step composition |
| 2 | BigCodeBench/501 | JSON→Excel transformation, partial 33% | **KEEP** — multi-step transformation |
| 3 | BigCodeBench/1116 | Mean/median/mode calc, partial 60%, value off | **KEEP** — composition |
| 4 | BigCodeBench/191 | Pet shop simulation, partial 22% | **KEEP** — multi-step random sim |
| 5 | BigCodeBench/407 | Excel→CSV, traceback | **KEEP** — multi-step transformation |
| 6 | BigCodeBench/326 | Find + run .bat files, partial 20% | **KEEP_WITH_CAVEAT** — could be subprocess runner artifact |
| 7 | BigCodeBench/696 | Random points in circle, ValueError shape | **KEEP** — geometric refinement |
| 8 | BigCodeBench/440 | Matrix×tensor product, shape mismatch, partial 30% | **KEEP** — algebraic refinement |
| 9 | BigCodeBench/189 | URL parse + JSON extract, "Invalid url input" | **DROP** — recall (URL handling) |
| 10 | BigCodeBench/285 | Hallucinated `mechanize.Session` | **DROP** — recall (API hallucination) |
| 11 | BigCodeBench/864 | Invented `PrettyDict` type | **DROP** — recall (type hallucination) |
| 12 | BigCodeBench/569 | KeyError schema mismatch on output | **DROP** — recall (output schema) |

**KEEP: 8** (or 7 conservative if dropping 326)
**DROP: 4**

## Smoke decision

**Gate**: ≥5 capability-relevant survivors (per `03_TESTING.md` §7a).
**Result**: 8 survivors (7 conservative). **GATE CLEARS.**

UT-family arc unlocks. Next step: full E0 (N=100-200) → if ≥10 survivors there → E1 (RDT card training) becomes priority per spec §7.

## Action items BEFORE full E0

Codex's runner needs the matplotlib/thread fix to avoid distorting the larger corpus's classification:

1. **Raise or remove RLIMIT_NPROC cap** in `r19b_e0_restricted_runner.py` — current 16 too tight for matplotlib Agg backend internals
2. **Add `MPLCONFIGDIR=<tmpdir>`** to sanitized env
3. **Force `matplotlib.use("Agg")`** in child before candidate/test import when matplotlib is in deps
4. Re-run analyzer end-to-end (NOT report-only reclassification) to validate fix

Codex owns the runner patch. Analyzer doesn't need changes (correctly partitions whatever the runner reports).

## Independent findings (hold regardless of E0 outcome)

1. **Stock Gemma 4 E4B has substantial multi-library reasoning failure surface.** ~83% raw failure rate (24/30 `failed`) on BigCodeBench multi-library at T=0. Validates BigCodeBench as the right corpus shape for testing reasoning gaps. Independent quantitative confirmation that the L24 deep-diffuse compositional gap (per `augmentation_thesis.md`) is real and broad.

2. **`architecture.md` MAX_TOKENS budget discipline rule worked.** Without the workflow plateau-detection rule firing at 3-in-a-row format_fail, the run would have completed and produced the WRONG conclusion ("Gemma can't even produce code on this corpus"). Workflow caught the bug 7 problems in, not 30. Receipt for future evals: *if your eval looks pathological in the first 3-5 samples, stop and find the one wrong line before completing*.

3. **Codex's `environment_unsupported` partition was load-bearing.** Without it, the matplotlib thread issue would have been silently scored as "Gemma can't reason" — exactly the false signal the partition design prevents. Three round of design pushback on Round 3 to add this partition was correct architecturally.

4. **Venv coverage at 96%.** Pre-installed `~/.venvs/e0_bigcodebench/` (16 packages) covered the seed=42 N=30 sample with only 2 native env_unsupported (BigCodeBench/13, 865). Codex's "first smoke quantifies unsupported rows, no implicit machine mutation" call was vindicated — explicit setup, low pollution, runner stayed honest.

## Collab receipts

3-round design + 4-message implementation collab. Verbatim lifts that earned places:

> *"first smoke quantifies unsupported rows instead of silently mutating the machine"* — codex Round 3, msg `1777128569287-0d806ff2`

> *"refinement-loop benefit requires structured output it can iteratively improve"* — claude Round 2 lift from R19 (`r19_d5_refinement_null.md:57-65`), msg `1777120398328-fd67f769`

> *"invert the ranking: failure-surface gate before loop-index"* — codex Round 2 close, msg `1777120468569-94c180db`

Plateau detection that caught the MAX_TOKENS bug: msg `1777130669651-fd06529a` (claude diagnosis post). Cross-review concur on matplotlib reclassification: msg `1777132358247-97f22262` (codex final).

## Files shipped

- `scripts/r53_fetch_corpora.py` — Phase 0 converter additions (claude)
- `scripts/r19b_e0_restricted_runner.py` (489 lines, codex)
- `scripts/r19b_e0_failure_surface.py` (claude)
- `scripts/r19b_e0_analyze.py` (262 lines, codex)
- `agents/distill/data/bigcodebench_raw.jsonl` (1140 records, 6.9MB)
- `/tmp/e0_raw_results.json` (this run's raw output, not committed)
- `/tmp/e0_classified_results.json` (analyzer output, not committed)

All files unstaged; landing/commit deferred to user greenlight.
