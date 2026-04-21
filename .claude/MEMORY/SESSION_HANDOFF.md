# Session Handoff — 2026-04-21 (R22 arc: decode-path wins + adapter bug find + R22 shipped)

## Goal

Opened with "what's next?" after prior-session `/handoff` committed the
PT+Delta R13-R21 arc. Session evolved into:

1. **Housekeeping** — committed prior SESSION_HANDOFF.md + decode
   rebench validating 25.00 tok/s D-path (42 tok/s claim
   unreproducible across 2 sessions 72h apart).
2. **R22 arc** — installed R21 deployable MQAR card on prod Gemma.
   7-round debug arc (R22a mechanism → R22b rounds 1-7) + R22e
   standalone-card sanity diagnostic revealed the entire "67% fired
   precision" narrative was an adapter-regex bug, not calibration.
   **TRUE result: +9/60 (21% relative lift), zero regressions.**
3. **Decode-path compute facades** — R22c `BaseConversionFacade`
   generalized R46.2's parse+safe_eval+step-through pattern to a new
   domain. 30% clean lift in 119s. Pattern-as-product validated.
4. **/update docs pass** — 3 commits (P0/P1/P2+fixup) rewriting stale
   CardSlot/R22 claims across 7 rule files + new
   `.claude/rules/compute_facades.md`.

Ended at a clean stopping point: 15 commits, no in-flight work, all
receipts + rule updates landed. "Mega session tomorrow" is set up
with 3 ordered queues (see Next Steps).

## Completed (15 commits, `946652c` → `c52cadc`)

### Session + housekeeping (2 commits)

| SHA | Purpose |
|---|---|
| `946652c` | Commit prior R13-R21 SESSION_HANDOFF.md |
| `6c66ffa` | Decode 4-path rebench — D=25.00 tok/s bit-identical to prior session (72h gap), 42 tok/s / 90% llama unreproducible. Receipt: `.claude/MEMORY/evals/2026-04-21_decode_paths_rebench.md` |

### R22 arc (10 commits)

| SHA | Round | Δ | Receipt |
|---|---|---|---|
| `8150e97` | R22a mechanism | 3/3 sanity | `2026-04-21_r22a_mqar_card_install.md` |
| `17024b8` | R22b r1 | -2 null (500-tok too easy) | `2026-04-21_r22b_round1_no_failure_surface.md` |
| `bcdf5d9` | R22b r2 | -2 (2W 4R, failure surface found) | `2026-04-21_r22b_round2_mixed_signal.md` |
| `ff2ddf6` | R22b r3 | "+2" post-hoc (FLAWED) | `2026-04-21_r22b_round3_margin_threshold.md` |
| `36de25d` | R22b r4 | 0 held-out | `2026-04-21_r22b_round4_holdout.md` |
| `7a7f347` | R22b r5 | -4 (residual-write bug exposed) | `2026-04-21_r22b_round5_6_gate_fix.md` |
| `e169d6d` | R22b r6 | 0 (write-gate fix) | same |
| `7db6eb9` | R22 close | r7 +1 + R22c +3 + R22d +1 + noise scaffolding | `2026-04-21_r22b_r7_r22cd_summary.md` |
| `c3eac18` | R22e sanity | **Adapter bug diagnosed** (100% standalone after 5-line regex fix) | summary receipt |
| `73df738` | **R22 TRUE result** | **+9/60 (21% rel), 0 regr** | `2026-04-21_r22b_r7_rerun_adapter_fixed.md` |

### Final R22 install config (deployable)

```python
install(m, card, layer_idx=30, ch_off=2480,
        write_margin=22.0, preserve=False)
hook.min_margin = 22.0
# Adapter: CARD_N_RANGE = {5, 10, 15}  (training distribution gate)
```

Four aligned gates. Per-cell on 60-prompt pooled distractor-confused
corpus:

```
   N    dist   mode              base    card    Δ
   5    500   confusing          5/10   10/10   +5
   5   1500   confusing_long     8/10   10/10   +2
  10    500   confusing          7/10    7/10    0   ← open follow-up
  10   1500   confusing_long     9/10    9/10    0   ← open follow-up
  15    500   confusing          7/10    8/10   +1
  15   1500   confusing_long     6/10    7/10   +1
                        OVERALL: 42/60 -> 51/60  Δ=+9 (21% rel, 0 regr)
```

### /update docs pass (3 commits)

| SHA | Tier | Files |
|---|---|---|
| `a2bade0` | P0 | `delta_rule.md` R22 section rewritten with TRUE result + R-delta-22 CANCELLED; `CLAUDE.md` three-install typology; `Substrate.md` preserve=True legacy caveat + R22 4-gate install workflow |
| `3a05198` | P1 | NEW `.claude/rules/compute_facades.md` (122 LOC); `workflow_part_1.md` adapter-robustness lesson; `augmentation_thesis.md` Tier-2 table updated; `embed_intelligence.md` min_margin tuning + write_margin alignment |
| `c52cadc` | P1-fixup + P2 | `CLAUDE.md` + `architecture.md` three-install cleanups; `tracing_roadmap.md` R-delta-22 CANCELLED row |

## In Progress

**None.** All 15 commits landed cleanly. Daemon running (`bin/gemma-run
--status` = PID 929999, warm), no orphan scripts, no uncommitted session
work at risk.

## ⚠ Uncommitted

All non-session (runtime caches + gitignored-adjacent). No
session-critical uncommitted files:

```
?? .cache/                       runtime cache (retrieval index, tq4 blocks, r22b JSONLs)
?? .claude/MEMORY/minutes/       auto-rotated session transcripts (3 files from today)
?? .claude/scheduled_tasks.lock  runtime lock
?? .codex/                       runtime
?? .port_sessions/               runtime
?? calm/.module_learning.json    runtime state
```

**Risk: NONE.** All shipped code, docs, and receipts are committed.

## Next Steps — two-phase arc, "ship frontier coding primitive → bootstrap recursive self-improvement"

**User direction (2026-04-21 end-of-session):** "i think we do it all,
ship frontier coding as first primitive then we use gemma for
recursive self improvement"

The full arc has two phases. Phase A ships the substrate capabilities
that collectively constitute "frontier coding primitive" on verifiable
workflows. Phase B uses the shipped primitive to recursively improve
the substrate itself (per `recursion.md`). Phase B requires Phase A
complete.

**Realistic scope**: Phase A is 15-25 hours = 1 long mega session or
2 normal ones. Phase B is another 8-15 hours once Phase A unlocks it.
Total: 2-3 sessions to go from today's state to shipping recursive
self-improvement demo.

### Phase A — frontier coding primitive (ordered by unlock dependency)

### 1. N=10 flat — the one remaining R22 diagnostic (15 min)

R22 shipped +9 with all gains in N=5 and N=15. N=10 cells completely
flat (7/10 and 9/10, unchanged by card). Three possible causes:

- (a) card's margin < 22.0 on those specific prompts — gate stays
  silent, Gemma's wrong answer unchanged
- (b) Gemma's wrong-answer logit gap > 50 boost — hook fires but
  can't override
- (c) adapter-regex remnant bug specific to N=10 key patterns

Diagnostic: write `scripts/r22f_n10_diag.py` that runs the card
standalone on r22b corpus's 20 N=10 prompts + logs card's argmax +
margin + parse_ok. 5 min script + 1 min run on warm daemon.

### 2. Add a second compute facade (1-3 hours)

`compute_facades.md` lists candidate queue. Pick ONE:
- **Modular arithmetic** (simplest — extends R46.2's `%` support
  with NL alias `mod` → `%`)
- **GCD/LCM of multi-digit ints** (new domain, clean Gemma failure
  surface)
- **Days-between dates** (new domain, uses `date_ops` backend)

Build pattern: copy `base_conversion.py`, swap parser + evaluate, 10-
probe A/B. Target Δ ≥ 20% with zero regressions. Commit as P1
with before/after table. Reinforces compute-facade-as-product line.

### 3. R22c + R22d RERUN with fixed adapter (30 min)

- R22c BaseConversionFacade didn't use the buggy adapter (uses its
  own parser), so no rerun needed for correctness — but worth
  confirming its 10/10 on a fresh run with this daemon.
- R22d N-fold retrieval test DID use the buggy adapter; needs rerun.
  Expected with fixed adapter: card lift across many more keys, not
  just the surgical 1 mem / 1 key result we saw.

### 4. (DEFERRED) Track A decode kernel queue

From `SESSION_HANDOFF_1.md` (pre-this-session):
- Round 6: q+k+v triple fusion (+3-5% expected)
- Round 7: per-shape autotune (BLOCK_M/num_warps/num_stages) (+5-10%)
- Round 8: fused flash-attn TILE_N blocking (+5-15% at large N)
- Round 9 (weeks): fused attention-layer mega-kernel

These remain parked. Decode bench stable at 25.00 tok/s across 2
sessions; no regression pressure. Revisit after product wins from
R22-follow-ups + compute facades compound enough commercial surface
to justify kernel deep dives.

### 5. (CANCELLED) R-delta-22 noise-augmented training

R22e invalidated this work stream. Scaffolding stays in tree
(`calm/hrm/memory_tasks.py::_gen_mqar_noisy` +
`scripts/train_pt_delta_mqar.py --noisy-frac`) for FUTURE cards
that genuinely show distribution shift AFTER adapter verification.
Do NOT retrain R21 — card is 100% standalone on clean adapter
outputs.

### 6. Tier-3 validation: ICD-10 recall card (4-6 hours, the moat demo)

R22 + R22c are both **tier-2** (augmenting weak Gemma circuits).
The substrate's moat claim requires **tier-3**: a capability where
Gemma has ZERO relevant prior. ICD-10 medical coding is the ideal
first demo:
- Gemma has no reliable mapping from specific codes (`J45.909`) to
  diagnoses — it fabricates plausibly-wrong answers. Per
  `augmentation_thesis.md §"Customer verticals = card decks"`,
  hospital stack = ICD-10 validator + drug-interaction DB + exact
  dosage.
- Pure memorization task — no compute, no reasoning. Perfect fit
  for `KnowledgeStore` + `build_recall_model()` compiled recall
  card (step-function indicators, 3 ReGLU per fact, proven pattern
  from session 30).
- Zero-shot Gemma failure rate on non-common codes is expected to
  be 60-80%. Card at 100% recall = **50pp+ absolute lift**,
  dwarfing R22's 15pp and R22c's 30%.

Canonical workflow:
1. **Failure-surface gate** (45 min) — pick 100 real ICD-10 codes
   from the public CMS 2024 code set (mix common + rare). Score
   stock Gemma on `"What is ICD-10 code <CODE>?"`. Keep prompts
   Gemma fails on. Target: ≥ 50 fail cases.
2. **Build recall card** (15 min) — `KnowledgeStore` with (code, diagnosis)
   pairs for the fail corpus. `build_recall_model()` compiles to
   `Small2DTransformer` recall card (d_model tuned to code-hash
   space). Install via `CardSlot(preserve=False, ...)` at L30 with
   `VerificationHook` biasing the DIAGNOSIS output (multi-token
   step-through bias per R11/R46.2 pattern).
3. **Adapter** (~1 hour) — pattern: `/ICD-10 code (\w\d+(?:\.\d+)?)/i`
   extracts the code from NL prompt. Hash to KnowledgeStore key.
   Simpler adapter than R22's MQAR — no distractor issue, single
   literal extraction.
4. **Live A/B** (~1 hour) — baseline Gemma on the 50-fail corpus,
   then with card. Expected: **50/50 card vs ~0/50 baseline** since
   card is exact on the failure set by construction.
5. **Commit with receipt**: "tier-3 validation, ICD-10 recall card
   50/50 on Gemma-fail corpus." Doc update: `commercial.md`
   §Customer verticals hospital stack now has concrete receipt;
   `augmentation_thesis.md` Tier-2/3 table gets a shipped tier-3
   row.

Risk: code-to-diagnosis text is long and tokenizes across many
Gemma BPE tokens. Step-through bias over 20-token diagnoses needs
the output sequence pre-computed. R11/R46.2 mechanism extends to
this but hasn't been tested at that length — may need
`max_tokens=80+` and careful start-bias timing.

If tier-3 ships tomorrow, we have: (a) tier-1 preserve free, (b)
tier-2 atlas across 2 install patterns with 4 shipped wins, (c)
tier-3 demo on a commercial-critical domain. That's the commercial
pitch deck complete.

### 7. Planner card (4-8 hours) — orchestration layer for multi-step workflows

The frontier-multi-step-coding path per `tracing_intelligence.md`
§"multi-step planning": a **decode-path PlannerFacade** that
decomposes NL goals into ordered facade calls. Tier-2 stacking
taken to its logical conclusion — orchestrate R46.2 (math) +
R22c (base conv) + `ast_repair` walker + code retrieval +
`KnowledgeStore` recall all from a single prompt.

**MVP scope (Option A — pure decode-path facade, no compiled card):**

Create `calm/llm_computer/facades/planner.py` following R46.2
skeleton:
- `parse(prompt)` → `(template_kind, sub_task_spec)` — regex
  catalog of task templates
- `orchestrate(spec)` → list of facade calls in order
- `_generate(prompt, orchestrated_steps, boost, max_tokens)` —
  executes each sub-facade, biases Gemma's decode to emit each
  step's result inline

Initial template catalog (5 classes, ~30 min each):
1. **Math + conversion chain** — "Compute a+b, convert result to
   hex" → R46.2 then R22c
2. **Bug + test** — "Fix this TypeError in <code>, then run tests"
   → `ast_repair` walker then sandbox
3. **Lookup + adapt** — "Find MBPP solution for task X, adapt to
   signature Y" → CodeExampleDB retrieve then ast_repair rename
4. **Data transform pipeline** — "Parse CSV, compute column
   mean, format as JSON" → CALM csv_ops + data_ops + json_ops
5. **Code gen + verify** — "Write function to do X, verify
   against tests" → Gemma natural + sandbox run_python + AST
   repair on failure

**Failure-surface gate first** (30 min):
- 10-15 multi-step prompts, each mixing 2-3 sub-facades
- Score stock Gemma end-to-end (check sub-task errors compound:
  e.g. wrong arithmetic then wrong conversion)
- Keep where Gemma fails ≥ 1 sub-step

**Measurement** (1 hour):
- Baseline: Gemma natural decode on the fail corpus
- With planner: orchestrated facade calls
- Target: **Δ ≥ 30% lift**, 0 regressions
- Per-sub-facade: track which sub-call delivered the fix

**Risks + mitigations:**
- *Template catalog too narrow* — starts with 5 classes; add
  more as failure modes surface. Don't block on exhaustive
  coverage.
- *Facade output integration* — R46.2 biases single numeric
  answer; planner needs to bias MULTIPLE intermediate results
  into Gemma's stream with markdown structure. May need a
  `bias_with_marker` extension (emit `<step>value</step>` then
  continue) — similar to R22c's `"Answer: "` suffix trick.
- *Step coordination* — Gemma may "finish" before planner's
  last sub-task fires. Use `max_tokens` aggressive (150+) and
  stop-on-pattern guards to hold the window open.

**Design path if MVP works:**
Option C (compiled planner card with channel-as-register state)
becomes the follow-up — `programs/planner.py` with
`dispatched_v4` opcode pattern routes sub-tasks to facade slots
via residual channels. That's tier-2 compiled (not designed from
scratch at a missing slot, so technically tier-2 by the circuit
typology; but it does COMPOSE existing facades in a way pure
decode-path can't — longer horizon, auditable state trace).

**Commercial framing** once shipped: "Substrate orchestrates
exact-compute facades via a parseable planner — the output is
inspectable as a sequence of deterministic calls with CALM
verification at each step. GPT-4's `<think>` block is opaque;
ours is a sequence diagram."

**Queue order for tomorrow**: do #6 ICD-10 first (tier-3 moat
demo is faster to prove out), then #7 planner. Combined, they
ship as the "Brain + Cards orchestration" story: tier-2 + tier-3
+ orchestration layer = frontier-competitive surface on
verifiable workflows.

### Phase A complete = frontier coding primitive shipped

Once items 1-3, 6, 7 ship (skip 4 parked + 5 cancelled), the
substrate has:
- R22 retrieval card (tier-2, CardSlot)
- R22c + 2nd compute facade (tier-2, decode-path)
- ICD-10 recall card (tier-3, CardSlot)
- PlannerFacade orchestration (tier-2 stacking composition)
- + pre-existing: CodeExampleDB, ast_repair walker, CALM 1002
  backends, sandbox, R46.2 multi-step math, R11 multiplier

**Integrated narrative**: given an NL task, PlannerFacade
decomposes → per sub-step invokes the right facade (compute /
retrieval / AST repair / recall) → CALM verifies each step →
outputs verifiable sequence diagram instead of opaque
`<think>` block. That's the frontier-multi-step-coding primitive
on **verifiable** workflows. Match GPT-4 on open-ended no, beat it
on auditable yes.

### Phase B — recursive self-improvement (follow-on session)

Per `.claude/rules/recursion.md`. Once Phase A primitives ship,
the substrate can use its OWN coding capability to extend itself.
Level 1 (card self-distill) already exists (`auto_upgrade.py`).
Level 2 (cards build cards, or MetaCard) is unshipped.

**Minimum viable recursive loop (Phase B MVP, ~8-12 hours):**

```
1. Substrate fails at domain X (CALM catches wrong answer)
2. Substrate retrieves similar existing facades from CodeExampleDB
   (8970 examples already indexed, hybrid retrieval already built)
3. Substrate generates new facade code via Gemma + PlannerFacade
   (uses Phase A's orchestration to write the facade)
4. Substrate runs new facade in sandbox (run_python exists)
5. Substrate validates via CALM oracle (safe_eval + tests)
6. If valid: add to facade registry, persist. If invalid:
   AST repair walker retries; if still bad, defer to human.
```

Integration targets:
- `AutoUpgradeEngine.commit()` already compiles recall cards;
  extend to commit NEW FACADE FILES to
  `calm/llm_computer/facades/*.py` with test corpus receipts
- `PlannerFacade` orchestrates the write-verify-install loop
- `CodeVerifierFacade` becomes the CALM oracle for new facades

**Why safe**: every card in the recursion chain is gated by
deterministic CALM tests (`recursion.md §"Why this is safe where
Self-Instruct / RLAIF fails"`). Cannot amplify drift the way
self-instruct / RLAIF does. Whatever survives has PASSED
objective correctness checks.

**Capability-completeness fixed point**: as recursion continues,
card library grows → each card covers more of its domain → Meta
gets better at spotting gaps → MetaMeta gets better at designing
meta variants. Asymptotically: for every task with verifiable
success criterion, substrate has a card that solves it exactly.
Per `recursion.md §"Capability completeness as a fixed point"`.

**Phase B queue (not numbered yet — activate after Phase A ships):**

- **B1**: Extend `AutoUpgradeEngine.commit()` to write facade
  files (not just recall card weights)
- **B2**: `MetaFacade` that given a failure trace proposes a new
  facade template (decode-path skeleton adaptation)
- **B3**: End-to-end Phase B demo: pick a domain Gemma fails at
  AND no facade exists for, run Phase B loop, measure if
  substrate self-builds a working facade. **This is the "it's
  self-improving" demo**.
- **B4**: Level 2 MetaCard — automate circuit probing from failure
  traces, route to appropriate tier (2 or 3)
- **B5**: Level 3 MetaMetaCard — watch MetaCard's failure modes,
  design variants. Speculative but tractable once B3 lands.

**Realistic timing**: each B-step is 1-3 hours. B1-B3 together =
~8-12 hours, shippable in one post-Phase-A session. B4-B5 are
weeks-scale research.

## Key Context (for cold-start tomorrow)

### The adapter-bug finding (the session's pivotal discovery)

R22b rounds 1-7 debugged the wrong thing. "Card confident-wrong at
67% precision" was actually "adapter picked wrong query key from
distractor prose". `parse_mqar_prompt`'s `_QUERY_RES` regex
`r"value of\s+([a-z])"` matched `"Previously the value of q rose to
2..."` inside distractor sentences BEFORE the real
`Question: What is the value of d?` at prompt tail. Adapter fed the
card the wrong question → card gave the right answer for the wrong
question → looked like a calibration gap.

Fix (commit `c3eac18`, 5 lines):
```python
question_idx = post_mem.lower().rfind("question:")
search_region = post_mem[question_idx:] if question_idx >= 0 else post_mem
# ... run query regexes on search_region, not full post_mem
```

30-second standalone diagnostic (`r22e_card_standalone_sanity.py`)
exposed it: card 14/33 before fix, 60/60 after. This lesson is now
canonical in `workflow_part_1.md` §"Adapter-robustness — the R22e
lesson".

### The decode-path vs CardSlot vs in-tensor typology

Session clarified three distinct install paths — documented in
`compute_facades.md` + `Substrate.md` §CardSlot + `CLAUDE.md`
§Brain+Cards. Rule for tomorrow:

- **Deterministic compute** (arithmetic, hex/binary, GCD, dates,
  financial) → decode-path facade. Zero VRAM, zero training,
  stacks freely, ~2-4 hours per new domain. **Ship first.**
- **Retrieval** (key→value lookup, NIAH-style) → CardSlot with
  R22 4-gate install. Days per card (train + adapter + tune +
  install).
- **Attention-circuit replacement** (R28/R42/R43 forced-attn
  compiled cards) → in-tensor `install_card_in_attention`.
  Requires FP32 host (~600 MB per layer).

### R22 TRUE result vs interim receipts

R22b rounds 1-7 receipts (kept as per-round debugging history)
report pre-adapter-fix numbers: -2/0/+2-post-hoc/0/-4/0/+1. All
superseded by `73df738`'s rerun-with-fixed-adapter result:
**+9/60, 21% rel, 0 regressions**. Agent-3 audit 2026-04-21
cross-referenced all files for stale numbers; 0 remain after
P0/P1/P2 commits.

### Methodology lesson banked

**Always run raw path (card standalone) on REAL adapter outputs,
not hand-crafted sanity strings.** R22a's `"a 3 b 7 c 1 ; b"`
sanity was fine but skipped the adapter. Had I run the card on
60 actual r22b prompts at round 2, the adapter bug would have
been visible immediately and the entire 6-round debug arc
avoided. Codified in `workflow_part_1.md` §"Always check two
things" card-install row + dedicated subsection §"Adapter-
robustness — the R22e lesson".

### Failed approaches (cite SHAs, don't retry)

- `17024b8` R22b r1: 500-tok neutral distractor doesn't break Gemma
- `ff2ddf6` R22b r3: post-hoc threshold sweep assumed hook-silent
  == no-card-installed. **Wrong assumption**; CardSlot's residual
  write fires independently. Only trust LIVE A/B at each threshold
  tested.
- `7a7f347` R22b r5: pre-adapter-fix "preserve=True pins channels"
  diagnosis was correct as far as it went (r6 fixed residual write,
  r7 fixed preserve). BUT the dominant effect was the adapter bug;
  fixing install alone would have topped out at Δ=+1 forever.
- R-delta-22 noise-augmented training: CANCELLED. Scaffolding
  kept; do not run on R21 card.

### Runtime state at session end

- Branch: `feature/multi-agent-qwen` at `c52cadc`
- HEAD: "docs(P1-fixup + P2): three-install typology in CLAUDE/architecture + R-delta-22 ruled-out"
- Gemma daemon: RUNNING (PID 929999, warm from today's runs)
- GPU: 12% util, 7863 MiB used / 86 MiB free (mostly model weights
  + tq4 preload at 5.17 GB + daemon activations)
- `/tmp/gemma_log`: ~26 KB accumulated from today's R22 arc
- `.cache/r22b/` present with results jsonls from 7 rounds + sanity

### Architecture.md claims that were validated

- 25.00 tok/s D-path decode (2026-04-21, 72h-stable across sessions)
- Gemma 4 E4B + R22 MQAR card at 4-gate config delivers +9/60 on
  distractor-confused retrieval
- R46.2 parse+safe_eval+step-through pattern generalizes (R22c
  BaseConversionFacade clean +3/10)

## Files in Project (session-shipped)

### New files

- `calm/llm_computer/facades/base_conversion.py` — R22c
  BaseConversionFacade (hex/binary → decimal)
- `.claude/rules/compute_facades.md` — new rule (decode-path facade
  typology + two proven instances + candidate queue)
- `scripts/r22_install_mqar_card.py` — R22 adapter + install
  helpers (280 LOC; `install()` + `parse_mqar_prompt()` + closures)
- `scripts/r22b_round{2..7}_preserve_false.py` — per-round test
  scripts; round 7 is the final canonical config
- `scripts/r22c_base_conversion.py` — 10-probe A/B test
- `scripts/r22d_nfold_retrieval.py` — 60-probe N-fold diagnostic
- `scripts/r22e_card_standalone_sanity.py` — **the 30-second
  diagnostic that exposed the adapter bug**. Keep as template.

### Receipts (`.claude/MEMORY/evals/`)

Eight new receipts for this session:
- `2026-04-21_decode_paths_rebench.md`
- `2026-04-21_r22a_mqar_card_install.md`
- `2026-04-21_r22b_round1_no_failure_surface.md`
- `2026-04-21_r22b_round2_mixed_signal.md`
- `2026-04-21_r22b_round3_margin_threshold.md`
- `2026-04-21_r22b_round4_holdout.md`
- `2026-04-21_r22b_round5_6_gate_fix.md`
- `2026-04-21_r22b_r7_r22cd_summary.md` (interim r7/r22c/r22d)
- **`2026-04-21_r22b_r7_rerun_adapter_fixed.md`** — TRUE result
  receipt (superseding all interim r7 claims)

### Modified code

- `scripts/train_pt_delta_mqar.py` — `--noisy-frac` flag (scaffolding)
- `calm/hrm/memory_tasks.py` — `_gen_mqar_noisy` + `gen_mqar_batch_noisy`
  (scaffolding, R-delta-22 CANCELLED but code kept)

### Modified docs (via /update, 3 commits)

- `.claude/CLAUDE.md` — three-install typology
- `.claude/rules/delta_rule.md` — R22 shipped section rewritten with
  TRUE result + R-delta-22 CANCELLED section
- `.claude/rules/Substrate.md` — preserve=True legacy caveat + R22
  4-gate install workflow
- `.claude/rules/architecture.md` — R22 install references updated
- `.claude/rules/augmentation_thesis.md` — Tier-2 table updated with
  R22 shipped + BaseConversionFacade row
- `.claude/rules/workflow_part_1.md` — adapter-robustness lesson
- `.claude/rules/embed_intelligence.md` — min_margin tuning +
  write_margin alignment
- `.claude/rules/tracing_roadmap.md` — R-delta-22 CANCELLED row

## Handoff verification

- Main context claims vs git state: **match.** All 15 commits
  confirmed via `git log --oneline -15`.
- `.claude/` line counts: all files under 500 LOC ceiling.
- `compute_facades.md` (new) referenced from CLAUDE.md,
  augmentation_thesis.md.
- `R-delta-22` CANCELLED in 2 places (delta_rule.md, tracing_roadmap.md).
- No stale `+1` / `43/60` / `2W 1R` claims remain; `+9` / `51/60` /
  `21% rel` / `9W 0R` is canonical.
- Final audit found 0 gaps from session → docs (3-agent audit
  pass + verification grep confirmed).
