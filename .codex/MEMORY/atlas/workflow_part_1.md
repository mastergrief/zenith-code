**IMPORTANT**: Assume nothing. Hypothesis, Build, Test, Commit & Iterate. First Principles thinking. Do not discount anything until it's built and tested!

Workflow Part 1:

**This is the default working loop for all work in this project.** Apply
it to CUDA kernels, Python harness changes, training scripts, quantize
runs, config edits — anything that has a measurable outcome. The
session-16 tq4 GEMV optimization went 7.9 → 37.7 tok/s in one sitting
using this loop; the same pattern applies to every non-UI task.

## Core principle: it works or it doesn't, it's better or it isn't

**No vibes.** Every claim of "done", "working", "better", "fixed" must
be backed by a measurement that came *after* the change. If you can't
point to a number, a passing test, a correct output, or a diff on an
artifact — you don't know.

- "Looks right" — not done.
- "Should be faster" — not better.
- "I think it's working now" — not working.
- "Probably fine" — not fine.

The only exit from an iteration is a measurement that says "yes, it
moved" or "no, it didn't". Intermediate states (half-applied edits,
unrun tests, partial builds) aren't progress — they're liabilities.

If there's no way to measure it (pure UI/frontend judgment calls,
stylistic choices, brand tone), say so explicitly and ask the user to
make the call. Everything else gets measured.

## The loop

1. **State the hypothesis.** One sentence: "X is the bottleneck / bug
   / missing piece because Y. Changing it to Z should move metric M
   by N%." Write this down in the conversation, even when obvious —
   it's the thing you'll compare against later.
2. **Pick the measurement first, not after.** Decide what number or
   assertion would *prove* the hypothesis before you write any code.
   If you can't specify one, the hypothesis is too vague; sharpen it.
3. **Minimal edit.** Smallest change that tests the hypothesis. Don't
   bundle unrelated improvements — they'll poison the measurement.
4. **Build / run / test.** Whatever turns the edit into something
   measurable.
5. **Measure.** Run the test you decided on in step 2. Run it twice
   if the metric is noisy.
6. **Binary decision.** Real movement (> noise, matches the
   prediction) → ship. Didn't move or moved wrong direction → revert.
   No "maybe", no "we'll come back to it".
7. **Ruled-out log.** If you reverted, write one line in the session
   notes saying what you tried and what the delta was. Future-you
   (and other sessions) won't retry a known-bad path.
8. **Next hypothesis.** Repeat.

The loop should be fast — under 5 minutes for small changes. If it
takes longer, the measurement is usually too heavy; find a lighter
proxy (a unit test, a single-function benchmark, a one-prompt chat
call instead of a full eval).

## Empirical timeline — minutes to hours, NOT "weeks to months"

Project's measured pace on this stack is minutes-to-hours, not the
weeks-to-months inherited from mechinterp literature. If a step
looks like it'll take days, your methodology or tooling is wrong;
revisit before committing the time. Full detail + reference points:
`.claude/rules/probing_methodology.md` §"Empirical timeline".

## Always check two things

Wherever possible, run a **raw-path measurement** and a
**user-facing measurement** on the same change. This is the single
most important discipline in this workflow.

**Why two:**
- A win on the raw path that doesn't move the user-facing path means
  the thing you fixed wasn't on the critical path. Don't ship.
- A win on the user-facing path that isn't in the raw path is usually
  measurement noise. Don't ship.
- Both move → real.

**Concrete examples:**

| Work type | Raw path | User-facing path |
|---|---|---|
| CUDA kernel opt | `llama-bench -n 64 -p 0 -r 3` | `curl /v1/chat/completions` w/ fixed prompt |
| Python harness fix | unit test / pytest | `printf "prompt\n/exit\n" \| zenith --effort max` |
| Config / prompt change | model output on N fixed test prompts | real conversation turn |
| Training data filter | schema validation + dedup count | loss curve on a few hundred steps |
| llama.cpp build flag | `llama-bench` tg + pp | actual inference of a long prompt |
| New CALM backend | function count + `pytest calm/tests/` | Gemma `run_auto()` — does precompute fire? |
| Cognitive module | router on flawed response (quality + issue count) | Engine V2 Gemma test — quality gap bad vs good |
| NL pattern | NL pattern count + precompute on test prompt | Gemma test — correct value injected? |
| Scoring/threshold | flawed response < 75%, good > 90% | self-heal trigger test on bad response |
| Substrate card install | card.forward standalone on REAL adapter-extracted inputs (not hand-crafted sanity) | Gemma + card A/B vs Gemma baseline on the SAME corpus |
| Triton kernel | bit-equivalent to PyTorch path (max abs diff < 1e-5) | end-to-end tok/s on a 30+ token decode |

If you only have one path, say so out loud and accept the reduced
confidence — but keep looking for the second.

### Adapter-robustness — the R22e lesson (2026-04-21)

**If a card install shows low effective precision on live inputs,
run the card STANDALONE on the adapter's extracted strings BEFORE
hypothesizing calibration, distribution-shift, or architectural
gaps.** The R22b arc burned 6 rounds tuning thresholds and installing
margin gates on a "67% card precision" signal that turned out to be
a 5-line regex bug in the adapter picking the wrong query key from
distractor prose containing `value of X` phrasings. 30-second
standalone diagnostic (`scripts/r22e_card_standalone_sanity.py`
pattern) exposed it: card was 100% on correctly-extracted inputs.

Rule: for install work with an adapter or parser, the two-measurement
pair is **(raw on REAL adapter outputs) + (user-facing A/B)**, not
(raw on synthetic inputs) + (user-facing). Synthetic sanity cases
(R22a's `"a 3 b 7 c 1 ; b"`) skip the adapter entirely, so adapter
bugs are invisible.

Sibling commits worth citing when this rule applies:
- `c3eac18` — R22e adapter-regex anchor fix (5 lines)
- `73df738` — TRUE result post-fix: Δ=+9 vs pre-fix Δ=+1

## Plateau detection

**When 3 iterations in a row each give < 2% movement → there's usually
one bug, not a tuning problem.** Stop micro-tuning and go looking for
one wrong line of code.

Symptoms you're in a bug-not-tuning situation:
- The ratio between your path and a comparable known-good reference is
  surprisingly large (> 2× for perf, or you're off by suspicious
  constants for correctness).
- Occupancy / resource budgets say there's headroom but you're not
  using it.
- Compute/bandwidth math says the thing should be N× faster than you
  see.

Session-16 example: 6 micro-opts in a row each moved 0–3% on a ~24
tok/s baseline. The real fix turned out to be one line — caching a
16-entry `__constant__` LUT in per-thread registers to avoid divergent-
index serialization on Ada — which moved +58% by itself. Plateaus
mean "go look for a bug", not "keep tuning the knob you were tuning".

## MAX_TOKENS budget discipline (R53.25 receipt)

Before diagnosing logic / substrate / sandbox / import failures,
verify output budget isn't clipping. Gemma 4 E4B trains at 131K
ctx, NIAH-validates at 220K — eval `max_tokens` defaults should
be ≥ 4K, not ≤ 400. R53.19v3 through R53.24 burned four null
rounds (substrate install, sandbox fix, import injection,
KVCacheTq4) before R53.25 showed MAX_TOKENS 400 → 900 alone lifts
`log_level_counts` 0/0 → 6/6 on R53.0. Budget was first-order
cause; every other "failure" was downstream of truncation.

**Centralized**: `calm/llm_computer/eval_defaults.py` exports
`EVAL_CTX_SIZE=32768` (pre-allocated tq4 KV), `EVAL_MAX_TOKENS=16384`
(AdaptiveBudget clamp), `ITERATION_N=5` (fast-iteration problem
count), and `FINAL_N=20` (commit-baseline problem count). Every
R-series script imports from here — changing the numbers changes
every eval consistently. Full spec + exception list (r51/r52
dual-gate `K_TOKENS=12` is a measurement design, not a budget):
`training.md` §"Substrate eval defaults".

**ITERATION_N / FINAL_N pattern**: every eval with a configurable
problem count defaults to `ITERATION_N` for the hypothesis-test-
iterate loop (fast feedback, ~10 min wall time on Gemma 4 E4B tq4
substrate). Bump to `FINAL_N` only for the round that goes into a
commit receipt — that's the measurement a future Claude reads from
the commit body to judge whether the change shipped. Pattern:
`MBPP_N = FINAL_N if os.environ.get("MBPP_FINAL") == "1" else ITERATION_N`
lets a single env-var flip toggle between modes. Never iterate at
FINAL_N — 40-min rounds violate the <5 min loop target.

Rule: when a Gemma failure is "no output / NoCode", check
`max_tokens` ≥ prompt + `<think>` + expected output BEFORE any
deeper diagnosis. When adding a new eval script, import from
`eval_defaults` rather than picking a number locally — the
pre-R53.25 200-400 defaults cost four null rounds of misdiagnosis.

## GPU bench discipline (R53.29 receipt)

Triton kernel bench variance on the 4070 Laptop is 20-30%
run-to-run without GPU stabilization. Protocol:

1. `heavy_warmup(3.0s)` — dense fp16 matmul loop to steady-state
   clock before timing.
2. `torch.cuda.Event(enable_timing=True)` not `time.time()` —
   GPU-side timestamps, immune to host jitter.
3. Median of 5 × 2000 iters per shape.
4. Same-process A/B, paired per-shape (not full-sweep × 2 — GPU
   cools between sweeps).
5. Correctness check (`torch.allclose`) BEFORE timing.

Reference: `scripts/bench_tq4_matvec.py` +
`scripts/test_tq4_matvec_v2_correctness.py`. Without this
discipline, R53.29's v2 shipped at -2.8 / -11.3 / -6.3% across
same-code runs — direction right, magnitude unknowable. With
discipline, v2 stabilized to -5 to -10% aggregate.

Rule: never declare a Triton kernel win on one run. Median ≥ 3;
A/B deltas must agree in SIGN across runs even if magnitude
varies. If sign flips, it's noise.

## Daemon state invariants (2026-04-22 facade-run receipt)

The Gemma daemon (`bin/gemma_daemon.py` + `bin/gemma-run`) preserves
`m` and `tok` across script runs but each script also mutates
hidden state on `m` that MUST be reset between unrelated runs:

1. **`m.verification_hooks`** is a list owned by
   `calm/llm_computer/gemma_substrate.py`. Every script that installs
   a `CardSlot` or `VerificationHook` appends to this list. `RESET_GLOBALS`
   does NOT clear it. Run 2 of r60a (ICD-10 facade) after r22d (MQAR
   card) produced pure-digit garbage output because MQAR's digit-bias
   hook was still attached, biasing every ICD-10 probe's tokens.

2. **`m.reserved_channels`** + `m.layers[idx].card_slots` — same
   failure mode; lingering channel reservations affect downstream
   scripts.

3. **Module cache**: the daemon does `exec(compile(code, line, "exec"),
   ns)` each run, but `sys.modules` is shared. Editing `recursion.py`
   or a facade module and re-running picks up the NEW script text,
   but the IMPORT of the module still returns the cached version.
   `--reset` doesn't help; `--quit` + `--start` does.

**Rules:**

- Every facade A/B script MUST call `clear_card_state()` at startup:
  ```python
  def clear_card_state():
      for lyr in m.layers:
          if hasattr(lyr, "card_slots"):
              lyr.card_slots = []
      m.verification_hooks = []
      m.reserved_channels = []

  clear_card_state()
  ```
  Pattern shipped in `scripts/r60a_icd10_failure_gate.py`,
  `scripts/r70a_planner_mixed.py`, `scripts/m1a_four_new_facades.py`,
  `scripts/r80a_recursion_demo.py`.

- After editing a facade module source file, run `bin/gemma-run --quit`
  then `bin/gemma-run --start` (not just `--reset`). Re-loading takes
  ~2-3 min but is the only way to pick up module-level code changes.
  `importlib.reload` inside the script helps for THAT run but doesn't
  affect subsequent runs from the same daemon.

- If a script's output shows digit-bias artifacts on prompts it
  shouldn't bias ("hello" → "0000...0"), lingering hook is the first
  hypothesis before suspecting the new code.

## Commit discipline — git log as progress changelog

- **Commit completed work before starting the next round.** Default
  rule. Once a round's measurements pass and the change is shippable,
  commit BEFORE starting the next hypothesis. Uncommitted measured
  work is a liability — a crash, an accidental `git stash`, a `reset
  --hard` to recover from a bad swing, all silently destroy hours.
  "I'll commit at the end of the session" is how R52/R53 left ~30
  files untracked and forced a paranoid handoff section. The marginal
  cost of `git add && git commit` is seconds; the cost of losing a
  passed round is hours. Commit, then iterate.
- **One round per commit.** Don't stack unrelated optimizations into
  one commit. When you bisect later, you want each commit to own one
  clear change.
- **Measurement in the message.** Every perf / correctness commit has
  a short before/after table in the body. Template:
  ```
  <subsystem>: <one-line what>

  <3-5 lines explaining why and how — the hypothesis>

  Metric (context + hardware):

    metric                before   after
    -----------           ------   ------
    <name>                <N>      <N>
    <name>                <N>      <N>

  Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
  ```
- **Checkpoint before risky swings.** Before a re-quantize, a struct
  layout change, a kernel rewrite, a training run — commit the current
  working state so `git reset --hard HEAD` is your rollback path.
- **`git log --oneline` is a readable perf history.** When someone
  (you, future-you, another session) asks "what changed and why did
  it get faster?" the answer is in the log.

## Long-running training supervision

**Current rule (eager `workflow.md`):** foreground training in a dedicated
shell; log to file; **no detach** (`setsid`, `nohup`, `disown`,
`run_in_background`, trailing `&` forbidden). Arm `bin/watch-wrap` Monitor
with failure + progress + stop-on filters. Canonical copy:
`workflow_part_2.md` §"Long-running training supervision (current — foreground,
no detach)".

**Historical receipts** (session-25 plateau kills, monitor-ship-at-right-
checkpoint examples) retained below for forensics only — do not revive
detached launch patterns.

Session-25 HRM autoregressive retraining killed a 1000-epoch run at epoch
200 when loss=0.04 but val_acc=51% — classic 900-sample / 108K-param
memorization gap; restarted with 2× data rather than waiting out 800 more
epochs.

A 500-epoch HRM run hit 99.7% per-token at epoch 100 → killed early because
the structurally-relevant gate (full-expression via verified mode) was already
saturated. Monitor lets you ship at the right checkpoint, not the scheduled-end
checkpoint.
