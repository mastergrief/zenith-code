**IMPORTANT**: Assume nothing. Hypothesis, Build, Test, Commit & Iterate. First Principles thinking. Do not discount anything until it's built and tested!

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
| Substrate card install | card.forward standalone == card.forward inside Gemma (hooked) | Gemma logits diff vs no-install baseline > noise |
| Triton kernel | bit-equivalent to PyTorch path (max abs diff < 1e-5) | end-to-end tok/s on a 30+ token decode |

If you only have one path, say so out loud and accept the reduced
confidence — but keep looking for the second.

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

Current defaults: `MAX_TOKENS_CEILING = 16384` +
`AdaptiveBudget` (tier-picked per-prompt, trivial 2K → deep 32K
clamped to 16K) in `scripts/r53_eval_complex.py` +
`scripts/r53_21_import_inject.py`. All R53 wrapper scripts bumped
to 4096-16384.

Rule: when a Gemma failure is "no output / NoCode", check
`max_tokens` ≥ prompt + `<think>` + expected output BEFORE any
deeper diagnosis.

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

Training runs that emit sparse eval lines (every N epochs, every M steps)
are the perfect use case for **Monitor + filtered tail**. The raw log
stays on disk; only the eval/error lines arrive in context as
notifications, so you can apply the plateau-detection loop in real time
instead of waiting out a full run.

Pattern:

```bash
# Kick off detached so CC/WSL crashes don't kill it
setsid env PYTHONPATH=. python3 -u -m calm.hrm.train --... \
  < /dev/null > /tmp/train.log 2>&1 &
disown -a
```

```
Monitor(command="tail -f /tmp/train.log | grep --line-buffered -E 'epoch|Error|Traceback'")
```

- `-u` on python to avoid stdout buffering (eval lines arrive immediately).
- `grep --line-buffered` is mandatory — without it pipe buffering holds
  events for minutes.
- Redirect stdin from `/dev/null` to avoid WSL interop stdin
  consumption (same class of bug as the `bin/zenith` tasklist.exe fix).

Each monitor notification is a plateau-detection checkpoint (see the
loop above). If training loss crashes to near-zero while val/eval
accuracy stays flat for 2–3 consecutive eval intervals, **kill and
intervene** on one hypothesis (data, capacity, LR, regularization).
Don't ride out the remaining epochs. Session-25 HRM autoregressive
retraining killed a 1000-epoch run at epoch 200 when loss=0.04 but
val_acc=51% — classic 900-sample / 108K-param memorization gap;
restarted with 2× data rather than waiting out 800 more epochs.

The same monitor pattern caught a more subtle case later in session 25:
a 500-epoch HRM run hit 99.7% per-token at epoch 100 → killed early
because the structurally-relevant gate (full-expression via verified
mode) was already saturated. Monitor lets you ship at the right
checkpoint, not the scheduled-end checkpoint.

**Harness shorthand**: `Bash(run_in_background=True)` launches the
training; `Monitor` with `tail -f /tmp/train.log | grep --line-buffered
-E "epoch.*done|eval:|DECISION:|Traceback|Error|Killed"` streams
milestones into the conversation. Don't poll — wait for monitor
notifications and continue other work in the meantime. This session
used the pattern for SubstrateLM, hybrid v1, Round 3, Round 4, and
v2 GPU runs without burning context on raw training logs.

The filter MUST cover failure signatures too, not just success — a
monitor that only matches "epoch done" stays silent through a crashloop
or OOM, making silence indistinguishable from "still running." Include
`Traceback|Error|Killed|OOM|FAILED|assert` in the alternation.

## Training-specific discipline (see `training.md`)

Two training disciplines live in `.claude/rules/training.md` to keep
them with other training content:

- **Safer-config for noisy-grad training** — `training.md` §"Safer-
  config for noisy-grad training". Canonical receipt: R52.2 divergence
  (batch=1 + lr=1e-3 + grad_clip=1.0) vs recovered config (batch=4 +
  lr=3e-4 + grad_clip=0.1 + warmup=200). Diagnose Adam momentum
  poisoning via EMA climb over 20+ steps.

- **GPU vs CPU decision rule for substrate training** — `training.md`
  §"GPU vs CPU for substrate training". CPU < 500K params / 128 tok /
  pure-Euclidean; GPU above. Observed 6× speedup (not 10-20×) because
  D5 launches kernels serially.

## Informative null results

A null result that diagnoses the failure mode IS shippable. Commit it
with the same before/after table discipline as a win — the diagnostic
value is the contribution. Examples from this session:

- **Round 3 fast-weights d_model scaling** (n=10 ceiling): +1.8pp on
  4× capacity vs n=5's +32.8pp on same scaling. Concluded capacity
  isn't the n=10 bottleneck — interference is structural.
  Commit `b46aff3` scopes Round 4 mechanisms (delta rule, gate).
- **Round 4 delta rule + write gate**: clean null across all 4 variants
  at n=10 (10.5-12.2%). Ruled out both hypothesized fixes; remaining
  candidates are normalized Schlag form, per-head FW, SRWM.
  Commit `762ab07` gives Round 5 design inputs directly.
- **Hybrid v1 HRM mode** (0% correct, 78% parseable): diagnosed as
  token-count imbalance + template variety deficit. Fed directly into
  v2's curriculum fixes (oversample, mode-loss weight, pool multi20).
  Commit `1384373`.

**Null-result commit pattern**:
```
<subsystem>: <mechanism> clean null (Round N)

Hypothesis: <one line>

Result: <metric table showing the null>

Diagnosis: <what we learned about the failure mode>

Next round: <specific scope change justified by the null>

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```

Rounds 3 and 4 commit bodies are the canonical templates.

## Sweet-spot search for tiny models

When the goal is **maximum capability per parameter**, search downward
not upward:

1. Start with the smallest config that could plausibly work
   (`hidden=16-32`, single layer).
2. Train. If the structurally-relevant gate fails (not per-token —
   the user-facing gate), scale the smallest knob first: more data,
   then more epochs, then capacity.
3. Only scale capacity (`hidden`, layers, heads) when data + epochs
   plateau below gate.
4. **The point isn't smallest possible — it's smallest sufficient.**
   You're looking for the size where structure-emission becomes
   reliable, not the size where everything is solved.

Session-25 example: tried `h=64 → h=128 → h=64+scratchpad → h=64+placeval`
first (each iteration kept memorization pressure on values), got stuck
at 37-57% full-expression. Then switched to **`h=32 + structure-only
loss`** (offload values to the LLM-Computer interpreter): 48K params,
145s training, 96.7% full-expression. The unlock was redefining what
the model is asked to learn, not making it bigger.

Corollary: **Per-token accuracy is misleading on trace-shaped targets.**
Trace targets contain a lot of trivially-predictable tokens (operators,
parens, copies of input). Per-token can be 94% while full-expression
sits at 43%. Always measure the user-facing gate; only use per-token
to spot regressions during training.

## Tool priority for diagnostics

In any iteration where the simple "edit + rerun" doesn't explain what
happened, escalate through these tools in order of cost. Stop at the
first one that tells you what you need.

1. **The existing fast bench / test.** Your raw measurement from the
   loop. Always try to solve the puzzle at this level first.
2. **Pre-build static inspection** — `nvcc -Xptxas=-v` for CUDA
   (register count, shmem, spills), `python -c "import ..."` for Python
   imports, `cargo check` for Rust. Free, no runtime required.
3. **Print / counter instrumentation.** Add a counter, atomic, log
   line, timing statement. Hand-rolled instrumentation in the code you
   own is usually faster and more reliable than external profilers on
   this WSL setup.
4. **"Remove it and see."** Wrap the suspected code in `if (false)`,
   comment out the call, swap in a stub — whatever temporarily
   eliminates the component. If the metric barely moves, the
   component is not the bottleneck. Worth doing early because it's
   cheap and proves a negative cleanly.
5. **External profilers (`ncu`, `nsys`, `perf`, etc.)** — nice when
   they work but **known broken on this WSL setup**: `ncu` fails on
   `ERR_NVGPUCTRPERM` (driver flag is kernel-load-time on WSL, can't
   fix at runtime), `nsys` ships without the `.qdstrm` importer in the
   Ubuntu package. **Don't block optimization on profilers.** Fall
   back to step 3 instrumentation.

## Probing-specific methodology gates

Three gates from session 33's R47-R50 arc that apply when probing an
LLM's activations. Full spec + examples:
`.claude/rules/probing_methodology.md`.

- **Prompt-format gate** (R47.2): verify baseline argmax-correct
  rate > 50% before interpreting ablation, else you're measuring
  shortcut circuits.
- **Task-rank vs PCA-rank** (R49.2, R50.5): variance rank is a lower
  bound on task-rank; validate with a projection-or-ablation test
  that measures accuracy preservation.
- **Superposition blinds ablation** (R48.1 → R50.3): "diffuse at
  neurons + strong at layer" = superposition suspect. Reach for
  TopK SAE, not L1.

## Pitfalls to avoid

- **Bundling.** Two changes in one build → the bench can't attribute
  the delta. Always separate.
- **"It didn't crash" is not a passing test.** Running the binary and
  seeing no error is one signal. A correctness assertion is another.
  Speedup is a third. Don't conflate them.
- **Letting a measurement drift justify itself.** If you benched at
  24.1 this morning and 24.3 now, that's noise, not improvement. Run
  more repetitions, not more optimism.
- **Skipping the correctness check when chasing perf.** Every perf
  iteration must still pass the cheapest correctness test. Two canonical
  checks:
  - **Math**: "17 × 23 = 391" via chat API — a working tq4 model
    trivially passes, a broken one garbles.
  - **CALM multi-domain**: `run_auto("What is sin(30°)? Is 4181 a
    Fibonacci number? Capital of France?")` — tests precompute, NL
    patterns, and knowledge backends in one prompt. All three should
    be precomputed and correct with 0 corrections.
- **Trusting a single number.** One run with r=1 can be off by 10%.
  For llama-bench, -r 3 to -r 5 is the minimum. For chat API calls,
  run the same prompt twice.
- **Editing before benching the baseline.** You can't compute a delta
  from a hypothetical starting point. Always re-bench after a pull /
  merge / environment change before claiming the next optimization
  moved anything.

## Feedback-loop validation pattern

For any system with a `learn` phase and an `apply` phase —
AutoLearner, ModuleLearner, pattern DBs, self-tuning components —
write three layers of tests in this order:

1. **Unit tests that prove the cycle closes.** Feed in a correction;
   assert the pattern is recorded; assert the next matching input
   produces the right `apply`-phase output. If this fails, the loop
   is open. See `calm/tests/test_auto_learn_loop.py::test_loop_closes_*`.
2. **Quantitative effectiveness harness.** Script that runs N
   corrections then measures hit rate on held-out inputs. Report the
   delta round-over-round — does adding more corrections monotonically
   improve coverage? `calm/closed_loop_eval.py` is the canonical
   example (90% → 100% over 3 rounds, 10× pattern compression via
   generalization).
3. **End-to-end integration test with mocked dependencies.** Mock the
   expensive component (LLM inference, external API) and exercise the
   full pipeline through the real facade. Assert the learned patterns
   actually get injected upstream at the next invocation. See
   `calm/tests/test_auto_calm_integration.py` — mocks `_generate`,
   proves `AutoCalmEngine` injects `Verified facts: 347 * 289 =
   100283` into the system prompt on round 2 after learning from
   round 1.

Shape gates on the `apply` phase matter too. Vector 1 phase 2 found a
real defect: function patterns were firing on every numeric prompt
(factorial(5) instantiated for `what is 5 * 7?`), flooding the system
prompt with irrelevant precomputes. Fix: require a shape keyword in
the prompt (function name, operator, or NL alias like "plus"/"times").
Add the shape-gate test alongside the loop-closes test.

Visibility matters once both halves work. `scripts/learning_dashboard.py`
prints the current state of both loops — pattern counts, hit counters,
recurring issues — in one command. Build the equivalent for any new
loop you add.

## CALM iteration pattern

The hypothesis-test-iterate loop applied to CALM intelligence scaling:

1. **Hypothesis**: "Adding backend X will make Gemma answer domain Y
   correctly via precompute" or "Fixing module Z's triggers will catch
   failure mode W."
2. **Raw measurement**: function count, NL pattern count, module count,
   `pytest calm/tests/` passing.
3. **Build**: write the backend/module/pattern. Minimal — one domain,
   one module, one fix per iteration.
4. **User-facing measurement**: Gemma test via `run_auto()` or Engine
   V2. Check: did precompute fire? Did the module catch the issue? Did
   quality score change? Did self-heal trigger?
5. **Quality gap test** (for cognitive work): run the router on a
   deliberately flawed response AND a good response. Bad should score
   < 75%, good should score > 90%. If the gap is < 15%, the modules
   aren't discriminating — sharpen triggers or add patterns.
6. **Commit with before/after table** including function count, pattern
   count, module count, quality scores.

The CALM loop is fast because backends are pure Python (no GPU, no
training, no inference). A full cycle — write backend, test functions,
check registration, run Gemma — takes 3-5 minutes. Cognitive module
changes take slightly longer because the Gemma test needs inference
time (~30-60s per prompt).

### Auto-upgrade extension (session 30)

CALM corrections now feed the substrate's persistent knowledge layer:
after the CALM loop catches an error → `AutoUpgradeEngine.commit()` →
corrections compile into substrate weights → save `.pt` → next session
the error is permanently fixed. This closes the loop from "CALM catches
errors" to "errors never recur" without retraining. See
`.claude/rules/calm.md` "Auto-Upgrade Loop" and
`calm/llm_computer/auto_upgrade.py`.

## Substrate install workflow (see `Substrate.md`)

Checklist for installing a card into prod Gemma lives in
`.claude/rules/Substrate.md` §"Install Workflow (checklist)".
6-step Allocate → Convert → Install → Verify → Register → Commit.

## When this workflow doesn't apply

- **UI / frontend / design work.** Subjective judgment calls that
  don't have binary outcomes. Defer to the user.
- **Exploratory research with no defined target.** e.g. "what can this
  model do?" — at that stage, demos and qualitative reads are the
  measurement. Still write down what you're looking for.
- **Pure discovery reading.** Reading code to understand the system
  doesn't need a metric. Once you start *changing* it, it does.

**IMPORTANT**: Assume nothing. Hypothesis, Build, Test, Commit & Iterate. First Principles thinking. Do not discount anything until it's built and tested!
