**IMPORTANT**: Assume nothing. Hypothesis, Build, Test, Commit & Iterate. First Principles thinking. Do not discount anything until it's built and tested!

# Workflow — hypothesis, test, iterate

**This is the default working loop for all work in this project.**
Apply it to CUDA kernels, Python harness changes, training scripts,
quantize runs, config edits — anything that has a measurable outcome.

> Historical receipts (session-16 perf anecdote, canonical adapter-
> robustness / MAX_TOKENS / GPU-bench / null-commit case studies,
> session-25 sweet-spot journey): see
> `MEMORY/atlas/workflow_part_1.md` + `MEMORY/atlas/workflow_part_2.md`.

## Core principle — it works or it doesn't

**No vibes.** Every claim of "done", "working", "better", "fixed"
must be backed by a measurement that came *after* the change. If
you can't point to a number, a passing test, a correct output, or
a diff on an artifact — you don't know.

- "Looks right" — not done.
- "Should be faster" — not better.
- "I think it's working now" — not working.
- "Probably fine" — not fine.

The only exit from an iteration is a measurement that says "yes, it
moved" or "no, it didn't". Intermediate states (half-applied edits,
unrun tests, partial builds) aren't progress — they're liabilities.

If there's no way to measure it (pure UI/frontend judgment, brand
tone), say so explicitly and ask the user to make the call.

## The loop

1. **State the hypothesis.** One sentence: "X is the bottleneck / bug
   / missing piece because Y. Changing it to Z should move metric M
   by N%." Write it down.
2. **Pick the measurement first, not after.** If you can't specify
   the number that would prove the hypothesis, sharpen the hypothesis.
3. **Minimal edit.** Smallest change that tests the hypothesis. Don't
   bundle unrelated improvements — they poison the measurement.
4. **Build / run / test.**
5. **Measure.** Run twice if noisy.
6. **Binary decision.** Real movement (> noise, matches prediction)
   → ship. Didn't move or wrong direction → revert. No "maybe".
7. **Ruled-out log.** If you reverted, write one line saying what
   you tried + the delta. Future-you won't retry the known-bad path.
8. **Next hypothesis.**

Target: **< 5 minutes per round.** If longer, the measurement is too
heavy — find a lighter proxy.

## Always check two things

Run a **raw-path measurement** AND a **user-facing measurement** on
the same change. The single most important discipline here.

- Win on raw path that doesn't move user-facing → fixed wrong thing.
- User-only "win" is usually noise.
- Both move → real.

| Work type | Raw path | User-facing path |
|---|---|---|
| CUDA kernel opt | `llama-bench -n 64 -p 0 -r 3` | chat completion w/ fixed prompt |
| Python harness fix | unit test / pytest | `printf "prompt\n/exit\n" \| zenith` |
| Config / prompt | output on N fixed prompts | real conversation turn |
| Training data filter | schema validation + dedup count | loss curve on a few hundred steps |
| New CALM backend | function count + `pytest calm/tests/` | Gemma `run_auto()` — does precompute fire? |
| Cognitive module | router on flawed response (quality + issue count) | Engine V2 Gemma test — quality gap bad vs good |
| Substrate card install | card.forward on REAL adapter-extracted inputs | Gemma+card A/B vs baseline |
| Triton kernel | bit-equiv to PyTorch (max abs diff < 1e-5) | end-to-end tok/s on 30+ tok decode |

If you only have one path, say so out loud and accept reduced confidence.

### Adapter-robustness rule

If a card install shows low effective precision on live inputs, run
the card STANDALONE on the adapter's extracted strings BEFORE
hypothesizing calibration, distribution-shift, or architectural gaps.
30-second standalone diagnostic exposes adapter regex bugs that would
otherwise eat multiple rounds of threshold-tuning.

Rule: for install work with an adapter or parser, the two-measurement
pair is **(raw on REAL adapter outputs) + (user-facing A/B)**, not
(raw on synthetic inputs) + (user-facing). Synthetic sanity cases
skip the adapter entirely, so adapter bugs are invisible.

## Plateau detection

**3 iterations < 2% each → it's a bug, not a tuning problem.** Stop
micro-tuning, find the one wrong line.

Symptoms you're in a bug-not-tuning situation:
- Ratio to a known-good reference is suspiciously large (>2× perf,
  off by suspect constants for correctness).
- Occupancy / resource budgets say there's headroom but you're not
  using it.
- Compute/bandwidth math says it should be N× faster than you see.

## Empirical pace

Project's measured pace on this stack is minutes-to-hours, not the
weeks-to-months inherited from mechinterp literature. If a step looks
like days, methodology is wrong. Revisit. Full detail:
`probing_methodology.md` §"Empirical timeline".

## MAX_TOKENS budget discipline

Before diagnosing logic / substrate / sandbox / import failures,
verify output budget isn't clipping. Gemma 4 E4B trains at 131K ctx.
Eval `max_tokens` defaults should be ≥ 4K, not ≤ 400.

**Centralized**: `calm/llm_computer/eval_defaults.py` exports
`EVAL_CTX_SIZE=32768`, `EVAL_MAX_TOKENS=16384`, `ITERATION_N=5`,
`FINAL_N=20`. Every eval script imports from here — changing the
numbers changes every eval consistently.

**ITERATION_N / FINAL_N pattern**: every eval with a configurable
problem count defaults to `ITERATION_N` for the fast-iteration loop.
Bump to `FINAL_N` only for the round that goes into a commit receipt.
Pattern: `MBPP_N = FINAL_N if os.environ.get("MBPP_FINAL") == "1" else ITERATION_N`.
Never iterate at FINAL_N — violates the <5 min loop target.

Rule: when a Gemma failure is "no output / NoCode", check
`max_tokens` ≥ prompt + `<think>` + expected output BEFORE any
deeper diagnosis. When adding a new eval script, import from
`eval_defaults` rather than picking a number locally.

## GPU bench discipline

Triton kernel bench variance on the 4070 Laptop is 20-30% run-to-run
without stabilization:

1. `heavy_warmup(3.0s)` — dense fp16 matmul to steady-state clock.
2. `torch.cuda.Event(enable_timing=True)` — GPU-side timestamps.
3. Median of 5 × 2000 iters per shape.
4. Same-process A/B, paired per-shape (not full-sweep × 2).
5. Correctness check (`torch.allclose`) BEFORE timing.

Reference: `scripts/bench_tq4_matvec.py` +
`scripts/test_tq4_matvec_v2_correctness.py`.

Rule: never declare a Triton kernel win on one run. Median ≥ 3;
A/B deltas must agree in SIGN across runs even if magnitude varies.
Sign flips = noise.

## Daemon state invariants

The Gemma daemon (`bin/gemma_daemon.py` + `bin/gemma-run`) preserves
`m` and `tok` across script runs but mutates hidden state that MUST
be reset:

1. `m.verification_hooks` is a list. Every script installing a
   `CardSlot` or `VerificationHook` appends. `RESET_GLOBALS` does
   NOT clear it.
2. `m.reserved_channels` + `m.layers[idx].card_slots` — same mode.
3. Module cache: `sys.modules` is shared. Editing a facade module
   and re-running picks up NEW script text but IMPORT returns cached.
   `--reset` doesn't help; `--quit` + `--start` does.

**Mandatory pattern**: every facade A/B script calls `clear_card_state()`
at startup:
```python
def clear_card_state():
    for lyr in m.layers:
        if hasattr(lyr, "card_slots"):
            lyr.card_slots = []
    m.verification_hooks = []
    m.reserved_channels = []

clear_card_state()
```

**Diagnostic rule**: if script output shows digit-bias artifacts on
prompts it shouldn't bias ("hello" → "0000..."), lingering hook is
the first hypothesis before suspecting new code.

## Commit discipline — git log as progress changelog

- **Commit completed work BEFORE starting the next round.** Default
  rule. Uncommitted measured work is a liability — a crash, an
  accidental `git stash`, a `reset --hard` silently destroys hours.
- **One round per commit.** Don't stack unrelated optimizations.
- **Measurement in the message.** Every perf / correctness commit
  has a short before/after table in the body:
  ```
  <subsystem>: <one-line what>

  <3-5 lines explaining why and how — the hypothesis>

  Metric (context + hardware):
    metric                before   after
    -----------           ------   ------
    <name>                <N>      <N>

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  ```
- **Checkpoint before risky swings.** Re-quantize, struct layout,
  kernel rewrite, training run — commit current state so
  `git reset --hard HEAD` is your rollback.
- **`git log --oneline` is a readable perf history.**

## Informative null results

A null that diagnoses the failure mode IS shippable. Commit with the
same before/after discipline — the diagnostic value is the contribution.

Pattern:
```
<subsystem>: <mechanism> clean null (Round N)

Hypothesis: <one line>
Result: <metric table showing the null>
Diagnosis: <what we learned about the failure mode>
Next round: <specific scope change justified by the null>
```

## Long-running training supervision

Detached + Monitor pattern:
```bash
setsid env PYTHONPATH=. python3 -u -m calm.hrm.train ... \
  < /dev/null > /tmp/train.log 2>&1 &
disown -a
```

```
Monitor(command="tail -f /tmp/train.log | grep --line-buffered -E 'epoch.*done|eval:|DECISION:|Traceback|Error|Killed|OOM|FAILED|assert'")
```

- `-u` on python: unbuffered stdout.
- `grep --line-buffered`: mandatory or events stall in pipe.
- Redirect stdin from `/dev/null` to avoid WSL interop stdin consumption.
- **Filter MUST cover failure signatures** not just success — silent
  monitor through OOM is indistinguishable from healthy. Include
  `Traceback|Error|Killed|OOM|FAILED|assert`.

Each notification = plateau-detection checkpoint. Loss-crashes-while-
val-flat for 2-3 evals → kill, change one hypothesis, restart.

## Sweet-spot search for tiny models

When goal is **maximum capability per parameter**, search downward:

1. Start with smallest config that could plausibly work (`hidden=16-32`).
2. Train. If the user-facing gate fails, scale smallest knob first:
   more data → more epochs → capacity.
3. Only scale capacity when data + epochs plateau below gate.
4. **The point isn't smallest possible — it's smallest sufficient.**

Corollary: **Per-token accuracy is misleading on trace-shaped targets.**
Trace targets contain many trivially-predictable tokens (operators,
parens, copies of input). Per-token can be 94% while full-expression
sits at 43%. Always measure the user-facing gate.

## Tool priority for diagnostics

Escalate in order of cost. Stop at the first one that tells you what
you need.

1. **Existing fast bench / test.** Always try this first.
2. **Pre-build static inspection** — `nvcc -Xptxas=-v` for CUDA,
   `python -c "import ..."` for Python, `cargo check` for Rust. Free.
3. **Print / counter instrumentation.** Hand-rolled instrumentation
   is usually faster than external profilers on this WSL setup.
4. **"Remove it and see."** Wrap in `if (false)`, comment out, swap
   stub. If metric barely moves, component isn't the bottleneck.
5. **External profilers (`ncu`, `nsys`, `perf`)** — known broken on
   this WSL setup. Don't block optimization on profilers.

## Probing-specific methodology gates

Three gates from mechinterp work. Full spec: `probing_methodology.md`.

- **Prompt-format gate**: verify baseline argmax-correct rate > 50%
  before interpreting ablation, else you're measuring shortcut circuits.
- **Task-rank vs PCA-rank**: variance rank is a lower bound on
  task-rank; validate with a projection-or-ablation test that
  measures accuracy preservation.
- **Superposition blinds ablation**: "diffuse at neurons + strong at
  layer" = superposition suspect. Reach for TopK SAE, not L1.

## Pitfalls

- **Bundling.** Two changes in one build → can't attribute the delta.
- **"Didn't crash" ≠ passing test.** No error is one signal.
  Correctness assertion + perf are separate signals.
- **Drift justifying itself.** 24.1 → 24.3 is noise, not improvement.
- **Skipping correctness when chasing perf.** Canonical math smoke:
  `17×23=391` via chat API. CALM multi-domain smoke:
  `run_auto("What is sin(30°)? Is 4181 a Fibonacci number? Capital of France?")`.
- **Trusting one number.** llama-bench `-r 3` minimum; chat prompts twice.
- **Editing before benching baseline.** Re-bench after pull / merge /
  env change before claiming the next optimization moved anything.

## Feedback-loop validation pattern

For any system with a `learn` phase and an `apply` phase — write
three layers of tests in this order:

1. **Unit tests that prove the cycle closes.** Feed in correction;
   assert pattern recorded; assert next matching input produces the
   right apply-phase output. If this fails, the loop is open.
2. **Quantitative effectiveness harness.** Script that runs N
   corrections then measures hit rate on held-out inputs. Report
   delta round-over-round.
3. **End-to-end integration test with mocked dependencies.** Mock
   the expensive component; assert learned patterns get injected
   upstream at the next invocation.

**Shape gates on apply phase matter.** Require a shape keyword before
instantiating a pattern (function name, operator, or NL alias) — else
patterns fire on every numeric prompt, flooding the system prompt
with irrelevant precomputes.

**Visibility**: `scripts/learning_dashboard.py` prints the current
state of both loops in one command. Build the equivalent for any new
loop you add.

## CALM iteration pattern

Hypothesis-test-iterate applied to CALM intelligence scaling:

1. **Hypothesis**: "Adding backend X will make Gemma answer domain Y"
   or "Fixing module Z's triggers catches failure mode W."
2. **Raw measurement**: function count, NL pattern count, module
   count, `pytest calm/tests/` passing.
3. **Build**: minimal — one domain, one module, one fix per iteration.
4. **User-facing**: Gemma test via `run_auto()` or Engine V2. Did
   precompute fire? Did module catch the issue? Quality score change?
   Self-heal trigger?
5. **Quality gap test**: router on flawed response AND good response.
   Bad < 75%, good > 90%. If gap < 15%, modules aren't discriminating.
6. **Commit with before/after table** including counts + scores.

Full CALM cycle: 3-5 minutes (backends are pure Python, no GPU).
Cognitive module changes need inference time (~30-60s per prompt).

### Auto-upgrade extension

CALM corrections feed the substrate's persistent knowledge layer:
CALM catches error → `AutoUpgradeEngine.commit()` → corrections
compile into substrate weights → save `.pt` → next session the error
is permanently fixed. Closes the loop from "CALM catches errors" to
"errors never recur" without retraining. See `calm.md` §"Auto-Upgrade
Loop" and `calm/llm_computer/auto_upgrade.py`.

## Substrate install workflow

Checklist for installing a card into prod Gemma lives in `Substrate.md`
§"Install Workflow". 6-step: Allocate → Convert → Install → Verify →
Register → Commit.

## When this workflow doesn't apply

- **UI / frontend / design work.** Subjective judgment. Defer to user.
- **Exploratory research with no defined target.** Demos and qualitative
  reads are the measurement at that stage. Still write down what
  you're looking for.
- **Pure discovery reading.** Reading code to understand doesn't need
  a metric. Once you start *changing* it, it does.

**IMPORTANT**: Assume nothing. Hypothesis, Build, Test, Commit & Iterate. First Principles thinking. Do not discount anything until it's built and tested!
