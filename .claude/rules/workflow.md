# Default Workflow — Hypothesis, Test, Iterate

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

## Commit discipline — git log as progress changelog

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
  iteration must still pass the cheapest correctness test. For this
  project the canonical check is "17 × 23 = 391" — a math prompt that
  a working tq4 model trivially passes and a broken one garbles. Run
  it every round.
- **Trusting a single number.** One run with r=1 can be off by 10%.
  For llama-bench, -r 3 to -r 5 is the minimum. For chat API calls,
  run the same prompt twice.
- **Editing before benching the baseline.** You can't compute a delta
  from a hypothetical starting point. Always re-bench after a pull /
  merge / environment change before claiming the next optimization
  moved anything.

## When this workflow doesn't apply

- **UI / frontend / design work.** Subjective judgment calls that
  don't have binary outcomes. Defer to the user.
- **Exploratory research with no defined target.** e.g. "what can this
  model do?" — at that stage, demos and qualitative reads are the
  measurement. Still write down what you're looking for.
- **Pure discovery reading.** Reading code to understand the system
  doesn't need a metric. Once you start *changing* it, it does.

For everything else: hypothesis, test, iterate, commit, repeat.
