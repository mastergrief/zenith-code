# R19 — D5 refinement loop null on MQAR (2026-04-21)

Hypothesis from ARC Prize blog (arcprize.org/blog/hrm-analysis):
HRM's "outer refinement loop" drives +13pp over single-pass
inference. Test whether D5 recurrence (iterate layer stack
`n_iterations` times per forward pass, weight-shared) gives a
similar data-efficiency lift on MQAR.

**Null on our task.** D5 at n_iters=2 doesn't break the R13-c
plateau at N=15 / 2K/N — similar 12-22% band, slight regression
vs plain PT.

## Config

  script:  experiment_r10_mqar.py --task mqar
           --per-N-train 2000 --n-values 15 --epochs 20
           --max-len 128 --chunkwise --chunk-size 32
           --n-iterations 2
  model:   185K params, single-head delta, n_iterations=2

## Result

  phase            best     final
  -------------    -----    -----
  Plain PT (n_iters unaffected) | 27%  | 22%
  PT+Δ (n_iters=2)             | 21%  | 16%

Gap at best-epoch: **-6pp (PT+Δ worse than plain PT).**

Compare against:

  Round        config                        N=15 best
  -----------  ---------------------------   ---------
  R13-c        H=1 n_iters=1 @ 2K/N          **19%**  ← baseline null
  R18          H=4 n_iters=1 @ 2K/N          **21%**  ← null
  R19          H=1 n_iters=2 @ 2K/N          **21%**  ← null
  R13-d        H=1 n_iters=1 @ 5K/N          **99%**  ← data solves it

All three architectural interventions at 2K/N converge to ~20%.
Data scaling to 5K/N cleanly solves. **Data is the primary knob,
architecture changes don't substitute.**

## Runtime check

n_iters=2 took 165s for PT+Δ phase vs single-iter ~80-90s —
confirms the loop actually ran twice. Not an implementation bug.

## Why ARC's finding didn't transfer

ARC Prize's +13pp was measured on ARC-AGI grid-reasoning tasks
where refinement operates on a spatial layout (model writes a
candidate grid, then refines it). MQAR is:
- Single-token output (not iteratively refineable)
- No spatial structure
- Retrieval task, not generation task

Refinement-loop benefit appears to require tasks where the model
emits a structured output it can iteratively improve. MQAR
retrieves a single value; there's nothing to refine *iteratively*.

Refinement MIGHT help at:
- Scratchpad-style generation (R16's null task — multi-step
  arithmetic where the model could "correct" intermediate
  values on a second pass)
- Larger compositional tasks (but requires building the task first)
- HRM's original grid-based reasoning domains

Not a blanket data-efficiency lever for retrieval.

## What this doesn't rule out

- D5 at higher n_iters (3, 4, 5) — diminishing returns expected
  per ARC's own curve (0→1 gave +13pp, 1→2 gave less)
- D5 on scratchpad (R16 retry) — genuinely testable, deferred
  since R16 had a deeper issue (arithmetic-memorization capacity)
- D5 on mixed-task training (R20) — where some tasks benefit
  and others don't, weighted transfer could net positive

## Revert decision

`n_iterations` config stays in tree behind default=1 (backward
compat preserved). Script flag `--n-iterations N` exposed for
future experiments but not default.

Production default remains: **single-head chunkwise DeltaNet at
n_iterations=1**, same as R14-b / R17 shipped.

## Architectural-lever audit (R18 + R19)

Two architectural interventions tested at R13-c's 2K/N × N=15
null baseline (19%):

  R18: multi-head (H=4) → 21%   (no lift, -6pp below plain PT)
  R19: D5 refinement (n_iters=2) → 21% (no lift, -6pp below plain PT)
  R13-d: data scaling to 5K/N → 99%  (clean solve)

**Pattern**: at this substrate scale, architecture changes don't
substitute for data. The R14-b scaling curve's "+5 on N needs 2×
data" is the binding constraint.

Consistent with ARC Prize's finding that HRM's architecture
contributed ~5pp vs plain transformer. Architecture isn't the big
lever; data (+ targeted mechanism like DeltaNet's Householder
recurrence) is.

## Raw log

- `/tmp/r19_n15_2k_iter2.final.log`

## Next

With R18 and R19 both null, items #3 (multi-head) and #4 (D5
training-time refinement) on the improvement roadmap are
deprioritized. **Next move is #5 (Gemma CardSlot install)** —
the actual product step. No more architectural ablations until
a commercially-relevant task exposes a genuine bottleneck.
