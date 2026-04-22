# FC8/FC9 — DT code-skeleton training

Trained a Delta-Transducer (DT) on `(problem_description, skeleton)` pairs
from CodeExampleDB. Deployable checkpoint: `dt_code_skel_best.pt`.

## Hypothesis

Install a DT that emits Python function skeletons on the L30 CardSlot
pathway to bias Gemma toward code emission on MBPP-style prompts (fixes
the `format_fail` failure class measured in FC7).

## Iteration receipt

| version | target | best val_autoreg | plateau ep |
|---|---|---:|---:|
| v1 | `def radian_degree(deg):` (literal names) | **0.035** (4/115) | ep 12 |
| v3 | `def FN(deg):` (generic placeholder) | **0.148** (17/115) | ep 20 |

**+11.3pp lift (4×)** from target refinement alone — function NAMES
require concept→identifier synthesis which is beyond DT's copy-augmented
design; function ARG STRUCTURE is copy-friendly.

v1 killed on plateau detection at epoch 15 per `workflow.md` §"plateau
= bug, not tuning". Iterated to v3. Checkpoint committed.

## Model

| spec | value |
|---|---|
| architecture | `CopyAugmentedDeltaNet` (copy-gate + DeltaNet Householder fast-weight backbone) |
| params | 191,941 (191.9K) |
| d_model | 64 |
| n_heads | 32 (d_head=2 invariant) |
| n_layers | 4 |
| max_len | 256 chars |
| vocab | 81 (code-specific: math-PT vocab + `:`) |
| training | 30 epochs, batch=32, lr=1e-3, scheduled sampling tf 1.0→0.3 |
| chunkwise | yes |
| wall time | ~10 min on RTX 4070 |

## Data

| spec | value |
|---|---|
| total pairs | 1,158 (mbpp + humanevalplus + bigcodebench + generated) |
| train/val | 1043 / 115 |
| unique skeletons (v1 literal) | 1,158 (uniform — no duplicates by name) |
| unique skeletons (v3 FN-placeholder) | 367 |
| top-10 skeleton coverage | ~400/1158 ≈ 35% |
| most common | `def FN(n):` (182×), `def FN(text):` (40×), `def FN(s):` (38×) |

## Remaining gap to ≥80% target

DT learned `def FN(` template perfectly (implicit from sample decodes).
Error mode concentrated in:
  - Arg-count mismatches: training labels like `def FN(test_list):` gen
    `def FN(t_t):` or `def FN(arr, numstest_list):`
  - Compound/keyword arg names: DT drops them to single chars

Hypothesis why 0.148 is the current ceiling:
  - Copy-gate IS firing (sample outputs show arg-name char fragments
    copied from prompt context) but not reaching names accurately
  - 1158 pairs may be insufficient; HumanEval/BigCodeBench could double
    the data
  - 191K-param DT is consistent with MQAR N=15 ceiling; may need more
    capacity for 367 distinct skeletons

## Deliverables committed

- `calm/hrm/code_dt_data.py` — data pipeline (1158 pairs, 367 skeletons)
- `scripts/train_code_dt.py` — DT training loop
- `calm/llm_computer/dt_install.py` — CardSlot install scaffold (R22 pattern)
- `calm/hrm/checkpoints/dt_code_skel_best.pt` (780 KB)
- `calm/hrm/checkpoints/dt_code_skel_metrics.json`
- `.claude/rules/delta_rule.md` — DT vocabulary adoption note at top

## Deferred

1. **Gemma install + MBPP A/B** — with 0.148 autoreg, DT is likely too
   noisy to help extraction rate on MBPP. Either improve DT first or
   install as-is and measure to see if even noisy bias helps.
2. **Iteration FC10**: bigger DT (d_model=128), more data (include all
   HumanEvalPlus + generated), or different target (just arg-count)
3. **Commit-candidate comparison**: train plain-PT baseline on same
   target to confirm DT's advantage beyond what plain-PT would hit

## Terminology

**DT (delta-transducer)** adopted 2026-04-22 as canonical product name
for copy-augmented DeltaNet. Implementation class `CopyAugmentedDeltaNet`
retained. See `.claude/rules/delta_rule.md` vocabulary note.
