# HRM — Overview

The *what* and *why* of the Hierarchical Reasoning Model. Motivation,
architecture at a glance, results, brain correspondence, caveats. No
math — that lives in [`02_Hierarchical_Convergence.md`](02_Hierarchical_Convergence.md)
and [`03_Training_Procedure.md`](03_Training_Procedure.md).

Based on: Wang et al., "Hierarchical Reasoning Model," Sapient
Intelligence, 2025.

See also: [`../00_INDEX.md`](../00_INDEX.md) for the full RESEARCH/
index covering HRM and Percepta's in-model-execution work.

---

## 1. TL;DR

HRM is a 27-million-parameter brain-inspired recurrent model, trained
from scratch on **1,000 examples per task with no pretraining and no
CoT supervision**, that outperforms frontier CoT models on three hard
reasoning benchmarks:

| Benchmark | HRM (27M, scratch) | Claude 3.7 (8K ctx) | o3-mini-high |
|---|---:|---:|---:|
| ARC-AGI-1 | **40.3%** | 21.2% | 34.5% |
| Sudoku-Extreme | near-100% | 0% | 0% |
| Maze-Hard 30×30 | near-100% | 0% | 0% |

The underlying argument: standard Transformers are **fundamentally
shallow** (fixed depth → complexity class AC⁰ / TC⁰), so end-to-end
they cannot execute algorithms that need polynomial time. Chain-of-
Thought is a workaround — externalizing reasoning into tokens — but
it's brittle, data-hungry, and slow. HRM achieves **effective
computational depth** differently: two coupled recurrent modules at
different timescales execute a sequence of nested stable computations
inside a single forward pass.

Four training innovations make it practical:

1. **Hierarchical convergence** — avoids the premature-convergence
   failure mode of standard RNNs. See
   [`02`](02_Hierarchical_Convergence.md).
2. **One-step gradient approximation** (grounded in Deep Equilibrium
   Models + Neumann series) — replaces BPTT, drops memory from `O(T)`
   to `O(1)`. See [`03`](03_Training_Procedure.md).
3. **Deep supervision** — multiple detached forward "segments" per
   example; dense gradient signal without stacking memory. See
   [`03`](03_Training_Procedure.md).
4. **Adaptive Computational Time (ACT)** — Q-learning halting head
   chooses how many segments to run. Gives inference-time scaling for
   free. See [`03`](03_Training_Procedure.md).

Two secondary findings are striking:

- **Inference-time scaling is free:** a model trained at `M_max = 8`
  gets better at `M_max = 16` with no retraining.
- **Brain correspondence:** after training, the H-module develops a
  substantially higher-dimensional representation (Participation Ratio
  ≈ 90) than the L-module (PR ≈ 30) — ratio ≈ 2.98, close to the
  ≈ 2.25 ratio observed in mouse cortex between high and low levels.
  Untrained HRM shows no such separation — it's *emergent* from
  training.

This is a possibility proof, not a universal replacement for
Transformers. HRM is small, single-task per training run, trained on
grid-based symbolic puzzles — not a general language model. What it
shows is that **architectural depth can substitute for scale** on
tasks that need algorithmic reasoning rather than pattern matching.

---

## 2. Motivation — why CoT isn't enough

### Transformers are shallow

"Deep learning" is a misleading name for the current frontier.
Production Transformers top out at 50–150 layers, and that depth is
**fixed at inference time**. From a circuit-complexity lens, fixed-
depth networks with polynomial-size layers fall into the class **TC⁰**
(constant-depth threshold circuits). They cannot solve problems that
require depth to grow with input size — which includes most things
"algorithmic": graph reachability, propositional SAT, multi-step
symbolic manipulation, backtracking search.

The HRM paper verifies this empirically on Sudoku-Extreme (Figure 2):

- **Widening** a Transformer: no improvement on Sudoku.
- **Deepening** a Transformer: real improvement, but saturates well
  short of optimal accuracy no matter how many layers you add.

This matches complexity theory: you don't escape TC⁰ by adding more
TC⁰.

### CoT is a crutch

The industry workaround is Chain-of-Thought: the model generates
intermediate reasoning tokens, effectively using its own output stream
as external memory. This does lift you out of the fixed-depth limit —
a CoT trace is a variable-length computation — but at real cost:

- **Brittleness.** One misstep or reordering can derail the whole
  trace.
- **Data hunger.** Training strong CoT needs enormous quantities of
  reasoning-style data.
- **Latency.** Every intermediate thought is a generated token, so
  complex reasoning is slow.
- **Language-tethered.** All intermediate state is linguistic, which
  forces translation at every step and constrains what can be
  represented.

HRM's authors argue — drawing on cognitive science — that **language
is a tool for communication, not the substrate of thought**. The brain
sustains long coherent reasoning in latent space without constantly
translating back to language.

### The natural fix is recurrence, but it's historically broken

"Do the reasoning in hidden state, not tokens" suggests a **recurrent
network** iterating on its hidden state. But recurrent nets have two
classic problems:

1. **Premature convergence.** Hidden state settles into a fixed point;
   update magnitudes shrink; subsequent iterations do nothing useful.
2. **BPTT memory cost.** Training through time stores every state;
   `O(T)` memory; biologically implausible; practical bottleneck for
   large batches.

HRM's contribution is architectural + training-procedure choices that
solve both problems together. See
[`02`](02_Hierarchical_Convergence.md) for the architectural mechanism
and [`03`](03_Training_Procedure.md) for the training procedure.

---

## 3. Architecture at a glance

HRM has four learnable modules:

- **Input network `f_I`** — tokens → hidden representation.
- **Low-level module `f_L`** — fast, detailed computation.
- **High-level module `f_H`** — slow, abstract planning.
- **Output network `f_O`** — H-state → prediction.

The key architectural choice is **temporal separation**: `f_L` runs
every step, `f_H` runs once every `T` steps. Inspired by the brain's
cortical areas operating at distinct intrinsic timescales — slow theta
waves (4–8 Hz) in higher areas, fast gamma (30–100 Hz) in sensory
areas — though the paper is explicit that HRM's modules are conceptual
abstractions, not literal oscillation simulations.

A single forward pass runs `N` cycles of `T` low-level steps each —
`N·T` steps total. Within each cycle, `f_L` updates repeatedly while
`f_H` holds fixed; at cycle end, `f_H` updates once based on the final
`f_L` state. After `N` cycles, the output is read from the final
H-state.

**The effective depth of this computation is `N·T`, not `T`**, because
of what happens at cycle boundaries. See
[`02`](02_Hierarchical_Convergence.md) for the mechanism and why it
matters.

---

## 4. Results

### Benchmarks

**ARC-AGI Challenge.** Abstraction and Reasoning Corpus. Each task
gives 2–3 input/output grid pairs and one test input; the model must
infer the abstract rule and produce the correct test output grid.
Famous for being resistant to pretraining — specifically rewards
inductive generalization to novel task structures, not pattern
matching.

**Sudoku-Extreme.** A new hard-Sudoku dataset compiled by the authors,
combining Kaggle/17-clue easy puzzles with Forum-Hard and Forum-
Extreme community-hard puzzles. Mean difficulty: **22 backtracks per
puzzle** (measured with tdoku solver's logic reductions), vs. 0.45 for
the recent Sudoku-Bench. 1,000 training examples for main results;
3,831,994 for analysis experiments ("Sudoku-Extreme-Full").

**Maze-Hard.** Optimal pathfinding in 30×30 mazes, difficulty filtered
to shortest-path length > 110. A path is correct only if it's both
**valid and shortest**. 1,000 examples train, 1,000 test.

### Headline numbers

From Figure 1:

| Model | Params | Training | ARC-AGI-1 | Sudoku-Extreme | Maze-Hard |
|---|---:|---|---:|---:|---:|
| o3-mini-high | large | full LLM + CoT | 34.5% | 0% | 0% |
| Claude 3.7 (8K ctx) | large | full LLM + CoT | 21.2% | 0% | 0% |
| Direct-pred Transformer (8-layer) | 27M | scratch, 1K ex. | ~20% | 0% | 0% |
| **HRM** | **27M** | **scratch, 1K ex.** | **40.3%** | **~100%** | **~100%** |

Three things jump out:

- **CoT LLMs score 0%** on Sudoku-Extreme and Maze-Hard. These
  benchmarks require long trees of backtracking search, and token-by-
  token CoT generation cannot maintain state reliably across that
  many steps.
- **"Direct-pred"** — a Transformer identical in size to HRM, trained
  the same way — fails on Sudoku/Maze and is roughly half of HRM on
  ARC-AGI-1. Same training budget, same data, same size, no
  hierarchical structure → order-of-magnitude worse. This isolates
  HRM's architectural contribution cleanly.
- **HRM beats o3-mini-high on ARC-AGI-1** (40.3% vs 34.5%) despite a
  tiny fraction of the parameters, no pretraining, no CoT. The
  headline comparison of the paper.

### What HRM appears to learn

The authors decode the H-module's state through the output head at
each timestep to visualize intermediate predictions (Figure 7).
Patterns are task-specific:

- **Maze-Hard:** initial multi-path exploration → elimination →
  path construction → refinement.
- **Sudoku-Extreme:** depth-first search with backtracking —
  tentative fills, contradictions, reverts.
- **ARC-AGI:** hill-climbing — incremental board adjustments, no
  visible backtracking.

Different tasks evoke *different latent algorithms* from the same
trained model. HRM isn't learning one algorithm — it's learning a
space of algorithms indexed by task.

---

## 5. Brain correspondence — at a high level

After training, HRM develops a **dimensionality hierarchy** that
mirrors one observed in mouse cortex.

**Participation Ratio (PR)** is a smooth "effective rank" — it
measures how many dimensions of a representation are actually being
used. Mouse-cortex data shows PR increases monotonically from low-
level sensory areas to high-level associative areas (Spearman
ρ = 0.79, p = 0.0003). Trained HRM reproduces this:

- L-module PR: 30.22
- H-module PR: 89.95
- Ratio: ≈ 2.98 (mouse cortex: ≈ 2.25)

Untrained HRM (random weights): both modules have similar, low PRs.
So the hierarchy is **emergent from training**, not architectural.

And H-module PR scales with task diversity — more unique trajectories
→ higher PR. L-module PR stays flat. The H-module expands its
representational capacity as it sees more task variety; the L-module
stays compact and focused.

This matters because standard deep networks often show **neural
collapse** — final-layer features converge to a low-dimensional
subspace aligned with class prototypes. HRM does the opposite, which
is consistent with high-dimensional representations supporting
cognitive flexibility.

Full analysis in [`02`](02_Hierarchical_Convergence.md) §8.

(Caveat: this is correlational evidence. Causally testing whether the
dimensionality hierarchy is *necessary* for reasoning would require
constraining PR during training, which is hard to interpret cleanly
in deep learning.)

---

## 6. Caveats

From the paper and implicit:

- **Task generality.** Demonstrated on symbolic grid tasks. Language,
  open-ended problem solving, multi-modal reasoning: untested.
- **No pretraining.** Results are from-scratch on tiny datasets per
  task. Whether HRM benefits from pretraining is unknown.
- **Single-task per training run.** Each benchmark uses a separately
  trained model. No transfer or multi-task results shown.
- **Per-task `N`, `T` tuning.** How to choose them is unclear; the
  paper picks values empirically.
- **One-step gradient approximation error.** Uncharacterized — see
  [`03`](03_Training_Procedure.md) §3.
- **Adversarial robustness.** Small recurrent models on tiny training
  sets can be fragile out-of-distribution. Not measured.
- **Scaling laws.** "27M is enough" isn't swept against parameter
  count.
- **No wall-clock comparison.** HRM reports accuracy, not latency or
  cost vs. CoT.

---

## 7. How HRM relates to other "LLMs can't compute" approaches

This project's RESEARCH/ notes cover several orthogonal answers to the
same underlying problem.

| Approach | Idea | Strengths | Weaknesses |
|---|---|---|---|
| **Tool use** | Model emits code; external runtime executes; result injected | Works today; any language; no arch changes | Opaque to gradients; round-trip latency; computation outside the model |
| **CALM** (this project) | Model generates naturally; external verifier re-computes claims on CPU and corrects | Works on any existing LLM; no retraining; deterministic backends | Catches errors after generation; verifier is separate from forward pass |
| **Percepta in-model execution** (see [`../01_LLM_Computer_Overview.md`](../01_LLM_Computer_Overview.md)) | Compile programs directly into transformer weights; execute in forward pass via 2D-head attention + HullKVCache | Provably correct; differentiable; no round-trip | Weights compiled not trained; training-at-scale unproven |
| **HRM** (this paper) | Two-timescale recurrent architecture; effective depth via hierarchical convergence; no CoT; tiny training data | Beats CoT on hard reasoning with 27M params; inference-time scalable; brain-inspired | Task-specific per training run; untested on language; no pretraining |

The four are **complementary, not competing**:

- **Tool use** is pragmatic and works today.
- **CALM** adds a correctness safety net on existing models.
- **Percepta** rewrites what a model fundamentally is (weights as
  compiled software).
- **HRM** shows you can get CoT-level reasoning *without* CoT if the
  architecture is right.

If you squint, HRM and Percepta's work are making the same argument
from different angles: **the fixed-depth Transformer is not the right
substrate for reasoning**. Percepta's answer is to replace the decoding
algorithm (2D heads + HullKVCache + compiled weights). HRM's answer is
to replace the depth mechanism (hierarchical recurrence + one-step
gradient). Both sidestep the assumption that adding layers and tokens
is how models get smarter.

---

## 8. Related work (brief)

**Algorithm learning in neural networks.** NTM, DNC, Neural GPU,
Recurrent Relational Networks — earlier attempts, all trained with
BPTT, limited by premature convergence and memory pressure.

**Brain-inspired reasoning.** Spaun, Tolman-Eichenbaum Machine,
neural sampling models — typically rely on hand-designed algorithms
or specialize to simple tasks.

**Hierarchical memory.** Clockwork RNN, Hierarchical Sequential
Models — similar multi-timescale structure but targeted at long-range
dependency modeling (memory), not reasoning depth.

**Adaptive halting.** ACT for RNNs (Graves), PonderNet — predecessors
of HRM's ACT mechanism; HRM's version uses Q-learning and is stable
without the usual tricks (see [`03`](03_Training_Procedure.md) §6).

**Universal Transformers**, **Looped Transformers** — Transformers
with recurrence over layers plus adaptive halting. Philosophical
cousins of HRM, without the hierarchical two-timescale structure.
