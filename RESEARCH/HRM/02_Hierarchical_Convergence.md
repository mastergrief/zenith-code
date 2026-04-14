# Hierarchical Convergence

The core architectural mechanism of HRM — how two coupled modules at
different timescales sustain long computations that a flat RNN can't.

**Concept-owner for:** the premature-convergence problem, HRM's
forward-pass dynamics, how cycle boundaries reset the L-module,
empirical forward-residual evidence, comparison to stacked RNN / deep
feedforward networks, connection to Deep Equilibrium Models, the
Participation Ratio dimensionality hierarchy.

See also: [`01_HRM_Overview.md`](01_HRM_Overview.md) for motivation
and results, [`03_Training_Procedure.md`](03_Training_Procedure.md)
for how to train this thing without BPTT.

---

## 1. The architectural mismatch

A CPU runs a program by reading a small bounded state and writing
back a small update — per-instruction work is essentially constant.

A **standard RNN** doing `N·T` iterations is the opposite. If its
dynamics have a fixed point, state rushes toward it and stops.
Effective depth ≈ `O(log(1/ε))` for convergence to tolerance `ε`,
regardless of how long you actually run. **Premature convergence.**
Further iterations are wasted compute.

You could try to slow convergence deliberately, but marginally stable
dynamical systems are one step away from diverging. Narrow operating
window.

A **standard deep Transformer** is the opposite failure: it has
plenty of operating room within each layer, but depth is fixed at
training time. Effective depth = model depth. No way to get more
without growing the model, which is what the field has been doing
(and running into scaling-wall arguments about).

HRM threads between these.

---

## 2. The HRM forward pass, formally

Four learnable components:

| Symbol | Role |
|---|---|
| `f_I(·; θ_I)` | input network |
| `f_L(·; θ_L)` | low-level module |
| `f_H(·; θ_H)` | high-level module |
| `f_O(·; θ_O)` | output network |

Input is projected once: `x̃ = f_I(x; θ_I)`. Initial states `z_L^0`,
`z_H^0` are sampled from a truncated normal (stdev = 1, truncation 2)
and **kept fixed** throughout training.

At each step `i = 1, …, N·T`:

    z_L^i = f_L(z_L^(i−1), z_H^(i−1), x̃; θ_L)              (every step)

    z_H^i = f_H(z_H^(i−1), z_L^(i−1); θ_H)   if i ≡ 0 (mod T)
          = z_H^(i−1)                         otherwise

After `N·T` steps:

    ŷ = f_O(z_H^(N·T); θ_O)

Two-way coupling:

- `z_H` is held fixed during a cycle — the L-module sees a constant
  high-level context for `T` consecutive steps.
- `z_L` updates continuously under that fixed context.
- At cycle end, `z_H` advances once using the converged `z_L`.

---

## 3. What happens in one cycle

Zoom in on a single high-level cycle, say cycle `k`. The L-module is
conditioned on `z_H^(k−1)`, which doesn't change during the cycle.
So for the `T` low-level steps, `z_L` evolves under a **fixed
recurrence relation**:

    z_L^i = f_L(z_L^(i−1), z_H^(k−1), x̃; θ_L)

This is a standard iterated map. If `f_L` (given fixed context) is
contractive, `z_L` converges toward a **local fixed point**
`z_L*(z_H^(k−1), x̃)` that depends on the current H-state.

Over `T` steps, `z_L` approaches that fixed point closely enough to
be useful. The paper relies on `T` being large enough for near-
convergence per cycle, but not so large that all cycles become
redundant.

At cycle end (`i ≡ 0 mod T`), the H-module takes a single step:

    z_H^k = f_H(z_H^(k−1), z_L^(kT); θ_H)

Now the new `z_H^k` differs from `z_H^(k−1)`, so the L-module's
recurrence relation changes. `z_L^(kT)` is the old cycle's fixed
point — but it's no longer a fixed point for the new recurrence. The
L-module is, in effect, **yanked out of its equilibrium** into a
different dynamics, where it now converges toward a new local fixed
point `z_L*(z_H^k, x̃)`.

---

## 4. Why this avoids premature convergence

Think about it as a trajectory through state space.

**Standard RNN:** approaches one fixed point asymptotically.
Trajectory length is effectively one (then loops at the fixed point).
One fixed point = one answer = bounded depth of reasoning.

**HRM:** approaches a sequence of `N` fixed points, one per cycle.
Trajectory length is `N` — each cycle contributes a distinct near-
equilibrium *jump* to the H-state. `N` fixed points = `N` reasoning
steps.

The key insight: **the H-module's state is doing the reasoning; the
L-module is the engine that resolves each intermediate problem.** The
L-module's job is to settle into an answer *given* a high-level
context. The H-module's job is to decide what the next context should
be, given the last answer.

Effectively, HRM performs `N` stable nested computations. Each is
cheap and well-behaved; the composition is deep and powerful.

### Effective depth is `N·T`

The "effective depth" of this system — the number of serial dependent
steps of useful computation — is `N·T`, not `T`, because the H-module
carries information across cycles and each cycle's L-module dynamics
depends on that accumulated information.

By contrast, a plain RNN run for `N·T` steps has effective depth
roughly `T` (the time it takes to converge), with the remaining
`(N−1)·T` steps being wasted. HRM recovers a factor of `N` in
effective depth over a plain RNN of the same nominal runtime.

---

## 5. Empirical evidence — forward residuals

The paper measures **forward residuals** — how much the state changes
per step — across step count (Figure 3):

- **Standard RNN.** Residuals decay rapidly to near-zero after a few
  steps. Effective depth exhausted. The state stops changing;
  further iterations are wasted.
- **Deep feedforward network.** Residuals are significant only at the
  first and last layers — the classic vanishing-gradient signature.
  The middle of the stack is effectively dead.
- **HRM.** L-module shows **residual spikes at cycle boundaries** —
  where `z_H` resets the context, kicking `z_L` out of equilibrium —
  and converges within each cycle. H-module progresses steadily cycle
  over cycle. Computational activity is **sustained** across the
  whole `N·T` run.

Both failure modes — RNN premature convergence, deep FF vanishing
activations — are eliminated.

### PCA trajectories

Alongside the residual plot, the paper shows PCA-projected
trajectories of the hidden states across training. The HRM H-module
traces a visibly progressing path through low-dimensional PCA space;
the L-module shows cyclic "oscillations" between local equilibria.
The standard RNN's trajectory collapses quickly into a small region.

---

## 6. Stability without instability

One subtle property of HRM's design is that it gets "sustained non-
convergence" **without** being unstable.

Standard tricks to keep an RNN active — tuning eigenvalues close to
1, gated skip connections, noise injection, spectral normalization —
trade convergence quality for stability. HRM doesn't: within a cycle,
the L-module **does** converge cleanly. It's the cycle boundary that
breaks equilibrium, not ongoing instability during the cycle.

This gives you both properties:

- Each cycle is well-conditioned — converges cleanly, produces a
  clean answer.
- The overall trajectory stays active for many cycles — each new
  cycle resets with a fresh context.

The separation of timescales is what enables this. If everything ran
at the same rate, you'd have to trade off stability against activity.
With two rates, you can have both.

---

## 7. Connection to Deep Equilibrium Models

HRM sits on top of **Deep Equilibrium Model (DEQ)** theory.

Standard DEQs:

- Iterate a recurrence `z = f(z; θ, x)` to a fixed point `z*`.
- Use the Implicit Function Theorem to get gradients at `z*` without
  unrolling — avoiding BPTT.
- In practice, converge to a single fixed point. If that fixed point
  is "good enough" for the task, great; if the task needs something
  deeper, you're stuck.

HRM generalizes this: rather than **one** fixed point, **`N` nested**
fixed points. The L-module inherits DEQ-style behavior within each
cycle (converges, supports IFT gradient). The H-module supervises
the progression between fixed points, turning a single equilibrium
into a trajectory of equilibria.

This is why the one-step gradient works for HRM but wouldn't work
equally well for a plain RNN — see
[`03`](03_Training_Procedure.md) §2–3 for the gradient derivation.

---

## 8. The Participation Ratio hierarchy

A striking empirical property of trained HRM: the H-module develops
a much higher-dimensional representation than the L-module, mirroring
a neuroscience finding about mouse cortex. This is the paper's most
speculative claim, but the numbers are suggestive.

### Participation Ratio in one line

For a representation with covariance matrix eigenvalues `{λ_i}`:

    PR = (Σ λ_i)² / Σ λ_i²

Intuition:

- If all eigenvalues are equal (isotropic representation), `PR = n`
  — the full dimensionality.
- If one eigenvalue dominates (rank-1 representation), `PR ≈ 1`.
- In between, PR is a smooth "effective rank" — the number of
  dimensions that actually carry variance.

### The mouse-cortex benchmark

Posani et al. measured PR of mouse cortical population activity
across areas. **PR increases monotonically from low-level sensory
areas** (e.g., primary somatosensory, SSp-n) **to high-level
associative areas** (e.g., secondary motor MOs, anterior cingulate
ACAd). Spearman ρ = 0.79, p = 0.0003. Higher brain regions represent
information in higher-dimensional spaces.

### Trained HRM (Figure 8)

Measured across Sudoku-solving trajectories:

- **L-module PR:** 30.22
- **H-module PR:** 89.95
- **Ratio:** ≈ 2.98 (mouse cortex: ≈ 2.25 — HRM is in the same
  ballpark)

**Scaling with task diversity.** Going from 10 unique Sudoku
trajectories to 100 *increases* H-module PR; L-module PR stays flat.
The H-module expands its representational capacity as it sees more
task variety; the L-module remains compact and focused on its fixed
role (resolving sub-problems given context).

### Control: untrained HRM

Identical architecture, random weights. Results: both modules have
similar, low, stable PRs (`z_L`: 42.09, `z_H`: 40.75, ratio ≈ 1).
**The dimensionality hierarchy is emergent from training**, not
baked into the architecture.

### Why this matters

Standard deep networks often exhibit **neural collapse** — final-
layer features compress toward a low-dimensional subspace aligned
with class prototypes. HRM does the opposite: the H-module expands
its dimensionality.

High-dimensional representations are considered critical for
**cognitive flexibility** — the ability to rapidly re-purpose
internal state for novel tasks. Neural collapse trades flexibility
for class-separability; HRM appears to trade class-separability for
flexibility.

This is consistent with the H-module needing to implement different
algorithms for different tasks (see
[`01`](01_HRM_Overview.md) §4's observation about HRM learning a
space of algorithms indexed by task): a high-dimensional state space
lets many different latent programs coexist.

### Caveats

The brain-correspondence claim is **correlational**. A causal test
would require constraining `z_H`'s PR during training and measuring
the effect on accuracy — but such interventions are hard to interpret
cleanly in deep learning (confounding effects on training dynamics).
Whether the high-PR H-module is *necessary* for reasoning, or just a
free by-product, remains open.

---

## 9. Open questions about the mechanism

- **What determines good `N` and `T`?** The paper picks values; no
  principled procedure. Empirically `N = 4, T = 16` works across
  tasks, but optimal values are task-dependent.
- **How contractive does `f_L` need to be?** The theory assumes
  per-cycle convergence. What happens at the boundary of stability?
  Does training implicitly regulate this?
- **Is the dimensionality hierarchy causally necessary?**
  Correlational evidence only — see §8 caveats.
- **Does hierarchical convergence help for non-symbolic tasks?**
  All demonstrated tasks are grid-based symbolic puzzles. Unclear
  if the same mechanism buys you anything on language or multi-modal
  inputs.
- **Does it compose with standard Transformer training?** Could you
  bolt HRM-style hierarchical convergence onto a pretrained LM?
  Open.
