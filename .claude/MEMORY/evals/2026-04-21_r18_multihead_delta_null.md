# R18 — Multi-head delta state null (2026-04-21)

Hypothesis: splitting DeltaNet's (D, D) state into H independent
(D/H, D/H) heads gives data-efficiency via per-head specialization.
**Null at H=4 on N=15 @ 2K/N:** PT+Delta (21%) actually regresses
vs plain PT (27%) by -6pp. Multi-head at substrate-constrained
d_model=64 reduces aggregate storage and hurts capability.

## Config

Compared against R13-c (H=1 at 2K/N × N=15 → 19% plateau).

  script:  experiment_r10_mqar.py --task mqar
           --per-N-train 2000 --n-values 15 --epochs 20
           --max-len 128 --chunkwise --chunk-size 32
           --n-delta-heads 4
  model:   185K params, d_model=64, n_delta_heads=4
           → per-head state (16, 16) = 256 scalars
           → aggregate 4 × 256 = 1024 scalars (vs single-head 4096)

## Result

  phase        PT best    PT+Δ (H=4) best    Gap
  -----------  --------   ----------------   -----
  N15 final    27%        21%                -6pp

Trajectory (PT+Δ):

  ep 1: N15=17%  loss 1.42
  ep 4: N15=21%  loss 0.97
  ep 8: N15=18%  loss 0.73
  ep12: N15=17%  loss 0.30  (memorization starting)
  ep16: N15=12%  loss 0.04  (val regressing)
  ep20: N15=11%  loss 0.004 (saturated memorization, 11% val)

Loss drops 3 orders of magnitude, val accuracy drops from early
20% to 11%. Classic overfitting signature, not mechanism
emergence.

## Why multi-head hurts at this scale

Aggregate DeltaNet state capacity:

  H=1  (current): 64² = 4096 scalars
  H=2:            2 × 32² = 2048
  H=4:            4 × 16² = 1024   ← R18
  H=8:            8 × 8²  = 512

At H=4, each head must hold ~N=15 (k, v) associations in a 16-dim
key space and 16-dim value space, i.e. (16, 16) = 256 scalars per
head. The theoretical near-orthogonal-key capacity at D=16 is
D/log(D) ≈ 6 — well below N=15. Each head is individually
over-capacity; no amount of per-head specialization recovers what
the single-head (64, 64) = 4096-scalar state provides natively.

Paper DeltaNet uses multi-head at d_head=64 or d_head=128, giving
per-head states of 4K-16K scalars — plenty of headroom. Our
substrate invariant d_head=2 for attention, combined with d_model=64,
leaves no room for the same approach.

## Data-vs-capacity framework, confirmed

R14/R14-b showed: "+5 on N needs 2× data" at single-head d_model=64.
R18 shows: splitting capacity across heads worsens the story. **For
THIS substrate, aggregate DeltaNet state size is a hard constraint
that data can't substitute for once N exceeds per-head capacity.**

Correct way to trade off more heads vs more data (in principle):
bigger d_model. At d_model=128, H=4 gives per-head (32, 32) = 1024,
aggregate 4096 — same as current single-head but with specialization.
That's R11a territory (ran at 2K/N × 40ep, nulled) but would be
cheap to re-run now with chunkwise. **If d_model=128 + H=4 beats
d_model=64 + H=1 at same data budget, multi-head as a pattern
still works — just needs bigger d_model to pay for it.**

## What this doesn't rule out

- **Multi-head at d_model=128+**: each head still has (32, 32) =
  1024 scalars, comparable to current single-head. Worth a re-run.
- **Multi-head with shared state but independent projections**:
  each head has its own W_q/W_k/W_v but reads/writes the same (D, D)
  state. Different specialization mechanism — routes different
  "views" of input into shared storage. Not tested; would need
  separate implementation.
- **Per-head β**: currently β shared across heads. Per-head β
  could help each head specialize, but at d_model=64 the state
  capacity is still the bottleneck.

## Revert decision

Multi-head at d_model=64 is a net regression. The implementation
stays in tree behind `n_delta_heads` config (default 1) for:
- Future use at d_model=128+ (if substrate invariant relaxes)
- Future ablations (per-head β, shared state variants)

Script flag `--n-delta-heads 4` exposed but not default.
Default config path remains the R14-b-validated single-head
chunkwise, which is the production-ready PT+Delta.

## Raw log

- `/tmp/r18_n15_2k_h4.final.log`

## Related rounds

- R13-c (H=1 at 2K/N × N=15): 19% plateau
- R13-d (H=1 at 5K/N × N=15): **99% solved** — data beats multi-head
- R14-b (H=1 at 10K/N × N=20): 99% solved — data still the lever

## Methodology receipt

Multi-head was item #3 on the improvement plan based on a
hand-wavy "more specialization should help" argument. Running it
gave a clean null that clarifies the actual constraint: **at
d_model=64 substrate, aggregate state capacity is load-bearing
above the per-N key-space-density threshold**.

Next items on the improvement plan:
- #4 Mixed-task training (doesn't depend on multi-head)
- #5 Gemma CardSlot install (doesn't depend on multi-head)
- #6 Schlag/write-gate at 2K/N (doesn't depend on multi-head)
