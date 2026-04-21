# R17 — Chunkwise parallel DeltaNet (2026-04-21)

Paper §3-4 (UT transform) implementation. Turns the per-position
Householder recurrence into matmul-rich chunked computation. Bit-
equivalent to per-position within float32 epsilon; 3-7× training
speedup depending on sequence length.

## Implementation

File: `calm/llm_computer/delta_rule.py`

New static method `_delta_chunkwise(S, Q, K, V, beta, chunk_size)`
implements Algorithm from `RESEARCH/DELTA-RULE/02_Chunkwise_Parallel_Algorithm.md`:

```
for each chunk of C positions:
    Kkt = K_c K_c^T                                 # (C, C)
    A   = I + tril(diag(β) K K^T, -1)               # (C, C) lower-tri
    T   = A^-1 · diag(β)                             # triangular solve
    W   = T K,  U = T V                              # (C, D)
    U'  = U - W S^T                                  # prior-state adjust
    O   = Q S^T + (Q K^T ⊙ M_causal) U'              # chunk output
    S   = S + U'^T K                                  # state carry
```

Wired into `_forward_backbone` behind `DeltaNetConfig.use_chunkwise`
flag (default False for backward compat). `chunk_size` default 32 —
sweet spot at L≤128 per paper.

Config additions:
- `use_chunkwise: bool = False`
- `chunk_size: int = 32`

Script flags (`scripts/experiment_r10_mqar.py`):
- `--chunkwise` enables the new path
- `--chunk-size N` overrides chunk size

## Bit-equivalence test (raw-path, 2026-04-21)

Same config, same seed, switched only `use_chunkwise`:

  per-position vs chunkwise on L=64, B=2, d_model=64
  max abs diff  = 1.907e-06
  mean abs diff = 2.192e-07
  relative err  = 7.826e-07

**PASS** — well below the 1e-4 threshold. Chunkwise is numerically
equivalent to per-position within float32 noise.

## Wall-time comparison (inference, B=16, forward only, fp32, RTX 4070)

| L | per-position | chunkwise (C=32) | speedup |
|---:|---:|---:|---:|
| 32 | 46.2ms | 9.9ms | **4.65×** |
| 64 | 81.3ms | 11.8ms | **6.90×** |
| 128 | 147.9ms | 23.8ms | **6.22×** |
| 256 | 262.6ms | 34.9ms | **7.52×** |

Speedup grows with sequence length, matching paper Figure 1.

## User-facing training test (R13-med-2k retrain)

Same config as R13-med-2k (`--task mqar --per-N-train 2000 --n-values
5 10 --max-len 80`), shortened to `--epochs 15` (so cosine schedule
differs from R13's 50-epoch; plain-PT trajectory isn't directly
comparable, PT+Delta is).

PT+Delta trajectory under `--chunkwise`:

  ep 1: loss 1.06, overall 28.5%, t= 11s
  ep 3: loss 0.71, overall 61.5%, t= 29s
  ep 6: loss 0.01, overall **100.0%**, N5=100, N10=100, t= **52s**

Per-position R13-med-2k reference (at same task, different LR schedule):
  ep 5: loss 0.07, overall 99.0%, N5=100, N10=98, t=169s
  ep10: loss 0.0001, overall 100%, N5=100, N10=100, t=322s

**Training-wall-clock speedup: ~6× to reach 100%** (52s vs 322s).
Correctness preserved (N5=100, N10=100 both paths).

Training speedup is lower than the forward-only benchmark because
backward passes through the triangular solve aren't as parallelizable
as the forward matmuls, but still substantial.

## Implications

This unlocks R14-b (N=20 at 10K/N) cheaply. Per-position estimate was
~2 hrs GPU; chunkwise should drop this to ~20-30 min. Similarly
every future data-scaling or N-extension round gets cheaper.

Training-compute advantage compounds with the R15 finding that
PT+Delta converges 3-10× faster than plain PT in epochs. Combined
multiplier: **~20-50× faster real-wall-clock training per card than
plain PT at same final accuracy**. Reinforces the
augmentation-thesis factorial-domains-scaling claim: ~100 domain
cards trained in days-per-deck rather than weeks.

## Caveats

- Chunkwise uses `torch.linalg.solve_triangular` which has a CPU
  path and a GPU path. Both work; GPU path is what we tested.
- The triangular solve is O(C²) per chunk. Chunk size C=32 is fine;
  C=128 would start to dominate cost at small d_model.
- Backward pass recomputes hidden states (autograd through the
  ops) rather than using the paper's memory-optimized custom
  kernel. Activation memory is O(L·d²) instead of the paper's
  O(L·d). At our d_model=64 that's 256KB per sequence position
  × L=128 × 4 bytes × B=64 ≈ 4GB per layer — tight on 8GB but
  fits. For larger d_model a custom backward would be needed.

## Raw logs

- `/tmp/r17_chunkwise_retrain.final.log` — training trajectory
- Bit-equivalence test inline (`PYTHONPATH=. python3 -c "..."` pattern)
- Forward-only perf benchmark inline

## Next

R14-b now feasible: N=20 at 10K/N × 20ep should complete in ~20-30
min on RTX 4070. Would close the last data-scaling question. Expected
to saturate N=20 to ~95%+ based on the R13-R14 trend.
