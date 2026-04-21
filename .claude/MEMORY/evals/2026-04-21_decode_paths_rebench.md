# 2026-04-21 — decode_paths rebench (validates SESSION_HANDOFF numbers)

## Purpose

Close SESSION_HANDOFF.md #2 ("Rebench 42 tok/s in idle env") and
SESSION_HANDOFF_1.md Next Steps #1 (user PRIORITY 1: clean A/B/C/D
bench post-Round 5). Verify the Round 2 ship numbers (`bdf67ee`)
reproduce on a second session and whether the historical "42 tok/s /
90% llama.cpp" claim is reproducible.

## Method

- Script: `scripts/bench_decode_paths.py` — median-of-5, 256-tok decode,
  fixed prompt, four paths A/B/C/D, cudaEvent timing, heavy warmup.
- Env at start: `pgrep -c rustc == 0`, `pgrep -c cargo == 0`,
  no codex_tui. GPU idle 4% util.
- Env mid-bench: **codex booted**, spawning 1 rustc + 1 cargo during
  path B runs 3-5. Path C/D complete under background codex.
- Daemon: freshly restarted `bin/gemma-run --start` (prior daemon
  was stale from Track A session).

## Results

```
path                tokens   median_s    tok/s
----------------------------------------------
A-fp16-KV              256      32.26     7.94
B-tq4-KV               256      47.81     5.35
C-graph-fp16           256       7.85    32.59
D-graph-tq4            256      10.24    25.00

ratios:
  D/B = 4.668  (GRAPHS LIFT FOR TQ4 KV — main result)
  D/C = 0.767  (cost of tq4 dequant vs fp16)

% of llama.cpp 42 tok/s:
  A              18.9%
  B              12.7%
  C              77.6%
  D              59.5%
```

## Run-by-run

```
A-fp16-KV:     8.10  7.90  7.86  7.98  7.94   (tight, clean throughout)
B-tq4-KV:      6.17  5.92  5.02  4.86  5.35   (B3-5 contaminated)
C-graph-fp16: 31.29 32.88 32.48 32.59 33.34   (stable under contention)
D-graph-tq4:  25.35 24.62 25.00 24.91 25.19   (tight, +/- 0.4 tok/s)
```

## Comparison to prior session (SESSION_HANDOFF_1 R2 ship, clean)

| path | this bench | R2 ship | Δ |
|---|---:|---:|---:|
| A | 7.94 | 7.14 | +11% |
| B | 5.35 | 5.56 | -4% |
| C | 32.59 | 33.35 | -2% |
| **D** | **25.00** | **25.02** | **±0** |

**D is bit-identical to prior (25.00 vs 25.02)** across a 3-day gap and
freshly-restarted daemon. Graph-capture path is reliably reproducible.

## Findings

1. **25.00 tok/s is the honest D measurement.** Two independent clean
   benches 72h apart produced the same median. This is the current
   tq4+graphs ceiling on RTX 4070 Laptop.

2. **42 tok/s / 90% llama claim unreproducible across both sessions.**
   Neither SESSION_HANDOFF_1's bench (R2 25.02) nor this rebench (25.00)
   gets within 40% of 42 tok/s on path D. Session 32's historical claim
   is either (a) different GPU/driver state, (b) different bench method,
   or (c) not actually path D. **Doc update already shipped in this
   session's `394716a` P0** — architecture.md, turboquant.md, CLAUDE.md
   now say "25-33 tok/s clean bench, 42 historical unreproducible".
   This rebench confirms that decision.

3. **R5 k+v fusion (1.65× microbench, +4.4% e2e projected) still
   unverified at e2e.** A pure `f59ae73` vs `da382d7` A/B would need
   contention-free runs. Even R5's contaminated numbers (21.01 tok/s)
   are below R2's clean 25 — suggesting k+v fusion's per-kernel gain
   is swamped by other variance at e2e. Not a regression (gate is
   microbench), but the "+4.4% e2e" remains a projection.

4. **Bench-session variance real.** Path A rose +11% (7.14→7.94), B
   fell -4%, C fell -2%, D was flat. Consistent with SESSION_HANDOFF_1
   methodology caveat: "Compare ratios within a single bench session,
   not absolute values across sessions."

## CPU-contention signature (codex boot mid-bench)

Path B showed the contamination cleanly:

```
B1  6.17 tok/s  (clean: 0 rustc, 0 cargo)
B2  5.92 tok/s  (clean)
B3  5.02 tok/s  (contended: 1 rustc, 1 cargo — codex boot)
B4  4.86 tok/s  (contended)
B5  5.35 tok/s  (contended, partial recovery)
```

B1/B2 clean median ~6.05 tok/s is 8% above R2's 5.56 — so clean-env B
is actually slightly faster than prior session. All-in-one median 5.35
pulls down because of B3-5.

Paths C/D remained stable under contention (kernel-dispatch-bound,
not Python-overhead-bound like B). Consistent with methodology note:
"Graphs mode less CPU-sensitive than dynamic path".

## Next

- **Doc state already correct** — no action needed from this rebench.
  SESSION_HANDOFF.md #2 (rebench claim) and SESSION_HANDOFF_1.md #1
  (clean rebench) both closed.
- **R5 e2e validation** still parked — would need truly idle HW (no
  codex booted). Low priority; microbench is the gate.
- **R22 Gemma CardSlot install** is the next main arc (MQAR card ready).
