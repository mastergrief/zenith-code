# Session Handoff — 2026-04-21 (decode-speedup + eval-defaults)

## Goal

Two tracks, pivoted mid-session:

1. **R13 MBPP walker eval** (primary start): rerun the invalidated R13 N=20 baseline with the full walker stack + sandbox fix, produce the first real MBPP lift count. Plus centralize ctx/max_tokens across all substrate eval scripts.
2. **Decode-speedup to llama.cpp parity** (primary end, user pivot): close the gap between our tq4+CUDA-Graphs decode and llama.cpp's ~42 tok/s on the same GGUF at 256-512K context (user constraint: tq4 KV is mandatory, no reverting to fp16).

User pivot at MBPP#4 prime_num (~26 min decode @ ~9 tok/s): "i think we need to skip r13 for now and get decode up to llama standards." Explicit authorization for weeks-level kernel work: "do this next: The remaining 40% to llama.cpp is Triton vs hand-tuned CUDA matvec performance."

## Completed (8 commits, `8d8a8ec` → `f59ae73`)

### Eval infrastructure (track 1 — pre-pivot)

- **`8d8a8ec`** — `eval_defaults.py`: `EVAL_CTX_SIZE=32768`, `EVAL_MAX_TOKENS=16384` centralized; 12 R-scripts migrated; `kv_max_len` param added to `generate()` for pre-allocated tq4 KV.
- **`805e539`** — Walker expansion: AST-based `_defines_function` extract (prevents docstring substring false-positives), 7th `ast_repair` rewrite `fuzzy_rename_function` (Jaccard ≥ 0.5 over FunctionDefs, rename def + call sites), `extract_undefined_name` (NameError driver), `scripts/bench_decode_paths.py` baseline bench, `bin/mbpp-rotate` shell wrapper, `ITERATION_N=5` / `FINAL_N=20` / `resolve_problem_window()` rotation helpers. +94 ast_repair tests (was 81). MBPP-specific cap `MAX_TOKENS=8192`.

### Decode-speedup push (track 2 — post-pivot, the main arc)

| Round | Commit | Intervention | Result |
|---|---|---|---|
| Baseline | `805e539` | no graphs, tq4 KV | 5.73 tok/s (13% of llama) |
| **R2** | **`bdf67ee`** | **CUDA Graphs for tq4 KV** | **25.02 tok/s (60%), 4.5× lift** |
| R3 | `aa46f2b` | batched pos_t (single tensor + `set_pos_all`) | null (24.62 within GPU variance) |
| R4 | `6b27b90` | `torch.compile` on `_tq4_linear_kernel` | null (24.28), reverted |
| bench | `1780bec` | N_RUNS 3→5 + noise caveat | receipt-only |
| **R5** | **`da382d7` + `f59ae73`** | **k+v kernel fusion via `tq4_linear_dual_triton`** | **1.65× per-kernel microbench, +4.4% e2e est** |

### Verbatim bench numbers (median per path, 256-tok decode)

| Round | A-fp16-KV | B-tq4-KV | C-graph-fp16 | D-graph-tq4 | notes |
|---|---:|---:|---:|---:|---|
| R2 baseline | 7.26 | 5.73 | 32.67 | — | clean |
| R2 ship | 7.14 | 5.56 | 33.35 | **25.02** | clean, D/B 4.5× |
| R3 | 7.20 | 5.59 | 29.93 | 24.62 | C drifted -10% same code |
| R4 | 6.71 | 5.50 | 28.96 | 24.28 | compile recompile overhead |
| R5 N=5 | 7.05 | 5.53 | 28.92 | 21.01 | **rustc-contaminated** (B runs 4-5 dropped to 4.95/3.36) |

### Microbench (k+v fusion, clean env)

| Layer | d_head_k | sep μs | fuse μs | speedup |
|---|---:|---:|---:|---:|
| 0 SWA | 256 (bpr=1) | 202.96 | 134.38 | **1.510× +51%** |
| 5 global | 512 (bpr=2) | 177.45 | 102.15 | **1.737× +74%** |
| 23 global | 512 (bpr=2) | 179.09 | 102.70 | **1.744× +74%** |
| **aggregate** | | **559.50** | **339.24** | **1.649× +64.9%** |

Correctness: max\|Δ\| ≤ 1e-6 (FP noise only). Per-layer saving 73.42 μs × 24 own-KV layers = 1.76 ms/step → **+4.4% e2e at 40ms/step baseline**.

### New mechanisms shipped

- `KVCacheTq4Static` (calm/llm_computer/gemma_substrate.py) — graph-capturable tq4 KV with GPU-resident `pos_t: Tensor[n_layers]`, `valid_mask_all: Tensor[n_layers, max_len]`, bpr ∈ {1,2}. `write_at_graph` uses `index_copy_(1, _bpr_offsets + pos_t*bpr, ...)`. `set_pos_all()` batches to ONE GPU op.
- `generate_with_graph_tq4()` — prefill on dynamic KVCacheTq4, byte-copy transfer to static, capture decode step, replay loop with GPU-tensor pos updates.
- `attn_kv_fused()` method on `GemmaLayer` — routes `k + v` through `tq4_linear_dual_triton` (existing primitive, also used by gate+up FFN fusion).
- Extended `fused_tq4_flash_attn_decode` Pi-unrotate to handle bpr=2 (global layers d_head=512).
- `scripts/bench_decode_paths.py` (4 paths, median-of-5) + `scripts/bench_attn_kv_fused.py` (microbench).

### Key decisions + rationale

- **R13 → decode pivot**: triggered by MBPP#4 prime_num 26-min decode; user authorized all-out decode work.
- **tq4 mandatory**: "we have to use tq4 for 256-512k context so no reverting to fp16" — fp16 KV at 32K ctx OOMs on 8 GB VRAM (5 GB weights + 7.3 GB KV).
- **N=5 iteration cadence**: aligns with `workflow.md` §"The loop should be fast — under 5 min per round". Codified in `eval_defaults.py`.
- **Rotation opt-in, not auto**: clean deltas require fixed problem set between iterations; rotation (window=1,2,3) is for generalization check AFTER converging.
- **Commit each null**: Rounds 3 & 4 shipped null-result commits with receipts per `workflow.md` §"Informative null results". Kept cleaner code from R3 despite no perf win.

## In Progress

**None.** All planned rounds landed with commit receipts. Round 5 clean end-to-end bench attempt aborted mid-run due to codex_tui + 5 rustc procs kicking in from user's `cargo check --workspace`. Microbench is the only clean Round 5 measurement; e2e projection of +4.4% is unverified by end-to-end data.

## ⚠ Uncommitted

`git status --short` — 34 entries, **0 session-critical**:

```
 M .claude/CLAUDE.md                              # USER doc reorg (references new _part_ files)
 D .claude/rules/calm.md                          # USER split → calm_part_1/2.md
 M .claude/rules/training.md                      # USER split (−289 lines → training_part_2.md)
 D .claude/rules/workflow.md                      # USER split → workflow_part_1/2.md
 M calm/hrm/checkpoints/meta_best.pt              # TEAMMATE binary
 M scripts/r52_train_student_kl.py                # TEAMMATE R52.1 work
?? .claude/rules/calm_part_1.md, calm_part_2.md   # USER doc reorg
?? .claude/rules/training_part_2.md               # USER doc reorg
?? .claude/rules/workflow_part_1.md, workflow_part_2.md  # USER doc reorg
?? calm/llm_computer/tq4_autograd.py              # TEAMMATE R52.1 (pre-session, Apr 18)
?? .cache/, .codex/, .port_sessions/, RESEARCH/…  # GITIGNORED caches / training artifacts
?? .claude/MEMORY/minutes/                        # GITIGNORED session transcripts
?? calm/hrm/checkpoints/copy_code_*.pt, math_*.pt  # TEAMMATE training artifacts
?? calm/llm_computer/checkpoints/substrate_hrmlm_v2*  # TEAMMATE
?? calm/llm_computer/r51/checkpoints/             # TEAMMATE R51 student
?? calm/llm_computer/synth/*.jsonl                # TEAMMATE
```

**Session-critical unintentionally uncommitted: NONE.** User-intentional doc reorg (calm.md / workflow.md / training.md split into _part_1/_part_2) is the biggest change and not this session's work — do not touch unless user asks.

## Next Steps (ordered by priority + user-stated focus)

1. **Clean end-to-end bench in idle environment** (PRIORITY 1, user's explicit ask). Verify Round 5 microbench's +4.4% projection on `bench_decode_paths.py` full A/B/C/D run. Runtime pre-req: `pgrep -c rustc == 0 && pgrep -c cargo == 0 && ! pgrep codex_tui`. Expected clean D ≈ 26.1 tok/s = ~62% of llama (not the 90% claim in architecture.md).

2. **Reconcile architecture.md's "42 tok/s / 90% of llama" claim**. User's final session question: "and at steady state we're at 90%?" — answer is NO based on today's measurements. Either (a) architecture.md was aspirational / theoretical, (b) measured before today's R53.34 fused flash-attn + KVCacheTq4Static changes, or (c) measured on truly idle hardware. Next session should re-bench on idle hardware and either confirm 42 tok/s or update the doc to 25-33 tok/s depending on path.

3. **Close decode gap to llama (user authorized weeks-scale work)**. Remaining ~38% gap to 42 tok/s llama target. Ordered kernel-work queue:
   - **Round 6 candidate — q+k+v triple fusion** (Task #23 still pending). Requires new Triton kernel handling heterogeneous out_features (q is 4× bigger than k/v on SWA, 2× on global). Expected +3-5% on top of Round 5.
   - **Round 7 — autotune BLOCK_M + num_warps + num_stages per Gemma shape**. Prior `_pick_block_m` heuristic was bench-tuned globally; per-shape lift may be +5-10%.
   - **Round 8 — extend fused flash-attn with TILE_N blocking** on V-side kernel (currently `for n in range(N)` serial per-program). Parallelize over N positions with atomic_add across (head, d_tile, N_tile) grid. Expected +5-15% at larger N.
   - **Round 9 (weeks) — fused attention-layer mega-kernel** (q/k/v projections + attention + weighted V in ONE Triton kernel, matches llama.cpp's layer-fused CUDA kernel pattern). Biggest potential lift but highest complexity.

4. **Head-to-head vs llama.cpp on same GGUF** (Task #21 still pending). Fixed prompt (100 tok) decoding 500 tok, identical conditions, confirm the 42 tok/s figure or establish the actual gap.

5. **R13 MBPP walker work parked** — when decode stabilizes, resume:
   - `bin/mbpp-rotate 0` → run r53_39 at ITERATION_N=5 for baseline
   - AST-extract + fuzzy_rename walker shipped already (`805e539`) — R13 should now pick up NameError false-positive (MBPP#1 first_repeated_char) as a walker lift
   - Tasks #13 (fence-opener logit hook) and #14 (hard-stop on closing fence) previously queued but deferred behind decode pivot

## Key Context

### Ruled-out paths (cite SHAs to avoid retry)

- **`aa46f2b` — batched pos_t as end-to-end perf win**: 42 × per-layer Python `set_pos` calls are only ~0.8 ms/step theoretical; actual delta below GPU variance. Kept the cleaner single-tensor design anyway.
- **`6b27b90` — torch.compile on `_tq4_linear_kernel`**: dynamic-shape compile lookup overhead exceeds the launch-overhead savings, which are already absorbed by CUDA Graphs. The "6× per-linear" microbench in `gemma_substrate.py:1417` is not end-to-end; graphs already capture that benefit. Don't retry on graph path.

### Methodology caveats

- **Bench-session variance**: path C (same code) measured 32.67 / 29.93 / 28.96 / 28.92 tok/s across 4 different bench sessions in this session. Drift is GPU clock state, not code regression. Compare ratios within a single bench session, not absolute values across sessions.
- **CPU contention invalidates bench**: rustc / cargo check / codex_tui at high CPU steals Python→CUDA dispatch cycles. Path D dropped 25 → 21 tok/s under contention. Always check `pgrep -c rustc; pgrep -c cargo; pgrep codex_tui` before benching.
- **Microbenches are preferred under contention** — kernel-level measurements (single op × N iters + cudaEvent) are less affected by global CPU load than full decode loops (294 kernel dispatches × Python overhead per step).

### Runtime state at session end

- Branch: `feature/multi-agent-qwen` at `f59ae73`
- HEAD: "decode: microbench validates k+v fusion — 1.65× per-kernel, +4.4% e2e"
- gemma daemon: alive (1 `python3 -u bin/gemma_daemon.py`) but idle post-bench
- GPU: 13% util, 1014 MiB used, 6935 MiB free
- `/tmp/gemma_in` FIFO live; `/tmp/gemma_log` ~3.7 KB; `/tmp/substrate_eval_rotation.json` absent (rotation unset)
- `.cache/r53_code_db/` present (90 MB, TF-IDF + dense indices)
- Background contention at session end: 1 rustc + 1 cargo (user's codex compile), 0 codex_tui

### Architecture.md claim to re-validate

`.claude/rules/architecture.md` has multiple instances of "42 tok/s steady decode... 90% of llama.cpp on same GGUF":
- architecture.md: "coherent output at 42 tok/s decode (160× over baseline, 90% of llama.cpp on the same GGUF)"
- Same claim propagated through turboquant.md and other docs

Today's clean measurements: D (tq4+graphs) = 25.02, C (fp16+graphs) = 33.35. **Neither matches the 42 tok/s / 90% claim.** Next session should:
- (a) rebench on truly idle hardware (overnight or when user is away)
- (b) if still 25/33, update all docs referencing "42 tok/s / 90%" to reflect measured floor
- (c) if kernel work closes the gap, commit the new number with same receipt discipline

## Files in Project (session-shipped)

### New files (3)

- `calm/llm_computer/eval_defaults.py` — centralized eval constants + rotation helpers (`EVAL_CTX_SIZE`, `EVAL_MAX_TOKENS`, `ITERATION_N`, `FINAL_N`, `resolve_problem_window`, `read/write_rotation_state`)
- `scripts/bench_decode_paths.py` — 4-path decode bench (A/B/C/D, median-of-5)
- `scripts/bench_attn_kv_fused.py` — k+v kernel microbench (3 layers × 5 runs × 200 iters, cudaEvent timing)
- `bin/mbpp-rotate` — shell wrapper for rotation state

### Modified code (substantial)

- `calm/llm_computer/gemma_substrate.py` — `KVCacheTq4Static` (+~200 LOC), `generate_with_graph_tq4()` (+~120 LOC), `attn_kv_fused()` method on GemmaLayer, `is_tq4_static` branches in `_forward_layer`, SWA trim skip for static tq4, Pi-reshape for bpr=2 q-rotation, `kv_max_len` param on `generate()`
- `calm/llm_computer/tq4_flash_attn.py` — `fused_tq4_flash_attn_decode` Pi-unrotate now handles bpr=2 via `(H, bpr, 256) @ pi` reshape
- `calm/llm_computer/facades/ast_repair.py` — `fuzzy_rename_function`, `extract_undefined_name`, `_name_similarity`, `_FunctionRenamer`, wired into `repair()` dispatch
- `calm/llm_computer/tests/test_ast_repair.py` — +13 fuzzy-rename tests (81 → 94 total passing)
- `scripts/r53_39_mbpp_walker.py` — AST-based `_defines_function` extract, MBPP-specific MAX_TOKENS=8192, `resolve_problem_window()` integration, `USE_TQ4_KV=True`, force-fence fallback
- 11 other R-series scripts — migrated to `from calm.llm_computer.eval_defaults import EVAL_CTX_SIZE, EVAL_MAX_TOKENS, ...`

### Modified docs

None (user owns the doc reorg — don't touch `.claude/rules/` files without explicit ask).

### Deleted

None.
