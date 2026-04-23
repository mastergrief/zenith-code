# Architecture — Historical receipts

Per-session validation detail, R-delta commits, decode-perf bench
receipts, production-feature commit-cited bug fixes, compiled-program
exhaustive-test counts. Current architecture: `.claude/rules/architecture.md`.
This file exists for archaeology — "which session shipped which
substrate validation", "which commit fixed which production bug",
"what were the specific measurements that anchored each claim".

## Substrate pattern session-by-session port

- **Session 30**: validated substrate-native demo through Level 5
  on `HybridGroupedSmall2DTransformer`. Key results:
  - Level 5 validated: 3 attention modes in one layer, zero cross-talk
  - Real Gemma bytes: 2 layers byte-installed from GGUF + card, one tensor
  - GPU: 68× speedup at 889M params on RTX 4070
  - Auto-upgrade: CALM → compile → persist, self-improving across sessions
  - HRM 90% autoreg: scheduled sampling, 15 min on RTX 4070
  - Compiled reasoning: comparison, logic, transitivity — exact
  - Persistent knowledge: 0/8 → 11/11 across 3 sessions

- **Session 32**: ported the full pattern to prod Gemma 4 E4B
  (`GemmaSubstrate`). `convert_layer_to_fp32` +
  `install_card_in_attention` + per-sub-head dispatch via
  `attention_partition` — three attention modes coexist in one Gemma
  layer with verified non-zero distinct diffs. Plus residual-additive
  `CardSlot` pattern for cards with custom forwards (PTs).

- **Sessions 31-34**: Gemma substrate loader evolution. 42 layers, GQA
  8Q/2KV, per-layer head dim, proportional RoPE, per-layer embedding
  injection. KVCacheTq4 multi-token prefill S≥1 (R53.28).

## DT / PT+Delta commits (R-delta arc)

- `31337f3` — R-delta-6a PT+Delta NL math (100% val autoreg)
- `63a49fc` — R-delta-20 DT defaults shipping
- `e6f2d5c` — cached decode `decode_greedy_cached` (1.18× plain-PT)
- `f5455f6` — session-32 chained CRLM (PT → adder_tiny →
  VerificationHook)
- `73df738` — R22 initial ship at min_margin=22.0
- `9691e06` — R22f recalibration to 14.5

Full DT arc: `MEMORY/atlas/delta_rule_arc.md`.

## Code-skeleton DT R-numbered flag details

Code-skeleton DT recipe requires R26 aux copy-loss + R27 split-before-aug
+ gate init -1.0 + EMA decay 0.995. Don't extrapolate retrieval
defaults. Full recipe: `.claude/rules/delta_rule.md` §"Code-skeleton
recipe" + `MEMORY/atlas/delta_rule_arc.md`.

## Decode bench receipts (2026-04-21 clean, median-of-5, RTX 4070)

| Path | tok/s | % llama.cpp (~42 tok/s) |
|---|---:|---:|
| tq4, no graphs | 7.14 | 17% |
| tq4 + graphs (`bdf67ee`, 4.5× lift) | 25.02 | 60% |
| fp16 + graphs | 33.35 | 79% |

Historical "42 tok/s / 90% llama" from session 32 unreproducible in
current bench — hardware/driver state dependent.

## Graph-captured tq4 decode (Track A, 2026-04-21)

Prefilled-then-replayed decode path that ships the 4.5× lift over
dynamic-KV decode (commit `bdf67ee`):

- **`KVCacheTq4Static`** — graph-safe subclass of `KVCacheTq4`.
  Shared `pos_t: (n_layers,)` long tensor,
  `valid_mask_all: (n_layers, max_len)` bool, per-layer `_bpr_offsets`.
  Position is a 0-d GPU tensor; writes use `index_copy_`; attention
  reads full `max_len` with additive mask — no Python-int slicing
  breaks graph capture. `set_pos_all()` coalesces 42 per-layer pos
  writes into one GPU op.

- **`GemmaLayer.attn_kv_fused`** — single Triton launch
  (`tq4_linear_dual_triton`) for K and V projections (commit
  `da382d7` / `f59ae73`). Falls back to separate `ak(x), av(x)` when
  `_gpu_qs` unset or Pi buffer unset. Microbench: 1.65× per-kernel
  aggregate (74% on global layers, 51% SWA); e2e +4.4% projected,
  unverified due to rustc contention at session end.

- **Pi bpr=2 extension** — `tq4_flash_attn.py:fused_tq4_flash_attn_decode`
  Pi-unrotate now handles bpr=2 via `(H, bpr, 256) @ pi` reshape.
  Gemma global layers (d_head=512) now work in the fused path.

- **`generate_with_graph_tq4()`** — prefill on dynamic `KVCacheTq4`,
  byte-copy transfer into `KVCacheTq4Static`, 3-iter side-stream
  warmup (warmup-slot invariant: warmup writes the same bytes the
  graph will write at prefill_len), capture one-step graph, replay
  `max_tokens-1` times.

- **`generate(..., kv_max_len)`** parameter — pre-allocates tq4 KV to
  `kv_max_len`; raises if `< len(ids) + max_tokens + 8`. Used by eval
  scripts pinned to `EVAL_CTX_SIZE=32768` from
  `calm/llm_computer/eval_defaults.py`.

Full bench receipt + methodology: `MEMORY/atlas/turboquant_arc.md`.

## Triton kernel + fused flash-attn receipts

- v2 matvec shared-mem LUT default: R53.29, -7% aggregate, commit
  `cbb8073`
- Fused flash-attn decode: R53.34, head-major tq4 K/V storage, reuses
  `tq4_matvec_triton` for scoring, parallel `_tq4_weighted_v_kernel`
  for V-side
- Default-on (`_use_fused_flash_attn=True`) with runtime N-gate
  `128 < cached_kv_len < 2048` per 2026-04-20 re-bench
- SWA layers fused; global d_head=512 fall back to memoized dequant

Full receipts: `MEMORY/atlas/turboquant_arc.md`.

## Fast weights round-by-round receipts

- **Round 1 (d_head=2 baseline)**: 99.1% on held-out 3-pair associative
  recall at d_head=2 (vs vanilla 35.3% — the mechanism works at this
  narrow head dimension, a novel empirical result with no prior
  literature).
- **Round 2 (fusion)**: fast weights stay silent when projections are
  silent — no interference with compiled programs.
- **Rounds 3 (d_model scaling)** and **4 (delta rule + write gate)**
  nulls: diagnosed the n=10 ceiling as structural interference
  (cross-key leakage), not capacity. Tests in
  `tests/test_substrate_extensions.py`.

## Compiled programs exhaustive-test counts

| Program | Test count | Session |
|---|---|---|
| `adder` | 10,000/10,000 exhaustive in 0.38s | session 21 |
| `adder_tiny` | 16/16 exhaustive | — |
| `gcd` | 256/256 | — |
| `factorial` | 9/9 | — |
| `is_prime` | 99/99 | — |
| `dispatched` | 279/279 | — |
| `retrieve_by_index` | 256/256 exhaustive | — |
| `retrieve_threshold` | 256/256 same-layer attn+FFN composition | — |
| `read_by_key` | 96/96 (4! perms × 4 queries) | — |
| `add_one` | 1,280 params | — |
| `copy_past` | 2,560 | — |
| `increment_counter` | 2,176 | — |
| `threshold` | 216 | — |

Each primitive has an `*_ir.py` IR-compiled counterpart; 3 of 4
bit-match the hand-wired version (`copy_past` differs only in head
packing; behavior identical).

Session 30 additions: `compiled_router` (ADD/MUL), `dispatched_v2`
(5 ops), `dispatched_v3` (9 ops), `dispatched_v4` (5 ops + cross-card
gating), `composed_sum_threshold` (inter-slot), `depth_compound`
(3-stage), `reasoning_engine` (comparison + logic + transitivity),
`compiled_in_gemma` (inside Gemma layer), `three_in_one_layer`
(Level 5: 3 modes one layer).

Full inventory: `.claude/rules/Substrate.md` §"Key Files" + atlas
§"Compiled-program validation receipts" in `Substrate_arc.md`.

## CRLM scaling-law empirics (session 26)

| Input language | Max chars | Per-token | Full-expression / structural |
|---|---:|---:|---:|
| Math expression echo (3-digit) | ~20 | 100% | 30/30, smoke 5/5 |
| NL templates ("what is X plus Y?") | ~30 | 99.8% | 29/30, smoke 5/5 |
| Word problems (names, pronouns, multi-step) | 78 | 99.7% | 30/30, smoke 5/5 |
| GSM-style (subordinate clauses, 3-4 terms) | 104 | 99.6% | 28/30 — **first observed ceiling** |

Multi-task HRM (`calm/hrm/checkpoints/multi_task_best.pt`) pools all
four domains into one 48K model: 100% per-token val_acc.

The GSM shortfall was digit transposition — fixed by copy mechanism
in session 31 (93%→95% held-out).

## PT cross-domain results (session 31)

| Domain | Val autoreg | Held-out | Training time | Max input |
|---|---|---|---|---|
| NL math | 100% | 200/200 | 38s | 30 chars |
| Word problems | 98% | 96/100 | 248s | 78 chars |
| GSM-style | 100% | 95/100 | 491s | 104 chars |
| Funcall reasoning | 86% | 171/200 | 611s | 88 chars |
| Logic reasoning | 86% | 88/100 | 910s | 121 chars |
| Creative writing | 96% | 97/100 | 255s | 65 chars |

Session 31 priority-order finding: steps 1-3 took 0%→100% (data),
68%→100% (mechanism), 74%→88% (split). Never needed step 4 (capacity).

## Production-feature commit-cited bug fixes

- **Output dedup `_find_halved_duplicate()`** (commit `3cf1a69`):
  catches `A+A` patterns with any/no separator.
- **Harness double-print fix** (commit `c11232a`): `_streaming_text`
  flag must not be reset in response event handler.
- **`--parallel 1` requirement** (commit `4644051`, session 2026-04-07):
  bin/zenith passes this; manual llama-server invocations must too.
- **Agent context limit lookup invariant** (session 2026-04-07):
  when `backend == "llamacpp"`, must call `detect_llamacpp_model()`
  and pass loaded GGUF path to `detect_context_limit()`. Don't pass
  literal string `"llamacpp"`.
- **Harness loaded-model cache invariant** (session 2026-04-07): both
  `/swap` and `/backend llamacpp` handlers must refresh cache + call
  `_compute_compact_threshold()`.
- **89% safe-ctx margin** (raised from 85% in session 2026-04-08): at
  default 256K ctx the binding constraint is Gemma 232960 = 227.5K,
  giving 29184 tokens of headroom (below max-effort 32768).
- **session 16 alignment fix**: tq4 block padded to 132 bytes (was
  130) so CUDA mmvq / fattn can issue aligned uint32 loads on `qs`.
  Breaks compatibility with pre-132-byte tq4 GGUFs.
- **NIAH validation** (session 2026-04-07): eval report at
  `.claude/MEMORY/evals/2026-04-07_summary_needle_comparison.md`.
  Gemma 4 E4B 200K single-needle PASS; multi-needle degrades to 4/5
  at 220K.

## Cross-refs

- Current architecture spec: `.claude/rules/architecture.md`
- Substrate install detail: `.claude/rules/Substrate.md` + `MEMORY/atlas/Substrate_arc.md`
- DT install arc: `MEMORY/atlas/delta_rule_arc.md`
- tq4 kernel + flash-attn bench: `MEMORY/atlas/turboquant_arc.md`
- Training session-by-session: `MEMORY/atlas/training_part_1.md` + `_part_2.md`
- Capability-gain receipts: `MEMORY/atlas/capability_gain_arc.md`
