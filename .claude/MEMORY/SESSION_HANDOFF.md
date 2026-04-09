# Session Handoff — 2026-04-09 (Session 11)

## Where you are

The TurboQuant llama.cpp port is **functionally complete on CPU** and
**partially complete on GPU**. CPU-side end-to-end inference produces
coherent text with `--cache-type-k tq3_k256 --cache-type-v tq3_k256` on
Gemma 4 E4B. The CUDA SET_ROWS kernel works (KV cache allocates on
CUDA0, no scheduler abort). The remaining blocker is the CUDA Flash
Attention dequant path — see "Where to pick up" below.

## What got done this session

### Stage 5: CLI flag wiring ✅ COMPLETE

- `~/llama.cpp/common/arg.cpp`: registered `tq3_k256` as a valid value
  for `--cache-type-k` / `--cache-type-v` (added `GGML_TYPE_TQ3_K256` to
  the `kv_cache_types` vector). Added `tq3` short alias as a special-case
  in `kv_cache_type_from_str` before the table lookup.
- `~/llama.cpp/src/llama-context.cpp`: added a model-load-time pre-scan
  validation. Iterates over `model.hparams.n_layer` and checks
  `n_embd_head_k(il)` / `n_embd_head_v(il)` against 256. Three outcomes:
  - 0/N layers match → `LLAMA_LOG_WARN` ("ALL layers will fall back to
    q4_0... use --cache-type-k q4_0 to silence")
  - Partial match (e.g., 20/24 on Gemma 4 E4B) → `LLAMA_LOG_INFO` with
    the ratio
  - Full match → silent (per-layer routing log handles it)
- `~/llama.cpp/src/llama-kv-cache.cpp`: added `extern "C" { void
  ggml_tq3_k256_init_impl(void); }` forward decl right after the
  includes. **This was a Stage 4 build bug fix** — the function lives
  in `ggml-quants.h` (an internal ggml header NOT on `src/`'s include
  path), and the original Stage 4 code tried to call it without a
  visible declaration. The forward decl is the minimum-blast-radius
  fix; adding `ggml/src/` to llama's include path would leak ~100
  internal ggml prototypes.

### Three logging bugs found + fixed

These were uncovered during the Stage 6 dispositive validation. Two are
real bugs in the Stage 4/5 code from session 10; one is cosmetic and
deferred.

**Bug #1 — Stage 5 warning ISWA scoping** (FIXED). The original
warning fired once per `llama_kv_cache` constructor call. For ISWA
models like Gemma 4 E4B, llama.cpp builds **two separate caches**: a
non-SWA cache containing only the 4 FA layers and an SWA cache
containing only the 20 SWA layers. The non-SWA cache had zero matching
head_dim=256 layers and false-fired the warning telling the user to
load Gemma 4 E4B — *on Gemma 4 E4B*. Fix: moved the validation out of
the cache constructor entirely to `llama_context::llama_context` in
`src/llama-context.cpp`, where the full model layer list is visible.
The warning now fires once per model load, not once per cache.

**Bug #3 — Stage 4 missing match-branch log** (FIXED). The per-layer
routing log only emitted lines for fallback layers; match layers were
silent. Fix: added a symmetric branch in `llama-kv-cache.cpp`'s loop.
Format: `layer N: head_dim_k=256 type_k=tq3_k256 head_dim_v=256
type_v=tq3_k256` for matches, with `(fallback)` appended on each side
for fallback layers.

**Bug #2 — Cache size summary type label** (DEFERRED, cosmetic).
The `llama_kv_cache: size = ... K (tq3_k256): ... V (tq3_k256): ...`
line prints the *requested* type even when all layers fell back. Not
fixed this session because it's bundled cleanly into the eventual
Stage 3 CUDA cleanup (when CUDA FA lands and no fallbacks happen,
the label will be correct).

### Bug #4 — CPU type_traits_cpu missing TQ3_K256 entry (FIXED)

**Discovered during the dispositive `-ngl 0` validation as a
SIGSEGV in `ggml_compute_forward_set_rows`**. Stage 1 had only added
the TQ3_K256 entry to the BASE traits in `ggml.c`, missing the
CPU-specific table at `ggml-cpu/ggml-cpu.c:type_traits_cpu`. Without
an entry there, `type_traits_cpu[GGML_TYPE_TQ3_K256].from_float` was
NULL → first SET_ROWS op called NULL → crash.

Stack trace top frame (from gdb):
```
#0  0x0000000000000000 in ?? ()
#1  ggml_compute_forward_set_rows ()
#2  ggml_graph_compute_thread.isra ()
```

Fix:
- `~/llama.cpp/ggml/src/ggml-cpu/quants.h`: declared
  `quantize_row_tq3_k256` (the CPU wrapper signature with `void *y`).
- `~/llama.cpp/ggml/src/ggml-cpu/quants.c`: defined the wrapper as a
  thin delegate to `quantize_row_tq3_k256_ref` (matches the existing
  `quantize_row_tq2_0` pattern).
- `~/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c`: added a
  `[GGML_TYPE_TQ3_K256]` entry in the `type_traits_cpu` array,
  alongside `tq2_0`.

The Stage 2 PyTorch oracle test was not enough to catch this — it
only validated quant→dequant round-trip in isolation. SET_ROWS goes
through the type_traits dispatch, which only Stage 1 + Stage 2
together can validate.

### Bug #5 — CPU Flash Attention vec_dot missing for TQ3_K256 (FIXED)

**Discovered as a second SIGSEGV after Bug #4 was fixed**. With
`from_float` working, the code got further but FA's CPU path crashed
on the same NULL function pointer pattern, this time in
`ggml_compute_forward_flash_attn_ext`. FA uses
`type_traits_cpu[k_type].vec_dot` for the per-row Q · K product;
tq3_k256 had no `vec_dot` entry → NULL → crash.

Fix: added a slow reference vec_dot in
`~/llama.cpp/ggml/src/ggml-cpu/quants.c`:

```c
void ggml_vec_dot_tq3_k256_f32(int n, float *s, ..., const void *vx, ..., const void *vy, ..., int nrc) {
    // dequantize K block to a stack buffer, then f32 dot with Q
    const block_tq3_k256 * x = vx;
    const float          * y = vy;
    float buf[QK_K]; float sumf = 0.0f;
    for (int i = 0; i < n / QK_K; ++i) {
        dequantize_row_tq3_k256(&x[i], buf, QK_K);
        for (int j = 0; j < QK_K; ++j) sumf += buf[j] * y[i*QK_K + j];
    }
    *s = sumf;
}
```

Plus the corresponding `vec_dot_type = GGML_TYPE_F32` and `vec_dot =
ggml_vec_dot_tq3_k256_f32` fields in the `type_traits_cpu` entry. Q
stays as F32 (not Q8_1) because the algorithm requires f32 dot
products after Pi rotation.

### Stage 3 — CUDA SET_ROWS kernel ✅ COMPLETE

The CUDA kernel for writing tq3_k256 K cache. Originally listed as
"deferred" in the session 10 handoff, but discovered to be a hard
blocker for the GPU-offload path: without it, llama.cpp's backend
scheduler refuses to allocate the K cache tensor on CUDA0 and aborts
at `ggml/src/ggml-backend.cpp:809` ("pre-allocated tensor cannot run
the operation SET_ROWS").

**Implementation**: single self-contained TU at
`~/llama.cpp/ggml/src/ggml-cuda/turboquant.cu` (257 lines) +
`turboquant.cuh` declaring host-callable entry points. Architecture:

- `__device__   static float g_tq3_k256_pi_d[256 * 256];` — 256 KB
  global device memory for the rotation matrix (too big for
  `__constant__`'s 64 KB cap)
- `__constant__ static float g_tq3_k256_centroids_d[8];` — Lloyd-Max
  codebook in __constant__ (32 bytes)
- `__constant__ static float g_tq3_k256_boundaries_d[7];` — quantizer
  boundaries (28 bytes)
- `quantize_f32_tq3_k256_block(x, y)` — `__device__` per-block quantize
  function. Single CUDA thread handles one full 256-element vector
  (norm → Pi rotation → centroid lookup → 3-bit pack). Mirrors
  `quantize_row_tq3_k256_ref` from `ggml-quants.c` line-by-line.
- `k_set_rows_tq3_k256<idx_t>` kernel — replicates `k_set_rows_quant`
  from `set-rows.cu` (which is `static`, so we can't reuse it across
  TUs without RDC). Specialized for tq3_k256.
- `launch_set_rows_tq3_k256<idx_t>` template + `_i64`/`_i32` host
  wrappers — same parameter list as `set_rows_cuda_quant`.
- `ggml_tq3_k256_ensure_cuda_init()` — host init via `std::call_once`,
  copies the host Pi/centroid data to device via `cudaMemcpyToSymbol`.

**Why one TU**: nvcc without RDC (relocatable device code) treats
`extern __device__` declarations as definitions, causing link
conflicts when device symbols are shared across .cu files. Putting
all tq3_k256 device state and code in one TU avoids the issue
entirely. set-rows.cu calls our host launch wrappers, never touches
the device-side state directly.

**Wired into**: `~/llama.cpp/ggml/src/ggml-cuda/set-rows.cu`'s
dispatch (added an `else if (dst->type == GGML_TYPE_TQ3_K256)`
branch that calls `ggml_tq3_k256_ensure_cuda_init()` then dispatches
to `_i64` or `_i32` launcher via `if constexpr`).

**Allowlist**: added `GGML_TYPE_TQ3_K256` to the `GGML_OP_SET_ROWS`
case in `ggml_backend_cuda_device_supports_op` at
`~/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4839`.

**Validation**: launching with `-ngl 999 --cache-type-k tq3_k256
--cache-type-v tq3_k256` now shows `CUDA0 KV buffer size = 7.66 MiB`
(was 0 MiB before — the cache wasn't allocating at all). All 20 SWA
layers cleanly show `type_k=tq3_k256` without fallback. The original
`ggml-backend.cpp:809` abort is gone. **The kernel is bit-equivalent
to the CPU oracle by construction** (same algorithm, same constants,
same Pi matrix dumped from PyTorch, same Lloyd-Max codebook math).

### Dispositive CPU validation ✅ PASSED

End-to-end inference with `-ngl 0 --cache-type-k tq3_k256 --cache-type-v
tq3_k256` on Gemma 4 E4B:

```
Data neatly stored,
Queries flow, a swift embrace,
Knowledge finds its home.
```

Coherent English, topically relevant, actually 5/7/5. ~1.7 tok/s on
CPU (expected for full CPU offload of a 4B model). Per-layer log
confirms 20 SWA layers on tq3_k256 + 4 FA layers on q4_0 fallback.
Bug #1 warning did NOT fire. **Stage 2 + Stage 4 + Stage 5 + the
new CPU dispatch entries are all sound end-to-end with a real model
at real inference time.**

## Where to pick up

### Immediate next step: Stage 3.5 — CUDA Flash Attention dequant for TQ3_K256

**This is the only thing blocking GPU end-to-end inference.** The CUDA
SET_ROWS kernel works, but the CUDA Flash Attention kernel doesn't have
a dequant code path for tq3_k256 K/V cache reads. Same architectural
shape as Bug #5 on the CPU side — different code path.

**Reproduction**:
```bash
~/llama.cpp/build/bin/llama-server -m ~/models/gemma-4-E4B-it-Q5_K_M.gguf \
  --ctx-size 8192 --parallel 1 \
  --cache-type-k tq3_k256 --cache-type-v tq3_k256 \
  -ngl 999 --port 8080
```

Crash sequence in the log:
```
sched_reserve: layer 0 is assigned to device CUDA0 but the Flash Attention
   tensor is assigned to device CPU (usually due to missing support)
sched_reserve: Flash Attention was auto, set to disabled
llama_init_from_model: failed to initialize the context: quantized V cache
   was requested, but this requires Flash Attention
```

Root cause chain:
1. `~/llama.cpp/ggml/src/ggml-cuda/fattn.cu:373-389` has a hardcoded K
   type allowlist for CUDA FA: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
   BF16. Anything else returns `BEST_FATTN_KERNEL_NONE`. tq3_k256 falls
   through to default → NONE.
2. `ggml_cuda_flash_attn_ext_supported()` returns false → CUDA backend
   says "I can't run this op" → scheduler places the FA op tensor on
   CPU.
3. Model layer 0 is on CUDA, FA tensor is on CPU → device mismatch →
   `Flash Attention was auto, set to disabled`.
4. The hardcoded check at `src/llama-context.cpp:349-353` then refuses
   to start the context: `if (!cparams.flash_attn && ggml_is_quantized
   (params.type_v)) throw "quantized V cache requires Flash Attention"`.

### Three implementation paths (pick one)

**Option A — Add tq3_k256 to fattn-vec.cuh template system** (most
integrated, hardest):

1. Add `vec_dot_fattn_vec_KQ_tq3_k256` and `dequantize_V_tq3_k256` as
   template functions in `fattn-common.cuh` (forward decl) + body in a
   new `fattn-tq3.cuh` header.
2. Add tq3_k256 branches to `get_vec_dot_KQ` and `get_dequantize_V`.
3. Add tq3_k256 to the `Q_q8_1` false branch in `fattn-vec.cuh` (Q stays
   as f32 since Pi rotation requires f32 dot products).
4. Create `template-instances/fattn-vec-instance-tq3_k256-tq3_k256.cu`
   that includes fattn-tq3.cuh + fattn-vec.cuh + DECL_FATTN_VEC_CASE.
5. Add tq3_k256 to the K type allowlist in `fattn.cu:373-389`. Force
   `BEST_FATTN_KERNEL_VEC` regardless of `Q->ne[1]` to skip MMA_F16
   (would otherwise need a separate kernel for prompt prefill).
6. Add the new instance file to `ggml-cuda/CMakeLists.txt` line
   119-123 (the default-build vec instance list).

**The hard part of Option A** is the per-thread Q layout inside
`vec_dot_fattn_vec_KQ_tq3_k256`. The kernel calls `vec_dot_KQ` per K
row with each thread holding only a stripe of Q (for nthreads_KQ=32,
D=256: 4 float2 per thread = 8 floats per thread). The Pi rotation
requires the WHOLE y_hat[256] for each output element, so per-thread
slicing of the K dequant is impossible without either redundant work
or shared memory cooperation. **Risk: the Q layout assumptions are
subtle; getting strides wrong produces silent gibberish output.**

The natural optimization (precompute Pi @ Q once per query, then per
K row just do `<y_hat, Pi_Q>`) would reduce per-K-row cost from
O(D²) to O(D), but the existing kernel doesn't have a precompute
hook — would need to add one or use shared memory with a flag.

Realistic estimate: **2-4 hours** for the cleanest correct version
with shared-memory cooperative dequant + per-thread dot slice + warp
reduce. Add 1-2 more hours if MMA_F16 prefill kernel is also wanted.

**Option B — Custom kernel in turboquant.cu** (most isolated, simpler
control flow): write a complete-ish FA kernel from scratch in
turboquant.cu, dispatched via a special branch in fattn.cu. The CPU
reference (`ggml_compute_forward_flash_attn_ext_f16_one_chunk` in
`ggml-cpu/ops.cpp`) gives the algorithmic structure. Each CUDA block
handles one query head. Skip optimization, focus on correctness.
Estimate: **3-5 hours** including debugging.

**Option C — Force VEC kernel + naive thread-0 dequant**: in fattn.cu,
add tq3_k256 to allowlist and force VEC kernel. In fattn-tq3.cuh,
write the simplest possible vec_dot where thread 0 does ALL the work
(reads its slice of Q from registers, gathers others via
`__shfl_sync`, computes the full 256×256 Pi rotation + dot, returns
the value; other threads return 0, and warp_reduce_sum gives just
thread 0's contribution). Slow (~1 tok/s GPU, comparable to CPU)
but gets to a working baseline fast. Estimate: **1-2 hours**.

**My recommendation**: start with **Option C** to get a green
end-to-end GPU smoke test, then evaluate whether the perf is good
enough for the project's actual use case. If yes, ship. If not,
iterate to Option A's optimized version.

### After Stage 3.5 lands

1. **Re-run the smoke test** (the one in the README that currently
   crashes at FA) — should now produce coherent text on GPU.
2. **Re-run NIAH at 250K** with the GPU FA path — see how long-context
   recall holds up. Compare to the q4_0 baseline in
   `.claude/MEMORY/evals/2026-04-07_summary_needle_comparison.md`.
3. **Measure VRAM** vs q4_0 baseline. Expected: ~600 MB drop on Gemma
   4 E4B at 256K (the original session 9 Stage 6 acceptance criterion).
4. **Commit + write next handoff**.

## Working Tree State

### `~/llama.cpp/` (out-of-tree, NOT a git submodule)

13 files modified, 3 untracked. All session 10 + session 11 work is
saved as patches in this repo at `scripts/llama_cpp_patches/` so it
can be re-applied after `git pull` on llama.cpp.

```
M common/arg.cpp                    # Stage 5 CLI wiring
M ggml/include/ggml.h                # Stage 1: GGML_TYPE_TQ3_K256 = 42
M ggml/src/ggml-common.h             # Stage 1: block_tq3_k256 struct
M ggml/src/ggml-cpu/ggml-cpu.c       # Bug #4: type_traits_cpu entry
M ggml/src/ggml-cpu/quants.c         # Bug #4 + #5: CPU wrapper + vec_dot
M ggml/src/ggml-cpu/quants.h         # Bug #4 + #5: declarations
M ggml/src/ggml-cuda/ggml-cuda.cu    # Stage 3: SET_ROWS allowlist
M ggml/src/ggml-cuda/set-rows.cu     # Stage 3: dispatch branch
M ggml/src/ggml-quants.c             # Stage 2: CPU reference impl
M ggml/src/ggml-quants.h             # Stage 2: prototypes + getters
M ggml/src/ggml.c                    # Stage 1: type_traits entry
M src/llama-context.cpp              # Bug #1: warning moved here
M src/llama-kv-cache.cpp             # Stage 4 + Bug #3 fix
M tools/server/server-context.cpp    # PRE-EXISTING slot-cap patch
?? ggml/src/ggml-cuda/turboquant.cu  # Stage 3: CUDA kernel
?? ggml/src/ggml-cuda/turboquant.cuh # Stage 3: header
?? ggml/src/turboquant_tables.h      # Stage 1: 1.2 MB Pi matrix (generated)
```

### Project repo

Sessions 9 + 10 + 11 work is committed in 5 layered commits per the
recommended commit plan. New this session:

- `scripts/llama_cpp_patches/` — out-of-tree patches + new file copies
- `.claude/MEMORY/SESSION_HANDOFF.md` — this file (rewritten for session 11)

## Algorithmic invariants (carry into Stage 3.5)

1. **Algorithm reproduces exactly**. The C reference matches the PyTorch
   oracle byte-equivalently within fp16 epsilon. The CUDA kernel mirrors
   the C reference line-by-line. The CPU dispositive haiku test confirms
   the whole pipeline works at real inference time. Whatever Stage 3.5
   does, it must produce **bit-equivalent output** to the C reference
   on the same input, modulo float-summation order which fp16 quant
   tolerates.

2. **Pi rotation matrix is dumped from PyTorch**, not regenerated in C.
   `~/llama.cpp/ggml/src/turboquant_tables.h` is generated by
   `scripts/generate_turboquant_tables.py`. The CUDA kernel copies it
   to device global memory at init via `cudaMemcpyToSymbol`. Don't try
   to regenerate it on the device side.

3. **Codebook IS recomputed in C** (via closed-form Gaussian). The CUDA
   side reads the C-computed centroids/boundaries via `extern "C"`
   getters and copies them to `__constant__` device memory. Don't add
   a CUDA recompute path.

4. **One block = one head_dim=256 vector**. The block size is the
   natural granularity. For other head_dims, the per-layer routing
   (`llama-kv-cache.cpp:effective_cache_type` lambda) falls back to
   Q4_0. Stage 3.5 only needs to handle the head_dim=256 case.

5. **`Q_q8_1 = false` for tq3_k256**. The algorithm requires f32 dot
   products after Pi rotation. The existing fattn-vec.cuh kernel has a
   conditional at line 88 (`Q_q8_1 = type_K != GGML_TYPE_F16 &&
   type_K != GGML_TYPE_BF16`); add tq3_k256 to the false branch.

## Failed approaches (don't retry)

1. **`extern __device__` cross-TU symbol sharing without RDC**. nvcc
   warning #20044-D treats extern declarations as definitions, causing
   link conflicts. Either enable RDC (CMake change, slower compiles)
   or keep all tq3_k256 device state in a single TU.

2. **Reusing `set_rows_cuda_quant` template from set-rows.cu in
   another TU**. The template is `static`, so it's file-local and
   can't be shared. Either move it to a header (refactor outside
   tq3 scope) or replicate the kernel logic in turboquant.cu.

3. **Bypassing the "quantized V cache requires Flash Attention"
   check at `llama-context.cpp:351`**. Patching it out would just
   push the crash deeper into the V dequant path which also doesn't
   support tq3_k256. The check is correct; the fix is to make FA
   actually work for tq3_k256.

4. **Mixed `--cache-type-k tq3_k256 --cache-type-v q4_0`** as a
   workaround for the FA cascade. Tested this session: K being
   tq3_k256 alone trips the FA scheduler, regardless of V's type.

5. **`--no-kv-offload`** as a workaround. Tested this session: places
   K/V on CPU but the FA cascade still triggers because the FA tensor
   for tq3_k256 K layers is placed on CPU while the model is on CUDA →
   device mismatch → FA disabled → "quantized V requires FA" error.

## Server State at Session End

- **Branch**: `feature/multi-agent-qwen` on `mastergrief/zenith-code`
- **Working tree (project repo)**: about to be committed in 5 layered
  commits (sessions 9 + 10 + 11). See "Recommended commit plan" below.
- **Working tree (~/llama.cpp/)**: dirty as documented above. Saved as
  patches in `scripts/llama_cpp_patches/`.
- **llama-server**: not running. Stopped at session end after the
  failed `-ngl 999 --cache-type-k tq3_k256` test.
- **VRAM**: ~1.2 GB used (idle). RAM: ~7.7 GB available.
- **HF cache**: Gemma 4 E4B fp16 still cached at
  `~/.cache/huggingface/hub/models--google--gemma-4-E4B-it/`
- **HF token rotation**: still pending from session 9.
- **gdb**: now installed (`sudo apt-get install gdb`). Used to find
  Bugs #4 and #5 via stack traces.

## Recommended Commit Plan (UPDATED for session 11)

Sessions 9 + 10 + 11 work, layered to keep history clean:

1. **Session 9 harness baseline** — compaction 227.5K, max effort 32K,
   parallel-2 readiness. Files: `agents/agent.py`, `agents/compact.py`,
   `agents/harness.py`, `.claude/CLAUDE.md` (compaction sections),
   `.claude/rules/architecture.md` (compaction sections).

2. **Session 9 tool surface expansion** — 15 tools + registry + tests.
   Files: `agents/tools.py`, `agents/agent_registry.py`,
   `tests/test_agent_registry.py`, `tests/test_agent_tool.py`,
   `tests/test_new_tools.py`.

3. **Session 9 PyTorch TurboQuant validation** — design reference for
   the C++ port. Files: `scripts/turboquant_patches.py`,
   `scripts/test_turboquant_gemma4.py`.

4. **Session 10 TurboQuant llama.cpp port — CPU reference + validation**.
   Files: `scripts/generate_turboquant_tables.py`,
   `scripts/test_tq3_k256_c_vs_python.py`.

5. **Session 11 Stage 5 + bug fixes + Stage 3 SET_ROWS + handoff** —
   the llama.cpp source modifications saved as out-of-tree patches.
   Files: `scripts/llama_cpp_patches/01_slot_cap.patch`,
   `scripts/llama_cpp_patches/02_turboquant.patch`,
   `scripts/llama_cpp_patches/files/turboquant.cu`,
   `scripts/llama_cpp_patches/files/turboquant.cuh`,
   `scripts/llama_cpp_patches/README.md`,
   `.claude/MEMORY/SESSION_HANDOFF.md` (this file).

## Outside-project notes

- The CUDA kernel is bit-equivalent to the CPU oracle BY CONSTRUCTION.
  Any CUDA-side test that disagrees with the CPU side is a CUDA bug,
  not a CPU bug. Use `scripts/test_tq3_k256_c_vs_python.py` first to
  re-validate the CPU side after any change.
- The slot-cap patch on `tools/server/server-context.cpp` is needed
  for any NIAH testing past `n_ctx_train`. Re-apply after `git pull`
  on llama.cpp.
- `bin/zenith` does NOT cd into the repo before exec'ing the harness
  — keep it that way (preserves user's cwd for `.zenithrc` lookup).
