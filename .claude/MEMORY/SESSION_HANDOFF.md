# Session Handoff — 2026-04-07 (Session 6, full)

## Goal

Started narrow: user asked to push 2 unpushed commits and then work through 4 follow-ups from the Session 5 Next Steps list — wire up `agents/config.py` (item 2), expand React/security training data (item 3), build specialist hot-swap infrastructure (item 4), write an end-to-end harness smoke test (item 5). Explicit instruction: "no agents" (do the work directly, not via spawned subagents).

Session expanded organically from there:
- Added Gemma 4 E4B as a validated alternative base model (user saw it on Hugging Face and asked about it)
- Built a full needle-in-haystack test harness (`scripts/needle_test.py`) with single/multi/distractor modes
- Patched llama.cpp locally to enable 256K context testing past trained max
- Ran Qwen-vs-Gemma base-model A/B eval and NIAH comparison at 256K
- Fixed two stacked bugs in the context-limit wiring that made the per-model compaction limits effectively dead code
- Bumped `CLAW_CTX` default from 65536 to 262144 (full 256K) after VRAM and NIAH validation
- Ran `/update` at the end to rewrite `CLAUDE.md` + `architecture.md` + `training.md` capturing everything above

## Completed

### Early: items 2/3/4/5 from prior handoff

**Pushed the 2 unpushed commits from Session 5** (`d48e959`, `2e3de0e`) to `origin/feature/multi-agent-qwen`.

**Item 2 — Wired up `agents/config.py`** (~30 → 61 lines):
- Added explicit `ENV_VARS` registry mapping config keys to `CLAW_*` names so historical conventions (`CLAW_CTX`, `CLAW_AUTO_COMPACT_TOKENS`) stay aligned rather than auto-prefixing
- Added `INT_KEYS` set for proper typing on numeric env vars
- Fixed `bin/claw` to expand `~` in the `--gguf` path and to NOT `cd` into the repo root before launching the harness (previously broke `.clawrc` / CLAUDE.md auto-discovery from arbitrary cwd)
- Wired `load_config()` into `agents/harness.py:main()` — previously defined but never called
- Added `--ctx-size` CLI flag to harness, plumbed through `Harness(ctx_size=...)`
- Added `bin/claw --gguf PATH` launcher flag (must be first arg), supports `~` expansion
- Verified end-to-end with `.clawrc` file in a temp dir showing the effort override took effect

**Item 5 — Smoke test suite** (`scripts/smoke_test_harness.py`, 7 tests):
- 5 fast command/config tests (command battery, CLAW_EFFORT env, `.clawrc` file pickup, CLI flag override, session save/list)
- 2 slower model tests (streaming display single-print regression check, tool-call with `read_file`)
- Fresh tmpdir per test isolates `.claw_sessions` per run
- `--quick` flag skips the model-based tests
- All 7 pass at session start; re-ran multiple times during refactors to catch regressions

**Item 3 — Training data expansion** (`agents/distill/data/coding_reasoning_claude.jsonl`, 488 → 507):
- New generator: `scripts/generate_react_security_examples.py` (~700 lines of Python containing the example corpus)
- 19 hand-written examples: 11 React (stale closure in setInterval, RSC boundaries, Rules of Hooks, custom hook extraction, Context perf, useTransition, Zustand vs alternatives, react-hook-form + zod, error boundaries + async, key prop gotchas) + 8 security (SSRF with `ipaddress` check, XSS via `dangerouslySetInnerHTML`/DOMPurify, IDOR / `FOR UPDATE` queries, rate limiting layers, password reset with token hashing + session invalidation, argon2 vs SHA-256 rationale, JWT validation checklist, file upload vulnerabilities, CSP basics)
- Each example follows the existing JSONL schema (`{messages: [system, user, assistant]}`) with `<think>` blocks in the assistant turn
- Validation: JSON parseable, structural invariants checked (3 messages, role order, think block present, min length)
- Ran `filter_reasoning --merge` to rebuild `claude_reasoning.jsonl` (1,320 → 1,339)

**Item 4 — Hot-swap infrastructure** (`agents/model_swap.py`, 407 lines, new):
- `LlamaServerManager` class: owns subprocess lifecycle, supports adopting externally-started servers via `/props` query
- `stop_any()` finds the listening PID via `/proc/net/tcp` + socket-inode walk in `/proc/*/fd` when we don't have the Popen handle (for killing servers started by `bin/claw` or a previous session)
- `swap()` is a no-op when the target path resolves to the same inode (uses `Path.resolve()` for comparison). **Important**: symlinks collapse through `.resolve()`, so symlink-based swap tests fall through to the no-op branch. Use `os.link()` (hard link) to get two independent canonical paths pointing at the same bytes.
- `discover_specialist_models()` scans `~/models/` for domain-named GGUFs (patterns: `specialist-<domain>*.gguf`, `Qwen*<domain>*.gguf`, `<domain>*.gguf`), prefers Q5_K_M
- `detect_llamacpp_model()` helper (moved to `agent.py` later in the session) queries `/props` for `model_path`
- `SpecialistCoordinator` rewrite (76 → 245 lines): auto-detects hot-swap eligibility (llama-server running + specialist GGUFs on disk), else falls back to the Ollama multi-model path. Overrides `run()` with `_run_hot_swap()` that swaps to the specialist before each delegate call and swaps back to base for the leader's next turn.
- Integration test (`scripts/test_model_swap.py`) does real kill+restart cycles via hard links, verified the full forward-and-back swap cycle works

### Gemma 4 E4B discovery and evaluation

User asked about Gemma 4 26B-A4B (MoE) while discussing future model options. Initial analysis concluded it wouldn't fit 8 GB VRAM (smallest quant IQ2_XXS is 9.88 GB) but investigation turned up the **E4B variant** on Hugging Face (`google/gemma-4-E4B-it`, 473K downloads, stock). Downloaded Q5_K_M (5.48 GB).

**Architecture highlights** (from `config.json`):
- 42 layers, hidden_size 2560, num_kv_heads **2** (aggressive GQA)
- `head_dim` 256, `sliding_window` 512, `max_position_embeddings` 131072 (128K trained)
- Gemma's Per-Layer Embeddings (PLE) + AltUp + Laurel efficiency tricks — the "E" in E4B stands for "Effective"
- Multimodal (vision projector `mmproj-*.gguf` available, not tested)

**5-prompt A/B eval vs fine-tuned Qwen 3.5 4B** (`scripts/eval_base_models.py`, report at `.claude/MEMORY/evals/2026-04-07_qwen4b_vs_gemma4_e4b.md`):
- Gemma won 5/0. Prompts: race condition, OOMKilled, architecture design, React, security
- Qwen failures were **correctness bugs**, not style issues: fabricated `process.on('beforeOOM')` Node.js API; SQL queue with no `FOR UPDATE SKIP LOCKED` + async/sync mixing; regex-on-hostname SSRF check that DNS rebinding bypasses; broken `useEffect` chain that only handled the first fetch
- Conclusion: **fine-tuning Qwen could not add correctness the base doesn't have**. Gemma's base knowledge wins over Qwen's style fine-tuning.
- Throughput: Gemma 47.5 tok/s vs Qwen 52.6 tok/s (Gemma 10% slower, 2.3× more verbose)

### Needle-in-haystack validation (the big effort)

User asked "don't all LLMs perform great in the first 30% context?" — prompted a deep dive into effective context vs advertised context. Built `scripts/needle_test.py` with 3 modes:

**Mode 1: `single` (one needle, one retrieval)**
**Mode 2: `multi` (5 needles at evenly-spaced depths, must find ALL)**
**Mode 3: `distractor` (1 PRIMARY + N decoys, must return primary only)**

**Test 1 (128K ctx)**: `2026-04-07_gemma4_e4b_needle.md` — Gemma 21/21 perfect recall at 4K–100K haystacks.

**llama.cpp patch** (`~/llama.cpp/tools/server/server-context.cpp:763-766`, outside repo): llama-server silently caps each slot's context at `n_ctx_train` regardless of `--ctx-size`. To test past Gemma's 128K trained max, I patched out the line `n_ctx_slot = n_ctx_train;` (commented out, kept the warning log). **One-line change, NOT upstreamed, must re-apply after `git pull` on llama.cpp**. Incremental rebuild took ~30s (just `server-context.cpp.o` + linking llama-server). Verified slot now allocates the full 262144.

**Test 2 (256K ctx, real range)** — after patch:
- Gemma 4 E4B single-needle: 21/21 at 4K–220K (68% past trained max via raw RoPE extrapolation — stayed perfect)
- Gemma 4 E4B multi-needle: 6/7 (fail only at 220K 4/5)
- Gemma 4 E4B distractor: 7/7 (after fair-scoring fix)
- Qwen 3.5 4B single-needle: 21/21 at same sizes
- Qwen 3.5 4B multi-needle: 5/7 (fail at 180K 4/5 AND 220K 3/5 **+ hallucinated one** — returned `14223-AZURE-MARTEN` when expected `14223-CRIMSON-EAGLE`, correct number, invented suffix)
- Qwen 3.5 4B distractor: 5/7 **non-monotonic** — passes 4K/32K, **fails 64K and 100K**, passes 130K/180K/220K. The 64/100K failures are classic "lost in the middle" attention dilution. Gemma's sliding-window attention protects against this.

**Totals: Gemma 34/35 (97%), Qwen 31/35 (89%).**

All 9 eval reports saved to `.claude/MEMORY/evals/`:
- `2026-04-07_gemma4_e4b_needle.md` (initial 128K)
- `2026-04-07_gemma4_e4b_needle_256k_single.md`
- `2026-04-07_gemma4_e4b_needle_256k_multi.md`
- `2026-04-07_gemma4_e4b_needle_256k_distractor.md`
- `2026-04-07_qwen4b_needle_256k_single.md`
- `2026-04-07_qwen4b_needle_256k_multi.md`
- `2026-04-07_qwen4b_needle_256k_distractor.md`
- `2026-04-07_qwen4b_vs_gemma4_e4b.md` (5-prompt coding eval)
- `2026-04-07_summary_needle_comparison.md` (authoritative cross-model summary)

### Context-limit wiring bug fixes (the tricky part)

Setting up the per-model compaction limits in `compact.py` (Gemma 200K, Qwen 130K, llamacpp 65K fallback) revealed **two stacked bugs** that made the new limits effectively dead code:

**Bug 1**: `agents/agent.py:174` passed the literal string `"llamacpp"` to `detect_context_limit()` when backend was llama.cpp, instead of the actual GGUF name. So per-model lookups in `MODEL_CONTEXT_LIMITS` never fired for any llama.cpp session — always matched the generic 65K fallback.

**Bug 2**: `agents/harness.py:_spawn()` always passed `max_context_tokens=self.ctx_size` to `Agent()`. Since `ctx_size` was always a truthy int from config, the `max_context_tokens or detect_context_limit(...)` expression in Agent.__init__ short-circuited — `detect_context_limit` was never even called.

**Fix** (stacked):
1. `agent.py`: added `detect_llamacpp_model()` helper that queries `/props` for `model_path`. Rewrote `Agent.__init__`'s context-limit resolution to: (a) honor an explicit `max_context_tokens` caller override, (b) for ollama backend use the model name, (c) for llamacpp backend query `/props` and pass the real GGUF path to `detect_context_limit()`.
2. `harness.py`: `Harness.__init__` queries `/props` once at construction and caches `self._loaded_llamacpp_model`. New `_compute_compact_threshold()` method returns `min(per-GGUF limit, int(ctx_size * 0.85))` — the 85% safe-ctx margin leaves headroom for the active turn's response before hitting the llama-server hard cap.
3. `/swap` command handler now refreshes `_loaded_llamacpp_model` and recomputes the threshold for every agent after a successful swap (prints the new threshold to console).
4. `/backend llamacpp` handler does the same.

**Verified live**: Harness constructed with `ctx_size=262144` and Gemma loaded → compact threshold = 200,000. Same with ctx_size=131072 → threshold = 111,411 (85% of ctx wins because it's smaller than the 200K model limit). Swap from Gemma to Qwen mid-session correctly recomputed threshold from 200K → 130K for all 3 agents.

### `CLAW_CTX` default bump and cross-file alignment

User said "start at full threshold":
- `bin/claw:41` — `CTX="${CLAW_CTX:-65536}"` → `CTX="${CLAW_CTX:-262144}"`
- `agents/config.py:14` — `"ctx_size": 65536` → `"ctx_size": 262144`
- Both files now agree on the 256K default

VRAM verified: Gemma 4 E4B at 256K uses ~6.7 GB; Qwen 3.5 4B at 256K uses ~7.3 GB. Both fit the 8 GB 4070 Laptop with ~1 GB headroom.

### `/update` run at end of session

Ran `/update` with the full `Audit → propose → implement` flow. User approved all proposed changes with "implement all":
- `.claude/CLAUDE.md`: 239 → 277 lines, 12 section rewrites + new "Needle-in-Haystack Validation" section citing the eval reports
- `.claude/rules/architecture.md`: 69 → 80 lines, 7 section updates + 3 new invariant rules (Agent context limit lookup, Harness loaded-model cache, 85% safe-ctx compaction margin), all cited to "session 2026-04-07"
- `.claude/rules/training.md`: 57 → 61 lines, 5 section updates + 3 new Known Issues (llama.cpp slot context cap patch, Gemma 4 GGUF rope-scaling metadata override, llama-server `--parallel` default)
- `.claude/rules/orchestration.md` and `.claude/rules/vdd.md`: no drift, not touched
- All files well under 500-line hard limit after updates
- `PYTHONPATH=. python3 -c "import agents.harness"` clean after every edit batch

## In Progress

### Uncommitted (large pile — organized below for commit groups)

**Group 1: Hot-swap infrastructure** (logical commit: "Add llama.cpp hot-swap via LlamaServerManager")
- `agents/model_swap.py` (new, 407 lines)
- `agents/specialist_coordinator.py` (76 → 245 lines, rewritten for auto-selecting hot-swap mode)
- `scripts/test_model_swap.py` (new, integration test with hard-link trick)

**Group 2: Config + ctx wiring** (logical commit: "Wire load_config + fix per-model compaction thresholds")
- `agents/config.py` (wired up, explicit env var registry, ctx_size default 262144)
- `agents/agent.py` (`detect_llamacpp_model` helper + fixed `Agent.__init__` lookup)
- `agents/harness.py` (`_compute_compact_threshold`, `/swap` command, cached `_loaded_llamacpp_model`, refresh on `/swap` + `/backend`)
- `agents/compact.py` (`MODEL_CONTEXT_LIMITS` per-GGUF entries + NIAH-validated comment)
- `bin/claw` (`--gguf` flag, `CLAW_CTX=262144` default, removed `cd`, updated header comment)

**Group 3: Needle-in-haystack validation** (logical commit: "Add NIAH test tooling + eval reports")
- `scripts/needle_test.py` (new, 3 modes: single/multi/distractor)
- `scripts/eval_base_models.py` (new, A/B base model evaluator)
- `scripts/smoke_test_harness.py` (new, 7-test harness smoke suite)
- `.claude/MEMORY/evals/` (9 reports, all committed or not committed based on `.gitignore`?) — verify

**Group 4: Training data expansion** (logical commit: "Expand training data with React/security examples")
- `agents/distill/data/coding_reasoning_claude.jsonl` (+19 examples, 488 → 507 lines)
- `scripts/generate_react_security_examples.py` (new, the generator)

**Group 5: Doc updates from `/update`** (logical commit: "Update CLAUDE.md + rules for Gemma/hot-swap/NIAH")
- `.claude/CLAUDE.md`
- `.claude/rules/architecture.md`
- `.claude/rules/training.md`

**Note on `.claude/MEMORY/SESSION_HANDOFF.md`**: shows 297 lines diff, but this is from pre-session state (modified before this session started). Not part of this session's work — will be overwritten by this handoff write.

### Outside the repo (re-apply after llama.cpp pull)

**llama.cpp local patch** at `~/llama.cpp/tools/server/server-context.cpp:763-766`:

```cpp
int n_ctx_slot = llama_n_ctx_seq(ctx);
if (n_ctx_slot > n_ctx_train) {
    // PATCHED (claw-code 2026-04-07): allow slot context past trained
    // limit so users can opt into extrapolation via --ctx-size + RoPE
    // scaling. Original behavior capped to n_ctx_train silently.
    SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - allowing extrapolation (patched)\n", n_ctx_slot, n_ctx_train);
    // n_ctx_slot = n_ctx_train;  // <-- patched out
}
```

Rebuild after re-applying: `cd ~/llama.cpp && cmake --build build --target llama-server -j$(nproc)` (~30 s incremental).

## Next Steps

1. **Commit the uncommitted pile in logical groups** (see Groups 1-5 above). Push to `origin/feature/multi-agent-qwen`. This session made ~720 insertions across 11 files, none committed yet.

2. **Flip the `bin/claw` default model to Gemma 4 E4B** (optional, user asked about this but didn't explicitly switch). One line: `bin/claw:33`'s fallback from `$HOME/models/Qwen3.5-4B.Q5_K_M.gguf` → `$HOME/models/gemma-4-E4B-it-Q5_K_M.gguf`. The fallback only fires when no `--gguf` and no `CLAW_MODEL` are set, so changing it is low-risk. User can still `CLAW_MODEL=$HOME/models/Qwen3.5-4B.Q5_K_M.gguf claw` to get Qwen back.

3. **Fix `bin/claw` to pass `--parallel 1` to llama-server** (known bug noted in `.claude/rules/training.md` Known Issues). Without it, llama-server splits `CLAW_CTX` across 4 default slots, so each slot gets only 65K at the new 262144 default. Add the flag to the `llama-server` invocation around `bin/claw:73-80`.

4. **Train Gemma 4 E4B on the distillation dataset**. This is the highest-ROI work for the project. Pipeline:
   - Update `agents/distill/config.py` `STUDENT_BASE` from `"Qwen/Qwen3.5-0.8B"` to `"google/gemma-4-E4B-it"` (or make it configurable)
   - Adapt `train_4b_cloud.py` for Gemma's architecture (AltUp/Laurel/PLE target modules may differ from Qwen's standard `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`)
   - Run on RunPod A6000 spot (~$0.25/hr, ~30-40 min per run, ~$0.15-0.20 per training run)
   - **Re-run needle test on fine-tuned Gemma after training** — catastrophic forgetting of long-context abilities is a real risk when fine-tuning on short examples. If recall drops from 200K to <100K, abandon that fine-tune and add long-context examples to the training mix.
   - Re-run the 5-prompt coding eval to see if fine-tuned Gemma beats stock Gemma

5. **(Optional) Run multi-needle with verbose logging to identify which specific needle Gemma missed at 220K**. The test currently only logs `found_codes: int`, not the identity. ~4 minutes of compute. Doesn't change the 200K threshold recommendation but would be informative.

6. **Harness improvements backlog**:
   - The `/swap` command uses substring matching; might want exact-match mode too for scripting
   - Add `/swap` to the smoke test suite to cover the model-change path
   - Consider showing the compaction threshold in the harness startup banner so users can see it without a command

7. **Specialist training once base is ready**. Hot-swap infrastructure is built and tested; just needs specialist GGUFs on disk at `~/models/specialist-<domain>*.gguf`. `SpecialistCoordinator` auto-detects them and enables hot-swap routing.

## Key Context

### The most important correction from this session

**Qwen 3.5 4B is trained at 256K native context, NOT 32K.** I told the user 32K earlier in the session based on mis-remembered notes; when I actually fetched the GGUF metadata via `/props` later, it returned `n_ctx_train = 262144`. This changes the story: Qwen and Gemma have the same trained max, and the only long-context differentiator is architecture (Gemma's sliding-window + GQA vs Qwen's full attention). The earlier claim "Gemma has more trained context" was wrong. The actual advantage is **Gemma's sliding-window attention protects against 'lost in the middle' failures** that Qwen exhibits at 64K/100K on distractor tests.

### The context-limit wiring was two bugs, not one

I initially thought it was one bug (pass the real GGUF name instead of "llamacpp"). After fixing that I realized the harness was still overriding `max_context_tokens` unconditionally, so the fixed lookup never even ran. Two stacked bugs: caller always passed truthy value, so the callee's fallback path was dead. **Lesson for future debugging**: when you fix a code path and it still doesn't work, check whether you're actually REACHING it with a non-short-circuited caller.

### The llama.cpp slot cap is invisible unless you grep

When I passed `--ctx-size 262144` and got `ctx: 131072` back from `/props`, I burned a lot of time trying different flag combinations (`--rope-scaling yarn`, `--rope-scale 2.0`, `--yarn-orig-ctx`, `--parallel 1`) before I noticed the warning line `srv load_model: the slot context (262144) exceeds the training context of the model (131072) - capping`. The cap is hardcoded in `server-context.cpp:765` and silently truncates unless you patch it. **If `/props` returns a smaller ctx than `--ctx-size` passed**, grep `/tmp/llama-server.log` for "capping" before chasing flags.

### Gemma 4 GGUF metadata override (YaRN flags don't work)

Gemma 4 E4B's GGUF bakes in `rope scaling = linear`. Passing `--rope-scaling yarn` is silently ignored — you can see `print_info: rope scaling = linear` in the log regardless of CLI args. Extrapolation past 128K is raw linear RoPE (which somehow works up to 220K on single-needle). Don't trust YaRN flags on Gemma 4; they're pure overhead (allocate buffers that don't get used — removing the flags saved ~560 MiB VRAM).

### `llama-server --parallel` default is 4

Without `--parallel 1`, llama-server splits `--ctx-size` across 4 slots, so each slot gets `ctx_size / 4`. Since the harness is always single-user, this is pure waste — and it made me chase phantom context caps early in the session before I found the `--parallel 1` flag. `bin/claw` does not currently pass this flag; the next step (fix #3 above) adds it.

### Hard links, not symlinks, for swap testing

My first swap integration test used `symlink.symlink_to(original)` to create a "different path" for the same GGUF. It didn't exercise the kill+restart path because `Path.resolve()` collapses the symlink to its target, and `LlamaServerManager.swap()` uses `.resolve()` for target comparison — so it became a no-op. Switched to `os.link()` (hard link) which gives two independent canonical paths pointing at the same inode. `.resolve()` doesn't collapse hard links, so the swap sees them as different and does a real kill+restart. Integration test then worked: 11.4s forward, 5.3s reverse (page-cache warmed).

### Distractor test scoring needed fair reasoning vs content split

First run of distractor test showed "PARTIAL" at 4K/32K because I was combining `reasoning_content` + `content` in the leak check. The model was correctly identifying the primary but MENTIONING distractors during its `<think>` block (which is correct reasoning behavior — "I see codes A, B, C, D, E; the primary is A"). Fixed by only checking `content` for distractor leaks, while still allowing primary to be found in either field. Also handled the edge case where Gemma at small context puts its entire response in `reasoning_content` with empty `content`.

### Failed / abandoned approaches

1. **YaRN scaling to extend Gemma past 128K trained max** — doesn't work (metadata override), abandoned in favor of raw extrapolation
2. **Symlink-based swap test** — collapsed through `.resolve()`, replaced with hard links
3. **Distractor test criteria "any mention is a leak"** — too strict, gave false PARTIAL at small context, fixed to check only final `content`
4. **"Park the old KV cache in RAM on swap"** — discussed with user but not implemented; would need a llama.cpp C++ patch (~200 lines, not worth the scope for this project)
5. **Bumping compaction limit without fixing the wiring** — I set Gemma's compaction limit to 180K early in the session but then found the wiring bugs meant it never fired. Had to fix the wiring before the limit did anything useful.

### Hardware / serving snapshot at session end

- Laptop: Acer Nitro AN17-42, RTX 4070 Laptop GPU 8 GB VRAM, 32 GB DDR5
- llama.cpp: patched local build, HEAD at `69c28f1` (currently 27 commits ahead of `761797f` pre-session). Patched once at `tools/server/server-context.cpp:765`.
- Models on disk at `~/models/`: `Qwen3.5-4B.Q5_K_M.gguf` (2.9 GB), `gemma-4-E4B-it-Q5_K_M.gguf` (5.48 GB, new this session)
- Ollama pulled: `qwen3.5:4b`, `qwen3.5:9b`, `qwen3:0.6b/4b/8b`, `qwen4b-fast:latest`, `qwen9b-fast:latest`, `reasoning-base:latest` (note: `qwen3.5:0.8b` is NOT currently pulled — older doc claims corrected this session)
- llama-server at session end: running Qwen 3.5 4B at 262144 ctx (from the post-`/update` sanity test). If you want Gemma, `/swap gemma` or restart with `claw --gguf ~/models/gemma-4-E4B-it-Q5_K_M.gguf`.
- VRAM at session end: ~7.5 GB used (Qwen at 256K + baseline)

## Files in Project

### New this session

- `agents/model_swap.py` — `LlamaServerManager` + `discover_specialist_models()` + `detect_llamacpp_model()`. Adopts external servers via `/props`, kills via `/proc/net/tcp` PID lookup, swap is no-op if target already loaded (`.resolve()` comparison).
- `scripts/needle_test.py` — NIAH test runner with 3 modes (`--mode single/multi/distractor`). Uses `agents/distill/data/coding_reasoning_claude.jsonl` as haystack filler. Outputs markdown report with recall grid + timing.
- `scripts/eval_base_models.py` — A/B evaluator for two GGUFs. Swaps between them via `LlamaServerManager`, runs 5 fixed prompts (race_condition, oomkilled, architecture, react, security), writes side-by-side markdown.
- `scripts/smoke_test_harness.py` — 7-test harness smoke suite. `--quick` flag for fast tier. Runs `claw` as subprocess, verifies command dispatch + config loading + model+tools end-to-end.
- `scripts/test_model_swap.py` — integration test for `LlamaServerManager`. Uses hard links to force real kill+restart cycles when only one GGUF is on disk.
- `scripts/generate_react_security_examples.py` — one-shot generator that produces 19 React + security training examples in JSONL and appends to `coding_reasoning_claude.jsonl`. Contains the example corpus inline.
- `.claude/MEMORY/evals/2026-04-07_*.md` — 9 eval reports (see list in Completed section). `_summary_needle_comparison.md` is the authoritative cross-model summary.

### Modified this session

- `agents/agent.py` (449 → 520 lines) — added `detect_llamacpp_model()` helper at top level, rewrote `Agent.__init__`'s context-limit resolution to query `/props` for real GGUF name when backend is llamacpp
- `agents/harness.py` (536 → 664 lines) — `/swap` command, `_compute_compact_threshold()`, cached `_loaded_llamacpp_model`, refresh on `/swap` + `/backend`, `--ctx-size` arg, `Harness(ctx_size=...)` parameter
- `agents/compact.py` (215 → 244 lines) — `MODEL_CONTEXT_LIMITS` per-GGUF entries: `gemma-4-e4b` / `gemma-4-E4B` → 200000, `qwen3.5-4b` / `Qwen3.5-4B` → 130000. Extended `detect_context_limit()` docstring with match-order note.
- `agents/config.py` (30 → 61 lines) — explicit `ENV_VARS` registry, `INT_KEYS` for numeric coercion, `ctx_size` default 262144, `effort` key added to defaults
- `agents/specialist_coordinator.py` (76 → 245 lines) — hot-swap mode auto-detection via `discover_specialist_models()`, `_run_hot_swap()` override, `_swap_to()` helper, `specialist_status` reporting mode
- `agents/distill/data/coding_reasoning_claude.jsonl` (488 → 507 lines) — +19 hand-written React/security examples (2026-04-07)
- `bin/claw` (stdin fix earlier + this session's `--gguf` flag + `CLAW_CTX=262144` default + removed `cd $SCRIPT_DIR` before exec + updated header comment block)
- `.claude/CLAUDE.md` (239 → 277 lines) — `/update` rewrite
- `.claude/rules/architecture.md` (69 → 80 lines) — `/update` rewrite with 3 new invariants
- `.claude/rules/training.md` (57 → 61 lines) — `/update` rewrite with 3 new Known Issues

### Reference (not touched)

- `agents/coordinator.py`, `agents/swarm.py`, `agents/permissions.py`, `agents/session.py`, `agents/history.py`, `agents/tools.py`, `agents/example.py` — unchanged
- `agents/distill/*.py` — pipeline scripts unchanged (didn't need touching for the React/security data addition)
- `.claude/rules/orchestration.md`, `.claude/rules/vdd.md` — no drift
