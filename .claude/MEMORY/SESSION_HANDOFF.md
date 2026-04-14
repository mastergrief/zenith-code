# Session Handoff — 2026-04-14 (Session 25)

## Goal

Two tracks in parallel:

1. **Push HRM (Hierarchical Reasoning Model) past the 51% per-token plateau** session 24 left it at, and ship a working math HRM end-to-end.
2. **Implement the Percepta March-2026 LLM-Computer research** (compile programs to transformer weights, HullKVCache, 2D heads) as a CPU-only prototype that pairs with HRM under the user's CRLM thesis: *intelligence partitioned into structure (learned) + values (compiled, exact)*.

User directives that shaped the session: "max the intelligence, scale the knowledge", "i want mega performance from tiny models", "scratchpad 1 and 4 hybrid", "lets just go for 3 [integration] straight away", "embed into HRM decoder", "reverse engineer hrm into llm_computer", "yes lets implement [scratchpad + structure-only]". No subagents per `feedback_no_agents`.

## Completed

### HRM training journey — landed sweet spot at 48K params

Four rounds in `calm/hrm/`:

| Round | Config | Params | Train | Per-token | Full-expr (verified) |
|---|---|---|---|---|---|
| 1a | enc-dec + digit-reversal, h=64 | 244K | 8 min | 51% (plateau) | ~15-25% inferred |
| 1c | scratchpad + `<call>/<end_call>` delegation, h=64 | 245K | 15 min | 94% | 43% |
| 1d | 1c + place-value decomp (mult only) | 245K | 16 min | 94% | 37% (worse — paren noise) |
| **1e** | **`--structure-only`, h=32, 1 layer each** | **48K** | **145s** | **99.7%** | **96.7%** |

**Round 1e is the production checkpoint**: `calm/hrm/checkpoints/math_structure_best.pt`. 30/30 held-out (seed 9999); 29/30 had structurally-matched-input HRM emission. Lone failure was `gcd(39, 39) → gcd(69, 39)` (one mis-copied digit). 5/5 held-out pass without verified mode being needed for that case because LLM-Computer's interpreter handles values regardless.

The decisive insight: scratchpad-with-intermediate-values made the model try to memorize arithmetic it couldn't see enough samples of. **Removing the value-memorization burden** (target = `problem + = + <eos>`, values handled by the LLM-Computer interpreter) collapsed the learning problem to "echo + terminate", which a 48K-param model nails in 145 seconds.

Code shipped to `calm/hrm/`:
- `model.py` — `HRMSeq2Seq` (encoder=bidirectional + nested L/H loops, decoder=causal + cross-attn, NO recurrence in decoder), legacy `HRM` kept; `Attention` got `is_causal` ctor flag; new `CrossAttention`, `DecoderBlock`, `HRMEncoder`, `HRMDecoder` classes.
- `data.py` — `MathSeq2SeqDataset` with three target modes: `--scratchpad` (full reduction trace), `--structure-only` (echo + `=`), default (raw answer + reverse digits). `<call>`/`<end_call>` added to `_SPECIAL`. Trace generators (`_trace_expression`, `_trace_function_only`) for scratchpad mode. Helpers `_maybe_reverse_digits` / `_unreverse_if_numeric`. `tokenize_trace` / `detokenize_trace` for `<call>`-aware tokenization.
- `train.py` — `HRMSeq2SeqTrainer` with `scratchpad`, `structure_only`, `reverse_digits` flags. Checkpoint filenames split: `math_seq2seq_best.pt`, `math_scratchpad_best.pt`, `math_structure_best.pt`. CLI flags: `--seq2seq --structure-only --scratchpad --no-reverse-digits --hidden --num-heads --l-layers --h-layers --dec-layers --max-enc --max-dec`.
- `inference.py` — `HRMSeq2SeqReasoner` with mode-aware decode loop. In scratchpad mode, intercepts `<end_call>` and routes via `safe_eval`, injects result tokens, resumes. `_extract_final_answer()` handles paren-heavy traces.
- `__init__.py` — exports updated.

### LLM-Computer prototype shipped (`calm/llm_computer/`)

CPU-only standalone subpackage. All tests pass.

- `model.py` — `Small2DTransformer` (vanilla PyTorch, `d_head=2`, `use_hard_max` flag). Standard `nn.MultiheadAttention`-shaped forward pass. Hand-max attention zeroed via `torch.argmax + scatter` instead of softmax.
- `hull_cache.py` — `HullKVCache` (online 2D convex hull, Andrew's monotone chain, separate upper+lower hulls). Supporting-point query in O(h). **108× speedup vs linear scan at N=2K** (verified). Sign convention: upper hull removes triple when `cross >= 0`, lower when `cross <= 0` (Andrew's standard, NOT my initial flipped version).
- `gate_graph.py` — IR with two node families:
  - Compute: `Const`, `BinOp` (op = "add"/"sub"/"mul"/"div"), `Delegate` (fn_name + args), `Result` (named output)
  - Hardware: `TokenInput`, `TokenOutput` (for the direct compile path)
- `compile.py` — minimal compile_graph for identity + linear-head programs (used by `add_one`-style demos). LookUp/ReGLU as first-class nodes NOT YET implemented; everything beyond identity is hand-wired in `programs/`.
- `parse.py` — `parse_expression()` via Python `ast.parse` → `GateGraph`. `extract_problem_from_trace()` strips `<call>...<end_call>` markers and returns pre-`=` segment.
- `interpret.py` — topo-walks `GateGraph`; `Delegate` → `safe_eval`. Float→int collapse if integral.
- `programs/`:
  - `add_one.py` (1,280 params, 8/8 pass): tok=identity, head=cyclic shift, attn+FFN zeroed.
  - `copy_past.py` (2,560 params, 4/4 pass): zero-q/k tie-break selects pos 0; upper-half v copies tok(input[0]) into upper-half residual; head reads upper half.
  - `increment_counter.py` (2,176 params, all pass): position embedding pos[p][V+p] = 1, head reads upper-half. Input-invariant (verified on `[3,7,1,4]` vs `[0,0,0,0]`).
  - `threshold.py` (216 params, 8/8 pass): FFN computes `1[input >= T] = ReLU(input-(T-1)) - ReLU(input-T)` via 2 ReGLU neurons. Bias channel via constant-1 in pos embedding dim 1 (since `nn.Linear(bias=False)`).
- `tests/test_hull_cache.py` — parabolic-keys exact lookup (100/100), random-query match-vs-linear-scan (correctness), wall-clock speedup gate (≥3×, hit 108×).
- `tests/test_parse_interpret.py` — round-trip parse+interpret matches safe_eval on 19 expressions; trace extraction strips `<call>` markers; end-to-end trace → answer.

### Eval pipeline updated (`scripts/eval_hrm_math.py`)

- `--verified` flag: HRM emits trace, `_verified_answer()` extracts the pre-`=` segment, parses it, interprets via LLM-Computer. Reports HRM emission used / structurally-matched-input.
- `_hrm_raw_emit()` exposes raw decoder output for inspecting what HRM actually produced.
- Smoke cases unchanged (5 canonical: 17×23, 347×289, gcd(48,180), factorial(7), fibonacci(12)).

### Documentation updated

- `.claude/CLAUDE.md` (395 → 453 lines): "Three active systems" → "Four"; new `## HRM + LLM-Computer` section before Distillation Pipeline (sweet spot, primitives, pipeline, training-track table); CALM section gets adjacent-track sentence pointing at `calm/llm_computer/`.
- `.claude/rules/architecture.md` (102 → 140 lines): file-org additions for `calm/hrm/` + `calm/llm_computer/`; new `## HRM + LLM-Computer Architecture` section after CALM (HRMSeq2Seq, three target modes, Small2DTransformer, HullKVCache, gate-graph IR, parser/interpreter, convergence pipeline).
- `.claude/rules/training.md` (146 → 189 lines): new `## HRM Training (CRLM workflow)` section before Dataset Quality (don't memorize values, sweet-spot config, per-token vs full-expression, plateau pattern, when to scale, pitfalls).
- `.claude/rules/workflow.md` (263 → 298 lines): extended monitor case study with the early-stop-at-99.7% example; new `## Sweet-spot search for tiny models` section (start downward, structure-only as the unlock, per-token misleading on trace targets).
- `.claude/rules/workflow.md` ALSO got a `## Long-running training supervision` section earlier in the session (Monitor + filtered tail pattern with `setsid` + `disown` + `< /dev/null` gotchas).

### Memory file added

- `feedback_training_monitor_loop.md` — initially saved, then **deleted** when user reminded me project-rules belong in `.claude/rules/`, not personal-preference memory. Pattern moved to `workflow.md`.

### Other commits

None — everything sits as uncommitted changes for the user to commit at their discretion. `feature/multi-agent-qwen` branch unchanged from session 24's last commit (`91721f2`).

### Reading materials added by user

- `RESEARCH/00_INDEX.md`, `01_LLM_Computer_Overview.md`, `02_Fast_Attention_2D_Heads.md`, `03_Compiling_Programs_to_Weights.md` — Percepta March-2026 paper rewrites. The LLM-Computer prototype is the implementation of these. Worth re-reading before extending the prototype.

## In Progress

### Round 1e training is finished and shipped

Round 1e checkpoint at `calm/hrm/checkpoints/math_structure_best.pt` is the production HRM. No background training is running.

### Smoke-case 3-digit failures are out-of-distribution

`347 * 289` and `gcd(48, 180)` fail in eval because `MathDataGenerator` caps operands at 2 digits. **Fix**: bump `_arithmetic_simple` and friends in `calm/hrm/data.py` to allow 3-digit operands, regenerate, retrain. Trivial 5-min experiment. NOT done because the architectural point (sweet spot found) was the priority.

### Full LLM-Computer compiler not yet done

`compile.py` only handles identity + linear-head programs. The four primitive programs are hand-wired in `programs/*.py`. Promoting `LookUp` and `ReGLU` to first-class gate-graph nodes + writing a compiler that routes them to attention heads / FFN neurons is the next big code milestone (Round 4 Layer 2 in the plan file). User is fully aware.

### Unfinished from session 24

- Rust Ollama match-arm changes (`rust/crates/api/src/client.rs`, `providers/mod.rs`) sit modified from session 22, untouched this session.
- `calm/learned_patterns.jsonl` — frequency bumps from prior Gemma testing, not committed.

## Next Steps

Ordered by leverage:

1. **Push 3-digit data range** (5 min). Edit `calm/hrm/data.py:_arithmetic_simple`/multi/parens to allow operands up to `randint(1, 999)`. Retrain Round 1e config (`--structure-only --hidden 32 --epochs 100 --problems 2000`). Re-eval `scripts/eval_hrm_math.py --verified`. Should hit 100% smoke including `347 * 289`. Cheapest demo polish.

2. **Round 4 Layer 2 — promote `LookUp` and `ReGLU` to gate-graph nodes + write compiler** (~2-3 days). Currently the four primitive programs hand-write weights in Python. Goal: declare them as `GateGraph` instances and have `compile.py` produce the same weight tensors automatically. Then a 2-digit adder built compositionally (LookUp from past + ReGLU step indicators per output digit). Plan file `/home/gabe/.claude/plans/twinkling-soaring-dongarra.md` has the full design. Files to extend: `gate_graph.py`, `compile.py`. New tests in `tests/test_compile.py`.

3. **Commit Round 1e + LLM-Computer** (~10 min). All work is uncommitted. Suggested commit messages:
   - `hrm: Round 1e — structure-only loss + tiny model (48K params, 145s, 96.7%)` for `calm/hrm/`, `scripts/eval_hrm_math.py`
   - `calm: llm_computer prototype — Small2DTransformer + HullKVCache + 4 primitive programs` for `calm/llm_computer/`
   - `docs: session 25 — HRM sweet spot + LLM-Computer prototype` for `.claude/CLAUDE.md` + `.claude/rules/*.md`
   - User has NOT been prompted to commit — they direct that explicitly.

4. **Compile a real program** (`adder.py`) end-to-end via the new compiler (after #2). Two-digit `a + b` using LookUp + ReGLU step indicators per output bit. This is the proof of compositionality.

5. **Integration #3 — HRM emits gate-graph tokens directly** (~1 week). Currently the parser handles arithmetic strings (Python AST). For domains where HRM's value is structural decomposition (logic, planning), the HRM should emit gate-graph tokens directly. New tokenizer + serialization. Out of scope for this session; flagged in plan.

6. **HullKVCache true O(log n) query** — current implementation is O(h) hull-walk. Hull stays tiny (h=14 at N=2K) so it's already fast, but a proper ternary search on the sorted hull would shave constants. Low priority.

7. **2D-head HRM decoder retrofit** (Round 1e's untested hypothesis). Replace HRM decoder's `d_head=16` softmax attention with `d_head=2` hard-max + `HullKVCache`. Test whether structure-only training still works with the constrained head dim. Speculative — paper flags 2D-head training-at-scale as the decisive open question.

## Key Context

### CRLM thesis crystallized

**HRM provides STRUCTURE (learned, modest scale). LLM-Computer provides VALUES (compiled, exact).** HRM size scales with **problem-language complexity**, not problem-difficulty. `factorial(100)` is no harder for HRM than `factorial(2)` — both emit ~14 tokens of identical structure. The compute substrate handles all values regardless of difficulty. This is now the production architecture for math; future domains (logic, code, NL→math) reuse the same split.

### Per-token val_acc lies on trace targets

A scratchpad trace contains 60-70% trivially-predictable tokens (operators, parens, prefix copy). Per-token hits 94% while full-expression sits at 43%. **Always measure full-expression via `--verified` mode** before declaring a checkpoint shippable. Per-token is only useful for spotting regressions during training.

### Failed approaches (don't repeat)

- **Place-value decomposition for multiplication didn't help.** Round 1d traces (`25 * 88 = (25*80 + 25*8) = ...`) made things WORSE (43% → 37%) because longer traces give more chances to error and paren noise added structural complexity without capacity gain.
- **Reverse-digit answers (Abacus-style)** — Round 1a — capped at 51% per-token. Generalization was the bottleneck, not digit ordering.
- **Scaling hidden 64 → 128** at the same data — Round 1c-equivalent — moved val_acc from 56% to 56%. Capacity wasn't the bottleneck.
- **Cosine LR scheduled to 0 too early kills learning.** A 50-epoch run hit 73% capped (LR≈0 by epoch 25); same model at 500 epochs hit 99.7% by epoch 100. Always set `--epochs` generously, rely on `best_val_acc` checkpoint selection.
- **Trying to compile `safe_eval` directly into Small2DTransformer weights for a math demo** — for the math case this is degenerate (safe_eval works directly). The pipeline value is in non-math domains; math is the proof-of-mechanism.
- **HullKVCache cross-product sign** — initial implementation flipped Andrew's convention (upper removed `cross <= 0`, lower `cross >= 0`). Must be upper removes `cross >= 0`, lower removes `cross <= 0`. Tests caught this; corrected.

### Workflow patterns reinforced

- **Sweet-spot search**: start downward, scale only when the structurally-relevant gate fails. Section now in `workflow.md`.
- **Monitor + grep-filtered tail**: stream eval lines from detached training; intervene on plateau, ship at the right checkpoint (not the scheduled-end checkpoint). Section in `workflow.md`.
- **No subagents in this project** (per `feedback_no_agents` memory). Direct Edit/Read/Bash/Grep only.
- **Inline harness demos** for quick "does this work" tests — use `printf "...\n/exit\n" | zenith --effort max > /tmp/zenith.log`.

### Hardware/environment

- RTX 4070 8 GB, WSL2 Ubuntu 24.04. Round 1e training fits in <2 GB VRAM at 48K params, batch=128, max_enc=32, max_dec=32.
- Gemma 4 E4B llama-server NOT running this session (no Gemma inference needed; HRM/LLM-Computer work was self-contained).
- `setsid` + `disown` + `< /dev/null` for any training run that takes >30s. The WSL stdin-consumption gotcha bit during early monitor work.
- `pgrep -f calm.hrm.train` returns `tail -f` monitor processes — they MATCH "hrm" in their log filename. Don't treat them as live training; check `ps aux | grep -E "hrm|train"` to disambiguate.

### Plan file

`/home/gabe/.claude/plans/twinkling-soaring-dongarra.md` has the full structured plan that this session followed (HRM Rounds 1a/1c/1d/1e + LLM-Computer Round 4 layers + integration #3 vision + the CLAUDE.md/rules-update plan). Worth re-reading before resuming work.

## Files in Project (new this session)

- `calm/hrm/model.py` — HRMSeq2Seq encoder-decoder + legacy HRM. `Attention` has `is_causal` ctor flag; `CrossAttention`, `DecoderBlock`, `HRMEncoder`, `HRMDecoder`, `HRMSeq2Seq` classes.
- `calm/hrm/data.py` — `MathSeq2SeqDataset` with scratchpad/structure-only/answer modes. `<call>`/`<end_call>` special tokens. `tokenize_trace`/`detokenize_trace`. Trace generators.
- `calm/hrm/train.py` — `HRMSeq2SeqTrainer`, CLI flags `--seq2seq --structure-only --scratchpad --no-reverse-digits --hidden --num-heads --l-layers --h-layers --dec-layers --max-enc --max-dec`.
- `calm/hrm/inference.py` — `HRMSeq2SeqReasoner` with `<end_call>` interception. `_extract_final_answer()` for paren-heavy traces.
- `calm/hrm/checkpoints/math_structure_best.pt` — **production HRM** (48K params, 99.7% per-token, 96.7% full-expression).
- `calm/llm_computer/model.py` — `Small2DTransformer` (`d_head=2`, hard-max option).
- `calm/llm_computer/hull_cache.py` — `HullKVCache` (Andrew's monotone chain, 108× speedup).
- `calm/llm_computer/gate_graph.py` — compute IR (`Const`, `BinOp`, `Delegate`, `Result`) + hardware IR (`TokenInput`, `TokenOutput`).
- `calm/llm_computer/compile.py` — minimal compile_graph (identity + linear head only; LookUp/ReGLU NOT yet first-class).
- `calm/llm_computer/parse.py` — `parse_expression` (AST→GateGraph), `extract_problem_from_trace`.
- `calm/llm_computer/interpret.py` — topo-walk interpreter, `Delegate` → `safe_eval`.
- `calm/llm_computer/programs/{add_one,copy_past,increment_counter,threshold}.py` — hand-wired primitive programs.
- `calm/llm_computer/tests/test_hull_cache.py` — correctness + speedup tests.
- `calm/llm_computer/tests/test_parse_interpret.py` — round-trip parse+interpret tests.
- `scripts/eval_hrm_math.py` — eval with `--verified` flag using LLM-Computer interpreter.
- `RESEARCH/00_INDEX.md`, `01_LLM_Computer_Overview.md`, `02_Fast_Attention_2D_Heads.md`, `03_Compiling_Programs_to_Weights.md` — Percepta paper reference.
- `.claude/CLAUDE.md` — updated with 4th system + HRM/LLM-Computer section.
- `.claude/rules/architecture.md` — new HRM + LLM-Computer architecture section.
- `.claude/rules/training.md` — new HRM training (CRLM workflow) section.
- `.claude/rules/workflow.md` — new sweet-spot search section + extended monitor case study.

## Files to ignore / consider cleaning up

- `calm/hrm/checkpoints/math_hrm_best.pt` — pre-redesign legacy (45% masked-mode), kept for reference.
- `calm/hrm/checkpoints/math_seq2seq_best.pt` — Round 1a ceiling artifact.
- `calm/hrm/checkpoints/math_scratchpad_best.pt` — Round 1c/1d artifact (94% per-token, 43% full-expression). Could delete.
- `/tmp/hrm_*.log` — training logs from this session.
- `/tmp/quantize.log` — unrelated, from prior session.
