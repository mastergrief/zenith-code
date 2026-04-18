# Session Handoff — 2026-04-18 (R51 arc, continuation of session 33+)

Branch: `feature/multi-agent-qwen`, HEAD `61a5093`. 6 commits this session
on top of prior handoff `1f9e711`.

## TL;DR

R51 tier-3 distillation arc shipped end-to-end. **Null result**: MSE-only
distillation of Gemma's L24 contribution is insufficient for token-space
task preservation despite 92.6% residual-space variance explained. This
is the second instance of the R50.5 pattern (reconstruction fidelity ≠
causal effect) at a different scale. Also shipped: workflow rule
hardening from 6 gaps observed this session.

## Commits (6)

| SHA | Subject |
|---|---|
| `0364edd` | rules: agent_teams workflow hardening (messaging + task-per-spawn + revision discipline) |
| `125ba11` | round-51.1-2: broad L24 capture + tier-3 student stub |
| `67da31a` | round-51.3: student training script (prompt-level stratified, masked MSE) |
| `0f89367` | round-51.4-5: student install (monkey-patch L24) + dual-gate eval harness |
| `61a5093` | round-51.5 clean null: MSE distillation insufficient for token preservation |
| — | (R51.3 real training run — artifact in `calm/llm_computer/r51/checkpoints/r51_student.pt`, 4.76 MB, not committed) |

## R51 arc — end-to-end

### R51.1 — Broad prompt bank + L24 capture

**Files**: `calm/llm_computer/r51/prompt_bank.py` (634 LOC),
`calm/llm_computer/r51/__init__.py`, `scripts/r51_capture_broad.py`
(198 LOC).

Six deterministic samplers × 500 prompts each = 3000 prompts across
multi/single arithmetic, translation, code, creative, factual. Capture
script saves to `/tmp/r51_captures_broad.pt` (840 MB):

- `X_in`: `[40983, 2560]` fp32 — residual entering L24 (student input)
- `X_out`: `[40983, 2560]` fp32 — L24 contribution delta (student target)
- `prompt_ids`, `prompt_lens`, `prompts` — preserve sequence boundaries
- `domain_ids`, `DOMAIN_NAMES`, `prompt_counts`, `positions_per_domain`

Run on live Gemma 4 E4B tq4 via `bin/gemma_daemon.py`. 3000/3000
processed, 0 skipped, 40983 total positions. Per-domain stats:

| domain | prompts | positions | mean\|\|in\|\| | mean\|\|out\|\| |
|---|---:|---:|---:|---:|
| multi | 500 | 8678 | 90.72 | 75.73 |
| single | 500 | 5049 | 93.17 | 76.39 |
| trans | 500 | 8951 | 95.74 | 78.67 |
| code | 500 | 8647 | 97.74 | 79.34 |
| creative | 500 | 5177 | 95.22 | 77.42 |
| factual | 500 | 4481 | 94.50 | 78.24 |

### R51.1b — Schema revision (mid-round)

Builder2's R51.2 design-decisions report flagged that if training
batches samples per-position (S=1 per row), the student's self-attention
collapses to identity — equivalent to training a 650K-param MLP with
pos embedding, discarding the Small2DTransformer's whole point.

Fix: capture script edited to save `prompt_ids + prompt_lens + prompts
list` so R51.3 can reconstitute per-prompt sequences for real attention
training. Filed as task #7 (blockedBy #1) per the new downstream-
revision rule. This session's canonical example of the rule in action.

### R51.2 — Student stub

**File**: `calm/llm_computer/r51/student.py` (123 LOC).

`R51Student`: 1,245,696 params (1.25M) at default config (d_model=128,
n_layers=2, d_ffn=512, max_len=256). `Small2DTransformer` core with
d_head=2 invariant honored. Xavier init on `in_proj`, **zero init on
`out_proj`** — untrained student outputs exactly 0.0 (verified in
self-test), so pre-training install is a literal no-op.

Forward: `[B, S, 2560] -> [B, S, 2560]` via in_proj + pos + layernorm
+ inline `_core_forward` (bypasses tok_embed/head — Small2DTransformer
has a tiny vocab=1 throwaway head) + out_proj.

### R51.3 — Training

**File**: `scripts/r51_train_student.py` (400 LOC).

Prompt-level stratified split: per-domain shuffle with seed=42, last
val_frac=0.1 per domain → val. Result: train 2700 prompts (450/domain),
val 300 prompts (50/domain).

Padded batch collate — `[K, S_max, 2560]` + `[K, S_max]` mask. Masked
MSE: `sum((pred - y)^2 * mask) / (mask.sum() * d_io)`. Adam + linear
warmup (200 steps) → constant LR 1e-3, grad_clip 1.0.

Real training on /tmp/r51_captures_broad.pt: **23.2 seconds on GPU,
2000 steps**, best val at step 2000. Checkpoint at
`calm/llm_computer/r51/checkpoints/r51_student.pt` (4.76 MB, not
committed).

Val MSE and variance-explained (null baseline: `mean ||X_out||² / d_io`
≈ 2.38):

| domain | val MSE | var explained |
|---|---:|---:|
| multi | 0.0726 | **96.9%** |
| single | 0.1069 | 95.5% |
| creative | 0.1644 | 93.1% |
| trans | 0.2254 | 90.5% |
| factual | 0.2307 | 90.3% |
| code | 0.2433 | 89.8% |
| **total** | **0.1752** | **92.6%** |

Train EMA 0.175, val total 0.175 — train/val virtually identical, no
overfitting, student is capacity-saturated.

### R51.4 — Install mechanism

**Files**: `calm/llm_computer/r51/install.py` (105 LOC),
`calm/llm_computer/r51/__init__.py` (+3 re-exports).

Install path **B — monkey-patch `m._forward_layer`**. Paths A (forward
hook on `m.layers[24]`) and C (CardSlot) ruled out during the worker
brief stage:
- A: `m.layers[i]` is a `GemmaLayer` weight container, not `nn.Module`
- C: CardSlot is strictly additive, cannot REPLACE a layer's output

The patch: when called with `layer_idx==24`, `patched(h, layer, i, ...)`
returns `h + student(h)` and skips Gemma's native L24 compute. Other
layers fall through to original bound method. `InstallHandle.detach()`
restores original; idempotent.

KV-cache safety: Gemma 4 E4B `n_layer_kv_from_start = 42 - 18 = 24`, so
L24 is the first shared-KV layer — it never owns its own KV. Skipping
L24's forward leaves no stale cache state.

### R51.5 — Dual-gate eval (NULL RESULT)

**File**: `scripts/r51_eval_dual_gate.py` (201 LOC).

Held-out corpus: `build_broad_corpus(seed=43, per_domain=20)` = 120
fresh prompts, disjoint from training's seed=42/per_domain=500.

Two-phase: baseline generations first (Gemma unmodified, k=12 greedy),
install once, installed generations for same prompts, detach in
try/finally. Pairing via zipped index. Metrics per domain: exact-K
match count + mean prefix-match.

**Gates** (both FAIL):
- Training-dist (multi ∪ single) mean prefix = **0.194** vs 0.80 → FAIL
- Off-dist (trans ∪ code ∪ creative ∪ factual) mean prefix = **0.342**
  vs 0.95 → FAIL

Per-domain:

| domain | exact-K | mean-prefix |
|---|---:|---:|
| multi | 0/20 | 0.1125 |
| single | 0/20 | 0.2750 |
| creative | 0/20 | 0.1167 |
| trans | 2/20 | 0.2583 |
| factual | 3/20 | 0.4042 |
| code | 5/20 | 0.5875 |

**Counterintuitive**: off-dist > training-dist. Arithmetic (lowest val
MSE 0.07 on multi) preserves WORST (0.11); code (highest val MSE 0.24)
preserves BEST (0.59). The missing 3-10% MSE on arithmetic must
contain sharp digit-selector / content-reader directions — MSE
averages over all 2560 channels, washing out load-bearing task-
critical directions.

## Diagnosis

**R51 hypothesis is falsified.** Training-space MSE 92.6% does not
translate to token-space task preservation. This is the SECOND
instance of the R50.5 pattern (SAE features ablation had 0% causal
effect despite 99.6% reconstruction): reconstruction fidelity at
the chosen metric ≠ causal effect on the user-facing task.

Rules updated to reflect:
- `.claude/rules/tracing_roadmap.md` §Ruled-out: new R51.5 row
- `.claude/rules/augmentation_thesis.md` §Circuit typology: deep-diffuse
  row extended; R51 added to empirical basis

## Rules hardening (independent of R51 result)

Six gaps observed mid-session, all shipped in commit `0364edd`:

1. `CLAUDE.md` + `agent_teams.md`: "2 iterations" → "2-3 iterations"
   consistency
2. Agent spawn step 2 is now explicit `TaskCreate` — one task per
   agent, no exceptions
3. "Ship: commit" disambiguated — task description carries up-front
   authorization; otherwise CLAUDE.md's "explicit ask" rule
4. "Lead never writes implementation" gained a **disjoint-files
   corollary** — lead may edit files the worker has no claim to if
   the brief declares the claim explicitly
5. New footgun: worker reports must surface design decisions + deferred
   items (builder2's S=1 catch was the canonical instance)
6. New footgun: **downstream revisions open a new task, not silent
   re-edit** (R51.1b is the canonical instance)

Also added: Workflow invariant at top of both single-worker and
multi-worker sections: "one `TaskCreate` per agent, no exceptions,"
applies equally to both patterns.

Single-worker workflow rewrote 9 → 10 steps, each naming its tool/
channel (Agent.prompt, TaskUpdate, SendMessage, shutdown_request).

## Current environment state

- Branch `feature/multi-agent-qwen` at `61a5093`
- GPU: check `nvidia-smi` — daemon still warm at ~5 GB VRAM unless
  shut down
- `bin/gemma_daemon.py` running: PID file at `/tmp/gemma_daemon.pid`.
  Stop with `bin/gemma-run --stop`
- `/tmp/r51_captures_broad.pt` (840 MB) — the captured activations,
  not committed but regeneratable
- `calm/llm_computer/r51/checkpoints/r51_student.pt` (4.76 MB) —
  trained student, not committed, regeneratable via
  `scripts/r51_train_student.py`
- Team `r51-broad-capture` at `~/.claude/teams/r51-broad-capture/`
  — builders 1-4 all shut down (or should be; verify on resume)

## Next-round candidates (R52+)

The null result narrows the question. Four directions, listed by
increasing cost:

1. **KL-divergence loss on downstream logits.** Train student so that
   swapping student(h_before) for L24's contribution preserves Gemma's
   final logits (not L24's own output). Directly optimizes the user-
   facing metric. Student capacity unchanged; only the loss changes.
   Probably the cheapest re-try.

2. **Scale student 10-40×.** Current 1.25M hit a capacity plateau at
   MSE 0.17 — residual error might disappear at 12-50M params.
   Training cost scales linearly; 1.25M was 23s on GPU, 50M would be
   ~15 min. Still cheap.

3. **Per-head / per-subspace distillation.** Instead of reproducing
   the whole 2560-d L24 contribution, train smaller students on
   specific subspaces (e.g., H4 V projection — the R17 arithmetic
   circuit's load-bearing 512-d). Risk: L24 multi-step is diffuse
   (not concentrated per R47.4), so per-head might fragment the signal.

4. **Pivot: stop trying tier-3 on L24.** L24's deep-diffuse shape has
   now failed at attention level (R47.4), FFN-neuron (R48.1), SAE
   features (R50.5), AND MSE distillation (R51.5). Possibly not
   reachable by any currently-known substrate mechanism. Pivot to
   tier-2 extensions (more hub-served capabilities under
   `HubInjectionCard`) or tier-3 attempts on concentrated circuits
   (L23 H4 arithmetic, L30 H4/H6, L37 H6 induction).

Recommendation if continuing: start with (1) KL-divergence. It's the
cheapest meaningful retry and directly measures what we actually want
(token preservation, not residual MSE). Keep the same student architecture
and capture schema; only swap the loss + supervise through Gemma's
downstream layers.

## First action on resume

1. **Read this handoff in full.**
2. **Verify daemon state**: `bin/gemma-run --status`. If running, keep
   warm for next round; if not, reload takes ~3 min.
3. **Review `.claude/rules/augmentation_thesis.md` §"Circuit typology"**
   — the deep-diffuse row now has four ruled-out install mechanisms.
   This constrains what R52 can reasonably try.
4. **Decide R52 direction with user**: KL-divergence (cheap retry) vs
   capacity scaling vs per-head vs pivot. Default recommendation is
   KL-divergence.
5. **Team `r51-broad-capture` cleanup**: `TeamDelete` once all builders
   idle (verify with `SendMessage shutdown_request`). Spawn fresh team
   for R52 with a tightened brief.
6. **Task list**: create new tasks for R52 rounds per the invariant
   rule.

## Artifacts worth keeping

- `/tmp/r51_captures_broad.pt` (840 MB) — regenerating takes ~8 min
  at ~2s/prompt on live Gemma. Keep for R52 retry experiments (same
  data, different loss).
- `calm/llm_computer/r51/checkpoints/r51_student.pt` (4.76 MB) — the
  baseline student for comparison in future runs.

## Workflow receipts (for rule validation)

Four workers spawned this session, all cleanly shut down. Each worker's
brief demanded "Design decisions worth flagging" + "Deferred / open"
sections:

- **Builder1 (R51.1)**: shipped prompt bank + capture script. Report
  flagged bank sizes + per-domain dedup strategy.
- **Builder2 (R51.2)**: shipped student stub. **Report caught S=1
  dead-attention issue** → triggered R51.1b capture revision. Canonical
  example of why the design-decisions rule exists.
- **Builder3 (R51.3)**: shipped training script. Report flagged
  padding+mask vs bucket-sort choice + LR schedule decision + loss
  normalization math.
- **Builder4 (R51.4+R51.5)**: shipped install + eval. Report flagged
  install-path reasoning (A/B/C trade-off), KV-cache safety, held-out
  seed disjointness. Combined-task spawn slightly against "one task
  per agent" but justified by tight coupling.

Six rule gaps observed during the session were all shipped as rule
edits (commit `0364edd`) by session end. The rules ate their own
dogfood — R51.1b task #7 was filed per the new downstream-revision
rule shipped EARLIER the same session.
