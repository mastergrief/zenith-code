# HRM-Text-1.58 deviations from upstream

**Phase 0 deliverable for task #51.** Explicit per-item account of
what we plan to differ from `sapientinc/HRM-Text` SHA
`056c4ecad217933b9db33dfb22e30a2f511315ed` AND what's banned outright.
Default = ZERO deviations. Each deviation must justify itself with
revisit criteria.

Companion to `00_ARCHITECTURE.md` (source-faithful audit).

## Provenance

- Locked route: gabe AUQ msg 1779449209719-a01eb795 (2026-05-22 ~11:30 UTC) — Park TRM-1.58 RDT-v2 line, pivot to true HRM-Text-1.58 1:1
- Codex park +1 + guardrails: msg 1779449372857-b1876182
- Codex Q1/Q2/Q3 routing: msg 1779449622291-8c9660ae (Q1 /tmp / Q2 staged capacity / Q3 two RESEARCH docs)
- Codex substrate-vs-arch clarification: msg 1779450510543-a9cc578c
- Gabe "make hrm-1.58 work, fast iterations": msg 1779450637583-e9e3f834
- Codex fast-iteration contract: msg 1779450605092-f16a229d

## REJECTED imports (NOT deviations from Sapient upstream — constraints on new code)

Things absent from `sapientinc/HRM-Text` AND banned from HRM-Text-1.58
internal architecture. They were considered, rejected, will not be
imported. Substrate compilation path is preserved GLOBALLY as adjacent
card/facade layer; integration later via explicit adapter, not
internal.

| Rejected | Why |
|---|---|
| `Small2DTransformer` / substrate `d_head=2` invariant | Absent from upstream; forcing breaks 1:1 hypothesis |
| Gate-graph IR / compiled-card hosting interface | Compilation is separate layer; HRM-Text is the trained-arch experiment |
| DeltaNet / Householder fast-weight recurrence | TRM/RDT-v2 path, already null-gated |
| Copy gate / pointer attention / DT/PT transducers | TRM/RDT-v2 path; absent from HRM upstream |
| ACT halt-head / Q-head / ponder cost / halt loss | TRM-1.58 scaffold patched onto RDT-v2; absent from HRM upstream which uses `bp_warmup` schedule instead |
| Custom carry between forward passes | This variant is explicitly `nocarry` |
| Pre-RMSNorm-as-separate-flag pattern | Upstream has `norm_type: pre|post` config, NOT a flag-bundle stacking |

## Phase 1 (FP/BF16 1:1 port) planned deviations

DEFAULT: zero. Below are minimum-necessary deviations needed for the
mini-capacity FP smoke. Each requires (a) what differs, (b) why, (c)
upstream baseline citation, (d) revisit criteria.

### D1.1 — Mini-capacity sizing

- **What**: Use `hidden_size=256, n_layers=4 (2 per H/L when half_layers=True), num_heads=2, head_dim=128, expansion=4`. Estimated params ~5-8M.
- **Why**: 8 GB consumer GPU + GSM8k char surface fast-iteration requires < 50M params. Smallest upstream config (`config/arch/size/B.yaml`) is 12 layers × 1024 hidden ~150M params, too large.
- **Upstream**: `config/arch/size/B.yaml` smallest = 12 layers × 1024 hidden × 8 heads × head_dim=128 × expansion=4.
- **Preserved invariants**: head_dim=128 (per codex Q2 — preserve upstream architectural ratio), expansion=4, init_type=lecun_normal, norm_type=pre, rope_theta=10000.
- **Revisit when**: Phase 1 FP smoke shows non-degenerate signal AND need to scale to Tier B (closer to upstream). Or smoke is degenerate → bump head_dim, hidden, or layers per Tier B trajectory.

### D1.2 — Flash-attention dependency

- **What**: Replace `flash_attn_varlen_prefixlm` + `flash_attn_with_kvcache` with PyTorch `F.scaled_dot_product_attention` using a constructed PrefixLM mask + manual KV cache append.
- **Why**: Upstream depends on Sapient's custom flash-attn 2.8.3 + custom `flash_attention_prefixlm_v2.py` kernel. Building these on consumer 8 GB hardware is fragile and out-of-scope for Phase 1 fast smoke.
- **Upstream**: `models/layers.py:11` imports `flash_attn_with_kvcache` from `flash_attn_interface`; `models/layers.py:10` imports `flash_attn_varlen_prefixlm` from `flash_attention_prefixlm_v2.py`.
- **Revisit when**: Phase 1 smoke confirms architecture is healthy AND we want to match upstream training-step wall time. Then port the flash-attn kernel OR mark this as a permanent fork divergence.

### D1.3 — Optimizer

- **What**: Use `torch.optim.AdamW(lr, betas=(0.9, 0.95), weight_decay=0.1)`.
- **Why**: Upstream uses custom `adam_atan2` (atan2-based update, `models/adam_atan2.py`). Building from source is fragile; AdamW is the closest standard.
- **Upstream**: `models/adam_atan2.py`. Configured per training loop in `pretrain.py`.
- **Revisit when**: Phase 1 architecture is locked AND training is stable; then A/B AdamW vs adam-atan2 if convergence quality matters.

### D1.4 — Distributed training assumptions

- **What**: Single-GPU only. No FSDP2, no `dist.all_reduce`, no `WrappedTensor` device-marshalling. `loss_divisor` computed locally (just `masks.sum().to(torch.float32)`).
- **Why**: Single-GPU consumer setup. FSDP2 is over-engineering for a smoke run.
- **Upstream**: `models/lm_head.py:55` allreduces `loss_divisor`; `models/common.py:21-32` `WrappedTensor` marshalls for FSDP2.
- **Revisit when**: Never within this arc unless scaling to multi-GPU.

### D1.5 — Tokenizer / dataset

- **What**: Reuse claw-code `Gsm8kTokenizer` (char-level, 98 tokens) + `Gsm8kDataset` for GSM8k char surface. Re-target HRM-Text's vocab=65536 down to 98.
- **Why**: Fast comparability with prior TRM-1.58 surface. GSM8k char + 17×23 smoke is the established eval bed.
- **Upstream**: `dataset_new.py` is for HLM corpora (large web text); `tokenizer` not in repo (HF uses a custom BPE in 1B checkpoint). Not portable to our GSM8k char surface.
- **Revisit when**: Phase 1 architecture works at char-vocab; consider porting upstream BPE for sentence-level surfaces in Phase 3.

### D1.6 — `flash_attention_prefixlm_v2` PrefixLM mask reimplementation

- **What**: Construct PrefixLM mask as `(L, L)` bool: bidirectional on positions ≤ sep, causal on positions > sep, for each sequence. Apply additively to attention logits before softmax inside `F.scaled_dot_product_attention(attn_mask=...)`.
- **Why**: PrefixLM is upstream-default (`config: prefix_lm=true`). Need to preserve semantics.
- **Upstream**: `models/flash_attention_prefixlm_v2.py` implements this in flash-attn kernel. Mask construction logic visible in that file (Phase 1 will inline an equivalent pure-PyTorch mask builder).
- **Revisit when**: Phase 2 native 1.58 ships and we want to match upstream attention numerics exactly.

### D1.7 — pretrain.py training loop

- **What**: Custom Phase 1 training loop (~150 lines) that wraps the HRM model with: AdamW, linear-warmup-cosine LR, F.cross_entropy loss, step-100 + step-200 saves via Slice 13m-style `--save-at-step` (port that ONE helper from feature/trm-1.58 with explicit provenance).
- **Why**: Upstream `pretrain.py` is FSDP2 + Hydra config + Wandb + adam-atan2 — too much surface area for Phase 1 smoke.
- **Upstream**: `pretrain.py` (392 lines).
- **Revisit when**: Never — this is a permanent simplification for the smoke.

## Phase 2 (native 1.58 bulk linears) planned deviations

These are the WHOLE POINT of the 1.58 work. Each is intentional.

### D2.1 — Replace bulk `LinearInit` with ternary BitLinear

- **What**: New `BitLinearInit(in_features, out_features, ...)` class. Forward path: `weight_quantized = absmean_ternary(weight)` (in {-1, 0, +1}, scaled by per-tensor mean abs), then `F.linear(x, weight_quantized * scale, bias)`. STE for backward (gradient flows through `weight` directly). FP/BF16 master weights persisted; ternary weights computed forward-only.
- **Why**: This IS the 1.58 hypothesis. Test whether HRM-Text's recurrence survives native ternary bulk linear training.
- **Bounded scope**: ONLY `gqkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`. NOT `lm_head`, NOT `embed_tokens`, NOT `zL_init`, NOT norms.
- **Revisit when**: Phase 2 A/B vs Phase 1 FP shows ternary either survives (continue iteration) or collapses (record null, pivot).

### D2.2 — KEEP FP/BF16 on stability-sensitive components

- **What**: `embed_tokens` (ScaledEmbeddingInit), `lm_head` (final projection), all `F.rms_norm` calls, `zL_init` buffer all stay FP/BF16. No ternary quantization on these.
- **Why**: BitNet b1.58 convention: norms, embd, output stay full-precision because their outputs/inputs have outsized dynamic range. Empirical from BitNet paper. Codex msg 1779450605092 contract: "Phase 2 native 1.58 should change only bulk linears first."
- **Revisit when**: Phase 2 ternary survives; THEN consider quantizing additional components A/B-style, one at a time.

### D2.3 — STE backward + grad clip

- **What**: Backward through `BitLinear`: `grad_weight_quantized → grad_weight` (no transformation). Apply `torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)` before opt.step.
- **Why**: Standard STE for ternary training. Grad clip prevents BitNet-typical early-training divergence.
- **Revisit when**: Empirical training instability; tighten clip OR add weight decay tuning OR investigate alternate STE shapes.

## Phase 3 (capacity scale-up) — gated

NO Phase 3 deviations unless Phase 2 A/B shows ternary HRM-Text-1.58
survives. Then revisit each Phase 1 deviation to see if it should be
relaxed at scale.

## Revisit triggers (summary)

| Trigger | Action |
|---|---|
| Phase 1 FP smoke degenerate at Tier A (D1.1) | Move to Tier B (head_dim/heads closer to upstream); not "more deviations" |
| Phase 1 FP smoke healthy | Proceed to Phase 2 ternary A/B at SAME mini-capacity |
| Phase 2 ternary collapses | Park HRM-Text-1.58 line per codex binary gate discipline |
| Phase 2 ternary survives + Phase 1 deviations look load-bearing | Revisit each deviation explicitly with codex; never silent |
| Phase 1/2 numerics drift from upstream | Port `flash_attention_prefixlm_v2` and/or `adam_atan2` |

## Versioning

This file is the contract for Phase 1 + 2 implementation. Any
deviation added during implementation requires a corresponding entry
here AND a codex gate. Implementation that contradicts this file is
a bug.
