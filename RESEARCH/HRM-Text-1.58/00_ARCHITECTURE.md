# HRM-Text Architecture — source-faithful audit

**Phase 0 deliverable for task #51.** Faithful description of upstream
`sapientinc/HRM-Text` architecture, opinion-free, citation-only. Used
as the contract for Phase 1 (1:1 FP/BF16 port) and Phase 2 (native
1.58 bulk linears).

## Source pin

| Field | Value |
|---|---|
| Repo | `github.com/sapientinc/HRM-Text` |
| Commit SHA | `056c4ecad217933b9db33dfb22e30a2f511315ed` |
| Date | 2026-05-22T07:29:59Z |
| Message | "quick fix" |
| Trained 1B checkpoint | `huggingface.co/sapientinc/HRM-Text-1B/raw/main/config.json` |
| Cached locally | `/tmp/sapientinc_hrm_text/` (160 KB, 22 files) |

All file:line citations below resolve against the pinned SHA.

## Top-level composition

`config/arch/net/hrm.yaml:1-2` declares the model factory:
```yaml
name: baselines.hrm_nocarry_bp_warmup@HierarchicalReasoningModel
head: lm_head@LMHead
```

So the canonical HRM model class is `HierarchicalReasoningModel`
(`models/baselines/hrm_nocarry_bp_warmup.py:46`), wrapped by `LMHead`
(`models/lm_head.py:18`). All other variants in `models/baselines/`
(trm_nocarry, ut_nocarry, rins_nocarry, transformer_wrapper) are
comparison baselines — NOT HRM.

## Module hierarchy

```
LMHead (lm_head.py:18)
├── embed_tokens : ScaledEmbeddingInit (layers.py:88)
├── model        : HierarchicalReasoningModel (hrm_nocarry_bp_warmup.py:46)
│   ├── H_level  : HierarchicalReasoningModelRecurrentBlock (hrm_nocarry_bp_warmup.py:26)
│   │   └── core : Transformer (transformer.py:99)  [n_layers // 2 blocks when half_layers=True]
│   ├── L_level  : HierarchicalReasoningModelRecurrentBlock
│   │   └── core : Transformer
│   └── zL_init  : nn.Buffer (hidden_size,) bf16, persistent, trunc_normal_init_ std=1.0
└── lm_head      : LinearInit  (head_hint dim → vocab_size, bias=False)
```

`Transformer.layers` is `nn.ModuleList[TransformerBlock]` (transformer.py:111).
Each `TransformerBlock` (transformer.py:65) = Attention + SwiGLU + RMSNorm.

## Attention block

`models/layers.py:116-155`. Fused projection design.

```python
self.gqkv_proj = LinearInit(
    hidden_size,
    head_dim,
    batch_out_features=(2 * num_heads + 2 * num_key_value_heads,),
    bias=False, init_std=init_std_in,
)
self.o_proj = LinearInit(head_dim * num_heads, hidden_size, bias=False, init_std=init_std_out)
```

Single linear `gqkv_proj` produces gate + query + key + value in one
matmul. Split order: `gqkv.split((num_heads, num_heads, num_key_value_heads,
num_key_value_heads), dim=-2)` (layers.py:135).

Forward (`layers.py:129-155`):
1. `gqkv = gqkv_proj(hidden_states)` → reshape to `(..., 2*h + 2*kvh, head_dim)`
2. Split into `gate, query, key, value`
3. RoPE applied to query + key (if `cos_sin` provided)
4. Training: `flash_attn_varlen_prefixlm(query, key, value, is_causal, ...)` (custom kernel in `flash_attention_prefixlm_v2.py`)
5. Inference: `flash_attn_with_kvcache(...)` from `flash_attn_interface`
6. **Gated output**: `attn_output = sigmoid(gate) * attn_output` (layers.py:154)
7. `o_proj(attn_output)`

`attn_type` ∈ `{"causal", "prefixlm"}` (layers.py:16). 1B config sets `prefix_lm=true`.

GQA support: `num_key_value_heads ≤ num_heads`. 1B config uses MHA
(`num_attention_heads == num_key_value_heads == 12`).

## SwiGLU MLP

`models/layers.py:158-168`. Fused gate+up projection.

```python
self.gate_up_proj = LinearInit(hidden_size, intermediate_size,
                                batch_out_features=(2,), bias=False, init_std=init_std_in)
self.down_proj    = LinearInit(intermediate_size, hidden_size,
                                bias=False, init_std=init_std_out)
```

Forward: `gate, up = gate_up_proj(x).chunk(2, dim=-1); down_proj(silu(gate) * up)`.

`intermediate_size` computed from `expansion` parameter
(transformer.py:42-46):
```python
return find_multiple(round(expansion * hidden_size * 2 / 3), 256)
```
Matches GLU parameter count to a vanilla transformer with the same
expansion. Rounded to multiple of 256.

## TransformerBlock

`models/transformer.py:65-96`. Pre-norm OR post-norm via runtime dispatch:

```python
self.forward = getattr(self, f"_forward_{config.norm_type}")
self.norm = lambda x: F.rms_norm(x, (x.shape[-1],), eps=config.norm_eps)
```

Pre-norm (`_forward_pre`, transformer.py:90):
```
x = x + attn(norm(x))
x = x + mlp(norm(x))
```

Post-norm (`_forward_post`, transformer.py:94):
```
x = norm(x + attn(x))
x = norm(x + mlp(x))
```

1B config uses pre-norm (per `config/arch/size/B.yaml:6` and HF
`config.json` rms_norm_eps=1e-6).

`F.rms_norm` is the unparameterized PyTorch built-in (no learned
weight/bias). norm_eps from config.

## Transformer

`models/transformer.py:99-128`. Standard stack.

- Optional `rotary_emb = RotaryEmbedding(head_dim, max_seq_len, base=rope_theta)` (transformer.py:106-108) when `pos_emb_type == "rope"`.
- `layers = ModuleList([TransformerBlock(config) for _ in range(n_layers)])` (transformer.py:111)
- Final norm: `F.rms_norm` if pre-norm, else identity (transformer.py:114-116)
- `create_cache` → `[Cache.create(...) for _ in range(n_layers)]` per-layer KV cache (transformer.py:119)

`head_hint` dict (transformer.py:102-103) carries `in.dim`/`in.init_std` + `out.dim`/`out.init_std` for `LMHead` init.

## HierarchicalReasoningModel — the H/L recurrence

`models/baselines/hrm_nocarry_bp_warmup.py:46-100`. The defining
feature of HRM-Text.

Config (`hrm_nocarry_bp_warmup.py:11-23`):
```python
class HierarchicalReasoningModelConfig(TransformerConfig):
    half_layers: bool = False     # Divide n_layers by 2, split evenly H/L
    H_cycles: int                 # outer loop count (1B: 2)
    L_cycles: int                 # inner loop count per H-cycle (1B: 3)
    bp_warmup_ratio: float = 0.0  # fraction of training steps for bp warmup
    bp_min_steps: int = 2
    bp_max_steps: int = 5
    H_override: Dict[str, Any] = {}  # per-config override for H_level
```

When `half_layers=True` (config/arch/net/hrm.yaml:4): `n_layers //= 2`
so EACH of H_level and L_level gets `n_layers/2` `TransformerBlock`s.

`HierarchicalReasoningModelRecurrentBlock` (hrm_nocarry_bp_warmup.py:26-43):
- Wraps a `Transformer(config)` as `.core`
- Forward: `return self.core(hidden_states + input_injection, **kwargs)` — **ADDITIVE input injection** (NOT gated, NOT residual write).

`HierarchicalReasoningModel.__init__` (hrm_nocarry_bp_warmup.py:46-73):
- `H_level = HierarchicalReasoningModelRecurrentBlock(TransformerConfig(**(config.model_dump() | config.H_override)))` — H can override config
- `L_level = HierarchicalReasoningModelRecurrentBlock(config)` — L always uses base config
- `zL_init = nn.Buffer(trunc_normal_init_(torch.empty(hidden_size, dtype=bf16), std=1.0), persistent=True)` — **L-level initial hidden state (persistent buffer, bf16 hardcoded)**
- `head_hint = H_level.core.head_hint` — LMHead derives shape from H

`forward` (hrm_nocarry_bp_warmup.py:75-91):

```python
z_H, z_L = x, self.zL_init
H_bp_steps = min(H_cycles, bp_steps - 1)
L_bp_steps = bp_steps - H_bp_steps

for i in range(H_cycles):
    for k in range(i * L_cycles, (i + 1) * L_cycles):
        with torch.set_grad_enabled(torch.is_grad_enabled() and (k >= H_cycles * L_cycles - L_bp_steps)):
            z_L = L_level(z_L, z_H, **seq_info, cache=cache["L"][k] if cache else None)

    with torch.set_grad_enabled(torch.is_grad_enabled() and (i >= H_cycles - H_bp_steps)):
        z_H = H_level(z_H, z_L, **seq_info, cache=cache["H"][i] if cache else None)

return None, z_H
```

Semantics:
- `z_L` initialized from persistent buffer `zL_init` (NOT recomputed per-batch)
- `z_H` initialized from input embedding `x`
- Inner loop: `L_cycles` L-steps where `z_L` reads `z_H` as input injection
- Outer loop: 1 H-step at the end of each inner loop, where `z_H` reads final `z_L` as input injection
- **Grad enabled selectively** per `bp_steps` schedule (the "bp_warmup" of the name)

Returns `(None, z_H)` — the `None` is the carry slot (this variant is `nocarry`).

KV cache structure (hrm_nocarry_bp_warmup.py:72-73):
```python
dict(H=[H_level.create_cache(**kwargs) for _ in range(H_cycles)],
     L=[L_level.create_cache(**kwargs) for _ in range(H_cycles * L_cycles)])
```
So `cache["L"][k]` is keyed by `k = i * L_cycles + inner_k`, and `cache["H"][i]` by outer H-step index.

## bp_warmup schedule

`hrm_nocarry_bp_warmup.py:93-97`:
```python
def compute_train_extra_args(self, train_state):
    warmup_steps = train_state.total_steps * self.bp_warmup_ratio
    progress = min(1.0, train_state.step / warmup_steps) if warmup_steps > 0 else 1.0
    return dict(bp_steps=self.bp_min_steps + int(progress * (self.bp_max_steps - self.bp_min_steps)))
```

`bp_steps` linearly ramps from `bp_min_steps` to `bp_max_steps` over the
first `bp_warmup_ratio * total_steps` training steps. After warmup,
`bp_steps = bp_max_steps`.

For HF 1B config + `hrm.yaml`: `bp_warmup_ratio=0.2`, `bp_min_steps=2`,
`bp_max_steps=5`. After warmup with `bp_steps=5`, `H_cycles=2`,
`L_cycles=3`:
- `H_bp_steps = min(H_cycles, bp_steps - 1) = min(2, 4) = 2`
- `L_bp_steps = bp_steps - H_bp_steps = 3`
- L grad enabled when `k >= H_cycles*L_cycles - L_bp_steps = 6 - 3 = 3`, so `k ∈ {3, 4, 5}` — **last 3 of 6 L updates**
- H grad enabled when `i >= H_cycles - H_bp_steps = 2 - 2 = 0`, so `i ∈ {0, 1}` — **both H updates**

Net: after warmup, gradients cover the last 3 of 6 L updates AND
both H updates.

## LMHead

`models/lm_head.py:18-74`.

Init:
- `embed_tokens = ScaledEmbeddingInit(vocab_size, head_hint.in.dim, init_std=head_hint.in.init_std)`
- `lm_head = LinearInit(head_hint.out.dim, vocab_size, bias=False, init_std=head_hint.out.init_std)`

NOT tied (1B config: `tie_word_embeddings: false`).

`ScaledEmbeddingInit` (layers.py:88-102): stores `scale = 1.0 / init_std`,
forward returns `scale * F.embedding(input, embedding_weight)`. For
1B: init_std=0.0255 → scale ≈ 39.19, matching HF config's
`embedding_scale: 39.191835884530846`.

Forward (lm_head.py:34-74):
1. `input_embedding = embed_tokens(batch["inputs"])`
2. `new_carry, logits = model(carry, input_embedding, ...)` — HRM forward
3. `logits = lm_head(logits)` — project to vocab
4. If `labels` in batch: F.cross_entropy with `IGNORE_LABEL_ID = -100` in FP32, with FSDP allreduce-averaged divisor
5. Metrics: `loss`, `accuracy` (per-token), `exact_accuracy` (per-sequence — all tokens correct AND has valid tokens)

## Initialization

`models/transformer.py:42-62`. Three init types selectable via
`init_type`:

| init_type | in_std | attn_out_std | ff_out_std |
|---|---|---|---|
| `fixed_normal` | `init_std` or 0.02 | same | same |
| `lecun_normal` (1B uses this) | `1/√hidden_size` | same as in_std | `1/√intermediate_size` |
| `megatron` | `init_std` or `1/√hidden_size` | `in_std / √(2*n_layers)` | same as attn_out_std |

`LinearInit` (layers.py:61-85) uses `trunc_normal_init_` (common.py:10-13):
```python
return tensor.normal_().fmod_(3.0).mul_(1.014762601732121 * std)
```
Approximate truncated normal: sample standard normal, fmod-3 truncation,
scale by `1.014762601732121 * std` to preserve std after truncation.

## Position embedding

`RotaryEmbedding` (layers.py:41-58). Standard RoPE with:
- `inv_freq = 1 / (base ** (arange(0, dim, 2) / dim))` (FP32)
- `freqs = outer(t, inv_freq)` where `t = arange(max_seq_len)`
- `emb = cat([freqs, freqs], dim=-1)` — half-rotation permutation
- Stored as non-persistent buffers `cos_cached`, `sin_cached`

Applied (layers.py:30-38) only to query + key (not value/gate):
```python
return ((x * cos.unsqueeze(-2)) + (rotate_half(x) * sin.unsqueeze(-2))).to(x.dtype)
```

`rotate_half(x)` = `cat((-x2, x1), dim=-1)` — half-dim rotation.

1B: `rope_theta=10000.0`, `max_position_embeddings=4096`.

## Flash-attention (PrefixLM)

`models/flash_attention_prefixlm_v2.py` (282 lines). Custom variable-
length flash-attn wrapper supporting PrefixLM masking. Required for
training; inference uses `flash_attn_with_kvcache` from
`flash_attn_interface` package (layers.py:11).

Not fully expanded here — Phase 1 port may need a fallback path
since the canonical kernel depends on Sapient's custom build of
flash-attn 2.8.3 (`docker/requirements/torch_extensions.txt` per repo
tree). For mini-capacity FP smoke we can fall back to PyTorch
`scaled_dot_product_attention` with a constructed PrefixLM mask;
deviation list will record this.

## Optimizer

`models/adam_atan2.py` (108 lines). Adam-atan2 variant (atan2-based
update). Not a standard PyTorch optimizer; required for source-
faithful 1B reproduction. Phase 1 mini-smoke can use standard
`AdamW` if adam-atan2 unavailable; deviation list records this.

## Capacity tiers (for Phase 1 sizing)

`config/arch/size/B.yaml` (smallest in repo):
```yaml
n_layers: 12       # split half → 6 per H/L when half_layers=True
hidden_size: 1024
num_heads: 8       # head_dim = 1024/8 = 128
expansion: 4
norm_type: pre
init_type: lecun_normal
rope_theta: 10000.0
```

1B HF config:
```
hidden_size=1536, num_attention_heads=12, head_dim=128,
n_layers=16 (8 per H/L), intermediate_size=4096
H_cycles=2, L_cycles=3
```

Note: 1B HF config (`n_layers=16, hidden_size=1536`) differs from BOTH
repo size YAMLs — B is `12 × 1024` and L is `24 × 1280`. The actual
training config used to produce the 1B HF checkpoint isn't pinned by
the repo tree alone (likely a custom YAML or runtime override). Phase 1
deviation list records this and pins our chosen mini-config explicitly.

## Training contract

`pretrain.py` (392 lines, not fully detailed here). Key contract per
`compute_train_extra_args`: training loop passes `train_state` (step
+ total_steps) to model, which returns `bp_steps` for the forward.
`pretrain.py` is the entry point — Phase 1 mini-smoke will write its
own training loop (NOT vendor `pretrain.py`) but match its contract.

Loss: F.cross_entropy in FP32 (lm_head.py:52). Loss reduction: sum,
divided by valid-token count, allreduce-averaged across distributed
processes.

## What HRM-Text does NOT have

These are absent from upstream as of pinned SHA. The 1:1 port
inherits the absence:

- No ACT head, no Q-head, no halt loss, no ponder cost (TRM-1.58 had these — they are NOT in HRM)
- No copy gate, no pointer attention (DT/PT had these — not in HRM)
- No `d_head=2` substrate decomposition; `head_dim` is a config (1B uses 128)
- No `Small2DTransformer`, no gate-graph IR, no compiled-card hosting interface
- No DeltaNet, no Householder fast-weight recurrence
- No GroupedAttention, no MQA above what GQA provides via `num_key_value_heads`
- No "carry" between forward passes — this variant is `nocarry`

These are the constraints on the 1:1 port: NEW HRM-Text-1.58 code
includes ONLY what is in upstream, period.
