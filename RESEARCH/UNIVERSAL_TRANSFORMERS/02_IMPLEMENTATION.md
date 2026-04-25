# UT-Family — Implementation

Concrete mechanics, math, schemas, and reference code shapes for
the unified UT/RDT/NAMM stack. Companion to `00_INDEX.md` (overview
+ decision), `01_ARCHITECTURE.md` (primitives + design spec),
`03_TESTING.md` (substrate-relevance gates).

## §1 Parcae LTI stability constraint

The load-bearing fix that makes deep RDT trainable.

### The problem

Training looped models is notoriously unstable. Two failure modes
dominate:

- **Residual explosion** — hidden state `h_t` grows unboundedly
  across loops
- **Loss spikes** — training diverges suddenly due to large
  spectral norms in injection parameters

### The dynamical-systems view

Recast the recurrence as a **discrete linear time-invariant (LTI)
dynamical system** over the residual stream. Ignoring the nonlinear
Transformer contribution, the recurrence becomes:

```
h_{t+1} = A · h_t + B · e
```

For this LTI system, stability is governed entirely by the
**spectral radius** of A:

- `ρ(A) < 1` → stable, convergent
- `ρ(A) ≥ 1` → unstable, divergent

Empirically (Parcae 2026): every divergent training run learned
`ρ(A) ≥ 1`. Every convergent run maintained `ρ(A) < 1`. Stability
correlates 1:1 with the spectral radius condition.

### The fix

Constrain injection parameters so stability is guaranteed by
construction:

```python
# Parameterize A as continuous negative diagonal
log_A = nn.Parameter(torch.zeros(d_model))
delta_t = nn.Parameter(torch.tensor(1.0))

def get_A():
    A_continuous = torch.diag(-torch.exp(log_A))      # negative diag
    A_discrete = torch.matrix_exp(delta_t * A_continuous)  # ZOH/Euler
    return A_discrete
```

Properties:

1. `-exp(log_A)` is **always negative** ⇒ continuous-time eigenvalues are negative
2. `matrix_exp(Δt · A_continuous)` is **always positive** with magnitude `< 1` ⇒ discrete-time eigenvalues lie in (0, 1)
3. Therefore `ρ(A_discrete) < 1` **always holds**, regardless of learning rate or batch noise

Result: looped model becomes significantly more robust to
hyperparameter selection and trains cleanly even at high learning
rates. Reference impl: `OpenMythos.recurrent.injection.get_A()`.

### Diagnostic check

```python
A = model.recurrent.injection.get_A()
rho = torch.linalg.eigvals(A).abs().max().item()
assert rho < 1.0, f"Spectral radius {rho:.4f} ≥ 1 → unstable"
```

## §2 NAMM 3-step pipeline

Computation flow per token, per layer:

### Step 1: Spectrogram

The attention values for each token across positions are converted
to a **spectrogram** — frequency-based representation analogous to
audio / medicine / seismology. Captures temporal frequency
structure of the attention pattern.

```
attn_values[token, position] : [seq_len]
  ↓ STFT or wavelet transform
spectrogram[token, freq_bin, time_bin] : [F, T]
```

### Step 2: EMA compression

Elementwise exponential moving average condenses the spectrogram
into a compact, fixed-size feature summary of the history of each
token's attention values.

```
features[token, channel] = EMA_α(spectrogram_flat[token])
                         : [d_compressed]
```

Output dimension is constant regardless of context length —
critical for the classifier downstream.

### Step 3: Classifier score

Small learned NN takes EMA features and outputs a scalar
**keep/forget score** per token.

```
score[token] = MLP(features[token])  : scalar
keep[token]  = (score[token] > threshold)
```

Tokens with `keep=False` are evicted from the KV cache; their slots
are reclaimed for new tokens or fresh capacity.

### The non-differentiability problem

Each decision is binary (keep or evict). Lost tokens are gone
forever — the operation is **not differentiable**, blocking gradient
methods. Hence:

## §3 Evolutionary training for NAMM

Standard SGD doesn't apply. NAMMs are trained via **evolution**:

```
1. Initialize population of NAMM classifiers (random init)
2. For each generation:
   a. Run each NAMM atop a frozen base LM on a held-out task
   b. Score by downstream task performance (perplexity / accuracy)
   c. Select top-k performers
   d. Mutate (parameter perturbation) to fill next generation
3. Repeat until convergence
```

Trial-and-error optimization handles the non-differentiable
keep/forget decisions naturally. Computational cost: substantial —
each generation requires N forward passes through a large LM. Sakana
trained on Llama-3-8B with significant compute budget.

### Universal-transfer mechanism

NAMMs operate on **attention matrices** — universal across
transformer layers. A single NAMM can be applied across the model's
layers and even transferred to entirely different transformers
**without any further training**:

- Llama-3-8B → Llama-70B (10× scale, same family)
- Llama-3-8B → Llava Next Video (different modality)
- Llama-3-8B → Decision Transformer (different task domain — RL)

In each case, NAMMs retain benefits by discarding redundant tokens
appropriate to the new domain (redundant video frames, suboptimal
actions). The conditioning surface is universal; the learned policy
generalizes.

## §4 ACT halting computation

Per-position halting in UT (Graves 2016 ACT applied per-symbol):

```python
# Per loop iteration t, per position i:
halting_logit[i, t] = halting_head(H[i, t])
halting_prob[i, t]  = sigmoid(halting_logit[i, t])
cumulative[i]      += halting_prob[i, t]

if cumulative[i] >= halt_threshold:    # e.g. 1.0 - epsilon
    halted[i] = True
    H[i, T_max] = H[i, t]              # copy state forward
```

When all positions halt OR `t == T_max`, encoder output is
`H[:, T_max]`.

**Halting probability ponder cost** (regularizer): `pondercost = sum_i (n_iters_i + remainder_i)` added to loss with small weight. Encourages early halting where possible.

**Modern framing**: ACT can be replaced with simpler **convergence
detector** — compare `||H_t - H_{t-1}||` against threshold. Less
expressive but no extra learned head.

## §5 Parcae scaling laws

First predictable scaling laws for looped training (Parcae 2026):

### Training scaling

For a fixed FLOP budget with fixed parameters:

- **Increasing mean recurrence** AND **reducing token count** yields
  lower loss than minimal loops on more data
- **Optimal recurrence** and **optimal token count** both follow
  power laws with consistent exponents across scales

### Inference scaling

More test-time loops → better quality following a **predictable
saturating exponential decay**. Gains real but diminishing. Mirrors
the inference-time scaling observed for chain-of-thought.

### Empirical anchor

At 770M parameters, a Parcae-trained looped model achieves the
downstream quality of a **1.3B fixed-depth Transformer** trained on
the same data. Roughly **half the parameters for the same quality**.

Implication: a looped model's apparent capability has two sources —
parameter count AND loop depth. Reporting parameter count alone
understates the model.

## §6 OpenMythos variant config table

Pre-configured RDT scales (from `OpenMythos.MythosConfig`):

| Variant | dim | n_experts | expert_dim | loop_iters | context | max_output |
|---|---|---|---|---|---|---|
| mythos_1b  | 2048  | 64  | 2048  | 16 | 4k  | 4k   |
| mythos_3b  | 3072  | 64  | 4096  | 16 | 4k  | 4k   |
| mythos_10b | 4096  | 128 | 5632  | 24 | 8k  | 4k   |
| mythos_50b | 6144  | 256 | 9728  | 32 | 8k  | 4k   |
| mythos_100b| 8192  | 256 | 13568 | 32 | 1M  | 128k |
| mythos_500b| 12288 | 512 | 23040 | 48 | 1M  | 128k |
| mythos_1t  | 16384 | 512 | 34560 | 64 | 1M  | 128k |

**Loop count scales with model size** — bigger models get more
iterations, not just more width. mythos_1t at 64 iterations is the
upper bound the open-source reconstruction proposes.

Training defaults (per `OpenMythos.training/3b_fine_web_edu.py`):
AdamW, FineWeb-Edu (sample-10BT), GPT-OSS-20B tokenizer, PyTorch
DDP, bfloat16 on H100/A100, linear warmup 2000 steps → cosine
decay, ~30B tokens (Chinchilla-adjusted for looping).

## §7 Loop-index RoPE encoding

Cheap addition to differentiate same-weight iterations:

```python
def loop_index_embedding(t: int, T: int, d_model: int) -> Tensor:
    """RoPE-style sinusoidal embedding of loop index t ∈ [0, T)."""
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, d_model, 2) / d_model
    )
    angles = t * freqs                        # [d_model // 2]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)])
    return emb                                # [d_model]

# Inside the recurrent block:
loop_emb = loop_index_embedding(t, T, d_model)
h = h + loop_emb.unsqueeze(0)                 # broadcast over batch
# ... rest of transformer block as usual
```

Properties:

- **Zero parameter cost** (deterministic encoding, like position RoPE)
- **Iterations functionally distinct** without breaking weight tying
- Compatible with input injection: `h_{t+1} = A·h_t + B·e + loop_emb(t) + Transformer(h_t, e)`
- Compatible with ACT halting: halted positions just keep the loop_emb at the iteration they halted at

Implementation cost on D5 (`recurrent_substrate.py`): one constant
encoding function + one add inside the iteration loop. ≤1 day.

## §8 MoE shared-expert mechanics

Sparse MoE in the Recurrent Block, with two complementary expert
populations:

### Routed experts (sparse)

```python
# n_experts experts total, each 1/m the size of a normal FFN
# top-K selected per token
router_logits = router(token)                 # [n_experts]
router_probs  = softmax(router_logits + bias) # bias for collapse prevention
top_k_idx     = topk(router_probs, K)         # [K]
out = sum(router_probs[i] * experts[i](token) for i in top_k_idx)
```

### Shared experts (dense, always-on)

```python
# n_shared_experts always active, no routing
shared_out = sum(shared_experts[i](token) for i in range(n_shared))
final = out + shared_out
```

### Routing-collapse prevention

Without intervention, routers learn to send all tokens to a small
set of "popular" experts. The fix: **dynamic bias term on router
logits**, adjusted online during training to keep load balanced
across experts. Doesn't distort the loss signal (unlike auxiliary
load-balance losses) and reaches healthier final routing.

Per-loop expert subset selection is a **side benefit** — as `h_t`
evolves across iterations, router output drifts, naturally
selecting different experts at different depths.

## §9 Continuous depth-wise batching

Downstream consequence of weight-tied recurrence: because all
tokens share the same recurrent block, the model can **exit the
loop at different depths for different tokens or sequences** —
processing easy inputs quickly and hard inputs with more iterations,
all within the same batch.

```
Batch:                Token A: 4 iterations (easy)  → output
                      Token B: 12 iterations (hard) → output
                      Token C: 6 iterations         → output
                      ...

All processed together; ACT-halted tokens skip subsequent loop iterations
```

Theoretical analysis (OpenMythos / Mixture-of-Depths Attention):
**2-3× improvements in inference throughput**. For a deployed model
serving many users simultaneously, this would be a substantial
efficiency gain.

Implementation requires kernel support for variable per-token
iteration count within a batch — non-trivial in current PyTorch but
feasible in custom CUDA / Triton.

## §10 Repo cross-refs (existing code to read before any implementation)

| Net-new mechanism | Existing code to extend / coexist with |
|---|---|
| Loop-index RoPE on D5 | `calm/llm_computer/recurrent_substrate.py` (D5 `n_iterations` loop) |
| ACT halting | Same file; add `halting_head` parameter, modify iteration loop |
| Parcae stability | Only relevant if D5 grows `B·e` injection; new module needed |
| NAMM (skip) | Would need `gemma_substrate.py` per-layer attention hook |
| RDT Tier-3 card for L24 | New card class subclassing `Small2DTransformer`; install via `CardSlot` (`Substrate.md` §"Card Installation") |

Comparison points (already shipped, **not** to be replaced):

- `calm/llm_computer/copy_augmented_delta.py` — DT Householder fast-weight recurrence on **sequence axis** (orthogonal to RDT depth axis). Both can coexist on the same card.
- `calm/llm_computer/facades/multistep_reasoning.py` (or equivalent in `programs/`) — `MultiStepReasoningFacade` solves multi-step CALC via parse → safe_eval → step-through bias. Complements RDT cards (which would target multi-step *reasoning* with no safe_eval reduction).

## §11 References (cited in this file)

- Parcae stability fix + scaling laws: `OPEN_MYTHOS.md` §"The Stability Problem" + §"Scaling Laws for Looped Models"; arXiv 2604.12946 (Prairie 2026)
- NAMM 3-step pipeline: `EUTM.md` §"Learning a Memory Framework with Evolution"
- ACT halting: `UT.md` §2.2 "Dynamic Halting"; Graves 2016 (cited therein)
- Continuous depth-wise batching: `OPEN_MYTHOS.md` §"Continuous Depth-wise Batching"; Mixture-of-Depths Attention arXiv 2603.15619
- MoE shared-expert design: `OPEN_MYTHOS.md` §"Mixture of Experts"; arXiv 2401.06066 (DeepSeek fine-grained expert segmentation)
- Per-loop LoRA option: `OPEN_MYTHOS.md` §"Parameter Reuse via LoRA Adaptation"; arXiv 2410.20672 (Bae 2024)
