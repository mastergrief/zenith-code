# Embed Intelligence — Delivery paths from compiled card to Gemma's output

A compiled card with the right answer is useless if Gemma doesn't
emit that answer. This file maps the mechanisms for routing card
output into Gemma's token stream, the tradeoffs, and what's been
measured.

For the install-mode (where the card's computation LIVES) see
`Substrate.md` §"Card Installation." This file is about the OUTPUT
path: card computation → Gemma's vocab logits → emitted tokens.

## Three delivery mechanisms

| Mechanism | Where it acts | Scope |
|---|---|---|
| **VerificationHook** | Head logits, after softcapping | Bias one Gemma token id by +boost |
| **Token-embedding projection** | Residual at position -1, late layer | Add Gemma's `token_embd[answer_id]` to residual. Downstream head turns into logit bias. |
| **Step-through digit bias** | Head logits, once per generation step | Bias the next-expected token at every decode step for a multi-token answer |

Round 9 measurements showed (1) and (2) are functionally equivalent
for single-token answers at late layers. Both are head-level biases
via different mathematical routes. (3) is the generalization for
multi-token answers — required for anything beyond single-digit.

## VerificationHook (`gemma_substrate.py`)

```python
hook = VerificationHook(card_slot, vocab_mapping={card_slot_vocab_id: gemma_token_id},
                         boost=50.0, min_margin=0.5)
gemma.verification_hooks.append(hook)
```

- Runs after head + logit softcapping (so it can override Gemma's
  natural argmax).
- Reads `card_slot.last_output[0, -1]`, argmaxes, looks up in
  `vocab_mapping`, adds `boost` to the corresponding Gemma logit.
- `min_margin` gates: only fires if `(peak - median) >= min_margin`
  on the card output. Prevents firing on unmatched keys (recall card
  with no match returns all-zero logits → `min_margin=0.5` → silent).
- Without `min_margin` guard, unrelated prompts (Paris/Berlin/Rome)
  can be corrupted by the card's default argmax — Round 6 bug.
- Single-token only. Fires once per forward pass.

Use for: single-token verified answers where the card's vocab is
small (digits, yes/no, enum slots).

## Token-embedding projection (Round 9)

```python
# In the CardSlot writer:
slot_argmax = int(card_out[0, -1].argmax())
gemma_token = vocab_mapping[slot_argmax]
embd = gemma.token_embd[torch.tensor([gemma_token])]  # (1, d_model)
embd = embd * math.sqrt(gemma.config.d_model)  # match Gemma's scale
h[..., -1, :] = h[..., -1, :] + strength * embd.squeeze(0)
```

Mechanism: write Gemma's OWN token embedding for the verified answer
into the residual at position -1. Gemma's downstream layers process
the residual as usual; the additive embedding shifts the final head's
argmax toward the injected token.

**Critical findings (must be known before using):**

- **Only works at late layers (33-41).** Round 10a ruled out early
  install — projection at layer 1 gives 0/7 with regressions; layer
  33 gives 4/7 clean.
- **Strength behavior is binary** (Round 10b). α < 1: silent.
  α ≥ 1: fires cleanly. No upper break point up to 50× tested.
  Default `strength=1.0` matches Gemma's native token-embedding scale.
- **Must scale by `sqrt(d_model)`** — Gemma's own `token_embd`
  lookup is multiplied by `sqrt(d_model)` at forward ingress;
  injected embeddings must follow the same convention.
- **Does NOT make Gemma "reason with the injected context"**
  (Round 10c). Continuation after injection is noisy — Gemma is
  pushed off-distribution. The mechanism is a late-layer head bias,
  not deep integration.

Use for: single-token verified answers where you want to avoid
adding a hook and prefer a pure residual-level intervention.
Essentially equivalent to VerificationHook in effect.

## Step-through digit bias (Round 11)

The mechanism that enables **multi-token verified answers.** Required
for any answer longer than one Gemma BPE token (multi-digit numbers,
multi-word phrases, code snippets).

```python
# Decompose verified answer into Gemma BPE token sequence
digit_ids = tokenizer.encode(str(verified_int))  # [<bos>, ▁, 3, 9, 1]
digit_ids = strip_bos(digit_ids)                  # [▁, 3, 9, 1]

# Autoregressive decode with per-step bias
for step in range(max_tokens):
    logits = gemma.forward(next_input, ...)
    if step < len(digit_ids):
        logits[0, -1, digit_ids[step]] += boost
    next_tok = int(logits[0, -1].argmax())
```

Key patterns:
- Each step biases ONE token — the next expected in the digit chain.
- After the chain, stop biasing; let Gemma continue naturally.
- Optional `wait_marker_tokens`: only start biasing AFTER Gemma
  emits a marker token (`▁`, `=`, `\n`) — useful if Gemma's preferred
  output has a prefix before the answer.
- Compatible with Gemma's BPE: 3-digit numbers tokenize as
  `[▁, d0, d1, d2]` so there are 3-4 biasing steps per answer.

Use for: multi-digit arithmetic, multi-word factual answers,
structured outputs where the card knows the answer as a single value
but Gemma's BPE tokenizes it across multiple tokens.

Round 11 measurement: baseline 5/10 → facade 10/10 on 2-digit ×
prompts. Three genuine arithmetic fixes (17×23, 47×19, 45×15).

## Which mechanism to use

| Situation | Mechanism |
|---|---|
| Answer is a single Gemma BPE token (digit 0-9, yes/no, short enum) | VerificationHook or token-embd projection |
| Answer is multi-digit (arithmetic > 9) | Step-through digit bias |
| Answer is multi-word (facts, code) | Step-through token bias |
| Card has an "unmatched key" state (recall card with no match) | Always use `min_margin ≥ 0.5` to suppress spurious fires |

## Avoiding regressions

**Unrelated prompts** (prompts whose PT parse fails or whose recall
key doesn't match any stored entry) must NOT trigger the delivery
mechanism. Gemma's natural output on them must flow through
unmodified.

Two guards, use both:

1. **Parse-state flag** in the CardSlot writer:
   ```python
   if self._parse_ok:
       # inject / bias
   else:
       card_out.zero_()  # in-place — hook reads last_output
   ```
   Critical that `card_out` is zeroed IN PLACE (CardSlot assigns
   `slot.last_output = card_out` AFTER the writer, so in-place
   mutations propagate; reassignment doesn't).

2. **min_margin on VerificationHook** (0.5 is a safe default):
   ```python
   VerificationHook(..., min_margin=0.5)
   ```
   Silences the hook when the card's peak logit is within 0.5 of its
   median (no confident answer).

Round 6 and Round 11 both needed both guards to avoid regressing
"The capital of France is Paris."

## Known deliverability surface

Round 9: single-token bias via hook or projection.
Round 11: multi-token bias via step-through.

Not yet tested:
- **Multi-position injection** (inject a SEQUENCE of embeddings at
  trailing positions in one forward, instead of per-step) — would
  let Gemma see the full verified answer as established context
  before its first emission. Speculative; may or may not deliver.
- **Feature-directed projection** — instead of token embeddings,
  inject along Sparse Autoencoder feature directions. Requires SAE
  training on Gemma 4 E4B which hasn't been done. Research-level.
- **Autoregressive facade re-firing** — re-run PT + card at each
  decode step with the growing context as input. Expensive
  (O(generation_length × PT_forward)) but would let the facade
  adapt its verified answer to what Gemma's emitted so far.

## Related rules

- `capability_gain.md` — measurement discipline
- `Substrate.md` — where the card's computation lives (install modes)
- `tracing_intelligence.md` — what's compilable in principle
- `workflow.md` — two-measurement discipline
