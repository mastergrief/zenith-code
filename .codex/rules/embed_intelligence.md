---
paths:
  - "calm/llm_computer/facades/**"
  - "calm/llm_computer/*installer*.py"
  - "calm/llm_computer/tied_embedding.py"
  - "scripts/test_token_embd_projection.py"
  - "scripts/test_multi_token_projection.py"
  - "scripts/test_projection_strength.py"
  - "scripts/*projection*.py"
---

# Embed Intelligence — Delivery paths from compiled card to Gemma's output

A compiled card with the right answer is useless if Gemma doesn't
emit that answer. This file maps the mechanisms for routing card
output into Gemma's token stream, the tradeoffs, and the rules.

For the install-mode (where the card's computation LIVES) see
`Substrate.md` §"Card Installation". This file is about the OUTPUT
path: card computation → Gemma's vocab logits → emitted tokens.

> Historical receipts (delivery-mechanism validation rounds,
> threshold-calibration arc, discipline-scope expansion commits,
> first-token-hook ruled-out measurement): see
> `MEMORY/atlas/embed_intelligence_arc.md`.

## Three delivery mechanisms

| Mechanism | Where it acts | Scope |
|---|---|---|
| **VerificationHook** | Head logits, after softcapping | Bias one Gemma token id by +boost |
| **Token-embedding projection** | Residual at position -1, late layer | Add Gemma's `token_embd[answer_id]` to residual. Downstream head turns into logit bias. |
| **Step-through digit bias** | Head logits, once per generation step | Bias the next-expected token at every decode step for a multi-token answer |

(1) and (2) are functionally equivalent for single-token answers at
late layers — both are head-level biases via different mathematical
routes. (3) is the generalization for multi-token answers — required
for anything beyond single-digit.

## VerificationHook

```python
hook = VerificationHook(card_slot,
                        vocab_mapping={card_slot_vocab_id: gemma_token_id},
                        boost=50.0, min_margin=0.5)
gemma.verification_hooks.append(hook)
```

- Runs after head + logit softcapping (so it can override Gemma's
  natural argmax).
- Reads `card_slot.last_output[0, -1]`, argmaxes, looks up in
  `vocab_mapping`, adds `boost` to the corresponding Gemma logit.
- `min_margin` gates: only fires if `(peak - median) >= min_margin`
  on the card output. Prevents firing on unmatched keys (recall
  card with no match returns all-zero logits → silent).
- Without `min_margin` guard, unrelated prompts can be corrupted by
  the card's default argmax.
- Single-token only. Fires once per forward pass.

### Threshold-calibration rule

**Tune `min_margin` per-card AND per-input-distribution bucket**,
not to a fixed 0.5. For retrieval cards on distractor-confused
corpora, margins are input-shape-dependent — e.g. margin distribution
differs substantially by N in MQAR, where higher N produces lower
p5 margins even when standalone accuracy is 100%.

Process:
1. Run the card standalone on a representative corpus **per
   input-distribution bucket**.
2. Plot (peak − median) margin distribution per bucket.
3. Pick threshold below the lowest p5 across buckets.

A single margin threshold that's correct for one bucket may
over-gate another if input shape varies. See `delta_rule.md`
§"retrieval-card install pattern" for the canonical MQAR calibration arc.

### `write_margin` == `min_margin` alignment rule

`card_output_fn` independently writes to the residual stream; without
a margin gate the write happens even when hook is silent, shifting
Gemma's head projection. Keep both gates aligned (same numeric value)
unless you have a specific reason to let one fire without the other.

Use for: single-token verified answers where the card's vocab is
small (digits, yes/no, enum slots).

## Token-embedding projection

```python
# In the CardSlot writer:
slot_argmax = int(card_out[0, -1].argmax())
gemma_token = vocab_mapping[slot_argmax]
embd = gemma.token_embd[torch.tensor([gemma_token])]  # (1, d_model)
embd = embd * math.sqrt(gemma.config.d_model)  # match Gemma's scale
h[..., -1, :] = h[..., -1, :] + strength * embd.squeeze(0)
```

Mechanism: write Gemma's OWN token embedding for the verified answer
into the residual at position -1. Downstream layers process as usual;
the additive embedding shifts the final head's argmax toward the
injected token.

**Critical rules**:

- **Only works at late layers (33-41).** Early install projects into
  layers that still process the residual as input, not as prediction
  pre-image — projection at layer 1 regresses; layer 33+ fires clean.
- **Strength behavior is binary**: α < 1 silent, α ≥ 1 fires cleanly.
  No upper break point observed up to 50×. Default `strength=1.0`
  matches Gemma's native token-embedding scale.
- **Must scale by `sqrt(d_model)`** — Gemma's own `token_embd` lookup
  is multiplied by `sqrt(d_model)` at forward ingress; injected
  embeddings must follow the same convention.
- **Does NOT make Gemma "reason with the injected context".**
  Continuation after injection is noisy — Gemma is pushed
  off-distribution. The mechanism is a late-layer head bias, not
  deep integration.

Use for: single-token verified answers where you want to avoid
adding a hook and prefer a pure residual-level intervention.
Essentially equivalent to VerificationHook in effect.

## Step-through digit bias

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
- Optional `wait_marker_tokens`: only start biasing AFTER Gemma emits
  a marker token (`▁`, `=`, `\n`) — useful if Gemma's preferred
  output has a prefix before the answer.
- Compatible with Gemma's BPE: 3-digit numbers tokenize as
  `[▁, d0, d1, d2]` so there are 3-4 biasing steps per answer.

Use for: multi-digit arithmetic, multi-word factual answers,
structured outputs where the card knows the answer as a single value
but Gemma's BPE tokenizes it across multiple tokens.

**Extension to N-op chains**: `MultiStepReasoningFacade` parses NL
infix (e.g. `"2 + 3 × 5 - 7"`) with parens and mixed precedence,
evaluates via `safe_eval` to the final answer, then emits one
step-through digit bias per intermediate AND final value. The
mechanism generalizes from single-op to N-op composition —
step-through biasing is the right embed mechanism for any verifier
that produces a multi-token numeric answer.

### `▁`-strip + POST_BIAS_BUDGET discipline

For prompt-terminators that end in a SPACE (e.g. `"Answer: "` with
trailing space), `tokenizer.encode(str(n))` returns
`[BOS, ▁, d0, d1, ...]` where `▁` is id 236743. The naïve
`strip_bos` path leaves `▁` in the bias chain, so step-0 biases a
SPACE — Gemma's natural `0` token at that position has logit ~57-66
and +50 boost on `▁` can't flip it. Result: bias starts one step
late, and the answer gets "0"-prefixed gibberish.

**Rule (applied to all integer-answer facades)**:

1. Strip BOTH BOS (id=2) AND leading `▁` (id=236743) from bias
   tokens. Step-0 then biases the first digit directly.
2. After the bias chain exhausts, Gemma sticks in a same-digit loop
   (a long run of one digit pasted after the real answer). Cap
   continuation at `POST_BIAS_BUDGET=4` natural tokens then break.
3. `_parse_int` caps digit-run matches at 12 chars to defeat any
   residual loop that survives POST_BIAS_BUDGET.

**Scope**: applied in integer-answer facades (`number_theory.py`,
`numeric_encode.py`, all `recursion.py`-generated facades via shared
`_TEMPLATE`). NOT backported to facades whose answer shapes don't
trigger the same-digit loop in practice — backport only if a new
facade exhibits the bug.

**For text-answer facades**: do NOT strip `▁` because the diagnosis
text begins with a capital letter and Gemma's BPE often merges the
leading space into the first-word token (e.g. `▁Type` as a single
token). The `▁`-strip is correct ONLY for integer-answer facades
whose bias is `[▁, digit0, digit1, ...]`.

**Boost tuning for stubborn priors**: some domains' code-analysis
format prior overwhelms step-through bias. Retry with
`boost * 3.0 = 150.0` plus in-context answer injection as last
resort. Some edge cases resist all step-through boost and need
different mechanism (prompt reshape or pure-DB bypass).

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
mechanism. Gemma's natural output must flow through unmodified.

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

2. **min_margin on VerificationHook** (0.5 is a safe default for
   single-token cards; higher for retrieval cards with N-dependent
   margins):
   ```python
   VerificationHook(..., min_margin=0.5)
   ```
   Silences the hook when the card's peak logit is within margin
   of its median (no confident answer).

Both guards together prevent regressions like "The capital of France
is Paris" getting overridden by an unrelated-card's default argmax.

## Known deliverability surface

- **Single-token bias via hook or projection**: proven.
- **Multi-token bias via step-through**: proven. Generalizes to
  N-op composition and text-answer recall.

Not yet tested:

- **Multi-position injection** (inject a SEQUENCE of embeddings at
  trailing positions in one forward, instead of per-step) — would
  let Gemma see the full verified answer as established context
  before its first emission. Speculative.
- **Feature-directed projection** — instead of token embeddings,
  inject along SAE feature directions. Requires SAE training on
  Gemma 4 E4B. Research-level.
- **Autoregressive facade re-firing** — re-run PT + card at each
  decode step with the growing context as input. Expensive
  (O(generation_length × PT_forward)) but would let the facade
  adapt its verified answer to what Gemma's emitted so far.

## Ruled out

**First-token bias for code tasks.** First-token bias
(`VerificationHook(vocab_mapping=PER_MARKER_TARGETS, boost=50,
min_margin=0.5)` where targets are `"def"/"class"` per-problem)
regresses on code prompts. Root cause: Gemma's first-token on code
prompts is uniformly confident (margin ~6-9 on whitespace/fence
openers), so `min_margin=0.5` never gates — hook always fires on
HIT, forces `def`/`class` before Gemma emits the fence →
code-without-fence → extractor fails.

**Rule**: first-token bias is the wrong delivery mechanism for
code. Either (a) mid-generation per-token hooks that fire on token
patterns (e.g. bias only after `def` is emitted), or (b) post-generation
AST-walker rewrite (tier-2 compiled card, no decode-time
intervention — see `compute_facades.md` for the walker pattern).
Confidence-gated hooks also failed at this site — the measurable
margin doesn't correlate with "Gemma is uncertain about format."

## Related rules

- `capability_gain.md` — measurement discipline
- `Substrate.md` — where the card's computation lives (install modes)
- `compute_facades.md` — decode-path facade tier-2 pattern
- `tracing_intelligence.md` — what's compilable in principle
- `workflow.md` — two-measurement discipline
- `MEMORY/atlas/embed_intelligence_arc.md` — delivery-mechanism receipts
