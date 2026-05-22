# HRM-Text-1.58 Tokenizer Contract

**Phase 3 deliverable for task #51 / board task `1779460303130-742c8cbd`.** Locks
the broad fixed tokenizer for the entire HRM-Text-1.58 checkpoint chain,
starting at Phase 3 R0 and continuing through R7. Codex msg 1779460698439
locked route A1 (byte-level UTF-8) per gabe's "broad and fixed" +
"this could replace gemma completely" framing.

## Ambition vs claim (wording gate)

Per codex wording gate (msg 1779460586464): the phrase "could replace Gemma"
captures the long-horizon ambition / hypothesis driving this tokenizer
choice. It is **NOT a current claim**. The actual falsifier remains
curriculum retention and generalization across R0-R7. If broad-tokenizer
R0-R3 shows primitive acquisition and retention, we can talk about scaling
toward general replacement. If not, we diagnose curriculum or model first.

## Choice: byte-level UTF-8 (Route A1)

Rationale: zero future OOV for any string content (math, English, code,
non-Latin, emoji, arbitrary internet text). The parameter cost is
negligible at Tier B (260-vocab embed_tokens + lm_head adds ~52K params
on top of 30M = 0.17%). ASCII text stays 1 token per char. Non-ASCII
becomes multi-token (e.g., `é` = 2 bytes). For Phase 3's math + English +
code surface, expected non-ASCII is ~0%.

### Vocab specification

Total vocab = 260 tokens, deterministic order, NEVER built from corpus.

| id range | content |
|---|---|
| 0 | `<pad>` |
| 1 | `<bos>` |
| 2 | `<eos>` |
| 3 | `<sep>` |
| 4-259 | byte values 0x00..0xff (rendered as `<byte:00>`..`<byte:ff>` in `vocab_as_list()`) |

### Normalizer contract

Normalizer version: `byte_utf8_v1`. **Identity normalizer** — no semantic
rewriting (no lowercasing, no Unicode NFKC normalization, no whitespace
collapsing). Input text is passed verbatim to `.encode("utf-8")`.

This is a deliberate choice for source-faithful byte-level: any
normalization (e.g., NFKC) would make the tokenizer lossy under
re-encoding and could break checkpoint compatibility if normalization
behavior changes across Python versions or libraries.

### Encoding / decoding

**Encode**:
```python
def encode(text: str) -> list[int]:
    return [b + 4 for b in text.encode("utf-8")]
```

**Decode**:
```python
def decode(ids: Iterable[int], stop_at_eos: bool = True) -> str:
    # Byte ids are flushed as bytes(...).decode("utf-8", errors="replace")
    # so partial multi-byte sequences become '?' instead of raising.
```

For ASCII code/math/English (Phase 3 corpus), round-trip is exact.
For non-ASCII (not expected in Phase 3 scope), error policy is `replace`
to avoid runtime crashes on partial sequences.

## Cross-arc compatibility

### Phase 2 GSM8k-tokenizer ckpts (NOT load-from parents for Phase 3)

Phase 2 ternary Tier B / Tier A ckpts use the Gsm8k char tokenizer
(vocab=98, ordered by `from_corpus(GSM8k_train+val)`). The
BroadTokenizer's vocab (260, deterministic byte order) is INCOMPATIBLE
both in size and in id mapping. Loading a Phase 2 ckpt into a Phase 3
broad-vocab model would either:
- shape-fail on embed_tokens (98 × 512 vs 260 × 512), OR
- shape-succeed only if a checkpoint surgery downsamples vocab dims —
  not implemented, would corrupt id mapping silently.

**Phase 3 R0 starts from RANDOM INIT** with the broad tokenizer. Phase 2
ckpts remain as **reference baselines** for cross-arc comparison (e.g.,
"Phase 3 R0's GSM8k probe accuracy at random init + light training vs
Phase 2 ternary Tier B 0-1/50 plateau").

`scripts/train_hrm_text_158.py:_validate_load_from_ckpt_compat` enforces
this with a hard-fail error message that explicitly mentions the
Phase 2 → Phase 3 boundary.

### Inter-rung ckpt chaining (within Phase 3)

`--load-from R(n-1)_best.pt` is the canonical path for rung continuation.
Validation hard-fails on:
1. `gsm8k_char_vocab` list mismatch (full equality required)
2. `gsm8k_normalizer_version` mismatch (must both be `byte_utf8_v1`)
3. `use_ternary_bulk` mismatch (cannot switch FP <-> ternary mid-chain)
4. Architecture field mismatch (`hidden_size`, `n_layers`, `num_heads`,
   `H_cycles`, `L_cycles`, `half_layers`, `expansion`, `max_seq_len`,
   `attn_type`, `init_type`, `norm_type`)

`--load-from` loads `model_state` ONLY. Optimizer state + LR schedule
reset per rung. Curriculum builds primitives via WEIGHTS continuity;
optimizer momentum + LR-warmup are intentionally fresh each rung to
isolate curriculum effect.

## Sequence-length impact

Byte-level eliminates OOV but may increase sequence length on non-ASCII
text. Step 0 includes a length-histogram gate that compares
BroadTokenizer vs GSM8k char tokenizer on GSM8k train/val at
`max_len=384`:

- Expected: identical for ASCII-only rows (~100% of GSM8k)
- Watch for: rows with apostrophe-like Unicode (curly quotes) or
  em-dashes that would multi-byte under UTF-8

Test asserts: dropped/too_long counts under BroadTokenizer at max_len=384
are within ±5% of GSM8k char tokenizer. Larger drift surfaces a
data-side issue before R0 training.

## Future-compatibility

A new normalizer_version triggers a chain break. If we ever change
normalization (e.g., add NFKC), the new tokenizer gets a fresh
normalizer_version, and old ckpts cannot be `--load-from`'d into models
using the new normalizer.

Migration path if needed: explicit embedding/lm_head migration in a
dedicated phase, NOT silent tokenizer drift.

## Capability blocks + checkpoint chain (design constraint)

Gabe direction (verbatim across two ai-room messages on 2026-05-22):
"yeah and each run builds on previous block/layer?" + "but they all
build in blocks/layers and checkpoints". This locks the curriculum
architecture as a **checkpoint chain of capability blocks**:

- "blocks/layers" here = **skill/curriculum strata + checkpoint strata**,
  NOT necessarily neural-layer additions per rung.
- R0-R3 architecture (hidden_size, n_layers, num_heads, H/L_cycles,
  half_layers, expansion, max_seq_len, attn_type, init_type, norm_type)
  stays FROZEN to isolate the curriculum signal from architecture
  noise. `validate_load_from_ckpt_compat` enforces this with hard-fail
  on any arch-field drift.
- Each rung's best ckpt is the **parent for the next rung**. Replay
  mix + retention probe verify the chain holds (G2 gate: no prior rung
  drops > 10% absolute).
- Future physical-layer / adapter / modality-codebook additions are
  deliberate **compatibility events** with their own checkpoint
  boundary + retention replay, NOT silent mid-chain mutations.

The asset being built is the checkpoint chain itself — each block
adds one capability, every prior block must keep working in the
combined network.

## Future multimodal extension (non-claim, scope-deferred)

Gabe direction (verbatim ai-room msg `1779461640781`): "can hrm be
trained multi-modally?". Answer: **yes in principle**, but NOT in
R0-R3 scope. The byte-level UTF-8 tokenizer solves text/code OOV;
it does NOT solve multimodality. Multimodality is a deliberate
future block, not implicit in the current tokenizer choice.

Two candidate paths (both compatible with the checkpoint chain
above; choice deferred to post-R7):

| Path | Mechanism | Tradeoff |
|---|---|---|
| **Discrete modality tokens** | Image patches / VQ tokens, audio codec tokens, action/sensor tokens become token streams with modality delimiters; HRM recurrence over a unified sequence. Reserved id ranges or a separate embedding table + typed tokens. | Preserves H/L recurrent core + broad-fixed-tokenizer idea. Minimal architectural deviation. **Lowest-risk path.** |
| **Per-modality encoders/adapters** | Visual / audio encoders project into HRM's hidden space; text tokens as one stream and modality embeddings as typed spans. | Higher capability ceiling. Heavier architectural deviation; harder to keep source-faithful and native 1.58. |

Suggested future shape (post-R7, NOT in Phase 3 scope):
1. Text / math R0-R7 checkpoint chain (current scope)
2. Code curriculum starting from best text/math ckpt, replay from prior rungs
3. Multimodal discrete-token block from that ckpt, adding reserved
   modality token range OR codebook OR adapter contract, with
   text/code replay
4. Joint cross-modal tasks ONLY after unimodal modality primitives are
   acquired in isolation

**Explicit non-claim**: this section captures the architecture's
*compatibility* with future multimodality. It does NOT claim the
current byte-level tokenizer or R0-R3 curriculum is multimodal-ready.
The wording gate above (ambition vs claim) applies here too —
multimodal capability is a hypothesis to test after curriculum
retention is validated, not something the substrate ships with.

## Test surface

Implemented in `calm/llm_computer/tests/test_hrm_text_158_curriculum.py`:

- `test_broad_tokenizer_vocab_deterministic` — same instance two times = same vocab list
- `test_broad_tokenizer_vocab_size_260` — 4 specials + 256 bytes
- `test_broad_tokenizer_ascii_roundtrip` — encode/decode bit-exact on code-smoke ASCII strings
- `test_broad_tokenizer_special_ids` — pad=0, bos=1, eos=2, sep=3
- `test_broad_tokenizer_assert_corpus_covered_never_raises` — byte-level is OOV-free
- `test_broad_tokenizer_encode_example_shape` — `<bos> q <sep> target <eos>` contract
- `test_load_from_vocab_mismatch_hard_fails` — Phase 2 ckpt rejected against broad-vocab config
- `test_load_from_normalizer_mismatch_hard_fails`
- `test_load_from_ternary_flag_mismatch_hard_fails`
- `test_load_from_arch_mismatch_hard_fails` (each arch field)
- `test_code_smoke_round_trip` — all 20 CODE_SMOKE_STRINGS encode + decode exactly
- `test_length_histogram_broad_vs_char` — GSM8k length under BroadTokenizer vs GSM8k char tokenizer, asserts dropped/too_long within ±5%

Step 0 commit ships these tests + the contract.
