"""SubstrateLM — Small2DTransformer as a decoder-only language model.

The session-27 handoff scoped this at §H.2: a Small2DTransformer trained on
`(prompt, response)` pairs, proving the substrate hosts general LM behavior.
If it works at any scale, the architectural path to unifying compiled
programs + HRM specialists + general LM into one `.pt` file is open.

MVP deliberately small (~1-10M params) — goal is "does the substrate learn
coherent text?" not "beats Gemma." Uses existing Claude-authored corpus
(`agents/distill/data/claude_reasoning.jsonl`) instead of distilling fresh
from Gemma — that data is already curated, ~910 high-quality examples.

Decoder-only GPT style. Causal mask (softmax), learned positional
embeddings, vanilla ReGLU FFN. Same primitives as compiled programs so
fusion works (see experiment_fast_weights_fusion.py).

Design choices:
  - Tokenizer: BPE trained on the corpus, vocab=8192. Small enough to
    keep embedding params modest (8192 × d_model), large enough to cover
    `<think>` tags + code tokens cleanly.
  - Chat format: `<|sys|>{sys}<|user|>{user}<|asst|>{asst}<|eos|>`.
    Loss masked to assistant tokens only (train_on_responses_only pattern).
  - Max sequence: 512 tokens. Truncation for longer responses; typical
    corpus examples fit under this.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from calm.llm_computer.model import Small2DConfig, Small2DTransformer


# ----- Special tokens used in chat formatting -----
SYS_TOKEN = "<|sys|>"
USER_TOKEN = "<|user|>"
ASST_TOKEN = "<|asst|>"
EOS_TOKEN = "<|eos|>"
PAD_TOKEN = "<|pad|>"
SPECIAL_TOKENS = [SYS_TOKEN, USER_TOKEN, ASST_TOKEN, EOS_TOKEN, PAD_TOKEN]


# ----- Corpus loading -----

def load_messages(jsonl_paths: Iterable[str | Path]) -> list[list[dict]]:
    """Load all messages-format examples from one or more jsonl files."""
    out = []
    for path in jsonl_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                if "messages" in ex:
                    out.append(ex["messages"])
    return out


def format_chat(messages: list[dict]) -> tuple[str, int]:
    """Flatten a messages list into training text + byte index where the
    assistant response starts (for loss masking)."""
    parts = []
    asst_start_char = -1
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            parts.append(f"{SYS_TOKEN}{content}")
        elif role == "user":
            parts.append(f"{USER_TOKEN}{content}")
        elif role == "assistant":
            pre = "".join(parts)
            asst_start_char = len(pre) + len(ASST_TOKEN)  # first char of assistant content
            parts.append(f"{ASST_TOKEN}{content}{EOS_TOKEN}")
        else:
            # Unknown role — drop.
            continue
    return "".join(parts), asst_start_char


# ----- Tokenizer -----

def train_bpe_tokenizer(texts: Iterable[str], vocab_size: int = 8192) -> Tokenizer:
    """Train a byte-level BPE tokenizer on `texts`. Adds special tokens."""
    tok = Tokenizer(BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|unk|>"] + SPECIAL_TOKENS,
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def encode_with_mask(
    tokenizer: Tokenizer, messages: list[dict], max_len: int,
) -> tuple[list[int], list[int]]:
    """Tokenize a full chat and compute a loss mask (1 for assistant tokens,
    0 for system + user tokens). Returns (input_ids, loss_mask), both
    truncated/padded to max_len.
    """
    text, _asst_start_char = format_chat(messages)
    enc = tokenizer.encode(text)
    ids = enc.ids[:max_len]

    # Find token positions for ASST_TOKEN and EOS_TOKEN to mask loss.
    asst_tok_id = tokenizer.token_to_id(ASST_TOKEN)
    eos_tok_id = tokenizer.token_to_id(EOS_TOKEN)
    mask = [0] * len(ids)
    in_asst = False
    for i, t in enumerate(ids):
        if t == asst_tok_id:
            in_asst = True
            continue  # don't train on the asst marker itself
        if in_asst:
            mask[i] = 1
        if t == eos_tok_id:
            break
    return ids, mask


# ----- Model builder -----

@dataclass
class SubstrateLMConfig:
    """MVP defaults: ~8M params, trainable on CPU in ~1-2 hours."""
    vocab_size: int = 8192
    d_model: int = 128      # d_head = 2 with n_heads=64
    n_heads: int = 64
    n_layers: int = 4
    d_ffn: int = 512
    max_len: int = 512
    use_hard_max: bool = False


def build_substrate_lm(cfg: SubstrateLMConfig) -> Small2DTransformer:
    s2d_cfg = Small2DConfig(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ffn=cfg.d_ffn,
        max_len=cfg.max_len,
        use_hard_max=cfg.use_hard_max,
    )
    assert s2d_cfg.d_head == 2, f"d_head must be 2, got {s2d_cfg.d_head}"
    return Small2DTransformer(s2d_cfg)


# ----- Training -----

def pad_batch(
    seqs: list[list[int]], masks: list[list[int]], pad_id: int, max_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-aligned pad to max_len. Returns (input_ids, labels, loss_mask)."""
    B = len(seqs)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    loss_mask = torch.zeros(B, max_len, dtype=torch.float)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        L = min(len(s), max_len)
        input_ids[i, :L] = torch.tensor(s[:L], dtype=torch.long)
        loss_mask[i, :L] = torch.tensor(m[:L], dtype=torch.float)
    return input_ids, loss_mask


def train_substrate_lm(
    model: Small2DTransformer,
    tokenizer: Tokenizer,
    examples: list[list[dict]],
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 3e-4,
    max_len: int = 512,
    log_every: int = 50,
    seed: int = 0,
) -> None:
    """Teacher-forced next-token prediction with loss masked to assistant
    response tokens only."""
    rng = random.Random(seed)
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    # Pre-tokenize everything once for speed.
    print(f"pre-tokenizing {len(examples)} examples...", flush=True)
    cache: list[tuple[list[int], list[int]]] = []
    for msgs in examples:
        ids, mask = encode_with_mask(tokenizer, msgs, max_len)
        if sum(mask) > 0:  # keep only examples with >= 1 assistant token
            cache.append((ids, mask))
    print(f"  {len(cache)} training examples after filtering", flush=True)

    steps_per_epoch = max(1, len(cache) // batch_size)
    t0 = time.time()
    global_step = 0
    for epoch in range(epochs):
        rng.shuffle(cache)
        epoch_loss = 0.0
        epoch_tokens = 0
        for step in range(steps_per_epoch):
            batch = cache[step * batch_size: (step + 1) * batch_size]
            if len(batch) < batch_size:
                continue
            seqs = [b[0] for b in batch]
            masks = [b[1] for b in batch]
            ids, loss_mask = pad_batch(seqs, masks, pad_id, max_len)
            # Standard next-token prediction: predict ids[t+1] from ids[:t+1].
            logits = model(ids[:, :-1])          # (B, L-1, V)
            targets = ids[:, 1:]                  # (B, L-1)
            mask = loss_mask[:, 1:]               # (B, L-1) — mask applies to targets
            # Flatten
            V = logits.size(-1)
            flat_logits = logits.reshape(-1, V)
            flat_targets = targets.reshape(-1)
            flat_mask = mask.reshape(-1)
            # Per-token CE, then mask + mean
            per_tok = F.cross_entropy(flat_logits, flat_targets, reduction="none")
            masked = per_tok * flat_mask
            n_tok = flat_mask.sum().clamp(min=1.0)
            loss = masked.sum() / n_tok
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += masked.sum().item()
            epoch_tokens += n_tok.item()
            global_step += 1
            if global_step % log_every == 0:
                ppl = torch.tensor(epoch_loss / max(1, epoch_tokens)).exp().item()
                elapsed = time.time() - t0
                print(
                    f"  epoch {epoch+1}/{epochs} step {global_step} "
                    f"loss={epoch_loss/max(1,epoch_tokens):.4f} ppl={ppl:.1f}  "
                    f"{elapsed:.1f}s",
                    flush=True,
                )
        ppl = torch.tensor(epoch_loss / max(1, epoch_tokens)).exp().item()
        elapsed = time.time() - t0
        print(
            f"epoch {epoch+1}/{epochs} done: mean_loss="
            f"{epoch_loss/max(1,epoch_tokens):.4f} ppl={ppl:.1f}  {elapsed:.1f}s",
            flush=True,
        )
    model.eval()


# ----- Sampling -----

@torch.no_grad()
def generate(
    model: Small2DTransformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 40,
    seed: int = 0,
) -> str:
    """Greedy / temperature sampling from the trained model.

    Prompt is wrapped in the chat format; generation stops at EOS_TOKEN.
    """
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    asst_id = tokenizer.token_to_id(ASST_TOKEN)
    max_len = model.config.max_len

    wrapped = f"{SYS_TOKEN}You are a helpful assistant{USER_TOKEN}{prompt}{ASST_TOKEN}"
    ids = tokenizer.encode(wrapped).ids
    ids = ids[:max_len - 1]  # leave room for at least one new token

    generated: list[int] = []
    for _ in range(max_new_tokens):
        if len(ids) >= max_len:
            break
        x = torch.tensor([ids], dtype=torch.long)
        logits = model(x)[0, -1, :]  # (V,)
        logits = logits / max(temperature, 1e-6)
        # top-k filter
        if top_k and top_k < logits.size(0):
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, 1, generator=rng).item())
        if next_id == eos_id:
            break
        ids.append(next_id)
        generated.append(next_id)

    return tokenizer.decode(generated)
