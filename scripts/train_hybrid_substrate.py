"""Hybrid SubstrateLM + SubstrateHRM training — does 1+1=3?

One Small2DTransformer trained on TWO modes simultaneously:
  <|lm|>  + Claude-authored reasoning data  (~1.5K examples)
  <|hrm|> + NL math templates              (~1.5K examples)

If the joint model performs at-or-better-than separately-trained baselines
on each mode, we have evidence that:
  1. Multi-task training doesn't degrade either capability (the must-have).
  2. Cross-task representations transfer (the "1+1=3" stretch).

Baselines on disk:
  LM:  substrate_lm_mvp.pt (1.25M params, ppl 424 on held-out claude data)
  HRM: substrate_hrm_nl_best.pt (180K params, 99.1% on NL math)

Hybrid uses the LM-scale architecture (1.25M params) so capacity matches the
LM baseline. HRM mode gets 7x more capacity than its standalone counterpart
which is fine — the question is whether shared representations help.

Eval at end:
  - LM: held-out claude reasoning, perplexity on response tokens
  - HRM: held-out NL math, expression generation + analytical verification
         (parse → interpret → check value)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from calm.expression import safe_eval
from calm.hrm.nl_data import NLMathDataGenerator
from calm.llm_computer.model import Small2DConfig, Small2DTransformer
from calm.llm_computer.parse import parse_expression
from calm.llm_computer.interpret import interpret


REPO_ROOT = Path(__file__).resolve().parent.parent
LM_DATA_PATHS = [
    REPO_ROOT / "agents" / "distill" / "data" / "claude_reasoning.jsonl",
    REPO_ROOT / "agents" / "distill" / "data" / "coding_reasoning_claude.jsonl",
]
CKPT_DIR = REPO_ROOT / "calm" / "llm_computer" / "checkpoints"


# Special tokens — adds <|lm|> and <|hrm|> mode markers on top of the
# substrate_lm.py base set.
LM_MODE = "<|lm|>"
HRM_MODE = "<|hrm|>"
SYS_TOKEN = "<|sys|>"
USER_TOKEN = "<|user|>"
ASST_TOKEN = "<|asst|>"
EOS_TOKEN = "<|eos|>"
PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"
SPECIAL_TOKENS = [LM_MODE, HRM_MODE, SYS_TOKEN, USER_TOKEN, ASST_TOKEN,
                  EOS_TOKEN, PAD_TOKEN]


# ----- Corpus assembly -----

def load_lm_examples(jsonl_paths) -> list[tuple[str, int]]:
    """Return (text, asst_start_char) pairs for LM training."""
    out = []
    for path in jsonl_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                msgs = ex.get("messages", [])
                parts = [LM_MODE]   # mode marker first
                asst_start = -1
                for m in msgs:
                    role = m["role"]
                    content = m["content"]
                    if role == "system":
                        parts.append(f"{SYS_TOKEN}{content}")
                    elif role == "user":
                        parts.append(f"{USER_TOKEN}{content}")
                    elif role == "assistant":
                        pre = "".join(parts)
                        asst_start = len(pre) + len(ASST_TOKEN)
                        parts.append(f"{ASST_TOKEN}{content}{EOS_TOKEN}")
                if asst_start > 0:
                    out.append(("".join(parts), asst_start))
    return out


def build_hrm_examples(n: int, seed: int = 42) -> list[tuple[str, int]]:
    """Generate NL math problems and format them for hybrid training.
    Format: `<|hrm|> {question} = {expression} <|eos|>`
    Loss-mask asst_start = position of expression text."""
    gen = NLMathDataGenerator(seed=seed)
    problems = gen.generate(n=n)
    out = []
    for p in problems:
        prefix = f"{HRM_MODE} {p.question} = "
        text = f"{prefix}{p.expression}{EOS_TOKEN}"
        out.append((text, len(prefix)))
    return out


# ----- Tokenizer -----

def train_bpe(texts: list[str], vocab_size: int) -> Tokenizer:
    tok = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[UNK_TOKEN] + SPECIAL_TOKENS,
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def encode_with_mask(tok: Tokenizer, text: str, asst_start_char: int,
                     max_len: int) -> tuple[list[int], list[int]]:
    """Tokenize and produce a per-token loss mask (1 for response tokens
    after the asst_start_char marker, 0 otherwise)."""
    enc = tok.encode(text)
    ids = enc.ids[:max_len]
    offsets = enc.offsets[:max_len]   # (start_char, end_char) per token
    mask = [
        1 if (start >= asst_start_char) else 0
        for (start, _end) in offsets
    ]
    return ids, mask


# ----- Training -----

def pad_batch(seqs, masks, pad_id, max_len):
    B = len(seqs)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    loss_mask = torch.zeros(B, max_len, dtype=torch.float)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        L = min(len(s), max_len)
        input_ids[i, :L] = torch.tensor(s[:L], dtype=torch.long)
        loss_mask[i, :L] = torch.tensor(m[:L], dtype=torch.float)
    return input_ids, loss_mask


def train(model, tok, examples, epochs, batch_size, lr, max_len, log_every=50):
    rng = random.Random(0)
    pad_id = tok.token_to_id(PAD_TOKEN)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    print(f"pre-tokenizing {len(examples)} examples...", flush=True)
    cache = []
    for text, asst_start in examples:
        ids, mask = encode_with_mask(tok, text, asst_start, max_len)
        if sum(mask) > 0:
            cache.append((ids, mask))
    print(f"  {len(cache)} after filter", flush=True)

    steps_per_epoch = max(1, len(cache) // batch_size)
    t0 = time.time()
    global_step = 0
    for epoch in range(epochs):
        rng.shuffle(cache)
        eloss, etoks = 0.0, 0
        for step in range(steps_per_epoch):
            batch = cache[step*batch_size:(step+1)*batch_size]
            if len(batch) < batch_size:
                continue
            seqs = [b[0] for b in batch]
            masks = [b[1] for b in batch]
            ids, lmask = pad_batch(seqs, masks, pad_id, max_len)
            logits = model(ids[:, :-1])
            targets = ids[:, 1:]
            mask = lmask[:, 1:]
            V = logits.size(-1)
            per = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                                    reduction="none")
            masked = per * mask.reshape(-1)
            n_tok = mask.sum().clamp(min=1.0)
            loss = masked.sum() / n_tok
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += masked.sum().item()
            etoks += n_tok.item()
            global_step += 1
            if global_step % log_every == 0:
                ppl = torch.tensor(eloss/max(1, etoks)).exp().item()
                print(f"  epoch {epoch+1}/{epochs} step {global_step} "
                      f"loss={eloss/max(1, etoks):.4f} ppl={ppl:.1f} "
                      f"{time.time()-t0:.1f}s", flush=True)
        ppl = torch.tensor(eloss/max(1, etoks)).exp().item()
        print(f"epoch {epoch+1}/{epochs} done  loss={eloss/max(1, etoks):.4f} "
              f"ppl={ppl:.1f}  {time.time()-t0:.1f}s", flush=True)
    model.eval()


# ----- Eval -----

@torch.no_grad()
def lm_perplexity(model, tok, examples, max_len, batch_size=8):
    """Mean cross-entropy ppl on response tokens of held-out examples."""
    pad_id = tok.token_to_id(PAD_TOKEN)
    cache = []
    for text, asst_start in examples:
        ids, mask = encode_with_mask(tok, text, asst_start, max_len)
        if sum(mask) > 0:
            cache.append((ids, mask))
    total_loss, total_tok = 0.0, 0
    for i in range(0, len(cache), batch_size):
        batch = cache[i:i+batch_size]
        seqs = [b[0] for b in batch]
        masks = [b[1] for b in batch]
        ids, lmask = pad_batch(seqs, masks, pad_id, max_len)
        logits = model(ids[:, :-1])
        targets = ids[:, 1:]
        mask = lmask[:, 1:]
        V = logits.size(-1)
        per = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                                reduction="none")
        masked = per * mask.reshape(-1)
        total_loss += masked.sum().item()
        total_tok += mask.sum().item()
    return torch.tensor(total_loss/max(1, total_tok)).exp().item()


@torch.no_grad()
def hrm_accuracy(model, tok, problems, max_len, max_new_tokens=40):
    """For each NL problem, generate from `<|hrm|> question = `, parse the
    generated expression, evaluate it, compare to ground-truth answer."""
    eos_id = tok.token_to_id(EOS_TOKEN)
    correct = 0
    parseable = 0
    structural_match = 0
    for p in problems:
        prompt = f"{HRM_MODE} {p.question} = "
        prompt_ids = tok.encode(prompt).ids
        ids = list(prompt_ids)
        new_tokens = []
        for _ in range(max_new_tokens):
            if len(ids) >= max_len:
                break
            x = torch.tensor([ids], dtype=torch.long)
            logits = model(x)[0, -1, :]
            nid = int(logits.argmax(-1).item())
            if nid == eos_id:
                break
            ids.append(nid)
            new_tokens.append(nid)
        gen_text = tok.decode(new_tokens).strip()
        # Parse + interpret
        try:
            gate_graph = parse_expression(gen_text.rstrip("=").strip())
            parseable += 1
            res = interpret(gate_graph)
            ans = str(res)
            if isinstance(res, float) and res == int(res):
                ans = str(int(res))
            if ans == p.answer:
                correct += 1
            if gen_text.replace(" ", "") == p.expression.replace(" ", ""):
                structural_match += 1
        except Exception:
            pass
    return {
        "correct": correct,
        "parseable": parseable,
        "structural_match": structural_match,
        "total": len(problems),
    }


# ----- Main -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--d-ffn", type=int, default=384)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--n-hrm-train", type=int, default=1500)
    ap.add_argument("--n-hrm-eval", type=int, default=100)
    ap.add_argument("--n-lm-eval", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print("=== Hybrid SubstrateLM + SubstrateHRM Training ===")
    print(f"  d_model={args.d_model}  n_layers={args.n_layers}  "
          f"vocab={args.vocab_size}  max_len={args.max_len}")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")

    # Load LM data
    lm_examples = load_lm_examples(LM_DATA_PATHS)
    print(f"\nloaded {len(lm_examples)} LM examples")

    # Generate HRM data — split into train + eval
    all_hrm = build_hrm_examples(args.n_hrm_train + args.n_hrm_eval, seed=42)
    hrm_train = all_hrm[:args.n_hrm_train]
    hrm_eval_text = all_hrm[args.n_hrm_train:]
    print(f"generated {len(hrm_train)} HRM train + {len(hrm_eval_text)} HRM eval")

    # LM eval: hold out last N from LM training set
    lm_train = lm_examples[:-args.n_lm_eval]
    lm_eval = lm_examples[-args.n_lm_eval:]
    print(f"LM train={len(lm_train)}  LM eval={len(lm_eval)}")

    # Combined training set
    combined = lm_train + hrm_train
    random.Random(args.seed).shuffle(combined)
    print(f"combined train={len(combined)}")

    # Train tokenizer on combined corpus
    print("\ntraining BPE tokenizer...")
    tok = train_bpe([t for t, _ in combined], vocab_size=args.vocab_size)
    print(f"  vocab: {tok.get_vocab_size()}")

    # Build model
    cfg = Small2DConfig(
        vocab_size=args.vocab_size, d_model=args.d_model,
        n_heads=args.d_model // 2,
        n_layers=args.n_layers, d_ffn=args.d_ffn,
        max_len=args.max_len, use_hard_max=False,
    )
    assert cfg.d_head == 2
    model = Small2DTransformer(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    # Train
    print("\n--- training ---")
    train(model, tok, combined,
          epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, max_len=args.max_len)

    # Save
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / "substrate_hybrid_mvp.pt"
    tok_path = CKPT_DIR / "substrate_hybrid_mvp_tokenizer.json"
    torch.save({"model_state": model.state_dict(), "config": cfg.__dict__},
               ckpt_path)
    tok.save(str(tok_path))
    print(f"\nsaved {ckpt_path.relative_to(REPO_ROOT)}")

    # Eval
    print("\n=== EVAL ===")
    print("\n[LM mode] perplexity on held-out claude_reasoning examples:")
    lm_ppl = lm_perplexity(model, tok, lm_eval, max_len=args.max_len)
    print(f"  hybrid LM ppl = {lm_ppl:.1f}  "
          f"(SubstrateLM MVP baseline = 424)")

    print("\n[HRM mode] expression generation on held-out NL math:")
    hrm_problems = [
        # build_hrm_examples returned (text, asst_start). Need to recover the
        # NLMathProblem to evaluate. Regenerate using the same seed offset.
    ]
    # Re-derive problems from generator with the same seed for eval split
    gen = NLMathDataGenerator(seed=42)
    all_problems = gen.generate(n=args.n_hrm_train + args.n_hrm_eval)
    hrm_problems = all_problems[args.n_hrm_train:]
    res = hrm_accuracy(model, tok, hrm_problems, max_len=args.max_len)
    print(f"  parseable:        {res['parseable']}/{res['total']}  "
          f"({res['parseable']/res['total']:.1%})")
    print(f"  structural match: {res['structural_match']}/{res['total']}  "
          f"({res['structural_match']/res['total']:.1%})")
    print(f"  correct value:    {res['correct']}/{res['total']}  "
          f"({res['correct']/res['total']:.1%})")
    print(f"\n  (SubstrateHRM_nl baseline at 180K params = 99.1%)")

    print("\n" + "=" * 60)
    # Decision: hybrid passes if (a) LM ppl <= ~500 (no major degradation
    # vs SubstrateLM 424), AND (b) HRM correct >= 30% (substrate learned
    # at least basic structure extraction).
    pass_lm = lm_ppl <= 500
    pass_hrm = res['correct'] / res['total'] >= 0.30
    if pass_lm and pass_hrm:
        print("DECISION: PASS — hybrid hosts both modes without collapse.")
        print(f"  1+1 >= 2: free dual capability in one model file.")
    elif pass_lm:
        print(f"DECISION: PARTIAL — LM mode preserved, HRM mode underperforms.")
        print(f"  Mode-conditioning works for LM but HRM needs more "
              f"capacity / epochs.")
    elif pass_hrm:
        print(f"DECISION: PARTIAL — HRM mode learned, LM mode degraded.")
        print(f"  HRM data dominates training; rebalance ratio.")
    else:
        print(f"DECISION: FAIL — both modes underperform their baselines.")
        print(f"  Capacity may be insufficient or mode tokens not separating.")
    print("=" * 60)


if __name__ == "__main__":
    main()
