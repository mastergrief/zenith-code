"""Minibench — isolate-test substrate primitives at tiny scale.

v2 bundled D3 mixed geometry + D5 recurrence + curriculum changes into
one 2-hour training run and failed. That violates the one-variable-per-
round rule. This script runs a fixed tiny training job (d_model=32,
n_layers=2, ~500 examples, ~200 steps, 3-10 min) with CLI flags for
each primitive in isolation so we can measure each one's effect on
a known-good baseline before bundling.

Task: single-family NL→structure (nl_data templates). This is the
simplest form of the task v2 failed on — if a primitive breaks this,
it'll break joint training too.

Usage:
    # baseline (pure euclidean, 1 iteration)
    python3 scripts/minibench_substrate.py --tag baseline

    # D3 isolated: hyperbolic on layer 1
    python3 scripts/minibench_substrate.py \\
        --geometry euclidean,hyperbolic --tag d3_hyperbolic

    # D5 isolated: 2 iterations
    python3 scripts/minibench_substrate.py --iterations 2 --tag d5_iter2

    # combined (what v2 did — expect to reproduce the failure at tiny scale)
    python3 scripts/minibench_substrate.py \\
        --geometry euclidean,hyperbolic --iterations 2 --tag combined

Each run prints a compact before/after table: final ppl, structural
correctness on 50 held-out prompts, wallclock.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from calm.hrm.nl_data import NLMathDataGenerator
from calm.llm_computer.combined_substrate import (
    CombinedConfig, CombinedSmall2DTransformer,
)
from calm.llm_computer.substrate_lm import (
    ASST_TOKEN, EOS_TOKEN, PAD_TOKEN, SYS_TOKEN, USER_TOKEN,
    train_bpe_tokenizer,
)


# ----- Fixed corpus -----

def build_corpus(n_train: int, n_eval: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Generate NL math problems and format as chat messages.

    Returns (train_examples, eval_examples) where each example is a
    messages list `[{"role": "user", ...}, {"role": "assistant", ...}]`.
    Assistant target is `<expression> =` — the HRM structure-only format.
    """
    gen = NLMathDataGenerator(seed=seed)
    probs = gen.generate(n_train + n_eval)
    # Seeded shuffle for reproducible split
    rng = random.Random(seed + 1)
    rng.shuffle(probs)

    def to_msgs(p) -> list[dict]:
        return [
            {"role": "user", "content": p.question},
            {"role": "assistant", "content": f"{p.expression} ="},
        ]

    train = [to_msgs(p) for p in probs[:n_train]]
    eval_ = [to_msgs(p) for p in probs[n_train:n_train + n_eval]]
    return train, eval_


def format_chat(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m["role"]
        tok = {"system": SYS_TOKEN, "user": USER_TOKEN, "assistant": ASST_TOKEN}[role]
        parts.append(tok + m["content"])
    parts.append(EOS_TOKEN)
    return "".join(parts)


def encode_with_mask(tokenizer, messages, max_len):
    text = format_chat(messages)
    enc = tokenizer.encode(text)
    ids = enc.ids[:max_len]
    asst_id = tokenizer.token_to_id(ASST_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    mask = [0] * len(ids)
    in_asst = False
    for i, t in enumerate(ids):
        if t == asst_id:
            in_asst = True
            continue
        if in_asst:
            mask[i] = 1
        if t == eos_id:
            break
    return ids, mask


def pad_batch(seqs, masks, pad_id, max_len):
    B = len(seqs)
    ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    lm = torch.zeros(B, max_len, dtype=torch.float)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        L = min(len(s), max_len)
        ids[i, :L] = torch.tensor(s[:L], dtype=torch.long)
        lm[i, :L] = torch.tensor(m[:L], dtype=torch.float)
    return ids, lm


# ----- Training -----

def run_training(
    model: CombinedSmall2DTransformer,
    tokenizer,
    train_msgs: list[list[dict]],
    steps: int,
    batch_size: int,
    lr: float,
    max_len: int,
    device: str,
    n_iterations: int,
    bf16: bool,
    log_every: int,
    seed: int,
) -> dict:
    model.train()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    pad_id = tokenizer.token_to_id(PAD_TOKEN)

    # Pre-tokenize once
    cache = []
    for msgs in train_msgs:
        ids, mask = encode_with_mask(tokenizer, msgs, max_len)
        if sum(mask) > 0:
            cache.append((ids, mask))

    rng = random.Random(seed)
    amp_dtype = torch.bfloat16 if bf16 and device == "cuda" else None
    t0 = time.time()
    running_loss, running_toks = 0.0, 0.0
    last_ppl = float("inf")

    for step in range(1, steps + 1):
        batch = rng.sample(cache, min(batch_size, len(cache)))
        seqs = [b[0] for b in batch]
        masks = [b[1] for b in batch]
        ids, lm = pad_batch(seqs, masks, pad_id, max_len)
        ids, lm = ids.to(device), lm.to(device)

        if amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(ids[:, :-1], n_iterations=n_iterations)
                targets = ids[:, 1:]
                mask = lm[:, 1:]
                V = logits.size(-1)
                per_tok = F.cross_entropy(
                    logits.reshape(-1, V), targets.reshape(-1),
                    reduction="none",
                )
                masked = per_tok * mask.reshape(-1)
                n_tok = mask.sum().clamp(min=1.0)
                loss = masked.sum() / n_tok
        else:
            logits = model(ids[:, :-1], n_iterations=n_iterations)
            targets = ids[:, 1:]
            mask = lm[:, 1:]
            V = logits.size(-1)
            per_tok = F.cross_entropy(
                logits.reshape(-1, V), targets.reshape(-1),
                reduction="none",
            )
            masked = per_tok * mask.reshape(-1)
            n_tok = mask.sum().clamp(min=1.0)
            loss = masked.sum() / n_tok

        opt.zero_grad()
        loss.backward()
        opt.step()

        running_loss += masked.sum().item()
        running_toks += n_tok.item()

        if step % log_every == 0:
            avg_loss = running_loss / max(1.0, running_toks)
            last_ppl = float(torch.tensor(avg_loss).exp().item())
            print(f"  step {step}/{steps} loss={avg_loss:.4f} ppl={last_ppl:.1f} "
                  f"{time.time()-t0:.1f}s", flush=True)
            running_loss, running_toks = 0.0, 0.0

    elapsed = time.time() - t0
    model.eval()
    return {"final_ppl": last_ppl, "wallclock_s": elapsed}


# ----- Eval: structural correctness on held-out prompts -----

@torch.no_grad()
def eval_structural(
    model: CombinedSmall2DTransformer,
    tokenizer,
    eval_msgs: list[list[dict]],
    max_len: int,
    device: str,
    n_iterations: int,
    max_new_tokens: int = 48,
) -> dict:
    """For each eval prompt, generate the assistant response and check
    whether it structurally matches the target expression (string equality
    up to whitespace, ignoring the trailing ` =`).
    """
    model.eval()
    asst_id = tokenizer.token_to_id(ASST_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    n_total = 0
    n_parse = 0        # emitted a non-empty answer
    n_structural = 0   # answer string-matched target expression (pre-`=`)

    for msgs in eval_msgs:
        # Build prompt up through `<|asst|>`
        user_msg = msgs[0]
        target_expr = msgs[1]["content"].rsplit("=", 1)[0].strip()
        prompt_text = SYS_TOKEN + "" + USER_TOKEN + user_msg["content"] + ASST_TOKEN
        ids = tokenizer.encode(prompt_text).ids
        ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

        generated: list[int] = []
        for _ in range(max_new_tokens):
            if ids.size(1) >= max_len:
                break
            logits = model(ids, n_iterations=n_iterations)
            next_id = int(logits[0, -1].argmax().item())
            if next_id == eos_id:
                break
            generated.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

        decoded = tokenizer.decode(generated).strip()
        # Strip trailing `=` and whitespace before comparing
        answer = decoded.rsplit("=", 1)[0].strip() if "=" in decoded else decoded
        n_total += 1
        if answer:
            n_parse += 1
        # Canonicalize whitespace on both sides
        if " ".join(answer.split()) == " ".join(target_expr.split()):
            n_structural += 1

    return {
        "n_total": n_total,
        "parse_rate": n_parse / max(1, n_total),
        "structural_rate": n_structural / max(1, n_total),
    }


# ----- Main -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline", help="label for this run")
    ap.add_argument("--geometry", default="euclidean,euclidean",
                    help="comma-separated per-layer geometry (len must == n-layers)")
    ap.add_argument("--iterations", type=int, default=1,
                    help="D5 n_iterations (shared weights, repeat forward)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--n-heads", type=int, default=16)   # d_head=2
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--d-ffn", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    geometries = args.geometry.split(",")
    assert len(geometries) == args.n_layers, (
        f"geometry must have {args.n_layers} entries, got {len(geometries)}"
    )

    print(f"=== minibench [{args.tag}] ===", flush=True)
    print(f"  device={device}  geometry={geometries}  iterations={args.iterations}",
          flush=True)
    print(f"  d_model={args.d_model} n_heads={args.n_heads} n_layers={args.n_layers} "
          f"d_ffn={args.d_ffn} max_len={args.max_len} vocab={args.vocab_size}",
          flush=True)

    # Corpus
    train_msgs, eval_msgs = build_corpus(args.n_train, args.n_eval, args.seed)
    texts = [format_chat(m) for m in train_msgs]
    tokenizer = train_bpe_tokenizer(texts, vocab_size=args.vocab_size)

    # Model
    cfg = CombinedConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ffn=args.d_ffn,
        max_len=args.max_len,
        use_hard_max=False,
        layer_geometries=geometries,
        default_iterations=args.iterations,
        max_iterations=8,
    )
    assert cfg.d_head == 2, f"d_head must be 2, got {cfg.d_head}"
    model = CombinedSmall2DTransformer(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}", flush=True)

    # Train
    print("--- training ---", flush=True)
    train_result = run_training(
        model=model,
        tokenizer=tokenizer,
        train_msgs=train_msgs,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
        device=device,
        n_iterations=args.iterations,
        bf16=args.bf16,
        log_every=args.log_every,
        seed=args.seed,
    )

    # Eval
    print("--- eval ---", flush=True)
    eval_result = eval_structural(
        model=model, tokenizer=tokenizer, eval_msgs=eval_msgs,
        max_len=args.max_len, device=device, n_iterations=args.iterations,
    )

    # Summary
    print("", flush=True)
    print("=== summary ===", flush=True)
    print(f"  tag:             {args.tag}", flush=True)
    print(f"  geometry:        {geometries}", flush=True)
    print(f"  iterations:      {args.iterations}", flush=True)
    print(f"  params:          {n_params:,}", flush=True)
    print(f"  final ppl:       {train_result['final_ppl']:.1f}", flush=True)
    print(f"  parse rate:      {eval_result['parse_rate']*100:.1f}%  "
          f"({int(eval_result['parse_rate']*eval_result['n_total'])}/"
          f"{eval_result['n_total']})", flush=True)
    print(f"  structural rate: {eval_result['structural_rate']*100:.1f}%  "
          f"({int(eval_result['structural_rate']*eval_result['n_total'])}/"
          f"{eval_result['n_total']})", flush=True)
    print(f"  wallclock:       {train_result['wallclock_s']:.1f}s", flush=True)
    print(f"  tok/s (train):   {args.steps*args.batch_size*args.max_len/train_result['wallclock_s']:.0f}",
          flush=True)


if __name__ == "__main__":
    main()
