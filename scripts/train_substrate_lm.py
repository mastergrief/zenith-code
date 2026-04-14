"""Train the MVP SubstrateLM on Claude-authored reasoning data.

End-to-end pipeline:
  1. Load `agents/distill/data/claude_reasoning.jsonl` and
     `coding_reasoning_claude.jsonl` — ~1.5K Claude-authored examples.
  2. Train a byte-level BPE tokenizer (vocab=8192) on the corpus.
  3. Build a 128-dim Small2DTransformer (~8M params, d_head=2).
  4. Fine-tune with CE loss masked to assistant tokens.
  5. Save checkpoint + tokenizer to calm/llm_computer/checkpoints/.
  6. Print N sample completions for inspection.

Goal (MVP bar): "does a Small2DTransformer trained on ~1.5K examples
produce grammatical, contextually appropriate completions on held-out
prompts?" If yes, the substrate hosts LM behavior and scaling is pure
engineering. If no, we learn the d_head=2 constraint limits LM training.

Runtime: ~30-90 min on CPU depending on epoch count.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from calm.llm_computer.substrate_lm import (
    SubstrateLMConfig,
    build_substrate_lm,
    generate,
    load_messages,
    train_bpe_tokenizer,
    train_substrate_lm,
    format_chat,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATHS = [
    REPO_ROOT / "agents" / "distill" / "data" / "claude_reasoning.jsonl",
    REPO_ROOT / "agents" / "distill" / "data" / "coding_reasoning_claude.jsonl",
]
CKPT_DIR = REPO_ROOT / "calm" / "llm_computer" / "checkpoints"


SAMPLE_PROMPTS = [
    "what is 2 + 2?",
    "how do i center a div in css",
    "my docker container keeps dying",
    "explain how a linked list works",
    "how does garbage collection work in python",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-ffn", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-name", type=str, default="substrate_lm_mvp")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print("=== SubstrateLM MVP Training ===")
    print(f"  corpus: {[str(p.relative_to(REPO_ROOT)) for p in DATA_PATHS]}")
    print(f"  d_model={args.d_model} n_layers={args.n_layers} "
          f"vocab={args.vocab_size} max_len={args.max_len}")
    print(f"  epochs={args.epochs} batch={args.batch_size} lr={args.lr}")

    # 1. Load corpus
    examples = load_messages(DATA_PATHS)
    print(f"\nloaded {len(examples)} examples")

    # 2. Train tokenizer on flattened corpus text
    print("training BPE tokenizer...")
    corpus_texts = [format_chat(msgs)[0] for msgs in examples]
    tokenizer = train_bpe_tokenizer(corpus_texts, vocab_size=args.vocab_size)
    print(f"  vocab: {tokenizer.get_vocab_size()} tokens")

    # 3. Build model
    assert args.d_model % 2 == 0
    cfg = SubstrateLMConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.d_model // 2,  # d_head = 2
        n_layers=args.n_layers,
        d_ffn=args.d_ffn,
        max_len=args.max_len,
    )
    model = build_substrate_lm(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    # 4. Train
    print("\n--- training ---")
    train_substrate_lm(
        model, tokenizer, examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
        seed=args.seed,
    )

    # 5. Save
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{args.ckpt_name}.pt"
    tok_path = CKPT_DIR / f"{args.ckpt_name}_tokenizer.json"
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
    }, ckpt_path)
    tokenizer.save(str(tok_path))
    print(f"\nsaved: {ckpt_path.relative_to(REPO_ROOT)}")
    print(f"saved: {tok_path.relative_to(REPO_ROOT)}")

    # 6. Sample completions
    print("\n=== sample generations ===")
    for prompt in SAMPLE_PROMPTS:
        print(f"\n> {prompt}")
        out = generate(model, tokenizer, prompt, max_new_tokens=150,
                       temperature=0.7, seed=args.seed)
        # Truncate print if very long.
        print(f"  {out[:400]}{'...' if len(out) > 400 else ''}")


if __name__ == "__main__":
    main()
