"""Quick REPL to chat with the SubstrateLM MVP checkpoint.

Usage:
    PYTHONPATH=. python3 scripts/chat_substrate_lm.py
    PYTHONPATH=. python3 scripts/chat_substrate_lm.py --temperature 0.9 --top-k 80

Commands:
    /exit, /quit  — leave
    /temp N       — set sampling temperature
    /topk N       — set top-k
    /maxtok N     — set max new tokens

Expect incoherent content at this scale (1.25M params, 13 min CPU training).
The format should look Claude-like: <think> blocks, numbered steps, code
markers. The words inside the format are semantic noise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from calm.llm_computer.substrate_lm import (
    SubstrateLMConfig,
    build_substrate_lm,
    generate,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO_ROOT / "calm" / "llm_computer" / "checkpoints"


def load_checkpoint(name: str = "substrate_lm_mvp"):
    ckpt_path = CKPT_DIR / f"{name}.pt"
    tok_path = CKPT_DIR / f"{name}_tokenizer.json"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = SubstrateLMConfig(**ckpt["config"])
    model = build_substrate_lm(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tok = Tokenizer.from_file(str(tok_path))
    # Re-attach ByteLevel decoder (not persisted by from_file in older
    # tokenizers builds) so decode() reconstructs readable text rather
    # than leaving raw BPE markers.
    tok.decoder = ByteLevelDecoder()

    n_params = sum(p.numel() for p in model.parameters())
    return model, tok, cfg, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="substrate_lm_mvp")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=180)
    args = ap.parse_args()

    model, tok, cfg, n_params = load_checkpoint(args.ckpt)
    print(f"loaded: {args.ckpt}")
    print(f"  params: {n_params:,}  d_model={cfg.d_model}  "
          f"n_layers={cfg.n_layers}  vocab={cfg.vocab_size}")
    print(f"  temp={args.temperature}  top_k={args.top_k}  "
          f"max_new_tokens={args.max_new_tokens}")
    print("\nType a prompt and press enter. Commands: /exit /temp N /topk N /maxtok N\n")

    temp = args.temperature
    top_k = args.top_k
    max_new = args.max_new_tokens
    seed = 0

    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        if not prompt:
            continue
        if prompt in ("/exit", "/quit"):
            return
        if prompt.startswith("/temp "):
            temp = float(prompt.split()[1])
            print(f"  temp={temp}")
            continue
        if prompt.startswith("/topk "):
            top_k = int(prompt.split()[1])
            print(f"  top_k={top_k}")
            continue
        if prompt.startswith("/maxtok "):
            max_new = int(prompt.split()[1])
            print(f"  max_new_tokens={max_new}")
            continue

        seed += 1
        out = generate(
            model, tok, prompt,
            max_new_tokens=max_new,
            temperature=temp,
            top_k=top_k,
            seed=seed,
        )
        print(f"\n{out}\n")


if __name__ == "__main__":
    main()
