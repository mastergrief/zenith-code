"""Eval SubstrateHRM on held-out NL → expression, verify via LLM-Computer.

Mirrors `scripts/eval_hrm_nl.py` shape but runs a decoder-only
Small2DTransformer instead of the encoder-decoder HRM. If SubstrateHRM
matches HRM's 97% on this task, Option 2 step 1 succeeds: the substrate
can host the parse/extract layer as well.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from calm.hrm.data import _CHAR_TO_ID, _ID_TO_CHAR
from calm.hrm.nl_data import NLMathDataGenerator
from calm.llm_computer.interpret import InterpreterError, interpret
from calm.llm_computer.parse import ParseError, parse_expression
from calm.llm_computer.substrate_hrm import build_substrate_hrm


def _decode(model, prompt: str, max_len: int, max_gen: int = 20) -> str:
    pad = _CHAR_TO_ID["<pad>"]
    bos = _CHAR_TO_ID["<bos>"]
    eos = _CHAR_TO_ID["<eos>"]
    sep = _CHAR_TO_ID["<sep>"]

    ids = [bos] + [_CHAR_TO_ID[c] for c in prompt if c in _CHAR_TO_ID] + [sep]
    device = next(model.parameters()).device
    for _ in range(max_gen):
        if len(ids) >= max_len:
            break
        padded = ids + [pad] * (max_len - len(ids))
        x = torch.tensor([padded], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
        nid = int(logits[0, len(ids) - 1, :].argmax().item())
        if nid == eos:
            break
        ids.append(nid)
    prefix_len = 1 + len(prompt) + 1  # <bos> prompt <sep>
    out = ""
    for tid in ids[prefix_len:]:
        if tid in (pad, bos, eos, sep):
            continue
        out += _ID_TO_CHAR.get(tid, "?")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="calm/hrm/checkpoints/substrate_hrm_nl_best.pt")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=9999)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if not Path(args.ckpt).exists():
        print(f"ERROR: {args.ckpt}", file=sys.stderr); sys.exit(1)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_substrate_hrm(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        d_ffn=cfg["d_ffn"], max_len=cfg["max_len"], use_hard_max=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    probs = NLMathDataGenerator(seed=args.seed).generate(args.n)
    struct, verified = 0, 0
    for p in probs:
        emit = _decode(model, p.question, max_len=cfg["max_len"])
        struct_ok = emit.replace(" ", "") == p.expression.replace(" ", "")
        struct += int(struct_ok)
        try:
            got = interpret(parse_expression(emit))
            expected = interpret(parse_expression(p.expression))
            if isinstance(got, float) and got == int(got):
                got = int(got)
            if isinstance(expected, float) and expected == int(expected):
                expected = int(expected)
            if got == expected:
                verified += 1
        except (ParseError, InterpreterError, ValueError):
            pass
        if args.verbose and not struct_ok:
            print(f"  [MISS] {p.question!r:40} → {emit!r} (exp {p.expression!r})")

    print(f"struct:    {struct}/{args.n} = {struct/args.n:.0%}")
    print(f"verified:  {verified}/{args.n} = {verified/args.n:.0%}")
    print(f"(val_acc during training: {ckpt.get('val_acc', 0):.0%})")


if __name__ == "__main__":
    main()
