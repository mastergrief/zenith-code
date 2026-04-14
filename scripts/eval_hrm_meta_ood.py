"""OOD eval for the meta-learning HRM (L3).

Evaluates the trained meta-HRM on TEST_FORMATS that were held out
entirely during training. The model sees 3 in-context examples of a
test format, then must produce the math expression for the 4th query.

If the meta-learning hypothesis holds, the model generalizes the
format pattern from the 3 demonstrations to the query — no weight
update, no fine-tune. Target: ≥70% full-expression accuracy (raw
per-token) AND ≥50% verified-mode accuracy where the model's emitted
expression is parsed and interpreted via LLM-Computer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from calm.hrm.data import _CHAR_TO_ID, _ID_TO_CHAR
from calm.hrm.inference import HRMSeq2SeqReasoner
from calm.hrm.meta_data import (
    MetaGenerator, TEST_FORMATS, TRAIN_FORMATS, _eval_expr,
)
from calm.llm_computer.interpret import InterpreterError, interpret
from calm.llm_computer.parse import (
    ParseError, extract_problem_from_trace, parse_expression,
)


def _encode_prefixed(reasoner, sample) -> torch.Tensor:
    pad = _CHAR_TO_ID["<pad>"]
    bos = _CHAR_TO_ID["<bos>"]
    eos = _CHAR_TO_ID["<eos>"]
    sep = _CHAR_TO_ID["<sep>"]

    def _chars(text: str):
        return [_CHAR_TO_ID[c] for c in text if c in _CHAR_TO_ID]

    ids = [bos]
    for ex_in, ex_out in sample.examples:
        ids += _chars(ex_in) + [sep] + _chars(ex_out) + [sep]
    ids += _chars(sample.query_in) + [eos]
    ids = ids[: reasoner.config.max_seq_len]
    while len(ids) < reasoner.config.max_seq_len:
        ids.append(pad)
    return torch.tensor([ids], dtype=torch.long, device=reasoner.device)


def _decode(reasoner, sample) -> str:
    pad = _CHAR_TO_ID["<pad>"]
    bos = _CHAR_TO_ID["<bos>"]
    eos = _CHAR_TO_ID["<eos>"]
    enc_t = _encode_prefixed(reasoner, sample)
    with torch.no_grad():
        mem = reasoner.model.encode(enc_t)
        dec = [bos]
        for _ in range(reasoner.config.max_dec_len - 1):
            padded = dec + [pad] * (reasoner.config.max_dec_len - len(dec))
            dt = torch.tensor([padded], dtype=torch.long, device=reasoner.device)
            logits = reasoner.model.decode_step(dt, mem)
            nid = int(logits[0, len(dec) - 1, :].argmax().item())
            if nid == eos:
                break
            dec.append(nid)
    out = ""
    for tid in dec[1:]:
        if tid in (pad, bos, eos):
            continue
        out += _ID_TO_CHAR.get(tid, "?")
    return out


def _eval_pool(reasoner, formats, n_per_format: int, seed: int, verbose: bool):
    gen = MetaGenerator(seed=seed, formats=formats)
    per_fmt: dict = {f: [0, 0, 0] for f in formats}  # [struct_ok, verified_ok, total]
    for fmt in formats:
        # Draw exactly n_per_format samples from THIS format
        single_gen = MetaGenerator(seed=seed, formats=(fmt,))
        samples = single_gen.generate(n_per_format)
        for s in samples:
            emit = _decode(reasoner, s)
            expr = extract_problem_from_trace(emit)

            struct_ok = (expr.strip() == s.query_expr.strip())
            verified_ok = False
            got_val = None
            try:
                got_val = interpret(parse_expression(expr))
                if isinstance(got_val, float) and got_val == int(got_val):
                    got_val = int(got_val)
                expected_val = _eval_expr(s.query_expr)
                verified_ok = (got_val == expected_val)
            except (ParseError, InterpreterError):
                verified_ok = False

            per_fmt[fmt][0] += int(struct_ok)
            per_fmt[fmt][1] += int(verified_ok)
            per_fmt[fmt][2] += 1

            if verbose and not verified_ok:
                qshort = s.query_in if len(s.query_in) < 50 else s.query_in[:47] + "..."
                print(f"  [FAIL] [{fmt:14}] {qshort:52} → emit={emit!r:30} "
                      f"(expected expr={s.query_expr!r})")
    return per_fmt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="calm/hrm/checkpoints/meta_best.pt")
    p.add_argument("--n", type=int, default=20, help="samples per format")
    p.add_argument("--seed", type=int, default=9999)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if not Path(args.ckpt).exists():
        print(f"ERROR: checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    reasoner = HRMSeq2SeqReasoner(args.ckpt)
    print(reasoner.info())
    print(f"\n# L3 held-out OOD — {len(TEST_FORMATS)} formats never seen in training\n")

    ood = _eval_pool(reasoner, TEST_FORMATS, args.n, args.seed, args.verbose)
    print(f"  {'format':<16} {'struct':>12} {'verified':>12}")
    tot_s, tot_v, tot_n = 0, 0, 0
    for fmt, (s, v, n) in ood.items():
        print(f"  {fmt:<16} {s}/{n} = {s/n:>5.0%}   {v}/{n} = {v/n:>5.0%}")
        tot_s += s; tot_v += v; tot_n += n
    print(f"  {'TOTAL':<16} {tot_s}/{tot_n} = {tot_s/tot_n:>5.0%}   "
          f"{tot_v}/{tot_n} = {tot_v/tot_n:>5.0%}")

    print(f"\n# Train format sanity (expect high accuracy)\n")
    train = _eval_pool(reasoner, TRAIN_FORMATS, max(args.n // 2, 5), args.seed + 1, False)
    tot_s, tot_v, tot_n = 0, 0, 0
    for fmt, (s, v, n) in train.items():
        tot_s += s; tot_v += v; tot_n += n
    print(f"  TRAIN total: struct {tot_s}/{tot_n} = {tot_s/tot_n:.0%}, "
          f"verified {tot_v}/{tot_n} = {tot_v/tot_n:.0%}")


if __name__ == "__main__":
    main()
