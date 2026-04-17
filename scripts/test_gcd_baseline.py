"""Round 12: probe Gemma's GCD failure surface before building facade.

Per capability_gain.md §failure-surface-gate, don't build the facade
unless Gemma genuinely fails ≥3 of the tested prompts. GCD has
genuine computation (Euclidean algorithm) but many small-operand
pairs are memorizable from training.

Test set spans operand sizes:
  - 1-digit pairs (gcd(6, 9)): likely memorized
  - 2-digit small (gcd(48, 18)): memorizable but requires compute
  - 2-digit medium (gcd(143, 77)): genuine Euclidean — 11 not obvious
  - 3-digit (gcd(1071, 462)): hard without computation
"""

from __future__ import annotations

import math
import os
import re
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


TESTS = [
    # Small (likely memorized / easy)
    ("What is the GCD of 6 and 9? Answer with just the number.", gcd(6, 9)),
    ("What is the GCD of 12 and 18? Answer with just the number.", gcd(12, 18)),
    ("What is the GCD of 4 and 8? Answer with just the number.", gcd(4, 8)),
    ("What is the GCD of 15 and 25? Answer with just the number.", gcd(15, 25)),
    # 2-digit small — must compute
    ("What is the GCD of 48 and 18? Answer with just the number.", gcd(48, 18)),
    ("What is the GCD of 24 and 36? Answer with just the number.", gcd(24, 36)),
    ("What is the GCD of 56 and 42? Answer with just the number.", gcd(56, 42)),
    ("What is the GCD of 81 and 27? Answer with just the number.", gcd(81, 27)),
    # 2-digit medium — real Euclidean challenge
    ("What is the GCD of 143 and 77? Answer with just the number.", gcd(143, 77)),
    ("What is the GCD of 221 and 91? Answer with just the number.", gcd(221, 91)),
    ("What is the GCD of 377 and 145? Answer with just the number.", gcd(377, 145)),
    ("What is the GCD of 255 and 85? Answer with just the number.", gcd(255, 85)),
    # 3-digit harder — genuinely hard
    ("What is the GCD of 1071 and 462? Answer with just the number.", gcd(1071, 462)),
    ("What is the GCD of 841 and 527? Answer with just the number.", gcd(841, 527)),
    ("What is the GCD of 1001 and 357? Answer with just the number.", gcd(1001, 357)),
    ("What is the GCD of 2024 and 1776? Answer with just the number.", gcd(2024, 1776)),
]


def generate(m, tok, prompt, max_tokens=180):
    from calm.llm_computer.gemma_substrate import KVCache
    ids = tok.encode(prompt)
    cache = KVCache(m.config.n_layers, device="cuda")
    gen = list(ids)
    with torch.no_grad():
        logits = m.forward(
            torch.tensor([gen]), device="cuda",
            kv_cache=cache, start_pos=0,
        )
        nxt = int(logits[0, -1].argmax())
        gen.append(nxt)
        for _ in range(max_tokens - 1):
            if nxt == tok.EOS_ID:
                break
            logits = m.forward(
                torch.tensor([[nxt]]), device="cuda",
                kv_cache=cache, start_pos=len(gen) - 1,
            )
            nxt = int(logits[0, -1].argmax())
            gen.append(nxt)
    return tok.decode(gen[len(ids):])


def check(expected, text):
    nums = re.findall(r"-?\d+", text.replace(",", ""))
    for n in nums:
        if int(n) == expected:
            return True, int(n)
    # Also check: is the first number the expected? (stricter)
    return False, (int(nums[0]) if nums else None)


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[gcd-baseline] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    print("\n=== baseline Gemma on GCD ===")
    results = []
    for prompt, expected in TESTS:
        text = generate(m, tok, prompt, max_tokens=180)
        ok, found = check(expected, text)
        results.append((prompt, expected, found, ok, text))
        short = text.replace("\n", "\\n")[:70]
        print(f"  {'✓' if ok else '✗'} exp={expected:<4} found={found} | {short!r}")

    pass_count = sum(1 for _, _, _, ok, _ in results if ok)
    print(f"\n  pass: {pass_count}/{len(TESTS)}")

    # Break down by difficulty
    print("\n========== by difficulty ==========")
    buckets = [
        ("1-digit small", results[:4]),
        ("2-digit small", results[4:8]),
        ("2-digit medium", results[8:12]),
        ("3-digit harder", results[12:16]),
    ]
    for name, subset in buckets:
        passes = sum(1 for _, _, _, ok, _ in subset if ok)
        print(f"  {name:<18} {passes}/{len(subset)}")

    failures = [r for r in results if not r[3]]
    print(f"\n========== failures ({len(failures)}) ==========")
    for prompt, expected, found, _, text in failures:
        short = text.replace("\n", "\\n")[:120]
        print(f"  exp={expected} found={found}")
        print(f"    prompt: {prompt[:60]!r}")
        print(f"    text:   {short!r}")


if __name__ == "__main__":
    sys.exit(main())
