"""Round 10d-revised: focused test on Gemma's HARD cases.

After the 40-token run showed truncation was masking as failure, this
run uses 180 tokens and focuses on prompts where Gemma has a real
chance of being wrong (not just formatting-verbose).

Categories (4 prompts each, 16 total):
  - 2-digit multiplication (clear capability test)
  - Order-of-operations (symbolic reasoning)
  - Primality (binary Y/N — answer must appear in text)
  - Factorial (lookup-like)
"""

from __future__ import annotations

import os
import re
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")


TESTS = [
    # 2-digit multiply
    ("what is 17 times 23? Answer with just the number.", 391),
    ("what is 34 times 12? Answer with just the number.", 408),
    ("what is 47 times 19? Answer with just the number.", 893),
    ("what is 99 times 99? Answer with just the number.", 9801),
    # Order of ops
    ("What is 2 plus 3 times 4? Answer with just the number.", 14),
    ("What is 10 minus 3 plus 2? Answer with just the number.", 9),
    ("What is 20 divided by 4 times 3? Answer with just the number.", 15),
    ("What is 100 minus 20 times 3? Answer with just the number.", 40),
    # Primality (explicit)
    ("Is 91 prime? Answer yes or no.", 0),  # 91 = 7*13
    ("Is 29 prime? Answer yes or no.", 1),
    ("Is 51 prime? Answer yes or no.", 0),  # 51 = 3*17
    ("Is 97 prime? Answer yes or no.", 1),
    # Factorial
    ("What is factorial of 5? Answer with just the number.", 120),
    ("What is factorial of 6? Answer with just the number.", 720),
    ("What is factorial of 7? Answer with just the number.", 5040),
    ("What is factorial of 8? Answer with just the number.", 40320),
]

CATEGORIES = ["2-digit ×", "order-of-ops", "primality", "factorial"]


def generate(m, tok, prompt, max_tokens=180):
    from calm.llm_computer.gemma_substrate import KVCache
    ids = tok.encode(prompt)
    cache = KVCache(m.config.n_layers, device="cuda")
    gen_ids = list(ids)
    with torch.no_grad():
        logits = m.forward(
            torch.tensor([gen_ids]), device="cuda",
            kv_cache=cache, start_pos=0,
        )
        nxt = int(logits[0, -1].argmax())
        gen_ids.append(nxt)
        for _ in range(max_tokens - 1):
            if nxt == tok.EOS_ID:
                break
            logits = m.forward(
                torch.tensor([[nxt]]), device="cuda",
                kv_cache=cache, start_pos=len(gen_ids) - 1,
            )
            nxt = int(logits[0, -1].argmax())
            gen_ids.append(nxt)
    return tok.decode(gen_ids[len(ids):])


def check(prompt, expected, text):
    if "prime" in prompt.lower():
        lower = text.lower()
        yes = "yes" in lower
        no = "no" in lower
        if yes and not no:
            return expected == 1, "yes"
        if no and not yes:
            return expected == 0, "no"
        return False, "neither"
    nums = re.findall(r"-?\d+", text.replace(",", ""))
    for n in nums:
        if int(n) == expected:
            return True, int(n)
    return False, (int(nums[0]) if nums else None)


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[hard-cases] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    results = []
    for i, (prompt, expected) in enumerate(TESTS):
        text = generate(m, tok, prompt, max_tokens=180)
        ok, found = check(prompt, expected, text)
        results.append((prompt, expected, text, ok, found))
        cat = CATEGORIES[i // 4]
        short = text.replace("\n", "\\n")[:80]
        print(f"  [{cat:<14}] {'✓' if ok else '✗'} expected={expected} "
              f"found={found} | {short!r}")

    print("\n========== PER-CATEGORY ==========")
    for ci, cat in enumerate(CATEGORIES):
        cat_results = results[ci * 4:(ci + 1) * 4]
        passes = sum(1 for _, _, _, ok, _ in cat_results if ok)
        print(f"  {cat:<16} {passes}/4")

    failures = [r for r in results if not r[3]]
    print(f"\n========== FAILURES ({len(failures)}) ==========")
    for prompt, expected, text, _, found in failures:
        short = text.replace("\n", "\\n")[:200]
        print(f"  expected={expected} found={found}")
        print(f"    prompt: {prompt!r}")
        print(f"    text:   {short!r}")


if __name__ == "__main__":
    sys.exit(main())
