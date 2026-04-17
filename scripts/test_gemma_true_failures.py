"""Round 10d: map Gemma's true arithmetic failure surface.

Earlier tests measured "first token after prompt" and falsely concluded
Gemma failed at single-digit addition. Gemma actually solves most
single-digit arithmetic in its continuation; it just prefers markdown
formatting.

Now: test Gemma on a gradient of arithmetic difficulty, parse the full
continuation for the answer, find where Gemma genuinely fails.

Categories tested (each with 5 prompts):
  1. Single-digit add                     — expected: Gemma passes
  2. Two-digit add                        — unknown
  3. Three-digit add                      — unknown
  4. Single-digit mul                     — expected: Gemma passes
  5. Two-digit mul                        — likely fails
  6. Mixed (order of operations)          — likely fails
  7. Three-operand chain (1+2+3)          — unknown
  8. Primality test                       — unknown
  9. Factorial                            — likely passes small, fails big
 10. Word problems with computation       — likely fails

For each prompt: generate 30 tokens, parse the numeric answer from
the continuation, check correctness via ground truth (Python compute).
"""

from __future__ import annotations

import os
import re
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")


# (prompt, expected_int_answer)
TESTS = [
    # 1. Single-digit add
    ("what is 2 plus 3 equals", 5),
    ("what is 4 plus 1 equals", 5),
    ("what is 5 plus 1 equals", 6),
    ("what is 2 plus 4 equals", 6),
    ("what is 3 plus 4 equals", 7),
    # 2. Two-digit add
    ("what is 27 plus 13 equals", 40),
    ("what is 58 plus 29 equals", 87),
    ("what is 44 plus 91 equals", 135),
    ("what is 76 plus 25 equals", 101),
    ("what is 89 plus 12 equals", 101),
    # 3. Three-digit add
    ("what is 347 plus 289 equals", 636),
    ("what is 612 plus 194 equals", 806),
    ("what is 823 plus 477 equals", 1300),
    ("what is 199 plus 201 equals", 400),
    ("what is 500 plus 500 equals", 1000),
    # 4. Single-digit multiply
    ("what is 3 times 4 equals", 12),
    ("what is 7 times 8 equals", 56),
    ("what is 9 times 9 equals", 81),
    ("what is 6 times 6 equals", 36),
    ("what is 5 times 7 equals", 35),
    # 5. Two-digit multiply
    ("what is 17 times 23 equals", 391),
    ("what is 34 times 12 equals", 408),
    ("what is 25 times 16 equals", 400),
    ("what is 47 times 19 equals", 893),
    ("what is 99 times 99 equals", 9801),
    # 6. Order of operations
    ("what is 2 plus 3 times 4 equals", 14),
    ("what is (2 plus 3) times 4 equals", 20),
    ("what is 10 minus 3 plus 2 equals", 9),
    ("what is 20 divided by 4 times 3 equals", 15),
    ("what is 100 minus 20 times 3 equals", 40),
    # 7. Three-operand chain
    ("what is 1 plus 2 plus 3 equals", 6),
    ("what is 5 plus 7 plus 11 equals", 23),
    ("what is 10 plus 20 plus 30 equals", 60),
    ("what is 4 plus 4 plus 4 plus 4 equals", 16),
    ("what is 2 plus 4 plus 6 plus 8 plus 10 equals", 30),
    # 8. Primality
    ("is 17 prime? answer yes or no.", 1),  # 1 = yes
    ("is 91 prime? answer yes or no.", 0),  # 91 = 7*13
    ("is 29 prime? answer yes or no.", 1),
    ("is 51 prime? answer yes or no.", 0),  # 51 = 3*17
    ("is 97 prime? answer yes or no.", 1),
    # 9. Factorial
    ("what is factorial of 4 equals", 24),
    ("what is factorial of 5 equals", 120),
    ("what is factorial of 6 equals", 720),
    ("what is factorial of 7 equals", 5040),
    ("what is factorial of 8 equals", 40320),
    # 10. Word problems
    ("Alice has 3 apples. Bob gives her 5 more. How many apples does Alice have? Answer with a number.", 8),
    ("A store sells pencils for 7 cents each. How much do 12 pencils cost in cents? Answer with a number.", 84),
    ("If a train travels 60 miles per hour for 4 hours, how far does it travel in miles? Answer with a number.", 240),
    ("Sarah has 45 cookies and eats 17. How many cookies are left? Answer with a number.", 28),
    ("A bag has 24 marbles divided equally among 6 kids. How many marbles per kid? Answer with a number.", 4),
]

CATEGORIES = [
    "1-digit +", "2-digit +", "3-digit +",
    "1-digit ×", "2-digit ×",
    "order-of-ops", "3+-operand chain",
    "primality", "factorial", "word problem",
]


def generate(m, tok, prompt, max_tokens=40):
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


def extract_int_answer(text: str, expected: int) -> tuple[bool, int | None]:
    """Find the first integer in `text` that matches `expected`.
    Also return the first integer found at all (for debugging)."""
    nums = re.findall(r"-?\d+", text.replace(",", ""))
    if not nums:
        return False, None
    first_int = int(nums[0])
    # Pass if ANY of the extracted integers matches expected (Gemma may
    # include the operands before the answer).
    for n in nums:
        if int(n) == expected:
            return True, first_int
    return False, first_int


def extract_yesno(text: str, expected: int) -> tuple[bool, str | None]:
    """For primality: find 'yes' or 'no' in lowercased continuation."""
    lower = text.lower()
    yes = "yes" in lower
    no = "no" in lower
    if yes and not no:
        return expected == 1, "yes"
    if no and not yes:
        return expected == 0, "no"
    if yes and no:
        # Both appear; take first
        first = "yes" if lower.index("yes") < lower.index("no") else "no"
        return (expected == 1 if first == "yes" else expected == 0), first
    return False, None


def check(prompt, expected, text):
    """Category-aware pass check. Primality uses yes/no; rest use ints."""
    if "prime" in prompt.lower():
        ok, found = extract_yesno(text, expected)
    else:
        ok, found = extract_int_answer(text, expected)
    return ok, found


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[true-failures] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 8, 20))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # Run all prompts
    results = []
    for i, (prompt, expected) in enumerate(TESTS):
        text = generate(m, tok, prompt, max_tokens=180)
        ok, found = check(prompt, expected, text)
        results.append((prompt, expected, text, ok, found))
        cat = CATEGORIES[i // 5]
        # Short log
        short_text = text.replace("\n", "\\n")[:60]
        print(f"  [{cat:<16}] {'✓' if ok else '✗'} expected={expected} "
              f"found={found} | {short_text!r}")

    # Per-category summary
    print("\n========== PER-CATEGORY SUMMARY ==========")
    for ci, cat in enumerate(CATEGORIES):
        cat_results = results[ci * 5:(ci + 1) * 5]
        passes = sum(1 for _, _, _, ok, _ in cat_results if ok)
        print(f"  {cat:<18} {passes}/5")

    # Save full continuations for failure analysis
    failures = [r for r in results if not r[3]]
    print(f"\n========== FAILURES ({len(failures)}) ==========")
    for prompt, expected, text, _, found in failures:
        short = text.replace("\n", "\\n")[:100]
        print(f"  expected={expected} found={found}")
        print(f"    prompt: {prompt!r}")
        print(f"    text:   {short!r}")


if __name__ == "__main__":
    sys.exit(main())
