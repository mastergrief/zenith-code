"""Round 11b: MultiplicationFacade — test if compiled multiplier fixes
Gemma's actual failures (17×23, 47×19).

Challenge: 2-3 digit products are MULTI-TOKEN in Gemma's BPE
(391 → [▁, 3, 9, 1]). Single-token VerificationHook / projection
biases only the next token. Solution: run generation step-by-step,
bias each step's output to the next digit in the verified answer.

Pipeline:
  1. PT emits the operand pair → Router parses → multiplier computes
     product as an integer.
  2. Decompose product into Gemma-BPE token sequence: [digits as
     individual tokens, with leading ▁ or similar].
  3. Generate step-by-step, at each step bias the logit for the
     expected-next-digit token by +50 if still within the digit chain.
     After the digit chain, let Gemma continue naturally.

Measurement:
  - Baseline Gemma on 17×23 and 47×19 (and 10 other 2-digit × cases
    where answer < 1000). Continuation-parsed answer.
  - Facade with step-through digit bias. Same continuations parsed.
  - Compare: domain correct with/without facade.
"""

from __future__ import annotations

import math
import os
import re
import sys

import torch

GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")


# 2-digit × 2-digit cases where a·b < 1000 (multiplier's valid range).
TESTS = [
    ("what is 17 times 23? Answer with just the number.", 391),
    ("what is 47 times 19? Answer with just the number.", 893),
    ("what is 34 times 12? Answer with just the number.", 408),
    ("what is 13 times 27? Answer with just the number.", 351),
    ("what is 21 times 38? Answer with just the number.", 798),
    ("what is 45 times 15? Answer with just the number.", 675),
    ("what is 11 times 11? Answer with just the number.", 121),
    ("what is 29 times 17? Answer with just the number.", 493),
    ("what is 32 times 25? Answer with just the number.", 800),
    ("what is 16 times 31? Answer with just the number.", 496),
]


def digits_as_gemma_tokens(tok, n: int) -> list[int]:
    """Encode `n` as the sequence of Gemma tokens (skipping <bos>).
    Includes a leading '▁' (space) token because Gemma naturally
    starts a number with one in its outputs."""
    s = str(n)
    ids = tok.encode(" " + s)  # " 391" → [<bos>, ▁▁, 3, 9, 1]
    # Drop <bos> and the double-space marker. Keep leading ▁ for
    # consistency with Gemma's natural emission style.
    if ids[0] == 2:  # <bos>
        ids = ids[1:]
    # The " " prefix typically becomes '▁▁' (double), but Gemma's own
    # outputs use single '▁' before a number. Use tok.encode("▁N").
    alt = tok.encode(s)
    if alt[0] == 2:
        alt = alt[1:]
    return alt  # ['▁', '3', '9', '1']


def parse_answer(text: str) -> int | None:
    """Extract first integer from continuation (ignore commas)."""
    nums = re.findall(r"-?\d+", text.replace(",", ""))
    return int(nums[0]) if nums else None


def generate_baseline(m, tok, prompt, max_tokens=60):
    """Autoregressive greedy decode, no injection."""
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


def generate_with_digit_bias(m, tok, prompt, digit_token_ids,
                             boost=50.0, max_tokens=60,
                             wait_marker_tokens=None):
    """Generate step-by-step. At each step, if we're inside the
    verified digit chain, add +boost to the next expected Gemma
    token. Otherwise let Gemma decide.

    wait_marker_tokens: optional list of token ids. If given, only
    START the digit chain AFTER Gemma has emitted one of these
    (heuristically, "▁", "=", etc. that signal "the answer is next").
    If None, start the chain on the very first emission.
    """
    from calm.llm_computer.gemma_substrate import KVCache
    ids = tok.encode(prompt)
    cache = KVCache(m.config.n_layers, device="cuda")
    gen = list(ids)

    # State: index into digit_token_ids. -1 = haven't started yet.
    digit_idx = 0 if wait_marker_tokens is None else -1

    with torch.no_grad():
        logits = m.forward(
            torch.tensor([gen]), device="cuda",
            kv_cache=cache, start_pos=0,
        )
        # Bias: if we're inside the digit chain, add +boost to next
        # expected digit.
        if digit_idx >= 0 and digit_idx < len(digit_token_ids):
            logits[0, -1, digit_token_ids[digit_idx]] += boost
            digit_idx += 1
        nxt = int(logits[0, -1].argmax())
        gen.append(nxt)

        for _ in range(max_tokens - 1):
            if nxt == tok.EOS_ID:
                break
            # If waiting for a marker and Gemma emitted one, activate.
            if digit_idx == -1 and nxt in (wait_marker_tokens or []):
                digit_idx = 0
            logits = m.forward(
                torch.tensor([[nxt]]), device="cuda",
                kv_cache=cache, start_pos=len(gen) - 1,
            )
            if digit_idx >= 0 and digit_idx < len(digit_token_ids):
                logits[0, -1, digit_token_ids[digit_idx]] += boost
                digit_idx += 1
            nxt = int(logits[0, -1].argmax())
            gen.append(nxt)

    return tok.decode(gen[len(ids):])


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer
    from calm.llm_computer.programs.multiplier import build_multiplier

    enable_triton_tq4(True)
    print("[mul-facade] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # Load multiplier
    print("[mul-facade] loading multiplier...")
    multiplier = build_multiplier().cuda().eval()

    # --- Baseline ---
    print("\n=== A: baseline Gemma (no facade) ===")
    baseline_results = []
    for prompt, expected in TESTS:
        text = generate_baseline(m, tok, prompt, max_tokens=60)
        got = parse_answer(text)
        ok = (got == expected)
        baseline_results.append((prompt, expected, got, ok))
        short = text.replace("\n", "\\n")[:60]
        print(f"  {'✓' if ok else '✗'} {prompt[:40]!r:<44} "
              f"expected={expected} got={got} | {short!r}")

    # --- Compute verified answers via multiplier (sanity) ---
    print("\n=== B: multiplier standalone verification ===")
    for prompt, expected in TESTS:
        # Extract operands from prompt
        nums = re.findall(r"\d+", prompt)
        a, b = int(nums[0]), int(nums[1])
        x = torch.tensor([[a, b]], device="cuda", dtype=torch.long)
        with torch.no_grad():
            got = int(multiplier(x)[0, 1].argmax().item())
        ok = got == expected
        print(f"  {'✓' if ok else '✗'} {a} × {b} = "
              f"{expected} (multiplier: {got})")

    # --- With step-through digit bias (facade simulation) ---
    print("\n=== C: step-through digit bias (multiplier → Gemma) ===")
    facade_results = []
    for prompt, expected in TESTS:
        # 1. Run multiplier on the operands
        nums = re.findall(r"\d+", prompt)
        a, b = int(nums[0]), int(nums[1])
        x = torch.tensor([[a, b]], device="cuda", dtype=torch.long)
        with torch.no_grad():
            verified = int(multiplier(x)[0, 1].argmax().item())
        # 2. Convert verified → Gemma token sequence (digits only)
        digit_ids = digits_as_gemma_tokens(tok, verified)
        # 3. Generate with per-step bias on the digit chain
        text = generate_with_digit_bias(
            m, tok, prompt, digit_ids,
            boost=50.0, max_tokens=60,
            wait_marker_tokens=None,  # start bias immediately
        )
        got = parse_answer(text)
        ok = got == expected
        facade_results.append((prompt, expected, got, ok))
        short = text.replace("\n", "\\n")[:60]
        print(f"  {'✓' if ok else '✗'} {prompt[:40]!r:<44} "
              f"expected={expected} got={got} | {short!r}")

    # --- Summary ---
    base_ok = sum(1 for _, _, _, ok in baseline_results if ok)
    fac_ok = sum(1 for _, _, _, ok in facade_results if ok)
    print(f"\n========== SUMMARY ==========")
    print(f"  baseline Gemma:         {base_ok}/{len(TESTS)}")
    print(f"  with multiplier facade: {fac_ok}/{len(TESTS)}")
    print(f"  improvement: +{fac_ok - base_ok}")


if __name__ == "__main__":
    sys.exit(main())
