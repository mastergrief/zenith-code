"""Round 44: validate HubInjectionCard by reproducing R42 + R43 numbers.

Raw-path gate: facade logits on one prompt must match R43's inline
ForcedAttentionOutput to numerical noise (max abs diff < 1e-4). Same
tensor math, different packaging — any divergence is a bug.

User-facing gate: facade reproduces R42 SV agreement (≥ 8/10 match,
|Δ| < 3.0), R43a comparison (≥ 17/18 match — R43 hit 18/18), and R43b
counting (≥ 6/6 match).

Commit gate: all three user-facing capability runs match their R42/R43
numbers exactly (or within 1 argmax, allowing for CUDA non-determinism
across runs).
"""

from __future__ import annotations

import math
import os
import random
import sys

import torch

GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")


# --- Prompt sets (identical to R42/R43 scripts) ---

SV_PROMPTS = [
    ("The cat that sits near the window", "sing"),
    ("The cats that sit near the window", "plur"),
    ("The dog with the red collar", "sing"),
    ("The dogs with the red collar", "plur"),
    ("The teacher with the students", "sing"),
    ("The teachers with the student", "plur"),
    ("The key to the cabinets", "sing"),
    ("The keys to the cabinet", "plur"),
    ("The farmer beside the horses", "sing"),
    ("The farmers beside the horse", "plur"),
]

COMPARISON_PROMPTS = []
random.seed(0)
for _ in range(20):
    a = random.randint(1, 9)
    b = random.randint(1, 9)
    if a == b:
        continue
    COMPARISON_PROMPTS.append(
        (f"Which is larger, {a} or {b}? Answer: ", max(a, b)))

COUNTING_PROMPTS = []
random.seed(0)
for _ in range(20):
    length = random.randint(4, 7)
    start = random.randint(1, 9)
    nums = list(range(start, start + length))
    nxt = start + length
    if nxt > 9:
        continue
    COUNTING_PROMPTS.append(
        ("Count: " + ", ".join(str(x) for x in nums) + ", ", nxt))


def run_sv(facade, tok):
    print(f"\n=== SV agreement ({len(SV_PROMPTS)} prompts) ===")
    matches, n, sum_abs = 0, 0, 0.0
    for prompt, _expected in SV_PROMPTS:
        ids = torch.tensor([tok.encode(prompt)], device="cuda")

        logits_base = facade.forward(ids, inject=False)
        base_argmax = int(logits_base[0, -1].argmax())
        base_tok = tok.id_to_token.get(base_argmax, '?')
        base_logit = logits_base[0, -1, base_argmax].item()

        logits_forced = facade.forward(ids, inject=True)
        force_argmax = int(logits_forced[0, -1].argmax())
        force_tok = tok.id_to_token.get(force_argmax, '?')
        force_logit = logits_forced[0, -1, base_argmax].item()

        match = force_argmax == base_argmax
        delta = force_logit - base_logit
        sum_abs += abs(delta)
        n += 1
        if match:
            matches += 1
        print(f"  {prompt!r:>42}  base={base_tok!r:<10} "
              f"forced={force_tok!r:<10} {'Y' if match else 'N'} "
              f"Δ={delta:+.2f}")
    print(f"  mean |Δ|={sum_abs/max(n,1):.3f}  matches={matches}/{n}")
    return matches, n, sum_abs


def run_numeric(facade, tok, prompts, cap_name, target_digit_match: bool):
    """Comparison / counting. Only score on prompts where Gemma gets
    the right answer at baseline (mirrors R43's `stripped != str(expected)`
    filter)."""
    print(f"\n=== {cap_name} ({len(prompts)} prompts) ===")
    matches, n_clean, sum_abs = 0, 0, 0.0
    for prompt, expected in prompts:
        ids = torch.tensor([tok.encode(prompt)], device="cuda")

        logits_base = facade.forward(ids, inject=False)
        base_argmax = int(logits_base[0, -1].argmax())
        base_tok = tok.id_to_token.get(base_argmax, '?')
        base_logit = logits_base[0, -1, base_argmax].item()

        if target_digit_match:
            stripped = base_tok.lstrip('▁')
            if stripped != str(expected):
                continue
        n_clean += 1

        logits_forced = facade.forward(ids, inject=True)
        force_argmax = int(logits_forced[0, -1].argmax())
        force_tok = tok.id_to_token.get(force_argmax, '?')
        force_logit = logits_forced[0, -1, base_argmax].item()

        match = force_argmax == base_argmax
        delta = force_logit - base_logit
        sum_abs += abs(delta)
        if match:
            matches += 1
        short = prompt if len(prompt) <= 40 else prompt[:40]
        print(f"  {short!r:>42}  base={base_tok!r:<6} "
              f"forced={force_tok!r:<6} {'Y' if match else 'N'} "
              f"Δ={delta:+.2f}")
    print(f"  mean |Δ|={sum_abs/max(n_clean,1):.3f}  "
          f"matches={matches}/{n_clean}")
    return matches, n_clean, sum_abs


def raw_path_equivalence(facade, tok):
    """Take one SV prompt, run facade AND the R43-style inline forced
    forward. Assert logits match to numerical noise."""
    from scripts.test_l23_forced_cross_task import (
        forward_with_forced_attn, get_natural_top_positions,
    )
    print(f"\n=== Raw-path equivalence vs R43 inline ===")
    # Use a comparison prompt (R43-style)
    prompt, _expected = COMPARISON_PROMPTS[0]
    ids = torch.tensor([tok.encode(prompt)], device="cuda")
    m = facade._installed_on

    # Same natural-top computation via R43 helper
    top_pos = get_natural_top_positions(m, ids, facade.target_layer)
    logits_r43 = forward_with_forced_attn(
        m, ids, pos_h1=top_pos[1], pos_h4=top_pos[4])

    # Facade path — pass same positions to remove detection-phase noise
    logits_facade = facade.forward(ids, inject=True, positions=top_pos)

    diff = (logits_r43 - logits_facade).abs().max().item()
    print(f"  prompt: {prompt!r}")
    print(f"  positions: H1={top_pos[1]}  H4={top_pos[4]}")
    print(f"  max |Δlogits|: {diff:.2e}")
    ok = diff < 1e-4
    print(f"  raw-path equivalence: {'PASS' if ok else 'FAIL'}")
    return ok, diff


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4)
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer
    from calm.llm_computer.facades import HubInjectionCard

    enable_triton_tq4(True)
    print("[r44] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    card = HubInjectionCard()
    card.install(m)
    print(f"[r44] installed: target L{card.target_layer}, "
          f"heads={card.heads}")

    # Raw-path gate
    rp_ok, rp_diff = raw_path_equivalence(card, tok)

    # User-facing gates
    m_sv, n_sv, d_sv = run_sv(card, tok)
    m_cmp, n_cmp, d_cmp = run_numeric(
        card, tok, COMPARISON_PROMPTS, "COMPARISON",
        target_digit_match=True)
    m_cnt, n_cnt, d_cnt = run_numeric(
        card, tok, COUNTING_PROMPTS, "COUNTING",
        target_digit_match=True)

    print(f"\n\n=== ROUND 44 SUMMARY ===")
    print(f"  raw-path:     diff={rp_diff:.2e}  "
          f"{'PASS' if rp_ok else 'FAIL'}")
    print(f"  R42 baseline (SV):         mean|Δ|=0.467  8/10")
    print(f"  R44 facade  (SV):          mean|Δ|={d_sv/max(n_sv,1):.3f}  "
          f"{m_sv}/{n_sv}")
    print(f"  R43a baseline (comp):      mean|Δ|=0.176  18/18")
    print(f"  R44 facade   (comp):       mean|Δ|={d_cmp/max(n_cmp,1):.3f}  "
          f"{m_cmp}/{n_cmp}")
    print(f"  R43b baseline (count):     mean|Δ|=0.528  6/6")
    print(f"  R44 facade   (count):      mean|Δ|={d_cnt/max(n_cnt,1):.3f}  "
          f"{m_cnt}/{n_cnt}")

    # Ship gate: within-1 argmax of published numbers
    sv_ok = m_sv >= 7 and d_sv / max(n_sv, 1) < 3.0
    cmp_ok = m_cmp >= min(17, n_cmp) and d_cmp / max(n_cmp, 1) < 3.0
    cnt_ok = m_cnt == n_cnt and d_cnt / max(n_cnt, 1) < 3.0

    print(f"\n  Gates: raw={rp_ok}  SV={sv_ok}  comp={cmp_ok}  cnt={cnt_ok}")
    if rp_ok and sv_ok and cmp_ok and cnt_ok:
        print(f"\n  ✓ HubInjectionCard VALIDATED. First hub-first")
        print(f"    compiled-card-as-facade shipped. 4-for-1 ROI preserved.")
    else:
        print(f"\n  ✗ Facade diverges from R42/R43 numbers. Investigate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
