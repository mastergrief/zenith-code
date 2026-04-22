"""R53a — test NumberTheoryFacade on prod Gemma.

Second compute facade (after R22c BaseConversion). Generalizes R46.2 +
R22c parse+safe_eval+step-through pattern to modular arithmetic + GCD
+ LCM — per the `compute_facades.md` candidate queue.

Hypothesis: Gemma 4 E4B fails on non-trivial modulo / GCD / LCM via
natural decoding, but succeeds when NumberTheoryFacade biases the emit
path to the decimal digit sequence of the exact result.

Corpus: 15 probes spanning easy (GCD(12,18)) to hard (GCD of coprime
large pairs, modulo across many digits). Baseline = stock Gemma,
card = facade.solve with use_bias=True.

Target: Δ ≥ 20% with zero regressions (compute-facades candidate-queue
rule).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
assert "m" in globals() and "tok" in globals(), (  # type: ignore[name-defined]
    "run via bin/gemma-run scripts/r53a_number_theory.py"
)

sys.path.insert(0, str(ROOT))
from calm.llm_computer.facades.number_theory import NumberTheoryFacade
from calm.llm_computer.facades.retrieval import _monkey_patch_fast_encode

_monkey_patch_fast_encode(tok)  # type: ignore[name-defined]


def main():
    # Mix: easy / medium / hard per op. 5 probes per op = 15 total.
    probes = [
        # --- Modulo ---
        ("What is 25 mod 7?", 4),              # easy
        ("What is 127 mod 13?", 10),           # medium
        ("What is 391 mod 23?", 0),            # medium (exact divisor)
        ("What is 1729 mod 97?", 80),          # hard (4-digit)
        ("What is 12345 mod 67?", 17),         # hard (5-digit)
        # --- GCD ---
        ("What is the GCD of 12 and 18?", 6),          # easy
        ("What is the GCD of 48 and 180?", 12),        # medium
        ("What is the GCD of 391 and 238?", 17),       # hard (3-digit coprime-ish)
        ("What is the greatest common divisor of 1729 and 935?", 1),  # coprime
        ("What is the GCD of 12345 and 6789?", 3),     # hard (5-digit)
        # --- LCM ---
        ("What is the LCM of 12 and 18?", 36),         # easy
        ("What is the LCM of 15 and 20?", 60),         # easy-medium
        ("What is the LCM of 48 and 180?", 720),       # medium
        ("What is the LCM of 100 and 75?", 300),       # medium
        ("What is the LCM of 127 and 91?", 11557),     # hard (5-digit result)
    ]
    print(f"[r53a] {len(probes)} probes (5 mod + 5 gcd + 5 lcm)")

    facade = NumberTheoryFacade(device="cuda")
    facade.install(m, tok)  # type: ignore[name-defined]

    print("\n=== BASELINE (no bias) vs FACADE (biased) ===")
    print(f"  {'prompt':<55}  {'expected':>10}  {'baseline':>10}  {'card':>10}")
    n_base_ok = 0
    n_card_ok = 0
    regressions = []
    t0 = time.time()
    for prompt, expected in probes:
        r0 = facade.solve(prompt, use_bias=False)
        r1 = facade.solve(prompt, use_bias=True)
        base_ok = r0.parsed_answer == expected
        card_ok = r1.parsed_answer == expected
        if base_ok:
            n_base_ok += 1
        if card_ok:
            n_card_ok += 1
        if base_ok and not card_ok:
            regressions.append({
                "prompt": prompt, "expected": expected,
                "baseline": r0.parsed_answer, "card": r1.parsed_answer,
                "op": r1.op, "operands": r1.operands,
            })
        mark_base = "✓" if base_ok else "✗"
        mark_card = "✓" if card_ok else "✗"
        short = prompt[:53] + ("..." if len(prompt) > 53 else "")
        print(f"  {short:<55}  {expected:>10}  "
              f"{str(r0.parsed_answer):>8}{mark_base}  "
              f"{str(r1.parsed_answer):>8}{mark_card}")

    elapsed = time.time() - t0
    print(f"\n=== SUMMARY (elapsed {elapsed:.1f}s) ===")
    print(f"  baseline: {n_base_ok}/{len(probes)}")
    print(f"  facade:   {n_card_ok}/{len(probes)}  (Δ={n_card_ok - n_base_ok:+d})")
    print(f"  regressions: {len(regressions)}")
    for r in regressions:
        print(f"    {r['prompt']!r} exp={r['expected']} base={r['baseline']} card={r['card']}")

    # Receipt
    recpath = (ROOT / ".claude" / "MEMORY" / "evals"
               / "2026-04-22_r53a_number_theory_facade.md")
    lines = [
        "# R53a — NumberTheoryFacade (second compute facade)",
        "",
        "Decode-path tier-2 facade (parse → safe_eval → step-through",
        "digit bias) for modular arithmetic + GCD + LCM. Generalizes",
        "R46.2 (infix arithmetic) and R22c (base conversion).",
        "",
        "## A/B (15-probe corpus, stock Gemma 4 E4B tq4)",
        "",
        f"| metric | value |",
        f"|---|---:|",
        f"| baseline | {n_base_ok}/{len(probes)} |",
        f"| facade   | {n_card_ok}/{len(probes)} |",
        f"| Δ        | {n_card_ok - n_base_ok:+d} |",
        f"| regressions | {len(regressions)} |",
        f"| wall time | {elapsed:.1f}s |",
        "",
        "## Corpus",
        "",
        "5 mod + 5 gcd + 5 lcm, mixed easy/medium/hard per op.",
        "",
    ]
    if regressions:
        lines += ["## Regressions", ""]
        for r in regressions:
            lines.append(f"- `{r['prompt']}` exp={r['expected']} "
                         f"base={r['baseline']} card={r['card']}")
        lines.append("")
    recpath.write_text("\n".join(lines) + "\n")
    print(f"\n[r53a] receipt → {recpath}")


main()
print("R53A_DONE")
