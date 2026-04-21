"""R22c — test BaseConversionFacade on prod Gemma.

R46.2-style tier-2 card for a new domain: hex/binary → decimal.
Demonstrates the parse → safe_eval → step-through bias pattern
generalizes beyond arithmetic infix chains.

Hypothesis: Gemma 4 E4B fails on non-trivial hex-to-decimal conversions
(e.g., 0xDEADBEEF = 3735928559) via natural decoding, but succeeds when
BaseConversionFacade biases the emit path toward the digit sequence.

Corpus: 10 probe prompts spanning easy (0xFF, 0b1010) to hard
(0xDEADBEEF, 0xCAFEBABE). Baseline = stock Gemma, card = facade.solve.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
assert "m" in globals() and "tok" in globals(), (  # type: ignore[name-defined]
    "run via bin/gemma-run scripts/r22c_base_conversion.py"
)

sys.path.insert(0, str(ROOT))
from calm.llm_computer.facades.base_conversion import BaseConversionFacade
from calm.llm_computer.facades.retrieval import _monkey_patch_fast_encode

_monkey_patch_fast_encode(tok)  # type: ignore[name-defined]


def main():
    # Probe corpus — mix of easy and hard conversions
    probes = [
        ("What is 0xFF in decimal?", 255),
        ("What is 0x100 in decimal?", 256),
        ("What is 0x7F in decimal?", 127),
        ("Convert 0xDEAD to decimal.", 57005),
        ("What is 0xCAFE in decimal?", 51966),
        ("What is 0xDEADBEEF in decimal?", 3735928559),
        ("What is 0xCAFEBABE in decimal?", 3405691582),
        ("What is 0b1010 in decimal?", 10),
        ("What is 0b11110000 in decimal?", 240),
        ("What is 0b10101010 in decimal?", 170),
    ]
    print(f"[r22c] {len(probes)} probes")

    facade = BaseConversionFacade(device="cuda")
    facade.install(m, tok)  # type: ignore[name-defined]

    print("\n=== BASELINE (no bias) vs FACADE (biased) ===")
    print(f"  {'prompt':<52}  {'expected':>12}  {'baseline':>10}  {'card':>10}")
    n_base_ok = 0
    n_card_ok = 0
    t0 = time.time()
    for prompt, expected in probes:
        # Baseline — facade.solve with use_bias=False
        r0 = facade.solve(prompt, use_bias=False)
        # With bias
        r1 = facade.solve(prompt, use_bias=True)
        base_ok = r0.parsed_answer == expected
        card_ok = r1.parsed_answer == expected
        if base_ok:
            n_base_ok += 1
        if card_ok:
            n_card_ok += 1
        mark_base = "✓" if base_ok else "✗"
        mark_card = "✓" if card_ok else "✗"
        short = prompt[:50] + ("..." if len(prompt) > 50 else "")
        print(f"  {short:<52}  {expected:>12}  "
              f"{str(r0.parsed_answer):>8}{mark_base}  "
              f"{str(r1.parsed_answer):>8}{mark_card}")

    print(f"\n=== SUMMARY (elapsed {time.time() - t0:.1f}s) ===")
    print(f"  baseline: {n_base_ok}/{len(probes)}")
    print(f"  facade:   {n_card_ok}/{len(probes)}  (Δ={n_card_ok - n_base_ok:+d})")


main()
print("R22C_DONE")
