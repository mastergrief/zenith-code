"""M1a — live A/B for 4 new auto-generated facades.

Ships Combinations, Permutations, Power, NextPrime via the Level-1
recursion pipeline (CALM-oracle validate → generate_facade → import →
install → A/B). Each spec follows the R46.2 / R22c / R53a skeleton;
zero human-written Python per facade.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
assert "m" in globals() and "tok" in globals(), (  # type: ignore[name-defined]
    "run via bin/gemma-run scripts/m1a_four_new_facades.py"
)

sys.path.insert(0, str(ROOT))
from calm.llm_computer.facades.retrieval import _monkey_patch_fast_encode
from calm.llm_computer.recursion import (
    COMBINATIONS_SPEC, PERMUTATIONS_SPEC, POWER_SPEC, NEXT_PRIME_SPEC,
    validate_facade, generate_facade, import_facade_class,
)

_monkey_patch_fast_encode(tok)  # type: ignore[name-defined]


def clear_card_state():
    for lyr in m.layers:  # type: ignore[name-defined]
        if hasattr(lyr, "card_slots"):
            lyr.card_slots = []
    m.verification_hooks = []  # type: ignore[name-defined]
    m.reserved_channels = []  # type: ignore[name-defined]


clear_card_state()


def score_numeric(expected: int, text: str) -> bool:
    nums = re.findall(r"-?\d+", text.replace(",", ""))
    for cand in nums[:3]:
        if int(cand) == expected:
            return True
    return False


# --- Oracle test sets (pre-generation gate) ---
ORACLES = {
    "Combinations": [((10, 3), 120), ((52, 5), 2598960), ((5, 2), 10), ((7, 4), 35), ((20, 4), 4845)],
    "Permutations": [((10, 3), 720), ((5, 2), 20), ((6, 4), 360), ((8, 3), 336), ((12, 4), 11880)],
    "Power":        [((2, 10), 1024), ((3, 5), 243), ((7, 4), 2401), ((2, 16), 65536), ((5, 6), 15625)],
    "NextPrime":    [((100,), 101), ((200,), 211), ((1000,), 1009), ((7,), 11), ((50,), 53)],
}

# --- Live A/B probes ---
PROBES = {
    "Combinations": [
        ("What is 10 choose 3?", 120),
        ("What is C(52, 5)?", 2598960),
        ("What is 7 choose 4?", 35),
        ("combinations of 20 taken 4", 4845),
        ("What is 8 choose 3?", 56),
    ],
    "Permutations": [
        ("What is 10 permute 3?", 720),
        ("What is P(8, 3)?", 336),
        ("permutations of 5 taken 2", 20),
        ("What is 12 permute 4?", 11880),
        ("What is P(6, 2)?", 30),
    ],
    "Power": [
        ("What is 2 to the power 10?", 1024),
        ("What is 3 to the 5th power?", 243),
        ("What is 7^4?", 2401),
        ("What is 2^16?", 65536),
        ("What is 5 raised to the 6?", 15625),
    ],
    "NextPrime": [
        ("What is the next prime after 100?", 101),
        ("What is the smallest prime greater than 200?", 211),
        ("What is the next prime after 1000?", 1009),
        ("next prime after 50", 53),
        ("prime after 37", 41),
    ],
}


SPECS = [
    ("Combinations", COMBINATIONS_SPEC),
    ("Permutations", PERMUTATIONS_SPEC),
    ("Power", POWER_SPEC),
    ("NextPrime", NEXT_PRIME_SPEC),
]


def run_ab(label, spec, oracle_cases, probes):
    print(f"\n--- {label} ---")
    passed, total = validate_facade(spec, oracle_cases)
    print(f"oracle validation: {passed}/{total}")
    if passed < total:
        print("  ✗ ORACLE REJECTED — skipping")
        return None

    path = generate_facade(spec, overwrite=True)
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        rel = path
    print(f"generated: {rel}  ({path.stat().st_size} bytes)")

    Cls = import_facade_class(spec)
    facade = Cls(device="cuda")
    facade.install(m, tok)  # type: ignore[name-defined]

    n_base_ok = 0
    n_card_ok = 0
    t0 = time.time()
    print(f"  {'prompt':<50} {'expected':>10} base    card    Δ")
    for prompt, expected in probes:
        r0 = facade.solve(prompt, use_bias=False)
        r1 = facade.solve(prompt, use_bias=True)
        base_ok = score_numeric(expected, r0.generated)
        card_ok = score_numeric(expected, r1.generated)
        if base_ok:
            n_base_ok += 1
        if card_ok:
            n_card_ok += 1
        bmark = "✓" if base_ok else "✗"
        cmark = "✓" if card_ok else "✗"
        p_short = prompt[:48] + ("..." if len(prompt) > 48 else "")
        print(f"  {p_short:<50} {expected:>10} "
              f"{str(r0.parsed_answer):>6}{bmark} "
              f"{str(r1.parsed_answer):>6}{cmark}")
    elapsed = time.time() - t0
    facade.detach()
    return {
        "label": label, "oracle": f"{passed}/{total}",
        "baseline": n_base_ok, "card": n_card_ok, "total": len(probes),
        "elapsed": elapsed, "path": str(rel),
    }


def main():
    print("========== M1a — 4 NEW AUTO-GENERATED FACADES ==========\n")
    results = []
    for label, spec in SPECS:
        r = run_ab(label, spec, ORACLES[label], PROBES[label])
        if r:
            results.append(r)

    print("\n\n========== SUMMARY ==========")
    total_base = sum(r["baseline"] for r in results)
    total_card = sum(r["card"] for r in results)
    total_probes = sum(r["total"] for r in results)
    for r in results:
        print(f"  {r['label']:<14} oracle={r['oracle']:<5} "
              f"base={r['baseline']}/{r['total']:<2} "
              f"card={r['card']}/{r['total']:<2} "
              f"Δ={r['card']-r['baseline']:+d}  ({r['elapsed']:.1f}s)")
    print(f"\n  TOTAL: base={total_base}/{total_probes}  "
          f"card={total_card}/{total_probes}  "
          f"Δ={total_card - total_base:+d}")

    # Receipt
    recpath = (ROOT / ".claude" / "MEMORY" / "evals"
               / "2026-04-22_m1a_four_new_facades.md")
    lines = [
        "# M1a — 4 new auto-generated facades",
        "",
        "Ships Combinations / Permutations / Power / NextPrime as",
        "decode-path facades via `calm/llm_computer/recursion.py`.",
        "Every facade is a `FacadeSpec` → oracle-validate →",
        "generate_facade → import → install → live A/B. Zero human-",
        "written Python per facade (the specs live in recursion.py",
        "module-level constants; the implementations are auto-generated).",
        "",
        "## Results",
        "",
        "| spec | oracle | baseline | card | Δ | wall |",
        "|---|:-:|:-:|:-:|:-:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['label']} | {r['oracle']} | "
            f"{r['baseline']}/{r['total']} | {r['card']}/{r['total']} | "
            f"{r['card']-r['baseline']:+d} | {r['elapsed']:.1f}s |"
        )
    lines += [
        "",
        f"**TOTAL**: baseline {total_base}/{total_probes} → "
        f"card {total_card}/{total_probes} (Δ={total_card - total_base:+d})",
        "",
        "## Generated files",
        "",
    ]
    for r in results:
        lines.append(f"- `{r['path']}`")
    recpath.write_text("\n".join(lines) + "\n")
    print(f"\n[m1a] receipt → {recpath}")


main()
print("M1A_DONE")
