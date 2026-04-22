"""R90a — HospitalDeck end-to-end demo.

Demonstrates the "customer verticals = card decks" pattern per
augmentation_thesis.md §"Customer verticals". A HospitalDeck bundles
medical-relevant facades (ICD-10 recall, DaysBetween, NumberTheory)
behind one entry point. Shows:

  1. ICD-10 diagnosis lookup (tier-3 text recall)
  2. Days-between clinical intervals (date arithmetic)
  3. Dosage/cycle GCD (NumberTheory)
  4. Mixed-domain planner routing correctness

Not an A/B — this is a functional demo showing the vertical deck
pattern works. The commercial claim is that a deck like this stands
up in hours with Level-2 MetaFacade (per commercial.md §"Decode-path
facade proliferation"), not days.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
assert "m" in globals() and "tok" in globals(), (  # type: ignore[name-defined]
    "run via bin/gemma-run scripts/r90a_hospital_deck_demo.py"
)

sys.path.insert(0, str(ROOT))

import importlib
import calm.llm_computer.facades.planner as planner_mod
importlib.reload(planner_mod)
import calm.llm_computer.facades.hospital_deck as hospital_mod
importlib.reload(hospital_mod)
from calm.llm_computer.facades.hospital_deck import HospitalDeck


def clear_card_state():
    for lyr in m.layers:  # type: ignore[name-defined]
        if hasattr(lyr, "card_slots"):
            lyr.card_slots = []
    m.verification_hooks = []  # type: ignore[name-defined]
    m.reserved_channels = []  # type: ignore[name-defined]


clear_card_state()
print("[r90a] cleared card state")


# Corpus: 9 probes spanning 3 medical-relevant domains.
PROBES = [
    # ICD-10 recall
    ("What is the diagnosis for ICD-10 code E11.9?", "icd10",
     ["diabetes", "mellitus"]),
    ("What is ICD-10 code J45.909?", "icd10",
     ["asthma"]),
    ("What is ICD-10 code T44.6X4D?", "icd10",
     ["poisoning", "alpha-adrenoreceptor"]),

    # Days between (admission/treatment intervals)
    ("How many days between 2024-01-15 and 2024-02-14?", "auto:days_between",
     ["30"]),
    ("days between 2023-06-01 and 2024-06-01", "auto:days_between",
     ["366"]),  # 2024 is leap
    ("days from 2024-09-01 to 2024-12-31", "auto:days_between",
     ["121"]),

    # GCD / dosage cycles (number_theory)
    ("What is GCD of 24 and 36?", "number_theory", ["12"]),
    ("What is LCM of 8 and 12?", "number_theory", ["24"]),

    # Multi-step arithmetic on doses
    ("What is 250 * 4?", "multi_step", ["1000"]),
]


def score_answer(expected_fragments: list[str], generated: str) -> bool:
    low = generated.lower()
    return all(frag.lower() in low for frag in expected_fragments)


def main():
    deck = HospitalDeck(
        device="cuda",
        icd10_db_path=ROOT / ".cache" / "icd10" / "icd10cm_codes_2022.json",
    )
    deck.build()
    deck.install(m, tok)  # type: ignore[name-defined]

    print(f"[r90a] shipped cards: {deck.shipped_cards()}")
    print()

    print("=== Phase 1: route classification ===")
    route_hits = 0
    for prompt, expected_tag, _ in PROBES:
        got = deck.classify(prompt)
        ok = got == expected_tag
        if ok:
            route_hits += 1
        mark = "✓" if ok else "✗"
        print(f"  {mark} expected={expected_tag!r:20} got={got!r:20} "
              f"{prompt!r}")
    print(f"\n  route: {route_hits}/{len(PROBES)}")

    print("\n=== Phase 2: answer correctness (deck with bias) ===")
    t0 = time.time()
    hits = 0
    recs = []
    for prompt, expected_tag, expected_fragments in PROBES:
        r = deck.solve(prompt, use_bias=True)
        ok = score_answer(expected_fragments, r.generated)
        if ok:
            hits += 1
        mark = "✓" if ok else "✗"
        print(f"  {mark} tag={r.facade!r:20} frags={expected_fragments!r:30} "
              f"got={r.generated[:50]!r}")
        recs.append({
            "prompt": prompt, "expected_tag": expected_tag, "tag": r.facade,
            "expected_fragments": expected_fragments,
            "generated": r.generated[:160],
            "used_bias": r.used_bias, "ok": ok,
        })
    elapsed = time.time() - t0
    print(f"\n  answer: {hits}/{len(PROBES)}  elapsed {elapsed:.1f}s")

    print("\n========== SUMMARY ==========")
    print(f"  corpus:     {len(PROBES)}")
    print(f"  route ok:   {route_hits}/{len(PROBES)}")
    print(f"  answer ok:  {hits}/{len(PROBES)}")

    recpath = (ROOT / ".claude" / "MEMORY" / "evals"
               / "2026-04-22_r90a_hospital_deck_demo.md")
    lines = [
        "# R90a — HospitalDeck verticalized composition demo",
        "",
        "Per `augmentation_thesis.md` §'Customer verticals = card decks'.",
        "Bundles ICD-10 recall + DaysBetween + NumberTheory under one",
        "deck entry point. Demonstrates the 'deck per vertical' pattern",
        "that scales to 100-domain hospital / legal / financial customers",
        "via Level-2 MetaFacade (hours per domain per recursion.md).",
        "",
        "## Corpus",
        "",
        f"{len(PROBES)} probes: 3 ICD-10 (incl. tier-3 stubborn code),",
        "3 date-arithmetic, 2 number-theory, 1 multi-step.",
        "",
        "## Results",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| total probes | {len(PROBES)} |",
        f"| route correct | {route_hits}/{len(PROBES)} |",
        f"| answer correct | {hits}/{len(PROBES)} |",
        f"| wall time | {elapsed:.1f}s |",
        "",
        "## Shipped cards in deck",
        "",
        f"{deck.shipped_cards()!r}",
        "",
    ]
    recpath.write_text("\n".join(lines) + "\n")
    print(f"\n[r90a] receipt → {recpath}")

    outjsonl = ROOT / ".cache" / "r90a_hospital_deck.jsonl"
    outjsonl.parent.mkdir(exist_ok=True)
    with outjsonl.open("w") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")


main()
print("R90A_DONE")
