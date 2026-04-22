"""R70c — PlannerFacade auto-facade dispatch A/B.

Verifies that DEFAULT_AUTO_FACADES (factorial, fibonacci, combinations,
permutations, power, next_prime) are routed correctly by PlannerFacade.classify()
and solve() delivers the exact CALM-oracle answer via step-through bias.

Pattern: mirror r70a (mixed single-facade corpus). 2 probes per auto-facade
= 12 probes. Route-accuracy + answer-accuracy A/B (use_bias=True vs False).
Zero-regression on existing facade corpus (implicit — auto-facade dispatch
sits BEFORE multi_step catch-all; no existing probe uses auto-facade keywords).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
assert "m" in globals() and "tok" in globals(), (  # type: ignore[name-defined]
    "run via bin/gemma-run scripts/r70c_planner_auto_facades.py"
)

sys.path.insert(0, str(ROOT))

# Force-reload to pick up any facade module edits since daemon start.
import importlib
import calm.llm_computer.facades.planner as planner_mod
importlib.reload(planner_mod)
from calm.llm_computer.facades.planner import PlannerFacade

import calm.llm_computer.facades.icd10_recall as icd10_mod
importlib.reload(icd10_mod)


def clear_card_state():
    for lyr in m.layers:  # type: ignore[name-defined]
        if hasattr(lyr, "card_slots"):
            lyr.card_slots = []
    m.verification_hooks = []  # type: ignore[name-defined]
    m.reserved_channels = []  # type: ignore[name-defined]


clear_card_state()
print("[r70c] cleared card state")


# Each probe: (prompt, expected_tag, expected_answer). 2 probes per facade.
PROBES = [
    # Factorial
    ("What is 5 factorial?",            "auto:factorial",    120),
    ("What is factorial of 7?",         "auto:factorial",    5040),
    # Fibonacci
    ("What is the 10th fibonacci?",     "auto:fibonacci",    55),
    ("fibonacci of 15",                 "auto:fibonacci",    610),
    # Combinations
    ("What is 10 choose 3?",            "auto:combinations", 120),
    ("C(8, 2)",                         "auto:combinations", 28),
    # Permutations
    ("What is 5 permute 3?",            "auto:permutations", 60),
    ("permutations(6, 2)",              "auto:permutations", 30),
    # Power
    ("2 to the power 10",               "auto:power",        1024),
    ("pow(3, 5)",                       "auto:power",        243),
    # Next prime
    ("next prime after 100",            "auto:next_prime",   101),
    ("smallest prime greater than 50",  "auto:next_prime",   53),
]


def score(generated: str, expected: int) -> bool:
    normalized = generated.replace(",", "")
    m_ = re.search(r"-?\d{1,15}", normalized)
    if not m_:
        return False
    return int(m_.group(0)) == expected


def main():
    planner = PlannerFacade(device="cuda", register_auto=True)
    tags = [t for t, _ in planner.auto_facades]
    print(f"[r70c] auto-facades: {tags}")

    # Classify-only check (pure, no Gemma)
    print("\n=== Phase 1: route classification ===")
    route_hits = 0
    for prompt, expected_tag, _exp_val in PROBES:
        got = planner.classify(prompt)
        ok = got == expected_tag
        if ok:
            route_hits += 1
        mark = "✓" if ok else "✗"
        print(f"  {mark} expected={expected_tag!r:22} got={got!r:22} {prompt!r}")
    print(f"\n  route: {route_hits}/{len(PROBES)}")

    planner.install(m, tok)  # type: ignore[name-defined]

    print("\n=== Phase 2: answer correctness (auto-facade with bias) ===")
    t0 = time.time()
    answer_hits = 0
    recs = []
    for prompt, expected_tag, expected_value in PROBES:
        r = planner.solve(prompt, use_bias=True)
        ok = score(r.generated, expected_value)
        if ok:
            answer_hits += 1
        mark = "✓" if ok else "✗"
        print(f"  {mark} tag={r.facade!r:22} exp={expected_value:<8} "
              f"got={r.generated[:60]!r}")
        recs.append({
            "prompt": prompt, "expected_tag": expected_tag, "tag": r.facade,
            "expected": expected_value, "generated": r.generated[:120],
            "used_bias": r.used_bias, "ok": ok,
        })
    elapsed = time.time() - t0
    print(f"\n  answer: {answer_hits}/{len(PROBES)}  elapsed {elapsed:.1f}s")

    print("\n=== Phase 3: baseline (use_bias=False) ===")
    t0 = time.time()
    base_hits = 0
    for rec in recs:
        prompt = rec["prompt"]
        r = planner.solve(prompt, use_bias=False)
        ok = score(r.generated, rec["expected"])
        rec["baseline_ok"] = ok
        rec["baseline_out"] = r.generated[:120]
        if ok:
            base_hits += 1
        mark = "✓" if ok else "✗"
        print(f"  {mark} exp={rec['expected']:<8} got={r.generated[:60]!r}")
    elapsed_b = time.time() - t0
    print(f"\n  baseline: {base_hits}/{len(PROBES)}  elapsed {elapsed_b:.1f}s")

    print("\n========== SUMMARY ==========")
    print(f"  route:    {route_hits}/{len(PROBES)}")
    print(f"  answer:   {answer_hits}/{len(PROBES)}")
    print(f"  baseline: {base_hits}/{len(PROBES)}  (Δ={answer_hits - base_hits:+d})")

    # Receipt
    recpath = (ROOT / ".claude" / "MEMORY" / "evals"
               / "2026-04-22_r70c_planner_auto_facades.md")
    lines = [
        "# R70c — PlannerFacade auto-facade dispatch A/B",
        "",
        "Registers auto-generated Level-1/2 facades (factorial, fibonacci,",
        "combinations, permutations, power, next_prime) with the Planner so",
        "user queries reach them instead of falling through to multi_step",
        "catch-all.",
        "",
        "## Corpus",
        "",
        f"{len(PROBES)} probes (2 per facade), covering canonical + variant",
        "regex patterns from the auto-generated `_PARSE_RES` list.",
        "",
        "## Results",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| total probes | {len(PROBES)} |",
        f"| route accuracy | {route_hits}/{len(PROBES)} |",
        f"| answer (with bias) | {answer_hits}/{len(PROBES)} |",
        f"| baseline (no bias) | {base_hits}/{len(PROBES)} |",
        f"| Δ (bias − baseline) | {answer_hits - base_hits:+d} |",
        f"| wall Phase 2 | {elapsed:.1f}s |",
        f"| wall Phase 3 | {elapsed_b:.1f}s |",
        "",
    ]
    recpath.write_text("\n".join(lines) + "\n")
    print(f"\n[r70c] receipt → {recpath}")

    outjsonl = ROOT / ".cache" / "r70c_planner_auto_facades.jsonl"
    outjsonl.parent.mkdir(exist_ok=True)
    with outjsonl.open("w") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")


main()
print("R70C_DONE")
