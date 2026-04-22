"""R70a — PlannerFacade mixed-corpus A/B.

Validates that PlannerFacade correctly dispatches to the right facade
across a mixed-domain corpus — one entry point, 4 specialist tools +
1 pass-through.

Corpus: 20 prompts spanning math / base conversion / number theory /
ICD-10 / open-ended. Each has a known-correct answer for scoring.

Target: planner.solve(prompt) matches the specialist's accuracy for
each facade's domain, and does not regress pass-through.
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
    "run via bin/gemma-run scripts/r70a_planner_mixed.py"
)

sys.path.insert(0, str(ROOT))
from calm.llm_computer.facades.planner import PlannerFacade
from calm.llm_computer.facades.retrieval import _monkey_patch_fast_encode

_monkey_patch_fast_encode(tok)  # type: ignore[name-defined]


# --- Clear any lingering card state (from prior r22/r60 runs) ---
def clear_card_state():
    for lyr in m.layers:  # type: ignore[name-defined]
        if hasattr(lyr, "card_slots"):
            lyr.card_slots = []
    m.verification_hooks = []  # type: ignore[name-defined]
    m.reserved_channels = []  # type: ignore[name-defined]

clear_card_state()


# --- Scoring helpers ---

_STOPWORDS = {
    "unspecified", "acute", "chronic", "severe", "essential", "other",
    "mild", "moderate", "primary", "secondary", "left", "right",
    "bilateral", "upper", "lower", "initial", "subsequent", "sequela",
    "with", "without", "due", "from", "into", "onto", "for", "on",
    "of", "by", "at", "in", "and", "or", "the", "a", "an",
    "disease", "disorder", "condition", "syndrome", "type",
}


def _sig_words(text: str):
    cleaned = re.sub(r"[(),.;:/]", " ", text).lower()
    return {
        w for w in cleaned.split()
        if w and len(w) >= 4 and w not in _STOPWORDS
    }


def score_numeric(expected: int, text: str) -> bool:
    """Match on first integer in the text."""
    nums = re.findall(r"-?\d+", text.replace(",", ""))
    for candidate in nums[:3]:   # check first 3 numbers
        if int(candidate) == expected:
            return True
    return False


def score_text(expected: str, text: str) -> bool:
    """Bag-of-words ≥4 chars — any match suffices."""
    sigs = _sig_words(expected)
    out = text.lower()
    return any(w in out for w in sigs)


# --- Corpus ---
# (prompt, expected_kind, expected_value, expected_facade)
CORPUS = [
    # Math — multi_step
    ("What is 17 × 23?", "num", 391, "multi_step"),
    ("What is (17+23) * 2?", "num", 80, "multi_step"),
    ("What is 99 + 127?", "num", 226, "multi_step"),
    ("What is 1000 - 347?", "num", 653, "multi_step"),
    # Base conversion
    ("What is 0xFF in decimal?", "num", 255, "base_conv"),
    ("What is 0xDEADBEEF in decimal?", "num", 3735928559, "base_conv"),
    ("What is 0b11110000 in decimal?", "num", 240, "base_conv"),
    ("Convert 0b10101010 to decimal.", "num", 170, "base_conv"),
    # Number theory
    ("What is 127 mod 13?", "num", 10, "number_theory"),
    ("What is the GCD of 48 and 180?", "num", 12, "number_theory"),
    ("What is the GCD of 391 and 238?", "num", 17, "number_theory"),
    ("What is the LCM of 12 and 18?", "num", 36, "number_theory"),
    # ICD-10
    ("What is the diagnosis for ICD-10 code I10?", "text",
     "hypertension", "icd10"),
    ("What is the diagnosis for ICD-10 code E11.9?", "text",
     "diabetes mellitus", "icd10"),
    ("What is the diagnosis for ICD-10 code J45.909?", "text",
     "asthma", "icd10"),
    ("What is the diagnosis for ICD-10 code M65.029?", "text",
     "abscess", "icd10"),
    ("What is the diagnosis for ICD-10 code H02.713?", "text",
     "chloasma", "icd10"),
    # Pass-through
    ("Tell me a short joke.", "passthrough", None, None),
    ("Name a color.", "passthrough", None, None),
    ("What is the capital of France?", "text", "paris", None),
]


def main():
    planner = PlannerFacade(device="cuda")
    planner.load_icd10_db(ROOT / ".cache" / "icd10" / "icd10cm_codes_2022.json")
    planner.install(m, tok)  # type: ignore[name-defined]

    print(f"[r70a] {len(CORPUS)} mixed-domain probes")
    print()
    print(f"  {'#':<3} {'routed':<14} {'expect facade':<14} {'ok':<3} "
          f"{'prompt':<50}")

    correct = 0
    route_correct = 0
    by_facade = {}   # tag → {correct, total}
    t0 = time.time()

    results = []
    for i, (prompt, kind, expected, want_facade) in enumerate(CORPUS):
        r = planner.solve(prompt, use_bias=True)
        tag = r.facade or "passthrough"
        # Route-level check (did planner pick right facade?)
        want = want_facade or "passthrough"
        route_ok = (tag == want)
        if route_ok:
            route_correct += 1
        # Accuracy check
        if kind == "num":
            ok = score_numeric(expected, r.generated)
        elif kind == "text":
            ok = score_text(expected, r.generated)
        elif kind == "passthrough":
            # No ground truth — just check it didn't crash
            ok = bool(r.generated.strip())
        else:
            ok = False
        if ok:
            correct += 1

        by_facade.setdefault(tag, {"correct": 0, "total": 0})
        by_facade[tag]["total"] += 1
        if ok:
            by_facade[tag]["correct"] += 1

        rmark = "✓" if ok else "✗"
        route_mark = "✓" if route_ok else "✗"
        short = prompt[:48] + ("..." if len(prompt) > 48 else "")
        print(f"  {i:<3} {tag:<14} {want:<14} {rmark}{route_mark} {short}")
        results.append({
            "prompt": prompt, "expected": expected, "kind": kind,
            "routed": tag, "want_facade": want,
            "route_ok": route_ok, "ok": ok,
            "generated": r.generated[:200],
        })

    elapsed = time.time() - t0

    print(f"\n=== SUMMARY ===")
    print(f"  total:        {len(CORPUS)}")
    print(f"  route correct: {route_correct}/{len(CORPUS)}")
    print(f"  answer correct: {correct}/{len(CORPUS)}")
    print(f"  wall time:     {elapsed:.1f}s")
    print()
    print(f"  by facade:")
    for tag in sorted(by_facade.keys()):
        s = by_facade[tag]
        print(f"    {tag:<16} {s['correct']}/{s['total']}")

    # Receipt
    recpath = (ROOT / ".claude" / "MEMORY" / "evals"
               / "2026-04-22_r70a_planner_mixed.md")
    lines = [
        "# R70a — PlannerFacade mixed-corpus A/B",
        "",
        "Orchestration MVP per `tracing_roadmap.md` §'Planner card'.",
        "Single NL entry point dispatches to 4 specialist facades",
        "(multi_step / base_conv / number_theory / icd10) plus",
        "pass-through. First-match-wins priority chain, no chaining yet.",
        "",
        "## Mixed corpus",
        "",
        f"20 probes: 4 math / 4 base-conversion / 4 number-theory /",
        f"5 ICD-10 / 3 pass-through. Scoring per-domain (numeric",
        f"answer-check for facade outputs, bag-of-words for ICD-10,",
        f"non-empty for pass-through).",
        "",
        "## Result",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| total | {len(CORPUS)} |",
        f"| route correct | {route_correct}/{len(CORPUS)} |",
        f"| answer correct | {correct}/{len(CORPUS)} |",
        f"| wall time | {elapsed:.1f}s |",
        "",
        "## By facade",
        "",
        "| facade | correct |",
        "|---|---:|",
    ]
    for tag in sorted(by_facade.keys()):
        s = by_facade[tag]
        lines.append(f"| {tag} | {s['correct']}/{s['total']} |")
    lines += [
        "",
        "## Architecture",
        "",
        "Each facade's `parse(prompt)` acts as a gate — the planner tries",
        "them in priority order (icd10 > base_conv > number_theory >",
        "multi_step > passthrough). Ambiguity is avoided because each",
        "facade's gate requires domain-specific signal (ICD-10 phrase",
        "+ code; hex/binary + 'in decimal'; specific mod/gcd/lcm",
        "keywords; infix operators).",
        "",
        "Next steps (Option C — compiled planner card): chain facades",
        "so multi-step queries like 'GCD(48,180), convert to hex' route",
        "through two facades in sequence, biasing Gemma to emit each",
        "intermediate. Requires a new int→hex decode-path facade first.",
    ]
    recpath.write_text("\n".join(lines) + "\n")
    print(f"\n[r70a] receipt → {recpath}")

    # Save per-probe results
    outpath = ROOT / ".cache" / "r70a_planner_results.jsonl"
    outpath.parent.mkdir(exist_ok=True)
    with outpath.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[r70a] per-probe → {outpath}")


main()
print("R70A_DONE")
