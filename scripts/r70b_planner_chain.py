"""R70b — PlannerFacade chain test: cross-domain 2-step queries.

Extends r70a (single-facade dispatch) with 2-step chains:
  "X in hex/binary/octal" where X is evaluated by the primary facade
  (number_theory or multi_step), then encoded via NumericEncodeFacade.

Corpus: 12 chain probes spanning:
  - NumberTheory → hex/binary/octal
  - MultiStep → hex/binary/octal

Target: planner classifies the chain correctly AND produces the right
encoded answer, matching the planner's ability to compose facades.
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
    "run via bin/gemma-run scripts/r70b_planner_chain.py"
)

sys.path.insert(0, str(ROOT))
from calm.llm_computer.facades.planner import PlannerFacade
from calm.llm_computer.facades.retrieval import _monkey_patch_fast_encode

_monkey_patch_fast_encode(tok)  # type: ignore[name-defined]


def clear_card_state():
    for lyr in m.layers:  # type: ignore[name-defined]
        if hasattr(lyr, "card_slots"):
            lyr.card_slots = []
    m.verification_hooks = []  # type: ignore[name-defined]
    m.reserved_channels = []  # type: ignore[name-defined]


clear_card_state()


def score_match(expected: str, text: str) -> bool:
    """Case-insensitive substring match. Expected may be multi-char
    (e.g. 'DEADBEEF', '1010', '777')."""
    return expected.lower() in text.lower()


# --- Corpus: (prompt, expected_primary_value, expected_encoded, base) ---
CORPUS = [
    # NumberTheory → hex
    ("What is the GCD of 48 and 180 in hex?", 12, "C", 16),
    ("What is 127 mod 13 in hex?", 10, "A", 16),
    ("What is the LCM of 12 and 18 in hex?", 36, "24", 16),
    # NumberTheory → binary
    ("What is 127 mod 13 in binary?", 10, "1010", 2),
    ("What is the GCD of 48 and 180 in binary?", 12, "1100", 2),
    # NumberTheory → octal
    ("What is the LCM of 15 and 20 in octal?", 60, "74", 8),
    # MultiStep → hex
    ("What is 17 * 23 in hex?", 391, "187", 16),
    ("What is 100 + 55 in hex?", 155, "9B", 16),
    ("What is (17+23)*2 in hex?", 80, "50", 16),
    # MultiStep → binary
    ("What is 10 + 22 in binary?", 32, "100000", 2),
    # MultiStep → octal
    ("What is 17 * 23 in octal?", 391, "607", 8),
    # Sanity: direct numeric_encode (no chain)
    ("What is 255 in hex?", 255, "FF", 16),
]


def main():
    planner = PlannerFacade(device="cuda")
    planner.load_icd10_db(ROOT / ".cache" / "icd10" / "icd10cm_codes_2022.json")
    planner.install(m, tok)  # type: ignore[name-defined]

    print(f"[r70b] {len(CORPUS)} chain probes")
    print(f"  {'#':<3} {'route':<30} {'ok':<3} prompt  → expected")

    n_route = 0
    n_answer = 0
    results = []
    t0 = time.time()

    for i, (prompt, primary_val, expected_enc, base) in enumerate(CORPUS):
        tag = planner.classify(prompt)
        expected_tag_prefix = "chain:" if i < len(CORPUS) - 1 else "numeric_encode"
        route_ok = (tag or "").startswith(expected_tag_prefix)
        if route_ok:
            n_route += 1

        r = planner.solve(prompt, use_bias=True)
        ok = score_match(expected_enc, r.generated)
        if ok:
            n_answer += 1

        mark = "✓" if ok else "✗"
        rmark = "✓" if route_ok else "✗"
        print(f"  {i:<3} {str(tag):<32} {mark}{rmark} "
              f"{prompt!r:<55} → {expected_enc!r}  "
              f"(actual parsed {r.parsed_value!r})")
        results.append({
            "prompt": prompt, "expected_primary": primary_val,
            "expected_encoded": expected_enc, "base": base,
            "tag": tag, "route_ok": route_ok, "answer_ok": ok,
            "parsed_value": str(r.parsed_value),
            "generated": r.generated[:200],
            "chain_steps": r.chain_steps,
        })

    elapsed = time.time() - t0
    print(f"\n=== SUMMARY ===")
    print(f"  route correct:  {n_route}/{len(CORPUS)}")
    print(f"  answer correct: {n_answer}/{len(CORPUS)}")
    print(f"  wall time:      {elapsed:.1f}s")

    # Receipt
    recpath = (ROOT / ".claude" / "MEMORY" / "evals"
               / "2026-04-22_r70b_planner_chain.md")
    lines = [
        "# R70b — PlannerFacade 2-step chain (Option C step-1)",
        "",
        "Extends r70a single-facade dispatch with cross-domain chains:",
        "'X in hex/binary/octal' where X is a primary sub-query evaluated",
        "by number_theory / multi_step, then encoded via",
        "`NumericEncodeFacade` (new this round, `numeric_encode.py`).",
        "",
        "## Corpus",
        "",
        "12 probes: 6 NumberTheory → base, 5 MultiStep → base, 1 direct",
        "NumericEncode. All expected answers are short (1-8 char)",
        "hex/binary/octal strings.",
        "",
        "## Result",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| corpus size | {len(CORPUS)} |",
        f"| route correct | {n_route}/{len(CORPUS)} |",
        f"| answer correct | {n_answer}/{len(CORPUS)} |",
        f"| wall time | {elapsed:.1f}s |",
        "",
        "## Notes",
        "",
        "Chain detect strips 'in <base>' suffix from the prompt and",
        "re-classifies the remainder. Primary facade runs with",
        "use_bias=False (we only want its computed integer value);",
        "numeric_encode then runs with use_bias=True to deliver the",
        "encoded form through Gemma's decode.",
    ]
    recpath.write_text("\n".join(lines) + "\n")
    print(f"\n[r70b] receipt → {recpath}")

    outpath = ROOT / ".cache" / "r70b_planner_chain_results.jsonl"
    outpath.parent.mkdir(exist_ok=True)
    with outpath.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[r70b] per-probe → {outpath}")


main()
print("R70B_DONE")
