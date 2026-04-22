"""M2a — MetaFacade Level-2 demo: substrate synthesizes FacadeSpecs.

Level-1 (shipped in r80a) required a hand-written `FacadeSpec` — a
human picked the regex catalog, eval_expr, and guard constants. Level-2
(`MetaFacade.from_oracle`) emits the spec itself given JUST
(oracle_fn_name, arity).

Demo strategy: pick 5 safe_eval oracle functions, synthesize specs via
MetaFacade, validate + generate + install via the existing Level-1
pipeline, run A/B.

The same Level-1 CALM gates still apply — oracle validation + ast.parse
+ live A/B — so Level-2 inherits Level-1's drift-free safety property.

Compare this to Level-1 (r80a/m1a): the ONLY difference is where the
FacadeSpec comes from. In Level-1 a human authored FACTORIAL_SPEC /
FIBONACCI_SPEC etc. In Level-2 the spec is materialized from a minimal
(name, arity) descriptor.
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
    "run via bin/gemma-run scripts/m2a_metafacade_demo.py"
)

sys.path.insert(0, str(ROOT))
from calm.llm_computer.facades.retrieval import _monkey_patch_fast_encode
from calm.llm_computer.recursion import (
    MetaFacade, validate_facade, generate_facade, import_facade_class,
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


# --- Oracle descriptors (what MetaFacade gets as input) ---
ORACLE_DESCRIPTORS = [
    dict(
        fn_name="factorial",
        arity=1,
        domain_name="FactorialMeta",
        module_name="factorial_meta",
        max_operand=15,
    ),
    dict(
        fn_name="combinations",
        arity=2,
        domain_name="CombinationsMeta",
        module_name="combinations_meta",
        max_operand=100,
        # Extra pattern for "choose" idiom, which isn't in the
        # canonical list (MetaFacade generates function-name-based
        # patterns only; domain-specific idioms go in extra_patterns)
        extra_patterns=[r"(-?\d+)\s+choose\s+(-?\d+)"],
    ),
    dict(
        fn_name="gcd",
        arity=2,
        domain_name="GcdMeta",
        module_name="gcd_meta",
        max_operand=100000,
    ),
    dict(
        fn_name="lcm",
        arity=2,
        domain_name="LcmMeta",
        module_name="lcm_meta",
        max_operand=100000,
    ),
    dict(
        fn_name="fibonacci",
        arity=1,
        domain_name="FibonacciMeta",
        module_name="fibonacci_meta",
        max_operand=50,
    ),
]

# --- Oracle test sets ---
ORACLES = {
    "FactorialMeta":    [((5,), 120), ((7,), 5040), ((10,), 3628800), ((12,), 479001600)],
    "CombinationsMeta": [((10, 3), 120), ((52, 5), 2598960), ((5, 2), 10), ((7, 4), 35)],
    "GcdMeta":          [((48, 180), 12), ((391, 238), 17), ((12, 18), 6), ((1001, 143), 143)],
    "LcmMeta":          [((12, 18), 36), ((48, 180), 720), ((15, 20), 60)],
    "FibonacciMeta":    [((10,), 55), ((15,), 610), ((20,), 6765), ((25,), 75025)],
}

# --- A/B probes using the Meta-synthesized patterns ---
PROBES = {
    "FactorialMeta": [
        ("What is factorial 7", 5040),
        ("factorial(10)", 3628800),
        ("factorial of 12", 479001600),
    ],
    "CombinationsMeta": [
        ("combinations(10, 3)", 120),
        ("combinations of 7 and 4", 35),
        ("20 choose 4", 4845),        # tests extra_pattern
    ],
    "GcdMeta": [
        ("gcd(48, 180)", 12),
        ("gcd of 391 and 238", 17),
        ("gcd of 1001 and 143", 143),
    ],
    "LcmMeta": [
        ("lcm(12, 18)", 36),
        ("lcm of 48 and 180", 720),
        ("lcm of 15 and 20", 60),
    ],
    "FibonacciMeta": [
        ("fibonacci(20)", 6765),
        ("fibonacci of 25", 75025),
        ("what is fibonacci 15", 610),
    ],
}


def main():
    print("========== M2a — LEVEL-2 METAFACADE DEMO ==========")
    print("Substrate synthesizes FacadeSpecs from just (fn_name, arity).\n")

    # Step 1: MetaFacade synthesizes all specs in one batch
    specs = MetaFacade.batch_from_oracles(ORACLE_DESCRIPTORS)
    print(f"MetaFacade.batch_from_oracles({len(ORACLE_DESCRIPTORS)} descriptors) -> "
          f"{len(specs)} FacadeSpecs\n")

    results = []
    for spec in specs:
        label = spec.name
        print(f"--- {label} ---")
        print(f"  synthesized {len(spec.parse_patterns)} regex pattern(s) + "
              f"eval_expr={spec.eval_expr!r}")

        # Step 2: CALM oracle gate
        passed, total = validate_facade(spec, ORACLES[label])
        print(f"  oracle validation: {passed}/{total}")
        if passed < total:
            print(f"  ✗ ORACLE REJECTED — skipping")
            continue

        # Step 3: generate .py file (ast.parse-gated inside)
        path = generate_facade(spec, overwrite=True)
        try:
            rel = path.relative_to(ROOT)
        except ValueError:
            rel = path
        print(f"  generated: {rel}")

        # Step 4: import + install + A/B
        Cls = import_facade_class(spec)
        facade = Cls(device="cuda")
        facade.install(m, tok)  # type: ignore[name-defined]

        probes = PROBES[label]
        n_base_ok = 0
        n_card_ok = 0
        t0 = time.time()
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
            print(f"    {prompt!r:<42} exp={expected:<10} "
                  f"base={str(r0.parsed_answer):<10}{bmark}  "
                  f"card={str(r1.parsed_answer):<10}{cmark}")
        elapsed = time.time() - t0
        facade.detach()
        print(f"  baseline: {n_base_ok}/{len(probes)}  "
              f"card: {n_card_ok}/{len(probes)}  "
              f"Δ={n_card_ok - n_base_ok:+d}  elapsed {elapsed:.1f}s\n")
        results.append({
            "label": label, "spec_name": spec.name,
            "oracle": f"{passed}/{total}",
            "baseline": n_base_ok, "card": n_card_ok,
            "total": len(probes), "elapsed": elapsed,
            "path": str(rel),
        })

    # Summary
    print("========== SUMMARY ==========")
    total_base = sum(r["baseline"] for r in results)
    total_card = sum(r["card"] for r in results)
    total_probes = sum(r["total"] for r in results)
    for r in results:
        print(f"  {r['label']:<20} oracle={r['oracle']:<5} "
              f"base={r['baseline']}/{r['total']:<2} "
              f"card={r['card']}/{r['total']:<2} "
              f"Δ={r['card']-r['baseline']:+d}")
    print(f"\n  TOTAL: base={total_base}/{total_probes}  "
          f"card={total_card}/{total_probes}  "
          f"Δ={total_card - total_base:+d}")
    print(f"\n  SUBSTRATE SYNTHESIZED {len(results)} FacadeSpec(s) + "
          f"generated {len(results)} .py file(s) from ONLY "
          f"(fn_name, arity) descriptors.")

    # Receipt
    recpath = (ROOT / ".claude" / "MEMORY" / "evals"
               / "2026-04-22_m2a_level2_metafacade.md")
    lines = [
        "# M2a — Level-2 MetaFacade demo",
        "",
        "Per `.claude/rules/recursion.md` §'Level 2'. MetaFacade",
        "synthesizes the FacadeSpec itself from just",
        "(oracle_fn_name, arity). All three Level-1 CALM gates still",
        "apply (oracle validation → ast.parse → live A/B); only the",
        "SPEC authorship moved from human to substrate.",
        "",
        "## Descriptors → specs",
        "",
    ]
    for d in ORACLE_DESCRIPTORS:
        lines.append(f"- `{d['fn_name']}` arity={d['arity']} → "
                     f"`{d.get('domain_name', d['fn_name'].capitalize())}` "
                     f"(module `{d.get('module_name', d['fn_name']+'_auto')}`)")
    lines += [
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
        f"card {total_card}/{total_probes} "
        f"(Δ={total_card - total_base:+d})",
        "",
        "## What MetaFacade replaced",
        "",
        "A hand-written FacadeSpec requires the author to think about:",
        "- Name conventions (PascalCase class, snake_case module)",
        "- Canonical NL patterns (fn(args) / fn of args / a fn b)",
        "- Integer capture groups with negative-number support",
        "- safe_eval template formatting ({a}, {b})",
        "- Arity-specific regex shapes",
        "- max_tokens + max_operand guards",
        "",
        "MetaFacade.from_oracle encodes all of these as a template",
        "function over (fn_name, arity). The user supplies ONLY:",
        "- safe_eval function name (must exist)",
        "- Arity (1 or 2)",
        "- Optional guard / extra-patterns overrides",
        "",
        "## Generated files",
        "",
    ]
    for r in results:
        lines.append(f"- `{r['path']}` ({r['label']})")
    recpath.write_text("\n".join(lines) + "\n")
    print(f"\n[m2a] receipt → {recpath}")


main()
print("M2A_DONE")
