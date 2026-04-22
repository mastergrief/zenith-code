"""R70d — PlannerFacade N-step chain dispatch A/B.

Tests 3-step (and 2-step variants) chains composed via 'then' /
', multiply by N' / ', in hex' connectives. Each test probe:
(prompt, expected_value_or_encoded).

Pattern follows r70b (existing 2-step chain test). Compares Planner
with N-step dispatch against baseline Gemma (pass-through, no facade).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
assert "m" in globals() and "tok" in globals(), (  # type: ignore[name-defined]
    "run via bin/gemma-run scripts/r70d_planner_nstep_chain.py"
)

sys.path.insert(0, str(ROOT))

import importlib
import calm.llm_computer.facades.planner as planner_mod
importlib.reload(planner_mod)
from calm.llm_computer.facades.planner import PlannerFacade


def clear_card_state():
    for lyr in m.layers:  # type: ignore[name-defined]
        if hasattr(lyr, "card_slots"):
            lyr.card_slots = []
    m.verification_hooks = []  # type: ignore[name-defined]
    m.reserved_channels = []  # type: ignore[name-defined]


clear_card_state()
print("[r70d] cleared card state")


# Each probe: (prompt, expected_str, kind)
# kind 'int' -> expect integer; kind 'hex'/'bin'/'oct' -> expect base repr
PROBES = [
    # 2-step chain: primary + arith
    ("What is the GCD of 48 and 180, then multiply by 3?", "36", "int"),
    ("What is factorial of 5, then plus 10?", "130", "int"),
    ("2 to the power 8, then times 2", "512", "int"),

    # 3-step chain: primary + arith + encode
    ("What is the GCD of 48 and 180, then multiply by 3, in hex?", "24", "hex"),
    ("factorial of 5, then plus 10, in binary", "10000010", "bin"),
    ("2 to the power 8, then times 2, in hex", "200", "hex"),

    # 3-step chain: primary + arith + arith
    ("What is factorial of 5, then plus 10, then times 2", "260", "int"),
    ("GCD of 48 and 180, then plus 4, then times 2", "32", "int"),

    # 4-step chain: primary + 2 arith + encode
    ("factorial of 5, then plus 10, then times 2, in hex", "104", "hex"),
]


def score(generated: str, expected: str, kind: str) -> bool:
    out = generated.lower().replace(",", "")
    exp = expected.lower()
    if kind == "int":
        m_ = re.search(r"-?\d{1,15}", out)
        if not m_:
            return False
        return int(m_.group(0)) == int(expected)
    # base-encoded: check substring match (hex/bin/oct digits)
    # Look for the expected token allowing for 0x/0b prefix or plain.
    if kind == "hex":
        return bool(re.search(rf"\b(?:0x)?{re.escape(exp)}\b", out))
    if kind == "bin":
        return bool(re.search(rf"\b(?:0b)?{re.escape(exp)}\b", out))
    if kind == "oct":
        return bool(re.search(rf"\b(?:0o)?{re.escape(exp)}\b", out))
    return False


def main():
    planner = PlannerFacade(device="cuda", register_auto=True)
    tags = [t for t, _ in planner.auto_facades]
    print(f"[r70d] auto-facades: {tags}")

    # Route classify: show which chain detection path fires
    print("\n=== Phase 1: chain segment parse ===")
    for prompt, expected, kind in PROBES:
        segs = planner._split_chain_steps(prompt)
        ops = [planner._parse_chain_op(s) for s in segs[1:]]
        print(f"  segs={len(segs)} {prompt!r} → {segs!r}, ops={ops}")

    planner.install(m, tok)  # type: ignore[name-defined]

    print("\n=== Phase 2: chain answer with bias ===")
    t0 = time.time()
    hits = 0
    recs = []
    for prompt, expected, kind in PROBES:
        r = planner.solve(prompt, use_bias=True)
        ok = score(r.generated, expected, kind)
        if ok:
            hits += 1
        mark = "✓" if ok else "✗"
        steps_str = " → ".join(str(s) for s in (r.chain_steps or []))
        print(f"  {mark} exp={expected:<10} got={r.generated[:50]!r}")
        print(f"      steps: {steps_str}")
        recs.append({
            "prompt": prompt, "expected": expected, "kind": kind,
            "facade": r.facade, "generated": r.generated[:120],
            "used_bias": r.used_bias, "chain_steps": [
                [str(s[0]), s[1]] for s in (r.chain_steps or [])
            ],
            "ok": ok,
        })
    elapsed = time.time() - t0
    print(f"\n  answer: {hits}/{len(PROBES)}  elapsed {elapsed:.1f}s")

    print("\n=== Phase 3: baseline Gemma (no facade) ===")
    t0 = time.time()
    base_hits = 0
    for rec in recs:
        r = planner.solve(rec["prompt"], use_bias=False)
        ok = score(r.generated, rec["expected"], rec["kind"])
        rec["baseline_ok"] = ok
        rec["baseline_out"] = r.generated[:120]
        if ok:
            base_hits += 1
        mark = "✓" if ok else "✗"
        print(f"  {mark} exp={rec['expected']:<10} got={r.generated[:50]!r}")
    elapsed_b = time.time() - t0
    print(f"\n  baseline: {base_hits}/{len(PROBES)}  elapsed {elapsed_b:.1f}s")

    print("\n========== SUMMARY ==========")
    print(f"  chain probes: {len(PROBES)}")
    print(f"  bias hits:    {hits}")
    print(f"  baseline:     {base_hits}")
    print(f"  Δ:            {hits - base_hits:+d}")

    recpath = (ROOT / ".claude" / "MEMORY" / "evals"
               / "2026-04-22_r70d_planner_nstep_chain.md")
    lines = [
        "# R70d — PlannerFacade N-step chain dispatch",
        "",
        "3+ step chain composition via 'then'/',' connectives:",
        "primary facade → arith-op N → (arith-op N)* → optional numeric-encode.",
        "",
        "## Corpus",
        "",
        f"{len(PROBES)} chain probes: 2-step, 3-step, 4-step variants across",
        "NumberTheory, auto-facades (factorial, power), MultiStep intermediate",
        "arithmetic, NumericEncode terminal.",
        "",
        "## Results",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| chain probes | {len(PROBES)} |",
        f"| bias hits | {hits}/{len(PROBES)} |",
        f"| baseline | {base_hits}/{len(PROBES)} |",
        f"| Δ (bias − baseline) | {hits - base_hits:+d} |",
        f"| wall Phase 2 | {elapsed:.1f}s |",
        f"| wall Phase 3 | {elapsed_b:.1f}s |",
        "",
    ]
    recpath.write_text("\n".join(lines) + "\n")
    print(f"\n[r70d] receipt → {recpath}")

    outjsonl = ROOT / ".cache" / "r70d_planner_nstep_chain.jsonl"
    outjsonl.parent.mkdir(exist_ok=True)
    with outjsonl.open("w") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")


main()
print("R70D_DONE")
