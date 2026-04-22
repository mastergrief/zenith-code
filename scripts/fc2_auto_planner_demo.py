"""FC2 — Auto-refactor planner end-to-end demo.

Takes messy code + test harness, runs the substrate's planner WITHOUT
human direction: detect_refactor_opportunities → build_plan →
execute_plan (sandbox-verified). Reports what got applied.

Scenario: an `Inventory.summary()` function written with several
refactor anti-patterns stacked together:
  - loop-with-append that could be a comprehension
  - single-use-local variables (candidates for inline)
  - mutable-default arg (static ast_repair opportunity)

Goal: demonstrate the autonomous pattern — user provides code + tests,
substrate figures out the refactor plan on its own.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calm.llm_computer.ast_refactor import detect_refactor_opportunities
from calm.llm_computer.refactor_planner import build_plan, execute_plan


MESSY = """
class Inventory:
    def summary(self, items):
        names = []
        for item in items:
            names.append(item['name'])

        total_value = 0
        for item in items:
            total_value += item['qty'] * item['price']

        over_threshold = []
        for item in items:
            if item['qty'] > 10:
                over_threshold.append(item['name'])

        return {
            'names': names,
            'total_value': total_value,
            'over_threshold': over_threshold,
            'count': len(items),
        }
""".strip()


TESTS = """

def _run_tests():
    inv = Inventory()
    items = [
        {'name': 'apple', 'qty': 5, 'price': 1.0},
        {'name': 'pen',   'qty': 20, 'price': 2.5},
        {'name': 'book',  'qty': 3, 'price': 15.0},
    ]
    r = inv.summary(items)
    assert r['names'] == ['apple', 'pen', 'book']
    assert r['total_value'] == 5*1.0 + 20*2.5 + 3*15.0
    assert r['over_threshold'] == ['pen']
    assert r['count'] == 3

    # Empty case
    r2 = inv.summary([])
    assert r2['names'] == []
    assert r2['total_value'] == 0
    assert r2['over_threshold'] == []
    assert r2['count'] == 0

    print('ALL_PASS')

_run_tests()
"""


def main():
    print("=" * 60)
    print("FC2 — Auto-refactor planner demo")
    print("=" * 60)
    print("\nInput:")
    print("-" * 60)
    print(MESSY)

    print("\n" + "-" * 60)
    print("Detected opportunities:")
    opps = detect_refactor_opportunities(MESSY)
    for o in opps:
        print(f"  [{o.severity:4}] {o.kind:22}  {o.location:30}  {o.detail}")

    print("\n" + "-" * 60)
    print("Built plan:")
    plan = build_plan(MESSY)
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step.primitive_name:32} kwargs={step.kwargs}")

    print("\n" + "-" * 60)
    print("Executing plan with sandbox-gated verification...")
    out = execute_plan(MESSY, TESTS)

    print("\n" + "-" * 60)
    print(f"Result: {out.summary}")
    print(f"Tests pass on final code: {out.tests_pass}")

    applied = [s for s in out.applied_steps if s.refactor_result.applied
               and s.test_passed]
    rolled_back = [s for s in out.applied_steps if s.refactor_result.applied
                   and s.test_passed is False]
    refused = [s for s in out.applied_steps if not s.refactor_result.applied]

    print(f"\nStep outcomes:")
    print(f"  {len(applied)} applied + verified")
    print(f"  {len(rolled_back)} rolled back (test regression)")
    print(f"  {len(refused)} refused by primitive (error or no-op)")
    for s in out.applied_steps:
        mark = ("✓" if s.refactor_result.applied and s.test_passed
                else ("↩" if s.test_passed is False else "—"))
        print(f"    {mark} {s.primitive_name}({s.kwargs})")
        if s.refactor_result.error:
            print(f"       err: {s.refactor_result.error}")

    print("\n" + "-" * 60)
    print("Final code:")
    print(out.final_code)

    # Receipt
    recpath = ROOT / ".claude" / "MEMORY" / "evals" / \
              "2026-04-22_fc2_auto_planner_demo.md"
    lines = [
        "# FC2 — Auto-refactor planner demo",
        "",
        "Substrate plans + executes a refactor session from a user's",
        "code + tests, zero human direction beyond the baseline harness.",
        "",
        "## Input",
        "",
        "```python", MESSY, "```", "",
        "## Opportunities detected",
        "",
        "| severity | kind | location | detail |",
        "|---|---|---|---|",
    ]
    for o in opps:
        lines.append(f"| {o.severity} | {o.kind} | {o.location} | {o.detail} |")
    lines.extend([
        "",
        "## Plan built",
        "",
        "| # | primitive | kwargs |",
        "|---|---|---|",
    ])
    for i, step in enumerate(plan, 1):
        lines.append(f"| {i} | {step.primitive_name} | {step.kwargs!r} |")
    lines.extend([
        "",
        "## Execution outcome",
        "",
        f"**{out.summary}**",
        "",
        f"- Final tests pass: {out.tests_pass}",
        f"- Applied + verified: {len(applied)}",
        f"- Rolled back: {len(rolled_back)}",
        f"- Refused (no-op/error): {len(refused)}",
        "",
        "## Final code",
        "",
        "```python", out.final_code, "```", "",
    ])
    recpath.write_text("\n".join(lines) + "\n")
    print(f"\nreceipt → {recpath}")
    return 0 if out.tests_pass else 1


if __name__ == "__main__":
    sys.exit(main())
