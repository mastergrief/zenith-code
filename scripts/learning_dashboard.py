"""Show the current state of the CALM feedback loops.

The Zenith harness has two independent learning loops:
  1. AutoLearner (`calm/auto_learn.py`) — learns pattern → precompute
     mappings from verifier corrections. Persists to
     `calm/learned_patterns.jsonl`.
  2. ModuleLearner (`calm/module_learning.py`) — learns recurring
     cognitive-module issues → system-prompt preventions. Persists to
     `calm/.module_learning.json`.

This CLI prints both in one place so an operator can tell at a glance
what the system has learned from usage.

Usage:
  PYTHONPATH=. python3 scripts/learning_dashboard.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from calm.auto_learn import AutoLearner
from calm.module_learning import ModuleLearner


def print_auto_learner(db_path: Path) -> None:
    print(f"\n# AutoLearner  ({db_path})")
    if not db_path.exists():
        print("  (no DB — no corrections logged yet)")
        return
    learner = AutoLearner(db_path=db_path)
    stats = learner.stats()
    if stats.get("total", 0) == 0:
        print("  (DB exists but empty)")
        return

    print(f"  total patterns: {stats['total']}")
    print(f"  total hits (precomputes fired): {stats.get('total_hits', 0)}")
    print(f"  cold patterns (never fired): {stats.get('cold_patterns', 0)}")
    by_type = stats.get("by_type", {})
    for t, n in sorted(by_type.items()):
        print(f"    {t}: {n}")
    print("\n  top patterns (by hits, then frequency):")
    print(f"    {'expression':<40} {'freq':>5} {'hits':>5}")
    for expr, freq, hits in stats["top_patterns"]:
        short = expr if len(expr) < 38 else expr[:35] + "..."
        print(f"    {short:<40} {freq:>5} {hits:>5}")


def print_module_learner(db_path: Path) -> None:
    print(f"\n# ModuleLearner  ({db_path})")
    if not db_path.exists():
        print("  (no DB — no module issues logged yet)")
        return
    ml = ModuleLearner(db_path=db_path)
    if not ml._trends:
        print("  (DB exists but empty)")
        return

    total = len(ml._trends)
    recurring = ml.recurring_issues
    print(f"  total tracked issues: {total}")
    print(f"  recurring (frequency >= 3, inject into prompts): {len(recurring)}")
    by_module = {}
    by_context = {}
    for t in ml._trends.values():
        by_module[t.module] = by_module.get(t.module, 0) + 1
        by_context[t.context] = by_context.get(t.context, 0) + 1
    print("\n  by module:")
    for m, n in sorted(by_module.items(), key=lambda x: -x[1]):
        print(f"    {m:<24} {n}")
    print("\n  by context:")
    for c, n in sorted(by_context.items(), key=lambda x: -x[1]):
        print(f"    {c:<24} {n}")

    if recurring:
        print("\n  recurring issues (will inject prevention on matching prompts):")
        print(f"    {'module':<22} {'context':<14} {'freq':>5}")
        for t in sorted(recurring, key=lambda t: -t.frequency)[:15]:
            issue_short = t.issue_type if len(t.issue_type) < 20 else t.issue_type[:17] + "..."
            print(f"    {t.module + ':' + issue_short[:12]:<22} {t.context:<14} {t.frequency:>5}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--auto-db", type=Path,
                   default=Path("calm/learned_patterns.jsonl"))
    p.add_argument("--module-db", type=Path,
                   default=Path("calm/.module_learning.json"))
    args = p.parse_args()

    print("=" * 60)
    print(" Zenith Learning Loops — Current State")
    print("=" * 60)
    print_auto_learner(args.auto_db)
    print_module_learner(args.module_db)
    print()


if __name__ == "__main__":
    main()
