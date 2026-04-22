"""R22f — diagnostic on R22's flat N=10 cells.

R22 shipped +9/60 with gains concentrated in N=5 (+7) and N=15 (+2).
The four N=10 cells (2 distractor lengths × 2 seeds × 5 replicas = 20
prompts) are completely flat: 7/10 → 7/10 and 9/10 → 9/10 baseline→card.

Three candidate explanations:
  (a) card margin < 22.0 on N=10 wrong-baseline prompts → gate stays
      silent → card output never influences Gemma
  (b) card fires (margin >= 22) on N=10 wrong-baseline prompts, but
      the wrong Gemma answer has a logit gap > 50 boost — hook can't
      flip
  (c) adapter-regex has a residual N=10-specific bug (e.g. adapter
      picks the wrong key for N=10 only, so card gives right answer
      for wrong question → fires but wrong digit)

This script is pure analysis — reads the existing round6_gated_write.jsonl
(R22 TRUE result cache) and disambiguates (a) vs (b) vs (c). Zero
daemon cost.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "r22b"


def main() -> None:
    path = CACHE / "round6_gated_write.jsonl"
    cases = [json.loads(line) for line in path.open()]
    n10 = [c for c in cases if c["n_pairs"] == 10]
    print(f"[r22f] loaded {len(cases)} cases, {len(n10)} at N=10")

    # Classify each N=10 prompt
    from calm.hrm.data import _ID_TO_CHAR
    by_cat = defaultdict(list)
    for c in n10:
        base_ok = c["baseline_top_tok"] == c["expected"]
        card_ok = c["card_top_tok"] == c["expected"]
        fired = c["fired"]
        if base_ok and card_ok:
            cat = "base✓ card✓ (correct, card silent or preserved)"
        elif base_ok and not card_ok:
            cat = "base✓ card✗ REGRESSION"
        elif not base_ok and card_ok:
            cat = "base✗ card✓ WIN"
        else:  # not base_ok and not card_ok
            if fired:
                cat = "base✗ card✗ FIRED-BUT-WRONG"  # suspect (b) or (c)
            else:
                cat = "base✗ card✗ silent (gate blocked)"  # suspect (a)
        by_cat[cat].append(c)

    print("\n=== N=10 category breakdown ===")
    for cat, items in sorted(by_cat.items()):
        print(f"  {cat}: {len(items)}")

    # For the silent-failures (hypothesis a), dump margins
    silent_fail = by_cat.get("base✗ card✗ silent (gate blocked)", [])
    if silent_fail:
        print(f"\n=== HYPOTHESIS (a): margin < 22.0 on wrong-baseline prompts ===")
        print(f"  {'seed':>10}  {'dist':>5}  {'q':>2}  {'exp':>3}  "
              f"{'base':>5}  {'argmax':>8}  {'peak':>7}  {'med':>7}  "
              f"{'margin':>7}")
        for c in silent_fail:
            argmax_char = _ID_TO_CHAR.get(c["card_argmax"], "?")
            argmax_match = (argmax_char == c["expected"])
            mark = "★" if argmax_match else " "
            print(f"  {c['seed']:>10}  {c['distractor_tokens']:>5}  "
                  f"{c['query_key']:>2}  {c['expected']:>3}  "
                  f"{c['baseline_top_tok']!r:>5}  "
                  f"{argmax_char!r:>4} {mark}  "
                  f"{c['card_peak']:>7.2f}  {c['card_median']:>7.2f}  "
                  f"{c['card_margin']:>7.2f}")

        # If card argmax matches expected despite low margin, we could
        # LOWER the threshold for N=10 and pick up those fixes.
        would_fix_if_lowered = [
            c for c in silent_fail
            if _ID_TO_CHAR.get(c["card_argmax"], "?") == c["expected"]
        ]
        print(f"\n  ★ card argmax correct despite silent: "
              f"{len(would_fix_if_lowered)}/{len(silent_fail)}")
        if would_fix_if_lowered:
            margins = [c["card_margin"] for c in would_fix_if_lowered]
            print(f"    margin range: {min(margins):.2f}–{max(margins):.2f}")
            print(f"    → lowering threshold to {min(margins):.2f} "
                  f"would pick up +{len(would_fix_if_lowered)} fixes "
                  "(potentially; need to verify zero N=10 REGRESSIONS "
                  "among currently-correct cases at the new threshold)")

    # For the fired-but-wrong (hypothesis b or c), dump why
    fired_wrong = by_cat.get("base✗ card✗ FIRED-BUT-WRONG", [])
    if fired_wrong:
        print(f"\n=== HYPOTHESIS (b)/(c): card fired but didn't fix ===")
        for c in fired_wrong:
            argmax_char = _ID_TO_CHAR.get(c["card_argmax"], "?")
            argmax_match = (argmax_char == c["expected"])
            ptype = "(c) argmax wrong — adapter/card failure" if not argmax_match \
                else "(b) argmax right, boost lost — Gemma margin > 50"
            print(f"  seed={c['seed']} dist={c['distractor_tokens']} "
                  f"q={c['query_key']} exp={c['expected']} "
                  f"base={c['baseline_top_tok']!r} "
                  f"card_top={c['card_top_tok']!r} "
                  f"card_argmax={argmax_char!r} margin={c['card_margin']:.2f}")
            print(f"    → {ptype}")

    # Looking at the "correct-at-baseline" cases to estimate how many
    # would regress if threshold were lowered — does the card argmax
    # match baseline on those too?
    correct_at_base = by_cat.get(
        "base✓ card✓ (correct, card silent or preserved)", [])
    if correct_at_base:
        print(f"\n=== base✓ card✓ cases — margin distribution ===")
        fired_correct = [c for c in correct_at_base if c["fired"]]
        silent_correct = [c for c in correct_at_base if not c["fired"]]
        print(f"  fired (margin >= 22.0): {len(fired_correct)}")
        print(f"  silent (margin < 22.0): {len(silent_correct)}")
        if silent_correct:
            # Check if card argmax would disagree with baseline at
            # lower thresholds → would cause regression
            bad_args = 0
            for c in silent_correct:
                argmax_char = _ID_TO_CHAR.get(c["card_argmax"], "?")
                if argmax_char != c["expected"]:
                    bad_args += 1
            margins = sorted(c["card_margin"] for c in silent_correct)
            print(f"    margin range: {margins[0]:.2f}–{margins[-1]:.2f}")
            print(f"    card_argmax wrong on {bad_args}/{len(silent_correct)} "
                  "of these (lowering threshold would regress these cases)")

    # Summary verdict
    print("\n=== VERDICT ===")
    any_silent = len(silent_fail)
    any_fired_wrong = len(fired_wrong)
    if any_silent and not any_fired_wrong:
        print("  Hypothesis (a) — margin gate blocks card on N=10 "
              "wrong-baseline prompts. N=10 card margins are below 22.0 "
              "on confusing-distractor prompts.")
    elif any_fired_wrong and not any_silent:
        print("  Hypothesis (b) or (c) — card fires but Gemma's wrong "
              "answer outweighs the boost, OR adapter mis-extracts "
              "the key for N=10.")
    elif any_silent and any_fired_wrong:
        print("  Mixed — both gate-silent and fire-but-wrong present.")
    else:
        print("  No N=10 failures detected — everything correct at baseline.")


main()
