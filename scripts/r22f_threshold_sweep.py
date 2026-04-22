"""R22f — threshold sweep to fix N=10/N=15 gate-silence.

Finding from r22f_live_parse_trace: card is 100% correct standalone
at N=10 (20/20) and N=15 (20/20), but margins cluster below the 22.0
gate threshold used in R22 TRUE result:
  N=10 p50=20.83 p5=15.21
  N=15 p50=18.63 p5=16.39

R22 shipped +9/60 with 41/60 gate-silent. The gate was over-calibrated
on N=5 margins (where margin is naturally higher with shorter key-space).

Hypothesis: lower min_margin to 14.5 (below observed p5 across all N)
→ fire on ~all 60 prompts → upper bound ~60/60 card accuracy because
standalone card is 100% on N=5/10/15.

Risk: card wrong-argmax at low margin could cause regressions on
currently-correct-baseline cases. Standalone data says zero wrong in
40/40 tested, but we rerun the FULL A/B to be sure.

Test: run the R22 full install on the pooled 60-prompt corpus with
three thresholds:
  - 22.0 (R22 TRUE baseline)
  - 18.0 (cuts some but keeps most confident)
  - 14.5 (fires on essentially everything)

Write per-threshold lift + regressions. Pick the threshold with max
lift and zero regressions.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "r22b"
CACHE.mkdir(parents=True, exist_ok=True)

assert "m" in globals() and "tok" in globals(), (  # type: ignore[name-defined]
    "run via bin/gemma-run scripts/r22f_threshold_sweep.py"
)

sys.path.insert(0, str(ROOT))
from calm.llm_computer.gemma_substrate import KVCache  # noqa
from calm.hrm.data import _CHAR_TO_ID  # noqa
from calm.llm_computer.facades.retrieval import _monkey_patch_fast_encode  # noqa
_monkey_patch_fast_encode(tok)  # type: ignore[name-defined]

# Reuse r22b_round2 corpus generator + install closures
_r2_src = (ROOT / "scripts" / "r22b_round2.py").read_text()
_r2_src = _r2_src.split("main()\nprint(\"R22B_R2_DONE\")")[0]
_ns = {"__name__": "_r22b_r2", "__file__": str(ROOT / "scripts" / "r22b_round2.py"),
       "m": m, "tok": tok}  # type: ignore[name-defined]
exec(compile(_r2_src, str(ROOT / "scripts" / "r22b_round2.py"), "exec"), _ns)
make_prompt = _ns["make_prompt"]
build_gemma_digit_ids = _ns["build_gemma_digit_ids"]
load_mqar_card = _ns["load_mqar_card"]
install = _ns["install"]
parse_mqar_prompt = _ns["parse_mqar_prompt"]
mqar_to_ids = _ns["mqar_to_ids"]


def clear_card_state():
    for lyr in m.layers:  # type: ignore[name-defined]
        if hasattr(lyr, "card_slots"):
            lyr.card_slots = []
    m.verification_hooks = []  # type: ignore[name-defined]
    m.reserved_channels = []  # type: ignore[name-defined]


def score(ids):
    cache = KVCache(m.config.n_layers, device="cuda")  # type: ignore[name-defined]
    with torch.no_grad():
        logits = m.forward(  # type: ignore[name-defined]
            torch.tensor([ids]), device="cuda",
            kv_cache=cache, start_pos=0,
        )
    top = int(logits[0, -1].argmax())
    return top, tok.id_to_token.get(top, "?")  # type: ignore[name-defined]


def build_corpus():
    seeds = [2026_04_22, 2026_04_23]
    cells = [
        (5,  500, "confusing"),
        (5,  1500, "confusing_long"),
        (10, 500, "confusing"),
        (10, 1500, "confusing_long"),
        (15, 500, "confusing"),
        (15, 1500, "confusing_long"),
    ]
    REPLICAS = 5
    candidates = []
    for seed in seeds:
        rng = random.Random(seed)
        for (n_pairs, dist_tok, mode) in cells:
            for r in range(REPLICAS):
                prompt, q_key, expected = make_prompt(
                    n_pairs, dist_tok, mode, rng)
                candidates.append({
                    "seed": seed, "n_pairs": n_pairs,
                    "distractor_tokens": dist_tok, "mode": mode,
                    "replica": r, "prompt": prompt,
                    "query_key": q_key, "expected": expected,
                    "ids": tok.encode(prompt),  # type: ignore[name-defined]
                })
    return candidates


def run_baseline(candidates):
    print("[r22f-sweep] BASELINE pass...")
    clear_card_state()
    t0 = time.time()
    for c in candidates:
        top, top_tok = score(c["ids"])
        c["baseline_top"] = top
        c["baseline_top_tok"] = top_tok
    print(f"  done in {time.time() - t0:.1f}s")


def run_with_threshold(candidates, threshold, label):
    print(f"\n[r22f-sweep] THRESHOLD={threshold} ({label}) ...")
    card = load_mqar_card(
        ROOT / "calm/hrm/checkpoints/copy_augmented_delta_mqar_best.pt")
    slot, state, hook = install(  # type: ignore[name-defined]
        m, card, layer_idx=30, ch_off=2480,
        write_margin=threshold, preserve=False,
    )
    hook.min_margin = threshold

    CARD_N_RANGE = {5, 10, 15}
    t0 = time.time()
    for c in candidates:
        if c["n_pairs"] not in CARD_N_RANGE:
            state["active"] = False
        else:
            mqar_str = parse_mqar_prompt(c["prompt"])
            if mqar_str is None:
                state["active"] = False
            else:
                state["mqar_ids"] = mqar_to_ids(mqar_str)
                state["active"] = True
        top, top_tok = score(c["ids"])
        c[f"card_top_t{threshold}"] = top
        c[f"card_top_tok_t{threshold}"] = top_tok
        co = slot.last_output
        last = co[0, -1].float()
        peak = last.max().item()
        med = last.median().item()
        c[f"card_margin_t{threshold}"] = peak - med
        c[f"card_argmax_t{threshold}"] = int(last.argmax().item())
        c[f"fired_t{threshold}"] = state["active"] and (peak - med) >= threshold
    print(f"  done in {time.time() - t0:.1f}s")


def report(candidates, threshold, digit_ids):
    base_correct = sum(
        1 for c in candidates
        if c["baseline_top"] == digit_ids[int(c["expected"])]
    )
    card_correct = sum(
        1 for c in candidates
        if c[f"card_top_t{threshold}"] == digit_ids[int(c["expected"])]
    )
    fired = sum(1 for c in candidates if c[f"fired_t{threshold}"])

    wins = [c for c in candidates
            if c["baseline_top"] != digit_ids[int(c["expected"])]
            and c[f"card_top_t{threshold}"] == digit_ids[int(c["expected"])]]
    regr = [c for c in candidates
            if c["baseline_top"] == digit_ids[int(c["expected"])]
            and c[f"card_top_t{threshold}"] != digit_ids[int(c["expected"])]]

    print(f"\n=== THRESHOLD = {threshold} ===")
    print(f"  baseline:   {base_correct}/{len(candidates)}")
    print(f"  with card:  {card_correct}/{len(candidates)}  "
          f"(Δ={card_correct - base_correct:+d})")
    print(f"  fired:      {fired}/{len(candidates)}")
    print(f"  WINS:       {len(wins)}")
    print(f"  REGR:       {len(regr)}")
    if regr:
        for c in regr:
            print(f"    seed={c['seed']} N={c['n_pairs']} dist={c['distractor_tokens']} "
                  f"q={c['query_key']} exp={c['expected']} "
                  f"base={c['baseline_top_tok']!r} "
                  f"card={c[f'card_top_tok_t{threshold}']!r} "
                  f"margin={c[f'card_margin_t{threshold}']:.2f} "
                  f"fired={c[f'fired_t{threshold}']}")

    # Per-cell
    print(f"\n  per-cell: {'N':>3} {'dist':>5} {'mode':>18}  base   card   Δ")
    cell_stats = defaultdict(lambda: {"base": 0, "card": 0, "total": 0})
    for c in candidates:
        k = (c["n_pairs"], c["distractor_tokens"], c["mode"])
        cell_stats[k]["total"] += 1
        if c["baseline_top"] == digit_ids[int(c["expected"])]:
            cell_stats[k]["base"] += 1
        if c[f"card_top_t{threshold}"] == digit_ids[int(c["expected"])]:
            cell_stats[k]["card"] += 1
    for k in sorted(cell_stats.keys()):
        s = cell_stats[k]
        delta = s["card"] - s["base"]
        sign = "+" if delta > 0 else ""
        print(f"    {k[0]:>3}  {k[1]:>5}  {k[2]:>18}  "
              f"{s['base']:>2}/{s['total']:<2}  {s['card']:>2}/{s['total']:<2}  "
              f"{sign}{delta}")

    return {
        "threshold": threshold,
        "baseline": base_correct,
        "card": card_correct,
        "delta": card_correct - base_correct,
        "wins": len(wins),
        "regr": len(regr),
        "fired": fired,
    }


def main():
    candidates = build_corpus()
    digit_ids = build_gemma_digit_ids()
    run_baseline(candidates)

    # Sweep thresholds from high (R22 shipped) to low (fire everywhere)
    thresholds = [22.0, 18.0, 14.5]
    for t in thresholds:
        run_with_threshold(candidates, t, {22.0: "R22 shipped",
                                            18.0: "mid",
                                            14.5: "below p5"}[t])

    print("\n\n========== SUMMARY ==========")
    summary = []
    for t in thresholds:
        r = report(candidates, t, digit_ids)
        summary.append(r)

    print("\n\n=== HEAD-TO-HEAD ===")
    print(f"  threshold  baseline  card   Δ    W   R    fired")
    for r in summary:
        print(f"  {r['threshold']:>8.1f}   {r['baseline']:>8}   "
              f"{r['card']:>4}  {r['delta']:>+3d}  {r['wins']:>2}  "
              f"{r['regr']:>2}   {r['fired']:>5}")

    # Save
    out = CACHE / "r22f_threshold_sweep.jsonl"
    with out.open("w") as f:
        for c in candidates:
            rec = {k: v for k, v in c.items() if k not in ("ids", "prompt")}
            f.write(json.dumps(rec) + "\n")
    print(f"\n[r22f-sweep] full data → {out}")

    # Receipt summary
    recpath = (ROOT / ".claude" / "MEMORY" / "evals"
               / "2026-04-22_r22f_threshold_sweep.md")
    lines = [
        "# R22f — threshold sweep",
        "",
        "Hypothesis: R22's `min_margin=22.0` over-gates N=10/N=15 (card",
        "margins cluster at p50=20.83 / p50=18.63 for those Ns). Lowering",
        "threshold should pick up silenced-but-correct cases. Card is 100%",
        "standalone on N=5/10/15 per r22f_live_parse_trace.",
        "",
        "## A/B results (60-prompt pooled R22 corpus)",
        "",
        "| threshold | baseline | card | Δ | wins | regressions | fired |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['threshold']:.1f} | {r['baseline']}/60 | {r['card']}/60 | "
            f"{r['delta']:+d} | {r['wins']} | {r['regr']} | {r['fired']}/60 |")
    lines += [
        "",
        "## Verdict",
        "",
        "TBD — pick threshold with max Δ and zero regressions.",
    ]
    recpath.write_text("\n".join(lines) + "\n")
    print(f"[r22f-sweep] receipt → {recpath}")


main()
print("R22F_SWEEP_DONE")
