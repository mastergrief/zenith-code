"""R22f live-parse trace — reproduce exact R22 corpus (real Gemma
tokenizer for distractor lengths) and trace parse_mqar_prompt on every
prompt. Cross-reference with round6_gated_write.jsonl to find the
discrepancy: offline parse says 60/60 OK, live run shows 41/60 silent.

Hypothesis: seed divergence (my offline regeneration uses char/3 proxy
for _approx_tokens, real run uses tok.encode length; the distractor
prose count diverges; rng state diverges across seed) — but the MEM
block and query key ARE drawn BEFORE distractor generation, so parse
should still succeed.

If this script shows 60/60 parse OK, then the culprit is not
parse_mqar_prompt but something in the Gemma forward path that produces
silent card output even when state["active"]=True.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

assert "m" in globals() and "tok" in globals(), (  # type: ignore[name-defined]
    "run via bin/gemma-run scripts/r22f_live_parse_trace.py"
)

from calm.llm_computer.facades.retrieval import _monkey_patch_fast_encode  # noqa
_monkey_patch_fast_encode(tok)  # type: ignore[name-defined]

# Load the exact parse + corpus code from r22b_round2 / r22_install
import importlib.util as _ilu
_r2_src = (ROOT / "scripts" / "r22b_round2.py").read_text()
_r2_src = _r2_src.split("main()\nprint(\"R22B_R2_DONE\")")[0]
_ns = {"__name__": "_r22b_r2", "__file__": str(ROOT / "scripts" / "r22b_round2.py"),
       "m": m, "tok": tok}  # type: ignore[name-defined]
exec(compile(_r2_src, str(ROOT / "scripts" / "r22b_round2.py"), "exec"), _ns)
make_prompt = _ns["make_prompt"]
parse_mqar_prompt = _ns["parse_mqar_prompt"]
mqar_to_ids = _ns["mqar_to_ids"]
load_mqar_card = _ns["load_mqar_card"]


def main():
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

    print("[r22f-trace] regenerating exact R22 corpus...")
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
                })
    print(f"  {len(candidates)} candidates")

    # Run parse_mqar_prompt on each — reasons
    print("[r22f-trace] running parse_mqar_prompt...")
    parse_ok_count = 0
    parse_fail_by_n = defaultdict(int)
    parse_ok_cases = []
    parse_fail_cases = []
    for c in candidates:
        mqar = parse_mqar_prompt(c["prompt"])
        c["parse_ok"] = mqar is not None
        c["mqar_str"] = mqar
        if mqar is not None:
            parse_ok_count += 1
            parse_ok_cases.append(c)
        else:
            parse_fail_by_n[c["n_pairs"]] += 1
            parse_fail_cases.append(c)

    print(f"  parse OK: {parse_ok_count}/{len(candidates)}")
    for n in sorted(parse_fail_by_n.keys()):
        print(f"    N={n} parse_fail: {parse_fail_by_n[n]}")

    # If parse succeeded on all, cross-ref with the cached jsonl to see
    # which ones were silent in the live run
    cache = ROOT / ".cache" / "r22b" / "round6_gated_write.jsonl"
    cached = {
        (c["seed"], c["n_pairs"], c["distractor_tokens"], c["mode"], c["replica"]): c
        for c in [json.loads(l) for l in cache.open()]
    }

    # Cross-index parse results with cached live-run silence
    print("\n[r22f-trace] cross-reference with .cache/r22b/round6_gated_write.jsonl")
    n_mismatch = 0
    by_n_mismatch = defaultdict(lambda: {"silent_but_parse_ok": 0, "silent_parse_fail": 0})
    for c in candidates:
        key = (c["seed"], c["n_pairs"], c["distractor_tokens"],
               c["mode"], c["replica"])
        live = cached.get(key)
        if not live:
            continue
        live_silent = live["card_margin"] == 0.0 and live["card_argmax"] == 0
        if live_silent:
            if c["parse_ok"]:
                by_n_mismatch[c["n_pairs"]]["silent_but_parse_ok"] += 1
                n_mismatch += 1
            else:
                by_n_mismatch[c["n_pairs"]]["silent_parse_fail"] += 1

    print(f"\n  MISMATCH CASES: parse_ok=True but live run silent = {n_mismatch}")
    for n in sorted(by_n_mismatch.keys()):
        print(f"    N={n}: parse_ok+silent={by_n_mismatch[n]['silent_but_parse_ok']}  "
              f"parse_fail+silent={by_n_mismatch[n]['silent_parse_fail']}")

    # Actually run the card standalone on parse-ok prompts and see
    # what margin it produces — is the card itself silent?
    print("\n[r22f-trace] running card standalone on parse-ok N>=10 prompts...")
    card = load_mqar_card(ROOT / "calm/hrm/checkpoints/copy_augmented_delta_mqar_best.pt")
    from calm.hrm.data import _ID_TO_CHAR

    by_n_standalone = defaultdict(lambda: {"correct": 0, "total": 0,
                                            "margins": []})
    sample_silent = []
    for c in [p for p in parse_ok_cases if p["n_pairs"] >= 10][:40]:
        ids = torch.tensor([mqar_to_ids(c["mqar_str"])], device="cuda")
        with torch.no_grad():
            out = card(ids)
        last = out[0, -1].float()
        argmax = int(last.argmax().item())
        argmax_char = _ID_TO_CHAR.get(argmax, "?")
        peak = last.max().item()
        med = last.median().item()
        margin = peak - med
        expected = c["expected"]
        is_correct = argmax_char == expected
        by_n_standalone[c["n_pairs"]]["total"] += 1
        by_n_standalone[c["n_pairs"]]["margins"].append(margin)
        if is_correct:
            by_n_standalone[c["n_pairs"]]["correct"] += 1
        # Look at the ones that the live run marked silent — are they
        # actually confident standalone?
        key = (c["seed"], c["n_pairs"], c["distractor_tokens"],
               c["mode"], c["replica"])
        live = cached.get(key)
        if live and live["card_margin"] == 0.0 and live["card_argmax"] == 0:
            if len(sample_silent) < 10:
                sample_silent.append({
                    "case": key, "mqar": c["mqar_str"][:60],
                    "expected": expected, "argmax": argmax_char,
                    "margin_standalone": margin, "correct": is_correct,
                })

    print("\n  STANDALONE CARD on parse-ok N>=10 (up to 40):")
    for n in sorted(by_n_standalone.keys()):
        s = by_n_standalone[n]
        if not s["total"]:
            continue
        margins = sorted(s["margins"])
        print(f"    N={n}: {s['correct']}/{s['total']} correct  "
              f"margin_p50={margins[len(margins)//2]:.2f} "
              f"p5={margins[max(0,len(margins)//20)]:.2f} "
              f"p95={margins[min(len(margins)-1, 19*len(margins)//20)]:.2f}")

    if sample_silent:
        print("\n  CASES silent in live run but standalone correct:")
        for s in sample_silent:
            print(f"    {s['case']} exp={s['expected']!r} "
                  f"standalone_argmax={s['argmax']!r} margin={s['margin_standalone']:.2f} "
                  f"correct={s['correct']}")


main()
print("R22F_TRACE_DONE")
