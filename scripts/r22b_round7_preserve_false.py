"""R22b round 7 — preserve=False test to close round 6's `q=v` mystery.

Round 5 finding: CardSlot's card_output_fn always writes card's log-probs
to reserved residual channels [2480:2560], regardless of whether the
VerificationHook fires. That write flows through head projection and can
shift Gemma's output even when hook is silent. 6 of 7 regressions in
round 5 had fired=False.

Fix (in scripts/r22_install_mqar_card.py::install): new `write_margin`
param. When card.margin < write_margin, card_output_fn zeros the logits
AND skips the residual write. Aligns the h write gate with the hook gate.

Test: same 60-prompt pooled corpus as round 5, write_margin=22.0
(same threshold as hook). Expected: residual write now only happens on
confident cases, net should match round 3's claim (+2 ish) or better.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "r22b"
CACHE.mkdir(parents=True, exist_ok=True)

assert "m" in globals() and "tok" in globals(), (  # type: ignore[name-defined]
    "run via bin/gemma-run scripts/r22b_round6.py"
)

sys.path.insert(0, str(ROOT))
from calm.llm_computer.gemma_substrate import KVCache  # noqa: E402
from calm.hrm.data import _CHAR_TO_ID  # noqa: E402
from calm.llm_computer.facades.retrieval import _monkey_patch_fast_encode  # noqa: E402

_monkey_patch_fast_encode(tok)  # type: ignore[name-defined]

# Reuse round 2 corpus generator (identical make_prompt)
import importlib.util as _ilu
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


def main():
    MIN_MARGIN = 22.0  # from round 3

    # Pool two seeds to double sample size
    seeds = [2026_04_22, 2026_04_23]
    # Use only confusing distractors — round 2 showed neutral=no signal
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
                })
    print(f"[r22b.r7] {len(candidates)} candidates "
          f"({len(cells)} cells × {REPLICAS} reps × {len(seeds)} seeds)")

    digit_ids = build_gemma_digit_ids()
    for c in candidates:
        c["ids"] = tok.encode(c["prompt"])  # type: ignore[name-defined]

    # --- BASELINE ---
    print("[r22b.r7] BASELINE pass...")
    clear_card_state()
    t0 = time.time()
    for c in candidates:
        top, top_tok = score(c["ids"])
        c["baseline_top"] = top
        c["baseline_top_tok"] = top_tok
    print(f"  done in {time.time() - t0:.1f}s")

    # --- Install + with-card ---
    print(f"[r22b.r7] installing card (min_margin={MIN_MARGIN})...")
    card = load_mqar_card(ROOT / "calm/hrm/checkpoints/copy_augmented_delta_mqar_best.pt")
    # R22b round 6 fix: pass write_margin to gate the residual write too.
    slot, state, hook = install(  # type: ignore[name-defined]
        m, card, layer_idx=30, ch_off=2480,
        write_margin=MIN_MARGIN, preserve=False,
    )
    hook.min_margin = MIN_MARGIN

    CARD_N_RANGE = {5, 10, 15}

    print("[r22b.r7] WITH-CARD pass...")
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
        c["card_top"] = top
        c["card_top_tok"] = top_tok
        co = slot.last_output
        last = co[0, -1].float()
        c["card_peak"] = last.max().item()
        c["card_median"] = last.median().item()
        c["card_margin"] = c["card_peak"] - c["card_median"]
        c["card_argmax"] = int(last.argmax().item())
        c["fired"] = state["active"] and c["card_margin"] >= MIN_MARGIN
    print(f"  done in {time.time() - t0:.1f}s")

    # --- Per-cell + overall ---
    from collections import defaultdict
    cell_base = defaultdict(lambda: {"solve": 0, "total": 0})
    cell_card = defaultdict(lambda: {"solve": 0, "total": 0})
    for c in candidates:
        k = (c["n_pairs"], c["distractor_tokens"], c["mode"])
        expected_id = digit_ids[int(c["expected"])]
        cell_base[k]["total"] += 1
        cell_card[k]["total"] += 1
        if c["baseline_top"] == expected_id:
            cell_base[k]["solve"] += 1
        if c["card_top"] == expected_id:
            cell_card[k]["solve"] += 1

    print("\n=== PER-CELL (pooled 2-seed) ===")
    print(f"  {'N':>3}  {'dist':>5}  {'mode':>18}  base    card    Δ")
    for k in sorted(cell_base.keys()):
        b, cc = cell_base[k], cell_card[k]
        delta = cc["solve"] - b["solve"]
        sign = "+" if delta > 0 else ""
        print(f"  {k[0]:>3}  {k[1]:>5}  {k[2]:>18}  "
              f"{b['solve']:>2}/{b['total']:<2}   "
              f"{cc['solve']:>2}/{cc['total']:<2}   {sign}{delta}")

    n_base = sum(
        1 for c in candidates
        if c["baseline_top"] == digit_ids[int(c["expected"])]
    )
    n_card = sum(
        1 for c in candidates
        if c["card_top"] == digit_ids[int(c["expected"])]
    )
    print(f"\n=== OVERALL (pooled, t={MIN_MARGIN}) ===")
    print(f"  baseline:  {n_base}/{len(candidates)}")
    print(f"  with card: {n_card}/{len(candidates)}  (Δ={n_card - n_base:+d})")

    n_fired = sum(1 for c in candidates if c["fired"])
    print(f"  hook fired: {n_fired}/{len(candidates)}")

    # Wins / regressions
    wins = []
    regr = []
    for c in candidates:
        expected_id = digit_ids[int(c["expected"])]
        base_ok = c["baseline_top"] == expected_id
        card_ok = c["card_top"] == expected_id
        if not base_ok and card_ok:
            wins.append(c)
        elif base_ok and not card_ok:
            regr.append(c)
    print(f"\n  WINS  (base✗ card✓): {len(wins)}")
    for c in wins:
        print(f"    seed={c['seed']} N={c['n_pairs']} dist={c['distractor_tokens']} "
              f"mode={c['mode']} q={c['query_key']} exp={c['expected']} "
              f"base={c['baseline_top_tok']!r} card={c['card_top_tok']!r} "
              f"margin={c['card_margin']:.2f}")
    print(f"  REGR  (base✓ card✗): {len(regr)}")
    for c in regr:
        print(f"    seed={c['seed']} N={c['n_pairs']} dist={c['distractor_tokens']} "
              f"mode={c['mode']} q={c['query_key']} exp={c['expected']} "
              f"base={c['baseline_top_tok']!r} card={c['card_top_tok']!r} "
              f"margin={c['card_margin']:.2f} fired={c['fired']}")

    # Save
    out = CACHE / "round6_gated_write.jsonl"
    with out.open("w") as f:
        for c in candidates:
            rec = {k: v for k, v in c.items() if k not in ("ids", "prompt")}
            f.write(json.dumps(rec) + "\n")
    print(f"\n[r22b.r7] results → {out}")


main()
print("R22B_R7_DONE")
