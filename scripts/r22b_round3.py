"""R22b round 3 — calibrate min_margin threshold via post-hoc sweep.

Round 2 showed 2 WINS / 4 REGRESSIONS with VerificationHook min_margin=0.5.
Hypothesis: card's (peak - median) margin distribution differs between
wins (high margin, confident-correct) and regressions (moderate margin,
confident-wrong). A threshold t exists such that card.margin >= t keeps
most wins and cuts most regressions.

Approach:
1. Run baseline (no card).
2. Run with-card but log card's (peak, median, margin, argmax) per prompt.
   The VerificationHook stays at min_margin=0.0 during forward (we decide
   post-hoc whether the boost should have fired).
3. Post-hoc sweep: for threshold t ∈ {0.5, 1, 2, 3, 5, 10, 20}, compute
   effective verdict per prompt:
     - if card.margin >= t: use card's mapped digit (hook would have fired)
     - else: use baseline's digit (hook silent)
   Count solves, wins, regressions per threshold.

Corpus: same 40 prompts as round 2 (same seed).
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
    "run via bin/gemma-run scripts/r22b_round3.py"
)

sys.path.insert(0, str(ROOT))
from calm.llm_computer.gemma_substrate import KVCache  # noqa: E402
from calm.hrm.data import _CHAR_TO_ID  # noqa: E402
from calm.llm_computer.facades.retrieval import _monkey_patch_fast_encode  # noqa: E402

_monkey_patch_fast_encode(tok)  # type: ignore[name-defined]

# Reuse round 2's corpus generator by importing that script's functions
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


def score_baseline(ids, digit_ids):
    cache = KVCache(m.config.n_layers, device="cuda")  # type: ignore[name-defined]
    with torch.no_grad():
        logits = m.forward(  # type: ignore[name-defined]
            torch.tensor([ids]), device="cuda",
            kv_cache=cache, start_pos=0,
        )
    top = int(logits[0, -1].argmax())
    return top


def score_with_card(ids, digit_ids, slot):
    """Run Gemma forward with card attached. Return Gemma's pre-hook
    argmax + card's (peak, median, argmax, margin) from slot.last_output."""
    cache = KVCache(m.config.n_layers, device="cuda")  # type: ignore[name-defined]
    with torch.no_grad():
        logits = m.forward(  # type: ignore[name-defined]
            torch.tensor([ids]), device="cuda",
            kv_cache=cache, start_pos=0,
        )
    # VerificationHook ran (if min_margin=0 it always fires). We want
    # Gemma's PRE-HOOK logits to compute post-hoc. With min_margin=0 fired
    # with boost=50, the emitted top is whatever beat the boosted card digit.
    # For post-hoc sweep we need the UNBIASED logits. Two options:
    #   (a) temporarily remove the hook before forward, then re-add.
    #   (b) use baseline pass's result (separate run).
    # We use (b) — baseline is already run separately.
    gemma_top = int(logits[0, -1].argmax())  # post-hook, for sanity
    card_out = slot.last_output
    last = card_out[0, -1].float()
    card_argmax = int(last.argmax().item())
    peak = last.max().item()
    med = last.median().item()
    margin = peak - med
    return {"gemma_post_hook_top": gemma_top,
            "card_argmax": card_argmax,
            "peak": peak, "median": med, "margin": margin}


def main():
    rng = random.Random(2026_04_22)  # same seed as round 2

    axes = []
    for n_pairs in (5, 10):
        for dist_tok, mode in [
            (500, "neutral"), (500, "confusing"),
            (1500, "neutral_long"), (1500, "confusing_long"),
        ]:
            axes.append((n_pairs, dist_tok, mode))

    REPLICAS = 5
    candidates = []
    for (n_pairs, dist_tok, mode) in axes:
        for r in range(REPLICAS):
            prompt, q_key, expected = make_prompt(n_pairs, dist_tok, mode, rng)
            candidates.append({
                "n_pairs": n_pairs, "distractor_tokens": dist_tok,
                "mode": mode, "replica": r, "prompt": prompt,
                "query_key": q_key, "expected": expected,
            })
    print(f"[r22b.r3] {len(candidates)} candidates")

    digit_ids = build_gemma_digit_ids()
    for c in candidates:
        c["ids"] = tok.encode(c["prompt"])  # type: ignore[name-defined]

    # --- BASELINE (no card) ---
    print("[r22b.r3] BASELINE pass...")
    clear_card_state()
    t0 = time.time()
    for c in candidates:
        c["baseline_top"] = score_baseline(c["ids"], digit_ids)
    print(f"  done in {time.time() - t0:.1f}s")

    # --- With card (min_margin=0 — always captures output, analyze post-hoc) ---
    print("[r22b.r3] installing MQAR card (min_margin=0 for capture)...")
    card = load_mqar_card(ROOT / "calm/hrm/checkpoints/copy_augmented_delta_mqar_best.pt")
    slot, state, hook = install(m, card, layer_idx=30, ch_off=2480)  # type: ignore[name-defined]
    hook.min_margin = 0.0  # always fire during capture; post-hoc decides

    CARD_N_RANGE = {5, 10, 15}

    print("[r22b.r3] CAPTURE pass (with card, log margins)...")
    t0 = time.time()
    for c in candidates:
        if c["n_pairs"] not in CARD_N_RANGE:
            state["mqar_ids"] = None
            state["active"] = False
            c["card_active"] = False
        else:
            mqar_str = parse_mqar_prompt(c["prompt"])
            if mqar_str is None:
                state["mqar_ids"] = None
                state["active"] = False
                c["card_active"] = False
            else:
                state["mqar_ids"] = mqar_to_ids(mqar_str)
                state["active"] = True
                c["card_active"] = True
        r = score_with_card(c["ids"], digit_ids, slot)
        c.update(r)
    print(f"  done in {time.time() - t0:.1f}s")

    # --- Post-hoc threshold sweep ---
    # For each threshold t, effective top =
    #   card's mapped digit (if card_active AND card.margin >= t AND
    #                        card_argmax in vocab_mapping)
    #   else baseline_top
    vocab_mapping = {
        _CHAR_TO_ID[str(d)]: digit_ids[d] for d in range(10)
    }

    def verdict_for_threshold(c, t):
        if not c["card_active"] or c["margin"] < t:
            return c["baseline_top"]
        gemma_tok = vocab_mapping.get(c["card_argmax"])
        if gemma_tok is None:
            return c["baseline_top"]
        return gemma_tok

    thresholds = [0.0, 0.5, 2.0, 5.0, 10.0, 15.0, 18.0, 19.0, 19.5,
                   20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 25.0]
    print("\n=== THRESHOLD SWEEP ===")
    print(f"  {'t':>6}  {'solves':>7}  {'wins':>5}  {'regr':>5}  {'net':>5}")
    baseline_solves = sum(
        1 for c in candidates
        if c["baseline_top"] == digit_ids[int(c["expected"])]
    )
    print(f"  {'base':>6}  {baseline_solves:>3}/{len(candidates):<3}  "
          f"  —       —      —")

    results = []
    for t in thresholds:
        solves = 0
        wins = 0
        regr = 0
        for c in candidates:
            expected_id = digit_ids[int(c["expected"])]
            eff_top = verdict_for_threshold(c, t)
            base_solved = c["baseline_top"] == expected_id
            eff_solved = eff_top == expected_id
            if eff_solved:
                solves += 1
            if not base_solved and eff_solved:
                wins += 1
            if base_solved and not eff_solved:
                regr += 1
        net = solves - baseline_solves
        sign = "+" if net > 0 else ""
        print(f"  {t:>6.1f}  {solves:>3}/{len(candidates):<3}  "
              f"{wins:>4}   {regr:>4}   {sign}{net}")
        results.append({"threshold": t, "solves": solves,
                         "wins": wins, "regr": regr, "net": net})

    # Show the best threshold + its margin distribution
    best = max(results, key=lambda r: r["net"])
    print(f"\n  best threshold: t={best['threshold']:.1f}  "
          f"net {best['net']:+d}  (wins {best['wins']}, regr {best['regr']})")

    # --- Margin distribution: wins vs regressions at t=0.5 (round-2 default) ---
    print("\n=== MARGIN DISTRIBUTION (t=0.5 baseline behavior) ===")
    print(f"  {'case':>6}  {'N':>2}  dist   mode             margin   base→card")
    for c in candidates:
        if not c["card_active"]:
            continue
        expected_id = digit_ids[int(c["expected"])]
        base_solved = c["baseline_top"] == expected_id
        # Simulate t=0.5 (round 2's gate)
        eff_top_05 = verdict_for_threshold(c, 0.5)
        eff_solved_05 = eff_top_05 == expected_id
        if not base_solved and eff_solved_05:
            kind = "WIN"
        elif base_solved and not eff_solved_05:
            kind = "REGR"
        else:
            continue
        base_ch = tok.id_to_token.get(c["baseline_top"], "?")  # type: ignore[name-defined]
        card_mapped = vocab_mapping.get(c["card_argmax"])
        card_ch = tok.id_to_token.get(card_mapped, "?") if card_mapped else "?"  # type: ignore[name-defined]
        print(f"  {kind:>6}  {c['n_pairs']:>2}  {c['distractor_tokens']:>4}  "
              f"{c['mode']:<18}  {c['margin']:>6.2f}   "
              f"{base_ch!r} → {card_ch!r}")

    # Save
    save = []
    for c in candidates:
        save.append({
            "n_pairs": c["n_pairs"], "distractor_tokens": c["distractor_tokens"],
            "mode": c["mode"], "replica": c["replica"],
            "query_key": c["query_key"], "expected": c["expected"],
            "card_active": c["card_active"],
            "baseline_top_id": c["baseline_top"],
            "card_argmax_card_vocab": c.get("card_argmax"),
            "card_peak": c.get("peak"),
            "card_median": c.get("median"),
            "card_margin": c.get("margin"),
        })
    out = CACHE / "round3_results.jsonl"
    with out.open("w") as f:
        for r in save:
            f.write(json.dumps(r) + "\n")
    print(f"\n[r22b.r3] results → {out}")


main()
print("R22B_R3_DONE")
