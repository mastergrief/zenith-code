"""R22d — N-fold retrieval test (multi-needle proxy).

Rather than asking Gemma for a single random key per <mem> prompt,
query EACH of the N keys in the mem block. Measures the card's
intrinsic retrieval reliability across the full key distribution,
not just the random sample of positions chosen by the other rounds.

If the card is 100% standalone on training-MQAR but only 50-70%
effective on adapter-extracted live prompts, the loss is distributed
across keys. This test isolates per-key reliability.

Corpus: 3 prompts × N=10 keys/prompt = 30 retrieval tests per seed,
2 seeds = 60 tests. Pooled run. Each test: same <mem>...</mem>
prefix + 500-tok confusing distractor, query varies to each of
the 10 keys.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
assert "m" in globals() and "tok" in globals(), (  # type: ignore[name-defined]
    "run via bin/gemma-run scripts/r22d_nfold_retrieval.py"
)

sys.path.insert(0, str(ROOT))
from calm.llm_computer.gemma_substrate import KVCache
from calm.hrm.data import _CHAR_TO_ID
from calm.llm_computer.facades.retrieval import _monkey_patch_fast_encode

_monkey_patch_fast_encode(tok)  # type: ignore[name-defined]

# Reuse r22b_round2.py's corpus helpers
import importlib.util as _ilu
_r2_src = (ROOT / "scripts" / "r22b_round2.py").read_text()
_r2_src = _r2_src.split("main()\nprint(\"R22B_R2_DONE\")")[0]
_ns = {"__name__": "_r22b_r2", "__file__": str(ROOT / "scripts" / "r22b_round2.py"),
       "m": m, "tok": tok}  # type: ignore[name-defined]
exec(compile(_r2_src, str(ROOT / "scripts" / "r22b_round2.py"), "exec"), _ns)
make_distractor = _ns["make_distractor"]
build_gemma_digit_ids = _ns["build_gemma_digit_ids"]
load_mqar_card = _ns["load_mqar_card"]
install = _ns["install"]
parse_mqar_prompt = _ns["parse_mqar_prompt"]
mqar_to_ids = _ns["mqar_to_ids"]


def make_prompt_for_key(keys, values, query_key, distractor_tokens,
                         mode, rng):
    """Build a full distractor-wrapped prompt querying `query_key`."""
    mem_body = " ".join(f"{k}={v}" for k, v in zip(keys, values))
    mem_block = f"<mem>{mem_body}</mem>"
    half = distractor_tokens // 2
    prefix = make_distractor(half, mode, rng)
    suffix = make_distractor(distractor_tokens - half, mode, rng)
    body = f"{prefix}\n\n{mem_block}\n\n{suffix}"
    return f"{body}\n\nQuestion: What is the value of {query_key}? Answer: "


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
    MIN_MARGIN = 22.0
    N_PROMPTS = 3
    N_PAIRS = 10
    DISTRACTOR = 500
    MODE = "confusing"
    seeds = [2026_04_22, 2026_04_23]

    key_pool = list("abcdefghijklmnopqrstuvwxyz")
    val_pool = [str(d) for d in range(10)]
    digit_ids = build_gemma_digit_ids()

    # Generate base mem blocks
    mems = []
    for seed in seeds:
        rng = random.Random(seed)
        for i in range(N_PROMPTS):
            keys = rng.sample(key_pool, N_PAIRS)
            values = [rng.choice(val_pool) for _ in range(N_PAIRS)]
            mems.append({"seed": seed, "idx": i, "keys": keys,
                          "values": values, "rng": rng})

    # Generate all query prompts
    probes = []
    for mem in mems:
        for k_idx, k in enumerate(mem["keys"]):
            v = mem["values"][k_idx]
            prompt = make_prompt_for_key(
                mem["keys"], mem["values"], k,
                DISTRACTOR, MODE, mem["rng"],
            )
            probes.append({
                "seed": mem["seed"], "mem_idx": mem["idx"],
                "key_idx": k_idx, "query_key": k, "expected": v,
                "prompt": prompt,
            })
    print(f"[r22d] {len(probes)} probes "
          f"({N_PROMPTS} prompts × {N_PAIRS} keys × {len(seeds)} seeds)")

    for p in probes:
        p["ids"] = tok.encode(p["prompt"])  # type: ignore[name-defined]

    # --- BASELINE ---
    print("[r22d] BASELINE pass...")
    clear_card_state()
    t0 = time.time()
    for p in probes:
        top, top_tok = score(p["ids"])
        p["baseline_top"] = top
        p["baseline_top_tok"] = top_tok
    print(f"  done in {time.time() - t0:.1f}s")

    # --- Install + with-card ---
    print(f"[r22d] installing card (write_margin=min_margin={MIN_MARGIN})...")
    card = load_mqar_card(ROOT / "calm/hrm/checkpoints/copy_augmented_delta_mqar_best.pt")
    slot, state, hook = install(  # type: ignore[name-defined]
        m, card, layer_idx=30, ch_off=2480,
        write_margin=MIN_MARGIN, preserve=True,
    )
    hook.min_margin = MIN_MARGIN

    print("[r22d] WITH-CARD pass...")
    t0 = time.time()
    for p in probes:
        mqar_str = parse_mqar_prompt(p["prompt"])
        if mqar_str is None:
            state["active"] = False
        else:
            state["mqar_ids"] = mqar_to_ids(mqar_str)
            state["active"] = True
        top, top_tok = score(p["ids"])
        p["card_top"] = top
        p["card_top_tok"] = top_tok
        co = slot.last_output
        last = co[0, -1].float()
        p["card_margin"] = (last.max() - last.median()).item()
        p["card_argmax"] = int(last.argmax().item())
    print(f"  done in {time.time() - t0:.1f}s")

    # --- Report: per-mem accuracy grid ---
    from collections import defaultdict
    by_mem = defaultdict(list)
    for p in probes:
        by_mem[(p["seed"], p["mem_idx"])].append(p)

    print("\n=== PER-MEM ACCURACY (baseline vs card) ===")
    total_base = 0
    total_card = 0
    for (seed, idx), plist in sorted(by_mem.items()):
        base_correct = sum(
            1 for p in plist
            if p["baseline_top"] == digit_ids[int(p["expected"])]
        )
        card_correct = sum(
            1 for p in plist
            if p["card_top"] == digit_ids[int(p["expected"])]
        )
        total_base += base_correct
        total_card += card_correct
        delta = card_correct - base_correct
        sign = "+" if delta > 0 else ""
        print(f"  seed={seed} mem#{idx}  base={base_correct:>2}/{len(plist):<2}  "
              f"card={card_correct:>2}/{len(plist):<2}  Δ={sign}{delta}")

    print(f"\n=== OVERALL ===")
    print(f"  baseline:  {total_base}/{len(probes)}")
    print(f"  with card: {total_card}/{len(probes)}  "
          f"(Δ={total_card - total_base:+d})")

    # Per-key-index accuracy (does position matter?)
    by_idx = defaultdict(lambda: {"base": 0, "card": 0, "total": 0})
    for p in probes:
        expected_id = digit_ids[int(p["expected"])]
        by_idx[p["key_idx"]]["total"] += 1
        if p["baseline_top"] == expected_id:
            by_idx[p["key_idx"]]["base"] += 1
        if p["card_top"] == expected_id:
            by_idx[p["key_idx"]]["card"] += 1
    print("\n=== BY KEY POSITION (within mem block) ===")
    print(f"  {'idx':>3}   base   card")
    for idx in sorted(by_idx.keys()):
        s = by_idx[idx]
        print(f"  {idx:>3}   {s['base']:>2}/{s['total']:<2}   "
              f"{s['card']:>2}/{s['total']:<2}")


main()
print("R22D_DONE")
