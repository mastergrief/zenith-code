"""R22b round 2 — confusing distractors + card in-distribution gate.

Hypothesis: Gemma's attention can be confused by prose that MENTIONS
key-value-adjacent patterns (e.g., "variable z rose to 8 last spring",
"the seventh row shows a value of 4"), even though such prose doesn't
match the strict <mem>k=v</mem> format. The PT+Delta MQAR card is
immune because its adapter only parses the literal <mem>...</mem> block.

Axes (round 2):
  - N_pairs in {5, 10}  (in card's training distribution {5,10,15},
    dropping round-1's N=3 which was OOD for card and trivial for Gemma)
  - distractor_mode in {neutral, confusing, neutral_long, confusing_long}
    - neutral: plain prose ~500 tok (baseline from round 1)
    - confusing: ~500 tok prose with scattered "variable X rose to Y"
      / "seven apples" phrasings that don't match <mem> regex
    - neutral_long: ~1500 tok plain prose
    - confusing_long: ~1500 tok confusing prose
  - needle_position: mid (fixed)

Adapter gate: the install() state dict now requires N_pairs ∈ {5..15}
before activating. Out-of-distribution N_pairs → card inactive, Gemma
natural output. (This fixes round-1's 2 regressions on N=3.)

= 2 × 4 = 8 cells × 5 replicas = 40 prompts.
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
    "run via bin/gemma-run scripts/r22b_round2.py"
)

sys.path.insert(0, str(ROOT))
from calm.llm_computer.gemma_substrate import KVCache  # noqa: E402
from calm.hrm.data import _CHAR_TO_ID  # noqa: E402
from calm.llm_computer.facades.retrieval import _monkey_patch_fast_encode  # noqa: E402

_monkey_patch_fast_encode(tok)  # type: ignore[name-defined]

import importlib.util as _ilu
_r22_src = (ROOT / "scripts" / "r22_install_mqar_card.py").read_text()
_r22_src = _r22_src.split("main()\nprint(\"R22_DONE\")")[0]
_ns = {"__name__": "_r22_mod",
       "__file__": str(ROOT / "scripts" / "r22_install_mqar_card.py"),
       "m": m, "tok": tok}  # type: ignore[name-defined]
exec(compile(_r22_src, str(ROOT / "scripts" / "r22_install_mqar_card.py"), "exec"),
     _ns)
load_mqar_card = _ns["load_mqar_card"]
install = _ns["install"]
parse_mqar_prompt = _ns["parse_mqar_prompt"]
mqar_to_ids = _ns["mqar_to_ids"]


# ============================================================================
# Distractor generators
# ============================================================================

_NEUTRAL = [
    "The sky grows dim as the evening settles on the quiet valley.",
    "Mountains rise sharply above the forest of pine and silver birch.",
    "Birds gather in the gray clouds before the autumn rains arrive.",
    "Rivers carve wide paths through stone that has stood for ages.",
    "Old libraries hold secrets that only patient readers may find.",
    "Travelers stop at the inn to rest and to share news of the road.",
    "Snow covers the hilltops long before it reaches the lower fields.",
    "Stars appear one by one as night spreads across the open sky.",
]

# Prose that mentions letter-digit associations in non-<mem>-regex form.
# Key patterns Gemma's attention might pick up but our regex will NOT match:
#   - "the value of X rose to Y"
#   - "X is a variable in the seventh row"
#   - "(X, Y) pair" — note the comma
_CONFUSING = [
    "Previously the value of q rose to 2 before the market closed.",
    "Our records suggest that variable m was set near 7 early last week.",
    "The seventh chapter of the book introduces a symbol named z.",
    "An analyst noted that the column labeled w reached around 5 in May.",
    "The ledger lists pair (s, 8) as the baseline entry for spring.",
    "Observers recorded that k trended toward 3 over the summer months.",
    "Notes from the workshop mention that h held at approximately 6.",
    "A footnote clarifies that d reached roughly 4 before the audit.",
    "The chart groups variable n near the value 9 in its left column.",
    "Eight apples sat in the basket beside the letter r on the shelf.",
]


def _approx_tokens(sent: str) -> int:
    return len(tok.encode(sent))  # type: ignore[name-defined]


def make_distractor(target_tokens: int, mode: str, rng: random.Random) -> str:
    if target_tokens <= 0:
        return ""
    pool = _CONFUSING if mode.startswith("confusing") else _NEUTRAL
    per = _approx_tokens(pool[0])
    n_sents = max(1, (target_tokens + per - 1) // per)
    return " ".join(rng.choice(pool) for _ in range(n_sents))


_KEY_POOL = list("abcdefghijklmnopqrstuvwxyz")
_VAL_POOL = [str(d) for d in range(10)]


def make_prompt(n_pairs: int, distractor_tokens: int, mode: str,
                rng: random.Random) -> tuple[str, str, str]:
    keys = rng.sample(_KEY_POOL, n_pairs)
    values = [rng.choice(_VAL_POOL) for _ in range(n_pairs)]
    q_idx = rng.randrange(n_pairs)
    q_key = keys[q_idx]
    expected = values[q_idx]
    mem_body = " ".join(f"{k}={v}" for k, v in zip(keys, values))
    mem_block = f"<mem>{mem_body}</mem>"
    half = distractor_tokens // 2
    prefix = make_distractor(half, mode, rng)
    suffix = make_distractor(distractor_tokens - half, mode, rng)
    body = f"{prefix}\n\n{mem_block}\n\n{suffix}"
    prompt = f"{body}\n\nQuestion: What is the value of {q_key}? Answer: "
    return prompt, q_key, expected


def build_gemma_digit_ids():
    return {d: tok.encode(f" {d}")[-1] for d in range(10)}  # type: ignore[name-defined]


def score(ids: list, expected: str, digit_ids: dict) -> dict:
    cache = KVCache(m.config.n_layers, device="cuda")  # type: ignore[name-defined]
    with torch.no_grad():
        logits = m.forward(  # type: ignore[name-defined]
            torch.tensor([ids]), device="cuda",
            kv_cache=cache, start_pos=0,
        )
    top = int(logits[0, -1].argmax())
    top_tok = tok.id_to_token.get(top, "?")  # type: ignore[name-defined]
    got = top_tok.lstrip(" ▁")
    expected_id = digit_ids[int(expected)]
    if top == expected_id:
        verdict = "solve"
    elif top in digit_ids.values():
        verdict = "wrong_digit"
    else:
        verdict = "non_digit"
    return {"top": top, "top_tok": top_tok, "got": got, "verdict": verdict}


def clear_card_state():
    for lyr in m.layers:  # type: ignore[name-defined]
        if hasattr(lyr, "card_slots"):
            lyr.card_slots = []
    m.verification_hooks = []  # type: ignore[name-defined]
    m.reserved_channels = []  # type: ignore[name-defined]


def main():
    rng = random.Random(2026_04_22)

    axes = []
    for n_pairs in (5, 10):
        for dist_tok, mode in [
            (500, "neutral"),
            (500, "confusing"),
            (1500, "neutral_long"),
            (1500, "confusing_long"),
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
    print(f"[r22b.r2] {len(candidates)} candidates "
          f"({len(axes)} cells × {REPLICAS} replicas)")

    digit_ids = build_gemma_digit_ids()

    print("[r22b.r2] pre-tokenizing...")
    t0 = time.time()
    for c in candidates:
        c["ids"] = tok.encode(c["prompt"])  # type: ignore[name-defined]
    print(f"  tokenized {len(candidates)} in {time.time() - t0:.1f}s")

    # --- Warmup ---
    clear_card_state()
    shape_lens = sorted({len(c["ids"]) for c in candidates})
    print(f"[r22b.r2] warmup: {len(shape_lens)} shapes")
    first_by_len = {}
    for c in candidates:
        ln = len(c["ids"])
        first_by_len.setdefault(ln, c)
    for ln in shape_lens:
        t0 = time.time()
        _ = score(first_by_len[ln]["ids"], first_by_len[ln]["expected"],
                  digit_ids)
        print(f"  S={ln:>4} warmup: {time.time() - t0:.1f}s")

    # --- BASELINE ---
    print("\n[r22b.r2] BASELINE pass...")
    clear_card_state()
    t0 = time.time()
    baseline = []
    for c in candidates:
        r = score(c["ids"], c["expected"], digit_ids)
        baseline.append(r)
    print(f"  done in {time.time() - t0:.1f}s")

    # --- Install card + with-card pass ---
    print("\n[r22b.r2] installing MQAR card...")
    ckpt_path = ROOT / "calm/hrm/checkpoints/copy_augmented_delta_mqar_best.pt"
    card = load_mqar_card(ckpt_path)
    slot, state, hook = install(m, card, layer_idx=30, ch_off=2480)  # type: ignore[name-defined]

    # Adapter gate: only fire card for N_pairs in card's training distribution
    CARD_N_RANGE = {5, 10, 15}

    print("\n[r22b.r2] WITH-CARD pass...")
    t0 = time.time()
    with_card = []
    for c in candidates:
        if c["n_pairs"] not in CARD_N_RANGE:
            state["mqar_ids"] = None
            state["active"] = False
            parse_ok = False
            gated_out = True
        else:
            mqar_str = parse_mqar_prompt(c["prompt"])
            if mqar_str is None:
                state["mqar_ids"] = None
                state["active"] = False
                parse_ok = False
            else:
                state["mqar_ids"] = mqar_to_ids(mqar_str)
                state["active"] = True
                parse_ok = True
            gated_out = False
        r = score(c["ids"], c["expected"], digit_ids)
        r["parse_ok"] = parse_ok
        r["gated_out"] = gated_out
        with_card.append(r)
    print(f"  done in {time.time() - t0:.1f}s")

    # --- Per-cell report ---
    from collections import defaultdict
    cell_base = defaultdict(lambda: {"solve": 0, "total": 0})
    cell_card = defaultdict(lambda: {"solve": 0, "total": 0})
    for b, w, c in zip(baseline, with_card, candidates):
        k = (c["n_pairs"], c["distractor_tokens"], c["mode"])
        cell_base[k]["total"] += 1
        cell_card[k]["total"] += 1
        if b["verdict"] == "solve":
            cell_base[k]["solve"] += 1
        if w["verdict"] == "solve":
            cell_card[k]["solve"] += 1

    print("\n=== PER-CELL (baseline vs with-card) ===")
    print(f"  {'N':>3}  {'dist':>5}  {'mode':>18}  base   card   Δ")
    for k in sorted(cell_base.keys()):
        b, c = cell_base[k], cell_card[k]
        delta = c["solve"] - b["solve"]
        sign = "+" if delta > 0 else ""
        print(f"  {k[0]:>3}  {k[1]:>5}  {k[2]:>18}  "
              f"{b['solve']}/{b['total']}    "
              f"{c['solve']}/{c['total']}    {sign}{delta}")

    n_base = sum(1 for b in baseline if b["verdict"] == "solve")
    n_card = sum(1 for w in with_card if w["verdict"] == "solve")
    print(f"\n=== OVERALL ===")
    print(f"  baseline:  {n_base}/{len(baseline)}")
    print(f"  with card: {n_card}/{len(with_card)}  (Δ={n_card - n_base:+d})")

    # Which prompts did Gemma miss but card solve (the product win) ?
    card_wins = [
        (c, b, w) for c, b, w in zip(candidates, baseline, with_card)
        if b["verdict"] != "solve" and w["verdict"] == "solve"
    ]
    card_regress = [
        (c, b, w) for c, b, w in zip(candidates, baseline, with_card)
        if b["verdict"] == "solve" and w["verdict"] != "solve"
    ]
    print(f"\n  card_WINS   (base✗ card✓): {len(card_wins)}")
    print(f"  card_REGRESS(base✓ card✗): {len(card_regress)}")
    for c, b, w in card_wins:
        print(f"    WIN   N={c['n_pairs']} dist={c['distractor_tokens']} "
              f"mode={c['mode']} q={c['query_key']} exp={c['expected']} "
              f"base={b['top_tok']!r} card={w['top_tok']!r}")
    for c, b, w in card_regress:
        print(f"    REGR  N={c['n_pairs']} dist={c['distractor_tokens']} "
              f"mode={c['mode']} q={c['query_key']} exp={c['expected']} "
              f"base={b['top_tok']!r} card={w['top_tok']!r}")

    # Save
    result_path = CACHE / "round2_results.jsonl"
    with result_path.open("w") as f:
        for c, b, w in zip(candidates, baseline, with_card):
            rec = {
                **{k: c[k] for k in ("n_pairs", "distractor_tokens", "mode",
                                      "replica", "query_key", "expected")},
                "n_tokens": len(c["ids"]),
                "baseline_verdict": b["verdict"],
                "baseline_top": b["top_tok"],
                "card_verdict": w["verdict"],
                "card_top": w["top_tok"],
                "parse_ok": w["parse_ok"],
                "gated_out": w["gated_out"],
            }
            f.write(json.dumps(rec) + "\n")
    print(f"\n[r22b.r2] results → {result_path}")


main()
print("R22B_R2_DONE")
