"""R22b gate+lift — find Gemma's failure surface AND measure card lift.

Combined pass (replaces the separate gate + lift scripts):
1. Generate candidate corpus.
2. Explicit warmup: run ONE forward per unique prefill shape BEFORE timing,
   so Triton autotune stalls (~60-120s per new shape) don't pollute the
   timing loop.
3. For each candidate: run stock-Gemma forward → baseline verdict. Then
   install card, run Gemma-with-card forward → lift verdict.
4. Partition by cell + report.

Axes (reduced from initial draft — 1500-token-distractor dropped because
the prod Gemma prefill at S>1000 is ~5-15s per call even with warmup,
making iteration too slow. 1500+ stressed in a dedicated later round):

  - N_pairs in {3, 5, 10}
  - distractor_tokens in {0, 500}
  - needle_position: mid (fixed)

= 6 cells × 5 replicas = 30 prompts.
= 60 forward passes (baseline + with-card per prompt).

Usage: bin/gemma-run scripts/r22b_gate.py
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
    "run via bin/gemma-run scripts/r22b_gate.py"
)

sys.path.insert(0, str(ROOT))
from calm.llm_computer.gemma_substrate import KVCache, CardSlot, VerificationHook  # noqa: E402
from calm.hrm.data import _CHAR_TO_ID, _ID_TO_CHAR, VOCAB_SIZE  # noqa: E402
from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta  # noqa: E402
from calm.llm_computer.facades.retrieval import _monkey_patch_fast_encode  # noqa: E402

# Patch the naive O(vocab × len) GemmaTokenizer.encode with a trie-backed
# O(len) version — 13,000× speedup, critical for this script's 30-prompt
# warm-up loop at 500-char prompts (otherwise ~60s just to tokenize).
_monkey_patch_fast_encode(tok)  # type: ignore[name-defined]

# Import R22a adapter helpers via importlib (scripts/ isn't a package).
import importlib.util as _ilu  # noqa: E402
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


_DISTRACTOR_SENTENCES = [
    "The sky grows dim as the evening settles on the quiet valley.",
    "Mountains rise sharply above the forest of pine and silver birch.",
    "Birds gather in the gray clouds before the autumn rains arrive.",
    "Rivers carve wide paths through stone that has stood for ages.",
    "Old libraries hold secrets that only patient readers may find.",
    "Travelers stop at the inn to rest and to share news of the road.",
    "Snow covers the hilltops long before it reaches the lower fields.",
    "Stars appear one by one as night spreads across the open sky.",
    "Children chase the leaves that drift slowly down from tall maples.",
    "Bakers rise before dawn to knead dough and heat the stone ovens.",
]


def make_distractor(target_tokens: int, rng: random.Random) -> str:
    if target_tokens <= 0:
        return ""
    per = len(tok.encode(_DISTRACTOR_SENTENCES[0]))  # type: ignore[name-defined]
    n_sents = max(1, (target_tokens + per - 1) // per)
    return " ".join(rng.choice(_DISTRACTOR_SENTENCES) for _ in range(n_sents))


_KEY_POOL = list("abcdefghijklmnopqrstuvwxyz")
_VAL_POOL = [str(d) for d in range(10)]


def make_prompt(n_pairs: int, distractor_tokens: int,
                rng: random.Random) -> tuple[str, str, str]:
    keys = rng.sample(_KEY_POOL, n_pairs)
    values = [rng.choice(_VAL_POOL) for _ in range(n_pairs)]
    q_idx = rng.randrange(n_pairs)
    q_key = keys[q_idx]
    expected = values[q_idx]
    mem_body = " ".join(f"{k}={v}" for k, v in zip(keys, values))
    mem_block = f"<mem>{mem_body}</mem>"
    if distractor_tokens == 0:
        body = mem_block
    else:
        half = distractor_tokens // 2
        prefix = make_distractor(half, rng)
        suffix = make_distractor(distractor_tokens - half, rng)
        body = f"{prefix}\n\n{mem_block}\n\n{suffix}"
    prompt = f"{body}\n\nQuestion: What is the value of {q_key}? Answer: "
    return prompt, q_key, expected


def build_gemma_digit_ids():
    return {d: tok.encode(f" {d}")[-1] for d in range(10)}  # type: ignore[name-defined]


def score(prompt: str, expected: str, digit_ids: dict,
          ids: list | None = None) -> dict:
    if ids is None:
        ids = tok.encode(prompt)  # type: ignore[name-defined]
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
    return {"top": top, "top_tok": top_tok, "got": got, "verdict": verdict,
            "n_tokens": len(ids)}


def clear_card_state():
    """Remove any installed CardSlots / VerificationHooks."""
    for lyr in m.layers:  # type: ignore[name-defined]
        if hasattr(lyr, "card_slots"):
            lyr.card_slots = []
    m.verification_hooks = []  # type: ignore[name-defined]
    m.reserved_channels = []  # type: ignore[name-defined]


def main():
    rng = random.Random(2026_04_21)

    # --- Generate candidates ---
    axes = [(n, d) for n in (3, 5, 10) for d in (0, 500)]
    REPLICAS = 5
    candidates = []
    for (n_pairs, dist_tok) in axes:
        for r in range(REPLICAS):
            prompt, q_key, expected = make_prompt(n_pairs, dist_tok, rng)
            candidates.append({
                "n_pairs": n_pairs, "distractor_tokens": dist_tok,
                "replica": r, "prompt": prompt,
                "query_key": q_key, "expected": expected,
            })
    print(f"[r22b] {len(candidates)} candidates "
          f"({len(axes)} cells × {REPLICAS} replicas)")

    digit_ids = build_gemma_digit_ids()

    # Pre-tokenize every candidate ONCE (tok.encode is ~50ms/prompt on 500-tok
    # inputs — re-tokenizing in loops cost 4+ min on the first attempt).
    print("[r22b] pre-tokenizing candidates...")
    t0 = time.time()
    for c in candidates:
        c["ids"] = tok.encode(c["prompt"])  # type: ignore[name-defined]
    print(f"  tokenized {len(candidates)} prompts in {time.time() - t0:.1f}s")

    # --- Warmup: one forward per unique prefill shape ---
    clear_card_state()
    shape_lens = sorted({len(c["ids"]) for c in candidates})
    print(f"[r22b] warmup: {len(shape_lens)} unique prefill lengths: "
          f"{shape_lens}")
    first_by_len = {}
    for c in candidates:
        ln = len(c["ids"])
        if ln not in first_by_len:
            first_by_len[ln] = c
    for ln in shape_lens:
        t0 = time.time()
        _ = score(first_by_len[ln]["prompt"],
                  first_by_len[ln]["expected"], digit_ids)
        elapsed = time.time() - t0
        print(f"  S={ln:>4} warmup: {elapsed:.1f}s")

    # --- Baseline pass (stock Gemma, no card) ---
    print("\n[r22b] BASELINE pass (stock Gemma)...")
    clear_card_state()
    t0 = time.time()
    baseline = []
    for i, c in enumerate(candidates):
        r = score(c["prompt"], c["expected"], digit_ids, ids=c["ids"])
        r.update({k: c[k] for k in ("n_pairs", "distractor_tokens",
                                      "replica", "query_key", "expected")})
        baseline.append(r)
    print(f"  done in {time.time() - t0:.1f}s")

    # --- Install card ---
    print("\n[r22b] installing MQAR card...")
    ckpt_path = ROOT / "calm/hrm/checkpoints/copy_augmented_delta_mqar_best.pt"
    card = load_mqar_card(ckpt_path)
    slot, state, hook = install(m, card, layer_idx=30, ch_off=2480)  # type: ignore[name-defined]

    # --- With-card pass ---
    print("\n[r22b] WITH-CARD pass...")
    t0 = time.time()
    with_card = []
    for i, c in enumerate(candidates):
        mqar_str = parse_mqar_prompt(c["prompt"])
        if mqar_str is None:
            state["mqar_ids"] = None
            state["active"] = False
            parse_ok = False
        else:
            state["mqar_ids"] = mqar_to_ids(mqar_str)
            state["active"] = True
            parse_ok = True
        r = score(c["prompt"], c["expected"], digit_ids, ids=c["ids"])
        r.update({k: c[k] for k in ("n_pairs", "distractor_tokens",
                                      "replica", "query_key", "expected")})
        r["parse_ok"] = parse_ok
        with_card.append(r)
    print(f"  done in {time.time() - t0:.1f}s")

    # --- Partition + report ---
    from collections import defaultdict
    cell_base = defaultdict(lambda: {"solve": 0, "wrong": 0, "total": 0})
    cell_card = defaultdict(lambda: {"solve": 0, "wrong": 0, "total": 0})
    for b, w in zip(baseline, with_card):
        k = (b["n_pairs"], b["distractor_tokens"])
        cell_base[k]["total"] += 1
        cell_card[k]["total"] += 1
        if b["verdict"] == "solve":
            cell_base[k]["solve"] += 1
        else:
            cell_base[k]["wrong"] += 1
        if w["verdict"] == "solve":
            cell_card[k]["solve"] += 1
        else:
            cell_card[k]["wrong"] += 1

    print("\n=== PER-CELL (baseline vs with-card) ===")
    print(f"  {'N':>3}  {'dist':>5}  base   card   Δ")
    for k in sorted(cell_base.keys()):
        b, c = cell_base[k], cell_card[k]
        delta = c["solve"] - b["solve"]
        sign = "+" if delta > 0 else ""
        print(f"  {k[0]:>3}  {k[1]:>5}  "
              f"{b['solve']}/{b['total']}    "
              f"{c['solve']}/{c['total']}    "
              f"{sign}{delta}")

    n_base = sum(1 for b in baseline if b["verdict"] == "solve")
    n_card = sum(1 for w in with_card if w["verdict"] == "solve")
    print(f"\n=== OVERALL ===")
    print(f"  baseline:  {n_base}/{len(baseline)}")
    print(f"  with card: {n_card}/{len(with_card)}  (Δ={n_card - n_base:+d})")

    # --- Save corpus + results ---
    result_path = CACHE / "gate_lift_results.jsonl"
    with result_path.open("w") as f:
        for b, w, c in zip(baseline, with_card, candidates):
            rec = {
                **c,
                "baseline_verdict": b["verdict"],
                "baseline_top": b["top_tok"],
                "baseline_got": b["got"],
                "card_verdict": w["verdict"],
                "card_top": w["top_tok"],
                "card_got": w["got"],
                "parse_ok": w["parse_ok"],
                "n_tokens": b["n_tokens"],
            }
            f.write(json.dumps(rec) + "\n")
    print(f"\n[r22b] results → {result_path}")

    # Fail corpus — baseline wrong AND with-card wrong (for next iteration)
    still_wrong = [
        (b, w, c) for b, w, c in zip(baseline, with_card, candidates)
        if b["verdict"] != "solve" and w["verdict"] != "solve"
    ]
    print(f"  still-wrong (both baseline AND card miss): {len(still_wrong)}")


main()
print("R22B_GATE_DONE")
