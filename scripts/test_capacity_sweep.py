"""Empirical capacity test for the 30-50 domains claim.

H1: The 30-50 domains figure is a projection from 1024 free sub-heads ×
35 SWA layers = 35,840 slots. Never measured. Channel pressure, not
sub-head pressure, is the likely binding constraint — every installed
card steals channels from Gemma's own residual stream (d_model=2560).

Measurement:
  - Baseline Gemma quality on 10 held-out prompt/expected-token pairs
  - Incremental install sweep: N ∈ {1, 5, 10, 20, 30}
  - Per step: report argmax match count + mean -log P(expected token)
  - Cards are add_one (compiled, 1,280 params, 8 channels / 4 sub-heads)
  - Hosts: 3 FP32 SWA layers (25, 33, 41) in shared-KV range

Decision: at what N does Gemma's argmax quality on the test set
meaningfully regress?
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")


# 10 held-out completions with confident next-token expectations.
TEST_CASES = [
    ("The capital of France is", " Paris"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Spain is", " Madrid"),
    ("The capital of Japan is", " Tokyo"),
    ("2 plus 2 equals", " 4"),
    ("Water freezes at", " 0"),
    ("The opposite of up is", " down"),
    ("Dogs bark, cats", " meow"),
    ("The first president of the USA was", " George"),
]


# FP32 hosts: all SWA, all in the shared-KV reuse range (≥24) so
# their K/V writes are ignored by downstream layers (architecturally
# free of propagated side effects).
HOSTS = [25, 33, 41]
# Per-card: d_card=8 channels, 4 sub-heads. Start from high channel and
# grow downward; start sub-heads from 0 and grow upward.
D_CARD = 8
SUB_HEADS_PER_CARD = 4
BASE_CH = 2400  # start of card channel range; descends toward 0
BASE_SH = 0     # start of sub-head range; ascends


def plan_install(card_idx: int) -> dict:
    """Deterministic card_idx → (host_layer, ch_off, sub_head_offset).
    Each host gets a distinct column in (ch, sh) space."""
    host_i = card_idx % len(HOSTS)
    slot_i = card_idx // len(HOSTS)
    return {
        "host_layer": HOSTS[host_i],
        "ch_off": BASE_CH - slot_i * D_CARD,
        "sub_head_offset": BASE_SH + slot_i * SUB_HEADS_PER_CARD,
    }


def eval_quality(m, tok):
    """Return (argmax_match_count, mean_nll)."""
    from calm.llm_computer.gemma_substrate import KVCache

    match = 0
    total_nll = 0.0
    n = 0
    for prompt, expected in TEST_CASES:
        prompt_ids = tok.encode(prompt)
        # Expected: run tokenizer on expected, take the first id AFTER
        # the prompt's BOS-augmented encoding. Simpler: encode (prompt +
        # expected), compare to prompt_ids, first mismatch = expected id.
        full_ids = tok.encode(prompt + expected)
        assert full_ids[: len(prompt_ids)] == prompt_ids, \
            f"tokenizer prefix mismatch on {prompt!r}"
        expected_id = full_ids[len(prompt_ids)]

        cache = KVCache(m.config.n_layers, device="cuda")
        with torch.no_grad():
            logits = m.forward(
                torch.tensor([prompt_ids]), device="cuda",
                kv_cache=cache, start_pos=0,
            )
        last = logits[0, -1].float()  # (V,)
        log_probs = F.log_softmax(last, dim=-1)
        nll = -log_probs[expected_id].item()
        total_nll += nll
        argmax = int(last.argmax())
        if argmax == expected_id:
            match += 1
        n += 1

    return match, total_nll / n


def ensure_host_fp32(m, converted: set, host_layer: int) -> float:
    if host_layer in converted:
        return 0.0
    t0 = time.time()
    m.convert_layer_to_fp32(host_layer)
    torch.cuda.synchronize()
    converted.add(host_layer)
    return time.time() - t0


def install_one(m, card_idx: int):
    from calm.llm_computer.programs.add_one import build_add_one
    plan = plan_install(card_idx)
    card = build_add_one(vocab_size=D_CARD)
    info = m.install_card_in_attention(
        card, layer_idx=plan["host_layer"],
        sub_head_offset=plan["sub_head_offset"],
        ch_off=plan["ch_off"], d_card=D_CARD, mode="hard_max",
    )
    return card, info


def fmt_vram() -> str:
    alloc = torch.cuda.memory_allocated() / 1e9
    resv = torch.cuda.memory_reserved() / 1e9
    return f"{alloc:.2f} GB alloc / {resv:.2f} GB reserved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="1,5,10,20,30",
                    help="comma-separated N install counts to sample")
    args = ap.parse_args()
    targets = sorted({int(x) for x in args.targets.split(",") if x})

    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[capacity] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 6, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    rows = []

    # N = 0 (pure baseline, no FP32 layers converted)
    print(f"\n=== N=0 (baseline) — VRAM {fmt_vram()} ===")
    match, nll = eval_quality(m, tok)
    rows.append((0, match, nll, torch.cuda.memory_allocated() / 1e9))
    print(f"  argmax: {match}/{len(TEST_CASES)}, mean NLL: {nll:.3f}")

    # Incremental install sweep. Cards are kept alive by holding their refs
    # (CardSlot's attn install is in-weight; compiled card refs can be dropped
    # because weights live in attn_q/k/v/output now).
    cards = []
    converted = set()
    next_card = 0
    for N in targets:
        while next_card < N:
            plan = plan_install(next_card)
            ensure_host_fp32(m, converted, plan["host_layer"])
            if plan["ch_off"] < 0:
                print(f"[capacity] halting: ch_off would go negative at card {next_card}")
                break
            card, _info = install_one(m, next_card)
            cards.append(card)
            next_card += 1
        print(f"\n=== N={next_card} — VRAM {fmt_vram()} ===")
        print(f"  hosts: {sorted(converted)}")
        match, nll = eval_quality(m, tok)
        rows.append((next_card, match, nll, torch.cuda.memory_allocated() / 1e9))
        print(f"  argmax: {match}/{len(TEST_CASES)}, mean NLL: {nll:.3f}")

    print("\n========== SUMMARY ==========")
    print(f"{'N':>4} {'argmax':>12} {'mean_NLL':>10} {'Δ NLL':>10} {'VRAM GB':>8}")
    base_nll = rows[0][2]
    for N, match, nll, vram in rows:
        dnll = nll - base_nll
        print(f"{N:>4} {match:>6}/{len(TEST_CASES):<5} "
              f"{nll:>10.3f} {dnll:>+10.3f} {vram:>8.2f}")

    # Verdict
    base_match = rows[0][1]
    worst_match = min(r[1] for r in rows)
    match_drop = base_match - worst_match
    print(f"\nbaseline argmax: {base_match}/{len(TEST_CASES)}")
    print(f"worst argmax:    {worst_match}/{len(TEST_CASES)}")
    print(f"max argmax drop: {match_drop}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
