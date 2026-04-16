"""Save/load round-trip test for GemmaSubstrate.

H1: torch.save fails on mmap views. Verify + fix.
H2: After fix, reloaded substrate produces identical logits to original.
H3: Installed cards (in-attention + CardSlot) persist through save/load.

Run in two phases to force process-boundary reload:
  phase=save   → load substrate, install cards, torch.save, record baseline logits
  phase=load   → torch.load in fresh process, compare logits vs baseline
"""

import argparse
import os
import sys
import time
import traceback

import torch
import torch.nn as nn


# Module-level helpers so CardSlot closures pickle cleanly.
def _cardslot_input(h):
    return h[..., 2400:2402]


def _cardslot_writer(h, card_out, ch_lo, ch_hi):
    h[..., ch_lo:ch_hi] = h[..., ch_lo:ch_hi] + card_out
    return h

GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
CKPT_PATH = "/tmp/test_substrate.pt"
BASELINE_PATH = "/tmp/test_substrate_baseline.pt"


def load_substrate_fresh():
    """Load from GGUF, enable Triton, preload to GPU."""
    from calm.llm_computer.gemma_substrate import GemmaSubstrate, enable_triton_tq4
    enable_triton_tq4(True)
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=512)
    m.preload_gpu("cuda")
    return m


def install_demo_cards(m):
    """Exercise all three install patterns:
      1. In-attention (add_one at layer 41, FP32-hosted)
      2. CardSlot residual-additive (threshold at layer 41)
      3. VerificationHook biasing Gemma logits from a CardSlot"""
    from calm.llm_computer.gemma_substrate import CardSlot, VerificationHook
    from calm.llm_computer.programs.add_one import build_add_one
    from calm.llm_computer.programs.threshold import build_threshold

    m.convert_layer_to_fp32(41)

    add_one = build_add_one()
    info = m.install_card_in_attention(
        add_one, layer_idx=41, sub_head_offset=0,
        ch_off=2400, d_card=8, mode="hard_max",
    )
    print(f"[install] in-attention add_one: {info}")

    # Use a tiny picklable nn.Linear card. Picks 2 channels from residual,
    # projects 2→2, adds result back. Exercises CardSlot save/load without
    # bringing a compiled-card's token-ID forward signature into play.
    tiny_card = nn.Linear(2, 2).to("cuda").eval()
    slot = CardSlot(layer_idx=41, ch_off=2544, card=tiny_card, d_card=2,
                    card_input_fn=_cardslot_input, use_full_residual=True,
                    output_fn=_cardslot_writer)
    slot.attach(m, preserve=False)
    print(f"[install] CardSlot(nn.Linear): layer=41 ch=[2544:2546]")

    # VerificationHook — bias Gemma logit with an arbitrary pre-set output.
    # The hook reads slot.last_output each forward; we don't need it to
    # produce a useful result here, just to survive save/load.
    vocab_map = {0: 236771}  # card token 0 → Gemma token '0'
    hook = VerificationHook(slot, vocab_map, boost=0.0)  # boost=0 = no-op
    m.verification_hooks.append(hook)
    print(f"[install] VerificationHook attached")

    return {"add_one": add_one, "tiny_card": tiny_card, "slot": slot,
            "hook": hook}


def get_first_logits(m, prompt="The capital of France is"):
    """Return (B, 1, V) FP32 last-position logits — full forward via KVCache."""
    from calm.llm_computer.gemma_substrate import KVCache
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)
    ids = torch.tensor([tok.encode(prompt)], device="cuda")
    cache = KVCache(m.config.n_layers, device="cuda")
    with torch.no_grad():
        logits = m.forward(ids, device="cuda", kv_cache=cache, start_pos=0)
    return logits.detach().cpu().float()


def phase_save():
    print("=" * 60)
    print("PHASE: save")
    print("=" * 60)
    m = load_substrate_fresh()
    m.warmup(seq_lens=(1, 6))
    cards = install_demo_cards(m)

    print("\n[save] recording baseline logits...")
    baseline = get_first_logits(m)
    print(f"  baseline.shape={baseline.shape} sum={baseline.sum().item():.4f}")
    torch.save(baseline, BASELINE_PATH)

    print("\n[save] torch.save(m, ...)")
    t0 = time.time()
    try:
        torch.save(m, CKPT_PATH, pickle_protocol=5)
        print(f"  saved in {time.time()-t0:.1f}s, "
              f"{os.path.getsize(CKPT_PATH)/1e9:.2f} GB")
        return 0
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


def phase_load():
    print("=" * 60)
    print("PHASE: load")
    print("=" * 60)
    # Must import BEFORE load so classes resolve.
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4, MmapTq4Linear
    )
    enable_triton_tq4(True)

    print(f"[load] torch.load from {CKPT_PATH} "
          f"({os.path.getsize(CKPT_PATH)/1e9:.2f} GB)")
    t0 = time.time()
    m = torch.load(CKPT_PATH, weights_only=False, map_location="cuda")
    print(f"  loaded in {time.time()-t0:.1f}s")
    assert MmapTq4Linear._shared_pi is not None, "Pi cache not restored"

    print("[load] logits check...")
    got = get_first_logits(m)
    baseline = torch.load(BASELINE_PATH)

    diff = (got - baseline).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    baseline_range = (baseline.max() - baseline.min()).item()
    print(f"  baseline range: {baseline_range:.2f}")
    print(f"  max abs diff:   {max_diff:.6f}")
    print(f"  mean abs diff:  {mean_diff:.6f}")

    # Argmax agreement on the LAST position (the one that matters)
    a_base = baseline[0, -1].argmax().item()
    a_got = got[0, -1].argmax().item()
    argmax_match = a_base == a_got
    print(f"  argmax baseline: {a_base}, reloaded: {a_got}, "
          f"{'MATCH' if argmax_match else 'MISMATCH'}")

    # Card-state survival checks
    print("\n[load] install-state checks...")
    from calm.llm_computer.gemma_substrate import FP32GemmaLinear
    layer41 = m.layers[41]
    fp32_ok = isinstance(layer41.attn_q, FP32GemmaLinear)
    print(f"  layer 41 attn_q is FP32: {fp32_ok}")
    partition_ok = (41 in m.attention_partition
                    and len(m.attention_partition[41]) == 1
                    and m.attention_partition[41][0][2] == "hard_max")
    print(f"  attention_partition[41] has hard_max entry: {partition_ok}")
    slots_ok = (hasattr(layer41, "card_slots") and len(layer41.card_slots) == 1
                and layer41.card_slots[0].d_card == 2)
    print(f"  layer 41 has 1 CardSlot (threshold): {slots_ok}")
    slot = layer41.card_slots[0]
    slot_fired = (hasattr(slot, "last_output")
                  and slot.last_output is not None)
    print(f"  CardSlot fired during forward: {slot_fired}")
    hooks_ok = (len(m.verification_hooks) == 1
                and m.verification_hooks[0].vocab_mapping == {0: 236771})
    print(f"  VerificationHook preserved: {hooks_ok}")

    tol = 1e-3 * max(baseline_range, 1.0)
    ok = (max_diff < tol and argmax_match and fp32_ok and partition_ok
          and slots_ok and slot_fired and hooks_ok)
    print(f"\n  verdict: {'PASS' if ok else 'FAIL'} (tol={tol:.4f})")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["save", "load", "both"], default="both")
    args = ap.parse_args()

    if args.phase == "save":
        return phase_save()
    if args.phase == "load":
        return phase_load()
    # both: re-exec ourselves for load to force process boundary
    rc = phase_save()
    if rc != 0:
        return rc
    import subprocess
    return subprocess.call([sys.executable, __file__, "--phase", "load"],
                           env={**os.environ, "PYTHONPATH": "."})


if __name__ == "__main__":
    sys.exit(main())
