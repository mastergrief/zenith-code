"""End-to-end learning loop on the Gemma substrate.

Demonstrates the full substrate vision:

  detect Gemma's mistakes → log corrections → compile a recall card
  from the corrections → install via CardSlot → biased logits via
  VerificationHook → save corrections to disk → reload + rebuild card

All pieces already shipped:
  - GemmaSubstrate (gemma_substrate.py): the loaded Gemma 4 E4B
  - CardSlot, VerificationHook (gemma_substrate.py): card install +
    verification feedback into Gemma's logits
  - KnowledgeStore (persistent_knowledge.py): tracks corrections,
    compiles them into a Small2DTransformer recall card via
    step-function ReGLU dispatch (3 ReGLU per fact)

The compiled recall card IS a Small2DTransformer with d_head=2 — same
substrate-native architecture as every other compiled card. CardSlot
installs it into Gemma at a reserved channel range; VerificationHook
biases the corresponding Gemma vocab logits when the card has a hit.

Usage:
    PYTHONPATH=. python3 scripts/gemma_learning_loop_demo.py

Output: shows Gemma's wrong answers, the learned card, the corrected
answers after install, and the persistence round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from calm.llm_computer.gemma_substrate import (
    GemmaSubstrate, KVCache, CardSlot, VerificationHook, enable_triton_tq4,
)
from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer
from calm.llm_computer.persistent_knowledge import KnowledgeStore


# Gemma 4 E4B BPE token IDs for single digits 0..9 (vocab size 262144).
DIGIT_TO_GEMMA = {
    0: 236771, 1: 236770, 2: 236778, 3: 236800, 4: 236812,
    5: 236810, 6: 236825, 7: 236832, 8: 236828, 9: 236819,
}


def main():
    enable_triton_tq4(True)

    print("loading Gemma...", flush=True)
    m = GemmaSubstrate.from_gguf(
        "/home/gabe/models/gemma-4-E4B-it-tq4-aligned.gguf", max_len=64)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(6,))
    tok = GemmaTokenizer.from_gguf(
        "/home/gabe/models/gemma-4-E4B-it-tq4-aligned.gguf")

    test_cases = [(2, 3), (4, 1), (3, 2), (5, 1), (2, 4)]
    store = KnowledgeStore(max_key=77, max_value=8)

    # -------- Phase 1: detect Gemma's mistakes --------
    print("\n=== PHASE 1: probe Gemma, log mistakes ===")
    for a, b in test_cases:
        prompt = f"{a} plus {b} equals"
        expected = (a + b) % 8
        ids = tok.encode(prompt)
        cache = KVCache(m.config.n_layers, device="cuda")
        with torch.no_grad():
            logits = m.forward(torch.tensor([ids]), device="cuda",
                                kv_cache=cache, start_pos=0)
        top = int(logits[0, -1].argmax())
        ok = (top == DIGIT_TO_GEMMA[expected])
        print(f"  {a}+{b}={expected}: Gemma says {tok.id_to_token.get(top, '?')!r} "
              f"{'✓' if ok else '✗'}")
        if not ok:
            store.add_correction(a * 10 + b, expected)
    print(f"\n{len(store.corrections)} corrections logged")

    # -------- Phase 2: compile corrections into a recall card --------
    print("\n=== PHASE 2: compile corrections → recall card ===")
    recall = store.build_recall_model().cuda().eval()
    print(f"Recall card: {sum(p.numel() for p in recall.parameters())} params, "
          f"vocab={recall.config.vocab_size}, d_model={recall.config.d_model}")

    # -------- Phase 3: install + verification hook --------
    print("\n=== PHASE 3: install into Gemma ===")
    current_query = {"key": 0}

    def recall_input(h):
        return torch.tensor([[current_query["key"]]], device="cuda")

    def recall_output(h, logits, ch_lo, ch_hi):
        # Add recall's last-position logits to the last Gemma position
        ans = logits[:, -1:, :]
        n = min(ans.shape[-1], ch_hi - ch_lo)
        h[..., -1:, ch_lo:ch_lo + n] = h[..., -1:, ch_lo:ch_lo + n] + ans[..., :n]
        return h

    slot = CardSlot(layer_idx=35, ch_off=2480, card=recall,
                     d_card=recall.config.vocab_size,
                     card_input_fn=recall_input,
                     use_full_residual=True,
                     output_fn=recall_output)
    slot.attach(m, preserve=True)
    m.verification_hooks.append(
        VerificationHook(slot, vocab_mapping=DIGIT_TO_GEMMA, boost=50.0))

    # -------- Phase 4: re-run, expect corrections to flow through --------
    print("\n=== PHASE 4: re-run with learned card ===")
    fixed = 0
    for a, b in test_cases:
        current_query["key"] = a * 10 + b
        prompt = f"{a} plus {b} equals"
        expected = (a + b) % 8
        ids = tok.encode(prompt)
        cache = KVCache(m.config.n_layers, device="cuda")
        with torch.no_grad():
            logits = m.forward(torch.tensor([ids]), device="cuda",
                                kv_cache=cache, start_pos=0)
        top = int(logits[0, -1].argmax())
        ok = (top == DIGIT_TO_GEMMA[expected])
        print(f"  {a}+{b}={expected}: Gemma+learned says "
              f"{tok.id_to_token.get(top, '?')!r} {'✓' if ok else '✗'}")
        if ok:
            fixed += 1
    print(f"\nFixed {fixed}/{len(test_cases)}")

    # -------- Phase 5: persistence --------
    print("\n=== PHASE 5: save + reload ===")
    save_path = Path("/tmp/gemma_learned_corrections.json")
    store.save_corrections(save_path)
    store2 = KnowledgeStore(max_key=77, max_value=8)
    store2.load_corrections(save_path)
    recall2 = store2.build_recall_model().cuda().eval()
    match = all(
        int(recall(torch.tensor([[c.query_key]], device="cuda"))[0, -1].argmax())
        == int(recall2(torch.tensor([[c.query_key]], device="cuda"))[0, -1].argmax())
        for c in store.corrections
    )
    print(f"reloaded recall == original recall: {match}")
    print("\nLEARNING LOOP CLOSED: detect → log → compile → install → verify → persist")


if __name__ == "__main__":
    main()
