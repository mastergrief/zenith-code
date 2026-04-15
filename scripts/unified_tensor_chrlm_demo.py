"""Unified tensor demo — one Small2DTransformer hosts multiple compiled cards.

Thesis test: "compiled routing + compiled cards produce correct
arithmetic via compiled paths without any training". The smallest
demonstrable form:

  1. Build a substrate (GroupedSmall2DTransformer) sized to host two cards.
  2. Install the 2-digit `adder` card at slot A (vocab 0..199, channels
     0..201, sub-heads 0..100, FFN slots 0..397, layers 0..0).
  3. Install the `compiled_router` card at slot B (vocab 200..300,
     channels 202..331, sub-heads 101..165, FFN slots 400..643,
     layers 0..1).
  4. Save the substrate's state_dict as a single `.pt` file.
  5. Reload into a fresh substrate with the same config.
  6. Forward "15 + 27" through the adder slot — verify pos-1 argmax
     decodes as 42.
  7. Forward "(a=7, b=5, op=MUL)" through the router slot — verify
     pos-2 argmax decodes as 35.

Pass: both checks match expected. Proves cards coexist under the
substrate, compose via disjoint slots, and survive serialization.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from calm.llm_computer.card_installer import CardSlot, install_compiled_card
from calm.llm_computer.grouped_small2d import (
    GroupedSmall2DConfig, GroupedSmall2DTransformer,
)
from calm.llm_computer.programs.adder import build_adder, VOCAB as ADDER_VOCAB
from calm.llm_computer.programs.compiled_router import (
    build_router, decode_output, MUL_SLOT_BASE, VOCAB as ROUTER_VOCAB,
)


def build_substrate_for_cards(adder, router) -> GroupedSmall2DTransformer:
    """Build a substrate sized to host adder + router side-by-side."""
    # Channel budget: adder (202) + router (130) = 332 channels
    # Sub-heads: 101 + 65 = 166 (matches d_model/2 since d_head=2)
    # FFN: adder uses 398 on layer 0; router uses 6 on layer 0, 238 on layer 1.
    #   Total needs max-per-layer sum = max(398 + 6, 238) = 404 minimum.
    #   Generous allocation: 398 for adder + 250 for router = 648.
    # Vocab: 200 (adder) + 101 (router) = 301, pad to 400.
    # Layers: router uses 2, adder uses 1 — substrate needs 2.
    # max_len: adder uses 4, router uses 3 — take 5.
    n_heads = 101 + 65 + 4  # 170 (extra free sub-heads as headroom)
    d_model = 2 * n_heads    # 340
    d_ffn = 648
    vocab = 400
    n_layers = 2
    max_len = 8
    cfg = GroupedSmall2DConfig(
        vocab_size=vocab,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        max_len=max_len,
        use_hard_max=False,
        layer_modes=tuple(["single"] * n_layers),
        layer_hard_max=tuple([True] * n_layers),
    )
    sub = GroupedSmall2DTransformer(cfg)
    with torch.no_grad():
        for p in sub.parameters():
            p.zero_()
    return sub


def main() -> None:
    print("[demo] building compiled cards...")
    adder = build_adder()                     # 2-digit adder, 1 layer
    router = build_router()                   # ADD/MUL dispatcher, 2 layers
    print(f"  adder:  d_model={adder.config.d_model} "
          f"n_heads={adder.config.n_heads} n_layers={adder.config.n_layers} "
          f"params={adder.param_count():,}")
    print(f"  router: d_model={router.config.d_model} "
          f"n_heads={router.config.n_heads} n_layers={router.config.n_layers} "
          f"params={router.param_count():,}")

    print("[demo] building substrate...")
    substrate = build_substrate_for_cards(adder, router)
    print(f"  substrate: d_model={substrate.config.d_model} "
          f"n_heads={substrate.config.n_heads} d_ffn={substrate.config.d_ffn} "
          f"n_layers={substrate.config.n_layers} vocab={substrate.config.vocab_size} "
          f"params={substrate.param_count():,}")

    # Slot A: adder at origin (0 offsets, vocab 0..199)
    adder_slot = CardSlot(ch_off=0, sh_off=0, ffn_off=0, tok_off=0)
    # Slot B: router after adder in every axis
    router_slot = CardSlot(
        ch_off=adder.config.d_model,       # 202
        sh_off=adder.config.n_heads,       # 101
        ffn_off=adder.config.d_ffn,        # 398 — disjoint FFN slots
        tok_off=ADDER_VOCAB,               # 200 — router vocab starts here
    )

    print("[demo] installing adder @ slot A, router @ slot B...")
    install_compiled_card(substrate, adder, adder_slot)
    install_compiled_card(substrate, router, router_slot)

    # Save
    tmp = Path(tempfile.mkdtemp()) / "unified_chrlm.pt"
    print(f"[demo] saving substrate to {tmp}...")
    torch.save({
        "state_dict": substrate.state_dict(),
        "config": substrate.config.__dict__,
    }, tmp)
    size_mb = tmp.stat().st_size / 1e6
    print(f"  file size: {size_mb:.1f} MB")

    # Reload into fresh instance
    print("[demo] reloading into fresh substrate...")
    reloaded = build_substrate_for_cards(adder, router)  # same config
    ckpt = torch.load(tmp, weights_only=True)
    reloaded.load_state_dict(ckpt["state_dict"])

    # Each card's head entries live in disjoint substrate vocab ranges.
    # A higher-level router (future work) selects which slot range to
    # read — the handoff's "compiled router" is a gate-graph step that
    # gates card outputs by detected input pattern. For this round the
    # thesis test is: per-card, does the correct slot win within its
    # own range? That validates "the compiled card hosted in the
    # substrate correctly computed the answer via compiled paths".

    ADDER_RANGE = (0, ADDER_VOCAB)                         # [0, 200)
    ROUTER_RANGE = (ADDER_VOCAB, ADDER_VOCAB + ROUTER_VOCAB)  # [200, 301)

    def argmax_in_range(logits_1d: torch.Tensor,
                        lo: int, hi: int) -> int:
        sub = logits_1d[lo:hi]
        return int(sub.argmax().item()) + lo

    # --- Test 1: adder "15 + 27 = 42" ---
    print("\n[demo] TEST 1 — adder: 15 + 27 = ?")
    x = torch.tensor([[15, 27]], dtype=torch.long)
    with torch.no_grad():
        logits = reloaded(x)
    pred_slot = argmax_in_range(logits[0, 1], *ADDER_RANGE)
    expected = 15 + 27
    ok_add = pred_slot == expected
    print(f"  argmax in adder range = {pred_slot}, expected {expected} — "
          f"{'PASS' if ok_add else 'FAIL'}")

    # --- Test 2: router "(a=7, b=5, op=MUL=1) → 35" ---
    print("[demo] TEST 2 — router: a=7 * b=5 = ?")
    a, b, op = 7, 5, 1
    x = torch.tensor([[a + ADDER_VOCAB, b + ADDER_VOCAB, op + ADDER_VOCAB]],
                     dtype=torch.long)
    with torch.no_grad():
        logits = reloaded(x)
    pred_substrate_tok = argmax_in_range(logits[0, 2], *ROUTER_RANGE)
    pred_router_slot = pred_substrate_tok - ADDER_VOCAB
    pred_value = decode_output(op, pred_router_slot)
    expected = a * b
    ok_mul = pred_value == expected
    print(f"  substrate token {pred_substrate_tok} → router slot "
          f"{pred_router_slot} → value {pred_value}, expected {expected} — "
          f"{'PASS' if ok_mul else 'FAIL'}")

    # --- Test 3: router ADD mode, 6 + 3 = 9 ---
    print("[demo] TEST 3 — router: a=6 + b=3 = ?")
    a, b, op = 6, 3, 0
    x = torch.tensor([[a + ADDER_VOCAB, b + ADDER_VOCAB, op + ADDER_VOCAB]],
                     dtype=torch.long)
    with torch.no_grad():
        logits = reloaded(x)
    pred_router_slot = argmax_in_range(logits[0, 2], *ROUTER_RANGE) - ADDER_VOCAB
    pred_value = decode_output(op, pred_router_slot)
    expected = a + b
    ok_add_router = pred_value == expected
    print(f"  router slot {pred_router_slot} → value {pred_value}, "
          f"expected {expected} — {'PASS' if ok_add_router else 'FAIL'}")

    # --- Summary ---
    all_ok = ok_add and ok_mul and ok_add_router
    print(f"\n[demo] OVERALL: {'PASS' if all_ok else 'FAIL'}")
    print("[demo] thesis: compiled routing + compiled cards produce correct")
    print(f"[demo]         arithmetic via compiled paths, no training — "
          f"{'VALIDATED' if all_ok else 'NOT VALIDATED'}")
    tmp.unlink()


if __name__ == "__main__":
    main()
